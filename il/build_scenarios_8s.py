#!/usr/bin/env python3
"""
il/build_scenarios_8s.py

Builds the 8 s imitation-learning scenarios (ML_DECISIONS.md sections 0, 2, 3).

From Scenariosssss/usable_scenarios.csv (status "clean as recorded" / "rescued ..."):
  1. window: 34 frames (8 s at 4.27 fps) starting PRE_FRAMES (1.5 s) before the behaviour window,
     slid to stay inside the frames where the behaviour vehicles are tracked continuously
  2. per pair keep a window only if it overlaps <= 50 % with the previously kept one   -> ~1,717
  3. re-check the 8 s window as count_usable_scenarios.py does: checks, >= 80 % in-box, spawnable,
     car contact (shift <= 0.7 m via apply_separation), kerb/median hit
  4. neighbours: real non-pedestrian vehicles within 30 m of the ego, nearest first, accepted while
     the CARLA-body gap to every accepted vehicle (shifted ego/other included) stays >= 0.5 m, the path
     covers >= 90 % of the window from its first frame, it is spawnable and not off-road; at most 4.
     Also reports how many would be accepted at 0.3 / 0.7 m.
  4b. windows that fail at 8 s get a 5 s try (still containing the behaviour window, <= 50 % overlap)
  5. route command per vehicle and frame: what the vehicle does at its next pass through the junction
     (left / straight / right / U-turn, from the CARLA path heading change); "straight" if none.

No CARLA server needed (needs Scenariosssss/town10hd_drivable.npz).

  python il/build_scenarios_8s.py                 # all; writes Scenariosssss/IL/scenarios_8s.json + .csv
  python il/build_scenarios_8s.py --limit 50      # quick test
"""

import argparse
import collections
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, BASE_DIR)

import scenario_checks as sc  # noqa: E402
import timemethod  # noqa: E402
from debug.check_scenarios import LABEL_BLUEPRINT_MAP as LBP  # noqa: E402
from debug.predict_collisions import load_map, off_road, CORNER_INSET_M, VALID_SPAWN  # noqa: E402
from debug import apply_separation as sep  # noqa: E402

SCEN = os.path.join(BASE_DIR, "Scenariosssss")
OUT_DIR = os.path.join(SCEN, "IL")
ORDER = ["Cut In", "Cut Out", "U-turn", "Overtaking Left", "Overtaking Right", "Vehicle Following"]

WINDOW_FRAMES = 34          # 8 s at 4.27 fps
FALLBACK_FRAMES = 21        # 5 s, for windows that fail at 8 s
PRE_FRAMES = 6              # window starts 1.5 s before the behaviour window
MAX_OVERLAP = 0.5           # keep a window if it overlaps <= 50 % with the previous kept one of the pair
NEIGHBOUR_RADIUS_M = 30.0
MAX_NEIGHBOURS = 4
NEIGHBOUR_GAP_M = 0.5       # decided clearance (ML_DECISIONS.md S2)
REPORT_GAPS = (0.3, 0.5, 0.7)
MIN_NEIGHBOUR_COVERAGE = 0.9
KERB_FRAMES = 5             # same threshold as count_usable_scenarios.py
JUNCTION_PAD = 8            # frames (~2 s) before/after the Center crossing used for the heading change
UTURN_SPAN = 30             # U-turn outside Center: heading reverses within this many frames (~7 s)
MIN_CHORD_M = 2.0           # a heading needs at least this much movement
CMDS = ["left", "straight", "right", "u-turn"]


# ==========================================
# WINDOWS
# ==========================================
def contiguous_range(frames_set, start, end):
    a, b = start, end
    while a - 1 in frames_set:
        a -= 1
    while b + 1 in frames_set:
        b += 1
    return a, b


def select_windows(usable, store):
    """8 s windows (slid inside the tracked range), then <= 50 % overlap per pair."""
    cands = []
    for r in usable.itertuples():
        ids = [int(r.ego)] + ([int(r.other)] if not np.isnan(r.other) else [])
        if any(t not in store.tracks for t in ids):
            continue
        lo, hi = -10**9, 10**9
        for t in ids:
            a, b = contiguous_range(set(store.tracks[t].index), int(r.start), int(r.end))
            lo, hi = max(lo, a), min(hi, b)
        if hi - lo + 1 < WINDOW_FRAMES:
            continue
        s8 = min(max(int(r.start) - PRE_FRAMES, lo), hi - WINDOW_FRAMES + 1)
        cands.append({"behaviour": r.behaviour, "ego": ids[0], "other": ids[1] if len(ids) == 2 else None,
                      "src_start": int(r.start), "src_end": int(r.end), "start": s8, "end": s8 + WINDOW_FRAMES - 1})
    kept, last = [], {}
    min_step = int(round(WINDOW_FRAMES * (1 - MAX_OVERLAP)))
    for c in sorted(cands, key=lambda c: (c["behaviour"], c["ego"], -1 if c["other"] is None else c["other"], c["start"])):
        key = (c["behaviour"], c["ego"], c["other"])
        if key in last and c["start"] < last[key] + min_step:
            continue
        kept.append(c)
        last[key] = c["start"]
    return kept


# ==========================================
# GEOMETRY HELPERS
# ==========================================
def footprint(store, tid):
    return sc.BLUEPRINT_FOOTPRINT[LBP.get(store.label(tid), "vehicle.nissan.micra")]


def path_dict(store, tid, frames, override=None):
    """{frame: (x, y)} CARLA positions in the window (None frames dropped); override = shifted path."""
    if override is not None:
        return override
    cp = store.carla_path(tid)
    return {f: cp[f] for f in frames if cp.get(f) is not None}


def min_gap_dicts(pa, pb, fa, fb):
    common = sorted(set(pa) & set(pb))
    if not common:
        return float("inf")
    A = np.array([pa[f] for f in common])
    B = np.array([pb[f] for f in common])
    ya = sc.path_yaws(A) if len(A) > 1 else [0.0]
    yb = sc.path_yaws(B) if len(B) > 1 else [0.0]
    return min(sc.box_gap(sc.box_corners(A[i], ya[i], *fa), sc.box_corners(B[i], yb[i], *fb)) for i in range(len(common)))


def kerb_hit(drivable, paths):
    grid, res, bounds = drivable
    off = 0
    for pts, (L, W) in paths:
        pts = np.asarray(pts)
        if len(pts) < 2:
            continue
        for p, y in zip(pts, sc.path_yaws(pts)):
            inner = sc.box_corners(p, y, L - 2 * CORNER_INSET_M, max(W - 2 * CORNER_INSET_M, 0.1))
            off += off_road(grid, res, bounds, *p) or sum(off_road(grid, res, bounds, *c) for c in inner) >= 2
    return off >= KERB_FRAMES


def spawnable(store, tid, start):
    first = store.tracks[tid].loc[store.tracks[tid].index >= start].iloc[0]
    return timemethod.classify_point((first.kf_cx_pixel, -first.kf_cy_pixel)) in VALID_SPAWN


# ==========================================
# ROUTE COMMAND
# ==========================================
_route_cache = {}


def _chord_heading(cp, frames):
    """Heading (rad) from the first to the last CARLA point of these frames, None if it barely moves."""
    pts = [cp[f] for f in frames if cp.get(f) is not None]
    if len(pts) < 2 or math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]) < MIN_CHORD_M:
        return None
    return math.atan2(pts[-1][1] - pts[0][1], pts[-1][0] - pts[0][0])


def _turn(h0, h1):
    d = math.degrees((h1 - h0 + math.pi) % (2 * math.pi) - math.pi)
    # CARLA is left-handed seen from above (x east, y south): positive yaw change = right turn
    return "u-turn" if abs(d) > 135 else ("right" if d > 30 else ("left" if d < -30 else "straight"))


def junction_passes(store, tid):
    """[(entry_frame, exit_frame, command)]: every pass through Center (heading change between the
    JUNCTION_PAD frames before and after it), plus U-turns made outside Center (median gaps on an arm:
    heading reverses > 135 deg within UTURN_SPAN frames)."""
    if tid in _route_cache:
        return _route_cache[tid]
    t = store.tracks[tid]
    frames = t.index.tolist()
    regions = [timemethod.classify_point((x, -y)) for x, y in zip(t.kf_cx_pixel, t.kf_cy_pixel)]
    cp = store.carla_path(tid)
    passes, i = [], 0
    while i < len(frames):
        if regions[i] != "Center":
            i += 1
            continue
        j = i
        while j + 1 < len(frames) and regions[j + 1] == "Center":
            j += 1
        h0 = _chord_heading(cp, frames[max(0, i - JUNCTION_PAD):i + 1])
        h1 = _chord_heading(cp, frames[j:j + JUNCTION_PAD + 1])
        passes.append((frames[i], frames[j], "straight" if h0 is None or h1 is None else _turn(h0, h1)))
        i = j + 1
    in_center = {f for (a, b, _) in passes for f in range(a, b + 1)}
    k = 0
    while k + UTURN_SPAN < len(frames):
        span = frames[k:k + UTURN_SPAN + 1]
        if not in_center.intersection(span):
            h0 = _chord_heading(cp, frames[max(0, k - 4):k + 1])
            h1 = _chord_heading(cp, frames[k + UTURN_SPAN:k + UTURN_SPAN + 5])
            if h0 is not None and h1 is not None and _turn(h0, h1) == "u-turn":
                passes.append((span[0], span[-1], "u-turn"))
                k += UTURN_SPAN
                continue
        k += 1
    passes.sort()
    _route_cache[tid] = passes
    return passes


def route_commands(store, tid, frames):
    """Command at each frame: the action at the next junction pass not yet finished."""
    passes = junction_passes(store, tid)
    out = []
    for f in frames:
        nxt = [c for (_, exit_f, c) in passes if exit_f >= f]
        out.append(nxt[0] if nxt else "straight")
    return out


# ==========================================
# NEIGHBOURS
# ==========================================
def build_frame_index(store):
    """frame -> (track ids, real xy metres) for all non-pedestrian tracks."""
    idx = collections.defaultdict(list)
    for tid, t in store.tracks.items():
        if int(t.labelID.iloc[0]) == sc.PEDESTRIAN_LABEL:
            continue
        for f, x, y in zip(t.index, t.kf_cx_meter, t.kf_cy_meter):
            idx[f].append((tid, x, y))
    return {f: (np.array([v[0] for v in vs]), np.array([[v[1], v[2]] for v in vs])) for f, vs in idx.items()}


def neighbour_candidates(store, fidx, ego, exclude, frames):
    """Non-pedestrian tracks within NEIGHBOUR_RADIUS_M of the ego in any window frame, nearest first."""
    best = {}
    ego_t = store.tracks[ego]
    for f in frames:
        if f not in fidx or f not in ego_t.index:
            continue
        tids, xy = fidx[f]
        e = ego_t.loc[f, ["kf_cx_meter", "kf_cy_meter"]].to_numpy(dtype=float)
        d = np.hypot(*(xy - e).T)
        for t, dist in zip(tids, d):
            if t in exclude or dist > NEIGHBOUR_RADIUS_M:
                continue
            best[int(t)] = min(best.get(int(t), 1e9), float(dist))
    return sorted(best, key=best.get), best


def pick_neighbours(store, drivable, cands, accepted, frames, gap_m):
    """Greedy nearest-first pick. accepted: [(tid, path_dict, footprint)]. Returns accepted neighbours
    and rejection reasons."""
    chosen, reasons = [], collections.Counter()
    pool = list(accepted)
    for t in cands:
        if len(chosen) >= MAX_NEIGHBOURS:
            break
        pd_ = path_dict(store, t, frames)
        kept = [f for f in frames if f in pd_]
        if len(kept) < MIN_NEIGHBOUR_COVERAGE * len(frames) or not kept or kept[0] != frames[0]:
            reasons["coverage"] += 1
            continue
        if not spawnable(store, t, frames[0]):
            reasons["spawn"] += 1
            continue
        fp = footprint(store, t)
        if kerb_hit(drivable, [(np.array([pd_[f] for f in kept]), fp)]):
            reasons["kerb"] += 1
            continue
        if any(min_gap_dicts(pd_, p, fp, q) < gap_m for _, p, q in pool):
            reasons["gap"] += 1
            continue
        chosen.append((t, pd_, fp))
        pool.append((t, pd_, fp))
    return chosen, reasons


# ==========================================
# ONE SCENARIO
# ==========================================
def build_one(c, store, drivable, fidx):
    s, e = c["start"], c["end"]
    frames = list(range(s, e + 1))
    ids = [c["ego"]] + ([c["other"]] if c["other"] is not None else [])
    rec = dict(c, status=None, shift_m=None, min_carla_gap_m=None, warnings="")
    if any(store.label(t) == sc.PEDESTRIAN_LABEL for t in ids):
        rec["status"] = "pedestrian"
        return rec, None
    if sc.inbox_ratio(store, ids, s, e) < 0.8:
        rec["status"] = "in-box < 80 %"
        return rec, None
    bps = [LBP.get(store.label(t), "vehicle.nissan.micra") for t in ids]
    if len(ids) == 2:
        ok, m, why = sc.check_pair(store, *ids, s, e, *bps, overtake=c["behaviour"].startswith("Overtaking"))
    else:
        ok, m, why = sc.check_single(store, ids[0], s, e)
    rec["warnings"] = "; ".join(m.get("warnings", []))
    if not ok:
        rec["status"] = "fails checks: " + "; ".join(why)
        return rec, None
    if not all(spawnable(store, t, s) for t in ids):
        rec["status"] = "can't spawn"
        return rec, None

    common = [f for f in frames if all(store.carla_path(t).get(f) is not None for t in ids)]
    P = [np.array([store.carla_path(t)[f] for f in common]) for t in ids]
    F = [sc.BLUEPRINT_FOOTPRINT[bp] for bp in bps]
    paths = {t: path_dict(store, t, frames) for t in ids}
    gap = m.get("min_sim_gap_m")
    rec["min_carla_gap_m"] = gap
    if len(ids) == 2 and gap is not None and gap < 0:
        if gap < sep.MIN_FIXABLE_GAP:
            rec["status"] = "car contact, too deep"
            return rec, None
        res = sep.separate(P[0], P[1], F[0], F[1], drivable)
        if res is None:
            rec["status"] = "car contact, shift fails"
            return rec, None
        if kerb_hit(drivable, [(res["pa"], F[0]), (res["pb"], F[1])]):
            rec["status"] = "shifted, kerb/median hit"
            return rec, None
        rec["shift_m"], rec["shift_method"] = res["shift_m"], res["method"]
        # shifted positions replace the common frames; frames outside `common` keep the original path
        for t, q in zip(ids, (res["pa"], res["pb"])):
            paths[t].update({f: (float(x), float(y)) for f, (x, y) in zip(common, q)})
        rec["status"] = "rescued <= 0.5 m" if res["shift_m"] <= 0.5 else "rescued 0.5-0.7 m"
    else:
        if kerb_hit(drivable, list(zip(P, F))):
            rec["status"] = "kerb/median hit"
            return rec, None
        rec["status"] = "clean as recorded"

    accepted = [(t, paths[t], f) for t, f in zip(ids, F)]
    cands, dist = neighbour_candidates(store, fidx, c["ego"], set(ids), frames)
    rec["neighbour_candidates"] = len(cands)
    picks = {}
    for g in REPORT_GAPS:
        chosen, reasons = pick_neighbours(store, drivable, cands, accepted, frames, g)
        rec[f"neighbours_at_{g}m"] = len(chosen)
        picks[g] = (chosen, reasons)
    chosen, reasons = picks[NEIGHBOUR_GAP_M]
    rec["neighbours"] = len(chosen)
    rec.update({f"rejected_{k}": v for k, v in reasons.items()})

    vehicles = []
    roles = ["ego", "other"][:len(ids)]
    for role, (t, pd_, fp) in zip(roles + ["neighbour"] * len(chosen), accepted + chosen):
        vehicles.append({
            "track": int(t), "role": role, "label": store.label(t),
            "blueprint": LBP.get(store.label(t), "vehicle.nissan.micra"), "footprint": list(fp),
            "distance_to_ego_m": None if role != "neighbour" else round(dist[t], 2),
            "frames": [f for f in frames if f in pd_],
            "xy": [[round(pd_[f][0], 3), round(pd_[f][1], 3)] for f in frames if f in pd_],
            "route_cmd": route_commands(store, t, [f for f in frames if f in pd_]),
        })
    scen = {k: rec[k] for k in ("behaviour", "ego", "other", "start", "end", "src_start", "src_end", "status",
                                "shift_m", "min_carla_gap_m", "warnings")}
    scen["shift_method"] = rec.get("shift_method")
    scen["vehicles"] = vehicles
    apply_uturn_command(scen)
    return rec, scen


def apply_uturn_command(scen, frame_to_base=None):
    """U-turn scenarios: the ego's command is "u-turn" until the annotated U-turn (src_end) is over. The
    heading rule misses 5 of 18: inside an 8 s window their path turns only 90-130 deg (the turn continues
    past the window or runs through a long, blended Center pass). frame_to_base maps a copy's frame to the
    base scenario's frame (augmented copies)."""
    if scen["behaviour"] != "U-turn":
        return scen
    ego = next(v for v in scen["vehicles"] if v["role"] == "ego")
    fb = frame_to_base or (lambda f: f)
    ego["route_cmd"] = ["u-turn" if fb(f) <= scen["src_end"] else c for f, c in zip(ego["frames"], ego["route_cmd"])]
    return scen


def try_short_window(c, store, drivable, fidx, spans):
    """5 s windows that still contain the 4.2 s behaviour window (start src_start-3 .. src_start, inside the
    tracked frames), skipping any that overlaps > 50 % with a kept window of the same pair. First that
    passes all checks wins; returns (rec, scen) or (None, None)."""
    ids = [c["ego"]] + ([c["other"]] if c["other"] is not None else [])
    lo, hi = -10**9, 10**9
    for t in ids:
        a, b = contiguous_range(set(store.tracks[t].index), c["src_start"], c["src_end"])
        lo, hi = max(lo, a), min(hi, b)
    tried = set()
    for off in range(-(FALLBACK_FRAMES - (c["src_end"] - c["src_start"] + 1)), 1):
        s = min(max(c["src_start"] + off, lo), hi - FALLBACK_FRAMES + 1)
        e = s + FALLBACK_FRAMES - 1
        if s in tried or s < lo:
            continue
        tried.add(s)
        if any(min(e, b) - max(s, a) + 1 > MAX_OVERLAP * FALLBACK_FRAMES
               for a, b in spans[(c["behaviour"], c["ego"], c["other"])]):
            continue
        rec, scen = build_one(dict(c, start=s, end=e), store, drivable, fidx)
        if scen is not None:
            return rec, scen
    return None, None


# ==========================================
# MAIN
# ==========================================
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=None, help="Build only the first N selected windows.")
    args = p.parse_args()
    t0 = time.time()

    usable = pd.read_csv(os.path.join(SCEN, "usable_scenarios.csv"))
    usable = usable[usable.status.str.startswith(("clean", "rescued"))]
    print(f"{len(usable):,} usable windows; loading all tracks ...", flush=True)
    store = sc.TrackStore()
    drivable = load_map()
    print(f"  {len(store.tracks):,} tracks loaded ({time.time() - t0:.0f} s)", flush=True)

    sel = select_windows(usable, store)
    print(f"{len(sel):,} 8 s windows after <= {MAX_OVERLAP:.0%} overlap selection", flush=True)
    if args.limit:
        rng = np.random.default_rng(0)
        sel = [sel[i] for i in sorted(rng.choice(len(sel), size=min(args.limit, len(sel)), replace=False))]
    fidx = build_frame_index(store)

    recs, scens = [], []
    spans = collections.defaultdict(list)  # (behaviour, ego, other) -> kept (start, end)

    def keep(rec, scen, length):
        rec["length_frames"] = length
        recs.append(rec)
        if scen is not None:
            scen["id"], scen["length_s"] = f"{len(scens):05d}", round(length / sc.REAL_WORLD_FPS, 1)
            scens.append(scen)
            spans[(rec["behaviour"], rec["ego"], rec["other"])].append((rec["start"], rec["end"]))

    failed = []
    for i, c in enumerate(sel):
        rec, scen = build_one(c, store, drivable, fidx)
        keep(rec, scen, WINDOW_FRAMES)
        if scen is None:
            failed.append(c)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(sel)} ({time.time() - t0:.0f} s)", flush=True)

    # 5 s fallback for windows that fail at 8 s (ML_DECISIONS.md D3)
    print(f"{len(failed):,} windows failed at 8 s; trying 5 s windows ...", flush=True)
    for c in failed:
        rec, scen = try_short_window(c, store, drivable, fidx, spans)
        if rec is not None:
            keep(rec, scen, FALLBACK_FRAMES)

    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = f"_sample{args.limit}" if args.limit else ""
    json.dump(scens, open(os.path.join(OUT_DIR, f"scenarios_8s{suffix}.json"), "w"))
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(OUT_DIR, f"scenarios_8s{suffix}.csv"), index=False)

    pd.set_option("display.width", 250)
    ok = df[df.status.str.startswith(("clean", "rescued"))]
    t = pd.DataFrame({
        "8 s windows": df[df.length_frames == WINDOW_FRAMES].groupby("behaviour").size(),
        "usable 8 s": ok[ok.length_frames == WINDOW_FRAMES].groupby("behaviour").size(),
        "usable 5 s": ok[ok.length_frames == FALLBACK_FRAMES].groupby("behaviour").size(),
        "usable": ok.groupby("behaviour").size(),
        "pairs": ok.assign(pair=ok.ego.astype(str) + "_" + ok.other.astype(str)).groupby("behaviour").pair.nunique(),
        "clean": ok[ok.status == "clean as recorded"].groupby("behaviour").size(),
        "rescued": ok[ok.status != "clean as recorded"].groupby("behaviour").size(),
    }).reindex(ORDER).fillna(0).astype(int)
    t.loc["Total"] = t.sum()
    print("\n" + t.to_string())
    print("\nWhy 8 s windows failed (before the 5 s fallback):")
    print(df[~df.status.str.startswith(("clean", "rescued"))].status.str.split(":").str[0].value_counts().to_string())
    if len(ok):
        print("\nNeighbours per usable scenario (mean / share with 4):")
        for g in REPORT_GAPS:
            col = ok[f"neighbours_at_{g}m"]
            print(f"  gap >= {g} m: mean {col.mean():.2f}, with 4: {(col == 4).mean():.0%}, with 0: {(col == 0).mean():.0%}")
        print(f"  candidates within {NEIGHBOUR_RADIUS_M:.0f} m: median {ok.neighbour_candidates.median():.0f}")
        rej = [c for c in ok.columns if c.startswith("rejected_")]
        print("  rejected candidates at 0.5 m (total):", {c[9:]: int(ok[c].fillna(0).sum()) for c in rej})
        cmds = collections.Counter(v["route_cmd"][0] for s in scens for v in s["vehicles"] if v["role"] == "ego")
        print("  ego route command at window start:", dict(cmds))
        ut = collections.Counter(v["route_cmd"][0] for s in scens if s["behaviour"] == "U-turn" for v in s["vehicles"] if v["role"] == "ego")
        print("  ... U-turn scenarios only:", dict(ut))
    print(f"\nsaved {OUT_DIR}/scenarios_8s{suffix}.json ({len(scens):,} scenarios) and .csv ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
