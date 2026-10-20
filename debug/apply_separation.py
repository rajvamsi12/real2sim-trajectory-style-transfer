#!/usr/bin/env python3
"""
debug/apply_separation.py

Removes small CARLA-body overlaps (about -0.5 m .. 0 m) by moving whole trajectories apart, so the
manoeuvre keeps its shape and timing. The CARLA vehicles are larger than the real ones (Micra
1.85 m vs ~1.65 m wide), so a real pass with a few cm to spare overlaps in CARLA.

Two ways, whichever needs the smaller change:
  sideways : both vehicles move away from each other, half the distance each, perpendicular to their
             own heading every frame (works on curves and in the junction)
  delay    : the vehicle behind is moved back along its own path (front-to-back contact: following,
             cut-in, cut-out)
The shift is the overlap plus a margin (0.2 m) for PID tracking error, at most MAX_SHIFT_M (0.7 m), and
only for overlaps down to MIN_FIXABLE_GAP (-0.5 m). The shifted paths must stay on the drivable area
(town10hd_drivable.npz, from predict_collisions.py --export-map).

  python debug/apply_separation.py --count                       # offline: how many could be fixed
  python debug/apply_separation.py --apply "<scenario dir>" ...  # shift built folders, redo Stage 3 (CARLA)
"""

import argparse
import collections
import json
import math
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, BASE_DIR)

import scenario_checks as sc  # noqa: E402

MIN_FIXABLE_GAP = -0.5    # overlaps deeper than this are tracking errors, not size differences
TARGET_GAP = 0.2          # gap required after the shift (PID tracking margin)
MAX_SHIFT_M = 0.7         # largest allowed shift; above ~0.5 m only about half replay cleanly in CARLA
MAX_STEPS = 20            # shift search: overlap + margin, then +0.05 m steps


def _yaws(pts):
    return np.array(sc.path_yaws(pts)) if len(pts) > 1 else np.zeros(len(pts))


def min_gap(pa, pb, fa, fb):
    ya, yb = _yaws(pa), _yaws(pb)
    return min(sc.box_gap(sc.box_corners(pa[i], ya[i], *fa), sc.box_corners(pb[i], yb[i], *fb)) for i in range(len(pa)))


def _worst_frame(pa, pb, fa, fb):
    ya, yb = _yaws(pa), _yaws(pb)
    g = [sc.box_gap(sc.box_corners(pa[i], ya[i], *fa), sc.box_corners(pb[i], yb[i], *fb)) for i in range(len(pa))]
    return int(np.argmin(g))


def shift_sideways(pa, pb, d, i):
    """Move a and b apart by d in total (d/2 each), perpendicular to each one's own heading."""
    ya, yb = _yaws(pa), _yaws(pb)
    na = np.stack([-np.sin(ya), np.cos(ya)], 1)      # left normals
    nb = np.stack([-np.sin(yb), np.cos(yb)], 1)
    side_a = -np.sign((pb[i] - pa[i]) @ na[i]) or 1.0  # a moves away from b
    side_b = -np.sign((pa[i] - pb[i]) @ nb[i]) or 1.0
    return pa + na * side_a * d / 2, pb + nb * side_b * d / 2


def shift_back(p, d):
    """Path moved back by d metres along itself (frame k shows where the vehicle was d metres earlier)."""
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    first = p[1] - p[0] if seg[0] > 1e-6 else np.array([1.0, 0.0])
    first = first / max(np.linalg.norm(first), 1e-9)
    out = []
    for sk in s - d:
        if sk >= 0:
            out.append([np.interp(sk, s, p[:, 0]), np.interp(sk, s, p[:, 1])])
        else:  # before the recorded start: continue straight backwards
            out.append(p[0] + first * sk)
    return np.array(out)


def separate(pa, pb, fa, fb, drivable=None):
    """Smallest change that brings the CARLA-body gap to >= TARGET_GAP. Returns dict or None."""
    gap0 = min_gap(pa, pb, fa, fb)
    if gap0 >= TARGET_GAP:
        return {"method": "none", "shift_m": 0.0, "gap_before": gap0, "gap_after": gap0, "pa": pa, "pb": pb}
    if gap0 < MIN_FIXABLE_GAP:
        return None
    i = _worst_frame(pa, pb, fa, fb)
    base_off = [_offroad_count(drivable, q) for q in (pa, pb)] if drivable is not None else None
    heading = np.array([math.cos(_yaws(pa)[i]), math.sin(_yaws(pa)[i])])
    a_behind = (pb[i] - pa[i]) @ heading > 0
    best = None
    for step in range(MAX_STEPS):
        d = (TARGET_GAP - gap0) + 0.05 * step
        if d > MAX_SHIFT_M + 1e-9:
            break
        options = [("sideways",) + shift_sideways(pa, pb, d, i)]
        if a_behind:
            options.append(("delay ego",) + (shift_back(pa, d), pb))
        else:
            options.append(("delay other",) + (pa, shift_back(pb, d)))
        for name, qa, qb in options:
            g = min_gap(qa, qb, fa, fb)
            if g < TARGET_GAP:
                continue
            # the shift must not put a vehicle off the drivable area more often than the original path
            if drivable is not None and any(_offroad_count(drivable, q) > b for q, b in zip((qa, qb), base_off)):
                continue
            cand = {"method": name, "shift_m": round(d, 3), "gap_before": gap0, "gap_after": g, "pa": qa, "pb": qb}
            if best is None or cand["shift_m"] < best["shift_m"]:
                best = cand
        if best is not None:
            return best
    return None


def _offroad_count(drivable, path):
    grid, res, bounds = drivable
    from debug.predict_collisions import off_road
    return sum(off_road(grid, res, bounds, x, y) for x, y in path)


def _load_drivable():
    from debug.predict_collisions import MAP_FILE, load_map
    return load_map() if os.path.exists(MAP_FILE) else None


# ==========================================
# OFFLINE COUNT
# ==========================================
def count():
    from debug.check_scenarios import behaviour_of, LABEL_BLUEPRINT_MAP as LBP
    fr = lambda v: int(str(v).split("_")[-1])
    drivable = _load_drivable()
    rows = []
    for path in ("Cut In/cut_in_out.csv", "Overtaking Left/overtaking.csv"):
        df = pd.read_csv(os.path.join(BASE_DIR, "Scenariosssss", path))
        rows += [{"ego": int(r.trackID), "other": int(r.other_id), "start": fr(r.start_frame), "end": fr(r.end_frame),
                  "b": behaviour_of(r)} for r in df.itertuples()]
    vf = os.path.join(BASE_DIR, "Scenariosssss", "Vehicle Following", "vehicle_following_candidates_relaxed.json")
    rows += [{"ego": int(r["F"]), "other": int(r["L"]), "start": int(r["s"]), "end": int(r["e"]), "b": "Vehicle Following"}
             for r in json.load(open(vf)) if r["n"] >= 18]
    store = sc.TrackStore(track_ids={r["ego"] for r in rows} | {r["other"] for r in rows})
    st = collections.defaultdict(collections.Counter)
    methods = collections.Counter()
    fixed = []
    for r in rows:
        ids = [r["ego"], r["other"]]
        if any(t not in store.tracks or store.label(t) == sc.PEDESTRIAN_LABEL for t in ids):
            continue
        if sc.inbox_ratio(store, ids, r["start"], r["end"]) < 0.8:
            continue
        bps = [LBP.get(store.label(t), "vehicle.nissan.micra") for t in ids]
        ok, m, _ = sc.check_pair(store, ids[0], ids[1], r["start"], r["end"], *bps, overtake=r["b"].startswith("Overtaking"))
        if not ok or m.get("min_sim_gap_m") is None:
            continue
        s = st[r["b"]]
        s["pass"] += 1
        g = m["min_sim_gap_m"]
        if g >= 0:
            s["no overlap"] += 1
            continue
        if g < MIN_FIXABLE_GAP:
            s["overlap < -0.5 m"] += 1
            continue
        s["overlap -0.5..0 m"] += 1
        ca, cb = store.carla_path(ids[0]), store.carla_path(ids[1])
        frames = [f for f in range(r["start"], r["end"] + 1) if ca.get(f) is not None and cb.get(f) is not None]
        res = separate(np.array([ca[f] for f in frames]), np.array([cb[f] for f in frames]),
                       sc.BLUEPRINT_FOOTPRINT[bps[0]], sc.BLUEPRINT_FOOTPRINT[bps[1]], drivable)
        if res is None:
            s["not fixable"] += 1
        else:
            s["fixed"] += 1
            methods[res["method"]] += 1
            fixed.append(dict(r, behaviour=r["b"], method=res["method"], shift_m=res["shift_m"],
                              gap_before=round(res["gap_before"], 3), gap_after=round(res["gap_after"], 3)))
    cols = ["pass", "no overlap", "overlap -0.5..0 m", "fixed", "not fixable", "overlap < -0.5 m"]
    table = pd.DataFrame({b: {c: st[b][c] for c in cols} for b in st}).T
    table.loc["Total"] = table.sum()
    print(table.to_string())
    print("methods used:", dict(methods))
    sh = pd.DataFrame(fixed)
    print(f"shift applied: median {sh.shift_m.median():.2f} m, 90th pct {sh.shift_m.quantile(.9):.2f} m, max {sh.shift_m.max():.2f} m")
    out = os.path.join(BASE_DIR, "Scenariosssss", "separation_fixable.json")
    sh.drop(columns=["b"]).to_json(out, orient="records", indent=1)
    print(f"saved {out}")


# ==========================================
# APPLY TO BUILT FOLDERS
# ==========================================
def apply(dirs):
    import carla
    from simulate_and_record_scenarios import AnnotationParser, STABILIZE_EKF_CSV, SIM_FPS, TOWN
    from overlay_scenario_bboxes import parse_summary_file
    drivable = _load_drivable()
    parser = AnnotationParser(STABILIZE_EKF_CSV)
    client = carla.Client("localhost", 2000)
    client.set_timeout(60.0)
    world = client.get_world()
    if TOWN not in world.get_map().name:
        world = client.load_world(TOWN)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / SIM_FPS
    world.apply_settings(settings)
    from debug.check_scenarios import LABEL_BLUEPRINT_MAP as LBP
    for d in dirs:
        summary = [f for f in os.listdir(d) if f.endswith("_summary.txt")][0]
        summary_data = parse_summary_file(os.path.join(d, summary))
        traj_path = os.path.join(d, "time_variant_trajectories.json")
        orig_path = os.path.join(d, "time_variant_trajectories_original.json")
        traj = json.load(open(orig_path if os.path.exists(orig_path) else traj_path))
        ids = list(traj)
        if len(ids) != 2:
            print(f"{d}: not a two-vehicle scenario, skipped")
            continue
        fa, fb = (sc.BLUEPRINT_FOOTPRINT[LBP.get(parser.get_label_for_track(int(t)), "vehicle.nissan.micra")] for t in ids)
        common = sorted({p["frame"] for p in traj[ids[0]]["carla_coordinates"]} & {p["frame"] for p in traj[ids[1]]["carla_coordinates"]})
        pa = np.array([[p["x"], p["y"]] for p in traj[ids[0]]["carla_coordinates"] if p["frame"] in common])
        pb = np.array([[p["x"], p["y"]] for p in traj[ids[1]]["carla_coordinates"] if p["frame"] in common])
        res = separate(pa, pb, fa, fb, drivable)
        if res is None:
            print(f"{d}: overlap not fixable")
            continue
        if not os.path.exists(orig_path):
            json.dump(traj, open(orig_path, "w"), indent=2)
        new = {}
        for tid, q in ((ids[0], res["pa"]), (ids[1], res["pb"])):
            new[tid] = {"carla_coordinates": [{"frame": f, "x": float(x), "y": float(y)} for f, (x, y) in zip(common, q)]}
        json.dump(new, open(traj_path, "w"), indent=2)
        info = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k not in ("pa", "pb")}
        json.dump(info, open(os.path.join(d, "separation.json"), "w"), indent=2)
        print(f"{d}: {info}")
        run_stage_3_spawns_from_shifted(world, d, summary_data, parser, new)
        world.tick()


def run_stage_3_spawns_from_shifted(world, d, summary_data, parser, new):
    """Stage 3 reads positions from the dataset, so patch the shifted first points into the spawn JSONs."""
    from debug.batch_process_scenarios import run_stage_3_spawns
    run_stage_3_spawns(d, summary_data, parser, dji_spawn_params=None, world=world)
    for name in ("spawn_positions_raw.json", "spawn_positions_retracted.json"):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            continue
        sp = json.load(open(p))
        for tid, s in sp.items():
            if tid in new and new[tid]["carla_coordinates"]:
                first = new[tid]["carla_coordinates"][0]
                s["x"], s["y"] = first["x"], first["y"]
        json.dump(sp, open(p, "w"), indent=2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--count", action="store_true")
    p.add_argument("--apply", nargs="+", metavar="DIR")
    args = p.parse_args()
    if args.count:
        count()
    elif args.apply:
        apply(args.apply)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
