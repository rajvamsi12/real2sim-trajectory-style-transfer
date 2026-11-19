#!/usr/bin/env python3
"""
il/augment_scenarios.py

Step 4 (ML_DECISIONS.md A1-A6): 10 augmented copies of every collision-free scenario
(Scenariosssss/IL/replay_cureE.csv, result "collision-free"):

  copies 0-4  scene noise   - every vehicle's path shifted sideways by a slowly varying offset
                              (constant + one slow sine, |offset| <= 0.5 m), the whole scene sped up or slowed
                              down by one factor in [0.9, 1.1] (positions resampled in time, so the vehicles
                              keep their timing relative to each other), and the start delayed by 0-2 frames
                              (up to 0.5 s). Scene rotation is not used (the road doesn't rotate with it).
  copies 3-6  ego start     - only the ego changes: either it starts 0.5-1.0 m beside its path and the offset
              offset        shrinks smoothly to 0 over RECOVER_S, or it starts on its path turned 5-10 deg
                              (spawn_yaw_offset_deg; the replay steers it back). The training label for these
                              copies is the real path.
  copies 7-9  ego speed     - only the ego changes (added 2026-11-10: the LHD model never learned to brake): it moves
              recovery      along its own real path but starts too fast (x1.15-1.35, 70 %) or too slow (x0.70-0.85),
                              its along-track lead/lag d(t) = (f - 1) v0 t (1 - t/T)^2 returns to 0 at T = SPEED_RECOVER_S,
                              so its future path (the label) brakes or speeds back to the real timing. An ego slower
                              than SPEED_MIN at the start instead starts SPEED_DV faster (d(t) = dv t (1 - t/T)^2) and
                              brakes back. The copy's path must never move backwards; dv / (f - 1) is halved up to
                              3 times until it does not, else the slot gets an ego-offset copy.
                              (2026-11-10, 2nd version: T 2.5 -> 3.5 s, fast x1.15-1.35 -> x1.2-1.5, SPEED_DV added;
                              the 1st version made 54 % fallbacks and a median max lead of 0.26 m)
  (was 5 scene noise + 5 ego offset until 2026-11-10)

Every copy is checked like the base scenarios; a failing copy is redrawn up to MAX_TRIES times, else dropped:
  - behaviour kept: ego vs other, longitudinal order at start and end and the direction of the lateral change
    (when it was > 0.5 m) are the same as in the base scenario
  - gaps: every pair keeps CARLA-body gap >= min(0.5 m, its gap in the base scenario) - 0.05 m
  - drivable area: no vehicle >= 5 frames off Town10HD's drivable raster (same rule as the builder)
The CARLA physics check comes after:  python il/replay_scenarios.py --scenarios .../scenarios_aug.json ...

  python il/augment_scenarios.py      # -> Scenariosssss/IL/scenarios_aug.json (+ .csv with one row per copy)
"""

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
sys.path.insert(0, SCRIPT_DIR)

import scenario_checks as sc  # noqa: E402
from debug.predict_collisions import load_map  # noqa: E402
from build_scenarios_8s import kerb_hit  # noqa: E402

IL_DIR = os.path.join(BASE_DIR, "Scenariosssss", "IL")
FPS = sc.REAL_WORLD_FPS
N_NOISE, N_EGO, N_SPEED = 3, 4, 3
MAX_TRIES = 5
OFFSET_CONST_SD = 0.15      # m, constant part of the sideways offset
OFFSET_WAVE_SD = 0.15       # m, amplitude of the slow sine
OFFSET_PERIOD_S = (4.0, 10.0)
OFFSET_MAX = 0.5            # m
SPEED_SCALE = (0.9, 1.1)
MAX_TRIM_FRAMES = 2         # start delayed by up to 2 frames (0.47 s)
EGO_OFFSET_M = (0.5, 1.0)
EGO_YAW_DEG = (5.0, 10.0)
RECOVER_S = 2.0
GAP_FLOOR = 0.5
GAP_TOL = 0.05
LAT_CHANGE_MIN = 0.5        # m, lateral change that has to keep its direction
SPEED_FAST, SPEED_SLOW, P_FAST = (1.2, 1.5), (0.70, 0.85), 0.7
SPEED_RECOVER_S = 3.5       # s, the ego is back on the real timing after this
SPEED_MIN = 1.5             # m/s, slower egos get an added start speed instead of a factor
SPEED_DV = (1.0, 2.0)       # m/s, added start speed for egos slower than SPEED_MIN
SPEED_SHRINK = 3            # halvings of the push tried when the path would move backwards


# ==========================================
# GEOMETRY
# ==========================================
def resample(frames, xy, t_new):
    """Position at times t_new (s from the vehicle's own first frame grid), linear between recorded frames."""
    t = (np.asarray(frames, float) - frames[0]) / FPS
    xy = np.asarray(xy, float)
    return np.stack([np.interp(t_new, t, xy[:, 0]), np.interp(t_new, t, xy[:, 1])], 1)


def normals(xy):
    yaw = np.array(sc.path_yaws(xy)) if len(xy) > 1 else np.zeros(len(xy))
    return np.stack([-np.sin(yaw), np.cos(yaw)], 1)


def pair_gaps(vehicles):
    """{(i, j): min CARLA-body gap over shared frames}."""
    out = {}
    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            a, b = vehicles[i], vehicles[j]
            fa = {f: k for k, f in enumerate(a["frames"])}
            common = [f for f in b["frames"] if f in fa]
            if not common:
                continue
            fb = {f: k for k, f in enumerate(b["frames"])}
            A = np.array([a["xy"][fa[f]] for f in common])
            B = np.array([b["xy"][fb[f]] for f in common])
            ya = sc.path_yaws(A) if len(A) > 1 else [0.0]
            yb = sc.path_yaws(B) if len(B) > 1 else [0.0]
            out[(i, j)] = min(sc.box_gap(sc.box_corners(A[k], ya[k], *a["footprint"]),
                                         sc.box_corners(B[k], yb[k], *b["footprint"])) for k in range(len(common)))
    return out


def relation(vehicles):
    """Ego vs other: (sign of longitudinal offset at start, at end, lateral change) along the ego's travel."""
    if len(vehicles) < 2:
        return None
    e, o = vehicles[0], vehicles[1]
    fo = {f: k for k, f in enumerate(o["frames"])}
    common = [(k, fo[f]) for k, f in enumerate(e["frames"]) if f in fo]
    if len(common) < 2:
        return None
    E = np.array([e["xy"][k] for k, _ in common])
    O = np.array([o["xy"][k] for _, k in common])
    d = E[-1] - E[0]
    if np.linalg.norm(d) < 1.0:
        return None
    r = d / np.linalg.norm(d)
    n = np.array([-r[1], r[0]])
    lon, lat = (E - O) @ r, (E - O) @ n
    return np.sign(lon[0]), np.sign(lon[-1]), lat[-1] - lat[0]


def same_behaviour(base_rel, new_rel):
    if base_rel is None:
        return True
    if new_rel is None:
        return False
    if base_rel[0] != new_rel[0] or base_rel[1] != new_rel[1]:
        return False
    if abs(base_rel[2]) > LAT_CHANGE_MIN and np.sign(base_rel[2]) != np.sign(new_rel[2]):
        return False
    return True


# ==========================================
# COPIES
# ==========================================
def scene_noise(base, rng):
    s = rng.uniform(*SPEED_SCALE)
    trim = int(rng.integers(0, MAX_TRIM_FRAMES + 1))
    n_base = base["end"] - base["start"] + 1
    n = min(n_base - trim, int((n_base - 1 - trim) / s) + 1)
    start = base["start"] + trim
    vehicles, params = [], []
    for v in base["vehicles"]:
        v0 = v["frames"][0]
        # new frame k (absolute start + k) shows the base scene at base time trim + s * k
        t_base = (trim + s * np.arange(n)) / FPS - (v0 - base["start"]) / FPS
        keep = (t_base >= 0) & (t_base <= (v["frames"][-1] - v0) / FPS + 1e-6)
        frames = [start + k for k in range(n) if keep[k]]
        if len(frames) < 2:
            return None, None
        xy = resample(v["frames"], v["xy"], t_base[keep])
        c, a = rng.normal(0, OFFSET_CONST_SD), rng.normal(0, OFFSET_WAVE_SD)
        per, ph = rng.uniform(*OFFSET_PERIOD_S), rng.uniform(0, 2 * math.pi)
        tt = np.arange(len(frames)) / FPS
        off = np.clip(c + a * np.sin(2 * math.pi * tt / per + ph), -OFFSET_MAX, OFFSET_MAX)
        xy = xy + normals(xy) * off[:, None]
        base_idx = [int(round(trim + s * (f - start))) - (v0 - base["start"]) for f in frames]
        cmd = [v["route_cmd"][min(max(i, 0), len(v["route_cmd"]) - 1)] for i in base_idx]
        vehicles.append(dict(v, frames=frames, xy=np.round(xy, 3).tolist(), route_cmd=cmd))
        params.append({"track": v["track"], "offset_const_m": round(c, 3), "offset_wave_m": round(a, 3),
                       "offset_period_s": round(per, 2), "offset_phase": round(ph, 3)})
    copy = dict(base, start=start, end=start + n - 1, length_s=round(n / FPS, 1), vehicles=vehicles)
    return copy, {"type": "scene_noise", "speed_scale": round(s, 3), "trim_frames": trim, "vehicles": params}


def ego_offset(base, rng):
    vehicles = [dict(v) for v in base["vehicles"]]
    e = vehicles[0]
    xy = np.asarray(e["xy"], float)
    sign = rng.choice([-1.0, 1.0])
    if rng.random() < 0.5:
        d = sign * rng.uniform(*EGO_OFFSET_M)
        tt = (np.asarray(e["frames"]) - e["frames"][0]) / FPS
        w = np.where(tt < RECOVER_S, 0.5 * (1 + np.cos(math.pi * tt / RECOVER_S)), 0.0)
        e["xy"] = np.round(xy + normals(xy) * (d * w)[:, None], 3).tolist()
        params = {"type": "ego_offset", "mode": "lateral", "offset_m": round(d, 3), "recover_s": RECOVER_S}
    else:
        deg = sign * rng.uniform(*EGO_YAW_DEG)
        e["spawn_yaw_offset_deg"] = round(float(deg), 2)
        params = {"type": "ego_offset", "mode": "heading", "yaw_offset_deg": round(float(deg), 2)}
    return dict(base, vehicles=vehicles), params


def ego_speed(base, rng):
    """Ego starts too fast / too slow along its real path and returns to the real timing within SPEED_RECOVER_S."""
    vehicles = [dict(v) for v in base["vehicles"]]
    e = vehicles[0]
    xy = np.asarray(e["xy"], float)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s_cum = np.r_[0.0, np.cumsum(seg)]
    tt = (np.asarray(e["frames"]) - e["frames"][0]) / FPS
    v0 = (s_cum[min(2, len(s_cum) - 1)]) / max(tt[min(2, len(tt) - 1)], 1e-6)
    T = SPEED_RECOVER_S
    if v0 < SPEED_MIN:
        mode, f = "added", None
        dv = rng.uniform(*SPEED_DV)
    else:
        mode = "factor"
        f = rng.uniform(*SPEED_FAST) if rng.random() < P_FAST else rng.uniform(*SPEED_SLOW)
        dv = (f - 1.0) * v0
    for shrink in range(SPEED_SHRINK + 1):
        lead = np.where(tt < T, dv * tt * (1.0 - tt / T) ** 2, 0.0)
        s_new = np.clip(s_cum + lead, 0.0, s_cum[-1])
        if np.diff(s_new).min() >= -1e-3:   # never backwards
            break
        dv *= 0.5
    else:
        copy, params = ego_offset(base, rng)
        return copy, dict(params, fallback_from="ego_speed")
    e["xy"] = np.round(np.stack([np.interp(s_new, s_cum, xy[:, 0]), np.interp(s_new, s_cum, xy[:, 1])], 1), 3).tolist()
    return dict(base, vehicles=vehicles), {"type": "ego_speed", "mode": mode,
                                           "speed_factor": None if f is None else round(float(1 + dv / v0), 3),
                                           "added_mps": round(float(dv), 2), "shrunk": shrink,
                                           "v0_mps": round(float(v0), 2), "recover_s": T,
                                           "max_lead_m": round(float(np.abs(s_new - s_cum).max()), 2)}


def check(copy, base_gaps, base_rel, drivable):
    v = copy["vehicles"]
    if not same_behaviour(base_rel, relation(v)):
        return "behaviour changed"
    g = pair_gaps(v)
    for k, gap in g.items():
        if gap < min(GAP_FLOOR, base_gaps.get(k, GAP_FLOOR)) - GAP_TOL:
            return "gap"
    if any(kerb_hit(drivable, [(np.asarray(x["xy"]), tuple(x["footprint"]))]) for x in v):
        return "off drivable area"
    return None


# ==========================================
# MAIN
# ==========================================
def main():
    t0 = time.time()
    rep = pd.read_csv(os.path.join(IL_DIR, "replay_cureE.csv"), dtype={"id": str})
    ok_ids = set(rep[rep.result == "collision-free"].id)
    scens = [s for s in json.load(open(os.path.join(IL_DIR, "scenarios_8s.json"))) if s["id"] in ok_ids]
    drivable = load_map()
    print(f"{len(scens):,} base scenarios", flush=True)
    copies, rows = [], []
    for i, base in enumerate(scens):
        base_gaps, base_rel = pair_gaps(base["vehicles"]), relation(base["vehicles"])
        rng = np.random.default_rng(int(base["id"]))
        for k in range(N_NOISE + N_EGO + N_SPEED):
            why = None
            for attempt in range(MAX_TRIES):
                if k < N_NOISE:
                    copy, params = scene_noise(base, rng)
                    if copy is None:
                        why = "too short"
                        continue
                elif k < N_NOISE + N_EGO:
                    copy, params = ego_offset(base, rng)
                else:
                    copy, params = ego_speed(base, rng)
                why = check(copy, base_gaps, base_rel, drivable)
                if why is None:
                    break
            row = {"id": f"{base['id']}_{k}", "base_id": base["id"], "behaviour": base["behaviour"], "copy": k,
                   "type": "scene_noise" if k < N_NOISE else ("ego_offset" if k < N_NOISE + N_EGO else "ego_speed"),
                   "tries": attempt + 1,
                   "result": "ok" if why is None else f"dropped: {why}"}
            rows.append(row)
            if why is None:
                copies.append(dict(copy, id=row["id"], base_id=base["id"], aug=params))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(scens)} ({time.time() - t0:.0f} s)", flush=True)
    json.dump(copies, open(os.path.join(IL_DIR, "scenarios_aug.json"), "w"))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(IL_DIR, "scenarios_aug.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n" + pd.crosstab(df.behaviour, [df.type, df.result.str.split(":").str[0]]).to_string())
    print("\ndrop reasons:", df[df.result != "ok"].result.value_counts().to_dict())
    print("tries needed (ok copies):", df[df.result == "ok"].tries.value_counts().sort_index().to_dict())
    print(f"saved {len(copies):,} copies -> {IL_DIR}/scenarios_aug.json ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
