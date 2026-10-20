#!/usr/bin/env python3
"""
debug/predict_collisions.py

Fast offline collision prediction for every scenario, without running CARLA per scenario.

For each scenario (the same pool as the behaviour table: no pedestrians, >= 80% in-box,
passing scenario_checks; vehicle following from vehicle_following_candidates_relaxed.json, 4.2 s):
  1. vehicle-vehicle : CARLA vehicle shapes placed on the converted paths every frame;
                       predicted collision if the smallest gap < --gap (default 0.15 m)
  2. static          : a vehicle shape is off the drivable area of Town10HD (driving lanes, shoulders,
                       parking, bike lanes) in >= --offroad-frames frames, e.g. a U-turn through the median
  Defaults tuned on 45 random scenarios replayed in CARLA (PID): see --validate.
  3. spawn           : the vehicle's first position is outside the 6 regions / Center, so Stage 3 cannot spawn it

  python debug/predict_collisions.py --export-map      # once; needs a CARLA server (-nullrhi is enough)
  python debug/predict_collisions.py                   # writes Scenariosssss/collision_prediction.csv
  python debug/predict_collisions.py --validate        # compares with sim_metrics_pid.json of built folders
"""

import argparse
import glob
import json
import math
import os
import sys

import cv2
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, BASE_DIR)

import scenario_checks as sc  # noqa: E402
import timemethod  # noqa: E402
from debug.check_scenarios import behaviour_of, LABEL_BLUEPRINT_MAP  # noqa: E402

SCENARIOS_DIR = os.path.join(BASE_DIR, "Scenariosssss")
MAP_FILE = os.path.join(SCENARIOS_DIR, "town10hd_drivable.npz")
POOL_FILE = os.path.join(SCENARIOS_DIR, "collision_pool.json")
OUT_FILE = os.path.join(SCENARIOS_DIR, "collision_prediction.csv")
RES = 0.2                       # m per drivable-grid cell
BOUNDS = (-160.0, 160.0, -160.0, 220.0)  # xmin, xmax, ymin, ymax (m) around the calibrated intersection
CORNER_INSET_M = 0.25           # box corners are tested this far inside the box (tolerates kerb overhang)
NAMES = {0: "3W", 1: "car", 2: "2W", 3: "bus", 5: "truck"}
VALID_SPAWN = {"Region_1", "Region_2", "Region_3", "Region_4", "Region_5", "Region_6", "Center"}


# ==========================================
# DRIVABLE AREA
# ==========================================
def export_map():
    """Rasterise the drivable surface of Town10HD (needs CARLA) into MAP_FILE."""
    import carla
    client = carla.Client("localhost", 2000)
    client.set_timeout(60.0)
    world = client.get_world()
    if "Town10HD" not in world.get_map().name:
        world = client.load_world("Town10HD")
    drivable = {carla.LaneType.Driving, carla.LaneType.Shoulder, carla.LaneType.Parking,
                carla.LaneType.Bidirectional, carla.LaneType.Biking}
    xmin, xmax, ymin, ymax = BOUNDS
    grid = np.zeros((int((ymax - ymin) / RES), int((xmax - xmin) / RES)), np.uint8)

    def stamp(wp):
        loc, f, r = wp.transform.location, wp.transform.get_forward_vector(), wp.transform.get_right_vector()
        hl, hw = 0.35, wp.lane_width / 2.0
        corners = [(loc.x + f.x * a + r.x * b, loc.y + f.y * a + r.y * b) for a, b in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]
        pts = np.array([[(x - xmin) / RES, (y - ymin) / RES] for x, y in corners], np.int32)
        cv2.fillPoly(grid, [pts], 1)

    seen = 0
    for wp in world.get_map().generate_waypoints(0.5):
        loc = wp.transform.location
        if not (xmin <= loc.x <= xmax and ymin <= loc.y <= ymax):
            continue
        stamp(wp)
        seen += 1
        for step in (lambda w: w.get_left_lane(), lambda w: w.get_right_lane()):
            side = step(wp)
            for _ in range(3):  # lanes next to the driving lane: shoulders, parking, bike lanes
                if side is None or side.lane_type not in drivable:
                    break
                if side.lane_type != carla.LaneType.Driving:
                    stamp(side)
                side = step(side)
    np.savez_compressed(MAP_FILE, grid=grid, res=RES, bounds=np.array(BOUNDS))
    print(f"Saved {MAP_FILE}: {grid.shape[1]}x{grid.shape[0]} cells, {grid.mean() * 100:.1f}% drivable, {seen} waypoints")


def load_map():
    d = np.load(MAP_FILE)
    return d["grid"], float(d["res"]), d["bounds"]


def off_road(grid, res, bounds, x, y):
    xmin, _, ymin, _ = bounds
    c, r = int((x - xmin) / res), int((y - ymin) / res)
    return not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]) or grid[r, c] == 0


# ==========================================
# POOL OF SCENARIOS
# ==========================================
def build_pool():
    fr = lambda v: int(str(v).split("_")[-1])
    pool = []
    for path in ("Cut In/cut_in_out.csv", "U-turn/u_turn.csv", "Overtaking Left/overtaking.csv"):
        df = pd.read_csv(os.path.join(SCENARIOS_DIR, path))
        rows = [{"ego": int(r.trackID), "other": int(r.other_id) if hasattr(r, "other_id") else None,
                 "start": fr(r.start_frame), "end": fr(r.end_frame), "behaviour": behaviour_of(r)} for r in df.itertuples()]
        pool += sc.dedupe_windows(rows)
    vf = os.path.join(SCENARIOS_DIR, "Vehicle Following", "vehicle_following_candidates_relaxed.json")
    if os.path.exists(vf):
        for r in json.load(open(vf)):
            if r["n"] >= 18:
                pool.append({"ego": int(r["F"]), "other": int(r["L"]), "start": int(r["s"]), "end": int(r["e"]),
                             "behaviour": "Vehicle Following"})
    return pool


# ==========================================
# PREDICTION
# ==========================================
def predict(args):
    grid, res, bounds = load_map()
    pool = build_pool()
    ids = {r["ego"] for r in pool} | {r["other"] for r in pool if r["other"] is not None}
    print(f"{len(pool):,} candidate windows; loading {len(ids):,} tracks ...")
    store = sc.TrackStore(track_ids=ids)
    rows, kept = [], []
    for r in pool:
        tids = [r["ego"]] + ([r["other"]] if r["other"] is not None else [])
        if any(t not in store.tracks or store.label(t) == sc.PEDESTRIAN_LABEL for t in tids):
            continue
        inbox = sc.inbox_ratio(store, tids, r["start"], r["end"])
        if inbox < 0.8:
            continue
        bps = [LABEL_BLUEPRINT_MAP.get(store.label(t), "vehicle.nissan.micra") for t in tids]
        if len(tids) == 2:
            ok, m, _ = sc.check_pair(store, tids[0], tids[1], r["start"], r["end"], bps[0], bps[1],
                                     overtake=r["behaviour"].startswith("Overtaking"))
        else:
            ok, m, _ = sc.check_single(store, tids[0], r["start"], r["end"])
        if not ok:
            continue
        kept.append(dict(r, inbox_ratio=round(inbox, 3)))

        # vehicle shapes along the converted paths, every frame both are present
        frames = [f for f in range(r["start"], r["end"] + 1)
                  if all(store.carla_path(t).get(f) is not None for t in tids)]
        boxes = {}
        for t, bp in zip(tids, bps):
            pts = [store.carla_path(t)[f] for f in frames]
            yaws = sc.path_yaws(pts) if len(pts) > 1 else [0.0] * len(pts)
            L, W = sc.BLUEPRINT_FOOTPRINT[bp]
            boxes[t] = [(p, y, sc.box_corners(p, y, L, W), sc.box_corners(p, y, L - 2 * CORNER_INSET_M, max(W - 2 * CORNER_INSET_M, 0.1)))
                        for p, y in zip(pts, yaws)]
        gap = min(sc.box_gap(a[2], b[2]) for a, b in zip(boxes[tids[0]], boxes[tids[1]])) if len(tids) == 2 and frames else float("nan")
        off_frames = sum(1 for t in tids for (p, _, _, inner) in boxes[t]
                         if off_road(grid, res, bounds, *p) or sum(off_road(grid, res, bounds, *c) for c in inner) >= 2)
        spawn_ok = all(timemethod.classify_point(store.tracks[t].loc[store.tracks[t].index >= r["start"]].iloc[0][["kf_cx_pixel", "kf_cy_pixel"]].to_numpy() * [1, -1]) in VALID_SPAWN for t in tids)
        pred_vehicle = bool(len(tids) == 2 and gap < args.gap)
        pred_static = off_frames >= args.offroad_frames
        rows.append({
            "behaviour": r["behaviour"], "ego": r["ego"], "other": r["other"], "start": r["start"], "end": r["end"],
            "types": "-".join(NAMES.get(store.label(t), "?") for t in tids),
            "min_carla_gap_m": round(gap, 3), "min_real_gap_m": m.get("min_real_gap_m"),
            "offroad_frames": off_frames, "spawn_ok": spawn_ok,
            "predicted_vehicle_collision": pred_vehicle, "predicted_static_hit": pred_static,
            "predicted_collision_free": not (pred_vehicle or pred_static) and spawn_ok,
            "warnings": "; ".join(m.get("warnings", [])),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, index=False)
    with open(POOL_FILE, "w") as f:
        json.dump(kept, f)
    print(f"Saved {OUT_FILE} ({len(df):,} scenarios) and {POOL_FILE}")
    summary = df.groupby("behaviour").agg(scenarios=("ego", "size"), vehicle_collision=("predicted_vehicle_collision", "sum"),
                                          static_hit=("predicted_static_hit", "sum"), spawn_fail=("spawn_ok", lambda s: (~s).sum()),
                                          collision_free=("predicted_collision_free", "sum"))
    summary.loc["Total"] = summary.sum()
    print(summary.to_string())


def validate(args):
    """Prediction vs. CARLA PID result for every built '<Behaviour> Validated' folder."""
    pred = pd.read_csv(OUT_FILE)
    pred["key"] = pred.behaviour + "|" + pred.ego.astype(str) + "|" + pred.other.fillna(-1).astype(int).astype(str) + "|" + pred.start.astype(str)
    pred = pred.set_index("key")
    from overlay_scenario_bboxes import parse_summary_file
    rows = []
    for d in sorted(glob.glob(os.path.join(SCENARIOS_DIR, "* Validated", "scenario_*"))):
        st, en, vids, _, ego = parse_summary_file(glob.glob(os.path.join(d, "*_summary.txt"))[0])
        b = os.path.basename(os.path.dirname(d)).replace(" Validated", "")
        other = [v for v in vids if v != ego]
        key = f"{b}|{ego}|{other[0] if other else -1}|{st}"
        if key not in pred.index:
            continue
        p = pred.loc[key]
        mf = os.path.join(d, "sim_metrics_pid.json")
        if os.path.exists(mf) and json.load(open(mf))["vehicles"] < len(vids):
            actual = "spawn failed"  # only part of the scenario could be spawned
        elif os.path.exists(mf):
            m = json.load(open(mf))
            actual = "vehicle" if m["collision_events"] else ("static" if m["static_collisions"] else
                     ("overlap" if (m["min_box_gap_m"] == m["min_box_gap_m"] and m["min_box_gap_m"] < 0) else "clean"))
        else:
            actual = "spawn failed"
        predicted = "clean" if p.predicted_collision_free else ("spawn failed" if not p.spawn_ok else
                    ("vehicle" if p.predicted_vehicle_collision else "static"))
        rows.append({"scenario": f"{b}/{os.path.basename(d)}", "predicted": predicted, "actual": actual,
                     "pred_gap": p.min_carla_gap_m, "offroad_frames": p.offroad_frames})
    v = pd.DataFrame(rows)
    print(v.to_string(index=False))
    bad_actual = v.actual != "clean"
    bad_pred = v.predicted != "clean"
    tp, fn = (bad_actual & bad_pred).sum(), (bad_actual & ~bad_pred).sum()
    fp, tn = (~bad_actual & bad_pred).sum(), (~bad_actual & ~bad_pred).sum()
    print(f"\n{len(v)} scenarios: problem caught {tp}/{tp + fn}, problem missed {fn}, clean wrongly flagged {fp}/{fp + tn}, clean kept {tn}")
    print(f"accuracy {(tp + tn) / max(len(v), 1) * 100:.0f}%")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export-map", action="store_true")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--gap", type=float, default=0.15, help="Predicted vehicle collision below this CARLA-shape gap (m).")
    p.add_argument("--offroad-frames", type=int, default=5, help="Predicted static hit at this many off-road vehicle-frames.")
    args = p.parse_args()
    if args.export_map:
        export_map()
    elif args.validate:
        validate(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
