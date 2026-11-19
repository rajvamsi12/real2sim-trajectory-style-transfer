#!/usr/bin/env python3
"""
il/diagnose_collisions.py

Why do scenarios collide in the CARLA replay (il/replay_scenarios.py)? Replays the given scenarios again
(same PID replay) and logs, per scenario:
  - first collision: time, the two roles, the static object type
  - per vehicle: CARLA bounding-box size, spawn yaw error vs. the path direction, worst cross-track drift
    from its recorded path, drift and along-track lag at the moment of the first collision
  - "planned" gaps: the recorded (reference) paths checked with the REAL CARLA box sizes, i.e. would the
    cars touch even with perfect tracking?  (min over all pairs, and for the colliding pair)
  - reference-path kerb frames with the real box sizes (off the Town10HD drivable raster)

  python il/diagnose_collisions.py --ids collisions --clean-sample 100
  -> Scenariosssss/IL/diagnose_pid.csv
"""

import argparse
import contextlib
import io
import json
import math
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, SCRIPT_DIR)

import carla  # noqa: E402
import scenario_checks as sc  # noqa: E402
from simulate_and_record_scenarios import actor_box, SIM_FPS, TOWN, DEFAULT_Z  # noqa: E402
from debug.batch_process_scenarios import SETTLE_TICKS  # noqa: E402
from debug.predict_collisions import load_map, off_road  # noqa: E402
from replay_scenarios import IL_DIR, timed_route, clear_world, spawn_vehicle  # noqa: E402


def cross_track(sv, x, y):
    """Distance from (x, y) to the vehicle's reference polyline."""
    xy = sv.timed_route[1]
    if len(xy) < 2:
        return float(np.hypot(*(xy[0] - [x, y])))
    a, b = xy[:-1], xy[1:]
    ab = b - a
    u = np.clip(((np.array([x, y]) - a) * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-9), 0, 1)
    p = a + ab * u[:, None]
    return float(np.min(np.hypot(p[:, 0] - x, p[:, 1] - y)))


def planned_gaps(sims, roles, ticks):
    """Min box gap between every pair on the reference paths, with the real CARLA extents."""
    dims = [(2 * sv.vehicle.bounding_box.extent.x, 2 * sv.vehicle.bounding_box.extent.y) for sv in sims]
    gaps = {}
    for k in range(ticks):
        t = k / SIM_FPS
        boxes = []
        for sv, (L, W) in zip(sims, dims):
            x, y, _, _ = sv.reference(t)
            xa, ya, _, _ = sv.reference(t + 0.1)
            xb, yb, _, _ = sv.reference(max(0.0, t - 0.1))
            yaw = math.atan2(ya - yb, xa - xb) if math.hypot(xa - xb, ya - yb) > 1e-3 else None
            boxes.append(None if yaw is None else sc.box_corners((x, y), yaw, L, W))
        for i in range(len(sims)):
            for j in range(i + 1, len(sims)):
                if boxes[i] is None or boxes[j] is None:
                    continue
                key = tuple(sorted((f"{roles[i]}{i}", f"{roles[j]}{j}")))
                gaps[key] = min(gaps.get(key, 1e9), sc.box_gap(boxes[i], boxes[j]))
    return gaps, dims


def ref_kerb_frames(sv, L, W, drivable, ticks):
    grid, res, bounds = drivable
    n = 0
    for k in range(0, ticks, 4):  # ~5 Hz, like the data
        t = k / SIM_FPS
        x, y, _, _ = sv.reference(t)
        xa, ya, _, _ = sv.reference(t + 0.1)
        xb, yb, _, _ = sv.reference(max(0.0, t - 0.1))
        if math.hypot(xa - xb, ya - yb) < 1e-3:
            continue
        corners = sc.box_corners((x, y), math.atan2(ya - yb, xa - xb), L, W)
        n += sum(off_road(grid, res, bounds, *c) for c in corners) >= 2
    return n


def diagnose(world, lib, store, drivable, s):
    clear_world(world)
    start = s["start"]
    sims, roles, spawn_err = [], [], []
    for v in s["vehicles"]:
        sv = spawn_vehicle(world, lib, store, v)
        if sv is None:
            for o in sims:
                o.destroy()
            return {"id": s["id"], "diag": "spawn failed", "spawn_failed_role": v["role"]}
        sv.set_timed_route(*timed_route(v["frames"], v["xy"], start), z=DEFAULT_Z)
        xy = np.asarray(v["xy"])
        far = next((p for p in xy[1:] if np.hypot(*(p - xy[0])) > 1.0), None)
        if far is not None:
            path_yaw = math.degrees(math.atan2(far[1] - xy[0][1], far[0] - xy[0][0]))
            spawn_err.append(abs((sv.vehicle.get_transform().rotation.yaw - path_yaw + 180) % 360 - 180))
        else:
            spawn_err.append(0.0)
        sims.append(sv)
        roles.append(v["role"])
    for sv in sims:
        sv.hold_in_gear()
    for _ in range(SETTLE_TICKS):
        world.tick()
    for sv in sims:
        sv.set_initial_speed(float(sv.timed_route[2][0]))
    ticks = int(round(s["length_s"] * SIM_FPS))
    clock, drift = 0.0, [0.0] * len(sims)
    first = None
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(ticks):
            for sv in sims:
                sv.update(clock, None)
            world.tick()
            clock += 1.0 / SIM_FPS
            states = []
            for i, sv in enumerate(sims):
                loc = sv.vehicle.get_location()
                ct = cross_track(sv, loc.x, loc.y)
                drift[i] = max(drift[i], ct)
                _, _, _, ref_s = sv.reference(clock)
                states.append((ct, ref_s - sv._arc_length_at(loc.x, loc.y)))
            if first is None:
                for i, sv in enumerate(sims):
                    if sv.collision_events:
                        _, oid, otype = sv.collision_events[0]
                        j = next((k for k, o in enumerate(sims) if o.vehicle.id == oid), None)
                        first = {"t": round(clock, 2), "a": roles[i], "b": roles[j] if j is not None else otype,
                                 "a_drift": round(states[i][0], 2), "a_lag": round(states[i][1], 2),
                                 "b_drift": round(states[j][0], 2) if j is not None else None,
                                 "b_lag": round(states[j][1], 2) if j is not None else None,
                                 "a_bp": sv.vehicle.type_id.split(".")[-1],
                                 "b_bp": sims[j].vehicle.type_id.split(".")[-1] if j is not None else otype,
                                 "a_speed": round(sv.vehicle.get_velocity().length(), 1),
                                 "a_idx": i, "b_idx": j}
                        break
    gaps, dims = planned_gaps(sims, roles, ticks)
    row = {"id": s["id"], "behaviour": s["behaviour"], "status": s["status"], "length_s": s["length_s"],
           "diag": "collision" if first else "clean",
           "planned_min_gap_all": round(min(gaps.values()), 3) if gaps else None,
           "max_drift_ego_other": round(max(d for d, r in zip(drift, roles) if r != "neighbour"), 2),
           "max_drift_neighbours": round(max([d for d, r in zip(drift, roles) if r == "neighbour"] or [0]), 2),
           "max_spawn_yaw_err": round(max(spawn_err), 1),
           "ref_kerb_frames_max": max(ref_kerb_frames(sv, L, W, drivable, ticks) for sv, (L, W) in zip(sims, dims))}
    if first:
        a, b = first.pop("a_idx"), first.pop("b_idx")
        row.update({f"first_{k}": v for k, v in first.items()})
        if b is not None:
            key = tuple(sorted((f"{roles[a]}{a}", f"{roles[b]}{b}")))
            row["planned_gap_pair"] = round(gaps.get(key, float("nan")), 3)
            row["spawn_yaw_err_pair"] = round(max(spawn_err[a], spawn_err[b]), 1)
        else:
            row["ref_kerb_frames_a"] = ref_kerb_frames(sims[a], *dims[a], drivable, ticks)
            row["spawn_yaw_err_pair"] = round(spawn_err[a], 1)
    for sv in sims:
        sv.destroy()
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clean-sample", type=int, default=100)
    args = p.parse_args()
    rep = pd.read_csv(os.path.join(IL_DIR, "replay_pid.csv"), dtype={"id": str})
    ids = set(rep[rep.result == "collision"].id)
    clean = rep[rep.result == "collision-free"].id.sample(args.clean_sample, random_state=0)
    ids |= set(clean)
    scens = [s for s in json.load(open(os.path.join(IL_DIR, "scenarios_8s.json"))) if s["id"] in ids]
    print(f"diagnosing {len(scens)} scenarios", flush=True)
    store = sc.TrackStore(track_ids={v["track"] for s in scens for v in s["vehicles"]})
    drivable = load_map()
    client = carla.Client("localhost", 2000)
    client.set_timeout(60.0)
    world = client.get_world()
    if TOWN not in world.get_map().name:
        world = client.load_world(TOWN)
    orig = world.get_settings()
    st = world.get_settings()
    st.synchronous_mode, st.fixed_delta_seconds = True, 1.0 / SIM_FPS
    world.apply_settings(st)
    lib = world.get_blueprint_library()
    rows = []
    try:
        for i, s in enumerate(scens):
            rows.append(diagnose(world, lib, store, drivable, s))
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(scens)}", flush=True)
    finally:
        clear_world(world)
        world.apply_settings(orig)
    out = os.path.join(IL_DIR, "diagnose_pid.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
