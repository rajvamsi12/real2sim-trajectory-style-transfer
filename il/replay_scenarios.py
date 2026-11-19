#!/usr/bin/env python3
"""
il/replay_scenarios.py

Replays the IL scenarios (Scenariosssss/IL/scenarios_8s.json, from build_scenarios_8s.py) in CARLA with
ALL their vehicles (ego, other, up to 4 neighbours) and records collisions, so only collision-free
scenarios are used (ML_DECISIONS.md section 9, V3).

Same replay as debug/batch_process_scenarios.render_carla_simulation_video: CapturingVehicle with the
time-indexed PID route (scc.SimVehicle.set_timed_route), spawn yaw from choose_spawn_yaw, settle in gear,
launch at the real speed, synchronous 20 Hz. Mode "pid" (physics, no avoidance) by default; "expert" =
kinematic replay of the recorded positions.

A scenario is collision-free when every vehicle spawns and no collision sensor fires (vehicle or static).
Results go to Scenariosssss/IL/replay_<mode>.csv (one row per scenario, resumable).

  ~/CARLA_0.9.15/CarlaUE4.sh -nullrhi -nosound -carla-rpc-port=2000 &
  python il/replay_scenarios.py --limit 10          # quick test
  python il/replay_scenarios.py                     # all
"""

import argparse
import contextlib
import io
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

import carla  # noqa: E402
import scenario_checks as sc  # noqa: E402
from simulate_and_record_scenarios import CapturingVehicle, actor_box, SIM_FPS, TOWN, DEFAULT_Z  # noqa: E402
from debug.batch_process_scenarios import choose_spawn_yaw, travel_yaw_deg, SETTLE_TICKS  # noqa: E402
from debug.predict_collisions import load_map, off_road  # noqa: E402

IL_DIR = os.path.join(BASE_DIR, "Scenariosssss", "IL")
COLUMNS = ["id", "behaviour", "ego", "other", "start", "length_s", "vehicles", "spawned", "spawn_failed",
           "vehicle_collisions", "collision_pairs", "static_hits", "static_hit_roles", "min_box_gap_m",
           "min_progress", "min_progress_ego_other", "result"]  # fixed order: spawn-failed rows lack some keys
SPAWN_LIFT_M = 0.5          # spawn this far above the road surface, the car then drops and settles
PATH_YAW_MIN_M = 2.0        # yaw_mode "path": direction once the recorded path has moved this far
KERB_TICKS = 24             # kinematic vehicles: off the drivable area this long (1.2 s, as the builder's
                            # 5 frames) counts as a static hit, since their collision sensor can't fire


# ==========================================
# HELPERS
# ==========================================
def timed_route(frames, xy, start_frame):
    """(times_s, xy, speeds_mps) like batch_process_scenarios.build_timed_route (gaps allowed)."""
    times = (np.asarray(frames) - start_frame) / sc.REAL_WORLD_FPS
    xy = np.asarray(xy, dtype=float)
    speeds = []
    for i in range(len(xy)):
        lo, hi = max(0, i - 2), min(len(xy) - 1, i + 2)
        dt = times[hi] - times[lo]
        speeds.append(float(np.linalg.norm(xy[hi] - xy[lo]) / dt) if dt > 0 else 0.0)
    return times, xy, np.array(speeds)


def clear_world(world):
    world.tick()
    for a in list(world.get_actors().filter("vehicle.*")) + list(world.get_actors().filter("sensor.*")):
        if a.is_alive:
            try:
                a.destroy()
            except Exception:
                pass
    world.tick()


def spawn_vehicle(world, lib, store, v, yaw_mode="lane"):
    """CapturingVehicle at the vehicle's first scenario position, or None.
    yaw_mode "lane": choose_spawn_yaw (nearest lane direction, flipped if opposite); "path": the direction of
    the recorded path once it has moved PATH_YAW_MIN_M (14 % of lane yaws are > 30 deg off it), else "lane"."""
    f0, (x, y) = v["frames"][0], v["xy"][0]
    t = store.tracks[v["track"]]
    row = t.iloc[int(np.argmin(np.abs(t.index.to_numpy() - f0)))]  # augmented copies may start on another frame
    real_pt = (row.kf_cx_pixel, -row.kf_cy_pixel)
    path = [tuple(p) for p in v["xy"]]
    yaw = travel_yaw_deg(path, min_travel=PATH_YAW_MIN_M) if yaw_mode == "path" else None
    if yaw is None:
        yaw, _ = choose_spawn_yaw(world, (x, y), path, row.heading_angle, real_pt, row.linear_speed_meter_per_sec)
    yaw += v.get("spawn_yaw_offset_deg", 0.0)  # augmented "ego start offset (heading)" copies
    wp = world.get_map().get_waypoint(carla.Location(x=x, y=y, z=0.0), project_to_road=True)
    z = (wp.transform.location.z if wp else 0.0) + SPAWN_LIFT_M
    bp = lib.find(v["blueprint"])
    name = f"{v['role']}_{v['track']}"
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", name)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return CapturingVehicle(world, bp, name, (x, y, max(z, DEFAULT_Z)), yaw, IL_DIR, enable_avoidance=False)
    except RuntimeError:
        return None


VEL_POS_GAIN = 3.0          # drive "velocity": 1/s, position error -> velocity correction
VEL_YAW_GAIN = 3.0          # 1/s, heading error -> yaw-rate correction
VEL_MAX_EXTRA = 3.0         # m/s above the reference speed the correction may add


def _ref_yaw(sv, t):
    xa, ya, _, _ = sv.reference(t + 0.1)
    xb, yb, _, _ = sv.reference(max(0.0, t - 0.1))
    return math.degrees(math.atan2(ya - yb, xa - xb)) if math.hypot(xa - xb, ya - yb) > 0.05 else None


def velocity_step(sv, clock, dt):
    """Drive "velocity": physics stays on (gravity, contacts, collision sensor), but instead of throttle/steer
    the car gets, every tick, the velocity and yaw rate of its recorded path plus a correction towards it."""
    sv._sim_time = clock
    if clock > sv.timed_route[0][-1]:
        sv.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        return
    x, y, _, _ = sv.reference(clock)
    x1, y1, _, _ = sv.reference(clock + dt)
    loc, vel = sv.vehicle.get_location(), sv.vehicle.get_velocity()
    vx = (x1 - x) / dt + VEL_POS_GAIN * (x - loc.x)
    vy = (y1 - y) / dt + VEL_POS_GAIN * (y - loc.y)
    cap = math.hypot(x1 - x, y1 - y) / dt + VEL_MAX_EXTRA
    sp = math.hypot(vx, vy)
    if sp > cap:
        vx, vy = vx * cap / sp, vy * cap / sp
    sv.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False))
    sv.vehicle.set_target_velocity(carla.Vector3D(vx, vy, vel.z))
    y0, y1_ = _ref_yaw(sv, clock), _ref_yaw(sv, clock + dt)
    if y0 is not None and y1_ is not None:
        yaw = sv.vehicle.get_transform().rotation.yaw
        rate = ((y1_ - y0 + 180) % 360 - 180) / dt + VEL_YAW_GAIN * ((y0 - yaw + 180) % 360 - 180)
        sv.vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, rate))


def tune_pid(sv):
    """Drive "pid_tuned": longer steering look-ahead and softer lateral gains (less weaving of slow
    vehicles and two-wheelers)."""
    from agents.navigation.controller import VehiclePIDController
    sv.TIMED_MIN_LOOKAHEAD_M = 4.0
    sv.TIMED_LOOKAHEAD_S = 0.8
    sv.controller = VehiclePIDController(
        sv.vehicle,
        args_lateral={"K_P": 1.2, "K_I": 0.0, "K_D": 0.1, "dt": 1.0 / SIM_FPS},
        args_longitudinal={"K_P": 1.0, "K_I": 0.05, "K_D": 0.0, "dt": 1.0 / SIM_FPS},
        max_brake=0.5, max_throttle=1.0)


# ==========================================
# ONE SCENARIO
# ==========================================
def replay(world, lib, store, s, mode, yaw_mode="lane", kinematic_roles=(), drivable=None, drive="pid"):
    """mode "expert" = every vehicle kinematic; otherwise PID, except roles in kinematic_roles (e.g. the
    neighbours, as a non-reactive log replay). Kinematic vehicles have no physics, so a collision with them
    is a box overlap (< 0 m) and a static hit is KERB_TICKS off the drivable area."""
    clear_world(world)
    start = s["start"]
    sims, roles, failed = [], [], []
    for v in s["vehicles"]:
        sv = spawn_vehicle(world, lib, store, v, yaw_mode)
        if sv is None:
            failed.append(v["role"])
            continue
        sv.set_timed_route(*timed_route(v["frames"], v["xy"], start), z=DEFAULT_Z)
        sims.append(sv)
        roles.append(v["role"])
    row = {"id": s["id"], "behaviour": s["behaviour"], "ego": s["ego"], "other": s["other"], "start": start,
           "length_s": s.get("length_s"), "vehicles": len(s["vehicles"]), "spawned": len(sims),
           "spawn_failed": ";".join(failed)}
    if failed:
        for sv in sims:
            sv.destroy()
        row["result"] = "spawn failed (" + ", ".join(failed) + ")"
        return row

    kin = [mode == "expert" or r in kinematic_roles for r in roles]
    for sv, k in zip(sims, kin):
        if k:
            sv.vehicle.set_simulate_physics(False)
        else:
            sv.hold_in_gear()
    for _ in range(SETTLE_TICKS if not all(kin) else 1):
        world.tick()
    for sv, k in zip(sims, kin):
        if k:
            continue
        if drive == "pid_tuned":
            tune_pid(sv)
        if drive == "velocity":
            sv.vehicle.apply_control(carla.VehicleControl(hand_brake=False, manual_gear_shift=False))
        else:
            sv.set_initial_speed(float(sv.timed_route[2][0]))

    ticks = int(round(s["length_s"] * SIM_FPS)) if s.get("length_s") else int((s["end"] - start + 1) / sc.REAL_WORLD_FPS * SIM_FPS)
    dt, clock, min_gap = 1.0 / SIM_FPS, 0.0, float("inf")
    pairs, static = set(), set()
    off_ticks = [0] * len(sims)
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(ticks):
            for sv, k in zip(sims, kin):
                if k:
                    x, y, _, _ = sv.reference(clock)
                    xa, ya, _, _ = sv.reference(clock + 0.1)
                    xb, yb, _, _ = sv.reference(max(0.0, clock - 0.1))
                    yaw = math.degrees(math.atan2(ya - yb, xa - xb)) if math.hypot(xa - xb, ya - yb) > 1e-3 \
                        else sv.vehicle.get_transform().rotation.yaw
                    sv.vehicle.set_transform(carla.Transform(carla.Location(x=x, y=y, z=DEFAULT_Z), carla.Rotation(yaw=yaw)))
                    sv._sim_time = clock
                elif drive == "velocity":
                    velocity_step(sv, clock, dt)
                else:
                    sv.update(clock, None)
            world.tick()
            clock += dt
            boxes = [actor_box(sv.vehicle) for sv in sims]
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    g = sc.box_gap(boxes[i], boxes[j])
                    min_gap = min(min_gap, g)
                    if g < 0 and (kin[i] or kin[j]):
                        pairs.add(tuple(sorted((roles[i], roles[j]))))
            if drivable is not None:
                for i, (b, k) in enumerate(zip(boxes, kin)):
                    if k and sum(off_road(*drivable, *c) for c in b) >= 2:
                        off_ticks[i] += 1
                        if off_ticks[i] == KERB_TICKS:
                            static.add((roles[i], "off drivable area"))

    ids = {sv.vehicle.id: r for sv, r in zip(sims, roles)}
    for sv, r in zip(sims, roles):
        for _, other_id, other_type in sv.collision_events:
            if other_type.startswith("vehicle."):
                pairs.add(tuple(sorted((r, ids.get(other_id, "unknown")))))
            else:
                static.add((r, other_type))
    prog = {r: sv.progress() for sv, r in zip(sims, roles) if sv.route_length() >= 3.0}
    row.update({
        "vehicle_collisions": len(pairs), "collision_pairs": ";".join("-".join(p) for p in sorted(pairs)),
        "static_hits": len(static), "static_hit_roles": ";".join(sorted({r for r, _ in static})),
        "min_box_gap_m": round(min_gap, 3) if min_gap != float("inf") else None,
        "min_progress": round(min(prog.values()), 3) if prog else 1.0,
        "min_progress_ego_other": round(min([p for r, p in prog.items() if r in ("ego", "other")] or [1.0]), 3),
    })
    row["result"] = "collision-free" if not pairs and not static else "collision"
    for sv in sims:
        sv.destroy()
    return row


# ==========================================
# MAIN
# ==========================================
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["pid", "expert"], default="pid")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--scenarios", default=os.path.join(IL_DIR, "scenarios_8s.json"))
    p.add_argument("--ids", help="CSV with an 'id' column: replay only these scenarios.")
    p.add_argument("--yaw", choices=["lane", "path"], default="lane", help="Spawn yaw (see spawn_vehicle).")
    p.add_argument("--kinematic", choices=["none", "neighbours", "all"], default="none",
                   help="Replay these vehicles kinematically on their recorded path ('all' = --mode expert).")
    p.add_argument("--drive", choices=["pid", "pid_tuned", "velocity"], default="pid",
                   help="Physics vehicles: PID throttle/steer, retuned PID, or velocity tracking (see velocity_step).")
    p.add_argument("--tag", help="Output name replay_<tag>.csv (default from the options).")
    p.add_argument("--port", type=int, default=2000, help="CARLA server port (2010 lets a second check run beside a render).")
    args = p.parse_args()

    scens = json.load(open(args.scenarios))
    if args.ids:
        keep = set(pd.read_csv(args.ids, dtype={"id": str}).id)
        scens = [s for s in scens if s["id"] in keep]
    if args.limit:
        rng = np.random.default_rng(1)
        scens = [scens[i] for i in sorted(rng.choice(len(scens), size=min(args.limit, len(scens)), replace=False))]
    kin_roles = {"none": (), "neighbours": ("neighbour",)}.get(args.kinematic, ())
    mode = "expert" if args.kinematic == "all" else args.mode
    tag = args.tag or (f"{mode}" + (f"_yaw-{args.yaw}" if args.yaw != "lane" else "")
                       + (f"_kin-{args.kinematic}" if args.kinematic not in ("none", "all") else ""))
    out = os.path.join(IL_DIR, f"replay_{tag}" + (f"_sample{args.limit}" if args.limit else "") + ".csv")
    done = set(pd.read_csv(out, dtype={"id": str}).id) if os.path.exists(out) else set()
    todo = [s for s in scens if s["id"] not in done]
    print(f"{len(scens):,} scenarios, {len(done):,} already done, {len(todo):,} to replay ({tag})", flush=True)
    drivable = load_map()

    store = sc.TrackStore(track_ids={v["track"] for s in todo for v in s["vehicles"]}, extra_columns=["heading_angle"]
                          if "heading_angle" not in sc.CSV_COLUMNS else ())
    client = carla.Client("localhost", args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    if TOWN not in world.get_map().name:
        world = client.load_world(TOWN)
    orig = world.get_settings()
    st = world.get_settings()
    st.synchronous_mode, st.fixed_delta_seconds = True, 1.0 / SIM_FPS
    world.apply_settings(st)
    lib = world.get_blueprint_library()

    t0 = time.time()
    try:
        for i, s in enumerate(todo):
            row = replay(world, lib, store, s, mode, args.yaw, kin_roles, drivable, args.drive)
            pd.DataFrame([row]).reindex(columns=COLUMNS).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            if (i + 1) % 25 == 0 or i + 1 == len(todo):
                el = time.time() - t0
                print(f"  {i + 1}/{len(todo)}  {el / (i + 1):.1f} s/scenario  eta {el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min", flush=True)
    finally:
        clear_world(world)
        world.apply_settings(orig)

    df = pd.read_csv(out)
    order = ["Cut In", "Cut Out", "U-turn", "Overtaking Left", "Overtaking Right", "Vehicle Following"]
    t = pd.crosstab(df.behaviour, df.result.str.split(" \\(").str[0]).reindex(order).fillna(0).astype(int)
    t.loc["Total"] = t.sum()
    pd.set_option("display.width", 200)
    print("\n" + t.to_string())
    col = df[df.result == "collision"]
    if len(col):
        involved = (col.collision_pairs.fillna("").astype(str).str.contains("ego|other")
                    | col.static_hit_roles.fillna("").astype(str).str.contains("ego|other"))
        print(f"\nscenarios with a collision involving ego/other: {int(involved.sum())} of {len(col)}")
    fails = df.spawn_failed.dropna().astype(str)
    print(f"spawn failures by role: {fails[fails != ''].str.split(';').explode().value_counts().to_dict()}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
