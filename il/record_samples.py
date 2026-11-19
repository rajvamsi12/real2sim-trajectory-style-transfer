#!/usr/bin/env python3
"""
il/record_samples.py

Steps 6/7 (ML_DECISIONS.md sections 0, 5, 6): records imitation-learning samples by replaying scenarios in
CARLA with physics (cure E: path spawn heading + velocity tracking). Every spawned vehicle is an ego (T2):
each gets its own front camera and nuScenes-like 32-channel LiDAR.

Sensors run on a 0.25 s grid (the planning rate). Per vehicle and grid time it saves
  cam_<k>.jpg     front camera, 256 x 256
  lidar_<k>.npz   one LiDAR sweep in the right-handed sensor frame (x fwd, y left, z up), int16 centimetres;
                  the PointPillars input is (x, y, z, 0) (the 4th value is the multi-sweep time lag)
and one row per SAMPLE in samples.csv:
  own state (speed m/s, longitudinal accel, yaw rate rad/s, bicycle-model steer equivalent), route command, behaviour info, and the
  label: the next 10 waypoints (0.25 s apart, 2.5 s) of the vehicle's path in its own frame at that moment
  (x forward, y left). For ego-offset copies the path is the recovery path; for the others the recorded one.
Samples (T4): every 0.5 s, every 0.25 s while the behaviour window is running; only where the history
(cam/lidar at t-0.25, t-0.5) exists and the 2.5 s label lies inside the vehicle's path.

  ~/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2010 &
  python il/record_samples.py --port 2010 --limit 5 --out Scenariosssss/IL/samples_test
"""

import argparse
import contextlib
import io
import json
import math
import os
import queue
import shutil
import sys
import time

import cv2
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, SCRIPT_DIR)

import carla  # noqa: E402
import scenario_checks as sc  # noqa: E402
from simulate_and_record_scenarios import SIM_FPS, TOWN, DEFAULT_Z  # noqa: E402
from debug.batch_process_scenarios import SETTLE_TICKS  # noqa: E402
from replay_scenarios import IL_DIR, timed_route, clear_world, spawn_vehicle, velocity_step  # noqa: E402

GRID_S = 0.25                 # sensor / planning grid
GRID_TICKS = int(round(GRID_S * SIM_FPS))
PLAIN_EVERY = 2               # plain driving: a sample every 2nd grid step (0.5 s)
HISTORY = 2                   # cam/lidar at t-0.25 and t-0.5 must exist
SPEED_HISTORY = (1, 2, 3, 4)  # captures back: own speed at t-0.25 ... t-1.0 s (columns speed_m025 ... speed_m100)
N_WP = 10                     # label: 10 waypoints, 0.25 s apart
CAM_SIZE, CAM_FOV = 256, 100
LIDAR_Z = 1.84
JPEG_QUALITY = 90
WHEELBASE_FRAC = 0.6          # wheelbase ~ 60 % of the body length
MIN_STEER_SPEED = 0.5         # m/s; below this the steer equivalent is set to 0
MAX_SCENARIO_ERRORS = 2       # a scenario that crashed the recorder this often is skipped
MAX_ERROR_STREAK = 3          # this many errors in a row: exit so the runner restarts CARLA


def lidar_bp(lib):
    bp = lib.find("sensor.lidar.ray_cast")
    for k, v in {"channels": "32", "range": "70", "rotation_frequency": str(SIM_FPS), "points_per_second": "700000",
                 "upper_fov": "10", "lower_fov": "-30", "dropoff_general_rate": "0.0",
                 "dropoff_intensity_limit": "1.0", "dropoff_zero_intensity": "0.0", "sensor_tick": "0.0"}.items():
        bp.set_attribute(k, v)
    return bp


def camera_bp(lib):
    bp = lib.find("sensor.camera.rgb")
    # Renders only every 0.25 s (measured: ~1/3 less time per tick than rendering every tick). The capture ticks
    # define the grid; LiDAR and state are taken from the same tick. (The LiDAR keeps sensor_tick 0: with 0.25 it
    # returns only half a sweep.)
    for k, v in {"image_size_x": str(CAM_SIZE), "image_size_y": str(CAM_SIZE), "fov": str(CAM_FOV),
                 "sensor_tick": str(GRID_S)}.items():
        bp.set_attribute(k, v)
    return bp


def to_ego(points_xy, ego_xy, ego_yaw_deg):
    """World (CARLA, left-handed) -> ego frame, right-handed: x forward, y left."""
    c, s = math.cos(math.radians(ego_yaw_deg)), math.sin(math.radians(ego_yaw_deg))
    d = np.asarray(points_xy) - np.asarray(ego_xy)
    fwd = d[:, 0] * c + d[:, 1] * s
    right = -d[:, 0] * s + d[:, 1] * c
    return np.stack([fwd, -right], 1)


def record(world, lib, store, s, out_dir, egos=None):
    """egos: roles that get sensors and samples (None = every vehicle); all vehicles are driven either way."""
    clear_world(world)
    sims, vs = [], []
    for v in s["vehicles"]:
        sv = spawn_vehicle(world, lib, store, v, "path")
        if sv is None:
            for o in sims:
                o.destroy()
            return None, "spawn failed"
        sv.set_timed_route(*timed_route(v["frames"], v["xy"], s["start"]), z=DEFAULT_Z)
        sims.append(sv)
        vs.append(v)
    for sv in sims:
        sv.hold_in_gear()
    for _ in range(SETTLE_TICKS):
        world.tick()
    for sv in sims:
        sv.vehicle.apply_control(carla.VehicleControl(hand_brake=False, manual_gear_shift=False))

    rec = [j for j, v in enumerate(vs) if egos is None or v["role"] in egos]
    sensors, cams, lidars = [], [], []
    for sv in (sims[j] for j in rec):
        ext = sv.vehicle.bounding_box.extent
        cq, lq = queue.Queue(), queue.Queue()
        cam = world.spawn_actor(camera_bp(lib), carla.Transform(carla.Location(x=0.6 * ext.x, z=2 * ext.z + 0.2)),
                                attach_to=sv.vehicle)
        lid = world.spawn_actor(lidar_bp(lib), carla.Transform(carla.Location(z=LIDAR_Z)), attach_to=sv.vehicle)
        cam.listen(cq.put)
        lid.listen(lq.put)
        sensors += [cam, lid]
        cams.append(cq)
        lidars.append(lq)

    n_ticks = int(round(s["length_s"] * SIM_FPS))
    beh_t0 = (s.get("src_start", s["start"]) - s["start"]) / sc.REAL_WORLD_FPS
    beh_t1 = (s.get("src_end", s["end"]) - s["start"]) / sc.REAL_WORLD_FPS
    dirs = []
    for v in (vs[j] for j in rec):
        d = os.path.join(out_dir, s["id"], f"veh_{v['track']}")
        os.makedirs(d, exist_ok=True)
        dirs.append(d)
    state = [{} for _ in rec]    # capture index k -> (x, y, yaw, speed, accel, yaw_rate, steer, t)
    rows, clock, dt, prev_speed = [], 0.0, 1.0 / SIM_FPS, [0.0] * len(rec)
    recent = [{} for _ in rec]  # frame -> (lidar points, kinematic state, sim time), last few ticks only

    def kinematics(sv):
        tf, vel, av = sv.vehicle.get_transform(), sv.vehicle.get_velocity(), sv.vehicle.get_angular_velocity()
        fwd = tf.get_forward_vector()
        speed = vel.x * fwd.x + vel.y * fwd.y
        # Velocity tracking doesn't turn the wheels, so record the bicycle-model steer equivalent of the motion
        # (same formula at test time with the PID): delta = atan(wheelbase * yaw_rate / speed), + = left
        yaw_rate = -math.radians(av.z)
        wheelbase = WHEELBASE_FRAC * 2 * sv.vehicle.bounding_box.extent.x
        steer = math.degrees(math.atan(wheelbase * yaw_rate / speed)) if abs(speed) > MIN_STEER_SPEED else 0.0
        return tf.location.x, tf.location.y, tf.rotation.yaw, speed, yaw_rate, steer

    def save_captures(i, wait):
        """Store every camera image that has arrived, with the LiDAR sweep and state of its own tick."""
        while True:
            try:
                cm = cams[i].get(timeout=wait) if wait else cams[i].get_nowait()
            except queue.Empty:
                return
            if cm.frame not in recent[i]:
                continue  # image from before the scenario clock started
            pts, kin, t = recent[i][cm.frame]
            k = len(state[i])
            img = np.frombuffer(cm.raw_data, np.uint8).reshape(cm.height, cm.width, 4)[:, :, :3]
            cv2.imwrite(os.path.join(dirs[i], f"cam_{k:03d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            np.savez_compressed(os.path.join(dirs[i], f"lidar_{k:03d}.npz"),
                                xyz_cm=np.clip(np.round(pts * 100), -32768, 32767).astype(np.int16))
            accel = (kin[3] - prev_speed[i]) / GRID_S if k else 0.0
            prev_speed[i] = kin[3]
            state[i][k] = kin[:4] + (accel,) + kin[4:] + (t,)

    with contextlib.redirect_stdout(io.StringIO()):
        for tick in range(1, n_ticks + 1):
            for sv in sims:
                velocity_step(sv, clock, dt)
            frame = world.tick()
            clock += dt
            for i, sv in enumerate(sims[j] for j in rec):
                lm = lidars[i].get(timeout=5.0)
                while lm.frame < frame:
                    lm = lidars[i].get(timeout=5.0)
                pts = np.frombuffer(lm.raw_data, np.float32).reshape(-1, 4)[:, :3].copy()
                pts[:, 1] *= -1
                recent[i][frame] = (pts, kinematics(sv), round(clock, 4))
                for f in [f for f in recent[i] if f < frame - 2 * GRID_TICKS]:
                    del recent[i][f]
                save_captures(i, wait=0)
        for i in range(len(rec)):
            save_captures(i, wait=0.5)  # images of the last tick may still be in flight

    for i, (sv, v) in enumerate((sims[j], vs[j]) for j in rec):
        times = sv.timed_route[0]
        for k, st in sorted(state[i].items()):
            t = st[7]
            if k - HISTORY not in state[i] or t + N_WP * GRID_S > times[-1] + 1e-6 or t < times[0]:
                continue
            in_beh = beh_t0 <= t <= beh_t1
            if not in_beh and k % PLAIN_EVERY:
                continue
            fut = np.array([sv.reference(t + j * GRID_S)[:2] for j in range(1, N_WP + 1)])
            wp = to_ego(fut, st[:2], st[2])
            f_real = s["start"] + t * sc.REAL_WORLD_FPS
            ci = int(np.argmin(np.abs(np.asarray(v["frames"]) - f_real)))
            row = {"scenario": s["id"], "base_id": s.get("base_id", s["id"]), "aug": (s.get("aug") or {}).get("type", "none"),
                   "behaviour": s["behaviour"], "role": v["role"], "track": v["track"], "blueprint": v["blueprint"],
                   "k": k, "t": t, "in_behaviour_window": in_beh, "route_cmd": v["route_cmd"][ci],
                   "speed": round(st[3], 3), "accel": round(st[4], 3), "yaw_rate_rps": round(st[5], 4), "steer_equiv_deg": round(st[6], 2),
                   # own speed 0.25 / 0.5 / 0.75 / 1.0 s ago (earliest capture if the scenario is younger); added 2026-11-10
                   **{f"speed_m{int(h * 25):03d}": round(state[i][max(k - h, min(state[i]))][3], 3) for h in SPEED_HISTORY},
                   "cam": [f"{s['id']}/veh_{v['track']}/cam_{k - h:03d}.jpg" for h in range(HISTORY, -1, -1)],
                   "lidar": [f"{s['id']}/veh_{v['track']}/lidar_{k - h:03d}.npz" for h in range(HISTORY, -1, -1)]}
            row.update({f"wp{j}_x": round(float(wp[j, 0]), 3) for j in range(N_WP)})
            row.update({f"wp{j}_y": round(float(wp[j, 1]), 3) for j in range(N_WP)})
            rows.append(row)
    collided = any(sv.collision_events for sv in sims)
    for x in sensors:
        x.stop()
        x.destroy()
    for sv in sims:
        sv.destroy()
    world.tick()
    return rows, ("collision during recording" if collided else "ok")


def wait_for_disk(args):
    """Pause (don't fail) while the output drive is unmounted or nearly full; re-check every minute."""
    warned = False
    while True:
        if args.require_mount and not os.path.ismount(args.require_mount):
            why = f"{args.require_mount} is not mounted"
        elif shutil.disk_usage(args.out).free / 1e9 < args.min_free_gb:
            why = f"less than {args.min_free_gb:.0f} GB free on {args.out}"
        else:
            if warned:
                print(f"  resumed {time.strftime('%H:%M:%S')}", flush=True)
            return
        if not warned:
            print(f"  PAUSED {time.strftime('%H:%M:%S')}: {why}", flush=True)
            warned = True
        time.sleep(60)


def dir_bytes(path):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=2010)
    p.add_argument("--scenarios", default=os.path.join(IL_DIR, "scenarios_8s.json"))
    p.add_argument("--ok-csv", default=os.path.join(IL_DIR, "replay_cureE.csv"),
                   help="Only scenarios marked collision-free here.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ids", nargs="*", help="Record only these scenario ids.")
    p.add_argument("--out", default=os.path.join(IL_DIR, "samples_test"))
    p.add_argument("--full", action="store_true",
                   help="Step 7: all collision-free base scenarios and copies (cure E), ego/other progress >= 85 %; egos per "
                        "ML_DECISIONS.md T4 (middle option): base + scene noise all vehicles, ego offset the ego only.")
    p.add_argument("--min-free-gb", type=float, default=20.0, help="Pause while less than this is free on --out.")
    p.add_argument("--require-mount", help="Pause while this path is not a mount point (e.g. /mnt/data).")
    p.add_argument("--per-type", type=int, help="With --full: only this many scenarios per type (test).")
    args = p.parse_args()

    def collision_free(csv):
        r = pd.read_csv(csv, dtype={"id": str}).drop_duplicates("id", keep="last")
        return set(r[(r.result == "collision-free") & (r.min_progress_ego_other >= 0.85)].id)

    if args.full:
        ok_base = collision_free(os.path.join(IL_DIR, "replay_cureE.csv"))
        ok_aug = collision_free(os.path.join(IL_DIR, "replay_aug_cureE.csv"))
        base = [s for s in json.load(open(os.path.join(IL_DIR, "scenarios_8s.json"))) if s["id"] in ok_base]
        aug = [s for s in json.load(open(os.path.join(IL_DIR, "scenarios_aug.json"))) if s["id"] in ok_aug]
        noise = [s for s in aug if s["aug"]["type"] == "scene_noise"]
        offset = [s for s in aug if s["aug"]["type"] in ("ego_offset", "ego_speed")]  # ego only
        if args.per_type:
            base, noise, offset = base[:args.per_type], noise[:args.per_type], offset[:args.per_type]
        jobs = [(s, None) for s in base + noise] + [(s, ("ego",)) for s in offset]
        print(f"full: {len(base)} base + {len(noise)} scene-noise (all vehicles), {len(offset)} ego-offset/speed (ego only)",
              flush=True)
    else:
        ok = set(pd.read_csv(args.ok_csv, dtype={"id": str}).query("result == 'collision-free'").id)
        scens = [s for s in json.load(open(args.scenarios)) if s["id"] in ok and (not args.ids or s["id"] in args.ids)]
        if args.limit:
            rng = np.random.default_rng(3)
            scens = [scens[i] for i in sorted(rng.choice(len(scens), min(args.limit, len(scens)), replace=False))]
        jobs = [(s, None) for s in scens]
    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, "samples.csv")
    timing_csv = os.path.join(args.out, "timing.csv")
    errors_csv = os.path.join(args.out, "errors.csv")
    done = set(pd.read_csv(timing_csv, dtype={"scenario": str}).scenario) if os.path.exists(timing_csv) else set()
    if os.path.exists(errors_csv):  # a scenario that crashed the recorder twice is skipped from now on
        e = pd.read_csv(errors_csv, dtype={"scenario": str}).scenario.value_counts()
        done |= set(e[e >= MAX_SCENARIO_ERRORS].index)
    jobs = [(s, e) for s, e in jobs if s["id"] not in done]
    scens = [s for s, _ in jobs]
    print(f"{len(done)} already done, {len(jobs)} to record", flush=True)
    store = sc.TrackStore(track_ids={v["track"] for s in scens for v in s["vehicles"]})
    client = carla.Client("localhost", args.port)
    client.set_timeout(120.0)
    world = client.get_world()
    if TOWN not in world.get_map().name:
        world = client.load_world(TOWN)
    orig = world.get_settings()
    st = world.get_settings()
    st.synchronous_mode, st.fixed_delta_seconds = True, 1.0 / SIM_FPS
    world.apply_settings(st)
    lib = world.get_blueprint_library()
    log, t0 = [], time.time()
    try:
        streak = 0
        for i, (s, egos) in enumerate(jobs):
            wait_for_disk(args)
            ts = time.time()
            try:
                rows, note = record(world, lib, store, s, args.out, egos)
                streak = 0
            except Exception as ex:  # log it and move on; several in a row means CARLA is gone -> exit, runner restarts
                streak += 1
                pd.DataFrame([{"scenario": s["id"], "time": time.strftime("%Y-%m-%d %H:%M:%S"), "error": repr(ex)[:300]}]) \
                    .to_csv(errors_csv, mode="a", header=not os.path.exists(errors_csv), index=False)
                print(f"  [{i + 1}/{len(jobs)}] {s['id']}: ERROR {ex!r}", flush=True)
                if streak >= MAX_ERROR_STREAK:
                    raise SystemExit(1)
                try:
                    clear_world(world)
                except Exception:
                    raise SystemExit(1)
                continue
            secs = time.time() - ts
            mb = dir_bytes(os.path.join(args.out, s["id"])) / 1e6 if rows is not None else 0.0
            n = len(rows) if rows else 0
            if rows:
                pd.DataFrame(rows).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
            log.append({"scenario": s["id"], "behaviour": s["behaviour"], "length_s": s["length_s"], "vehicles": len(s["vehicles"]), "recorded": "all" if egos is None else "+".join(egos),
                        "aug": (s.get("aug") or {}).get("type", "none"),
                        "samples": n, "seconds": round(secs, 1), "MB": round(mb, 1), "note": note})
            pd.DataFrame([log[-1]]).to_csv(os.path.join(args.out, "timing.csv"), mode="a",
                                           header=not os.path.exists(os.path.join(args.out, "timing.csv")), index=False)
            print(f"  [{i + 1}/{len(scens)}] {s['id']} {s['behaviour']}: {n} samples, {secs:.0f} s, {mb:.1f} MB, {note}", flush=True)
    finally:
        clear_world(world)
        world.apply_settings(orig)
    d = pd.DataFrame(log)
    if len(d):
        print(f"\n{d.samples.sum()} samples from {len(d)} scenarios: {d.seconds.sum() / max(d.samples.sum(), 1):.2f} s and "
              f"{d.MB.sum() / max(d.samples.sum(), 1):.3f} MB per sample (sensor files shared by overlapping samples)")


if __name__ == "__main__":
    main()
