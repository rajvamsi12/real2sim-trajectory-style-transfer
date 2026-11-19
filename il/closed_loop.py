#!/usr/bin/env python3
"""
il/closed_loop.py

Step 11 (ML_DECISIONS.md E1, M10): closed-loop test of the IL driver in CARLA on the held-out base scenarios (time-split
test blocks, collision-free in replay_cureE.csv). The ego is driven by a "driver"; every other vehicle replays its
recorded path with cure E (velocity tracking, physics on, non-reactive log replay). All vehicles have physics.
Drivers:
  expert    PID on the ego's recorded path (how well the controller alone does; = replay mode "pid" for the ego)
  constvel  PID on a straight constant-speed path from the current state, replanned every 0.25 s (baseline)
  <driver>+ttc  the same driver plus the TTC safety layer of simulate_and_record_scenarios.py (2026-11-13): a predicted box
            conflict (< 0.4 m, constant velocity, 3 s) with a vehicle AHEAD -> sooner than 1.5 s: full brake; later: speed
            capped at the threat's speed along our heading and a 0.5 m nudge away from it if that stays on a driving lane.
            Rows report how long it braked / followed (shield_brake_s, shield_follow_s).
  <run>     a trained model (Scenariosssss/IL/runs/<run>/model.pt): every 0.25 s the ego's camera + LiDAR go through the
            frozen encoders exactly as in il/cache_features.py (JPEG q90 and cm rounding as recorded); the model's
            top-scored trajectory (10 x 0.25 s) becomes the PID's timed route (scc.SimVehicle, same PID as the replay)
Until 3 frames of sensor history exist (0.5 s) every driver follows the recorded path.
Ego blueprint: the scenario's own (--types own) or each of the 5 types (--types all); vehicle-type input = that blueprint.
One CSV row per (scenario, ego blueprint, driver): collision with a vehicle / static object, off the drivable area,
progress along the recorded path, distance to the recorded position at the same time (mean / max / final), stuck time,
min gap to other vehicles, accel / jerk. Resumable.

  ~/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic -carla-rpc-port=2000 &
  python il/closed_loop.py --drivers expert constvel s_K1 reg_C_droplr --limit 5 --tag smoke
"""

import argparse
import contextlib
import io
import json
import math
import os
import queue
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, SCRIPT_DIR)

import carla  # noqa: E402
import scenario_checks as sc  # noqa: E402
import simulate_and_record_scenarios as SR  # noqa: E402
import scc  # noqa: E402
from simulate_and_record_scenarios import (SIM_FPS, TOWN, DEFAULT_Z, actor_box, find_primary_threat, nudge_offset,  # noqa: E402
                                           AVOIDANCE_IMMINENT_TTC_S)
from debug.batch_process_scenarios import SETTLE_TICKS, get_camera_intrinsic_matrix, world_to_pixel  # noqa: E402
from debug.predict_collisions import load_map, off_road  # noqa: E402
from replay_scenarios import IL_DIR, KERB_TICKS, timed_route, clear_world, spawn_vehicle, velocity_step  # noqa: E402
from record_samples import (lidar_bp, camera_bp, GRID_S, GRID_TICKS, JPEG_QUALITY, WHEELBASE_FRAC, MIN_STEER_SPEED,  # noqa: E402
                            LIDAR_Z)
from cache_features import Encoders, IMG_MEAN, IMG_STD, POOL  # noqa: E402
from model import N_WP, HISTORY, ROUTE_CMDS, VEHICLE_TYPES, STATE_SCALE, SPEED_HISTORY, make_state  # noqa: E402
import train as T  # noqa: E402

RUNS = os.path.join(IL_DIR, "runs")
STUCK_SPEED, EXPERT_MOVING = 0.5, 2.0   # m/s: ego below this while the recorded vehicle is above that = stuck
MAX_ERROR_STREAK = 3
SMOOTH_SPEED = os.environ.get("CL_INSTANT_SPEED") != "1"   # see episode()
TRACE = os.environ.get("CL_TRACE") == "1"
TICK_TRACE = os.environ.get("CL_TICK") == "1"
VIDEO_DIR = os.environ.get("CL_VIDEO")   # write a top-down review video per episode into this folder (4 Nov)
VIDEO_W, VIDEO_H, VIDEO_FOV = 960, 720, 90.0
VIDEO_COLORS = {"ego": (60, 220, 60), "other": (0, 150, 255), "neighbour": (255, 220, 0)}  # BGR
PROBE = os.environ.get("CL_PROBE") == "1"   # expert drives, model only predicts (diagnosis 30 Oct)
ACCEL_MODE = os.environ.get("CL_ACCEL", "raw")   # model accel input: raw (0.25 s) | smooth (1 s) | zero; diagnosis 30 Oct


class OnlineEncoders(Encoders):
    """The step-8 encoders on in-memory sensor data, with the same JPEG / int16-cm loss as the recorded files."""

    @torch.no_grad()
    def camera_img(self, bgr):
        _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1] / 255.0
        x = torch.tensor(((im - IMG_MEAN) / IMG_STD).transpose(2, 0, 1)[None], dtype=torch.float32, device=self.device)
        return self.seg(pixel_values=x).last_hidden_state.half().float()[0]           # (256, 8, 8)

    @torch.no_grad()
    def lidar_pts(self, pts):
        xyz = np.clip(np.round(pts * 100), -32768, 32767).astype(np.int16).astype(np.float32) / 100.0
        p = torch.tensor(np.c_[xyz, np.zeros(len(xyz))], dtype=torch.float32, device=self.device)
        f = F.avg_pool2d(self.pp.bev(p)[0][:, self.rows, self.cols][None], POOL)[0]  # (384, 16, 20)
        c, h, w = f.shape
        v = f.reshape(c, -1).T - self.pca[0]
        return (v @ self.pca[1].T).T.reshape(-1, h, w).transpose(1, 2).half().float()  # (48, 20, 16)


def test_scenarios(any_split=False, split_name="test"):
    """Base scenarios in the time-split test blocks (val blocks with split_name 'val', every split with any_split) that
    replay collision-free with cure E."""
    base = json.load(open(os.path.join(IL_DIR, "scenarios_8s.json")))
    r = pd.read_csv(os.path.join(IL_DIR, "replay_cureE.csv"), dtype={"id": str}).drop_duplicates("id", keep="last")
    ok = set(r[(r.result == "collision-free") & (r.min_progress_ego_other >= 0.85)].id)
    names = {"T": "train", "V": "val", "X": "test"}
    split = lambda s: names[T.SPLIT_PATTERN[int((s["start"] + s["end"]) / 2 / T.REAL_FPS // T.SPLIT_BLOCK_S) % len(T.SPLIT_PATTERN)]]
    return [dict(s, split=split(s)) for s in base if s["id"] in ok and (any_split or split(s) == split_name)]


def arc_length_at(xy, s_cum, x, y):
    a, b = xy[:-1], xy[1:]
    ab = b - a
    u = np.clip(((np.array([x, y]) - a) * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-9), 0.0, 1.0)
    proj = a + ab * u[:, None]
    i = int(np.argmin(np.hypot(proj[:, 0] - x, proj[:, 1] - y)))
    return float(s_cum[i] + u[i] * (s_cum[i + 1] - s_cum[i]))


def kinematics(v):
    """Same as record_samples.record.kinematics: x, y, yaw deg, speed, yaw rate (rad/s, + left), steer equivalent deg."""
    tf, vel, av = v.get_transform(), v.get_velocity(), v.get_angular_velocity()
    fwd = tf.get_forward_vector()
    speed = vel.x * fwd.x + vel.y * fwd.y
    yaw_rate = -math.radians(av.z)
    wheelbase = WHEELBASE_FRAC * 2 * v.bounding_box.extent.x
    steer = math.degrees(math.atan(wheelbase * yaw_rate / speed)) if abs(speed) > MIN_STEER_SPEED else 0.0
    return tf.location.x, tf.location.y, tf.rotation.yaw, speed, yaw_rate, steer


def from_ego(wp, x, y, yaw_deg):
    """Ego frame (x forward, y left) -> CARLA world (left-handed); inverse of record_samples.to_ego."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    fwd, right = wp[:, 0], -wp[:, 1]
    return np.stack([x + fwd * c - right * s, y + fwd * s + right * c], 1)


SPEED_KI = float(os.environ.get("CL_SPEED_KI", "2.0"))   # 1/s; 0 = plain scc.SimVehicle PID
SPEED_I_MAX = 3.0                                         # m/s, anti-windup clamp


def ego_update(sv, clock, speed_i, dt, shield=None):
    """scc.SimVehicle._timed_update plus an integral on the speed error that is kept across replans.
    Without it the PID settles ~0.1 m/s below its target; every 0.25 s plan starts from the measured speed, so the ego
    lost ~0.1 m/s per replan (measured 29 Oct: constvel 4.7 -> 2.8 m/s in 4 s on a Micra). Returns the new integral.
    shield: dict(world=..., brake=0, follow=0[, brake_s, gap_m]) -> TTC safety layer (module docstring), counts its ticks;
    brake_s / gap_m override AVOIDANCE_IMMINENT_TTC_S (1.5 s) / AVOIDANCE_MIN_GAP_M (0.4 m) for tuning."""
    if SPEED_KI == 0.0:
        sv.update(clock, None)
        return 0.0
    if sv._constant_velocity_until is not None and clock >= sv._constant_velocity_until:
        sv.release_launch_hold()
    if clock > sv.timed_route[0][-1]:
        sv.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        return speed_i
    cap, offset = None, 0.0
    if shield is not None:
        SR.AVOIDANCE_MIN_GAP_M = shield.get("gap_m", 0.4)   # read by predict_box_conflict at call time
        ttc, threat = find_primary_threat(shield["world"], sv.vehicle)
        if threat is not None and ttc < shield.get("brake_s", AVOIDANCE_IMMINENT_TTC_S):
            shield["brake"] += 1
            sv.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            return 0.0
        if threat is not None:
            shield["follow"] += 1
            fwd = sv.vehicle.get_transform().get_forward_vector()
            tv = threat.get_velocity()
            cap = max(0.0, tv.x * fwd.x + tv.y * fwd.y)
            offset = nudge_offset(shield["world"], sv.vehicle, threat)
    _, _, ref_speed, ref_s = sv.reference(clock)
    _, _, _, ahead_s = sv.reference(clock + sv.TIMED_LOOKAHEAD_S)
    loc = sv.vehicle.get_location()
    own_s = sv._arc_length_at(loc.x, loc.y)
    tx, ty = sv._point_at_arc_length(max(ahead_s, own_s + sv.TIMED_MIN_LOOKAHEAD_M))
    if offset:
        fwd = sv.vehicle.get_transform().get_forward_vector()
        tx, ty = tx - fwd.y * offset, ty + fwd.x * offset          # + = left
    target_wp = scc.DummyWaypoint(carla.Transform(carla.Location(x=tx, y=ty, z=sv._route_z)))
    want = max(0.0, ref_speed + sv.TIMED_SPEED_GAIN * (ref_s - own_s))
    if cap is not None:
        want = min(want, cap)
    v = kinematics(sv.vehicle)[3]
    speed_i = float(np.clip(speed_i + SPEED_KI * (want - v) * dt, -SPEED_I_MAX, SPEED_I_MAX)) if want > 0.1 else 0.0
    sv.vehicle.apply_control(sv.controller.run_step(max(0.0, want + speed_i) * 3.6, target_wp))
    return speed_i


def traj_image(base, trails, roles, ei, real_px, driver, hits, out):
    """Whole-episode paths on the first top-down frame: ego coloured by time (blue = start, red = end), the other vehicles
    thin in their role colour, the real recorded ego path as a thin white line on a driven (non-expert) episode."""
    img = cv2.addWeighted(base, 0.75, np.zeros_like(base), 0.25, 0)
    for j, (tr, r) in enumerate(zip(trails, roles)):
        if j != ei and len(tr) > 1:
            cv2.polylines(img, [np.array(tr, np.int32)], False, VIDEO_COLORS[r], 2, cv2.LINE_AA)
            cv2.circle(img, tr[-1], 5, VIDEO_COLORS[r], -1, cv2.LINE_AA)
    if driver != "expert":
        rp = [q for q in real_px if q]
        if len(rp) > 1:
            cv2.polylines(img, [np.array(rp, np.int32)], False, (255, 255, 255), 2, cv2.LINE_AA)
    tr = trails[ei]
    for i in range(1, len(tr)):
        f = i / max(len(tr) - 1, 1)
        cv2.line(img, tr[i - 1], tr[i], (int(255 * (1 - f)), 40, int(255 * f)), 4, cv2.LINE_AA)
    if tr:
        cv2.circle(img, tr[0], 8, (255, 0, 0), -1, cv2.LINE_AA)
        cv2.rectangle(img, (tr[-1][0] - 7, tr[-1][1] - 7), (tr[-1][0] + 7, tr[-1][1] + 7), (0, 0, 255), -1)
    title = "GROUND TRUTH (recorded path)" if driver == "expert" else f"AI MODEL ({driver})"
    cv2.rectangle(img, (0, 0), (img.shape[1], 62), (25, 25, 25), -1)
    cv2.putText(img, title + ("   - COLLISION" if hits else ""), (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (110, 110, 255) if hits else (240, 240, 240), 2, cv2.LINE_AA)
    legend = "ego: blue = start -> red = end" + ("   white = real path" if driver != "expert" else "") + \
             "   orange = other   cyan = neighbours"
    cv2.putText(img, legend, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.imwrite(out, img)


def episode(world, lib, store, s, driver, model, enc, drivable, ego_bp, device):
    shield = None
    if driver.endswith("+ttc"):
        driver, shield = driver[:-len("+ttc")], {"world": world, "brake": 0, "follow": 0}
    clear_world(world)
    sims, roles, ego_v = [], [], None
    for v in s["vehicles"]:
        if v["role"] == "ego":
            v = dict(v, blueprint=ego_bp)
            ego_v = v
        sv = spawn_vehicle(world, lib, store, v, "path")
        if sv is None:
            for o in sims:
                o.destroy()
            return {"result": "spawn failed"}
        sv.set_timed_route(*timed_route(v["frames"], v["xy"], s["start"]), z=DEFAULT_Z)
        sims.append(sv)
        roles.append(v["role"])
    ei = roles.index("ego")
    ego = sims[ei]
    rec_t, rec_xy, rec_v = (a.copy() for a in ego.timed_route)
    rec_s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(rec_xy, axis=0), axis=1))])
    for sv in sims:
        sv.hold_in_gear()
    for _ in range(SETTLE_TICKS):
        world.tick()
    for i, sv in enumerate(sims):
        if i == ei:
            sv.set_initial_speed(float(rec_v[0]))
        else:
            sv.vehicle.apply_control(carla.VehicleControl(hand_brake=False, manual_gear_shift=False))

    ext = ego.vehicle.bounding_box.extent
    cq, lq = queue.Queue(), queue.Queue()
    cam = world.spawn_actor(camera_bp(lib), carla.Transform(carla.Location(x=0.6 * ext.x, z=2 * ext.z + 0.2)),
                            attach_to=ego.vehicle)
    lid = world.spawn_actor(lidar_bp(lib), carla.Transform(carla.Location(z=LIDAR_Z)), attach_to=ego.vehicle)
    cam.listen(cq.put)
    lid.listen(lq.put)
    vcam = vwriter = None
    if VIDEO_DIR:
        # fixed top-down camera over all recorded paths, same placement for every driver of a scenario
        allxy = np.vstack([v["xy"] for v in s["vehicles"]])
        cx, cy = allxy.mean(0)
        span = max(np.ptp(allxy[:, 0]), np.ptp(allxy[:, 1])) + 20.0
        vfov = 2 * math.atan(math.tan(math.radians(VIDEO_FOV / 2)) * VIDEO_H / VIDEO_W)
        vtf = carla.Transform(carla.Location(x=float(cx), y=float(cy), z=float(max(25.0, span / 2 / math.tan(vfov / 2)))),
                              carla.Rotation(pitch=-90.0, yaw=0.0))
        vbp = lib.find("sensor.camera.rgb")
        vbp.set_attribute("image_size_x", str(VIDEO_W))
        vbp.set_attribute("image_size_y", str(VIDEO_H))
        vbp.set_attribute("fov", str(VIDEO_FOV))
        vcam = world.spawn_actor(vbp, vtf)
        vq = queue.Queue()
        vcam.listen(vq.put)
        vK = get_camera_intrinsic_matrix(VIDEO_W, VIDEO_H, VIDEO_FOV)
        os.makedirs(VIDEO_DIR, exist_ok=True)
        vpath = os.path.join(VIDEO_DIR, f"{s['id']}_{ego_bp.split('.')[-1]}_{driver}.mp4")
        vwriter = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"), float(SIM_FPS), (VIDEO_W, VIDEO_H))
        vtrail, vfirst, vtrails = [], None, [[] for _ in sims]

    vtype = torch.zeros(1, len(VEHICLE_TYPES), device=device)
    vtype[0, VEHICLE_TYPES.index(ego_bp)] = 1.0
    frames_real = np.asarray(ego_v["frames"])
    n_ticks = int(round(s["length_s"] * SIM_FPS))
    dt, clock = 1.0 / SIM_FPS, 0.0
    hist, recent, prev_speed = [], {}, None
    dist, speeds, gaps, plans = [], [], [], 0
    off_ticks = off_run = stuck_ticks = 0
    t_model, speed_i, speed_hist = 0.0, 0.0, []
    hit_from = None
    with contextlib.redirect_stdout(io.StringIO()):
        for tick in range(1, n_ticks + 1):
            for i, sv in enumerate(sims):
                if i == ei:
                    sv._sim_time = clock
                    speed_i = ego_update(sv, clock, speed_i, dt, shield if driver != "expert" else None)
                    if TICK_TRACE and sv.timed_route is not None:
                        _, _, rsp, rs = sv.reference(clock)
                        l = sv.vehicle.get_location()
                        own = sv._arc_length_at(l.x, l.y)
                        c = sv.vehicle.get_control()
                        print(f"TICK {clock:.2f} speed {kinematics(sv.vehicle)[3]:.2f} ref_speed {rsp:.2f} lag {rs - own:.2f} "
                              f"target {max(0.0, rsp + sv.TIMED_SPEED_GAIN * (rs - own)):.2f} thr {c.throttle:.2f} brk {c.brake:.2f}",
                              file=sys.stderr)
                else:
                    velocity_step(sv, clock, dt)
            frame = world.tick()
            clock += dt
            lm = lq.get(timeout=5.0)
            while lm.frame < frame:
                lm = lq.get(timeout=5.0)
            pts = np.frombuffer(lm.raw_data, np.float32).reshape(-1, 4)[:, :3].copy()
            pts[:, 1] *= -1
            kin = kinematics(ego.vehicle)
            recent[frame] = (pts, kin, round(clock, 4))
            for f in [f for f in recent if f < frame - 20]:
                del recent[f]
            while True:
                try:
                    cm = cq.get_nowait()
                except queue.Empty:
                    break
                if cm.frame not in recent:
                    continue
                cpts, ckin, ct = recent[cm.frame]
                if SMOOTH_SPEED and cm.frame - GRID_TICKS in recent:
                    # speed over the last 0.25 s (signed along the heading): the PID's throttle/brake switching makes the
                    # instantaneous speed jump +-0.5 m/s per tick, and copying a low sample into each plan ratchets the
                    # speed down (measured 29 Oct: constvel 7.4 -> 0.4 m/s)
                    pk = recent[cm.frame - GRID_TICKS][1]
                    h = math.radians(ckin[2])
                    v = ((ckin[0] - pk[0]) * math.cos(h) + (ckin[1] - pk[1]) * math.sin(h)) / GRID_S
                    ckin = ckin[:3] + (v,) + ckin[4:]
                img = np.frombuffer(cm.raw_data, np.uint8).reshape(cm.height, cm.width, 4)[:, :, :3]
                t0 = time.time()
                hist = (hist + [(enc.camera_img(img), enc.lidar_pts(cpts))])[-HISTORY:]
                accel = float(np.clip((ckin[3] - prev_speed) / GRID_S, -T.ACCEL_CLIP, T.ACCEL_CLIP)) if prev_speed is not None else 0.0
                prev_speed = ckin[3]
                speed_hist = (speed_hist + [ckin[3]])[-5:]
                if ACCEL_MODE == "zero":
                    accel = 0.0
                elif ACCEL_MODE == "smooth" and len(speed_hist) == 5:   # over the last 1 s instead of 0.25 s
                    accel = float(np.clip((speed_hist[-1] - speed_hist[0]) / (4 * GRID_S), -T.ACCEL_CLIP, T.ACCEL_CLIP))
                if driver == "expert" or len(hist) < HISTORY:
                    t_model += time.time() - t0
                    continue
                if driver == "constvel":
                    wp = np.stack([max(ckin[3], 0.0) * GRID_S * np.arange(1, N_WP + 1), np.zeros(N_WP)], 1)
                else:
                    ci = int(np.argmin(np.abs(frames_real - (s["start"] + ct * sc.REAL_WORLD_FPS))))
                    route = torch.zeros(1, len(ROUTE_CMDS), device=device)
                    route[0, ROUTE_CMDS.index(ego_v["route_cmd"][ci])] = 1.0
                    # speed history like the recorder: own speed 0.25 ... 1.0 s ago, the earliest one while younger
                    past = [speed_hist[max(len(speed_hist) - 1 - h, 0)] for h in SPEED_HISTORY]
                    st = torch.tensor(make_state(getattr(model, "state_layout", "accel"), [ckin[3]], [past], [accel],
                                                 [ckin[4]], [ckin[5]]), dtype=torch.float32, device=device)
                    with torch.no_grad():
                        w, scr, _ = model(torch.stack([h[0] for h in hist])[None], torch.stack([h[1] for h in hist])[None],
                                          st, route, vtype)
                    wp = w[0, int(scr[0].argmax())].cpu().numpy()
                t_model += time.time() - t0
                if PROBE:
                    # the expert keeps driving the recorded route; only log what the model would have planned
                    own = arc_length_at(rec_xy, rec_s, ckin[0], ckin[1])
                    fut = float(np.interp(ct + N_WP * GRID_S, rec_t, rec_s)) - own
                    plan = float(np.sum(np.linalg.norm(np.diff(np.vstack([[0.0, 0.0], wp]), axis=0), axis=1)))
                    print(f"PROBE {s['id']} {ego_bp} t {ct:.2f} speed {ckin[3]:.3f} acc {accel:.3f} plan_m {plan:.3f} "
                          f"rec_m {fut:.3f} cv_m {max(ckin[3], 0.0) * N_WP * GRID_S:.3f} wp0 {wp[0][0]:.3f}", file=sys.stderr)
                    continue
                xy = np.vstack([[ckin[0], ckin[1]], from_ego(wp, ckin[0], ckin[1], ckin[2])])
                sp = np.r_[max(ckin[3], 0.0), np.linalg.norm(np.diff(xy, axis=0), axis=1) / GRID_S]
                ego.set_timed_route(ct + GRID_S * np.arange(N_WP + 1), xy, sp, z=DEFAULT_Z)
                plans += 1
                if TRACE:
                    print(f"TRACE t {ct:.2f} clock {clock:.2f} speed {ckin[3]:.2f} acc {accel:.2f} wp0 {wp[0].round(2)} wp_end {wp[-1].round(2)} speed_i {speed_i:.2f} "
                          f"rec_speed {np.interp(ct, rec_t, rec_v):.2f} ctrl {ego.vehicle.get_control()}", file=sys.stderr)
                    tf = ego.vehicle.get_transform()
                    bx = actor_box(ego.vehicle)
                    print(f"TRACE2 pos {tf.location.x:.1f},{tf.location.y:.1f},{tf.location.z:.2f} pitch {tf.rotation.pitch:.1f} "
                          f"roll {tf.rotation.roll:.1f} rec_pos {np.interp(ct, rec_t, rec_xy[:, 0]):.1f},{np.interp(ct, rec_t, rec_xy[:, 1]):.1f} "
                          f"gap {min(sc.box_gap(bx, actor_box(o.vehicle)) for j, o in enumerate(sims) if j != ei):.2f} "
                          f"offroad_corners {sum(off_road(*drivable, *c) for c in bx) if drivable is not None else -1} "
                          f"events {len(ego.collision_events)}", file=sys.stderr)
            # metrics
            rx, ry = np.interp(clock, rec_t, rec_xy[:, 0]), np.interp(clock, rec_t, rec_xy[:, 1])
            dist.append(math.hypot(kin[0] - rx, kin[1] - ry))
            speeds.append(kin[3])
            if kin[3] < STUCK_SPEED and np.interp(clock, rec_t, rec_v) > EXPERT_MOVING:
                stuck_ticks += 1
            if hit_from is None and ego.collision_events:
                by_id = {o.vehicle.id: o.vehicle for o in sims}
                for _, oid, typ in ego.collision_events:
                    if typ.startswith("vehicle.") and oid in by_id:
                        fwd = ego.vehicle.get_transform().get_forward_vector()
                        rel = by_id[oid].get_location() - ego.vehicle.get_location()
                        hit_from = "ahead" if rel.x * fwd.x + rel.y * fwd.y > 0 else "behind"
                        break
            box = actor_box(ego.vehicle)
            gaps.append(min([sc.box_gap(box, actor_box(o.vehicle)) for j, o in enumerate(sims) if j != ei] or [np.inf]))
            if drivable is not None and sum(off_road(*drivable, *c) for c in box) >= 2:
                off_ticks += 1
                off_run = max(off_run, off_ticks)
            else:
                off_ticks = 0
            if vwriter is not None:
                im = vq.get(timeout=5.0)
                while im.frame < frame:
                    im = vq.get(timeout=5.0)
                img = np.frombuffer(im.raw_data, np.uint8).reshape(im.height, im.width, 4)[:, :, :3].copy()
                if vfirst is None:
                    vfirst = img.copy()
                for j, sv in enumerate(sims):
                    q = world_to_pixel(sv.vehicle.get_location(), vtf, vK)
                    if q:
                        vtrails[j].append(q)
                pr = world_to_pixel(carla.Location(x=float(rx), y=float(ry), z=DEFAULT_Z), vtf, vK)
                pe = world_to_pixel(ego.vehicle.get_location(), vtf, vK)
                if pe:
                    vtrail.append(pe)
                if len(vtrail) > 1:
                    cv2.polylines(img, [np.array(vtrail, np.int32)], False, (60, 220, 60), 2, cv2.LINE_AA)
                if pr:
                    cv2.circle(img, pr, 9, (255, 255, 255), 2, cv2.LINE_AA)   # where the real vehicle is at this time
                for sv, r in zip(sims, roles):
                    q = world_to_pixel(sv.vehicle.get_location(), vtf, vK)
                    if q and 0 <= q[0] < VIDEO_W and 0 <= q[1] < VIDEO_H:
                        hit = sv is ego and bool(ego.collision_events)
                        col = (0, 0, 255) if hit else VIDEO_COLORS[r]
                        cv2.circle(img, q, 6, col, -1, cv2.LINE_AA)
                        cv2.putText(img, r + (" CRASH" if hit else ""), (q[0] + 8, q[1] - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
                cv2.rectangle(img, (0, 0), (VIDEO_W, 34), (25, 25, 25), -1)
                cv2.putText(img, f"t {clock:4.1f} s   ego speed {kin[3]:4.1f} m/s   real {np.interp(clock, rec_t, rec_v):4.1f} m/s   "
                                 f"off by {dist[-1]:4.1f} m", (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
                vwriter.write(img)

    ids = {sv.vehicle.id: r for sv, r in zip(sims, roles)}
    veh_hits = sorted({ids.get(o, "unknown") for _, o, typ in ego.collision_events if typ.startswith("vehicle.")})
    static_hits = sorted({typ for _, _, typ in ego.collision_events if not typ.startswith("vehicle.")})
    loc = ego.vehicle.get_location()
    acc = np.diff(speeds) * SIM_FPS
    jerk = np.diff(acc) * SIM_FPS
    for x in (cam, lid) + ((vcam,) if vcam is not None else ()):
        x.stop()
        x.destroy()
    if vwriter is not None:
        vwriter.release()
        if vfirst is not None:
            traj_image(vfirst, vtrails, roles, ei, [world_to_pixel(carla.Location(x=float(x), y=float(y), z=DEFAULT_Z), vtf, vK)
                                                    for x, y in rec_xy], driver, ego.collision_events,
                       vpath.replace(".mp4", "_trajectory.png"))
    for sv in sims:
        sv.destroy()
    world.tick()
    return {
        "result": "collision" if veh_hits or static_hits else ("off road" if off_run >= KERB_TICKS else "ok"),
        "vehicle_hit": ";".join(veh_hits), "static_hit": ";".join(static_hits), "hit_from": hit_from,
        "off_road_s": round(off_run / SIM_FPS, 2),
        "progress": round(arc_length_at(rec_xy, rec_s, loc.x, loc.y) / max(rec_s[-1], 1e-6), 3),
        "route_m": round(float(rec_s[-1]), 1),
        "dist_mean_m": round(float(np.mean(dist)), 3), "dist_max_m": round(float(np.max(dist)), 3),
        "dist_final_m": round(float(dist[-1]), 3), "stuck_s": round(stuck_ticks / SIM_FPS, 2),
        "min_gap_m": round(float(np.min(gaps)), 3), "mean_speed": round(float(np.mean(speeds)), 3),
        "mean_abs_accel": round(float(np.mean(np.abs(acc))), 3), "p95_abs_jerk": round(float(np.percentile(np.abs(jerk), 95)), 2),
        "plans": plans, "model_s": round(t_model, 2),
        "shield_brake_s": round(shield["brake"] / SIM_FPS, 2) if shield else None,
        "shield_follow_s": round(shield["follow"] / SIM_FPS, 2) if shield else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--drivers", nargs="+", default=["expert", "constvel", "s_K1", "reg_C_droplr"])
    p.add_argument("--types", choices=["own", "all"], default="own")
    p.add_argument("--limit", type=int, default=None, help="Only the first N scenarios (fixed random order).")
    p.add_argument("--tag", default="own")
    p.add_argument("--ids", nargs="*", help="Only these scenario ids.")
    p.add_argument("--any-split", action="store_true", help="Scenarios from train/val too (showcase videos only).")
    p.add_argument("--split", choices=["test", "val"], default="test",
                   help="val: checkpoint selection (il/select_ckpt_closed_loop.py), so the test scenarios stay unseen.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    scens = test_scenarios(args.any_split, args.split)
    rng = np.random.default_rng(0)
    scens = [scens[i] for i in rng.permutation(len(scens))]
    scens = [x for x in scens if not args.ids or x["id"] in args.ids][:args.limit]
    out = os.path.join(IL_DIR, f"closed_loop_{args.tag}.csv")
    done = set()
    if os.path.exists(out):
        d = pd.read_csv(out, dtype={"scenario": str})
        done = set(zip(d.scenario, d.ego_bp, d.driver))
    models = {}
    for dr in args.drivers:
        base = dr[:-len("+ttc")] if dr.endswith("+ttc") else dr
        if base not in ("expert", "constvel"):
            run, _, ep = base.partition("@")   # 'run@ep05' = that run's ckpt_ep05.pt
            ck = torch.load(os.path.join(RUNS, run, f"ckpt_{ep}.pt" if ep else "model.pt"), weights_only=False)
            m = T.model_from_args(ck["args"]).to(args.device).eval()
            m.load_state_dict(ck["model"])
            models[dr] = m
    enc = OnlineEncoders(args.device)
    drivable = load_map()
    store = sc.TrackStore(track_ids={v["track"] for s in scens for v in s["vehicles"]}, extra_columns=["heading_angle"]
                          if "heading_angle" not in sc.CSV_COLUMNS else ())
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
    jobs = [(s, bp, dr) for s in scens
            for bp in ([v["blueprint"] for v in s["vehicles"] if v["role"] == "ego"] if args.types == "own" else VEHICLE_TYPES)
            for dr in args.drivers if (s["id"], bp, dr) not in done]
    print(f"{len(scens)} test scenarios, {len(done)} episodes done, {len(jobs)} to run", flush=True)
    t_all, streak = time.time(), 0
    try:
        for i, (s, bp, dr) in enumerate(jobs):
            ts = time.time()
            try:
                row = episode(world, lib, store, s, dr, models.get(dr), enc, drivable, bp, args.device)
                streak = 0
            except Exception as ex:
                streak += 1
                print(f"  [{i + 1}/{len(jobs)}] {s['id']} {bp} {dr}: ERROR {ex!r}", flush=True)
                if streak >= MAX_ERROR_STREAK:
                    raise SystemExit(1)
                clear_world(world)
                continue
            own = [v["blueprint"] for v in s["vehicles"] if v["role"] == "ego"][0]
            row = {"scenario": s["id"], "behaviour": s["behaviour"], "length_s": s["length_s"], "ego_bp": bp,
                   "own_bp": bp == own, "driver": dr, **row, "seconds": round(time.time() - ts, 1)}
            pd.DataFrame([row]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            print(f"  [{i + 1}/{len(jobs)}] {s['id']} {s['behaviour']} {bp.split('.')[-1]} {dr}: {row['result']}, "
                  f"dist {row.get('dist_mean_m')} m, progress {row.get('progress')}, {row['seconds']} s", flush=True)
    finally:
        clear_world(world)
        world.apply_settings(orig)
    d = pd.read_csv(out, dtype={"scenario": str})
    d = d[d.result != "spawn failed"]
    g = d.groupby("driver").agg(n=("scenario", "size"), collision=("result", lambda r: (r == "collision").mean()),
                                off_road=("result", lambda r: (r == "off road").mean()), progress=("progress", "mean"),
                                dist_mean=("dist_mean_m", "mean"), dist_final=("dist_final_m", "mean"),
                                stuck_s=("stuck_s", "mean"), jerk95=("p95_abs_jerk", "mean"))
    print("\n" + g.round(3).to_string())
    print(f"saved {out} ({(time.time() - t_all) / 60:.1f} min)")


if __name__ == "__main__":
    main()
