#!/usr/bin/env python3
"""
il/closed_loop_multi.py

Closed-loop test where the TTC safety layer acts on EVERY vehicle (Raj's professor, 2026-11-13): the driver drives the ego
only (as in il/closed_loop.py); every other vehicle follows its recorded path on the same PID controller (physics on)
instead of the non-reactive velocity replay, so with '+ttc' it can brake / yield too. With --all-vehicles the driver
drives every vehicle (each with its own camera + LiDAR); not the default.
Drivers (same names as il/closed_loop.py):
  expert          ego: PID on its recorded path
  constvel        ego: PID on a straight constant-speed path from its current state, replanned every 0.25 s
  <run>[@epNN|@cl] ego: the trained model, from its own sensors, speed history, route command and blueprint
  <driver>+ttc[:B:G]  plus the TTC layer on EVERY vehicle (closed_loop.ego_update): predicted box gap < G m (default 0.4)
                  with a vehicle ahead sooner than B s (default 1.5) -> full brake, later -> speed cap + 0.5 m nudge
Behaviour check: the driven ego/other paths are sampled at the recorded drone frames and compared with the recorded ones by
augment_scenarios.relation / same_behaviour (who is ahead at start / end, direction of the lateral change > 0.5 m).
One CSV row per (scenario, driver): collisions (any driven vehicle; ego; ego/other pair), behaviour kept, per-vehicle
mean speed vs recorded, progress, stuck time, time the TTC layer braked / followed. Resumable.

  python il/closed_loop_multi.py --port 2000 --split val --limit 50 --tag multi_tune \\
      --drivers rhd_scratch@cl+ttc rhd_scratch@cl+ttc:1.0:0.4 rhd_scratch@cl+ttc:1.0:0.2
"""

import argparse
import contextlib
import io
import math
import os
import queue
import sys
import time

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import closed_loop as CL  # noqa: E402
from closed_loop import (carla, sc, SIM_FPS, TOWN, DEFAULT_Z, SETTLE_TICKS, load_map, off_road, IL_DIR, KERB_TICKS,  # noqa: E402
                         timed_route, clear_world, spawn_vehicle, lidar_bp, camera_bp, GRID_S, GRID_TICKS, LIDAR_Z,
                         N_WP, HISTORY, ROUTE_CMDS, VEHICLE_TYPES, SPEED_HISTORY, make_state, T, RUNS, STUCK_SPEED,
                         EXPERT_MOVING, MAX_ERROR_STREAK, OnlineEncoders, test_scenarios, arc_length_at, kinematics,
                         from_ego, ego_update, actor_box)
from augment_scenarios import relation, same_behaviour  # noqa: E402
from behaviour_v2 import behaviour2  # noqa: E402
import idm as IDM  # noqa: E402
from model import lead_feats  # noqa: E402
import cv2  # noqa: E402
from closed_loop import (VIDEO_DIR, VIDEO_W, VIDEO_H, VIDEO_FOV, VIDEO_COLORS, traj_image, get_camera_intrinsic_matrix,  # noqa: E402
                         world_to_pixel)

FPS = sc.REAL_WORLD_FPS
TM = None                    # carla.TrafficManager, only for the "autopilot" driver
AUTOPILOTS = ("autopilot", "autopilot_stock")   # "autopilot": recorded path and speed given; "autopilot_stock": defaults
AUTOPILOT_PATH_STEP = 2.0    # m between the route points given to the Traffic Manager


def parse_driver(name):
    """'run@cl+ttc:1.0:0.2' -> ('run@cl', {'brake_s': 1.0, 'gap_m': 0.2}) ; no '+ttc' -> (name, None)."""
    base, _, ttc = name.partition("+ttc")
    if not name[len(base):].startswith("+ttc"):
        return name, None
    parts = [float(x) for x in ttc.split(":")[1:]] if ttc else []
    return base, {"brake_s": parts[0] if parts else 1.5, "gap_m": parts[1] if len(parts) > 1 else 0.4}


def driven_paths(s, trails):
    """Each vehicle's driven path at the recorded drone frames that fall inside the driven time."""
    out = []
    for v, tr in zip(s["vehicles"], trails):
        tr = np.asarray(tr)
        t = (np.asarray(v["frames"]) - s["start"]) / FPS
        keep = t <= tr[-1, 0]
        out.append(dict(v, frames=[f for f, k in zip(v["frames"], keep) if k],
                        xy=np.stack([np.interp(t[keep], tr[:, 0], tr[:, 1]), np.interp(t[keep], tr[:, 0], tr[:, 2])], 1).tolist()))
    return out


def pair(vehicles):
    by = {v["role"]: v for v in vehicles}
    return [by[r] for r in ("ego", "other") if r in by]


def episode(world, lib, store, s, driver, model, enc, drivable, device, all_vehicles=False):
    base, ttc = parse_driver(driver)
    roles = [v["role"] for v in s["vehicles"]]
    driven = list(range(len(roles))) if all_vehicles else [roles.index("ego")]   # the rest: PID on the recorded path
    clear_world(world)
    sims = []
    for v in s["vehicles"]:
        sv = spawn_vehicle(world, lib, store, v, "path")
        if sv is None:
            for o in sims:
                o.destroy()
            return {"result": "spawn failed"}
        sv.set_timed_route(*timed_route(v["frames"], v["xy"], s["start"]), z=DEFAULT_Z)
        sims.append(sv)
    n = len(sims)
    rec = []
    for sv in sims:
        t_, xy_, v_ = (a.copy() for a in sv.timed_route)
        rec.append((t_, xy_, v_, np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy_, axis=0), axis=1))]))
    for sv in sims:
        sv.hold_in_gear()
    for _ in range(SETTLE_TICKS):
        world.tick()
    for sv, r in zip(sims, rec):
        sv.set_initial_speed(float(r[2][0]))
    use_model = base not in ("expert", "constvel", "autopilot", "autopilot_stock", "idm")
    sensors, cqs, lqs = [], {}, {}
    if use_model:
        for i in driven:
            sv = sims[i]
            ext = sv.vehicle.bounding_box.extent
            cq, lq = queue.Queue(), queue.Queue()
            cam = world.spawn_actor(camera_bp(lib), carla.Transform(carla.Location(x=0.6 * ext.x, z=2 * ext.z + 0.2)),
                                    attach_to=sv.vehicle)
            lid = world.spawn_actor(lidar_bp(lib), carla.Transform(carla.Location(z=LIDAR_Z)), attach_to=sv.vehicle)
            cam.listen(cq.put)
            lid.listen(lq.put)
            sensors += [cam, lid]
            cqs[i], lqs[i] = cq, lq
    vtypes = []
    for v in s["vehicles"]:
        vt = torch.zeros(1, len(VEHICLE_TYPES), device=device)
        vt[0, VEHICLE_TYPES.index(v["blueprint"])] = 1.0
        vtypes.append(vt)
    shields = [dict(world=world, brake=0, follow=0, **ttc) if ttc else None for _ in sims]
    ei = roles.index("ego")
    if base == "autopilot":   # CARLA Traffic Manager drives the ego: its recorded path as the route, its recorded mean speed
        ev = sims[ei].vehicle                                  # as the target speed, lights and signs ignored (none recorded)
        ev.set_autopilot(True, TM.get_port())
        TM.ignore_lights_percentage(ev, 100.0)
        TM.ignore_signs_percentage(ev, 100.0)
        TM.auto_lane_change(ev, False)
        TM.set_desired_speed(ev, 3.6 * float(np.mean(rec[ei][2])))
        path = [rec[ei][1][0]]
        for q in rec[ei][1][1:]:
            if np.linalg.norm(q - path[-1]) >= AUTOPILOT_PATH_STEP:
                path.append(q)
        if len(path) > 1:
            TM.set_path(ev, [carla.Location(x=float(q[0]), y=float(q[1]), z=DEFAULT_Z) for q in path])
    if base == "autopilot_stock":   # stock CARLA Autopilot: Traffic Manager defaults, nothing from the recording but the start
        sims[ei].vehicle.set_autopilot(True, TM.get_port())   # pose and start speed (own route, speed limit, lights, lane changes)
    vcam = vwriter = None
    if VIDEO_DIR:   # fixed top-down camera over all recorded paths, same placement for every driver of a scenario
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
        vpath = os.path.join(VIDEO_DIR, f"{s['id']}_{driver.replace(':', '_')}.mp4")
        vwriter = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"), float(SIM_FPS), (VIDEO_W, VIDEO_H))
        vfirst, vtrails = None, [[] for _ in sims]
        title = "CARLA FROM DRONE (recorded paths)" if base == "expert" and not ttc else \
            ("AI MODEL ego" if use_model else base) + (f" + TTC on all ({ttc['brake_s']} s / {ttc['gap_m']} m)" if ttc else "")
    n_ticks = int(round(s["length_s"] * SIM_FPS))
    dt, clock = 1.0 / SIM_FPS, 0.0
    hist = [[] for _ in sims]
    recent = [{} for _ in sims]
    speed_hist = [[] for _ in sims]
    speed_i = [0.0] * n
    prev_speed = [None] * n
    pos_hist = [[] for _ in sims]             # (clock, x, y, yaw) per tick, for const-vel speed and the behaviour check
    trails = [[] for _ in sims]
    ticklog = [[] for _ in sims]              # (clock, x, y, yaw deg, speed, yaw rate, TTC layer: 0 off / 1 speed cap / 2 full brake)
    sh_seen = [(0, 0)] * n
    speeds = [[] for _ in sims]
    stuck = [0] * n
    off_ticks, off_run = [0] * n, [0] * n
    t_model, plans = 0.0, 0
    idm_p, idm_ext, idm_path = None, None, {}
    use_lead = use_model and getattr(model, "lead", False)   # lead-vehicle input from the simulator (privileged, il/lead_input.py)
    lead_ext = np.array([[sv.vehicle.bounding_box.extent.x, sv.vehicle.bounding_box.extent.y] for sv in sims])
    all_kins = {}                                            # frame -> every vehicle's kinematics at that tick
    with contextlib.redirect_stdout(io.StringIO()):
        for tick in range(1, n_ticks + 1):
            for i, sv in enumerate(sims):
                sv._sim_time = clock
                if base in AUTOPILOTS and i == ei:              # the Traffic Manager applies the control
                    if sv._constant_velocity_until is not None and clock >= sv._constant_velocity_until:
                        sv.release_launch_hold()
                    continue
                speed_i[i] = ego_update(sv, clock, speed_i[i], dt, shields[i])
            frame = world.tick()
            clock += dt
            kins = [kinematics(sv.vehicle) for sv in sims]
            for i in range(n):
                pos_hist[i].append((clock, kins[i][0], kins[i][1]))
                pos_hist[i] = pos_hist[i][-(GRID_TICKS + 1):]
            if base == "constvel" and tick % GRID_TICKS == 0 and clock >= HISTORY * GRID_S:
                for i in driven:
                    sv, k = sims[i], kins[i]
                    (_, x0, y0) = pos_hist[i][0]
                    h = math.radians(k[2])
                    v = max(0.0, ((k[0] - x0) * math.cos(h) + (k[1] - y0) * math.sin(h)) / GRID_S)
                    wp = np.stack([v * GRID_S * np.arange(1, N_WP + 1), np.zeros(N_WP)], 1)
                    xy = np.vstack([[k[0], k[1]], from_ego(wp, k[0], k[1], k[2])])
                    sv.set_timed_route(clock + GRID_S * np.arange(N_WP + 1), xy, np.full(N_WP + 1, v), z=DEFAULT_Z)
                    plans += 1
            if base == "idm" and tick % GRID_TICKS == 0:   # IDM picks the speed along the recorded path (il/idm.py)
                if idm_p is None:
                    idm_p = IDM.load_params()
                    idm_ext = np.array([[sv.vehicle.bounding_box.extent.x, sv.vehicle.bounding_box.extent.y] for sv in sims])
                K = np.array(kins)
                for i in driven:
                    sv, k = sims[i], kins[i]
                    if idm_path.get(i) is None:            # recorded path, repeated points dropped, 150 m straight on at the end
                        q = [rec[i][1][0]]
                        for pt in rec[i][1][1:]:
                            if np.linalg.norm(pt - q[-1]) >= 0.05:
                                q.append(pt)
                        h = math.radians(k[2])
                        d = (q[-1] - q[-2]) / np.linalg.norm(q[-1] - q[-2]) if len(q) > 1 else np.array([math.cos(h), math.sin(h)])
                        q = np.vstack(q + [q[-1] + 150.0 * d])
                        idm_path[i] = (q, np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
                    q, cum = idm_path[i]
                    v = max(k[3], 0.0)
                    gap, dv = IDM.lead(i, K[:, 0], K[:, 1], np.radians(K[:, 2]), K[:, 3], idm_ext)
                    acc = float(IDM.idm_accel(v, gap, dv, idm_p["v0"][s["vehicles"][i]["blueprint"]], idm_p["T"], idm_p["s0"],
                                              idm_p["a"], idm_p["b"]))
                    vs = np.clip(v + acc * GRID_S * np.arange(N_WP + 1), 0.0, None)
                    at = arc_length_at(q, cum, k[0], k[1]) + np.r_[0.0, np.cumsum((vs[:-1] + vs[1:]) / 2 * GRID_S)]
                    sv.set_timed_route(clock + GRID_S * np.arange(N_WP + 1),
                                       np.stack([np.interp(at, cum, q[:, 0]), np.interp(at, cum, q[:, 1])], 1), vs, z=DEFAULT_Z)
                    plans += 1
            if use_lead:
                all_kins[frame] = np.array(kins)
                for f in [f for f in all_kins if f < frame - 20]:
                    del all_kins[f]
            if use_model:
                for i in driven:
                    sv = sims[i]
                    lm = lqs[i].get(timeout=5.0)
                    while lm.frame < frame:
                        lm = lqs[i].get(timeout=5.0)
                    pts = np.frombuffer(lm.raw_data, np.float32).reshape(-1, 4)[:, :3].copy()
                    pts[:, 1] *= -1
                    recent[i][frame] = (pts, kins[i], round(clock, 4))
                    for f in [f for f in recent[i] if f < frame - 20]:
                        del recent[i][f]
                    while True:
                        try:
                            cm = cqs[i].get_nowait()
                        except queue.Empty:
                            break
                        if cm.frame not in recent[i]:
                            continue
                        cpts, ckin, ct = recent[i][cm.frame]
                        if CL.SMOOTH_SPEED and cm.frame - GRID_TICKS in recent[i]:
                            pk = recent[i][cm.frame - GRID_TICKS][1]
                            h = math.radians(ckin[2])
                            vv = ((ckin[0] - pk[0]) * math.cos(h) + (ckin[1] - pk[1]) * math.sin(h)) / GRID_S
                            ckin = ckin[:3] + (vv,) + ckin[4:]
                        img = np.frombuffer(cm.raw_data, np.uint8).reshape(cm.height, cm.width, 4)[:, :, :3]
                        t0 = time.time()
                        hist[i] = (hist[i] + [(enc.camera_img(img), enc.lidar_pts(cpts))])[-HISTORY:]
                        accel = float(np.clip((ckin[3] - prev_speed[i]) / GRID_S, -T.ACCEL_CLIP, T.ACCEL_CLIP)) \
                            if prev_speed[i] is not None else 0.0
                        prev_speed[i] = ckin[3]
                        speed_hist[i] = (speed_hist[i] + [ckin[3]])[-5:]
                        if len(hist[i]) < HISTORY:
                            t_model += time.time() - t0
                            continue
                        v = s["vehicles"][i]
                        ci = int(np.argmin(np.abs(np.asarray(v["frames"]) - (s["start"] + ct * FPS))))
                        route = torch.zeros(1, len(ROUTE_CMDS), device=device)
                        route[0, ROUTE_CMDS.index(v["route_cmd"][ci])] = 1.0
                        past = [speed_hist[i][max(len(speed_hist[i]) - 1 - h, 0)] for h in SPEED_HISTORY]
                        st = torch.tensor(make_state(getattr(model, "state_layout", "accel"), [ckin[3]], [past], [accel],
                                                     [ckin[4]], [ckin[5]]), dtype=torch.float32, device=device)
                        if use_lead:
                            K = all_kins[cm.frame].copy()
                            K[i, 3] = ckin[3]
                            st = torch.cat([st, torch.tensor(lead_feats(*IDM.lead(i, K[:, 0], K[:, 1], np.radians(K[:, 2]), K[:, 3], lead_ext)),
                                                             device=device)], 1)
                        with torch.no_grad():
                            w, scr, _ = model(torch.stack([h[0] for h in hist[i]])[None],
                                              torch.stack([h[1] for h in hist[i]])[None], st, route, vtypes[i])
                        wp = w[0, int(scr[0].argmax())].cpu().numpy()
                        t_model += time.time() - t0
                        xy = np.vstack([[ckin[0], ckin[1]], from_ego(wp, ckin[0], ckin[1], ckin[2])])
                        sp = np.r_[max(ckin[3], 0.0), np.linalg.norm(np.diff(xy, axis=0), axis=1) / GRID_S]
                        sv.set_timed_route(ct + GRID_S * np.arange(N_WP + 1), xy, sp, z=DEFAULT_Z)
                        plans += 1
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
                if len(vtrails[ei]) > 1:
                    cv2.polylines(img, [np.array(vtrails[ei], np.int32)], False, (60, 220, 60), 2, cv2.LINE_AA)
                rt, rxy = rec[ei][0], rec[ei][1]
                pr = world_to_pixel(carla.Location(x=float(np.interp(clock, rt, rxy[:, 0])), y=float(np.interp(clock, rt, rxy[:, 1])),
                                                   z=DEFAULT_Z), vtf, vK)
                if pr and use_model or pr and base == "constvel":
                    cv2.circle(img, pr, 9, (255, 255, 255), 2, cv2.LINE_AA)   # where the real ego is at this time
                for j, (sv, r) in enumerate(zip(sims, roles)):
                    q = world_to_pixel(sv.vehicle.get_location(), vtf, vK)
                    if q and 0 <= q[0] < VIDEO_W and 0 <= q[1] < VIDEO_H:
                        hit = bool(sv.collision_events)
                        braking = shields[j] is not None and sv.vehicle.get_control().brake > 0.9
                        col = (0, 0, 255) if hit else VIDEO_COLORS[r]
                        cv2.circle(img, q, 6, col, -1, cv2.LINE_AA)
                        cv2.putText(img, r + (" CRASH" if hit else (" TTC" if braking else "")), (q[0] + 8, q[1] - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
                cv2.rectangle(img, (0, 0), (VIDEO_W, 58), (25, 25, 25), -1)
                cv2.putText(img, f"{title}   #{s['id']} {s['behaviour']}", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (240, 240, 240), 1, cv2.LINE_AA)
                cv2.putText(img, f"t {clock:4.1f} s   ego speed {kins[ei][3]:4.1f} m/s   real {np.interp(clock, rt, rec[ei][2]):4.1f} m/s",
                            (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)
                vwriter.write(img)
            for i, sv in enumerate(sims):
                k = kins[i]
                trails[i].append((clock, k[0], k[1]))
                sh = shields[i]
                layer = 0 if sh is None else (2 if sh["brake"] > sh_seen[i][0] else (1 if sh["follow"] > sh_seen[i][1] else 0))
                if sh is not None:
                    sh_seen[i] = (sh["brake"], sh["follow"])
                ticklog[i].append((clock, k[0], k[1], k[2], k[3], k[4], layer))
                speeds[i].append(k[3])
                rt, rxy, rv, _ = rec[i]
                if k[3] < STUCK_SPEED and rt[0] <= clock <= rt[-1] and np.interp(clock, rt, rv) > EXPERT_MOVING:
                    stuck[i] += 1
                if drivable is not None and sum(off_road(*drivable, *c) for c in actor_box(sv.vehicle)) >= 2:
                    off_ticks[i] += 1
                    off_run[i] = max(off_run[i], off_ticks[i])
                else:
                    off_ticks[i] = 0
    for x in sensors + ([vcam] if vcam is not None else []):
        x.stop()
        x.destroy()
    if vwriter is not None:
        vwriter.release()
        if vfirst is not None:
            real_px = [world_to_pixel(carla.Location(x=float(x), y=float(y), z=DEFAULT_Z), vtf, vK) for x, y in rec[ei][1]]
            traj_image(vfirst, vtrails, roles, ei, real_px, "expert" if base == "expert" and not ttc else title,
                       any(sv.collision_events for sv in sims), vpath.replace(".mp4", "_trajectory.png"))
    ids = {sv.vehicle.id: i for i, sv in enumerate(sims)}
    hits = set()                                       # vehicle pairs (i, j) in contact, i < j
    static = [False] * n
    for i, sv in enumerate(sims):
        for _, oid, typ in sv.collision_events:
            if typ.startswith("vehicle.") and oid in ids:
                hits.add(tuple(sorted((i, ids[oid]))))
            elif not typ.startswith("vehicle."):
                static[i] = True
    in_coll = [static[i] or any(i in p for p in hits) for i in range(n)]
    oi = roles.index("other") if "other" in roles else None
    base_rel = relation(pair(s["vehicles"]))
    new_rel = relation(pair(driven_paths(s, trails)))
    b2 = behaviour2(pair(s["vehicles"]), pair(driven_paths(s, trails)))
    if os.environ.get("CL_TRAILS"):                    # driven + recorded paths for offline metrics (il/behaviour_v2.py)
        os.makedirs(os.environ["CL_TRAILS"], exist_ok=True)
        np.savez(os.path.join(os.environ["CL_TRAILS"], f"{s['id']}__{driver.replace(':', '_')}.npz"),
                 scenario=s["id"], driver=driver, behaviour=s["behaviour"], roles=np.array(roles), start=s["start"], fps=FPS,
                 ego_collision=bool(in_coll[ei]), any_collision=bool(any(in_coll)),
                 dt=dt, **{f"trail{i}": np.asarray(t, float) for i, t in enumerate(trails)},
                 **{f"tick{i}": np.asarray(t, float) for i, t in enumerate(ticklog)},
                 **{f"ext{i}": np.array([sv.vehicle.bounding_box.extent.x, sv.vehicle.bounding_box.extent.y]) for i, sv in enumerate(sims)},
                 **{f"coll{i}": np.array([(tc, ids.get(oid, -1)) for tc, oid, _ in sv.collision_events], float).reshape(-1, 2)
                    for i, sv in enumerate(sims)},
                 **{f"frames{i}": np.asarray(v["frames"]) for i, v in enumerate(s["vehicles"])},
                 **{f"xy{i}": np.asarray(v["xy"], float) for i, v in enumerate(s["vehicles"])})
    prog = []
    for i, sv in enumerate(sims):
        loc = sv.vehicle.get_location()
        prog.append(arc_length_at(rec[i][1], rec[i][3], loc.x, loc.y) / max(rec[i][3][-1], 1e-6))
    # mean speed driven vs recorded, inside each vehicle's recorded time window
    clocks = np.arange(1, n_ticks + 1) * dt
    drv_speed, rec_speed = [], []
    for i in range(n):
        m = (clocks >= rec[i][0][0]) & (clocks <= rec[i][0][-1])
        drv_speed.append(float(np.mean(np.asarray(speeds[i])[m])) if m.any() else 0.0)
        rec_speed.append(float(np.mean(np.interp(clocks[m], rec[i][0], rec[i][2]))) if m.any() else 0.0)
    if base in AUTOPILOTS:
        sims[ei].vehicle.set_autopilot(False, TM.get_port())
    for sv in sims:
        sv.destroy()
    world.tick()
    sh = [x for x in shields if x]
    return {
        "result": "collision" if any(in_coll) else ("off road" if max(off_run) >= KERB_TICKS else "ok"),
        "vehicles": n, "vehicles_in_collision": int(sum(in_coll)), "vehicle_pairs_hit": len(hits),
        "ego_collision": bool(in_coll[ei]), "ego_other_hit": bool(oi is not None and tuple(sorted((ei, oi))) in hits),
        "off_road_vehicles": int(sum(r >= KERB_TICKS for r in off_run)),
        "behaviour_checked": base_rel is not None,
        "behaviour_kept": bool(same_behaviour(base_rel, new_rel)) if base_rel is not None else None,
        **b2,
        "ego_progress": round(prog[ei], 3), "mean_progress": round(float(np.mean(prog)), 3),
        "speed_vs_recorded": round(sum(drv_speed) / max(sum(rec_speed), 1e-6), 3),
        "ego_speed_vs_recorded": round(drv_speed[ei] / max(rec_speed[ei], 1e-6), 3),
        "stuck_s_per_vehicle": round(float(np.mean(stuck)) / SIM_FPS, 2),
        "shield_brake_s_per_vehicle": round(float(np.mean([x["brake"] for x in sh])) / SIM_FPS, 2) if sh else None,
        "shield_follow_s_per_vehicle": round(float(np.mean([x["follow"] for x in sh])) / SIM_FPS, 2) if sh else None,
        "plans": plans, "model_s": round(t_model, 2),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--drivers", nargs="+", required=True)
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ids", nargs="*")
    p.add_argument("--tag", default="multi")
    p.add_argument("--device", default="cuda")
    p.add_argument("--all-vehicles", action="store_true", help="The driver drives every vehicle, not only the ego.")
    p.add_argument("--any-split", action="store_true", help="Scenarios from train/val too (showcase folders only).")
    p.add_argument("--no-render", action="store_true", help="No rendering (drivers without camera / LiDAR only).")
    args = p.parse_args()
    scens = test_scenarios(args.any_split, args.split)
    rng = np.random.default_rng(0)
    scens = [scens[i] for i in rng.permutation(len(scens))]
    scens = [x for x in scens if not args.ids or x["id"] in args.ids][:args.limit]
    out = os.path.join(IL_DIR, f"closed_loop_{args.tag}.csv")
    done = set()
    if os.path.exists(out):
        d = pd.read_csv(out, dtype={"scenario": str})
        done = set(zip(d.scenario, d.driver))
    models = {}
    for dr in args.drivers:
        b, _ = parse_driver(dr)
        if b not in ("expert", "constvel", "autopilot", "autopilot_stock", "idm") and b not in models:
            run, _, ep = b.partition("@")
            ck = torch.load(os.path.join(RUNS, run, f"ckpt_{ep}.pt" if ep else "model.pt"), weights_only=False)
            m = T.model_from_args(ck["args"]).to(args.device).eval()
            m.load_state_dict(ck["model"])
            models[b] = m
    enc = OnlineEncoders(args.device) if models else None
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
    if args.no_render:
        assert not models, "--no-render: model drivers need the camera"
        st.no_rendering_mode = True
    world.apply_settings(st)
    lib = world.get_blueprint_library()
    global TM
    if any(parse_driver(dr)[0] in AUTOPILOTS for dr in args.drivers):
        TM = client.get_trafficmanager(args.port + 6000)
        TM.set_synchronous_mode(True)
        TM.set_random_device_seed(int(os.environ.get("CL_TM_SEED", 0)))   # run number in repeated runs
    jobs = [(s, dr) for s in scens for dr in args.drivers if (s["id"], dr) not in done]
    print(f"{len(scens)} {args.split} scenarios, {len(done)} episodes done, {len(jobs)} to run", flush=True)
    t_all, streak = time.time(), 0
    try:
        for i, (s, dr) in enumerate(jobs):
            ts = time.time()
            try:
                row = episode(world, lib, store, s, dr, models.get(parse_driver(dr)[0]), enc, drivable, args.device,
                              args.all_vehicles)
                streak = 0
            except Exception as ex:
                streak += 1
                print(f"  [{i + 1}/{len(jobs)}] {s['id']} {dr}: ERROR {ex!r}", flush=True)
                if streak >= MAX_ERROR_STREAK:
                    raise SystemExit(1)
                clear_world(world)
                continue
            row = {"scenario": s["id"], "behaviour": s["behaviour"], "length_s": s["length_s"], "driver": dr, **row,
                   "seconds": round(time.time() - ts, 1)}
            pd.DataFrame([row]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            print(f"  [{i + 1}/{len(jobs)}] {s['id']} {s['behaviour']} {dr}: {row['result']}, "
                  f"behaviour kept {row.get('behaviour_kept')}, {row['seconds']} s", flush=True)
    finally:
        clear_world(world)
        if TM is not None:
            TM.set_synchronous_mode(False)
        world.apply_settings(orig)
    summary(out)
    print(f"saved {out} ({(time.time() - t_all) / 60:.1f} min)")


def summary(out):
    d = pd.read_csv(out, dtype={"scenario": str})
    d = d[d.result != "spawn failed"]
    chk = d[d.behaviour_checked.astype(str) == "True"]
    g = d.groupby("driver").agg(n=("scenario", "size"), any_collision=("result", lambda r: (r == "collision").mean()),
                                ego_collision=("ego_collision", "mean"), ego_other_hit=("ego_other_hit", "mean"),
                                speed_vs_rec=("speed_vs_recorded", "mean"), progress=("mean_progress", "mean"),
                                stuck_s=("stuck_s_per_vehicle", "mean"),
                                brake_s=("shield_brake_s_per_vehicle", "mean"), follow_s=("shield_follow_s_per_vehicle", "mean"))
    g["behaviour_kept"] = chk.groupby("driver").behaviour_kept.apply(lambda x: (x.astype(str) == "True").mean())
    g["checked"] = chk.groupby("driver").size()
    if "behaviour2_checked" in d:   # behaviour check v2 (il/behaviour_v2.py), the one reported since 2026-11-14
        c2 = d[d.behaviour2_checked.astype(str) == "True"]
        g["behaviour2_kept"] = c2.groupby("driver").behaviour2_kept.apply(lambda x: (x.astype(str) == "True").mean())
        g["kept2_no_ego_coll"] = c2.groupby("driver").apply(
            lambda x: ((x.behaviour2_kept.astype(str) == "True") & (x.ego_collision.astype(str) != "True")).mean())
        g["checked2"] = c2.groupby("driver").size()
    pd.set_option("display.width", 220)
    print("\n" + g.round(3).to_string())


if __name__ == "__main__":
    main()
