#!/usr/bin/env python3
"""
il/test_pointpillars_carla.py

Checks the pretrained PointPillars encoder (il/pointpillars.py) on real CARLA LiDAR sweeps:
spawns an ego with a nuScenes-like 32-channel LiDAR plus parked vehicles around it in Town10HD,
takes one sweep and a 10-sweep accumulation (as in nuScenes training), runs the nuScenes detection head and compares detected cars with the CARLA
ground truth. If the weights loaded correctly and transfer to CARLA LiDAR, most nearby cars are found.
Also times the encoder on GPU and CPU.

  ~/CARLA_0.9.15/CarlaUE4.sh -nullrhi -nosound -carla-rpc-port=2000 &
  python il/test_pointpillars_carla.py
"""

import os
import random
import sys
import time

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import pointpillars as pp  # noqa: E402

WEIGHTS = os.path.join(SCRIPT_DIR, "weights", "pointpillars_nus_secfpn_f19d00a3.pth")
LIDAR_Z = 1.84                  # nuScenes LIDAR_TOP height above ground
N_SPAWNS = 3                    # ego positions tried
MATCH_M = 2.0
EVAL_RANGE_M = 40.0
VEHICLE_CLASSES = {0, 1, 3, 6}  # car, truck, bus, motorcycle
SWEEPS = 10                     # nuScenes training used 10 accumulated sweeps (0.5 s at 20 Hz)


def lidar_bp(lib):
    bp = lib.find("sensor.lidar.ray_cast")
    for k, v in {"channels": "32", "range": "70", "rotation_frequency": "20", "points_per_second": "700000",
                 "upper_fov": "10", "lower_fov": "-30", "dropoff_general_rate": "0.0",
                 "dropoff_intensity_limit": "1.0", "dropoff_zero_intensity": "0.0"}.items():
        bp.set_attribute(k, v)
    return bp


def sweep_points(measurements, current):
    """Points of these sweeps in the current sensor frame, right-handed, 4th value = time lag (s):
    the checkpoint's 4 input values are x, y, z and the nuScenes multi-sweep time lag, not intensity."""
    inv = np.array(current.transform.get_inverse_matrix())
    out = []
    for m in measurements:
        p = np.frombuffer(m.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
        if m is not current:
            p = (inv @ np.array(m.transform.get_matrix()) @ np.c_[p, np.ones(len(p))].T).T[:, :3]
        lag = np.full((len(p), 1), current.timestamp - m.timestamp)
        out.append(np.c_[p[:, 0], -p[:, 1], p[:, 2], lag])  # CARLA (left-handed) -> right-handed
    return np.concatenate(out).astype(np.float32)


def to_sensor(world_loc, sensor_tf):
    """World point -> right-handed sensor frame (x forward, y left, z up)."""
    m = np.array(sensor_tf.get_inverse_matrix())
    p = m @ np.array([world_loc.x, world_loc.y, world_loc.z, 1.0])
    return np.array([p[0], -p[1], p[2]])


def main():
    import carla
    client = carla.Client("localhost", 2000)
    client.set_timeout(60.0)
    world = client.get_world()
    if "Town10HD" not in world.get_map().name:
        world = client.load_world("Town10HD")
    orig = world.get_settings()
    s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds = True, 0.05
    world.apply_settings(s)
    lib = world.get_blueprint_library()
    random.seed(0)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = pp.load_encoder(WEIGHTS, with_head=True, device=dev)
    print(f"weights loaded (strict), device {dev}")

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    results = []
    for trial in range(N_SPAWNS):
        actors = []
        try:
            ego_tf = spawn_points[trial]
            ego = world.spawn_actor(lib.find("vehicle.nissan.micra"), ego_tf)
            actors.append(ego)
            near = [p for p in spawn_points if 4.0 < p.location.distance(ego_tf.location) < EVAL_RANGE_M]
            cars = [b for b in lib.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
            for p in near[:20]:
                a = world.try_spawn_actor(random.choice(cars), p)
                if a:
                    actors.append(a)
            sensor = world.spawn_actor(lidar_bp(lib), carla.Transform(carla.Location(z=LIDAR_Z)), attach_to=ego)
            actors.append(sensor)
            frames = []
            sensor.listen(frames.append)
            for _ in range(SWEEPS + 5):
                world.tick()
            m = frames[-1]
            raw = sweep_points(frames[-1:], m)
            gt = [to_sensor(a.get_transform().location, m.transform) for a in actors[1:-1]]
            gt = [g for g in gt if np.hypot(g[0], g[1]) < EVAL_RANGE_M]
            for mode in ("1 sweep", "10 sweeps"):
                pts = torch.from_numpy(raw if mode == "1 sweep" else sweep_points(frames[-SWEEPS:], m)).to(dev)
                scale = mode
                with torch.no_grad():
                    boxes, scores, labels = model.detect(pts, score_thr=0.3)
                keep = torch.tensor([int(l) in VEHICLE_CLASSES for l in labels.tolist()], dtype=torch.bool)
                det = boxes[keep][:, :2].cpu().numpy() if keep.any() else np.zeros((0, 2))
                found = sum(1 for g in gt if len(det) and np.min(np.hypot(*(det - g[:2]).T)) < MATCH_M)
                false = sum(1 for d in det if not gt or np.min([np.hypot(*(d - g[:2])) for g in gt]) > MATCH_M)
                results.append((trial, scale, len(pts), len(gt), found, len(det), false))
                print(f"spawn {trial} {scale}: {len(pts):,} points, {len(gt)} cars within {EVAL_RANGE_M:.0f} m, "
                      f"found {found}, vehicle detections (score > 0.3, no NMS) {len(det)}, not near a car {false}")
            if trial == 0:
                pts = torch.from_numpy(raw).to(dev)
                for d in ([dev] + (["cpu"] if dev != "cpu" else [])):
                    mm = model.to(d)
                    p = pts.to(d)
                    with torch.no_grad():
                        mm.bev(p)
                        if d == "cuda":
                            torch.cuda.synchronize()
                        t0 = time.time()
                        for _ in range(5):
                            f = mm.bev(p)
                        if d == "cuda":
                            torch.cuda.synchronize()
                    print(f"encoder on {d}: {(time.time() - t0) / 5 * 1000:.0f} ms per sweep, output {tuple(f.shape)}")
                model.to(dev)
                if dev == "cuda":
                    print(f"GPU memory peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
            sensor.stop()
        finally:
            for a in reversed(actors):
                a.destroy()
            world.tick()
    world.apply_settings(orig)
    for mode in ("1 sweep", "10 sweeps"):
        sel = [r for r in results if r[1] == mode]
        print(f"TOTAL {mode}: found {sum(r[4] for r in sel)}/{sum(r[3] for r in sel)} cars, "
              f"{sum(r[6] for r in sel)} detections not near a car")


if __name__ == "__main__":
    main()
