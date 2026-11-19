"""
il/proof_lhd.py

Evidence that the old drone -> CARLA conversion (timemethod.py, "drone pixel to carla coordinates.txt") puts Gwalior's
left-hand traffic against CARLA Town10HD's right-hand lanes (10 Nov 2026).

  select  score every old base scenario (Scenariosssss/IL/scenarios_8s.json): per moving vehicle step (outside junctions,
          within 2.5 m of a driving lane) is the travel direction against the nearest CARLA lane? Writes
          Scenariosssss/IL/proof_lhd/scores.csv and prints the 10 worst (distinct pairs, spread over behaviours).
  render  per scenario id: 1_real_drone.mp4, 2_carla_converted.mp4 (cure E physics replay, camera turned like the drone
          view), 3_trajectory_drone.png (real paths on the drone frame), 4_trajectory_carla.png (converted paths on the
          empty CARLA map with CARLA's lane directions as white arrows), info.txt

  python il/proof_lhd.py select
  python il/proof_lhd.py render --ids 00012 00100 ... --out Scenariosssss/IL/proof_lhd
"""

import argparse
import json
import math
import os
import queue
import sys

import cv2
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)

import carla  # noqa: E402
import render_compare_video as R  # noqa: E402
import scenario_checks as sc  # noqa: E402
from debug.batch_process_scenarios import get_camera_intrinsic_matrix, world_to_pixel  # noqa: E402
from simulate_and_record_scenarios import SIM_FPS, TOWN  # noqa: E402

IL = R.IL_DIR
OUT = os.path.join(IL, "proof_lhd")
ROLE_COL = R.COLORS   # BGR: ego green, other orange, neighbour cyan


def connect(port=2000):
    client = carla.Client("localhost", port)
    client.set_timeout(120.0)
    world = client.get_world()
    if TOWN not in world.get_map().name:
        world = client.load_world(TOWN)
    return world


def lane_steps(m, xy, step=3):
    """[(i, |heading - lane yaw| deg)] for moving steps near a driving lane, outside junctions."""
    out = []
    for i in range(0, len(xy) - step, step):
        d = xy[i + step] - xy[i]
        if np.hypot(*d) < 1.0:
            continue
        wp = m.get_waypoint(carla.Location(x=float(xy[i, 0]), y=float(xy[i, 1]), z=0.5), project_to_road=True,
                            lane_type=carla.LaneType.Driving)
        if wp is None or wp.is_junction:
            continue
        if math.hypot(wp.transform.location.x - xy[i, 0], wp.transform.location.y - xy[i, 1]) > 2.5:
            continue
        out.append((i, abs((math.degrees(math.atan2(d[1], d[0])) - wp.transform.rotation.yaw + 180) % 360 - 180)))
    return out


def select(port):
    m = connect(port).get_map()
    scens = json.load(open(os.path.join(IL, "scenarios_8s.json")))
    rep = pd.read_csv(os.path.join(IL, "replay_cureE.csv"), dtype={"id": str}).drop_duplicates("id", keep="last")
    ok = set(rep[rep.result == "collision-free"].id)
    rows = []
    for s in scens:
        n = against = ego_n = ego_against = 0
        moved = 0.0
        for v in s["vehicles"]:
            xy = np.array(v["xy"], float)
            st = lane_steps(m, xy)
            a = sum(d > 135 for _, d in st)
            n, against = n + len(st), against + a
            if v["role"] == "ego":
                ego_n, ego_against = len(st), a
                moved = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
        rows.append({"id": s["id"], "behaviour": s["behaviour"], "pair": f"{s['ego']}_{s['other']}", "length_s": s["length_s"],
                     "status": s["status"], "collision_free": s["id"] in ok, "steps": n,
                     "against_pct": round(100 * against / max(n, 1), 1), "ego_steps": ego_n,
                     "ego_against_pct": round(100 * ego_against / max(ego_n, 1), 1), "ego_moved_m": round(moved, 1)})
    d = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    d.to_csv(os.path.join(OUT, "scores.csv"), index=False)
    # worst = every checked step of every vehicle against the lane, the most such steps, clean 8 s, replays collision-free
    c = d[(d.ego_against_pct == 100) & (d.against_pct >= 99) & d.collision_free & (d.length_s == 8.0)]
    c = c.assign(score=c.steps * c.against_pct / 100).sort_values("score", ascending=False).drop_duplicates("pair")
    quota = {"Overtaking Left": 2, "Overtaking Right": 2, "Vehicle Following": 2, "U-turn": 2, "Cut Out": 1, "Cut In": 1}
    pick = []
    for b, q in quota.items():
        pick += list(c[c.behaviour == b].head(q).id)
    pick += list(c[~c.id.isin(pick)].head(10 - len(pick)).id)
    print(f"{len(d)} scenarios scored; {len(c)} with every checked step against the lane")
    print(d.against_pct.describe().round(1).to_string())
    print(c[c.id.isin(pick)].to_string(index=False))
    print("PICK", " ".join(pick))


def arrow_path(img, pts, col, thick, every=6):
    pts = [p for p in pts if p is not None]
    if len(pts) < 2:
        return
    cv2.polylines(img, [np.array(pts, np.int32)], False, col, thick, cv2.LINE_AA)
    for i in range(every, len(pts), every):
        cv2.arrowedLine(img, pts[i - 2], pts[i], col, thick + 1, cv2.LINE_AA, tipLength=1.2)
    cv2.circle(img, pts[0], thick + 4, col, -1, cv2.LINE_AA)
    cv2.arrowedLine(img, pts[-3] if len(pts) > 2 else pts[0], pts[-1], col, thick + 1, cv2.LINE_AA, tipLength=0.9)


def bar(img, lines):
    cv2.rectangle(img, (0, 0), (img.shape[1], 24 * len(lines) + 10), (25, 25, 25), -1)
    for k, (t, col) in enumerate(lines):
        cv2.putText(img, t, (12, 24 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA)


def carla_traj_image(world, s, crop_w, m, steps_by_role):
    """Empty CARLA map from the same camera as the converted video, CARLA lane directions (white arrows) and the converted
    paths with travel-direction arrows."""
    R.clear_world(world)
    xy = np.vstack([v["xy"] for v in s["vehicles"]])
    cx, cy = xy.mean(0)
    height = max(20.0, crop_w * R.M_PER_PX / 2 / math.tan(math.radians(R.FOV / 2)))
    yaw, _ = R.camera_yaw(STORE, s)
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(R.HALF_W))
    bp.set_attribute("image_size_y", str(R.HALF_H))
    bp.set_attribute("fov", str(R.FOV))
    tf = carla.Transform(carla.Location(x=float(cx), y=float(cy), z=float(height)), carla.Rotation(pitch=-90.0, yaw=yaw))
    cam = world.spawn_actor(bp, tf)
    q = queue.Queue()
    cam.listen(q.put)
    for _ in range(3):
        world.tick()
    im = q.get(timeout=10.0)
    while not q.empty():
        im = q.get()
    cam.stop()
    cam.destroy()
    img = np.frombuffer(im.raw_data, np.uint8).reshape(im.height, im.width, 4)[:, :, :3].copy()
    img = cv2.addWeighted(img, 0.7, np.zeros_like(img), 0.3, 0)
    K = get_camera_intrinsic_matrix(R.HALF_W, R.HALF_H, R.FOV)
    half = height * math.tan(math.radians(R.FOV / 2)) * 1.5
    for wp in m.generate_waypoints(4.0):
        L = wp.transform.location
        if wp.lane_type != carla.LaneType.Driving or wp.is_junction or abs(L.x - cx) > half or abs(L.y - cy) > half:
            continue
        f = wp.transform.get_forward_vector()
        a = world_to_pixel(L, tf, K)
        b = world_to_pixel(carla.Location(x=L.x + 2.2 * f.x, y=L.y + 2.2 * f.y, z=L.z), tf, K)
        if a and b:
            cv2.arrowedLine(img, a, b, (235, 235, 235), 2, cv2.LINE_AA, tipLength=0.5)
    for v in sorted(s["vehicles"], key=lambda v: v["role"] == "ego"):
        pts = [world_to_pixel(carla.Location(x=float(x), y=float(y), z=0.5), tf, K) for x, y in v["xy"]]
        arrow_path(img, pts, ROLE_COL[v["role"]], 4 if v["role"] == "ego" else 2, every=4)
    a = sum(steps_by_role.values(), [])
    bar(img, [(f"CARLA, OLD conversion   #{s['id']} {s['behaviour']}", (240, 240, 240)),
              (f"white arrows = CARLA lane direction;  coloured = converted paths (dot = start, arrow = travel)", (200, 200, 200)),
              (f"moving steps AGAINST the CARLA lane: {sum(d > 135 for d in a)}/{len(a)}   (ego "
               f"{sum(d > 135 for d in steps_by_role['ego'])}/{len(steps_by_role['ego'])})", (110, 110, 255))])
    return img


def drone_traj_image(frames, s, boxes, crop):
    x0, y0, w, h = crop
    sx, sy = R.HALF_W / w, R.HALF_H / h
    img = frames[len(frames) // 2].copy()
    img = cv2.addWeighted(img, 0.75, np.zeros_like(img), 0.25, 0)
    for v in sorted(s["vehicles"], key=lambda v: v["role"] == "ego"):
        tb = boxes.get(v["track"], {})
        pts = [tuple(((tb[f].mean(0) - [x0, y0]) * [sx, sy]).astype(int)) for f in sorted(tb) if s["start"] <= f <= s["end"]]
        arrow_path(img, pts, ROLE_COL[v["role"]], 4 if v["role"] == "ego" else 2, every=4)
    bar(img, [(f"REAL DRONE (Gwalior, left-hand traffic)   #{s['id']} {s['behaviour']}", (240, 240, 240)),
              ("real paths from the annotations (dot = start, arrow = travel); green ego, orange other, cyan neighbours",
               (200, 200, 200))])
    return img


def render(ids, out, port):
    global STORE
    world = connect(port)
    m = world.get_map()
    orig = world.get_settings()
    st = world.get_settings()
    st.synchronous_mode, st.fixed_delta_seconds = True, 1.0 / SIM_FPS
    world.apply_settings(st)
    lib = world.get_blueprint_library()
    scens = {s["id"]: s for s in json.load(open(os.path.join(IL, "scenarios_8s.json")))}
    sel = [scens[i] for i in ids]
    STORE = sc.TrackStore(track_ids={v["track"] for s in sel for v in s["vehicles"]})
    all_boxes = R.load_boxes(sel)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    try:
        for k, s in enumerate(sel, 1):
            d = os.path.join(out, f"{k:02d}_{s['id']}_{s['behaviour'].replace(' ', '_')}")
            os.makedirs(d, exist_ok=True)
            n = int(round(s["length_s"] * SIM_FPS))
            roles = {v["track"]: v["role"] for v in s["vehicles"]}
            boxes = all_boxes[s["id"]]
            left, _, crop_w = R.drone_frames(s, boxes, roles, n)
            crop = R.drone_crop(s, boxes)
            right, note = R.carla_frames(world, lib, STORE, s, n, crop_w, roles)
            w = cv2.VideoWriter(os.path.join(d, "1_real_drone.mp4"), fourcc, float(SIM_FPS), (R.HALF_W, R.HALF_H))
            for i, f in enumerate(left):
                f = f.copy()
                bar(f, [(f"REAL DRONE (Gwalior)  #{s['id']} {s['behaviour']}  t {i / SIM_FPS:4.1f} s", (240, 240, 240))])
                w.write(f)
            w.release()
            w = cv2.VideoWriter(os.path.join(d, "2_carla_converted.mp4"), fourcc, float(SIM_FPS), (R.HALF_W, R.HALF_H))
            for i, f in enumerate(right or []):
                f = f.copy()
                bar(f, [(f"CARLA, OLD conversion  #{s['id']}  t {i / SIM_FPS:4.1f} s  [{note}]", (240, 240, 240))])
                w.write(f)
            w.release()
            cv2.imwrite(os.path.join(d, "3_trajectory_drone.png"), drone_traj_image(left, s, boxes, crop))
            steps = {"ego": []}
            for v in s["vehicles"]:
                steps.setdefault(v["role"], [])
                steps[v["role"]] += [a for _, a in lane_steps(m, np.array(v["xy"], float))]
            cv2.imwrite(os.path.join(d, "4_trajectory_carla.png"), carla_traj_image(world, s, crop_w, m, steps))
            a = sum(steps.values(), [])
            open(os.path.join(d, "info.txt"), "w").write(
                f"Scenario {s['id']}: {s['behaviour']}, ego track {s['ego']}, other {s['other']}, drone frames "
                f"{s['start']}-{s['end']}, {s['length_s']} s, status {s['status']}\n"
                f"Moving vehicle steps (outside junctions, within 2.5 m of a CARLA driving lane) against the CARLA lane "
                f"direction: {sum(x > 135 for x in a)}/{len(a)}; ego {sum(x > 135 for x in steps['ego'])}/{len(steps['ego'])}\n"
                f"CARLA replay (cure E, physics on): {note}\n"
                "Conversion: OLD 6-region matrices (timemethod.py / 'drone pixel to carla coordinates.txt').\n"
                "1 real drone video, 2 the same seconds in CARLA after conversion, 3 real paths on the drone frame,\n"
                "4 converted paths on the CARLA map with CARLA's lane directions (white arrows).\n")
            print(f"  [{k}/{len(sel)}] {s['id']} {s['behaviour']}: {note}, against {sum(x > 135 for x in a)}/{len(a)}", flush=True)
    finally:
        R.clear_world(world)
        world.apply_settings(orig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["select", "render"])
    p.add_argument("--ids", nargs="*")
    p.add_argument("--out", default=OUT)
    p.add_argument("--port", type=int, default=2000)
    a = p.parse_args()
    if a.what == "select":
        select(a.port)
    else:
        render(a.ids, a.out, a.port)
