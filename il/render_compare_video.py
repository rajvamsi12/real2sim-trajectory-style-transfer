#!/usr/bin/env python3
"""
il/render_compare_video.py

Side-by-side review video (ML_DECISIONS.md section 9): left = the real drone footage of the scenario
(DJI_0003-002.MP4, cropped around its vehicles, recorded boxes drawn), right = the same seconds replayed in
CARLA Town10HD with physics (cure E: path spawn heading + velocity tracking), top-down camera turned so its
"up" matches the drone image's "up". Both halves run at 20 fps over the same time span. Only the scenario's
spawned vehicles (ego, other, up to 4 neighbours) are boxed / labelled on both sides.

Needs a GPU-rendered server:  ~/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic &

  python il/render_compare_video.py --pick 10 --out Scenariosssss/IL/review_10.mp4
"""

import argparse
import ast
import contextlib
import io
import math
import os
import queue
import sys

import cv2
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, SCRIPT_DIR)

import carla  # noqa: E402
import scenario_checks as sc  # noqa: E402
import timemethod  # noqa: E402
from overlay_scenario_bboxes import DEFAULT_SCENARIO_FPS  # noqa: E402
from simulate_and_record_scenarios import SIM_FPS, TOWN, DEFAULT_Z  # noqa: E402
from debug.batch_process_scenarios import SETTLE_TICKS, get_camera_intrinsic_matrix, world_to_pixel  # noqa: E402
from replay_scenarios import IL_DIR, timed_route, clear_world, spawn_vehicle, velocity_step  # noqa: E402

VIDEO = os.path.join(BASE_DIR, "DJI_0003-002.MP4")
HALF_W, HALF_H, HEADER_H = 960, 720, 64
M_PER_PX = 0.152            # drone pixel -> metres (measured from the region matrices)
MARGIN_PX = 110
MIN_CROP_W_PX = 520
FOV = 90.0
COLORS = {"ego": (60, 220, 60), "other": (0, 150, 255), "neighbour": (255, 220, 0)}  # BGR


# ==========================================
# DRONE SIDE
# ==========================================
def load_boxes(scens):
    """{scenario id: {track: {frame: corners px (4, 2)}}} from the EKF boxes, shifted by the raw-vs-stabilised
    offset of that row; one pass over the CSV for all scenarios."""
    want = {}
    for s in scens:
        for v in s["vehicles"]:
            want.setdefault(v["track"], []).append((s["id"], s["start"] - 1, s["end"] + 1))
    out = {s["id"]: {} for s in scens}
    for chunk in pd.read_csv(os.path.join(BASE_DIR, "stabilize_ekf_final2.csv"), chunksize=400_000,
                             usecols=["frameID", "trackID", "o_cx_pixel", "o_cy_pixel", "kf_cx_pixel", "kf_cy_pixel",
                                      "kf_obbox_coord_pixel"]):
        chunk = chunk[chunk.trackID.isin(want)]
        for r in chunk.itertuples():
            f, tid = int(r.frameID[6:]), int(r.trackID)
            for sid, f0, f1 in want[tid]:
                if f0 <= f <= f1:
                    c = np.array(ast.literal_eval(r.kf_obbox_coord_pixel), dtype=float)
                    c += [r.o_cx_pixel - r.kf_cx_pixel, r.o_cy_pixel - r.kf_cy_pixel]
                    out[sid].setdefault(tid, {})[f] = c
    return out


def box_at(frames_boxes, a):
    """Corners at fractional annotation frame a (linear between neighbouring frames), or None."""
    lo, hi = math.floor(a), math.ceil(a)
    if lo in frames_boxes and hi in frames_boxes:
        w = a - lo
        return frames_boxes[lo] * (1 - w) + frames_boxes[hi] * w
    near = min(frames_boxes, key=lambda f: abs(f - a), default=None)
    return frames_boxes[near] if near is not None and abs(near - a) <= 1 else None


def drone_crop(scen, boxes):
    pts = np.vstack([c for tb in boxes.values() for f, c in tb.items() if scen["start"] <= f <= scen["end"]])
    x0, y0 = pts.min(0) - MARGIN_PX
    x1, y1 = pts.max(0) + MARGIN_PX
    w = max(x1 - x0, (y1 - y0) * HALF_W / HALF_H, MIN_CROP_W_PX)
    h = w * HALF_H / HALF_W
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return cx - w / 2, cy - h / 2, w, h


def drone_frames(scen, boxes, roles, n):
    """n frames at SIM_FPS starting at the scenario's first annotation frame."""
    cap = cv2.VideoCapture(VIDEO)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    s0 = scen["start"] * src_fps / DEFAULT_SCENARIO_FPS
    want = [int(round(s0 + k * src_fps / SIM_FPS)) for k in range(n)]
    cap.set(cv2.CAP_PROP_POS_FRAMES, want[0])
    x, y, w, h = drone_crop(scen, boxes)
    sx, sy = HALF_W / w, HALF_H / h
    out, cur, frame = [], want[0] - 1, None
    for k, target in enumerate(want):
        while cur < target:
            ok, frame = cap.read()
            cur += 1
            if not ok:
                break
        img = np.zeros((HALF_H, HALF_W, 3), np.uint8)
        if frame is not None:
            M = np.float32([[sx, 0, -x * sx], [0, sy, -y * sy]])
            img = cv2.warpAffine(frame, M, (HALF_W, HALF_H), borderValue=(30, 30, 30))
        a = scen["start"] + k * DEFAULT_SCENARIO_FPS / SIM_FPS
        for tid, role in roles.items():
            c = box_at(boxes.get(tid, {}), a)
            if c is None:
                continue
            p = ((c - [x, y]) * [sx, sy]).astype(np.int32)
            cv2.polylines(img, [p], True, COLORS[role], 2, cv2.LINE_AA)
            cv2.putText(img, role, tuple(p.min(0) - [0, 6]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[role], 1, cv2.LINE_AA)
        out.append(img)
    cap.release()
    return out, (x + w / 2, y + h / 2), w


# ==========================================
# CARLA SIDE
# ==========================================
def camera_yaw(store, scen):
    """Yaw of a top-down camera whose image 'up' is the drone image 'up', from the region matrix at the
    ego's first position. Also returns whether the CARLA view is mirrored relative to the drone image."""
    row = store.tracks[scen["ego"]].loc[scen["vehicles"][0]["frames"][0]]
    pt = (row.kf_cx_pixel, -row.kf_cy_pixel)
    region = timemethod.classify_point(pt)
    if region not in timemethod.transformation_matrices:
        region = timemethod.get_closest_region(pt)
    M = timemethod.transformation_matrices[region]
    up = (M[0][1], M[1][1])          # image up = matrix input (0, +1) in (u, -v)
    right = (M[0][0], M[1][0])
    yaw = math.atan2(up[1], up[0])
    mirrored = (-math.sin(yaw) * right[0] + math.cos(yaw) * right[1]) < 0
    return math.degrees(yaw), mirrored


def carla_frames(world, lib, store, scen, n, crop_w_px, roles):
    clear_world(world)
    sims, rl = [], []
    for v in scen["vehicles"]:
        sv = spawn_vehicle(world, lib, store, v, "path")
        if sv is None:
            for o in sims:
                o.destroy()
            return None, "spawn failed"
        sv.set_timed_route(*timed_route(v["frames"], v["xy"], scen["start"]), z=DEFAULT_Z)
        sims.append(sv)
        rl.append(v["role"])
    for sv in sims:
        sv.hold_in_gear()
    for _ in range(SETTLE_TICKS):
        world.tick()
    for sv in sims:
        sv.vehicle.apply_control(carla.VehicleControl(hand_brake=False, manual_gear_shift=False))

    xy = np.vstack([v["xy"] for v in scen["vehicles"]])
    cx, cy = xy.mean(0)
    height = max(20.0, crop_w_px * M_PER_PX / 2 / math.tan(math.radians(FOV / 2)))
    yaw, _ = camera_yaw(store, scen)
    bp = lib.find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(HALF_W))
    bp.set_attribute("image_size_y", str(HALF_H))
    bp.set_attribute("fov", str(FOV))
    cam_tf = carla.Transform(carla.Location(x=float(cx), y=float(cy), z=float(height)), carla.Rotation(pitch=-90.0, yaw=yaw))
    cam = world.spawn_actor(bp, cam_tf)
    q = queue.Queue()
    cam.listen(q.put)
    world.tick()
    while not q.empty():
        q.get()
    K = get_camera_intrinsic_matrix(HALF_W, HALF_H, FOV)
    out, clock, dt, hit = [], 0.0, 1.0 / SIM_FPS, False
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(n):
            for sv in sims:
                velocity_step(sv, clock, dt)
            world.tick()
            clock += dt
            img = q.get(timeout=5.0)
            frame = np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4)[:, :, :3].copy()
            for sv, r in zip(sims, rl):
                p = world_to_pixel(sv.vehicle.get_location(), cam_tf, K)
                if p and 0 <= p[0] < HALF_W and 0 <= p[1] < HALF_H:
                    crashed = bool(sv.collision_events)
                    hit |= crashed
                    col = (0, 0, 255) if crashed else COLORS[r]
                    cv2.circle(frame, p, 6, col, -1)
                    cv2.putText(frame, r + (" CRASH" if crashed else ""), (p[0] + 8, p[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            out.append(frame)
    cam.stop()
    cam.destroy()
    for sv in sims:
        sv.destroy()
    world.tick()
    return out, ("collision during render" if hit else "ok")


# ==========================================
# MAIN
# ==========================================
def header(scen, k, n, note):
    h = np.full((HEADER_H, 2 * HALF_W, 3), 25, np.uint8)
    other = "" if scen["other"] is None else f" / other {scen['other']}"
    t1 = (f"#{scen['id']}  {scen['behaviour']}  ({scen['length_s']} s)   ego {scen['ego']}{other}   "
          f"frames {scen['start']}-{scen['end']}   t = {k / SIM_FPS:4.1f} s")
    cv2.putText(h, t1, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(h, "LEFT: real drone (Gwalior)", (14, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(h, "RIGHT: CARLA Town10HD, physics on (mirrored: left- to right-hand traffic)" + (f"   [{note}]" if note != "ok" else ""),
                (HALF_W + 14, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    for i, (r, c) in enumerate(COLORS.items()):
        cv2.putText(h, r, (2 * HALF_W - 330 + i * 105, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
    return h


def pick(n, seed=0):
    rep = pd.read_csv(os.path.join(IL_DIR, "replay_cureE.csv"), dtype={"id": str})
    ok = rep[(rep.result == "collision-free") & (rep.min_progress_ego_other >= 0.95)]
    want = [("Cut In", 1), ("Cut Out", 1), ("U-turn", 1), ("Overtaking Left", 2), ("Overtaking Right", 2),
            ("Vehicle Following", 2)]
    ids = []
    for b, k in want:
        g = ok[(ok.behaviour == b) & (ok.length_s == 8.0)]
        g = g if len(g) else ok[ok.behaviour == b]
        ids += list(g.sample(min(k, len(g)), random_state=seed).id)
    rest = ok[(ok.length_s < 8.0) & ~ok.id.isin(ids)]
    ids += list(rest.sample(max(0, n - len(ids)), random_state=seed).id)
    return ids[:n]


def pick_showcase(n=500):
    """ML_DECISIONS.md V1/V2: one scenario per distinct vehicle pair, collision-free with cure E; every Cut In /
    Cut Out / U-turn pair, the rest split evenly over Overtaking Left / Right and Vehicle Following. Within a pair
    prefer an 8 s window, then the best ego/other progress."""
    rep = pd.read_csv(os.path.join(IL_DIR, "replay_cureE.csv"), dtype={"id": str})
    ok = rep[(rep.result == "collision-free") & (rep.min_progress_ego_other >= 0.85)].copy()
    ok["pair"] = ok.behaviour + "|" + ok.ego.astype(str) + "|" + ok.other.astype(str)
    ok = ok.sort_values(["length_s", "min_progress_ego_other"], ascending=False).drop_duplicates("pair")
    ids = list(ok[ok.behaviour.isin(["Cut In", "Cut Out", "U-turn"])].id)
    rest = ["Overtaking Left", "Overtaking Right", "Vehicle Following"]
    left = n - len(ids)
    for i, b in enumerate(rest):
        k = left // len(rest) + (1 if i < left % len(rest) else 0)
        g = ok[ok.behaviour == b]
        ids += list(g.sample(min(k, len(g)), random_state=0).id)
    if len(ids) < n:  # a behaviour ran short: top up from the others
        extra = ok[ok.behaviour.isin(rest) & ~ok.id.isin(ids)]
        ids += list(extra.sample(min(n - len(ids), len(extra)), random_state=0).id)
    order = ["Cut In", "Cut Out", "U-turn", "Overtaking Left", "Overtaking Right", "Vehicle Following"]
    return list(ok.set_index("id").loc[ids].assign(o=lambda d: d.behaviour.map(order.index))
                .sort_values(["o", "length_s"], ascending=[True, False]).index)


def join_clips(clips_dir, out):
    files = sorted(f for f in os.listdir(clips_dir) if f.endswith(".mp4"))
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), float(SIM_FPS), (2 * HALF_W, HALF_H + HEADER_H))
    for f in files:
        cap = cv2.VideoCapture(os.path.join(clips_dir, f))
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            writer.write(fr)
        cap.release()
    writer.release()
    print(f"joined {len(files)} clips -> {out} ({os.path.getsize(out) / 1e9:.2f} GB)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pick", type=int, default=10)
    p.add_argument("--showcase", type=int, help="Pick this many with the V1/V2 rule (e.g. 500).")
    p.add_argument("--ids", nargs="*")
    p.add_argument("--out", default=os.path.join(IL_DIR, "review_10.mp4"))
    p.add_argument("--clips-dir", help="Write one mp4 per scenario here (resumable: existing clips are skipped).")
    p.add_argument("--join", action="store_true", help="Only join the clips in --clips-dir into --out.")
    args = p.parse_args()
    if args.join:
        join_clips(args.clips_dir, args.out)
        return
    import json
    import time
    ids = args.ids or (pick_showcase(args.showcase) if args.showcase else pick(args.pick))
    if args.clips_dir:
        os.makedirs(args.clips_dir, exist_ok=True)
        pd.DataFrame({"order": range(len(ids)), "id": ids}).to_csv(os.path.join(args.clips_dir, "selection.csv"), index=False)
        done = {f.split("_")[2].split(".")[0] for f in os.listdir(args.clips_dir) if f.startswith("clip_")}
        order = {i: k for k, i in enumerate(ids)}
        ids = [i for i in ids if i not in done]
        print(f"{len(order)} selected, {len(done)} already rendered, {len(ids)} to render", flush=True)
    scens = {s["id"]: s for s in json.load(open(os.path.join(IL_DIR, "scenarios_8s.json")))}
    sel = [scens[i] for i in ids]
    print("scenarios:", [(s["id"], s["behaviour"], s["length_s"]) for s in sel], flush=True)
    store = sc.TrackStore(track_ids={v["track"] for s in sel for v in s["vehicles"]})

    client = carla.Client("localhost", 2000)
    client.set_timeout(120.0)
    world = client.get_world()
    if TOWN not in world.get_map().name:
        world = client.load_world(TOWN)
    orig = world.get_settings()
    st = world.get_settings()
    st.synchronous_mode, st.fixed_delta_seconds = True, 1.0 / SIM_FPS
    world.apply_settings(st)
    lib = world.get_blueprint_library()

    all_boxes = load_boxes(sel)
    fourcc, size = cv2.VideoWriter_fourcc(*"mp4v"), (2 * HALF_W, HALF_H + HEADER_H)
    writer = None if args.clips_dir else cv2.VideoWriter(args.out, fourcc, float(SIM_FPS), size)
    log_path = os.path.join(args.clips_dir, "log.csv") if args.clips_dir else args.out.replace(".mp4", ".csv")
    log, t0 = [], time.time()
    try:
        for i, s in enumerate(sel):
            n = int(round(s["length_s"] * SIM_FPS))
            roles = {v["track"]: v["role"] for v in s["vehicles"]}
            boxes = all_boxes[s["id"]]
            left, _, crop_w = drone_frames(s, boxes, roles, n)
            right, note = carla_frames(world, lib, store, s, n, crop_w, roles)
            if right is None:
                print(f"  {s['id']}: {note}, skipped", flush=True)
                continue
            w = writer
            if args.clips_dir:
                path = os.path.join(args.clips_dir, f"clip_{order[s['id']]:04d}_{s['id']}.tmp.mp4")
                w = cv2.VideoWriter(path, fourcc, float(SIM_FPS), size)
            for k in range(n):
                w.write(np.vstack([header(s, k, n, note), np.hstack([left[k], right[k]])]))
            if args.clips_dir:
                w.release()
                os.replace(path, path.replace(".tmp.mp4", ".mp4"))
            yaw, mirrored = camera_yaw(store, s)
            row = {"id": s["id"], "behaviour": s["behaviour"], "length_s": s["length_s"], "carla": note,
                   "camera_yaw": round(yaw, 1), "carla_view_mirrored_vs_drone": mirrored}
            log.append(row)
            if args.clips_dir:
                pd.DataFrame([row]).to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)
            el = time.time() - t0
            print(f"  [{i + 1}/{len(sel)}] {s['id']} {s['behaviour']}: {note}   {el / (i + 1):.0f} s/clip, "
                  f"eta {el / (i + 1) * (len(sel) - i - 1) / 3600:.1f} h", flush=True)
    finally:
        if writer:
            writer.release()
        clear_world(world)
        world.apply_settings(orig)
    if not args.clips_dir:
        pd.DataFrame(log).to_csv(log_path, index=False)
        print(f"saved {args.out} ({os.path.getsize(args.out) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
