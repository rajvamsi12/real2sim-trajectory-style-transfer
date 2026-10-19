#!/usr/bin/env python3
"""
debug/render_scenario_evidence.py

Builds one scenario folder with everything needed to inspect a replay problem:

  <folder>/
    *_summary.txt, raw_matrix_trajectories.json, time_variant_trajectories.json,
    spawn_positions_raw.json, spawn_positions_retracted.json      Stages 1-3
    real_world_trajectory.png      both tracks on the drone background map
    real_drone_video.mp4           DJI footage with the EKF boxes (overlay_scenario_bboxes.py)
    closest_moment_real.png        the drone frame where the recorded centres are closest, zoomed
    closest_moment_zoom.png        raw vs. boxed drone crops just before, at and after that frame
    frame<N>_all_tracks.png        every tracked box around the pair at that frame (crowding, duplicates)
    centre_distance.png            centre distance over time vs. what real-size vehicles need
    sim_metrics_<mode>.json

  CARLA outputs, with --camera (needs a GPU-rendered server, CarlaUE4.sh -RenderOffScreen):
    trajectory_comparison.png                matrix (red) vs time-method (cyan) trajectories on the CARLA map
    trajectory_video_expert_replay.mp4       bird's-eye camera, positions set exactly from the data
    trajectory_video_without_avoidance.mp4   bird's-eye camera, PID-driven with physics
  without --camera (works headless with -nullrhi, no GPU), drawn from the simulated actor states:
    carla_sim_expert.mp4, carla_sim_pid.mp4

  python debug/render_scenario_evidence.py --category "Overtaking Left" --ego 565 --other 1342 --start 853 --camera
"""

import argparse
import ast
import math
import os
import subprocess
import sys

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, BASE_DIR)

import carla  # noqa: E402
import pandas as pd  # noqa: E402
import scenario_checks as sc  # noqa: E402
from overlay_scenario_bboxes import DEFAULT_SCENARIO_FPS, annotation_frame_to_source_frame  # noqa: E402
from simulate_and_record_scenarios import (  # noqa: E402
    AnnotationParser, CapturingVehicle, STABILIZE_EKF_CSV, SIM_FPS, TOWN, DEFAULT_Z, REAL_WORLD_FPS,
)
from debug.batch_process_scenarios import (  # noqa: E402
    run_stage_1_matrix, run_stage_2_time_method, run_stage_3_spawns, build_timed_route, SETTLE_TICKS, actor_box,
    render_carla_simulation_video, render_trajectory_comparison_map,
)
from generate_5_same_direction_micra_scenarios import render_real_world_drone_trajectory_image  # noqa: E402

SCENARIOS_DIR = os.path.join(BASE_DIR, "Scenariosssss")
VIDEO = os.path.join(BASE_DIR, "DJI_0003-002.MP4")
CSV_BY_CATEGORY = {
    "Overtaking Left": "Overtaking Left/overtaking.csv", "Overtaking Right": "Overtaking Left/overtaking.csv",
    "Cut In": "Cut In/cut_in_out.csv", "Cut Out": "Cut In/cut_in_out.csv",
}
# Typical real vehicle footprints (L x W, m) used only for the "what real vehicles need" line
TYPICAL = {0: (2.6, 1.4), 1: (3.6, 1.65), 2: (1.9, 0.75), 3: (9.0, 2.5), 5: (6.0, 2.3)}
LABEL_NAMES = {0: "3-wheeler", 1: "car", 2: "2-wheeler", 3: "bus", 5: "truck"}


def find_row(category, ego, other, start):
    df = pd.read_csv(os.path.join(SCENARIOS_DIR, CSV_BY_CATEGORY[category]))
    start_col = df.start_frame.str.split("_").str[-1].astype(int)
    row = df[(df.trackID == ego) & (df.other_id == other) & (start_col == start)].iloc[0]
    return int(str(row.end_frame).split("_")[-1]), row


def closest_moment(store, ego, other, start, end):
    """Frame where the recorded centres are closest, with distance, gap with CSV-like box sizes etc."""
    we, wo = store.window(ego, start, end), store.window(other, start, end)
    frames = sorted(set(we.index) & set(wo.index))
    pe = we.loc[frames, ["kf_cx_meter", "kf_cy_meter"]].to_numpy()
    po = wo.loc[frames, ["kf_cx_meter", "kf_cy_meter"]].to_numpy()
    dist = np.linalg.norm(pe - po, axis=1)
    ye, yo = sc.path_yaws(pe), sc.path_yaws(po)
    le, lo = store.label(ego), store.label(other)
    typ_gap = [sc.box_gap(sc.box_corners(pe[i], ye[i], *TYPICAL[le]), sc.box_corners(po[i], yo[i], *TYPICAL[lo]))
               for i in range(len(frames))]
    return frames, dist, np.array(typ_gap), we, wo


def render_closest_still(out_path, frame, we, wo, ego, other, dist_m, typ_gap):
    cap = cv2.VideoCapture(VIDEO)
    src = annotation_frame_to_source_frame(frame, cap.get(cv2.CAP_PROP_FPS) or 30.0, DEFAULT_SCENARIO_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, src)
    ok, img = cap.read()
    cap.release()
    if not ok:
        print(f"  Could not read source frame {src}")
        return
    pts_all = []
    for tid, w, color in ((ego, we, (0, 255, 0)), (other, wo, (0, 165, 255))):
        poly = np.array(ast.literal_eval(w.at[frame, "kf_obbox_coord_pixel"]), dtype=np.int32)
        c = (int(round(w.at[frame, "kf_cx_pixel"])), int(round(w.at[frame, "kf_cy_pixel"])))
        cv2.polylines(img, [poly], True, color, 2, cv2.LINE_AA)
        cv2.circle(img, c, 5, color, -1)
        cv2.putText(img, str(tid), (c[0] + 8, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        pts_all += poly.tolist() + [c]
    pts_all = np.array(pts_all)
    x0, y0 = pts_all.min(axis=0) - 120
    x1, y1 = pts_all.max(axis=0) + 120
    h, w_ = img.shape[:2]
    crop = img[max(0, y0):min(h, y1), max(0, x0):min(w_, x1)]
    crop = cv2.resize(crop, None, fx=800 / crop.shape[1], fy=800 / crop.shape[1], interpolation=cv2.INTER_CUBIC)
    banner = np.zeros((90, crop.shape[1], 3), np.uint8)
    cv2.putText(banner, f"Annotation frame {frame} (video frame {src})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(banner, f"Recorded centres {dist_m:.2f} m apart; real-size vehicles would overlap by {-typ_gap:.2f} m",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.imwrite(out_path, np.vstack([banner, crop]))
    print(f"  Saved {os.path.basename(out_path)}")


def render_zoom_strip(out_path, frames, i, we, wo, ego, other):
    """Raw (top) and boxed (bottom) drone crops a few frames before, at and after frame index i."""
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    picks = sorted({frames[max(0, i - 3)], frames[i], frames[min(len(frames) - 1, i + 3)]})
    raw_tiles, box_tiles = [], []
    for f in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, annotation_frame_to_source_frame(f, fps, DEFAULT_SCENARIO_FPS))
        ok, img = cap.read()
        if not ok:
            continue
        boxed = img.copy()
        for w, col in ((we, (0, 255, 0)), (wo, (0, 165, 255))):
            cv2.polylines(boxed, [np.array(ast.literal_eval(w.at[f, "kf_obbox_coord_pixel"]), np.int32)], True, col, 1)
        c = np.mean([[w.at[f, "kf_cx_pixel"], w.at[f, "kf_cy_pixel"]] for w in (we, wo)], axis=0).astype(int)
        x0, y0 = np.maximum(c - 60, 0)
        for im, tiles, tag in ((img, raw_tiles, "raw"), (boxed, box_tiles, "boxes")):
            tile = cv2.resize(im[y0:y0 + 120, x0:x0 + 120], (360, 360), interpolation=cv2.INTER_CUBIC)
            cv2.putText(tile, f"frame {f} {tag}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            tiles.append(tile)
    cap.release()
    cv2.imwrite(out_path, np.vstack([np.hstack(raw_tiles), np.hstack(box_tiles)]))
    print(f"  Saved {os.path.basename(out_path)}")


def render_all_tracks(out_path, frame, ego, other):
    """Every tracked box within ~35 m of the pair at one frame, to show crowding and duplicate tracks."""
    cols = ["frameID", "trackID", "labelID", "kf_cx_pixel", "kf_cy_pixel", "kf_obbox_coord_pixel"]
    fid = f"frame_{frame:06d}"
    df = pd.concat(ch[ch.frameID == fid] for ch in pd.read_csv(STABILIZE_EKF_CSV, usecols=cols, chunksize=500_000))
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, annotation_frame_to_source_frame(frame, cap.get(cv2.CAP_PROP_FPS) or 30.0, DEFAULT_SCENARIO_FPS))
    ok, img = cap.read()
    cap.release()
    if not ok:
        return
    c = df[df.trackID.isin([ego, other])][["kf_cx_pixel", "kf_cy_pixel"]].mean().to_numpy().astype(int)
    x0, y0 = np.maximum(c - 250, 0)
    near = df[(abs(df.kf_cx_pixel - c[0]) < 250) & (abs(df.kf_cy_pixel - c[1]) < 250)]
    label_cols = {0: (255, 0, 255), 1: (0, 255, 255), 2: (255, 255, 0), 3: (0, 0, 255), 4: (200, 200, 200), 5: (0, 0, 255)}
    for r in near.itertuples():
        col = (0, 255, 0) if r.trackID == ego else (0, 165, 255) if r.trackID == other else label_cols.get(r.labelID, (255, 255, 255))
        cv2.polylines(img, [np.array(ast.literal_eval(r.kf_obbox_coord_pixel), np.int32)], True, col, 1)
        cv2.putText(img, f"{r.trackID}:{r.labelID}", (int(r.kf_cx_pixel) + 6, int(r.kf_cy_pixel) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
    tile = cv2.resize(img[y0:y0 + 500, x0:x0 + 500], (900, 900), interpolation=cv2.INTER_CUBIC)
    cv2.putText(tile, f"frame {frame}: all tracks near {ego}/{other} (track:label, 0=3W 1=car 2=2W 4=ped)",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(out_path, tile)
    print(f"  Saved {os.path.basename(out_path)}")


def render_distance_plot(out_path, frames, dist, typ_gap, ego, other, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = (np.array(frames) - frames[0]) / REAL_WORLD_FPS
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
    ax.plot(t, dist, "o-", label="recorded centre-to-centre distance")
    ax.plot(t, dist - typ_gap, "--", color="gray", label="distance real-size vehicles need at that angle")
    bad = typ_gap < 0
    ax.scatter(t[bad], dist[bad], color="red", zorder=5, label="real-size vehicles would overlap")
    i = int(np.argmin(dist))
    ax.annotate(f"{dist[i]:.2f} m", (t[i], dist[i]), textcoords="offset points", xytext=(8, -14), color="red")
    ax.set_xlabel("time in scenario (s)")
    ax.set_ylabel("metres")
    ax.set_title(f"Vehicle {ego} ({labels[0]}) vs {other} ({labels[1]}), frames {frames[0]}-{frames[-1]}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {os.path.basename(out_path)}")


def replay_and_record(world, sc_dir, start, mode):
    """Runs the scenario in CARLA and returns per-tick states [(t, {name: (corners, x, y)}, gap, collided)]."""
    import json
    with open(os.path.join(sc_dir, "spawn_positions_retracted.json")) as f:
        spawns = json.load(f)
    with open(os.path.join(sc_dir, "time_variant_trajectories.json")) as f:
        t_data = json.load(f)
    for a in list(world.get_actors().filter("vehicle.*")) + list(world.get_actors().filter("sensor.*")):
        a.destroy()
    world.tick()
    lib = world.get_blueprint_library()
    vehicles = {}
    for tid, s in spawns.items():
        route = build_timed_route(tid, t_data, start)
        if route is None:
            continue
        bp = lib.find(s["blueprint"])
        yaw = sc.path_yaw_deg([tuple(p) for p in route[1][:8]]) or s["yaw"]
        v = CapturingVehicle(world, bp, f"vehicle_{tid}", (s["x"], s["y"], s["z"]), yaw, sc_dir, enable_avoidance=False)
        v.set_timed_route(*route, z=DEFAULT_Z)
        vehicles[f"vehicle_{tid}"] = v
    if mode == "expert":
        for v in vehicles.values():
            v.vehicle.set_simulate_physics(False)
        world.tick()
    else:
        for v in vehicles.values():
            v.hold_in_gear()
        for _ in range(SETTLE_TICKS):
            world.tick()
        for v in vehicles.values():
            v.set_initial_speed(float(v.timed_route[2][0]))
    duration = max(v.timed_route[0][-1] for v in vehicles.values()) + 1.0 / REAL_WORLD_FPS
    states, t = [], 0.0
    names = list(vehicles)
    for _ in range(int(duration * SIM_FPS)):
        for v in vehicles.values():
            if mode == "expert":
                x, y, _, _ = v.reference(t)
                xa, ya, _, _ = v.reference(t + 0.1)
                xb, yb, _, _ = v.reference(max(0.0, t - 0.1))
                yaw_e = math.degrees(math.atan2(ya - yb, xa - xb)) if math.hypot(xa - xb, ya - yb) > 1e-3 \
                    else v.vehicle.get_transform().rotation.yaw
                v.vehicle.set_transform(carla.Transform(carla.Location(x=x, y=y, z=DEFAULT_Z), carla.Rotation(yaw=yaw_e)))
            else:
                v.update(t, vehicles)
        world.tick()
        t += 1.0 / SIM_FPS
        snap = {n: (actor_box(vehicles[n].vehicle), vehicles[n].vehicle.get_location().x, vehicles[n].vehicle.get_location().y)
                for n in names}
        gap = sc.box_gap(snap[names[0]][0], snap[names[1]][0]) if len(names) == 2 else float("nan")
        states.append((t, snap, gap, any(v.has_collided for v in vehicles.values())))
    routes = {n: vehicles[n].timed_route[1] for n in names}
    for v in vehicles.values():
        v.destroy()
    world.tick()
    return states, routes


def render_topdown_video(out_path, world, states, routes, title):
    """Top-down video of the CARLA run: map lanes, recorded routes, vehicle boxes, gap and collision state."""
    all_xy = np.vstack(list(routes.values()))
    cx, cy = all_xy.mean(axis=0)
    span = max(np.ptp(all_xy[:, 0]), np.ptp(all_xy[:, 1]), 20.0) + 16.0
    W, H = 1280, 720
    ppm = min(W, H - 80) / span

    def px(x, y):  # CARLA x right, y down on screen (top-down view as seen from above)
        return int(W / 2 + (x - cx) * ppm), int(80 + (H - 80) / 2 + (y - cy) * ppm)

    base = np.full((H, W, 3), 235, np.uint8)
    for wp in world.get_map().generate_waypoints(1.0):
        loc = wp.transform.location
        if abs(loc.x - cx) > span or abs(loc.y - cy) > span or wp.lane_type != carla.LaneType.Driving:
            continue
        r = wp.transform.get_right_vector()
        hw = wp.lane_width / 2
        for s in (-1, 1):
            cv2.circle(base, px(loc.x + r.x * hw * s, loc.y + r.y * hw * s), 1, (150, 150, 150), -1)
        cv2.circle(base, px(loc.x, loc.y), 1, (200, 200, 200), -1)
    colors = [(0, 160, 0), (0, 120, 255)]
    for (name, xy), col in zip(routes.items(), colors):
        cv2.polylines(base, [np.array([px(*p) for p in xy], np.int32)], False, col, 1, cv2.LINE_AA)

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), float(SIM_FPS), (W, H))
    min_gap = min(s[2] for s in states)
    for t, snap, gap, collided in states:
        img = base.copy()
        for (name, (corners, x, y)), col in zip(snap.items(), colors):
            fill = (0, 0, 255) if gap < 0 else col
            cv2.fillPoly(img, [np.array([px(*c) for c in corners], np.int32)], fill)
            cv2.polylines(img, [np.array([px(*c) for c in corners], np.int32)], True, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, name.replace("vehicle_", ""), (px(x, y)[0] + 12, px(x, y)[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        cv2.rectangle(img, (0, 0), (W, 70), (30, 30, 30), -1)
        cv2.putText(img, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        status = f"t={t:4.2f}s   box gap {gap:+.2f} m   (min {min_gap:+.2f} m)" + ("   COLLISION SENSOR FIRED" if collided else "")
        cv2.putText(img, status, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if gap < 0 or collided else (180, 255, 180), 2)
        writer.write(img)
    writer.release()
    print(f"  Saved {os.path.basename(out_path)} (min box gap {min_gap:+.2f} m)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--category", required=True, help='Behaviour folder, e.g. "Overtaking Left".')
    p.add_argument("--ego", type=int, required=True)
    p.add_argument("--other", type=int, required=True)
    p.add_argument("--start", type=int, required=True, help="start_frame of the CSV row")
    p.add_argument("--camera", action="store_true", help="Film the CARLA runs with a bird's-eye camera (needs GPU rendering).")
    p.add_argument("--camera-z", type=float, default=22.0, help="Bird's-eye camera height in metres.")
    args = p.parse_args()

    end, _ = find_row(args.category, args.ego, args.other, args.start)
    name = f"scenario_{args.ego}_{args.other}_{args.start}"
    sc_dir = os.path.join(SCENARIOS_DIR, args.category, name)
    os.makedirs(sc_dir, exist_ok=True)
    ids = [args.ego, args.other]
    summary = os.path.join(sc_dir, f"{name}_summary.txt")
    with open(summary, "w") as f:
        f.write(f"Start Frame: {args.start}\nEnd Frame: {end}\nEgo ID: {args.ego}\n\nHighlight IDs:\n")
        f.write("".join(f"- {i}\n" for i in ids) + "\nVehicles:\n" + "".join(f"- {i}\n" for i in ids))
    summary_data = (args.start, end, ids, ids, args.ego)
    print(f"{sc_dir}  (frames {args.start}-{end})")

    store = sc.TrackStore(STABILIZE_EKF_CSV, track_ids=ids, extra_columns=["kf_obbox_coord_pixel"])
    frames, dist, typ_gap, we, wo = closest_moment(store, args.ego, args.other, args.start, end)
    i = int(np.argmin(dist))
    labels = [LABEL_NAMES.get(store.label(t), "?") for t in ids]
    print(f"  Closest recorded centres: {dist[i]:.2f} m at frame {frames[i]}; typical-size gap {typ_gap[i]:+.2f} m")
    render_distance_plot(os.path.join(sc_dir, "centre_distance.png"), frames, dist, typ_gap, args.ego, args.other, labels)
    render_closest_still(os.path.join(sc_dir, "closest_moment_real.png"), frames[i], we, wo, args.ego, args.other, dist[i], typ_gap[i])
    render_zoom_strip(os.path.join(sc_dir, "closest_moment_zoom.png"), frames, i, we, wo, args.ego, args.other)
    render_all_tracks(os.path.join(sc_dir, f"frame{frames[i]}_all_tracks.png"), frames[i], args.ego, args.other)

    parser = AnnotationParser(STABILIZE_EKF_CSV)
    render_real_world_drone_trajectory_image(sc_dir, args.ego, args.other, args.start, end, f"{args.ego} vs {args.other}", parser)
    subprocess.run([sys.executable, os.path.join(BASE_DIR, "overlay_scenario_bboxes.py"), "--summary", summary,
                    "--annotations", STABILIZE_EKF_CSV, "--video", VIDEO,
                    "--output", os.path.join(sc_dir, "real_drone_video.mp4")], check=True)

    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    world = client.get_world()
    if TOWN not in world.get_map().name:
        world = client.load_world(TOWN)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / SIM_FPS
    world.apply_settings(settings)

    run_stage_1_matrix(sc_dir, summary_data, parser)
    run_stage_2_time_method(sc_dir, summary_data, parser)
    run_stage_3_spawns(sc_dir, summary_data, parser, dji_spawn_params=None, world=world)
    world.tick()

    if args.camera:
        world.set_weather(carla.WeatherParameters.ClearNoon)
        render_trajectory_comparison_map(client, world, sc_dir, summary_data)
        for _ in range(5):  # let the server finish destroying that camera before the replays clean up actors
            world.tick()
        for mode in ("expert", "pid"):
            render_carla_simulation_video(client, world, sc_dir, summary, summary_data, parser, None, False,
                                          mode=mode, render=True, camera_z=args.camera_z)
        return
    for mode, label in (("expert", "EXPERT: positions set exactly from the drone data (physics off)"),
                        ("pid", "PID: cars driven by the controller along the recorded paths (physics on)")):
        states, routes = replay_and_record(world, sc_dir, args.start, mode)
        render_topdown_video(os.path.join(sc_dir, f"carla_sim_{mode}.mp4"), world, states, routes,
                             f"CARLA Town10HD | {name} | {label}")


if __name__ == "__main__":
    main()
