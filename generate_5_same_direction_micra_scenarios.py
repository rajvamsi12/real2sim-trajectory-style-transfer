#!/usr/bin/env python3
"""
Generate complete deliverables for 5 Same-Direction Dual Nissan Micra Overtaking Scenarios:
1. Row 545   (Ego: 1304, Other: 1133, Frames: 944..961)
2. Row 5256  (Ego: 2294, Other: 782,  Frames: 689..706)
3. Row 532   (Ego: 1514, Other: 1332, Frames: 939..956)
4. Row 19112 (Ego: 149,  Other: 2278, Frames: 351..368)
5. Row 528   (Ego: 1514, Other: 1332, Frames: 936..953)

Generates per scenario:
- 3 Videos: real_drone_video.mp4, trajectory_video_without_avoidance.mp4, trajectory_video_with_avoidance.mp4
- 2 Images: real_world_trajectory.png, trajectory_comparison.png
"""

import os
import sys
import json
import math
import subprocess
import cv2
import numpy as np
import carla

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from simulate_and_record_scenarios import AnnotationParser, SIM_FPS
from scenario_checks import TrackStore, check_pair
from debug.batch_process_scenarios import (
    run_stage_1_matrix,
    run_stage_2_time_method,
    run_stage_3_spawns,
    render_trajectory_comparison_map,
    render_carla_simulation_video,
)

STABILIZE_EKF_CSV = os.path.join(SCRIPT_DIR, "stabilize_ekf_final2.csv")
REAL_DRONE_VIDEO_PATH = os.path.join(SCRIPT_DIR, "DJI_0003-002.MP4")
DRONE_MAP_PATH = os.path.join(SCRIPT_DIR, "drone_background_map.png")
SCENARIOS_DIR = os.path.join(SCRIPT_DIR, "Scenariosssss")

TARGET_SCENARIOS = [
    {"row": 545,   "folder": "scenario_0545",  "ego": 1304, "other": 1133, "start": 944, "end": 961,  "label": "Row 545 (Ego:1304, Other:1133)"},
    {"row": 5256,  "folder": "scenario_5256",  "ego": 2294, "other": 782,  "start": 689, "end": 706,  "label": "Row 5256 (Ego:2294, Other:782)"},
    {"row": 532,   "folder": "scenario_0532",  "ego": 1514, "other": 1332, "start": 939, "end": 956,  "label": "Row 532 (Ego:1514, Other:1332)"},
    {"row": 19112, "folder": "scenario_19112", "ego": 149,  "other": 2278, "start": 351, "end": 368,  "label": "Row 19112 (Ego:149, Other:2278)"},
    {"row": 528,   "folder": "scenario_0528",  "ego": 1514, "other": 1332, "start": 936, "end": 953,  "label": "Row 528 (Ego:1514, Other:1332)"}
]

def render_real_world_drone_trajectory_image(scenario_dir, ego_id, other_id, start_frame, end_frame, label_str, annotation_parser):
    """Render static real-world drone map image with full trajectories plotted."""
    out_img_path = os.path.join(scenario_dir, "real_world_trajectory.png")
    print(f"  Rendering Real World Trajectory Image -> {os.path.basename(out_img_path)}...")

    if not os.path.exists(DRONE_MAP_PATH):
        print(f"Error: Map image not found at {DRONE_MAP_PATH}")
        return

    map_img = cv2.imread(DRONE_MAP_PATH)
    all_ids = [ego_id, other_id]

    for tid in all_ids:
        if tid in annotation_parser.track_data:
            t_dict = annotation_parser.track_data[tid]
            traj = t_dict.get("trajectory", {})
            pts = []
            for f in range(start_frame, end_frame + 1):
                if f in traj:
                    mx, my_neg = traj[f]
                    my = -my_neg
                    pts.append((int(round(mx)), int(round(my))))
            
            is_ego = (tid == ego_id)
            color = (0, 255, 0) if is_ego else (0, 165, 255) # Green for EGO, Orange for Other
            
            for i in range(1, len(pts)):
                cv2.line(map_img, pts[i-1], pts[i], color, 4 if is_ego else 3, cv2.LINE_AA)
            
            if pts:
                cv2.circle(map_img, pts[0], 8 if is_ego else 6, (0, 255, 255), -1)
                cv2.circle(map_img, pts[-1], 8 if is_ego else 6, color, -1)
                
                lbl = f"EGO {tid} (Micra)" if is_ego else f"VEH {tid} (Micra)"
                u, v = pts[0]
                cv2.rectangle(map_img, (u - 2, v - 22), (u + len(lbl)*12, v), (0, 0, 0), -1)
                cv2.putText(map_img, lbl, (u, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # Header banner
    cv2.rectangle(map_img, (20, 20), (950, 80), (20, 20, 30), -1)
    cv2.rectangle(map_img, (20, 20), (950, 80), (0, 255, 0), 2)
    cv2.putText(map_img, f"DUAL MICRA TRAJECTORY MAP | {label_str} | Frames [{start_frame}..{end_frame}]", (35, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imwrite(out_img_path, map_img)
    print(f"  Saved {os.path.basename(out_img_path)}")

def render_real_drone_video(scenario_dir, summary_txt_path):
    """Render real moving drone MP4 video (DJI_0003-002.MP4) with live camera footage and vehicle bounding boxes overlaid."""
    out_video_path = os.path.join(scenario_dir, "real_drone_video.mp4")
    print(f"  Rendering Real Drone MP4 Video -> {os.path.basename(out_video_path)}...")
    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "overlay_scenario_bboxes.py"),
        "--summary", summary_txt_path,
        "--annotations", STABILIZE_EKF_CSV,
        "--video", REAL_DRONE_VIDEO_PATH,
        "--output", out_video_path
    ]
    subprocess.run(cmd, check=True)
    print(f"  Saved {os.path.basename(out_video_path)}")

def main():
    force = "--force" in sys.argv

    # Reject scenarios an exact replay cannot survive (overlapping real boxes, tracking glitches,
    # no side-by-side clearance). --force renders them anyway, e.g. to show the failure.
    store = TrackStore(STABILIZE_EKF_CSV, track_ids={i for s in TARGET_SCENARIOS for i in (s["ego"], s["other"])})
    runnable = []
    for sc in TARGET_SCENARIOS:
        ok, _, reasons = check_pair(store, sc["ego"], sc["other"], sc["start"], sc["end"])
        if ok or force:
            runnable.append(sc)
        if not ok:
            print(f"{'FORCED' if force else 'SKIPPED'} {sc['label']}: " + "; ".join(reasons))
    if not runnable:
        print("No scenario passed scenario_checks.check_pair (use --force to render anyway).")
        return

    print("Loading EKF dataset from CSV...")
    parser = AnnotationParser(STABILIZE_EKF_CSV)

    print("Connecting to CARLA simulator on localhost:2000...")
    client = carla.Client("localhost", 2000)
    client.set_timeout(15.0)
    world = client.load_world("Town10HD")
    world.set_weather(carla.WeatherParameters.ClearNoon)
    # The replay ticks the world itself at a fixed 20 Hz (the PID gains assume dt = 1/20 s)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / SIM_FPS
    world.apply_settings(settings)

    for sc in runnable:
        sc_folder = os.path.join(SCENARIOS_DIR, "Overtaking Left", sc["folder"])
        os.makedirs(sc_folder, exist_ok=True)

        ego_id = sc["ego"]
        other_id = sc["other"]
        start_frame = sc["start"]
        end_frame = sc["end"]
        all_ids = [ego_id, other_id]

        print(f"\n=======================================================")
        print(f"Processing Scenario: {sc['label']}")
        print(f"  Folder: {sc_folder}")
        print(f"  Ego ID: {ego_id}, Other ID: {other_id}, Frames: [{start_frame} .. {end_frame}]")
        print(f"=======================================================")

        summary_txt = os.path.join(sc_folder, f"{sc['folder']}_summary.txt")
        with open(summary_txt, "w") as f:
            f.write(f"Start Frame: {start_frame}\n")
            f.write(f"End Frame: {end_frame}\n")
            f.write(f"Ego ID: {ego_id}\n\n")
            f.write("Highlight IDs:\n")
            for v_id in all_ids:
                f.write(f"- {v_id}\n")
            f.write("\nVehicles:\n")
            for v_id in all_ids:
                f.write(f"- {v_id}\n")

        summary_data = (start_frame, end_frame, all_ids, all_ids, ego_id)

        # Stage 1: Matrix Conversion
        run_stage_1_matrix(sc_folder, summary_data, parser)

        # Stage 2: Time Method Conversion
        run_stage_2_time_method(sc_folder, summary_data, parser)

        # Stage 3: Spawn Extraction
        run_stage_3_spawns(sc_folder, summary_data, parser, dji_spawn_params=None, world=world)

        # Stage 4 Images
        render_real_world_drone_trajectory_image(sc_folder, ego_id, other_id, start_frame, end_frame, sc["label"], parser)
        render_trajectory_comparison_map(client, world, sc_folder, summary_data)

        # Stage 4 Videos
        render_real_drone_video(sc_folder, summary_txt)

        render_carla_simulation_video(
            client, world, sc_folder, summary_txt, summary_data,
            parser, dji_spawn_params=None, enable_avoidance=False
        )

        render_carla_simulation_video(
            client, world, sc_folder, summary_txt, summary_data,
            parser, dji_spawn_params=None, enable_avoidance=True
        )

        print(f"Successfully generated all 5 deliverables for {sc['label']}!")

    print("\nAll 5 Same-Direction Dual Nissan Micra Overtaking Scenarios successfully generated!")

if __name__ == "__main__":
    main()
