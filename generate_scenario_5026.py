#!/usr/bin/env python3
"""
Generate complete deliverables for Overtaking Row 5026 (Ego: 1200, Other: 2871, Frames: 634..709, Duration: 17.6s, 100% In-Box):
1. 3 Videos:
   - real_drone_video.mp4 (Live moving drone camera footage from DJI_0003-002.MP4)
   - trajectory_video_without_avoidance.mp4 (Top-down CARLA video without avoidance)
   - trajectory_video_with_avoidance.mp4 (Top-down CARLA video with active collision avoidance)
2. Images:
   - trajectory_comparison.png (CARLA map trajectory overlay image)
   - real_world_trajectory.png (Real world drone map trajectory plot image)
"""

import os
import sys
import json
import math
import subprocess
import cv2
import numpy as np
import matplotlib.pyplot as plt
import carla

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from simulate_and_record_scenarios import AnnotationParser
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

# Target Scenario Details for Row 5026
EGO_ID = 1200
OTHER_ID = 2871
START_FRAME = 634
END_FRAME = 709
ALL_IDS = [EGO_ID, OTHER_ID]
SCENARIO_FOLDER = os.path.join(SCENARIOS_DIR, "Overtaking Left", "scenario_5026")

def render_real_world_drone_trajectory_image(scenario_dir, summary_data, annotation_parser):
    """Render static real-world drone map image with full trajectories plotted."""
    out_img_path = os.path.join(scenario_dir, "real_world_trajectory.png")
    print(f"  Rendering Real World Trajectory Image -> {os.path.basename(out_img_path)}...")

    if not os.path.exists(DRONE_MAP_PATH):
        print(f"Error: Map image not found at {DRONE_MAP_PATH}")
        return

    map_img = cv2.imread(DRONE_MAP_PATH)
    orig_h, orig_w = map_img.shape[:2]

    # Draw trajectories on map image
    for tid in ALL_IDS:
        if tid in annotation_parser.track_data:
            t_dict = annotation_parser.track_data[tid]
            traj = t_dict.get("trajectory", {})
            pts = []
            for f in range(START_FRAME, END_FRAME + 1):
                if f in traj:
                    mx, my_neg = traj[f]
                    my = -my_neg
                    pts.append((int(round(mx)), int(round(my))))
            
            is_ego = (tid == EGO_ID)
            color = (0, 255, 0) if is_ego else (0, 165, 255) # Green for EGO, Orange for Other
            
            # Draw line polyline
            for i in range(1, len(pts)):
                cv2.line(map_img, pts[i-1], pts[i], color, 4 if is_ego else 3, cv2.LINE_AA)
            
            # Start/End circles
            if pts:
                cv2.circle(map_img, pts[0], 8 if is_ego else 6, (0, 255, 255), -1)
                cv2.circle(map_img, pts[-1], 8 if is_ego else 6, color, -1)
                
                # Label
                lbl = f"EGO {tid}" if is_ego else f"VEH {tid}"
                u, v = pts[0]
                cv2.rectangle(map_img, (u - 2, v - 22), (u + len(lbl)*12, v), (0, 0, 0), -1)
                cv2.putText(map_img, lbl, (u, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # Header banner
    cv2.rectangle(map_img, (20, 20), (900, 80), (20, 20, 30), -1)
    cv2.rectangle(map_img, (20, 20), (900, 80), (0, 255, 0), 2)
    cv2.putText(map_img, f"REAL WORLD TRAJECTORY MAP | Frames [{START_FRAME}..{END_FRAME}] (17.6s)", (35, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imwrite(out_img_path, map_img)
    print(f"  Saved {os.path.basename(out_img_path)}")

def render_real_drone_video(scenario_dir):
    """Render real moving drone MP4 video (DJI_0003-002.MP4) with live camera footage and vehicle bounding boxes overlaid."""
    out_video_path = os.path.join(scenario_dir, "real_drone_video.mp4")
    summary_path = os.path.join(scenario_dir, "scenario_5026_summary.txt")
    
    print(f"  Rendering Real Drone MP4 Video -> {os.path.basename(out_video_path)}...")
    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "overlay_scenario_bboxes.py"),
        "--summary", summary_path,
        "--annotations", STABILIZE_EKF_CSV,
        "--video", REAL_DRONE_VIDEO_PATH,
        "--output", out_video_path
    ]
    subprocess.run(cmd, check=True)
    print(f"  Saved {os.path.basename(out_video_path)}")

def main():
    os.makedirs(SCENARIO_FOLDER, exist_ok=True)
    print(f"=======================================================")
    print(f"Processing Overtaking Row 5026")
    print(f"  Folder: {SCENARIO_FOLDER}")
    print(f"  Ego ID: {EGO_ID}, Other ID: {OTHER_ID}, Frames: [{START_FRAME} .. {END_FRAME}] (17.6s)")
    print(f"=======================================================")

    print("Loading EKF dataset from CSV...")
    parser = AnnotationParser(STABILIZE_EKF_CSV)

    print("Connecting to CARLA simulator on localhost:2000...")
    client = carla.Client("localhost", 2000)
    client.set_timeout(15.0)
    world = client.load_world("Town10HD")
    world.set_weather(carla.WeatherParameters.ClearNoon)

    # Write summary text file
    summary_txt = os.path.join(SCENARIO_FOLDER, "scenario_5026_summary.txt")
    with open(summary_txt, "w") as f:
        f.write(f"Start Frame: {START_FRAME}\n")
        f.write(f"End Frame: {END_FRAME}\n")
        f.write(f"Ego ID: {EGO_ID}\n\n")
        f.write("Highlight IDs:\n")
        for v_id in ALL_IDS:
            f.write(f"- {v_id}\n")
        f.write("\nVehicles:\n")
        for v_id in ALL_IDS:
            f.write(f"- {v_id}\n")

    summary_data = (START_FRAME, END_FRAME, ALL_IDS, ALL_IDS, EGO_ID)

    # Stage 1: Matrix Conversion
    run_stage_1_matrix(SCENARIO_FOLDER, summary_data, parser)

    # Stage 2: Time Method Conversion
    run_stage_2_time_method(SCENARIO_FOLDER, summary_data, parser)

    # Stage 3: Spawn Extraction
    run_stage_3_spawns(SCENARIO_FOLDER, summary_data, parser, dji_spawn_params=None, world=world)

    # Stage 4 Images: Real World Trajectory Image & CARLA Trajectory Comparison Image
    render_real_world_drone_trajectory_image(SCENARIO_FOLDER, summary_data, parser)
    render_trajectory_comparison_map(client, world, SCENARIO_FOLDER, summary_data)

    # Stage 4 Videos:
    # 1. Real Drone MP4 Video Overlay (DJI_0003-002.MP4)
    render_real_drone_video(SCENARIO_FOLDER)

    # 2. CARLA Top-Down Video Without Avoidance
    render_carla_simulation_video(
        client, world, SCENARIO_FOLDER, summary_txt, summary_data,
        parser, dji_spawn_params=None, enable_avoidance=False
    )

    # 3. CARLA Top-Down Video With Avoidance
    render_carla_simulation_video(
        client, world, SCENARIO_FOLDER, summary_txt, summary_data,
        parser, dji_spawn_params=None, enable_avoidance=True
    )

    print(f"\nAll deliverables for Overtaking Row 5026 successfully generated in {SCENARIO_FOLDER}!")

if __name__ == "__main__":
    main()
