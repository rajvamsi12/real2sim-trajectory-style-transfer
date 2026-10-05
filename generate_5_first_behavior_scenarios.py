#!/usr/bin/env python3
"""
Generate complete deliverables for the FIRST 100% IN-BOX UNIQUE scenario for 5 behaviors:
1. Cut Out
2. Cut In
3. U-turn
4. Overtaking Left
5. Overtaking Right

For each scenario, generates:
- scenario_0001_summary.txt
- Stage 1, 2, 3 data JSONs & spawn plots
- trajectory_comparison.png (Trajectory plot)
- trajectory_video_without_avoidance.mp4 (CARLA top-down video with EGO vehicle clearly marked in Green HUD)
- real_drone_video.mp4 (Real moving drone video from DJI_0003-002.MP4 with EGO and vehicle bounding boxes overlaid)
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

from simulate_and_record_scenarios import AnnotationParser
from timemethod import classify_point
from debug.batch_process_scenarios import (
    run_stage_1_matrix,
    run_stage_2_time_method,
    run_stage_3_spawns,
    render_trajectory_comparison_map,
    render_carla_simulation_video,
)

STABILIZE_EKF_CSV = os.path.join(SCRIPT_DIR, "stabilize_ekf_final2.csv")
REAL_DRONE_VIDEO_PATH = os.path.join(SCRIPT_DIR, "DJI_0003-002.MP4")
SCENARIOS_DIR = os.path.join(SCRIPT_DIR, "Scenariosssss")

# First strictly unique 100% in-box scenarios for the 5 behaviors
TARGET_SCENARIOS = {
    "Cut Out": {"ego_id": 22, "other_ids": [135], "start_frame": 71, "end_frame": 83},
    "Cut In": {"ego_id": 235, "other_ids": [250], "start_frame": 169, "end_frame": 181},
    "U-turn": {"ego_id": 0, "other_ids": [], "start_frame": 526, "end_frame": 556},
    "Overtaking Left": {"ego_id": 190, "other_ids": [2827], "start_frame": 122, "end_frame": 139},
    "Overtaking Right": {"ego_id": 10, "other_ids": [163], "start_frame": 62, "end_frame": 79},
}

def render_real_drone_video(scenario_dir):
    """Render real moving drone MP4 video (DJI_0003-002.MP4) with real camera footage and vehicle bounding boxes overlaid."""
    out_video_path = os.path.join(scenario_dir, "real_drone_video.mp4")
    summary_path = os.path.join(scenario_dir, "scenario_0001_summary.txt")
    
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
    print("Loading EKF dataset from CSV...")
    parser = AnnotationParser(STABILIZE_EKF_CSV)

    print("Connecting to CARLA simulator on localhost:2000...")
    client = carla.Client("localhost", 2000)
    client.set_timeout(15.0)
    world = client.load_world("Town10HD")
    world.set_weather(carla.WeatherParameters.ClearNoon)

    for behavior_name, info in TARGET_SCENARIOS.items():
        sc_folder = os.path.join(SCENARIOS_DIR, behavior_name, "scenario_0001")
        os.makedirs(sc_folder, exist_ok=True)

        ego_id = info["ego_id"]
        other_ids = info["other_ids"]
        sf = info["start_frame"]
        ef = info["end_frame"]
        all_ids = [ego_id] + [o for o in other_ids if o != ego_id]

        print(f"\n=======================================================")
        print(f"Processing First Scenario for: {behavior_name}")
        print(f"  Folder: {sc_folder}")
        print(f"  Ego ID: {ego_id}, Others: {other_ids}, Frames: [{sf} .. {ef}]")
        print(f"=======================================================")

        # Create scenario_0001_summary.txt in standard format for overlay_scenario_bboxes.py
        summary_txt = os.path.join(sc_folder, "scenario_0001_summary.txt")
        with open(summary_txt, "w") as f:
            f.write(f"Start Frame: {sf}\n")
            f.write(f"End Frame: {ef}\n")
            f.write(f"Ego ID: {ego_id}\n\n")
            f.write("Highlight IDs:\n")
            for v_id in all_ids:
                f.write(f"- {v_id}\n")
            f.write("\nVehicles:\n")
            for v_id in all_ids:
                f.write(f"- {v_id}\n")

        summary_data = (sf, ef, all_ids, all_ids, ego_id)

        # Stage 1: Matrix Conversion
        run_stage_1_matrix(sc_folder, summary_data, parser)

        # Stage 2: Time Method Conversion
        run_stage_2_time_method(sc_folder, summary_data, parser)

        # Stage 3: Spawn Extraction
        run_stage_3_spawns(sc_folder, summary_data, parser, dji_spawn_params=None, world=world)

        # Stage 4: Real Moving Drone MP4 Video Overlay (DJI_0003-002.MP4)
        render_real_drone_video(sc_folder)

        # Stage 4: Top-Down CARLA Video
        render_trajectory_comparison_map(client, world, sc_folder, summary_data)
        render_carla_simulation_video(
            client, world, sc_folder, summary_txt, summary_data,
            parser, dji_spawn_params=None, enable_avoidance=False
        )

    print("\nAll 5 first behavior scenarios successfully generated with real drone MP4 video footage!")

if __name__ == "__main__":
    main()
