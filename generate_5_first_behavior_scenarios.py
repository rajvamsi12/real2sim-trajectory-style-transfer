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
- real_drone_video.mp4 (Real drone map video with EGO vehicle clearly marked in Neon Green BBox & active vehicles only)
"""

import os
import sys
import json
import math
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
DRONE_MAP_PATH = os.path.join(SCRIPT_DIR, "drone_background_map.png")
SCENARIOS_DIR = os.path.join(SCRIPT_DIR, "Scenariosssss")

# First strictly unique 100% in-box scenarios for the 5 behaviors
TARGET_SCENARIOS = {
    "Cut Out": {"ego_id": 22, "other_ids": [135], "start_frame": 71, "end_frame": 83},
    "Cut In": {"ego_id": 235, "other_ids": [250], "start_frame": 169, "end_frame": 181},
    "U-turn": {"ego_id": 0, "other_ids": [], "start_frame": 526, "end_frame": 556},
    "Overtaking Left": {"ego_id": 190, "other_ids": [2827], "start_frame": 122, "end_frame": 139},
    "Overtaking Right": {"ego_id": 10, "other_ids": [163], "start_frame": 62, "end_frame": 79},
}

def render_real_drone_video(scenario_dir, summary_data, annotation_parser):
    """Render real drone map video with EGO vehicle clearly highlighted in Neon Green and only active vehicles drawn."""
    start_frame, end_frame, vehicle_ids, highlight_ids, ego_id = summary_data
    out_video_path = os.path.join(scenario_dir, "real_drone_video.mp4")
    print(f"  Rendering Real Drone Video -> {os.path.basename(out_video_path)}...")

    if not os.path.exists(DRONE_MAP_PATH):
        print(f"Error: Map image not found at {DRONE_MAP_PATH}")
        return

    base_map = cv2.imread(DRONE_MAP_PATH)
    orig_h, orig_w = base_map.shape[:2]

    # Target render dimensions (1280x720)
    target_w, target_h = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(out_video_path, fourcc, 10.0, (target_w, target_h))

    for frame in range(start_frame, end_frame + 1):
        frame_img = base_map.copy()

        # Render active participating vehicles for this frame ONLY
        for tid in vehicle_ids:
            if tid in annotation_parser.track_data:
                t_dict = annotation_parser.track_data[tid]
                traj = t_dict.get("trajectory", {})
                # FIX: Only draw vehicle if it is actively present on this exact frame!
                if frame in traj:
                    mx, my_neg = traj[frame]
                    my = -my_neg  # Image space Y is positive downwards
                    u, v = int(round(mx)), int(round(my))

                    is_ego = (tid == ego_id)
                    color = (0, 255, 0) if is_ego else (0, 165, 255)  # Green for EGO, Orange for others
                    box_size = 35 if is_ego else 22
                    thickness = 4 if is_ego else 2

                    # Draw Bounding Box
                    x1, y1 = max(0, u - box_size), max(0, v - box_size)
                    x2, y2 = min(orig_w - 1, u + box_size), min(orig_h - 1, v + box_size)
                    cv2.rectangle(frame_img, (x1, y1), (x2, y2), color, thickness)
                    cv2.circle(frame_img, (u, v), 6 if is_ego else 4, color, -1)

                    # Label text
                    lbl = f"EGO vehicle_{tid}" if is_ego else f"vehicle_{tid}"
                    cv2.rectangle(frame_img, (x1 - 2, y1 - 22), (x1 + len(lbl)*11, y1), (0, 0, 0), -1)
                    cv2.putText(frame_img, lbl, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55 if is_ego else 0.45, color, 2 if is_ego else 1, cv2.LINE_AA)

        # Header status banner
        cv2.rectangle(frame_img, (20, 20), (850, 80), (20, 20, 30), -1)
        cv2.rectangle(frame_img, (20, 20), (850, 80), (0, 255, 0), 2)
        cv2.putText(frame_img, f"REAL DRONE VIDEO | Frame {frame} | EGO VEHICLE: vehicle_{ego_id}", (35, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

        resized_frame = cv2.resize(frame_img, (target_w, target_h))
        video_writer.write(resized_frame)

    video_writer.release()
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

        # Create scenario_0001_summary.txt
        summary_txt = os.path.join(sc_folder, "scenario_0001_summary.txt")
        with open(summary_txt, "w") as f:
            f.write(f"Start Frame: {sf}\n")
            f.write(f"End Frame: {ef}\n")
            f.write(f"All Vehicle IDs: {all_ids}\n")
            f.write(f"Highlight Vehicle IDs: {all_ids}\n")
            f.write(f"Ego Vehicle ID: {ego_id}\n")

        summary_data = (sf, ef, all_ids, all_ids, ego_id)

        # Stage 1: Matrix Conversion
        run_stage_1_matrix(sc_folder, summary_data, parser)

        # Stage 2: Time Method Conversion
        run_stage_2_time_method(sc_folder, summary_data, parser)

        # Stage 3: Spawn Extraction
        run_stage_3_spawns(sc_folder, summary_data, parser, dji_spawn_params=None, world=world)

        # Stage 4: Real Drone Video & Visual Plots
        render_real_drone_video(sc_folder, summary_data, parser)

        # Stage 4: Top-Down CARLA Video (Without Avoidance only as requested)
        render_trajectory_comparison_map(client, world, sc_folder, summary_data)
        render_carla_simulation_video(
            client, world, sc_folder, summary_txt, summary_data,
            parser, dji_spawn_params=None, enable_avoidance=False
        )

    print("\nAll 5 first behavior scenarios successfully generated!")

if __name__ == "__main__":
    main()
