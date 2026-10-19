#!/usr/bin/env python3
"""
debug/batch_process_scenarios.py

Modular Scenario Processing & Rendering Pipeline.
Processes scenarios in Scenariosssss to generate:
  1. Trajectories (With and Without Time Method) -> trajectory_with_time_method.png & trajectory_without_time_method.png
  2. Spawn Positions (With and Without Retraction) -> spawn_positions_with_retraction.png & spawn_positions_without_retraction.png
  3. Trajectory Videos (With and Without Collision Avoidance) -> trajectory_video_with_avoidance.mp4 & trajectory_video_without_avoidance.mp4

Usage:
  /home/raj/miniconda3/envs/carla_env/bin/python debug/batch_process_scenarios.py --target Overtaking/1
  /home/raj/miniconda3/envs/carla_env/bin/python debug/batch_process_scenarios.py --target Dense
  /home/raj/miniconda3/envs/carla_env/bin/python debug/batch_process_scenarios.py --all
"""

# OpenCV MUST be imported before carla to prevent C++ symbol conflict
import cv2
import os
import sys
import glob
import json
import math
import time
import queue
import argparse
import numpy as np

# Python path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Legacy Python < 3.8 egg check
if sys.version_info < (3, 8):
    egg_path = "/home/raj/CARLA_0.9.15/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
    if os.path.exists(egg_path) and egg_path not in sys.path:
        sys.path.insert(0, egg_path)

# Import existing codebase modules
try:
    import convertion
    import timemethod
    from overlay_scenario_bboxes import parse_summary_file
    from simulate_and_record_scenarios import (
        AnnotationParser,
        load_dji_spawn_params,
        convert_real_heading_to_carla_yaw,
        build_spawn_search_transforms,
        LABEL_BLUEPRINT_MAP,
        CapturingVehicle,
        calculate_continuous_speeds,
        SIM_FPS,
        CARLA_HOST,
        CARLA_PORT,
        TOWN,
        DEFAULT_Z,
        SPAWN_SAFETY_MARGIN,
        REAL_WORLD_FPS,
        actor_box,
    )
    from scenario_checks import box_gap, path_yaw_deg
except ImportError as e:
    print(f"Warning loading codebase modules: {e}")

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False
    print("Warning: carla library not found in Python environment.")

DEFAULT_SCENARIOS_DIR = os.path.join(BASE_DIR, "Scenariosssss")
STABILIZE_EKF_CSV = os.path.join(BASE_DIR, "stabilize_ekf_final2.csv")
DEFAULT_JSON = STABILIZE_EKF_CSV if os.path.exists(STABILIZE_EKF_CSV) else os.path.join(BASE_DIR, "default.json")
DEFAULT_DJI_DATA = os.path.join(BASE_DIR, "dji0003_data.csv")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Scenario conversion, spawn extraction, and video rendering pipeline."
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target scenario or category, e.g. 'Overtaking/1', 'Dense/scenario_0001', 'Dense', 'all'.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all categories (100 scenarios).",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help="Comma-separated stages (1,2,3,4 or 'all'). Stage 1: Matrix, 2: TimeMethod, 3: Spawns, 4: Videos & Map Photos.",
    )
    parser.add_argument(
        "--scenarios-dir",
        type=str,
        default=DEFAULT_SCENARIOS_DIR,
        help="Path to Scenariosssss directory.",
    )
    parser.add_argument(
        "--json-file",
        type=str,
        default=DEFAULT_JSON,
        help="Path to annotation default.json file.",
    )
    return parser.parse_args()


def resolve_scenario_paths(scenarios_dir, target_arg, run_all):
    if run_all or target_arg == "all":
        summaries = glob.glob(os.path.join(scenarios_dir, "**/*_summary.txt"), recursive=True)
        return sorted(list(set(os.path.dirname(s) for s in summaries)))

    if not target_arg:
        print("Error: Please specify --target <category/scenario> or --all.")
        sys.exit(1)

    target_arg = target_arg.strip("/")
    parts = target_arg.split("/")

    all_categories = [d for d in os.listdir(scenarios_dir) if os.path.isdir(os.path.join(scenarios_dir, d))]

    matching_category = None
    for cat in all_categories:
        if cat.lower().strip() == parts[0].lower().strip():
            matching_category = cat
            break

    if len(parts) == 1:
        if matching_category:
            category_dir = os.path.join(scenarios_dir, matching_category)
            summaries = glob.glob(os.path.join(category_dir, "**/*_summary.txt"), recursive=True)
            return sorted(list(set(os.path.dirname(s) for s in summaries)))
        else:
            print(f"Error: Category '{parts[0]}' not found in {scenarios_dir}.")
            print(f"Available categories: {all_categories}")
            sys.exit(1)

    if len(parts) == 2:
        cat_name, sc_name = parts[0], parts[1]
        if matching_category:
            cat_name = matching_category

        category_dir = os.path.join(scenarios_dir, cat_name)
        if sc_name.isdigit():
            sc_folder_name = f"scenario_{int(sc_name):04d}"
        else:
            sc_folder_name = sc_name

        sc_dir = os.path.join(category_dir, sc_folder_name)
        if os.path.exists(sc_dir):
            return [sc_dir]

        print(f"Error: Scenario directory '{sc_dir}' does not exist.")
        sys.exit(1)

    return []


# ==========================================
# 3D TO 2D CAMERA PROJECTION HELPERS
# ==========================================
def get_camera_intrinsic_matrix(width, height, fov):
    focal = width / (2.0 * math.tan(fov * math.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = focal
    K[1, 1] = focal
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0
    return K


def world_to_pixel(world_point, cam_transform, K):
    """
    Projects a CARLA 3D world Location (x,y,z) to 2D image pixel coordinates (u,v).
    """
    world_matrix = np.array(cam_transform.get_inverse_matrix())
    point_loc = np.array([world_point.x, world_point.y, world_point.z, 1.0])
    point_in_cam = np.dot(world_matrix, point_loc)

    # Convert CARLA camera frame (X-forward, Y-right, Z-up) to OpenCV frame (X-right, Y-down, Z-forward)
    point_in_cam_cv = np.array([point_in_cam[1], -point_in_cam[2], point_in_cam[0]])

    if point_in_cam_cv[2] <= 0:
        return None

    pixel = np.dot(K, point_in_cam_cv)
    u = int(round(pixel[0] / pixel[2]))
    v = int(round(pixel[1] / pixel[2]))
    return (u, v)


def draw_overlay_legend(image, title, items):
    """
    Draws a clean, dark semi-transparent legend box on top of an OpenCV BGR image.
    items: list of (label_text, (B, G, R))
    """
    h, w = image.shape[:2]
    box_w, box_h = 650, 50 + len(items) * 35
    overlay = image.copy()
    cv2.rectangle(overlay, (20, 20), (20 + box_w, 20 + box_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.82, image, 0.18, 0, image)
    cv2.rectangle(image, (20, 20), (20 + box_w, 20 + box_h), (220, 220, 220), 2)

    cv2.putText(image, title, (32, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    for idx, (text, color) in enumerate(items):
        y_pos = 88 + idx * 32
        cv2.circle(image, (45, y_pos - 6), 9, color, -1)
        cv2.putText(image, text, (68, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)


# ==========================================
# STAGE 1: MATRIX METHOD (WITHOUT TIME METHOD)
# ==========================================
def run_stage_1_matrix(scenario_dir, summary_data, annotation_parser):
    print(f"  [Stage 1] Running Pure Matrix Conversion (without time method)...")
    start_frame, end_frame, vehicle_ids, highlight_ids, ego_id = summary_data

    matrix_trajectories = {}
    for tid in vehicle_ids:
        if tid not in annotation_parser.track_data:
            continue
        
        traj_dict = annotation_parser.track_data[tid]["trajectory"]
        carla_pts = []
        raw_pts = []
        
        for frame in range(start_frame, end_frame + 1):
            if frame in traj_dict:
                real_pt = traj_dict[frame]
                carla_pt = convertion.transform_to_carla(real_pt)
                if carla_pt is not None:
                    carla_pts.append({
                        "frame": frame,
                        "x": float(carla_pt[0]),
                        "y": float(carla_pt[1]),
                    })
                    raw_pts.append({
                        "frame": frame,
                        "real_x": float(real_pt[0]),
                        "real_y": float(real_pt[1]),
                    })

        matrix_trajectories[str(tid)] = {
            "carla_coordinates": carla_pts,
            "real_coordinates": raw_pts,
        }

    output_path = os.path.join(scenario_dir, "raw_matrix_trajectories.json")
    with open(output_path, "w") as f:
        json.dump(matrix_trajectories, f, indent=2)
    print(f"  Saved matrix trajectories to {os.path.basename(output_path)}")
    return matrix_trajectories


# ==========================================
# STAGE 2: TIME METHOD (WITH TIME-VARIANT SMOOTHING)
# ==========================================
def run_stage_2_time_method(scenario_dir, summary_data, annotation_parser):
    print(f"  [Stage 2] Running Time-Variant Trajectory Conversion (with time method)...")
    start_frame, end_frame, vehicle_ids, highlight_ids, ego_id = summary_data

    time_trajectories = {}
    for tid in vehicle_ids:
        if tid not in annotation_parser.track_data:
            continue
        
        traj_dict = annotation_parser.track_data[tid]["trajectory"]

        # Solution 2: Analyze the ENTIRE vehicle trajectory first to extract full entry/exit context
        all_frames = sorted(traj_dict.keys())
        all_real_coords = [traj_dict[f] for f in all_frames]
        all_carla_coords = timemethod.transform_trajectory_with_analysis(all_real_coords)

        # Slice transformed CARLA coordinates for scenario range [start_frame .. end_frame]
        carla_pts = []
        for frame, c_pt in zip(all_frames, all_carla_coords):
            if start_frame <= frame <= end_frame and c_pt is not None:
                carla_pts.append({
                    "frame": frame,
                    "x": float(c_pt[0]),
                    "y": float(c_pt[1]),
                })

        time_trajectories[str(tid)] = {
            "carla_coordinates": carla_pts,
        }

    output_path = os.path.join(scenario_dir, "time_variant_trajectories.json")
    with open(output_path, "w") as f:
        json.dump(time_trajectories, f, indent=2)
    print(f"  Saved time-variant trajectories to {os.path.basename(output_path)}")
    return time_trajectories


# ==========================================
# STAGE 3: SPAWN POSITIONS (WITH & WITHOUT RETRACTION)
# ==========================================
def run_stage_3_spawns(scenario_dir, summary_data, annotation_parser, dji_spawn_params, world=None):
    print(f"  [Stage 3] Extracting Vehicle Spawn Positions (with & without retraction)...")
    start_frame, end_frame, vehicle_ids, highlight_ids, ego_id = summary_data

    raw_spawns = {}
    retracted_spawns = {}
    stage3_test_actors = []

    for tid in vehicle_ids:
        if tid not in annotation_parser.track_data:
            continue

        t_data = annotation_parser.track_data[tid]
        traj_dict = t_data["trajectory"]
        headings_dict = t_data.get("headings", {})

        sc_frames = [f for f in sorted(traj_dict.keys()) if start_frame <= f <= end_frame]
        if not sc_frames:
            continue
        init_frame = sc_frames[0]
        real_pt = traj_dict[init_frame]

        # Spawn Rule: Only spawn if initial position is in Region_1..Region_6 or Center
        init_region = timemethod.classify_point(real_pt)
        valid_regions = {'Region_1', 'Region_2', 'Region_3', 'Region_4', 'Region_5', 'Region_6', 'Center'}
        if init_region not in valid_regions:
            print(f"  [Stage 3 Filter] Skipping vehicle_{tid}: initial position {real_pt} is in '{init_region}' (outside 6 regions & Center)")
            continue

        all_frames = sorted(traj_dict.keys())
        all_real_coords = [traj_dict[f] for f in all_frames]
        all_carla_coords = timemethod.transform_trajectory_with_analysis(all_real_coords)

        init_idx = all_frames.index(init_frame)
        carla_pt = all_carla_coords[init_idx]
        if carla_pt is None:
            continue

        # Spawn heading from the CARLA path over the next few frames. The EKF heading conversion
        # (convert_real_heading_to_carla_yaw) was 60-74 deg off for slow vehicles in 0528/0532,
        # so cars spawned pointing across their neighbour.
        carla_yaw = path_yaw_deg(all_carla_coords[init_idx:init_idx + 8])
        if carla_yaw is None:
            carla_yaw = 0.0

        label_id = t_data.get("label_id", t_data.get("label", 1))
        if label_id is None:
            label_id = 1
        bp_name = LABEL_BLUEPRINT_MAP.get(label_id, "vehicle.bmw.grandtourer")

        # RAW SPAWN (Without Retraction)
        raw_spawns[str(tid)] = {
            "vehicle_id": tid,
            "label_id": label_id,
            "blueprint": bp_name,
            "start_frame": init_frame,
            "x": float(carla_pt[0]),
            "y": float(carla_pt[1]),
            "z": DEFAULT_Z,
            "pitch": 0.0,
            "yaw": float(carla_yaw),
            "roll": 0.0,
        }

        # RETRACTED SPAWN (With Retraction)
        base_loc = carla.Location(x=carla_pt[0], y=carla_pt[1], z=DEFAULT_Z) if CARLA_AVAILABLE and world else None
        if CARLA_AVAILABLE and world and base_loc:
            candidates = build_spawn_search_transforms(
                world, base_loc, carla_yaw,
                annotation_parser=annotation_parser,
                tid=tid, interaction_frame=init_frame
            )
            chosen_tf = None
            bp = world.get_blueprint_library().find(bp_name)
            for cand_tf, strategy in candidates:
                # Lateral shifts (+-2/4 m) put the car beside its own path, so the PID then steers
                # diagonally across the neighbouring lane. Only retract along the path.
                if strategy.startswith("lateral"):
                    continue
                test_actor = world.try_spawn_actor(bp, cand_tf)
                if test_actor is not None:
                    chosen_tf = cand_tf
                    stage3_test_actors.append(test_actor)
                    break
            if chosen_tf is None:
                print(f"  [Stage 3] vehicle_{tid}: no collision-free spawn along its path; not spawned")
        else:
            chosen_tf = None

        if chosen_tf:
            retracted_spawns[str(tid)] = {
                "vehicle_id": tid,
                "label_id": label_id,
                "blueprint": bp_name,
                "start_frame": init_frame,
                "x": float(chosen_tf.location.x),
                "y": float(chosen_tf.location.y),
                "z": float(chosen_tf.location.z),
                "pitch": float(chosen_tf.rotation.pitch),
                "yaw": float(chosen_tf.rotation.yaw),
                "roll": float(chosen_tf.rotation.roll),
            }

    for act in stage3_test_actors:
        if act and act.is_alive:
            act.destroy()

    raw_path = os.path.join(scenario_dir, "spawn_positions_raw.json")
    with open(raw_path, "w") as f:
        json.dump(raw_spawns, f, indent=2)

    retracted_path = os.path.join(scenario_dir, "spawn_positions_retracted.json")
    with open(retracted_path, "w") as f:
        json.dump(retracted_spawns, f, indent=2)

    print(f"  Saved raw and retracted spawn position JSONs.")
    return raw_spawns, retracted_spawns


# ==========================================
# STAGE 4: RENDER MAP PHOTOS & VIDEOS
# ==========================================
def render_spawn_positions_without_retraction_map(client, world, scenario_dir, summary_data):
    """
    Renders visual photo of RAW vehicle spawn points (dots without vehicle 3D models) on the real CARLA map.
    Projects 3D spawn coordinates directly onto 2D image pixels using OpenCV for 100% visible RED DOTS.
    Output: spawn_positions_without_retraction.png
    """
    if not CARLA_AVAILABLE or not world:
        return

    spawns_path = os.path.join(scenario_dir, "spawn_positions_raw.json")
    if not os.path.exists(spawns_path):
        return

    with open(spawns_path, "r") as f:
        spawns_data = json.load(f)

    if not spawns_data:
        return

    for actor in world.get_actors().filter("vehicle.*"):
        try: actor.destroy()
        except Exception: pass

    coords_x = [s["x"] for s in spawns_data.values()]
    coords_y = [s["y"] for s in spawns_data.values()]
    min_x, max_x = min(coords_x), max(coords_x)
    min_y, max_y = min(coords_y), max(coords_y)
    avg_x, avg_y = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
    span = max(max_x - min_x, max_y - min_y, 35.0)

    # Adaptive, lower camera height (35m - 60m) for close-up clear map framing
    cam_z = max(45.0, min(75.0, span * 1.2))

    bp_library = world.get_blueprint_library()
    cam_bp = bp_library.find("sensor.camera.rgb")
    cam_w, cam_h = 1920, 1080
    cam_fov = 90.0
    cam_bp.set_attribute("image_size_x", str(cam_w))
    cam_bp.set_attribute("image_size_y", str(cam_h))
    cam_bp.set_attribute("fov", str(cam_fov))

    cam_tf = carla.Transform(
        carla.Location(x=avg_x, y=avg_y, z=cam_z),
        carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
    )

    camera = world.spawn_actor(cam_bp, cam_tf)
    img_queue = queue.Queue()
    camera.listen(img_queue.put)

    carla_img = None
    for _ in range(12):
        world.tick()
        try:
            carla_img = img_queue.get(timeout=1.0)
            if carla_img: break
        except queue.Empty: pass

    if carla_img:
        arr = np.frombuffer(carla_img.raw_data, dtype=np.dtype("uint8"))
        arr = np.reshape(arr, (carla_img.height, carla_img.width, 4))
        frame_bgr = arr[:, :, :3].copy()

        K = get_camera_intrinsic_matrix(cam_w, cam_h, cam_fov)

        # Draw RED DOTS & Labels directly on captured image pixels using 3D-to-2D projection
        for tid_str, s_info in spawns_data.items():
            loc_3d = carla.Location(x=s_info["x"], y=s_info["y"], z=0.5)
            pixel = world_to_pixel(loc_3d, cam_tf, K)
            if pixel:
                u, v = pixel
                if 0 <= u < cam_w and 0 <= v < cam_h:
                    # Draw prominent RED dot
                    cv2.circle(frame_bgr, (u, v), 12, (30, 30, 255), -1)
                    cv2.circle(frame_bgr, (u, v), 14, (255, 255, 255), 2)
                    
                    label = f"V{tid_str} (Raw)"
                    cv2.putText(frame_bgr, label, (u + 16, v + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame_bgr, label, (u + 16, v + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2, cv2.LINE_AA)

        draw_overlay_legend(
            frame_bgr,
            f"RAW SPAWN POSITIONS (BEFORE RETRACTION) - {os.path.basename(scenario_dir)}",
            [("RED DOT = Raw Vehicle Spawn Position (Before Retraction)", (30, 30, 255))]
        )

        out_path = os.path.join(scenario_dir, "spawn_positions_without_retraction.png")
        cv2.imwrite(out_path, frame_bgr)
        print(f"  Saved {os.path.basename(out_path)}")

    try: camera.stop(); camera.destroy()
    except Exception: pass


def render_spawn_positions_with_retraction_map(client, world, scenario_dir, summary_data):
    """
    Renders visual photo of RETRACTED vehicle spawn points (dots without vehicle 3D models) on the real CARLA map.
    Projects 3D spawn coordinates directly onto 2D image pixels using OpenCV for 100% visible GREEN DOTS.
    Output: spawn_positions_with_retraction.png
    """
    if not CARLA_AVAILABLE or not world:
        return

    spawns_path = os.path.join(scenario_dir, "spawn_positions_retracted.json")
    if not os.path.exists(spawns_path):
        return

    with open(spawns_path, "r") as f:
        spawns_data = json.load(f)

    if not spawns_data:
        return

    for actor in world.get_actors().filter("vehicle.*"):
        try: actor.destroy()
        except Exception: pass

    coords_x = [s["x"] for s in spawns_data.values()]
    coords_y = [s["y"] for s in spawns_data.values()]
    min_x, max_x = min(coords_x), max(coords_x)
    min_y, max_y = min(coords_y), max(coords_y)
    avg_x, avg_y = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
    span = max(max_x - min_x, max_y - min_y, 35.0)

    cam_z = max(45.0, min(75.0, span * 1.2))

    bp_library = world.get_blueprint_library()
    cam_bp = bp_library.find("sensor.camera.rgb")
    cam_w, cam_h = 1920, 1080
    cam_fov = 90.0
    cam_bp.set_attribute("image_size_x", str(cam_w))
    cam_bp.set_attribute("image_size_y", str(cam_h))
    cam_bp.set_attribute("fov", str(cam_fov))

    cam_tf = carla.Transform(
        carla.Location(x=avg_x, y=avg_y, z=cam_z),
        carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
    )

    camera = world.spawn_actor(cam_bp, cam_tf)
    img_queue = queue.Queue()
    camera.listen(img_queue.put)

    carla_img = None
    for _ in range(12):
        world.tick()
        try:
            carla_img = img_queue.get(timeout=1.0)
            if carla_img: break
        except queue.Empty: pass

    if carla_img:
        arr = np.frombuffer(carla_img.raw_data, dtype=np.dtype("uint8"))
        arr = np.reshape(arr, (carla_img.height, carla_img.width, 4))
        frame_bgr = arr[:, :, :3].copy()

        K = get_camera_intrinsic_matrix(cam_w, cam_h, cam_fov)

        # Draw GREEN DOTS & Labels directly on captured image pixels using 3D-to-2D projection
        for tid_str, s_info in spawns_data.items():
            loc_3d = carla.Location(x=s_info["x"], y=s_info["y"], z=0.5)
            pixel = world_to_pixel(loc_3d, cam_tf, K)
            if pixel:
                u, v = pixel
                if 0 <= u < cam_w and 0 <= v < cam_h:
                    # Draw prominent GREEN dot
                    cv2.circle(frame_bgr, (u, v), 12, (30, 255, 30), -1)
                    cv2.circle(frame_bgr, (u, v), 14, (255, 255, 255), 2)

                    label = f"V{tid_str} (Retracted)"
                    cv2.putText(frame_bgr, label, (u + 16, v + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame_bgr, label, (u + 16, v + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 80), 2, cv2.LINE_AA)

        draw_overlay_legend(
            frame_bgr,
            f"SPAWN POSITIONS WITH RETRACTION (RETRACTED) - {os.path.basename(scenario_dir)}",
            [("GREEN DOT = Safety Retracted Spawn Position (After Retraction)", (30, 255, 30))]
        )

        out_path = os.path.join(scenario_dir, "spawn_positions_with_retraction.png")
        cv2.imwrite(out_path, frame_bgr)
        print(f"  Saved {os.path.basename(out_path)}")

    try: camera.stop(); camera.destroy()
    except Exception: pass


def render_trajectory_image_without_time_method(client, world, scenario_dir, summary_data):
    """
    Renders pure matrix transformed trajectories (without time method) on real CARLA map using 3D-to-2D image projection.
    Output: trajectory_without_time_method.png
    """
    if not CARLA_AVAILABLE or not world:
        return

    raw_path = os.path.join(scenario_dir, "raw_matrix_trajectories.json")
    if not os.path.exists(raw_path):
        return

    with open(raw_path, "r") as f:
        raw_data = json.load(f)

    if not raw_data:
        return

    all_coords = []
    for tid_str, r_info in raw_data.items():
        r_pts = r_info.get("carla_coordinates", [])
        for pt in r_pts:
            all_coords.append((pt["x"], pt["y"]))

    if not all_coords:
        return

    coords_x = [c[0] for c in all_coords]
    coords_y = [c[1] for c in all_coords]
    min_x, max_x = min(coords_x), max(coords_x)
    min_y, max_y = min(coords_y), max(coords_y)
    avg_x, avg_y = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
    span = max(max_x - min_x, max_y - min_y, 40.0)

    # Adaptive camera height (50m - 90m) ensuring full trajectory fits in frame
    cam_z = max(55.0, min(100.0, span * 1.3))

    bp_library = world.get_blueprint_library()
    cam_bp = bp_library.find("sensor.camera.rgb")
    cam_w, cam_h = 1920, 1080
    cam_fov = 90.0
    cam_bp.set_attribute("image_size_x", str(cam_w))
    cam_bp.set_attribute("image_size_y", str(cam_h))
    cam_bp.set_attribute("fov", str(cam_fov))

    cam_tf = carla.Transform(
        carla.Location(x=avg_x, y=avg_y, z=cam_z),
        carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
    )

    camera = world.spawn_actor(cam_bp, cam_tf)
    img_queue = queue.Queue()
    camera.listen(img_queue.put)

    carla_img = None
    for _ in range(12):
        world.tick()
        try:
            carla_img = img_queue.get(timeout=1.0)
            if carla_img: break
        except queue.Empty: pass

    if carla_img:
        arr = np.frombuffer(carla_img.raw_data, dtype=np.dtype("uint8"))
        arr = np.reshape(arr, (carla_img.height, carla_img.width, 4))
        frame_bgr = arr[:, :, :3].copy()

        K = get_camera_intrinsic_matrix(cam_w, cam_h, cam_fov)

        # Draw Matrix Trajectories in RED directly on image pixels
        for tid_str, r_info in raw_data.items():
            r_pts = r_info.get("carla_coordinates", [])
            if len(r_pts) >= 2:
                pixel_pts = []
                for pt in r_pts:
                    px = world_to_pixel(carla.Location(x=pt["x"], y=pt["y"], z=0.5), cam_tf, K)
                    if px: pixel_pts.append(px)

                if len(pixel_pts) >= 2:
                    pts_arr = np.array(pixel_pts, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame_bgr, [pts_arr], isClosed=False, color=(30, 30, 255), thickness=4, lineType=cv2.LINE_AA)
                    cv2.circle(frame_bgr, pixel_pts[0], 8, (0, 255, 0), -1) # Start Green
                    cv2.circle(frame_bgr, pixel_pts[-1], 8, (0, 0, 255), -1) # End Red
                    
                    label = f"V{tid_str}"
                    cv2.putText(frame_bgr, label, (pixel_pts[0][0] + 12, pixel_pts[0][1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        draw_overlay_legend(
            frame_bgr,
            f"TRAJECTORIES WITHOUT TIME METHOD (PURE MATRIX) - {os.path.basename(scenario_dir)}",
            [
                ("RED LINE = Pure Matrix Method Trajectory (Without Smoothing)", (30, 30, 255)),
                ("GREEN DOT = Trajectory Start Point", (0, 255, 0)),
                ("RED DOT = Trajectory End Point", (0, 0, 255)),
            ]
        )

        out_path = os.path.join(scenario_dir, "trajectory_without_time_method.png")
        cv2.imwrite(out_path, frame_bgr)
        print(f"  Saved {os.path.basename(out_path)}")

    try: camera.stop(); camera.destroy()
    except Exception: pass


def render_trajectory_image_with_time_method(client, world, scenario_dir, summary_data):
    """
    Renders time-variant smoothed trajectories (with time method) on real CARLA map using 3D-to-2D image projection.
    Output: trajectory_with_time_method.png
    """
    if not CARLA_AVAILABLE or not world:
        return

    time_path = os.path.join(scenario_dir, "time_variant_trajectories.json")
    if not os.path.exists(time_path):
        return

    with open(time_path, "r") as f:
        time_data = json.load(f)

    if not time_data:
        return

    all_coords = []
    for tid_str, t_info in time_data.items():
        t_pts = t_info.get("carla_coordinates", [])
        for pt in t_pts:
            all_coords.append((pt["x"], pt["y"]))

    if not all_coords:
        return

    coords_x = [c[0] for c in all_coords]
    coords_y = [c[1] for c in all_coords]
    min_x, max_x = min(coords_x), max(coords_x)
    min_y, max_y = min(coords_y), max(coords_y)
    avg_x, avg_y = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
    span = max(max_x - min_x, max_y - min_y, 40.0)

    cam_z = max(55.0, min(100.0, span * 1.3))

    bp_library = world.get_blueprint_library()
    cam_bp = bp_library.find("sensor.camera.rgb")
    cam_w, cam_h = 1920, 1080
    cam_fov = 90.0
    cam_bp.set_attribute("image_size_x", str(cam_w))
    cam_bp.set_attribute("image_size_y", str(cam_h))
    cam_bp.set_attribute("fov", str(cam_fov))

    cam_tf = carla.Transform(
        carla.Location(x=avg_x, y=avg_y, z=cam_z),
        carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
    )

    camera = world.spawn_actor(cam_bp, cam_tf)
    img_queue = queue.Queue()
    camera.listen(img_queue.put)

    carla_img = None
    for _ in range(12):
        world.tick()
        try:
            carla_img = img_queue.get(timeout=1.0)
            if carla_img: break
        except queue.Empty: pass

    if carla_img:
        arr = np.frombuffer(carla_img.raw_data, dtype=np.dtype("uint8"))
        arr = np.reshape(arr, (carla_img.height, carla_img.width, 4))
        frame_bgr = arr[:, :, :3].copy()

        K = get_camera_intrinsic_matrix(cam_w, cam_h, cam_fov)

        # Draw Time Method Trajectories in CYAN directly on image pixels
        for tid_str, t_info in time_data.items():
            t_pts = t_info.get("carla_coordinates", [])
            if len(t_pts) >= 2:
                pixel_pts = []
                for pt in t_pts:
                    px = world_to_pixel(carla.Location(x=pt["x"], y=pt["y"], z=0.5), cam_tf, K)
                    if px: pixel_pts.append(px)

                if len(pixel_pts) >= 2:
                    pts_arr = np.array(pixel_pts, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame_bgr, [pts_arr], isClosed=False, color=(255, 255, 0), thickness=4, lineType=cv2.LINE_AA)
                    cv2.circle(frame_bgr, pixel_pts[0], 8, (0, 255, 0), -1) # Start Green
                    cv2.circle(frame_bgr, pixel_pts[-1], 8, (0, 0, 255), -1) # End Red
                    
                    label = f"V{tid_str}"
                    cv2.putText(frame_bgr, label, (pixel_pts[0][0] + 12, pixel_pts[0][1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        draw_overlay_legend(
            frame_bgr,
            f"TRAJECTORIES WITH TIME METHOD (TIME-VARIANT) - {os.path.basename(scenario_dir)}",
            [
                ("CYAN LINE = Time Method Trajectory (With Time-Variant Smoothing)", (255, 255, 0)),
                ("GREEN DOT = Trajectory Start Point", (0, 255, 0)),
                ("RED DOT = Trajectory End Point", (0, 0, 255)),
            ]
        )

        out_path = os.path.join(scenario_dir, "trajectory_with_time_method.png")
        cv2.imwrite(out_path, frame_bgr)
        print(f"  Saved {os.path.basename(out_path)}")

    try: camera.stop(); camera.destroy()
    except Exception: pass


def render_spawn_positions_comparison_map(client, world, scenario_dir, summary_data):
    """
    Renders COMBINED visual photo comparing Raw Spawn Points (Red Dots) vs Retracted Spawn Points (Green Dots) on real CARLA map.
    Output: spawn_positions_comparison.png
    """
    if not CARLA_AVAILABLE or not world:
        return

    raw_path = os.path.join(scenario_dir, "spawn_positions_raw.json")
    retracted_path = os.path.join(scenario_dir, "spawn_positions_retracted.json")

    if not os.path.exists(raw_path) and not os.path.exists(retracted_path):
        return

    raw_data, retracted_data = {}, {}
    if os.path.exists(raw_path):
        with open(raw_path, "r") as f: raw_data = json.load(f)
    if os.path.exists(retracted_path):
        with open(retracted_path, "r") as f: retracted_data = json.load(f)

    for actor in world.get_actors().filter("vehicle.*"):
        try: actor.destroy()
        except Exception: pass

    all_spawns = list(raw_data.values()) + list(retracted_data.values())
    if not all_spawns:
        return

    coords_x = [s["x"] for s in all_spawns]
    coords_y = [s["y"] for s in all_spawns]
    min_x, max_x = min(coords_x), max(coords_x)
    min_y, max_y = min(coords_y), max(coords_y)
    avg_x, avg_y = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
    span = max(max_x - min_x, max_y - min_y, 35.0)

    cam_z = max(45.0, min(75.0, span * 1.2))

    bp_library = world.get_blueprint_library()
    cam_bp = bp_library.find("sensor.camera.rgb")
    cam_w, cam_h = 1920, 1080
    cam_fov = 90.0
    cam_bp.set_attribute("image_size_x", str(cam_w))
    cam_bp.set_attribute("image_size_y", str(cam_h))
    cam_bp.set_attribute("fov", str(cam_fov))

    cam_tf = carla.Transform(
        carla.Location(x=avg_x, y=avg_y, z=cam_z),
        carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
    )

    camera = world.spawn_actor(cam_bp, cam_tf)
    img_queue = queue.Queue()
    camera.listen(img_queue.put)

    carla_img = None
    for _ in range(12):
        world.tick()
        try:
            carla_img = img_queue.get(timeout=1.0)
            if carla_img: break
        except queue.Empty: pass

    if carla_img:
        arr = np.frombuffer(carla_img.raw_data, dtype=np.dtype("uint8"))
        arr = np.reshape(arr, (carla_img.height, carla_img.width, 4))
        frame_bgr = arr[:, :, :3].copy()

        K = get_camera_intrinsic_matrix(cam_w, cam_h, cam_fov)

        # Draw RAW SPAWNS (Red Dots)
        for tid_str, s_info in raw_data.items():
            loc_3d = carla.Location(x=s_info["x"], y=s_info["y"], z=0.5)
            pixel = world_to_pixel(loc_3d, cam_tf, K)
            if pixel:
                u, v = pixel
                if 0 <= u < cam_w and 0 <= v < cam_h:
                    cv2.circle(frame_bgr, (u, v), 11, (30, 30, 255), -1)
                    cv2.circle(frame_bgr, (u, v), 13, (255, 255, 255), 2)
                    cv2.putText(frame_bgr, f"V{tid_str} (Raw)", (u + 14, v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 255), 2, cv2.LINE_AA)

        # Draw RETRACTED SPAWNS (Green Dots) & Shift Lines
        for tid_str, s_info in retracted_data.items():
            loc_3d = carla.Location(x=s_info["x"], y=s_info["y"], z=0.5)
            pixel_ret = world_to_pixel(loc_3d, cam_tf, K)

            if pixel_ret:
                u, v = pixel_ret
                if 0 <= u < cam_w and 0 <= v < cam_h:
                    cv2.circle(frame_bgr, (u, v), 11, (30, 255, 30), -1)
                    cv2.circle(frame_bgr, (u, v), 13, (255, 255, 255), 2)
                    cv2.putText(frame_bgr, f"V{tid_str} (Retracted)", (u + 14, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 255, 80), 2, cv2.LINE_AA)

                    # Draw shift line from Raw to Retracted
                    if tid_str in raw_data:
                        r_info = raw_data[tid_str]
                        pixel_raw = world_to_pixel(carla.Location(x=r_info["x"], y=r_info["y"], z=0.5), cam_tf, K)
                        if pixel_raw:
                            cv2.line(frame_bgr, pixel_raw, pixel_ret, (0, 255, 255), 2, cv2.LINE_AA)

        draw_overlay_legend(
            frame_bgr,
            f"SPAWN POSITIONS COMPARISON (RAW VS RETRACTED) - {os.path.basename(scenario_dir)}",
            [
                ("RED DOT = Raw Spawn Position (Before Retraction)", (30, 30, 255)),
                ("GREEN DOT = Safety Retracted Spawn Position (After Retraction)", (30, 255, 30)),
                ("YELLOW LINE = Retraction Shift Vector", (0, 255, 255)),
            ]
        )

        out_path = os.path.join(scenario_dir, "spawn_positions_comparison.png")
        cv2.imwrite(out_path, frame_bgr)
        print(f"  Saved {os.path.basename(out_path)}")

    try: camera.stop(); camera.destroy()
    except Exception: pass


def render_trajectory_comparison_map(client, world, scenario_dir, summary_data):
    """
    Renders COMBINED visual photo comparing Matrix Method Trajectories (Red) vs Time Method Trajectories (Cyan) on real CARLA map.
    Output: trajectory_comparison.png
    """
    if not CARLA_AVAILABLE or not world:
        return

    raw_path = os.path.join(scenario_dir, "raw_matrix_trajectories.json")
    time_path = os.path.join(scenario_dir, "time_variant_trajectories.json")

    if not os.path.exists(raw_path) and not os.path.exists(time_path):
        return

    raw_data, time_data = {}, {}
    if os.path.exists(raw_path):
        with open(raw_path, "r") as f: raw_data = json.load(f)
    if os.path.exists(time_path):
        with open(time_path, "r") as f: time_data = json.load(f)

    all_coords = []
    for tid_str, r_info in raw_data.items():
        for pt in r_info.get("carla_coordinates", []):
            all_coords.append((pt["x"], pt["y"]))
    for tid_str, t_info in time_data.items():
        for pt in t_info.get("carla_coordinates", []):
            all_coords.append((pt["x"], pt["y"]))

    if not all_coords:
        return

    coords_x = [c[0] for c in all_coords]
    coords_y = [c[1] for c in all_coords]
    min_x, max_x = min(coords_x), max(coords_x)
    min_y, max_y = min(coords_y), max(coords_y)
    avg_x, avg_y = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
    span = max(max_x - min_x, max_y - min_y, 40.0)

    cam_z = max(55.0, min(100.0, span * 1.3))

    bp_library = world.get_blueprint_library()
    cam_bp = bp_library.find("sensor.camera.rgb")
    cam_w, cam_h = 1920, 1080
    cam_fov = 90.0
    cam_bp.set_attribute("image_size_x", str(cam_w))
    cam_bp.set_attribute("image_size_y", str(cam_h))
    cam_bp.set_attribute("fov", str(cam_fov))

    cam_tf = carla.Transform(
        carla.Location(x=avg_x, y=avg_y, z=cam_z),
        carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
    )

    camera = world.spawn_actor(cam_bp, cam_tf)
    img_queue = queue.Queue()
    camera.listen(img_queue.put)

    carla_img = None
    for _ in range(12):
        world.tick()
        try:
            carla_img = img_queue.get(timeout=1.0)
            if carla_img: break
        except queue.Empty: pass

    if carla_img:
        arr = np.frombuffer(carla_img.raw_data, dtype=np.dtype("uint8"))
        arr = np.reshape(arr, (carla_img.height, carla_img.width, 4))
        frame_bgr = arr[:, :, :3].copy()

        K = get_camera_intrinsic_matrix(cam_w, cam_h, cam_fov)

        # Draw Matrix Trajectories in RED
        for tid_str, r_info in raw_data.items():
            r_pts = r_info.get("carla_coordinates", [])
            if len(r_pts) >= 2:
                pixel_pts = []
                for pt in r_pts:
                    px = world_to_pixel(carla.Location(x=pt["x"], y=pt["y"], z=0.5), cam_tf, K)
                    if px: pixel_pts.append(px)
                if len(pixel_pts) >= 2:
                    pts_arr = np.array(pixel_pts, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame_bgr, [pts_arr], isClosed=False, color=(30, 30, 255), thickness=3, lineType=cv2.LINE_AA)

        # Draw Time Method Trajectories in CYAN
        for tid_str, t_info in time_data.items():
            t_pts = t_info.get("carla_coordinates", [])
            if len(t_pts) >= 2:
                pixel_pts = []
                for pt in t_pts:
                    px = world_to_pixel(carla.Location(x=pt["x"], y=pt["y"], z=0.5), cam_tf, K)
                    if px: pixel_pts.append(px)
                if len(pixel_pts) >= 2:
                    pts_arr = np.array(pixel_pts, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame_bgr, [pts_arr], isClosed=False, color=(255, 255, 0), thickness=3, lineType=cv2.LINE_AA)
                    cv2.putText(frame_bgr, f"V{tid_str}", (pixel_pts[0][0] + 12, pixel_pts[0][1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        draw_overlay_legend(
            frame_bgr,
            f"TRAJECTORY COMPARISON (MATRIX VS TIME METHOD) - {os.path.basename(scenario_dir)}",
            [
                ("RED LINE = Matrix Method Trajectory (Without Time Method)", (30, 30, 255)),
                ("CYAN LINE = Time Method Trajectory (With Time-Variant Smoothing)", (255, 255, 0)),
            ]
        )

        out_path = os.path.join(scenario_dir, "trajectory_comparison.png")
        cv2.imwrite(out_path, frame_bgr)
        print(f"  Saved {os.path.basename(out_path)}")

    try: camera.stop(); camera.destroy()
    except Exception: pass


SIM_MODES = ("expert", "pid", "pid_avoid")
SETTLE_TICKS = 20  # 1 s at 20 Hz before the scenario clock starts
VIDEO_NAMES = {
    "expert": "trajectory_video_expert_replay.mp4",
    "pid": "trajectory_video_without_avoidance.mp4",
    "pid_avoid": "trajectory_video_with_avoidance.mp4",
}


def build_timed_route(tid_str, t_data, start_frame):
    """(times_s, xy, speeds_mps) from the Stage 2 trajectory, keyed by real frame number (gaps allowed)."""
    pts = t_data.get(tid_str, {}).get("carla_coordinates", [])
    if len(pts) < 2:
        return None
    times = np.array([(p["frame"] - start_frame) / REAL_WORLD_FPS for p in pts])
    xy = np.array([(p["x"], p["y"]) for p in pts])
    # Central difference over +-2 frames: speed consistent with the CARLA path the car must follow
    k = 2
    speeds = []
    for i in range(len(xy)):
        lo, hi = max(0, i - k), min(len(xy) - 1, i + k)
        dt = times[hi] - times[lo]
        speeds.append(float(np.linalg.norm(xy[hi] - xy[lo]) / dt) if dt > 0 else 0.0)
    return times, xy, np.array(speeds)


def render_carla_simulation_video(client, world, scenario_dir, summary_path, summary_data, annotation_parser,
                                  dji_spawn_params, enable_avoidance, mode=None, render=True, camera_z=55.0):
    """
    Replays a scenario in CARLA and (optionally) records a top-down video.

    mode:
      "expert"    - kinematic replay: physics off, set_transform to the real position every tick.
                    Any box overlap here is in the data itself, not caused by the controller.
      "pid"       - PID tracking of the time-indexed real trajectory, no avoidance.
      "pid_avoid" - same, plus box-based follower avoidance.
    If mode is None it follows enable_avoidance (pid_avoid / pid), as the existing callers expect.
    camera_z is the height (m) of the bird's-eye camera above the middle of the routes.

    Returns a metrics dict (also saved as sim_metrics_<mode>.json) or None if nothing spawned.
    """
    if not CARLA_AVAILABLE:
        print("  Skipping CARLA video render: carla library not available.")
        return None

    mode = mode or ("pid_avoid" if enable_avoidance else "pid")
    assert mode in SIM_MODES, mode
    out_video_name = VIDEO_NAMES[mode]
    output_video_path = os.path.join(scenario_dir, out_video_name)
    print(f"  Replaying in CARLA ({mode})" + (f" -> {out_video_name}..." if render else " (headless)..."))

    # Every mode uses the same inputs (retracted spawns + time-variant trajectories)
    spawns_path = os.path.join(scenario_dir, "spawn_positions_retracted.json")
    traj_path = os.path.join(scenario_dir, "time_variant_trajectories.json")
    if not os.path.exists(spawns_path):
        spawns_path = os.path.join(scenario_dir, "spawn_positions_raw.json")
    if not os.path.exists(spawns_path):
        print("  Error: No spawn position JSON found to spawn vehicles.")
        return None

    with open(spawns_path, "r") as f:
        spawns_data = json.load(f)
    t_data = {}
    if os.path.exists(traj_path):
        with open(traj_path, "r") as tf_file:
            t_data = json.load(tf_file)

    start_frame, end_frame, vehicle_ids, highlight_ids, ego_id = summary_data
    scenario_duration_sec = (end_frame - start_frame + 1) / REAL_WORLD_FPS
    total_sim_ticks = int(scenario_duration_sec * SIM_FPS)
    dt = 1.0 / SIM_FPS

    for actor in list(world.get_actors().filter("vehicle.*")) + list(world.get_actors().filter("sensor.*")):
        try: actor.destroy()
        except Exception: pass

    blueprint_library = world.get_blueprint_library()
    vehicles = {}
    spawn_locations = []

    for tid_str, s_info in spawns_data.items():
        tid = int(tid_str)
        route = build_timed_route(tid_str, t_data, start_frame)
        if route is None:
            print(f"  Skipping vehicle_{tid}: no Stage 2 trajectory")
            continue
        bp = blueprint_library.find(s_info.get("blueprint", "vehicle.nissan.micra"))
        vname = f"vehicle_{tid}"
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", vname)

        # Heading from the path (see run_stage_3_spawns); also corrects spawn JSONs written before that fix
        yaw = path_yaw_deg([tuple(p) for p in route[1][:8]])
        yaw = s_info["yaw"] if yaw is None else yaw
        try:
            sim_v = CapturingVehicle(world, bp, vname, (s_info["x"], s_info["y"], s_info["z"]), yaw,
                                     scenario_dir, enable_avoidance=(mode == "pid_avoid"))
        except Exception as e:
            print(f"  Failed to spawn {vname}: {e}")
            continue
        sim_v.set_timed_route(*route, z=DEFAULT_Z)
        vehicles[vname] = sim_v
        spawn_locations.append((s_info["x"], s_info["y"]))

    if not vehicles:
        print("  Warning: No vehicles could be spawned in CARLA.")
        return None

    if mode == "expert":
        for v in vehicles.values():
            v.vehicle.set_simulate_physics(False)
        world.tick()
    else:
        # Let the cars land and engage 1st gear before the clock starts, then launch them at their
        # real speed. Without this they lost ~1.5 s of a ~4 s scenario sitting in gear 0.
        for v in vehicles.values():
            v.hold_in_gear()
        for _ in range(SETTLE_TICKS):
            world.tick()
        for v in vehicles.values():
            v.set_initial_speed(float(v.timed_route[2][0]))

    camera, img_queue, writer = None, None, None
    cam_width, cam_height = 1280, 720
    if render:
        # Centre the bird's-eye camera on the middle of all routes (not just the spawns)
        route_xy = np.vstack([v.timed_route[1] for v in vehicles.values()])
        avg_x, avg_y = route_xy.mean(axis=0)
        cam_bp = blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(cam_width))
        cam_bp.set_attribute("image_size_y", str(cam_height))
        cam_bp.set_attribute("fov", "90")
        cam_tf = carla.Transform(carla.Location(x=float(avg_x), y=float(avg_y), z=camera_z), carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0))
        camera = world.spawn_actor(cam_bp, cam_tf)
        img_queue = queue.Queue()
        camera.listen(img_queue.put)
        writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), float(SIM_FPS), (cam_width, cam_height))
        K_video = get_camera_intrinsic_matrix(cam_width, cam_height, 90.0)
        for _ in range(5):
            world.tick()
            try: img_queue.get(timeout=1.0)
            except queue.Empty: pass

    sim_clock = 0.0
    recorded_frames = 0
    min_gap, overlap_ticks, avoidance_ticks = float("inf"), 0, 0
    names = list(vehicles)

    for step in range(total_sim_ticks):
        for v in vehicles.values():
            if mode == "expert":
                x, y, _, _ = v.reference(sim_clock)
                xa, ya, _, _ = v.reference(sim_clock + 0.1)
                xb, yb, _, _ = v.reference(max(0.0, sim_clock - 0.1))
                yaw_e = math.degrees(math.atan2(ya - yb, xa - xb)) if math.hypot(xa - xb, ya - yb) > 1e-3 \
                    else v.vehicle.get_transform().rotation.yaw
                v.vehicle.set_transform(carla.Transform(carla.Location(x=x, y=y, z=DEFAULT_Z), carla.Rotation(yaw=yaw_e)))
            else:
                v.update(sim_clock, vehicles)
                avoidance_ticks += 1 if v.last_avoidance_event else 0

        world.tick()
        sim_clock += dt

        # Box-gap bookkeeping works in every mode (the collision sensor needs physics)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                g = box_gap(actor_box(vehicles[names[i]].vehicle), actor_box(vehicles[names[j]].vehicle))
                min_gap = min(min_gap, g)
                overlap_ticks += 1 if g < 0.0 else 0

        if not render:
            continue
        try:
            carla_img = img_queue.get(timeout=2.0)
        except queue.Empty:
            continue
        arr = np.frombuffer(carla_img.raw_data, dtype=np.dtype("uint8"))
        frame_bgr = np.reshape(arr, (carla_img.height, carla_img.width, 4))[:, :, :3].copy()

        banner_text, banner_bg_color = {
            "expert": ("EXPERT REPLAY (KINEMATIC, NO PHYSICS)", (160, 90, 30)),
            "pid": ("COLLISION AVOIDANCE: DISABLED", (30, 30, 200)),
            "pid_avoid": ("COLLISION AVOIDANCE: ENABLED", (30, 160, 30)),
        }[mode]
        cv2.rectangle(frame_bgr, (20, 20), (560, 65), banner_bg_color, -1)
        cv2.rectangle(frame_bgr, (20, 20), (560, 65), (255, 255, 255), 2)
        cv2.putText(frame_bgr, banner_text, (32, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        for vname, vobj in vehicles.items():
            if not (vobj.vehicle and vobj.vehicle.is_alive):
                continue
            px = world_to_pixel(vobj.vehicle.get_location(), cam_tf, K_video)
            if not (px and 0 <= px[0] < cam_width and 0 <= px[1] < cam_height):
                continue
            u, v = px
            if vobj.has_collided:
                # Drawn in every mode; the old code hid crashes in the with-avoidance video
                cv2.circle(frame_bgr, (u, v), 14, (0, 0, 255), -1)
                cv2.circle(frame_bgr, (u, v), 18, (0, 0, 255), 3)
                cv2.putText(frame_bgr, f"{vname} [CRASH / IMPACT]", (u + 16, v + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
            elif vobj.last_avoidance_event:
                cv2.circle(frame_bgr, (u, v), 14, (0, 255, 255), -1)
                cv2.circle(frame_bgr, (u, v), 18, (0, 255, 255), 3)
                cv2.putText(frame_bgr, f"{vname} [AVOIDANCE ACTIVE]", (u + 16, v + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                color = (255, 255, 0) if mode == "pid_avoid" else (200, 200, 200)
                cv2.circle(frame_bgr, (u, v), 7, color, -1)
                cv2.putText(frame_bgr, vname, (u + 10, v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(frame_bgr)
        recorded_frames += 1

    # Vehicle-vehicle collisions counted once per pair; static hits (kerbs, signs, vegetation) separately
    collided_pairs, static_hits = set(), 0
    for vname, v in vehicles.items():
        types = {}
        for _, other_id, other_type in v.collision_events:
            if other_type.startswith("vehicle."):
                collided_pairs.add(frozenset((v.vehicle.id, other_id)))
            else:
                types[other_type] = True
        static_hits += len(types)
    metrics = {
        "mode": mode,
        "vehicles": len(vehicles),
        "collision_events": len(collided_pairs),
        "static_collisions": static_hits,
        "min_box_gap_m": round(min_gap, 3) if min_gap != float("inf") else None,
        "box_overlap_ticks": overlap_ticks,
        "avoidance_ticks": avoidance_ticks,
        # Progress of near-stationary vehicles (route < 3 m) is dominated by noise, so they are left out
        "min_progress": round(min([v.progress() for v in vehicles.values() if v.route_length() >= 3.0] or [1.0]), 3),
        "sim_ticks": total_sim_ticks,
    }
    if metrics["min_box_gap_m"] is None:
        metrics["min_box_gap_m"] = float("nan")
    with open(os.path.join(scenario_dir, f"sim_metrics_{mode}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if writer:
        writer.release()
    if camera:
        try: camera.stop(); camera.destroy()
        except Exception: pass
    for v in vehicles.values():
        try: v.destroy()
        except Exception: pass
    world.tick()  # flush the destroys so the next run's actor list doesn't contain them

    if render:
        size_mb = os.path.getsize(output_video_path) / (1024 * 1024) if os.path.exists(output_video_path) else 0.0
        print(f"  Saved {out_video_name}: {recorded_frames} frames, {size_mb:.2f} MB")
    print(f"  Metrics ({mode}): {metrics}")
    return metrics


def main():
    args = parse_arguments()
    scenario_dirs = resolve_scenario_paths(args.scenarios_dir, args.target, args.all)

    if not scenario_dirs:
        print("No matching scenario directories found.")
        return

    if args.stages == "all":
        active_stages = [1, 2, 3, 4]
    else:
        active_stages = [int(s.strip()) for s in args.stages.split(",") if s.strip().isdigit()]

    print(f"\n=======================================================")
    print(f"Batch Processing {len(scenario_dirs)} scenario(s)")
    print(f"Active Stages: {active_stages}")
    print(f"=======================================================\n")

    print(f"Loading annotation data from {args.json_file}...")
    annotation_parser = AnnotationParser(args.json_file)
    dji_spawn_params = load_dji_spawn_params(DEFAULT_DJI_DATA) if os.path.exists(DEFAULT_DJI_DATA) else {}

    world = None
    client = None
    if CARLA_AVAILABLE and (4 in active_stages or 3 in active_stages):
        try:
            print(f"Connecting to CARLA simulator on {CARLA_HOST}:{CARLA_PORT}...")
            client = carla.Client(CARLA_HOST, CARLA_PORT)
            client.set_timeout(10.0)
            world = client.get_world()
            if TOWN not in world.get_map().name:
                print(f"Loading CARLA world {TOWN}...")
                world = client.load_world(TOWN)
                time.sleep(2)

            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / SIM_FPS
            world.apply_settings(settings)
        except Exception as e:
            print(f"Notice: Could not connect to CARLA server ({e}).")

    for idx, sc_dir in enumerate(scenario_dirs, 1):
        rel_path = os.path.relpath(sc_dir, args.scenarios_dir)
        print(f"[{idx}/{len(scenario_dirs)}] Processing Scenario: {rel_path}")

        summary_files = glob.glob(os.path.join(sc_dir, "*_summary.txt"))
        if not summary_files:
            print(f"  Warning: No summary file found in {sc_dir}. Skipping.")
            continue

        summary_path = summary_files[0]
        try:
            summary_data = parse_summary_file(summary_path)
        except Exception as e:
            print(f"  Error parsing summary file {summary_path}: {e}")
            continue

        # Stage 1: Matrix conversion without time method
        if 1 in active_stages:
            run_stage_1_matrix(sc_dir, summary_data, annotation_parser)

        # Stage 2: Time method conversion
        if 2 in active_stages:
            run_stage_2_time_method(sc_dir, summary_data, annotation_parser)

        # Stage 3: Spawn extraction
        if 3 in active_stages:
            run_stage_3_spawns(sc_dir, summary_data, annotation_parser, dji_spawn_params, world=world)

        # Stage 4: Render photos on CARLA map & trajectory simulation videos
        if 4 in active_stages:
            if world and client:
                render_spawn_positions_without_retraction_map(client, world, sc_dir, summary_data)
                render_spawn_positions_with_retraction_map(client, world, sc_dir, summary_data)
                render_spawn_positions_comparison_map(client, world, sc_dir, summary_data)
                render_trajectory_image_without_time_method(client, world, sc_dir, summary_data)
                render_trajectory_image_with_time_method(client, world, sc_dir, summary_data)
                render_trajectory_comparison_map(client, world, sc_dir, summary_data)

                render_carla_simulation_video(
                    client, world, sc_dir, summary_path, summary_data,
                    annotation_parser, dji_spawn_params, enable_avoidance=False
                )
                render_carla_simulation_video(
                    client, world, sc_dir, summary_path, summary_data,
                    annotation_parser, dji_spawn_params, enable_avoidance=True
                )

        print("")

    print("Batch processing complete!")


if __name__ == "__main__":
    main()
