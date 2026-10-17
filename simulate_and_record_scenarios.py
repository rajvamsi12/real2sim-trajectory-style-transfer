# OpenCV (cv2) MUST be imported BEFORE carla to avoid C++ shared library symbol conflicts!
import cv2
import numpy as np
import os
import sys
import glob
import json
import csv
import math
import time
import carla
import weakref
import random
import shutil
import logging
import re
import select
import queue
from functools import reduce
from itertools import combinations

# Add CARLA egg path if needed (only for legacy Python < 3.8)
if sys.version_info < (3, 8):
    egg_path = "/home/raj/CARLA_0.9.15/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
    if os.path.exists(egg_path) and egg_path not in sys.path:
        sys.path.insert(0, egg_path)

# Import sibling modules from this checkout (not a hardcoded path, so branches/worktrees use their own code)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from overlay_scenario_bboxes import parse_summary_file
from timemethod import (
    transform_trajectory_with_analysis,
    classify_point,
    transformation_matrices,
)
from scc import SimVehicle, DummyWaypoint
from scenario_checks import box_corners, box_gap, path_yaw_deg

# --- Configuration (Identical to scenario_new4.py) ---
SIM_FPS = 20

STABILIZE_EKF_CSV = os.path.join(SCRIPT_DIR, "stabilize_ekf_final2.csv")
INPUT_FILE = STABILIZE_EKF_CSV if os.path.exists(STABILIZE_EKF_CSV) else os.path.join(SCRIPT_DIR, "default.json")
TIDS_FILE = os.path.join(SCRIPT_DIR, "tids.txt")
DJI_DATA_FILE = os.path.join(SCRIPT_DIR, "dji0003_data.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "_scenario_output")
SCENARIOS_DIR = os.path.join(SCRIPT_DIR, "Scenariosssss")

CARLA_HOST = "localhost"
CARLA_PORT = 2000
TOWN = "Town10HD"

VEHICLE_BLUEPRINT = "vehicle.bmw.grandtourer"
DEFAULT_Z = 0.6

REAL_WORLD_FPS = 4.27
SPEED_CORRECTION_FACTOR = 1.0

# --- Spawn Retraction Configuration (Enhanced for tight Cut-In/Overtaking scenarios) ---
SPAWN_SAFETY_MARGIN = 2.5
MAX_RETRACTION_ATTEMPTS = 25
RETRACTION_STEP = 0.5
SNAP_FALLBACK_DISTANCE = 1.5
LATERAL_SHIFT_OPTIONS = [2.0, -2.0, 4.0, -4.0]
Z_ELEVATION_OFFSETS = [0.0, 0.2, 0.5, 1.0, 1.5, 2.0]

# --- Real-time Avoidance Configuration ---
# A vehicle only reacts to a predicted bounding-box conflict with a vehicle AHEAD of it
# (the follower yields), so a clean overtake with real clearance is left untouched.
AVOIDANCE_HORIZON_S = 3.0          # constant-velocity prediction horizon
AVOIDANCE_STEP_S = 0.1
AVOIDANCE_MIN_GAP_M = 0.4          # predicted box gap below this counts as a conflict
AVOIDANCE_IMMINENT_TTC_S = 1.5     # conflict sooner than this -> full brake, PID skipped
AVOIDANCE_SEARCH_RADIUS_M = 30.0
AVOIDANCE_NUDGE_M = 0.5            # lateral offset away from the threat, only if still on a driving lane
AVOIDANCE_LOG_INTERVAL_S = 1.0

# --- Vehicle Label Mapping (DJI_003 Dataset Rule: Label 4 is Pedestrian -> EXCLUDED) ---
LABEL_BLUEPRINT_MAP = {
    0: "vehicle.micro.microlino",    # 3-wheeler (Real avg length: 2.11m, CARLA: 2.21m)
    1: "vehicle.nissan.micra",       # 4-wheeler (Real avg length: 2.70m, CARLA: 3.63m)
    2: "vehicle.kawasaki.ninja",     # 2-wheeler (Real avg length: 1.90m, CARLA: 2.04m)
    3: "vehicle.volkswagen.t2",      # Bus (CARLA: 4.48m)
    4: None,                         # Pedestrian -> STRICTLY EXCLUDED FROM VEHICLE SCENARIOS
    5: "vehicle.mercedes.sprinter"   # Truck (Real avg length: 3.90m, CARLA: 5.92m)
}

# Real Dataset Dimension Override Table (Radius = Half Length)
APPROX_BLUEPRINT_CLEARANCE_RADII = {
    "vehicle.kawasaki.ninja": 0.95,    # 1.90m / 2 (Real Motorcycle length)
    "vehicle.micro.microlino": 1.05,   # 2.11m / 2 (Real 3-wheeler length)
    "vehicle.nissan.micra": 1.35,      # 2.70m / 2 (Real 4-wheeler car length override)
    "vehicle.bmw.grandtourer": 1.35,   # 2.70m / 2 (Real 4-wheeler car length override)
    "vehicle.mercedes.sprinter": 1.95, # 3.90m / 2 (Real Truck length)
    "vehicle.volkswagen.t2": 2.24,     # 4.48m / 2 (Real Bus length)
}


def get_existing_vehicle_clearance_radius(vehicle):
    if not vehicle or not hasattr(vehicle, 'bounding_box'):
        return 2.0
    extent = vehicle.bounding_box.extent
    return math.hypot(extent.x, extent.y)


def get_candidate_clearance_radius(blueprint_name):
    return APPROX_BLUEPRINT_CLEARANCE_RADII.get(blueprint_name, 2.5)


def normalize_angle_deg(angle):
    return ((angle + 180.0) % 360.0) - 180.0


def angular_error_deg(a, b):
    return abs(normalize_angle_deg(a - b))


def convert_real_heading_to_carla_yaw(real_heading_deg, real_point, trajectory_yaw_hint=None):
    if real_heading_deg is None or real_point is None:
        return None

    region = classify_point(real_point)
    if region not in transformation_matrices:
        return None

    M = transformation_matrices[region]
    A = np.array([[M[0][0], M[0][1]], [M[1][0], M[1][1]]], dtype=float)

    theta = math.radians(real_heading_deg)
    c, s = math.cos(theta), math.sin(theta)

    candidate_dirs = [
        np.array([c, s], dtype=float),
        np.array([c, -s], dtype=float),
        np.array([s, c], dtype=float),
        np.array([s, -c], dtype=float),
    ]

    candidate_yaws = []
    for direction in candidate_dirs:
        mapped = A @ direction
        if np.linalg.norm(mapped) < 1e-9:
            continue
        candidate_yaws.append(math.degrees(math.atan2(mapped[1], mapped[0])))

    if not candidate_yaws:
        return None
    if trajectory_yaw_hint is None:
        return normalize_angle_deg(candidate_yaws[0])

    best = min(candidate_yaws, key=lambda y: angular_error_deg(y, trajectory_yaw_hint))
    return normalize_angle_deg(best)


def actor_box(actor, t=0.0):
    """Bounding-box corners of a vehicle t seconds ahead, assuming constant velocity."""
    tf = actor.get_transform()
    vel = actor.get_velocity()
    ext = actor.bounding_box.extent
    center = (tf.location.x + vel.x * t, tf.location.y + vel.y * t)
    return box_corners(center, math.radians(tf.rotation.yaw), 2.0 * ext.x, 2.0 * ext.y)


def predict_box_conflict(ego, other, horizon=AVOIDANCE_HORIZON_S, step=AVOIDANCE_STEP_S):
    """First time (s) within the horizon at which the predicted boxes come closer than AVOIDANCE_MIN_GAP_M."""
    t = 0.0
    while t <= horizon + 1e-9:
        if box_gap(actor_box(ego, t), actor_box(other, t)) < AVOIDANCE_MIN_GAP_M:
            return t
        t += step
    return None


def is_ahead_of(ego, other):
    """True if `other` is in front of `ego` along ego's heading (ego is then the follower and yields)."""
    fwd = ego.get_transform().get_forward_vector()
    rel = other.get_location() - ego.get_location()
    return rel.x * fwd.x + rel.y * fwd.y > 0.0


def find_primary_threat(world, ego):
    """(ttc_s, actor) of the soonest predicted conflict with a vehicle ahead, or (None, None)."""
    ego_loc = ego.get_location()
    best = (None, None)
    for a in world.get_actors().filter("vehicle.*"):
        if a.id == ego.id or not a.is_alive:
            continue
        if ego_loc.distance(a.get_location()) > AVOIDANCE_SEARCH_RADIUS_M or not is_ahead_of(ego, a):
            continue
        ttc = predict_box_conflict(ego, a)
        if ttc is not None and (best[0] is None or ttc < best[0]):
            best = (ttc, a)
    return best


def nudge_offset(world, ego, threat):
    """Signed lateral offset (m, + = left) away from the threat, or 0.0 if that would leave the driving lanes."""
    tf = ego.get_transform()
    fwd = tf.get_forward_vector()
    left = carla.Vector3D(-fwd.y, fwd.x, 0.0)
    rel = threat.get_location() - tf.location
    offset = -AVOIDANCE_NUDGE_M if (rel.x * left.x + rel.y * left.y) > 0 else AVOIDANCE_NUDGE_M
    probe = carla.Location(tf.location.x + left.x * offset * 2.0, tf.location.y + left.y * offset * 2.0, tf.location.z)
    # project_to_road=False: None means the probe is off the road; we do NOT snap to a lane centre
    wp = world.get_map().get_waypoint(probe, project_to_road=False, lane_type=carla.LaneType.Driving)
    return offset if wp is not None else 0.0


def get_backward_vector(yaw_degrees):
    yaw_rad = math.radians(yaw_degrees)
    return carla.Vector3D(x=-math.cos(yaw_rad), y=-math.sin(yaw_rad), z=0.0)


def get_right_vector(yaw_degrees):
    yaw_rad = math.radians(yaw_degrees)
    return carla.Vector3D(x=-math.sin(yaw_rad), y=math.cos(yaw_rad), z=0.0)


def make_spawn_transform(location, yaw):
    return carla.Transform(location, carla.Rotation(yaw=yaw))


def snap_to_driving_transform(world, location, fallback_yaw):
    waypoint = world.get_map().get_waypoint(
        location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    if waypoint is None:
        return make_spawn_transform(location, fallback_yaw)
    wp_loc = waypoint.transform.location
    snapped_loc = carla.Location(x=wp_loc.x, y=wp_loc.y, z=location.z)
    return make_spawn_transform(snapped_loc, fallback_yaw)


# --- Trajectory-based Retraction (Identical to scenario_new4.py) ---
def build_spawn_search_transforms(world, base_spawn_point, initial_yaw, annotation_parser=None, tid=None, interaction_frame=None):
    backward = get_backward_vector(initial_yaw)
    right = get_right_vector(initial_yaw)
    raw_candidates = []
    seen_raw = set()

    def add_raw_candidate(location, yaw, strategy):
        for z_off in Z_ELEVATION_OFFSETS:
            loc = carla.Location(x=location.x, y=location.y, z=location.z + z_off)
            key = (
                round(loc.x, 2),
                round(loc.y, 2),
                round(loc.z, 2),
                round(yaw, 2),
            )
            if key in seen_raw:
                continue
            seen_raw.add(key)
            raw_candidates.append((make_spawn_transform(loc, yaw), f"{strategy}_z{z_off:.1f}"))

    # 1. Base Spawn Point
    add_raw_candidate(base_spawn_point, initial_yaw, "original")

    # 2. TRAJECTORY-BASED RETRACTION: Retract backward along actual historical dataset waypoints in default.json
    if annotation_parser is not None and tid is not None and interaction_frame is not None:
        if tid in annotation_parser.track_data:
            traj_dict = annotation_parser.track_data[tid]["trajectory"]
            prior_frames = sorted([f for f in traj_dict.keys() if f < interaction_frame], reverse=True)
            for p_frame in prior_frames[:15]:
                p_pt = traj_dict[p_frame]
                next_pt = traj_dict.get(p_frame + 1, p_pt)
                carla_p_pts = transform_trajectory_with_analysis([p_pt, next_pt])
                if len(carla_p_pts) >= 1 and carla_p_pts[0] is not None:
                    p_loc = carla.Location(x=carla_p_pts[0][0], y=carla_p_pts[0][1], z=base_spawn_point.z)
                    p_yaw = estimate_trajectory_yaw([pt for pt in carla_p_pts if pt is not None]) if len([pt for pt in carla_p_pts if pt is not None]) >= 2 else initial_yaw
                    add_raw_candidate(p_loc, p_yaw, f"traj_retract_frame_{p_frame}")

    # 3. Vector-based Backward Retraction
    for i in range(MAX_RETRACTION_ATTEMPTS):
        dist = (i + 1) * RETRACTION_STEP
        add_raw_candidate(
            base_spawn_point + backward * dist,
            initial_yaw,
            f"retract_{dist:.1f}m",
        )

    # 4. Lateral Shifts (+2.0m, -2.0m, +4.0m, -4.0m) + Retraction
    for shift in LATERAL_SHIFT_OPTIONS:
        shifted_base = base_spawn_point + right * shift
        add_raw_candidate(
            shifted_base,
            initial_yaw,
            f"lateral_{shift:+.1f}m",
        )
        for i in range(MAX_RETRACTION_ATTEMPTS):
            dist = (i + 1) * RETRACTION_STEP
            add_raw_candidate(
                shifted_base + backward * dist,
                initial_yaw,
                f"lateral_{shift:+.1f}m_retract_{dist:.1f}m",
            )

    # 5. Driving Road Snapping
    all_candidates = list(raw_candidates)
    seen_snapped = set()
    for raw_transform, strategy in raw_candidates:
        snapped_transform = snap_to_driving_transform(
            world,
            raw_transform.location,
            raw_transform.rotation.yaw,
        )
        key = (
            round(snapped_transform.location.x, 2),
            round(snapped_transform.location.y, 2),
            round(snapped_transform.location.z, 2),
            round(snapped_transform.rotation.yaw, 2),
        )
        if key in seen_snapped:
            continue
        seen_snapped.add(key)
        all_candidates.append((snapped_transform, f"{strategy}_snap"))

    return all_candidates


class AnnotationParser:
    def __init__(self, filepath):
        self.track_data = {}
        self.frame_data = {}
        self.load_data(filepath)

    def load_data(self, filepath):
        if not os.path.exists(filepath):
            logging.error(f"Annotation file not found at {filepath}")
            return

        if filepath.endswith(".csv"):
            import pandas as pd
            print(f"Loading EKF dataset from CSV: {filepath}...")
            df = pd.read_csv(filepath)
            for row in df.itertuples():
                tid = int(row.trackID)
                fr_str = str(row.frameID)
                fr = int(fr_str.split('_')[-1]) if 'frame_' in fr_str else int(fr_str)
                label_id = int(row.labelID)

                if tid not in self.track_data:
                    self.track_data[tid] = {
                        "label_id": label_id,
                        "trajectory": {},
                        "headings": {},
                        "speeds": {},
                    }

                cx = float(row.kf_cx_pixel)
                cy = -float(row.kf_cy_pixel)
                self.track_data[tid]["trajectory"][fr] = (cx, cy)
                self.track_data[tid]["headings"][fr] = float(row.heading_angle)
                self.track_data[tid]["speeds"][fr] = float(row.linear_speed_meter_per_sec)

                if fr not in self.frame_data:
                    self.frame_data[fr] = []
                self.frame_data[fr].append(tid)
            print(f"Loaded {len(self.track_data)} track IDs from EKF CSV dataset.")
        else:
            with open(filepath, "r") as f:
                data = json.load(f)

            for frame_idx, item in enumerate(data.get("items", [])):
                self.frame_data[frame_idx] = []
                for ann in item.get("annotations", []):
                    attributes = ann.get("attributes", {})
                    tid = attributes.get("track_id")
                    label_id = ann.get("label_id")

                    if tid is not None:
                        if tid not in self.track_data:
                            self.track_data[tid] = {
                                "label_id": label_id,
                                "trajectory": {},
                                "headings": {},
                                "speeds": {},
                            }

                        bbox = ann.get("bbox")
                        if bbox and len(bbox) == 4:
                            x, y, w, h = bbox
                            center_x = x + w / 2.0
                            center_y = -(y + h / 2.0)
                            self.track_data[tid]["trajectory"][frame_idx] = (center_x, center_y)

                        self.frame_data[frame_idx].append(tid)

    def get_sliced_trajectory_coords(self, tid, start_frame, num_frames):
        coords = []
        if tid in self.track_data:
            end_frame = start_frame + num_frames
            for fr in range(start_frame, end_frame):
                if fr in self.track_data[tid]["trajectory"]:
                    coords.append(self.track_data[tid]["trajectory"][fr])
        return coords

    def get_sliced_trajectory_frames(self, tid, start_frame, num_frames):
        frames = []
        if tid in self.track_data:
            sorted_frames = sorted(self.track_data[tid]["trajectory"].keys())
            end_frame = start_frame + num_frames
            frames = [fr for fr in sorted_frames if start_frame <= fr < end_frame]
        return frames

    def get_label_for_track(self, tid):
        if tid in self.track_data:
            return self.track_data[tid].get("label_id")
        return None


class CapturingVehicle(SimVehicle):
    def __init__(self, world, bp, name, start, yaw, output_path, start_delay=0, start_capture_time=0, end_capture_time=float("inf"), enable_avoidance=True):
        # CARLA's default limits (brake 0.3, throttle 0.75) can neither stop a car within the avoidance
        # horizon nor match the real accelerations, so the replay uses the full range
        super().__init__(world, bp, name, start, yaw, start_delay, max_brake=1.0, max_throttle=1.0)
        self.enable_avoidance = enable_avoidance
        self.collision_sensor = None
        self.collision_events = []          # (sim_time, other_actor_id, other_type_id), every event
        self.last_avoidance_event = None
        self._sim_time = 0.0
        self._last_avoidance_log_time = -1.0
        self._setup_sensors()

    @property
    def has_collided(self):
        return bool(self.collision_events)

    def _setup_sensors(self):
        collision_bp = self.world.get_blueprint_library().find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: CapturingVehicle._on_collision(weakref.ref(self), event))

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self or not self.collision_sensor or not self.collision_sensor.is_alive:
            return
        if not self.collision_events:
            logging.warning(f"!!! Collision DETECTED for {self.name} with {event.other_actor.type_id}")
        self.collision_events.append((self._sim_time, event.other_actor.id, event.other_actor.type_id))

    def _apply_avoidance(self, simulation_time):
        """Sets speed cap / lateral offset for this tick. Returns True if it applied a full brake itself."""
        self.speed_cap_mps = None
        self.lateral_offset_m = 0.0
        self.last_avoidance_event = None

        ttc, threat = find_primary_threat(self.world, self.vehicle)
        if threat is None:
            return False

        threat_name = threat.attributes.get("role_name", threat.type_id)
        if ttc < AVOIDANCE_IMMINENT_TTC_S:
            action = "brake"
            self.release_launch_hold()  # constant velocity would otherwise override the brake
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        else:
            # Match the threat's speed along our heading (follow it) and edge away if the lane allows
            action = "follow"
            fwd = self.vehicle.get_transform().get_forward_vector()
            tv = threat.get_velocity()
            self.speed_cap_mps = max(0.0, tv.x * fwd.x + tv.y * fwd.y)
            self.lateral_offset_m = nudge_offset(self.world, self.vehicle, threat)

        self.last_avoidance_event = f"{self.name}: {action} (Threat: {threat_name}, TTC {ttc:.1f}s)"
        if simulation_time - self._last_avoidance_log_time >= AVOIDANCE_LOG_INTERVAL_S:
            print(f"  [AVOIDANCE ACTIVATED] {self.last_avoidance_event}")
            self._last_avoidance_log_time = simulation_time
        return action == "brake"

    def update(self, simulation_time, sim_vehicles):
        self._sim_time = simulation_time
        if self.enable_avoidance and self.vehicle and simulation_time >= self.start_delay:
            if self._apply_avoidance(simulation_time):
                return True   # full brake already applied; running the PID now would overwrite it
        return super().update(simulation_time, sim_vehicles)

    def destroy(self):
        if self.collision_sensor and self.collision_sensor.is_listening:
            self.collision_sensor.stop()
        time.sleep(0.05)
        if self.collision_sensor and self.collision_sensor.is_alive:
            self.collision_sensor.destroy()
        super().destroy()


def estimate_trajectory_yaw(carla_pts):
    if len(carla_pts) < 2:
        return 0.0
    dx = carla_pts[1][0] - carla_pts[0][0]
    dy = carla_pts[1][1] - carla_pts[0][1]
    return math.degrees(math.atan2(dy, dx))


def get_initial_yaw(world, carla_pts, spawn_point, real_heading_deg=None, real_point=None):
    trajectory_yaw = estimate_trajectory_yaw(carla_pts)

    if real_heading_deg is not None and real_point is not None:
        dji_yaw = convert_real_heading_to_carla_yaw(real_heading_deg, real_point, trajectory_yaw_hint=trajectory_yaw)
        if dji_yaw is not None:
            return dji_yaw

    if world is not None:
        wp = world.get_map().get_waypoint(spawn_point, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp:
            return wp.transform.rotation.yaw

    return trajectory_yaw


def is_spawn_location_safe(target_spawn_loc, existing_vehicles, margin=3.5):
    for vehicle in existing_vehicles:
        if vehicle.get_location().distance(target_spawn_loc) < margin:
            return False
    return True


def cleanup_world(client):
    try:
        world = client.get_world()
        vehicle_list = world.get_actors().filter("vehicle.*")
        sensors = world.get_actors().filter("sensor.*")
        actors_to_destroy = list(vehicle_list) + list(sensors)
        if actors_to_destroy:
            client.apply_batch([carla.command.DestroyActor(x) for x in actors_to_destroy])
        time.sleep(0.2)
    except Exception as e:
        logging.error(f"Error during world cleanup: {e}")


def draw_scenario_overlay(world, vehicles, scenario_id, life_time):
    if not vehicles:
        return
    spectator = world.get_spectator()
    spec_loc = spectator.get_location()
    label_loc = carla.Location(x=spec_loc.x + 20.0, y=spec_loc.y - 40.0, z=0.0)
    world.debug.draw_string(
        label_loc,
        f"SCENARIO {scenario_id}",
        draw_shadow=True,
        color=carla.Color(r=255, g=0, b=0),
        life_time=life_time,
        persistent_lines=False,
    )


def parse_frame_index(frame_id):
    if frame_id is None:
        return None
    match = re.search(r"(\d+)$", str(frame_id))
    return int(match.group(1)) if match else None


def load_dji_spawn_params(csv_path):
    if not os.path.exists(csv_path):
        return {}
    params = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                tid = int(row["trackID"])
                frame_idx = parse_frame_index(row["frameID"])
                speed = float(row["linear_speed"]) * 3.6 if row.get("linear_speed") else None
                yaw = float(row["heading_angle"]) if row.get("heading_angle") else None
                if tid not in params:
                    params[tid] = []
                params[tid].append((frame_idx, speed, yaw))
            except (ValueError, KeyError):
                continue
    return params


def get_dji_spawn_values_for_frame_exact(dji_spawn_params, tid, start_frame):
    per_frame_rows = dji_spawn_params.get(tid)
    if not per_frame_rows:
        return None, None, None
    for frame_idx, speed_kmh, yaw_deg in per_frame_rows:
        if frame_idx == start_frame:
            return yaw_deg, speed_kmh, frame_idx
    return None, None, None


# --- Step 3 & 4: Continuous Speeds from Trajectory Points ---
def calculate_continuous_speeds(points, fps, correction_factor):
    speeds_kmh, step = [], 7
    if len(points) < step + 1:
        return [0] * len(points)
    # points[i] -> points[i + step] spans `step` frame intervals (was (step + 1), 12.5% too slow)
    time_interval = step / fps
    for i in range(len(points) - step):
        p1, p2 = points[i], points[i + step]
        distance = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        raw_speed_mps = distance / time_interval if time_interval > 0 else 0
        corrected_speed_mps = raw_speed_mps * correction_factor
        speeds_kmh.append(corrected_speed_mps * 3.6)
    speeds_kmh.extend([speeds_kmh[-1]] * step if speeds_kmh else [0] * step)
    return speeds_kmh


def prepare_track_spawn_data(annotation_parser, tid, interaction_frame, num_frames_required, dji_spawn_params):
    label_id = annotation_parser.get_label_for_track(tid)
    if label_id == 4:
        return None

    trajectory_frames = annotation_parser.get_sliced_trajectory_frames(tid, interaction_frame, num_frames_required)
    real_pts = annotation_parser.get_sliced_trajectory_coords(tid, interaction_frame, num_frames_required)
    if len(real_pts) < 2:
        return None

    # Points outside the 6 regions come back as None (trimmed); drop them before using the path
    carla_pts = [pt for pt in transform_trajectory_with_analysis(real_pts) if pt is not None]
    if len(carla_pts) < 2:
        return None

    speeds = calculate_continuous_speeds(carla_pts, REAL_WORLD_FPS, SPEED_CORRECTION_FACTOR)
    dynamic_route = [(pt[0], pt[1], DEFAULT_Z, speed) for pt, speed in zip(carla_pts, speeds)]

    # Spawn heading from the CARLA path itself. The EKF-heading conversion guessed among 4 axis
    # conventions and was 60-74 deg off for slow vehicles (scenarios 0528/0532).
    spawn_yaw_deg = path_yaw_deg(carla_pts)

    spawn_point = carla.Location(
        x=dynamic_route[0][0],
        y=dynamic_route[0][1],
        z=dynamic_route[0][2],
    )
    return {
        "tid": tid,
        "label_id": label_id,
        "carla_pts": carla_pts,
        "dynamic_route": dynamic_route,
        "spawn_point": spawn_point,
        "spawn_yaw_deg": spawn_yaw_deg,
    }


def run_single_scenario(client, world, summary_path, scenario_id_str, annotation_parser, dji_spawn_params):
    scenario_dir = os.path.dirname(summary_path)
    output_video_path = os.path.join(scenario_dir, f"{scenario_id_str}_carla_sim.mp4")

    print(f"\n=======================================================", flush=True)
    print(f"Simulating & Recording Scenario (SCENARIO_NEW4 IDENTICAL PIPELINE): {scenario_id_str}", flush=True)
    print(f"=======================================================", flush=True)

    cleanup_world(client)

    start_frame, end_frame, vehicle_ids, highlight_ids, ego_id = parse_summary_file(summary_path)
    num_frames_required = max(1, end_frame - start_frame + 1)

    scenario_duration_sec = num_frames_required / REAL_WORLD_FPS
    total_sim_ticks = int(round(scenario_duration_sec * SIM_FPS))
    
    print(f"Summary Annotation Range: frames {start_frame} -> {end_frame} ({num_frames_required} frames = {scenario_duration_sec:.2f}s)", flush=True)
    print(f"CARLA Simulation Ticks  : {total_sim_ticks} frames @ {SIM_FPS} FPS ({scenario_duration_sec:.2f}s video)", flush=True)
    print(f"Target Vehicles ({len(vehicle_ids)}): {vehicle_ids}", flush=True)

    dt = world.get_settings().fixed_delta_seconds
    bp_library = world.get_blueprint_library()

    scenario_tracks = {}
    for tid in vehicle_ids:
        spawn_data = prepare_track_spawn_data(annotation_parser, tid, start_frame, num_frames_required, dji_spawn_params)
        if spawn_data:
            scenario_tracks[tid] = spawn_data

    vehicles = {}
    spawned_vehicles_actors = []

    for tid in scenario_tracks:
        track_data = scenario_tracks[tid]
        label_id = track_data["label_id"]
        carla_pts = track_data["carla_pts"]
        dynamic_route = track_data["dynamic_route"]
        blueprint_name = LABEL_BLUEPRINT_MAP.get(label_id, VEHICLE_BLUEPRINT)
        base_spawn_point = track_data["spawn_point"]

        initial_yaw = track_data.get("spawn_yaw_deg")
        if initial_yaw is None:
            initial_yaw = get_initial_yaw(world, carla_pts, base_spawn_point)

        vname = f"vehicle_{tid}"
        vehicle_bp = bp_library.find(blueprint_name)
        if vehicle_bp.has_attribute("role_name"):
            vehicle_bp.set_attribute("role_name", vname)

        spawn_candidates = build_spawn_search_transforms(
            world,
            base_spawn_point,
            initial_yaw,
            annotation_parser=annotation_parser,
            tid=tid,
            interaction_frame=start_frame,
        )
        sim_v = None

        for candidate_transform, strategy in spawn_candidates:
            if not is_spawn_location_safe(candidate_transform.location, spawned_vehicles_actors, SPAWN_SAFETY_MARGIN):
                continue
            try:
                sim_v = CapturingVehicle(
                    world,
                    vehicle_bp,
                    vname,
                    (candidate_transform.location.x, candidate_transform.location.y, candidate_transform.location.z),
                    candidate_transform.rotation.yaw,
                    scenario_dir,
                )
                print(f"  [{vname}] Spawned using strategy: {strategy}", flush=True)
                break
            except RuntimeError:
                continue

        if sim_v is None:
            for z_off in Z_ELEVATION_OFFSETS:
                try:
                    sim_v = CapturingVehicle(
                        world,
                        vehicle_bp,
                        vname,
                        (base_spawn_point.x, base_spawn_point.y, base_spawn_point.z + z_off),
                        initial_yaw,
                        scenario_dir,
                    )
                    if sim_v and sim_v.vehicle:
                        break
                except RuntimeError:
                    continue

        if sim_v is None or sim_v.vehicle is None:
            print(f"  FAILED to spawn TID {tid} ({blueprint_name})", flush=True)
            continue

        print(f"  Successfully spawned TID {tid} ({blueprint_name}) with yaw={initial_yaw:.2f}°", flush=True)
        sim_v.set_dynamic_route(dynamic_route)
        vehicles[vname] = sim_v
        spawned_vehicles_actors.append(sim_v.vehicle)
        world.tick()

    print(f"Total spawned vehicles for {scenario_id_str}: {len(vehicles)}/{len(vehicle_ids)}", flush=True)

    if not vehicles:
        print("  No vehicles spawned. Skipping recording.", flush=True)
        return

    # Setup Overhead Spectator RGB Camera
    avg_loc = np.mean([[v.vehicle.get_location().x, v.vehicle.get_location().y] for v in vehicles.values()], axis=0)

    cam_bp = bp_library.find("sensor.camera.rgb")
    cam_width, cam_height = 1280, 720
    cam_bp.set_attribute("image_size_x", str(cam_width))
    cam_bp.set_attribute("image_size_y", str(cam_height))
    cam_bp.set_attribute("fov", "90")

    cam_transform = carla.Transform(
        carla.Location(x=avg_loc[0], y=avg_loc[1], z=50.0),
        carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
    )

    camera = world.spawn_actor(cam_bp, cam_transform)
    image_queue = queue.Queue()
    camera.listen(image_queue.put)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, float(SIM_FPS), (cam_width, cam_height))

    for _ in range(5):
        world.tick()
        try:
            image_queue.get(timeout=1.0)
        except queue.Empty:
            pass

    sim_clock = 0.0
    overlay_life_time = max(dt * 1.5, 0.1)
    recorded_frames = 0

    for step in range(total_sim_ticks):
        draw_scenario_overlay(world, vehicles, scenario_id_str, overlay_life_time)

        for v in vehicles.values():
            v.update(sim_clock, vehicles)

        world.tick()
        sim_clock += dt

        try:
            carla_image = image_queue.get(timeout=2.0)
            array = np.frombuffer(carla_image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (carla_image.height, carla_image.width, 4))
            frame_bgr = array[:, :, :3]
            writer.write(frame_bgr)
            recorded_frames += 1
        except queue.Empty:
            pass

    writer.release()
    try:
        camera.stop()
        camera.destroy()
    except Exception:
        pass

    cleanup_world(client)

    size_mb = os.path.getsize(output_video_path) / (1024 * 1024) if os.path.exists(output_video_path) else 0.0
    print(f"Saved SCENARIO_NEW4 IDENTICAL video: {output_video_path} ({recorded_frames} frames, {scenario_duration_sec:.2f}s, {size_mb:.2f} MB)", flush=True)


def run():
    print(f"Connecting to CARLA simulator on {CARLA_HOST}:{CARLA_PORT}...", flush=True)
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(20.0)

    world = client.get_world()
    if TOWN not in world.get_map().name:
        print(f"Loading world map {TOWN}...", flush=True)
        world = client.load_world(TOWN)
        time.sleep(2)

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / SIM_FPS
    world.apply_settings(settings)

    print("Loading annotation dataset...", flush=True)
    annotation_parser = AnnotationParser(INPUT_FILE)
    dji_spawn_params = load_dji_spawn_params(DJI_DATA_FILE)

    # TARGET ALL 16 SCENARIOS ACROSS ALL SUBFOLDERS IN Scenariosssss
    summaries = glob.glob(os.path.join(SCENARIOS_DIR, "**/*_summary.txt"), recursive=True)
    summaries.sort()

    print(f"Found {len(summaries)} scenario summaries to simulate & record with SCENARIO_NEW4 IDENTICAL PIPELINE.", flush=True)

    for idx, summary_path in enumerate(summaries, 1):
        scenario_name = os.path.splitext(os.path.basename(summary_path))[0]
        if scenario_name.endswith("_summary"):
            scenario_id_str = scenario_name[:-len("_summary")]
        else:
            scenario_id_str = scenario_name

        print(f"\n--- Progress: [{idx}/{len(summaries)}] Processing {scenario_id_str} ({os.path.relpath(summary_path, SCENARIOS_DIR)}) ---", flush=True)
        run_single_scenario(client, world, summary_path, scenario_id_str, annotation_parser, dji_spawn_params)

    world.apply_settings(original_settings)
    print(f"\nCompleted simulating & recording ALL 16 scenario videos!", flush=True)


if __name__ == "__main__":
    run()
