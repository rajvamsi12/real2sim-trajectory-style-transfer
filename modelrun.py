import os
import json
import math
import time
import carla
import weakref
import random
import numpy as np
import shutil
import logging
import re
import statistics
import gc
from collections import deque
from functools import reduce
from itertools import combinations

# --- ML Imports ---
import torch
from PIL import Image
import pickle5 as pickle
from transformers import AutoImageProcessor

# 1. Disable Gradients Globally (VRAM Optimization)
torch.set_grad_enabled(False)

# Try importing the user's model
try:
    from train3 import BehavioralCloningSegFormer
except ImportError:
    print("WARNING: Could not import 'BehavioralCloningSegFormer' from 'train3.py'.")
    print("Ensure 'train3.py' is in the same directory.")

# --- SCC/TimeMethod Imports ---
try:
    from timemethod import transform_trajectory_with_analysis
    from scc import SimVehicle, DummyWaypoint
except ImportError:
    print("ERROR: Could not import 'transform_trajectory_with_analysis' or 'SimVehicle'.")
    print("Please ensure 'timemethod.py' and 'scc.py' exist and are in the same directory.")
    exit()


# --- Configuration ---
TOTAL_IMAGES_GOAL = 1_000_000
SCENARIO_DURATION_SEC = 15
SIM_FPS = 20
IMAGES_PER_VEHICLE_PER_SCENARIO = int(SCENARIO_DURATION_SEC * SIM_FPS)
IMAGE_CAPTURE_TOLERANCE = 0.98

INPUT_FILE = "default.json"
TIDS_FILE = "tids.txt"
OUTPUT_DIR = "_scenario_output_disabled" 
STATS_FILE = "scenario_stats.csv" 
SNAPSHOT_DIR = "scenario_snapshots" 

CARLA_HOST = "localhost"
CARLA_PORT = 2000
TOWN = "Town10HD"

VEHICLE_BLUEPRINT = "vehicle.bmw.grandtourer"
DEFAULT_Z = 0.6

REAL_WORLD_FPS = 4
SPEED_CORRECTION_FACTOR = 4

MIN_ACTORS_FOR_SCENARIO = 3
MIN_VEHICLE_SPEED_KMH = 1.0
SPLIT_LARGE_GROUPS = False

SCENARIOS_PER_WORLD_RELOAD = 10

# --- Spawn Retraction Configuration ---
SPAWN_SAFETY_MARGIN = 3.5
MAX_RETRACTION_ATTEMPTS = 10
RETRACTION_STEP = 0.5

# --- Real-time Avoidance Configuration ---
AVOIDANCE_HORIZON_S = 3.5
AVOIDANCE_IMMINENT_TTC_S = 1.5

# --- Grouping Configuration ---
DISTANCE_THRESHOLD = 100.0 

# --- Frame Regions ---
FRAME_REGIONS = {
    # 1: (0, 88),
    1: (89, 177),
    # 2: (178, 252),
    # 2: (253, 327),  
    # 3: (328, 401),
    # 3: (402, 475),
    # 4: (476, 536),
    # 4: (530, 590),
    # 5: (591, 685),
    # 5: (686, 779),
    # 6: (780, 845),
    # 6: (846, 910),
    # 7: (884, 944),
}

# -----------------------------
# Global ML State
# -----------------------------
EGO_LATEST_IMAGE = None

def ego_image_callback(image):
    global EGO_LATEST_IMAGE
    EGO_LATEST_IMAGE = image

# -----------------------------
# Helper Functions for Detection
# -----------------------------
def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def detect_following(world, ego_actor):
    try:
        ego_wp = world.get_map().get_waypoint(ego_actor.get_location())
        ego_vel = ego_actor.get_velocity()
        ego_speed = math.hypot(ego_vel.x, ego_vel.y)

        for veh in world.get_actors().filter("vehicle.*"):
            if veh.id == ego_actor.id:
                continue
            
            # Simple check if veh is alive
            if not veh.is_alive: continue

            veh_loc = veh.get_location()
            if distance(ego_actor.get_location(), veh_loc) > 50: continue

            wp = world.get_map().get_waypoint(veh_loc)

            # same lane & road
            if wp.road_id == ego_wp.road_id and wp.lane_id == ego_wp.lane_id:
                d = distance(ego_actor.get_location(), veh_loc)

                # Check if 'veh' is in front of 'ego'
                # Dot product of ego forward vector and vector to other vehicle
                fwd = ego_actor.get_transform().get_forward_vector()
                vec_to_veh = carla.Vector3D(veh_loc.x - ego_actor.get_location().x, veh_loc.y - ego_actor.get_location().y, 0)
                
                # Normalize 2D
                vec_len = math.hypot(vec_to_veh.x, vec_to_veh.y)
                if vec_len > 0:
                    dot = (fwd.x * vec_to_veh.x + fwd.y * vec_to_veh.y) / vec_len
                    if dot > 0.5: # Roughly in front (within ~60 deg cone)
                         if d < 40 and d > 2: # Within following distance
                            if ego_speed > 0.5:
                                return True
    except Exception:
        pass # Waypoint errors can happen if off-road
    return False

def detect_dense_traffic(world, ego_actor):
    ego_loc = ego_actor.get_location()
    count = 0
    for veh in world.get_actors().filter("vehicle.*"):
        if veh.id == ego_actor.id:
            continue
        if distance(ego_loc, veh.get_location()) < 40:
            count += 1
    return count 

def detect_lane_change(lane_history):
    if len(lane_history) < 5:
        return False
    # Check if lane ID changed significantly (sustained change)
    first_chunk = list(lane_history)[:5]
    last_chunk = list(lane_history)[-5:]
    
    # Mode of first few vs mode of last few
    start_lane = max(set(first_chunk), key=first_chunk.count)
    end_lane = max(set(last_chunk), key=last_chunk.count)

    return start_lane != end_lane

# --- New Relaxed Overtaking Detection ---
def longitudinal_projection(ego_xy, ego_yaw, other_xy):
    """
    Projects relative position of 'other' onto ego forward axis.
    Returns signed longitudinal distance:
    >0 = ahead of ego
    <0 = behind ego
    """
    ex, ey = ego_xy
    ox, oy = other_xy

    # relative vector
    rx = ox - ex
    ry = oy - ey

    # ego heading unit vector
    heading = np.array([np.cos(ego_yaw), np.sin(ego_yaw)])

    # projection scalar
    return rx * heading[0] + ry * heading[1]

def detect_relaxed_overtake(ego_traj, ego_yaws, other_trajs):
    """
    ego_traj   : list of (x,y)
    ego_yaws   : list of yaw angles (radians)
    other_trajs: dict vehicle_id -> list of (x,y)

    Returns True if ANY vehicle was behind first and ahead last
    """

    if len(ego_traj) < 2:
        return False

    ego_start = ego_traj[0]
    ego_end   = ego_traj[-1]

    # Use first and last valid yaws
    yaw_start = ego_yaws[0]
    yaw_end   = ego_yaws[-1]

    for vid, traj in other_trajs.items():
        if len(traj) < 2:
            continue

        # We compare start state vs end state
        # Note: traj length might differ slightly if spawn time differs, 
        # but usually synchronous mode keeps steps aligned.
        # We use the first recorded point and last recorded point for that vehicle.
        
        other_start = traj[0]
        other_end   = traj[-1]

        d0 = longitudinal_projection(ego_start, yaw_start, other_start)
        d1 = longitudinal_projection(ego_end,   yaw_end,   other_end)

        # Rule: Start BEHIND (<0) and End AHEAD (>0) relative to Ego frame
        # OR Start AHEAD (>0) and End BEHIND (<0) (Being overtaken) - Wait, 
        # "Overtaking" usually means Ego passes them. 
        # If Ego passes them: Other starts AHEAD (>0) and ends BEHIND (<0).
        # If Ego IS overtaken: Other starts BEHIND (<0) and ends AHEAD (>0).
        
        # Original request said: "if ego was behind another vehicle at start and ego ends ahead of the same vehicle at end"
        # Ego Behind Start -> d0 (other relative to ego) > 0 (Other is ahead)
        # Ego Ahead End -> d1 (other relative to ego) < 0 (Other is behind)
        
        # BUT the provided snippet logic was: `if d0 < 0 and d1 > 0: return True`
        # d0 < 0 means "Other is behind Ego". d1 > 0 means "Other is ahead of Ego".
        # This describes "Being Overtaken" (Someone passes Ego).
        
        # Let's stick to the prompt's explicit code:
        # "if d0 < 0 and d1 > 0: return True"
        # This technically detects if SOMEONE ELSE overtakes the Ego.
        
        # However, usually "Overtaking scenario" implies Ego is the active agent.
        # If Ego overtakes someone: They start Ahead (d0 > 0) and end Behind (d1 < 0).
        
        # I will implement BOTH checks or clarify. 
        # The prompt code `if d0 < 0 and d1 > 0` detects "Being Overtaken".
        # I will add the "Ego Overtakes" case too for completeness or switch if that's what you meant.
        # Given "replace the old overtaking mechanism", and usually we want to know if Ego passed someone.
        # I will interpret "Overtake" as "Ego passes someone".
        # Ego passes someone: They were >0, now <0.
        
        # Wait, let's re-read the "Relaxed Overtaking" logic provided in the prompt:
        # "if ego was behind another vehicle at start and ego ends ahead of the same vehicle at end"
        # Ego Behind -> Other Ahead -> d0 > 0
        # Ego Ahead -> Other Behind -> d1 < 0
        
        # The code snippet provided in the prompt was:
        # `if d0 < 0 and d1 > 0: return True`
        # This snippet actually checks if OTHER was behind and ended up ahead.
        
        # I will implement "Ego Overtaking" (Ego passes someone).
        if d0 > 0 and d1 < 0:
            return True

    return False

def classify_intersection_type(world, ego_positions):
    m = world.get_map()
    # Sample a few points from history to check for intersections
    step = max(1, len(ego_positions) // 10)
    for i in range(0, len(ego_positions), step):
        p = ego_positions[i]
        # Ensure 'p' is a carla.Location
        loc = p if isinstance(p, carla.Location) else carla.Location(x=p.x, y=p.y, z=p.z)
        
        try:
            wp = m.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp and wp.is_junction:
                junction = wp.get_junction()
                # Heuristic: count unique roads connected to this junction
                # Get bounding box of junction
                bb = junction.bounding_box
                # Find all waypoints inside junction is hard, use topology around it?
                # Simpler: check wp.next/previous road IDs at junction entry
                
                # Let's trust the provided logic or simplify:
                # Provided logic used junction.get_waypoints which returns pairs (entry, exit)
                topology = junction.get_waypoints(carla.LaneType.Driving)
                road_connections = set()
                for a, b in topology:
                    road_connections.add(a.road_id)
                    road_connections.add(b.road_id)
                
                n = len(road_connections)
                if n == 3: return "3-way"
                if n >= 4: return "4-way"
        except Exception:
            pass # Map errors
            
    return "none"

def classify_turn_from_trajectory(ego_positions):
    """
    Classify whether ego executed a turn based on heading change.
    Returns one of:
    straight, left_turn, right_turn, slight_left_turn, slight_right_turn
    """

    if len(ego_positions) < 5:
        return "unknown"

    # build heading angles from positions
    yaws = []
    for i in range(1, len(ego_positions)):
        p1 = ego_positions[i-1]
        p2 = ego_positions[i]

        dx = p2.x - p1.x
        dy = p2.y - p1.y

        if abs(dx) + abs(dy) < 1e-4:
            continue

        yaw = math.degrees(math.atan2(dy, dx))
        yaws.append(yaw)

    if len(yaws) < 2:
        return "unknown"

    initial = yaws[0]
    final = yaws[-1]

    # normalize to [-180, 180]
    delta = final - initial
    while delta > 180:
        delta -= 360
    while delta < -180:
        delta += 360

    abs_delta = abs(delta)

    # thresholds (you can tune)
    if abs_delta < 15:
        return "straight"

    if 15 <= abs_delta < 60:
        return "slight_left_turn" if delta > 0 else "slight_right_turn"

    return "left_turn" if delta > 0 else "right_turn"

# -----------------------------
# Ego ML Controller
# -----------------------------
class EgoMLController:
    def __init__(self, vehicle, world, model_path, scaler_path, model_name):
        self.vehicle = vehicle
        self.world = world

        camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "640")
        camera_bp.set_attribute("image_size_y", "480")
        camera_bp.set_attribute("fov", "90")

        camera_tf = carla.Transform(
            carla.Location(x=1.5, z=2.3),
            carla.Rotation(pitch=0)
        )

        self.cam = world.spawn_actor(camera_bp, camera_tf, attach_to=vehicle)
        self.cam.listen(ego_image_callback)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model from {model_path}...")
        self.model = BehavioralCloningSegFormer(model_name=model_name).to(self.device)
        
        state = torch.load(model_path, map_location=self.device, pickle_module=pickle)
        self.model.load_state_dict(state)
        self.model.eval()

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        self.processor = AutoImageProcessor.from_pretrained(model_name)

        self.Kp = 1.0
        self.Ki = 0.05
        self.Kd = 0.1
        self.integral = 0.0
        self.prev_error = 0.0
        
        print("EgoMLController initialized successfully.")

    def tick(self):
        global EGO_LATEST_IMAGE
        if EGO_LATEST_IMAGE is None:
            return

        img_bgra = np.frombuffer(
            EGO_LATEST_IMAGE.raw_data, dtype=np.uint8
        ).reshape((EGO_LATEST_IMAGE.height, EGO_LATEST_IMAGE.width, 4))

        img_rgb = img_bgra[:, :, :3][:, :, ::-1]
        pil_image = Image.fromarray(img_rgb)

        with torch.no_grad():
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
            normalized_output = self.model(inputs["pixel_values"])
            real_prediction = self.scaler.inverse_transform(normalized_output.cpu().numpy())[0]

        target_speed = float(real_prediction[0])
        steer_deg = float(real_prediction[1])

        v = self.vehicle.get_velocity()
        current_speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)

        error = target_speed - current_speed
        self.integral += error * 0.05
        derivative = (error - self.prev_error) / 0.05
        self.prev_error = error

        pid_out = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        throttle = float(np.clip(pid_out, 0, 1)) if pid_out > 0 else 0.0
        brake = float(np.clip(-pid_out, 0, 1)) if pid_out < 0 else 0.0
        steer = float(np.clip(steer_deg / 60.0, -1.0, 1.0))

        self.vehicle.apply_control(carla.VehicleControl(
            throttle=throttle,
            brake=brake,
            steer=steer
        ))

    def destroy(self):
        if self.cam and self.cam.is_alive:
            self.cam.stop()
            self.cam.destroy()


# -----------------------------
# Hybrid avoidance helpers
# -----------------------------
def predict_ttc(p0, v0, p1, v1, safety_r=3.0, horizon=AVOIDANCE_HORIZON_S):
    dp_x, dp_y = p0.x - p1.x, p0.y - p1.y
    dv_x, dv_y = v0.x - v1.x, v0.y - v1.y
    dv2 = dv_x*dv_x + dv_y*dv_y
    if dv2 == 0.0:
        return 0.0 if (dp_x*dp_x + dp_y*dp_y) <= safety_r*safety_r else None
    t_star = - (dp_x*dv_x + dv_y*dv_y) / dv2
    if t_star < 0.0 or t_star > horizon:
        return None
    cx, cy = dp_x + dv_x * t_star, dp_y + dv_y * t_star
    d2 = cx*cx + cy*cy
    return t_star if d2 <= safety_r*safety_r else None


def avoidance_adjustment(world, ego, target_loc, threat_loc, max_offset=0.8):
    vel = ego.get_velocity()
    d = math.hypot(vel.x, vel.y) or 1.0
    fwd = carla.Vector3D(vel.x/d, vel.y/d, 0.0)
    left = carla.Vector3D(-fwd.y, fwd.x, 0.0)

    ego_loc = ego.get_location()
    to_threat = carla.Vector3D(threat_loc.x - ego_loc.x, threat_loc.y - ego_loc.y, 0.0)

    side = carla.Vector3D(-left.x, -left.y, 0.0) if (to_threat.x*left.x + to_threat.y*left.y) > 0 else left
    nudged = carla.Location(
        target_loc.x + side.x * max_offset,
        target_loc.y + side.y * max_offset,
        target_loc.z
    )

    wp = world.get_map().get_waypoint(nudged, project_to_road=True, lane_type=carla.LaneType.Driving)
    return wp.transform.location if wp else None


def blend_back(current_loc, annotated_loc, alpha=0.2):
    return carla.Location(
        x=alpha*annotated_loc.x + (1-alpha)*current_loc.x,
        y=alpha*annotated_loc.y + (1-alpha)*current_loc.y,
        z=annotated_loc.z
    )


def safe_next_target(world, ego, base_target_loc, base_target_speed, horizon=AVOIDANCE_HORIZON_S):
    ego_loc = ego.get_location()
    ego_vel = ego.get_velocity()
    threats = []
    
    nearby_actors = []
    for a in world.get_actors().filter('vehicle.*'):
        if a.id == ego.id: continue
        if ego_loc.distance(a.get_location()) < 40.0:
            nearby_actors.append(a)

    for a in nearby_actors:
        a_loc = a.get_location()
        a_vel = a.get_velocity()
        ttc = predict_ttc(ego_loc, ego_vel, a_loc, a_vel, horizon=horizon)
        if ttc is not None:
            threats.append((ttc, a, a_loc))
            
    if not threats:
        return base_target_loc, base_target_speed, False

    threats.sort(key=lambda x: x[0])
    ttc, primary_threat, threat_loc = threats[0]
    
    if ttc < AVOIDANCE_IMMINENT_TTC_S:
        return base_target_loc, 0.0, True
        
    nudged = avoidance_adjustment(world, ego, base_target_loc, threat_loc, max_offset=0.8)

    if nudged:
        nudge_is_safe = True
        vec_to_nudge = nudged - ego_loc
        predicted_vel = vec_to_nudge.make_unit_vector() * ego_vel.length()

        for other_ttc, other_threat, other_threat_loc in threats:
            if other_threat.id == primary_threat.id:
                continue
            
            future_ttc = predict_ttc(ego_loc, predicted_vel, other_threat_loc, other_threat.get_velocity(), horizon=horizon)
            if future_ttc is not None and future_ttc < 1.0:
                nudge_is_safe = False
                break
        
        if nudge_is_safe:
            return nudged, max(0.833, 0.5*base_target_speed), True
        
    return base_target_loc, max(0.833, 0.5*base_target_speed), True


# -----------------------------
# AnnotationParser
# -----------------------------
class AnnotationParser:
    def __init__(self, filepath):
        self.track_data = {}
        self.frame_data = {}
        self._load_and_parse(filepath)

    def _load_and_parse(self, filepath):
        logging.info(f"Loading and parsing annotation file: {filepath}...")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Annotation file not found at {filepath}")
        with open(filepath, 'r') as f:
            data = json.load(f)
        for frame_num, item in enumerate(data.get('items', [])):
            self.frame_data[frame_num] = []
            for ann in item.get('annotations', []):
                tid = ann.get('attributes', {}).get('track_id')
                bbox = tuple(ann.get('bbox'))
                if tid is not None and bbox and len(bbox) == 4:
                    self.frame_data[frame_num].append({'track_id': tid, 'bbox': bbox})
                    if tid not in self.track_data:
                        self.track_data[tid] = {'trajectory': {}}
                    self.track_data[tid]['trajectory'][frame_num] = bbox
        logging.info("Annotation parsing complete.")
        logging.info(f"Found {len(self.track_data)} unique tracks across {len(self.frame_data)} frames.")

    def get_track_positions_at_frame(self, frame_num):
        positions = {}
        if frame_num in self.frame_data:
            for ann in self.frame_data[frame_num]:
                tid = ann['track_id']
                x, y, w, h = ann['bbox']
                positions[tid] = (x + w/2, -(y + h/2))
        return positions

    def get_frames_for_track(self, tid):
        return sorted(self.track_data[tid]['trajectory'].keys()) if tid in self.track_data else []

    def get_sliced_trajectory_coords(self, tid, start_frame, num_frames):
        coords = []
        if tid in self.track_data:
            sorted_frames = sorted(self.track_data[tid]['trajectory'].keys())
            end_frame = start_frame + num_frames
            for f in [fr for fr in sorted_frames if start_frame <= fr < end_frame]:
                x, y, w, h = self.track_data[tid]['trajectory'][f]
                coords.append((x+w/2, -(y+h/2)))
        return coords


# -----------------------------
# CapturingVehicle
# -----------------------------
class CapturingVehicle(SimVehicle):
    def __init__(self, world, bp, name, start, yaw, output_path, start_delay=0, start_capture_time=0, end_capture_time=float('inf'), is_ego=False, scenario_id=0, mode=""):
        super().__init__(world, bp, name, start, yaw, start_delay)
        self.output_path = os.path.join(output_path, self.name)
        self.image_path = None 
        
        self.camera_ego = None
        self.camera_bird = None
        self.collision_sensor = None
        self.angular_velocities = []
        self.linear_velocities = []
        self.image_count = 0
        self.has_collided = False
        self.start_capture_time = start_capture_time
        self.end_capture_time = end_capture_time
        self.is_capturing = False
        self._recent_risky = 0
        self._blend_alpha = 0.2
        self._safe_frames_to_blend = 10
        self.time_of_first_collision = None
        
        # --- SNAPSHOT CONFIG ---
        self.is_ego = is_ego
        self.scenario_id = scenario_id
        self.mode = mode
        self.snapshot_taken_ego = False
        self.snapshot_taken_bird = False
        
        self._setup_sensors()

    def _setup_sensors(self):
        # 1. Collision Sensor (Always)
        collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: CapturingVehicle._on_collision(weakref.ref(self), event))

        # 2. Cameras (Only if Expert Ego)
        if self.is_ego and self.mode == "EXPERT":
            # -- Camera A: Ego View (Driver) --
            bp_ego = self.world.get_blueprint_library().find('sensor.camera.rgb')
            bp_ego.set_attribute('image_size_x', '800')
            bp_ego.set_attribute('image_size_y', '600')
            bp_ego.set_attribute('fov', '90')
            tf_ego = carla.Transform(carla.Location(x=1.5, z=2.4))
            
            self.camera_ego = self.world.spawn_actor(bp_ego, tf_ego, attach_to=self.vehicle)
            self.camera_ego.listen(lambda image: CapturingVehicle._save_snapshot(weakref.ref(self), image, "ego"))

            # -- Camera B: Bird's Eye View (Top Down) --
            bp_bird = self.world.get_blueprint_library().find('sensor.camera.rgb')
            bp_bird.set_attribute('image_size_x', '1024')
            bp_bird.set_attribute('image_size_y', '1024')
            bp_bird.set_attribute('fov', '90')
            tf_bird = carla.Transform(carla.Location(x=0, z=30), carla.Rotation(pitch=-90))
            
            self.camera_bird = self.world.spawn_actor(bp_bird, tf_bird, attach_to=self.vehicle)
            self.camera_bird.listen(lambda image: CapturingVehicle._save_snapshot(weakref.ref(self), image, "birdseye"))

    @staticmethod
    def _save_snapshot(weak_self, image, view_type):
        """
        view_type: 'ego' or 'birdseye'
        """
        self = weak_self()
        if not self: return
        
        # Only take snapshot if capturing, not already taken, and frame count > 20 (settle time)
        if self.is_capturing and self.image_count > 20:
            
            if view_type == "ego" and not self.snapshot_taken_ego:
                filename = f"scenario_{self.scenario_id}_expert_ego.png"
                filepath = os.path.join(SNAPSHOT_DIR, filename)
                image.save_to_disk(filepath)
                self.snapshot_taken_ego = True
                
            elif view_type == "birdseye" and not self.snapshot_taken_bird:
                filename = f"scenario_{self.scenario_id}_expert_birdseye.png"
                filepath = os.path.join(SNAPSHOT_DIR, filename)
                image.save_to_disk(filepath)
                self.snapshot_taken_bird = True

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self or not self.collision_sensor or not self.collision_sensor.is_alive: return
        if not self.has_collided:
            self.has_collided = True
            # Log timestamp for Time To First Collision
            self.time_of_first_collision = self.world.get_snapshot().timestamp.elapsed_seconds
            logging.warning(f"!!! Collision DETECTED for {self.name} with {event.other_actor.type_id}")

    def get_image_count(self):
        return self.image_count

    def _apply_avoidance(self):
        if not hasattr(self, "dense_route") or not self.vehicle or not self.vehicle.is_alive or self.wp_index >= len(self.dense_route): return
        
        original_wp, original_spd = self.dense_route[self.wp_index]
        adj_loc, adj_spd, risky = safe_next_target(self.world, self.vehicle, original_wp.transform.location, original_spd)

        if risky:
            self.dense_route[self.wp_index] = (DummyWaypoint(carla.Transform(adj_loc, original_wp.transform.rotation)), adj_spd)
            self._recent_risky = 0
        elif 0 < self._recent_risky < self._safe_frames_to_blend:
            blended = blend_back(self.vehicle.get_transform().location, original_wp.transform.location, self._blend_alpha)
            self.dense_route[self.wp_index] = (DummyWaypoint(carla.Transform(blended, original_wp.transform.rotation)), original_spd)
            self._recent_risky += 1
        else:
            self._recent_risky = max(self._recent_risky, self._safe_frames_to_blend+1)

    def update(self, simulation_time, sim_vehicles):
        self.is_capturing = self.start_capture_time <= simulation_time <= self.end_capture_time
        if self.is_capturing:
            self.image_count += 1

        try:
            self._apply_avoidance()
        except Exception:
            pass
        return super().update(simulation_time, sim_vehicles)

    def save_captured_data(self):
        return

    def destroy(self):
        if self.camera_ego and self.camera_ego.is_listening: self.camera_ego.stop()
        if self.camera_bird and self.camera_bird.is_listening: self.camera_bird.stop()
        if self.collision_sensor and self.collision_sensor.is_listening: self.collision_sensor.stop()
        
        time.sleep(0.05)
        
        if self.camera_ego and self.camera_ego.is_alive: self.camera_ego.destroy()
        if self.camera_bird and self.camera_bird.is_alive: self.camera_bird.destroy()
        if self.collision_sensor and self.collision_sensor.is_alive: self.collision_sensor.destroy()
        super().destroy()

# -----------------------------
# Helper Functions
# -----------------------------
def calculate_continuous_speeds(points, fps, correction_factor):
    speeds_kmh, step = [], 7
    if len(points) < step + 1: return [0] * len(points)
    time_interval = (step + 1) / fps
    for i in range(len(points) - step):
        p1, p2 = points[i], points[i + step]
        distance = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        raw_speed_mps = distance / time_interval if time_interval > 0 else 0
        corrected_speed_mps = raw_speed_mps * correction_factor
        speeds_kmh.append(corrected_speed_mps * 3.6)
    speeds_kmh.extend([speeds_kmh[-1]] * step if speeds_kmh else [0] * step)
    return speeds_kmh

def is_spawn_location_safe(target_spawn_loc, existing_vehicles, margin):
    for vehicle in existing_vehicles:
        if vehicle.get_location().distance(target_spawn_loc) < margin:
            return False
    return True

# -----------------------------
# Metric Helpers
# -----------------------------
def calculate_route_completion_distance_percentage(driven_path, expert_path):
    """
    Calculates completion based on distance projected onto the expert path.
    driven_path: list of carla.Location or (x,y) tuples
    expert_path: list of (x,y) tuples (reference)
    """
    if not expert_path or len(expert_path) < 2:
        return 0.0
    if not driven_path:
        return 0.0

    # Calculate total expert path length
    total_length = 0.0
    for i in range(len(expert_path) - 1):
        total_length += math.hypot(expert_path[i+1][0] - expert_path[i][0], 
                                   expert_path[i+1][1] - expert_path[i][1])
    
    if total_length == 0: return 100.0

    # Find the closest point on expert path for the LAST point of driven path
    last_pos = driven_path[-1]
    last_x, last_y = (last_pos.x, last_pos.y) if isinstance(last_pos, carla.Location) else last_pos

    min_dist = float('inf')
    closest_idx = 0

    for i, pt in enumerate(expert_path):
        dist = math.hypot(last_x - pt[0], last_y - pt[1])
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
            
    # Calculate distance covered up to that closest point
    covered_dist = 0.0
    for i in range(closest_idx):
        covered_dist += math.hypot(expert_path[i+1][0] - expert_path[i][0], 
                                   expert_path[i+1][1] - expert_path[i][1])
                                   
    return min(100.0, (covered_dist / total_length) * 100.0)


# -----------------------------
# Main Scenario Runner
# -----------------------------
def run_scenario(world, scenario_id, all_track_ids, annotation_parser, interaction_frame, mode="ML", expert_trajectory=None):
    # mode: "ML" or "EXPERT"
    # expert_trajectory: If mode="ML", this list of (x,y) tuples is the ground truth from the EXPERT run.
    
    vehicles, spawned_vehicles_actors = {}, []
    scenario_output_path = "dummy_path" 
    end_status = "UNKNOWN"
    
    ego_controller = None
    ego_name = None

    # --- METRIC CONTAINERS ---
    ego_positions = []
    ego_speeds = []
    yaw_rates = []
    ego_ttc_values = []
    lane_history = deque(maxlen=50) # Buffer for lane change detection
    
    # --- DATA COLLECTION FOR OVERTAKE DETECTION ---
    ego_yaws = []
    # We will need other vehicle positions too. Since we don't know IDs yet,
    # we initialize this dict after spawning or check existence during loop.
    other_positions = {} 

    try:
        dt = world.get_settings().fixed_delta_seconds
        bp_library = world.get_blueprint_library()
        vehicle_bp = bp_library.find(VEHICLE_BLUEPRINT)
        num_frames_required = int(SCENARIO_DURATION_SEC * REAL_WORLD_FPS)

        # NOTE: track_ids are passed in PRE-SHUFFLED from main(), so we iterate directly.
        # This ensures ML and EXPERT runs use the same vehicle mapping.
        
        for tid in all_track_ids:
            try:
                real_pts = annotation_parser.get_sliced_trajectory_coords(tid, interaction_frame, num_frames_required)
                if len(real_pts) < 8: continue
                carla_pts = transform_trajectory_with_analysis(real_pts)
                if len(carla_pts) < 8: continue

                speeds = calculate_continuous_speeds(carla_pts, REAL_WORLD_FPS, SPEED_CORRECTION_FACTOR)
                dynamic_route = [(pt[0], pt[1], DEFAULT_Z, speed) for pt, speed in zip(carla_pts, speeds)]
                
                if not dynamic_route or sum(s for _,_,_,s in dynamic_route)/len(dynamic_route) < MIN_VEHICLE_SPEED_KMH: continue

                p1, p2 = carla_pts[0], carla_pts[7]
                initial_yaw = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
                
                original_spawn_point = carla.Location(x=dynamic_route[0][0], y=dynamic_route[0][1], z=dynamic_route[0][2])
                safe_spawn_point, is_safe = original_spawn_point, is_spawn_location_safe(original_spawn_point, spawned_vehicles_actors, SPAWN_SAFETY_MARGIN)
                
                if not is_safe:
                    logging.warning(f"Spawn for {tid} occupied. Retracting.")
                    yaw_rad = math.radians(initial_yaw)
                    backwards = carla.Vector3D(x=-math.cos(yaw_rad), y=-math.sin(yaw_rad))
                    for i in range(MAX_RETRACTION_ATTEMPTS):
                        dist = (i + 1) * RETRACTION_STEP
                        new_loc = original_spawn_point + backwards * dist
                        if is_spawn_location_safe(new_loc, spawned_vehicles_actors, SPAWN_SAFETY_MARGIN):
                            safe_spawn_point, is_safe = new_loc, True
                            logging.info(f"SUCCESS: Safe spawn for {tid} after retracting {dist:.2f}m.")
                            break
                if not is_safe:
                    logging.error(f"FAILED: No safe spawn for {tid}. Skipping.")
                    continue

                # --- VISUAL MARKING ---
                is_ego = (len(vehicles) == 0) # First spawned is Ego
                if vehicle_bp.has_attribute('color'):
                    if is_ego:
                        vehicle_bp.set_attribute('color', '255,0,0') # RED for EGO
                    else:
                        vehicle_bp.set_attribute('color', '180,180,180') # GREY for others

                vname = f"vehicle_{tid}"
                
                # Setup Snapshot Args for Ego
                sim_v = CapturingVehicle(world, vehicle_bp, vname, 
                                         (safe_spawn_point.x, safe_spawn_point.y, safe_spawn_point.z), 
                                         initial_yaw, scenario_output_path, 
                                         start_capture_time=0.0, end_capture_time=SCENARIO_DURATION_SEC,
                                         is_ego=is_ego, scenario_id=scenario_id, mode=mode)
                                         
                sim_v.set_dynamic_route(dynamic_route)
                vehicles[vname] = sim_v
                spawned_vehicles_actors.append(sim_v.vehicle)
                world.tick()
            except Exception as e:
                logging.error(f"[SCENARIO {scenario_id}] FAILED TO SPAWN {tid}: {e}")
                continue

        if len(vehicles) < 2:
            return None, "SPAWN_FAILURE", None

        if len(vehicles) > 0:
            logging.info(f"[SCENARIO {scenario_id} - {mode}] Spawned {len(vehicles)} vehicles. Starting sim.")
            
            ego_name = list(vehicles.keys())[0] # Pick the first vehicle as Ego
            ego_vehicle_wrapper = vehicles[ego_name]
            ego_vehicle_actor = ego_vehicle_wrapper.vehicle
            
            # --- PREP OVERTAKE CONTAINERS ---
            other_positions = {v_name: [] for v_name in vehicles if v_name != ego_name}

            # --- SETUP REFERENCE TRAJECTORY ---
            if mode == "EXPERT":
                # In Expert mode, the reference is the dense_route (theoretical) 
                # but we will return the ACTUAL driven path as ground truth for the next run.
                # FIX: dense_route contains (Waypoint, Speed), not (x,y,z,speed).
                # We need to extract location from the Waypoint object.
                reference_traj_for_metrics = []
                for item in ego_vehicle_wrapper.dense_route:
                    if len(item) == 2:
                        # Likely (Waypoint, Speed) or (DummyWaypoint, Speed)
                        wp, _ = item
                        loc = wp.transform.location
                        reference_traj_for_metrics.append((loc.x, loc.y))
                    elif len(item) >= 2:
                         # Fallback if structure is different but indexable
                         # Assuming first item has transform.location
                         try:
                             loc = item[0].transform.location
                             reference_traj_for_metrics.append((loc.x, loc.y))
                         except AttributeError:
                             # If item[0] is just a coordinate? unlikely based on error.
                             pass
            else:
                # In ML mode, we prefer using the recorded expert trajectory if available
                if expert_trajectory and len(expert_trajectory) > 5:
                    reference_traj_for_metrics = expert_trajectory
                    logging.info("Using RECORDED EXPERT PATH as ground truth for ML metrics.")
                else:
                    # Fallback to theoretical route
                    reference_traj_for_metrics = []
                    for item in ego_vehicle_wrapper.dense_route:
                        if len(item) == 2:
                            wp, _ = item
                            loc = wp.transform.location
                            reference_traj_for_metrics.append((loc.x, loc.y))
                    logging.warning("No expert path provided. Using theoretical route as ground truth.")

            # --- INITIALIZE ML EGO CONTROLLER IF MODE IS ML ---
            if mode == "ML":
                logging.info(f"--- Assigning ML Controller to Ego Vehicle: {ego_name} ---")
                try:
                    ego_controller = EgoMLController(
                        ego_vehicle_actor, 
                        world,
                        model_path="lr_0.0001_bs_110_wspeed_0.5_frozen-20251021T053025Z-1-001/lr_0.0001_bs_110_wspeed_0.5_frozen/best_model.pth",
                        scaler_path="label_scaler.pkl",
                        model_name="nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
                    )
                except Exception as e:
                    logging.error(f"Failed to initialize ML controller: {e}")
                    return None, "ML_INIT_FAILURE", None
            else:
                 logging.info(f"--- Ego Vehicle: {ego_name} controlled by EXPERT (SCC) ---")

            spectator = world.get_spectator()
            avg_loc = np.mean([[v.vehicle.get_location().x, v.vehicle.get_location().y] for v in vehicles.values()], axis=0)
            spectator.set_transform(carla.Transform(carla.Location(x=avg_loc[0], y=avg_loc[1], z=50), carla.Rotation(pitch=-90)))
            
            start_time = world.get_snapshot().timestamp.elapsed_seconds
            sim_clock, max_sim_time = 0.0, SCENARIO_DURATION_SEC + 5.0
            
            # --- SIMULATION LOOP ---
            while sim_clock < max_sim_time:
                if all(v.get_image_count() >= IMAGES_PER_VEHICLE_PER_SCENARIO * IMAGE_CAPTURE_TOLERANCE for v in vehicles.values()):
                    end_status = "COMPLETED"; break
                if any(v.has_collided for v in vehicles.values()):
                    end_status = "COLLISION"; break
                
                world.tick()
                sim_clock += dt
                
                # --- LOGGING METRICS (Every Tick) ---
                ego_loc = ego_vehicle_actor.get_location()
                ego_vel = ego_vehicle_actor.get_velocity()
                ego_ang = ego_vehicle_actor.get_angular_velocity()
                
                ego_positions.append(ego_loc)
                ego_speeds.append(ego_vel)
                yaw_rates.append(ego_ang.z)
                
                # Collect Yaw for Overtake logic
                ego_tf = ego_vehicle_actor.get_transform()
                ego_yaws.append(math.radians(ego_tf.rotation.yaw))

                # Collect other vehicle positions for Overtake logic
                for v_name, v_wrapper in vehicles.items():
                    if v_name != ego_name:
                        if v_wrapper.vehicle.is_alive:
                            pos = v_wrapper.vehicle.get_location()
                            other_positions[v_name].append((pos.x, pos.y))

                # Track Lane ID for detection
                try:
                    wp_current = world.get_map().get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                    if wp_current:
                        lane_history.append(wp_current.lane_id)
                except:
                    pass

                # Optimized Min TTC Calculation (Nearest 40m only)
                for other in world.get_actors().filter('vehicle.*'):
                    if other.id == ego_vehicle_actor.id: continue
                    
                    if ego_loc.distance(other.get_location()) < 40.0:
                        ttc = predict_ttc(ego_loc, ego_vel, other.get_location(), other.get_velocity())
                        if ttc:
                            ego_ttc_values.append(ttc)

                # --- UPDATE VEHICLES ---
                for name, v in vehicles.items():
                    if name == ego_name and ego_controller:
                        v.is_capturing = v.start_capture_time <= sim_clock <= v.end_capture_time
                        if v.is_capturing: v.image_count += 1
                        
                        if v.vehicle and v.vehicle.is_alive:
                            loc = v.vehicle.get_location()
                            label = "AI MODEL"
                            color = carla.Color(r=255, g=0, b=0)
                            world.debug.draw_string(loc + carla.Location(z=2.5), label, draw_shadow=False, color=color, life_time=dt*1.1)

                        ego_controller.tick()
                    elif name == ego_name and mode == "EXPERT":
                         v.is_capturing = v.start_capture_time <= sim_clock <= v.end_capture_time
                         if v.is_capturing: v.image_count += 1
                         
                         if v.vehicle and v.vehicle.is_alive:
                            loc = v.vehicle.get_location()
                            label = "EXPERT"
                            color = carla.Color(r=0, g=255, b=0)
                            world.debug.draw_string(loc + carla.Location(z=2.5), label, draw_shadow=False, color=color, life_time=dt*1.1)
                            
                         v.update(sim_clock, vehicles)
                    else:
                        v.update(sim_clock, vehicles)
            else:
                end_status = "TIMEOUT"
        
        # --- POST-SCENARIO METRIC CALCULATION ---
        
        # 1. Collisions
        ego_collision_count = 1 if ego_vehicle_wrapper.has_collided else 0

        # 2. Time to First Collision
        ttfc = None
        if ego_vehicle_wrapper.time_of_first_collision:
             ttfc = ego_vehicle_wrapper.time_of_first_collision - start_time

        # Filter out short runs
        if len(ego_positions) < 10:
            ade = None
            fde = None
            jerk_rms = None
            yaw_std = None
            completion = 0.0
        else:
            # 3. ADE (Average Displacement Error) - TIME ALIGNED
            ego_xy = [(p.x, p.y) for p in ego_positions]

            if reference_traj_for_metrics and len(reference_traj_for_metrics) > 1:
                
                # match trajectory lengths (time-aligned)
                min_len = min(len(ego_xy), len(reference_traj_for_metrics))

                ade_sum = 0.0
                for i in range(min_len):
                    ex, ey = ego_xy[i]
                    rx, ry = reference_traj_for_metrics[i]
                    ade_sum += math.hypot(ex - rx, ey - ry)

                ade = ade_sum / min_len

            else:
                ade = None

            # 4. FDE (Final Displacement Error)
            if reference_traj_for_metrics:
                last_ego = ego_xy[-1]
                last_ref = reference_traj_for_metrics[-1]
                fde = math.hypot(last_ego[0] - last_ref[0], last_ego[1] - last_ref[1])
            else:
                fde = 0.0

            # 6. Jerk RMS
            jerk_rms = 0.0
            if len(ego_speeds) > 2:
                acc = []
                for i in range(1, len(ego_speeds)):
                    v1 = np.array([ego_speeds[i-1].x, ego_speeds[i-1].y, ego_speeds[i-1].z])
                    v2 = np.array([ego_speeds[i].x, ego_speeds[i].y, ego_speeds[i].z])
                    a_vec = (v2 - v1) / dt
                    acc.append(a_vec)
                
                jerks = []
                for i in range(1, len(acc)):
                    j_vec = (acc[i] - acc[i-1]) / dt
                    jerks.append(np.linalg.norm(j_vec))
                
                if jerks:
                    jerk_rms = math.sqrt(sum(j*j for j in jerks)/len(jerks))

            # 7. Angular Velocity Std Dev
            yaw_std = statistics.pstdev(yaw_rates) if len(yaw_rates) > 1 else 0.0

            # 8. Route Completion % (Distance based)
            completion = calculate_route_completion_distance_percentage(ego_positions, reference_traj_for_metrics)

        # 5. Minimum TTC
        min_ttc = min(ego_ttc_values) if ego_ttc_values else None
        
        # 9. Time Completion Percentage
        time_completion_pct = (sim_clock / SCENARIO_DURATION_SEC) * 100.0
        
        # 10. Scenario Detection Tags
        tags = []
        # Updated: Using relaxed overtake detection (Ego overtaking others)
        # We need to format trajectories for the function
        # ego_xy already computed for ADE
        ego_traj_simple = [(p.x, p.y) for p in ego_positions]
        if detect_relaxed_overtake(ego_traj_simple, ego_yaws, other_positions): 
            tags.append("overtaking")
        
        traffic_level = detect_dense_traffic(world, ego_vehicle_actor)
        tags.append(f"dense_{traffic_level}")
        
        if detect_lane_change(lane_history): tags.append("lane_change")
        
        itype = classify_intersection_type(world, ego_positions)
        if itype != "none": tags.append(f"{itype}_intersection")
        
        if detect_following(world, ego_vehicle_actor): tags.append("following")
        
        tags_str = ";".join(tags)

        # 11. Turn Type
        turn_type = classify_turn_from_trajectory(ego_positions)

        metrics = {
            "scenario_id": scenario_id,
            "mode": mode,
            "collisions": ego_collision_count,
            "ttfc": ttfc,
            "ade": ade,
            "fde": fde,
            "min_ttc": min_ttc,
            "jerk_rms": jerk_rms,
            "yaw_std": yaw_std,
            "completion": completion,
            "time_completion": time_completion_pct,
            "tags": tags_str,
            "turn_type": turn_type,
            "end_status": end_status
        }
        
        logging.info(f"[METRICS] {metrics}")

        # Return metrics AND the driven path (for use as ground truth in ML run)
        # We return simple list of tuples [(x,y), ...] to be pickleable/simple
        driven_path_xy = [(p.x, p.y) for p in ego_positions]
        
        return metrics, end_status, driven_path_xy

    finally:
        # Cleanup ML Controller (Memory Fix)
        if ego_controller:
            ego_controller.destroy()
            del ego_controller
        
        # Explicitly clear CUDA cache and Python garbage
        torch.cuda.empty_cache()
        gc.collect()
            
        if vehicles:
            logging.info(f"[SCENARIO {scenario_id}] Cleaning up {len(vehicles)} vehicles.")
            for v in vehicles.values(): v.destroy()
            time.sleep(0.2)

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def parse_tids_file(filepath):
    if not os.path.exists(filepath): raise FileNotFoundError(f"TID file not found: {filepath}")
    region_tids = {}
    with open(filepath, 'r') as f: content = f.read()
    for match in re.findall(r'(\d+)\.\s*\{([\d\s,]+)\}', content):
        region_tids[int(match[0])] = sorted([int(i) for i in re.split(r'[\s,]+', match[1].strip()) if i])
    logging.info(f"Loaded track IDs for {len(region_tids)} regions from {filepath}.")
    return region_tids

def find_common_interaction_frame(vehicle_ids, parser, frame_range):
    if not vehicle_ids or len(vehicle_ids) < 2: return None
    list_of_frame_sets = [set(parser.get_frames_for_track(tid)) for tid in vehicle_ids]
    if not all(list_of_frame_sets): return None
    common_frames = reduce(set.intersection, list_of_frame_sets)
    start, end = frame_range
    window = int(SCENARIO_DURATION_SEC * REAL_WORLD_FPS)
    valid_frames = [f for f in common_frames if start <= f <= end - (window - 1)]
    return random.choice(valid_frames) if valid_frames else None

def find_best_vehicle_group(vehicle_ids, parser, frame_range):
    if len(vehicle_ids) < MIN_ACTORS_FOR_SCENARIO: return None, None
    for size in range(len(vehicle_ids), MIN_ACTORS_FOR_SCENARIO - 1, -1):
        for subgroup in combinations(vehicle_ids, size):
            frame = find_common_interaction_frame(list(subgroup), parser, frame_range)
            if frame: return list(subgroup), frame
    return None, None

def find_consecutive_groups(tids, min_size, split_groups=True):
    if not tids or len(tids) < min_size:
        return []

    sorted_tids = sorted(list(set(tids))) 
    all_sequences = []
    if not sorted_tids: return []
    
    current_sequence = [sorted_tids[0]]
    for i in range(1, len(sorted_tids)):
        if sorted_tids[i] == sorted_tids[i-1] + 1:
            current_sequence.append(sorted_tids[i])
        else:
            if len(current_sequence) > 0:
                all_sequences.append(current_sequence)
            current_sequence = [sorted_tids[i]]
    
    if len(current_sequence) > 0:
        all_sequences.append(current_sequence)

    final_groups = []
    for seq in all_sequences:
        if len(seq) < min_size:
            continue
        
        if split_groups:
            num_splits = 1
            if len(seq) > 14:
                num_splits = 3
            elif len(seq) > 7:
                num_splits = 2
            
            if num_splits > 1:
                split_groups_list = np.array_split(seq, num_splits)
                for group in split_groups_list:
                    group_list = group.tolist()
                    if len(group_list) >= min_size:
                        final_groups.append(group_list)
            else:
                final_groups.append(seq)
        else:
            final_groups.append(seq)
            
    return final_groups

def cleanup_world(client):
    try:
        world = client.get_world()
        vehicle_list = world.get_actors().filter('vehicle.*')
        if vehicle_list:
            logging.info(f"Destroying {len(vehicle_list)} vehicles from previous runs.")
            client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])
        time.sleep(0.5)
    except Exception as e:
        logging.error(f"Error during world cleanup: {e}")

def main():
    setup_logging()
    
    # Initialize Stats File
    with open(STATS_FILE, "w") as f:
        # Added TimeCompletionPct and Tags to header
        header = "ScenarioID,Mode,Collisions,TTFC,ADE,FDE,MinTTC,JerkRMS,YawStd,Completion,TimeCompletionPct,Tags,TurnType,EndStatus\n"
        f.write(header)

    if os.path.exists(SNAPSHOT_DIR): shutil.rmtree(SNAPSHOT_DIR)
    os.makedirs(SNAPSHOT_DIR)

    parser = AnnotationParser(INPUT_FILE)
    region_tids = parse_tids_file(TIDS_FILE)
    
    processed_track_ids, total_images_captured, scenario_count = set(), 0, 0

    client = None
    world = None
    original_settings = None
    try:
        client = carla.Client(CARLA_HOST, CARLA_PORT)
        client.set_timeout(60.0)

        cleanup_world(client)

        world = client.get_world()
        original_settings = world.get_settings()
        
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / SIM_FPS
        world.apply_settings(settings)
        time.sleep(2)
        
        for region_num, frame_range in sorted(FRAME_REGIONS.items(), reverse=True):
            if total_images_captured >= TOTAL_IMAGES_GOAL: break

            logging.info(f"{'='*60}\nProcessing Region {region_num} (Frames {frame_range[0]}-{frame_range[1]})")
            
            eligible_ids = sorted([tid for tid in region_tids.get(region_num, []) if tid not in processed_track_ids])
            
            # --- HELPER TO RUN PAIR OF SCENARIOS ---
            def process_group(group_to_run):
                nonlocal scenario_count, total_images_captured
                
                # SHUFFLE ONCE HERE to determine Ego identity for both runs
                current_group_shuffled = list(group_to_run)
                random.shuffle(current_group_shuffled)
                
                scenario_count += 1
                
                if scenario_count > 1 and scenario_count % SCENARIOS_PER_WORLD_RELOAD == 0:
                    client.reload_world()
                    world_obj = client.get_world()
                    world_obj.apply_settings(settings)
                    time.sleep(2)
                else:
                    world_obj = world

                # --- RUN 1: EXPERT (Capture Reference) ---
                logging.info(f"--- Starting Scenario {scenario_count} [EXPERT MODE] ---")
                
                metrics_exp, status_exp, expert_path = run_scenario(world_obj, scenario_count, current_group_shuffled, parser, interaction_frame, mode="EXPERT")
                
                if metrics_exp:
                    with open(STATS_FILE, "a") as f:
                        # Added TimeCompletionPct and Tags to output string
                        line = f"{metrics_exp['scenario_id']},{metrics_exp['mode']},{metrics_exp['collisions']},{metrics_exp['ttfc']},{metrics_exp['ade'] or ''},{metrics_exp['fde'] or ''},{metrics_exp['min_ttc'] or ''},{metrics_exp['jerk_rms'] or ''},{metrics_exp['yaw_std'] or ''},{metrics_exp['completion']:.2f},{metrics_exp['time_completion']:.2f},\"{metrics_exp['tags']}\",{metrics_exp['turn_type']},{metrics_exp['end_status']}\n"
                        f.write(line)

                time.sleep(1) 

                # --- RUN 2: ML MODEL (Compare against captured Expert Path) ---
                logging.info(f"--- Starting Scenario {scenario_count} [ML MODE] ---")
                
                metrics_ml, status_ml, _ = run_scenario(world_obj, scenario_count, current_group_shuffled, parser, interaction_frame, mode="ML", expert_trajectory=expert_path)
                
                if metrics_ml:
                    with open(STATS_FILE, "a") as f:
                        line = f"{metrics_ml['scenario_id']},{metrics_ml['mode']},{metrics_ml['collisions']},{metrics_ml['ttfc']},{metrics_ml['ade'] or ''},{metrics_ml['fde'] or ''},{metrics_ml['min_ttc'] or ''},{metrics_ml['jerk_rms'] or ''},{metrics_ml['yaw_std'] or ''},{metrics_ml['completion']:.2f},{metrics_ml['time_completion']:.2f},\"{metrics_ml['tags']}\",{metrics_ml['turn_type']},{metrics_ml['end_status']}\n"
                        f.write(line)

                total_images_captured += 100 

            if region_num in (1, 2):
                candidate_groups = find_consecutive_groups(eligible_ids, MIN_ACTORS_FOR_SCENARIO, SPLIT_LARGE_GROUPS)
                logging.info(f"Found {len(candidate_groups)} potential consecutive groups in Region {region_num}.")
                
                for group in candidate_groups:
                    if total_images_captured >= TOTAL_IMAGES_GOAL: break
                    if any(tid in processed_track_ids for tid in group): continue
                    
                    scenario_vehicles, interaction_frame = find_best_vehicle_group(group, parser, frame_range)
                    
                    if scenario_vehicles and interaction_frame:
                        processed_track_ids.update(group)
                        process_group(scenario_vehicles)
                    else:
                        processed_track_ids.update(group)
            else:
                logging.info(f"Using iterative distance-based grouping for Region {region_num}.")
                while eligible_ids:
                    if total_images_captured >= TOTAL_IMAGES_GOAL: break

                    anchor_tid = eligible_ids[0]
                    
                    interaction_frame = frame_range[0]
                    all_positions = parser.get_track_positions_at_frame(interaction_frame)

                    if anchor_tid not in all_positions:
                        eligible_ids.pop(0)
                        processed_track_ids.add(anchor_tid)
                        continue
                    
                    anchor_pos = all_positions[anchor_tid]
                    new_group = [anchor_tid]
                    
                    for other_tid in eligible_ids[1:]:
                        if other_tid in all_positions:
                            other_pos = all_positions[other_tid]
                            distance = math.hypot(anchor_pos[0] - other_pos[0], anchor_pos[1] - other_pos[1])
                            if distance <= DISTANCE_THRESHOLD:
                                new_group.append(other_tid)

                    processed_track_ids.update(new_group)
                    eligible_ids = [tid for tid in eligible_ids if tid not in new_group]

                    if len(new_group) >= MIN_ACTORS_FOR_SCENARIO:
                        window = int(SCENARIO_DURATION_SEC * REAL_WORLD_FPS)
                        if interaction_frame > frame_range[1] - (window - 1):
                            continue
                        
                        process_group(new_group)

    except KeyboardInterrupt:
        logging.info("--- User interrupted script ---")
    except Exception as e:
        logging.critical(f"CRITICAL ERROR: {e}", exc_info=True)
    finally:
        if client:
            cleanup_world(client)
            if world and original_settings:
                world.apply_settings(original_settings)
        logging.info("Script finished.")

if __name__ == "__main__":
    main()