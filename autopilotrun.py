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

INPUT_FILE = "selected_annotations.json"
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
    2: (178, 327),
    3: (328, 475),
    4: (476, 590),
    5: (591, 779),
    # 6: (780, 910),
    # 7: (911, 944),
    # 1: (0, 177),
}

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
                fwd = ego_actor.get_transform().get_forward_vector()
                vec_to_veh = carla.Vector3D(veh_loc.x - ego_actor.get_location().x, veh_loc.y - ego_actor.get_location().y, 0)
                
                vec_len = math.hypot(vec_to_veh.x, vec_to_veh.y)
                if vec_len > 0:
                    dot = (fwd.x * vec_to_veh.x + fwd.y * vec_to_veh.y) / vec_len
                    if dot > 0.5: 
                         if d < 40 and d > 2: 
                            if ego_speed > 0.5:
                                return True
    except Exception:
        pass 
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
    first_chunk = list(lane_history)[:5]
    last_chunk = list(lane_history)[-5:]
    
    start_lane = max(set(first_chunk), key=first_chunk.count)
    end_lane = max(set(last_chunk), key=last_chunk.count)

    return start_lane != end_lane

# --- Relaxed Overtaking Detection ---
def longitudinal_projection(ego_xy, ego_yaw, other_xy):
    ex, ey = ego_xy
    ox, oy = other_xy
    rx = ox - ex
    ry = oy - ey
    heading = np.array([np.cos(ego_yaw), np.sin(ego_yaw)])
    return rx * heading[0] + ry * heading[1]

def detect_relaxed_overtake(ego_traj, ego_yaws, other_trajs):
    if len(ego_traj) < 2: return False

    ego_start = ego_traj[0]
    ego_end   = ego_traj[-1]
    yaw_start = ego_yaws[0]
    yaw_end   = ego_yaws[-1]

    for vid, traj in other_trajs.items():
        if len(traj) < 2: continue
        other_start = traj[0]
        other_end   = traj[-1]

        d0 = longitudinal_projection(ego_start, yaw_start, other_start)
        d1 = longitudinal_projection(ego_end,   yaw_end,   other_end)

        # Ego passes someone: They started Ahead (>0) and ended Behind (<0)
        if d0 > 0 and d1 < 0:
            return True

    return False

def classify_intersection_type(world, ego_positions):
    m = world.get_map()
    step = max(1, len(ego_positions) // 10)
    for i in range(0, len(ego_positions), step):
        p = ego_positions[i]
        loc = p if isinstance(p, carla.Location) else carla.Location(x=p.x, y=p.y, z=p.z)
        try:
            wp = m.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp and wp.is_junction:
                junction = wp.get_junction()
                topology = junction.get_waypoints(carla.LaneType.Driving)
                road_connections = set()
                for a, b in topology:
                    road_connections.add(a.road_id)
                    road_connections.add(b.road_id)
                n = len(road_connections)
                if n == 3: return "3-way"
                if n >= 4: return "4-way"
        except Exception:
            pass
    return "none"

def classify_turn_from_trajectory(ego_positions):
    if len(ego_positions) < 5: return "unknown"
    yaws = []
    for i in range(1, len(ego_positions)):
        p1 = ego_positions[i-1]
        p2 = ego_positions[i]
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        if abs(dx) + abs(dy) < 1e-4: continue
        yaw = math.degrees(math.atan2(dy, dx))
        yaws.append(yaw)

    if len(yaws) < 2: return "unknown"

    initial = yaws[0]
    final = yaws[-1]
    delta = final - initial
    while delta > 180: delta -= 360
    while delta < -180: delta += 360
    abs_delta = abs(delta)

    if abs_delta < 15: return "straight"
    if 15 <= abs_delta < 60: return "slight_left_turn" if delta > 0 else "slight_right_turn"
    return "left_turn" if delta > 0 else "right_turn"


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


def safe_next_target(world, ego, base_target_loc, base_target_speed, horizon=AVOIDANCE_HORIZON_S):
    # Simplified for non-ego vehicles
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
    
    return base_target_loc, 0.0, True # Just stop if threat


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
    def __init__(self, world, bp, name, start, yaw, output_path, start_delay=0, start_capture_time=0, end_capture_time=float('inf'), is_ego=False, scenario_id=0):
        super().__init__(world, bp, name, start, yaw, start_delay)
        self.output_path = os.path.join(output_path, self.name)
        self.image_path = None 
        
        self.collision_sensor = None
        self.image_count = 0
        self.has_collided = False
        self.start_capture_time = start_capture_time
        self.end_capture_time = end_capture_time
        self.is_capturing = False
        self.time_of_first_collision = None
        
        # --- SNAPSHOT CONFIG ---
        self.is_ego = is_ego
        self.scenario_id = scenario_id
        self.snapshot_taken = False
        
        self._setup_sensors()

    def _setup_sensors(self):
        # 1. Collision Sensor (Always)
        collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: CapturingVehicle._on_collision(weakref.ref(self), event))

        # 2. Camera (Only if Ego - for snapshot)
        if self.is_ego:
            # Bird's Eye View (Top Down)
            bp_bird = self.world.get_blueprint_library().find('sensor.camera.rgb')
            bp_bird.set_attribute('image_size_x', '1024')
            bp_bird.set_attribute('image_size_y', '1024')
            bp_bird.set_attribute('fov', '90')
            tf_bird = carla.Transform(carla.Location(x=0, z=30), carla.Rotation(pitch=-90))
            
            self.camera = self.world.spawn_actor(bp_bird, tf_bird, attach_to=self.vehicle)
            self.camera.listen(lambda image: CapturingVehicle._save_snapshot(weakref.ref(self), image))

    @staticmethod
    def _save_snapshot(weak_self, image):
        self = weak_self()
        if not self: return
        
        # Only take snapshot if capturing, not already taken, and frame count > 20 (settle time)
        if self.is_capturing and self.image_count > 20 and not self.snapshot_taken:
            filename = f"scenario_{self.scenario_id}_autopilot.png"
            filepath = os.path.join(SNAPSHOT_DIR, filename)
            image.save_to_disk(filepath)
            logging.info(f"📸 Snapshot saved: {filepath}")
            self.snapshot_taken = True

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

    def update(self, simulation_time, sim_vehicles):
        self.is_capturing = self.start_capture_time <= simulation_time <= self.end_capture_time
        if self.is_capturing:
            self.image_count += 1
        
        # For non-ego vehicles, we still run simple avoidance logic
        if not self.is_ego:
             # Basic safety
             pass
             
        return super().update(simulation_time, sim_vehicles)

    def save_captured_data(self):
        return

    def destroy(self):
        if hasattr(self, 'camera') and self.camera and self.camera.is_listening: self.camera.stop()
        if self.collision_sensor and self.collision_sensor.is_listening: self.collision_sensor.stop()
        
        time.sleep(0.05)
        
        if hasattr(self, 'camera') and self.camera and self.camera.is_alive: self.camera.destroy()
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
    if not expert_path or len(expert_path) < 2: return 0.0
    if not driven_path: return 0.0

    total_length = 0.0
    for i in range(len(expert_path) - 1):
        total_length += math.hypot(expert_path[i+1][0] - expert_path[i][0], 
                                   expert_path[i+1][1] - expert_path[i][1])
    
    if total_length == 0: return 100.0

    last_pos = driven_path[-1]
    last_x, last_y = (last_pos.x, last_pos.y) if isinstance(last_pos, carla.Location) else last_pos

    min_dist = float('inf')
    closest_idx = 0

    for i, pt in enumerate(expert_path):
        dist = math.hypot(last_x - pt[0], last_y - pt[1])
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
            
    covered_dist = 0.0
    for i in range(closest_idx):
        covered_dist += math.hypot(expert_path[i+1][0] - expert_path[i][0], 
                                   expert_path[i+1][1] - expert_path[i][1])
                                   
    return min(100.0, (covered_dist / total_length) * 100.0)


# -----------------------------
# Main Scenario Runner
# -----------------------------
def run_scenario(world, scenario_id, all_track_ids, annotation_parser, interaction_frame):
    vehicles, spawned_vehicles_actors = {}, []
    end_status = "UNKNOWN"
    
    ego_name = None

    # --- METRIC CONTAINERS ---
    ego_positions = []
    ego_speeds = []
    yaw_rates = []
    ego_ttc_values = []
    lane_history = deque(maxlen=50)
    
    ego_yaws = []
    other_positions = {} 

    try:
        dt = world.get_settings().fixed_delta_seconds
        bp_library = world.get_blueprint_library()
        vehicle_bp = bp_library.find(VEHICLE_BLUEPRINT)
        num_frames_required = int(SCENARIO_DURATION_SEC * REAL_WORLD_FPS)

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
                
                sim_v = CapturingVehicle(world, vehicle_bp, vname, 
                                         (safe_spawn_point.x, safe_spawn_point.y, safe_spawn_point.z), 
                                         initial_yaw, "dummy_path", 
                                         start_capture_time=0.0, end_capture_time=SCENARIO_DURATION_SEC,
                                         is_ego=is_ego, scenario_id=scenario_id)
                                         
                sim_v.set_dynamic_route(dynamic_route)
                vehicles[vname] = sim_v
                spawned_vehicles_actors.append(sim_v.vehicle)
                world.tick()
            except Exception as e:
                logging.error(f"[SCENARIO {scenario_id}] FAILED TO SPAWN {tid}: {e}")
                continue

        if len(vehicles) < 2:
            return {
                "scenario_id": scenario_id,
                "mode": "AUTOPILOT",
                "collisions": 0,
                "ttfc": None,
                "ade": None,
                "fde": None,
                "min_ttc": None,
                "jerk_rms": None,
                "yaw_std": None,
                "completion": 0.0,
                "time_completion": 0.0,
                "tags": "spawn_failure",
                "turn_type": "unknown",
                "end_status": "SPAWN_FAILURE"
            }

        logging.info(f"[SCENARIO {scenario_id} - AUTOPILOT] Spawned {len(vehicles)} vehicles. Starting sim.")
        
        ego_name = list(vehicles.keys())[0] # Pick the first vehicle as Ego
        ego_vehicle_wrapper = vehicles[ego_name]
        ego_vehicle_actor = ego_vehicle_wrapper.vehicle
        
        # --- ENABLE AUTOPILOT FOR EGO ---
        ego_vehicle_actor.set_autopilot(True)
        logging.info(f"--- Ego Vehicle: {ego_name} controlled by AUTOPILOT ---")
        
        # --- PREP OVERTAKE CONTAINERS ---
        other_positions = {v_name: [] for v_name in vehicles if v_name != ego_name}

        # --- SETUP REFERENCE TRAJECTORY FROM ANNOTATIONS ---
        # NOTE: dense_route format is usually [(Waypoint, speed), ...] or similar from scc/timemethod
        # We need to extract x,y from it.
        reference_traj_for_metrics = []
        for item in ego_vehicle_wrapper.dense_route:
            # Handles different formats if dense_route changes
            if hasattr(item, 'transform'): # direct waypoint object
                 loc = item.transform.location
                 reference_traj_for_metrics.append((loc.x, loc.y))
            elif isinstance(item, (tuple, list)):
                if hasattr(item[0], 'transform'): # (Waypoint, Speed)
                    loc = item[0].transform.location
                    reference_traj_for_metrics.append((loc.x, loc.y))
                elif len(item) >= 2: # (x, y, z, speed)
                    reference_traj_for_metrics.append((item[0], item[1]))

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
            
            ego_tf = ego_vehicle_actor.get_transform()
            ego_yaws.append(math.radians(ego_tf.rotation.yaw))

            for v_name, v_wrapper in vehicles.items():
                if v_name != ego_name:
                    if v_wrapper.vehicle.is_alive:
                        pos = v_wrapper.vehicle.get_location()
                        other_positions[v_name].append((pos.x, pos.y))

            try:
                wp_current = world.get_map().get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                if wp_current:
                    lane_history.append(wp_current.lane_id)
            except:
                pass

            for other in world.get_actors().filter('vehicle.*'):
                if other.id == ego_vehicle_actor.id: continue
                if ego_loc.distance(other.get_location()) < 40.0:
                    ttc = predict_ttc(ego_loc, ego_vel, other.get_location(), other.get_velocity())
                    if ttc:
                        ego_ttc_values.append(ttc)

            # --- UPDATE VEHICLES ---
            for name, v in vehicles.items():
                if name == ego_name:
                    v.is_capturing = v.start_capture_time <= sim_clock <= v.end_capture_time
                    if v.is_capturing: v.image_count += 1
                    
                    if v.vehicle and v.vehicle.is_alive:
                        loc = v.vehicle.get_location()
                        label = "AUTOPILOT"
                        color = carla.Color(r=0, g=0, b=255) # BLUE
                        world.debug.draw_string(loc + carla.Location(z=2.5), label, draw_shadow=False, color=color, life_time=dt*1.1)
                    
                    # No manual control applied here, autopilot handles it
                    v.update(sim_clock, vehicles) # Still call update for capture logic
                else:
                    v.update(sim_clock, vehicles)
        else:
            end_status = "TIMEOUT"
    
        # --- METRIC CALCULATION ---
        ego_collision_count = 1 if ego_vehicle_wrapper.has_collided else 0
        
        ttfc = None
        if ego_vehicle_wrapper.time_of_first_collision:
             ttfc = ego_vehicle_wrapper.time_of_first_collision - start_time

        if len(ego_positions) < 10:
            ade, fde, jerk_rms, yaw_std, completion = None, None, None, None, 0.0
        else:
            ego_xy = [(p.x, p.y) for p in ego_positions]

            if reference_traj_for_metrics and len(reference_traj_for_metrics) > 1:
                min_len = min(len(ego_xy), len(reference_traj_for_metrics))
                ade_sum = 0.0
                for i in range(min_len):
                    ex, ey = ego_xy[i]
                    rx, ry = reference_traj_for_metrics[i]
                    ade_sum += math.hypot(ex - rx, ey - ry)
                ade = ade_sum / min_len
            else:
                ade = None

            if reference_traj_for_metrics:
                last_ego = ego_xy[-1]
                last_ref = reference_traj_for_metrics[-1]
                fde = math.hypot(last_ego[0] - last_ref[0], last_ego[1] - last_ref[1])
            else:
                fde = 0.0

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

            yaw_std = statistics.pstdev(yaw_rates) if len(yaw_rates) > 1 else 0.0
            completion = calculate_route_completion_distance_percentage(ego_positions, reference_traj_for_metrics)

        min_ttc = min(ego_ttc_values) if ego_ttc_values else None
        time_completion_pct = (sim_clock / SCENARIO_DURATION_SEC) * 100.0
        
        tags = []
        ego_traj_simple = [(p.x, p.y) for p in ego_positions]
        if detect_relaxed_overtake(ego_traj_simple, ego_yaws, other_positions): tags.append("overtaking")
        traffic_level = detect_dense_traffic(world, ego_vehicle_actor)
        tags.append(f"dense_{traffic_level}")
        if detect_lane_change(lane_history): tags.append("lane_change")
        itype = classify_intersection_type(world, ego_positions)
        if itype != "none": tags.append(f"{itype}_intersection")
        if detect_following(world, ego_vehicle_actor): tags.append("following")
        tags_str = ";".join(tags)

        turn_type = classify_turn_from_trajectory(ego_positions)

        metrics = {
            "scenario_id": scenario_id,
            "mode": "AUTOPILOT",
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
        return metrics

    finally:
        if vehicles:
            logging.info(f"[SCENARIO {scenario_id}] Cleaning up {len(vehicles)} vehicles.")
            
            # --- FIX 1: DISABLE AUTOPILOT BEFORE DESTROY ---
            for v in vehicles.values():
                if v.vehicle and v.vehicle.is_alive:
                    try:
                        v.vehicle.set_autopilot(False)
                    except:
                        pass
            
            if world:
                try:
                    world.tick() # Ensure disabling processes
                except:
                    pass
            
            for v in vehicles.values(): 
                v.destroy()
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
    
    with open(STATS_FILE, "w") as f:
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
            
            # --- HELPER TO RUN SCENARIO ONCE (AUTOPILOT) ---
            def process_group(group_to_run):
                nonlocal scenario_count, total_images_captured
                
                # Shuffle once to pick random ego
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

                logging.info(f"--- Starting Scenario {scenario_count} [AUTOPILOT MODE] ---")
                
                metrics = run_scenario(world_obj, scenario_count, current_group_shuffled, parser, interaction_frame)
                
                if metrics:
                    with open(STATS_FILE, "a") as f:
                        line = f"{metrics['scenario_id']},{metrics['mode']},{metrics['collisions']},{metrics['ttfc']},{metrics['ade'] or ''},{metrics['fde'] or ''},{metrics['min_ttc'] or ''},{metrics['jerk_rms'] or ''},{metrics['yaw_std'] or ''},{metrics['completion']:.2f},{metrics['time_completion']:.2f},\"{metrics['tags']}\",{metrics['turn_type']},{metrics['end_status']}\n"
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