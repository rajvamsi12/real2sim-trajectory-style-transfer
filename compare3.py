import os
import json
import math
import time
import carla
import random
import numpy as np
import shutil
import logging
import re
import csv
import queue
import hashlib
import weakref
from functools import reduce
from itertools import combinations

# --- ML Imports ---
import torch
from PIL import Image
import pickle5 as pickle
from transformers import AutoImageProcessor

# 1. Deterministic Reproducibility
random.seed(84)
np.random.seed(84)
torch.manual_seed(84)
torch.set_grad_enabled(False)

# --- Custom Module Imports ---
try:
    from train3 import BehavioralCloningSegFormer
    from timemethod import transform_trajectory_with_analysis
    from scc import SimVehicle, DummyWaypoint
except ImportError as e:
    print(f"CRITICAL ERROR: Missing required modules. {e}")
    exit()

# ==========================================
# STEP 0: CONSTANTS & CONFIGURATION
# ==========================================
SCENARIO_DURATION_SEC = 15
SIM_FPS = 20
REAL_WORLD_FPS = 4
CARLA_HOST = "localhost"
CARLA_PORT = 2000
TOWN = "Town10HD"
VEHICLE_BLUEPRINT = "vehicle.bmw.grandtourer"
DEFAULT_Z = 0.6
MIN_ACTORS_FOR_SCENARIO = 3
MAX_ACTORS_FOR_SCENARIO = 10 
DISTANCE_THRESHOLD = 100.0
SPEED_CORRECTION_FACTOR = 4

MAX_SCENARIOS = 100 
TOTAL_IMAGES_GOAL = 10000 
SCENARIOS_PER_WORLD_RELOAD = 10
SPLIT_LARGE_GROUPS = False

MODES = ["EXPERT", "ML", "AUTOPILOT"] 
EGO_COLOR = {
    "AUTOPILOT": "0,0,255", # Blue
    "EXPERT": "0,255,0",    # Green
    "ML": "255,0,0"         # Red
}

# --- Shared Config from Scenarios 6/7 ---
SPAWN_SAFETY_MARGIN = 3.5
MAX_RETRACTION_ATTEMPTS = 10
RETRACTION_STEP = 0.5
AVOIDANCE_HORIZON_S = 3.5
AVOIDANCE_IMMINENT_TTC_S = 1.5

REPLAY_DIR = "scenario_replays"
VIDEO_DIR = "scenario_videos"
STATS_FILE = "scenario_stats.csv"
METADATA_FILE = "scenario_metadata.csv"
INPUT_FILE = "selected_annotations.json"
TIDS_FILE = "tids.txt"

FRAME_REGIONS = {
    1: (0, 177),
    2: (178, 327),
    3: (328, 475),
    4: (476, 590),
    5: (591, 779),
    6: (780, 910),
    7: (911, 944),
}

EGO_LATEST_IMAGE = None
TM = None

def ego_image_callback(image):
    global EGO_LATEST_IMAGE
    EGO_LATEST_IMAGE = image

# ==========================================
# STEP 1: HYBRID AVOIDANCE HELPERS
# ==========================================
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
    nudged = carla.Location(target_loc.x + side.x * max_offset, target_loc.y + side.y * max_offset, target_loc.z)
    wp = world.get_map().get_waypoint(nudged, project_to_road=True, lane_type=carla.LaneType.Driving)
    return wp.transform.location if wp else None

def blend_back(current_loc, annotated_loc, alpha=0.2):
    return carla.Location(
        x=alpha*annotated_loc.x + (1-alpha)*current_loc.x,
        y=alpha*annotated_loc.y + (1-alpha)*current_loc.y,
        z=annotated_loc.z
    )

def safe_next_target(world, ego, base_target_loc, base_target_speed, horizon=AVOIDANCE_HORIZON_S):
    ego_loc, ego_vel = ego.get_location(), ego.get_velocity()
    threats = []
    for a in world.get_actors().filter('vehicle.*'):
        if a.id == ego.id: continue
        a_loc = a.get_location()
        if ego_loc.distance(a_loc) > 35.0: continue
        ttc = predict_ttc(ego_loc, ego_vel, a_loc, a.get_velocity(), horizon=horizon)
        if ttc is not None: threats.append((ttc, a, a_loc))
    if not threats: return base_target_loc, base_target_speed, False
    threats.sort(key=lambda x: x[0])
    ttc, primary_threat, threat_loc = threats[0]
    if ttc < AVOIDANCE_IMMINENT_TTC_S: return base_target_loc, 0.0, True
    nudged = avoidance_adjustment(world, ego, base_target_loc, threat_loc, max_offset=0.8)
    if nudged:
        nudge_is_safe = True
        predicted_vel = (nudged - ego_loc).make_unit_vector() * ego_vel.length()
        for other_ttc, other_threat, other_threat_loc in threats[1:]:
            future_ttc = predict_ttc(ego_loc, predicted_vel, other_threat_loc, other_threat.get_velocity(), horizon=horizon)
            if future_ttc is not None and future_ttc < 1.0:
                nudge_is_safe = False; break
        if nudge_is_safe: return nudged, max(0.833, 0.5*base_target_speed), True
    return base_target_loc, max(0.833, 0.5*base_target_speed), True

# ==========================================
# STEP 2: ANNOTATION PARSER
# ==========================================
class AnnotationParser:
    def __init__(self, filepath):
        self.track_data = {}
        self.frame_data = {}
        self._load_and_parse(filepath)

    def _load_and_parse(self, filepath):
        logging.info(f"Loading annotations from {filepath}...")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        with open(filepath, 'r') as f:
            data = json.load(f)
        for frame_num, item in enumerate(data.get('items', [])):
            self.frame_data[frame_num] = []
            for ann in item.get('annotations', []):
                tid = ann.get('attributes', {}).get('track_id')
                bbox = tuple(ann.get('bbox'))
                if tid is not None and bbox:
                    self.frame_data[frame_num].append((tid, bbox))
                    if tid not in self.track_data:
                        self.track_data[tid] = {'trajectory': {}}
                    self.track_data[tid]['trajectory'][frame_num] = bbox

    def get_sliced_trajectory_coords(self, tid, start_frame, num_frames):
        coords = []
        if tid in self.track_data:
            end_frame = start_frame + num_frames
            for f in range(start_frame, end_frame):
                if f in self.track_data[tid]['trajectory']:
                    x, y, w, h = self.track_data[tid]['trajectory'][f]
                    coords.append((x + w/2, -(y + h/2)))
        return coords

    def get_positions_in_frame(self, frame):
        pos = {}
        if frame in self.frame_data:
            for item in self.frame_data[frame]:
                tid, bbox = item[0], item[1]
                x, y, w, h = bbox
                pos[tid] = (x + w/2, -(y + h/2))
        return pos
    
    def get_frames_for_track(self, tid):
        return sorted(self.track_data[tid]['trajectory'].keys()) if tid in self.track_data else []

# ==========================================
# STEP 3: REPLAY VEHICLE WITH AVOIDANCE
# ==========================================
class ReplayVehicle(SimVehicle):
    def __init__(self, world, bp, name, start_pose, is_ego=False, scenario_id=0, mode="EXPERT"):
        super().__init__(world, bp, name, (start_pose.location.x, start_pose.location.y, start_pose.location.z), start_pose.rotation.yaw, 0)
        self.is_ego, self.mode, self.scenario_id, self.cameras = is_ego, mode, scenario_id, []
        self.local_frame = 0
        self._recent_risky = 0
        self._blend_alpha = 0.2
        self._safe_frames_to_blend = 10
        if self.is_ego: self._setup_cameras()

    def _setup_cameras(self):
        base_dir = os.path.join(VIDEO_DIR, f"scenario_{self.scenario_id}", self.mode)
        self.ego_dir, self.bird_dir = os.path.join(base_dir, "ego"), os.path.join(base_dir, "bird")
        os.makedirs(self.ego_dir, exist_ok=True); os.makedirs(self.bird_dir, exist_ok=True)
        
        bp_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp_cam.set_attribute('image_size_x', '640'); bp_cam.set_attribute('image_size_y', '480')
        self.cam_ego = self.world.spawn_actor(bp_cam, carla.Transform(carla.Location(x=1.5, z=2.3)), attach_to=self.vehicle)
        self.cam_ego.listen(lambda img: self._save_img(img, self.ego_dir, True))
        
        bp_bird = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp_bird.set_attribute('image_size_x', '800'); bp_bird.set_attribute('image_size_y', '800')
        self.cam_bird = self.world.spawn_actor(bp_bird, carla.Transform(carla.Location(x=0, z=40), carla.Rotation(pitch=-90)), attach_to=self.vehicle)
        self.cam_bird.listen(lambda img: self._save_img(img, self.bird_dir, False))
        self.cameras.extend([self.cam_ego, self.cam_bird])

    def _save_img(self, image, path, is_ego):
        if is_ego: ego_image_callback(image)
        if self.local_frame % 5 == 0:
            image.save_to_disk(f"{path}/{self.local_frame:06d}.png")

    def _apply_avoidance(self):
        """Active Avoidance and Path Blending."""
        if not hasattr(self, "dense_route") or not self.vehicle or not self.vehicle.is_alive or self.wp_index >= len(self.dense_route): 
            return
        
        original_wp, original_spd = self.dense_route[self.wp_index]
        adj_loc, adj_spd, risky = safe_next_target(self.world, self.vehicle, original_wp.transform.location, original_spd)

        if risky:
            # Active Avoidance: Apply nudge/halt to the temporary route
            self.dense_route[self.wp_index] = (DummyWaypoint(carla.Transform(adj_loc, original_wp.transform.rotation)), adj_spd)
            self._recent_risky = 0
        elif 0 < self._recent_risky < self._safe_frames_to_blend:
            # Path Blending: Smoothly return to annotated route
            blended = blend_back(self.vehicle.get_transform().location, original_wp.transform.location, self._blend_alpha)
            self.dense_route[self.wp_index] = (DummyWaypoint(carla.Transform(blended, original_wp.transform.rotation)), original_spd)
            self._recent_risky += 1
        else:
            self._recent_risky = max(self._recent_risky, self._safe_frames_to_blend + 1)

    def update(self, simulation_time, sim_vehicles):
        try:
            if not (self.is_ego and self.mode == "AUTOPILOT"):
                self._apply_avoidance()
        except: pass
        return super().update(simulation_time, sim_vehicles)

    def destroy(self):
        for cam in self.cameras:
            if cam.is_listening: cam.stop()
            if cam.is_alive: cam.destroy()
        super().destroy()

# ==========================================
# STEP 4: ML CONTROLLER
# ==========================================
class EgoMLController:
    def __init__(self, vehicle, model_path, scaler_path, model_name):
        self.vehicle, self.device = vehicle, torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BehavioralCloningSegFormer(model_name=model_name).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, pickle_module=pickle))
        self.model.eval()
        with open(scaler_path, "rb") as f: self.scaler = pickle.load(f)
        self.processor, self.integral, self.prev_error = AutoImageProcessor.from_pretrained(model_name), 0.0, 0.0

    def tick(self):
        if EGO_LATEST_IMAGE is None: return
        array = np.frombuffer(EGO_LATEST_IMAGE.raw_data, dtype=np.uint8).reshape((EGO_LATEST_IMAGE.height, EGO_LATEST_IMAGE.width, 4))
        pil_img = Image.fromarray(array[:, :, :3][:, :, ::-1])
        with torch.no_grad():
            inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
            preds = self.scaler.inverse_transform(self.model(inputs["pixel_values"]).cpu().numpy())[0]
        v = self.vehicle.get_velocity(); curr_speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
        error = preds[0] - curr_speed; self.integral += error * 0.05
        derivative = (error - self.prev_error) / 0.05; self.prev_error = error
        pid = 1.0 * error + 0.05 * self.integral + 0.1 * derivative
        self.vehicle.apply_control(carla.VehicleControl(throttle=float(np.clip(pid, 0, 1)) if pid > 0 else 0.0, brake=float(np.clip(-pid, 0, 1)) if pid < 0 else 0.0, steer=float(np.clip(preds[1]/60.0, -1.0, 1.0))))

# ==========================================
# STEP 5: UTILS & GROUPING LOGIC
# ==========================================
def calculate_continuous_speeds(points, fps, correction_factor):
    speeds, step = [], 7
    if len(points) < step + 1: return [0] * len(points)
    time_interval = (step + 1) / fps
    for i in range(len(points) - step):
        dist = math.hypot(points[i+step][0] - points[i][0], points[i+step][1] - points[i][1])
        speeds.append((dist / time_interval) * correction_factor * 3.6)
    speeds.extend([speeds[-1]] * step)
    return speeds

def is_spawn_safe(loc, actors, margin):
    for a in actors:
        if a.get_location().distance(loc) < margin: return False
    return True

def cleanup_world(world):
    actors = world.get_actors()
    for s in actors.filter('sensor.*'): 
        if s.is_alive: s.destroy()
    for v in actors.filter('vehicle.*'): 
        if v.is_alive: v.destroy()
    for _ in range(5): world.tick()

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
    if not tids or len(tids) < min_size: return []
    sorted_tids = sorted(list(set(tids)))
    all_sequences = []
    current_sequence = [sorted_tids[0]]
    for i in range(1, len(sorted_tids)):
        if sorted_tids[i] == sorted_tids[i-1] + 1:
            current_sequence.append(sorted_tids[i])
        else:
            if len(current_sequence) >= min_size: all_sequences.append(current_sequence)
            current_sequence = [sorted_tids[i]]
    if len(current_sequence) >= min_size: all_sequences.append(current_sequence)

    final_groups = []
    for seq in all_sequences:
        if split_groups:
            num_splits = 3 if len(seq) > 14 else (2 if len(seq) > 7 else 1)
            if num_splits > 1:
                for g in np.array_split(seq, num_splits):
                    if len(g.tolist()) >= min_size: final_groups.append(g.tolist())
            else: final_groups.append(seq)
        else: final_groups.append(seq)
    return final_groups

class ScenarioRecorder:
    def __init__(self, scenario_id, mode, ego_name):
        self.scenario_id, self.mode, self.ego_name = scenario_id, mode, ego_name
        self.data = {"scenario_id": scenario_id, "mode": mode, "vehicles": {}}
        os.makedirs(REPLAY_DIR, exist_ok=True)
    def record_step(self, vehicles_dict):
        for name, actor in vehicles_dict.items():
            if not actor.is_alive: continue
            t, v = actor.get_transform(), actor.get_velocity()
            entry = {"x": round(t.location.x, 3), "y": round(t.location.y, 3), "yaw": round(t.rotation.yaw, 3), "speed": round(math.hypot(v.x, v.y), 3), "is_ego": (name == self.ego_name)}
            if name not in self.data["vehicles"]: self.data["vehicles"][name] = []
            self.data["vehicles"][name].append(entry)
    def save(self):
        with open(os.path.join(REPLAY_DIR, f"scenario_{self.scenario_id}_{self.mode}.json"), "w") as f: json.dump(self.data, f)

# ==========================================
# STEP 6: RUNNER
# ==========================================
def run_scenario(world, scenario_id, group_tids, parser, start_frame, mode, target_ego_tid):
    global EGO_LATEST_IMAGE
    EGO_LATEST_IMAGE = None
    cleanup_world(world)
    
    distance_travelled, ego_traj_history, ego_total_route_length, elapsed, collided = 0.0, [], 0.0, 0.0, False
    vehicles, actors_map, col_sensor, spectator = {}, {}, None, world.get_spectator()
    ego_tid = target_ego_tid
    recorder = ScenarioRecorder(scenario_id, mode, f"veh_{ego_tid}")

    try:
        for tid in group_tids:
            raw_pts = parser.get_sliced_trajectory_coords(tid, start_frame, int(SCENARIO_DURATION_SEC * REAL_WORLD_FPS))
            if len(raw_pts) < 8: continue
            carla_pts = transform_trajectory_with_analysis(raw_pts)
            speeds = calculate_continuous_speeds(carla_pts, REAL_WORLD_FPS, SPEED_CORRECTION_FACTOR)
            route = [(p[0], p[1], DEFAULT_Z, s) for p, s in zip(carla_pts, speeds)]
            if len(route) < 8: continue
            
            is_ego = (tid == ego_tid)
            if is_ego:
                for i in range(len(route)-1): ego_total_route_length += carla.Location(route[i][0],route[i][1],route[i][2]).distance(carla.Location(route[i+1][0],route[i+1][1],route[i+1][2]))
            
            # Stable Yaw: 7-frame gap logic
            p1, p2 = carla_pts[0], carla_pts[7]
            initial_yaw = math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))
            
            # Spawn Retraction Logic
            orig_spawn = carla.Location(route[0][0], route[0][1], route[0][2])
            safe_loc, spawned_successfully = orig_spawn, False
            if is_spawn_safe(orig_spawn, list(actors_map.values()), SPAWN_SAFETY_MARGIN):
                spawned_successfully = True
            else:
                yaw_rad = math.radians(initial_yaw)
                back_vec = carla.Vector3D(-math.cos(yaw_rad), -math.sin(yaw_rad), 0)
                for i in range(MAX_RETRACTION_ATTEMPTS):
                    test_loc = orig_spawn + back_vec * ((i+1) * RETRACTION_STEP)
                    if is_spawn_safe(test_loc, list(actors_map.values()), SPAWN_SAFETY_MARGIN):
                        safe_loc, spawned_successfully = test_loc, True
                        break
            
            if not spawned_successfully: continue

            start_tf = carla.Transform(safe_loc, carla.Rotation(yaw=initial_yaw))
            bp = world.get_blueprint_library().find(VEHICLE_BLUEPRINT)
            bp.set_attribute("color", EGO_COLOR.get(mode, "255,0,0") if is_ego else "128,128,128")
            
            try:
                sim_v = ReplayVehicle(world, bp, f"veh_{tid}", start_tf, is_ego=is_ego, scenario_id=scenario_id, mode=mode)
                sim_v.set_dynamic_route(route)
                vehicles[tid], actors_map[f"veh_{tid}"] = sim_v, sim_v.vehicle
            except: continue
        
        world.tick()
        if ego_tid not in vehicles: raise RuntimeError("Ego failed spawn.")
        ego_actor = vehicles[ego_tid].vehicle
        
        col_sensor = world.spawn_actor(world.get_blueprint_library().find('sensor.other.collision'), carla.Transform(), attach_to=ego_actor)
        def on_col(e): nonlocal collided; collided = True
        col_sensor.listen(on_col)

        ml_ctrl = EgoMLController(ego_actor, "lr_0.0001_bs_110_wspeed_0.5_frozen-20251021T053025Z-1-001/lr_0.0001_bs_110_wspeed_0.5_frozen/best_model.pth", "label_scaler.pkl", "nvidia/segformer-b1-finetuned-cityscapes-1024-1024") if mode == "ML" else None
        
        if mode == "AUTOPILOT":
            ego_actor.set_autopilot(True, TM.get_port())
            TM.ignore_lights_percentage(ego_actor, 100); TM.distance_to_leading_vehicle(ego_actor, 2.0)

        dt = 1.0 / SIM_FPS
        while elapsed < SCENARIO_DURATION_SEC and ego_actor.is_alive:
            if ego_tid in vehicles: vehicles[ego_tid].local_frame += 1
            world.tick()
            elapsed += dt
            spectator.set_transform(carla.Transform(ego_actor.get_location() + carla.Location(z=50), carla.Rotation(pitch=-90)))
            curr_loc = ego_actor.get_location()
            if ego_traj_history: distance_travelled += ego_traj_history[-1].distance(curr_loc)
            ego_traj_history.append(curr_loc)
            recorder.record_step(actors_map)
            
            for tid, v in vehicles.items():
                if tid == ego_tid:
                    if mode == "ML": ml_ctrl.tick()
                v.update(elapsed, list(vehicles.values()))

    except Exception as e:
        logging.error(f"Error: {e}"); collided = True
    finally:
        recorder.save()
        if mode == "AUTOPILOT" and ego_tid in vehicles:
            try: vehicles[ego_tid].vehicle.set_autopilot(False)
            except: pass
        
        metrics = {
            "scenario_id": scenario_id, "mode": mode, "collisions": 1 if collided else 0,
            "ttfc": round(elapsed, 2) if collided else -1,
            "completion": round(min(distance_travelled / (ego_total_route_length + 1e-6), 1.0), 2),
            "time_completion": round(elapsed / SCENARIO_DURATION_SEC, 2),
            "end_status": "COLLISION" if collided else "SUCCESS"
        }
        for v in vehicles.values(): v.destroy()
        if col_sensor and col_sensor.is_alive: col_sensor.stop(); col_sensor.destroy()
        for _ in range(3): world.tick()
        return metrics, ego_traj_history

# ==========================================
# STEP 7: MAIN
# ==========================================
def main():
    global TM
    logging.basicConfig(level=logging.INFO)
    if not os.path.exists(STATS_FILE):
        with open(STATS_FILE, "w") as f: f.write("ScenarioID,Mode,Collisions,TTFC,Completion,TimeCompletionPct,EndStatus\n")

    parser = AnnotationParser(INPUT_FILE)
    region_tids = {}
    content = open(TIDS_FILE, 'r').read() if os.path.exists(TIDS_FILE) else ""
    for m in re.findall(r'(\d+)\.\s*\{([\d\s,]+)\}', content): region_tids[int(m[0])] = sorted([int(i) for i in re.split(r'[\s,]+', m[1].strip()) if i])

    client = carla.Client(CARLA_HOST, CARLA_PORT); client.set_timeout(20.0)
    scenario_count = 0
    processed_track_ids = set()

    def process_group(group_to_run, interaction_frame):
        nonlocal scenario_count
        current_group_shuffled = list(group_to_run)
        random.shuffle(current_group_shuffled)
        target_ego_tid = current_group_shuffled[0]
        scenario_count += 1
        
        if scenario_count > 1 and scenario_count % SCENARIOS_PER_WORLD_RELOAD == 0:
            client.reload_world()
        
        world_obj = client.get_world()
        global TM
        TM = client.get_trafficmanager(8000); TM.set_synchronous_mode(True)
        settings = world_obj.get_settings()
        settings.synchronous_mode, settings.fixed_delta_seconds = True, 1.0 / SIM_FPS
        world_obj.apply_settings(settings)

        for mode in MODES:
            logging.info(f"--- Scenario {scenario_count} [{mode}] ---")
            metrics, _ = run_scenario(world_obj, scenario_count, current_group_shuffled, parser, interaction_frame, mode, target_ego_tid)
            if metrics:
                with open(STATS_FILE, "a") as f:
                    f.write(f"{metrics['scenario_id']},{metrics['mode']},{metrics['collisions']},{metrics['ttfc']},{metrics['completion']:.2f},{metrics['time_completion']:.2f},{metrics['end_status']}\n")

    try:
        for region_num, frame_range in FRAME_REGIONS.items():
            if scenario_count >= MAX_SCENARIOS: break
            # Only use IDs that haven't been used in a scenario yet
            eligible_ids = sorted([tid for tid in region_tids.get(region_num, []) if tid not in processed_track_ids])
            
            if region_num in (1, 2):
                # Regions 1 & 2: Use consecutive grouping
                candidate_groups = find_consecutive_groups(eligible_ids, MIN_ACTORS_FOR_SCENARIO, SPLIT_LARGE_GROUPS)
                for group in candidate_groups:
                    if scenario_count >= MAX_SCENARIOS: break
                    if any(tid in processed_track_ids for tid in group): continue
                    
                    scenario_vehicles, interaction_frame = find_best_vehicle_group(group, parser, frame_range)
                    if scenario_vehicles and interaction_frame:
                        processed_track_ids.update(group)
                        process_group(scenario_vehicles, interaction_frame)
            else:
                # Other Regions: Iterative distance-based grouping
                while eligible_ids:
                    if scenario_count >= MAX_SCENARIOS: break
                    anchor_tid = eligible_ids[0]
                    interaction_frame = frame_range[0]
                    all_positions = parser.get_positions_in_frame(interaction_frame)
                    
                    if anchor_tid not in all_positions: 
                        eligible_ids.pop(0)
                        processed_track_ids.add(anchor_tid)
                        continue
                    
                    anchor_pos = all_positions[anchor_tid]
                    new_group = [anchor_tid]
                    for other_tid in eligible_ids[1:]:
                        if other_tid in all_positions:
                            if math.hypot(anchor_pos[0]-all_positions[other_tid][0], anchor_pos[1]-all_positions[other_tid][1]) <= DISTANCE_THRESHOLD:
                                new_group.append(other_tid)
                    
                    processed_track_ids.update(new_group)
                    eligible_ids = [tid for tid in eligible_ids if tid not in new_group]
                    
                    if len(new_group) >= MIN_ACTORS_FOR_SCENARIO:
                        window = int(SCENARIO_DURATION_SEC * REAL_WORLD_FPS)
                        if interaction_frame > frame_range[1] - (window - 1): continue
                        process_group(new_group, interaction_frame)
    except KeyboardInterrupt: pass
    finally: logging.info("Done.")

if __name__ == "__main__":
    main()