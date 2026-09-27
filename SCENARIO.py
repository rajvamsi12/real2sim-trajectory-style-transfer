import os
import json
import math
import time
import carla
import random
import numpy as np
import logging
import re
from functools import reduce
from itertools import combinations

# Ensure these files exist in your directory
try:
    from timemethod import transform_trajectory_with_analysis
    from scc import SimVehicle, DummyWaypoint
except ImportError:
    print("ERROR: Could not import 'transform_trajectory_with_analysis' or 'SimVehicle'.")
    exit()

# --- Configuration ---
SCENARIO_DURATION_SEC = 15
SIM_FPS = 20
INPUT_FILE = "selected_annotations.json"
TIDS_FILE = "tids.txt"

CARLA_HOST = "localhost"
CARLA_PORT = 2000
DEFAULT_Z = 1.0  # Increased for safer spawning in Town10HD

REAL_WORLD_FPS = 4
SPEED_CORRECTION_FACTOR = 4
MIN_ACTORS_FOR_SCENARIO = 3
MIN_VEHICLE_SPEED_KMH = 0.5
SCENARIOS_PER_WORLD_RELOAD = 10

# --- Spawn Retraction Configuration ---
SPAWN_SAFETY_MARGIN = 3.5
MAX_RETRACTION_ATTEMPTS = 10
RETRACTION_STEP = 0.5

# --- Grouping Configuration ---
DISTANCE_THRESHOLD = 100.0 # Meters for distance-based grouping
SPLIT_LARGE_GROUPS = False

# --- Label to CARLA Blueprint Mapping ---
LABEL_TO_BLUEPRINT = {
    0: "vehicle.vespa.zx125",
    1: "vehicle.tesla.model3",
    2: "vehicle.kawasaki.ninja",
    3: "vehicle.bus.city",
    4: "walker.pedestrian.0001",
    5: "vehicle.carlamotors.carlacola"
}

# --- Frame Regions ---
FRAME_REGIONS = {
    2: (178, 327),
    3: (328, 475),
    4: (476, 590),
    5: (591, 779),
    6: (780, 910),
    7: (911, 944),
    1: (0, 177),
}

# -----------------------------
# AnnotationParser
# -----------------------------
class AnnotationParser:
    def __init__(self, filepath):
        self.track_data = {}
        self.frame_data = {}
        self._load_and_parse(filepath)

    def _load_and_parse(self, filepath):
        logging.info(f"Parsing annotations: {filepath}")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Annotation file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        for frame_idx, item in enumerate(data.get('items', [])):
            self.frame_data[frame_idx] = []
            for ann in item.get('annotations', []):
                attr = ann.get('attributes', {})
                tid = attr.get('track_id')
                label_id = ann.get('label_id', 1) 
                bbox = ann.get('bbox')
                
                if tid is not None and bbox:
                    self.frame_data[frame_idx].append({'track_id': tid, 'bbox': bbox})
                    if tid not in self.track_data:
                        self.track_data[tid] = {'trajectory': {}, 'label_id': label_id}
                    self.track_data[tid]['trajectory'][frame_idx] = bbox
        logging.info(f"Parsed {len(self.track_data)} tracks.")

    def get_label_for_track(self, tid):
        return self.track_data.get(tid, {}).get('label_id', 1)

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
# ScenarioActor (Simplified)
# -----------------------------
class ScenarioActor(SimVehicle):
    def __init__(self, world, bp, name, start, yaw):
        super().__init__(world, bp, name, start, yaw, start_delay=0)

    def update(self, simulation_time, sim_vehicles):
        return super().update(simulation_time, sim_vehicles)

# -----------------------------
# Helpers
# -----------------------------
def calculate_continuous_speeds(points, fps, correction_factor):
    speeds_kmh, step = [], 7
    if len(points) < step + 1: return [0] * len(points)
    time_interval = (step + 1) / fps
    for i in range(len(points) - step):
        p1, p2 = points[i], points[i + step]
        distance = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        raw_speed_mps = distance / time_interval if time_interval > 0 else 0
        speeds_kmh.append(raw_speed_mps * correction_factor * 3.6)
    speeds_kmh.extend([speeds_kmh[-1]] * step if speeds_kmh else [0] * step)
    return speeds_kmh

def is_spawn_location_safe(target_spawn_loc, existing_actors, margin):
    for actor in existing_actors:
        if actor.get_location().distance(target_spawn_loc) < margin:
            return False
    return True

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
    sorted_tids = sorted(list(set(tids))) 
    if not sorted_tids: return []
    all_sequences, current_sequence = [], [sorted_tids[0]]
    for i in range(1, len(sorted_tids)):
        if sorted_tids[i] == sorted_tids[i-1] + 1:
            current_sequence.append(sorted_tids[i])
        else:
            all_sequences.append(current_sequence)
            current_sequence = [sorted_tids[i]]
    all_sequences.append(current_sequence)
    
    final_groups = []
    for seq in all_sequences:
        if len(seq) < min_size: continue
        if split_groups and len(seq) > 7:
            for g in np.array_split(seq, 2 if len(seq) <= 14 else 3):
                if len(g.tolist()) >= min_size: final_groups.append(g.tolist())
        else:
            final_groups.append(seq)
    return final_groups

# -----------------------------
# Main Runner
# -----------------------------
def run_scenario(world, scenario_id, all_track_ids, annotation_parser, interaction_frame):
    actors = {}
    spawned_actors_list = []
    carla_map = world.get_map()
    
    try:
        dt = world.get_settings().fixed_delta_seconds
        bp_library = world.get_blueprint_library()
        num_frames_required = int(SCENARIO_DURATION_SEC * REAL_WORLD_FPS)

        for tid in all_track_ids:
            try:
                # 1. Get Trajectory
                real_pts = annotation_parser.get_sliced_trajectory_coords(tid, interaction_frame, num_frames_required)
                if len(real_pts) < 10: continue
                carla_pts = transform_trajectory_with_analysis(real_pts)
                if not carla_pts: continue

                # 2. Setup Base Route & Transform
                label = annotation_parser.get_label_for_track(tid)
                bp = bp_library.find(LABEL_TO_BLUEPRINT.get(label, "vehicle.tesla.model3"))
                speeds = calculate_continuous_speeds(carla_pts, REAL_WORLD_FPS, SPEED_CORRECTION_FACTOR)
                dynamic_route = [(pt[0], pt[1], DEFAULT_Z, spd) for pt, spd in zip(carla_pts, speeds)]
                
                # 3. Initial Yaw calculation from trajectory
                p1, p2 = carla_pts[0], carla_pts[7]
                initial_yaw = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
                
                # 4. Project to Road for better spawning AND update Yaw to match Lane
                orig_spawn_loc = carla.Location(x=dynamic_route[0][0], y=dynamic_route[0][1], z=DEFAULT_Z)
                wp = carla_map.get_waypoint(orig_spawn_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                
                if wp:
                    orig_spawn_loc = wp.transform.location
                    orig_spawn_loc.z += 0.1 # Slight offset to avoid clipping road
                    
                    # UPDATE: Align initial_yaw with lane direction to prevent inconsistent poses
                    # We blend them slightly to preserve the original motion intent
                    lane_yaw = wp.transform.rotation.yaw
                    # initial_yaw = (0.7 * initial_yaw) + (0.3 * lane_yaw)
                    initial_yaw = lane_yaw
                # 5. Spawn Retraction Logic
                safe_loc, is_safe = orig_spawn_loc, is_spawn_location_safe(orig_spawn_loc, spawned_actors_list, SPAWN_SAFETY_MARGIN)
                if not is_safe:
                    yaw_rad = math.radians(initial_yaw)
                    backwards = carla.Vector3D(x=-math.cos(yaw_rad), y=-math.sin(yaw_rad))
                    for i in range(MAX_RETRACTION_ATTEMPTS):
                        new_loc = orig_spawn_loc + backwards * ((i + 1) * RETRACTION_STEP)
                        if is_spawn_location_safe(new_loc, spawned_actors_list, SPAWN_SAFETY_MARGIN):
                            safe_loc, is_safe = new_loc, True
                            break
                
                if not is_safe:
                    logging.warning(f"[Scenario {scenario_id}] No safe spawn for actor_{tid}. Skipping.")
                    continue

                # 6. Spawning with robust try/except
                vname = f"actor_{tid}_L{label}"
                actor = ScenarioActor(world, bp, vname, (safe_loc.x, safe_loc.y, safe_loc.z), initial_yaw)
                actor.set_dynamic_route(dynamic_route)
                actors[vname] = actor
                spawned_actors_list.append(actor.vehicle)
                
                # Immediate tick to register spawn
                world.tick()
                
            except (RuntimeError, Exception) as e:
                logging.error(f"[Scenario {scenario_id}] FAILED TO SPAWN actor_{tid}: {e}")
                continue

        if not actors: return "SKIPPED"

        logging.info(f"Scenario {scenario_id}: Playing with {len(actors)} actors.")
        spectator = world.get_spectator()
        avg_x = np.mean([a.vehicle.get_location().x for a in actors.values()])
        avg_y = np.mean([a.vehicle.get_location().y for a in actors.values()])
        spectator.set_transform(carla.Transform(carla.Location(x=avg_x, y=avg_y, z=40), carla.Rotation(pitch=-90)))

        sim_clock = 0.0
        while sim_clock < SCENARIO_DURATION_SEC:
            world.tick()
            sim_clock += dt
            for a in actors.values(): a.update(sim_clock, actors)
        return "SUCCESS"

    finally:
        for a in actors.values():
            try: a.destroy()
            except: pass

def parse_tids_file(filepath):
    region_tids = {}
    if not os.path.exists(filepath): return region_tids
    with open(filepath, 'r') as f: content = f.read()
    for match in re.findall(r'(\d+)\.\s*\{([\d\s,]+)\}', content):
        region_tids[int(match[0])] = sorted([int(i) for i in re.split(r'[\s,]+', match[1].strip()) if i])
    return region_tids

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    parser = AnnotationParser(INPUT_FILE)
    region_tids = parse_tids_file(TIDS_FILE)
    processed_track_ids, scenario_count = set(), 0

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(10.0)
    world = client.get_world()
    
    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds = True, 1.0 / SIM_FPS
    world.apply_settings(settings)

    try:
        for region_num, frame_range in sorted(FRAME_REGIONS.items(), reverse=False):
            logging.info(f"\nProcessing Region {region_num}")
            eligible_ids = sorted([tid for tid in region_tids.get(region_num, []) if tid not in processed_track_ids])
            
            if region_num in (1, 2):
                groups = find_consecutive_groups(eligible_ids, MIN_ACTORS_FOR_SCENARIO, SPLIT_LARGE_GROUPS)
                for group in groups:
                    if any(tid in processed_track_ids for tid in group): continue
                    v_group, frame = find_best_vehicle_group(group, parser, frame_range)
                    if v_group and frame:
                        processed_track_ids.update(group)
                        scenario_count += 1
                        if scenario_count % SCENARIOS_PER_WORLD_RELOAD == 0:
                            client.reload_world()
                            world = client.get_world(); world.apply_settings(settings)
                        run_scenario(world, scenario_count, v_group, parser, frame)
            else:
                while eligible_ids:
                    anchor_tid = eligible_ids.pop(0)
                    interaction_frame = frame_range[0]
                    all_pos = parser.get_track_positions_at_frame(interaction_frame)
                    if anchor_tid not in all_pos: continue
                    
                    new_group = [anchor_tid]
                    anchor_pos = all_pos[anchor_tid]
                    for other_tid in eligible_ids[:]:
                        if other_tid in all_pos:
                            dist = math.hypot(anchor_pos[0] - all_pos[other_tid][0], anchor_pos[1] - all_pos[other_tid][1])
                            if dist <= DISTANCE_THRESHOLD:
                                new_group.append(other_tid); eligible_ids.remove(other_tid)
                    
                    if len(new_group) >= MIN_ACTORS_FOR_SCENARIO:
                        processed_track_ids.update(new_group)
                        scenario_count += 1
                        if scenario_count % SCENARIOS_PER_WORLD_RELOAD == 0:
                            client.reload_world()
                            world = client.get_world(); world.apply_settings(settings)
                        run_scenario(world, scenario_count, new_group, parser, interaction_frame)

    finally:
        try:
            settings.synchronous_mode = False
            world.apply_settings(settings)
        except: pass
        logging.info("Done.")

if __name__ == "__main__":
    main()