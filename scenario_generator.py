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

from timemethod import transform_trajectory_with_analysis
from scc import SimVehicle


# --- Configuration ---
TOTAL_IMAGES_GOAL = 1_000_000
SCENARIO_DURATION_SEC = 15.0  # Scenario duration in seconds
SIM_FPS = 10  # Simulation frames per second (image capture rate)
IMAGES_PER_VEHICLE_PER_SCENARIO = int(SCENARIO_DURATION_SEC * SIM_FPS)
IMAGE_CAPTURE_TOLERANCE = 0.98  # Threshold for minimum images to accept scenario

INPUT_FILE = "selected_annotations.json"
TIDS_FILE = "tids.txt" # <<< NEW: File with track IDs for each region
OUTPUT_DIR = "_scenario_output"

CARLA_HOST = "localhost"
CARLA_PORT = 2000
TOWN = "Town10HD"

VEHICLE_BLUEPRINT = "vehicle.bmw.grandtourer"
DEFAULT_Z = 0.6  # Default spawn height for vehicles

REAL_WORLD_FPS = 4  # FPS of original annotated data
SPEED_CORRECTION_FACTOR = 4

MIN_PROXIMITY_M = 1.0
MAX_PROXIMITY_M = 15.0 
MIN_ACTORS_FOR_SCENARIO = 3  
MIN_TEMPORAL_OVERLAP_FRAMES = int(SCENARIO_DURATION_SEC * REAL_WORLD_FPS) + 10  
LOOKBACK_FRAMES = 10  

# <<< NEW: Define the structured frame regions and their corresponding track IDs >>>
FRAME_REGIONS = {
    1: (0, 177),
    2: (178, 327),
    3: (328, 475),
    4: (476, 590),
    5: (591, 779),
    6: (780, 910),
    7: (911, 944),
}

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
                track_id = ann.get('attributes', {}).get('track_id')
                bbox = tuple(ann.get('bbox'))
                if track_id is not None and bbox and len(bbox) == 4:
                    vehicle_info = {'track_id': track_id, 'bbox': bbox}
                    self.frame_data[frame_num].append(vehicle_info)
                    if track_id not in self.track_data:
                        self.track_data[track_id] = {'trajectory': {}}
                    self.track_data[track_id]['trajectory'][frame_num] = bbox
        logging.info("Annotation parsing complete.")
        logging.info(f"Found {len(self.track_data)} unique tracks across {len(self.frame_data)} frames.")

    def get_all_track_ids(self):
        return list(self.track_data.keys())

    def get_frames_for_track(self, track_id):
        if track_id in self.track_data:
            return sorted(list(self.track_data[track_id]['trajectory'].keys()))
        return []

    def get_bbox_at_frame(self, track_id, frame_num):
        return self.track_data.get(track_id, {}).get('trajectory', {}).get(frame_num)

    def get_vehicles_in_frame(self, frame_num):
        return self.frame_data.get(frame_num, [])

    def get_sliced_trajectory_coords(self, track_id, start_frame, num_frames):
        coords = []
        if track_id in self.track_data:
            sorted_frames = sorted(self.track_data[track_id]['trajectory'].keys())
            end_frame = start_frame + num_frames
            slice_frames = [frame for frame in sorted_frames if start_frame <= frame < end_frame]
            for frame in slice_frames:
                bbox = self.track_data[track_id]['trajectory'][frame]
                x, y, w, h = bbox
                mid_x, mid_y = x + w / 2, y + h / 2
                coords.append((mid_x, -mid_y))
        return coords


class CapturingVehicle(SimVehicle):
    def __init__(self, world, bp, name, start, yaw, output_path, start_delay=0, start_capture_time=0, end_capture_time=float('inf')):
        super().__init__(world, bp, name, start, yaw, start_delay)
        self.output_path = os.path.join(output_path, self.name)
        self.image_path = os.path.join(self.output_path, 'images')
        os.makedirs(self.image_path, exist_ok=True)
        self.camera = None
        self.collision_sensor = None
        self.angular_velocities = []
        self.linear_velocities = []
        self.image_count = 0
        self.has_collided = False
        self.start_capture_time = start_capture_time
        self.end_capture_time = end_capture_time
        self.is_capturing = False
        self._setup_sensors()

    def _setup_sensors(self):
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(lambda image: CapturingVehicle._parse_image(weakref.ref(self), image))
        collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: CapturingVehicle._on_collision(weakref.ref(self), event))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self or not self.camera or not self.camera.is_alive: return
        if not self.is_capturing: return
        self.image_count += 1
        image.save_to_disk(os.path.join(self.image_path, f'{image.frame:06d}.png'))

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self or not self.collision_sensor or not self.collision_sensor.is_alive: return
        if not self.has_collided:
            logging.warning(f"!!! COLLISION DETECTED for vehicle {self.name} with {event.other_actor.type_id}")
            self.has_collided = True

    def get_image_count(self):
        return self.image_count

    def update(self, simulation_time, sim_vehicles):
        self.is_capturing = self.start_capture_time <= simulation_time <= self.end_capture_time
        if self.vehicle and self.vehicle.is_alive and self.is_capturing:
            ang_vel = self.vehicle.get_angular_velocity()
            self.angular_velocities.append({'timestamp': simulation_time, 'x': ang_vel.x, 'y': ang_vel.y, 'z': ang_vel.z})
            lin_vel = self.vehicle.get_velocity()
            speed_ms = lin_vel.length()
            self.linear_velocities.append({'timestamp': simulation_time, 'x': lin_vel.x, 'y': lin_vel.y, 'z': lin_vel.z, 'speed_ms': speed_ms})
        return super().update(simulation_time, sim_vehicles)

    def save_captured_data(self):
        if self.linear_velocities:
            with open(os.path.join(self.output_path, 'linear_velocity.json'), 'w') as f:
                json.dump(self.linear_velocities, f, indent=4)
        if self.angular_velocities:
            with open(os.path.join(self.output_path, 'angular_velocity.json'), 'w') as f:
                json.dump(self.angular_velocities, f, indent=4)

    def destroy(self):
        if self.camera and self.camera.is_listening:
            self.camera.stop()
        if self.collision_sensor and self.collision_sensor.is_listening:
            self.collision_sensor.stop()
        time.sleep(0.05)
        if self.camera and self.camera.is_alive:
            self.camera.destroy()
        if self.collision_sensor and self.collision_sensor.is_alive:
            self.collision_sensor.destroy()
        super().destroy()


def calculate_continuous_speeds(points, fps, correction_factor):
    speeds_kmh = []
    step = 7
    if len(points) < step + 1:
        return [0] * len(points)
    time_interval = (step + 1) / fps
    for i in range(len(points) - step):
        p1, p2 = points[i], points[i + step]
        distance = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        raw_speed_mps = distance / time_interval if time_interval > 0 else 0
        corrected_speed_mps = raw_speed_mps * correction_factor
        speeds_kmh.append(corrected_speed_mps * 3.6)
    speeds_kmh.extend([speeds_kmh[-1]] * step if speeds_kmh else [0] * step)
    return speeds_kmh


def run_scenario(scenario_id, anchor_track_id, actor_track_ids, annotation_parser, random_frame):
    vehicles = {}
    world = None
    original_settings = None
    client = None
    scenario_output_path = os.path.join(OUTPUT_DIR, f"scenario_{scenario_id:04d}")
    scenario_had_collision = False
    images_generated_this_run = 0

    try:
        client = carla.Client(CARLA_HOST, CARLA_PORT)
        client.set_timeout(60.0)

        logging.info(f"[SCENARIO {scenario_id}] Reloading world...")
        client.reload_world()
        world = client.get_world()

        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / SIM_FPS
        world.apply_settings(settings)
        dt = settings.fixed_delta_seconds

        bp_library = world.get_blueprint_library()
        vehicle_bp = bp_library.find(VEHICLE_BLUEPRINT)
        all_track_ids = [anchor_track_id] + actor_track_ids

        warmup_duration_sec = LOOKBACK_FRAMES / REAL_WORLD_FPS
        start_capture_time = warmup_duration_sec
        end_capture_time = warmup_duration_sec + SCENARIO_DURATION_SEC

        sim_start_frame = max(0, random_frame - LOOKBACK_FRAMES)
        num_frames_required = LOOKBACK_FRAMES + int(SCENARIO_DURATION_SEC * REAL_WORLD_FPS)

        logging.info(f"[SCENARIO {scenario_id}] Interaction frame {random_frame}, sim from frame {sim_start_frame}")
        logging.info(f"[SCENARIO {scenario_id}] Warm-up: {warmup_duration_sec:.2f}s, capturing from {start_capture_time:.2f}s to {end_capture_time:.2f}s")

        for tid in all_track_ids:
            try:
                real_pts = annotation_parser.get_sliced_trajectory_coords(tid, sim_start_frame, num_frames_required)
                if len(real_pts) < 8:
                    logging.warning(f"[SCENARIO {scenario_id}] Skipping track {tid}: insufficient raw points in slice ({len(real_pts)})")
                    continue
            
                carla_pts = transform_trajectory_with_analysis(real_pts)
                
                if len(carla_pts) < 8:
                    logging.warning(f"[SCENARIO {scenario_id}] Skipping track {tid}: not enough points after transformation ({len(carla_pts)}).")
                    continue

                speeds = calculate_continuous_speeds(carla_pts, REAL_WORLD_FPS, SPEED_CORRECTION_FACTOR)
                dynamic_route = [(pt[0], pt[1], DEFAULT_Z, speed) for pt, speed in zip(carla_pts, speeds)]

                p1, p2 = carla_pts[0], carla_pts[7]
                initial_yaw = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
                
                spawn_point = dynamic_route[0][:3]
                vname = f"vehicle_{tid}"

                logging.info(f"--> Attempting to spawn vehicle {tid}...")
                logging.info(f"    Initial REAL-WORLD pixel coords: {real_pts[0]}")
                logging.info(f"    Transformed CARLA spawn point: {spawn_point}")

                sim_v = CapturingVehicle(
                    world, vehicle_bp, vname, spawn_point, initial_yaw,
                    output_path=scenario_output_path,
                    start_capture_time=start_capture_time,
                    end_capture_time=end_capture_time
                )
                sim_v.set_dynamic_route(dynamic_route)
                vehicles[vname] = sim_v
                world.tick()
            
            except RuntimeError as e:
                logging.error(f"[SCENARIO {scenario_id}] FAILED TO SPAWN vehicle {tid} at CARLA coords {spawn_point}. Error: {e}")
                logging.error(f"    This corresponds to initial REAL-WORLD pixel coords: {real_pts[0] if 'real_pts' in locals() else 'N/A'}")
                continue
            except IndexError as e:
                logging.warning(f"[SCENARIO {scenario_id}] Skipping vehicle {tid} due to data processing error (IndexError): {e}")
                continue
            except Exception as e:
                logging.error(f"[SCENARIO {scenario_id}] An unexpected error occurred while processing vehicle {tid}: {e}")
                continue

        if len(vehicles) < MIN_ACTORS_FOR_SCENARIO:
            logging.warning(f"[SCENARIO {scenario_id}] ABORTED: only {len(vehicles)} vehicles spawned, need at least {MIN_ACTORS_FOR_SCENARIO}")
            return 0, False

        if len(vehicles) > 0:
            avg_x = sum(v.vehicle.get_location().x for v in vehicles.values()) / len(vehicles)
            avg_y = sum(v.vehicle.get_location().y for v in vehicles.values()) / len(vehicles)
            spectator = world.get_spectator()
            spectator.set_transform(carla.Transform(carla.Location(x=avg_x, y=avg_y, z=50), carla.Rotation(pitch=-90)))
            logging.info(f"[SCENARIO {scenario_id}] Spawned {len(vehicles)} vehicles. Starting simulation...")

            sim_clock = 0.0
            max_sim_time = end_capture_time + 10.0
            min_images_required = int(IMAGES_PER_VEHICLE_PER_SCENARIO * IMAGE_CAPTURE_TOLERANCE)

            while sim_clock < max_sim_time:
                if all(v.get_image_count() >= min_images_required for v in vehicles.values()):
                    logging.info(f"[SCENARIO {scenario_id}] All vehicles captured at least {min_images_required} images. Finishing.")
                    break
                if any(v.has_collided for v in vehicles.values()):
                    scenario_had_collision = True
                    logging.warning(f"[SCENARIO {scenario_id}] Collision detected. Ending scenario early.")
                    break
                world.tick()
                sim_clock += dt
                for v in vehicles.values():
                    v.update(sim_clock, vehicles)
            else:
                logging.warning(f"[SCENARIO {scenario_id}] Simulation timed out at {max_sim_time:.1f}s.")

        if not scenario_had_collision and any(v.has_collided for v in vehicles.values()):
            scenario_had_collision = True

        if scenario_had_collision:
            logging.warning(f"[SCENARIO {scenario_id}] Discarded due to collision. No data saved.")
            if os.path.exists(scenario_output_path):
                shutil.rmtree(scenario_output_path)
            return 0, False
        else:
            for v in vehicles.values():
                v.save_captured_data()
            images_generated_this_run = len(vehicles) * IMAGES_PER_VEHICLE_PER_SCENARIO
            logging.info(f"[SCENARIO {scenario_id}] Captured ~{images_generated_this_run} images.")
            return images_generated_this_run, True

    except Exception as e:
        logging.error(f"Error during scenario {scenario_id}: {e}")
        return 0, False

    finally:
        logging.info(f"[SCENARIO {scenario_id}] Cleaning up...")
        if vehicles:
            logging.info(f"Destroying {len(vehicles)} vehicles and sensors...")
            for v in vehicles.values():
                v.destroy()
            time.sleep(0.2)
        if client and original_settings is not None:
            try:
                world.apply_settings(original_settings)
                logging.info(f"[SCENARIO {scenario_id}] World settings restored.")
            except Exception as e:
                logging.error(f"Failed to restore world settings: {e}")

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def parse_tids_file(filepath):
    """ Parses the tids.txt file to get a dictionary of region -> track_ids. """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Track ID file not found at {filepath}")
    
    region_tids = {}
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Use regex to find all regions and their corresponding ID sets
    matches = re.findall(r'(\d+)\.\s*\{([\d\s,]+)\}', content)
    
    for match in matches:
        region_num = int(match[0])
        # Extract numbers from the matched string
        ids_str = match[1].strip()
        # Split by comma or space and filter out empty strings
        ids = [int(i) for i in re.split(r'[\s,]+', ids_str) if i]
        region_tids[region_num] = ids
        
    logging.info(f"Loaded track IDs for {len(region_tids)} regions from {filepath}.")
    return region_tids

# <<< MODIFIED: Function now searches within a specific frame range and eligible IDs >>>
def find_proximate_actors(anchor_id, parser, processed_track_ids, frame_range, eligible_ids):
    all_frames_for_anchor = parser.get_frames_for_track(anchor_id)
    if not all_frames_for_anchor:
        return None, None

    # Find frames for the anchor that are within the specified region
    valid_frame_range_for_anchor = [
        f for f in all_frames_for_anchor 
        if frame_range[0] <= f <= frame_range[1] and f >= LOOKBACK_FRAMES
    ]
    if not valid_frame_range_for_anchor:
        return None, None

    # Choose a random frame *from the valid range* to test for a scenario
    random_frame = random.choice(valid_frame_range_for_anchor)
    anchor_bbox = parser.get_bbox_at_frame(anchor_id, random_frame)
    if not anchor_bbox:
        return None, None

    anchor_pos_2d = (anchor_bbox[0] + anchor_bbox[2] / 2, anchor_bbox[1] + anchor_bbox[3] / 2)
    vehicles_in_frame = parser.get_vehicles_in_frame(random_frame)
    proximate_actors = []

    for other_vehicle in vehicles_in_frame:
        other_id = other_vehicle['track_id']
        # Check if the other vehicle is eligible for this region and not already used
        if other_id == anchor_id or other_id in processed_track_ids or other_id not in eligible_ids:
            continue

        actor_frames = parser.get_frames_for_track(other_id)
        try:
            current_frame_index = actor_frames.index(random_frame)
            if len(actor_frames) - current_frame_index < MIN_TEMPORAL_OVERLAP_FRAMES:
                continue
        except ValueError:
            continue

        actor_bbox = other_vehicle['bbox']
        actor_pos_2d = (actor_bbox[0] + actor_bbox[2] / 2, actor_bbox[1] + actor_bbox[3] / 2)
        distance = math.hypot(anchor_pos_2d[0] - actor_pos_2d[0], anchor_pos_2d[1] - actor_pos_2d[1])

        if MIN_PROXIMITY_M <= distance <= MAX_PROXIMITY_M:
            proximate_actors.append(other_id)

    if len(proximate_actors) >= 2:
        return proximate_actors, random_frame

    return None, None


def main():
    setup_logging()
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logging.info(f"Created output directory: {OUTPUT_DIR}")

    parser = AnnotationParser(INPUT_FILE)
    # <<< NEW: Load track IDs for each region from tids.txt >>>
    region_tids = parse_tids_file(TIDS_FILE)

    processed_track_ids = set()
    total_images_captured = 0
    scenario_count = 0
    valid_scenarios_summary = []

    logging.info(f"Starting structured scenario search. Goal: {TOTAL_IMAGES_GOAL:,} images.")

    # <<< MODIFIED: Loop through each region sequentially >>>
    for region_num, frame_range in sorted(FRAME_REGIONS.items()):
        logging.info(f"======================================================")
        logging.info(f"Processing Region {region_num} (Frames {frame_range[0]}-{frame_range[1]})")
        logging.info(f"======================================================")

        if total_images_captured >= TOTAL_IMAGES_GOAL:
            logging.info(f"Image goal reached. Halting processing.")
            break

        eligible_track_ids = region_tids.get(region_num, [])
        if not eligible_track_ids:
            logging.warning(f"No track IDs specified for Region {region_num}. Skipping.")
            continue
        
        # Iterate through potential anchors from the eligible list for this region
        for anchor_id in eligible_track_ids:
            if total_images_captured >= TOTAL_IMAGES_GOAL:
                break
            if anchor_id in processed_track_ids:
                continue

            # Find actors within the specified frame range and from the eligible list
            proximate_actors, random_frame = find_proximate_actors(
                anchor_id, parser, processed_track_ids, frame_range, set(eligible_track_ids)
            )

            if proximate_actors:
                scenario_count += 1
                logging.info(f"--- Found Scenario {scenario_count} in Region {region_num} ---")
                logging.info(f"  Anchor Vehicle: {anchor_id}")
                logging.info(f"  Actors       : {proximate_actors} (found at frame {random_frame})")

                images_in_run, is_valid = run_scenario(scenario_count, anchor_id, proximate_actors, parser, random_frame)

                if is_valid and images_in_run > 0:
                    total_images_captured += images_in_run
                    valid_scenarios_summary.append({
                        'scenario_id': scenario_count, 'anchor_id': anchor_id,
                        'actor_ids': proximate_actors, 'found_at_frame': random_frame,
                        'region': region_num
                    })
                    logging.info(f"--- Scenario {scenario_count} Complete ---")
                    logging.info(f"Total images so far: {total_images_captured:,} / {TOTAL_IMAGES_GOAL:,}")

                # Mark all used vehicles as processed so they aren't used again
                processed_track_ids.add(anchor_id)
                for actor_id in proximate_actors:
                    processed_track_ids.add(actor_id)

    logging.info("======================================================")
    logging.info("All regions processed or image goal reached.")
    logging.info("--- Summary of Valid Scenarios ---")

    if not valid_scenarios_summary:
        logging.info("No valid scenarios were successfully captured.")
    else:
        for summary in valid_scenarios_summary:
            logging.info(f"  - Region {summary['region']}, Scenario {summary['scenario_id']}: Anchor={summary['anchor_id']}, Actors={summary['actor_ids']}")

    logging.info(f"Total scenarios generated: {len(valid_scenarios_summary)}")
    logging.info(f"Total unique vehicles used: {len(processed_track_ids)}")
    logging.info("Script finished.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"A critical error occurred in the main loop: {e}", exc_info=True)
    finally:
        logging.info("Script finished.")
