# ========================= carla_body_server.py (fixed) =========================
import carla
import random
import numpy as np
from flask import Flask, request, jsonify, send_file
from PIL import Image
import io
import threading
import time
import csv
from datetime import datetime
import math

app = Flask(__name__)

# --- Global variables (protected with state_lock) ---
state_lock = threading.Lock()
LATEST_IMAGE = None  # {'raw': bytes, 'width': int, 'height': int}
vehicle = None
client = None
actor_list = []  # sensors and vehicle (actors that should be destroyed)
HAS_COLLIDED = False
LANE_DEPARTURES = 0
METRICS_FILE = "carla_metrics.csv"  # CSV file to save metrics


# --- Sensor callbacks ---
def on_image(image):
    """Store raw bytes and dimensions instead of the CARLA Image object."""
    global LATEST_IMAGE
    try:
        raw_bytes = bytes(image.raw_data)
        w, h = image.width, image.height
        with state_lock:
            LATEST_IMAGE = {"raw": raw_bytes, "width": w, "height": h}
    except Exception as e:
        print("[ERROR] on_image exception:", e)


def on_collision(event):
    global HAS_COLLIDED
    try:
        other = event.other_actor
        other_id = other.id if other is not None else "unknown"
        msg = f"!!! COLLISION with {getattr(other, 'type_id', other_id)} (id={other_id}) !!!"
        with state_lock:
            if not HAS_COLLIDED:
                HAS_COLLIDED = True
                print("[DEBUG]", msg)
            else:
                print("[DEBUG] collision event but HAS_COLLIDED already True")
    except Exception as e:
        print("[ERROR] on_collision exception:", e)


def on_lane_invasion(event):
    global LANE_DEPARTURES
    try:
        lane_types = {str(lt.type) for lt in event.crossed_lane_markings}
        with state_lock:
            LANE_DEPARTURES += 1
            count = LANE_DEPARTURES
        if 'Solid' in lane_types or 'LaneMarkingType.Solid' in lane_types:
            print(f"[DEBUG] --- Solid Lane Departure #{count} ---")
        else:
            print(f"[DEBUG] --- Lane Invasion (non-solid) #{count} ---")
    except Exception as e:
        print("[ERROR] on_lane_invasion exception:", e)


# --- Save metrics to CSV ---
def save_metrics():
    with state_lock:
        collided = HAS_COLLIDED
        lane_dep = LANE_DEPARTURES
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = ["timestamp", "collided", "lane_departures"]
    row = [timestamp, collided, lane_dep]

    try:
        with open(METRICS_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(header)
            writer.writerow(row)
        print(f"[DEBUG] Metrics saved: {row}")
    except Exception as e:
        print("[ERROR] Error saving metrics:", e)


# --- Flask endpoints ---
@app.route("/latest_image")
def latest_image():
    global LATEST_IMAGE
    with state_lock:
        img_info = None if LATEST_IMAGE is None else dict(LATEST_IMAGE)
    if img_info is None:
        print("[DEBUG] /latest_image: No image yet")
        return "No image yet", 404

    try:
        raw = img_info["raw"]
        w, h = img_info["width"], img_info["height"]
        img_array = np.frombuffer(raw, dtype=np.uint8)
        expected_len = w * h * 4
        if img_array.size != expected_len:
            print(f"[WARN] /latest_image: expected {expected_len} bytes but got {img_array.size}")
            img_array = np.resize(img_array, expected_len)
        img_array = img_array.reshape((h, w, 4))
        rgb = img_array[:, :, :3][:, :, ::-1]  # BGRA → RGB
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        # print("[DEBUG] /latest_image: returning JPEG")
        return send_file(buf, mimetype="image/jpeg")
    except Exception as e:
        print("[ERROR] /latest_image exception:", e)
        return "Internal server error", 500


@app.route("/apply_control", methods=["POST"])
def apply_control():
    global vehicle
    data = request.json or {}
    with state_lock:
        local_vehicle = vehicle
    if local_vehicle is None:
        print("[DEBUG] /apply_control: Vehicle not ready")
        return jsonify({"error": "Vehicle not ready"}), 400

    try:
        throttle = float(data.get("throttle", 0.0))
        steer = float(data.get("steer", 0.0))
        brake = float(data.get("brake", 0.0))
        local_vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
        print(f"[DEBUG] /apply_control: throttle={throttle:.3f} steer={steer:.3f} brake={brake:.3f}")
        return jsonify({"status": "applied"})
    except Exception as e:
        print("[ERROR] /apply_control exception:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/status")
def status():
    with state_lock:
        local_client = client
        local_vehicle = vehicle
        collided = HAS_COLLIDED
        lane_dep = LANE_DEPARTURES
    speed_kmh = None
    try:
        if local_vehicle is not None:
            vel = local_vehicle.get_velocity()
            speed_m_s = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            speed_kmh = speed_m_s * 3.6
            # print(f"[DEBUG] /status: speed_m_s={speed_m_s:.3f} m/s speed_kmh={speed_kmh:.3f}")
        else:
            print("[DEBUG] /status: vehicle is None")
    except Exception as e:
        print("[ERROR] /status velocity exception:", e)

    return jsonify({
        "connected": local_client is not None,
        "has_collided": collided,
        "lane_departures": lane_dep,
        "current_speed_kmh": speed_kmh
    })


# --- Spectator follow thread ---
def spectator_follow_loop(sleep_time=0.05, distance_behind=6.0, height=3.0):
    """Keep spectator behind the vehicle for better camera view."""
    print("[DEBUG] spectator_follow_loop started")
    while True:
        with state_lock:
            local_vehicle = vehicle
            local_client = client
        if local_vehicle is None or local_client is None:
            time.sleep(0.2)
            continue

        try:
            world = local_client.get_world()
            spectator = world.get_spectator()
            v_transform = local_vehicle.get_transform()
            forward = v_transform.get_forward_vector()

            spec_loc = carla.Location(
                x=v_transform.location.x - forward.x * distance_behind,
                y=v_transform.location.y - forward.y * distance_behind,
                z=v_transform.location.z + height
            )
            yaw = v_transform.rotation.yaw
            spec_rot = carla.Rotation(pitch=-10.0, yaw=yaw, roll=0.0)
            spectator.set_transform(carla.Transform(spec_loc, spec_rot))
        except Exception as e:
            # print("[ERROR] spectator_follow_loop exception:", e) # Can be noisy
            pass

        time.sleep(sleep_time)


# --- CARLA setup ---
def setup_carla():
    global client, vehicle, actor_list
    print("[INFO] Connecting to CARLA on port 2000...")
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    # Spawn vehicle
    vehicle_bp = bp_lib.find('vehicle.tesla.model3')
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points found in the map")
    spawn_point = random.choice(spawn_points)
    vehicle_actor = world.spawn_actor(vehicle_bp, spawn_point)
    with state_lock:
        vehicle = vehicle_actor
        actor_list.append(vehicle_actor)
    print(f"[INFO] Spawned vehicle at {spawn_point.location} (id={vehicle_actor.id})")

    # --- NEW SECTION: Spawn AI Traffic ---
    print("[INFO] Spawning AI traffic...")
    try:
        traffic_manager = client.get_trafficmanager(8000) # Default TM port
        traffic_manager.set_global_distance_to_leading_vehicle(5.0)
        blueprints = bp_lib.filter('vehicle.*')


        NUM_VEHICLES = 50

        # Get available spawn points, excluding the player's
        available_spawn_points = [sp for sp in spawn_points if sp.location != spawn_point.location]
        random.shuffle(available_spawn_points)

        spawned_ai_count = 0
        for n, transform in enumerate(available_spawn_points):
            if spawned_ai_count >= NUM_VEHICLES:
                break # We've spawned enough

            bp = random.choice(blueprints)

            # Filter out problematic vehicles
            if 'bicycle' in bp.id or 'motorcycle' in bp.id or 'carlacola' in bp.id:
                continue
            
            try:
                ai_vehicle = world.spawn_actor(bp, transform)
                ai_vehicle.set_autopilot(True, traffic_manager.get_port())
                traffic_manager.vehicle_percentage_speed_difference(ai_vehicle, 20.0) # 20% slower than speed limit
                traffic_manager.auto_lane_change(ai_vehicle, True)
                
                with state_lock:
                    actor_list.append(ai_vehicle) # Add to cleanup list
                spawned_ai_count += 1

            except Exception as e:
                # print(f"[WARN] Could not spawn AI vehicle: {e}") # Can be very noisy
                pass
        
        print(f"[INFO] Successfully spawned {spawned_ai_count} AI vehicles (Target was {NUM_VEHICLES}).")
    
    except Exception as e:
        print(f"[ERROR] Could not set up AI traffic: {e}")
    # --- END NEW SECTION ---

    # Camera
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '800')
    cam_bp.set_attribute('image_size_y', '600')
    cam_bp.set_attribute('fov', '90')
    cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle_actor)
    with state_lock:
        actor_list.append(camera)
    camera.listen(lambda img: on_image(img))
    print(f"[INFO] RGB camera attached (id={camera.id})")

    # Collision sensor
    collision_bp = bp_lib.find('sensor.other.collision')
    collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle_actor)
    with state_lock:
        actor_list.append(collision_sensor)
    collision_sensor.listen(lambda event: on_collision(event))
    print(f"[INFO] Collision sensor attached (id={collision_sensor.id})")

    # Lane invasion sensor
    lane_bp = bp_lib.find('sensor.other.lane_invasion')
    lane_sensor = world.spawn_actor(lane_bp, carla.Transform(), attach_to=vehicle_actor)
    with state_lock:
        actor_list.append(lane_sensor)
    lane_sensor.listen(lambda event: on_lane_invasion(event))
    print(f"[INFO] Lane invasion sensor attached (id={lane_sensor.id})")

    print("[INFO] Sensors attached successfully.")

    # Start spectator follow thread
    spec_thread = threading.Thread(target=spectator_follow_loop, daemon=True, name="SpectatorFollow")
    spec_thread.start()
    print("[INFO] Spectator follow thread started.")


def cleanup():
    global client, actor_list
    print("[INFO] cleanup: saving metrics and destroying actors")
    save_metrics()

    with state_lock:
        actors_to_destroy = list(actor_list)
        actor_list.clear()

    print(f"[INFO] Destroying {len(actors_to_destroy)} actors...")
    for a in actors_to_destroy:
        try:
            if a is not None:
                a.destroy()
        except Exception as e:
            print(f"[ERROR] cleanup destroy actor exception: {e}")

    print("[INFO] Cleanup complete.")


if __name__ == "__main__":
    try:
        setup_carla()
        print("[INFO] Starting Flask server on port 5000...")
        app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt — shutting down.")
    except Exception as e:
        print("[ERROR] Unhandled exception in main:", e)
    finally:
        cleanup()

