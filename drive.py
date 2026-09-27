import sys
import os
import random
import argparse
import time
import math

# ---------------- ML imports FIRST (important) ----------------
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor
import pickle5 as pickle


# ---------------- Add CARLA egg path ----------------
egg_paths = [
    "/home/raj/CARLA_0.9.15/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg",
    "/home/dasari-raj-vamsi/Desktop/CARLA_0.9.15/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
]

for egg_path in egg_paths:
    if os.path.exists(egg_path):
        sys.path.insert(0, egg_path)
        print(f"Successfully added CARLA egg: {egg_path}")
        break



# ---------------- Your model ----------------
from train3 import BehavioralCloningSegFormer


# ==============================================================================
# Global state
# ==============================================================================

LATEST_IMAGE = None          # front camera image for model
HAS_COLLIDED = False
LANE_DEPARTURES = 0


# ==============================================================================
# Callbacks
# ==============================================================================

def on_collision(event):
    global HAS_COLLIDED
    if not HAS_COLLIDED:
        print(f"!!! COLLISION DETECTED with actor: {event.other_actor.type_id} !!!")
        HAS_COLLIDED = True


def on_lane_invasion(event):
    global LANE_DEPARTURES
    lane_types = {str(lt.type) for lt in event.crossed_lane_markings}
    if "Solid" in lane_types:
        LANE_DEPARTURES += 1
        print(f"--- Solid Lane Departure #{LANE_DEPARTURES} Detected ---")


def on_image(image):
    global LATEST_IMAGE
    LATEST_IMAGE = image


# ==============================================================================
# Main
# ==============================================================================

def drive_with_model(args):
    global HAS_COLLIDED, LANE_DEPARTURES, LATEST_IMAGE

    import carla  # import here

    HAS_COLLIDED = False
    LANE_DEPARTURES = 0
    LATEST_IMAGE = None

    client = None
    actor_list = []

    try:
        # -------- Connect --------
        print("Connecting to CARLA simulator...")
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        world = client.get_world()
        spawn_points = world.get_map().get_spawn_points()

        # -------- Load model --------
        print("Loading trained model and scaler...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = BehavioralCloningSegFormer(model_name=args.model_name).to(device)

        state = torch.load(args.model_path, map_location=device, pickle_module=pickle)
        model.load_state_dict(state)
        model.eval()

        # scaler
        with open(args.scaler_path, "rb") as f:
            scaler = pickle.load(f)

        image_processor = AutoImageProcessor.from_pretrained(args.model_name)

        print("Model and scaler loaded successfully.")

        # -------- Spawn vehicle --------
        vehicle_bp = world.get_blueprint_library().find("vehicle.tesla.model3")

        spawn_point = (
            random.choice(spawn_points)
            if args.spawn_point == -1
            else spawn_points[args.spawn_point]
        )

        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)

        print(f"Spawned vehicle at: {spawn_point.location}")

        # ==============================================================================
        # CAMERAS
        # ==============================================================================

        # ------ FRONT CAMERA — MODEL INPUT ------
        front_cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        front_cam_bp.set_attribute("image_size_x", "800")
        front_cam_bp.set_attribute("image_size_y", "600")
        front_cam_bp.set_attribute("fov", "90")

        front_cam_transform = carla.Transform(
            carla.Location(x=1.5, z=2.4),
            carla.Rotation(pitch=0)
        )

        front_camera = world.spawn_actor(front_cam_bp, front_cam_transform, attach_to=vehicle)
        front_camera.listen(on_image)
        actor_list.append(front_camera)

        # ------ THIRD PERSON CHASE CAMERA ------
        chase_cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        chase_cam_bp.set_attribute("image_size_x", "1024")
        chase_cam_bp.set_attribute("image_size_y", "768")
        chase_cam_bp.set_attribute("fov", "90")

        # positioned BEHIND+ABOVE car
        chase_cam_transform = carla.Transform(
            carla.Location(x=-8.0, z=4.0),   # back 8m, up 4m
            carla.Rotation(pitch=-10, yaw=0, roll=0)
        )

        chase_camera = world.spawn_actor(
            chase_cam_bp,
            chase_cam_transform,
            attach_to=vehicle
        )
        actor_list.append(chase_camera)

        # spectator follows chase cam
        spectator = world.get_spectator()

        def follow_loop():
            while True:
                spectator.set_transform(chase_camera.get_transform())
                world.wait_for_tick()

        import threading
        threading.Thread(target=follow_loop, daemon=True).start()

        print("Front and chase cameras attached.")

        # ------ Sensors ------
        collision_bp = world.get_blueprint_library().find("sensor.other.collision")
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
        collision_sensor.listen(on_collision)
        actor_list.append(collision_sensor)

        lane_bp = world.get_blueprint_library().find("sensor.other.lane_invasion")
        lane_sensor = world.spawn_actor(lane_bp, carla.Transform(), attach_to=vehicle)
        lane_sensor.listen(on_lane_invasion)
        actor_list.append(lane_sensor)

        time.sleep(2)

        # -------- PID --------
        Kp, Ki, Kd = 1.0, 0.05, 0.1
        integral_error = 0.0
        previous_error = 0.0

        print("\n--- AI Driving Started ---")

        start = time.time()

        while time.time() - start < args.duration:

            if HAS_COLLIDED:
                print("Collision detected. Stopping.")
                break

            world.wait_for_tick()

            if LATEST_IMAGE is None:
                continue

            # image to numpy
            img_bgra = np.frombuffer(
                LATEST_IMAGE.raw_data, dtype=np.uint8
            ).reshape((LATEST_IMAGE.height, LATEST_IMAGE.width, 4))

            img_rgb = img_bgra[:, :, :3][:, :, ::-1]
            pil_image = Image.fromarray(img_rgb)

            # forward pass
            with torch.no_grad():
                inputs = image_processor(images=pil_image, return_tensors="pt").to(device)
                normalized = model(inputs["pixel_values"])
                real = scaler.inverse_transform(normalized.cpu().numpy())[0]

            speed_pred, steer_deg = real[0], real[1]

            # vehicle speed
            v = vehicle.get_velocity()
            current_speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)

            # PID
            error = speed_pred - current_speed
            integral_error += error * 0.05
            derivative = (error - previous_error) / 0.05
            previous_error = error

            pid_out = Kp * error + Ki * integral_error + Kd * derivative

            throttle = float(np.clip(pid_out, 0, 1)) if pid_out > 0 else 0.0
            brake = float(np.clip(-pid_out, 0, 1)) if pid_out < 0 else 0.0

            steer = float(np.clip(steer_deg / 60.0, -1.0, 1.0))

            vehicle.apply_control(carla.VehicleControl(
                throttle=throttle,
                brake=brake,
                steer=steer
            ))

        else:
            print("Duration completed successfully.")

    except Exception as e:
        print(f"ERROR: {e}")

    finally:
        print("\n--- Session Summary ---")
        print("Solid Lane Departures:", LANE_DEPARTURES)
        print("Collision:", HAS_COLLIDED)

        print("Destroying actors...")
        if client is not None:
            try:
                client.apply_batch([carla.command.DestroyActor(a) for a in actor_list])
            except:
                pass

        print("Done.")


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--scaler_path", default="label_scaler.pkl")
    parser.add_argument("--model_name",
                        default="nvidia/segformer-b1-finetuned-cityscapes-1024-1024")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--spawn-point", type=int, default=-1)

    args = parser.parse_args()
    drive_with_model(args)
