#!/usr/bin/env python3
"""
Generate 2 Comparison Images for the 6 Regions + Central Region:
1. real_world_6_regions.png -> High-res Real Drone Aerial Map with 6 region boxes & Center
2. carla_map_6_regions.png -> CARLA Town10HD Top-Down Overhead view with projected 6 3D region boxes & Center
"""

import os
import sys
import math
import queue
import cv2
import numpy as np
import carla

# Path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DRONE_MAP_PATH = os.path.join(SCRIPT_DIR, "drone_background_map.png")
OUT_REAL_IMAGE = os.path.join(SCRIPT_DIR, "real_world_6_regions.png")
OUT_CARLA_IMAGE = os.path.join(SCRIPT_DIR, "carla_map_6_regions.png")

# 6 Regions Drone Pixel Polygons (Note: In image space, positive Y is downwards: y_img = -y_annot)
REAL_REGION_POLYGONS = {
    'Region 1': np.array([[1243, 895], [1293, 895], [1324, 1510], [1274, 1510]], dtype=np.int32),
    'Region 2': np.array([[1363, 892], [1413, 892], [1394, 1509], [1338, 1509]], dtype=np.int32),
    'Region 3': np.array([[1432, 735], [1435, 785], [1943, 724], [1938, 677]], dtype=np.int32),
    'Region 4': np.array([[1435, 798], [1440, 848], [1948, 788], [1943, 738]], dtype=np.int32),
    'Region 5': np.array([[1190, 644], [1166, 696], [548, 344], [569, 297]], dtype=np.int32),
    'Region 6': np.array([[1159, 704], [1137, 755], [520, 401], [541, 352]], dtype=np.int32),
}

# 6 Regions CARLA 3D World Coordinates (in meters)
CARLA_REGION_BOXES_3D = {
    'Region 1': [(-30.8, 23.0), (-30.8, 30.0), (69.2, 30.0), (69.2, 22.9)],
    'Region 2': [(-30.8, 11.4), (-30.8, 18.1), (69.2, 19.1), (69.2, 11.7)],
    'Region 3': [(-47.0, 4.4), (-54.7, 4.4), (-54.7, -72.5), (-46.9, -72.5)],
    'Region 4': [(-39.4, 4.4), (-47.0, 4.4), (-46.9, -72.5), (-39.4, -72.5)],
    'Region 5': [(-46.8, 35.1), (-55.3, 35.2), (-55.3, 141.8), (-46.8, 141.8)],
    'Region 6': [(-38.1, 35.1), (-46.8, 35.1), (-46.8, 141.8), (-38.1, 141.8)],
}

REGION_COLORS_BGR = {
    'Region 1': (0, 0, 255),      # Bright Red
    'Region 2': (0, 140, 255),    # Orange
    'Region 3': (0, 220, 0),      # Bright Green
    'Region 4': (255, 200, 0),    # Cyan/Yellow
    'Region 5': (255, 50, 50),    # Blue
    'Region 6': (200, 0, 200),    # Magenta / Purple
    'Center':   (0, 255, 255),    # Bright Yellow
}

def get_camera_intrinsic_matrix(w, h, fov=90.0):
    focal = w / (2.0 * math.tan(fov * math.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = focal
    K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def world_to_pixel(loc, cam_transform, K, w, h):
    matrix = np.array(cam_transform.get_matrix())
    R = matrix[:3, :3]
    cam_pos = np.array([cam_transform.location.x, cam_transform.location.y, cam_transform.location.z])
    pt_world = np.array([loc[0], loc[1], 0.5])
    
    pt_cam_carla = R.T @ (pt_world - cam_pos)
    pt_cv = np.array([pt_cam_carla[1], -pt_cam_carla[2], pt_cam_carla[0]])
    
    if pt_cv[2] <= 0:
        return None
        
    px_h = K @ pt_cv
    u = int(px_h[0] / px_h[2])
    v = int(px_h[1] / px_h[2])
    return (u, v)

def generate_real_world_image():
    print(f"Generating Real-World Drone Map Image -> {os.path.basename(OUT_REAL_IMAGE)}...")
    img = cv2.imread(DRONE_MAP_PATH)
    if img is None:
        print(f"Error: Could not load {DRONE_MAP_PATH}")
        return

    overlay = img.copy()

    # Draw Central Intersection Circle (Center: u=1344, v=756, R=220px)
    center_pt = (1344, 756)
    cv2.circle(overlay, center_pt, 220, REGION_COLORS_BGR['Center'], -1)
    cv2.circle(img, center_pt, 220, REGION_COLORS_BGR['Center'], 3)
    cv2.putText(img, "Center Region", (center_pt[0] - 80, center_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3, cv2.LINE_AA)

    # Draw the 6 Region Polygons
    for name, poly in REAL_REGION_POLYGONS.items():
        color = REGION_COLORS_BGR[name]
        cv2.fillPoly(overlay, [poly], color)
        cv2.polylines(img, [poly], True, color, 4, cv2.LINE_AA)

        # Polygon centroid for label positioning
        cx = int(np.mean(poly[:, 0]))
        cy = int(np.mean(poly[:, 1]))
        
        # Label background box
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(img, (cx - tw//2 - 5, cy - th//2 - 5), (cx + tw//2 + 5, cy + th//2 + 5), (0, 0, 0), -1)
        cv2.putText(img, name, (cx - tw//2, cy + th//2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # Blend overlay for semi-transparency
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    # Add header title banner
    cv2.rectangle(img, (20, 20), (1050, 95), (20, 20, 30), -1)
    cv2.rectangle(img, (20, 20), (1050, 95), (0, 255, 255), 2)
    cv2.putText(img, "REAL DRONE AERIAL MAP - 6 CALIBRATED REGIONS & CENTER", (35, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3, cv2.LINE_AA)

    cv2.imwrite(OUT_REAL_IMAGE, img)
    print(f"Saved: {OUT_REAL_IMAGE}")

def generate_carla_map_image():
    print(f"Generating CARLA Top-Down Map Image -> {os.path.basename(OUT_CARLA_IMAGE)}...")
    
    # Connect to CARLA
    client = carla.Client("localhost", 2000)
    client.set_timeout(15.0)
    world = client.load_world("Town10HD")
    world.set_weather(carla.WeatherParameters.ClearNoon)

    cam_width, cam_height = 1920, 1080
    cam_x, cam_y, cam_z = -50.0, 10.0, 260.0  # Top-down camera view

    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(cam_width))
    cam_bp.set_attribute("image_size_y", str(cam_height))
    cam_bp.set_attribute("fov", "90")

    cam_tf = carla.Transform(
        carla.Location(x=cam_x, y=cam_y, z=cam_z),
        carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
    )

    camera = world.spawn_actor(cam_bp, cam_tf)
    img_q = queue.Queue()
    camera.listen(img_q.put)

    for _ in range(5):
        world.tick()
        try: img_q.get(timeout=1.0)
        except queue.Empty: pass

    carla_img = img_q.get(timeout=3.0)
    camera.stop()
    camera.destroy()

    arr = np.frombuffer(carla_img.raw_data, dtype=np.dtype("uint8"))
    arr = np.reshape(arr, (carla_img.height, carla_img.width, 4))
    img = arr[:, :, :3].copy()
    overlay = img.copy()

    K = get_camera_intrinsic_matrix(cam_width, cam_height, 90.0)

    # Draw Central Intersection Region in CARLA (Center: X=-47.52m, Y=20.71m, R_meters=33m)
    center_carla = (-47.52, 20.71)
    center_px = world_to_pixel(center_carla, cam_tf, K, cam_width, cam_height)
    if center_px:
        r_px = int(33.0 * (K[0, 0] / cam_z))
        cv2.circle(overlay, center_px, r_px, REGION_COLORS_BGR['Center'], -1)
        cv2.circle(img, center_px, r_px, REGION_COLORS_BGR['Center'], 3)
        cv2.putText(img, "Center Region", (center_px[0] - 70, center_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    # Draw the 6 CARLA 3D Region Boxes
    for name, pts_3d in CARLA_REGION_BOXES_3D.items():
        color = REGION_COLORS_BGR[name]
        px_list = []
        for p in pts_3d:
            px = world_to_pixel(p, cam_tf, K, cam_width, cam_height)
            if px:
                px_list.append(px)

        if len(px_list) == 4:
            poly = np.array(px_list, dtype=np.int32)
            cv2.fillPoly(overlay, [poly], color)
            cv2.polylines(img, [poly], True, color, 3, cv2.LINE_AA)

            cx = int(np.mean(poly[:, 0]))
            cy = int(np.mean(poly[:, 1]))

            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.rectangle(img, (cx - tw//2 - 4, cy - th//2 - 4), (cx + tw//2 + 4, cy + th//2 + 4), (0, 0, 0), -1)
            cv2.putText(img, name, (cx - tw//2, cy + th//2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # Blend overlay
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    # Add header title banner
    cv2.rectangle(img, (20, 20), (980, 95), (20, 20, 30), -1)
    cv2.rectangle(img, (20, 20), (980, 95), (0, 255, 255), 2)
    cv2.putText(img, "CARLA TOWN10HD MAP - 6 CALIBRATED REGIONS & CENTER", (35, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3, cv2.LINE_AA)

    cv2.imwrite(OUT_CARLA_IMAGE, img)
    print(f"Saved: {OUT_CARLA_IMAGE}")

def main():
    generate_real_world_image()
    generate_carla_map_image()
    print("\nBoth region comparison images successfully generated!")

if __name__ == "__main__":
    main()
