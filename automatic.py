import math
import numpy as np
import json
import os
import carla
import time

# --- Updated Affine transformation matrices ---
transformation_matrices = {
    'East': np.array([
        [-0.008071, 0.084431, 32.491831],
        [0.125319, 0.001716, -141.511227]
    ]),
    "West": np.array([[0.03755768, 0.07700213, -37.13919422],
                     [0.11282144, -0.0729956, -178.56198166]]),
    "North": np.array([[0.01753432, 0.13574565, 31.54156408],
                      [0.12600283, -0.0015926, -148.24315321]]),
    "South": np.array([[-0.00233274, 0.13773347, 65.20834494],
                      [0.09464766, 0.01271084, -93.9143594]]),
     "Center Intersection": np.array([[ 0.02973276,  0.11443728,  0.48077602],
                                         [ 0.11123173,  0.00531112, -122.21446525]])

}

region_definitions = {
    'West': [(1212, -605), (1125, -760), (0, -130), (100, 0), (1212, -605)],
    'South': [(1200, -890), (1450, -890), (1450, -1512), (1200, -1512), (1200, -890)],
    'East': [(1425, -885), (1425, -690), (2688, -580), (2688, -730), (1425, -885)],
    'North': [(1425, -680), (1220, -635), (1280, 0), (1430, 0), (1425, -680)]
}

def point_in_polygon(point, polygon):
    x, y = point
    inside = False
    n = len(polygon)
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    x_inters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= x_inters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def classify_point(point):
    for region, polygon in region_definitions.items():
        if point_in_polygon(point, polygon):
            return region
    # Center intersection check
    cx, cy = (1344, -756)
    distance = math.hypot(point[0]-cx, point[1]-cy)
    return 'Center Intersection' if distance <= 300 else 'Unclassified'

def blended_transform(real_point):
    region = classify_point(real_point)
    if region == 'Center Intersection':
        # Center coordinates
        cx, cy = (1344, -756)
        x, y = real_point
        
        # Calculate direction vector from center
        dx = x - cx
        dy = y - cy
        
        # Determine neighboring region based on dominant direction
        if abs(dx) > abs(dy):  # East/West dominant
            neighbor_region = 'East' if dx > 0 else 'West'
        else:  # North/South dominant
            neighbor_region = 'North' if dy > 0 else 'South'
        
        # Calculate blend factor based on distance from center
        distance = math.hypot(dx, dy)
        blend_factor = min(1, distance / 200)  # 200 = blend radius
        
        # Get matrices
        center_matrix = transformation_matrices['Center Intersection']
        neighbor_matrix = transformation_matrices[neighbor_region]
        
        # Linear blend between matrices
        blended_matrix = (1 - blend_factor) * center_matrix + blend_factor * neighbor_matrix
        return tuple(blended_matrix @ np.array([x, y, 1]))
    else:
        return transform_to_carla(real_point)


def transform_to_carla(real_point):
    region = classify_point(real_point)
    if region not in transformation_matrices:
        return None
    x, y = real_point
    affine_matrix = transformation_matrices[region]
    return tuple(affine_matrix @ np.array([x, y, 1]))

def load_json_file_safely(filepath):
    if not os.path.isfile(filepath):
        print(f"File not found: {filepath}")
        return None
    if os.path.getsize(filepath) == 0:
        print(f"File is empty: {filepath}")
        return None
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error in {filepath}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error reading {filepath}: {e}")
        return None

def get_carla_coords_for_track(input_file, target_track_id):
    data = load_json_file_safely(input_file)
    if data is None:
        print("Aborting: could not load input JSON.")
        return []
    carla_coords = []
    for item_idx, item in enumerate(data.get('items', [])):
        for ann in item.get('annotations', []):
            if ann.get('attributes', {}).get('track_id') == target_track_id:
                bbox = ann.get('bbox')
                if bbox and len(bbox) == 4:
                    x, y, w, h = bbox
                    mid_x = x + w / 2
                    mid_y = y + h / 2
                    mid_y_flipped = -mid_y  # Flip y if needed for your system
                    real_coord = (mid_x, mid_y_flipped)
                    carla_coord = blended_transform(real_coord)  # Use blended transform here
                    if carla_coord is not None:
                        carla_coords.append((carla_coord[0], carla_coord[1]))
    return carla_coords

def main():
    # --- Get CARLA coordinates from annotation file ---
    input_file = 'selected_annotations.json'
    target_track_id = 1162  # Change as needed
    carla_coords = get_carla_coords_for_track(input_file, target_track_id)
    if not carla_coords:
        print("No CARLA coordinates found for given track_id.")
        return

    # --- Connect to CARLA and place markers ---
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    try:
        world = client.load_world("Town10HD")
        spectator = world.get_spectator()
        camera_transform = carla.Transform(
            carla.Location(x=-50, y=20, z=80),
            carla.Rotation(pitch=-90, yaw=0)
        )
        spectator.set_transform(camera_transform)

        for idx, (x, y) in enumerate(carla_coords):
            point_location = carla.Location(x=x, y=y, z=0.0)
            world.debug.draw_string(
                point_location,
                "o",
                color=carla.Color(r=255, g=0, b=0),
                life_time=60,
                persistent_lines=True
            )
            print(f"Marker {idx+1} placed at ({x}, {y})")
            time.sleep(0.05)  # Slight delay for visualization

        print(f"Placed {len(carla_coords)} markers. The simulation is now running...")
        print("Press Ctrl+C to exit.")
        while True:
            world.tick()
    except KeyboardInterrupt:
        print("\nExiting simulation...")
    finally:
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

if __name__ == '__main__':
    main()
