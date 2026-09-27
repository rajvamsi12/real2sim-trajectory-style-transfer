import math
import numpy as np
import json
import os

# Precomputed affine transformation matrices for each region
transformation_matrices = {
    'East': np.array([[-0.00961, 0.0874, 37.558], [0.128, -0.0169, -159.376]]),
    'West': np.array([[0.0248, 0.069, -29.205], [0.1209, -0.0678, -183.564]]),
    'North': np.array([[0.00135, 0.138, 54.305], [0.1505, 0.0006, -180.504]]),
    'South': np.array([[0.01824, 0.1271, 25.640], [0.1369, -0.0033, -168.171]]),
    'Center Intersection': np.array([
        [-0.045097, 0.103548, 99.947758],
        [0.174581, -0.022419, -230.880645]
    ])
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
    return 'Center Intersection' if distance <= 200 else 'Unclassified'

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

def process_track_id_bboxes(input_file, target_track_id):
    data = load_json_file_safely(input_file)
    if data is None:
        print("Aborting: could not load input JSON.")
        return

    for item_idx, item in enumerate(data.get('items', [])):
        for ann in item.get('annotations', []):
            if ann.get('attributes', {}).get('track_id') == target_track_id:
                bbox = ann.get('bbox')
                if bbox and len(bbox) == 4:
                    # Calculate midpoint
                    x, y, w, h = bbox
                    mid_x = x + w / 2
                    mid_y = y + h / 2
                    mid_y_flipped = -mid_y
                    
                    # Transform to CARLA coordinates
                    real_coord = (mid_x, mid_y_flipped)
                    carla_coord = transform_to_carla(real_coord)
                    
                    # Print results
                    print(f"Frame {item_idx}:")
                    print(f"  Midpoint (real): {real_coord}")
                    print(f"  Transformed (CARLA): {carla_coord}\n")

# Example usage:
process_track_id_bboxes('selected_annotations.json', 1162)