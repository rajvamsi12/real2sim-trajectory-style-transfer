import math
import numpy as np
import json
import os

# --- OLD 4-REGION MATRICES (COMMENTED OUT AS REQUESTED) ---
# transformation_matrices_old = {
#     'East': np.array([[-0.00961, 0.0874, 37.558], [0.128, -0.0169, -159.376]]),
#     'West': np.array([[0.0248, 0.069, -29.205], [0.1209, -0.0678, -183.564]]),
#     'North': np.array([[0.00135, 0.138, 54.305], [0.1505, 0.0006, -180.504]]),
#     'South': np.array([[0.01824, 0.1271, 25.640], [0.1369, -0.0033, -168.171]]),
#     'Center Intersection': np.array([
#         [-0.045097, 0.103548, 99.947758],
#         [0.174581, -0.022419, -230.880645]
#     ])
# }

# --- NEW 6-REGION MATRICES ---
transformation_matrices = {
    'Region_1': np.array([
        [ 9.15789638e-16, -1.62601626e-01, -1.76328455e+02],
        [ 1.41000000e-01,  7.18861789e-03, -1.45854187e+02]
    ]),
    'Region_2': np.array([
        [-4.99830858e-16, -1.62074554e-01, -1.75370502e+02],
        [ 1.32966643e-01, -5.79459667e-03, -1.74976481e+02]
    ]),
    'Region_3': np.array([
        [-1.84595207e-02,  1.58203852e-01,  9.58468818e+01],
        [-1.50247737e-01, -1.15397598e-02,  2.11009803e+02]
    ]),
    'Region_4': np.array([
        [-1.75309936e-02,  1.49335088e-01,  1.04903592e+02],
        [-1.49625364e-01, -1.42148462e-02,  2.07787633e+02]
    ]),
    'Region_5': np.array([
        [ 7.67514596e-02,  1.35989693e-01, -5.03501531e+01],
        [-1.37220307e-01,  6.20209705e-02,  2.38349570e+02]
    ]),
    'Region_6': np.array([
        [ 8.02705481e-02,  1.40459425e-01, -3.21354719e+01],
        [-1.38938860e-01,  5.93620991e-02,  2.37906466e+02]
    ])
}

region_definitions = {
    'Region_1': [(1243, -895), (1293, -895), (1324, -1510), (1274, -1510), (1243, -895)],
    'Region_2': [(1363, -892), (1413, -892), (1394, -1509), (1338, -1509), (1363, -892)],
    'Region_3': [(1432, -735), (1435, -785), (1943, -724), (1938.5, -677), (1432, -735)],
    'Region_4': [(1435, -798), (1440, -848), (1948, -788), (1943.5, -738), (1435, -798)],
    'Region_5': [(1190, -644), (1166, -696), (548, -344), (569.5, -297), (1190, -644)],
    'Region_6': [(1159, -704), (1137, -755), (520.5, -401), (541, -352.5), (1159, -704)]
}

center_x, center_y = 1344, -756
center_radius = 220

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
    distance = math.hypot(point[0]-center_x, point[1]-center_y)
    return 'Center' if distance <= center_radius else 'Unclassified'

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

if __name__ == '__main__':
    process_track_id_bboxes('selected_annotations.json', 1162)