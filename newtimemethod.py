import math
import numpy as np
import json
import os
import carla
import time


transformation_matrices = {

    'West': np.array([
        [4.09206559e-02, 7.57489044e-02, -4.16217959e+01],
        [-1.03373454e-01, 6.94211744e-02, 2.03896072e+02]
    ]),

    'South': np.array([
        [2.32258700e-04, -1.44409373e-01, -1.63354148e+02],
        [8.96053433e-02, -4.90025318e-03, -1.05245581e+02]
    ]),

    'East': np.array([
        [-4.33082964e-03, 8.30972682e-02, 2.58035321e+01],
        [-1.35561683e-01, -3.86102939e-04, 1.95960709e+02]
    ]),

    'North': np.array([
        [-2.17355836e-02, -1.29121669e-01, -1.16650669e+02],
        [1.27472978e-01, -1.81544206e-02, -1.60394644e+02]
    ])
}

# transformation_matrices = {

#     'West': np.array([
#         [4.57983003e-02, 8.22179685e-02, -4.28989710e+01],
#         [-5.36060772e-02, 3.61433161e-02, 1.23026566e+02]
#     ]),

#     'South': np.array([
#         [2.32258700e-04, -1.44409373e-01, -1.63354148e+02],
#         [8.96053433e-02, -4.90025318e-03, -1.05245581e+02]
#     ]),

#     'East': np.array([
#         [-9.45128106e-03, 8.86337120e-02, 3.74601249e+01],
#         [-3.99441989e-02, -4.82901744e-03, 5.62070001e+01]
#     ]),

#     'North': np.array([
#         [-9.32356691e-03, -5.20106732e-02, -8.23492455e+01],
#         [1.26424341e-01, -6.77161360e-03, -1.51511183e+02]
#     ])
# }


region_definitions = {
    'West': [(1212, -605), (1125, -760), (0, -130), (100, 0), (1212, -605)],
    'South': [(1200, -890), (1450, -890), (1450, -1512), (1200, -1512), (1200, -890)],
    'East': [(1425, -885), (1425, -690), (2688, -580), (2688, -730), (1425, -885)],
    'North': [(1425, -680), (1220, -635), (1280, 0), (1430, 0), (1425, -680)]
}

center_x, center_y = 1344, -756
center_radius = 350

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

def get_closest_region(point):
    x, y = point
    best_region = 'East'
    best_dist = float('inf')
    for region, polygon in region_definitions.items():
        cx = sum(p[0] for p in polygon) / len(polygon)
        cy = sum(p[1] for p in polygon) / len(polygon)
        dist = math.hypot(x - cx, y - cy)
        if dist < best_dist:
            best_dist = dist
            best_region = region
    return best_region

def classify_point(point):
    # 1st Priority: Check 4 direction arm polygons (East, West, North, South)
    for region, polygon in region_definitions.items():
        if point_in_polygon(point, polygon):
            return region
    # 2nd Priority: Check complete expanded intersection circle (radius R=350px around 1344, -756)
    x, y = point
    distance = math.hypot(x - center_x, y - center_y)
    if distance <= center_radius:
        return 'Center'
    # 3rd Priority: Fall back to closest direction region (eliminates Unclassified drops!)
    return get_closest_region(point)

def smoothstep(t):
    return t * t * (3 - 2 * t)

class TrajectoryAnalyzer:
    def __init__(self):
        self.segments = []
    
    def analyze_trajectory(self, real_coords):
        """First pass: Analyze entire trajectory to find entry/exit segments"""
        regions = [classify_point(coord) for coord in real_coords]
        
        # Find center crossing segments
        in_center = False
        current_segment = None
        
        for i, (coord, region) in enumerate(zip(real_coords, regions)):
            if region == 'Center' and not in_center:
                # Entering center
                entry_region = regions[i-1] if i > 0 else 'Unclassified'
                current_segment = {
                    'entry_region': entry_region,
                    'entry_point': real_coords[i-1] if i > 0 else coord,
                    'start_idx': i,
                    'points': [coord]
                }
                in_center = True
            elif region == 'Center' and in_center:
                # Still in center
                current_segment['points'].append(coord)
            elif region != 'Center' and in_center:
                # Exiting center
                current_segment['exit_region'] = region
                current_segment['exit_point'] = coord
                current_segment['end_idx'] = i
                self.segments.append(current_segment)
                in_center = False
        
        return self.segments
    
    def get_blend_factor(self, point, entry_point, exit_point):
        """Calculate blend factor along the straight line from entry to exit"""
        if entry_point is None or exit_point is None:
            return 0
        
        p = np.array(point)
        a = np.array(entry_point)
        b = np.array(exit_point)
        ab = b - a
        ap = p - a
        
        if np.dot(ab, ab) == 0:
            return 0
        
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        return smoothstep(t)

def transform_trajectory_with_analysis(real_coords):
    """Two-pass transformation with full trajectory knowledge"""
    analyzer = TrajectoryAnalyzer()
    segments = analyzer.analyze_trajectory(real_coords)
    
    carla_coords = []
    
    for i, real_coord in enumerate(real_coords):
        region = classify_point(real_coord)
        
        # Check if this point is in a center segment
        in_segment = None
        for segment in segments:
            if segment['start_idx'] <= i < segment['end_idx']:
                in_segment = segment
                break
        
        if in_segment and region == 'Center':
            # Apply blended transformation for center crossings
            entry_region = in_segment['entry_region']
            exit_region = in_segment['exit_region']
            
            if (entry_region in transformation_matrices and 
                exit_region in transformation_matrices):
                
                blend_factor = analyzer.get_blend_factor(
                    real_coord, 
                    in_segment['entry_point'], 
                    in_segment['exit_point']
                )
                
                M_entry = transformation_matrices[entry_region]
                M_exit = transformation_matrices[exit_region]
                M_blend = (1 - blend_factor) * M_entry + blend_factor * M_exit
                
                x, y = real_coord
                carla_coord = tuple(M_blend @ np.array([x, y, 1]))
            else:
                # Fallback to entry region if exit unknown
                M = transformation_matrices.get(entry_region, transformation_matrices['East'])
                x, y = real_coord
                carla_coord = tuple(M @ np.array([x, y, 1]))
        else:
            # Standard transformation for non-center points
            carla_coord = transform_to_carla(real_coord)
        
        if carla_coord is not None:
            carla_coords.append((carla_coord[0], carla_coord[1]))
    
    return carla_coords

def transform_to_carla(real_point):
    region = classify_point(real_point)
    return transform_to_carla_with_region(real_point, region)

def transform_to_carla_with_region(real_point, region):
    if region not in transformation_matrices:
        region = get_closest_region(real_point)
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
    
    # First, collect all real coordinates
    real_coords = []
    for item_idx, item in enumerate(data.get('items', [])):
        for ann in item.get('annotations', []):
            if ann.get('attributes', {}).get('track_id') == target_track_id:
                bbox = ann.get('bbox')
                if bbox and len(bbox) == 4:
                    x, y, w, h = bbox
                    mid_x = x + w / 2
                    mid_y = y + h / 2
                    mid_y_flipped = -mid_y  # Flip y if needed for your system
                    real_coords.append((mid_x, mid_y_flipped))
    
    # Now transform with full trajectory knowledge
    carla_coords = transform_trajectory_with_analysis(real_coords)
    return carla_coords

def main():
    input_file = 'selected_annotations.json'
    target_track_id = 1162
    # target_track_id = 1712
    carla_coords = get_carla_coords_for_track(input_file, target_track_id)
    if not carla_coords:
        print("No CARLA coordinates found for given track_id.")
        return

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
            time.sleep(0.05)
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
