import math
import numpy as np

# Region perspective transformation matrices (Drone image pixels -> CARLA world meters)
transformation_matrices = {
    1: np.array([[ 0.052, -0.001, -12.4], [ 0.001,  0.051, -15.2], [ 0.000,  0.000,   1.0]]),
    2: np.array([[ 0.053, -0.001, -10.1], [ 0.001,  0.052, -14.8], [ 0.000,  0.000,   1.0]]),
    3: np.array([[ 0.054, -0.002,  -8.5], [ 0.002,  0.053, -13.9], [ 0.000,  0.000,   1.0]]),
    4: np.array([[ 0.051, -0.001, -14.2], [ 0.001,  0.050, -16.1], [ 0.000,  0.000,   1.0]]),
    5: np.array([[ 0.052, -0.001, -12.0], [ 0.001,  0.051, -15.5], [ 0.000,  0.000,   1.0]]),
    6: np.array([[ 0.053, -0.002,  -9.8], [ 0.002,  0.052, -14.2], [ 0.000,  0.000,   1.0]]),
    7: np.array([[ 0.050, -0.001, -15.1], [ 0.001,  0.049, -17.0], [ 0.000,  0.000,   1.0]]),
    8: np.array([[ 0.051, -0.001, -13.0], [ 0.001,  0.050, -16.2], [ 0.000,  0.000,   1.0]]),
    9: np.array([[ 0.052, -0.002, -11.1], [ 0.002,  0.051, -15.0], [ 0.000,  0.000,   1.0]]),
}

def transform_trajectory_with_analysis(coords, region_id=5):
    """Transforms image pixel coordinates into CARLA world coordinates (meters). Returns valid=False on failure."""
    if not coords or len(coords) < 2:
        return {"valid": False, "carla_points": [], "reason": "INSUFFICIENT_POINTS"}
        
    pts = np.asarray(coords, dtype=float)
    if not np.isfinite(pts).all():
        return {"valid": False, "carla_points": [], "reason": "NON_FINITE_PIXEL_COORDINATES"}
        
    M = transformation_matrices.get(region_id, transformation_matrices[5])
    
    # Apply affine/perspective transformation
    ones = np.ones((pts.shape[0], 1))
    pts_h = np.hstack([pts, ones])
    transformed_h = (M @ pts_h.T).T
    
    # Normalize homogeneous coordinates
    carla_pts = transformed_h[:, :2] / transformed_h[:, 2:3]
    
    if not np.isfinite(carla_pts).all():
        return {"valid": False, "carla_points": [], "reason": "NON_FINITE_TRANSFORMED_COORDINATES"}
        
    return {
        "valid": True,
        "carla_points": carla_pts.tolist(),
        "reason": None
    }

def calculate_trajectory_distances(world_points):
    """Calculates path length (meters), net displacement (meters), and max step jump (meters)."""
    points = np.asarray(world_points, dtype=float)

    if points.ndim != 2 or points.shape[0] < 2:
        raise ValueError("Trajectory must contain at least two points")

    if points.shape[1] < 2:
        raise ValueError("Trajectory must contain at least two spatial coordinates")

    if not np.isfinite(points[:, :2]).all():
        raise ValueError("Trajectory contains invalid world coordinates")

    xy = points[:, :2]
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)

    path_length_m = float(steps.sum())
    net_displacement_m = float(np.linalg.norm(xy[-1] - xy[0]))
    max_step_m = float(steps.max()) if len(steps) > 0 else 0.0

    return {
        "path_length_m": round(path_length_m, 2),
        "net_displacement_m": round(net_displacement_m, 2),
        "max_step_m": round(max_step_m, 2),
    }

def wrap_angle_rad(angle):
    """Wraps angle in radians to [-pi, pi]"""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def wrap_angle_deg(angle_deg):
    """Wraps angle in degrees to [-180, 180]"""
    while angle_deg > 180.0: angle_deg -= 360.0
    while angle_deg < -180.0: angle_deg += 360.0
    return angle_deg
