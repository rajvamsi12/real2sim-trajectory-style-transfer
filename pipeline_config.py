import os

PIPELINE_DIR = "/home/raj/Desktop/carlascripts/validation_pipeline"
STABILIZE_EKF_CSV = "/home/raj/Desktop/carlascripts/stabilize_ekf_final2.csv"
INPUT_ANNOTATIONS_FILE = STABILIZE_EKF_CSV if os.path.exists(STABILIZE_EKF_CSV) else "/home/raj/Desktop/carlascripts/default.json"
VIDEO_FILE = "/home/raj/Desktop/carlascripts/DJI_0003-002.MP4"
TIDS_FILE = os.path.join(PIPELINE_DIR, "tids.txt")
AUDIT_LOG_FILE = os.path.join(PIPELINE_DIR, "candidate_validation_audit.json")
RUN_MANIFEST_FILE = os.path.join(PIPELINE_DIR, "run_manifest.json")
PROTECTED_MANIFEST_FILE = os.path.join(PIPELINE_DIR, "protected_scenarios.json")
SCENARIOS_DIR = "/home/raj/Desktop/carlascripts/Scenariosssss"

PYTHON_BIN = "/home/raj/miniconda3/envs/carla_env/bin/python"

PIPELINE_CONFIG = {
    "temporal": {
        "target_sample_fps": 4.28274,
        "target_frame_count": 87, # 87 frame indices [f0, f0+86]
        "target_timestamp_span_s": 20.081,
        "encoded_playback_duration_s": 20.314,
    },
    "ego": {
        "min_path_length_m": 40.0, # Physically calibrated for drone FOV (Max dataset distance in 20s is 90.3m)
        "max_step_jump_m": 15.0, # Max plausible single-frame step jump
    },
    "focus": {
        "proximity_radius_px": 220.0, # ~14.0 meters
        "proximity_radius_m": 14.0,
        "min_track_coverage_ratio": 1.0, # 100% continuous coverage required
    },
    "dense": {
        "trigger_min_vehicles": 4,
        "max_total_vehicles": 8, # 8 total vehicles including Ego
    },
    "sparse": {
        "min_total_vehicles": 2,
        "max_total_vehicles": 3, # 2 to 3 total vehicles including Ego
        "min_pairwise_separation_m": 19.0, # 300 px
    },
    "intersection": {
        "x_min": 1150.0,
        "x_max": 1650.0,
        "y_min": 500.0,
        "y_max": 950.0,
    },
    "preservation": {
        "protect_existing_scenarios": True,
        "allow_overwrite": False,
    }
}

CATEGORIES = [
    "4-way intersection",
    "Dense",
    "Following",
    "Lane Change",
    "Left",
    "Overtaking ",
    "Right",
    "Sparse",
    "slight left",
    "slight right"
]

PROTECTED_SCENARIOS = [
    {"category": "Left", "scenario_id": "scenario_0001", "ego_id": 709},
    {"category": "Following", "scenario_id": "scenario_0001", "ego_id": 2489},
    {"category": "Following", "scenario_id": "scenario_0002", "ego_id": 273},
]
