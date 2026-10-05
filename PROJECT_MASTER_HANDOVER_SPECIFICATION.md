# 📌 PROJECT MASTER HANDOVER SPECIFICATION & ARCHITECTURE REFERENCE
**Repository**: [`real2sim-trajectory-style-transfer`](https://github.com/rajvamsi12/real2sim-trajectory-style-transfer)  
**Latest Commit Hash**: `d769764a81265ad4eed1a4e5d6f9671c1f3e589c` (`main` branch)  
**Date**: October 5, 2026  

---

## 1. 🎯 Project Overview & Core Problem Statement

The goal of this project is a **3-Way Comparative Benchmark** analyzing autonomous and human driving behavior under identical traffic scenario conditions in CARLA Town10HD:

1. **Mode 1: `EXPERT` (Real Ground Vehicles)**: Replays exact human driving behavior extracted from real aerial drone video dataset (`default.json`).
2. **Mode 2: `AUTOPILOT` (CARLA Rule-Based Traffic Manager)**: Standard CARLA Traffic Manager controlling the vehicles.
3. **Mode 3: `ML` (Goal-Conditioned Neural Network Policy)**: Trained machine learning model controlling the vehicles.

### Benchmark Metrics Tracked (`compare3.py`, `carla_metrics.csv`, `combined_stats.csv`):
* 💥 **Collision Rate / Collisions per km**
* ⏱️ **Time-To-Collision (TTC) & Safety Margins**
* 📈 **Speed & Acceleration Smoothness (Jerk profile)**
* 📍 **Trajectory Deviation (Lateral & Longitudinal deviation vs Expert)**
* 🏆 **Driving Score & Route Completion Rate**

---

## 2. 🗺️ Regional Affine Transformation Matrices (6-Region Calibration)

The pipeline converts drone image pixel coordinates $(u, -v)$ (where $v$ is negative in dataset convention) into CARLA 3D world coordinates $(X,Y)$ in meters:

$$\begin{bmatrix} X_{\text{CARLA}} \\ Y_{\text{CARLA}} \end{bmatrix} = M_{\text{Region}} \times \begin{bmatrix} u \\ -v \\ 1 \end{bmatrix}$$

### 6-Region Affine Matrices (`convertion.py` & `timemethod.py`):
```python
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
```
*Old 4-region matrices (`East`, `West`, `North`, `South`) are kept **commented out** in `convertion.py` and `timemethod.py`.*

### Region Bounding Polygons & Central Intersection Region:
```python
region_definitions = {
    'Region_1': [(1243, -895), (1293, -895), (1324, -1510), (1274, -1510), (1243, -895)],
    'Region_2': [(1363, -892), (1413, -892), (1394, -1509), (1338, -1509), (1363, -892)],
    'Region_3': [(1432, -735), (1435, -785), (1943, -724), (1938.5, -677), (1432, -735)],
    'Region_4': [(1435, -798), (1440, -848), (1948, -788), (1943.5, -738), (1435, -798)],
    'Region_5': [(1190, -644), (1166, -696), (548, -344), (569.5, -297), (1190, -644)],
    'Region_6': [(1159, -704), (1137, -755), (520.5, -401), (541, -352.5), (1159, -704)]
}

center_x, center_y = 1344, -756
center_radius = 220  # 220 pixels on drone map (corresponding to 33.0m in CARLA)
```

---

## 3. ⚙️ Execution Rules & Pipeline Rules

1. **Vehicle Spawn Rule (`debug/batch_process_scenarios.py`)**:
   - Vehicles are only spawned if their initial position at `init_frame` falls inside `Region_1` .. `Region_6` OR inside the `Center` region. If outside, they are skipped.
2. **Strict Trajectory Trimming (`timemethod.py`)**:
   - Waypoint points falling into `Unclassified` space (outside all 6 region boxes AND outside Center) are **strictly trimmed / filtered out**.
3. **Full-Context Trajectory Analysis**:
   - Stages 2 & 3 analyze the vehicle's entire 15–20 second trajectory across all recorded frames *first* to determine true entry/exit regions before cropping to scenario frame range `[start_frame .. end_frame]`.

---

## 4. 🛠️ Interactive Tools & Visualizers

* **`click_real_map_coordinates.py`**: Pygame interactive picker on `drone_background_map.png`. Click anywhere to query exact `default.json` coordinates `(X, -Y)`. Supports Zoom In/Out (mouse wheel), WASD/drag panning, and 30s screen timer.
* **`click_carla_map_coordinates.py`**: Pygame interactive picker on CARLA Town10HD overhead view.
* **`generate_6_regions_comparison_images.py`**: Renders `real_world_6_regions.png` and `carla_map_6_regions.png`.

---

## 5. 🧠 Mode 3 Machine Learning Architecture Specification

### Inputs:
1. **Front RGB Camera Image**: Processed via SegFormer encoder (`nvidia/mit-b0`).
2. **LiDAR BEV Point Cloud Grid**: 2D BEV projection ($256 \times 256$) for 360° obstacle awareness.
3. **Target Destination Waypoint $[X_{\text{target}}, Y_{\text{target}}]$**: Local ego-frame target 6m ahead.
4. **Ego Speed & High-Level Road Command** (`RoadOption`).

### Output Strategy:
* Predicts $K=6$ future waypoints at 0.5s intervals (3.0s horizon) + Target Speed.
* A **PID / Pure Pursuit controller** turns waypoints into throttle, brake, and steering actions.
* Auxiliary heads for direct steering/throttle/brake for baseline ablation.

### Symmetric Augmentations:
* When applying horizontal flip augmentation (`Image.FLIP_LEFT_RIGHT`):
  - Mirror LiDAR BEV along $Y$-axis ($y \rightarrow -y$).
  - Negate target point and waypoint $Y$-coordinates ($y \rightarrow -y$).
  - Swap left/right high-level commands.

---

## 🚀 Quick Execution Commands

```bash
cd /home/raj/Desktop/carlascripts

# 1. Start CARLA Server Daemon
/home/raj/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low &

# 2. Run Pipeline for any scenario (e.g. 4-way intersection Scenario 1 or Overtaking Scenario 10)
./debug/run_all_scenarios_pipeline.sh --target "4-way intersection/1"

# 3. Interactive Real Drone Map Coordinate Picker
python click_real_map_coordinates.py

# 4. Generate 6-Region Comparison Map Images
python generate_6_regions_comparison_images.py

# 5. Run 3-Way Driving Benchmark Comparison (Expert vs Autopilot vs ML)
python compare3.py
```
