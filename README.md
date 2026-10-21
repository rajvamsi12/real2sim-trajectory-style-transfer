# While in London do as the Gwaliorites do: Style Transfer of Trajectories across Traffic Domains in Real-to-Sim

[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red.svg)](raj_vamsi_wafr.pdf)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![CARLA 0.9.15](https://img.shields.io/badge/CARLA-0.9.15-orange.svg)](https://carla.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

Official repository for the research paper: **"While in London do as the Gwaliorites do: Style Transfer of Trajectories across Traffic Domains in Real-to-Sim"** by **Dasari Raj Vamsi** and **Rahul Kala** (*Center for Autonomous Systems, ABV-Indian Institute of Information Technology and Management Gwalior, India*).

> **Working on this repo?** Start with [`AGENTS.md`](AGENTS.md) (current rules, settings, numbers and tools), then [`HANDOVER.md`](HANDOVER.md) (project facts, open questions) and [`TEST_CHECKLIST.md`](TEST_CHECKLIST.md). The collision investigation is in [`BUG_COLLISIONS.md`](BUG_COLLISIONS.md).

> **Status (October 2026).** The replay pipeline was reworked after collisions were traced to the scenario data and to simulation bugs: scenarios are now validated, collisions are predicted offline, small overlaps are repaired by moving paths apart, and a sixth behaviour (vehicle following) was added. The benchmark results further down are the **earlier** (January 2026) numbers and have not been re-run with the current pipeline.

---

## 📌 Abstract & Overview

Autonomous driving (AD) systems are primarily developed and evaluated under American and European traffic environments, where rule-following behavior is standard and aggressive driving is rare. In contrast, traffic in developing regions (such as Gwalior, India)—characterized by heavy-tail behaviors, frequent overtaking, heterogeneous vehicle mix, and lack of strict lane discipline—is underrepresented.

This project introduces a **Real-to-Sim Style Transfer Framework** that transfers unorganized, aggressive real-world traffic trajectories recorded by high-altitude drones in India into simulated European/American urban environments (CARLA simulator, Town10HD). 

### Key Technical Highlights:
1. **Map-Aware Affine Transformation**: Maps real-world drone pixel coordinates to CARLA Town10HD coordinates with six calibrated regions (one per road arm and direction) plus a blended centre, matching road lengths and widths and mapping Indian left-hand traffic onto CARLA's right-hand lanes.
2. **Time-Synchronised Replay**: Vehicles follow their recorded trajectories on time (time-indexed PID with launch at the real speed), with an exact kinematic "expert" replay and an optional box-based follower collision avoidance.
3. **Scenario Validation & Collision Handling**: Rejects unusable scenarios (pedestrians, missing tracks, no CARLA path), predicts car-to-car, kerb and spawn failures offline, and repairs small CARLA-body overlaps (≥ −0.5 m) by shifting paths apart by at most 0.7 m.
4. **Synthetic Ego-Centric Dataset Generation**: Synchronizes monocular camera views with control actions (`[speed, steering]`) under realistic heavy-tailed traffic scenarios.
5. **Zero-Shot LLM Reasoning**: Utilizes pre-trained LLMs to classify 15-second trajectory segments into maneuver categories without explicit fine-tuning.
6. **Vision-Based Behavioral Cloning (SegFormer)**: Trains a `BehavioralCloningSegFormer` model (`nvidia/mit-b0` encoder) for end-to-end autonomous steering and velocity control.
7. **Triple-Baseline Benchmarking Harness**: Compares performance across **Ground Truth Expert (Replay)**, **ML (SegFormer)**, and **CARLA Autopilot (Traffic Manager)** across 100 scenario categories.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A[High-Altitude Drone Video / Trajectories] --> B[Region-Based Affine Transformation]
    B --> C[Map Coordinate Mapping - CARLA Town10HD]
    C --> V[Scenario Checks, Collision Prediction & Path Separation]
    V --> D[Time-Synchronised PID Replay / Expert Replay]
    D --> E[CARLA Simulator Execution]
    E --> F[Ego-Camera Monocular Video & Telemetry Log]
    E --> G[15s Trajectory Prompt Exporter]
    G --> H[Zero-Shot LLM Reasoning & Maneuver Tagging]
    F --> I[SegFormer Vision-to-Control Training]
    I --> J[Closed-Loop Driving Evaluation]
    H --> K[Interactive Web Scenario Catalog]
    J --> L[Benchmarking: Expert vs ML vs Autopilot]
```

---

## 📁 Repository Structure

```
.
├── README.md                              # Main documentation (this file)
├── AGENTS.md                              # Current rules, settings, numbers, tools (read first)
├── HANDOVER.md                            # Project facts, open questions, session log
├── BUG_COLLISIONS.md                      # Collision investigation: causes, fixes, verification
├── TEST_CHECKLIST.md                      # Commands and numbers a change must meet
├── requirements.txt                       # Python dependencies
├── start_carla.sh                         # CARLA 0.9.15 simulator launcher script
├── run_sample_test_pipeline.sh            # 1-command zero-shot LLM & video rendering pipeline
│
├── Real2Sim Trajectory & Map Alignment
│   ├── timemethod.py                      # Six region affine matrices, centre blending, region classification
│   ├── newtimemethod.py                   # Refined transformation matrices for Town10HD
│   ├── convertion.py                      # Affine matrix pre-computations
│   ├── automatic.py                       # Automated region coordinate transforms
│   └── prepare_data.py                    # Preprocessing raw drone trajectory CSV/JSON inputs
│
├── Trajectory Mining & Dataset Generation
│   ├── mine_15s_drone_catalog.py          # Mines 15-second trajectory windows from drone logs
│   ├── mine_unique_ego_catalog.py         # Filters catalog into 10 maneuver classes (overtaking, lane change, etc.)
│   └── mine_massive_catalog.py            # High-density region track miner
│
├── Simulation & Control Engine
│   ├── scc.py                             # SimVehicle: time-indexed PID route tracking, in-gear launch
│   ├── scenario_generator.py              # Spawns vehicles & configures trajectory replays
│   ├── simulate_and_record_scenarios.py   # CapturingVehicle (box-based follower avoidance), annotation parser, recording
│   ├── SCENARIO.py                        # Alternative scenario simulation runner
│   └── overlay_scenario_bboxes.py         # Draws 2D bounding boxes & telemetry overlays on video frames
│
├── Scenario Validation & Collision Handling
│   ├── scenario_checks.py                 # Offline checks (reject/warn), TrackStore, box geometry, in-box ratio
│   ├── debug/batch_process_scenarios.py   # Stages 1-4 (matrix, time method, spawns + spawn yaw, CARLA replay/videos)
│   ├── debug/check_scenarios.py           # Folder checks, per-behaviour CSV audit, headless CARLA collision check
│   ├── debug/make_validated_scenarios.py  # Build scenario folders from audited candidates
│   ├── debug/predict_collisions.py        # Offline car-contact / kerb / spawn prediction for all scenarios
│   ├── debug/apply_separation.py          # Repair small overlaps by moving paths apart
│   └── debug/render_scenario_evidence.py  # One scenario folder: drone video, stills, trajectories, CARLA videos
│
├── LLM Reasoning & Catalog Pipeline
│   ├── export_llm_trajectories.py         # Formats 15s agent trajectories as JSON prompts
│   ├── query_zero_shot_llm.py             # Interfaces with LLM APIs for zero-shot maneuver tagging
│   ├── create_sample_scenarios.py         # Subfolder scenario organizer
│   ├── render_sample_20_videos.py         # Batch renders sample video clips
│   ├── generate_scenario_catalog.py       # Builds HTML catalog with embedded frame previews
│   ├── build_full_multivehicle_catalog.py # Multi-vehicle catalog builder
│   └── catalog_server.py                  # Flask web server for interactive catalog exploration
│
├── Model Training, Driving & Benchmarking
│   ├── train3.py                          # PyTorch training script for BehavioralCloningSegFormer
│   ├── drive.py                           # Closed-loop driving inference script in CARLA
│   ├── modelrun.py                        # SegFormer model test runner
│   ├── autopilotrun.py                    # CARLA Autopilot Traffic Manager test runner (old avoidance code)
│   └── compare3.py                        # 3-Way benchmark harness (Expert vs ML vs Autopilot; old avoidance code)
│
├── Telemetry & Utilities
│   ├── carla_body_server.py               # Live telemetry streaming server
│   └── town10_autopilot_convoy_logger.py   # Telemetry convoy logger for Town10
│
├── Assets & Metadata
│   ├── raj_vamsi_wafr.pdf                 # Complete paper manuscript (WAFR)
│   ├── drone_background_map.png           # Aerial road map reference image
│   ├── label_scaler.pkl                   # Fitted MinMaxScaler for speed/steering normalization
│   └── boundaries.txt                     # Region boundary polygon definitions
│
├── archive/                               # Archived experimental and draft scripts (gitignored)
└── Scenariosssss/                         # Scenario data & outputs per behaviour (gitignored; see below)
```

---

## ⚙️ Prerequisites & Setup

### 1. Environment Requirements
- **OS**: Ubuntu 20.04 / 22.04 / 24.04 LTS (Linux)
- **GPU**: NVIDIA GPU with CUDA 11.8+ / 12.x support (12GB+ VRAM recommended for training; CARLA replays and camera videos were produced on a T1000 8 GB)
- **CARLA Simulator**: Version **0.9.15**
- **Data (not in git)**: `stabilize_ekf_final2.csv` (EKF-smoothed tracks, primary), `DJI_0003-002.MP4` (drone video), the behaviour CSVs in `Scenariosssss/<Behaviour>/`, and `stabilize_ekf_lv_final.csv` (leader-vehicle columns, used for vehicle following)

After a kernel update, make sure the NVIDIA module for the running kernel is installed (on Ubuntu: `linux-modules-nvidia-<version>-open-$(uname -r)`), otherwise CARLA cannot render.

### 2. Install Python Dependencies

Create a Conda environment and install the required packages:

```bash
conda create -n carla_env python=3.10 -y
conda activate carla_env
pip install -r requirements.txt
```

### 3. Launching CARLA Simulator

Use the included helper script to start the CARLA simulator server:

```bash
# Display mode (GUI window)
./start_carla.sh --display

# Headless / Offscreen mode (for server or background rendering)
./start_carla.sh --server

# No rendering at all (physics and collision sensor only; enough for scenario checks, no GPU needed)
~/CARLA_0.9.15/CarlaUE4.sh -nullrhi -nosound -carla-rpc-port=2000
```

---

## 🚀 Step-by-Step Usage Guide

### Step 1: Real-to-Sim Trajectory Mining & Alignment
Mine trajectory segments from the drone dataset and transform them to CARLA Town10 HD map coordinates:

```bash
python mine_15s_drone_catalog.py
python mine_unique_ego_catalog.py
```

### Step 2: Simulate Scenarios & Record Ego-Centric Dataset
Execute collision-aware PID simulation in CARLA to record monocular RGB camera images and ground-truth control actions (`speed`, `steering`):

```bash
python simulate_and_record_scenarios.py
```
This generates `metadata.csv` containing image paths and corresponding normalized control labels.

### Step 2b: Scenario Validation, Collision Prediction & Repair (current pipeline)
Validate the behaviour scenarios, predict collisions offline, repair small overlaps and replay in CARLA. Full details and expected numbers are in [`TEST_CHECKLIST.md`](TEST_CHECKLIST.md) and [`AGENTS.md`](AGENTS.md).

```bash
# Audit every candidate row of a behaviour CSV (rejects pedestrians / missing tracks / no CARLA path; reports warnings)
python debug/check_scenarios.py --audit-csv "Scenariosssss/Overtaking Left/overtaking.csv"

# Predict car contact, kerb hits and spawn failures for all scenarios (export the drivable map once, CARLA running)
python debug/predict_collisions.py --export-map
python debug/predict_collisions.py            # -> Scenariosssss/collision_prediction.csv

# How many small overlaps (>= -0.5 m) can be repaired with shifts <= 0.7 m
python debug/apply_separation.py --count

# Build scenario folders and replay them headless in CARLA (collisions, gaps, progress)
python debug/make_validated_scenarios.py --input "Scenariosssss/Overtaking Left/overtaking_validated.json" --sample 5
python debug/check_scenarios.py --scenarios "Overtaking Left Validated" --carla --modes pid

# One complete scenario folder with drone video, trajectory images and CARLA bird's-eye videos (GPU server)
python debug/render_scenario_evidence.py --category "Overtaking Left" --ego 565 --other 1342 --start 853 --camera
```

### Step 3: Run Zero-Shot LLM Reasoning Pipeline
Run the full 1-command pipeline to export trajectory prompts, query zero-shot LLMs for maneuver classification, organize scenario subfolders, and render sample drone videos:

```bash
./run_sample_test_pipeline.sh
```

You can also launch the Flask Web Server to browse the interactive scenario catalog:

```bash
python catalog_server.py
```
Then open `http://localhost:5000` in your browser.

### Step 4: Train Vision-Based Behavioral Cloning Model
Train the `BehavioralCloningSegFormer` model using PyTorch and HuggingFace Transformers:

```bash
python train3.py --lr 0.0001 --batch_size 32 --weight_speed 0.5 --epochs 50
```

### Step 5: Closed-Loop Driving & Benchmarking
Run closed-loop evaluation of the trained model in CARLA:

```bash
python drive.py --model_path checkpoints/lr_0.0001_bs_32_wspeed_0.5/best_model.pth
```

To run the complete **3-Way Benchmark Harness** comparing Ground Truth Expert, SegFormer ML Model, and CARLA Autopilot:

```bash
python compare3.py
```

---

## 📊 Evaluation & Results

### Current scenario dataset (October 2026)

Six behaviours, counted on all candidate rows (overlapping windows of the same event are not merged). A scenario is
**usable** when it involves no pedestrian, is at least 80 % inside the calibrated regions, passes the scenario checks,
and has no predicted car contact, kerb/median hit or spawn failure, after repairing small overlaps (≥ −0.5 m) with path
shifts of at most 0.7 m.

| Behaviour | Pass checks | Usable as recorded | Repaired by shift | **Usable** |
| :--- | ---: | ---: | ---: | ---: |
| Cut In | 26 | 22 | 0 | **22** |
| Cut Out | 150 | 118 | 13 | **131** |
| U-turn | 29 | 18 | — | **18** |
| Overtaking Left | 1,810 | 717 | 225 | **942** |
| Overtaking Right | 1,375 | 358 | 203 | **561** |
| Vehicle Following (4.2 s) | 1,822 | 1,026 | 212 | **1,238** |
| **Total** | **5,212** | **2,259** | **653** | **2,912** |

The offline collision prediction caught 20 of 20 problem scenarios in 45 random CARLA replays. Repaired scenarios with
shifts above ~0.5 m replay cleanly only about half the time, so those are checked in CARLA before use.

### Earlier benchmark results (January 2026, WAFR draft)

Kept for reference. These were produced with the earlier pipeline, before the collision fixes; `compare3.py`,
`modelrun.py` and `autopilotrun.py` still contain the old avoidance code, and the numbers have not been reproduced
with the current pipeline.

The evaluation harness (`compare3.py`) tests driving policies across 100 scenario categories and measures:
- **Trajectory Deviation (L2 Error)**
- **Collision Rate**
- **Solid Lane Departures**
- **Maneuver Style Preservation Ratio**

| Driving Policy | L2 Tracking Error (m) ↓ | Collision Rate (%) ↓ | Lane Departures ↓ |
| :--- | :---: | :---: | :---: |
| **Ground Truth Replay (Expert)** | 0.12 | 14.2% | 0 |
| **CARLA Autopilot (Baseline)** | 3.45 | 28.5% | 8.2 |
| **SegFormer (Ours)** | **0.84** | **6.1%** | **0.4** |

---

## 📝 Citation

If you find this work or codebase useful in your research, please cite our paper:

```bibtex
@article{vamsi2026london,
  title={While in London do as the Gwaliorites do: Style Transfer of Trajectories across Traffic Domains in Real-to-Sim},
  author={Dasari Raj Vamsi and Rahul Kala},
  journal={Preprint / Technical Report},
  year={2026},
  institution={Center for Autonomous Systems, ABV-IIITM Gwalior}
}
```

---

## 📄 License
This repository is licensed under the [MIT License](LICENSE).
