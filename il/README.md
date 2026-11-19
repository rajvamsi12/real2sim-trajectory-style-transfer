# Imitation-learning driver (`il/`)

An imitation-learning driver for CARLA Town10HD (right-hand traffic), trained on real drone tracks from Gwalior.
It drives the ego vehicle from its own sensors: camera, LiDAR, its own state and a route command.
The full decision log is in `ML_DECISIONS.md`; the current status is in `IL_STATUS.md`.

## Pipeline

| Step | Script |
|---|---|
| 8 s scenarios from the drone tracks, glitch filter, U-turn commands | `build_scenarios_8s.py`, `filter_glitches.py`, `fix_uturn_commands.py`, `mapping_rhd.py` |
| Augmentation (scene noise, ego offset, ego speed recovery) | `augment_scenarios.py`, `aug_reuse.py` |
| Replay in CARLA and record samples (camera, LiDAR, state, future path) | `replay_scenarios.py`, `record_samples.py` |
| Frozen sensor features: SegFormer-B0 camera encoder, PointPillars BEV LiDAR + PCA | `cache_features.py`, `pointpillars.py` |
| Model and training | `model.py`, `train.py` |
| Checkpoint choice in closed loop | `select_ckpt_closed_loop.py` |
| Closed-loop evaluation (9 drivers, TTC safety layer, per-tick logs) | `closed_loop_multi.py`, `closed_loop.py` |
| Metrics, behaviour check, distribution distances, multi-run summary | `metrics_v2.py`, `behaviour_v2.py`, `dist_v2.py`, `compare_runs.py`, `summary_runs.py` |
| Baselines | `idm.py` (IDM fitted on the training tracks), Traffic Manager Autopilot in `closed_loop_multi.py` |
| Lead-vehicle input (ground truth, privileged) | `lead_input.py` |
| Offline braking probe | `probe_offline.py` |

`run_*.sh` are the chain scripts that ran each step on the lab machine.

## Weights (not in the repository)

`il/weights/` needs:
- `pointpillars_nus_secfpn_f19d00a3.pth`: mmdetection3d PointPillars, `hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d` (nuScenes).
- `segformer_b0_cityscapes/`: SegFormer-B0 fine-tuned on Cityscapes, Hugging Face Transformers format.
- `lidar_pca.npz`: our fixed PCA of the LiDAR features (included; refit with `cache_features.py --fit-pca`).

Data, features, trails and checkpoints are not in the repository (about 70 GB).

## Result: 10 closed-loop runs

114 test scenarios × 9 drivers × 10 runs. The model drives the ego; all other vehicles follow their recorded paths
with physics on. TTC = time-to-collision safety layer on every vehicle (0.7 s, 0.2 m). Mean over the 10 runs.

| Driver | Ego collision % | Behaviour kept % (of 56) | Kept and no ego collision % | Progress | Median headway (s) |
|---|---|---|---|---|---|
| Expert (recording replayed) | 2.5 | 100 | 98.6 | 0.94 | 1.07 |
| `rhd_lead` + TTC | 1.1 | 66.6 | 65.0 | 0.62 | 1.85 |
| IDM | 25.6 | 74.6 | 62.9 | 0.80 | 1.46 |
| `rhd_scratch` + TTC | 1.4 | 62.0 | 60.0 | 0.59 | 1.83 |
| Autopilot (recorded path + speed) | 9.1 | 68.8 | 60.0 | 0.54 | 1.91 |
| Constant velocity + TTC | 1.2 | 61.8 | 59.3 | 0.51 | 1.62 |
| `rhd_lead` | 36.3 | 58.6 | 38.0 | 0.72 | 1.11 |
| Autopilot (stock) | 6.3 | 42.7 | 38.0 | 0.62 | 1.44 |
| `rhd_scratch` | 31.0 | 58.4 | 37.0 | 0.67 | 1.11 |

`rhd_scratch` is the main model. `rhd_lead` adds the ground-truth gap and closing speed to the vehicle ahead (privileged).
Full tables, spread between runs and paired comparisons: `results/il_rhd_10runs/`.
