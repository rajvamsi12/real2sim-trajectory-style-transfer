# HANDOVER — read this first

The single, current status file. It replaces `PROJECT_MASTER_HANDOVER_SPECIFICATION.md` and the
untracked `HANDOVER_CONTEXT_SUMMARY.md`. Keep it short. Update it at the end of every session
(see "Session log" at the bottom). Last updated: 2026-10-18.

## What this repo does (one paragraph)

It takes vehicle tracks from a drone video of a Gwalior intersection (`stabilize_ekf_final2.csv`,
EKF-smoothed, about 4.27 fps) and maps them into CARLA Town10HD with six per-region affine
matrices (`timemethod.py`; Indian left-hand traffic maps onto CARLA right-hand lanes). It then mines
behaviours (cut-in/out, U-turn, overtaking) and replays them in CARLA: expert replay, PID tracking,
and PID with avoidance. The goal is to render videos and datasets, and to compare Expert vs CARLA
Autopilot vs a learned policy.

## Pipeline and where the code is

| Step | Code | Output (per scenario folder in `Scenariosssss/<Behaviour>/scenario_*`) |
|---|---|---|
| Candidate rows | **not in this repo** (see Open questions) | `overtaking.csv`, `cut_in_out.csv`, `u_turn.csv` |
| Validate candidates | `scenario_checks.py`, `debug/check_scenarios.py --audit-csv` | `overtaking_validated.json` |
| Stage 1 pure matrix | `debug/batch_process_scenarios.py:run_stage_1_matrix` → `convertion.py` | `raw_matrix_trajectories.json` |
| Stage 2 time method | `run_stage_2_time_method` → `timemethod.transform_trajectory_with_analysis` | `time_variant_trajectories.json` |
| Stage 3 spawns | `run_stage_3_spawns` | `spawn_positions_raw.json`, `spawn_positions_retracted.json` |
| Stage 4 replay | `render_carla_simulation_video(mode=expert/pid/pid_avoid)` using `CapturingVehicle` (`simulate_and_record_scenarios.py`) and `SimVehicle` (`scc.py`) | `trajectory_video_*.mp4`, `sim_metrics_<mode>.json` |
| Drone video overlay | `overlay_scenario_bboxes.py` against `DJI_0003-002.MP4` | `real_drone_video.mp4` |
| Drivers | `generate_5_same_direction_micra_scenarios.py` (runs checks first), `generate_scenario_<row>.py`, `debug/make_validated_scenarios.py` | |
| Benchmark / ML | `compare3.py`, `autopilotrun.py`, `modelrun.py`, `train3.py`, `drive.py` | **still on the old avoidance code** |

## Facts you must not re-derive

- **Data source:** `stabilize_ekf_final2.csv`, not `default.json`. Pixel y is negated (`-kf_cy_pixel`) before the region matrices are applied.
- **Real frame rate:** 4.27–4.28 fps (from `timestamp_sec`). A 18-frame window is 4.2 s.
- **Labels, with sizes measured from the EKF boxes (median L × W):**

  | Label | Meaning | Size (L × W) | Blueprint | Blueprint size (L × W) |
  |---|---|---|---|---|
  | 0 | 3-wheeler | 2.4 × 1.8 m | `microlino` | 2.21 × 1.48 |
  | 1 | car | 3.6 × 1.8 m | `nissan.micra` | 3.63 × 1.85 |
  | 2 | 2-wheeler | 1.8 × 1.2 m | `kawasaki.ninja` | 2.04 × 0.80 |
  | 3 | bus | 9.0 × 2.4 m | `volkswagen.t2` | 4.48 × 2.07 — too short, see Open questions |
  | 4 | **pedestrian** | 1.5 × 1.5 m, p90 speed 0.6 m/s | excluded | — |
  | 5 | truck | 6.0 × 2.4 m | `mercedes.sprinter` | 5.92 × 1.99 |

- **Most candidates are not usable.** Of 6,549 unique overtaking candidates, 3,270 include a pedestrian and most of the rest have real boxes that overlap. Passing per behaviour (relaxed rules): Cut In 1/8, Cut Out 8/27, U-turn 29/33, Overtaking Left 73/3,995, Overtaking Right 51/2,554. Details are in `BUG_COLLISIONS.md`.
- **Validated scenario sets:** `Scenariosssss/<Behaviour> Validated/`, 37 in total: Cut In 1, Cut Out 6, U-turn 10, Overtaking Left 10, Overtaking Right 10. Use these, not the old hand-picked folders. The ones without warnings are all clean in CARLA; the per-scenario results are in `BUG_COLLISIONS.md`.
- **Vehicle Following** exists only as an empty folder: there is no CSV and there are no scenarios yet.
- **`cut_in_out.csv`** in `Cut In/` and `Cut Out/` is the same file, as is `overtaking.csv` in `Overtaking Left/` and `Overtaking Right/`. The `cut_type` and `direction` columns split them.
- **Overtaking geometry:** side by side needs `lateral ≥ (W1+W2)/2 + margin`. Slide 16's `< W_avg` is wrong.
- **CARLA does not need the GPU for checks.** `CarlaUE4.sh -nullrhi` runs physics and the collision sensor headless. Videos and camera images need the GPU (`-RenderOffScreen`).
- **CARLA start-up quirk:** a freshly spawned car sits in gear 0 for about 1.2 s and loses any set velocity. Always use `hold_in_gear()`, settle ticks, then `set_initial_speed()`.

## Rules (constraints)

1. Never render or benchmark a scenario that fails `scenario_checks.check_pair` / `check_single` unless you are deliberately showing a failure (`--force`).
   - Rejecting checks: pedestrian; missing frames; real gap < 0.2 m; CARLA gap < 0.3 m; no CARLA path.
   - Warnings only (Raj, 2026-10-18): glitches, trimmed paths, overtaking geometry. There is no on-road check.
   - Look at the warnings before using a scenario: every replay problem so far was in a scenario with warnings.
2. Label 4 is a pedestrian: never spawn it as a vehicle.
3. Spawn yaw comes from the CARLA path (`path_yaw_deg`), not from the EKF `heading_angle`.
4. Replays run in synchronous mode at 20 Hz (the PID gains assume `dt = 0.05`).
5. "Done" means `TEST_CHECKLIST.md` passes, not that a video exists.
6. One logical change per commit; read the diff before committing.

## Current state

- Branch `fix/collisions` (in the worktree `~/Desktop/carlascripts-fix`) holds the collision fixes. **Not merged into `main` and not pushed.**
- `main`'s working tree has your own uncommitted edits (the clearance-radii table, two deleted files) and untracked scripts. The branch doesn't touch them.
- The NVIDIA driver isn't loaded on kernel 7.0.0-38: the module package for that kernel was never installed. Fix: `sudo apt install linux-modules-nvidia-595-open-7.0.0-38-generic linux-modules-nvidia-595-open-generic-hwe-24.04`, then reboot.
- Videos in `Scenariosssss/` were rendered with the old code. They still show the collisions. The validated folders don't have videos yet (that needs the GPU).
- Stale non-Python files were moved to `~/Desktop/carlascripts_trash_2026-10-18/` (nothing was permanently deleted). Two tracked ones were removed on this branch. The list is in "Cleanup" below.

## Open questions (need Raj)

1. Where are the README results (Expert 14.2% / Autopilot 28.5% / SegFormer 6.1% collision rate) from? They aren't reproducible from this repo, and the benchmark scripts use the old, buggy avoidance.
2. Label 3 (9 m bus) is mapped to a 4.48 m VW T2. The MTP deck maps label 5 to the T2; the code maps 5 to the Sprinter. Which is intended?
3. The deck says the real car length is 2.70 m (from `avg_length` in the behaviour CSVs), but the EKF boxes give 3.6 m. Which source is right?
4. Where is the code that produced `overtaking.csv` and `cut_in_out.csv` (the slide 16 rule)? The rule needs fixing at the source too.
5. `PROJECT_REPORT_MASTER_SUMMARY.md` (untracked) lists TC-03 "100% collision-free spawn" and TC-05 avoidance as PASS. Both are contradicted by `BUG_COLLISIONS.md`.

## Cleanup (2026-10-18)

Moved to `~/Desktop/carlascripts_trash_2026-10-18/`, from the main checkout (it's untracked/ignored, so git can't restore it; restore by moving it back):

| File | Reason |
|---|---|
| `test/` (6 mp4, 47 MB) | Old `scenario_new3`/`scenario_new4` test renders from Aug 27; no code references them |
| `cursor_2.6.21_amd64.deb` (176 MB) | Cursor IDE installer, not part of the project |
| `index.html` (174 MB) | Byte-identical copy of `scenario_catalog.html` (`mine_15s_drone_catalog.py` regenerates it) |
| `selected_annotations(old).json` (360 MB) | Superseded annotations (the symlink points to `selected_annotations (1).json`); no references |
| `HANDOVER_CONTEXT_SUMMARY.md` | Superseded by this file |
| `Scenariosssss/Overtaking Left/.~lock.overtaking.csv#` | Stale LibreOffice lock file |
| All `__pycache__/` folders | Bytecode caches for 4 Python versions; regenerated automatically |

Removed on this branch with `git rm` (restorable from git history):
- `PROJECT_MASTER_HANDOVER_SPECIFICATION.md`: superseded by this file.
- `test_map_plot.png`: a one-off test plot that nothing references.

Kept on purpose:
- The old scenario videos: they are the "before" evidence for the bug.
- `final.mp4` / `input.mp4` / `output.mp4`, `output/`, `_scenario_output*`, `scenario_snapshots/` and `validation_pipeline/`: purpose unclear, or referenced by code or `metadata.csv`. Raj to decide.

## Session log

- **2026-10-18 — Claude (Opus 5.5), with Raj.**
  - **Did:** diagnosed the driver issue; traced the collisions (`BUG_COLLISIONS.md`); added scenario validation, time-indexed tracking, launch fix and new avoidance; added `TEST_CHECKLIST.md` and this file.
  - **Left:** port the avoidance to `compare3.py` / `modelrun.py` / `autopilotrun.py`; re-render videos for the validated folders on the GPU; answer the open questions; fix slide 16; find usable Cut In data; create Vehicle Following data.
  - **Watch out:** the old scenario folders are known-bad on purpose. Don't use them as examples.
