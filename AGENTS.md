# AGENTS.md — context for AI agents working on carlascripts

Read this first, then `HANDOVER.md` (project facts), `BUG_COLLISIONS.md` (the collision investigation)
and `TEST_CHECKLIST.md` (how to verify). Where this file and `HANDOVER.md` disagree, **this file is newer**
(rules were relaxed after HANDOVER.md was written). Written 2026-10-20 from a Claude Code session with Raj.

## 1. Setup and access

| Item | Value |
|---|---|
| Machine | Dell, Ubuntu 24.04, `ssh raj@192.168.83.163` (key login), GPU NVIDIA T1000 8 GB |
| Repo | `~/Desktop/carlascripts`, branch `main`, remote `origin` = github.com/rajvamsi12/real2sim-trajectory-style-transfer (**not pushed**; local main is ahead) |
| Python | `~/miniconda3/envs/carla_env/bin/python` (3.10) |
| CARLA | `~/CARLA_0.9.15/CarlaUE4.sh`, port 2000, Town10HD |
| Headless (no camera, checks only) | `CarlaUE4.sh -nullrhi -nosound -carla-rpc-port=2000` |
| With cameras/videos (needs GPU driver) | `CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic -carla-rpc-port=2000` |
| Data | `stabilize_ekf_final2.csv` (1.2 GB, primary). Behaviour CSVs in `Scenariosssss/<Behaviour>/` (`overtaking.csv` and `cut_in_out.csv` are each shared by two folders; `direction` / `cut_type` split them) |
| Raj's Mac working folder | `/Users/rajvamsi/Downloads/L4DC_2027_Neural_ICBF` (a different project; generated scenario folders were copied there) |

GPU driver: after kernel 7.0.0-38 the NVIDIA module was missing. Fixed by Raj with
`sudo apt install linux-modules-nvidia-595-open-7.0.0-38-generic linux-modules-nvidia-595-open-generic-hwe-24.04` + reboot.
Agents cannot use sudo.

## 2. Working preferences (Raj)

- Short, plain explanations; tables over prose; say clearly what is measured vs. guessed.
- Ask when a request is ambiguous; propose a method before big runs; don't build folders or run hours-long jobs unasked.
- Raj handles deleting/cleaning **Python** files himself; agents may move stale non-Python files to a dated trash folder (never `rm` data).
- `main` on the Dell has Raj's **own uncommitted edits** (`simulate_and_record_scenarios.py` clearance-radii table,
  `generate_5_first_behavior_scenarios.py`, two deleted files). Never commit them; put new code in other files
  (e.g. `debug/batch_process_scenarios.py`) or stage only your hunks.
- One logical change per commit; commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## 3. Current rules (decided in this session)

**Reject** a scenario (`scenario_checks.check_pair` / `check_single`): a vehicle is a pedestrian (label 4); a track
is missing; vehicles present in < 80 % of the window; no CARLA path (fewer than 2 points in the 6 regions/Center).

**Warnings only** (scenario stays valid): speed jump > 3 m/s per frame; heading flip > 60° per frame while moving;
path trimmed (< 90 % of window in the regions); real drone boxes < 0.2 m apart; CARLA bodies < 0.3 m apart;
overtaking geometry (ego doesn't move past / no behind→ahead / side-by-side gap too small / no side-by-side frame).
No on-road check in scenario building (removed at Raj's request).

**Filters:** at least **80 % in-box** (both vehicles' points inside Region_1..6 or Center, the MTP "% in-box").
Duplicate windows are **not** merged in Raj's tables (all CSV rows counted).

**Collision-free decision (offline, `debug/predict_collisions.py`):**
- car-car: CARLA vehicle shapes on the converted paths; `min_carla_gap < 0` → contact
- kerb/median: shape off the Town10HD drivable raster (`Scenariosssss/town10hd_drivable.npz`) in ≥ 5 vehicle-frames
- spawn: first position outside the 6 regions/Center → cannot be spawned

**Repair small overlaps (`debug/apply_separation.py`) — FIXED SETTINGS:** overlap limit **−0.5 m**
(deeper = reject; mostly tracking errors), maximum shift **0.7 m**, target gap after shift 0.2 m, shifted
paths must stay on the drivable area. Sideways shift (half each, perpendicular to own heading) for side-by-side
contact, or the rear vehicle moved back along its path for front-to-back contact; the smaller one wins.

**Spawn yaw (`choose_spawn_yaw` in `debug/batch_process_scenarios.py`):** spawn point inside the 6 regions →
nearest CARLA driving lane's direction (even if the point is on a shoulder); otherwise the lane direction if on a
non-junction driving lane; flipped 180° if > 90° from the travel direction. Else (junction/off-road): EKF heading
× region matrix if speed ≥ 1 m/s, otherwise path direction once moved 1 m. `heading_angle` = `atan2(-vy, vx)` in
image pixels (y-up, same as the region-matrix input `(u, -v)`).

## 4. Key findings (with evidence)

- Original collisions came from **bad scenario data**, not CARLA: overlapping drone boxes, ghost tracks/ID errors
  (e.g. track 149; 565_1342 has two car IDs crossing over at most one real vehicle at 0.55 m), pedestrians labelled as
  "overtakers" (half of `overtaking.csv`), plus simulation bugs (fixed): spawn yaw, start from rest, CARLA gear-0
  start, speed window, distance-indexed replay, broken avoidance.
- Drone boxes are larger than vehicles (2-wheeler box 1.2 m wide vs ~0.8 m real), so box overlap ≠ vehicle overlap.
- CARLA Micra (1.85 m wide) is wider than Gwalior cars; matters only in very tight car-car passes (1514_1332).
- One arm (CARLA x ≈ −52) puts vehicles on lanes of the opposite direction / onto shoulders — likely a lateral
  offset in that region's matrix. Unresolved.
- U-turns through the median gap in Gwalior hit Town10HD's planted median (10 of 29).
- `stabilize_ekf_lv_final.csv` (on Raj's Mac, Downloads): same 2,955 tracks/frames/labels as final2, positions
  within ~1 px, plus leader columns (`ID_LV`, `Space_Gap`, `Space_Headway`, `Speed_Diff`, `direction`); 5,441
  duplicate rows = vehicles with two leaders. Behaviour CSVs were built from final2, not this file.
- **Vehicle Following** (no CSV) is mined from the lv file: exactly one leader, `direction` = 1, no pedestrians,
  same leader for ≥ 18 frames (4.2 s), not overlapping an overtake/cut/U-turn of either vehicle.
  Candidates: `Scenariosssss/Vehicle Following/vehicle_following_candidates_relaxed.json` (`n` = window length).

## 5. Numbers (all CSV rows, pass checks, ≥ 80 % in-box)

| Behaviour | Pass checks | Usable as recorded | Rescued, shift ≤ 0.5 m | Rescued, shift ≤ 0.7 m (cumulative) |
|---|---|---|---|---|
| Cut In | 26 | 22 | 0 | 0 |
| Cut Out | 150 | 118 | 13 | 13 |
| U-turn | 29 | 18 | — | — |
| Overtaking Left | 1,810 | 717 | 128 | 225 |
| Overtaking Right | 1,375 | 358 | 131 | 203 |
| Vehicle Following | 1,822 | 1,026 | 127 | 212 |
| **Total** | **5,212** | **2,259** | **399** | **653** |

So with the fixed settings (−0.5 m, 0.7 m): **≈ 2,900 usable** (2,259 + ~653; the exact-search count is ~720 rescued).
CARLA validation: offline prediction caught 20/20 problems on 45 random scenarios (8/25 clean wrongly flagged at
0.15 m threshold; 4/25 at 0 m). Shifted scenarios replayed clean ~75 % for shifts ≤ 0.45 m, ~50 % for 0.5–0.7 m —
**replay shifted scenarios with 0.5–0.7 m shifts in CARLA and keep only clean ones.**

## 6. Tools added in this session (all under `debug/` unless noted)

| Script | Purpose |
|---|---|
| `scenario_checks.py` (root) | Offline checks, `TrackStore`, box geometry, `inbox_ratio` |
| `check_scenarios.py` | `--scenarios <folder>` checks; `--audit-csv` per-behaviour audit; `--carla --modes pid` headless replay with collision/gap/progress summary; `--sample/--seed` |
| `make_validated_scenarios.py` | Build `<Behaviour> Validated/` folders from a candidates JSON (`--sample`, `--seed`, `--min-inbox`, `--count 0` = all) |
| `batch_process_scenarios.py` | Stages 1–4; `render_carla_simulation_video(mode=expert|pid|pid_avoid, render, camera_z)`; `choose_spawn_yaw` |
| `render_scenario_evidence.py` | One full scenario folder: drone video, stills, trajectory images, CARLA bird's-eye videos (`--camera`) |
| `predict_collisions.py` | `--export-map` (once, CARLA), predict all → `Scenariosssss/collision_prediction.csv`, `--validate` |
| `apply_separation.py` | `--count` fixable overlaps; `--apply <dirs>` shift built folders (keeps `time_variant_trajectories_original.json`, writes `separation.json`, redoes Stage 3) |

Replay modes: `expert` = recorded positions, physics off (data check); `pid` = physics + time-indexed PID
(the realistic one); `pid_avoid` = plus follower avoidance.

## 7. Gotchas

- `pgrep -f`/`pkill -f` with a pattern that also appears in your own ssh command line kills your session.
  Kill CARLA with `kill $(ps -eo pid,args | grep "[C]arlaUE4-Linux-Shipping" | awk '{print $1}')`.
- A CARLA server left running can hang (port 2000 open, GPU 0 %) and new servers then segfault on bind. Kill it first.
- Destroying a camera and then destroying actors in the same frame segfaulted CARLA; tick the world in between
  (already done in the code).
- Fresh CARLA vehicles sit in gear 0 for ~1.2 s; always `hold_in_gear()` → settle ticks → `set_initial_speed()`.
- The 1.2 GB CSV makes any full audit take ~8 min; `check_pair` ≈ 0.05–0.1 s per scenario.
- `Scenariosssss/` is gitignored data; generated folders and JSON/CSV results live there, not in git.

## 8. Generated outputs

On the Dell (`Scenariosssss/`): `collision_prediction.csv`, `separation_fixable.json`, `town10hd_drivable.npz`,
`Separation Before After/` (10 scenarios, before/after-shift CARLA videos), `* Validated/` test folders.
On Raj's Mac (`~/Downloads/L4DC_2027_Neural_ICBF/`): `scenario_565_1342_853/`, `overtaking_random5/`,
`overtaking_cars5/`, `separation_before_after/`, `collision_prediction.csv`, `deep_overlap_histogram.png`,
`shift_rescue_curve.png`, `overlap_shift_needed.csv`.
Stale non-Python files were moved to `~/Desktop/carlascripts_trash_2026-10-18/` (see HANDOVER.md → Cleanup).

## 9. Open items

- Port the new avoidance to `compare3.py`, `modelrun.py`, `autopilotrun.py` (still old, buggy copies; README benchmark numbers come from them).
- Check the region matrix of the CARLA x ≈ −52 arm (lane-direction flips, shoulder placement).
- Confirm the meaning of `direction` = 0 in the lv file (41 % of rows).
- Answer HANDOVER.md open questions (README results source, label 3 bus → 4.5 m van, real car length 2.70 vs 3.6 m, generator of the behaviour CSVs, PROJECT_REPORT test claims).
- Correct slide 16 (overtaking rule `lateral < W_avg` selects overlapping pairs).
- Verify the 0.5–0.7 m shifted scenarios in CARLA before using them.
