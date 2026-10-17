# BUG: vehicles collide in the CARLA overtaking replays

| | |
|---|---|
| **Status** | Fixed on branch `fix/collisions`; checked headless in CARLA (see Verification). Videos not re-rendered yet (needs the GPU driver). |
| **Found** | 2026-10-18, while reviewing the 5 "same-direction dual Micra" overtaking scenarios (0545, 5256, 0532, 19112, 0528) |
| **Investigated / fixed by** | Claude (Opus 5.5) in a Claude Code session with Raj |

## Symptom

- `trajectory_video_without_avoidance.mp4`: both cars collide within about 0.8 s and stay stuck together.
- `trajectory_video_with_avoidance.mp4`: "AVOIDANCE BRAKE ACTIVE" stays on for the whole clip. The cars barely move, so the overtake never happens.
- The with-avoidance video never drew a crash marker, even when one happened (the renderer only drew crashes in the no-avoidance video).

## How it was traced

1. Per-frame comparison for each scenario of the real EKF boxes (`kf_obbox_coord_meter`) and the CARLA blueprint boxes placed on the Stage 2 paths, using a separating-axis box gap.
2. Still frames from the rendered videos, to confirm.
3. A headless CARLA replay (`-nullrhi`), logging reference vs actual position, speed, gear and control every few ticks.

## Root causes (most important first)

### A. The scenarios are physically impossible (data)

The real annotated boxes already overlap, so an exact replay **must** collide:

| Scenario | Min real box gap | Notes |
|---|---|---|
| 5256 | −1.27 m | centres 0.91 m apart |
| 0528 / 0532 | −1.14 m | same vehicle pair, windows 3 frames apart (duplicate) |
| 0545 | −0.37 m | |
| 19112 | −0.62 m | ego speed jumps 4 m/s per frame and its heading flips about 150°: an ID switch, not an overtake |
| Overtaking Left/scenario_0001 | −0.34 m | the "ego" (track 190) is a **pedestrian** (label 4) |

Across all of `overtaking.csv`: 26,629 rows, but only **6,549** unique ones (the same pair is listed in many overlapping windows).
- **3,270** of those include a pedestrian.
- Of the 3,279 without a pedestrian, the minimum real box gap is:
  - below −0.5 m: 1,391
  - −0.5 to 0 m: 1,095
  - 0 to 0.2 m: 325
  - 0.2 m or more: 468

So most "overtakes" are boxes that overlap in the annotations themselves (detection or tracking noise), not real passing manoeuvres.

### B. The overtaking rule selects overlapping pairs (slide 16 of the MTP deck)

The rule accepts a pair when the lateral separation is **smaller** than the vehicle width (`d·|sin φ| < W_avg`). Two cars side by side need at least half of each width plus some clearance. The correct condition is:
`lateral ≥ (W_ego + W_other)/2 + margin`, checked while the cars overlap longitudinally.
This is fixed in `scenario_checks.check_pair`. The slide itself still needs correcting.

### C. Spawn problems

- **Wrong spawn heading.** The yaw came from `convert_real_heading_to_carla_yaw`, which guesses among 4 axis conventions using one noisy EKF heading. In 0528 the ego spawned at −142.7° while its path heads −68.6° (74° off); in 0532 it was 60° off. The cars started pointed across their neighbour.
- **Sideways spawn candidates.** Stage 3 retraction also tried ±2 m and ±4 m lateral shifts. That puts a car beside its own path, and the PID then steers it diagonally across the next lane.
- **Vehicles started from rest.** In 5256 the real ego is doing 7 m/s and the whole window is only 4.2 s.
- **CARLA start-up quirk** (found during this fix). A car spawned 0.6 m up drops onto the road and sits in gear 0 for about 1.2 s; any launch velocity bleeds off because the wheels aren't turning. In 0545 the ego was 5.9 m behind its reference after 3 s.

### D. Controller problems

- **Speeds 12.5% too low.** `calculate_continuous_speeds` divided a 7-interval distance by 8 intervals' time.
- **Replay driven by distance, not time.** A car moved on once it was within 3 m of its next waypoint, so the two cars drifted apart in time and their relative positions stopped matching the real ones.
- **Weak limits.** The CARLA `VehiclePIDController` defaults are `max_brake=0.3` and `max_throttle=0.75`. They can't stop a car within the avoidance horizon or match the real accelerations.

### E. Avoidance bugs (`CapturingVehicle`)

- **The brake never took effect.** `brake=0.9` was applied, then overwritten in the same tick by the PID's `apply_control`.
- **The "nudge" wasn't a nudge.** `get_waypoint(project_to_road=True)` snapped the offset point to the nearest lane centre, possibly the threat's lane.
- **The route was destroyed.** The next 40 waypoints were overwritten with one point at 20% speed and never restored. `blend_back` never ran: the counter was reset to 0, then tested with `0 < x`.
- **Avoidance fired constantly.** Anything within 4 m (centre to centre) was a threat. In these scenarios the centres are always 2–4 m apart, and both cars reacted to each other.

## Fix (branch `fix/collisions`)

| Area | Change | Where |
|---|---|---|
| A, B | `check_pair` rejects: pedestrians; real box gap < 0.2 m; speed jump > 3 m/s per frame; heading flip > 60° while moving; no behind→ahead flip; side-by-side lateral gap < half widths + 0.3 m; path trimmed (< 90% of the window kept, or not starting at the window start); CARLA box gap < 0.3 m. Duplicates of the same pair are removed. | `scenario_checks.py` |
| A | Generators skip failing scenarios (use `--force` to render them anyway) | `generate_5_same_direction_micra_scenarios.py` |
| C | Spawn yaw comes from the CARLA path. Lateral spawn shifts dropped. Fixed a stale-`tid` bug in the Stage 3 cleanup. | `debug/batch_process_scenarios.py`, `simulate_and_record_scenarios.py` |
| C | Cars settle in 1st gear for 1 s, then launch at the real speed (held for 0.3 s while the drivetrain catches up) | `scc.py` (`hold_in_gear`, `set_initial_speed`) |
| A | (Removed 2026-10-18 at Raj's request) Skipping validated scenarios whose CARLA path leaves the driving lanes | — |
| D | Speed window divides by `step/fps` | `simulate_and_record_scenarios.py` |
| D | Time-indexed tracking: the steering target is where the real car is at t + 0.5 s (at least 2.5 m ahead); speed = real speed + 0.8 × along-track lag. The replay uses full throttle; tracking brake is capped at 0.5, and emergency stops use 1.0. | `scc.py` (`set_timed_route`), `CapturingVehicle` |
| E | Avoidance rewritten. Constant-velocity box prediction over 3 s; only the follower (threat ahead) reacts. Conflict < 1.5 s ahead: full brake, PID skipped that tick. Otherwise: follow the threat's speed, plus a 0.5 m offset only if it stays on a driving lane. The route is never modified. | `simulate_and_record_scenarios.py` |
| — | Replay modes `expert` (kinematic, physics off), `pid`, `pid_avoid`. Box-gap, collision and progress metrics saved as `sim_metrics_<mode>.json`. Headless option. Crash marker drawn in every mode. | `debug/batch_process_scenarios.py` |

## What did not work / was ruled out

- **The affine transform.** CARLA centre-to-centre distances match the real ones within about 6%, so the transform itself isn't shrinking gaps.
- **`set_target_velocity` on its own.** It bleeds off within a few ticks (wheels not spinning). Hence the settle-in-gear step plus `enable_constant_velocity` for the first 0.3 s.

## Verification

Headless CARLA 0.9.15 (`-nullrhi`), Town10HD, 20 Hz. Commands are in `TEST_CHECKLIST.md`.

**Old scenarios, with the new code.** They still fail, as expected, since the data overlaps:
- `expert` replay overlaps in 5 of 6.
- `pid` collides in 5 of 6.
- `pid_avoid` avoids the collision in 0001, 0528 and 0545. It cannot in 0532, 5256 and 19112, which already overlap at spawn or from an ID switch.

**Tracking after the launch fix (0545).** Position error is at most 0.28 m and both cars cover 100% of their route; before the fix the error reached 5.9 m and coverage was 61%.

**Audit of every candidate**, by first failing check (each scenario counted once per pair and window):

| Behaviour | Unique candidates | Pass | Main reasons for rejection |
|---|---|---|---|
| Cut In | 8 | **0** | 4 real boxes overlap, 3 leave the 6 regions, 1 too tight for CARLA boxes |
| Cut Out | 27 | 7 | 16 real boxes overlap |
| U-turn | 33 | 25 | 3 pedestrians, 4 tracking glitches |
| Overtaking Left | 3,995 | 25 | 2,237 pedestrian, 1,494 real boxes overlap |
| Overtaking Right | 2,554 | 15 | 1,306 real boxes overlap, 1,033 pedestrian |
| Vehicle Following | — | — | folder is empty: no CSV or scenarios exist yet |

With the strict rules, `make_validated_scenarios.py` also dropped candidates whose CARLA path leaves the driving lanes (that check has since been removed). This happened to 23 of the passing candidates: U-turns across a median that has a gap only in Gwalior, and vehicles on a shoulder that Town10HD doesn't have. Before this check, they hit vegetation or a traffic sign.

**Closed-loop result with the strict rules** (headless CARLA, before the relaxation below): **34 validated scenarios**: Cut Out 4, U-turn 10, Overtaking Left 10, Overtaking Right 10.

| Mode | Vehicle collisions | Static hits | Min box gap | Route covered |
|---|---|---|---|---|
| `expert` | 0 / 34 | 0 / 34 | ≥ 0.20 m | 100% |
| `pid` | 0 / 34 | 0 / 34 | ≥ 0.26 m | ≥ 92% |
| `pid_avoid` | 0 / 34 | 0 / 34 | ≥ 0.38 m | ≥ 86%, except Overtaking Left 873 (11%: the follower stops behind a slow car) |

Overtaking Right was run 3 times in a row with identical results. One earlier run recorded a single `pid_avoid` collision (829) that did not reproduce. It came before the tracking-brake cap (below), so treat it as a possible flake to watch for.

During verification, `max_brake=1.0` made the speed PID swing between full throttle and full brake. Tracking braking is now capped at 0.5; only the emergency stop uses 1.0.

### Relaxed rules (Raj's decision, 2026-10-18)

Only these checks now **reject** a scenario:
- a pedestrian;
- a track missing from more than 20% of the window;
- real boxes closer than 0.2 m;
- CARLA boxes closer than 0.3 m;
- no CARLA path at all.

These are now **warnings only**: speed or heading jumps, trimmed paths, and the overtaking geometry (moves past, behind→ahead, side-by-side clearance). The on-road (driving lane) check was removed.

Audit with the relaxed rules:

| Behaviour | Pass | of which with warnings |
|---|---|---|
| Cut In | 1 / 8 | 1 |
| Cut Out | 8 / 27 | 1 |
| U-turn | 29 / 33 | 4 |
| Overtaking Left | 73 / 3,995 | 48 |
| Overtaking Right | 51 / 2,554 | 36 |

CARLA replay of 37 built scenarios (Cut In 1, Cut Out 6, U-turn 10, Overtaking Left 10, Overtaking Right 10):
- **Clean:** no collision-sensor hits between vehicles in any mode. Every scenario **without** warnings passes all criteria.
- **Problems, all in scenarios that have warnings:**
  - **2 cannot spawn** (U-turn 766, Overtaking Right 75): the path starts outside the 6 regions.
  - **1 static hit** (Overtaking Right 379, off-road), with box overlap of −0.17 m.
  - **Box overlap with no sensor event** (CARLA's collision shape is smaller than its bounding box):
    - Overtaking Left 24: −0.31 m (expert);
    - Overtaking Left 334: −0.12 m (pid);
    - Overtaking Left 95: −0.55 m (expert) and −0.35 m (pid). Here the other vehicle's path is trimmed to 3 of 18 frames, so it stands still and gets driven into.
  - **Low route progress** (glitchy or trimmed tracks): Overtaking Right 201 (25%), Overtaking Right 469 (61%), Overtaking Left 24 (76%).
- **Cut In 135:** only one of its two vehicles spawns (the other starts outside the regions), so it is collision-free trivially.

## Still open

- `compare3.py`, `modelrun.py` and `autopilotrun.py` have their **own copies** of the old avoidance code (same bugs). The 3-way benchmark numbers come from them, so they should switch to `CapturingVehicle` from `simulate_and_record_scenarios.py`.
- **Cut In has no usable scenario** (0 of 8). Vehicle Following has no data yet.
- Only 40 of 6,549 overtaking candidates pass. If that is too few, the main lever is the annotation quality (overlapping boxes), not the simulator.
- Avoidance is conservative: in Overtaking Left 873 the follower stops behind a slow vehicle and covers only 11% of its route.
- The videos for validated scenarios need re-rendering on the GPU server.
