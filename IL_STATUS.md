# IL_STATUS.md: where the imitation-learning driver stands (handoff)

Last updated **2026-11-16**. Read this first in a new chat, then the newest entries at the bottom of `ML_DECISIONS.md` (every decision and result, dated).
Copies: Mac `~/Downloads/projectsss/` and Dell `~/Desktop/carlascripts/`. Keep both in sync with scp and check md5.
**Precedence:** where docs disagree, `ML_DECISIONS.md` (log) and this file are newest. Next come `AGENTS.md`, then `HANDOVER.md`, `README.md` and `BUG_COLLISIONS.md`, which describe the October 2026 replay pipeline.

## Goal (one line)
An autonomous AI driver in CARLA Town10HD that drives like Gwalior drivers. It is trained by imitation learning on real drone tracks
and sees only its own sensors (camera, LiDAR, own state, route command).

## Current state (2026-11-16, right-hand traffic)
- **Everything is right-hand traffic (RHD) since 2026-11-10.** The old mapping put Gwalior's left-hand traffic against Town10HD's lanes. All LHD outputs are archived in `Scenariosssss/IL_lhd_archive/`. The LHD results further down are history only.
- **Current best driver:** `rhd_scratch@cl` (trained from scratch, checkpoint picked closed loop), plus the **TTC safety layer on every vehicle at 0.7 s / 0.2 m**. The model drives the ego only; all other vehicles follow their recorded paths.
- **Main finding (10 runs, 2026-11-16):** the model is about constant velocity and hasn't learned to brake (it predicts 5.5 % of the expert's slow-downs in the closed-loop probe, 22.8 % in the offline probe). Alone it crashes in 31 % of test scenarios; with the TTC layer 1.4 %, but then it equals const-vel + TTC on behaviour kept (62.0 vs 61.8 %) and drives slower (median speed 0.66 x recorded, headway 1.83 s vs the real 1.07 s).
- **Nothing is running on the Dell** (10-run evaluation finished 2026-11-16 17:02 IST). Model improvement is paused (Raj).
- **Dell needs Raj (16 Nov):** `/` is full (0 free), `/mnt/data` is not mounted since a reboot, and the Dell copy of `ML_DECISIONS.md` is damaged by the full disk; the Mac copy is the good one, re-sync both md files when there is space. Mac: `results_10runs.xlsx` (16 measures; extend with the rest from the Dell).
- **10-run result** (114 test scenarios, 9 drivers, `il/summary_runs.py`, `Scenariosssss/IL/summary_10runs.log`; mean, sd over runs in `ML_DECISIONS.md`). Ego collision % / behaviour kept % of 56 / kept and no ego collision % / progress: expert 2.5 / 100 / 98.6 / 0.94; rhd_lead+ttc 1.1 / 66.6 / 65.0 / 0.62; rhd_scratch+ttc 1.4 / 62.0 / 60.0 / 0.59; const-vel+ttc 1.2 / 61.8 / 59.3 / 0.51; idm 25.6 / 74.6 / 62.9 / 0.80; autopilot (recorded path + speed) 9.1 / 68.8 / 60.0 / 0.54; autopilot_stock 6.3 / 42.7 / 38.0 / 0.62; rhd_scratch 31.0 / 58.4 / 37.0 / 0.67; rhd_lead 36.3 / 58.6 / 38.0 / 0.72. rhd_lead (privileged ground-truth lead input) crashes more without TTC (10 of 10 runs) and is about 5 points ahead with TTC (9 of 10 runs).

## Evaluation tooling and baselines (2026-11-14 to 16; full evaluation done, 10 runs)
- **Behaviour check v2** (`il/behaviour_v2.py`, the only one reported from now on): ego vs other along the recorded ego path; same vehicle ahead at start and end; a pass on the same side; no pass: final gap within 5 m of the recorded gap; not checkable if the recorded ego travels < 5 m. 20-scenario trial: expert 100 %, const-vel+ttc 62.5 %, rhd_scratch 43.8 %, rhd_scratch+ttc 68.8 % (16 checkable).
- **Per-tick logs**: `CL_TRAILS=<dir>` on `il/closed_loop_multi.py` saves every vehicle every tick. `il/metrics_v2.py` (jerk, accelerations, min TTC, TTC-layer share, path deviation, collisions per km, impact speed) and `il/dist_v2.py` (Wasserstein / Jensen-Shannon distance to the expert replay for speed, acceleration, jerk, headway, pass clearance) read them offline.
- **New drivers** in `closed_loop_multi.py`: `autopilot` (Traffic Manager, recorded path as route, recorded mean speed as target; report it as "Autopilot (recorded path + speed)"), `autopilot_stock` (Traffic Manager defaults, only the start pose and speed from the recording; added 2026-11-15) and `idm` (IDM fitted on the train tracks: T 0.39 s, s0 2.02 m, a 1.63, b 1.62 m/s², v0 per vehicle type; `il/idm.py`, `idm_params.json`). Both ran on 3 scenarios without errors.
- **Braking retrain** (`il/run_brake_train.sh`): slow-down samples ×5 (`rhd_brk_w5`) predicts 64 % of the expert's slow-down offline (rhd_scratch 23 %) but plans shorter everywhere (speed-ups −46 %): a slow-down bias, not learned braking. Hiding the speed history (`rhd_brk_hd`) changes nothing. Not run closed loop.
- **Two evaluation runs, means** (run 1 `eval_v2`, run 2 `eval_v2_run2`; `il/compare_runs.py`): ego collision / behaviour kept: expert 2.6 / 100 %; const-vel+ttc 1.3 / 61.6; rhd_scratch 31.6 / 59.8; **rhd_scratch+ttc 2.2 / 60.7**; autopilot 10.1 / 68.8; idm 24.6 / 74.1. One run has several points of noise on behaviour kept (up to 9 of 56 scenarios flip between runs); model+ttc and const-vel+ttc are equal within it.
- **Full evaluation done 2026-11-15** (114 test scenarios, `closed_loop_eval_v2.csv`, `trails_eval_v2/metrics_v2.csv`, `dist_v2.csv`; details in `ML_DECISIONS.md`). Ego collision / behaviour kept (of 56) / ego progress: expert 2.6 % / 100 % / 0.94; const-vel+ttc 1.8 / 60.7 / 0.51; rhd_scratch 29.8 / 58.9 / 0.67; **rhd_scratch+ttc 3.5 / 64.3 / 0.56**; autopilot 9.6 / 67.9 / 0.54; idm 23.7 / 73.2 / 0.80. Model+ttc and const-vel+ttc are not separable; the plain model is closest to the real headway (1.08 vs 1.05 s) but crashes; the TTC layer about doubles the headway (1.94 s).

- **Lead-vehicle input, ground truth (2026-11-15, `rhd_lead`, `--lead`)**: 3 extra state columns (has lead, gap, closing speed from `idm.lead`; privileged). Offline it works where it applies: closing on a lead within 4 s, slow-down predicted 17.5 -> 55.4 %. **Closed loop it does not help** (114 scenarios, one run, `closed_loop_eval_lead.csv`): ego collision rhd_lead 37.7 % (rhd_scratch 31.6 mean), speed 1.29 x recorded (0.97); with TTC 2.6 % collisions, behaviour kept 66.1 %, kept and no collision 62.5 %, progress 0.62 (rhd_scratch+ttc 2.2 / 60.7 / 57.1 / 0.57), inside one-run noise. Cause not diagnosed. `rhd_scratch@cl` + TTC stays the reported driver.

## Open decisions for Raj (next step not chosen yet)
Done since this list was first written: Autopilot baselines (recorded path + speed, and stock), IDM baseline, 10 evaluation runs, ground-truth lead-vehicle input (`rhd_lead`).
Still open, none started:
1. Lead-vehicle input: with the TTC layer about 5 points ahead on kept-and-no-collision (9 of 10 runs), without it more collisions (10 of 10 runs). Open: diagnose the crashes from the saved trails (offline) before any LiDAR-detector version.
2. DAgger-style data (the model drives, and the expert's correction is recorded) to teach recovery and braking.
3. Behaviour check for the 58 slow scenarios (recorded ego travels < 5 m): trial done with a straight reference line; tolerance not chosen (2 m suggested); not yet in `il/behaviour_v2.py`.
4. Which claim the paper makes (sensor-driving model vs traffic model), and ablations.
5. Extend `results_10runs.xlsx` (Mac) with the remaining measures and per-run values from the Dell.

## RHD pipeline (all done, measured)
| Step | Result | Output / log |
|---|---|---|
| Mapping | Corrected matrices in `timemethod.py` (each region takes its partner's CARLA points, 1↔2, 3↔4, 5↔6). 98.2 % of moving steps go with the CARLA lane (old 3.9 %) | `drone pixel to carla coordinates_RHD.txt`, `il/mapping_rhd.py`; LHD block kept as a comment |
| Usable windows | 2,873 (from 5,212 candidates with a status) | `il/run_rhd_rebuild.sh`, `rhd_rebuild.log` |
| 8 s scenarios | 1,620 | `Scenariosssss/IL/scenarios_8s.json` (`scenarios_8s.unfiltered.json` before the filter) |
| Glitch filter (`il/filter_glitches.py`) | 1,341 kept. Ego/other speed jump > 3 m/s per frame, heading flip > 60° while moving, or EKF drift > 1.5 m drops the scenario; a bad neighbour is removed alone (406 removed) | `filter_glitches.csv` |
| CARLA check (cure E, headless) | **1,191 collision-free** (95 collisions, 55 spawn failures) | `replay_cureE.csv` |
| 500-scenario drone vs CARLA video | all 500 ok | Mac `review_video/drone_vs_carla_500_RHD.mp4` |
| Augmentation **v2**, 3 + 4 + 3 per base | 11,120 copies (scene noise, ego offset, ego speed recovery). 7,818 were identical to v1, so their check was reused (`il/aug_reuse.py`) | `scenarios_aug.json`; v1 in `aug_v1/`; `rhd_part4.log` |
| CARLA check of copies | **10,854 collision-free** (225 collisions, 41 spawn failures) | `replay_aug_cureE.csv` |
| Recording | 12,000 scenarios, **433,778 samples**, 119 GB, 19 collided while recording (left out); done 2026-11-12 02:53 | `/mnt/data/raj_il/samples_rhd/`, `il/run_rhd_part4.sh` + `run_rhd_part4b.sh` |
| Feature cache | 30,939 vehicle folders, same LiDAR PCA as the LHD model; done 2026-11-13 06:11 | SSD `Scenariosssss/IL/feats_rhd/`, `il/run_rhd_feats.sh`, `rhd_feats.log` |
| Training | `rhd_ft` (fine-tuned from `runs/lhd_s_K1`) and `rhd_scratch`: setting C, 1 mode, 20 epochs × 65,536 samples, speed-history state, along-track weight 2, checkpoint every 4 epochs | `Scenariosssss/IL/runs/rhd_*`, `il/run_rhd_train.sh`, `rhd_train.log` |
| Checkpoint pick | closed loop on 50 val scenarios (fewest collisions, then speed closest to the expert, then path distance): rhd_ft ep16, **rhd_scratch ep08** → `ckpt_cl.pt` | `il/select_ckpt_closed_loop.py`, `closed_loop_sel_rhd_*.csv` |

Augmentation v2 speed copies: T = 3.5 s; fast copies ×1.2–1.5; egos below 1.5 m/s start 1–2 m/s faster (added) and brake back. The path may never move backwards (the push is halved up to 3 times, otherwise the copy falls back to ego offset).

## RHD results (measured)
**Open loop, test top-mode ADE:** rhd_ft 0.482 m, rhd_scratch 0.481 m, const-vel 0.499 m.

**Closed loop, ego only, others replayed non-reactively** (114 test scenarios; `closed_loop_rhd_test.csv`, `closed_loop_rhd_ttc.csv`):

| Driver | Collision | Speed vs expert |
|---|---|---|
| Expert | 1.8 % | 1.00 |
| Const-vel | 24.6 % | 1.05 |
| Old LHD s_K1 | 37.7 % | 1.26 |
| rhd_ft@cl | 36.0 % | 1.45 |
| rhd_scratch@cl | 28.9 % | 1.18 |
| rhd_scratch@cl + TTC on the ego | 9.6 % (1.8 % hit from ahead, 7.9 % from behind) | 0.65 |

**Probe** (`CL_PROBE=1`: the expert drives, the model only predicts; `rhd_probe.log`): when the expert slows down, the model predicts 5.5 % of the slow-down.

**Professor's setup: model drives the ego, TTC on every vehicle** (`il/closed_loop_multi.py`; the other vehicles follow their recorded paths on the same PID so they can brake; 114 test scenarios, 79 with a checkable behaviour; `closed_loop_ttcall_test.csv`):

| Driver | Any collision | Ego collision | Behaviour kept | Ego speed vs recorded | Ego progress |
|---|---|---|---|---|---|
| Expert | 9.6 % | 1.8 % | 100 % | 1.08 | 0.94 |
| Expert + TTC | 4.4 | 0 | 97.5 | 1.04 | 0.87 |
| Const-vel | 30.7 | 24.6 | 68.4 | 0.94 | 0.64 |
| Const-vel + TTC | 5.3 | 0.9 | 68.4 | 0.55 | 0.50 |
| Model | 36.8 | 30.7 | 63.3 | 1.13 | 0.68 |
| **Model + TTC 0.7/0.2** | **5.3** | **0.9** | **70.9** | 0.74 | 0.58 |
| Model + TTC 1.5/0.4 | 4.4 | 0 | 63.3 | 0.65 | 0.52 |

Behaviour kept, model + TTC vs model alone: Following 77 vs 54 %, Overtaking Left 47 vs 80 %, Overtaking Right 71 vs 100 %.

**TTC tuning** (50 val scenarios, 31 checkable; `closed_loop_ttcall_tune.csv`, `il/run_ttc_all.sh`, `il/run_ttc_tune2.sh`). The two numbers mean: brake fully if a conflict is predicted sooner than B seconds, and count a conflict when the boxes come closer than G metres.

| B s / G m | Any collision | Ego collision | Behaviour kept | Ego speed |
|---|---|---|---|---|
| 1.5 / 0.4 (code default) | 4 % | 0 % | 38.7 % | 0.56 |
| 1.0 / 0.4 | 2 | 0 | 48.4 | 0.56 |
| 1.0 / 0.2 | 2 | 0 | 51.6 | 0.65 |
| **0.7 / 0.2 (pick)** | 4 | 0 | 54.8 | 0.62 |
| 0.5 / 0.1 | 12 | 4 | 54.8 | 0.67 |
| 0.5 / 0.01 | 10 | 4 | 58.1 | 0.69 |
| 0.3 / 0.01 | 10 | 4 | 54.8 | 0.69 |

Looser than 0.7/0.2 brings crashes back without keeping more behaviours. Overtaking Left stays at 56 % from 1.0/0.2 down, so TTC isn't what limits overtakes there.

**Showcase** (5 scenarios per behaviour class: drone video, CARLA-from-drone video + trajectory image, model + TTC 0.7/0.2 video + trajectory image, `results.csv`): Mac `review_video/showcase_ttc_2026-11-13/`.

## Closed-loop tools
- `il/closed_loop.py`: model on the ego, others replayed non-reactively (cure E). Drivers: `expert`, `constvel`, `<run>`, `<run>@epNN`, `<run>@cl`, plus a `+ttc` suffix. Use `--split val|test`; `CL_PROBE=1` runs the probe; `CL_VIDEO=<dir>` writes videos.
- `il/closed_loop_multi.py` (professor's setup, default): model on the ego, other vehicles on PID along their recorded paths. `+ttc[:B:G]` applies TTC to every vehicle. Also reports the behaviour check and ego speed/progress. `--all-vehicles` puts the model on every vehicle (built, set aside); `--any-split` is for showcase folders.
- TTC layer = follower avoidance from `simulate_and_record_scenarios.py` (`find_primary_threat`, box conflict predicted 3 s ahead at constant velocity). A conflict sooner than B s → full brake; a later one → speed capped at the threat's speed, plus a 0.5 m nudge.

## What one training example is (RHD)
One row of `/mnt/data/raj_il/samples_rhd/samples.csv` is one vehicle at one time t, plus the files that row references.
- **Inputs**
  - 3 front-camera JPGs, 256×256, fov 100, at t−0.5 / t−0.25 / t.
  - 3 LiDAR sweeps (32-ch at 1.84 m).
  - Own state.
  - Route command.
  - Vehicle type: one-hot of the blueprint (Micra, Ninja, Microlino, Sprinter, T2).
- **Own state, layout `speedhist`:** speed, the speed change over the last 0.25 / 0.5 / 0.75 / 1.0 s (columns `speed_m025 … speed_m100`), yaw rate, and steer. Steer is the bicycle-model equivalent, because velocity tracking doesn't turn the wheels.
- **Label:** 10 waypoints 0.25 s apart (2.5 s) in the vehicle's own frame. For ego-offset and ego-speed copies it is the recovery path.
- **Training settings:**
  - Split by video time: 20 s blocks, pattern `TTTXTTTTVT`.
  - Behaviour-window samples are weighted ×2; stopped samples ×0.3.
  - Accel is clipped to ±10 m/s².
- **Cached features replace camera and LiDAR in training:**
  - SegFormer-B0: 256×8×8 per frame.
  - PointPillars BEV crop x[−8,32) y[−16,16), 2 m cells, PCA 384 → 48.
- Behaviour is an **output head only**, never an input.
- Look at one sample: `il/show_sample.py`.

## Model (il/model.py)
- Branches: camera conv → 256; LiDAR conv → 512; state + route + vehicle-type MLP → 64.
- Fusion: concat → 512.
- Decoder: 2-layer GRU, outputs offsets on top of the constant-velocity path (`cv_prior`), plus mode scores and a behaviour head.
- **Current runs use 1 mode** (`s_K1` setting). 6 modes tied with 1 mode in the LHD sweep.
- Loss: winner-takes-all L1, with the along-track error weighted ×2 (`--along-w`), plus score CE and 0.1 × behaviour CE.
- Train options: `--state speedhist|accel`, `--along-w`, `--init-from <run>`, `--ckpt-every`. Old LHD runs load unchanged with `--state accel`.

## Machines
| | |
|---|---|
| Dell | `ssh raj@192.168.83.163` (key login), repo `~/Desktop/carlascripts` (branch `main`), Python `~/miniconda3/envs/carla_env/bin/python`, GPU T1000 8 GB |
| CARLA | `~/CARLA_0.9.15/CarlaUE4.sh`; `-nullrhi` for collision checks, `-RenderOffScreen` for sensors/video; ports 2000 / 2010 |
| Big disk | `/mnt/data` (NTFS `/dev/sda4`, 875 GB, 356 GB free on 2026-11-15; **not mounted since the reboot of 2026-11-16**). **Not in fstab:** after a reboot Raj re-mounts it (sudo). Use only our folder, `/mnt/data/raj_il/` (`samples_rhd/` = current, `samples/` = old LHD). Never touch `om`'s data there |
| SSD `/` | **0 free on 2026-11-16** (was 85 GB on 2026-11-15; cause not found, not the evaluation) (features `feats_rhd`; the LHD `feats_full` is in `IL_lhd_archive/`) |
| Network | the Dell has no internet (SSL proxy). Download weights on the Mac and scp them; load with `weights_only` |
| ffmpeg | none on the Dell; convert videos on the Mac (`/opt/homebrew/bin/ffmpeg`, H.264) |

## Gotchas learned the hard way
- Launch long jobs as a detached script: `ssh -f host 'cd repo && setsid nohup bash il/x.sh > /dev/null 2>&1 < /dev/null &'`. An ssh drop kills a CARLA started inside that ssh session.
- Never `pkill -f <pattern>` over ssh: it can kill your own ssh shell. Use `ps -eo pid,args | grep "[x]"` and then kill by PID.
- An interrupted closed-loop run can leave CARLA hung in sync mode. Restart CARLA (the chains' `stop_carla` waits, then force-kills after 30 s).
- The GPU holds ~5 GB while CARLA renders. Don't cache features or train on the GPU at the same time (closed-loop runs hit OOM during caching).
- Terminal tabs in the desktop app die when the app reloads. Restart the Mac monitor (`monitor_rhd.py`) or run it in the Mac's Terminal app.
- Recorded speed is the signed forward speed, so it can be slightly negative while rolling back.
- PointPillars' 4th input is the time lag (0), **not intensity**.
- CSV writers must use a fixed column order (`reindex(columns=COLUMNS)`).
- Physics is not exactly repeatable: a rerun of a clean scenario can collide (~1–3 %).
- **Annotation quality** (measured 2026-11-11, `il/annotation_audit.py`, on the 1,692 LHD base scenarios): 452 had a speed jump > 3 m/s per frame, 91 a heading flip, 849 EKF drift > 1 m, 58 two IDs on one vehicle. This led to the glitch filter. 10 worst cases: Mac `review_video/annotation_problems_2026-11-11/`.

## Rules (Raj)
- Physics on for every vehicle, no kinematic cures.
- Ask before big or long runs and before installing packages. Measure first.
- Short answers, tables; say what is measured vs guessed.
- No sudo by agents. Don't touch `om`'s data. Never ask for a GitHub token. Don't commit Raj's uncommitted edits in `main`.
- Raj deletes Python files himself; other files go to a dated trash folder, never `rm`. Trash folders: Dell `~/trash_YYYY-MM-DD/`, Mac `~/Downloads/projectsss/trash_YYYY-MM-DD/`.

---

## History: left-hand traffic run (October 2026, archived, superseded by RHD)
These outputs are in `Scenariosssss/IL_lhd_archive/` and `/mnt/data/raj_il/samples/`. They used the wrong-side mapping, so the results are only for comparison.

| # | Step | Result (measured) |
|---|---|---|
| 1–2 | 8 s scenarios + cure-E check | 1,692 scenarios (1,333 × 8 s + 359 × 5 s), 1,607 collision-free |
| 4 | Augmentation ×10 (5 scene noise + 5 ego offset) | 14,730 collision-free copies |
| 5 | PointPillars on CARLA LiDAR | `il/pointpillars.py`, finds 30/43 cars |
| 7 | Recording | 16,234 scenarios, 797,340 samples, 217 GiB (~49 h) |
| 8 | Feature cache | 1,628,606 frames, ~29.6 h, 66 ms/frame |
| 9 | Train `full` (6 modes, 30 epochs) | Overfit after epoch 1. Test top-mode ADE 0.740 vs const-vel 0.810 m |
| 9b | Regularisation sweep | Setting C (dropout 0.3, feature dropout 0.2, weight decay 0.05, lr 1e-4) best: test 0.750; it stops the overfit but doesn't lower the floor |
| 9c | Route ablation | The route command doesn't help (test 0.750 with vs 0.718 without) |
| 9d | Overnight sweep (13 runs, 2026-10-29) | Best ~0.72 m: C seed 1, no-route, **1 mode** (`s_K1`) and 3 modes all tie (seed noise ~0.03 m). Worse: score weight 2, state noise, no LiDAR. Conclusion: the data limits it, not the settings |
| 11 | Closed loop (159 test scenarios) | After the speed-handover fix (v2): collisions expert 3.8 %, const-vel 27.7 %, models 42–44 %, driving 1.25–1.37 × expert speed. The probe showed the model predicts almost none of the slow-downs |

Key decisions from that period, still in force (details in `ML_DECISIONS.md`):
- Vehicle type is an input.
- Split by video time.
- Stopped samples weighted ×0.3; behaviour samples ×2.
- Features live on the SSD.
- Constant-velocity prior in the decoder.
- Best checkpoint chosen on val top-mode ADE (RHD: closed loop).
