# Test Checklist

A change counts as **done** only when the checks below give the expected output.
"The video rendered" is not a check.

All commands run from the repo root with `~/miniconda3/envs/carla_env/bin/python` (written `python` below).

## 0. CARLA server (only for steps 3–4)

No GPU is needed for the checks. `-nullrhi` turns rendering off, but physics and the collision sensor still work:

```bash
~/CARLA_0.9.15/CarlaUE4.sh -nullrhi -nosound -carla-rpc-port=2000 &
```

Videos and camera images need the real GPU server (`-RenderOffScreen`), so a working NVIDIA driver.

## 1. Compile

```bash
python -m py_compile scc.py simulate_and_record_scenarios.py scenario_checks.py debug/*.py
```

Expected: no output.

## 2. Offline scenario checks (no CARLA, ~5 s per category)

```bash
python debug/check_scenarios.py --scenarios "Overtaking Left"
```

Works for any folder under `Scenariosssss/`. Two-vehicle folders use `check_pair`; one-vehicle folders (U-turn) use `check_single`.
The original hand-picked folders (Cut In, Cut Out, U-turn, Overtaking Left/Right `scenario_0001`, and 0528/0532/0545/5256/19112) mostly **FAIL**. They are kept as known-bad examples (see `BUG_COLLISIONS.md`).
A scenario you intend to render must print **PASS**.

## 3. Candidate audit (no CARLA)

```bash
python debug/check_scenarios.py --audit-csv "Scenariosssss/Cut In/cut_in_out.csv"     # Cut In + Cut Out, ~1 min
python debug/check_scenarios.py --audit-csv "Scenariosssss/U-turn/u_turn.csv"          # ~1 min
python debug/check_scenarios.py --audit-csv "Scenariosssss/Overtaking Left/overtaking.csv"  # Left + Right, ~8 min
```

Each prints `<Behaviour>: PASS n / m` with a breakdown by first failing check, and writes `<csv>_validated.json`.
Last run's numbers are in `BUG_COLLISIONS.md` → Verification.

## 4. Closed-loop collision check (CARLA, ~10 s per scenario and mode)

```bash
python debug/make_validated_scenarios.py --input "Scenariosssss/Cut In/cut_in_out_validated.json" --count 10
python debug/check_scenarios.py --scenarios "Cut Out Validated" --carla
```

Same for `U-turn/u_turn_validated.json` and `Overtaking Left/overtaking_validated.json`.
Folders are created as `<Behaviour> Validated/`.

Every scenario marked `valid yes` must show, in **all three** modes (`expert`, `pid`, `pid_avoid`):

| Metric | Required |
|---|---|
| `col` (collision-sensor events: vehicles **and** static objects) | `0` |
| `gap` (minimum box gap between vehicles, m; `nan` for one-vehicle scenarios) | `> 0` |
| `prog` (route covered, vehicles with a route of 3 m or more) | `>= 85%` for `expert` and `pid` |

The exit code is 1 if a valid scenario collides or fails to spawn.
Per-scenario numbers are also saved as `sim_metrics_<mode>.json` in each folder.

How to read a failure:
- **`expert` overlaps:** the data or the coordinate transform is wrong. The controller is not involved.
- **`expert` clean but `pid` collides:** the controller or the spawn is wrong.
- **`pid_avoid` progress far below `pid`:** avoidance is firing without a real conflict.

## 5. Before committing

- Read the diff (`git diff --stat`, then `git diff`), not the AI's summary of it.
- One logical change per commit, and the commit message says why.
- If behaviour changed, update `HANDOVER.md` (and `BUG_COLLISIONS.md` if it concerns collisions).
