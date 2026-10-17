#!/usr/bin/env python3
"""
debug/make_validated_scenarios.py

Turns candidates that passed scenario_checks (overtaking_validated.json, written by
check_scenarios.py --audit-csv) into scenario folders and runs Stages 1-3 on them.

  python debug/make_validated_scenarios.py --count 10
  python debug/check_scenarios.py --scenarios "Overtaking Validated" --carla

Needs a CARLA server for Stage 3 (spawn dry-runs); -nullrhi is enough.
"""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, BASE_DIR)

import carla  # noqa: E402
from simulate_and_record_scenarios import AnnotationParser, STABILIZE_EKF_CSV, SIM_FPS, TOWN  # noqa: E402
from debug.batch_process_scenarios import run_stage_1_matrix, run_stage_2_time_method, run_stage_3_spawns  # noqa: E402

SCENARIOS_DIR = os.path.join(BASE_DIR, "Scenariosssss")
DEFAULT_INPUT = os.path.join(SCENARIOS_DIR, "Overtaking Left", "overtaking_validated.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--category", default="Overtaking Validated")
    args = p.parse_args()

    with open(args.input) as f:
        candidates = json.load(f)
    # One scenario per ego vehicle, so the sample is not dominated by a single long track
    picked, seen = [], set()
    for c in candidates:
        if c["ego"] not in seen:
            picked.append(c)
            seen.add(c["ego"])
        if len(picked) == args.count:
            break

    parser = AnnotationParser(STABILIZE_EKF_CSV)
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    world = client.get_world()
    if TOWN not in world.get_map().name:
        world = client.load_world(TOWN)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / SIM_FPS
    world.apply_settings(settings)

    for c in picked:
        name = f"scenario_{c['ego']}_{c['other']}_{c['start']}"
        sc_dir = os.path.join(SCENARIOS_DIR, args.category, name)
        os.makedirs(sc_dir, exist_ok=True)
        ids = [c["ego"], c["other"]]
        with open(os.path.join(sc_dir, f"{name}_summary.txt"), "w") as f:
            f.write(f"Start Frame: {c['start']}\nEnd Frame: {c['end']}\nEgo ID: {c['ego']}\n\nHighlight IDs:\n")
            f.write("".join(f"- {i}\n" for i in ids))
            f.write("\nVehicles:\n" + "".join(f"- {i}\n" for i in ids))
        summary_data = (c["start"], c["end"], ids, ids, c["ego"])
        print(f"\n{name}: real gap {c.get('min_real_gap_m')} m, sim gap {c.get('min_sim_gap_m')} m")
        run_stage_1_matrix(sc_dir, summary_data, parser)
        run_stage_2_time_method(sc_dir, summary_data, parser)
        run_stage_3_spawns(sc_dir, summary_data, parser, dji_spawn_params=None, world=world)
        world.tick()


if __name__ == "__main__":
    main()
