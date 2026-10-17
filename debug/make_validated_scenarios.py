#!/usr/bin/env python3
"""
debug/make_validated_scenarios.py

Turns candidates that passed scenario_checks (*_validated.json, written by
check_scenarios.py --audit-csv) into scenario folders "<Behaviour> Validated/" and runs
Stages 1-3 on them. Up to --count scenarios per behaviour.

  python debug/make_validated_scenarios.py --input "Scenariosssss/Cut In/cut_in_out_validated.json"
  python debug/check_scenarios.py --scenarios "Cut In Validated" --carla

Needs a CARLA server for Stage 3 (spawn dry-runs); -nullrhi is enough.
"""

import argparse
import json
import random
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
    p.add_argument("--count", type=int, default=10, help="Scenarios per behaviour (0 = all, no one-per-ego limit).")
    p.add_argument("--min-inbox", type=float, default=0.8, help="Minimum share of points inside the 6 regions/Center.")
    p.add_argument("--sample", type=int, default=None, help="Pick N random candidates per behaviour (seed 0) instead of the first ones.")
    args = p.parse_args()

    with open(args.input) as f:
        candidates = json.load(f)
    if args.sample:
        candidates = [c for c in candidates if c.get("inbox_ratio", 1.0) >= args.min_inbox]
        by_b = {}
        for c in candidates:
            by_b.setdefault(c.get("behaviour", "Overtaking"), []).append(c)
        rng = random.Random(0)
        candidates = [c for b in sorted(by_b) for c in rng.sample(by_b[b], min(args.sample, len(by_b[b])))]
        args.count = 0
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

    # One scenario per ego vehicle and behaviour, so a sample is not dominated by one long track
    seen, per_behaviour = set(), {}
    for c in candidates:
        b = c.get("behaviour", "Overtaking")
        if c.get("inbox_ratio", 1.0) < args.min_inbox:
            continue
        if args.count and ((b, c["ego"]) in seen or per_behaviour.get(b, 0) >= args.count):
            continue
        seen.add((b, c["ego"]))
        ids = [c["ego"]] + ([c["other"]] if c.get("other") is not None else [])
        name = "scenario_" + "_".join(str(i) for i in ids) + f"_{c['start']}"
        sc_dir = os.path.join(SCENARIOS_DIR, f"{b} Validated", name)
        os.makedirs(sc_dir, exist_ok=True)
        with open(os.path.join(sc_dir, f"{name}_summary.txt"), "w") as f:
            f.write(f"Start Frame: {c['start']}\nEnd Frame: {c['end']}\nEgo ID: {c['ego']}\n\nHighlight IDs:\n")
            f.write("".join(f"- {i}\n" for i in ids))
            f.write("\nVehicles:\n" + "".join(f"- {i}\n" for i in ids))
        summary_data = (c["start"], c["end"], ids, ids, c["ego"])
        print(f"\n{name}: real gap {c.get('min_real_gap_m')} m, sim gap {c.get('min_sim_gap_m')} m, "
              f"warnings: {c.get('warnings') or 'none'}")
        run_stage_1_matrix(sc_dir, summary_data, parser)
        run_stage_2_time_method(sc_dir, summary_data, parser)
        run_stage_3_spawns(sc_dir, summary_data, parser, dji_spawn_params=None, world=world)
        world.tick()
        per_behaviour[b] = per_behaviour.get(b, 0) + 1
    print(f"\nBuilt: {per_behaviour}")


if __name__ == "__main__":
    main()
