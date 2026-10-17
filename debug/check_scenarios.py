#!/usr/bin/env python3
"""
debug/check_scenarios.py  --  the numeric test checklist (see TEST_CHECKLIST.md).

Offline (no CARLA needed):
  python debug/check_scenarios.py --scenarios "Overtaking Left"
      Checks every scenario folder (its *_summary.txt) with scenario_checks.check_pair.
  python debug/check_scenarios.py --audit-csv "Scenariosssss/Overtaking Left/overtaking.csv"
      Runs the same checks over every candidate row (overtaking.csv, cut_in_out.csv or u_turn.csv)
      and prints, per behaviour, how many survive and why the rest fail.

In CARLA (server must be running; -nullrhi is enough, no GPU rendering needed):
  python debug/check_scenarios.py --scenarios "Overtaking Left" --carla
      Replays each scenario headless in 3 modes (expert / pid / pid_avoid) and reports
      col=<vehicle collisions>/<static hits>, minimum box gap and route progress.

Exit code is 1 if any checked scenario fails, so it can gate a pipeline run.
"""

import argparse
import collections
import glob
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, BASE_DIR)

import scenario_checks as sc  # noqa: E402

SCENARIOS_DIR = os.path.join(BASE_DIR, "Scenariosssss")
LABEL_BLUEPRINT_MAP = {0: "vehicle.micro.microlino", 1: "vehicle.nissan.micra", 2: "vehicle.kawasaki.ninja",
                       3: "vehicle.volkswagen.t2", 5: "vehicle.mercedes.sprinter"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenarios", help="Category or category/scenario folder under Scenariosssss, or 'all'.")
    p.add_argument("--audit-csv", help="overtaking.csv to audit row by row.")
    p.add_argument("--limit", type=int, default=None, help="Audit only the first N rows.")
    p.add_argument("--csv", default=sc.DEFAULT_CSV, help="EKF dataset CSV.")
    p.add_argument("--carla", action="store_true", help="Also replay headless in CARLA and count collisions.")
    p.add_argument("--modes", default="expert,pid,pid_avoid", help="CARLA modes to run.")
    return p.parse_args()


def find_summaries(target):
    root = SCENARIOS_DIR if target in (None, "all") else os.path.join(SCENARIOS_DIR, target)
    return sorted(glob.glob(os.path.join(root, "**", "*_summary.txt"), recursive=True))


def bp_for(store, tid):
    return LABEL_BLUEPRINT_MAP.get(store.label(tid), "vehicle.nissan.micra")


def check_folders(args):
    from overlay_scenario_bboxes import parse_summary_file

    summaries = find_summaries(args.scenarios)
    parsed = [(s, parse_summary_file(s)) for s in summaries]
    ids = {t for _, (_, _, vids, _, _) in parsed for t in vids}
    print(f"Loading {len(ids)} tracks from {os.path.basename(args.csv)} ...")
    store = sc.TrackStore(args.csv, track_ids=ids)

    results = []
    for path, (start, end, vids, _, ego) in parsed:
        name = os.path.relpath(os.path.dirname(path), SCENARIOS_DIR)
        others = [v for v in vids if v != ego]
        if ego is None:
            print(f"SKIP {name}: summary has no ego")
            continue
        if not others:
            ok, m, reasons = sc.check_single(store, ego, start, end)
            results.append((name, ok, m, reasons))
            print(f"{'PASS' if ok else 'FAIL'} {name}  ego {ego} (single vehicle)")
            for r in reasons:
                print(f"       - {r}")
            continue
        overtake = "overtak" in name.lower()
        for other in others:
            ok, m, reasons = sc.check_pair(store, ego, other, start, end,
                                           bp_for(store, ego), bp_for(store, other), overtake=overtake)
            results.append((name, ok, m, reasons))
            print(f"{'PASS' if ok else 'FAIL'} {name}  ego {ego} vs {other}  "
                  f"real gap {m.get('min_real_gap_m')}  sim gap {m.get('min_sim_gap_m')}")
            for r in reasons:
                print(f"       - {r}")
    return store, results


def behaviour_of(row):
    """Scenariosssss folder name for one CSV row (overtaking.csv / cut_in_out.csv / u_turn.csv)."""
    if hasattr(row, "cut_type"):
        return "Cut In" if str(row.cut_type).strip() == "in" else "Cut Out"
    if hasattr(row, "other_id"):
        return "Overtaking Left" if str(row.direction).strip() == "left" else "Overtaking Right"
    return "U-turn"


def audit_csv(args):
    import pandas as pd

    df = pd.read_csv(args.audit_csv)
    if args.limit:
        df = df.head(args.limit)
    frame = lambda v: int(str(v).split("_")[-1])
    rows = [{"ego": int(r.trackID), "other": int(r.other_id) if hasattr(r, "other_id") else None,
             "start": frame(r.start_frame), "end": frame(r.end_frame), "behaviour": behaviour_of(r)}
            for r in df.itertuples()]
    unique = sc.dedupe_windows(rows)
    ids = {r["ego"] for r in unique} | {r["other"] for r in unique if r["other"] is not None}
    print(f"{len(rows)} rows, {len(unique)} after removing overlapping duplicates of the same pair. "
          f"Loading {len(ids)} tracks ...")
    store = sc.TrackStore(args.csv, track_ids=ids)

    first_reason = {}
    totals = collections.Counter(r["behaviour"] for r in unique)
    passed = []
    for i, r in enumerate(unique, 1):
        if r["other"] is None:
            ok, m, reasons = sc.check_single(store, r["ego"], r["start"], r["end"])
        elif r["ego"] not in store.tracks or r["other"] not in store.tracks:
            ok, m, reasons = False, {}, ["track missing from CSV"]
        else:
            ok, m, reasons = sc.check_pair(store, r["ego"], r["other"], r["start"], r["end"],
                                           bp_for(store, r["ego"]), bp_for(store, r["other"]),
                                           overtake=r["behaviour"].startswith("Overtaking"))
        if ok:
            passed.append(dict(m, behaviour=r["behaviour"]))
        else:
            first_reason.setdefault(r["behaviour"], collections.Counter())[_reason_key(reasons[0])] += 1
        if i % 1000 == 0:
            print(f"  {i}/{len(unique)} checked, {len(passed)} pass so far")

    for behaviour in sorted(totals):
        n_pass = sum(1 for p in passed if p["behaviour"] == behaviour)
        print(f"\n{behaviour}: PASS {n_pass} / {totals[behaviour]} unique candidates")
        for k, v in first_reason.get(behaviour, collections.Counter()).most_common():
            print(f"  {v:6d}  {k}")
    print(f"\nTotal: PASS {len(passed)} / {len(unique)} unique candidates ({len(rows)} rows)")
    out = os.path.splitext(args.audit_csv)[0] + "_validated.json"
    with open(out, "w") as f:
        json.dump(passed, f, indent=1)
    print(f"Passing candidates written to {out}")
    return passed, unique


def _reason_key(reason):
    for key in ("pedestrian", "real boxes", "speed jumps", "heading flips", "behind to ahead",
                "side-by-side lateral", "no side-by-side", "CARLA boxes", "trimmed outside", "shared frames", "frames in window", "does not move past"):
        if key in reason:
            return key
    return reason


def run_carla(args, store, results):
    import carla
    from simulate_and_record_scenarios import CARLA_HOST, CARLA_PORT, TOWN, SIM_FPS, AnnotationParser
    from debug.batch_process_scenarios import render_carla_simulation_video
    from overlay_scenario_bboxes import parse_summary_file

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(30.0)
    world = client.get_world()
    if TOWN not in world.get_map().name:
        world = client.load_world(TOWN)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / SIM_FPS
    world.apply_settings(settings)

    parser = AnnotationParser(args.csv)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    failed = False
    print(f"\n{'scenario':40s} {'valid':5s} " + " ".join(f"{m:>22s}" for m in modes))
    for name in sorted({r[0] for r in results}):
        valid = all(ok for n, ok, _, _ in results if n == name)
        sc_dir = os.path.join(SCENARIOS_DIR, name)
        summary = glob.glob(os.path.join(sc_dir, "*_summary.txt"))[0]
        cells = []
        for mode in modes:
            metrics = render_carla_simulation_video(client, world, sc_dir, summary, parse_summary_file(summary),
                                                    parser, None, enable_avoidance=(mode == "pid_avoid"),
                                                    mode=mode, render=False)
            if not metrics:
                cells.append("spawn failed")
                failed = True
                continue
            cells.append(f"col={metrics['collision_events']}/{metrics['static_collisions']} "
                         f"gap={metrics['min_box_gap_m']:+.2f} prog={metrics['min_progress']:.0%}")
            if valid and (metrics["collision_events"] or metrics["static_collisions"]):
                failed = True
        print(f"{name:40s} {'yes' if valid else 'no':5s} " + " ".join(f"{c:>22s}" for c in cells))
    return failed


def main():
    args = parse_args()
    if not args.scenarios and not args.audit_csv:
        print("Give --scenarios and/or --audit-csv")
        return 2
    failed = False
    if args.audit_csv:
        audit_csv(args)
    if args.scenarios:
        store, results = check_folders(args)
        failed |= any(not ok for _, ok, _, _ in results)
        if args.carla:
            failed |= run_carla(args, store, results)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
