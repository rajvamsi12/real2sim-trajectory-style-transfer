#!/usr/bin/env python3
"""
il/select_ckpt_closed_loop.py

Checkpoint choice by closed-loop driving (Raj approved 2026-11-10): open-loop val ADE hid that the LHD model drove
1.25-1.37 x the expert's speed and collided in 42-44 % of the test scenarios. Every ckpt_epNN.pt of a run (train.py
--ckpt-every) drives the same VAL scenarios (time-split val blocks, so the test scenarios stay unseen) next to the expert
via il/closed_loop.py; the pick is the fewest collisions, then mean speed closest to the expert's, then the smallest mean
distance from the recorded path. Writes runs/<run>/cl_selection.csv and copies the pick to runs/<run>/ckpt_cl.pt
(closed-loop driver name for later: '<run>@cl').

  python il/select_ckpt_closed_loop.py --run rhd_ft --limit 50 --port 2000
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IL_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Scenariosssss", "IL")
RUNS = os.path.join(IL_DIR, "runs")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--analyse-only", action="store_true", help="Only read the existing closed-loop csv.")
    args = p.parse_args()
    ckpts = sorted(glob.glob(os.path.join(RUNS, args.run, "ckpt_ep*.pt")))
    if not ckpts:
        sys.exit(f"no ckpt_ep*.pt in runs/{args.run} (train with --ckpt-every)")
    drivers = ["expert"] + [f"{args.run}@{os.path.basename(c)[5:-3]}" for c in ckpts]
    tag = f"sel_{args.run}"
    if not args.analyse_only:
        cmd = [sys.executable, "-u", os.path.join(SCRIPT_DIR, "closed_loop.py"), "--split", "val", "--limit", str(args.limit),
               "--port", str(args.port), "--tag", tag, "--drivers", *drivers]
        r = subprocess.run(cmd)
        if r.returncode:
            sys.exit(r.returncode)
    d = pd.read_csv(os.path.join(IL_DIR, f"closed_loop_{tag}.csv"), dtype={"scenario": str})
    d = d[d.result != "spawn failed"]
    ex = d[d.driver == "expert"].set_index(["scenario", "ego_bp"]).mean_speed.rename("expert_speed")
    d = d.join(ex, on=["scenario", "ego_bp"])
    g = d[d.driver != "expert"].groupby("driver").agg(
        n=("scenario", "size"), collision=("result", lambda r: (r == "collision").mean()),
        off_road=("result", lambda r: (r == "off road").mean()), progress=("progress", "mean"),
        speed_vs_expert=("mean_speed", "sum"), expert=("expert_speed", "sum"), dist_mean=("dist_mean_m", "mean"))
    g["speed_vs_expert"] = g.speed_vs_expert / g.expert
    g = g.drop(columns="expert")
    g["speed_off"] = (g.speed_vs_expert - 1).abs()
    g = g.sort_values(["collision", "speed_off", "dist_mean"])
    e = d[d.driver == "expert"]
    print(f"expert: {len(e)} episodes, collision {(e.result == 'collision').mean():.3f}")
    print(g.round(3).to_string())
    g.round(4).to_csv(os.path.join(RUNS, args.run, "cl_selection.csv"))
    best = g.index[0]
    src = os.path.join(RUNS, args.run, f"ckpt_{best.split('@')[1]}.pt")
    shutil.copy(src, os.path.join(RUNS, args.run, "ckpt_cl.pt"))
    print(f"picked {best} -> runs/{args.run}/ckpt_cl.pt (driver {args.run}@cl)")


if __name__ == "__main__":
    main()
