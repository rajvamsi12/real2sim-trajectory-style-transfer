"""Lead-vehicle input for the model (Raj, 2026-11-15 "go ahead with ground truth first"): bumper gap to the nearest vehicle
ahead in the vehicle's corridor and the closing speed, the two numbers IDM uses (idm.lead, same function here, in training
and in the closed loop). Ground truth = privileged: in training from the scenario's reference paths at the sample's time,
in the closed loop from the simulator's vehicle states.

Per sample -> <samples dir>/lead.csv: scenario, track, k, lead_gap (m, inf = no lead within 40 m), lead_dv (m/s, + = closing).
Position / heading / speed of every vehicle at the sample time t come from its timed route (the path the replay tracks,
position error ~0.1 m); the sample's own speed is the recorded one. Heading = path direction, held while the path is still.

  python il/lead_input.py --samples /mnt/data/raj_il/samples_rhd
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import idm as IDM  # noqa: E402

IL_DIR = IDM.IL_DIR
REAL_FPS = 4.27
MOVE_M = 0.15     # a path step shorter than this gives no heading


def route(v, start):
    """times, xy, speeds (replay_scenarios.timed_route) and the heading at every route point (nan = never moves)."""
    t = (np.asarray(v["frames"], float) - start) / REAL_FPS
    xy = np.asarray(v["xy"], float)
    n = len(xy)
    sp, yaw = np.zeros(n), np.full(n, np.nan)
    for i in range(n):
        lo, hi = max(0, i - 2), min(n - 1, i + 2)
        sp[i] = np.linalg.norm(xy[hi] - xy[lo]) / (t[hi] - t[lo]) if t[hi] > t[lo] else 0.0
        d = xy[min(n - 1, i + 1)] - xy[max(0, i - 1)]
        if np.linalg.norm(d) >= MOVE_M:
            yaw[i] = np.arctan2(d[1], d[0])
        elif i:
            yaw[i] = yaw[i - 1]
    good = np.flatnonzero(~np.isnan(yaw))
    if len(good):
        yaw[:good[0]] = yaw[good[0]]
    return t, xy, sp, yaw


def scene_at(routes, t):
    x = np.array([np.interp(t, r[0], r[1][:, 0]) for r in routes])
    y = np.array([np.interp(t, r[0], r[1][:, 1]) for r in routes])
    sp = np.array([np.interp(t, r[0], r[2]) for r in routes])
    yaw = np.array([r[3][int(np.abs(r[0] - t).argmin())] for r in routes])
    return x, y, yaw, sp


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--samples", required=True)
    args = a.parse_args()
    t0 = time.time()
    scen = {s["id"]: s for s in json.load(open(os.path.join(IL_DIR, "scenarios_8s.json")))}
    aug = os.path.join(IL_DIR, "scenarios_aug.json")
    if os.path.exists(aug):
        scen.update({s["id"]: s for s in json.load(open(aug))})
    d = pd.read_csv(os.path.join(args.samples, "samples.csv"), dtype={"scenario": str}, usecols=["scenario", "track", "k", "t", "speed"])
    missing = set(d.scenario) - set(scen)
    if missing:
        sys.exit(f"{len(missing)} sample scenarios are not in the scenario files, e.g. {sorted(missing)[:5]}")
    gap, dv, no_heading = np.full(len(d), np.inf), np.zeros(len(d)), 0
    for sid, g in d.groupby("scenario", sort=False):
        s = scen[sid]
        V = s["vehicles"]
        routes = [route(v, s["start"]) for v in V]
        ext = np.array([[v["footprint"][0] / 2, v["footprint"][1] / 2] for v in V])
        at = {v["track"]: i for i, v in enumerate(V)}
        for t, gt in g.groupby("t", sort=False):
            x, y, yaw, sp = scene_at(routes, t)
            bad = np.isnan(yaw)
            if bad.any():       # a vehicle that never moves: heading of the nearest vehicle that has one
                for i in np.flatnonzero(bad):
                    ok = np.flatnonzero(~bad)
                    yaw[i] = yaw[ok[np.hypot(x[ok] - x[i], y[ok] - y[i]).argmin()]] if len(ok) else 0.0
            for row, track, own in zip(gt.index, gt.track, gt.speed):
                i = at[track]
                no_heading += bool(bad[i])
                v = sp.copy()
                v[i] = own
                gap[row], dv[row] = IDM.lead(i, x, y, yaw, v, ext)
    out = d[["scenario", "track", "k"]].copy()
    out["lead_gap"], out["lead_dv"] = np.round(gap, 3), np.round(dv, 3)
    out.to_csv(os.path.join(args.samples, "lead.csv"), index=False)
    has = np.isfinite(gap)
    print(f"{len(d)} samples, {has.mean() * 100:.1f} % with a lead vehicle; gap median {np.median(gap[has]):.2f} m, "
          f"10th-90th {np.percentile(gap[has], 10):.2f}-{np.percentile(gap[has], 90):.2f} m; closing speed 10th-90th "
          f"{np.percentile(dv[has], 10):.2f} to {np.percentile(dv[has], 90):.2f} m/s; {no_heading} samples of vehicles that never move "
          f"(heading borrowed); {time.time() - t0:.0f} s")
    print("saved", os.path.join(args.samples, "lead.csv"))


if __name__ == "__main__":
    main()
