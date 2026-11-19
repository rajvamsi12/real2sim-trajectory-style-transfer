#!/usr/bin/env python3
"""
il/filter_glitches.py

Annotation-glitch filter (Raj, 2026-11-10), run after il/build_scenarios_8s.py and before the CARLA check. Inside each
scenario window, from stabilize_ekf_final2.csv (same measures as il/annotation_audit.py):
  speed jump   change of speed between frames > 3 m/s (EKF positions)       -> ID switch / tracking glitch
  heading flip change of travel direction > 60 deg while moving > 1.5 m/s
  EKF drift    EKF track > 1.5 m from the stabilised detection of the same frame
Ego or other glitches -> the scenario is dropped (the ego's path is the training label, the other defines the behaviour).
A glitching neighbour -> only that neighbour is removed (no replacement).

Writes scenarios_8s.json (filtered; the input is kept as scenarios_8s.unfiltered.json) and filter_glitches.csv
(one row per scenario: what was dropped and why).

  python il/filter_glitches.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)

import scenario_checks as sc  # noqa: E402
from annotation_audit import EXTRA, vehicle_metrics  # noqa: E402

IL = os.path.join(os.path.dirname(SCRIPT_DIR), "Scenariosssss", "IL")
MAX_SPEED_JUMP, MAX_HEADING_FLIP, MAX_EKF_DRIFT = 3.0, 60.0, 1.5


def glitches(store, track, start, end):
    t = store.window(int(track), start, end)
    if t is None or len(t) < 3:
        return []
    m = vehicle_metrics(t)
    out = []
    if m.speed_jump.max() > MAX_SPEED_JUMP:
        out.append(f"speed jump {m.speed_jump.max():.1f}")
    if m.heading_flip.max() > MAX_HEADING_FLIP:
        out.append(f"heading flip {m.heading_flip.max():.0f}")
    if m.ekf_drift.max() > MAX_EKF_DRIFT:
        out.append(f"EKF drift {m.ekf_drift.max():.1f}")
    return out


def main():
    src = os.path.join(IL, "scenarios_8s.json")
    keep_src = os.path.join(IL, "scenarios_8s.unfiltered.json")
    scens = json.load(open(keep_src if os.path.exists(keep_src) else src))
    store = sc.TrackStore(track_ids={int(v["track"]) for s in scens for v in s["vehicles"]}, extra_columns=EXTRA)
    kept, rows = [], []
    for s in scens:
        row = {"id": s["id"], "behaviour": s["behaviour"], "dropped": False, "reason": "", "neighbours_removed": 0}
        main_bad = []
        for v in s["vehicles"]:
            if v["role"] in ("ego", "other"):
                main_bad += [f"{v['role']} {g}" for g in glitches(store, v["track"], s["start"], s["end"])]
        if main_bad:
            row.update(dropped=True, reason="; ".join(main_bad))
            rows.append(row)
            continue
        vs, removed = [], []
        for v in s["vehicles"]:
            g = glitches(store, v["track"], s["start"], s["end"]) if v["role"] == "neighbour" else []
            (removed if g else vs).append(v)
            if g:
                removed[-1] = f"{v['track']}: {', '.join(g)}"
        row.update(neighbours_removed=len(removed), reason="; ".join(removed))
        kept.append(dict(s, vehicles=vs))
        rows.append(row)
    if not os.path.exists(keep_src):
        os.replace(src, keep_src)
    json.dump(kept, open(src + ".tmp", "w"))
    os.replace(src + ".tmp", src)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(IL, "filter_glitches.csv"), index=False)
    pd.set_option("display.width", 200)
    t = pd.DataFrame({"before": d.groupby("behaviour").size(), "dropped": d.groupby("behaviour").dropped.sum(),
                      "kept": d[~d.dropped].groupby("behaviour").size(),
                      "neighbours removed": d.groupby("behaviour").neighbours_removed.sum()}).fillna(0).astype(int)
    t.loc["Total"] = t.sum()
    print(t.to_string())
    print(f"saved {src} ({len(kept):,} scenarios; unfiltered kept as {os.path.basename(keep_src)}) and filter_glitches.csv")


if __name__ == "__main__":
    main()
