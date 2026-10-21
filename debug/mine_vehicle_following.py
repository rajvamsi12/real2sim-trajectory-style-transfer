#!/usr/bin/env python3
"""
debug/mine_vehicle_following.py

Mines Vehicle Following scenarios (one vehicle following another, doing nothing else) from the
leader-vehicle columns of stabilize_ekf_lv_final.csv. There is no behaviour CSV for this behaviour.

A follower-frame counts when:
  - the follower has exactly ONE leader (ID_LV) in that frame (vehicles with two leaders get two rows),
  - direction == 1 (leader travels the same way),
  - neither vehicle is a pedestrian (label 4).
Consecutive frames with the same leader form a stretch; stretches are cut into windows of at most
WIN (18 frames = 4.2 s). A window is kept when:
  - neither vehicle is part of an overtake, cut-in/out or U-turn (behaviour CSVs) overlapping it,
  - both vehicles are >= 80 % inside the 6 regions / Center,
  - it passes scenario_checks.check_pair (overtake=False).
Lane and speed are deliberately NOT required (Raj, 2026-10-20). Downstream code uses windows with n >= 18.

Positions, labels and checks come from stabilize_ekf_final2.csv; only ID_LV / direction come from the lv file.

  python debug/mine_vehicle_following.py [--lv-csv stabilize_ekf_lv_final.csv]
  -> Scenariosssss/Vehicle Following/vehicle_following_candidates_relaxed.json
"""

import argparse
import collections
import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, BASE_DIR)

import scenario_checks as sc  # noqa: E402
from debug.check_scenarios import LABEL_BLUEPRINT_MAP as LBP  # noqa: E402

WIN = 18
MIN_LENGTHS = {"any (>= 2 frames)": 2, ">= 1 s (5 frames)": 5, ">= 2 s (9 frames)": 9, ">= 4.2 s (18 frames)": 18}
NAMES = {0: "3W", 1: "car", 2: "2W", 3: "bus", 5: "truck"}
SCEN = os.path.join(BASE_DIR, "Scenariosssss")
OUT = os.path.join(SCEN, "Vehicle Following", "vehicle_following_candidates_relaxed.json")


def busy_windows():
    """(track -> [(start, end)]) for every vehicle in an overtake, cut-in/out or U-turn row."""
    fr = lambda v: int(str(v).split("_")[-1])
    busy = collections.defaultdict(list)
    for path in ("Overtaking Left/overtaking.csv", "Cut In/cut_in_out.csv"):
        for r in pd.read_csv(os.path.join(SCEN, path)).itertuples():
            for t in (r.trackID, r.other_id):
                busy[t].append((fr(r.start_frame), fr(r.end_frame)))
    for r in pd.read_csv(os.path.join(SCEN, "U-turn", "u_turn.csv")).itertuples():
        busy[r.trackID].append((fr(r.start_frame), fr(r.end_frame)))
    return busy


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lv-csv", default=os.path.join(BASE_DIR, "stabilize_ekf_lv_final.csv"))
    args = p.parse_args()

    lv = pd.read_csv(args.lv_csv, usecols=["frameID", "trackID", "labelID", "ID_LV", "direction"])
    lv["f"] = lv.frameID.str.split("_").str[-1].astype(int)
    lv = lv[lv.groupby(["f", "trackID"]).ID_LV.transform("size") == 1].dropna(subset=["ID_LV"])
    lv["L"] = lv.ID_LV.astype(int)
    lab = lv.drop_duplicates("trackID").set_index("trackID").labelID
    lv["labL"] = lv.L.map(lab)
    c = lv[(lv.direction == 1) & (lv.labelID != sc.PEDESTRIAN_LABEL) & (lv.labL != sc.PEDESTRIAN_LABEL)].rename(columns={"trackID": "F"})
    print(f"follower-frames (one leader, same direction, no pedestrians): {len(c):,}")

    c = c.sort_values(["F", "L", "f"])
    c["run"] = ((c.f.diff() != 1) | (c.F.diff() != 0) | (c.L.diff() != 0)).cumsum()
    runs = c.groupby("run").agg(F=("F", "first"), L=("L", "first"), start=("f", "min"), end=("f", "max"), n=("f", "size"))
    print(f"same-leader stretches: {len(runs):,}; length (frames) median {runs.n.median():.0f}, "
          f"90th pct {runs.n.quantile(.9):.0f}, max {runs.n.max()}")

    busy = busy_windows()
    is_busy = lambda t, s, e: any(a <= e and s <= b for a, b in busy.get(t, []))
    wins = []
    for r in runs[runs.n >= 2].itertuples():
        for s in range(r.start, r.end + 1, WIN):
            e = min(s + WIN - 1, r.end)
            if e - s + 1 >= 2:
                wins.append((int(r.F), int(r.L), s, e, e - s + 1))
    w = pd.DataFrame(wins, columns=["F", "L", "s", "e", "n"])
    w["busy"] = [is_busy(F, s, e) or is_busy(L, s, e) for F, L, s, e in zip(w.F, w.L, w.s, w.e)]
    store = sc.TrackStore(track_ids=set(w.F) | set(w.L))
    w["inbox"] = [sc.inbox_ratio(store, [F, L], s, e) for F, L, s, e in zip(w.F, w.L, w.s, w.e)]

    cand = w[(~w.busy) & (w.inbox >= 0.8)].copy()
    oks, nowarn = [], []
    for r in cand.itertuples():
        ok, m, _ = sc.check_pair(store, r.F, r.L, r.s, r.e, LBP[store.label(r.F)], LBP[store.label(r.L)], overtake=False)
        oks.append(ok)
        nowarn.append(not m.get("warnings"))
    cand["ok"], cand["nowarn"] = oks, nowarn
    cand["types"] = [NAMES[store.label(F)] + "-" + NAMES[store.label(L)] for F, L in zip(cand.F, cand.L)]

    print(f"\n{'minimum length':24s} {'windows':>8s} {'not in other behaviour':>23s} {'>=80% in-box':>13s} "
          f"{'pass checks':>12s} {'no warnings':>12s} {'distinct pairs':>15s} {'car-car':>8s}")
    for name, k in MIN_LENGTHS.items():
        a = w[w.n >= k]
        b = a[~a.busy]
        d = cand[(cand.n >= k) & cand.ok]
        print(f"{name:24s} {len(a):8,} {len(b):23,} {(b.inbox >= 0.8).sum():13,} {len(d):12,} {d.nowarn.sum():12,} "
              f"{d.groupby(['F', 'L']).ngroups:15,} {(d.types == 'car-car').sum():8,}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cand[cand.ok].to_json(OUT, orient="records", indent=1)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
