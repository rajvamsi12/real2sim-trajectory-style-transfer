#!/usr/bin/env python3
"""
debug/count_usable_scenarios.py

The per-behaviour "usable scenarios" table (README / AGENTS.md section 5), with one status per scenario.

For every candidate row (all CSV rows, overlapping windows NOT merged) plus Vehicle Following windows
(>= 18 frames, from mine_vehicle_following.py):
  1. skip pedestrians, < 80 % in-box, and scenarios failing scenario_checks      -> not counted
  2. can't spawn (first position outside the 6 regions / Center)                  -> "can't spawn"
  3. no car contact (CARLA-body gap >= 0 m):
       kerb/median hit (>= 5 off-road vehicle-frames)                              -> "kerb/median hit"
       else                                                                        -> "clean as recorded"
  4. car contact: deeper than -0.5 m                                              -> "car contact, too deep"
     otherwise apply_separation.separate (shift <= 0.7 m, gap >= 0.2 m after):
       no valid shift                                                              -> "car contact, shift fails"
       shifted path hits kerb/median                                               -> "shifted, kerb/median hit"
       shift <= 0.5 m / 0.5-0.7 m                                                  -> "rescued <= 0.5 m" / "rescued 0.5-0.7 m"
Usable = clean as recorded + rescued.

Needs Scenariosssss/town10hd_drivable.npz (predict_collisions.py --export-map). No CARLA server needed.

  python debug/count_usable_scenarios.py      # prints the table, writes Scenariosssss/usable_scenarios.csv
"""

import collections
import json
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, BASE_DIR)

import scenario_checks as sc  # noqa: E402
import timemethod  # noqa: E402
from debug.check_scenarios import behaviour_of, LABEL_BLUEPRINT_MAP as LBP  # noqa: E402
from debug.predict_collisions import load_map, off_road, CORNER_INSET_M, VALID_SPAWN  # noqa: E402
from debug import apply_separation as sep  # noqa: E402

SCEN = os.path.join(BASE_DIR, "Scenariosssss")
OUT = os.path.join(SCEN, "usable_scenarios.csv")
ORDER = ["Cut In", "Cut Out", "U-turn", "Overtaking Left", "Overtaking Right", "Vehicle Following"]
STATUSES = ["can't spawn", "kerb/median hit", "car contact, too deep", "car contact, shift fails",
            "shifted, kerb/median hit", "clean as recorded", "rescued <= 0.5 m", "rescued 0.5-0.7 m"]


def load_rows():
    fr = lambda v: int(str(v).split("_")[-1])
    rows = []
    for path in ("Cut In/cut_in_out.csv", "U-turn/u_turn.csv", "Overtaking Left/overtaking.csv"):
        for r in pd.read_csv(os.path.join(SCEN, path)).itertuples():
            rows.append({"ego": int(r.trackID), "other": int(r.other_id) if hasattr(r, "other_id") else None,
                         "start": fr(r.start_frame), "end": fr(r.end_frame), "behaviour": behaviour_of(r)})
    vf = os.path.join(SCEN, "Vehicle Following", "vehicle_following_candidates_relaxed.json")
    rows += [{"ego": int(r["F"]), "other": int(r["L"]), "start": int(r["s"]), "end": int(r["e"]),
              "behaviour": "Vehicle Following"} for r in json.load(open(vf)) if r["n"] >= 18]
    return rows


def main():
    grid, res, bounds = load_map()

    def kerb_hit(paths):
        off = 0
        for pts, (L, W) in paths:
            for p, y in zip(pts, sc.path_yaws(pts)):
                inner = sc.box_corners(p, y, L - 2 * CORNER_INSET_M, max(W - 2 * CORNER_INSET_M, 0.1))
                off += off_road(grid, res, bounds, *p) or sum(off_road(grid, res, bounds, *c) for c in inner) >= 2
        return off >= 5

    rows = load_rows()
    store = sc.TrackStore(track_ids={r["ego"] for r in rows} | {r["other"] for r in rows if r["other"] is not None})
    counts = collections.defaultdict(collections.Counter)
    out = []
    for r in rows:
        b = r["behaviour"]
        counts[b]["rows"] += 1
        ids = [r["ego"]] + ([r["other"]] if r["other"] is not None else [])
        if any(t not in store.tracks or store.label(t) == sc.PEDESTRIAN_LABEL for t in ids):
            continue
        if sc.inbox_ratio(store, ids, r["start"], r["end"]) < 0.8:
            continue
        bps = [LBP.get(store.label(t), "vehicle.nissan.micra") for t in ids]
        if len(ids) == 2:
            ok, m, _ = sc.check_pair(store, *ids, r["start"], r["end"], *bps, overtake=b.startswith("Overtaking"))
        else:
            ok, m, _ = sc.check_single(store, ids[0], r["start"], r["end"])
        if not ok:
            continue
        counts[b]["pass checks"] += 1
        shift, gap = None, m.get("min_sim_gap_m")
        first = lambda t: store.tracks[t].loc[store.tracks[t].index >= r["start"]].iloc[0]
        if not all(timemethod.classify_point((first(t).kf_cx_pixel, -first(t).kf_cy_pixel)) in VALID_SPAWN for t in ids):
            status = "can't spawn"
        else:
            frames = [f for f in range(r["start"], r["end"] + 1) if all(store.carla_path(t).get(f) is not None for t in ids)]
            P = [np.array([store.carla_path(t)[f] for f in frames]) for t in ids]
            F = [sc.BLUEPRINT_FOOTPRINT[bp] for bp in bps]
            if gap is None or gap >= 0:
                status = "kerb/median hit" if kerb_hit(list(zip(P, F))) else "clean as recorded"
            elif gap < sep.MIN_FIXABLE_GAP:
                status = "car contact, too deep"
            else:
                res_ = sep.separate(P[0], P[1], F[0], F[1], (grid, res, bounds))
                if res_ is None:
                    status = "car contact, shift fails"
                elif kerb_hit([(res_["pa"], F[0]), (res_["pb"], F[1])]):
                    status = "shifted, kerb/median hit"
                else:
                    shift = res_["shift_m"]
                    status = "rescued <= 0.5 m" if shift <= 0.5 else "rescued 0.5-0.7 m"
        counts[b][status] += 1
        out.append(dict(r, status=status, min_carla_gap_m=gap, shift_m=shift, warnings="; ".join(m.get("warnings", []))))

    t = pd.DataFrame({b: {c: counts[b][c] for c in ["rows", "pass checks"] + STATUSES} for b in ORDER}).T
    t.loc["Total"] = t.sum()
    t["USABLE (<= 0.5 m)"] = t["clean as recorded"] + t["rescued <= 0.5 m"]
    t["USABLE (<= 0.7 m)"] = t["USABLE (<= 0.5 m)"] + t["rescued 0.5-0.7 m"]
    pd.set_option("display.width", 250)
    print(t.to_string())
    pd.DataFrame(out).to_csv(OUT, index=False)
    print(f"\nsaved {OUT} ({len(out):,} scenarios with a status)")


if __name__ == "__main__":
    main()
