"""Repeated evaluation runs: per driver the mean and standard deviation over runs, and paired differences between drivers.

  python il/summary_runs.py eval_v2 eval_v2_run2 ... eval_v2_run10     # tags: closed_loop_<tag>.csv, trails_<tag>/{metrics,dist}_v2.csv
"""
import os
import sys

import numpy as np
import pandas as pd

IL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Scenariosssss", "IL")
T = lambda x: x.astype(str) == "True"  # noqa: E731


def one(tag):
    d = pd.read_csv(os.path.join(IL, f"closed_loop_{tag}.csv"), dtype={"scenario": str})
    m = pd.read_csv(os.path.join(IL, f"trails_{tag}", "metrics_v2.csv"), dtype={"scenario": str})
    d = d.merge(m[["scenario", "driver", "min_ttc_s", "ttc_low_%", "mean_speed", "jerk_lon_rms", "path_dev_m"]],
                on=["scenario", "driver"], how="left")
    d["ec"], d["k"], d["ch"] = T(d.ego_collision), T(d.behaviour2_kept), T(d.behaviour2_checked)
    g, c = d.groupby("driver"), d[d.ch].groupby("driver")
    t = pd.DataFrame({"episodes": g.size(), "ego_collision_%": 100 * g.ec.mean(), "behaviour_kept_%": 100 * c.k.mean(),
                      "kept_no_ego_coll_%": 100 * c.apply(lambda x: (x.k & ~x.ec).mean()),
                      "ego_progress": g.ego_progress.mean(), "speed_vs_rec_median": g.ego_speed_vs_recorded.median(),
                      "stands_still_%": 100 * g.ego_progress.apply(lambda x: (x < 0.1).mean()),
                      "min_ttc_s": g.min_ttc_s.mean(), "ttc_low_%": g["ttc_low_%"].mean(), "mean_speed": g.mean_speed.mean(),
                      "jerk_lon_rms": g.jerk_lon_rms.mean(), "path_dev_m": g.path_dev_m.mean()})
    q = pd.read_csv(os.path.join(IL, f"trails_{tag}", "dist_v2.csv"))
    for name in ("headway_s", "gap_ahead_m"):
        x = q[(q.quantity == name)].set_index("driver")
        t[name + "_median"] = x["median"]
        t[name + "_wasserstein"] = x["wasserstein"]
    return t, d


def main():
    tags = sys.argv[1:]
    R = [one(t) for t in tags]
    S = pd.concat({t: r[0] for t, r in zip(tags, R)}, names=["run"])
    pd.set_option("display.width", 260)
    mean, sd = S.groupby("driver").mean(), S.groupby("driver").std()
    print(f"{len(tags)} runs. mean over runs:\n{mean.round(2).T.to_string()}\n\nstandard deviation over runs:\n{sd.round(2).T.to_string()}")
    for col in ("ego_collision_%", "behaviour_kept_%", "kept_no_ego_coll_%"):
        print(f"\n{col} per run:\n{S[col].unstack(0).round(1).to_string()}")
    pairs = [("rhd_lead@ep08", "rhd_scratch@cl"), ("rhd_lead@ep08+ttc:0.7:0.2", "rhd_scratch@cl+ttc:0.7:0.2"),
             ("rhd_lead@ep08+ttc:0.7:0.2", "constvel+ttc:0.7:0.2"), ("rhd_scratch@cl+ttc:0.7:0.2", "constvel+ttc:0.7:0.2"),
             ("autopilot", "autopilot_stock"), ("idm", "rhd_lead@ep08+ttc:0.7:0.2")]
    print("\npaired over runs (a - b): mean difference, sd of the difference, runs with a > b / a < b")
    for a, b in pairs:
        for col in ("ego_collision_%", "behaviour_kept_%", "kept_no_ego_coll_%", "ego_progress"):
            x = (S[col].xs(a, level="driver") - S[col].xs(b, level="driver")).dropna()
            print(f"  {a} - {b}  {col}: {x.mean():+.2f}  sd {x.std():.2f}  {int((x > 0).sum())}/{int((x < 0).sum())} of {len(x)}")
    D = pd.concat({t: r[1].set_index(["scenario", "driver"]) for t, r in zip(tags, R)}, names=["run"])
    e = D.ec.unstack("run")
    k = D[D.ch].k.unstack("run")
    f = pd.DataFrame({"ego collision: scenarios never": (e.sum(axis=1) == 0).groupby("driver").sum(),
                      "always": (e.sum(axis=1) == e.shape[1]).groupby("driver").sum(),
                      "sometimes": ((e.sum(axis=1) > 0) & (e.sum(axis=1) < e.shape[1])).groupby("driver").sum(),
                      "kept: never": (k.sum(axis=1) == 0).groupby("driver").sum(),
                      "kept always": (k.sum(axis=1) == k.shape[1]).groupby("driver").sum(),
                      "kept sometimes": ((k.sum(axis=1) > 0) & (k.sum(axis=1) < k.shape[1])).groupby("driver").sum()})
    print(f"\nscenarios by outcome over the runs (ego collision of 114, kept of the checkable):\n{f.to_string()}")
    S.to_csv(os.path.join(IL, f"summary_{len(tags)}runs.csv"))


if __name__ == "__main__":
    main()
