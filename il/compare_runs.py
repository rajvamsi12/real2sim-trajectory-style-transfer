"""Repeated evaluation runs side by side: per driver, each run's value and the mean, plus how many scenarios change their
behaviour-kept / ego-collision outcome between runs.

  python il/compare_runs.py eval_v2 eval_v2_run2        # tags: closed_loop_<tag>.csv and trails_<tag>/metrics_v2.csv
"""
import os
import sys

import pandas as pd

IL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Scenariosssss", "IL")
T = lambda x: x.astype(str) == "True"  # noqa: E731
EXTRA = ["mean_speed", "jerk_lon_rms", "min_ttc_s", "ttc_low_%", "layer_brake_%", "layer_cap_%", "path_dev_m"]


def table(tag):
    d = pd.read_csv(os.path.join(IL, f"closed_loop_{tag}.csv"), dtype={"scenario": str})
    d = d[d.result != "spawn failed"]
    m = os.path.join(IL, f"trails_{tag}", "metrics_v2.csv")
    if os.path.exists(m):
        d = d.merge(pd.read_csv(m, dtype={"scenario": str})[["scenario", "driver"] + EXTRA], on=["scenario", "driver"], how="left")
    d["kept"] = T(d.behaviour2_kept)
    d["checked"] = T(d.behaviour2_checked)
    d["ecoll"] = T(d.ego_collision)
    return d


def summary(d):
    g, c = d.groupby("driver"), d[d.checked].groupby("driver")
    t = pd.DataFrame({"ego_collision_%": 100 * g.ecoll.mean(), "any_collision_%": 100 * g.result.apply(lambda r: (r == "collision").mean()),
                      "behaviour_kept_%": 100 * c.kept.mean(), "kept_no_ego_coll_%": 100 * c.apply(lambda x: (x.kept & ~x.ecoll).mean()),
                      "ego_progress": g.ego_progress.mean()})
    for k in EXTRA:
        if k in d:
            t[k] = g[k].mean()
    return t


def main():
    tags = sys.argv[1:]
    D = {t: table(t) for t in tags}
    S = {t: summary(d) for t, d in D.items()}
    pd.set_option("display.width", 250)
    for col in S[tags[0]].columns:
        t = pd.DataFrame({tag: S[tag][col] for tag in tags})
        t["mean"], t["range"] = t.mean(axis=1), t.max(axis=1) - t.min(axis=1)
        print(f"\n{col}:\n{t.round(2).to_string()}")
    a, b = (D[t].set_index(["scenario", "driver"]) for t in tags[:2])
    j = a.join(b, lsuffix="_1", rsuffix="_2", how="inner")
    ck = j[j.checked_1]
    f = pd.DataFrame({"checkable": ck.groupby("driver").size(),
                      "kept_changed": ck[ck.kept_1 != ck.kept_2].groupby("driver").size(),
                      "ego_collision_changed": j[j.ecoll_1 != j.ecoll_2].groupby("driver").size()}).fillna(0).astype(int)
    print(f"\nscenarios whose outcome differs between {tags[0]} and {tags[1]}:\n{f.to_string()}")


if __name__ == "__main__":
    main()
