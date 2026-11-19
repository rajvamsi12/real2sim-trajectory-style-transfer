"""Behaviour check v2 (2026-11-14). Used by closed_loop_multi.py (columns behaviour2_*) and offline on saved paths.

Old check (augment_scenarios.relation / same_behaviour): who is ahead at start / end along the straight line from the ego's
start to its end, plus the direction of the start-to-end lateral change (> 0.5 m).
v2, ego vs other, both measured along the RECORDED ego path (arc length s, signed lateral offset d):
  1. same vehicle ahead at the start and at the end as recorded;
  2. a pass (ahead/behind flips): the side is the lateral offset at the moment the two are level, and must match;
  3. no pass (following / being followed): the driven gap (s_ego - s_other) at the last driven frame is within TOL m of the
     recorded gap at that frame.
Not checkable: no "other" vehicle, < 2 common frames, or the recorded ego travels < MIN_TRAVEL m (slow traffic: every
driver passes, the scenario says nothing).

  python il/behaviour_v2.py Scenariosssss/IL/trails_behav_v2 --tol 5 10
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

STEP = 0.5          # m, spacing of the reference path points
EXTEND = 200.0      # m, straight extension of the reference path at both ends
MIN_TRAVEL = 5.0    # m, recorded ego path length below this = not checkable
TOL = 5.0           # m, gap tolerance of rule 3


def reference(xy, min_travel=MIN_TRAVEL):
    """Recorded ego path as a polyline with >= STEP m between points, extended straight at both ends; None if too short."""
    xy = np.asarray(xy, float)
    keep = [xy[0]]
    for p in xy[1:]:
        if np.linalg.norm(p - keep[-1]) >= STEP:
            keep.append(p)
    r = np.array(keep)
    if len(r) < 2 or np.linalg.norm(np.diff(r, axis=0), axis=1).sum() < min_travel:
        return None
    d0, d1 = r[1] - r[0], r[-1] - r[-2]
    return np.vstack([r[0] - EXTEND * d0 / np.linalg.norm(d0), r, r[-1] + EXTEND * d1 / np.linalg.norm(d1)])


def project(ref, pts):
    """(arc length, signed lateral offset) of each point on the polyline."""
    a, d = ref[:-1], np.diff(ref, axis=0)
    seg = np.linalg.norm(d, axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    out = []
    for p in np.asarray(pts, float):
        u = np.clip(((p - a) * d).sum(1) / seg ** 2, 0, 1)
        k = int(np.linalg.norm(p - (a + u[:, None] * d), axis=1).argmin())
        out.append((cum[k] + u[k] * seg[k], (d[k, 0] * (p[1] - a[k, 1]) - d[k, 1] * (p[0] - a[k, 0])) / seg[k]))
    return np.array(out)


def rel(ref, e, o):
    """frames, longitudinal gap (ego - other) and lateral offset (ego - other) at the common frames; None if < 2."""
    fo = {int(f): k for k, f in enumerate(o["frames"])}
    common = [(k, fo[int(f)]) for k, f in enumerate(e["frames"]) if int(f) in fo]
    if len(common) < 2:
        return None
    E = project(ref, np.asarray(e["xy"], float)[[k for k, _ in common]])
    O = project(ref, np.asarray(o["xy"], float)[[k for _, k in common]])
    return np.array([int(e["frames"][k]) for k, _ in common]), E[:, 0] - O[:, 0], E[:, 1] - O[:, 1]


def side_at_level(lon, lat):
    i = int(np.argmax(np.sign(lon) != np.sign(lon[0])))
    w = lon[i - 1] / (lon[i - 1] - lon[i]) if lon[i - 1] != lon[i] else 0.0
    return lat[i - 1] + w * (lat[i] - lat[i - 1])


def behaviour2(rec_pair, drv_pair, tol=TOL, min_travel=MIN_TRAVEL):
    """rec_pair / drv_pair: [ego, other] dicts with frames and xy (recorded / driven, the driven ones sampled at the recorded
    frames). -> dict(behaviour2_checked, behaviour2_kept, behaviour2_kind, gap_err_m). tol None = no gap rule."""
    out = {"behaviour2_checked": False, "behaviour2_kept": None, "behaviour2_kind": None, "gap_err_m": None}
    if len(rec_pair) < 2:
        return out
    ref = reference(rec_pair[0]["xy"], min_travel)
    rec = rel(ref, *rec_pair) if ref is not None else None
    if rec is None:
        return out
    s0, s1 = np.sign(rec[1][0]), np.sign(rec[1][-1])
    out["behaviour2_checked"] = True
    out["behaviour2_kind"] = ("ego passes" if s1 > 0 else "ego is passed") if s0 != s1 else ("ego behind" if s0 < 0 else "ego ahead")
    drv = rel(ref, *drv_pair) if len(drv_pair) == 2 and min(len(v["frames"]) for v in drv_pair) > 1 else None
    if drv is None:
        out["behaviour2_kept"] = False
        return out
    out["gap_err_m"] = round(float(drv[1][-1] - rec[1][rec[0] == drv[0][-1]][0]), 2)
    if np.sign(drv[1][0]) != s0 or np.sign(drv[1][-1]) != s1:
        out["behaviour2_kept"] = False
    elif s0 != s1:
        out["behaviour2_kept"] = bool(np.sign(side_at_level(rec[1], rec[2])) == np.sign(side_at_level(drv[1], drv[2])))
    else:
        out["behaviour2_kept"] = bool(tol is None or abs(out["gap_err_m"]) <= tol)
    return out


def sample(frames, start, fps, trail):
    """A driven trail (clock, x, y, ...) at the recorded frames that fall inside the driven time."""
    t = (np.asarray(frames) - start) / fps
    k = t <= trail[-1, 0]
    return {"frames": [int(f) for f in np.asarray(frames)[k]],
            "xy": np.stack([np.interp(t[k], trail[:, 0], trail[:, 1]), np.interp(t[k], trail[:, 0], trail[:, 2])], 1)}


def pairs_from_npz(z):
    roles = [str(r) for r in z["roles"]]
    if "other" not in roles:
        return [], []
    idx = [roles.index("ego"), roles.index("other")]
    start, fps = int(z["start"]), float(z["fps"])
    return ([{"frames": z[f"frames{i}"], "xy": z[f"xy{i}"]} for i in idx],
            [sample(z[f"frames{i}"], start, fps, z[f"trail{i}"]) for i in idx])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir")
    p.add_argument("--tol", type=float, nargs="+", default=[TOL])
    a = p.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.dir, "*.npz"))):
        z = np.load(f)
        rec, drv = pairs_from_npz(z)
        row = {"scenario": str(z["scenario"]), "behaviour": str(z["behaviour"]), "driver": str(z["driver"]),
               "ego_collision": bool(z["ego_collision"])}
        if rec:
            row["rec_travel_m"] = round(float(np.linalg.norm(np.diff(rec[0]["xy"], axis=0), axis=1).sum()), 1)
        for t in a.tol:
            b = behaviour2(rec, drv, t)
            row.update({"checked": b["behaviour2_checked"], "kind": b["behaviour2_kind"], "gap_err_m": b["gap_err_m"],
                        f"kept_tol{t:g}": b["behaviour2_kept"]})
        rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(a.dir, "behaviour_v2.csv"), index=False)
    c = d[d.checked].copy()
    cols = [f"kept_tol{t:g}" for t in a.tol]
    for k in cols:
        c[k] = c[k].astype(float)
        c[k + "&clean"] = c[k] * (~c.ego_collision)
    pd.set_option("display.width", 250)
    print(f"{d.scenario.nunique()} scenarios, {c.scenario.nunique()} checkable; not checkable: "
          f"{sorted(set(d.scenario) - set(c.scenario))}")
    print("\nKept %, by driver:\n", (100 * c.groupby("driver")[cols + [k + "&clean" for k in cols]].mean()).round(1)
          .assign(n=c.groupby("driver").size()).to_string())
    c["group"] = np.where(c.kind.isin(["ego passes", "ego is passed"]), "pass", "no pass")
    print("\nKept %, by driver and kind:\n", (100 * c.groupby(["driver", "group"])[cols].mean()).round(1)
          .assign(n=c.groupby(["driver", "group"]).size()).to_string())


if __name__ == "__main__":
    main()
