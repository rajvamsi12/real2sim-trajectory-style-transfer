"""Driving metrics from the per-tick logs saved by closed_loop_multi.py (CL_TRAILS=<dir>), ego only. Offline, no CARLA.

Per episode (inside the ego's recorded time window, and up to its first collision):
  comfort    longitudinal / lateral acceleration (95th percentile of |a|) and jerk (RMS, 95th percentile); speed and yaw rate
             are smoothed over SMOOTH_S first (physics + PID noise)
  safety     min TTC and share of time with TTC < TTC_LOW_S: the ego's and every other vehicle's box moved at constant
             velocity, first time within TTC_HORIZON_S the boxes are closer than TTC_GAP_M (same idea as the TTC layer, but
             for every driver and against all vehicles); impact speed = relative speed at the ego's first contact
  TTC layer  share of ego ticks with a full brake / a speed cap, full-brake onsets per km
  tracking   mean distance to the recorded ego path, mean distance to the recorded position at the same time
Joined with closed_loop_<tag>.csv (collisions, behaviour check v2, progress) when --csv is given.

  python il/metrics_v2.py Scenariosssss/IL/trails_eval --csv Scenariosssss/IL/closed_loop_eval.csv
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

SMOOTH_S = 0.5
TTC_HORIZON_S, TTC_STEP_S, TTC_GAP_M, TTC_LOW_S = 3.0, 0.1, 0.2, 1.0
TTC_EVERY_S = 0.1       # the TTC is evaluated this often
NEAR_M = 30.0           # other vehicles further than this are skipped


def smooth(x, n):
    if n <= 1 or len(x) < 2:
        return np.asarray(x, float)
    pad = n // 2
    return np.convolve(np.pad(np.asarray(x, float), (pad, n - 1 - pad), mode="edge"), np.ones(n) / n, mode="valid")


def corners(x, y, yaw, ext):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[x + sx * ext[0] * c - sy * ext[1] * s, y + sx * ext[0] * s + sy * ext[1] * c]
                     for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))])


def seg_dist(p, a, b):
    d = b - a
    u = np.clip(((p - a) * d).sum(-1) / max(float(d @ d), 1e-12), 0, 1)
    return np.linalg.norm(p - (a + u[:, None] * d), axis=1).min()


def box_gap(A, B):
    """Distance between two convex quadrilaterals (0 if they overlap)."""
    for P in (A, B):
        for k in range(4):
            e = P[(k + 1) % 4] - P[k]
            ax = np.array([-e[1], e[0]])
            a, b = A @ ax, B @ ax
            if a.max() < b.min() or b.max() < a.min():
                break
        else:
            continue
        break
    else:
        return 0.0
    return min(min(seg_dist(A, B[k], B[(k + 1) % 4]) for k in range(4)), min(seg_dist(B, A[k], A[(k + 1) % 4]) for k in range(4)))


def ttc_at(e, others, ext_e, ext_o):
    """e / others: (x, y, yaw rad, speed). First time within the horizon the constant-velocity boxes are within TTC_GAP_M."""
    best = np.inf
    ve = e[3] * np.array([np.cos(e[2]), np.sin(e[2])])
    for o, xo in zip(others, ext_o):
        if np.hypot(o[0] - e[0], o[1] - e[1]) > NEAR_M:
            continue
        vo = o[3] * np.array([np.cos(o[2]), np.sin(o[2])])
        for t in np.arange(0.0, min(best, TTC_HORIZON_S + 1e-9), TTC_STEP_S):
            if box_gap(corners(e[0] + ve[0] * t, e[1] + ve[1] * t, e[2], ext_e),
                       corners(o[0] + vo[0] * t, o[1] + vo[1] * t, o[2], xo)) < TTC_GAP_M:
                best = t
                break
    return best


def poly_dist(path, pts):
    a, d = path[:-1], np.diff(path, axis=0)
    L2 = np.maximum((d * d).sum(1), 1e-12)
    out = []
    for p in pts:
        u = np.clip(((p - a) * d).sum(1) / L2, 0, 1)
        out.append(np.linalg.norm(p - (a + u[:, None] * d), axis=1).min())
    return np.array(out)


def episode(z):
    roles = [str(r) for r in z["roles"]]
    ei = roles.index("ego")
    dt, start, fps = float(z["dt"]), int(z["start"]), float(z["fps"])
    E = z[f"tick{ei}"]
    t_rec = (z[f"frames{ei}"] - start) / fps
    coll = z[f"coll{ei}"]
    t_coll = float(coll[:, 0].min()) if len(coll) else np.inf
    m = (E[:, 0] >= t_rec[0]) & (E[:, 0] <= t_rec[-1]) & (E[:, 0] < t_coll)
    row = {"scenario": str(z["scenario"]), "driver": str(z["driver"]), "behaviour": str(z["behaviour"]),
           "ticks": int(m.sum()), "ego_collision_logged": bool(len(coll))}
    if m.sum() < 5:
        return row
    n = max(1, int(round(SMOOTH_S / dt)))
    speed, yaw_rate = smooth(E[:, 4], n), smooth(E[:, 5], n)
    a_lon = np.gradient(speed, dt)
    a_lat = speed * yaw_rate
    j_lon, j_lat = np.gradient(smooth(a_lon, n), dt), np.gradient(smooth(a_lat, n), dt)
    dist = float(np.linalg.norm(np.diff(E[m][:, 1:3], axis=0), axis=1).sum())
    row.update({"dist_m": round(dist, 2), "time_s": round(m.sum() * dt, 2), "mean_speed": round(float(E[m, 4].mean()), 3),
                "acc_lon_p95": np.percentile(np.abs(a_lon[m]), 95), "acc_lat_p95": np.percentile(np.abs(a_lat[m]), 95),
                "jerk_lon_rms": np.sqrt((j_lon[m] ** 2).mean()), "jerk_lon_p95": np.percentile(np.abs(j_lon[m]), 95),
                "jerk_lat_rms": np.sqrt((j_lat[m] ** 2).mean()), "jerk_lat_p95": np.percentile(np.abs(j_lat[m]), 95)})
    layer = E[m, 6]
    onsets = int(((layer == 2) & (np.r_[0, layer[:-1]] != 2)).sum())
    row.update({"layer_brake_%": 100 * float((layer == 2).mean()), "layer_cap_%": 100 * float((layer == 1).mean()),
                "brake_onsets": onsets})
    # TTC against every other vehicle that is in the world at that time (all are spawned for the whole episode)
    others = [i for i in range(len(roles)) if i != ei]
    step = max(1, int(round(TTC_EVERY_S / dt)))
    ttcs = []
    for k in np.flatnonzero(m)[::step]:
        e = (E[k, 1], E[k, 2], np.radians(E[k, 3]), E[k, 4])
        o = [(z[f"tick{i}"][k, 1], z[f"tick{i}"][k, 2], np.radians(z[f"tick{i}"][k, 3]), z[f"tick{i}"][k, 4]) for i in others]
        ttcs.append(ttc_at(e, o, z[f"ext{ei}"], [z[f"ext{i}"] for i in others]))
    ttcs = np.array(ttcs)
    row.update({"min_ttc_s": float(min(ttcs.min(), TTC_HORIZON_S)) if len(ttcs) else TTC_HORIZON_S,
                "ttc_low_%": 100 * float((ttcs < TTC_LOW_S).mean()) if len(ttcs) else 0.0})
    rec_xy = z[f"xy{ei}"]
    row["path_dev_m"] = float(poly_dist(rec_xy, E[m][:, 1:3]).mean()) if len(rec_xy) > 1 else None
    row["pos_err_m"] = float(np.hypot(E[m, 1] - np.interp(E[m, 0], t_rec, rec_xy[:, 0]), E[m, 2] - np.interp(E[m, 0], t_rec, rec_xy[:, 1])).mean())
    if len(coll):                                         # relative speed just before the first contact
        k = max(int(np.searchsorted(E[:, 0], t_coll)) - 1, 0)
        j = int(coll[coll[:, 0].argmin(), 1])
        ve = E[k, 4] * np.array([np.cos(np.radians(E[k, 3])), np.sin(np.radians(E[k, 3]))])
        if j >= 0:
            O = z[f"tick{j}"]
            ve = ve - O[k, 4] * np.array([np.cos(np.radians(O[k, 3])), np.sin(np.radians(O[k, 3]))])
        row["impact_speed"] = float(np.linalg.norm(ve))
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir")
    p.add_argument("--csv", help="closed_loop_<tag>.csv of the same run (collisions, behaviour check, progress).")
    a = p.parse_args()
    d = pd.DataFrame([episode(np.load(f)) for f in sorted(glob.glob(os.path.join(a.dir, "*.npz")))])
    if a.csv:
        c = pd.read_csv(a.csv, dtype={"scenario": str})
        keep = [k for k in ("scenario", "driver", "result", "ego_collision", "behaviour2_checked", "behaviour2_kept", "gap_err_m",
                            "ego_progress", "ego_speed_vs_recorded") if k in c]
        d = d.merge(c[keep], on=["scenario", "driver"], how="left")
    d.to_csv(os.path.join(a.dir, "metrics_v2.csv"), index=False)
    g = d.groupby("driver")
    mean_cols = [k for k in ("mean_speed", "acc_lon_p95", "acc_lat_p95", "jerk_lon_rms", "jerk_lon_p95", "jerk_lat_rms", "jerk_lat_p95",
                             "min_ttc_s", "ttc_low_%", "layer_brake_%", "layer_cap_%", "path_dev_m", "pos_err_m", "impact_speed",
                             "ego_progress", "ego_speed_vs_recorded") if k in d]
    t = g[mean_cols].mean()
    t.insert(0, "episodes", g.size())
    km = g.dist_m.sum() / 1000
    t["km"] = km
    t["brake_onsets_per_km"] = g.brake_onsets.sum() / km
    coll = "ego_collision" if "ego_collision" in d else "ego_collision_logged"
    t["ego_collision_%"] = 100 * g[coll].apply(lambda x: (x.astype(str) == "True").mean())
    t["ego_collisions_per_km"] = g[coll].apply(lambda x: (x.astype(str) == "True").sum()) / km
    if "behaviour2_checked" in d:
        c2 = d[d.behaviour2_checked.astype(str) == "True"]
        t["behaviour_kept_%"] = 100 * c2.groupby("driver").behaviour2_kept.apply(lambda x: (x.astype(str) == "True").mean())
        t["kept_no_ego_coll_%"] = 100 * c2.groupby("driver").apply(
            lambda x: ((x.behaviour2_kept.astype(str) == "True") & (x[coll].astype(str) != "True")).mean())
        t["checkable"] = c2.groupby("driver").size()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(t.T.round(3).to_string())
    print("\nwrote", os.path.join(a.dir, "metrics_v2.csv"))


if __name__ == "__main__":
    main()
