"""How close each driver's driving is to the real drivers', as distribution distances. Offline, from the per-tick logs
saved by closed_loop_multi.py (CL_TRAILS=<dir>); ego only, inside its recorded time window and up to its first collision.

Reference = the "expert" driver of the same run (the recorded Gwalior paths replayed in CARLA with the same physics and
controller), so simulator effects cancel. Quantities, pooled over all episodes of a driver:
  speed, longitudinal acceleration, longitudinal jerk, lateral acceleration   (smoothed as in metrics_v2.py)
  time headway   bumper gap to the nearest vehicle ahead in the ego's corridor / ego speed (ego faster than HEADWAY_MIN_SPEED)
  gap ahead      that bumper gap, m
  pass clearance side clearance between the boxes when the ego is level with a vehicle it passes or is passed by
Distances: Wasserstein-1 (in the quantity's unit) and Jensen-Shannon distance (0 = same, 1 = disjoint; shared 30 bins).
Also printed: the raw drone speed of the same egos vs the expert replay (how much the replay itself changes speed).

  python il/dist_v2.py Scenariosssss/IL/trails_eval
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_v2 import smooth, SMOOTH_S  # noqa: E402

HEADWAY_MIN_SPEED = 1.0   # m/s
AHEAD_MAX_M = 40.0
BINS = 30
MIN_N = 30                 # samples needed on both sides for a distance
QUANT = ["speed", "acc_lon", "jerk_lon", "acc_lat", "headway_s", "gap_ahead_m", "pass_clearance_m"]


def episode(z):
    roles = [str(r) for r in z["roles"]]
    ei = roles.index("ego")
    dt, start, fps = float(z["dt"]), int(z["start"]), float(z["fps"])
    E = z[f"tick{ei}"]
    t_rec = (z[f"frames{ei}"] - start) / fps
    coll = z[f"coll{ei}"]
    t_coll = float(coll[:, 0].min()) if len(coll) else np.inf
    m = (E[:, 0] >= t_rec[0]) & (E[:, 0] <= t_rec[-1]) & (E[:, 0] < t_coll)
    out = {q: np.array([]) for q in QUANT}
    if m.sum() < 5:
        return out, np.array([])
    n = max(1, int(round(SMOOTH_S / dt)))
    speed, yaw_rate = smooth(E[:, 4], n), smooth(E[:, 5], n)
    a_lon = np.gradient(speed, dt)
    out.update({"speed": E[m, 4], "acc_lon": a_lon[m], "jerk_lon": np.gradient(smooth(a_lon, n), dt)[m], "acc_lat": (speed * yaw_rate)[m]})
    # other vehicles in the ego frame (x forward, y to the side), per tick
    yaw = np.radians(E[:, 3])
    c, s = np.cos(yaw), np.sin(yaw)
    ext_e = z[f"ext{ei}"]
    gap = np.full(len(E), np.inf)
    clear = []
    for i in range(len(roles)):
        if i == ei:
            continue
        O, ext_o = z[f"tick{i}"], z[f"ext{i}"]
        k = min(len(O), len(E))
        dx, dy = O[:k, 1] - E[:k, 1], O[:k, 2] - E[:k, 2]
        fwd, side = dx * c[:k] + dy * s[:k], -dx * s[:k] + dy * c[:k]
        bumper = fwd - ext_e[0] - ext_o[0]
        ahead = (bumper > 0) & (bumper < AHEAD_MAX_M) & (np.abs(side) < ext_e[1] + ext_o[1])
        gap[:k] = np.where(ahead, np.minimum(gap[:k], bumper), gap[:k])
        cross = np.flatnonzero((np.sign(fwd[1:]) != np.sign(fwd[:-1])) & m[1:k]) + 1      # level with this vehicle
        clear += [abs(side[j]) - ext_e[1] - ext_o[1] for j in cross if abs(side[j]) < 6.0]
    has = m & np.isfinite(gap)
    out["gap_ahead_m"] = gap[has]
    mv = has & (E[:, 4] > HEADWAY_MIN_SPEED)
    out["headway_s"] = gap[mv] / E[mv, 4]
    out["pass_clearance_m"] = np.array(clear)
    # raw drone speed of the same ego (finite difference of the recorded positions, same smoothing)
    xy = z[f"xy{ei}"]
    drone = smooth(np.linalg.norm(np.diff(xy, axis=0), axis=1) * fps / np.maximum(np.diff(z[f"frames{ei}"]), 1), max(1, int(round(SMOOTH_S * fps)))) \
        if len(xy) > 2 else np.array([])
    return out, drone


def js(a, b):
    lo, hi = np.percentile(np.r_[a, b], [0.5, 99.5])
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, BINS + 1)
    pa, pb = np.histogram(np.clip(a, lo, hi), edges)[0], np.histogram(np.clip(b, lo, hi), edges)[0]
    return float(jensenshannon(pa / pa.sum(), pb / pb.sum(), base=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir")
    p.add_argument("--reference", default="expert")
    a = p.parse_args()
    pool, drone = {}, []
    for f in sorted(glob.glob(os.path.join(a.dir, "*.npz"))):
        z = np.load(f)
        if "tick0" not in z:
            continue
        o, dr = episode(z)
        d = pool.setdefault(str(z["driver"]), {q: [] for q in QUANT})
        for q in QUANT:
            d[q].append(o[q])
        if str(z["driver"]) == a.reference:
            drone.append(dr)
    pool = {dr: {q: np.concatenate(v) if v else np.array([]) for q, v in d.items()} for dr, d in pool.items()}
    if a.reference not in pool:
        sys.exit(f"no '{a.reference}' episodes in {a.dir}")
    ref = pool[a.reference]
    rows = []
    for dr, d in pool.items():
        for q in QUANT:
            x, r = d[q], ref[q]
            row = {"driver": dr, "quantity": q, "n": len(x), "mean": x.mean() if len(x) else None,
                   "median": np.median(x) if len(x) else None, "p95": np.percentile(x, 95) if len(x) else None}
            if len(x) >= MIN_N and len(r) >= MIN_N:   # fewer samples: no distance
                row["wasserstein"], row["js"] = wasserstein_distance(x, r), js(x, r)
            rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(a.dir, "dist_v2.csv"), index=False)
    pd.set_option("display.width", 250)
    for k in ("wasserstein", "js", "median", "n"):
        print(f"\n{k} (vs {a.reference}):" if k in ("wasserstein", "js") else f"\n{k}:")
        print(t.pivot(index="quantity", columns="driver", values=k).reindex(QUANT).round(3).to_string())
    if drone and len(np.concatenate(drone)):
        dr = np.concatenate(drone)
        print(f"\nraw drone speed of the same egos vs the {a.reference} replay: Wasserstein {wasserstein_distance(dr, ref['speed']):.3f} m/s, "
              f"mean {dr.mean():.2f} vs {ref['speed'].mean():.2f} m/s")
    print("\nwrote", os.path.join(a.dir, "dist_v2.csv"))


if __name__ == "__main__":
    main()
