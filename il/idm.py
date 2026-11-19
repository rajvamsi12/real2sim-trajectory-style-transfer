"""Intelligent Driver Model (Treiber et al. 2000) as a rule-based baseline: fit on the recorded Gwalior tracks, and the
helpers the closed-loop driver "idm" uses (il/closed_loop_multi.py).

  accel = a * (1 - (v / v0)^4 - (s* / s)^2),   s* = s0 + max(0, v*T + v*dv / (2*sqrt(a*b)))
  v own speed, s bumper gap to the lead vehicle, dv = v - v_lead (closing speed)

Lead vehicle = the nearest vehicle ahead whose box overlaps the follower's corridor sideways (traffic here is not lane-based,
so "same lane" has no meaning). The driver follows the ego's recorded path and only chooses the speed; it reads the lead's
true position and speed from the simulator (privileged, unlike the learned model).

Fit (train-split base scenarios only, each vehicle and frame once): v0 per vehicle type = 95th percentile of its recorded
speed while moving; T and s0 = slope and intercept of the median gap against speed in steady following; a = 90th percentile
of the recorded accelerations, b = 90th percentile of the recorded decelerations (while moving).

  python il/idm.py            # fit -> Scenariosssss/IL/idm_params.json
"""
import json
import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IL_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Scenariosssss", "IL")
PARAMS = os.path.join(IL_DIR, "idm_params.json")
DELTA = 4.0
AHEAD_MAX_M = 40.0
MIN_GAP = 0.1            # m, floor of the gap in the formula
ACC_MIN = -8.0           # m/s^2, floor of the IDM output
TEXTBOOK = {"T": 1.5, "s0": 2.0, "a": 1.0, "b": 1.5}
STEADY_GAP_MAX, STEADY_DV, STEADY_ACC = 15.0, 0.5, 0.5   # steady following: gap m, |closing speed| m/s, |acc| m/s^2


def idm_accel(v, gap, dv, v0, T, s0, a, b):
    """v, gap (np.inf = no lead), dv: floats or arrays."""
    v = np.maximum(v, 0.0)
    s_star = s0 + np.maximum(0.0, v * T + v * dv / (2.0 * np.sqrt(a * b)))
    inter = np.where(np.isfinite(gap), (s_star / np.maximum(gap, MIN_GAP)) ** 2, 0.0)
    return np.clip(a * (1.0 - (v / v0) ** DELTA - inter), ACC_MIN, a)


def lead(i, x, y, yaw, speed, ext):
    """Vehicle i's lead among all vehicles at one instant. x, y, yaw (rad), speed: (N,); ext: (N, 2) half length, half width.
    -> (bumper gap m or inf, closing speed v_i - v_lead along i's heading)."""
    c, s = np.cos(yaw[i]), np.sin(yaw[i])
    dx, dy = x - x[i], y - y[i]
    fwd, side = dx * c + dy * s, -dx * s + dy * c
    bumper = fwd - ext[i, 0] - ext[:, 0]
    ok = (bumper > 0) & (bumper < AHEAD_MAX_M) & (np.abs(side) < ext[i, 1] + ext[:, 1])
    ok[i] = False
    if not ok.any():
        return np.inf, 0.0
    j = int(np.flatnonzero(ok)[bumper[ok].argmin()])
    return float(bumper[j]), float(speed[i] - speed[j] * np.cos(yaw[j] - yaw[i]))


def load_params():
    return json.load(open(PARAMS))


def track_kinematics(frames, xy, fps):
    """Speed (smoothed over 3 frames), heading (held while nearly still) and acceleration of one recorded track."""
    t = np.asarray(frames, float) / fps
    xy = np.asarray(xy, float)
    d = np.gradient(xy, t, axis=0)
    speed = np.convolve(np.pad(np.linalg.norm(d, axis=1), 1, mode="edge"), np.ones(3) / 3, mode="valid")
    yaw = np.arctan2(d[:, 1], d[:, 0])
    for k in range(len(yaw)):
        if np.linalg.norm(d[k]) < 0.3:
            yaw[k] = yaw[k - 1] if k else np.nan
    if np.isnan(yaw).any():                       # leading still frames take the first valid heading
        good = np.flatnonzero(~np.isnan(yaw))
        yaw[:] = np.where(np.isnan(yaw), yaw[good[0]] if len(good) else 0.0, yaw)
    return speed, yaw, np.gradient(speed, t)


def main():
    import sys
    import pandas as pd
    sys.path.insert(0, SCRIPT_DIR)
    import train as T

    base = json.load(open(os.path.join(IL_DIR, "scenarios_8s.json")))
    fps = T.REAL_FPS
    is_train = lambda s: T.SPLIT_PATTERN[int((s["start"] + s["end"]) / 2 / fps // T.SPLIT_BLOCK_S) % len(T.SPLIT_PATTERN)] == "T"  # noqa: E731
    rows, seen = [], set()
    n_scen = 0
    for s in base:
        if not is_train(s):
            continue
        n_scen += 1
        V = [v for v in s["vehicles"] if len(v["frames"]) >= 5]
        kin = [track_kinematics(v["frames"], v["xy"], fps) for v in V]
        ext = np.array([[v["footprint"][0] / 2, v["footprint"][1] / 2] for v in V])
        at = [{int(f): k for k, f in enumerate(v["frames"])} for v in V]
        for f in range(s["start"], s["end"] + 1):
            here = [i for i in range(len(V)) if f in at[i]]
            if not here:
                continue
            x = np.array([V[i]["xy"][at[i][f]][0] for i in here])
            y = np.array([V[i]["xy"][at[i][f]][1] for i in here])
            yaw = np.array([kin[i][1][at[i][f]] for i in here])
            sp = np.array([kin[i][0][at[i][f]] for i in here])
            for n, i in enumerate(here):
                key = (V[i]["track"], f)
                if key in seen:
                    continue
                seen.add(key)
                gap, dv = lead(n, x, y, yaw, sp, ext[here])
                rows.append((V[i]["blueprint"], sp[n], gap, dv, kin[i][2][at[i][f]]))
    d = pd.DataFrame(rows, columns=["type", "v", "gap", "dv", "acc"])
    d["acc"] = d.acc.clip(-4, 4)
    v0 = d[d.v > 0.5].groupby("type").v.quantile(0.95).round(2).to_dict()
    d["v0"] = d.type.map(v0)
    d = d.dropna(subset=["v0"])
    print(f"{n_scen} train scenarios, {len(d)} vehicle-frames ({np.isfinite(d.gap).mean() * 100:.0f} % with a lead vehicle)")
    print("v0 per type (95th percentile of the speed while moving):", v0)
    fol = d[np.isfinite(d.gap) & (d.v > 1.0)]
    print(f"recorded followers moving > 1 m/s: median gap {fol.gap.median():.2f} m, median time headway {(fol.gap / fol.v).median():.2f} s, "
          f"10th-90th percentile headway {(fol.gap / fol.v).quantile(0.1):.2f}-{(fol.gap / fol.v).quantile(0.9):.2f} s")
    d.to_csv(os.path.join(IL_DIR, "idm_fit_samples.csv"), index=False)
    # T and s0 from the gaps drivers keep in steady following (IDM equilibrium far below v0: gap = s0 + v*T); a and b from
    # the recorded accelerations. (A least-squares fit on the acceleration itself collapses to "never react": the 4.27 Hz
    # tracks are too noisy, all four parameters end at their bounds and the error is above that of zero acceleration.)
    st = d[np.isfinite(d.gap) & (d.gap < STEADY_GAP_MAX) & (d.dv.abs() < STEADY_DV) & (d.acc.abs() < STEADY_ACC)]
    bins = pd.cut(st.v, np.arange(0, 7.5, 0.5))
    g = st.groupby(bins, observed=True).agg(v=("v", "mean"), gap=("gap", "median"), n=("gap", "size"))
    g = g[g.n >= 50]
    T_, s0 = np.polyfit(g.v, g.gap, 1, w=np.sqrt(g.n))
    mov = d[d.v > 0.5]
    a_ = float(mov.acc[mov.acc > 0.1].quantile(0.9))
    b_ = float(-mov.acc[mov.acc < -0.1].quantile(0.1))
    p = {"T": round(float(T_), 3), "s0": round(float(max(s0, 0.2)), 3), "a": round(a_, 3), "b": round(b_, 3)}
    print(f"steady following ({len(st)} vehicle-frames: gap < {STEADY_GAP_MAX} m, |dv| < {STEADY_DV} m/s, |acc| < {STEADY_ACC} m/s^2), median gap by speed:")
    print(g.round(2).to_string())
    print("fitted:", p, "| textbook:", TEXTBOOK)
    A = [d[k].to_numpy() for k in ("v", "gap", "dv", "v0", "acc")]
    res = lambda q: idm_accel(A[0], A[1], A[2], A[3], *q) - A[4]  # noqa: E731
    rmse = lambda r: float(np.sqrt((r ** 2).mean()))  # noqa: E731
    lead_m = np.isfinite(A[1])
    print(f"acceleration RMSE, all / with a lead: fitted {rmse(res(list(p.values()))):.3f} / {rmse(res(list(p.values()))[lead_m]):.3f}, "
          f"textbook {rmse(res(list(TEXTBOOK.values()))):.3f} / {rmse(res(list(TEXTBOOK.values()))[lead_m]):.3f}, "
          f"zero acceleration {rmse(A[4]):.3f} / {rmse(A[4][lead_m]):.3f} m/s^2")
    json.dump({"v0": v0, **p, "delta": DELTA, "samples": len(d), "fit": "il/idm.py, train-split base scenarios"}, open(PARAMS, "w"), indent=1)
    print("saved", PARAMS)


if __name__ == "__main__":
    main()
