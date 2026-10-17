#!/usr/bin/env python3
"""
scenario_checks.py

Offline sanity checks for a two-vehicle scenario BEFORE it is replayed in CARLA.
No CARLA server is needed (timemethod only imports the carla module for types).

A scenario is rejected when:
  1. Either track is a pedestrian (label_id 4).
  2. The real-world EKF boxes already touch/overlap  -> any exact replay must collide.
  3. A track "teleports" (speed or heading jumps)     -> tracking / ID-switch glitch.
  4. The "overtake" never puts the two vehicles side by side with clearance
     (fixes the slide-16 rule, which accepted lateral separation < vehicle width).
  5. The CARLA blueprints, placed on the transformed paths, overlap or come
     closer than MIN_SIM_GAP_M (PID tracking needs some margin).

See BUG_COLLISIONS.md for the evidence behind each threshold.
"""

import ast
import math
import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(SCRIPT_DIR, "stabilize_ekf_final2.csv")

REAL_WORLD_FPS = 4.27
PEDESTRIAN_LABEL = 4

MIN_REAL_GAP_M = 0.2              # real boxes must stay this far apart
MIN_SIM_GAP_M = 0.3               # CARLA boxes on the replayed paths
MAX_SPEED_JUMP_MPS = 3.0          # per frame (0.23 s), i.e. ~13 m/s^2
MAX_HEADING_JUMP_DEG = 60.0       # per frame, only checked while moving
HEADING_CHECK_MIN_SPEED_MPS = 1.0
OVERTAKE_LATERAL_MARGIN_M = 0.3   # side-by-side clearance on top of half widths
MIN_PATH_COVERAGE = 0.9           # share of the window each CARLA path must keep after trimming
MIN_EGO_TRAVEL_M = 5.0            # an overtaking ego must actually drive, and further than the other

# CARLA footprints (length, width) in metres, from the handover blueprint table.
# check_scenarios.py --carla replaces these with the live bounding boxes.
BLUEPRINT_FOOTPRINT = {
    "vehicle.micro.microlino": (2.21, 1.48),
    "vehicle.nissan.micra": (3.63, 1.85),
    "vehicle.kawasaki.ninja": (2.04, 0.80),
    "vehicle.volkswagen.t2": (4.48, 2.07),
    "vehicle.mercedes.sprinter": (5.92, 1.99),
}

CSV_COLUMNS = [
    "frameID", "trackID", "labelID",
    "kf_cx_pixel", "kf_cy_pixel",
    "kf_cx_meter", "kf_cy_meter",
    "linear_speed_meter_per_sec", "heading_angle",
    "kf_obbox_coord_meter",
]


# ==========================================
# DATA LOADING
# ==========================================
class TrackStore:
    """Loads only the columns (and optionally only the tracks) the checks need."""

    def __init__(self, csv_path=DEFAULT_CSV, track_ids=None, chunksize=500_000):
        wanted = set(int(t) for t in track_ids) if track_ids is not None else None
        parts = []
        for chunk in pd.read_csv(csv_path, usecols=CSV_COLUMNS, chunksize=chunksize):
            if wanted is not None:
                chunk = chunk[chunk["trackID"].isin(wanted)]
            parts.append(chunk)
        df = pd.concat(parts, ignore_index=True)
        df["frame"] = df["frameID"].str.rsplit("_", n=1).str[-1].astype(int)
        self.tracks = {int(tid): g.set_index("frame").sort_index() for tid, g in df.groupby("trackID")}
        self._carla_paths = {}

    def label(self, tid):
        return int(self.tracks[tid]["labelID"].iloc[0])

    def window(self, tid, start, end):
        t = self.tracks.get(tid)
        return None if t is None else t.loc[(t.index >= start) & (t.index <= end)]

    def carla_path(self, tid):
        """Same conversion as Stage 2: full-track time method, keyed by frame (None = trimmed)."""
        if tid not in self._carla_paths:
            import timemethod
            t = self.tracks[tid]
            real = list(zip(t["kf_cx_pixel"].to_numpy(), (-t["kf_cy_pixel"]).to_numpy()))
            pts = timemethod.transform_trajectory_with_analysis(real)
            self._carla_paths[tid] = dict(zip(t.index.tolist(), pts))
        return self._carla_paths[tid]


# ==========================================
# GEOMETRY
# ==========================================
def box_corners(center, yaw_rad, length, width):
    f = np.array([math.cos(yaw_rad), math.sin(yaw_rad)])
    l = np.array([-f[1], f[0]])
    c = np.asarray(center, dtype=float)
    return [c + sx * f * length / 2 + sy * l * width / 2 for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]


def box_gap(a, b):
    """Separating-axis distance between two convex quads (negative = overlap depth)."""
    best = -float("inf")
    for poly in (a, b):
        for i in range(len(poly)):
            edge = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm < 1e-9:
                continue
            n /= norm
            pa = [p @ n for p in a]
            pb = [p @ n for p in b]
            best = max(best, min(pb) - max(pa), min(pa) - max(pb))
    return best


def parse_obbox(text):
    return [np.array(p, dtype=float) for p in ast.literal_eval(text)]


def obbox_dims(corners):
    e1 = np.linalg.norm(corners[1] - corners[0])
    e2 = np.linalg.norm(corners[2] - corners[1])
    return max(e1, e2), min(e1, e2)  # (length, width)


def path_yaws(points, k=2):
    """Heading at each point from a +-k point chord, robust to EKF jitter."""
    pts = np.asarray(points, dtype=float)
    yaws = []
    for i in range(len(pts)):
        a, b = pts[max(0, i - k)], pts[min(len(pts) - 1, i + k)]
        d = b - a
        yaws.append(math.atan2(d[1], d[0]) if np.hypot(*d) > 1e-6 else (yaws[-1] if yaws else 0.0))
    return yaws


def path_yaw_deg(carla_pts, k=3):
    """Spawn yaw from the first k segments of the CARLA path (replaces the EKF-heading guess)."""
    pts = [p for p in carla_pts if p is not None]
    if len(pts) < 2:
        return None
    a, b = np.asarray(pts[0]), np.asarray(pts[min(k, len(pts) - 1)])
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


# ==========================================
# CHECKS
# ==========================================
def check_pair(store, ego, other, start, end, ego_bp="vehicle.nissan.micra", other_bp="vehicle.nissan.micra",
               overtake=True, footprints=None):
    """Returns (ok, metrics, reasons) for one ego/other scenario window."""
    footprints = footprints or BLUEPRINT_FOOTPRINT
    reasons, m = [], {"ego": ego, "other": other, "start": start, "end": end}

    for tid in (ego, other):
        if tid not in store.tracks:
            return False, m, [f"track {tid} missing from CSV"]
        if store.label(tid) == PEDESTRIAN_LABEL:
            reasons.append(f"track {tid} is a pedestrian (label 4)")

    we, wo = store.window(ego, start, end), store.window(other, start, end)
    frames = sorted(set(we.index) & set(wo.index))
    if len(frames) < 0.8 * (end - start + 1):
        return False, m, reasons + [f"only {len(frames)} shared frames"]

    # 2. real-world box gap
    gaps, dims_e, dims_o = [], [], []
    for f in frames:
        be, bo = parse_obbox(we.at[f, "kf_obbox_coord_meter"]), parse_obbox(wo.at[f, "kf_obbox_coord_meter"])
        gaps.append(box_gap(be, bo))
        dims_e.append(obbox_dims(be))
        dims_o.append(obbox_dims(bo))
    m["min_real_gap_m"] = round(min(gaps), 2)
    if min(gaps) < MIN_REAL_GAP_M:
        reasons.append(f"real boxes overlap/touch (min gap {min(gaps):.2f} m at frame {frames[int(np.argmin(gaps))]})")

    # 3. tracking glitches
    for tid, w in ((ego, we), (other, wo)):
        w = w.loc[frames]
        sp = w["linear_speed_meter_per_sec"].to_numpy()
        jump = float(np.max(np.abs(np.diff(sp)))) if len(sp) > 1 else 0.0
        if jump > MAX_SPEED_JUMP_MPS:
            reasons.append(f"track {tid} speed jumps {jump:.1f} m/s in one frame")
        xy = w[["kf_cx_meter", "kf_cy_meter"]].to_numpy()
        yaw = np.degrees(path_yaws(xy))
        dyaw = np.abs((np.diff(yaw) + 180) % 360 - 180)
        moving = sp[1:] > HEADING_CHECK_MIN_SPEED_MPS
        if len(dyaw) and np.any(dyaw[moving] > MAX_HEADING_JUMP_DEG):
            reasons.append(f"track {tid} heading flips {float(np.max(dyaw[moving])):.0f} deg in one frame")

    # 4. overtake geometry (side-by-side phase needs lateral clearance)
    if overtake:
        pe = we.loc[frames, ["kf_cx_meter", "kf_cy_meter"]].to_numpy()
        po = wo.loc[frames, ["kf_cx_meter", "kf_cy_meter"]].to_numpy()
        travel = pe[-1] - pe[0]
        other_travel = np.linalg.norm(po[-1] - po[0])
        if np.linalg.norm(travel) < MIN_EGO_TRAVEL_M or np.linalg.norm(travel) <= other_travel:
            # A (nearly) parked ego has no meaningful direction; the behind->ahead flip is then noise
            reasons.append(f"ego does not move past the other (travels {np.linalg.norm(travel):.1f} m vs {other_travel:.1f} m)")
        else:
            r = travel / np.linalg.norm(travel)
            n = np.array([-r[1], r[0]])
            lon, lat = (pe - po) @ r, np.abs((pe - po) @ n)
            le, wde = np.median(dims_e, axis=0)
            lo, wdo = np.median(dims_o, axis=0)
            m["lon_start_m"], m["lon_end_m"] = round(float(lon[0]), 2), round(float(lon[-1]), 2)
            if not (lon[0] < 0 < lon[-1]):
                reasons.append("ego does not go from behind to ahead of the other vehicle")
            abreast = np.abs(lon) < (le + lo) / 2
            need = (wde + wdo) / 2 + OVERTAKE_LATERAL_MARGIN_M
            if abreast.any():
                m["min_lateral_abreast_m"] = round(float(lat[abreast].min()), 2)
                if lat[abreast].min() < need:
                    reasons.append(f"side-by-side lateral gap {lat[abreast].min():.2f} m < {need:.2f} m needed")
            else:
                reasons.append("no side-by-side phase sampled (passes through the other vehicle between frames)")

    # 5. CARLA blueprint gap along the transformed paths
    # A track trimmed early (left the 6 regions) stops dead in the replay and gets driven into,
    # so each track must keep MIN_PATH_COVERAGE of the window, starting at the window start.
    ce, co = store.carla_path(ego), store.carla_path(other)
    for tid, path in ((ego, ce), (other, co)):
        kept = [f for f in frames if path.get(f) is not None]
        if len(kept) < MIN_PATH_COVERAGE * len(frames) or not kept or kept[0] != frames[0]:
            reasons.append(f"track {tid} path trimmed outside the 6 regions ({len(kept)}/{len(frames)} frames kept)")
    sim_frames = [f for f in frames if ce.get(f) is not None and co.get(f) is not None]
    if len(sim_frames) >= 2:
        ye = path_yaws([ce[f] for f in sim_frames])
        yo = path_yaws([co[f] for f in sim_frames])
        (le, wde), (lo, wdo) = footprints[ego_bp], footprints[other_bp]
        sim_gaps = [box_gap(box_corners(ce[f], ye[i], le, wde), box_corners(co[f], yo[i], lo, wdo))
                    for i, f in enumerate(sim_frames)]
        m["min_sim_gap_m"] = round(min(sim_gaps), 2)
        if min(sim_gaps) < MIN_SIM_GAP_M:
            reasons.append(f"CARLA boxes come within {min(sim_gaps):.2f} m (< {MIN_SIM_GAP_M} m)")

    return not reasons, m, reasons


def dedupe_windows(rows):
    """Drop rows whose (ego, other) pair overlaps an earlier kept window. rows: dicts with ego/other/start/end."""
    kept, last_end = [], {}
    for r in sorted(rows, key=lambda r: (r["ego"], r["other"], r["start"])):
        key = (r["ego"], r["other"])
        if key in last_end and r["start"] <= last_end[key]:
            continue
        kept.append(r)
        last_end[key] = r["end"]
    return kept
