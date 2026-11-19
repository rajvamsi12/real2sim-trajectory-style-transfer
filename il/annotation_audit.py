"""
il/annotation_audit.py

Where the drone annotations do not follow the vehicles (11 Nov 2026), in the 1,692 base scenarios the IL data was built
from (Scenariosssss/IL/scenarios_8s.json). Per scenario vehicle, inside the 8 s window, from stabilize_ekf_final2.csv:
  speed_jump  max change of speed between frames (EKF positions, m/s per frame; > 3 = ID switch / glitch, see
              carlascripts_BUG_COLLISIONS.md)
  heading_flip max change of travel direction between frames while moving > 1.5 m/s (deg; > 60 = glitch)
  ekf_drift   max distance between the EKF track and the stabilised detection of the same frame (m): the box
              sliding off the vehicle it was detected on
  overlap     min EKF box gap to another scenario vehicle (m; < 0 = the two boxes overlap)
  duplicate   frames where two scenario vehicles' centres are < 1.2 m apart (two IDs on one vehicle)

  select  write Scenariosssss/IL/annotation_audit/scores.csv and print the 10 worst (2 per problem type, distinct pairs)
  render  per id: 1_real_drone.mp4 (EKF boxes on the raw video), 2_carla_converted.mp4, 3_trajectory_drone.png (EKF path
          as a line, raw detections as dots, the worst moment circled in red), 4_trajectory_carla.png, info.txt

  python il/annotation_audit.py select
  python il/annotation_audit.py render --ids ...
"""

import argparse
import ast
import json
import math
import os
import sys

import cv2
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)

import proof_lhd as P  # noqa: E402
import render_compare_video as R  # noqa: E402
import scenario_checks as sc  # noqa: E402

IL = R.IL_DIR
OUT = os.path.join(IL, "annotation_audit")
DT = 1.0 / sc.REAL_WORLD_FPS
EXTRA = ["o_cx_pixel", "o_cy_pixel", "stab_cx_pixel", "stab_cy_pixel", "stab_cx_meter", "stab_cy_meter"]
TYPES = ["speed_jump", "heading_flip", "ekf_drift", "overlap", "duplicate"]


def vehicle_metrics(t):
    """Per-frame problem values for one track window (index = frame)."""
    p = t[["kf_cx_meter", "kf_cy_meter"]].to_numpy(float)
    d = np.diff(p, axis=0)
    v = np.r_[np.nan, np.hypot(*d.T) / DT]
    hd = np.r_[np.nan, np.degrees(np.arctan2(d[:, 1], d[:, 0]))]
    jump = np.r_[np.nan, np.abs(np.diff(v))]
    flip = np.r_[np.nan, np.abs((np.diff(hd) + 180) % 360 - 180)]
    moving = (v > 1.5) & (np.r_[np.nan, v[:-1]] > 1.5)
    flip = np.where(moving, flip, np.nan)
    drift = np.hypot(*(p - t[["stab_cx_meter", "stab_cy_meter"]].to_numpy(float)).T)
    return pd.DataFrame({"speed_jump": jump, "heading_flip": flip, "ekf_drift": drift}, index=t.index)


def scenario_metrics(store, s):
    row = {"id": s["id"], "behaviour": s["behaviour"], "pair": f"{s['ego']}_{s['other']}", "length_s": s["length_s"],
           "status": s["status"]}
    worst = {}
    wins = {}
    for v in s["vehicles"]:
        t = store.window(int(v["track"]), s["start"], s["end"])
        if t is None or len(t) < 3:
            continue
        wins[v["track"]] = t
        m = vehicle_metrics(t)
        for k in ("speed_jump", "heading_flip", "ekf_drift"):
            if m[k].notna().any() and m[k].max() > worst.get(k, (-1,))[0]:
                worst[k] = (float(m[k].max()), v["track"], v["role"], int(m[k].idxmax()))
    tr = list(wins)
    gap_best, dup_n, dup_at = (np.inf, None, None, None), 0, None
    for i in range(len(tr)):
        for j in range(i + 1, len(tr)):
            a, b = wins[tr[i]], wins[tr[j]]
            for f in a.index.intersection(b.index):
                ca, cb = a.loc[f, ["kf_cx_meter", "kf_cy_meter"]].to_numpy(float), b.loc[f, ["kf_cx_meter", "kf_cy_meter"]].to_numpy(float)
                if np.hypot(*(ca - cb)) < 1.2:
                    dup_n += 1
                    dup_at = dup_at or (tr[i], tr[j], int(f))
                if np.hypot(*(ca - cb)) > 8:
                    continue
                g = sc.box_gap([np.array(q, float) for q in ast.literal_eval(a.loc[f, "kf_obbox_coord_meter"])],
                               [np.array(q, float) for q in ast.literal_eval(b.loc[f, "kf_obbox_coord_meter"])])
                if g < gap_best[0]:
                    gap_best = (float(g), f"{tr[i]}+{tr[j]}", None, int(f))
    worst["overlap"] = gap_best
    worst["duplicate"] = (dup_n, f"{dup_at[0]}+{dup_at[1]}" if dup_at else None, None, dup_at[2] if dup_at else None)
    for k in TYPES:
        val, trk, role, f = worst.get(k, (np.nan, None, None, None))
        row[k] = round(val, 2) if np.isfinite(val) else np.nan
        row[f"{k}_track"] = trk if role is None else f"{trk} ({role})"
        row[f"{k}_t_s"] = round((f - s["start"]) * DT, 1) if f is not None else np.nan
    return row


def select():
    scens = json.load(open(os.path.join(IL, "scenarios_8s.json")))
    store = sc.TrackStore(track_ids={int(v["track"]) for s in scens for v in s["vehicles"]}, extra_columns=EXTRA)
    d = pd.DataFrame([scenario_metrics(store, s) for s in scens])
    os.makedirs(OUT, exist_ok=True)
    d.to_csv(os.path.join(OUT, "scores.csv"), index=False)
    pd.set_option("display.width", 250)
    print(f"{len(d)} scenarios scored")
    print("speed_jump > 3 m/s per frame:", int((d.speed_jump > 3).sum()), "| heading_flip > 60 deg:", int((d.heading_flip > 60).sum()),
          "| ekf_drift > 1 m:", int((d.ekf_drift > 1).sum()), "| overlap < 0 m:", int((d.overlap < 0).sum()),
          "| duplicate >= 3 frames:", int((d.duplicate >= 3).sum()))
    print(d[TYPES].describe().round(2).to_string())
    order = {"speed_jump": False, "heading_flip": False, "ekf_drift": False, "overlap": True, "duplicate": False}
    pick, pairs = [], set()
    for k, asc in order.items():
        c = d.dropna(subset=[k]).sort_values(k, ascending=asc)
        n = 0
        for _, r in c.iterrows():
            if r.id in pick or r.pair in pairs:
                continue
            pick.append(r.id)
            pairs.add(r.pair)
            n += 1
            print(f"  {k:12s} {r.id} {r.behaviour:18s} value {r[k]} track {r[k + '_track']} at t {r[k + '_t_s']} s")
            if n == 2:
                break
    print("PICK", " ".join(pick))


def mark(img, xy, text):
    cv2.circle(img, xy, 26, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(img, text, (xy[0] + 30, xy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)


def render(ids, out, port):
    import carla  # noqa: F401
    world = P.connect(port)
    orig = world.get_settings()
    st = world.get_settings()
    st.synchronous_mode, st.fixed_delta_seconds = True, 1.0 / R.SIM_FPS
    world.apply_settings(st)
    lib = world.get_blueprint_library()
    scens = {s["id"]: s for s in json.load(open(os.path.join(IL, "scenarios_8s.json")))}
    scores = pd.read_csv(os.path.join(OUT, "scores.csv"), dtype={"id": str}).set_index("id")
    sel = [scens[i] for i in ids]
    store = sc.TrackStore(track_ids={int(v["track"]) for s in sel for v in s["vehicles"]}, extra_columns=EXTRA)
    P.STORE = store
    all_boxes = R.load_boxes(sel)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    try:
        for k, s in enumerate(sel, 1):
            r = scores.loc[s["id"]]
            # the problem this scenario was picked for = its most extreme metric relative to the threshold
            rel = {"speed_jump": r.speed_jump / 3, "heading_flip": r.heading_flip / 60, "ekf_drift": r.ekf_drift / 1.0,
                   "overlap": -r.overlap / 0.5, "duplicate": r.duplicate / 3}
            kind = max(rel, key=lambda x: rel[x] if np.isfinite(rel[x]) else -1)
            d = os.path.join(out, f"{k:02d}_{s['id']}_{kind}")
            os.makedirs(d, exist_ok=True)
            n = int(round(s["length_s"] * R.SIM_FPS))
            roles = {v["track"]: v["role"] for v in s["vehicles"]}
            boxes = all_boxes[s["id"]]
            left, _, crop_w = R.drone_frames(s, boxes, roles, n)
            x0, y0, cw, ch = R.drone_crop(s, boxes)
            sx, sy = R.HALF_W / cw, R.HALF_H / ch
            right, note = R.carla_frames(world, lib, store, s, n, crop_w, roles)
            t_bad = float(r[f"{kind}_t_s"])
            for name, frames, title in (("1_real_drone.mp4", left, "REAL DRONE, EKF boxes"),
                                        ("2_carla_converted.mp4", right or [], "CARLA, converted")):
                w = cv2.VideoWriter(os.path.join(d, name), fourcc, float(R.SIM_FPS), (R.HALF_W, R.HALF_H))
                for i, f in enumerate(frames):
                    f = f.copy()
                    near = abs(i / R.SIM_FPS - t_bad) < 0.6
                    P.bar(f, [(f"{title}  #{s['id']} {s['behaviour']}  t {i / R.SIM_FPS:4.1f} s"
                               + (f"   <-- {kind} here" if near else ""), (110, 110, 255) if near else (240, 240, 240))])
                    w.write(f)
                w.release()
            # drone path image: EKF track (line) vs raw detections (dots) in raw-video pixels
            img = left[min(len(left) - 1, int(t_bad * R.SIM_FPS))].copy()
            img = cv2.addWeighted(img, 0.75, np.zeros_like(img), 0.25, 0)
            bad_px = None
            for v in sorted(s["vehicles"], key=lambda v: v["role"] == "ego"):
                t = store.window(int(v["track"]), s["start"], s["end"])
                off = t[["o_cx_pixel", "o_cy_pixel"]].to_numpy(float) - t[["stab_cx_pixel", "stab_cy_pixel"]].to_numpy(float)
                kf = t[["kf_cx_pixel", "kf_cy_pixel"]].to_numpy(float) + off
                det = t[["o_cx_pixel", "o_cy_pixel"]].to_numpy(float)
                kp = [tuple(((q - [x0, y0]) * [sx, sy]).astype(int)) for q in kf]
                col = R.COLORS[v["role"]]
                P.arrow_path(img, kp, col, 3 if v["role"] == "ego" else 2, every=4)
                for q in det:
                    cv2.circle(img, tuple(((q - [x0, y0]) * [sx, sy]).astype(int)), 3, (255, 255, 255), -1, cv2.LINE_AA)
                trk = str(r[f"{kind}_track"]).split(" ")[0].split("+")[0]
                if str(v["track"]) == trk:
                    i = int(np.argmin(np.abs((t.index.to_numpy() - s["start"]) * DT - t_bad)))
                    bad_px = kp[i]
            if bad_px:
                mark(img, bad_px, kind)
            P.bar(img, [(f"REAL DRONE  #{s['id']} {s['behaviour']}  frame at t {t_bad:.1f} s", (240, 240, 240)),
                        ("line = EKF track (what we used), white dots = raw detections; green ego, orange other, cyan neighbours",
                         (200, 200, 200)),
                        (f"{kind} = {r[kind]} ({r[kind + '_track']}) at t {t_bad} s", (110, 110, 255))])
            cv2.imwrite(os.path.join(d, "3_trajectory_drone.png"), img)
            steps = {"ego": []}
            for v in s["vehicles"]:
                steps.setdefault(v["role"], [])
            cimg = P.carla_traj_image(world, s, crop_w, world.get_map(), steps)
            P.bar(cimg, [(f"CARLA, converted paths  #{s['id']} {s['behaviour']}  (white arrows = CARLA lane direction)",
                          (240, 240, 240)),
                         ("converted paths: dot = start, arrow = travel; green ego, orange other, cyan neighbours", (200, 200, 200)),
                         (f"{kind} = {r[kind]} ({r[kind + '_track']}) at t {t_bad} s", (110, 110, 255))])
            cv2.imwrite(os.path.join(d, "4_trajectory_carla.png"), cimg)
            open(os.path.join(d, "info.txt"), "w").write(
                f"Scenario {s['id']}: {s['behaviour']}, ego track {s['ego']}, other {s['other']}, drone frames {s['start']}-{s['end']}, "
                f"{s['length_s']} s, build status '{s['status']}', build warnings '{s.get('warnings', '')}'\n"
                f"Picked for: {kind} = {r[kind]} on {r[kind + '_track']} at t = {t_bad} s (look there in the videos)\n"
                "All problem values in this window (thresholds from carlascripts_BUG_COLLISIONS.md / AGENTS.md):\n"
                + "".join(f"  {x:12s} {r[x]!s:>7s}  ({r[x + '_track']}, t {r[x + '_t_s']} s)\n" for x in TYPES)
                + "  speed_jump m/s per frame (> 3 = glitch); heading_flip deg while moving (> 60 = glitch); ekf_drift m between\n"
                "  EKF track and the detection; overlap m between EKF boxes (< 0 = overlap); duplicate = frames with two\n"
                "  vehicles' centres < 1.2 m apart.\n"
                f"CARLA replay (cure E, physics on): {note}\n"
                "1 real drone video with the EKF boxes, 2 the same seconds in CARLA after conversion, 3 EKF paths (lines) vs\n"
                "raw detections (white dots) on the drone frame at the bad moment, 4 the converted paths on the CARLA map.\n")
            print(f"  [{k}/{len(sel)}] {s['id']} {kind} {r[kind]} at t {t_bad}: {note}", flush=True)
    finally:
        R.clear_world(world)
        world.apply_settings(orig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["select", "render"])
    p.add_argument("--ids", nargs="*")
    p.add_argument("--out", default=OUT)
    p.add_argument("--port", type=int, default=2000)
    a = p.parse_args()
    select() if a.what == "select" else render(a.ids, a.out, a.port)
