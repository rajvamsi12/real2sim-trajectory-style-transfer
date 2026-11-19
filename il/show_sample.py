#!/usr/bin/env python3
"""
il/show_sample.py

Draws everything the model receives for ONE training sample (3 camera frames, 3 LiDAR sweeps seen from above, own
state, route command) and its label (the real driver's next 10 positions), and prints the raw CSV row.

  python il/show_sample.py --behaviour "Overtaking Right" --out /tmp/sample.png
"""

import argparse
import ast
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SAMPLES = "/mnt/data/raj_il/samples"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--behaviour", default="Overtaking Right")
    p.add_argument("--out", default="/tmp/sample.png")
    p.add_argument("--seed", type=int, default=4)
    a = p.parse_args()
    t = pd.read_csv(os.path.join(SAMPLES, "samples.csv"), dtype={"scenario": str, "base_id": str})
    c = t[(t.behaviour == a.behaviour) & (t.role == "ego") & (t.aug == "none") & (t.in_behaviour_window.astype(str) == "True")
          & (t.speed > 3)]
    r = c.sample(1, random_state=a.seed).iloc[0]
    print("ROW AS STORED IN samples.csv:")
    for k, v in r.items():
        print(f"  {k}: {v}")

    cams = ast.literal_eval(r.cam)
    lids = ast.literal_eval(r.lidar)
    wp = np.array([[r[f"wp{j}_x"], r[f"wp{j}_y"]] for j in range(10)])
    fig = plt.figure(figsize=(18, 10.5), dpi=110)
    fig.suptitle(f"One training sample  ·  scenario {r.scenario}  ·  {r.behaviour}  ·  role {r.role}  ·  t = {r.t:.2f} s",
                 fontsize=16, fontweight="bold")
    labels = ["t − 0.5 s", "t − 0.25 s", "t (now)"]
    for i, (cp, lp) in enumerate(zip(cams, lids)):
        ax = fig.add_subplot(3, 4, i + 1)
        ax.imshow(cv2.imread(os.path.join(SAMPLES, cp))[:, :, ::-1])
        ax.set_title(f"Camera {labels[i]}", fontsize=11)
        ax.axis("off")
        ax = fig.add_subplot(3, 4, 5 + i)
        xyz = np.load(os.path.join(SAMPLES, lp))["xyz_cm"] / 100.0
        m = (np.abs(xyz[:, 0]) < 40) & (np.abs(xyz[:, 1]) < 40) & (xyz[:, 2] > -1.6)   # drop the ground
        ax.scatter(-xyz[m, 1], xyz[m, 0], s=0.4, c=xyz[m, 2], cmap="viridis", vmin=-1.6, vmax=1.5)
        ax.add_patch(plt.Rectangle((-0.9, -1.8), 1.8, 3.6, color="red"))
        ax.set_xlim(-25, 25)
        ax.set_ylim(-15, 40)
        ax.set_aspect("equal")
        ax.set_title(f"LiDAR from above {labels[i]} ({len(xyz):,} pts, ground removed)", fontsize=10)
        ax.set_xlabel("right ← → left ... (m)" if False else "← left   lateral (m)   right →", fontsize=8)
        ax.tick_params(labelsize=7)
    # label
    ax = fig.add_subplot(3, 4, (4, 8))
    ax.plot(-wp[:, 1], wp[:, 0], "o-", color="tab:orange", lw=2, ms=6, label="label: real driver's next 2.5 s")
    ax.add_patch(plt.Rectangle((-0.9, -1.8), 1.8, 3.6, color="red", label="this car (now)"))
    for j in (0, 4, 9):
        ax.annotate(f"+{(j + 1) * 0.25:.2f} s", (-wp[j, 1], wp[j, 0]), textcoords="offset points", xytext=(8, -4), fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-3, max(20, wp[-1, 0] + 3))
    ax.grid(alpha=0.3)
    ax.set_title("OUTPUT TO LEARN: 10 waypoints (car's own frame)", fontsize=11, fontweight="bold")
    ax.set_xlabel("← left   lateral (m)   right →", fontsize=9)
    ax.set_ylabel("forward (m)", fontsize=9)
    ax.legend(fontsize=9, loc="lower right")
    # state
    ax = fig.add_subplot(3, 4, (9, 12))
    ax.axis("off")
    txt = (f"OWN STATE (input)\n"
           f"   speed  {r.speed:6.2f} m/s   ({r.speed * 3.6:.0f} km/h)\n"
           f"   acceleration  {r.accel:6.2f} m/s²\n"
           f"   yaw rate  {r.yaw_rate_rps:6.3f} rad/s\n"
           f"   steer (equivalent)  {r.steer_equiv_deg:6.1f}°\n\n"
           f"ROUTE COMMAND (input):  {r.route_cmd}\n\n"
           f"Not inputs (bookkeeping): vehicle {r.blueprint.split('.')[-1]}, track {r.track}, "
           f"in behaviour window = {r.in_behaviour_window}, augmentation = {r.aug}")
    ax.text(0.0, 0.95, txt, fontsize=13, family="monospace", va="top")
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
