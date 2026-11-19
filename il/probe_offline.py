"""Offline probe: how much of the expert's slow-down / speed-up a model predicts, on recorded samples (no CARLA).

Same definition as the CARLA probe (ttc_report.py): 2.5 s path length of the plan, the recorded future and const-vel, for
samples moving > 0.5 m/s; "expert slows" = recorded path >= 0.5 m shorter than const-vel; share predicted =
(plan - const-vel) / (recorded - const-vel), from the group means. Un-augmented samples of one split only.

  python il/probe_offline.py --data /mnt/data/raj_il/samples_rhd:Scenariosssss/IL/feats_rhd --runs rhd_scratch rhd_brk_w5
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T  # noqa: E402


def length(wp):
    return torch.cat([torch.zeros_like(wp[:, :1]), wp], 1).diff(dim=1).norm(dim=-1).sum(1)


@torch.no_grad()
def probe(path, t, device):
    ck = torch.load(path, weights_only=False, map_location=device)
    model = T.model_from_args(ck["args"]).to(device).eval()
    model.load_state_dict(ck["model"])
    loader = DataLoader(T.Samples(t, ck["args"].get("state", "accel"), ck["args"].get("lead", False)), batch_size=256, num_workers=4)
    rows = []
    for cam, lid, st, rt, vt, tgt, _, _ in loader:
        cam, lid, st, rt, vt, tgt = (x.to(device) for x in (cam, lid, st, rt, vt, tgt))
        wp, sc, _ = model(cam, lid, st, rt, vt)
        top = wp[torch.arange(len(wp)), sc.argmax(1)]
        speed = st[:, 0] * 10.0
        rows.append(torch.stack([speed, length(top), length(tgt), speed.clamp_min(0) * 2.5, (top - tgt).norm(dim=-1).mean(-1)], 1).cpu())
    p = pd.DataFrame(torch.cat(rows).numpy(), columns=["speed", "plan", "rec", "cv", "ade"])
    p = p[p.speed > 0.5]
    out = {"ckpt": os.path.basename(path)[:-3], "epoch": ck.get("epoch"), "n": len(p), "ADE": round(p.ade.mean(), 3)}
    for name, m in (("slow", p.rec - p.cv < -0.5), ("fast", p.rec - p.cv > 0.5)):
        q = p[m]
        out[f"n_{name}"] = len(q)
        out[f"{name}_share_%"] = round(100 * (q.plan.mean() - q.cv.mean()) / (q.rec.mean() - q.cv.mean()), 1)
        out[f"{name}_ADE"] = round(q.ade.mean(), 3)
    out["steady_plan-cv_m"] = round((p.plan - p.cv)[(p.rec - p.cv).abs() <= 0.5].mean(), 3)
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--data", nargs="+", required=True)
    a.add_argument("--runs", nargs="+", required=True)
    a.add_argument("--split", default="test")
    a.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = a.parse_args()
    t = T.load_table(args.data)
    t = t[T.has_feats(t)]
    t = t[(t.split == args.split) & (t.aug == "none")]
    print(f"{len(t)} un-augmented {args.split} samples", flush=True)
    rows = []
    for run in args.runs:
        for c in sorted(glob.glob(os.path.join(T.RUNS, run, "ckpt_*.pt"))):
            rows.append({"run": run, **probe(c, t, args.device)})
            print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(T.RUNS, "..", f"probe_offline_{args.split}.csv"), mode="a", index=False)
    pd.set_option("display.width", 250)
    print(d.to_string(index=False))


if __name__ == "__main__":
    main()
