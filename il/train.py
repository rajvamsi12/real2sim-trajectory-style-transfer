#!/usr/bin/env python3
"""
il/train.py

Step 9 (ML_DECISIONS.md M6-M9, T2, T3): trains il/model.ILDriver on cached encoder features.

Data: one or more recordings (samples.csv from il/record_samples.py) and their feature caches (il/cache_features.py).
Split (T3, by video time): the ~5.5 min drone video is cut into SPLIT_BLOCK_S blocks in the repeating pattern
SPLIT_PATTERN (T train, V val, X test); a base scenario and all its copies go to the block of the base window's midpoint.
Train samples whose real vehicle is also in val/test at the same second are dropped (and test samples that share one
with val), so no vehicle is seen at the same moment on two sides (measured 2026-10-27: 0 % leak, 1.4 % of samples dropped;
the md5 split by pair leaked ~50 %). --split pair keeps the old md5 split for comparison.
Weights (T2): samples of the ego/other inside the behaviour window count WEIGHT_BEHAVIOUR x; they are also recorded
twice as densely (0.25 s vs 0.5 s), so the effective over-sampling is ~2 x WEIGHT_BEHAVIOUR = 4x (plan: 3-5x).
Regularisation (the full run overfit after epoch 1): --wd, --dropout, --feat-drop, --epoch-samples (val every N train
samples instead of every 655k), --patience (early stop).
Checkpoints: model.pt = best on val top-mode ADE; last.pt = model + optimiser + scheduler + position, saved every epoch
and every --save-minutes; --resume continues from last.pt (mid-epoch too).
Early check: after --cv-check-epoch epochs (default 3) the best val top-mode ADE must be below val constant-velocity ADE;
otherwise the run stops (exit code 3, cv_check.json written, last.pt kept) instead of training for hours on a bad setup.
Inputs include the vehicle type (one-hot blueprint), so bike and car samples can drive differently; at test time = the spawned blueprint.
Loss: winner-takes-all over 6 modes + mode-score cross-entropy + 0.1 x behaviour cross-entropy (output head only).
Metrics: minADE6 / minFDE6, ADE / FDE of the top-scored mode, behaviour accuracy, and a constant-velocity baseline.
The best checkpoint is chosen on val ADE of the top-scored mode (the one the car drives), not minADE6.
Scenarios that collided while recording (timing.csv note) are left out.

  python il/train.py --data Scenariosssss/IL/samples_pilot:Scenariosssss/IL/feats_pilot --epochs 30 --name pilot
  python il/train.py ... --no-route --name pilot_noroute        # ablation (M5)
"""

import argparse
import ast
import collections
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from model import ILDriver, wta_loss, make_state, N_WP, ROUTE_CMDS, VEHICLE_TYPES, BEHAVIOURS, STATE_SCALE, SPEED_HISTORY, lead_feats  # noqa: E402

IL_DIR = os.path.join(BASE_DIR, "Scenariosssss", "IL")
RUNS = os.path.join(IL_DIR, "runs")
WEIGHT_BEHAVIOUR = 2.0   # x 2 from the denser sampling = ~4x effective
WEIGHT_STOPPED = 0.3   # stopped (< 0.5 m/s) and still stopped 2.5 s later; samples that start moving keep full weight
ACCEL_CLIP = 10.0      # m/s^2, recorded accel has noisy spikes
SPLITS = (0.8, 0.1, 0.1)   # --split pair only
SPLIT_BLOCK_S = 20.0
SPLIT_PATTERN = "TTTXTTTTVT"   # rotation chosen so Cut In stays in train and U-turn / Cut Out reach test
REAL_FPS = 4.27               # scenario_checks.REAL_WORLD_FPS (drone frames per second)
MAX_OPEN_VEHICLES = 128  # per worker; each keeps 2 memory-mapped files open (limit 1,024 open files per process)


# ==========================================
# DATA
# ==========================================
def pair_of_base(base_ids):
    """base scenario id -> 'behaviour|ego|other' (the split unit)."""
    scen = json.load(open(os.path.join(IL_DIR, "scenarios_8s.json")))
    m = {s["id"]: f"{s['behaviour']}|{s['ego']}|{s['other']}" for s in scen}
    return {b: m[b] for b in base_ids}


def split_of(pair):
    h = int(hashlib.md5(pair.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if h < SPLITS[0] else ("val" if h < SPLITS[0] + SPLITS[1] else "test")


def time_split(t):
    """'train' / 'val' / 'test' / 'drop' per sample (see the module docstring)."""
    base = {s["id"]: s for s in json.load(open(os.path.join(IL_DIR, "scenarios_8s.json")))}
    start = {k: s["start"] for k, s in base.items()}
    aug = os.path.join(IL_DIR, "scenarios_aug.json")
    if os.path.exists(aug):
        start.update({s["id"]: s["start"] for s in json.load(open(aug))})
    names = {"T": "train", "V": "val", "X": "test"}
    block = {b: names[SPLIT_PATTERN[int((s["start"] + s["end"]) / 2 / REAL_FPS // SPLIT_BLOCK_S) % len(SPLIT_PATTERN)]]
             for b, s in base.items()}
    split = t.base_id.map(block)
    sec = ((t.scenario.map(start) + t.t * REAL_FPS) / REAL_FPS).round().astype(int)
    key = t.track.astype(str) + "_" + sec.astype(str)
    held, val = set(key[split != "train"]), set(key[split == "val"])
    split = split.where(~((split == "train") & key.isin(held)), "drop")
    return split.where(~((split == "test") & key.isin(val)), "drop")


def load_table(data_args, split="time", slow_w=1.0):
    frames = []
    for d in data_args:
        samples, feats = d.split(":")
        t = pd.read_csv(os.path.join(samples, "samples.csv"), dtype={"scenario": str, "base_id": str})
        timing = os.path.join(samples, "timing.csv")
        if os.path.exists(timing):
            tm = pd.read_csv(timing, dtype={"scenario": str})
            collided = set(tm.scenario[tm.note.astype(str).str.startswith("collision")])
            drop = t.scenario.isin(collided)
            print(f"{samples}: {len(collided)} scenarios collided while recording, {drop.sum()} samples left out")
            t = t[~drop]
        lead = os.path.join(samples, "lead.csv")   # il/lead_input.py: gap and closing speed to the lead vehicle
        if os.path.exists(lead):
            n0 = len(t)
            t = t.merge(pd.read_csv(lead, dtype={"scenario": str}), on=["scenario", "track", "k"], how="left", validate="one_to_one")
            assert len(t) == n0 and t.lead_gap.notna().all(), "lead.csv does not match samples.csv"
        t["feats"] = feats
        frames.append(t)
    t = pd.concat(frames, ignore_index=True)
    t["base_id"] = t.base_id.str.zfill(5)
    t["ks"] = t.cam.map(lambda s: [int(p.rsplit("_", 1)[1].split(".")[0]) for p in ast.literal_eval(s)])
    pairs = pair_of_base(set(t.base_id))
    t["pair"] = t.base_id.map(pairs)
    t["split"] = t.pair.map(split_of) if split == "pair" else time_split(t)
    if (t.split == "drop").any():
        print(f"{(t.split == 'drop').sum()} samples dropped: same vehicle and second as a val/test sample")
        t = t[t.split != "drop"]
    in_beh = t.in_behaviour_window.astype(str) == "True"
    t["beh_label"] = np.where((t.role == "ego") & in_beh, t.behaviour.map(BEHAVIOURS.index), 0)
    t["weight"] = np.where(t.role.isin(["ego", "other"]) & in_beh, WEIGHT_BEHAVIOUR, 1.0)
    stays = (t.speed.abs() < 0.5) & (np.hypot(t.wp9_x, t.wp9_y) < 0.5)
    t["weight"] = np.where(stays, t.weight * WEIGHT_STOPPED, t.weight)
    if slow_w != 1.0:   # the expert slows: its 2.5 s path is >= 0.5 m shorter than const-vel (the probe's definition)
        wp = t[[f"wp{j}_{a}" for j in range(N_WP) for a in "xy"]].to_numpy(np.float32).reshape(-1, N_WP, 2)
        length = np.linalg.norm(np.diff(np.concatenate([np.zeros((len(t), 1, 2), np.float32), wp], 1), axis=1), axis=-1).sum(1)
        slows = (t.speed.to_numpy() > 0.5) & (length < t.speed.to_numpy() * N_WP * 0.25 - 0.5)
        t["weight"] = np.where(slows, t.weight * slow_w, t.weight)
        print(f"slow-down samples: {slows.mean() * 100:.1f} % of all, weighted x{slow_w}")
    t["accel"] = t.accel.clip(-ACCEL_CLIP, ACCEL_CLIP)
    return t


class Samples(Dataset):
    def __init__(self, t, state_layout="accel", lead=False):
        self.t = t.reset_index(drop=True)
        self.cache = collections.OrderedDict()   # LRU, at most MAX_OPEN_VEHICLES entries
        self.wp = t[[f"wp{j}_{a}" for j in range(N_WP) for a in "xy"]].to_numpy(np.float32).reshape(-1, N_WP, 2)
        past = t[[f"speed_m{h * 25:03d}" for h in SPEED_HISTORY]].to_numpy() if state_layout == "speedhist" else None
        self.state = make_state(state_layout, t.speed.to_numpy(), past, t.accel.to_numpy(), t.yaw_rate_rps.to_numpy(),
                                t.steer_equiv_deg.to_numpy()).astype(np.float32)
        if lead:
            self.state = np.concatenate([self.state, lead_feats(t.lead_gap.to_numpy(), t.lead_dv.to_numpy())], 1)
        self.route = np.eye(len(ROUTE_CMDS), dtype=np.float32)[t.route_cmd.map(ROUTE_CMDS.index).to_numpy()]
        self.vtype = np.eye(len(VEHICLE_TYPES), dtype=np.float32)[t.blueprint.map(VEHICLE_TYPES.index).to_numpy()]

    def feats(self, feats_dir, scenario, track):
        key = (feats_dir, scenario, track)
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            p = os.path.join(feats_dir, scenario, f"veh_{track}")
            ks = np.load(p + "_k.npy")
            self.cache[key] = (np.load(p + "_cam.npy", mmap_mode="r"), np.load(p + "_lidar.npy", mmap_mode="r"),
                               {int(k): i for i, k in enumerate(ks)})
            if len(self.cache) > MAX_OPEN_VEHICLES:
                self.cache.popitem(last=False)   # dropping the memmaps closes their files
        return self.cache[key]

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        r = self.t.iloc[i]
        cam, lid, idx = self.feats(r.feats, r.scenario, r.track)
        rows = [idx[k] for k in r.ks]
        return (torch.from_numpy(np.asarray(cam[rows], np.float32)), torch.from_numpy(np.asarray(lid[rows], np.float32)),
                torch.from_numpy(self.state[i]), torch.from_numpy(self.route[i]), torch.from_numpy(self.vtype[i]),
                torch.from_numpy(self.wp[i]),
                torch.tensor(r.beh_label), torch.tensor(r.weight, dtype=torch.float32))


class EpochSampler(torch.utils.data.Sampler):
    """Shuffled order fixed by (seed, epoch), so a resumed run can continue mid-epoch at the same position.
    With per_epoch < n_total each epoch is a fresh random subset of that size (more frequent val checks); epoch 0 = all."""
    def __init__(self, n_total, per_epoch=0, seed=0):
        self.n_total, self.seed, self.epoch, self.start = n_total, seed, 0, 0
        self.n = min(per_epoch, n_total) if per_epoch > 0 else n_total

    def set_epoch(self, epoch, start=0):
        self.epoch, self.start = epoch, start

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 100003 + self.epoch)
        n = self.n_total if self.epoch == 0 else self.n
        return iter(torch.randperm(self.n_total, generator=g)[self.start:n].tolist())

    def __len__(self):
        return (self.n_total if self.epoch == 0 else self.n) - self.start


def save_atomic(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def has_feats(t):
    ok = []
    for f, s, tr in zip(t.feats, t.scenario, t.track):
        ok.append(os.path.exists(os.path.join(f, s, f"veh_{tr}_k.npy")))
    return np.array(ok)


# ==========================================
# TRAIN / EVAL
# ==========================================
STATE_NOISE = {"accel": torch.tensor([0.05, 0.2, 0.05, 0.05]),   # scaled units: 0.5 m/s, 1 m/s^2, 0.05 rad/s, 1.5 deg
               "speedhist": torch.tensor([0.05, 0.2, 0.2, 0.2, 0.2, 0.05, 0.05])}


def model_from_args(a):
    """ILDriver from a run's args dict (older runs lack the newer keys and get the defaults)."""
    return ILDriver(use_route=not a.get("no_route", False), dropout=a.get("dropout", 0.0), feat_drop=a.get("feat_drop", 0.0),
                    n_modes=a.get("modes", 6), width=a.get("width", 1.0),
                    use_cam=not a.get("no_cam", False), use_lidar=not a.get("no_lidar", False),
                    state_layout=a.get("state", "accel"), lead=a.get("lead", False))


def init_from(model, run):
    """Fine-tune start: weights of an earlier run. An 'accel' run into a 'speedhist' model: the first state layer's accel
    column goes to the 0.25 s speed change (same scaled value, model.make_state), the 0.5-1.0 s columns start at 0."""
    ck = torch.load(os.path.join(RUNS, run, "model.pt"), weights_only=False, map_location="cpu")
    old, new = ck["model"], model.state_dict()
    src_layout = ck["args"].get("state", "accel")
    w = old["state.0.weight"]
    if src_layout == "accel" and model.state_layout == "speedhist":
        nw = torch.zeros_like(new["state.0.weight"])
        nw[:, 0], nw[:, 1] = w[:, 0], w[:, 1]                       # speed, accel -> 0.25 s speed change
        nw[:, 1 + len(SPEED_HISTORY):] = w[:, 2:]                   # yaw rate, steer, route, vehicle type
        old = dict(old, **{"state.0.weight": nw})
    missing = [k for k in new if k not in old or old[k].shape != new[k].shape]
    if missing:
        sys.exit(f"--init-from {run}: these weights don't fit: {missing}")
    model.load_state_dict(old)
    print(f"initialised from {run} ({src_layout} -> {model.state_layout})", flush=True)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    m = {"n": 0, "minADE6": 0.0, "minFDE6": 0.0, "ADE_top": 0.0, "FDE_top": 0.0, "ADE_constvel": 0.0, "FDE_constvel": 0.0,
         "beh_correct": 0, "beh_n": 0}
    t_wp = torch.arange(1, N_WP + 1, device=device) * 0.25
    for cam, lid, st, rt, vt, tgt, beh, _ in loader:
        cam, lid, st, rt, vt, tgt, beh = (x.to(device) for x in (cam, lid, st, rt, vt, tgt, beh))
        wp, sc, bl = model(cam, lid, st, rt, vt)
        d = (wp - tgt[:, None]).norm(dim=-1)                       # (B, K, 10)
        ade, fde = d.mean(-1), d[..., -1]
        top = sc.argmax(1)
        cv = torch.stack([st[:, 0:1] * 10.0 * t_wp, torch.zeros_like(t_wp)[None].expand(len(st), -1)], -1)
        dcv = (cv - tgt).norm(dim=-1)
        b = len(tgt)
        m["n"] += b
        m["minADE6"] += ade.min(1).values.sum().item()
        m["minFDE6"] += fde.min(1).values.sum().item()
        m["ADE_top"] += ade.gather(1, top[:, None]).sum().item()
        m["FDE_top"] += fde.gather(1, top[:, None]).sum().item()
        m["ADE_constvel"] += dcv.mean(-1).sum().item()
        m["FDE_constvel"] += dcv[:, -1].sum().item()
        sel = beh > 0
        m["beh_correct"] += (bl.argmax(1)[sel] == beh[sel]).sum().item()
        m["beh_n"] += sel.sum().item()
    n = max(m["n"], 1)
    out = {k: round(v / n, 3) for k, v in m.items() if k not in ("n", "beh_correct", "beh_n")}
    out["behaviour_acc_in_window"] = round(m["beh_correct"] / m["beh_n"], 3) if m["beh_n"] else None
    out["n"] = m["n"]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", required=True, help="samples_dir:feats_dir (one or more)")
    p.add_argument("--name", default="run")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4, help="AdamW weight decay.")
    p.add_argument("--dropout", type=float, default=0.0, help="Dropout on the fused feature.")
    p.add_argument("--feat-drop", type=float, default=0.0, help="Channel dropout on the cached camera/LiDAR features.")
    p.add_argument("--epoch-samples", type=int, default=0,
                   help="Train samples per epoch (random subset each epoch, val after each); 0 = all.")
    p.add_argument("--modes", type=int, default=6, help="Number of trajectory modes (1 = single path).")
    p.add_argument("--width", type=float, default=1.0, help="Scales all hidden sizes (0.5 = ~1/4 of the parameters).")
    p.add_argument("--no-cam", action="store_true", help="Ablation: camera features set to zero.")
    p.add_argument("--no-lidar", action="store_true", help="Ablation: LiDAR features set to zero.")
    p.add_argument("--state-noise", type=float, default=0.0,
                   help="Training only: Gaussian noise on the scaled state (std x this; 1 = 0.5 m/s, 1 m/s^2, 0.05 rad/s, 1.5 deg).")
    p.add_argument("--score-w", type=float, default=0.5, help="Weight of the mode-score cross-entropy.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=0, help="Stop after this many epochs without a new best val (0 = off).")
    p.add_argument("--no-route", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--split", choices=["time", "pair"], default="time")
    p.add_argument("--save-minutes", type=float, default=30.0, help="Also save last.pt this often inside an epoch.")
    p.add_argument("--resume", action="store_true", help="Continue from <run>/last.pt.")
    p.add_argument("--state", choices=["accel", "speedhist"], default="accel",
                   help="State input: accel (LHD runs) or speedhist (speed + its change over the last 0.25-1.0 s).")
    p.add_argument("--along-w", type=float, default=1.0, help="Loss weight of the along-path error (1 = plain L1).")
    p.add_argument("--init-from", help="Fine-tune: start from this run's model.pt (runs/<name>).")
    p.add_argument("--slow-w", type=float, default=1.0, help="Loss weight of samples where the expert slows down.")
    p.add_argument("--hist-drop", type=float, default=0.0,
                   help="speedhist state: share of training samples whose speed history is hidden (set to 0).")
    p.add_argument("--lead", action="store_true",
                   help="Extra state input: gap and closing speed to the lead vehicle (ground truth, il/lead_input.py).")
    p.add_argument("--ckpt-every", type=int, default=0,
                   help="Also save ckpt_epNN.pt every N epochs (for il/select_ckpt_closed_loop.py); 0 = off.")
    p.add_argument("--cv-check-epoch", type=int, default=3,
                   help="Stop if val top-mode ADE is not below const-vel after this epoch (0 = off).")
    args = p.parse_args()
    torch.manual_seed(args.seed)

    t = load_table(args.data, args.split, args.slow_w)
    keep = has_feats(t)
    if (~keep).any():
        print(f"{(~keep).sum()} samples without cached features are left out")
    t = t[keep]
    print("samples per split:", t.split.value_counts().to_dict(), "| pairs per split:",
          t.groupby("split").pair.nunique().to_dict(), flush=True)
    sampler = EpochSampler(int((t.split == "train").sum()), args.epoch_samples, args.seed)
    loaders = {s: DataLoader(Samples(t[t.split == s], args.state, args.lead), batch_size=args.batch, sampler=sampler if s == "train" else None,
                             num_workers=args.workers, drop_last=False) for s in ("train", "val", "test") if (t.split == s).any()}
    n_batches = -(-sampler.n // args.batch)

    model = model_from_args(vars(args)).to(args.device)
    if args.init_from and not (args.resume and os.path.exists(os.path.join(RUNS, args.name, "last.pt"))):
        init_from(model, args.init_from)
    print(f"model parameters: {sum(x.numel() for x in model.parameters()) / 1e6:.2f} M, device {args.device}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * n_batches)
    out_dir = os.path.join(RUNS, args.name)
    os.makedirs(out_dir, exist_ok=True)
    last_pt = os.path.join(out_dir, "last.pt")
    hist, best, ep0, done0, tot0, n0 = [], float("inf"), 1, 0, 0.0, 0
    best_ep = 0
    if args.resume and os.path.exists(last_pt):
        ck = torch.load(last_pt, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        hist, best, ep0, done0, tot0, n0 = ck["hist"], ck["best"], ck["epoch"], ck["batches_done"], ck["tot"], ck["n"]
        best_ep = ck.get("best_ep", len(hist))
        print(f"resumed from {last_pt}: epoch {ep0}, batch {done0}/{n_batches}, best val ADE_top {best}", flush=True)

    def save_last(ep, batches_done, tot, n):
        save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "hist": hist,
                     "best": best, "epoch": ep, "batches_done": batches_done, "tot": tot, "n": n, "best_ep": best_ep, "args": vars(args)}, last_pt)

    for ep in range(ep0, args.epochs + 1):
        model.train()
        done, tot, n = (done0, tot0, n0) if ep == ep0 else (0, 0.0, 0)
        sampler.set_epoch(ep, done * args.batch)
        t0 = t_save = time.time()
        for cam, lid, st, rt, vt, tgt, beh, w in loaders["train"]:
            cam, lid, st, rt, vt, tgt, beh, w = (x.to(args.device) for x in (cam, lid, st, rt, vt, tgt, beh, w))
            if args.state_noise:
                sn = STATE_NOISE[args.state].to(st.device)
                st = st + torch.randn_like(st) * torch.cat([sn, sn.new_zeros(st.shape[1] - len(sn))]) * args.state_noise
            if args.hist_drop and args.state == "speedhist":   # the model cannot just copy its recent speed change
                st = st.clone()
                st[torch.rand(len(st), device=st.device) < args.hist_drop, 1:1 + len(SPEED_HISTORY)] = 0.0
            wp, sc, bl = model(cam, lid, st, rt, vt)
            loss, _ = wta_loss(wp, sc, tgt, w, args.score_w, args.along_w)
            loss = loss + 0.1 * (torch.nn.functional.cross_entropy(bl, beh, reduction="none") * w).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(tgt)
            n += len(tgt)
            done += 1
            if time.time() - t_save > args.save_minutes * 60:
                save_last(ep, done, tot, n)
                t_save = time.time()
        row = {"epoch": ep, "train_loss": round(tot / n, 4), "seconds": round(time.time() - t0, 1)}
        if "val" in loaders:
            v = evaluate(model, loaders["val"], args.device)
            row.update({f"val_{k}": x for k, x in v.items()})
            if v["ADE_top"] < best:
                best, best_ep = v["ADE_top"], ep
                save_atomic({"model": model.state_dict(), "args": vars(args), "epoch": ep}, os.path.join(out_dir, "model.pt"))
        if args.ckpt_every and ep % args.ckpt_every == 0:
            save_atomic({"model": model.state_dict(), "args": vars(args), "epoch": ep}, os.path.join(out_dir, f"ckpt_ep{ep:02d}.pt"))
        hist.append(row)
        pd.DataFrame(hist).to_csv(os.path.join(out_dir, "history.csv"), index=False)
        save_last(ep + 1, 0, 0.0, 0)
        if "val" in loaders and ep == args.cv_check_epoch:
            ok = best < v["ADE_constvel"]
            json.dump({"epoch": ep, "best_val_ADE_top": best, "val_ADE_constvel": v["ADE_constvel"], "passed": ok},
                      open(os.path.join(out_dir, "cv_check.json"), "w"), indent=1)
            print(f"  CV CHECK epoch {ep}: best val ADE_top {best} vs const-vel {v['ADE_constvel']} -> "
                  + ("OK" if ok else "FAILED, stopping (resume with --resume --cv-check-epoch 0 to continue anyway)"), flush=True)
            if not ok:
                sys.exit(3)
        print("  " + "  ".join(f"{k} {x}" for k, x in row.items() if k in ("epoch", "train_loss", "val_minADE6", "val_minFDE6",
                                                                           "val_ADE_top", "val_ADE_constvel", "seconds")), flush=True)
        if args.patience and "val" in loaders and ep - best_ep >= args.patience:
            print(f"  early stop: no new best val ADE_top for {args.patience} epochs (best {best}, epoch {best_ep})", flush=True)
            break
    sampler.set_epoch(0)   # full train set for the final evaluation (a resumed epoch started mid-way)
    if os.path.exists(os.path.join(out_dir, "model.pt")):
        model.load_state_dict(torch.load(os.path.join(out_dir, "model.pt"), weights_only=True)["model"])
    final = {s: evaluate(model, l, args.device) for s, l in loaders.items()}
    json.dump({"args": vars(args), "best_val_ADE_top": best, "final": final}, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
    for s, v in final.items():
        print(f"{s:5s} {v}")
    print(f"saved {out_dir}")


if __name__ == "__main__":
    main()
