#!/usr/bin/env python3
"""
il/cache_features.py

Step 8 (ML_DECISIONS.md M8/M9): runs the two FROZEN pretrained encoders once over every recorded frame and stores
their outputs, so training never touches images or point clouds.

  camera  SegFormer-B0 (Cityscapes, il/weights/segformer_b0_cityscapes) encoder, last stage: 256 x 8 x 8
  LiDAR   PointPillars (nuScenes, il/pointpillars.py) BEV map 384 x 200 x 200 (0.5 m cells, +-50 m), cropped to
          x in [-8, 32) m (forward), y in [-16, 16) m, average-pooled to 2 m cells (20 x 16) and projected from 384 to
          PCA_DIMS channels with a fixed PCA (fitted once by --fit-pca, saved in il/weights/lidar_pca.npz)

Output per recorded vehicle: <feats>/<scenario>/veh_<track>_cam.npy (K, 256, 8, 8) and _lidar.npy (K, PCA_DIMS, 20, 16),
float16, one row per captured frame, and _k.npy with each row's frame number k (cam_<k>.jpg / lidar_<k>.npz).
Resumable: existing files are skipped.

  python il/cache_features.py --samples Scenariosssss/IL/samples_pilot --fit-pca       # once
  python il/cache_features.py --samples Scenariosssss/IL/samples_pilot --feats Scenariosssss/IL/feats_pilot
"""

import argparse
import glob
import os
import re
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import pointpillars as pp  # noqa: E402

SEG_DIR = os.path.join(SCRIPT_DIR, "weights", "segformer_b0_cityscapes")
PP_WEIGHTS = os.path.join(SCRIPT_DIR, "weights", "pointpillars_nus_secfpn_f19d00a3.pth")
PCA_FILE = os.path.join(SCRIPT_DIR, "weights", "lidar_pca.npz")
PCA_DIMS = 48
BEV_CELL_M = 0.5                     # PointPillars output cell (400 pillars of 0.25 m, stride 2)
CROP_X = (-8.0, 32.0)                # forward
CROP_Y = (-16.0, 16.0)               # left
POOL = 4                             # 0.5 m -> 2 m cells
IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD = np.array([0.229, 0.224, 0.225])
BATCH = 32


class Encoders:
    def __init__(self, device):
        from transformers import SegformerForSemanticSegmentation
        self.device = device
        self.seg = SegformerForSemanticSegmentation.from_pretrained(SEG_DIR, use_safetensors=True).segformer.eval().to(device)
        self.pp = pp.load_encoder(PP_WEIGHTS, device=device)
        x0, y0 = pp.PC_RANGE[0], pp.PC_RANGE[1]
        # BEV rows = y, cols = x (pointpillars.voxelize); crop in cells
        self.cols = slice(int((CROP_X[0] - x0) / BEV_CELL_M), int((CROP_X[1] - x0) / BEV_CELL_M))
        self.rows = slice(int((CROP_Y[0] - y0) / BEV_CELL_M), int((CROP_Y[1] - y0) / BEV_CELL_M))
        self.pca = None
        if os.path.exists(PCA_FILE):
            z = np.load(PCA_FILE)
            self.pca = (torch.tensor(z["mean"], device=device), torch.tensor(z["components"], device=device))

    @torch.no_grad()
    def camera(self, paths):
        imgs = [cv2.imread(p)[:, :, ::-1] / 255.0 for p in paths]
        x = torch.tensor(np.stack([(im - IMG_MEAN) / IMG_STD for im in imgs]).transpose(0, 3, 1, 2), dtype=torch.float32,
                         device=self.device)
        return self.seg(pixel_values=x).last_hidden_state.half().cpu().numpy()  # (B, 256, 8, 8)

    @torch.no_grad()
    def lidar_pooled(self, path):
        """Cropped, pooled BEV map (384, 16, 20) for one sweep: rows = y (left), cols = x (forward)."""
        xyz = np.load(path)["xyz_cm"].astype(np.float32) / 100.0
        pts = torch.tensor(np.c_[xyz, np.zeros(len(xyz))], dtype=torch.float32, device=self.device)  # 4th value: time lag 0
        bev = self.pp.bev(pts)[0][:, self.rows, self.cols]
        return F.avg_pool2d(bev[None], POOL)[0]

    @torch.no_grad()
    def lidar(self, path):
        f = self.lidar_pooled(path)                      # (384, 16, 20)
        c, h, w = f.shape
        v = f.reshape(c, -1).T - self.pca[0]             # (h*w, 384)
        return (v @ self.pca[1].T).T.reshape(-1, h, w).transpose(1, 2).half().cpu().numpy()  # (PCA_DIMS, 20, 16): x, y


def vehicle_dirs(samples_dir):
    return sorted(d for d in glob.glob(os.path.join(samples_dir, "*", "veh_*")) if os.path.isdir(d))


def frames(vdir, prefix, ext):
    ks = sorted(int(re.search(r"_(\d+)\.", os.path.basename(p)).group(1)) for p in glob.glob(os.path.join(vdir, f"{prefix}_*.{ext}")))
    return [os.path.join(vdir, f"{prefix}_{k:03d}.{ext}") for k in ks], ks


def fit_pca(enc, samples_dir, n_sweeps):
    rng = np.random.default_rng(0)
    paths = [p for d in vehicle_dirs(samples_dir) for p in frames(d, "lidar", "npz")[0]]
    paths = [paths[i] for i in rng.choice(len(paths), min(n_sweeps, len(paths)), replace=False)]
    X = torch.cat([enc.lidar_pooled(p).reshape(384, -1).T for p in paths]).double()
    mean = X.mean(0)
    cov = (X - mean).T @ (X - mean) / (len(X) - 1)
    val, vec = torch.linalg.eigh(cov)
    order = torch.argsort(val, descending=True)
    val, vec = val[order], vec[:, order]
    explained = (val[:PCA_DIMS].sum() / val.sum()).item()
    np.savez(PCA_FILE, mean=mean.float().cpu().numpy(), components=vec[:, :PCA_DIMS].T.float().cpu().numpy(),
             explained=explained, n_vectors=len(X))
    print(f"PCA {PCA_DIMS} of 384 channels from {len(paths)} sweeps ({len(X):,} cells): {explained * 100:.1f} % variance kept "
          f"(16: {val[:16].sum() / val.sum() * 100:.1f} %, 32: {val[:32].sum() / val.sum() * 100:.1f} %, 64: {val[:64].sum() / val.sum() * 100:.1f} %)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", required=True)
    p.add_argument("--feats", help="Output folder (default <samples>/../feats_<name>).")
    p.add_argument("--fit-pca", action="store_true")
    p.add_argument("--pca-sweeps", type=int, default=400)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    enc = Encoders(args.device)
    if args.fit_pca:
        fit_pca(enc, args.samples, args.pca_sweeps)
        return
    if enc.pca is None:
        sys.exit("no PCA yet: run with --fit-pca first")
    out = args.feats or os.path.join(os.path.dirname(args.samples.rstrip("/")), "feats_" + os.path.basename(args.samples.rstrip("/")))
    dirs = vehicle_dirs(args.samples)
    t0, n_frames = time.time(), 0
    for i, d in enumerate(dirs):
        scen, veh = os.path.basename(os.path.dirname(d)), os.path.basename(d)
        od = os.path.join(out, scen)
        cam_f, lid_f, k_f = (os.path.join(od, f"{veh}_{x}.npy") for x in ("cam", "lidar", "k"))
        if all(os.path.exists(f) for f in (cam_f, lid_f, k_f)):
            continue
        cams, kc = frames(d, "cam", "jpg")
        lids, kl = frames(d, "lidar", "npz")
        if not cams or kc != kl:
            print(f"  skip {d}: camera/LiDAR frames don't match", flush=True)
            continue
        cam = np.concatenate([enc.camera(cams[j:j + BATCH]) for j in range(0, len(cams), BATCH)])
        lid = np.stack([enc.lidar(p) for p in lids])
        os.makedirs(od, exist_ok=True)
        for f, a in ((cam_f, cam), (lid_f, lid), (k_f, np.array(kc, np.int16))):
            np.save(f + ".tmp.npy", a)
            os.replace(f + ".tmp.npy", f)
        n_frames += len(cams)
        if (i + 1) % 25 == 0 or i + 1 == len(dirs):
            el = time.time() - t0
            print(f"  {i + 1}/{len(dirs)} vehicles, {n_frames} frames, {el / max(n_frames, 1) * 1000:.0f} ms per frame", flush=True)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
