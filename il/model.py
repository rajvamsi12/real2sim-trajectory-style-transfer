#!/usr/bin/env python3
"""
il/model.py

Trainable part of the imitation-learning driver (ML_DECISIONS.md section 0, M1-M9), on top of the cached frozen
encoder features (il/cache_features.py):

  camera  3 frames (t-0.5, t-0.25, t) x SegFormer-B0 stage 4 (256 x 8 x 8)   -> conv head -> 256
  LiDAR   3 frames x PointPillars BEV, PCA 48 x 20 x 16 (2 m cells)         -> conv head -> 512
  state   speed, accel, yaw rate, steer equivalent + route command (4)
          + vehicle type (5, one-hot blueprint = the vehicle being driven)  -> MLP       -> 64
  fusion  concat 832 -> MLP 512
  decoder K = 6 modes; per mode a 2-layer GRU rolls out 10 waypoints (0.25 s apart) autoregressively
  heads   mode scores (K) and behaviour class (output only, never an input; M3/M4)
  regul.  optional dropout on the fused feature and channel dropout on the cached features (both off by default;
          the full run without them overfit after epoch 1, IL_STATUS.md step 9)
  options n_modes (1 = plain regression, no mode choice), width (scales all hidden sizes), use_cam / use_lidar (ablations)
  prior   each mode = constant-velocity path (current speed, straight ahead) + the decoder's offsets, so a wrongly
          picked mode stays near const-vel (pilot, 3 seeds: held-out top-mode ADE 0.69-0.83 m vs 1.18-1.37 m without)

No other vehicle's state goes in (M1); the route command can be zeroed for the no-route ablation (M5).

State layouts (2026-11-12): 'accel' (the LHD runs) = speed, accel, yaw rate, steer; 'speedhist' = speed, then the
speed change over the last 0.25 / 0.5 / 0.75 / 1.0 s (speed - speed_mXXX, scaled by 1.25 / 2.5 / 3.75 / 5 so that the
first one equals the old scaled accel: (v - v_0.25s) / 0.25 / 5), yaw rate, steer. The LHD model could not tell a car that
is braking from one that is not from a single noisy accel; it never learned to slow down (IL_STATUS closed-loop diagnosis).
"""

import numpy as np
import torch
import torch.nn as nn

N_WP, N_MODES, HISTORY = 10, 6, 3
ROUTE_CMDS = ["left", "straight", "right", "u-turn"]
VEHICLE_TYPES = ["vehicle.nissan.micra", "vehicle.kawasaki.ninja", "vehicle.micro.microlino",
                 "vehicle.mercedes.sprinter", "vehicle.volkswagen.t2"]
BEHAVIOURS = ["plain", "Cut In", "Cut Out", "U-turn", "Overtaking Left", "Overtaking Right", "Vehicle Following"]
STATE_DIMS = 4
STATE_SCALE = np.array([10.0, 5.0, 1.0, 30.0], np.float32)   # speed m/s, accel m/s^2, yaw rate rad/s, steer deg
SPEED_HISTORY = (1, 2, 3, 4)                                 # x 0.25 s, the recorder's speed_m025 ... speed_m100
STATE_LAYOUT_DIMS = {"accel": 4, "speedhist": 3 + len(SPEED_HISTORY)}
LEAD_DIMS, LEAD_MAX_M = 3, 40.0                              # optional lead-vehicle input (il/lead_input.py), 2026-11-15


def lead_feats(gap, dv):
    """(N, 3) scaled lead-vehicle input: has a lead (0/1), bumper gap / 10 (no lead = 40 m), closing speed / 5 (no lead = 0)."""
    gap, dv = np.atleast_1d(np.asarray(gap, np.float32)), np.atleast_1d(np.asarray(dv, np.float32))
    has = np.isfinite(gap) & (gap < LEAD_MAX_M)
    return np.stack([has.astype(np.float32), np.where(has, gap, LEAD_MAX_M) / 10.0, np.where(has, dv, 0.0) / 5.0], 1).astype(np.float32)


def make_state(layout, speed, past, accel, yaw_rate, steer):
    """Scaled state rows (N, dims). speed, accel, yaw_rate, steer: (N,); past: (N, 4) own speed 0.25 ... 1.0 s ago."""
    speed, yaw_rate, steer = (np.asarray(x, np.float32) for x in (speed, yaw_rate, steer))
    if layout == "accel":
        return np.stack([speed / 10.0, np.asarray(accel, np.float32) / 5.0, yaw_rate, steer / 30.0], 1)
    past = np.asarray(past, np.float32).reshape(len(speed), len(SPEED_HISTORY))
    dv = [(speed - past[:, j]) / (1.25 * h) for j, h in enumerate(SPEED_HISTORY)]
    return np.stack([speed / 10.0] + dv + [yaw_rate, steer / 30.0], 1)
WP_DT = 0.25


class ILDriver(nn.Module):
    def __init__(self, cam_ch=256, lidar_ch=48, use_route=True, cv_prior=True, dropout=0.0, feat_drop=0.0,
                 n_modes=N_MODES, width=1.0, use_cam=True, use_lidar=True, state_layout="accel", lead=False):
        super().__init__()
        self.lead = lead       # state input carries lead_feats (3 more columns, after the state columns)
        self.state_layout = state_layout
        self.use_route, self.cv_prior, self.use_cam, self.use_lidar = use_route, cv_prior, use_cam, use_lidar
        self.k = n_modes
        wc, wl, wf, wg = (max(16, int(round(x * width))) for x in (256, 512, 512, 256))   # width 1 = the original sizes
        self.wg = wg
        self.feat_drop = nn.Dropout2d(feat_drop)   # drops whole encoder channels of the cached features (training only)
        self.drop = nn.Dropout(dropout)            # on the fused feature, before and after the fusion MLP
        c = cam_ch * HISTORY
        self.cam = nn.Sequential(
            nn.Conv2d(c, wc, 1), nn.ReLU(), nn.Conv2d(wc, wc, 3, stride=2, padding=1), nn.ReLU(),     # 8x8 -> 4x4
            nn.Flatten(), nn.Linear(wc * 16, wc), nn.ReLU())
        lc = lidar_ch * HISTORY
        self.lidar = nn.Sequential(
            nn.Conv2d(lc, wl // 4, 3, padding=1), nn.ReLU(),
            nn.Conv2d(wl // 4, wl // 4, 3, stride=2, padding=1), nn.ReLU(),                          # 20x16 -> 10x8
            nn.Conv2d(wl // 4, wl // 4, 3, stride=2, padding=1), nn.ReLU(),                          # -> 5x4
            nn.Flatten(), nn.Linear(wl // 4 * 20, wl), nn.ReLU())
        self.state = nn.Sequential(nn.Linear(STATE_LAYOUT_DIMS[state_layout] + (LEAD_DIMS if lead else 0) + len(ROUTE_CMDS) + len(VEHICLE_TYPES), 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(wc + wl + 64, wf), nn.ReLU(), nn.Linear(wf, wf), nn.ReLU())
        self.mode_emb = nn.Embedding(n_modes, wf)
        self.h0 = nn.Linear(wf, 2 * wg)
        self.gru = nn.GRU(input_size=2 + wf, hidden_size=wg, num_layers=2, batch_first=True)
        self.out = nn.Linear(wg, 2)
        self.score = nn.Linear(wf, n_modes)
        self.behaviour = nn.Linear(wf, len(BEHAVIOURS))

    def forward(self, cam, lidar, state, route, vtype):
        """cam (B, 3, 256, 8, 8), lidar (B, 3, 48, 20, 16), state (B, 4 or 7, see make_state), route (B, 4) one-hot, vtype (B, 5) one-hot.
        Returns waypoints (B, K, 10, 2) in the ego frame (x forward, y left, metres), mode logits (B, K),
        behaviour logits (B, 7)."""
        b = cam.shape[0]
        if not self.use_route:
            route = torch.zeros_like(route)
        if not self.use_cam:          # input ablations: the head still runs, on zeros
            cam = torch.zeros_like(cam)
        if not self.use_lidar:
            lidar = torch.zeros_like(lidar)
        f = torch.cat([self.cam(self.feat_drop(cam.flatten(1, 2))), self.lidar(self.feat_drop(lidar.flatten(1, 2))),
                       self.state(torch.cat([state, route, vtype], 1))], 1)
        f = self.drop(self.fuse(self.drop(f)))                              # (B, 512)
        fm = f[:, None] + self.mode_emb.weight[None]                         # (B, K, 512)
        fm = fm.reshape(b * self.k, -1)
        h = self.h0(fm).reshape(-1, 2, self.wg).transpose(0, 1).contiguous() # (2, B*K, 256)
        wp, prev = [], torch.zeros(b * self.k, 2, device=cam.device)
        for _ in range(N_WP):
            o, h = self.gru(torch.cat([prev, fm], 1)[:, None], h)
            prev = prev + self.out(o[:, 0])                                  # predict the step, accumulate
            wp.append(prev)
        wp = torch.stack(wp, 1).reshape(b, self.k, N_WP, 2)
        if self.cv_prior:
            t = torch.arange(1, N_WP + 1, device=cam.device) * WP_DT
            speed = state[:, 0:1] * float(STATE_SCALE[0])                    # state is scaled by STATE_SCALE
            wp = wp + torch.stack([speed * t, torch.zeros_like(speed * t)], -1)[:, None]
        return wp, self.score(f), self.behaviour(f)


def wta_loss(wp, scores, target, weight=None, score_w=0.5, along_w=1.0):
    """Winner-takes-all: L1 of the best mode (by mean distance) + cross-entropy that the scores pick that mode.
    along_w != 1: the error is split into the part along the target path (its direction at that waypoint; straight ahead
    while it hardly moves) and across it, and the along part is weighted along_w (closed loop the LHD model ran
    1.25-1.37 x the expert's speed: too-long plans cost little in the plain L1)."""
    e = wp - target[:, None]                                                  # (B, K, 10, 2)
    if along_w == 1.0:
        err = e.abs().sum(-1).mean(-1)                                        # (B, K) mean L1 per mode
    else:
        prev = torch.cat([torch.zeros_like(target[:, :1]), target[:, :-1]], 1)
        d = target - prev
        n = d.norm(dim=-1, keepdim=True)
        tan = torch.where(n > 0.05, d / n.clamp_min(1e-6), torch.tensor([1.0, 0.0], device=d.device))[:, None]
        along = (e * tan).sum(-1)
        across = e[..., 0] * -tan[..., 1] + e[..., 1] * tan[..., 0]
        err = (along_w * along.abs() + across.abs()).mean(-1)
    best = err.argmin(1)
    reg = err.gather(1, best[:, None])[:, 0]
    cls = nn.functional.cross_entropy(scores, best, reduction="none")
    loss = reg + score_w * cls
    if weight is not None:
        loss = loss * weight
    return loss.mean(), best
