#!/usr/bin/env python3
"""
il/pointpillars.py

Pure-PyTorch PointPillars LiDAR encoder that loads the official mmdetection3d nuScenes checkpoint
(hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d, il/weights/pointpillars_nus_secfpn_f19d00a3.pth) without
installing mmcv / mmdet3d (their compiled ops don't match torch 2.13 in carla_env).

Same layers as the mmdet3d config: HardVFE (2 x 64, cluster + voxel-centre offsets) -> PointPillarsScatter
(400 x 400, 0.25 m pillars, x/y in [-50, 50] m, z in [-5, 3] m) -> SECOND [64, 128, 256] -> SECONDFPN
(3 x 128). Output: BEV feature map (B, 384, 200, 200), 0.5 m per cell. The detection head is only used
to verify the weights (il/test_pointpillars_carla.py); the driving model drops it (ML_DECISIONS.md M8/M9).

Input points: (N, 4) x, y, z, intensity in a right-handed sensor frame (z up), as nuScenes LIDAR_TOP.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VOXEL = (0.25, 0.25, 8.0)
PC_RANGE = (-50.0, -50.0, -5.0, 50.0, 50.0, 3.0)
GRID = (400, 400)              # ny, nx
MAX_POINTS = 64
MAX_VOXELS = 40000             # test-time value of the config
BN = dict(eps=1e-3, momentum=0.01)
CLASSES = ["car", "truck", "trailer", "bus", "construction_vehicle", "bicycle", "motorcycle",
           "pedestrian", "traffic_cone", "barrier"]
# anchor z and size (x, y, z) per range, from pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py
ANCHOR_Z = [-1.80032795, -1.74440365, -1.68526504, -1.67339111, -1.61785072, -1.80984986, -1.763965]
ANCHOR_SIZES = [[4.60718145, 1.95017717, 1.72270761], [6.73778078, 2.4560939, 2.73004906],
                [12.01320693, 2.87427237, 3.81509561], [1.68452161, 0.60058911, 1.27192197],
                [0.7256437, 0.66344886, 1.75748069], [0.40359262, 0.39694519, 1.06232151],
                [0.48578221, 2.49008838, 0.98297065]]


# ==========================================
# VOXELISATION
# ==========================================
def voxelize(points):
    """Hard voxelisation: up to MAX_POINTS points per pillar, up to MAX_VOXELS pillars.
    Returns voxels (M, MAX_POINTS, 4), num_points (M,), coors (M, 4) = (batch, z, y, x)."""
    x0, y0, z0, x1, y1, z1 = PC_RANGE
    keep = ((points[:, 0] >= x0) & (points[:, 0] < x1) & (points[:, 1] >= y0) & (points[:, 1] < y1)
            & (points[:, 2] >= z0) & (points[:, 2] < z1))
    pts = points[keep]
    ix = ((pts[:, 0] - x0) / VOXEL[0]).long()
    iy = ((pts[:, 1] - y0) / VOXEL[1]).long()
    vid = iy * GRID[1] + ix
    vid_sorted, order = torch.sort(vid, stable=True)
    pts = pts[order]
    uniq, inverse, counts = torch.unique_consecutive(vid_sorted, return_inverse=True, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(len(pts), device=pts.device) - starts[inverse]
    sel = (rank < MAX_POINTS) & (inverse < MAX_VOXELS)
    uniq, counts = uniq[:MAX_VOXELS], counts[:MAX_VOXELS].clamp(max=MAX_POINTS)
    voxels = pts.new_zeros((len(uniq), MAX_POINTS, pts.shape[1]))
    voxels[inverse[sel], rank[sel]] = pts[sel]
    coors = torch.stack([torch.zeros_like(uniq), torch.zeros_like(uniq), uniq // GRID[1], uniq % GRID[1]], 1)
    return voxels, counts, coors


# ==========================================
# NETWORK
# ==========================================
class VFELayer(nn.Module):
    def __init__(self, cin, cout, cat_max):
        super().__init__()
        self.cat_max = cat_max
        self.norm = nn.BatchNorm1d(cout, **BN)
        self.linear = nn.Linear(cin, cout, bias=False)

    def forward(self, x):
        x = F.relu(self.norm(self.linear(x).permute(0, 2, 1)).permute(0, 2, 1))
        agg = x.max(dim=1, keepdim=True)[0]
        return torch.cat([x, agg.expand_as(x)], 2) if self.cat_max else agg.squeeze(1)


class HardVFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.vfe_layers = nn.ModuleList([VFELayer(10, 64, True), VFELayer(128, 64, False)])

    def forward(self, voxels, num_points, coors):
        mean = voxels[:, :, :3].sum(1, keepdim=True) / num_points.type_as(voxels).view(-1, 1, 1)
        centre = torch.stack([coors[:, 3] * VOXEL[0] + VOXEL[0] / 2 + PC_RANGE[0],
                              coors[:, 2] * VOXEL[1] + VOXEL[1] / 2 + PC_RANGE[1],
                              coors[:, 1] * VOXEL[2] + VOXEL[2] / 2 + PC_RANGE[2]], 1).type_as(voxels)
        feats = torch.cat([voxels, voxels[:, :, :3] - mean, voxels[:, :, :3] - centre.unsqueeze(1)], -1)
        mask = torch.arange(voxels.shape[1], device=voxels.device)[None] < num_points[:, None]
        x = feats * mask.unsqueeze(-1).type_as(feats)
        for layer in self.vfe_layers:
            x = layer(x)
        return x


class SECOND(nn.Module):
    def __init__(self, cin=64, layer_nums=(3, 5, 5), strides=(2, 2, 2), couts=(64, 128, 256)):
        super().__init__()
        blocks, c = [], cin
        for n, s, co in zip(layer_nums, strides, couts):
            layers = [nn.Conv2d(c, co, 3, stride=s, padding=1, bias=False), nn.BatchNorm2d(co, **BN), nn.ReLU()]
            for _ in range(n):
                layers += [nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co, **BN), nn.ReLU()]
            blocks.append(nn.Sequential(*layers))
            c = co
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        outs = []
        for b in self.blocks:
            x = b(x)
            outs.append(x)
        return outs


class SECONDFPN(nn.Module):
    def __init__(self, cins=(64, 128, 256), ups=(1, 2, 4), cout=128):
        super().__init__()
        self.deblocks = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(ci, cout, u, stride=u, bias=False), nn.BatchNorm2d(cout, **BN), nn.ReLU())
            for ci, u in zip(cins, ups)])

    def forward(self, xs):
        return torch.cat([d(x) for d, x in zip(self.deblocks, xs)], 1)


class Anchor3DHead(nn.Module):
    """nuScenes head (14 anchors x 10 classes per cell); only used to check the pretrained weights."""

    def __init__(self, cin=384, n_anchors=14, n_classes=10, code=9):
        super().__init__()
        self.conv_cls = nn.Conv2d(cin, n_anchors * n_classes, 1)
        self.conv_reg = nn.Conv2d(cin, n_anchors * code, 1)
        self.conv_dir_cls = nn.Conv2d(cin, n_anchors * 2, 1)


class PointPillarsEncoder(nn.Module):
    def __init__(self, with_head=False):
        super().__init__()
        self.pts_voxel_encoder = HardVFE()
        self.pts_backbone = SECOND()
        self.pts_neck = SECONDFPN()
        self.pts_bbox_head = Anchor3DHead() if with_head else None

    def bev(self, points):
        """points (N, 4) -> BEV features (1, 384, 200, 200)."""
        voxels, num, coors = voxelize(points)
        feats = self.pts_voxel_encoder(voxels, num, coors)
        canvas = feats.new_zeros(feats.shape[1], GRID[0] * GRID[1])
        canvas[:, coors[:, 2] * GRID[1] + coors[:, 3]] = feats.t()
        x = canvas.view(1, feats.shape[1], *GRID)
        return self.pts_neck(self.pts_backbone(x))

    def forward(self, points):
        return self.bev(points)

    def detect(self, points, score_thr=0.3, top_k=200):
        """Decoded boxes (x, y, z, w, l, h, yaw), scores, class ids; no NMS (verification only)."""
        f = self.bev(points)
        h = self.pts_bbox_head
        H, W = f.shape[-2:]
        cls = h.conv_cls(f).permute(0, 2, 3, 1).reshape(-1, len(CLASSES)).sigmoid()
        reg = h.conv_reg(f).permute(0, 2, 3, 1).reshape(-1, 9)
        anchors = make_anchors(H, W, f.device)
        score, label = cls.max(1)
        idx = torch.nonzero(score > score_thr).squeeze(1)
        idx = idx[score[idx].argsort(descending=True)[:top_k]]
        return decode(anchors[idx], reg[idx]), score[idx], label[idx]


def make_anchors(H, W, device):
    """AlignedAnchor3DRangeGenerator (align_corner=False, reshape_out=True): order (H, W, 7 sizes, 2 rots)."""
    xs = torch.linspace(-49.6, 49.6, W + 1, device=device)
    ys = torch.linspace(-49.6, 49.6, H + 1, device=device)
    xs = (xs + (xs[1] - xs[0]) / 2)[:W]
    ys = (ys + (ys[1] - ys[0]) / 2)[:H]
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    out = []
    for z, (sx, sy, sz) in zip(ANCHOR_Z, ANCHOR_SIZES):
        for rot in (0.0, math.pi / 2):
            a = torch.stack([xx, yy, torch.full_like(xx, z), torch.full_like(xx, sx), torch.full_like(xx, sy),
                             torch.full_like(xx, sz), torch.full_like(xx, rot)], -1)
            out.append(a)
    return torch.stack(out, 2).reshape(-1, 7)


def decode(a, d):
    """DeltaXYZWLHRBBoxCoder.decode for the first 7 values (velocity ignored)."""
    xa, ya, za, wa, la, ha, ra = a.unbind(-1)
    xt, yt, zt, wt, lt, ht, rt = d[:, :7].unbind(-1)
    za = za + ha / 2
    diag = torch.sqrt(la ** 2 + wa ** 2)
    hg = torch.exp(ht) * ha
    return torch.stack([xt * diag + xa, yt * diag + ya, zt * ha + za - hg / 2,
                        torch.exp(wt) * wa, torch.exp(lt) * la, hg, rt + ra], -1)


def load_encoder(path, with_head=False, device="cpu"):
    """Loads the mmdet3d checkpoint (weights_only, so no pickled code runs)."""
    sd = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
    m = PointPillarsEncoder(with_head=with_head)
    keep = {k: v for k, v in sd.items() if with_head or not k.startswith("pts_bbox_head")}
    m.load_state_dict(keep, strict=True)
    return m.to(device).eval()
