"""GPU batch augmentation (spatial + intensity). x: (B,C,D,H,W) float, y: (B,R,D,H,W) float in {0,1}."""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def _rand(n, lo, hi, dev):
    return torch.rand(n, device=dev) * (hi - lo) + lo


def random_affine(x, y, p_rot=0.2, p_scale=0.2, max_deg=15.0, scale=(0.85, 1.25)):
    B = x.shape[0]
    dev = x.device
    theta = torch.eye(3, 4, device=dev).unsqueeze(0).repeat(B, 1, 1)
    for b in range(B):
        R = torch.eye(3, device=dev)
        if torch.rand(1).item() < p_rot:
            ax, ay, az = [math.radians(float(v)) for v in _rand(3, -max_deg, max_deg, dev)]
            cx, sx, cy, sy, cz, sz = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay), math.cos(az), math.sin(az)
            Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], device=dev)
            Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], device=dev)
            Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], device=dev)
            R = Rz @ Ry @ Rx
        if torch.rand(1).item() < p_scale:
            R = R / float(_rand(1, *scale, dev))
        theta[b, :, :3] = R
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    xo = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    yo = F.grid_sample(y, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    return xo, yo


def random_flip(x, y):
    for ax in (2, 3, 4):
        flip = torch.rand(x.shape[0], device=x.device) < 0.5
        if flip.any():
            x = torch.where(flip.view(-1, 1, 1, 1, 1), x.flip(ax), x)
            y = torch.where(flip.view(-1, 1, 1, 1, 1), y.flip(ax), y)
    return x, y


def intensity_aug(x, p_noise=0.1, p_bright=0.15, p_contrast=0.15, p_gamma=0.15):
    B, C = x.shape[:2]
    dev = x.device
    x = x.clone()
    for b in range(B):
        if torch.rand(1).item() < p_noise:
            x[b] += torch.randn_like(x[b]) * float(_rand(1, 0.0, 0.1, dev))
        for c in range(C):
            if torch.rand(1).item() < p_bright:
                x[b, c] *= float(_rand(1, 0.75, 1.25, dev))
            if torch.rand(1).item() < p_contrast:
                m = x[b, c].mean()
                x[b, c] = (x[b, c] - m) * float(_rand(1, 0.75, 1.25, dev)) + m
            if torch.rand(1).item() < p_gamma:
                v = x[b, c]
                lo, hi = v.min(), v.max()
                rng = (hi - lo).clamp_min(1e-6)
                g = float(_rand(1, 0.7, 1.5, dev))
                x[b, c] = ((v - lo) / rng).pow(g) * rng + lo
    return x


def augment_batch(x, y):
    x, y = random_flip(x, y)
    x, y = random_affine(x, y)
    x = intensity_aug(x)
    return x, y
