"""Whole-volume inference: Gaussian-weighted sliding window (+ optional mirror TTA)."""
from __future__ import annotations
import itertools, time
from typing import Sequence
import numpy as np
import torch
import torch.nn.functional as F


def gaussian_importance(patch: Sequence[int], sigma_scale: float = 1.0 / 8) -> torch.Tensor:
    coords = [torch.arange(p, dtype=torch.float32) - (p - 1) / 2 for p in patch]
    g = [torch.exp(-(c ** 2) / (2 * (p * sigma_scale) ** 2)) for c, p in zip(coords, patch)]
    w = g[0][:, None, None] * g[1][None, :, None] * g[2][None, None, :]
    return (w / w.max()).clamp_min(1e-4)


def _starts(n: int, p: int, step: int):
    if n <= p:
        return [0]
    s = list(range(0, n - p + 1, step))
    if s[-1] != n - p:
        s.append(n - p)
    return s


@torch.no_grad()
def sliding_window(model, x: torch.Tensor, patch: Sequence[int] = (128, 128, 128), overlap: float = 0.5,
                   tta: bool = False, amp: bool = True, device=None, batch: int = 1) -> torch.Tensor:
    """x: (C,D,H,W) normalised float tensor (CPU or GPU). Returns probabilities (R,D,H,W) on CPU."""
    model.eval()
    device = device or next(model.parameters()).device
    C, D, H, W = x.shape
    P = tuple(patch)
    pad = [max(0, p - s) for p, s in zip(P, (D, H, W))]
    xp = F.pad(x[None], (0, pad[2], 0, pad[1], 0, pad[0]))[0] if any(pad) else x
    Dp, Hp, Wp = xp.shape[1:]
    steps = [max(1, int(p * (1 - overlap))) for p in P]
    imp = gaussian_importance(P).to(device)
    acc, wsum = None, torch.zeros((Dp, Hp, Wp), device="cpu")
    flips = [()] if not tta else [f for r in range(4) for f in itertools.combinations((2, 3, 4), r)]
    use_amp = amp and device.type == "cuda"
    for z in _starts(Dp, P[0], steps[0]):
        for y in _starts(Hp, P[1], steps[1]):
            for w in _starts(Wp, P[2], steps[2]):
                patch_x = xp[:, z:z + P[0], y:y + P[1], w:w + P[2]][None].to(device, dtype=torch.float32)
                prob = 0
                for fl in flips:
                    inp = patch_x.flip(fl) if fl else patch_x
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                        out = model(inp)
                    out = torch.sigmoid(out.float())
                    prob = prob + (out.flip(fl) if fl else out)
                prob = (prob / len(flips))[0] * imp
                if acc is None:
                    acc = torch.zeros((prob.shape[0], Dp, Hp, Wp))
                acc[:, z:z + P[0], y:y + P[1], w:w + P[2]] += prob.cpu()
                wsum[z:z + P[0], y:y + P[1], w:w + P[2]] += imp.cpu()
    out = acc / wsum.clamp_min(1e-8)
    return out[:, :D, :H, :W]


def predict_regions(model, img: np.ndarray, patch, overlap=0.5, tta=False, thr=0.5, device=None):
    """Returns (binary regions uint8 (3,D,H,W), probabilities float16, seconds)."""
    t0 = time.perf_counter()
    x = torch.from_numpy(np.asarray(img, dtype=np.float32))
    prob = sliding_window(model, x, patch, overlap, tta, device=device)
    if (device is not None and device.type == "cuda") or torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return (prob.numpy() > thr).astype(np.uint8), prob.numpy().astype(np.float16), dt
