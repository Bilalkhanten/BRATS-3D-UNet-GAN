"""Evaluation metrics on binary masks (numpy). Conventions follow the BraTS challenge:
  * both masks empty            -> Dice = 1, HD95 = 0
  * exactly one mask empty      -> Dice = 0, HD95 = HD95_PENALTY  (= diagonal of 240x240x155 volume)
HD95 is the 95th percentile of the *pooled* set of directed surface distances.
"""
from __future__ import annotations
from typing import Dict, Sequence
import numpy as np
from scipy import ndimage as ndi

HD95_PENALTY = float(np.sqrt(240 ** 2 + 240 ** 2 + 155 ** 2))  # 373.13


def confusion(pred: np.ndarray, gt: np.ndarray):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(pred.size - tp - fp - fn)
    return tp, fp, fn, tn


def dice(pred, gt) -> float:
    tp, fp, fn, _ = confusion(pred, gt)
    if tp + fp + fn == 0:
        return 1.0
    return 2.0 * tp / (2 * tp + fp + fn)


def iou(pred, gt) -> float:
    tp, fp, fn, _ = confusion(pred, gt)
    if tp + fp + fn == 0:
        return 1.0
    return tp / (tp + fp + fn)


def precision(pred, gt) -> float:
    tp, fp, fn, _ = confusion(pred, gt)
    if tp + fp == 0:
        return 1.0 if fn == 0 else 0.0
    return tp / (tp + fp)


def sensitivity(pred, gt) -> float:
    tp, fp, fn, _ = confusion(pred, gt)
    if tp + fn == 0:
        return 1.0 if fp == 0 else 0.0
    return tp / (tp + fn)


def specificity(pred, gt) -> float:
    tp, fp, fn, tn = confusion(pred, gt)
    return tn / (tn + fp) if tn + fp > 0 else 1.0


def _surface(mask: np.ndarray) -> np.ndarray:
    er = ndi.binary_erosion(mask, structure=ndi.generate_binary_structure(3, 1), border_value=0)
    return mask & ~er


def hd95(pred, gt, spacing: Sequence[float] = (1.0, 1.0, 1.0), penalty: float = HD95_PENALTY) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    pe, ge = pred.any(), gt.any()
    if not pe and not ge:
        return 0.0
    if pe != ge:
        return penalty
    sp, sg = _surface(pred), _surface(gt)
    d_to_g = ndi.distance_transform_edt(~sg, sampling=spacing)
    d_to_p = ndi.distance_transform_edt(~sp, sampling=spacing)
    d = np.concatenate([d_to_g[sp], d_to_p[sg]])
    return float(np.percentile(d, 95))


def region_metrics(pred_regions: np.ndarray, gt_regions: np.ndarray, names=("WT", "TC", "ET"),
                   spacing=(1.0, 1.0, 1.0), with_hd: bool = True) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i, n in enumerate(names):
        p, g = pred_regions[i].astype(bool), gt_regions[i].astype(bool)
        out[f"dice_{n}"] = dice(p, g)
        out[f"iou_{n}"] = iou(p, g)
        out[f"prec_{n}"] = precision(p, g)
        out[f"sens_{n}"] = sensitivity(p, g)
        out[f"spec_{n}"] = specificity(p, g)
        if with_hd:
            out[f"hd95_{n}"] = hd95(p, g, spacing)
        out[f"vol_gt_{n}"] = float(g.sum())
        out[f"vol_pred_{n}"] = float(p.sum())
    return out


def enforce_hierarchy(regions: np.ndarray) -> np.ndarray:
    """ET subset of TC subset of WT (regions ordered WT, TC, ET)."""
    r = regions.astype(bool).copy()
    r[1] &= r[0]
    r[2] &= r[1]
    return r
