"""Segmentation-driven radiomic-style features for LGG/HGG classification.

Features are computed from the PREDICTED masks (never the ground truth), so the grade classifier
sees exactly what a deployed system would see.
"""
from __future__ import annotations
from typing import Dict
import numpy as np
from scipy import ndimage as ndi
from . import MODALITIES


def _surface_voxels(mask: np.ndarray) -> int:
    er = ndi.binary_erosion(mask, structure=ndi.generate_binary_structure(3, 1), border_value=0)
    return int((mask & ~er).sum())


def regions_to_compartments(reg: np.ndarray) -> Dict[str, np.ndarray]:
    wt, tc, et = reg[0].astype(bool), reg[1].astype(bool), reg[2].astype(bool)
    return {"WT": wt, "TC": tc, "ET": et, "NCR": tc & ~et, "ED": wt & ~tc}


def extract_features(img: np.ndarray, reg: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> Dict[str, float]:
    """img: (4,D,H,W) normalised; reg: (3,D,H,W) binary [WT,TC,ET]."""
    vox = float(np.prod(spacing))
    comp = regions_to_compartments(reg)
    f: Dict[str, float] = {}
    vols = {k: float(m.sum()) * vox for k, m in comp.items()}
    for k, v in vols.items():
        f[f"logvol_{k}"] = float(np.log1p(v))
    eps = 1e-6
    f["ratio_ET_WT"] = vols["ET"] / (vols["WT"] + eps)
    f["ratio_ET_TC"] = vols["ET"] / (vols["TC"] + eps)
    f["ratio_TC_WT"] = vols["TC"] / (vols["WT"] + eps)
    f["ratio_NCR_TC"] = vols["NCR"] / (vols["TC"] + eps)
    f["ratio_ED_WT"] = vols["ED"] / (vols["WT"] + eps)
    for k in ("WT", "TC", "ET"):
        m = comp[k]
        A = _surface_voxels(m) * (vox ** (2 / 3))
        V = vols[k]
        f[f"sphericity_{k}"] = float((np.pi ** (1 / 3)) * (6 * V) ** (2 / 3) / A) if A > 0 and V > 0 else 0.0
        if m.any():
            idx = np.argwhere(m)
            ext = idx.max(0) - idx.min(0) + 1
            f[f"extent_{k}"] = float(V / (np.prod(ext) * vox))
            f[f"maxdiam_{k}"] = float(ext.max() * np.mean(spacing))
        else:
            f[f"extent_{k}"] = 0.0
            f[f"maxdiam_{k}"] = 0.0
    for k, m in comp.items():
        for mi, mod in enumerate(MODALITIES):
            v = img[mi][m]
            if v.size:
                f[f"{mod}_{k}_mean"] = float(v.mean())
                f[f"{mod}_{k}_std"] = float(v.std())
                f[f"{mod}_{k}_p10"] = float(np.percentile(v, 10))
                f[f"{mod}_{k}_p90"] = float(np.percentile(v, 90))
            else:
                f[f"{mod}_{k}_mean"] = f[f"{mod}_{k}_std"] = f[f"{mod}_{k}_p10"] = f[f"{mod}_{k}_p90"] = 0.0
    # enhancement contrast: T1ce vs T1 inside ET and TC
    for k in ("ET", "TC"):
        m = comp[k]
        f[f"enh_contrast_{k}"] = float(img[1][m].mean() - img[0][m].mean()) if m.any() else 0.0
    return f


FEATURE_NAMES = None
