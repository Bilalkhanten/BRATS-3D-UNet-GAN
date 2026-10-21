"""BraTS 2020 data handling: discovery, pre-processing cache, patient-level splits, patch sampling.

Label convention (BraTS 2020): 1 = necrotic/non-enhancing core (NCR/NET), 2 = peritumoural edema (ED),
4 = enhancing tumour (ET).  Evaluation regions: WT = {1,2,4}, TC = {1,4}, ET = {4}.

Important design decisions (they differ deliberately from the published study this thesis extends):
  * every split is made at *patient* level and stratified by tumour grade (HGG/LGG);
  * no crop is ever centred on the ground-truth tumour at inference time: the whole brain is
    segmented with a sliding window; tumour-centred sampling is used only to draw *training* patches;
  * BraTS contains no tumour-free subjects, so classification is HGG-vs-LGG grading, not tumour/no-tumour.
"""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from . import MODALITIES
from .utils import read_json, write_json

_SUFFIX = {"t1": ["_t1.nii.gz", "_t1.nii"], "t1ce": ["_t1ce.nii.gz", "_t1ce.nii"],
           "t2": ["_t2.nii.gz", "_t2.nii"], "flair": ["_flair.nii.gz", "_flair.nii"]}


# ------------------------------------------------------------------ discovery
def labels_to_regions(seg: np.ndarray) -> np.ndarray:
    """(D,H,W) BraTS labels -> (3,D,H,W) uint8 binary masks [WT, TC, ET]."""
    wt = seg > 0
    tc = (seg == 1) | (seg == 4)
    et = seg == 4
    return np.stack([wt, tc, et]).astype(np.uint8)


def find_training_root(root: str) -> Path:
    root = Path(root)
    for cand in [root] + [p for p in root.rglob("*") if p.is_dir()][:2000]:
        if any(cand.glob("BraTS20_Training_*")) or any(cand.glob("BraTS*_Training_*")):
            return cand
    raise FileNotFoundError(f"no BraTS training folders under {root}")


def read_grades(train_root: Path) -> Dict[str, str]:
    """Map BraTS20 subject id -> 'HGG' | 'LGG' from name_mapping.csv (shipped with BraTS 2020)."""
    grades: Dict[str, str] = {}
    for f in list(train_root.glob("name_mapping.csv")) + list(train_root.parent.glob("name_mapping.csv")):
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                sid = row.get("BraTS_2020_subject_ID") or row.get("Brats20ID")
                g = (row.get("Grade") or "").strip().upper()
                if sid and g in ("HGG", "LGG"):
                    grades[sid] = g
        if grades:
            break
    return grades


def find_cases(root: str) -> List[dict]:
    tr = find_training_root(root)
    grades = read_grades(tr)
    cases = []
    for d in sorted(p for p in tr.iterdir() if p.is_dir() and "Training" in p.name):
        files = {m: next((str(d / (d.name + s)) for s in _SUFFIX[m] if (d / (d.name + s)).exists()), None) for m in MODALITIES}
        seg = [str(p) for p in d.iterdir() if ("seg" in p.name.lower() or "segm" in p.name.lower()) and p.suffix in (".gz", ".nii")]
        if any(v is None for v in files.values()) or not seg:
            print(f"[warn] incomplete case skipped: {d.name}")
            continue
        cases.append({"id": d.name, "paths": files, "seg": seg[0], "grade": grades.get(d.name)})
    return cases


# ------------------------------------------------------------- preprocessing
def normalise_volume(vol: np.ndarray, brain: np.ndarray, pct=(0.5, 99.5)) -> np.ndarray:
    """Clip to the [0.5, 99.5] percentile range of in-brain voxels, then z-score in-brain; outside brain = 0."""
    out = np.zeros_like(vol, dtype=np.float32)
    v = vol[brain].astype(np.float32)
    if v.size == 0:
        return out
    lo, hi = np.percentile(v, pct)
    v = np.clip(v, lo, hi)
    mu, sd = v.mean(), v.std() + 1e-8
    out[brain] = (v - mu) / sd
    return out


def preprocess_case(case: dict, out_dir: Path, margin: int = 4, n_fg_coords: int = 4096, seed: int = 0) -> dict:
    import nibabel as nib
    imgs, spacing = [], (1.0, 1.0, 1.0)
    for m in MODALITIES:
        nii = nib.load(case["paths"][m])
        imgs.append(np.asarray(nii.dataobj, dtype=np.float32))
        spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])
    img = np.stack(imgs)                                        # (4,D,H,W)
    seg = np.asarray(nib.load(case["seg"]).dataobj).astype(np.uint8)
    brain = (img != 0).any(axis=0)
    idx = np.argwhere(brain)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + margin + 1, brain.shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    img, seg, brain = img[(slice(None),) + sl], seg[sl], brain[sl]
    img = np.stack([normalise_volume(img[m], brain) for m in range(img.shape[0])])
    rng = np.random.default_rng(seed)
    fg = np.argwhere(seg > 0)
    if len(fg) > n_fg_coords:
        fg = fg[rng.choice(len(fg), n_fg_coords, replace=False)]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{case['id']}_img.npy", img.astype(np.float16))
    np.save(out_dir / f"{case['id']}_seg.npy", seg)
    np.save(out_dir / f"{case['id']}_fg.npy", fg.astype(np.int32))
    meta = {"id": case["id"], "grade": case["grade"], "bbox_lo": lo.tolist(), "bbox_hi": hi.tolist(),
            "orig_shape": list(brain.shape), "shape": list(seg.shape), "spacing": spacing}
    write_json(out_dir / f"{case['id']}_meta.json", meta)
    return meta


# --------------------------------------------------------------------- splits
def make_splits(cases: Sequence[dict], seed: int = 42, test_frac: float = 0.2, k: int = 5) -> dict:
    """Patient-level, grade-stratified: a held-out test set, then k folds over the remainder."""
    rng = np.random.default_rng(seed)
    by_grade: Dict[str, List[str]] = {}
    for c in cases:
        by_grade.setdefault(c.get("grade") or "UNK", []).append(c["id"])
    test, folds = [], [[] for _ in range(k)]
    for g, ids in sorted(by_grade.items()):
        ids = list(ids)
        rng.shuffle(ids)
        n_test = int(round(test_frac * len(ids)))
        test += ids[:n_test]
        for j, cid in enumerate(ids[n_test:]):
            folds[j % k].append(cid)
    splits = {"seed": seed, "test": sorted(test), "folds": [sorted(f) for f in folds]}
    assert_no_leakage(splits)
    return splits


def assert_no_leakage(splits: dict) -> None:
    seen = set(splits["test"])
    assert len(seen) == len(splits["test"]), "duplicate in test"
    for f in splits["folds"]:
        s = set(f)
        assert not (s & seen), f"patient overlap between partitions: {sorted(s & seen)[:3]}"
        seen |= s


def fold_ids(splits: dict, fold: int) -> Tuple[List[str], List[str], List[str]]:
    """(train, val, test) ids for one CV fold. Test is the common held-out set."""
    val = splits["folds"][fold]
    train = [i for j, f in enumerate(splits["folds"]) if j != fold for i in f]
    return train, val, splits["test"]


# ------------------------------------------------------------------- loading
def load_case(cache_dir, cid: str, mmap: bool = True):
    cache_dir = Path(cache_dir)
    img = np.load(cache_dir / f"{cid}_img.npy", mmap_mode="r" if mmap else None)
    seg = np.load(cache_dir / f"{cid}_seg.npy", mmap_mode="r" if mmap else None)
    meta = read_json(cache_dir / f"{cid}_meta.json")
    return img, seg, meta


class PatchStream(IterableDataset):
    """Infinite stream of training patches (image, region-masks, grade).

    With probability fg_prob a patch is centred (with jitter) on a random tumour voxel;
    otherwise its location is uniform over the brain bounding box."""

    def __init__(self, cache_dir, ids: Sequence[str], patch: Sequence[int] = (128, 128, 128),
                 fg_prob: float = 0.67, seed: int = 0, jitter: int = 16):
        self.cache_dir, self.ids, self.patch = Path(cache_dir), list(ids), tuple(patch)
        self.fg_prob, self.seed, self.jitter = fg_prob, seed, jitter
        self.meta = {i: read_json(self.cache_dir / f"{i}_meta.json") for i in self.ids}

    def _crop(self, arr, lo, hi_shape, pad_value=0):
        P = self.patch
        out_shape = arr.shape[:-3] + P
        out = np.full(out_shape, pad_value, dtype=arr.dtype)
        src = [slice(max(l, 0), min(l + p, s)) for l, p, s in zip(lo, P, arr.shape[-3:])]
        dst = [slice(max(0, -l), max(0, -l) + (sl.stop - sl.start)) for l, sl in zip(lo, src)]
        out[(Ellipsis,) + tuple(dst)] = arr[(Ellipsis,) + tuple(src)]
        return out

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng(self.seed * 1000 + wid)
        P = np.array(self.patch)
        while True:
            cid = self.ids[rng.integers(len(self.ids))]
            img, seg, meta = load_case(self.cache_dir, cid)
            fg = np.load(self.cache_dir / f"{cid}_fg.npy")
            shape = np.array(seg.shape)
            if len(fg) and rng.random() < self.fg_prob:
                centre = fg[rng.integers(len(fg))] + rng.integers(-self.jitter, self.jitter + 1, size=3)
            else:
                centre = np.array([rng.integers(0, s) for s in shape])
            lo = (centre - P // 2).astype(int)
            lo = np.clip(lo, -(P // 4), shape - (3 * P // 4))
            x = self._crop(np.asarray(img), lo, shape).astype(np.float32)
            y = labels_to_regions(self._crop(np.asarray(seg), lo, shape))
            g = {"HGG": 1.0, "LGG": 0.0}.get(meta.get("grade"), -1.0)
            yield torch.from_numpy(x), torch.from_numpy(y.astype(np.float32)), torch.tensor(g)
