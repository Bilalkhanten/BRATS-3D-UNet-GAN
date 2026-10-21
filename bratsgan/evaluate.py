"""Per-case evaluation with sliding-window inference."""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import pandas as pd
import torch

from . import REGIONS
from .data import labels_to_regions, load_case
from .infer import predict_regions
from .metrics import enforce_hierarchy, region_metrics
from .models import build_model
from .utils import read_json, write_json


def evaluate_ids(model, cache_dir, ids: Sequence[str], patch, overlap=0.5, tta=False, device=None,
                 save_pred_dir: Optional[str] = None, with_hd: bool = True, hierarchy: bool = True,
                 progress: bool = False) -> pd.DataFrame:
    rows = []
    if save_pred_dir:
        Path(save_pred_dir).mkdir(parents=True, exist_ok=True)
    for n, cid in enumerate(ids):
        img, seg, meta = load_case(cache_dir, cid, mmap=False)
        gt = labels_to_regions(np.asarray(seg))
        pred, _, dt = predict_regions(model, img, patch, overlap, tta, device=device)
        if hierarchy:
            pred = enforce_hierarchy(pred).astype(np.uint8)
        m = region_metrics(pred, gt, REGIONS, tuple(meta.get("spacing", (1, 1, 1))), with_hd)
        m.update(id=cid, grade=meta.get("grade"), seconds=dt)
        rows.append(m)
        if save_pred_dir:
            np.save(Path(save_pred_dir) / f"{cid}_pred.npy", pred)
        if progress:
            print(f"[{n + 1}/{len(ids)}] {cid} dice WT/TC/ET = {m['dice_WT']:.3f}/{m['dice_TC']:.3f}/{m['dice_ET']:.3f} ({dt:.1f}s)", flush=True)
    df = pd.DataFrame(rows)
    cols = ["id", "grade"] + [c for c in df.columns if c not in ("id", "grade")]
    return df[cols]


def summarise(df: pd.DataFrame) -> dict:
    out = {}
    for c in df.columns:
        if c in ("id", "grade"):
            continue
        out[c] = {"mean": float(df[c].mean()), "std": float(df[c].std(ddof=1)), "median": float(df[c].median())}
    return out


def load_for_eval(cfg: dict, ckpt: str, device):
    model = build_model(cfg).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory containing config.json and best.pt")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--no-hd", action="store_true")
    a = ap.parse_args()
    run = Path(a.run)
    cfg = read_json(run / "config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_for_eval(cfg, str(run / "best.pt"), device)
    splits = read_json(cfg["splits"])
    from .data import fold_ids
    _, val, test = fold_ids(splits, cfg["fold"])
    ids = val if a.split == "val" else test
    df = evaluate_ids(model, cfg["cache_dir"], ids, cfg["patch"], cfg["final_overlap"], a.tta or cfg["final_tta"], device,
                      save_pred_dir=str(run / f"pred_{a.split}"), with_hd=not a.no_hd, hierarchy=cfg["hierarchy"], progress=True)
    df.to_csv(run / f"{a.split}_metrics.csv", index=False)
    write_json(run / f"{a.split}_summary.json", summarise(df))


if __name__ == "__main__":
    main()
