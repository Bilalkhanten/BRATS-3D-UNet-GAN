"""Analysis of training runs (used for the phantom verification study *and* for the BraTS results).

All numbers that appear in the results chapter are produced here from the files that the training
code writes (val_metrics.csv, test_metrics.csv, train_log.jsonl, config.json).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bratsgan import stats as S  # noqa: E402

REG = ("WT", "TC", "ET")


# ------------------------------------------------------------------ loading
def run_dir(runs: Path, name: str):
    d = Path(runs) / name
    return d if (d / "test_metrics.csv").exists() else None


def load_metrics(dirs, split="test"):
    """Per-patient metrics averaged over the given run directories (seeds or fold models)."""
    frames = []
    for d in dirs:
        f = Path(d) / f"{split}_metrics.csv"
        if not f.exists():
            return None
        frames.append(pd.read_csv(f).set_index("id"))
    ids = frames[0].index
    for f in frames[1:]:
        ids = ids.intersection(f.index)
    num = [c for c in frames[0].columns if c not in ("grade",) and pd.api.types.is_numeric_dtype(frames[0][c])]
    mean = sum(f.loc[ids, num] for f in frames) / len(frames)
    mean["grade"] = frames[0].loc[ids, "grade"]
    return add_composite(mean)


def add_composite(df):
    df = df.copy()
    df["dice_mean"] = df[[f"dice_{r}" for r in REG]].mean(axis=1)
    if all(f"hd95_{r}" in df for r in REG):
        df["hd95_mean"] = df[[f"hd95_{r}" for r in REG]].mean(axis=1)
    return df


def read_log(d):
    f = Path(d) / "train_log.jsonl"
    if not f.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(x) for x in f.read_text().splitlines() if x.strip()])


# --------------------------------------------------------------- statistics
def ci(x, n_boot=10_000, seed=0):
    m, lo, hi = S.bootstrap_ci(np.asarray(x, float), n_boot=n_boot, seed=seed)
    return m, lo, hi


def paired(a: pd.DataFrame, b: pd.DataFrame, col: str, n_boot=10_000):
    ids = a.index.intersection(b.index)
    x, y = a.loc[ids, col].values, b.loc[ids, col].values
    d, lo, hi = S.paired_bootstrap_diff(x, y, n_boot=n_boot)
    return {"n": int(len(ids)), "diff": d, "lo": lo, "hi": hi, "p": S.wilcoxon_paired(x, y),
            "cliff": S.cliffs_delta(x, y), "dz": S.cohens_dz(x, y)}


def fmt(m, nd=3):
    return f"{m:.{nd}f}"


def fmt_ci(t, nd=3):
    m, lo, hi = t
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def sig_mark(lo, hi):
    return "+" if lo > 0 else ("-" if hi < 0 else "0")


# ----------------------------------------------------------------- grade classification
def grade_rows(J: dict):
    """Rows (label, cv_auc, test_auc, acc, sens, spec, bacc, ece) from the JSON written by scripts/grade_classification.py."""
    names = {"pred_logreg": "Predicted masks, logistic regression", "pred_gboost": "Predicted masks, gradient boosting",
             "gt_logreg": "Ground-truth masks, logistic regression", "gt_gboost": "Ground-truth masks, gradient boosting"}
    rows = []
    for k, lab in names.items():
        if k not in J:
            continue
        r = J[k]
        t, c = r["test"], r["cv"]
        rows.append((lab, f"{c['auc']:.3f} [{c['auc_ci'][0]:.3f}, {c['auc_ci'][1]:.3f}]", f"{t['auc']:.3f} [{t['auc_ci'][0]:.3f}, {t['auc_ci'][1]:.3f}]",
                     f"{t['accuracy']:.3f}", f"{t['sensitivity']:.3f}", f"{t['specificity']:.3f}", f"{t['balanced_accuracy']:.3f}", f"{r['test_ece']:.3f}"))
    return rows


def grade_table_md(J: dict, label: str, caption: str) -> str:
    rows = grade_rows(J)
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return (f"@tab {label} | widths=2.0,1.35,1.35,0.6,0.6,0.6,0.6,0.5 | align=lccccccc | size=7.5\n{caption}\n"
            "| Features from | CV AUC [95% CI] | Test AUC [95% CI] | Acc | Sens | Spec | Bal. acc | ECE |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n"
            f"{body}\n@end\n")
