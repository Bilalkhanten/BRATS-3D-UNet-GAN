"""Statistics: bootstrap CIs, paired tests with multiplicity correction, effect sizes, classification metrics."""
from __future__ import annotations
from typing import Dict, Sequence, Tuple
import numpy as np
from scipy import stats as sst


def bootstrap_ci(x: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0, stat=np.mean) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = stat(x[idx], axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(x)), float(lo), float(hi)


def paired_bootstrap_diff(a, b, n_boot=10_000, alpha=0.05, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    return bootstrap_ci(d, n_boot, alpha, seed)


def wilcoxon_paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return 1.0
    return float(sst.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)


def holm(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def cliffs_delta(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def cohens_dz(a, b) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(d.mean() / (d.std(ddof=1) + 1e-12))


def roc_auc(y, s) -> float:
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = sst.rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def binary_report(y, prob, thr: float = 0.5, n_boot: int = 5000, seed: int = 0) -> Dict[str, float]:
    y, prob = np.asarray(y).astype(int), np.asarray(prob, float)
    pred = (prob >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    out = {"n": len(y), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
           "accuracy": (tp + tn) / len(y), "sensitivity": sens, "specificity": spec,
           "balanced_accuracy": np.nanmean([sens, spec]), "auc": roc_auc(y, prob)}
    rng = np.random.default_rng(seed)
    aucs, accs = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        aucs.append(roc_auc(y[i], prob[i])); accs.append(((prob[i] >= thr).astype(int) == y[i]).mean())
    out["auc_ci"] = [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]
    out["accuracy_ci"] = [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]
    return out


def expected_calibration_error(y, prob, bins: int = 10) -> float:
    y, prob = np.asarray(y), np.asarray(prob)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob > lo) & (prob <= hi) if lo > 0 else (prob >= lo) & (prob <= hi)
        if m.any():
            ece += m.mean() * abs(y[m].mean() - prob[m].mean())
    return float(ece)
