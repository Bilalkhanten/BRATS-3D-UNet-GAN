"""Aggregate all runs into results/ (tidy CSVs + summary.json) used by the thesis builder.

    python scripts/collect_results.py --runs runs --out results --baseline a0_unet
"""
import argparse, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bratsgan.stats import bootstrap_ci, cliffs_delta, cohens_dz, holm, paired_bootstrap_diff, wilcoxon_paired
from bratsgan.utils import JsonlLogger, write_json

REG = ("WT", "TC", "ET")


def gather(runs: Path):
    exps = {}
    for d in sorted(runs.iterdir()):
        if not d.is_dir() or "_f" not in d.name:
            continue
        exp, fold = d.name.rsplit("_f", 1)
        if not fold.isdigit():
            continue
        exps.setdefault(exp, {})[int(fold)] = d
    return exps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs"); ap.add_argument("--out", default="results"); ap.add_argument("--baseline", default="a0_unet")
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    exps = gather(runs)
    summary = {"experiments": {}, "comparisons": []}
    oof_tables, test_tables = {}, {}
    for exp, folds in exps.items():
        val = [pd.read_csv(d / "val_metrics.csv").assign(fold=k) for k, d in folds.items() if (d / "val_metrics.csv").exists()]
        tst = [pd.read_csv(d / "test_metrics.csv").assign(fold=k) for k, d in folds.items() if (d / "test_metrics.csv").exists()]
        if not val:
            continue
        oof = pd.concat(val, ignore_index=True)
        oof_tables[exp] = oof
        oof.to_csv(out / f"oof_{exp}.csv", index=False)
        e = {"n_folds": len(folds), "oof": {}, "test": {}}
        for m in ("dice", "hd95", "prec", "sens", "iou"):
            for r in REG:
                c = f"{m}_{r}"
                if c in oof:
                    mean, lo, hi = bootstrap_ci(oof[c].values)
                    e["oof"][c] = {"mean": mean, "ci95": [lo, hi], "sd": float(oof[c].std(ddof=1))}
        e["oof"]["seconds"] = {"mean": float(oof["seconds"].mean()), "sd": float(oof["seconds"].std(ddof=1))}
        if tst:
            t = pd.concat(tst, ignore_index=True)
            tc = t.groupby("id").mean(numeric_only=True).reset_index()   # average over the fold models, per case
            test_tables[exp] = tc
            tc.to_csv(out / f"test_{exp}.csv", index=False)
            for m in ("dice", "hd95"):
                for r in REG:
                    c = f"{m}_{r}"
                    mean, lo, hi = bootstrap_ci(tc[c].values)
                    per_fold = t.groupby("fold")[c].mean()
                    e["test"][c] = {"mean": mean, "ci95": [lo, hi], "fold_mean": float(per_fold.mean()), "fold_sd": float(per_fold.std(ddof=1))}
        # training curves
        logs = {k: JsonlLogger(d / "train_log.jsonl").read() for k, d in folds.items()}
        write_json(out / f"curves_{exp}.json", logs)
        summary["experiments"][exp] = e
    # paired comparisons on OOF Dice (same cases across experiments)
    names = list(oof_tables)
    base = a.baseline
    pairs = [(n, base) for n in names if n != base and base in oof_tables]
    pvals, rows = [], []
    for n, b in pairs:
        A_, B_ = oof_tables[n].set_index("id"), oof_tables[b].set_index("id")
        ids = A_.index.intersection(B_.index)
        for r in REG:
            x, y = A_.loc[ids, f"dice_{r}"].values, B_.loc[ids, f"dice_{r}"].values
            d, lo, hi = paired_bootstrap_diff(x, y)
            rows.append({"a": n, "b": b, "region": r, "n": len(ids), "mean_diff": d, "ci95": [lo, hi], "p_wilcoxon": wilcoxon_paired(x, y),
                         "cliffs_delta": cliffs_delta(x, y), "cohens_dz": cohens_dz(x, y)})
            pvals.append(rows[-1]["p_wilcoxon"])
    if pvals:
        for row, adj in zip(rows, holm(pvals)):
            row["p_holm"] = float(adj)
    summary["comparisons"] = rows
    write_json(out / "summary.json", summary)
    print(f"collected {len(summary['experiments'])} experiments -> {out}/summary.json")


if __name__ == "__main__":
    main()
