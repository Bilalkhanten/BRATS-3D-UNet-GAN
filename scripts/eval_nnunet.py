"""Per-patient metrics of nnU-Net predictions, with exactly the conventions of bratsgan.metrics.

nnU-Net is trained with scripts/export_nnunet.py (region-based labels).  In the region-based setting the
predicted label map is expressed in the order of `regions_class_order` = [1, 2, 3] = [WT, TC, ET], and later
regions overwrite earlier ones, so that  WT = pred > 0,  TC = pred >= 2,  ET = pred == 3.
(Use --label-space labels if you trained a plain multi-class model with labels 1=NCR, 2=ED, 3=ET:
 WT = pred > 0, TC = pred in {1, 3}, ET = pred == 3.)  Check this against the dataset.json of your nnU-Net version.

    python scripts/eval_nnunet.py --root /data/BraTS2020 --splits splits.json \
        --val-pred "nnunet_pred/val_f{k}" --test-pred "nnunet_pred/test_f{k}" --out nnunet

Writes  nnunet/val_metrics.csv   (every development patient, predicted by the fold model that did not train on it)
        nnunet/test_metrics.csv  (every test patient: metrics averaged over the five fold models, as for our models)
Both files have the columns of the test_metrics.csv written by bratsgan.train, so the thesis builder reads them directly.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bratsgan.data import find_cases, labels_to_regions  # noqa: E402
from bratsgan.metrics import region_metrics  # noqa: E402
from bratsgan.utils import read_json  # noqa: E402


def to_regions(pred, space):
    if space == "regions":
        wt, tc, et = pred > 0, pred >= 2, pred == 3
    else:
        wt, tc, et = pred > 0, np.isin(pred, (1, 3)), pred == 3
    return np.stack([wt, tc, et]).astype(np.uint8)


def evaluate(cases, ids, pred_dir, space):
    import nibabel as nib
    rows = []
    for cid in ids:
        c = cases[cid]
        gt_nii = nib.load(c["seg"])
        gt = labels_to_regions(np.asarray(gt_nii.dataobj).astype(np.uint8))
        f = Path(pred_dir) / f"{cid}.nii.gz"
        pred = to_regions(np.asarray(nib.load(str(f)).dataobj).astype(np.uint8), space)
        spacing = tuple(float(s) for s in gt_nii.header.get_zooms()[:3])
        m = region_metrics(pred, gt, spacing=spacing)
        m.update(id=cid, grade=c["grade"])
        rows.append(m)
    df = pd.DataFrame(rows)
    return df[["id", "grade"] + [x for x in df.columns if x not in ("id", "grade")]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--val-pred", required=True, help="directory pattern with {k}: predictions of fold model k on its validation fold")
    ap.add_argument("--test-pred", required=True, help="directory pattern with {k}: predictions of fold model k on the test set")
    ap.add_argument("--out", default="nnunet")
    ap.add_argument("--label-space", choices=["regions", "labels"], default="regions")
    a = ap.parse_args()
    splits = read_json(a.splits)
    cases = {c["id"]: c for c in find_cases(a.root)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    K = len(splits["folds"])
    val = [evaluate(cases, splits["folds"][k], a.val_pred.format(k=k), a.label_space) for k in range(K)]
    pd.concat(val, ignore_index=True).to_csv(out / "val_metrics.csv", index=False)
    tests = [evaluate(cases, splits["test"], a.test_pred.format(k=k), a.label_space).set_index("id") for k in range(K)]
    num = [c for c in tests[0].columns if c != "grade"]
    mean = sum(t[num] for t in tests) / K
    mean["grade"] = tests[0]["grade"]
    mean.reset_index().to_csv(out / "test_metrics.csv", index=False)
    print(f"wrote {out}/val_metrics.csv ({sum(len(v) for v in val)} patients) and test_metrics.csv ({len(mean)} patients)")


if __name__ == "__main__":
    main()
