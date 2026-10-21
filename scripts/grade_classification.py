"""LGG/HGG grade classification from PREDICTED segmentations.

Protocol (leak-free):
  1. Out-of-fold (OOF) predictions: every cross-validation case is segmented by the model that did NOT train on it.
  2. Classifier CV: the classifier is trained on the OOF features of 4 folds and tested on the 5th.
  3. Final test: the classifier is trained on all OOF features and applied to held-out test cases,
     whose features come from each fold model; probabilities are averaged over the fold models.
  4. Upper bound: the same pipeline using ground-truth masks (shows the cost of segmentation error).

    python scripts/grade_classification.py --exp a5_full --runs runs --cache cache --splits splits.json --out results/grade_a5_full.json
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from bratsgan.data import labels_to_regions, load_case
from bratsgan.features import extract_features
from bratsgan.stats import binary_report, expected_calibration_error
from bratsgan.utils import read_json, write_json


def feats_for(cache, cid, regions):
    img, _, meta = load_case(cache, cid, mmap=False)
    f = extract_features(np.asarray(img, np.float32), regions, tuple(meta.get("spacing", (1, 1, 1))))
    f["id"], f["y"] = cid, 1 if meta["grade"] == "HGG" else 0
    return f


def lr_model():
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
    return GridSearchCV(pipe, {"logisticregression__C": [0.01, 0.03, 0.1, 0.3, 1, 3]}, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc")


def gb_model():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, class_weight="balanced", random_state=0)


MODELS = {"logreg": lr_model, "gboost": gb_model}


def run(a):
    splits = read_json(a.splits)
    K = len(splits["folds"])
    oof_rows, test_rows = {"pred": [], "gt": []}, {"pred": {}, "gt": []}
    fold_of = {}
    for k in range(K):
        run_dir = Path(a.runs) / f"{a.exp}_f{k}"
        for cid in splits["folds"][k]:
            fold_of[cid] = k
            pred = np.load(run_dir / "pred_val" / f"{cid}_pred.npy")
            _, seg, _ = load_case(a.cache, cid, mmap=False)
            oof_rows["pred"].append(feats_for(a.cache, cid, pred))
            oof_rows["gt"].append(feats_for(a.cache, cid, labels_to_regions(np.asarray(seg))))
        for cid in splits["test"]:
            pred = np.load(run_dir / "pred_test" / f"{cid}_pred.npy")
            test_rows["pred"].setdefault(k, []).append(feats_for(a.cache, cid, pred))
    for cid in splits["test"]:
        _, seg, _ = load_case(a.cache, cid, mmap=False)
        test_rows["gt"].append(feats_for(a.cache, cid, labels_to_regions(np.asarray(seg))))
    results = {}
    for source in ("pred", "gt"):
        oof = pd.DataFrame(oof_rows[source])
        feat_cols = [c for c in oof.columns if c not in ("id", "y")]
        X, y = oof[feat_cols].values, oof["y"].values
        folds = np.array([fold_of[i] for i in oof["id"]])
        for mname, mk in MODELS.items():
            cvp = np.zeros(len(y))
            for k in range(K):
                tr, te = folds != k, folds == k
                m = mk().fit(X[tr], y[tr]); cvp[te] = m.predict_proba(X[te])[:, 1]
            final = mk().fit(X, y)
            if source == "pred":
                probs = np.mean([final.predict_proba(pd.DataFrame(test_rows["pred"][k])[feat_cols].values)[:, 1] for k in range(K)], axis=0)
                tdf = pd.DataFrame(test_rows["pred"][0])
            else:
                tdf = pd.DataFrame(test_rows["gt"]); probs = final.predict_proba(tdf[feat_cols].values)[:, 1]
            key = f"{source}_{mname}"
            results[key] = {"cv": binary_report(y, cvp), "test": binary_report(tdf["y"].values, probs),
                            "cv_ece": expected_calibration_error(y, cvp), "test_ece": expected_calibration_error(tdf["y"].values, probs)}
            pd.DataFrame({"id": oof["id"], "y": y, "prob": cvp}).to_csv(Path(a.out).with_suffix(f".{key}.cv.csv"), index=False)
            pd.DataFrame({"id": tdf["id"], "y": tdf["y"], "prob": probs}).to_csv(Path(a.out).with_suffix(f".{key}.test.csv"), index=False)
            print(key, "CV AUC %.3f | test AUC %.3f acc %.3f" % (results[key]["cv"]["auc"], results[key]["test"]["auc"], results[key]["test"]["accuracy"]))
    write_json(a.out, results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True); ap.add_argument("--runs", default="runs")
    ap.add_argument("--cache", default="cache"); ap.add_argument("--splits", default="splits.json")
    ap.add_argument("--out", required=True)
    run(ap.parse_args())
