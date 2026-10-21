"""LGG/HGG classification from segmentation-derived features, using the reference masks of synthetic phantoms.

Shows the feature extractor (101 features) and the evaluation helpers. With reference masks the task is easy by
construction; the thesis compares this with features of *predicted* masks (see scripts/grade_classification.py).

    python examples/04_grade_from_masks.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bratsgan.data import labels_to_regions  # noqa: E402
from bratsgan.features import extract_features  # noqa: E402
from bratsgan.stats import binary_report  # noqa: E402
from bratsgan.synthetic import make_phantom  # noqa: E402


def main(n_hgg=30, n_lgg=30):
    rows, y = [], []
    for g, n in (("HGG", n_hgg), ("LGG", n_lgg)):
        for i in range(n):
            vols, label, _ = make_phantom(size=40, grade=g, seed=1000 + i + (0 if g == "HGG" else 500), difficulty="hard")
            rows.append(extract_features(vols, labels_to_regions(label)))
            y.append(int(g == "HGG"))
    names = sorted(rows[0])
    X = np.array([[r[k] for k in names] for r in rows])
    y = np.array(y)
    print(f"{X.shape[0]} phantoms, {X.shape[1]} features")
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    prob = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    rep = binary_report(y, prob, n_boot=500)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rep.items() if not isinstance(v, (list, tuple))})


if __name__ == "__main__":
    main()
