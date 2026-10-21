# Results in this repository

`results/` holds the outputs of the grade-classification experiments on **synthetic phantoms** (not BraTS):

| File pattern | Content |
| --- | --- |
| `grade_phantom_a5.json`, `grade_hard_a5.json` | summaries (AUC, accuracy, sensitivity, specificity, intervals) for the easy and hard phantom benchmarks |
| `grade_*_a5.<source>_<classifier>.<cv|test>.csv` | per-patient predicted probabilities (`id,y,prob`) for features from predicted (`pred`) or reference (`gt`) masks, logistic regression (`logreg`) or gradient boosting (`gboost`) |

What the synthetic studies show, stated conservatively:

* On the easy phantoms all variants are close to saturation; they test the pipeline, not the architecture.
* On the hard phantoms the differences between variants are of the same size as the variation between seeds and folds. The
  pre-registered hypotheses H1 to H4 of the thesis are **not supported** there, and H5 (non-inferiority to nnU-Net) was not
  evaluated because no nnU-Net was trained on that benchmark.
* Grade classification from predicted masks is unstable on the hard phantoms (cross-validated AUC about 0.57, test AUC about 0.92
  on a very small test set), whereas reference masks give AUC 1.0 by construction.

These statements describe synthetic data only. The BraTS tables of the thesis are produced from the output of the experiments in
[`RUNBOOK.md`](../RUNBOOK.md) and read "pending" until those exist.
