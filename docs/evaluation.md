# Evaluation

## Inference

`bratsgan.infer.sliding_window` runs the network over the whole volume with overlapping windows (`final_overlap`, default 0.5),
Gaussian importance weighting of each window, a final flush at the volume border, and normalisation by the accumulated weights.
Optional mirror test-time augmentation (`final_tta`). The network outputs logits; the window average is taken over probabilities.
`predict_regions` thresholds at 0.5 and returns the binary regions, the probabilities and the inference time.

## Metrics (`bratsgan/metrics.py`)

Computed per patient and per region (WT, TC, ET) over the whole volume:

| Metric | Definition |
| --- | --- |
| Dice | 2TP / (2TP + FP + FN) |
| IoU | TP / (TP + FP + FN) |
| Sensitivity, precision, specificity | TP/(TP+FN), TP/(TP+FP), TN/(TN+FP) |
| HD95 | 95th percentile of the symmetric surface distances (in voxels or in mm via `spacing`) |

**Empty-mask conventions.** If both masks are empty, Dice is 1 and HD95 is 0; if exactly one is empty, Dice is 0 and HD95 is the
penalty 373.13 (the diagonal of the BraTS volume), as in the challenge's ranking convention. This matters for ET in LGG cases,
which often contain no enhancing tumour.

## Per-run and aggregate evaluation

```bash
python -m bratsgan.evaluate --run runs/a5_full_f0 --split test --tta
python scripts/collect_results.py --runs runs --out results --baseline a0_unet
```

`collect_results.py` writes `oof_<exp>.csv` (out-of-fold validation metrics), `test_<exp>.csv`, `curves_<exp>.json` (training curves)
and `summary.json` (per-variant means with bootstrap intervals, paired comparisons against the baseline, Holm-adjusted p-values).

## Statistics (`bratsgan/stats.py`)

* percentile **bootstrap** intervals over patients (`bootstrap_ci`), **paired bootstrap** of differences (`paired_bootstrap_diff`);
* **Wilcoxon signed-rank** test with **Holm** correction over the family of comparisons (`wilcoxon_paired`, `holm`);
* effect sizes: Cliff's delta (`cliffs_delta`) and the paired Cohen's d_z (`cohens_dz`);
* `roc_auc`, `binary_report` (accuracy, sensitivity, specificity, AUC with bootstrap intervals) and `expected_calibration_error`.

The unit of analysis is the patient; the five fold models of one variant are treated as realisations of the same model and their
per-patient scores are averaged on the common test set.

## Grade classification (LGG vs HGG)

`bratsgan/features.py` computes 101 features from a segmentation and the four images (log-volumes of the compartments and their ratios, maximal diameter,
extent, sphericity, enhancement contrast, and intensity statistics per compartment and sequence). Features are computed from the
**predicted** masks, never from the reference masks, so that the classifier sees what a deployed system would see.

```bash
python scripts/grade_classification.py --exp a5_full --runs runs --cache cache --splits splits.json --out results/grade_a5_full.json
```

The script cross-validates logistic regression and gradient boosting on the development patients, evaluates once on the held-out
test patients, and also reports the same pipeline on the reference masks as an upper bound. `examples/04_grade_from_masks.py`
shows the feature extractor on synthetic phantoms.
