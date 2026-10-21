#!/usr/bin/env bash
# Experiment matrix. Tier 1 = 5-fold CV for baseline + full model (required for the thesis claims).
# Tier 2 = cumulative ablation on fold 0.  Tier 3 = sensitivity sweeps on fold 0.
# Usage: bash scripts/run_experiments.sh [tier1|tier2|tier3|all] [extra --set overrides, e.g. epochs_seg=100]
set -euo pipefail
TIER=${1:-tier1}; shift || true
EXTRA="$@"
run() { python -m bratsgan.train --config configs/$1.json --set name=$1_f$2 fold=$2 $EXTRA; }
if [[ $TIER == tier1 || $TIER == all ]]; then
  for k in 0 1 2 3 4; do run a0_unet $k; run a5_full $k; done
fi
if [[ $TIER == tier2 || $TIER == all ]]; then
  for c in a1_res a2_att a3_maf a4_tversky; do run $c 0; done
fi
if [[ $TIER == tier3 || $TIER == all ]]; then
  for c in g_lam001 g_lam020 g_hinge g_nosmooth t_dice_eq t_fn_heavy; do run $c 0; done
fi
python scripts/collect_results.py --runs runs --out results --baseline a0_unet
for k in 0; do :; done
python scripts/grade_classification.py --exp a5_full --runs runs --cache cache --splits splits.json --out results/grade_a5_full.json || true
