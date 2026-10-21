#!/usr/bin/env bash
# Follow-up to phantom_study.sh: (i) residual-only variant a1_res (fold 0, seeds 1-2) and
# (ii) the full model on folds 1 and 2 (seed 1) so that out-of-fold grade classification can be run.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
W=${1:-$ROOT/runs/phantom_study}
while ! grep -q ALL_DONE $W/progress.txt; do sleep 60; done
COMMON="base=8 depth=4 max_ch=64 patch=[48,48,48] batch_size=2 lr=1e-3 iters_per_epoch=50 epochs_seg=20 epochs_adv=8 val_every=4 num_workers=0 amp=false adv_ramp_epochs=2 ndf=8 out_dir=$W/runs cache_dir=$W/cache splits=$W/splits.json"
for f in 1 2; do
  n=a5_full_s1_f$f
  [ -f $W/runs/$n/test_metrics.csv ] || python -m bratsgan.train --config configs/a5_full.json --set name=$n seed=1 fold=$f $COMMON > $W/log_$n.txt 2>&1
  echo "done $n $(date +%T)" >> $W/progress2.txt
done
for s in 1 2; do
  n=a1_res_s${s}_f0
  [ -f $W/runs/$n/test_metrics.csv ] || python -m bratsgan.train --config configs/a1_res.json --set name=$n seed=$s fold=0 $COMMON > $W/log_$n.txt 2>&1
  echo "done $n $(date +%T)" >> $W/progress2.txt
done
echo ALL_DONE >> $W/progress2.txt
