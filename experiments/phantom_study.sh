#!/usr/bin/env bash
# Implementation-verification study on synthetic phantoms (NOT a BraTS result).
# 5 variants x 2 seeds on one patient-level split of 100 phantoms (75 HGG / 25 LGG, 48^3 voxels).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
W=${1:-$ROOT/runs/phantom_study}
mkdir -p $W
if [ ! -d $W/raw ]; then
python - <<PY
import sys; sys.path.insert(0,'.')
from bratsgan.synthetic import write_brats_layout
write_brats_layout("$W/raw", n_hgg=75, n_lgg=25, size=48, seed=2026)
PY
python scripts/prepare_data.py --root $W/raw --cache $W/cache --splits $W/splits.json --workers 2 --folds 3 --test-frac 0.2
fi
COMMON="base=8 depth=4 max_ch=64 patch=[48,48,48] batch_size=2 lr=1e-3 iters_per_epoch=50 epochs_seg=20 epochs_adv=8 val_every=4 num_workers=0 amp=false adv_ramp_epochs=2 ndf=8 fold=0 out_dir=$W/runs cache_dir=$W/cache splits=$W/splits.json"
for s in 1 2; do
  for c in a0_unet a2_att a3_maf a4_tversky a5_full; do
    if [ ! -f $W/runs/${c}_s${s}_f0/test_metrics.csv ]; then
      python -m bratsgan.train --config configs/$c.json --set name=${c}_s${s}_f0 seed=$s $COMMON > $W/log_${c}_s${s}.txt 2>&1
    fi
    echo "done $c seed $s $(date +%T)" >> $W/progress.txt
  done
done
echo ALL_DONE >> $W/progress.txt
