#!/usr/bin/env bash
# End-to-end test on synthetic NIfTI phantoms (BraTS folder layout). CPU-friendly (~minutes).
# This verifies that the code runs; it says nothing about BraTS performance.
set -euo pipefail
W=${1:-/tmp/bratsgan_smoke}; rm -rf $W; mkdir -p $W
python - <<PY
import sys; sys.path.insert(0,'.')
from bratsgan.synthetic import write_brats_layout
write_brats_layout("$W/raw", n_hgg=18, n_lgg=8, size=48, seed=7)
PY
python scripts/prepare_data.py --root $W/raw --cache $W/cache --splits $W/splits.json --workers 2 --folds 2 --test-frac 0.25
TINY="base=8 depth=4 max_ch=64 patch=[32,32,32] batch_size=2 iters_per_epoch=12 epochs_seg=3 epochs_adv=2 val_every=1 num_workers=0 amp=false adv_ramp_epochs=1 ndf=8"
for k in 0 1; do
  python -m bratsgan.train --config configs/a0_unet.json --set name=a0_unet_f$k fold=$k out_dir=$W/runs cache_dir=$W/cache splits=$W/splits.json $TINY
  python -m bratsgan.train --config configs/a5_full.json  --set name=a5_full_f$k  fold=$k out_dir=$W/runs cache_dir=$W/cache splits=$W/splits.json $TINY
done
python scripts/collect_results.py --runs $W/runs --out $W/results --baseline a0_unet
python scripts/grade_classification.py --exp a5_full --runs $W/runs --cache $W/cache --splits $W/splits.json --out $W/results/grade_a5_full.json
echo SMOKE TEST PASSED
