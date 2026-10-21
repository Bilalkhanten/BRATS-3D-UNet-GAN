#!/usr/bin/env bash
# Hard synthetic benchmark (NOT a BraTS result): 160 phantoms (120 HGG / 40 LGG, 48^3), smaller lesions, thin enhancing
# shells, reduced tumour contrast, blurred boundaries, stronger noise and bias field (bratsgan.synthetic, difficulty="hard").
# Order of the runs = priority: tier A (ablation chain, seed 1), tier B (sensitivity, seed 1), tier C (ablation chain, seed 2),
# tier D (folds 1-2 of A0 and A5, for cross-validation and the grade pipeline).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
W=${1:-$ROOT/runs/hard_study}
mkdir -p $W
if [ ! -d $W/raw ]; then
python - <<PY
import sys; sys.path.insert(0,'.')
from bratsgan.synthetic import write_brats_layout
write_brats_layout("$W/raw", n_hgg=120, n_lgg=40, size=48, seed=2027, difficulty="hard")
PY
python scripts/prepare_data.py --root $W/raw --cache $W/cache --splits $W/splits.json --workers 2 --folds 3 --test-frac 0.2
fi
COMMON="base=8 depth=4 max_ch=64 patch=[48,48,48] batch_size=2 lr=1e-3 iters_per_epoch=50 epochs_seg=32 epochs_adv=8 val_every=4 num_workers=0 amp=false adv_ramp_epochs=2 ndf=8 out_dir=$W/runs cache_dir=$W/cache splits=$W/splits.json"
run() { # config seed fold
  local c=$1 s=$2 f=$3
  local n=${c}_s${s}_f${f}
  if [ ! -f $W/runs/$n/test_metrics.csv ]; then
    rm -rf $W/runs/$n
    python -m bratsgan.train --config configs/$c.json --set name=$n seed=$s fold=$f $COMMON > $W/log_$n.txt 2>&1
  fi
  echo "done $n $(date +%T)" >> $W/progress.txt
}
for c in a0_unet a1_res a2_att a3_maf a4_tversky a5_full; do run $c 1 0; done
for c in t_dice_eq t_fn_heavy g_lam001 g_lam020 g_hinge g_nosmooth; do run $c 1 0; done
for c in a0_unet a1_res a2_att a3_maf a4_tversky a5_full; do run $c 2 0; done
for f in 1 2; do run a5_full 1 $f; run a0_unet 1 $f; done
echo ALL_DONE >> $W/progress.txt
