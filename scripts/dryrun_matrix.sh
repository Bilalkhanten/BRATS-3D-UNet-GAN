#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."; export PYTHONPATH=$PWD OMP_NUM_THREADS=2
W=${1:-/tmp/bratsgan_dryrun}   # directory that already holds cache/ and splits.json (see RUNBOOK, dry run)
TINY="base=8 depth=4 max_ch=64 patch=[32,32,32] batch_size=2 iters_per_epoch=20 epochs_seg=8 epochs_adv=3 val_every=2 num_workers=0 amp=false adv_ramp_epochs=1 ndf=8"
run() { python -m bratsgan.train --config configs/$1.json --set name=$1_f$2 fold=$2 out_dir=$W/runs cache_dir=$W/cache splits=$W/splits.json $TINY > $W/log_$1_f$2.txt 2>&1; echo "$1 f$2 done $(date +%T)" >> $W/progress.txt; }
for k in 0 1 2 3 4; do run a0_unet $k; run a5_full $k; done
for c in a1_res a2_att a3_maf a4_tversky; do run $c 0; done
for c in g_lam001 g_lam020 g_hinge g_nosmooth t_dice_eq t_fn_heavy; do run $c 0; done
python scripts/collect_results.py --runs $W/runs --out $W/results --baseline a0_unet >> $W/progress.txt 2>&1
python scripts/grade_classification.py --exp a5_full --runs $W/runs --cache $W/cache --splits $W/splits.json --out $W/results/grade_a5_full.json >> $W/progress.txt 2>&1
echo ALL_DONE >> $W/progress.txt
