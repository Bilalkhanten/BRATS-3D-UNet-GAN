# Training

## Schedule

Training has two stages. In the **segmentation stage** (`epochs_seg`) the generator is trained with the segmentation loss alone
(AdamW, polynomial learning-rate decay, gradient clipping). In the **adversarial stage** (`epochs_adv`, only if `gan=true`) a
discriminator is trained alternately and its loss, weighted by `adv_weight` and ramped up over `adv_ramp_epochs`, is added to the
generator loss. Validation runs every `val_every` epochs by sliding-window inference on the validation fold; the checkpoint with the
highest mean validation Dice (`best.pt`) is used for the final evaluation.

Patches (`patch`, default 128^3) are sampled by `data.PatchStream`: with probability `fg_prob` the patch is centred on a tumour voxel,
otherwise at a random position. Augmentation (`augment.py`) is applied on the GPU: random affine transforms, flips and intensity
changes.

## Running one experiment

```bash
python -m bratsgan.train --config configs/a5_full.json --set name=a5_full_f0 fold=0
# equivalent:
bratsgan train --config configs/a5_full.json --set name=a5_full_f0 fold=0
```

`--set` takes JSON values (`patch=[64,64,64]`, `amp=false`). For a quick CPU run use `configs/quick.json` or the overrides of
`scripts/smoke_test.sh`.

## The experiment matrix

`scripts/run_experiments.sh` runs the matrix in three tiers:

| Tier | Runs | Purpose |
| --- | --- | --- |
| `tier1` | A0 and A5, five folds | cross-validated comparison of baseline and full model |
| `tier2` | A1 to A4, fold 0 | cumulative ablation |
| `tier3` | sensitivity runs (`t_*`, `g_*`), fold 0 | Tversky parameters, adversarial weight, loss type, smoothing |

```bash
bash scripts/run_experiments.sh tier1 epochs_seg=300
```

## Outputs of a run (`runs/<name>/`)

| File | Content |
| --- | --- |
| `config.json` | the resolved configuration |
| `train_log.jsonl` | one record per epoch (losses, learning rate, validation Dice, discriminator accuracy) |
| `best.pt`, `last.pt` | best checkpoint (by validation Dice) and the latest state for resuming |
| `val_metrics.csv`, `test_metrics.csv` | per-patient metrics of the best checkpoint |
| `val_summary.json`, `test_summary.json` | means, standard deviations and medians |
| `pred_val/`, `pred_test/` | predicted regions as `.npy` |

Training resumes automatically from `last.pt` if the run directory already exists.
