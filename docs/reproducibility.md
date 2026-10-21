# Reproducibility and the status of the evidence

## Determinism

`seed` seeds Python, NumPy and PyTorch (`utils.seed_everything`). `deterministic=true` additionally requests deterministic
kernels, at a cost in speed. Runs on different hardware or PyTorch versions are not bitwise identical; the thesis therefore reports
the variation between seeds (two seeds in the synthetic studies) next to every difference between variants.

Every run stores the resolved `config.json`, and `splits.json` is created once and shared by all experiments.

## What counts as evidence

| Source | Evidence about BraTS? |
| --- | --- |
| Unit tests, analytical and numerical verification (`experiments/verify.py`) | No: they test that the code does what the derivations say |
| Synthetic phantom studies (`experiments/phantom_study.sh`, `hard_study.sh`) | No: phantoms are far simpler than patient data |
| A dry run on synthetic data in the BraTS layout (`scripts/dryrun_matrix.sh`) | No: it only shows that every command of the BraTS pipeline runs |
| Runs of `scripts/run_experiments.sh` on the real BraTS 2020 data | Yes, once they have been run |

No BraTS result is included in this repository.

## Running the BraTS experiments

See [`RUNBOOK.md`](../RUNBOOK.md) for the order of the commands, the GPU time and what to collect.

## Dry run of the pipeline

```bash
bash scripts/dryrun_matrix.sh /tmp/dryrun
```

generates 60 synthetic volumes in the BraTS layout, prepares them, trains the matrix with tiny settings and runs the collection
and the grade classification. The numbers are meaningless; the point is that every command works before a long GPU job is started.
