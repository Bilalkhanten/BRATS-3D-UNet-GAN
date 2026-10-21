# Experiments behind the thesis

| File | Purpose |
| --- | --- |
| `verify.py` | analytical and numerical verification (gradients, losses, cost model, sliding window); writes `runs/verification/verify.json` |
| `phantom_study.sh`, `phantom_followup.sh` | synthetic phantom study (easy setting) and follow-up runs |
| `hard_study.sh` | hard synthetic benchmark (160 phantoms, ablation chain, sensitivity runs, folds) |
| `latency.py` | inference latency of the variants |
| `figstyle.py` | shared matplotlib style and palette |
| `diagrams*.py` | conceptual diagrams used in the thesis |

All studies run on synthetic data and are **not** evidence about BraTS performance. Outputs go to `runs/<study>/`.
