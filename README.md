# BRATS-3D-UNet-GAN

[![CI](https://github.com/Bilalkhanten/BRATS-3D-UNet-GAN/actions/workflows/ci.yml/badge.svg)](https://github.com/Bilalkhanten/BRATS-3D-UNet-GAN/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Attention-residual **3D U-Net with adversarial refinement** for multimodal brain-tumour segmentation (BraTS 2020: whole tumour,
tumour core, enhancing tumour) and **LGG/HGG grade classification** from the predicted masks. The repository accompanies the
PhD thesis *3D Brain Tumor Segmentation and Grade Classification with 3D U-Net and GANs* and contains the complete pipeline, the
experiment configurations, the statistical analysis and the generator of the thesis document.

> **Status.** No BraTS result is included. The data cannot be redistributed and the GPU runs must be performed by the user
> ([`RUNBOOK.md`](RUNBOOK.md)). Everything that *is* included was verified on synthetic data: unit tests, analytical/numerical
> verification, and phantom studies. These show that the code works; they are not evidence about accuracy on patient data
> (see [docs/reproducibility.md](docs/reproducibility.md)).

## Features

* **Configurable 3D U-Net** with residual blocks, additive attention gates, a modality-attentive fusion (MAF) stem and deep
  supervision; every component is a switch, giving the six-step ablation chain A0 to A5.
* **Losses**: Dice, Tversky (asymmetric, optionally focal) and BCE on the three nested regions; **conditional 3D PatchGAN**
  discriminator with LSGAN or hinge objective, label smoothing, instance noise and a ramped weight.
* **Leak-free data handling**: patient-level, grade-stratified splits with an explicit leakage assertion.
* **Whole-volume evaluation**: Gaussian-weighted sliding-window inference, Dice / IoU / sensitivity / precision / HD95 with the
  BraTS empty-mask conventions, bootstrap intervals, paired tests with Holm correction, effect sizes, calibration.
* **Grade classification** from 101 segmentation-derived features of the *predicted* masks (logistic regression, gradient boosting).
* **Synthetic phantoms** in the BraTS folder layout to test every step without data access, plus a ready-made sample in `data/sample/`.
* **nnU-Net v2 export/evaluation** on identical splits for a fair baseline.
* **Reproducible thesis builder** that regenerates every table and figure of the results chapter from the run outputs.

## Installation

```bash
git clone https://github.com/Bilalkhanten/BRATS-3D-UNet-GAN.git
cd BRATS-3D-UNet-GAN
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

For a CUDA build of PyTorch and Docker usage see [docs/installation.md](docs/installation.md).

## Quick start (no data needed)

```bash
pytest -m "not slow"                          # unit tests, seconds
python examples/01_synthetic_quickstart.py    # draw two phantoms
python examples/02_train_tiny_model.py        # generate -> prepare -> train -> evaluate, minutes on a CPU
bash scripts/smoke_test.sh                    # full pipeline incl. cross-validation folds and grading
```

or step by step with the command-line tool:

```bash
bratsgan synth   --out /tmp/raw --hgg 12 --lgg 4 --size 40
bratsgan prepare --root /tmp/raw --cache /tmp/cache --splits /tmp/splits.json --folds 2 --test-frac 0.25
bratsgan train   --config configs/a5_full.json --set name=demo cache_dir=/tmp/cache splits=/tmp/splits.json \
                 base=8 depth=4 max_ch=64 patch=[32,32,32] iters_per_epoch=10 epochs_seg=4 epochs_adv=2 num_workers=0 amp=false
bratsgan info    --config configs/a5_full.json          # parameters and GMACs
```

## Training on BraTS 2020

```bash
python scripts/prepare_data.py --root /data/MICCAI_BraTS2020_TrainingData --cache cache --splits splits.json --workers 8
bash scripts/run_experiments.sh tier1                 # A0 and A5, 5-fold cross-validation
python scripts/collect_results.py --runs runs --out results --baseline a0_unet
python scripts/grade_classification.py --exp a5_full --runs runs --cache cache --splits splits.json --out results/grade_a5_full.json
```

The full order of commands, the GPU time and what to collect are in [`RUNBOOK.md`](RUNBOOK.md).

## Repository layout

| Path | Contents |
| --- | --- |
| `bratsgan/` | the package: `data`, `models`, `losses`, `train`, `infer`, `evaluate`, `metrics`, `stats`, `features`, `augment`, `complexity`, `synthetic`, `cli` |
| `configs/` | one JSON file per experiment (A0 to A5, sensitivity runs, `quick.json`) |
| `scripts/` | data preparation, experiment matrix, result collection, grade classification, nnU-Net export/evaluation, smoke and dry-run tests |
| `examples/` | five runnable examples (phantoms, tiny training, inference + overlay, grading, variant costs) |
| `notebooks/` | quickstart and results-exploration notebooks |
| `experiments/` | verification and synthetic-study scripts behind the thesis, figure generators |
| `tests/` | unit, integration and example tests (`pytest`) |
| `data/sample/` | four tiny synthetic phantoms in the BraTS layout |
| `results/` | outputs of the grade-classification experiments on synthetic phantoms |
| `docs/` | documentation ([index](docs/index.md)) |
| `thesis/` | generator of the thesis document |

## Documentation

[Installation](docs/installation.md) · [Data](docs/data.md) · [Architecture](docs/architecture.md) · [Training](docs/training.md) ·
[Evaluation](docs/evaluation.md) · [Configuration reference](docs/configuration.md) · [Reproducibility](docs/reproducibility.md) ·
[Results](docs/results.md) · [Thesis builder](docs/thesis.md) · [FAQ](docs/faq.md)

## Citation

If you use this code, please cite it (see [`CITATION.cff`](CITATION.cff)). The BraTS data must be cited according to the
organisers' instructions (Menze et al., 2015; Bakas et al., 2017, 2018).

## License

MIT, see [`LICENSE`](LICENSE). The BraTS data are distributed under their own terms and are not part of this repository.
