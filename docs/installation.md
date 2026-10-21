# Installation

The code was tested with Python 3.13 and the versions listed in `requirements.txt`; Python 3.10 or newer is required.

## Virtual environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

PyTorch is installed from PyPI by default (CPU or the default CUDA build for your platform). For a specific CUDA build follow the
selector at <https://pytorch.org/get-started/locally/> *before* installing the package, for example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
```

## Check the installation

```bash
pytest -m "not slow"          # seconds
pytest                        # includes a short end-to-end training run (about a minute on a CPU)
bash scripts/smoke_test.sh    # full pipeline on synthetic phantoms, a few minutes on a CPU
```

## Docker (CPU)

```bash
docker build -t bratsgan .
docker run --rm bratsgan                      # runs the tests
docker run --rm -v "$PWD/runs:/workspace/runs" bratsgan python examples/02_train_tiny_model.py /workspace/runs/demo
```

## Optional components

| Extra | Install | Used for |
| --- | --- | --- |
| `thesis` | `pip install -e ".[thesis]"` plus `pandoc`, LibreOffice and `poppler-utils` | building the thesis document (`thesis/`) |
| `nnunet` | `pip install -e ".[nnunet]"` | the nnU-Net v2 baseline (`scripts/export_nnunet.py`, `scripts/eval_nnunet.py`) |
