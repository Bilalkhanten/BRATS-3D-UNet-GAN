# FAQ

**`CUDA out of memory`.** Reduce `patch` (for example `[96,96,96]`), `batch_size`, or `base`. Mixed precision (`amp=true`) is the
default on CUDA.

**Training is slow on a CPU.** Use the tiny settings of `scripts/smoke_test.sh` or `examples/02_train_tiny_model.py`; the full
configuration is meant for a GPU.

**`FileNotFoundError: ... splits.json`.** Run `scripts/prepare_data.py` (or `bratsgan prepare`) first; `cache_dir` and `splits`
in the configuration must point to its outputs.

**The ET Dice of LGG cases is 0 or 1.** That is the empty-mask convention (see [Evaluation](evaluation.md)): both masks empty gives
1, one empty gives 0.

**Why are there no BraTS results in the repository?** The data cannot be redistributed and the GPU runs have to be performed
by the user; see [Reproducibility](reproducibility.md).

**How do I add a variant?** Copy a file in `configs/`, change the keys, and run `python -m bratsgan.train --config configs/<new>.json`.
New keys require a default in `bratsgan/config.py` and an entry in `scripts/gen_config_docs.py` (run `make docs`).
