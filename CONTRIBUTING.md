# Contributing

Thank you for considering a contribution.

1. Fork the repository and create a branch (`git checkout -b feature/short-name`).
2. Install the development environment: `pip install -e ".[dev]"` and `pre-commit install`.
3. Make the change and add or update tests in `tests/`.
4. Run `ruff check bratsgan tests scripts examples` and `pytest`. If you changed training or data code also run
   `bash scripts/smoke_test.sh`.
5. If you added a configuration key, add a default in `bratsgan/config.py` and a description in `scripts/gen_config_docs.py`, then run
   `make docs`.
6. Open a pull request using the template and describe what changed and how it was tested.

## Guidelines

* Keep the unit of analysis the **patient**: any new split or statistic must respect the patient-level partition.
* Do not commit data, checkpoints or run directories (`runs/`, `cache/`, `*.pt`); they are git-ignored.
* Results on synthetic data must be labelled as such in code, documentation and figures.
* Prefer small, readable functions with a docstring that states shapes and conventions (for example `(4, D, H, W)`).
