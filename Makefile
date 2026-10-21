.PHONY: help install test smoke lint docs sample thesis clean
PY ?= python

help:
	@echo "install  - editable install with development tools"
	@echo "test     - unit tests (fast)"
	@echo "smoke    - end-to-end test on synthetic phantoms (CPU, minutes)"
	@echo "lint     - ruff"
	@echo "docs     - regenerate docs/configuration.md from bratsgan.config"
	@echo "sample   - regenerate the sample phantoms in data/sample"
	@echo "thesis   - build the thesis document (needs --template, pandoc, LibreOffice)"

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

smoke:
	bash scripts/smoke_test.sh

lint:
	ruff check bratsgan tests scripts examples

docs:
	$(PY) scripts/gen_config_docs.py

sample:
	$(PY) scripts/make_sample_data.py

thesis:
	$(PY) thesis/build_thesis.py thesis/out/thesis.docx --template $(TEMPLATE)

clean:
	rm -rf runs/* .pytest_cache .ruff_cache build dist *.egg-info
	touch runs/.gitkeep
