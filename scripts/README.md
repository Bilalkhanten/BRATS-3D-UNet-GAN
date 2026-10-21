# Scripts

| Script | Purpose |
| --- | --- |
| `prepare_data.py` | pre-process BraTS into the cache and write patient-level splits |
| `run_experiments.sh` | experiment matrix in tiers (`tier1`, `tier2`, `tier3`, `all`) |
| `collect_results.py` | aggregate runs into tidy CSVs and `summary.json` with statistics |
| `grade_classification.py` | LGG/HGG classification from predicted (and reference) masks |
| `export_nnunet.py`, `eval_nnunet.py` | nnU-Net v2 baseline on identical splits |
| `smoke_test.sh` | end-to-end test on synthetic phantoms |
| `dryrun_matrix.sh` | dry run of the whole BraTS pipeline on synthetic data in the BraTS layout |
| `make_sample_data.py` | regenerate `data/sample/` |
| `gen_config_docs.py` | regenerate `docs/configuration.md` |
