# RUNBOOK — producing the BraTS results for the thesis

The thesis is complete except for the numbers that need a GPU and the BraTS data. This file lists exactly what to run.
Nothing below changes the protocol; do **not** change patch size, schedule, seeds or splits unless you also report the change
(a deviation, such as a smaller patch because of GPU memory, must be stated in the thesis).

## 0. What you will produce and send back

| Item | Produced by | Needed for |
|---|---|---|
| `runs/{exp}_f{k}/{config.json, train_log.jsonl, val_metrics.csv, test_metrics.csv}` | `bratsgan.train` | all tables, curves |
| `results/grade_a5_full.json` (+ the `.csv` files next to it) | `grade_classification.py` | H4, grade table |
| `nnunet/val_metrics.csv`, `nnunet/test_metrics.csv` | `eval_nnunet.py` | H5, comparison table |
| `latency.json` (optional) | `experiments/latency.py` | latency table |
| `gpu.txt`, `env.txt` | commands in step 9 | hardware statement |

## 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # install the CUDA build of torch for your driver
pytest tests                               # 23 tests, seconds on a CPU
```

## 2. Data

Download the BraTS 2020 **training** release (369 cases) under the terms of the challenge. The folder you pass must
contain `MICCAI_BraTS2020_TrainingData/` with one folder per case (`*_t1`, `*_t1ce`, `*_t2`, `*_flair`, `*_seg` NIfTI files)
and the file `name_mapping.csv` (it carries the HGG/LGG label).

## 3. Optional end-to-end check on miniature phantoms (CPU, minutes)

```bash
bash scripts/smoke_test.sh
```

## 3b. Reproducing the hard synthetic benchmark (CPU, several hours)

Section 4.4 of the thesis is generated from `runs/hard_study`. To reproduce it on any machine (about 20 minutes per run
on two CPU cores, 22 runs):

```bash
bash experiments/hard_study.sh /path/to/hard_study      # generates the 160 hard phantoms, trains all runs, writes progress.txt
python thesis/build_thesis.py thesis.docx --hard /path/to/hard_study --template /path/to/your/reference_thesis.docx
```

If the directory does not exist the builder uses the frozen copy of the analysed text in `thesis/assets/hard_frozen`.

## 4. Preprocess and split (once)

```bash
python scripts/prepare_data.py --root /data/BraTS2020 --cache cache --splits splits.json --workers 8
```

Expected: 74 test patients (59 HGG, 15 LGG) and five folds of 60/59/59/59/58 patients. The script asserts that no patient
occurs twice. **Keep `splits.json`**: the nnU-Net baseline must use the same file.

## 5. Measure the time per iteration (pilot), then plan

```bash
python -m bratsgan.train --config configs/a5_full.json --set name=pilot fold=0 epochs_seg=1 epochs_adv=1 iters_per_epoch=40 val_every=1
```

`epoch_seconds` in `runs/pilot_f0/train_log.jsonl` divided by 40 is the time per iteration `t`. Every run of the thesis
has 350 x 250 = 87,500 generator updates, so one run takes about `87500 x t` (the adversarial stage is slightly slower per
iteration). Tier 1 has 10 runs, Tier 2 has 4, Tier 3 has 6 (20 runs in total). Decide from this whether to run all tiers.
`configs/quick.json` (100 + 20 epochs of 200 iterations) shortens every run to about a quarter; it is for rehearsals and
must not be mixed with full runs in one table.

## 6. Experiments

```bash
bash scripts/run_experiments.sh tier1     # A0 and A5 on folds 0-4  (required: CV, test, H4, H5)
bash scripts/run_experiments.sh tier2     # A1-A4 on fold 0         (ablation, H1)
bash scripts/run_experiments.sh tier3     # six sensitivity runs    (H2, H3, adversarial sensitivity)
```

Each call also runs `collect_results.py` and the grade classification. Runs can be executed on several GPUs in parallel
with the same command pattern, for example:

```bash
CUDA_VISIBLE_DEVICES=1 python -m bratsgan.train --config configs/a4_tversky.json --set name=a4_tversky_f0 fold=0
```

The run name must be `{config}_f{fold}` (the analysis finds runs by name). If a run is interrupted, delete its directory and repeat it.

If the GPU runs out of memory, the order of preference is: `batch_size=1`, then `patch=[96,96,96]`. Report whichever you used.

## 7. Grade classification (needs the five fold models of A5)

```bash
python scripts/grade_classification.py --exp a5_full --runs runs --cache cache --splits splits.json --out results/grade_a5_full.json
```

## 8. nnU-Net baseline (H5)

```bash
pip install nnunetv2
export nnUNet_raw=... nnUNet_preprocessed=... nnUNet_results=...
python scripts/export_nnunet.py --root /data/BraTS2020 --splits splits.json --nnunet-raw $nnUNet_raw --dataset-id 501
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
for k in 0 1 2 3 4; do nnUNetv2_train 501 3d_fullres $k; done
```

Predict, for each fold model `k`, its validation fold and the test set (the exported test cases are in `imagesTs`; the
validation cases are listed in `splits_final.json` — copy their images into a folder first):

```bash
nnUNetv2_predict -i <val_images_fold_k> -o nnunet_pred/val_f$k  -d 501 -c 3d_fullres -f $k
nnUNetv2_predict -i $nnUNet_raw/Dataset501_BraTS2020/imagesTs -o nnunet_pred/test_f$k -d 501 -c 3d_fullres -f $k
python scripts/eval_nnunet.py --root /data/BraTS2020 --splits splits.json \
    --val-pred "nnunet_pred/val_f{k}" --test-pred "nnunet_pred/test_f{k}" --out nnunet
```

Check the `dataset.json` written by `export_nnunet.py` against the documentation of your nnU-Net version (the region syntax
has changed between versions). `eval_nnunet.py` assumes the region-based output convention described in its docstring
(`--label-space labels` for a plain multi-class model). Note the version of nnU-Net and the configuration you used.

## 9. Hardware statement and optional latency

```bash
nvidia-smi > gpu.txt; pip freeze > env.txt
python experiments/latency.py --patch 128 --repeats 10 --out latency.json
```

## 10. What to send back

```bash
tar czf brats_results.tgz runs/*/config.json runs/*/train_log.jsonl runs/*/val_metrics.csv runs/*/test_metrics.csv \
    results nnunet latency.json gpu.txt env.txt splits.json
```

(`splits.json` lists patient identifiers only.) With this archive the results chapter, the hypothesis verdicts, the abstract
and the discussion can be regenerated.

## 11. Rebuilding the thesis yourself (optional)

Requires `pandoc`, LibreOffice (`soffice`), `pdftotext` and `python-docx`.

```bash
mkdir -p brats && tar xzf brats_results.tgz -C brats           # layout: brats/runs, brats/results, brats/nnunet
python thesis/build_thesis.py thesis.docx --brats brats --template /path/to/your/reference_thesis.docx
```

Open the result in Word and press Ctrl+A, F9 to update the table of contents and the lists (page numbers in the file are
computed by LibreOffice and can differ by a page in Word). The abstract text (`thesis/content/abstract.txt`), Chapters 5 and 6
and the summary table of Chapter 6 describe the *status* of the results and must be re-read after the numbers are in.

## 12. Things that are easy to get wrong

* Different seeds, `quick.json`, or a different patch in one table: not comparable; state it or rerun.
* Evaluating on the test set during development: the test set is only touched by `train.py` at the end of each run and by
  the analysis; do not tune on it.
* Moving `splits.json` between runs: the leakage check fails or, worse, silently differs. Generate it once.

## Dry run of the BraTS pipeline on synthetic data (implementation check, not results)

To check that the whole BraTS path (training matrix, aggregation, grading, thesis tables) works before using real data:

1. `python -c "from bratsgan.synthetic import write_brats_layout; write_brats_layout('DR/raw', n_hgg=45, n_lgg=15, size=48, seed=11)"`
2. `python scripts/prepare_data.py --root DR/raw --cache DR/cache --splits DR/splits.json --folds 5 --test-frac 0.25`
3. `bash scripts/dryrun_matrix.sh DR` (20 tiny runs, about 30 minutes on two CPU cores)
4. `python thesis/build_thesis.py DRYRUN.docx --brats DR --dryrun`

The `--dryrun` flag puts a notice at the top of the BraTS section and the abstract. The numbers produced this way are not BraTS results. For the real experiment use step 7 of the procedure in Appendix B without `--dryrun`.
