# Examples

| Script | What it shows | Time |
| --- | --- | --- |
| `01_synthetic_quickstart.py` | generate and draw an HGG-like and an LGG-like phantom | seconds |
| `02_train_tiny_model.py` | generate, pre-process, split, train (A5) and evaluate a tiny network | about 30 s on a CPU |
| `03_inference_and_visualise.py RUN CASE` | segment a case with a trained run and save an overlay | seconds |
| `04_grade_from_masks.py` | feature extraction and cross-validated LGG/HGG classification | seconds |
| `05_compare_variants.py [PATCH]` | parameters and GMACs of A0 to A5 | seconds |

Run them from the repository root. The numbers they print come from synthetic data or from the architecture and are not BraTS results.
