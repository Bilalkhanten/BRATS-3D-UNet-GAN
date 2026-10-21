"""Regenerate the tiny sample dataset in data/sample (BraTS 2020 folder layout, 32^3 phantoms).

    python scripts/make_sample_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bratsgan.synthetic import write_brats_layout  # noqa: E402

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "sample"
    root = write_brats_layout(out, n_hgg=3, n_lgg=1, size=32, seed=11, difficulty="easy")
    print("wrote", root)
