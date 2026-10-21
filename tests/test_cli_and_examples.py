import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(*args, cwd=ROOT, timeout=600):
    return subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True, timeout=timeout)


def test_cli_help_lists_commands():
    r = _run("-m", "bratsgan.cli", "--help")
    assert r.returncode == 0
    for c in ("synth", "prepare", "train", "evaluate", "info"):
        assert c in r.stdout


def test_cli_synth_prepare_roundtrip(tmp_path):
    assert _run("-m", "bratsgan.cli", "synth", "--out", str(tmp_path / "raw"), "--hgg", "3", "--lgg", "1", "--size", "24").returncode == 0
    r = _run("-m", "bratsgan.cli", "prepare", "--root", str(tmp_path / "raw"), "--cache", str(tmp_path / "c"),
             "--splits", str(tmp_path / "s.json"), "--folds", "2", "--test-frac", "0.25")
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "s.json").exists() and len(list((tmp_path / "c").glob("*_img.npy"))) == 4


def test_sample_data_can_be_prepared(tmp_path):
    r = _run("-m", "bratsgan.cli", "prepare", "--root", str(ROOT / "data" / "sample"), "--cache", str(tmp_path / "c"),
             "--splits", str(tmp_path / "s.json"), "--folds", "2", "--test-frac", "0.25")
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("script", ["01_synthetic_quickstart.py", "04_grade_from_masks.py", "05_compare_variants.py"])
def test_fast_examples_run(script, tmp_path):
    args = ["examples/" + script] + ([str(tmp_path / "out.png")] if script.startswith("01") else (["32"] if script.startswith("05") else []))
    r = _run(*args)
    assert r.returncode == 0, r.stderr


@pytest.mark.slow
def test_train_example_runs(tmp_path):
    r = _run("examples/02_train_tiny_model.py", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "runs" / "demo" / "best.pt").exists()
