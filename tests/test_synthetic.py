import numpy as np

from bratsgan.data import labels_to_regions
from bratsgan.synthetic import make_phantom, write_brats_layout


def test_phantom_shapes_and_labels():
    vols, lab, brain = make_phantom(size=32, grade="HGG", seed=1)
    assert vols.shape == (4, 32, 32, 32) and lab.shape == (32, 32, 32) and brain.shape == (32, 32, 32)
    assert set(np.unique(lab)) <= {0, 1, 2, 4}
    assert (lab == 4).sum() > 0  # HGG-like phantoms have an enhancing shell


def test_phantom_is_deterministic():
    a = make_phantom(size=24, grade="LGG", seed=7)[0]
    b = make_phantom(size=24, grade="LGG", seed=7)[0]
    c = make_phantom(size=24, grade="LGG", seed=8)[0]
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_regions_are_nested():
    for g in ("HGG", "LGG"):
        _, lab, _ = make_phantom(size=32, grade=g, seed=3)
        wt, tc, et = labels_to_regions(lab)
        assert (et <= tc).all() and (tc <= wt).all()


def test_hard_setting_is_harder():
    easy = make_phantom(size=48, grade="HGG", seed=2, difficulty="easy")[1]
    hard = make_phantom(size=48, grade="HGG", seed=2, difficulty="hard")[1]
    assert (hard > 0).sum() < (easy > 0).sum()  # smaller lesions


def test_write_brats_layout(tmp_path):
    root = write_brats_layout(tmp_path, n_hgg=2, n_lgg=1, size=16, seed=0)
    cases = sorted(p.name for p in (root / "MICCAI_BraTS2020_TrainingData").iterdir() if p.is_dir())
    assert cases == ["BraTS20_Training_001", "BraTS20_Training_002", "BraTS20_Training_003"]
    assert (root / "MICCAI_BraTS2020_TrainingData" / "name_mapping.csv").exists()
