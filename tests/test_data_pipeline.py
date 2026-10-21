import numpy as np

from bratsgan.data import (assert_no_leakage, find_cases, fold_ids, load_case, make_splits, normalise_volume,
                           preprocess_case)
from bratsgan.synthetic import write_brats_layout


def test_find_preprocess_and_load(tmp_path):
    raw = write_brats_layout(tmp_path / "raw", n_hgg=2, n_lgg=2, size=24, seed=4)
    cases = find_cases(str(raw))
    assert [c["grade"] for c in cases] == ["HGG", "HGG", "LGG", "LGG"]
    meta = preprocess_case(cases[0], tmp_path / "cache")
    img, seg, m = load_case(tmp_path / "cache", cases[0]["id"], mmap=False)
    assert img.shape[0] == 4 and img.shape[1:] == seg.shape == tuple(meta["shape"])
    assert m["grade"] == "HGG"


def test_normalisation_is_zscore_inside_brain():
    rng = np.random.default_rng(0)
    vol = rng.normal(100, 20, (16, 16, 16)).astype(np.float32)
    brain = np.zeros(vol.shape, bool)
    brain[4:12, 4:12, 4:12] = True
    out = normalise_volume(vol, brain)
    assert abs(out[brain].mean()) < 1e-4 and abs(out[brain].std() - 1) < 1e-2
    assert (out[~brain] == 0).all()


def test_splits_are_disjoint_stratified_and_reproducible():
    cases = [{"id": f"c{i:03d}", "grade": "HGG" if i % 4 else "LGG"} for i in range(80)]
    s1, s2 = make_splits(cases, seed=3, test_frac=0.25, k=4), make_splits(cases, seed=3, test_frac=0.25, k=4)
    assert s1 == s2
    assert_no_leakage(s1)
    allids = s1["test"] + [i for f in s1["folds"] for i in f]
    assert sorted(allids) == sorted(c["id"] for c in cases)
    n_lgg_test = sum(1 for i in s1["test"] if int(i[1:]) % 4 == 0)
    assert n_lgg_test == round(0.25 * 20)  # grade-stratified test set


def test_fold_ids_partition():
    cases = [{"id": f"c{i:02d}", "grade": "HGG"} for i in range(30)]
    s = make_splits(cases, seed=0, test_frac=0.2, k=3)
    for k in range(3):
        tr, va, te = fold_ids(s, k)
        assert not set(tr) & set(va) and not set(tr) & set(te) and not set(va) & set(te)
