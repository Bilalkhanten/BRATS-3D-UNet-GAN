import numpy as np

from bratsgan.complexity import count_params, profile
from bratsgan.data import labels_to_regions
from bratsgan.features import extract_features
from bratsgan.models import UNet3D
from bratsgan.synthetic import make_phantom


def test_feature_vector_is_finite_and_has_101_entries():
    vols, lab, _ = make_phantom(size=32, grade="HGG", seed=1)
    f = extract_features(vols, labels_to_regions(lab))
    assert len(f) == 101
    assert all(np.isfinite(v) for v in f.values())


def test_features_handle_missing_enhancing_region():
    vols, lab, _ = make_phantom(size=32, grade="LGG", seed=6)
    reg = labels_to_regions(lab)
    reg[2] = 0  # no enhancing tumour predicted
    f = extract_features(vols, reg)
    assert all(np.isfinite(v) for v in f.values())


def test_features_distinguish_grades_on_reference_masks():
    def logvol_et(g, s):
        vols, lab, _ = make_phantom(size=32, grade=g, seed=s)
        return extract_features(vols, labels_to_regions(lab))["logvol_ET"]
    hgg = np.mean([logvol_et("HGG", s) for s in range(5)])
    lgg = np.mean([logvol_et("LGG", s) for s in range(5)])
    assert hgg > lgg


def test_profile_matches_parameter_count_and_scales_with_volume():
    m = UNet3D(base=4, depth=3, max_ch=16)
    a = profile(m, (1, 4, 16, 16, 16))
    b = profile(m, (1, 4, 32, 32, 32))
    assert a["params"] == count_params(m)
    assert 7.0 < b["gmacs"] / a["gmacs"] < 9.0  # cost is proportional to the number of voxels (x8)
