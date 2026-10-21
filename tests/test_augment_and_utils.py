import json

import numpy as np
import torch

from bratsgan.augment import augment_batch, intensity_aug, random_flip
from bratsgan.utils import JsonlLogger, jsonable, read_json, seed_everything, write_json


def test_augment_preserves_shapes_and_binary_labels():
    x = torch.randn(2, 4, 16, 16, 16)
    y = (torch.rand(2, 3, 16, 16, 16) > 0.7).float()
    xa, ya = augment_batch(x.clone(), y.clone())
    assert xa.shape == x.shape and ya.shape == y.shape
    assert set(torch.unique(ya).tolist()) <= {0.0, 1.0}


def test_flip_applies_identical_transform_to_image_and_label():
    torch.manual_seed(0)
    x = torch.arange(8 ** 3, dtype=torch.float32).reshape(1, 1, 8, 8, 8).repeat(1, 4, 1, 1, 1)
    y = x[:, :3].clone()
    for _ in range(10):
        xf, yf = random_flip(x.clone(), y.clone())
        assert torch.equal(xf[:, :3], yf)


def test_intensity_aug_leaves_labels_untouched_by_design():
    x = torch.randn(1, 4, 8, 8, 8)
    out = intensity_aug(x.clone(), p_noise=1, p_bright=1, p_contrast=1, p_gamma=0)
    assert out.shape == x.shape and not torch.equal(out, x)


def test_seed_everything_reproducible():
    seed_everything(5)
    a = (torch.rand(3), np.random.rand(3))
    seed_everything(5)
    b = (torch.rand(3), np.random.rand(3))
    assert torch.equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_json_helpers_roundtrip(tmp_path):
    obj = {"a": np.float32(1.5), "b": np.arange(3), "c": (1, 2)}
    write_json(tmp_path / "x.json", obj)
    assert read_json(tmp_path / "x.json") == {"a": 1.5, "b": [0, 1, 2], "c": [1, 2]}
    assert json.dumps(obj, default=jsonable)  # jsonable is the `default=` hook of json.dump


def test_jsonl_logger_appends(tmp_path):
    log = JsonlLogger(tmp_path / "log.jsonl")
    log.log(epoch=0, loss=1.0)
    log.log(epoch=1, loss=0.5)
    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    assert [json.loads(l)["epoch"] for l in lines] == [0, 1]
