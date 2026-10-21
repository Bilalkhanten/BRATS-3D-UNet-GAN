"""Default hyper-parameters. Every experiment is a JSON file that overrides a subset of these."""
from __future__ import annotations
import json
from pathlib import Path

DEFAULTS = dict(
    name="exp", cache_dir="cache", splits="splits.json", fold=0, out_dir="runs",
    # architecture
    in_ch=4, out_ch=3, base=32, depth=5, max_ch=320, residual=True, attention=True, fusion="maf",
    norm="instance", deep_supervision=True, dropout=0.0,
    # data / schedule
    patch=[128, 128, 128], batch_size=2, iters_per_epoch=250, epochs_seg=300, epochs_adv=50, num_workers=4, fg_prob=0.67,
    # optimisation
    lr=3e-4, weight_decay=1e-5, poly_power=0.9, grad_clip=12.0,
    # segmentation loss
    w_dice=1.0, w_tv=1.0, w_bce=1.0, tv_alpha=0.3, tv_beta=0.7, tv_gamma=1.0,
    # adversarial refinement
    gan=True, ndf=32, d_lr=2e-4, d_betas=[0.5, 0.999], adv_weight=0.05, adv_ramp_epochs=5, d_steps=1,
    adv_loss="lsgan", real_smooth=0.05, d_noise=0.05,
    # evaluation
    val_every=5, val_overlap=0.5, final_overlap=0.5, final_tta=False, hierarchy=True,
    # misc
    seed=42, amp=True, deterministic=False, device="auto",
)


def load_config(path: str | None, overrides: dict | None = None) -> dict:
    cfg = dict(DEFAULTS)
    if path:
        cfg.update(json.loads(Path(path).read_text()))
    if overrides:
        cfg.update(overrides)
    return cfg


def parse_overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out
