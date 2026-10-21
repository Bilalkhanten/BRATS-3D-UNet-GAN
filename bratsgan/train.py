"""Two-stage training: (1) supervised pre-training of the segmentation network,
(2) adversarial refinement with a conditional 3-D PatchGAN discriminator.

Models that do not use the GAN continue stage 1 for the same total number of epochs, so that
GAN and non-GAN variants are compared at an equal number of generator updates.

Usage:  python -m bratsgan.train --config configs/full.json --set fold=0 name=full_f0
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from . import REGIONS
from .augment import augment_batch
from .config import load_config, parse_overrides
from .data import PatchStream, fold_ids, labels_to_regions, load_case
from .evaluate import evaluate_ids, summarise
from .infer import predict_regions
from .losses import SegmentationLoss, adv_ramp, d_loss_hinge, d_loss_lsgan, g_loss_hinge, g_loss_lsgan
from .metrics import dice, enforce_hierarchy
from .models import Discriminator3D, build_model
from .utils import JsonlLogger, read_json, seed_everything, write_json


def poly_lr(base_lr, epoch, total, power):
    return base_lr * (1 - epoch / total) ** power


@torch.no_grad()
def quick_validate(model, cache_dir, ids, patch, overlap, device, hierarchy=True):
    model.eval()
    ds = {k: [] for k in REGIONS}
    for cid in ids:
        img, seg, _ = load_case(cache_dir, cid, mmap=False)
        gt = labels_to_regions(np.asarray(seg))
        pred, _, _ = predict_regions(model, img, patch, overlap, False, device=device)
        if hierarchy:
            pred = enforce_hierarchy(pred)
        for i, k in enumerate(REGIONS):
            ds[k].append(dice(pred[i], gt[i]))
    out = {f"val_dice_{k}": float(np.mean(v)) for k, v in ds.items()}
    out["val_dice_mean"] = float(np.mean(list(out.values())))
    model.train()
    return out


def run(cfg: dict) -> Path:
    seed_everything(cfg["seed"], cfg["deterministic"])
    device = torch.device("cuda" if (cfg["device"] == "auto" and torch.cuda.is_available()) else (cfg["device"] if cfg["device"] != "auto" else "cpu"))
    out = Path(cfg["out_dir"]) / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", cfg)
    log = JsonlLogger(out / "train_log.jsonl")

    splits = read_json(cfg["splits"])
    train_ids, val_ids, _ = fold_ids(splits, cfg["fold"])
    stream = PatchStream(cfg["cache_dir"], train_ids, cfg["patch"], cfg["fg_prob"], cfg["seed"])
    loader = iter(DataLoader(stream, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], pin_memory=device.type == "cuda",
                             persistent_workers=cfg["num_workers"] > 0))

    G = build_model(cfg).to(device)
    seg_loss = SegmentationLoss(cfg["w_dice"], cfg["w_tv"], cfg["w_bce"], cfg["tv_alpha"], cfg["tv_beta"], cfg["tv_gamma"])
    opt_g = torch.optim.AdamW(G.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    D = opt_d = None
    use_amp = cfg["amp"] and device.type == "cuda"
    total_epochs = cfg["epochs_seg"] + cfg["epochs_adv"]
    best, start_epoch = -1.0, 0
    ck = out / "last.pt"
    if ck.exists():
        st = torch.load(ck, map_location=device)
        G.load_state_dict(st["G"]); opt_g.load_state_dict(st["opt_g"]); best, start_epoch = st["best"], st["epoch"] + 1
        print(f"resumed from epoch {start_epoch}")

    def make_d():
        d = Discriminator3D(cfg["in_ch"], cfg["out_ch"], cfg["ndf"]).to(device)
        return d, torch.optim.Adam(d.parameters(), lr=cfg["d_lr"], betas=tuple(cfg["d_betas"]))

    d_loss_fn = d_loss_lsgan if cfg["adv_loss"] == "lsgan" else d_loss_hinge
    g_loss_fn = g_loss_lsgan if cfg["adv_loss"] == "lsgan" else g_loss_hinge
    adv_step = 0
    G.train()
    for epoch in range(start_epoch, total_epochs):
        stage = "adv" if (cfg["gan"] and epoch >= cfg["epochs_seg"]) else "seg"
        if stage == "adv" and D is None:
            D, opt_d = make_d()
            if ck.exists() and "D" in torch.load(ck, map_location=device):
                st = torch.load(ck, map_location=device); D.load_state_dict(st["D"]); opt_d.load_state_dict(st["opt_d"])
        lr = poly_lr(cfg["lr"], epoch, total_epochs, cfg["poly_power"])
        for g in opt_g.param_groups:
            g["lr"] = lr
        t0 = time.time()
        agg = {}
        for it in range(cfg["iters_per_epoch"]):
            x, y, _ = next(loader)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x, y = augment_batch(x, y)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = G(x)
            loss, parts = seg_loss(logits, y)
            stats = {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in parts.items()}
            if stage == "adv":
                p_fake = torch.sigmoid(logits[0].float() if isinstance(logits, (list, tuple)) else logits.float())
                lam = cfg["adv_weight"] * adv_ramp(adv_step, cfg["adv_ramp_epochs"] * cfg["iters_per_epoch"])
                # --- discriminator
                y_real = y * (1 - cfg["real_smooth"]) + 0.5 * cfg["real_smooth"]
                for _ in range(cfg["d_steps"]):
                    noise_r = cfg["d_noise"] * torch.randn_like(y_real)
                    noise_f = cfg["d_noise"] * torch.randn_like(p_fake)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                        d_real = D(x, y_real + noise_r)
                        d_fake = D(x, p_fake.detach() + noise_f)
                    ld = d_loss_fn(d_real.float(), d_fake.float())
                    opt_d.zero_grad(set_to_none=True); ld.backward(); opt_d.step()
                # --- generator
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    d_out = D(x, p_fake)
                lg = g_loss_fn(d_out.float())
                loss = loss + lam * lg
                stats.update(adv=float(lg), d_loss=float(ld), lam=lam,
                             d_acc_real=float((d_real > 0.5).float().mean()), d_acc_fake=float((d_fake < 0.5).float().mean()))
                adv_step += 1
            opt_g.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), cfg["grad_clip"])
            opt_g.step()
            stats["loss"] = float(loss)
            for k, v in stats.items():
                agg.setdefault(k, []).append(v)
        rec = {k: float(np.mean(v)) for k, v in agg.items()}
        rec.update(epoch=epoch, stage=stage, lr=lr, epoch_seconds=time.time() - t0)
        if (epoch + 1) % cfg["val_every"] == 0 or epoch == total_epochs - 1:
            rec.update(quick_validate(G, cfg["cache_dir"], val_ids, cfg["patch"], cfg["val_overlap"], device, cfg["hierarchy"]))
            if rec["val_dice_mean"] > best:
                best = rec["val_dice_mean"]
                torch.save({"model": G.state_dict(), "epoch": epoch, "val": rec}, out / "best.pt")
                rec["is_best"] = True
        log.log(**rec)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
        st = {"G": G.state_dict(), "opt_g": opt_g.state_dict(), "best": best, "epoch": epoch}
        if D is not None:
            st.update(D=D.state_dict(), opt_d=opt_d.state_dict())
        torch.save(st, ck)
    if not (out / "best.pt").exists():
        torch.save({"model": G.state_dict(), "epoch": total_epochs - 1}, out / "best.pt")
    # ---- final evaluation with the best checkpoint
    st = torch.load(out / "best.pt", map_location=device)
    G.load_state_dict(st["model"]); G.eval()
    _, val_ids, test_ids = fold_ids(splits, cfg["fold"])
    for name, ids in (("val", val_ids), ("test", test_ids)):
        df = evaluate_ids(G, cfg["cache_dir"], ids, cfg["patch"], cfg["final_overlap"], cfg["final_tta"], device,
                          save_pred_dir=str(out / f"pred_{name}"), hierarchy=cfg["hierarchy"])
        df.to_csv(out / f"{name}_metrics.csv", index=False)
        write_json(out / f"{name}_summary.json", summarise(df))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides (JSON values)")
    a = ap.parse_args()
    run(load_config(a.config, parse_overrides(a.set)))


if __name__ == "__main__":
    main()
