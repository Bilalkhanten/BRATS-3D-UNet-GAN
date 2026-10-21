"""Chapter 4, section 4.3: synthetic phantom study (implementation verification; NOT BraTS evidence).

Every number in the generated text is computed here from the run directories written by bratsgan.train.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "experiments"))
import analysis as AN  # noqa: E402
from bratsgan import stats as S  # noqa: E402

FIGDIR = HERE / "figs_phantom"
SUPP = ""
ASSET_FIG = HERE / "assets" / "figs"
FROZEN = HERE / "assets" / "phantom_frozen"
VARIANTS = [("a0_unet", "A0", "Plain 3D U-Net"), ("a1_res", "A1", "+ residual blocks"), ("a2_att", "A2", "+ attention gates"),
            ("a3_maf", "A3", "+ MAF stem"), ("a4_tversky", "A4", "+ Tversky loss"), ("a5_full", "A5", "+ adversarial term (full)")]
SEEDS = (1, 2)
W = Path(os.environ.get("BRATSGAN_PHANTOM_STUDY", HERE.parent / "runs" / "phantom_study"))


def _dirs(runs, cfg, fold=0):
    ds = [Path(runs) / f"{cfg}_s{s}_f{fold}" for s in SEEDS]
    return [d for d in ds if (d / "test_metrics.csv").exists()]


def _pending(msg):
    return f"~{msg}"


def build(phantom_dir=None) -> str:
    global SUPP
    base = Path(phantom_dir or W)
    runs = base / "runs"
    FIGDIR.mkdir(exist_ok=True)
    from figstyle import C, LS, MK, plt, setup
    setup()

    avail = []
    for cfg, tag, name in VARIANTS:
        ds = _dirs(runs, cfg)
        if ds:
            avail.append((cfg, tag, name, ds))
    if not avail:
        fz = FROZEN / "ph.txt"
        if fz.exists():
            SUPP = (FROZEN / "supp.txt").read_text().replace("{FIGDIR}", str(FIGDIR))
            FIGDIR.mkdir(exist_ok=True)
            import shutil
            for f in FROZEN.glob("*.png"):
                if not (FIGDIR / f.name).exists():
                    shutil.copy(f, FIGDIR / f.name)
            return fz.read_text().replace("{FIGDIR}", str(FIGDIR)).replace("{ASSETFIG}", str(ASSET_FIG))
        return "@section[phantom] Synthetic Phantom Study\n~The phantom runs were not available when this document was built.\n\n@subsection[v10] Diagnostics of the discriminator\n~Not available.\n"

    splits = json.loads((base / "splits.json").read_text())
    n_test = len(splits["test"])
    n_tr = len(splits["folds"][0]) if isinstance(splits["folds"][0], list) else None
    cfg0 = json.loads((avail[0][3][0] / "config.json").read_text())

    out = []
    add = out.append

    # -------------------------------------------------------------- design
    add("@section[phantom] Synthetic Phantom Study\n")
    seeds_used = sorted({int(d.name.split("_s")[1].split("_")[0]) for *_, ds in avail for d in ds})
    add(f"""The second kind of evidence is an end-to-end run of the complete training and evaluation pipeline on synthetic data with known ground truth. Its purpose is to establish that the pipeline, as implemented, learns, converges, evaluates, aggregates and tests as designed; it is not intended to rank the model variants for brain tumours (see the note in Section @sec:r-overview).

@subsection[ph-design] Design

One hundred phantoms of ${cfg0['patch'][0]}^3$ voxels were generated with the generator described in Appendix @app:phantom-gen: 75 are labelled HGG-like and 25 LGG-like, with an ellipsoidal brain with three tissue classes, four MR-like contrasts, a smooth multiplicative bias field, Rician-like noise, and a nested tumour (necrotic core, enhancing shell, oedema) written with the BraTS labels $\\{{1,2,4\\}}$. HGG-like phantoms have a strongly enhancing shell and a large core, LGG-like phantoms weak or absent enhancement. @fig:ph-examples shows two examples. The phantoms were split at the level of the phantom with the same code as for BraTS: a grade-stratified test set of {n_test} phantoms and three cross-validation folds on the remaining {100-n_test}; only fold 0 was trained.

@fig ph-examples | file={ASSET_FIG}/phantom_examples.png | width=6.2
Two synthetic phantoms (upper row: HGG-like, lower row: LGG-like). From left to right: T1, T1ce, T2, FLAIR, and the ground-truth labels.
@end

The network variants A0 to A5 are the cumulative steps of the ablation of Chapter 3 ({', '.join(t for _, t, _, _ in avail)} available in this build), each built with {cfg0['depth']} levels, {cfg0['base']} base channels (at most {cfg0['max_ch']}) and trained for {cfg0['epochs_seg']} epochs of {cfg0['iters_per_epoch']} iterations with batch size {cfg0['batch_size']} on $48^3$ patches (A5 adds {cfg0['epochs_adv']} adversarial epochs; the other variants are trained for the same total number of generator updates), AdamW with learning rate $10^{{-3}}$, on the CPU in single precision. The width and depth are therefore much smaller than the BraTS configuration of Chapter 3 (base width 32, depth 5), but the code path is the same. Each variant was trained with seed(s) {', '.join(map(str, seeds_used))}; where several seeds exist the per-phantom metrics are averaged over seeds before any statistic is computed, so that the unit of analysis remains the phantom.
""")

    # -------------------------------------------------------------- results
    frames = {}
    for cfg, tag, name, ds in avail:
        frames[cfg] = AN.load_metrics(ds, "test")
    rows, per_seed = [], {}
    for cfg, tag, name, ds in avail:
        df = frames[cfg]
        sm = []
        for d in ds:
            m = AN.add_composite(pd.read_csv(d / "test_metrics.csv"))
            sm.append(m["dice_mean"].mean())
        per_seed[cfg] = sm
        c = AN.ci(df["dice_mean"].values)
        sd = np.std(sm, ddof=1) if len(sm) > 1 else float("nan")
        rows.append((tag, name, len(ds), df["dice_WT"].mean(), df["dice_TC"].mean(), df["dice_ET"].mean(), c, df["hd95_mean"].mean(), sd))
    tab = "\n".join("| " + " | ".join([r[0] + " " + r[1], str(r[2]), AN.fmt(r[3], 4), AN.fmt(r[4], 4), AN.fmt(r[5], 4), AN.fmt_ci(r[6], 4), AN.fmt(r[7], 2),
                                       ("%.4f" % r[8]) if r[8] == r[8] else "n/a"]) + " |" for r in rows)
    best = max(rows, key=lambda r: r[6][0])
    worst = min(rows, key=lambda r: r[6][0])
    spread = best[6][0] - worst[6][0]
    all_dice = np.concatenate([frames[c]["dice_mean"].values for c in frames])
    lo_case = min(frames[c]["dice_mean"].min() for c in frames)
    add(f"""@subsection[ph-res] Segmentation accuracy on the held-out phantoms

@tab:ph-main lists the accuracy on the {n_test} test phantoms. All variants reach a mean Dice above {min(r[6][0] for r in rows):.3f} for the average over the three regions; the largest difference between the means of any two variants is {spread:.4f} ({best[0]} against {worst[0]}), and the lowest Dice of any single test phantom and variant is {lo_case:.3f}. The task is therefore close to saturation: phantoms contain no anatomy, no artefacts and no label ambiguity, and the contrast between the tumour compartments is large compared with the noise. Under these conditions the pipeline can be shown to work, but it cannot discriminate between architectures, and the differences below must not be interpreted as evidence about the value of any component.

@tab ph-main | widths=1.75,0.5,0.75,0.75,0.75,1.35,0.7,0.7 | align=lcccccrc | size=8
Test results of the phantom study (fold 0, {n_test} test phantoms). Per-phantom metrics are averaged over seeds; the interval is a 95% bootstrap interval of the mean over phantoms. HD95 in voxels (mean over WT, TC, ET). Last column: standard deviation of the mean Dice between seeds.
| Variant | Seeds | Dice WT | Dice TC | Dice ET | Dice mean [95% CI] | HD95 | SD seeds |
| --- | --- | --- | --- | --- | --- | --- | --- |
{tab}
@end
""")

    # steps
    steps = []
    order = [a for a in avail]
    for (c1, t1, n1, _), (c2, t2, n2, _) in zip(order[:-1], order[1:]):
        pd_ = AN.paired(frames[c2], frames[c1], "dice_mean")
        ph_ = AN.paired(frames[c2], frames[c1], "hd95_mean")
        steps.append((f"{t2} vs {t1}", pd_, ph_))
    holm = S.holm([s[1]["p"] for s in steps]) if steps else []
    srows = "\n".join("| " + " | ".join([s[0], AN.fmt_ci((s[1]["diff"], s[1]["lo"], s[1]["hi"]), 4), AN.fmt_p(float(h)),
                                         AN.fmt_ci((s[2]["diff"], s[2]["lo"], s[2]["hi"]), 2), f"{s[1]['dz']:.2f}"]) + " |"
                      for s, h in zip(steps, holm))
    n_sig = int(sum(1 for s in steps if s[1]["lo"] > 0 or s[1]["hi"] < 0))
    seed_sd = [np.std(v, ddof=1) for v in per_seed.values() if len(v) > 1]
    msd = float(np.mean(seed_sd)) if seed_sd else float("nan")
    mde = 2.80 * np.std(frames[avail[-1][0]]["dice_mean"].values - frames[avail[0][0]]["dice_mean"].values, ddof=1) / np.sqrt(n_test)
    maxd = max((abs(x[1]["diff"]) for x in steps), default=float("nan"))
    msd_txt = f"{msd:.4f}" if msd == msd else "not available (single seed)"
    cmp_txt = (f"The mean seed-to-seed standard deviation of the mean Dice is {msd:.4f}, " +
               ("larger than" if msd > maxd else "smaller than") + f" the largest step difference ({maxd:.4f})") if msd == msd else \
        f"With a single seed the seed-to-seed variability cannot be estimated; the largest step difference is {maxd:.4f}"
    add(f"""@tab:ph-steps gives the paired differences between successive variants. Of the {len(steps)} comparisons, {n_sig} have a bootstrap interval that excludes zero, and the largest absolute difference is {maxd:.4f} Dice units. {cmp_txt}. With {n_test} test phantoms the minimum detectable paired difference of @eq:mde is about {mde:.4f} for the comparison of the first and the last variant. Where an interval excludes zero, the difference is a property of this phantom population and of the training budget of the study; it is small in absolute terms and is not used as evidence for or against any component on patient data.

@tab ph-steps | widths=1.2,1.9,0.9,1.9,0.7 | align=lcccc | size=8.5
Paired differences between successive variants on the phantom test set (later minus earlier). The p-value is a two-sided Wilcoxon signed-rank test with Holm correction over the comparisons of the table. Dice difference with 95% bootstrap interval; HD95 difference in voxels; last column: Cohen's $d_z$ of the Dice difference.
| Comparison | $\\Delta$ Dice mean [95% CI] | p (Holm) | $\\Delta$ HD95 [95% CI] | $d_z$ |
| --- | --- | --- | --- | --- |
{srows}
@end
""")

    # curves
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    sel = [a for a in avail if a[0] in ("a0_unet", "a5_full")] or avail[:1]
    for k, (cfg, tag, name, ds) in enumerate(sel):
        L = AN.read_log(ds[0])
        ax[0].plot(L["epoch"], L["loss"], color=[C["blue"], C["orange"]][k], ls=["-", "--"][k], label=f"{tag} {name}")
        v = L.dropna(subset=["val_dice_mean"])
        ax[1].plot(v["epoch"], 1 - v["val_dice_mean"], color=[C["blue"], C["orange"]][k], ls=["-", "--"][k], marker=MK[k], ms=4, label=f"{tag} {name}")
    ax[0].set_yscale("log"); ax[1].set_yscale("log")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training generator loss"); ax[0].set_title("(a) training loss")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("1 $-$ validation Dice (mean)"); ax[1].set_title("(b) validation error")
    for a_ in ax:
        a_.axvline(cfg0["epochs_seg"] - 0.5, color=C["grey"], lw=0.8, ls=":")
    ax[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGDIR / "ph_curves.png"); plt.close(fig)
    L_all = AN.read_log(avail[0][3][0])
    L0 = L_all.dropna(subset=["val_dice_mean"])
    add(f"""@subsection[ph-curves] Convergence

@fig ph-curves | file={FIGDIR}/ph_curves.png | width=6.2
Training curves of the phantom study (seed 1): (a) training loss of the generator, (b) one minus the validation Dice, both on a logarithmic axis. The dotted vertical line marks the start of the adversarial epochs.
@end

@fig:ph-curves shows that the training loss of the first run falls by a factor of {L_all['loss'].iloc[0]/L_all['loss'].iloc[-1]:.0f} between the first and the last epoch and that the validation error falls from {1-L0['val_dice_mean'].iloc[0]:.2f} at epoch {int(L0['epoch'].iloc[0])} to {1-L0['val_dice_mean'].iloc[-1]:.4f} at epoch {int(L0['epoch'].iloc[-1])} for {avail[0][1]}. {_final_err_sentence(avail, cfg0)}
""")

    # D diagnostics
    a5 = [a for a in avail if a[0] == "a5_full"]
    if a5:
        cfg, tag, name, ds = a5[0]
        fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7))
        summ = []
        for k, d in enumerate(ds):
            L = AN.read_log(d)
            A_ = L[L["stage"] == "adv"]
            if A_.empty:
                continue
            ax[0].plot(A_["epoch"], A_["d_acc_real"], color=C["blue"], ls=LS[k % 4], alpha=0.9)
            ax[0].plot(A_["epoch"], A_["d_acc_fake"], color=C["orange"], ls=LS[k % 4], alpha=0.9)
            ax[1].plot(A_["epoch"], A_["d_loss"], color=C["violet"], ls=LS[k % 4])
            ax[1].plot(A_["epoch"], A_["adv"], color=C["aqua"], ls=LS[k % 4])
            ax[2].plot(A_["epoch"], A_["lam"], color=C["grey"], ls=LS[k % 4])
            summ.append((d.name, A_["d_acc_real"].iloc[-1], A_["d_acc_fake"].iloc[-1], A_["d_loss"].iloc[-1], A_["adv"].iloc[-1], A_["lam"].iloc[-1],
                         A_["d_acc_real"].mean(), A_["d_acc_fake"].mean()))
        ax[0].plot([], [], color=C["blue"], label="real"); ax[0].plot([], [], color=C["orange"], label="fake")
        ax[1].plot([], [], color=C["violet"], label="$D$ loss"); ax[1].plot([], [], color=C["aqua"], label="adversarial term of $G$")
        ax[0].set_ylim(-0.02, 1.02); ax[0].set_title("(a) $D$ accuracy"); ax[0].legend(frameon=False, loc="center right")
        ax[1].set_title("(b) adversarial losses"); ax[1].legend(frameon=False)
        ax[2].set_title("(c) weight $\\lambda$")
        for a_ in ax:
            a_.set_xlabel("epoch")
        fig.tight_layout(); fig.savefig(FIGDIR / "ph_d.png"); plt.close(fig)
        mr = float(np.mean([s[6] for s in summ])) if summ else float("nan")
        mf = float(np.mean([s[7] for s in summ])) if summ else float("nan")
        lastr = float(np.mean([s[1] for s in summ])) if summ else float("nan")
        lastf = float(np.mean([s[2] for s in summ])) if summ else float("nan")
        lam_end = summ[0][5] if summ else float("nan")
        trivial = (mr > 0.95 and mf > 0.95)
        add(f"""@subsection[v10] Diagnostics of the discriminator

@fig:ph-d shows the quantities that the training code records at every adversarial epoch for the {len(summ)} run(s) of the full model: the accuracy of the discriminator on real and generated label maps (fraction of patch-level outputs above or below 0.5), the two adversarial losses, and the weight $\\lambda$ of the adversarial term, which rises linearly to {lam_end:.3f} over {cfg0['adv_ramp_epochs']} epochs.

@fig ph-d | file={FIGDIR}/ph_d.png | width=6.4
Discriminator diagnostics of the full model during the adversarial epochs: (a) accuracy on real and generated label maps, (b) discriminator loss and adversarial loss of the generator, (c) weight of the adversarial term. Line styles distinguish seeds.
@end

Averaged over the adversarial epochs, the discriminator classifies {100*mr:.1f}% of the real and {100*mf:.1f}% of the generated outputs correctly; at the last epoch the figures are {100*lastr:.1f}% and {100*lastf:.1f}%. {"A discriminator that is correct on both classes almost always has separated them trivially, the situation analysed in Section @sec:gan-model: the label smoothing and instance noise cannot prevent it when the generator output is very close to a binary map and the real maps are exactly binary, and the adversarial gradient then carries little information beyond what the supervised loss already provides." if trivial else "The discriminator is therefore not trivially separating the two classes, which is the regime in which the adversarial gradient is informative; this is the behaviour that the label smoothing, the instance noise and the small weight of the adversarial term were designed to produce."} The diagnostics are recorded for the BraTS runs in exactly the same way and are reported in Section @sec:brats-res; they should be read together with the Dice and HD95 differences, because a stable discriminator without an effect on the metrics is the expected outcome of a small adversarial weight.
""")

    else:
        add("@subsection[v10] Diagnostics of the discriminator\n\n~The full model (A5) was not yet trained when this version was built.\n")

    # maps
    add(_maps_section(avail, base, runs, plt, C))

    # grade
    add(_grade_section(base, runs))
    SUPP = _supp(avail)
    text = "\n".join(out)
    if complete_for_freeze(avail):
        FROZEN.mkdir(parents=True, exist_ok=True)
        (FROZEN / "ph.txt").write_text(text.replace(str(FIGDIR), "{FIGDIR}").replace(str(ASSET_FIG), "{ASSETFIG}"))
        (FROZEN / "supp.txt").write_text(SUPP.replace(str(FIGDIR), "{FIGDIR}"))
    return text


# ---------------------------------------------------------------------------
def _load_model(d, device="cpu"):
    import torch
    from bratsgan.models import build_model
    cfg = json.loads((d / "config.json").read_text())
    m = build_model(cfg)
    st = torch.load(d / "best.pt", map_location=device)
    m.load_state_dict(st["model"] if "model" in st else st)
    return m.eval(), cfg


def _crop(img, seg, p):
    """crop/pad a (C,D,H,W) volume to p^3 around the whole-tumour centroid"""
    idx = np.argwhere(np.asarray(seg) > 0)
    c = idx.mean(0).round().astype(int)
    lo = [int(np.clip(ci - p // 2, 0, max(s - p, 0))) for ci, s in zip(c, img.shape[1:])]
    sl = tuple(slice(l, l + p) for l in lo)
    x = np.asarray(img)[(slice(None),) + sl]
    y = np.asarray(seg)[sl]
    pad = [(0, 0)] + [(0, p - s) for s in x.shape[1:]]
    return np.pad(x, pad), np.pad(y, [(0, p - s) for s in y.shape])


def _maps_section(avail, base, runs, plt, C):
    import torch
    import torch.nn.functional as F
    from bratsgan.data import fold_ids, load_case
    have = {a[0]: a for a in avail}
    splits = json.loads((base / "splits.json").read_text())
    _, _, test_ids = fold_ids(splits, 0)
    cache = json.loads((have[next(iter(have))][3][0] / "config.json").read_text())["cache_dir"]
    txt = ["@subsection[ph-maps] Internal behaviour of the attention gates and of the fusion stem\n"]
    p = 48
    # ---- attention gates (A2 or later variants with attention)
    att = [a for a in avail if a[0] in ("a2_att", "a3_maf", "a4_tversky", "a5_full")]
    if att:
        cfg, tag, name, ds = att[0]
        inside, outside = [], []
        ex = None
        for d in ds:
            m, c = _load_model(d)
            for cid in test_ids:
                img, seg, meta = load_case(cache, cid, mmap=False)
                x, y = _crop(img, seg, p)
                with torch.no_grad():
                    m(torch.from_numpy(x[None]).float())
                brain = (np.abs(x).sum(0) > 0)
                row_in, row_out, maps = [], [], []
                for g in m.gates:
                    a_ = F.interpolate(g.last_alpha, size=(p, p, p), mode="trilinear", align_corners=False)[0, 0].numpy()
                    maps.append(a_)
                    row_in.append(float(a_[y > 0].mean()) if (y > 0).any() else np.nan)
                    row_out.append(float(a_[brain & (y == 0)].mean()))
                inside.append(row_in); outside.append(row_out)
                if ex is None and d == ds[0] and meta.get("grade") == "HGG":
                    ex = (x, y, maps, cid)
        inside, outside = np.array(inside), np.array(outside)
        ng = inside.shape[1]
        rows = "\n".join(f"| {gi} | {np.nanmean(inside[:, gi]):.3f} | {np.nanmean(outside[:, gi]):.3f} | {np.nanmean(inside[:, gi]) - np.nanmean(outside[:, gi]):+.3f} |" for gi in range(ng))
        fig, ax = plt.subplots(1, ng + 1, figsize=(1.9 * (ng + 1), 2.2))
        x, y, maps, cid = ex
        zc = int(np.argwhere(y > 0)[:, 0].mean())
        ax[0].imshow(x[1, zc], cmap="gray"); ax[0].contour(y[zc] > 0, levels=[0.5], colors=[C["red"]], linewidths=0.8); ax[0].set_title("T1ce + WT", fontsize=8)
        for gi in range(ng):
            im = ax[gi + 1].imshow(maps[gi][zc], cmap="viridis", vmin=0, vmax=1)
            ax[gi + 1].contour(y[zc] > 0, levels=[0.5], colors=["white"], linewidths=0.6)
            ax[gi + 1].set_title(f"gate {gi}", fontsize=8)
        for a_ in ax:
            a_.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        fig.savefig(FIGDIR / "ph_gates.png"); plt.close(fig)
        txt.append(f"""@fig ph-gates | file={FIGDIR}/ph_gates.png | width=6.4
Attention coefficients $\\alpha$ of the gates of variant {tag} on an HGG-like test phantom (central axial slice through the tumour), upsampled to the input resolution. Gate 0 acts on the highest-resolution skip connection. The white contour is the whole-tumour ground truth.
@end

@fig:ph-gates shows the coefficients of the gates of the first available variant with attention ({tag}, {len(ds)} seed(s)) and @tab:ph-gates their mean inside and outside the whole tumour, over the {len(test_ids)} test phantoms. The gates are not inactive: the mean coefficient differs between tumour and the remaining brain at {int(np.sum(np.abs(np.nanmean(inside,0)-np.nanmean(outside,0))>0.02))} of {ng} levels by more than 0.02. Whether the gates increase the weight of the lesion (positive difference) or of the surrounding tissue is a property of the learned function; the table reports it without interpretation, and the same analysis is applied to the BraTS models in Section @sec:brats-res.

@tab ph-gates | widths=1.0,1.5,1.7,1.5 | align=cccc | size=9
Mean attention coefficient inside the whole tumour and in the remaining brain tissue (phantom test set).
| Gate | Inside tumour | Outside tumour (brain) | Difference |
| --- | --- | --- | --- |
{rows}
@end
""")
    else:
        txt.append("~Attention analysis requires a trained variant with gates.\n")
    # ---- MAF weights
    mf = [a for a in avail if a[0] in ("a3_maf", "a4_tversky", "a5_full")]
    if mf:
        cfg, tag, name, ds = mf[0]
        W_ = {"HGG": [], "LGG": []}
        for d in ds:
            m, c = _load_model(d)
            for cid in test_ids:
                img, seg, meta = load_case(cache, cid, mmap=False)
                x, y = _crop(img, seg, p)
                with torch.no_grad():
                    m(torch.from_numpy(x[None]).float())
                W_[meta["grade"]].append(m.fusion.last_weights[0].mean(1).numpy())
        mods = ["T1", "T1ce", "T2", "FLAIR"]
        h, l = np.mean(W_["HGG"], 0), np.mean(W_["LGG"], 0)
        sh = np.std(W_["HGG"], 0)
        rows = "\n".join(f"| {mods[i]} | {h[i]:.3f} | {l[i]:.3f} | {sh[i]:.3f} |" for i in range(4))
        fig, ax = plt.subplots(figsize=(3.6, 2.4))
        xx = np.arange(4)
        ax.bar(xx - 0.19, h, 0.36, color=C["blue"], label="HGG-like")
        ax.bar(xx + 0.19, l, 0.36, color=C["orange"], hatch="//", edgecolor="white", label="LGG-like")
        ax.axhline(0.25, color=C["grey"], ls=":", lw=1)
        ax.set_xticks(xx); ax.set_xticklabels(mods); ax.set_ylabel("mean fusion weight"); ax.legend(frameon=False)
        fig.savefig(FIGDIR / "ph_maf.png"); plt.close(fig)
        spread = float(np.abs(h - 0.25).max())
        if spread < 0.01 and float(sh.max()) < 0.01:
            maf_close = ("The fusion weights are uniform to within the precision of the table: the stem does not select among the modalities, neither on average nor between phantoms, and the weights of the two grades are indistinguishable. "
                         "The weights are computed from globally pooled descriptors of stems that end in an instance normalisation, which standardises every channel of every volume and thereby removes most of the volume-specific information that the pooled descriptor could carry; "
                         "this is a plausible reason for the uniform weights, which the present experiment does not isolate (it would require a variant without normalisation in the stems). "
                         "As a consequence the stem behaves as a fixed averaging of four modality-specific branches followed by a mixing convolution, and a benefit of this component cannot be attributed to modality selection. This design limitation is discussed in Chapter 5.")
        else:
            maf_close = ("The weights are computed from globally pooled descriptors and therefore vary only between volumes, not between positions, which limits the amount of modality selection that the stem can express; this design choice is revisited in Chapter 5.")
        txt.append(f"""@fig ph-maf | file={FIGDIR}/ph_maf.png | width=3.8
Mean weights of the modality-attentive fusion stem (variant {tag}, averaged over the stem channels and over the test phantoms of each grade). The dotted line is the uniform weight 1/4.
@end

@tab:ph-maf and @fig:ph-maf give the weights that the fusion stem assigns to the four modalities (variant {tag}). A uniform allocation corresponds to 1/4; the largest deviation from it, in the mean over the HGG-like test phantoms, is {spread:.3f}. The weights for HGG-like and LGG-like phantoms differ by at most {float(np.abs(h-l).max()):.3f}, and their standard deviation across phantoms is at most {float(sh.max()):.3f}.

@tab ph-maf | widths=1.2,1.6,1.6,1.9 | align=cccc | size=9
Fusion weights per modality (mean over stem channels and test phantoms).
| Modality | HGG-like | LGG-like | SD across HGG-like phantoms |
| --- | --- | --- | --- |
{rows}
@end

{maf_close}
""")
    return "\n".join(txt)


def _grade_section(base, runs):
    """Grade classification on phantoms via scripts/grade_classification.py (only if the 3 fold models of A5 exist)."""
    res = HERE.parent / "results"
    res.mkdir(exist_ok=True)
    f = res / "grade_phantom_a5.json"
    ok = all((runs / f"a5_full_s1_f{k}" / "test_metrics.csv").exists() for k in range(3))
    head = "@subsection[ph-grade] Grade classification pipeline\n\n"
    if not ok:
        return head + "~The grade-classification check requires the full model trained on all three folds; it was not available when this version was built.\n"
    if not f.exists():
        subprocess.run([sys.executable, str(HERE.parent / "scripts" / "grade_classification.py"), "--exp", "a5_full_s1", "--runs", str(runs),
                        "--cache", str(base / "cache"), "--splits", str(base / "splits.json"), "--out", str(f)], check=True,
                       stdout=subprocess.DEVNULL)
    J = json.loads(f.read_text())
    return head + _grade_text(J)


def _grade_text(J):
    rows = AN.grade_rows(J)
    best = next((r for r in rows if r[0].startswith("Predicted masks, logistic")), rows[0])
    gt = next((r for r in rows if r[0].startswith("Ground-truth masks, logistic")), None)
    t = AN.grade_table_md(J, "ph-grade-tab", "Grade classification on the phantoms (three folds, 80 development phantoms, 20 test phantoms). Predicted masks come from the out-of-fold models of the full model (seed 1).")
    s = (f"The out-of-fold protocol of Section @sec:grade was run on the phantoms with the full model (A5, seed 1) trained on each of the three folds. @tab:ph-grade-tab gives the result: with the features of the predicted masks and logistic regression, the test AUC is {best[2]} and the cross-validated AUC {best[1]}"
         + (f"; with the ground-truth masks the test AUC is {gt[2]}" if gt else "") + ". The phantom grades differ by construction in the enhancement of the shell, so a high AUC is expected and shows that the feature extraction, the leak-free fold assignment and the classifier code work end to end; with only 20 test phantoms, of which about five are LGG-like, the intervals are wide and the number carries no information about the achievable AUC on BraTS.\n\n")
    return s + t


def _final_err_sentence(avail, cfg0):
    ends = {}
    for cfg, tag, name, ds in avail:
        e = []
        for d in ds:
            L = AN.read_log(d).dropna(subset=["val_dice_mean"])
            e.append(1 - L["val_dice_mean"].iloc[-1])
        ends[tag] = float(np.mean(e))
    return "The final validation errors (mean over seeds) are " + ", ".join(f"{t}: {v:.4f}" for t, v in ends.items()) + "."


def _supp(avail):
    rows, drows = [], []
    for cfg, tag, name, ds in avail:
        for d in ds:
            m = AN.add_composite(pd.read_csv(d / "test_metrics.csv"))
            L = AN.read_log(d)
            v = L.dropna(subset=["val_dice_mean"])
            best_ep = int(v.loc[v["val_dice_mean"].idxmax(), "epoch"]) if len(v) else -1
            rows.append(f"| {tag} | {d.name.split('_s')[-1].split('_')[0]} | {m['dice_WT'].mean():.4f} | {m['dice_TC'].mean():.4f} | {m['dice_ET'].mean():.4f} | {m['dice_mean'].mean():.4f} | {m['hd95_mean'].mean():.3f} | {best_ep} | {L['epoch_seconds'].sum() / 60:.1f} |")
            A_ = L[L["stage"] == "adv"]
            if cfg == "a5_full" and len(A_):
                drows.append(f"| {d.name.split('_s')[-1].split('_')[0]} | {A_['d_acc_real'].mean():.3f} | {A_['d_acc_fake'].mean():.3f} | {A_['d_loss'].mean():.4f} | {A_['adv'].mean():.4f} |")
    out = f"""
@section[d-ph] Supplementary Tables of the Phantom Study

@tab ph-seed | widths=0.5,0.5,0.8,0.8,0.8,0.9,0.7,0.8,0.9 | align=ccccccccc | size=8
Test results of every phantom run (fold 0; 20 test phantoms). HD95 in voxels, mean over regions; best epoch: epoch of the selected checkpoint; time: total training time in minutes on two CPU cores.
| Variant | Seed | Dice WT | Dice TC | Dice ET | Dice mean | HD95 | Best epoch | Time (min) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}
@end
"""
    if drows:
        out += f"""
@tab ph-dacc | widths=0.8,1.6,1.6,1.4,1.7 | align=ccccc | size=8.5
Discriminator diagnostics of the full model per seed, averaged over the adversarial epochs.
| Seed | Accuracy on real maps | Accuracy on generated maps | $D$ loss | Adversarial loss of $G$ |
| --- | --- | --- | --- | --- |
{chr(10).join(drows)}
@end
"""
    return out


def complete_for_freeze(avail):
    """freeze the generated chapter text only when every variant and seed has finished"""
    return len(avail) == len(VARIANTS) and all(len(ds) == len(SEEDS) for *_, ds in avail)
