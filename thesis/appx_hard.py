"""Appendix E: supplementary results of the hard synthetic benchmark (all figures and tables computed from the run directories).

Nothing here is BraTS evidence.  Every number comes from runs/<variant>_s<seed>_f<fold>/{test,val}_metrics.csv, train_log.jsonl,
pred_test/*.npy, from the cached phantoms, or from the saved checkpoints (best.pt); the figures are drawn from the same sources.
"""
from __future__ import annotations

import itertools
import json
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

REG = AN.REG
VARS = [("a0_unet", "A0"), ("a1_res", "A1"), ("a2_att", "A2"), ("a3_maf", "A3"), ("a4_tversky", "A4"), ("a5_full", "A5")]
ALLCFG = VARS + [("t_dice_eq", "T0.5"), ("t_fn_heavy", "T0.8"), ("g_lam001", "G.01"), ("g_lam020", "G.2"), ("g_hinge", "Gh"), ("g_nosmooth", "Gn")]
LABCOL = {1: "#2a78d6", 2: "#1baf7a", 4: "#f2c230"}


def _tex(s):
    return s.replace("_", "\\_") if False else s


def build(H, base: Path, plt, C, figdir: Path) -> str:
    import torch
    from bratsgan.data import fold_ids, labels_to_regions, load_case
    from ch4_phantom import _load_model
    import ch4_gallery as G

    figdir.mkdir(exist_ok=True)
    splits = json.loads((base / "splits.json").read_text())
    _, _, test_ids = fold_ids(splits, 0)
    d0 = H.dirs("a5_full", 0, (1,))
    if not d0:
        return "@appendix[hard-supp] Supplementary Results of the Hard Synthetic Benchmark\n\n~The runs of the hard benchmark were not available.\n"
    cache = json.loads((d0[0] / "config.json").read_text())["cache_dir"]
    test = {c: H.test(c) for c, _ in VARS}
    test1 = {c: H.test(c, seeds=(1,)) for c, _ in ALLCFG}
    out = ["@appendix[hard-supp] Supplementary Results of the Hard Synthetic Benchmark\n",
           "This appendix collects the complete tables and figures behind Section @sec:hard. All values are computed from the run directories of the hard benchmark (160 synthetic phantoms; 32 test phantoms of fold 0; 128 development phantoms in three folds) by the same code that produced the chapter; none of them is a result on BraTS. Phantoms are identified by the last three digits of their case number.\n"]
    add = out.append
    ids = list(test["a5_full"].index)
    info = {}
    for cid in ids:
        img, seg, meta = load_case(cache, cid, mmap=False)
        info[cid] = {"grade": meta.get("grade", ""), "wt": int((np.asarray(seg) > 0).sum())}

    # ------------------------------------------------------------------ E.1 per-phantom Dice of all variants
    add("@section[x-pp] Per-Phantom Dice of the Variants\n")
    rows = []
    for cid in sorted(ids, key=lambda c: info[c]["wt"]):
        v = [f"{test[c].loc[cid, 'dice_mean']:.3f}" for c, _ in VARS]
        rows.append(f"| {cid[-3:]} | {info[cid]['grade']} | {info[cid]['wt']:,} | " + " | ".join(v) + " |")
    add(f"""@tab:x-pp lists the mean Dice score over the three regions of every variant on every one of the {len(ids)} test phantoms of fold 0, sorted by the volume of the whole tumour (average over the two seeds). The table is the raw material of the paired statistics of @tab:hd-steps; it also shows the dependence on lesion size that Section @sec:hd-dist analyses.

@tab x-pp | widths=0.6,0.6,1.0,0.7,0.7,0.7,0.7,0.7,0.7 | align=cccrcccccc | size=8
Mean Dice (WT, TC, ET) of the six variants on each test phantom, sorted by the whole-tumour volume (voxels).
| Phantom | Grade | WT volume | A0 | A1 | A2 | A3 | A4 | A5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}
@end
""")
    # per-region by-grade table
    rows = []
    for c, t in VARS:
        df = test[c].copy(); df["grade"] = [info[i]["grade"] for i in df.index]
        cells = []
        for g in ("HGG", "LGG"):
            s_ = df[df["grade"] == g]
            cells += [f"{s_[f'dice_{r}'].mean():.3f}" for r in REG]
        rows.append(f"| {t} | " + " | ".join(cells) + " |")
    ng = {g: sum(1 for i in ids if info[i]["grade"] == g) for g in ("HGG", "LGG")}
    add(f"""@tab:x-grade gives the Dice scores by grade. The LGG-like phantoms ({ng['LGG']} of {len(ids)}) have little or no enhancing tumour, so that their ET score is not comparable with that of the HGG-like phantoms ({ng['HGG']}); the empty-mask convention of Chapter 3 applies to them.

@tab x-grade | widths=0.8,0.75,0.75,0.75,0.75,0.75,0.75 | align=lcccccc | size=8.5
Mean Dice per region by grade of the phantom (test set, fold 0, two seeds).
| Variant | HGG WT | HGG TC | HGG ET | LGG WT | LGG TC | LGG ET |
| --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}
@end
""")

    # ------------------------------------------------------------------ E.2 out-of-fold table
    add("@section[x-oof] Out-of-Fold Scores of the Development Phantoms\n")
    o0, o5 = H.oof("a0_unet"), H.oof("a5_full")
    if o0 is not None and o5 is not None:
        j = o0[["fold", "dice_mean"]].join(o5[["dice_mean"]], rsuffix="_5")
        j["grade"] = [load_case(cache, i, mmap=False)[2].get("grade", "") for i in j.index]
        j = j.sort_index()
        half = (len(j) + 1) // 2
        left, right = j.iloc[:half], j.iloc[half:]
        rows = []
        for k in range(half):
            a = left.iloc[k]; ai = left.index[k]
            cell = f"| {ai[-3:]} | {int(a['fold'])} | {a['grade']} | {a['dice_mean']:.3f} | {a['dice_mean_5']:.3f} |"
            if k < len(right):
                b = right.iloc[k]; bi = right.index[k]
                cell += f" {bi[-3:]} | {int(b['fold'])} | {b['grade']} | {b['dice_mean']:.3f} | {b['dice_mean_5']:.3f} |"
            else:
                cell += " | | | | |"
            rows.append(cell)
        add(f"""Every development phantom is segmented by the model of the fold that did not train on it. @tab:x-oof gives the out-of-fold mean Dice of A0 and A5 for all {len(j)} phantoms (seed 1); the pooled difference of the two networks is the one reported in Section @sec:hd-cv.

@tab x-oof | widths=0.55,0.4,0.5,0.55,0.55,0.55,0.4,0.5,0.55,0.55 | align=cccccccccc | size=7.5
Out-of-fold mean Dice of A0 and A5 for the development phantoms (seed 1); fold of the phantom and grade.
| Phantom | Fold | Grade | A0 | A5 | Phantom | Fold | Grade | A0 | A5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}
@end
""")
        # figure: distribution of per-phantom differences by fold
        diff = j["dice_mean_5"] - j["dice_mean"]
        dm, dlo, dhi = AN.ci(diff.values)
        fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
        ax[0].hist(diff, bins=24, color=C["blue"], edgecolor="white")
        ax[0].axvline(0, color=C["ink"], lw=0.9); ax[0].axvline(diff.mean(), color=C["orange"], ls="--", lw=1.4, label=f"mean {diff.mean():+.4f}")
        ax[0].set_xlabel("A5 minus A0 (mean Dice)"); ax[0].set_ylabel("phantoms"); ax[0].legend(frameon=False, fontsize=8)
        ax[0].set_title("(a) all development phantoms", fontsize=9)
        data = [diff[j["fold"] == k].values for k in sorted(j["fold"].unique())]
        bp = ax[1].boxplot(data, patch_artist=True, medianprops=dict(color=C["ink"]))
        for b_ in bp["boxes"]:
            b_.set_facecolor(C["aqua"]); b_.set_alpha(0.6)
        ax[1].axhline(0, color=C["ink"], lw=0.9); ax[1].set_xticklabels([f"fold {int(k)}" for k in sorted(j["fold"].unique())])
        ax[1].set_ylabel("A5 minus A0"); ax[1].set_title("(b) by fold", fontsize=9)
        fig.tight_layout(); fig.savefig(figdir / "x_oof.png"); plt.close(fig)
        add(f"""@fig x-oof | file={figdir}/x_oof.png | width=6.2
Distribution of the per-phantom difference A5 minus A0 in out-of-fold mean Dice: (a) all {len(j)} development phantoms, (b) by fold.
@end

The distribution in @fig:x-oof(a) has a mean of {diff.mean():+.4f} and a standard deviation of {diff.std():.3f}, i.e. {diff.std() / max(abs(diff.mean()), 1e-9):.0f} times the mean, ({'the 95% bootstrap interval of the mean, ' + f'[{dlo:+.4f}, {dhi:+.4f}], ' + ('includes zero' if dlo <= 0 <= dhi else 'excludes zero')}); the boxes of panel (b) show that the three folds differ in the position of the median ({', '.join(f'{np.median(d):+.3f}' for d in data)}).
""")

    # ------------------------------------------------------------------ E.3 remaining gallery
    add("@section[x-gal] Remaining Test Phantoms\n")
    shown = set()
    for p_ in sorted(figdir.glob("gal_g*.png")):
        pass
    gal_used = set(G_used_ids(H, base, cache, ids, info, test))
    rest = [c for c in sorted(ids, key=lambda c: -test["a5_full"].loc[c, "dice_mean"]) if c not in gal_used]
    chunks = [rest[i:i + 4] for i in range(0, len(rest), 4)]
    labrows = []
    figtxt = []
    for gi, chunk in enumerate(chunks):
        fig, ax = plt.subplots(len(chunk), 5, figsize=(6.6, 1.4 * len(chunk) + 0.4))
        ax = np.atleast_2d(ax)
        for i, cid in enumerate(chunk):
            img, seg, meta = load_case(cache, cid, mmap=False)
            img, seg = np.asarray(img), np.asarray(seg)
            pr = G._labels_from_regions(np.load(d0[0] / "pred_test" / f"{cid}_pred.npy"))
            pos = np.argwhere(seg > 0); z = int(pos[:, 0].mean())
            ax[i, 0].imshow(img[1, z], cmap="gray"); ax[i, 1].imshow(img[3, z], cmap="gray")
            ax[i, 2].imshow(img[1, z], cmap="gray"); ax[i, 2].imshow(G._rgb(seg[z]))
            ax[i, 3].imshow(img[1, z], cmap="gray"); ax[i, 3].imshow(G._rgb(pr[z]))
            ax[i, 4].imshow(img[1, z], cmap="gray", alpha=0.6); ax[i, 4].imshow(G._err(seg[z], pr[z]))
            for a_ in ax[i]:
                a_.set_xticks([]); a_.set_yticks([]); a_.grid(False)
            ax[i, 0].set_ylabel(cid[-3:], fontsize=7)
            r = test["a5_full"].loc[cid]
            labrows.append(f"| {gi + 1} | {cid[-3:]} | {info[cid]['grade']} | {info[cid]['wt']:,} | {r['dice_WT']:.3f} | {r['dice_TC']:.3f} | {r['dice_ET']:.3f} |")
        for k, t in enumerate(("T1ce", "FLAIR", "Reference", "Detection (A5)", "Error")):
            ax[0, k].set_title(t, fontsize=8.5)
        fig.tight_layout(); fn = figdir / f"x_gal{gi + 1}.png"; fig.savefig(fn); plt.close(fig)
        figtxt.append(f"@fig x-gal{gi + 1} | file={fn} | width=6.4\nRemaining test phantoms, part {gi + 1} of {len(chunks)} (sorted by decreasing mean Dice of A5): T1ce, FLAIR, reference, detection of A5 and error map in the central axial slice through the tumour. Colours as in @fig:hd-g1.\n@end\n")
    add(f"""The gallery of Section @sec:hd-gal shows 14 of the {len(ids)} test phantoms. @fig:x-gal1 to @fig:x-gal{len(chunks)} show the remaining {len(rest)} in the order of decreasing mean Dice of A5, so that the complete test set can be inspected, and @tab:x-gal gives their volumes and scores.

{chr(10).join(figtxt)}
@tab x-gal | widths=0.5,0.7,0.6,1.0,0.8,0.8,0.8 | align=cccrccc | size=8.5
Phantoms of @fig:x-gal1 to @fig:x-gal{len(chunks)}: figure part, phantom, grade, whole-tumour volume (voxels) and Dice of A5 per region.
| Part | Phantom | Grade | WT volume | Dice WT | Dice TC | Dice ET |
| --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(labrows)}
@end
""")

    # ------------------------------------------------------------------ E.4 training curves of all runs
    add("@section[x-curves] Training Curves of All Runs\n")
    fig, ax = plt.subplots(3, 4, figsize=(6.8, 5.4), sharex=True)
    fig2, ax2 = plt.subplots(3, 4, figsize=(6.8, 5.4), sharex=True)
    nrun = 0
    for i, (c, t) in enumerate(ALLCFG):
        a_, b_ = ax.flat[i], ax2.flat[i]
        for s_, col in ((1, C["blue"]), (2, C["orange"])):
            L = AN.read_log(H.runs / f"{c}_s{s_}_f0")
            if L is None or L.empty:
                continue
            nrun += 1
            v = L.dropna(subset=["val_dice_mean"])
            a_.plot(v["epoch"], v["val_dice_mean"], color=col, marker="o", ms=2.5, lw=1.3, label=f"seed {s_}")
            b_.plot(L["epoch"], L["loss"], color=col, lw=1.3)
        a_.set_title(t, fontsize=8.5); b_.set_title(t, fontsize=8.5)
        if i == 0:
            a_.legend(frameon=False, fontsize=6.5)
    for a_ in list(ax[-1]) + list(ax2[-1]):
        a_.set_xlabel("epoch", fontsize=8)
    for a_ in list(ax[:, 0]):
        a_.set_ylabel("validation Dice", fontsize=8)
    for a_ in list(ax2[:, 0]):
        a_.set_ylabel("training loss", fontsize=8)
    fig.tight_layout(); fig.savefig(figdir / "x_curves_val.png"); plt.close(fig)
    fig2.tight_layout(); fig2.savefig(figdir / "x_curves_loss.png"); plt.close(fig2)
    add(f"""@fig:x-cv-val and @fig:x-cv-loss show the validation Dice and the training loss of every one of the {nrun} runs on fold 0 (twelve configurations, two seeds where available). The scale of the loss differs between configurations because the Tversky and adversarial terms are added to it; only the shape is comparable.

@fig x-cv-val | file={figdir}/x_curves_val.png | width=6.4
Validation Dice (mean over the three regions) against the epoch for all configurations of the hard benchmark (fold 0).
@end

@fig x-cv-loss | file={figdir}/x_curves_loss.png | width=6.4
Training loss of the generator against the epoch for all configurations of the hard benchmark (fold 0).
@end
""")

    # ------------------------------------------------------------------ E.5 secondary metrics
    add("@section[x-sec] Secondary Metrics by Variant\n")
    metrics = [("sens", "Sensitivity"), ("prec", "Precision"), ("hd95", "HD95 (voxels)")]
    for key, name in metrics:
        rows = []
        for c, t in VARS:
            df = test[c]
            ci_ = [AN.ci(df[f"{key}_{r}"].values) for r in REG]
            nd = 2 if key == "hd95" else 3
            rows.append(f"| {t} | " + " | ".join(f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]" for m, lo, hi in ci_) + " |")
        add(f"""@tab x-{key} | widths=0.7,1.9,1.9,1.9 | align=lccc | size=8
{name} of the six variants per region on the test phantoms (mean over phantoms and seeds, 95% bootstrap interval).
| Variant | WT | TC | ET |
| --- | --- | --- | --- |
{chr(10).join(rows)}
@end
""")
    # figure: sens-prec-hd95 per variant (bars)
    fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.5))
    xs = np.arange(len(VARS)); w = 0.26
    for k, (key, ttl) in enumerate((("sens", "sensitivity"), ("prec", "precision"), ("hd95", "HD95 (voxels)"))):
        for ri, r in enumerate(REG):
            ax[k].bar(xs + (ri - 1) * w, [test[c][f"{key}_{r}"].mean() for c, _ in VARS], w, color=[C["blue"], C["orange"], C["aqua"]][ri], label=r,
                      hatch=["", "//", ".."][ri], edgecolor="white")
        ax[k].set_xticks(xs); ax[k].set_xticklabels([t for _, t in VARS], fontsize=7.5); ax[k].set_title(ttl, fontsize=9)
    ax[0].legend(frameon=False, fontsize=6.5, loc="lower left", ncol=3, bbox_to_anchor=(0, -0.42)); fig.subplots_adjust(bottom=0.3)
    fig.tight_layout(); fig.savefig(figdir / "x_secondary.png", bbox_inches="tight"); plt.close(fig)
    et_sens = [test[c]["sens_ET"].mean() for c, _ in VARS]; et_prec = [test[c]["prec_ET"].mean() for c, _ in VARS]
    pen = {t: int((test[c]["hd95_ET"] > 100).sum()) for c, t in VARS}
    pen_txt = (f" The HD95 of the enhancing tumour is dominated by the empty-mask penalty of Chapter 3: {min(pen.values())} to {max(pen.values())} of the {len(ids)} test phantoms have an HD95 above 100 voxels for ET depending on the variant (reference or prediction empty while the other is not), and these phantoms alone determine the mean."
               if max(pen.values()) > 0 else "")
    add(f"""@fig:x-secondary summarises the three metrics. For the enhancing tumour the sensitivity lies between {min(et_sens):.3f} and {max(et_sens):.3f} and the precision between {min(et_prec):.3f} and {max(et_prec):.3f} over the variants: the networks recover most of the reference volume and pay for it with a low precision, which is the same over-segmentation as in Section @sec:hd-tv.{pen_txt}

@fig x-secondary | file={figdir}/x_secondary.png | width=6.4
Sensitivity, precision and HD95 of the six variants for WT, TC and ET (test phantoms of fold 0, mean over phantoms and seeds).
@end
""")

    # ------------------------------------------------------------------ E.6 all pairwise comparisons
    add("@section[x-pair] All Pairwise Comparisons\n")
    pairs = list(itertools.combinations(VARS, 2))
    res_rows, ps = [], []
    for (c1, t1), (c2, t2) in pairs:
        q = AN.paired(test[c2], test[c1], "dice_mean")
        res_rows.append((t2, t1, q)); ps.append(q["p"])
    holm = S.holm(np.array(ps)) if hasattr(S, "holm") else np.array(ps)
    rows = []
    M = np.zeros((len(VARS), len(VARS)))
    names = [t for _, t in VARS]
    for (t2, t1, q), ph in zip(res_rows, holm):
        rows.append(f"| {t2} $-$ {t1} | {AN.fmt_ci((q['diff'], q['lo'], q['hi']), 3)} | {AN.fmt_p(q['p'])} | {AN.fmt_p(float(ph))} | {q.get('dz', float('nan')):.2f} |")
        M[names.index(t2), names.index(t1)] = q["diff"]; M[names.index(t1), names.index(t2)] = -q["diff"]
    add(f"""@tab:x-pair gives the paired difference of the mean Dice for all {len(pairs)} pairs of the six variants, with the Holm correction applied over the {len(pairs)} comparisons (the chapter corrects over the five successive comparisons only). @fig:x-pair shows the differences as a matrix.

@tab x-pair | widths=1.2,2.0,0.9,0.9,0.7 | align=lcccc | size=8
Paired differences of the mean Dice between all pairs of variants (later minus earlier variant; 95% bootstrap interval; Wilcoxon p unadjusted and with Holm correction over the {len(pairs)} comparisons; Cohen's $d_z$).
| Comparison | $\\Delta$ Dice [95% CI] | p | p (Holm) | $d_z$ |
| --- | --- | --- | --- | --- |
{chr(10).join(rows)}
@end
""")
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    im = ax.imshow(M, cmap="RdBu", vmin=-abs(M).max(), vmax=abs(M).max())
    ax.set_xticks(range(len(VARS))); ax.set_yticks(range(len(VARS))); ax.set_xticklabels(names); ax.set_yticklabels(names); ax.grid(False)
    for i in range(len(VARS)):
        for j_ in range(len(VARS)):
            if i != j_:
                ax.text(j_, i, f"{M[i, j_]:+.3f}", ha="center", va="center", fontsize=6.5)
    ax.set_xlabel("reference variant"); ax.set_ylabel("variant")
    fig.colorbar(im, ax=ax, fraction=0.046, label="$\\Delta$ mean Dice"); fig.tight_layout(); fig.savefig(figdir / "x_pairmat.png"); plt.close(fig)
    add(f"""@fig x-pair | file={figdir}/x_pairmat.png | width=3.9
Matrix of the paired differences in mean Dice (row variant minus column variant) on the test phantoms.
@end
""")

    # ------------------------------------------------------------------ E.7 scatter and Bland-Altman
    add("@section[x-ba] Agreement of A0 and A5 per Phantom\n")
    fig, ax = plt.subplots(2, 3, figsize=(6.8, 4.4))
    t0, t5 = test["a0_unet"], test["a5_full"]
    stats_ba = []
    for k, r in enumerate(REG):
        a_, b_ = t0[f"dice_{r}"].values, t5[f"dice_{r}"].values
        ax[0, k].scatter(a_, b_, s=14, color=C["blue"]); lim = [min(a_.min(), b_.min()) - 0.02, 1.0]
        ax[0, k].plot(lim, lim, color=C["grey"], ls=":"); ax[0, k].set_xlabel("A0"); ax[0, k].set_title(r, fontsize=9)
        m_, d_ = (a_ + b_) / 2, b_ - a_
        ax[1, k].scatter(m_, d_, s=14, color=C["orange"]); ax[1, k].axhline(d_.mean(), color=C["ink"], lw=1)
        sd = d_.std(ddof=1)
        for off in (-1.96, 1.96):
            ax[1, k].axhline(d_.mean() + off * sd, color=C["grey"], ls="--", lw=0.9)
        ax[1, k].set_xlabel("mean of A0 and A5"); stats_ba.append((r, d_.mean(), d_.mean() - 1.96 * sd, d_.mean() + 1.96 * sd, float(np.corrcoef(a_, b_)[0, 1])))
    ax[0, 0].set_ylabel("A5"); ax[1, 0].set_ylabel("A5 minus A0")
    fig.tight_layout(); fig.savefig(figdir / "x_ba.png"); plt.close(fig)
    rows = "\n".join(f"| {r} | {m:+.4f} | [{lo:+.3f}, {hi:+.3f}] | {rho:.3f} |" for r, m, lo, hi, rho in stats_ba)
    add(f"""@fig:x-ba (upper row) plots the Dice of A5 against that of A0 for every test phantom and (lower row) the differences against the means, with the mean difference and the limits of agreement (mean $\\pm$ 1.96 standard deviations). @tab:x-ba lists the numbers. The half-width of the limits of agreement is {', '.join(f'{(hi - lo) / 2:.3f}' for _, _, lo, hi, _ in stats_ba)} for WT, TC and ET, against mean differences of {', '.join(f'{m:+.3f}' for _, m, _, _, _ in stats_ba)}; the correlation between the scores of the two networks is {', '.join(f'{rho:.2f}' for *_, rho in stats_ba)}. The two networks therefore rank the phantoms in nearly the same order, and the difference between them is small compared with the variation between phantoms.

@fig x-ba | file={figdir}/x_ba.png | width=6.4
Agreement of A0 and A5 on the test phantoms: scatter plots of the Dice (upper row) and Bland--Altman plots (lower row) for WT, TC and ET.
@end

@tab x-ba | widths=1.0,1.2,2.0,1.2 | align=cccc | size=9
Bland-Altman statistics of A5 against A0 (Dice, test phantoms): mean difference, limits of agreement and Pearson correlation.
| Region | Mean difference | Limits of agreement | Correlation |
| --- | --- | --- | --- |
{rows}
@end
""")

    # ------------------------------------------------------------------ E.8 voxel confusion
    add("@section[x-conf] Voxel Confusion Matrices\n")
    labs = [0, 1, 2, 4]
    names_l = ["background", "NCR (1)", "ED (2)", "ET (4)"]
    conf = {}
    for c, t in (("a0_unet", "A0"), ("a5_full", "A5")):
        dd = H.dirs(c, 0, (1,))[0]
        Cm = np.zeros((4, 4), np.int64)
        for cid in test_ids:
            _, seg, _ = load_case(cache, cid, mmap=False); seg = np.asarray(seg)
            pr = G._labels_from_regions(np.load(dd / "pred_test" / f"{cid}_pred.npy"))
            for gi, gl in enumerate(labs):
                m = seg == gl
                for pi, pl in enumerate(labs):
                    Cm[gi, pi] += int((m & (pr == pl)).sum())
        conf[t] = Cm
    for t, Cm in conf.items():
        rn = Cm / np.maximum(Cm.sum(1, keepdims=True), 1)
        rows = "\n".join(f"| {names_l[i]} | " + " | ".join(f"{100 * rn[i, j_]:.1f}" for j_ in range(4)) + f" | {int(Cm[i].sum()):,} |" for i in range(4))
        add(f"""@tab x-conf{t.lower()} | widths=1.4,0.9,0.9,0.9,0.9,1.1 | align=lccccr | size=8.5
Voxel confusion matrix of {t} on the test phantoms (rows: reference label, columns: predicted label, entries in per cent of the reference voxels; last column: number of reference voxels). The label map is obtained from the three regions with the hierarchy ET in TC in WT.
| Reference | background | NCR | ED | ET | Voxels |
| --- | --- | --- | --- | --- | --- |
{rows}
@end
""")
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.9))
    for a_, (t, Cm) in zip(ax, conf.items()):
        rn = Cm / np.maximum(Cm.sum(1, keepdims=True), 1)
        im = a_.imshow(rn, cmap="Blues", vmin=0, vmax=1); a_.grid(False)
        a_.set_xticks(range(4)); a_.set_yticks(range(4)); a_.set_xticklabels(["bg", "NCR", "ED", "ET"]); a_.set_yticklabels(["bg", "NCR", "ED", "ET"])
        for i in range(4):
            for j_ in range(4):
                a_.text(j_, i, f"{100 * rn[i, j_]:.0f}", ha="center", va="center", fontsize=7, color="white" if rn[i, j_] > 0.5 else "black")
        a_.set_title(t, fontsize=9); a_.set_xlabel("predicted")
    ax[0].set_ylabel("reference")
    fig.tight_layout(); fig.savefig(figdir / "x_conf.png"); plt.close(fig)
    r0, r5 = conf["A0"] / np.maximum(conf["A0"].sum(1, keepdims=True), 1), conf["A5"] / np.maximum(conf["A5"].sum(1, keepdims=True), 1)
    add(f"""@tab:x-confa0 and @tab:x-confa5 give the confusion between the four labels, and @fig:x-conf shows them as matrices. The enhancing tumour is recovered for {100 * r0[3, 3]:.1f}% (A0) and {100 * r5[3, 3]:.1f}% (A5) of its voxels; the remainder is assigned to the necrotic or non-enhancing core ({100 * r0[3, 1]:.1f}% and {100 * r5[3, 1]:.1f}%) or to the oedema, {'so that the errors are confusions within the tumour more than losses to the background' if (r5[3, 0] < r5[3, 1] + r5[3, 2] and r0[3, 0] < r0[3, 1] + r0[3, 2]) else 'which includes a notable loss to the background'}. {'A0 assigns the label of the necrotic and non-enhancing core to ' + f'{100 * r0[1, 1]:.1f}% of the reference voxels of that label and ' + f'{100 * r0[1, 3]:.1f}% of them to the enhancing tumour; A5 does so for {100 * r5[1, 1]:.1f}% and {100 * r5[1, 3]:.1f}%.'}

@fig x-conf | file={figdir}/x_conf.png | width=6.2
Confusion matrices between the four labels (rows normalised) for A0 and A5 on the test phantoms.
@end
""")

    # ------------------------------------------------------------------ probabilities of all variants
    probs = {}
    for c, t in VARS:
        dd = H.dirs(c, 0, (1,))
        if not dd:
            continue
        m, cfg = _load_model(dd[0])
        pp = {}
        for cid in test_ids:
            img, seg, meta = load_case(cache, cid, mmap=False)
            x = np.asarray(img)
            D_, H_, W_ = x.shape[1:]
            pad = [(0, 0)] + [(0, (-n) % 8) for n in (D_, H_, W_)]
            with torch.no_grad():
                pr = torch.sigmoid(m(torch.from_numpy(np.pad(x, pad)[None]).float()))[0].numpy()[:, :D_, :H_, :W_]
            pp[cid] = pr
        probs[c] = pp

    # ------------------------------------------------------------------ E.9 calibration of all variants
    add("@section[x-cal] Calibration of All Variants\n")
    nb = 10; edges = np.linspace(0, 1, nb + 1)
    ece, curves = {}, {}
    for c, t in VARS:
        for k, r in enumerate(REG):
            P_, T_ = [], []
            for cid in test_ids:
                img, seg, _ = load_case(cache, cid, mmap=False)
                x = np.asarray(img); y = labels_to_regions(np.asarray(seg)).astype(np.float32); brain = np.abs(x).sum(0) > 0
                P_.append(probs[c][cid][k][brain]); T_.append(y[k][brain])
            p, tt = np.concatenate(P_), np.concatenate(T_)
            idx = np.clip(np.digitize(p, edges) - 1, 0, nb - 1)
            e = 0.0; xs, ys = [], []
            for b in range(nb):
                s = idx == b
                if s.sum() == 0:
                    continue
                e += s.mean() * abs(tt[s].mean() - p[s].mean()); xs.append(p[s].mean()); ys.append(tt[s].mean())
            ece[(c, r)] = e; curves[(c, r)] = (xs, ys)
    fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.6), sharey=True)
    cols = [C["blue"], C["orange"], C["aqua"], C["grey"], C["violet"], C["red"]]
    for k, r in enumerate(REG):
        ax[k].plot([0, 1], [0, 1], color=C["grey"], ls=":", lw=1)
        for i, (c, t) in enumerate(VARS):
            ax[k].plot(*curves[(c, r)], color=cols[i], marker=["o", "s", "^", "D", "v", "x"][i], ms=3, lw=1.1, label=t)
        ax[k].set_title(r, fontsize=9); ax[k].set_xlabel("predicted probability")
    ax[0].set_ylabel("observed frequency"); ax[2].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout(); fig.savefig(figdir / "x_cal.png"); plt.close(fig)
    rows = "\n".join(f"| {t} | " + " | ".join(f"{ece[(c, r)]:.4f}" for r in REG) + " |" for c, t in VARS)
    add(f"""@fig:x-cal and @tab:x-cal extend the calibration analysis of Section @sec:hd-calib from A0 and A5 to all six variants (test phantoms of fold 0, seed 1, ten equal-width bins over the brain voxels).

@fig x-cal | file={figdir}/x_cal.png | width=6.4
Reliability diagrams of the six variants for WT, TC and ET (test phantoms, fold 0, seed 1).
@end

@tab x-cal | widths=1.0,1.1,1.1,1.1 | align=cccc | size=9
Expected calibration error of the six variants per region.
| Variant | WT | TC | ET |
| --- | --- | --- | --- |
{rows}
@end
""")

    # ------------------------------------------------------------------ E.10 ET probability maps
    add("@section[x-prob] Probability Maps of the Enhancing Tumour\n")
    pick = [c for c in sorted(ids, key=lambda c: -info[c]["wt"]) if info[c]["grade"] == "HGG"][2:6]
    show = [("a0_unet", "A0"), ("a3_maf", "A3"), ("a4_tversky", "A4"), ("a5_full", "A5")]
    fig, ax = plt.subplots(len(pick), 2 + len(show), figsize=(6.8, 1.35 * len(pick) + 0.5))
    for i, cid in enumerate(pick):
        img, seg, _ = load_case(cache, cid, mmap=False); img, seg = np.asarray(img), np.asarray(seg)
        z = int(np.argwhere(seg > 0)[:, 0].mean())
        ax[i, 0].imshow(img[1, z], cmap="gray"); ax[i, 1].imshow(img[1, z], cmap="gray"); ax[i, 1].contour(seg[z] == 4, levels=[0.5], colors=["white"], linewidths=0.9)
        for j_, (c, t) in enumerate(show):
            ax[i, 2 + j_].imshow(probs[c][cid][2][z], cmap="magma", vmin=0, vmax=1)
            ax[i, 2 + j_].contour(seg[z] == 4, levels=[0.5], colors=["white"], linewidths=0.7)
            if i == 0:
                ax[0, 2 + j_].set_title(t, fontsize=8.5)
        for a_ in ax[i]:
            a_.set_xticks([]); a_.set_yticks([]); a_.grid(False)
        ax[i, 0].set_ylabel(cid[-3:], fontsize=7)
    ax[0, 0].set_title("T1ce", fontsize=8.5); ax[0, 1].set_title("ET contour", fontsize=8.5)
    fig.tight_layout(); fig.savefig(figdir / "x_prob.png"); plt.close(fig)
    ratio = {t: np.mean([(probs[c][cid][2] > 0.5).sum() / max((np.asarray(load_case(cache, cid, mmap=False)[1]) == 4).sum(), 1) for cid in pick]) for c, t in show}
    add(f"""@fig:x-prob shows the predicted probability of the enhancing tumour of four networks on four HGG-like phantoms with large lesions (central axial slice; the white contour is the reference ET). On these phantoms the ratio of the number of voxels with a probability above 0.5 to the reference volume is {', '.join(f'{ratio[t]:.2f} ({t})' for _, t in show)}. A probability map that fills the whole lesion instead of the shell is the visible form of the over-segmentation analysed in Section @sec:hd-tv.

@fig x-prob | file={figdir}/x_prob.png | width=6.4
Probability maps of the enhancing tumour of A0, A3, A4 and A5 on four HGG-like test phantoms; white contour: reference enhancing tumour.
@end
""")

    # ------------------------------------------------------------------ E.11 slice profile
    add("@section[x-slice] Dice along the Slice Axis\n")
    fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.5), sharey=True)
    prof = {}
    for c, t, col in (("a0_unet", "A0", C["blue"]), ("a5_full", "A5", C["orange"])):
        dd = H.dirs(c, 0, (1,))[0]
        acc = {r: [] for r in REG}
        for cid in test_ids:
            _, seg, _ = load_case(cache, cid, mmap=False); gt = labels_to_regions(np.asarray(seg))
            pr = np.load(dd / "pred_test" / f"{cid}_pred.npy")
            zs = np.argwhere(gt[0] > 0)[:, 0]; z0, z1 = zs.min(), zs.max()
            for k, r in enumerate(REG):
                v = []
                for z in np.linspace(z0, z1, 12).round().astype(int):
                    a, b = gt[k][z] > 0, pr[k][z] > 0
                    s = a.sum() + b.sum()
                    v.append(np.nan if (a.sum() == 0) else 2 * (a & b).sum() / s)
                acc[r].append(v)
        prof[t] = {r: np.nanmean(np.array(acc[r], float), 0) for r in REG}
        for k, r in enumerate(REG):
            ax[k].plot(np.linspace(0, 1, 12), prof[t][r], color=col, marker="o", ms=3, label=t)
    for k, r in enumerate(REG):
        ax[k].set_title(r, fontsize=9); ax[k].set_xlabel("relative slice position", fontsize=8)
    ax[0].set_ylabel("slice-wise Dice"); ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(figdir / "x_slice.png"); plt.close(fig)
    mid = {t: {r: float(prof[t][r][4:8].mean()) for r in REG} for t in prof}
    edge = {t: {r: float(np.r_[prof[t][r][:2], prof[t][r][-2:]].mean()) for r in REG} for t in prof}
    rows = "\n".join(f"| {t} | " + " | ".join(f"{mid[t][r]:.3f} / {edge[t][r]:.3f}" for r in REG) + " |" for t in prof)
    slice_txt = (f"For WT and TC the accuracy is highest in the central slices (A5: {mid['A5']['WT']:.3f} and {mid['A5']['TC']:.3f}) and falls towards the poles of the lesion ({edge['A5']['WT']:.3f} and {edge['A5']['TC']:.3f}), where partial volume and the blur of the labels dominate. "
                 f"For ET the profile does not follow this pattern: the mean over the central positions ({mid['A5']['ET']:.3f} for A5, {mid['A0']['ET']:.3f} for A0) is {'lower' if mid['A5']['ET'] < edge['A5']['ET'] else 'higher'} than that over the outer positions ({edge['A5']['ET']:.3f} and {edge['A0']['ET']:.3f}). "
                 "A plausible reason, which the data do not test, is the geometry of the hard phantoms: the enhancing tumour is a thin shell, so that a central slice cuts it as a ring whose thickness is a few voxels and a boundary error of one voxel removes a large fraction of it, whereas the slices near the poles cut it tangentially as a compact disc. "
                 "The ranking of the two networks along the axis changes between positions, in agreement with the small and unstable difference of the means.")
    add(f"""@fig:x-slice plots the Dice score of single axial slices, averaged over the test phantoms, against the relative position of the slice between the first and the last slice that contains the tumour; slices without reference voxels of the region are excluded. @tab:x-slice gives the mean over the four central positions and over the four outer positions.

@fig x-slice | file={figdir}/x_slice.png | width=6.4
Slice-wise Dice of A0 and A5 against the relative slice position within the tumour extent (mean over the test phantoms).
@end

@tab x-slice | widths=1.0,1.5,1.5,1.5 | align=cccc | size=9
Slice-wise Dice in the central part / at the ends of the tumour extent (mean over test phantoms).
| Variant | WT | TC | ET |
| --- | --- | --- | --- |
{rows}
@end

{slice_txt}
""")

    # ------------------------------------------------------------------ E.12 grading diagnostics
    add("@section[x-grade2] Diagnostics of the Grade Classification\n")
    res = HERE.parent / "results"
    from sklearn.metrics import roc_curve, auc as skauc
    combos = [("pred_logreg", "predicted masks, logistic", C["blue"], "-"), ("pred_gboost", "predicted masks, boosting", C["orange"], "--"),
              ("gt_logreg", "reference masks, logistic", C["aqua"], "-."), ("gt_gboost", "reference masks, boosting", C["grey"], ":")]
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.9))
    nloaded = 0
    for key, lab, col, ls in combos:
        for kk, split in enumerate(("cv", "test")):
            f = res / f"grade_hard_a5.{key}.{split}.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f); nloaded += 1
            fpr, tpr, _ = roc_curve(df["y"], df["prob"])
            ax[kk].plot(fpr, tpr, color=col, ls=ls, label=f"{lab} (AUC {skauc(fpr, tpr):.2f})")
    for a_, ttl in zip(ax, ("(a) cross-validated (development phantoms)", "(b) test phantoms")):
        a_.plot([0, 1], [0, 1], color=C["ink"], ls=":", lw=0.8); a_.set_xlabel("false positive rate"); a_.set_title(ttl, fontsize=8.5)
    ax[0].set_ylabel("true positive rate"); ax[0].legend(frameon=False, fontsize=6, loc="lower right"); ax[1].legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout(); fig.savefig(figdir / "x_roc.png"); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.5))
    for a_, split, ttl in zip(ax, ("cv", "test"), ("(a) cross-validated", "(b) test")):
        f = res / f"grade_hard_a5.pred_logreg.{split}.csv"
        if f.exists():
            df = pd.read_csv(f)
            a_.hist([df[df["y"] == 0]["prob"], df[df["y"] == 1]["prob"]], bins=10, range=(0, 1), color=[C["orange"], C["blue"]], label=["LGG-like", "HGG-like"], edgecolor="white")
        a_.set_xlabel("predicted probability of HGG"); a_.set_title(ttl, fontsize=9)
    ax[0].set_ylabel("phantoms"); ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(figdir / "x_gprob.png"); plt.close(fig)
    t_df = pd.read_csv(res / "grade_hard_a5.pred_logreg.test.csv"); c_df = pd.read_csv(res / "grade_hard_a5.pred_logreg.cv.csv")
    def cm(df):
        pred = (df["prob"] >= 0.5).astype(int); y = df["y"].astype(int)
        return int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 0)).sum())
    a_, b_ = cm(c_df), cm(t_df)
    a_t = int((t_df['y'] == 1).sum())
    add(f"""@fig:x-roc shows the ROC curves of the four combinations of mask source and classifier of @tab:hd-grade-tab, and @fig:x-gprob the distribution of the predicted probability of the high grade for the logistic model on predicted masks. @tab:x-gcm gives the confusion counts at the threshold 0.5.

@fig x-roc | file={figdir}/x_roc.png | width=6.4
ROC curves of the grade classifiers on the hard benchmark: (a) cross-validated predictions on the development phantoms, (b) predictions on the test phantoms.
@end

@fig x-gprob | file={figdir}/x_gprob.png | width=6.2
Predicted probability of the high grade (logistic regression, features of the predicted masks) for HGG-like and LGG-like phantoms: (a) cross-validated, (b) test.
@end

@tab x-gcm | widths=1.8,0.9,0.9,0.9,0.9 | align=lcccc | size=9
Confusion counts of the logistic model on features of predicted masks at the threshold 0.5 (HGG-like is the positive class).
| Set | True positive | False negative | False positive | True negative |
| --- | --- | --- | --- | --- |
| Cross-validated (development) | {a_[0]} | {a_[1]} | {a_[2]} | {a_[3]} |
| Test | {b_[0]} | {b_[1]} | {b_[2]} | {b_[3]} |
@end

The curves of the models on the reference masks rise to the corner of the plot in both panels, whereas those on the predicted masks stay close to the diagonal in the cross-validated panel and rise above it in the test panel, which is the discrepancy discussed in Section @sec:hd-grade. The histograms show where it comes from: in the cross-validated panel a large group of phantoms of both grades receives a probability close to zero or close to one irrespective of its true grade, and in the test panel the {a_t} HGG-like phantoms are spread over a wide range of probabilities while the few LGG-like phantoms lie at its lower end. The confusion counts of @tab:x-gcm rest on {len(t_df)} test phantoms, of which {int((t_df['y'] == 0).sum())} are LGG-like, so that a change of the classification of a single phantom moves the sensitivity or specificity by several per cent.
""")

    return "\n".join(out)


def G_used_ids(H, base, cache, ids, info, test):
    """Phantoms that already appear in the gallery of Section hd-gal (recorded by ch4_gallery.build)."""
    import ch4_gallery
    return list(ch4_gallery.USED)
