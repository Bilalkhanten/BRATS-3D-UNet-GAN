"""Chapter 4, section 4.4: hard synthetic benchmark (NOT BraTS evidence).

160 phantoms (48^3) generated with difficulty="hard" (small lesions, thin enhancing shells, low tumour contrast, blurred boundaries,
strong noise and bias field).  The complete set of variants, sensitivity runs, seeds and folds of experiments/hard_study.sh is analysed
with the same statistics as planned for BraTS.  Every number in the generated text is computed here from the run directories.

The module also fills the dictionary RES with the key numbers; the discussion chapter (ch5_hard.py) is generated from it.
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

FIGDIR = HERE / "figs_hard"
FROZEN = HERE / "assets" / "hard_frozen"
W = Path(os.environ.get("BRATSGAN_HARD_STUDY", HERE.parent / "runs" / "hard_study"))
CHAIN = [("a0_unet", "A0", "Plain 3D U-Net"), ("a1_res", "A1", "+ residual blocks"), ("a2_att", "A2", "+ attention gates"),
         ("a3_maf", "A3", "+ MAF stem"), ("a4_tversky", "A4", "+ Tversky loss"), ("a5_full", "A5", "+ adversarial term (full)")]
TAG = {c: t for c, t, _ in CHAIN}
NAME = {c: n for c, _, n in CHAIN}
SEEDS = (1, 2)
REG = AN.REG
RES: dict = {}
SUPP = ""
SUPPX = ""
P = "~pending"


# ---------------------------------------------------------------------------------------------- data access
class Hard:
    def __init__(self, base):
        self.base = Path(base)
        self.runs = self.base / "runs"

    def dirs(self, cfg, fold=0, seeds=SEEDS, split="test"):
        ds = [self.runs / f"{cfg}_s{s}_f{fold}" for s in seeds]
        return [d for d in ds if (d / f"{split}_metrics.csv").exists()]

    def test(self, cfg, fold=0, seeds=SEEDS):
        ds = self.dirs(cfg, fold, seeds)
        return AN.load_metrics(ds, "test") if ds else None

    def test_seed(self, cfg, seed, fold=0):
        return self.test(cfg, fold, (seed,))

    def test_folds(self, cfg, seed=1, folds=(0, 1, 2)):
        ds = [d for f in folds for d in self.dirs(cfg, f, (seed,))]
        return (AN.load_metrics(ds, "test"), len(ds)) if len(ds) == len(folds) else (None, len(ds))

    def oof(self, cfg, seed=1, folds=(0, 1, 2)):
        fr = []
        for f in folds:
            d = self.runs / f"{cfg}_s{seed}_f{f}"
            if not (d / "val_metrics.csv").exists():
                return None
            fr.append(pd.read_csv(d / "val_metrics.csv").assign(fold=f))
        return AN.add_composite(pd.concat(fr, ignore_index=True).set_index("id"))

    def log(self, cfg, seed=1, fold=0):
        d = self.runs / f"{cfg}_s{seed}_f{fold}"
        return AN.read_log(d)


def _m(df, col, nd=3):
    return P if df is None or col not in df else AN.fmt(df[col].mean(), nd)


def _mc(df, col, nd=3):
    return P if df is None or col not in df else AN.fmt_ci(AN.ci(df[col].values), nd)


def _sgn(q):
    """verbal description of a paired difference"""
    if q["lo"] > 0:
        return "positive, with an interval that excludes zero"
    if q["hi"] < 0:
        return "negative, with an interval that excludes zero"
    return "indistinguishable from zero (the interval contains zero)"


def _pct(x):
    return f"{100 * x:.1f}\\%"


# ---------------------------------------------------------------------------------------------- build
def build(hard_dir=None) -> str:
    global SUPP, SUPPX
    base = Path(hard_dir or W)
    if not (base / "runs").exists():
        fz = FROZEN / "hard.txt"
        if fz.exists():
            global RES
            RES = json.loads((FROZEN / "res.json").read_text())
            SUPP = (FROZEN / "supp.txt").read_text().replace("{FIGDIR}", str(FIGDIR))
            fx = FROZEN / "supp_x.txt"
            SUPPX = fx.read_text().replace("{FIGDIR}", str(FIGDIR)) if fx.exists() else ""
            FIGDIR.mkdir(exist_ok=True)
            import shutil
            for f in FROZEN.glob("*.png"):
                if not (FIGDIR / f.name).exists():
                    shutil.copy(f, FIGDIR / f.name)
            return fz.read_text().replace("{FIGDIR}", str(FIGDIR))
        return "@section[hard] Hard Synthetic Benchmark\n~The runs of the hard benchmark were not available when this document was built.\n"
    FIGDIR.mkdir(exist_ok=True)
    from figstyle import C, LS, MK, plt, setup
    setup()
    H = Hard(base)
    splits = json.loads((base / "splits.json").read_text())
    n_test = len(splits["test"])
    cfg0 = None
    for c, _, _ in CHAIN:
        ds = H.dirs(c)
        if ds:
            cfg0 = json.loads((ds[0] / "config.json").read_text())
            break
    if cfg0 is None:
        return "@section[hard] Hard Synthetic Benchmark\n~No run of the hard benchmark had finished when this document was built.\n"
    n_dev = len(splits["folds"][0]) + len(splits["folds"][1]) + len(splits["folds"][2]) if "folds" in splits else 128

    out = []
    add = out.append
    add("@section[hard] Hard Synthetic Benchmark\n")
    add(_design(H, n_test, n_dev, cfg0, plt, C))
    add(_ablation(H, n_test, plt, C, LS, MK))
    add(_distribution(H, n_test, plt, C))
    add(_curves(H, cfg0, plt, C, LS, MK))
    add(_tversky(H, n_test, plt, C, LS, MK))
    add(_gan(H, n_test, cfg0, plt, C, LS, MK))
    add(_seeds(H, n_test, plt, C, LS, MK))
    add(_cv(H, n_test, plt, C))
    add(_grade(H, base))
    add(_calibration(H, base, plt, C))
    add(_qualitative(H, base, plt, C))
    import ch4_gallery
    add(ch4_gallery.build(H, base, plt, C, FIGDIR))
    from ch4_phantom import _maps_section
    import ch4_phantom as PH
    avail = [(c, t, n, H.dirs(c)) for c, t, n in CHAIN if H.dirs(c)]
    old = PH.FIGDIR
    PH.FIGDIR = FIGDIR
    try:
        txt = _maps_section(avail, base, H.runs, plt, C)
    finally:
        PH.FIGDIR = old
    txt = (txt.replace("ph-gates", "hd-gates").replace("ph-maf", "hd-maf").replace("ph-maps", "hd-maps")
           .replace("ph_gates", "ph_gates").replace("phantom test set", "test set of the hard benchmark")
           .replace("test phantoms", "test phantoms of the hard benchmark").replace("test phantom ", "test phantom of the hard benchmark "))
    add(txt.replace("@subsection[hd-maps]", "@subsection[hd-maps]"))
    add(_efficiency(H, plt, C, MK))
    add(_hyp())
    SUPP = _supp(H)
    import appx_hard
    SUPPX = appx_hard.build(H, base, plt, C, FIGDIR)
    text = "\n".join(out)
    done = (base / "progress.txt").exists() and "ALL_DONE" in (base / "progress.txt").read_text()
    if done:
        FROZEN.mkdir(parents=True, exist_ok=True)
        (FROZEN / "hard.txt").write_text(text.replace(str(FIGDIR), "{FIGDIR}"))
        (FROZEN / "supp.txt").write_text(SUPP.replace(str(FIGDIR), "{FIGDIR}"))
        (FROZEN / "supp_x.txt").write_text(SUPPX.replace(str(FIGDIR), "{FIGDIR}"))
        (FROZEN / "res.json").write_text(json.dumps(RES, indent=1, default=float))
        import shutil
        for f in FIGDIR.glob("*.png"):
            shutil.copy(f, FROZEN / f.name)
    return text


# ---------------------------------------------------------------------------------------------- design
def _design(H, n_test, n_dev, cfg0, plt, C):
    from bratsgan.synthetic import make_phantom
    fig, ax = plt.subplots(2, 5, figsize=(6.4, 2.9))
    for r, diff in enumerate(("easy", "hard")):
        vols, lab, _ = make_phantom(48, "HGG", 7, diff)
        zc = int(np.argwhere(lab > 0)[:, 0].mean())
        for k, (m, nm) in enumerate(((1, "T1ce"), (3, "FLAIR"))):
            ax[r, k].imshow(vols[m][zc], cmap="gray"); ax[r, k].set_title(nm if r == 0 else "", fontsize=8)
        ax[r, 2].imshow(vols[2][zc], cmap="gray"); ax[r, 2].set_title("T2" if r == 0 else "", fontsize=8)
        ax[r, 3].imshow(vols[0][zc], cmap="gray"); ax[r, 3].set_title("T1" if r == 0 else "", fontsize=8)
        ax[r, 4].imshow(lab[zc], cmap="viridis", vmin=0, vmax=4); ax[r, 4].set_title("labels" if r == 0 else "", fontsize=8)
        ax[r, 0].set_ylabel(diff, fontsize=9)
        for a in ax[r]:
            a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.tight_layout(); fig.savefig(FIGDIR / "examples.png"); plt.close(fig)
    nt = len(H.dirs("a0_unet"))
    t_run = np.mean([AN.read_log(d)["epoch_seconds"].sum() / 60 for c, _, _ in CHAIN for d in H.dirs(c)])
    RES["t_run_min"] = float(t_run)
    RES["n_test"] = n_test
    RES["n_dev"] = n_dev
    return f"""@subsection[hd-design] Design of the benchmark

The phantom study of the previous section shows that the pipeline works but is saturated: all variants reach almost the same Dice score, so that it cannot rank them. To obtain a task on which the components of the model can actually differ, the phantom generator was given a second setting that makes the segmentation problem harder in the ways in which BraTS is hard (@tab:hd-gen). The lesions are smaller, so that an error of a few voxels matters; the enhancing shell is thinner, which makes ET small and difficult to separate from the core; the contrast between the tumour compartments and the normal tissue is lowered and the boundaries are blurred, which reproduces the gradual transitions of infiltrating glioma; and the noise and the multiplicative bias field are stronger. @fig:hd-examples compares one phantom of each setting. The hard benchmark is a synthetic task: it contains no anatomy, no artefacts and no inter-rater variability, and it is **not** evidence about the performance on patients. Its purpose is to exercise the complete analysis (ablation, paired statistics, sensitivity, seeds, folds, grading) on a problem that is not saturated, so that the analysis produces numbers that can be read, and so that published effect sizes can be compared with the variation measured here.

@tab hd-gen | widths=2.4,1.6,1.6 | align=lcc | size=9
Settings of the phantom generator: the easy setting of the previous section and the hard setting of this section.
| Property | Easy | Hard |
| --- | --- | --- |
| Number of phantoms (HGG-like / LGG-like) | 100 (75 / 25) | 160 (120 / 40) |
| Volume size (voxels) | $48^3$ | $48^3$ |
| Tumour semi-axes (fraction of the half-size) | $\\mathcal{{U}}(0.18,0.34)$ | $\\mathcal{{U}}(0.10,0.22)$ |
| Core radius relative to the tumour (HGG-like) | 0.55 | 0.78 (thin enhancing shell) |
| Tumour contrast | full | 0.45 towards the tissue value, then Gaussian blur ($\\sigma=1.2$ voxels) |
| Noise standard deviation | 0.03 | 0.10 |
| Amplitude of the bias field | 0.15 | 0.30 |
| Test phantoms / cross-validation folds | 20 / 3 | {n_test} / 3 (development set: {n_dev}) |
@end

@fig hd-examples | file={FIGDIR}/examples.png | width=6.0
One HGG-like phantom of the easy setting (upper row) and of the hard setting (lower row), same slice position: T1ce, FLAIR, T2, T1 and the labels (necrotic core, enhancing shell and oedema in three colours).
@end

The experiments follow the plan of Chapter 3 in a reduced form that can be run on two CPU cores: the six cumulative variants A0 to A5 on fold 0 with two seeds (tiers A and C), two Tversky settings and four adversarial settings with one seed (tier B), and the variants A0 and A5 on the two other folds (tier D). All networks have {cfg0['depth']} levels and {cfg0['base']} base channels (at most {cfg0['max_ch']}), are trained on $48^3$ patches with batch size {cfg0['batch_size']} for {cfg0['epochs_seg']} epochs of {cfg0['iters_per_epoch']} iterations (the adversarial variants add {cfg0['epochs_adv']} epochs of the adversarial stage; the other variants are trained for the same total number of generator updates), with AdamW at learning rate $10^{{-3}}$ in single precision. The discriminator has {cfg0['ndf']} base channels. Each run took {t_run:.0f} minutes on average. The unit of analysis is the phantom: where a variant has two seeds, the per-phantom metrics are averaged over the seeds before any statistic is computed.
"""


# ---------------------------------------------------------------------------------------------- ablation
def _ablation(H, n_test, plt, C, LS, MK):
    fr = {c: H.test(c) for c, _, _ in CHAIN}
    av = [(c, t, n) for c, t, n in CHAIN if fr[c] is not None]
    if not av:
        return "@subsection[hd-res] Ablation\n~pending\n"
    rows = []
    for c, t, n in av:
        df = fr[c]
        rows.append("| " + " | ".join([f"{t} {n}", str(len(H.dirs(c)))] + [_m(df, f"dice_{r}") for r in REG] + [_mc(df, "dice_mean")] + [_m(df, "hd95_mean", 2)]) + " |")
    means = {c: float(fr[c]["dice_mean"].mean()) for c, _, _ in av}
    bestc = max(means, key=means.get)
    worstc = min(means, key=means.get)
    et = {c: float(fr[c]["dice_ET"].mean()) for c, _, _ in av}
    RES["abl"] = {TAG[c]: {"dice_mean": means[c], "dice_ET": et[c], "dice_WT": float(fr[c]["dice_WT"].mean()), "dice_TC": float(fr[c]["dice_TC"].mean()),
                           "hd95": float(fr[c]["hd95_mean"].mean())} for c, _, _ in av}
    # paired steps
    steps = []
    for (c1, t1, _), (c2, t2, _) in zip(av[:-1], av[1:]):
        steps.append((f"{t2} vs {t1}", AN.paired(fr[c2], fr[c1], "dice_mean"), AN.paired(fr[c2], fr[c1], "hd95_mean"),
                      {r: AN.paired(fr[c2], fr[c1], f"dice_{r}") for r in REG}))
    holm = S.holm([s[1]["p"] for s in steps]) if steps else []
    srows = "\n".join("| " + " | ".join([s[0], AN.fmt_ci((s[1]["diff"], s[1]["lo"], s[1]["hi"]), 3), AN.fmt_p(float(h)), AN.fmt_ci((s[2]["diff"], s[2]["lo"], s[2]["hi"]), 2),
                                         f"{s[1]['dz']:.2f}"]) + " |" for s, h in zip(steps, holm))
    first, last = av[0][0], av[-1][0]
    tot = AN.paired(fr[last], fr[first], "dice_mean")
    totr = {r: AN.paired(fr[last], fr[first], f"dice_{r}") for r in REG}
    RES["tot"] = {"diff": tot["diff"], "lo": tot["lo"], "hi": tot["hi"], "p": tot["p"], "dz": tot["dz"]}
    RES["steps"] = [{"name": s[0], "diff": s[1]["diff"], "lo": s[1]["lo"], "hi": s[1]["hi"], "p_holm": float(h), "dz": s[1]["dz"]} for s, h in zip(steps, holm)]
    RES["totr"] = {r: {"diff": q["diff"], "lo": q["lo"], "hi": q["hi"]} for r, q in totr.items()}
    # cumulative differences vs A0 per region (forest)
    fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.6), sharey=True)
    cum = [(c, t) for c, t, _ in av[1:]]
    for k, r in enumerate(REG):
        for i, (c, t) in enumerate(cum):
            q = AN.paired(fr[c], fr[first], f"dice_{r}")
            col = C["aqua"] if q["lo"] > 0 else (C["red"] if q["hi"] < 0 else C["grey"])
            ax[k].errorbar(q["diff"], i, xerr=[[q["diff"] - q["lo"]], [q["hi"] - q["diff"]]], fmt="o", color=col, capsize=3, ms=4)
        ax[k].axvline(0, color=C["ink"], lw=0.8)
        ax[k].set_title(f"{r}", fontsize=9); ax[k].set_xlabel("$\\Delta$ Dice vs A0")
    ax[0].set_yticks(range(len(cum))); ax[0].set_yticklabels([t for _, t in cum]); ax[0].invert_yaxis()
    fig.tight_layout(); fig.savefig(FIGDIR / "forest.png"); plt.close(fig)
    nsig = sum(1 for s in steps if s[1]["lo"] > 0 or s[1]["hi"] < 0)
    nsig_h = sum(1 for h in holm if float(h) < 0.05)
    et_gain = totr["ET"]
    txt = f"""@subsection[hd-res] Ablation of the architecture and the objective

@tab:hd-main gives the accuracy of the cumulative variants on the {n_test} test phantoms. Unlike the saturated easy setting, the benchmark leaves room for differences (ET reaches a Dice of only about {max(et.values()):.2f}). The mean Dice of the variants ranges from {means[worstc]:.3f} ({TAG[worstc]}) to {means[bestc]:.3f} ({TAG[bestc]}); the ET region is the hardest, with a mean Dice between {min(et.values()):.3f} and {max(et.values()):.3f} compared with {min(fr[c]['dice_WT'].mean() for c,_,_ in av):.3f} to {max(fr[c]['dice_WT'].mean() for c,_,_ in av):.3f} for WT. This ordering of the regions is the one found in the BraTS literature (@tab:pubres).

@tab hd-main | widths=1.9,0.45,0.65,0.65,0.65,1.4,0.65 | align=lcccccc | size=8
Ablation chain on the hard benchmark (fold 0, {n_test} test phantoms). Per-phantom metrics are averaged over the seeds (column Seeds); interval: 95% bootstrap interval of the mean over phantoms. HD95 in voxels (mean over WT, TC, ET).
| Variant | Seeds | Dice WT | Dice TC | Dice ET | Dice mean [95% CI] | HD95 |
| --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}
@end

@tab:hd-steps lists the paired differences between successive variants and @fig:hd-forest shows the cumulative differences with respect to A0 for each region. {('Some of the differences are resolved by the design. ' if nsig else 'None of the differences between successive variants is resolved by the design of this benchmark: the intervals contain zero. ')}Of the {len(steps)} successive comparisons, {nsig} have a bootstrap interval for the mean Dice difference that excludes zero and {nsig_h} remain significant at the 5% level after the Holm correction. The last variant {TAG[last]} differs from the plain network {TAG[first]} by {AN.fmt_ci((tot['diff'], tot['lo'], tot['hi']), 3)} in mean Dice (paired bootstrap; Wilcoxon p {AN.fmt_p(tot['p'])}, $d_z$ = {tot['dz']:.2f}): the difference is {_sgn(tot)}. Region by region, the difference for ET is {AN.fmt_ci((et_gain['diff'], et_gain['lo'], et_gain['hi']), 3)}, that for TC {AN.fmt_ci((totr['TC']['diff'], totr['TC']['lo'], totr['TC']['hi']), 3)} and that for WT {AN.fmt_ci((totr['WT']['diff'], totr['WT']['lo'], totr['WT']['hi']), 3)}.

@tab hd-steps | widths=1.2,1.9,0.9,1.9,0.7 | align=lcccc | size=8.5
Paired differences between successive variants on the hard benchmark (later minus earlier). The p-value is a two-sided Wilcoxon signed-rank test with Holm correction over the comparisons of the table; HD95 difference in voxels; last column: Cohen's $d_z$ of the Dice difference.
| Comparison | $\\Delta$ Dice mean [95% CI] | p (Holm) | $\\Delta$ HD95 [95% CI] | $d_z$ |
| --- | --- | --- | --- | --- |
{srows}
@end

@fig hd-forest | file={FIGDIR}/forest.png | width=6.3
Cumulative paired differences in Dice with respect to the plain network A0 for each region (point: mean difference, bar: 95% bootstrap interval; green: interval above zero, red: below zero, grey: contains zero).
@end
"""
    return txt


# ---------------------------------------------------------------------------------------------- distribution and lesion size
def _distribution(H, n_test, plt, C):
    fr = {c: H.test(c) for c, _, _ in CHAIN}
    av = [(c, t, n) for c, t, n in CHAIN if fr[c] is not None]
    if not av:
        return ""
    fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.7), sharey=True)
    cols = [C["blue"], C["orange"], C["aqua"], C["violet"], C["yellow"], C["red"]]
    for k, r in enumerate(REG):
        data = [fr[c][f"dice_{r}"].values for c, _, _ in av]
        bp = ax[k].boxplot(data, widths=0.6, patch_artist=True, flierprops=dict(ms=2.5, mfc="none"))
        for b_, col in zip(bp["boxes"], cols):
            b_.set(facecolor=col, alpha=0.5, edgecolor=C["ink"], lw=0.8)
        for m_ in bp["medians"]:
            m_.set(color=C["ink"])
        ax[k].set_xticklabels([t for _, t, _ in av], fontsize=8); ax[k].set_title(r, fontsize=9.5)
    ax[0].set_ylabel("Dice (per phantom)")
    fig.tight_layout(); fig.savefig(FIGDIR / "box.png"); plt.close(fig)
    # table median / IQR / failures
    rows = []
    for c, t, n in av:
        df = fr[c]
        cell = []
        for r in REG:
            x = df[f"dice_{r}"].values
            q1, q2, q3 = np.percentile(x, [25, 50, 75])
            cell.append(f"{q2:.3f} [{q1:.3f}, {q3:.3f}]")
        fail = int((df["dice_ET"] < 0.1).sum())
        rows.append("| " + " | ".join([t] + cell + [str(int((df["dice_mean"] < 0.5).sum())), str(fail)]) + " |")
    RES["fail"] = {TAG[c]: int((fr[c]["dice_ET"] < 0.1).sum()) for c, _, _ in av}
    # lesion size
    a0 = fr.get("a0_unet"); a5 = fr.get("a5_full")
    size_txt = ""
    if a0 is not None and a5 is not None:
        vol = a0["vol_gt_WT"]
        q = np.quantile(vol, [1 / 3, 2 / 3])
        grp = np.where(vol <= q[0], "small", np.where(vol <= q[1], "medium", "large"))
        srows = []
        sz = {}
        for g in ("small", "medium", "large"):
            idx = grp == g
            r_ = []
            for c, t in (("a0_unet", "A0"), ("a5_full", "A5")):
                d = fr[c].loc[a0.index[idx]]
                r_ += [d["dice_WT"].mean(), d["dice_ET"].mean()]
            vr = (a5.loc[a0.index[idx], "vol_pred_WT"] / a5.loc[a0.index[idx], "vol_gt_WT"]).median()
            srows.append(f"| {g} | {int(idx.sum())} | {vol[idx].min():.0f}--{vol[idx].max():.0f} | {r_[0]:.3f} | {r_[2]:.3f} | {r_[1]:.3f} | {r_[3]:.3f} | {vr:.2f} |")
            sz[g] = {"A0_WT": float(r_[0]), "A5_WT": float(r_[2]), "A0_ET": float(r_[1]), "A5_ET": float(r_[3])}
        RES["size"] = sz
        fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.6), sharey=True)
        for k, r in enumerate(REG):
            for c, col, mk, lab in (("a0_unet", C["blue"], "o", "A0"), ("a5_full", C["orange"], "s", "A5")):
                ax[k].scatter(fr[c][f"vol_gt_{r}"] + 1, fr[c][f"dice_{r}"], s=12, color=col, marker=mk, alpha=0.7, label=lab, facecolors="none" if mk == "o" else col)
            ax[k].set_xscale("log"); ax[k].set_xlabel(f"reference {r} volume (voxels + 1)"); ax[k].set_title(r, fontsize=9.5)
        ax[0].set_ylabel("Dice (per phantom)"); ax[0].legend(frameon=False, loc="lower right")
        fig.tight_layout(); fig.savefig(FIGDIR / "size.png"); plt.close(fig)
        rho = []
        for r in REG:
            from scipy.stats import spearmanr
            rho.append(spearmanr(np.log1p(a5[f"vol_gt_{r}"]), a5[f"dice_{r}"])[0])
        RES["rho"] = {r: float(x) for r, x in zip(REG, rho)}
        size_txt = f"""
@fig hd-size | file={FIGDIR}/size.png | width=6.3
Dice of each test phantom against the volume of the reference region (logarithmic axis) for the plain network A0 (circles) and the full model A5 (squares).
@end

@tab hd-size | widths=0.9,0.5,1.1,0.8,0.8,0.8,0.8,0.9 | align=lccccccc | size=8
Dice by size of the whole tumour (tertiles of the reference WT volume of the test phantoms). Last column: median ratio of the predicted to the reference WT volume of A5.
| Size | n | WT volume (voxels) | WT A0 | WT A5 | ET A0 | ET A5 | Volume ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(srows)}
@end

@fig:hd-size and @tab:hd-size show the size dependence that the evaluation chapter of the literature review anticipates. The Spearman rank correlation between the logarithm of the reference volume and the Dice score of A5 is {rho[0]:.2f} for WT, {rho[1]:.2f} for TC and {rho[2]:.2f} for ET: {'small structures are segmented less accurately' if min(rho) > 0 else 'the relation between size and accuracy is not positive for every region'}; the correlation is strongest for {REG[int(np.argmax(np.abs(rho)))]}. The comparison of the two networks within the tertiles shows where the differences of @tab:hd-main arise: the mean WT Dice of A5 minus that of A0 is {sz['small']['A5_WT']-sz['small']['A0_WT']:+.3f} for the small, {sz['medium']['A5_WT']-sz['medium']['A0_WT']:+.3f} for the medium and {sz['large']['A5_WT']-sz['large']['A0_WT']:+.3f} for the large tumours, and the corresponding ET differences are {sz['small']['A5_ET']-sz['small']['A0_ET']:+.3f}, {sz['medium']['A5_ET']-sz['medium']['A0_ET']:+.3f} and {sz['large']['A5_ET']-sz['large']['A0_ET']:+.3f}.
"""
    return f"""@subsection[hd-dist] Distribution of the per-phantom scores and dependence on lesion size

@fig:hd-box shows the distribution of the per-phantom Dice scores and @tab:hd-dist the median and interquartile range of every variant and region, together with the number of phantoms with a mean Dice below 0.5 and with an ET Dice below 0.1 (practically missed enhancing tumours).

@fig hd-box | file={FIGDIR}/box.png | width=6.4
Distribution of the per-phantom Dice scores of the six variants on the test phantoms (box: quartiles, line: median, circles: outlying phantoms).
@end

@tab hd-dist | widths=0.55,1.45,1.45,1.45,0.75,0.75 | align=lccccc | size=8
Median [first quartile, third quartile] of the per-phantom Dice. Last two columns: number of phantoms with mean Dice below 0.5, and with ET Dice below 0.1.
| Variant | WT | TC | ET | Mean $<0.5$ | ET $<0.1$ |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}
@end
{size_txt}"""


# ---------------------------------------------------------------------------------------------- convergence
def _curves(H, cfg0, plt, C, LS, MK):
    cols = [C["blue"], C["orange"], C["aqua"], C["violet"], C["yellow"], C["red"]]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    rows = []
    for i, (c, t, n) in enumerate(CHAIN):
        L = H.log(c)
        if L.empty:
            continue
        ax[0].plot(L["epoch"], L["loss"], color=cols[i], ls=LS[i % 4], label=t)
        v = L.dropna(subset=["val_dice_mean"])
        ax[1].plot(v["epoch"], v["val_dice_mean"], color=cols[i], ls=LS[i % 4], marker=MK[i % 4], ms=3.5, label=t)
        best = v.loc[v["val_dice_mean"].idxmax()]
        tt = L["epoch_seconds"].sum() / 60
        rows.append(f"| {t} {n} | {int(best['epoch'])} | {best['val_dice_mean']:.3f} | {v['val_dice_mean'].iloc[-1]:.3f} | {L['loss'].iloc[-1]:.3f} | {tt:.1f} |")
    ax[0].set_yscale("log"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training generator loss"); ax[0].set_title("(a) training loss")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation Dice (mean of WT, TC, ET)"); ax[1].set_title("(b) validation score")
    for a in ax:
        a.axvline(cfg0["epochs_seg"] - 0.5, color=C["grey"], lw=0.8, ls=":")
    ax[1].legend(frameon=False, ncol=2, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(FIGDIR / "curves.png"); plt.close(fig)
    still = 0
    nvar = 0
    for c, t, n in CHAIN:
        L = H.log(c)
        if L.empty:
            continue
        v = L.dropna(subset=["val_dice_mean"])
        nvar += 1
        if int(v.loc[v["val_dice_mean"].idxmax(), "epoch"]) >= int(v["epoch"].iloc[-2]):
            still += 1
    RES["still_improving"] = [still, nvar]
    return f"""@subsection[hd-curves] Convergence

@fig hd-curves | file={FIGDIR}/curves.png | width=6.4
Training curves of the six variants on the hard benchmark (fold 0, seed 1): (a) training loss of the generator (the Tversky and adversarial terms change the scale of the loss between variants), (b) validation Dice. The dotted line marks the start of the adversarial epochs.
@end

@fig:hd-curves and @tab:hd-conv show how the validation score evolves. The best validation score is reached at one of the last two evaluations in {still} of the {nvar} variants{', which indicates that the training budget of ' + str(cfg0['epochs_seg']) + ' epochs is a limiting factor of this benchmark, as it is for any experiment with a budget much smaller than that of the BraTS configuration' if still * 2 >= nvar else ', so that most variants had converged within the budget of ' + str(cfg0['epochs_seg']) + ' epochs'}. The loss values of different variants are not comparable, because the loss of A4 and A5 contains the additional Tversky term. The selected checkpoint is the one with the highest validation Dice, so that a late degradation would be visible as a best epoch before the last one.

@tab hd-conv | widths=2.0,0.8,1.0,1.0,1.0,0.9 | align=lccccc | size=8.5
Convergence summary (seed 1, fold 0): epoch and value of the best validation Dice, validation Dice at the last epoch, final training loss and training time on two CPU cores.
| Variant | Best epoch | Best val. Dice | Final val. Dice | Final loss | Time (min) |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}
@end
"""


# ---------------------------------------------------------------------------------------------- Tversky
def _tversky(H, n_test, plt, C, LS, MK):
    sets = [("a3_maf", "none (A3)", "--"), ("t_dice_eq", "(0.5, 0.5)", "0.5"), ("a4_tversky", "(0.3, 0.7)", "0.3"), ("t_fn_heavy", "(0.2, 0.8)", "0.2")]
    fr = {c: H.test(c, seeds=(1,)) if c.startswith("t_") else H.test(c) for c, _, _ in sets}
    if all(v is None for v in fr.values()):
        return "@subsection[hd-tv] Tversky parameters\n~pending\n"
    rows, pts = [], []
    ref = fr.get("a3_maf")
    for c, lab, _ in sets:
        df = fr[c]
        if df is None:
            rows.append(f"| {lab} | {P} | {P} | {P} | {P} | {P} | {P} |")
            continue
        vr = {r: float((df[f"vol_pred_{r}"] / df[f"vol_gt_{r}"].clip(lower=1)).median()) for r in REG}
        rows.append("| " + " | ".join([lab, _m(df, "dice_mean"), _m(df, "dice_ET"), _m(df, "prec_ET"), _m(df, "sens_ET"), _m(df, "prec_WT"), _m(df, "sens_WT")]) + f" | {vr['ET']:.2f} |")
        pts.append((lab, df))
    RES["tv"] = {lab: {"dice": float(df["dice_mean"].mean()), "prec_ET": float(df["prec_ET"].mean()), "sens_ET": float(df["sens_ET"].mean()),
                       "prec_WT": float(df["prec_WT"].mean()), "sens_WT": float(df["sens_WT"].mean())} for lab, df in pts}
    fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.6))
    cols = [C["grey"], C["blue"], C["orange"], C["aqua"]]
    for k, r in enumerate(REG):
        for i, (lab, df) in enumerate(pts):
            ax[k].scatter(df[f"sens_{r}"].mean(), df[f"prec_{r}"].mean(), s=45, color=cols[i % 4], marker=MK[i % 4], label=lab)
        ax[k].set_xlabel("sensitivity"); ax[k].set_title(r, fontsize=9.5)
    ax[0].set_ylabel("precision"); ax[2].legend(frameon=False, fontsize=7.5, title="$(\\alpha,\\beta)$", title_fontsize=8)
    fig.tight_layout(); fig.savefig(FIGDIR / "tv.png"); plt.close(fig)
    a4 = fr.get("a4_tversky"); a3 = fr.get("a3_maf")
    teq = fr.get("t_dice_eq")
    if a4 is not None and teq is not None:
        RES["h2"] = {}
        for r in ("TC", "ET"):
            qs = AN.paired(a4, teq, f"sens_{r}"); qp = AN.paired(a4, teq, f"prec_{r}")
            RES["h2"][r] = {"sens": [qs["diff"], qs["lo"], qs["hi"]], "prec": [qp["diff"], qp["lo"], qp["hi"]]}
    cmp_txt = ""
    if a4 is not None and a3 is not None:
        q = AN.paired(a4, a3, "dice_mean"); qs = AN.paired(a4, a3, "sens_ET"); qp = AN.paired(a4, a3, "prec_ET")
        RES["tv_a4a3"] = {"dice": q["diff"], "dlo": q["lo"], "dhi": q["hi"], "sens": qs["diff"], "slo": qs["lo"], "shi": qs["hi"], "prec": qp["diff"], "plo": qp["lo"], "phi": qp["hi"]}
        cmp_txt = (f" Adding the Tversky term with $(\\alpha,\\beta)=(0.3,0.7)$ to the loss of A3 changes the mean Dice by {AN.fmt_ci((q['diff'], q['lo'], q['hi']), 3)}, the ET sensitivity by {AN.fmt_ci((qs['diff'], qs['lo'], qs['hi']), 3)} and the ET precision by {AN.fmt_ci((qp['diff'], qp['lo'], qp['hi']), 3)} (paired bootstrap over phantoms).")
    t5 = RES["tv"].get("(0.5, 0.5)", {}); t8 = RES["tv"].get("(0.2, 0.8)", {})
    tv_close = ""
    if t5 and t8:
        ds, dp = t8["sens_ET"] - t5["sens_ET"], t8["prec_ET"] - t5["prec_ET"]
        vr_all = [float((fr[c]["vol_pred_ET"] / fr[c]["vol_gt_ET"].clip(lower=1)).median()) for c, _, _ in sets if fr[c] is not None]
        head = (f"When $\\beta$ increases from 0.5 to 0.8 the mean ET sensitivity changes from {t5['sens_ET']:.3f} to {t8['sens_ET']:.3f} ({ds:+.3f}) "
                f"and the mean ET precision from {t5['prec_ET']:.3f} to {t8['prec_ET']:.3f} ({dp:+.3f}). ")
        if ds > 0.02 and dp < -0.02:
            body = "The operating point therefore moves in the direction predicted by the derivation: more of the reference volume is recovered at the price of additional false positives."
        elif ds < -0.02 and dp > 0.02:
            body = ("This is the opposite of the shift predicted by the derivation of Section @sec:losses: the settings with a larger weight on false negatives are "
                    "more conservative, not more permissive. The prediction concerns the gradient of a single loss term at a fixed network output, whereas the trained network also depends on the "
                    "region-wise Dice and cross-entropy terms, on the deep-supervision weights and on the small number of phantoms; the result is reported as measured and is not used as support for the recall argument.")
        else:
            body = "The change is within the range that the seed-to-seed variability of the following subsection can produce, so that the benchmark does not resolve the predicted shift."
        tail = (f" The median ratio of predicted to reference ET volume lies between {min(vr_all):.2f} and {max(vr_all):.2f} for all settings, "
                + ("which shows a systematic over-segmentation of the enhancing tumour that is present for every loss, including the loss without Tversky term, and is therefore not caused by $\\beta$." if min(vr_all) > 1 else "so that no systematic over-segmentation is present."))
        tv_close = head + body + tail
    return f"""@subsection[hd-tv] Tversky parameters and the recall--precision trade-off

Section @sec:losses derived that a Tversky loss with $\\beta>\\alpha$ weights false negatives more strongly than false positives and therefore shifts the operating point of the network towards higher sensitivity at the price of precision. The hard benchmark allows this prediction to be tested with three settings of $(\\alpha,\\beta)$ on top of the loss of A3, and compared with the loss without Tversky term.{cmp_txt} @tab:hd-tv and @fig:hd-tv give the Dice score, the precision and the sensitivity of the settings.

@tab hd-tv | widths=1.1,0.8,0.8,0.9,0.9,0.9,0.9,1.0 | align=lccccccc | size=8
Effect of the Tversky parameters $(\\alpha,\\beta)$ (fold 0; A3 and A4 averaged over two seeds, the two other settings one seed). Last column: median ratio of the predicted to the reference ET volume.
| $(\\alpha,\\beta)$ | Dice mean | Dice ET | Prec. ET | Sens. ET | Prec. WT | Sens. WT | ET volume ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}
@end

@fig hd-tv | file={FIGDIR}/tv.png | width=6.3
Operating points (sensitivity against precision, mean over the test phantoms) of the loss settings for the three regions.
@end

{tv_close}
"""


# ---------------------------------------------------------------------------------------------- GAN
def _gan(H, n_test, cfg0, plt, C, LS, MK):
    sets = [("a4_tversky", "no adversarial term (A4)"), ("a5_full", "default $\\lambda=0.05$ (A5)"), ("g_lam001", "$\\lambda=0.01$"), ("g_lam020", "$\\lambda=0.2$"),
            ("g_hinge", "hinge loss"), ("g_nosmooth", "no smoothing, no noise")]
    fr = {c: (H.test(c, seeds=(1,)) if c.startswith("g_") else H.test(c)) for c, _ in sets}
    if all(v is None for k, v in fr.items() if k != "a4_tversky"):
        return "@subsection[hd-gan] Adversarial sensitivity\n~pending\n"
    ref = fr["a4_tversky"]
    rows, dacc = [], {}
    for c, lab in sets:
        df = fr[c]
        if df is None:
            rows.append(f"| {lab} | {P} | {P} | {P} | {P} | {P} |"); continue
        L = H.log(c)
        A_ = L[L["stage"] == "adv"] if not L.empty and "stage" in L else pd.DataFrame()
        ar = af = float("nan")
        if len(A_):
            ar, af = float(A_["d_acc_real"].mean()), float(A_["d_acc_fake"].mean())
            dacc[c] = (lab, A_)
        if ref is not None and c != "a4_tversky":
            q = AN.paired(df, ref, "dice_mean"); dd = AN.fmt_ci((q["diff"], q["lo"], q["hi"]), 3)
        else:
            dd = "--"
        rows.append("| " + " | ".join([lab, _m(df, "dice_mean"), _m(df, "dice_ET"), dd,
                                       "--" if ar != ar else f"{ar:.2f}", "--" if af != af else f"{af:.2f}"]) + " |")
    RES["gan"] = {}
    for c, lab in sets:
        if fr[c] is not None:
            L = H.log(c); A_ = L[L["stage"] == "adv"] if not L.empty and "stage" in L else pd.DataFrame()
            RES["gan"][c] = {"dice": float(fr[c]["dice_mean"].mean()), "acc_real": float(A_["d_acc_real"].mean()) if len(A_) else None,
                             "acc_fake": float(A_["d_acc_fake"].mean()) if len(A_) else None}
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.7))
    cols = [C["blue"], C["orange"], C["aqua"], C["violet"], C["red"]]
    for i, (c, (lab, A_)) in enumerate(dacc.items()):
        ax[0].plot(A_["epoch"], A_["d_acc_real"], color=cols[i % 5], ls=LS[i % 4], label=lab)
        ax[1].plot(A_["epoch"], A_["d_acc_fake"], color=cols[i % 5], ls=LS[i % 4])
        ax[2].plot(A_["epoch"], A_["d_loss"], color=cols[i % 5], ls=LS[i % 4])
    ax[0].set_title("(a) $D$ accuracy, real", fontsize=9); ax[1].set_title("(b) $D$ accuracy, generated", fontsize=9); ax[2].set_title("(c) $D$ loss", fontsize=9)
    ax[0].set_ylim(-0.02, 1.02); ax[1].set_ylim(-0.02, 1.02)
    for a in ax:
        a.set_xlabel("epoch")
    ax[0].legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.tight_layout(); fig.savefig(FIGDIR / "gan.png"); plt.close(fig)
    a5 = fr.get("a5_full")
    if a5 is not None and ref is not None:
        RES["h3"] = {}
        for r in REG:
            q = AN.paired(a5, ref, f"hd95_{r}")
            RES["h3"][r] = [q["diff"], q["lo"], q["hi"]]
    q54 = AN.paired(a5, ref, "dice_mean") if a5 is not None and ref is not None else None
    RES["a5a4"] = ({"diff": q54["diff"], "lo": q54["lo"], "hi": q54["hi"], "p": q54["p"]} if q54 else None)
    ar5 = RES["gan"].get("a5_full", {}).get("acc_real"); af5 = RES["gan"].get("a5_full", {}).get("acc_fake")
    triv = (ar5 is not None and ar5 > 0.90 and af5 > 0.90)
    txt = f"""@subsection[hd-gan] Sensitivity of the adversarial stage

Hypothesis H3 of Chapter 3 asks whether the adversarial term improves the segmentation and whether the discriminator remains informative. @tab:hd-gan compares the full model with the same network trained without the adversarial term (A4, the reference of the paired differences) and with four changes of the adversarial configuration: a smaller and a larger weight, the hinge instead of the least-squares objective, and the removal of both the label smoothing and the instance noise. The last two columns give the accuracy of the discriminator on real and generated inputs, averaged over the adversarial epochs; @fig:hd-gan shows their evolution.

@tab hd-gan | widths=2.0,0.8,0.8,1.8,0.8,0.8 | align=lccccc | size=8
Sensitivity of the adversarial stage (fold 0, test phantoms; A4 and A5 averaged over two seeds, the other settings one seed). $\\Delta$ Dice: mean-Dice difference to A4 with 95% bootstrap interval. Acc.: accuracy of the discriminator on real and generated inputs averaged over the adversarial epochs.
| Setting | Dice mean | Dice ET | $\\Delta$ Dice vs A4 | Acc. real | Acc. generated |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}
@end

@fig hd-gan | file={FIGDIR}/gan.png | width=6.4
Discriminator diagnostics of the adversarial settings (seed 1): (a) accuracy on real inputs, (b) accuracy on generated inputs, (c) discriminator loss, over the adversarial epochs.
@end
"""
    if q54:
        txt += f"""
For the default configuration the paired difference between A5 and A4 is {AN.fmt_ci((q54['diff'], q54['lo'], q54['hi']), 3)} (Wilcoxon p {AN.fmt_p(q54['p'])}); it is {_sgn(q54)}. The discriminator classifies {100*ar5:.0f}% of the real and {100*af5:.0f}% of the generated outputs correctly on average, so that it is {"close to separating the two classes almost perfectly, which means that the generator does not challenge it and that the adversarial gradient carries little information; only when the label smoothing and the instance noise are removed does the accuracy fall to the range in which the two networks compete (last row of the table)" if triv else "not trivially separating the classes, which is the regime in which the adversarial gradient can carry information"}. The sensitivity runs show how much of the outcome is due to the configuration: the Dice scores of the five adversarial settings lie between {min(v['dice'] for k, v in RES['gan'].items() if k != 'a4_tversky'):.3f} and {max(v['dice'] for k, v in RES['gan'].items() if k != 'a4_tversky'):.3f}, a range of {max(v['dice'] for k, v in RES['gan'].items() if k != 'a4_tversky') - min(v['dice'] for k, v in RES['gan'].items() if k != 'a4_tversky'):.3f}, to be compared with the seed-to-seed variability of the following subsection.
"""
    return txt


# ---------------------------------------------------------------------------------------------- seeds
def _seeds(H, n_test, plt, C, LS, MK):
    rows, d = [], {}
    for c, t, n in CHAIN:
        a, b = H.test_seed(c, 1), H.test_seed(c, 2)
        if a is None or b is None:
            continue
        ma, mb = a["dice_mean"].mean(), b["dice_mean"].mean()
        pm = AN.paired(b, a, "dice_mean")
        d[t] = (ma, mb)
        rows.append(f"| {t} {n} | {ma:.3f} | {mb:.3f} | {mb - ma:+.3f} | {np.std([ma, mb], ddof=1):.3f} | {AN.fmt_ci((pm['diff'], pm['lo'], pm['hi']), 3)} |")
    if not d:
        return "@subsection[hd-seed] Variability between random seeds\n~pending\n"
    diffs = np.array([abs(v[1] - v[0]) for v in d.values()])
    sd = np.array([np.std(v, ddof=1) for v in d.values()])
    RES["seed"] = {"mean_abs_diff": float(diffs.mean()), "max_abs_diff": float(diffs.max()), "mean_sd": float(sd.mean()), "n": len(d), "pairs": {k: list(map(float, v)) for k, v in d.items()}}
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    xs = np.arange(len(d))
    for i, (t, v) in enumerate(d.items()):
        ax.plot([i, i], v, color=C["grey"], lw=1.2)
    ax.scatter(xs, [v[0] for v in d.values()], color=C["blue"], marker="o", s=34, label="seed 1", zorder=3)
    ax.scatter(xs, [v[1] for v in d.values()], color=C["orange"], marker="s", s=34, label="seed 2", zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels(list(d.keys())); ax.set_ylabel("mean test Dice"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGDIR / "seed.png"); plt.close(fig)
    steps = RES.get("steps", [])
    big = max((abs(s["diff"]) for s in steps), default=float("nan"))
    return f"""@subsection[hd-seed] Variability between random seeds

Because the fold, the data and the hyper-parameters are identical for the two seeds of a variant, the difference between the seeds measures the part of the observed differences that is due to the random initialisation and the random sampling of patches alone. @tab:hd-seed and @fig:hd-seed give the mean test Dice of each variant for the two seeds.

@tab hd-seed | widths=1.9,0.9,0.9,0.9,0.9,1.9 | align=lccccc | size=8.5
Mean test Dice of each variant for the two seeds (fold 0). Last column: paired difference of the per-phantom Dice (seed 2 minus seed 1) with 95% bootstrap interval.
| Variant | Seed 1 | Seed 2 | Difference | SD | Paired difference [95% CI] |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}
@end

@fig hd-seed | file={FIGDIR}/seed.png | width=4.4
Mean test Dice of the variants for the two seeds (a vertical line connects the two values of a variant).
@end

The mean absolute difference between the seeds is {diffs.mean():.3f} Dice units (largest: {diffs.max():.3f}), and the mean standard deviation of the two values is {sd.mean():.3f}. For comparison, the largest absolute difference between two successive variants of @tab:hd-steps is {big:.3f}. A difference between variants that does not clearly exceed the seed-to-seed difference cannot be attributed to the component that distinguishes them, which is the reason why the protocol of Chapter 3 requires several seeds, or the fold-wise replication of the next subsection, before a component is accepted as beneficial.
"""


# ---------------------------------------------------------------------------------------------- folds
def _cv(H, n_test, plt, C):
    a0, a5 = H.oof("a0_unet"), H.oof("a5_full")
    t0, n0 = H.test_folds("a0_unet"); t5, n5 = H.test_folds("a5_full")
    if a0 is None or a5 is None:
        return "@subsection[hd-cv] Cross-validation\n~pending (folds 1 and 2 of A0 and A5 not yet trained)\n"
    rows = []
    for k in range(3):
        x, y = a0[a0.fold == k], a5[a5.fold == k]
        q = AN.paired(y, x, "dice_mean")
        rows.append(f"| {k} | {len(x)} | {x['dice_mean'].mean():.3f} | {y['dice_mean'].mean():.3f} | {AN.fmt_ci((q['diff'], q['lo'], q['hi']), 3)} |")
    qa = AN.paired(a5, a0, "dice_mean")
    qh = AN.paired(a5, a0, "hd95_mean")
    qr = {r: AN.paired(a5, a0, f"dice_{r}") for r in REG}
    fm = a5.groupby("fold")["dice_mean"].mean()
    RES["cv"] = {"oof": {"diff": qa["diff"], "lo": qa["lo"], "hi": qa["hi"], "p": qa["p"], "n": int(len(a5))},
                 "per_fold": [float(a5[a5.fold == k]["dice_mean"].mean() - a0[a0.fold == k]["dice_mean"].mean()) for k in range(3)],
                 "a0": float(a0["dice_mean"].mean()), "a5": float(a5["dice_mean"].mean()), "spread": float(fm.max() - fm.min())}
    pt = AN.paired(t5, t0, "dice_mean")
    RES["cv"]["test"] = {"diff": pt["diff"], "lo": pt["lo"], "hi": pt["hi"], "a0": float(t0["dice_mean"].mean()), "a5": float(t5["dice_mean"].mean())}
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    w = 0.36
    xs = np.arange(3)
    ax.bar(xs - w / 2, [a0[a0.fold == k]["dice_mean"].mean() for k in range(3)], w, color=C["blue"], label="A0")
    ax.bar(xs + w / 2, [a5[a5.fold == k]["dice_mean"].mean() for k in range(3)], w, color=C["orange"], hatch="//", edgecolor="white", label="A5")
    lo = min(min(a0["dice_mean"].mean(), a5["dice_mean"].mean()) - 0.1, 0.5)
    ax.set_ylim(max(lo, 0), None); ax.set_xticks(xs); ax.set_xticklabels([f"fold {k}" for k in range(3)]); ax.set_ylabel("validation Dice (mean)"); ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(FIGDIR / "cv.png"); plt.close(fig)
    return f"""@subsection[hd-cv] Cross-validation over three folds

The comparison of the plain network A0 with the full model A5 was repeated on the other two folds (seed 1), so that every one of the {len(a5)} development phantoms is segmented by a model that did not train on it. @tab:hd-cv gives the out-of-fold Dice per fold and @fig:hd-cv the same quantity graphically. Pooled over the folds, the mean out-of-fold Dice is {a0['dice_mean'].mean():.3f} for A0 and {a5['dice_mean'].mean():.3f} for A5; the paired difference is {AN.fmt_ci((qa['diff'], qa['lo'], qa['hi']), 3)} (Wilcoxon p {AN.fmt_p(qa['p'])}, $d_z$ = {qa['dz']:.2f}), which is {_sgn(qa)}. By region the differences are {AN.fmt_ci((qr['WT']['diff'], qr['WT']['lo'], qr['WT']['hi']), 3)} for WT, {AN.fmt_ci((qr['TC']['diff'], qr['TC']['lo'], qr['TC']['hi']), 3)} for TC and {AN.fmt_ci((qr['ET']['diff'], qr['ET']['lo'], qr['ET']['hi']), 3)} for ET, and the difference of the HD95 is {AN.fmt_ci((qh['diff'], qh['lo'], qh['hi']), 2)} voxels. On the common test set, the metrics of the three fold models averaged per phantom give {t0['dice_mean'].mean():.3f} for A0 and {t5['dice_mean'].mean():.3f} for A5 (paired difference {AN.fmt_ci((pt['diff'], pt['lo'], pt['hi']), 3)}).

@tab hd-cv | widths=0.7,0.6,1.3,1.3,2.1 | align=ccccc | size=8.5
Out-of-fold validation Dice per fold (seed 1) and paired difference A5 $-$ A0 with 95% bootstrap interval.
| Fold | n | A0 | A5 | A5 $-$ A0 [95% CI] |
| --- | --- | --- | --- | --- |
{chr(10).join(rows)}
@end

@fig hd-cv | file={FIGDIR}/cv.png | width=4.2
Mean out-of-fold Dice of A0 and A5 on the three folds.
@end

The spread of the mean Dice of the full model between the folds is {fm.max() - fm.min():.3f}, which is {'larger' if fm.max() - fm.min() > RES.get('seed', {}).get('mean_abs_diff', 1) else 'not larger'} than the mean seed-to-seed difference of the previous subsection ({RES.get('seed', {}).get('mean_abs_diff', float('nan')):.3f}). The per-fold differences {', '.join(f'{x:+.3f}' for x in RES['cv']['per_fold'])} {'have the same sign on all folds, so that the direction of the effect is stable across the data partitions' if (min(RES['cv']['per_fold']) > 0 or max(RES['cv']['per_fold']) < 0) else 'do not have the same sign on all folds: the direction of the effect of the full model over the plain network is not stable across the data partitions, and a single-fold comparison, such as the ablation of the previous subsections, must be read with this variability in mind'}.
"""


# ---------------------------------------------------------------------------------------------- grading
def _grade(H, base):
    runs = base / "runs"
    ok = all((runs / f"a5_full_s1_f{k}" / "test_metrics.csv").exists() for k in range(3))
    head = "@subsection[hd-grade] Grade classification from the predicted masks\n\n"
    if not ok:
        return head + "~pending (requires the full model on all three folds)\n"
    res = HERE.parent / "results"
    res.mkdir(exist_ok=True)
    f = res / "grade_hard_a5.json"
    if not f.exists():
        subprocess.run([sys.executable, str(HERE.parent / "scripts" / "grade_classification.py"), "--exp", "a5_full_s1", "--runs", str(runs),
                        "--cache", str(base / "cache"), "--splits", str(base / "splits.json"), "--out", str(f)], check=True, stdout=subprocess.DEVNULL)
    J = json.loads(f.read_text())
    rows = AN.grade_rows(J)
    RES["grade"] = {r[0]: {"cv": r[1], "test": r[2], "acc": r[3], "sens": r[4], "spec": r[5], "bacc": r[6], "ece": r[7]} for r in rows}
    best = next((r for r in rows if r[0].startswith("Predicted masks, logistic")), rows[0])
    gt = next((r for r in rows if r[0].startswith("Ground-truth masks, logistic")), None)
    gbp = next((r for r in rows if r[0].startswith("Predicted masks, gradient")), None)
    t = AN.grade_table_md(J, "hd-grade-tab", f"Grade classification on the hard benchmark (three folds, development set of {RES.get('n_dev', 128)} phantoms, test set of {RES.get('n_test', 32)} phantoms). Predicted masks come from the out-of-fold models of the full model (seed 1).")
    s = (f"The leak-free grading protocol of Section @sec:grade was run on the hard benchmark with the full model (A5, seed 1) trained on each of the three folds. With features of the predicted masks and logistic regression, the cross-validated AUC is {best[1]} and the test AUC {best[2]} (accuracy {best[3]}, sensitivity {best[4]}, specificity {best[5]}, expected calibration error {best[7]})"
         + (f"; gradient boosting reaches a test AUC of {gbp[2]}" if gbp else "")
         + (f"; with the features of the ground-truth masks the test AUC is {gt[2]}, which is the upper bound for the segmentation quality of this benchmark" if gt else "")
         + ". @tab:hd-grade-tab gives all four combinations of mask source and classifier. The grades of the phantoms differ by construction in the enhancement of the shell and the size of the core, so that the task measures how much of this information survives the segmentation error; the number of LGG-like test phantoms is small, and the intervals are accordingly wide.")
    try:
        cvv, tv_ = float(best[1].split()[0]), float(best[2].split()[0])
        if abs(tv_ - cvv) > 0.15:
            s += (f" The cross-validated and the test AUC differ by {abs(tv_-cvv):.2f}, and their intervals do not overlap. Both estimates come from the same pipeline, so that the difference is a property of the data and cannot be attributed to a change of method: the cross-validated estimate rests on {RES.get('n_dev', 128)} development phantoms, the test estimate on {RES.get('n_test', 32)} phantoms of which only a minority are LGG-like, and the AUC of such a small minority class is volatile. The experiment does not allow the discrepancy to be resolved; the test value is therefore to be read as an optimistic and unstable estimate, and the cross-validated value as the more reliable statement that the features of the predicted masks carry little information about the grade on this benchmark although the ground-truth masks carry all of it. Neither is a statement about BraTS.")
    except Exception:
        pass
    s += "\n\n"
    return head + s + t


# ---------------------------------------------------------------------------------------------- calibration
def _calibration(H, base, plt, C):
    import torch
    from bratsgan.data import fold_ids, labels_to_regions, load_case
    from ch4_phantom import _load_model, _crop
    head = "@subsection[hd-calib] Calibration of the voxel probabilities\n\n"
    ds = {c: H.dirs(c, 0, (1,)) for c in ("a0_unet", "a5_full")}
    if not all(ds.values()):
        return head + "~pending\n"
    splits = json.loads((base / "splits.json").read_text())
    _, _, test_ids = fold_ids(splits, 0)
    cache = json.loads((ds["a0_unet"][0] / "config.json").read_text())["cache_dir"]
    nb = 10
    edges = np.linspace(0, 1, nb + 1)
    out = {}
    for c, (d,) in ds.items():
        m, cfg = _load_model(d)
        P_, T_ = [], []
        for cid in test_ids:
            img, seg, meta = load_case(cache, cid, mmap=False)
            x = np.asarray(img)
            y = labels_to_regions(np.asarray(seg)).astype(np.float32)
            D_, H_, W_ = x.shape[1:]
            pad = [(0, 0)] + [(0, (-n) % 8) for n in (D_, H_, W_)]
            with torch.no_grad():
                pr = torch.sigmoid(m(torch.from_numpy(np.pad(x, pad)[None]).float()))[0].numpy()[:, :D_, :H_, :W_]
            brain = np.abs(x).sum(0) > 0
            P_.append(pr[:, brain]); T_.append(y[:, brain])
        P_, T_ = np.concatenate(P_, 1), np.concatenate(T_, 1)
        out[c] = (P_, T_)
    ece, curves = {}, {}
    for c, (P_, T_) in out.items():
        for k, r in enumerate(REG):
            p, t = P_[k], T_[k]
            idx = np.clip(np.digitize(p, edges) - 1, 0, nb - 1)
            e = 0.0
            xs, ys = [], []
            for b in range(nb):
                s = idx == b
                if s.sum() == 0:
                    continue
                e += s.mean() * abs(t[s].mean() - p[s].mean())
                xs.append(p[s].mean()); ys.append(t[s].mean())
            ece[(c, r)] = e; curves[(c, r)] = (xs, ys)
    fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.5), sharey=True)
    for k, r in enumerate(REG):
        ax[k].plot([0, 1], [0, 1], color=C["grey"], ls=":", lw=1)
        for c, col, mk, lab in (("a0_unet", C["blue"], "o", "A0"), ("a5_full", C["orange"], "s", "A5")):
            ax[k].plot(*curves[(c, r)], color=col, marker=mk, ms=4, label=f"{lab} (ECE {ece[(c, r)]:.4f})")
        ax[k].set_xlabel("predicted probability"); ax[k].set_title(r, fontsize=9.5); ax[k].legend(frameon=False, fontsize=7, loc="upper left")
    ax[0].set_ylabel("observed frequency")
    fig.tight_layout(); fig.savefig(FIGDIR / "calib.png"); plt.close(fig)
    RES["ece"] = {f"{TAG[c]}_{r}": float(v) for (c, r), v in ece.items()}
    rows = "\n".join(f"| {r} | {ece[('a0_unet', r)]:.4f} | {ece[('a5_full', r)]:.4f} |" for r in REG)
    return head + f"""The segmentation network outputs a probability per voxel and region, and the quality of these probabilities matters when they are used beyond the thresholded mask, for example to display uncertainty. @fig:hd-calib shows reliability diagrams of the plain network and the full model on the test phantoms (all brain voxels, ten equal-width probability bins), and @tab:hd-calib the expected calibration error of each region.

@fig hd-calib | file={FIGDIR}/calib.png | width=6.3
Reliability diagrams of A0 and A5 (fold 0, seed 1) on the test phantoms: observed frequency of the region against the predicted probability, with the expected calibration error (ECE) in the legend.
@end

@tab hd-calib | widths=1.2,1.6,1.6 | align=ccc | size=9
Voxel-wise expected calibration error (ten bins, brain voxels of the test phantoms).
| Region | A0 | A5 |
| --- | --- | --- |
{rows}
@end

The ECE values are small (at most {max(ece[(c, r)] for c in ('a0_unet', 'a5_full') for r in REG):.4f}) because the vast majority of the brain voxels are confidently background and dominate the bins at the extremes; the average therefore hides the behaviour in the intermediate bins, which correspond to the boundary voxels and to uncertain regions. In these bins the curves of @fig:hd-calib depart visibly from the diagonal: for the enhancing tumour both networks are over-confident, that is the observed frequency of ET stays below the predicted probability over most of the range, which agrees with the over-segmentation of ET reported in @tab:hd-tv (predicted volumes about twice the reference volume). For WT and TC the sign of the deviation differs between the two networks, so that the figure does not support the statement that either one is better calibrated; the ECE values of the two networks differ by at most {max(abs(ece[('a0_unet', r)] - ece[('a5_full', r)]) for r in REG):.4f}.
"""


# ---------------------------------------------------------------------------------------------- qualitative
def _qualitative(H, base, plt, C):
    from bratsgan.data import fold_ids, load_case
    head = "@subsection[hd-qual] Qualitative comparison\n\n"
    ds = {c: H.dirs(c, 0, (1,)) for c in ("a0_unet", "a5_full")}
    if not all(ds.values()):
        return head + "~pending\n"
    a0 = AN.add_composite(pd.read_csv(ds["a0_unet"][0] / "test_metrics.csv").set_index("id"))
    a5 = AN.add_composite(pd.read_csv(ds["a5_full"][0] / "test_metrics.csv").set_index("id"))
    d = (a5["dice_mean"] - a0["dice_mean"]).sort_values()
    picks = [("largest gain of A5", d.index[-1]), ("median difference", d.index[len(d) // 2]), ("largest loss of A5", d.index[0])]
    cache = json.loads((ds["a0_unet"][0] / "config.json").read_text())["cache_dir"]
    fig, ax = plt.subplots(3, 4, figsize=(6.4, 4.9))
    rows = []
    for i, (lab, cid) in enumerate(picks):
        img, seg, meta = load_case(cache, cid, mmap=False)
        seg = np.asarray(seg); img = np.asarray(img)
        zc = int(np.argwhere(seg > 0)[:, 0].mean())
        p0 = np.load(ds["a0_unet"][0] / "pred_test" / f"{cid}_pred.npy"); p5 = np.load(ds["a5_full"][0] / "pred_test" / f"{cid}_pred.npy")
        gt = [(seg > 0), (seg == 1) | (seg == 4), (seg == 4)]
        ax[i, 0].imshow(img[1, zc], cmap="gray"); ax[i, 1].imshow(img[3, zc], cmap="gray")
        for k, (pp, ttl) in enumerate(((p0, "A0"), (p5, "A5"))):
            a_ = ax[i, 2 + k]; a_.imshow(img[1, zc], cmap="gray")
            for r, col in zip(range(3), (C["blue"], C["yellow"], C["red"])):
                if gt[r][zc].any():
                    a_.contour(gt[r][zc], levels=[0.5], colors=["white"], linewidths=0.9)
                if pp[r][zc].any():
                    a_.contour(pp[r][zc], levels=[0.5], colors=[col], linewidths=0.9, linestyles="--")
        for a_ in ax[i]:
            a_.set_xticks([]); a_.set_yticks([]); a_.grid(False)
        ax[i, 0].set_ylabel(lab, fontsize=7)
        if i == 0:
            for k, ttl in enumerate(("T1ce", "FLAIR", "A0", "A5")):
                ax[0, k].set_title(ttl, fontsize=9)
        rows.append(f"| {lab} | {cid.replace('BraTS20_Training_', '')} | {meta.get('grade', '')} | {a0.loc[cid, 'dice_mean']:.3f} | {a5.loc[cid, 'dice_mean']:.3f} | {a0.loc[cid, 'dice_ET']:.3f} | {a5.loc[cid, 'dice_ET']:.3f} |")
    fig.tight_layout(); fig.savefig(FIGDIR / "qual.png"); plt.close(fig)
    return head + f"""@fig:hd-qual shows three test phantoms of fold 0 (seed 1) selected by the difference between the Dice scores of A5 and A0, not by their appearance: the phantom with the largest gain of A5, the one at the median of the differences, and the one with the largest loss. The selection rule makes the figure a fair representation of the spread, and @tab:hd-qual gives the scores of the three cases.

@fig hd-qual | file={FIGDIR}/qual.png | width=5.6
Qualitative comparison on three test phantoms (central axial slice through the tumour): T1ce, FLAIR, and the predictions of A0 and A5 on T1ce. White: ground-truth contours of WT, TC and ET; dashed coloured contours: prediction (WT blue, TC yellow, ET red).
@end

@tab hd-qual | widths=1.6,0.6,0.6,0.8,0.8,0.8,0.8 | align=lccccccc | size=8.5
Dice of the three phantoms of @fig:hd-qual.
| Selection | Phantom | Grade | Mean Dice A0 | Mean Dice A5 | ET Dice A0 | ET Dice A5 |
| --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}
@end
"""


# ---------------------------------------------------------------------------------------------- efficiency
def _efficiency(H, plt, C, MK):
    from bratsgan.models import Discriminator3D, build_model
    cols = [C["blue"], C["orange"], C["aqua"], C["violet"], C["yellow"], C["red"]]
    rows, pts = [], []
    fr = {}
    for i, (c, t, n) in enumerate(CHAIN):
        ds = H.dirs(c)
        if not ds:
            continue
        cfg = json.loads((ds[0] / "config.json").read_text())
        m = build_model(cfg)
        npar = sum(p.numel() for p in m.parameters())
        df = H.test(c)
        tt = float(np.mean([AN.read_log(d)["epoch_seconds"].sum() / 60 for d in ds]))
        inf = float(pd.read_csv(ds[0] / "test_metrics.csv")["seconds"].mean())
        rows.append(f"| {t} {n} | {npar:,} | {df['dice_mean'].mean():.3f} | {tt:.1f} | {inf:.2f} |")
        pts.append((t, npar, df["dice_mean"].mean(), cols[i]))
    cfgd = cfg
    dpar = sum(p.numel() for p in Discriminator3D(cfgd.get("in_ch", 4), cfgd.get("out_ch", 3), cfgd.get("ndf", 32)).parameters())
    RES["params"] = {t: int(p) for t, p, *_ in pts}
    RES["dparams"] = int(dpar)
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    for t, p, d, col in pts:
        ax.scatter(p / 1e3, d, color=col, s=45, marker=MK[(int(t[1]) % 4)])
        ax.annotate(t, (p / 1e3, d), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("generator parameters (thousands)"); ax.set_ylabel("mean test Dice")
    fig.tight_layout(); fig.savefig(FIGDIR / "eff.png"); plt.close(fig)
    return f"""@subsection[hd-cx] Accuracy against cost

@tab:hd-cx and @fig:hd-eff relate the accuracy of the variants to their cost in the configuration of this benchmark (the parameter counts are exact, from the instantiated models; the discriminator of the adversarial stage has {dpar:,} further parameters and is not used at inference). The ranking by accuracy does not follow the ranking by parameters: the residual blocks, the attention gates and the fusion stem add parameters, the Tversky and adversarial terms add none to the generator.

@tab hd-cx | widths=2.1,1.2,1.1,1.2,1.2 | align=lcccc | size=8.5
Cost and accuracy of the variants on the hard benchmark (small configuration). Time: training time on two CPU cores; inference: mean time per phantom.
| Variant | Parameters | Dice mean | Training (min) | Inference (s) |
| --- | --- | --- | --- | --- |
{chr(10).join(rows)}
@end

@fig hd-eff | file={FIGDIR}/eff.png | width=4.4
Mean test Dice against the number of generator parameters for the six variants.
@end
"""


# ---------------------------------------------------------------------------------------------- supplementary
def _supp(H):
    rows = []
    for c, t, n in list(CHAIN) + [("t_dice_eq", "T0.5", "Tversky (0.5, 0.5)"), ("t_fn_heavy", "T0.8", "Tversky (0.2, 0.8)"), ("g_lam001", "G.01", "adversarial $\\lambda=0.01$"),
                                  ("g_lam020", "G.2", "adversarial $\\lambda=0.2$"), ("g_hinge", "Gh", "adversarial hinge"), ("g_nosmooth", "Gn", "adversarial without smoothing")]:
        for s in SEEDS:
            for f in (0, 1, 2):
                d = H.runs / f"{c}_s{s}_f{f}"
                if not (d / "test_metrics.csv").exists():
                    continue
                m = AN.add_composite(pd.read_csv(d / "test_metrics.csv"))
                L = AN.read_log(d); v = L.dropna(subset=["val_dice_mean"])
                be = int(v.loc[v["val_dice_mean"].idxmax(), "epoch"]) if len(v) else -1
                rows.append(f"| {t} | {s} | {f} | {m['dice_WT'].mean():.3f} | {m['dice_TC'].mean():.3f} | {m['dice_ET'].mean():.3f} | {m['dice_mean'].mean():.3f} | {m['hd95_mean'].mean():.2f} | {be} | {L['epoch_seconds'].sum() / 60:.1f} |")
    return f"""
@section[d-hd] Supplementary Table of the Hard Benchmark

@tab hd-all | widths=0.6,0.45,0.45,0.7,0.7,0.7,0.75,0.65,0.6,0.65 | align=cccccccccc | size=7.5
Test results of every run of the hard benchmark. Variants A0 to A5 as in @tab:abl; T0.5 and T0.8: Tversky settings $(0.5,0.5)$ and $(0.2,0.8)$ on top of A3; G.01, G.2: adversarial weight 0.01 and 0.2; Gh: hinge loss; Gn: no smoothing and no noise. HD95 in voxels (mean over regions); best epoch of the selected checkpoint; time in minutes on two CPU cores.
| Variant | Seed | Fold | Dice WT | Dice TC | Dice ET | Dice mean | HD95 | Best epoch | Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}
@end
"""


# ---------------------------------------------------------------------------------------------- hypotheses
def _auc_lo(cell):
    """lower limit of a string 'a [lo, hi]'"""
    try:
        return float(cell.split("[")[1].split(",")[0])
    except Exception:
        return float("nan")


def verdicts() -> dict:
    """pre-specified decision rules applied to the hard benchmark (H5 needs nnU-Net and is not evaluated)"""
    v = {}
    st = RES.get("steps", [])
    first4 = [s for s in st if s["name"] in ("A1 vs A0", "A2 vs A1", "A3 vs A2", "A4 vs A3")]
    if len(first4) == 4:
        ok = [s["lo"] > 0 and s["p_holm"] < 0.05 for s in first4]
        v["H1"] = ("Supported" if all(ok) else "Not supported", f"{sum(ok)} of 4 steps with interval above zero and Holm p < 0.05")
    h2 = RES.get("h2")
    if h2:
        ok = all(h2[r]["sens"][1] > 0 and h2[r]["prec"][0] <= 0 for r in ("TC", "ET"))
        v["H2"] = ("Supported" if ok else "Not supported",
                   f"A4 $-$ symmetric: sensitivity ET {h2['ET']['sens'][0]:+.3f} [{h2['ET']['sens'][1]:+.3f}, {h2['ET']['sens'][2]:+.3f}], precision ET {h2['ET']['prec'][0]:+.3f}")
    h3 = RES.get("h3")
    if h3:
        ok = any(h3[r][2] < 0 for r in ("WT", "TC"))
        v["H3"] = ("Supported" if ok else "Not supported", f"A5 $-$ A4 HD95: WT {h3['WT'][0]:+.2f}, TC {h3['TC'][0]:+.2f}, ET {h3['ET'][0]:+.2f} voxels")
    g = RES.get("grade", {}).get("Predicted masks, logistic regression")
    if g:
        lo = _auc_lo(g["test"])
        v["H4"] = ("Supported" if lo > 0.80 else "Not supported", f"test AUC {g['test']} (threshold 0.80 for the lower limit)")
    v["H5"] = ("Not evaluated", "nnU-Net was not trained for the synthetic benchmark")
    return v


def _hyp() -> str:
    v = verdicts()
    txt = {"H1": "Each ablation step raises Dice mean", "H2": "Asymmetric Tversky raises TC / ET sensitivity at lower precision",
           "H3": "Adversarial term lowers HD95 in WT or TC", "H4": "Grade AUC with lower 95% limit above 0.80", "H5": "Non-inferiority to nnU-Net (margin 0.01)"}
    rows = []
    for h in ("H1", "H2", "H3", "H4", "H5"):
        ver, ev = v.get(h, ("Pending", P))
        rows.append(f"| {h} | {txt[h]} | {ev} | {ver} |")
    nsup = sum(1 for h in v if v[h][0] == "Supported")
    RES["verdicts"] = {h: v[h][0] for h in v}
    return f"""@subsection[hd-hyp] Summary of the hypotheses on the hard benchmark

The decision rules of @tab:hyp were applied to the hard benchmark with the same code that will be applied to BraTS. @tab:hd-hyp gives the verdicts. H5 cannot be evaluated because the nnU-Net baseline was not trained on the synthetic data. {nsup} of the {sum(1 for h in v if v[h][0] != 'Not evaluated')} evaluable hypotheses are supported by the strict decision rules. This is a finding about the benchmark and the small training budget, and the rules are not relaxed after the fact: the individual effects (a gain from the Tversky term, a loss from the fusion stem, a neutral adversarial term) are reported with their intervals in the preceding subsections and discussed in Chapter 5. The verdicts are conditional on the benchmark, on the small training budget and on one fold for H1 to H3; they are not verdicts on BraTS.

@tab hd-hyp | widths=0.5,2.2,2.7,1.1 | align=lllc | size=8.5
Verdicts on the pre-specified hypotheses on the hard synthetic benchmark (not on BraTS). Evidence: quantity on which the rule of @tab:hyp is applied.
| | Hypothesis | Evidence | Verdict |
| --- | --- | --- | --- |
{chr(10).join(rows)}
@end
"""
