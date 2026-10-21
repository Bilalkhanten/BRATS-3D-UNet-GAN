"""Chapter 4, section 4.4: BraTS 2020 results.

All numbers are computed here from the files written by the training / evaluation scripts.  If a file does not exist,
the corresponding cell is rendered as '~pending' (grey italics) and the associated hypothesis verdict is 'Pending'.
Nothing in this module contains a performance number.

Expected layout of the results directory (see RUNBOOK):
    runs/{exp}_f{k}/{config.json, train_log.jsonl, val_metrics.csv, test_metrics.csv, pred_test/*.npy}
    results/grade_a5_full.json                      (scripts/grade_classification.py)
    nnunet/{val_metrics.csv, test_metrics.csv}      (scripts/eval_nnunet.py; per-case, same columns as test_metrics.csv)
    latency.json                                    (experiments/latency.py)
"""
from __future__ import annotations

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

DRYRUN = False
FIGDIR = HERE / "figs_brats"
REG = AN.REG
P = "~pending"
CHAIN = [("a0_unet", "A0", "Plain 3D U-Net"), ("a1_res", "A1", "+ residual blocks"), ("a2_att", "A2", "+ attention gates"),
         ("a3_maf", "A3", "+ MAF stem"), ("a4_tversky", "A4", "+ Tversky loss"), ("a5_full", "A5", "+ adversarial (full)")]


class Data:
    def __init__(self, root):
        self.root = Path(root) if root else None
        self.runs = self.root / "runs" if self.root else None
        self._c = {}

    def fold_dirs(self, exp, split):
        if not self.runs:
            return {}
        out = {}
        for k in range(5):
            d = self.runs / f"{exp}_f{k}"
            if (d / f"{split}_metrics.csv").exists():
                out[k] = d
        return out

    def oof(self, exp):
        """out-of-fold validation metrics; requires all five folds"""
        ds = self.fold_dirs(exp, "val")
        if len(ds) < 5:
            return None
        fr = [pd.read_csv(d / "val_metrics.csv").assign(fold=k) for k, d in ds.items()]
        return AN.add_composite(pd.concat(fr, ignore_index=True).set_index("id"))

    def test(self, exp, fold0=False):
        ds = self.fold_dirs(exp, "test")
        if fold0:
            ds = {k: d for k, d in ds.items() if k == 0}
            if not ds:
                return None
        elif len(ds) < 5:
            return None
        return AN.load_metrics(list(ds.values()), "test")

    def ext(self, which):
        if not self.root:
            return None
        f = self.root / "nnunet" / f"{which}_metrics.csv"
        if not f.exists():
            return None
        return AN.add_composite(pd.read_csv(f).set_index("id"))

    def grade(self):
        if not self.root:
            return None
        f = self.root / "results" / "grade_a5_full.json"
        return json.loads(f.read_text()) if f.exists() else None


def _m(df, col, nd=3):
    return P if df is None or col not in df else AN.fmt(df[col].mean(), nd)


def _mc(df, col, nd=3):
    return P if df is None or col not in df else AN.fmt_ci(AN.ci(df[col].values), nd)


def _pfig(path, text):
    from figstyle import C, plt, setup
    setup()
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    ax.set_facecolor("#f1f1f1"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(C["grey"]); s.set_linestyle("--")
    ax.text(0.5, 0.5, text, ha="center", va="center", color=C["grey"], style="italic", fontsize=11, transform=ax.transAxes)
    fig.savefig(path); plt.close(fig)


def build(brats_dir=None) -> str:
    FIGDIR.mkdir(exist_ok=True)
    from figstyle import C, LS, MK, plt, setup
    setup()
    D = Data(brats_dir)
    out = []
    add = out.append
    anyres = bool(D.fold_dirs("a5_full", "test") or D.fold_dirs("a0_unet", "test"))
    add("@section[brats-res] BraTS 2020 Results\n")
    if DRYRUN:
        add("""@note | title=DRY-RUN BUILD: pipeline test, not BraTS results
Every table and figure of this section was generated from a *dry run* of the complete BraTS pipeline on a small synthetic dataset in the BraTS folder layout (60 synthetic volumes of $48^3$ voxels, 15 test cases, five folds) with a tiny training budget (8 + 3 epochs of 20 iterations). It tests that the scripts, the output files and this builder work end to end. The numbers are **not** BraTS 2020 results, they carry no information about the performance on patients, and statements in the text about 295 development and 74 test patients refer to the real experiment. The nnU-Net rows come from a 5-epoch run of nnU-Net v2 on the same synthetic data (250 iterations per epoch, about ten times more updates than the other networks), so the comparison with nnU-Net is not budget-matched. Do not cite or compare these values with published results.
@end
""")
    add("""All numbers of this section are computed by the analysis code from the output files of the training and evaluation scripts (per-patient metrics, training logs, predicted masks). Cells shown as *pending* correspond to experiments whose output files were not available when this document was built.
""")

    if not anyres:
        return _compact(D)

    # ----------------------------------------------------------------- CV
    A0o, A5o, NNo = D.oof("a0_unet"), D.oof("a5_full"), D.ext("val")
    rows = []
    for lab, df in (("A0 Plain 3D U-Net", A0o), ("A5 Full model", A5o), ("nnU-Net 3d_fullres", NNo)):
        rows.append("| " + " | ".join([lab, str(len(df)) if df is not None else P] + [_m(df, f"dice_{r}") for r in REG] + [_mc(df, "dice_mean")] + [_m(df, f"hd95_{r}", 2) for r in REG]) + " |")
    add(f"""@subsection[cv] Five-fold cross-validation

@tab:cv gives the out-of-fold accuracy: every one of the 295 development patients is segmented by the model that did not train on it, and the metrics are averaged over patients. The interval is the 95% bootstrap interval of the mean over patients (folds are not resampled separately).

@tab cv | widths=1.6,0.45,0.6,0.6,0.6,1.35,0.6,0.6,0.6 | align=lccccccccc | size=7.5
Out-of-fold results over the five folds (295 patients). Dice: WT, TC, ET and the mean of the three; HD95 in mm.
| Model | n | Dice WT | Dice TC | Dice ET | Dice mean [95% CI] | HD95 WT | HD95 TC | HD95 ET |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}
@end
""")
    # per fold
    rows = []
    for k in range(5):
        a = A0o[A0o.fold == k] if A0o is not None else None
        b = A5o[A5o.fold == k] if A5o is not None else None
        n_ = nn_ = None
        rows.append("| " + " | ".join([str(k), str(len(a)) if a is not None else P, _m(a, "dice_mean"), _m(b, "dice_mean"),
                                        (AN.fmt(b["dice_mean"].mean() - a["dice_mean"].mean(), 3) if a is not None and b is not None else P)]) + " |")
    spread = ""
    if A0o is not None and A5o is not None:
        fm = A5o.groupby("fold")["dice_mean"].mean()
        spread = f" The mean Dice of the full model varies between {fm.min():.3f} and {fm.max():.3f} across the folds, i.e. by {fm.max()-fm.min():.3f}, which is the order of magnitude that must be kept in mind when comparing single-fold ablation results."
    add(f"""@tab fold | widths=0.8,0.8,1.5,1.5,1.5 | align=ccccc | size=9
Mean Dice (average of WT, TC, ET) per validation fold.
| Fold | n | A0 | A5 | A5 $-$ A0 |
| --- | --- | --- | --- | --- |
{chr(10).join(rows)}
@end

@tab:fold reports the same quantities per fold.{spread}
""")

    # ----------------------------------------------------------------- test
    A0t, A5t, NNt = D.test("a0_unet"), D.test("a5_full"), D.ext("test")
    rows = []
    for lab, df in (("A0 Plain 3D U-Net", A0t), ("A5 Full model", A5t), ("nnU-Net 3d_fullres", NNt)):
        rows.append("| " + " | ".join([lab] + [_m(df, f"dice_{r}") for r in REG] + [_mc(df, "dice_mean")] + [_m(df, f"hd95_{r}", 2) for r in REG]) + " |")
    sub = []
    for lab, df in (("A0", A0t), ("A5", A5t), ("nnU-Net", NNt)):
        for g in ("HGG", "LGG"):
            s = None if df is None else df[df.grade == g]
            sub.append("| " + " | ".join([f"{lab} {g}", str(len(s)) if s is not None else P, _m(s, "dice_WT"), _m(s, "dice_TC"), _m(s, "dice_ET"), _m(s, "hd95_WT", 2), _m(s, "hd95_TC", 2), _m(s, "hd95_ET", 2)]) + " |")
    pair = []
    for lab, a, b in (("A5 $-$ A0", A5t, A0t), ("nnU-Net $-$ A5", NNt, A5t)):
        for r in REG + ("mean",):
            col = f"dice_{r}"
            if a is None or b is None:
                pair.append(f"| {lab} | Dice {r} | {P} | {P} | {P} |"); continue
            q = AN.paired(a, b, col)
            pair.append(f"| {lab} | Dice {r} | {AN.fmt_ci((q['diff'], q['lo'], q['hi']), 3)} | {AN.fmt_p(q['p'])} | {q['dz']:.2f} |")
    add(f"""@subsection[test] Held-out test set

The 74 test patients were never used for training, model selection or any design decision. For the models trained on the five folds, the metric of each patient is the mean over the five fold models (the protocol of Section @sec:eval-protocol). @tab:test and @tab:test-sub give the results and @tab:test-pair the paired differences; @fig:box shows the distribution of the per-patient Dice scores.

@tab test | widths=1.6,0.6,0.6,0.6,1.35,0.6,0.6,0.6 | align=lccccccc | size=7.5
Test-set results (74 patients). Dice: WT, TC, ET and mean with 95% bootstrap interval; HD95 in mm.
| Model | Dice WT | Dice TC | Dice ET | Dice mean [95% CI] | HD95 WT | HD95 TC | HD95 ET |
| --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}
@end

@tab test-sub | widths=1.5,0.5,0.7,0.7,0.7,0.7,0.7,0.7 | align=lccccccc | size=7.5
Test-set results by grade (59 HGG and 15 LGG patients; intervals for subgroups of this size are optimistic, Section @sec:v6s).
| Model, grade | n | Dice WT | Dice TC | Dice ET | HD95 WT | HD95 TC | HD95 ET |
| --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(sub)}
@end

@tab test-pair | widths=1.3,1.0,2.0,0.9,0.7 | align=llccc | size=8.5
Paired differences of the Dice scores on the test set (95% bootstrap interval, two-sided Wilcoxon *p*, Cohen's $d_z$).
| Comparison | Quantity | Difference [95% CI] | p | $d_z$ |
| --- | --- | --- | --- | --- |
{chr(10).join(pair)}
@end
""")
    if A0t is not None and A5t is not None:
        fig, ax = plt.subplots(figsize=(6.2, 3.0))
        data, pos, cols = [], [], []
        for i, r in enumerate(REG):
            for j, (df, c) in enumerate(((A0t, C["blue"]), (A5t, C["orange"]), (NNt, C["aqua"]))):
                if df is None:
                    continue
                data.append(df[f"dice_{r}"].values); pos.append(i + 0.27 * (j - 1)); cols.append(c)
        bp = ax.boxplot(data, positions=pos, widths=0.22, patch_artist=True, showfliers=True, flierprops=dict(ms=2.5, mfc="none"))
        for b_, c in zip(bp["boxes"], cols):
            b_.set(facecolor=c, alpha=0.5, edgecolor=C["ink"], lw=0.8)
        for m_ in bp["medians"]:
            m_.set(color=C["ink"])
        ax.set_xticks(range(3)); ax.set_xticklabels(REG); ax.set_ylabel("Dice (per patient)")
        ax.plot([], [], color=C["blue"], lw=6, alpha=0.5, label="A0"); ax.plot([], [], color=C["orange"], lw=6, alpha=0.5, label="A5")
        if NNt is not None:
            ax.plot([], [], color=C["aqua"], lw=6, alpha=0.5, label="nnU-Net")
        ax.legend(frameon=False, loc="lower left")
        fig.savefig(FIGDIR / "box.png"); plt.close(fig)
    else:
        _pfig(FIGDIR / "box.png", "Figure pending: per-patient Dice distributions\n(requires test_metrics.csv of A0, A5, nnU-Net)")
    add(f"""@fig box | file={FIGDIR}/box.png | width=6.0
Distribution of the per-patient Dice scores on the test set for the three regions (box: quartiles, line: median, circles: outlying patients).
@end
""")

    # ----------------------------------------------------------------- ablation
    ab = {t: D.test(e, fold0=True) for e, t, _ in CHAIN}
    rows = []
    for e, t, n in CHAIN:
        df = ab[t]
        rows.append("| " + " | ".join([f"{t} {n}"] + [_m(df, f"dice_{r}") for r in REG] + [_mc(df, "dice_mean")] + [_m(df, "hd95_mean", 2)]) + " |")
    add(f"""@subsection[abl-res] Ablation of the architecture and the objective

@tab:abl-res reports the cumulative ablation of @tab:abl on the common test set; all variants were trained on fold 0 with seed 42.

@tab abl-res | widths=2.1,0.7,0.7,0.7,1.5,0.7 | align=lccccc | size=8
Ablation chain on the test set (fold-0 models, 74 patients). HD95 is the mean over the three regions (mm).
| Variant | Dice WT | Dice TC | Dice ET | Dice mean [95% CI] | HD95 mean |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}
@end
""")
    h1_rows, h1_ok, h1_have = [], [], True
    pv = []
    steps = [("a1_res", "a0_unet"), ("a2_att", "a1_res"), ("a3_maf", "a2_att"), ("a4_tversky", "a3_maf")]
    tag = {e: t for e, t, _ in CHAIN}
    res = []
    for e1, e0 in steps:
        a, b = ab[tag[e1]], ab[tag[e0]]
        if a is None or b is None:
            h1_have = False
            res.append(None)
        else:
            res.append(AN.paired(a, b, "dice_mean"))
    ph = S.holm([r["p"] for r in res]) if h1_have else [None] * 4
    for i, (e1, e0) in enumerate(steps):
        if res[i] is None:
            h1_rows.append(f"| {tag[e1]} vs {tag[e0]} | {P} | {P} | {P} | {P} |")
        else:
            r = res[i]
            if ph[i] is None:
                h1_rows.append(f"| {tag[e1]} vs {tag[e0]} | {AN.fmt_ci((r['diff'], r['lo'], r['hi']), 4)} | {P} | {r['dz']:.2f} | {P} |")
                continue
            ok = r["lo"] > 0 and ph[i] < 0.05
            h1_ok.append(ok)
            h1_rows.append(f"| {tag[e1]} vs {tag[e0]} | {AN.fmt_ci((r['diff'], r['lo'], r['hi']), 4)} | {AN.fmt_p(float(ph[i]))} | {r['dz']:.2f} | {'yes' if ok else 'no'} |")
    add(f"""@tab h1 | widths=1.4,2.0,0.9,0.7,1.0 | align=lcccc | size=8.5
Pre-specified tests of H1: paired differences of the mean Dice between successive variants on the test set (later minus earlier), with Holm-adjusted Wilcoxon *p* over the four comparisons.
| Comparison | $\\Delta$ Dice mean [95% CI] | p (Holm) | $d_z$ | Supports H1? |
| --- | --- | --- | --- | --- |
{chr(10).join(h1_rows)}
@end
""")
    H1 = ("Pending" if not h1_have else ("Supported" if all(h1_ok) else (f"Not supported ({sum(h1_ok)} of 4 steps)" if sum(h1_ok) else "Not supported")))
    H1ev = P if not h1_have else f"{sum(h1_ok)} of 4 steps with interval above zero"

    # ----------------------------------------------------------------- tversky H2 / GAN H3
    A4, A5f, teq, tfn = ab["A4"], ab["A5"], D.test("t_dice_eq", True), D.test("t_fn_heavy", True)
    rows = []
    for lab, df in (("$\\alpha=\\beta=0.5$ (t_dice_eq)", teq), ("$\\alpha=0.3,\\ \\beta=0.7$ (A4)", A4), ("$\\alpha=0.2,\\ \\beta=0.8$ (t_fn_heavy)", tfn)):
        rows.append("| " + " | ".join([lab] + [_m(df, f"{m}_{r}") for r in ("TC", "ET") for m in ("sens", "prec")] + [_m(df, "dice_mean")]) + " |")
    h2 = []
    h2_ok = True
    h2_have = A4 is not None and teq is not None
    for r in ("TC", "ET"):
        if not h2_have:
            h2.append(f"| {r} | {P} | {P} |"); h2_ok = False; continue
        s_ = AN.paired(A4, teq, f"sens_{r}")
        p_ = AN.paired(A4, teq, f"prec_{r}")
        h2.append(f"| {r} | {AN.fmt_ci((s_['diff'], s_['lo'], s_['hi']), 3)} | {AN.fmt_ci((p_['diff'], p_['lo'], p_['hi']), 3)} |")
        h2_ok &= (s_["lo"] > 0 and p_["diff"] <= 0)
    H2 = "Pending" if not h2_have else ("Supported" if h2_ok else "Not supported")
    H2ev = P if not h2_have else "sensitivity and precision differences, @tab:tv-pair"
    add(f"""@subsection[tv-res] Asymmetric Tversky loss

@tab tv | widths=2.2,0.65,0.65,0.65,0.65,0.7 | align=lccccc | size=8
Effect of the asymmetry of the Tversky term (fold-0 models, test set): sensitivity and precision of TC and ET, and the mean Dice.
| Setting | Sens TC | Prec TC | Sens ET | Prec ET | Dice mean |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}
@end

@tab tv-pair | widths=1.0,2.4,2.4 | align=lcc | size=8.5
Pre-specified test of H2: paired difference A4 minus the symmetric setting, sensitivity and precision (95% bootstrap interval).
| Region | $\\Delta$ Sensitivity [95% CI] | $\\Delta$ Precision [95% CI] |
| --- | --- | --- |
{chr(10).join(h2)}
@end
""")
    a, b = A5f, A4
    g_rows, hd_ok = [], False
    have3 = a is not None and b is not None
    for r in REG + ("mean",):
        for m in ("dice", "hd95"):
            col = f"{m}_{r}"
            if not have3:
                g_rows.append(f"| {m.upper() if m=='hd95' else 'Dice'} {r} | {P} | {P} |"); continue
            q = AN.paired(a, b, col)
            g_rows.append(f"| {'HD95' if m=='hd95' else 'Dice'} {r} | {AN.fmt_ci((q['diff'], q['lo'], q['hi']), 3 if m=='dice' else 2)} | {AN.fmt_p(q['p'])} |")
            if m == "hd95" and r in ("WT", "TC") and q["hi"] < 0:
                hd_ok = True
    H3 = "Pending" if not have3 else ("Supported" if hd_ok else "Not supported")
    H3ev = P if not have3 else "HD95 and Dice differences, @tab:gan-pair"
    sweep = []
    for lab, e in (("$\\lambda_{\\max}=0.01$", "g_lam001"), ("$\\lambda_{\\max}=0.05$ (A5)", "a5_full"), ("$\\lambda_{\\max}=0.20$", "g_lam020"),
                   ("hinge loss", "g_hinge"), ("no smoothing / noise", "g_nosmooth"), ("no adversarial term (A4)", "a4_tversky")):
        df = D.test(e, True)
        L = AN.read_log(D.runs / f"{e}_f0") if D.runs and (D.runs / f"{e}_f0").exists() else pd.DataFrame()
        accs = P
        if not L.empty and "d_acc_real" in L:
            A_ = L[L["stage"] == "adv"]
            accs = f"{A_['d_acc_real'].mean():.2f} / {A_['d_acc_fake'].mean():.2f}" if len(A_) else P
        elif e == "a4_tversky":
            accs = "--"
        sweep.append("| " + " | ".join([lab, _m(df, "dice_mean"), _m(df, "hd95_mean", 2), accs]) + " |")
    add(f"""@subsection[gan-res] Adversarial refinement

@tab gan-pair | widths=1.5,2.6,0.9 | align=lcc | size=8.5
Pre-specified test of H3: paired difference A5 (with adversarial term) minus A4 (without), test set (95% bootstrap interval; HD95 in mm, negative values are improvements).
| Quantity | Difference [95% CI] | p |
| --- | --- | --- |
{chr(10).join(g_rows)}
@end

@tab gan-sweep | widths=2.3,0.9,0.9,1.6 | align=lccc | size=8.5
Sensitivity analysis of the adversarial term (fold-0 models, test set). Last column: mean accuracy of the discriminator on real / generated maps during the adversarial epochs.
| Setting | Dice mean | HD95 mean | $D$ accuracy (real / fake) |
| --- | --- | --- | --- |
{chr(10).join(sweep)}
@end
""")
    # training curves + D diag (fold 0)
    d0 = D.runs / "a5_full_f0" if D.runs else None
    if d0 is not None and (d0 / "train_log.jsonl").exists():
        L = AN.read_log(d0)
        fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
        ax[0].plot(L["epoch"], L["loss"], color=C["blue"]); ax[0].set_yscale("log"); ax[0].set_xlabel("epoch"); ax[0].set_title("(a) training loss")
        v = L.dropna(subset=["val_dice_mean"])
        ax[1].plot(v["epoch"], v["val_dice_mean"], color=C["orange"], marker="o", ms=3); ax[1].set_xlabel("epoch"); ax[1].set_title("(b) validation Dice (mean of WT, TC, ET)")
        A_ = L[L["stage"] == "adv"]
        if len(A_):
            for a_ in ax:
                a_.axvline(A_["epoch"].iloc[0] - 0.5, color=C["grey"], ls=":", lw=0.8)
        fig.tight_layout(); fig.savefig(FIGDIR / "curves.png"); plt.close(fig)
    else:
        _pfig(FIGDIR / "curves.png", "Figure pending: training curves of the full model, fold 0\n(requires runs/a5_full_f0/train_log.jsonl)")
    add(f"""@fig curves | file={FIGDIR}/curves.png | width=6.2
Training and validation curves of the full model on fold 0. The dotted line marks the start of the adversarial epochs.
@end
""")

    # ----------------------------------------------------------------- nnunet H5
    H5have = NNt is not None and A5t is not None
    if H5have:
        q = AN.paired(NNt, A5t, "dice_mean")
        H5 = "Supported" if q["hi"] < 0.01 else "Not supported"
        H5ev = f"baseline $-$ full: {AN.fmt_ci((q['diff'], q['lo'], q['hi']), 4)} (margin 0.01)"
    else:
        H5, H5ev = "Pending", P
    add(f"""@subsection[nn-res] Comparison with nnU-Net

The comparison with the nnU-Net baseline is given in @tab:test and @tab:test-pair; the decision on H5 uses the upper limit of the interval of the difference (baseline minus full model) of the mean Dice, against the margin 0.01, in the last row of the hypothesis table (@tab:hyp-res). {"" if H5have else "Neither the nnU-Net predictions nor the full-model test metrics are available in this version."}
""")

    # ----------------------------------------------------------------- grade H4
    J = D.grade()
    if J:
        add("@subsection[grade-res] LGG/HGG classification\n")
        add(AN.grade_table_md(J, "grade-tab", "Grade classification from segmentation-derived features (CV: out-of-fold cross-validation over the 295 development patients; test: 74 held-out patients, mean over the five fold models). ECE: expected calibration error."))
        r = J.get("pred_logreg")
        lo = r["test"]["auc_ci"][0]
        H4 = "Supported" if lo > 0.80 else "Not supported"
        H4ev = f"test AUC {r['test']['auc']:.3f}, lower limit {lo:.3f} (threshold 0.80)"
    else:
        add("@subsection[grade-res] LGG/HGG classification\n")
        add(f"""@tab grade-tab | widths=2.0,1.35,1.35,0.6,0.6,0.6,0.6,0.5 | align=lccccccc | size=7.5
Grade classification from segmentation-derived features.
| Features from | CV AUC [95% CI] | Test AUC [95% CI] | Acc | Sens | Spec | Bal. acc | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Predicted masks, logistic regression | {P} | {P} | {P} | {P} | {P} | {P} | {P} |
| Predicted masks, gradient boosting | {P} | {P} | {P} | {P} | {P} | {P} | {P} |
| Ground-truth masks, logistic regression | {P} | {P} | {P} | {P} | {P} | {P} | {P} |
| Ground-truth masks, gradient boosting | {P} | {P} | {P} | {P} | {P} | {P} | {P} |
@end
""")
        H4, H4ev = "Pending", P

    # ----------------------------------------------------------------- error analysis
    rows_sz, rows_et = [], []
    for tag_, df in (("A0", ab["A0"]), ("A5", ab["A5"])):
        if df is None or "vol_gt_WT" not in df:
            rows_sz.append(f"| {tag_} | {P} | {P} | {P} | {P} |")
            rows_et.append(f"| {tag_} | {P} | {P} | {P} | {P} | {P} |")
            continue
        q = df["vol_gt_WT"].quantile([1 / 3, 2 / 3]).values
        grp = pd.cut(df["vol_gt_WT"], [-np.inf, q[0], q[1], np.inf], labels=["small", "medium", "large"])
        cells = [f"{df.loc[grp == g, 'dice_WT'].mean():.3f} (n={int((grp == g).sum())})" for g in ("small", "medium", "large")]
        rows_sz.append(f"| {tag_} | {cells[0]} | {cells[1]} | {cells[2]} | {q[0]:.0f} / {q[1]:.0f} |")
        has = df["vol_gt_ET"] > 0
        fp = (df.loc[~has, "vol_pred_ET"] > 0)
        rows_et.append(f"| {tag_} | {int(has.sum())} | {int((~has).sum())} | {AN.fmt(df.loc[has, 'dice_ET'].mean(), 3) if has.any() else 'n/a'} | "
                       f"{(AN.fmt(float(fp.mean()), 2) if (~has).any() else 'n/a')} | {AN.fmt(df.loc[~has, 'hd95_ET'].mean(), 1) if (~has).any() else 'n/a'} |")
    add(f"""@subsection[err-res] Dependence on lesion size and absent enhancing tumour

Two structural properties of the data explain much of the variation of the scores. First, the Dice coefficient depends on the size of the region: a given boundary error removes a larger fraction of a small lesion. @tab:size splits the test patients into terciles of the reference whole-tumour volume and gives the Dice of WT per tercile. Second, many LGG patients (and some HGG patients) have no enhancing tumour at all, and for them the Dice of ET is 0 or 1 by convention and the HD95 is 0 or 373.13 mm. @tab:et-empty separates the patients with and without reference ET, and reports for the latter the fraction in which the model predicts any enhancing voxel (a false alarm) and the mean HD95. Both analyses use the fold-0 models of A0 and A5 on the test set, so that every number refers to one trained model.

@tab size | widths=0.8,1.6,1.6,1.6,1.6 | align=ccccc | size=8.5
Dice of the whole tumour by tercile of the reference volume (test set, fold-0 models); the last column gives the tercile limits in voxels.
| Model | Small | Medium | Large | Limits (voxels) |
| --- | --- | --- | --- | --- |
{chr(10).join(rows_sz)}
@end

@tab et-empty | widths=0.7,1.1,1.2,1.3,1.4,1.3 | align=cccccc | size=8
Patients with and without reference enhancing tumour (test set, fold-0 models): numbers, mean Dice of ET among patients with ET, fraction of patients without ET in whom any ET voxel is predicted, and their mean HD95 of ET (mm).
| Model | With ET | Without ET | Dice ET (with) | False-alarm fraction | HD95 ET (without) |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(rows_et)}
@end
""")

    # ----------------------------------------------------------------- runtime
    lat = D.root / "latency.json" if D.root else None
    rt = []
    for e, t, n in CHAIN:
        df = ab[t]
        rt.append(f"| {t} {n} | {AN.fmt(df['seconds'].mean(), 1) if df is not None and 'seconds' in df else P} |")
    add(f"""@subsection[rt-res] Inference time

@tab rt | widths=2.8,2.2 | align=lc | size=9
Mean time to segment one patient (whole volume, sliding window with 50% overlap, no mirroring) as recorded during the test evaluation; it depends on the GPU used, which is stated in the RUNBOOK of the experiment.
| Variant | Seconds per patient |
| --- | --- |
{chr(10).join(rt)}
@end
""")

    if lat is not None and lat.exists():
        L = json.loads(lat.read_text())
        rows_l = "\n".join(f"| {r['variant']} | {1000*r['median_s']:.0f} | {1000*r['min_s']:.0f} | {1000*r['max_s']:.0f} |" for r in L["rows"])
        add(f"""@tab lat-gpu | widths=2.4,1.3,1.3,1.3 | align=lccc | size=9
Forward-pass time (milliseconds) of the generator variants on a ${L['patch']}^3$ patch, batch size one, on the experiment machine ({L['device']}, PyTorch {L['torch']}).
| Variant | Median | Minimum | Maximum |
| --- | --- | --- | --- |
{rows_l}
@end
""")
    else:
        add("~Forward-pass latency on the experiment GPU: pending (latency.json).\n")

    if DRYRUN:
        t5, t0 = D.test("a5_full"), D.test("a0_unet")
        def _row(lab, df):
            v = [f"{100 * df[f'dice_{r}'].mean():.2f}" for r in ("ET", "WT", "TC")]
            return f"| **{lab}** | synthetic BraTS layout, test (15 cases) | {v[0]} | {v[1]} | {v[2]} | Dry run, not BraTS |"
        if t5 is not None and t0 is not None:
            add(f"""@subsection[brats-ref] Reference values from the literature and the dry run

@tab:brats-ref sets the published Dice scores (percent) of @tab:pubres beside the values of the dry run of this build. The dry-run rows come from a pipeline test on synthetic data with a tiny training budget; they are not BraTS results and must not be compared with the published rows.

@tab brats-ref | widths=1.9,1.7,0.7,0.7,0.7,1.2 | align=llcccl | size=8.5
Published Dice scores (%) and the values of the dry run (dry-run rows are not BraTS results).
| Reference | Data, set | ET | WT | TC | Note |
| --- | --- | --- | --- | --- | --- |
| nnU-Net, BraTS 2020 entry | BraTS 2020, test | 82.03 | 88.95 | 85.06 | Ensemble of 25 models |
| Myronenko, ensemble | BraTS 2018, test | 76.64 | 88.39 | 81.54 | 191 test cases |
| No New-Net | BraTS 2018, test | 77.88 | 87.81 | 80.62 | Second place |
{_row("This thesis, A5 (DRY RUN)", t5)}
{_row("This thesis, A0 (DRY RUN)", t0)}
@end
""")
    # ----------------------------------------------------------------- hypotheses
    rows = [("H1", "Each ablation step raises Dice mean", H1ev, H1), ("H2", "Asymmetric Tversky raises TC / ET sensitivity at lower precision", H2ev, H2),
            ("H3", "Adversarial term lowers HD95 in WT or TC", H3ev, H3), ("H4", "Grade AUC with lower 95% limit above 0.80", H4ev, H4),
            ("H5", "Non-inferiority to nnU-Net (margin 0.01)", H5ev, H5)]
    add(f"""@subsection[hyp-sum] Summary of the hypotheses

@tab hyp-res | widths=0.5,2.4,2.4,1.1 | align=lllc | size=8.5
Verdicts on the pre-specified hypotheses, computed by the analysis code from the rules of @tab:hyp. "Pending" means that the required experiment output was not available.
| | Hypothesis | Evidence | Verdict |
| --- | --- | --- | --- |
{chr(10).join('| ' + ' | '.join([a, b, c, ('~' + d if d == 'Pending' else d)]) + ' |' for a, b, c, d in rows)}
@end
""")
    return "\n".join(out)


def _compact(D) -> str:
    """Section shown when no BraTS output exists: status, plan of the analyses, hypothesis table (all pending), real latency table."""
    rows = [("Five-fold cross-validation", "runs/{a0_unet,a5_full}_f{0-4}/val_metrics.csv", "Out-of-fold Dice and HD95 of A0, A5 and nnU-Net; per-fold table"),
            ("Held-out test set (74 patients)", "runs/*/test_metrics.csv, nnunet/test_metrics.csv", "Dice and HD95 per region, by grade, paired differences, box plots"),
            ("Ablation A0--A5", "runs/{a0_unet,...,a5_full}_f0/test_metrics.csv", "Ablation table, paired tests of H1 with Holm correction"),
            ("Tversky parameters", "runs/{t_dice_eq,a4_tversky,t_fn_heavy}_f0", "Sensitivity and precision of TC and ET, test of H2"),
            ("Adversarial term", "runs/{g_*,a4_tversky,a5_full}_f0", "Paired differences A5 $-$ A4, sensitivity sweep, discriminator accuracy, test of H3"),
            ("Grade classification", "results/grade_a5_full.json", "AUC, accuracy, calibration for predicted and reference masks, test of H4"),
            ("Comparison with nnU-Net", "nnunet/{val,test}_metrics.csv", "Test of H5 (non-inferiority margin 0.01)"),
            ("Lesion size, empty ET, inference time", "runs/*/test_metrics.csv", "Dice by size tercile, ET false alarms, seconds per patient")]
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    hyp = [("H1", "Each ablation step raises Dice mean"), ("H2", "Asymmetric Tversky raises TC / ET sensitivity at lower precision"),
           ("H3", "Adversarial term lowers HD95 in WT or TC"), ("H4", "Grade AUC with lower 95% limit above 0.80"), ("H5", "Non-inferiority to nnU-Net (margin 0.01)")]
    hrows = "\n".join(f"| {a} | {b} | {P} | ~Pending |" for a, b in hyp)
    txt = f"""@section[brats-res] BraTS 2020 Results

@note | title=Status of this section
The BraTS 2020 experiments require a GPU and the BraTS data, which were not available in the environment in which this document was prepared. **No BraTS number has been estimated, extrapolated or taken from the synthetic studies, and none appears in this thesis.** The code, the protocol, the pre-specified hypotheses and the analysis are complete and tested; the evidence that can be produced without BraTS is given in Sections @sec:phantom and @sec:hard. This section lists the analyses that will be generated from the output files of the training scripts and gives the verdicts of the pre-specified hypotheses, which read "Pending" until then.
@end

@tab brats-plan | widths=1.5,2.4,2.6 | align=lll | size=8
BraTS 2020 analyses of the framework: the experiment, the output files that the training scripts write (relative to the results directory, see the RUNBOOK) and the tables and figures that the thesis builder generates from them.
| Analysis | Input files | Generated output |
| --- | --- | --- |
{body}
@end

When the output files exist, the builder replaces this section by the full results section: the tables of the cross-validation (295 patients) and of the test set (74 patients), the paired-difference tables of the ablation, the sensitivity analyses, the grading table, the box plots and the error analyses, all computed by the same code that produced the corresponding analyses of the synthetic studies. The command sequence is given in the RUNBOOK that accompanies the thesis, and the number of runs and generator updates of the complete matrix is given in @tab:cost of Appendix @app:plan-app.

@subsection[brats-ref] Reference values from the literature

The BraTS cells of this thesis are pending, and no value has been estimated for them. @tab:brats-ref gives the published Dice scores (percent) that the finished BraTS results will be compared with, copied from @tab:pubres, where the primary sources are cited. The last row is the place for the results of this thesis. The published entries use other editions and test sets, so that the comparison is an orientation and not a test; the paired comparison with the plain network A0 and with nnU-Net on the same patients is the evidence of this thesis.

@tab brats-ref | widths=1.9,1.7,0.7,0.7,0.7,1.2 | align=llcccl | size=8.5
Published Dice scores (%) as reference for the BraTS results of this thesis (values of @tab:pubres), and the row reserved for the thesis.
| Reference | Data, set | ET | WT | TC | Note |
| --- | --- | --- | --- | --- | --- |
| nnU-Net, BraTS 2020 entry | BraTS 2020, test | 82.03 | 88.95 | 85.06 | Ensemble of 25 models |
| Myronenko, ensemble | BraTS 2018, test | 76.64 | 88.39 | 81.54 | 191 test cases |
| No New-Net | BraTS 2018, test | 77.88 | 87.81 | 80.62 | Second place |
| **This thesis, A5 (full model)** | BraTS 2020, test (74 cases) | ~pending | ~pending | ~pending | To be filled from the runs |
| **This thesis, A0 (plain U-Net)** | BraTS 2020, test (74 cases) | ~pending | ~pending | ~pending | To be filled from the runs |
@end

@subsection[hyp-sum] Summary of the hypotheses

@tab hyp-res | widths=0.5,2.4,2.4,1.1 | align=lllc | size=8.5
Verdicts on the pre-specified hypotheses on BraTS 2020, computed by the analysis code from the rules of @tab:hyp. "Pending" means that the required experiment output was not available. The corresponding analyses on the hard synthetic benchmark are reported in Section @sec:hard.
| | Hypothesis | Evidence | Verdict |
| --- | --- | --- | --- |
{hrows}
@end
"""
    lat = D.root / "latency.json" if D.root else None
    if lat is None or not lat.exists():
        lat = HERE / "assets" / "latency.json"
    if lat.exists():
        L = json.loads(lat.read_text())
        rows_l = "\n".join(f"| {r['variant']} | {1000*r['median_s']:.0f} | {1000*r['min_s']:.0f} | {1000*r['max_s']:.0f} |" for r in L["rows"])
        txt += f"""
@tab lat-gpu | widths=2.4,1.3,1.3,1.3 | align=lccc | size=9
Measured forward-pass time (milliseconds) of the generator variants on a ${L['patch']}^3$ patch, batch size one, with the BraTS configuration of the network, on the machine on which the thesis was prepared ({L['device']}, PyTorch {L['torch']}).
| Variant | Median | Minimum | Maximum |
| --- | --- | --- | --- |
{rows_l}
@end
"""
    return txt
