"""Discussion chapter: section generated from the results of the hard synthetic benchmark (dictionary ch4_hard.RES) and from
published values that were checked against the primary sources.  No BraTS number of this thesis is involved."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "experiments"))

# Published values (Dice, per cent) as stated in the primary sources; order ET, WT, TC unless noted.
MYR_VAL_ENS = (82.33, 91.00, 86.68)      # Myronenko 2018, validation, ensemble of 10 (BraTS 2018, 66 cases)
MYR_TEST = (76.64, 88.39, 81.54)          # Myronenko 2018, test (191 cases)
ISE18_VAL = (80.87, 91.26, 86.34)         # Isensee et al. 2018, validation ensemble
ISE18_TEST = (77.88, 87.81, 80.62)        # Isensee et al. 2018, test
ISE20_VAL = (79.89, 91.24, 85.06)         # Isensee et al. 2020, validation ensemble
ISE20_TEST = (82.03, 88.95, 85.06)        # Isensee et al. 2020, test
SEGAN_GAIN = (2.0, 5.0, 7.0)              # SegAN minus U-Net, BRATS 2015 test: ET 0.66-0.64, WT 0.85-0.80, TC 0.70-0.63


def _d(a, b):
    return tuple(round(y - x, 2) for x, y in zip(a, b))


def build(RES: dict) -> str:
    if not RES or "abl" not in RES:
        return "@section[d-hard] Evidence from the Hard Synthetic Benchmark\n\n~The hard benchmark was not available when this document was built.\n"
    from figstyle import C, plt, setup
    from ch4_hard import FIGDIR
    setup()
    FIGDIR.mkdir(exist_ok=True)
    tot, steps = RES["tot"], RES.get("steps", [])
    # ------------------------------------------------------------------ effect sizes in perspective
    pub = [("SegAN minus U-Net\\\\(BRATS 2015, test)", np.mean(np.abs(SEGAN_GAIN)), "adv"),
           ("Validation to test\\\\(Isensee 2018)", np.mean(np.abs(_d(ISE18_VAL, ISE18_TEST))), "gap"),
           ("Validation to test\\\\(Myronenko 2018)", np.mean(np.abs(_d(MYR_VAL_ENS, MYR_TEST))), "gap"),
           ("Validation to test\\\\(Isensee 2020)", np.mean(np.abs(_d(ISE20_VAL, ISE20_TEST))), "gap")]
    mine = [("A5 minus A0\\\\(hard benchmark)", 100 * abs(tot["diff"]), "mine")]
    if "seed" in RES:
        mine.append(("Seed to seed\\\\(hard benchmark)", 100 * RES["seed"]["mean_abs_diff"], "mine"))
    if "cv" in RES:
        mine.append(("Spread over folds\\\\(hard benchmark)", 100 * RES["cv"]["spread"], "mine"))
    items = pub + mine
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ys = np.arange(len(items))[::-1]
    colmap = {"adv": C["orange"], "gap": C["grey"], "mine": C["blue"]}
    hatch = {"adv": "//", "gap": "", "mine": ""}
    for y, (lab, v, k) in zip(ys, items):
        ax.barh(y, v, color=colmap[k], hatch=hatch[k], edgecolor="white", height=0.62)
        ax.text(v + 0.08, y, f"{v:.2f}", va="center", fontsize=8)
    ax.set_yticks(ys); ax.set_yticklabels([i[0].replace("\\\\", "\n") for i in items], fontsize=7.5)
    ax.set_xlabel("mean absolute difference in Dice (percentage points, mean over the three regions)")
    ax.barh([], [], color=C["orange"], hatch="//", label="published adversarial gain"); ax.barh([], [], color=C["grey"], label="published validation-to-test gap")
    ax.barh([], [], color=C["blue"], label="this thesis (synthetic)"); ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout(); fig.savefig(FIGDIR / "effects.png"); plt.close(fig)

    lit_rows = [
        ("SegAN minus U-Net, BRATS 2015 test", *[f"{x:+.1f}" for x in SEGAN_GAIN], "Adversarial loss versus the authors' own baseline"),
        ("Validation to test, ensemble, BraTS 2018", *[f"{x:+.2f}" for x in _d(MYR_VAL_ENS, MYR_TEST)], "Same method, different case sets"),
        ("Validation to test, ensemble, BraTS 2018", *[f"{x:+.2f}" for x in _d(ISE18_VAL, ISE18_TEST)], "Same method, different case sets"),
        ("Validation to test, ensemble, BraTS 2020", *[f"{x:+.2f}" for x in _d(ISE20_VAL, ISE20_TEST)], "Same method, different case sets"),
    ]
    a = RES["abl"]
    tr = RES["totr"]
    mine_rows = [("A5 minus A0, hard benchmark, test", *[f"{100 * tr[r]['diff']:+.2f}" for r in ("ET", "WT", "TC")], "Paired difference over the test phantoms")]
    if "seed" in RES:
        mine_rows.append(("Mean absolute difference between two seeds, hard benchmark", "--", "--", "--", f"{100 * RES['seed']['mean_abs_diff']:.2f} points, mean over the six variants"))
    if "cv" in RES:
        mine_rows.append(("Spread of the A5 Dice over three folds, hard benchmark", "--", "--", "--", f"{100 * RES['cv']['spread']:.2f} points"))
    rows = "\n".join("| " + " | ".join(r) + " |" for r in lit_rows + mine_rows)

    # ------------------------------------------------------------------ component ranking
    seed_ref = RES.get("seed", {}).get("mean_abs_diff")
    comp_rows = []
    names = {"A1 vs A0": "residual blocks", "A2 vs A1": "attention gates", "A3 vs A2": "modality-attentive fusion stem",
             "A4 vs A3": "Tversky loss", "A5 vs A4": "adversarial term"}
    reading = []
    for s in steps:
        nm = names.get(s["name"], s["name"])
        ratio = (abs(s["diff"]) / seed_ref) if seed_ref else float("nan")
        if s["lo"] > 0 and s["p_holm"] < 0.05:
            rd = "improves, beyond Holm-corrected chance"
        elif s["hi"] < 0 and s["p_holm"] < 0.05:
            rd = "harms, beyond Holm-corrected chance"
        elif s["lo"] > 0 or s["hi"] < 0:
            rd = "interval excludes zero, not after Holm correction"
        else:
            rd = "not distinguishable from zero"
        reading.append((nm, s, ratio, rd))
        comp_rows.append(f"| {nm} ({s['name']}) | {s['diff']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}] | {s['p_holm']:.3f} | {ratio:.1f} | {rd} |" if s["p_holm"] >= 0.001 else
                         f"| {nm} ({s['name']}) | {s['diff']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}] | <0.001 | {ratio:.1f} | {rd} |")
    n_imp = sum(1 for r in reading if r[3].startswith("improves"))
    n_harm = sum(1 for r in reading if r[3].startswith("harms"))
    n_nd = sum(1 for r in reading if r[3].startswith("not dist"))
    best = max(reading, key=lambda r: r[1]["diff"]) if reading else None
    worst = min(reading, key=lambda r: r[1]["diff"]) if reading else None

    # figure: component differences against the seed-to-seed band
    if steps:
        fig, ax = plt.subplots(figsize=(6.0, 2.8))
        for i, (nm, st, ratio, rd) in enumerate(reading):
            col = C["aqua"] if rd.startswith("improves") else (C["red"] if rd.startswith("harms") else C["grey"])
            ax.errorbar(st["diff"], i, xerr=[[st["diff"] - st["lo"]], [st["hi"] - st["diff"]]], fmt="o", color=col, capsize=3, ms=5)
        if seed_ref:
            ax.axvspan(-seed_ref, seed_ref, color=C["yellow"], alpha=0.25, label="mean seed-to-seed difference")
        ax.axvline(0, color=C["ink"], lw=0.8)
        ax.set_yticks(range(len(reading))); ax.set_yticklabels([r[0] for r in reading], fontsize=8); ax.invert_yaxis()
        ax.set_xlabel("$\\Delta$ mean Dice (later minus earlier variant)")
        if seed_ref:
            ax.legend(frameon=False, fontsize=8, loc="lower right")
        fig.tight_layout(); fig.savefig(FIGDIR / "comp_noise.png"); plt.close(fig)
    ver = RES.get("verdicts", {})
    vtxt = ", ".join(f"{h}: {v.lower()}" for h, v in ver.items())

    surp = []
    tv = RES.get("tv", {})
    t5, t8 = tv.get("(0.5, 0.5)"), tv.get("(0.2, 0.8)")
    if t5 and t8:
        surp.append(("Tversky weights $\\beta>\\alpha$ raise ET sensitivity", f"ET sensitivity {t5['sens_ET']:.3f} to {t8['sens_ET']:.3f}, precision {t5['prec_ET']:.3f} to {t8['prec_ET']:.3f} from $\\beta=0.5$ to $0.8$",
                     "Opposite direction; all settings over-segment ET (volume ratio near 2)", "Treat the derivation as a statement about the gradient, not about the trained operating point; re-measure on BraTS"))
    g = RES.get("gan", {})
    if "a5_full" in g and g["a5_full"].get("acc_real") is not None:
        surp.append(("The discriminator stays informative", f"accuracy {100*g['a5_full']['acc_real']:.0f}% real, {100*g['a5_full']['acc_fake']:.0f}% generated; Dice range over five settings {100*(max(v['dice'] for k_, v in g.items() if k_ != 'a4_tversky') - min(v['dice'] for k_, v in g.items() if k_ != 'a4_tversky')):.2f} points",
                     "Near-perfect separation; adversarial term neutral", "Monitor $D$ accuracy on BraTS; weaken the discriminator or the smoothing before drawing conclusions on the adversarial gain"))
    surp.append(("The fusion stem selects modalities", "weights uniform (1/4) for all phantoms of both grades (@tab:hd-maf)", "No selection; stem acts as fixed averaging", "Test a variant that pools the descriptor before the instance normalisation"))
    gr = RES.get("grade", {}).get("Predicted masks, logistic regression")
    if gr:
        surp.append(("Grade is recoverable from predicted masks", f"cross-validated AUC {gr['cv']}, test AUC {gr['test']}", "Cross-validated and test estimates disagree; ground-truth masks give AUC 1.0", "Report the cross-validated estimate; enlarge the test set"))
    surp_rows = "\n".join("| " + " | ".join(r) + " |" for r in surp)
    surp_txt = f"""@subsection[d-hard-s] Where the expectations of the design were not met

A pre-specified analysis is useful only if its negative outcomes are reported with the same care as its positive ones. @tab:d-surprise lists the four places at which the hard benchmark contradicted an expectation that the design chapter had derived or assumed.

@tab d-surprise | widths=1.5,2.0,1.6,2.0 | align=llll | size=8
Expectations of the design that the hard benchmark did not confirm.
| Expectation | Measured | Reading | Consequence |
| --- | --- | --- | --- |
{surp_rows}
@end

The first three are likely to be properties of the design rather than of the data. The Tversky derivation of Section @sec:losses concerns the gradient mass that false negatives and false positives contribute at a given network output; the trained operating point also depends on the Dice and cross-entropy terms and on the deep supervision, so that a loss that weights false negatives more strongly can, in a trained network, still end at a more conservative operating point. The uniform fusion weights follow from the construction of the stem: the pooled descriptor is computed after an instance normalisation, which equalises the channel statistics of the modalities and leaves the scoring network almost no signal; this is a hypothesis that agrees with the observation and that was not isolated by an experiment. The adversarial term is neutral, and the discriminator separates real from generated label maps almost perfectly, which is consistent with a generator that receives little useful feedback; the experiment does not prove this mechanism. The fourth item, the grading from predicted masks, depends on the small number of LGG-like phantoms in the test set and cannot be resolved with this sample. None of these four findings depends on BraTS, and none can be extrapolated to BraTS; they are listed because each of them identifies a check (operating point, fusion weights, discriminator accuracy, size of the minority class) that the BraTS analysis of the RUNBOOK includes (operating-point table, fusion weights, discriminator accuracy, grading with intervals).

"""
    txt = f"""@section[d-hard] Evidence from the Hard Synthetic Benchmark

The hard benchmark of Section @sec:hard is the only place in this thesis where all components of the model are compared with paired statistics on a task that is not saturated. This section reads that evidence against the pre-specified hypotheses and against the effect sizes that are reported in the literature, and states what the evidence does and does not transfer to BraTS.

@subsection[d-hard-h] Reading the components

@tab:d-comp ranks the five components by the paired difference of the mean Dice that they produce when added to the preceding variant. The last but one column relates the difference to the seed-to-seed difference of Section @sec:hd-seed: a ratio below one means that the difference is smaller than the variation that two seeds of the same variant already show.

@tab d-comp | widths=1.9,1.9,0.7,0.7,1.9 | align=lcccl | size=8
Components of the model ranked in the order of the ablation chain: paired difference of the mean Dice on the hard benchmark (later minus earlier variant, 95% bootstrap interval), Holm-corrected Wilcoxon p, ratio of the absolute difference to the mean absolute seed-to-seed difference, and reading.
| Component (comparison) | $\\Delta$ Dice [95% CI] | p (Holm) | Ratio to seed difference | Reading |
| --- | --- | --- | --- | --- |
{chr(10).join(comp_rows)}
@end

@fig d-comp | file={FIGDIR}/comp_noise.png | width=5.8
Paired difference of the mean Dice caused by each component (point: mean, bar: 95% bootstrap interval; green: above zero, red: below zero, grey: contains zero). The shaded band is the mean absolute difference between two seeds of the same variant.
@end

Of the {len(reading)} components, {n_imp} improve the mean Dice beyond chance, {n_harm} reduce it and {n_nd} are indistinguishable from zero. The largest positive difference is that of the {best[0] if best else '--'} ({best[1]['diff'] if best else float('nan'):+.3f}), the smallest that of the {worst[0] if worst else '--'} ({worst[1]['diff'] if worst else float('nan'):+.3f}). The verdicts of the pre-specified rules (@tab:hd-hyp) are: {vtxt}. Two cautions apply to any reading of this kind. First, the components are added cumulatively, so that a difference is the effect of a component *in the presence of the preceding ones*; an attention gate, for example, may be redundant after residual blocks and useful without them. Second, the benchmark is small and the budget short, so that components with a slow effect on training (the adversarial term, which acts only in the last epochs) are structurally disadvantaged.

{surp_txt}@subsection[d-hard-e] Effect sizes in perspective

The reliability of a reported improvement depends on its size relative to the other sources of variation. @tab:d-eff collects, for the studies of Chapter 2 whose numbers were checked against the primary sources, the differences that they report between an adversarially trained and a baseline network and the differences between the validation and the test results of the same method, and sets them beside the corresponding quantities measured on the hard benchmark. @fig:d-eff shows the mean absolute value over the three regions.

@tab d-eff | widths=2.9,0.6,0.6,0.6,1.8 | align=lcccl | size=8
Differences in Dice (percentage points) reported in the literature and measured in this thesis. Columns ET, WT, TC as in @tab:pubres.
| Quantity | ET | WT | TC | Remark |
| --- | --- | --- | --- | --- |
{rows}
@end

@fig d-eff | file={FIGDIR}/effects.png | width=6.2
Mean absolute difference in Dice (percentage points, mean over ET, WT, TC) of published adversarial gains, published validation-to-test gaps and the effects measured on the hard benchmark.
@end

The published adversarial gain (SegAN, {np.mean(np.abs(SEGAN_GAIN)):.1f} points on average) is of the same order as the validation-to-test gaps of the same challenge entries ({np.mean(np.abs(_d(ISE18_VAL, ISE18_TEST))):.1f} and {np.mean(np.abs(_d(MYR_VAL_ENS, MYR_TEST))):.1f} points in 2018). A gain of this size, obtained on one test set against one baseline, therefore cannot be distinguished from the variation that a change of case set produces, which is the observation on which the evaluation protocol of this thesis is built. On the hard benchmark the effect of the full model against the plain network is {100 * abs(tot['diff']):.2f} points, the seed-to-seed difference is {100 * RES['seed']['mean_abs_diff'] if 'seed' in RES else float('nan'):.2f} points and the spread over three folds {100 * RES['cv']['spread'] if 'cv' in RES else float('nan'):.2f} points.

@subsection[d-hard-t] What transfers to BraTS

@tab:d-transfer separates the findings of the hard benchmark by their expected transferability. The classification is a judgement about mechanisms, not a result: only the BraTS runs can establish whether a finding holds on patient data.

@tab d-transfer | widths=2.5,2.0,2.0 | align=lll | size=8.5
Findings of the hard benchmark and their expected transferability to BraTS.
| Finding | Mechanism | Expected to transfer? |
| --- | --- | --- |
| The analysis pipeline (paired statistics, Holm correction, seed and fold replication, grading, calibration) runs end to end and reproduces itself from the output files | Software; independent of the data | Yes, by construction |
| ET is the hardest region and the most size-dependent | Small structures, boundary error relative to size | Yes: the same ordering is found in every BraTS entry of @tab:pubres |
| The ranking and sign of component effects | Depends on the signal-to-noise ratio, the size of the data set and the training budget | Not expected: must be re-measured on BraTS |
| The size of the adversarial effect and the discriminator accuracy | Depends on the realism of the label maps and on the generator capacity | Not expected; the diagnostics should be monitored on BraTS |
| The direction of the recall--precision shift produced by the asymmetric Tversky weights | Depends on the interaction of the loss with the other terms; measured opposite to the derivation here | Not expected; must be re-measured on BraTS |
| The seed-to-seed and fold-to-fold variation | Depends on data-set size and difficulty | The order of magnitude may differ; it must be measured on BraTS |
@end
"""
    return txt


def _ci(q, nd=3):
    return f"{q['diff']:+.{nd}f} [{q['lo']:+.{nd}f}, {q['hi']:+.{nd}f}]"


def abstract_sentence(RES: dict) -> str:
    """Sentence(s) of the abstract about the hard benchmark; every number comes from RES."""
    if not RES or "tot" not in RES:
        return "A harder synthetic benchmark was not available when this document was built."
    st = {x["name"]: x for x in RES["steps"]}
    t, cv, sd = RES["tot"], RES["cv"], RES["seed"]
    return (f"Because the phantom study is saturated, a harder synthetic benchmark with {RES['n_dev'] + RES['n_test']} volumes carries the controlled comparison: "
            f"the full model improves the mean Dice over the plain 3D U-Net by {t['diff']:.3f} [{t['lo']:.3f}, {t['hi']:.3f}] on the test phantoms of one fold, "
            f"but by {cv['oof']['diff']:.3f} [{cv['oof']['lo']:.3f}, {cv['oof']['hi']:.3f}] out of fold, which is of the order of the seed-to-seed difference of {sd['mean_abs_diff']:.3f}; "
            f"the fusion stem lowers the Dice by {abs(st['A3 vs A2']['diff']):.3f}, the Tversky term raises it by {st['A4 vs A3']['diff']:.3f}, and the adversarial term has no measurable effect, "
            f"the discriminator separating real from generated label maps almost perfectly. None of the evaluable pre-specified hypotheses is supported on this benchmark, which is synthetic and is not evidence about patients.")


def conclusion_section(RES: dict) -> str:
    """Section of Chapter 6 on the findings of the hard benchmark, generated from RES."""
    if not RES or "tot" not in RES:
        return ""
    st = {x["name"]: x for x in RES["steps"]}
    t, cv, sd = RES["tot"], RES["cv"], RES["seed"]
    tv = RES["tv"]; t5, t8 = tv["(0.5, 0.5)"], tv["(0.2, 0.8)"]
    g = RES["gan"]; gr = RES["grade"]["Predicted masks, logistic regression"]
    ver = RES["verdicts"]
    return f"""@section[c-hard] Findings of the Hard Synthetic Benchmark

The hard benchmark of Section @sec:hard produced the only numerical comparison of the components in this thesis, and its outcome is mixed. The numbers are those of the synthetic task and say nothing about the performance on patients; they say what the analysis can resolve and where the design did not behave as derived.

*Components.* Added one by one, the residual blocks ({st['A1 vs A0']['diff']:+.3f}) and the attention gates ({st['A2 vs A1']['diff']:+.3f}) gave small positive differences that do not survive the Holm correction; the modality-attentive fusion stem lowered the mean Dice by {abs(st['A3 vs A2']['diff']):.3f}; the Tversky term raised it by {st['A4 vs A3']['diff']:+.3f}; the adversarial term changed it by {RES['a5a4']['diff']:+.3f} [{RES['a5a4']['lo']:+.3f}, {RES['a5a4']['hi']:+.3f}]. The full model exceeds the plain network by {t['diff']:.3f} on the test phantoms of fold 0, but by {cv['oof']['diff']:.3f} [{cv['oof']['lo']:.3f}, {cv['oof']['hi']:.3f}] out of fold, with differences per fold of {', '.join(f'{x:+.3f}' for x in cv['per_fold'])}: the benefit is not stable, and it is comparable with the seed-to-seed difference of {sd['mean_abs_diff']:.3f}.

*Operating point.* The Tversky weights moved the enhancing-tumour operating point opposite to the derivation (sensitivity {t5['sens_ET']:.3f} to {t8['sens_ET']:.3f}, precision {t5['prec_ET']:.3f} to {t8['prec_ET']:.3f} from $\\beta=0.5$ to $0.8$), and all networks over-segmented the enhancing tumour by a factor of about two.

*Adversarial stage and fusion stem.* The discriminator classified {100 * g['a5_full']['acc_real']:.0f}% of the real and {100 * g['a5_full']['acc_fake']:.0f}% of the generated label maps correctly, and the fusion stem assigned uniform weights to the four modalities.

*Grading.* From predicted masks the cross-validated AUC was {gr['cv']} and the test AUC {gr['test']}, against 1.0 from the reference masks.

With the decision rules fixed in advance, the verdicts were {', '.join(f'{h}: {v.lower()}' for h, v in ver.items())}. These outcomes are reported without adjustment of the rules. They do not weaken the case for the protocol: a protocol that cannot return a negative answer is not a protocol, and the benchmark showed that this one can. They do show that each component of the proposed model needs to be justified on BraTS, with several seeds, before any claim is made about it.

"""
