"""Chapter 4, sections 4.1-4.2: analytical and numerical verification (numbers read from verify.json)."""
from __future__ import annotations

FIG = str(__import__("pathlib").Path(__file__).resolve().parent / "assets" / "figs")


def _rows(rows, fmt):
    return "\n".join("| " + " | ".join(fmt(r)) + " |" for r in rows)


def build(V: dict, ledger: list[tuple[str, str, str]], tests: dict) -> str:
    out = []
    add = out.append

    # ------------------------------------------------------------------ 4.1
    add("@chapter[res] Results\n")
    add("@section[r-overview] Overview of the Evidence\n")
    add("""This chapter presents three kinds of evidence, in increasing proximity to the clinical question and in decreasing order of what could be completed within the computing environment in which the thesis was prepared. @tab:ledger lists them and states, for each, whether the corresponding numbers are complete in this version of the document.

@tab ledger | widths=1.7,2.9,1.9 | align=lll | size=9
Evidence ledger of the results chapter.
""" + "| Kind of evidence | Content | Status in this version |\n| --- | --- | --- |\n" + _rows(ledger, lambda r: [r[0], r[1], r[2]]) + """
@end

The verification experiments (Sections @sec:verif and @sec:phantom) test whether the implementation behaves as the theory of Chapter 3 predicts and whether the experimental pipeline is free of defects. They use no patient data. The BraTS results (Section @sec:brats-res) are the evidence on which the hypotheses of Chapter 1 are decided.

@note | title=What the verification experiments can and cannot show
Synthetic phantoms are generated with known ground truth and with contrast, shape and noise models chosen by the author. They are easier than real MRI in some respects (no anatomy, no acquisition artefacts, no inter-rater ambiguity) and unrepresentative in others. A method that performs well on phantoms may fail on patients, and the ranking of the model variants on phantoms may differ from the ranking on patients. Accordingly, the phantom results of this chapter are used only to confirm that the code works and that the diagnostics behave sensibly; *no performance claim about brain tumours is derived from them.*
@end
""")

    # ------------------------------------------------------------------ 4.2
    add("@section[verif] Analytical and Numerical Verification\n")
    add("All results in this section are computed by the script *experiments/verify.py* of the code base, by exact calculation, automatic differentiation or Monte-Carlo simulation. None of them requires a trained model.\n")

    # V1
    c = V["v1_constraints"]
    add(f"""@subsection[v1-s] Dice and IoU: feasibility region

Equation @eq:dice-iou states that the mean IoU of a set of cases lies between $\\bar d/(2-\\bar d)$ and $\\bar d$. @fig:v1 (a) shows the convexity of $j(d)=d/(2-d)$ that underlies the lower bound, and (b) the feasible region. A Monte-Carlo check was made with beta-distributed per-case Dice scores (400,000 draws per setting): in all {len(V['v1_dice_iou'])} settings the simulated mean IoU lay inside the bounds, and the gap between the mean IoU and the IoU of the mean Dice (the Jensen gap) grew with the dispersion of the Dice scores, from {min(r['jensen_gap'] for r in V['v1_dice_iou']):.4f} to {max(r['jensen_gap'] for r in V['v1_dice_iou']):.4f}.

@fig v1 | file={FIG}/v1_dice_iou.png | width=6.2
Relation between Dice and IoU. (a) IoU as a convex function of Dice; the chord shows that the mean of two cases has a higher IoU than the IoU of their mean Dice. (b) Feasible region of the pair of mean Dice and mean IoU; the red marker is the pair reported in the preliminary study.
@end

The bound is a practical consistency test. For a mean Dice of 0.99, the mean IoU cannot be lower than {c['iou_floor_for_dice_0.99']:.4f}; conversely, a mean IoU of 0.97 implies a mean Dice not larger than {c['max_dice_for_iou_0.97']:.4f}. The pair (Dice, IoU) = (0.99, 0.97) reported in the preliminary study therefore cannot be the pair of means over the same set of cases. The voxel-wise precision (97.56%) and recall (98.37%) reported there give, by the harmonic mean, an F-score, which for binary masks is the Dice coefficient, of {c['f1_from_P0.9756_R0.9837']:.4f}, which is also below 0.99. These observations do not indicate which of the reported values is in error; they illustrate why the protocol of this thesis computes all metrics from one confusion matrix per patient and region, and why consistency checks of this kind are applied before any number is reported.
""")

    # V2
    g = V["v2_gradient_mass"]

    def share(r, k):
        return r[k]["tumour_mass"] / (r[k]["tumour_mass"] + r[k]["bg_mass"])

    sel = [r for r in g if r["frac"] in (0.002, 0.01, 0.05, 0.2)]
    rows = _rows(sel, lambda r: [f"{100*r['frac']:.1f}%", f"{100*share(r,'bce'):.1f}", f"{100*share(r,'dice'):.1f}", f"{100*share(r,'tversky'):.1f}",
                                 f"{r['bce']['tumour_mass']:.4f}", f"{r['dice']['tumour_mass']:.4f}", f"{r['tversky']['tumour_mass']:.4f}"])
    r0, r1 = g[0], g[-1]
    add(f"""@subsection[v2] Gradient mass under class imbalance

To verify the analysis of Section @sec:losses, a volume of 10,000 voxels was simulated with a tumour fraction between 0.2% and 40%, with predicted probabilities of 0.30 on tumour voxels and 0.05 on background voxels (a moderately trained network). The gradients $|\\partial\\mathcal{{L}}/\\partial z_i|$ with respect to the logits were obtained by automatic differentiation for the cross-entropy, the soft Dice loss and the Tversky loss ($\\alpha=0.3$, $\\beta=0.7$), and summed over tumour and over background voxels. @tab:v2 and @fig:v2 give the result.

@fig v2 | file={FIG}/v2_gradient_mass.png | width=6.2
Gradient mass with respect to the logits as a function of the tumour fraction, for the cross-entropy, the Dice loss and the Tversky loss. (a) Absolute mass on tumour voxels (logarithmic axes). (b) Share of the total gradient mass that falls on tumour voxels.
@end

@tab v2 | widths=0.9,0.8,0.8,0.8,0.95,0.95,1.0 | align=lcccccc | size=8.5
Share of the gradient mass on tumour voxels (columns 2 to 4, in %) and absolute gradient mass on tumour voxels (columns 5 to 7) for the three losses.
| Tumour fraction | BCE share | Dice share | Tversky share | BCE mass | Dice mass | Tversky mass |
| --- | --- | --- | --- | --- | --- | --- |
{rows}
@end

For the cross-entropy the tumour's share of the gradient is only {100*share(r0,'bce'):.1f}% at a tumour fraction of 0.2% and its absolute mass is proportional to the tumour fraction (it changes by a factor of {r1['bce']['tumour_mass']/r0['bce']['tumour_mass']:.0f} between the smallest and the largest fraction, exactly the change of the fraction, 200), as predicted by the analysis. For the Dice loss the share at the same fraction is {100*share(r0,'dice'):.1f}%, and the absolute mass changes only by a factor of {r1['dice']['tumour_mass']/r0['dice']['tumour_mass']:.0f}. The Tversky loss behaves like the Dice loss with a slightly larger share on the tumour ({100*share(r0,'tversky'):.1f}% at 0.2%) and a smaller share on the background at large fractions, which is the effect of down-weighting false positives. The overlap losses are therefore, in the quantitative sense of the analysis, far less sensitive to class imbalance than the cross-entropy; the cross-entropy nevertheless remains useful in the hybrid loss, because its gradient stays bounded and informative when the overlap losses become noisy.
""")

    # V9
    v9 = V["v9"]
    add(f"""@subsection[v9s] Asymmetry of the Tversky loss

@fig:v9 shows the loss as a function of the numbers of false negatives and false positives for a region with {v9['TP']} true-positive voxels. The Dice loss is symmetric about the diagonal. The Tversky loss with $(\\alpha,\\beta)=(0.3,0.7)$ penalises false negatives more strongly: 200 missed voxels cost {v9['tv_at_fn200_fp0']:.4f}, whereas 200 spurious voxels cost {v9['tv_at_fn0_fp200']:.4f}, a ratio of {v9['tv_at_fn200_fp0']/v9['tv_at_fn0_fp200']:.2f}. For small error counts the ratio tends to the value $\\beta/\\alpha=2.33$ of @eq:cost-ratio; for the Dice loss, 200 missed or 200 spurious voxels both cost {v9['dice_at_fn200_fp0']:.4f}.

@fig v9 | file={FIG}/v9_dice_vs_tversky.png | width=6.0
Loss surfaces of the Dice loss (a) and the Tversky loss with $\\alpha=0.3$, $\\beta=0.7$ (b) as a function of false negatives and false positives for 1,000 true-positive voxels; the dashed line marks equal numbers of both errors.
@end
""")

    # V3 complexity & RF
    cx = V["v3_complexity"]
    rows = _rows(cx[:4], lambda r: [r["name"], f"{r['params']/1e6:.3f}", f"{r['gmacs']:.1f}", f"{r['gflops']:.1f}", f"{r['activation_GB_fp32']:.2f}"])
    d = cx[4]
    a0, a1, a2, a3 = cx[:4]
    rf_e, rf_m = V["v3_rf_early"], V["v3_rf_maf"]
    rows_rf = _rows(list(zip(rf_e, rf_m)), lambda t: [str(t[0]["stage"]), str(t[0]["stride"]), str(t[0]["rf"]), str(t[1]["rf"])])
    add(f"""@subsection[v3] Model complexity and receptive field

@tab:cx lists the exact number of trainable parameters, the multiply-accumulate operations (GMACs) and the memory of the activations for one forward pass on a $128^3$ patch (batch size one, four channels), for the four generator variants. The counts were obtained with the hooks described in Section @sec:arch on the meta device, and the discriminator ({d['params']/1e6:.3f} million parameters) is listed for completeness.

@tab cx | widths=2.3,0.95,0.9,0.9,1.4 | align=lcccc | size=9
Complexity of the generator variants for one $128^3$ patch (forward pass). Activations: stored outputs of convolution, normalisation and activation layers in single precision.
| Variant | Parameters (M) | GMACs | GFLOPs | Activations (GB) |
| --- | --- | --- | --- | --- |
{rows}
@end

Residual blocks add {a1['params']-a0['params']:,} parameters ({100*(a1['params']/a0['params']-1):.1f}%) and {a1['gmacs']-a0['gmacs']:.1f} GMACs ({100*(a1['gmacs']/a0['gmacs']-1):.1f}%) through the $1\\times1\\times1$ projections and the additional stored activations. The attention gates add {a2['params']-a1['params']:,} parameters and {a2['gmacs']-a1['gmacs']:.1f} GMACs ({100*(a2['gmacs']/a1['gmacs']-1):.1f}%): they are cheap because they operate with $C/2$ channels and use $1\\times1\\times1$ kernels. The modality-attentive fusion stem adds only {a3['params']-a2['params']:,} parameters but {a3['gmacs']-a2['gmacs']:.1f} GMACs ({100*(a3['gmacs']/a2['gmacs']-1):.1f}%) and {a3['activation_GB_fp32']-a2['activation_GB_fp32']:.2f} GB of activations ({100*(a3['activation_GB_fp32']/a2['activation_GB_fp32']-1):.0f}%), because it is evaluated at full resolution (four per-modality $3\\times3\\times3$ stems with 16 channels, followed by a mixing convolution over 80 channels) and because the first encoder block now receives 32 instead of four input channels. A model with the full complement of components therefore needs {a3['activation_GB_fp32']/a0['activation_GB_fp32']:.1f} times the activation memory of the plain network at the same patch size, which is the practical cost to be weighed against any accuracy gain in the ablation.

@tab rf | widths=1.2,1.3,1.9,2.1 | align=cccc | size=9
Theoretical receptive field (voxels per axis) of the encoder after each level, from the recursion of @eq:rf.
| Level | Cumulative stride | Receptive field, early fusion | Receptive field, MAF stem |
| --- | --- | --- | --- |
{rows_rf}
@end

@tab:rf gives the receptive fields. The values confirm the statement of Chapter 3: at the bottleneck the encoder reaches {rf_e[-1]['rf']} voxels ({rf_m[-1]['rf']} with the MAF stem), i.e. less than the $128$-voxel patch, so that the patch size is not wasted on a receptive field that could never be exploited.
""")

    if V.get("latency"):
        L = V["latency"]
        rows_l = "\n".join(f"| {r['variant']} | {1000*r['median_s']:.0f} | {1000*r['min_s']:.0f} | {1000*r['max_s']:.0f} |" for r in L["rows"])
        base_t, last_t = L["rows"][0]["median_s"], L["rows"][-1]["median_s"]
        add(f"""To relate the operation counts to time, the forward pass of each variant (full width, five levels, no deep supervision) was timed on a ${L['patch']}^3$ patch with batch size one on the machine on which the thesis was prepared ({L['device']}, PyTorch {L['torch']}; median of {L['repeats']} repetitions after one warm-up pass, @tab:lat). The full model took {last_t/base_t:.2f} times as long as the plain network, against a ratio of {a3['gmacs']/a0['gmacs']:.2f} for the multiply-accumulate counts of @tab:cx. These timings describe that machine only; the inference time on the GPU used for the BraTS experiments is recorded with the test results (Section @sec:brats-res).

@tab lat | widths=2.4,1.3,1.3,1.3 | align=lccc | size=9
Measured forward-pass time (milliseconds) of the generator variants on a ${L['patch']}^3$ patch, batch size one, on the preparation machine ({L['device']}).
| Variant | Median | Minimum | Maximum |
| --- | --- | --- | --- |
{rows_l}
@end
""")

    # V7 windows
    w = V["v7_windows"]

    def get(shape0, patch, ov):
        return next(x["windows"] for x in w if x["shape"][0] == shape0 and x["patch"] == patch and abs(x["overlap"] - ov) < 1e-9)

    rows = []
    for shp, lab in ((240, "240 x 240 x 155 (full BraTS)"), (160, "160 x 192 x 144 (typical brain crop)")):
        for p in (96, 128):
            rows.append([lab, str(p)] + [str(get(shp, p, ov)) for ov in (0.0, 0.25, 0.5, 0.75)])
    add(f"""@subsection[v7s] Cost of sliding-window inference

@tab:win gives the number of windows needed to cover a volume for different patch sizes and overlaps, from the exact enumeration used by the implementation (including the final window flush with the border). With the default patch size of 128 and 50% overlap, a typical cropped brain of $160\\times192\\times144$ voxels needs {get(160,128,0.5)} windows and the uncropped BraTS volume {get(240,128,0.5)}. Increasing the overlap to 75% raises the cost to {get(160,128,0.75)} and {get(240,128,0.75)} windows respectively, i.e. by a factor of {get(160,128,0.75)/get(160,128,0.5):.1f} and {get(240,128,0.75)/get(240,128,0.5):.1f}; with test-time mirroring the cost is multiplied by a further factor of eight.

@tab win | widths=2.6,0.8,0.8,0.8,0.8,0.8 | align=lccccc | size=9
Number of windows for sliding-window inference as a function of patch size and overlap.
| Volume (voxels) | Patch | 0% | 25% | 50% | 75% |
| --- | --- | --- | --- | --- | --- |
{chr(10).join('| ' + ' | '.join(r) + ' |' for r in rows)}
@end
""")

    # V5 slice dependence
    s5 = V["v5_slice_dependence"]
    add(f"""@subsection[v5s] Dependence between slices

To illustrate the dependence structure that makes slice-level partitioning and slice-level statistics unsafe, the tumour area per axial slice was computed for the {s5['n_patients']} synthetic phantoms of the verification study ({s5['n_slices']:,} tumour-containing slices, on average {s5['mean_tumour_slices']} per phantom). The correlation of the tumour area between adjacent slices is {s5['adjacent_slice_corr']:.3f}, and the intra-patient correlation coefficient of the one-way random-effects model is $\\rho={s5['icc1']:.3f}$. With $m={s5['mean_tumour_slices']}$ slices per patient, the Kish design effect is $1+(m-1)\\rho={s5['design_effect']:.2f}$, so that the {s5['n_slices']:,} slices carry the information of about {s5['effective_n']:.0f} independent observations for this quantity. For real patients the dependence of segmentation *errors* between neighbouring slices is at least as strong, because errors follow the boundary of the same lesion. The patient, not the slice, is therefore the unit of partitioning and of analysis in this thesis.
""")

    # V6 / V8
    cv = V["v6_coverage"]
    rows = _rows(cv, lambda r: [str(r["n"]), f"{100*r['cov_bootstrap']:.1f}", f"{100*r['cov_t']:.1f}", f"{r['width_bootstrap']:.3f}", f"{r['width_t']:.3f}"])
    m = V["v8_mde"]
    rows_m = _rows([x for x in m if x["n"] in (20, 74, 295)],
                   lambda r: [f"{r['sd_diff']:.2f}", str(r["n"]), f"{r['mde']:.4f}"])
    n74 = next(x for x in cv if x["n"] == 74)
    add(f"""@subsection[v6s] Coverage of the confidence intervals and detectable differences

The percentile bootstrap interval is only approximately valid for small samples and skewed distributions, such as Dice scores that pile up near one. To check its coverage, 400 samples of size $n$ were drawn from a beta distribution with mean 0.88 and standard deviation 0.10 (skewed to the left like segmentation scores), and the fraction of 95% intervals that contained the true mean was recorded for the percentile bootstrap (800 resamples) and for the Student-$t$ interval (@tab:cov, @fig:v6).

@tab cov | widths=1.0,1.4,1.4,1.4,1.4 | align=ccccc | size=9
Empirical coverage (%) of nominal 95% intervals and mean interval width for the bootstrap and the $t$ interval.
| n | Coverage, bootstrap | Coverage, $t$ | Width, bootstrap | Width, $t$ |
| --- | --- | --- | --- | --- |
{rows}
@end

@fig v6 | file={FIG}/v6_ci_coverage.png | width=6.0
Coverage of nominal 95% intervals as a function of the number of test cases (a) and width of the bootstrap interval (b), for skewed Dice-like data.
@end

Both intervals under-cover when $n\\le40$ (the bootstrap more so, {100*cv[0]['cov_bootstrap']:.1f}% at $n=10$) and reach the nominal level from about $n=74$ onwards ({100*n74['cov_bootstrap']:.1f}% for the bootstrap and {100*n74['cov_t']:.1f}% for the $t$ interval at the size of the planned test set). The intervals computed on the 74-patient test set can thus be read at face value, whereas intervals computed for subgroups of ten or twenty patients (for example, the LGG patients of the test set) are optimistic and are labelled as such. The half-width of the interval at $n=74$ is about {n74['width_bootstrap']/2:.3f} Dice units.

@tab:mde gives the minimum detectable paired difference of @eq:mde. For the test set of 74 patients and a standard deviation of the paired differences of 0.05, differences below {next(x['mde'] for x in m if x['n']==74 and abs(x['sd_diff']-0.05)<1e-9):.3f} Dice units cannot be reliably detected; the same threshold is {next(x['mde'] for x in m if x['n']==20 and abs(x['sd_diff']-0.05)<1e-9):.3f} for 20 patients, which quantifies the limited power of the phantom study of Section @sec:phantom and of subgroup analyses.

@tab mde | widths=1.9,1.2,2.0 | align=ccc | size=9
Minimum detectable paired difference (two-sided level 0.05, power 0.8).
| Standard deviation of differences | n | Minimum detectable difference |
| --- | --- | --- |
{rows_m}
@end
""")

    # tests
    add(f"""@subsection[tests-s] Software verification

The automated test suite of the code base contains {tests['n']} tests, of which {tests['passed']} passed in the final run ({tests['seconds']:.0f} s on the CPU of the preparation environment). They cover the metrics against hand-computed values (including the empty-mask conventions, the penalty of 373.13 mm and the scaling of HD95 with the voxel spacing), the gradients of the Dice and Tversky losses against finite differences, the Tversky loss reducing to the Dice loss for $\\alpha=\\beta=\\tfrac12$ and penalising missed voxels more than spurious ones for $\\beta>\\alpha$, the output shapes and the deep-supervision interface of the networks, the range of the attention coefficients and the normalisation of the fusion weights, the exact reconstruction of a constant model by the sliding window on a volume that is not a multiple of the patch, the absence of patient overlap between partitions and the detection of a duplicated patient, the in-brain standardisation, and the Holm correction, the AUC and the bootstrap interval. In addition, an end-to-end smoke test runs the complete pipeline on a miniature synthetic data set: conversion of NIfTI files, preprocessing, patient-level split, training of the baseline and of the adversarial model, evaluation, aggregation, and grade classification.
""")
    return "\n".join(out)
