"""Appendices A-D of the thesis (configuration tables are generated from the code's own DEFAULTS)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from bratsgan.config import DEFAULTS as D  # noqa: E402


def _src(path, a, b, strip_decorators=True):
    ls = (ROOT / path).read_text().splitlines()[a - 1:b]
    return "\n".join(l for l in ls if not (strip_decorators and l.lstrip().startswith("@")))


def _tab(rows):
    return "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)


def appA() -> str:
    p = D["patch"][0]
    cfg = [
        ("Architecture", "Base width / depth / maximum width", f"{D['base']} / {D['depth']} / {D['max_ch']}"),
        ("", "Normalisation, activation", f"{D['norm']} normalisation, LeakyReLU(0.01)"),
        ("", "Deep supervision", f"{D['deep_supervision']} (heads at 1/1, 1/2, 1/4; weights 4/7, 2/7, 1/7)"),
        ("", "Dropout", D["dropout"]),
        ("Data", "Patch size / batch size", f"${p}^3$ / {D['batch_size']}"),
        ("", "Foreground-biased sampling probability", D["fg_prob"]),
        ("Schedule", "Iterations per epoch", D["iters_per_epoch"]),
        ("", "Segmentation / adversarial epochs", f"{D['epochs_seg']} / {D['epochs_adv']}"),
        ("", "Generator updates in total", f"{(D['epochs_seg'] + D['epochs_adv']) * D['iters_per_epoch']:,}"),
        ("", "Validation every (epochs), overlap", f"{D['val_every']}, {D['val_overlap']}"),
        ("Optimisation", "Generator: AdamW, learning rate, weight decay", f"{D['lr']}, {D['weight_decay']}"),
        ("", "Learning-rate schedule", f"polynomial, power {D['poly_power']}"),
        ("", "Gradient-norm clipping", D["grad_clip"]),
        ("", "Mixed precision (GPU)", f"bfloat16 autocast: {D['amp']}"),
        ("Loss", "Weights Dice / Tversky / BCE", f"{D['w_dice']} / {D['w_tv']} / {D['w_bce']}"),
        ("", "Tversky $\\alpha$ (false positives) / $\\beta$ (false negatives)", f"{D['tv_alpha']} / {D['tv_beta']}"),
        ("", "Focal exponent $\\gamma$", D["tv_gamma"]),
        ("Adversarial", "Discriminator base width", D["ndf"]),
        ("", "Discriminator optimiser", f"Adam, learning rate {D['d_lr']}, betas {tuple(D['d_betas'])}"),
        ("", "Adversarial weight $\\lambda_{\\max}$, ramp (epochs)", f"{D['adv_weight']}, {D['adv_ramp_epochs']}"),
        ("", "Objective", f"{D['adv_loss']} (alternative: hinge)"),
        ("", "Real-label smoothing, instance noise $\\sigma$", f"{D['real_smooth']}, {D['d_noise']}"),
        ("", "Discriminator steps per generator step", D["d_steps"]),
        ("Inference", "Window overlap, mirroring, threshold", f"{D['final_overlap']}, {D['final_tta']}, 0.5"),
        ("", "Hierarchy projection ET $\\subset$ TC $\\subset$ WT", D["hierarchy"]),
        ("", "Seed", D["seed"]),
    ]
    rows = _tab(cfg)
    var = [
        ("A0", "plain 3D U-Net", "no", "no", "early", "0", "no"),
        ("A1", "+ residual", "yes", "no", "early", "0", "no"),
        ("A2", "+ attention", "yes", "yes", "early", "0", "no"),
        ("A3", "+ MAF", "yes", "yes", "MAF", "0", "no"),
        ("A4", "+ Tversky", "yes", "yes", "MAF", "1", "no"),
        ("A5", "+ adversarial", "yes", "yes", "MAF", "1", "yes"),
        ("t_dice_eq", "A4 with $\\alpha=\\beta=0.5$", "yes", "yes", "MAF", "1", "no"),
        ("t_fn_heavy", "A4 with $\\alpha=0.2$, $\\beta=0.8$", "yes", "yes", "MAF", "1", "no"),
        ("g_lam001", "A5 with $\\lambda_{\\max}=0.01$", "yes", "yes", "MAF", "1", "yes"),
        ("g_lam020", "A5 with $\\lambda_{\\max}=0.20$", "yes", "yes", "MAF", "1", "yes"),
        ("g_hinge", "A5 with hinge objective", "yes", "yes", "MAF", "1", "yes"),
        ("g_nosmooth", "A5 without smoothing and noise", "yes", "yes", "MAF", "1", "yes"),
    ]
    mods = [("`data.py`", "BraTS discovery, preprocessing cache, patient-level grade-stratified splits, leakage assertion, foreground-biased patch sampler"),
            ("`augment.py`", "GPU augmentation: flips, 90-degree rotations, intensity scaling and shift, gamma, Gaussian noise and blur, bias field"),
            ("`models.py`", "3D U-Net family with optional residual blocks, additive attention gates, MAF stem, deep supervision; conditional 3D PatchGAN"),
            ("`losses.py`", "Soft Dice, Tversky, BCE with deep supervision; LSGAN and hinge objectives; adversarial ramp"),
            ("`train.py`", "Two-stage training loop, checkpointing by validation Dice, final evaluation of the best checkpoint"),
            ("`infer.py`", "Gaussian-weighted sliding-window inference, optional mirroring, region thresholding"),
            ("`metrics.py`", "Dice, IoU, precision, sensitivity, specificity, HD95 with the BraTS conventions for empty masks"),
            ("`evaluate.py`", "Per-patient evaluation of a model on a list of cases; command-line entry point"),
            ("`features.py`", "101 segmentation-derived features for grade classification"),
            ("`stats.py`", "Bootstrap intervals, Wilcoxon test, Holm correction, Cliff's delta, Cohen's $d_z$, AUC, calibration error"),
            ("`complexity.py`", "Exact parameter, MAC and activation counts by hooks on the meta device"),
            ("`synthetic.py`", "Synthetic phantoms and a writer for the BraTS folder layout")]
    return f"""@appendix[impl-app] Implementation Details

@section[a-env] Software Environment

The verification experiments of Chapter 4 were run with Python 3.13, PyTorch 2.14, NumPy 2.5, SciPy 1.18, pandas 3.0 and scikit-learn 1.9 on a two-core CPU without a graphics processor. The BraTS experiments require a CUDA-capable GPU with enough memory for two $128^3$ patches with four channels (the activation memory of one forward pass of the full model is given in @tab:cx); the package versions used for a given experiment are stored with its results through the configuration file and the environment listing of the RUNBOOK. All random number generators (Python, NumPy and PyTorch) are seeded from the configuration; bit-wise determinism on GPUs is not guaranteed unless the option *deterministic* is set, at a cost in speed.

@section[a-cfg] Configuration Reference

@tab cfg | widths=1.1,2.9,2.5 | align=lll | size=8
Default configuration of the full model (the file *bratsgan/config.py*). Every experiment is a JSON file that overrides a subset of these values.
| Group | Parameter | Value |
| --- | --- | --- |
{rows}
@end

The twelve experiment files of *configs/* override this configuration as follows: A0 to A5 add, cumulatively, residual blocks, attention gates, the MAF stem, the Tversky term and the adversarial stage; *t_dice_eq* and *t_fn_heavy* change the Tversky weights of A4 to $(0.5,0.5)$ and $(0.2,0.8)$; *g_lam001*, *g_lam020*, *g_hinge* and *g_nosmooth* change the adversarial weight to 0.01 and 0.20, use the hinge objective, or remove the label smoothing and the instance noise.

@section[a-code] Code Organisation

The package *bratsgan* contains the modules *data.py* (BraTS discovery, preprocessing cache, patient-level grade-stratified splits with the leakage assertion, foreground-biased patch sampler), *augment.py* (GPU augmentation), *models.py* (the 3D U-Net family and the conditional 3D PatchGAN), *losses.py* (Dice, Tversky, BCE with deep supervision; LSGAN and hinge objectives), *train.py* (two-stage training loop), *infer.py* (Gaussian-weighted sliding-window inference), *metrics.py* (Dice, IoU, precision, sensitivity, HD95 with the BraTS conventions), *evaluate.py*, *features.py* (the 101 grading features), *stats.py* (bootstrap, Wilcoxon, Holm, Cliff's delta, $d_z$, AUC, calibration error), *complexity.py* (exact parameter and operation counts) and *synthetic.py* (phantoms).

The scripts directory contains *prepare_data.py* (conversion and splitting), *run_experiments.sh* (the experiment matrix), *collect_results.py* (aggregation and paired comparisons), *grade_classification.py* (the leak-free grading protocol), *export_nnunet.py* and *eval_nnunet.py* (the baseline) and *smoke_test.sh* (an end-to-end test on miniature phantoms). The directory *tests* holds the 23 automated tests, and *experiments* the scripts that produced the verification results of Chapter 4 and the figures of Chapters 3 and 4.

@section[a-feat] Features of the Grade Classifier

@tab feat | widths=1.5,3.8,0.7 | align=llc | size=8.5
The 101 features computed from the predicted masks and the images (*bratsgan/features.py*). Compartments: WT, TC, ET, necrotic and non-enhancing core NCR = TC $\\setminus$ ET, and oedema ED = WT $\\setminus$ TC; $V_k$ is the volume of compartment $k$ in mm³.
| Group | Definition | Number |
| --- | --- | --- |
| Log-volume | $\\log(1+V_k)$ for $k\\in$ WT, TC, ET, NCR, ED | 5 |
| Volume ratios | ET/WT, ET/TC, TC/WT, NCR/TC, ED/WT | 5 |
| Sphericity | $\\pi^{1/3}(6V_k)^{2/3}/A_k$ with the surface area $A_k$ from the surface voxels, for WT, TC, ET | 3 |
| Extent | volume divided by the volume of the bounding box, for WT, TC, ET | 3 |
| Maximum diameter | longest side of the bounding box (mm), for WT, TC, ET | 3 |
| Intensity | mean, standard deviation, 10th and 90th percentile of each of the four modalities in each of the five compartments | 80 |
| Enhancement contrast | mean T1ce minus mean T1 inside ET and inside TC | 2 |
| Total | | 101 |
@end

Empty compartments (frequent for ET in LGG) contribute zeros for their features, a convention that is applied identically to predicted and reference masks.

@section[a-tests] Automated Tests

The test file *tests/test_core.py* has 21 test functions (23 cases after parametrisation). They check the metrics (Dice and IoU against hand-computed values, the conventions for empty masks, the scaling of HD95 with the voxel spacing, the relation $J=D/(2-D)$ and the Jensen bound, the mapping of labels to regions); the losses (perfect and opposite predictions, the reduction of the Tversky loss to the Dice loss for $\\alpha=\\beta=\\tfrac12$, the larger penalty of missed voxels for $\\beta>\\alpha$, finite-difference gradient checks, deep-supervision shapes); the networks (output shapes of four variants, attention coefficients in $(0,1)$, fusion weights that sum to one, the patch-map output of the discriminator, gradient flow to all parameters); the inference (peak of the window weights at the centre, exact reconstruction of a constant model on a volume that is not a multiple of the patch size); the data handling (disjoint stratified patient-level partitions, detection of a duplicated patient by the leakage assertion, in-brain standardisation); and the statistics (Holm adjustment, AUC, bootstrap interval).
"""


def appB() -> str:
    t1 = (D["epochs_seg"] + D["epochs_adv"]) * D["iters_per_epoch"]
    runs = [("Tier 1", "A0, A5 on folds 0 to 4", 10), ("Tier 2", "A1, A2, A3, A4 on fold 0", 4), ("Tier 3", "six sensitivity runs on fold 0", 6)]
    rows = _tab([(a, b, c, f"{c * t1:,}") for a, b, c in runs] + [("Total", "", sum(r[2] for r in runs), f"{sum(r[2] for r in runs) * t1:,}")])
    return f"""@appendix[plan-app] Experiment Plan and Reproducibility

@section[b-steps] Procedure

The BraTS results are produced by seven commands of the code base, collected in the RUNBOOK that accompanies the thesis: (1) obtain the BraTS 2020 training release (369 cases) under the terms of the challenge; (2) preprocess and split with *scripts/prepare_data.py* (bounding-box crop with a margin of four voxels, never to the tumour; percentile clipping; in-brain standardisation; patient-level grade-stratified partition with the leakage assertion); (3) run the optional smoke test *scripts/smoke_test.sh*; (4) train the three tiers with *scripts/run_experiments.sh*; (5) aggregate with *scripts/collect_results.py* and *scripts/grade_classification.py*; (6) train and evaluate the nnU-Net baseline with *scripts/export_nnunet.py* and *scripts/eval_nnunet.py*; (7) rebuild the thesis with *thesis/build_thesis.py --brats*; every table cell without an output file remains pending.

@section[b-cost] Computational Cost

All variants use the same number of generator updates ({t1:,}); @tab:cost lists the runs and updates per tier. The wall-clock time depends on the hardware and is not estimated: the time per iteration should be measured with a short pilot (*configs/quick.json*) and multiplied by the numbers of the table.

@tab cost | widths=0.9,2.6,0.8,1.6 | align=llcc | size=9
Number of training runs and generator updates per tier ({t1:,} updates per run).
| Tier | Content | Runs | Generator updates |
| --- | --- | --- | --- |
{rows}
@end

@section[b-check] Reproducibility Checklist

- Every experiment is a JSON file with recorded overrides; the splits are fixed and checked for patient overlap on every load; no number of Chapter 4 is typed by hand; the tests and the smoke test should pass before any BraTS run.
"""


def appC() -> str:
    return r"""@appendix[deriv-app] Additional Derivations

@section[c-dice] Dice--IoU Relation and Jensen Bounds

Let a case have $T$ true-positive, $F$ false-positive and $M$ missed voxels. Then $D=2T/(2T+F+M)$ and $J=T/(T+F+M)$, and

@eq dice-to-iou
\frac{D}{2-D}=\frac{2T}{2(2T+F+M)-2T}=\frac{2T}{2T+2F+2M}=J .
@end

The function $j(d)=d/(2-d)$ has $j'(d)=2/(2-d)^2>0$ and $j''(d)=4/(2-d)^3>0$ on $[0,1]$, so it is increasing and convex. Jensen's inequality gives $\bar J=\frac1n\sum_i j(D_i)\ge j(\bar D)$ for the mean over cases. Moreover $j(d)\le d$ for $d\in[0,1]$, because $d/(2-d)\le d$ is equivalent to $d\le 1$; hence $\bar J\le\bar D$. Together, $\bar D/(2-\bar D)\le\bar J\le\bar D$, which is the bound used in Section @sec:v1-s. The first inequality is an equality if and only if all cases have the same Dice score, and the gap grows with their dispersion: a second-order expansion gives $\bar J-j(\bar D)\approx\tfrac12 j''(\bar D)\,s_D^2=2s_D^2/(2-\bar D)^3$ for the sample variance $s_D^2$ of the Dice scores.

@section[c-triv] The Trivial-Separation Pitfall

Consider the conditional discriminator of Section @sec:gan-model with inputs $(\mathbf{x},\mathbf{y})$, where the real label maps $\mathbf{y}$ are exactly binary and the generated maps $\hat{\mathbf{y}}=\sigma(\mathbf{z})$ take values in the open interval $(0,1)$ at every voxel. Then the supports of the two distributions are disjoint: the real distribution is concentrated on the vertices of the hypercube $\{0,1\}^n$, and the generated distribution gives probability zero to every vertex.

@note | title=Proposition (vanishing adversarial gradient)
Let $D$ be any discriminator that equals 0 on an open set $U$ containing the support of the generated distribution and 1 at the vertices. Then $D$ attains the minimum of the least-squares discriminator loss, and $\nabla_{\hat{\mathbf{y}}}D=0$ on $U$, so that the adversarial term of the generator loss has zero gradient with respect to the segmentation network.
@end

*Proof.* The least-squares discriminator loss $\tfrac12\mathbb{E}[(D(\mathbf{y})-1)^2]+\tfrac12\mathbb{E}[D(\hat{\mathbf{y}})^2]$ is non-negative and equals zero for the stated $D$. On the open set $U$ the function is constant, so its gradient vanishes there. Since the generator loss $\tfrac12\mathbb{E}[(D(\hat{\mathbf{y}})-1)^2]$ depends on the generator only through $D(\hat{\mathbf{y}})$, the chain rule gives a zero gradient. $\square$

For a network $D$ with bounded Lipschitz constant (spectral normalisation) the function cannot jump from 1 at a vertex to 0 on a set at distance $\varepsilon$ if $\varepsilon L<1$; the separation is then not exact, but it is the more complete the closer the generated maps are to binary, which is the case once the segmentation network is well trained. The remedy used in this thesis is to add Gaussian noise of standard deviation $\sigma$ to both inputs: the distributions of the noisy inputs are the convolutions of the original distributions with a Gaussian density, which have strictly positive densities everywhere, so that the optimal discriminator $p_r^{\sigma}/(p_r^{\sigma}+p_g^{\sigma})$ is a smooth function with, in general, non-vanishing gradient [@arjovsky_bottou; @sonderby]. Label smoothing limits the confidence of the discriminator on the real maps in the same direction. The weight $\lambda$ of the adversarial term is kept small and ramped up, so that the supervised terms dominate whenever the adversarial gradient is uninformative.

@section[c-mde] Minimum Detectable Difference

Let $d_1,\dots,d_n$ be paired differences with mean $\delta$ and standard deviation $\sigma_d$. For a two-sided test of level $\alpha$ based on $z=\bar d/(\sigma_d/\sqrt n)$ with an approximately normal sampling distribution, the null hypothesis $\delta=0$ is rejected if $|z|>z_{1-\alpha/2}$. If the true mean is $\delta>0$, then $z$ is approximately normal with mean $\delta\sqrt n/\sigma_d$ and unit variance, so that the power is $\Phi(\delta\sqrt n/\sigma_d-z_{1-\alpha/2})$ up to a negligible term. Setting the power to $1-\beta$ and solving for $\delta$ gives

@eq mde-deriv
\delta_{\min}=\big(z_{1-\alpha/2}+z_{1-\beta}\big)\frac{\sigma_d}{\sqrt n},\qquad z_{0.975}+z_{0.80}=1.960+0.842=2.802,
@end

which is @eq:mde. For $n=74$ and $\sigma_d=0.05$ it gives $\delta_{\min}=2.802\times0.05/\sqrt{74}=0.0163$.

@section[c-deff] Design Effect

Suppose that each of $n$ patients contributes $m$ observations (slices) with common variance $\sigma^2$ and intra-patient correlation $\rho$. The variance of the mean of all $nm$ observations is

@eq deff
\operatorname{Var}(\bar x)=\frac{\sigma^2}{nm}\big[1+(m-1)\rho\big],
@end

so that the effective number of independent observations is $nm/\mathrm{deff}$ with $\mathrm{deff}=1+(m-1)\rho$ [@kish]. With $\rho=0$ the factor is 1; with $\rho=1$ the $nm$ observations carry the information of $n$. The values of Section @sec:v5s follow by inserting the intra-patient correlation estimated by the one-way random-effects model.

@section[c-rf] Receptive Field of the Encoder

With $r_0=1$ and $j_0=1$, the recursion $r_l=r_{l-1}+(k_l-1)\,j_{l-1}$, $j_l=s_l j_{l-1}$ gives the receptive field after layer $l$ with kernel size $k_l$ and stride $s_l$ [@erf]. At the first level the two $3\times3\times3$ convolutions give $r=5$. At each lower level the first convolution has stride two ($j$ doubles) and the second has stride one:

The MAF stem adds a $3\times3\times3$ convolution before the first level (the $1\times1\times1$ mixing convolution does not change the receptive field), which raises every value by two (7, 13, 25, 49, 97), in agreement with @tab:rf.
"""


def appD(V: dict, phantom_rows: str = "") -> str:
    g = V["v2_gradient_mass"]

    def share(r, k):
        return 100 * r[k]["tumour_mass"] / (r[k]["tumour_mass"] + r[k]["bg_mass"])

    v2 = _tab([(f"{100*r['frac']:.1f}%", f"{share(r,'bce'):.1f}", f"{share(r,'dice'):.1f}", f"{share(r,'tversky'):.1f}") for r in g])
    v1 = _tab([(r["mu"], r["sd"], f"{r['mean_dice']:.4f}", f"{r['mean_iou']:.4f}", f"{r['mean_dice']/(2-r['mean_dice']):.4f}", f"{r['jensen_gap']:.4f}") for r in V["v1_dice_iou"]])
    w = _tab([("x".join(map(str, x["shape"])), x["patch"], x["overlap"], x["windows"]) for x in V["v7_windows"]])
    m = _tab([(x["sd_diff"], x["n"], f"{x['mde']:.4f}") for x in V["v8_mde"]])
    return f"""@appendix[phantom-gen] Synthetic Phantom Generator and Supplementary Tables

@section[d-gen] The Phantom Generator

The generator (*bratsgan/synthetic.py*) draws, for each phantom, an ellipsoidal brain with semi-axes $(0.85, 0.72, 0.62)$ of the half-width of the volume, inside which three tissue classes are defined: a ventricle-like inner ellipsoid (cerebrospinal fluid), white matter inside the radius 0.78 of the normalised ellipsoidal coordinate, and grey matter in the rim. Base intensities per tissue are assigned for the four contrasts (T1, T1ce, T2, FLAIR). A tumour is placed at a random position (offsets within $\\pm0.3$) with random semi-axes in $[0.18, 0.34]$; its boundary is perturbed by smooth noise. The oedema is an ellipsoid 1.25 to 1.6 times larger than the tumour.

*HGG-like phantoms* have a necrotic core of 55% of the tumour radius and an enhancing shell between the core and the outer boundary of the tumour; the enhancement is strong in T1ce (intensity 1.00 against 0.20 for the core). *LGG-like phantoms* have a non-enhancing core that fills the outer ellipsoid (intensity 0.45 in T1ce) and, with probability 0.25, a small enhancing focus (35% of the radius, intensity 0.90 in T1ce); otherwise no enhancing tissue. Tumour compartments modify the T1, T2 and FLAIR intensities with fixed values (for example, oedema is bright in T2 and FLAIR). Each contrast is multiplied by a smooth bias field $\\exp(0.15\\,n)$ with $n$ a unit-variance Gaussian-filtered noise field (scale 12 voxels), and Rician-like noise is obtained by adding Gaussian noise (standard deviation 0.03) to the real and imaginary parts and taking the modulus. Labels follow the BraTS convention (1: core, 2: oedema, 4: enhancing tumour).

The phantoms are deliberately simpler than patient data: the contrast between compartments is high, the geometry is smooth and convex, there are no vessels, no partial-volume interfaces between lesion and sulci, and no inter-rater ambiguity. They are used to test the code, not to estimate accuracy.

The *hard* setting used in Section @sec:hard changes the following: the tumour semi-axes are drawn from $[0.10, 0.22]$, the core of an HGG-like phantom has 78% of the tumour radius (a thinner enhancing shell), the tumour contrast is reduced to 45% of the difference to the tissue value and the labelled tumour is blurred with a Gaussian of standard deviation 1.2 voxels, the noise standard deviation is 0.10 and the bias-field amplitude 0.30.

{phantom_rows}"""


def appE() -> str:
    return r"""@appendix[algo-app] Algorithms

This appendix lists, in pseudocode, the procedures of Chapter 3 that are not given there. They correspond one-to-one to functions of the code base.

@section[e-split] Patient-Level Grade-Stratified Partition

@algo split
Construction of the test set and the cross-validation folds (data.py).
Input: patients with grade labels, seed, test fraction f = 0.2, number of folds K = 5
rng <- random generator(seed)
for each grade g in sorted(grades):
    ids <- patients of grade g;  shuffle(ids, rng)
    n_test <- round(f * |ids|)
    test <- test + ids[0 : n_test]
    for j, id in enumerate(ids[n_test :]):
        fold[j mod K] <- fold[j mod K] + [id]
assert no patient occurs in two of {test, fold 0, ..., fold K-1}
Output: test (74 patients), folds of sizes 60, 59, 59, 59, 58 for BraTS 2020
@end

Rounding is done per grade, so the test set contains 59 HGG and 15 LGG patients; the folds inherit the grade proportion of the development pool up to a difference of one patient.

@section[e-hd] Region Metrics and HD95

@algo hd
Metrics of one region for one patient (metrics.py).
Input: binary masks P (prediction) and G (reference), voxel spacing
tp, fp, fn <- counts;  Dice <- 1 if tp+fp+fn = 0 else 2 tp / (2 tp + fp + fn)
if P and G are both empty:   HD95 <- 0
else if exactly one is empty: HD95 <- sqrt(240^2 + 240^2 + 155^2) = 373.13
else:
    S_P <- P minus erosion(P, 6-neighbourhood);  S_G <- G minus erosion(G, 6-neighbourhood)
    d_G <- Euclidean distance transform of the complement of S_G (with spacing)
    d_P <- Euclidean distance transform of the complement of S_P (with spacing)
    D <- concat( d_G[S_P], d_P[S_G] )               # pooled directed distances
    HD95 <- percentile(D, 95)
Output: Dice, HD95 (and IoU, precision, sensitivity, specificity from the same counts)
@end

@section[e-win] Gaussian-Weighted Sliding Window

@algo window
Whole-volume inference (infer.py).
Input: normalised volume x (C x D x H x W), model G, patch size P, overlap o = 0.5
pad x with zeros at the far borders so that every side is at least P
step_a <- max(1, floor(P_a (1 - o)))   for each axis a
w <- Gaussian weight of the patch (sigma = P/8 per axis), normalised to maximum 1, floor 1e-4
for each window start (z, y, x) on the grid of starts, the last start flushed to the border:
    p <- sigmoid( G( x[:, window] ) )                # optionally averaged over mirrored inputs
    acc[:, window] <- acc[:, window] + p * w
    wsum[window]   <- wsum[window] + w
prob <- (acc / wsum) cropped to the original size
regions <- (prob > 0.5), then ET <- ET and TC, TC <- TC and WT
Output: binary maps for WT, TC, ET
@end

@section[e-boot] Bootstrap Interval and Holm Correction

@algo boot
Percentile bootstrap of a paired difference and Holm adjustment (stats.py).
Input: paired per-patient scores a_i, b_i (i = 1..n); B = 10,000
d_i <- a_i - b_i;  rng <- random generator(0)
for b = 1..B:   draw n indices with replacement;  m_b <- mean(d[indices])
CI <- (2.5th, 97.5th percentile of m_1..m_B);  estimate <- mean(d)
p <- two-sided Wilcoxon signed-rank test on (a, b)       # p = 1 if all d_i = 0
Holm over m comparisons with ordered p-values p_(1) <= ... <= p_(m):
    p_adj(i) <- max over j <= i of min(1, (m - j + 1) p_(j));  restore original order
Output: estimate, CI, p, p_adj
@end

@section[e-oof] Evaluation Order and Aggregation

The order of aggregation matters for the figures of Chapter 4. For the cross-validation tables, the per-patient metrics of the five validation folds are concatenated (each development patient is segmented once, by the model that did not train on it) and averaged over the 295 patients. For the test tables, each of the 74 test patients is segmented by each of the five fold models, the per-patient metric is averaged over the five models, and the mean over patients is reported; the paired comparisons use the per-patient averages. The fold models of one variant are thus treated as five noisy realisations of the same model; the average over them reduces the variance of the training run but not the variance due to the sampling of the test patients.
"""


def appF() -> str:
    files = [
        ("bratsgan/config.py", "Configuration defaults", "All hyper-parameters and their default values; a configuration file overrides only the keys it names, and every run stores the resolved configuration in its output directory."),
        ("bratsgan/data.py", "Data pipeline", "Reading of the BraTS layout, label conversion to the three nested regions, cropping to the brain bounding box, intensity normalisation, the patient-level grade-stratified split with its leakage assertion, and the patch sampler."),
        ("bratsgan/models.py", "Networks", "The configurable generator (residual blocks, additive attention gates, modality-attentive fusion stem, deep supervision) and the conditional 3D PatchGAN discriminator."),
        ("bratsgan/losses.py", "Losses", "Soft Dice, Tversky and binary cross-entropy terms with deep supervision, the hierarchy projection, and the least-squares adversarial objectives with label smoothing and instance noise."),
        ("bratsgan/train.py", "Training loop", "Optimisers, the polynomial learning-rate schedule, the two-stage schedule, gradient clipping, validation and the final test pass."),
        ("bratsgan/infer.py", "Sliding-window inference", "Gaussian-weighted whole-volume inference with border flushing and weight normalisation."),
        ("bratsgan/metrics.py", "Evaluation metrics", "Dice, HD95 with the empty-mask conventions, sensitivity, precision and volumetric agreement."),
        ("bratsgan/stats.py", "Statistical analysis", "Percentile bootstrap, paired bootstrap, Wilcoxon test with Holm adjustment and Cliff's delta."),
        ("bratsgan/features.py", "Grade-classification features", "The 101 features computed from a segmentation and the four images."),
    ]
    keep = ("bratsgan/losses.py", "bratsgan/metrics.py", "bratsgan/stats.py")
    files = [f for f in files if f[0] in keep]
    out = ["@appendix[src-app] Source Listings of the Principal Modules", "",
           "This appendix reproduces, without abbreviation, the modules that realise the loss functions, the evaluation metrics and the statistical analysis, so that the corresponding definitions of Chapter 3 can be checked against the code. Comments are preserved; blank lines are not drawn. The complete package, with the tests, the configuration files and the scripts, accompanies the thesis (see the RUNBOOK).", ""]
    for k, (path, title, desc) in enumerate(files, 1):
        text = (ROOT / path).read_text().rstrip().splitlines()
        body = "\n".join(l for l in text if l.strip())
        out += [f"@section[src-{k}] {title}", "", f"*{path}* ({len(text)} lines). {desc}", "", "@code", body, "@end", ""]
    return "\n".join(out)


def appN() -> str:
    """Appendix of notation: symbols, abbreviations and metric definitions used in Chapters 3 to 6."""
    sym = [
        (r"$\mathbf{x}$", r"Input volume of four modalities (T1, T1ce, T2, FLAIR), $\mathbf{x}\in\mathbb{R}^{4\times N}$", r"3"),
        (r"$N$", r"Number of voxels of the (cropped) volume or of a patch", r"3"),
        (r"$L$, $\mathbf{y}$", r"Label map with values $\{0,1,2,4\}$ and its encoding as three nested binary regions $(y_{\mathrm{WT}},y_{\mathrm{TC}},y_{\mathrm{ET}})$", r"3"),
        (r"$f_\theta$, $G$", r"Segmentation network (generator) with parameters $\theta$", r"3"),
        (r"$D_\psi$", r"Conditional 3D PatchGAN discriminator", r"3"),
        (r"$\mathbf{z}$, $\mathbf{p}$", r"Logits and probabilities, $\mathbf{p}=\sigma(\mathbf{z})$", r"3"),
        (r"$\sigma$", r"Logistic function", r"3"),
        (r"$\hat{y}_r$", r"Binary prediction of region $r$ after thresholding and projection onto the hierarchy", r"3"),
        (r"$\mathrm{TP}_r,\mathrm{FP}_r,\mathrm{FN}_r$", r"True positives, false positives and false negatives of region $r$ (soft counts in the losses)", r"3"),
        (r"$\mathcal{L}_{\mathrm{Dice}},\mathcal{L}_{\mathrm{TV}},\mathcal{L}_{\mathrm{BCE}}$", r"Dice, Tversky and binary cross-entropy losses", r"3"),
        (r"$(\alpha,\beta)$", r"Tversky weights of false positives and false negatives; default $(0.3,0.7)$", r"3"),
        (r"$\epsilon$", r"Smoothing constant of the overlap losses, $10^{-5}$", r"3"),
        (r"$\mathcal{L}_{\mathrm{seg}}^{\mathrm{DS}}$", r"Deep-supervision segmentation loss with weights $(4/7,2/7,1/7)$", r"3"),
        (r"$\lambda$", r"Weight of the adversarial term in the generator objective", r"3"),
        (r"$\alpha(v)$", r"Attention coefficient of an additive attention gate at voxel $v$", r"3"),
        (r"$W_s,W_g,\boldsymbol{\psi}$", r"$1\times1\times1$ convolutions of the attention gate", r"3"),
        (r"$g_\phi$", r"Grade classifier with parameters $\phi$", r"3"),
        (r"$\mathbf{u}$", r"Feature vector (101 features) computed from images and predicted regions", r"3"),
        (r"$\mathcal{S}_m$", r"Expected value of the metric $m$ under the data distribution", r"3"),
        (r"$\Delta$", r"Paired difference of a metric between two variants (later minus earlier)", r"4"),
        (r"$d_z$", r"Standardised paired effect size (mean difference divided by the standard deviation of the differences)", r"4"),
        (r"$\delta$", r"Cliff's delta, a non-parametric effect size in $[-1,1]$", r"4"),
        (r"$B$", r"Number of bootstrap resamples (10,000)", r"4"),
    ]
    abbr = [
        (r"BraTS", r"Brain Tumor Segmentation Challenge"), (r"HGG, LGG", r"High-grade and low-grade glioma"),
        (r"WT, TC, ET", r"Whole tumour, tumour core, enhancing tumour (nested regions)"),
        (r"NCR/NET, ED", r"Necrotic and non-enhancing tumour core, peritumoural oedema"),
        (r"MAF", r"Modality-attentive fusion (stem of the network)"), (r"GAN", r"Generative adversarial network"),
        (r"LSGAN", r"Least-squares GAN objective"), (r"PatchGAN", r"Discriminator that scores overlapping patches"),
        (r"HD95", r"95th-percentile Hausdorff distance"), (r"IoU", r"Intersection over union (Jaccard index)"),
        (r"ECE", r"Expected calibration error"), (r"AUC", r"Area under the ROC curve"), (r"CI", r"Confidence interval"),
        (r"MDE", r"Minimum detectable effect"), (r"CV", r"Cross-validation"), (r"OOF", r"Out-of-fold"),
        (r"A0 to A5", r"Ablation variants: baseline, residual, attention, MAF stem, Tversky loss, full model with adversarial term"),
    ]
    metr = [
        (r"Dice", r"$\mathrm{Dice}=\dfrac{2\,\mathrm{TP}}{2\,\mathrm{TP}+\mathrm{FP}+\mathrm{FN}}$"),
        (r"IoU", r"$\mathrm{IoU}=\dfrac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}+\mathrm{FN}}=\dfrac{\mathrm{Dice}}{2-\mathrm{Dice}}$"),
        (r"Sensitivity", r"$\mathrm{TP}/(\mathrm{TP}+\mathrm{FN})$"),
        (r"Precision", r"$\mathrm{TP}/(\mathrm{TP}+\mathrm{FP})$"),
        (r"Specificity", r"$\mathrm{TN}/(\mathrm{TN}+\mathrm{FP})$"),
        (r"HD95", r"95th percentile of the symmetric set of surface distances; 373.13 voxels if exactly one of the two masks is empty"),
        (r"ECE", r"$\sum_b \frac{n_b}{n}\,d_b$ over ten equal-width probability bins, with $d_b$ the absolute difference between the observed frequency and the mean predicted probability in bin $b$"),
    ]
    out = ["@appendix[notation-app] Notation, Abbreviations and Metric Definitions", "",
           "This appendix collects the symbols, abbreviations and metric definitions used in Chapters 3 to 6, so that the tables and figures of the results can be read without returning to the methodology. The last column of @tab:sym gives the chapter in which the symbol is introduced.", "",
           "@section[not-sym] Symbols", "",
           "@tab sym | widths=1.5,4.4,0.6 | align=llc | size=9", "Symbols used in the thesis.",
           "| Symbol | Meaning | Chapter |", "| --- | --- | --- |"]
    out += [f"| {a} | {b} | {c} |" for a, b, c in sym]
    out += ["@end", "", "@section[not-abbr] Abbreviations", "",
            "@tab abbr | widths=1.5,4.9 | align=ll | size=9", "Abbreviations.", "| Abbreviation | Meaning |", "| --- | --- |"]
    out += [f"| {a} | {b} |" for a, b in abbr]
    out += ["@end", "", "@section[not-met] Metric Definitions", "",
            "The metrics are computed per patient and per region from the binary masks; the table gives the definition used by the code (*bratsgan/metrics.py*).", "",
            "@tab metdef | widths=1.4,5.0 | align=ll | size=9", "Definitions of the evaluation metrics.", "| Metric | Definition |", "| --- | --- |"]
    out += [f"| {a} | {b} |" for a, b in metr]
    out += ["@end", ""]
    return "\n".join(out)


def appX(x: str) -> str:
    """Appendix of supplementary results of the hard benchmark (built by appx_hard.build, frozen with the chapter)."""
    return x or "@appendix[hard-supp] Supplementary Results of the Hard Synthetic Benchmark\n\n~The runs of the hard benchmark were not available when this document was built.\n"


if __name__ == "__main__":
    print(appA()[:500])
