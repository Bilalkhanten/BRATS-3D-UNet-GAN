"""Implementation-verification experiments that need no trained model.
All numbers are computed here (analytic or by simulation/autograd); nothing is typed in by hand.
Outputs: runs/verification/{verify.json, figs/*.png}"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from scipy import stats as sst
from figstyle import C, LS, MK, SERIES, plt, setup
from bratsgan.complexity import count_params, profile
from bratsgan.infer import _starts
from bratsgan.losses import soft_dice_loss, tversky_loss
from bratsgan.models import Discriminator3D, UNet3D
from bratsgan.synthetic import make_phantom

setup()
OUT = Path(__file__).resolve().parents[1] / "runs" / "verification"; FIG = OUT / "figs"; FIG.mkdir(parents=True, exist_ok=True)
R = {}


def save(fig, name):
    fig.savefig(FIG / name); plt.close(fig)


# ---------------------------------------------------------------- V1 Dice <-> IoU
def beta_ab(mu, sd):
    k = mu * (1 - mu) / sd ** 2 - 1
    return mu * k, (1 - mu) * k


rng = np.random.default_rng(0)
rows = []
for mu in (0.85, 0.90, 0.95, 0.99):
    for sd in (0.02, 0.05, 0.10):
        if sd ** 2 >= mu * (1 - mu):
            continue
        a, b = beta_ab(mu, sd)
        d = rng.beta(a, b, 400_000)
        iou = d / (2 - d)
        rows.append({"mu": mu, "sd": sd, "mean_dice": float(d.mean()), "mean_iou": float(iou.mean()),
                     "iou_of_mean_dice": float(d.mean() / (2 - d.mean())), "jensen_gap": float(iou.mean() - d.mean() / (2 - d.mean()))})
R["v1_dice_iou"] = rows
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
x = np.linspace(0, 1, 200)
ax[0].plot(x, x / (2 - x), color=C["blue"], label=r"$\mathrm{IoU}=D/(2-D)$")
ax[0].plot(x, x, color=C["grey"], ls=":", label="IoU = Dice")
d1, d2 = 0.80, 0.99
ax[0].plot([d1, d2], [d1 / (2 - d1), d2 / (2 - d2)], color=C["orange"], ls="--", label="chord (mean of two cases)")
ax[0].set_xlabel("Dice $D$"); ax[0].set_ylabel("IoU"); ax[0].legend(frameon=False, loc="upper left"); ax[0].set_title("(a) IoU is a convex function of Dice")
dd = np.linspace(0.90, 0.995, 200)
ax[1].plot(dd, dd / (2 - dd), color=C["blue"], label=r"lower bound: $\bar D/(2-\bar D)$")
ax[1].fill_between(dd, dd / (2 - dd), 1, color=C["blue"], alpha=0.08)
ax[1].scatter([0.99], [0.97], color=C["red"], zorder=5, label="reported pair (0.99, 0.97)")
ax[1].annotate("infeasible: needs\nmean IoU $\\geq$ 0.980", (0.99, 0.97), (0.915, 0.985), arrowprops=dict(arrowstyle="->", color=C["soft"]), fontsize=8.5)
ax[1].set_xlabel(r"mean Dice $\bar D$"); ax[1].set_ylabel(r"mean IoU $\bar I$"); ax[1].set_ylim(0.82, 1.002)
ax[1].legend(frameon=False, loc="lower right"); ax[1].set_title("(b) feasible region for $(\\bar D,\\bar I)$")
fig.tight_layout(); save(fig, "v1_dice_iou.png")
R["v1_constraints"] = {"iou_floor_for_dice_0.99": 0.99 / 1.01, "max_dice_for_iou_0.97": 2 * 0.97 / 1.97,
                       "f1_from_P0.9756_R0.9837": 2 * 0.9756 * 0.9837 / (0.9756 + 0.9837)}

# ---------------------------------------------------------------- V2 gradient mass vs tumour fraction
fracs = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
N = (100, 10, 10)  # 10,000 voxels
g_rows = []
for f in fracs:
    n = int(np.prod(N)); nt = max(1, int(round(f * n)))
    t = torch.zeros(n); t[:nt] = 1
    z = torch.where(t > 0, torch.tensor(-0.85), torch.tensor(-2.94)).clone().requires_grad_(True)   # p_tumour~0.30, p_bg~0.05
    out = {}
    for name, fn in (("bce", lambda p, tt, zz: torch.nn.functional.binary_cross_entropy_with_logits(zz, tt)),
                     ("dice", lambda p, tt, zz: soft_dice_loss(p.view(1, 1, *N), tt.view(1, 1, *N))),
                     ("tversky", lambda p, tt, zz: tversky_loss(p.view(1, 1, *N), tt.view(1, 1, *N), 0.3, 0.7))):
        z.grad = None
        p = torch.sigmoid(z); fn(p, t, z).backward()
        gm = z.grad.abs()
        out[name] = {"tumour_mass": float(gm[t > 0].sum()), "bg_mass": float(gm[t == 0].sum())}
    g_rows.append({"frac": f, **out})
R["v2_gradient_mass"] = g_rows
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
for i, (k, lab) in enumerate((("bce", "BCE"), ("dice", "Dice"), ("tversky", r"Tversky ($\alpha$=0.3, $\beta$=0.7)"))):
    tm = np.array([r[k]["tumour_mass"] for r in g_rows]); bm = np.array([r[k]["bg_mass"] for r in g_rows])
    ax[0].plot(fracs, tm, color=SERIES[i], ls=LS[i], marker=MK[i], ms=4, label=lab)
    ax[1].plot(fracs, tm / (tm + bm), color=SERIES[i], ls=LS[i], marker=MK[i], ms=4, label=lab)
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("tumour fraction of volume"); ax[0].set_ylabel(r"$\sum_{i\in \mathrm{tumour}}|\partial L/\partial z_i|$")
ax[0].set_title("(a) gradient mass on tumour voxels"); ax[0].legend(frameon=False, loc="lower right")
ax[1].set_xscale("log"); ax[1].set_xlabel("tumour fraction of volume"); ax[1].set_ylabel("share of gradient mass on tumour")
ax[1].set_title("(b) share of total gradient mass"); fig.tight_layout(); save(fig, "v2_gradient_mass.png")

# ------------------------------------------------------- V9 Dice vs Tversky penalty surface
TPv = 1000
fn_ = np.arange(0, 401, 5); fp_ = np.arange(0, 401, 5)
FN, FP = np.meshgrid(fn_, fp_)
dice_l = 1 - 2 * TPv / (2 * TPv + FP + FN)
tv_l = 1 - TPv / (TPv + 0.3 * FP + 0.7 * FN)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for a_, L, t in ((ax[0], dice_l, "(a) Dice loss"), (ax[1], tv_l, r"(b) Tversky loss, $\alpha$=0.3, $\beta$=0.7")):
    cs = a_.contourf(FN, FP, L, levels=np.linspace(0, 0.5, 11), cmap="Blues"); a_.contour(FN, FP, L, levels=[0.1, 0.2, 0.3], colors=C["ink"], linewidths=0.6)
    a_.plot([0, 400], [0, 400], color=C["orange"], ls="--", lw=1.0); a_.set_xlabel("false negatives (voxels)"); a_.set_title(t); a_.grid(False)
ax[0].set_ylabel("false positives (voxels)"); fig.colorbar(cs, ax=ax, shrink=0.85, label="loss"); save(fig, "v9_dice_vs_tversky.png")
R["v9"] = {"TP": TPv, "dice_at_fn200_fp0": float(1 - 2 * TPv / (2 * TPv + 200)), "dice_at_fn0_fp200": float(1 - 2 * TPv / (2 * TPv + 200)),
           "tv_at_fn200_fp0": float(1 - TPv / (TPv + 0.7 * 200)), "tv_at_fn0_fp200": float(1 - TPv / (TPv + 0.3 * 200))}

# -------------------------------------------------------- V3 complexity and receptive field
variants = [("plain 3D U-Net", dict(residual=False, attention=False, fusion="early")),
            ("+ residual blocks", dict(residual=True, attention=False, fusion="early")),
            ("+ attention gates", dict(residual=True, attention=True, fusion="early")),
            ("+ modality-attentive fusion", dict(residual=True, attention=True, fusion="maf"))]
cx = []
for name, kw in variants:
    m = UNet3D(base=32, depth=5, max_ch=320, **kw)
    p = profile(m, (1, 4, 128, 128, 128)); p["name"] = name; cx.append(p)
D = Discriminator3D(4, 3, 32)
cx.append({"name": "discriminator (PatchGAN, 3D)", "params": count_params(D), **{k: float("nan") for k in ("gmacs", "gflops", "activation_elems_M", "activation_GB_fp32")}})
R["v3_complexity"] = cx


def receptive_field(maf: bool, depth=5):
    r, j, out = 1, 1, []
    def conv(k, s):
        nonlocal r, j
        r += (k - 1) * j; j *= s
    if maf:
        conv(3, 1)
    for i in range(depth):
        conv(3, 1 if i == 0 else 2); conv(3, 1)
        out.append({"stage": i, "stride": j, "rf": r})
    return out


R["v3_rf_early"] = receptive_field(False); R["v3_rf_maf"] = receptive_field(True)

# ---------------------------------------------------------------- V7 sliding-window counts
def n_windows(shape, patch, overlap):
    step = [max(1, int(p * (1 - overlap))) for p in patch]
    n = 1
    for s, p, st in zip(shape, patch, step):
        n *= len(_starts(max(s, p), p, st))
    return n


win = []
for shape, label in (((240, 240, 155), "full BraTS volume"), ((160, 192, 144), "typical brain crop")):
    for patch in (96, 128):
        for ov in (0.0, 0.25, 0.5, 0.75):
            win.append({"shape": shape, "label": label, "patch": patch, "overlap": ov, "windows": n_windows(shape, (patch,) * 3, ov)})
R["v7_windows"] = win

# --------------------------------------------------- V5 slice dependence (design effect) on phantoms
def slice_dependence(vols_labels):
    areas = []
    for lab in vols_labels:
        a = (lab > 0).sum(axis=(1, 2)).astype(float)
        areas.append(a)
    A = np.stack(areas)                                  # patients x slices
    adj = [np.corrcoef(a[:-1], a[1:])[0, 1] for a in A if a.std() > 0]
    # one-way ANOVA ICC(1) of slice-level tumour area among tumour-containing slices
    grp = [a[a > 0] for a in A]
    m = int(round(np.mean([len(g) for g in grp]))); k = len(grp)
    gm = np.concatenate(grp).mean()
    msb = sum(len(g) * (g.mean() - gm) ** 2 for g in grp) / (k - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in grp) / (sum(len(g) for g in grp) - k)
    icc = (msb - msw) / (msb + (m - 1) * msw)
    deff = 1 + (m - 1) * icc
    return {"n_patients": k, "mean_tumour_slices": m, "adjacent_slice_corr": float(np.mean(adj)), "icc1": float(icc), "design_effect": float(deff),
            "n_slices": int(sum(len(g) for g in grp)), "effective_n": float(sum(len(g) for g in grp) / deff)}


labs = [make_phantom(48, "HGG" if i < 75 else "LGG", 5000 + i)[1] for i in range(100)]
R["v5_slice_dependence"] = slice_dependence(labs)

# ------------------------------------------------------------- V6 bootstrap coverage
def coverage(n, mu=0.88, sd=0.10, reps=400, boot=800, seed=0):
    rng = np.random.default_rng(seed)
    a, b = beta_ab(mu, sd)
    cov_b = cov_t = 0; w_b = []; w_t = []
    for _ in range(reps):
        x = rng.beta(a, b, n)
        idx = rng.integers(0, n, (boot, n)); bm = x[idx].mean(1)
        lo, hi = np.percentile(bm, [2.5, 97.5]); cov_b += lo <= mu <= hi; w_b.append(hi - lo)
        se = x.std(ddof=1) / np.sqrt(n); tcrit = sst.t.ppf(0.975, n - 1)
        cov_t += (x.mean() - tcrit * se <= mu <= x.mean() + tcrit * se); w_t.append(2 * tcrit * se)
    return cov_b / reps, cov_t / reps, float(np.mean(w_b)), float(np.mean(w_t))


cv = []
for n in (10, 20, 40, 74, 150, 300):
    cb, ct, wb, wt = coverage(n)
    cv.append({"n": n, "cov_bootstrap": cb, "cov_t": ct, "width_bootstrap": wb, "width_t": wt})
R["v6_coverage"] = cv
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
ns = [r["n"] for r in cv]
ax[0].plot(ns, [r["cov_bootstrap"] for r in cv], color=C["blue"], marker="o", ms=4, label="percentile bootstrap")
ax[0].plot(ns, [r["cov_t"] for r in cv], color=C["orange"], ls="--", marker="s", ms=4, label="Student-$t$")
ax[0].axhline(0.95, color=C["grey"], ls=":", lw=1); ax[0].set_xscale("log"); ax[0].set_ylim(0.85, 1.0)
ax[0].set_xlabel("number of test cases $n$"); ax[0].set_ylabel("empirical coverage of nominal 95% CI"); ax[0].legend(frameon=False, loc="lower right"); ax[0].set_title("(a) coverage, skewed Dice ($\\mu$=0.88, $\\sigma$=0.10)")
ax[1].plot(ns, [r["width_bootstrap"] for r in cv], color=C["blue"], marker="o", ms=4, label="mean CI width")
ax[1].set_xscale("log"); ax[1].set_xlabel("number of test cases $n$"); ax[1].set_ylabel("95% CI width (Dice)"); ax[1].set_title("(b) precision vs sample size")
fig.tight_layout(); save(fig, "v6_ci_coverage.png")

# ------------------------------------------------------- V8 minimum detectable paired difference
mde = []
for sd_d in (0.03, 0.05, 0.08):
    for n in (20, 74, 295):
        z = sst.norm.ppf(0.975) + sst.norm.ppf(0.80)
        mde.append({"sd_diff": sd_d, "n": n, "mde": float(z * sd_d / np.sqrt(n))})
R["v8_mde"] = mde

# ----------------------------------------------------- phantom examples figure
fig, axes = plt.subplots(2, 6, figsize=(7.4, 2.9))
for r_, (g, sd) in enumerate((("HGG", 11), ("LGG", 4))):
    v, lab, _ = make_phantom(64, g, sd)
    zc = int(np.argwhere(lab > 0)[:, 0].mean())
    for c_, nm in enumerate(("T1", "T1ce", "T2", "FLAIR")):
        axes[r_, c_].imshow(v[c_, zc], cmap="gray", vmin=0, vmax=np.percentile(v[c_], 99.5)); axes[r_, c_].set_title(nm if r_ == 0 else "", fontsize=9)
    cm = np.zeros(lab[zc].shape + (3,)); cm[lab[zc] == 2] = [0.9, 0.75, 0.2]; cm[lab[zc] == 1] = [0.15, 0.45, 0.85]; cm[lab[zc] == 4] = [0.9, 0.25, 0.2]
    axes[r_, 4].imshow(cm); axes[r_, 4].set_title("labels" if r_ == 0 else "", fontsize=9)
    axes[r_, 5].imshow(v[3, zc], cmap="gray"); axes[r_, 5].contour(lab[zc] > 0, levels=[0.5], colors=[C["orange"]], linewidths=0.9); axes[r_, 5].set_title("WT contour" if r_ == 0 else "", fontsize=9)
    axes[r_, 0].set_ylabel(g, fontsize=10)
for a_ in axes.ravel():
    a_.set_xticks([]); a_.set_yticks([]); a_.grid(False)
    for s in a_.spines.values(): s.set_visible(False)
fig.tight_layout(); save(fig, "phantom_examples.png")

(OUT / "verify.json").write_text(json.dumps(R, indent=1))
print(json.dumps({k: R[k] for k in ("v1_constraints", "v5_slice_dependence", "v3_rf_early", "v3_rf_maf")}, indent=1)); print([ (r['n'],round(r['cov_bootstrap'],3),round(r['cov_t'],3)) for r in cv])
print(cx[0]['params'], cx[3]['params'], cx[3]['gmacs'])
