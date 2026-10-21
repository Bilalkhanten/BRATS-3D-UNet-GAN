"""Conceptual diagrams drawn with matplotlib (vector-quality PNGs for the thesis)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import C, plt, setup
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle, Ellipse
setup()
FIG = Path(__file__).resolve().parents[1] / "runs" / "verification" / "figs"; FIG.mkdir(parents=True, exist_ok=True)
FILL = {"data": "#e8f0fb", "net": "#fdebe4", "loss": "#e4f5ee", "eval": "#f0eefb", "grey": "#f1f1ef"}
EDGE = {"data": C["blue"], "net": C["orange"], "loss": C["aqua"], "eval": C["violet"], "grey": C["grey"]}


def box(ax, x, y, w, h, text, kind="grey", fs=8.5, bold=False, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=FILL[kind], ec=EDGE[kind], lw=1.2, ls=ls)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=C["ink"])
    return (x, y, w, h)


def arrow(ax, p, q, color=None, style="-|>", lw=1.2, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, lw=lw, color=color or C["soft"], connectionstyle=f"arc3,rad={rad}", ls=ls))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off"); ax.grid(False)
    return fig, ax


def save(fig, name):
    fig.savefig(FIG / name); plt.close(fig)


# ------------------------------------------------------------- 1. system pipeline
fig, ax = canvas(7.6, 4.4)
ax.text(0.05, 4.2, "Training", fontsize=10, fontweight="bold"); ax.text(0.05, 2.0, "Inference and grading", fontsize=10, fontweight="bold")
box(ax, 0.05, 3.2, 1.2, 0.75, "BraTS 2020\n4 MRI sequences\n+ labels", "data", fs=8)
box(ax, 1.5, 3.2, 1.2, 0.75, "Pre-processing\ncrop, z-score", "data", fs=8)
box(ax, 2.95, 3.2, 1.3, 0.75, "Patch sampler\n(tumour-biased)\n+ augmentation", "data", fs=8)
box(ax, 4.5, 3.2, 1.45, 0.75, "Generator $G$\n3D U-Net\n(res + att. + MAF)", "net", fs=8)
box(ax, 6.2, 3.62, 1.35, 0.5, "Segmentation loss\nDice+Tversky+BCE", "loss", fs=7)
box(ax, 6.2, 2.85, 1.35, 0.62, "Discriminator $D$\n(image, mask)\n\u2192 patch scores", "net", fs=7)
for a_, b_ in (((1.25, 3.58), (1.5, 3.58)), ((2.7, 3.58), (2.95, 3.58)), ((4.25, 3.58), (4.5, 3.58))):
    arrow(ax, a_, b_)
arrow(ax, (5.95, 3.85), (6.2, 3.87)); arrow(ax, (5.95, 3.4), (6.2, 3.2))
arrow(ax, (6.87, 2.85), (6.87, 2.5), color=C["aqua"]); ax.text(6.93, 2.6, "adversarial loss to $G$", fontsize=7, color=C["soft"], va="center")
ax.text(0.05, 2.55, "stage 1: $G$ only  $\\rightarrow$  stage 2: $G$ and $D$\n(adversarial refinement)", fontsize=8, color=C["soft"], style="italic", va="center")
box(ax, 0.05, 0.85, 1.2, 0.75, "Held-out volume\n(whole brain)", "data", fs=8)
box(ax, 1.5, 0.85, 1.3, 0.75, "Sliding window\nGaussian weights\n50% overlap", "eval", fs=8)
box(ax, 3.05, 0.85, 1.25, 0.75, "Threshold +\nhierarchy\n$ET\\subset TC\\subset WT$", "eval", fs=8)
box(ax, 4.6, 1.2, 1.45, 0.55, "Segmentation metrics\nDice, HD95, ...", "eval", fs=7.5)
box(ax, 4.6, 0.5, 1.45, 0.55, "Radiomic features\n$\\rightarrow$ LGG / HGG", "eval", fs=7.5)
box(ax, 6.3, 0.85, 1.25, 0.75, "Statistics\nCI, Wilcoxon,\nHolm", "eval", fs=8)
for a_, b_ in (((1.25, 1.22), (1.5, 1.22)), ((2.8, 1.22), (3.05, 1.22)), ((4.3, 1.4), (4.6, 1.5)), ((4.3, 1.05), (4.6, 0.8)), ((6.05, 1.5), (6.3, 1.35)), ((6.05, 0.8), (6.3, 1.1))):
    arrow(ax, a_, b_)
arrow(ax, (5.2, 3.2), (2.15, 1.62), color=C["grey"], ls="--", rad=0.0); ax.text(3.75, 2.05, "best checkpoint", fontsize=7.5, color=C["soft"], rotation=27, ha="center")
save(fig, "d_pipeline.png")

# ----------------------------------------------------------------- 2. U-Net architecture
fig, ax = canvas(7.4, 4.5); ax.set_xlim(-0.15, 7.4)
widths = [32, 64, 128, 256, 320]
W_, H_ = 0.8, 0.42
ex = [0.15 + 0.85 * i for i in range(4)]; dx = [6.45 - 0.85 * i for i in range(4)]; yy = [3.2 - 0.7 * i for i in range(5)]
box(ax, 0.0, 3.95, 1.25, 0.4, "Input: 4 MRI\nsequences", "data", fs=7)
box(ax, 1.5, 3.95, 1.45, 0.4, "MAF stem\n(modality-attentive)", "data", fs=7)
arrow(ax, (1.25, 4.15), (1.5, 4.15)); arrow(ax, (2.2, 3.95), (0.6, 3.64), rad=0.25)
for i in range(4):
    box(ax, ex[i], yy[i], W_, H_, f"Enc {i}\n{widths[i]} ch", "net", fs=7)
    box(ax, dx[i], yy[i], W_, H_, f"Dec {i}\n{widths[i]} ch", "net", fs=7)
    mid = ((ex[i] + W_ + dx[i]) / 2, yy[i] + H_ / 2)
    ax.add_patch(Circle(mid, 0.115, fc="#fff7e0", ec=C["yellow"], lw=1.1)); ax.text(mid[0], mid[1], "AG", fontsize=6, ha="center", va="center")
    arrow(ax, (ex[i] + W_, mid[1]), (mid[0] - 0.115, mid[1]), color=C["grey"], style="-")
    arrow(ax, (mid[0] + 0.115, mid[1]), (dx[i], mid[1]), color=C["grey"], style="-" if i == 3 else "-|>")
for i in range(3):
    arrow(ax, (ex[i] + W_ / 2, yy[i]), (ex[i + 1] + W_ / 2, yy[i + 1] + H_))
    arrow(ax, (dx[i + 1] + W_ / 2, yy[i + 1] + H_), (dx[i] + W_ / 2, yy[i]))
box(ax, 3.3, yy[4], W_, H_, "Bottleneck\n320 ch", "net", fs=7)
arrow(ax, (ex[3] + W_ / 2, yy[3]), (3.55, yy[4] + H_)); arrow(ax, (4.1, yy[4] + H_ / 2), (dx[3] + W_ / 2, yy[3]))
for i, nm in ((0, "head: full res"), (1, "head: 1/2 res"), (2, "head: 1/4 res")):
    box(ax, dx[i] + 0.0, yy[i] + 0.62, W_, 0.24, nm, "loss", fs=5.8)
    arrow(ax, (dx[i] + W_ / 2, yy[i] + H_), (dx[i] + W_ / 2, yy[i] + 0.62))
ax.text(0.1, 0.1, "AG = additive attention gate on the skip connection.\nDown-sampling: strided 3x3x3 convolution.\nUp-sampling: transposed convolution.\nOutput: 3 sigmoid channels (WT, TC, ET);\nheads at 1/2 and 1/4 resolution are used only\nfor deep supervision during training.", fontsize=6.6, color=C["soft"], va="bottom")
save(fig, "d_unet.png")

# ---------------------------------------------------------------- 3. MAF module
fig, ax = canvas(7.6, 3.0)
for m, nm in enumerate(("T1", "T1ce", "T2", "FLAIR")):
    y = 2.1 - m * 0.55
    box(ax, 0.05, y, 0.65, 0.38, nm, "data", fs=8)
    box(ax, 0.95, y, 1.45, 0.38, f"stem {m+1}: conv-norm-act", "net", fs=7)
    box(ax, 2.75, y, 0.6, 0.38, "GAP", "eval", fs=8)
    arrow(ax, (0.7, y + 0.19), (0.95, y + 0.19)); arrow(ax, (2.4, y + 0.19), (2.75, y + 0.19))
    arrow(ax, (3.35, y + 0.19), (3.7, 1.55))
box(ax, 3.7, 1.15, 1.0, 0.8, "shared MLP\nscores $e_m$", "eval", fs=7.5)
box(ax, 4.95, 1.15, 1.05, 0.8, "softmax over\nmodalities\n$\\Rightarrow a_m$", "eval", fs=7.5)
arrow(ax, (4.7, 1.55), (4.95, 1.55))
box(ax, 6.25, 1.75, 1.3, 0.5, "fused = $\\sum_m a_m \\odot f_m$", "net", fs=7.5)
box(ax, 6.25, 0.85, 1.3, 0.6, "concat[fused, $f_1,\\ldots,f_4$]\n+ 1x1x1 conv", "net", fs=7)
arrow(ax, (6.0, 1.6), (6.25, 1.95)); arrow(ax, (6.9, 1.75), (6.9, 1.45))
ax.text(0.05, 2.8, "Modality-attentive fusion (MAF) stem", fontsize=9.5, fontweight="bold")
ax.set_ylim(-0.35, 3.0); ax.text(0.05, -0.3, "Stem outputs $f_m$ are pooled (GAP) to a per-channel descriptor, scored by a shared MLP, and weighted by a softmax over the four\nmodalities; the fused map and the four stem outputs are concatenated and mixed into the first encoder width.", fontsize=7.2, color=C["soft"], va="bottom")
save(fig, "d_maf.png")

# ------------------------------------------------------------ 4. attention gate
fig, ax = canvas(7.0, 3.2)
box(ax, 0.1, 1.85, 0.9, 0.5, "skip features $x$\n$C_x$ channels", "data", fs=7.5)
box(ax, 0.1, 0.35, 0.9, 0.5, "gating signal $g$\n(from decoder)", "data", fs=7)
box(ax, 1.5, 1.85, 0.8, 0.5, "$W_x$ (1×1×1)", "net", fs=8)
box(ax, 1.5, 0.35, 0.8, 0.5, "$W_g$ (1×1×1)", "net", fs=8)
ax.add_patch(Circle((2.8, 1.35), 0.17, fc="white", ec=C["soft"])); ax.text(2.8, 1.35, "+", ha="center", va="center", fontsize=12)
box(ax, 3.2, 1.1, 0.7, 0.5, "ReLU", "net", fs=8)
box(ax, 4.15, 1.1, 0.9, 0.5, "$\\psi$ (1×1×1)\nsigmoid", "net", fs=7.5)
box(ax, 5.3, 1.1, 0.7, 0.5, "$\\alpha\\in(0,1)$", "eval", fs=8)
ax.add_patch(Circle((6.5, 2.1), 0.17, fc="white", ec=C["soft"])); ax.text(6.5, 2.1, "×", ha="center", va="center", fontsize=12)
box(ax, 6.2, 0.45, 0.75, 0.5, "$\\hat{x}=\\alpha x$", "net", fs=8)
arrow(ax, (1.0, 2.1), (1.5, 2.1)); arrow(ax, (1.0, 0.6), (1.5, 0.6)); arrow(ax, (2.3, 2.1), (2.65, 1.45)); arrow(ax, (2.3, 0.6), (2.65, 1.25))
arrow(ax, (2.97, 1.35), (3.2, 1.35)); arrow(ax, (3.9, 1.35), (4.15, 1.35)); arrow(ax, (5.05, 1.35), (5.3, 1.35))
arrow(ax, (6.0, 1.45), (6.4, 1.97)); ax.plot([0.55, 0.55, 6.5], [2.35, 2.62, 2.62], color=C["grey"], lw=1.2); arrow(ax, (6.5, 2.62), (6.5, 2.27), color=C["grey"])
arrow(ax, (6.5, 1.93), (6.55, 0.95))
ax.text(0.1, 2.95, "Additive attention gate", fontsize=9.5, fontweight="bold")
save(fig, "d_attgate.png")

# ------------------------------------------------------------- 5. adversarial game
fig, ax = canvas(7.0, 3.1)
box(ax, 0.1, 2.1, 0.9, 0.6, "MRI patch $x$", "data")
box(ax, 1.4, 2.1, 1.2, 0.6, "Generator $G$\n(3D U-Net)", "net")
box(ax, 3.0, 2.1, 0.9, 0.6, "prediction\n$\\hat{y}=\\sigma(G(x))$", "grey", fs=7.5)
box(ax, 3.0, 0.4, 0.9, 0.6, "ground truth\n$y$", "grey", fs=8)
box(ax, 4.5, 1.2, 1.25, 0.8, "Discriminator $D$\n3D PatchGAN\nspectral norm", "net", fs=8)
box(ax, 6.1, 1.35, 0.8, 0.5, "patch\nscores", "eval", fs=8)
arrow(ax, (1.0, 2.4), (1.4, 2.4)); arrow(ax, (2.6, 2.4), (3.0, 2.4)); arrow(ax, (3.9, 2.4), (4.5, 1.8)); arrow(ax, (3.9, 0.7), (4.5, 1.35)); arrow(ax, (5.75, 1.6), (6.1, 1.6))
ax.plot([0.55, 0.55, 5.1], [2.1, 0.08, 0.08], color=C["blue"], ls="--", lw=1.2); arrow(ax, (5.1, 0.08), (5.1, 1.2), color=C["blue"], ls="--")
ax.text(0.7, 0.16, "image $x$ is also fed to $D$ (conditional discrimination)", fontsize=7.5, color=C["soft"])
ax.text(6.5, 2.35, "$D$: real $\\rightarrow$ 1, fake $\\rightarrow$ 0\n$G$: fool $D$ + segment", fontsize=8, color=C["soft"], ha="center")
ax.text(0.1, 2.9, "Conditional adversarial refinement", fontsize=9.5, fontweight="bold")
save(fig, "d_gan.png")

# ------------------------------------------------------- 6. evaluation protocol
fig, ax = canvas(7.4, 2.9)
ax.text(0.05, 2.7, "Patient-level, grade-stratified evaluation protocol (BraTS 2020, 369 cases)", fontsize=9.5, fontweight="bold")
ax.add_patch(Rectangle((0.1, 1.85), 5.6, 0.5, fc="#e8f0fb", ec=C["blue"])); ax.text(2.9, 2.1, "development pool (80%) — 5-fold cross-validation", ha="center", va="center", fontsize=8.5)
ax.add_patch(Rectangle((5.8, 1.85), 1.45, 0.5, fc="#fdebe4", ec=C["orange"])); ax.text(6.52, 2.1, "held-out test (20%)", ha="center", va="center", fontsize=8.5)
for k in range(5):
    x0 = 0.1 + k * 1.12
    for j in range(5):
        col = "#fff7e0" if j == k else "#ffffff"
        ax.add_patch(Rectangle((x0 + 0.04 + j * 0.04 - 0.0, 0.0), 0, 0))
for k in range(5):
    y = 1.35 - k * 0.27
    for j in range(5):
        ax.add_patch(Rectangle((0.1 + j * 1.12, y), 1.12, 0.22, fc="#fdf2cc" if j == k else "#e8f0fb", ec=C["grey"], lw=0.6))
        ax.text(0.1 + j * 1.12 + 0.56, y + 0.11, "validation (OOF)" if j == k else "train", ha="center", va="center", fontsize=6.5, color=C["ink"])
    ax.text(5.8, y + 0.11, f"model {k + 1}", fontsize=7, va="center")
arrow(ax, (6.35, 0.65), (6.85, 0.65), color=C["soft"])
ax.text(6.9, 0.65, "applied to test;\nmean over 5 models", fontsize=7, va="center")
ax.text(0.1, -0.1, "Every development case is segmented once by a model that never saw it (out-of-fold, OOF); the grade classifier is trained on OOF features only.", fontsize=7.5, color=C["soft"])
ax.set_ylim(-0.25, 2.9)
save(fig, "d_protocol.png")

# --------------------------------------------------------------- 7. label hierarchy
fig, ax = canvas(5.2, 2.9)
ax.add_patch(Ellipse((1.5, 1.45), 2.8, 2.4, fc="#fdf2cc", ec=C["yellow"], lw=1.3))
ax.add_patch(Ellipse((1.55, 1.4), 1.9, 1.6, fc="#e1ecfa", ec=C["blue"], lw=1.3))
ax.add_patch(Ellipse((1.6, 1.35), 0.9, 0.8, fc="#fbd9d6", ec=C["red"], lw=1.3))
ax.text(0.75, 2.3, "WT", fontsize=10, fontweight="bold"); ax.text(1.0, 1.95, "TC", fontsize=10, fontweight="bold", color=C["blue"]); ax.text(1.45, 1.32, "ET", fontsize=10, fontweight="bold", color=C["red"])
ax.text(3.1, 2.25, "edema (label 2)\n$\\in$ WT only", fontsize=8.5, va="center"); ax.text(3.1, 1.5, "necrotic / non-enhancing\n(label 1)  $\\in$ TC, WT", fontsize=8.5, va="center"); ax.text(3.1, 0.75, "enhancing tumour (label 4)\n$\\in$ ET, TC, WT", fontsize=8.5, va="center")
ax.text(0.1, 0.15, "WT = {1,2,4}   TC = {1,4}   ET = {4}   (nested: $ET\subset TC\subset WT$)", fontsize=8.5, color=C["soft"])
save(fig, "d_regions.png")
print("diagrams ok")
