"""Schematics of existing methods for the literature-review chapter.

Every figure is an ORIGINAL redrawing made for this thesis from the description in the cited paper;
it is not a copy of the published figure.  Captions state 'redrawn by the author after [ref]'.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import C, plt, setup
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Polygon, Circle
setup()
FIG = Path(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "thesis" / "assets" / "figs")); FIG.mkdir(parents=True, exist_ok=True)
FILL = {"data": "#e8f0fb", "net": "#fdebe4", "loss": "#e4f5ee", "eval": "#f0eefb", "grey": "#f1f1ef", "att": "#fff7e0"}
EDGE = {"data": C["blue"], "net": C["orange"], "loss": C["aqua"], "eval": C["violet"], "grey": C["grey"], "att": C["yellow"]}


def box(ax, x, y, w, h, text, kind="grey", fs=8, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05", fc=FILL[kind], ec=EDGE[kind], lw=1.2, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=C["ink"])


def arrow(ax, p, q, color=None, style="-|>", lw=1.2, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, lw=lw, color=color or C["soft"], connectionstyle=f"arc3,rad={rad}", ls=ls))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off"); ax.grid(False)
    return fig, ax


def slab(ax, x, ymid, w, h, kind="net", d=0.0, label=None, fs=6.5):
    """Feature-map stack: width ~ channels, height ~ resolution; d > 0 adds an oblique 3-D depth."""
    y = ymid - h / 2
    ax.add_patch(Rectangle((x, y), w, h, fc=FILL[kind], ec=EDGE[kind], lw=1.0))
    if d > 0:
        ax.add_patch(Polygon([(x, y + h), (x + d, y + h + d), (x + w + d, y + h + d), (x + w, y + h)], fc=FILL[kind], ec=EDGE[kind], lw=1.0))
        ax.add_patch(Polygon([(x + w, y), (x + w + d, y + d), (x + w + d, y + h + d), (x + w, y + h)], fc=FILL[kind], ec=EDGE[kind], lw=1.0))
    if label:
        ax.text(x + w / 2, y - 0.12, label, ha="center", va="top", fontsize=fs, color=C["soft"])


def save(fig, name):
    fig.savefig(FIG / name); plt.close(fig)


def unet_like(name, title, chans, bottom, three_d, in_lab, out_lab, conv_txt, pool_txt, up_txt, extra=None, skip_lab="copy"):
    fig, ax = canvas(7.6, 5.0)
    ax.text(0.05, 4.82, title, fontsize=10, fontweight="bold")
    ax.text(0.05, 4.55, conv_txt, fontsize=7, color=C["soft"])
    n = len(chans)
    H = [1.15, 0.85, 0.62, 0.45]
    YC = [3.55, 2.7, 1.98, 1.35]
    W = lambda c: 0.14 + 0.0006 * c
    STEP = 0.7
    xe = [0.5 + STEP * i for i in range(n)]
    xb = 0.5 + STEP * n
    xd = [xb + STEP * (n - i) for i in range(n)]
    d = 0.08 if three_d else 0.0
    for i in range(n):
        slab(ax, xe[i], YC[i], W(chans[i]), H[i], "net", d, f"{chans[i]}")
        slab(ax, xd[i], YC[i], W(chans[i]), H[i], "net", d, f"{chans[i]}")
        y0 = YC[i] + 0.05
        arrow(ax, (xe[i] + W(chans[i]) + d + 0.03, y0), (xd[i] - 0.03, y0), color=C["grey"], ls="--", lw=1.0)
        if i == 0:
            ax.text((xe[0] + xd[0]) / 2 + 0.3, y0 + 0.08, skip_lab, fontsize=7, color=C["soft"], ha="center")
    ybot = 0.85
    slab(ax, xb, ybot, W(bottom), 0.38, "net", d, f"{bottom}")
    for i in range(n - 1):
        arrow(ax, (xe[i] + W(chans[i]) + d + 0.03, YC[i] - H[i] * 0.3), (xe[i + 1] - 0.03, YC[i + 1] + H[i + 1] * 0.3), color=C["orange"], lw=1.2)
        arrow(ax, (xd[i + 1] + W(chans[i + 1]) + d + 0.03, YC[i + 1] + H[i + 1] * 0.3), (xd[i] - 0.03, YC[i] - H[i] * 0.3), color=C["aqua"], lw=1.2)
    arrow(ax, (xe[-1] + W(chans[-1]) + d + 0.03, YC[-1] - H[-1] * 0.3), (xb - 0.03, ybot + 0.1), color=C["orange"], lw=1.2)
    arrow(ax, (xb + W(bottom) + d + 0.03, ybot + 0.1), (xd[-1] - 0.03, YC[-1] - H[-1] * 0.3), color=C["aqua"], lw=1.2)
    ax.text(0.05, YC[0] - H[0] / 2 - 0.5, in_lab, fontsize=6.8, color=C["soft"], va="top")
    ax.text(xd[0] + W(chans[0]) + d + 0.1, YC[0], out_lab, fontsize=7, color=C["soft"], va="center")
    leg = [(C["orange"], pool_txt, "-"), (C["aqua"], up_txt, "-"), (C["grey"], "skip connection (feature concatenation)", "--")]
    for k, (col, t, ls_) in enumerate(leg):
        ax.plot([0.05, 0.4], [0.62 - 0.2 * k] * 2, color=col, lw=1.6, ls=ls_)
        ax.text(0.5, 0.62 - 0.2 * k, t, fontsize=7, va="center", color=C["soft"])
    if extra:
        ax.text(4.9, 0.05, extra, fontsize=7, va="bottom", color=C["soft"])
    ax.set_ylim(0.0, 5.0)
    save(fig, name)


# 1. U-Net (Ronneberger et al.)
unet_like("f_unet2d.png", "U-Net: contracting path, expanding path and skip connections (2-D)", [64, 128, 256, 512], 1024, False,
          "input image\n(1 channel)", "1x1 conv\n$\\rightarrow$ class\nscores", "Each block is a pair of 3x3 convolutions followed by ReLU; numbers give the channel count of the block.",
          "2x2 max-pooling, stride 2 (channels double)", "2x2 up-convolution (channels halve)",
          extra="Convolutions are unpadded, so the contracted\nfeature maps are cropped before concatenation.")

# 2. 3-D U-Net (Cicek et al.)
unet_like("f_unet3d.png", "3-D U-Net: the same layout with volumetric operators", [64, 128, 256, 512], 512, True,
          "input volume\n(1 channel)", "1x1x1 conv\n$\\rightarrow$ labels", "Each block is two 3x3x3 convolutions with batch normalisation and ReLU; numbers give the channel count (simplified).",
          "2x2x2 max-pooling, stride 2", "2x2x2 up-convolution, stride 2",
          extra="The number of channels is doubled before pooling\nso that the bottleneck is not too narrow;\ntrained from sparsely annotated volumes.")

# 3. Attention U-Net gate
fig, ax = canvas(7.6, 3.5)
ax.text(0.05, 3.3, "Attention gate of Attention U-Net (grid attention)", fontsize=10, fontweight="bold")
box(ax, 0.1, 2.15, 1.0, 0.55, "skip features\n$x^l$ (fine grid)", "data", fs=7.5)
box(ax, 0.1, 0.45, 1.0, 0.55, "gating signal $g$\n(coarser grid)", "data", fs=7.5)
box(ax, 1.55, 2.15, 1.0, 0.55, "$W_x$: 1x1x1 conv\nstride 2", "net", fs=7.5)
box(ax, 1.55, 0.45, 1.0, 0.55, "$W_g$: 1x1x1 conv", "net", fs=7.5)
ax.add_patch(Circle((3.05, 1.4), 0.17, fc="white", ec=C["soft"])); ax.text(3.05, 1.4, "+", ha="center", va="center", fontsize=12)
box(ax, 3.4, 1.15, 0.75, 0.5, "ReLU\n$\\sigma_1$", "net", fs=7.5)
box(ax, 4.4, 1.15, 0.95, 0.5, "$\\psi$: 1x1x1 conv\nsigmoid $\\sigma_2$", "net", fs=7)
box(ax, 5.6, 1.15, 0.95, 0.5, "trilinear\nresampling", "eval", fs=7.5)
ax.add_patch(Circle((7.0, 2.4), 0.17, fc="white", ec=C["soft"])); ax.text(7.0, 2.4, "×", ha="center", va="center", fontsize=12)
box(ax, 6.55, 0.55, 0.95, 0.5, "gated features\n$\\hat{x}^l=\\alpha\\, x^l$", "att", fs=7)
arrow(ax, (1.1, 2.42), (1.55, 2.42)); arrow(ax, (1.1, 0.72), (1.55, 0.72)); arrow(ax, (2.55, 2.42), (2.92, 1.5)); arrow(ax, (2.55, 0.72), (2.92, 1.3))
arrow(ax, (3.22, 1.4), (3.4, 1.4)); arrow(ax, (4.15, 1.4), (4.4, 1.4)); arrow(ax, (5.35, 1.4), (5.6, 1.4)); arrow(ax, (6.55, 1.55), (6.93, 2.28))
ax.plot([0.6, 0.6, 7.0], [2.7, 3.02, 3.02], color=C["grey"], lw=1.2); arrow(ax, (7.0, 3.02), (7.0, 2.58), color=C["grey"]); arrow(ax, (7.0, 2.22), (7.02, 1.05))
ax.text(5.6, 0.2, "coefficients $\\alpha\\in[0,1]$ at the fine grid", fontsize=7, color=C["soft"])
save(fig, "f_attn_gate.png")

# 4. SegAN
fig, ax = canvas(7.6, 3.7)
ax.text(0.05, 3.5, "SegAN: segmentor, critic and the multi-scale L1 objective", fontsize=10, fontweight="bold")
box(ax, 0.1, 2.0, 0.9, 0.6, "image $x$", "data")
box(ax, 1.35, 2.0, 1.25, 0.6, "Segmentor $S$\n(encoder-decoder\nwith skips)", "net", fs=7.5)
box(ax, 2.95, 2.0, 0.95, 0.6, "predicted mask\n$S(x)$", "grey", fs=7.5)
box(ax, 2.95, 0.5, 0.95, 0.6, "reference\nmask $y$", "grey", fs=7.5)
box(ax, 4.25, 2.0, 1.0, 0.6, "$x\\circ S(x)$", "data", fs=8.5)
box(ax, 4.25, 0.5, 1.0, 0.6, "$x\\circ y$", "data", fs=8.5)
box(ax, 5.6, 1.1, 1.0, 1.0, "Critic $C$\n(shared weights)\nfeature maps\nat $L$ layers", "net", fs=7.2)
box(ax, 6.85, 1.25, 0.65, 0.7, "mean abs.\nerror over\nlayers", "loss", fs=6.8)
arrow(ax, (1.0, 2.3), (1.35, 2.3)); arrow(ax, (2.6, 2.3), (2.95, 2.3)); arrow(ax, (3.9, 2.3), (4.25, 2.3)); arrow(ax, (3.9, 0.8), (4.25, 0.8))
arrow(ax, (5.25, 2.3), (5.6, 1.85)); arrow(ax, (5.25, 0.8), (5.6, 1.35)); arrow(ax, (6.6, 1.6), (6.85, 1.6))
ax.plot([0.55, 0.55, 4.75], [2.6, 3.12, 3.12], color=C["blue"], ls="--", lw=1.1); arrow(ax, (4.75, 3.12), (4.75, 2.62), color=C["blue"], ls="--")
ax.plot([0.55, 0.55, 4.75], [2.0, 0.28, 0.28], color=C["blue"], ls="--", lw=1.1); arrow(ax, (4.75, 0.28), (4.75, 0.5), color=C["blue"], ls="--")
ax.text(2.6, 3.2, "the image multiplies both masks element-wise", fontsize=7, color=C["soft"], ha="center")
ax.text(0.1, -0.12, "$S$ minimises and $C$ maximises the same objective, $\\mathcal{L}=\\frac{1}{N}\\sum_n \\frac{1}{L}\\sum_{i=1}^{L}\\left\\|f_C^{\\,i}(x_n\\circ S(x_n))-f_C^{\\,i}(x_n\\circ y_n)\\right\\|_1$.", fontsize=8, color=C["ink"])
ax.set_ylim(-0.3, 3.7)
save(fig, "f_segan.png")

# 5. nnU-Net
fig, ax = canvas(7.6, 3.8)
ax.text(0.05, 3.6, "nnU-Net: configuration of a segmentation pipeline from the data set", fontsize=10, fontweight="bold")
box(ax, 0.1, 1.6, 1.3, 1.2, "Training cases\n\n$\\rightarrow$ data set\nfingerprint:\nimage size, spacing,\nintensity statistics,\nclass ratios", "data", fs=6.8)
col = [("Fixed parameters", ["loss: Dice + cross-entropy", "optimiser, learning-rate schedule", "data augmentation recipe", "deep supervision", "normalisation: instance, leaky ReLU"], 1.75, "grey"),
       ("Rule-based parameters", ["resampling, intensity normalisation", "patch size and batch size", "network topology (depth, pooling)", "low-resolution configuration", "(chosen by rules from the fingerprint)"], 3.7, "net"),
       ("Empirical parameters", ["cross-validation of 2-D, 3-D full-res.", "and cascade candidates", "selection of the best configuration", "or ensemble", "post-processing (largest component)"], 5.65, "eval")]
for t, lines, x, k in col:
    box(ax, x, 1.0, 1.8, 2.3, "", k)
    ax.text(x + 0.9, 3.12, t, ha="center", fontsize=8, fontweight="bold")
    for j, l in enumerate(lines):
        ax.text(x + 0.9, 2.78 - 0.4 * j, l, ha="center", va="center", fontsize=6.4, color=C["ink"])
arrow(ax, (1.4, 2.2), (1.75, 2.2)); arrow(ax, (3.55, 2.2), (3.7, 2.2)); arrow(ax, (5.5, 2.2), (5.65, 2.2))
box(ax, 5.65, 0.1, 1.8, 0.55, "predictions on new cases", "data", fs=7.5); arrow(ax, (6.55, 1.0), (6.55, 0.65))
ax.text(0.1, 0.55, "No manual design step: the same recipe is applied to every data set\nand the rules adapt the architecture to the image geometry.", fontsize=7, color=C["soft"], va="center")
save(fig, "f_nnunet.png")

# 6. Swin UNETR
fig, ax = canvas(7.6, 4.0)
ax.text(0.05, 3.8, "Swin UNETR: shifted-window transformer encoder with a convolutional decoder", fontsize=10, fontweight="bold")
box(ax, 0.05, 2.35, 0.8, 0.75, "4-channel\n3-D input", "data", fs=7.2)
box(ax, 1.0, 2.35, 0.8, 0.75, "patch\npartition,\nembedding", "net", fs=6.8)
xs = [2.0, 3.45, 4.9, 6.35]
for i, x in enumerate(xs):
    box(ax, x, 2.3, 1.05, 0.85, f"Swin stage {i+1}\n(W-MSA,\nSW-MSA)", "net", fs=6.6)
    if i < 3:
        arrow(ax, (x + 1.05, 2.72), (xs[i + 1], 2.72)); ax.text(x + 1.25, 2.84, "patch\nmerging", fontsize=5.5, color=C["soft"], ha="center", va="bottom")
arrow(ax, (0.85, 2.72), (1.0, 2.72)); arrow(ax, (1.8, 2.72), (2.0, 2.72))
for i, x in enumerate(xs):
    box(ax, x, 0.95, 1.05, 0.7, "residual conv\nblock" + ("" if i == 3 else "\n+ up-conv"), "loss", fs=6.4)
    arrow(ax, (x + 0.525, 2.3), (x + 0.525, 1.65), color=C["grey"], ls="--", lw=1.0)
    if i > 0:
        arrow(ax, (x, 1.3), (xs[i - 1] + 1.05, 1.3))
box(ax, 0.6, 0.95, 1.2, 0.7, "residual conv\nblock (input)", "loss", fs=6.4)
arrow(ax, (0.45, 2.35), (0.8, 1.65), color=C["grey"], ls="--", lw=1.0)
arrow(ax, (2.0, 1.3), (1.8, 1.3))
box(ax, 0.3, 0.12, 1.7, 0.4, "1x1x1 conv, sigmoid (3 channels)", "data", fs=6.6); arrow(ax, (1.2, 0.95), (1.2, 0.52))
ax.text(7.45, 0.55, "Dashed arrows: output of each Swin stage passes through a residual\nconvolution block to the decoder level of matching resolution;\nthe last stage feeds the decoder directly.", fontsize=6.4, color=C["soft"], ha="right", va="center")
save(fig, "f_swinunetr.png")
print("literature figures done")
