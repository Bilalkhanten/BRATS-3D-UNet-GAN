"""Figures for the methodology chapter. Data-driven panels use the real code of the package
(normalise_volume, PatchStream, augment, gaussian_importance, metrics, Discriminator3D) on synthetic phantoms.
usage: python diagrams_meth.py OUT_DIR
"""
import itertools
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np
import torch
from scipy import ndimage as ndi
from figstyle import C, plt, setup
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle, Polygon

setup()
FIG = Path(sys.argv[1] if len(sys.argv) > 1 else HERE.parent / "thesis" / "assets" / "figs"); FIG.mkdir(parents=True, exist_ok=True)
FILL = {"data": "#e8f0fb", "net": "#fdebe4", "loss": "#e4f5ee", "eval": "#f0eefb", "grey": "#f1f1ef", "att": "#fff7e0"}
EDGE = {"data": C["blue"], "net": C["orange"], "loss": C["aqua"], "eval": C["violet"], "grey": C["grey"], "att": C["yellow"]}
STATS = {}


def box(ax, x, y, w, h, text, kind="grey", fs=8, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05", fc=FILL[kind], ec=EDGE[kind], lw=1.2, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=C["ink"])


def arrow(ax, p, q, color=None, style="-|>", lw=1.2, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, lw=lw, color=color or C["soft"], connectionstyle=f"arc3,rad={rad}", ls=ls))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off"); ax.grid(False)
    return fig, ax


def save(fig, name):
    fig.savefig(FIG / name); plt.close(fig)


from bratsgan.synthetic import make_phantom  # noqa: E402
from bratsgan.data import normalise_volume, PatchStream, labels_to_regions  # noqa: E402


def contour(ax, m, color, lw=1.0):
    if m.any():
        ax.contour(m.astype(float), levels=[0.5], colors=[color], linewidths=lw)


# ===================================================================== 1. preprocessing
def fig_preproc():
    vols, lab, brain = make_phantom(64, "HGG", 11)
    pad = np.zeros((4, 96, 96, 96), np.float32); L = np.zeros((96, 96, 96), np.uint8)
    off = (14, 20, 16)
    sl = tuple(slice(o, o + 64) for o in off)
    pad[(slice(None),) + sl] = vols * 600 + 40 * (vols > 0)   # arbitrary scanner scale
    L[sl] = lab
    br = (pad != 0).any(0)
    idx = np.argwhere(br); lo = np.maximum(idx.min(0) - 4, 0); hi = np.minimum(idx.max(0) + 5, br.shape)
    z = int(np.argmax((L > 0).sum((1, 2))))
    fig = plt.figure(figsize=(7.4, 4.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.38, wspace=0.32)
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(pad[3, z], cmap="gray"); ax.add_patch(Rectangle((lo[2], lo[1]), hi[2] - lo[2], hi[1] - lo[1], fill=False, ec=C["red"], lw=1.4, ls="--"))
    ax.set_title("(a) FLAIR as stored, with the crop box", fontsize=8.5); ax.axis("off"); ax.grid(False)
    ax.text(lo[2], lo[1] - 2, "non-zero box + 4 voxels", color=C["red"], fontsize=6.5, va="bottom")
    crop = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    fl = pad[3][crop]; bc = br[crop]; zc = z - int(lo[0])
    nm = normalise_volume(fl, bc)
    ax = fig.add_subplot(gs[0, 1]); ax.imshow(fl[zc], cmap="gray"); ax.set_title("(b) cropped volume, raw intensities", fontsize=8.5); ax.axis("off"); ax.grid(False)
    ax = fig.add_subplot(gs[0, 2]); ax.imshow(nm[zc], cmap="gray", vmin=-3, vmax=3); contour(ax, L[crop][zc] > 0, C["yellow"], 0.9)
    ax.set_title("(c) after clipping and z-score", fontsize=8.5); ax.axis("off"); ax.grid(False)
    v = fl[bc]; lo_p, hi_p = np.percentile(v, (0.5, 99.5)); vc = np.clip(v, lo_p, hi_p); zs = (vc - vc.mean()) / (vc.std() + 1e-8)
    for k, (arr, ttl, xl) in enumerate(((v, "raw (in-brain voxels)", "intensity"), (vc, "clipped to the 0.5-99.5 percentile range", "intensity"), (zs, "z-scored (mean 0, s.d. 1)", "z-score"))):
        ax = fig.add_subplot(gs[1, k]); ax.hist(arr, bins=60, color=C["blue"], alpha=0.85)
        ax.set_title(f"({'def'[k]}) {ttl}", fontsize=8.5); ax.set_xlabel(xl, fontsize=8); ax.tick_params(labelsize=7)
        if k == 0:
            for p_, nm_ in ((lo_p, "0.5 %"), (hi_p, "99.5 %")):
                ax.axvline(p_, color=C["red"], ls="--", lw=1.0); ax.text(p_, ax.get_ylim()[1] * 0.92, nm_, color=C["red"], fontsize=6.5, ha="center")
        if k == 0:
            ax.set_ylabel("voxel count", fontsize=8)
    STATS["preproc"] = dict(raw_shape=[96, 96, 96], crop_shape=[int(h - l) for l, h in zip(lo, hi)], pct_lo=float(lo_p), pct_hi=float(hi_p),
                            z_mean=float(zs.mean()), z_std=float(zs.std()), n_clipped=int(((v < lo_p) | (v > hi_p)).sum()), n_brain=int(v.size))
    save(fig, "m_preproc.png")


# ===================================================================== 2. patch sampling and augmentation
def fig_patch():
    torch.manual_seed(0)
    tmp = Path(tempfile.mkdtemp())
    ids = []
    for k in range(6):
        vols, lab, brain = make_phantom(96, "HGG" if k % 4 else "LGG", 100 + k)
        cid = f"c{k}"; ids.append(cid)
        img = np.stack([normalise_volume(vols[m], brain) for m in range(4)])
        np.save(tmp / f"{cid}_img.npy", img.astype(np.float16)); np.save(tmp / f"{cid}_seg.npy", lab)
        np.save(tmp / f"{cid}_fg.npy", np.argwhere(lab > 0).astype(np.int32))
        (tmp / f"{cid}_meta.json").write_text(json.dumps({"id": cid, "grade": "HGG", "shape": list(lab.shape)}))
    P = (48, 48, 48)
    probs = [0.0, 0.33, 0.67, 1.0]
    frac_any, frac_vox = [], []
    for p in probs:
        st = PatchStream(tmp, ids, P, fg_prob=p, seed=3)
        n_any, fv = 0, []
        for x, y, g in itertools.islice(iter(st), 500):
            t = y[0].sum().item()
            n_any += t > 0; fv.append(t / np.prod(P))
        frac_any.append(n_any / 500); frac_vox.append(float(np.mean(fv)))
    STATS["patch"] = dict(p=probs, frac_with_tumour=frac_any, mean_tumour_voxel_fraction=frac_vox, patch=list(P), volume=[96, 96, 96])
    # panel (a): illustration of the sampling rule on one case
    rng = np.random.default_rng(5)
    lab = np.load(tmp / "c1_seg.npy"); fg = np.load(tmp / "c1_fg.npy")
    z = int(np.argmax((lab > 0).sum((1, 2))))
    img = np.load(tmp / "c1_img.npy").astype(np.float32)
    fig = plt.figure(figsize=(7.4, 3.9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1], hspace=0.45, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0:2]); ax.imshow(img[3, z], cmap="gray"); contour(ax, lab[z] > 0, C["yellow"], 1.0)
    for k in range(14):
        if rng.random() < 0.67:
            c = fg[rng.integers(len(fg))] + rng.integers(-16, 17, 3); col = C["red"]
        else:
            c = np.array([rng.integers(0, s) for s in lab.shape]); col = C["blue"]
        lo = np.clip(c - 24, -12, 96 - 36)
        ax.add_patch(Rectangle((lo[2], lo[1]), 48, 48, fill=False, ec=col, lw=0.9, alpha=0.9))
    ax.set_title("(a) patch footprints (axial projection): tumour-centred (red), uniform (blue)", fontsize=7.6); ax.axis("off"); ax.grid(False)
    ax = fig.add_subplot(gs[0, 2])
    xs = np.arange(len(probs))
    ax.bar(xs, frac_any, 0.6, color=C["blue"])
    for i in range(len(probs)):
        ax.text(xs[i], frac_any[i] + 0.02, f"{frac_any[i]:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(xs); ax.set_xticklabels([f"{p:.2f}" for p in probs], fontsize=7); ax.set_xlabel("$p_{fg}$", fontsize=8)
    ax.set_ylim(0, 1.15); ax.set_title("(b) patches containing\ntumour", fontsize=7.6)
    ax = fig.add_subplot(gs[0, 3])
    ax.bar(xs, [100 * v for v in frac_vox], 0.6, color=C["orange"])
    for i in range(len(probs)):
        ax.text(xs[i], 100 * frac_vox[i] + 0.4, f"{100 * frac_vox[i]:.1f}", ha="center", fontsize=6.5)
    ax.set_xticks(xs); ax.set_xticklabels([f"{p:.2f}" for p in probs], fontsize=7); ax.set_xlabel("$p_{fg}$", fontsize=8)
    ax.set_ylim(0, 19); ax.set_title("(c) mean tumour share of a\npatch (%), 500 patches each", fontsize=7.6)
    # panel (c): augmentation on one patch
    from bratsgan.augment import random_flip, random_affine, intensity_aug
    st = PatchStream(tmp, ["c1"], P, fg_prob=1.0, seed=8)
    for x, y, g in iter(st):
        cen = np.argwhere(y[0].numpy() > 0)
        if len(cen) and np.abs(cen.mean(0)[1:] - 24).max() < 5 and y[0].sum() > 600:
            break
    x, y = x[None], y[None]
    torch.manual_seed(2)
    xf, yf = random_flip(x, y)
    xa, ya = random_affine(x, y, p_rot=1.0, p_scale=1.0)
    xi = intensity_aug(x, 1.0, 1.0, 1.0, 1.0)
    zz = int(y[0, 2].sum((1, 2)).argmax()) if y[0, 2].sum() > 0 else int(y[0, 0].sum((1, 2)).argmax())
    for j, (xx, yy, t) in enumerate(((x, y, "(d) original"), (xf, yf, "random flips"), (xa, ya, "rotation and scaling"), (xi, y, "intensity changes"))):
        ax = fig.add_subplot(gs[1, j]); ax.imshow(xx[0, 1, zz], cmap="gray"); contour(ax, yy[0, 0, zz].numpy() > 0, C["yellow"], 0.9); contour(ax, yy[0, 2, zz].numpy() > 0, C["red"], 0.9)
        ax.set_title(t, fontsize=7.5); ax.axis("off"); ax.grid(False)
    save(fig, "m_patch.png")


# ===================================================================== 3. building blocks
def fig_blocks():
    fig, ax = canvas(7.4, 4.3)
    ax.text(0.05, 4.12, "Building blocks of the generator", fontsize=10, fontweight="bold")
    ax.text(0.05, 3.75, "(a) plain block (variant A0)", fontsize=8.5, fontweight="bold")
    y = 3.05
    box(ax, 0.1, y, 0.7, 0.5, "input\n$C_{in}$", "data", fs=7)
    box(ax, 1.05, y, 1.5, 0.5, "conv 3x3x3\n(stride $s$) $\\rightarrow$ IN\n$\\rightarrow$ LeakyReLU", "net", fs=6.6)
    box(ax, 2.8, y, 1.5, 0.5, "conv 3x3x3\n$\\rightarrow$ IN $\\rightarrow$\nLeakyReLU", "net", fs=6.6)
    box(ax, 4.55, y, 0.7, 0.5, "output\n$C_{out}$", "data", fs=7)
    for a, b in (((0.8, y + 0.25), (1.05, y + 0.25)), ((2.55, y + 0.25), (2.8, y + 0.25)), ((4.3, y + 0.25), (4.55, y + 0.25))):
        arrow(ax, a, b)
    ax.text(5.45, y + 0.25, "the first convolution of\nevery encoder level $\\geq$ 1\nhas stride $s=2$", fontsize=6.8, color=C["soft"], va="center")
    ax.text(0.05, 2.5, "(b) residual block (variants A1-A5)", fontsize=8.5, fontweight="bold")
    y = 1.05
    box(ax, 0.1, y + 0.1, 0.7, 0.5, "input\n$C_{in}$", "data", fs=7)
    box(ax, 1.05, y + 0.6, 1.5, 0.5, "conv 3x3x3 (stride $s$)\n$\\rightarrow$ IN $\\rightarrow$ LeakyReLU", "net", fs=6.4)
    box(ax, 2.8, y + 0.6, 1.5, 0.5, "conv 3x3x3\n$\\rightarrow$ IN", "net", fs=6.6)
    box(ax, 1.05, y - 0.4, 3.25, 0.5, "shortcut: identity if $C_{in}=C_{out}$ and $s=1$,\notherwise 1x1x1 conv (stride $s$) $\\rightarrow$ IN", "grey", fs=6.4)
    ax.add_patch(Circle((4.65, y + 0.35), 0.17, fc="white", ec=C["soft"])); ax.text(4.65, y + 0.35, "+", ha="center", va="center", fontsize=11)
    box(ax, 5.0, y + 0.1, 0.9, 0.5, "LeakyReLU\n(slope 0.01)", "net", fs=6.6)
    box(ax, 6.15, y + 0.1, 0.7, 0.5, "output\n$C_{out}$", "data", fs=7)
    arrow(ax, (0.8, y + 0.35), (1.05, y + 0.85)); arrow(ax, (2.55, y + 0.85), (2.8, y + 0.85)); arrow(ax, (4.3, y + 0.85), (4.55, y + 0.45))
    arrow(ax, (0.8, y + 0.25), (1.05, y - 0.15)); arrow(ax, (4.3, y - 0.15), (4.55, y + 0.25)); arrow(ax, (4.82, y + 0.35), (5.0, y + 0.35)); arrow(ax, (5.9, y + 0.35), (6.15, y + 0.35))
    ax.text(0.05, 0.1, "IN = instance normalisation (affine). Every convolution is bias-free because the normalisation follows it.\nDecoder blocks use the same block with stride 1 on the concatenation of the up-sampled features and the (gated) skip features.", fontsize=6.8, color=C["soft"])
    save(fig, "m_blocks.png")


# ===================================================================== 4. deep supervision and region outputs
def fig_deepsup():
    fig, ax = canvas(7.4, 3.6)
    ax.text(0.05, 3.42, "Deep supervision and region-based outputs", fontsize=10, fontweight="bold")
    for i, (nm, wt, res) in enumerate((("decoder level 0", "4/7", "full resolution"), ("decoder level 1", "2/7", "1/2 resolution"), ("decoder level 2", "1/7", "1/4 resolution"))):
        y = 2.35 - 0.8 * i
        box(ax, 0.1, y, 1.3, 0.55, nm, "net", fs=7.5)
        box(ax, 1.7, y, 1.35, 0.55, f"1x1x1 conv\n3 logits ({res})", "net", fs=6.6)
        box(ax, 3.35, y, 1.5, 0.55, "target: region masks\n(nearest-neighbour\ndown-sampling)" if i else "target: region masks\n(WT, TC, ET)", "data", fs=6.2)
        box(ax, 5.15, y, 1.15, 0.55, "$L_{seg}$ = Dice\n+ Tversky + BCE", "loss", fs=6.4)
        ax.text(6.75, y + 0.28, f"weight\n{wt}", fontsize=8, ha="center", va="center", fontweight="bold")
        arrow(ax, (1.4, y + 0.28), (1.7, y + 0.28)); arrow(ax, (3.05, y + 0.28), (3.35, y + 0.28), style="-"); arrow(ax, (4.85, y + 0.28), (5.15, y + 0.28)); arrow(ax, (6.3, y + 0.28), (6.45, y + 0.28), style="-")
        ax.annotate("", xy=(3.2, y + 0.28), xytext=(3.2, y + 0.28))
    ax.plot([7.1, 7.1], [0.85, 2.9], color=C["grey"], lw=1.2); arrow(ax, (7.1, 1.85), (7.3, 1.85), style="-|>")
    ax.text(7.32, 1.85, "$\\sum$", fontsize=11, va="center")
    ax.text(0.1, 0.12, "At inference only the full-resolution head is used: sigmoid, threshold 0.5, then the projection $ET\\subset TC\\subset WT$\n(TC $\\leftarrow$ TC $\\wedge$ WT, ET $\\leftarrow$ ET $\\wedge$ TC). Weights $w_i=2^{-i}/\\sum_j 2^{-j}$.", fontsize=7, color=C["soft"])
    ax.set_xlim(0, 7.6)
    save(fig, "m_deepsup.png")


# ===================================================================== 5. discriminator
def fig_disc():
    from bratsgan.models import Discriminator3D
    D = Discriminator3D(4, 3, 32).eval()
    x = torch.zeros(1, 4, 128, 128, 128); s = torch.zeros(1, 3, 128, 128, 128)
    shapes, h = [], torch.cat([x, s], 1)
    shapes.append(("input", tuple(h.shape[1:])))
    convs = [m for m in D.net if hasattr(m, "weight") or hasattr(m, "weight_orig")]
    with torch.no_grad():
        for m in D.net:
            h = m(h)
            if not isinstance(m, torch.nn.LeakyReLU):
                shapes.append(("conv", tuple(h.shape[1:])))
    nparams = sum(p.numel() for p in D.parameters())
    # effective parameter count (spectral norm keeps weight_orig + u,v buffers): count conv weights only
    ncw = 0
    for m in D.net:
        if isinstance(m, torch.nn.Conv3d):
            w = m.weight_orig if hasattr(m, "weight_orig") else m.weight
            ncw += w.numel() + m.bias.numel()
    rf, jump = 1, 1
    for k, s_ in ((4, 2), (4, 2), (4, 2), (4, 1), (4, 1)):
        rf += (k - 1) * jump; jump *= s_
    STATS["disc"] = dict(shapes=[(n, list(s_)) for n, s_ in shapes], params=int(ncw), receptive_field=int(rf), n_patch_scores=int(np.prod(shapes[-1][1][1:])))
    fig, ax = canvas(7.4, 3.6)
    ax.text(0.05, 3.42, "Conditional 3-D PatchGAN discriminator ($128^3$ input patch)", fontsize=10, fontweight="bold")
    box(ax, 0.05, 1.5, 0.9, 1.0, "image $x$ (4 ch)\n+\nsegmentation\n(3 ch)\nconcatenated:\n7 ch", "data", fs=6.2)
    names = ["conv 4x4x4, s=2\nSN, LReLU(0.2)", "conv 4x4x4, s=2\nSN, LReLU(0.2)", "conv 4x4x4, s=2\nSN, LReLU(0.2)", "conv 4x4x4, s=1\nSN, LReLU(0.2)", "conv 4x4x4, s=1\nSN (no activation)"]
    for i in range(5):
        x0 = 1.2 + 1.2 * i
        hh = 1.35 - 0.2 * i
        box(ax, x0, 2.0 - hh / 2, 1.0, hh, names[i], "net", fs=5.8)
        sh = shapes[i + 1][1]
        ax.text(x0 + 0.5, 2.0 - hh / 2 - 0.17, f"{sh[0]} x {sh[1]}$^3$", ha="center", fontsize=7, color=C["ink"])
        arrow(ax, (x0 - 0.25 if i else 0.95, 2.0), (x0, 2.0))
    ax.text(7.35, 1.65, "patch\nscores", fontsize=7.5, ha="center", va="center", color=C["soft"])
    ax.text(0.05, 0.62, f"SN = spectral normalisation. Output: {int(np.prod(shapes[-1][1][1:])):,} patch scores ({shapes[-1][1][1]}$^3$), each with a receptive field of {rf}$^3$ voxels.\nTrainable convolution parameters: {ncw:,} (ndf = 32). A real patch pairs the image with the reference masks, a fake patch with the generator's probabilities.", fontsize=7, color=C["soft"])
    ax.set_ylim(0.3, 3.6)
    save(fig, "m_disc.png")


# ===================================================================== 6. schedule
def fig_schedule():
    from bratsgan.train import poly_lr
    from bratsgan.losses import adv_ramp
    E, Es, Ea, it, ramp = 350, 300, 50, 250, 5
    ep = np.arange(E)
    lr = [poly_lr(3e-4, e, E, 0.9) for e in ep]
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6), gridspec_kw={"wspace": 0.4})
    ax = axs[0]; ax.plot(ep, np.array(lr) * 1e4, color=C["blue"]); ax.axvspan(Es, E, color=C["orange"], alpha=0.12)
    ax.set_xlabel("epoch"); ax.set_ylabel("learning rate ($\\times 10^{-4}$)"); ax.set_title("(a) polynomial decay, power 0.9", fontsize=8.5)
    ax.text(Es + 2, 2.2, "adv.\nstage", fontsize=6.5, color=C["orange"])
    steps = np.arange(0, 12 * it)
    ax = axs[1]; ax.plot(steps / it, [0.05 * adv_ramp(int(s), ramp * it) for s in steps], color=C["orange"])
    ax.axvline(ramp, color=C["grey"], ls=":"); ax.set_xlabel("epochs since the start of the adversarial stage"); ax.set_ylabel("adversarial weight $\\lambda$")
    ax.set_title("(b) linear warm-up to $\\lambda_{max}=0.05$", fontsize=8.5)
    ax = axs[2]; ax.axis("off"); ax.grid(False)
    rows = [("A0-A4 (no GAN)", Es, Es + Ea, False), ("A5 and GAN variants", Es, Es + Ea, True)]
    for k, (nm, a, b, adv) in enumerate(rows):
        y = 0.62 - 0.4 * k
        ax.add_patch(Rectangle((0, y), a / E, 0.25, fc="#e8f0fb", ec=C["blue"])); ax.add_patch(Rectangle((a / E, y), (b - a) / E, 0.25, fc="#fdebe4" if adv else "#e8f0fb", ec=C["orange"] if adv else C["blue"], hatch="//" if adv else None))
        ax.text(0, y + 0.31, nm, fontsize=7.2)
    ax.text(a / E / 2, 0.745, "300 epochs: $G$ only", ha="center", fontsize=6.5, va="center"); ax.text(a / E / 2, 0.345, "300 epochs: $G$ only", ha="center", fontsize=6.5, va="center"); ax.text(0.5 * (1 + Es / E), 0.745 - 0.4, "$G{+}D$", ha="center", fontsize=6.3, va="center", color=C["orange"]); ax.text(0.5 * (1 + Es / E), 0.745, "$G$", ha="center", fontsize=6.3, va="center", color=C["blue"])
    ax.text(0.0, 0.0, "250 iterations per epoch, batch 2, 87,500\ngenerator updates in every variant", fontsize=7, color=C["soft"], va="center")
    ax.set_xlim(0, 1.0); ax.set_ylim(-0.12, 1.0); ax.set_title("(c) stages", fontsize=8.5)
    save(fig, "m_schedule.png")


# ===================================================================== 7. sliding window
def fig_window():
    from bratsgan.infer import gaussian_importance, _starts
    P, n = 128, 200
    w1 = gaussian_importance((P, 1, 1))[:, 0, 0].numpy()  # separable factor along one axis (the other factors are 1)
    st = _starts(n, P, 64)
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw={"wspace": 0.38, "width_ratios": [1, 1.25, 1]})
    ax = axs[0]; xs = np.arange(P); ax.plot(xs, w1, color=C["blue"]); ax.fill_between(xs, 0, w1, color=C["blue"], alpha=0.12)
    ax.axvline(P / 2 - 0.5 - P / 8, color=C["grey"], ls=":"); ax.axvline(P / 2 - 0.5 + P / 8, color=C["grey"], ls=":")
    ax.text(P / 2, 0.28, "$\\pm\\sigma$,\n$\\sigma=P/8$", ha="center", fontsize=7, color=C["soft"]); ax.set_xlabel("position in the window (voxels)"); ax.set_ylabel("importance weight")
    ax.set_title("(a) Gaussian importance", fontsize=8.5)
    ax = axs[1]; total = np.zeros(n); xx = np.arange(n)
    for k, s in enumerate(st):
        ax.fill_between(np.arange(s, s + P), k * 0.0, np.full(P, 0.0) + w1, alpha=0.0)
        seg = np.zeros(n); seg[s:s + P] = w1; ax.plot(xx, seg + 0.0, lw=1.0, color=[C["blue"], C["orange"], C["aqua"]][k % 3], label=f"window {k + 1} (start {s})"); total += seg
    ax.plot(xx, total, color=C["ink"], lw=1.6, ls="--", label="sum of weights"); ax.legend(fontsize=6.2, loc="upper left", ncol=1, frameon=False)
    ax.set_ylim(0, 2.6); ax.set_xlabel("voxel along one axis (length 200)"); ax.set_title("(b) overlap 0.5, last window flushed", fontsize=8.5)
    ax = axs[2]
    Lz = 200; steps = [_starts(Lz, P, 64)] * 2
    ax.set_xlim(-5, 205); ax.set_ylim(205, -5); ax.set_aspect("equal"); ax.grid(False)
    cols = [C["blue"], C["orange"], C["aqua"], C["violet"], C["red"], C["yellow"], C["grey"], C["magenta"], C["blue"]]
    k = 0
    for a in st:
        for b in st:
            ax.add_patch(Rectangle((b, a), P, P, fc=cols[k % 9], ec=cols[k % 9], lw=0.9, alpha=0.16)); k += 1
    ax.add_patch(Rectangle((0, 0), Lz, Lz, fill=False, ec=C["ink"], lw=1.4)); ax.set_xticks([0, 100, 200]); ax.set_yticks([0, 100, 200])
    ax.set_title(f"(c) {len(st) ** 2} windows in a slice (3-D: {len(st) ** 3})", fontsize=8.5)
    STATS["window"] = dict(P=P, n=n, starts=st, sum_min=float(total.min()), sum_max=float(total.max()), overlap=0.5)
    save(fig, "m_window.png")


# ===================================================================== 8. features
def fig_features():
    groups = [("Volume (log)", 5, "WT, TC, ET, NCR, ED"), ("Compartment ratios", 5, "ET/WT, ET/TC, TC/WT, NCR/TC, ED/WT"), ("Shape", 9, "sphericity, extent, max. diameter of WT, TC, ET"),
              ("Intensity", 80, "mean, s.d., 10th and 90th percentile of 4 sequences in 5 compartments"), ("Enhancement contrast", 2, "T1ce $-$ T1 in ET and TC")]
    fig, ax = plt.subplots(figsize=(7.4, 2.7))
    cols = [C["blue"], C["orange"], C["aqua"], C["violet"], C["yellow"]]
    left = 0
    for (n, k, d), c in zip(groups, cols):
        ax.barh(0, k, left=left, color=c, ec="white", height=0.55)
        ax.text(left + k / 2, 0.0, str(k), color="white" if k > 4 else C["ink"], ha="center", va="center", fontsize=8, fontweight="bold")
        left += k
    ax.set_xlim(0, 101); ax.set_ylim(-1.9, 0.55); ax.set_yticks([]); ax.set_xlabel("feature index (101 in total)"); ax.grid(False)
    for i, ((n, k, d), c) in enumerate(zip(groups, cols)):
        y = -0.55 - 0.27 * i
        ax.add_patch(Rectangle((0.5, y - 0.07), 2.2, 0.14, color=c)); ax.text(3.5, y, f"{n} ({k}): {d}", fontsize=7.2, va="center")
    ax.spines["left"].set_visible(False)
    save(fig, "m_features.png")
    assert sum(g[1] for g in groups) == 101


# ===================================================================== 9. metrics illustration
def fig_metrics():
    import bratsgan.metrics as M
    vols, lab, brain = make_phantom(64, "HGG", 21)
    gt = np.isin(lab, [1, 2, 4])
    pred = ndi.shift(gt.astype(float), (1.5, 2.0, 1.0), order=0) > 0.5
    pred = ndi.binary_dilation(pred, iterations=1)
    blob = np.zeros_like(gt); blob[50:54, 10:14, 10:14] = True
    pred = pred | blob
    d, i_, h95 = M.dice(pred, gt), M.iou(pred, gt), M.hd95(pred, gt)
    sp, sg = M._surface(pred), M._surface(gt)
    dg = ndi.distance_transform_edt(~sg); dp = ndi.distance_transform_edt(~sp)
    dist = np.concatenate([dg[sp], dp[sg]])
    hmax = float(dist.max())
    z = int(np.argmax(gt.sum((1, 2))))
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw={"wspace": 0.5, "width_ratios": [1, 1, 1.25]})
    ax = axs[0]; ax.imshow(vols[3][z], cmap="gray"); contour(ax, gt[z], C["aqua"], 1.3); contour(ax, pred[z], C["orange"], 1.3); ax.axis("off"); ax.grid(False)
    ax.set_title("(a) reference (green) and\nprediction (orange)", fontsize=8)
    ax = axs[1]
    tp = (pred & gt).sum(); fp = (pred & ~gt).sum(); fn = (~pred & gt).sum()
    ax.bar([0, 1, 2], [tp, fp, fn], color=[C["aqua"], C["orange"], C["red"]]); ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["TP", "FP", "FN"]); ax.set_ylabel("voxels")
    ax.set_title(f"(b) Dice = {d:.3f}, IoU = {i_:.3f}\n$2TP/(2TP+FP+FN)$", fontsize=8)
    ax = axs[2]; ax.hist(dist, bins=40, color=C["blue"], alpha=0.85)
    ax.axvline(h95, color=C["red"], ls="--"); ax.axvline(hmax, color=C["grey"], ls=":")
    ax.text(h95, ax.get_ylim()[1] * 0.85, f" HD95 = {h95:.1f}", color=C["red"], fontsize=7, ha="left"); ax.text(hmax, ax.get_ylim()[1] * 0.6, f"max = {hmax:.1f} ", color=C["soft"], fontsize=7, ha="right")
    ax.set_xlabel("directed surface distance (voxels)"); ax.set_ylabel("surface voxels"); ax.set_title("(c) pooled surface distances", fontsize=8)
    STATS["metrics"] = dict(dice=float(d), iou=float(i_), hd95=float(h95), hd_max=hmax, tp=int(tp), fp=int(fp), fn=int(fn), n_surface=int(dist.size))
    save(fig, "m_metrics.png")


# ===================================================================== 10. one adversarial iteration
def fig_iteration():
    fig, ax = canvas(7.4, 3.7)
    ax.text(0.05, 3.52, "One training iteration of the adversarial stage", fontsize=10, fontweight="bold")
    box(ax, 0.05, 2.65, 1.05, 0.6, "batch $(x, y)$\nafter augmentation", "data", fs=6.8)
    box(ax, 1.4, 2.65, 1.2, 0.6, "$G(x)$: logits\nof the 3 heads", "net", fs=7)
    box(ax, 2.9, 2.65, 1.55, 0.6, "$L_{seg}$ (Dice + Tversky\n+ BCE, deep supervision)", "loss", fs=6.6)
    box(ax, 5.1, 1.75, 2.1, 1.1, "$L=L_{seg}+\\lambda_t L_G$\ngradient clipping (12)\nAdamW update of $G$", "net", fs=7)
    box(ax, 1.4, 1.7, 1.2, 0.6, "$\\hat p=\\sigma(\\mathrm{head}_0)$", "grey", fs=7.5)
    box(ax, 2.9, 1.55, 1.55, 0.7, "(2) G step:\n$L_G=\\frac{1}{2}(D(x,\\hat p)-1)^2$", "loss", fs=6.8)
    box(ax, 0.05, 0.45, 1.05, 0.6, "smoothed target\n$\\tilde y=0.95\\,y+0.025$", "grey", fs=6.4)
    box(ax, 2.9, 0.3, 1.55, 0.95, "(1) D step (noise $\\sigma=0.05$):\n$D(x,\\tilde y)\\rightarrow 1$\n$D(x,\\hat p\\,\\mathrm{detached})\\rightarrow 0$", "net", fs=6.2)
    box(ax, 5.1, 0.45, 2.1, 0.6, "Adam update of $D$\n(lr $2\\times10^{-4}$, $\\beta_1=0.5$)", "net", fs=6.8)
    arrow(ax, (1.1, 2.95), (1.4, 2.95)); arrow(ax, (2.6, 2.95), (2.9, 2.95)); arrow(ax, (2.0, 2.65), (2.0, 2.3))
    ax.plot([4.45, 6.15], [2.95, 2.95], color=C["soft"], lw=1.2); arrow(ax, (6.15, 2.95), (6.15, 2.85))
    arrow(ax, (2.6, 2.0), (2.9, 1.95)); arrow(ax, (4.45, 1.9), (5.1, 2.1))
    arrow(ax, (2.2, 1.7), (2.9, 1.0)); arrow(ax, (1.1, 0.75), (2.9, 0.75)); arrow(ax, (4.45, 0.75), (5.1, 0.75))
    ax.text(0.05, 0.05, "$\\lambda_t=0.05\\cdot\\min(1,\\,t/5\\ \\mathrm{epochs})$. $L_{seg}$ reaches the generator directly; the adversarial term acts through $D$. Probabilities, not hard masks, are fed to $D$.", fontsize=6.6, color=C["soft"])
    save(fig, "m_iteration.png")


# ===================================================================== 11. parameters / shapes table data
def table_shapes():
    from bratsgan.models import UNet3D
    G = UNet3D(4, 3, 32, 5, 320, True, True, "maf", "instance", True, 0.0)
    rows = []
    cnt = lambda m: sum(p.numel() for p in m.parameters())
    rows.append(("MAF stem", "4 x $128^3$", f"32 x $128^3$", cnt(G.fusion)))
    res = [128, 64, 32, 16, 8]
    for i in range(5):
        cin = "32" if i == 0 else str(G.chs[i - 1])
        rows.append((f"Encoder level {i}", f"{cin} x ${res[max(i - 1, 0)] if i else 128}^3$", f"{G.chs[i]} x ${res[i]}^3$", cnt(G.enc[i])))
    for i in reversed(range(4)):
        g = cnt(G.gates[i])
        rows.append((f"Decoder level {i}", f"{2 * G.chs[i]} x ${res[i]}^3$ (concat.)", f"{G.chs[i]} x ${res[i]}^3$", cnt(G.up[i]) + cnt(G.dec[i]) + g))
    rows.append(("Output heads (3)", "levels 0, 1, 2", "3 x (full, 1/2, 1/4)", sum(cnt(h) for h in G.heads)))
    tot = cnt(G)
    STATS["shapes"] = dict(rows=[list(r) for r in rows], total=int(tot), gates=[int(cnt(G.gates[i])) for i in range(4)])
    assert sum(r[3] for r in rows) == tot, (sum(r[3] for r in rows), tot)


if __name__ == "__main__":
    for f in (fig_preproc, fig_patch, fig_blocks, fig_deepsup, fig_disc, fig_schedule, fig_window, fig_features, fig_metrics, fig_iteration, table_shapes):
        f()
        print("ok", f.__name__, flush=True)
    (FIG.parent / "meth_stats.json").write_text(json.dumps(STATS, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(json.dumps({k: v for k, v in STATS.items() if k != "shapes"}, default=str)[:1500])
