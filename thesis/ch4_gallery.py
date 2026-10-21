"""Gallery of samples and detections on the hard synthetic benchmark (seven figures, one table).

Every panel is computed from the cached phantoms and from the prediction files written by the training scripts
(runs/<variant>_s1_f0/pred_test/*_pred.npy); nothing is drawn by hand.  The phantoms are test phantoms of fold 0 and every
phantom appears in at most one figure, so that the seven figures show 14 different samples.  Selection rules are stated in the text.
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

USED: list = []
LAB_COL = {1: "#2a78d6", 2: "#1baf7a", 4: "#f2c230"}   # necrotic/non-enhancing core, oedema, enhancing tumour
ERR_COL = {1: "#1baf7a", 2: "#f2c230", 3: "#e34a33", 4: "#2a78d6"}  # agree, wrong class, false positive, missed


def _labels_from_regions(p):
    """(3,D,H,W) binary WT/TC/ET -> BraTS label map {0,1,2,4} with the hierarchy ET in TC in WT."""
    wt, tc, et = (p[0] > 0), (p[1] > 0) & (p[0] > 0), (p[2] > 0) & (p[0] > 0)
    out = np.zeros(wt.shape, np.uint8)
    out[wt] = 2
    out[tc] = 1
    out[et] = 4
    return out


def _rgb(lab):
    im = np.zeros(lab.shape + (4,), float)
    for k, c in LAB_COL.items():
        m = lab == k
        im[m] = matplotlib_rgba(c)
    return im


def matplotlib_rgba(c, a=1.0):
    import matplotlib.colors as mc
    r = mc.to_rgba(c)
    return (r[0], r[1], r[2], a)


def _err(gt, pr):
    e = np.zeros(gt.shape + (4,), float)
    both = (gt > 0) & (pr > 0)
    e[both & (gt == pr)] = matplotlib_rgba(ERR_COL[1], 0.85)
    e[both & (gt != pr)] = matplotlib_rgba(ERR_COL[2], 0.95)
    e[(gt == 0) & (pr > 0)] = matplotlib_rgba(ERR_COL[3], 0.95)
    e[(gt > 0) & (pr == 0)] = matplotlib_rgba(ERR_COL[4], 0.95)
    return e


def _dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s


def _slice(vol, axis, idx):
    return np.take(vol, idx, axis=axis if vol.ndim == 3 else axis + 1)


def build(H, base, plt, C, figdir: Path) -> str:
    from bratsgan.data import load_case
    cfgs = {c: H.dirs(c, 0, (1,)) for c in ("a0_unet", "a3_maf", "a5_full", "t_fn_heavy")}
    head = "@subsection[hd-gal] Gallery of samples and detections\n\n"
    if not all(v for k, v in cfgs.items() if k in ("a0_unet", "a5_full")):
        return head + "~pending\n"
    dirs = {k: v[0] for k, v in cfgs.items() if v}
    cache = json.loads((dirs["a5_full"] / "config.json").read_text())["cache_dir"]
    m5 = AN.add_composite(pd.read_csv(dirs["a5_full"] / "test_metrics.csv").set_index("id"))
    ids = list(m5.index)
    info = {}
    for cid in ids:
        img, seg, meta = load_case(cache, cid, mmap=False)
        seg = np.asarray(seg)
        info[cid] = {"grade": meta.get("grade", ""), "wt": int((seg > 0).sum()), "et": int((seg == 4).sum())}
    used: list[str] = []

    def pick(cands, n=2):
        out = [c for c in cands if c not in used][:n]
        used.extend(out)
        return out

    hgg = [c for c in ids if info[c]["grade"] == "HGG"]
    lgg = [c for c in ids if info[c]["grade"] == "LGG"]
    rank_h = sorted(hgg, key=lambda c: -m5.loc[c, "dice_mean"])
    rank_l = sorted(lgg, key=lambda c: -m5.loc[c, "dice_mean"])
    med_h = sorted(hgg, key=lambda c: abs(m5.loc[c, "dice_mean"] - np.median([m5.loc[x, "dice_mean"] for x in hgg])))
    by_size = sorted(ids, key=lambda c: info[c]["wt"])
    worst = sorted(ids, key=lambda c: m5.loc[c, "dice_mean"])
    rng = np.random.default_rng(2026)
    rest_a = list(rng.permutation(ids))
    rest_b = list(rng.permutation(ids))
    big = sorted(ids, key=lambda c: -info[c]["wt"])
    plan = [
        ("g1", "best HGG-like detections of A5", pick(rank_h), "a5_full"),
        ("g2", "HGG-like phantoms at the median Dice of A5", pick(med_h), "a5_full"),
        ("g3", "LGG-like phantoms (weak or absent enhancement), A5", pick(rank_l if len(rank_l) >= 2 else lgg), "a5_full"),
        ("g4", "the smallest lesions, A5", pick(by_size), "a5_full"),
        ("g5", "the lowest Dice of A5 (failure cases)", pick(worst), "a5_full"),
        ("g6", "different networks on the same phantoms", pick(rest_a), None),
        ("g7", "orthogonal views, large lesions, A5", pick(big), "a5_full"),
    ]

    USED[:] = list(used)

    def load(cid):
        img, seg, meta = load_case(cache, cid, mmap=False)
        return np.asarray(img), np.asarray(seg)

    def pred(cfg, cid):
        return _labels_from_regions(np.load(dirs[cfg] / "pred_test" / f"{cid}_pred.npy"))

    def zc(seg, axis=0):
        pos = np.argwhere(seg > 0)
        return int(pos[:, axis].mean()) if len(pos) else seg.shape[axis] // 2

    def clean(ax):
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    rows, texts, figs = [], [], []
    for gid, title, cs, cfg in plan:
        if not cs:
            continue
        if gid == "g6":
            variants = [(k, t) for k, t in (("a0_unet", "A0"), ("a3_maf", "A3"), ("t_fn_heavy", "T0.8"), ("a5_full", "A5")) if k in dirs]
            ncol = 2 + len(variants)
            fig, ax = plt.subplots(len(cs), ncol, figsize=(6.6, 1.15 * ncol * 0 + 1.55 * len(cs) + 0.4))
            ax = np.atleast_2d(ax)
            for i, cid in enumerate(cs):
                img, seg = load(cid); z = zc(seg)
                ax[i, 0].imshow(img[1, z], cmap="gray"); ax[i, 1].imshow(img[1, z], cmap="gray"); ax[i, 1].imshow(_rgb(seg[z]))
                for j, (k, t) in enumerate(variants):
                    pr = pred(k, cid)
                    ax[i, 2 + j].imshow(img[1, z], cmap="gray"); ax[i, 2 + j].imshow(_rgb(pr[z]))
                    if i == 0:
                        ax[0, 2 + j].set_title(t, fontsize=8.5)
                for a_ in ax[i]:
                    clean(a_)
                ax[i, 0].set_ylabel(f"{cid[-3:]}", fontsize=7)
            ax[0, 0].set_title("T1ce", fontsize=8.5); ax[0, 1].set_title("Reference", fontsize=8.5)
        elif gid == "g7":
            fig, ax = plt.subplots(len(cs), 6, figsize=(6.6, 1.35 * len(cs) + 0.4))
            ax = np.atleast_2d(ax)
            for i, cid in enumerate(cs):
                img, seg = load(cid); pr = pred(cfg, cid)
                for v, (axis, name) in enumerate(((0, "axial"), (1, "coronal"), (2, "sagittal"))):
                    idx = zc(seg, axis)
                    bg = _slice(img[1], axis, idx) if True else None
                    ax[i, 2 * v].imshow(bg, cmap="gray", origin="lower"); ax[i, 2 * v].imshow(_rgb(_slice(seg, axis, idx)), origin="lower")
                    ax[i, 2 * v + 1].imshow(bg, cmap="gray", origin="lower"); ax[i, 2 * v + 1].imshow(_rgb(_slice(pr, axis, idx)), origin="lower")
                    if i == 0:
                        ax[0, 2 * v].set_title(f"{name}: reference", fontsize=7.5); ax[0, 2 * v + 1].set_title(f"{name}: A5", fontsize=7.5)
                for a_ in ax[i]:
                    clean(a_)
                ax[i, 0].set_ylabel(f"{cid[-3:]}", fontsize=7)
        else:
            fig, ax = plt.subplots(len(cs), 5, figsize=(6.6, 1.55 * len(cs) + 0.4))
            ax = np.atleast_2d(ax)
            for i, cid in enumerate(cs):
                img, seg = load(cid); pr = pred(cfg, cid); z = zc(seg)
                ax[i, 0].imshow(img[1, z], cmap="gray"); ax[i, 1].imshow(img[3, z], cmap="gray")
                ax[i, 2].imshow(img[1, z], cmap="gray"); ax[i, 2].imshow(_rgb(seg[z]))
                ax[i, 3].imshow(img[1, z], cmap="gray"); ax[i, 3].imshow(_rgb(pr[z]))
                ax[i, 4].imshow(img[1, z], cmap="gray", alpha=0.6); ax[i, 4].imshow(_err(seg[z], pr[z]))
                for a_ in ax[i]:
                    clean(a_)
                ax[i, 0].set_ylabel(f"{cid[-3:]}", fontsize=7)
            for k, t in enumerate(("T1ce", "FLAIR", "Reference", "Detection (A5)", "Error")):
                ax[0, k].set_title(t, fontsize=8.5)
        fig.tight_layout()
        fn = figdir / f"gal_{gid}.png"
        fig.savefig(fn); plt.close(fig)
        figs.append((gid, title, cs, fn))
        for cid in cs:
            r = m5.loc[cid]
            rows.append(f"| {gid[1:]} | {cid[-3:]} | {info[cid]['grade']} | {info[cid]['wt']:,} | {info[cid]['et']:,} | {r['dice_WT']:.3f} | {r['dice_TC']:.3f} | {r['dice_ET']:.3f} |")

    cap = {
        "g1": "The two HGG-like test phantoms with the highest mean Dice of A5. Columns: T1ce, FLAIR, reference labels, detection of A5, and error map. Central axial slice through the tumour.",
        "g2": "The two HGG-like test phantoms whose mean Dice of A5 is closest to the median over the HGG-like test phantoms. Columns as in the preceding figure.",
        "g3": "LGG-like test phantoms (non-enhancing core, rarely a small enhancing focus). Columns as in the preceding figures; a coloured pixel in the detection that has no counterpart in the reference marks a false alarm.",
        "g4": "The two test phantoms with the smallest whole-tumour volume (that are not shown in the preceding figures). Columns as before.",
        "g5": "The two test phantoms with the lowest mean Dice of A5 (failure cases). Columns as before.",
        "g6": "Detections of four networks on the same two test phantoms: the plain network A0, the network with the fusion stem A3, the network trained with the Tversky weights $(0.2,0.8)$ (T0.8) and the full model A5.",
        "g7": "Orthogonal views (axial, coronal, sagittal) through the centre of the whole tumour of the two test phantoms with the largest whole-tumour volume, with the reference and the detection of A5 side by side.",
    }
    legend = " Label colours: blue, necrotic or non-enhancing core; green, oedema; yellow, enhancing tumour. Error map: green, correct label; amber, tumour voxel with the wrong label; red, false positive; blue, missed tumour voxel."
    body = []
    for gid, title, cs, fn in figs:
        extra = legend if gid in ("g1", "g3") else ""
        body.append(f"@fig hd-{gid} | file={fn} | width=6.4\n{cap[gid]}{extra}\n@end\n")
    # quantitative remarks computed from the data
    lg = [c for g_, t_, cs_, f_ in figs if g_ == "g3" for c in cs_]
    fa = []
    for c in lg:
        pr = pred("a5_full", c); seg = load(c)[1]
        fa.append((int((pr == 4).sum()), int((seg == 4).sum())))
    wst = [c for g_, t_, cs_, f_ in figs if g_ == "g5" for c in cs_]
    g6 = [c for g_, t_, cs_, f_ in figs if g_ == "g6" for c in cs_]
    g6t = []
    for c in g6:
        seg = load(c)[1]
        parts = [f"reference {int((seg == 4).sum())}"] + [f"{t} {int((pred(k, c) == 4).sum())}" for k, t in (("a0_unet", "A0"), ("a3_maf", "A3"), ("t_fn_heavy", "T0.8"), ("a5_full", "A5")) if k in dirs]
        g6t.append(f"phantom {c[-3:]}: " + ", ".join(parts))
    g6txt = "; ".join(g6t)
    txt = f"""The qualitative comparison of the previous subsection shows three phantoms selected by the difference between two networks. This subsection shows further samples selected by rules that are fixed before the images are drawn, so that the gallery covers the range of the detections and not only their successes. The test set of fold 0 has {len(ids)} phantoms ({len(hgg)} HGG-like and {len(lgg)} LGG-like), and each of the seven figures shows two phantoms that appear in no other figure, 14 in total. The selections are: the HGG-like phantoms with the highest Dice of A5 (@fig:hd-g1), those at the median (@fig:hd-g2), LGG-like phantoms (@fig:hd-g3), the smallest lesions (@fig:hd-g4), the lowest Dice (@fig:hd-g5), two phantoms drawn at random with a fixed seed and detected by four different networks (@fig:hd-g6), and the largest lesions in three orthogonal views (@fig:hd-g7). @tab:hd-gal gives the reference volumes and the Dice scores of the 14 phantoms.

{chr(10).join(body)}
@tab hd-gal | widths=0.5,0.7,0.6,1.1,1.0,0.8,0.8,0.8 | align=cccrrccc | size=8.5
Samples of the gallery: figure number of the gallery (the order of @fig:hd-g1 to @fig:hd-g7), phantom, grade, volumes of the whole tumour and of the enhancing tumour in voxels, and Dice of A5 (fold 0, seed 1) for the three regions.
| Fig. | Phantom | Grade | WT volume | ET volume | Dice WT | Dice TC | Dice ET |
| --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}
@end

Three features of the gallery deserve comment. First, the boundary of the whole tumour follows the reference closely in the figures, and the errors concentrate on the interface between the compartments and on the enhancing tumour, which is thin and low in contrast in the hard setting. Second, the failure cases of @fig:hd-g5 have the lowest Dice scores of the test set ({', '.join(f"{m5.loc[c, 'dice_mean']:.3f}" for c in wst)}) and are distinguished by the size of the reference regions ({', '.join(f"{info[c]['wt']:,}" for c in wst)} voxels of whole tumour, against a median of {int(np.median([info[c]['wt'] for c in ids])):,}) and not by an artefact of the benchmark. Third, the compartment that the networks disagree on is the enhancing tumour. On the two phantoms of @fig:hd-g6 the numbers of voxels labelled as enhancing are: {g6txt}. A network that labels a multiple of the reference volume as enhancing, as the images of A0 and A3 show for these phantoms, makes the enhancing label cover the core and the shell together; this is the over-segmentation of the enhancing tumour that the volume ratios of @tab:hd-tv quantify over the whole test set. For the LGG-like phantoms of @fig:hd-g3 the reference contains {', '.join(str(e) for _, e in fa)} enhancing voxels, and A5 labels {', '.join(str(p) for p, _ in fa)} voxels as enhancing; a detection of enhancement in a phantom whose reference has little or none is a false alarm of the kind that the empty-mask conventions of Chapter 3 are designed to penalise. The images are a selection and are not a statistical statement; the statistics are in the preceding subsections.
"""
    return head + txt
