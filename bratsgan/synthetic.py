"""Synthetic BraTS-like phantoms.

These are used ONLY to test that the code is correct and to run implementation-verification
experiments. They carry no information about performance on real patients.

Each phantom has an ellipsoidal 'brain' with three tissue classes, four MR-like contrasts
(T1, T1ce, T2, FLAIR), a smooth multiplicative bias field, Rician-like noise, and a nested tumour
(necrotic core -> enhancing shell -> edema) written with BraTS labels {1,2,4}. HGG phantoms have
a strongly enhancing shell and a large necrotic core; LGG phantoms have weak/absent enhancement,
so grade is learnable in principle.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi


def _smooth_noise(shape, sigma, rng, amp=1.0):
    n = ndi.gaussian_filter(rng.standard_normal(shape), sigma)
    return amp * n / (n.std() + 1e-8)


def make_phantom(size: int = 64, grade: str = "HGG", seed: int = 0, difficulty: str = "easy"):
    """difficulty="easy" is the default generator used so far (random stream unchanged).
    difficulty="hard" gives smaller lesions, thinner enhancing shells, lower tumour contrast,
    blurred boundaries (partial-volume effect), stronger noise and a stronger bias field."""
    hard = difficulty == "hard"
    rng = np.random.default_rng(seed)
    S = (size, size, size)
    zz, yy, xx = np.meshgrid(*[np.linspace(-1, 1, s) for s in S], indexing="ij")
    brain = (zz / 0.85) ** 2 + (yy / 0.72) ** 2 + (xx / 0.62) ** 2 < 1.0
    # tissues: CSF (ventricle-like inner ellipsoid), WM (inner), GM (rim)
    r = np.sqrt((zz / 0.85) ** 2 + (yy / 0.72) ** 2 + (xx / 0.62) ** 2)
    csf = (((zz / 0.22) ** 2 + (yy / 0.12) ** 2 + (xx / 0.35) ** 2) < 1) & brain
    wm = (r < 0.78) & brain & ~csf
    gm = brain & ~wm & ~csf
    t1 = np.where(wm, 0.80, np.where(gm, 0.55, np.where(csf, 0.15, 0.0)))
    t2 = np.where(wm, 0.35, np.where(gm, 0.55, np.where(csf, 0.95, 0.0)))
    fl = np.where(wm, 0.30, np.where(gm, 0.45, np.where(csf, 0.05, 0.0)))
    t1c = t1.copy()
    # tumour geometry
    c = rng.uniform(-0.3, 0.3, 3)
    ax = rng.uniform(0.10, 0.22, 3) if hard else rng.uniform(0.18, 0.34, 3)
    ax_e = ax * rng.uniform(1.25, 1.6)
    rough = _smooth_noise(S, 3.0, rng, 0.08)

    def ell(a, k=1.0):
        d = ((zz - c[0]) / a[0]) ** 2 + ((yy - c[1]) / a[1]) ** 2 + ((xx - c[2]) / a[2]) ** 2
        return np.sqrt(d) + rough < k

    edema = ell(ax_e) & brain
    outer = ell(ax) & brain
    if grade == "HGG":
        core = ell(ax * (0.78 if hard else 0.55)) & brain
        enh = outer & ~core
        label = np.zeros(S, np.uint8)
        label[edema] = 2
        label[core] = 1
        label[enh] = 4
        t1c_enh, t1c_core = 1.00, 0.20
    else:
        core = ell(ax * 0.80) & brain
        enh = np.zeros(S, bool)
        if rng.random() < 0.25:  # a small enhancing focus in a minority of LGG, as in real data
            enh = ell(ax * 0.35, 1.0) & brain
        label = np.zeros(S, np.uint8)
        label[edema] = 2
        label[outer] = 1
        label[enh] = 4
        t1c_enh, t1c_core = 0.90, 0.45
    tis = (t1.copy(), t2.copy(), fl.copy())  # tissue-only maps (tumour not yet written), used by the hard mode
    ed_m, co_m, en_m = label == 2, label == 1, label == 4
    t1 = np.where(ed_m, 0.45, t1); t1 = np.where(co_m, 0.25, t1); t1 = np.where(en_m, 0.60, t1)
    t1c = np.where(ed_m, 0.45, t1); t1c = np.where(co_m, t1c_core, t1c); t1c = np.where(en_m, t1c_enh, t1c)
    t2 = np.where(ed_m, 0.90, t2); t2 = np.where(co_m, 0.80, t2); t2 = np.where(en_m, 0.70, t2)
    fl = np.where(ed_m, 0.95, fl); fl = np.where(co_m, 0.60, fl); fl = np.where(en_m, 0.75, fl)
    if hard:
        tum = ed_m | co_m | en_m
        base = (tis[0], tis[0], tis[1], tis[2])  # t1ce tissue baseline equals t1
        k = 0.45  # tumour contrast is reduced to 45 % of its easy-mode difference to the tissue
        mm = []
        for m, b in zip((t1, t1c, t2, fl), base):
            m2 = np.where(tum, b + k * (m - b), m)
            mm.append(ndi.gaussian_filter(m2, 1.2) * brain)
        t1, t1c, t2, fl = mm
    sig_noise, amp_bias = (0.10, 0.30) if hard else (0.03, 0.15)
    vols = []
    for m in (t1, t1c, t2, fl):
        bias = np.exp(_smooth_noise(S, 12.0, rng, amp_bias))
        v = m * bias
        v = np.sqrt((v + rng.normal(0, sig_noise, S)) ** 2 + rng.normal(0, sig_noise, S) ** 2)  # Rician-like
        v = v * brain
        vols.append(v.astype(np.float32))
    return np.stack(vols), label, brain  # (4,D,H,W) order t1,t1ce,t2,flair


def write_brats_layout(root, n_hgg: int = 24, n_lgg: int = 8, size: int = 64, seed: int = 0, difficulty: str = "easy") -> Path:
    """Write phantoms as NIfTI in the BraTS 2020 folder layout (so the real I/O path is exercised)."""
    import nibabel as nib
    root = Path(root) / "MICCAI_BraTS2020_TrainingData"
    root.mkdir(parents=True, exist_ok=True)
    rows = ["Grade,BraTS_2017_subject_ID,BraTS_2018_subject_ID,TCIA_subject_ID,BraTS_2019_subject_ID,BraTS_2020_subject_ID"]
    aff = np.eye(4)
    k = 0
    for g, n in (("HGG", n_hgg), ("LGG", n_lgg)):
        for i in range(n):
            k += 1
            cid = f"BraTS20_Training_{k:03d}"
            d = root / cid
            d.mkdir(exist_ok=True)
            vols, lab, _ = make_phantom(size, g, seed * 10_000 + k, difficulty)
            for name, v in zip(("t1", "t1ce", "t2", "flair"), vols):
                nib.save(nib.Nifti1Image(v, aff), str(d / f"{cid}_{name}.nii.gz"))
            nib.save(nib.Nifti1Image(lab, aff), str(d / f"{cid}_seg.nii.gz"))
            rows.append(f"{g},,,,,{cid}")
    (root / "name_mapping.csv").write_text("\n".join(rows) + "\n")
    return root.parent
