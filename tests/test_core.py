import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pytest, torch
from bratsgan import metrics as M
from bratsgan.data import assert_no_leakage, labels_to_regions, make_splits, normalise_volume
from bratsgan.infer import gaussian_importance, sliding_window
from bratsgan.losses import SegmentationLoss, soft_dice_loss, tversky_loss
from bratsgan.models import AttentionGate3D, Discriminator3D, ModalityAttentiveFusion, UNet3D
from bratsgan.stats import bootstrap_ci, holm, roc_auc


# ------------------------------------------------------------------ metrics
def test_dice_iou_hand_computed():
    a = np.zeros((4, 4, 4), bool); b = np.zeros((4, 4, 4), bool)
    a[:2] = True; b[1:3] = True              # |A|=32, |B|=32, |A&B|=16
    assert M.dice(a, b) == pytest.approx(2 * 16 / 64)
    assert M.iou(a, b) == pytest.approx(16 / 48)
    assert M.precision(a, b) == pytest.approx(0.5) and M.sensitivity(a, b) == pytest.approx(0.5)


def test_empty_conventions():
    z = np.zeros((6, 6, 6), bool); o = z.copy(); o[2:4, 2:4, 2:4] = True
    assert M.dice(z, z) == 1.0 and M.hd95(z, z) == 0.0
    assert M.dice(o, z) == 0.0 and M.hd95(o, z) == pytest.approx(373.13, abs=0.01)


def test_hd95_shift():
    a = np.zeros((20, 20, 20), bool); a[5:12, 5:12, 5:12] = True
    b = np.roll(a, 3, axis=0)
    assert M.hd95(a, a) == 0.0
    assert 2.0 <= M.hd95(a, b) <= 3.5          # a 3-voxel shift
    assert M.hd95(a, b, spacing=(2, 2, 2)) == pytest.approx(2 * M.hd95(a, b), rel=1e-6)


def test_dice_iou_identity_and_jensen():
    rng = np.random.default_rng(0)
    d = np.clip(rng.normal(0.99, 0.03, 20000), 0, 1)
    iou_each = d / (2 - d)
    assert iou_each.mean() >= d.mean() / (2 - d.mean()) - 1e-12   # convexity => mean IoU >= IoU(mean Dice)


def test_regions():
    seg = np.zeros((3, 3, 3), np.uint8); seg[0, 0, 0] = 1; seg[1, 1, 1] = 2; seg[2, 2, 2] = 4
    r = labels_to_regions(seg)
    assert r[0].sum() == 3 and r[1].sum() == 2 and r[2].sum() == 1


# -------------------------------------------------------------------- losses
def test_dice_loss_perfect_and_worst():
    t = (torch.rand(1, 3, 6, 6, 6) > 0.7).float()
    assert soft_dice_loss(t, t).item() < 1e-4
    assert soft_dice_loss(1 - t, t).item() > 0.99


def test_tversky_reduces_to_dice():
    p = torch.rand(2, 3, 5, 5, 5); t = (torch.rand_like(p) > 0.6).float()
    assert tversky_loss(p, t, 0.5, 0.5).item() == pytest.approx(soft_dice_loss(p, t).item(), rel=1e-6)


def test_tversky_penalises_fn_more():
    t = torch.zeros(1, 1, 8, 8, 8); t[..., :4, :, :] = 1
    p_fn = t.clone(); p_fn[..., 3, :, :] = 0           # misses some positives
    p_fp = t.clone(); p_fp[..., 4, :, :] = 1           # adds same number of false positives
    assert tversky_loss(p_fn, t, 0.3, 0.7) > tversky_loss(p_fp, t, 0.3, 0.7)


def test_gradcheck_overlap_losses():
    p = torch.rand(1, 2, 3, 3, 3, dtype=torch.double, requires_grad=True)
    t = (torch.rand(1, 2, 3, 3, 3, dtype=torch.double) > 0.5)
    t = t.double()
    assert torch.autograd.gradcheck(lambda q: soft_dice_loss(q, t), (p,))
    assert torch.autograd.gradcheck(lambda q: tversky_loss(q, t, 0.3, 0.7), (p,))


def test_deep_supervision_loss_shapes():
    logits = [torch.randn(1, 3, 16, 16, 16), torch.randn(1, 3, 8, 8, 8), torch.randn(1, 3, 4, 4, 4)]
    t = (torch.rand(1, 3, 16, 16, 16) > 0.7).float()
    l, parts = SegmentationLoss()(logits, t)
    assert l.ndim == 0 and set(parts) == {"dice", "tversky", "bce"}


# -------------------------------------------------------------------- models
@pytest.mark.parametrize("res,att,fus", [(False, False, "early"), (True, True, "early"), (True, True, "maf")])
def test_unet_shapes(res, att, fus):
    m = UNet3D(base=4, depth=4, max_ch=16, residual=res, attention=att, fusion=fus)
    x = torch.randn(2, 4, 16, 16, 16)
    m.train(); out = m(x)
    assert isinstance(out, list) and out[0].shape == (2, 3, 16, 16, 16) and out[1].shape[-1] == 8
    m.eval(); assert m(x).shape == (2, 3, 16, 16, 16)


def test_attention_alpha_in_unit_interval():
    g = AttentionGate3D(8, 8); x = torch.randn(1, 8, 6, 6, 6)
    y = g(x, torch.randn(1, 8, 6, 6, 6))
    a = g.last_alpha
    assert a.min() >= 0 and a.max() <= 1 and y.shape == x.shape


def test_maf_weights_sum_to_one():
    f = ModalityAttentiveFusion(4, 4, 8)
    f(torch.randn(2, 4, 8, 8, 8))
    w = f.last_weights                       # B,M,C
    assert torch.allclose(w.sum(1), torch.ones_like(w.sum(1)), atol=1e-5)


def test_discriminator_output_is_patch_map():
    d = Discriminator3D(4, 3, 8, n_layers=3)
    o = d(torch.randn(1, 4, 32, 32, 32), torch.rand(1, 3, 32, 32, 32))
    assert o.shape[1] == 1 and o.ndim == 5


def test_backward_runs():
    m = UNet3D(base=4, depth=3, max_ch=16); m.train()
    x = torch.randn(1, 4, 16, 16, 16); t = (torch.rand(1, 3, 16, 16, 16) > 0.8).float()
    l, _ = SegmentationLoss()(m(x), t); l.backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


# ------------------------------------------------------------------ inference
def test_gaussian_importance_peak_in_centre():
    w = gaussian_importance((16, 16, 16))
    assert w.max() == pytest.approx(1.0) and w[8, 8, 8] > w[0, 0, 0]


def test_sliding_window_matches_full_forward_for_fully_conv_const():
    class Const(torch.nn.Module):
        def __init__(self): super().__init__(); self.p = torch.nn.Parameter(torch.zeros(1))
        def forward(self, x): return torch.full((x.shape[0], 3) + x.shape[2:], 2.0)
    out = sliding_window(Const(), torch.randn(4, 20, 23, 25), patch=(16, 16, 16), overlap=0.5, device=torch.device("cpu"))
    assert out.shape == (3, 20, 23, 25)
    assert torch.allclose(out, torch.sigmoid(torch.tensor(2.0)).expand_as(out), atol=1e-5)  # perfect weighted average


# ----------------------------------------------------------------- data/splits
def test_splits_patient_level_and_stratified():
    cases = [{"id": f"c{i}", "grade": "HGG" if i < 80 else "LGG"} for i in range(100)]
    s = make_splits(cases, seed=1, test_frac=0.2, k=5)
    assert_no_leakage(s)
    allids = s["test"] + [i for f in s["folds"] for i in f]
    assert sorted(allids) == sorted(c["id"] for c in cases)
    lg = lambda ids: sum(i in {f"c{j}" for j in range(80, 100)} for i in ids)
    assert lg(s["test"]) == 4 and all(lg(f) in (3, 4) for f in s["folds"])


def test_leakage_detected():
    with pytest.raises(AssertionError):
        assert_no_leakage({"test": ["a"], "folds": [["a", "b"], ["c"]]})


def test_normalise_in_brain_zero_mean_unit_var():
    rng = np.random.default_rng(0)
    v = rng.normal(100, 20, (16, 16, 16)).astype(np.float32); brain = np.zeros_like(v, bool); brain[2:14, 2:14, 2:14] = True
    n = normalise_volume(v, brain)
    assert abs(n[brain].mean()) < 1e-4 and abs(n[brain].std() - 1) < 0.05 and (n[~brain] == 0).all()


# ------------------------------------------------------------------------ stats
def test_holm_and_auc_and_bootstrap():
    adj = holm([0.01, 0.04, 0.03])
    assert adj[0] == pytest.approx(0.03) and adj[2] == pytest.approx(0.06) and adj[1] == pytest.approx(0.06)
    assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])) == pytest.approx(0.75)
    m, lo, hi = bootstrap_ci(np.arange(100.0), n_boot=2000)
    assert lo < m < hi
