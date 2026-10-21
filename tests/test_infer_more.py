import torch

from bratsgan.infer import _starts, gaussian_importance, sliding_window


def test_window_starts_cover_axis_with_final_flush():
    for n, p, step in [(100, 32, 16), (33, 32, 16), (32, 32, 16), (50, 48, 24)]:
        starts = _starts(n, p, step)
        assert starts[0] == 0 and starts[-1] + p >= n
        covered = set()
        for s in starts:
            covered |= set(range(s, min(s + p, n)))
        assert covered == set(range(n))


def test_gaussian_importance_peaks_in_centre():
    w = gaussian_importance((16, 16, 16))
    assert w.shape == (16, 16, 16)
    assert w[8, 8, 8] > w[0, 0, 0] and (w > 0).all()


def test_sliding_window_returns_sigmoid_of_a_constant_logit():
    class Const(torch.nn.Module):
        def forward(self, x):
            out = torch.full((x.shape[0], 3, *x.shape[2:]), 0.25)
            return out
    x = torch.randn(4, 20, 23, 21)
    p = sliding_window(Const(), x, (16, 16, 16), 0.5, False, device=torch.device("cpu"))
    assert p.shape == (3, 20, 23, 21)
    assert torch.allclose(p, torch.full_like(p, torch.sigmoid(torch.tensor(0.25)).item()), atol=1e-5)  # logits in, probabilities out
