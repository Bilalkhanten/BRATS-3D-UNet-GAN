"""Network definitions.

* UNet3D       : encoder-decoder with optional residual blocks, optional additive
                 attention gates on the skip connections, optional modality-attentive
                 fusion (MAF) stem, and optional deep supervision.
* Discriminator3D : conditional 3-D PatchGAN with spectral normalisation. It sees the
                 image *and* a segmentation (ground truth or prediction).
"""
from __future__ import annotations
from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ---------------------------------------------------------------- building blocks
def make_norm(kind: str, ch: int) -> nn.Module:
    if kind == "instance":
        return nn.InstanceNorm3d(ch, affine=True)
    if kind == "group":
        return nn.GroupNorm(min(8, ch), ch)
    if kind == "batch":
        return nn.BatchNorm3d(ch)
    raise ValueError(f"unknown norm {kind}")


def make_act() -> nn.Module:
    return nn.LeakyReLU(0.01, inplace=True)


class ConvNormAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 3, stride: int = 1, norm: str = "instance"):
        super().__init__(
            nn.Conv3d(cin, cout, k, stride=stride, padding=k // 2, bias=False),
            make_norm(norm, cout),
            make_act(),
        )


class PlainBlock(nn.Module):
    """Two (conv-norm-act) layers; the first one may down-sample."""

    def __init__(self, cin: int, cout: int, stride: int = 1, norm: str = "instance"):
        super().__init__()
        self.body = nn.Sequential(ConvNormAct(cin, cout, 3, stride, norm), ConvNormAct(cout, cout, 3, 1, norm))

    def forward(self, x):
        return self.body(x)


class ResBlock(nn.Module):
    """y = act( F(x) + S(x) ),  F = conv-norm-act-conv-norm,  S = identity or 1x1x1 projection."""

    def __init__(self, cin: int, cout: int, stride: int = 1, norm: str = "instance"):
        super().__init__()
        self.c1 = ConvNormAct(cin, cout, 3, stride, norm)
        self.c2 = nn.Sequential(nn.Conv3d(cout, cout, 3, padding=1, bias=False), make_norm(norm, cout))
        if cin != cout or stride != 1:
            self.skip = nn.Sequential(nn.Conv3d(cin, cout, 1, stride=stride, bias=False), make_norm(norm, cout))
        else:
            self.skip = nn.Identity()
        self.act = make_act()

    def forward(self, x):
        return self.act(self.c2(self.c1(x)) + self.skip(x))


class AttentionGate3D(nn.Module):
    """Additive attention gate (Oktay et al., 2018) at a single resolution.

        q    = ReLU( W_x x + W_g g )
        alpha = sigmoid( psi(q) )            in (0,1), one scalar per voxel
        x_hat = alpha * x
    x: skip features (C_x channels), g: gating signal (decoder features already
    up-sampled to the resolution of x).
    """

    def __init__(self, cx: int, cg: int, cint: Optional[int] = None):
        super().__init__()
        cint = cint or max(cx // 2, 1)
        self.wx = nn.Conv3d(cx, cint, 1, bias=False)
        self.wg = nn.Conv3d(cg, cint, 1, bias=True)
        self.psi = nn.Conv3d(cint, 1, 1, bias=True)
        self.last_alpha: Optional[torch.Tensor] = None

    def forward(self, x, g):
        q = F.relu(self.wx(x) + self.wg(g), inplace=True)
        alpha = torch.sigmoid(self.psi(q))
        self.last_alpha = alpha.detach()
        return x * alpha


class ModalityAttentiveFusion(nn.Module):
    """Modality-attentive fusion (MAF) stem.

    Every modality m has its own stem f_m = stem_m(x_m). A shared MLP scores the
    globally pooled descriptor z_m = GAP(f_m) per channel; a softmax over modalities
    gives weights a_m (sum_m a_m = 1 per channel). The fused map is sum_m a_m * f_m
    and a 1x1x1 convolution mixes [fused, f_1,...,f_M] into the first encoder width.
    """

    def __init__(self, n_mod: int, ch: int, out_ch: int, norm: str = "instance"):
        super().__init__()
        self.n_mod = n_mod
        self.stems = nn.ModuleList([ConvNormAct(1, ch, 3, 1, norm) for _ in range(n_mod)])
        self.score = nn.Sequential(nn.Linear(ch, max(ch // 2, 1)), nn.ReLU(inplace=True), nn.Linear(max(ch // 2, 1), ch))
        self.mix = nn.Sequential(nn.Conv3d(ch * (n_mod + 1), out_ch, 1, bias=False), make_norm(norm, out_ch), make_act())
        self.last_weights: Optional[torch.Tensor] = None

    def forward(self, x):
        feats = [stem(x[:, m : m + 1]) for m, stem in enumerate(self.stems)]
        z = torch.stack([f.mean(dim=(2, 3, 4)) for f in feats], dim=1)  # B,M,C
        a = torch.softmax(self.score(z), dim=1)                            # B,M,C
        self.last_weights = a.detach()
        fused = sum(a[:, m, :, None, None, None] * feats[m] for m in range(self.n_mod))
        return self.mix(torch.cat([fused] + feats, dim=1))


# ------------------------------------------------------------------------- U-Net
class UNet3D(nn.Module):
    def __init__(
        self,
        in_ch: int = 4,
        out_ch: int = 3,
        base: int = 32,
        depth: int = 5,
        max_ch: int = 320,
        residual: bool = True,
        attention: bool = True,
        fusion: str = "maf",          # "early" | "maf"
        norm: str = "instance",
        deep_supervision: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert depth >= 3 and fusion in ("early", "maf")
        self.depth, self.deep_supervision = depth, deep_supervision
        chs = [min(base * 2 ** i, max_ch) for i in range(depth)]
        self.chs = chs
        Block = ResBlock if residual else PlainBlock

        if fusion == "maf":
            self.fusion = ModalityAttentiveFusion(in_ch, max(base // 2, 4), chs[0], norm)
            first_in = chs[0]
        else:
            self.fusion = None
            first_in = in_ch

        enc = [Block(first_in, chs[0], 1, norm)]
        for i in range(1, depth):
            enc.append(Block(chs[i - 1], chs[i], 2, norm))
        self.enc = nn.ModuleList(enc)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

        self.up = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(depth - 1):  # decoder stage i merges enc[i] and up(enc/dec[i+1])
            self.up.append(nn.ConvTranspose3d(chs[i + 1], chs[i], 2, stride=2))
            self.gates.append(AttentionGate3D(chs[i], chs[i]) if attention else nn.Identity())
            self.dec.append(Block(2 * chs[i], chs[i], 1, norm))
        n_heads = min(3, depth - 1) if deep_supervision else 1
        self.heads = nn.ModuleList([nn.Conv3d(chs[i], out_ch, 1) for i in range(n_heads)])
        self.attention = attention

    @property
    def bottleneck_channels(self) -> int:
        return self.chs[-1]

    def forward(self, x, return_feat: bool = False):
        if self.fusion is not None:
            x = self.fusion(x)
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            skips.append(x)
        feat = self.drop(skips[-1])
        h = feat
        outs: List[torch.Tensor] = []
        for i in reversed(range(self.depth - 1)):
            u = self.up[i](h)
            s = skips[i]
            if self.attention:
                s = self.gates[i](s, u)
            h = self.dec[i](torch.cat([u, s], dim=1))
            if i < len(self.heads):
                outs.append(self.heads[i](h))
        outs = outs[::-1]  # [full-res, 1/2, 1/4]
        logits: Union[torch.Tensor, List[torch.Tensor]]
        logits = outs if (self.deep_supervision and self.training) else outs[0]
        return (logits, feat) if return_feat else logits


# ---------------------------------------------------------------- discriminator
class Discriminator3D(nn.Module):
    """Conditional 3-D PatchGAN. Input = concat(image [C_img], segmentation [C_seg]).

    Output is a map of per-patch scores (LSGAN, no sigmoid)."""

    def __init__(self, img_ch: int = 4, seg_ch: int = 3, ndf: int = 32, n_layers: int = 4, use_sn: bool = True):
        super().__init__()
        sn = spectral_norm if use_sn else (lambda m: m)
        layers: List[nn.Module] = []
        cin, cout = img_ch + seg_ch, ndf
        for i in range(n_layers):
            stride = 2 if i < n_layers - 1 else 1
            layers += [sn(nn.Conv3d(cin, cout, 4, stride=stride, padding=1)), nn.LeakyReLU(0.2, inplace=True)]
            cin, cout = cout, min(cout * 2, ndf * 8)
        layers.append(sn(nn.Conv3d(cin, 1, 4, stride=1, padding=1)))
        self.net = nn.Sequential(*layers)

    def forward(self, img, seg):
        return self.net(torch.cat([img, seg], dim=1))


def build_model(cfg: dict) -> UNet3D:
    keys = ["in_ch", "out_ch", "base", "depth", "max_ch", "residual", "attention", "fusion", "norm", "deep_supervision", "dropout"]
    return UNet3D(**{k: cfg[k] for k in keys if k in cfg})
