"""Loss functions.

All overlap losses operate on *probabilities* p = sigmoid(z) in [0,1], shape (B,C,D,H,W),
against binary targets t of the same shape. Sums run over batch and space jointly
("batch Dice"), and the loss is averaged over the C region channels.

    TP = sum(p t),  FP = sum(p (1-t)),  FN = sum((1-p) t)
    Dice     = (2 TP + e) / (2 TP + FP + FN + e)
    Tversky  = (TP + e) / (TP + a FP + b FN + e)         (a=b=1/2 reduces to Dice)
"""
from __future__ import annotations
from typing import Sequence, Union
import torch
import torch.nn.functional as F


def _sums(p: torch.Tensor, t: torch.Tensor):
    dims = (0, 2, 3, 4)
    tp = (p * t).sum(dims)
    fp = (p * (1 - t)).sum(dims)
    fn = ((1 - p) * t).sum(dims)
    return tp, fp, fn


def soft_dice_loss(p, t, eps: float = 1e-5):
    tp, fp, fn = _sums(p, t)
    return 1.0 - ((2 * tp + eps) / (2 * tp + fp + fn + eps)).mean()


def tversky_loss(p, t, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-5, gamma: float = 1.0):
    """alpha weights false positives, beta false negatives. gamma<1 gives the focal Tversky loss."""
    tp, fp, fn = _sums(p, t)
    ti = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return ((1.0 - ti) ** gamma).mean()


class SegmentationLoss(torch.nn.Module):
    """L_seg = w_dice L_Dice + w_tv L_Tversky + w_bce L_BCE with deep supervision.

    Deep-supervision weights are w_i = 2^-i normalised to sum to one."""

    def __init__(self, w_dice=1.0, w_tv=1.0, w_bce=1.0, alpha=0.3, beta=0.7, gamma=1.0):
        super().__init__()
        self.w_dice, self.w_tv, self.w_bce = w_dice, w_tv, w_bce
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def single(self, logits, target):
        p = torch.sigmoid(logits.float())
        parts = {}
        total = logits.new_zeros(()).float()
        if self.w_dice:
            parts["dice"] = soft_dice_loss(p, target); total = total + self.w_dice * parts["dice"]
        if self.w_tv:
            parts["tversky"] = tversky_loss(p, target, self.alpha, self.beta, gamma=self.gamma); total = total + self.w_tv * parts["tversky"]
        if self.w_bce:
            parts["bce"] = F.binary_cross_entropy_with_logits(logits.float(), target); total = total + self.w_bce * parts["bce"]
        return total, parts

    def forward(self, logits: Union[torch.Tensor, Sequence[torch.Tensor]], target: torch.Tensor):
        if torch.is_tensor(logits):
            return self.single(logits, target)
        w = torch.tensor([0.5 ** i for i in range(len(logits))])
        w = (w / w.sum()).tolist()
        total, parts_main = 0.0, {}
        for i, (lg, wi) in enumerate(zip(logits, w)):
            tgt = target if lg.shape[2:] == target.shape[2:] else F.interpolate(target, size=lg.shape[2:], mode="nearest")
            l, parts = self.single(lg, tgt)
            total = total + wi * l
            if i == 0:
                parts_main = parts
        return total, parts_main


# ----------------------------------------------------------------- adversarial
def d_loss_lsgan(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    return 0.5 * (((d_real - 1) ** 2).mean() + (d_fake ** 2).mean())


def g_loss_lsgan(d_fake: torch.Tensor) -> torch.Tensor:
    return 0.5 * ((d_fake - 1) ** 2).mean()


def d_loss_hinge(d_real, d_fake):
    return 0.5 * (F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean())


def g_loss_hinge(d_fake):
    return -d_fake.mean()


def adv_ramp(step: int, ramp_steps: int) -> float:
    """Linear warm-up of the adversarial weight from 0 to 1."""
    return 1.0 if ramp_steps <= 0 else min(1.0, step / float(ramp_steps))
