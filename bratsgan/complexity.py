"""Exact parameter counts, multiply-accumulate (MAC) counts and activation footprints.

Counts are obtained by running a forward pass on the *meta* device (no memory is allocated), with hooks on
Conv3d / ConvTranspose3d / Linear. They are therefore exact for the implemented architecture."""
from __future__ import annotations
import copy
from typing import Dict, Sequence
import torch
import torch.nn as nn


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


@torch.no_grad()
def profile(model: nn.Module, input_shape: Sequence[int], *extra_inputs) -> Dict[str, float]:
    n_params = count_params(model)
    model = copy.deepcopy(model).to("meta")
    macs = [0]
    act = [0]
    hooks = []

    def conv_hook(mod, inp, out):
        k = 1
        for s in mod.kernel_size:
            k *= s
        if isinstance(mod, nn.ConvTranspose3d):
            spatial = inp[0].shape[2:].numel()
            macs[0] += inp[0].shape[0] * mod.in_channels * mod.out_channels * k * spatial // mod.groups
        else:
            spatial = out.shape[2:].numel()
            macs[0] += out.shape[0] * (mod.in_channels // mod.groups) * mod.out_channels * k * spatial
        act[0] += out.numel()

    def lin_hook(mod, inp, out):
        macs[0] += inp[0].numel() // inp[0].shape[-1] * mod.in_features * mod.out_features

    def other_hook(mod, inp, out):
        if torch.is_tensor(out):
            act[0] += out.numel()

    for mod in model.modules():
        if isinstance(mod, (nn.Conv3d, nn.ConvTranspose3d)):
            hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(lin_hook))
        elif isinstance(mod, (nn.InstanceNorm3d, nn.GroupNorm, nn.BatchNorm3d, nn.LeakyReLU, nn.ReLU)):
            hooks.append(mod.register_forward_hook(other_hook))
    was_training = model.training
    model.eval()
    x = torch.empty(*input_shape, device="meta")
    model(x, *extra_inputs)
    model.train(was_training)
    for h in hooks:
        h.remove()
    return {"params": n_params, "gmacs": macs[0] / 1e9, "gflops": 2 * macs[0] / 1e9,
            "activation_elems_M": act[0] / 1e6, "activation_GB_fp32": act[0] * 4 / 1e9}
