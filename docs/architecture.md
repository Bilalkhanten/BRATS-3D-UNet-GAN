# Architecture

## Generator: 3D U-Net

`bratsgan.models.UNet3D` is a five-level encoder-decoder (`depth=5`) with `base=32` channels at the first level, doubled per level
up to `max_ch=320`, instance normalisation and LeakyReLU. Downsampling is done by the stride-2 convolution of the first block of each level,
upsampling by transposed convolutions. Each component can be switched on independently, which defines the **ablation chain**:

| Variant | Config | Adds |
| --- | --- | --- |
| A0 | `configs/a0_unet.json` | plain 3D U-Net, early fusion of the four sequences, Dice + BCE |
| A1 | `a1_res.json` | residual blocks |
| A2 | `a2_att.json` | additive attention gates on the skip connections |
| A3 | `a3_maf.json` | modality-attentive fusion (MAF) stem |
| A4 | `a4_tversky.json` | Tversky loss (alpha = 0.3, beta = 0.7) |
| A5 | `a5_full.json` | adversarial refinement (LSGAN, conditional PatchGAN) |

### Attention gate

For a skip feature `s` and gating signal `g` from the coarser level, `q = ReLU(W_s s + W_g g)` and
`alpha = sigmoid(psi^T q)`; the gated skip feature is `alpha * s`. All three weights are 1x1x1 convolutions.

### Modality-attentive fusion

The MAF stem gives every sequence its own convolutional stem. A shared MLP scores the globally pooled descriptor of each modality
per channel, and a softmax across the modalities yields weights that sum to one; the weighted sum of the modality features is
concatenated with the individual features and mixed by a 1x1x1 convolution into the first encoder width. The weights of the last
forward pass are available as `ModalityAttentiveFusion.last_weights` for inspection.

### Deep supervision

With `deep_supervision=true` the three coarsest decoder levels have auxiliary 1x1x1 outputs; the loss weights are 4/7, 2/7 and
1/7 (halving per level).

## Discriminator

`bratsgan.models.Discriminator3D` is a conditional 3D PatchGAN with spectral normalisation. It receives the four-channel image
together with a three-channel segmentation (the reference or the generator output) and returns a map of patch scores. It is used
in the adversarial stage only and is discarded at inference.

## Losses (`bratsgan/losses.py`)

* `soft_dice_loss`, `tversky_loss(alpha, beta, gamma)` and binary cross-entropy, averaged over the three regions;
  `SegmentationLoss` combines them with the weights `w_dice`, `w_tv`, `w_bce` and applies deep supervision.
* Tversky: `TP / (TP + alpha FP + beta FN)`; `alpha` penalises false positives and `beta` false negatives. `alpha = beta = 0.5`
  recovers Dice.
* Adversarial: least-squares (`d_loss_lsgan`, `g_loss_lsgan`) or hinge objectives with one-sided label smoothing and instance
  noise; the adversarial weight is ramped up linearly over `adv_ramp_epochs`.

## Cost

`bratsgan.complexity.profile(model, input_shape)` returns exact parameter and multiply-accumulate counts from a forward pass on the
`meta` device, so no memory is allocated:

```bash
bratsgan info --config configs/a5_full.json
python examples/05_compare_variants.py 128
```
