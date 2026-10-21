# Configurations

Each file overrides a subset of the defaults in `bratsgan/config.py` (see [docs/configuration.md](../docs/configuration.md)).

| File | Description |
| --- | --- |
| `a0_unet.json` | baseline: plain 3D U-Net, early fusion, Dice + BCE |
| `a1_res.json` | A0 + residual blocks |
| `a2_att.json` | A1 + attention gates |
| `a3_maf.json` | A2 + modality-attentive fusion stem |
| `a4_tversky.json` | A3 + Tversky loss (0.3, 0.7) |
| `a5_full.json` | A4 + adversarial refinement (full model) |
| `t_dice_eq.json`, `t_fn_heavy.json` | Tversky sensitivity: (0.5, 0.5) and (0.2, 0.8) on top of A3 |
| `g_lam001.json`, `g_lam020.json` | adversarial weight 0.01 and 0.2 |
| `g_hinge.json`, `g_nosmooth.json` | hinge loss; no label smoothing and no instance noise |
| `quick.json` | shortened schedule for a first run on real data |
