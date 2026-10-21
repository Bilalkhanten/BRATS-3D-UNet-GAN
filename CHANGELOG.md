# Changelog

## 1.0.0

* Configurable 3D U-Net (residual blocks, attention gates, MAF stem, deep supervision) and conditional 3D PatchGAN.
* Dice / Tversky / BCE losses, LSGAN and hinge adversarial objectives, two-stage training schedule.
* BraTS 2020 pre-processing, patient-level grade-stratified splits with leakage assertion.
* Sliding-window inference, per-patient metrics with BraTS empty-mask conventions, bootstrap/paired statistics.
* LGG/HGG grade classification from features of the predicted masks.
* Synthetic phantom generator (easy and hard settings), phantom studies and verification experiments.
* nnU-Net v2 export and evaluation scripts for a baseline on identical splits.
* Thesis document builder.
* Command-line tool `bratsgan`, examples, notebooks, CI, Docker image, documentation.
