"""Generate one HGG-like and one LGG-like phantom and save a PNG with the four sequences and the labels.

    python examples/01_synthetic_quickstart.py [output.png]
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bratsgan import MODALITIES  # noqa: E402
from bratsgan.synthetic import make_phantom  # noqa: E402

LABEL_COLOURS = {1: (0.16, 0.47, 0.84), 2: (0.11, 0.69, 0.48), 4: (0.95, 0.76, 0.19)}


def overlay(label_slice):
    rgb = np.zeros(label_slice.shape + (4,))
    for k, c in LABEL_COLOURS.items():
        rgb[label_slice == k] = (*c, 1.0)
    return rgb


def main(out="phantoms.png"):
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.6))
    for row, grade in enumerate(("HGG", "LGG")):
        vols, label, _brain = make_phantom(size=64, grade=grade, seed=3)
        z = int(np.argwhere(label > 0)[:, 0].mean())  # axial slice through the tumour
        for col, name in enumerate(MODALITIES):
            axes[row, col].imshow(vols[col, z], cmap="gray")
            axes[row, col].set_title(f"{grade} {name}", fontsize=9)
        axes[row, 4].imshow(vols[1, z], cmap="gray")
        axes[row, 4].imshow(overlay(label[z]))
        axes[row, 4].set_title("labels (1 NCR, 2 ED, 4 ET)", fontsize=9)
        print(f"{grade}: voxels per label =", {int(k): int((label == k).sum()) for k in np.unique(label)})
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main(*sys.argv[1:2])
