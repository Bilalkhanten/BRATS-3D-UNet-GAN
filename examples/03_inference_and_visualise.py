"""Segment one case with a trained run and save an overlay figure (reference | prediction | error).

    python examples/03_inference_and_visualise.py RUN_DIR CASE_ID [output.png]

RUN_DIR is a directory written by ``bratsgan.train`` (it contains config.json and best.pt). CASE_ID is a case of the cache
the run was trained on, for example BraTS20_Training_005.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bratsgan.data import labels_to_regions, load_case  # noqa: E402
from bratsgan.evaluate import load_for_eval  # noqa: E402
from bratsgan.infer import predict_regions  # noqa: E402
from bratsgan.metrics import enforce_hierarchy, region_metrics  # noqa: E402
from bratsgan.utils import read_json  # noqa: E402


def main(run_dir, case_id, out="overlay.png"):
    run = Path(run_dir)
    cfg = read_json(run / "config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_for_eval(cfg, str(run / "best.pt"), device)
    img, seg, _meta = load_case(cfg["cache_dir"], case_id, mmap=False)
    pred, _prob, secs = predict_regions(model, np.asarray(img, np.float32), cfg["patch"], cfg["final_overlap"], device=device)
    pred = enforce_hierarchy(pred)
    gt = labels_to_regions(np.asarray(seg))
    scores = region_metrics(pred, gt)
    print(f"inference {secs:.2f} s;", {k: round(float(v), 3) for k, v in scores.items() if k.startswith("dice")})

    z = int(np.argwhere(gt[0] > 0)[:, 0].mean())
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
    for a, (title, m) in zip(ax, (("reference (WT / TC / ET)", gt), ("prediction", pred), ("error", None))):
        a.imshow(np.asarray(img)[1, z], cmap="gray")
        if m is not None:
            a.imshow(np.ma.masked_equal(m[0, z] + m[1, z] + m[2, z], 0), cmap="viridis", alpha=0.6, vmin=0, vmax=3)
        else:
            fp, fn = (pred[0, z] > 0) & ~(gt[0, z] > 0), (gt[0, z] > 0) & ~(pred[0, z] > 0)
            a.imshow(np.ma.masked_equal(fp * 1 + fn * 2, 0), cmap="autumn", alpha=0.8)
        a.set_title(title, fontsize=9)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(*sys.argv[1:4])
