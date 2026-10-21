"""End-to-end demonstration on synthetic data: generate phantoms, pre-process, split, train a tiny network, evaluate.

This runs on a CPU in a few minutes. It demonstrates the pipeline; the numbers are NOT BraTS results.

    python examples/02_train_tiny_model.py [work_dir]
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bratsgan.config import load_config  # noqa: E402
from bratsgan.data import find_cases, make_splits, preprocess_case  # noqa: E402
from bratsgan.synthetic import write_brats_layout  # noqa: E402
from bratsgan.train import run  # noqa: E402
from bratsgan.utils import read_json, write_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TINY = dict(base=8, depth=4, max_ch=64, patch=[32, 32, 32], batch_size=2, iters_per_epoch=10, epochs_seg=4, epochs_adv=2,
            val_every=2, num_workers=0, amp=False, adv_ramp_epochs=1, ndf=8)


def main(work=None):
    work = Path(work or tempfile.mkdtemp(prefix="bratsgan_demo_"))
    raw = write_brats_layout(work / "raw", n_hgg=12, n_lgg=4, size=40, seed=5)
    cases = find_cases(str(raw))
    for c in cases:
        preprocess_case(c, work / "cache")
    write_json(work / "splits.json", make_splits(cases, seed=1, test_frac=0.25, k=2))

    cfg = load_config(str(ROOT / "configs" / "a5_full.json"),
                      dict(TINY, name="demo", out_dir=str(work / "runs"), cache_dir=str(work / "cache"), splits=str(work / "splits.json")))
    out = run(cfg)
    summary = read_json(out / "test_summary.json")
    print("\nrun directory:", out)
    for r in ("WT", "TC", "ET"):
        print(f"test {r}: Dice {summary[f'dice_{r}']['mean']:.3f}  HD95 {summary[f'hd95_{r}']['mean']:.1f} voxels")
    print("(tiny network, a few epochs, synthetic data: these numbers only show that the pipeline runs)")


if __name__ == "__main__":
    main(*sys.argv[1:2])
