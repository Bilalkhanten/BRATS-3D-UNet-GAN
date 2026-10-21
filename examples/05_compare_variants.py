"""Parameter count and cost of the six ablation variants A0-A5 (and the discriminator) for a given patch size.

    python examples/05_compare_variants.py [patch_size]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bratsgan.complexity import count_params, profile  # noqa: E402
from bratsgan.config import load_config  # noqa: E402
from bratsgan.models import Discriminator3D, build_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ["a0_unet", "a1_res", "a2_att", "a3_maf", "a4_tversky", "a5_full"]


def main(patch=64):
    patch = int(patch)
    print(f"input 1 x 4 x {patch}^3\n")
    print(f"{'variant':<12}{'parameters':>14}{'GMACs':>10}")
    for v in VARIANTS:
        cfg = load_config(str(ROOT / "configs" / f"{v}.json"))
        model = build_model(cfg)
        r = profile(model, (1, cfg["in_ch"], patch, patch, patch))
        print(f"{v:<12}{int(r['params']):>14,}{r['gmacs']:>10.1f}")
    d = Discriminator3D()
    print(f"\ndiscriminator parameters: {count_params(d):,} (training only; not used at inference)")


if __name__ == "__main__":
    main(*sys.argv[1:2])
