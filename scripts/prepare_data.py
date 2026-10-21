"""Pre-process BraTS 2020 into the cache and create patient-level, grade-stratified splits.

    python scripts/prepare_data.py --root /data/BraTS2020 --cache cache --splits splits.json
"""
import argparse, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bratsgan.data import find_cases, make_splits, preprocess_case
from bratsgan.utils import write_json


def _job(args):
    case, out = args
    return preprocess_case(case, Path(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--splits", default="splits.json")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=0.2)
    a = ap.parse_args()
    cases = find_cases(a.root)
    print(f"{len(cases)} cases found; grades: " + str({g: sum(c['grade'] == g for c in cases) for g in ('HGG', 'LGG', None)}))
    with ProcessPoolExecutor(a.workers) as ex:
        list(ex.map(_job, [(c, a.cache) for c in cases]))
    splits = make_splits(cases, a.seed, a.test_frac, a.folds)
    write_json(a.splits, splits)
    print(f"test: {len(splits['test'])}  folds: {[len(f) for f in splits['folds']]}  -> {a.splits}")


if __name__ == "__main__":
    main()
