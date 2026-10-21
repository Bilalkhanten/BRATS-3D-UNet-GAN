"""Command-line entry point: ``bratsgan <command>``.

Commands
--------
synth      write synthetic phantoms in the BraTS 2020 folder layout
prepare    pre-process a BraTS-layout folder into the cache and write patient-level splits
train      train one experiment (same options as ``python -m bratsgan.train``)
evaluate   evaluate a finished run (same options as ``python -m bratsgan.evaluate``)
info       print the parameter count and cost of an architecture variant
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _synth(a) -> None:
    from .synthetic import write_brats_layout

    root = write_brats_layout(a.out, n_hgg=a.hgg, n_lgg=a.lgg, size=a.size, seed=a.seed, difficulty=a.difficulty)
    print(f"wrote {a.hgg + a.lgg} phantoms to {root}")


def _prepare(a) -> None:
    from concurrent.futures import ProcessPoolExecutor

    from .data import find_cases, make_splits, preprocess_case
    from .utils import write_json

    cases = find_cases(a.root)
    print(f"{len(cases)} cases found")
    out = Path(a.cache)
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            list(ex.map(preprocess_case, cases, [out] * len(cases)))
    else:
        for c in cases:
            preprocess_case(c, out)
    splits = make_splits(cases, a.seed, a.test_frac, a.folds)
    write_json(a.splits, splits)
    print(f"test: {len(splits['test'])}  folds: {[len(f) for f in splits['folds']]}  -> {a.splits}")


def _info(a) -> None:
    from .config import load_config, parse_overrides
    from .complexity import profile
    from .models import build_model

    cfg = load_config(a.config, parse_overrides(a.set))
    model = build_model(cfg)
    shape = (1, cfg["in_ch"], *cfg["patch"])
    r = profile(model, shape)
    for k, v in r.items():
        print(f"{k:>24s}: {v:,.3f}" if isinstance(v, float) else f"{k:>24s}: {v:,}")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="bratsgan", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("synth", help="write synthetic phantoms (BraTS layout)")
    p.add_argument("--out", required=True)
    p.add_argument("--hgg", type=int, default=24)
    p.add_argument("--lgg", type=int, default=8)
    p.add_argument("--size", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--difficulty", choices=["easy", "hard"], default="easy")
    p.set_defaults(fn=_synth)

    p = sub.add_parser("prepare", help="pre-process and split")
    p.add_argument("--root", required=True)
    p.add_argument("--cache", default="cache")
    p.add_argument("--splits", default="splits.json")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.set_defaults(fn=_prepare)

    p = sub.add_parser("info", help="parameters and cost of a configuration")
    p.add_argument("--config", default=None)
    p.add_argument("--set", nargs="*", default=[])
    p.set_defaults(fn=_info)

    sub.add_parser("train", help="train an experiment (see python -m bratsgan.train -h)", add_help=False)
    sub.add_parser("evaluate", help="evaluate a run (see python -m bratsgan.evaluate -h)", add_help=False)

    if argv and argv[0] in ("train", "evaluate"):
        sys.argv = [f"bratsgan {argv[0]}"] + argv[1:]
        if argv[0] == "train":
            from .train import main as m
        else:
            from .evaluate import main as m
        m()
        return 0
    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
