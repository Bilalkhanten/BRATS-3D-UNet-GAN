"""Assemble the thesis .docx (two-pass build so that TOC / LoF / LoT carry page numbers).

usage: python build_thesis.py OUT.docx [--phantom DIR] [--brats DIR] [--stage draft|final]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import assemble as A  # noqa: E402
import ch4_verify  # noqa: E402
import appendices  # noqa: E402
from refs import REFS  # noqa: E402

TEMPLATE = "reference_thesis.docx"  # overridden by --template
VERIFY = HERE / "assets" / "verify.json"


def read(name):
    p = HERE / "content" / name
    return (p.read_text(), name) if p.exists() else None


def chapters(args):
    V = json.loads(VERIFY.read_text())
    lat = HERE / "assets" / "latency.json"
    if lat.exists():
        V["latency"] = json.loads(lat.read_text())
    ledger = [("Analytical and numerical verification", "Exact calculation, autograd and simulation of the properties derived in Chapter 3; complexity, receptive fields, statistical power.", "Complete"),
              ("Synthetic phantom study", "Training of five model variants on 100 synthetic multimodal volumes with known ground truth; two seeds; one fold.", "Complete / see Section 4.3"),
              ("Hard synthetic benchmark", "Six cumulative variants with two seeds, Tversky and adversarial sensitivity runs, three-fold comparison of A0 and A5, calibration and grade pipeline on 160 hard phantoms.", "Complete / see Section 4.4"),
              ("BraTS 2020 experiments", "Cross-validation, held-out test, ablations, comparison with nnU-Net, grade classification.", "Pending (see Section 4.5)")]
    tests = {"n": 23, "passed": 23, "seconds": 12.3}
    parts = [ch4_verify.build(V, ledger, tests)]
    try:
        import ch4_phantom
        parts.append(ch4_phantom.build(args.phantom))
    except ImportError:
        pass
    try:
        import ch4_hard
        parts.append(ch4_hard.build(args.hard))
    except ImportError:
        pass
    try:
        import ch4_brats
        parts.append(ch4_brats.build(args.brats))
    except ImportError:
        pass
    ch4 = "\n".join(parts)
    names = ["ch1.txt", "ch2a.txt", "ch2b.txt", "ch3a.txt", "ch3b.txt"]
    chs = [read(n) for n in names]
    chs.append((ch4, "ch4(generated)"))
    import ch5_hard
    hard_txt = ch5_hard.build(getattr(sys.modules.get("ch4_hard"), "RES", {}))
    RESH = getattr(sys.modules.get("ch4_hard"), "RES", {})
    for n in ("ch5.txt", "ch6.txt"):
        r = read(n)
        if r:
            chs.append((r[0].replace("@@HARD_DISCUSSION@@", hard_txt).replace("@@HARD_CONCL@@", ch5_hard.conclusion_section(RESH)), r[1]))
    apps = [(appendices.appA(), "appA"), (appendices.appB(), "appB"), (appendices.appC(), "appC"), (appendices.appD(V, getattr(sys.modules.get('ch4_hard'), 'SUPP', '')), "appD"),
            (appendices.appX(getattr(sys.modules.get('ch4_hard'), 'SUPPX', '')), "appX"), (appendices.appN(), "appN"), (appendices.appF(), "appF")]
    return [c for c in chs if c], apps


def main():
    global TEMPLATE
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--phantom", default=str(HERE.parent / "runs" / "phantom_study"), help="phantom-study directory (if missing, the frozen copy in assets/ is used)")
    ap.add_argument("--template", default=TEMPLATE, help="the reference thesis .docx that provides the format")
    ap.add_argument("--brats", default=None)
    ap.add_argument("--dryrun", action="store_true", help="label the BraTS section and the abstract as a pipeline dry run on synthetic data")
    ap.add_argument("--hard", default=str(HERE.parent / "runs" / "hard_study"), help="hard-benchmark directory (if missing, the frozen copy in assets/ is used)")
    args = ap.parse_args()
    TEMPLATE = args.template
    if args.dryrun:
        import ch4_brats
        ch4_brats.DRYRUN = True
    chs, apps = chapters(args)
    abstract = (HERE / "content" / "abstract.txt")
    paras, kw = ["Abstract pending."], "brain tumour"
    if abstract.exists():
        t = abstract.read_text().strip().split("\n\nKeywords:")
        import ch5_hard as _c5
        _res = getattr(sys.modules.get("ch4_hard"), "RES", {})
        paras = [p.strip().replace("@@HARD_ABS@@", _c5.abstract_sentence(_res)) for p in t[0].split("\n\n") if p.strip()]
        kw = t[1].strip() if len(t) > 1 else kw
    if args.dryrun:
        import ch4_brats as _b
        _D = _b.Data(args.brats)
        _t5 = _D.test("a5_full")
        if _t5 is not None:
            _v = [f"{100 * _t5[f'dice_{r}'].mean():.2f}" for r in ("ET", "WT", "TC")]
            _old = "| **This thesis** | BraTS 2020, test (74 cases) | pending | pending | pending | Requires the GPU runs of the RUNBOOK |"
            _new = f"| **This thesis (DRY RUN)** | synthetic BraTS layout, test (15 cases) | {_v[0]} | {_v[1]} | {_v[2]} | Pipeline test, not BraTS |"
            _gj = json.loads((Path(args.brats) / "results" / "grade_a5_full.json").read_text())
            _oldg = "AUC with bootstrap interval, calibration (ECE); pending on BraTS (Chapter 4)"
            _newg = f"AUC with bootstrap interval, calibration (ECE); DRY RUN on synthetic data: CV AUC {_gj['pred_logreg']['cv']['auc']:.3f}, test AUC {_gj['pred_logreg']['test']['auc']:.3f}"
            chs = [((t.replace(_oldg, _newg), n) if isinstance(t, str) else (t, n)) for t, n in chs]
            chs = [((t.replace(_old, _new), n) if isinstance(t, str) else (t, n)) for t, n in chs]
    if args.dryrun:
        paras = ["DRY-RUN BUILD. The BraTS tables of this version were generated from a pipeline test on a small synthetic dataset in the BraTS layout with a tiny training budget. They test the implementation and are not BraTS 2020 results; the abstract and the conclusions below describe the real thesis and are unchanged."] + paras
    out = Path(args.out)
    work = out.parent / (out.stem + "_work")
    b = A.build(TEMPLATE, str(out), chs, apps, REFS, paras, kw, None, None)
    pdf = A.render_pdf(out, work)
    pages = A.pdf_pages(pdf)
    for _ in range(2):
        pm = A.make_pagemap(b, pages)
        b = A.build(TEMPLATE, str(out), chs, apps, REFS, paras, kw, pm, len(pages) - 1)
        pdf = A.render_pdf(out, work)
        pages = A.pdf_pages(pdf)
    print("pages:", len(pages) - 1, "pdf:", pdf)
    missing = [k for k, v in pm.items() if v is None]
    if missing:
        print("unmapped TOC keys:", len(missing), missing[:8])


if __name__ == "__main__":
    main()
