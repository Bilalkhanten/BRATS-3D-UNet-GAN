"""Assemble the thesis .docx.

    python assemble.py --template REF.docx --out thesis.docx [--results RESULTS_DIR]

Two passes are made so the table of contents and the lists of figures/tables
carry correct page numbers (pages are obtained by rendering with LibreOffice).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import docx  # noqa: E402
from docx.oxml import parse_xml  # noqa: E402
from docx.oxml.ns import nsdecls, qn  # noqa: E402

import tbuild as T  # noqa: E402
from tbuild import el, norm, para, ppr, run  # noqa: E402

TITLE = "3D BRAIN TUMOR SEGMENTATION AND GRADE CLASSIFICATION WITH 3D U-NET AND GANS"
TITLE_CASE = "3D Brain Tumor Segmentation and Grade Classification with 3D U-Net and GANs"
YEAR = "2026"

STYLE_XML = {
    "FigureCaption": '<w:style %s w:type="paragraph" w:customStyle="1" w:styleId="FigureCaption"><w:name w:val="Figure Caption"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:jc w:val="center"/></w:pPr><w:rPr><w:sz w:val="22"/></w:rPr></w:style>',
    "TableCaption": '<w:style %s w:type="paragraph" w:customStyle="1" w:styleId="TableCaption"><w:name w:val="Table Caption"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:jc w:val="center"/></w:pPr><w:rPr><w:sz w:val="22"/></w:rPr></w:style>',
    "TOC1": '<w:style %s w:type="paragraph" w:styleId="TOC1"><w:name w:val="toc 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:uiPriority w:val="39"/><w:unhideWhenUsed/><w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style>',
    "TOC2": '<w:style %s w:type="paragraph" w:styleId="TOC2"><w:name w:val="toc 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:uiPriority w:val="39"/><w:unhideWhenUsed/><w:rPr><w:sz w:val="22"/></w:rPr></w:style>',
    "TOC3": '<w:style %s w:type="paragraph" w:styleId="TOC3"><w:name w:val="toc 3"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:uiPriority w:val="39"/><w:unhideWhenUsed/><w:rPr><w:sz w:val="22"/></w:rPr></w:style>',
    "TableofFigures": '<w:style %s w:type="paragraph" w:styleId="TableofFigures"><w:name w:val="table of figures"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:uiPriority w:val="99"/><w:unhideWhenUsed/><w:rPr><w:sz w:val="22"/></w:rPr></w:style>',
    "Heading3": '<w:style %s w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:next w:val="BodyText"/><w:uiPriority w:val="9"/><w:qFormat/><w:pPr><w:keepNext/><w:keepLines/><w:outlineLvl w:val="2"/></w:pPr><w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style>',
}


def setup_styles(doc):
    styles = doc.styles.element
    for sid, xml in STYLE_XML.items():
        for old in styles.findall(qn("w:style")):
            if old.get(qn("w:styleId")) == sid:
                styles.remove(old)
        styles.append(parse_xml(xml % nsdecls("w")))


def set_par_text(p_el, text):
    """Replace visible text of a paragraph, keeping the first text run's format."""
    runs = [r for r in p_el.iter(qn("w:r")) if r.find(qn("w:t")) is not None]
    if not runs:
        return False
    first = runs[0]
    ts = first.findall(qn("w:t"))
    ts[0].text = text
    ts[0].set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    for t in ts[1:]:
        t.text = ""
    for r in runs[1:]:
        for t in r.findall(qn("w:t")):
            t.text = ""
    return True


def replace_in_runs(p_el, old, new):
    for t in p_el.iter(qn("w:t")):
        if t.text and old in t.text:
            t.text = t.text.replace(old, new)
            return True
    return False


def patch_front(doc):
    body = doc.element.body
    kids = list(body)
    set_par_text(kids[9], TITLE)
    set_par_text(kids[25], TITLE)
    set_par_text(kids[12], "[Student Full Name]")
    set_par_text(kids[6], "[Program]")
    # year: runs '202','5'
    for idx in (23, 55):
        for t in kids[idx].iter(qn("w:t")):
            if t.text == "5":
                t.text = "6"
    # acknowledgements placeholder
    set_par_text(kids[75], "[To be completed by the author.]")
    # delete everything after the front matter except final sectPr
    for k in kids[82:-1]:
        body.remove(k)


# ---------------------------------------------------------------------------
def abbreviations():
    items = [
        ("3D", "Three-dimensional"), ("AMP", "Automatic mixed precision"),
        ("AG", "Attention gate"), ("AUC", "Area under the ROC curve"),
        ("BCE", "Binary cross-entropy"), ("BraTS", "Brain Tumor Segmentation (challenge)"),
        ("CI", "Confidence interval"), ("CNN", "Convolutional neural network"),
        ("CV", "Cross-validation"), ("DSC", "Dice similarity coefficient"),
        ("ECE", "Expected calibration error"), ("ED", "Peritumoral edema"),
        ("ET", "Enhancing tumor"), ("FLAIR", "Fluid-attenuated inversion recovery"),
        ("FN / FP", "False negative / false positive"), ("GAN", "Generative adversarial network"),
        ("GBM", "Glioblastoma"), ("GMACs", "Giga multiply-accumulate operations"),
        ("HD95", "95th-percentile Hausdorff distance"), ("HGG / LGG", "High-grade / low-grade glioma"),
        ("IoU", "Intersection over union"), ("LSGAN", "Least-squares GAN"),
        ("MAF", "Modality-attentive fusion"), ("MDE", "Minimum detectable effect"),
        ("MRI", "Magnetic resonance imaging"), ("NCR/NET", "Necrotic and non-enhancing tumor core"),
        ("nnU-Net", "No-new-U-Net (self-configuring segmentation framework)"),
        ("OOF", "Out-of-fold"), ("ROC", "Receiver operating characteristic"),
        ("T1 / T1ce / T2", "T1-weighted / contrast-enhanced T1 / T2-weighted MRI"),
        ("TC", "Tumor core"), ("TTA", "Test-time augmentation"),
        ("WHO", "World Health Organization"), ("WT", "Whole tumor"),
    ]
    return items


def build_abbrev(b: T.Builder):
    b.front_heading("LIST OF ABBREVIATIONS", toc=False)
    for a, d in abbreviations():
        pp = ppr("BodyText", jc="left", after=40, line=276, left=2160, hanging=2160,
                 tabs=[(2160, "left", None)])
        b.add(para(pp, [run(a, bold=True, size=11), run("\t" + d, size=11)]))


def build_abstract(b: T.Builder, abstract_paras, keywords, total_pages):
    # page 1: abstract header block (as in the reference)
    pp = ppr("Heading1", jc="center", before=0, after=360, line=360, page_break=True, keep_next=True,
             left=0)
    b.add(para(pp, [run("ABSTRACT", bold=True, size=14)]))
    lines = [
        (TITLE, True), ("", False),
        ("[Author’s Surname, Name],", False), ("[M.Sc./PhD], [Department], Altınbaş University,", False), ("", False),
        ("Supervisor: [Supervisor]", False), ("Co-Supervisor: [Co-Supervisor, if any]", False), ("", False),
        (f"Date: [Month] {YEAR}", False),
        (f"Pages: {total_pages if total_pages else '___'}", False),
    ]
    for t, bold in lines:
        b.add(para(ppr("BodyText", jc="center", after=60, line=360), [run(t, bold=bold, size=12)] if t else []))
    # page 2: abstract text
    first = True
    for t in abstract_paras:
        pp = ppr("BodyText", jc="both", after=120, line=300, page_break=first)
        b.add(para(pp, b.inline(t)))
        first = False
    b.add(para(ppr("BodyText", jc="both", before=120, after=140, line=300),
               [run("Keywords: ", bold=True)] + [run(keywords)]))


# ---------------------------------------------------------------------------
def anchor(b: T.Builder):
    p = para(ppr("BodyText", after=0), [])
    b.add(p)
    return p


def fill_at(b: T.Builder, anch, fn):
    saved = b.sectPr
    b.sectPr = anch
    try:
        fn()
    finally:
        b.sectPr = saved
    anch.getparent().remove(anch)


_ORD = {
    "pPr": "pStyle keepNext keepLines pageBreakBefore framePr widowControl numPr suppressLineNumbers pBdr shd tabs suppressAutoHyphens kinsoku wordWrap overflowPunct topLinePunct autoSpaceDE autoSpaceDN bidi adjustRightInd snapToGrid spacing ind contextualSpacing mirrorIndents suppressOverlap jc textDirection textAlignment textboxTightWrap outlineLvl divId cnfStyle rPr sectPr pPrChange".split(),
    "rPr": "rStyle rFonts b bCs i iCs caps smallCaps strike dstrike outline shadow emboss imprint noProof snapToGrid vanish webHidden color spacing w kern position sz szCs highlight u effect bdr shd fitText vertAlign rtl cs em lang eastAsianLayout specVanish oMath".split(),
    "tblPr": "tblStyle tblpPr tblOverlap bidiVisual tblStyleRowBandSize tblStyleColBandSize tblW jc tblCellSpacing tblInd tblBorders shd tblLayout tblCellMar tblLook".split(),
    "tcPr": "cnfStyle tcW gridSpan hMerge vMerge tcBorders shd noWrap tcMar textDirection tcFitText vAlign hideMark".split(),
}


def fix_order(root):
    """Put the children of property elements in the order of the OOXML schema (Word is strict about it)."""
    W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    n = 0
    for name, order in _ORD.items():
        for e in root.iter("{%s}%s" % (W, name)):
            kids = list(e)
            def key(c):
                ln = c.tag.split("}")[-1]
                return order.index(ln) if ln in order else len(order)
            srt = sorted(kids, key=key)
            if srt != kids:
                for c in kids:
                    e.remove(c)
                for c in srt:
                    e.append(c)
                n += 1
    return n


def build(template, out, content_chapters, content_appendices, refs, abstract_paras, keywords,
          pagemap=None, total_pages=None):
    b = T.Builder(template, refs, pagemap, total_pages)
    setup_styles(b.doc)
    patch_front(b.doc)
    blocks = []
    for txt, src in content_chapters + content_appendices:
        blocks += T.parse(txt, src)
    b.blocks = blocks
    b.number_blocks()
    b.collect_math()
    # abstract math (none expected) – registered through inline() would fail, keep plain text
    b.convert_math()

    build_abstract(b, abstract_paras, keywords, total_pages)
    a_toc = anchor(b)
    a_lof = anchor(b)
    a_lot = anchor(b)
    build_abbrev(b)
    n_chap_blocks = 0
    # chapters
    first_app = next((i for i, x in enumerate(blocks) if x.kind == "appendix"), len(blocks))
    b.emit_blocks(blocks[:first_app])
    a_ref = anchor(b)
    b.emit_blocks(blocks[first_app:])
    # references (after all citations are known)
    fill_at(b, a_ref, b.references)
    # the references precede the appendices in the document: move their TOC entry accordingly
    ents = b.toc_entries
    ri = next((i for i, e in enumerate(ents) if e[2] == "REFERENCES"), None)
    ai = next((i for i, e in enumerate(ents) if e[2].startswith("APPENDIX")), None)
    if ri is not None and ai is not None and ri > ai:
        ents.insert(ai, ents.pop(ri))
    fill_at(b, a_toc, b.build_toc)

    def lists():
        pass

    def lof():
        ents = [(1, f"{lab} {cap}") for lab, cap in b.fig_entries]
        sf = lambda lvl: ppr("TableofFigures", jc="left", after=60, line=276, left=1100, hanging=1100,
                             right=600, tabs=[(T.TEXT_W, "right", "dot")])
        b.toc_block(ents, 'TOC \\h \\z \\t "Figure Caption,1"', sf, "LIST OF FIGURES",
                    lambda t: b.pagemap.get(entry_key(*t.split(" ", 1))))

    def lot():
        ents = [(1, f"{lab} {cap}") for lab, cap in b.tab_entries]
        sf = lambda lvl: ppr("TableofFigures", jc="left", after=60, line=276, left=1100, hanging=1100,
                             right=600, tabs=[(T.TEXT_W, "right", "dot")])
        b.toc_block(ents, 'TOC \\h \\z \\t "Table Caption,1"', sf, "LIST OF TABLES",
                    lambda t: b.pagemap.get(entry_key(*t.split(" ", 1))))

    fill_at(b, a_lof, lof)
    fill_at(b, a_lot, lot)
    # core properties
    cp = b.doc.core_properties
    cp.title = TITLE_CASE
    cp.author = ""
    cp.last_modified_by = ""
    cp.subject = "Doctoral thesis"
    fix_order(b.doc.element)
    fix_order(b.doc.styles.element)
    b.doc.save(out)
    return b


# ---------------------------------------------------------------------------
def render_pdf(docx_path, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(outdir), str(docx_path)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
    return outdir / (Path(docx_path).stem + ".pdf")


def pdf_pages(pdf):
    txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    return txt.split("\f")


def entry_key(lab, cap):
    words = re.sub(r"[*$`]", "", cap).split()[:2]
    return norm(lab + " " + " ".join(words))


def make_pagemap(builder: T.Builder, pages):
    """page of every TOC / LoF / LoT entry: first page of the body on which a line starts with the entry text"""
    pm = {}
    lines = [[norm(l) for l in p.split("\n")] for p in pages]
    start = 0
    for i, L in enumerate(lines):
        if any(l == "LIST OF ABBREVIATIONS" for l in L):
            start = i + 1
    cands = [t for (_, _, t) in builder.toc_entries]
    cands += [entry_key(lab, cap) for lab, cap in builder.fig_entries]
    cands += [entry_key(lab, cap) for lab, cap in builder.tab_entries]
    for t in cands:
        key = norm(t)
        pref = key[:45]
        hit = None
        for i in range(start, len(pages)):
            if any(l.startswith(pref) or (len(l) >= 20 and pref.startswith(l)) for l in lines[i]):
                hit = i + 1
                break
        pm[key] = hit
    return pm
