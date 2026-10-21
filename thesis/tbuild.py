"""Thesis builder: light markup -> .docx in the Altinbas reference format.

The reference thesis supplies the cover, title, approval and declaration pages
(front matter, body children 0..81).  Everything from the abstract onwards is
generated here with direct formatting that mimics the reference (Times New
Roman 12 pt, 1.5 line spacing, justified, 1-inch margins).

Markup (one directive per line, blocks end with ``@end``):

    @chapter [label] TITLE            numbered chapter (page break)
    @section [label] TITLE            1.1.
    @subsection [label] Title         1.1.1.
    @subsubsection Title              run-in bold-italic, not in TOC
    @eq label ... @end                LaTeX display equation, numbered
    @fig label | file=PATH | width=5.8 \n caption \n @end
    @tab label | widths=1.5,1,1 | align=lcc \n caption \n | a | b | \n @end
    @algo label \n caption \n lines \n @end
    @code \n lines \n @end
    @note \n paragraph(s) \n @end     shaded call-out
    @pagebreak
    - bullet / 1. numbered (one item per line)
Inline: **bold**  *italic*  `code`  $latex$  [@key1; @key2]
        @fig:label @tab:label @eq:label @sec:label @chap:label @app:label
"""
from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Emu, Inches, Pt

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
FONT = "Times New Roman"
MONO = "Courier New"
TEXT_W = 9360  # twips, 6.5 in


# --------------------------------------------------------------------------
# helpers for raw OOXML
# --------------------------------------------------------------------------
def el(tag, **attrs):
    e = OxmlElement(tag)
    for k, v in attrs.items():
        e.set(qn(k.replace("_", ":", 1)), str(v))
    return e


def rpr(bold=False, italic=False, mono=False, size=None, color=None, sup=False,
        sub=False, caps=False, underline=False):
    r = el("w:rPr")
    if mono:
        r.append(el("w:rFonts", w_ascii=MONO, w_hAnsi=MONO, w_cs=MONO))
    if bold:
        r.append(el("w:b"))
        r.append(el("w:bCs"))
    if italic:
        r.append(el("w:i"))
        r.append(el("w:iCs"))
    if caps:
        r.append(el("w:caps"))
    if color:
        r.append(el("w:color", w_val=color))
    if size:
        r.append(el("w:sz", w_val=int(size * 2)))
        r.append(el("w:szCs", w_val=int(size * 2)))
    if underline:
        r.append(el("w:u", w_val="single"))
    if sup:
        r.append(el("w:vertAlign", w_val="superscript"))
    if sub:
        r.append(el("w:vertAlign", w_val="subscript"))
    return r


def run(text, **kw):
    r = el("w:r")
    pr = rpr(**kw)
    if len(pr):
        r.append(pr)
    parts = text.split("\t")
    for i, part in enumerate(parts):
        if i:
            r.append(el("w:tab"))
        if part:
            t = el("w:t")
            t.text = part
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            r.append(t)
    return r


def fld_runs(instr, cached="", **kw):
    """Complex field as a list of runs (begin, instr, separate, cached, end)."""
    out = []
    r = el("w:r"); r.append(el("w:fldChar", w_fldCharType="begin")); out.append(r)
    r = el("w:r"); it = el("w:instrText"); it.text = f" {instr} "
    it.set("{http://www.w3.org/XML/1998/namespace}space", "preserve"); r.append(it); out.append(r)
    r = el("w:r"); r.append(el("w:fldChar", w_fldCharType="separate")); out.append(r)
    out.append(run(cached, **kw))
    r = el("w:r"); r.append(el("w:fldChar", w_fldCharType="end")); out.append(r)
    return out


def ppr(style="BodyText", jc=None, before=None, after=None, line=None, left=None,
        right=None, hanging=None, first=None, keep_next=False, keep_lines=False,
        page_break=False, tabs=None, outline=None, shade=None, border_left=None,
        widow=True):
    p = el("w:pPr")
    if style:
        p.append(el("w:pStyle", w_val=style))
    if keep_next:
        p.append(el("w:keepNext"))
    if keep_lines:
        p.append(el("w:keepLines"))
    if page_break:
        p.append(el("w:pageBreakBefore"))
    if widow:
        p.append(el("w:widowControl"))
    if border_left:
        b = el("w:pBdr"); b.append(el("w:left", w_val="single", w_sz=24, w_space=6, w_color=border_left)); p.append(b)
    if shade:
        p.append(el("w:shd", w_val="clear", w_color="auto", w_fill=shade))
    if tabs:
        t = el("w:tabs")
        for pos, kind, leader in tabs:
            a = dict(w_val=kind, w_pos=pos)
            if leader:
                a["w_leader"] = leader
            t.append(el("w:tab", **a))
        p.append(t)
    sp = {}
    if before is not None:
        sp["w_before"] = before
    if after is not None:
        sp["w_after"] = after
    if line is not None:
        sp["w_line"] = line
        sp["w_lineRule"] = "auto"
    if sp:
        p.append(el("w:spacing", **sp))
    ind = {}
    if left is not None:
        ind["w_left"] = left
    if right is not None:
        ind["w_right"] = right
    if hanging is not None:
        ind["w_hanging"] = hanging
    if first is not None:
        ind["w_firstLine"] = first
    if ind:
        p.append(el("w:ind", **ind))
    if jc:
        p.append(el("w:jc", w_val=jc))
    if outline is not None:
        p.append(el("w:outlineLvl", w_val=outline))
    return p


def para(pPr_el, children=()):
    p = el("w:p")
    p.append(pPr_el)
    for c in children:
        p.append(c)
    return p


# --------------------------------------------------------------------------
# math: LaTeX -> OMML through pandoc (batch)
# --------------------------------------------------------------------------
def latex_to_omml(snips: list[tuple[str, bool]]) -> list:
    """snips: list of (latex, display). returns list of lxml m:oMath elements."""
    if not snips:
        return []
    lines = []
    for tex, disp in snips:
        tex = tex.strip()
        lines.append(("$$%s$$" if disp else "$%s$") % tex)
        lines.append("")
    with tempfile.TemporaryDirectory() as td:
        md = Path(td) / "m.md"
        out = Path(td) / "m.docx"
        md.write_text("\n".join(lines), encoding="utf8")
        subprocess.run(["pandoc", str(md), "-f", "markdown+tex_math_dollars", "-o", str(out)], check=True)
        import zipfile
        from lxml import etree
        xml = zipfile.ZipFile(out).read("word/document.xml")
        root = etree.fromstring(xml)
    ns = {"w": W_NS, "m": M_NS}
    paras = root.findall(".//w:body/w:p", ns)
    res = []
    for (tex, _), p in zip(snips, paras):
        om = p.find(".//m:oMath", ns)
        if om is None:
            raise ValueError(f"pandoc could not convert math: {tex!r}")
        res.append(om)
    if len(paras) != len(snips):
        raise ValueError(f"math count mismatch {len(paras)} vs {len(snips)}")
    return res


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
class Block:
    def __init__(self, kind, **kw):
        self.kind = kind
        self.__dict__.update(kw)

    def __repr__(self):
        return f"Block({self.kind})"


def _opts(s):
    out = {}
    for part in s.split("|")[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def parse(text: str, source="") -> list[Block]:
    lines = text.split("\n")
    blocks: list[Block] = []
    i = 0
    buf: list[str] = []

    def flush():
        nonlocal buf
        if buf:
            blocks.append(Block("para", text=" ".join(s.strip() for s in buf)))
            buf = []

    while i < len(lines):
        ln = lines[i].rstrip()
        s = ln.strip()
        i += 1
        if not s:
            flush(); continue
        if s.startswith("%%"):  # comment
            continue
        m = re.match(r"@(chapter|section|subsection|subsubsection|appendix)(?:\[([\w\-]+)\])?\s+(.*)$", s)
        if m:
            flush()
            blocks.append(Block(m.group(1), label=m.group(2), title=m.group(3).strip()))
            continue
        if s == "@pagebreak":
            flush(); blocks.append(Block("pagebreak")); continue
        m = re.match(r"@(eq|fig|tab|algo|code|note)(?=\s|$)(.*)$", s)
        if m:
            flush()
            kind, rest = m.group(1), m.group(2)
            body = []
            while i < len(lines) and lines[i].strip() != "@end":
                body.append(lines[i].rstrip("\n"))
                i += 1
            i += 1
            head = rest.split("|")[0].strip()
            opts = _opts(rest)
            if kind == "eq":
                blocks.append(Block("eq", label=head or None, tex="\n".join(body).strip()))
            elif kind == "fig":
                cap = " ".join(b.strip() for b in body if b.strip())
                blocks.append(Block("fig", label=head, cap=cap, **opts))
            elif kind == "tab":
                cap_lines = [b.strip() for b in body if b.strip() and not b.strip().startswith("|")]
                rows = [b.strip() for b in body if b.strip().startswith("|")]
                table = []
                for r in rows:
                    cells = [c.strip() for c in r.strip().strip("|").split("|")]
                    if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                        continue
                    table.append(cells)
                blocks.append(Block("tab", label=head, cap=" ".join(cap_lines), rows=table, **opts))
            elif kind == "algo":
                nonblank = [b for b in body]
                cap = nonblank[0].strip() if nonblank else ""
                blocks.append(Block("algo", label=head, cap=cap, lines=nonblank[1:]))
            elif kind == "code":
                blocks.append(Block("code", lines=body))
            elif kind == "note":
                paras, cur = [], []
                for b in body:
                    if b.strip():
                        cur.append(b.strip())
                    elif cur:
                        paras.append(" ".join(cur)); cur = []
                if cur:
                    paras.append(" ".join(cur))
                blocks.append(Block("note", paras=paras, title=opts.get("title")))
            continue
        if re.match(r"- ", s):
            flush()
            items = [s[2:].strip()]
            while i < len(lines) and lines[i].lstrip().startswith("- "):
                items.append(lines[i].strip()[2:].strip()); i += 1
            blocks.append(Block("bullets", items=items)); continue
        if re.match(r"\d+\. ", s):
            flush()
            items = [re.sub(r"^\d+\.\s+", "", s)]
            while i < len(lines) and re.match(r"\s*\d+\. ", lines[i]):
                items.append(re.sub(r"^\s*\d+\.\s+", "", lines[i]).strip()); i += 1
            blocks.append(Block("numbered", items=items)); continue
        buf.append(s)
    flush()
    return blocks


# --------------------------------------------------------------------------
# the builder
# --------------------------------------------------------------------------
INLINE = re.compile(
    r"(?P<math>(?<!\\)\$(?P<mt>[^$]+?)\$)"
    r"|(?P<code>`(?P<ct>[^`]+)`)"
    r"|(?P<bold>\*\*(?P<bt>.+?)\*\*)"
    r"|(?P<ital>(?<![\w*])\*(?P<it>[^\s*][^*]*?)\*(?![\w*]))"
    r"|(?P<cite>\[@(?P<ck>[^\]]+)\])"
    r"|(?P<ref>@(?P<rk>fig|tab|eq|sec|chap|app|alg):(?P<rl>[\w\-]+))"
)


class Builder:
    def __init__(self, template: str, refs: dict, pagemap: dict | None = None,
                 total_pages: int | None = None):
        self.doc = docx.Document(template)
        self.body = self.doc.element.body
        self.refs = refs  # key -> formatted reference string
        self.pagemap = pagemap or {}
        self.total_pages = total_pages
        self.blocks: list[Block] = []
        self.labels: dict[str, str] = {}
        self.cite_order: list[str] = []
        self.math: list[tuple[str, bool]] = []
        self.math_map: dict[str, int] = {}
        self.toc_entries: list[tuple[int, str, str]] = []  # level, number, title
        self.fig_entries: list[tuple[str, str]] = []
        self.tab_entries: list[tuple[str, str]] = []
        self.sectPr = self.body.find(qn("w:sectPr"))
        self.chapter_no = 0
        self.in_appendix = False
        self.sec_no = [0, 0, 0]

    # ---------------------------------------------------------- numbering --
    def number_blocks(self):
        ch = 0          # chapter counter
        app = 0         # appendix counter
        sec = [0, 0]
        counters = {"fig": 0, "tab": 0, "eq": 0, "algo": 0}
        cur = ""
        for b in self.blocks:
            if b.kind in ("chapter", "appendix"):
                if b.kind == "appendix":
                    app += 1
                    b.num = chr(ord("A") + app - 1)
                else:
                    ch += 1
                    b.num = str(ch)
                cur = b.num
                sec = [0, 0]
                counters = {k: 0 for k in counters}
                if b.label:
                    word = "Appendix " if b.kind == "appendix" else "Chapter "
                    self.labels["chap:" + b.label] = b.num
                    self.labels["app:" + b.label] = b.num
            elif b.kind == "section":
                sec[0] += 1; sec[1] = 0
                b.num = f"{cur}.{sec[0]}"
                if b.label:
                    self.labels["sec:" + b.label] = b.num
            elif b.kind == "subsection":
                sec[1] += 1
                b.num = f"{cur}.{sec[0]}.{sec[1]}"
                if b.label:
                    self.labels["sec:" + b.label] = b.num
            elif b.kind in counters:
                counters[b.kind] += 1
                b.num = f"{cur}.{counters[b.kind]}"
                if b.label:
                    key = {"eq": "eq:", "fig": "fig:", "tab": "tab:", "algo": "alg:"}[b.kind] + b.label
                    word = {"eq": "(%s)", "fig": "Figure %s", "tab": "Table %s", "algo": "Algorithm %s"}[b.kind]
                    self.labels[key] = word % b.num

    # ----------------------------------------------------------- math ------
    def reg_math(self, tex, display):
        key = ("D" if display else "I") + tex.strip()
        if key not in self.math_map:
            self.math_map[key] = len(self.math)
            self.math.append((tex.strip(), display))
        return self.math_map[key]

    def collect_math(self):
        def scan(s):
            for m in INLINE.finditer(s):
                if m.group("math"):
                    self.reg_math(m.group("mt"), False)

        for b in self.blocks:
            if b.kind == "eq":
                self.reg_math(b.tex, True)
            elif b.kind in ("para", "chapter", "section", "subsection", "subsubsection", "appendix"):
                scan(getattr(b, "text", getattr(b, "title", "")))
            elif b.kind in ("bullets", "numbered"):
                for it in b.items:
                    scan(it)
            elif b.kind == "note":
                for p in b.paras:
                    scan(p)
            elif b.kind in ("fig", "algo"):
                scan(b.cap)
            elif b.kind == "tab":
                scan(b.cap)
                for r in b.rows:
                    for c in r:
                        scan(c)

    def convert_math(self):
        self.omml = latex_to_omml(self.math)

    def math_el(self, tex, display):
        return copy.deepcopy(self.omml[self.math_map[("D" if display else "I") + tex.strip()]])

    # ---------------------------------------------------------- citations --
    def cite_num(self, key):
        key = key.strip().lstrip("@")
        if key not in self.refs:
            raise KeyError(f"unknown reference key: {key}")
        if key not in self.cite_order:
            self.cite_order.append(key)
        return self.cite_order.index(key) + 1

    def fmt_cites(self, keys):
        nums = sorted({self.cite_num(k) for k in keys.split(";")})
        groups, cur = [], [nums[0]]
        for n in nums[1:]:
            if n == cur[-1] + 1:
                cur.append(n)
            else:
                groups.append(cur); cur = [n]
        groups.append(cur)
        parts = []
        for g in groups:
            if len(g) >= 3:
                parts.append(f"[{g[0]}]–[{g[-1]}]")
            else:
                parts.extend(f"[{n}]" for n in g)
        return ", ".join(parts)

    # ------------------------------------------------------------ inline --
    def inline(self, text, **base):
        """-> list of lxml children (runs / oMath)."""
        out = []
        pos = 0

        def add_text(t, **kw):
            if t and not kw.get("mono"):
                t = t.replace("--", "–")
            if t:
                out.append(run(t, **{**base, **kw}))

        for m in INLINE.finditer(text):
            add_text(text[pos:m.start()])
            pos = m.end()
            if m.group("math"):
                out.append(self.math_el(m.group("mt"), False))
            elif m.group("code"):
                add_text(m.group("ct"), mono=True, size=(base.get("size") or 12) - 1.5)
            elif m.group("bold"):
                add_text(m.group("bt"), bold=True)
            elif m.group("ital"):
                add_text(m.group("it"), italic=True)
            elif m.group("cite"):
                add_text(self.fmt_cites(m.group("ck")))
            elif m.group("ref"):
                key = m.group("rk") + ":" + m.group("rl")
                if key not in self.labels:
                    raise KeyError(f"unknown label {key}")
                add_text(self.labels[key])
        add_text(text[pos:])
        return out

    # --------------------------------------------------------- emitters ---
    def add(self, element):
        self.sectPr.addprevious(element)

    def p_body(self, text, **kw):
        pp = ppr("BodyText", jc="both", after=120, line=360, **kw)
        self.add(para(pp, self.inline(text)))

    def p_plain(self, text, jc="both", size=None, after=120, line=360, **kw):
        pp = ppr("BodyText", jc=jc, after=after, line=line, **kw)
        base = {"size": size} if size else {}
        self.add(para(pp, self.inline(text, **base)))

    def heading_chapter(self, b):
        if b.kind == "appendix":
            label = f"APPENDIX {b.num}"
            text = f"{label}. {b.title.replace('--', '–').upper()}"
        else:
            text = f"{b.num}. {b.title.replace('--', '–').upper()}"
        prev = self.sectPr.getprevious()
        if (prev is not None and prev.tag.endswith('}p') and not list(prev.iter(el('w:r').tag))
                and not list(prev.iter(el('w:sectPr').tag))
                and any(x.get(el('w:line').tag if False else '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}line')=='120' for x in prev.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}spacing'))):
            prev.getparent().remove(prev)   # trailing spacer would spill onto a blank page
        pp = ppr("Heading1", jc="center", before=0, after=360, line=360, page_break=True,
                 keep_next=True, outline=0, left=0)
        self.add(para(pp, [run(text, bold=True, size=14)]))
        self.toc_entries.append((1, "", text))

    def heading_section(self, b):
        text = f"{b.num}.\t{b.title.replace('--', '–').upper()}"
        pp = ppr("Heading2", jc="left", before=280, after=120, line=360, left=720, hanging=720,
                 keep_next=True, outline=1, tabs=[(720, "left", None)])
        self.add(para(pp, [run(text, bold=True, size=12)]))
        self.toc_entries.append((2, "", f"{b.num}. {b.title.replace('--', '–').upper()}"))

    def heading_subsection(self, b):
        text = f"{b.num}.\t{b.title.replace('--', '–')}"
        pp = ppr("Heading3", jc="left", before=200, after=100, line=360, left=900, hanging=900,
                 keep_next=True, outline=2, tabs=[(900, "left", None)])
        self.add(para(pp, [run(text, bold=True, size=12)]))
        self.toc_entries.append((3, "", f"{b.num}. {b.title.replace('--', '–')}"))

    def heading_subsub(self, b):
        pp = ppr("BodyText", jc="left", before=120, after=60, line=360, keep_next=True)
        self.add(para(pp, self.inline(b.title, bold=True, italic=True)))

    def eq_block(self, b):
        tbl = el("w:tbl")
        tp = el("w:tblPr")
        tp.append(el("w:tblW", w_w=TEXT_W, w_type="dxa"))
        tp.append(el("w:jc", w_val="center"))
        tp.append(el("w:tblLayout", w_type="fixed"))
        cm = el("w:tblCellMar")
        for side in ("top", "bottom"):
            cm.append(el("w:" + side, w_w=40, w_type="dxa"))
        tp.append(cm)
        tbl.append(tp)
        g = el("w:tblGrid")
        widths = [800, 7760, 800]
        for w in widths:
            g.append(el("w:gridCol", w_w=w))
        tbl.append(g)
        tr = el("w:tr")
        trp = el("w:trPr"); trp.append(el("w:cantSplit")); tr.append(trp)

        def cell(w, children, jc):
            tc = el("w:tc")
            tcp = el("w:tcPr")
            tcp.append(el("w:tcW", w_w=w, w_type="dxa"))
            tcp.append(el("w:vAlign", w_val="center"))
            tc.append(tcp)
            tc.append(para(ppr("BodyText", jc=jc, after=0, line=240, left=0, first=0), children))
            return tc

        tr.append(cell(widths[0], [], "left"))
        om = self.math_el(b.tex, True)
        omp = el("m:oMathPara")
        omp.append(om)
        tr.append(cell(widths[1], [omp], "center"))
        tr.append(cell(widths[2], [run(f"({b.num})", size=12)], "right"))
        tbl.append(tr)
        self.add(para(ppr("BodyText", after=0, line=120), []))
        self.add(tbl)
        self.add(para(ppr("BodyText", after=60, line=120), []))

    def fig_block(self, b):
        path = b.file
        if not os.path.isabs(path):
            path = str(Path(__file__).resolve().parent / "assets" / path)
        width = float(getattr(b, "width", 5.8))
        pic_p = self.doc.add_paragraph()
        r = pic_p.add_run()
        r.add_picture(path, width=Inches(width))
        pe = pic_p._p
        self.body.remove(pe)
        pPr = pe.find(qn("w:pPr"))
        if pPr is not None:
            pe.remove(pPr)
        pe.insert(0, ppr("BodyText", jc="center", before=120, after=60, line=240, keep_next=True))
        self.add(pe)
        cap = self.inline(b.cap, size=11)
        label = run(f"Figure {b.num}. ", bold=True, size=11)
        self.add(para(ppr("FigureCaption", jc="center", after=200, line=252, left=200, right=200),
                      [label] + cap))
        self.fig_entries.append((f"Figure {b.num}.", self.plain(b.cap)))

    def tab_block(self, b):
        cap = self.inline(b.cap, size=11)
        label = run(f"Table {b.num}. ", bold=True, size=11)
        self.add(para(ppr("TableCaption", jc="center", before=160, after=80, line=252, left=200, right=200,
                          keep_next=True), [label] + cap))
        self.tab_entries.append((f"Table {b.num}.", self.plain(b.cap)))
        rows = b.rows
        ncol = max(len(r) for r in rows)
        widths_in = [float(x) for x in b.widths.split(",")] if hasattr(b, "widths") else [6.5 / ncol] * ncol
        scale = 6.5 / sum(widths_in) if sum(widths_in) > 6.5 else 1.0
        widths = [int(w * scale * 1440) for w in widths_in]
        align = getattr(b, "align", "l" + "c" * (ncol - 1))
        fsz = float(getattr(b, "size", 9.5))
        tbl = el("w:tbl")
        tp = el("w:tblPr")
        tp.append(el("w:tblStyle", w_val="TableGrid"))
        tp.append(el("w:tblW", w_w=sum(widths), w_type="dxa"))
        tp.append(el("w:jc", w_val="center"))
        tp.append(el("w:tblLayout", w_type="fixed"))
        cm = el("w:tblCellMar")
        for side, w in (("top", 30), ("left", 70), ("bottom", 30), ("right", 70)):
            cm.append(el("w:" + side, w_w=w, w_type="dxa"))
        tp.append(cm)
        tbl.append(tp)
        g = el("w:tblGrid")
        for w in widths:
            g.append(el("w:gridCol", w_w=w))
        tbl.append(g)
        for ri, r in enumerate(rows):
            tr = el("w:tr")
            trp = el("w:trPr"); trp.append(el("w:cantSplit"))
            if ri == 0:
                trp.append(el("w:tblHeader"))
            tr.append(trp)
            for ci in range(ncol):
                txt = r[ci] if ci < len(r) else ""
                tc = el("w:tc")
                tcp = el("w:tcPr")
                tcp.append(el("w:tcW", w_w=widths[ci], w_type="dxa"))
                tcp.append(el("w:vAlign", w_val="center"))
                if ri == 0:
                    tcp.append(el("w:shd", w_val="clear", w_color="auto", w_fill="E7E6E6"))
                tc.append(tcp)
                a = align[ci] if ci < len(align) else "c"
                jc = {"l": "left", "c": "center", "r": "right"}[a]
                kw = {}
                if ri == 0:
                    kw["bold"] = True
                if txt.startswith("~"):
                    txt = txt[1:]
                    kw.update(italic=True, color="7F7F7F")
                pp = ppr("BodyText", jc=jc, after=0, line=240, left=0, first=0,
                         keep_next=(ri < len(rows) - 1 and len(rows) <= 14))
                tc.append(para(pp, self.inline(txt, size=fsz, **kw)))
                tr.append(tc)
            tbl.append(tr)
        self.add(tbl)
        self.add(para(ppr("BodyText", after=160, line=120), []))

    def box(self, paras_runs, fill="F2F2F2", border=None, mono=False, size=10, title=None):
        """single-cell shaded table; paras_runs = list of list-of-children."""
        tbl = el("w:tbl")
        tp = el("w:tblPr")
        tp.append(el("w:tblW", w_w=TEXT_W, w_type="dxa"))
        tp.append(el("w:jc", w_val="center"))
        bd = el("w:tblBorders")
        for side in ("top", "left", "bottom", "right"):
            if side == "left" and border:
                bd.append(el("w:left", w_val="single", w_sz=24, w_space=0, w_color=border))
            elif side == "left":
                bd.append(el("w:left", w_val="single", w_sz=4, w_space=0, w_color="BFBFBF"))
            else:
                bd.append(el("w:" + side, w_val="single" if not border else "nil", w_sz=4, w_space=0, w_color="BFBFBF"))
        tp.append(bd)
        tp.append(el("w:tblLayout", w_type="fixed"))
        cm = el("w:tblCellMar")
        for side, w in (("top", 80), ("left", 140), ("bottom", 80), ("right", 140)):
            cm.append(el("w:" + side, w_w=w, w_type="dxa"))
        tp.append(cm)
        tbl.append(tp)
        g = el("w:tblGrid"); g.append(el("w:gridCol", w_w=TEXT_W)); tbl.append(g)
        tr = el("w:tr")
        trp = el("w:trPr"); trp.append(el("w:cantSplit")); tr.append(trp)
        tc = el("w:tc")
        tcp = el("w:tcPr"); tcp.append(el("w:tcW", w_w=TEXT_W, w_type="dxa"))
        tcp.append(el("w:shd", w_val="clear", w_color="auto", w_fill=fill)); tc.append(tcp)
        for children, jc in paras_runs:
            tc.append(para(ppr("BodyText", jc=jc, after=40, line=252 if not mono else 228, left=0, first=0), children))
        tr.append(tc)
        tbl.append(tr)
        self.add(tbl)
        self.add(para(ppr("BodyText", after=160, line=120), []))

    def algo_block(self, b):
        self.add(para(ppr("TableCaption", jc="left", before=160, after=80, line=252, keep_next=True),
                      [run(f"Algorithm {b.num}. ", bold=True, size=11)] + self.inline(b.cap, size=11)))
        rows = []
        for ln in b.lines:
            if not ln.strip():
                continue
            rows.append(([run(ln.rstrip(), mono=True, size=9)], "left"))
        self.box(rows, fill="F7F7F7", mono=True)

    def code_block(self, b):
        rows = [([run(ln.rstrip(), mono=True, size=7.5)], "left") for ln in b.lines if ln.strip() != ""]
        self.box(rows, fill="F7F7F7", mono=True)

    def note_block(self, b):
        rows = []
        if b.title:
            rows.append((self.inline(b.title, bold=True, size=11), "left"))
        for p in b.paras:
            rows.append((self.inline(p, size=11), "both"))
        self.box(rows, fill="EEF3FA", border="2A78D6")

    def list_block(self, b, numbered):
        for k, it in enumerate(b.items, 1):
            lead = f"{k}.\t" if numbered else "•\t"
            pp = ppr("BodyText", jc="both", after=60, line=360, left=720, hanging=360,
                     tabs=[(720, "left", None)])
            self.add(para(pp, [run(lead)] + self.inline(it)))

    # -------------------------------------------------------- front/aux ---
    def emit_blocks(self, blocks):
        for b in blocks:
            k = b.kind
            if k in ("chapter", "appendix"):
                self.heading_chapter(b)
            elif k == "section":
                self.heading_section(b)
            elif k == "subsection":
                self.heading_subsection(b)
            elif k == "subsubsection":
                self.heading_subsub(b)
            elif k == "para":
                self.p_body(b.text)
            elif k == "eq":
                self.eq_block(b)
            elif k == "fig":
                self.fig_block(b)
            elif k == "tab":
                self.tab_block(b)
            elif k == "algo":
                self.algo_block(b)
            elif k == "code":
                self.code_block(b)
            elif k == "note":
                self.note_block(b)
            elif k == "bullets":
                self.list_block(b, False)
            elif k == "numbered":
                self.list_block(b, True)
            elif k == "pagebreak":
                r = el("w:r"); r.append(el("w:br", w_type="page"))
                self.add(para(ppr("BodyText", after=0), [r]))

    def front_heading(self, text, page_break=True, toc=False):
        pp = ppr("Heading1", jc="center", before=0, after=360, line=360, page_break=page_break,
                 keep_next=True, outline=0 if toc else None, left=0)
        self.add(para(pp, [run(text, bold=True, size=14)]))
        if toc:
            self.toc_entries.append((1, "", text))

    def toc_block(self, entries, instr, style_for, title, page_for, tab_pos=TEXT_W):
        """Field-based TOC with cached entries."""
        self.front_heading(title, toc=False)
        first = True
        n = len(entries)
        for idx, (lvl, text) in enumerate(entries):
            pg = page_for(text)
            kids = []
            if first:
                for r in fld_runs(instr, "")[:3]:
                    kids.append(r)
                first = False
            kids.append(run(text))
            kids.append(run("\t"))
            kids.append(run(str(pg) if pg else ""))
            if idx == n - 1:
                r = el("w:r"); r.append(el("w:fldChar", w_fldCharType="end")); kids.append(r)
            self.add(para(style_for(lvl), kids))

    def build_toc(self):
        ents = [(l, t) for (l, _, t) in self.toc_entries if l <= 3]

        def style_for(lvl):
            left = {1: 440, 2: 880, 3: 1540}[lvl]
            hang = {1: 440, 2: 440, 3: 660}[lvl]
            return ppr(f"TOC{lvl}", jc="left", before=100 if lvl == 1 else 0, after=20, line=276,
                       left=left, hanging=hang, right=600,
                       tabs=[(TEXT_W, "right", "dot")])

        self.toc_block(ents, 'TOC \\o "1-3" \\h \\z \\u', style_for, "TABLE OF CONTENTS",
                       lambda t: self.pagemap.get(norm(t)))

    def build_lists(self):
        def sf(lvl):
            return ppr("TableofFigures", jc="left", after=60, line=276, left=1100, hanging=1100, right=600,
                       tabs=[(TEXT_W, "right", "dot")])

        ents = [(1, f"{lab} {cap}") for lab, cap in self.fig_entries]
        self.toc_block(ents, 'TOC \\h \\z \\t "Figure Caption,1"', sf, "LIST OF FIGURES",
                       lambda t: self.pagemap.get(norm(t.split(" ")[0])))
        ents = [(1, f"{lab} {cap}") for lab, cap in self.tab_entries]
        self.toc_block(ents, 'TOC \\h \\z \\t "Table Caption,1"', sf, "LIST OF TABLES",
                       lambda t: self.pagemap.get(norm(t.split(" ")[0])))

    def references(self):
        self.front_heading("REFERENCES", toc=True)
        for n, key in enumerate(self.cite_order, 1):
            pp = ppr("BodyText", jc="left", after=80, line=264, left=567, hanging=567,
                     tabs=[(567, "left", None)])
            self.add(para(pp, [run(f"[{n}]\t", size=11)] + self.inline_ref(self.refs[key])))

    _MACROS = {r"\\alpha": "α", r"\\beta": "β", r"\\gamma": "γ", r"\\lambda": "λ", r"\\Delta": "Δ", r"\\delta": "δ", r"\\sigma": "σ", r"\\rho": "ρ",
               r"\\times": "×", r"\\ell": "ℓ", r"\\leq": "≤", r"\\le": "≤", r"\\geq": "≥", r"\\ge": "≥", r"\\pm": "±", r"\\ ": " ", r"\\,": " ", r"\\;": " ",
               r"\\to": "→", r"\\cdot": "·", r"\\in": "∈", r"\\subset": "⊂", r"\\approx": "≈", r"\\mathcal": "", r"\\mathrm": "", r"\\mathbf": "", r"\\text": ""}
    _SUP = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "-": "⁻"}

    def plain(self, text):
        """caption text for the lists of figures / tables: refs resolved, markup and LaTeX removed"""
        def ref(m):
            return self.labels[m.group("rk") + ":" + m.group("rl")]
        t = re.sub(r"\[@(?P<ck>[^\]]+)\]", lambda m: self.fmt_cites(m.group("ck").replace("@", "")), text)
        t = re.sub(r"@(?P<rk>fig|tab|eq|sec|chap|app|alg):(?P<rl>[\w\-]+)", ref, t)
        def mat(m):
            x = m.group(1)
            for k, v in self._MACROS.items():
                x = re.sub(k, v, x)
            x = re.sub(r"\\(?:bar|hat|tilde)\{?([A-Za-z])\}?", r"\1", x)
            x = re.sub(r"\^\{?([0-9-]+)\}?", lambda mm: "".join(self._SUP.get(c, c) for c in mm.group(1)), x)
            x = re.sub(r"_\{([^}]*)\}", r"_\1", x)
            x = x.replace("{", "").replace("}", "").replace("\\", "")
            return x
        t = re.sub(r"(?<!\\)\$([^$]+?)\$", mat, t)
        t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
        t = re.sub(r"(?<![\w*])\*([^\s*][^*]*?)\*(?![\w*])", r"\1", t)
        t = t.replace("`", "")
        return re.sub(r"\s+", " ", t).strip()

    def inline_ref(self, s):
        # references use *italic* for venue; keep it simple: italic via *...*
        out = []
        pos = 0
        for m in re.finditer(r"\*(.+?)\*", s):
            if m.start() > pos:
                out.append(run(s[pos:m.start()], size=11))
            out.append(run(m.group(1), italic=True, size=11))
            pos = m.end()
        if pos < len(s):
            out.append(run(s[pos:], size=11))
        return out


def norm(s):
    return re.sub(r"\s+", " ", s.replace(" ", " ").replace("\t", " ")).strip()
