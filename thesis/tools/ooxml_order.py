import sys, zipfile
from lxml import etree
W="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ORD={
"pPr":"pStyle keepNext keepLines pageBreakBefore framePr widowControl numPr suppressLineNumbers pBdr shd tabs suppressAutoHyphens kinsoku wordWrap overflowPunct topLinePunct autoSpaceDE autoSpaceDN bidi adjustRightInd snapToGrid spacing ind contextualSpacing mirrorIndents suppressOverlap jc textDirection textAlignment textboxTightWrap outlineLvl divId cnfStyle rPr sectPr pPrChange".split(),
"rPr":"rStyle rFonts b bCs i iCs caps smallCaps strike dstrike outline shadow emboss imprint noProof snapToGrid vanish webHidden color spacing w kern position sz szCs highlight u effect bdr shd fitText vertAlign rtl cs em lang eastAsianLayout specVanish oMath".split(),
"tblPr":"tblStyle tblpPr tblOverlap bidiVisual tblStyleRowBandSize tblStyleColBandSize tblW jc tblCellSpacing tblInd tblBorders shd tblLayout tblCellMar tblLook".split(),
"tcPr":"cnfStyle tcW gridSpan hMerge vMerge tcBorders shd noWrap tcMar textDirection tcFitText vAlign hideMark".split(),
}
z=zipfile.ZipFile(sys.argv[1]); bad=0
for part in ("word/document.xml","word/styles.xml"):
    t=etree.fromstring(z.read(part))
    for name,order in ORD.items():
        for el in t.iter("{%s}%s"%(W,name)):
            idx=[]
            for c in el:
                ln=etree.QName(c).localname
                if ln in order: idx.append(order.index(ln))
                else: idx.append(-1)
            known=[i for i in idx if i>=0]
            if known!=sorted(known):
                bad+=1
                if bad<8: print(part,name,[etree.QName(c).localname for c in el])
print("order violations:",bad)
