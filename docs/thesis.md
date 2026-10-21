# Thesis builder

`thesis/` contains the generator of the thesis document: the chapter sources (`thesis/content/*.txt`, a small directive language for
chapters, sections, figures, tables, algorithms and equations), the generators of the results chapter (`ch4_*.py`, `ch5_hard.py`,
`appendices.py`, `appx_hard.py`), and the document assembler (`build_thesis.py`, `assemble.py`, `tbuild.py`).

Every number in the results chapter is computed from the output files of the experiments when the document is built; tables for
experiments that have not been run show *pending*.

## Requirements

`pip install -e ".[thesis]"`, `pandoc` (LaTeX to Word equations), LibreOffice (field/page-number refresh and PDF rendering) and
`poppler-utils`. A reference `.docx` that provides the page format is required and is **not** included in the repository:

```bash
python thesis/build_thesis.py thesis/out/thesis.docx --template /path/to/reference_thesis.docx
```

Useful options: `--phantom DIR` and `--hard DIR` (directories of the synthetic studies, default `runs/phantom_study` and
`runs/hard_study`; if they are missing, the frozen copies in `thesis/assets/` are used), `--brats DIR` (BraTS results), `--dryrun`.

## Quality checks

```bash
python thesis/tools/ooxml_order.py thesis/out/thesis.docx    # schema-order check of the generated OOXML
python thesis/tools/cites.py 5                               # citation usage per reference (run from thesis/)
```

After opening the document in Word press Ctrl+A and F9 to refresh the table of contents and the lists of figures and tables.
