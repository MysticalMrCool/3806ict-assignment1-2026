# Report sources

LaTeX sources for the LNCS report. The compiled PDF
(`Holland_Ben_Assignment1.pdf`) lives in the assignment root.

## Files

- `main.tex` — report body (abstract, sections, figures, tables)
- `appendix_ai.tex` — Generative-AI declaration appendix (`\input` from `main.tex`)
- `references.bib` — bibliography database
- `llncs.cls`, `splncs04.bst` — Springer LNCS class and BibTeX style
  (unmodified, redistributed for self-contained builds)

## Building the PDF

Requires a LaTeX distribution with `pdflatex` and `bibtex`.

```sh
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or open `main.tex` on Overleaf — the LNCS class is included so no template
project is needed.

## Where the numbers come from

The values in Table 1 and Table 2 are taken from
`../results/bench_summary.txt`. Regenerate with `python -m src.bench` from
the assignment root.
