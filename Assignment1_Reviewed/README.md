# Assignment 1: reviewed report and reproducible experiments

The report is **EECE5644_Assignment1_Reviewed.pdf**. It uses native LaTeX equations and includes the mathematical analysis, results, plots, AI declaration, data provenance, and experiment source appendix. Study_Guide.md is a separate learning companion.

## Run the experiments

```sh
python -m pip install -r requirements.txt
python run_all.py
```

The runner downloads original UCI datasets when absent and runs all questions plus verification. Results appear in `out/`; plots appear in `figs/`. Run from any directory. NumPy 2+ is needed for `numpy.trapezoid`.

## Build the PDF

Install Tectonic or a LaTeX distribution providing pdfLaTeX, then run:

```sh
python build_latex.py
```

The builder regenerates LaTeX tables from saved JSON, compiles `report.tex`, and writes `EECE5644_Assignment1_Reviewed.pdf`. Tectonic may download TeX packages on the first build. No ReportLab, machine-specific Python package path, or rasterized equation assets are needed. `build/` contains disposable compiler output.

## What changed after the presentation review

- Replaced the earlier ReportLab report with native LaTeX based on the merged report from `Assignment1/`.
- Removed draft labels and standardized equations, headings, captions, and page numbering.
- Retained the verified AUC and preprocessing corrections and explicit validation terminology.
- Tightened remaining claims about LDA, covariance rank, and Gaussian diagnostics.
- Preserved the study guide as a separate resource.

`build_report.py` delegates to `build_latex.py` for compatibility with the earlier entry point. The existing `original_source/` directory preserves provenance, not the current source. The root `report.tex` and `gen_tables.py` are current.

## Evidence and provenance

`out/verification.json` records the numerical checks. `data/sources.json` records original UCI archive URLs and SHA-256 hashes. Synthetic seeds and data are unchanged. The main results use all-sample training evaluation as the assignment requests; validation estimates are reported separately. AI assistance from Claude and Codex is disclosed in the report without claiming independent student verification.
