# EECE5644 Fall 2026 — Assignment 1

Code, data and report for Assignment 1 (Bayesian decision theory, ROC analysis, naive Bayes, Fisher LDA, ERM with loss matrices, Gaussian classifiers on real data).

The graded write-up is **`EECE5644_Assignment1_DavidAbu.pdf`**.

## Contents

| Path | What it is |
|---|---|
| `q1.py` | Q1: ERM/LRT with the true pdf, naive Bayes, Fisher LDA, ROC curves |
| `q2.py` | Q2: 3-class Gaussian-mixture problem, MAP and ERM with Λ₁₀ / Λ₁₀₀ |
| `q3.py` | Q3: regularized Gaussian classifiers on Wine Quality and HAR |
| `common.py` | Shared helpers (Gaussian log-pdf, ROC sweep) |
| `gen_tables.py` | Writes the LaTeX tables in `out/` from the result files |
| `download_data.py` | Fetches the Q3 datasets from the official UCI archives if `data/` is missing them |
| `report.tex` | LaTeX source of the report |
| `generated_data/` | The exact 10K-sample data sets used in Q1 and Q2 (CSV and `.npz`) |
| `data/` | Q3 data: `winequality-white.csv` and the UCI HAR files in `data/har/` |
| `out/` | Numerical results (JSON) and generated tables |
| `figs/` | All figures in the report |

## Data

- **Q1** — `generated_data/q1_samples_10k.csv`: columns `x1..x4`, `label` (0/1). Seed 5644.
- **Q2** — `generated_data/q2_samples_10k.csv`: columns `x1..x3`, `label` (1/2/3), `component` (1–4). Seed 2026.
- **Q3** — [UCI Wine Quality](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) (white wine, 4898 × 11) and [UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) (7352 + 2947 = 10299 × 561). The copies in `data/` come from the public mirrors `github.com/zygmuntz/wine-quality` and `github.com/srvds/Human-Activity-Recognition`. An independent rerun on the official UCI archives (wine SHA-256 `3ed56667f4b8…0304d4`, HAR SHA-256 `c00b803081a5…1031`) gave identical Q3 results.

The scripts regenerate the Q1/Q2 data from their fixed seeds, so rerunning them reproduces these files exactly.

## Reproduce

Requires Python 3 with NumPy, SciPy, pandas and Matplotlib.

```bash
python download_data.py   # only downloads what's missing from data/
python q1.py
python q2.py
python q3.py        # about a minute (HAR has 561 features)
python gen_tables.py
pdflatex report.tex && pdflatex report.tex
```

## Acknowledgement

Prepared with the assistance of Claude (Anthropic) and reviewed by OpenAI Codex, as declared in the report. Codex's corrections (ROC AUC with tied scores, training-only scaling during tuning, validation terminology, data provenance) are merged into this version.
