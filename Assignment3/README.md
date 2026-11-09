# Assignment 3: MLP posterior estimation and GMM order selection

The report is **EECE5644_Assignment3.pdf**. It contains the process, math, results, figures, software checks, the AI-use declaration and a code appendix. **EECE5644_Assignment3.docx** is an editable Word version with the same content; its AI-use statement is a highlighted placeholder to edit. The questions are in `Assignment3_Questions.pdf`.

| Question | Topic | Code |
|---|---|---|
| Q1 | 4-class 3-D Gaussian data; true-pdf MAP classifier; softplus MLPs with hidden size chosen by 10-fold CV (min P(error)), trained by maximum likelihood with random restarts; tested on 100,000 samples | `mlp.py`, `q1.py`, `q1_diagnostics.py`, `q1_cv_sensitivity.py` |
| Q2 | 4-component 2-D GMM with two overlapping components; GMM order 1–10 selected by 10-fold CV log-likelihood on 10/100/1000 samples, repeated 100 times | `q2.py` |

## Run the experiments

```sh
python -m pip install -r requirements.txt
python q1.py                  # ~20 min on 2 cores; writes q1_results.json and figures
python q1_diagnostics.py      # convergence diagnostics, confidence intervals, decision-region slice
python q1_cv_sensitivity.py   # CV re-run with 5x iteration budget for N=500 and N=1000
python q2.py                  # ~10 min on 2 cores; writes q2_results.json, q2_curves.npz and figures
python q2.py --plots          # redraw Q2 figures from saved results only
```

Run from this folder. Every script uses fixed seeds. The diagnostics and sensitivity scripts re-create the exact datasets of `q1.py` and assert that its results reproduce. The console output of the runs used in the report is saved in `q1_log.txt`, `q1_diag_log.txt` and `q2_log.txt`.

The MLP is implemented directly in NumPy (`mlp.py`): softplus hidden layer, softmax output, average cross-entropy minimized by SciPy's L-BFGS with an analytic gradient. The gradient is checked against finite differences in `q1.py`. GMMs are fitted with scikit-learn's `GaussianMixture` (EM).

## Build the PDF

```sh
python build_report.py                       # fills report_template.tex with numbers from the JSON results
xelatex report.tex && xelatex report.tex     # rename the output to EECE5644_Assignment3.pdf
```

The report uses `fontspec`, so it needs XeLaTeX rather than pdfLaTeX.

## Key results

| | Result |
|---|---|
| Q1 true-pdf MAP classifier | test P(error) = 0.1484 |
| Q1 MLP, N = 100 / 500 / 1000 / 5000 / 10000 | test P(error) = 0.2508 / 0.1807 / 0.1523 / 0.1501 / 0.1496 |
| Q1 CV-selected hidden units | 3 / 6 / 3 / 3 / 12 |
| Q2 most-selected order, N = 10 / 100 / 1000 | K = 1 (93%) / K = 3 (92%) / K = 4 (83%) |

AI assistance is disclosed in the report.
