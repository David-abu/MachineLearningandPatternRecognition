# Assignment 2: logistic classifiers, ML/MAP estimation, MAP localization

The report is **EECE5644_Assignment2.pdf**. It contains the derivations, results, plots, the AI-use declaration, and a code appendix. The questions are in `Assignment2_Questions.pdf`.

| Question | Topic | Code |
|---|---|---|
| Q1 | Min-P(error) classifier for a Gaussian-mixture problem, ROC, ML-trained logistic-linear and logistic-quadratic classifiers on 20/200/2000 samples | `q1.py` |
| Q2 | ML and MAP (Gaussian prior, γ sweep) estimators for cubic polynomial regression | `q2.py` |
| Q3 | MAP vehicle localization from K range measurements, objective contour plots | `q3.py` |
| Q4 | Duda–Hart–Stork Problem 2.13 (reject option) | derivation in the report |
| Q5 | ML and Dirichlet-MAP estimators for a categorical distribution | derivation in the report |

## Run the experiments

```sh
python -m pip install -r requirements.txt
python q1.py
python q2.py
python q3.py
```

Run from this folder. Each script uses a fixed seed and writes its plots to `figs/`; numerical results print to the console.

`course_provided/` holds the instructor's `hw2q2.py` and `hw2q2.m` data generators. `q2.py` reproduces the same generator logic (plotting removed, seed added) with 100 training and 1000 validation samples.

## Build the PDF

```sh
xelatex report.tex && xelatex report.tex
```

The report uses `fontspec` (DejaVu Sans Mono for the code appendix), so it needs XeLaTeX rather than pdfLaTeX. Rename the output to `EECE5644_Assignment2.pdf` if rebuilding.

## Key results

| | Validation result |
|---|---|
| Q1 optimal classifier | P(error) = 0.2793 |
| Q1 logistic-quadratic, N=2000 | P(error) = 0.2795 |
| Q1 logistic-linear, best | P(error) = 0.379 |
| Q2 ML | MSE = 5.455 |
| Q2 best MAP (γ ≈ 8e-6, selected on the validation set, so optimistic) | MSE = 4.427 |
| Q3 MAP error, K=1 → K=4 | 0.650 → 0.147 |

AI assistance from Claude is disclosed in the report.
