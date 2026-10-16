# Assignment 1: reviewed implementation

The original uploaded PDF and ZIP have not been modified. This folder contains a corrected, rerun working draft based on that implementation. AI assistance is described in the report.

## Run the experiments

With Python 3.10+:

```
python -m pip install -r requirements.txt
python run_all.py
```

The runner downloads the original UCI datasets if absent, then runs all questions and verification. It can be invoked from any directory. Q1 includes an additional independent two-million-observation Monte Carlo estimate. HAR requires substantial memory; allow approximately 1-2 GB of available RAM. Results and figures are regenerated deterministically from the specified seeds, subject to numerical library differences.

To regenerate the PDF separately:

```
python build_report.py
```

PDF fonts use Windows Arial when available; the experiments do not require Windows. The supplied original LaTeX file and original code are retained under `original_source/` for comparison and are not the revised report source. `build_report.py` is the revised report source. `gen_tables.py` is an optional legacy table exporter, not required by the runner.

## Files

- `EECE5644_Assignment1_Reviewed.pdf`: complete revised report, plots, and source appendix.
- `Study_Guide.md`: explanations of what each experiment does and why.
- `Review_Notes.md`: corrections and limits of the verification.
- `q1.py`, `q2.py`, `q3.py`, `common.py`: experiment code.
- `download_data.py`: original UCI archives, SHA-256 records, and required-data extraction.
- `run_all.py`: one-command experiment runner.
- `verify_results.py`: independent checks of numerical calculations and saved results.
- `out/`: JSON summaries, generated datasets, full ROC traces, and verification record.
- `figs/`: every report plot, plus equation images used in PDF generation.
- `data/sources.json`: exact archive URLs, sizes, and hashes.
- `logs/`: execution output.

Training results are reported on all data as the instructor requests. Wine CV and the HAR official-split validation are separate tuning diagnostics. The latter was used to choose alpha and is not an untouched final test.
