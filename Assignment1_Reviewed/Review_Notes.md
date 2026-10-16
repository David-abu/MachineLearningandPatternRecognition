# Review notes

## What was verified

All three questions were run on this computer. Synthetic parameters match the assignment. Q1 and Q2 use deterministic seeds and reuse their datasets. Q3 uses original UCI archives; exact SHA-256 hashes are recorded in data/sources.json. 22 checks passed; see out/verification.json.

## Corrections

1. **Reproducibility:** the original q3.py referenced /home/claude/data and the ZIP omitted all datasets, generated figures, and numerical outputs. The revised implementation uses a project-relative path and provides the downloader and runner.
2. **ROC integration:** sorting FPR alone broke traversal order among equal FPR values. Integrating the original descending ROC directly corrects this; a tied-score example is checked against an independent rank-based AUC formula.
3. **Validation preprocessing:** the original code standardized using all samples before CV or held-out evaluation. Scaling now fits each training portion separately. The all-sample fit remains unchanged as required by the assignment.
4. **Validation terminology:** the HAR official test split selects alpha, so its score is reported as validation, not an independent final test.
5. **Numerical interpretation:** covariance regularization produces eigenvalues at least lambda, with equality possible; HAR rank deficiency does not establish a specific nonlinear causal relationship between features; training error is not necessarily monotonic in alpha; fitted/regularized Mahalanobis Q-Q plots are descriptive, not calibrated tests.
6. **Provenance:** the earlier model-name and mirror byte-identity claims are not independently verified. The new report acknowledges both Claude's original assistance and Codex's revision without claiming David has already verified it.
7. **Saved evidence:** all Q1 thresholds, FPR, TPR, FNR, and error arrays are saved in out/q1_roc_*.npz. JSON summaries and regenerated figures support all report numbers.

## Numerical outcome

The major reported errors reproduce: Q1 approximately 0.02593 / 0.03248 / 0.05210; Q2 risks 0.1749 / 0.3387 / 0.4882; wine training error 0.5032666; HAR training error 0.0030100. Selected alpha values remain 0.001 for wine and 0.0001 for HAR. ROC AUCs change slightly after the correction. Some validation scores change after training-only scaling.

## Limits

The supplied original report was read as text, not independently validated line by line. Reruns and numerical checks support the revised implementation; they do not certify the validity of the Gaussian modeling assumptions. No external final-test performance is claimed. All-sample scoring is intentional because the assignment asks for it. The completed report and study guide should be reviewed and understood before submission; no Canvas submission was made.
