"""Shared helpers for EECE5644 Assignment 1."""
import os
import numpy as np

# output folders used by every script
os.makedirs("out", exist_ok=True)
os.makedirs("figs", exist_ok=True)


def gauss_logpdf(X, m, C):
    """Log of the multivariate Gaussian pdf g(x|m,C), evaluated row-wise on X (N x d).
    Uses a Cholesky factorisation for numerical stability."""
    X = np.atleast_2d(X)
    d = X.shape[1]
    L = np.linalg.cholesky(C)
    Z = np.linalg.solve(L, (X - m).T)          # d x N, Z = L^{-1}(x - m)
    maha = np.sum(Z ** 2, axis=0)              # (x-m)^T C^{-1} (x-m)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2 * np.pi) + logdet + maha)


def roc_sweep(score, labels):
    """Sweep a threshold t over all observed score values for the rule D = 1 iff score > t.
    Returns thresholds (starting at -inf, so the curve starts at (1,1)), P(D=1|L=0) and P(D=1|L=1).
    The largest observed score gives the finite end point (0,0)."""
    score = np.asarray(score, float)
    labels = np.asarray(labels).astype(int)
    n1, n0 = labels.sum(), (1 - labels).sum()
    order = np.argsort(score)
    s_sorted = score[order]
    l_sorted = labels[order]
    # number of samples with score <= s_sorted[k]  -> those are decided 0 at threshold s_sorted[k]
    c1_le = np.cumsum(l_sorted)
    c0_le = np.cumsum(1 - l_sorted)
    # keep only the last index of each run of tied scores
    last = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    thr = np.r_[-np.inf, s_sorted[last]]
    tpr = np.r_[1.0, (n1 - c1_le[last]) / n1]
    fpr = np.r_[1.0, (n0 - c0_le[last]) / n0]
    return thr, fpr, tpr


def auc(fpr, tpr):
    o = np.argsort(fpr, kind="stable")
    return np.trapezoid(tpr[o], fpr[o])
