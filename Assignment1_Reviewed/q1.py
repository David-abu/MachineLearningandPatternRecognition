"""Question 1: ERM / LRT with true pdf, Naive Bayes, and Fisher LDA on a 2-class 4-D Gaussian problem."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from common import gauss_logpdf, roc_sweep, auc

rng = np.random.default_rng(5644)
N = 10000
P = np.array([0.35, 0.65])
m0 = -0.5 * np.ones(4)
C0 = 0.25 * np.array([[2, -0.5, 0.3, 0],
                      [-0.5, 1, -0.5, 0],
                      [0.3, -0.5, 1, 0],
                      [0, 0, 0, 2]])
m1 = np.ones(4)
C1 = np.array([[1, 0.3, -0.2, 0],
               [0.3, 2, 0.3, 0],
               [-0.2, 0.3, 1, 0],
               [0, 0, 0, 3]])
for C in (C0, C1):
    assert np.all(np.linalg.eigvalsh(C) > 0)


def sample(n, rng):
    L = (rng.random(n) >= P[0]).astype(int)          # P(L=1)=0.65
    X = np.empty((n, 4))
    X[L == 0] = rng.multivariate_normal(m0, C0, (L == 0).sum())
    X[L == 1] = rng.multivariate_normal(m1, C1, (L == 1).sum())
    return X, L


X, L = sample(N, rng)
np.savez("out/q1_data.npz", X=X, L=L)                 # same data set used for every part
N0, N1 = (L == 0).sum(), (L == 1).sum()
res = {"N0": int(N0), "N1": int(N1)}

gamma_theory = P[0] / P[1]                            # 0-1 loss: (l10-l00)P0 / ((l01-l11)P1)
res["gamma_theory"] = gamma_theory


def analyse(score, name, theory_thr=None, priors=P):
    """score: decision statistic, D=1 iff score > thr. Returns dict + ROC arrays."""
    thr, fpr, tpr = roc_sweep(score, L)
    perr = fpr * priors[0] + (1 - tpr) * priors[1]
    k = int(np.argmin(perr))
    # any threshold in [thr_k, thr_{k+1}) gives the same decisions; report the midpoint
    t_lo = thr[k]
    t_hi = thr[k + 1] if k + 1 < len(thr) else np.inf
    t_mid = 0.5 * (t_lo + t_hi) if np.isfinite(t_lo) and np.isfinite(t_hi) else t_lo
    out = dict(name=name, k=k, thr_mid=float(t_mid), thr_lo=float(t_lo), thr_hi=float(t_hi),
               perr_min=float(perr[k]), tpr=float(tpr[k]), fpr=float(fpr[k]),
               auc=float(auc(fpr, tpr)), n_thr=len(thr))
    if theory_thr is not None:
        D = score > theory_thr
        out["theory_tpr"] = float(D[L == 1].mean())
        out["theory_fpr"] = float(D[L == 0].mean())
        out["theory_perr"] = float(out["theory_fpr"] * P[0] + (1 - out["theory_tpr"]) * P[1])
    return out, (fpr, tpr, perr, thr)


# ---------- Part A: true pdf ----------
llr_A = gauss_logpdf(X, m1, C1) - gauss_logpdf(X, m0, C0)          # ln p(x|1) - ln p(x|0)
rA, curA = analyse(llr_A, "True pdf (ERM)", np.log(gamma_theory))
rA["gamma_emp"] = float(np.exp(rA["thr_mid"]))
rA["gamma_emp_lo"] = float(np.exp(rA["thr_lo"]))
rA["gamma_emp_hi"] = float(np.exp(rA["thr_hi"]))

# ---------- Part B: naive Bayes (diagonal covariances with true variances) ----------
C0d, C1d = np.diag(np.diag(C0)), np.diag(np.diag(C1))
llr_B = gauss_logpdf(X, m1, C1d) - gauss_logpdf(X, m0, C0d)
rB, curB = analyse(llr_B, "Naive Bayes", np.log(gamma_theory))
rB["gamma_emp"] = float(np.exp(rB["thr_mid"]))
rB["gamma_emp_lo"] = float(np.exp(rB["thr_lo"]))
rB["gamma_emp_hi"] = float(np.exp(rB["thr_hi"]))

# ---------- Part C: Fisher LDA from sample estimates ----------
mu0_hat, mu1_hat = X[L == 0].mean(0), X[L == 1].mean(0)
S0_hat = np.cov(X[L == 0].T, bias=True)          # sample-average (1/N_j) estimators
S1_hat = np.cov(X[L == 1].T, bias=True)
Sb = np.outer(mu1_hat - mu0_hat, mu1_hat - mu0_hat)
Sw = S0_hat + S1_hat                              # equal weights, priors ignored (as in class)
evals, evecs = eigh(Sb, Sw)                       # generalised eigenproblem Sb w = lambda Sw w
w = evecs[:, np.argmax(evals)]
w = w / np.linalg.norm(w)
if w @ mu1_hat < w @ mu0_hat:                     # orient so that class 1 projects higher
    w = -w
y = X @ w
rC, curC = analyse(y, "Fisher LDA", priors=np.array([N0, N1]) / N)  # sample-count estimate
rC["count_check"] = float(np.mean((y > rC["thr_mid"]) != L))
w_closed = np.linalg.solve(Sw, mu1_hat - mu0_hat)
w_closed /= np.linalg.norm(w_closed)
res["lda"] = dict(w=w.tolist(), w_closed=w_closed.tolist(), eig=float(evals.max()),
                  mu0_hat=mu0_hat.tolist(), mu1_hat=mu1_hat.tolist(),
                  S0_hat=S0_hat.tolist(), S1_hat=S1_hat.tolist())

# ---------- reference: theoretical min P(error) via a large independent Monte Carlo ----------
Xbig, Lbig = sample(2_000_000, np.random.default_rng(1))
llr_big = gauss_logpdf(Xbig, m1, C1) - gauss_logpdf(Xbig, m0, C0)
res["perr_bayes_mc"] = float(np.mean((llr_big > np.log(gamma_theory)) != Lbig))
llr_bigB = gauss_logpdf(Xbig, m1, C1d) - gauss_logpdf(Xbig, m0, C0d)
res["perr_nb_mc_theorythr"] = float(np.mean((llr_bigB > np.log(gamma_theory)) != Lbig))

res["A"], res["B"], res["C"] = rA, rB, rC
for key, curve in [("A", curA), ("B", curB), ("C", curC)]:
    fpr, tpr, perr, thresholds = curve
    np.savez(f"out/q1_roc_{key}.npz", fpr=fpr, tpr=tpr, fnr=1-tpr, perr=perr, thresholds=thresholds)
res["perr_bayes_mc_se"] = float(np.sqrt(res["perr_bayes_mc"]*(1-res["perr_bayes_mc"])/len(Lbig)))

json.dump(res, open("out/q1_results.json", "w"), indent=2)

# ---------- figures ----------
def roc_plot(ax, cur, r, title, color, theory=True):
    fpr, tpr, _, _ = cur
    ax.plot(fpr, tpr, color=color, lw=1.6, label=f"ROC (AUC = {r['auc']:.4f})")
    ax.plot(r["fpr"], r["tpr"], "o", ms=10, mfc="none", mec="red", mew=2.2,
            label=f"Empirical min-P(error): P(e) = {r['perr_min']:.4f}")
    if theory and "theory_tpr" in r:
        ax.plot(r["theory_fpr"], r["theory_tpr"], "x", ms=10, color="black", mew=2.2,
                label=f"Theoretical $\\gamma^*=0.538$: P(e) = {r['theory_perr']:.4f}")
    ax.plot([0, 1], [0, 1], ":", color="gray", lw=0.8)
    ax.set_xlabel(r"False positive rate $P(D=1\mid L=0)$")
    ax.set_ylabel(r"True positive rate $P(D=1\mid L=1)$")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)


for cur, r, t, c, fn in [(curA, rA, "Part A: ERM with true pdf", "C0", "q1a_roc"),
                         (curB, rB, "Part B: Naive Bayes (diagonal covariances)", "C2", "q1b_roc"),
                         (curC, rC, r"Part C: Fisher LDA, $w^T x$ vs. $\tau$", "C1", "q1c_roc")]:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    roc_plot(ax, cur, r, t, c)
    # zoomed inset near the operating point
    ins = ax.inset_axes([0.45, 0.32, 0.5, 0.36])
    ins.plot(cur[0], cur[1], color=c, lw=1.4)
    ins.plot(r["fpr"], r["tpr"], "o", ms=8, mfc="none", mec="red", mew=2)
    if "theory_tpr" in r:
        ins.plot(r["theory_fpr"], r["theory_tpr"], "x", ms=8, color="black", mew=2)
    ins.set_xlim(0, 0.2); ins.set_ylim(0.8, 1.0); ins.grid(alpha=0.3); ins.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(f"figs/{fn}.png", dpi=170); plt.close(fig)

# comparison plot
fig, ax = plt.subplots(figsize=(6.2, 5.2))
for cur, r, c in [(curA, rA, "C0"), (curB, rB, "C2"), (curC, rC, "C1")]:
    ax.plot(cur[0], cur[1], color=c, lw=1.5, label=f"{r['name']} (AUC {r['auc']:.4f}, min P(e) {r['perr_min']:.4f})")
    ax.plot(r["fpr"], r["tpr"], "o", ms=8, mfc="none", mec=c, mew=2)
ax.set_xlim(0, 0.3); ax.set_ylim(0.7, 1.0); ax.grid(alpha=0.3)
ax.set_xlabel(r"$P(D=1\mid L=0)$"); ax.set_ylabel(r"$P(D=1\mid L=1)$")
ax.set_title("ROC comparison (zoomed); circles = min-P(error) points")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig("figs/q1_compare.png", dpi=170); plt.close(fig)

# P(error) vs threshold
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, cur, r, xl in [(axs[0], curA, rA, r"$\ln\gamma$"), (axs[1], curB, rB, r"$\ln\gamma$"),
                       (axs[2], curC, rC, r"$\tau$")]:
    _, _, perr, thr = cur
    ok = np.isfinite(thr)
    ax.plot(thr[ok], perr[ok], lw=1.2)
    ax.axvline(r["thr_mid"], color="red", ls="--", lw=1, label=f"empirical min at {r['thr_mid']:.3f}")
    if r is not rC:
        ax.axvline(np.log(gamma_theory), color="k", ls=":", lw=1.2, label=fr"$\ln\gamma^*$ = {np.log(gamma_theory):.3f}")
    lo, hi = np.percentile(thr[ok], [2, 98])
    ax.set_xlim(lo, hi); ax.set_xlabel(xl); ax.set_ylabel("P(error)"); ax.set_title(r["name"])
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("figs/q1_perr_vs_thr.png", dpi=160); plt.close(fig)

print(json.dumps({k: v for k, v in res.items() if k != "lda"}, indent=1))
print("w", w, "w_closed", w_closed)
