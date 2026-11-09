"""EECE5644 Assignment 3 - Question 2
GMM model-order selection with 10-fold cross-validation (validation log-likelihood),
for datasets of 10, 100 and 1000 samples drawn from a 4-component 2-D GMM in which two
components overlap strongly. Repeated R times to estimate the selection rate of each order.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json, sys, time, warnings
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import multivariate_normal as mvn
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figs", exist_ok=True)
SEED = 5644
R = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 100        # repetitions
Ns = [10, 100, 1000]
Ks = list(range(1, 11))
NFOLD = 10
REG_COVAR = 1e-3        # covariance floor added to each diagonal (see report)
N_INIT = 3              # EM restarts per fit (k-means++ initializations)

# ---------------- true GMM ----------------
w_true = np.array([0.30, 0.25, 0.25, 0.20])
m_true = np.array([[0.0, 0.0], [1.6, 0.6], [6.0, 4.0], [-5.0, 5.0]])
C_true = np.array([[[1.0, 0.3], [0.3, 0.8]],
                   [[0.7, -0.2], [-0.2, 1.0]],
                   [[1.5, 0.6], [0.6, 0.9]],
                   [[0.6, 0.0], [0.0, 1.8]]])
avg_eig = [np.linalg.eigvalsh(C).mean() for C in C_true]
dist12 = np.linalg.norm(m_true[0] - m_true[1])


def sample_gmm(N, rng):
    comp = rng.choice(4, size=N, p=w_true)
    X = np.zeros((N, 2))
    for k in range(4):
        i = comp == k
        X[i] = rng.multivariate_normal(m_true[k], C_true[k], i.sum())
    return X, comp


def true_logpdf(X):
    return np.logaddexp.reduce(np.column_stack(
        [np.log(w_true[k]) + mvn.logpdf(X, m_true[k], C_true[k]) for k in range(4)]), axis=1)


# ---------------- one repetition ----------------
def one_run(N, rep):
    rng = np.random.default_rng([SEED, N, rep])
    X, _ = sample_gmm(N, rng)
    folds = KFold(n_splits=NFOLD, shuffle=True, random_state=int(rng.integers(1 << 31))).split(X)
    val_ll = np.zeros(len(Ks))
    for tr, va in folds:
        for j, K in enumerate(Ks):
            if K > len(tr):            # cannot fit more components than training samples
                val_ll[j] = -np.inf
                continue
            gm = GaussianMixture(n_components=K, covariance_type="full", reg_covar=REG_COVAR,
                                 n_init=N_INIT, init_params="k-means++", max_iter=500,
                                 tol=1e-4, random_state=int(rng.integers(1 << 31)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gm.fit(X[tr])                                  # ML (EM) on training folds
            val_ll[j] += gm.score_samples(X[va]).sum()         # validation log-likelihood
    val_ll /= N                                                # average per sample
    return Ks[int(np.argmax(val_ll))], val_ll


def run_experiment():
    print(f"distance between means of components 1 and 2 = {dist12:.3f}; "
          f"sum of their average covariance eigenvalues = {avg_eig[0] + avg_eig[1]:.3f}")
    t0 = time.time()
    sel = {}; curves = {}
    for N in Ns:
        out = Parallel(n_jobs=2)(delayed(one_run)(N, r) for r in range(R))
        sel[N] = np.array([o[0] for o in out]); curves[N] = np.array([o[1] for o in out])
        rate = np.array([np.mean(sel[N] == K) for K in Ks])
        print(f"N={N:5d}: selection rate by K = " +
              " ".join(f"{K}:{100*r:.0f}%" for K, r in zip(Ks, rate)) +
              f"   mode={Ks[int(np.argmax(rate))]}  mean={sel[N].mean():.2f}  [{time.time()-t0:.0f}s]",
              flush=True)

    # true-density reference: expected per-sample log-likelihood (entropy estimate)
    Xbig, _ = sample_gmm(200000, np.random.default_rng(1))
    ll_true = float(true_logpdf(Xbig).mean())
    print(f"E[log p_true(x)] ≈ {ll_true:.4f}")

    res = {"R": R, "Ns": Ns, "Ks": Ks, "dist12": float(dist12), "avg_eig": [float(a) for a in avg_eig],
           "ll_true": ll_true,
           "selected": {N: sel[N].tolist() for N in Ns},
           "rates": {N: [float(np.mean(sel[N] == K)) for K in Ks] for N in Ns}}
    with open("q2_results.json", "w") as f:
        json.dump(res, f, indent=1)
    np.savez("q2_curves.npz", **{f"N{N}": curves[N] for N in Ns})
    print(f"experiment done in {time.time()-t0:.0f}s")


def make_figures():
    """All figures are drawn from the saved results, so they can be redrawn without rerunning."""
    res = json.load(open("q2_results.json"))
    res["rates"] = {int(k): v for k, v in res["rates"].items()}
    ll_true = res["ll_true"]
    npz = np.load("q2_curves.npz"); curves = {N: npz[f"N{N}"] for N in Ns}
    # ---------------- figures ----------------
    # (a) true GMM and one dataset of each size
    gx, gy = np.meshgrid(np.linspace(-9, 11, 300), np.linspace(-4, 10, 220))
    Z = np.exp(true_logpdf(np.column_stack([gx.ravel(), gy.ravel()]))).reshape(gx.shape)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, N in zip(axes, Ns):
        X, comp = sample_gmm(N, np.random.default_rng([SEED, N, 0]))
        ax.contour(gx, gy, Z, levels=8, colors="gray", linewidths=0.8)
        for k, mk in zip(range(4), "o^sD"):
            ax.scatter(*X[comp == k].T, s=12 if N < 1000 else 5, marker=mk, label=f"component {k+1}")
        ax.plot(*m_true.T, "k+", ms=12, mew=2)
        ax.set_title(f"N = {N} (repetition 1)"); ax.set_aspect("equal")
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Q2: true GMM density contours with sample datasets (components 1 and 2 overlap)")
    fig.tight_layout(); fig.savefig("figs/q2_data.png", dpi=140); plt.close(fig)

    # (b) selection-rate bar chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    wbar = 0.27
    for i, N in enumerate(Ns):
        r = np.array(res["rates"][N]) * 100
        ax.bar(np.array(Ks) + (i - 1) * wbar, r, wbar, label=f"N = {N}")
        for K, v in zip(Ks, r):
            if v >= 1:
                ax.text(K + (i - 1) * wbar, v + 1, f"{v:.0f}", ha="center", fontsize=7)
    ax.axvline(4, color="k", ls=":", lw=1); ax.text(4.05, 92, "true order", fontsize=8)
    ax.set_xticks(Ks); ax.set_xlabel("selected number of components K")
    ax.set_ylabel(f"selection rate over {R} repetitions (%)"); ax.set_ylim(0, 100)
    ax.set_title("Q2: GMM order selected by 10-fold cross-validation")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig("figs/q2_selection_rates.png", dpi=160); plt.close(fig)

    # (c) validation log-likelihood curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, N in zip(axes, Ns):
        cv = curves[N].copy()
        cv_f = np.where(np.isfinite(cv), cv, np.nan)
        cv_f[:, np.all(np.isnan(cv_f), axis=0)] = np.nan
        med = np.nanmedian(cv_f, axis=0)
        lo, hi = np.nanpercentile(cv_f, [10, 90], axis=0)
        if N == 10:
            # held-out scores span three orders of magnitude here: plot the NEGATIVE
            # validation log-likelihood on a log axis (lower is better)
            ax.fill_between(Ks, -hi, -lo, alpha=0.25, label="10–90% across repetitions")
            ax.plot(Ks, -med, "o-", label="median")
            ax.axhline(-ll_true, color="r", ls="--", lw=1, label="true-pdf expected value")
            ax.set_yscale("log"); ax.set_ylabel("NEGATIVE avg. validation log-likelihood (log scale)")
            ax.text(9.6, -ll_true * 1.3, "K=10 infeasible\n(9 training points)", fontsize=7, ha="right")
        else:
            ax.fill_between(Ks, lo, hi, alpha=0.25, label="10–90% across repetitions")
            ax.plot(Ks, med, "o-", label="median")
            ax.axhline(ll_true, color="r", ls="--", lw=1, label="true-pdf expected log-lik.")
            ax.set_ylabel("average validation log-likelihood")
            ax.set_ylim(min(np.nanmin(lo[:6]), ll_true) - 0.3, max(np.nanmax(hi), ll_true) + 0.2)
        ax.axvline(4, color="k", ls=":", lw=1)
        ax.set_xticks(Ks); ax.set_xlabel("K"); ax.set_title(f"N = {N}")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Q2: 10-fold CV validation log-likelihood per sample vs. number of components")
    fig.tight_layout(); fig.savefig("figs/q2_cv_curves.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    # usage: python q2.py [R]        -> run experiment with R repetitions, then draw figures
    #        python q2.py --plots    -> redraw figures from saved results only
    if "--plots" not in sys.argv:
        run_experiment()
    make_figures()
