"""EECE5644 Assignment 3 - Question 1
4-class, 3-D Gaussian data; theoretically optimal MAP classifier; 2-layer MLPs whose
hidden-layer size P is chosen by 10-fold cross-validation (min P(error)), trained by
maximum likelihood with random restarts, and evaluated on a 100,000-sample test set.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json, time
import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.optimize import approx_fprime
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mlp import MLP

os.makedirs("figs", exist_ok=True)
SEED = 5644
rng = np.random.default_rng(SEED)
C, d = 4, 3

# ---------------- true data distribution ----------------
def rot(a, b, c):
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

scale = 0.85                                   # tuned so Bayes error is ~15%
means = scale * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
eigs = [[1.0, 0.6, 0.3], [0.8, 0.8, 0.4], [1.2, 0.5, 0.5], [0.7, 0.6, 0.9]]
angs = [(0.3, 0.5, 0.2), (1.0, -0.4, 0.8), (-0.6, 0.9, -0.3), (0.2, 0.2, 1.2)]
covs = [rot(*a) @ np.diag(e) @ rot(*a).T for a, e in zip(angs, eigs)]
priors = np.full(C, 1 / C)


def generate(N):
    L = rng.choice(C, size=N, p=priors)
    X = np.zeros((N, d))
    for c in range(C):
        i = L == c
        X[i] = rng.multivariate_normal(means[c], covs[c], i.sum())
    return X, L


def map_classify(X):
    # min-P(error) rule: argmax_c  ln p(x|c) + ln P(c)
    return np.column_stack([mvn.logpdf(X, means[c], covs[c]) + np.log(priors[c])
                            for c in range(C)]).argmax(axis=1)


Ns = [100, 500, 1000, 5000, 10000]
train = {N: generate(N) for N in Ns}
Xtest, Ltest = generate(100_000)

# ---------------- theoretically optimal classifier ----------------
D_opt = map_classify(Xtest)
pe_opt = float(np.mean(D_opt != Ltest))
conf_opt = np.array([[np.sum((D_opt == i) & (Ltest == j)) for j in range(C)] for i in range(C)])
print(f"Test class counts: {np.bincount(Ltest, minlength=C).tolist()}")
print(f"Optimal (true-pdf MAP) classifier test P(error) = {pe_opt:.4f}")
print("Confusion counts (rows = decision, cols = true label):\n", conf_opt)

# ---------------- tool sanity checks ----------------
chk = MLP(d, 5, C); r0 = np.random.default_rng(0)
Xc, Yc = r0.normal(size=(40, d)), np.eye(C)[r0.integers(0, C, 40)]
th = chk.init(r0)
num = approx_fprime(th, lambda t: chk.nll_grad(t, Xc, Yc)[0], 1e-6)
ana = chk.nll_grad(th, Xc, Yc)[1]
grad_rel_err = float(np.linalg.norm(num - ana) / np.linalg.norm(ana))
print(f"Gradient check: relative error between analytic and finite-difference gradient = {grad_rel_err:.2e}")
chk.theta = th; chk.mu, chk.sd = np.zeros(d), np.ones(d)
print("Softmax outputs sum to 1:", np.allclose(chk.predict_proba(Xc).sum(axis=1), 1.0))
# a sufficiently large MLP trained on lots of data should approach the Bayes classifier
# (checked at the end via the N=10000 result)

# ---------------- model-order selection by 10-fold CV ----------------
P_grid = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
K = 10
CV_RESTARTS, CV_MAXITER = 2, 1000
FINAL_RESTARTS, FINAL_MAXITER = 10, 3000


def cv_job(X, L, P, tr, va, seed):
    m = MLP(d, P, C).fit(X[tr], L[tr], restarts=CV_RESTARTS,
                         rng=np.random.default_rng(seed), maxiter=CV_MAXITER)
    return int(np.sum(m.predict(X[va]) != L[va]))


results = {"pe_opt": pe_opt, "conf_opt": conf_opt.tolist(), "P_grid": P_grid,
           "grad_rel_err": grad_rel_err, "sets": {}}
t0 = time.time()
for N in Ns:
    X, L = train[N]
    folds = list(KFold(n_splits=K, shuffle=True, random_state=SEED).split(X))
    # check the folds partition the data: disjoint validation sets that cover every sample once
    allva = np.concatenate([va for _, va in folds])
    assert np.array_equal(np.sort(allva), np.arange(N))
    assert all(len(np.intersect1d(tr, va)) == 0 for tr, va in folds)
    jobs = [(P, f) for P in P_grid for f in range(K)]
    errs = Parallel(n_jobs=2)(delayed(cv_job)(X, L, P, *folds[f], SEED + 1000 * P + f)
                              for P, f in jobs)
    cv_err = {P: sum(e for (p, _), e in zip(jobs, errs) if p == P) / N for P in P_grid}
    P_best = min(P_grid, key=lambda P: (cv_err[P], P))      # ties -> smaller network
    # final ML training on the whole training set with many restarts
    final = MLP(d, P_best, C).fit(X, L, restarts=FINAL_RESTARTS,
                                  rng=np.random.default_rng(SEED + N), maxiter=FINAL_MAXITER)
    D = final.predict(Xtest)
    pe_test = float(np.mean(D != Ltest))
    pe_train = float(np.mean(final.predict(X) != L))
    results["sets"][N] = {"cv_err": cv_err, "P_best": P_best, "pe_test": pe_test,
                          "pe_train": pe_train, "train_nll": float(final.train_nll),
                          "n_params": final.n_params}
    print(f"N={N:6d}: CV P(error) by P = " + ", ".join(f"{P}:{cv_err[P]:.3f}" for P in P_grid))
    print(f"          best P = {P_best} ({final.n_params} params), train P(err) = {pe_train:.4f}, "
          f"train NLL = {final.train_nll:.4f}, TEST P(error) = {pe_test:.4f}   [{time.time()-t0:.0f}s]",
          flush=True)

with open("q1_results.json", "w") as f:
    json.dump(results, f, indent=1)

# ---------------- figures ----------------
fig = plt.figure(figsize=(6.5, 5.5))
ax = fig.add_subplot(projection="3d")
X, L = train[1000]
for c, mk in zip(range(C), "o^sD"):
    ax.scatter(*X[L == c].T, s=6, marker=mk, alpha=0.6, label=f"class {c+1}")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_zlabel("$x_3$")
ax.set_title("Q1: training set with N = 1000"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("figs/q1_data.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.6))
for N in Ns:
    ce = results["sets"][N]["cv_err"]
    line, = ax.plot(P_grid, [ce[P] for P in P_grid], "o-", ms=4, label=f"N = {N}")
    Pb = results["sets"][N]["P_best"]
    ax.plot(Pb, ce[Pb], "*", ms=14, color=line.get_color(), mec="k")
ax.axhline(pe_opt, color="k", ls="--", lw=1, label=f"optimal classifier ({pe_opt:.3f})")
ax.set_xscale("log", base=2); ax.set_xticks(P_grid, [str(p) for p in P_grid])
ax.set_xlabel("number of hidden perceptrons P"); ax.set_ylabel("10-fold CV estimate of P(error)")
ax.set_title("Q1: model-order selection (stars = selected P)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("figs/q1_cv.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.6))
pe = [results["sets"][N]["pe_test"] for N in Ns]
ax.semilogx(Ns, pe, "o-", lw=2, ms=7, label="trained MLP (CV-selected P)")
for N, p in zip(Ns, pe):
    ax.annotate(f"P={results['sets'][N]['P_best']}\n{p:.4f}", (N, p), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=8)
ax.axhline(pe_opt, color="r", ls="--", label=f"theoretically optimal classifier ({pe_opt:.4f})")
ax.set_xlabel("number of training samples (log scale)"); ax.set_ylabel("test-set empirical P(error)")
ax.set_title("Q1: test P(error) of MLP classifiers vs. training-set size")
ax.set_ylim(min(pe_opt, min(pe)) - 0.02, max(pe) + 0.05)
ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig("figs/q1_test_perror.png", dpi=160); plt.close(fig)
print(f"done in {time.time()-t0:.0f}s")
