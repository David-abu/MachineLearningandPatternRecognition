"""EECE5644 Assignment 3 - Question 1 diagnostics (run after q1.py).
Re-creates the exact datasets and final MLPs of q1.py (same seeds) and records:
  * optimizer convergence of every restart of every final fit,
  * convergence of the cross-validation fits (re-run with their original seeds),
  * 95% Wilson intervals for each test P(error),
  * the test confusion matrix of the N=10000 MLP,
  * a decision-region slice at x3 = 0 (true-pdf MAP vs N=10000 MLP).
Imports q1's data definitions without re-running its experiment.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json
import numpy as np
from scipy.stats import multivariate_normal as mvn, norm
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mlp import MLP

# ---- reproduce q1.py's random stream exactly (same generator, same draw order) ----
SEED = 5644
rng = np.random.default_rng(SEED)
C, d = 4, 3
exec(open("q1.py").read().split("# ---------------- true data distribution ----------------")[1]
     .split("Ns = [100")[0])                        # rot, means, covs, priors, generate, map_classify
Ns = [100, 500, 1000, 5000, 10000]
train = {N: generate(N) for N in Ns}
Xtest, Ltest = generate(100_000)
res1 = json.load(open("q1_results.json"))
assert abs(np.mean(map_classify(Xtest) != Ltest) - res1["pe_opt"]) < 1e-12, "data stream mismatch"
P_grid = res1["P_grid"]


def wilson(k, n, z=norm.ppf(0.975)):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def cv_status(X, L, P, tr, va, seed):
    m = MLP(d, P, C).fit(X[tr], L[tr], restarts=2, rng=np.random.default_rng(seed), maxiter=1000)
    return int(np.sum(m.predict(X[va]) != L[va])), m.converged, [r[1] for r in m.restart_log]


out = {"final": {}, "cv": {}}
n_test = len(Ltest)
k_opt = int(np.sum(map_classify(Xtest) != Ltest))
out["opt_ci"] = wilson(k_opt, n_test)
models = {}
for N in Ns:
    X, L = train[N]
    Pb = res1["sets"][str(N)]["P_best"]
    m = MLP(d, Pb, C).fit(X, L, restarts=10, rng=np.random.default_rng(SEED + N), maxiter=3000)
    k = int(np.sum(m.predict(Xtest) != Ltest))
    assert abs(k / n_test - res1["sets"][str(N)]["pe_test"]) < 1e-12, f"final model mismatch N={N}"
    models[N] = m
    out["final"][N] = {"P": Pb, "test_errors": k, "ci": wilson(k, n_test),
                       "best_converged": m.converged, "best_iters": m.n_iter,
                       "restarts_converged": int(sum(r[1] for r in m.restart_log)),
                       "restart_nll_min": min(r[0] for r in m.restart_log),
                       "restart_nll_max": max(r[0] for r in m.restart_log)}
    print(f"N={N}: P={Pb} test errors={k} 95% CI=({out['final'][N]['ci'][0]:.4f},{out['final'][N]['ci'][1]:.4f}) "
          f"best restart converged={m.converged} ({m.n_iter} it); converged restarts "
          f"{out['final'][N]['restarts_converged']}/10; restart NLL range "
          f"[{out['final'][N]['restart_nll_min']:.4f}, {out['final'][N]['restart_nll_max']:.4f}]", flush=True)

    # CV convergence (same seeds as q1.py, so the error counts must reproduce)
    folds = list(KFold(n_splits=10, shuffle=True, random_state=SEED).split(X))
    jobs = [(P, f) for P in P_grid for f in range(10)]
    st = Parallel(n_jobs=2)(delayed(cv_status)(X, L, P, *folds[f], SEED + 1000 * P + f) for P, f in jobs)
    for P in P_grid:
        errs = sum(s[0] for (p, _), s in zip(jobs, st) if p == P)
        assert abs(errs / N - res1["sets"][str(N)]["cv_err"][str(P)]) < 1e-12, f"CV mismatch N={N} P={P}"
    retained_ok = sum(s[1] for s in st)
    all_ok = sum(sum(s[2]) for s in st)
    out["cv"][N] = {"retained_converged": retained_ok, "retained_total": len(st),
                    "runs_converged": all_ok, "runs_total": 2 * len(st)}
    print(f"   CV fits: retained solution converged {retained_ok}/{len(st)}; "
          f"all restarts converged {all_ok}/{2*len(st)}", flush=True)

conf = np.array([[np.sum((models[10000].predict(Xtest) == i) & (Ltest == j)) for j in range(C)] for i in range(C)])
out["conf_mlp10k"] = conf.tolist()
print("N=10000 MLP confusion (rows decision, cols label):\n", conf)
json.dump(out, open("q1_diagnostics.json", "w"), indent=1, default=float)

# decision-region slice at x3 = 0
g = np.linspace(-3.5, 3.5, 351)
G1, G2 = np.meshgrid(g, g)
Gx = np.column_stack([G1.ravel(), G2.ravel(), np.zeros(G1.size)])
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
cmap = matplotlib.colors.ListedColormap(["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
for ax, (title, D) in zip(axes, [("true-pdf MAP classifier", map_classify(Gx)),
                                 (f"MLP, N=100 (P={models[100].P})", models[100].predict(Gx)),
                                 (f"MLP, N=10000 (P={models[10000].P})", models[10000].predict(Gx))]):
    ax.imshow(D.reshape(G1.shape), origin="lower", extent=[g[0], g[-1], g[0], g[-1]], cmap=cmap,
              vmin=-0.5, vmax=3.5, alpha=0.85)
    ax.contour(G1, G2, map_classify(Gx).reshape(G1.shape), levels=[0.5, 1.5, 2.5], colors="k",
               linewidths=0.8, linestyles="--")
    ax.set_title(title); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
handles = [matplotlib.patches.Patch(color=cmap(c), label=f"decide class {c+1}") for c in range(C)]
axes[2].legend(handles=handles, fontsize=8, loc="lower right")
fig.suptitle("Q1: decision regions on the slice $x_3=0$ (dashed = optimal boundaries)")
fig.tight_layout(); fig.savefig("figs/q1_decision_slice.png", dpi=140); plt.close(fig)

# test P(error) with Wilson intervals (replaces the plain plot)
fig, ax = plt.subplots(figsize=(7, 4.6))
pe = np.array([out["final"][N]["test_errors"] / n_test for N in Ns])
lo = np.array([out["final"][N]["ci"][0] for N in Ns]); hi = np.array([out["final"][N]["ci"][1] for N in Ns])
ax.errorbar(Ns, pe, yerr=[pe - lo, hi - pe], fmt="o-", lw=2, ms=7, capsize=4,
            label="trained MLP (CV-selected P), 95% interval")
for N, p in zip(Ns, pe):
    ax.annotate(f"P={out['final'][N]['P']}\n{p:.4f}", (N, p), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=8)
ax.axhline(res1["pe_opt"], color="r", ls="--", label=f"theoretically optimal classifier ({res1['pe_opt']:.4f})")
ax.axhspan(*out["opt_ci"], color="r", alpha=0.12)
ax.set_xscale("log")
ax.set_xlabel("number of training samples (log scale)"); ax.set_ylabel("test-set empirical P(error)")
ax.set_title("Q1: test P(error) of MLP classifiers vs. training-set size")
ax.set_ylim(0.13, pe.max() + 0.04); ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig("figs/q1_test_perror.png", dpi=160); plt.close(fig)
print("diagnostics done")
