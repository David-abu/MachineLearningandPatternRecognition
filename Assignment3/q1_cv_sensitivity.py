"""EECE5644 Assignment 3 - Q1 sensitivity check (run after q1.py).
Re-runs the 10-fold CV for N=500 and N=1000 with a 5x larger L-BFGS iteration budget
(5000 instead of 1000) to test whether non-converged CV fits change the selected P.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json
import numpy as np
from scipy.stats import multivariate_normal as mvn
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from mlp import MLP

SEED = 5644
rng = np.random.default_rng(SEED)
C, d = 4, 3
exec(open("q1.py").read().split("# ---------------- true data distribution ----------------")[1]
     .split("Ns = [100")[0])
Ns = [100, 500, 1000, 5000, 10000]
train = {N: generate(N) for N in Ns}
res1 = json.load(open("q1_results.json"))
P_grid = res1["P_grid"]


def job(X, L, P, tr, va, seed):
    m = MLP(d, P, C).fit(X[tr], L[tr], restarts=2, rng=np.random.default_rng(seed), maxiter=5000)
    return int(np.sum(m.predict(X[va]) != L[va])), m.converged


out = {}
for N in (500, 1000):
    X, L = train[N]
    folds = list(KFold(n_splits=10, shuffle=True, random_state=SEED).split(X))
    jobs = [(P, f) for P in P_grid for f in range(10)]
    r = Parallel(n_jobs=2)(delayed(job)(X, L, P, *folds[f], SEED + 1000 * P + f) for P, f in jobs)
    cv = {P: sum(e for (p, _), (e, _) in zip(jobs, r) if p == P) / N for P in P_grid}
    conv = sum(c for _, c in r)
    Pb = min(P_grid, key=lambda P: (cv[P], P))
    orig = res1["sets"][str(N)]
    out[N] = {"cv_err_5000": cv, "P_best_5000": Pb, "converged": conv, "total": len(r)}
    print(f"N={N}: maxiter 1000 -> P={orig['P_best']}; maxiter 5000 -> P={Pb}; converged {conv}/{len(r)}")
    print("   P:        " + " ".join(f"{P:>6}" for P in P_grid))
    print("   it=1000:  " + " ".join(f"{orig['cv_err'][str(P)]:6.3f}" for P in P_grid))
    print("   it=5000:  " + " ".join(f"{cv[P]:6.3f}" for P in P_grid), flush=True)
json.dump(out, open("q1_cv_sensitivity.json", "w"), indent=1)
