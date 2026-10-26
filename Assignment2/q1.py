"""EECE5644 Assignment 2 - Question 1
Min-P(error) classifier for a 2-class Gaussian-mixture problem, plus ML-trained
logistic-linear and logistic-quadratic approximations of the class posterior.
"""
import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

rng = np.random.default_rng(5644)
os.makedirs("figs", exist_ok=True)

# ---------------- True data distribution ----------------
priors = np.array([0.6, 0.4])
means = {0: [np.array([-0.9, -1.1]), np.array([0.8, 0.75])],
         1: [np.array([-1.1, 0.9]), np.array([0.9, -0.75])]}
C = np.diag([0.75, 1.25])
w_mix = [0.5, 0.5]


def generate(N):
    labels = (rng.random(N) >= priors[0]).astype(int)          # P(L=1)=0.4
    X = np.zeros((N, 2))
    for l in (0, 1):
        idx = np.where(labels == l)[0]
        comp = (rng.random(idx.size) >= w_mix[0]).astype(int)
        for c in (0, 1):
            sub = idx[comp == c]
            X[sub] = rng.multivariate_normal(means[l][c], C, size=sub.size)
    return X, labels


def class_pdf(X, l):
    return sum(w_mix[c] * mvn.pdf(X, means[l][c], C) for c in (0, 1))


datasets = {n: generate(n) for n in (20, 200, 2000)}
Xv, Lv = generate(10000)

# ---------------- Part 1: theoretically optimal classifier ----------------
# Decide 1 iff ln p(x|1) - ln p(x|0) > ln(P0/P1)   (0-1 loss -> MAP rule)
score = np.log(class_pdf(Xv, 1)) - np.log(class_pdf(Xv, 0))
gamma_map = np.log(priors[0] / priors[1])


def roc(score, labels):
    thr = np.concatenate(([-np.inf], np.sort(score), [np.inf]))
    n1, n0 = (labels == 1).sum(), (labels == 0).sum()
    order = np.argsort(score)
    s_sorted, l_sorted = score[order], labels[order]
    # decisions: 1 if score > thr ; count of positives above each threshold
    tp = np.concatenate(([n1], n1 - np.cumsum(l_sorted == 1), [0]))
    fp = np.concatenate(([n0], n0 - np.cumsum(l_sorted == 0), [0]))
    return fp / n0, tp / n1, thr


def perr(dec, labels):
    return np.mean(dec != labels)


pfa, pd, thr = roc(score, Lv)
p_err_emp = (pfa * (Lv == 0).mean() + (1 - pd) * (Lv == 1).mean())
i_emp = np.argmin(p_err_emp)
dec_opt = (score > gamma_map).astype(int)
pe_opt = perr(dec_opt, Lv)
pfa_opt = np.mean(dec_opt[Lv == 0] == 1)
pd_opt = np.mean(dec_opt[Lv == 1] == 1)

print(f"Validation class counts: N0={np.sum(Lv==0)}, N1={np.sum(Lv==1)}")
print(f"Theoretical threshold gamma = P0/P1 = {priors[0]/priors[1]:.3f} (ln = {gamma_map:.4f})")
print(f"Min-P(error) classifier on D10K: P(error) = {pe_opt:.4f}  (P_FA={pfa_opt:.4f}, P_D={pd_opt:.4f})")
print(f"Empirically best threshold on ROC: ln gamma = {thr[i_emp]:.4f}, P(error) = {p_err_emp[i_emp]:.4f}")
counts = {(d, l): int(np.sum((dec_opt == d) & (Lv == l))) for d in (0, 1) for l in (0, 1)}
print("Decision/label counts (D,L):", counts)

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot(pfa, pd, lw=1.8, label="ROC (log-likelihood-ratio score)")
ax.plot(pfa_opt, pd_opt, "r*", ms=15, label=f"Theoretical min-P(error), γ=1.5\nP(error)={pe_opt:.4f}")
ax.plot(pfa[i_emp], pd[i_emp], "go", mfc="none", ms=11, label=f"Empirical min-P(error)\nP(error)={p_err_emp[i_emp]:.4f}")
ax.plot([0, 1], [0, 1], "k:", lw=0.8)
ax.set_xlabel("P(false alarm)  P(D=1|L=0)")
ax.set_ylabel("P(correct detection)  P(D=1|L=1)")
ax.set_title("Q1 Part 1: ROC of the optimal classifier on $D^{10K}_{validate}$")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("figs/q1_roc.png", dpi=160); plt.close(fig)

# grid for decision boundaries
gx, gy = np.meshgrid(np.linspace(-4.5, 4.5, 300), np.linspace(-5, 5, 300))
G = np.column_stack([gx.ravel(), gy.ravel()])
g_score = (np.log(class_pdf(G, 1)) - np.log(class_pdf(G, 0)) - gamma_map).reshape(gx.shape)


def scatter_with_boundary(ax, X, L, dec, Z, title):
    correct = dec == L
    for l, mk in ((0, "o"), (1, "^")):
        ax.scatter(*X[(L == l) & correct].T, s=6, marker=mk, c="tab:green", alpha=0.5,
                   label=f"L={l} correct")
        ax.scatter(*X[(L == l) & ~correct].T, s=8, marker=mk, c="tab:red", alpha=0.7,
                   label=f"L={l} incorrect")
    ax.contour(gx, gy, Z, levels=[0], colors="k", linewidths=1.6)
    ax.set_title(title, fontsize=9); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-5, 5)


fig, ax = plt.subplots(figsize=(6, 5.5))
scatter_with_boundary(ax, Xv, Lv, dec_opt, g_score,
                      f"Optimal (MAP) decision boundary on $D^{{10K}}_{{validate}}$, P(error)={pe_opt:.4f}")
ax.legend(fontsize=7, markerscale=2, loc="upper right")
fig.tight_layout(); fig.savefig("figs/q1_opt_boundary.png", dpi=160); plt.close(fig)

# ---------------- Part 2: logistic models trained by ML ----------------
def z_lin(X):
    return np.column_stack([np.ones(len(X)), X])


def z_quad(X):
    x1, x2 = X[:, 0], X[:, 1]
    return np.column_stack([np.ones(len(X)), x1, x2, x1**2, x1 * x2, x2**2])


def nll(w, Z, L):
    a = Z @ w
    # -sum[ l*log h + (1-l)*log(1-h) ] written stably with logaddexp
    return np.mean(np.logaddexp(0, a) - L * a)


def nll_grad(w, Z, L):
    h = 1 / (1 + np.exp(-(Z @ w)))
    return Z.T @ (h - L) / len(L)


def train(Z, L):
    res = minimize(nll, np.zeros(Z.shape[1]), args=(Z, L), jac=nll_grad,
                   method="BFGS", options={"gtol": 1e-6, "maxiter": 5000})
    return res.x, res


results = {}
for name, zf in (("linear", z_lin), ("quadratic", z_quad)):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    for j, n in enumerate((20, 200, 2000)):
        Xt, Lt = datasets[n]
        w, res = train(zf(Xt), Lt)
        # classify: h(x)≈P(L=1|x); min-P(error) rule decides 1 iff h>=0.5 (i.e. w^T z >= 0)
        dec_t = (zf(Xt) @ w >= 0).astype(int)
        dec_v = (zf(Xv) @ w >= 0).astype(int)
        pe_t, pe_v = perr(dec_t, Lt), perr(dec_v, Lv)
        results[(name, n)] = dict(w=w, pe_train=pe_t, pe_val=pe_v, nll=res.fun, it=res.nit, ok=res.success)
        print(f"logistic-{name:9s} N={n:5d}: train P(err)={pe_t:.4f}  validate P(err)={pe_v:.4f}  "
              f"NLL/N={res.fun:.4f} iters={res.nit} converged={res.success}  w={np.round(w,3)}")
        Zg = (zf(G) @ w).reshape(gx.shape)
        scatter_with_boundary(axes[0, j], Xt, Lt, dec_t, Zg,
                              f"logistic-{name}, trained on $D^{{{n}}}_{{train}}$ (train P(err)={pe_t:.3f})")
        axes[0, j].contour(gx, gy, g_score, levels=[0], colors="gray", linestyles="--", linewidths=1)
        scatter_with_boundary(axes[1, j], Xv, Lv, dec_v, Zg,
                              f"same model on $D^{{10K}}_{{validate}}$ (P(err)={pe_v:.4f})")
        axes[1, j].contour(gx, gy, g_score, levels=[0], colors="gray", linestyles="--", linewidths=1)
    axes[0, 0].legend(fontsize=7, markerscale=2, loc="upper right")
    fig.suptitle(f"Q1 Part 2: logistic-{name} classifiers (black = learned boundary, gray dashed = optimal)")
    fig.tight_layout(); fig.savefig(f"figs/q1_logistic_{name}.png", dpi=140); plt.close(fig)

# summary bar chart
fig, ax = plt.subplots(figsize=(6.5, 4))
xs = np.arange(3)
for k, name in enumerate(("linear", "quadratic")):
    vals = [results[(name, n)]["pe_val"] for n in (20, 200, 2000)]
    ax.bar(xs + (k - 0.5) * 0.35, vals, 0.35, label=f"logistic-{name}")
    for x, v in zip(xs + (k - 0.5) * 0.35, vals):
        ax.text(x, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
ax.axhline(pe_opt, color="r", ls="--", label=f"optimal classifier ({pe_opt:.3f})")
ax.set_xticks(xs, ["$N=20$", "$N=200$", "$N=2000$"])
ax.set_ylabel("Estimated P(error) on $D^{10K}_{validate}$")
ax.set_title("Q1: validation P(error) vs training-set size and model form")
ax.legend(fontsize=8); ax.set_ylim(0, 0.6)
fig.tight_layout(); fig.savefig("figs/q1_summary.png", dpi=160); plt.close(fig)

np.save("figs/q1_results.npy", {"pe_opt": pe_opt, "results": results}, allow_pickle=True)
