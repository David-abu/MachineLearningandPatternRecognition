"""Question 2: 3-class, 3-D Gaussian-mixture problem; MAP (0-1 loss) and ERM with Lambda_10, Lambda_100."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from common import gauss_logpdf

rng = np.random.default_rng(2026)
N = 10000
priors = np.array([0.3, 0.3, 0.4])

# Four component means on a regular tetrahedron with edge length a = 2.5
a = 2.5
means = np.array([[0, 0, 0],
                  [a, 0, 0],
                  [a / 2, a * np.sqrt(3) / 2, 0],
                  [a / 2, a * np.sqrt(3) / 6, a * np.sqrt(2 / 3)]])
covs = np.array([[[1.0, 0.3, -0.2], [0.3, 0.8, 0.1], [-0.2, 0.1, 1.2]],
                 [[0.9, -0.2, 0.1], [-0.2, 1.1, 0.3], [0.1, 0.3, 0.8]],
                 [[1.2, 0.2, 0.0], [0.2, 0.9, -0.3], [0.0, -0.3, 1.0]],
                 [[0.8, 0.0, 0.2], [0.0, 1.0, 0.0], [0.2, 0.0, 1.1]]])
for C in covs:
    assert np.all(np.linalg.eigvalsh(C) > 0)
# class 1 <- component 0, class 2 <- component 1, class 3 <- 0.5*comp 2 + 0.5*comp 3
avg_std = np.mean([np.sqrt(np.trace(C) / 3) for C in covs])
dists = [np.linalg.norm(means[i] - means[j]) for i in range(4) for j in range(i + 1, 4)]

# ---- 1. generate data ----
L = rng.choice([1, 2, 3], size=N, p=priors)
comp = np.where(L == 1, 0, np.where(L == 2, 1, 2 + (rng.random(N) < 0.5)))
X = np.empty((N, 3))
for c in range(4):
    idx = comp == c
    X[idx] = rng.multivariate_normal(means[c], covs[c], idx.sum())
np.savez("out/q2_data.npz", X=X, L=L, comp=comp)


def class_loglik(X):
    """N x 3 matrix of ln p(x|L=j)."""
    l1 = gauss_logpdf(X, means[0], covs[0])
    l2 = gauss_logpdf(X, means[1], covs[1])
    l3 = np.logaddexp(np.log(0.5) + gauss_logpdf(X, means[2], covs[2]),
                      np.log(0.5) + gauss_logpdf(X, means[3], covs[3]))
    return np.c_[l1, l2, l3]


ll = class_loglik(X)
lj = ll + np.log(priors)                                     # ln p(x|j)P(j)
post = np.exp(lj - np.logaddexp.reduce(lj, axis=1, keepdims=True))   # P(L=j|x)

Lambda = {"0-1": 1 - np.eye(3),
          "10": np.array([[0, 1, 10], [1, 0, 10], [1, 1, 0]], float),
          "100": np.array([[0, 1, 100], [1, 0, 100], [1, 1, 0]], float)}

res = {"means": means.tolist(), "covs": covs.tolist(), "avg_std": avg_std,
       "pairwise_dist": dists, "class_counts": [int((L == j).sum()) for j in (1, 2, 3)]}

for key, Lam in Lambda.items():
    R = post @ Lam.T                       # R[n,i] = sum_j Lam[i,j] P(j|x_n)  (conditional risk)
    D = np.argmin(R, axis=1) + 1
    conf = np.zeros((3, 3))
    counts = np.zeros((3, 3), int)
    for i in (1, 2, 3):
        for j in (1, 2, 3):
            counts[i - 1, j - 1] = np.sum((D == i) & (L == j))
            conf[i - 1, j - 1] = counts[i - 1, j - 1] / np.sum(L == j)
    risk = float(np.mean(Lam[D - 1, L - 1]))            # sample average of incurred loss
    perr = float(np.mean(D != L))
    res[key] = dict(conf=conf.tolist(), counts=counts.tolist(), risk=risk, perr=perr,
                    dec_counts=[int((D == i).sum()) for i in (1, 2, 3)],
                    exp_risk_cond=float(np.mean(R.min(axis=1))))
    np.save(f"out/q2_D_{key}.npy", D)

    # ---- 3-D scatter plot ----
    fig = plt.figure(figsize=(12, 6.2))
    markers = {1: ".", 2: "^", 3: "s"}
    sizes = {1: 30, 2: 10, 3: 8}
    ok = D == L
    for p, (elev, azim) in enumerate([(22, -60), (12, 30)]):
        ax = fig.add_subplot(1, 2, p + 1, projection="3d")
        for j in (1, 2, 3):
            for good, col in ((True, "green"), (False, "red")):
                idx = (L == j) & (ok == good)
                ax.scatter(*X[idx].T, marker=markers[j], s=sizes[j],
                           c=col, alpha=0.35 if good else 0.8, linewidths=0.3)
        ax.view_init(elev, azim)
        ax.set_box_aspect(None, zoom=1.0)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_zlabel("$x_3$")
    handles = [Line2D([], [], marker=markers[j], ls="", color="gray", ms=7, label=f"L = {j}") for j in (1, 2, 3)]
    handles += [Line2D([], [], marker="o", ls="", color="green", label="correct"),
                Line2D([], [], marker="o", ls="", color="red", label="incorrect")]
    name = "MAP (0-1 loss)" if key == "0-1" else fr"ERM with $\Lambda_{{{key}}}$"
    fig.suptitle(f"{name}: P(error) = {perr:.4f}, estimated min. expected risk = {risk:.4f}")
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.07, wspace=0.08)
    fig.savefig(f"figs/q2_scatter_{key.replace('-', '')}.png", dpi=150); plt.close(fig)

# ---- confusion-matrix heat maps side by side ----
fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, key in zip(axs, Lambda):
    M = np.array(res[key]["conf"])
    ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                    color="white" if M[i, j] > 0.6 else "black", fontsize=10)
    ax.set_xticks(range(3), ["L=1", "L=2", "L=3"]); ax.set_yticks(range(3), ["D=1", "D=2", "D=3"])
    t = "0-1 loss (MAP)" if key == "0-1" else fr"$\Lambda_{{{key}}}$"
    ax.set_title(f"{t}\nrisk {res[key]['risk']:.4f}, P(e) {res[key]['perr']:.4f}", fontsize=10)
fig.suptitle(r"Confusion matrices $P(D=i\mid L=j)$ (columns sum to 1)")
fig.tight_layout(); fig.savefig("figs/q2_confusions.png", dpi=150); plt.close(fig)

# ---- risk/error trade-off as the class-3 loss weight grows ----
ws = np.r_[1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 200, 500]
rk, pe, p3 = [], [], []
for wgt in ws:
    Lam = np.array([[0, 1, wgt], [1, 0, wgt], [1, 1, 0]], float)
    D = np.argmin(post @ Lam.T, axis=1) + 1
    rk.append(np.mean(Lam[D - 1, L - 1])); pe.append(np.mean(D != L)); p3.append(np.mean(D == 3))
res["sweep"] = dict(w=ws.tolist(), risk=rk, perr=pe, frac_D3=p3)
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.semilogx(ws, pe, "o-", label="P(error)")
ax.semilogx(ws, p3, "s-", label="fraction decided D=3")
ax.semilogx(ws, rk, "^-", label="estimated expected risk")
ax.set_xlabel("loss for misclassifying a class-3 sample (w)"); ax.grid(alpha=0.3, which="both")
ax.legend(); ax.set_title("Effect of the class-3 loss weight")
fig.tight_layout(); fig.savefig("figs/q2_sweep.png", dpi=150); plt.close(fig)

json.dump(res, open("out/q2_results.json", "w"), indent=2)
for key in Lambda:
    print(key, res[key]["risk"], res[key]["perr"], res[key]["dec_counts"])
    print(np.round(np.array(res[key]["conf"]), 4))
print("avg std", avg_std, "dists", np.round(dists, 3), res["class_counts"])
