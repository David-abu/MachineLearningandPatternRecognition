"""Question 3: Gaussian class-conditional (QDA-type) minimum-P(error) classifiers on real data."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2
from common import gauss_logpdf

DATA = "/home/claude/data"


# ------------------------------------------------------------------ data
def load_wine():
    df = pd.read_csv(f"{DATA}/wine_white.csv", sep=";")
    X = df.drop(columns="quality").to_numpy(float)
    y = df["quality"].to_numpy(int)
    return X, y, list(df.columns[:-1])


def load_har():
    Xtr = np.loadtxt(f"{DATA}/har/X_train.txt"); Xte = np.loadtxt(f"{DATA}/har/X_test.txt")
    ytr = np.loadtxt(f"{DATA}/har/y_train.txt", int); yte = np.loadtxt(f"{DATA}/har/y_test.txt", int)
    names = [l.split()[1] for l in open(f"{DATA}/har/activity_labels.txt")]
    return np.vstack([Xtr, Xte]), np.r_[ytr, yte], names, len(ytr)


# ------------------------------------------------------------------ model
def fit(X, y, alpha):
    """Per-class priors (counts), sample-average means and covariances, plus
    lambda_j = alpha * trace(C_j) / rank(C_j) added to the diagonal."""
    model = {}
    for c in np.unique(y):
        Xc = X[y == c]
        m = Xc.mean(0)
        C = (Xc - m).T @ (Xc - m) / len(Xc)            # 1/N_c sample-average estimator
        r = np.linalg.matrix_rank(C)
        lam = alpha * np.trace(C) / r
        model[c] = dict(prior=len(Xc) / len(X), m=m, C=C + lam * np.eye(X.shape[1]), lam=lam,
                        rank=int(r), n=len(Xc), cond_raw=float(np.linalg.cond(C)),
                        cond_reg=float(np.linalg.cond(C + lam * np.eye(X.shape[1]))))
    return model


def predict(model, X):
    cls = np.array(sorted(model))
    S = np.column_stack([gauss_logpdf(X, model[c]["m"], model[c]["C"]) + np.log(model[c]["prior"]) for c in cls])
    return cls[np.argmax(S, 1)]


def confusion(D, y, cls):
    """M[i,j] = P(D=cls_i | L=cls_j), counts as well."""
    K = len(cls)
    cnt = np.zeros((K, K), int)
    for a, ci in enumerate(cls):
        for b, cj in enumerate(cls):
            cnt[a, b] = np.sum((D == ci) & (y == cj))
    return cnt / cnt.sum(0, keepdims=True), cnt


def stratified_folds(y, k, rng):
    f = np.empty(len(y), int)
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        f[idx] = np.arange(len(idx)) % k
    return f


def heat(ax, M, cnt, ticks, title):
    ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if cnt[:, j].sum() == 0:
                continue
            ax.text(j, i, f"{M[i, j]:.2f}\n({cnt[i, j]})", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(ticks)), [f"L={t}" for t in ticks], fontsize=8, rotation=0)
    ax.set_yticks(range(len(ticks)), [f"D={t}" for t in ticks], fontsize=8)
    ax.set_title(title, fontsize=10)


def pca(X, k):
    Xc = X - X.mean(0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T, s ** 2 / np.sum(s ** 2)


res = {}
rng = np.random.default_rng(0)
alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1]

# =================================================================== WINE
Xw_raw, yw, wnames = load_wine()
mu_w, sd_w = Xw_raw.mean(0), Xw_raw.std(0)
Xw = (Xw_raw - mu_w) / sd_w                     # z-score (see report for why)
cls_w = np.unique(yw)
res["wine_counts"] = {int(c): int((yw == c).sum()) for c in cls_w}

folds = stratified_folds(yw, 5, rng)
sweep_w = []
for a in alphas:
    m = fit(Xw, yw, a)
    tr_err = float(np.mean(predict(m, Xw) != yw))
    cv_err = []
    for f in range(5):
        mf = fit(Xw[folds != f], yw[folds != f], a)
        cv_err.append(np.mean(predict(mf, Xw[folds == f]) != yw[folds == f]))
    sweep_w.append(dict(alpha=a, train=tr_err, cv=float(np.mean(cv_err))))
res["wine_sweep"] = sweep_w
alpha_w = min(sweep_w, key=lambda r: r["cv"])["alpha"]
mw = fit(Xw, yw, alpha_w)
Dw = predict(mw, Xw)
Mw, Cw = confusion(Dw, yw, cls_w)
res["wine"] = dict(alpha=alpha_w, perr=float(np.mean(Dw != yw)), conf=Mw.tolist(), counts=Cw.tolist(),
                   classes=cls_w.tolist(),
                   per_class={int(c): dict(n=mw[c]["n"], prior=mw[c]["prior"], rank=mw[c]["rank"],
                                           lam=mw[c]["lam"], cond_raw=mw[c]["cond_raw"],
                                           cond_reg=mw[c]["cond_reg"]) for c in cls_w},
                   majority_err=float(1 - max(np.mean(yw == c) for c in cls_w)))
# same classifier on raw (unstandardised) features, for comparison
mw_raw = fit(Xw_raw, yw, alpha_w)
res["wine"]["perr_raw_features"] = float(np.mean(predict(mw_raw, Xw_raw) != yw))
# within-one-grade accuracy
res["wine"]["within1"] = float(np.mean(np.abs(Dw - yw) <= 1))

# =================================================================== HAR
Xh, yh, hnames, ntr = load_har()
mu_h, sd_h = Xh.mean(0), Xh.std(0)
Xhs = (Xh - mu_h) / sd_h
cls_h = np.unique(yh)
res["har_counts"] = {int(c): int((yh == c).sum()) for c in cls_h}
sweep_h = []
for a in alphas:
    m = fit(Xhs, yh, a)
    tr_err = float(np.mean(predict(m, Xhs) != yh))
    # held-out check on the dataset's own subject-disjoint train/test split
    mt = fit(Xhs[:ntr], yh[:ntr], a)
    te_err = float(np.mean(predict(mt, Xhs[ntr:]) != yh[ntr:]))
    sweep_h.append(dict(alpha=a, train=tr_err, test=te_err))
    print("HAR", a, tr_err, te_err, flush=True)
res["har_sweep"] = sweep_h
alpha_h = min(sweep_h, key=lambda r: r["test"])["alpha"]
mh = fit(Xhs, yh, alpha_h)
Dh = predict(mh, Xhs)
Mh, Ch = confusion(Dh, yh, cls_h)
res["har"] = dict(alpha=alpha_h, perr=float(np.mean(Dh != yh)), conf=Mh.tolist(), counts=Ch.tolist(),
                  names=hnames,
                  per_class={int(c): dict(n=mh[c]["n"], prior=mh[c]["prior"], rank=mh[c]["rank"],
                                          lam=mh[c]["lam"], cond_raw=mh[c]["cond_raw"],
                                          cond_reg=mh[c]["cond_reg"]) for c in cls_h})
json.dump(res, open("out/q3_results.json", "w"), indent=2, default=float)

# =================================================================== FIGURES
# confusion matrices
fig, ax = plt.subplots(figsize=(6.4, 5.4))
heat(ax, Mw, Cw, cls_w, fr"Wine (white): $P(D=i\mid L=j)$, P(error) = {res['wine']['perr']:.4f}")
fig.tight_layout(); fig.savefig("figs/q3_wine_conf.png", dpi=160); plt.close(fig)
fig, ax = plt.subplots(figsize=(6.4, 5.4))
heat(ax, Mh, Ch, cls_h, fr"HAR: $P(D=i\mid L=j)$, P(error) = {res['har']['perr']:.4f}")
fig.tight_layout(); fig.savefig("figs/q3_har_conf.png", dpi=160); plt.close(fig)

# regularisation sweep
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
axs[0].semilogx(alphas, [r["train"] for r in sweep_w], "o-", label="all-sample (training) error")
axs[0].semilogx(alphas, [r["cv"] for r in sweep_w], "s-", label="5-fold CV error")
axs[0].axvline(alpha_w, color="k", ls=":"); axs[0].set_title("Wine"); axs[0].set_xlabel(r"$\alpha$")
axs[1].semilogx(alphas, [r["train"] for r in sweep_h], "o-", label="all-sample (training) error")
axs[1].semilogx(alphas, [r["test"] for r in sweep_h], "s-", label="held-out error (official split)")
axs[1].axvline(alpha_h, color="k", ls=":"); axs[1].set_title("HAR"); axs[1].set_xlabel(r"$\alpha$")
for a in axs:
    a.set_ylabel("P(error)"); a.grid(alpha=0.3, which="both"); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig("figs/q3_alpha_sweep.png", dpi=160); plt.close(fig)

# PCA projections
cmap = plt.get_cmap("tab10")
for nm, X, y, cls, lab in [("wine", Xw, yw, cls_w, lambda c: f"q={c}"),
                           ("har", Xhs, yh, cls_h, lambda c: hnames[c - 1])]:
    Z, evr = pca(X, 3)
    fig = plt.figure(figsize=(13, 4.6))
    ax1 = fig.add_subplot(1, 3, 1); ax2 = fig.add_subplot(1, 3, 2); ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    for k, c in enumerate(cls):
        i = y == c
        kw = dict(s=4, alpha=0.45, color=cmap(k), label=lab(c))
        ax1.scatter(Z[i, 0], Z[i, 1], **kw); ax2.scatter(Z[i, 0], Z[i, 2], **kw)
        ax3.scatter(Z[i, 0], Z[i, 1], Z[i, 2], **kw)
    ax1.set_xlabel(f"PC1 ({evr[0]:.1%})"); ax1.set_ylabel(f"PC2 ({evr[1]:.1%})")
    ax2.set_xlabel("PC1"); ax2.set_ylabel(f"PC3 ({evr[2]:.1%})")
    ax3.set_xlabel("PC1"); ax3.set_ylabel("PC2"); ax3.set_zlabel("PC3")
    ax1.legend(markerscale=4, fontsize=7, loc="best")
    fig.suptitle(f"{'White wine' if nm == 'wine' else 'HAR'}: first three principal components (standardised features)")
    fig.tight_layout(); fig.savefig(f"figs/q3_{nm}_pca.png", dpi=150); plt.close(fig)

# wine: feature-subset scatter + marginal histograms
fig, axs = plt.subplots(1, 3, figsize=(13, 4))
pairs = [("alcohol", "density"), ("alcohol", "volatile acidity"), ("residual sugar", "chlorides")]
for ax, (fx, fy) in zip(axs, pairs):
    ix, iy = wnames.index(fx), wnames.index(fy)
    for k, c in enumerate(cls_w):
        i = yw == c
        ax.scatter(Xw_raw[i, ix], Xw_raw[i, iy], s=4, alpha=0.5, color=cmap(k), label=f"q={c}")
    ax.set_xlabel(fx); ax.set_ylabel(fy)
axs[0].legend(markerscale=4, fontsize=7)
fig.suptitle("White wine: raw feature pairs")
fig.tight_layout(); fig.savefig("figs/q3_wine_pairs.png", dpi=150); plt.close(fig)

fig, axs = plt.subplots(2, 6, figsize=(15, 5))
for k, ax in enumerate(axs.ravel()):
    if k >= len(wnames):
        ax.axis("off"); continue
    ax.hist(Xw_raw[yw == 6, k], bins=50, density=True, color="C0", alpha=0.7)
    ax.set_title(wnames[k], fontsize=9); ax.tick_params(labelsize=7)
fig.suptitle("White wine, quality-6 class: marginal histograms (a Gaussian model implies symmetric, bell-shaped marginals)")
fig.tight_layout(); fig.savefig("figs/q3_wine_hist.png", dpi=140); plt.close(fig)

# HAR: a few raw features
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, (fx, fy) in zip(axs, [(0, 1), (40, 52)]):   # 0-based column indices into the 561 features
    for k, c in enumerate(cls_h):
        i = yh == c
        ax.scatter(Xh[i, fx], Xh[i, fy], s=3, alpha=0.4, color=cmap(k), label=hnames[c - 1])
    ax.set_xlabel(f"feature {fx + 1}"); ax.set_ylabel(f"feature {fy + 1}")
axs[0].legend(markerscale=4, fontsize=7)
fig.suptitle("HAR: two raw feature pairs")
fig.tight_layout(); fig.savefig("figs/q3_har_pairs.png", dpi=150); plt.close(fig)

# Gaussianity check: squared Mahalanobis distances vs chi^2_d quantiles (Q-Q)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
for ax, (nm, X, y, mdl, cs) in zip(axs, [("Wine", Xw, yw, mw, [5, 6, 7]), ("HAR", Xhs, yh, mh, [1, 4, 6])]):
    d = X.shape[1]
    for c in cs:
        Xc = X[y == c]
        m = Xc.mean(0); C = mdl[c]["C"]
        Lc = np.linalg.cholesky(C)
        r2 = np.sort(np.sum(np.linalg.solve(Lc, (Xc - m).T) ** 2, 0))
        q = chi2.ppf((np.arange(1, len(r2) + 1) - 0.5) / len(r2), d)
        lab = f"q={c}" if nm == "Wine" else hnames[c - 1]
        ax.plot(q, r2, ".", ms=2, label=lab)
    lim = ax.get_xlim()
    ax.plot(lim, lim, "k--", lw=1, label="y = x (Gaussian)")
    ax.set_xlabel(fr"$\chi^2_{{{d}}}$ quantiles"); ax.set_ylabel("sorted squared Mahalanobis distance")
    ax.set_title(f"{nm}: Q-Q check of the Gaussian model"); ax.legend(markerscale=5, fontsize=8)
    if nm == "Wine":
        ax.set_ylim(0, 150)
    ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("figs/q3_qq.png", dpi=150); plt.close(fig)

print(json.dumps({k: res[k] for k in ("wine_sweep", "har_sweep")}, indent=1))
print("wine", alpha_w, res["wine"]["perr"], res["wine"]["perr_raw_features"], res["wine"]["within1"], res["wine"]["majority_err"])
print(np.round(Mw, 3)); print(Cw)
print("har", alpha_h, res["har"]["perr"]); print(np.round(Mh, 4)); print(Ch)
for c in cls_w: print(c, res["wine"]["per_class"][int(c)])
for c in cls_h: print(c, res["har"]["per_class"][int(c)])
