"""EECE5644 Assignment 2 - Question 2
ML and MAP estimation of a cubic polynomial regression y = c(x,w) + v, x in R^2.
Data generator reproduced from the course-provided hw2q2.py (plotting removed,
seeded RNG added for reproducibility). Ntrain=100, Nvalidate=1000.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("figs", exist_ok=True)
np.random.seed(5644)


# ---------- course-provided generator (hw2q2.py), unchanged logic ----------
def generateData(N):
    priors = [.3, .4, .3]
    meanVectors = np.array([[-10, 0, 10], [0, 0, 0], [10, 0, -10]])
    cov = np.zeros((3, 3, 3))
    cov[:, :, 0] = [[1, 0, -3], [0, 1, 0], [-3, 0, 15]]
    cov[:, :, 1] = [[8, 0, 0], [0, .5, 0], [0, 0, .5]]
    cov[:, :, 2] = [[1, 0, -3], [0, 1, 0], [-3, 0, 15]]
    x = np.zeros((3, N))
    u = np.random.random(N)
    thresholds = np.append(np.cumsum(priors), 1)
    for l in range(3):
        ind = np.where(u <= thresholds[l])[0]
        u[ind] = 1.1
        x[:, ind] = np.random.multivariate_normal(meanVectors[:, l], cov[:, :, l], len(ind)).T
    return x


def hw2q2(Ntrain=100, Nvalidate=1000):
    d = generateData(Ntrain); xT, yT = d[0:2, :].T, d[2, :]
    d = generateData(Nvalidate); xV, yV = d[0:2, :].T, d[2, :]
    return xT, yT, xV, yV


# ---------- model ----------
def z_cubic(X):
    """All monomials of total degree <= 3 in (x1,x2): 10 features."""
    x1, x2 = X[:, 0], X[:, 1]
    return np.column_stack([np.ones(len(X)), x1, x2,
                            x1**2, x1 * x2, x2**2,
                            x1**3, x1**2 * x2, x1 * x2**2, x2**3])


def w_ml(Z, y):
    return np.linalg.solve(Z.T @ Z, Z.T @ y)


def w_map(Z, y, gamma, sigma2):
    return np.linalg.solve(Z.T @ Z + (sigma2 / gamma) * np.eye(Z.shape[1]), Z.T @ y)


def mse(Z, y, w):
    return np.mean((y - Z @ w) ** 2)


gammas = np.logspace(-10, 6, 161)

# ---------- main experiment (one dataset) ----------
xT, yT, xV, yV = hw2q2()
ZT, ZV = z_cubic(xT), z_cubic(xV)
wML = w_ml(ZT, yT)
resid = yT - ZT @ wML
sigma2 = resid @ resid / (len(yT) - ZT.shape[1])   # unbiased noise-variance estimate from ML fit
mse_ml_train, mse_ml_val = mse(ZT, yT, wML), mse(ZV, yV, wML)
print(f"sigma^2 estimate (from ML residuals) = {sigma2:.4f}")
print(f"ML : train MSE = {mse_ml_train:.4f}, validation MSE = {mse_ml_val:.4f}")
print("w_ML =", np.round(wML, 4))

W = np.array([w_map(ZT, yT, g, sigma2) for g in gammas])
mse_map_val = np.array([mse(ZV, yV, w) for w in W])
mse_map_tr = np.array([mse(ZT, yT, w) for w in W])
ib = np.argmin(mse_map_val)
print(f"MAP best gamma = {gammas[ib]:.3g}: validation MSE = {mse_map_val[ib]:.4f} "
      f"(train MSE {mse_map_tr[ib]:.4f})")
for g in (1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6):
    w = w_map(ZT, yT, g, sigma2)
    print(f"  gamma=1e{int(np.log10(g)):+d}: val MSE={mse(ZV, yV, w):9.4f}  ||w||={np.linalg.norm(w):.4f}  "
          f"||w_MAP - w_ML||={np.linalg.norm(w - wML):.4f}")
print("w_MAP(best) =", np.round(W[ib], 4))

# ---------- repeated trials to show the trend isn't a one-dataset artifact ----------
R = 100
curves = np.zeros((R, len(gammas))); ml_vals = np.zeros(R)
for r in range(R):
    a, b, c, d = hw2q2()
    Za, Zc = z_cubic(a), z_cubic(c)
    wm = w_ml(Za, b)
    s2 = np.sum((b - Za @ wm) ** 2) / (len(b) - 10)
    ml_vals[r] = mse(Zc, d, wm)
    curves[r] = [mse(Zc, d, w_map(Za, b, g, s2)) for g in gammas]
best_g = gammas[np.argmin(curves, axis=1)]
print(f"\n{R} trials: median ML val MSE = {np.median(ml_vals):.3f}; "
      f"median best-MAP val MSE = {np.median(curves.min(axis=1)):.3f}; "
      f"median best gamma = {np.median(best_g):.3g} (IQR {np.percentile(best_g,25):.3g}-{np.percentile(best_g,75):.3g})")
print(f"fraction of trials where best MAP beats ML: {np.mean(curves.min(axis=1) < ml_vals):.2f}")

# ---------- figures ----------
fig = plt.figure(figsize=(11, 4.5))
for k, (X, y, t) in enumerate(((xT, yT, "Training set (N=100)"), (xV, yV, "Validation set (N=1000)"))):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], y, s=6)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_zlabel("y"); ax.set_title(t)
fig.tight_layout(); fig.savefig("figs/q2_data.png", dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.semilogx(gammas, mse_map_val, lw=2, label="MAP, validation MSE")
ax.semilogx(gammas, mse_map_tr, lw=1.2, ls="--", label="MAP, training MSE")
ax.axhline(mse_ml_val, color="r", ls=":", label=f"ML validation MSE = {mse_ml_val:.3f}")
ax.plot(gammas[ib], mse_map_val[ib], "k*", ms=12, label=f"best γ = {gammas[ib]:.2g} (MSE {mse_map_val[ib]:.3f})")
ax.set_xlabel("prior variance γ  (w ~ N(0, γI))"); ax.set_ylabel("average squared error")
ax.set_title("Q2: validation error of MAP estimator vs γ")
ax.set_yscale("log"); ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig("figs/q2_mse_vs_gamma.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
for j in range(10):
    ax.semilogx(gammas, W[:, j], lw=1.2)
ax.set_xlabel("γ"); ax.set_ylabel("coefficient value")
ax.set_title("Q2: MAP coefficients shrink to 0 as γ→0 and approach $w_{ML}$ as γ→∞")
for j in range(10):
    ax.plot(gammas[-1] * 1.6, wML[j], "k<", ms=5)
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("figs/q2_coeff_path.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
lo, med, hi = np.percentile(curves, [10, 50, 90], axis=0)
ax.fill_between(gammas, lo, hi, alpha=0.25, label="10–90% band")
ax.semilogx(gammas, med, lw=2, label="median MAP validation MSE")
ax.axhline(np.median(ml_vals), color="r", ls=":", label="median ML validation MSE")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("γ"); ax.set_ylabel("validation MSE"); ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
ax.set_title(f"Q2: {R} independent train/validation draws")
fig.tight_layout(); fig.savefig("figs/q2_mse_trials.png", dpi=160); plt.close(fig)
