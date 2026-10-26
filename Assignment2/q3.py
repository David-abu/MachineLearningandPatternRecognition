"""EECE5644 Assignment 2 - Question 3
MAP vehicle localization from K noisy range measurements to landmarks on the unit circle.
Objective (to minimize):  J(x,y) = sum_i (r_i - d_i(x,y))^2 / sigma_i^2 + x^2/sx^2 + y^2/sy^2
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("figs", exist_ok=True)
rng = np.random.default_rng(5644)

sigma = 0.3          # range-noise std (all landmarks)
sx = sy = 0.25       # prior std about the origin

# true position: uniform inside the unit circle
rad, ang = np.sqrt(rng.random()), 2 * np.pi * rng.random()
pT = np.array([rad * np.cos(ang), rad * np.sin(ang)])
print(f"True position = ({pT[0]:.4f}, {pT[1]:.4f})")


def landmarks(K):
    th = 2 * np.pi * np.arange(K) / K
    return np.column_stack([np.cos(th), np.sin(th)])


def measure(L, p):
    d = np.linalg.norm(L - p, axis=1)
    r = d + sigma * rng.standard_normal(len(d))
    while np.any(r < 0):                       # reject and resample negative ranges
        bad = r < 0
        r[bad] = d[bad] + sigma * rng.standard_normal(bad.sum())
    return r


def J(X, Y, L, r):
    val = X**2 / sx**2 + Y**2 / sy**2
    for (lx, ly), ri in zip(L, r):
        val = val + (ri - np.hypot(X - lx, Y - ly)) ** 2 / sigma**2
    return val


g = np.linspace(-2, 2, 801)
X, Y = np.meshgrid(g, g)
levels = [0.5, 1, 2, 4, 8, 16, 32, 64, 128]   # identical contour levels for every K

fig, axes = plt.subplots(2, 2, figsize=(11, 10.5))
for K, ax in zip((1, 2, 3, 4), axes.ravel()):
    L = landmarks(K)
    r = measure(L, pT)
    Z = J(X, Y, L, r)
    i = np.unravel_index(np.argmin(Z), Z.shape)
    pMAP = np.array([X[i], Y[i]])
    # area of J <= Jmin + 2: objective-based uncertainty proxy (~63% mass under a local 2-D Gaussian approx.)
    area = np.sum(Z <= Z.min() + 2) * (g[1] - g[0]) ** 2
    print(f"K={K}: ranges={np.round(r,3)}  MAP≈({pMAP[0]:.3f},{pMAP[1]:.3f})  "
          f"|MAP-true|={np.linalg.norm(pMAP-pT):.3f}  Jmin={Z.min():.3f}  area(J<=Jmin+2)={area:.4f}")
    cs = ax.contour(X, Y, Z, levels=levels, cmap="viridis")
    ax.clabel(cs, fontsize=7, fmt="%g")
    ax.contour(X, Y, Z, levels=[Z.min() + 2], colors="r", linewidths=1, linestyles="--")
    ax.plot(*L.T, "o", mfc="none", mec="k", ms=10, mew=2, label="landmarks")
    ax.plot(*pT, "+", c="r", ms=16, mew=2.5, label="true position")
    ax.plot(*pMAP, "x", c="b", ms=10, mew=2, label="MAP (grid argmin)")
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, ls=":", color="gray"))
    ax.set_title(f"K = {K}:  |MAP − true| = {np.linalg.norm(pMAP-pT):.3f}")
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
axes[0, 0].legend(fontsize=8, loc="lower left")
fig.suptitle(f"Q3: MAP objective contours (same levels {levels[0]}…{levels[-1]} in every panel), "
             f"σ=0.3, σx=σy=0.25\nred dashed = J ≤ Jmin+2 region", fontsize=11)
fig.tight_layout(); fig.savefig("figs/q3_contours.png", dpi=150); plt.close(fig)

# Monte Carlo over many noise draws (and random true positions) to back up the visual trend
T = 500
gc = np.linspace(-1.5, 1.5, 241)
Xc, Yc = np.meshgrid(gc, gc)
err = np.zeros((T, 4)); area = np.zeros((T, 4))
for t in range(T):
    rr, aa = np.sqrt(rng.random()), 2 * np.pi * rng.random()
    p = np.array([rr * np.cos(aa), rr * np.sin(aa)])
    for k, K in enumerate((1, 2, 3, 4)):
        L = landmarks(K); r = measure(L, p)
        Z = J(Xc, Yc, L, r)
        i = np.unravel_index(np.argmin(Z), Z.shape)
        err[t, k] = np.hypot(Xc[i] - p[0], Yc[i] - p[1])
        area[t, k] = np.sum(Z <= Z.min() + 2) * (gc[1] - gc[0]) ** 2
print(f"\nMonte Carlo ({T} random true positions & noise draws):")
for k in range(4):
    print(f"  K={k+1}: mean |MAP-true| = {err[:,k].mean():.3f}  median = {np.median(err[:,k]):.3f}  "
          f"mean area(J<=Jmin+2) = {area[:,k].mean():.4f}")
