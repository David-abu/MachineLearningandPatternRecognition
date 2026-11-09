"""Minimal 2-layer MLP for C-class posterior estimation, written in NumPy.

Architecture:  x (d) -> hidden layer of P softplus perceptrons -> C-way softmax.
Training:      maximum likelihood = minimum average cross-entropy, full-batch
               L-BFGS (scipy.optimize.minimize) with an analytic gradient.
"""
import numpy as np
from scipy.optimize import minimize


def softplus(a):
    # smooth ReLU: log(1 + e^a), computed stably
    return np.logaddexp(0.0, a)


def sigmoid(a):
    return 0.5 * (1.0 + np.tanh(0.5 * a))      # derivative of softplus, overflow-safe


def log_softmax(A):
    A = A - A.max(axis=1, keepdims=True)
    return A - np.log(np.exp(A).sum(axis=1, keepdims=True))


class MLP:
    def __init__(self, d, P, C):
        self.d, self.P, self.C = d, P, C
        self.shapes = [(P, d), (P,), (C, P), (C,)]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.n_params = sum(self.sizes)
        self.theta = None
        self.mu = np.zeros(d)
        self.sd = np.ones(d)

    # ---- parameter packing ----
    def unpack(self, theta):
        out, k = [], 0
        for s, n in zip(self.shapes, self.sizes):
            out.append(theta[k:k + n].reshape(s)); k += n
        return out

    def init(self, rng):
        # Glorot-style random initialization (a new draw for every restart)
        W1 = rng.normal(0, np.sqrt(2.0 / (self.d + self.P)), (self.P, self.d))
        b1 = rng.normal(0, 0.5, self.P)
        W2 = rng.normal(0, np.sqrt(2.0 / (self.P + self.C)), (self.C, self.P))
        b2 = np.zeros(self.C)
        return np.concatenate([W1.ravel(), b1, W2.ravel(), b2])

    # ---- forward pass ----
    def _forward(self, theta, X):
        W1, b1, W2, b2 = self.unpack(theta)
        A1 = X @ W1.T + b1               # N x P pre-activations
        H = softplus(A1)                 # N x P hidden outputs
        A2 = H @ W2.T + b2               # N x C logits
        return A1, H, log_softmax(A2)

    # ---- objective: average negative log-likelihood and its gradient ----
    def nll_grad(self, theta, X, Y):
        """X: N x d (standardized), Y: N x C one-hot."""
        W1, b1, W2, b2 = self.unpack(theta)
        A1, H, logp = self._forward(theta, X)
        N = X.shape[0]
        nll = -np.sum(Y * logp) / N
        G2 = (np.exp(logp) - Y) / N      # dNLL/dA2 for softmax + cross-entropy
        gW2 = G2.T @ H
        gb2 = G2.sum(axis=0)
        G1 = (G2 @ W2) * sigmoid(A1)     # back through softplus
        gW1 = G1.T @ X
        gb1 = G1.sum(axis=0)
        return nll, np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2])

    # ---- public API ----
    def fit(self, X, labels, restarts=5, rng=None, maxiter=3000):
        """ML training with `restarts` random initializations; keeps the solution
        with the highest training log-likelihood (lowest average NLL)."""
        rng = np.random.default_rng() if rng is None else rng
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0) + 1e-12   # standardize with training stats
        Xs = (X - self.mu) / self.sd
        Y = np.eye(self.C)[labels]
        best = None
        self.restart_log = []            # (final NLL, converged?, iterations) for every restart
        for _ in range(restarts):
            res = minimize(self.nll_grad, self.init(rng), args=(Xs, Y), jac=True,
                           method="L-BFGS-B", options={"maxiter": maxiter, "gtol": 1e-7})
            self.restart_log.append((float(res.fun), bool(res.success), int(res.nit)))
            if best is None or res.fun < best.fun:
                best = res
        self.theta = best.x
        self.train_nll = best.fun
        self.converged = bool(best.success)
        self.n_iter = int(best.nit)
        return self

    def predict_proba(self, X):
        _, _, logp = self._forward(self.theta, (X - self.mu) / self.sd)
        return np.exp(logp)

    def predict(self, X):
        # MAP rule with the MLP posterior estimates (0-1 loss -> argmax posterior)
        _, _, logp = self._forward(self.theta, (X - self.mu) / self.sd)
        return logp.argmax(axis=1)
