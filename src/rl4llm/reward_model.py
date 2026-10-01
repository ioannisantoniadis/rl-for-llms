"""A learned proxy reward on the toy language, and best-of-n sampling against it (Chapter 5).

The proxy is a Bradley-Terry reward model (Christiano et al. 2017; Ouyang et al. 2022) fit by
maximum likelihood to preference pairs. Pairs are sampled from pi_ref and labelled by a simulated
annotator who prefers y1 with probability sigmoid(sharpness * (r*(y1) - r*(y2))), where r* is the
true reward. The model is deliberately *misspecified*: it is linear in per-position token
features, so it cannot represent "the tokens sum to the target mod 5" exactly, and it is fit on
finite data. Near pi_ref it tracks the true reward; optimized hard, it is exploited. That is the
whole mechanism of reward-model overoptimization, in miniature, and every quantity is exact.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit


def position_features(lang) -> np.ndarray:
    """``(N, L*V)`` one-hot features: which token sits at each position."""
    N = lang.n_sequences
    F = np.zeros((N, lang.L * lang.V))
    for t in range(lang.L):
        F[np.arange(N), t * lang.V + lang.sequences[:, t]] = 1.0
    return F


def sum_features(lang) -> np.ndarray:
    """``(N, V)`` one-hot of the token sum mod V: a feature set that *can* express the true reward."""
    s = lang.sequences.sum(1) % lang.V
    return np.eye(lang.V)[s]


def fit_bradley_terry(lang, ref_logp, true_reward, n_pairs, sharpness=3.0, l2=1e-2, seed=0, features=None):
    """Fit a per-prompt linear Bradley-Terry reward model; returns the ``(P, N)`` proxy reward.

    ``features`` defaults to ``position_features`` (misspecified for the sum-mod-V reward);
    ``sum_features`` gives a model that can represent the true reward exactly.
    """
    rng = np.random.default_rng(seed)
    F = position_features(lang) if features is None else features
    proxy = np.zeros_like(true_reward, dtype=float)
    for p in range(lang.n_prompts):
        prob = np.exp(ref_logp[p])
        prob /= prob.sum()
        y1 = rng.choice(lang.n_sequences, n_pairs, p=prob)
        y2 = rng.choice(lang.n_sequences, n_pairs, p=prob)
        first = rng.random(n_pairs) < expit(sharpness * (true_reward[p, y1] - true_reward[p, y2]))
        d = np.where(first[:, None], F[y1] - F[y2], F[y2] - F[y1])  # chosen minus rejected

        def nll(w, d=d):
            return -log_expit(d @ w).mean() + l2 * w @ w

        def grad(w, d=d):
            return -(expit(-(d @ w))[:, None] * d).mean(0) + 2 * l2 * w

        w = minimize(nll, np.zeros(F.shape[1]), jac=grad, method="L-BFGS-B").x
        proxy[p] = F @ w
    return proxy


def best_of_n_logprobs(ref_logp: np.ndarray, score: np.ndarray, n: int) -> np.ndarray:
    """Exact log-distribution of the best of ``n`` i.i.d. samples from pi_ref, ranked by ``score``.

    With responses sorted by score and F_k the cumulative pi_ref mass up to the k-th,
    P(best = y_k) = F_k^n - F_{k-1}^n (ties broken by index; the proxy here has no exact ties).
    """
    out = np.empty_like(ref_logp)
    for p in range(ref_logp.shape[0]):
        order = np.argsort(score[p], kind="stable")
        prob = np.exp(ref_logp[p, order])
        F = np.cumsum(prob)
        F_prev = np.concatenate([[0.0], F[:-1]])
        pb = np.clip(F, 0, 1) ** n - np.clip(F_prev, 0, 1) ** n
        res = np.empty_like(pb)
        res[order] = pb
        out[p] = np.log(np.maximum(res, 1e-300))
    return out
