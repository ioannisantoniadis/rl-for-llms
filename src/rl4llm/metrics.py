"""Exact metrics on the enumerable testbed: KL divergences, expected rewards, pass@k.

Everything here is an exact sum over the response space, never a Monte Carlo estimate. That is the
point of the testbed: an algorithm's *estimates* (sampled KL, sampled reward) can always be
compared with the *truth* computed here.
"""

from __future__ import annotations

import numpy as np

__all__ = ["entropy", "expected", "kl", "pass_at_k"]


def kl(logp: np.ndarray, logq: np.ndarray) -> np.ndarray:
    """Exact ``KL(p || q) = sum_y p(y) (log p(y) - log q(y))`` along the last axis.

    With ``p = pi_theta`` and ``q = pi_ref`` this is the *reverse* KL that appears in the
    KL-regularized objective; ``kl(logp_data, logp_model)`` is the *forward* KL that SFT minimizes.
    """
    p = np.exp(logp)
    # 0 * log 0 = 0: mask zero-probability entries so -inf - x does not produce nan.
    with np.errstate(invalid="ignore"):
        diff = np.where(p > 0, logp - logq, 0.0)
    return (p * diff).sum(axis=-1)


def expected(logp: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Exact ``E_{y~p}[values(y)]`` along the last axis."""
    return (np.exp(logp) * values).sum(axis=-1)


def entropy(logp: np.ndarray) -> np.ndarray:
    """Exact Shannon entropy (nats) along the last axis."""
    p = np.exp(logp)
    return -(p * np.where(p > 0, logp, 0.0)).sum(axis=-1)


def pass_at_k(logp: np.ndarray, correct: np.ndarray, k: int | np.ndarray) -> np.ndarray:
    """Exact pass@k: the probability that at least one of ``k`` i.i.d. samples is correct.

    ``pass@k = 1 - (1 - p_correct)^k`` with ``p_correct = sum_y pi(y) 1[correct(y)]``. (The
    unbiased estimator of Chen et al. 2021 is for when ``p_correct`` must itself be estimated from
    ``n >= k`` samples; here it is known exactly.)
    """
    p_correct = expected(logp, correct.astype(float))
    k = np.asarray(k)
    return 1.0 - (1.0 - p_correct[..., None]) ** k


def prefix_values(lang, logp: np.ndarray, reward: np.ndarray) -> np.ndarray:
    """Exact state values ``V^pi(s_t) = E_pi[r | x, y_<t]`` along every response.

    Returns a ``(P, N, L + 1)`` array: entry ``[x, i, t]`` is the expected final reward given the
    prompt and the first ``t`` tokens of response ``i`` (``t = 0`` is the prompt alone, ``V(x)``;
    ``t = L`` is the reward of the complete response). With a single terminal reward and
    ``gamma = 1`` this is the value function of the token-level MDP, the quantity a critic tries to
    learn. Computed by marginalizing the exact joint distribution over the unseen suffix.
    """
    P, V, L = logp.shape[0], lang.V, lang.L
    p = np.exp(logp).reshape((P,) + (V,) * L)
    pr = (np.exp(logp) * reward).reshape((P,) + (V,) * L)
    out = np.empty((P, lang.n_sequences, L + 1))
    for t in range(L + 1):
        axes = tuple(range(t + 1, L + 1))
        mass = p.sum(axis=axes) if axes else p
        val = pr.sum(axis=axes) if axes else pr
        v = np.where(mass > 0, val / np.where(mass > 0, mass, 1.0), 0.0).reshape(P, V**t)
        # Index of each response's length-t prefix in base V.
        idx = lang.sequences[:, :t] @ (V ** np.arange(t - 1, -1, -1)) if t > 0 else np.zeros(lang.n_sequences, int)
        out[:, :, t] = v[:, idx]
    return out
