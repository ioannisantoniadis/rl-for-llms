"""Baselines, advantages and Monte Carlo KL estimators.

These are the small pieces of classical RL that survive into LLM post-training (Chapters 2, 3
and 8). Each function works on plain arrays of per-sample quantities, so it can be checked
against exact expectations computed on the testbed.
"""

from __future__ import annotations

import numpy as np

__all__ = ["advantages", "kl_estimator", "ppo_clip_grad_ratio", "ppo_clip_objective"]


def advantages(rewards: np.ndarray, baseline: str = "rloo", eps: float = 1e-8) -> np.ndarray:
    """Per-sample advantages for a ``(n_groups, G)`` array of rewards (G samples per prompt).

    ``baseline`` is one of:

    ``"none"``   ``A_i = r_i``. Plain REINFORCE: unbiased, highest variance.
    ``"mean"``   ``A_i = r_i - mean(r)``. Group mean *including* sample ``i``; the gradient is
                 biased by a factor ``(G-1)/G`` (the sample's own reward leaks into its baseline).
    ``"rloo"``   ``A_i = r_i - mean_{j != i} r_j``. Leave-one-out (Kool et al. 2019; Ahmadian et
                 al. 2024): unbiased, because the baseline is independent of sample ``i``.
                 Algebraically ``G/(G-1)`` times the ``"mean"`` advantage.
    ``"grpo"``   ``A_i = (r_i - mean(r)) / (std(r) + eps)``. GRPO's group normalization (Shao et
                 al. 2024): rescales each prompt's advantages by that prompt's reward spread.
                 A group with zero variance gets zero advantage (and so contributes no gradient).
    """
    r = np.asarray(rewards, dtype=float)
    G = r.shape[-1]
    mean = r.mean(axis=-1, keepdims=True)
    if baseline == "none":
        return r.copy()
    if baseline == "mean":
        return r - mean
    if baseline == "rloo":
        return (r - mean) * G / (G - 1)
    if baseline == "grpo":
        return (r - mean) / (r.std(axis=-1, keepdims=True) + eps)
    raise ValueError(f"unknown baseline {baseline!r}")


def kl_estimator(logp: np.ndarray, logq: np.ndarray, kind: str = "k3") -> np.ndarray:
    """Per-sample Monte Carlo estimators of ``KL(p || q)`` for samples drawn from ``p``.

    Following Schulman (2020), with ``r = q(x) / p(x)``:

    ``k1 = -log r``              unbiased; high variance; can be negative.
    ``k2 = (log r)^2 / 2``       biased (its mean is an f-divergence that matches KL to second
                                 order); low variance; always >= 0.
    ``k3 = (r - 1) - log r``     unbiased; low variance; always >= 0. GRPO's choice (Shao et al.
                                 2024), with ``p = pi_theta`` and ``q = pi_ref``.

    "Unbiased" is about the *value* under samples from ``p``. It is not a statement about the
    *gradient* of the estimator used as a loss (Chapter 3).
    """
    log_r = logq - logp
    if kind == "k1":
        return -log_r
    if kind == "k2":
        return 0.5 * log_r**2
    if kind == "k3":
        return np.expm1(log_r) - log_r
    raise ValueError(f"unknown KL estimator {kind!r}")


def ppo_clip_objective(ratio: np.ndarray, adv: np.ndarray, eps: float = 0.2, eps_high=None):
    """PPO's clipped surrogate ``min(rho A, clip(rho, 1-eps, 1+eps_high) A)`` per sample.

    ``eps_high`` defaults to ``eps`` (symmetric, Schulman et al. 2017); DAPO's "clip-higher"
    decouples it (Yu et al. 2025).
    """
    eps_high = eps if eps_high is None else eps_high
    return np.minimum(ratio * adv, np.clip(ratio, 1 - eps, 1 + eps_high) * adv)


def ppo_clip_grad_ratio(ratio: np.ndarray, adv: np.ndarray, eps: float = 0.2, eps_high=None):
    """Derivative of ``ppo_clip_objective`` with respect to the ratio (a.e.).

    It equals ``A`` where the unclipped term is the active one, and ``0`` where the clipped term
    is active and the clip binds: ``A > 0`` and ``rho > 1 + eps_high`` (no further reward for
    making an already-up-weighted good sample even likelier), or ``A < 0`` and ``rho < 1 - eps``
    (no further push down on an already-down-weighted bad sample). In the other two corners
    (``A > 0, rho < 1 - eps`` and ``A < 0, rho > 1 + eps``) the gradient is *not* clipped: the
    ``min`` picks the unclipped, more pessimistic term, so mistakes can always be undone.
    """
    eps_high = eps if eps_high is None else eps_high
    zero = ((adv > 0) & (ratio > 1 + eps_high)) | ((adv < 0) & (ratio < 1 - eps))
    return np.where(zero, 0.0, adv)


def gae(token_rewards: np.ndarray, values: np.ndarray, gamma: float = 1.0, lam: float = 1.0) -> np.ndarray:
    """Generalized advantage estimation (Schulman et al. 2016) for a batch of finite episodes.

    ``token_rewards`` is ``(n, T)``: the reward received after each action (for a sequence-level
    reward, zeros except the last column). ``values`` is ``(n, T + 1)``: the critic's estimate of
    each state, including the terminal one (whose value is 0 for a finished episode; this module
    lets the caller pass it). Returns ``(n, T)`` advantages

        A_t = sum_l (gamma lam)^l delta_{t+l},   delta_t = r_t + gamma V(s_{t+1}) - V(s_t).

    ``lam = 1`` gives the Monte Carlo advantage ``G_t - V(s_t)`` (unbiased for any baseline V, high
    variance); ``lam = 0`` gives the one-step TD error (low variance, biased unless V is exact).
    """
    r = np.asarray(token_rewards, dtype=float)
    v = np.asarray(values, dtype=float)
    delta = r + gamma * v[:, 1:] - v[:, :-1]
    adv = np.zeros_like(delta)
    running = np.zeros(r.shape[0])
    for t in range(r.shape[1] - 1, -1, -1):
        running = delta[:, t] + gamma * lam * running
        adv[:, t] = running
    return adv
