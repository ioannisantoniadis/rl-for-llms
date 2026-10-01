"""Bias and variance of policy-gradient estimators against the exact gradient (Chapters 2 and 8).

Shared by the Chapter 2 and Chapter 8 figures, so both use one harness. Everything is measured at a
fixed policy against the exact gradient of J(theta) = mean_x E_{y~pi}[r(x, y)], computed by
enumeration.
"""

from __future__ import annotations

import numpy as np

from rl4llm.estimators import advantages, gae
from rl4llm.metrics import expected, prefix_values
from rl4llm.toy_language import grad_log_likelihood, sample_sequences, sequence_logprobs

SEQUENCE_ESTIMATORS = ("none", "mean", "rloo", "grpo", "oracle_V(x)", "critic_V(s_t)")


def exact_gradient(lang, logits, reward):
    lp = sequence_logprobs(lang, logits)
    P, N = lp.shape
    w = (np.exp(lp) * reward).ravel()
    return grad_log_likelihood(lang, logits, np.repeat(np.arange(P), N), np.tile(np.arange(N), P), w) / P


def estimate(lang, logits, reward, rng, G, kind, values=None, lam=1.0):
    """One stochastic gradient estimate from G samples per prompt."""
    P = lang.n_prompts
    prompts = np.repeat(np.arange(P), G)
    ys = sample_sequences(lang, logits, rng, prompts)
    r = reward[prompts, ys]
    if kind in ("none", "mean", "rloo", "grpo"):
        w = advantages(r.reshape(P, G), kind).ravel()
    elif kind == "oracle_V(x)":
        lp = sequence_logprobs(lang, logits)
        w = r - expected(lp, reward)[prompts]
    elif kind in ("critic_V(s_t)", "gae"):
        # Token-level advantages from a critic. ``values`` is (P, N, L+1) critic estimates.
        v = values[prompts, ys]  # (n, L+1)
        v = v.copy()
        v[:, -1] = 0.0  # terminal state after the last token has value 0; reward arrives there
        tr = np.zeros((len(ys), lang.L))
        tr[:, -1] = r
        w = gae(tr, v, gamma=1.0, lam=lam if kind == "gae" else 1.0)
    else:
        raise ValueError(kind)
    return grad_log_likelihood(lang, logits, prompts, ys, w) / (P * G)


def bias_variance(lang, logits, reward, G, kind, n_draws=2000, seed=0, values=None, lam=1.0):
    """Relative squared bias, relative variance and cosine(mean estimate, exact gradient)."""
    rng = np.random.default_rng(seed)
    g = exact_gradient(lang, logits, reward).ravel()
    est = np.array([estimate(lang, logits, reward, rng, G, kind, values, lam).ravel() for _ in range(n_draws)])
    m = est.mean(0)
    gg = g @ g
    bias2 = ((m - g) @ (m - g)) / gg
    var = ((est - m) ** 2).sum(1).mean() / gg
    cos = (m @ g) / np.sqrt((m @ m) * gg)
    return bias2, var, cos


def exact_values(lang, logits, reward):
    return prefix_values(lang, sequence_logprobs(lang, logits), reward)


def effective_prompt_weights(lang, logits, reward, G, kind, n_draws=3000, seed=0):
    """Per-prompt effective weight of an estimator: the projection of its mean estimate onto the exact
    per-prompt gradient, divided by that gradient's squared norm. An unbiased estimator gives 1 for
    every prompt; a constant rescaling c gives c; GRPO's std normalization gives a weight that depends
    on the prompt's difficulty. (Parameters of different prompts are disjoint in the tabular policy,
    so each prompt's slice of the gradient can be read off separately.)"""
    rng = np.random.default_rng(seed)
    g = exact_gradient(lang, logits, reward)
    m = np.mean([estimate(lang, logits, reward, rng, G, kind) for _ in range(n_draws)], axis=0)
    w = []
    for p in range(lang.n_prompts):
        gp, mp = g[p].ravel(), m[p].ravel()
        w.append((mp @ gp) / (gp @ gp))
    return np.array(w)
