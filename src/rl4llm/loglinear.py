"""A sequence-level log-linear policy with *shared* features, for experiments where the prefix-tabular
policy is too flexible (Chapter 7: likelihood displacement).

pi_theta(y | x) is proportional to exp(theta_x . phi(y)), with phi the per-position token one-hot
features of rl4llm.reward_model.position_features. Two responses that share most tokens share most
features, so a gradient step on one moves the other: the regime in which Razin et al. (2025) show
that DPO can lower the likelihood of the preferred response.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logsumexp

from rl4llm.reward_model import position_features


class LogLinearPolicy:
    def __init__(self, lang):
        self.lang = lang
        self.F = position_features(lang)  # (N, d)
        self.theta = np.zeros((lang.n_prompts, self.F.shape[1]))

    def logprobs(self, theta=None) -> np.ndarray:
        th = self.theta if theta is None else theta
        lp = (self.F @ th.T).T
        return lp - logsumexp(lp, axis=1, keepdims=True)

    def fit_mle(self, target_logp, steps=3000, lr=0.5):
        """Maximum-likelihood (moment-matching) fit to a target distribution, e.g. pi_ref."""
        for _ in range(steps):
            g = (np.exp(target_logp) - np.exp(self.logprobs())) @ self.F
            self.theta += lr * g
        return self

    def dpo_step(self, ref_logp, beta, prompts, chosen, rejected, lr):
        """One gradient-ascent step on the mean DPO log-likelihood of the given pairs."""
        lp = self.logprobs()
        margin = beta * ((lp - ref_logp)[prompts, chosen] - (lp - ref_logp)[prompts, rejected])
        coef = beta * expit(-margin)
        g = np.zeros_like(self.theta)
        np.add.at(g, prompts, coef[:, None] * (self.F[chosen] - self.F[rejected]))
        self.theta += lr * g / len(prompts) * self.lang.n_prompts


def one_token_edits(lang, reward, n_per_prompt, rng):
    """Pairs (chosen correct, rejected = chosen with one token changed so that it is wrong)."""
    V, L = lang.V, lang.L
    out = []
    for p in range(lang.n_prompts):
        correct = np.flatnonzero(reward[p] > 0)
        for _ in range(n_per_prompt):
            yw = rng.choice(correct)
            s = lang.sequences[yw].copy()
            t = rng.integers(L)
            s[t] = (s[t] + rng.integers(1, V)) % V
            out.append((p, yw, int(s @ (V ** np.arange(L - 1, -1, -1)))))
    return np.array(out).T
