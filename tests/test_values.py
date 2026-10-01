"""Exact value functions and GAE (Chapter 2)."""

import numpy as np

from rl4llm.estimators import gae
from rl4llm.metrics import expected, prefix_values


def test_prefix_values_endpoints(lang, ref_logp, reward):
    v = prefix_values(lang, ref_logp, reward)
    assert np.allclose(v[:, :, 0], expected(ref_logp, reward)[:, None])  # V(x) = E[r]
    assert np.allclose(v[:, :, -1], reward)  # V(complete response) = its reward


def test_prefix_values_satisfy_bellman(lang, ref_logits, ref_logp, reward):
    """V(s_t) = sum_a pi(a|s_t) V(s_t, a): the Bellman equation with gamma = 1, no step reward."""
    from rl4llm.toy_language import token_logprobs

    v = prefix_values(lang, ref_logp, reward)
    tok = np.exp(token_logprobs(lang, ref_logits))  # pi(y_t | s_t) for each response/position
    P, N, L = tok.shape
    for t in range(L):
        # Group responses by their length-t prefix; average V(s_{t+1}) weighted by pi(y_t | s_t)
        # over the distinct next tokens.
        pref = lang.prefix_index[:, t]
        for p in range(P):
            lhs = v[p, :, t]
            rhs = np.zeros(N)
            for k in np.unique(pref):
                m = pref == k
                # each distinct next token appears V**(L-t-1) times in the group; deduplicate
                _, first = np.unique(lang.sequences[m, t], return_index=True)
                idx = np.flatnonzero(m)[first]
                rhs[m] = (tok[p, idx, t] * v[p, idx, t + 1]).sum()
            assert np.allclose(lhs, rhs)


def test_gae_special_cases(rng):
    n, T = 6, 4
    r = np.zeros((n, T))
    r[:, -1] = rng.normal(size=n)
    v = np.concatenate([rng.normal(size=(n, T)), np.zeros((n, 1))], axis=1)
    mc = gae(r, v, lam=1.0)
    assert np.allclose(mc, r.sum(1, keepdims=True) - v[:, :-1])  # G_t - V(s_t), gamma = 1
    td = gae(r, v, lam=0.0)
    assert np.allclose(td, r + v[:, 1:] - v[:, :-1])
