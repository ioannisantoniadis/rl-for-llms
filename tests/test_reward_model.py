"""Best-of-n and the proxy reward model (Chapter 5)."""

import numpy as np

from rl4llm.metrics import kl
from rl4llm.reward_model import best_of_n_logprobs, fit_bradley_terry


def test_best_of_n_is_a_distribution_and_matches_sampling(lang, ref_logp, reward, rng):
    score = rng.normal(size=reward.shape)
    bon = best_of_n_logprobs(ref_logp, score, 4)
    assert np.allclose(np.exp(bon).sum(1), 1.0)
    p = 0
    draws = rng.choice(lang.n_sequences, size=(100_000, 4), p=np.exp(ref_logp[p]))
    best = draws[np.arange(len(draws)), np.argmax(score[p][draws], axis=1)]
    freq = np.bincount(best, minlength=lang.n_sequences) / len(best)
    assert np.abs(freq - np.exp(bon[p])).max() < 5e-3


def test_best_of_n_kl_bound(ref_logp, rng):
    """KL(best-of-n || pi_ref) <= log n - (n-1)/n, with near-equality when no response is likely
    (the formula is exact for continuous distributions; Gao et al. 2022 use it)."""
    score = rng.normal(size=ref_logp.shape)
    for n in (2, 4, 16, 64):
        k = kl(best_of_n_logprobs(ref_logp, score, n), ref_logp)
        bound = np.log(n) - (n - 1) / n
        assert np.all(k <= bound + 1e-9)


def test_proxy_reward_tracks_truth_near_reference(lang, ref_logp, reward):
    proxy = fit_bradley_terry(lang, ref_logp, reward, n_pairs=2000)
    pr = np.exp(ref_logp)
    for p in range(lang.n_prompts):
        # pi_ref-weighted correlation between proxy and true reward is clearly positive
        m1, m2 = (pr[p] * proxy[p]).sum(), (pr[p] * reward[p]).sum()
        cov = (pr[p] * (proxy[p] - m1) * (reward[p] - m2)).sum()
        assert cov > 0
