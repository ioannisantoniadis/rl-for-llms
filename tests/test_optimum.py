"""The hub of the book (Chapter 6): the closed-form optimum of the KL-regularized objective."""

import numpy as np

from rl4llm.methods import exact
from rl4llm.methods.rft import rejection_sample
from rl4llm.metrics import expected, kl
from rl4llm.toy_language import (
    kl_regularized_objective,
    log_partition,
    optimal_logprobs,
    sequence_logprobs,
)

BETA = 0.5


def test_closed_form_matches_direct_optimization(lang, ref_logits, ref_logp, reward):
    """Gradient ascent on the exact objective (no closed form used) lands on pi*."""
    logits, _ = exact.train(lang, ref_logits, reward, BETA, steps=800, lr=0.05, eval_every=800)
    learned = sequence_logprobs(lang, logits)
    target = optimal_logprobs(ref_logp, reward, BETA)
    assert kl(learned, target).max() < 1e-6
    assert np.allclose(
        kl_regularized_objective(learned, ref_logp, reward, BETA),
        kl_regularized_objective(target, ref_logp, reward, BETA),
        atol=1e-6,
    )


def test_objective_is_reverse_kl_to_optimum(lang, ref_logp, reward, rng):
    """E_pi[r] - beta KL(pi||pi_ref) = beta log Z - beta KL(pi||pi*), for *any* policy pi."""
    target = optimal_logprobs(ref_logp, reward, BETA)
    logZ = log_partition(ref_logp, reward, BETA)
    for _ in range(5):
        logits = rng.normal(scale=2.0, size=lang.logits_shape)
        lp = sequence_logprobs(lang, logits)
        lhs = kl_regularized_objective(lp, ref_logp, reward, BETA)
        rhs = BETA * logZ - BETA * kl(lp, target)
        assert np.allclose(lhs, rhs)


def test_beta_limits(ref_logp, reward):
    """Large beta -> pi_ref; small beta -> all mass on the reward-maximizing responses."""
    assert kl(optimal_logprobs(ref_logp, reward, 1e4), ref_logp).max() < 1e-6
    sharp = optimal_logprobs(ref_logp, reward, 1e-3)
    assert expected(sharp, reward).min() > 1 - 1e-6


def test_rejection_sampling_draws_from_optimum(lang, ref_logits, ref_logp, reward, rng):
    """Accepting pi_ref samples with prob exp((r - r_max)/beta) yields exact samples of pi*."""
    prompts, seqs = rejection_sample(lang, ref_logits, reward, BETA, 200_000, rng)
    target = np.exp(optimal_logprobs(ref_logp, reward, BETA))
    for p in range(lang.n_prompts):
        freq = np.bincount(seqs[prompts == p], minlength=lang.n_sequences)
        freq = freq / freq.sum()
        assert np.abs(freq - target[p]).max() < 6e-3


def test_beta_zero_limit_is_reference_restricted_to_best(ref_logp, reward):
    lim = optimal_logprobs(ref_logp, reward, 0.0)
    assert np.allclose(np.exp(lim).sum(1), 1)
    assert np.allclose(kl(optimal_logprobs(ref_logp, reward, 1e-3), lim), 0, atol=1e-6)
