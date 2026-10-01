"""The testbed's bookkeeping: the policy is a valid distribution and its gradient is right."""

import numpy as np

from rl4llm.toy_language import (
    ToyLanguage,
    grad_log_likelihood,
    logits_from_sequence_logprobs,
    sample_sequences,
    sequence_logprobs,
)


def test_sizes(lang):
    assert lang.n_sequences == 625
    assert lang.n_prefixes == 1 + 5 + 25 + 125
    assert len(np.unique(lang.sequences, axis=0)) == 625


def test_policy_normalizes(lang, ref_logits, rng):
    for logits in (ref_logits, rng.normal(size=lang.logits_shape)):
        assert np.allclose(np.exp(sequence_logprobs(lang, logits)).sum(axis=1), 1.0)


def test_logits_round_trip(lang, rng):
    target = rng.normal(size=(lang.n_prompts, lang.n_sequences))
    target -= np.log(np.exp(target).sum(axis=1, keepdims=True))
    back = sequence_logprobs(lang, logits_from_sequence_logprobs(lang, target))
    assert np.allclose(back, target, atol=1e-12)


def test_grad_log_likelihood_matches_finite_differences(rng):
    lang = ToyLanguage(V=3, L=3, n_prompts=2)
    logits = rng.normal(size=lang.logits_shape)
    prompts = np.array([0, 1, 1, 0])
    seqs = np.array([3, 7, 20, 26])
    w = rng.normal(size=4)

    def f(z):
        return (w * sequence_logprobs(lang, z)[prompts, seqs]).sum()

    g = grad_log_likelihood(lang, logits, prompts, seqs, w)
    num = np.zeros_like(logits)
    h = 1e-6
    for idx in np.ndindex(logits.shape):
        d = np.zeros_like(logits)
        d[idx] = h
        num[idx] = (f(logits + d) - f(logits - d)) / (2 * h)
    assert np.allclose(g, num, atol=1e-6)


def test_token_level_weights_reduce_to_sequence_weights(lang, ref_logits, rng):
    prompts = rng.integers(0, lang.n_prompts, size=10)
    seqs = rng.integers(0, lang.n_sequences, size=10)
    w = rng.normal(size=10)
    a = grad_log_likelihood(lang, ref_logits, prompts, seqs, w)
    b = grad_log_likelihood(lang, ref_logits, prompts, seqs, np.repeat(w[:, None], lang.L, 1))
    assert np.allclose(a, b)


def test_sampler_matches_distribution(lang, ref_logits, ref_logp, rng):
    prompts = np.zeros(200_000, dtype=int)
    seqs = sample_sequences(lang, ref_logits, rng, prompts)
    freq = np.bincount(seqs, minlength=lang.n_sequences) / len(seqs)
    assert np.abs(freq - np.exp(ref_logp[0])).max() < 5e-3
