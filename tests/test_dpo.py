"""DPO's population minimizer is exactly pi* (Chapter 7)."""

import numpy as np

from rl4llm.methods import dpo
from rl4llm.methods._common import Adam
from rl4llm.metrics import kl
from rl4llm.toy_language import ToyLanguage, make_reference, optimal_logprobs, sequence_logprobs

BETA = 0.5


def _fit_population(lang, ref_logits, reward, pair_logp=None, steps=600):
    ref_logp = sequence_logprobs(lang, ref_logits)
    logits, opt = ref_logits.copy(), Adam(0.05)
    for _ in range(steps):
        g = dpo.population_gradient(lang, logits, ref_logp, reward, BETA, pair_logp)
        logits = opt.step(logits, g)
    return sequence_logprobs(lang, logits), optimal_logprobs(ref_logp, reward, BETA)


def test_population_dpo_recovers_optimum(lang, ref_logits, reward):
    learned, target = _fit_population(lang, ref_logits, reward)
    assert kl(learned, target).max() < 1e-6


def test_population_dpo_recovers_optimum_with_other_pair_distribution(rng):
    """The pair-sampling distribution does not matter, as long as it has full support:
    the implicit reward is identified up to a per-prompt constant, which normalization removes."""
    lang = ToyLanguage(V=3, L=3, n_prompts=2)
    ref_logits = make_reference(lang, seed=3)
    reward = rng.normal(size=(lang.n_prompts, lang.n_sequences))
    mu = rng.normal(size=(lang.n_prompts, lang.n_sequences))
    mu -= np.log(np.exp(mu).sum(axis=1, keepdims=True))
    learned, target = _fit_population(lang, ref_logits, reward, pair_logp=mu, steps=1500)
    assert kl(learned, target).max() < 1e-6


def test_dpo_gradient_vanishes_at_optimum(lang, ref_logits, ref_logp, reward):
    from rl4llm.toy_language import logits_from_sequence_logprobs

    at_opt = logits_from_sequence_logprobs(lang, optimal_logprobs(ref_logp, reward, BETA))
    g = dpo.population_gradient(lang, at_opt, ref_logp, reward, BETA)
    assert np.abs(g).max() < 1e-12
