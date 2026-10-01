"""Baselines, KL estimators and the PPO clip (Chapters 2, 3 and 8), checked exactly."""

import numpy as np

from rl4llm.estimators import kl_estimator, ppo_clip_grad_ratio, ppo_clip_objective
from rl4llm.metrics import kl
from rl4llm.toy_language import grad_log_likelihood, sequence_logprobs


def _exact_pg(lang, logits, values):
    """sum_y pi(y|x) values(x,y) grad log pi(y|x), summed over prompts."""
    lp = sequence_logprobs(lang, logits)
    P, N = lp.shape
    w = (np.exp(lp) * values).ravel()
    return grad_log_likelihood(lang, logits, np.repeat(np.arange(P), N), np.tile(np.arange(N), P), w)


def test_score_function_has_zero_mean(lang, rng):
    logits = rng.normal(size=lang.logits_shape)
    g = _exact_pg(lang, logits, np.ones((lang.n_prompts, lang.n_sequences)))
    assert np.abs(g).max() < 1e-12


def test_baseline_does_not_change_expected_gradient(lang, reward, rng):
    logits = rng.normal(size=lang.logits_shape)
    g = _exact_pg(lang, logits, reward)
    per_prompt_b = rng.normal(size=(lang.n_prompts, 1))
    for b in (0.37, per_prompt_b):
        assert np.allclose(_exact_pg(lang, logits, reward - b), g, atol=1e-12)


def test_kl_estimators_bias_exact(lang, ref_logp, rng):
    """k1 and k3 are unbiased for KL(pi || pi_ref) under samples from pi; k2 is not."""
    logits = rng.normal(size=lang.logits_shape)
    lp = sequence_logprobs(lang, logits)
    p = np.exp(lp)
    true = kl(lp, ref_logp)
    for kind, unbiased in (("k1", True), ("k2", False), ("k3", True)):
        mean = (p * kl_estimator(lp, ref_logp, kind)).sum(axis=1)
        assert np.allclose(mean, true) == unbiased
    assert (kl_estimator(lp, ref_logp, "k3") >= -1e-12).all()


def test_ppo_clip_gradient_zero_exactly_where_claimed():
    eps = 0.2
    rho = np.linspace(0.5, 1.5, 1001)
    h = 1e-7
    for A in (1.3, -0.7):
        adv = np.full_like(rho, A)
        num = (ppo_clip_objective(rho + h, adv, eps) - ppo_clip_objective(rho - h, adv, eps)) / (2 * h)
        ana = ppo_clip_grad_ratio(rho, adv, eps)
        away = np.abs(np.abs(rho - 1) - eps) > 1e-3  # skip the kinks themselves
        assert np.allclose(num[away], ana[away], atol=1e-5)
        if A > 0:
            assert np.all(ana[rho > 1 + eps + 1e-3] == 0) and np.all(ana[rho < 1 - eps] == A)
        else:
            assert np.all(ana[rho < 1 - eps - 1e-3] == 0) and np.all(ana[rho > 1 + eps] == A)
