"""How a KL penalty is implemented changes which divergence it actually regularizes (Chapter 3)."""

import numpy as np

from rl4llm.kl_penalties import forward_kl, penalty_gradient, reverse_kl
from rl4llm.toy_language import ToyLanguage, make_reference


def _numerical_grad(f, logits, h=1e-6):
    g = np.zeros_like(logits)
    for idx in np.ndindex(logits.shape):
        d = np.zeros_like(logits)
        d[idx] = h
        g[idx] = (f(logits + d) - f(logits - d)) / (2 * h)
    return g


def _setup(rng):
    lang = ToyLanguage(V=3, L=3, n_prompts=2)
    ref = make_reference(lang, seed=5)
    logits = ref + rng.normal(scale=0.7, size=ref.shape)
    return lang, ref, logits


def test_k1_in_reward_and_k2_as_loss_follow_reverse_kl(rng):
    lang, ref, logits = _setup(rng)
    target = _numerical_grad(lambda z: reverse_kl(lang, z, ref), logits)
    for kind in ("k1_reward", "k2_loss"):
        assert np.allclose(penalty_gradient(lang, logits, ref, kind), target, atol=1e-6)


def test_k1_as_loss_has_zero_expected_gradient(rng):
    lang, ref, logits = _setup(rng)
    assert np.abs(penalty_gradient(lang, logits, ref, "k1_loss")).max() < 1e-12


def test_k3_as_loss_follows_forward_kl(rng):
    lang, ref, logits = _setup(rng)
    target = _numerical_grad(lambda z: forward_kl(lang, z, ref), logits)
    assert np.allclose(penalty_gradient(lang, logits, ref, "k3_loss"), target, atol=1e-6)
    rev = _numerical_grad(lambda z: reverse_kl(lang, z, ref), logits)
    assert not np.allclose(target, rev, atol=1e-3)  # and the two genuinely differ here
