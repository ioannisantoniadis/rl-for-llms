"""Proper scoring rules as rewards (Chapter 10)."""

import numpy as np

from rl4llm.calibration import BUCKETS, CalibrationTask, exact_optimum


def test_brier_reward_is_proper():
    """For a fixed success probability q, E[-(p - Y)^2] is maximized at p = q."""
    p = np.linspace(0, 1, 1001)
    for q in (0.0, 0.13, 0.5, 0.8, 1.0):
        val = -(q * (1 - p) ** 2 + (1 - q) * p**2)
        assert abs(p[np.argmax(val)] - q) < 1e-3


def test_optima_of_the_three_rewards():
    task = CalibrationTask()
    q_best = task.q.max(axis=1)
    for kind in ("binary", "combined"):
        opt = exact_optimum(task, kind)
        assert np.allclose(task.q[np.arange(4), opt], q_best)  # both pick the best answer ...
    opt = exact_optimum(task, "combined")
    nearest = BUCKETS[np.abs(BUCKETS[None, :] - q_best[:, None]).argmin(1)]
    assert np.allclose(task.conf[opt], nearest)  # ... combined also states its probability honestly
    opt = exact_optimum(task, "brier")
    assert task.q[np.arange(4), opt].min() < 0.05  # Brier alone prefers a surely-wrong answer somewhere


def test_mixture_brier_is_rlvr_plus_gini(rng):
    """Gao & Wang (2026), Prop. 1 / Cor. 1, checked numerically: with J(u, q) = 2 u.q - |u|^2, the
    score of the mixture p = E[u] equals the mean per-sample score plus E|u - p|^2; with one-hot
    sampled answers that is (2 P(correct) - 1) + (1 - |p|^2)."""
    K = 6
    q = rng.dirichlet(np.ones(K))
    us = rng.dirichlet(np.ones(K), size=50)
    p = us.mean(0)

    def J(u):
        return 2 * u @ q - u @ u

    assert np.isclose(J(p), np.mean([J(u) for u in us]) + np.mean(((us - p) ** 2).sum(1)))
    rlvr = 2 * p @ q - 1  # E_{A~p} J(e_A, q)
    gini = 1 - p @ p
    assert np.isclose(J(p), rlvr + gini)
