"""IPO's distinct target and DPO's likelihood displacement with shared features (Chapter 7)."""

import numpy as np

from rl4llm.loglinear import LogLinearPolicy, one_token_edits
from rl4llm.methods import dpo
from rl4llm.metrics import kl
from rl4llm.toy_language import optimal_logprobs


def test_ipo_target_differs_from_dpo_target(ref_logp, reward):
    ipo = dpo.ipo_target_logprobs(ref_logp, reward, 0.5)
    assert np.allclose(np.exp(ipo).sum(1), 1)
    assert kl(ipo, optimal_logprobs(ref_logp, reward, 0.5)).mean() > 0.05


def test_dpo_often_displaces_preferred_likelihood_with_shared_features(lang, ref_logp, reward):
    pol0 = LogLinearPolicy(lang).fit_mle(ref_logp)
    lin_ref = pol0.logprobs()
    d_chosen, margins = [], []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        pol = LogLinearPolicy(lang)
        pol.theta = pol0.theta.copy()
        pr, yw, yl = one_token_edits(lang, reward, 100, rng)
        before = pol.logprobs()
        for _ in range(600):
            pol.dpo_step(lin_ref, 0.5, pr, yw, yl, lr=0.2)
        after = pol.logprobs()
        dc = (after[pr, yw] - before[pr, yw]).mean()
        dr = (after[pr, yl] - before[pr, yl]).mean()
        d_chosen.append(dc)
        margins.append(dc - dr)
    assert np.median(d_chosen) < 0  # preferred responses usually become less likely ...
    assert min(margins) > 0  # ... while the margin always grows
