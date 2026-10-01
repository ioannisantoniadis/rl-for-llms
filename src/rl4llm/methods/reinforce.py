"""REINFORCE with a choice of baseline (none / mean / leave-one-out / group-normalized).

Method card (Chapter 2 derives it; Chapter 8 places RLOO and GRPO-style normalization):

    Signal                scalar reward per response, minus a per-sample KL penalty
                          beta * log(pi_theta/pi_ref) ("KL in the reward", k1 estimator)
    Samples from (q)      pi_theta itself: G fresh samples per prompt each step (on-policy)
    Per-sample weight (w) advantage of the shaped reward: r - beta log(pi/pi_ref) - baseline
    Stability             none beyond the KL-shaped reward and a small learning rate
    Credit granularity    sequence level: one advantage broadcast to every token
    Models in memory      policy, reference (+ reward function)
    Targets               the KL-regularized objective; its optimum is pi* exactly
    Classical lineage     Williams (1992) REINFORCE with a Monte Carlo baseline;
                          "rloo" is Kool et al. (2019) / Ahmadian et al. (2024)

Why KL-in-the-reward gives an unbiased gradient: the score-function gradient of
E[-beta log(pi/pi_ref)] is E[-beta log(pi/pi_ref) grad log pi] - beta E[grad log pi], and the last
expectation is zero. So treating the per-sample log-ratio as part of the reward is exact.
"""

from __future__ import annotations

import numpy as np

from rl4llm.estimators import advantages
from rl4llm.methods._common import Adam, History
from rl4llm.toy_language import (
    grad_log_likelihood,
    optimal_logprobs,
    sample_sequences,
    sequence_logprobs,
)


def train(
    lang,
    ref_logits,
    reward,
    beta,
    steps=500,
    group_size=8,
    baseline="rloo",
    lr=0.05,
    seed=0,
    eval_every=10, lr_final_frac=1.0):
    rng = np.random.default_rng(seed)
    P, G = lang.n_prompts, group_size
    ref_logp = sequence_logprobs(lang, ref_logits)
    opt_logp = optimal_logprobs(ref_logp, reward, beta)
    logits, opt, hist = ref_logits.copy(), Adam(lr, total_steps=steps, final_frac=lr_final_frac), History()
    prompts = np.repeat(np.arange(P), G)
    for step in range(steps + 1):
        if step % eval_every == 0:
            hist.record(step, lang, logits, ref_logp, opt_logp, reward)
        if step == steps:
            break
        seqs = sample_sequences(lang, logits, rng, prompts)  # q = pi_theta (on-policy)
        logp = sequence_logprobs(lang, logits)
        shaped = reward[prompts, seqs] - beta * (logp[prompts, seqs] - ref_logp[prompts, seqs])
        w = advantages(shaped.reshape(P, G), baseline).ravel()  # w = advantage
        grad = grad_log_likelihood(lang, logits, prompts, seqs, w) / (P * G)
        logits = opt.step(logits, grad)
    return logits, hist
