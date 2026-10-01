"""GRPO (Shao et al. 2024) with switches for the 2025 modifications (Chapter 8).

Method card (as published):

    Signal                scalar reward per response (often verifiable)
    Samples from (q)      pi_old: G responses per prompt, reused for ``epochs`` updates
    Per-sample weight (w) group-normalized advantage (r_i - mean) / std, broadcast to every token,
                          times the clipped-ratio factor; per-sequence 1/|o_i| then mean over the group
    Stability             PPO-style clip; KL to pi_ref *in the loss* via the k3 estimator per token
    Credit granularity    sequence-level advantage broadcast to all tokens
    Models in memory      policy, reference (no critic; no reward model if verifiable)
    Targets               approximately the KL-regularized objective; with k3 as a loss the
                          regularizer is the forward KL (Chapter 3), so not exactly pi*
    Classical lineage     REINFORCE with a Monte Carlo (group) baseline + PPO's trust region

Switches:
    advantage     "grpo" (mean/std), "dr_grpo" (mean only; Liu et al. 2025), "rloo"
    kl            "k3_loss" (published), "k1_reward" (KL in the reward: targets pi* exactly), "none"
                  (DAPO drops it)
    aggregation   "seq_mean" (published: 1/|o_i| per sequence), "token_mean" (DAPO), "constant"
                  (Dr. GRPO: a fixed normalizer)
    eps_low/high  clip range; DAPO's clip-higher uses eps_high > eps_low
    dynamic_sampling  drop groups whose rewards are all equal (DAPO)
    kl_is_weight  multiply the k3 term by pi_theta/pi_old (Zhang et al.'s correction)
"""

from __future__ import annotations

import numpy as np

from rl4llm.estimators import advantages, ppo_clip_grad_ratio
from rl4llm.methods._common import Adam, History
from rl4llm.toy_language import (
    grad_log_likelihood,
    optimal_logprobs,
    sample_sequences,
    sequence_logprobs,
    token_logprobs,
)


def train(
    lang,
    ref_logits,
    reward,
    beta,
    steps=300,
    group_size=8,
    epochs=1,
    eps_low=0.2,
    eps_high=0.2,
    advantage="grpo",
    kl="k3_loss",
    aggregation="seq_mean",
    dynamic_sampling=False,
    kl_is_weight=False,
    lr=0.05,
    seed=0,
    eval_every=5,
    lengths_fn=None, lr_final_frac=1.0, callback=None):
    """``lengths_fn(seqs) -> (n,)`` gives response lengths for variable-length variants; tokens past
    a response's length must already carry zero weight in the caller's reward/mask conventions."""
    rng = np.random.default_rng(seed)
    P, G, L = lang.n_prompts, group_size, lang.L
    ref_logp = sequence_logprobs(lang, ref_logits)
    opt_logp = optimal_logprobs(ref_logp, reward, beta)
    ref_tok_all = token_logprobs(lang, ref_logits)
    logits, opt, hist = ref_logits.copy(), Adam(lr, total_steps=steps * epochs, final_frac=lr_final_frac), History()
    prompts = np.repeat(np.arange(P), G)
    for step in range(steps + 1):
        if step % eval_every == 0:
            hist.record(step, lang, logits, ref_logp, opt_logp, reward)
            if callback is not None:
                callback(step, logits)
        if step == steps:
            break
        seqs = sample_sequences(lang, logits, rng, prompts)
        old_tok = token_logprobs(lang, logits)[prompts, seqs]
        ref_tok = ref_tok_all[prompts, seqs]
        r = reward[prompts, seqs].astype(float)
        if kl == "k1_reward":
            r = r - beta * (old_tok - ref_tok).sum(1)
        A = advantages(r.reshape(P, G), "grpo" if advantage == "grpo" else ("mean" if advantage == "dr_grpo" else "rloo")).ravel()
        keep = np.ones(len(seqs), bool)
        if dynamic_sampling:
            keep = np.repeat(reward[prompts, seqs].reshape(P, G).std(1) > 0, G)
        lengths = np.full(len(seqs), L) if lengths_fn is None else lengths_fn(seqs)
        mask = (np.arange(L)[None, :] < lengths[:, None]).astype(float)
        if aggregation == "seq_mean":
            norm = mask / lengths[:, None] / max(keep.sum(), 1)
        elif aggregation == "token_mean":
            norm = mask / max((mask * keep[:, None]).sum(), 1)
        elif aggregation == "constant":
            norm = mask / (L * max(keep.sum(), 1))
        else:
            raise ValueError(aggregation)
        norm = norm * keep[:, None]
        for _ in range(epochs):
            new_tok = token_logprobs(lang, logits)[prompts, seqs]
            ratio = np.exp(new_tok - old_tok)
            A_t = np.repeat(A[:, None], L, 1)
            w = ppo_clip_grad_ratio(ratio, A_t, eps_low, eps_high) * ratio
            if kl == "k3_loss":
                coef = 1.0 - np.exp(ref_tok - new_tok)  # pathwise d k3 / d log pi
                if kl_is_weight:
                    k3 = np.exp(ref_tok - new_tok) - 1.0 - (ref_tok - new_tok)
                    coef = ratio * (k3 + coef)
                w = w - beta * coef
            logits = opt.step(logits, grad_log_likelihood(lang, logits, prompts, seqs, w * norm))
    return logits, hist
