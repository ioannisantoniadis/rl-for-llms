"""On-policy distillation (Agarwal et al. 2024, GKD; Lu & Thinking Machines Lab 2025).

Method card (Chapter 9):

    Signal                the teacher's per-token log-probabilities
    Samples from (q)      the student pi_theta itself (on-policy)
    Per-sample weight (w) per token: -(log pi_theta(y_t|s_t) - log pi_T(y_t|s_t)), the negative
                          per-token reverse KL, used directly as the advantage with a discount of zero
                          (as in the Thinking Machines recipe; ``reward_to_go=True`` sums future terms)
    Stability             none needed beyond a small learning rate: the signal is dense
    Credit granularity    token level, exactly where the student departs from the teacher
    Models in memory      student, teacher
    Targets               the teacher distribution (reverse KL); here the teacher is pi* so the
                          target coincides with the other methods'
    Classical lineage     DAgger (the expert labels states the learner visits) with a KL loss;
                          policy gradient with a dense, shaped reward
"""

from __future__ import annotations

import numpy as np

from rl4llm.methods._common import Adam, History
from rl4llm.toy_language import (
    grad_log_likelihood,
    sample_sequences,
    sequence_logprobs,
    token_logprobs,
)


def train(lang, ref_logits, teacher_logits, reward, steps=300, group_size=8, lr=0.05, seed=0,
          eval_every=5, reward_to_go=False, lr_final_frac=1.0):
    rng = np.random.default_rng(seed)
    P, G = lang.n_prompts, group_size
    ref_logp = sequence_logprobs(lang, ref_logits)
    teacher_logp = sequence_logprobs(lang, teacher_logits)
    teacher_tok_all = token_logprobs(lang, teacher_logits)
    logits, opt, hist = ref_logits.copy(), Adam(lr, total_steps=steps, final_frac=lr_final_frac), History()
    prompts = np.repeat(np.arange(P), G)
    for step in range(steps + 1):
        if step % eval_every == 0:
            hist.record(step, lang, logits, ref_logp, teacher_logp, reward)
        if step == steps:
            break
        seqs = sample_sequences(lang, logits, rng, prompts)
        stu = token_logprobs(lang, logits)[prompts, seqs]
        adv = -(stu - teacher_tok_all[prompts, seqs])  # negative per-token reverse KL
        if reward_to_go:
            adv = np.cumsum(adv[:, ::-1], axis=1)[:, ::-1]
        g = grad_log_likelihood(lang, logits, prompts, seqs, adv) / (len(seqs) * lang.L)
        logits = opt.step(logits, g)
    return logits, hist


def train_offpolicy(lang, ref_logits, teacher_logits, reward, steps=300, group_size=8, lr=0.05,
                    seed=0, eval_every=5, lr_final_frac=1.0):
    """Off-policy (sequence-level) distillation: SFT on fresh samples drawn from the *teacher*.

    Method card: signal = teacher samples; q = pi_T; w = 1; minimizes the forward KL(pi_T || pi_theta)
    (mass-covering); lineage = behavior cloning of the teacher.
    """
    rng = np.random.default_rng(seed)
    P, G = lang.n_prompts, group_size
    ref_logp = sequence_logprobs(lang, ref_logits)
    teacher_logp = sequence_logprobs(lang, teacher_logits)
    logits, opt, hist = ref_logits.copy(), Adam(lr, total_steps=steps, final_frac=lr_final_frac), History()
    prompts = np.repeat(np.arange(P), G)
    for step in range(steps + 1):
        if step % eval_every == 0:
            hist.record(step, lang, logits, ref_logp, teacher_logp, reward)
        if step == steps:
            break
        seqs = sample_sequences(lang, teacher_logits, rng, prompts)
        g = grad_log_likelihood(lang, logits, prompts, seqs, np.ones(len(seqs))) / len(seqs)
        logits = opt.step(logits, g)
    return logits, hist
