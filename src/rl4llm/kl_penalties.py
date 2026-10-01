"""Exact expected gradients of the ways LLM recipes implement a KL penalty (Chapter 3).

A KL penalty to pi_ref can be implemented in several ways that look interchangeable and are not:

``"k1_reward"``   subtract beta * log(pi/pi_ref) from each sample's reward and use the score
                  function (classic RLHF; Stiennon et al. 2020, Ouyang et al. 2022). The expected
                  gradient is the gradient of the *reverse* KL(pi || pi_ref).
``"k1_loss"``     add beta * mean(log pi - log pi_ref) to the loss and differentiate it directly.
                  Its expected gradient is E[grad log pi] = 0: no regularization at all.
``"k2_loss"``     add beta * mean(0.5 (log pi - log pi_ref)^2): gradient-equivalent to "k1_reward"
                  on-policy (Liu et al. 2025, Thm 5.1).
``"k3_loss"``     add beta * mean(pi_ref/pi - 1 - log(pi_ref/pi)), sequence level, differentiated
                  directly with the samples held fixed. Its expected gradient is
                  E_pi[(1 - pi_ref/pi) grad log pi] = -sum_y pi_ref(y) grad log pi(y), which is the
                  gradient of the *forward* KL(pi_ref || pi).
``"k3_loss_token"`` the same, per token and summed over positions (GRPO's implementation; Shao et
                  al. 2024): forward KL at each visited state, averaged over the policy's states.

Every variant is returned as a per-token weight on grad log pi(y_t | s_t), so it plugs into the same
``grad_log_likelihood`` as everything else, and the expectation is exact (all responses enumerated,
weighted by pi_theta). All gradients are of the penalty itself (to be *subtracted*, times beta).
"""

from __future__ import annotations

import numpy as np

from rl4llm.toy_language import grad_log_likelihood, sequence_logprobs, token_logprobs

KINDS = ("k1_reward", "k1_loss", "k2_loss", "k3_loss", "k3_loss_token")


def penalty_gradient(lang, logits, ref_logits, kind: str) -> np.ndarray:
    """Exact on-policy expected gradient of the KL penalty implemented as ``kind`` (mean over prompts)."""
    lp_tok = token_logprobs(lang, logits)  # (P, N, L)
    ref_tok = token_logprobs(lang, ref_logits)
    lp = lp_tok.sum(-1)
    ref = ref_tok.sum(-1)
    pi = np.exp(lp)  # sampling weights: on-policy
    log_ratio = lp - ref
    if kind in ("k1_reward", "k2_loss"):
        w = np.repeat(log_ratio[..., None], lang.L, axis=-1)
    elif kind == "k1_loss":
        w = np.ones_like(lp_tok)
    elif kind == "k3_loss":
        w = np.repeat((1.0 - np.exp(-log_ratio))[..., None], lang.L, axis=-1)
    elif kind == "k3_loss_token":
        w = 1.0 - np.exp(ref_tok - lp_tok)
    else:
        raise ValueError(kind)
    P, N = lp.shape
    w = (pi[..., None] * w).reshape(P * N, lang.L)
    prompts = np.repeat(np.arange(P), N)
    seqs = np.tile(np.arange(N), P)
    return grad_log_likelihood(lang, logits, prompts, seqs, w) / P


def reverse_kl(lang, logits, ref_logits) -> float:
    lp, ref = sequence_logprobs(lang, logits), sequence_logprobs(lang, ref_logits)
    return float((np.exp(lp) * (lp - ref)).sum(1).mean())


def forward_kl(lang, logits, ref_logits) -> float:
    lp, ref = sequence_logprobs(lang, logits), sequence_logprobs(lang, ref_logits)
    return float((np.exp(ref) * (ref - lp)).sum(1).mean())
