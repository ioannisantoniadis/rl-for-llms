"""Exact gradient ascent on the KL-regularized objective: the reference curve.

Not a practical method (it needs a sum over every possible response), but on the toy testbed it
is the *true* policy gradient, so it is the yardstick that sampled methods are compared against.

    J(theta) = E_{x} [ E_{y~pi_theta}[r(x,y)] - beta KL(pi_theta(.|x) || pi_ref(.|x)) ]

    grad J = E_x sum_y pi_theta(y|x) [ r(x,y) - beta log(pi_theta(y|x)/pi_ref(y|x)) ] grad log pi_theta(y|x)

(the "-beta" term that would multiply ``sum_y pi grad log pi`` vanishes, because that sum is the
gradient of ``sum_y pi = 1``). In the book's (q, w) form: q is pi_theta itself, enumerated rather
than sampled, and w is pi_theta(y|x) times the KL-shaped reward.
"""

from __future__ import annotations

import numpy as np

from rl4llm.methods._common import Adam, History
from rl4llm.toy_language import (
    ToyLanguage,
    grad_log_likelihood,
    optimal_logprobs,
    sequence_logprobs,
)


def exact_gradient(lang: ToyLanguage, logits, ref_logp, reward, beta) -> np.ndarray:
    """Exact gradient of J (averaged over prompts) with respect to the logits."""
    logp = sequence_logprobs(lang, logits)
    w = np.exp(logp) * (reward - beta * (logp - ref_logp))  # (P, N)
    P, N = w.shape
    prompts = np.repeat(np.arange(P), N)
    seqs = np.tile(np.arange(N), P)
    return grad_log_likelihood(lang, logits, prompts, seqs, w.ravel()) / P


def train(lang, ref_logits, reward, beta, steps=500, lr=0.05, eval_every=10, lr_final_frac=1.0):
    ref_logp = sequence_logprobs(lang, ref_logits)
    opt_logp = optimal_logprobs(ref_logp, reward, beta)
    logits, opt, hist = ref_logits.copy(), Adam(lr, total_steps=steps, final_frac=lr_final_frac), History()
    for step in range(steps + 1):
        if step % eval_every == 0:
            hist.record(step, lang, logits, ref_logp, opt_logp, reward)
        if step < steps:
            logits = opt.step(logits, exact_gradient(lang, logits, ref_logp, reward, beta))
    return logits, hist
