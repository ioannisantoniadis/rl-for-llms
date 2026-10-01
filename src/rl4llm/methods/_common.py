"""Shared training plumbing: an Adam optimizer and an exact evaluation history.

Every method in this package updates the same prefix-tabular logits with the same optimizer, so
differences between methods in an experiment come from the *method* (its sampling distribution
``q`` and its weights ``w``), not from optimizer details.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rl4llm.metrics import expected, kl
from rl4llm.toy_language import ToyLanguage, sequence_logprobs


class Adam:
    """Plain Adam (Kingma & Ba 2015) on a single array, for *ascent* on an objective.

    Optional cosine learning-rate decay from ``lr`` to ``lr * final_frac`` over ``total_steps``
    calls (off by default). Experiments that compare methods use the same schedule for all of them.
    """

    def __init__(self, lr: float = 0.05, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8,
                 total_steps: int | None = None, final_frac: float = 1.0):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.total_steps, self.final_frac = total_steps, final_frac
        self.m = self.v = None
        self.t = 0

    def current_lr(self) -> float:
        if not self.total_steps:
            return self.lr
        frac = min(self.t / self.total_steps, 1.0)
        return self.lr * (self.final_frac + (1 - self.final_frac) * 0.5 * (1 + np.cos(np.pi * frac)))

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m, self.v = np.zeros_like(params), np.zeros_like(params)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad**2
        m_hat = self.m / (1 - self.b1**self.t)
        v_hat = self.v / (1 - self.b2**self.t)
        return params + self.current_lr() * m_hat / (np.sqrt(v_hat) + self.eps)


@dataclass
class History:
    """Exact diagnostics recorded during training (all averaged over prompts)."""

    step: list = field(default_factory=list)
    kl_to_opt: list = field(default_factory=list)  # KL(pi_theta || pi*): the headline metric
    kl_to_ref: list = field(default_factory=list)  # KL(pi_theta || pi_ref)
    reward: list = field(default_factory=list)  # E_{pi_theta}[r]

    def record(self, step, lang: ToyLanguage, logits, ref_logp, opt_logp, reward) -> None:
        logp = sequence_logprobs(lang, logits)
        self.step.append(step)
        self.kl_to_opt.append(float(kl(logp, opt_logp).mean()))
        self.kl_to_ref.append(float(kl(logp, ref_logp).mean()))
        self.reward.append(float(expected(logp, reward).mean()))
