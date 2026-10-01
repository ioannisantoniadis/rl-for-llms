"""Answer-plus-confidence task for rewarding calibration (Chapter 10).

This is the book's first-principles reconstruction of "RL for calibrated decisions", **not** any
vendor's algorithm. A response is an answer (the first three tokens of the toy language) followed by
a stated confidence (the last token: bucket c in {0..4} means probability (c + 0.5) / 5, i.e. 0.1,
0.3, 0.5, 0.7, 0.9). Whether an answer is correct is random: answer a to prompt x succeeds with a
hidden probability q_x(a). That randomness is the uncertainty the model cannot resolve, so the
honest confidence for answer a is q_x(a).

Rewards (Y ~ Bernoulli(q_x(a)) is the outcome, p the stated confidence):

``"binary"``   Y                       correctness only (RLVR-style); confidence is ignored
``"brier"``    -(p - Y)^2              a proper scoring rule alone; ignores whether Y = 1
``"combined"`` Y - (p - Y)^2           correctness + Brier, the form Damani et al. (2025) use

For a fixed answer, E[-(p - Y)^2] = -(q(1-p)^2 + (1-q)p^2) is maximized at p = q (propriety). With
p = q, E[combined] = q - q(1 - q) = q^2, increasing in q: the best answer *and* an honest confidence.
E[brier] = -q(1 - q) is maximized at q in {0, 1}: Brier alone rewards answers whose outcome is
predictable, including answers that are surely wrong.

Everything is evaluated exactly by enumeration, as elsewhere in rl4llm.
"""

from __future__ import annotations

import numpy as np

from rl4llm.estimators import advantages
from rl4llm.methods._common import Adam
from rl4llm.toy_language import (
    ToyLanguage,
    grad_log_likelihood,
    make_reference,
    sample_sequences,
    sequence_logprobs,
)

BUCKETS = (np.arange(5) + 0.5) / 5  # stated probabilities 0.1 .. 0.9


class CalibrationTask:
    def __init__(self, seed: int = 0, overconfident_prior: float = 0.55):
        self.lang = ToyLanguage(V=5, L=4, n_prompts=4)
        rng = np.random.default_rng(seed)
        seqs = self.lang.sequences
        self.answer = seqs[:, 0] * 25 + seqs[:, 1] * 5 + seqs[:, 2]  # 0..124
        self.conf = BUCKETS[seqs[:, 3]]  # stated probability of each response
        # Hidden success probability of each answer: a logistic of random token features, so that
        # answers range from almost surely wrong to almost surely right.
        # Prompts range from hard (no answer is very likely to be right) to easy.
        w = rng.normal(0, 0.7, size=(self.lang.n_prompts, 3, 5))
        bias = np.array([-2.6, -1.6, -0.6, 0.8])[:, None]
        score = bias + sum(w[:, t, :][:, seqs[:, t]] for t in range(3))
        self.q = 1 / (1 + np.exp(-score))  # (P, N): success probability of response's answer
        # Reference policy: answers from the usual MLE "pretraining"; confidence skewed high
        # (an overconfident prior), the same at every answer.
        ref = make_reference(self.lang, seed=seed + 1, concentration=0.6)
        prior = np.array([0.05, 0.05, 0.10, 0.25, overconfident_prior])
        prior = prior / prior.sum()
        off3 = 1 + 5 + 25
        ref[:, off3:, :] = np.log(prior)[None, None, :]
        self.ref_logits = ref

    # --- rewards -------------------------------------------------------------------------------
    def expected_reward(self, kind: str) -> np.ndarray:
        """(P, N) expected reward of every response under the outcome distribution."""
        q, p = self.q, self.conf[None, :]
        brier = -(q * (1 - p) ** 2 + (1 - q) * p**2)
        if kind == "binary":
            return q
        if kind == "brier":
            return brier
        if kind == "combined":
            return q + brier
        raise ValueError(kind)

    def sample_reward(self, rng, prompts, seqs, kind: str) -> np.ndarray:
        y = (rng.random(len(seqs)) < self.q[prompts, seqs]).astype(float)
        p = self.conf[seqs]
        if kind == "binary":
            return y
        if kind == "brier":
            return -((p - y) ** 2)
        if kind == "combined":
            return y - (p - y) ** 2
        raise ValueError(kind)

    # --- exact evaluation ----------------------------------------------------------------------
    def metrics(self, logits) -> dict:
        prob = np.exp(sequence_logprobs(self.lang, logits))  # (P, N)
        acc = (prob * self.q).sum(1).mean()
        mass = np.array([prob[:, self.conf == b].sum() for b in BUCKETS]) / prob.shape[0]
        hit = np.array([(prob[:, self.conf == b] * self.q[:, self.conf == b]).sum() for b in BUCKETS]) / prob.shape[0]
        freq = np.where(mass > 0, hit / np.maximum(mass, 1e-12), np.nan)
        ece = np.nansum(mass * np.abs(freq - BUCKETS))
        confident_wrong = (prob * (1 - self.q) * (self.conf[None, :] >= 0.7)).sum(1).mean()
        mean_conf = (prob * self.conf[None, :]).sum(1).mean()
        return {"accuracy": acc, "bucket_mass": mass, "bucket_accuracy": freq, "ece": ece,
                "confident_wrong": confident_wrong, "mean_confidence": mean_conf}


def exact_optimum(task: CalibrationTask, kind: str) -> np.ndarray:
    """Per prompt, the response (answer + confidence) with the highest expected reward."""
    return task.expected_reward(kind).argmax(axis=1)


def train(task: CalibrationTask, kind: str, steps=600, group_size=8, lr=0.05, seed=0,
          lr_final_frac=0.05, eval_every=20):
    """RLOO on sampled rewards (a contextual bandit: one decision, an immediate outcome)."""
    rng = np.random.default_rng(seed)
    lang = task.lang
    P, G = lang.n_prompts, group_size
    prompts = np.repeat(np.arange(P), G)
    logits = task.ref_logits.copy()
    opt = Adam(lr, total_steps=steps, final_frac=lr_final_frac)
    history = []
    for step in range(steps + 1):
        if step % eval_every == 0:
            history.append((step, task.metrics(logits)))
        if step == steps:
            break
        seqs = sample_sequences(lang, logits, rng, prompts)
        r = task.sample_reward(rng, prompts, seqs, kind)
        w = advantages(r.reshape(P, G), "rloo").ravel()
        logits = opt.step(logits, grad_log_likelihood(lang, logits, prompts, seqs, w) / (P * G))
    return logits, history
