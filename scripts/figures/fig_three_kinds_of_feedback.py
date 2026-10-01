"""Three kinds of feedback, one model, one gradient: what changes is what you observe and who chose it.

This figure makes visible that supervised learning and RL use the same machinery (the same
prefix-tabular policy, the same optimizer, the same call to grad_log_likelihood) and differ only in
the learning signal. With *instructive* feedback every sample is a correct answer, so learning speed
does not depend on how good the learner already is. With *evaluative* feedback on *self-generated*
samples, a hard prompt yields almost no signal at first (only ~2% of the learner's own answers are
rewarded), and the signal grows as the learner improves, because the learner's policy is choosing
its own training data. With evaluative feedback on a *fixed log* of someone else's samples, the
useful fraction of data never changes, so the learner cannot speed itself up.

Real computation: three training runs on the toy testbed (no KL term, reward = verifiable 0/1
correctness), with exact P(correct) for the hardest and easiest prompt after every step, and the
fraction of each step's samples that carried a nonzero gradient weight.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
from _theme import FAMILY_COLOR, INK_SECONDARY, apply_theme, savefig

from rl4llm.estimators import advantages
from rl4llm.methods._common import Adam
from rl4llm.metrics import expected
from rl4llm.toy_language import (
    default_testbed,
    grad_log_likelihood,
    optimal_logprobs,
    sample_sequences,
    sequence_logprobs,
)

apply_theme()

lang, ref, reward = default_testbed()
P, G, STEPS, LR, SEEDS = lang.n_prompts, 8, 300, 0.05, 5
prompts = np.repeat(np.arange(P), G)
ref_logp = sequence_logprobs(lang, ref)
# The "expert" behind the demonstrations: pi_ref restricted to correct answers (pi* as beta -> 0).
expert_logp = optimal_logprobs(ref_logp, reward, 1e-3)


def run(regime: str, seed: int):
    rng = np.random.default_rng(seed)
    logits, opt = ref.copy(), Adam(LR)
    p_correct, useful = [], []
    for _ in range(STEPS + 1):
        lp = sequence_logprobs(lang, logits)
        p_correct.append(expected(lp, reward))
        if regime == "instructive":
            # Labels: someone else's correct answers. Data distribution fixed, every sample useful.
            ys = np.array([rng.choice(lang.n_sequences, p=np.exp(expert_logp[p])) for p in prompts])
            w = np.ones(len(ys))
        elif regime == "self":
            # Evaluative feedback on the learner's own samples (REINFORCE, leave-one-out baseline).
            ys = sample_sequences(lang, logits, rng, prompts)
            w = advantages(reward[prompts, ys].reshape(P, G), "rloo").ravel()
        else:
            # Evaluative feedback on a fixed log of pi_ref's samples: reward-weighted likelihood.
            ys = sample_sequences(lang, ref, rng, prompts)
            w = reward[prompts, ys]
        useful.append((reward[prompts, ys] > 0).reshape(P, G).mean(axis=1))
        logits = opt.step(logits, grad_log_likelihood(lang, logits, prompts, ys, w) / len(ys))
    return np.array(p_correct), np.array(useful)


regimes = {
    "instructive": ("Instructive feedback: demonstrations\n(supervised learning / SFT)", FAMILY_COLOR["imitation"], "-"),
    "self": ("Evaluative feedback, self-generated samples\n(RL: REINFORCE on its own answers)", FAMILY_COLOR["policy_gradient"], "-"),
    "logged": ("Evaluative feedback, fixed log of another\npolicy's samples (offline reward-weighted MLE)", FAMILY_COLOR["policy_gradient"], "--"),
}
results = {k: [run(k, s) for s in range(SEEDS)] for k in regimes}

samples = np.arange(STEPS + 1) * G * P
HARD, EASY = 0, 2
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
for ax, prompt, title in ((axes[0], HARD, "Hard prompt"), (axes[1], EASY, "Easy prompt")):
    for k, (label, color, ls) in regimes.items():
        curves = np.array([r[0][:, prompt] for r in results[k]])
        ax.fill_between(samples, curves.min(0), curves.max(0), color=color, alpha=0.12, lw=0)
        ax.plot(samples, np.median(curves, 0), color=color, ls=ls, label=label)
    base = expected(ref_logp, reward)[prompt]
    ax.axhline(base, color=INK_SECONDARY, lw=0.8, ls=":")
    ax.text(samples[-1], base + 0.015, r"$\pi_{\mathrm{ref}}$", ha="right", va="bottom", fontsize=9, color=INK_SECONDARY)
    ax.set_title(f"{title}: P(correct) under " + r"$\pi_{\mathrm{ref}}$" + f" = {base:.2f}")
    ax.set_xlabel("training samples consumed")
    ax.set_ylabel("P(correct)  (exact)")
    ax.set_ylim(0, 1.02)
axes[0].legend(loc="center right", fontsize=8.2)

ax = axes[2]
W = 15  # moving-average window (steps); each step has only G = 8 samples for this prompt
for k, (label, color, ls) in regimes.items():
    u = np.array([np.array(r[1])[:-1, HARD] for r in results[k]]).mean(0)  # mean over seeds
    smooth = np.convolve(u, np.ones(W) / W, mode="valid")
    ax.plot(samples[W - 1 : W - 1 + len(smooth)], smooth, color=color, ls=ls)
ax.set_title("Hard prompt: share of samples that were correct")
ax.set_xlabel("training samples consumed")
ax.set_ylabel("fraction of samples with reward 1")
ax.set_ylim(0, 1.02)
ax.text(samples[-1] * 0.97, 0.94, "demonstrations: always 1", ha="right", fontsize=8.5, color=INK_SECONDARY)
ax.text(samples[-1] * 0.97, 0.07, "fixed log: stays at " + r"$\pi_{\mathrm{ref}}$" + "'s hit rate (0.02)", ha="right", fontsize=8.5, color=INK_SECONDARY)
ax.text(samples[-1] * 0.55, 0.55, "own samples: almost no signal at first,\nthen more as the learner improves", ha="left", fontsize=8.5, color=INK_SECONDARY)

savefig(fig, "three_kinds_of_feedback")
for k in regimes:
    pc = np.median(np.array([r[0] for r in results[k]]), 0)
    print(k, "final P(correct) per prompt:", pc[-1].round(3), " at 1/3 of training:", pc[STEPS // 3].round(3))
