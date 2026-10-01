"""The SFT gradient and the policy gradient are the same operation with different q and w.

This figure makes visible that supervised fine-tuning and REINFORCE both do one thing, raise
log pi_theta of some responses, and differ only in which responses they touch (someone else's
demonstrations vs the learner's own samples) and in the weight on each (always +1 vs a signed
advantage that also pushes wrong answers *down*). Their expected gradients differ accordingly: the
coefficient on each response's grad log pi(y) is q(y) w(y), which for SFT is the expert's
probability of y (correct answers only, wherever the expert puts mass) and for REINFORCE is
pi(y)(r(y) - b): positive on correct answers *in proportion to how likely the policy already finds
them*, negative on wrong ones.

Real computation: one batch of 16 responses for one prompt with each method's actual weights; and
the exact expected coefficient q(y) w(y) for all 625 responses (the "gradient coefficient" of
Shao et al. 2024, Sec. 5.2).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
from _theme import CATEGORICAL, FAMILY_COLOR, INK_SECONDARY, MUTED, apply_theme, savefig

from rl4llm.estimators import advantages
from rl4llm.toy_language import (
    default_testbed,
    optimal_logprobs,
    sample_sequences,
    sequence_logprobs,
)

apply_theme()
lang, ref, reward = default_testbed()
X, B = 1, 16  # prompt (P(correct) = 0.19 under pi_ref), batch size
ref_logp = sequence_logprobs(lang, ref)
expert_logp = optimal_logprobs(ref_logp, reward, 1e-3)  # pi_ref restricted to correct answers
order = np.argsort(-ref_logp[X])  # x-axis: responses ranked by pi_ref probability
rank = np.empty_like(order)
rank[order] = np.arange(len(order))
correct = reward[X] > 0

rng = np.random.default_rng(3)
prompts = np.full(B, X)
demo = rng.choice(lang.n_sequences, size=B, p=np.exp(expert_logp[X]))
own = sample_sequences(lang, ref, rng, prompts)
w_own = advantages(reward[X, own][None, :], "rloo").ravel()


p_ref = np.exp(ref_logp[X])
d_sft = np.exp(expert_logp[X])  # q = expert, w = 1
d_pg = p_ref * (reward[X] - (p_ref * reward[X]).sum())  # q = pi_ref, w = r - E[r]

fig, axes = plt.subplots(2, 2, figsize=(13, 6.6), gridspec_kw={"width_ratios": [1, 1.25]}, sharex="col")
rows = [
    ("SFT: someone else's correct answers, weight +1", demo, np.ones(B), d_sft, FAMILY_COLOR["imitation"]),
    ("REINFORCE: the policy's own samples, weight = advantage", own, w_own, d_pg, FAMILY_COLOR["policy_gradient"]),
]
for i, (title, ys, w, delta, color) in enumerate(rows):
    ax = axes[i, 0]
    # Aggregate duplicate samples so each response appears once with its summed weight.
    uniq, inv = np.unique(ys, return_inverse=True)
    wsum = np.bincount(inv, weights=w)
    for y, ww in zip(uniq, wsum):
        c = color if correct[y] else CATEGORICAL[7]
        ax.vlines(rank[y] + 1, 0, ww, color=c, lw=2)
        ax.plot(rank[y] + 1, ww, "o", color=c, ms=5)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xscale("log")
    ax.set_ylabel("summed weight $w$ in the batch")
    ax.set_title(title, fontsize=11)
    ax = axes[i, 1]
    ax.scatter(rank[~correct] + 1, delta[~correct], s=7, color=CATEGORICAL[7], alpha=0.6, label="wrong answers")
    ax.scatter(rank[correct] + 1, delta[correct], s=9, color=color, alpha=0.9, label="correct answers")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xscale("log")
    ax.set_ylabel(r"expected coefficient $q(y)\,w(y)$")
    ax.set_title(r"Expected gradient: coefficient on $\nabla\log\pi(y \mid x)$, all 625 responses", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right")
for ax in axes[1]:
    ax.set_xlabel(r"responses ranked by $\pi_{\mathrm{ref}}(y \mid x)$  (1 = most likely)")
axes[0, 0].text(0.98, 0.92, "red: wrong answers\ncolored: correct answers", transform=axes[0, 0].transAxes,
                ha="right", va="top", fontsize=8.5, color=INK_SECONDARY)

savefig(fig, "sft_vs_reinforce_gradient")
print("SFT coefficient mass on correct:", d_sft[correct].sum().round(3), "| PG positive mass:",
      d_pg[d_pg > 0].sum().round(4), "negative mass:", d_pg[d_pg < 0].sum().round(4))
