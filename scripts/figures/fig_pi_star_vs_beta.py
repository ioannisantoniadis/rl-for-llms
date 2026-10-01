"""The closed-form optimum pi* = pi_ref exp(r / beta) / Z, as beta varies.

This figure makes visible that beta interpolates between the reference policy (beta -> infinity)
and pi_ref restricted to the highest-reward responses (beta -> 0), by multiplying every correct
response's probability by the same factor exp(1/beta) and renormalizing. The *relative*
probabilities among correct responses, and among wrong ones, never change: pi* keeps pi_ref's
preferences inside each reward level. Every beta gives one point on the reward-KL frontier.

Real computation: exact pi*_beta by the closed form (rl4llm.toy_language.optimal_logprobs) for one
prompt of the testbed; exact expected reward, KL from pi_ref and entropy, averaged over prompts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
from _theme import (
    CATEGORICAL,
    FAMILY_COLOR,
    INK,
    INK_SECONDARY,
    MUTED,
    SEQUENTIAL_BLUE,
    apply_theme,
    savefig,
)

from rl4llm.metrics import entropy, expected, kl
from rl4llm.toy_language import default_testbed, optimal_logprobs, sequence_logprobs

apply_theme()
lang, ref, reward = default_testbed()
ref_logp = sequence_logprobs(lang, ref)
X = 1  # P(correct) = 0.19 under pi_ref
order = np.argsort(-ref_logp[X])
correct = reward[X, order] > 0

fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), gridspec_kw={"width_ratios": [1.35, 1]})
ax = axes[0]
shown = [(np.inf, r"$\pi_{\mathrm{ref}}$ ($\beta\to\infty$)", MUTED), (1.0, r"$\beta = 1$", SEQUENTIAL_BLUE[5]),
         (0.5, r"$\beta = 0.5$", SEQUENTIAL_BLUE[7]), (0.15, r"$\beta = 0.15$", SEQUENTIAL_BLUE[9])]
rank = np.arange(1, lang.n_sequences + 1)
for b, label, c in shown:
    lp = ref_logp if np.isinf(b) else optimal_logprobs(ref_logp, reward, b)
    p = np.exp(lp[X, order])
    ax.plot(rank[~correct], p[~correct], color=c, lw=1.4, label=label, zorder=2)
    ax.scatter(rank[correct], p[correct], s=12, color=c, zorder=3)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(1e-6, 1)
ax.set_xlabel(r"responses ranked by $\pi_{\mathrm{ref}}(y\mid x)$")
ax.set_ylabel(r"probability")
ax.set_title("One prompt: lines = wrong answers, dots = correct answers", fontsize=11.5)
ax.legend(fontsize=9, loc="lower left")

ax = axes[1]
betas = np.geomspace(50, 0.03, 200)
er, klr, H = [], [], []
for b in betas:
    lp = optimal_logprobs(ref_logp, reward, b)
    er.append(expected(lp, reward).mean())
    klr.append(kl(lp, ref_logp).mean())
    H.append(entropy(lp).mean())
ax.plot(1 / betas, er, color=FAMILY_COLOR["policy_gradient"], label=r"expected reward $\mathbb{E}_{\pi^\star}[r]$")
ax.plot(1 / betas, klr, color=CATEGORICAL[1], label=r"$\mathrm{KL}(\pi^\star\,\|\,\pi_{\mathrm{ref}})$")
ax.plot(1 / betas, H, color=INK, ls="--", lw=1.4, label=r"entropy $H(\pi^\star)$ (nats)")
ax.set_xscale("log")
ax.set_xlabel(r"$1/\beta$  (strength of the reward relative to the KL)")
ax.set_title("Averaged over prompts", fontsize=11.5)
ax.legend(fontsize=9, loc="center left")
ax.axvline(2, color=MUTED, lw=0.8, ls=":")
ax.text(2.1, 2.3, r"$\beta=0.5$, used in" + "\nmost figures", fontsize=8.5, color=INK_SECONDARY)
savefig(fig, "pi_star_vs_beta")
