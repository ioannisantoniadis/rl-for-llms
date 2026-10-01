"""Learning by imitating a better distribution: where filtering, weighting and distillation land.

This figure makes visible that (1) supervised learning on samples filtered or weighted to look like
pi* reaches pi* (exact-acceptance rejection sampling, reward-weighted regression with weights
exp(r/beta)), while *iterated hard filtering* (keep only correct samples from the current policy,
fine-tune, repeat: the STaR / ReST / expert-iteration pattern) runs past even the beta -> 0 limit and
collapses onto a few correct answers; (2) best-of-n sampling against a reward model traces a curve
just below the KL-regularized frontier, and the common formula log n - (n-1)/n overstates its KL;
(3) a teacher's per-token log-probabilities are a denser signal per sampled response than a
sequence-level reward.

Real computation. Panel 1: rl4llm.methods.rft (accept / weighted / iterative filter) on the
default testbed, exact reward and KL. Panel 2: exact best-of-n distributions
(rl4llm.reward_model.best_of_n_logprobs) against a learned Bradley-Terry proxy, vs pi*_beta of the
same proxy. Panel 3: on-policy distillation, off-policy distillation and RLOO toward the same pi*
(teacher = pi*), 4 samples per prompt per step, median of 3 seeds.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
from _theme import CATEGORICAL, FAMILY_COLOR, INK, INK_SECONDARY, MUTED, apply_theme, savefig

from rl4llm.methods import distill, reinforce, rft
from rl4llm.metrics import entropy, expected, kl
from rl4llm.reward_model import best_of_n_logprobs, fit_bradley_terry
from rl4llm.toy_language import (
    default_testbed,
    logits_from_sequence_logprobs,
    optimal_logprobs,
    sequence_logprobs,
)

apply_theme()
lang, ref, reward = default_testbed()
ref_logp = sequence_logprobs(lang, ref)
AQ, BLUE = FAMILY_COLOR["imitation"], FAMILY_COLOR["policy_gradient"]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))


def frontier(r_for_pol, r_measure, betas):
    out = []
    for b in betas:
        lp = optimal_logprobs(ref_logp, r_for_pol, b)
        out.append((kl(lp, ref_logp).mean(), expected(lp, r_measure).mean()))
    return np.array(out)


# Panel 1: filtering and weighting.
ax = axes[0]
fr = frontier(reward, reward, np.geomspace(30, 0.02, 150))
ax.plot(fr[:, 0], fr[:, 1], color=MUTED, lw=1.4, label=r"$\pi^\star_\beta$, every $\beta$")
lim = optimal_logprobs(ref_logp, reward, 0.0)
ax.plot(kl(lim, ref_logp).mean(), 1.0, "D", color=INK, ms=7, zorder=5)
ax.annotate(r"$\beta\to0$: $\pi_{\mathrm{ref}}$ restricted" + "\nto correct answers", (kl(lim, ref_logp).mean(), 1.0),
            xytext=(0.05, 0.93), fontsize=8.3, color=INK_SECONDARY, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
for b, mk in ((1.0, "o"), (0.5, "s"), (0.25, "^")):
    _, h = rft.train(lang, ref, reward, b, steps=800, n_proposals_per_prompt=100_000, batch_size=512, lr_final_frac=0.02, eval_every=800)
    ax.plot(h.kl_to_ref[-1], h.reward[-1], mk, color=AQ, ms=8, mec="white", zorder=4,
            label=rf"exact-acceptance RFT, $\beta={b}$")
_, h = rft.train_weighted(lang, ref, reward, 0.5, steps=800, n_samples_per_prompt=100_000, batch_size=512, lr_final_frac=0.02, eval_every=800)
ax.plot(h.kl_to_ref[-1], h.reward[-1], "x", color=INK, ms=9, mew=2, zorder=6, label=r"reward-weighted regression, $\beta=0.5$")
logits_it, h = rft.train_iterative_filter(lang, ref, reward, 0.5, iterations=10)
ax.plot(h.kl_to_ref, h.reward, "-o", color=CATEGORICAL[7], ms=4, lw=1.4, label="iterated hard filtering (STaR/ReST-style)")
ax.set_xlabel(r"$\mathrm{KL}(\pi_\theta\,\|\,\pi_{\mathrm{ref}})$  (exact)")
ax.set_ylabel(r"$\mathbb{E}_{\pi_\theta}[r]$  (exact)")
ax.set_title("Filtering targets: $\\pi^\\star$, or past it")
ax.legend(fontsize=7.6, loc="lower right")
ax.set_xlim(-0.05, 3.5)

# Panel 2: best-of-n vs the KL-regularized frontier for a learned reward model.
ax = axes[1]
proxy = fit_bradley_terry(lang, ref_logp, reward, n_pairs=2000)
pf = frontier(proxy, proxy, np.geomspace(30, 0.02, 150))
ax.plot(pf[:, 0], pf[:, 1], color=MUTED, lw=1.4, label=r"$\pi^\star_\beta$ for the reward model")
ns = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]
bon = np.array([(kl(best_of_n_logprobs(ref_logp, proxy, n), ref_logp).mean(),
                 expected(best_of_n_logprobs(ref_logp, proxy, n), proxy).mean()) for n in ns])
bound = np.array([np.log(n) - (n - 1) / n for n in ns])
ax.plot(bon[:, 0], bon[:, 1], "o-", color=AQ, ms=5, label="best-of-$n$ (exact KL)")
ax.plot(bound, bon[:, 1], "o--", color=AQ, ms=4, alpha=0.6, mfc="white", label=r"same, at the formula $\log n - \frac{n-1}{n}$")
for n, (x, y) in zip(ns, bon):
    if n in (2, 8, 64, 1024):
        ax.annotate(f"$n={n}$", (x, y), xytext=(4, -12), textcoords="offset points", fontsize=8, color=INK_SECONDARY)
ax.set_xlabel(r"$\mathrm{KL}(\pi\,\|\,\pi_{\mathrm{ref}})$")
ax.set_ylabel("expected reward-model score")
ax.set_title("Best-of-$n$ sits just below the frontier")
ax.legend(fontsize=7.8, loc="lower right")
ax.set_xlim(-0.1, 6.5)

# Panel 3: dense vs sparse signal.
ax = axes[2]
opt_logp = optimal_logprobs(ref_logp, reward, 0.5)
teacher = logits_from_sequence_logprobs(lang, opt_logp)
STEPS, G = 300, 4
runs = {
    "on-policy distillation (per-token teacher log-probs)": (AQ, "-", lambda s: distill.train(lang, ref, teacher, reward, steps=STEPS, group_size=G, seed=s, lr_final_frac=0.05)[1]),
    "off-policy distillation (SFT on teacher samples)": (AQ, "--", lambda s: distill.train_offpolicy(lang, ref, teacher, reward, steps=STEPS, group_size=G, seed=s, lr_final_frac=0.05)[1]),
    "RLOO (one reward per response)": (BLUE, "-", lambda s: reinforce.train(lang, ref, reward, 0.5, steps=STEPS, group_size=G, seed=s, lr_final_frac=0.05, eval_every=5)[1]),
}
for name, (c, ls, run) in runs.items():
    hs = [run(s) for s in range(3)]
    k = np.median([h.kl_to_opt for h in hs], 0)
    ax.plot(np.array(hs[0].step) * G * lang.n_prompts, k, color=c, ls=ls, label=name)
ax.set_yscale("log")
ax.set_xlabel("responses sampled")
ax.set_ylabel(r"$\mathrm{KL}(\pi_\theta\,\|\,\pi^\star)$")
ax.set_title("A teacher's log-probabilities are a dense reward")
ax.legend(fontsize=7.8, loc="upper right")

savefig(fig, "imitation")
print("iterated filter final entropy", entropy(sequence_logprobs(lang, logits_it)).round(2),
      "beta->0 limit KLref", round(kl(lim, ref_logp).mean(), 2), "entropy", entropy(lim).round(2))
print("best-of-n exact KL vs bound:", [(n, round(x, 3), round(b, 3)) for n, (x, _), b in zip(ns, bon, bound)])
