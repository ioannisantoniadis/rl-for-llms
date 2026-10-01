"""Estimating a KL vs regularizing with one: k1, k2, k3, and where each implementation converges.

This figure makes visible that (top) k3 is an unbiased, low-variance estimate of the *value*
KL(pi || pi_ref) when samples come from pi, which is why GRPO chose it, while k2 is biased and k1
is noisy; but (bottom) the way the estimator is turned into a *gradient* decides which divergence
is actually regularized. Putting k1 in the reward (or k2 in the loss) reaches the exact optimum
pi*; using k3 as a loss follows the forward KL and converges to a different policy (more reward,
much further from pi_ref, higher entropy); using k1 as a loss regularizes nothing.

Real computation. Top: exact mean and standard deviation of each single-sample, sequence-level
estimator under pi*_beta. Note that k3's relative standard deviation grows far from pi_ref (the
ratio pi_ref/pi is heavy-tailed under pi), so its low-noise advantage holds near pi_ref only for a range of beta (all 625 responses enumerated). Bottom: exact-gradient
training of E[r] - beta * penalty (beta = 0.5) with each implementation
(rl4llm.kl_penalties.penalty_gradient; tests/test_kl_penalties.py checks the gradients).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
from _theme import CATEGORICAL, FAMILY_COLOR, INK, INK_SECONDARY, MUTED, apply_theme, savefig

from rl4llm.estimators import kl_estimator
from rl4llm.kl_penalties import penalty_gradient
from rl4llm.methods._common import Adam
from rl4llm.methods.exact import exact_gradient
from rl4llm.metrics import expected, kl
from rl4llm.toy_language import default_testbed, optimal_logprobs, sequence_logprobs

apply_theme()
lang, ref, reward = default_testbed()
ref_logp = sequence_logprobs(lang, ref)
BETA = 0.5
est_color = {"k1": INK_SECONDARY, "k2": CATEGORICAL[3], "k3": FAMILY_COLOR["policy_gradient"]}
est_label = {"k1": r"$k_1 = \log(\pi/\pi_{\mathrm{ref}})$", "k2": r"$k_2 = \frac{1}{2}\log^2(\pi/\pi_{\mathrm{ref}})$",
             "k3": r"$k_3 = \pi_{\mathrm{ref}}/\pi - 1 - \log(\pi_{\mathrm{ref}}/\pi)$"}

# Top row: estimator properties along pi*_beta.
betas = np.geomspace(20, 0.12, 40)
true, mean, std = [], {k: [] for k in est_color}, {k: [] for k in est_color}
for b in betas:
    lp = optimal_logprobs(ref_logp, reward, b)
    p = np.exp(lp)
    true.append(kl(lp, ref_logp).mean())
    for k in est_color:
        v = kl_estimator(lp, ref_logp, k)
        m = (p * v).sum(1)
        mean[k].append(m.mean())
        std[k].append(np.sqrt((p * (v - m[:, None]) ** 2).sum(1)).mean())
true = np.array(true)

fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))
ax = axes[0, 0]
ax.plot(true, true, color=MUTED, lw=1, ls="--", label="true KL")
for k, c in est_color.items():
    ax.plot(true, mean[k], color=c, lw=2 if k != "k1" else 3.2, ls="-" if k != "k3" else (0, (4, 2)), label=est_label[k])
ax.set_xlabel(r"true $\mathrm{KL}(\pi\,\|\,\pi_{\mathrm{ref}})$  along $\pi^\star_\beta$")
ax.set_ylabel("mean of the single-sample estimate")
ax.set_title("Value: $k_1$ and $k_3$ are unbiased, $k_2$ is not")
ax.legend(fontsize=8.6, loc="upper left")
ax = axes[0, 1]
for k, c in est_color.items():
    ax.plot(true, np.array(std[k]) / true, color=c, label=k)
ax.set_yscale("log")
ax.set_xlabel(r"true $\mathrm{KL}(\pi\,\|\,\pi_{\mathrm{ref}})$")
ax.set_ylabel("std of one sample / true KL")
ax.set_title("Noise: $k_3$ is quiet near $\\pi_{\\mathrm{ref}}$, not far from it")
ax.legend(fontsize=9)

# Bottom row: what each implementation of the penalty converges to.
impl = {
    "k1_reward": (r"$k_1$ in the reward (classic RLHF)", INK, "-"),
    "k2_loss": (r"$k_2$ as a loss", CATEGORICAL[3], (0, (2, 2))),
    "k3_loss": (r"$k_3$ as a loss, per sequence", FAMILY_COLOR["policy_gradient"], "-"),
    "k3_loss_token": (r"$k_3$ as a loss, per token (GRPO)", FAMILY_COLOR["policy_gradient"], "--"),
    "k1_loss": (r"$k_1$ as a loss (no effect)", CATEGORICAL[7], ":"),
}
opt_logp = optimal_logprobs(ref_logp, reward, BETA)
STEPS = 1500
runs = {}
for kind in impl:
    logits, opt, hist = ref.copy(), Adam(0.05), []
    for s in range(STEPS + 1):
        lp = sequence_logprobs(lang, logits)
        hist.append((kl(lp, opt_logp).mean(), kl(lp, ref_logp).mean(), expected(lp, reward).mean()))
        if s < STEPS:
            g = exact_gradient(lang, logits, ref_logp, reward, 0.0) - BETA * penalty_gradient(lang, logits, ref, kind)
            logits = opt.step(logits, g)
    runs[kind] = np.array(hist)

ax = axes[1, 0]
for kind, (label, color, ls) in impl.items():
    ax.plot(np.arange(STEPS + 1), np.maximum(runs[kind][:, 0], 1e-6), color=color, ls=ls, label=label)
ax.set_yscale("log")
ax.set_ylim(1e-6, 10)
ax.set_xlabel("training step (exact gradients)")
ax.set_ylabel(r"$\mathrm{KL}(\pi_\theta\,\|\,\pi^\star)$")
ax.set_title(r"Regularizer: only $k_1$-in-reward / $k_2$-loss reach $\pi^\star$")
ax.legend(fontsize=8.3, loc="center right")

ax = axes[1, 1]
fb = np.geomspace(30, 0.05, 120)
fr = [(kl(optimal_logprobs(ref_logp, reward, b), ref_logp).mean(), expected(optimal_logprobs(ref_logp, reward, b), reward).mean()) for b in fb]
fx, fy = np.array(fr).T
ax.plot(fx, fy, color=MUTED, lw=1.3)
ax.text(1.6, 0.86, r"$\pi^\star_\beta$, every $\beta$", fontsize=8.5, color=INK_SECONDARY)
for kind, (label, color, ls) in impl.items():
    h = runs[kind]
    ax.plot(h[:, 1], h[:, 2], color=color, ls=ls, lw=1.3)
    ax.plot(h[-1, 1], h[-1, 2], "o", color=color, ms=7, mec="white", mew=1)
ax.plot(kl(opt_logp, ref_logp).mean(), expected(opt_logp, reward).mean(), "*", color=INK, ms=15, mec="white", zorder=5)
ax.annotate(r"$\pi^\star$ ($\beta=0.5$)", (kl(opt_logp, ref_logp).mean(), expected(opt_logp, reward).mean()),
            xytext=(0.9, 0.55), fontsize=9, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
ax.set_xscale("log")
ax.set_xlim(0.03, 5)
ax.set_xlabel(r"reverse $\mathrm{KL}(\pi_\theta\,\|\,\pi_{\mathrm{ref}})$")
ax.set_ylabel(r"expected reward $\mathbb{E}_{\pi_\theta}[r]$")
ax.set_title("Where each implementation ends up")

savefig(fig, "kl_estimators")
for kind in impl:
    h = runs[kind][-1]
    print(f"{kind:14s} KL->pi*={h[0]:.4f} revKL={h[1]:.3f} E[r]={h[2]:.3f}")
