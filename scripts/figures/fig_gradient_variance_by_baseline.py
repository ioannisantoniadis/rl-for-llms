"""What a baseline buys: invariance to the reward's offset, and (with a critic) a bias-variance knob.

This figure makes visible that (1) subtracting a baseline never changes the expected policy
gradient but makes its variance independent of the reward's offset, whereas plain REINFORCE's error
grows with the square of the offset (a learned reward model's offset is arbitrary); (2) with a 0/1
reward and rare successes, zero already is a good baseline, so baselines matter most once the policy
is good; (3) GAE's lambda trades variance for bias when the critic is imperfect.

Real computation: 1,500 independent gradient estimates per point (8 samples per prompt unless
varied), each compared with the exact gradient of E[r] computed by enumeration. Relative MSE =
E||g_hat - g||^2 / ||g||^2. The "critic" curves use the exact state value V^pi(s_t) (the best a
critic can do), and in panel 3 that value plus a fixed per-state error (std 0.15).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
from _theme import CATEGORICAL, FAMILY_COLOR, INK_SECONDARY, MUTED, apply_theme, savefig
from matplotlib.ticker import FuncFormatter, NullFormatter

from rl4llm.experiments_gradvar import bias_variance, exact_values
from rl4llm.toy_language import (
    default_testbed,
    logits_from_sequence_logprobs,
    optimal_logprobs,
    sequence_logprobs,
)

apply_theme()
N_DRAWS = 1500

lang, ref, reward = default_testbed()
ref_logp = sequence_logprobs(lang, ref)
trained = logits_from_sequence_logprobs(lang, optimal_logprobs(ref_logp, reward, 0.5))

est_style = {
    "none": ("no baseline (REINFORCE)", INK_SECONDARY, "-", "o"),
    "rloo": ("leave-one-out (RLOO)", FAMILY_COLOR["policy_gradient"], "-", "s"),
    "oracle_V(x)": (r"exact $V(x)$ per prompt", CATEGORICAL[6], "--", "^"),
    "critic_V(s_t)": (r"exact per-token critic $V(s_t)$", CATEGORICAL[2], ":", "D"),
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

# Panel 1: reward offset at pi_ref.
offsets = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 3.0])
ax = axes[0]
for k, (label, color, ls, mk) in est_style.items():
    mse = []
    for c in offsets:
        rr = reward + c
        vals = exact_values(lang, ref, rr)
        b, v, _ = bias_variance(lang, ref, rr, 8, k, n_draws=N_DRAWS, values=vals)
        mse.append(b + v)
    ax.plot(offsets, mse, color=color, ls=ls, marker=mk, ms=5, label=label)
ax.set_yscale("log")
ax.set_xlabel(r"reward offset $c$  (training on $r + c$)")
ax.set_ylabel("relative MSE of the gradient estimate")
ax.set_title(r"At $\pi_{\mathrm{ref}}$: a baseline removes the offset")
ax.legend(fontsize=8.3, loc="upper left")

# Panel 2: group size at a good policy (85% success), offset 0.
Gs = np.array([2, 4, 8, 16, 32])
ax = axes[1]
vals = exact_values(lang, trained, reward)
for k, (label, color, ls, mk) in est_style.items():
    mse = [sum(bias_variance(lang, trained, reward, G, k, n_draws=N_DRAWS, values=vals)[:2]) for G in Gs]
    ax.plot(Gs, mse, color=color, ls=ls, marker=mk, ms=5, label=label)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(Gs, [str(g) for g in Gs])
ax.set_xlabel(r"samples per prompt $G$")
ax.set_ylabel("relative MSE")
ax.set_title("At a good policy (85% correct), offset 0")

# Panel 3: GAE lambda with an imperfect critic.
rng = np.random.default_rng(7)
P = lang.n_prompts
noise = np.zeros_like(vals)
noise[:, :, 0] = rng.normal(0, 0.15, size=(P, 1))
for t in range(1, lang.L):
    e = rng.normal(0, 0.15, size=(P, lang.n_prefixes))
    noise[:, :, t] = e[:, lang.prefix_index[:, t]]
lams = np.array([0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
bias2, var = [], []
for lam in lams:
    b, v, _ = bias_variance(lang, trained, reward, 8, "gae", n_draws=N_DRAWS, values=vals + noise, lam=lam)
    bias2.append(b)
    var.append(v)
ax = axes[2]
ax.plot(lams, var, color=FAMILY_COLOR["policy_gradient"], marker="o", ms=5, label="variance")
ax.plot(lams, bias2, color=CATEGORICAL[7], marker="s", ms=5, ls="--", label=r"squared bias")
ax.plot(lams, np.array(var) + np.array(bias2), color=MUTED, lw=1.2, label="MSE (sum)")
ax.set_xlabel(r"GAE $\lambda$  (0 = one-step TD, 1 = Monte Carlo)")
ax.set_ylabel("relative to $\\|g\\|^2$")
ax.set_title("GAE with an imperfect critic")
ax.legend(fontsize=8.5, loc="center right")
ax.set_ylim(0, None)

axes[0].set_yticks([1, 2, 5, 10, 20, 50])
axes[1].set_yticks([0.5, 1, 2, 5, 10])
for ax in axes[:2]:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
savefig(fig, "gradient_variance_by_baseline")
print("lambda sweep bias2:", np.round(bias2, 4), "var:", np.round(var, 3))
