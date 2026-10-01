"""Reward-model overoptimization on the toy language: Goodhart's law, measured exactly.

This figure makes visible that (1) optimizing a learned proxy reward harder (more KL from pi_ref)
first raises and then lowers the true reward while the proxy keeps rising, for both best-of-n and
KL-regularized RL; (2) a larger KL penalty stops training earlier along the curve (the "akin to
early stopping" behavior Gao et al. 2022 report), but in this toy a small or zero penalty also takes
a worse path than the proxy's optimum, which Gao et al. did not observe in their setting; (3) the cause is
misspecification, not noise: more preference data does not remove the peak, while a reward model
whose features can express the true rule does not overoptimize.

Real computation. A Bradley-Terry reward model, linear in per-position token features, is fit to
preference pairs sampled from pi_ref and labelled by a simulated annotator from the true 0/1
reward (rl4llm.reward_model). Optimization against it: exact pi*_beta(proxy) for a sweep of beta;
exact best-of-n distributions; exact-gradient RL with the KL in the reward. All rewards and KLs are
exact sums over the 625 responses, averaged over the four prompts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
from _theme import CATEGORICAL, FAMILY_COLOR, INK, MUTED, apply_theme, savefig

from rl4llm.methods._common import Adam
from rl4llm.methods.exact import exact_gradient
from rl4llm.metrics import expected, kl
from rl4llm.reward_model import best_of_n_logprobs, fit_bradley_terry, sum_features
from rl4llm.toy_language import default_testbed, optimal_logprobs, sequence_logprobs

apply_theme()
lang, ref, true_r = default_testbed()
ref_logp = sequence_logprobs(lang, ref)
proxy = fit_bradley_terry(lang, ref_logp, true_r, n_pairs=2000)
betas = np.geomspace(20, 0.01, 60)


def path(reward_for_policy, measure):
    """KL and the measured rewards along pi*_beta(reward_for_policy)."""
    out = []
    for b in betas:
        lp = optimal_logprobs(ref_logp, reward_for_policy, b)
        out.append([kl(lp, ref_logp).mean()] + [expected(lp, m).mean() for m in measure])
    return np.array(out)


fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# Panel 1: the overoptimization curves.
ax = axes[0]
gold = path(true_r, [true_r])
rl_path = path(proxy, [true_r, proxy])
ns = np.unique(np.geomspace(1, 4096, 40).astype(int))
bon = []
for n in ns:
    lp = best_of_n_logprobs(ref_logp, proxy, int(n))
    bon.append([kl(lp, ref_logp).mean(), expected(lp, true_r).mean(), expected(lp, proxy).mean()])
bon = np.array(bon)
ax.plot(gold[:, 0], gold[:, 1], color=MUTED, lw=1.3, label=r"best possible: $\pi^\star_\beta$ for the true reward")
ax.plot(rl_path[:, 0], rl_path[:, 1], color=FAMILY_COLOR["policy_gradient"], label=r"optimum of the proxy, $\pi^\star_\beta$(proxy): true reward")
ax.plot(rl_path[:, 0], rl_path[:, 2] - rl_path[0, 2] + rl_path[0, 1], color=FAMILY_COLOR["policy_gradient"], ls="--",
        label="  ... proxy reward (shifted to start at the same point)")
ax.plot(bon[:, 0], bon[:, 1], color=FAMILY_COLOR["imitation"], label="best-of-$n$ against the proxy: true reward")
ax.plot(bon[:, 0], bon[:, 2] - bon[0, 2] + bon[0, 1], color=FAMILY_COLOR["imitation"], ls="--", label="  ... proxy reward (shifted)")
ax.set_xlim(0, 7)
ax.set_ylim(0, 1.25)
ax.set_xlabel(r"$\mathrm{KL}(\pi\,\|\,\pi_{\mathrm{ref}})$  (optimization pressure)")
ax.set_ylabel("expected reward (exact)")
ax.set_title("Proxy keeps rising, true reward peaks and falls")
ax.legend(fontsize=7.8, loc="upper right", bbox_to_anchor=(1.0, 0.86))

# Panel 2: RL trajectories with different KL penalties (exact gradients, KL in the reward).
ax = axes[1]
ax.plot(rl_path[:, 0], rl_path[:, 1], color=MUTED, lw=1.2, label=r"$\pi^\star_\beta$(proxy) path")
pen = [0.0, 0.05, 0.1, 0.2, 0.5]
shades = [INK, CATEGORICAL[6], FAMILY_COLOR["policy_gradient"], CATEGORICAL[2], CATEGORICAL[3]]
for beta, c in zip(pen, shades):
    logits, opt, traj = ref.copy(), Adam(0.02), []
    for s in range(1501):
        if s % 3 == 0:
            lp = sequence_logprobs(lang, logits)
            traj.append((kl(lp, ref_logp).mean(), expected(lp, true_r).mean()))
        logits = opt.step(logits, exact_gradient(lang, logits, ref_logp, proxy, beta))
    traj = np.array(traj)
    ax.plot(traj[:, 0], traj[:, 1], color=c, lw=1.4, label=rf"RL, KL penalty $\beta={beta}$")
    ax.plot(traj[-1, 0], traj[-1, 1], "o", color=c, ms=6, mec="white", mew=1)
ax.set_xlim(0, 7)
ax.set_ylim(0, 0.85)
ax.set_xlabel(r"$\mathrm{KL}(\pi_\theta\,\|\,\pi_{\mathrm{ref}})$")
ax.set_ylabel("true reward (exact)")
ax.set_title("RL against the proxy, different KL penalties")
ax.legend(fontsize=7.8, loc="lower left")

# Panel 3: more data vs a better-specified reward model.
ax = axes[2]
for n, c in zip((250, 1000, 4000, 16000), (CATEGORICAL[3], CATEGORICAL[2], FAMILY_COLOR["policy_gradient"], CATEGORICAL[6])):
    pr = fit_bradley_terry(lang, ref_logp, true_r, n_pairs=n, seed=1)
    pth = path(pr, [true_r])
    ax.plot(pth[:, 0], pth[:, 1], color=c, label=f"position features, {n} pairs/prompt")
good = fit_bradley_terry(lang, ref_logp, true_r, n_pairs=1000, seed=1, features=sum_features(lang))
pth = path(good, [true_r])
ax.plot(pth[:, 0], pth[:, 1], color=INK, ls="--", label="features that can express the rule, 1000 pairs")
ax.plot(gold[:, 0], gold[:, 1], color=MUTED, lw=1.2, label="true reward itself")
ax.set_xlim(0, 7)
ax.set_ylim(0, 1.02)
ax.set_xlabel(r"$\mathrm{KL}(\pi\,\|\,\pi_{\mathrm{ref}})$ along $\pi^\star_\beta$(proxy)")
ax.set_ylabel("true reward (exact)")
ax.set_title("More data does not fix a misspecified proxy")
ax.legend(fontsize=7.6, loc="lower left")

savefig(fig, "overoptimization")
i = np.argmax(rl_path[:, 1])
print(f"proxy optimum path: peak true {rl_path[i, 1]:.3f} at KL {rl_path[i, 0]:.2f}; end true {rl_path[-1, 1]:.3f} at KL {rl_path[-1, 0]:.2f}")
j = np.argmax(bon[:, 1])
print(f"best-of-n: peak true {bon[j, 1]:.3f} at n={ns[j]} KL {bon[j, 0]:.2f}; n={ns[-1]} true {bon[-1, 1]:.3f}")
print("gold frontier at KL~1:", np.interp(1.0, gold[:, 0], gold[:, 1]).round(3))
