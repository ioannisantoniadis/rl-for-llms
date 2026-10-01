"""All roads lead to pi*: every method in the book, trained on the toy language, measured against the
exact optimum of the KL-regularized objective.

This figure makes visible that methods with completely different training signals (on-policy policy
gradients with or without a critic, offline logistic regression on preference pairs, supervised
learning on filtered samples, and per-token distillation from a teacher) are estimators of the same
target distribution pi*, and that what separates them in practice is the estimator (gradient noise,
finite offline data, how the KL is implemented), not where they are trying to go. The one exception
is instructive: GRPO with its KL estimator in the loss regularizes a different divergence and
converges elsewhere (Chapter 3).

Real computation: every curve is a training run of an rl4llm.methods implementation (same Adam
optimizer and the same cosine learning-rate decay for every method, beta = 0.5); KL(pi_theta || pi*)
and the reward/KL coordinates are exact sums over all 625 responses, averaged over the four
prompts. Sampled methods: median over 3 seeds.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
from _theme import CATEGORICAL, FAMILY_COLOR, INK, INK_SECONDARY, MUTED, apply_theme, savefig

from rl4llm.methods import distill, dpo, exact, grpo, ppo, reinforce, rft
from rl4llm.metrics import expected, kl
from rl4llm.toy_language import (
    default_testbed,
    logits_from_sequence_logprobs,
    optimal_logprobs,
    sequence_logprobs,
)

apply_theme()
BETA, STEPS, EVERY, DECAY, SEEDS = 0.5, 1000, 10, 0.02, (0, 1, 2)
FLOOR = 1e-5

lang, ref, reward = default_testbed()
ref_logp = sequence_logprobs(lang, ref)
opt_logp = optimal_logprobs(ref_logp, reward, BETA)
teacher = logits_from_sequence_logprobs(lang, opt_logp)
common = {"steps": STEPS, "eval_every": EVERY, "lr_final_frac": DECAY}

PG, PREF, IMIT = FAMILY_COLOR["policy_gradient"], FAMILY_COLOR["preference"], FAMILY_COLOR["imitation"]
GR = CATEGORICAL[6]
methods = {
    # name: (family panel, color, linestyle, runner(seed) -> History)
    "RLOO (no critic)": (0, PG, ":", lambda s: reinforce.train(lang, ref, reward, BETA, seed=s, **common)[1]),
    "PPO-RLHF (critic + GAE)": (0, PG, "-", lambda s: ppo.train(lang, ref, reward, BETA, lr=0.02, seed=s, **common)[1]),
    "GRPO, KL in the reward": (0, GR, "-", lambda s: grpo.train(lang, ref, reward, BETA, kl="k1_reward", epochs=2, seed=s, **common)[1]),
    "GRPO as published ($k_3$ in the loss)": (0, GR, "--", lambda s: grpo.train(lang, ref, reward, BETA, kl="k3_loss", epochs=2, seed=s, **common)[1]),
    "DPO, infinite pairs": (1, PREF, "-", lambda s: dpo.train_population(lang, ref, reward, BETA, **common)[1]),
    "DPO, 100k pairs/prompt": (1, PREF, "--", lambda s: dpo.train(lang, ref, reward, BETA, n_pairs_per_prompt=100_000, batch_size=512, seed=s, **common)[1]),
    "DPO, 5k pairs/prompt": (1, PREF, ":", lambda s: dpo.train(lang, ref, reward, BETA, n_pairs_per_prompt=5_000, batch_size=512, seed=s, **common)[1]),
    "Rejection-sampling FT, 100k proposals": (2, IMIT, "--", lambda s: rft.train(lang, ref, reward, BETA, n_proposals_per_prompt=100_000, batch_size=512, seed=s, **common)[1]),
    "Rejection-sampling FT, 5k proposals": (2, IMIT, ":", lambda s: rft.train(lang, ref, reward, BETA, n_proposals_per_prompt=5_000, batch_size=512, seed=s, **common)[1]),
    "On-policy distillation (teacher $= \\pi^\\star$)": (2, IMIT, "-", lambda s: distill.train(lang, ref, teacher, reward, seed=s, **common)[1]),
}
deterministic = {"DPO, infinite pairs"}
_, ref_hist = exact.train(lang, ref, reward, BETA, **common)

results = {}
for name, (_, _, _, run) in methods.items():
    seeds = (0,) if name in deterministic else SEEDS
    hs = [run(s) for s in seeds]
    results[name] = {
        "step": np.array(hs[0].step),
        "kl_opt": np.median([h.kl_to_opt for h in hs], 0),
        "kl_ref": np.median([h.kl_to_ref for h in hs], 0),
        "reward": np.median([h.reward for h in hs], 0),
    }

fig = plt.figure(figsize=(15, 9.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.38, wspace=0.25)
titles = ["Policy gradients (on-policy)", "Direct preference optimization (offline)",
          "Imitating a better distribution"]
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
for i, ax in enumerate(axes):
    ax.plot(ref_hist.step, np.maximum(ref_hist.kl_to_opt, FLOOR), color=INK, ls=":", lw=1.4,
            label="exact gradient (reference)")
    for name, (panel, color, ls, _) in methods.items():
        if panel == i:
            r = results[name]
            ax.plot(r["step"], np.maximum(r["kl_opt"], FLOOR), color=color, ls=ls, label=name)
    ax.set_yscale("log")
    ax.set_ylim(FLOOR * 0.7, 2)
    ax.set_title(titles[i], fontsize=11.5)
    ax.set_xlabel("training step")
    ax.legend(fontsize=7.6, loc="lower right", bbox_to_anchor=(1.0, 0.1))
axes[0].set_ylabel(r"$\mathrm{KL}(\pi_\theta\,\|\,\pi^\star)$  (exact)")

# Bottom left: the reward-KL plane with every method's endpoint.
bottom = gs[1, :].subgridspec(1, 2, width_ratios=[1.55, 1], wspace=0.62)
ax = fig.add_subplot(bottom[0])
betas = np.geomspace(20, 0.12, 120)
fr = np.array([(kl(optimal_logprobs(ref_logp, reward, b), ref_logp).mean(),
                expected(optimal_logprobs(ref_logp, reward, b), reward).mean()) for b in betas])
ax.plot(fr[:, 0], fr[:, 1], color=MUTED, lw=1.4, zorder=1)
ax.text(1.05, 0.42, r"gray: $\pi^\star_\beta$ for every $\beta$, the most reward" + "\nreachable at each KL budget",
        fontsize=8.5, color=INK_SECONDARY)
tx, ty = kl(opt_logp, ref_logp).mean(), expected(opt_logp, reward).mean()
for name, (_, color, ls, _) in methods.items():
    r = results[name]
    ax.plot(r["kl_ref"], r["reward"], color=color, ls=ls, lw=1.0, alpha=0.6)
    ax.plot(r["kl_ref"][-1], r["reward"][-1], "o", color=color, ms=6.5, mec="white", mew=1, zorder=4)
ax.plot(tx, ty, "*", color=INK, ms=16, mec="white", mew=1, zorder=6)
ax.annotate(r"$\pi^\star$ at $\beta = 0.5$", (tx, ty), xytext=(0.08, 0.72), fontsize=9.5,
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
grpo_k3 = results["GRPO as published ($k_3$ in the loss)"]
ax.annotate("GRPO, $k_3$ in the loss:\na different target", (grpo_k3["kl_ref"][-1], grpo_k3["reward"][-1]),
            xytext=(1.2, 0.62), fontsize=8.5, color=INK_SECONDARY,
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
ax.plot(0, expected(ref_logp, reward).mean(), "s", color=INK_SECONDARY, ms=7, zorder=5)
ax.text(0.03, expected(ref_logp, reward).mean() - 0.03, r"$\pi_{\mathrm{ref}}$", fontsize=9.5, color=INK_SECONDARY)
ax.set_xlim(-0.03, 1.9)
ax.set_ylim(0.28, 0.9)
ax.set_xlabel(r"$\mathrm{KL}(\pi_\theta\,\|\,\pi_{\mathrm{ref}})$  (exact)")
ax.set_ylabel(r"expected reward $\mathbb{E}_{\pi_\theta}[r]$  (exact)")
ax.set_title("Where every run ends, in the reward–KL plane", fontsize=11.5)

# Bottom right: final distance to pi*, sorted.
ax = fig.add_subplot(bottom[1])
names = sorted(methods, key=lambda n: results[n]["kl_opt"][-1])
vals = [max(results[n]["kl_opt"][-1], FLOOR) for n in names]
ax.barh(range(len(names)), vals, color=[methods[n][1] for n in names], height=0.65)
ax.set_xscale("log")
ax.set_yticks(range(len(names)), [n.replace(" (teacher $= \\pi^\\star$)", "") for n in names], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel(r"final $\mathrm{KL}(\pi_\theta\,\|\,\pi^\star)$")
ax.set_title("Final distance to the target", fontsize=11.5)

savefig(fig, "all_roads_to_pi_star")
for n in names:
    r = results[n]
    print(f"{n:44s} KL->pi* {r['kl_opt'][-1]:.2e}  KLref {r['kl_ref'][-1]:.3f}  E[r] {r['reward'][-1]:.3f}")
print("pi*:", round(tx, 3), round(ty, 3))
