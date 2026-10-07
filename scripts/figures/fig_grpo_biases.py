"""What GRPO's group statistics do: zero-signal groups, difficulty reweighting, and length bias.

This figure makes visible three consequences of computing a baseline from a group of G samples per
prompt: (1) a group whose rewards are all equal carries exactly zero signal, and with 0/1 rewards
this is the common case for very hard and very easy prompts unless G is large (DAPO's dynamic
sampling drops such groups); (2) dividing by the group's standard deviation reweights prompts by
difficulty (leave-one-out keeps every prompt at weight 1); (3) normalizing each response's loss by
its own length (GRPO's 1/|o_i|) gives short correct answers larger per-token updates and penalizes
long wrong answers less, so correct answers shrink and wrong answers grow relative to a constant
normalizer (Dr. GRPO).

Real computation. Panel 1: P(all G rewards equal) = p^G + (1-p)^G, with the testbed's prompts marked
at their exact pi_ref success rates. Panel 2: the mean of 1,500 sampled gradient estimates per
policy, projected onto the exact per-prompt gradient (rl4llm.experiments_gradvar), at policies
pi*_beta of varying difficulty. Panel 3: GRPO training on the variable-length (EOS) testbed with
per-sequence vs constant normalization, no KL term, exact mean lengths, median of 3 seeds.
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

from rl4llm.experiments_gradvar import effective_prompt_weights
from rl4llm.methods import grpo
from rl4llm.metrics import expected
from rl4llm.toy_language import (
    default_testbed,
    eos_testbed,
    logits_from_sequence_logprobs,
    optimal_logprobs,
    sequence_logprobs,
)

apply_theme()
lang, ref, reward = default_testbed()
ref_logp = sequence_logprobs(lang, ref)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))

# Panel 1: zero-variance groups.
ax = axes[0]
p = np.linspace(0, 1, 401)
for G, c in zip((2, 4, 8, 16, 64), (SEQUENTIAL_BLUE[3], SEQUENTIAL_BLUE[5], SEQUENTIAL_BLUE[7], SEQUENTIAL_BLUE[8], SEQUENTIAL_BLUE[9])):
    ax.plot(p, p**G + (1 - p) ** G, color=c, label=f"$G = {G}$")
pr = expected(ref_logp, reward)
for x in pr:
    ax.plot(x, x**8 + (1 - x) ** 8, "o", color=CATEGORICAL[1], ms=7, mec="white", zorder=5)
ax.text(0.03, 0.72, "testbed prompts under $\\pi_{\\mathrm{ref}}$\n(orange, at $G = 8$)", fontsize=8.5, color=INK_SECONDARY)
ax.set_xlabel("P(correct) for the prompt")
ax.set_ylabel("P(all $G$ rewards equal) = P(zero signal)")
ax.set_title("A group with no variation carries no signal")
ax.legend(fontsize=8.5, loc="upper center")

# Panel 2: effective per-prompt weight vs difficulty.
ax = axes[1]
pts = {"grpo": [], "rloo": [], "mean": []}
for b in (None, 3.0, 1.0, 0.6, 0.3, 0.18, 0.12):
    L = ref if b is None else logits_from_sequence_logprobs(lang, optimal_logprobs(ref_logp, reward, b))
    pc = expected(sequence_logprobs(lang, L), reward)
    for kind, acc in pts.items():
        w = effective_prompt_weights(lang, L, reward, 8, kind, n_draws=1500)
        for x, y in zip(pc, w):
            if 0.01 < x < 0.99:  # gradients vanish at the extremes; ratios become noise
                acc.append((x, y))
styles = {"grpo": (FAMILY_COLOR["policy_gradient"], "o", "GRPO: $(r - \\bar r)/\\mathrm{std}$"),
          "mean": (CATEGORICAL[3], "^", "group mean incl. self (Dr. GRPO)"),
          "rloo": (INK, "s", "leave-one-out (RLOO)")}
for kind, (c, mk, lab) in styles.items():
    a = np.array(pts[kind])
    ax.scatter(a[:, 0], a[:, 1], color=c, marker=mk, s=26, label=lab, zorder=3)
ax.axhline(1, color=MUTED, lw=0.8)
ax.axhline(7 / 8, color=CATEGORICAL[3], lw=0.8, ls=":")
ax.set_xlabel("P(correct) for the prompt")
ax.set_ylabel("effective weight on the prompt's true gradient")
ax.set_title("Group normalization reweights prompts ($G = 8$)")
ax.legend(fontsize=8.3, loc="upper center")
ax.set_ylim(0, 3.2)

# Panel 3: length bias on the EOS testbed.
ax = axes[2]
elang, eref, er = eos_testbed()
STEPS = 400


def run(aggregation, advantage, seed):
    out = []

    def cb(step, logits):
        q = np.exp(sequence_logprobs(elang, logits))
        wrong, right = q * (1 - er), q * er
        out.append(((wrong * elang.lengths).sum(1) / wrong.sum(1)).mean().item())
        out.append(((right * elang.lengths).sum(1) / right.sum(1)).mean().item())

    grpo.train(elang, eref, er, 0.0, steps=STEPS, kl="none", aggregation=aggregation, advantage=advantage,
               group_size=8, lr=0.02, seed=seed, eval_every=10, lengths_fn=lambda s: elang.lengths[s], callback=cb)
    a = np.array(out).reshape(-1, 2)
    return a


steps = np.arange(0, STEPS + 1, 10)
for agg, adv, ls, lab in (("seq_mean", "grpo", "-", "GRPO (1/|o| per response)"),
                          ("constant", "dr_grpo", "--", "Dr. GRPO (constant normalizer)")):
    runs = np.array([run(agg, adv, s) for s in (0, 1, 2)])
    med = np.median(runs, 0)
    print(f"{lab}: length at step 0 wrong {med[0, 0]:.2f} correct {med[0, 1]:.2f}; "
          f"final wrong {med[-1, 0]:.2f} correct {med[-1, 1]:.2f}")
    ax.plot(steps, med[:, 0], color=CATEGORICAL[7], ls=ls, label=f"{lab}: wrong answers")
    ax.plot(steps, med[:, 1], color=FAMILY_COLOR["imitation"], ls=ls, label=f"{lab}: correct answers")
ax.set_xlabel("training step")
ax.set_ylabel("mean response length (tokens, exact)")
ax.set_title("Per-response length normalization biases length")
ax.legend(fontsize=7.8, loc="lower left")

# Not plotted: the constant normalizer alone (std division kept), isolating the aggregation's effect.
med = np.median([run("constant", "grpo", s) for s in (0, 1, 2)], 0)
print(f"GRPO with a constant normalizer (std kept): final wrong {med[-1, 0]:.2f} correct {med[-1, 1]:.2f}")

savefig(fig, "grpo_biases")
a = np.array(pts["grpo"])
print(f"GRPO weights: min {a[:, 1].min():.2f} max {a[:, 1].max():.2f}")
print("testbed prompts zero-signal prob at G=8:", np.round(pr**8 + (1 - pr) ** 8, 3))
