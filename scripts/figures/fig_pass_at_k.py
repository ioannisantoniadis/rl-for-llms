"""Pass@k after RL: when does RL raise pass@1 but lower pass@k at large k?

This figure makes visible that (1) on a single prompt, any change that raises the probability of a
correct answer raises pass@k = 1 - (1 - p)^k at *every* k, so per-prompt sharpening alone cannot
produce the crossover reported by Yue et al. (2025); with a fully expressive per-prompt policy the
KL-regularized optimum never falls below the reference; (2) a crossover appears when one set of
parameters serves several problems whose good answers compete: RL moves the shared distribution toward
what is rewarded most often, the success rate on the rest falls, and pass@k at large k drops below the
base model's even as pass@1 rises.

Real computation. Left: exact pass@k on the default testbed for pi_ref and pi*_beta. Middle and right: a
deliberately extreme "shared" setting, one distribution over the 625 responses serving five tasks
(target sum 0..4 mod 5) that appear in training with frequencies 0.40/0.30/0.15/0.10/0.05; exact
pi*_beta for the frequency-weighted reward, and an unregularized RLOO run on it; pass@k weighted by
the same task frequencies.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
from _theme import (
    FAMILY_COLOR,
    MUTED,
    SEQUENTIAL_BLUE,
    apply_theme,
    savefig,
)

from rl4llm.methods import reinforce
from rl4llm.metrics import pass_at_k
from rl4llm.toy_language import (
    ToyLanguage,
    default_testbed,
    make_reference,
    optimal_logprobs,
    sequence_logprobs,
)

apply_theme()
ks = np.unique(np.round(np.geomspace(1, 256, 30)).astype(int))
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1, 1, 0.85]})

# Left: per-prompt, fully expressive.
ax = axes[0]
lang, ref, reward = default_testbed()
rl = sequence_logprobs(lang, ref)
ax.plot(ks, pass_at_k(rl, reward, ks).mean(0), color=MUTED, lw=2.2, label=r"$\pi_{\mathrm{ref}}$ (base)")
for b, c in ((1.0, SEQUENTIAL_BLUE[5]), (0.5, SEQUENTIAL_BLUE[7]), (0.15, SEQUENTIAL_BLUE[9])):
    ax.plot(ks, pass_at_k(optimal_logprobs(rl, reward, b), reward, ks).mean(0), color=c, label=rf"$\pi^\star_\beta$, $\beta={b}$")
ax.set_xscale("log", base=2)
ax.set_xlabel("$k$ (samples per problem)")
ax.set_ylabel("pass@$k$ (exact, mean over prompts)")
ax.set_title("One policy per prompt: RL never hurts pass@$k$")
ax.legend(fontsize=8.5, loc="lower right")

# Middle/right: one shared distribution serving five tasks.
lang1 = ToyLanguage(n_prompts=1)
ref1 = make_reference(lang1, seed=1, concentration=0.6)
rl1 = sequence_logprobs(lang1, ref1)
sums = lang1.sequences.sum(1) % 5
correct = np.stack([(sums == t).astype(float) for t in range(5)])  # (task, N)
freq = np.array([0.40, 0.30, 0.15, 0.10, 0.05])
rbar = (freq[:, None] * correct).sum(0)[None, :]


def task_success(logp):
    return (np.exp(logp[0])[None, :] * correct).sum(1)


def weighted_pass(p):
    return (freq[:, None] * (1 - (1 - p[:, None]) ** ks[None, :])).sum(0)


ax = axes[1]
base_p = task_success(rl1)
ax.plot(ks, weighted_pass(base_p), color=MUTED, lw=2.2, label="base")
policies = {}
for b, c in ((0.3, SEQUENTIAL_BLUE[5]), (0.1, SEQUENTIAL_BLUE[7]), (0.03, SEQUENTIAL_BLUE[9])):
    p = task_success(optimal_logprobs(rl1, rbar, b))
    policies[rf"$\pi^\star_\beta$, $\beta={b}$"] = p
    ax.plot(ks, weighted_pass(p), color=c, label=rf"$\pi^\star_\beta$, $\beta={b}$")
logits_rl, _ = reinforce.train(lang1, ref1, rbar, 0.0, steps=1500, group_size=16, seed=0, eval_every=1500)
p_rl = task_success(sequence_logprobs(lang1, logits_rl))
policies["RLOO, no KL"] = p_rl
ax.plot(ks, weighted_pass(p_rl), color=FAMILY_COLOR["policy_gradient"], ls="--", lw=2.2, label="RLOO, no KL (sampled)")
ax.set_xscale("log", base=2)
ax.set_xlabel("$k$ (samples per problem)")
ax.set_ylabel("pass@$k$ (exact, weighted by task frequency)")
ax.set_title("One shared policy, five tasks: a crossover")
ax.legend(fontsize=8.5, loc="upper left")

ax = axes[2]
x = np.arange(5)
ax.bar(x - 0.2, base_p, width=0.4, color=MUTED, label="base")
ax.bar(x + 0.2, p_rl, width=0.4, color=FAMILY_COLOR["policy_gradient"], label="after RL (RLOO, no KL)")
ax.set_xticks(x, [f"task {t}\n{int(f * 100)}%" for t, f in enumerate(freq)], fontsize=9)
ax.set_xlabel("task (share of training prompts)")
ax.set_ylabel("P(correct) on the task (exact)")
ax.set_title("Success moves to the common tasks")
ax.legend(fontsize=8.5, loc="upper right")

savefig(fig, "pass_at_k")
print("base task p", base_p.round(3), "RL task p", p_rl.round(3))
print("weighted pass@1 base", weighted_pass(base_p)[0].round(3), "RL", weighted_pass(p_rl)[0].round(3),
      "| pass@256 base", weighted_pass(base_p)[-1].round(3), "RL", weighted_pass(p_rl)[-1].round(3))
cross = ks[np.argmax(weighted_pass(p_rl) < weighted_pass(base_p))]
print("first k where RL is below base:", cross)
