"""What going offline costs DPO, what IPO changes, and when the chosen response's likelihood falls.

This figure makes visible that (1) DPO on a fixed dataset approaches pi* and then drifts away as it
overfits the dataset's (often deterministic) empirical preferences, worse with less data, while
online DPO, which labels fresh pairs from the current policy, keeps converging; (2) IPO does not
drift, because it regresses each log-ratio gap to a bounded target, but it converges to a
*different* distribution, the reference tilted by the preference probability rather than the
reward; (3) with a policy whose responses share features, DPO can lower the log-probability of the
preferred responses it is trained on while widening the margin ("likelihood displacement").

Real computation: rl4llm.methods.dpo (offline, online and IPO variants, exact KL to pi* and to IPO's
exact target, rl4llm.methods.dpo.ipo_target_logprobs); and rl4llm.loglinear for panel 3 (a
log-linear policy over per-position token features, fit to pi_ref by maximum likelihood, trained
with DPO on 100 pairs per prompt whose rejected response differs from the chosen one by a single
token; median and range over 10 random pair sets. With larger pair sets the effect still appears
in most, but not all, draws).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
from _theme import CATEGORICAL, FAMILY_COLOR, INK, INK_SECONDARY, MUTED, apply_theme, savefig

from rl4llm.loglinear import LogLinearPolicy, one_token_edits
from rl4llm.methods import dpo
from rl4llm.metrics import kl
from rl4llm.toy_language import default_testbed, optimal_logprobs, sequence_logprobs

apply_theme()
lang, ref, reward = default_testbed()
ref_logp = sequence_logprobs(lang, ref)
BETA, TAU, STEPS, EVERY = 0.5, 0.5, 800, 10
ipo_target = dpo.ipo_target_logprobs(ref_logp, reward, TAU)
common = {"steps": STEPS, "eval_every": EVERY, "batch_size": 512, "lr_final_frac": 0.02}
OR = FAMILY_COLOR["preference"]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
ax = axes[0]
for n, ls in ((5_000, ":"), (20_000, "--"), (100_000, "-.")):
    _, h = dpo.train(lang, ref, reward, BETA, n_pairs_per_prompt=n, **common)
    ax.plot(h.step, h.kl_to_opt, color=OR, ls=ls, label=f"offline DPO, {n // 1000}k pairs/prompt")
_, h = dpo.train(lang, ref, reward, BETA, online=True, **common)
ax.plot(h.step, h.kl_to_opt, color=OR, lw=2.4, label="online DPO (fresh pairs from $\\pi_\\theta$)")
ax.set_yscale("log")
ax.set_xlabel("training step")
ax.set_ylabel(r"$\mathrm{KL}(\pi_\theta\,\|\,\pi^\star)$  (exact)")
ax.set_title("Offline DPO drifts; online DPO does not")
ax.legend(fontsize=8.2, loc="lower left")

ax = axes[1]
ipo_dist = []


def record_ipo(step, logits, data):
    ipo_dist.append((step, kl(sequence_logprobs(lang, logits), ipo_target).mean()))


_, h = dpo.train(lang, ref, reward, BETA, n_pairs_per_prompt=20_000, variant="ipo", tau=TAU,
                 callback=record_ipo, **common)
s, d = np.array(ipo_dist).T
ax.plot(h.step, h.kl_to_opt, color=OR, ls="--", label=r"IPO, 20k pairs: KL to DPO's $\pi^\star$")
ax.plot(s, d, color=OR, ls="-.", label="IPO, 20k pairs: KL to IPO's own target")
ax.axhline(kl(ipo_target, optimal_logprobs(ref_logp, reward, BETA)).mean(), color=MUTED, lw=0.9, ls=":")
ax.text(STEPS, 0.2, "distance between the two targets", ha="right", fontsize=8.3, color=INK_SECONDARY)
ax.set_yscale("log")
ax.set_xlabel("training step")
ax.set_ylabel("KL (exact)")
ax.set_title("IPO: no drift, but a different target")
ax.legend(fontsize=8.2, loc="center right")

ax = axes[2]
base = LogLinearPolicy(lang).fit_mle(ref_logp)
lin_ref = base.logprobs()
SEEDS_LL, N_PAIRS, STEPS_LL = range(10), 100, 600
runs = []
for seed in SEEDS_LL:
    rng = np.random.default_rng(seed)
    pol = LogLinearPolicy(lang)
    pol.theta = base.theta.copy()
    pr, yw, yl = one_token_edits(lang, reward, N_PAIRS, rng)
    traj = []
    for step in range(STEPS_LL + 1):
        lp = pol.logprobs()
        if step % 10 == 0:
            traj.append((lp[pr, yw].mean(), lp[pr, yl].mean(), np.log((np.exp(lp) * reward).sum(1)).mean()))
        pol.dpo_step(lin_ref, BETA, pr, yw, yl, lr=0.2)
    traj = np.array(traj)
    runs.append(traj - traj[0])
runs = np.array(runs)  # (seed, time, [chosen, rejected, logPcorrect])
steps_ll = np.arange(0, STEPS_LL + 1, 10)
series = [(runs[:, :, 0], FAMILY_COLOR["imitation"], "-", "preferred (correct) responses"),
          (runs[:, :, 1], CATEGORICAL[7], "-", "rejected (one token changed, wrong)"),
          (runs[:, :, 0] - runs[:, :, 1], INK, "--", "margin (preferred minus rejected)"),
          (runs[:, :, 2], MUTED, ":", "log P(correct) over all responses")]
for vals, c, ls, lab in series:
    ax.fill_between(steps_ll, vals.min(0), vals.max(0), color=c, alpha=0.12, lw=0)
    ax.plot(steps_ll, np.median(vals, 0), color=c, ls=ls, label=lab)
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xlabel("training step")
ax.set_ylabel("change in mean log-probability")
ax.set_title("Shared features: the preferred response often falls")
ax.legend(fontsize=8.2, loc="lower left")

savefig(fig, "direct_preference")
final = runs[:, -1, :]
print("displacement over seeds: d(chosen)", np.round(final[:, 0], 3), "d(rejected) median",
      np.round(np.median(final[:, 1]), 3), "margin median", np.round(np.median(final[:, 0] - final[:, 1]), 3),
      "logPcorrect median", np.round(np.median(final[:, 2]), 3))
print("IPO final: KL to pi*", round(h.kl_to_opt[-1], 4), "KL to own target", round(d[-1], 4))
