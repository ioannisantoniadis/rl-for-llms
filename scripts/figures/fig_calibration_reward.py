"""Correctness reward vs a proper-scoring-rule reward: what each makes the policy say about itself.

This figure makes visible that (1) a binary correctness reward improves accuracy but leaves the
stated confidence wherever the reference policy put it (here, overconfident), so confident wrong
answers stay common; (2) a Brier score alone makes the stated confidence honest but does not care
whether the answer is right, and drifts toward answers that are predictably wrong; (3) correctness
plus the Brier score gives nearly the accuracy of the binary reward *and* honest probabilities, which
makes abstention (answer only when the stated probability is high) actually work.

Real computation: the answer-plus-confidence task of rl4llm.calibration (a first-principles
reconstruction, not any vendor's method), trained with RLOO on sampled outcomes, 2,500 steps,
16 samples per prompt, median of 3 seeds; reliability, accuracy and confident-wrong rates are exact
expectations under each trained policy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
from _theme import CATEGORICAL, FAMILY_COLOR, INK_SECONDARY, MUTED, apply_theme, savefig

from rl4llm.calibration import BUCKETS, CalibrationTask, train
from rl4llm.toy_language import sequence_logprobs

apply_theme()
task = CalibrationTask()
STEPS, G, SEEDS = 2500, 16, (0, 1, 2)
style = {
    "reference": ("reference policy", MUTED, "s"),
    "binary": ("binary correctness reward", FAMILY_COLOR["policy_gradient"], "o"),
    "brier": ("Brier score only", CATEGORICAL[3], "^"),
    "combined": ("correctness + Brier", FAMILY_COLOR["calibration"], "D"),
}
policies = {"reference": [task.ref_logits]}
for kind in ("binary", "brier", "combined"):
    policies[kind] = [train(task, kind, steps=STEPS, group_size=G, seed=s, eval_every=STEPS)[0] for s in SEEDS]


def summary(logits_list):
    ms = [task.metrics(lg) for lg in logits_list]
    return {k: np.nanmedian([m[k] for m in ms], 0) for k in ms[0]}


def risk_coverage(logits_list):
    curves = []
    for lg in logits_list:
        prob = np.exp(sequence_logprobs(task.lang, lg))
        cov, acc = [], []
        for tau in BUCKETS:
            keep = task.conf[None, :] >= tau - 1e-9
            c = (prob * keep).sum(1).mean()
            a = (prob * keep * task.q).sum(1).mean() / max(c, 1e-12)
            cov.append(c)
            acc.append(a)
        curves.append((cov, acc))
    return np.median(np.array(curves), 0)


S = {k: summary(v) for k, v in policies.items()}
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.05, 1, 1.05]})

ax = axes[0]
ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="perfect calibration")
for k, (lab, c, mk) in style.items():
    m = S[k]
    ok = m["bucket_mass"] > 0.005
    ax.plot(BUCKETS[ok], m["bucket_accuracy"][ok], color=c, lw=1.2, alpha=0.7)
    ax.scatter(BUCKETS[ok], m["bucket_accuracy"][ok], s=30 + 400 * m["bucket_mass"][ok], color=c, marker=mk,
               edgecolor="white", zorder=3, label=lab)
ax.set_xlabel("stated probability of being correct")
ax.set_ylabel("actual probability of being correct")
ax.set_title("Reliability (marker area = share of answers)")
ax.legend(fontsize=8.2, loc="upper left")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax = axes[1]
names = list(style)
metrics = [("accuracy", "accuracy"), ("ece", "calibration error (ECE)"), ("confident_wrong", "confident and wrong\n(stated ≥ 0.7, actually wrong)")]
x = np.arange(len(metrics))
wbar = 0.2
for i, k in enumerate(names):
    vals = [S[k][m] for m, _ in metrics]
    ax.bar(x + (i - 1.5) * wbar, vals, width=wbar, color=style[k][1], label=style[k][0])
    for xi, v in zip(x + (i - 1.5) * wbar, vals):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.3, color=INK_SECONDARY)
ax.set_xticks(x, [lab for _, lab in metrics], fontsize=9)
ax.set_ylabel("rate (exact)")
ax.set_title("What each reward produces")
ax.set_ylim(0, 0.8)

ax = axes[2]
for k in ("reference", "binary", "combined"):
    cov, acc = risk_coverage(policies[k])
    lab, c, mk = style[k]
    ax.plot(cov, acc, "-", marker=mk, color=c, label=lab)
ax.set_xlabel("share of questions answered (abstain below a confidence threshold)")
ax.set_ylabel("accuracy on the questions answered")
ax.set_title("Calibration is what makes abstention work")
ax.legend(fontsize=8.5, loc="center left")
ax.invert_xaxis()

savefig(fig, "calibration_reward")
for k in names:
    print(f"{k:9s} acc {S[k]['accuracy']:.3f} ece {S[k]['ece']:.3f} confident_wrong {S[k]['confident_wrong']:.3f} "
          f"mean_conf {S[k]['mean_confidence']:.3f}")
for k in ("binary", "combined"):
    cov, acc = risk_coverage(policies[k])
    print(k, "coverage", np.round(cov, 2), "selective acc", np.round(acc, 2))
print("best achievable accuracy:", task.q.max(1).mean().round(3))
