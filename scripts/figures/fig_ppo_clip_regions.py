"""PPO's clipped surrogate and its gradient, region by region.

This figure makes visible that the clip removes the gradient only where an update would push a
sample's probability ratio further in the direction the advantage already favors (A > 0 and ratio
above 1 + eps; A < 0 and ratio below 1 - eps), and keeps the full gradient where the ratio has
moved the *wrong* way, so a bad step can always be undone. DAPO's "clip-higher" (eps_high = 0.28)
only widens the upper limit for positive advantages.

Real computation: rl4llm.estimators.ppo_clip_objective and ppo_clip_grad_ratio (the derivative is
checked against finite differences in tests/test_estimators.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
from _theme import CATEGORICAL, FAMILY_COLOR, INK_SECONDARY, MUTED, apply_theme, savefig

from rl4llm.estimators import ppo_clip_grad_ratio, ppo_clip_objective

apply_theme()
EPS, EPS_HIGH = 0.2, 0.28
rho = np.linspace(0.4, 1.6, 1201)
blue = FAMILY_COLOR["policy_gradient"]

fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.6), sharex=True)
for col, A in enumerate((1.0, -1.0)):
    adv = np.full_like(rho, A)
    ax = axes[0, col]
    ax.plot(rho, rho * A, color=MUTED, ls="--", lw=1.4, label=r"unclipped $\rho A$")
    ax.plot(rho, ppo_clip_objective(rho, adv, EPS), color=blue, label=r"PPO: $\min(\rho A,\ \mathrm{clip}(\rho)A)$")
    if A > 0:
        ax.plot(rho, ppo_clip_objective(rho, adv, EPS, EPS_HIGH), color=CATEGORICAL[6], ls=":",
                label=r"DAPO clip-higher ($\epsilon_{\mathrm{high}}=0.28$)")
    ax.set_title(f"Advantage $A = {A:+.0f}$" + (": make this response likelier" if A > 0 else ": make it less likely"))
    ax.set_ylabel("surrogate objective")
    ax.legend(fontsize=8.3, loc="upper left" if A > 0 else "upper right")
    ax = axes[1, col]
    g = ppo_clip_grad_ratio(rho, adv, EPS)
    ax.plot(rho, g, color=blue)
    if A > 0:
        ax.plot(rho, ppo_clip_grad_ratio(rho, adv, EPS, EPS_HIGH), color=CATEGORICAL[6], ls=":")
    zero = (g == 0)
    ax.fill_between(rho, -1.3, 1.3, where=zero, color=CATEGORICAL[7], alpha=0.10, lw=0)
    ax.text(1.42 if A > 0 else 0.58, 0.4,
            "gradient\nclipped to 0", ha="center", va="center", fontsize=9, color=CATEGORICAL[7])
    ax.text(0.62 if A > 0 else 1.38, 0.55 * A, "full gradient kept:\nratio moved the\nwrong way, so the\nstep can be undone",
            ha="center", va="center", fontsize=8.5, color=INK_SECONDARY)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel(r"probability ratio $\rho = \pi_\theta(y_t \mid s_t)\,/\,\pi_{\mathrm{old}}(y_t \mid s_t)$")
    ax.set_ylabel(r"$\partial\,\mathrm{objective} / \partial\rho$")
    for a in (axes[0, col], ax):
        a.axvline(1, color=INK_SECONDARY, lw=0.7)
        for b in (1 - EPS, 1 + EPS):
            a.axvline(b, color=MUTED, lw=0.7, ls=":")

savefig(fig, "ppo_clip_regions")
