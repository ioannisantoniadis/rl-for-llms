"""Forward vs reverse KL: fitting one Gaussian to a two-mode target.

This figure makes visible that minimizing the reverse KL(q || p) (the direction in the
KL-regularized RL objective, where q is the policy being trained) locks onto one mode of the
target and ignores the other, while minimizing the forward KL(p || q) (the direction supervised
learning / maximum likelihood minimizes) spreads q over both modes, putting mass where the target
has almost none. Reverse KL also has two separate local minima, one per mode.

Real computation: both KLs are computed by numerical integration on a fine grid and minimized
over the Gaussian's mean and log-std with scipy (several starts for the reverse KL).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
from _theme import FAMILY_COLOR, INK, INK_SECONDARY, MUTED, apply_theme, savefig
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm

apply_theme()
x = np.linspace(-20, 22, 16001)
dx = x[1] - x[0]
p = 0.6 * norm.pdf(x, -2.0, 0.8) + 0.4 * norm.pdf(x, 3.5, 1.0)
EPS = 1e-300


def q_of(theta):
    return norm.pdf(x, theta[0], np.exp(theta[1]))


def reverse_kl(theta):  # KL(q || p)
    q = q_of(theta)
    return np.sum(q * (np.log(q + EPS) - np.log(p + EPS))) * dx


def forward_kl(theta):  # KL(p || q)
    q = q_of(theta)
    return np.sum(p * (np.log(p + EPS) - np.log(q + EPS))) * dx


fwd = minimize(forward_kl, [0.0, 0.0], method="Nelder-Mead").x
rev = [minimize(reverse_kl, [m, 0.0], method="Nelder-Mead").x for m in (-3.0, 4.0)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
ax1.fill_between(x, p, color=MUTED, alpha=0.25, lw=0)
ax1.plot(x, p, color=INK, lw=1.4, label="target $p$ (two modes)")
ax1.plot(x, q_of(fwd), color=FAMILY_COLOR["imitation"], lw=2.2,
         label=r"argmin forward $\mathrm{KL}(p\,\|\,q)$: covers both modes")
ax1.plot(x, q_of(rev[0]), color=FAMILY_COLOR["policy_gradient"], lw=2.2,
         label=r"argmin reverse $\mathrm{KL}(q\,\|\,p)$: picks one mode")
ax1.plot(x, q_of(rev[1]), color=FAMILY_COLOR["policy_gradient"], lw=1.6, ls="--",
         label="reverse KL, the other local minimum")
ax1.set_xlim(-6, 8)
ax1.set_xlabel("$y$")
ax1.set_ylabel("density")
ax1.set_title("Fitting one Gaussian $q$ to a bimodal target")
ax1.legend(fontsize=8.4, loc="upper right")

mus = np.linspace(-3.5, 5.0, 171)
best_rev = [minimize_scalar(lambda s, m=m: reverse_kl([m, s]), bounds=(-2.5, 1.5), method="bounded").fun for m in mus]
best_fwd = [minimize_scalar(lambda s, m=m: forward_kl([m, s]), bounds=(-2.5, 1.5), method="bounded").fun for m in mus]
ax2.plot(mus, best_fwd, color=FAMILY_COLOR["imitation"], label="forward KL")
ax2.plot(mus, best_rev, color=FAMILY_COLOR["policy_gradient"], label="reverse KL")
ax2.set_ylim(0, 2.0)
ax2.set_xlabel(r"mean of $q$  (best width at each mean)")
ax2.set_ylabel("KL (nats)")
ax2.set_title("The objectives along the mean")
ax2.legend(fontsize=9)
ax2.text(0.75, 1.55, "reverse KL: a separate\nminimum at each mode", fontsize=8.6, color=INK_SECONDARY, ha="center")

savefig(fig, "forward_vs_reverse_kl")
print("forward fit mean/std:", fwd[0].round(2), np.exp(fwd[1]).round(2),
      "| reverse fits:", [(r[0].round(2), np.exp(r[1]).round(2), round(reverse_kl(r), 3)) for r in rev])
