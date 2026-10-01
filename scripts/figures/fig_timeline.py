"""Timeline of the ideas in the book, from REINFORCE (1992) to RLCD (2026).

This figure makes visible that the lineage is a handful of classical ideas recombined over three
decades, with a burst of named variants after 2023 once the KL-regularized target and verifiable rewards
made new routes to it cheap to try.

Real computation: the dated milestones in scripts/method_data.py (TIMELINE), one row per milestone in
date order, colored by family; and the per-year count of methods in the unified table.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collections import Counter

import matplotlib.pyplot as plt
from _theme import FAMILY_COLOR, INK, INK_SECONDARY, MUTED, apply_theme, savefig
from method_data import TIMELINE, M

apply_theme()
COLOR = {"classical": INK_SECONDARY, "policy_gradient": FAMILY_COLOR["policy_gradient"],
         "preference": FAMILY_COLOR["preference"], "imitation": FAMILY_COLOR["imitation"],
         "calibration": FAMILY_COLOR["calibration"]}
LANE = {"calibration": 4, "policy_gradient": 3, "classical": 2, "preference": 1, "imitation": 0}
NAME = {"calibration": "calibration rewards", "policy_gradient": "policy gradients", "classical": "classical RL",
        "preference": "direct preference", "imitation": "imitation / distillation"}


rows = sorted(TIMELINE, key=lambda t: (t[0], LANE[t[2]]))
fig, (ax, axb) = plt.subplots(1, 2, figsize=(14, 8.2), gridspec_kw={"width_ratios": [3.2, 1]})
n = len(rows)
prev_year = None
for i, (year, label, fam) in enumerate(rows):
    y = n - 1 - i
    if year != prev_year:
        ax.text(0.0, y, str(year), ha="right", va="center", fontsize=10, color=INK_SECONDARY, fontweight="bold")
        prev_year = year
    ax.plot([0.25], [y], "o", color=COLOR[fam], ms=10, mec="white", zorder=3)
    ax.text(0.45, y, label, ha="left", va="center", fontsize=9.5, color=INK)
ax.plot([0.25, 0.25], [-0.5, n - 0.5], color=MUTED, lw=1, zorder=1)
ax.set_xlim(-0.6, 6)
ax.set_ylim(-0.8, n - 0.2)
ax.axis("off")
handles = [plt.Line2D([0], [0], marker="o", ls="", ms=9, mfc=COLOR[f], mec="white", label=NAME[f]) for f in LANE]
ax.legend(handles=handles, fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 0.0))
ax.set_title("A dated lineage, from REINFORCE (1992) to RLCD (2026)", loc="left")

counts = Counter(m["year"] for m in M if m["year"])
years = sorted(counts)
axb.barh([str(y) for y in years], [counts[y] for y in years], color=MUTED, height=0.6)
axb.invert_yaxis()
axb.set_xlabel("methods in the unified table")
axb.set_title("Named variants per year", fontsize=11)
savefig(fig, "timeline")
