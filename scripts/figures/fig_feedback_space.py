"""The feedback space: where every method sits by what its signal says and who generated its data.

This figure makes visible Thesis 1 of the book: supervised learning, imitation, contextual bandits and
RL are regions of one space whose axes are the kind of feedback (instructive labels, comparisons,
evaluative rewards) and the data regime (a fixed dataset vs samples the learner generates), and every
LLM post-training method sits somewhere in it. Marker shape shows the third axis, the granularity of
credit (per sequence or per token).

Real computation: positions are read from the `feedback` and `data` fields of scripts/method_data.py
(the records behind the unified table); methods that share a position are grouped into one labelled
point.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
from _theme import FAMILY_COLOR, INK, INK_SECONDARY, MUTED, apply_theme, savefig
from method_data import M

apply_theme()
COLOR = {"classical": INK_SECONDARY, "policy_gradient": FAMILY_COLOR["policy_gradient"],
         "preference": FAMILY_COLOR["preference"], "imitation": FAMILY_COLOR["imitation"],
         "calibration": FAMILY_COLOR["calibration"]}

groups = {}
for m in M:
    if m["data"] is None:
        continue  # RLCD (TypeSafe): data regime undisclosed
    groups.setdefault((m["data"], m["feedback"]), []).append(m)

fig, ax = plt.subplots(figsize=(12.5, 7.2))
regions = [
    (-0.08, 0.25, -0.08, 0.12, "supervised learning / imitation"),
    (0.6, 1.3, 0.55, 1.1, "contextual bandit / RL\n(evaluative, self-generated)"),
    (-0.08, 0.25, 0.8, 1.1, "offline (evaluative, fixed data)"),
    (-0.08, 0.25, 0.3, 0.62, "offline preference learning"),
]
for x0, x1, y0, y1, lab in regions:
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color=MUTED, alpha=0.08, lw=0))
    ax.text(x0 + 0.01, y1 - 0.01, lab, fontsize=8.5, color=INK_SECONDARY, va="top", style="italic")
XMAP = {0.0: 0.0, 0.5: 0.38, 0.85: 0.70, 1.0: 1.0}  # spread the data-regime positions
for (x, y), ms in sorted(groups.items()):
    xp = XMAP[x]
    right = xp < 0.6 or xp == 1.0  # label side: left of the slightly-off-policy column, else right
    for i, m in enumerate(sorted(ms, key=lambda m: (m["family"], m["short"]))):
        yy = y - i * 0.052
        token = m["credit"].startswith(("token", "per step"))
        ax.scatter([xp], [yy], s=120, marker="s" if token else "o", color=COLOR[m["family"]],
                   edgecolor="white", lw=1.2, zorder=3)
        ax.text(xp + (0.025 if right else -0.025), yy, m["short"], ha="left" if right else "right",
                va="center", fontsize=8.6, color=INK, zorder=4)
ax.set_xlim(-0.1, 1.32)
ax.set_ylim(-0.1, 1.13)
ax.set_xticks([0, 0.38, 0.70, 1.0], ["fixed dataset\n(offline)", "iterated /\nfiltered", "slightly off-policy\n(batch reuse)", "on-policy"])
ax.set_yticks([0, 0.25, 0.5, 1.0], ["instructive\n(labels, demos)", "teacher log-probs", "comparisons /\nbinary labels", "evaluative\n(rewards)"])
ax.set_xlabel("who generated the training data")
ax.set_ylabel("what the feedback says")
ax.set_title("The feedback space: every method in the book (circle = sequence-level credit, square = token-level)")
handles = [plt.Line2D([0], [0], marker="o", ls="", ms=9, mfc=c, mec="white", label=lab)
           for lab, c in (("classical RL", COLOR["classical"]), ("policy gradients", COLOR["policy_gradient"]),
                          ("direct preference", COLOR["preference"]), ("imitation / distillation", COLOR["imitation"]),
                          ("calibration rewards", COLOR["calibration"]))]
ax.legend(handles=handles, fontsize=8.5, loc="lower right")
savefig(fig, "feedback_space")
