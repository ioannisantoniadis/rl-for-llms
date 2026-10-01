"""The method lineage: every method in the book as a node, with an edge from the method it modifies.

This figure makes visible that the post-training zoo is one tree rooted in two classical ideas (the
policy gradient and behavior cloning), that after PPO-based RLHF it branches into a few families
(direct preference methods, critic-free policy gradients, filtered imitation and distillation,
calibration rewards), and that most named methods change exactly one thing about their parent.

Real computation: a networkx DiGraph built from the `parent` field of scripts/method_data.py (the same
records that generate the unified table), laid out by generation (x: the longest chain of modifications from a root) and family (y lane),
with the publication year in each label.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from _theme import FAMILY_COLOR, INK, INK_SECONDARY, MUTED, apply_theme, savefig
from method_data import M

apply_theme()
COLOR = {"classical": INK_SECONDARY, "policy_gradient": FAMILY_COLOR["policy_gradient"],
         "preference": FAMILY_COLOR["preference"], "imitation": FAMILY_COLOR["imitation"],
         "calibration": FAMILY_COLOR["calibration"]}
LANE = {"classical": 2.0, "policy_gradient": 3.2, "preference": 0.8, "imitation": -0.6, "calibration": 4.6}
LABEL = {"classical": "classical RL", "policy_gradient": "policy gradients", "preference": "direct preference",
         "imitation": "imitation / distillation", "calibration": "calibration rewards"}

G = nx.DiGraph()
for m in M:
    G.add_node(m["name"], **m)
for m in M:
    if m["parent"]:
        G.add_edge(m["parent"], m["name"])


# x = generation: the length of the longest chain of modifications from a root method.
depth = {}
for n in nx.topological_sort(G):
    preds = list(G.predecessors(n))
    depth[n] = 0 if not preds else 1 + max(depth[p] for p in preds)
pos = {m["name"]: np.array([depth[m["name"]], LANE[m["family"]]], dtype=float) for m in M}
groups = {}
for m in M:
    groups.setdefault((m["family"], depth[m["name"]]), []).append(m["name"])
for names in groups.values():
    names = sorted(names, key=lambda n: (G.nodes[n]["year"] or 0, n))
    for i, n in enumerate(names):
        pos[n][1] += ((len(names) - 1) / 2 - i) * 0.5

fig, ax = plt.subplots(figsize=(15, 8.4))
for fam, y in LANE.items():
    ax.axhspan(y - 0.6, y + 0.6, color=COLOR[fam], alpha=0.05, lw=0)
    ax.text(-1.25, y, LABEL[fam], ha="left", va="center", fontsize=9.5, color=COLOR[fam], fontweight="bold")
nx.draw_networkx_edges(G, pos, ax=ax, edge_color=MUTED, width=1.1, arrows=True, arrowsize=10,
                       connectionstyle="arc3,rad=0.08", node_size=500, min_target_margin=8)
for m in M:
    x, y = pos[m["name"]]
    ax.scatter([x], [y], s=150, color=COLOR[m["family"]], edgecolor="white", linewidth=1.2, zorder=3)
    yr = f" '{str(m['year'])[2:]}" if m["year"] else ""
    if depth[m["name"]] == 0:  # roots: label below, clear of the outgoing arrows
        ax.text(x, y - 0.2, m["short"] + yr, ha="center", va="top", fontsize=8.6, color=INK, zorder=4)
    else:
        ax.text(x + 0.09, y, m["short"] + yr, ha="left", va="center", fontsize=8.6, color=INK, zorder=4)
ax.set_xticks([])
for d, lab in enumerate(["roots", "1st modification", "2nd", "3rd", "4th", "5th"][: max(depth.values()) + 1]):
    ax.text(d, 5.55, lab, ha="center", va="bottom", fontsize=9, color=INK_SECONDARY)
ax.set_yticks([])
ax.set_xlim(-1.3, max(depth.values()) + 0.9)
ax.set_ylim(-1.7, 5.7)
ax.grid(axis="y", visible=False)
for s in ("left",):
    ax.spines[s].set_visible(False)
ax.set_title("The method lineage: each arrow points from a method to the one that modifies it", pad=10)
savefig(fig, "method_lineage")
