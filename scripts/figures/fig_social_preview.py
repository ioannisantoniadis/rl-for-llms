"""1280x640 social-preview banner (README hero / GitHub link preview), generated from the method lineage.

Not a teaching figure: it re-renders the lineage graph of fig_method_lineage.py at banner size with
the book's title baked in, as the sibling repositories do.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import networkx as nx
from _theme import FAMILY_COLOR, INK, INK_SECONDARY, MUTED, SURFACE, apply_theme
from method_data import M

apply_theme()
COLOR = {"classical": INK_SECONDARY, "policy_gradient": FAMILY_COLOR["policy_gradient"],
         "preference": FAMILY_COLOR["preference"], "imitation": FAMILY_COLOR["imitation"],
         "calibration": FAMILY_COLOR["calibration"]}
LANE = {"classical": 2.0, "policy_gradient": 3.2, "preference": 0.8, "imitation": -0.4, "calibration": 4.4}
G = nx.DiGraph()
for m in M:
    G.add_node(m["name"], **m)
for m in M:
    if m["parent"]:
        G.add_edge(m["parent"], m["name"])
depth = {}
for n in nx.topological_sort(G):
    preds = list(G.predecessors(n))
    depth[n] = 0 if not preds else 1 + max(depth[p] for p in preds)
pos = {}
groups = {}
for m in M:
    groups.setdefault((m["family"], depth[m["name"]]), []).append(m["name"])
for (fam, d), names in groups.items():
    for i, n in enumerate(sorted(names)):
        pos[n] = (d, LANE[fam] + ((len(names) - 1) / 2 - i) * 0.42)

fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
fig.patch.set_facecolor(SURFACE)
ax = fig.add_axes([0.38, 0.06, 0.6, 0.88])
ax.axis("off")
nx.draw_networkx_edges(G, pos, ax=ax, edge_color=MUTED, width=1.0, arrows=True, arrowsize=8,
                       connectionstyle="arc3,rad=0.08", node_size=300)
for m in M:
    x, y = pos[m["name"]]
    ax.scatter([x], [y], s=120, color=COLOR[m["family"]], edgecolor="white", linewidth=1, zorder=3)
    # A surface-colored backing keeps edges that leave a node from striking through its label.
    ax.text(x + 0.08, y, m["short"], fontsize=7.5, va="center", color=INK, zorder=4,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": SURFACE, "edgecolor": "none"})
fig.text(0.04, 0.70, "Reinforcement Learning\nfor Language Models,\nfrom First Principles", fontsize=27,
         fontweight="bold", color=INK, va="top")
fig.text(0.04, 0.36, "RLHF, PPO, DPO, GRPO and their relatives\nas one idea with different approximations,\n"
         "checked exactly on an enumerable toy language.", fontsize=12.5, color=INK_SECONDARY, va="top")
fig.text(0.04, 0.08, "rl-for-llms", fontsize=11, color=MUTED)
out = Path(__file__).resolve().parents[2] / "docs" / "images" / "social_preview.png"
fig.savefig(out, dpi=200, facecolor=SURFACE)
print(f"wrote {out.relative_to(Path(__file__).resolve().parents[2])}")
