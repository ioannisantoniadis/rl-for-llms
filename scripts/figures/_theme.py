"""Shared matplotlib styling for every figure script in this directory.

Every ``fig_*.py`` script imports ``apply_theme()`` and the palette constants from here instead
of picking colors ad hoc, so the figures across all chapters read as one visual system. Figures
are rendered once as static PNGs into ``docs/images/`` and committed; Quarto does not execute
Python at render time (see ``scripts/figures/README.md``).

Palette: the same fixed categorical order, sequential ramp and surfaces as this author's other
Quarto books (loss-functions-lab, optimization-lab, modern-ai-systems-and-methods). The first five
categorical slots were run through the dataviz palette validator on 2026-10-01: lightness,
chroma, CVD separation and normal-vision floor all PASS on the light surface. Contrast vs the
surface is a WARN for aqua, yellow and magenta, so every multi-series figure must carry a legend
*and* direct labels or distinct line styles (identity never by color alone).

Book-specific addition: ``FAMILY_COLOR``. A method family keeps the same color in every figure
(e.g. on-policy policy-gradient methods are always blue), so a reader can track "the PPO/GRPO
family" or "the DPO family" across chapters. Variants within a family are told apart by line
style, never by inventing a new hue.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# Fixed categorical order: always index into this, never let matplotlib auto-cycle.
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

SEQUENTIAL_BLUE = [
    "#eaf2fc", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef",
    "#6da7ec", "#5598e7", "#3987e5", "#1f6dc9", "#0d366b",
]

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

# Method families -> fixed color (the book's map, Chapter 0, uses the same assignment).
FAMILY_COLOR = {
    "policy_gradient": CATEGORICAL[0],  # REINFORCE, RLOO, PPO, GRPO and its fixes
    "preference": CATEGORICAL[1],  # DPO, IPO, KTO, SimPO, ORPO
    "imitation": CATEGORICAL[2],  # SFT, best-of-N, rejection sampling FT, distillation
    "calibration": CATEGORICAL[4],  # proper-scoring-rule rewards (Chapter 10)
    "reference": INK,  # exact computations: pi*, the true gradient
}

FIGURE_DPI = 200
REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGES = REPO_ROOT / "docs" / "images"


def apply_theme() -> None:
    """Call once at the top of every fig_*.py before creating any figure."""
    plt.rcParams.update(
        {
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRIDLINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.grid": True,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 11,
            "axes.titlesize": 12.5,
            "axes.titleweight": "600",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "lines.linewidth": 2.0,
            "mathtext.fontset": "cm",
        }
    )


def savefig(fig, name: str) -> Path:
    """Save ``docs/images/<name>.png`` with the shared padding convention."""
    IMAGES.mkdir(parents=True, exist_ok=True)
    path = IMAGES / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {path.relative_to(REPO_ROOT)}")
    return path
