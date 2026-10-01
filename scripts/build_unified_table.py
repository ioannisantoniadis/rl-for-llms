"""Generate the unified table (Chapter 12 and the standalone page) from scripts/method_data.py.

Writes docs/includes/unified-table-full.md (every method-card field) and
docs/includes/unified-table-compact.md (the five fields Chapter 12 shows). Run from the repo root:
    uv run python scripts/build_unified_table.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from method_data import FIELDS, HEADERS, M

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "includes"
FAMILY = {"classical": "classical RL", "policy_gradient": "policy gradient", "preference": "preference",
          "imitation": "imitation / distillation", "calibration": "calibration"}


def cell(text: str) -> str:
    return str(text).replace("|", r"\|")


def table(fields, link_prefix):
    head = "| Method | Family | " + " | ".join(HEADERS[f] for f in fields) + " |"
    sep = "|" + "---|" * (len(fields) + 2)
    rows = []
    for m in M:
        name = f"[{m['name']}]({link_prefix}{m['chapter']}.qmd)"
        rows.append("| " + name + " | " + FAMILY[m["family"]] + " | " + " | ".join(cell(m[f]) for f in fields) + " |")
    return "\n".join([head, sep, *rows]) + "\n"


OUT.mkdir(parents=True, exist_ok=True)
(OUT / "unified-table-full.md").write_text(table(FIELDS, "chapters/"))
(OUT / "unified-table-compact.md").write_text(table(["signal", "q", "w", "targets", "changed"], ""))
print(f"wrote {len(M)} rows to docs/includes/")
