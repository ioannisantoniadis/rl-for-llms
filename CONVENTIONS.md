# Authoring conventions for rl-for-llms

This file is the single source of truth for how chapters in this book are written. It exists so
that chapters written at different times read as one book. Read
`docs/appendix-notation.qmd` first (it defines every symbol), then this file. `SPEC.md` is the
conceptual contract; this file turns it into mechanics.

Not part of the rendered book. Not referenced by `_quarto.yml`.

## What this book is, in one paragraph

Not a catalogue of post-training methods. One argument: RL differs from supervised learning in
where the data comes from and what the feedback says, not in the model or the loss (Thesis 1).
The policy gradient is weighted maximum likelihood on samples the learner chose (Thesis 2), so
almost every method is a choice of **sampling distribution $q$** and **per-sample weight $w$**.
Nearly every LLM preference/RL method targets one distribution, the KL-regularized optimum
$\pi^\star \propto \pi_{\text{ref}}\exp(r/\beta)$ (Thesis 3). LLM post-training is an unusually
easy RL problem in some ways and an unusually hard one in others (Thesis 4), and a method's name
hides six independent design choices (Thesis 5). Every chapter must leave the reader seeing the
continuity with the RL they learned as a student.

## The two standing rules (from `SPEC.md`)

1. **Research first, never write from memory.** Before a chapter is written, its primary sources
   are read and logged in `research-log.md` (URL, what was checked, what could not be verified).
   Equations, defaults (token aggregation, KL placement, estimator choice) and reported results
   come from the paper, not from a secondary summary and not from recall. Search for later work
   that corrects or disputes each method.
2. **It is an RL book.** The test for every paragraph: does it help the reader understand an RL
   technique, or how a modern technique relates to classical RL? If not, cut it or link to the
   sibling repo that owns the topic.

## Chapter file and frontmatter

`docs/chapters/NN-slug.qmd`, minimal frontmatter:

```yaml
---
title: "Human-Readable Chapter Title"
---
```

## The chapter template

Method and derivation chapters follow this shape. Framing and synthesis chapters (the map, Ch 1,
Ch 4, Ch 11, Ch 12) may deviate, but **never drop the closing `## Connections` section**.

1. **Opening paragraph(s), no header.** The problem in plain language, and which previous
   method's limitation motivates this one.
2. **`## The problem`**: what exactly is being optimized or estimated, and what is still missing.
3. **`## Assumptions`**: stated as assumptions, with what breaks if each is wrong.
4. **`## Derivation`**: the actual math, step by step. Never skip a *conceptual* step.
5. **The resulting objective, boxed:**
   ```markdown
   ::: {.callout-note title="Definition — <Name>"}
   $$ ... $$
   :::
   ```
6. **Method card, immediately after the definition box** (see below), if the chapter introduces
   a method.
7. **`## Interpretation`**: what does optimizing this *actually* produce? (Which distribution?
   $\pi^\star$? Something sharper? The empirical optimum of a finite dataset?)
8. **`## Behavior and failure modes`**: with at least one computed figure where a figure helps.
9. **`## Connections`**: names 2–4 neighboring chapters and says *why* each is related.

Long chapters (2,500–4,000 words of prose; equations, code and figures don't count) are split
into clearly headed `##` sections so they can be read in sittings. The template's headers may
appear once per method inside such a chapter (e.g. Ch 7 has a short problem/derivation/card
sequence per DPO variant) rather than once per chapter.

## The method card

Every method gets a card, a `callout-tip`, immediately after its definition box. The fields are
fixed and always in this order, because the unified table (`SPEC.md` §7) is assembled from them:

```markdown
::: {.callout-tip title="Method card — <Name>"}
| | |
|---|---|
| **Signal** | ... |
| **Samples from** ($q$) | ... |
| **Per-sample weight** ($w$) | ... |
| **Stability** | ... |
| **Credit granularity** | ... |
| **Models in memory** | ... |
| **Targets** | ... |
| **Classical lineage** | ... |
:::
```

Every field is filled from the primary source. If a field cannot be determined (RLCD), write
"undisclosed" rather than guessing. If the method does not fit the $(q, w)$ form cleanly, say so
in the card ("does not fit: ...") and explain why in the prose; that is itself informative.

The implementation in `src/rl4llm/methods/` carries the same card in its docstring. Keep the two
consistent.

## Classical-lineage callouts

Whenever an LLM method reuses a classical RL idea, add a short callout naming it. This is the
book's main device for connecting the reader's student-era RL to the new vocabulary:

```markdown
::: {.callout-important title="Classical lineage — this is a Monte Carlo baseline"}
One to three sentences: the classical idea, where the reader met it (Sutton & Barto chapter, or
an earlier chapter), and what changed in the LLM setting.
:::
```

Use the title pattern "Classical lineage — this is <idea>" verbatim, so the callouts are easy to
find and list.

## Evidence levels

Every empirical or frontier claim is one of three kinds, and the prose says which:

- **Mathematical fact**: derived in the book or proved in a cited paper. State plainly.
- **Replicated empirical finding**: reported by several independent groups. "Several groups
  report ..." with citations.
- **Contested or recent claim**: one paper, a vendor, or an open debate. "The paper reports ...",
  dated ("as of 2026"), with the counter-evidence if any exists.

Vendor performance claims (RLCD/Jev in particular) are always attributed and never restated as
fact. A first-principles reconstruction is never presented as a vendor's method; it gets a
separate callout titled "A first-principles reconstruction, not the vendor's algorithm".

## Distinctions to never blur

From `SPEC.md` §13: RL (the problem) vs an RL algorithm vs RLHF (a pipeline); reward vs return vs
value vs advantage; reward model vs verifier vs judge vs critic; on-policy vs off-policy vs
offline (say which one each method is); forward vs reverse KL, and KL-in-reward vs KL-in-loss;
optimizing the KL-regularized objective vs reaching its optimum ("equivalent in the limit" is not
"equivalent in practice"); mathematical result vs replicated finding vs contested claim;
token-level vs sequence-level ratios, advantages and KLs.

Two more found during Phase 1 research:

- **An unbiased estimate of a value is not an unbiased estimate of its gradient.** ($k_3$ is
  unbiased for $\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$ under on-policy samples; the gradient of
  $k_3$ used as a loss is not $\nabla\mathrm{KL}$.)
- **KL to $\pi_{\text{old}}$ (a trust region) vs KL to $\pi_{\text{ref}}$ (a regularizer toward a
  fixed prior).** Original PPO's KL-penalty variant is the former; RLHF's is the latter.

## Math and code mechanics

- Inline `$...$`, display `$$...$$`, using only the symbols in `docs/appendix-notation.qmd`. A new
  symbol goes into the appendix in the same change.
- **Quarto does not execute code in this project.** Fenced `python` blocks are display-only:
  short excerpts of the real implementations in `src/rl4llm/`, kept in sync by hand. Actual
  computation lives in `scripts/figures/fig_*.py` and produces committed PNGs.
- Citations: `[@bibkey]` against `docs/references.bib`. Every entry is checked against the source
  and logged in `research-log.md`. Never fabricate an author list, year, venue or result.
- **Cross-reference chapters by title and link, never by bare number in running prose** ("the
  [policy-gradient chapter](02-policy-gradients.qmd)", not "Chapter 2"), because numbering shifts
  if chapters move (a lesson from `math-conceptual-map`). Method cards and tables may use
  "Ch 6" style shorthand, because they are regenerated with the unified table.
- Date-stamp anything frontier: "as of October 2026".

## Figures

- One script per figure: `scripts/figures/fig_<slug>.py`. It starts by importing the shared theme
  and ends with `savefig(fig, "<slug>")`, which writes `docs/images/<slug>.png`:
  ```python
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).resolve().parent))
  from _theme import apply_theme, savefig, FAMILY_COLOR, CATEGORICAL, INK_SECONDARY

  apply_theme()
  ```
- Figures that run an experiment import from `rl4llm` (the testbed and methods), so a figure is
  always the real code, never a re-implementation. Run with `uv run python scripts/figures/fig_<slug>.py`.
- **Every figure is a real computation**, never decoration. Before writing one, be able to finish
  the sentence "this figure makes visible that ...", and put that sentence in the script's
  docstring.
- **Color follows the method family** (`FAMILY_COLOR` in `_theme.py`): policy-gradient methods
  blue, preference methods orange, imitation/distillation aqua, calibration magenta, exact
  reference computations black. Variants inside a family differ by line style. Every
  multi-series figure has a legend *and* direct labels or distinct line styles (the palette's
  contrast check is a WARN for three slots).
- Embed with an id and a caption that states the lesson:
  ```markdown
  ![One sentence on what is plotted. One sentence on the lesson.](../images/<slug>.png){#fig-<slug>}
  ```
- Run the script and **look at the PNG** before the chapter is considered done (label collisions
  and clipped text are only caught by looking).
- Stochastic experiments use fixed seeds and show spread over seeds (median plus a band) when the
  claim is about typical behavior.

## Code

- `src/rl4llm/` is numpy + scipy only. Methods are 50–150 lines, written to be read, with the
  method card in the module docstring.
- Every method ends in `grad_log_likelihood(lang, logits, prompts, responses, weights)`. Keep it
  that way: it is Thesis 2 made executable.
- Tests check derivations numerically against exact enumeration (`tests/`), as in
  `optimization-lab`. A new derivation claimed in the book gets a test when it is checkable.
- `uv run ruff check .` and `uv run pytest` both pass before a chapter is called done.

## File map

Chapter files, appendices and their order are fixed in `docs/_quarto.yml`. Do not rename or
reorder without updating it and `ROADMAP.md`.
