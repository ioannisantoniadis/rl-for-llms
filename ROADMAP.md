# Roadmap

Status, plan and decision log for the book. `SPEC.md` is the conceptual contract;
`research-log.md` is the audit trail of sources; this file records **what will be built, in what
order, and why each structural decision was made**.

Status markers: `planned` · `researching` · `drafted` · `done` (passes the §15 quality bar) ·
`blocked`.

---

## Phase 2: build (2026-10-01): **done** (see the Phase 2 log below)

## Phase 1: scope (2026-10-01): **done, approved**

- [x] Read `SPEC.md` end to end; read the sibling repos (`loss-functions-lab` in full;
      `modern-ai-systems-and-methods` Ch 10 and the RL parts of Ch 12, plus its landscape-graph
      script; `optimization-lab` `CLAUDE.md`, `pyproject.toml`, CI and tests layout;
      `transformer-atlas` `MAP.md`; `math-conceptual-map` content conventions).
- [x] Initial research pass: 38 core papers verified by arXiv API metadata, about 30 technical
      claims checked against paper text, RLCD re-searched. Everything is in `research-log.md`.
- [x] Scaffold: `README.md`, `CONVENTIONS.md`, `ROADMAP.md`, `CLAUDE.md`, `pyproject.toml` (uv),
      `docs/_quarto.yml`, `docs/appendix-notation.qmd`, `scripts/figures/_theme.py`,
      `.github/workflows/docs.yml` (Pages) and `ci.yml` (ruff + pytest), stub pages for every
      chapter and appendix. `quarto render docs` builds with no warnings.
- [x] Toy testbed `src/rl4llm/toy_language.py` with exact $\pi^\star$; `metrics.py`,
      `estimators.py`; methods `exact`, `reinforce` (none/mean/RLOO/GRPO baselines), `dpo`
      (sampled and population), `rft` (exact-acceptance rejection sampling + SFT).
- [x] The §10 tests and a few more: 17 tests pass (`uv run pytest`), `ruff check` clean.
- [x] First version of the signature figure, `docs/images/all_roads_to_pi_star.png`, with
      four methods (see "What the first figure showed" below).

### What the first figure showed (and why it changes the plan slightly)

With $\beta = 0.5$ and a verifiable 0/1 reward:

- **Exact gradient ascent** and **DPO on the population (infinite-pair) loss** reach $\pi^\star$
  to machine precision (KL ≈ 1e-7 to 1e-12). This is the framework working: two completely
  different objectives with the same optimum.
- **RLOO** (8 samples per prompt, KL in the reward) converges to a noise floor around
  KL ≈ 3e-2 set by gradient noise at a fixed learning rate. It keeps closing the gap slowly.
- **Rejection-sampling FT** (20k proposals per prompt) plateaus at KL ≈ 5e-2 and then drifts up
  slightly: SFT converges to the *empirical* distribution of a finite accepted sample, not to
  $\pi^\star$.
- **DPO on 20k sampled pairs per prompt** approaches $\pi^\star$ and then **drifts away**
  (KL ≈ 0.12 → 0.17). This is the overfitting Azar et al. (2023) predict: with a finite dataset
  many responses are always chosen or always rejected, so the empirical preference is
  deterministic and the implicit reward is pushed without bound.

So the figure says something sharper than "all roads lead to $\pi^\star$": **every route aims at
$\pi^\star$, and what separates them in practice is the estimator (noise, finite data, offline
drift), not the target.** That is exactly the spec's "equivalent in the limit is not equivalent
in practice" (§13). The final figure keeps both the exact/infinite-data and the practical versions
side by side. Its right panel (the reward–KL plane with the exact $\pi^\star_\beta$ frontier) also
previews Ch 5's overoptimization plots. No framework rethink is needed (§16, item 5).

---

## The chapter plan (proposed final)

**Decision: keep the spec's 12 chapters plus the map, in the spec's order. No merges, no splits.**
Reasons:

1. The research pass found nothing that changes the conceptual order. Every chapter's
   prerequisites are still satisfied by earlier chapters. In particular the spec's bottom-up
   order (policy gradient → trust region/KL → LLM mapping → rewards → the hub → routes to the hub)
   is what makes Ch 6's "all roads" argument land.
2. The densest chapter is Ch 8 (RLOO, ReMax, GRPO, Dr. GRPO, DAPO, GSPO, CISPO). It fits in
   ~4,000 words if each 2025 method is presented as *one bias and its fix* (which is how the
   papers themselves frame them; verified), not as a full method write-up. Splitting it would
   produce two chapters about GRPO. Keep.
3. The author approved 12 chapters deliberately (§4). Research did not produce a reason strong
   enough to override that.

**Changes within the structure** (all are additions or re-weightings inside existing chapters,
listed here so they are not silent):

| # | Change | Where | Why |
|---|---|---|---|
| C1 | New appendix **"The Toy Testbed"** documenting the enumerable language once; chapters link to it | Appendices | Every experiment chapter uses it; documenting it once keeps chapters on RL. First introduced in prose in Ch 2 (its first experiment). |
| C2 | **Unified table as its own page** at the end of Part IV (as well as summarized in Ch 12) | `unified-table.qmd` | Spec §7 asks for a standalone page; placing it in the book's TOC (rather than an orphan page) keeps it navigable. Static first; a small JS filter later only if cheap (§8). |
| C3 | Ch 3 gains a subsection **"an unbiased value is not an unbiased gradient"**: $k_3$ as a loss vs $k_1$ in the reward | Ch 3, revisited in Ch 8 | Research: Liu et al. 2025 (arXiv 2510.01555) and Zhang et al. (ICLR 2026, arXiv 2505.17508) show GRPO's $k_3$-as-loss gradient is a biased approximation, and that off-policy use lacks an IS correction. The testbed can show it exactly. |
| C4 | Ch 3 and the terminology-traps appendix state the **precise** PPO/KL fact: original PPO had a KL-penalty variant to $\pi_{\text{old}}$; LLM "PPO" adds a KL to a *fixed* $\pi_{\text{ref}}$ | Ch 3, appendix | Spec §4 wording was imprecise (verified against the PPO paper, §4 and Eq. 9). |
| C5 | Ch 5 states that Gao et al. found the KL penalty acts **like early stopping** on the gold-reward–KL frontier, and reproduces that test on the toy | Ch 5 | Verified in Gao et al. §3.6; sharpens the spec's "the KL penalty only partly helps". |
| C6 | Ch 5 is careful that Lightman et al. use PRMs to **rerank** samples (verification), not as an RL reward | Ch 5 | Avoids overclaiming what process supervision has been shown to do *for RL*. To confirm in full text. |
| C7 | Ch 9 opens from STaR's own §3.1 ("an approximation to an RL-style policy gradient objective") | Ch 9 | Primary-source anchor for the chapter's thesis (verified). |
| C8 | Ch 9 distinguishes **exact-acceptance rejection sampling** (targets $\pi^\star$ exactly) from **hard filtering** (best-of-N, thresholds: a sharper, different target) | Ch 9 | Falls out of the testbed; makes "filtering approximates sampling from $\pi^\star$" precise instead of loose. |
| C9 | Ch 10 layer 3 adds **OpenJev-RLCD** (Gao & Wang, arXiv 2609.38850, 2026-09-30) as an *independent, unrefereed* reconstruction, explicitly not evidence about TypeSafe's method | Ch 10 | Published after the spec was drafted. |
| C10 | Ch 10 reports Guo et al.'s calibration numbers **as the paper frames them** (pooled ECE 0.047; median per-benchmark ECE 0.168 vs 0.074 expected under perfect calibration; base-rate mismatch), not as "above acceptable thresholds" | Ch 10 | Wording discrepancy with the spec; see `research-log.md`. |
| C11 | Ch 6 hosts the signature figure (with every method implemented by then: PPO+critic+GAE, GRPO, DPO, RFT, on-policy distillation). Chs 7–9 each get their own failure-mode figure instead of re-showing it | Ch 6 | One canonical place for figure 9; the method chapters show what is specific to each route. |
| C12 | Ch 4 cites the origin of "exposure bias" correctly (Ranzato et al. 2016 lead) alongside DAgger for compounding errors | Ch 4 | The spec pairs the term with DAgger; the term and the analysis come from different papers. To confirm. |

### Chapter status

| Ch | File | Title | Status | Key primary sources | Figures |
|---|---|---|---|---|---|
| 0 | `00-map.qmd` | The Map | **done** | (synthesis) | F1, F2 |
| 1 | `01-what-makes-a-problem-rl.qmd` | What Makes a Problem RL | **done** | Sutton & Barto 2018; Li 2010; Ross & Bagnell 2010; Ross 2011; Ranzato 2016 | F19 |
| 2 | `02-policy-gradients.qmd` | Policy Gradients | **done** | Williams 1992; Schulman et al. 2016 (GAE); Kool et al. 2019; Shao et al. 2024 §5.2 | F3, F4 |
| 3 | `03-keeping-updates-stable.qmd` | Keeping Updates Stable | **done** | TRPO; PPO; Schulman 2020; Liu et al. 2025; Zhang et al. 2025 | F5, F6, F7 |
| 4 | `04-the-llm-as-a-policy.qmd` | The LLM as a Policy | **done** | Ross et al. 2011; Ranzato et al. 2016; Ahmadian et al. 2024; Li et al. 2023; Tülu 3; DeepSeek-R1; Noukhovitch et al. 2025 | F20 + classical-vs-LLM table |
| 5 | `05-rewards.qmd` | Where Rewards Come From, and How They Break | **done** | Christiano 2017; Bai 2022; Lee 2023; Tülu 3; Uesato 2022; Lightman 2023; Gao 2022 | F10 |
| 6 | `06-the-shared-target.qmd` | The Shared Target, and RLHF with PPO | **done** | Korbak 2022; Levine 2018; Ziegler 2019; Stiennon 2020; Ouyang 2022 | F8, F9 |
| 7 | `07-direct-preference-methods.qmd` | Direct Preference Methods | **done** | Rafailov 2023; Azar 2023; Ethayarajh 2024; Meng 2024; Hong 2024; Razin 2025 | F11, F11b |
| 8 | `08-policy-gradients-without-a-critic.qmd` | Policy Gradients Without a Critic | **done** | Ahmadian 2024; Li 2023; Shao 2024; Liu 2025; Yu 2025; Zheng 2025; MiniMax 2025 | F4b, F12, F12b |
| 9 | `09-imitating-a-better-distribution.qmd` | Learning by Imitating a Better Distribution | **done** | Zelikman 2022; Gulcehre 2023; Dong 2023; Peters & Schaal 2007; Peng 2019; Agarwal 2024; Lu 2025 | F16, F17 |
| 10 | `10-rewarding-calibration.qmd` | Rewarding Calibration: RLCD and Proper Scoring Rules | **done** | TypeSafe 2026; Guo 2026; Damani 2025; Wu 2025; Zhu 2026; Gao & Wang 2026; Gneiting & Raftery 2007; Yang 2023 | F14 |
| 11 | `11-what-rl-does-to-a-model.qmd` | What RL Actually Does to a Model | **done** | Yue 2025; Liu 2025 (ProRL); Chu 2025; Kirk 2023; Shenfeld 2025; Korbak 2022; Zhou 2024 (ArCHer) | F13 |
| 12 | `12-decision-guide.qmd` | Decision Guide and the Unified Table | **done** | (synthesis) | (tables) |
| — | `unified-table.qmd` | The Unified Table | **done** (generated from `scripts/method_data.py`, with a filter box) | method cards | — |
| A | `appendix-notation.qmd` | Notation | **done** | — | — |
| A | `appendix-testbed.qmd` | The Toy Testbed | **done** | — | F18 |
| A | `appendix-acronyms.qmd` | Acronym Decoder | **done** | — | — |
| A | `appendix-terminology-traps.qmd` | Terminology Traps | **done** | — | — |
| A | `appendix-timeline.qmd` | Timeline | **done** | dates from `research-log.md` | F15 |
| A | `appendix-further-reading.qmd` | Further Reading | **done** | — | — |

Writing order (spec §16): Part I (Chs 1–3) → Part II (Chs 4–5) → Ch 6 → Chs 7–10 → Chs 11–12 →
map, unified table and appendices (they summarize what the chapters established). Each chapter:
research → log → write → figures run and inspected → §15 checklist → status here.

---

## Figure list

Every figure is one script, `scripts/figures/fig_<slug>.py`, and every one is a real computation.
"Makes visible that ..." is the sentence the script's docstring must finish.

| # | Slug | Ch | Makes visible that ... | Computation | Status |
|---|---|---|---|---|---|
| F1 | `feedback_space` | 0 | supervised learning, bandits, full RL and imitation are corners of one three-axis space, and every LLM post-training method sits somewhere in it | Each method's (feedback, data, consequence) coordinates read from its method card, laid out on a 3-axis projection | **done** |
| F2 | `method_lineage` | 0 | the method zoo is one tree with four post-RLHF branches, and each node inherits identifiable classical pieces | networkx layout of a typed lineage graph (edges labelled by "what changed") | **done** (layout by generation, not year: a year axis was unreadable) |
| F3 | `sft_vs_reinforce_gradient` | 2 | SFT and REINFORCE apply the same push-up operation to different sequences with different (signed) weights | One batch per method + exact expected coefficient $q(y)w(y)$ for all 625 responses | **done** |
| F4 | `gradient_variance_by_baseline` | 2 | a baseline leaves the expected gradient unchanged and makes it invariant to the reward's offset; its variance benefit depends on reward and policy (measured: none wins at a mostly-wrong policy with 0/1 reward); GAE's λ trades variance for bias | 1,500 estimates per point vs the exact gradient: offset sweep, G sweep, λ sweep | **done** (claim revised by evidence; see research log) |
| F4b | (panel of F4) | 8 | group normalization rescales prompts by reward spread, which reweights the objective (not just its variance) | Same harness, per-prompt effective weight | **done** (panel 2 of `grpo_biases`) |
| F5 | `ppo_clip_regions` | 3 | the clip only removes gradient where an update would go *further* in a direction already taken; mistakes can always be undone | Clipped surrogate and its derivative vs $\rho$, for $A>0$ and $A<0$ (incl. DAPO's asymmetric clip) | **done** |
| F6 | `forward_vs_reverse_kl` | 3 | the reverse KL fits one mode of a bimodal target and the forward KL smears across both | Fit a single Gaussian to a two-component mixture by minimizing each KL numerically | **done** |
| F7 | `kl_estimators` | 3 | $k_3$ is unbiased with low variance as a *value*, but its *gradient* as a loss is not $\nabla\mathrm{KL}$, and the gap grows as the policy moves | Exact mean/std of $k_1,k_2,k_3$ along $\pi^\star_\beta$; exact-gradient training with five KL implementations (fixed points) | **done** (k3 noise claim corrected by evidence) |
| F8 | `pi_star_vs_beta` | 6 | $\beta$ interpolates between $\pi_{\text{ref}}$ ($\beta\to\infty$) and the reward argmax ($\beta\to0$), and every $\beta$ gives one point on the reward–KL frontier | Exact $\pi^\star_\beta$ on the testbed for a range of $\beta$ (candidate for a slider, §8) | **done** (static; slider still optional) |
| F9 | `all_roads_to_pi_star` | 6 (preview on landing page) | all the methods target the same exact $\pi^\star$; noise and finite/offline data, not the target, separate them | Training runs of every method, exact KL to $\pi^\star$ | **done** (final: exact, RLOO, PPO+critic+GAE, GRPO ×2, DPO ×3, RFT ×2, on-policy distillation) |
| F10 | `overoptimization` | 5 | the proxy reward keeps rising while the true reward peaks and falls as KL grows, and a KL penalty mostly moves you *along* that curve (early stopping) rather than improving it | Proxy RM (limited-feature Bradley–Terry fit to noisy preferences from $r^\star$); RL and best-of-$n$ against it; exact true reward vs exact KL; KL-penalty sweep (tests Gao §3.6 qualitatively) | **done** (KL-penalty result partly differs from Gao et al.; reported) |
| F11 | `direct_preference` (merged F11 + F11b) | 7 | offline DPO on finite pairs first approaches $\pi^\star$ then overfits; IPO's bounded target stops the drift; more data or online pairs fix it | DPO / IPO / online DPO on the testbed with growing datasets | **done** |
| F11b | `dpo_likelihood_displacement` | 7 | DPO can lower the likelihood of the *chosen* response while increasing the margin | Chosen/rejected log-probs during DPO. **Risk:** may need a shared-feature (log-linear) policy rather than the tabular one; see E10 | **done** as panel 3 of `direct_preference`; needed the log-linear policy (fallback (d) used) |
| F12 | `grpo_length_bias` | 8 | per-sequence length normalization makes long wrong answers cheaper, so length grows on wrong answers; Dr. GRPO's aggregation removes the drift | EOS-variant testbed; GRPO vs Dr. GRPO vs DAPO token-level aggregation | **done** (merged into `grpo_biases`) |
| F12b | `zero_variance_groups` | 8 | a group whose rewards are all equal carries exactly zero signal, and how often that happens depends on difficulty and group size | Exact probability that a group of size $G$ is all-correct/all-wrong, per prompt difficulty; DAPO's dynamic sampling | **done** (merged into `grpo_biases`) |
| F13 | `pass_at_k_sharpening` | 11 | RL toward $\pi^\star$ raises pass@1 but can lower pass@$k$ at large $k$, because reverse-KL optimization concentrates mass | Exact pass@$k$ for $\pi_{\text{ref}}$ and $\pi^\star_\beta$ (and trained policies) | **done** (`pass_at_k`) |
| F14 | `calibration_reward` | 10 | a binary correctness reward makes "always commit, confidently" optimal, while a Brier reward makes the stated confidence match the true hit rate | Answer + confidence-bucket task; train with each reward; reliability diagrams and confident-wrong rate | **done** |
| F15 | `timeline` | appendix | the lineage is a few ideas recombined over 30 years, with a burst after 2023 | Dated nodes from `research-log.md`, laid out by date | **done** (vertical list layout) |
| F16 | `best_of_n_vs_pi_star` | 9 | best-of-$N$ hits a different target than $\pi^\star$; its KL from the reference follows $\log N - (N-1)/N$ | Exact best-of-$N$ distribution by enumeration; analytic KL check | **done** (merged into `imitation`) |
| F17 | `dense_vs_sparse_reward` | 9 | a teacher's per-token log-probability is a dense reward that learns far faster per sample than a sequence-level reward | On-policy distillation vs RLOO toward the same $\pi^\star$ (teacher = exact $\pi^\star$), KL vs samples consumed | **done** (merged into `imitation`) |
| F18 | `testbed_overview` | appendix | the toy language has a genuine non-uniform prior, prompts of different difficulty, and a computable optimum | $\pi_{\text{ref}}$, reward and $\pi^\star$ for each prompt | **dropped**: covered by `pi_star_vs_beta` and the testbed appendix's tables |
| F19 | `three_kinds_of_feedback` | 1 | the same model and gradient under instructive, self-generated evaluative and logged evaluative feedback learn at different speeds for reasons traceable to the three properties; RL also collapses onto one correct answer while SFT reproduces a spread | Three training runs, exact P(correct), 5 seeds | **done** |
| F20 | `where_credit_lives` | 4 | sequence-level credit puts the same advantage on every token; the exact token-level advantage concentrates where the outcome is decided; both telescope to the same total | Exact $V^\pi(s_t)$ by enumeration, two rewards | **done** |
| F0 | `social_preview` | README | (banner, generated from F2) | — | **done** |

**Interactivity** (§8): the unified table has a filter box (a few lines of inline JS). The $\beta$
slider on F8 was not built; the static figure with four values of $\beta$ plus the right-hand panel
over all $\beta$ carries the same lesson. Left as an optional follow-up.

---

## Experiment list

All experiments run on the shared testbed. New code needed is listed per experiment.

| # | Experiment | Ch | Needs | Status |
|---|---|---|---|---|
| E1 | Gradient variance by baseline: MSE of sampled gradients vs the exact gradient | 2, 8 | `experiments_gradvar.py` (done); critic = exact $V^\pi(s_t)$ (+ noise for GAE) via `metrics.prefix_values` | **done** for Ch 2; Ch 8 adds GRPO per-prompt weighting |
| E2 | PPO clip regions | 3 | `estimators.ppo_clip_*` (done, tested) | **done** |
| E3 | Forward vs reverse KL on a bimodal target | 3 | standalone (scipy) | **done** |
| E4 | KL estimators: value bias/variance and gradient bias as the policy drifts | 3 | `kl_penalties.py` (done, tested) | **done** |
| E5 | KL in the reward vs KL in the loss: where does each fixed point land? | 3, 8 | exact version done in Ch 3 (`kl_penalties.py`); sampled GRPO version in Ch 8 | **done** (exact) |
| E6 | SFT vs REINFORCE gradient contributions | 2 | figure script only | **done** |
| E7 | Overoptimization: proxy RM, RL and best-of-$n$, KL-penalty sweep | 5 | `reward_model.py` (done, tested) | **done** |
| E8 | $\pi^\star$ across $\beta$ | 6 | closed form | **done** |
| E9 | All roads lead to $\pi^\star$ | 6 | `methods/ppo.py`, `grpo.py`, `distill.py` (done, tested) | **done** |
| E10 | DPO offline drift, IPO, online DPO, likelihood displacement | 7 | `dpo.py` variants + `loglinear.py` (done, tested) | **done** |
| E11 | GRPO length bias and zero-variance groups | 8 | `eos_testbed()` (done) | **done** |
| E12 | Best-of-$N$ vs exact rejection sampling | 9 | `rft.py`, `reward_model.best_of_n_logprobs` | **done** |
| E13 | On-policy distillation, dense vs sparse | 9 | `distill.py` (done) | **done** (advantage ~2×; no growth with length in the toy) |
| E14 | Correctness vs Brier reward | 10 | `calibration.py` (done, tested) | **done** |
| E15 | Pass@$k$ sharpening | 11 | `metrics.pass_at_k` | **done** |

**Testbed extensions this implies** (each documented in the testbed appendix when added):
(a) an EOS / variable-length variant (E11); (b) a learned proxy reward model (E7); (c) an
answer-plus-confidence task (E14); (d) possibly a shared-feature log-linear policy class for
likelihood displacement (E10), since a fully tabular policy may not reproduce it. (d) will be
tried with the tabular policy first; the fallback is recorded here if needed.

**Prompt difficulty spread.** Under the current $\pi_{\text{ref}}$, $P(\text{correct})$ ranges
0.15–0.30 across the four prompts. E1/F4b and E11 need a wider spread (one "hard" prompt near 0.02,
one "easy" near 0.7). Planned fix: tilt the per-prompt Markov chains when building the corpus.
This changes $\pi_{\text{ref}}$, so it is done before Phase 2's first figure, and F9 is regenerated.

---

## RLCD (spec §9, rule 4): what was found, sources, confidence

As of **2026-10-01**:

- **Mechanism:** still **undisclosed** by TypeSafe AI. The only primary source is the launch blog
  post (dated 2026-09-28). It discloses the name, the goal (calibrated decisions with
  "epistemically honest probabilities"), typed outputs with probabilities, and a "parallel
  sampler"; it does not disclose the algorithm, reward, data or architecture. **Confidence:
  high.**
- **Independent evaluation:** Guo et al., arXiv 2609.29429 (2026-09-24): strong zero-shot
  *detection* (median AUROC 0.886); good pooled calibration but within-benchmark calibration
  off by a base-rate mismatch (numbers in `research-log.md`). It does not describe the mechanism.
  **Confidence: high** on the claims, medium on exact figures until re-read by hand.
- **New since the spec:** Gao & Wang, "OpenJev-RLCD", arXiv 2609.38850 (2026-09-30), an
  independent open implementation built on proper scoring rules, which claims "nothing about
  Jev's internal algorithm". It will appear in Ch 10 layer 3, labelled as such.
- **Date:** resolved: the post is dated 2026-09-15; 2026-09-28 is the site-build timestamp.
- **Re-checked before writing Ch 10 (2026-10-01):** still unpublished; Layer 2 remains the book's reconstruction.

---

## Decisions log

- **numpy + scipy only, no torch.** The spec's `pyproject` sketch lists torch (CPU). The
  prefix-tabular policy needs only the softmax score function, which is ten lines of numpy, and
  computing gradients by hand keeps Thesis 2 visible in the code
  (`grad_log_likelihood(..., weights)` is called by every method). It also keeps CI and figure
  runs fast (the full first figure takes ~20 s). Revisit only if a later experiment needs a neural
  policy (E10 fallback); a small log-linear policy can still be done in numpy.
- **Prefix-tabular policy, not a per-position table or a GRU.** A per-position table cannot
  represent $\pi^\star$ in general, so "does method X reach $\pi^\star$?" would be confounded with
  "can the architecture represent it?". The prefix-tabular policy is fully expressive, still
  autoregressive and token-level. Defaults: $V=5$, $L=4$ (625 responses), 4 prompts.
- **$\pi_{\text{ref}}$ is a genuine MLE fit** to a corpus sampled from per-prompt Markov chains,
  with additive smoothing (so it has full support; DPO's identifiability needs that).
- **Adam for every method**, same learning rate by default, so method comparisons are not
  optimizer comparisons.
- **Figures pre-generated, not rendered by Quarto** (the `loss-functions-lab` convention), but
  figure scripts import the real `rl4llm` package, so a figure is always the tested code.
- **Same palette as the sibling sites**, validated with the dataviz validator; colors assigned
  per *method family* (`FAMILY_COLOR`), constant across all figures.
- **The book uses $\pi^\star$ (not $\pi^*$)**, matching `loss-functions-lab` Ch 16.
- **Cross-reference chapters by link and title, not by number** in prose (lesson from
  `math-conceptual-map`).
- **A CI workflow (ruff + pytest)** in addition to the Pages workflow, as in `optimization-lab`.
- **bib entries generated from the arXiv API**, so titles and author lists are exact; author
  lists of 100+ (DeepSeek-R1, MiniMax-M1) collapsed to the corporate author.

## Author decisions (2026-10-01, after the Phase 1 report)

1. Chapter plan approved as proposed (12 chapters + map, changes C1–C12).
2. Optional real-model notebook: **skipped for now**.
3. numpy + scipy only (no torch): **approved**.
4. OpenJev-RLCD in Ch 10 layer 3, labelled as third-party and unrefereed: **approved**.

## Unverified or not-yet-checked items (carried into Phase 2)

- ~~ReMax's baseline~~ confirmed (reward of the greedy response), Ch 4.
- ~~Korbak et al. 2022 venue~~ confirmed: Findings of EMNLP 2022 (Ch 6).
- ~~Lightman et al. 2023~~ confirmed: reranking only, no RL (Ch 5).
- ~~Origin of "exposure bias"~~ confirmed (Ranzato et al. 2016), Ch 1.
- ~~Guo et al. 2026 calibration figures~~ re-read verbatim (Ch 10).
- ~~OpenJev-RLCD decomposition~~ re-derived and tested (Ch 10).
- BibTeX entries still to add (papers verified via abstract page during Phase 1, but the arXiv
  API rate-limited the batch export): Razin 2024, Damani 2025, Wu 2025, Zhu 2026, Guo 2026,
  Gao & Wang 2026, Liu 2025 (KL), Zhang 2025 (RPG); also Sutton & Barto 2018, Gneiting & Raftery
  2007, Kool et al. 2019, Ranzato et al. 2016.

## Non-goals

From `SPEC.md` §14: not a classical RL textbook, not a survey of every preference-optimization
variant, not an infrastructure guide, not an alignment-philosophy book, not a leaderboard. The
optional real-model notebook (§10) is **skipped for now** (author decision, 2026-10-01).

## Phase 2 log (2026-10-01)

Chapters were written in the spec's order (1–3, 4–5, 6, 7–10, 11–12), then the map, the unified
table and the appendices. For each: research → `research-log.md` → code + tests → figures run and
inspected → prose → render → §15 check. Decisions and findings made along the way:

- **Default testbed retuned** before the first chapter figure (seed 1, concentration 0.3, targets
  (1, 2, 0, 0)) so the prompts span difficulty 0.02–0.71; the Phase 1 figure was regenerated.
- **Library growth** beyond the Phase 1 plan: `kl_penalties.py`, `reward_model.py`, `loglinear.py`,
  `calibration.py`, `experiments_gradvar.py`, `metrics.prefix_values`, `estimators.gae`, an EOS
  variant (`eos_testbed`), and methods `ppo`, `grpo` (with Dr. GRPO / DAPO / CISPO-style switches),
  `distill` (on- and off-policy), `rft.train_weighted`, `rft.train_iterative_filter`, DPO variants
  (IPO, SimPO, online). 38 tests.
- **Optional cosine learning-rate decay** added to the shared Adam (off by default) so the signature
  figure shows convergence rather than optimizer noise floors; applied identically to every method.
- **Findings that differ from the spec's expectations** (each reported in the chapter and logged):
  (1) a baseline does not always reduce variance; with 0/1 rewards and rare success plain REINFORCE
  had the lowest error (Ch 2); (2) $k_3$'s noise advantage holds only near $\pi_{\text{ref}}$, and
  "$k_3$ as a loss" follows the forward KL, changing the target (Ch 3); (3) in the toy, small KL
  penalties take a worse path than the proxy optimum, unlike Gao et al.'s finding (Ch 5); (4) the
  tabular policy does not show likelihood displacement on informative pairs; a shared-feature policy
  does, in most but not all pair sets (Ch 7); (5) GRPO's std normalization further weakens its
  effective KL (Ch 6/8); (6) iterated hard filtering collapses past the $\beta\to0$ limit (Ch 9);
  (7) the dense-signal advantage of on-policy distillation is ~2× and does not grow with length in
  the toy (Ch 9); (8) a pass@$k$ crossover requires RL to lower success on some problems (Ch 11).
- **RLCD**: re-checked before Ch 10; still undisclosed. The blog's date corrected to 2026-09-15
  (2026-09-28 is the site-build timestamp); the FAQ's training-data answer, missed by the Phase 1
  fetch, recovered from the raw page.
- **Single source of truth for the synthesis**: `scripts/method_data.py` generates the unified table
  (both versions), the lineage graph, the feedback-space map and the timeline.
- **Chapter lengths** (words incl. tables): Chs 1–10 ≈ 2,200–3,300; Ch 11 ≈ 2,000; Ch 12 ≈ 1,650.
  Below the spec's 2,500–4,000 for some chapters; not padded. Flagged for the author.

## Issue #1 remediation (2026-10-07)

Fixed the defects reported by the cross-repository quality assessment (GitHub issue #1); details and
the re-verified numbers are in `research-log.md`. Decisions made along the way:

- **Every number quoted in prose is printed by a figure script.** Where a quoted number was not
  printed (Ch 1 entropies, Ch 6/8 "without std division", Ch 7 log-probabilities), the script now
  prints it; seed-dependent numbers are quoted as a median and range over the figure's seeds.
- **Ch 8's constant-normalizer comparison** is now a controlled one: an unplotted run with only the
  aggregation changed (2.6 tokens), with the plotted Dr. GRPO run (3.0) named as such.
- **Method-specific coefficients** no longer reuse $\gamma$ or $\lambda$ (reserved for the discount
  and GAE): $c_{\text{ptx}}$, $c_D$/$c_U$, $c_{\text{OR}}$, $c_{\text{corr}}$ and SimPO's margin $m$,
  each with the paper's own letter noted. Schulman's ratio is $u$; Ch 10's true success probability is
  $\eta$. All added to the notation appendix.
- **Bib keys follow the year field**: `zhang2026design`, `wang2023mathshepherd`, `xiong2023iterative`.
- Not reproduced: the README "unclosed italic asterisk" (every `*` in `README.md` is paired).
