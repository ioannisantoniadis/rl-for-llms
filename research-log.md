# Research log

Every source consulted for this book, what it was used for, and what could not be verified.
This is the audit trail required by `SPEC.md` §12. Entries are grouped by phase and then by
chapter. Status markers:

- **✓ verified**: checked against the primary source (paper text, official report or official
  blog), with what was checked stated explicitly.
- **~ partly verified**: metadata or abstract checked; the specific technical claim still needs
  a full-text read before the chapter that uses it is written.
- **✗ discrepancy**: the spec (or a sibling repo) says something the primary source does not
  support. The discrepancy is described and the book follows the source.
- **? unverifiable**: could not be confirmed from a primary source.

Secondary sources (explainers, blogs that are not the method's own) are listed only when they
were used for orientation, and are never the authority for a technical claim.

---

## Phase 1: initial verification pass (2026-10-01)

### Method

1. Bibliographic metadata (exact title, first authors, submission date) for 38 arXiv papers was
   pulled in one batch from the arXiv API (`export.arxiv.org/api/query?id_list=...`). All 38
   exist and match the titles and first authors the spec gives; see the table below.
2. For the claims the spec relies on most heavily, the paper's own text was read (arXiv HTML
   rendering, or PDF extracted with `pdftotext` when no HTML exists) and the relevant
   equations or sentences checked.
3. RLCD was searched for from scratch (§9 of the spec), including for anything published since
   the spec was drafted.

### Bibliographic verification (arXiv API, 2026-10-01)

| arXiv | Date (v1) | Title (exact) | First authors | Status |
|---|---|---|---|---|
| 1011.0686 | 2010-11-02 | A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning | Ross, Gordon, Bagnell | ✓ (AISTATS 2011) |
| 1502.05477 | 2015-02-19 | Trust Region Policy Optimization | Schulman, Levine, Moritz, Jordan, Abbeel | ✓ |
| 1506.02438 | 2015-06-08 | High-Dimensional Continuous Control Using Generalized Advantage Estimation | Schulman, Moritz, Levine, Jordan, Abbeel | ✓ |
| 1707.06347 | 2017-07-20 | Proximal Policy Optimization Algorithms | Schulman, Wolski, Dhariwal, Radford, Klimov | ✓ |
| 1706.03741 | 2017-06-12 | Deep reinforcement learning from human preferences | Christiano, Leike, Brown, Martic, ... | ✓ |
| 1805.00909 | 2018-05-02 | Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review | Levine | ✓ |
| 1909.08593 | 2019-09-18 | Fine-Tuning Language Models from Human Preferences | Ziegler, Stiennon, Wu, Brown, ... | ✓ |
| 1910.00177 | 2019-10-01 | Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning | Peng, Kumar, Zhang, Levine | ✓ |
| 2009.01325 | 2020-09-02 | Learning to summarize from human feedback | Stiennon, Ouyang, Wu, Ziegler, ... | ✓ |
| 2203.02155 | 2022-03-04 | Training language models to follow instructions with human feedback | Ouyang, Wu, Jiang, Almeida, ... | ✓ |
| 2203.14465 | 2022-03-28 | STaR: Bootstrapping Reasoning With Reasoning | Zelikman, Wu, Mu, Goodman | ✓ |
| 2205.11275 | 2022-05-23 | RL with KL penalties is better viewed as Bayesian inference | Korbak, Perez, Buckley | ✓ (venue: EMNLP 2022 per fetch; confirm Findings vs main in Ph. 2) |
| 2210.10760 | 2022-10-19 | Scaling Laws for Reward Model Overoptimization | Gao, Schulman, Hilton | ✓ |
| 2211.14275 | 2022-11-25 | Solving math word problems with process- and outcome-based feedback | Uesato, Kushman, Kumar, Song, ... | ✓ |
| 2212.08073 | 2022-12-15 | Constitutional AI: Harmlessness from AI Feedback | Bai, Kadavath, Kundu, Askell, ... | ✓ |
| 2304.06767 | 2023-04-13 | RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment | Dong, Xiong, Goyal, Zhang, ... | ✓ |
| 2305.18290 | 2023-05-29 | Direct Preference Optimization: Your Language Model is Secretly a Reward Model | Rafailov, Sharma, Mitchell, Ermon, Manning, Finn | ✓ |
| 2305.20050 | 2023-05-31 | Let's Verify Step by Step | Lightman, Kosaraju, Burda, Edwards, ... | ✓ |
| 2306.13649 | 2023-06-23 | On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes | Agarwal, Vieillard, Zhou, Stanczyk, ... | ✓ (arXiv 2023, ICLR 2024) |
| 2307.12950 | 2023-07-24 | RLCD: Reinforcement Learning from Contrastive Distillation for Language Model Alignment | Yang, Klein, Celikyilmaz, Peng, Tian | ✓ (the acronym collision is real) |
| 2308.08998 | 2023-08-17 | Reinforced Self-Training (ReST) for Language Modeling | Gulcehre, Le Paine, Srinivasan, Konyushkova, ... | ✓ |
| 2309.00267 | 2023-09-01 | RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback | Lee, Phatale, Mansoor, Mesnard, ... | ✓ (note the title changed across versions; cite the current one) |
| 2310.10505 | 2023-10-16 | ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models | Li, Xu, Zhang, Lin, ... | ✓ |
| 2310.12036 | 2023-10-18 | A General Theoretical Paradigm to Understand Learning from Human Preferences | Azar, Rowland, Piot, Guo, ... | ✓ |
| 2402.01306 | 2024-02-02 | KTO: Model Alignment as Prospect Theoretic Optimization | Ethayarajh, Xu, Muennighoff, Jurafsky, Kiela | ✓ |
| 2402.03300 | 2024-02-05 | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | Shao, Wang, Zhu, Xu, ... | ✓ |
| 2402.14740 | 2024-02-22 | Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs | Ahmadian, Cremer, Gallé, Fadaee, ... | ✓ |
| 2403.07691 | 2024-03-12 | ORPO: Monolithic Preference Optimization without Reference Model | Hong, Lee, Thorne | ✓ |
| 2405.14734 | 2024-05-23 | SimPO: Simple Preference Optimization with a Reference-Free Reward | Meng, Xia, Chen | ✓ |
| 2411.15124 | 2024-11-22 | Tulu 3: Pushing Frontiers in Open Language Model Post-Training | Lambert, Morrison, Pyatkin, Huang, ... | ✓ |
| 2501.12948 | 2025-01-22 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | DeepSeek-AI (Guo, Yang, Zhang, ...) | ✓ |
| 2501.17161 | 2025-01-28 | SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training | Chu, Zhai, Yang, Tong, ... | ✓ |
| 2503.14476 | 2025-03-18 | DAPO: An Open-Source LLM Reinforcement Learning System at Scale | Yu, Zhang, Zhu, Yuan, ... | ✓ |
| 2503.20783 | 2025-03-26 | Understanding R1-Zero-Like Training: A Critical Perspective | Liu, Chen, Li, Qi, ... | ✓ (this is "Dr. GRPO") |
| 2504.13837 | 2025-04-18 | Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? | Yue, Chen, Lu, Zhao, ... | ✓ |
| 2506.13585 | 2025-06-16 | MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention | MiniMax | ✓ (CISPO is introduced here) |
| 2507.18071 | 2025-07-24 | Group Sequence Policy Optimization | Zheng, Liu, Li, Chen, ... (Qwen) | ✓ |
| 2509.04259 | 2025-09-04 | RL's Razor: Why Online Reinforcement Learning Forgets Less | Shenfeld, Pari, Agrawal | ✓ (spec flagged "verify": it exists and says what the spec says) |

Non-arXiv items checked:

| Source | Status | Notes |
|---|---|---|
| Williams (1992), "Simple statistical gradient-following algorithms for connectionist reinforcement learning", *Machine Learning* 8:229–256 | ✓ | Springer: <https://link.springer.com/article/10.1007/BF00992696> |
| Peters & Schaal (2007), "Reinforcement learning by reward-weighted regression for operational space control", ICML 2007, pp. 745–750 | ✓ | dblp: <https://dblp.org/rec/conf/icml/PetersS07.html>; doi 10.1145/1273496.1273590 |
| Schulman (2020-03-07), "Approximating KL Divergence" | ✓ | <http://joschu.net/blog/kl-approx.html>, full text read (see Ch 3 below) |
| Thinking Machines Lab (Kevin Lu), "On-Policy Distillation", 2025-10-27 | ✓ | <https://thinkingmachines.ai/blog/on-policy-distillation/> |
| Ranzato, Chopra, Auli, Zaremba (2015), "Sequence Level Training with Recurrent Neural Networks", arXiv 1511.06732, ICLR 2016 | ~ | Lead for the origin of the term "exposure bias" (the abstract describes the train/test discrepancy but does not use the term; check the body in Phase 2) |
| Sutton & Barto (2018), Gneiting & Raftery (2007), Kool et al. (2019) | ~ | Standard references; bibliographic details to be confirmed when the BibTeX entries are written |

### Technical claims checked against the source text

**DeepSeekMath / GRPO (Shao et al., 2024), arXiv HTML read.** ✓
- GRPO objective: $\frac1G\sum_i \frac{1}{|o_i|}\sum_t \{\min[\rho_{i,t}\hat A_{i,t}, \mathrm{clip}(\rho_{i,t},1\pm\epsilon)\hat A_{i,t}] - \beta\,\mathbb{D}_{KL}[\pi_\theta\|\pi_{ref}]\}$:
  per-sequence length normalization, then mean over the group. KL is **in the loss**, not
  in the reward, and the paper says so explicitly ("instead of adding KL penalty in the reward").
- KL term: $\frac{\pi_{ref}}{\pi_\theta} - \log\frac{\pi_{ref}}{\pi_\theta} - 1$ per token, which is
  Schulman's $k_3$ with $r = \pi_{ref}/\pi_\theta$. The paper calls it "unbiased" and "guaranteed to
  be positive".
- Outcome-supervision advantage: $\hat A_{i,t} = (r_i - \mathrm{mean}(\mathbf r))/\mathrm{std}(\mathbf r)$, broadcast to every token.
- §5.2 "Towards to a Unified Paradigm": general gradient
  $\nabla_\theta\mathcal J_{\mathcal A} = \mathbb E_{(q,o)\sim\mathcal D}[\frac{1}{|o|}\sum_t GC_{\mathcal A}(q,o,t,\pi_{rf})\nabla_\theta\log\pi_\theta(o_t|q,o_{<t})]$,
  with three components: **data source** $\mathcal D$, **reward function** $\pi_{rf}$, **algorithm**
  $\mathcal A$ producing the **gradient coefficient** $GC$. Table 10 classifies SFT, RFT, DPO, Online
  RFT, PPO, GRPO along these axes. The spec's attribution is correct. The book's $(q, w)$ form is
  this, with $GC \leftrightarrow w$ and data source $\leftrightarrow q$.
- Iterative GRPO retrains the reward model with a 10% replay of historical data.

**Schulman, "Approximating KL Divergence" (2020-03-07), full text.** ✓
- Estimates $\mathrm{KL}[q,p]$ from samples $x\sim q$, with $r = p(x)/q(x)$:
  $k_1 = -\log r$ (unbiased, high variance, negative for some samples);
  $k_2 = \frac12(\log r)^2$ (biased, low variance; its expectation is an f-divergence that agrees with KL to second order);
  $k_3 = (r-1) - \log r$ (unbiased, always ≥ 0, a Bregman divergence / control-variate construction).
- Reported numbers: for $q=N(0,1)$, $p=N(0.1,1)$ (true KL 0.005), bias/true and stdev/true are
  k1: 0, 20; k2: 0.002, 1.42; k3: 0, 1.42. For $p=N(1,1)$ (true KL 0.5): k1: 0, 2; k2: 0.25, 1.73; k3: 0, 1.7.
  (Note: the post's prose and code swap which Gaussian is $p$ and which is $q$; for equal-variance
  Gaussians KL is symmetric, so the numbers are unaffected.)
- Mapping to GRPO: $q = \pi_\theta$, $p = \pi_{ref}$, so $k_3$ estimates the **reverse** KL
  $\mathrm{KL}(\pi_\theta\|\pi_{ref})$, unbiasedly **only when samples come from $\pi_\theta$**.

**KL estimators as losses: later work (found in this pass).** ~
- K. Liu, J. K. Liu, M. Chen, Y. Liu, "Rethinking KL Regularization in RLHF: From Value Estimation
  to Gradient Optimization", arXiv 2510.01555 (2025-10-02). Abstract: $k_1$ in the reward is the
  principled reverse-KL regularizer; "$k_1$ in reward" is gradient-equivalent to "$k_2$ as loss"
  (on-policy); GRPO's "$k_3$ as loss" is "merely a first-order, biased approximation"; off-policy
  implementations need an importance-sampling correction.
- Y. Zhang, Y. Liu, H. Yuan, Y. Yuan, Q. Gu, A. C.-C. Yao, "On the Design of KL-Regularized Policy
  Gradient Algorithms for LLM Reasoning", arXiv 2505.17508, ICLR 2026. Abstract: identifies and
  fixes an importance-weighting issue in GRPO's KL term.
- **Consequence for the book.** The spec's test ("$k_3$ is unbiased for KL under samples from
  $\pi$") is correct, and the testbed checks it. But "unbiased value estimate" ≠ "correct
  gradient". Ch 3 must make that distinction, and the testbed can show it exactly (expected
  gradient of the $k_3$ loss vs the true $\nabla\mathrm{KL}$). Full-text read of both papers is
  scheduled before Ch 3 is written.

**PPO (Schulman et al., 2017), PDF text extracted.** ✓ / ✗ (spec nuance)
- §4 "Adaptive KL Penalty Coefficient": the objective is $\hat{\mathbb E}_t[\rho_t\hat A_t - \beta\,\mathrm{KL}[\pi_{\theta_{old}}(\cdot|s_t),\pi_\theta(\cdot|s_t)]]$,
  presented as an alternative or addition to clipping. The authors report it performed worse than
  clipping. The KL is to the **previous policy**, not to a fixed reference.
- Eq. 9: $L^{CLIP+VF+S} = \hat{\mathbb E}_t[L^{CLIP}_t - c_1 L^{VF}_t + c_2 S[\pi_\theta](s_t)]$, with an entropy bonus.
- **✗ Discrepancy with the spec's "terminology traps" (§4 Appendices):** the spec says "'PPO' in
  LLM papers usually includes a KL penalty that the original PPO did not have." The original paper
  *does* have a KL-penalty variant, but its KL is to $\pi_{old}$ (a moving trust region), not to a
  frozen $\pi_{ref}$. The book will state the precise version: LLM "PPO" adds a KL to a **fixed
  reference policy**, usually folded into the per-token reward, which original PPO did not have.

**Stiennon et al. (2020), arXiv HTML read.** ✓
- Reward used for RL: $R(x,y) = r_\theta(x,y) - \beta\log[\pi^{RL}_\phi(y|x)/\pi^{SFT}(y|x)]$.
- Stated purposes of the KL term: (1) "acts as an entropy bonus, encouraging the policy to explore
  and deterring it from collapsing to a single mode"; (2) keeps outputs in the region the reward
  model was trained on.
- PPO with "each time step is a BPE token", $\gamma = 1$; value function is a separate transformer,
  initialized from the reward model.

**InstructGPT (Ouyang et al., 2022), arXiv HTML read.** ✓
- PPO-ptx objective: $\mathbb E_{(x,y)\sim D_{\pi^{RL}_\phi}}[r_\theta(x,y) - \beta\log(\pi^{RL}_\phi(y|x)/\pi^{SFT}(y|x))] + \gamma\,\mathbb E_{x\sim D_{pretrain}}[\log\pi^{RL}_\phi(x)]$.
- Value function initialized from the RM. RM trained on all $\binom K2$ comparisons per prompt as
  one batch element, $K\in[4,9]$.

**Gao, Schulman & Hilton (2022), PDF text extracted.** ✓
- $d := \sqrt{D_{KL}(\pi\|\pi_{init})}$; best-of-$n$: $R_{bon}(d) = d(\alpha_{bon} - \beta_{bon}d)$;
  RL: $R_{RL}(d) = d(\alpha_{RL} - \beta_{RL}\log d)$. Gold-RM-labels-a-proxy-RM synthetic setup.
- §3.6: varying the KL penalty, gold score depends only on $\mathrm{KL}_{RL}$; "the effect of the
  penalty on the gold score is akin to early stopping", with no improvement to the gold-reward–KL
  frontier (flagged by the authors as possibly hyperparameter-sensitive). They set the KL penalty to
  0 for their other experiments.
- **Supports and sharpens** the spec's "what the KL penalty does and does not protect against".

**Tülu 3 (Lambert et al., 2024).** ✓ "We introduce a new final finetuning stage – Reinforcement
Learning with Verifiable Rewards (RLVR)." Constant reward on success. The spec's attribution of the
*term* to Tülu 3 is correct. (Earlier work also trained on verifiable correctness, e.g. STaR and
DeepSeekMath's rule-based rewards, so the book should credit the term, not the idea, to Tülu 3.)

**Dr. GRPO (Liu et al., 2025, "Understanding R1-Zero-Like Training").** ✓
- Response-level length bias from $1/|o_i|$: shorter correct responses get larger per-token
  updates, and "longer responses are penalized less" among incorrect ones, so the policy drifts
  toward longer incorrect responses.
- Question-level difficulty bias from std normalization: low-std (too easy or too hard) questions
  get higher weight.
- Fix: remove both normalizations, which recovers a PPO-style objective with an unbiased baseline.
- Also: DeepSeek-V3-Base already shows self-reflection, and Qwen2.5 base models may have been
  pretrained on concatenated question-answer text.

**DAPO (Yu et al., 2025).** ✓ Clip-Higher with $\epsilon_{low}=0.2$, $\epsilon_{high}=0.28$. Dynamic
Sampling filters groups where all outputs receive the same reward (zero advantage → zero
gradient). Token-level loss aggregation (sum over all tokens in the batch / total tokens).
Overlong reward shaping with a soft-punishment interval. The KL term is removed ("the model
distribution can diverge significantly from the initial model, thus this restriction is not
necessary"). 50 on AIME 2024 with Qwen2.5-32B base. The spec's description is correct.

**GSPO (Zheng et al., 2025).** ✓ Sequence-level ratio $s_i(\theta) = (\pi_\theta(y_i|x)/\pi_{\theta_{old}}(y_i|x))^{1/|y_i|}$
(length-normalized), clipped at the sequence level, with a group-normalized advantage. Motivation:
token-level importance weights from single samples are "ill-posed" and accumulate high-variance
noise; stabilizes MoE training without "Routing Replay". Clip range ~3e-4 / 4e-4.

**CISPO (MiniMax-M1, 2025).** ✓ $\mathcal J = \mathbb E[\frac{1}{\sum|o_i|}\sum_i\sum_t \mathrm{sg}(\hat r_{i,t})\hat A_{i,t}\log\pi_\theta(o_{i,t}|\cdot)]$ with
$\hat r = \mathrm{clip}(r, 1-\epsilon^{IS}_{low}, 1+\epsilon^{IS}_{high})$ (lower bound effectively off). Motivation:
low-probability "reflective" tokens ("However", "Recheck", "Wait", "Aha") get large ratios and are
clipped out of the gradient by PPO/GRPO after the first update. Reports a 2× speedup over DAPO.
In $(q,w)$ form this is exactly "clip the weight $w$, keep the gradient".

**Back to Basics (Ahmadian et al., 2024).** ✓ RLOO estimator
$\frac1k\sum_i[R(y^{(i)},x) - \frac{1}{k-1}\sum_{j\ne i}R(y^{(j)},x)]\nabla\log\pi(y^{(i)}|x)$, credited to Kool et al. (2019).
Argues that PPO's variance-reduction machinery is unnecessary for RLHF, because the pretrained
policy concentrates probability mass, and that modelling the full completion as a single action
(bandit) is enough. Reports REINFORCE beating PPO and RLOO beating DPO, RAFT and PPO on their setups.

**ReMax (Li et al., 2023).** ~ Abstract: exploits "fast simulation, deterministic transitions, and
trajectory-level rewards"; ~46% GPU memory saving vs PPO for a 7B model. The baseline (reward of the
greedy response) is not stated in the abstract; confirm in full text before Ch 8.

**IPO / ΨPO (Azar et al., 2023).** ✓ $\max_\pi \mathbb E[\Psi(p^*(y\succ y'|x))] - \tau D_{KL}(\pi\|\pi_{ref})$;
IPO loss $\mathbb E[(h_\pi(y_w,y_l) - \tfrac{\tau^{-1}}{2})^2]$. Argument: with deterministic or
near-deterministic preferences, Bradley–Terry needs infinite reward gaps, so DPO's KL regularization
becomes ineffective. RLHF escapes this through reward-model underfitting.

**KTO (Ethayarajh et al., 2024).** ✓ $v(x,y) = \lambda_D\sigma(\beta(r_\theta - z_0))$ for desirable and
$\lambda_U\sigma(\beta(z_0 - r_\theta))$ for undesirable outputs, with $r_\theta = \log\pi_\theta/\pi_{ref}$, and
$z_0$ a batch-estimated KL ("biased but convenient"), not backpropagated through. Needs only a
binary desirable/undesirable signal. HALO / prospect-theory framing.

**SimPO (Meng et al., 2024).** ✓ $-\log\sigma(\frac{\beta}{|y_w|}\log\pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log\pi_\theta(y_l|x) - \gamma)$.
Reference-free, length-normalized, with a target margin $\gamma$. Motivation: mismatch between
DPO's implicit reward and the average-log-likelihood used at generation time.

**ORPO (Hong et al., 2024).** ✓ $\mathcal L = \mathcal L_{SFT} + \lambda\mathcal L_{OR}$,
$\mathcal L_{OR} = -\log\sigma(\log\frac{\mathrm{odds}_\theta(y_w|x)}{\mathrm{odds}_\theta(y_l|x)})$, with
$\mathrm{odds} = P/(1-P)$ and $P$ the length-normalized (geometric-mean) likelihood. No reference
model, single stage.

**DPO likelihood displacement.** ✓ (lead confirmed) Razin, Malladi, Bhaskar, Chen, Arora, Hanin,
"Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization", arXiv
2410.08847, ICLR 2025. Documents that the preferred response's likelihood often *decreases*
during DPO, sometimes catastrophically. A CHES embedding-similarity score predicts it. Other leads
to read for Ch 7: Pal et al. 2024 (Smaug / DPO-Positive, arXiv 2402.13228).

**STaR (Zelikman et al., 2022), PDF text extracted.** ✓ §3.1: "STaR can be seen as an
approximation to an RL-style policy gradient objective." Treat the rationale as a latent variable,
use reward $\mathbb 1(\hat y = y)$, and apply the log-derivative trick. The indicator discards the
gradient of failed samples, which is the filtering step. The approximations are greedy decoding
and multiple gradient steps on the same batch. **This gives Ch 9's thesis a primary-source anchor.**

**ReST (Gulcehre et al., 2023).** ✓ Grow (sample from the policy) / Improve (filter with an
increasing threshold schedule, e.g. [0.0, 0.7, 0.8, 0.9, 0.95, 0.99], then fine-tune). Behavior
cloning (NLL) on the filtered data worked best among the offline-RL losses tried. Explicitly related
to Expert Iteration and Self-Imitation Learning. Task: machine translation.

**GKD (Agarwal et al., 2023/ICLR 2024).** ✓ $\mathcal L_{GKD} = (1-\lambda)\mathbb E_{(x,y)\sim(X,Y)}[D(p_T\|p_S^\theta)(y|x)] + \lambda\mathbb E_{x}\mathbb E_{y\sim p_S}[D(p_T\|p_S^\theta)(y|x)]$
with $D\in$ {forward KL, reverse KL, generalized JSD}. Framed as imitation learning with
train/inference mismatch (DAgger analogy). Can be combined with RLHF/RLAIF.

**Thinking Machines on-policy distillation (Lu, 2025-10-27).** ✓ The student samples and the
teacher scores every token. Per-token reverse KL to the teacher is used as the (negative) reward,
which is dense ("O(N) bits per episode"). Credits DAgger, process reward modeling, Agarwal et al.,
Gu et al., and Qwen3. Reported compute savings vs off-policy distillation and RL (vendor-lab
numbers, reported as such).

**DeepSeek-R1 (2025).** ✓ R1-Zero: GRPO from the base model, rule-based accuracy + format rewards.
R1: cold-start SFT → reasoning RL (adds a language-consistency reward) → rejection sampling + SFT
→ second RL stage (rule-based + model-based rewards). Explicitly avoids neural outcome or process
reward models for reasoning because of reward hacking. Distillation into smaller models. The
spec's "canonical public example of interleaved SFT and RL" is correct.

**Korbak, Perez & Buckley (2022).** ✓ Abstract: standard RL fine-tuning "leads to distribution
collapse"; "KL-regularised RL is equivalent to variational inference: approximating a Bayesian
posterior which specifies how to update a prior LM to conform with evidence provided by the reward
function."

**Yue et al. (2025).** ✓ RLVR models beat their base at small $k$; base models reach higher
pass@$k$ at large $k$; reasoning "originate[s] from and [is] bounded by the base model." NeurIPS 2025
oral (per the arXiv comments field). Present as a finding with an active debate (Ch 11 must
search for replies and rebuttals).

**Chu et al. (2025).** ✓ GeneralPoints and V-IRL tasks; "SFT stabilizes the model's output format,
enabling subsequent RL to achieve its performance gains."

**Shenfeld, Pari & Agrawal (2025), "RL's Razor".** ✓ Forgetting is predicted by
$\mathrm{KL}(\pi_{ft}\|\pi_{base})$ on the new task; "on-policy RL is implicitly biased towards
KL-minimal solutions among the many that solve the new task, whereas SFT can converge to
distributions arbitrarily far from the base model."

**Process supervision.** ✓ Uesato et al. (2022): outcome supervision gives similar final-answer
error with less labelling, but process-based feedback (or a learned RM emulating it) is needed for
correct reasoning steps (GSM8K reasoning error 14.0% → 3.4%). Lightman et al. (2023): process
supervision outperforms outcome supervision on MATH, 78% on a representative subset, PRM800K
released. **Caution for Ch 5:** Lightman et al. use the reward models to *rerank* samples
(best-of-$N$ verification), not as an RL reward. Confirm in full text, and do not describe PRMs as
having been shown to help *RL training* on the strength of that paper alone.

---

### RLCD (spec §9): status as of 2026-10-01

**Primary source.** TypeSafe AI, "Introducing System One Models & Jev", blog, dated
**2026-09-15** (see the correction below; Phase 1 recorded 2026-09-28) (<https://typesafe.ai/blog/introducing-system-one-models-and-jev>). Read in full.
✓ Confirmed:
- Expansion "Reinforcement Learning for Calibrated Decisions".
- Goal: "calibrated decisions: answers with epistemically honest probabilities on System One tasks".
- "a new stack entirely focused on automation: with a new model architecture, parallel sampler for
  maximum efficiency, and training method we call Reinforcement Learning for Calibrated Decisions".
- Output: "type-safe structured values", "all answers accompanied with calibrated probabilities and
  confidence scores"; "parallel" sampler "generates all outputs in a single query".
- **Not disclosed:** training algorithm, reward function, architecture details. *(Corrected in
  Ch 10 research, 2026-10-01: the FAQ does answer the data question, in collapsed accordion content
  the fetch summarizer missed: "We make all the data ourselves ... if you want to find out more,
  we'd have to hire you". It also says Jev "is neither small nor an LLM" and that TypeSafe
  "deliberately chose not to publish performance against public benchmarks".)*
- Vendor claims (to be reported only as claims): 70–500 ms latency; "40x–200x faster" than
  frontier models; input $0.042/MTok, output free; zero hallucinations / zero type errors;
  "193.6x faster, 444.6x cheaper" in tested workflows.
- **✗ Resolved discrepancy (Ch 10):** the article's visible date, rendered next to its title, is
  **15 September 2026**; "Published Sep 28, 2026" is the Framer site-build timestamp in the page
  metadata, which the Phase 1 fetch reported as the post date. The spec's "28 September 2026" is
  likewise the build date. The book dates the announcement to 15 September 2026.
- ~~Unverified detail from the spec~~ **Resolved (Ch 10):** the blog does say "Jev outputs all
  probabilities in parallel instead of autoregressively generating by token" (raw page text).

**Has the mechanism been published since the spec was written?** **No, not by TypeSafe.** Web
searches on 2026-10-01 for an RLCD paper, technical report or reward function found none.
Secondary explainers agree ("no paper, no published reward function, no dataset description").

**New since the spec: a third-party reconstruction.** Zhimin Gao & Pichao Wang, "OpenJev-RLCD: A
Working RLCD Implementation", arXiv **2609.38850**, submitted **2026-09-30** (two days before this
log entry; unrefereed). ~ Read via arXiv HTML:
- Not affiliated with TypeSafe; states "we claim nothing about Jev's internal algorithm".
- Policy samples a rationale, then the method reads the next-token distribution over $K$ option
  tokens. Stage 1 rewards the per-rationale Brier score (shifted form $2u_y - \|u\|^2$, strictly
  proper). Stage 2 adds a score-function policy gradient with a KL anchor.
- Proposition: the mixture objective = mean per-rationale score + a "diversity bonus"
  $\mathbb E_r\|u(r) - p\|^2$. Corollary: "the calibrated group reward is RLVR + Gini bonus", i.e.
  RLVR is the mixture objective without its diversity term.
- Qwen3-1.7B on MMLU-Pro, GSM8K-Verify and ChaosNLI. Reports Brier 0.135 on GSM8K-Verify and
  selective-decision gains over GRPO.
- **Use in the book:** in Ch 10 layer 3 as one public, *independent* attempt, clearly labelled as
  unrefereed, third-party and very recent. It is **not** evidence about TypeSafe's method. Its
  "RLVR + Gini bonus" decomposition is a checkable mathematical claim; verify it in full before
  using it, and possibly reproduce it on the toy testbed.

**Independent evaluation.** Guo, Liu, Deng, Li, Zhao, Wu, Chen, Zhang, Zhang, "Just Ask Jev:
Reinforcement Learning for Calibrated Decisions as a Zero-Shot Detector of AI Alignment
Failures", arXiv **2609.29429**, submitted 2026-09-24. ✓ metadata/abstract.
- Its main result is *detection*: RLCDAlignBench, ten failure types, 44 benchmarks, five target
  models; median AUROC 0.886 zero-shot with a single generic question; 63× cheaper than LLM-judge
  scorers.
- Does not describe RLCD's mechanism ("RLCD trains a model to return calibrated decisions instead
  of generated text").
- **✗ Discrepancy (wording) with the spec.** The spec says the paper "reports pooled calibration
  but per-benchmark calibration error above acceptable thresholds". What the paper reports (read
  via the HTML full text; exact wording to be re-checked by hand in Phase 2, because the fetch
  summarizer mangled one word) is: pooled ECE **0.047** (reliability curve close to the diagonal);
  **median per-benchmark ECE 0.168 vs 0.074 expected under perfect calibration** at those sample
  sizes; **24 of 31** benchmarks exceed the null's 95th percentile; the error is a **base-rate
  mismatch** (mean probability misses the positive rate by a median 0.125) rather than a ranking
  problem (median within-benchmark AUROC 0.905). The comparison is against a *null distribution
  under perfect calibration*, not an "acceptable threshold". The book will state it that way.

**Calibration-reward research (layer 3 of Ch 10).**
- Damani, Puri, Slocum, Shenfeld, Choshen, Kim, Andreas, "Beyond Binary Rewards: Training LMs to
  Reason About Their Uncertainty" (RLCR), arXiv 2507.16806, 2025-07-22. ✓ Reward = binary
  correctness + Brier score on a verbalized confidence. They prove that this (or any bounded proper
  scoring rule) yields models that are both accurate and calibrated.
- Wu, Liu, Zeng, Zhan, Cai, Huang, "Mitigating LLM Hallucination via Behaviorally Calibrated
  Reinforcement Learning", arXiv 2512.19920, 2025-12-22. ✓ Argues binary RLVR rewards "encourag[e]
  guessing whenever correctness probability exceeds zero". Trains with strictly proper scoring
  rules and abstention / claim flagging. Qwen3-4B-Instruct.
- Yinglun Zhu, "Closing the Reflection Gap: A Free Calibration Bonus for Agentic RL" (RefGRPO),
  arXiv 2606.14211, 2026-06-12. ✓ A calibration bonus that contrasts the agent's self-assessment
  with the observed outcome, plus a dynamic coefficient schedule.
- **Acronym collision.** ✓ Yang, Klein, Celikyilmaz, Peng, Tian, "RLCD: Reinforcement Learning
  from Contrastive Distillation for Language Model Alignment", arXiv 2307.12950 (2023).

**Secondary sources consulted for orientation only (never cited as authority):**
sanity.io glossary, befailproof.ai, systemonemodels.org, mindstudio.ai, turingpost.com,
dsebastien.net, saulius.io, note.com (wayne_chang), prefactor.tech. They agree that the objective
is public and the mechanism is not.

**Confidence.** High that the mechanism is undisclosed as of 2026-10-01. High in the bibliographic
details above. Medium on the exact Guo et al. calibration numbers until re-read by hand.

---

### Sibling-repo observations (not edited; reported to the author)

- `loss-functions-lab/README.md` and `scripts/figures/README.md` refer to `fig_social_preview.py`,
  which does not exist in that repo's `scripts/figures/` (it exists in
  `modern-ai-systems-and-methods`).
- `loss-functions-lab` notation reserves bare $\pi$ for permutations. This book uses $\pi$ for
  policies throughout. Not an error, but the notation appendix states the difference explicitly.
- `modern-ai-systems-and-methods` Ch 10 (RL and bandits): no factual errors found on the points
  this book overlaps with. Its MDP definition uses $\gamma\in[0,1)$; LLM post-training uses
  $\gamma = 1$ on finite episodes, which this book explains rather than contradicts.

---

## Phase 2: per-chapter research

### Testbed change before Phase 2 figures (2026-10-01)

`default_testbed()` fixes seed 1, Dirichlet concentration 0.3 and per-prompt targets (1, 2, 0, 0),
chosen by scanning seeds so P(correct) under $\pi_{\text{ref}}$ is 0.017 / 0.187 / 0.707 / 0.439.
The signature figure was regenerated on it. Not a source; recorded because it changes numbers.

### Ch 1: What Makes a Problem RL (2026-10-01)

| Source | Used for | Status |
|---|---|---|
| Sutton & Barto, *RL: An Introduction*, 2nd ed. (2018; 2020 online PDF, <http://incompleteideas.net/book/RLbook2020.pdf>) | Quotes: §1.1 "trial-and-error search and delayed reward"; ch. 2 intro "evaluates the actions taken rather than instructs", definitions of purely evaluative / purely instructive feedback; §2.9 associative search = contextual bandits, "intermediate between the k-armed bandit problem and the full reinforcement learning problem"; §11.3 deadly triad ("instability and divergence"); §13.1 advantages of policy parameterization ("can approach a deterministic policy", stochastic policies, "any way, as long as" differentiable, ANN preferences) | ✓ full text extracted with pdftotext, quotes checked verbatim |
| Li, Chu, Langford, Schapire (2010), WWW 2010, arXiv 1003.0146 | Contextual-bandit news recommendation as the click example | ✓ |
| Ross & Bagnell (2010), "Efficient Reductions for Imitation Learning", AISTATS, PMLR 9:661–668 | Origin of the $T^2\epsilon$ compounding-error bound | ✓ bibliographic (PMLR page); the bound itself checked as quoted in Ross et al. 2011 |
| Ross, Gordon & Bagnell (2011), AISTATS, arXiv 1011.0686 | Theorem 2.1 ($J(\pi)\le J(\pi^*)+T^2\epsilon$), tightness remark, DAgger | ✓ PDF text |
| Ranzato, Chopra, Auli, Zaremba (2016), ICLR, arXiv 1511.06732 | The term "exposure bias" (quoted definition) and that MIXER trains text generators with REINFORCE | ✓ PDF text (resolves the Phase 1 open item) |
| Mudgal et al. (2024), "Controlled Decoding from Language Models", ICML 2024, arXiv 2310.17022 | Value function ("prefix scorer") guiding decoding, KL-regularized framing | ✓ abstract |
| Snell et al. (2022), ILQL, arXiv 2206.11871 | Offline Q-learning for language generation | ✓ abstract; venue not confirmed (cited as arXiv) |

Experiment result quoted in the chapter (`fig_three_kinds_of_feedback.py`, seeds 0–4): on the
hard prompt, the median P(correct) after 300 steps is 0.93 (SFT), 0.99 (REINFORCE/RLOO) and 0.36
(fixed log). Collapse check (seed 0): the SFT policy's entropy on the hard prompt is 3.61 nats with
19 responses above 1% mass (the expert's: 3.50 nats, 19 responses); REINFORCE's is 0.13 nats with
one response above 1%.


### Ch 2: Policy Gradients (2026-10-01)

| Source | Used for | Status |
|---|---|---|
| Williams (1992), *Machine Learning* 8:229–256 | REINFORCE / score-function estimator; baseline in the original work | ✓ bibliographic; attribution of the baseline to Williams checked in Sutton & Barto §13.4 bibliographical remarks |
| Sutton & Barto (2018) §§13.2–13.4 + chapter notes | Policy gradient theorem (first obtained by Marbach & Tsitsiklis and independently Sutton et al. 2000, per the notes); REINFORCE §13.3; baseline "can be any function, even a random variable, as long as it does not vary with a" §13.4 | ✓ full text |
| Schulman, Moritz, Levine, Jordan, Abbeel (2016), GAE, ICLR 2016, arXiv 1506.02438 | GAE definition eq. (16), special cases (17)–(18), and the quoted bias/variance statements (Sec. 3); TD(λ) analogy quote | ✓ PDF text |
| Kool, van Hoof, Welling (2019), "Buy 4 REINFORCE Samples, Get a Baseline for Free!", ICLR 2019 workshop (DeepRLStructPred), openreview r1lgTGL5DE | Leave-one-out baseline | ✓ bibliographic + abstract (via search results / mlanthology) |
| Ahmadian et al. (2024) | RLOO brought to RLHF | ✓ (Phase 1) |
| Shao et al. (2024) §5.2 | The (q, w) form's prior statement ("data source", "reward function", "gradient coefficient") | ✓ (Phase 1) |

**Finding that departs from the spec's expectation (recorded per the "follow the evidence" rule).**
The spec anticipated F4 as "a baseline cuts variance". Measured on the testbed with a 0/1 reward at
$\pi_{\text{ref}}$ (mostly-wrong policy), plain REINFORCE had the *lowest* relative MSE (1.23 vs
1.45–1.67 for the baselined estimators at G = 8), because zero is already near the mean reward.
Baselines win once (a) the reward has an offset (no-baseline MSE 79 at offset 3; baselined
estimators unchanged) or (b) the policy is good (85% correct: 3.58 none vs 2.53–2.88 baselined at
G = 8). The chapter reports this as measured and states the general lesson (invariance to offset
always; variance reduction depends on reward and policy). GAE sweep with an imperfect critic
(exact $V^\pi$ + fixed N(0, 0.15²) error per state, G = 8): variance 2.54 → 1.59 and squared bias
0.001 → 0.10 as λ goes 1 → 0.

### Ch 3: Keeping Updates Stable (2026-10-01)

| Source | Used for | Status |
|---|---|---|
| Schulman, Levine, Moritz, Jordan, Abbeel (2015), TRPO, arXiv 1502.05477 | Monotonic-improvement bound ("monotonically improving sequence of policies", secs. 3–4); practical averaged-KL constraint called "a heuristic approximation" (sec. 4); δ = 0.01 "for all experiments" (sec. 8); KL direction $D_{KL}(\pi_{old}\|\pi_\theta)$ | ✓ PDF text |
| Schulman et al. (2017), PPO, arXiv 1707.06347 | Eq. 7 clipped surrogate and the quoted motivation (sec. 3); "CPI" = conservative policy iteration; K epochs of minibatch SGD/Adam; sec. 4 adaptive KL penalty to $\pi_{old}$ "performed worse"; eq. 9 with value loss and entropy bonus | ✓ PDF text |
| Stiennon et al. (2020) | KL term "acts as an entropy bonus ... deterring it from collapsing to a single mode" + keeps outputs in-distribution for the RM | ✓ (Phase 1) |
| Schulman (2020) KL blog | $k_1,k_2,k_3$ definitions and properties; his test used KLs of 0.005 and 0.5 | ✓ (Phase 1) |
| Shao et al. (2024), DeepSeekMath | Sec. 4.1.1 verbatim: "instead of adding KL penalty in the reward, GRPO regularizes by directly adding the KL divergence ... to the loss, avoiding complicating the calculation of Â"; PPO per-token reward $r_t = r_\phi - \beta\log(\pi_\theta/\pi_{ref})$; sec. 4.2: β = 0.04, 64 samples per question | ✓ arXiv HTML |
| Liu, Liu, Chen, Liu (2025), arXiv 2510.01555 | Thm 5.1 ("k1 in reward" ≡ "k2 as loss" on-policy); "k1 as loss" has zero expected gradient; "k3 as loss" coefficient $1-\pi_{ref}/\pi_\theta$, a first-order biased approximation; off-policy IS correction | ✓ arXiv HTML (full text via fetch) |
| Zhang, Liu, Yuan, Yuan, Gu, Yao (ICLR 2026), arXiv 2505.17508 | $\mathbb{E}_{\pi_\theta}[k_3]$ = unnormalized reverse KL (Remark 3.5); GRPO's KL term lacks the importance weight $\pi_\theta/\pi_{old}$ | ✓ arXiv HTML |
| Yu et al. (2025), DAPO | Removes the KL term; quoted reason | ✓ (Phase 1) |

**Derived in the book and checked numerically** (`tests/test_kl_penalties.py`, finite differences):
the on-policy expected gradient of "k3 as a loss" equals $\nabla D_{KL}(\pi_{ref}\|\pi_\theta)$ (forward
KL). This reconciles the two 2025 papers: Liu et al.'s "biased approximation" is, exactly, the
forward-KL gradient; Zhang et al.'s importance-weighted version recovers the reverse-KL gradient.
Exact-gradient training (β = 0.5, default testbed): k1-in-reward and k2-loss reach π* (KL < 1e-6);
k3-loss sequence-level converges to KL(π‖π*) = 0.13, E[r] = 0.709, reverse KL 0.564 (π*: 0.636,
0.289); per-token k3 to 0.59 / 0.730 / 1.065; k1-loss to 2.59 / 1.000 / 3.60 (no regularization).

**Correction caught by looking at the figure.** The first draft of the estimator figure titled the
noise panel "k3 and k2 are far less noisy than k1". The exact computation shows k3's relative std
grows with distance from π_ref and exceeds k1's beyond KL ≈ 0.5 (heavy-tailed π_ref/π under π;
consistent with Liu et al.'s χ²-instability remark). The title and the chapter text state the
measured behavior.

### Ch 4: The LLM as a Policy (2026-10-01)

| Source | Used for | Status |
|---|---|---|
| Li et al. (2023), ReMax, arXiv 2310.10505 | The three properties of RLHF, quoted verbatim ("fast simulation", "deterministic transitions", "trajectory-level rewards"); value model ~50% of GPU memory; **baseline = reward of the greedy response** $b_\theta(x) = r(x, \bar a_{1:T})$ (resolves the Phase 1 open item) | ✓ arXiv HTML; venue listed as ICML in keywords only, so cited as arXiv |
| Ahmadian et al. (2024) | "modeling partial sequences is an unnecessary undertaking"; "probability mass is concentrated on a few tokens at each generation step" | ✓ (Phase 1 quotes) |
| Stiennon et al. (2020) | PPO with "each time step is a BPE token", γ = 1 | ✓ (Phase 1) |
| Lambert et al. (2024), Tülu 3, arXiv 2411.15124 | Abstract: SFT, DPO and RLVR stages; "a novel method we call Reinforcement Learning with Verifiable Rewards" | ✓ abstract |
| DeepSeek-AI (2025), DeepSeek-R1 | R1-Zero (RL on base, rule-based accuracy + format rewards); R1 = cold-start SFT → reasoning RL → rejection sampling + SFT → second RL | ✓ (Phase 1) |
| Chu et al. (2025) | "SFT stabilizes the model's output format, enabling subsequent RL to achieve its performance gains" | ✓ (Phase 1) |
| Noukhovitch, Huang, Xhonneux, Hosseini, Agarwal, Courville (2025), "Asynchronous RLHF", ICLR 2025, arXiv 2410.18252 | Off-policy tolerance in asynchronous RLHF; online DPO most robust to off-policy data | ✓ abstract |
| Ross & Bagnell 2010; Ross et al. 2011; Ranzato et al. 2016; Sutton & Barto §11.3 | Behavior cloning / compounding errors / exposure bias / deadly triad | ✓ (Ch 1) |

Experiment (`fig_where_credit_lives.py`): exact per-token advantages under π_ref. For the default
sum-mod-5 reward, mean |A_t| by position is 0.13 / 0.07 / 0.09 / 0.14 vs 0.31 for the broadcast
sequence-level advantage; for a first-token reward, 0.15 / 0 / 0 / 0. The first draft's
description ("decided only by the last token") was wrong for the default reward, because π_ref's
last-token distribution is not uniform; the caption and text state the computed behavior.
Interpretive (not sourced) statements in the classical-vs-LLM table are phrased as reasons, and
the deadly-triad remark is explicitly hedged ("whether or not that was the reason").

### Ch 5: Where Rewards Come From, and How They Break (2026-10-01)

| Source | Used for | Status |
|---|---|---|
| Christiano et al. (2017), arXiv 1706.03741 (NeurIPS 2017) | Reward learning from pairwise preferences; "less than one percent" of interactions; "follows the Bradley-Terry model" (PDF line ~316) | ✓ PDF text |
| Ouyang et al. (2022) | RM loss over all $\binom{K}{2}$ pairs, K = 4–9 | ✓ (Phase 1) |
| Bai et al. (2022), Constitutional AI | SL phase (critique and revise) and RL phase quotes, "RL from AI Feedback (RLAIF)" | ✓ abstract |
| Lee et al. (2023), RLAIF vs RLHF, ICML 2024 (PMLR 235) | "comparable performance to RLHF"; d-RLAIF from an off-the-shelf LLM | ✓ abstract |
| Zheng et al. (2023), MT-Bench / Chatbot Arena, NeurIPS 2023 D&B | LLM judges > 80% agreement; position, verbosity, self-enhancement biases | ✓ abstract |
| Lambert et al. (2024), Tülu 3 | RLVR quote | ✓ (Phase 1) |
| DeepSeek-AI (2025), R1 | "neural reward models—whether outcome-based or process-based—to reasoning tasks" not used, due to reward hacking | ✓ (Phase 1) |
| Uesato et al. (2022) | Outcome vs process quote; 14.0% → 3.4% reasoning error | ✓ (Phase 1) |
| Lightman et al. (2023), arXiv 2305.20050 | 78%, PRM800K; **evaluated "by its ability to perform best-of-N search"; "We do not attempt to improve the generator with reinforcement learning"** (resolves Phase 1 open item C6) | ✓ PDF text |
| Wang et al. (2023/24), Math-Shepherd, arXiv 2312.08935 | "potential to deduce the correct answer"; completer with N rollouts; soft estimation = frequency of reaching the correct answer (eq. 4) = MC value estimate; used for reranking and step-by-step PPO | ✓ PDF text; venue not confirmed, cited as arXiv |
| Gao, Schulman, Hilton (2022) | Functional forms, sec. 3.6 KL-penalty quotes, hyperparameter-sensitivity caveat | ✓ (Phase 1) |
| Singhal, Goyal, Xu, Durrett (2024), COLM 2024, arXiv 2310.03716 | Length drives most reward improvement; length-only reward reproduces most gains | ✓ abstract |
| Sharma et al. (2024), ICLR 2024, arXiv 2310.13548 | Sycophancy quote incl. "a non-negligible fraction of the time" | ✓ PDF text (venue from header) |
| Baker et al. (2025), arXiv 2503.11926 | `exit(0)` and `raise SkipTest` hacks (quoted), "quickly get reinforced and become systemic", obfuscated reward hacking | ✓ PDF text |

Experiment (`fig_overoptimization.py`; `rl4llm/reward_model.py`, tested): misspecified linear BT
proxy (2,000 pairs/prompt, annotator sharpness 3). Along π*_β(proxy): true reward 0.338 → peak
0.725 at KL 0.87 → 0.238 at KL 7.3; true-reward frontier at KL 1 = 0.869. Best-of-n vs proxy:
peak 0.753 at n = 30 (KL 1.48), 0.062 at n = 4096. RL with KL penalties (exact gradients):
β = 0.5 stops at KL 0.40 / 0.63; β = 0.2 ends near the peak (0.725); smaller β overshoot, and
β ∈ {0, 0.05} trajectories lie *below* the π*_β(proxy) path at equal KL. **This differs from Gao et
al.'s finding** (penalty does not change the frontier); the chapter reports both and says the toy
cannot settle which holds at scale. Data sweep: 250–16,000 pairs all show the peak-and-decline; a
well-specified feature set (sum mod 5) tracks the true reward.

### Ch 6: The Shared Target, and RLHF with PPO (2026-10-01)

| Source | Used for | Status |
|---|---|---|
| Ziegler et al. (2019), arXiv 1909.08593 | Eq. 2 $R(x,y) = r(x,y) - \beta\log\frac{\pi(y\mid x)}{\rho(y\mid x)}$; "a penalty with expectation β KL(π, ρ)"; constant or adaptive β; KL "following Jaques et al. (2017; 2019)"; purposes (entropy bonus, valid range of r, coherence) | ✓ PDF text |
| Jaques et al. (2017), "Sequence Tutor ... with KL-control", ICML 2017, pp. 1645–1654 | Origin of KL-control fine-tuning of sequence models | ~ bibliographic details from Ziegler et al.'s reference list (a primary source citing it); not opened directly |
| Korbak, Perez, Buckley (2022), Findings of EMNLP 2022, pp. 1083–1091 | Eqs. (5)–(7): posterior $\pi^*_{KL-RL} = \pi_0\exp(r/\beta)/Z$, equals argmax of $J_{KL-RL}$, $J \propto -D_{KL}(\pi_\theta, \pi^*)$; ELBO; "equivalent to variational inference" | ✓ PDF text; venue via ACL Anthology search result |
| Levine (2018), arXiv 1805.00909 | Max-ent RL "equivalent to exact probabilistic inference in the case of deterministic dynamics" | ✓ abstract |
| Stiennon et al. (2020); Ouyang et al. (2022) | Separate value transformer initialized from RM; value initialized from RM; PPO-ptx "minimize performance regressions on public NLP datasets", "alignment tax" | ✓ arXiv HTML |
| Li et al. (2023), ReMax | Value model ≈ 50% of GPU memory | ✓ (Ch 4) |

Experiments: `fig_pi_star_vs_beta.py` (closed form); `fig_all_roads_to_pi_star.py` final version
(1000 steps, cosine LR decay to 2% for every method, 3 seeds): final KL(π‖π*) = DPO-population
~0 (machine precision); on-policy distillation 5.4e-3; RFT 100k 8.4e-3; RLOO 9.2e-3; PPO-RLHF
1.3e-2; DPO 100k 1.9e-2; GRPO with KL in reward 2.2e-2; RFT 5k 6.3e-2; DPO 5k 0.29; GRPO as
published (k3 loss) 1.08 (KL from π_ref 1.83 vs π*'s 0.29). Ablation: GRPO-k3 *without* std
normalization ends at 0.32–0.51 (3 seeds) vs 0.81–1.10 with it, supporting the chapter's statement
that std normalization further weakens the effective KL. New methods (`ppo.py`, `grpo.py`,
`distill.py`) are covered by `tests/test_methods_converge.py`.

### Ch 7: Direct Preference Methods (2026-10-01)

| Source | Used for | Status |
|---|---|---|
| Rafailov et al. (2023), DPO, arXiv 2305.18290 | Loss; gradient; weighting quotes ("the examples are weighed by how much higher the implicit reward model r̂θ rates the dispreferred completions", "higher weight when reward estimate is wrong"); Theorem 1 wording; initializing π_ref by maximizing likelihood of preferred completions when π_SFT unavailable | ✓ PDF text |
| Azar et al. (2023), ΨPO / IPO | ΨPO objective; IPO loss; deterministic-preference overfitting argument; RLHF protected by RM underfitting | ✓ (Phase 1) |
| Ethayarajh et al. (2024), KTO | Value function, z0 ("biased but convenient", not backpropagated), binary signal, prospect theory | ✓ (Phase 1) |
| Meng, Xia, Chen (2024), SimPO | Loss, reference-free, length normalization, margin γ, reward/likelihood mismatch motivation | ✓ (Phase 1) |
| Hong, Lee, Thorne (2024), ORPO | L_SFT + λ L_OR, odds, length-normalized P, "without a reference model in a single-step manner" | ✓ (Phase 1) |
| Razin et al. (2025), ICLR 2025, arXiv 2410.08847 | "preferred responses often decreases during training"; No/Never → Yes example; refusal 74.4% → 33.4%; CHES | ✓ PDF text |
| Guo et al. (2024), OAIF, arXiv 2402.04792 | "collected ahead of training and never updated"; sample two responses "from the current model" | ✓ abstract |
| Xiong et al. (2023/24), arXiv 2312.11456 | Iterative DPO from a reverse-KL-regularized contextual bandit | ✓ abstract |
| Noukhovitch et al. (2025) | Online DPO most robust to off-policyness | ✓ (Ch 4) |

Experiments (`fig_direct_preference.py`; `rl4llm/methods/dpo.py` variants; `rl4llm/loglinear.py`;
tests in `test_preference_variants.py`): offline DPO final KL(π‖π*) ≈ 0.19 / 0.05 / 0.02 for
5k / 20k / 100k pairs per prompt, online DPO ≈ 0.012. IPO (τ = 0.5, 20k pairs): KL to its own exact
target 0.0046, to π* 0.187; KL(IPO target ‖ π*) = 0.18; IPO target E[r] = 0.41 vs π*'s 0.64.
**Likelihood displacement — a finding that departs from a naive reading of the spec:** with the
prefix-tabular policy, informative pairs' chosen log-prob *rises* (−1.78 → −1.20, 5k pairs); the
average chosen log-prob falls only because 69% of pairs are reward ties labelled by coin flip.
Razin-style displacement appears with a shared-feature log-linear policy on one-token-edit pairs,
but **not in every draw**: over 10 random pair sets (100 pairs/prompt, 600 steps), Δ log π(chosen) <
0 in 6/10 (median −0.34), rejected fell in all (median −0.67), margin grew in all (median +0.39),
log P(correct) median −0.07. A first single-seed run (300 pairs) and a sweep over pair-set sizes
(100–600) showed the same sign variability (14/20 negative). An early test written from one seed
failed on another seed; the test and the chapter were changed to report the distribution. The
chapter ties the effect to Razin et al.'s similarity mechanism and says it depends on the pairs.

### Ch 8: Policy Gradients Without a Critic (2026-10-01)

| Source | Used for | Status |
|---|---|---|
| Ahmadian et al. (2024); Kool et al. (2019) | RLOO, quotes, results | ✓ (Phase 1, Ch 2) |
| Li et al. (2023), ReMax | Greedy baseline; three properties; ~46% memory | ✓ (Ch 4) |
| Rennie et al. (2017), SCST, CVPR 2017, arXiv 1612.00563 | Lineage of the greedy baseline: "utilizes the output of its own test-time inference algorithm to normalize the rewards" | ✓ abstract |
| Shao et al. (2024), DeepSeekMath | GRPO objective; G = 64, β = 0.04 | ✓ (Phase 1, Ch 3) |
| DeepSeek-AI (2025), R1 | GRPO in R1-Zero / R1 | ✓ (Phase 1) |
| Liu et al. (2025), Dr. GRPO, arXiv 2503.20783 | Difficulty bias quote; length bias quotes; "simply remove the 1/|o_i| and std normalization terms"; implementation note "replace the mask.sum(axis=dim) with a constant value (e.g., generation budget)"; "prevents the response length from growing wildly"; DeepSeek-V3-Base self-reflection | ✓ arXiv HTML |
| Yu et al. (2025), DAPO | Four techniques, quotes, ε values, KL removal, AIME result | ✓ (Phase 1) |
| Zheng et al. (2025), GSPO | Sequence ratio, "ill-posed" token ratios, clip ranges, Routing Replay | ✓ (Phase 1) |
| MiniMax (2025), M1 / CISPO | Objective, reflective tokens, 2× over DAPO | ✓ (Phase 1) |
| Espeholt et al. (2018), IMPALA, arXiv 1802.01561 | Lineage: V-trace's "truncated importance sampling (IS)" weights | ✓ PDF text |

Experiments (`fig_grpo_biases.py`; `experiments_gradvar.effective_prompt_weights`;
`toy_language.eos_testbed`): zero-signal probability at G = 8 for the four prompts under π_ref =
0.869 / 0.19 / 0.063 / 0.011. Effective per-prompt weight at G = 8 (1,500 draws): RLOO ≈ 1.0;
group-mean-incl.-self ≈ 0.875; GRPO 1.88–2.68, highest at extreme difficulty, saturating far below
1/std. EOS testbed (L = 6, V = 5 incl. EOS, 15,625 responses), no KL, 3 seeds: per-response
normalization → correct answers 4.53 → ~2.2 tokens vs ~3.0 with constant normalizer; wrong answers
3.97 → ~4.8 vs ~4.65. The toy's larger effect is on correct answers; the chapter says so and
attributes "growing wildly" to Liu et al.'s real-scale results.

### Ch 9: Learning by Imitating a Better Distribution (2026-10-01)

| Source | Used for | Status |
|---|---|---|
| Zelikman et al. (2022), STaR, sec. 3.1 | "can be seen as an approximation to an RL-style policy gradient objective"; indicator discards failed rationales' gradients; greedy decoding; multiple gradient steps | ✓ (Phase 1, PDF text) |
| Dong et al. (2023), RAFT, TMLR | Filter-and-fine-tune; quote on RL inefficiencies and instabilities | ✓ abstract |
| Yuan et al. (2023), arXiv 2308.01825 | Names RFT: "generate and collect correct reasoning paths as augmented fine-tuning datasets" | ✓ abstract |
| Gulcehre et al. (2023), ReST | Grow/Improve, thresholds, BC best, relation to expert iteration | ✓ (Phase 1) |
| Anthony, Tian, Barber (2017), Expert Iteration, arXiv 1705.08439 | Tree search as expert, network as apprentice; Hex | ✓ abstract (venue not confirmed; cited as arXiv) |
| Peters & Schaal (2007); Peng et al. (2019), AWR | RWR weights exp(R/β); AWR update and derivation: optimal policy ∝ μ(a\|s) exp((R − V)/β) | ✓ AWR arXiv HTML |
| Beirami et al. (2025), ICML 2025, arXiv 2401.01879 | log n − (n−1)/n is an upper bound on the best-of-n KL, not exact | ✓ abstract |
| Gao et al. (2022) | Used the formula: "KL_bon = log n − (n−1)/n [Stiennon et al., 2020, Appendix G.3]" | ✓ PDF text |
| Kim & Rush (2016), Sequence-Level KD, EMNLP 2016 | Off-policy distillation = SFT on teacher outputs | ✓ abstract |
| Agarwal et al. (2023/ICLR 2024), GKD | Objective, divergences, imitation-learning framing, "can be easily combined with RL fine-tuning" | ✓ (Phase 1) |
| Lu & Thinking Machines Lab (2025) | "we set the per-token advantage to the negative reverse KL"; "a discount factor of zero"; O(N) bits per episode; compute savings (lab report) | ✓ blog |
| DeepSeek-AI (2025), R1 | Rejection sampling from the RL checkpoint to build SFT data | ✓ (Phase 1) |

Experiments (`fig_imitation.py`; `rft.train_weighted`, `rft.train_iterative_filter`,
`distill.train_offpolicy`): exact-acceptance RFT (100k proposals) and RWR land on π*_β (RWR at β = 0.5:
KL(π‖π*) 0.0072, tested). Iterated hard filtering (64 samples/prompt, 40 SFT steps, 10 iterations):
E[r] → 1.0 by iteration 4, KL from π_ref → 3.17 vs 1.72 for π_ref restricted to correct answers;
entropy per prompt [0, 0.38, 0.35, 0] vs [3.5, 1.36, 0.32, 0.85]. Best-of-n exact KL vs formula:
n = 8: 0.886 vs 1.204; n = 64: 1.765 vs 3.175; n = 1024: 4.605 vs 5.932. Dense vs sparse (L = 4,
4 samples/prompt/step, 3 seeds): KL(π‖π*) after 4,800 responses ≈ 0.017 (on-policy distillation),
0.030 (RLOO), 0.040 (off-policy distillation). Length sweep L = 2..6: RLOO/OPD ratio 1.22, 1.71,
2.18, 1.70, 1.80, so the toy does **not** show the advantage growing with length; the chapter says so.

### Ch 10: Rewarding Calibration (2026-10-01)

**§9 rule 1 re-check (2026-10-01):** fresh web search and a re-read of the raw TypeSafe page found
no published RLCD mechanism, technical report or reward function. Secondary explainers
(sanity.io, saulius.io, systemonemodels.org, jev-ai.org, jevai.site, dsebastien.net, note.com)
continue to state that only the name and objective are public. Layer 2 of the chapter therefore
remains the book's reconstruction.

| Source | Used for | Status |
|---|---|---|
| TypeSafe AI blog (15 Sep 2026), raw HTML read | All Layer 1 quotes: stack description, RLHF/RLVR/RLCD comparison, "type-safe structured values", calibrated probabilities, "outputs all probabilities in parallel instead of autoregressively generating by token", speed/cost claims, "can't hallucinate", "never makes type errors", "193.6x faster, 444.6x cheaper", caveats ("Extraordinary claims require extraordinary evidence", OpenRouter bias "almost certainly", "We can't prove it isn't subsidized"), FAQ answers (data made in-house; "neither small nor an LLM"; no public benchmarks by choice) | ✓ raw page text |
| Guo et al. (2026), arXiv 2609.29429 | Detection results (abstract); calibration paragraph verbatim: "calibrated when pooled but not within a benchmark", ECE 0.047 pooled, 0.168 vs 0.074 null, 24/31, base-rate mismatch 0.125, AUROC 0.905. ("NOUL" is one of the paper's question types, not a typo.) | ✓ PDF text (resolves the Phase 1 "re-check by hand" item) |
| Wu et al. (2025), arXiv 2512.19920 | "encouraging guessing whenever correctness probability exceeds zero"; proper scoring rules; abstention / claim flagging | ✓ (Phase 1) |
| Damani et al. (2025), RLCR, arXiv 2507.16806 | Reward $\mathbb{1}_{y\equiv y^*} - (q - \mathbb{1}_{y\equiv y^*})^2$; boundedness condition $S(p,1) - S(p,0) < \lambda$; log-loss can favor "incorrect answers with zero confidence"; four-part output; ECE 0.37 → 0.03 on HotpotQA; OOD results | ✓ arXiv HTML |
| Gneiting & Raftery (2007), JASA 102(477):359–378 | Proper scoring rules | ✓ bibliographic (RePEc/mindat listing) |
| Zhu (2026), RefGRPO, arXiv 2606.14211 | Reflection gap; free calibration bonus | ✓ (Phase 1) |
| Gao & Wang (2026), OpenJev-RLCD, arXiv 2609.38850 | Prop. 1 (variance decomposition), Cor. 1 ("RLVR + Gini bonus"), per-rationale objective; "we claim nothing about Jev's internal algorithm" | ✓ PDF text; **both identities re-derived and checked numerically** (`tests/test_calibration.py`) |
| Yang et al. (2023), RLCD (contrastive distillation), ICLR 2024 | Positive/negative prompts → preference pairs → preference model → RL | ✓ abstract |

Experiment (`rl4llm/calibration.py`, `fig_calibration_reward.py`; 2,500 steps RLOO, G = 16, 3 seeds;
best achievable accuracy 0.654): reference acc 0.365 / ECE 0.408 / confident-wrong 0.508; binary
0.627 / 0.333 / 0.318; Brier-only 0.471 / 0.033 / 0.071; correctness + Brier 0.611 / 0.052 / 0.072.
Abstention at threshold ≥ 0.7: combined coverage 0.49 at accuracy 0.86; binary coverage 0.45 at
0.68. Exact optima (tested): binary and combined pick the best answer; combined states the bucket
nearest its success probability; Brier-only picks a surely-wrong answer on three of four prompts.

### Ch 11: What RL Actually Does to a Model (2026-10-01)

| Source | Used for | Status |
|---|---|---|
| Yue et al. (2025), arXiv 2504.13837 (NeurIPS 2025 oral per arXiv comments) | Small-k vs large-k quotes; "originate from and are bounded by the base model" | ✓ (Phase 1) |
| Liu et al. (2025), ProRL, arXiv 2505.24864 | Counter-evidence: prolonged RL with KL control and reference resets; "consistently outperform base models across a wide range of pass@k evaluations"; correlation with base competence and duration | ✓ abstract (NeurIPS 2025 poster per search result; cited as arXiv) |
| Chu et al. (2025) | Abstract quotes on RL generalizing / SFT memorizing; SFT stabilizes format | ✓ abstract |
| Kirk et al. (2023), arXiv 2310.06452 | "RLHF generalises better than SFT to new inputs..."; "significantly reduces output diversity" | ✓ abstract (venue not confirmed; cited as arXiv) |
| Shenfeld, Pari, Agrawal (2025) | RL's Razor quotes | ✓ (Phase 1) |
| Korbak et al. (2022) | "distribution collapse" | ✓ (Phase 1) |
| Zhou et al. (2024), ArCHer, arXiv 2402.19446 | Single-turn RL limits quote; hierarchical value-based multi-turn RL | ✓ abstract |
| Zhu (2026); Baker et al. (2025); Noukhovitch et al. (2025) | Agentic calibration bonus; environmental reward hacking; async off-policyness | ✓ (earlier chapters) |

Experiment (`fig_pass_at_k.py`): per-prompt π*_β has pass@k ≥ π_ref at every k (mathematical fact:
pass@k increasing in p). Shared-distribution toy (one policy, five competing tasks, training
frequencies 0.40/0.30/0.15/0.10/0.05): unregularized RLOO → task success [0.999, 0, 0, 0, 0] from
base [0.22, 0.15, 0.28, 0.18, 0.18]; frequency-weighted pass@1 0.20 → 0.40, pass@256 1.0 → 0.42,
crossover from k = 3. Labelled in the chapter as a deliberately extreme mechanism illustration, not a
reproduction of Yue et al.
