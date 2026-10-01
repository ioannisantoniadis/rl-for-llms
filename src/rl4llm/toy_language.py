"""The enumerable toy language: the shared testbed for every experiment in the book.

Why this exists
---------------
Real LLM post-training can only be judged by proxies: a benchmark score, a reward-model score,
an estimated KL. This testbed is small enough that **every** quantity the theory talks about can
be computed exactly, by enumeration:

- the response space has ``V ** L`` elements (625 by default), so a policy's full distribution
  over responses fits in a small array;
- the closed-form optimum of the KL-regularized objective, ``pi*(y|x) ∝ pi_ref(y|x) exp(r/beta)``,
  its normalizer ``Z(x)``, exact KLs, exact expected rewards and the exact policy gradient are
  all a few lines of numpy.

Every algorithm in ``rl4llm.methods`` can therefore be measured against ground truth (for
example, ``KL(pi_theta || pi*)`` over training) instead of being eyeballed.

The pieces
----------
``ToyLanguage``
    The vocabulary, the response length, the prompts, and the bookkeeping that maps every
    response to the prefixes it passes through.

The policy
    A **prefix-tabular autoregressive policy**: one softmax over the next token for every
    (prompt, prefix) pair. Its parameters are an array of logits with shape
    ``(n_prompts, n_prefixes, V)``. This is the "lookup table" end of the model spectrum. It is
    fully expressive (any distribution over responses can be represented exactly, via the chain
    rule), which is what lets every method be checked for convergence to the *exact* ``pi*``
    rather than to "the best ``pi*`` the architecture can represent". It is still autoregressive
    and token-level, so token-level credit, per-token ratios and per-token KLs mean the same thing
    here as in an LLM.

The reference policy
    ``make_reference`` "pretrains" the policy by maximum likelihood on a synthetic corpus: each
    prompt has its own Markov chain over tokens, a corpus is sampled from it, and the tabular MLE
    (counts plus additive smoothing) is the resulting ``pi_ref``. It has a genuine non-uniform
    prior, and it is *not* the corpus distribution (finite data, smoothing). That mirrors the real
    situation, where ``pi_ref`` is whatever pretraining produced.

Rewards
    ``correctness_reward`` is a verifiable rule ("the tokens sum to the prompt's target, mod V"),
    the toy analogue of an RLVR checker. Other rewards (a learned proxy, a graded reward) are
    added by the chapters that need them.

Conventions
-----------
Log-probabilities are natural logs. Arrays indexed by response use the lexicographic order of
``ToyLanguage.sequences``. "Per prompt" arrays have the prompt as their leading axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

__all__ = [
    "ToyLanguage",
    "correctness_reward",
    "default_testbed",
    "eos_testbed",
    "grad_log_likelihood",
    "kl_regularized_objective",
    "log_partition",
    "log_softmax",
    "logits_from_sequence_logprobs",
    "make_reference",
    "optimal_logprobs",
    "sample_sequences",
    "sequence_logprobs",
    "token_logprobs",
]


# ---------------------------------------------------------------------------------------------
# The language
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ToyLanguage:
    """Vocabulary size ``V``, response length ``L``, and ``n_prompts`` prompts (contexts).

    Attributes computed on construction:

    ``sequences``      ``(N, L)`` int array of every response, lexicographic order, ``N = V**L``.
    ``prefix_index``   ``(N, L)`` int array: ``prefix_index[i, t]`` is the index of the prefix
                       ``sequences[i, :t]`` (the *state* in which token ``t`` is chosen).
    ``n_prefixes``     number of distinct prefixes of length ``0 .. L-1``:
                       ``1 + V + ... + V**(L-1)``.
    ``targets``        ``(n_prompts,)`` the verifiable target of each prompt (used by
                       ``correctness_reward``). Pass ``target_values`` to choose them; the
                       default is ``0, 1, 2, ...`` mod ``V``.
    """

    V: int = 5
    L: int = 4
    n_prompts: int = 4
    target_values: tuple[int, ...] | None = None
    eos: bool = False
    sequences: np.ndarray = field(init=False, repr=False)
    prefix_index: np.ndarray = field(init=False, repr=False)
    n_prefixes: int = field(init=False)
    targets: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        V, L = self.V, self.L
        # All V**L responses, lexicographic: the response with index i is i written in base V.
        idx = np.arange(V**L)
        seqs = np.stack([(idx // V ** (L - 1 - t)) % V for t in range(L)], axis=1)
        # Prefixes of length t occupy a contiguous block starting at offset[t]; inside the block a
        # prefix is identified by its base-V value.
        offsets = np.concatenate([[0], np.cumsum([V**t for t in range(L)])])
        pidx = np.zeros((V**L, L), dtype=np.int64)
        for t in range(L):
            value = np.zeros(V**L, dtype=np.int64)
            for s in range(t):
                value = value * V + seqs[:, s]
            pidx[:, t] = offsets[t] + value
        object.__setattr__(self, "sequences", seqs)
        object.__setattr__(self, "prefix_index", pidx)
        object.__setattr__(self, "n_prefixes", int(offsets[L]))
        if self.target_values is None:
            targets = np.arange(self.n_prompts) % V
        else:
            targets = np.asarray(self.target_values, dtype=np.int64)
            assert len(targets) == self.n_prompts, "one target per prompt"
        object.__setattr__(self, "targets", targets)
        # Variable-length variant: token V-1 is end-of-sequence. A response ends at its first EOS
        # (inclusive); every position after it is padding whose next-token distribution is forced
        # to EOS, so it carries probability 1 and no gradient.
        if self.eos:
            is_eos = seqs == V - 1
            first = np.where(is_eos.any(1), is_eos.argmax(1), L - 1)
            object.__setattr__(self, "lengths", first + 1)
            object.__setattr__(self, "token_mask", (np.arange(L)[None, :] <= first[:, None]).astype(float))
            # Prefixes that already contain an EOS: their conditionals are forced.
            post = np.zeros(int(offsets[L]), bool)
            for t in range(1, L):
                post[pidx[:, t][is_eos[:, :t].any(1)]] = True
            object.__setattr__(self, "post_eos_prefix", post)
        else:
            object.__setattr__(self, "lengths", np.full(V**L, L))
            object.__setattr__(self, "token_mask", np.ones((V**L, L)))
            object.__setattr__(self, "post_eos_prefix", np.zeros(int(offsets[L]), bool))

    @property
    def n_sequences(self) -> int:
        return self.V**self.L

    @property
    def logits_shape(self) -> tuple[int, int, int]:
        """Shape of a prefix-tabular policy's parameter array."""
        return (self.n_prompts, self.n_prefixes, self.V)

    def clamp_eos(self, logits: np.ndarray) -> np.ndarray:
        """Force EOS after an EOS (no-op unless ``eos``). Returns a new array."""
        if not self.eos:
            return logits
        out = logits.copy()
        out[:, self.post_eos_prefix, :] = -50.0
        out[:, self.post_eos_prefix, self.V - 1] = 0.0
        return out

    def decode(self, i: int) -> str:
        """Human-readable form of response ``i``, e.g. ``"a c c e"``."""
        return " ".join("abcdefghijklmnopqrstuvwxyz"[k] for k in self.sequences[i])


# ---------------------------------------------------------------------------------------------
# The prefix-tabular policy
# ---------------------------------------------------------------------------------------------


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax over the last axis."""
    return logits - logsumexp(logits, axis=-1, keepdims=True)


def token_logprobs(lang: ToyLanguage, logits: np.ndarray) -> np.ndarray:
    """``(P, N, L)`` array of ``log pi(y_t | x, y_<t)`` for every prompt, response and position."""
    lp = log_softmax(logits)  # (P, n_prefixes, V)
    # Gather the log-prob of the token actually chosen at each position of each response.
    return lp[:, lang.prefix_index, lang.sequences]


def sequence_logprobs(lang: ToyLanguage, logits: np.ndarray) -> np.ndarray:
    """``(P, N)`` array of ``log pi(y | x)``: the chain rule, a sum of per-token log-probs."""
    return token_logprobs(lang, logits).sum(axis=-1)


def grad_log_likelihood(
    lang: ToyLanguage,
    logits: np.ndarray,
    prompt_idx: np.ndarray,
    seq_idx: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Gradient w.r.t. ``logits`` of ``sum_i w_i * log pi(y_i | x_i)``.

    This one function is the book's Thesis 2 in code. SFT, REINFORCE, PPO, GRPO, DPO, rejection
    sampling and distillation all call it; they differ only in *which* ``(x_i, y_i)`` they pass
    (the sampling distribution ``q``) and *which* ``weights`` (the per-sample coefficient ``w``).

    ``weights`` may be ``(n,)`` (one weight per response, broadcast to every token: sequence-level
    credit) or ``(n, L)`` (one weight per token: token-level credit, as with a critic or a teacher).

    For a softmax, ``d log softmax(z)_k / dz = onehot(k) - softmax(z)``, accumulated at every
    prefix the response passes through.
    """
    prompt_idx = np.asarray(prompt_idx)
    seq_idx = np.asarray(seq_idx)
    w = np.asarray(weights, dtype=float)
    if w.ndim == 1:
        w = np.repeat(w[:, None], lang.L, axis=1)
    probs = np.exp(log_softmax(logits))  # (P, n_prefixes, V)
    pref = lang.prefix_index[seq_idx]  # (n, L)
    toks = lang.sequences[seq_idx]  # (n, L)
    pr = np.repeat(prompt_idx[:, None], lang.L, axis=1)  # (n, L)
    grad = np.zeros_like(logits)
    # -w * softmax at every visited (prompt, prefix) ...
    np.add.at(grad, (pr, pref), -w[..., None] * probs[pr, pref])
    # ... plus w on the token actually taken.
    np.add.at(grad, (pr, pref, toks), w)
    return grad


def logits_from_sequence_logprobs(lang: ToyLanguage, seq_logp: np.ndarray) -> np.ndarray:
    """Inverse of ``sequence_logprobs``: the prefix-tabular logits of a joint distribution.

    Given ``(P, N)`` log-probabilities over whole responses, return logits whose per-prefix
    conditionals reproduce it exactly (the chain rule run backwards). Prefixes of zero mass get
    uniform conditionals. Used to represent ``pi*`` (or any target) as a policy.
    """
    P, V, L = seq_logp.shape[0], lang.V, lang.L
    logits = np.zeros(lang.logits_shape)
    joint = seq_logp.reshape((P,) + (V,) * L)
    offsets = np.concatenate([[0], np.cumsum([V**t for t in range(L)])])
    for t in range(L):
        # Log-mass of each prefix of length t+1: marginalize out the remaining positions.
        axes = tuple(range(t + 2, L + 1))
        m = logsumexp(joint, axis=axes) if axes else joint  # (P, V, ..., V) with t+1 V-axes
        m = m.reshape(P, V**t, V)  # (prompt, prefix of length t, next token)
        logits[:, offsets[t] : offsets[t + 1], :] = np.where(np.isfinite(m), m, -1e9)
    return logits


def sample_sequences(
    lang: ToyLanguage,
    logits: np.ndarray,
    rng: np.random.Generator,
    prompt_idx: np.ndarray,
) -> np.ndarray:
    """Sample one response per entry of ``prompt_idx``. Returns response indices.

    Sampling is done from the exact joint distribution (it is enumerable), which is
    distributionally identical to sampling token by token.
    """
    prompt_idx = np.asarray(prompt_idx)
    probs = np.exp(sequence_logprobs(lang, logits))
    out = np.empty(len(prompt_idx), dtype=np.int64)
    for p in np.unique(prompt_idx):
        mask = prompt_idx == p
        out[mask] = rng.choice(lang.n_sequences, size=int(mask.sum()), p=probs[p] / probs[p].sum())
    return out


# ---------------------------------------------------------------------------------------------
# The reference policy: "pretraining" by maximum likelihood on a synthetic corpus
# ---------------------------------------------------------------------------------------------


def make_reference(
    lang: ToyLanguage,
    seed: int = 0,
    corpus_size: int = 4000,
    smoothing: float = 0.5,
    concentration: float = 0.6,
) -> np.ndarray:
    """Return the logits of ``pi_ref``, fit by tabular MLE to a synthetic corpus.

    Each prompt gets its own first-token distribution and token-transition matrix, drawn from a
    Dirichlet with the given ``concentration`` (small values give peaky, "opinionated" chains). A
    corpus of ``corpus_size`` responses per prompt is sampled from that Markov chain, and the
    prefix-tabular model is fit by maximum likelihood with additive ``smoothing`` (the MAP
    estimate under a symmetric Dirichlet prior), so every response keeps nonzero probability.
    """
    rng = np.random.default_rng(seed)
    V, L = lang.V, lang.L
    counts = np.zeros(lang.logits_shape)
    for p in range(lang.n_prompts):
        init = rng.dirichlet(np.full(V, concentration))
        trans = rng.dirichlet(np.full(V, concentration), size=V)
        seq = np.empty((corpus_size, L), dtype=np.int64)
        seq[:, 0] = rng.choice(V, size=corpus_size, p=init)
        for t in range(1, L):
            u = rng.random(corpus_size)
            cdf = np.cumsum(trans[seq[:, t - 1]], axis=1)
            seq[:, t] = np.minimum((u[:, None] > cdf).sum(axis=1), V - 1)
        if lang.eos:
            after = np.cumsum(seq == V - 1, axis=1) - (seq == V - 1) > 0
            seq[after] = V - 1
        seq_id = np.zeros(corpus_size, dtype=np.int64)
        for t in range(L):
            seq_id = seq_id * V + seq[:, t]
        np.add.at(counts, (p, lang.prefix_index[seq_id], seq), 1.0)
    return lang.clamp_eos(np.log(counts + smoothing))


# ---------------------------------------------------------------------------------------------
# Rewards and the exact optimum
# ---------------------------------------------------------------------------------------------


def correctness_reward(lang: ToyLanguage) -> np.ndarray:
    """``(P, N)`` verifiable reward: 1 if the response's tokens sum to the prompt's target mod V."""
    sums = lang.sequences.sum(axis=1) % lang.V
    return (sums[None, :] == lang.targets[:, None]).astype(float)


def log_partition(ref_logp: np.ndarray, reward: np.ndarray, beta: float) -> np.ndarray:
    """``(P,)`` array of ``log Z(x) = log sum_y pi_ref(y|x) exp(r(x,y)/beta)``."""
    return logsumexp(ref_logp + reward / beta, axis=1)


def optimal_logprobs(ref_logp: np.ndarray, reward: np.ndarray, beta: float) -> np.ndarray:
    """Exact ``log pi*(y|x)`` for the KL-regularized objective (the book's hub, Chapter 6).

    ``pi*(y|x) = pi_ref(y|x) exp(r(x,y)/beta) / Z(x)``: the reference policy, exponentially tilted
    by the reward and renormalized.
    """
    if beta <= 0:  # the beta -> 0 limit: pi_ref restricted to the highest-reward responses
        best = reward >= reward.max(axis=1, keepdims=True)
        lp = np.where(best, ref_logp, -np.inf)
        return lp - logsumexp(lp, axis=1, keepdims=True)
    return ref_logp + reward / beta - log_partition(ref_logp, reward, beta)[:, None]


def kl_regularized_objective(
    logp: np.ndarray, ref_logp: np.ndarray, reward: np.ndarray, beta: float
) -> np.ndarray:
    """``(P,)`` exact ``E_{y~pi}[r] - beta KL(pi || pi_ref)`` per prompt."""
    p = np.exp(logp)
    return (p * (reward - beta * (logp - ref_logp))).sum(axis=1)


# ---------------------------------------------------------------------------------------------
# The canonical testbed used by every figure in the book
# ---------------------------------------------------------------------------------------------

#: Seed and Dirichlet concentration of the corpus-generating Markov chains, and the target of
#: each prompt. Chosen (by scanning seeds, see ROADMAP.md) so the four prompts span a wide range of
#: difficulty under pi_ref: P(correct) is about 0.02, 0.19, 0.71 and 0.44 for prompts 0..3.
#: Experiments about difficulty (group normalization, zero-variance groups, pass@k) need a hard
#: and an easy prompt.
DEFAULT_SEED = 1
DEFAULT_CONCENTRATION = 0.3
DEFAULT_TARGETS = (1, 2, 0, 0)


def default_testbed() -> tuple[ToyLanguage, np.ndarray, np.ndarray]:
    """Return ``(lang, ref_logits, reward)``: the testbed every chapter's experiments share.

    Changing anything here changes pi_ref and pi* for every figure; regenerate them all.
    """
    lang = ToyLanguage(target_values=DEFAULT_TARGETS)
    ref_logits = make_reference(lang, seed=DEFAULT_SEED, concentration=DEFAULT_CONCENTRATION)
    return lang, ref_logits, correctness_reward(lang)


def eos_testbed(L: int = 6, seed: int = 4, concentration: float = 0.6):
    """Variable-length testbed for length effects (Chapter 8): V = 5 with token 4 = EOS, responses of
    1..L tokens. Reward: 1 if the content tokens (before EOS) sum to the prompt's target mod 4 and
    the response has at least one content token. Length is never rewarded directly.
    """
    lang = ToyLanguage(V=5, L=L, n_prompts=4, eos=True, target_values=(0, 1, 2, 3))
    ref_logits = make_reference(lang, seed=seed, concentration=concentration)
    content = np.where(lang.sequences == lang.V - 1, 0, lang.sequences) * (lang.token_mask > 0)
    sums = content.sum(1) % 4
    has_content = lang.lengths > 1
    reward = ((sums[None, :] == lang.targets[:, None]) & has_content[None, :]).astype(float)
    return lang, ref_logits, reward
