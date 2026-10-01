"""Direct Preference Optimization (Rafailov et al. 2023), offline, on Bradley-Terry pairs.

Method card (Chapter 7):

    Signal                pairwise preferences y+ > y-, generated here by a Bradley-Terry
                          model of the true reward: P(y1 > y2) = sigmoid(r(y1) - r(y2))
    Samples from (q)      a fixed dataset of pairs drawn from pi_ref (offline: never resampled)
    Per-sample weight (w) +beta * sigmoid(-margin) on y+, the same with a minus sign on y-,
                          where margin = beta[log pi/pi_ref (y+) - log pi/pi_ref (y-)]:
                          "weighted by how wrong the implicit ordering is"
    Stability             the reference log-ratio inside the loss; no sampling, no clip
    Credit granularity    sequence level
    Models in memory      policy, reference (log-probs can be precomputed)
    Targets               pi* exactly, in the infinite-data limit with full-support pairs
                          (``population_gradient`` + the test suite check this); with finite
                          data, the optimum of the *empirical* loss instead
    Classical lineage     none in classical RL: logistic regression (Bradley-Terry MLE) on an
                          implicit reward beta log(pi/pi_ref); targets an RL optimum without RL
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from rl4llm.methods._common import Adam, History
from rl4llm.toy_language import (
    grad_log_likelihood,
    optimal_logprobs,
    sample_sequences,
    sequence_logprobs,
)


def make_preference_pairs(lang, ref_logits, reward, n_pairs_per_prompt, rng):
    """Sample (prompt, chosen, rejected) triples: responses from pi_ref, labels from Bradley-Terry."""
    P = lang.n_prompts
    prompts = np.repeat(np.arange(P), n_pairs_per_prompt)
    y1 = sample_sequences(lang, ref_logits, rng, prompts)
    y2 = sample_sequences(lang, ref_logits, rng, prompts)
    p_first = expit(reward[prompts, y1] - reward[prompts, y2])
    first_wins = rng.random(len(prompts)) < p_first
    chosen = np.where(first_wins, y1, y2)
    rejected = np.where(first_wins, y2, y1)
    return prompts, chosen, rejected


def dpo_gradient(lang, logits, ref_logp, beta, prompts, chosen, rejected, variant="dpo", tau=0.5,
                 simpo_gamma=0.5):
    """Ascent direction of the mean preference objective over the given pairs.

    ``variant``:
      "dpo"    -log sigmoid(beta * (h_w - h_l)), h = log pi/pi_ref       (Rafailov et al. 2023)
      "ipo"    (h_w - h_l - 1/(2 tau))^2                                 (Azar et al. 2023)
      "simpo"  -log sigmoid(beta/|y| (log pi_w - log pi_l) - gamma), no reference (Meng et al. 2024);
               here all responses have length L, so |y| is a constant
    Every variant returns +coef on grad log pi(y_w) and -coef on grad log pi(y_l): they differ only in
    the per-pair weight, the "w" of the book's (q, w) form.
    """
    logp = sequence_logprobs(lang, logits)
    h = logp - ref_logp  # implicit reward / beta
    if variant == "dpo":
        margin = beta * (h[prompts, chosen] - h[prompts, rejected])
        coef = beta * expit(-margin)  # large when the implicit ordering is wrong
    elif variant == "ipo":
        gap = h[prompts, chosen] - h[prompts, rejected]
        coef = -2.0 * (gap - 1.0 / (2.0 * tau))  # regress the gap to a fixed target
    elif variant == "simpo":
        k = beta / lang.L
        margin = k * (logp[prompts, chosen] - logp[prompts, rejected]) - simpo_gamma
        coef = k * expit(-margin)
    else:
        raise ValueError(variant)
    idx_p = np.concatenate([prompts, prompts])
    idx_y = np.concatenate([chosen, rejected])
    w = np.concatenate([coef, -coef])
    return grad_log_likelihood(lang, logits, idx_p, idx_y, w) / len(prompts)


def population_gradient(lang, logits, ref_logp, reward, beta, pair_logp=None):
    """Exact gradient of the *expected* DPO log-likelihood: infinite data, pairs from ``pair_logp``.

    Each ordered pair (y, y') is drawn with probability mu(y) mu(y') (``mu = pi_ref`` by default)
    and labelled by Bradley-Terry on ``reward``. Used by the tests to show that the population
    minimizer is exactly pi*.
    """
    mu = np.exp(ref_logp if pair_logp is None else pair_logp)  # (P, N)
    h = sequence_logprobs(lang, logits) - ref_logp
    P, N = h.shape
    w = np.empty((P, N))
    for p in range(P):
        M = mu[p][:, None] * mu[p][None, :]
        S = expit(beta * (h[p][:, None] - h[p][None, :])) - expit(reward[p][:, None] - reward[p][None, :])
        w[p] = -2.0 * beta * (M * S).sum(axis=1)  # ascent weight on log pi(y)
    prompts = np.repeat(np.arange(P), N)
    seqs = np.tile(np.arange(N), P)
    return grad_log_likelihood(lang, logits, prompts, seqs, w.ravel()) / P


def ipo_target_logprobs(ref_logp, reward, tau, pair_logp=None):
    """Exact optimum of IPO's population objective: pi_ref tilted by the *preference probability*.

    Azar et al.'s Psi-PO with Psi = identity maximizes E_{y~pi, y'~mu}[p(y > y')] - tau KL(pi||pi_ref),
    whose optimum is pi_ref(y) exp(p(y > mu) / tau) / Z with p(y > mu) = E_{y'~mu} sigmoid(r(y) - r(y')).
    This is a different target from DPO's pi* (which tilts by r / beta), unless preferences happen to
    make the two coincide.
    """
    from rl4llm.toy_language import optimal_logprobs

    mu = np.exp(ref_logp if pair_logp is None else pair_logp)
    pref = np.einsum("pj,pij->pi", mu, expit(reward[:, :, None] - reward[:, None, :]))
    return optimal_logprobs(ref_logp, pref, tau)


def train(
    lang,
    ref_logits,
    reward,
    beta,
    steps=500,
    n_pairs_per_prompt=2000,
    batch_size=256,
    lr=0.05,
    seed=0,
    eval_every=10, lr_final_frac=1.0, variant="dpo", tau=0.5, online=False, callback=None):
    """Offline DPO-family training on a fixed pair dataset, or ``online=True``: every step draws a
    fresh batch of pairs from the *current* policy and labels it (online / iterative DPO with an
    on-the-fly annotator, as in OAIF). ``callback(step, logits, data)`` is called at eval steps."""
    rng = np.random.default_rng(seed)
    ref_logp = sequence_logprobs(lang, ref_logits)
    opt_logp = optimal_logprobs(ref_logp, reward, beta)
    data = None if online else make_preference_pairs(lang, ref_logits, reward, n_pairs_per_prompt, rng)
    logits, opt, hist = ref_logits.copy(), Adam(lr, total_steps=steps, final_frac=lr_final_frac), History()
    for step in range(steps + 1):
        if step % eval_every == 0:
            hist.record(step, lang, logits, ref_logp, opt_logp, reward)
            if callback is not None:
                callback(step, logits, data)
        if step == steps:
            break
        if online:
            pb, cb, rb = make_preference_pairs(lang, logits, reward, batch_size // lang.n_prompts, rng)
        else:
            b = rng.choice(len(data[0]), size=batch_size, replace=False)
            pb, cb, rb = data[0][b], data[1][b], data[2][b]
        grad = dpo_gradient(lang, logits, ref_logp, beta, pb, cb, rb, variant=variant, tau=tau)
        logits = opt.step(logits, grad)
    return logits, hist


def train_population(lang, ref_logits, reward, beta, steps=500, lr=0.05, eval_every=10, lr_final_frac=1.0):
    """DPO on the *expected* loss (infinite preference data, pairs from pi_ref): no data noise."""
    ref_logp = sequence_logprobs(lang, ref_logits)
    opt_logp = optimal_logprobs(ref_logp, reward, beta)
    logits, opt, hist = ref_logits.copy(), Adam(lr, total_steps=steps, final_frac=lr_final_frac), History()
    for step in range(steps + 1):
        if step % eval_every == 0:
            hist.record(step, lang, logits, ref_logp, opt_logp, reward)
        if step < steps:
            logits = opt.step(logits, population_gradient(lang, logits, ref_logp, reward, beta))
    return logits, hist
