"""Rejection-sampling fine-tuning: imitate samples filtered to look like pi*.

Method card (Chapter 9):

    Signal                scalar reward, used only to accept or reject samples
    Samples from (q)      pi_ref (one offline round here; iterated variants resample from the
                          current policy: STaR, ReST, RAFT, expert iteration)
    Per-sample weight (w) 1 for accepted samples, 0 for rejected ones
    Stability             none needed: it is supervised learning on a fixed dataset
    Credit granularity    sequence level
    Models in memory      policy (+ the reference only as the proposal sampler)
    Targets               with acceptance probability exp((r - r_max)/beta), the accepted
                          samples are *exact* draws from pi* (rejection sampling), so SFT on them
                          targets pi*; hard filters (best-of-N, threshold) target a different,
                          sharper distribution
    Classical lineage     rejection sampling + behavior cloning; reward-weighted regression
                          (Peters & Schaal 2007) with a 0/1 weight
"""

from __future__ import annotations

import numpy as np

from rl4llm.methods._common import Adam, History
from rl4llm.toy_language import (
    grad_log_likelihood,
    optimal_logprobs,
    sample_sequences,
    sequence_logprobs,
)


def rejection_sample(lang, proposal_logits, reward, beta, n_proposals_per_prompt, rng):
    """Draw from pi_ref and accept with probability exp((r - max r)/beta).

    The accepted samples are distributed exactly as pi_ref * exp(r/beta) / Z = pi*.
    Returns (prompts, responses) of the accepted samples.
    """
    P = lang.n_prompts
    prompts = np.repeat(np.arange(P), n_proposals_per_prompt)
    ys = sample_sequences(lang, proposal_logits, rng, prompts)
    r = reward[prompts, ys]
    r_max = reward.max(axis=1)[prompts]
    accept = rng.random(len(ys)) < np.exp((r - r_max) / beta)
    return prompts[accept], ys[accept]


def sft_gradient(lang, logits, prompts, seqs):
    """Ascent direction of the mean log-likelihood of a dataset: w = 1 on every sample."""
    return grad_log_likelihood(lang, logits, prompts, seqs, np.ones(len(seqs))) / len(seqs)


def train(
    lang,
    ref_logits,
    reward,
    beta,
    steps=500,
    n_proposals_per_prompt=4000,
    batch_size=256,
    lr=0.05,
    seed=0,
    eval_every=10, lr_final_frac=1.0):
    rng = np.random.default_rng(seed)
    ref_logp = sequence_logprobs(lang, ref_logits)
    opt_logp = optimal_logprobs(ref_logp, reward, beta)
    prompts, seqs = rejection_sample(lang, ref_logits, reward, beta, n_proposals_per_prompt, rng)
    logits, opt, hist = ref_logits.copy(), Adam(lr, total_steps=steps, final_frac=lr_final_frac), History()
    for step in range(steps + 1):
        if step % eval_every == 0:
            hist.record(step, lang, logits, ref_logp, opt_logp, reward)
        if step == steps:
            break
        b = rng.choice(len(seqs), size=min(batch_size, len(seqs)), replace=False)
        logits = opt.step(logits, sft_gradient(lang, logits, prompts[b], seqs[b]))
    return logits, hist


def train_weighted(lang, ref_logits, reward, beta, steps=500, n_samples_per_prompt=4000,
                   batch_size=256, lr=0.05, seed=0, eval_every=10, lr_final_frac=1.0):
    """Reward-weighted regression with exponential weights (Peters & Schaal 2007; Peng et al. 2019):
    supervised learning on samples from pi_ref, each weighted by exp(r / beta) (self-normalized per
    prompt). Its population optimum is pi_ref exp(r/beta) / Z = pi*: a soft version of the exact
    rejection sampler above, using every sample instead of discarding some."""
    rng = np.random.default_rng(seed)
    ref_logp = sequence_logprobs(lang, ref_logits)
    opt_logp = optimal_logprobs(ref_logp, reward, beta)
    P = lang.n_prompts
    prompts = np.repeat(np.arange(P), n_samples_per_prompt)
    ys = sample_sequences(lang, ref_logits, rng, prompts)
    w = np.exp((reward[prompts, ys] - reward.max(1)[prompts]) / beta)
    w = w / np.bincount(prompts, weights=w, minlength=P)[prompts] * n_samples_per_prompt
    logits, opt, hist = ref_logits.copy(), Adam(lr, total_steps=steps, final_frac=lr_final_frac), History()
    for step in range(steps + 1):
        if step % eval_every == 0:
            hist.record(step, lang, logits, ref_logp, opt_logp, reward)
        if step == steps:
            break
        b = rng.choice(len(ys), size=batch_size, replace=False)
        g = grad_log_likelihood(lang, logits, prompts[b], ys[b], w[b]) / batch_size
        logits = opt.step(logits, g)
    return logits, hist


def train_iterative_filter(lang, ref_logits, reward, beta, iterations=8, samples_per_prompt=64,
                           sft_steps=40, batch_size=128, lr=0.05, seed=0):
    """Iterated hard filtering: the STaR / ReST / expert-iteration pattern (Zelikman et al. 2022;
    Gulcehre et al. 2023; Anthony et al. 2017). Each iteration samples from the *current* policy,
    keeps only responses with reward 1 (w = 1[correct], negative samples get no gradient), and
    fine-tunes on them for ``sft_steps`` steps. ``beta`` is used only to report KL to pi*_beta.

    Returns the final logits and a History recorded once per iteration.
    """
    rng = np.random.default_rng(seed)
    ref_logp = sequence_logprobs(lang, ref_logits)
    opt_logp = optimal_logprobs(ref_logp, reward, beta)
    P = lang.n_prompts
    logits, hist = ref_logits.copy(), History()
    hist.record(0, lang, logits, ref_logp, opt_logp, reward)
    for it in range(1, iterations + 1):
        prompts = np.repeat(np.arange(P), samples_per_prompt)
        ys = sample_sequences(lang, logits, rng, prompts)
        keep = reward[prompts, ys] > 0
        kp, ky = prompts[keep], ys[keep]
        opt = Adam(lr)
        for _ in range(sft_steps if len(ky) else 0):
            b = rng.choice(len(ky), size=min(batch_size, len(ky)), replace=False)
            logits = opt.step(logits, sft_gradient(lang, logits, kp[b], ky[b]))
        hist.record(it, lang, logits, ref_logp, opt_logp, reward)
    return logits, hist
