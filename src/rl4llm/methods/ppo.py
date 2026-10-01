"""PPO-based RLHF: actor, critic, reference and reward, as in Stiennon et al. (2020) / Ouyang et al. (2022).

Method card (Chapter 6):

    Signal                scalar reward per response, plus a per-token KL penalty folded into the
                          reward: r_t = -beta log(pi_old(y_t|s_t)/pi_ref(y_t|s_t)), and r(x, y) added
                          at the last token
    Samples from (q)      pi_old: G responses per prompt, reused for several epochs (slightly
                          off-policy)
    Per-sample weight (w) per token: rho_t * A_t where the clip does not bind, 0 where it does
                          (d/dtheta of min(rho A, clip(rho) A) = A * rho * grad log pi when active)
    Stability             ratio clipping; KL to pi_ref in the reward
    Credit granularity    token level: GAE advantages from a learned critic
    Models in memory      policy, reference, reward (model), critic
    Targets               the KL-regularized objective; optimum pi*
    Classical lineage     PPO (actor-critic + GAE + clipped trust region) on the token-level MDP

The critic is tabular over (prompt, prefix) states, initialized to the reference policy's expected
reward per prompt (Stiennon et al. initialize theirs from the reward model), and regressed toward
the GAE returns A_t + V(s_t) each epoch.
"""

from __future__ import annotations

import numpy as np

from rl4llm.estimators import gae, ppo_clip_grad_ratio
from rl4llm.methods._common import Adam, History
from rl4llm.metrics import expected
from rl4llm.toy_language import (
    grad_log_likelihood,
    optimal_logprobs,
    sample_sequences,
    sequence_logprobs,
    token_logprobs,
)


def train(
    lang,
    ref_logits,
    reward,
    beta,
    steps=300,
    group_size=8,
    epochs=4,
    eps=0.2,
    lam=0.95,
    lr=0.05,
    critic_lr=0.5,
    seed=0,
    eval_every=5, lr_final_frac=1.0):
    rng = np.random.default_rng(seed)
    P, G, L = lang.n_prompts, group_size, lang.L
    ref_logp = sequence_logprobs(lang, ref_logits)
    opt_logp = optimal_logprobs(ref_logp, reward, beta)
    ref_tok_all = token_logprobs(lang, ref_logits)
    critic = np.repeat(expected(ref_logp, reward)[:, None], lang.n_prefixes, axis=1)
    logits, opt, hist = ref_logits.copy(), Adam(lr, total_steps=steps * epochs, final_frac=lr_final_frac), History()
    prompts = np.repeat(np.arange(P), G)
    pref = None
    for step in range(steps + 1):
        if step % eval_every == 0:
            hist.record(step, lang, logits, ref_logp, opt_logp, reward)
        if step == steps:
            break
        seqs = sample_sequences(lang, logits, rng, prompts)
        pref = lang.prefix_index[seqs]  # (n, L) state of each token
        old_tok = token_logprobs(lang, logits)[prompts, seqs]
        ref_tok = ref_tok_all[prompts, seqs]
        r_t = -beta * (old_tok - ref_tok)  # per-token KL penalty in the reward (k1)
        r_t[:, -1] += reward[prompts, seqs]
        values = np.concatenate([critic[prompts[:, None], pref], np.zeros((len(seqs), 1))], axis=1)
        adv = gae(r_t, values, gamma=1.0, lam=lam)
        returns = adv + values[:, :-1]
        for _ in range(epochs):
            new_tok = token_logprobs(lang, logits)[prompts, seqs]
            ratio = np.exp(new_tok - old_tok)
            w = ppo_clip_grad_ratio(ratio, adv, eps) * ratio
            logits = opt.step(logits, grad_log_likelihood(lang, logits, prompts, seqs, w) / (len(seqs) * L))
            # Critic: tabular regression toward the GAE returns (average error per visited state).
            err = np.zeros_like(critic)
            cnt = np.zeros_like(critic)
            np.add.at(err, (np.repeat(prompts[:, None], L, 1), pref), returns - critic[prompts[:, None], pref])
            np.add.at(cnt, (np.repeat(prompts[:, None], L, 1), pref), 1.0)
            critic += critic_lr * err / np.maximum(cnt, 1.0)
    return logits, hist
