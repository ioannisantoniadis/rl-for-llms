"""Token-level vs sequence-level credit: where does the credit for a terminal reward actually live?

This figure makes visible that with one reward at the end of a response, the sequence-level view
(the bandit view: one advantage r - V(x), broadcast to every token) puts the same credit on every
token, while the exact token-level advantage A(s_t, y_t) = V(s_{t+1}) - V(s_t) concentrates it on
the tokens that actually decide the outcome. Both views give the same expected gradient; they
differ in how the credit is spread within one sample, which is what critics, process rewards and
dense teacher signals try to recover.

Real computation: exact values V^pi(s_t) under pi_ref by enumeration (rl4llm.metrics.prefix_values),
for two verifiable rewards on the same language: the default one (tokens sum to a target mod 5,
decided only by the last token) and a variant decided by the first token (first token equals the
target). Shown: E_pi |advantage| at each position, averaged over prompts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
from _theme import FAMILY_COLOR, INK_SECONDARY, MUTED, apply_theme, savefig

from rl4llm.metrics import prefix_values
from rl4llm.toy_language import default_testbed, sequence_logprobs

apply_theme()
lang, ref, sum_reward = default_testbed()
ref_logp = sequence_logprobs(lang, ref)
first_reward = (lang.sequences[None, :, 0] == lang.targets[:, None]).astype(float)
p = np.exp(ref_logp)


def credit(reward):
    v = prefix_values(lang, ref_logp, reward)  # (P, N, L+1)
    tok = v[:, :, 1:] - v[:, :, :-1]  # exact per-token advantage (gamma = 1, terminal reward)
    seq = reward[..., None] - v[:, :, :1]  # sequence-level advantage, same for every token
    tok_mag = (p[..., None] * np.abs(tok)).sum(1).mean(0)
    seq_mag = (p[..., None] * np.abs(np.repeat(seq, lang.L, axis=2))).sum(1).mean(0)
    # Sanity: the per-token advantages telescope to the sequence advantage.
    assert np.allclose(tok.sum(-1), seq[..., 0])
    return tok_mag, seq_mag


fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), sharey=True)
pos = np.arange(1, lang.L + 1)
for ax, reward, title in ((axes[0], sum_reward, "Reward: tokens sum to the target (mod 5)\ncredit spread over all positions"),
                          (axes[1], first_reward, "Reward: first token equals the target\ncredit entirely on token 1")):
    tok_mag, seq_mag = credit(reward)
    w = 0.38
    ax.bar(pos - w / 2, seq_mag, width=w, color=MUTED, label="sequence level: $r - V(x)$ on every token")
    ax.bar(pos + w / 2, tok_mag, width=w, color=FAMILY_COLOR["policy_gradient"],
           label="token level: exact $A(s_t, y_t) = V(s_{t+1}) - V(s_t)$")
    ax.set_xticks(pos, [f"token {t}" for t in pos])
    ax.set_title(title, fontsize=11.5)
    for x, v in zip(pos + w / 2, tok_mag):
        ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=8.5, color=INK_SECONDARY)
axes[0].set_ylabel(r"mean $|\mathrm{advantage}|$ under $\pi_{\mathrm{ref}}$")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.06))
savefig(fig, "where_credit_lives")
