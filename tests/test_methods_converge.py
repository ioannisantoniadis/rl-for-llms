"""Every implemented route toward pi* actually gets there on the testbed (Chapters 6-9)."""

import pytest

from rl4llm.methods import distill, grpo, ppo, reinforce
from rl4llm.toy_language import logits_from_sequence_logprobs, optimal_logprobs

BETA, STEPS, DECAY = 0.5, 400, 0.02


@pytest.mark.parametrize(
    "name, run",
    [
        ("rloo", lambda l, ref, r: reinforce.train(l, ref, r, BETA, steps=STEPS, lr_final_frac=DECAY)),
        ("ppo", lambda l, ref, r: ppo.train(l, ref, r, BETA, steps=STEPS, lr=0.02, lr_final_frac=DECAY)),
        ("grpo_kl_in_reward", lambda l, ref, r: grpo.train(l, ref, r, BETA, steps=STEPS, kl="k1_reward",
                                                           epochs=2, lr_final_frac=DECAY)),
    ],
)
def test_on_policy_methods_reach_pi_star(lang, ref_logits, reward, name, run):
    _, hist = run(lang, ref_logits, reward)
    assert hist.kl_to_opt[-1] < 0.05, name


def test_on_policy_distillation_reaches_teacher(lang, ref_logits, ref_logp, reward):
    teacher = logits_from_sequence_logprobs(lang, optimal_logprobs(ref_logp, reward, BETA))
    _, hist = distill.train(lang, ref_logits, teacher, reward, steps=STEPS, lr_final_frac=DECAY)
    assert hist.kl_to_opt[-1] < 0.05


def test_grpo_with_k3_in_the_loss_does_not_target_pi_star(lang, ref_logits, reward):
    _, hist = grpo.train(lang, ref_logits, reward, BETA, steps=STEPS, kl="k3_loss", epochs=2,
                         lr_final_frac=DECAY)
    assert hist.kl_to_opt[-1] > 0.1


def test_reward_weighted_regression_reaches_pi_star(lang, ref_logits, reward):
    from rl4llm.methods import rft

    _, hist = rft.train_weighted(lang, ref_logits, reward, BETA, steps=STEPS, n_samples_per_prompt=50_000,
                                 batch_size=512, lr_final_frac=DECAY)
    assert hist.kl_to_opt[-1] < 0.05
