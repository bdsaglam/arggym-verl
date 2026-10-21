"""On fixed completions, the verl reward pays what the TRL reward pays, through verl's own plumbing."""

import numpy as np
import pytest
import torch

from arggym_verl.reward import ArgGymRewardManager, score_completion
from conftest import QWEN3_06B

EOS, PAD, OTHER = 151645, 151643, 42  # Qwen3: <|im_end|>, <|endoftext|>, any token
STOP = {EOS, PAD}

CASES = [  # (text, gold, ids)
    ("<think>\n12 + 6 = 18\n</think>\n\nThe answer is \\boxed{18}.", 18.0, [OTHER, EOS]),
    ("so \\boxed{17}", 18.0, [OTHER, EOS]),
    ("first \\boxed{3}, then \\boxed{18}", 18.0, [OTHER, EOS]),
    ("\\boxed{\\$1,234}", 1234.0, [OTHER, EOS]),
    ("\\boxed{\\frac{1}{2}}", 0.5, [OTHER, EOS]),
    ("no box at all: 18", 18.0, [OTHER, EOS]),
    ("cut off but \\boxed{18}", 18.0, [OTHER, OTHER]),
    ("ends on padding \\boxed{-3.50}", -3.5, [OTHER, PAD]),
    ("", 1.0, []),
]


def trl_rewards():
    from rl import gsm8k

    fn = gsm8k.make_reward(sorted(STOP), gather=lambda x: x)
    texts, golds, ids = zip(*CASES)
    return fn(list(texts), list(ids), list(golds), id=[f"q{i}" for i in range(len(CASES))])


def test_gsm8k_reward_matches_trl():
    ours = [score_completion("gsm8k", t, ids, {"gold": g}, STOP)["score"]
            for t, g, ids in CASES]
    assert ours == trl_rewards()


def test_truncated_pays_zero_and_counts_as_failure():
    info = score_completion("gsm8k", "\\boxed{18}", [OTHER], {"gold": 18.0}, STOP)
    assert (info["score"], info["acc"], info["truncated"]) == (0.0, 0.0, 1.0)


@pytest.mark.skipif(not QWEN3_06B.exists(), reason="Qwen3-0.6B tokenizer not cached")
def test_manager_decodes_the_sampled_ids_as_trl_does():
    """verl's reward-loop input: padded responses plus an attention mask; text is decoded
    with special tokens skipped, as TRL decodes completions."""
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer

    from verl import DataProto

    tok = AutoTokenizer.from_pretrained(str(QWEN3_06B))
    text = "<think>\n6 * 3 = 18\n</think>\n\nSo \\boxed{18}."
    ids = tok(text, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    width = len(ids) + 5
    responses = torch.tensor([ids + [tok.pad_token_id] * 5])
    mask = torch.tensor([[1] * len(ids) + [0] * 5])
    data = DataProto.from_dict(
        tensors={"responses": responses, "attention_mask": mask,
                 "prompts": torch.zeros(1, 3, dtype=torch.long)},
        non_tensors={"data_source": np.array(["gsm8k"], dtype=object),
                     "extra_info": np.array([{"gold": 18.0, "split": "dev", "id": "test-0"}],
                                            dtype=object)})
    assert responses.shape[1] == width
    manager = ArgGymRewardManager(OmegaConf.create({}), tok, None)
    out = manager.loop.run_until_complete(manager.run_single(data))
    assert out["reward_score"] == 1.0
    assert out["reward_extra_info"]["truncated"] == 0.0
    assert out["reward_extra_info"]["length"] == len(ids)
    from rl import gsm8k

    trl_text = tok.batch_decode([ids], skip_special_tokens=True)[0]
    assert gsm8k.correct(trl_text, 18.0)


def arggym_dev_rows():
    """Six dev rows of experiment 003's pool, from the committed copy (conftest.ARGGYM_POOL_ROOT)."""
    from conftest import ARGGYM_POOL_ROOT, TRL_ARGGYM
    from rl import config as rl_config
    from rl import pool

    cfg = rl_config.load(str(TRL_ARGGYM), [f"output_root={ARGGYM_POOL_ROOT}"])
    elicit = pool.elicitation(cfg["prompt"]["elicitation"])
    return pool.trl_rows(pool.load_split(cfg, "dev")[:6], cfg["prompt"]["template"], elicit), \
        cfg["prompt"]["template"]


def test_arggym_reward_matches_trl():
    """Same scorer and payout: a truncated completion pays 0 in both (rl-003); only a row
    the scorer refuses, None in TRL, pays 0 here."""
    from rl import reward as arggym_reward

    rows, template = arggym_dev_rows()
    texts = ["<think>\nhmm\n</think>\n\n<answer>yes</answer>", "no tags at all",
             "<think>\nloop loop", "</think><answer></answer>", "<answer>A</answer>", ""]
    ids = [[OTHER, EOS], [OTHER, EOS], [OTHER, OTHER], [OTHER, PAD], [OTHER, EOS], []]
    trl = arggym_reward.make_reward(sorted(STOP), template, gather=lambda x: x)(
        texts, ids, [r["row_json"] for r in rows])
    ours = [score_completion("arggym", t, i, {"row_json": r["row_json"], "template": template},
                             STOP) for t, i, r in zip(texts, ids, rows)]
    for t, o in zip(trl, ours, strict=True):
        assert o["score"] == (0.0 if t is None else t)
        assert (t is None) == (o["refused"] == 1.0)
    assert ours[2]["truncated"] == 1.0 and trl[2] == 0.0 and ours[2]["score"] == 0.0
