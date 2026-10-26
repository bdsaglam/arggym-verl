"""On fixed completions, the verl reward pays what ArgGYM's TRL reward pays, through verl's own plumbing.

ArgGYM's rewards and scorer records come from fixtures/, written by its own code.
"""

import json

import numpy as np
import pytest
import torch

from arggym_verl.reward import ArgGymRewardManager, score_completion
from conftest import QWEN3_06B, fixture

EOS, PAD, OTHER = 151645, 151643, 42  # Qwen3: <|im_end|>, <|endoftext|>, any token
STOP = {EOS, PAD}



def test_gsm8k_reward_matches_arggyms():
    """ArgGYM's TRL reward (rl.gsm8k.make_reward) on fixed completions, from fixtures/."""
    want = fixture("gsm8k_rewards.json")
    stop = set(want["stop_ids"])
    for case in want["cases"]:
        info = score_completion("gsm8k", case["text"], case["ids"], {"gold": case["gold"]}, stop)
        assert info["score"] == case["reward"], case["text"]
        assert info["boxed"] == float(case["boxed"] is not None)


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
    from arggym_verl.vendor import rl_gsm8k as gsm8k

    trl_text = tok.batch_decode([ids], skip_special_tokens=True)[0]
    assert gsm8k.correct(trl_text, 18.0)


def dev_rows():
    """Experiment 003's dev rows by id, from the committed pool."""
    from arggym_verl.vendor import rl_config, rl_pool
    from conftest import ARGGYM_POOL_ROOT, TASK_ARGGYM

    cfg = rl_config.load(str(TASK_ARGGYM), [f"output_root={ARGGYM_POOL_ROOT}"])
    return {r["id"]: r for r in rl_pool.load_split(cfg, "dev")}


def test_arggym_reward_matches_arggyms():
    """ArgGYM's reward and its eval scorer's record on seven completions of one dev row per
    task: the reference answer finished, truncated, inside the reasoning only and without
    tags; a wrong answer; an unfinished thought; nothing. A truncated completion pays 0
    in both; a row the scorer refused would pay 0 here and None in TRL."""
    from arggym_verl.vendor import rl_reward

    want, rows = fixture("arggym_rewards.json"), dev_rows()
    stop = set(want["stop_ids"])
    for case in want["cases"]:
        row = rows[case["row_id"]]
        extra = {"row_json": json.dumps(row), "template": want["template"]}
        ours = score_completion("arggym", case["text"], case["ids"], extra, stop)
        assert ours["score"] == (0.0 if case["reward"] is None else case["reward"]), case["case"]
        assert ours["refused"] == float(case["reward"] is None)
        rec = rl_reward.score(case["text"], row, bool(ours["truncated"]), want["template"])
        assert {k: rec[k] for k in case["record"]} == case["record"], (case["row_id"], case["case"])
    assert {c["case"] for c in want["cases"] if c["reward"] == 1.0} == {
        "reference", "reference_in_reasoning_only", "reference_without_tags"}


def test_verl_loads_the_manager_from_its_file_path():
    """verl imports reward.py from its path, outside sys.modules (reward_manager.source
    importlib); a dataclass defined in that module fails to build there."""
    from pathlib import Path

    from verl.utils.import_utils import load_extern_object

    path = Path(__file__).resolve().parents[1] / "reward.py"
    assert load_extern_object(str(path), "ArgGymRewardManager").__name__ == "ArgGymRewardManager"
