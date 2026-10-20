"""The verl parquet holds exactly the rows, prompts and gold answers ArgGYM's selection gives.

The screen output is committed under data/, so a missing or partial screen fails
here (rl.gsm8k.screened checks all 8,792 questions are present) rather than skipping.
"""
import json

import pandas as pd
import pytest

from arggym_verl import data
from conftest import ROOT, SCREEN, TRL_GSM8K


@pytest.fixture(scope="module")
def trl_and_verl(tmp_path_factory):
    from rl import config as rl_config
    from rl import gsm8k

    cfg = rl_config.load(str(TRL_GSM8K), [f"data.screen.path={SCREEN}"])
    train, dev = gsm8k.load_splits(cfg["data"])
    out = tmp_path_factory.mktemp("gsm8k")
    data.write(data.build(cfg), out, {})
    return ({"train": gsm8k.trl_rows(train), "dev": gsm8k.trl_rows(dev)},
            {s: pd.read_parquet(out / f"{s}.parquet") for s in ("train", "dev")})


@pytest.mark.parametrize("split", ["train", "dev"])
def test_same_ids_in_the_same_order(trl_and_verl, split):
    trl, verl = trl_and_verl
    assert [r["id"] for r in trl[split]] == [e["id"] for e in verl[split]["extra_info"]]


@pytest.mark.parametrize("split", ["train", "dev"])
def test_same_prompts_and_gold(trl_and_verl, split):
    trl, verl = trl_and_verl
    df = verl[split]
    for r, prompt, extra, agent in zip(trl[split], df["prompt"], df["extra_info"],
                                       df["agent_name"], strict=True):
        assert [dict(m) for m in prompt] == r["prompt"]
        assert extra["gold"] == r["gold"]
        assert agent == split


def test_dev_is_200_screened_test_rows(trl_and_verl):
    _, verl = trl_and_verl
    ids = [e["id"] for e in verl["dev"]["extra_info"]]
    assert len(ids) == len(set(ids)) == 200
    assert all(i.startswith("test-") for i in ids)
    assert all(e["id"].startswith("train-") for e in verl["train"]["extra_info"])


@pytest.mark.parametrize("split", ["train", "dev"])
def test_committed_manifest_has_the_same_ids(trl_and_verl, split):
    """The parquet a run reads was written with these ids (data/gsm8k-hard/manifest.json)."""
    _, verl = trl_and_verl
    manifest = json.loads((ROOT / "data/gsm8k-hard/manifest.json").read_text())
    assert manifest["ids"][split] == [e["id"] for e in verl[split]["extra_info"]]
