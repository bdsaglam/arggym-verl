"""The verl parquet holds exactly the rows, prompts and gold answers ArgGYM's selection gives.

GSM8K: the screen output is committed under data/, so a missing or partial screen fails
here (rl.gsm8k.screened checks all 8,792 questions are present) rather than skipping.
ArgGYM: experiment 003's pool under data/arggym-pool; without its train.jsonl the
ArgGYM tests fail (rl.pool.load_split refuses a missing split).
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


@pytest.fixture(scope="module")
def arggym_trl_and_verl(tmp_path_factory):
    """Experiment 003's rows from rl.pool, and the parquet arggym_verl.data writes from them.

    Needs the pool's train.jsonl under data/arggym-pool (README, "Data"); fails without it.
    """
    from conftest import ARGGYM_POOL_ROOT, TRL_ARGGYM
    from rl import config as rl_config
    from rl import pool

    cfg = rl_config.load(str(TRL_ARGGYM), [f"output_root={ARGGYM_POOL_ROOT}"])
    trl = {"train": pool.restrict(pool.load_split(cfg, "train"), cfg["train_cells"]),
           "dev": pool.load_split(cfg, "dev")}
    out = tmp_path_factory.mktemp("arggym")
    data.write(data.build(cfg), out, {})
    return cfg, trl, {s: pd.read_parquet(out / f"{s}.parquet") for s in ("train", "dev")}


@pytest.mark.parametrize("split", ["train", "dev"])
def test_arggym_same_ids_as_003(arggym_trl_and_verl, split):
    _, trl, verl = arggym_trl_and_verl
    ids = [e["id"] for e in verl[split]["extra_info"]]
    assert ids == [r["id"] for r in trl[split]]
    manifest = json.loads((ROOT / "data/arggym-qwen3.5-4b/manifest.json").read_text())
    assert manifest["ids"][split] == ids


def test_arggym_trains_on_the_57_cells_and_evaluates_on_all_96(arggym_trl_and_verl):
    cfg, _, verl = arggym_trl_and_verl
    listed = {(t, lv) for t, levels in cfg["train_cells"].items() for lv in levels}
    train = {(e["task"], e["level"]) for e in verl["train"]["extra_info"]}
    dev = [(e["task"], e["level"]) for e in verl["dev"]["extra_info"]]
    assert len(listed) == 57 and train == listed
    assert len(verl["train"]) == 5700
    assert len(dev) == len(set(dev)) == 96


def test_arggym_prompts_are_003s(arggym_trl_and_verl):
    from rl import pool

    cfg, trl, verl = arggym_trl_and_verl
    elicit = pool.elicitation(cfg["prompt"]["elicitation"])
    for split in ("train", "dev"):
        want = pool.trl_rows(trl[split], cfg["prompt"]["template"], elicit)
        for r, prompt, extra in zip(want, verl[split]["prompt"], verl[split]["extra_info"],
                                    strict=True):
            assert [dict(m) for m in prompt] == r["prompt"]
            assert extra["row_json"] == r["row_json"]
