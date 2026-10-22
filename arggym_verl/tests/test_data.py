"""The verl parquet holds exactly the rows, prompts and gold answers of ArgGYM's selection.

The expected values are ArgGYM's own outputs (fixtures/, the committed manifests and
pool summary); the parquet is written here by `arggym_verl.data` from the vendored code.
GSM8K reads the committed screen and the Hub's openai/gsm8k. ArgGYM reads experiment
003's pool under data/arggym-pool; without its train.jsonl (scripts/arggym/prepare_data.sh)
the ArgGYM tests fail.
"""
import hashlib
import json

import pandas as pd
import pytest

from arggym_verl import data
from conftest import ARGGYM_POOL_ROOT, ROOT, TASK_ARGGYM, TASK_GSM8K, fixture


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def written(cfg, tmp_path_factory, name):
    out = tmp_path_factory.mktemp(name)
    data.write(data.build(cfg), out, {})
    return {s: pd.read_parquet(out / f"{s}.parquet") for s in ("train", "dev")}


def messages(prompt):
    return [dict(m) for m in prompt]


@pytest.fixture(scope="module")
def gsm8k(tmp_path_factory):
    from arggym_verl.vendor import rl_config

    return written(rl_config.load(str(TASK_GSM8K)), tmp_path_factory, "gsm8k")


@pytest.mark.parametrize("split", ["train", "dev"])
def test_gsm8k_rows_prompts_and_gold_are_arggyms(gsm8k, split):
    df, want = gsm8k[split], fixture("gsm8k_prompts.json")
    got = [[e["id"], messages(p), e["gold"]] for p, e in zip(df["prompt"], df["extra_info"])]
    assert got[:2] == [[r["id"], r["prompt"], r["gold"]] for r in want["examples"][split]]
    assert sha(got) == want["sha256"][split]
    assert (df["agent_name"] == split).all()


@pytest.mark.parametrize("split", ["train", "dev"])
def test_gsm8k_committed_manifest_has_the_same_ids(gsm8k, split):
    """The parquet a run reads was written with these ids (data/gsm8k-hard/manifest.json)."""
    manifest = json.loads((ROOT / "data/gsm8k-hard/manifest.json").read_text())
    assert manifest["ids"][split] == [e["id"] for e in gsm8k[split]["extra_info"]]


def test_gsm8k_dev_is_200_screened_test_rows(gsm8k):
    ids = [e["id"] for e in gsm8k["dev"]["extra_info"]]
    assert len(ids) == len(set(ids)) == 200
    assert all(i.startswith("test-") for i in ids)
    assert all(e["id"].startswith("train-") for e in gsm8k["train"]["extra_info"])


@pytest.fixture(scope="module")
def arggym_cfg():
    from arggym_verl.vendor import rl_config

    return rl_config.load(str(TASK_ARGGYM), [f"output_root={ARGGYM_POOL_ROOT}"])


@pytest.fixture(scope="module")
def arggym(arggym_cfg, tmp_path_factory):
    return written(arggym_cfg, tmp_path_factory, "arggym")


@pytest.mark.parametrize("split", ["train", "dev"])
def test_pool_is_003s(arggym_cfg, split):
    """The pool's rows hash to what ArgGYM's rl.pool recorded when it built 003's pool."""
    from arggym.core.freeze import taskset_hash

    from arggym_verl.vendor import rl_pool

    assert rl_pool.pool_dir(arggym_cfg).name == "0df63cbebe86"
    summary = json.loads((rl_pool.pool_dir(arggym_cfg) / "summary.json").read_text())
    assert taskset_hash(rl_pool.load_split(arggym_cfg, split)) == summary[split]["taskset_hash"]


@pytest.mark.parametrize("split", ["train", "dev"])
def test_arggym_ids_prompts_and_rows_are_003s(arggym, split):
    df, want = arggym[split], fixture("arggym_prompts.json")
    got = [[e["id"], messages(p)] for p, e in zip(df["prompt"], df["extra_info"])]
    assert got[:2] == [[r["id"], r["prompt"]] for r in want["examples"][split]]
    assert sha(got) == want["sha256"][split]
    assert sha([e["row_json"] for e in df["extra_info"]]) == want["row_json_sha256"][split]
    manifest = json.loads((ROOT / "data/arggym-qwen3.5-4b/manifest.json").read_text())
    assert manifest["ids"][split] == [e["id"] for e in df["extra_info"]]


def test_arggym_trains_on_the_57_cells_and_evaluates_on_all_96(arggym_cfg, arggym):
    listed = {(t, lv) for t, levels in arggym_cfg["train_cells"].items() for lv in levels}
    train = {(e["task"], e["level"]) for e in arggym["train"]["extra_info"]}
    dev = [(e["task"], e["level"]) for e in arggym["dev"]["extra_info"]]
    assert len(listed) == 57 and train == listed
    assert len(arggym["train"]) == 5700
    assert len(dev) == len(set(dev)) == 96


def test_pool_refuses_seed_windows_that_overlap_the_eval_tasksets(arggym_cfg):
    """rl.pool's guard, on the eval specs shipped in arggym_verl/vendor/conf."""
    from arggym_verl.vendor import rl_config, rl_pool

    rl_pool.check_windows(arggym_cfg)
    bad = rl_config.apply(arggym_cfg, ["data.dev.seed_start=0"])
    with pytest.raises(SystemExit, match="seed windows overlap"):
        rl_pool.check_windows(bad)


def test_eval_keys_cover_both_eval_tasksets():
    """7,200 standard rows (the lite taskset is a subset): one question key each."""
    from arggym_verl.vendor import rl_pool

    questions, theories = rl_pool.eval_keys()
    assert len(questions) == 7200 and 0 < len(theories) <= 7200
