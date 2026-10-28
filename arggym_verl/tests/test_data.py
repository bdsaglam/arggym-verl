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


# ---------------------------------------------------------------- v3: arggym-004, levels 1-12

TASK_004 = ROOT / "arggym_verl/tasks/arggym-004.yaml"


@pytest.fixture(scope="module")
def cfg004():
    from arggym_verl.vendor import rl_config

    return rl_config.load(str(TASK_004), [f"output_root={ARGGYM_POOL_ROOT}"])


@pytest.fixture(scope="module")
def v3(cfg004, tmp_path_factory):
    return written(cfg004, tmp_path_factory, "arggym-v3")


def test_004_pool_holds_003s_rows_on_levels_1_to_8(arggym_cfg, cfg004):
    """Each (task, level, ordering) cell is frozen on its own seed scan, so the 1-12 pool's
    L1-8 rows are 003's, row for row; its dev split keeps 003's 96 rows on L1-8. Only
    `metadata.source_index`, a row's position in its file, differs, as between ArgGYM's
    lite and standard tasksets (vendor/conf/taskset-lite.yaml)."""
    from arggym.core.freeze import taskset_hash

    from arggym_verl.vendor import rl_pool

    def content(r):
        return json.dumps({**r, "metadata": {k: v for k, v in r["metadata"].items()
                                             if k != "source_index"}}, sort_keys=True, default=str)

    assert rl_pool.pool_dir(cfg004).name == "16f6114d9306"
    for split in ("train", "dev"):
        new, old = rl_pool.load_split(cfg004, split), rl_pool.load_split(arggym_cfg, split)
        summary = json.loads((rl_pool.pool_dir(cfg004) / "summary.json").read_text())
        assert taskset_hash(new) == summary[split]["taskset_hash"]
        low = [r for r in new if r["metadata"]["level"] <= 8]
        assert [content(r) for r in low] == [content(r) for r in old], split
    train = rl_pool.load_split(cfg004, "train")
    assert len(train) == 12 * 12 * 4 * 25


def test_v3_trains_on_its_82_cells_and_evaluates_003s_96_dev_rows(arggym_cfg, cfg004, v3):
    listed = {(t, lv) for t, levels in cfg004["train_cells"].items() for lv in levels}
    train = [(e["task"], e["level"]) for e in v3["train"]["extra_info"]]
    assert len(listed) == 82 and set(train) == listed and len(train) == 8200
    manifest003 = json.loads((ROOT / "data/arggym-qwen3.5-4b/manifest.json").read_text())
    dev = [e["id"] for e in v3["dev"]["extra_info"]]
    assert dev == manifest003["ids"]["dev"] and len(dev) == 96
    # The cells both runs train on have the same rows, in the same order.
    both = listed & {(t, lv) for t, levels in arggym_cfg["train_cells"].items() for lv in levels}
    assert len(both) == 56
    old = [i for i in manifest003["ids"]["train"] if (i.split("/")[0], int(i.split("/")[1][1:])) in both]
    new = [e["id"] for e in v3["train"]["extra_info"] if (e["task"], e["level"]) in both]
    assert new == old
    manifest = json.loads((ROOT / "data/arggym-qwen3.5-4b-v3/manifest.json").read_text())
    assert manifest["ids"] == {"train": [e["id"] for e in v3["train"]["extra_info"]], "dev": dev}
    assert manifest["dev_levels"] == list(range(1, 9))


def test_v3_dev_without_dev_levels_is_all_144_cells(cfg004):
    from arggym_verl.vendor import rl_config

    rows = data.build(rl_config.apply(cfg004, ["dev_levels=null"]))["dev"]
    cells = [(e["extra_info"]["task"], e["extra_info"]["level"]) for e in rows]
    assert len(cells) == len(set(cells)) == 144
