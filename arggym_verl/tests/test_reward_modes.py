"""`reward.mode`: binary pays what the reward paid before the modes existed, partial and
native pay what reward.py says at every edge, and exact success stays what the
curriculum, dev and selection read."""
import json
from types import SimpleNamespace

import pytest

from arggym_verl.reward import BINARY, RewardShape, score_completion, step_metrics
from conftest import ROOT, fixture

BASELINE = "data/arggym-reward/floors-0df63cbebe86.json"
PARTIAL = {"mode": "partial", "partial_weight": 0.3, "partial_tasks": ["status_query"],
           "partial_baseline": BASELINE}


def compose(*overrides):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "arggym_verl/config"), version_base=None):
        return compose(config_name="arggym-qwen3.5-4b",
                       overrides=["hydra.searchpath=[pkg://verl.trainer.config]", *overrides])


def record(success, score, truncated=False, no_region=False):
    """`rl_reward.score`'s record, the fields the reward reads; success None is a refusal."""
    return {"success": success, "score": score, "truncated": truncated,
            "no_answer_region": no_region}


@pytest.fixture(scope="module")
def rows():
    from arggym_verl.vendor import rl_config, rl_pool
    from conftest import ARGGYM_POOL_ROOT, TASK_ARGGYM

    cfg = rl_config.load(str(TASK_ARGGYM), [f"output_root={ARGGYM_POOL_ROOT}"])
    return {r["id"]: r for r in rl_pool.load_split(cfg, "dev")}


# ---------------------------------------------------------------- binary is today's reward


@pytest.mark.parametrize("overrides", [(), ("reward.mode=binary",)])
def test_binary_is_byte_identical_to_the_reward_before_modes(rows, overrides):
    """Every logged field of every fixture completion, as train and as dev row, under the
    config's default mode, against the reward as it was before `reward.mode` existed."""
    shape = RewardShape.from_config(compose(*overrides).reward)
    assert shape == BINARY
    want, cases = fixture("arggym_rewards.json"), fixture("arggym_reward_infos_binary.json")["cases"]
    by_case = {(c["row_id"], c["case"]): c for c in want["cases"]}
    for case in cases:
        c = by_case[(case["row_id"], case["case"])]
        extra = {"row_json": json.dumps(rows[c["row_id"]]), "template": want["template"],
                 "split": case["split"]}
        info = score_completion("arggym", c["text"], c["ids"], extra, set(want["stop_ids"]), shape)
        assert json.dumps(info) == json.dumps(case["info"]), (case["row_id"], case["case"])


# ---------------------------------------------------------------- partial and native at the edges


@pytest.fixture(scope="module")
def partial():
    return RewardShape.from_config(PARTIAL)


def test_partial_edges(partial):
    b = partial.floors[("status_query", 1)]
    assert b == pytest.approx(1 / 3, abs=1e-3)  # all_justified on three claims, one each status
    pay = lambda rec: partial.pay(rec, "status_query", 1)  # noqa: E731
    assert pay(record(True, 1.0)) == 1.0
    assert pay(record(True, 1.0, truncated=True)) == 0.0
    assert pay(record(False, 0.9, truncated=True)) == 0.0
    assert pay(record(None, None)) == 0.0
    assert pay(record(False, 0.0)) == 0.0
    assert pay(record(False, b - 0.01)) == 0.0
    assert pay(record(False, b)) == 0.0
    assert pay(record(False, 0.8)) == pytest.approx(0.3 * (0.8 - b) / (1 - b))
    # An exact answer beats any near miss.
    assert pay(record(False, 0.9999)) < 0.3 < pay(record(True, 1.0))


def test_partial_floor_is_the_rows_level(partial):
    assert partial.floors[("status_query", 5)] != partial.floors[("status_query", 1)]
    s, b5 = 0.8, partial.floors[("status_query", 5)]
    assert partial.pay(record(False, s), "status_query", 5) == pytest.approx(0.3 * (s - b5) / (1 - b5))


def test_a_task_outside_the_list_stays_binary(partial):
    for task in ("semantics_query", "attack"):
        assert partial.pay(record(False, 0.9), task, 1) == 0.0
        assert partial.pay(record(True, 0.75), task, 1) == 1.0
        assert partial.pay(record(True, 0.75, truncated=True), task, 1) == 0.0


def test_native_edges():
    native = RewardShape.from_config({"mode": "native"})
    pay = lambda rec: native.pay(rec, "attack", 1)  # noqa: E731
    assert pay(record(True, 1.0)) == 1.0
    assert pay(record(True, 0.75)) == 0.75  # a construction success, half the efficiency term
    assert pay(record(False, 0.1)) == 0.1  # a construction failure's progress, capped at 0.25
    assert pay(record(False, 0.6)) == 0.6
    assert pay(record(True, 1.0, truncated=True)) == 0.0
    assert pay(record(False, 0.6, truncated=True)) == 0.0
    assert pay(record(None, None)) == 0.0
    assert pay(record(True, 1.0, no_region=True)) == 0.0
    assert pay(record(False, 0.4, no_region=True)) == 0.0


def status_answer(row, wrong: int) -> str:
    """The row's gold statuses, `wrong` of them changed."""
    gold = row["metadata"]["gold"]["gold"]
    flip = {"JUSTIFIED": "overruled", "OVERRULED": "undecided", "UNDECIDED": "justified"}
    lines = [f"{c}: {flip[s] if i < wrong else s.lower()}" for i, (c, s) in enumerate(gold.items())]
    return "<think>\nworking\n</think>\n\n<answer>\n" + "\n".join(lines) + "\n</answer>"


@pytest.mark.parametrize("mode", ["partial", "native"])
def test_scored_near_miss_is_paid_in_training_only(rows, partial, mode):
    """A real near miss through the eval's scorer: paid in training, 0 on dev, `acc` 0 in both."""
    shape = partial if mode == "partial" else RewardShape.from_config({"mode": "native"})
    row = next(r for r in rows.values() if r["task"] == "status_query")
    level, eos = row["metadata"]["level"], 151645
    text = status_answer(row, wrong=1)
    infos = {split: score_completion("arggym", text, [42, eos], {
        "row_json": json.dumps(row), "template": "xml_tags", "split": split}, {eos}, shape)
        for split in ("train", "dev")}
    s = infos["train"]["partial_score"]
    assert 0 < s < 1
    want = (0.3 * max(0.0, (s - partial.floors[("status_query", level)])
                      / (1 - partial.floors[("status_query", level)]))
            if mode == "partial" else s)
    assert infos["train"]["score"] == pytest.approx(want) and want > 0
    assert infos["dev"]["score"] == 0.0
    assert infos["train"]["acc"] == infos["dev"]["acc"] == 0.0
    exact = score_completion("arggym", status_answer(row, wrong=0), [42, eos], {
        "row_json": json.dumps(row), "template": "xml_tags", "split": "train"}, {eos}, shape)
    assert exact["score"] == exact["acc"] == 1.0


# ---------------------------------------------------------------- success stays success


def test_curriculum_counts_success_not_the_paid_reward():
    """`step_outcomes` reads `acc`, `truncated` and `refused`: partial pay is no pass."""
    from arggym_verl.curriculum import step_outcomes

    cur = SimpleNamespace(draws=[{"rows": [0]}], observed=0)
    infos = [{"index": 0, "id": "r0", "task": "status_query", "level": 1}] * 4
    rewards = [{"score": 1.0, "acc": 1.0, "truncated": 0.0, "refused": 0.0},
               {"score": 0.25, "acc": 0.0, "truncated": 0.0, "refused": 0.0},
               {"score": 0.1, "acc": 0.0, "truncated": 0.0, "refused": 0.0},
               {"score": 0.0, "acc": 1.0, "truncated": 1.0, "refused": 0.0}]
    assert step_outcomes(cur, infos, rewards) == [(("status_query", 1), 1, 4)]


def test_step_metrics_split_paid_reward_from_success():
    infos = [{"id": i, "task": t} for i, t in zip("aaaabbbbcccc", ["status_query"] * 8 + ["attack"] * 4)]
    rewards = [{"score": s, "acc": a, "truncated": 0.0, "refused": 0.0} for s, a in [
        (1.0, 1.0), (0.2, 0.0), (0.1, 0.0), (0.0, 0.0),   # a: success varies, reward varies
        (0.2, 0.0), (0.1, 0.0), (0.0, 0.0), (0.0, 0.0),   # b: all fail, partial pay varies
        (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]]  # c: all fail, all 0
    m = step_metrics(infos, rewards)
    assert m["reward/no_success_variance_share"] == pytest.approx(2 / 3)
    assert m["reward/no_reward_variance_share"] == pytest.approx(1 / 3)
    assert m["reward/success_mean"] == pytest.approx(1 / 12)
    assert m["reward/paid_mean"] == pytest.approx(1.6 / 12)
    assert m["reward/paid_mean/status_query"] == pytest.approx(1.6 / 8)
    assert m["reward/success_mean/status_query"] == pytest.approx(1 / 8)
    assert m["reward/success_mean/attack"] == 0.0


def test_step_metrics_leave_refusals_out_of_success():
    infos = [{"id": "a", "task": "t"}] * 2
    rewards = [{"score": 0.0, "acc": 0.0, "truncated": 0.0, "refused": 1.0},
               {"score": 1.0, "acc": 1.0, "truncated": 0.0, "refused": 0.0}]
    m = step_metrics(infos, rewards)
    assert m["reward/success_mean"] == 1.0
    assert m["reward/no_success_variance_share"] == 1.0
    assert m["reward/no_reward_variance_share"] == 0.0


# ---------------------------------------------------------------- config


def test_config_default_is_binary_and_the_partial_block_is_set():
    r = compose().reward
    assert r.mode == "binary"
    assert r.partial_weight == 0.3
    assert list(r.partial_tasks) == ["status_query", "defeat_diagnosis", "formalization"]
    assert (ROOT / r.partial_baseline).exists()


def test_gsm8k_config_has_no_mode_and_is_binary():
    from hydra import compose as hcompose, initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "arggym_verl/config"), version_base=None):
        cfg = hcompose(config_name="gsm8k-hard-qwen3-0.6b",
                       overrides=["hydra.searchpath=[pkg://verl.trainer.config]"])
    assert "mode" not in cfg.reward
    assert RewardShape.from_config(cfg.reward) == BINARY


def test_partial_override_resolves_floors_per_cell():
    shape = RewardShape.from_config(compose("reward.mode=partial").reward)
    assert shape.mode == "partial" and shape.partial_weight == 0.3
    assert shape.partial_tasks == {"status_query", "defeat_diagnosis", "formalization"}
    assert {t for t, _ in shape.floors} == shape.partial_tasks
    assert len(shape.floors) == 3 * 8
    cells = json.loads((ROOT / BASELINE).read_text())["by_task_level"]
    assert shape.floors[("defeat_diagnosis", 4)] == cells["defeat_diagnosis|L4"]["floor"]


def test_native_override():
    shape = RewardShape.from_config(compose("reward.mode=native").reward)
    assert shape.mode == "native"


@pytest.mark.parametrize("bad, match", [
    ({"mode": "shaped"}, "one of binary, partial, native"),
    ({**PARTIAL, "partial_weight": 1.0}, r"must be in \(0, 1\)"),
    ({**PARTIAL, "partial_weight": 0.0}, r"must be in \(0, 1\)"),
    ({**PARTIAL, "partial_tasks": []}, "no reward.partial_tasks"),
    ({**PARTIAL, "partial_tasks": ["status_query", "no_such_task"]}, "no floor for"),
])
def test_config_refusals(bad, match):
    with pytest.raises(ValueError, match=match):
        RewardShape.from_config(bad)


def test_manager_reads_the_mode_from_the_config():
    from omegaconf import OmegaConf

    from arggym_verl.reward import ArgGymRewardManager

    tok = SimpleNamespace(eos_token_id=1, pad_token_id=0)
    m = ArgGymRewardManager(OmegaConf.create({"reward": PARTIAL}), tok, None)
    assert m.shape == RewardShape.from_config(PARTIAL)
    assert ArgGymRewardManager(OmegaConf.create({}), tok, None).shape == BINARY


def test_baselines_file_says_its_source():
    data = json.loads((ROOT / BASELINE).read_text())
    assert "arggym.floors" in data["source"] and "0df63cbebe86" in data["source"]
    assert len(data["by_task_level"]) == 12 * 8
    assert all(0 <= c["floor"] < 1 and c["n"] == 100 for c in data["by_task_level"].values())
