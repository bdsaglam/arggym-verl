"""The v3 cell selection and curriculum prior (arggym_verl.prior): the paid rate by hand on
two cells, the committed prior file and arggym-004.yaml's train_cells reproduced from the
committed sources, and no test-set eval of a trained checkpoint among them."""
import json

import pytest
import yaml

from conftest import ROOT

PRIOR = ROOT / "data/arggym-prior/qwen3.5-4b-pooled-paid.json"
TASK_004 = ROOT / "arggym_verl/tasks/arggym-004.yaml"


@pytest.fixture(scope="module")
def table():
    from arggym_verl import prior

    return prior.load_table()


@pytest.fixture(scope="module")
def committed():
    return json.loads(PRIOR.read_text())


def test_paid_rate_by_hand(committed):
    """status_query L6 (L1-8, a partial task): p = 003's 3/32 + 0/8 + 0/8 dev = 3/48. The
    base eval's wrong answers average (0.8672 - 0.25) / 0.75 = 0.8229 against its floor
    0.375, near (0.8229 - 0.375) / 0.625 = 0.7167; the runs' 16 wrong dev answers sum
    6.2794 + 4.6599 near-miss terms, 0.6837 each; near is the mean, 0.7002, and
    q = 0.0625 + 0.9375 * 0.3 * 0.7002 = 0.2594.
    defeat_diagnosis L12 (L9-12): the base eval alone, 1/8; wrong answers average
    (0.4288 - 0.125) / 0.875 = 0.3472 against the floor 0.075, near 0.2943, q 0.2022."""
    sq = committed["by_task_level"]["status_query|L6"]
    near = ((0.8672 - 0.25) / 0.75 - 0.375) / 0.625 / 2 + (6.279352 + 4.659919) / 16 / 2
    assert (sq["successes"], sq["samples"]) == (3, 48)
    assert sq["near"] == pytest.approx(near) == pytest.approx(0.7002, abs=1e-4)
    assert sq["success_rate"] == pytest.approx(3 / 48 + (1 - 3 / 48) * 0.3 * near)
    assert sq["success_rate"] == pytest.approx(0.2594, abs=1e-4)
    assert sq["n_scored"] == 16 and sq["train"]
    dd = committed["by_task_level"]["defeat_diagnosis|L12"]
    near = ((0.4288 - 0.125) / 0.875 - 0.075) / 0.925
    assert dd["success_rate"] == pytest.approx(0.125 + 0.875 * 0.3 * near)
    assert dd["success_rate"] == pytest.approx(0.2022, abs=1e-4)
    assert (dd["n_scored"], dd["evidence"]) == (8, "base 1/8")


def test_prior_file_is_the_build_of_the_committed_sources(table, committed):
    from arggym_verl import prior

    assert json.loads(json.dumps(prior.prior(table))) == committed
    assert len(committed["by_task_level"]) == 12 * 12


def test_train_cells_are_the_rules(table):
    """arggym-004.yaml's train_cells are the cells `prior` selects: q within [0.15, 0.85], or
    [0.12, 0.85] on a partial task; 82 cells."""
    from arggym_verl import prior

    cells = yaml.safe_load(TASK_004.read_text())["train_cells"]
    assert cells == prior.train_cells(table)
    assert sum(map(len, cells.values())) == 82
    for k, c in table.items():
        task = k.split("|")[0]
        low = 0.12 if task in prior.PARTIAL_TASKS else 0.15
        assert c["train"] == (low <= c["q"] <= 0.85), k
        if task not in prior.PARTIAL_TASKS:
            assert c["q"] == c["p"] and c["near"] == 0


def test_003s_pooled_counts_give_its_57_cells():
    """The parser reads 003's comments right: its rule, pooled rate in [0.15, 0.85], gives
    its train_cells."""
    from arggym_verl import prior

    pooled = prior.pooled_003()
    assert len(pooled) == 96
    want = yaml.safe_load((ROOT / prior.TASK_003).read_text())["train_cells"]
    got = {}
    for (task, level), (s, n) in sorted(pooled.items()):
        got.setdefault(task, [])
        if 0.15 <= s / n <= 0.85:
            got[task].append(level)
    assert got == want


def test_sources_hold_no_test_eval_of_a_trained_checkpoint(committed):
    """The sources are the base model's eval and the training-time observations of the v2 and
    partial runs (training groups, step-20 dev)."""
    from arggym_verl import prior

    sources = sorted(p.name for p in (ROOT / "data/arggym-prior/sources").iterdir())
    assert sources == ["hf-qwen3.5-4b__xml_tags__cot.metrics.json",
                       "qwen3.5-4b-v2-partial-observed.json"]
    base = json.loads((ROOT / prior.BASE).read_text())["_meta"]
    assert base["endpoint"]["model"] == "Qwen/Qwen3.5-4B" and base["taskset"] == "data/taskset-lite.jsonl"
    observed = json.loads((ROOT / prior.OBSERVED).read_text())["sources"]
    assert sorted(observed) == ["arggym-qwen3.5-4b-partial dev step 20",
                                "arggym-qwen3.5-4b-partial train steps 1-34",
                                "arggym-qwen3.5-4b-v2 dev step 20",
                                "arggym-qwen3.5-4b-v2 train steps 1-20"]
    # L9-12 rest on the base eval alone.
    for k, c in committed["by_task_level"].items():
        if int(k.split("|L")[1]) > 8:
            assert c["evidence"].startswith("base ") and c["samples"] == 8


def test_tally_pays_what_the_reward_pays():
    """A pass is a success that did not truncate; a wrong answer's near-miss term is
    max(0, (s - b) / (1 - b)), 0 when truncated; a refusal is left out."""
    from arggym_verl.prior import tally

    recs = [{"id": "t/L2/o/s1", "acc": 1, "truncated": 0, "refused": 0, "partial_score": 1.0},
            {"id": "t/L2/o/s1", "acc": 1, "truncated": 1, "refused": 0, "partial_score": 1.0},
            {"id": "t/L2/o/s1", "acc": 0, "truncated": 0, "refused": 0, "partial_score": 0.6},
            {"id": "t/L2/o/s1", "acc": 0, "truncated": 0, "refused": 0, "partial_score": 0.1},
            {"id": "t/L2/o/s1", "acc": 0, "truncated": 0, "refused": 1, "partial_score": 0.9}]
    c = tally(recs, {"t|L2": 0.2})["t|L2"]
    assert (c["n"], c["success"], c["n_wrong"]) == (4, 1, 3)
    assert c["near_sum"] == pytest.approx((0.6 - 0.2) / 0.8)
