"""The curriculum draws and learns as ArgGYM's TRL pipeline's, one block a step, and resumes.

The TRL side is fixtures/curriculum_trace.json: ArgGYM's `rl.train.build_curriculum` on
003's config and training rows, set to one block of 16 prompts a step with each prompt
yielded once (verl's batch), fed 40 steps of `completions` below in its own order. The
verl side is `arggym_verl.curriculum` on the training parquet and the ArgGYM config at
003's weight (margin 0, floor 0.01), fed the same completions in a shuffled order, as
`_balance_batch` leaves them, in the reward's logged fields. The ArgGYM config's own
weight, with a margin, is tested on its own below.
"""
import json
import random
from types import SimpleNamespace

import pytest
import torch


def compose(*overrides):
    from hydra import compose, initialize_config_dir

    from conftest import ROOT

    with initialize_config_dir(config_dir=str(ROOT / "arggym_verl/config"), version_base=None):
        return compose(config_name="arggym-qwen3.5-4b",
                       overrides=["hydra.searchpath=[pkg://verl.trainer.config]", *overrides])


@pytest.fixture(scope="module")
def arggym():
    """The config at the trace's settings: 16 prompts a step and 003's weight, p(1 - p) + 0.01.
    The mechanics (draws, updates, resume) do not depend on either."""
    return compose("data.train_batch_size=16", "data.curriculum.margin=0.0",
                   "data.curriculum.floor=0.01")


@pytest.fixture(scope="module")
def configured():
    """The ArgGYM config as committed."""
    return compose()


@pytest.fixture(scope="module")
def parquet():
    import datasets

    from conftest import ROOT

    return datasets.load_dataset("parquet", split="train", data_files=str(
        ROOT / "data/arggym-qwen3.5-4b/train.parquet"))


@pytest.fixture(scope="module")
def trace():
    from conftest import fixture

    return fixture("curriculum_trace.json")


def cell_key(cell):
    return f"{cell[0]}|L{cell[1]}"


def ours(arggym, parquet):
    from arggym_verl import curriculum as cur

    return cur.build(arggym.data, SimpleNamespace(dataframe=parquet))


def completions(step, rows, extra_info, n=6):
    """A synthetic step: n completions a drawn row, success depending on the row so cells
    move apart; some truncated, a few refused. One dict per completion with verl's
    (extra_info, reward_extra_info). The fixture's TRL side was fed the same draws."""
    rng = random.Random(step)
    out = []
    for r in rows:
        info = extra_info[r]
        p = (r % 7) / 6
        for _ in range(n):
            refused = rng.random() < 0.03
            success = None if refused else rng.random() < p
            truncated = rng.random() < 0.1
            out.append({"info": info,
                        "reward": {"acc": float(bool(success)), "refused": float(refused),
                                   "truncated": float(truncated)}})
    return out


def feed(curriculum, batch, shuffle_seed):
    from arggym_verl import curriculum as cur

    batch = list(batch)
    random.Random(shuffle_seed).shuffle(batch)
    curriculum.update(cur.step_outcomes(curriculum, [c["info"] for c in batch],
                                        [c["reward"] for c in batch]))


def test_config_is_003s_curriculum_with_a_margin(arggym, configured):
    """003's prior, half-life, table interval and seed; the weight's margin 0.09 and floor 0.005
    are the config's own. At margin 0 and floor 0.01 it is 003's block exactly."""
    from conftest import fixture

    exp003 = fixture("exp003.json")
    want = {**exp003["curriculum"], "prior": "data/arggym-prior/hf-qwen3.5-4b__xml_tags__cot.json"}
    assert dict(arggym.data.curriculum) == {**want, "margin": 0.0}
    assert exp003["curriculum"]["floor"] == 0.01
    assert dict(configured.data.curriculum) == {**want, "margin": 0.09, "floor": 0.005}
    assert configured.data.seed == exp003["seed"] == 0
    assert configured.data.dataloader_num_workers == 0


def test_weight_is_p1mp_less_the_margin_plus_the_floor():
    """max(0, p(1 - p) - margin) + floor; margin 0 is ArgGYM's p(1 - p) + floor."""
    from arggym_verl.curriculum import StepCurriculum
    from arggym_verl.vendor.rl_curriculum import Curriculum

    # Beta(1, 1) pass rates 0.1, 0.5, 0.9, 0.95, 0.75 from (successes, scored).
    prior = {("a", 1): (0, 8), ("a", 2): (4, 8), ("a", 3): (8, 8), ("a", 4): (18, 18), ("a", 5): (5, 6)}
    cells = {c: [i] for i, c in enumerate(prior)}
    args = dict(prompts_per_step=4, num_generations=1, steps_per_generation=1, seed=0)
    sharp = StepCurriculum(cells, prior, floor=0.005, margin=0.09, **args)
    p = {c: sharp.pass_rate(c) for c in cells}
    assert [round(p[c], 3) for c in cells] == [0.1, 0.5, 0.9, 0.95, 0.75]
    w = sharp.weights()
    assert w[("a", 1)] == pytest.approx(0.005) and w[("a", 3)] == pytest.approx(0.005)
    assert w[("a", 4)] == 0.005
    assert w[("a", 2)] == pytest.approx(0.25 - 0.09 + 0.005)
    assert w[("a", 5)] == pytest.approx(0.1875 - 0.09 + 0.005)
    # The table (draw shares, entropy) reads the same weights.
    assert sum(r["draw_share"] for r in sharp.table()) == pytest.approx(1)
    plain = StepCurriculum(cells, prior, floor=0.01, **args)
    assert plain.weights() == Curriculum(cells, prior, floor=0.01, **args).weights()


def test_draw_entropy_at_step_0(configured, parquet):
    """The configured weight on the base model's eval sharpens the first draw only a little:
    4.003 nats against 4.031 for 003's weight and 4.043 for uniform over the 57 cells. Of
    the 57 training cells only one starts at p <= 0.1 or p >= 0.9 (0 or 8 of 8 passes); the
    rest sit at p 0.2-0.8, where p(1 - p) - 0.09 varies from 0.07 to 0.16."""
    import math

    from arggym_verl import curriculum as cur

    sharp = ours(configured, parquet)
    assert cur.draw_entropy(sharp) == pytest.approx(4.003, abs=1e-3)
    old = cur.build(compose("data.curriculum.margin=0.0", "data.curriculum.floor=0.01").data,
                    SimpleNamespace(dataframe=parquet))
    assert cur.draw_entropy(old) == pytest.approx(4.031, abs=1e-3)
    assert math.log(57) == pytest.approx(4.043, abs=1e-3)
    excluded = [c for c in sharp.cells if sharp.weights()[c] == pytest.approx(sharp.floor)]
    assert len(excluded) == 1


def test_zero_signal_share():
    """Groups (one row's completions) whose rewards are all equal: all passed or all failed."""
    from arggym_verl.curriculum import zero_signal_share

    infos = [{"id": r} for r in "aaaabbbbcccc"]
    scores = [1, 1, 1, 1] + [0, 0, 0, 0] + [0, 1, 0, 0]
    assert zero_signal_share(infos, [{"score": float(x)} for x in scores]) == pytest.approx(2 / 3)


def test_cells_and_prior_are_trls(arggym, parquet, trace):
    mine = ours(arggym, parquet)
    assert {cell_key(c): v for c, v in mine.cells.items()} == trace["initial"]["cells"]
    assert {cell_key(c): v for c, v in mine.counts.items()} == trace["initial"]["counts"]
    assert len(mine.cells) == 57 and sum(map(len, mine.cells.values())) == 5700
    # Every training cell has a base-eval prior (8 samples a cell).
    assert all(sum(v) == 8 for v in mine.counts.values())


def test_matches_trl_step_by_step(arggym, parquet, trace):
    """Same prompts drawn, same estimates after every step's update, for 40 steps."""
    mine = ours(arggym, parquet)
    extra_info = parquet["extra_info"]
    m_it = iter(mine)
    for step, want in enumerate(trace["steps"]):
        m_rows = [next(m_it) for _ in range(16)]
        assert m_rows == want["rows"], step
        assert mine.draws[step]["cells"] == want["cells"]
        feed(mine, completions(step, m_rows, extra_info), shuffle_seed=step)  # verl's order
        assert {cell_key(c): v for c, v in mine.counts.items()} == want["counts"], step
        assert mine.draws[step]["outcomes"] == want["outcomes"]
    assert json.loads(json.dumps(mine.state_dict())) == trace["final_state"]
    # The estimates moved away from the prior.
    assert len({round(mine.pass_rate(c), 3) for c in mine.cells}) > 40


class Rows(torch.utils.data.Dataset):
    """What verl's dataloader needs from RLHFDataset: one dict a position."""

    def __init__(self, parquet):
        self.dataframe = parquet
        self.extra_info = parquet["extra_info"]

    def __len__(self):
        return len(self.extra_info)

    def __getitem__(self, i):
        return {"raw_prompt": [], "extra_info": self.extra_info[i]}


def loader(dataset, sampler):
    from torchdata.stateful_dataloader import StatefulDataLoader
    from verl.utils.dataset.rl_dataset import collate_fn

    return StatefulDataLoader(dataset=dataset, batch_size=1, num_workers=0, drop_last=True,
                              collate_fn=collate_fn, sampler=sampler)


def run_steps(dl, it, curriculum, steps, extra_info):
    """verl's order: fetch a step's 16 prompts one at a time, then feed the outcomes."""
    seen = []
    for step in steps:
        rows = [int(next(it)["extra_info"][0]["index"]) for _ in range(16)]
        feed(curriculum, completions(step, rows, extra_info), shuffle_seed=step)
        seen.append((rows, {c: list(v) for c, v in curriculum.counts.items()}))
    return seen


def test_resume_restores_the_estimates(arggym, parquet, tmp_path):
    """verl's data.pt (the dataloader's state_dict) carries the curriculum; a resumed run
    draws and learns as the uninterrupted one."""
    data = Rows(parquet)
    first = ours(arggym, parquet)
    dl = loader(data, first)
    it = iter(dl)
    run_steps(dl, it, first, range(7), data.extra_info)
    torch.save(dl.state_dict(), tmp_path / "data.pt")  # trainer_base._save_checkpoint
    after = run_steps(dl, it, first, range(7, 12), data.extra_info)

    resumed = ours(arggym, parquet)
    assert resumed.counts != first.counts
    dl2 = loader(data, resumed)
    dl2.load_state_dict(torch.load(tmp_path / "data.pt", weights_only=False))  # _load_checkpoint
    assert run_steps(dl2, iter(dl2), resumed, range(7, 12), data.extra_info) == after
    assert resumed.state_dict() == first.state_dict()


def test_dataloader_draws_a_block_only_after_the_previous_outcomes(arggym, parquet):
    """No index is pulled ahead: 16 fetches draw one block; a 17th before the update is
    refused; after the update it draws the next block."""
    data = Rows(parquet)
    c = ours(arggym, parquet)
    it = iter(loader(data, c))
    assert c.blocks == 0  # creating the iterator draws nothing
    rows = [int(next(it)["extra_info"][0]["index"]) for _ in range(16)]
    assert (c.blocks, c.observed) == (1, 0)
    with pytest.raises(RuntimeError, match="before its outcomes were added"):
        next(it)
    c2 = ours(arggym, parquet)
    it2 = iter(loader(data, c2))
    assert [int(next(it2)["extra_info"][0]["index"]) for _ in range(16)] == rows
    feed(c2, completions(0, rows, data.extra_info), shuffle_seed=0)
    next(it2)
    assert (c2.blocks, c2.observed) == (2, 1)


def test_setup_refuses_draws_ahead(arggym):
    from omegaconf import OmegaConf

    from arggym_verl.curriculum import check_setup

    check_setup(arggym, "colocate_async", 1)
    for key, value in [("data.dataloader_num_workers", 8),
                       ("trainer.v1.colocate_async.num_warmup_batches", 1),
                       ("algorithm.filter_groups.enable", True)]:
        cfg = OmegaConf.create(OmegaConf.to_container(arggym))
        OmegaConf.update(cfg, key, value, force_add=True)
        with pytest.raises(ValueError, match="data.curriculum"):
            check_setup(cfg, "colocate_async", 1)
    with pytest.raises(ValueError, match="generation runs ahead"):
        check_setup(arggym, "separate_async", 1)
    with pytest.raises(ValueError, match="parameter_sync_step"):
        check_setup(arggym, "colocate_async", 2)


def test_metrics(arggym, parquet):
    import math

    from arggym_verl import curriculum as cur

    c = ours(arggym, parquet)
    it = iter(c)
    rows = [next(it) for _ in range(16)]
    feed(c, completions(0, rows, parquet["extra_info"]), shuffle_seed=0)
    m = cur.metrics(c, step=1, table_every=5)
    assert sum(m[f"curriculum/level_share/L{lv}"] for lv in range(1, 9)) == pytest.approx(1)
    assert m["curriculum/groups"] == 16
    assert 0 < m["curriculum/draw_entropy"] < math.log(57)
    assert sum(v for k, v in m.items() if k.startswith("curriculum/draw_share/")) == pytest.approx(1)
    assert len([k for k in m if k.startswith("curriculum/pass_rate/")]) == 57
    assert {"curriculum/mean_signal", "curriculum/mean_pass_rate",
            "curriculum/distinct_cells"} <= set(m)
    rows = [next(it) for _ in range(16)]
    feed(c, completions(1, rows, parquet["extra_info"]), shuffle_seed=1)
    assert not any(k.startswith("curriculum/pass_rate/") for k in cur.metrics(c, 2, 5))


# ---------------------------------------------------------------- v3: exponent, task balance, paid rate


def synthetic(**kwargs):
    """Three tasks over Beta(1, 1) pass rates: a has cells at p 0.5, 0.75 and 0.1; b one cell
    at 0.75; c two cells at the floor (0.1, 0.95)."""
    from arggym_verl.curriculum import StepCurriculum

    prior = {("a", 1): (4, 8), ("a", 2): (5, 6), ("a", 3): (0, 8), ("b", 1): (5, 6),
             ("c", 1): (0, 8), ("c", 2): (18, 18)}
    cells = {c: list(range(10 * i, 10 * i + 10)) for i, c in enumerate(prior)}
    args = dict(prompts_per_step=18, num_generations=1, steps_per_generation=1, seed=0)
    return StepCurriculum(cells, prior, **{**args, **kwargs})


def shares(c):
    return {(r["task"], r["level"]): r["draw_share"] for r in c.table()}


def test_normalized_weight_at_k1_draws_as_the_current_weight():
    """(max(0, p(1 - p) - m) / (0.25 - m))^1 + f / (0.25 - m) is the current weight times
    1 / (0.25 - m): the same draw shares and the same draws. The current floor 0.005 is
    0.03125 normalized."""
    current = synthetic(margin=0.09, floor=0.005)
    normalized = synthetic(margin=0.09, floor=0.005 / 0.16, power=1)
    w0, w1 = current.cell_weights(), normalized.cell_weights()
    assert all(w1[c] == pytest.approx(w0[c] / 0.16) for c in w0)
    assert shares(normalized) == pytest.approx(shares(current))
    assert normalized.draw() == current.draw()
    assert w1[("a", 1)] == pytest.approx(1 + 0.03125)


def test_k2_squares_the_normalized_signal():
    """k = 2: p 0.5 weighs 1 + floor, p 0.75 ((0.1875 - 0.09) / 0.16)^2 + floor = 0.372 +
    floor, p 0.1 and 0.95 the floor alone; the ratio of the two in-band cells grows from
    1.6 (k = 1) to 2.6."""
    w = synthetic(margin=0.09, floor=0.03, power=2).cell_weights()
    assert w[("a", 1)] == pytest.approx(1.03)
    assert w[("a", 2)] == pytest.approx(((0.1875 - 0.09) / 0.16) ** 2 + 0.03)
    assert w[("a", 3)] == w[("c", 1)] == w[("c", 2)] == pytest.approx(0.03)
    assert w[("a", 1)] > w[("a", 2)] > w[("a", 3)]
    k1 = synthetic(margin=0.09, floor=0.03, power=1).cell_weights()
    assert w[("a", 1)] / w[("a", 2)] > k1[("a", 1)] / k1[("a", 2)]


def test_task_balanced_draw_gives_each_task_its_best_cells_share():
    """A task gets max_c w_c / sum over tasks of max_c w_c of the draws, whatever its number of
    cells, and its cells share that in proportion to w_c; a task of floor cells gets the
    floor's share. Checked on the table and on 10^4 draws (+-0.01)."""
    from collections import Counter

    c = synthetic(margin=0.09, floor=0.03, power=2, task_balanced=True, prompts_per_step=10_000)
    w = c.cell_weights()
    best = {"a": w[("a", 1)], "b": w[("b", 1)], "c": 0.03}
    total = sum(best.values())
    s = shares(c)
    for t in best:
        assert sum(v for (task, _), v in s.items() if task == t) == pytest.approx(best[t] / total)
    a_total = w[("a", 1)] + w[("a", 2)] + w[("a", 3)]
    assert s[("a", 2)] == pytest.approx(best["a"] / total * w[("a", 2)] / a_total)
    c.draw()
    drawn = Counter(x["task"] for x in c.draws[0]["cells"])
    for t in best:
        assert drawn[t] / 10_000 == pytest.approx(best[t] / total, abs=0.01)
    # Cell-proportional, task a's three cells would outweigh b's one.
    flat = shares(synthetic(margin=0.09, floor=0.03, power=2))
    assert sum(v for (t, _), v in flat.items() if t == "a") > sum(
        v for (t, _), v in s.items() if t == "a")


def test_paid_tasks_add_the_paid_reward_and_binary_tasks_their_passes():
    """On a paid task a group adds the sum of its rewards (`score`: 1 for a pass, partial
    credit for a near miss, 0 for a truncation); elsewhere its passes. Refusals are left out."""
    from arggym_verl.curriculum import step_outcomes

    cur = SimpleNamespace(draws=[{"rows": [0, 1]}], observed=0,
                          paid_tasks=frozenset({"status_query"}))
    info = {0: {"index": 0, "id": "r0", "task": "status_query", "level": 3},
            1: {"index": 1, "id": "r1", "task": "attack", "level": 2}}
    comps = [(1, 1.0, 1.0, 0.0, 0.0), (0, 1.0, 1.0, 0.0, 0.0), (0, 0.25, 0.0, 0.0, 0.0),
             (1, 0.0, 0.0, 0.0, 0.0), (0, 0.0, 1.0, 1.0, 0.0), (0, 0.1, 0.0, 0.0, 0.0),
             (1, 0.0, 1.0, 1.0, 0.0), (0, 0.0, 0.0, 0.0, 1.0), (1, 1.0, 1.0, 0.0, 0.0)]
    infos = [info[r] for r, *_ in comps]
    rewards = [{"score": s, "acc": a, "truncated": t, "refused": x} for _, s, a, t, x in comps]
    (c0, s0, n0), (c1, s1, n1) = step_outcomes(cur, infos, rewards)
    assert (c0, n0) == (("status_query", 3), 4) and s0 == pytest.approx(1.35)
    assert (c1, s1, n1) == (("attack", 2), 2, 4)
    # Without paid tasks every group counts passes, as before.
    cur.paid_tasks = frozenset()
    assert step_outcomes(cur, infos, rewards)[0] == (("status_query", 3), 1, 4)


def test_paid_tasks_come_from_the_reward_block():
    from arggym_verl.curriculum import paid_tasks

    assert paid_tasks({"mode": "binary", "partial_tasks": ["status_query"]}) == []
    assert paid_tasks({"mode": "native", "partial_tasks": ["status_query"]}) == []
    assert paid_tasks(None) == []
    assert paid_tasks({"mode": "partial", "partial_tasks": ["status_query", "perturbation"]}) == [
        "status_query", "perturbation"]


def test_resume_round_trips_fractional_paid_counts():
    """A paid task's counts are fractional; state_dict -> JSON -> load_state_dict restores them,
    and the resumed curriculum draws as the uninterrupted one."""
    from arggym_verl.curriculum import step_outcomes

    kw = dict(margin=0.09, floor=0.03, power=2, task_balanced=True, paid_tasks={"a"})
    first = synthetic(**kw)
    for step in range(3):
        rows = first.draw()
        infos, rewards = [], []
        for r in rows:
            task, level = next(c for c, rs in first.cells.items() if r in rs)
            for k in range(4):
                infos.append({"index": r, "id": f"{r}", "task": task, "level": level})
                rewards.append({"score": [1.0, 0.2, 0.05, 0.0][k], "acc": float(k == 0),
                                "truncated": 0.0, "refused": 0.0})
        first.update(step_outcomes(first, infos, rewards))
    assert any(s != int(s) for s, _ in first.counts.values())
    state = json.loads(json.dumps(first.state_dict()))
    resumed = synthetic(**kw)
    resumed.load_state_dict(state)
    assert resumed.counts == first.counts and resumed.weights() == first.weights()
    assert resumed.draw() == first.draw()
