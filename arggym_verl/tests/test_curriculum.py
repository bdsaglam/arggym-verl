"""The curriculum draws and learns as the TRL pipeline's, one block a step, and resumes.

The TRL side is `rl.train.build_curriculum` on 003's config and training rows, as
`rl.train` calls it, set to one block of 16 prompts a step with each prompt
yielded once (verl's batch). The verl side is `arggym_verl.curriculum` on the
training parquet and the ArgGYM config, fed the step's completions in a shuffled
order, as `_balance_batch` leaves them, in the reward's logged fields.
"""
import random
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def arggym():
    from hydra import compose, initialize_config_dir

    from conftest import ROOT

    with initialize_config_dir(config_dir=str(ROOT / "arggym_verl/config"), version_base=None):
        return compose(config_name="arggym-qwen3.5-4b",
                       overrides=["hydra.searchpath=[pkg://verl.trainer.config]"])


@pytest.fixture(scope="module")
def parquet():
    import datasets

    from conftest import ROOT

    return datasets.load_dataset("parquet", split="train", data_files=str(
        ROOT / "data/arggym-qwen3.5-4b/train.parquet"))


@pytest.fixture(scope="module")
def trl_setup():
    from conftest import ARGGYM_POOL_ROOT, TRL_ARGGYM
    from rl import config as rl_config
    from rl import pool

    cfg = rl_config.load(str(TRL_ARGGYM), [f"output_root={ARGGYM_POOL_ROOT}"])
    rows = pool.trl_rows(pool.restrict(pool.load_split(cfg, "train"), cfg.get("train_cells")),
                         cfg["prompt"]["template"], pool.elicitation(cfg["prompt"]["elicitation"]))
    return cfg, rows


def trl_curriculum(trl_setup, prompts_per_step):
    from rl.train import build_curriculum

    cfg, rows = trl_setup
    cfg = {**cfg, "prompts_per_step": prompts_per_step,
           "grpo": {**cfg["grpo"], "num_generations": 1, "gradient_accumulation_steps": 1}}
    return build_curriculum(cfg, rows)


def ours(arggym, parquet):
    from arggym_verl import curriculum as cur

    return cur.build(arggym.data, SimpleNamespace(dataframe=parquet))


def completions(step, rows, extra_info, n=6):
    """A synthetic step: n completions a drawn row, success depending on the row so cells
    move apart; some truncated, a few refused. One dict per completion, both pipelines'
    fields: TRL's reward record and verl's (extra_info, reward_extra_info)."""
    rng = random.Random(step)
    out = []
    for r in rows:
        info = extra_info[r]
        p = (r % 7) / 6
        for _ in range(n):
            refused = rng.random() < 0.03
            success = None if refused else rng.random() < p
            truncated = rng.random() < 0.1
            out.append({"trl": {"id": info["id"], "task": info["task"], "level": info["level"],
                                "success": success, "truncated": truncated},
                        "info": info,
                        "reward": {"acc": float(bool(success)), "refused": float(refused),
                                   "truncated": float(truncated)}})
    return out


def feed(curriculum, batch, shuffle_seed):
    from arggym_verl import curriculum as cur

    batch = list(batch)
    random.Random(shuffle_seed).shuffle(batch)
    curriculum.update(cur.step_outcomes(curriculum, [c["info"] for c in batch],
                                        [c["reward"] for c in batch]))


def test_config_is_003s_curriculum(arggym, trl_setup):
    cfg, _ = trl_setup
    assert dict(arggym.data.curriculum) == cfg["curriculum"]
    assert arggym.data.seed == cfg["seed"] == 0
    assert arggym.data.dataloader_num_workers == 0


def test_cells_and_prior_are_trls(arggym, parquet, trl_setup):
    trl, mine = trl_curriculum(trl_setup, 16), ours(arggym, parquet)
    assert mine.cells == trl.cells and len(mine.cells) == 57
    assert sum(map(len, mine.cells.values())) == 5700
    assert mine.counts == trl.counts
    # Every training cell has a base-eval prior (8 samples a cell).
    assert all(sum(v) == 8 for v in mine.counts.values())


def test_matches_trl_step_by_step(arggym, parquet, trl_setup):
    """Same prompts drawn, same estimates after every step's update, for 40 steps."""
    from rl.curriculum import group_outcomes

    trl, mine = trl_curriculum(trl_setup, 16), ours(arggym, parquet)
    extra_info = parquet["extra_info"]
    t_it, m_it = iter(trl), iter(mine)
    for step in range(40):
        t_rows = [next(t_it) for _ in range(16)]
        m_rows = [next(m_it) for _ in range(16)]
        assert m_rows == t_rows, step
        assert mine.draws[step]["cells"] == trl.draws[step]["cells"]
        batch = completions(step, m_rows, extra_info)
        trl.update(group_outcomes([c["trl"] for c in batch]))  # TRL's batch order
        feed(mine, batch, shuffle_seed=step)  # verl's balanced order
        assert mine.counts == trl.counts, step
        assert mine.draws[step]["outcomes"] == trl.draws[step]["outcomes"]
    assert mine.state_dict() == trl.state_dict()
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
