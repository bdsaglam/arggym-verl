"""ArgGYM's pass-rate curriculum as verl's training sampler.

The sampler is the TRL pipeline's own class, `rl.curriculum.Curriculum`, imported
from the pinned ArgGYM code: each step's prompts are drawn from the (task, level)
cells in proportion to p(1 - p) + floor, p being the cell's pass rate, started
from the base model's eval through a Beta(1, 1) prior and updated from every
training group with a half-life counted in groups of the cell; rows within a cell
come without replacement. `data.curriculum` configures it as the `curriculum:`
block of a TRL config does (prior, half_life, floor, table_every); the seed is
`data.seed` and the prompts per step `data.train_batch_size`.

Differences from TRL's use of the same class, all in how it is driven:

- The dataloader yields each prompt once (`num_generations` 1,
  `steps_per_generation` 1); verl's agent loop makes the `rollout.n` completions.
- No lag. verl fetches a step's prompts one at a time at the start of the step
  (`prepare_step`; `gen_batch_size` is 1 in the async modes) and the trainer
  feeds the step's outcomes back in `_compute_advantage`, so block k+1 is drawn
  after block k's outcomes are added. TRL's dataloader drew two blocks ahead.
  `dataloader_num_workers` must be 0: worker processes make the dataloader
  pull indices ahead of the step.
- Resuming: the sampler's `state_dict` goes into verl's `data.pt` with the
  dataloader's (torchdata saves a sampler that has one). Checkpoints fall
  between steps, after the update and before the next draw, so the current
  state is the one TRL's `state_for_step` would return.

What counts as a pass is `rl.curriculum.group_outcomes`'s rule on the reward's
logged fields: a success that did not truncate; a completion the scorer refused
is left out.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Sequence

from rl.curriculum import Cell, Curriculum, base_pass_counts, group_outcomes, summary


class StepCurriculum(Curriculum):
    """TRL's `Curriculum`, refusing to draw before the previous block's outcomes are in.

    verl asks for more prompts mid-step when it evicts a failed group and refills
    it (`ReplayBufferAsync.wait_for_sampleable`); that request would draw the next
    block from estimates that miss the current step.
    """

    def draw(self) -> List[int]:
        if self.observed != self.blocks:
            raise RuntimeError(
                f"curriculum: a prompt was requested after block {self.blocks - 1} was drawn and "
                f"before its outcomes were added (blocks drawn {self.blocks}, observed "
                f"{self.observed}); verl drew prompts beyond one batch a step")
        return super().draw()

    def update(self, outcomes) -> None:
        if self.observed != self.blocks - 1:
            raise RuntimeError(f"curriculum: outcomes without a drawn block (blocks drawn "
                               f"{self.blocks}, observed {self.observed})")
        super().update(outcomes)


def cells_of(dataset) -> Dict[Cell, List[int]]:
    """Dataset positions per (task, level), as `rl.train.build_curriculum` builds them."""
    cells: Dict[Cell, List[int]] = {}
    for i, info in enumerate(dataset.dataframe["extra_info"]):
        if info["index"] != i:
            raise RuntimeError(f"training row {i} has extra_info.index {info['index']}: the dataset "
                               f"dropped or reordered rows, so positions are not ArgGYM's indices")
        cells.setdefault((info["task"], int(info["level"])), []).append(i)
    return cells


def check_setup(config, trainer_mode: str, parameter_sync_step: int) -> None:
    """Refuse a configuration in which a step's draw would not see the previous step's outcomes."""
    data, v1 = config.data, config.trainer.v1
    problems = []
    if data.get("dataloader_num_workers", 0) != 0:
        problems.append("data.dataloader_num_workers must be 0 (workers prefetch indices)")
    if (v1.get(trainer_mode) or {}).get("num_warmup_batches", 0):
        problems.append(f"trainer.v1.{trainer_mode}.num_warmup_batches must be 0 (a batch ahead)")
    if trainer_mode not in ("sync", "colocate_async"):
        problems.append(f"trainer_mode {trainer_mode}: generation runs ahead of training")
    if parameter_sync_step != 1:
        problems.append("parameter_sync_step must be 1 (one update per batch)")
    fg = config.algorithm.get("filter_groups", None)
    if (fg and fg.get("enable", False)) or v1.sampler.get("sync_refill_failed_groups", False):
        problems.append("group filtering and refills draw prompts beyond one batch a step")
    if v1.sampler.get("custom_sampler", {}).get("path"):
        problems.append("a custom replay buffer may sample other than one step's groups")
    if problems:
        raise ValueError("data.curriculum: " + "; ".join(problems))


def build(data_config, dataset) -> StepCurriculum:
    from rl import pool

    c = data_config.curriculum
    return StepCurriculum(cells_of(dataset), base_pass_counts(str(pool.ROOT / c.prior)),
                          prompts_per_step=data_config.train_batch_size, num_generations=1,
                          steps_per_generation=1, seed=data_config.seed,
                          half_life=c.half_life, floor=c.floor)


def step_outcomes(curriculum: Curriculum, infos: Sequence[Dict[str, Any]],
                  rewards: Sequence[Dict[str, Any]]):
    """(cell, successes, completions) per group of the step, in the order the block drew them.

    `infos` are the trajectories' `extra_info` (the dataset row's), `rewards` their
    `reward_extra_info` (arggym_verl.reward); one entry per completion.
    """
    rows = curriculum.draws[curriculum.observed]["rows"]
    first: Dict[int, int] = {}
    for pos, r in enumerate(rows):
        first.setdefault(r, pos)
    records = []
    for info, rew in zip(infos, rewards, strict=True):
        if info["index"] not in first:
            raise RuntimeError(f"curriculum: row {info['index']} was trained but not drawn in "
                               f"block {curriculum.observed}")
        records.append((first[info["index"]],
                        {"id": info["id"], "task": info["task"], "level": int(info["level"]),
                         "success": None if rew["refused"] else bool(rew["acc"]),
                         "truncated": bool(rew["truncated"])}))
    records.sort(key=lambda x: x[0])
    return group_outcomes([r for _, r in records])


def metrics(curriculum: Curriculum, step: int, table_every: int) -> Dict[str, float]:
    """The step's draw and, every `table_every` steps and at step 1, every cell's estimate.

    The block is the one trained at `step`; the estimates and `draw_entropy` are
    after its outcomes, so they describe the distribution the next step draws from.
    """
    draw = curriculum.draws[curriculum.observed - 1]
    out = summary(draw)
    levels = Counter(c["level"] for c in draw["cells"])
    for lv in sorted({lv for _, lv in curriculum.cells}):
        out[f"curriculum/level_share/L{lv}"] = levels[lv] / len(draw["cells"])
    out["curriculum/groups"] = float(len(draw.get("outcomes", [])))
    rows = curriculum.table()
    out["curriculum/draw_entropy"] = -sum(r["draw_share"] * math.log(r["draw_share"]) for r in rows)
    if step % table_every == 0 or step == 1:
        for r in rows:
            cell = f"{r['task']}/L{r['level']}"
            out[f"curriculum/pass_rate/{cell}"] = r["pass_rate"]
            out[f"curriculum/draw_share/{cell}"] = r["draw_share"]
    return out
