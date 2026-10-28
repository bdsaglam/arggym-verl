"""ArgGYM's pass-rate curriculum as verl's training sampler.

The sampler is ArgGYM's own class, `Curriculum` (vendored as
`arggym_verl.vendor.rl_curriculum`): each step's prompts are drawn from the (task, level)
cells in proportion to a weight of each cell's pass rate p (below). p starts
from the base model's eval through a Beta(1, 1) prior and is updated from every
training group with a half-life counted in groups of the cell; rows within a cell
come without replacement. `data.curriculum` configures it as the `curriculum:` block of a TRL
config does (prior, half_life, floor, table_every), plus `margin`; the seed is
`data.seed` and the prompts per step `data.train_batch_size`.

The weight is max(0, p(1 - p) - margin) + floor. ArgGYM's is the case margin = 0,
p(1 - p) + floor. A margin of 0.09 leaves cells with p <= 0.1 or p >= 0.9 the
floor alone: they are drawn rarely, but still now and then, so their estimates
can move back. With `power` k set the weight is normalized so the exponent reads,
(max(0, p(1 - p) - margin) / (0.25 - margin))^k + floor: 1 + floor at p = 0.5. At
k = 1 it is the unnormalized weight times 1 / (0.25 - margin), floor scaled alike
(0.005 / 0.16 = 0.03125), so the draw is the same.

With `task_balanced` the draw has two levels: a task gets a share in proportion to
its best cell's weight, max_c w_c, and its cells share that in proportion to w_c.
A task is worth its best cell, so the number of its cells in band does not
multiply its share, and a task whose cells all sit at the floor gets a share in
proportion to the floor. Without it, cells are drawn in proportion to w_c.

What a group adds to its cell is its passes on a binary task. On a task the reward
pays partial credit on (`reward.mode: partial`, `reward.partial_tasks`), it adds
the group's paid reward, the sum of its completions' rewards, so the estimate is
the paid rate q the reward pays on average and the prior starts from
(`arggym_verl.prior`); `curriculum/pass_rate/<cell>` is that rate on those tasks.

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
  between steps, after the update and before the next draw.

What counts as a pass is `group_outcomes`'s rule on the reward's
logged fields: a success that did not truncate; a completion the scorer refused
is left out. On a paid task the reward's `score` counts instead; it pays 0 for a
truncation.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence

from arggym_verl.vendor.rl_curriculum import Cell, Curriculum, base_pass_counts, summary


class StepCurriculum(Curriculum):
    """TRL's `Curriculum`, with a margin and an exponent on the weight, an optional
    task-balanced draw, the paid rate on paid tasks, and refusing to draw before the
    previous block's outcomes are in.

    verl asks for more prompts mid-step when it evicts a failed group and refills
    it (`ReplayBufferAsync.wait_for_sampleable`); that request would draw the next
    block from estimates that miss the current step.
    """

    def __init__(self, *args, margin: float = 0.0, power: Optional[float] = None,
                 task_balanced: bool = False, paid_tasks: Iterable[str] = (), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.margin = margin
        self.power = power
        self.task_balanced = task_balanced
        #: Tasks whose groups add their paid reward rather than their passes.
        self.paid_tasks = frozenset(paid_tasks)

    def cell_weights(self) -> Dict[Cell, float]:
        """w_c per cell (module docstring)."""
        out = {}
        for c in self.cells:
            p = self.pass_rate(c)
            signal = max(0.0, p * (1 - p) - self.margin)
            if self.power is not None:
                signal = (signal / (0.25 - self.margin)) ** self.power
            out[c] = signal + self.floor
        return out

    def weights(self) -> Dict[Cell, float]:
        """What the draw is proportional to (the vendored `draw` and `table` read it): w_c,
        or with `task_balanced` each cell's two-level share."""
        w = self.cell_weights()
        if not self.task_balanced:
            return w
        by_task: Dict[str, Dict[Cell, float]] = {}
        for c, x in w.items():
            by_task.setdefault(c[0], {})[c] = x
        best = {t: max(ws.values()) for t, ws in by_task.items()}
        total = sum(best.values())
        return {c: best[t] / total * x / sum(ws.values())
                for t, ws in by_task.items() for c, x in ws.items()}

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
    """Dataset positions per (task, level), as ArgGYM's `rl.train.build_curriculum` builds them."""
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
    c = data.get("curriculum") or {}
    if c.get("task_balanced", False) and not c.get("floor", 0) > 0:
        # A task whose cells all sit at or below the margin would weigh 0 / 0.
        problems.append("task_balanced needs floor > 0")
    if problems:
        raise ValueError("data.curriculum: " + "; ".join(problems))


def paid_tasks(reward_config) -> List[str]:
    """The tasks whose groups the curriculum counts at their paid reward: `reward.partial_tasks`
    under `reward.mode: partial`, none otherwise."""
    reward = reward_config or {}
    if reward.get("mode", "binary") != "partial":
        return []
    return list(reward.get("partial_tasks") or [])


def build(data_config, dataset, paid: Iterable[str] = ()) -> StepCurriculum:
    from arggym_verl.vendor.rl_pool import ROOT

    c = data_config.curriculum
    return StepCurriculum(cells_of(dataset), base_pass_counts(str(ROOT / c.prior)),
                          prompts_per_step=data_config.train_batch_size, num_generations=1,
                          steps_per_generation=1, seed=data_config.seed,
                          half_life=c.half_life, floor=c.floor, margin=c.get("margin", 0.0),
                          power=c.get("power", None), task_balanced=c.get("task_balanced", False),
                          paid_tasks=paid)


def step_outcomes(curriculum: Curriculum, infos: Sequence[Dict[str, Any]],
                  rewards: Sequence[Dict[str, Any]]):
    """(cell, successes, completions) per group of the step, in the order the block drew them.

    `infos` are the trajectories' `extra_info` (the dataset row's), `rewards` their
    `reward_extra_info` (arggym_verl.reward); one entry per completion. A completion
    counts 1 when it succeeded and did not truncate (`group_outcomes`'s rule), or, on a
    task in the curriculum's `paid_tasks`, its paid reward `score`. A refused one is
    left out, and a group with nothing left is dropped.
    """
    rows = curriculum.draws[curriculum.observed]["rows"]
    paid = getattr(curriculum, "paid_tasks", frozenset())
    first: Dict[int, int] = {}
    for pos, r in enumerate(rows):
        first.setdefault(r, pos)
    records = []
    for info, rew in zip(infos, rewards, strict=True):
        if info["index"] not in first:
            raise RuntimeError(f"curriculum: row {info['index']} was trained but not drawn in "
                               f"block {curriculum.observed}")
        if rew["refused"]:
            continue
        value = (float(rew["score"]) if info["task"] in paid
                 else int(bool(rew["acc"]) and not rew["truncated"]))
        records.append((first[info["index"]], info["id"], (info["task"], int(info["level"])), value))
    records.sort(key=lambda x: x[0])
    groups: Dict[str, list] = {}
    for _, gid, cell, value in records:
        g = groups.setdefault(gid, [cell, 0, 0])
        g[1] += value
        g[2] += 1
    return [tuple(g) for g in groups.values()]


def zero_signal_share(infos: Sequence[Dict[str, Any]], rewards: Sequence[Dict[str, Any]]) -> float:
    """The share of the step's groups whose completions were all paid the same reward:
    all passed or all failed, so every advantage in the group is 0."""
    groups: Dict[str, set] = {}
    for info, rew in zip(infos, rewards, strict=True):
        groups.setdefault(info["id"], set()).add(float(rew["score"]))
    return sum(len(scores) < 2 for scores in groups.values()) / len(groups)


def draw_entropy(curriculum: Curriculum) -> float:
    """Entropy in nats of the distribution the next block's cells are drawn from."""
    return -sum(r["draw_share"] * math.log(r["draw_share"]) for r in curriculum.table()
                if r["draw_share"] > 0)


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
    out["curriculum/draw_entropy"] = draw_entropy(curriculum)
    if step % table_every == 0 or step == 1:
        for r in rows:
            cell = f"{r['task']}/L{r['level']}"
            out[f"curriculum/pass_rate/{cell}"] = r["pass_rate"]
            out[f"curriculum/draw_share/{cell}"] = r["draw_share"]
    return out
