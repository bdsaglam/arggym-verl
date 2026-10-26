"""The reward, as a verl reward manager: ArgGYM's scorers on the exact sampled token ids.

Configured with

    reward.reward_manager.source=importlib
    reward.reward_manager.name=ArgGymRewardManager
    reward.reward_manager.module.path=arggym_verl/reward.py

verl's default manager hands a reward function the decoded text only. The TRL
pipeline also needs the token ids, to call a completion truncated when its last
token is neither EOS nor padding, so this manager reads the ids itself and
decodes them as TRL does (`skip_special_tokens=True`).

GSM8K pays 1 when the last `\\boxed{}` is the gold number (`rl_gsm8k.correct`),
0 otherwise, and 0 when truncated; the completion stays in the loss.

ArgGYM scores through `rl_reward.score`, which is `evals_score.score_one`, the
scorer of ArgGYM's eval harness. What a training completion is paid is `reward.mode`
(`RewardShape`):

- `binary` (the default): `rl_reward.reward_of`, 1 for a finished success, 0 for a
  failure or a truncated completion, which stays in the loss.
- `partial`: on the tasks in `reward.partial_tasks`, a finished failure is paid
  w * max(0, (s - b) / (1 - b)), s the scorer's partial score, b the score of an
  answer that reads nothing on the row's (task, level) (`reward.partial_baseline`,
  written by `arggym_verl.floors`) and w `reward.partial_weight` < 1, so an exact
  answer always beats a near miss. Other tasks are paid as in `binary`.
- `native`: the scorer's own score on every task, 0 for a completion with no
  answer region.

In every mode a truncated completion pays 0, and so does a row the scorer refused
(`reward_of` returns None for it; verl has no way to leave a completion out of its
group's baseline). `refused` is logged so a refusal is never silent. Dev
completions are paid as in `binary` whatever the mode, and `acc`, which dev
selection, the curriculum and the step's success metrics read, is the scorer's
success in every mode.

`reward_extra_info` is what verl averages into the dev metrics
(`val-aux/<data_source>/<key>/mean@N`); `acc` is the selection metric
(`val-core/<data_source>/acc/mean@N`). With `ARGGYM_VERL_REWARD_LOG` set to a
directory, every scored completion is also appended there as one JSON line
(`rewards-<pid>.jsonl`): training truncation and groups without signal are not
in verl's step metrics, and `arggym_verl.report` computes them from these lines.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Mapping, Optional, Sequence, Tuple

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase


ROOT = Path(__file__).resolve().parents[1]
MODES = ("binary", "partial", "native")


def is_truncated(ids: Sequence[int], stop_ids: set) -> bool:
    """ArgGYM's rule (its TRL reward functions): the last token is neither EOS nor padding."""
    return bool(ids) and ids[-1] not in stop_ids


@dataclass(frozen=True)
class RewardShape:
    """What a training completion of an ArgGYM row is paid (module docstring)."""

    mode: str = "binary"
    partial_weight: float = 0.0
    partial_tasks: FrozenSet[str] = frozenset()
    #: (task, level) -> b, the floor of the cell; only the partial tasks' cells.
    floors: Mapping[Tuple[str, int], float] = field(default_factory=dict)

    @classmethod
    def from_config(cls, reward: Optional[Mapping[str, Any]]) -> "RewardShape":
        """From verl's `reward` block; no `mode` key is `binary`."""
        reward = reward or {}
        mode = reward.get("mode", "binary")
        if mode not in MODES:
            raise ValueError(f"reward.mode {mode!r}: one of {', '.join(MODES)}")
        if mode != "partial":
            return cls(mode)
        w = float(reward.get("partial_weight") or 0)
        if not 0 < w < 1:
            raise ValueError(f"reward.partial_weight {w}: must be in (0, 1), so an exact answer "
                             f"is paid more than any near miss")
        tasks = frozenset(reward.get("partial_tasks") or ())
        if not tasks:
            raise ValueError("reward.mode partial with no reward.partial_tasks")
        path = Path(reward.get("partial_baseline") or "")
        cells = json.loads((path if path.is_absolute() else ROOT / path).read_text())["by_task_level"]
        floors = {}
        for key, cell in cells.items():
            task, level = key.split("|L")
            if task in tasks:
                floors[(task, int(level))] = float(cell["floor"])
        missing = sorted(tasks - {t for t, _ in floors})
        if missing:
            raise ValueError(f"reward.partial_baseline {path} has no floor for {missing}")
        over = sorted(k for k, b in floors.items() if b >= 1)
        if over:
            raise ValueError(f"reward.partial_baseline: floor 1 or above on {over}")
        return cls(mode, w, tasks, floors)

    def pay(self, record: Dict[str, Any], task: str, level: int) -> float:
        """The reward of one scored training completion (`rl_reward.score`'s record)."""
        from arggym_verl.vendor.rl_reward import reward_of

        binary = reward_of(record)
        if binary is None or record["truncated"]:
            return 0.0
        if self.mode == "native":
            return 0.0 if record["no_answer_region"] else float(record["score"])
        if self.mode == "partial" and task in self.partial_tasks and not binary:
            b = self.floors[(task, level)]
            return self.partial_weight * max(0.0, (float(record["score"]) - b) / (1 - b))
        return binary


BINARY = RewardShape()


def score_completion(data_source: str, text: str, ids: Sequence[int], extra_info: Dict[str, Any],
                     stop_ids: set, shape: RewardShape = BINARY) -> Dict[str, float]:
    """The reward and what is logged beside it; every value is a float, as verl averages them."""
    truncated = is_truncated(ids, stop_ids)
    if data_source == "gsm8k":
        from arggym_verl.vendor import rl_gsm8k as gsm8k

        success = gsm8k.correct(text, float(extra_info["gold"]))
        return {"score": 0.0 if truncated else float(success),
                # ArgGYM's rl.gsm8k.batch_metrics: a truncated completion counts as a failure.
                "acc": float(success and not truncated),
                "truncated": float(truncated), "length": float(len(ids)),
                "boxed": float(gsm8k.last_boxed(text) is not None)}
    if data_source == "arggym":
        from arggym_verl.vendor import rl_reward as arggym_reward

        row = json.loads(extra_info["row_json"])
        rec = arggym_reward.score(text, row, truncated, extra_info["template"])
        refused = rec["success"] is None
        if extra_info.get("split") == "train" and shape.mode != "binary":
            paid = shape.pay(rec, row["task"], int(row["metadata"]["level"]))
        else:
            paid = arggym_reward.reward_of(rec)
        return {"score": 0.0 if paid is None else paid,
                # ArgGYM's rl.reward.batch_metrics: success over the rows the scorer accepted,
                # a truncated completion scored as it reads, as in eval.
                "acc": float(bool(rec["success"])), "refused": float(refused),
                "partial_score": float(rec["score"] or 0.0), "truncated": float(truncated),
                "length": float(len(ids)),
                "no_answer_region": float(bool(rec.get("no_answer_region")))}
    raise ValueError(f"unknown data_source {data_source!r}")


def passed(reward_info: Mapping[str, Any]) -> Optional[bool]:
    """The curriculum's pass (`group_outcomes`): a success that did not truncate; None if refused."""
    if reward_info.get("refused"):
        return None
    return bool(reward_info["acc"]) and not reward_info["truncated"]


def step_metrics(infos: Sequence[Mapping[str, Any]],
                 rewards: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    """A training step's paid reward beside its exact success, overall and per task.

    `infos` are the completions' `extra_info`, `rewards` their `reward_extra_info`.
    A group is a row's completions. With partial credit a group whose completions
    all fail can still differ in reward, so the share of groups with no reward
    variance (all advantages 0) falls below the share with no success variance.
    """
    if not infos:
        return {}
    groups: Dict[str, Tuple[set, set]] = defaultdict(lambda: (set(), set()))
    by_task: Dict[str, Tuple[list, list]] = defaultdict(lambda: ([], []))
    paid_all, passed_all = [], []
    for info, rew in zip(infos, rewards, strict=True):
        paid, ok = float(rew["score"]), passed(rew)
        scores, outcomes = groups[info["id"]]
        scores.add(paid)
        paid_all.append(paid)
        by_task[info.get("task", "")][0].append(paid)
        if ok is not None:
            outcomes.add(ok)
            passed_all.append(ok)
            by_task[info.get("task", "")][1].append(ok)
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    out = {"reward/paid_mean": mean(paid_all),
           "reward/no_reward_variance_share": mean([len(g[0]) < 2 for g in groups.values()]),
           "reward/no_success_variance_share": mean([len(g[1]) < 2 for g in groups.values()])}
    if passed_all:
        out["reward/success_mean"] = mean(passed_all)
    for task, (paid, ok) in sorted(by_task.items()):
        if task:
            out[f"reward/paid_mean/{task}"] = mean(paid)
            if ok:
                out[f"reward/success_mean/{task}"] = mean(ok)
    return out


class ArgGymRewardManager(RewardManagerBase):
    def __init__(self, config, tokenizer, compute_score, reward_router_address=None,
                 reward_model_tokenizer=None, **_: Any):
        super().__init__(config, tokenizer, compute_score)
        self.stop_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id} - {None}
        self.shape = RewardShape.from_config(config.get("reward") if config else None)
        self.log_dir: Optional[str] = os.environ.get("ARGGYM_VERL_REWARD_LOG")

    async def run_single(self, data: DataProto) -> dict:
        item = data[-1]
        response = item.batch["responses"]
        n_valid = int(item.batch["attention_mask"][-response.shape[-1]:].sum())
        ids = response[:n_valid].tolist()
        text = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(ids, skip_special_tokens=True))
        extra = dict(item.non_tensor_batch.get("extra_info") or {})
        info = await self.loop.run_in_executor(
            None, lambda: score_completion(item.non_tensor_batch["data_source"], text, ids, extra,
                                           self.stop_ids, self.shape))
        if self.log_dir:
            self._append(item, extra, info)
        return {"reward_score": info["score"], "reward_extra_info": info}

    def _append(self, item, extra: Dict[str, Any], info: Dict[str, float]) -> None:
        fields = item.non_tensor_batch.get("tool_extra_fields") or {}
        record = {"split": extra.get("split"), "id": extra.get("id"),
                  "uid": str(item.non_tensor_batch.get("uid", "")),
                  "version": fields.get("global_steps"), **info}
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, f"rewards-{os.getpid()}.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")
