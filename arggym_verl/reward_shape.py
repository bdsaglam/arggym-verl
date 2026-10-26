"""What a training completion of an ArgGYM row is paid: `reward.mode` (arggym_verl.reward).

Its own module because verl loads `reward.py` from its file path
(`reward.reward_manager.source=importlib`) without registering it in `sys.modules`,
and a dataclass defined there fails to build: `dataclasses` looks its module up.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
MODES = ("binary", "partial", "native")


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
