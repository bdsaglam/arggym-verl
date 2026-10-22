# Vendored from ArgGYM rl/gsm8k.py (see arggym_verl/vendor/__init__.py; MIT, ./LICENSE).
# Kept: the rows, the screen filter, the prompt and the `\boxed{}` check. Removed: the TRL
# reward function and its batch metrics.
"""GSM8K as a positive control: the same trainer on a task GRPO is known to move.

`task: gsm8k` in a config swaps the data, the prompt, the reward and the
metrics for these; the trainer, vLLM, the weight sync, the loss and dev
evaluation are the ArgGYM run's own. Training rows are HF `openai/gsm8k`
`train`; dev is a seeded subset of `test`, with any question `train` also holds
dropped.

The reward is 1 when the last `\\boxed{...}` in the completion is the gold
number after `####`, compared as numbers once commas and dollar signs are
stripped, and 0 otherwise. A truncated completion pays 0, as the eval scores
it, and stays in the loss (`mask_truncated_completions: false`), as in the
ArgGYM reward. Returning `None` instead takes it out of the baseline and the
loss, so the only behaviour the gradient punishes is committing to a wrong
answer: at LR 1e-4 on Qwen3-0.6B, truncation rose from 2% to 38% in 30 steps
and dev success fell from 0.805 to 0.735.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)")


def last_boxed(text: str) -> Optional[str]:
    """The content of the last `\\boxed{...}`, nested braces included; None if there is none."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i = start + len("\\boxed{")
    depth = 1
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i:j]
    return None


def to_number(text: Optional[str]) -> Optional[float]:
    """`1,234`, `$18`, `\\$-3.50` as a float; None for anything that is not one number."""
    if text is None:
        return None
    s = re.sub(r"\\\$|\$|,|\s", "", text)
    return float(s) if _NUMBER.fullmatch(s) else None


def gold(answer: str) -> float:
    """The number after `####` in a GSM8K answer."""
    value = to_number(answer.split("####")[-1])
    if value is None:
        raise ValueError(f"no number after ####: {answer[-80:]!r}")
    return value


def correct(text: str, gold_value: float) -> bool:
    value = to_number(last_boxed(text))
    return value is not None and abs(value - gold_value) < 1e-6


def load_splits(data: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Training rows from `train`, `dev_size` rows of `test` drawn with `dev_seed`.

    With `screen`, both keep only the questions whose base-model passes
    (`rl.gsm8k_screen`) lie in `screen.passes`, inclusive.
    """
    from datasets import load_dataset

    ds = load_dataset(data["dataset"], data["config"], revision=data["revision"])
    keep = screened(data["screen"]) if data.get("screen") else None
    train = [{"id": f"train-{i}", **r} for i, r in enumerate(ds["train"])]
    seen = {r["question"].strip() for r in train}
    test = [{"id": f"test-{i}", **r} for i, r in enumerate(ds["test"])
            if r["question"].strip() not in seen]
    if keep is not None:
        train = [r for r in train if r["id"] in keep]
        test = [r for r in test if r["id"] in keep]
    dev = random.Random(data["dev_seed"]).sample(test, data["dev_size"])
    return train, dev


def screened(screen: Dict[str, Any]) -> set[str]:
    """The ids whose passes lie in `screen.passes`; every question must have been screened."""
    lo, hi = screen["passes"]
    counts = {}
    for shard in sorted(Path(screen["path"]).glob("shard-*.jsonl")):
        for line in shard.read_text().splitlines():
            r = json.loads(line)
            if r["samples"] != screen["samples"]:
                raise ValueError(f"{shard}: {r['id']} has {r['samples']} samples, "
                                 f"not {screen['samples']}")
            counts[r["id"]] = r["passes"]
    if len(counts) != 7473 + 1319:
        raise ValueError(f"{screen['path']} screens {len(counts)} questions of 8,792; "
                         f"a shard is missing")
    return {i for i, p in counts.items() if lo <= p <= hi}


def trl_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"prompt": [{"role": "user", "content": f"{r['question']}\n{INSTRUCTION}"}],
             "gold": gold(r["answer"]), "id": r["id"]} for r in rows]
