# Vendored from ArgGYM rl/reward.py (see arggym_verl/vendor/__init__.py; MIT, ./LICENSE).
# Kept: `split_reasoning`, `score`, `reward_of`. Removed: the TRL reward function,
# its batch metrics and the distributed gather.
"""The reward: ArgGYM's strict success, scored by the code that scores every eval run.

A completion is split into reasoning and answer where vLLM's `qwen3` reasoning
parser splits it, at `</think>`, and handed to `evals.score.score_one` -- the
function that scores every eval run -- so the answer region, the fallback into
the reasoning trace and the scorer are the harness's own. The reward is
`success` as 1.0 or 0.0. The partial-credit `score` is logged and never paid:
a policy that reads nothing already earns a large share of it.

A truncated completion, meaning the last token is neither EOS nor padding, pays
0 and stays in the loss (`mask_truncated_completions: false`, which `rl.train`
requires). Returning `None` for it instead takes it out of the baseline and the
loss, so the only behaviour the gradient punishes is committing to a wrong
answer, and running to the cap becomes the safe move: on GSM8K this raised
truncation from 2% to 38% in 30 steps (`rl.gsm8k`). Eval scores a truncated
completion on its text, so the two part where that text already holds a
correct answer: 1 of the 119 truncated completions in the base eval of
Qwen3.5-4B (`results/hf-qwen3.5-4b__xml_tags__cot`). Paying it would pay for
running on after answering. The logged success rate is eval's, truncated text
scored as it stands.

One case returns `None`, which TRL treats as unscorable: the completion leaves
the group baseline and gets advantage 0. That is a row the scorer refused
(`score_one` reports `scorer_refused`): a fault in the row or the install, which
eval also leaves out of the score, so no answer could earn it a 0 or a 1.

Everything the reward computes is also logged, once per batch, over the whole
batch: every rank gathers the batch before computing a metric, so a group split
across ranks is still one group.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from arggym_verl.vendor.evals_score import score_one

THINK_END = "</think>"


def split_reasoning(text: str) -> Dict[str, str]:
    """Reasoning and completion, as the eval server's reasoning parser returns them."""
    head, sep, tail = text.partition(THINK_END)
    if not sep:
        return {"reasoning": "", "completion": text}
    return {"reasoning": head.replace("<think>", "", 1).strip(), "completion": tail.strip()}


def score(text: str, row: Dict[str, Any], truncated: bool, template: str) -> Dict[str, Any]:
    """The eval harness's record for this completion."""
    gen = {"id": row["id"], "task": row["task"], "level": row["metadata"]["level"],
           "ordering": row["metadata"]["ordering"], "truncated": truncated,
           **split_reasoning(text)}
    return score_one(gen, row, template)


def reward_of(record: Dict[str, Any]) -> Optional[float]:
    """1.0 for a finished success, 0.0 for a failure or a truncation, None if unscorable."""
    if record["success"] is None:
        return None
    return 1.0 if record["success"] and not record["truncated"] else 0.0
