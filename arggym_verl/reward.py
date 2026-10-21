"""The reward, as a verl reward manager: ArgGYM's scorers on the exact sampled token ids.

Configured with

    reward.reward_manager.source=importlib
    reward.reward_manager.name=ArgGymRewardManager
    reward.reward_manager.module.path=arggym_verl/reward.py

verl's default manager hands a reward function the decoded text only. The TRL
pipeline also needs the token ids, to call a completion truncated when its last
token is neither EOS nor padding, so this manager reads the ids itself and
decodes them as TRL does (`skip_special_tokens=True`).

GSM8K pays 1 when the last `\\boxed{}` is the gold number (`rl.gsm8k.correct`),
0 otherwise, and 0 when truncated; the completion stays in the loss.

ArgGYM scores through `rl.reward.score`, which is `evals.score.score_one`, the
eval's scorer, and pays `rl.reward.reward_of`: 1 for a finished success, 0 for a
failure or a truncated completion, which stays in the loss. Where `reward_of`
returns None (the scorer refused the row) verl has no way to leave a completion
out of its group's baseline, so it pays 0; `refused` is logged so a refusal is
never silent.

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
from typing import Any, Dict, Optional, Sequence

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase


def is_truncated(ids: Sequence[int], stop_ids: set) -> bool:
    """TRL's rule (`rl/reward.py`, `rl/gsm8k.py`): the last token is neither EOS nor padding."""
    return bool(ids) and ids[-1] not in stop_ids


def score_completion(data_source: str, text: str, ids: Sequence[int], extra_info: Dict[str, Any],
                     stop_ids: set) -> Dict[str, float]:
    """The reward and what is logged beside it; every value is a float, as verl averages them."""
    truncated = is_truncated(ids, stop_ids)
    if data_source == "gsm8k":
        from rl import gsm8k

        success = gsm8k.correct(text, float(extra_info["gold"]))
        return {"score": 0.0 if truncated else float(success),
                # rl.gsm8k.batch_metrics: a truncated completion counts as a failure.
                "acc": float(success and not truncated),
                "truncated": float(truncated), "length": float(len(ids)),
                "boxed": float(gsm8k.last_boxed(text) is not None)}
    if data_source == "arggym":
        from rl import reward as arggym_reward

        row = json.loads(extra_info["row_json"])
        rec = arggym_reward.score(text, row, truncated, extra_info["template"])
        paid = arggym_reward.reward_of(rec)
        refused = rec["success"] is None
        return {"score": 0.0 if paid is None else paid,
                # rl.reward.batch_metrics: success over the rows the scorer accepted,
                # a truncated completion scored as it reads, as in eval.
                "acc": float(bool(rec["success"])), "refused": float(refused),
                "partial_score": float(rec["score"] or 0.0), "truncated": float(truncated),
                "length": float(len(ids)),
                "no_answer_region": float(bool(rec.get("no_answer_region")))}
    raise ValueError(f"unknown data_source {data_source!r}")


class ArgGymRewardManager(RewardManagerBase):
    def __init__(self, config, tokenizer, compute_score, reward_router_address=None,
                 reward_model_tokenizer=None, **_: Any):
        super().__init__(config, tokenizer, compute_score)
        self.stop_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id} - {None}
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
                                           self.stop_ids))
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
