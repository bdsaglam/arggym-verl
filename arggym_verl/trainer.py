"""verl's V1 trainer, plus a check that vLLM serves the policy being trained and,
with `data.curriculum`, ArgGYM's curriculum sampler.

`install(mode)` replaces the trainer registered for `mode` by a subclass with
`ServedPolicyCheck` and `CurriculumFeedback` mixed in; nothing else about the
trainer changes, and the mode string stays verl's, so every branch verl takes on
it is unchanged. Without `data.curriculum`, `CurriculumFeedback` does nothing.

Every training step has, for each sampled token, three log-probs: vLLM's
(`rollout_log_probs`), the trainer's with the adapter (`old_log_probs`), and the
trainer's with the adapter off (`ref_log_prob`, verl's reference-in-actor path
for LoRA, which the KL loss computes anyway). Per token, the trainer-side shift
is d = trained - base and the served shift is s = vllm - base. The check fits
s = track * d by least squares over the step's tokens:

- vLLM serves the adapter: track ~1 (below 1 when the rollout is one weight
  version stale, ~v/(k-1) for version v against the trainer's k-1).
- vLLM serves the base model: track ~0.

The engines' kernel gap is the regression's noise, so `served/track_se` =
std(residual) / sqrt(sum d^2) shrinks as the adapter moves. The check arms once
`served/track_se` is below `TRACK_SE_ARM` (at step 1 the adapter is untouched
and d is 0) and then stops the run if `served/track` falls below `TRACK_FAIL`.
Only trajectories generated entirely by a trained adapter (weight version >= 1)
enter; a version-0 rollout is the base model by construction.

The same step logs the trainer-vs-vLLM precision gap per token: mean and max of
|trained - vllm|, and the share of tokens whose IS weight exceeds the
truncation threshold.
"""
from __future__ import annotations

import math
from typing import Dict

import torch

#: The check arms when base (track 0) and adapter (track >= ~0.5) are this many SEs from TRACK_FAIL.
TRACK_SE_ARM = 0.05
#: Armed, a served shift below this fraction of the trainer's shift stops the run.
TRACK_FAIL = 0.25


def served_stats(rollout: torch.Tensor, trained: torch.Tensor, base: torch.Tensor,
                 mask: torch.Tensor, is_threshold: float) -> Dict[str, float]:
    """Token-level statistics over `mask`; all tensors (batch, response_length)."""
    m = mask.bool()
    r, t, b = rollout[m].double(), trained[m].double(), base[m].double()
    n = r.numel()
    if n == 0:
        return {"served/armed": 0.0, "served/tokens": 0.0}
    gap = (t - r).abs()
    d, s = t - b, r - b
    stats = {"served/adapter_shift_mean": d.abs().mean().item(),
             "served/logp_gap_mean": gap.mean().item(),
             "served/logp_gap_max": gap.max().item(),
             "served/is_truncated_fraction": ((t - r).exp() > is_threshold).double().mean().item(),
             "served/tokens": float(n),
             "served/armed": 0.0}
    dd = (d * d).sum().item()
    if dd > 0 and n > 1:
        track = (s * d).sum().item() / dd
        se = (s - track * d).std().item() / math.sqrt(dd)
        stats.update({"served/track": track, "served/track_se": se,
                      "served/armed": float(se <= TRACK_SE_ARM)})
    return stats


def check_served(stats: Dict[str, float]) -> None:
    if stats.get("served/armed") and stats["served/track"] < TRACK_FAIL:
        raise RuntimeError(
            f"vLLM's log-probs moved {stats['served/track']:.2f} (+- {stats['served/track_se']:.2f}) "
            f"of the trained adapter's shift from the base model (adapter shift "
            f"{stats['served/adapter_shift_mean']:.2e} nats/token): the rollout server is not "
            f"serving the policy being trained")


class ServedPolicyCheck:
    def _compute_advantage(self, batch, metrics: dict):
        # Runs after verl's old-log-prob and (with a KL loss) reference passes, before the update.
        rc = self.config.algorithm.get("rollout_correction", None) or {}
        if self.ref_in_actor and not rc.get("bypass_mode", False):
            if not self.use_reference_policy:
                # No KL loss: compute the adapter-off pass verl would compute for one.
                batch = self._compute_ref_log_prob(batch, metrics={})
                batch.extra_info.pop("no_lora_adapter", None)
            stats = self._served_stats(batch, float(rc.get("rollout_is_threshold", 2.0)))
            metrics.update(stats)
            check_served(stats)
        return super()._compute_advantage(batch, metrics)

    def _served_stats(self, batch, is_threshold: float) -> Dict[str, float]:
        import transfer_queue as tq

        # Trajectories generated entirely by a trained adapter; padding rows repeat real ones.
        keys = [k for k, tag in zip(batch.keys, batch.tags)
                if not tag.get("is_padding", False) and (tag.get("min_global_steps") or 0) >= 1]
        if not keys:
            return {"served/armed": 0.0, "served/tokens": 0.0}
        fields = ["rollout_log_probs", "old_log_probs", "ref_log_prob", "response_mask"]
        data = tq.kv_batch_get(keys=keys, partition_id=batch.partition_id, select_fields=fields)
        missing = [f for f in fields if f not in data.keys()]
        if missing:
            raise RuntimeError(f"served-policy check: {missing} not in the transfer queue")
        data = data.to_padded_tensor()
        return served_stats(*(data[f] for f in fields), is_threshold=is_threshold)


class CurriculumFeedback:
    """Draws training prompts with `arggym_verl.curriculum` and feeds it each step's outcomes.

    The sampler replaces verl's in the training dataloader (`_init_dataloader`).
    Each step's outcomes are read from the transfer queue in `_compute_advantage`,
    which runs once a step on the driver, after the rewards and before the update;
    the next step's prompts are fetched after it (`prepare_step`).
    """

    curriculum = None

    def _init_dataloader(self):
        super()._init_dataloader()
        if not self.config.data.get("curriculum"):
            return
        from torchdata.stateful_dataloader import StatefulDataLoader

        from arggym_verl import curriculum as cur

        cur.check_setup(self.config, self.trainer_mode, self.parameter_sync_step)
        self.curriculum = cur.build(self.config.data, self.train_dataset)
        verls = self.train_dataloader
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset, batch_size=verls.batch_size, num_workers=0,
            drop_last=verls.drop_last, collate_fn=verls.collate_fn, sampler=self.curriculum)

    def _compute_advantage(self, batch, metrics: dict):
        if self.curriculum is not None:
            self._feed_curriculum(batch, metrics)
        return super()._compute_advantage(batch, metrics)

    def _feed_curriculum(self, batch, metrics: dict) -> None:
        import transfer_queue as tq

        from arggym_verl import curriculum as cur

        keys = [k for k, tag in zip(batch.keys, batch.tags) if not tag.get("is_padding", False)]
        data = tq.kv_batch_get(keys=keys, partition_id=batch.partition_id,
                               select_fields=["extra_info", "extra_fields"])
        infos = data["extra_info"].tolist()
        rewards = [f["reward_extra_info"] for f in data["extra_fields"].tolist()]
        self.curriculum.update(cur.step_outcomes(self.curriculum, infos, rewards))
        metrics.update(cur.metrics(self.curriculum, self.global_steps,
                                   self.config.data.curriculum.table_every))


def install(mode: str) -> type:
    """Register `mode`'s trainer with both mixins; returns the new class."""
    from verl.trainer.ppo.v1 import trainer_base

    base = trainer_base.TRAINER_REGISTRY[mode]
    if issubclass(base, ServedPolicyCheck):
        return base
    checked = type(f"Checked{base.__name__}", (ServedPolicyCheck, CurriculumFeedback, base), {})
    trainer_base.TRAINER_REGISTRY[mode] = checked
    return checked
