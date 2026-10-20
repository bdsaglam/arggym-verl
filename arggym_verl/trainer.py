"""verl's V1 trainer, plus a check that vLLM serves the policy being trained.

`install(mode)` replaces the trainer registered for `mode` by a subclass with
`ServedPolicyCheck` mixed in; nothing else about the trainer changes, and the
mode string stays verl's, so every branch verl takes on it is unchanged.

Every training step already has, for each sampled token, the log-prob vLLM
reported (`rollout_log_probs`) and the trainer's own (`old_log_probs`,
recomputed with the adapter). The check adds one forward pass with the adapter
off (`no_lora_adapter`, verl's own reference-in-actor path for LoRA) and asks
which of the two vLLM is closer to. Per token,

    x = (vllm - base)^2 - (vllm - trained)^2

has mean +delta^2 when vLLM serves the adapter and -delta^2 when it serves the
base model, delta being the adapter's shift; the kernel gap between the engines
is in both terms and only adds variance. `served/z`, the mean of x over its
standard error, grows with every step once the adapter has moved; a value
below `SERVED_Z_FAIL` stops the run.

The same step logs the trainer-vs-vLLM precision gap per token: mean and max of
|trained - vllm|, and the share of tokens whose IS weight exceeds the
truncation threshold.
"""
from __future__ import annotations

import math
from typing import Dict

import torch

#: Below this, vLLM is closer to the base model than to the trained one beyond doubt.
SERVED_Z_FAIL = -5.0


def served_stats(rollout: torch.Tensor, trained: torch.Tensor, base: torch.Tensor,
                 mask: torch.Tensor, is_threshold: float) -> Dict[str, float]:
    """Token-level statistics over `mask`; all tensors (batch, response_length)."""
    m = mask.bool()
    r, t, b = rollout[m].double(), trained[m].double(), base[m].double()
    n = r.numel()
    if n == 0:
        return {}
    gap = (t - r).abs()
    x = (r - b) ** 2 - (r - t) ** 2
    std = x.std().item() if n > 1 else 0.0
    z = x.mean().item() / (std / math.sqrt(n)) if std > 0 else 0.0
    return {"served/z": z,
            "served/excess_sq_mean": x.mean().item(),
            "served/adapter_shift_mean": (t - b).abs().mean().item(),
            "served/logp_gap_mean": gap.mean().item(),
            "served/logp_gap_max": gap.max().item(),
            "served/is_truncated_fraction": ((t - r).exp() > is_threshold).double().mean().item(),
            "served/tokens": float(n)}


def check_served(stats: Dict[str, float]) -> None:
    z = stats.get("served/z", 0.0)
    if z < SERVED_Z_FAIL:
        raise RuntimeError(
            f"vLLM's log-probs are closer to the base model than to the trained adapter "
            f"(served/z {z:.1f}, adapter shift {stats['served/adapter_shift_mean']:.2e}): "
            f"the rollout server is not serving the policy being trained")


class ServedPolicyCheck:
    def _compute_old_log_prob(self, batch, metrics: dict):
        import transfer_queue as tq

        batch = super()._compute_old_log_prob(batch, metrics)
        rc = self.config.algorithm.get("rollout_correction", None) or {}
        if not self.ref_in_actor or rc.get("bypass_mode", False):
            return batch  # no adapter to switch off, or old_log_probs are vLLM's own
        if not self.use_reference_policy:
            # Adapter off: the base model's log-probs, as verl computes a LoRA reference.
            batch = self._compute_ref_log_prob(batch, metrics={})
            batch.extra_info.pop("no_lora_adapter", None)
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id,
                               select_fields=["rollout_log_probs", "old_log_probs",
                                              "ref_log_prob", "response_mask"])
        data = data.to_padded_tensor()
        fields = ("rollout_log_probs", "old_log_probs", "ref_log_prob", "response_mask")
        width = min(data[k].shape[-1] for k in fields)
        r, t, b, m = (data[k][..., :width] for k in fields)
        stats = served_stats(r, t, b, m,
                             float(rc.get("rollout_is_threshold", 2.0)))
        metrics.update(stats)
        check_served(stats)
        return batch


def install(mode: str) -> type:
    """Register `mode`'s trainer with the check mixed in; returns the new class."""
    from verl.trainer.ppo.v1 import trainer_base

    base = trainer_base.TRAINER_REGISTRY[mode]
    if issubclass(base, ServedPolicyCheck):
        return base
    checked = type(f"Checked{base.__name__}", (ServedPolicyCheck, base), {})
    trainer_base.TRAINER_REGISTRY[mode] = checked
    return checked
