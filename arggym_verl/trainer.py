"""verl's V1 trainer, plus a check that vLLM serves the policy being trained,
with `data.curriculum` ArgGYM's curriculum sampler, the step's paid reward beside
its exact success, with `trainer.auto_stop` a clean stop when the policy drifts,
and a `selection.json` kept current after every dev evaluation.

`install(mode)` replaces the trainer registered for `mode` by a subclass with
`ServedPolicyCheck`, `CurriculumFeedback`, `RewardMetrics`, `AutoStop` and
`DevSelection` mixed in;
nothing else about the trainer changes, and the mode string stays verl's, so every
branch verl takes on it is unchanged. Without `data.curriculum`,
`CurriculumFeedback` does nothing; without `trainer.auto_stop`, `AutoStop` does
nothing.

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

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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
        self.curriculum = cur.build(self.config.data, self.train_dataset,
                                    cur.paid_tasks(self.config.get("reward")))
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
        infos = _as_list(data["extra_info"])
        rewards = [f["reward_extra_info"] for f in _as_list(data["extra_fields"])]
        self.curriculum.update(cur.step_outcomes(self.curriculum, infos, rewards))
        metrics.update(cur.metrics(self.curriculum, self.global_steps,
                                   self.config.data.curriculum.table_every))
        metrics["curriculum/zero_signal_share"] = cur.zero_signal_share(infos, rewards)


class RewardMetrics:
    """Logs `arggym_verl.reward.step_metrics` each step: the paid reward (verl's
    `critic/score` is the same number) beside the exact success the curriculum and
    dev count, the shares of groups with no reward and no success variance, and both
    means per task. Read from the transfer queue in `_compute_advantage`, after the
    rewards and before the update."""

    def _compute_advantage(self, batch, metrics: dict):
        import transfer_queue as tq

        from arggym_verl.reward import step_metrics

        keys = [k for k, tag in zip(batch.keys, batch.tags) if not tag.get("is_padding", False)]
        data = tq.kv_batch_get(keys=keys, partition_id=batch.partition_id,
                               select_fields=["extra_info", "extra_fields"])
        rewards = [f["reward_extra_info"] for f in _as_list(data["extra_fields"])]
        metrics.update(step_metrics(_as_list(data["extra_info"]), rewards))
        return super()._compute_advantage(batch, metrics)


class StopRule:
    """When the run should stop, from each step's metrics.

    - `actor/ppo_kl` above `ppo_kl_max` for `ppo_kl_steps` consecutive steps: the
      update moves the policy further each step than PPO's trust region assumes.
    - `rollout_corr/kl`, vLLM's log-probs against the trainer's on the sampled
      tokens, above `rollout_kl_factor` times its mean over steps
      1..`baseline_steps`, checked from the step after: the policy has moved
      where the two engines disagree more than they did at the start.

    Steps 59-60 of the run that destabilised trip both (ppo_kl 1.15e-3, 1.80e-3;
    rollout_corr/kl 0.087 against a step 1-5 mean of 0.043).
    """

    def __init__(self, ppo_kl_max: float, ppo_kl_steps: int, rollout_kl_factor: float,
                 baseline_steps: int) -> None:
        self.ppo_kl_max, self.ppo_kl_steps = ppo_kl_max, ppo_kl_steps
        self.rollout_kl_factor, self.baseline_steps = rollout_kl_factor, baseline_steps
        self.over = 0  # consecutive steps with ppo_kl above its bound
        self.baseline: Dict[int, float] = {}  # step -> rollout_corr/kl, steps 1..baseline_steps
        self.warned = False

    @classmethod
    def from_config(cls, c) -> "StopRule":
        return cls(float(c.ppo_kl_max), int(c.ppo_kl_steps), float(c.rollout_kl_factor),
                   int(c.baseline_steps))

    def observe(self, step: int, metrics: Dict[str, Any]) -> Optional[str]:
        """Add step `step`'s metrics; the reason to stop after it, or None."""
        reasons = []
        kl = metrics.get("actor/ppo_kl")
        if kl is not None:
            self.over = self.over + 1 if kl > self.ppo_kl_max else 0
            if self.over >= self.ppo_kl_steps:
                reasons.append(f"actor/ppo_kl {kl:.2e} above {self.ppo_kl_max:.0e} for "
                               f"{self.over} consecutive steps")
        rk = metrics.get("rollout_corr/kl")
        if rk is not None:
            if step <= self.baseline_steps:
                self.baseline[step] = rk
            elif len(self.baseline) < self.baseline_steps:
                if not self.warned:
                    self.warned = True
                    print(f"auto-stop: the rollout_corr/kl rule is off: steps "
                          f"{sorted(self.baseline)} of 1-{self.baseline_steps} have the metric",
                          flush=True)
            else:
                ref = sum(self.baseline.values()) / len(self.baseline)
                if ref > 0 and rk > self.rollout_kl_factor * ref:
                    reasons.append(f"rollout_corr/kl {rk:.4f} above {self.rollout_kl_factor:g} x "
                                   f"its step 1-{self.baseline_steps} mean {ref:.4f}")
        return "; ".join(reasons) or None

    def replay(self, records: Iterable[Dict[str, Any]], before: int) -> None:
        """Rebuild the state from a resumed run's metrics file: each step below `before`,
        its last record (a resumed run logs the steps after its checkpoint again)."""
        last: Dict[int, Dict[str, Any]] = {}
        for rec in records:
            if 1 <= rec["step"] < before:
                last.setdefault(rec["step"], {}).update(rec["data"])
        for step in sorted(last):
            self.observe(step, last[step])


class AutoStop:
    """Stops the run cleanly at the end of a step that trips `StopRule` (`trainer.auto_stop`).

    The rule reads the step's metrics after its update (`step`). The stop happens in
    `on_step_end`, which verl's `fit` calls after the step's checkpoint is saved; a
    step off the `save_freq` grid is saved here first. The step then finishes as any
    other (metrics, rollouts) and `fit`'s `while` loop ends; `on_train_end` runs.
    The driver log gets one line starting `AUTO-STOP:`, and the run directory a file
    `AUTO-STOP` with the reason, which run.sh refuses to resume past.

    The stop step is also evaluated on dev. `fit` validates when its `is_last_step`,
    computed before the step from the original `total_training_steps`
    (trainer_base.py:482), or the step is on the `test_freq` grid (:505-506); lowering
    `total_training_steps` does not flip that local, so a stop off the grid would leave
    its checkpoint unevaluated. `on_step_end` then validates itself, after the weight
    sync (`super().on_step_end()`, as `fit` validates after it) and into the step's
    metrics, which `fit` logs with the step (:528); `DevSelection` writes
    `selection.json` with the checkpoint just saved in it. A stop that `fit` evaluates
    anyway is evaluated once.
    """

    stop_rule: Optional[StopRule] = None
    stop_reason: Optional[str] = None
    #: The current step's metrics dict, which `fit` logs at the end of the step.
    _step_metrics: Optional[dict] = None

    def step(self, metrics: dict, timing_raw: dict):
        batch = super().step(metrics, timing_raw)
        self._step_metrics = metrics
        c = self.config.trainer.get("auto_stop")
        if c:
            if self.stop_rule is None:
                self.stop_rule = StopRule.from_config(c)
                self.stop_rule.replay(_metrics_records(), before=self.global_steps)
            self.stop_reason = self.stop_rule.observe(self.global_steps, metrics)
        return batch

    def on_step_end(self):
        t = self.config.trainer
        validate = False
        if self.stop_reason:
            last = self.global_steps >= self.total_training_steps
            saved = t.save_freq > 0 and (last or self.global_steps % t.save_freq == 0)
            if not saved:
                self._save_checkpoint()
            # fit validates this step itself when it is its last or on the test_freq grid.
            validate = t.test_freq > 0 and not last and self.global_steps % t.test_freq != 0
            line = (f"step {self.global_steps}: {self.stop_reason}. Checkpoint "
                    f"global_step_{self.global_steps} saved; the run ends after this step.")
            print(f"AUTO-STOP: {line}", flush=True)
            (Path(t.default_local_dir).parent / "AUTO-STOP").write_text(line + "\n")
            # fit's loop runs while global_steps <= total_training_steps; it increments
            # global_steps after logging this step.
            self.total_training_steps = self.global_steps
        super().on_step_end()
        if validate:
            from verl.utils.debug import marked_timer

            with marked_timer("testing", self.timing_raw, color="green"):
                self.on_validate_begin()
                val = self._validate()
                self.on_validate_end()
            self._step_metrics.update(val)


class DevSelection:
    """Rewrites the run's `selection.json` (`arggym_verl.report`) after every dev evaluation.

    verl saves a step's checkpoint before it evaluates dev and logs the dev scores
    with the step's other metrics, after both. The file is written here, with the
    evaluation's scores in hand, so it never trails the latest dev score.
    """

    def _validate(self, *args, **kwargs):
        val = super()._validate(*args, **kwargs)
        path = os.environ.get("VERL_FILE_LOGGER_PATH")
        if val and path:
            from arggym_verl import report

            step = self.global_steps
            records = [r for r in _metrics_records() if r["step"] <= step]
            curve = report.dev_curve(records + [{"step": step, "data": val}])
            selection = report.select(curve, Path(self.config.trainer.default_local_dir))
            (Path(path).parent / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        return val


def _metrics_records() -> List[Dict[str, Any]]:
    """The run's metrics records so far, oldest first: verl's file logger
    (`VERL_FILE_LOGGER_PATH`, set by run.sh) truncates its file at each launch, and run.sh
    keeps the earlier launches' files beside it (`arggym_verl.report.metrics_files`)."""
    from arggym_verl import report

    path = os.environ.get("VERL_FILE_LOGGER_PATH")
    if not path:
        return []
    return report.read_jsonl(report.metrics_files(Path(path).parent))


def _as_list(field) -> list:
    """A non-tensor field from the queue: a NonTensorStack (`.tolist()`), or for a field
    of dicts, a tensordict LinkedList, which is already a list."""
    return field.tolist() if hasattr(field, "tolist") else list(field)


def install(mode: str) -> type:
    """Register `mode`'s trainer with the mixins; returns the new class."""
    from verl.trainer.ppo.v1 import trainer_base

    base = trainer_base.TRAINER_REGISTRY[mode]
    if issubclass(base, ServedPolicyCheck):
        return base
    checked = type(f"Checked{base.__name__}",
                   (ServedPolicyCheck, CurriculumFeedback, RewardMetrics, AutoStop, DevSelection,
                    base), {})
    trainer_base.TRAINER_REGISTRY[mode] = checked
    return checked
