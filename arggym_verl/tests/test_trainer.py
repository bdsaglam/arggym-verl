"""The served-policy check: vLLM serving the adapter passes, vLLM serving the base fails.

The step-order tests run verl's own `PPOTrainer._step_once` over stub passes
that write to an in-memory transfer queue with the real one's field semantics.
"""
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from arggym_verl.trainer import (TRACK_FAIL, ServedPolicyCheck, check_served, install,
                                 served_stats)


def engines(serve_adapter: bool, shift: float = 0.02, noise: float = 0.08, n=(64, 512), seed=0):
    g = torch.Generator().manual_seed(seed)
    base = -torch.rand(n, generator=g) * 3
    trained = base + shift * torch.randn(n, generator=g)
    served = trained if serve_adapter else base
    rollout = served + noise * torch.randn(n, generator=g)  # BF16 vs FP32 kernel gap
    mask = torch.ones(n)
    mask[:, 400:] = 0
    return rollout, trained, base, mask


def test_adapter_served_passes_even_when_the_kernel_gap_is_larger_than_the_update():
    stats = served_stats(*engines(True), is_threshold=2.0)
    assert stats["served/armed"] == 1.0
    assert stats["served/track"] == pytest.approx(1.0, abs=0.1)
    check_served(stats)


def test_base_served_fails():
    stats = served_stats(*engines(False), is_threshold=2.0)
    assert stats["served/armed"] == 1.0
    assert abs(stats["served/track"]) < 0.1
    with pytest.raises(RuntimeError, match="not serving the policy"):
        check_served(stats)


def test_one_version_stale_adapter_passes():
    """Served version 1 against trainer version 2 under steady drift: track ~0.5."""
    r, t, b, m = engines(True)
    t2 = b + 2 * (t - b)
    stats = served_stats(r, t2, b, m, is_threshold=2.0)
    assert stats["served/armed"] == 1.0
    assert stats["served/track"] == pytest.approx(0.5, abs=0.05)
    check_served(stats)


def test_untrained_adapter_does_not_arm():
    r, _, b, m = engines(True)
    stats = served_stats(r, b, b, m, is_threshold=2.0)
    assert stats["served/armed"] == 0.0 and stats["served/adapter_shift_mean"] == 0.0
    assert "served/track" not in stats
    check_served(stats)


def test_tiny_shift_does_not_arm():
    """Below the noise, neither outcome is decided, so the run goes on."""
    stats = served_stats(*engines(False, shift=1e-4), is_threshold=2.0)
    assert stats["served/armed"] == 0.0
    check_served(stats)


def test_gap_and_truncation_are_over_masked_tokens():
    r, t, b, m = engines(True)
    r[:, 450] = t[:, 450] - 5.0  # outside the mask: must not count
    stats = served_stats(r, t, b, m, is_threshold=2.0)
    assert stats["served/tokens"] == 64 * 400
    assert stats["served/logp_gap_max"] < 1.0
    assert stats["served/is_truncated_fraction"] == pytest.approx(
        ((t - r)[:, :400].exp() > 2.0).double().mean().item())


def test_install_wraps_the_registered_mode():
    from verl.trainer.ppo.v1 import get_trainer_cls
    from verl.trainer.ppo.v1.trainer_colocate_async import PPOTrainerColocateAsync

    cls = install("colocate_async")
    assert get_trainer_cls("colocate_async") is cls
    assert issubclass(cls, ServedPolicyCheck) and issubclass(cls, PPOTrainerColocateAsync)
    assert install("colocate_async") is cls


# ---------------------------------------------------------------- verl's step order


class FakeQueue:
    """kv_batch_get/put over a dict, with transfer_queue's semantics: a field absent from
    every requested key is dropped from the result, one absent from some keys raises."""

    def __init__(self):
        self.rows = {}

    def put(self, keys, fields):
        for i, k in enumerate(keys):
            self.rows.setdefault(k, {}).update({f: v[i] for f, v in fields.items()})

    def kv_batch_get(self, keys, partition_id, select_fields):
        present = [f for f in select_fields if any(f in self.rows[k] for k in keys)]
        if any(f not in self.rows[k] for f in present for k in keys):
            raise ValueError("Some fields are not ready in all the requested keys!")
        return Padded({f: torch.stack([self.rows[k][f] for k in keys]) for f in present})


class Padded(dict):
    def to_padded_tensor(self):
        return self


class StubPasses:
    """verl's trainer with every worker call replaced; `_step_once` is verl's own."""

    from verl.trainer.ppo.v1.trainer_base import PPOTrainer
    _step_once = PPOTrainer._step_once

    def __init__(self, queue, use_kl, serve_adapter, versions):
        from transfer_queue import KVBatchMeta

        self.config = OmegaConf.create({
            "algorithm": {"rollout_correction": {"rollout_is": "token", "rollout_is_threshold": 2.0,
                                                 "bypass_mode": False}},
            "actor_rollout_ref": {"rollout": {"temperature": 1.0}},
            "trainer": {"critic_warmup": 0}})
        self.use_reference_policy = use_kl
        self.ref_in_actor, self.use_critic, self.global_steps = True, False, 3
        self.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=object())
        self.queue, self.calls = queue, []
        n = len(versions)
        r, t, b, m = engines(serve_adapter, n=(n, 512))
        stale = torch.tensor([v == 0 for v in versions])
        r[stale] = b[stale] + 0.08 * torch.randn(int(stale.sum()), 512)  # version 0 is the base
        self.tensors = {"rollout_log_probs": r, "old_log_probs": t, "ref": b, "response_mask": m}
        keys = [f"k{i}" for i in range(n)]
        tags = [{"min_global_steps": v, "max_global_steps": v} for v in versions]
        self.replay_buffer = SimpleNamespace(
            sample=lambda **kw: (KVBatchMeta(keys=keys, tags=tags, partition_id="train"), {}))

    def on_sample_begin(self):
        pass

    def on_sample_end(self):
        pass

    def _balance_batch(self, batch, metrics):
        return batch

    def _compute_old_log_prob(self, batch, metrics):
        self.calls.append("old")
        self.queue.put(batch.keys, {f: self.tensors[f]
                                    for f in ("rollout_log_probs", "old_log_probs", "response_mask")})
        return batch

    def _compute_ref_log_prob(self, batch, metrics):
        self.calls.append("ref")
        batch.extra_info["no_lora_adapter"] = True
        self.queue.put(batch.keys, {"ref_log_prob": self.tensors["ref"]})
        return batch

    def _compute_advantage(self, batch, metrics):
        self.calls.append("adv")
        return batch

    def _update_actor(self, batch, metrics):
        self.calls.append("update")
        if self.use_reference_policy:  # the KL loss reads the reference
            self.queue.kv_batch_get(batch.keys, batch.partition_id, ["ref_log_prob"])["ref_log_prob"]
        return batch


class Checked(ServedPolicyCheck, StubPasses):
    pass


@pytest.fixture
def queue(monkeypatch):
    import transfer_queue

    q = FakeQueue()
    monkeypatch.setattr(transfer_queue, "kv_batch_get", q.kv_batch_get)
    return q


def step(trainer):
    metrics = {}
    trainer._step_once(metrics, {}, sample_batch_size=None)
    return metrics


def test_kl_on_reuses_verls_reference_pass(queue):
    trainer = Checked(queue, use_kl=True, serve_adapter=True, versions=[2] * 64)
    metrics = step(trainer)
    assert trainer.calls == ["old", "ref", "adv", "update"]
    assert metrics["served/armed"] == 1.0 and metrics["served/track"] > 0.9


def test_kl_off_adds_the_adapter_off_pass_once(queue):
    trainer = Checked(queue, use_kl=False, serve_adapter=True, versions=[2] * 64)
    metrics = step(trainer)
    assert trainer.calls == ["old", "ref", "adv", "update"]
    assert metrics["served/track"] > 0.9


@pytest.mark.parametrize("use_kl", [True, False])
def test_base_served_stops_the_step_before_the_update(queue, use_kl):
    trainer = Checked(queue, use_kl=use_kl, serve_adapter=False, versions=[2] * 64)
    with pytest.raises(RuntimeError, match="not serving the policy"):
        step(trainer)
    assert "update" not in trainer.calls


def test_version_0_rollouts_are_left_out(queue):
    """Rows generated by the untouched adapter are the base model's, rightly; they must not fail."""
    trainer = Checked(queue, use_kl=True, serve_adapter=True, versions=[0] * 32 + [1] * 32)
    metrics = step(trainer)
    assert metrics["served/tokens"] == 32 * 400
    assert metrics["served/track"] > 1 - TRACK_FAIL


def test_only_version_0_rollouts_do_not_arm(queue):
    trainer = Checked(queue, use_kl=True, serve_adapter=False, versions=[0] * 64)
    metrics = step(trainer)
    assert metrics["served/armed"] == 0.0 and trainer.calls[-1] == "update"
