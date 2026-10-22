"""The served-policy check: vLLM serving the adapter passes, vLLM serving the base fails.

The step-order tests run verl's own `PPOTrainer._step_once` over stub passes
that write to an in-memory transfer queue with the real one's field semantics.
"""
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from arggym_verl.trainer import (TRACK_FAIL, CurriculumFeedback, ServedPolicyCheck, check_served,
                                 install, served_stats)


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
    assert issubclass(cls, ServedPolicyCheck) and issubclass(cls, CurriculumFeedback)
    assert issubclass(cls, PPOTrainerColocateAsync)
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
        return Padded({f: torch.stack(v) if isinstance(v[0], torch.Tensor)
                       else LinkedList(v) if f == "extra_info" else Listed(v)
                       for f in present for v in [[self.rows[k][f] for k in keys]]})


class Padded(dict):
    def to_padded_tensor(self):
        return self


class LinkedList(list):
    """`extra_info` as the queue returned it on GPU: tensordict's LinkedList, no `.tolist()`."""


class Listed(list):
    """A non-tensor field, as the queue returns one (`.tolist()`)."""

    def tolist(self):
        return list(self)


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


# ---------------------------------------------------------------- the curriculum in verl's step order


class Rows(torch.utils.data.Dataset):
    """What verl's dataloader reads from RLHFDataset: the training parquet's rows."""

    def __init__(self):
        import datasets

        from conftest import ROOT

        self.dataframe = datasets.load_dataset("parquet", split="train", data_files=str(
            ROOT / "data/arggym-qwen3.5-4b/train.parquet"))
        self.extra_info = self.dataframe["extra_info"]

    def __len__(self):
        return len(self.extra_info)

    def __getitem__(self, i):
        return {"raw_prompt": [], "extra_info": self.extra_info[i]}


class StepStub(StubPasses):
    """verl's `step`: `prepare_step` fetches the batch through verl's own dataloader path,
    a stub agent loop puts `rollout.n` scored completions a prompt into the queue, and
    `_step_once` samples them back in a shuffled order, as `_balance_batch` leaves them."""

    from verl.trainer.ppo.v1.trainer_base import PPOTrainer
    step = PPOTrainer.step
    prepare_step = PPOTrainer.prepare_step
    _add_batch_to_generate = PPOTrainer._add_batch_to_generate
    _next_train_batch = PPOTrainer._next_train_batch
    _fetch_one_gen_batch = PPOTrainer._fetch_one_gen_batch

    def __init__(self, queue, config, dataset):
        from transfer_queue import KVBatchMeta

        n = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        super().__init__(queue, use_kl=True, serve_adapter=True, versions=[2] * n)
        self.config, self.train_dataset = config, dataset
        self.trainer_mode, self.parameter_sync_step = config.trainer.v1.trainer_mode, 1
        self.global_steps = 1
        self.submitted = []
        self.replay_buffer = SimpleNamespace(sample=lambda **kw: (KVBatchMeta(
            keys=self.submitted, tags=[{"min_global_steps": 2}] * n, partition_id="train"), {}))
        self._init_dataloader()

    def _init_dataloader(self):
        """verl's, on this dataset: one prompt a fetch in the async modes, verl's sampler."""
        from torchdata.stateful_dataloader import StatefulDataLoader
        from verl.trainer.ppo.utils import create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset, batch_size=1,
            num_workers=self.config.data.dataloader_num_workers, drop_last=True,
            collate_fn=collate_fn, sampler=create_rl_sampler(self.config.data, self.train_dataset))
        self.train_dataloader_it = None

    def _submit_batch_to_rollout(self, batch):
        import random

        infos = list(batch["extra_info"])
        self.calls.append(("generate", len(infos)))
        keys = []
        for uid, info in zip(list(batch["uid"]), infos, strict=True):
            for s in range(self.config.actor_rollout_ref.rollout.n):
                key = f"{uid}_{s}_0"
                self.queue.rows[key] = {"extra_info": info, "extra_fields": {"reward_extra_info": {
                    "acc": float((info["index"] + s) % 3 == 0), "refused": 0.0, "truncated": 0.0}}}
                keys.append(key)
        random.Random(len(self.calls)).shuffle(keys)
        self.submitted = keys
        return len(infos)


class CurriculumChecked(ServedPolicyCheck, CurriculumFeedback, StepStub):
    pass


@pytest.fixture(scope="module")
def arggym_cfg():
    from hydra import compose, initialize_config_dir

    from conftest import ROOT

    with initialize_config_dir(config_dir=str(ROOT / "arggym_verl/config"), version_base=None):
        cfg = compose(config_name="arggym-qwen3.5-4b",
                      overrides=["hydra.searchpath=[pkg://verl.trainer.config]"])
    cfg.data.gen_batch_size = 1  # trainer_base._init_dataloader, in the async modes
    return cfg


def test_curriculum_learns_each_step_before_the_next_draw(queue, arggym_cfg):
    """In verl's step order, step k's outcomes reach the curriculum (in `_compute_advantage`,
    before the update) before step k+1's prompts are drawn, and no prompt is drawn ahead."""
    trainer = CurriculumChecked(queue, arggym_cfg, Rows())
    c = trainer.curriculum
    assert trainer.train_dataloader.sampler is c and trainer.train_dataloader.num_workers == 0
    draw, update = c.draw, c.update
    c.draw = lambda: (trainer.calls.append(("draw", c.blocks)), draw())[1]
    c.update = lambda outcomes: (trainer.calls.append(("learn", c.observed)), update(outcomes))[1]
    for k in range(3):
        metrics = {}
        trainer.step(metrics, {})
        assert (c.blocks, c.observed) == (k + 1, k + 1)
        assert metrics["curriculum/groups"] == arggym_cfg.data.train_batch_size and metrics["served/armed"] == 1.0
        assert ("curriculum/pass_rate/attack/L1" in metrics) == (trainer.global_steps == 1)
        trainer.global_steps += 1
    step = ["draw", ("generate", arggym_cfg.data.train_batch_size), "old", "ref", "learn", "adv", "update"]
    assert [x if x[0] == "generate" or isinstance(x, str) else x[0] for x in trainer.calls] == step * 3
    assert [x[1] for x in trainer.calls if x[0] in ("draw", "learn")] == [0, 0, 1, 1, 2, 2]
    # The trained groups are the drawn ones, one per prompt.
    assert [o["n"] for o in c.draws[2]["outcomes"]] == [arggym_cfg.actor_rollout_ref.rollout.n] * arggym_cfg.data.train_batch_size


def test_without_curriculum_verls_sampler_stays(queue, arggym_cfg):
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(OmegaConf.to_container(arggym_cfg))
    del cfg.data["curriculum"]
    trainer = CurriculumChecked(queue, cfg, Rows())
    assert trainer.curriculum is None
    assert type(trainer.train_dataloader.sampler).__name__ == "RandomSampler"
