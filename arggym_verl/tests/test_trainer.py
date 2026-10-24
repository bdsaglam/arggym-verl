"""The trainer mixins: the served-policy check (vLLM serving the adapter passes, vLLM
serving the base fails), the curriculum feed, the auto-stop and the dev selection.

The step-order tests run verl's own `PPOTrainer._step_once`, `step` and `fit` over
stub passes that write to an in-memory transfer queue with the real one's field
semantics.
"""
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from arggym_verl.trainer import (TRACK_FAIL, AutoStop, CurriculumFeedback, DevSelection,
                                 ServedPolicyCheck, StopRule, check_served, install, served_stats)


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
    for mixin in (ServedPolicyCheck, CurriculumFeedback, AutoStop, DevSelection):
        assert issubclass(cls, mixin), mixin
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
                acc = float((info["index"] + s) % 3 == 0)
                self.queue.rows[key] = {"extra_info": info, "extra_fields": {"reward_extra_info": {
                    "score": acc, "acc": acc, "refused": 0.0, "truncated": 0.0}}}
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
        assert 0 <= metrics["curriculum/zero_signal_share"] <= 1
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


# ---------------------------------------------------------------- auto-stop


def rule():
    return StopRule(ppo_kl_max=1e-3, ppo_kl_steps=2, rollout_kl_factor=2.0, baseline_steps=5)


def feed(r, seq):
    """seq: (ppo_kl, rollout_corr/kl) per step from step 1; the steps whose reason is set."""
    return {step: why for step, (pk, rk) in enumerate(seq, start=1)
            if (why := r.observe(step, {"actor/ppo_kl": pk, "rollout_corr/kl": rk}))}


def test_ppo_kl_stops_after_two_consecutive_steps_above():
    seq = [(1e-4, 0.04)] * 5 + [(2e-3, 0.04), (5e-4, 0.04), (2e-3, 0.04), (-3e-3, 0.04),
                                (1.1e-3, 0.04), (1.2e-3, 0.04)]
    stops = feed(rule(), seq)
    assert list(stops) == [11] and "actor/ppo_kl 1.20e-03 above 1e-03 for 2 consecutive" in stops[11]
    assert feed(rule(), [(1e-3, 0.04)] * 8) == {}  # at the bound is not above it


def test_rollout_kl_stops_above_twice_its_step_1_to_5_mean():
    base = [(0.0, k) for k in (0.03, 0.05, 0.04, 0.035, 0.045)]  # mean 0.04
    assert feed(rule(), base + [(0.0, 0.079), (0.0, 0.07)]) == {}
    stops = feed(rule(), base + [(0.0, 0.079), (0.0, 0.081)])
    assert list(stops) == [7] and "rollout_corr/kl 0.0810 above 2 x its step 1-5 mean 0.0400" in stops[7]
    # Steps 1-5 set the baseline and never stop the run, however high.
    assert feed(rule(), [(0.0, 0.04)] * 4 + [(0.0, 0.5), (0.0, 0.1)]) == {}


def test_stops_where_the_destabilised_run_did():
    """The last ArgGYM run (LR 3e-5, card sampling): (actor/ppo_kl, rollout_corr/kl) at steps
    1-5 and 55-62. Both rules first fire at step 60, two steps after entropy began to climb."""
    first = [(-1.78e-4, 0.04355), (-1.10e-4, 0.04329), (-1.50e-4, 0.04123), (-9.93e-5, 0.04102),
             (-5.42e-5, 0.04582)]
    late = {55: (7.05e-4, 0.0581), 56: (1.47e-4, 0.0585), 57: (7.38e-4, 0.0623),
            58: (7.61e-4, 0.0606), 59: (1.15e-3, 0.0753), 60: (1.80e-3, 0.0869),
            61: (-4.86e-4, 0.0896), 62: (2.18e-3, 0.0999)}
    r = rule()
    for step, (pk, rk) in enumerate(first, start=1):
        assert r.observe(step, {"actor/ppo_kl": pk, "rollout_corr/kl": rk}) is None
    stops = {s: r.observe(s, {"actor/ppo_kl": pk, "rollout_corr/kl": rk}) for s, (pk, rk) in late.items()}
    assert min(s for s, why in stops.items() if why) == 60
    assert "actor/ppo_kl" in stops[60] and "rollout_corr/kl" in stops[60]


def test_replay_restores_the_state_of_a_resumed_run():
    seq = [(1e-4, 0.04)] * 5 + [(1e-4, 0.05), (2e-3, 0.05)]
    live = rule()
    feed(live, seq)
    records = [{"step": 0, "data": {"val-core/arggym/acc/mean@8": 0.3}}]
    records += [{"step": s, "data": {"actor/ppo_kl": pk, "rollout_corr/kl": rk}}
                for s, (pk, rk) in enumerate(seq, start=1)]
    records += [{"step": 8, "data": {"actor/ppo_kl": 9.0, "rollout_corr/kl": 9.0}}]  # after the checkpoint
    resumed = rule()
    resumed.replay(records, before=8)
    assert (resumed.over, resumed.baseline) == (live.over, live.baseline) == (1, {s: 0.04 for s in range(1, 6)})
    assert "actor/ppo_kl" in resumed.observe(8, {"actor/ppo_kl": 2e-3, "rollout_corr/kl": 0.05})


class FitStub:
    """verl's `fit` loop with every worker call replaced; `step` returns the scripted metrics."""

    from verl.trainer.ppo.v1.trainer_base import PPOTrainer
    fit = PPOTrainer.fit

    def __init__(self, script, total=10, save_freq=5, auto_stop=True):
        trainer = {"project_name": "p", "experiment_name": "e", "logger": [], "val_before_train": False,
                   "total_epochs": 1, "save_freq": save_freq, "test_freq": 0, "rollout_data_dir": None,
                   "default_local_dir": "/nonexistent"}
        if auto_stop:
            trainer["auto_stop"] = {"ppo_kl_max": 1e-3, "ppo_kl_steps": 2, "rollout_kl_factor": 2.0,
                                    "baseline_steps": 5}
        self.config = OmegaConf.create({"trainer": trainer, "global_profiler": {"steps": None}})
        self.script, self.global_steps, self.total_training_steps = script, 0, total
        self.steps_per_epoch = total
        self.saved, self.logged, self.synced, self.ended = [], [], [], False

    def step(self, metrics, timing_raw):
        pk, rk = self.script[self.global_steps - 1]
        metrics.update({"actor/ppo_kl": pk, "rollout_corr/kl": rk})
        return SimpleNamespace(keys=[], partition_id="train")

    def _save_checkpoint(self):
        self.saved.append(self.global_steps)

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        self.logged.append(global_steps)

    def on_step_end(self):  # colocate_async: the weight sync
        self.synced.append(self.global_steps)

    def on_train_end(self):
        self.ended = True

    def _reissue_inflight_prompts(self):
        pass

    def _consume_sync_metrics(self):
        return {}

    def _shutdown_dump_executor(self):
        pass

    def __getattr__(self, name):  # on_step_begin, _start_profiling, ...: verl's no-op hooks
        if name.startswith(("on_", "_start_", "_stop_")):
            return lambda *a, **k: None
        raise AttributeError(name)


class StoppingFit(AutoStop, FitStub):
    pass


@pytest.fixture
def quiet_fit(monkeypatch):
    import transfer_queue
    from verl.trainer.ppo.v1 import trainer_base

    class Nothing:
        def __init__(self, *a, **k):
            pass

        def log(self, *a, **k):
            pass

    for name in ("Tracking", "ValidationGenerationsLogger", "DapoFilteredRewardTableLogger"):
        monkeypatch.setattr(trainer_base, name, Nothing)
    monkeypatch.setattr(trainer_base, "SkipManager",
                        SimpleNamespace(init=lambda config: None, set_step=lambda step: None))
    monkeypatch.setattr(transfer_queue, "kv_clear", lambda **k: None)
    monkeypatch.delenv("VERL_FILE_LOGGER_PATH", raising=False)


@pytest.mark.parametrize("stop_at, saved", [(7, [5, 7]), (10, [5, 10])])
def test_auto_stop_ends_verls_fit_after_the_step_and_its_checkpoint(quiet_fit, capsys, stop_at, saved):
    """In verl's own `fit`, a tripped step is saved (off the save_freq grid too), logs its
    metrics, prints AUTO-STOP and is the last step; on_train_end runs as at a normal end."""
    script = [(1e-4, 0.04)] * 20
    script[stop_at - 2] = script[stop_at - 1] = (2e-3, 0.04)
    trainer = StoppingFit(script, total=20)
    trainer.fit(agent_loop_manager=None)
    assert trainer.logged == list(range(1, stop_at + 1))
    assert trainer.saved == saved and trainer.ended
    assert trainer.synced == trainer.logged
    out = [line for line in capsys.readouterr().out.splitlines() if line.startswith("AUTO-STOP:")]
    assert len(out) == 1 and f"step {stop_at}:" in out[0] and "actor/ppo_kl" in out[0]


def test_without_auto_stop_the_run_goes_on(quiet_fit):
    trainer = StoppingFit([(2e-3, 0.5)] * 6, total=6, auto_stop=False)
    trainer.fit(agent_loop_manager=None)
    assert trainer.logged == list(range(1, 7)) and trainer.saved == [5, 6]


# ---------------------------------------------------------------- dev selection


class ValidateStub:
    def __init__(self, ckpt_root, step, val):
        self.config = OmegaConf.create({"trainer": {"default_local_dir": str(ckpt_root)}})
        self.global_steps, self.val = step, val

    def _validate(self):
        return dict(self.val)


class Selecting(DevSelection, ValidateStub):
    pass


def test_selection_is_written_with_the_dev_scores_just_evaluated(tmp_path, monkeypatch):
    """verl saved step 40's checkpoint and evaluated dev; its scores are not yet in
    metrics.jsonl (verl logs them at the end of the step). selection.json names step 40."""
    import json

    key = "val-core/arggym/acc/mean@8"
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text("".join(json.dumps({"step": s, "data": {key: v}}) + "\n"
                               for s, v in [(0, 0.29), (20, 0.41), (30, 0.40)]))
    for s in (20, 40):
        (tmp_path / "checkpoints" / f"global_step_{s}").mkdir(parents=True)
    monkeypatch.setenv("VERL_FILE_LOGGER_PATH", str(metrics))
    val = Selecting(tmp_path / "checkpoints", 40, {key: 0.48})._validate()
    assert val == {key: 0.48}
    sel = json.loads((tmp_path / "selection.json").read_text())
    assert (sel["best_step"], sel["best_metric"], sel["base_metric"]) == (40, 0.48, 0.29)
    assert sel["dev"] == {"0": 0.29, "20": 0.41, "30": 0.40, "40": 0.48}
