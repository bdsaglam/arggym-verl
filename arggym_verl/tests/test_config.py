"""The GSM8K config composes, follows verl's LoRA recipe, and keeps the task's own values.

The recipe is examples/tuning/lora/run_qwen3_8b_fsdp.sh; its defaults are read
from the script itself, so a change on either side fails here.
"""
import math
import re

import pytest
import yaml

from conftest import ROOT

RECIPE = ROOT / "examples/tuning/lora/run_qwen3_8b_fsdp.sh"


@pytest.fixture(scope="module")
def cfg():
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "arggym_verl/config"), version_base=None):
        return compose(config_name="gsm8k-hard-qwen3-0.6b",
                       overrides=["hydra.searchpath=[pkg://verl.trainer.config]"])


@pytest.fixture(scope="module")
def recipe():
    """The recipe's `name=${ENV:-default}` values and its `key=value` overrides."""
    text = RECIPE.read_text()
    values = {k: v for k, v in re.findall(r"^(\w+)=\$\{\w+:-([^}]*)\}", text, re.M)}
    overrides = dict(re.findall(r"^\s+([\w.]+)=(\S+)$", text, re.M))
    return values, overrides


def agent_loops(cfg):
    return {a["name"]: a for a in yaml.safe_load(open(cfg.actor_rollout_ref.rollout.agent.agent_loop_config_path))}


def test_follows_the_lora_recipe(cfg, recipe):
    v, o = recipe
    m, a, r, alg = (cfg.actor_rollout_ref.model, cfg.actor_rollout_ref.actor,
                    cfg.actor_rollout_ref.rollout, cfg.algorithm)
    assert (m.lora_rank, m.lora_alpha) == (int(v["lora_rank"]), int(v["lora_alpha"])) == (64, 32)
    assert m.target_modules == "all-linear"
    assert cfg.data.train_batch_size == int(v["train_batch_size"]) == 64
    assert a.ppo_mini_batch_size == int(v["ppo_mini_batch_size"]) == 32
    assert a.ppo_epochs == 1
    assert r.n == int(v["rollout_n"]) == 5
    assert a.use_kl_loss and o["actor_rollout_ref.actor.use_kl_loss"] == "True"
    assert a.kl_loss_coef == float(v["kl_loss_coef"]) == 0.001
    assert a.kl_loss_type == o["actor_rollout_ref.actor.kl_loss_type"] == "low_var_kl"
    assert alg.use_kl_in_reward is False and o["algorithm.use_kl_in_reward"] == "False"
    assert a.entropy_coeff == float(v["entropy_coeff"]) == 0
    assert alg.adv_estimator == o["algorithm.adv_estimator"] == "grpo"
    # verl's defaults, which the recipe does not override.
    assert alg.norm_adv_by_std_in_grpo is True
    assert (a.clip_ratio_low, a.clip_ratio_high) == (0.2, 0.2)
    assert a.loss_agg_mode == "token-mean"
    assert a.optim.lr_scheduler_type == "constant"
    assert a.optim.lr_warmup_steps == -1 and a.optim.lr_warmup_steps_ratio == 0.0
    assert a.optim.weight_decay == 0.01
    assert a.optim.clip_grad == 1.0
    assert r.load_format == o["actor_rollout_ref.rollout.load_format"] == "safetensors"
    assert r.layered_summon and o["actor_rollout_ref.rollout.layered_summon"] == "True"


def test_lr_is_the_lora_guides_not_the_recipes(cfg, recipe):
    """The one hyperparameter departure: verl's LoRA guide over the recipe's 3e-6."""
    guide = (ROOT / "docs/advance/ppo_lora.rst").read_text()
    assert "increase the value of learning rate by an order of magnitude" in guide
    assert "actor_rollout_ref.actor.optim.lr=3e-5" in guide
    assert float(recipe[0]["actor_lr"]) == 3e-6
    assert cfg.actor_rollout_ref.actor.optim.lr == 3e-5


def test_batch_arithmetic_pads_nothing_on_two_gpus(cfg):
    a, r, t = cfg.actor_rollout_ref.actor, cfg.actor_rollout_ref.rollout, cfg.trainer
    ranks = t.n_gpus_per_node * t.nnodes
    assert ranks == 2
    completions = cfg.data.train_batch_size * r.n
    # trainer_base._get_required_batch_multiple: verl pads the batch up to a multiple of this.
    multiple = math.lcm(ranks, a.ppo_mini_batch_size * r.n)
    assert completions % multiple == 0
    assert completions // (a.ppo_mini_batch_size * r.n) == 2  # optimizer steps per batch
    assert completions % math.lcm(3, a.ppo_mini_batch_size * r.n) != 0  # 3 GPUs would pad
    assert ranks % r.tensor_model_parallel_size == 0


def test_async_mode_with_adapter_serving(cfg):
    m, r = cfg.actor_rollout_ref.model, cfg.actor_rollout_ref.rollout
    # Only the hybrid engine's (naive) weight sync sends vLLM the adapter.
    assert cfg.actor_rollout_ref.hybrid_engine
    assert cfg.trainer.v1.trainer_mode == "colocate_async"
    assert cfg.trainer.v1.colocate_async.num_warmup_batches == 1
    assert r.checkpoint_engine.backend == "naive"
    assert not m.lora.merge
    assert r.dtype == "bfloat16"


def test_rollout_correction_is_token_is_capped_at_2(cfg):
    """The documented departure: present, token-level, cap 2, decoupled (pi_old from the trainer)."""
    rc = cfg.algorithm.rollout_correction
    assert rc is not None
    assert (rc.rollout_is, rc.rollout_is_threshold, rc.bypass_mode) == ("token", 2.0, False)
    assert rc.rollout_rs is None
    assert cfg.actor_rollout_ref.rollout.calculate_log_probs
    assert cfg.actor_rollout_ref.actor.fsdp_config.model_dtype == "fp32"


def test_lengths(cfg):
    r = cfg.actor_rollout_ref.rollout
    loops = agent_loops(cfg)
    assert loops["train"]["sampling"] == {"max_tokens": 8192}
    assert "max_tokens" not in loops["dev"]["sampling"]
    assert cfg.data.max_response_length == 32768
    assert cfg.data.max_prompt_length + cfg.data.max_response_length <= r.max_model_len
    # A micro-batch holds the longest training sequence, in training and in both log-prob passes.
    longest = cfg.data.max_prompt_length + 8192
    assert cfg.actor_rollout_ref.actor.ppo_max_token_len_per_gpu >= longest
    assert r.log_prob_max_token_len_per_gpu >= longest
    assert cfg.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu >= longest


def test_training_cap_reaches_vllm():
    """The agent loop's cap overrides what verl passes, and refuses one above response_length."""
    import asyncio
    from unittest import mock

    from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop

    from arggym_verl.agent_loop import SamplingAgentLoop

    loop = SamplingAgentLoop.__new__(SamplingAgentLoop)
    loop.sampling = {"max_tokens": 8192}
    seen = {}

    async def run(self, sampling_params, **kwargs):
        seen.update(sampling_params)

    with mock.patch.object(SingleTurnAgentLoop, "run", run):
        asyncio.run(loop.run({"temperature": 1.0, "max_tokens": 32768}))
    assert seen == {"temperature": 1.0, "max_tokens": 8192}

    with mock.patch.object(SingleTurnAgentLoop, "__init__", lambda self, *a, **k: None):
        too_long = SamplingAgentLoop.__new__(SamplingAgentLoop)
        too_long.response_length = 4096
        with pytest.raises(ValueError, match="above rollout.response_length"):
            SamplingAgentLoop.__init__(too_long, sampling={"max_tokens": 8192})


def test_task_sampling_and_schedule(cfg):
    r, t = cfg.actor_rollout_ref.rollout, cfg.trainer
    m = cfg.actor_rollout_ref.model
    assert m.path.endswith("c1899de289a04d12100db370d81485cdf75e47ca")
    assert (r.temperature, r.top_p, r.top_k) == (1.0, 1.0, -1)
    v = r.val_kwargs
    assert (v.temperature, v.top_p, v.top_k, v.n) == (0.6, 0.95, 20, 4)
    assert dict(cfg.data.apply_chat_template_kwargs) == {"enable_thinking": True}
    assert t.val_before_train and t.test_freq == 10 and t.save_freq == 10
    assert t.max_actor_ckpt_to_keep is None


def test_validates_with_kl_on(cfg):
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config

    assert need_reference_policy(cfg)
    validate_config(config=cfg, use_reference_policy=True, use_critic=need_critic(cfg))


# ArgGYM on Qwen3.5-4B, three GPUs: the GSM8K config's LoRA and IS on experiment 003's task.


@pytest.fixture(scope="module")
def arggym():
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "arggym_verl/config"), version_base=None):
        return compose(config_name="arggym-qwen3.5-4b",
                       overrides=["hydra.searchpath=[pkg://verl.trainer.config]"])


@pytest.fixture(scope="module")
def exp003():
    """003's settings, read by ArgGYM's rl.config from its run and eval-model configs."""
    from conftest import fixture

    return fixture("exp003.json")


def test_arggym_keeps_the_gsm8k_lora_loss_and_is(cfg, arggym):
    """The recipe's LoRA, loss, advantage and the token IS, as the GSM8K control."""
    a, g = arggym.actor_rollout_ref, cfg.actor_rollout_ref
    assert (a.model.lora_rank, a.model.lora_alpha, a.model.target_modules) == (64, 32, "all-linear")
    assert not a.model.lora.merge
    for key in ("use_kl_loss", "kl_loss_type", "clip_ratio_low", "clip_ratio_high", "loss_agg_mode",
                "entropy_coeff", "ppo_epochs"):
        assert a.actor[key] == g.actor[key], key
    for key in ("optimizer", "weight_decay", "clip_grad", "betas"):
        assert a.actor.optim[key] == g.actor.optim[key], key
    assert arggym.algorithm == cfg.algorithm
    rc = arggym.algorithm.rollout_correction
    assert (rc.rollout_is, rc.rollout_is_threshold, rc.bypass_mode) == ("token", 2.0, False)
    assert a.rollout.calculate_log_probs and a.rollout.logprobs_mode == "processed_logprobs"
    assert (a.rollout.load_format, a.rollout.layered_summon) == ("safetensors", True)
    assert arggym.trainer.v1.trainer_mode == "colocate_async"


def test_arggym_lr_is_cosine_from_1e5_to_1e6_after_3_warmup_steps(arggym):
    """verl's own scheduler on the config's optimizer block, stepped once a training step as
    engine_workers.update_actor steps it, with total_training_steps as trainer_base sets it."""
    from types import SimpleNamespace

    import torch
    from verl.utils.config import omega_conf_to_dataclass
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    o = arggym.actor_rollout_ref.actor.optim
    assert (o.lr, o.lr_scheduler_type, o.min_lr_ratio, o.lr_warmup_steps) == (1e-5, "cosine", 0.1, 3)
    optim = omega_conf_to_dataclass(o)
    optim.total_training_steps = arggym.trainer.total_training_steps  # trainer_base, parameter_sync_step 1
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=optim.lr)
    sched = FSDPEngine._build_lr_scheduler(SimpleNamespace(optimizer_config=optim, rank=1), opt)
    lrs = []
    for _ in range(arggym.trainer.total_training_steps):
        lrs.append(opt.param_groups[0]["lr"])  # both mini-batches of the step
        opt.step()
        sched.step()
    assert lrs[:3] == pytest.approx([1e-5 / 3, 2e-5 / 3, 1e-5])  # step 1 trains, at 1/3
    assert lrs[3] == pytest.approx(1e-5, rel=1e-3)  # the cosine starts
    assert lrs[50] == pytest.approx(0.55e-5, rel=0.05)  # halfway down the cosine
    assert lrs[-1] == pytest.approx(1e-6) and min(lrs) == pytest.approx(1e-6)
    assert all(a >= b for a, b in zip(lrs[2:], lrs[3:]))


def test_arggym_kl_is_logged_not_trained_on(arggym):
    """KL coefficient 0 with the KL loss on: verl still runs the reference pass (which the
    served-policy check reads) and logs actor/kl_loss, and the gradient does not depend
    on the reference. verl's own ppo_loss on a two-sequence batch."""
    import torch
    from tensordict import TensorDict
    from verl.trainer.ppo.utils import need_reference_policy
    from verl.utils import tensordict_utils as tu
    from verl.utils.config import omega_conf_to_dataclass
    from verl.workers.utils.losses import ppo_loss

    a = arggym.actor_rollout_ref.actor
    assert a.use_kl_loss and a.kl_loss_coef == 0.0
    assert need_reference_policy(arggym)
    actor = omega_conf_to_dataclass(a)
    B, P, R = 2, 3, 4

    def loss(ref):
        logp = (-torch.rand(B * (P + R), generator=torch.Generator().manual_seed(1))).requires_grad_()
        data = TensorDict({"prompts": torch.ones(B, P, dtype=torch.long),
                           "responses": torch.ones(B, R, dtype=torch.long),
                           "attention_mask": torch.ones(B, P + R, dtype=torch.long),
                           "response_mask": torch.ones(B, R), "old_log_probs": -torch.ones(B, R),
                           "advantages": torch.tensor([[1.0] * R, [-1.0] * R]), "ref_log_prob": ref},
                          batch_size=[B])
        tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=B * R, global_batch_size=B)
        out, metrics = ppo_loss(actor, {"log_probs": logp}, data)
        out.backward()
        return out, logp.grad, metrics

    near, g_near, _ = loss(-torch.ones(B, R))
    far, g_far, metrics = loss(-3 * torch.ones(B, R))
    assert torch.equal(g_near, g_far) and g_far.abs().sum() > 0
    assert near.item() == far.item()
    assert metrics["kl_coef"] == 0.0 and metrics["kl_loss"].values[0] > 0.1


def test_arggym_batch_arithmetic_pads_nothing_on_three_gpus(arggym):
    """18 prompts x 16 = 288 completions; mini-batch 9 x 16 = 144; lcm(3, 144) = 144 divides 288."""
    a, r, t = arggym.actor_rollout_ref.actor, arggym.actor_rollout_ref.rollout, arggym.trainer
    ranks = t.n_gpus_per_node * t.nnodes
    assert ranks == 3 and r.n == 16 and r.tensor_model_parallel_size == 1
    assert (arggym.data.train_batch_size, a.ppo_mini_batch_size) == (18, 9)
    completions, mini = arggym.data.train_batch_size * r.n, a.ppo_mini_batch_size * r.n
    assert (completions, mini, math.lcm(ranks, mini)) == (288, 144, 144)
    assert completions % math.lcm(ranks, mini) == 0  # trainer_base._get_required_batch_multiple
    assert completions // mini == 2  # optimizer steps per batch, as the recipe's
    # 96 a replica: 48 decode at once, the other half queue.
    assert (completions // ranks, r.max_num_seqs) == (96, 48)
    # The old 16-prompt batch would pad: lcm(3, 8 x 16) = 384 does not divide 256.
    assert (16 * r.n) % math.lcm(ranks, 8 * r.n) != 0


def test_arggym_task_is_003s(arggym, exp003):
    """Model revision, thinking, the training cap, dev sampling and cap, the step count."""
    r, g = arggym.actor_rollout_ref.rollout, exp003["grpo"]
    assert arggym.actor_rollout_ref.model.path.endswith(exp003["model_revision"])
    assert dict(arggym.data.apply_chat_template_kwargs) == exp003["chat_template_kwargs"] \
        == {"enable_thinking": True}
    loops = agent_loops(arggym)
    assert loops["train"]["sampling"]["max_tokens"] == g["max_completion_length"] == 40960
    assert not g["mask_truncated_completions"]
    # Dev: the eval config's sampling, cap and presence penalty.
    dev = exp003["dev_sampling"]
    v = r.val_kwargs
    assert (v.temperature, v.top_p, v.top_k) == (dev["temperature"], dev["top_p"], dev["top_k"])
    assert (dev["min_p"], dev["repetition_penalty"]) == (0.0, 1.0)  # vLLM's defaults
    assert loops["dev"]["sampling"] == {"presence_penalty": dev["presence_penalty"]} \
        == {"presence_penalty": 1.5}
    assert arggym.data.max_response_length == dev["max_completion_length"] == 65536
    assert r.max_model_len == exp003["vllm_max_model_len"] == 73728
    assert arggym.data.max_prompt_length + arggym.data.max_response_length <= r.max_model_len
    assert arggym.trainer.total_training_steps == g["max_steps"] == 100


def test_arggym_schedule(arggym):
    """Dev at step 0 and every 20 steps, 8 samples a row; a checkpoint every 5 steps, so
    every dev step has one; the auto-stop's thresholds."""
    t, v = arggym.trainer, arggym.actor_rollout_ref.rollout.val_kwargs
    assert t.val_before_train and (t.test_freq, t.save_freq, v.n) == (20, 5, 8)
    assert t.test_freq % t.save_freq == 0 and t.max_actor_ckpt_to_keep is None
    assert dict(t.auto_stop) == {"ppo_kl_max": 1e-3, "ppo_kl_steps": 2, "rollout_kl_factor": 2.0,
                                 "baseline_steps": 5}


def sampling_sent(cfg, validate: bool):
    """What reaches vLLM for one row: verl's AgentLoopWorker.generate_sequences builds the
    sampling params from rollout.* (train) or rollout.val_kwargs.* (dev), and the row's agent
    loop, named by its `agent_name` as data.py writes it, merges its `sampling` over them."""
    import asyncio
    from types import SimpleNamespace
    from unittest import mock

    import numpy as np
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
    from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
    from verl.protocol import DataProto

    from arggym_verl.agent_loop import SamplingAgentLoop

    loops = agent_loops(cfg)
    seen = {}

    async def run_loop(sampling_params, trajectory, *, agent_name, trace=True, **kwargs):
        loop = SamplingAgentLoop.__new__(SamplingAgentLoop)
        loop.sampling = dict(loops[agent_name]["sampling"])
        return await loop.run(sampling_params)

    async def single_turn_run(self, sampling_params, **kwargs):
        seen.update(sampling_params)

    worker = SimpleNamespace(rollout_config=cfg.actor_rollout_ref.rollout, _run_agent_loop=run_loop,
                             _postprocess=lambda outputs, **kw: outputs)
    split = "dev" if validate else "train"
    batch = DataProto.from_dict(non_tensors={"agent_name": np.array([split], dtype=object)},
                                meta_info={"validate": validate, "global_steps": 1})
    with mock.patch.object(SingleTurnAgentLoop, "run", single_turn_run):
        asyncio.run(AgentLoopWorker.generate_sequences(worker, batch))
    return seen


def test_arggym_trains_on_plain_sampling_and_evaluates_dev_with_the_cards(arggym):
    train = sampling_sent(arggym, validate=False)
    assert {k: train[k] for k in ("temperature", "top_p", "top_k", "presence_penalty", "max_tokens")} \
        == {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "presence_penalty": 0.0, "max_tokens": 40960}
    assert train["repetition_penalty"] == 1.0 and train["logprobs"]
    dev = sampling_sent(arggym, validate=True)
    assert {k: dev[k] for k in ("temperature", "top_p", "top_k", "presence_penalty")} \
        == {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}
    assert "max_tokens" not in dev  # dev runs to rollout.response_length, 65,536


def test_arggym_runs_on_its_own_batch(arggym):
    """No batch generated ahead: the curriculum draws each step after the previous one's outcomes."""
    from arggym_verl.curriculum import check_setup

    assert arggym.trainer.v1.colocate_async.num_warmup_batches == 0
    check_setup(arggym, arggym.trainer.v1.trainer_mode, 1)


def test_arggym_memory_layout(arggym):
    a, r = arggym.actor_rollout_ref, arggym.actor_rollout_ref.rollout
    longest = arggym.data.max_prompt_length + agent_loops(arggym)["train"]["sampling"]["max_tokens"]
    for limit in (a.actor.ppo_max_token_len_per_gpu, r.log_prob_max_token_len_per_gpu,
                  a.ref.log_prob_max_token_len_per_gpu):
        assert limit >= longest == 49152
    assert a.actor.use_dynamic_bsz and a.model.use_remove_padding
    assert a.model.enable_gradient_checkpointing
    assert a.actor.entropy_from_logits_with_chunking
    assert not a.actor.fsdp_config.param_offload and not a.actor.fsdp_config.optimizer_offload
    assert r.gpu_memory_utilization == 0.8 and r.max_num_seqs == 48
    assert r.free_cache_engine and not r.enable_prefix_caching
    assert r.engine_kwargs.vllm == {"language_model_only": True}
    assert a.actor.checkpoint.save_lora_only


def test_arggym_validates_with_kl_on(arggym):
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config

    assert need_reference_policy(arggym)
    validate_config(config=arggym, use_reference_policy=True, use_critic=need_critic(arggym))


def test_lora_targets_are_the_text_linears_and_vllm_serves_each(arggym, exp003):
    """The adapter verl builds on Qwen3.5-4B (meta device, no weights) covers every Linear of
    the text model, as 003's run.json lists them, and vLLM's Qwen3.5 has a LoRA-capable layer
    for each: a packed one (packed_modules_mapping) or a parallel linear of that name.
    The served-policy check confirms at runtime that vLLM applies them."""
    import collections
    import inspect

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig

    from conftest import QWEN35_4B
    from verl.utils.model import get_hf_auto_model_class

    m = arggym.actor_rollout_ref.model
    hf_config = AutoConfig.from_pretrained(str(QWEN35_4B))
    with torch.device("meta"):
        model = get_hf_auto_model_class(hf_config=hf_config).from_config(hf_config)
    assert type(model).__name__ == "Qwen3_5ForConditionalGeneration"
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=m.lora_rank,
                                             lora_alpha=m.lora_alpha, target_modules=m.target_modules,
                                             exclude_modules=m.exclude_modules, bias="none"))
    adapted = [n for n, mod in model.named_modules() if getattr(mod, "lora_A", None)]
    assert all(".language_model.layers." in n for n in adapted)
    kinds = collections.Counter(n.rsplit(".", 1)[-1] for n in adapted)
    assert kinds == {"q_proj": 8, "k_proj": 8, "v_proj": 8, "o_proj": 8,
                     "in_proj_qkv": 24, "in_proj_z": 24, "in_proj_a": 24, "in_proj_b": 24,
                     "out_proj": 24, "gate_proj": 32, "up_proj": 32, "down_proj": 32}

    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
    from vllm.model_executor.models import qwen2_moe, qwen3_5, qwen3_next

    packed = qwen3_5.Qwen3_5ForConditionalGeneration.packed_modules_mapping
    source = "".join(inspect.getsource(mod) for mod in (qwen3_5, qwen3_next, qwen2_moe,
                                                        qwen_gdn_linear_attn))
    linear = set(re.findall(r"self\.(\w+)\s*=\s*(?:Row|Column)ParallelLinear\(", source))
    assert set(kinds) <= {s for subs in packed.values() for s in subs} | linear
