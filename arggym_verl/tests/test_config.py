"""The verl config composes, its batch arithmetic holds, and it carries the TRL run's decisions."""
import pytest
import yaml

from conftest import ROOT, TRL_GSM8K


@pytest.fixture(scope="module")
def cfg():
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "arggym_verl/config"), version_base=None):
        return compose(config_name="gsm8k-hard-qwen3-0.6b",
                       overrides=["hydra.searchpath=[pkg://verl.trainer.config]"])


@pytest.fixture(scope="module")
def trl():
    from rl import config as rl_config

    return rl_config.load(str(TRL_GSM8K)), rl_config.dev_sampling(rl_config.load(str(TRL_GSM8K)))


def agent_loops(cfg):
    return {a["name"]: a for a in yaml.safe_load(open(cfg.actor_rollout_ref.rollout.agent.agent_loop_config_path))}


def test_batch_arithmetic(cfg, trl):
    trl_cfg, _ = trl
    g, a, r, t = trl_cfg["grpo"], cfg.actor_rollout_ref.actor, cfg.actor_rollout_ref.rollout, cfg.trainer
    # One optimizer step per 16 prompts x 8 completions, as in TRL.
    assert cfg.data.train_batch_size == trl_cfg["prompts_per_step"]
    assert r.n == g["num_generations"]
    assert cfg.data.train_batch_size // a.ppo_mini_batch_size * a.ppo_epochs == g["num_iterations"]
    # The batch splits evenly over the ranks, so verl pads nothing in.
    assert cfg.data.train_batch_size * r.n % (t.n_gpus_per_node * t.nnodes) == 0
    assert a.ppo_mini_batch_size * r.n % (t.n_gpus_per_node * t.nnodes) == 0
    assert (t.n_gpus_per_node * t.nnodes) % r.tensor_model_parallel_size == 0


def test_async_mode_with_adapter_serving(cfg):
    m, r = cfg.actor_rollout_ref.model, cfg.actor_rollout_ref.rollout
    # Only the hybrid engine's (naive) weight sync sends vLLM the adapter.
    assert cfg.actor_rollout_ref.hybrid_engine
    assert cfg.trainer.v1.trainer_mode == "colocate_async"
    assert r.checkpoint_engine.backend == "naive"
    assert not m.lora.merge
    # verl's LoRA recipe (examples/tuning/lora/run_qwen3_8b_fsdp.sh).
    assert r.load_format == "safetensors" and r.layered_summon
    assert r.dtype == "bfloat16"


def test_ppo_ratio_uses_the_trainers_old_log_probs(cfg):
    """No phantom clipping: vLLM's log-probs never sit in the PPO ratio's denominator."""
    rc = cfg.algorithm.rollout_correction
    assert rc.bypass_mode is False
    assert cfg.actor_rollout_ref.actor.ppo_epochs == 1
    assert cfg.actor_rollout_ref.rollout.calculate_log_probs
    assert cfg.actor_rollout_ref.actor.fsdp_config.model_dtype == "fp32"


def test_lengths(cfg, trl):
    trl_cfg, dev = trl
    r = cfg.actor_rollout_ref.rollout
    loops = agent_loops(cfg)
    assert loops["train"]["sampling"]["max_tokens"] == trl_cfg["grpo"]["max_completion_length"]
    assert "max_tokens" not in loops["dev"]["sampling"]
    assert cfg.data.max_response_length == dev["max_completion_length"] == 32768
    assert cfg.data.max_prompt_length + cfg.data.max_response_length <= r.max_model_len
    assert r.max_model_len == trl_cfg["vllm"]["max_model_len"]
    # A micro-batch holds the longest training sequence.
    assert cfg.actor_rollout_ref.actor.ppo_max_token_len_per_gpu >= (
        cfg.data.max_prompt_length + trl_cfg["grpo"]["max_completion_length"])


def test_algorithm_matches_trl(cfg, trl):
    g = trl[0]["grpo"]
    a, alg = cfg.actor_rollout_ref.actor, cfg.algorithm
    assert alg.adv_estimator == "grpo"
    assert alg.norm_adv_by_std_in_grpo is False and g["scale_rewards"] == "none"
    assert a.loss_agg_mode == "token-mean" and g["loss_type"] == "dapo"
    assert (a.clip_ratio_low, a.clip_ratio_high) == (g["epsilon"], g["epsilon_high"])
    assert not a.use_kl_loss and not alg.use_kl_in_reward and g["beta"] == 0.0
    assert a.optim.lr == g["learning_rate"]
    assert a.optim.lr_warmup_steps == g["warmup_steps"]
    assert a.optim.lr_scheduler_type == "constant"
    assert a.optim.weight_decay == g["weight_decay"]
    assert a.optim.clip_grad == g["max_grad_norm"]
    rc = alg.rollout_correction
    assert (rc.rollout_is, rc.rollout_is_threshold, rc.bypass_mode) == (
        "token", g["vllm_importance_sampling_clip_max"], False)
    assert g["vllm_importance_sampling_mode"] == "token_truncate"


def test_lora_matches_trl(cfg, trl):
    lora, m = trl[0]["lora"], cfg.actor_rollout_ref.model
    assert (m.lora_rank, m.lora_alpha, m.target_modules) == (lora["r"], lora["alpha"],
                                                             lora["target_modules"])
    assert m.path.endswith(trl[0]["model"]["revision"])


def test_sampling_matches_trl(cfg, trl):
    g, dev = trl[0]["grpo"], trl[1]
    r = cfg.actor_rollout_ref.rollout
    assert (r.temperature, r.top_p) == (g["temperature"], g["top_p"])
    assert r.top_k == -1 and g["top_k"] == 0  # both mean "off"
    v = r.val_kwargs
    assert (v.temperature, v.top_p, v.top_k) == (dev["temperature"], dev["top_p"], dev["top_k"])
    assert dev["min_p"] == 0.0 and dev["presence_penalty"] == 0.0
    assert v.n == g["num_generations_eval"] and v.do_sample
    assert dict(cfg.data.apply_chat_template_kwargs) == {"enable_thinking": True}


def test_schedule(cfg, trl):
    g, t = trl[0]["grpo"], cfg.trainer
    assert t.total_training_steps == g["max_steps"]
    assert t.test_freq == g["eval_steps"] and t.val_before_train == g["eval_on_start"]
    assert t.save_freq == g["save_steps"]
