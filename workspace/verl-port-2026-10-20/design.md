# Design: ArgGYM GRPO on verl

verl v0.9.1 (tag `v0.9.1`, `1876b06d`, 2026-09-20; the latest release on
2026-10-20). Paths below are in this repo, which is that tag plus `arggym_verl/`.

## The three features, and where verl provides them

| Feature | Entry point | Status in v0.9.1 |
|---|---|---|
| GRPO | `algorithm.adv_estimator=grpo` (`verl/trainer/ppo/core_algos.py`, `docs/algo/grpo.md`) | stable |
| Async RL | `verl.trainer.main_ppo` with `trainer.use_v1=true` (the default, `verl/trainer/config/ppo_trainer.yaml:232`) and `trainer.v1.trainer_mode=separate_async` (`verl/trainer/ppo/v1/trainer_separate_async.py`, `docs/advance/v1_async_trainer.md`) | main library; only the hybrid step switching is marked experimental, and it is off here |
| Token-in/token-out | the agent loop: `SingleTurnAgentLoop` (`verl/experimental/agent_loop/single_turn_agent_loop.py`) builds prompt ids, calls the vLLM server's `generate(prompt_ids, ...)` (`verl/workers/rollout/vllm_rollout/vllm_async_server.py:559`), keeps the returned ids and log-probs | default rollout path (`rollout.mode=async`, `agent.default_agent_loop=single_turn_agent`), but the module lives under `verl/experimental/` and `docs/advance/agent_loop.rst` still says "status: alpha, the API may change" |

So all three exist; the agent loop carries an alpha label though every rollout
in v0.9.1 goes through it. The older async paths, `verl/experimental/fully_async_policy`
and `one_step_off_policy`, are experimental and superseded: the V1 doc presents
`separate_async` with `hybrid_engine=False` as the replacement for "v0
fully-async semantics".

## Repo layout: a clone of the release tag

verl's install guide (`docs/start/install.rst`, "Install with uv") runs
everything from the verl repo root on its committed `uv.lock`
(`uv run --frozen --extra vllm --extra fsdp`), and custom tasks plug in through
config: a reward manager by file path (`reward.reward_manager.source=importlib`,
`docs/advance/reward_loop.rst`), agent loops by a YAML of `_target_`s
(`rollout.agent.agent_loop_config_path`, `verl/experimental/agent_loop/agent_loop.py:534`).
So the repo is the tag with our package beside verl's, and no verl file is
edited but one: the `hybrid_engine=False` branch of
`PPOTrainerSeparateAsync._setup` passed `flush=True` to `logger.info` and
raised `TypeError` before the first step (commit `f9b9654f`, a one-line
upstream bug fix, its own commit). `scripts/arggym/setup.sh` syncs verl's lock as committed (`torch
2.11.0+cu130`, `vllm 0.24.0`, `transformers 5.9.0`, `peft 0.19.1`, `ray
2.55.1`), then adds `python-argumentation` (ArgGYM's scorer dependency) and
`trackio` with `uv pip`. Re-locking with them as a new extra was tried and
dropped: it re-resolved verl's wheelhouse packages, and flash-attn came back
as a torch-2.13 build that fails to load against torch 2.11 (undefined symbol
`c10::ValueError`). The lock's own flash-attn URL is dead (404), so the setup
installs the wheelhouse's torch-2.11 build of flash-attn 2.8.3 instead.

ArgGYM code is imported, not copied: the launcher puts `$ARGGYM_ROOT` (the
ArgGYM checkout) on `PYTHONPATH`, and `arggym_verl` calls `rl.gsm8k`,
`rl.pool`, `rl.reward` (hence `evals.score.score_one`, the eval's scorer)
directly. The run records the checkout's commit and branch.

## What arggym_verl adds

- `data.py`: writes verl's parquet from the TRL config with ArgGYM's own
  selection (`rl.gsm8k.load_splits`, `rl.pool.load_split`) and prompt messages.
  GSM8K hard: 1,322 training questions, 200 dev.
- `reward.py`: a reward manager. verl's default one hands the reward function
  decoded text only; TRL's truncation rule needs the ids (last token neither
  EOS nor pad), so this manager reads them, decodes them as TRL does
  (`skip_special_tokens=True`) and calls ArgGYM's scorer.
- `agent_loop.py`: `SingleTurnAgentLoop` with per-split sampling overrides
  (`max_tokens`, `presence_penalty`), selected per row by `agent_name`. verl has
  one response length for training and validation and no presence penalty in
  its sampling config.
- `report.py`: dev curve, best checkpoint, and per-version training truncation
  and no-signal groups, which verl's step metrics lack.
- `config/*.yaml`: one Hydra config per task over verl's `ppo_trainer`.

## Decisions, and each one's source

| Decision | Setting | Source |
|---|---|---|
| GPU split as TRL: vLLM on GPU 0, training on 1-2 | `hybrid_engine=false`, `rollout.n_gpus_per_node=1`, `trainer.n_gpus_per_node=2` | `docs/advance/v1_async_trainer.md` ("Separate Async without Hybrid Replicas") |
| One optimizer step per 16 prompts x 8 | `train_batch_size=16`, `ppo_mini_batch_size=16`, `parameter_sync_step=1`, `rollout.n=8` | invariant `train_batch_size = parameter_sync_step x ppo_mini_batch_size`, `trainer_separate_async.py:56` |
| At most one step off-policy, nothing dropped | `sampler.max_off_policy_threshold=2`, `max_off_policy_strategy=wait` | span rule `global_steps - submitted + 1`, `verl/trainer/ppo/v1/replay_buffer.py:528,553`; `drop` would discard late groups, which are the long completions |
| Off-policy correction | `rollout_correction: {rollout_is: token, rollout_is_threshold: 2.0, bypass_mode: false}` (decoupled PPO: trainer recomputes pi_old; weight pi_old/pi_rollout truncated at 2) | `docs/algo/rollout_corr.md` preset `decoupled_token_is`; the analogue of TRL's `token_truncate` cap 2 |
| GRPO without std scaling, DAPO token mean, clip 0.2/0.28, no KL | `norm_adv_by_std_in_grpo=false`, `loss_agg_mode=token-mean`, `clip_ratio_low/high`, `use_kl_loss=false` | `docs/algo/grpo.md` (Dr. GRPO section), `docs/algo/dapo.md` |
| LoRA r32 / alpha 32 / all-linear, LR 2e-5 constant after 5 warmup steps, wd 0, grad clip 1 | `model.lora_rank/lora_alpha/target_modules`, `optim.*` | `docs/advance/ppo_lora.rst`; scheduler `verl/workers/engine/fsdp/transformer_impl.py:521` |
| Training sampling T 1, top-p 1, top-k off; 8,192 cap | `rollout.temperature/top_p/top_k`; cap via the `train` agent loop | TRL config |
| Dev: card sampling, 4 samples, 32,768 cap, at step 0 and every 10 steps | `rollout.val_kwargs`, `data.max_response_length=32768`, `trainer.test_freq=10`, `val_before_train=true` | `evals/conf/model/hf-qwen3-0.6b.yaml` |
| Sampled log-probs | `rollout.logprobs_mode=processed_logprobs`, `calculate_log_probs=true`; `VLLM_USE_FLASHINFER_SAMPLER=0` | as the TRL server (`rl/launch.py`) |
| Logging | `trainer.logger=[console, file, trackio]`; dev generations to `outputs/<run>/dev` | `verl/utils/tracking.py:88` (trackio adapter) |

## Where verl forces a difference from the TRL run

1. **LoRA weights reach vLLM merged, in BF16.** Async training syncs the
   standalone server through a checkpoint engine (`nccl`), which calls
   `get_per_tensor_param()` with its defaults (`verl/workers/engine_workers.py:782`).
   With `lora.merge=false` that yields the base weights without the adapter
   (`collect_lora_params(base_sync_done=False)` skips every `lora_` tensor,
   `verl/utils/fsdp_utils.py:715`), and the receiving side loads them as plain
   weights (`verl/checkpoint_engine/base.py:355`): the server would keep
   sampling the base model while the adapter trains, with no error. Only the
   colocated (naive) sync knows adapters. So `lora.merge=true`: the delta is
   merged in FP32 and sent as BF16 (`_merged_lora_per_tensor_param`,
   `transformer_impl.py:1063`). This is the TRL pipeline's `merged` mode, not
   the `adapter` mode of the reference run, and it has the rounding loss the
   TRL review measured (at LR 1e-5 on Qwen3.5-4B, 23% of the update reached the
   server after 40 steps). At 2e-5 on Qwen3-0.6B the share is not measured.
   The alternatives are full fine-tuning (no adapter; BF16 rounding of a full
   weight update is the norm for every RL pipeline) or `colocate_async`, which
   keeps adapter sync but does not overlap rollout with training
   (`trainer_colocate_async.py`: generation is paused while the trainer runs).
2. **Rollouts are up to one step stale.** TRL is synchronous. Here a group may
   come from the previous weights and a long completion may span two weight
   versions (partial rollout). The truncated IS weight corrects the gap; watch
   `training/off_policy/*` and `rollout_corr/*`.
3. **Loss denominator.** TRL divides the DAPO loss by the tokens of completions
   with a nonzero advantage (`signal_tokens`); verl's `token-mean` divides by
   all tokens of the mini-batch, DAPO's own definition. Under Adam this only
   reweights steps relative to one another (TRL review, section 3).
4. **No checkpoint selection inside training.** verl saves every 10 steps and
   keeps them all; `arggym_verl.report` picks the best by dev success and
   compares it with step 0, as TRL's `selection.json`.
5. **ArgGYM only: no curriculum, and `None` rewards become 0.** verl samples
   prompts uniformly (a custom sampler is possible but not built). A completion
   TRL scores `None` (truncated, or a row the scorer refuses) cannot leave its
   group's baseline in verl; it pays 0 and `refused` is logged. Branch
   `rl-003` makes truncated pay 0 in TRL too.
6. **Checkpoint state.** verl's FSDP checkpoint holds the whole model state,
   not only the adapter; about 2.5 GB a step for Qwen3-0.6B.

## GPU smoke status

Two attempts on 2026-10-20 got through Ray start-up, FSDP2 model and LoRA
init, the reward loop and the checkpoint engine, then stopped on environment
and upstream faults, both fixed since: the flash-attn wheel (setup above) and
the `flush=True` bug. A third run was stopped when the GPUs were reassigned,
before the step-0 dev eval. No rollout, dev score or training step has run yet.

## Not checked

- The served share of the merged update (point 1) and whether 2e-5 still learns
  through it; the smoke run reports dev at steps 0 and 3 only.
- Qwen3.5-4B on this stack (vision checkpoint, Gated DeltaNet, 46k-token
  sequences on padded SDPA): composed and validated on CPU, never run.
