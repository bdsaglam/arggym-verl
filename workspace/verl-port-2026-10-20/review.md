# Review: ArgGYM GRPO port on verl (branch `arggym`, e769b22e..3fae188f on v0.9.1 `1876b06d`)

Reviewed on CPU only, 2026-10-20. Every "verified" claim comes from reading the
named lines or running the CPU tests; nothing below was executed on a GPU.
"Suspicion" marks reasoning from the code that no run has confirmed. Paths
without a repo prefix are in this repo; `rl/...` and `evals/...` are in the
ArgGYM checkout at `b433e6f`.

Direction change during the review: TRL is no longer the reference. Section 3
therefore identifies the verl recipe to follow and lists the overrides; the
TRL-parity checks in the config and tests are reported as carry-overs to
remove.

## Findings, ranked

### 1. The served-policy mixin crashes the run as soon as a KL loss is on (blocks the recipe switch)

Verified. `ServedPolicyCheck._compute_old_log_prob` (`arggym_verl/trainer.py:67-89`)
computes the adapter-off log-probs itself only when `self.use_reference_policy`
is false, and then reads `ref_log_prob` from the transfer queue unconditionally
(`trainer.py:78-80`). verl computes `ref_log_prob` in `_step_once` *after*
`_compute_old_log_prob` (`verl/trainer/ppo/v1/trainer_base.py:598-605`). So with
`actor.use_kl_loss=true`, the field is missing at that point and
`tq.kv_batch_get` raises `ValueError("Some fields are not ready in all the
requested keys!")` (transfer_queue `kv_batch_get`, read from the installed
package) on step 1. The current config has no KL, so it hides this; the LoRA
recipe (section 3) has `use_kl_loss=True`, and the mixin must change before
that config runs. Fix: compute the adapter-off pass inside the mixin whenever
it is needed (drop the `use_reference_policy` branch), or move the check to a
hook that runs after step 5, e.g. wrap `_compute_advantage`. No test covers
this path (section 5).

### 2. `colocate_async` with one warmup batch trains step 1 on the 16 fastest groups of 32, step 2 on the 16 slowest (comparison validity)

Verified by reading, not run. The warmup batch is submitted in `on_train_begin`
after `global_steps` is already 1 (`trainer_base.py:468,479`), and
`prepare_step` submits the step-1 batch with the same tag 1
(`trainer_base.py:540-542,1508,1513`). At step 1 the `wait` rule does not block
(`replay_buffer.py:551-555`: `1 - 1 + 1 < 2`), so `sample` takes the first 16
groups that finish out of the 32 in flight (`replay_buffer.py:383-392` sorts by
submission tag only; ties fall in set order). Shorter completions finish
first, so step 1 trains on a short-biased batch and step 2 (which must wait for
every tag-1 group, `replay_buffer.py:551-555`) on the long remainder. From step
3 on each step trains on exactly its own 16-prompt batch, with the prefetch
batch's already-finished completions one full weight version stale and its
unfinished ones spanning two versions (`llm_server.py:281-347`;
`trainer_base.py:1993-1995` logs this as `trajectory_staleness` 1 and
`trajectory_spans` 2). design.md's "one step of staleness" covers the steady
state but not the step-1/2 composition. Consequences: (a) step-1 and step-2
`critic/score`, length and truncation metrics are not comparable with any
synchronous run; (b) the smoke "expected at step 1" table in design.md is read
on a biased batch. Options: `num_warmup_batches: 0` removes both the bias and
the staleness and is then equivalent to `sync` (the warmup batch is the only
source of asynchrony in this mode, `trainer_colocate_async.py:34-48`); or keep
1 and read smoke metrics from step 3 on.

### 3. vLLM silently serves the base model whenever the adapter is not loaded; the check that guards this is blind for the first steps

Verified: `vllm_async_server.py:652-660` builds a `LoRARequest` only if
`VLLM_LORA_INT_ID in await self.engine.list_loras()`, otherwise
`lora_request=None` and the request runs on the base weights with no log line.
The naive sync path removes the old adapter and adds the new one on every
`update_weights` (`vllm_rollout/utils.py:270-273,363-375`), and `on_init_end`
syncs once before validation (`trainer_colocate_async.py:30-32`), so in the
normal path the adapter is always present. The `served/z` check
(`trainer.py:36-63`) is the only guard, and by construction it reads 0 at
steps 1 and 2: LoRA B is zero at init (`hf_model.yaml:131`,
`lora_B_init_method: zero`) and the first optimizer step runs at LR 0
(`torch_functional.py:784-787`, `current_step=0` with 5 warmup steps;
`zero_indexed_step: true`). Suspicion: with the engine gap at ~0.08 nats per
token and an adapter shift of order 1e-3, `z` grows roughly as
`shift * sqrt(tokens) / (2 * gap)`, so a dropped adapter is caught only after
the shift reaches a few 1e-3, i.e. some tens of steps at the recipe LR. State
this in the smoke checklist: a non-negative `served/z` in the first steps
proves nothing. The check also costs one extra forward over the batch per step
(`trainer_base.py:1710-1733`); with KL on (section 3) verl pays that forward
anyway and the mixin should reuse it, which is the same fix as finding 1.

### 4. The config carries TRL hyperparameters that no verl recipe uses (now the wrong reference)

Verified against `examples/tuning/lora/run_qwen3_8b_fsdp.sh`. The full list and
the proposed overrides are in section 3 below. The ones that change the
algorithm: LR 2e-5 with 5 warmup steps, LoRA r32/a32, 16 prompts x 8, one
mini-batch per batch, no KL, clip 0.2/0.28, no std normalisation,
weight decay 0. `arggym_verl/tests/test_config.py:76-118` pins all of them to
the TRL config and will have to be rewritten.

### 5. Three GPUs pad every batch by 3x; two GPUs are forced by arithmetic, in both the current and the recipe config

Verified: `_balance_batch` upsamples the batch to
`lcm(dp_size, ppo_mini_batch_size * rollout.n)` (`trainer_base.py:1604-1636`).
Current config: `lcm(3, 128) = 384` rows for 128 real ones; recipe values
(mini 32 x n 5 = 160): `lcm(3, 160) = 480`. Any 3-GPU run needs
`ppo_mini_batch_size * n` divisible by 3 (e.g. n=6, mini 16 or 32). design.md
states the 2-GPU choice; the arithmetic is only tested for 2
(`test_config.py:35-38`).

### 6. Tests: what they pin and what they miss

Ran `CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q arggym_verl/tests`:
23 passed, 0 skipped, 11.8 s. Verified gaps:

- `test_data.py:8` skips the whole data-parity module when the screen
  directory under `.claude/worktrees/agent-a671d3be36862eff3/` is absent. That
  path is a Claude worktree, not a durable location; when it is cleaned the
  only test of "same rows as ArgGYM" disappears without failing. Either copy
  the screen output into a durable path (`outputs/rl/gsm8k-screen/...` in
  ArgGYM2, where the TRL config already expects it, `rl/configs/gsm8k-hard-qwen3-0.6b-adapter.yaml:34`)
  or make the skip a failure.
- `test_trainer.py` exercises `served_stats` on synthetic tensors and the
  registry patch, never the mixin's `_compute_old_log_prob` against verl's step
  order. A fake trainer with `use_reference_policy=True` would have caught
  finding 1.
- `test_config.py:53-60` asserts `bypass_mode is False` and
  `model_dtype == "fp32"` but nothing asserts `rollout_correction` is present
  at all; with the recipe's default (`rollout_correction: None`,
  `verl/trainer/config/algorithm.py:667`) the test's `rc.bypass_mode` would
  raise, which is accidental protection rather than a pinned decision.
- Nothing tests the 8,192 training cap across a partial-rollout resume. The
  cap survives (`llm_server.py:254-333` rewrites `max_tokens` per attempt), but
  that is the one path where the agent-loop override could be lost.
- `test_prompt_tokens.py` compares verl's token builder with
  `apply_chat_template` for one synthetic row and pins thinking-on. Good, but
  it is the only test of the prompt path and uses a hand-written question
  rather than a dataset row.
- `report.py` has no test.

### 7. Minor, verified

- `actor.grad_clip: 1.0` (`gsm8k-hard-qwen3-0.6b.yaml:58`) is the deprecated
  alias of `optim.clip_grad` (`verl/workers/config/optimizer.py:54-61`); both
  are 1.0, so harmless, but drop one.
- `actor.calculate_entropy: true` is redundant: `actor/entropy` is computed in
  `_compute_old_log_prob` regardless (`trainer_base.py:1668-1696`); in the
  update it only adds the entropy forward cost.
- `rollout.val_kwargs.do_sample` is ignored by the V1 agent loop
  (`agent_loop_tq.py:73-76` copies only `top_p`, `top_k`, `temperature`);
  dev sampling still happens because temperature is 0.6, so the value is
  decorative.
- design.md "Expected at step 1: `actor/pg_clipfrac` ~0" holds only while the
  mini-batch equals the batch. Under the recipe (64 prompts, mini 32) the
  second mini-batch has a ratio away from 1 and a nonzero clip fraction is
  normal.
- The verl core patch `f9b9654f` removes `flush=True` from a
  `logger.info(...)` call in `trainer_separate_async.py:181`; `logging.Logger.info`
  takes no such kwarg, so the patch is correct and harmless. It is dead code
  for `colocate_async`. design.md says main fixed it too; not checked here.
- `data/gsm8k-hard/*.parquet` is gitignored (`.gitignore:125`); only the
  manifest with ids is committed, so a fresh checkout must regenerate the
  parquet from the ArgGYM checkout at the recorded sha. Fine, but say so in
  the README.
- `data/arggym-qwen3.5-4b/` is untracked in the working tree (9,600 train /
  96 dev rows, ArgGYM sha b433e6f).

## 1. What actually runs

All verified by reading unless marked.

- **Trainer mode.** `trainer.use_v1: true`, `trainer_mode: colocate_async`
  resolves to `PPOTrainerColocateAsync` (`trainer_colocate_async.py:19-20`)
  through `TaskRunnerV1.run` (`main_ppo.py:139-141`). `arggym_verl.main`
  subclasses the Ray actor class and calls `install()` inside the actor before
  `super().run`, so the registry swap happens in the process that reads it
  (`main.py:20-26`, `trainer.py:92-101`, `main_ppo.py:102-103`).
- **GRPO.** `adv_estimator: grpo` -> `compute_grpo_outcome_advantage`
  (`core_algos.py:268-332`), grouped by `uid`, mean baseline, std off by
  config. Loss `compute_policy_loss_vanilla` (`core_algos.py:1286-1378`),
  token-mean over the whole mini-batch across DP ranks
  (`transformer_impl.py:729-734` all-reduces `batch_num_tokens`;
  `losses.py:63-66`; `core_algos.py:1170-1175`).
- **Async.** One batch prefetched (`trainer_base.py:638-665`), abort + sleep
  at sample time, sync + resume at step end (`trainer_colocate_async.py:37-48`).
  Partial rollouts resume with the cumulative `max_tokens` budget
  (`llm_server.py:254-333`). Rollout and training never overlap in time
  (`on_sample_end` sleeps the replicas before the trainer runs). See finding 2
  for what the prefetch actually buys and costs.
- **Token-in/token-out.** `SamplingAgentLoop` -> `SingleTurnAgentLoop.run`
  builds prompt ids with the continuous-token builder from
  `data.apply_chat_template_kwargs` (`agent_loop.py:242-251`,
  `single_turn_agent_loop.py:54-60`), sends `prompt_ids` to
  `vLLMHttpServer.generate` which wraps them in `TokensPrompt`
  (`vllm_async_server.py:636-642`), and keeps the returned ids and log-probs
  (`single_turn_agent_loop.py:84-98`). The merge for single-turn is a plain
  concatenation (`continuous_token.py:197-202`); nothing is decoded and
  re-encoded, and no EOS is appended, so the reward's truncation rule sees
  exactly what vLLM produced. The trainer scores `responses` from the queue
  (`agent_loop_tq.py:184-202`). The reward manager slices the valid response
  the same way verl's own does (`reward.py:81-91` vs `naive.py:34-40`).
- **Dev sampling.** Validation overrides `top_p/top_k/temperature` from
  `val_kwargs` and uses `val_kwargs.n` (`agent_loop_tq.py:73-76,115`); the
  `dev` agent loop sets no `max_tokens`, so the server falls back to
  `response_length` = 32,768 (`vllm_async_server.py:607-614`). Training rows
  pass `max_tokens: 8192` through the loop (`agent_loop.py:29-30`,
  `agent_loops-gsm8k.yaml:4-7`); `SamplingAgentLoop.__init__` refuses a cap
  above `response_length`.
- **LoRA as an adapter.** `ref_in_actor` is true for `lora_rank > 0`
  (`trainer_base.py:248-251`); the V1 trainer forces
  `checkpoint_engine.backend = "naive"` (`trainer_base.py:395-396`); the naive
  branch of `update_weights` (`engine_workers.py:779-826`) skips the base sync
  because `load_format: safetensors` sets `base_sync_done = True`
  (`engine_workers.py:673`), then sends only LoRA tensors plus `peft_config`
  (`get_per_tensor_param(base_sync_done=True)`), which vLLM loads via
  `add_lora(TensorLoRARequest)` (`utils.py:363-375`). vLLM is started with
  `enable_lora`, `max_loras=1`, `max_lora_rank` (`vllm_async_server.py:417-435`)
  and sleeps at level 1 (`:1294-1305`). `lora.merge: false` keeps
  `peft_merge` off (`engine_workers.py:675`). The adapter-served path is
  covered upstream by `tests/workers/test_engine_workers_lora_sync.py`
  (exists; not run here).
- **Reward.** Rule-based reward with no RM -> reward computed inside the agent
  loop workers (`reward_loop.py:313-321`), so `_validate` never sleeps the
  replicas mid-validation (`trainer_base.py:1101-1104` is skipped). Dev
  metrics land as `val-core/gsm8k/acc/mean@4` because `acc` is present in
  `reward_extra_info` (`trainer_base.py:1365-1385`,
  `metric_utils.py:961-971`); `report.py:43-44` searches exactly that key.

## 2. Precision and the PPO ratio

Verified:

- `bypass_mode: false` -> `_compute_old_log_prob` recomputes `old_log_probs`
  with the actor engine (`trainer_base.py:1655-1696`); the ratio in the loss is
  `exp(log_prob - old_log_prob)` (`core_algos.py:1336-1339`). vLLM's
  `rollout_log_probs` enter only as the IS weight
  `exp(old_log_prob - rollout_log_prob)`, clamped above at 2.0 with no lower
  bound for a float threshold (`rollout_corr_helper.py:568-602`), detached
  (`:627`), multiplied into `pg_losses` (`core_algos.py:1363-1365`). `rollout_rs`
  is null, so no token is zeroed. This matches "truncated IS, cap 2".
- Trainer dtypes: `model_dtype` defaults to fp32 for training
  (`transformer_impl.py:258-262`; composed config shows `model_dtype: fp32`),
  so the base weights and the LoRA factors are FP32 masters (no cast at
  `transformer_impl.py:370-375` because base and adapter already agree);
  FSDP2 `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)`
  (`transformer_impl.py:393-399,455-457`); forward under bf16 autocast
  (`transformer_impl.py:1559-1565`). The log-prob pass and the update pass
  share this engine and precision (`engine_workers.py:396-440` and
  `:710-719` both go through `self.actor`), so `old_log_probs` and
  `log_prob` differ only by nondeterminism in the single-mini-batch case.
- vLLM: base in BF16 from the safetensors path (`rollout.dtype: bfloat16`);
  LoRA tensors leave the trainer in their FP32 master dtype
  (`fsdp_utils.py:715-760` casts nothing); vLLM's LoRA dtype is left at its
  default, so the cast to BF16 happens inside vLLM's `add_lora`. Not read in
  vLLM; design.md's "adapter in BF16" is consistent with this but is
  vLLM-side behaviour I did not verify.
- `logprobs_mode: processed_logprobs` with T=1, top_p=1, top_k=-1 equals raw
  log-probs for training; the IS weight compares like with like.
- The served-is-trained test: `x = (r-b)^2 - (r-t)^2` has mean `+delta^2`
  when vLLM serves the adapter and `-delta^2` when it serves the base, with
  the kernel gap in both terms (`trainer.py:45`). The sign logic is right. The
  `rollout`/`trained`/`base` tensors are fetched after `to_padded_tensor` on
  the same response-aligned nested fields, so the `min(width)` trim at
  `trainer.py:83-84` is a no-op safeguard. The blind window is finding 3.
- Not checked: the HF blog design.md quotes ("Defeating the trainer-generator
  precision mismatch"); design.md itself says it was not read.

## 3. Which verl recipe to follow, and the override list

Candidates read: `examples/tuning/lora/run_qwen3_8b_fsdp.sh` (GRPO + LoRA,
GSM8K, Qwen3-8B, FSDP, vLLM adapter serving: "canonical LoRA fine-tuning
baseline"), `examples/tuning/lora/run_qwen3_8b_merge_fsdp.sh` (same with
merge: LR 1e-5, r32/a64, batch 128, mini 64, n 8),
`examples/grpo_trainer/run_qwen3_4b_fsdp.sh` (full FT, LR 1e-6, batch 512,
mini 256, n 5, KL 0.001), `examples/grpo_trainer/run_qwen3_8b_fsdp.sh` (full
FT, LR 1e-6, batch 1024, mini 256, n 5), `examples/grpo_trainer/run_qwen3_5_2b_openr1_fsdp.sh`
(full FT, LR 5e-7, batch 64, mini 16, n 5, KL 0.02, entropy chunking: the
template for the later Qwen3.5 run), `examples/rollout_correction/run_qwen2_5_7b_fsdp.sh`
(RLOO, sequence IS, bypass mode: not a GRPO recipe), `docs/algo/dapo.md`,
`docs/algo/grpo.md`, `docs/algo/rollout_corr.md`, `docs/advance/ppo_lora.rst`.
There is no `recipe/dapo` directory at v0.9.1.

**Follow `examples/tuning/lora/run_qwen3_8b_fsdp.sh`.** It is the only recipe
that combines GRPO, LoRA served as an adapter, GSM8K and Qwen3, and
`examples/tuning/README.md:9-15` names it the canonical baseline. Its values
and verl's defaults where it is silent:

| Setting | Recipe / verl default | Current config | Verdict |
|---|---|---|---|
| `actor.optim.lr` | 3e-6 | 2e-5 | TRL carry-over: revert to 3e-6. Flag: `docs/advance/ppo_lora.rst:164` says to raise the LR "by an order of magnitude" for LoRA, and its FSDP reference at `:183-186` uses 3e-5 with r32/a32; the merge recipe uses 1e-5. 3e-6 is only 3x the full-FT 1e-6 of `run_qwen3_4b_fsdp.sh`. Default to the recipe; if dev does not move in the smoke, 1e-5 is the next value with a verl source. |
| `actor.optim.lr_warmup_steps` | -1 (none; `fsdp.yaml`, `transformer_impl.py:509-517`) | 5 | TRL carry-over: drop. |
| `actor.optim.weight_decay` | 0.01 (default) | 0.0 | TRL carry-over: drop the override. |
| `model.lora_rank` / `lora_alpha` | 64 / 32 | 32 / 32 | TRL carry-over: revert to 64/32. (`ppo_lora.rst:171` says r>=32 is enough for a 0.5B model; 64 is the recipe.) |
| `model.target_modules` | `all-linear` (default, `hf_model.yaml:49`) | `all-linear` | same |
| `data.train_batch_size` | 64 | 16 | TRL carry-over: 64. |
| `actor.ppo_mini_batch_size` | 32 | 16 | TRL carry-over: 32 (two optimizer steps per batch; the clip then does work on the second). |
| `rollout.n` | 5 | 8 | TRL carry-over: 5. |
| `actor.use_kl_loss` / `kl_loss_coef` / `kl_loss_type` | True / 0.001 / low_var_kl | false | TRL carry-over: enable. Requires finding 1 fixed first. The reference is the base model through `disable_adapter` (`trainer_base.py:1718-1722`). |
| `actor.clip_ratio_low` / `high` | 0.2 / 0.2 (default) | 0.2 / 0.28 | DAPO value from TRL: revert to defaults. |
| `algorithm.norm_adv_by_std_in_grpo` | True (default) | false | Dr.GRPO value from TRL: revert. |
| `actor.loss_agg_mode` | token-mean (default) | token-mean | same |
| `actor.entropy_coeff` | 0 | 0 | same |
| `actor.calculate_entropy` | false (default) | true | drop (redundant, finding 7). |
| `actor.grad_clip` | not set (`optim.clip_grad` 1.0) | 1.0 | drop the deprecated alias. |
| `algorithm.rollout_correction` | not set (`None`, `algorithm.py:667`): no IS weight, old log-probs still recomputed | token / 2.0 / bypass false | Departure to keep, with a source: it is the YAML block `docs/algo/rollout_corr.md:170-182` documents, and `colocate_async` (not in the recipe either) makes rollouts one version stale, which is what the weight corrects. If the recipe is to be followed literally, remove it and watch `training/rollout_probs_diff_*` instead. |
| `trainer.v1.trainer_mode` | sync (default) | colocate_async | Spec requirement, not recipe. See finding 2 for the cost; `num_warmup_batches: 0` would make it equivalent to sync. |
| `trainer.v1.sampler.max_off_policy_threshold` / strategy | 8 / drop (default) | 2 / wait | Async choice, not in any recipe; keep if colocate_async stays. |
| `data.max_prompt_length` | 512 | 1024 | task-neutral (longest prompt 235 tokens); keep. |
| `data.max_response_length` + train cap | 1024 | 32768 dev / 8192 train | task-defining: keep (thinking averages ~3.1k tokens per the coordinator; not measured here). |
| `rollout.val_kwargs` | T 0 / n 1 / greedy | T 0.6, top_p 0.95, top_k 20, n 4 | task-defining: keep. |
| `trainer.val_before_train` | False | True | task-defining (step-0 score): keep. |
| `trainer.test_freq` / `save_freq` / `total_training_steps` | 5 / 20 / 15 epochs | 10 / 10 / 200 | budget choice; keep. |
| `rollout.gpu_memory_utilization` | 0.6 | 0.5 | choice; either. |
| `rollout.tensor_model_parallel_size` | 2 (8B) | 1 (0.6B) | model-size choice; keep 1. |
| `actor.ppo_max_token_len_per_gpu` / `rollout.log_prob_max_token_len_per_gpu` | 12288 | 16384 | must hold 1024 + 8192; keep 16384. |
| `data.filter_overlong_prompts` | True | false + `truncation: error` | equivalent for GSM8K; keep. |
| `trainer.n_gpus_per_node` | 2 | 2 | forced by finding 5 in both. |
| `reward.reward_manager`, `rollout.agent.agent_loop_config_path`, data files | n/a | ours | task-defining: keep. |

Proposed override block (Hydra, on top of the current yaml after removing the
TRL values):

```
actor_rollout_ref.model.lora_rank=64
actor_rollout_ref.model.lora_alpha=32
actor_rollout_ref.actor.optim.lr=3e-6
actor_rollout_ref.actor.optim.lr_warmup_steps=-1
actor_rollout_ref.actor.optim.weight_decay=0.01
data.train_batch_size=64
actor_rollout_ref.actor.ppo_mini_batch_size=32
actor_rollout_ref.rollout.n=5
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.001
actor_rollout_ref.actor.kl_loss_type=low_var_kl
actor_rollout_ref.actor.clip_ratio_low=0.2
actor_rollout_ref.actor.clip_ratio_high=0.2
algorithm.norm_adv_by_std_in_grpo=True
# keep: algorithm.rollout_correction={rollout_is: token, rollout_is_threshold: 2.0, bypass_mode: false}
# delete from the yaml: actor.grad_clip, actor.calculate_entropy
```

Batch arithmetic under these values on 2 GPUs: 64 x 5 = 320 completions,
mini 32 x 5 = 160, `lcm(2, 160) = 160`, 320 % 160 = 0: no padding, two
optimizer steps per batch. The step cost rises 2.5x in rollouts (320 vs 128
completions at up to 8,192 tokens), and the 200-step budget becomes 400
optimizer steps; decide whether `total_training_steps` stays 200.

For the later Qwen3.5-4B run, `run_qwen3_5_2b_openr1_fsdp.sh` is the closest
template (LR 5e-7 full FT, KL 0.02, mini 16, `entropy_from_logits_with_chunking`,
`entropy_checkpointing`), but it is full fine-tuning on a 2B model; there is
no LoRA + Qwen3.5 recipe in the tree. `arggym-qwen3.5-4b.yaml` currently
inherits the GSM8K config and sets LR 1e-5; that will need the same
re-derivation once the GSM8K smoke settles the LoRA LR.

## 4. Silent-failure risks

- vLLM falls back to the base model when no adapter is loaded, without a log
  line (`vllm_async_server.py:652-660`); the only guard is `served/z`, which is
  blind early (finding 3).
- `layered_summon` silently falls back to a full summon with a warning when
  the layered gather returns nothing (`fsdp_utils.py:731-736`). Harmless for
  correctness; it only changes peak memory.
- `rollout.val_kwargs.do_sample` and `n` are read from different places;
  `do_sample` is ignored (`agent_loop_tq.py:73-76`).
- `data.truncation: error` plus `filter_overlong_prompts: false`: a prompt
  over 1,024 tokens raises rather than being cut. Right for this data.
- `reward.py:92-93` writes the reward log only when `ARGGYM_VERL_REWARD_LOG`
  is set (`run.sh:25` sets it); `report.py` silently prints an empty table if
  the directory is missing.
- `report.py:48-49` selects only among steps whose checkpoint directory
  exists; a dev score at a non-save step is ignored without notice. With
  `test_freq == save_freq` this is moot.
- `ServedPolicyCheck` leaves `calculate_entropy: False` and
  `compute_loss: False` in `batch.extra_info` after the adapter-off pass
  (`trainer_base.py:1712-1719`). `_update_actor` resets `calculate_entropy`
  (`trainer_base.py:1845-1847`) and the train path never reads
  `compute_loss` (`engine_workers.py:344-369`), so no effect today; it is
  fragile.
- `_compute_ref_log_prob` leaves `no_lora_adapter: True` in the trainer-side
  `extra_info`; the worker pops its own copy (`engine_workers.py:403`) and
  the train path does not read it, so upstream is safe, and the mixin pops it
  anyway (`trainer.py:77`).
- The `f9b9654f` patch is correct (section "Minor").
- `setup.sh` replaces the locked flash-attn wheel with a different wheelhouse
  release and installs two extra packages outside the lock; not reproducible
  from `uv.lock` alone. Not verified (no install run here).

## 5. Tests

See finding 6. Summary of what is pinned today: parquet ids/prompts/gold equal
ArgGYM's selection (when the screen path exists); reward equals ArgGYM's on
nine fixed GSM8K completions and six ArgGYM rows (the ArgGYM part also
depends on a Claude-worktree pool path, `test_reward.py:78`); prompt token ids
equal `apply_chat_template` for one synthetic row; the config composes and
carries the TRL values; `served_stats` separates adapter-served from
base-served on synthetic data with a 4x noise-to-shift ratio; `install`
patches the registry. Not pinned: the mixin against verl's step order with KL
on (finding 1), the recipe values (section 3), 3-GPU padding, the training
cap across a resume, `report.py`.

## Smoke checklist corrections to design.md

- Read the step-1 table from step 3 on, or set `num_warmup_batches: 0`
  (finding 2).
- `served/z` is 0 at steps 1-2 by construction and small for several steps
  after (finding 3); treat only a sustained negative value as a failure.
- With mini-batch 32 of 64, `actor/pg_clipfrac` is nonzero on the second
  mini-batch and does not indicate phantom clipping.
- Expect `actor/kl_loss` and `rollout_corr/*` both present once the recipe
  config lands; the KL reference is the base model, so `actor/kl_loss` should
  grow slowly from 0.
