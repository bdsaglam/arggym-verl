# Design: ArgGYM GRPO on verl

verl v0.9.1 (tag `v0.9.1`, `1876b06d`, 2026-09-20), the latest release on
2026-10-20. Paths are in this repo: that tag plus `arggym_verl/` and two
patches to verl itself (listed under "Patches to verl"). Where main
(`ddf70072`, 2026-10-20) differs, it says so.

## The three features, and where verl provides them

| Feature | Entry point | Status |
|---|---|---|
| GRPO | `algorithm.adv_estimator=grpo` (`verl/trainer/ppo/core_algos.py`, `docs/algo/grpo.md`) | stable |
| Async RL | `verl.trainer.main_ppo`, V1 trainer (`trainer.use_v1=true`, the default, `verl/trainer/config/ppo_trainer.yaml:232`), `trainer.v1.trainer_mode` = `colocate_async` or `separate_async` (`verl/trainer/ppo/v1/`, `docs/advance/v1_async_trainer.md`) | main library; only `separate_async`'s step switching is labelled experimental |
| Token-in/token-out | `SingleTurnAgentLoop` (`verl/experimental/agent_loop/single_turn_agent_loop.py`): prompt ids in, the vLLM server's `generate(prompt_ids)` (`verl/workers/rollout/vllm_rollout/vllm_async_server.py:559`), sampled ids and log-probs out | every rollout goes through it, but it lives under `verl/experimental/` and `docs/advance/agent_loop.rst` still says "status: alpha" |

`verl/experimental/fully_async_policy` and `one_step_off_policy` are older,
experimental async paths that the V1 trainer supersedes.

## Hyperparameters: verl's LoRA recipe

The GSM8K control follows `examples/tuning/lora/run_qwen3_8b_fsdp.sh`, which
`examples/tuning/README.md` names the canonical LoRA baseline. It is the only
recipe in the tree with GRPO, LoRA served as an adapter, GSM8K and Qwen3
together. The task defines the data, reward, lengths, dev sampling and eval
schedule; every other value is the recipe's, or verl's default where the
recipe is silent. `arggym_verl/tests/test_config.py` reads the recipe's
defaults out of the script and pins the config to them.

| Setting | Value | Source |
|---|---|---|
| LoRA | rank 64, alpha 32, all-linear | recipe; `target_modules` default |
| Batch | 64 prompts x 5 completions = 320; mini-batch 32 prompts, so two optimizer steps per batch; 1 PPO epoch | recipe |
| Optimizer | AdamW, LR **3e-5** constant, no warmup, weight decay 0.01, grad clip 1.0 | verl's LoRA guide for the LR (below); verl defaults (`verl/trainer/config/optim/fsdp.yaml`) for the rest, as in the recipe |
| KL | loss `low_var_kl`, coefficient 0.001, against the base model; none in the reward | recipe |
| Loss | `token-mean`, clip 0.2 / 0.2, entropy coefficient 0 | verl defaults (`verl/trainer/config/actor/actor.yaml`), as in the recipe |
| Advantage | GRPO, std normalization on | recipe + verl default |
| Off-policy correction | token-level truncated IS, cap 2, decoupled (`bypass_mode: false`) | departure: async staleness (below) |
| Async | `colocate_async`, 1 warmup batch, staleness bound 2 with `wait` | not in the recipe (its mode is `sync`); section "Async" |
| GPUs | 2, each an FSDP2 rank and a vLLM replica | recipe's default count; forced (below) |

Task-defining, unchanged: the screened GSM8K hard rows (dev 200, seed 0),
ArgGYM's scorer, thinking on, training rollouts at T 1.0 / top_p 1.0 / no
top-k with an 8,192-token cap, dev at T 0.6 / top_p 0.95 / top_k 20, 4
samples, 32,768-token cap, dev at step 0 and every 10 steps, a checkpoint
every 10 steps, best checkpoint by dev success.

**LR 3e-5, not the recipe's 3e-6.** verl's LoRA guide says to raise the LR
by an order of magnitude for LoRA (`docs/advance/ppo_lora.rst:164`) and its
FSDP reference uses 3e-5 at r32/a32 (`ppo_lora.rst:183-186`). The recipe's
3e-6 is 3x the full fine-tuning LR of `examples/grpo_trainer/run_qwen3_4b_fsdp.sh`
(1e-6), not 10x. The merge recipe (`run_qwen3_8b_merge_fsdp.sh`) uses 1e-5.

**Truncated IS, cap 2.** The recipe sets no `rollout_correction`
(`verl/trainer/config/algorithm.py:667`, default `None`). `colocate_async`
trains on completions that are one weight version stale or span two
versions (section "Async"), and the weight `pi_old / pi_rollout` corrects for
that. The block is the one `docs/algo/rollout_corr.md:170-182` documents.

**GPUs.** `_balance_batch` pads every batch to a multiple of
`lcm(ranks, mini-batch completions)` (`trainer_base._get_required_batch_multiple`,
`verl/trainer/ppo/v1/trainer_base.py:1604-1621`). With mini-batch 32 x 5 = 160
completions, two ranks need `lcm(2, 160) = 160`, which divides the 320 of a
batch; three would pad 320 to `lcm(3, 160) = 480`. So the GSM8K control runs on
two GPUs, and the ArgGYM run takes six completions a prompt to run on three
(section "ArgGYM run").

The step budget stays 200 steps: 12,800 prompts, about 9.7 passes over the
1,322 training questions.

## LoRA: the path chosen, and why

What verl ships (v0.9.1 and main alike):

- **Docs:** `docs/advance/ppo_lora.rst`. LoRA runs through PEFT with
  FSDP/FSDP2 and vLLM. `lora.merge` (default false) decides whether vLLM gets
  the adapter or merged weights. `load_format=safetensors` is required, and
  `layered_summon=True` is recommended.
- **Recipes:** `examples/tuning/lora/`. `run_qwen3_8b_fsdp.sh` serves the
  adapter; `run_qwen3_8b_merge_fsdp.sh` merges. All of them run the default
  `trainer_mode: sync` on the hybrid engine.
- **Tests:** `tests/workers/test_engine_workers_lora_sync.py` covers the
  hybrid sync's two branches: adapter mode sends base weights once, then
  adapter deltas; merge mode sends merged weights. No test or recipe
  combines LoRA with an async mode.

How the adapter reaches vLLM, by sync path:

- **Hybrid engine (`sync`, `colocate_async`, and `separate_async`'s hybrid
  replicas).** The V1 trainer forces `checkpoint_engine.backend = "naive"`
  for its colocated replicas (`trainer_base.py:396`). The naive branch of
  `update_weights` (`verl/workers/engine_workers.py:739-826`) skips the base
  sync with `load_format=safetensors`; each sync sends only the LoRA tensors
  and `peft_config`. vLLM removes the old adapter and loads the new one
  (`vllm_rollout/utils.py:270-273,363-375`), and every request names it
  (`vllm_async_server.py`, `generate`). This is adapter serving.
- **Standalone replicas (`separate_async`, the only mode with rollout on
  separate GPUs).** These sync through a checkpoint engine whose branch calls
  `get_per_tensor_param()` with its defaults (`engine_workers.py:782`; main:
  `:816`) and drops the `peft_config`. With `merge=false` the server gets the
  base weights with every `lora_` tensor skipped
  (`verl/utils/fsdp_utils.py:715`) and would sample the base model for the
  whole run; with the patch below it now refuses every request instead. With
  `merge=true` the delta is merged into the FP32 master weight and cast to
  BF16 before sending (`verl/workers/engine/fsdp/transformer_impl.py:1063-1084`),
  so any per-element delta below half a BF16 ulp of its weight rounds away.

Chosen: the recipe's adapter serving (`lora.merge=false`,
`load_format=safetensors`, `layered_summon=true`) on the hybrid engine, in
`trainer_mode: colocate_async`, verl's async mode for that engine.
`separate_async` can only serve this run merged in BF16.

## Async: what `colocate_async` does, step by step

`trainer_colocate_async.py:30-48`: weights sync once at init, a warmup batch
is submitted at train start, and every step aborts and sleeps the vLLM
replicas once its batch is sampled (`on_sample_end`), trains, then syncs the
new weights and resumes generation (`on_step_end`). Rollout and training
never run at the same time; the asynchrony is the batch generated ahead.
An aborted completion resumes under the new weights with its remaining
token budget (`verl/workers/rollout/llm_server.py:254-347`), so the 8,192 cap
holds across the resume.

Weight version v is the adapter after v optimizer steps; the trainer at step
k holds version k-1, and the server is told the version at each sync. Each
prompt is tagged with the step at which it was submitted
(`trainer_base.py:1508,1513`). With `num_warmup_batches: 1`:

- **Step 1.** `global_steps` is already 1 when `on_train_begin` submits the
  warmup batch (`trainer_base.py:468,479`), and `prepare_step` submits the
  step-1 batch with the same tag 1 (`trainer_base.py:1540-1541`). The `wait`
  rule does not block (`replay_buffer.py:550-553`: `1 - 1 + 1 < 2`), so the
  sampler takes the first 64 groups to finish out of 128 in flight
  (`replay_buffer.py:388-391` orders by tag only). Step 1 trains on the
  shortest completions, all from version 0.
- **Step 2.** The wait rule now holds the step until every tag-1 group has
  finished (`2 - 1 + 1 >= 2`), so step 2 trains on the 64 longest groups of
  the first 128: mostly version 0, the unfinished tails resumed under
  version 1.
- **Step k >= 3.** Step k trains on the batch submitted at step k-1. Its
  completions that finished during step k-1's generation window are one
  version stale (version k-2 against the trainer's k-1); the rest were
  aborted and resumed, so they span versions k-2 and k-1.
  `training/off_policy/trajectory_staleness_worst` reads 1 and
  `trajectory_spans/max` 2 (`trainer_base.py:1993-1995`).

Consequences: step-1 and step-2 score, length and truncation metrics are not
comparable with a synchronous run or with each other, since step 1 is biased
short and step 2 long. Read training metrics from step 3. The truncated IS
weight corrects the one-version staleness from step 3 on.
`num_warmup_batches: 0` would remove both effects and make the mode
equivalent to `sync`.

## The served model is the trained one

Two guards, one in vLLM and one in the trainer.

**vLLM refuses to serve the base model in place of the adapter.**
Upstream, `vLLMHttpServer.generate` sent a request with no `lora_request`
whenever the adapter was missing from the engine, i.e. to the base model,
without a log line. It also checked before the request parked behind a
weight sync's closed gate, where the old adapter is already removed, so a
request arriving in that window ran on the base weights after resume. The
patch checks after admission (the gate barrier keeps the next sync from
starting before `engine.generate`) and raises, with an error log, when the
adapter is not loaded (`arggym_verl/tests/test_vllm_lora_request.py`).

**The trainer checks that vLLM's log-probs moved with the adapter**
(`arggym_verl/trainer.py`). Each step has three log-probs per sampled token:
vLLM's (`rollout_log_probs`), the trainer's with the adapter
(`old_log_probs`), and with the adapter off (`ref_log_prob`, verl's
reference-in-actor path for LoRA, which the KL loss computes anyway). With
trainer shift d = trained - base and served shift s = vllm - base, the check
fits s = track * d over the step's tokens. `served/track` is ~1 when vLLM
serves the adapter (about 0.5-1 under one version of staleness early in the
run, approaching 1 as steps accumulate) and ~0 when it serves the base. The
engines' kernel gap is the noise, so the fit's standard error
`served/track_se` falls as the adapter moves. The check:

- uses only trajectories generated entirely by a trained adapter
  (`min_global_steps >= 1`): steps 1 and 2 have none, and `served/tokens` is 0;
- arms once `served/track_se <= 0.05` (`served/armed` = 1);
- once armed, stops the run before the update if `served/track < 0.25`.

It wraps `_compute_advantage`, which runs after verl's old-log-prob and
reference passes (`trainer_base.py:600,605,614`), so with the KL loss on it
costs no extra forward pass. `arggym_verl/tests/test_trainer.py` runs it
inside verl's own `_step_once` with KL on and off.

When it arms depends on the shift per token and on the kernel gap, neither
measured yet. With ~10^6 response tokens per step and a gap of ~0.08 nats
(the blog value quoted below), it arms at a mean shift of ~1.6e-3 nats per
token: an estimate, not a measurement.

## Precision, and the PPO ratio

Phantom clipping (HF blog "Defeating the trainer-generator precision mismatch
in TRL", as quoted to us; not read here) is what happens when vLLM's log-probs
sit in the PPO ratio's denominator: the engine gap puts tokens outside the
clip range at step 1, and those tokens get zero gradient. verl avoids it:

- **Ratio.** `bypass_mode: false` (decoupled PPO). The trainer recomputes
  pi_old with the same engine and precision it trains with
  (`trainer_base._compute_old_log_prob`, `trainer_base.py:1649`). With
  `bypass_mode: true` the denominator would be vLLM's log-probs
  (`trainer_base.py:1657-1663`).
- **Clipping.** The loss uses `ratio = exp(log_prob - old_log_prob)`
  (`core_algos.py:1336-1339`). On the first mini-batch of a step the ratio
  is 1 up to kernel nondeterminism; on the second, the policy has taken one
  optimizer step, so the clip works as PPO intends.
- **vLLM gap.** The gap enters only as the token IS weight pi_old/pi_rollout,
  clamped at 2. It multiplies the loss (`core_algos.py:1364-1365`): it
  reweights tokens and never zeroes them, because `rollout_rs` (rejection)
  is off.

Dtypes:

| Where | Dtype | Why |
|---|---|---|
| FSDP2 master weights, LoRA factors, AdamW state | FP32 (`fsdp_config.model_dtype: fp32`, the default for training, `transformer_impl.py:258-264`) | an Adam step on a BF16 weight below 2^-8 of it is lost |
| Trainer forward/backward | BF16 compute, FP32 gradient reduce (FSDP2 `MixedPrecisionPolicy`, `transformer_impl.py:393-399,455`) | verl's default |
| vLLM base weights | BF16 (`rollout.dtype`), the published checkpoint, loaded from safetensors | vendor precision |
| vLLM adapter | vLLM LoRA dtype `auto`, BF16 (vLLM-side; not read in vLLM) | the factors are rounded on their own, not added into a weight, so no delta falls below a weight's ulp |
| Sampled log-probs | `logprobs_mode: processed_logprobs` | GSM8K (T 1, top_p 1, no top-k): the raw log-probs. ArgGYM (top_p 0.95, top_k 20, presence penalty 1.5): the log-probs of the distribution sampled from, as TRL's server returns them; the IS weight then also covers the gap to the trainer's full distribution |

## Smoke checklist

Read training metrics from step 3 (section "Async").

| Metric | Expected | Wrong if |
|---|---|---|
| `val-core/gsm8k/acc/mean@4` at step 0 | near 0.58, the TRL run's step 0 on the same rows, model and dev sampling | far off: data, prompt or scorer |
| `actor/pg_clipfrac` | small but nonzero: the second mini-batch has a ratio away from 1 | ~0.15 from step 1: phantom clipping |
| `actor/kl_loss` | 0 at step 1 (adapter B is zero), growing slowly | absent: KL not on |
| `served/logp_gap_mean`, `_max` | mean of order 1e-2, max a few nats on rare tokens | mean near 1: wrong model or tokens |
| `rollout_corr/rollout_is_eff_sample_size`, `rollout_is_mean`, `rollout_is_ratio_fraction_high` | ESS ~0.99, mean ~1, clamped share below 1% | ESS far below 1: stale or mismatched rollouts |
| `served/tokens` | 0 at steps 1-2, positive from step 3 | 0 from step 3: version tags missing |
| `served/armed`, `served/track` | armed within a few steps of step 3; track 0.5-1 | the run stops itself if track < 0.25 once armed; never armed: the adapter is not moving |
| `training/off_policy/trajectory_staleness_worst/max`, `trajectory_spans/max` | 1 and 2 from step 3 | above 2: the staleness bound failed |
| driver log | no "LoRA adapter ... is not loaded" error | that error: vLLM lost the adapter |

## Comparing with the TRL run

The TRL pipeline in ArgGYM (`rl/`; GSM8K on branch `rl-training`, ArgGYM's
experiment 003 on `rl-003`) uses its own hyperparameters; this pipeline uses
verl's recipe, so the two are compared on the same task (rows, reward, lengths,
sampling, dev split), not the same algorithm settings. Beyond the
hyperparameters:

1. **Staleness.** TRL is synchronous. The GSM8K run trains one version stale
   from step 3, with the truncated IS weight; the ArgGYM run trains each step
   on its own batch, as TRL does.
2. **Loss denominator.** TRL's DAPO loss divides by the tokens of completions
   with a nonzero advantage. verl's `token-mean` divides by all tokens of the
   mini-batch across ranks.
3. **No checkpoint selection inside training.** Every 10-step checkpoint is
   kept. `arggym_verl.report` picks the best by dev success and compares it
   with step 0.
4. **ArgGYM only.**
   - Both draw prompts with the same curriculum class and settings; TRL's
     draws lag the outcomes by two steps, verl's by none (section
     "Curriculum").
   - A completion the scorer refuses is `None` in TRL and pays 0 here, logged
     as `refused`. A truncated completion pays 0 and stays in the loss in both.

## Patches to verl

- `f9b9654f`: drops `flush=True` from a `logger.info` call in
  `trainer_separate_async.py`'s `hybrid_engine=False` branch. Not on this
  run's path.
- `vllm_async_server.py`, `generate`: the adapter check above.

## Repo and environment

The repo follows verl's install guide (`docs/start/install.rst`): run from the
repo root on the committed `uv.lock`. Custom pieces plug in through config:

- the reward manager by file path (`reward.reward_manager.source=importlib`,
  `docs/advance/reward_loop.rst`);
- agent loops by `_target_` YAML (`rollout.agent.agent_loop_config_path`);
- the served-policy check and the curriculum through our entry point,
  `arggym_verl.main`, which mixes both into the V1 trainer.

`scripts/arggym/setup.sh` syncs the lock (`torch 2.11.0+cu130`,
`vllm 0.24.0`, `transformers 5.9.0`, `peft 0.19.1`, `ray 2.55.1`), with two
changes:

- **flash-attn.** The lock's flash-attn URL is dead (404), and the
  wheelhouse index now serves a torch-2.13 build that fails to load
  (undefined symbol `c10::ValueError`). The script installs the wheelhouse's
  torch-2.11 build instead.
- **Two additions:** `python-argumentation` (ArgGYM's scorer) and `trackio`.

ArgGYM code is imported from `$ARGGYM_ROOT` on `PYTHONPATH`, not copied:
`rl.gsm8k`, `rl.pool` (with `rl.pool.restrict`, branch `rl-003`), and
`rl.reward`, which calls `evals.score.score_one`. `ARGGYM_ROOT` defaults to
`.arggym`, a clone of the ArgGYM repository that `setup.sh` checks out at the
commit in `scripts/arggym/arggym-rev` (`a5b05fb`, experiment 003's). A clone
pinned by commit stays put when the main ArgGYM checkout changes branch or a
worktree is cleaned up; a branch name or a worktree path would not. Each run
records the commit. Two pieces of ArgGYM output are data, copied rather than
imported: the GSM8K screen (`data/gsm8k-screen/`, from ArgGYM at `79d9fe2`) and
003's pool (`data/arggym-pool/pools/0df63cbebe86/`, byte-identical to the one
003's run reads; `train.jsonl` is gitignored for size).

## ArgGYM run

`arggym_verl/config/arggym-qwen3.5-4b.yaml`, three GPUs. The algorithm is the
GSM8K config's (section "Hyperparameters"): LR 3e-5, LoRA r64/a32, KL
`low_var_kl` 0.001, mini-batch half the batch, std normalization, no warmup,
weight decay 0.01, token IS capped at 2. The task is ArgGYM's experiment 003
(`rl/configs/qwen3.5-4b-003.yaml` at `a5b05fb`), and `test_config.py` reads
each of these values from 003's own config:

| Setting | Value | Source |
|---|---|---|
| Rows | training: 5,700 rows of 57 (task, level) cells; dev: 96 rows, one per cell | 003's `train_cells` through `rl.pool.restrict`; 003's pool |
| Reward | strict success, 1 or 0; a truncated completion pays 0 and stays in the loss | `rl.reward` at `a5b05fb` |
| Model | Qwen3.5-4B, revision `851bf6e`, BF16 | 003 |
| Prompts | `xml_tags`, `cot`, thinking on | 003 and the eval model config |
| Training rollouts | T 1.0, top_p 0.95, top_k 20, presence penalty 1.5, 40,960-token cap | 003 |
| Dev | the same sampling (min_p 0, repetition penalty 1), 65,536-token cap, 2 samples a row | the eval config through `rl.config.dev_sampling`; 003 |
| Schedule | 100 steps; dev at step 0 and every 10; a checkpoint every 10 | 003's step count |
| Batch | 16 prompts x 6 = 96 completions; mini-batch 8 x 6 = 48 | below |
| Prompts | drawn by 003's curriculum: prior from the base eval, half-life 10 groups, floor 0.01 | 003's `curriculum:` block; section "Curriculum" |

**Batch.** Six completions a prompt is the count nearest the recipe's 5 that
three ranks divide: a mini-batch of M prompts is then 6M completions, a
multiple of 3, so `lcm(3, 6M) = 6M`, which divides the batch's 12M. Sixteen
prompts put 32 sequences on each vLLM replica, about half its KV cache at the
lengths 003 measured (below), and keep a step near half an hour; 24 would fill
three quarters of it and lengthen every step by about a quarter.

**Each step on its own batch** (`num_warmup_batches: 0`). With a batch
generated ahead, the completions still running when a step starts are aborted
and resumed after the weight sync, and `verl/workers/rollout/llm_server.py:291-296` resumes one by
sending prompt plus tokens so far as the new prompt. vLLM's presence penalty
counts output tokens only (`vllm/model_executor/layers/utils.py`,
`apply_penalties`), so a resumed completion would sample its remainder as if its
first part had never been penalized. With completions of ~28k tokens most of a
prefetched batch would resume. Without the batch ahead, each step's completions
come from the weights that train on them: staleness 0, steps 1 and 2 unbiased,
and the IS weight covers only the engine gap and the gap between the sampled
(penalized, top-k, top-p) distribution and the trainer's full one. The step
then waits for its longest completion, as 003's synchronous TRL run does.

**Qwen3.5.**

- verl loads the checkpoint as `Qwen3_5ForConditionalGeneration`
  (`AutoModelForImageTextToText`, `verl/utils/model.py:711`), as its Qwen3.5
  recipes do; the vision tower is inside the FSDP model, frozen, without an
  adapter (`exclude_modules: .*visual.*`). verl's Qwen3.5 patch
  (`verl/models/transformers/monkey_patch.py:500-542`) gives the Gated DeltaNet
  layers a packed-sequence forward with fla's `chunk_gated_delta_rule`.
  `causal_conv1d` is not installed, so the short convolution runs verl's PyTorch
  fallback: slower, not different.
- `all-linear` adapts 12 module kinds, 129.9M parameters at rank 64: q/k/v/o in
  the 8 full-attention layers, `in_proj_qkv`, `in_proj_z`, `in_proj_a`,
  `in_proj_b` and `out_proj` in the 24 Gated DeltaNet layers, gate/up/down in all
  32. 003's run lists the same kinds and counts.
- vLLM starts with `--language-model-only` (`engine_kwargs.vllm`), as ArgGYM
  serves it. vLLM 0.24's `Qwen3_5ForConditionalGeneration` packs
  `in_proj_qkv`+`in_proj_z` into `in_proj_qkvz`, `in_proj_b`+`in_proj_a` into
  `in_proj_ba`, q/k/v into `qkv_proj` and gate/up into `gate_up_proj`, and has
  `o_proj`, `out_proj` and `down_proj` as row-parallel linears, so every target
  has a LoRA layer (`test_config.py`). 003's TRL run probed all 12 kinds at its
  first sync, on vLLM 0.30, and vLLM applied each; here the served-policy check
  confirms it.
- Prefix caching off, entropy in chunks, gradient checkpointing and padding
  removal on, as in `examples/grpo_trainer/run_qwen3_5_27b_fsdp.sh`. Those
  recipes also offload parameters and optimizer to CPU; this run does not need
  to (below).

**Memory per 80 GB GPU.** Estimates from the shapes; only the run measures
them (`actor/perf/max_memory_allocated_gb`, vLLM's KV log line).

Training phase, vLLM asleep (weights in CPU memory, KV cache freed):

| Item | Arithmetic | GB |
|---|---|---|
| FP32 base weights, one shard of three | 4.54B x 4 B / 3 | 6.1 |
| LoRA weights, gradients, AdamW state (FP32) | 129.9M x 16 B / 3 | 0.7 |
| Layer inputs kept by checkpointing, one 49,152-token micro-batch | 49,152 x 2,560 x 2 B x 32 layers | 8.1 |
| Its BF16 logits (flash-attn's cross-entropy writes the gradient in place) | 49,152 x 248,320 x 2 B | 24.4 |
| One layer recomputed and backpropagated, its unsharded BF16 weights, the tied embedding | | ~6 |
| Trainer total | | ~45 |

The same arithmetic gives 7.3 GB for the GSM8K run (16,384-token micro-batches,
logits 5.0 GB), which peaked at 7.6 GB allocated. vLLM's sleeping remainder
(CUDA graphs, context) adds a few GB. If the trainer runs out of memory,
`actor_rollout_ref.model.use_fused_kernels=true` computes log-probs 512 tokens
at a time (`verl/utils/experimental/torch_functional.py`) and removes the
logits term.

Rollout phase: vLLM takes `gpu_memory_utilization` 0.8 = 64 GB: text-model
weights 8.4 GB (4.21B x 2 B), the adapter 0.3 GB, activations and CUDA graphs
~4 GB, and ~51 GB for the cache. The trainer keeps its 6.8 GB of shards, so
the GPU holds ~72 GB. A token's KV is 32 KiB (8 full-attention layers x K and
V x 4 heads x 256 x 2 B) and a sequence's Gated DeltaNet state ~49 MiB (24
layers x 32 heads x 128 x 128 x FP32), so the cache holds ~1.5M tokens. 003's
server (60 GiB, ~1.9M tokens) held 52 running sequences at 67% usage, ~24k
tokens each. A training step's 32 sequences a replica then need ~0.8M tokens;
dev's 64 a replica would need ~1.5M, so `max_num_seqs: 48` runs 48 (~1.2M)
and queues the rest rather than let vLLM preempt long completions.

**Step time, estimated.** Training: on GSM8K the three passes (old log-prob,
reference, update) ran at ~8,100 tokens/s per GPU on Qwen3-0.6B. Qwen3.5-4B has
8x the non-embedding parameters (3.57B against 0.44B) and attends over ~30k
tokens, so assume ~1/9 of that, ~900 tokens/s; a step's 96 x ~28k = 2.7M tokens
are 0.9M a GPU, ~17 minutes. Generation is bound by the longest completion:
003's server decoded each sequence at 25-55 tokens/s (3,000 tokens/s over 64
sequences, 442 over 8), so a completion at the 40,960 cap takes 12-25 minutes,
while the step's 2.6M tokens at 1.2-2k tokens/s a replica take ~10. A step
should take ~30 minutes (25-40), a dev evaluation ~30 (192 completions of ~29k,
the longest at 65,536), and the run of 100 steps with 11 evaluations ~2.5
days. 003 on TRL takes ~49 minutes a step.

**What only the GPU run settles.**

- That vLLM 0.24 applies the adapter to the packed Gated DeltaNet projections:
  `served/armed`, then `served/track` near 1. Sampled log-probs include the
  presence penalty and the top-k/top-p renormalization, which the trainer's do
  not, so the check's noise is far above GSM8K's 0.02 nats and it may arm later.
- Trainer and vLLM memory against the estimates above, and preemptions (vLLM's
  log lines in `driver.log`).
- verl's Qwen3.5 training path at 49,152-token micro-batches (fla kernels,
  padding removal, LoRA on the Gated DeltaNet projections), which no CPU test
  reaches.
- Step 0: `val-core/arggym/acc/mean@2` near 0.328, 003's step 0 on the same rows
  and sampling (its `dev/step_00000.jsonl`).
- The importance weights: `rollout_corr/rollout_is_eff_sample_size` and the
  clamped share, now that the sampled distribution is not the trainer's.
- Step time, and the share of groups without signal (`arggym_verl.report`;
  002, on all 96 cells, had 0.38-0.58).
- That the trainer reads every completion's `extra_info` and
  `reward_extra_info` from the real transfer queue: `curriculum/groups` 16
  from step 1. The CPU test reads them from a stand-in queue.

The GSM8K smoke checklist applies with three changes: `served/tokens` is
positive from step 2 (no batch ahead), `training/off_policy/*` staleness is 0,
and training metrics are comparable from step 1.

## Curriculum

Every ArgGYM run draws its training prompts with experiment 003's curriculum.
The sampler is ArgGYM's own class, `rl.curriculum.Curriculum` at `a5b05fb`,
imported from `.arggym`; `arggym_verl/curriculum.py` builds it and drives it.
Each step draws 16 prompts from the (task, level) cells in proportion to
p(1 - p) + 0.01, p being the cell's pass rate. p starts from the base model's
eval (`results/hf-qwen3.5-4b__xml_tags__cot/metrics.json`, 8 samples a cell)
through a Beta(1, 1) prior, and every training group adds its successes out of
its completions to its cell, after the cell's counts decay by 2^(-1/10). A
pass is a success that did not truncate; a completion the scorer refused is
left out (`rl.curriculum.group_outcomes`, applied to the reward's logged
`acc`, `truncated` and `refused`). Rows within a cell come without replacement,
reshuffled when used up. The seed is `data.seed` (0, as 003's).

**Where it plugs in.** verl v0.9.1 has no option for the prompt sampler:
`create_rl_sampler` (`verl/trainer/ppo/utils.py:140`) builds a random or
sequential sampler and nothing else. `trainer.v1.sampler.custom_sampler`
(`ppo_trainer.yaml:308-318`) replaces the replay buffer
(`trainer_base._build_replay_buffer`, `docs/extend_guide.rst:103-108`), which
picks among trajectories already generated, so it cannot choose prompts, and
verl has no hook that feeds outcomes to a sampler. The trainer mixin
`CurriculumFeedback` (`arggym_verl/trainer.py`) does both:

- `_init_dataloader` rebuilds verl's training dataloader with the curriculum
  as its sampler, with verl's batch size, collate function and `drop_last`.
- `_compute_advantage` reads each completion's `extra_info` (task, level,
  row index) and `reward_extra_info` from the transfer queue for the step's
  batch, groups them by row in the order the block drew them, and calls
  `Curriculum.update` before verl computes the advantages. It runs on the
  driver once a step, after the rewards and before the update.

**No lag.** In `colocate_async` verl fetches one prompt at a time
(`gen_batch_size` 1, `trainer_base.py:741-753`), 16 a step, in `prepare_step`
at the start of the step. The curriculum draws a block when the first prompt
of the step is fetched, so block k+1 is drawn after block k's outcomes are
in. TRL's dataloader drew two blocks ahead, so an outcome at step t first
shaped block t+2 there. Two settings keep it so, and `curriculum.check_setup`
refuses a run without them:

- `data.dataloader_num_workers: 0`. verl's default of 8 worker processes makes
  the dataloader pull indices ahead (two per worker), which draws the next
  block during the current step.
- `num_warmup_batches: 0`, no group filtering, no failed-group refill,
  `parameter_sync_step` 1, and `sync` or `colocate_async`: each fetches
  exactly one batch a step.

The async replay buffer still refills a group whose rollout failed
(`replay_buffer.py:580-583`), which would fetch a 17th prompt before the
update. The curriculum then raises rather than draw the next block from
estimates that miss the current step; TRL's run also stops on a failed
generation.

**Resuming.** torchdata's dataloader saves a sampler that has `state_dict`
inside its own state (`_BatchSamplerIterator.state_dict`), so the cell
counts, the per-cell row orders and positions, and the random generator go into
verl's `data.pt` with the rest. Checkpoints fall between steps, after the
update and before the next draw, which is the state TRL's `state_for_step`
returns for that step. A checkpoint of a run without the curriculum does not
load into one with it (its sampler state is the random sampler's).

**Logged** (metrics file and trackio): every step `curriculum/mean_signal`,
`mean_pass_rate`, `distinct_cells` (TRL's), `curriculum/level_share/L1..L8`
(the step's prompts by level), `curriculum/groups`, and
`curriculum/draw_entropy`, the entropy in nats of the distribution the next
step draws from (uniform over 57 cells is 4.04); at step 1 and every 5 steps
(`table_every`), `curriculum/pass_rate/<task>/L<level>` and
`curriculum/draw_share/<task>/L<level>` for every cell.

**Not ported:** `algorithm.filter_groups` (DAPO's dynamic sampling). 003 drops
no groups; it divides its loss by the tokens of completions with a nonzero
advantage instead, which verl's `token-mean` does not do (section "Comparing
with the TRL run").

`arggym_verl/tests/test_curriculum.py` builds the TRL side with
`rl.train.build_curriculum` on 003's config and rows and checks, for 40 steps
of synthetic outcomes fed in a shuffled order, that both draw the same rows and
hold the same counts; that a run resumed from `data.pt` draws and learns as the
uninterrupted one; and that the dataloader draws no block before the previous
one's outcomes. `test_trainer.py` runs verl's own `step` (prompt fetch through
`_fetch_one_gen_batch`, then `_step_once`) for three steps and checks the order
draw, generate, old log-probs, reference, curriculum update, advantage, actor
update.

## Results: GSM8K hard control

Run `outputs/gsm8k-hard-recipe`: `gsm8k-hard-qwen3-0.6b.yaml` on two GPUs,
verl at `60866134`, ArgGYM at `b433e6f`, 100 steps.

Dev success (`val-core/gsm8k/acc/mean@4`, 200 rows x 4 samples):

| Step | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dev | 0.610 | 0.609 | 0.644 | 0.645 | 0.670 | 0.663 | 0.666 | 0.696 | 0.689 | 0.700 | 0.733 |

Still rising at step 100, the best checkpoint (`selection.json`).

Precision, from step 3:

- `actor/pg_clipfrac` ~5e-4 (no phantom clipping); `actor/kl_loss` 0.001 at
  step 3, 0.031 at step 100.
- Importance weights: ESS 0.998, mean 1.000, clamped share below 1e-5.
- `served/logp_gap_mean` 0.017-0.019 nats.
- `served/armed` at step 3; `served/track` 0.50 at step 3, 0.67 at step 5, 0.85
  at step 10, 0.95 at step 20, 0.98 at step 100. Early on, one version of
  staleness is large next to the adapter's total shift; it shrinks as the
  shift grows.
- Trainer peak 7.6 GB allocated.

Time: a step takes 212 s (median from step 3): generation 140, old log-probs 15,
reference 12, update 45, weight sync 1.2. A dev evaluation takes 6-14 minutes.

The ArgGYM config composes and passes verl's validation on CPU; it has not run
on GPU.
