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

**Two GPUs.** `_balance_batch` pads every batch to a multiple of
`lcm(ranks, mini-batch completions)` (`trainer_base._get_required_batch_multiple`,
`verl/trainer/ppo/v1/trainer_base.py:1604-1621`). With mini-batch 32 x 5 = 160
completions, two ranks need `lcm(2, 160) = 160`, which divides the 320 of a
batch; three would pad 320 to `lcm(3, 160) = 480`.

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
| Sampled log-probs | `logprobs_mode: processed_logprobs` | at T 1, top_p 1, no top-k, the raw log-probs |

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

The TRL pipeline in ArgGYM (`rl/`, branch `rl-training`) uses its own
hyperparameters; this pipeline uses verl's recipe, so the two are compared on
the same task (rows, reward, lengths, dev sampling, schedule), not the same
algorithm settings. Beyond the hyperparameters:

1. **Staleness.** TRL is synchronous; this run trains one version stale from
   step 3, with the truncated IS weight.
2. **Loss denominator.** TRL's DAPO loss divides by the tokens of completions
   with a nonzero advantage. verl's `token-mean` divides by all tokens of the
   mini-batch across ranks.
3. **No checkpoint selection inside training.** Every 10-step checkpoint is
   kept. `arggym_verl.report` picks the best by dev success and compares it
   with step 0.
4. **ArgGYM only.**
   - There is no curriculum: prompts are drawn uniformly.
   - A completion TRL scores `None` pays 0 here, and refusals are logged as
     `refused`. Branch `rl-003` makes a truncated completion pay 0 in TRL too.
   - The ArgGYM config inherits the GSM8K control's algorithm (recipe values,
     KL, LR 3e-5, rollout n 5) and overrides batch, lengths, sampling and LR
     1e-5. Its values have not been re-derived for Qwen3.5-4B; verl has no
     LoRA recipe for it (`examples/grpo_trainer/run_qwen3_5_2b_openr1_fsdp.sh`
     is full fine-tuning). It composes on CPU and has never run.

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
- the served-policy check through our entry point, `arggym_verl.main`.

`scripts/arggym/setup.sh` syncs the lock (`torch 2.11.0+cu130`,
`vllm 0.24.0`, `transformers 5.9.0`, `peft 0.19.1`, `ray 2.55.1`), with two
changes:

- **flash-attn.** The lock's flash-attn URL is dead (404), and the
  wheelhouse index now serves a torch-2.13 build that fails to load
  (undefined symbol `c10::ValueError`). The script installs the wheelhouse's
  torch-2.11 build instead.
- **Two additions:** `python-argumentation` (ArgGYM's scorer) and `trackio`.

ArgGYM code is imported from `$ARGGYM_ROOT` on `PYTHONPATH`, not copied:
`rl.gsm8k`, `rl.pool`, and `rl.reward`, which calls `evals.score.score_one`.
Each run records that checkout's commit. The GSM8K screen output is the one
exception: it is data, copied into `data/gsm8k-screen/` from ArgGYM at
`79d9fe2`.

## GPU status

Earlier smoke attempts got through Ray start-up, model and LoRA init, the
reward loop and the checkpoint engine, then stopped on the flash-attn wheel
(fixed by the setup script). No rollout, dev score or training step has run,
and this config has not run on GPU.
