# Design: ArgGYM GRPO on verl

verl v0.9.1 (tag `v0.9.1`, `1876b06d`, 2026-09-20), the latest release on
2026-10-20. Paths are in this repo, which is that tag, unmodified, plus
`arggym_verl/`. Where main (`ddf70072`, 2026-10-20) differs, it says so.

## The three features, and where verl provides them

| Feature | Entry point | Status |
|---|---|---|
| GRPO | `algorithm.adv_estimator=grpo` (`verl/trainer/ppo/core_algos.py`, `docs/algo/grpo.md`) | stable |
| Async RL | `verl.trainer.main_ppo`, V1 trainer (`trainer.use_v1=true`, the default, `verl/trainer/config/ppo_trainer.yaml:232`), `trainer.v1.trainer_mode` = `colocate_async` or `separate_async` (`verl/trainer/ppo/v1/`, `docs/advance/v1_async_trainer.md`) | main library; only `separate_async`'s step switching is labelled experimental |
| Token-in/token-out | `SingleTurnAgentLoop` (`verl/experimental/agent_loop/single_turn_agent_loop.py`): prompt ids in, the vLLM server's `generate(prompt_ids)` (`verl/workers/rollout/vllm_rollout/vllm_async_server.py:559`), sampled ids and log-probs out | every rollout goes through it, but it lives under `verl/experimental/` and `docs/advance/agent_loop.rst` still says "status: alpha" |

`verl/experimental/fully_async_policy` and `one_step_off_policy` are older,
experimental async paths that the V1 trainer supersedes.

## LoRA: the path chosen, and why

What verl ships (v0.9.1 and main alike):

- **Docs:** `docs/advance/ppo_lora.rst`. LoRA runs through PEFT with
  FSDP/FSDP2 and vLLM. `lora.merge` (default false) decides whether vLLM gets
  the adapter or merged weights. `load_format=safetensors` is required, and
  `layered_summon=True` is recommended.
- **Recipes:** `examples/tuning/lora/`.
  - `run_qwen3_8b_fsdp.sh` is the "canonical LoRA baseline", adapter served:
    `lora_rank`/`lora_alpha`, `load_format=safetensors`, `layered_summon=True`,
    LR 3e-6 at rank 64 / alpha 32.
  - `run_qwen3_8b_merge_fsdp.sh` merges, with LR 1e-5 at rank 32 / alpha 64.
  - `run_qwen3_8b_from_adapter_fsdp.sh`, the VL script and the Megatron one
    complete the set.
  - All of them run the default `trainer_mode: sync` on the hybrid engine
    (`examples/tuning/README.md`: "rollout still serves the adapter via
    `load_format=safetensors + layered_summon`").
- **Tests:** `tests/workers/test_engine_workers_lora_sync.py` covers the
  hybrid sync's two branches: adapter mode sends base weights once, then
  adapter deltas; merge mode sends merged weights. `tests/utils/test_fsdp_lora_merge.py`
  covers the merge. No test or recipe combines LoRA with an async mode.

How the adapter reaches vLLM, by sync path:

- **Hybrid engine (`sync`, `colocate_async`, and `separate_async`'s hybrid
  replicas).** The V1 trainer forces `checkpoint_engine.backend = "naive"`
  for its colocated replicas (`verl/trainer/ppo/v1/trainer_base.py:396`).
  The naive branch of `update_weights` (`verl/workers/engine_workers.py:739-826`)
  syncs the base weights once (or not at all with `load_format=safetensors`).
  After that, each sync sends only the LoRA tensors and `peft_config`. vLLM
  loads them as a LoRA adapter (`vllm_rollout/utils.py:248`) and every request
  names it (`vllm_async_server.py:652-660`). This is adapter serving.
- **Standalone replicas (`separate_async`, the only mode with rollout on
  separate GPUs).** These sync through a checkpoint engine (nccl/nixl/...),
  whose branch calls `get_per_tensor_param()` with its defaults
  (`engine_workers.py:782`; main: `:816`) and drops the `peft_config`.
  - With `merge=false`, that yields the base weights with every `lora_`
    tensor skipped (`collect_lora_params(base_sync_done=False)`,
    `verl/utils/fsdp_utils.py:715`). The server then loads them as plain
    weights (`verl/checkpoint_engine/base.py:355`) and samples the base model
    for the whole run, with no error.
  - With `merge=true`, the delta is merged into the FP32 master weight and
    the result is cast to BF16 before sending (`_merged_lora_per_tensor_param`,
    `verl/workers/engine/fsdp/transformer_impl.py:1063-1084`). Any
    per-element delta below half a BF16 ulp of its weight rounds away; that
    is the loss the TRL review measured.

Chosen: the canonical recipe's adapter serving (`lora.merge=false`,
`load_format=safetensors`, `layered_summon=true`) on the hybrid engine, in
`trainer_mode: colocate_async`, verl's async mode for that engine. The
alternative async mode, `separate_async`, can only serve this run merged in
BF16. LoRA r=32, alpha=32, all-linear and LR 2e-5 follow the TRL run. The
recipes use other values (3e-6 at r64/a32; 1e-5 at r32/a64), all with KL loss
on Qwen3-8B, and `ppo_lora.rst` says only "increase the learning rate by an
order of magnitude", so they give no reason to depart from TRL.

What `colocate_async` keeps and loses (`trainer_colocate_async.py`):

- **Kept.**
  - One batch is generated ahead (`num_warmup_batches=1`).
  - A completion that is still running at a weight sync is aborted, then
    resumed under the new weights, rather than holding the step (partial
    rollout).
  - The staleness bound and the off-policy correction work as in any V1
    async mode.
- **Lost.** Rollout and training never run at the same time. `on_sample_end`
  aborts and sleeps the vLLM replicas, and `on_step_end` syncs weights and
  resumes generation.
- **GPUs.** The run uses 2 GPUs, both doing both jobs. Over 3, verl would pad
  every 128-completion batch to `lcm(3, 128) = 384` rows
  (`trainer_base._get_required_batch_multiple`).

This combination (LoRA adapter serving plus `colocate_async`) is in verl's
code but in no recipe or test. The fallback is `trainer.v1.trainer_mode=sync`,
the recipes' own mode.

## The served model is the trained one: `arggym_verl.trainer`

The hybrid sync could still fail silently; vLLM accepts a LoRA load that maps
no module. So every training step tests it, using what verl already computes:

- vLLM's log-probs of the sampled tokens (`rollout_log_probs`).
- The trainer's log-probs with the adapter (`old_log_probs`).
- One extra forward pass with the adapter off. This is verl's own
  reference-in-actor path for LoRA (`no_lora_adapter`,
  `engine_workers.py:403,425`; `trainer_base._compute_ref_log_prob`).

Per token, `x = (vllm - base)^2 - (vllm - trained)^2` averages `+delta^2` if
vLLM serves the adapter and `-delta^2` if it serves the base model. The
engines' kernel gap appears in both terms and only adds variance, so the test
works even when the update is smaller than that gap. That is the case where
the TRL pipeline needed probe adapters. `served/z` is the mean of `x` over its
standard error; the run stops when it drops below -5.

At step 1 the adapter is untouched, so `z` is 0 by construction. From the
first steps after that, `z` should be positive and grow. The check is a mixin
installed over verl's registered trainer class by `arggym_verl.main`, which is
otherwise `verl.trainer.main_ppo`.

## Precision, and the PPO ratio

Phantom clipping (HF blog "Defeating the trainer-generator precision mismatch
in TRL", as quoted to us; not read here) is what happens when vLLM's log-probs
sit in the PPO ratio's denominator. The engine gap then puts tokens outside the
clip range at step 1, and those tokens get exactly zero gradient. verl avoids
it as follows:

- **Ratio.** `bypass_mode: false` (decoupled PPO). The trainer recomputes
  pi_old with the same engine and precision it trains with
  (`trainer_base._compute_old_log_prob`, `trainer_base.py:1649`). With
  `bypass_mode: true` the ratio's denominator would be vLLM's log-probs
  instead (`trainer_base.py:1657-1663`).
- **Clipping.** The loss uses `ratio = exp(log_prob - old_log_prob)`
  (`core_algos.py:1336-1339`). With one PPO epoch and one mini-batch, the
  ratio is 1 up to kernel nondeterminism, so the clip zeroes nothing.
- **vLLM gap.** The gap enters only as the token IS weight pi_old/pi_rollout,
  clamped at 2 (`rollout_correction.rollout_is: token`, threshold 2.0). It
  multiplies the loss (`core_algos.py:1364-1365`): it reweights tokens and
  never zeroes them, because `rollout_rs` (rejection) is off. This is TRL's
  `token_truncate`, cap 2.

Dtypes:

| Where | Dtype | Why |
|---|---|---|
| FSDP2 master weights, LoRA factors, AdamW state | FP32 (`fsdp_config.model_dtype: fp32`, the default for training, `transformer_impl.py:258-264`) | an Adam step on a BF16 weight below 2^-8 of it is lost |
| Trainer forward/backward | BF16 compute, FP32 gradient reduce (FSDP2 `MixedPrecisionPolicy`, `transformer_impl.py:393-399,455`) | the TRL trainer's autocast BF16 |
| vLLM base weights | BF16 (`rollout.dtype`), the published checkpoint, loaded from safetensors | vendor precision |
| vLLM adapter | vLLM LoRA dtype `auto`, which is BF16 | the factors are rounded on their own, not added into a weight, so no delta falls below a weight's ulp |
| Sampled log-probs | `logprobs_mode: processed_logprobs` | the distribution actually sampled from |

## Logged per step, and the smoke checklist

| Metric | Meaning | Expected at step 1 | TRL reference (GSM8K runs) |
|---|---|---|---|
| `actor/pg_clipfrac` (+ `_lower`) | share of tokens where the clip is active | ~0 (below 1e-3); anything near 0.15 means phantom clipping | `clip_ratio/region_mean` exactly 0.0 every step |
| `served/logp_gap_mean`, `served/logp_gap_max` | mean and max \|trainer - vLLM\| log-prob per token | mean of order 1e-2, max a few nats on rare tokens | the quoted blog: mean \|beta\| ~0.08 on Qwen3-0.6B (stated there for BF16 vLLM against an FP32 trainer) |
| `served/is_truncated_fraction`, `rollout_corr/rollout_is_ratio_fraction_high` | share of tokens whose IS weight is clamped at 2 | below 1% | `is_zeroed_fraction` 0 (TRL zeroes nothing either) |
| `rollout_corr/rollout_is_eff_sample_size`, `rollout_is_mean` | ESS of the IS weights, their mean | ESS ~0.99, mean ~1 | ESS 0.998 |
| `training/rollout_probs_diff_*`, `rollout_corr/kl` | verl's own engine-gap diagnostics | small | - |
| `served/z`, `served/adapter_shift_mean` | served-policy test, adapter shift | 0 and 0 at step 1; positive and growing after | - |
| `training/off_policy/trajectory_staleness*` | weight versions behind | 0-1 | 0 (synchronous) |

## Where the run differs from the TRL run

1. **One step of staleness.** A group can come from the previous weights, and
   a completion can span two weight versions. The truncated IS weight covers
   both; TRL is synchronous.
2. **Loss denominator.** TRL divides the DAPO loss by the tokens of
   completions with a nonzero advantage. verl's `token-mean` divides by all
   tokens of the mini-batch, which is DAPO's own definition. Under Adam this
   only reweights steps against one another.
3. **2 GPUs, not 3** (see above).
4. **No checkpoint selection inside training.** Every 10-step checkpoint is
   kept. `arggym_verl.report` picks the best by dev success and compares it
   with step 0, as TRL's `selection.json` does.
5. **ArgGYM only.**
   - There is no curriculum: prompts are drawn uniformly.
   - A completion TRL scores `None` pays 0 here, and refusals are logged as
     `refused`. Branch `rl-003` makes a truncated completion pay 0 in TRL too.
   - The configuration (vision checkpoint, Gated DeltaNet, 46k-token
     sequences) is composed and validated on CPU but has never run.

## Repo and environment

The repo follows verl's install guide (`docs/start/install.rst`): run from the
repo root on the committed `uv.lock`. Custom pieces plug in through config:

- the reward manager by file path (`reward.reward_manager.source=importlib`,
  `docs/advance/reward_loop.rst`);
- agent loops by `_target_` YAML (`rollout.agent.agent_loop_config_path`);
- the served-policy check through our entry point.

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
Each run records that checkout's commit.

## GPU smoke status

Three attempts on 2026-10-20 got through Ray start-up, model and LoRA init,
the reward loop and the checkpoint engine. They then stopped on the flash-attn
wheel (fixed by the setup script) and on a `logger.info(flush=True)` bug in
`separate_async`'s `hybrid_engine=False` branch. That bug is fixed on verl
main and is irrelevant now that the run uses `colocate_async`. The GPUs were
then reassigned. No rollout, dev score or training step has run.
