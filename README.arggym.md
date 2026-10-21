# ArgGYM GRPO on verl

This branch (`arggym`) is upstream verl v0.9.1 (`1876b06d`) plus one package,
`arggym_verl/`, that trains ArgGYM's two RL tasks with verl's own GRPO, async
trainer and token-in/token-out rollouts, and two small patches to verl (one
makes vLLM refuse to serve the base model in place of the LoRA adapter). It is
the second pipeline beside the TRL one in the ArgGYM repo (`rl/`): same data,
reward code and model, with verl's own recipe for the algorithm
(`examples/tuning/lora/run_qwen3_8b_fsdp.sh`, LR from verl's LoRA guide).
Upstream's README is `README.md`; the design, the hyperparameters and their
sources are in `workspace/verl-port-2026-10-20/design.md`.

## Results

GSM8K hard control (Qwen3-0.6B, 2 GPUs, `outputs/gsm8k-hard-recipe`), dev
success (`val-core/gsm8k/acc/mean@4`, 200 rows x 4 samples):

| Step | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dev | 0.610 | 0.609 | 0.644 | 0.645 | 0.670 | 0.663 | 0.666 | 0.696 | 0.689 | 0.700 | 0.733 |

Still rising at step 100. The trainer and vLLM agree throughout: PPO clip
fraction about 5e-4, importance-weight ESS 0.998, and `served/track` 0.50 at
step 3, 0.85 at step 10, 0.95 at step 20 and 0.98 at step 100. A step takes
about 3.5 minutes, a dev evaluation 6 to 14. design.md, "Results", has the
detail.

## Setup

```
scripts/arggym/setup.sh
```

This syncs `.venv` from verl's own `uv.lock` (`--extra fsdp --extra vllm`),
replaces flash-attn with the wheelhouse build for the lock's torch 2.11 (the
lock's URL is dead and the index now serves a torch-2.13 build that does not
load), and adds `python-argumentation` (ArgGYM's scorer) and `trackio`.

It also clones ArgGYM into `.arggym` and checks out the commit in
`scripts/arggym/arggym-rev`: `a5b05fb`, experiment 003's commit on ArgGYM's
branch `rl-003`, which the main ArgGYM checkout does not have. The data
selection, prompts and scorers are imported from there: `run.sh` and the tests
default `ARGGYM_ROOT` to `.arggym`, and the run records its commit in
`outputs/<run>/versions.txt`. To move to another ArgGYM commit, change
`arggym-rev` and rerun the last lines of `setup.sh`.

## Data

The parquet files are gitignored; each split directory commits a
`manifest.json` with the ids, the ArgGYM commit and the command's arguments.
A fresh checkout writes the parquet with ArgGYM's own selection code, from the
TRL run's config, so both pipelines see the same ids:

```
export PYTHONPATH=$PWD:$PWD/.arggym
# GSM8K hard control: the screened questions, dev 200 test rows, seed 0. The screen
# (Qwen3-0.6B, 8 samples per question) is committed under data/gsm8k-screen/, copied
# from ArgGYM at 79d9fe2.
.venv/bin/python -m arggym_verl.data .arggym/rl/configs/gsm8k-hard-qwen3-0.6b-adapter.yaml \
    --out data/gsm8k-hard data.screen.path=$PWD/data/gsm8k-screen/qwen3-0.6b-c1899de
# ArgGYM: experiment 003's rows, training restricted to its 57 cells (5,700 rows),
# dev all 96 cells
.venv/bin/python -m arggym_verl.data $PWD/.arggym/rl/configs/qwen3.5-4b-003.yaml \
    --out data/arggym-qwen3.5-4b output_root=$PWD/data/arggym-pool
```

The ArgGYM rows come from the pool `rl.pool` built for 003,
`data/arggym-pool/pools/0df63cbebe86/`. Its `dev.jsonl` and `summary.json` are
committed; `train.jsonl` (60 MB) is not. Copy it from the ArgGYM checkout that
ran 003 (`outputs/rl/pools/0df63cbebe86/train.jsonl`), or rebuild the pool
with `python -m rl.pool rl/configs/qwen3.5-4b-003.yaml
output_root=<abs path>/data/arggym-pool` from `.arggym` in ArgGYM's own
environment. The data tests fail without it.

## Runs

Each GPU holds an FSDP2 trainer rank and a vLLM replica that serves the LoRA
adapter (verl's hybrid engine, `trainer_mode: colocate_async`; design.md says
why not separate rollout GPUs).

```
# GSM8K: 2 GPUs, 64 prompts x 5 completions a step, one batch generated ahead
CUDA_VISIBLE_DEVICES=0,1 scripts/arggym/run.sh gsm8k-hard-qwen3-0.6b
# ArgGYM: 3 GPUs, 16 prompts x 6 completions a step, each step on its own batch
CUDA_VISIBLE_DEVICES=0,1,2 scripts/arggym/run.sh arggym-qwen3.5-4b
# smoke: 5 steps, dev at step 0 and at the last step
RUN=smoke-gsm8k CUDA_VISIBLE_DEVICES=0,1 scripts/arggym/run.sh gsm8k-hard-qwen3-0.6b \
    trainer.total_training_steps=5 trainer.resume_mode=disable
```

Both take two optimizer steps per batch (mini-batch half the batch), and both
batch sizes divide so that verl pads nothing (design.md, "GPUs").

`run.sh` calls `arggym_verl.main`, which is `verl.trainer.main_ppo` plus a
per-step check that vLLM serves the trained adapter (`served/*` metrics). Once
the adapter has moved far enough for the check to decide (`served/armed`), the
run stops if vLLM's log-probs did not move with it (`served/track` below
0.25). The smoke checklist is in design.md; on GSM8K read training metrics
from step 3, since steps 1 and 2 split the first two batches by completion
length.

Any `key=value` after the config name is a Hydra override of verl's config.
Everything lands in `outputs/<run>/`: `checkpoints/global_step_N`, `rollouts/`
and `dev/` (one JSONL of generations per step), `metrics.jsonl`, `rewards/`
(one line per scored completion), `driver.log`. Metrics also go to trackio:
`TRACKIO_DIR=outputs/trackio trackio show --project arggym-verl`.

After a run (or during one):

```
.venv/bin/python -m arggym_verl.report outputs/<run>
```

writes `selection.json` (the checkpoint with the best dev success, the step-0
score, and whether it beats it) and prints training success, truncation and
the share of groups without signal per weight version.

What to read: `val-core/<task>/acc/mean@N` (dev success, the selection
metric), `val-aux/<task>/truncated/mean@N`, `critic/score/mean` (training
reward), `actor/entropy`, `actor/kl_loss`, `actor/pg_clipfrac`,
`served/logp_gap_*`, `rollout_corr/*` (importance weights:
`rollout_is_mean`, `rollout_is_eff_sample_size`,
`rollout_is_ratio_fraction_high`), `served/track` and `served/armed`,
`training/off_policy/*` (staleness of the trained groups),
`actor/perf/max_memory_allocated_gb`. The ArgGYM run also logs vLLM's own
lines (running and waiting requests, KV cache usage) to `driver.log`.

## Tests (CPU)

```
CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q arggym_verl/tests
```

They check that the parquet holds the ids, prompts and gold answers of
ArgGYM's selection (the committed GSM8K screen; 003's pool, its 57 training
cells and 96 dev rows); that the reward matches ArgGYM's on fixed completions,
through verl's reward-loop input, a truncated completion paying 0 in both;
that verl's agent loop sends vLLM the prompt token ids ArgGYM sends (Qwen3, and
Qwen3.5 through its processor on every dev row); that both configs compose,
pass verl's validation with the KL loss on and pad no batch on their GPU
count; that the GSM8K config carries the recipe's values (read from the recipe
script), the LR from verl's LoRA guide, the rollout correction and the 8,192
training cap; that the ArgGYM config keeps that algorithm and takes 003's
model revision, thinking, caps, sampling and dev samples from 003's own config;
that the adapter covers the 12 Linear kinds of Qwen3.5's text model and vLLM's
Qwen3.5 has a LoRA layer for each; that the served-policy check runs inside
verl's own step order with KL on and off and stops a base-served run; and that
vLLM's server refuses a LoRA request when the adapter is not loaded.
