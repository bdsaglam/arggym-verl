# ArgGYM GRPO on verl

This branch (`arggym`) is upstream verl v0.9.1 (`1876b06d`) plus one package,
`arggym_verl/`, that trains ArgGYM's two RL tasks with verl's own GRPO, async
trainer and token-in/token-out rollouts, and two small patches to verl (one
makes vLLM refuse to serve the base model in place of the LoRA adapter). It is
the second pipeline beside the TRL one in the ArgGYM repo (`rl/`, branch
`rl-training`): same data, reward code and model, with verl's own recipe for
the algorithm (`examples/tuning/lora/run_qwen3_8b_fsdp.sh`, LR from verl's
LoRA guide). Upstream's README is `README.md`; the design, the hyperparameters
and their sources are in `workspace/verl-port-2026-10-20/design.md`.

## Setup

```
scripts/arggym/setup.sh
```

This syncs `.venv` from verl's own `uv.lock` (`--extra fsdp --extra vllm`),
replaces flash-attn with the wheelhouse build for the lock's torch 2.11 (the
lock's URL is dead and the index now serves a torch-2.13 build that does not
load), and adds `python-argumentation` (ArgGYM's scorer) and `trackio`.

The data selection, prompts and scorers are imported from an ArgGYM checkout,
`ARGGYM_ROOT` (default `/home/baris/repos/ArgGYM2`), put on `PYTHONPATH` by the
launcher; the run records that checkout's commit in `outputs/<run>/versions.txt`.

## Data

The parquet files are gitignored; each split directory commits a
`manifest.json` with the ids, the ArgGYM commit and the command's arguments.
A fresh checkout writes the parquet with ArgGYM's own selection code, from
the TRL run's config, so both pipelines see the same ids:

```
export PYTHONPATH=$PWD:$ARGGYM_ROOT
# GSM8K hard control: the screened questions, dev 200 test rows, seed 0. The screen
# (Qwen3-0.6B, 8 samples per question) is committed under data/gsm8k-screen/, copied
# from ArgGYM at 79d9fe2.
.venv/bin/python -m arggym_verl.data $ARGGYM_ROOT/rl/configs/gsm8k-hard-qwen3-0.6b-adapter.yaml \
    --out data/gsm8k-hard data.screen.path=$PWD/data/gsm8k-screen/qwen3-0.6b-c1899de
# ArgGYM: the pool rl.pool built (output_root is where its pools/ directory lives)
.venv/bin/python -m arggym_verl.data $ARGGYM_ROOT/rl/configs/qwen3.5-4b-curriculum.yaml \
    --out data/arggym-qwen3.5-4b output_root=<abs path>/outputs/rl
```

## Runs

Two GPUs, each holding an FSDP2 trainer rank and a vLLM replica that serves the
LoRA adapter (verl's hybrid engine, `trainer_mode: colocate_async`; design.md
says why not three, and why not separate rollout GPUs). A step is 64 prompts x
5 completions, two optimizer steps of 32 prompts:

```
CUDA_VISIBLE_DEVICES=0,1 scripts/arggym/run.sh gsm8k-hard-qwen3-0.6b
CUDA_VISIBLE_DEVICES=0,1 scripts/arggym/run.sh arggym-qwen3.5-4b
# smoke: 5 steps, dev at step 0 and at the last step
RUN=smoke-gsm8k CUDA_VISIBLE_DEVICES=0,1 scripts/arggym/run.sh gsm8k-hard-qwen3-0.6b \
    trainer.total_training_steps=5 trainer.resume_mode=disable
```

`run.sh` calls `arggym_verl.main`, which is `verl.trainer.main_ppo` plus a
per-step check that vLLM serves the trained adapter (`served/*` metrics). Once
the adapter has moved far enough for the check to decide (`served/armed`), the
run stops if vLLM's log-probs did not move with it (`served/track` below
0.25). The smoke checklist is in design.md; read training metrics from step 3,
since steps 1 and 2 split the first two batches by completion length.

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
`training/off_policy/*` (staleness of the trained groups).

## Tests (CPU)

```
CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q arggym_verl/tests
```

They check that the parquet holds the ids, prompts and gold answers of
ArgGYM's selection on the committed screen; that the reward matches ArgGYM's on
fixed completions, through verl's reward-loop input; that verl's agent loop
sends vLLM the prompt token ids ArgGYM sends; that the config composes, passes
verl's validation with the KL loss on, pads no batch on 2 GPUs, carries the
recipe's values (read from the recipe script), the LR from verl's LoRA guide,
the rollout correction and the 8,192 training cap; that the served-policy check
runs inside verl's own step order with KL on and off and stops a base-served
run; and that vLLM's server refuses a LoRA request when the adapter is not
loaded.
