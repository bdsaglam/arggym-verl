# ArgGYM GRPO on verl

This branch (`arggym`) is upstream verl v0.9.1 (`1876b06d`) plus one package,
`arggym_verl/`, that trains ArgGYM's two RL tasks with verl's own GRPO, async
trainer and token-in/token-out rollouts. It is the second pipeline beside the
TRL one in the ArgGYM repo (`rl/`, branch `rl-training`): same data, same
reward code, same model, so a metric that moves in both can be trusted.
Upstream's README is `README.md`; the design and every place this run differs
from the TRL run are in `workspace/verl-port-2026-10-20/design.md`.

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

The splits are written from the TRL run's own config by ArgGYM's own selection
code, so both pipelines see the same ids:

```
export PYTHONPATH=$PWD:$ARGGYM_ROOT
# GSM8K hard control: the screened questions, dev 200 test rows, seed 0
.venv/bin/python -m arggym_verl.data $ARGGYM_ROOT/rl/configs/gsm8k-hard-qwen3-0.6b-adapter.yaml \
    --out data/gsm8k-hard data.screen.path=<abs path>/outputs/rl/gsm8k-screen/qwen3-0.6b-c1899de
# ArgGYM: the pool rl.pool built (output_root is where its pools/ directory lives)
.venv/bin/python -m arggym_verl.data $ARGGYM_ROOT/rl/configs/qwen3.5-4b-curriculum.yaml \
    --out data/arggym-qwen3.5-4b output_root=<abs path>/outputs/rl
```

## Runs

Two GPUs, each holding an FSDP2 trainer rank and a vLLM replica that serves the
LoRA adapter (verl's hybrid engine, `trainer_mode: colocate_async`; design.md
says why not three, and why not separate rollout GPUs):

```
CUDA_VISIBLE_DEVICES=0,1 scripts/arggym/run.sh gsm8k-hard-qwen3-0.6b
CUDA_VISIBLE_DEVICES=0,1 scripts/arggym/run.sh arggym-qwen3.5-4b
# smoke: 3 steps, dev at step 0 and at the last step
RUN=smoke-gsm8k CUDA_VISIBLE_DEVICES=0,1 scripts/arggym/run.sh gsm8k-hard-qwen3-0.6b \
    trainer.total_training_steps=3 trainer.resume_mode=disable
```

`run.sh` calls `arggym_verl.main`, which is `verl.trainer.main_ppo` plus a
per-step check that vLLM serves the trained adapter (`served/*` metrics; the
run stops if vLLM is closer to the base model). The smoke checklist with the
expected step-1 values is in design.md.

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
reward), `actor/entropy`, `actor/pg_clipfrac`, `served/logp_gap_*`, `rollout_corr/*` (importance
weights: `rollout_is_mean`, `rollout_is_eff_sample_size`,
`rollout_is_ratio_fraction_high`), `served/z`,
`training/off_policy/*` (staleness of the trained groups).

## Tests (CPU)

```
CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q arggym_verl/tests
```

They check that the parquet holds the TRL run's ids, prompts and gold answers;
that the reward matches TRL's on fixed completions, through verl's reward-loop
input; that verl's agent loop sends vLLM the prompt token ids TRL sends; and that
the config composes, passes verl's validation, satisfies the async batch
invariant, and carries the TRL run's hyperparameters.
