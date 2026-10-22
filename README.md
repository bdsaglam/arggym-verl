# arggym-verl: GRPO with LoRA on ArgGYM, on verl

This repository trains language models with GRPO and a LoRA adapter on
[ArgGYM](https://github.com/ieddeveci/ArgGYM)'s argumentation tasks, using
[verl](https://github.com/volcengine/verl) v0.9.1's own trainer: its async V1
trainer (`colocate_async`), token-in/token-out rollouts through vLLM, and its
LoRA recipe (`examples/tuning/lora/run_qwen3_8b_fsdp.sh`, LR from verl's LoRA
guide). It is upstream verl plus one package, `arggym_verl/`, and two small
patches to verl. It runs two tasks:

- **GSM8K hard control** (Qwen3-0.6B, 2 GPUs): GSM8K questions the base model
  solves 2 to 6 times in 8. A task GRPO is known to move, to check the pipeline.
- **ArgGYM** (Qwen3.5-4B, 3 GPUs): the task of ArgGYM's experiment 003, 57
  (task, level) cells for training and all 96 for dev, prompts drawn by a
  pass-rate curriculum.

The design, every hyperparameter and its source, and the full results are in
[docs/arggym/design.md](docs/arggym/design.md). verl's own README is
[README.verl.md](README.verl.md).

## Results

GSM8K hard control, Qwen3-0.6B, dev success over 200 rows x 4 samples
(`val-core/gsm8k/acc/mean@4`), 100 steps:

| Step | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dev | 0.610 | 0.609 | 0.644 | 0.645 | 0.670 | 0.663 | 0.666 | 0.696 | 0.689 | 0.700 | 0.733 |

ArgGYM, Qwen3.5-4B, dev success over 96 rows x 2 samples
(`val-core/arggym/acc/mean@2`; one standard error is about 0.035). This run drew
its prompts uniformly from the 57 training cells; the curriculum run, the
configuration in this repository, is pending. In progress at step 41 of 100:

| Step | 0 | 10 | 20 | 30 | 40 |
|---|---|---|---|---|---|
| Dev | 0.286 | 0.328 | 0.406 | 0.396 | 0.484 |

In both runs the trainer and vLLM agree (no phantom clipping, importance-weight
ESS 0.96 or above) and vLLM provably serves the trained adapter (`served/track`
near 1 or above). design.md, "Results", has the detail.

## Requirements

- Linux x86_64 with NVIDIA GPUs: 2 for GSM8K, 3 for ArgGYM. Tested on A100
  80GB; the ArgGYM trainer peaks at about 40 GB allocated, and vLLM takes 80%
  of each GPU while generating.
- An NVIDIA driver for CUDA 13.0 (580 or newer). The venv brings its own CUDA
  runtime: torch 2.11.0+cu130, cuDNN 9.19, NCCL 2.28.9, vLLM 0.24.0, flash-attn
  2.8.3, transformers 5.9.0, peft 0.19.1, Ray 2.55.1. The runs have worked with an
  old system CUDA toolkit (11.5) because `run.sh` turns off FlashInfer's sampler,
  which would JIT-compile with it.
- [uv](https://docs.astral.sh/uv/) (built with 0.9.13) and git. uv installs
  Python 3.12 if it is missing.
- Network access to GitHub and the Hugging Face Hub. Both models and the GSM8K
  dataset are public; no token is needed.
- Disk: about 11 GB for the venv, 10 GB for the two models, 60 MB for data.
  Checkpoints: about 28 GB for a 100-step GSM8K run (full trainer state) and
  about 1.5 GB per checkpoint for ArgGYM (adapter and optimizer only).

## Quickstart

```bash
git clone https://github.com/bdsaglam/arggym-verl && cd arggym-verl

# The venv (.venv) and the two pinned model snapshots.
scripts/arggym/setup.sh

# The parquet files under data/ (a few minutes; see "Data").
scripts/arggym/prepare_data.sh

# CPU tests: about a minute, no GPU.
CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q arggym_verl/tests

# A 5-step GSM8K smoke run on 2 GPUs, dev at step 0 and at the last step.
RUN=smoke-gsm8k CUDA_VISIBLE_DEVICES=0,1 scripts/arggym/run.sh gsm8k-hard-qwen3-0.6b \
    trainer.total_training_steps=5 trainer.resume_mode=disable

# The two runs.
CUDA_VISIBLE_DEVICES=0,1 scripts/arggym/run.sh gsm8k-hard-qwen3-0.6b
CUDA_VISIBLE_DEVICES=0,1,2 scripts/arggym/run.sh arggym-qwen3.5-4b
```

On A100s a GSM8K step takes about 3.5 minutes and a dev evaluation 6 to 14
(the reported run stopped at step 100 of the config's 200), and an ArgGYM step
about 26 minutes and a dev evaluation 30 to 55; its 100 steps take about 2.5 days.
Start runs in tmux or similar.

`run.sh` takes a config name under `arggym_verl/config/` and any Hydra overrides
of verl's config after it. `RUN` names the output directory (default: the config
name); a run with the same `RUN` resumes from its latest checkpoint.

## Configs

| Config | Model | GPUs | Batch | Steps |
|---|---|---|---|---|
| `gsm8k-hard-qwen3-0.6b` | Qwen3-0.6B @ `c1899de` | 2 | 64 prompts x 5 completions, one batch generated ahead | 200 |
| `arggym-qwen3.5-4b` | Qwen3.5-4B @ `851bf6e` | 3 | 12 prompts x 16 completions, each step on its own batch | 100 |

Both use verl's LoRA recipe: rank 64, alpha 32, all linear layers, LR 3e-5
constant, KL loss 0.001 against the base model, token-mean loss, clip 0.2,
two optimizer steps per batch, truncated importance sampling capped at 2. Both
evaluate dev at step 0 and every 10 steps and keep every 10-step checkpoint.
The ArgGYM config inherits the GSM8K one and changes the task: data, sampling
(the model card's, presence penalty 1.5), 40,960-token training and 65,536-token
dev caps, the curriculum. The model paths point into the Hugging Face cache
(`HF_HUB_CACHE`, which `run.sh` sets from your Hugging Face settings).

### Other GPU counts

Each GPU holds one FSDP2 trainer rank and one vLLM replica, so the GPU count is
`trainer.n_gpus_per_node`. verl pads every batch to a multiple of
`lcm(GPUs, ppo_mini_batch_size x rollout.n)`, so pick a count for which that
divides `train_batch_size x rollout.n`, or the padding trains on duplicated
completions:

- GSM8K: 320 completions, mini-batch 160. 2, 4, 5, 8 or 10 GPUs pad nothing;
  3 would pad to 480.
- ArgGYM: 96 completions, mini-batch 48. 2, 3, 4, 6 or 8 GPUs pad nothing. One
  GPU does not fit: the unsharded FP32 trainer weights (18 GB) beside vLLM's 64 GB.

For example, ArgGYM on 4 GPUs:
`CUDA_VISIBLE_DEVICES=0,1,2,3 scripts/arggym/run.sh arggym-qwen3.5-4b trainer.n_gpus_per_node=4`.
Changing the batch or `rollout.n` instead changes the experiment. With fewer GPUs
each vLLM replica decodes more sequences at once; ArgGYM's `rollout.max_num_seqs`
(48) caps that and queues the rest. `test_config.py` checks the arithmetic for the
committed counts.

## Outputs and monitoring

Everything lands in `outputs/<RUN>/`: `checkpoints/global_step_N`, `rollouts/`
and `dev/` (one JSONL of generations per step), `metrics.jsonl`, `rewards/` (one
line per scored completion), `driver.log`, and `versions.txt` (the verl commit and
the `arggym` version). Metrics also go to trackio, stored locally:

```bash
TRACKIO_DIR=outputs/trackio .venv/bin/trackio show --project arggym-verl
```

After or during a run,

```bash
.venv/bin/python -m arggym_verl.report outputs/<RUN>
```

writes `selection.json` (the checkpoint with the best dev success, the step-0
score, and whether it beats it) and prints training success, truncation and the
share of groups without signal per weight version.

What to read: `val-core/<task>/acc/mean@N` (dev success, the selection metric),
`val-aux/<task>/truncated/mean@N`, `critic/score/mean` (training reward),
`actor/entropy`, `actor/kl_loss`, `actor/pg_clipfrac`, `served/track` and
`served/armed`, `served/logp_gap_*`, `rollout_corr/*` (importance weights),
`training/off_policy/*` (staleness), `actor/perf/max_memory_allocated_gb`. For
the curriculum: `curriculum/mean_signal`, `curriculum/draw_entropy` (4.04 nats
is uniform over the 57 cells), `curriculum/level_share/L*`, and every 5 steps
each cell's `curriculum/pass_rate/*` and `curriculum/draw_share/*`. On GSM8K,
read training metrics from step 3: steps 1 and 2 split the first two batches by
completion length (design.md, "Async"). design.md, "Smoke checklist", lists the
expected values.

## Guards: precision and the served policy

- **No phantom clipping.** The PPO ratio's denominator is the trainer's own
  recomputed log-probs (`bypass_mode: false`), not vLLM's; vLLM's log-probs enter
  only as a token importance weight capped at 2, which reweights tokens and never
  zeroes them. Master weights, LoRA factors and optimizer state are FP32;
  compute is BF16; vLLM serves the published BF16 checkpoint plus the adapter.
- **vLLM serves the adapter, not the base model.** A patch to verl makes vLLM
  refuse a request, with an error, when the adapter is not loaded (upstream
  silently served the base model). Each step, the trainer also checks that
  vLLM's log-probs moved with the adapter (`served/track`, about 1 when it
  serves the adapter, about 0 for the base). Once the adapter has moved enough
  for the check to decide (`served/armed`), the run stops if `served/track`
  falls below 0.25.

design.md, "The served model is the trained one" and "Precision, and the PPO
ratio", has the derivation.

## Data

`scripts/arggym/prepare_data.sh` writes four parquet files with
`python -m arggym_verl.data <task config> --out <dir>`:

- `data/gsm8k-hard/`: from `openai/gsm8k` at a pinned revision (downloaded from
  the Hub) and the committed screen in `data/gsm8k-screen/qwen3-0.6b-c1899de/`
  (8 Qwen3-0.6B samples per question, written by ArgGYM's `rl/gsm8k_screen.py`).
  1,322 training questions, 200 dev.
- `data/arggym-qwen3.5-4b/`: from experiment 003's pool in
  `data/arggym-pool/pools/0df63cbebe86/`. The `arggym` library generates the
  pool from fixed seeds, dropping every item that appears in ArgGYM's eval
  tasksets; the result is byte-identical to the pool ArgGYM built for 003.
  Its `dev.jsonl` and `summary.json` are committed (so `git status` shows any
  difference); `train.jsonl` (60 MB) is not. 5,700 training rows, 96 dev.

Each output directory also gets a committed `manifest.json` with the row ids.
The task configs are `arggym_verl/tasks/*.yaml`. The curriculum's starting pass
rates are `data/arggym-prior/hf-qwen3.5-4b__xml_tags__cot.json`, the base
model's ArgGYM eval per cell; the file names its source.

## Repository layout

```
arggym_verl/            our package
  main.py               entry point: verl's main_ppo plus the two trainer mixins
  trainer.py            served-policy check, curriculum feedback (mixins on verl's trainer)
  curriculum.py         ArgGYM's curriculum as verl's training sampler
  reward.py             reward manager: ArgGYM's scorers on the sampled token ids
  agent_loop.py         per-split sampling overrides on verl's single-turn agent loop
  data.py               task config -> parquet
  report.py             best checkpoint and training-side metrics of a run
  config/               Hydra configs (the two runs, their agent loops)
  tasks/                task configs: the rows each run trains and evaluates on
  vendor/               ArgGYM's training glue, vendored (MIT; see below)
  tests/                CPU tests; fixtures/ are outputs of ArgGYM's own code
scripts/arggym/         setup.sh, prepare_data.sh, run.sh, write_fixtures.py
data/                   screen, pool dev split, prior, manifests (parquet is generated)
docs/arggym/design.md   design, hyperparameters, results, known gaps
verl/ examples/ ...     upstream verl v0.9.1
```

ArgGYM itself enters in two ways. Its library, `arggym` (task generators and
grader), is a dependency, installed from git at a pinned commit
(`scripts/arggym/requirements.txt`; it is not on PyPI). Its training and eval
code, which the library does not include, is vendored in `arggym_verl/vendor/`
from ArgGYM commit `a5b05fb`: data selection, prompts, the eval harness's
scorer and the curriculum. `arggym_verl/vendor/__init__.py` lists each file's
source, and the parity tests compare the vendored code with outputs of
ArgGYM's own code (`arggym_verl/tests/fixtures/`).
`scripts/arggym/write_fixtures.py` rewrites those fixtures from an ArgGYM
checkout after re-vendoring.

## Relation to upstream verl

The history is upstream verl's up to the v0.9.1 tag (`1876b06d`), then our
commits. Outside `arggym_verl/`, `scripts/arggym/`, `data/` and `docs/arggym/`,
two commits change verl:

- `verl/workers/rollout/vllm_rollout/vllm_async_server.py`: refuse a LoRA request
  when the adapter is not loaded, checked after admission (the served-policy
  guard above).
- `verl/trainer/ppo/v1/trainer_separate_async.py`: drop an invalid `flush=True`
  from a logger call (not on these runs' path).

To move to a newer verl, rebase our commits onto its tag
(`git remote add upstream https://github.com/volcengine/verl.git`, then
`git fetch upstream --tags && git rebase --onto <tag> v0.9.1`), resolve the two
patches, rerun `scripts/arggym/setup.sh` (the lock and the flash-attn wheel URL
change with verl), and run the CPU tests. Several tests pin behaviour read from
verl's code (step order, batch padding, the LoRA recipe's values), so a verl
change that matters fails there first.

## License

Apache-2.0, as verl ([LICENSE](LICENSE), [Notice.txt](Notice.txt)); our additions
are under the same license. The files in `arggym_verl/vendor/` are ArgGYM's,
under the MIT license in [arggym_verl/vendor/LICENSE](arggym_verl/vendor/LICENSE).
