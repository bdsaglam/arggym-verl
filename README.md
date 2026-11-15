# arggym-verl: GRPO with LoRA on ArgGYM, on verl

This repository trains Qwen3.5-4B with GRPO and a LoRA adapter on
[ArgGYM](https://github.com/ieddeveci/ArgGYM)'s formal argumentation tasks. It is
[verl](https://github.com/volcengine/verl) v0.9.1 plus one package, `arggym_verl/`,
and a few small patches to verl. A GSM8K control on Qwen3-0.6B checks the pipeline
on a task GRPO is known to move.

- [docs/arggym/usage.md](docs/arggym/usage.md): configs in detail, other GPU counts,
  outputs and metrics, guards, data, layout, and our patches to verl.
- [docs/arggym/design.md](docs/arggym/design.md): the design, every hyperparameter
  and its source, and the full results.
- [README.verl.md](README.verl.md): verl's own README.

## Checkpoints

LoRA adapters (rank 64, alpha 32) for
[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) at revision `851bf6e`,
from the paper's run. Dev success is over 96 ArgGYM problems x 8 samples; the base
model scores 0.309.

| Step | Dev | Adapter |
|---|---|---|
| 60 | 0.527 | [bdsaglam/Qwen3.5-4B-ArgGYM-GRPO-step60-LoRA](https://huggingface.co/bdsaglam/Qwen3.5-4B-ArgGYM-GRPO-step60-LoRA) |
| 80 | 0.569 | [bdsaglam/Qwen3.5-4B-ArgGYM-GRPO-step80-LoRA](https://huggingface.co/bdsaglam/Qwen3.5-4B-ArgGYM-GRPO-step80-LoRA) |
| 100 | 0.539 | [bdsaglam/Qwen3.5-4B-ArgGYM-GRPO-step100-LoRA](https://huggingface.co/bdsaglam/Qwen3.5-4B-ArgGYM-GRPO-step100-LoRA) |
| **140** (the paper's) | **0.613** | [bdsaglam/Qwen3.5-4B-ArgGYM-GRPO-step140-LoRA](https://huggingface.co/bdsaglam/Qwen3.5-4B-ArgGYM-GRPO-step140-LoRA) |
| 200 (last) | 0.645 | [bdsaglam/Qwen3.5-4B-ArgGYM-GRPO-step200-LoRA](https://huggingface.co/bdsaglam/Qwen3.5-4B-ArgGYM-GRPO-step200-LoRA) |

Each model card gives the training recipe, results and loading code. To serve one
with vLLM, start the base with `--enable-lora --max-lora-rank 64` and register the
adapter with `--lora-modules`. Sample with the base model's thinking-mode settings
(temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5) and allow 40k-65k
output tokens.

## Results

Dev success of the paper's run (steps 1-100, then a continuation to 200), with the
model card's sampling and a 65,536-token cap:

| Step | Base | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 | 180 | 200 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dev | 0.309 | 0.418 | 0.447 | 0.527 | 0.569 | 0.539 | 0.572 | **0.613** | 0.611 | 0.641 | 0.645 |

Step 140 gains +0.31 over the base model (paired bootstrap 95% CI +0.25 to +0.37).
Per-task results, the GSM8K control and the earlier runs are in design.md, "Results".

## Requirements

- Linux with NVIDIA GPUs: 3 for ArgGYM, 2 for GSM8K. Tested on A100 80GB.
- An NVIDIA driver for CUDA 13.0 (580 or newer); the venv brings its own CUDA runtime.
- [uv](https://docs.astral.sh/uv/) and git, and network access to GitHub and the
  Hugging Face Hub (no token needed).
- Disk: about 21 GB for the venv and models, plus about 1.5 GB per ArgGYM checkpoint.

Pinned versions are in usage.md, "Requirements", and `scripts/arggym/setup.sh`.

## Quickstart

```bash
git clone https://github.com/bdsaglam/arggym-verl && cd arggym-verl
scripts/arggym/setup.sh           # the venv and the two pinned models
scripts/arggym/prepare_data.sh    # the parquet files under data/
CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q arggym_verl/tests   # CPU tests, about a minute

# The paper's run: steps 1-100, then the continuation from step 100 to 200.
CUDA_VISIBLE_DEVICES=0,1,2 scripts/arggym/run.sh arggym-qwen3.5-4b-v3-lr3e-5-matched
scripts/arggym/seed_from.sh outputs/arggym-qwen3.5-4b-v3-lr3e-5-matched \
    outputs/arggym-qwen3.5-4b-v3-lr3e-5-matched-cont 100
CUDA_VISIBLE_DEVICES=0,1,2 scripts/arggym/run.sh arggym-qwen3.5-4b-v3-lr3e-5-matched-cont

# A checkpoint as a PEFT adapter and a merged bf16 model (CPU only).
.venv/bin/python scripts/arggym/export_lora.py \
    --run outputs/arggym-qwen3.5-4b-v3-lr3e-5-matched-cont --step 140 --no-verify
```

`run.sh` takes a config name under `arggym_verl/config/` and any Hydra overrides
after it; a run with the same `RUN` resumes from its latest checkpoint. On A100s an
ArgGYM step takes about 75 minutes, so 100 steps take about 6 days; start runs in
tmux. Outputs land in `outputs/<RUN>/`. usage.md covers relaunching after a crash,
the GSM8K control, monitoring, and other GPU counts.

## Configs

| Config | Model | GPUs | Steps | What it is |
|---|---|---|---|---|
| `gsm8k-hard-qwen3-0.6b` | Qwen3-0.6B | 2 | 200 | the pipeline control |
| `arggym-qwen3.5-4b` | Qwen3.5-4B | 3 | 100 | first ArgGYM recipe, levels 1-8 |
| `arggym-qwen3.5-4b-v3` | Qwen3.5-4B | 3 | 100 | levels 1-12, partial credit, reworked curriculum |
| `arggym-qwen3.5-4b-v3-lr3e-5-matched` | Qwen3.5-4B | 3 | 1-100 | the paper's run: v3 at LR 3e-5, trainer log-probs matched to the sampler |
| `arggym-qwen3.5-4b-v3-lr3e-5-matched-cont` | Qwen3.5-4B | 3 | 101-200 | its continuation at LR 1e-5 |

usage.md, "What the configs set", gives every setting.

## License

Apache-2.0, as verl ([LICENSE](LICENSE), [Notice.txt](Notice.txt)). The files in
`arggym_verl/vendor/` are ArgGYM's, under the MIT license in
[arggym_verl/vendor/LICENSE](arggym_verl/vendor/LICENSE).
