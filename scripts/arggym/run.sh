#!/usr/bin/env bash
# One run of an arggym_verl config: verl's main_ppo with the served-policy check (arggym_verl.main).
#
#   scripts/arggym/run.sh gsm8k-hard-qwen3-0.6b [hydra overrides ...]
#   RUN=smoke scripts/arggym/run.sh gsm8k-hard-qwen3-0.6b trainer.total_training_steps=3
#
# Outputs go to outputs/$RUN (RUN defaults to the config name): checkpoints,
# rollouts, dev generations, metrics.jsonl, the reward log, and the driver log.
# Metrics also go to trackio under outputs/trackio (`trackio show --project arggym-verl`).
set -euo pipefail

CONFIG=${1:?config name under arggym_verl/config}
shift
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUN=${RUN:-$CONFIG}
OUT="$ROOT/outputs/$RUN"
mkdir -p "$OUT"

export ARGGYM_VERL_ROOT="$ROOT"
export ARGGYM_ROOT=${ARGGYM_ROOT:-/home/baris/repos/ArgGYM2}
# ArgGYM's rl/ and evals/ (data selection, prompts, scorers) are imported from its checkout.
export PYTHONPATH="$ROOT:$ARGGYM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TRACKIO_DIR="$ROOT/outputs/trackio"
export VERL_FILE_LOGGER_PATH="$OUT/metrics.jsonl"
export ARGGYM_VERL_REWARD_LOG="$OUT/rewards"
export TOKENIZERS_PARALLELISM=false
# As the TRL server (rl/launch.py): vLLM's own sampler, not FlashInfer's, whose
# top-k/top-p kernel JIT-compiles with the system nvcc, too old on this machine.
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$ROOT/.venv/bin:$PATH"

{
  echo "verl $(git -C "$ROOT" rev-parse HEAD)"
  echo "arggym $(git -C "$ARGGYM_ROOT" rev-parse HEAD) ($(git -C "$ARGGYM_ROOT" branch --show-current))"
} > "$OUT/versions.txt"

"$ROOT/.venv/bin/python" -m arggym_verl.main \
  --config-path="$ROOT/arggym_verl/config" --config-name="$CONFIG" \
  "hydra.searchpath=[pkg://verl.trainer.config]" \
  trainer.experiment_name="$RUN" \
  ray_kwargs.ray_init.runtime_env.py_executable="$ROOT/.venv/bin/python" \
  "$@" 2>&1 | tee "$OUT/driver.log"
