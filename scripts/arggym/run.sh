#!/usr/bin/env bash
# One run of an arggym_verl config: verl's main_ppo with the served-policy check (arggym_verl.main).
#
#   scripts/arggym/run.sh gsm8k-hard-qwen3-0.6b [hydra overrides ...]
#   RUN=smoke scripts/arggym/run.sh gsm8k-hard-qwen3-0.6b trainer.total_training_steps=3
#   CUDA_VISIBLE_DEVICES=0,1,2 scripts/arggym/run.sh arggym-qwen3.5-4b
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

# A run that stopped itself (arggym_verl.trainer.AutoStop) is not resumed by accident.
if [ -e "$OUT/AUTO-STOP" ]; then
  echo "$OUT stopped itself: $(cat "$OUT/AUTO-STOP")" >&2
  echo "Delete $OUT/AUTO-STOP to resume it anyway." >&2
  exit 1
fi
# verl's file logger and tee truncate their files: keep a resumed run's earlier ones,
# which arggym_verl.report and the trainer read back (metrics.<time>.jsonl, oldest first).
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
for f in metrics.jsonl driver.log; do
  if [ -s "$OUT/$f" ]; then mv "$OUT/$f" "$OUT/${f%.*}.$STAMP.${f##*.}"; fi
done

export ARGGYM_VERL_ROOT="$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TRACKIO_DIR="$ROOT/outputs/trackio"
export VERL_FILE_LOGGER_PATH="$OUT/metrics.jsonl"
export ARGGYM_VERL_REWARD_LOG="$OUT/rewards"
export TOKENIZERS_PARALLELISM=false
# As ArgGYM's TRL server (rl/launch.py): vLLM's own sampler, not FlashInfer's, whose
# top-k/top-p kernel JIT-compiles with the system nvcc and fails when that is older
# than the venv's CUDA 13.0.
export VLLM_USE_FLASHINFER_SAMPLER=0
# The configs read the pinned model snapshots from the Hugging Face cache.
export HF_HUB_CACHE=${HF_HUB_CACHE:-$("$ROOT/.venv/bin/python" -c 'from huggingface_hub import constants; print(constants.HF_HUB_CACHE)')}
export PATH="$ROOT/.venv/bin:$PATH"

{
  echo "verl $(git -C "$ROOT" rev-parse HEAD)"
  echo "arggym $("$ROOT/.venv/bin/python" -c 'from importlib.metadata import version; print(version("arggym"))')"
} > "$OUT/versions.txt"

"$ROOT/.venv/bin/python" -m arggym_verl.main \
  --config-path="$ROOT/arggym_verl/config" --config-name="$CONFIG" \
  "hydra.searchpath=[pkg://verl.trainer.config]" \
  trainer.experiment_name="$RUN" \
  ray_kwargs.ray_init.runtime_env.py_executable="$ROOT/.venv/bin/python" \
  "$@" 2>&1 | tee "$OUT/driver.log"
