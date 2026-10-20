#!/usr/bin/env bash
# The venv: verl v0.9.1's own uv.lock, plus what arggym_verl needs beside it.
#
# The lock's flash-attn URL (wheelhouse release `flash-attention-v2.8.3`) is gone
# (404), and the wheelhouse's index now serves a torch-2.13 build of the same
# version that fails to load against the lock's torch 2.11 (undefined symbol
# c10::ValueError). So flash-attn is skipped in the sync and installed from the
# wheelhouse's torch-2.11 release.
set -euo pipefail
cd "$(dirname "$0")/../.."
uv sync --frozen --extra fsdp --extra vllm --no-install-package flash-attn
uv pip install --python .venv/bin/python --no-deps \
  "flash-attn @ https://github.com/verl-project/verl-wheelhouse/releases/download/flash-attention-v2.8.3-py3.12-torch2.11.0/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"
# ArgGYM's scorer dependency, and trackio for local metrics.
uv pip install --python .venv/bin/python python-argumentation==2.0.2 trackio==0.42.0
