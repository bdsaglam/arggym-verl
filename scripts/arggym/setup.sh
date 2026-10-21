#!/usr/bin/env bash
# The venv: verl v0.9.1's own uv.lock, plus what arggym_verl needs beside it.
#
# The lock's flash-attn URL (wheelhouse release `flash-attention-v2.8.3`) is gone
# (404), and the wheelhouse's index now serves a torch-2.13 build of the same
# version that fails to load against the lock's torch 2.11 (undefined symbol
# c10::ValueError). So flash-attn is skipped in the sync and installed from the
# wheelhouse's torch-2.11 release.
#
# ArgGYM's code (data selection, prompts, scorers) is cloned into .arggym at the
# commit in scripts/arggym/arggym-rev: experiment 003's commit on branch rl-003,
# which is not in the main ArgGYM checkout's branch. A pinned clone does not move
# when that checkout changes branch or a worktree is removed. ARGGYM_SRC is the
# repository to clone from.
set -euo pipefail
cd "$(dirname "$0")/../.."
uv sync --frozen --extra fsdp --extra vllm --no-install-package flash-attn
uv pip install --python .venv/bin/python --no-deps \
  "flash-attn @ https://github.com/verl-project/verl-wheelhouse/releases/download/flash-attention-v2.8.3-py3.12-torch2.11.0/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"
# ArgGYM's scorer dependency, and trackio for local metrics.
uv pip install --python .venv/bin/python python-argumentation==2.0.2 trackio==0.42.0
# ArgGYM's code at the pinned commit.
ARGGYM_SRC=${ARGGYM_SRC:-/home/baris/repos/ArgGYM2}
[ -d .arggym ] || git clone --quiet --no-checkout "$ARGGYM_SRC" .arggym
git -C .arggym fetch --quiet origin
git -C .arggym checkout --quiet --detach "$(cat scripts/arggym/arggym-rev)"
