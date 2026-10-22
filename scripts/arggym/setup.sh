#!/usr/bin/env bash
# Everything a run needs besides the data: the venv and the two models.
#
#   scripts/arggym/setup.sh
#
# The venv is verl v0.9.1's own uv.lock on Python 3.12, plus what arggym_verl needs
# beside it. The lock's flash-attn URL (wheelhouse release `flash-attention-v2.8.3`)
# is gone (404), and the wheelhouse's index now serves a torch-2.13 build of the same
# version that fails to load against the lock's torch 2.11 (undefined symbol
# c10::ValueError). So flash-attn is skipped in the sync and installed from the
# wheelhouse's torch-2.11 release.
#
# requirements.txt adds ArgGYM's library (`arggym`, from git at a pinned commit; it is
# not on PyPI) and trackio. ArgGYM's training glue is vendored in arggym_verl/vendor/.
#
# The models are the Hub revisions the configs pin, downloaded into the Hugging Face
# cache (HF_HUB_CACHE, or HF_HOME/hub, default ~/.cache/huggingface/hub).
set -euo pipefail
cd "$(dirname "$0")/../.."

uv sync --frozen --python 3.12 --extra fsdp --extra vllm --no-install-package flash-attn
uv pip install --python .venv/bin/python --no-deps \
  "flash-attn @ https://github.com/verl-project/verl-wheelhouse/releases/download/flash-attention-v2.8.3-py3.12-torch2.11.0/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"
# ArgGYM's repository stores archived eval outputs in Git LFS, and at least one LFS
# object is missing on the remote, so checking out the commit with git-lfs installed
# fails. The library needs none of them.
GIT_LFS_SKIP_SMUDGE=1 uv pip install --python .venv/bin/python -r scripts/arggym/requirements.txt

# The pinned model snapshots: the GSM8K control's and the ArgGYM run's.
.venv/bin/hf download Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca
.venv/bin/hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
