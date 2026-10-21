import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# ArgGYM's code at the commit scripts/arggym/setup.sh checks out (scripts/arggym/arggym-rev).
ARGGYM_ROOT = Path(os.environ.get("ARGGYM_ROOT", ROOT / ".arggym"))
# rl.gsm8k_screen's output for Qwen3-0.6B (8 samples per GSM8K question), written by
# ArgGYM at 79d9fe2 (branch rl-gsm8k-control) and copied here unchanged.
SCREEN = ROOT / "data/gsm8k-screen/qwen3-0.6b-c1899de"
TRL_GSM8K = ARGGYM_ROOT / "rl/configs/gsm8k-hard-qwen3-0.6b-adapter.yaml"
# Experiment 003's config, and a byte-identical copy of the rl.pool output its run reads
# (pools/0df63cbebe86). dev.jsonl and summary.json are committed; train.jsonl (60 MB) is
# copied locally (README, "Data"), and the tests that need it fail without it.
TRL_ARGGYM = ARGGYM_ROOT / "rl/configs/qwen3.5-4b-003.yaml"
ARGGYM_POOL_ROOT = ROOT / "data/arggym-pool"
QWEN35_4B = Path.home() / (".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/"
                           "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
QWEN3_06B = Path.home() / (".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/"
                           "c1899de289a04d12100db370d81485cdf75e47ca")

os.environ.setdefault("ARGGYM_VERL_ROOT", str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for p in (str(ARGGYM_ROOT), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
