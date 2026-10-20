import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARGGYM_ROOT = Path(os.environ.get("ARGGYM_ROOT", "/home/baris/repos/ArgGYM2"))
SCREEN = ARGGYM_ROOT / (".claude/worktrees/agent-a671d3be36862eff3/outputs/rl/gsm8k-screen/"
                        "qwen3-0.6b-c1899de")
TRL_GSM8K = ARGGYM_ROOT / "rl/configs/gsm8k-hard-qwen3-0.6b-adapter.yaml"
QWEN3_06B = Path.home() / (".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/"
                           "c1899de289a04d12100db370d81485cdf75e47ca")

os.environ.setdefault("ARGGYM_VERL_ROOT", str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for p in (str(ARGGYM_ROOT), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
