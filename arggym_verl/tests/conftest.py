import json
import os
import sys
from pathlib import Path

from huggingface_hub import constants as hf_constants

ROOT = Path(__file__).resolve().parents[2]
# Golden outputs of ArgGYM's own code at the commit arggym_verl/vendor is copied from;
# each file names its source.
FIXTURES = Path(__file__).resolve().parent / "fixtures"
# The task configs `python -m arggym_verl.data` reads.
TASK_GSM8K = ROOT / "arggym_verl/tasks/gsm8k-hard.yaml"
TASK_ARGGYM = ROOT / "arggym_verl/tasks/arggym-003.yaml"
# Experiment 003's pool. dev.jsonl and summary.json are committed; train.jsonl (60 MB) is
# built by scripts/arggym/prepare_data.sh, and the tests that read it fail without it.
ARGGYM_POOL_ROOT = ROOT / "data/arggym-pool"
# The pinned snapshots in the Hugging Face cache (scripts/arggym/setup.sh downloads them).
HF_HUB_CACHE = Path(hf_constants.HF_HUB_CACHE)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE)
QWEN35_4B = HF_HUB_CACHE / "models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
QWEN3_06B = HF_HUB_CACHE / "models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

os.environ.setdefault("ARGGYM_VERL_ROOT", str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def fixture(name: str):
    return json.loads((FIXTURES / name).read_text())
