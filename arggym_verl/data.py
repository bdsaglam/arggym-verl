"""Write a task's training and dev splits as the parquet files verl reads.

    python -m arggym_verl.data $ARGGYM_ROOT/rl/configs/gsm8k-hard-qwen3-0.6b-adapter.yaml \
        --out data/gsm8k-hard data.screen.path=$PWD/data/gsm8k-screen/qwen3-0.6b-c1899de

The first argument is the TRL run's own config, read by ArgGYM's `rl.config`, and
the rows come from ArgGYM's own selection code: `rl.gsm8k.load_splits` for
`task: gsm8k`, `rl.pool.load_split` for ArgGYM. So both pipelines train and
evaluate on the same ids, with the same prompt messages.

Each row carries `agent_name`, `train` or `dev`, which picks the agent loop and so
the sampling overrides of that split (`arggym_verl/config/agent_loops-*.yaml`).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence

from rl import config as rl_config

#: verl's `data_source` per task; dev metrics are logged as `val-core/<data_source>/acc/mean@N`.
DATA_SOURCE = {"gsm8k": "gsm8k", "arggym": "arggym"}


def gsm8k_rows(cfg: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    from rl import gsm8k

    train, dev = gsm8k.load_splits(cfg["data"])
    out = {}
    for split, rows in (("train", train), ("dev", dev)):
        out[split] = [
            {"data_source": DATA_SOURCE["gsm8k"], "prompt": r["prompt"], "agent_name": split,
             "reward_model": {"style": "rule", "ground_truth": repr(r["gold"])},
             "extra_info": {"index": i, "id": r["id"], "split": split, "gold": r["gold"]}}
            for i, r in enumerate(gsm8k.trl_rows(rows))]
    return out


def arggym_rows(cfg: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    from rl import pool

    elicit = pool.elicitation(cfg["prompt"]["elicitation"])
    template = cfg["prompt"]["template"]
    out = {}
    for split in ("train", "dev"):
        rows = pool.trl_rows(pool.load_split(cfg, split), template, elicit)
        out[split] = [
            {"data_source": DATA_SOURCE["arggym"], "prompt": r["prompt"], "agent_name": split,
             "reward_model": {"style": "rule", "ground_truth": r["id"]},
             "extra_info": {"index": i, "id": r["id"], "split": split, "task": r["task"],
                            "level": r["level"], "template": template,
                            "row_json": r["row_json"]}}
            for i, r in enumerate(rows)]
    return out


def build(cfg: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    task = cfg.get("task", "arggym")
    if task == "gsm8k":
        return gsm8k_rows(cfg)
    if task == "arggym":
        return arggym_rows(cfg)
    raise SystemExit(f"task {task!r}: gsm8k or arggym")


def _git_sha(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write(splits: Dict[str, List[Dict[str, Any]]], out: Path, manifest: Dict[str, Any]) -> None:
    import pandas as pd

    out.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        pd.DataFrame(rows).to_parquet(out / f"{split}.parquet", index=False)
    manifest = {**manifest, **{f"n_{s}": len(r) for s, r in splits.items()},
                "ids": {s: [r["extra_info"]["id"] for r in rows] for s, rows in splits.items()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="the TRL run's config in the ArgGYM checkout")
    p.add_argument("overrides", nargs="*", help="key.sub=value, as rl.launch takes them")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args(argv)
    cfg = rl_config.load(a.config, a.overrides)
    splits = build(cfg)
    write(splits, a.out, {"config": a.config, "overrides": a.overrides, "data": cfg["data"],
                          "arggym_git_sha": _git_sha(rl_config.ROOT)})
    print(json.dumps({s: len(r) for s, r in splits.items()}), "->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
