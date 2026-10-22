"""Write a task's training and dev splits as the parquet files verl reads.

    python -m arggym_verl.data arggym_verl/tasks/gsm8k-hard.yaml --out data/gsm8k-hard
    python -m arggym_verl.data arggym_verl/tasks/arggym-003.yaml --out data/arggym-qwen3.5-4b

scripts/arggym/prepare_data.sh runs both.

The first argument is a task config (`arggym_verl/tasks/`), the `task`, `data`,
`prompt` and `train_cells` blocks of the ArgGYM TRL run it reproduces; `key=value`
arguments override it. The rows come from ArgGYM's selection code, vendored in
`arggym_verl.vendor`: `rl_gsm8k.load_splits` for `task: gsm8k`, `rl_pool.load_split`
for ArgGYM, with the training rows restricted to `train_cells` by `rl_pool.restrict`.
An ArgGYM pool that is not there yet is built first (`rl_pool.build`: the `arggym`
library generates and freezes the rows from fixed seeds, a few minutes on 16 cores).

Each row carries `agent_name`, `train` or `dev`, which picks the agent loop and so
the sampling overrides of that split (`arggym_verl/config/agent_loops-*.yaml`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from arggym_verl.vendor import rl_config

ROOT = Path(__file__).resolve().parents[1]

#: verl's `data_source` per task; dev metrics are logged as `val-core/<data_source>/acc/mean@N`.
DATA_SOURCE = {"gsm8k": "gsm8k", "arggym": "arggym"}


def gsm8k_rows(cfg: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    from arggym_verl.vendor import rl_gsm8k as gsm8k

    data = cfg["data"]
    if data.get("screen"):
        data = {**data, "screen": {**data["screen"], "path": str(ROOT / data["screen"]["path"])}}
    train, dev = gsm8k.load_splits(data)
    out = {}
    for split, rows in (("train", train), ("dev", dev)):
        out[split] = [
            {"data_source": DATA_SOURCE["gsm8k"], "prompt": r["prompt"], "agent_name": split,
             "reward_model": {"style": "rule", "ground_truth": repr(r["gold"])},
             "extra_info": {"index": i, "id": r["id"], "split": split, "gold": r["gold"]}}
            for i, r in enumerate(gsm8k.trl_rows(rows))]
    return out


def arggym_rows(cfg: Dict[str, Any], workers: int | None = None) -> Dict[str, List[Dict[str, Any]]]:
    from arggym.core.freeze import usable_cores

    from arggym_verl.vendor import rl_pool as pool

    if not all((pool.pool_dir(cfg) / f"{s}.jsonl").exists() for s in ("train", "dev")):
        print(f"building the pool in {pool.pool_dir(cfg)}")
        pool.build(cfg, workers or min(64, usable_cores()))
    elicit = pool.elicitation(cfg["prompt"]["elicitation"])
    template = cfg["prompt"]["template"]
    out = {}
    for split in ("train", "dev"):
        rows = pool.load_split(cfg, split)
        if split == "train":
            # Training draws only from the config's `train_cells`; dev keeps every cell.
            rows = pool.restrict(rows, cfg.get("train_cells"))
        rows = pool.trl_rows(rows, template, elicit)
        out[split] = [
            {"data_source": DATA_SOURCE["arggym"], "prompt": r["prompt"], "agent_name": split,
             "reward_model": {"style": "rule", "ground_truth": r["id"]},
             "extra_info": {"index": i, "id": r["id"], "split": split, "task": r["task"],
                            "level": r["level"], "template": template,
                            "row_json": r["row_json"]}}
            for i, r in enumerate(rows)]
    return out


def build(cfg: Dict[str, Any], workers: int | None = None) -> Dict[str, List[Dict[str, Any]]]:
    task = cfg.get("task", "arggym")
    if task == "gsm8k":
        return gsm8k_rows(cfg)
    if task == "arggym":
        return arggym_rows(cfg, workers)
    raise SystemExit(f"task {task!r}: gsm8k or arggym")


def write(splits: Dict[str, List[Dict[str, Any]]], out: Path, manifest: Dict[str, Any]) -> None:
    import pandas as pd

    out.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        pd.DataFrame(rows).to_parquet(out / f"{split}.parquet", index=False)
    manifest = {**manifest, **{f"n_{s}": len(r) for s, r in splits.items()},
                "ids": {s: [r["extra_info"]["id"] for r in rows] for s, rows in splits.items()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    from importlib.metadata import version

    from arggym_verl import vendor

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="a task config under arggym_verl/tasks/")
    p.add_argument("overrides", nargs="*", help="key.sub=value")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--workers", type=int, default=None, help="pool build processes")
    a = p.parse_args(argv)
    cfg = rl_config.load(a.config, a.overrides)
    splits = build(cfg, a.workers)
    config = Path(a.config).resolve()
    write(splits, a.out, {
        "config": str(config.relative_to(ROOT)) if config.is_relative_to(ROOT) else str(config),
        "overrides": a.overrides, "data": cfg["data"], "train_cells": cfg.get("train_cells"),
        "vendored_from": vendor.SOURCE_COMMIT, "arggym_version": version("arggym")})
    print(json.dumps({s: len(r) for s, r in splits.items()}), "->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
