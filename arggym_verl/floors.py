"""The partial-credit floors of `reward.mode: partial`: what an answer that reads nothing scores.

    python -m arggym_verl.floors arggym_verl/tasks/arggym-003.yaml --out data/arggym-reward/floors-0df63cbebe86.json

A floor is ArgGYM's own (`arggym.floors`, the baseline its eval reports beside each
task): the best of a fixed set of uninformed answers, such as every claim given one
status, the theory copied back, a bare status line or nothing, each scored by the
eval's scorer. A strategy is chosen once per task and may read a row only to learn
which coordinates it asks about.

The floor is measured per (task, level) on the task config's whole training pool
(100 rows a cell). A fitted strategy (`semantics_query`'s one status per
semantics) is fitted on the task's rows and measured on each level's, as
`arggym.floors` asks of a subgroup. The strategy that wins is picked per level:
across levels the best constant differs, and a policy can learn a cell's constant.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Sequence

from arggym_verl.vendor import rl_config

ROOT = Path(__file__).resolve().parents[1]


def task_floors(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One task's floor over all its rows and per level."""
    from arggym.core.floors import floor_for

    value, best, *_ = floor_for(rows)
    out = {"floor": value, "strategy": best, "n": len(rows), "levels": {}}
    for level in sorted({r["metadata"]["level"] for r in rows}):
        sub = [r for r in rows if r["metadata"]["level"] == level]
        v, b, *_ = floor_for(sub, rows)
        out["levels"][level] = {"floor": v, "strategy": b, "n": len(sub)}
    return out


def measure(rows: Sequence[Dict[str, Any]], workers: int) -> Dict[str, Dict[str, Any]]:
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    tasks = sorted(by_task)
    with ProcessPoolExecutor(workers) as ex:
        return dict(zip(tasks, ex.map(task_floors, [by_task[t] for t in tasks])))


def main(argv: Sequence[str] | None = None) -> int:
    from importlib.metadata import version

    import arggym
    from arggym.core.freeze import SCORING_VERSION

    from arggym_verl.vendor import rl_pool

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="a task config under arggym_verl/tasks/")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args(argv)
    cfg = rl_config.load(a.config)
    rows = rl_pool.load_split(cfg, "train")
    by_task = measure(rows, a.workers)
    pool = rl_pool.pool_dir(cfg).resolve().relative_to(ROOT)
    out = {
        "source": (f"arggym.floors (FLOORS_VERSION {arggym.FLOORS_VERSION}) on every row of "
                   f"{pool}/train.jsonl, the pool of {a.config}: per (task, level), the best "
                   f"uninformed strategy's mean score, a fitted map fitted per task; "
                   f"`python -m arggym_verl.floors {a.config} --out <this file>`"),
        "arggym_version": version("arggym"), "scoring_version": SCORING_VERSION,
        "floors_version": arggym.FLOORS_VERSION,
        "by_task_level": {f"{t}|L{lv}": cell for t, f in by_task.items()
                          for lv, cell in f["levels"].items()},
        "by_task": {t: {k: f[k] for k in ("floor", "strategy", "n")} for t, f in by_task.items()},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    for t, f in by_task.items():
        print(f"{t:26s} {f['floor']:.4f} {f['strategy']:18s} "
              + " ".join(f"L{lv}={c['floor']:.3f}" for lv, c in f["levels"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
