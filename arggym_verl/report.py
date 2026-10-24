"""A run's dev curve, the selected checkpoint, and the training metrics verl does not log.

    python -m arggym_verl.report outputs/<run>

Reads `metrics.jsonl` (verl's file logger) and `rewards/*.jsonl` (one line per
scored completion, `arggym_verl.reward`). Writes `selection.json`: the
checkpoint with the best dev success, beside the step-0 score and whether it
beats it, as the TRL pipeline's `selection.json` does; the trainer writes the
same file after every dev evaluation (`arggym_verl.trainer.DevSelection`), and
this command, run during a dev evaluation, misses that step. Dev success is picked as
the maximum of many evaluations; quote the selected checkpoint's eval-taskset
score, not this one.

Training rows are grouped by the weight version vLLM held when the completion
finished (`version`). A group (one prompt's completions, one `uid`) has no
signal when all its rewards are equal: its advantages are all zero.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_jsonl(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    out = []
    for p in paths:
        out.extend(json.loads(line) for line in p.read_text().splitlines() if line.strip())
    return out


def dev_curve(metrics: List[Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
    """step -> dev metrics, from the `val-core/` and `val-aux/` keys verl logs."""
    curve: Dict[int, Dict[str, float]] = {}
    for rec in metrics:
        dev = {k: v for k, v in rec["data"].items() if k.startswith(("val-core/", "val-aux/"))}
        if dev:
            curve.setdefault(rec["step"], {}).update(dev)
    return curve


def select(curve: Dict[int, Dict[str, float]], ckpt_root: Path) -> Dict[str, Any]:
    key = next((k for row in curve.values() for k in row
                if k.startswith("val-core/") and "/acc/mean@" in k), None)
    if key is None:
        return {"metric": None}
    scores = {s: row[key] for s, row in curve.items() if key in row}
    saved = {s: s for s in scores if (ckpt_root / f"global_step_{s}").exists()}
    best = max(saved, key=lambda s: (scores[s], -s)) if saved else None
    base = scores.get(0)
    return {"metric": key, "base_metric": base,
            "best_step": best, "best_metric": scores.get(best),
            "best_checkpoint": str(ckpt_root / f"global_step_{best}") if best else None,
            "improved_over_base": None if best is None or base is None else scores[best] > base,
            "dev": {str(s): scores[s] for s in sorted(scores)}}


def train_by_version(rows: List[Dict[str, Any]]) -> Dict[Any, Dict[str, float]]:
    by_version: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("split") == "train":
            by_version[r.get("version")].append(r)
    out = {}
    for version, rs in sorted(by_version.items(), key=lambda kv: (kv[0] is None, kv[0])):
        groups: Dict[str, List[float]] = defaultdict(list)
        for r in rs:
            groups[r["uid"]].append(r["score"])
        n = len(rs)
        out[version] = {
            "completions": n, "success": sum(r["acc"] for r in rs) / n,
            "truncated": sum(r["truncated"] for r in rs) / n,
            "length": sum(r["length"] for r in rs) / n,
            "groups": len(groups),
            "groups_no_signal": sum(len(set(g)) < 2 for g in groups.values()) / len(groups)}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    a = p.parse_args(argv)
    metrics = read_jsonl([a.run_dir / "metrics.jsonl"])
    selection = select(dev_curve(metrics), a.run_dir / "checkpoints")
    (a.run_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection, indent=2))
    train = train_by_version(read_jsonl(sorted((a.run_dir / "rewards").glob("*.jsonl"))))
    print(f"{'version':>7} {'n':>5} {'success':>8} {'trunc':>6} {'length':>7} {'no_signal':>9}")
    for v, m in train.items():
        print(f"{v!s:>7} {m['completions']:>5} {m['success']:>8.3f} {m['truncated']:>6.3f} "
              f"{m['length']:>7.0f} {m['groups_no_signal']:>9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
