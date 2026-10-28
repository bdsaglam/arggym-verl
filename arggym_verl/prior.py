"""The v3 run's training cells and curriculum prior, both read as the paid rate.

    python -m arggym_verl.prior observe --out data/arggym-prior/sources/qwen3.5-4b-v2-partial-observed.json \\
        outputs/arggym-qwen3.5-4b-v2:20 outputs/arggym-qwen3.5-4b-partial:34
    python -m arggym_verl.prior build --out data/arggym-prior/qwen3.5-4b-pooled-paid.json

Under `reward.mode: partial` a cell pays more than its exact success p: a finished
wrong answer on a `partial_tasks` task earns w * max(0, (s - b) / (1 - b)), s its
partial score and b the cell's floor. The cell's paid rate is the expected reward,

    q = p + (1 - p) * w * near,

`near` the mean of max(0, (s - b) / (1 - b)) over the cell's wrong answers (0 for a
task outside `partial_tasks`). A cell trains when 0.15 <= q <= 0.85: at 0.15, 16
independent completions all agree with probability 0.07 (arggym-003.yaml gives the
rule at 8). On a partial task the lower bound is 0.12 = w * 0.4 (failures 40% of the
way from floor to full): its all-fail groups still differ in reward, so the
agreement argument behind 0.15 does not bind.

The evidence per (task, level), test-set evals of trained checkpoints excluded:

- L1-8, p: experiment 003's pooled counts (the comments of `tasks/arggym-003.yaml`:
  the base eval on taskset-lite, 002's training groups and dev), plus the v2 and
  partial runs' training groups and their step-20 dev (8 samples a row), read
  from those runs' `rewards/` and `dev/20.jsonl` by `observe` and committed in
  `sources/`. A completion passes when it succeeds and does not truncate (the
  curriculum's rule); a refused one is left out.
- L1-8, near: the mean of two estimates, the base eval's and the runs'. The base
  eval's is max(0, (m - b) / (1 - b)), m the mean score of its wrong answers
  (from the cell's `mean` and `success_rate`) and b the eval's floor; the runs'
  is the mean over their wrong answers of the reward's own term, a truncated one
  counting 0, b the training pool's floor (`data/arggym-reward/`).
- L9-12: the base eval alone (8 samples a cell), p and near.

The prior file carries q as `success_rate`, so the curriculum starts every cell at
the rate it pays and `base_pass_counts` (vendored) reads it unchanged. Its
`n_scored` is 16 for L1-8, one group's worth, though the pooled evidence is
larger: the v2 and partial evidence comes from a policy 20-34 steps into training,
so the prior should yield to the run's first group of the cell, not anchor it.
L9-12 keep the base eval's 8.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
Cell = Tuple[str, int]

TASK_003 = "arggym_verl/tasks/arggym-003.yaml"
#: ArgGYM's results/hf-qwen3.5-4b__xml_tags__cot/metrics.json, the base model's eval on
#: taskset-lite (8 samples a cell, L1-15), copied as is.
BASE = "data/arggym-prior/sources/hf-qwen3.5-4b__xml_tags__cot.metrics.json"
OBSERVED = "data/arggym-prior/sources/qwen3.5-4b-v2-partial-observed.json"
FLOORS_003 = "data/arggym-reward/floors-0df63cbebe86.json"
#: The v3 config's reward (arggym-qwen3.5-4b-v3.yaml); test_config.py checks they agree.
PARTIAL_TASKS = ("status_query", "defeat_diagnosis", "formalization", "perturbation")
PARTIAL_WEIGHT = 0.3
LEVELS = tuple(range(1, 13))
BAND = (0.15, 0.85)
#: The lower bound of q on a partial task: w * 0.4.
NEAR_LOW = 0.12
#: The prior's sample count on L1-8 (module docstring).
PRIOR_N = 16


def key(cell: Cell) -> str:
    return f"{cell[0]}|L{cell[1]}"


def parse_key(k: str) -> Cell:
    task, level = k.split("|L")
    return task, int(level)


# ---------------------------------------------------------------- sources


def pooled_003(path: str = TASK_003) -> Dict[Cell, Tuple[int, int]]:
    """(successes, samples) per cell from the train_cells comments of arggym-003.yaml,
    lines `# task: in L1 8/14, ...;` and their `#   out ...` continuations."""
    out: Dict[Cell, Tuple[int, int]] = {}
    task = None
    for line in (ROOT / path).read_text().splitlines():
        m = re.match(r"\s*# (\w+): in (.*)", line)
        if m:
            task, rest = m.group(1), m.group(2)
        elif task and line.strip().startswith("#"):
            rest = line
        else:
            task = None
            continue
        for mm in re.finditer(r"L(\d+) (\d+)/(\d+)", rest):
            out[(task, int(mm.group(1)))] = (int(mm.group(2)), int(mm.group(3)))
    if len(out) != 96:
        raise ValueError(f"{path}: {len(out)} pooled cells, expected 12 tasks x 8 levels")
    return out


def near_base(m: Dict[str, Any]) -> Optional[float]:
    """max(0, (m - b) / (1 - b)) over an eval cell's wrong answers; None if all passed."""
    sr = m["success_rate"]
    if sr >= 1:
        return None
    wrong = (m["mean"] - sr) / (1 - sr)
    b = m["floor"]
    return max(0.0, (wrong - b) / (1 - b))


def tally(records: Iterable[Dict[str, Any]], floors: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Per cell: scored completions, passes, wrong answers and the sum of their near-miss terms.

    A record is a scored completion with `id` (task/L<level>/...), `acc`, `truncated`,
    `refused` and `partial_score`, as the reward logs it.
    """
    out: Dict[str, Dict[str, float]] = {}
    for r in records:
        if r.get("refused"):
            continue
        task, level = r["id"].split("/")[:2]
        k = f"{task}|{level}"
        c = out.setdefault(k, {"n": 0, "success": 0, "n_wrong": 0, "near_sum": 0.0})
        c["n"] += 1
        if r["acc"] and not r["truncated"]:
            c["success"] += 1
            continue
        c["n_wrong"] += 1
        if not r["truncated"]:
            b = floors[k]
            c["near_sum"] += max(0.0, (float(r["partial_score"]) - b) / (1 - b))
    for c in out.values():
        c["near_sum"] = round(c["near_sum"], 6)
    return dict(sorted(out.items()))


def observe(runs: Sequence[Tuple[Path, int]], floors_path: str = FLOORS_003) -> Dict[str, Any]:
    """The training groups of steps 1..steps and the step-20 dev of each run, tallied per cell."""
    from arggym_verl.report import read_jsonl

    floors = {k: c["floor"] for k, c in
              json.loads((ROOT / floors_path).read_text())["by_task_level"].items()}
    sources = {}
    for run_dir, steps in runs:
        name = run_dir.name
        # Step k is generated by weight version k - 1 (no batch ahead: num_warmup_batches 0).
        train = [r for r in read_jsonl(sorted((run_dir / "rewards").glob("*.jsonl")))
                 if r["split"] == "train" and r["version"] < steps]
        if len({r["version"] for r in train}) != steps:
            raise SystemExit(f"{run_dir}: steps 1-{steps} are not all in rewards/")
        sources[f"{name} train steps 1-{steps}"] = tally(train, floors)
        dev = [{**r, "id": r["gts"]} for r in read_jsonl([run_dir / "dev" / "20.jsonl"])]
        sources[f"{name} dev step 20"] = tally(dev, floors)
    return {"source": ("python -m arggym_verl.prior observe: per (task, level), scored "
                       "completions (refused left out), passes (success, not truncated), "
                       "wrong answers and the sum over them of max(0, (s - b) / (1 - b)), "
                       f"a truncated one 0, b the floor of {floors_path}"),
            "sources": sources}


# ---------------------------------------------------------------- the paid rate


def cell_table(base: Dict[str, Dict[str, Any]], observed: Dict[str, Dict[str, Dict[str, float]]],
               pooled: Dict[Cell, Tuple[int, int]], partial_tasks: Iterable[str] = PARTIAL_TASKS,
               weight: float = PARTIAL_WEIGHT, levels: Iterable[int] = LEVELS
               ) -> Dict[str, Dict[str, Any]]:
    """p, its sample count, near and q per (task, level) (module docstring), and whether
    the cell trains."""
    partial = set(partial_tasks)
    tasks = sorted({parse_key(k)[0] for k in base})
    out = {}
    for task in tasks:
        for level in levels:
            k = key((task, level))
            b = base[k]
            nb = near_base(b)
            if level <= 8:
                s, n = pooled[(task, level)]
                parts = [f"003 {s}/{n}"]
                wrong = near = 0.0
                for name, cells in observed.items():
                    c = cells.get(k)
                    if c:
                        s, n = s + c["success"], n + c["n"]
                        wrong, near = wrong + c["n_wrong"], near + c["near_sum"]
                        parts.append(f"{name} {c['success']}/{c['n']}")
                estimates = [x for x in (nb, near / wrong if wrong else None) if x is not None]
            else:
                n = b["n_scored"]
                s = round(b["success_rate"] * n)
                parts = [f"base {s}/{n}"]
                estimates = [nb] if nb is not None else []
            p = s / n
            nm = sum(estimates) / len(estimates) if task in partial and estimates else 0.0
            q = p + (1 - p) * weight * nm
            low = NEAR_LOW if task in partial else BAND[0]
            out[k] = {"p": p, "successes": s, "samples": n, "near": nm, "q": q,
                      "train": low <= q <= BAND[1], "evidence": ", ".join(parts)}
    return out


def train_cells(table: Dict[str, Dict[str, Any]]) -> Dict[str, List[int]]:
    """{task: [levels]} of the cells that train, every task listed."""
    out: Dict[str, List[int]] = {}
    for k, c in table.items():
        task, level = parse_key(k)
        out.setdefault(task, [])
        if c["train"]:
            out[task].append(level)
    return {t: sorted(lv) for t, lv in sorted(out.items())}


def prior(table: Dict[str, Dict[str, Any]], partial_tasks: Iterable[str] = PARTIAL_TASKS,
          weight: float = PARTIAL_WEIGHT) -> Dict[str, Any]:
    """The curriculum's prior file: `success_rate` q over `n_scored` samples per cell."""
    cells = {}
    for k, c in table.items():
        n = PRIOR_N if parse_key(k)[1] <= 8 else c["samples"]
        cells[k] = {"n_scored": n, "success_rate": c["q"], **{f: c[f] for f in (
            "p", "near", "successes", "samples", "train", "evidence")}}
    return {
        "source": (f"python -m arggym_verl.prior build: the paid rate q = p + (1 - p) * w * near "
                   f"per (task, level) (arggym_verl/prior.py). L1-8 from {TASK_003}'s pooled "
                   f"counts and {OBSERVED}; L9-12 from {BASE} alone, the base model's eval on "
                   f"taskset-lite. No test-set eval of a trained checkpoint enters. n_scored is "
                   f"{PRIOR_N} on L1-8 and the base eval's 8 on L9-12."),
        "partial_tasks": sorted(partial_tasks), "partial_weight": weight,
        "band": list(BAND), "near_low": NEAR_LOW, "by_task_level": cells}


def load_table(partial_tasks: Iterable[str] = PARTIAL_TASKS,
               weight: float = PARTIAL_WEIGHT) -> Dict[str, Dict[str, Any]]:
    """`cell_table` on the committed sources."""
    base = json.loads((ROOT / BASE).read_text())["by_task_level"]
    observed = json.loads((ROOT / OBSERVED).read_text())["sources"]
    return cell_table(base, observed, pooled_003(), partial_tasks, weight)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("observe", help="tally runs' training groups and step-20 dev per cell")
    o.add_argument("runs", nargs="+", help="<run dir>:<steps>")
    o.add_argument("--out", required=True, type=Path)
    b = sub.add_parser("build", help="the cell table, train_cells and the prior file")
    b.add_argument("--out", required=True, type=Path)
    a = p.parse_args(argv)
    if a.cmd == "observe":
        runs = [(Path(r.rsplit(":", 1)[0]), int(r.rsplit(":", 1)[1])) for r in a.runs]
        out = observe(runs)
    else:
        table = load_table()
        out = prior(table)
        print(f"{'cell':30s} {'p':>5} {'n':>4} {'near':>5} {'q':>5} train  evidence")
        for k, c in table.items():
            print(f"{k:30s} {c['p']:5.2f} {c['samples']:4d} {c['near']:5.2f} {c['q']:5.2f} "
                  f"{'*' if c['train'] else ' ':>5}  {c['evidence']}")
        cells = train_cells(table)
        print(f"{sum(map(len, cells.values()))} cells:")
        for t, lv in cells.items():
            print(f"  {t}: {lv}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
