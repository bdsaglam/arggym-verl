# Vendored from ArgGYM evals/taskset.py (see arggym_verl/vendor/__init__.py; MIT, ./LICENSE).
# Kept: `load` and `check`. Removed: `select`.
"""Reading a frozen taskset, and slicing one without rebuilding it.

A taskset is a JSONL whose first line is a manifest and whose remaining lines are
rows (`arggym/core/freeze.py:217`). Everything a scorer needs is on the row, so
nothing here imports a generator.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import arggym


class TasksetCorrupt(SystemExit):
    """A taskset file does not hold what its manifest says it holds."""


def load(path: str, verify: bool = True) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """The manifest and the rows.

    The hash is recomputed from the rows rather than read out of the manifest.
    Comparing the manifest against a copy of itself answers "do these two
    strings match", not "are these the questions that hash names": a file
    truncated to eight lines loads seven rows, passes a string comparison, and
    gets a full taskset's hash printed beside numbers measured on seven items.
    Blake2b over a few thousand rows costs milliseconds.

    A file with no manifest line is refused, because a taskset with no hash
    disables every check downstream at once and does it quietly: `score.py`
    compares the run's `None` against the file's `None`, finds them equal, and
    scores completions against whatever gold the file now holds. Row ids are
    stable across regenerations, so the unknown-id check cannot catch it either.
    `verify=False` reads such a file deliberately.
    """
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        first = json.loads(f.readline())
        manifest = first.get("__manifest__")
        if manifest is None:
            manifest, _ = {}, rows.append(first)
        rows.extend(json.loads(line) for line in f if line.strip())
    if verify:
        if not manifest:
            raise TasksetCorrupt(
                f"{path} has no manifest line, so it names no taskset_hash and "
                f"nothing downstream can tell it from the taskset a run was "
                f"actually generated against. Freeze it with `arggym freeze`, "
                f"or load it with verify=False and accept that its scores name "
                f"no questions.")
        check(path, manifest, rows)
    return manifest, rows


def check(path: str, manifest: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    """Refuse a file whose contents do not match its own manifest."""
    want_n = manifest.get("n_items")
    if want_n is not None and want_n != len(rows):
        raise TasksetCorrupt(
            f"{path} holds {len(rows)} rows and its manifest says {want_n}. A "
            f"short taskset scored against a full manifest publishes the whole "
            f"grid's identity beside a subset's numbers.")
    want = manifest.get("taskset_hash")
    if want is None:
        return
    got = arggym.taskset_hash(rows)
    if got != want:
        raise TasksetCorrupt(
            f"{path} hashes to {got} and its manifest claims {want}. The rows "
            f"have been edited, reordered or truncated since the freeze, so the "
            f"hash a report would cite names questions this file does not hold.")

