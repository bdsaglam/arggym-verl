# Vendored from ArgGYM evals/values.py (see arggym_verl/vendor/__init__.py; MIT, ./LICENSE).
# Kept: `decode`, which `evals_score.score_one` calls for a solver that answers with a
# value. Text completions, the only kind this pipeline scores, never reach it.
"""JSON back to the value `arggym.score_row_value` expects."""
from __future__ import annotations

from typing import Any, Dict

import arggym


def decode(payload: Any, row: Dict[str, Any]) -> Any:
    """JSON back to the value `score_row_value` expects.

    `ordered_sequence` and `record_list` are already JSON, and a `label_map`
    keyed by strings is too, so those pass through untouched.
    """
    shape = row["metadata"].get("answer_shape")
    if shape == "operation_list":
        return arggym.ops_from_json(payload)
    if shape == "label_map" and isinstance(payload, list):
        return {tuple(k): v for k, v in payload}
    return payload
