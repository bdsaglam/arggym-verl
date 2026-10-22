# Vendored from ArgGYM evals/score.py (see arggym_verl/vendor/__init__.py; MIT, ./LICENSE).
# Kept: `answer_of` and `score_one`, the per-completion scorer every ArgGYM eval run
# uses. Removed: run aggregation, latency and coverage statistics, the CLI.
# Changed: `values.decode` is imported from `evals_values` as `decode`.
"""Score one completion against its frozen row, as ArgGYM's eval harness does."""
from __future__ import annotations

from typing import Any, Dict, Optional

import arggym

from arggym_verl.vendor.evals_prompt import region
from arggym_verl.vendor.evals_values import decode


def answer_of(gen: Dict[str, Any], template: Optional[str]) -> Dict[str, Any]:
    """The text to score, and what had to be done to find it.

    The completion is read first. Only if it holds no answer region is the
    reasoning trace consulted, because a model that reaches an answer mid-thought
    and never submits it has done something different from one that submits, and
    the flag says which. The fallback never changes a score that a submitted
    answer would have got; it only rescues one that was never submitted.
    """
    body, found = region(gen.get("completion") or "", template)
    if found or not gen.get("reasoning"):
        return {"answer": body, "no_answer_region": not found, "answer_in_cot": False}
    merged, found_cot = region(f"{gen['reasoning']}\n{gen.get('completion') or ''}",
                               template)
    return {"answer": merged if found_cot else body,
            "no_answer_region": not found_cot, "answer_in_cot": found_cot}


def score_one(gen: Dict[str, Any], row: Dict[str, Any],
              template: Optional[str]) -> Dict[str, Any]:
    """One scored record. An API error is never a zero."""
    out: Dict[str, Any] = {
        "id": gen["id"], "task": gen["task"], "level": gen["level"],
        "ordering": gen["ordering"], "api_error": gen.get("error"),
        # What kind of failure it was, as `client.py` labelled it at the moment
        # it happened. Absent on generations made before that label existed,
        # which `_error_kinds` reports as `unclassified` rather than guessing
        # the kind back out of the message.
        "api_error_kind": gen.get("error_kind"),
        "truncated": bool(gen.get("truncated")),
        "completion_tokens": (gen.get("usage") or {}).get("completion_tokens"),
        "latency_s": gen.get("latency_s"),
        # The slowest single request, which is what `timeout_s` bounds, and how
        # many of this row's requests expired. The second is the only record of
        # a row that hit the wall and then answered.
        "attempt_latency_s": gen.get("attempt_latency_s"),
        "requests_timed_out": gen.get("requests_timed_out"),
        "attempts": gen.get("attempts"),
    }
    # Every documented field is present on every record, whatever happened.
    # `samples.jsonl` is read outside this process, and a reader that has to
    # guess whether a missing key means false or means "not applicable" is
    # being invited to guess wrong.
    out.update(scorer_refused=False, zero_with_region=False)
    if gen.get("error"):
        # No score, no success, no zero. The item was not measured.
        out.update(score=None, success=None, reason="api_error",
                   no_answer_region=None, answer_in_cot=None)
        return out

    by_value = gen.get("value") is not None
    if by_value:
        # A solver that produced the answer as a value did its own parsing, so
        # nothing is extracted and no fence was ever asked for.
        out.update(no_answer_region=False, answer_in_cot=False)
    else:
        found = answer_of(gen, template)
        out.update(no_answer_region=found["no_answer_region"],
                   answer_in_cot=found["answer_in_cot"])

    try:
        result = (arggym.score_row_value(decode(gen["value"], row), row)
                  if by_value
                  else arggym.score_row(found["answer"], row))
    except Exception as e:  # noqa: BLE001
        # `score_row` refuses a row it cannot score -- a mismatched engine
        # version, a missing field. That is a fault in the taskset or the
        # install, not an answer worth zero, so it is flagged rather than
        # averaged. The previous harness caught this with a broad except, called
        # it 0.0, and published a cell of forty items scoring exactly 0.000.
        out.update(score=None, success=None, reason=f"{type(e).__name__}: {e}",
                   scorer_refused=True)
        return out

    out.update(score=result.score, success=result.success, reason=result.reason,
               diagnostics=result.diagnostics, scorer_refused=False)
    # A well-formed answer that scores zero is the interesting case: either the
    # model reasoned wrongly, which is a result, or the scorer is wrong, which
    # is a bug. Counting them is how the second gets noticed.
    out["zero_with_region"] = (result.score == 0.0 and not out["no_answer_region"])
    return out

