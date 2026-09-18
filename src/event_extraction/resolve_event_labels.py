"""
src/event_extraction/resolve_event_labels.py

Event label resolver (Stage 2c).

Fuses rule-based labels (event_type_rule) with LLM semantic labels
(event_type_llm) to produce final event_type and claim_status.

Fusion logic (conservative):
  - Non-claim rows (no claim_id) → None always
  - delegate_subtask → rule wins always (structural, not semantic)
  - propose_claim with no parents → rule wins (unambiguous)
  - Rule and LLM agree → use that label
  - LLM has a label with confidence >= threshold → LLM wins for semantic cases
  - LLM is None or low confidence → rule wins

The LLM only overrides on semantic claim events (merge/revise/contradict/endorse).
It never overrides delegation or structural graph decisions.

Writes to each row:
  event_type        — final resolved label
  claim_status      — derived from event_type
  event_type_source — "rule" | "llm" | "agreed" | "none"

Usage:
    from event_extraction.resolve_event_labels import resolve_event_labels

    rows = annotate_event_types(rows)
    rows = annotate_event_types_llm(rows, llm)
    rows = resolve_event_labels(rows)
    subtask_tree, claim_dag, cascades = build_all(rows)
"""

from __future__ import annotations

from typing import Dict, List, Optional

# Claim status mapping (mirrors event_extractor.EVENT_TO_CLAIM_STATUS)
_EVENT_TO_CLAIM_STATUS: Dict[str, Optional[str]] = {
    "propose_claim":    "proposed",
    "revise_claim":     "revised",
    "contradict_claim": "contradicted",
    "merge_claims":     "merged",
    "endorse_claim":    "endorsed",
    "delegate_subtask": None,
}

# Semantic events the LLM classifier is trusted to disambiguate
_LLM_ELIGIBLE = {"merge_claims", "revise_claim", "contradict_claim", "endorse_claim"}

# Minimum LLM confidence to override the rule label
_CONFIDENCE_THRESHOLD = 0.6


def resolve_event_type(row: dict) -> tuple[Optional[str], str]:
    """
    Resolve the final event_type for one row.

    Returns (event_type, source) where source is one of:
      "none"    — no claim_id, not a claim event
      "rule"    — rule label used
      "llm"     — LLM label overrode rule
      "agreed"  — both agreed on same label
    """
    claim_id = row.get("claim_id")
    rule     = row.get("event_type_rule")
    llm      = row.get("event_type_llm")
    conf     = row.get("event_type_llm_confidence")
    parents  = row.get("parent_claim_ids", []) or []

    # Non-claim rows
    if not claim_id:
        return None, "none"

    # Delegation is always structural — rule wins
    if rule == "delegate_subtask":
        return rule, "rule"

    # propose_claim with no parents is unambiguous — rule wins
    if rule == "propose_claim" and not parents:
        return rule, "rule"

    # No LLM label available — fall back to rule
    if llm is None:
        return rule, "rule"

    # Rule and LLM agree
    if rule == llm:
        return rule, "agreed"

    # LLM overrides rule for semantic claim events when confidence is sufficient.
    # conf=None (missing/failed) must NOT override — treat as low confidence.
    if rule in _LLM_ELIGIBLE and conf is not None and conf >= _CONFIDENCE_THRESHOLD:
        return llm, "llm"

    # Default: rule wins
    return rule, "rule"


def resolve_event_labels(rows: List[dict]) -> List[dict]:
    """
    Write final event_type, claim_status, and event_type_source to all rows.

    Must be called after both annotate_event_types() and
    annotate_event_types_llm(). Safe to call without the LLM pass
    (will use rule labels for all rows in that case).
    """
    from collections import Counter
    for row in rows:
        event_type, source = resolve_event_type(row)
        row["event_type"]        = event_type
        row["claim_status"]      = (
            _EVENT_TO_CLAIM_STATUS.get(event_type)
            if event_type is not None else None
        )
        row["event_type_source"] = source

    # Observability: log label source distribution so callers can verify
    # that the LLM classifier is actually being used (not silently disabled).
    sources = Counter(
        r.get("event_type_source") for r in rows if r.get("event_type_source")
    )
    types   = Counter(
        r.get("event_type") for r in rows if r.get("event_type")
    )
    import logging
    _log = logging.getLogger(__name__)
    _log.debug("event_type_source distribution: %s", dict(sources))
    _log.debug("event_type distribution:        %s", dict(types))

    return rows