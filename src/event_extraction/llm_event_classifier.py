"""
src/event_extraction/llm_event_classifier.py

LLM-based semantic event classifier (Stage 2b).

Role: disambiguate the hard semantic cases that rule-based classification
cannot reliably separate:
  - merge_claims vs revise_claim vs contradict_claim on multi-parent rows
  - corrective single-parent rows vs plain endorsements
  - synthesis rows that look like merges but are actually corrections

This classifier is:
  - POST-HOC: reads frozen trace rows, never affects execution
  - NON-PARTICIPATING: not one of the agents in the run
  - CHEAP: only called on rows flagged needs_llm_event_classification=True
  - CONSERVATIVE: its output is fused with rule labels by resolve_event_labels

Label set (same as rule classifier, minus delegate_subtask which stays rule-only):
  propose_claim | endorse_claim | revise_claim | contradict_claim | merge_claims

Usage:
    from event_extraction.llm_event_classifier import annotate_event_types_llm

    rows = annotate_event_types(rows)           # rule pass
    rows = annotate_event_types_llm(rows, llm)  # semantic pass
    rows = resolve_event_labels(rows)           # final labels
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional


SEMANTIC_LABELS = {
    "propose_claim",
    "endorse_claim",
    "revise_claim",
    "contradict_claim",
    "merge_claims",
}

_CLASSIFIER_SYSTEM_PROMPT = """\
You are labeling the semantic role of one claim in a multi-agent reasoning trace.
You must choose exactly one label from:
  propose_claim | endorse_claim | revise_claim | contradict_claim | merge_claims

Definitions:
  propose_claim:    Introduces a new claim without materially depending on parent claims.
  endorse_claim:    Mainly supports or affirms a parent claim without correcting it.
  revise_claim:     Corrects, updates, or resolves an error in one or more parent claims.
                    Includes recomputing a wrong value, selecting the correct answer
                    among conflicting claims, or resolving a disagreement.
  contradict_claim: Explicitly rejects or opposes a parent claim as wrong or incompatible.
  merge_claims:     Combines multiple parent claims into a synthesis without primarily
                    correcting or rejecting them.

Rules:
  - If the claim resolves a disagreement between parent claims by recomputing or
    selecting the correct value → revise_claim.
  - If the claim mainly combines compatible parent claims → merge_claims.
  - If the claim mainly rejects a parent claim → contradict_claim.
  - If the claim mainly agrees with and repeats a parent claim → endorse_claim.
  - If there are no meaningful parent dependencies → propose_claim.

Return JSON only: {"label": "<one_label>", "confidence": <0.0_to_1.0>}"""


# ── Input formatting ──────────────────────────────────────────────────────────

def build_claim_lookup(rows: List[dict]) -> Dict[str, dict]:
    """Build a claim_id → row lookup for parent text retrieval."""
    return {
        r["claim_id"]: r
        for r in rows
        if r.get("claim_id")
    }


def format_row_for_llm(row: dict, claim_lookup: Dict[str, dict]) -> str:
    """
    Format one row + its parent claims as structured text for the classifier.
    Keeps context tight: reasoning_text truncated to 400 chars per row.
    """
    lines = [
        "Current claim:",
        f"  claim_id:      {row.get('claim_id', '?')}",
        f"  subtask_id:    {row.get('subtask_id', '?')}",
        f"  agent_role:    {row.get('role') or row.get('agent_role', '?')}",
        f"  reasoning:     {(row.get('reasoning_text') or '')[:400]}",
    ]

    parent_ids = row.get("parent_claim_ids", []) or []
    if parent_ids:
        lines.append("\nParent claims:")
        for i, pid in enumerate(parent_ids[:5]):   # cap at 5 parents
            prow = claim_lookup.get(pid, {})
            lines.append(f"  {i+1}. claim_id: {pid}")
            lines.append(f"     agent_role: {prow.get('role') or prow.get('agent_role', '?')}")
            lines.append(f"     reasoning:  {(prow.get('reasoning_text') or '')[:400]}")
    else:
        lines.append("\nParent claims: none")

    return "\n".join(lines)


# ── LLM call ──────────────────────────────────────────────────────────────────

def _call_classifier(user_content: str, llm: Any) -> tuple[Optional[str], Optional[float]]:
    """
    Call the LLM classifier for one row.
    Returns (label, confidence) or (None, None) on failure.
    """
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        # temperature=0 for deterministic, reproducible classification
        classifier_llm = llm.bind(temperature=0) if hasattr(llm, "bind") else llm
        response = classifier_llm.invoke([
            SystemMessage(content=_CLASSIFIER_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ])
        raw = response.content if hasattr(response, "content") else str(response)
        raw = raw.strip()
        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        s, e = raw.find("{"), raw.rfind("}")
        if s == -1 or e == -1:
            return None, None
        parsed     = json.loads(raw[s:e+1])
        label      = parsed.get("label", "").strip()
        confidence = float(parsed.get("confidence", 0.5))
        if label not in SEMANTIC_LABELS:
            return None, 0.0
        return label, confidence
    except Exception:
        return None, 0.0


# ── Batch annotator ───────────────────────────────────────────────────────────

def annotate_event_types_llm(
    rows:         List[dict],
    llm:          Any,
    batch_size:   int = 20,
    retry_delay:  float = 1.0,
) -> List[dict]:
    """
    LLM semantic classification pass (Stage 2b).

    Only processes rows flagged needs_llm_event_classification=True by the
    rule pass. All other rows get event_type_llm=None, confidence=None.

    Writes to each row:
      event_type_llm             — semantic label or None
      event_type_llm_confidence  — float 0-1 or None

    Args:
        rows:        annotated rows from annotate_event_types()
        llm:         LangChain LLM instance. Prefer a separate/stronger model
                     than the participating agents (e.g. gpt-4o instead of
                     gpt-4o-mini) to avoid same-model correlation bias.
        batch_size:  rows to process per second to stay within rate limits
        retry_delay: seconds between batches
    """
    claim_lookup = build_claim_lookup(rows)

    candidates = [r for r in rows if r.get("needs_llm_event_classification")]
    non_cands  = [r for r in rows if not r.get("needs_llm_event_classification")]

    # Non-candidates get null LLM fields
    for row in non_cands:
        row["event_type_llm"]            = None
        row["event_type_llm_confidence"] = None

    # Process candidates in batches
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        for row in batch:
            user_content = format_row_for_llm(row, claim_lookup)
            label, conf  = _call_classifier(user_content, llm)
            row["event_type_llm"]            = label
            row["event_type_llm_confidence"] = conf
            row["event_type_llm_error"]      = (label is None)

        if i + batch_size < len(candidates):
            time.sleep(retry_delay)

    return rows