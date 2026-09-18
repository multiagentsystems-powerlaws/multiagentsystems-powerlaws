"""
src/event_extraction/event_extractor.py

Post-hoc event type inference from structured traces.
Lives in src/event_extraction/ alongside coordination.py, graph_builder.py, tce.py.

Classification rules (priority order):
  1. delegate_subtask   new subtask_id + valid parent + topology/role allows it
  2. contradict_claim   STRICT — explicit signal OR strong contradiction keywords only
  3. revise_claim       explicit correction signal or keywords (any parent count)
  4. merge_claims       STRUCTURAL fallback — multi-parent, not corrective
  5. endorse_claim      supports_claims non-empty, no correction
  6. propose_claim      default

  Merge is a structural fallback, not a first-pass filter. Contradiction and
  revision are checked first so that corrective multi-parent reconciliations
  are classified as revisions rather than merges. This matches the paper's
  modeling assumption: merge fan-in measures unconflicted synthesis, while
  revision waves capture corrective dynamics.

Design philosophy:
  - Merge is structural, not semantic. Any claim with >=2 parents is a merge.
    Requiring synthesis language would undercount merges and distort tail distributions.
  - Contradiction is strict. "but", "however", "instead" are excluded from
    correction patterns — they appear in non-contradictory revisions and would
    inflate false contradictions, which inflates heavy tails incorrectly.
  - False negatives are acceptable. False positives break the DAG.
  - Delegation requires explicit assignment signal, not just a new subtask_id,
    to prevent misclassification of reused or delayed subtask IDs.

Revision chain propagation uses parent-based inheritance as primary method,
with forward frontier propagation as secondary pass for cross-row chains.

claim_status is derived deterministically from event_type:
  propose_claim   → proposed
  revise_claim    → revised
  contradict_claim→ contradicted
  merge_claims    → merged
  endorse_claim   → endorsed
  delegate_subtask→ None (work event, not a claim status)
"""

from __future__ import annotations

import re
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from loggers.trace_schema import EventType, ClaimStatus


# ── Event type constants ──────────────────────────────────────

DELEGATE   : EventType = "delegate_subtask"
MERGE      : EventType = "merge_claims"
CONTRADICT : EventType = "contradict_claim"
REVISE     : EventType = "revise_claim"
ENDORSE    : EventType = "endorse_claim"
PROPOSE    : EventType = "propose_claim"

EVENT_TO_CLAIM_STATUS: Dict[str, Optional[ClaimStatus]] = {
    PROPOSE:    "proposed",
    REVISE:     "revised",
    CONTRADICT: "contradicted",
    MERGE:      "merged",
    ENDORSE:    "endorsed",
    DELEGATE:   None,
}

# ── Regex patterns ────────────────────────────────────────────
#
# Correction: excludes "but", "however", "instead" — these appear in
# non-contradictory revisions and would cause false contradictions if
# shared across both patterns.
#
# Contradiction: strong adversarial language only. Conservative by design.

_CORRECTION_RE = re.compile(
    r"\b(incorrect|wrong|error|mistake|fix|correct(?:ion)?|should be|actually|"
    r"disagree|conflicting|recalculate|recomputed|correct total|"
    r"revised total|actual total|right answer|not right)\b",
    re.IGNORECASE,
)

_CONTRADICTION_RE = re.compile(
    r"\b(contradict|conflict|incompatible|oppose|reject|"
    r"cannot be both|mutually exclusive|fundamentally disagree)\b",
    re.IGNORECASE,
)

DELEGATION_TOPOLOGIES = {"star", "tree", "hybrid_modular"}
DELEGATION_ROLES      = {"hub", "supervisor", "coordinator", "bridge"}


# ── Core classifier ───────────────────────────────────────────

def infer_event_type(
    row: dict,
    known_subtask_ids: Set[str],
    topology_allows_delegation: bool,
) -> Optional[EventType]:
    # parent_claim_ids = topology-computed visible claim IDs
    parent_claim_ids: List[str] = row.get("parent_claim_ids", []) or row.get("claims_visible", []) or []
    subtask_id: str             = row.get("subtask_id", "") or ""
    parent_subtask_id           = row.get("parent_subtask_id")
    reasoning: str              = row.get("reasoning_text", "") or ""

    # Fix 1: read signals from top-level TraceRow fields, not a nested dict.
    # TraceRow emits these as flat fields — "coordination_signals" is never
    # populated as a nested dict in practice.
    signals = {
        "requested_subtask_creation":  row.get("requested_subtask_creation", False),
        "proposed_assignee":           row.get("proposed_assignee"),
        "synthesis_of_multiple_inputs": row.get("synthesis_of_multiple_inputs", False),
        "explicit_disagreement_with":  row.get("explicit_disagreement_with") or [],
        "explicit_correction_of":      row.get("explicit_correction_of") or [],
        "supports_claims":             row.get("supports_claims") or [],
    }

    # 1. delegate_subtask
    # Requires: new subtask_id, valid parent_subtask_id, topology/role permits it,
    # AND an explicit assignment signal to prevent misclassification of reused IDs.
    if (
        subtask_id
        and subtask_id not in known_subtask_ids
        and parent_subtask_id
        and topology_allows_delegation
        and (
            signals["requested_subtask_creation"]
            or signals["proposed_assignee"] is not None
        )
    ):
        return DELEGATE

    # Fix 2: non-claim rows return None — no claim_id means no claim event.
    claim_id = row.get("claim_id")
    if not claim_id:
        return None

    # Fix 3: contradiction and revision are checked BEFORE merge.
    # Priority order matters: a corrective multi-parent row (explicit correction
    # citing 2 parents) should be REVISE, not MERGE. Merge is the structural
    # fallback for uncorrective multi-parent claims only.

    # 2. contradict_claim — STRICT
    # Only explicit signal or strong adversarial language.
    # "but", "however", "instead" deliberately excluded — they appear in revisions.
    if signals["explicit_disagreement_with"]:
        return CONTRADICT
    if len(parent_claim_ids) >= 1 and _CONTRADICTION_RE.search(reasoning):
        return CONTRADICT

    # 3. revise_claim — explicit correction signal or correction keywords
    # Checked before merge: a corrective claim that cites multiple parents
    # (e.g. resolves a conflict between two prior claims) is a revision, not
    # a structural merge.
    if len(parent_claim_ids) >= 1:
        if signals["explicit_correction_of"] or _CORRECTION_RE.search(reasoning):
            return REVISE

    # 4. merge_claims — STRUCTURAL fallback
    # Any multi-parent claim that is not corrective is a merge by definition.
    # Synthesis language not required: requiring it would undercount merges
    # and distort the merge fan-in tail distribution.
    if len(parent_claim_ids) >= 2:
        return MERGE

    # 5. endorse_claim
    if signals["supports_claims"] and len(parent_claim_ids) >= 1:
        return ENDORSE

    # 6. propose_claim (default)
    return PROPOSE


# ── Batch annotator ───────────────────────────────────────────

def annotate_event_types(rows: List[dict]) -> List[dict]:
    """
    Rule-based annotation pass (Stage 2a).

    Writes event_type_rule, claim_status_rule, and needs_llm_event_classification
    to every row. Does NOT write final event_type — that is done by
    resolve_event_labels.resolve_event_labels() after the LLM pass.

    For backward compatibility, also writes event_type = event_type_rule so
    the pipeline works without the LLM pass if resolve_event_labels is not called.

    needs_llm_event_classification is True for rows where the rule-based
    classifier may be unreliable:
      - multi-parent claim rows (merge vs revise vs contradict ambiguity)
      - single-parent rows classified as merge/revise/contradict/endorse
        (semantic content determines the correct label)
    """
    rows = sorted(rows, key=lambda r: (r.get("step_id", 0), r.get("timestamp", 0)))
    known_subtask_ids: Set[str] = set()

    for row in rows:
        topology = row.get("topology", "")
        role     = row.get("role", "")
        allows   = topology in DELEGATION_TOPOLOGIES or role in DELEGATION_ROLES

        event_type = infer_event_type(row, known_subtask_ids, allows)
        row["event_type_rule"]   = event_type
        row["claim_status_rule"] = (
            EVENT_TO_CLAIM_STATUS[event_type] if event_type is not None else None
        )
        # Backward-compat: also write final labels (overwritten by resolver if used)
        row["event_type"]   = event_type
        row["claim_status"] = row["claim_status_rule"]

        # Flag rows that benefit from LLM disambiguation
        parents = row.get("parent_claim_ids", []) or []
        row["needs_llm_event_classification"] = bool(
            row.get("claim_id") and (
                len(parents) >= 2
                or event_type in {MERGE, REVISE, CONTRADICT, ENDORSE}
                or (event_type == PROPOSE and len(parents) == 1)
            )
        )

        if row.get("subtask_id"):
            known_subtask_ids.add(row["subtask_id"])

    _assign_revision_chains(rows)
    _assign_contradiction_groups(rows, window_steps=3)
    _assign_merge_ids(rows)
    return rows


# ── Grouping ID assignment ────────────────────────────────────

def _assign_revision_chains(rows: List[dict]) -> None:
    """
    Assign revision_chain_id to all revise_claim rows.

    Primary method: parent-based inheritance.
      If a revision's parent already has a revision_chain_id, inherit it.
      This correctly handles branching chains and non-contiguous rows.

    Secondary method: forward frontier propagation.
      For chains that start mid-run, propagate the new chain_id forward
      to any unassigned revision that references a claim already in the chain.
    """
    claim_to_idx: Dict[str, int] = {
        row["claim_id"]: i
        for i, row in enumerate(rows)
        if row.get("claim_id")
    }

    for i, row in enumerate(rows):
        if row.get("event_type_rule") != REVISE or row.get("revision_chain_id"):
            continue
        parent_ids = row.get("parent_claim_ids", [])
        if not parent_ids:
            continue

        # Primary: inherit from parent
        parent_idx = claim_to_idx.get(parent_ids[0])
        if parent_idx is not None and rows[parent_idx].get("revision_chain_id"):
            row["revision_chain_id"] = rows[parent_idx]["revision_chain_id"]
            continue

        # Secondary: start a new chain and propagate forward
        chain_id = f"rev_{uuid.uuid4().hex[:8]}"
        row["revision_chain_id"] = chain_id
        frontier: Set[str] = {row["claim_id"]}
        for j in range(i + 1, len(rows)):
            r = rows[j]
            if r.get("event_type") != REVISE or r.get("revision_chain_id"):
                continue
            if set(r.get("parent_claim_ids", [])) & frontier:
                r["revision_chain_id"] = chain_id
                frontier.add(r["claim_id"])


def _assign_contradiction_groups(rows: List[dict], window_steps: int = 3) -> None:
    """
    Group contradictory claims targeting the same parent within a temporal window.
    Burst size (distinct agents per group) is computed downstream in cascade_metrics,
    where agent_id deduplication happens over each group's rows.
    """
    contra_by_parent: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for i, row in enumerate(rows):
        if row.get("event_type_rule") == CONTRADICT:
            for pid in row.get("parent_claim_ids", []):
                contra_by_parent[pid].append((row.get("step_id", 0), i))

    for _pid, entries in contra_by_parent.items():
        entries.sort(key=lambda x: x[0])
        groups: List[List[int]] = []
        current: List[int] = []
        last_step: Optional[int] = None
        for step, idx in entries:
            if last_step is None or (step - last_step) <= window_steps:
                current.append(idx)
            else:
                if current:
                    groups.append(current)
                current = [idx]
            last_step = step
        if current:
            groups.append(current)
        for group in groups:
            gid = f"con_{uuid.uuid4().hex[:8]}"
            for idx in group:
                rows[idx]["contradiction_group_id"] = gid


def _assign_merge_ids(rows: List[dict]) -> None:
    for row in rows:
        if row.get("event_type_rule") == MERGE:
            row["merge_id"] = f"merge_{uuid.uuid4().hex[:8]}"