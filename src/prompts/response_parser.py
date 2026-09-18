"""
prompts/response_parser.py
---------------------------
Parses agent JSON responses into structured event fields.
Also validates the action contract constraints and rejects/relabels
structurally invalid events before they hit the log.

Shared / topology-agnostic version:
- normalizes actions
- validates generic structural constraints
- parses optional critique / novelty fields
- avoids topology-specific parent rewriting
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional

ACTION_TO_EVENT = {
    "propose_claim":    "propose_claim",
    "revise_claim":     "revise_claim",
    "contradict_claim": "contradict_claim",
    "merge_claims":     "merge_claims",
    "delegate_subtask": "delegate_subtask",
    "complete_subtask": "complete_subtask",
    "endorse_claim":    "endorse_claim",
    "finalize_answer":  "finalize_answer",
    # Short aliases the model sometimes uses
    "propose":    "propose_claim",
    "revise":     "revise_claim",
    "contradict": "contradict_claim",
    "merge":      "merge_claims",
    "delegate":   "delegate_subtask",
    "complete":   "complete_subtask",
    "endorse":    "endorse_claim",
    "finalize":   "finalize_answer",
}


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:6]}"


def _safe_float(value: Any, default: float = 0.7) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _extract_json(raw: str) -> Optional[dict]:
    """Try multiple strategies to extract a JSON object from raw output."""
    raw = (raw or "").strip()
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    m = re.search(r'\{[^{}]*"action"[^{}]*\}', raw, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return None


def _coerce_parent_ids(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            loaded = json.loads(s)
            if isinstance(loaded, list):
                return [str(x).strip() for x in loaded if str(x).strip()]
        except Exception:
            pass
        return [part.strip() for part in s.split(",") if part.strip()]

    return []


def _validate_and_fix(parsed: dict) -> dict:
    """
    Enforce generic action contract constraints.
    Topology-specific lineage logic should stay outside this parser.
    """
    action = parsed.get("action", "propose_claim")
    action = ACTION_TO_EVENT.get(action, "propose_claim")

    pids = _coerce_parent_ids(parsed.get("parent_claim_ids", []))
    parsed["parent_claim_ids"] = pids

    # merge_claims needs >= 2 parents
    if action == "merge_claims" and len(pids) < 2:
        if len(pids) == 1:
            action = "revise_claim"
            parsed["revision_chain_id"] = parsed.get("revision_chain_id") or _new_id("rev")
            parsed["trigger_claim_id"] = parsed.get("trigger_claim_id") or pids[0]
            parsed["reason_for_revision"] = parsed.get("reason_for_revision") or "incompleteness"
        else:
            action = "propose_claim"
            parsed["parent_claim_ids"] = []

    # revise_claim needs exactly 1 parent
    if action == "revise_claim" and len(parsed["parent_claim_ids"]) != 1:
        if len(parsed["parent_claim_ids"]) == 0:
            action = "propose_claim"
        elif len(parsed["parent_claim_ids"]) > 1:
            action = "merge_claims"

    # contradict_claim needs exactly 1 parent
    if action == "contradict_claim" and len(parsed["parent_claim_ids"]) != 1:
        if len(parsed["parent_claim_ids"]) == 0:
            action = "propose_claim"
        elif len(parsed["parent_claim_ids"]) > 1:
            action = "merge_claims"

    # endorse_claim should generally point to one claim; if not, keep as-is
    # rather than forcing topology-specific fallback behavior.
    if action == "endorse_claim" and len(parsed["parent_claim_ids"]) > 1:
        action = "merge_claims"

    # propose_claim must have empty parents
    if action == "propose_claim":
        parsed["parent_claim_ids"] = []

    # delegate_subtask needs target_agent_id
    if action == "delegate_subtask" and not parsed.get("target_agent_id"):
        action = "propose_claim"
        parsed["parent_claim_ids"] = []

    if action == "merge_claims":
        pids = parsed.get("parent_claim_ids", [])
        parsed["merge_num_inputs"] = len(pids)
        parsed["merge_num_unique_agents"] = len(set(pids))
        if not parsed.get("merge_id"):
            parsed["merge_id"] = _new_id("merge")

    if action == "revise_claim":
        pids = parsed.get("parent_claim_ids", [])
        if not parsed.get("revision_chain_id"):
            parsed["revision_chain_id"] = _new_id("rev")
        if not parsed.get("trigger_claim_id") and pids:
            parsed["trigger_claim_id"] = pids[0]
        if not parsed.get("reason_for_revision"):
            parsed["reason_for_revision"] = "incompleteness"

    if action == "contradict_claim":
        pids = parsed.get("parent_claim_ids", [])
        if not parsed.get("contradiction_group_id"):
            parsed["contradiction_group_id"] = _new_id("con")
        if not parsed.get("trigger_claim_id") and pids:
            parsed["trigger_claim_id"] = pids[0]

    if action == "endorse_claim":
        pids = parsed.get("parent_claim_ids", [])
        if not parsed.get("endorsed_claim_id") and len(pids) == 1:
            parsed["endorsed_claim_id"] = pids[0]
        if not parsed.get("endorsement_reason"):
            critique = (parsed.get("critique") or "").strip()
            parsed["endorsement_reason"] = critique or "validation"
        if not parsed.get("support_type"):
            parsed["support_type"] = "validation"

    if not parsed.get("claim_id"):
        parsed["claim_id"] = _new_id("claim")

    if not parsed.get("subtask_id"):
        parsed["subtask_id"] = _new_id("sub")

    parsed["action"] = action
    parsed["event_type"] = action
    return parsed


def parse_agent_response(
    raw: str,
    prior_claim_id: Optional[str] = None,
    root_claim_id: Optional[str] = None,
    prior_subtask_id: Optional[str] = None,
    root_subtask_id: Optional[str] = None,
    available_claim_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Parse raw agent output into a validated, contract-compliant event dict.

    Notes:
    - prior_claim_id and available_claim_ids are accepted for interface compatibility,
      but topology-specific parent fallback / visibility enforcement should be handled
      by the calling topology code.
    """
    result = {
        "action":                  "propose_claim",
        "event_type":              "propose_claim",
        "claim_id":                _new_id("claim"),
        "claim_text":              "",
        "claim_type":              "intermediate_claim",
        "root_claim_id":           root_claim_id or "",
        "parent_claim_ids":        [],
        "revision_chain_id":       None,
        "trigger_claim_id":        None,
        "reason_for_revision":     None,
        "contradiction_group_id":  None,
        "merge_id":                None,
        "merge_num_inputs":        None,
        "merge_num_unique_agents": None,
        "subtask_id":              _new_id("sub"),
        "subtask_text":            None,
        "parent_subtask_id":       prior_subtask_id,
        "root_subtask_id":         root_subtask_id,
        "target_agent_id":         None,
        "subtask_type":            None,
        "endorsed_claim_id":       None,
        "endorsement_reason":      None,
        "support_type":            None,
        "confidence":              0.7,
        "critique":                "",
        "novelty":                 "",
        "answer":                  "",
        "raw_text":                raw,
    }

    parsed = _extract_json(raw)

    if parsed is None:
        m = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", raw or "", re.IGNORECASE)
        fallback_text = m.group(1).strip() if m else (raw or "")[:300].strip()

        result["claim_text"] = fallback_text
        result["answer"] = fallback_text

        if root_claim_id:
            result["root_claim_id"] = root_claim_id
        else:
            result["root_claim_id"] = result["claim_id"]

        return result

    parsed = _validate_and_fix(parsed)

    answer_text = parsed.get("answer")
    claim_text = parsed.get("claim_text")

    final_text = ""
    if isinstance(claim_text, str) and claim_text.strip():
        final_text = claim_text.strip()
    elif isinstance(answer_text, str) and answer_text.strip():
        final_text = answer_text.strip()

    result.update({
        "action":                  parsed["action"],
        "event_type":              parsed["action"],
        "claim_id":                parsed.get("claim_id") or result["claim_id"],
        "claim_text":              final_text,
        "claim_type":              parsed.get("claim_type", "intermediate_claim"),
        "parent_claim_ids":        parsed.get("parent_claim_ids", []),
        "revision_chain_id":       parsed.get("revision_chain_id"),
        "trigger_claim_id":        parsed.get("trigger_claim_id"),
        "reason_for_revision":     parsed.get("reason_for_revision"),
        "contradiction_group_id":  parsed.get("contradiction_group_id"),
        "merge_id":                parsed.get("merge_id"),
        "merge_num_inputs":        parsed.get("merge_num_inputs"),
        "merge_num_unique_agents": parsed.get("merge_num_unique_agents"),
        "subtask_id":              parsed.get("subtask_id") or result["subtask_id"],
        "subtask_text":            parsed.get("subtask_text"),
        "parent_subtask_id":       parsed.get("parent_subtask_id", prior_subtask_id),
        "root_subtask_id":         parsed.get("root_subtask_id", root_subtask_id),
        "target_agent_id":         parsed.get("target_agent_id"),
        "subtask_type":            parsed.get("subtask_type"),
        "endorsed_claim_id":       parsed.get("endorsed_claim_id"),
        "endorsement_reason":      parsed.get("endorsement_reason"),
        "support_type":            parsed.get("support_type"),
        "confidence":              _safe_float(parsed.get("confidence", 0.7), 0.7),
        "critique":                (parsed.get("critique") or "").strip(),
        "novelty":                 (parsed.get("novelty") or "").strip(),
        "answer":                  final_text,
    })

    if not result["root_claim_id"]:
        if root_claim_id:
            result["root_claim_id"] = root_claim_id
        elif result["action"] == "propose_claim":
            result["root_claim_id"] = result["claim_id"]

    return result


def event_type_from_action(action: str) -> str:
    return ACTION_TO_EVENT.get(action, "propose_claim")
