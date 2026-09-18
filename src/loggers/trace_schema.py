"""
src/loggers/trace_schema.py

Canonical schemas for live agent output and runtime trace rows.

Lives in src/loggers/ alongside schemas.py and event_bus.py.

Key design rules:
  - message_id is orchestrator-owned, never agent-generated
  - claim_id is agent-generated (reasoning artifact)
  - content is typed via ContentOutput, not a free dict
  - event_type, role, status use Literal types for strictness
  - All post-hoc fields (event_type, root_claim_id, *_chain_id, etc.)
    are None at emit time and filled by event_extractor + dag_builder
"""

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field
import uuid


# ── Enums as Literals ────────────────────────────────────────

EventType = Literal[
    "propose_claim",
    "revise_claim",
    "contradict_claim",
    "merge_claims",
    "delegate_subtask",
    "endorse_claim",
]

AgentRole = Literal[
    "worker",
    "hub",
    "coordinator",
    "supervisor",
    "bridge",
    "synthesizer",
]

SubtaskStatus = Literal["active", "complete", "blocked"]

ClaimStatus = Literal["proposed", "revised", "contradicted", "merged", "endorsed"]


# ── Live agent output ─────────────────────────────────────────

class SubtaskOutput(BaseModel):
    subtask_id: str
    parent_subtask_id: Optional[str] = None
    assigned_by: Optional[str] = None          # agent_id that delegated this
    status: SubtaskStatus = "active"


class ContentOutput(BaseModel):
    reasoning: str = ""                         # concise reasoning summary
    answer: str = ""
    confidence: Optional[float] = None          # 0.0–1.0


class ProvenanceOutput(BaseModel):
    references_used: List[str] = Field(default_factory=list)    # message_ids
    claims_referenced: List[str] = Field(default_factory=list)  # claim_ids
    tools_used: List[str] = Field(default_factory=list)
    evidence_refs: List[str] = Field(default_factory=list)


class CoordinationSignals(BaseModel):
    """
    Optional hints the agent can emit when naturally applicable.
    Not a forced action grammar — never fabricate these.
    The extractor uses them as supporting signals, not sole criteria.
    """
    requested_subtask_creation: bool = False
    proposed_assignee: Optional[str] = None            # agent_id
    synthesis_of_multiple_inputs: bool = False          # True if merging ≥2 prior claims
    explicit_disagreement_with: List[str] = Field(default_factory=list)  # claim_ids
    explicit_correction_of: List[str] = Field(default_factory=list)      # claim_ids
    supports_claims: List[str] = Field(default_factory=list)             # claim_ids


class AgentOutput(BaseModel):
    """
    What the LLM returns per turn.

    message_id is NOT included — it is orchestrator-owned and assigned
    when wrapping this into a TraceRow. claim_id is agent-generated
    because it is a reasoning artifact referenced by subsequent claims.
    """
    claim_id: str = Field(default_factory=lambda: f"claim_{uuid.uuid4().hex[:12]}")
    parent_claim_ids: List[str] = Field(default_factory=list)

    subtask: SubtaskOutput
    content: ContentOutput
    provenance: ProvenanceOutput
    coordination_signals: CoordinationSignals = Field(default_factory=CoordinationSignals)

    final_answer: Optional[str] = None     # only for synthesizer / final node

    class Config:
        extra = "ignore"    # tolerate extra fields from LLM output


# ── Canonical trace row ───────────────────────────────────────

class TraceRow(BaseModel):
    """
    One row per agent turn, written to events.jsonl by the orchestrator.

    message_id is assigned here by the orchestrator, not by the agent.

    Post-hoc fields (event_type, root_claim_id, *_chain_id, claim_depth,
    subtask_depth, claim_status, secondary_root_claim_ids) are None at
    emit time and filled by event_extraction/event_extractor.py and
    observables/dag_builder.py.
    """
    # Run metadata
    run_id: str
    benchmark: str
    task_id: str
    task_family: str
    topology: str
    seed: int
    num_agents: int

    # Turn metadata
    step_id: int
    timestamp: float
    agent_id: str
    role: AgentRole

    # Orchestrator-assigned identifiers
    message_id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
    visible_message_ids: List[str] = Field(default_factory=list)
    neighbor_ids: List[str] = Field(default_factory=list)

    # Subtask lineage
    subtask_id: str
    parent_subtask_id: Optional[str] = None
    assigned_agent: Optional[str] = None
    subtask_status: SubtaskStatus = "active"

    # Claim lineage (root_claim_id filled post hoc by dag_builder)
    claim_id: str
    parent_claim_ids: List[str] = Field(default_factory=list)
    root_claim_id: Optional[str] = None
    secondary_root_claim_ids: List[str] = Field(default_factory=list)

    # Content
    reasoning_text: str = ""
    final_answer_text: Optional[str] = None
    message_length: int = 0
    confidence: Optional[float] = None

    # Provenance
    references_used:   List[str] = Field(default_factory=list)
    claims_visible:    List[str] = Field(default_factory=list)  # topology-visible claim IDs (structural)
    claims_referenced: List[str] = Field(default_factory=list)  # LLM-cited claim IDs (semantic)
    tools_used:        List[str] = Field(default_factory=list)
    evidence_refs:     List[str] = Field(default_factory=list)

    # Coordination signals (from agent — not event labels)
    requested_subtask_creation: bool = False
    proposed_assignee: Optional[str] = None
    synthesis_of_multiple_inputs: bool = False
    explicit_disagreement_with: List[str] = Field(default_factory=list)
    explicit_correction_of: List[str] = Field(default_factory=list)
    supports_claims: List[str] = Field(default_factory=list)

    # Post-hoc fields — filled by event_extractor + dag_builder, NOT by agent
    event_type: Optional[EventType] = None
    claim_status: Optional[ClaimStatus] = None
    revision_chain_id: Optional[str] = None
    contradiction_group_id: Optional[str] = None
    merge_id: Optional[str] = None
    claim_depth: Optional[int] = None
    subtask_depth: Optional[int] = None

    class Config:
        extra = "ignore"


# ── Output format string shown inside the agent prompt ────────
#
# Rules:
#   - No message_id field (orchestrator-owned)
#   - parent_claim_ids restricted to actually visible prior claims
#   - reasoning described as "concise" not "step-by-step"
#   - coordination_signals optional, never fabricated

AGENT_OUTPUT_FORMAT = """\
Output exactly one JSON object with these fields:

{
  "claim_id": "<new unique id, e.g. claim_abc123>",
  "parent_claim_ids": ["<only prior visible claim ids this output actually builds on, or [] if none>"],
  "subtask": {
    "subtask_id": "<your current subtask id>",
    "parent_subtask_id": "<id of parent subtask, or null>",
    "assigned_by": "<agent_id that assigned this, or null>",
    "status": "active"
  },
  "content": {
    "reasoning": "<concise reasoning summary>",
    "answer": "<your best current answer>",
    "confidence": <0.0 to 1.0>
  },
  "provenance": {
    "references_used": ["<message_ids you used>"],
    "claims_referenced": ["<claim_ids you referenced>"],
    "tools_used": [],
    "evidence_refs": []
  },
  "coordination_signals": {
    "requested_subtask_creation": false,
    "proposed_assignee": null,
    "synthesis_of_multiple_inputs": false,
    "explicit_disagreement_with": [],
    "explicit_correction_of": [],
    "supports_claims": []
  }
}

Fill coordination_signals only when naturally applicable — do not fabricate.\
"""