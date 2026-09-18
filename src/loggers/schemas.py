"""
schemas.py
----------
Pydantic v2 models for every loggable event in the MAS power-law study.

Models:
  - AgentEvent    (one per agent action)
  - GraphSnapshot (one per topology window / step)
  - RunConfig     (one per run, written at start)
  - RunMetadata   (one per run, written at end with outcomes)

H2 outcome metrics are recorded in RunMetadata so that regime structure
(influence Gini, coalition persistence, bridge traffic) can be joined to
actual task quality and efficiency for each run.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import time


# ─────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────

class EventType(str, Enum):
    DELEGATE_SUBTASK       = "delegate_subtask"
    COMPLETE_SUBTASK       = "complete_subtask"
    PROPOSE_CLAIM          = "propose_claim"
    REVISE_CLAIM           = "revise_claim"
    CONTRADICT_CLAIM       = "contradict_claim"
    MERGE_CLAIMS           = "merge_claims"
    ENDORSE_CLAIM          = "endorse_claim"
    CONSULT_AGENT          = "consult_agent"
    TOOL_CALL              = "tool_call"
    FINALIZE_ANSWER        = "finalize_answer"
    ROUTE_MESSAGE          = "route_message"


class ClaimType(str, Enum):
    ROOT_CLAIM             = "root_claim"
    INTERMEDIATE_CLAIM     = "intermediate_claim"
    FINAL_CLAIM            = "final_claim"
    CRITIQUE               = "critique"
    SYNTHESIS              = "synthesis"


class ClaimStatus(str, Enum):
    PROPOSED     = "proposed"
    REVISED      = "revised"
    CONTRADICTED = "contradicted"
    MERGED       = "merged"
    ACCEPTED     = "accepted"
    REJECTED     = "rejected"
    UNRESOLVED   = "unresolved"


class SubtaskType(str, Enum):
    PLANNING    = "planning"
    RETRIEVAL   = "retrieval"
    CRITIQUE    = "critique"
    SYNTHESIS   = "synthesis"
    CODING      = "coding"
    VALIDATION  = "validation"


class SubtaskStatus(str, Enum):
    CREATED     = "created"
    ASSIGNED    = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED   = "completed"
    MERGED      = "merged"
    DROPPED     = "dropped"


class RevisionReason(str, Enum):
    FACTUAL_ERROR      = "factual_error"
    INCONSISTENCY      = "inconsistency"
    INCOMPLETENESS     = "incompleteness"
    BETTER_DECOMPOSITION = "better_decomposition"
    ALTERNATIVE_PLAN   = "alternative_plan"
    MERGE_CONFLICT     = "merge_conflict"


class ResolutionStatus(str, Enum):
    RESOLVED   = "resolved"
    UNRESOLVED = "unresolved"
    ESCALATED  = "escalated"


class EndorsementType(str, Enum):
    AGREEMENT         = "agreement"
    REUSE             = "reuse"
    VALIDATION        = "validation"
    COALITION_SUPPORT = "coalition_support"


class TopologyName(str, Enum):
    CHAIN              = "chain"
    STAR               = "star"
    TREE               = "tree"
    FULL_MESH          = "full_mesh"
    SPARSE_MESH        = "sparse_mesh"
    HYBRID_MODULAR     = "hybrid_modular"
    DYNAMIC_REPUTATION = "dynamic_reputation"


class MemoryType(str, Enum):
    NONE           = "none"
    SLIDING_WINDOW = "sliding_window"
    FULL_HISTORY   = "full_history"
    SUMMARIZED     = "summarized"


class RoutingStrategy(str, Enum):
    UNIFORM_RANDOM     = "uniform_random"
    PLANNER_ASSIGNED   = "planner_assigned"
    REPUTATION_BASED   = "reputation_based"
    LOCAL_NEIGHBORHOOD = "local_neighborhood"


# ─────────────────────────────────────────────
# Core event model
# ─────────────────────────────────────────────

class AgentEvent(BaseModel):
    """One atomic loggable action from an agent."""

    # ── Identity ──
    run_id:          str
    benchmark:       str
    task_id:         str
    task_family:     str
    difficulty:      str
    topology:        TopologyName
    architecture:    str
    num_agents:      int
    seed:            int
    step_id:         int
    timestamp:       float = Field(default_factory=time.time)
    agent_id:        str
    agent_role:      str = "peer"

    # ── Event type ──
    event_type: EventType

    # ── Target ──
    target_agent_id:   Optional[str]  = None
    target_claim_id:   Optional[str]  = None
    target_subtask_id: Optional[str]  = None

    # ── Message / action metadata ──
    message_id:            Optional[str]   = None
    message_length_tokens: Optional[int]   = None
    message_length_chars:  Optional[int]   = None
    tool_name:             Optional[str]   = None
    tool_args_summary:     Optional[str]   = None
    confidence_score:      Optional[float] = None
    action_success:        Optional[bool]  = None

    # ── Tokens / compute ──
    tokens_input:            Optional[int]   = None
    tokens_output:           Optional[int]   = None
    tokens_total_event:      Optional[int]   = None
    latency_ms:              Optional[float] = None
    compute_cost_estimate:   Optional[float] = None
    num_tool_calls:          int = 0
    num_external_retrievals: int = 0

    # ── Claim graph fields ──
    claim_id:                    Optional[str]        = None
    claim_type:                  Optional[ClaimType]  = None
    parent_claim_ids:            List[str]             = Field(default_factory=list)
    root_claim_id:               Optional[str]        = None
    claim_depth:                 Optional[int]        = None
    claim_text_hash:             Optional[str]        = None
    claim_semantic_embedding_id: Optional[str]        = None
    claim_status:                Optional[ClaimStatus] = None

    # ── Subtask graph fields ──
    subtask_id:           Optional[str]           = None
    parent_subtask_id:    Optional[str]           = None
    root_subtask_id:      Optional[str]           = None
    subtask_depth:        Optional[int]           = None
    subtask_type:         Optional[SubtaskType]   = None
    subtask_assigned_by:  Optional[str]           = None
    subtask_assigned_to:  Optional[str]           = None
    subtask_status:       Optional[SubtaskStatus] = None
    subtask_branch_index: Optional[int]           = None

    # ── Contradiction / revision fields ──
    contradiction_group_id:  Optional[str]              = None
    revision_chain_id:       Optional[str]              = None
    trigger_claim_id:        Optional[str]              = None
    contradiction_window_id: Optional[str]              = None
    reason_for_revision:     Optional[RevisionReason]   = None
    resolution_status:       Optional[ResolutionStatus] = None
    num_agents_involved:     Optional[int]              = None

    # ── Merge fields ──
    merge_id:                   Optional[str]  = None
    merge_parent_claim_ids:     List[str]       = Field(default_factory=list)
    merge_parent_subtask_ids:   List[str]       = Field(default_factory=list)
    merge_num_inputs:           Optional[int]  = None
    merge_num_unique_agents:    Optional[int]  = None
    merge_synthesizer_agent_id: Optional[str]  = None
    merge_output_claim_id:      Optional[str]  = None
    merge_success:              Optional[bool] = None

    # ── Endorsement / coalition fields ──
    endorsed_claim_id:      Optional[str]             = None
    endorsed_agent_id:      Optional[str]             = None
    endorsement_strength:   Optional[float]           = None
    endorsement_reason:     Optional[str]             = None
    support_type:           Optional[EndorsementType] = None
    interaction_context_id: Optional[str]             = None
    local_group_id:         Optional[str]             = None
    same_claim_lineage:     Optional[bool]            = None
    same_subtask_branch:    Optional[bool]            = None

    # ── Agent dynamic state snapshot ──
    visible_neighbors:            List[str]       = Field(default_factory=list)
    memory_size:                  Optional[int]   = None
    recent_message_count:         Optional[int]   = None
    recent_tool_calls:            Optional[int]   = None
    current_subtask_id_state:     Optional[str]   = None
    current_claim_focus:          Optional[str]   = None
    local_belief_state_hash:      Optional[str]   = None
    local_belief_entropy:         Optional[float] = None
    agent_influence_score_so_far: Optional[float] = None
    agent_degree_so_far:          Optional[int]   = None
    agent_bridge_score_so_far:    Optional[float] = None

    # ── Semantic diversity ──
    claim_cluster_id:           Optional[str]   = None
    pairwise_cosine_similarity: Optional[float] = None
    divergence_from_final:      Optional[float] = None


# ─────────────────────────────────────────────
# Graph snapshot (per step / window)
# ─────────────────────────────────────────────

class GraphSnapshot(BaseModel):
    """Topology state snapshot taken at a given step."""
    run_id:     str
    step:       int
    topology:   str
    num_agents: int

    # Structural
    edge_list:             List[tuple]         = Field(default_factory=list)
    edge_weights:          Dict[str, float]    = Field(default_factory=dict)
    agent_degrees:         Dict[str, int]      = Field(default_factory=dict)
    weighted_degrees:      Dict[str, float]    = Field(default_factory=dict)
    bridge_agents:         List[str]           = Field(default_factory=list)
    community_assignments: Dict[str, int]      = Field(default_factory=dict)

    # Graph metrics
    graph_entropy:          float = 0.0
    average_path_length:    float = 0.0
    clustering_coefficient: float = 0.0
    modularity:             float = 0.0
    centralization_index:   float = 0.0

    # Sparse activation tracking
    active_agents_this_step: List[str] = Field(default_factory=list)
    num_active_this_step:    int       = 0

    # Influence
    influence_scores: Dict[str, float] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# Run configuration (written at run start)
# ─────────────────────────────────────────────

class RunConfig(BaseModel):
    """Complete configuration for one experimental run."""
    run_id:      str
    benchmark:   str
    task_id:     str
    task_family: str
    difficulty:  str

    task_decomposability:    Optional[float] = None
    task_conflict_level:     Optional[float] = None
    task_requires_tools:     bool = False
    task_requires_synthesis: bool = False

    topology:                 TopologyName
    architecture:             str
    routing_strategy:         RoutingStrategy = RoutingStrategy.PLANNER_ASSIGNED
    memory_type:              MemoryType = MemoryType.SLIDING_WINDOW
    endorsement_persistence:  bool = False
    scarcity_condition:       bool = False

    num_agents:    int
    max_steps:     int = 50
    seed:          int
    run_seed:      int
    task_seed:     int
    topology_seed: int

    model_name:     str   = "gpt-4o-mini"
    temperature:    float = 0.7
    config_hash:    Optional[str] = None
    git_commit_hash: Optional[str] = None


# ─────────────────────────────────────────────
# Run outcomes (written at run end)
# ─────────────────────────────────────────────

class RunMetadata(BaseModel):
    """
    Final outcomes written when a run completes.

    Organized into four sections:
      1. Core task outcome       — for H1/H3 baseline
      2. Completeness metrics    — for H2 regime-quality analysis
      3. Efficiency metrics      — for H2 quality-adjusted performance
      4. Activation summary      — for H3 scaling
      5. Regime structure hints  — derived at extraction time, stored here
    """
    run_id: str

    # ── 1. Core task outcome ──────────────────────────────────
    task_success:  Optional[bool]  = None   # binary pass/fail
    task_score:    Optional[float] = None   # benchmark-native scalar (0–1)

    # Benchmark-specific quality fields
    # SWE-bench
    swe_patch_applied:   Optional[bool] = None  # patch applied without error
    swe_tests_passed:    Optional[int]  = None  # number of tests passing
    swe_tests_total:     Optional[int]  = None  # total tests in suite
    swe_files_modified:  Optional[int]  = None  # files touched by patch
    # GAIA
    gaia_exact_match:    Optional[bool]  = None  # exact string match
    gaia_rubric_score:   Optional[float] = None  # rubric/partial credit
    gaia_tools_used:     Optional[int]   = None  # number of tool calls used
    # MARBLE / coordination
    marble_subgoals_completed:   Optional[int]   = None
    marble_constraints_satisfied: Optional[int]  = None
    marble_team_objective_met:   Optional[bool]  = None
    # REALM / planning
    realm_plan_valid:            Optional[bool]  = None
    realm_recovered_from_disruption: Optional[bool] = None
    realm_num_replans:           Optional[int]   = None
    realm_dependency_satisfaction_rate: Optional[float] = None

    # ── 2. Completeness metrics (H2) ─────────────────────────
    # These connect regime structure to actual problem-solving progress.
    num_subtasks_total:         int   = 0    # subtasks created during run
    num_subtasks_completed:     int   = 0    # subtasks with status=completed
    num_subtasks_open_final:    int   = 0    # subtasks still open at end
    completion_ratio:           float = 0.0  # completed / total (0 if total=0)

    num_claims_total:           int   = 0    # total claim nodes in DAG
    num_claims_merged:          int   = 0    # claims resolved via merge
    num_claims_unresolved_final: int  = 0    # contradicted/open at end
    coherence_score:            float = 0.0  # 1 - (unresolved / total)

    num_revisions_total:        int   = 0    # total revise_claim events
    num_contradictions_total:   int   = 0    # total contradict_claim events
    num_merges_total:           int   = 0    # total merge_claims events
    num_endorsements_total:     int   = 0    # total endorse_claim events

    # Integration: fraction of terminal claims that were merged into final answer
    # (higher = more synthesis, less fragmentation)
    integration_score:          float = 0.0  # merged_claims / terminal_claims

    # ── 3. Efficiency metrics (H2) ────────────────────────────
    # Used to compare regimes on quality-adjusted performance.
    tokens_total:               int   = 0
    messages_total:             int   = 0
    num_coordination_events_total: int = 0
    wall_time_seconds:          Optional[float] = None

    # Derived efficiency scores (computed at extraction time, stored here)
    # All are Optional — only meaningful when task_score > 0
    success_per_token:          Optional[float] = None  # task_score / tokens_total
    completion_per_token:       Optional[float] = None  # completion_ratio / tokens_total
    quality_adjusted_efficiency: Optional[float] = None  # task_score / wall_time_seconds

    # ── 4. Activation summary (H3) ───────────────────────────
    num_unique_agents_activated: int   = 0
    unique_agents_touched:       int   = 0
    mean_active_per_step:        float = 0.0
    activation_rate:             float = 0.0   # unique_touched / num_agents
    active_agents_per_step:      Dict[str, int] = Field(default_factory=dict)

    # ── 5. Extra / free-form ─────────────────────────────────
    extra: Dict[str, Any] = Field(default_factory=dict)