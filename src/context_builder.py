"""
src/context_builder.py

Assembles the full prompt for one agent turn.

Enforces a fixed context-token budget by truncating PRIOR OUTPUTS — the
only unbounded-size component — dropping the oldest entries first when full
history would overflow the budget.

What is NOT injected:
  - decision rules ("if prior claim -> revise_claim")
  - action grammar or event labels
  - revision_chain_id, merge_id, contradiction_group_id

These are computed post hoc by event_extraction/event_extractor.py
and observables/dag_builder.py.

Prompt composition order:
  1. BASE_PROMPT
  2. topology_addendum
  3. task_addendum
  4. runtime_state
  5. AGENT_OUTPUT_FORMAT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from prompts.base_prompt import BASE_PROMPT
from loggers.trace_schema import AGENT_OUTPUT_FORMAT
from prompts.topology_addenda import get_topology_addendum
from prompts.task_addenda import get_task_addendum

# Context budget for prompt assembly.
CONTEXT_BUDGET_TOKENS = 4000


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class AgentContextSpec:
    # Required
    task: str
    agent_id: str
    agent_role: str    # worker | hub | coordinator | supervisor | bridge | synthesizer
    topology: str      # chain | star | tree | full_mesh | sparse_mesh | hybrid_modular | dynamic_reputation
    task_family: str   # qa | reasoning | coding | planning | coordination | critique | synthesis

    # Runtime state
    local_goal: str = ""
    subtask_id: str = ""
    parent_subtask_id: Optional[str] = None
    step: int = 0
    max_steps: int = 0   # 0 = not enforced by this builder

    # Must already be filtered to visible messages for this agent
    # and ordered chronologically by the orchestrator.
    neighbor_ids: List[str] = field(default_factory=list)

    # Tuple: (agent_id, message_id, claim_id, full_text)
    # Must already be filtered to topology-visible messages for this agent
    # and ordered chronologically by the orchestrator.
    prior_outputs: List[Tuple[str, str, str, str]] = field(default_factory=list)

    available_tools: List[str] = field(default_factory=list)
    extra_context: str = ""


def build_context(spec: AgentContextSpec) -> str:
    """
    Assemble the full prompt for one agent turn.
    Returns the prompt string.

    Raises ValueError for unknown topology or task_family — fail loudly
    so misconfigured experimental conditions are caught immediately.
    """
    try:
        topology_addendum = get_topology_addendum(spec.topology)
    except KeyError as e:
        raise ValueError(
            f"Invalid topology in AgentContextSpec: '{spec.topology}'. "
            f"Check prompts/topology_addenda.py for valid keys."
        ) from e

    try:
        task_addendum = get_task_addendum(spec.task_family)
    except KeyError as e:
        raise ValueError(
            f"Invalid task_family in AgentContextSpec: '{spec.task_family}'. "
            f"Check prompts/task_addenda.py for valid keys."
        ) from e

    parts: List[str] = [
        BASE_PROMPT,
        topology_addendum,
        task_addendum,
    ]

    state_parts: List[str] = []

    header = f"AGENT: {spec.agent_id}  ROLE: {spec.agent_role}  STEP: {spec.step}"
    if spec.max_steps > 0:
        header += f"/{spec.max_steps}"
    state_parts.append(header)

    state_parts.append(
        f"NEIGHBORS: {', '.join(spec.neighbor_ids)}"
        if spec.neighbor_ids else "NEIGHBORS: none"
    )

    if spec.subtask_id:
        sub_line = f"SUBTASK: {spec.subtask_id}"
        if spec.parent_subtask_id:
            sub_line += f"  (parent: {spec.parent_subtask_id})"
        state_parts.append(sub_line)

    state_parts.append(f"TASK:\n{spec.task}")

    if spec.local_goal:
        state_parts.append(f"YOUR GOAL:\n{spec.local_goal}")

    if spec.available_tools:
        state_parts.append(f"TOOLS: {', '.join(spec.available_tools)}")

    if spec.extra_context:
        state_parts.append(f"ADDITIONAL CONTEXT:\n{spec.extra_context}")

    if spec.prior_outputs:
        prior_lines = [
            "PRIOR OUTPUTS",
            "Cite message_id and claim_id in your provenance when you build on these:",
        ]
        entries = [
            f"\n[{aid}]  message_id={mid}  claim_id={cid}\n{text}"
            for aid, mid, cid, text in spec.prior_outputs
        ]

        # Fixed-cost parts (everything except prior outputs) must fit first;
        # whatever budget remains goes to the most recent prior outputs,
        # dropping the oldest entries when full history would overflow it.
        fixed_cost = _estimate_tokens(
            "\n\n".join([BASE_PROMPT, topology_addendum, task_addendum,
                         "\n".join(state_parts), AGENT_OUTPUT_FORMAT])
        )
        budget_remaining = max(0, CONTEXT_BUDGET_TOKENS - fixed_cost)

        used = _estimate_tokens("\n".join(prior_lines))
        kept: List[str] = []
        for entry in reversed(entries):
            cost = _estimate_tokens(entry)
            if used + cost > budget_remaining and kept:
                break
            kept.append(entry)
            used += cost
        kept.reverse()

        if len(kept) < len(entries):
            prior_lines.append(
                f"\n[... {len(entries) - len(kept)} earlier output(s) omitted "
                f"to fit the {CONTEXT_BUDGET_TOKENS}-token context budget ...]"
            )
        state_parts.append("\n".join(prior_lines + kept))

    parts.append("\n".join(state_parts))
    parts.append(AGENT_OUTPUT_FORMAT)

    return "\n\n".join(p.strip() for p in parts if p.strip())


# Max visible neighbors per step for sparse topologies.
# This is a communication constraint applied by the orchestrator,
# not a token budget enforced here.
VISIBLE_NEIGHBOR_CAP: dict[int, int] = {8: 8, 16: 10, 32: 12, 64: 14, 128: 16}


def get_visible_neighbor_cap(n: int) -> int:
    """Return the neighbor visibility cap for a system of n agents."""
    for threshold in sorted(VISIBLE_NEIGHBOR_CAP.keys(), reverse=True):
        if n >= threshold:
            return VISIBLE_NEIGHBOR_CAP[threshold]
    return n


# Token budget for LLM completions. Used by _acall_llm in base.py.
MAX_COMPLETION_TOKENS = 1000