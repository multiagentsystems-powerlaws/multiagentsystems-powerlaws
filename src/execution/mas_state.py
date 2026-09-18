"""
execution/mas_state.py
----------------------
Canonical MASState TypedDict used by all topologies and the runner.

All dict / list fields carry explicit Annotated reducers so that
parallel fan-out branches (star, full_mesh) can write concurrently
without LangGraph raising InvalidUpdateError.

Import this instead of defining MASState in base.py.
"""

from __future__ import annotations

from typing import Any, Annotated, Dict, List, Optional, TypedDict
import operator

from langchain_core.messages import BaseMessage


# ─────────────────────────────────────────────────────────────
# Reducer helpers
# ─────────────────────────────────────────────────────────────

def _merge_dicts(a: Dict, b: Dict) -> Dict:
    """Merge two dicts: b wins on key conflicts. Safe for parallel writes."""
    if not a:
        return b
    if not b:
        return a
    return {**a, **b}


def _keep_last(a: Any, b: Any) -> Any:
    """For scalar fields written by parallel branches: last non-None wins."""
    return b if b is not None else a


def _max_int(a: int, b: int) -> int:
    """For step counter: keep the highest value seen."""
    return max(a or 0, b or 0)


# ─────────────────────────────────────────────────────────────
# Canonical state
# ─────────────────────────────────────────────────────────────

class MASState(TypedDict):
    """
    Shared mutable state that flows through every LangGraph node.

    Fields
    ------
    messages
        Full conversation history; new messages are appended (operator.add).
    task
        The original task string; never mutated after init.
    current_agent
        ID of the agent currently executing; last-write wins.
    step
        Global step counter; max-wins across parallel branches.
    claims
        claim_id → claim dict; branches merge by union (b wins on conflict).
    subtasks
        subtask_id → subtask dict; same merge policy.
    influence
        agent_id → cumulative influence score; merged by union.
    agent_outputs
        agent_id → latest text output; merged by union.
    final_answer
        Set once by the synthesizer node; last non-None wins.
    metadata
        Free-form per-run metadata dict; merged by union.
    """
    messages:       Annotated[List[BaseMessage],    operator.add]
    task:           str
    current_agent:  Annotated[str,                  _keep_last]
    step:           Annotated[int,                  _max_int]
    claims:         Annotated[Dict[str, Any],       _merge_dicts]
    subtasks:       Annotated[Dict[str, Any],       _merge_dicts]
    influence:      Annotated[Dict[str, float],     _merge_dicts]
    agent_outputs:  Annotated[Dict[str, str],       _merge_dicts]
    final_answer:   Annotated[Optional[str],        _keep_last]
    metadata:       Annotated[Dict[str, Any],       _merge_dicts]


def initial_state(task: str) -> MASState:
    """Return a zero-value MASState for a new run."""
    from langchain_core.messages import HumanMessage
    return MASState(
        messages=[HumanMessage(content=task)],
        task=task,
        current_agent="",
        step=0,
        claims={},
        subtasks={},
        influence={},
        agent_outputs={},
        final_answer=None,
        metadata={},
    )
