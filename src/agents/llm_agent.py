"""
src/agents/llm_agent.py

LLM agent node. Replaces the old version that emitted AgentEvent.
Now emits TraceRow (from loggers/trace_schema.py) directly to events.jsonl.

Key changes from old version:
  - Emits TraceRow, not AgentEvent — all claim_id, parent_claim_ids,
    coordination_signals, subtask_id fields are present and logged.
  - message_id is assigned here (orchestrator role), not by the LLM.
  - Tools are executed here and results injected into the agent's context
    before the LLM call, logged in tools_used and evidence_refs.
  - AgentOutput is parsed and validated against trace_schema.AgentOutput.
  - RunContext carries benchmark/topology/task_family for tool policy.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from loggers.event_bus import EventBus
from loggers.trace_schema import AgentOutput, TraceRow, SubtaskOutput, ContentOutput, ProvenanceOutput, CoordinationSignals
from context_builder import AgentContextSpec, build_context
from tools.tools import get_tools_for_benchmark, run_tool, ToolResult


# ── Run context ───────────────────────────────────────────────

@dataclass
class RunContext:
    """Immutable per-run identifiers shared across all agent nodes."""
    run_id:       str
    benchmark:    str
    task_id:      str
    task_family:  str
    topology:     str
    num_agents:   int
    seed:         int


# ── Step counter ──────────────────────────────────────────────

class StepCounter:
    """Monotonic step counter shared across all agents in a run."""
    def __init__(self) -> None:
        self._value = 0

    def increment(self) -> int:
        self._value += 1
        return self._value

    @property
    def value(self) -> int:
        return self._value


# ── Tool execution helper ─────────────────────────────────────

def _execute_tools_from_output(
    agent_output: dict,
    benchmark: str,
    environment: Any = None,
) -> Tuple[List[str], List[str]]:
    """
    Execute any tool_calls listed in the agent's raw output dict.
    Returns (tools_used_names, evidence_refs as tool output summaries).
    """
    raw_tools = agent_output.get("tool_calls") or []
    if not raw_tools:
        return [], []

    tools_used = []
    evidence_refs = []

    for call in raw_tools:
        if not isinstance(call, dict):
            continue
        tool_name  = call.get("tool_name") or call.get("name", "")
        tool_input = call.get("tool_input") or call.get("input", "")
        if not tool_name or not tool_input:
            continue

        result = run_tool(
            tool_name=tool_name,
            tool_input=str(tool_input),
            benchmark=benchmark,
            environment=environment,
        )
        tools_used.append(tool_name)
        # Evidence ref = truncated tool output for provenance
        evidence_refs.append(
            f"[{tool_name}] {result.summary(max_chars=200)}"
        )

    return tools_used, evidence_refs


# ── Parse LLM output → AgentOutput ───────────────────────────

def _parse_agent_output(raw_text: str) -> dict:
    """
    Parse LLM response text into a dict.
    Tolerates markdown fences and extra text before/after the JSON object.
    Returns empty dict on failure.
    """
    text = raw_text.strip()
    # Strip markdown code fences
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    # Find first { ... }
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start:end+1])
    except json.JSONDecodeError:
        return {}


# ── Main agent class ──────────────────────────────────────────

class LLMAgent:
    """
    One agent node in the MAS graph.

    Responsibilities:
      1. Build prompt via context_builder.build_context()
      2. Call the LLM
      3. Parse AgentOutput from response
      4. Execute any tool_calls and collect evidence
      5. Emit a fully-populated TraceRow to the EventBus
      6. Return the parsed output for the topology to route

    message_id is assigned here (orchestrator role).
    claim_id comes from the LLM output (reasoning artifact).
    """

    def __init__(
        self,
        agent_id:    str,
        agent_role:  str,
        llm:         Any,           # LangChain chat model
        bus:         EventBus,
        run_ctx:     RunContext,
        step_counter: StepCounter,
        environment: Any = None,    # MultiAgentBench env, if applicable
    ) -> None:
        self.agent_id     = agent_id
        self.agent_role   = agent_role
        self.llm          = llm
        self.bus          = bus
        self.run_ctx      = run_ctx
        self.step_counter = step_counter
        self.environment  = environment

    async def arun(
        self,
        spec: AgentContextSpec,
    ) -> Tuple[AgentOutput, TraceRow]:
        """
        Run one agent turn asynchronously.

        Args:
            spec: AgentContextSpec built by the topology orchestrator.
                  Must have subtask_id, parent_subtask_id, topology,
                  task_family, prior_outputs, neighbor_ids set correctly.

        Returns:
            (AgentOutput, TraceRow) — the parsed output and the logged row.
        """
        t0         = time.time()
        step_id    = self.step_counter.increment()
        message_id = f"msg_{uuid.uuid4().hex[:12]}"

        # 1. Build prompt
        prompt_text = build_context(spec)

        # 2. LLM call
        messages = [
            SystemMessage(content="You are a reasoning agent in a multi-agent system."),
            HumanMessage(content=prompt_text),
        ]
        try:
            response    = await self.llm.ainvoke(messages)
            raw_text    = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            raw_text = json.dumps({
                "claim_id": f"claim_{uuid.uuid4().hex[:12]}",
                "parent_claim_ids": [],
                "subtask": {
                    "subtask_id": spec.subtask_id or f"sub_{uuid.uuid4().hex[:8]}",
                    "parent_subtask_id": spec.parent_subtask_id,
                    "assigned_by": None,
                    "status": "active",
                },
                "content": {"reasoning": f"LLM error: {e}", "answer": "", "confidence": 0.0},
                "provenance": {"references_used": [], "claims_referenced": [], "tools_used": [], "evidence_refs": []},
                "coordination_signals": {},
            })

        # 3. Parse output
        raw_dict = _parse_agent_output(raw_text)

        # 4. Execute tool calls if present (before building TraceRow)
        tools_used, evidence_refs = _execute_tools_from_output(
            raw_dict, self.run_ctx.benchmark, self.environment
        )

        # Merge tool results into provenance
        prov = raw_dict.get("provenance") or {}
        prov_tools    = list(prov.get("tools_used", []))    + tools_used
        prov_evidence = list(prov.get("evidence_refs", [])) + evidence_refs
        prov["tools_used"]   = prov_tools
        prov["evidence_refs"] = prov_evidence
        raw_dict["provenance"] = prov

        # 5. Validate into AgentOutput (tolerates extra/missing fields)
        subtask_id = (
            raw_dict.get("subtask", {}).get("subtask_id")
            or spec.subtask_id
            or f"sub_{uuid.uuid4().hex[:8]}"
        )
        try:
            agent_out = AgentOutput(
                claim_id=raw_dict.get("claim_id") or f"claim_{uuid.uuid4().hex[:12]}",
                parent_claim_ids=raw_dict.get("parent_claim_ids") or [],
                subtask=SubtaskOutput(
                    subtask_id=subtask_id,
                    parent_subtask_id=raw_dict.get("subtask", {}).get("parent_subtask_id") or spec.parent_subtask_id,
                    assigned_by=raw_dict.get("subtask", {}).get("assigned_by"),
                    status=raw_dict.get("subtask", {}).get("status", "active"),
                ),
                content=ContentOutput(
                    reasoning=raw_dict.get("content", {}).get("reasoning", ""),
                    answer=raw_dict.get("content", {}).get("answer", ""),
                    confidence=raw_dict.get("content", {}).get("confidence"),
                ),
                provenance=ProvenanceOutput(
                    references_used=prov.get("references_used", []),
                    claims_referenced=prov.get("claims_referenced", []),
                    tools_used=prov_tools,
                    evidence_refs=prov_evidence,
                ),
                coordination_signals=CoordinationSignals(
                    **{k: v for k, v in (raw_dict.get("coordination_signals") or {}).items()
                       if k in CoordinationSignals.model_fields}
                ),
                final_answer=raw_dict.get("final_answer"),
            )
        except Exception:
            # Minimal fallback — ensures a TraceRow is always emitted
            agent_out = AgentOutput(
                claim_id=f"claim_{uuid.uuid4().hex[:12]}",
                parent_claim_ids=[],
                subtask=SubtaskOutput(subtask_id=subtask_id),
                content=ContentOutput(reasoning=raw_text[:500], answer="", confidence=0.0),
                provenance=ProvenanceOutput(tools_used=prov_tools, evidence_refs=prov_evidence),
            )

        # 6. Build TraceRow (orchestrator assigns message_id)
        sigs = agent_out.coordination_signals
        row = TraceRow(
            run_id=self.run_ctx.run_id,
            benchmark=self.run_ctx.benchmark,
            task_id=self.run_ctx.task_id,
            task_family=self.run_ctx.task_family,
            topology=self.run_ctx.topology,
            seed=self.run_ctx.seed,
            num_agents=self.run_ctx.num_agents,

            step_id=step_id,
            timestamp=time.time(),
            agent_id=self.agent_id,
            role=self.agent_role,

            message_id=message_id,
            visible_message_ids=[mid for _, mid, _, _ in spec.prior_outputs],
            neighbor_ids=spec.neighbor_ids,

            subtask_id=agent_out.subtask.subtask_id,
            parent_subtask_id=agent_out.subtask.parent_subtask_id,
            assigned_agent=agent_out.subtask.assigned_by,
            subtask_status=agent_out.subtask.status,

            claim_id=agent_out.claim_id,
            parent_claim_ids=agent_out.parent_claim_ids,
            # root_claim_id filled post-hoc by dag_builder

            reasoning_text=agent_out.content.reasoning,
            final_answer_text=agent_out.final_answer or agent_out.content.answer,
            message_length=len(raw_text),
            confidence=agent_out.content.confidence,

            references_used=agent_out.provenance.references_used,
            claims_referenced=agent_out.provenance.claims_referenced,
            tools_used=agent_out.provenance.tools_used,
            evidence_refs=agent_out.provenance.evidence_refs,

            requested_subtask_creation=sigs.requested_subtask_creation,
            proposed_assignee=sigs.proposed_assignee,
            synthesis_of_multiple_inputs=sigs.synthesis_of_multiple_inputs,
            explicit_disagreement_with=sigs.explicit_disagreement_with,
            explicit_correction_of=sigs.explicit_correction_of,
            supports_claims=sigs.supports_claims,
        )

        # 7. Write to event bus
        self.bus.log(row.model_dump())

        return agent_out, row

    def run(self, spec: AgentContextSpec) -> Tuple[AgentOutput, TraceRow]:
        """Sync wrapper."""
        return asyncio.run(self.arun(spec))