"""
src/prompts/task_addenda.py
----------------------------
Task-family-specific addenda injected after the topology addendum.

Design rule: these addenda shape domain-specific REASONING STYLE only.
They do NOT prescribe coordination event labels or interaction protocols.

Covers the 7 task families in the current benchmark suite:
  qa, reasoning, coding, planning, coordination, critique, synthesis

Usage:
    from prompts.task_addenda import TASK_ADDENDA
    addendum = TASK_ADDENDA["qa"]
    addendum = TASK_ADDENDA["planning"]
"""

from __future__ import annotations

TASK_ADDENDA: dict[str, str] = {

    "qa": """\
Task family: QA.
Prioritize factual grounding.
Resolve ambiguity explicitly — state which interpretation you are using.
Separate evidence from inference: distinguish clearly between what is known and what is inferred.
Return a direct answer when possible; flag if the question is unanswerable.\
""",

    "reasoning": """\
Task family: Reasoning.
Track assumptions carefully and make them explicit when they matter.
Compare plausible explanations before committing to one.
Prefer globally consistent reasoning over isolated local plausibility.
When uncertain, state the main source of uncertainty.\
""",

    "coding": """\
Task family: Coding.
Localize the issue clearly before proposing a fix.
Reason explicitly about code behavior, constraints, and edge cases.
Prefer concrete fixes over vague commentary.
If the fix is partial or uncertain, state the main remaining unknowns.\
""",

    "planning": """\
Task family: Planning.
Produce a feasible, dependency-aware plan rather than an idealized one.
Track important constraints such as time, budget, tools, or API limits when relevant.
Identify dependencies, bottlenecks, and opportunities for parallel execution when they matter.
If constraints make the plan infeasible, say so and propose a realistic alternative.\
""",

    "coordination": """\
Task family: Coordination.
Distinguish clearly between independent and interdependent subproblems.
Reason about prerequisites, blocking dependencies, and likely bottlenecks.
Prefer solutions that reduce unnecessary coordination overhead.
Flag points where one delayed component could stall the rest of the work.\
""",

    "critique": """\
Task family: Critique.
Identify specific errors or gaps; avoid generic feedback.
Distinguish factual errors, reasoning errors, and framing errors.
Propose concrete corrections for the issues you identify.
If the prior work is substantially correct, say so briefly and precisely.\
""",

    "synthesis": """\
Task family: Synthesis.
Integrate prior outputs into a coherent unified answer.
Identify where prior inputs agree, disagree, or complement one another when relevant.
Resolve important contradictions rather than obscuring them.
Prefer a well-justified integrated conclusion over simple concatenation.\
""",
}


def get_task_addendum(task_family: str) -> str:
    """
    Return the addendum for the given task family.
    Raises KeyError for unknown families so misconfigurations are caught early.
    """
    if task_family not in TASK_ADDENDA:
        available = list(TASK_ADDENDA.keys())
        raise KeyError(
            f"Unknown task family '{task_family}'. Available: {available}"
        )
    return TASK_ADDENDA[task_family]