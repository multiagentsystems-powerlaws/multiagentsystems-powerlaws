"""
src/prompts/topology_addenda.py
--------------------------------
Topology-specific addenda injected after the shared base prompt.

Design rule: these addenda constrain COMMUNICATION and VISIBILITY only.
They do NOT prescribe coordination event labels (revise_claim, merge_claims, etc.).
Those are inferred post hoc from traces.
Do not assume or invent communication outside what is explicitly provided.

Each addendum tells the agent:
  - who they can communicate with
  - what neighborhood is visible
  - what their role is (hub / worker / coordinator / etc.)
  - whether they may assign subtasks

Usage:
    from prompts.topology_addenda import TOPOLOGY_ADDENDA
    addendum = TOPOLOGY_ADDENDA["chain"]          # inject into prompt
    addendum = TOPOLOGY_ADDENDA["dynamic_reputation"]
"""

from __future__ import annotations

TOPOLOGY_ADDENDA: dict[str, str] = {

    "chain": """\
Topology: Chain.
You may use only the task, your own assigned state, and the visible output \
from the immediately preceding agent in the chain.
You cannot communicate backward or skip positions.
Your job is to improve, refine, or correct the visible reasoning before you.
If you identify errors or omissions, address them explicitly.
You do not have access to the full system state — only what is explicitly shown.\
""",

    "star": """\
Topology: Star.
If role == hub:
- You may assign subtasks to peripheral agents.
- You collect returned outputs and synthesize them into a final answer.
- Only synthesize from outputs actually received; do not fabricate worker results.
If role == worker:
- You only communicate with the hub.
- Solve your assigned subtask and return your best grounded reasoning.
- Do not assume access to other workers' outputs unless explicitly shown.
You do not have access to the full system state — only what is explicitly shown.\
""",

    "tree": """\
Topology: Hierarchical (Tree).
You may communicate only with your direct parent and/or direct children \
as specified in your runtime state.
Workers are typically assigned local subtasks.
Intermediate agents may receive outputs from children and combine them.
Supervisory agents may integrate information upward.
Do not assume global visibility outside your local hierarchy.
You do not have access to the full system state — only what is explicitly shown.\
""",

    "full_mesh": """\
Topology: Fully Connected.
You may use all currently visible peer outputs shown in your context.
Identify the strongest reasoning, detect gaps or contradictions, and \
integrate or correct as needed.
Do not echo consensus without independent justification.
You do not have access to the full system state — only what is explicitly shown.\
""",

    "sparse_mesh": """\
Topology: Sparse Mesh.
You may communicate only with your listed neighbors.
You do not have global visibility.
Reason from local information only.
If neighbors disagree, compare their arguments explicitly and justify \
which line of reasoning you follow.
You do not have access to the full system state — only what is explicitly shown.\
""",

    "hybrid_modular": """\
Topology: Hybrid Modular.
You belong to a local team (cluster). Within your cluster you have full \
visibility. Across clusters, only bridge agents relay information.
If you are a bridge agent: summarize and relay your cluster's outputs \
to connected clusters, preserving key disagreements when relevant. Summarize cluster outputs for external communication while preserving key disagreements when relevant.
If you are an intra-cluster worker: solve your assigned subtask using \
only your cluster's visible context.
You do not have access to the full system state — only what is explicitly shown.\
""",

    "dynamic_reputation": """\
Topology: Dynamic Reputation.
You have access only to peers selected by the current routing mechanism, \
based on their reputation scores at this step.
Review consulted outputs critically.
Do not blindly trust high-reputation peers — evaluate their reasoning, \
not their score.
Evaluate consulted outputs critically and form your own grounded conclusion.
You do not have access to the full system state — only what is explicitly shown.\
""",
}


def get_topology_addendum(topology_name: str) -> str:
    """
    Return the addendum for the given topology name.
    Raises KeyError for unknown topologies so misconfigurations are caught early.
    """
    if topology_name not in TOPOLOGY_ADDENDA:
        available = list(TOPOLOGY_ADDENDA.keys())
        raise KeyError(
            f"Unknown topology '{topology_name}'. Available: {available}"
        )
    return TOPOLOGY_ADDENDA[topology_name]