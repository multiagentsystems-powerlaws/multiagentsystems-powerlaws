"""
full_mesh.py  —  Fully Connected / Complete Graph topology
-----------------------------------------------------------
Structure
---------
  Every agent communicates with every other agent.
  All agents have full visibility of every other agent's latest output.

  agent_0 ↔ agent_1 ↔ agent_2 ↔ ... ↔ agent_{N-1}
  (all edges present, both directions)

Execution model
---------------
  Round-based: each round, active agents (selected via sparse activation)
  produce outputs sequentially. All agents have full visibility regardless
  of whether they were active that round.
  After MAX_ROUNDS, agent_0 synthesizes all outputs into a final answer.

  Design note: sparse activation means not all agents speak every round.
  This is scalable full-mesh (all-to-all visibility, cost-controlled
  activation), not canonical full-mesh (all agents active every round).
  This is intentional for large-N scaling experiments.

Power-law observables
---------------------
- Tests whether heavy tails are endogenous (topology-agnostic) or
  require structural constraints
- Contradiction bursts: with full visibility, every agent sees every
  disagreement → potentially large contradiction groups
- Endorsement clustering: agents may converge on a few dominant claims

Logging contract
----------------
Round 0:    propose_claim — no parents, each agent starts its own branch
Round r>0:  event_type=None — extractor infers revise/contradict/endorse
            from parent_claim_ids and coordination signals
Synthesis:  event_type=None — extractor detects merge from parent_claim_ids >= 2
            final claim stored in claims, metadata pointer updated
All claims: explicit claim_id and root_claim_id stored in state["claims"]
Metadata:   latest_claim_by_agent pointer map maintained each round

Prompt construction:
  - Both the per-round and synthesis nodes build their prompt via
    build_context(AgentContextSpec(...)), with prior_outputs supplied as
    (agent_id, message_id, claim_id, text) tuples.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END

from .base import (
    BaseTopology, MASState,
    make_agent_id, new_id, text_hash,
    _ordered_claim_ids, _root_claim_from_parents, _update_meta,
)
from context_builder import build_context, AgentContextSpec
from loggers.schemas import EventType, TopologyName

MAX_ROUNDS = 3


# ─────────────────────────────────────────────────────────────────
# Topology
# ─────────────────────────────────────────────────────────────────

class FullMeshTopology(BaseTopology):
    """Complete graph: every agent sees every other agent's output each round."""

    def name(self) -> TopologyName:
        return TopologyName.FULL_MESH

    def agent_ids(self) -> List[str]:
        return [make_agent_id("agent", i) for i in range(self.num_agents)]

    def edge_list(self) -> List[Tuple[str, str]]:
        ids   = self.agent_ids()
        edges = []
        for i, a in enumerate(ids):
            for j, b in enumerate(ids):
                if i != j:
                    edges.append((a, b))
        return edges

    def build_graph(self) -> Any:
        ids     = self.agent_ids()
        builder = StateGraph(MASState)

        # ── Round nodes ───────────────────────────────────────────────

        def make_round_node(round_idx: int):
            def round_fn(state: MASState) -> Dict:
                new_outputs    = dict(state["agent_outputs"])
                new_influence  = dict(state["influence"])
                new_claims     = dict(state["claims"])
                claim_ptr_updates: Dict[str, str] = {}

                # Sparse activation: cap active agents for large N
                from sparse_activation import select_active_agents
                _active = select_active_agents(
                    all_agents=ids, step_idx=round_idx,
                    agent_outputs=new_outputs, strategy="frontier",
                )
                # Record activation at current step before incrementing
                self._record_activation(self._step, _active)

                for agent_id in _active:
                    self._step += 1
                    self._maybe_snapshot()

                    # Peers with output this round (for lineage)
                    peers_with_output = [
                        p for p in ids if p != agent_id and new_outputs.get(p)
                    ]

                    # Event type and parent claim IDs — structural, not heuristic
                    if round_idx == 0:
                        ev_type             = "propose_claim"
                        py_parent_claim_ids = []
                    else:
                        ev_type             = None  # post-hoc: extractor infers revise/contradict/endorse
                        # Parents = ordered latest claims from visible peers
                        py_parent_claim_ids = _ordered_claim_ids(
                            state, peers_with_output, new_claims
                        )

                    claim_id      = new_id("claim")
                    root_claim_id = _root_claim_from_parents(
                        state, py_parent_claim_ids, claim_id, new_claims
                    )

                    ptr_map = dict((state.get("metadata") or {}).get("latest_claim_by_agent") or {})
                    prior_outputs_list = [
                        (p,
                         f"msg_mesh_r{round_idx}_{p}",
                         ptr_map.get(p, ""),
                         new_outputs[p])
                        for p in peers_with_output
                    ]

                    spec = AgentContextSpec(
                        task=state["task"],
                        agent_id=agent_id,
                        agent_role="worker",
                        topology="full_mesh",
                        task_family=self.task_family,
                        local_goal="Give your updated best answer. Note any contradictions.",
                        subtask_id=claim_id,
                        step=round_idx,
                        neighbor_ids=[p for p in ids if p != agent_id],
                        prior_outputs=prior_outputs_list,
                        available_tools=self._tool_names,
                    )
                    system_prompt = (
                        f"You are agent {agent_id} in a fully-connected deliberation "
                        f"(round {round_idx + 1}/{MAX_ROUNDS}). "
                        "Update your answer. Correct errors you find in peers."
                    )
                    user_content = build_context(spec)

                    output = self._call_llm(
                        agent_id=agent_id,
                        agent_role="peer",
                        system_prompt=system_prompt,
                        user_content=user_content,
                        event_type=ev_type,
                        claim_id=claim_id,
                        claim_depth=round_idx,
                        claim_status=None,  # post-hoc: extractor assigns revised/contradicted/merged
                        parent_claim_ids=py_parent_claim_ids,
                        root_claim_id=root_claim_id,
                        subtask_id=claim_id,
                        visible_neighbors=[p for p in ids if p != agent_id],
                        num_agents_involved=len(ids) - 1,
                        # No fake endorsement fields
                    )

                    new_outputs[agent_id]   = output
                    new_influence[agent_id] = new_influence.get(agent_id, 0) + 1
                    new_claims[claim_id]    = {
                        "agent_id":        agent_id,
                        "claim_id":        claim_id,
                        "parent_claim_ids": py_parent_claim_ids,
                        "root_claim_id":   root_claim_id,
                        "depth":           round_idx,
                        "text_hash":       text_hash(output),
                    }
                    claim_ptr_updates[agent_id] = claim_id

                new_meta = _update_meta(state, claim_ptr_updates)

                return {
                    "messages":      [AIMessage(content=f"Round {round_idx} complete", name="system")],
                    "agent_outputs": new_outputs,
                    "influence":     new_influence,
                    "claims":        new_claims,
                    "metadata":      new_meta,
                    "step":          self._step,
                }
            return round_fn

        # ── Synthesis node ────────────────────────────────────────────

        def synthesis(state: MASState) -> Dict:
            self._step += 1
            self._maybe_snapshot()

            aggregator = ids[0]

            # Collect ordered latest claim per agent — explicit, not keys()
            all_agent_claim_ids = _ordered_claim_ids(state, ids)
            final_claim_id      = new_id("claim")
            root_claim_id       = _root_claim_from_parents(
                state, all_agent_claim_ids, final_claim_id
            )

            # (original used 3-tuple (aid, cid, text) — old AgentContextSpec signature)
            active_agents = [aid for aid in ids if state["agent_outputs"].get(aid)]
            active_cids   = _ordered_claim_ids(state, active_agents)
            prior_outputs_list = [
                (aid,
                 f"msg_mesh_synth_{aid}",
                 cid,
                 state["agent_outputs"][aid])
                for aid, cid in zip(active_agents, active_cids)
            ]

            # (original had `user_content, _ = build_context(AgentContextSpec(...))`)
            spec = AgentContextSpec(
                task=state["task"],
                agent_id=aggregator,
                agent_role="synthesizer",
                topology="full_mesh",
                task_family=self.task_family,
                local_goal="Synthesize all agents. Resolve contradictions into one final answer.",
                subtask_id=final_claim_id,
                step=self._step,
                neighbor_ids=[a for a in ids if a != aggregator],
                prior_outputs=prior_outputs_list,
                available_tools=self._tool_names,
            )
            user_content = build_context(spec)

            output = self._call_llm(
                agent_id=aggregator,
                agent_role="synthesizer",
                system_prompt=(
                    "You are the synthesizer of a fully-connected multi-agent deliberation. "
                    "All agents have debated. Produce one final coherent answer."
                ),
                user_content=user_content,
                event_type=None,  # post-hoc: extractor detects merge from parent_claim_ids >= 2
                claim_id=final_claim_id,
                claim_depth=MAX_ROUNDS,
                # Explicit ordered parent claim IDs — all agents
                parent_claim_ids=all_agent_claim_ids,
                subtask_id=final_claim_id,
                root_claim_id=root_claim_id,
    
    
    
    
            )

            # Store final claim
            new_claims = dict(state["claims"])
            new_claims[final_claim_id] = {
                "agent_id":        aggregator,
                "claim_id":        final_claim_id,
                "parent_claim_ids": all_agent_claim_ids,
                "root_claim_id":   root_claim_id,
                "depth":           MAX_ROUNDS,
                "text_hash":       text_hash(output),
            }

            new_outputs   = dict(state["agent_outputs"])
            new_outputs["synthesizer_final"] = output
            new_influence = dict(state["influence"])
            new_influence[aggregator] = new_influence.get(aggregator, 0) + self.num_agents
            self._record_activation(self._step, [aggregator])

            new_meta = _update_meta(state, {aggregator: final_claim_id})

            return {
                "messages":      [AIMessage(content=output, name=aggregator)],
                "agent_outputs": new_outputs,
                "influence":     new_influence,
                "claims":        new_claims,
                "metadata":      new_meta,
                "final_answer":  output,
                "step":          self._step,
            }

        # ── Wire graph ────────────────────────────────────────────────
        round_names = [f"round_{r}" for r in range(MAX_ROUNDS)]
        for r, rname in enumerate(round_names):
            builder.add_node(rname, make_round_node(r))
        builder.add_node("synthesis", synthesis)

        builder.set_entry_point(round_names[0])
        for i in range(len(round_names) - 1):
            builder.add_edge(round_names[i], round_names[i + 1])
        builder.add_edge(round_names[-1], "synthesis")
        builder.add_edge("synthesis", END)

        return builder.compile()