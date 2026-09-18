"""
src/loggers/event_bus.py

Unchanged except for one addition: log(dict) method.

llm_agent.py calls bus.log(row.model_dump()) to write TraceRow dicts.
The existing log_event(AgentEvent) path is preserved for all topology
files that still use _acall_llm → _log_event → bus.log_event(AgentEvent).

Both paths write to events.jsonl. The post-hoc pipeline (run_pipeline.py)
reads all rows regardless of which schema produced them, since both
AgentEvent and TraceRow share the fields that event_extractor needs:
  event_type, claim_id, parent_claim_ids, subtask_id, coordination_signals.

The two schemas can coexist in events.jsonl during the migration period.
Once all topologies are updated to use llm_agent.py, log_event() can be removed.
"""

from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from .schemas import (
    AgentEvent,
    GraphSnapshot,
    RunConfig,
    RunMetadata,
    TopologyName,
)


class EventBus:
    """Thread-unsafe but fast JSONL event logger."""

    def __init__(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self._events_file    = open(run_dir / "events.jsonl",    "a", encoding="utf-8")
        self._snapshots_file = open(run_dir / "snapshots.jsonl", "a", encoding="utf-8")
        self._event_count = 0

    # ── Event logging ──────────────────────────────────────────────────

    def log_event(self, event: AgentEvent) -> None:
        """Existing path: topology nodes log via _log_event → AgentEvent."""
        self._events_file.write(event.model_dump_json() + "\n")
        self._events_file.flush()
        self._event_count += 1

    def log(self, row: Dict[str, Any]) -> None:
        """
        New path: llm_agent.py logs TraceRow dicts directly.
        Accepts any dict and writes it as a JSON line.
        No schema validation here — TraceRow is already validated upstream.
        """
        self._events_file.write(json.dumps(row, default=str) + "\n")
        self._events_file.flush()
        self._event_count += 1

    # ── Graph snapshot ─────────────────────────────────────────────────

    def log_snapshot(
        self,
        run_id:       str,
        step:         int,
        topology:     TopologyName | str,
        agent_ids:    List[str],
        edge_list:    List[Tuple[str, str]],
        edge_weights: Optional[Dict[str, float]] = None,
        influence_scores: Optional[Dict[str, float]] = None,
    ) -> None:
        topo_str     = topology.value if isinstance(topology, TopologyName) else str(topology)
        edge_weights = edge_weights or {}

        G = nx.DiGraph()
        G.add_nodes_from(agent_ids)
        for src, dst in edge_list:
            w = edge_weights.get(f"{src}->{dst}", 1.0)
            G.add_edge(src, dst, weight=w)

        degrees = [G.degree(n) for n in agent_ids]

        weighted_degrees: Dict[str, float] = {
            n: sum(edge_weights.get(f"{n}->{nb}", 1.0) for nb in G.successors(n))
            for n in agent_ids
        }
        agent_degrees: Dict[str, int] = dict(G.degree())

        try:
            bc = nx.betweenness_centrality(G)
            median_bc = sorted(bc.values())[len(bc) // 2] if bc else 0.0
            bridge_agents = [n for n, v in bc.items() if v > median_bc and v > 0]
        except Exception:
            bridge_agents = []

        try:
            communities = list(nx.community.greedy_modularity_communities(G.to_undirected()))
            community_assignments: Dict[str, int] = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    community_assignments[node] = i
            modularity = nx.community.modularity(G.to_undirected(), communities)
        except Exception:
            community_assignments = {n: 0 for n in agent_ids}
            modularity = 0.0

        m = G.number_of_edges() or 1
        try:
            graph_entropy = -sum(
                (d / (2 * m)) * math.log(d / (2 * m))
                for _, d in G.degree() if d > 0
            )
        except Exception:
            graph_entropy = 0.0

        try:
            undirected = G.to_undirected()
            largest_cc = max(nx.connected_components(undirected), key=len)
            sub = undirected.subgraph(largest_cc)
            avg_path = nx.average_shortest_path_length(sub) if len(sub) > 1 else 0.0
        except Exception:
            avg_path = 0.0

        try:
            clustering = nx.average_clustering(G.to_undirected())
        except Exception:
            clustering = 0.0

        try:
            n = len(agent_ids)
            max_deg = max(degrees) if degrees else 0
            star_sum = (n - 1) * (n - 2)
            centralization = (
                sum(max_deg - d for d in degrees) / star_sum if star_sum > 0 else 0.0
            )
        except Exception:
            centralization = 0.0

        snap = GraphSnapshot(
            run_id=run_id,
            step=step,
            topology=topo_str,
            num_agents=len(agent_ids),
            edge_list=list(edge_list),
            edge_weights=edge_weights,
            agent_degrees=agent_degrees,
            weighted_degrees=weighted_degrees,
            bridge_agents=bridge_agents,
            community_assignments=community_assignments,
            graph_entropy=round(graph_entropy, 4),
            average_path_length=round(avg_path, 4),
            clustering_coefficient=round(clustering, 4),
            modularity=round(modularity, 4),
            centralization_index=round(centralization, 4),
            influence_scores=influence_scores or {},
        )
        self._snapshots_file.write(snap.model_dump_json() + "\n")
        self._snapshots_file.flush()

    # ── Run metadata ───────────────────────────────────────────────────

    def write_run_config(self, config: RunConfig) -> None:
        path = self.run_dir / "run_config.json"
        path.write_text(config.model_dump_json(indent=2))

    def flush_run_outcome(self, metadata: RunMetadata) -> None:
        path = self.run_dir / "run_metadata.json"
        path.write_text(metadata.model_dump_json(indent=2))

    # ── Housekeeping ───────────────────────────────────────────────────

    def close(self) -> None:
        self._events_file.close()
        self._snapshots_file.close()

    @property
    def event_count(self) -> int:
        return self._event_count

    def name(self) -> str:
        return self.run_dir.name