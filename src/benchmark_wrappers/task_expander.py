"""
src/benchmark_wrappers/task_expander.py

Benchmark-conditioned workload expansion module (paper Section H).

Paper spec:
    K = min(5, |B(b,d)|)      seed tasks sampled from benchmark b, domain d
    A = 5                      target agents per task
    M = ceil(N / (K * A))     expanded tasks generated per seed
    Execution pool = K * M     expanded nodes only — seeds are evaluation anchors

Design contracts (fixes vs previous versions):
───────────────────────────────────────────────
1.  execution_pool = expanded_nodes only.
    Root is handled separately as a synthesis stage.
    Docstring, allocation, and pool_size all agree.

2.  No forced minimum-1 allocation.
    _allocate_agents uses largest-remainder with no floor override,
    so sum(budgets) == N exactly. active_agent_fraction is renamed
    allocated_agent_fraction() to clarify it is a pre-run planning
    statistic, not a runtime observation.

3.  Root depends on expanded_nodes, not seed_nodes.
    Seeds are not executed, so root cannot synthesise from their outputs.
    Root synthesises from the outputs of the expanded execution pool.

4.  LLM expansion requires ALL M tasks to be valid.
    Padding by duplication is rejected. If fewer than M valid tasks are
    returned after max_retries, a RuntimeError is raised for real runs.
    Duplicate descriptions and objectives are detected and rejected.

5.  Empty benchmark_pool guard in build().

6.  active_agent_fraction() → allocated_agent_fraction().

7.  _build_sparse_dag uses a temporary set to prevent duplicate edges.

8.  Cross-cluster edge direction is documented as a design choice.

9.  _extract_source_facts_from_original() is marked as a weak heuristic
    validator only; structured facts come from LLM output.

10. evaluate_accuracy() documents the missing bridge: caller must map
    root/expanded outputs back to seed.agent_answer before calling.
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional


# ── Constants ─────────────────────────────────────────────────────────────────

K_MAX    = 5   # K = min(5, |B(b,d)|)   paper Section H
A_TARGET = 5   # target agents per task  paper Section H


# ── Types ─────────────────────────────────────────────────────────────────────

DomainType = Literal["qa", "reasoning", "coding", "planning", "coordination"]
NodeType   = Literal["seed", "expanded", "root"]


# ── Scaling formulas ──────────────────────────────────────────────────────────

def num_seed_tasks(pool_size: int) -> int:
    """K = min(5, |B(b,d)|)."""
    return min(K_MAX, pool_size)


def num_expanded_per_seed(N: int, K: int) -> int:
    """M = ceil(N / (K * A)).  Requires K >= 1."""
    if K <= 0:
        raise ValueError(f"K must be >= 1, got {K}")
    return max(1, math.ceil(N / (K * A_TARGET)))


def total_pool_size(N: int, K: int) -> int:
    return K * num_expanded_per_seed(N, K)


def agents_per_task(N: int, K: int) -> float:
    pool = total_pool_size(N, K)
    return N / pool if pool > 0 else 0.0


# ── TaskNode ──────────────────────────────────────────────────────────────────

@dataclass
class TaskNode:
    """
    One node in the workload tree.

    Structured fields ensure agents receive grounded, fact-preserving
    prompts assembled by build_node_prompt() rather than free-text
    descriptions that can drift from the original task.

    agent_answer is populated by graph_runner at runtime, not here.
    For seed nodes, agent_answer must be set by the caller (e.g. from
    the root or expanded outputs) before evaluate_accuracy() is called.
    """
    node_id:              str
    description:          str
    node_type:            NodeType
    seed_parent_id:       Optional[str]
    depends_on:           List[str]

    # Structured fields — populated by LLM expansion or synthetic fallback
    objective:            str       = ""
    source_facts:         List[str] = field(default_factory=list)
    required_constraints: List[str] = field(default_factory=list)
    expected_answer_type: str       = "text"

    # Metadata
    ground_truth:         Optional[str] = None
    benchmark_source:     Optional[str] = None
    agent_budget:         int = 0

    # ── Semantic DAG and fact-scope fields ────────────────────────────────────
    dependency_policy: str = "independent"
    # Controls whether _build_sparse_dag may add within-cluster edges:
    #   "independent"    — no same-cluster predecessors (parallel subtasks)
    #   "may_depend"     — sparse predecessor wiring allowed (coding/planning)
    #   "requires_prior" — must depend on at least one prior node (chained)

    fact_scope: str = "local"
    # Describes the fact coverage this node was given:
    #   "local"  — partial/role-specific fact view
    #   "full"   — complete fact set (root, synthesis, or M=1 collapse)

    fact_sufficient: bool = True
    # Set by _validate_fact_sufficiency() after task creation.
    # If False, source_facts were augmented with fallback full facts.

    # Runtime — set by graph_runner after execution
    agent_answer:         Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "node_id":              self.node_id,
            "node_type":            self.node_type,
            "description":          self.description,
            "objective":            self.objective,
            "source_facts":         self.source_facts,
            "required_constraints": self.required_constraints,
            "expected_answer_type": self.expected_answer_type,
            "seed_parent_id":       self.seed_parent_id,
            "depends_on":           self.depends_on,
            "dependency_policy":    self.dependency_policy,
            "fact_scope":           self.fact_scope,
            "fact_sufficient":      self.fact_sufficient,
            "agent_budget":         self.agent_budget,
            "ground_truth":         self.ground_truth,
            "benchmark_source":     self.benchmark_source,
        }


# ── TaskTree ──────────────────────────────────────────────────────────────────

@dataclass
class TaskTree:
    """
    Fully expanded workload tree ready for agent execution.

    Execution contract:
        execution_pool  = expanded_nodes only.
        root_node       = synthesis stage, run after all execution_pool nodes.
        seed_nodes      = evaluation anchors only — NOT executed.

    Root depends on expanded_nodes (fix #3): root synthesises from the
    outputs agents actually produce, not from seed nodes that are never run.

    pool_size = len(expanded_nodes) = K * M.
    """
    benchmark:        str
    domain:           DomainType
    N:                int
    K:                int
    M:                int
    seed_nodes:       List[TaskNode]
    expanded_nodes:   List[TaskNode]
    root_node:        TaskNode
    dependency_dag:   Dict[str, List[str]]  # node_id → [downstream node_ids]
    agent_allocation: Dict[str, int] = field(default_factory=dict)

    @property
    def all_nodes(self) -> List[TaskNode]:
        return self.seed_nodes + self.expanded_nodes + [self.root_node]

    @property
    def execution_pool(self) -> List[TaskNode]:
        """
        Nodes agents actually execute: expanded_nodes only.
        Seeds are not in this list — they are evaluation anchors.
        Root is not in this list — it is the synthesis stage run after.
        """
        return self.expanded_nodes

    @property
    def pool_size(self) -> int:
        """K * M — expanded nodes only."""
        return len(self.expanded_nodes)

    @property
    def total_nodes(self) -> int:
        return len(self.all_nodes)

    def allocated_agent_fraction(self) -> float:
        """
        Fraction of N agents assigned to execution_pool + root nodes.
        This is a pre-run planning statistic, not a runtime observation.
        Seeds are excluded from allocation (they are not executed).
        """
        if self.N <= 0:
            return 0.0
        assigned = sum(self.agent_allocation.values())
        return assigned / self.N

    def node_by_id(self, node_id: str) -> Optional[TaskNode]:
        return next((n for n in self.all_nodes if n.node_id == node_id), None)

    def to_dict(self) -> dict:
        return {
            "benchmark":           self.benchmark,
            "domain":              self.domain,
            "N":                   self.N,
            "K":                   self.K,
            "M":                   self.M,
            "pool_size":           self.pool_size,
            "total_nodes":         self.total_nodes,
            "nodes":               [n.to_dict() for n in self.all_nodes],
            "dependency_dag":      self.dependency_dag,
            "agent_allocation":    self.agent_allocation,
        }

    def summary(self) -> str:
        m = validate_tree(self)
        return "\n".join([
            f"\n{'='*65}",
            f"Task Tree | {self.benchmark.upper()} | {self.domain.upper()} | N={self.N}",
            f"{'='*65}",
            f"  Seed tasks (K)             : {self.K}  (evaluation anchors, not executed)",
            f"  Expanded per seed (M)      : {self.M}",
            f"  Execution pool (K*M)       : {self.pool_size}",
            f"  Total nodes incl. root     : {self.total_nodes}",
            f"  Agents per task (target A) : {m['agents_per_task']:.2f}  (paper target: ~5)",
            f"  Allocated agent fraction   : {m['allocated_agent_fraction']:.1%}",
            f"  Root depends on            : {len(self.root_node.depends_on)} expanded nodes",
            f"  Avg deps per exec node     : {m['avg_deps_per_node']:.1f}",
            f"  Max DAG depth              : {m['max_dag_depth']}",
            "",
        ])


# ── Prompt assembly ───────────────────────────────────────────────────────────

def build_node_prompt(
    node:        TaskNode,
    dep_outputs: Optional[Dict[str, str]] = None,
) -> str:
    """
    Assemble a structured, grounded prompt for one TaskNode.
    Used by graph_runner per node — replaces free-text concatenation.
    """
    parts = [f"TASK:\n{node.description}"]

    if node.objective:
        parts.append(f"\nOBJECTIVE:\n{node.objective}")

    if node.source_facts:
        parts.append(
            "\nSOURCE FACTS (use these exactly — do not modify, omit, or assume away):\n"
            + "\n".join(f"  - {f}" for f in node.source_facts)
        )

    if node.required_constraints:
        parts.append(
            "\nREQUIRED CONSTRAINTS:\n"
            + "\n".join(f"  - {c}" for c in node.required_constraints)
        )

    if node.expected_answer_type and node.expected_answer_type != "text":
        parts.append(f"\nEXPECTED ANSWER TYPE: {node.expected_answer_type}")

    if dep_outputs:
        dep_lines = [
            f"[{dep_id}]:\n{ans[:500]}"
            for dep_id, ans in dep_outputs.items()
            if ans
        ]
        if dep_lines:
            parts.append(
                "\nPRIOR TASK OUTPUTS (results from dependency nodes):\n"
                + "\n\n".join(dep_lines)
            )

    return "\n".join(parts)


# ── LLM expansion ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a task decomposition expert. Given a benchmark task, generate "
    "structured sub-tasks that decompose it for parallel agent execution.\n"
    "CRITICAL RULES:\n"
    "  1. Copy exact quantities, entities, and constraints from the original "
    "task verbatim into source_facts. Do NOT paraphrase.\n"
    "  2. Do NOT introduce assumptions not in the original "
    "(no equal rates, no symmetric values, no invented data).\n"
    "  3. Each sub-task must be genuinely narrower — not a restatement.\n"
    "  4. Sub-tasks must be collectively exhaustive: solving all of them "
    "must be sufficient to answer the original.\n"
    "  5. Every sub-task must have a unique description and objective.\n"
    "Return ONLY valid JSON with no markdown or preamble."
)

_DOMAIN_INSTRUCTION: Dict[str, str] = {
    "qa": (
        "Generate {M} factual QA sub-questions, each narrower than the original "
        "and answerable from the source facts alone."
    ),
    "reasoning": (
        "Generate {M} reasoning sub-problems, each a genuine computation or "
        "logical deduction using only the exact quantities and constraints in "
        "the original. Do not assume symmetry, equal rates, or any unstated value."
    ),
    "coding": (
        "Generate {M} coding sub-tasks, each a specific implementable function, "
        "module, or bug fix. Preserve all interface constraints and type signatures."
    ),
    "planning": (
        "Generate {M} planning sub-problems, each covering a specific "
        "sub-route, time window, resource subset, or agent subgroup. "
        "Preserve all hard constraints exactly."
    ),
    "coordination": (
        "Generate {M} coordination sub-tasks, each addressing a specific "
        "agent interaction, resource allocation, or scheduling decision. "
        "Preserve all agent counts, resource limits, and timing constraints."
    ),
}

_USER_TEMPLATE = """\
Original task:
{description}

{domain_instruction}

Generate exactly {M} sub-tasks. Each must copy exact source facts verbatim.
Every sub-task must have a UNIQUE description and objective — no duplicates.

Return JSON only:
{{
  "expanded_tasks": [
    {{
      "id": "e1",
      "description": "<complete sub-task statement>",
      "objective": "<one sentence: exactly what must be computed or solved>",
      "source_facts": ["<exact fact verbatim from original>", ...],
      "required_constraints": ["<constraint that must not be violated>"],
      "expected_answer_type": "<numeric | boolean | ranked_list | code_patch | text>"
    }}
  ]
}}
"""


def _extract_source_facts_heuristic(description: str) -> List[str]:
    """
    Weak heuristic validator: extract numeric facts and named entities.

    NOTE: This is a coarse sanity check only. Structured source facts
    should come primarily from the LLM expansion output under constrained
    prompting. This function is not a reliable semantic extractor for
    code tasks, symbolic reasoning, or non-quantity-heavy text.
    """
    facts: List[str] = []

    # Numeric facts: extract quantity-bearing phrases verbatim
    numeric = re.findall(
        r'[\w\s]*\d+[\.,]?\d*\s*'
        r'(?:widgets|hours|staff|revenue|units|items|tasks|agents|'
        r'steps|days|minutes|km|kg|%|K|M|B|\$)[^\.,;]*',
        description, re.IGNORECASE
    )
    facts.extend(f.strip() for f in numeric if len(f.strip()) > 5)

    # Also capture target statements explicitly (e.g. "650-widget/day target")
    target = re.search(
        r'\b\d+[\.,]?\d*[\s\-\w/]*(?:target|goal|limit|threshold|quota)\b',
        description, re.IGNORECASE
    )
    if target:
        facts.append(target.group().strip())

    # Named entities — filter question/auxiliary words that are not real entities
    _BAD_ENTITIES = {
        "Which", "What", "When", "Where", "Why", "How", "Does", "Do",
        "Is", "Are", "Was", "Were", "Has", "Have", "Can", "Will", "Show",
        "Answer", "Give", "Find", "Compute", "Calculate", "Determine",
    }
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', description)
    facts.extend(e for e in entities if e not in _BAD_ENTITIES)

    return list(dict.fromkeys(facts))[:10]


# ── Part 4: Numeric-aware fact slicing ───────────────────────────────────────
# Applied in _slice_reasoning_facts: target_facts always include all numeric
# facts so computation-dependent subtasks are never under-specified.
# This is a general fix — not reasoning-specific.

# ── Part 3: Fact sufficiency validator ───────────────────────────────────────

_OBJECTIVE_SUFFICIENCY_RULES: Dict[str, dict] = {
    # objective keyword → {required fact patterns}
    # Each rule checks that source_facts contain enough ingredients for
    # the subtask objective. Generic enough to cover all domains.
    "total":      {"min_numeric": 2},
    "sum":        {"min_numeric": 2},
    "aggregate":  {"min_numeric": 2},
    "highest":    {"min_facts":   2},
    "compare":    {"min_facts":   2},
    "maximum":    {"min_facts":   2},
    "rank":       {"min_facts":   2},
    "target":     {"needs_numeric": True, "needs_target_keyword": True},
    "meet":       {"needs_numeric": True, "needs_target_keyword": True},
    "threshold":  {"needs_numeric": True, "needs_target_keyword": True},
    "patch":      {"needs_content": True},
    "implement":  {"needs_content": True},
    "fix":        {"needs_content": True},
}

_TARGET_KEYWORDS = {"target", "goal", "threshold", "quota", "meet", "exceed",
                    "below", "above", "requirement", "limit"}


def _validate_fact_sufficiency(
    objective:    str,
    source_facts: List[str],
    all_facts:    List[str],
) -> tuple:
    """
    Check whether source_facts contain enough ingredients for the objective.

    Returns (is_sufficient: bool, augmented_facts: List[str]).
    If not sufficient, returns (False, all_facts) so the caller can fall back
    to the full fact set.

    Rules are generic across domains:
      - "total/sum/aggregate" → need at least 2 numeric facts
      - "highest/compare/maximum/rank" → need at least 2 facts
      - "target/meet/threshold" → need a numeric fact + a target keyword fact
      - "patch/implement/fix" → need some non-trivial content

    This is a heuristic validator, not a semantic oracle. It prevents the
    most common failure mode (missing required numbers or target statements)
    without overfitting to any specific benchmark or task type.
    """
    obj_lower = objective.lower()
    facts     = source_facts

    # Find the matching rule (first keyword match wins)
    rule: dict = {}
    for kw, r in _OBJECTIVE_SUFFICIENCY_RULES.items():
        if kw in obj_lower:
            rule = r
            break

    if not rule:
        return True, facts   # No rule → assume sufficient

    # min_numeric: need at least N facts containing a digit
    if "min_numeric" in rule:
        n_numeric = sum(1 for f in facts if re.search(r'\d', f))
        if n_numeric < rule["min_numeric"]:
            return False, all_facts

    # min_facts: need at least N facts total
    if "min_facts" in rule:
        if len(facts) < rule["min_facts"]:
            return False, all_facts

    # needs_numeric + needs_target_keyword
    if rule.get("needs_numeric"):
        if not any(re.search(r'\d', f) for f in facts):
            return False, all_facts
    if rule.get("needs_target_keyword"):
        has_target = any(
            any(kw in f.lower() for kw in _TARGET_KEYWORDS)
            for f in facts
        )
        if not has_target:
            return False, all_facts

    # needs_content: at least one fact with > 10 chars
    if rule.get("needs_content"):
        if not any(len(f) > 10 for f in facts):
            return False, all_facts

    return True, facts


def _normalize(s: str) -> str:
    """Lowercase, collapse whitespace — for fuzzy substring matching."""
    return " ".join(s.lower().split())


def _validate_expanded_task(
    task: dict,
    original_description: str,
    original_numbers: List[str],
    seen_descriptions: set,
    seen_objectives: set,
) -> Optional[str]:
    """
    Validate one expanded task dict.
    Returns None if valid, error string if invalid.

    Checks:
      - required fields present and non-empty
      - source_facts non-empty
      - each source_fact is a verbatim (normalized) substring of the original
        OR the original contains every token in the fact (strict containment)
      - at least one original number appears in expanded content
      - no assumption-introducing language
      - description and objective are unique across this expansion batch
    """
    for f in ["id", "description", "objective", "source_facts"]:
        if not task.get(f):
            return f"missing or empty field '{f}'"

    desc   = task["description"].strip()
    obj    = task["objective"].strip()
    facts  = task.get("source_facts", [])

    if not facts:
        return "source_facts is empty"

    # Uniqueness across this expansion batch
    if desc.lower() in seen_descriptions:
        return f"duplicate description: '{desc[:60]}'"
    if obj.lower() in seen_objectives:
        return f"duplicate objective: '{obj[:60]}'"

    # Verbatim fact check: each source_fact must be a normalized substring of
    # the original description, OR every non-trivial token in the fact must
    # appear in the original. This enforces that the LLM is not inventing facts.
    orig_norm = _normalize(original_description)
    stopwords = {"a", "an", "the", "and", "or", "of", "in", "on", "at",
                 "to", "is", "are", "was", "were", "for", "with", "from"}
    unfaithful = []
    for fact in facts:
        fact_norm   = _normalize(fact)
        # Accept if the normalized fact appears as a substring
        if fact_norm in orig_norm:
            continue
        # Accept if every content token in the fact appears in the original
        tokens = [t for t in fact_norm.split() if t not in stopwords and len(t) > 2]
        if tokens and all(t in orig_norm for t in tokens):
            continue
        unfaithful.append(fact[:60])
    if unfaithful:
        return (
            f"source_facts not found verbatim in original — "
            f"unfaithful facts: {unfaithful[:3]}"
        )

    # At least one original number must appear in expanded content
    combined = (desc + " " + obj + " " + " ".join(facts)).lower()
    if original_numbers:
        if not any(n in combined for n in original_numbers):
            return (
                f"no original quantities found in expanded task — "
                f"expected one of {original_numbers[:5]}"
            )

    # Reject unsupported assumptions
    bad_phrases = [
        "same rate", "equal rate", "same output", "assume equal",
        "assuming same", "same for each", "uniform rate", "equal distribution",
    ]
    for phrase in bad_phrases:
        if phrase in combined:
            return f"assumption-introducing phrase: '{phrase}'"

    return None


def _call_llm_expand(
    seed_task:   dict,
    domain:      DomainType,
    M:           int,
    model:       str = "gpt-4o-mini",
    max_retries: int = 3,
) -> List[dict]:
    """
    Expand one seed task into exactly M valid, unique structured sub-tasks.

    Requires ALL M tasks to pass validation (fix #4).
    Duplication padding is rejected — if M valid tasks cannot be produced
    after max_retries, raises RuntimeError.
    """
    import os
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    original   = seed_task["description"]
    orig_nums  = re.findall(r'\b\d+\.?\d*\b', original)

    domain_instr = _DOMAIN_INSTRUCTION.get(
        domain, _DOMAIN_INSTRUCTION["reasoning"]
    ).format(M=M)

    user_msg = _USER_TEMPLATE.format(
        description=original,
        domain_instruction=domain_instr,
        M=M,
    )

    last_err: Exception = RuntimeError("No attempts made.")
    rejection_notes = ""

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=4096,
                temperature=0.5,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg + rejection_notes},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()
            s, e = raw.find("{"), raw.rfind("}")
            if s == -1 or e == -1:
                raise ValueError("No JSON object found in response.")
            parsed = json.loads(raw[s:e+1])
            tasks  = parsed.get("expanded_tasks", [])
            if not tasks:
                raise ValueError("expanded_tasks list is empty.")

            # Validate all tasks; track uniqueness within this batch
            seen_descs = set()
            seen_objs  = set()
            valid: List[dict] = []
            reasons: List[str] = []

            for t in tasks:
                reason = _validate_expanded_task(
                    t, original, orig_nums, seen_descs, seen_objs
                )
                if reason is None:
                    seen_descs.add(t["description"].strip().lower())
                    seen_objs.add(t["objective"].strip().lower())
                    valid.append(t)
                else:
                    reasons.append(f"  task {t.get('id','?')}: {reason}")

            if len(valid) >= M:
                return valid[:M]

            # Not enough valid tasks — tell the model what went wrong
            rejection_notes = (
                f"\n\nPREVIOUS ATTEMPT REJECTED ({len(valid)}/{M} valid):\n"
                + "\n".join(reasons)
                + f"\n\nRegenerate all {M} tasks fixing these issues. "
                "Ensure every sub-task is unique and copies exact source facts."
            )
            raise ValueError(
                f"{len(valid)}/{M} tasks passed validation: {reasons}"
            )

        except Exception as ex:
            last_err = ex
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"LLM expansion failed for '{seed_task.get('node_id','?')}' "
        f"after {max_retries} attempts: {last_err}"
    ) from last_err


# ── Synthetic expansion (no API) ──────────────────────────────────────────────

def _slice_reasoning_facts(description: str) -> dict:
    """
    Partition source facts into per-role views for coordination stress mode.

    Returns a dict with four views:
      shift_facts:  entity-level facts (e.g. per-shift or per-department data)
      total_facts:  aggregation-relevant facts + shift facts
      target_facts: threshold/goal facts + total facts (full context for decision)
      all_facts:    complete set (used by root and fallback)

    This gives each subtask a partial but coherent view of the problem,
    forcing real information synthesis at the root stage rather than
    trivial agreement from full shared context.
    """
    all_facts = _extract_source_facts_heuristic(description)

    shift_facts: List[str] = []
    total_facts: List[str] = []
    target_facts: List[str] = []

    # Entity-level facts: "Name: <quantity> in <quantity>" patterns
    for m in re.finditer(
        r'[A-Z][a-z]+:\s*[^.?!;]*\d+[^.?!;]*', description
    ):
        shift_facts.append(m.group(0).strip())

    # Total/aggregate facts
    for m in re.finditer(
        r'[^.?!;]*\b(total|overall|combined|aggregate|sum)\b[^.?!;]*',
        description, re.IGNORECASE
    ):
        total_facts.append(m.group(0).strip())

    # Target/threshold facts
    for m in re.finditer(
        r'[^.?!;]*\b(target|goal|threshold|quota|meet|exceed|below|above|requirement)\b[^.?!;]*',
        description, re.IGNORECASE
    ):
        target_facts.append(m.group(0).strip())

    def _dedup(lst: List[str]) -> List[str]:
        return list(dict.fromkeys(lst))

    # Always include all numeric facts in target_facts so computation-dependent
    # subtasks (total+target) are never under-specified. This is the general
    # fix: any subtask whose objective requires computation must see the numbers.
    numeric_facts = [f for f in all_facts if re.search(r'\d', f)]

    sf = _dedup(shift_facts) or all_facts
    tf = _dedup(total_facts + shift_facts) or all_facts
    xf = _dedup(target_facts + numeric_facts) or all_facts   # always has numbers

    return {
        "shift_facts":  sf,
        "total_facts":  tf,
        "target_facts": xf,
        "all_facts":    all_facts,
    }


def _synthetic_expand_reasoning(
    seed_task:          dict,
    M:                  int,
    seed_idx:           int,
    stress_coordination: bool = True,
) -> List[dict]:
    """
    Operator-aware, M-aware, coordination-stress synthetic decomposition
    for reasoning tasks.

    Base behaviour (always):
      - Detects operators (per-unit/compare, total, target, ranking).
      - Packs them into exactly M tasks without truncating any operator.
      - Merges compatible operators when M is tight (fixes the earlier drop bug).

    Coordination stress mode (stress_coordination=True, default):
      - Each subtask receives a PARTIAL fact view, not the global fact list.
        This forces real information synthesis at the root — agents cannot
        trivially agree because they see different subsets of the problem.
      - When M >= 3, adds one alternate-derivation task that computes an
        aggregate measure that may conflict with the per-entity maximum.
        This induces revise_claim and contradict_claim events naturally.
      - Constraints explicitly prevent tasks from pre-solving the whole
        problem or using facts outside their assigned view.

    stress_coordination=False gives the clean cooperative baseline
    (useful for correctness checks and plumbing tests).
    """
    description  = seed_task["description"]
    source_facts = _extract_source_facts_heuristic(description)
    desc_lower   = description.lower()
    is_numeric   = bool(re.search(r'\d', description))

    fact_views = _slice_reasoning_facts(description) if stress_coordination else {
        "shift_facts":  source_facts,
        "total_facts":  source_facts,
        "target_facts": source_facts,
        "all_facts":    source_facts,
    }

    # ── Operator detection ────────────────────────────────────────────────────
    has_per_unit = any(k in desc_lower for k in
                       ["per hour", "per day", "per shift", "per unit",
                        "per staff", "per agent", "rate", "productivity"])
    has_compare  = any(k in desc_lower for k in
                       ["highest", "largest", "most", "lowest", "least",
                        "which shift", "which department", "compare", "best"])
    has_total    = any(k in desc_lower for k in
                       ["total", "overall", "sum", "combined", "aggregate"])
    has_target   = any(k in desc_lower for k in
                       ["target", "goal", "meet", "exceed", "below", "above",
                        "threshold", "quota", "requirement"])
    has_ranking  = any(k in desc_lower for k in
                       ["rank", "order", "sort"])
    # Comparison implies implicit ranking
    if has_compare:
        has_ranking = True

    bucket_a = has_per_unit or has_compare   # rate/compare
    bucket_b = has_total or has_target       # total/target
    bucket_c = has_ranking                   # ranking

    # ── M=1: everything in one task ───────────────────────────────────────────
    if M == 1:
        return [{
            "id": "e1",
            "description": (
                "Solve the full reasoning problem: compute all needed intermediate "
                "quantities, identify the correct comparison outcome, and check any "
                "stated target or threshold. Use only the exact values provided."
            ),
            "objective":   "Produce all intermediate computations and the final answer.",
            "source_facts":         fact_views["all_facts"],
            "required_constraints": [
                "Use only the facts explicitly stated in source_facts.",
                "Do not introduce values not present in the original task.",
            ],
            "expected_answer_type": "numeric" if is_numeric else "text",
            "dependency_policy":    "independent",
            "fact_scope":           "full",
        }]

    tasks: List[dict] = []

    # ── Bucket A: per-unit rate + comparison ──────────────────────────────────
    if bucket_a:
        tasks.append({
            "id":          f"e{len(tasks)+1}",
            "description": (
                "Compute the per-unit rate for each entity using only the "
                "entity-level facts provided and identify which has the highest rate."
            ),
            "objective":   "Determine the per-unit rate for each entity and find the maximum.",
            "source_facts":         fact_views["shift_facts"],
            "required_constraints": [
                "Compute rates exactly from the stated values — do not assume equal rates.",
                "Use only the facts in source_facts. Do not use target or aggregate facts.",
                "Solve only the objective stated here — do not answer the full original problem.",
            ],
            "expected_answer_type": "numeric",
            "dependency_policy":    "independent",
        })

    # ── Bucket B: total + target ──────────────────────────────────────────────
    if bucket_b:
        if has_total and has_target:
            desc_str = (
                "Compute the total combined output by summing all stated individual "
                "values, then determine whether it meets the stated target or threshold."
            )
            obj_str = "Compute total output and determine whether the target is met."
            atype   = "boolean"
        elif has_total:
            desc_str = (
                "Compute the total combined output by summing all individual values "
                "stated in the problem. Do not estimate or infer missing values."
            )
            obj_str = "Compute the exact total by summing all stated individual values."
            atype   = "numeric"
        else:
            desc_str = (
                "Compare the relevant computed value against the stated target or "
                "threshold and determine whether the condition is met."
            )
            obj_str = "Determine whether the computed value meets the stated target."
            atype   = "boolean"

        tasks.append({
            "id":          f"e{len(tasks)+1}",
            "description": desc_str,
            "objective":   obj_str,
            "source_facts":         fact_views["target_facts"],
            "required_constraints": [
                "Use only the values explicitly stated in source_facts.",
                "State clearly whether any condition is met, with justification.",
                "Solve only the objective stated here — do not answer the full original problem.",
            ],
            "expected_answer_type": atype,
            "dependency_policy":    "independent",
        })

    # ── Bucket C: ranking ─────────────────────────────────────────────────────
    # ── Stress task: alternate derivation (induces contradiction/revision) ────
    # Prioritized over ranking bucket when stress_coordination=True and M>=3,
    # because ranking is already implicit in bucket A (comparison). The stress
    # task computes an aggregate average rate which may conflict with the
    # per-entity maximum, creating genuine revision/contradiction pressure.
    if stress_coordination and M >= 3 and len(tasks) < M and (has_per_unit or has_compare):
        tasks.append({
            "id":          f"e{len(tasks)+1}",
            "description": (
                "Compute the overall average rate across all entities and explain "
                "how it compares to the maximum per-entity rate. Note any discrepancy."
            ),
            "objective": (
                "Derive an aggregate average rate and identify how it differs "
                "from the maximum single-entity rate."
            ),
            "source_facts":         fact_views["all_facts"],
            "required_constraints": [
                "Use exact stated values only — no estimation.",
                "Explicitly distinguish average aggregate rate from maximum per-entity rate.",
                "Do not confuse the two measures.",
            ],
            "expected_answer_type": "numeric",
            "dependency_policy":    "independent",
        })

    # ── Bucket C: ranking ─────────────────────────────────────────────────────
    # Added after stress task so stress task takes priority when M is tight.
    if bucket_c and len(tasks) < M:
        tasks.append({
            "id":          f"e{len(tasks)+1}",
            "description": (
                "Rank all entities in the problem by the relevant metric from "
                "highest to lowest, using only the entity-level facts provided."
            ),
            "objective":   "Produce a ranked ordering of all entities by the relevant metric.",
            "source_facts":         fact_views["shift_facts"],
            "required_constraints": [
                "Rank based only on computed values from source_facts.",
                "Include all entities mentioned in the task.",
                "Do not infer missing values.",
            ],
            "expected_answer_type": "ranked_list",
            "dependency_policy":    "independent",
        })

    # ── Pad to M with generic intermediate tasks ──────────────────────────────
    while len(tasks) < M:
        i = len(tasks)
        tasks.append({
            "id":          f"e{i+1}",
            "description": (
                f"Compute one intermediate quantity needed for the final answer "
                f"(sub-computation {i+1} of {M}). Use only the stated values."
            ),
            "objective":   f"Compute intermediate quantity {i+1} from source facts.",
            "source_facts":         fact_views["all_facts"],
            "required_constraints": [
                "Use only the facts explicitly stated in source_facts.",
                "Do not introduce values not present in the original task.",
                "Solve only the intermediate step stated — do not answer the full problem.",
            ],
            "expected_answer_type": "numeric" if is_numeric else "text",
            "dependency_policy":    "independent",
        })

    # Re-index IDs
    for i, t in enumerate(tasks):
        t["id"] = f"e{i+1}"

    return tasks[:M]


def _synthetic_expand(
    seed_task:           dict,
    domain:              DomainType,
    M:                   int,
    seed_idx:            int,
    stress_coordination: bool = True,
) -> List[dict]:
    """
    Deterministic structured expansion for testing without API calls.

    For reasoning tasks: routes to _synthetic_expand_reasoning() with
    operator-aware, M-aware, coordination-stress decomposition.

    For other domains: structured clause-based fallback.

    stress_coordination=True (default): partial fact views + alternate
      derivation task to induce revisions and contradictions.
    stress_coordination=False: clean cooperative baseline for plumbing tests.
    """
    if domain == "reasoning":
        return _synthetic_expand_reasoning(
            seed_task, M, seed_idx,
            stress_coordination=stress_coordination,
        )

    # ── Non-reasoning fallback ────────────────────────────────────────────────
    description  = seed_task["description"]
    source_facts = _extract_source_facts_heuristic(description)

    # Split by meaningful clauses — avoid single-word fragments
    clauses = [s.strip() for s in re.split(r'[.?!;]', description)
               if len(s.strip()) > 25]
    if not clauses:
        clauses = [description]

    verb = {
        "qa": "sub-question", "coding": "sub-task",
        "planning": "sub-plan", "coordination": "sub-task",
    }.get(domain, "sub-task")

    tasks = []
    for i in range(M):
        clause = clauses[i % len(clauses)]
        tasks.append({
            "id":          f"e{i+1}",
            "description": (
                f"{verb.capitalize()} {seed_idx+1}.{i+1} "
                f"(part {i+1} of {M}): {clause}"
            ),
            "objective": (
                f"Solve sub-problem {i+1} of {M}: {clause[:80]}"
            ),
            "source_facts":         source_facts[:5] or [description[:100]],
            "required_constraints": [
                "Use only the facts explicitly stated in source_facts.",
                "Do not introduce values not present in the original task.",
            ],
            "expected_answer_type": (
                "numeric" if re.search(r'\d', description) else "text"
            ),
            "dependency_policy":    "may_depend",
        })
    return tasks


# ── Sparse DAG construction ───────────────────────────────────────────────────

def _build_sparse_dag(
    expanded_nodes: List[TaskNode],
    root_node:      TaskNode,
    rng:            random.Random,
    avg_deps:       float = 2.0,
) -> Dict[str, List[str]]:
    """
    Build a semantics-aware sparse DAG over expanded_nodes + root.
    Seeds are NOT included — they are not in the execution DAG.

    DAG edges are governed by node.dependency_policy (Part 2):
      "independent"    — no within-cluster edges; node runs in parallel.
      "may_depend"     — sparse predecessor wiring as before (coding/planning).
      "requires_prior" — must depend on the immediately prior cluster node.

    Cross-cluster edges (15% probability) are only added to nodes whose
    policy is "may_depend" or "requires_prior" — never to "independent" nodes.

    Root always depends on ALL expanded nodes regardless of policy.
    Duplicate edges prevented via per-node set.
    """
    all_exec  = expanded_nodes + [root_node]
    forward_dag: Dict[str, List[str]] = {n.node_id: [] for n in all_exec}

    clusters: Dict[str, List[TaskNode]] = {}
    for n in expanded_nodes:
        clusters.setdefault(n.seed_parent_id or "_none", []).append(n)

    # Within-cluster wiring — respects dependency_policy
    for seed_id, cluster in clusters.items():
        for i, node in enumerate(cluster):
            predecessors = cluster[:i]
            if not predecessors:
                continue

            policy = node.dependency_policy

            if policy == "independent":
                # No same-cluster edges — node is a parallel branch
                continue

            elif policy == "requires_prior":
                # Chain to the immediately prior node only
                pred    = predecessors[-1]
                dep_set = set(node.depends_on)
                if pred.node_id not in dep_set:
                    node.depends_on.append(pred.node_id)
                    dep_set.add(pred.node_id)
                    forward_dag[pred.node_id].append(node.node_id)

            else:  # "may_depend" — sparse wiring
                k      = min(len(predecessors), max(1, round(avg_deps)))
                chosen = rng.sample(predecessors, k)
                dep_set: set = set(node.depends_on)
                for pred in chosen:
                    if pred.node_id not in dep_set:
                        node.depends_on.append(pred.node_id)
                        dep_set.add(pred.node_id)
                        forward_dag[pred.node_id].append(node.node_id)

    # Cross-cluster edges — only for may_depend / requires_prior nodes
    cluster_keys = list(clusters.keys())
    for seed_id, cluster in clusters.items():
        others = [k for k in cluster_keys if k != seed_id]
        if not others:
            continue
        for node in cluster:
            if node.dependency_policy == "independent":
                continue
            if rng.random() < 0.15:
                other_nodes = clusters[rng.choice(others)]
                if other_nodes:
                    pred    = rng.choice(other_nodes)
                    dep_set = set(node.depends_on)
                    if pred.node_id not in dep_set:
                        node.depends_on.append(pred.node_id)
                        forward_dag[pred.node_id].append(node.node_id)

    # Root depends on ALL expanded nodes (fix #3)
    # Root synthesises from actual execution outputs.
    root_node.depends_on = [n.node_id for n in expanded_nodes]
    for n in expanded_nodes:
        forward_dag[n.node_id].append(root_node.node_id)

    return forward_dag


# ── Agent allocation ──────────────────────────────────────────────────────────

def _allocate_agents(tree: TaskTree) -> TaskTree:
    """
    Distribute exactly N agents across execution_pool + root (fix #2).

    Uses largest-remainder method: guarantees sum == N exactly.
    No forced minimum of 1 — some nodes can receive 0 budget at small N.
    Seeds are excluded (not in execution_pool).

    Pool nodes weighted 1.0, root weighted 1.5.
    """
    exec_nodes = tree.execution_pool   # expanded_nodes only
    all_w      = exec_nodes + [tree.root_node]
    N          = tree.N

    if not all_w or N <= 0:
        return tree

    weights  = [1.0] * len(exec_nodes) + [1.5]
    total_w  = sum(weights)
    exact    = [w / total_w * N for w in weights]
    floors   = [int(x) for x in exact]
    diff     = N - sum(floors)

    # Give remainder to nodes with largest fractional parts
    fracs = sorted(
        ((exact[i] - floors[i], i) for i in range(len(floors))),
        reverse=True,
    )
    for _, idx in fracs[:diff]:
        floors[idx] += 1

    allocation: Dict[str, int] = {}
    for node, budget in zip(all_w, floors):
        allocation[node.node_id] = budget
        node.agent_budget        = budget

    tree.agent_allocation = allocation
    return tree


# ── Main class ────────────────────────────────────────────────────────────────

class TaskExpander:
    """
    Benchmark-conditioned workload expansion (paper Section H).
    """

    def __init__(
        self,
        benchmark: str,
        domain:    DomainType,
        seed:      int = 42,
        model:     str = "gpt-4o-mini",
    ):
        self.benchmark = benchmark
        self.domain    = domain
        self.seed      = seed
        self.model     = model

    def build(
        self,
        N:                   int,
        benchmark_pool:      List[dict],
        use_llm:             bool = True,
        stress_coordination: bool = True,
    ) -> TaskTree:
        """
        Build the workload tree for N agents.

        Args:
            N:                   agent count for this run
            benchmark_pool:      list of dicts with keys:
                                   "node_id"     str
                                   "description" str
                                 optional: "ground_truth", "benchmark_source"
            use_llm:             True  = LLM-based structured expansion
                                 False = deterministic synthetic expansion (tests)
            stress_coordination: True (default) = partial fact views + alternate
                                 derivation task to induce coordination pressure.
                                 False = clean cooperative baseline (plumbing tests).
                                 Only applies when use_llm=False.

        Raises:
            ValueError: if benchmark_pool is empty (fix #5)
        """
        # Fix #5: guard against empty pool
        if not benchmark_pool:
            raise ValueError(
                "benchmark_pool must be non-empty. "
                "Pass at least one task dict with 'node_id' and 'description'."
            )

        rng = random.Random(self.seed)
        K   = num_seed_tasks(len(benchmark_pool))

        # ── Step 1: Sample K seed tasks ──────────────────────────────────────
        sampled    = rng.sample(benchmark_pool, K)
        seed_nodes = [
            TaskNode(
                node_id=t["node_id"],
                description=t["description"],
                node_type="seed",
                seed_parent_id=None,
                depends_on=[],
                objective=f"Evaluate: {t['description'][:100]}",
                source_facts=_extract_source_facts_heuristic(t["description"]),
                required_constraints=[
                    "Use only the facts explicitly given in this task."
                ],
                expected_answer_type=(
                    "numeric" if re.search(r'\d', t["description"]) else "text"
                ),
                ground_truth=t.get("ground_truth"),
                benchmark_source=t.get("benchmark_source", self.benchmark),
                dependency_policy="independent",
                fact_scope="full",
                fact_sufficient=True,
            )
            for t in sampled
        ]

        # ── Step 2: Expand each seed into M structured sub-tasks ─────────────
        M = num_expanded_per_seed(N, K)
        expanded_nodes: List[TaskNode] = []

        for idx, seed_node in enumerate(seed_nodes):
            seed_dict = {
                "node_id":     seed_node.node_id,
                "description": seed_node.description,
            }

            raw_tasks = (
                _call_llm_expand(seed_dict, self.domain, M, model=self.model)
                if use_llm
                else _synthetic_expand(seed_dict, self.domain, M, seed_idx=idx,
                                       stress_coordination=stress_coordination)
            )

            # Trim to M. Padding only allowed in synthetic mode.
            # LLM mode must return exactly M valid tasks (enforced in _call_llm_expand).
            raw_tasks = raw_tasks[:M]
            if len(raw_tasks) < M:
                if use_llm:
                    raise RuntimeError(
                        f"LLM expansion for seed '{seed_node.node_id}' returned "
                        f"{len(raw_tasks)} tasks but {M} required."
                    )
            while len(raw_tasks) < M:
                i = len(raw_tasks)
                raw_tasks.append({
                    "id":          f"e{i+1}",
                    "description": f"Sub-task {i+1} of {M}: {seed_node.description[:80]}",
                    "objective":   f"Solve part {i+1} of {M} (seed {idx+1}).",
                    "source_facts":         seed_node.source_facts,
                    "required_constraints": ["Use only original task facts."],
                    "expected_answer_type": seed_node.expected_answer_type,
                })

            for t in raw_tasks:
                raw_facts  = t.get("source_facts", seed_node.source_facts)
                all_facts  = seed_node.source_facts
                policy     = t.get("dependency_policy", "independent")

                # Part 3: validate fact sufficiency and fallback if needed
                is_suff, final_facts = _validate_fact_sufficiency(
                    objective=t.get("objective", ""),
                    source_facts=raw_facts,
                    all_facts=all_facts,
                )

                expanded_nodes.append(TaskNode(
                    node_id=f"{seed_node.node_id}_exp_{t['id']}",
                    description=t["description"],
                    node_type="expanded",
                    seed_parent_id=seed_node.node_id,
                    depends_on=[],
                    objective=t.get("objective", ""),
                    source_facts=final_facts,
                    required_constraints=t.get("required_constraints", []),
                    expected_answer_type=t.get("expected_answer_type", "text"),
                    benchmark_source=self.benchmark,
                    dependency_policy=policy,
                    fact_scope=t.get("fact_scope", "local"),
                    fact_sufficient=is_suff,
                ))

        # ── Step 3: Root synthesis node ───────────────────────────────────────
        # Root carries the original seed descriptions and facts so the synthesis
        # prompt (via build_node_prompt) remains grounded in the original problem,
        # not only in potentially drifted sub-task outputs.
        all_seed_facts = []
        all_seed_descs = []
        for sn in seed_nodes:
            all_seed_descs.append(f"[{sn.node_id}]: {sn.description}")
            all_seed_facts.extend(sn.source_facts)
        # Deduplicate facts while preserving order
        seen_facts: set = set()
        deduped_facts: List[str] = []
        for f in all_seed_facts:
            if f not in seen_facts:
                deduped_facts.append(f)
                seen_facts.add(f)

        root_node = TaskNode(
            node_id="root_000",
            description=(
                f"Synthesise the outputs from all {len(expanded_nodes)} "
                f"sub-tasks into one complete, coherent answer to the original "
                f"{self.benchmark} {self.domain} problem.\n\n"
                f"Original problem(s):\n" + "\n".join(all_seed_descs)
            ),
            node_type="root",
            seed_parent_id=None,
            depends_on=[],   # filled by _build_sparse_dag
            objective="Produce one final unified answer from all sub-task outputs, faithful to the original problem facts.",
            source_facts=deduped_facts,   # original seed facts carried forward
            required_constraints=[
                "Reconcile any conflicting answers from sub-tasks.",
                "Remain faithful to the original problem facts in source_facts.",
                "Produce one unified answer that directly addresses the original problem.",
            ],
            expected_answer_type="text",
            dependency_policy="requires_prior",   # root must see expanded outputs
            fact_scope="full",
            fact_sufficient=True,
        )

        # ── Step 4: Build sparse DAG ──────────────────────────────────────────
        # Seeds are NOT included in the execution DAG.
        # Root depends on all expanded_nodes (fix #3).
        dag = _build_sparse_dag(expanded_nodes, root_node, rng)

        # ── Step 5: Assemble and allocate ─────────────────────────────────────
        tree = TaskTree(
            benchmark=self.benchmark,
            domain=self.domain,
            N=N,
            K=K,
            M=M,
            seed_nodes=seed_nodes,
            expanded_nodes=expanded_nodes,
            root_node=root_node,
            dependency_dag=dag,
        )
        return _allocate_agents(tree)


# ── Validation ────────────────────────────────────────────────────────────────

def validate_tree(tree: TaskTree) -> dict:
    """
    Compute health metrics for a TaskTree.
    Returns dict matching paper Figure 18 / Table H targets.
    """
    from collections import deque

    exec_nodes = tree.execution_pool + [tree.root_node]
    id_set     = {n.node_id for n in exec_nodes}
    in_deg     = {n.node_id: 0 for n in exec_nodes}
    fwd: Dict[str, List[str]] = {n.node_id: [] for n in exec_nodes}
    for n in exec_nodes:
        for dep in n.depends_on:
            if dep in id_set:
                in_deg[n.node_id] += 1
                fwd[dep].append(n.node_id)

    depths: Dict[str, int] = {n.node_id: 0 for n in exec_nodes}
    queue  = deque(nid for nid, d in in_deg.items() if d == 0)
    while queue:
        nid = queue.popleft()
        for child in fwd.get(nid, []):
            depths[child] = max(depths[child], depths[nid] + 1)
            in_deg[child] -= 1
            if in_deg[child] == 0:
                queue.append(child)

    total_edges = sum(len(n.depends_on) for n in exec_nodes)
    n_nodes     = len(exec_nodes)
    max_e       = n_nodes * (n_nodes - 1) / 2

    return {
        "N":                       tree.N,
        "K":                       tree.K,
        "M":                       tree.M,
        "pool_size":               tree.pool_size,
        "total_nodes":             tree.total_nodes,
        "agents_per_task":         round(agents_per_task(tree.N, tree.K), 2),
        "allocated_agent_fraction": round(tree.allocated_agent_fraction(), 3),
        "avg_deps_per_node":       round(total_edges / n_nodes, 2) if n_nodes else 0.0,
        "max_dag_depth":           max(depths.values(), default=0),
        "edge_density":            round(total_edges / max_e, 4) if max_e > 0 else 0.0,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_accuracy(tree: TaskTree) -> Optional[float]:
    """
    Accuracy evaluated at seed nodes (ground truth anchors).

    IMPORTANT (fix #10): seeds are not executed, so seed.agent_answer
    must be set by the caller before this function is called.

    Caller responsibility:
        The root node (or a post-processing step) must map its final answer
        back to each relevant seed node:

            for seed in tree.seed_nodes:
                seed.agent_answer = extract_answer_for_seed(
                    root_output, seed
                )
            score = evaluate_accuracy(tree)

    Returns None if no seeds have agent_answer set.
    """
    scored = [
        n for n in tree.seed_nodes
        if n.ground_truth is not None and n.agent_answer is not None
    ]
    if not scored:
        return None
    correct = sum(
        1 for n in scored
        if n.agent_answer.strip().lower() == n.ground_truth.strip().lower()
    )
    return correct / len(scored)


# ── Serialisation ─────────────────────────────────────────────────────────────

def save_tree(tree: TaskTree, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(tree.to_dict(), indent=2))


def load_tree_dict(path: str) -> dict:
    return json.loads(Path(path).read_text())


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--N",         type=int, default=64)
    ap.add_argument("--benchmark", type=str, default="gaia")
    ap.add_argument("--domain",    type=str, default="reasoning")
    ap.add_argument("--use-llm",   action="store_true")
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--pool-size", type=int, default=150)
    args = ap.parse_args()

    print("\nScaling table:")
    print(f"{'N':>6}  {'K':>4}  {'M':>5}  {'K*M':>6}  {'agents/task':>12}")
    print("-" * 40)
    for N in [8, 16, 32, 64, 128, 256, 512]:
        K   = num_seed_tasks(args.pool_size)
        M   = num_expanded_per_seed(N, K)
        apt = agents_per_task(N, K)
        print(f"{N:>6}  {K:>4}  {M:>5}  {K*M:>6}  {apt:>12.2f}")

    print()

    fake_pool = [
        {
            "node_id":     f"task_{i:04d}",
            "description": (
                "A factory runs three shifts. Morning: 240 widgets in 8 hours. "
                "Afternoon: 180 widgets in 6 hours. Night: 200 widgets in 10 hours. "
                f"Variant {i}."
            ),
            "ground_truth":     "morning",
            "benchmark_source": args.benchmark,
        }
        for i in range(args.pool_size)
    ]

    expander = TaskExpander(
        benchmark=args.benchmark, domain=args.domain, seed=args.seed
    )
    tree = expander.build(N=args.N, benchmark_pool=fake_pool, use_llm=args.use_llm)
    print(tree.summary())

    # Structural checks
    m = validate_tree(tree)

    # Correct root dependencies (fix #3)
    exp_ids  = {n.node_id for n in tree.expanded_nodes}
    seed_ids = {n.node_id for n in tree.seed_nodes}
    root_deps = set(tree.root_node.depends_on)
    assert root_deps == exp_ids, (
        f"Root must depend on all expanded nodes.\n"
        f"  Missing: {exp_ids - root_deps}\n"
        f"  Extra:   {root_deps - exp_ids}"
    )
    assert not (root_deps & seed_ids), "Root must NOT depend on seed nodes"

    # Seeds not in execution pool (fix #1)
    pool_ids = {n.node_id for n in tree.execution_pool}
    assert not (seed_ids & pool_ids), "Seeds must not be in execution_pool"

    # Exact agent sum (fix #2)
    total = sum(tree.agent_allocation.values())
    assert total == tree.N, f"Agent sum {total} != N={tree.N}"

    # No self-loops
    for n in tree.all_nodes:
        assert n.node_id not in n.depends_on, f"Self-loop at {n.node_id}"

    # No duplicate edges
    for n in tree.all_nodes:
        assert len(n.depends_on) == len(set(n.depends_on)), \
            f"Duplicate deps at {n.node_id}: {n.depends_on}"

    # Empty pool guard (fix #5)
    try:
        expander.build(N=args.N, benchmark_pool=[])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("All checks passed.")
    out = f"/tmp/tree_{args.benchmark}_N{args.N}.json"
    save_tree(tree, out)
    print(f"Saved to {out}")