"""
src/metrics/inequality.py

Inequality statistics over agent influence distributions.
Lives in src/metrics/ — the existing empty package.

Used by cascade_metrics.compute_agent_influence() callers
and visualization modules (lorenz_curves.py etc.).
"""

from __future__ import annotations

from typing import List


def gini(values: List[float]) -> float:
    """Gini coefficient of a distribution. Returns 0.0 for empty or zero-sum."""
    if not values or sum(values) == 0:
        return 0.0
    n = len(values)
    sorted_vals = sorted(values)
    cumsum = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(sorted_vals))
    g = cumsum / (n * sum(sorted_vals))
    return max(0.0, min(1.0, g))


def top_k_share(values: List[float], k: int) -> float:
    """Fraction of total held by the top-k agents."""
    k = min(k, len(values))
    if not values or sum(values) == 0:
        return 0.0
    return sum(sorted(values, reverse=True)[:k]) / sum(values)


def effective_n(values: List[float]) -> float:
    """Effective number of agents (inverse Herfindahl index)."""
    total = sum(values)
    if total == 0:
        return 0.0
    shares = [v / total for v in values if v > 0]
    hhi = sum(s ** 2 for s in shares)
    return 1.0 / hhi if hhi > 0 else 0.0