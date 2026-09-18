"""
prompts/templates.py
--------------------
Loads and fills prompt templates for any topology × task_family combination.

Usage
-----
    from prompts.templates import PromptBuilder

    builder = PromptBuilder()
    system_prompt = builder.system(
        topology="star",
        task_family="planning",
        role="hub",
        # any extra template variables:
        agent_index=0, num_agents=8,
    )
    # Returns: base_peer + topology addendum + task_family addendum,
    #          with {variables} filled in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

_PROMPT_DIR = Path(__file__).parent


def _load(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def _fill(template: str, **kwargs: Any) -> str:
    """Fill {variable} placeholders; leave unfilled ones as-is."""
    for key, val in kwargs.items():
        template = template.replace(f"{{{key}}}", str(val) if val is not None else "")
    return template


class PromptBuilder:
    """Builds fully-resolved system prompts for any agent in any topology."""

    def __init__(self, prompt_dir: Optional[Path] = None) -> None:
        root = prompt_dir or _PROMPT_DIR
        self._base      = _load(root / "base_peer.txt")
        self._topologies: dict[str, str] = {}
        self._families:   dict[str, str] = {}

        for p in (root / "topology").glob("*.txt"):
            self._topologies[p.stem] = _load(p)
        for p in (root / "task_family").glob("*.txt"):
            self._families[p.stem] = _load(p)

    def system(
        self,
        topology:    str,
        task_family: str,
        **kwargs: Any,
    ) -> str:
        """
        Assemble a system prompt:
            base_peer + topology/{topology}.txt + task_family/{task_family}.txt

        All {variable} placeholders in all three sections are filled with kwargs.
        """
        topo_block   = self._topologies.get(topology, "")
        family_block = self._families.get(task_family, "")

        parts = [self._base]
        if topo_block:
            parts += ["", "--- TOPOLOGY CONTEXT ---", topo_block]
        if family_block:
            parts += ["", "--- TASK TYPE CONTEXT ---", family_block]

        combined = "\n".join(parts)
        return _fill(combined, **kwargs)

    def list_topologies(self) -> list[str]:
        return sorted(self._topologies.keys())

    def list_families(self) -> list[str]:
        return sorted(self._families.keys())


# Module-level singleton for convenience
_default_builder: Optional[PromptBuilder] = None


def get_builder() -> PromptBuilder:
    global _default_builder
    if _default_builder is None:
        _default_builder = PromptBuilder()
    return _default_builder


def build_system_prompt(topology: str, task_family: str, **kwargs: Any) -> str:
    return get_builder().system(topology, task_family, **kwargs)
