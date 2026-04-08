"""
src/prompts/action_contract.py

DEPRECATED shim — kept only so existing imports don't crash immediately.
The output contract has moved to src/loggers/trace_schema.py.

Migrate any import of OUTPUT_CONTRACT or ACTION_SCHEMA to:
    from loggers.trace_schema import AGENT_OUTPUT_FORMAT
"""

from loggers.trace_schema import AGENT_OUTPUT_FORMAT  # noqa: F401

# Backward-compat aliases — remove once all call sites are migrated.
OUTPUT_CONTRACT      = AGENT_OUTPUT_FORMAT
ACTION_SCHEMA        = AGENT_OUTPUT_FORMAT
DECISION_RULES_SHORT = AGENT_OUTPUT_FORMAT