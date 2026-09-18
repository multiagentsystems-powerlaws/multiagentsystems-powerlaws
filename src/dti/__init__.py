"""
src/dti/

Deficit-Triggered Integration (DTI) — paper Section 6 + Appendix C.

Public API:
    from dti import DTIMonitor, DTIParams
    from dti import estimate_dti_params_from_sweep, load_dti_params
"""

from dti.dti import (
    DTIMonitor,
    DTIParams,
    estimate_dti_params_from_sweep,
    load_dti_params,
)

__all__ = [
    "DTIMonitor",
    "DTIParams",
    "estimate_dti_params_from_sweep",
    "load_dti_params",
]