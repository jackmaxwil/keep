"""Compatibility alias for the enforcement-native spend snapshot contract.

The worker-start readiness source can be imported directly from an isolated
system Python, where the repository ``src`` directory is not on ``sys.path``.
Restore that package root before resolving the enforcement-native contract so
its relative dependency on the campaign authority remains intact.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping


try:
    from glm52_enforcement import glm52_gpu_spend_snapshot as _native
except ModuleNotFoundError as error:
    if error.name != "glm52_enforcement":
        raise
    _source_root = str(Path(__file__).resolve().parents[2])
    if _source_root not in sys.path:
        sys.path.insert(0, _source_root)
    from glm52_enforcement import glm52_gpu_spend_snapshot as _native

QUALIFICATION_MAX_SECONDS = _native.QUALIFICATION_MAX_SECONDS
GpuSpendSnapshotError = _native.GpuSpendSnapshotError


def build_gpu_spend_snapshot(**kwargs: Any) -> dict[str, object]:
    """Build a GPU-spend snapshot through the enforcement-native contract."""
    return _native.build_gpu_spend_snapshot(**kwargs)


def validate_gpu_spend_snapshot(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate a GPU-spend snapshot through the enforcement-native contract."""
    return _native.validate_gpu_spend_snapshot(value)
