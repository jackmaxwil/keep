"""Compatibility facade for the enforcement-native watchdog policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


try:
    from glm52_enforcement import glm52_campaign_watchdog as _native
except ModuleNotFoundError as error:
    if error.name != "glm52_enforcement":
        raise
    _native_path = (
        Path(__file__).resolve().parents[2]
        / "glm52_enforcement/glm52_campaign_watchdog.py"
    )
    _native_name = "_glm52_enforcement_campaign_watchdog"
    _native_spec = importlib.util.spec_from_file_location(_native_name, _native_path)
    if _native_spec is None or _native_spec.loader is None:
        raise RuntimeError(f"cannot load {_native_path}")
    _native = importlib.util.module_from_spec(_native_spec)
    sys.modules[_native_name] = _native
    _native_spec.loader.exec_module(_native)

for _name, _value in vars(_native).items():
    if _name not in {
        "__builtins__",
        "__cached__",
        "__file__",
        "__loader__",
        "__name__",
        "__package__",
        "__spec__",
    }:
        globals()[_name] = _value
