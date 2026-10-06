"""Compatibility facade for the enforcement-native must-start contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping


_MOUNT_ROOT_NAME = "glm52-worker-start-v2"
_MOUNTED_NATIVE_FILENAME = "glm52_sky_must_start_native.py"
_NATIVE_MODULE_NAME = "glm52_sky_must_start_native"


def _load_native_path(native_path: Path) -> object:
    _existing_native = sys.modules.get(_NATIVE_MODULE_NAME)
    if _existing_native is not None:
        _existing_path = getattr(_existing_native, "__file__", None)
        if (
            not isinstance(_existing_path, str)
            or Path(_existing_path).resolve() != native_path
        ):
            raise RuntimeError("must-start native policy module identity drifted")
        return _existing_native
    _native_spec = importlib.util.spec_from_file_location(
        _NATIVE_MODULE_NAME,
        native_path,
    )
    if _native_spec is None or _native_spec.loader is None:
        raise RuntimeError(f"cannot load {native_path}")
    native = importlib.util.module_from_spec(_native_spec)
    sys.modules[_NATIVE_MODULE_NAME] = native
    _native_spec.loader.exec_module(native)
    return native


_facade_path = Path(__file__).resolve()
_mounted_native_path = _facade_path.with_name(_MOUNTED_NATIVE_FILENAME)
if _facade_path.parent.name == _MOUNT_ROOT_NAME:
    if not _mounted_native_path.is_file() or _mounted_native_path.is_symlink():
        raise RuntimeError("mounted must-start native policy is unavailable")
    _native = _load_native_path(_mounted_native_path)
else:
    try:
        from glm52_enforcement import glm52_sky_must_start as _native
    except ModuleNotFoundError as error:
        if error.name != "glm52_enforcement":
            raise
        _native = _load_native_path(
            _facade_path.parents[2] / "glm52_enforcement/glm52_sky_must_start.py"
        )

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


def build_must_start_controller_observation(**kwargs: Any) -> dict[str, object]:
    """Compatibility route for the enforcement-native observation builder."""
    return _native.build_must_start_controller_observation(**kwargs)


def validate_must_start_controller_observation(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Compatibility route for the enforcement-native observation validator."""
    return _native.validate_must_start_controller_observation(value)
