"""Compatibility alias for the enforcement-native H100 contract.

Qualification shell paths load this module directly with the system Python,
outside an installed package.  Fall back to the sibling enforcement source in
that mode while retaining the ordinary package-import route.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_MOUNT_ROOT_NAME = "glm52-worker-start-v2"
_MOUNTED_NATIVE_FILENAME = "glm52_h100_qualification_native.py"
_NATIVE_MODULE_NAME = "glm52_h100_qualification_native"


def _load_native_path(native_path: Path) -> object:
    existing = sys.modules.get(_NATIVE_MODULE_NAME)
    if existing is not None:
        existing_path = getattr(existing, "__file__", None)
        if (
            not isinstance(existing_path, str)
            or Path(existing_path).resolve() != native_path
        ):
            raise RuntimeError("H100 native policy module identity drifted")
        return existing
    specification = importlib.util.spec_from_file_location(
        _NATIVE_MODULE_NAME, native_path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {native_path}")
    native = importlib.util.module_from_spec(specification)
    sys.modules[_NATIVE_MODULE_NAME] = native
    specification.loader.exec_module(native)
    return native


_facade_path = Path(__file__).resolve()
_mounted_native_path = _facade_path.with_name(_MOUNTED_NATIVE_FILENAME)
if _facade_path.parent.name == _MOUNT_ROOT_NAME:
    if not _mounted_native_path.is_file() or _mounted_native_path.is_symlink():
        raise RuntimeError("mounted H100 native policy is unavailable")
    _native = _load_native_path(_mounted_native_path)
else:
    try:
        from glm52_enforcement import glm52_h100_qualification as _native
    except ModuleNotFoundError as error:
        if error.name != "glm52_enforcement":
            raise
        _native = _load_native_path(
            _facade_path.parents[2]
            / "glm52_enforcement/glm52_h100_qualification.py"
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
