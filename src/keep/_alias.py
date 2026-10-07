"""Small helpers for KEEP/RAMP compatibility package aliases."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterable, Sequence
from importlib.machinery import ModuleSpec, all_suffixes
from importlib.util import spec_from_loader
from pathlib import Path
from types import ModuleType
from typing import Any


class _AliasLoader:
    """Loader that hands back an already-imported legacy module."""

    def __init__(self, legacy_name: str) -> None:
        self._legacy_name = legacy_name
        self._legacy_spec: ModuleSpec | None = None

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        module = importlib.import_module(self._legacy_name)
        self._legacy_spec = getattr(module, "__spec__", None)
        return module

    def exec_module(self, module: ModuleType) -> None:
        # The legacy module body already ran under its own name; there is
        # nothing to execute. ``module_from_spec`` unconditionally rewrote
        # ``__spec__`` with the public alias spec, so put the legacy spec back:
        # the module keeps reporting its own identity and stays reloadable.
        if self._legacy_spec is not None:
            module.__spec__ = self._legacy_spec


class _AliasFinder:
    """Resolve registered public child names to their legacy modules.

    Aliasing is lazy in the six.moves sense: nothing is imported until someone
    actually imports the public name, and what lands in ``sys.modules`` is the
    legacy module object itself, so ``keep.quant.rtn is mlx_vq.quant.rtn``.
    """

    def __init__(self) -> None:
        self._aliases: dict[str, str] = {}

    def register(self, public_name: str, legacy_name: str) -> None:
        current = self._aliases.get(public_name)
        if current is not None and current != legacy_name:
            raise RuntimeError(
                f"{public_name} is already aliased to {current}; refusing to "
                f"re-alias it to {legacy_name}"
            )
        self._aliases[public_name] = legacy_name

    def legacy_name_for(self, public_name: str) -> str | None:
        return self._aliases.get(public_name)

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        legacy_name = self._aliases.get(fullname)
        if legacy_name is None:
            return None
        return spec_from_loader(fullname, _AliasLoader(legacy_name))


_FINDER = _AliasFinder()


def _install_finder() -> None:
    if any(finder is _FINDER for finder in sys.meta_path):
        return
    # Ahead of ``PathFinder``: an alias package's ``__path__`` also contains the
    # legacy directory, so the default machinery would happily load a second,
    # distinct copy of the legacy module under the public name and break the
    # ``is`` identity these aliases exist to provide.
    sys.meta_path.insert(0, _FINDER)


def _physical_module_path(own_path: Sequence[str], child: str) -> str | None:
    """Return the path of a module named ``child`` living in ``own_path``."""

    top = child.partition(".")[0]
    for entry in own_path:
        directory = Path(entry)
        package_init = directory / top / "__init__.py"
        if package_init.is_file():
            return str(package_init)
        for suffix in all_suffixes():
            candidate = directory / f"{top}{suffix}"
            if candidate.is_file():
                return str(candidate)
    return None


def install_alias_package(
    public_name: str,
    legacy_name: str,
    namespace: dict[str, Any],
    child_modules: Iterable[str] = (),
) -> tuple[Any, Any]:
    """Expose a legacy package through a new public package name.

    The wrapper keeps imports like ``keep.quant`` lightweight: child modules
    named in ``child_modules`` are registered with a meta-path finder and only
    imported when someone imports them, at which point ``keep.quant.rtn`` and
    ``mlx_vq.quant.rtn`` resolve to the same module object.
    """

    own_path = [str(entry) for entry in namespace.get("__path__", ())]
    children = tuple(child_modules)

    for child in children:
        shadowed = _physical_module_path(own_path, child)
        if shadowed is not None:
            raise RuntimeError(
                f"{public_name}.{child} is registered as an alias of "
                f"{legacy_name}.{child}, but a physical module already exists at "
                f"{shadowed}; the alias would shadow it. Drop {child!r} from "
                "child_modules or rename the file."
            )

    legacy = importlib.import_module(legacy_name)
    namespace["LEGACY_PACKAGE"] = legacy_name
    namespace["__all__"] = list(getattr(legacy, "__all__", ()))
    if hasattr(legacy, "__path__"):
        # Search this package's own directory first, then fall back to the
        # legacy path. Python already populated ``__path__`` with the alias
        # package's own directory before running this init code; preserving it
        # (rather than replacing it outright) lets real submodules that
        # physically live here -- the cutover target for new code -- resolve
        # from this directory (e.g. ``ramp.models.registry``), while
        # legacy-only submodules keep resolving through the aliased path.
        # Names in ``child_modules`` are resolved by the alias finder before
        # any path search happens, which is why a physical file of the same
        # name is rejected above instead of being silently shadowed.
        legacy_path = [path for path in legacy.__path__ if path not in own_path]
        namespace["__path__"] = own_path + legacy_path

    # Exported names are *not* copied into this namespace: several legacy
    # packages (e.g. ``mlx_vq.models``) resolve their exports through their own
    # module ``__getattr__``, so touching them here would import every adapter
    # -- and therefore MLX -- just to alias the package. ``__getattr__`` below
    # forwards on first access instead, which keeps ``__all__`` accurate and
    # attribute identity intact.

    if children:
        _install_finder()
        for child in children:
            _FINDER.register(f"{public_name}.{child}", f"{legacy_name}.{child}")

    def __getattr__(name: str) -> Any:
        legacy_child = _FINDER.legacy_name_for(f"{public_name}.{name}")
        if legacy_child is not None:
            return importlib.import_module(f"{public_name}.{name}")
        return getattr(legacy, name)

    def __dir__() -> list[str]:
        return sorted(set(namespace) | set(dir(legacy)))

    return __getattr__, __dir__


def alias_module(public_name: str, legacy_name: str) -> ModuleType:
    """Bind ``public_name`` to the module object imported from ``legacy_name``.

    Callable from a shim module's own body (``sys.modules[__name__] =
    alias_module(__name__, ...)``), so the binding must overwrite the
    partially-initialized shim that the import machinery already registered.
    """

    module = importlib.import_module(legacy_name)
    sys.modules[public_name] = module
    return module
