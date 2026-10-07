"""The native-extension loader must resolve once per process, not per dispatch.

Regression guard for the DSV4 M=1 expert-projection deficit: ``load_native`` was
re-globbing the build tree and re-executing the 3.5 MB extension on every kernel
call, which dominated measured VQ E8P projection time.
"""

from __future__ import annotations

import sys

from ramp.kernels import nax


def test_load_native_returns_one_module_per_process() -> None:
    if not nax.is_available():
        return
    first = nax.load_native()
    assert nax.load_native() is first
    assert sys.modules.get("_vqnax") is first


def test_import_module_fast_path_hits_after_first_load() -> None:
    if not nax.is_available():
        return
    nax.load_native()
    import importlib

    assert importlib.import_module("_vqnax") is nax.load_native()


def test_native_candidates_have_no_duplicates() -> None:
    candidates = nax._native_candidates()
    assert len(candidates) == len({path.resolve() for path in candidates})


def test_repo_root_and_kernel_dir_are_cached() -> None:
    assert nax._repo_root() is nax._repo_root()
    assert nax._kernel_dir() is nax._kernel_dir()
    assert nax._kernel_dir().name == "kernels"


def test_is_available_is_memoized() -> None:
    nax.is_available()
    assert nax.is_available.cache_info().hits >= 1
