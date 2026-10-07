"""Family adapter registry: registration, lookup, lazy resolution."""

import subprocess
import sys
from pathlib import Path

import pytest

from ramp.models import registry


def test_default_families_registered():
    families = registry.list_families()
    assert "glm52" in families
    assert "glm45_air" in families
    assert "qwen_moe" in families


def test_get_family_returns_binding():
    binding = registry.get_family("glm52")
    assert binding.architecture == "glm_moe_dsa"
    assert binding.adapter_module == "ramp.models.glm52_vq_adapter"
    assert binding.bind_vq_experts_symbol == "bind_glm52_vq_experts"


def test_unknown_family_raises_with_choices():
    with pytest.raises(registry.FamilyRegistryError) as excinfo:
        registry.get_family("nope")
    assert "glm52" in str(excinfo.value)


def test_register_and_unregister_family():
    binding = registry.FamilyBinding(
        family="test_family",
        architecture="test_arch",
        adapter_module="ramp.models.glm52_vq_adapter",
        model_args_symbol="GLM52VQModelArgs",
        bind_vq_experts_symbol="bind_glm52_vq_experts",
        bind_non_vq_weights_symbol="bind_glm52_non_vq_weights",
        has_unbound_symbol="has_unbound_vq_experts",
        profile_name=None,
    )
    registry.register_family(binding)
    try:
        assert "test_family" in registry.list_families()
    finally:
        registry.unregister_family("test_family")
    assert "test_family" not in registry.list_families()


def test_duplicate_registration_rejected():
    with pytest.raises(registry.FamilyRegistryError):
        registry.register_family(registry.get_family("glm52"))


def test_qwen_moe_binding_uses_repo_model_type():
    binding = registry.get_family("qwen_moe")
    assert binding.architecture == "qwen3_5_moe"
    # ``qwen_moe_adapter`` is a pure binder: it defines no args/config class,
    # so there is no symbol to resolve for ``model_args``.
    assert binding.model_args_symbol is None


def test_default_binding_architecture_matches_profile():
    from ramp.models.profiles import get_profile

    checked = 0
    for family in registry.list_families():
        binding = registry.get_family(family)
        if binding.profile_name is None:
            continue
        profile = get_profile(binding.profile_name)
        assert binding.architecture == profile.architecture, (
            f"{family}: binding architecture {binding.architecture!r} != profile "
            f"{binding.profile_name!r} architecture {profile.architecture!r}"
        )
        checked += 1
    assert checked >= 3


def test_resolve_family_allows_absent_model_args():
    # ``ramp.models.profiles`` is a headless-safe stand-in adapter module: it
    # imports no MLX and exposes real module-level callables.
    binding = registry.FamilyBinding(
        family="test_no_args",
        architecture="test_arch",
        adapter_module="ramp.models.profiles",
        model_args_symbol=None,
        bind_vq_experts_symbol="get_profile",
        bind_non_vq_weights_symbol=None,
        has_unbound_symbol="list_profiles",
        profile_name=None,
    )
    registry.register_family(binding)
    try:
        resolved = registry.resolve_family("test_no_args")
    finally:
        registry.unregister_family("test_no_args")
    assert resolved.model_args is None
    assert resolved.bind_non_vq_weights is None
    assert resolved.bind_vq_experts.__name__ == "get_profile"
    assert resolved.has_unbound.__name__ == "list_profiles"


def test_empty_symbol_names_rejected():
    with pytest.raises(registry.FamilyRegistryError):
        registry.FamilyBinding(
            family="test_bad",
            architecture="test_arch",
            adapter_module="ramp.models.profiles",
            model_args_symbol="",
            bind_vq_experts_symbol="get_profile",
            bind_non_vq_weights_symbol=None,
            has_unbound_symbol="list_profiles",
        )


def test_registry_is_visible_on_ramp_models():
    import ramp.models

    assert ramp.models.registry is registry
    assert "registry" in ramp.models.__all__


_HEADLESS_PROBE = """
import sys

import ramp.models.registry
import ramp.models.registry

mlx_loaded = sorted(name for name in sys.modules if name == "mlx" or name.startswith("mlx."))
assert "mlx.core" not in sys.modules, mlx_loaded
assert sys.modules["ramp.models.registry"] is sys.modules["ramp.models.registry"]

adapters = sorted(name for name in sys.modules if name.endswith("_adapter"))
assert not adapters, adapters
print("ok")
"""


def test_registry_import_does_not_load_mlx():
    """Importing the registry must stay headless: no ``mlx.core``, no adapters."""

    result = subprocess.run(
        [sys.executable, "-c", _HEADLESS_PROBE],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ok"


