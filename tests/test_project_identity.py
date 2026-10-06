import importlib
import subprocess
import sys


def test_keep_method_imports_and_legacy_alias_still_work():
    keep = importlib.import_module("keep")
    legacy = importlib.import_module("mlx_vq")

    assert keep.PROJECT_NAME == "KEEP"
    assert keep.METHOD_NAME == "KL-distilled Expert Encoding and Precision"
    assert keep.verify_environment is legacy.verify_environment

    keep_rtn = importlib.import_module("keep.quant.rtn")
    legacy_rtn = importlib.import_module("mlx_vq.quant.rtn")
    assert keep_rtn is legacy_rtn

    keep_e8 = importlib.import_module("keep.vq.e8")
    legacy_e8 = importlib.import_module("mlx_vq.codebook.e8")
    assert keep_e8 is legacy_e8


def test_keep_public_subpackages_alias_legacy_method_modules():
    aliases = {
        "keep.build.highlevel": "mlx_vq.build.highlevel",
        "keep.convert.inspect_hf": "mlx_vq.convert.inspect_hf",
        "keep.io.schema": "mlx_vq.io.schema",
        "keep.quality.prompts": "mlx_vq.quality.prompts",
        "keep.validate.glm45_air_vq": "mlx_vq.validate.glm45_air_vq",
    }

    for public_name, legacy_name in aliases.items():
        assert importlib.import_module(public_name) is importlib.import_module(legacy_name)


def test_ramp_engine_imports_alias_runtime_modules():
    ramp = importlib.import_module("ramp")

    assert ramp.ENGINE_NAME == "RAMP"
    assert ramp.ENGINE_FULL_NAME == "Routed Accelerated MoE Pipeline"

    ramp_vq_qmv = importlib.import_module("ramp.kernels.vq_qmv")
    legacy_vq_qmv = importlib.import_module("mlx_vq.kernels.vq_qmv")
    assert ramp_vq_qmv is legacy_vq_qmv

    ramp_switch = importlib.import_module("ramp.ops.vq_switch")
    legacy_switch = importlib.import_module("mlx_vq.ops.vq_switch")
    assert ramp_switch is legacy_switch


def test_ramp_public_subpackages_alias_legacy_runtime_modules():
    aliases = {
        "ramp.benchmark.metrics": "mlx_vq.benchmark.metrics",
        "ramp.models.glm45_air_vq_adapter": "mlx_vq.models.glm45_air_vq_adapter",
        "ramp.models.profiles": "mlx_vq.models.profiles",
        "ramp.models.qwen_moe_adapter": "mlx_vq.models.qwen_moe_adapter",
        "ramp.nn.switch_linear": "mlx_vq.nn.switch_linear",
    }

    for public_name, legacy_name in aliases.items():
        assert importlib.import_module(public_name) is importlib.import_module(legacy_name)


def test_mutable_registry_modules_are_single_instances():
    """Registries with process-global mutable state must not be duplicated.

    ``profiles`` and ``highlevel`` each hold a module-level registry dict.
    Before these were declared as alias children, importing the ``ramp``/``keep``
    name loaded a *second* copy from the shared ``__path__``, so a converter
    kind registered through one name was invisible through the other.
    """

    import keep.build
    import ramp.models

    profiles = importlib.import_module("ramp.models.profiles")
    legacy_profiles = importlib.import_module("mlx_vq.models.profiles")
    assert profiles is legacy_profiles
    assert ramp.models.profiles is legacy_profiles

    highlevel = importlib.import_module("keep.build.highlevel")
    legacy_highlevel = importlib.import_module("mlx_vq.build.highlevel")
    assert highlevel is legacy_highlevel
    assert keep.build.highlevel is legacy_highlevel

    # A registration through the public name is visible through the legacy one.
    profiles.register_converter("_identity_probe_converter")
    try:
        assert "_identity_probe_converter" in legacy_profiles.converter_kinds()
    finally:
        legacy_profiles.unregister_converter("_identity_probe_converter")


_LAZY_CHILD_PROBE = """
import sys

import ramp.models

# ``mlx_vq.models`` resolves its exports lazily, so aliasing it must not pull in
# any adapter module (and with them, MLX). ``ramp.models.registry`` is a real
# local module, not an alias, and is imported on purpose.
eager = sorted(
    name
    for name in sys.modules
    if name.startswith(("mlx_vq.models.", "ramp.models."))
    and not name.endswith("registry")
)
assert not eager, eager
assert "mlx.core" not in sys.modules

import ramp.models.glm52_vq_adapter
import mlx_vq.models.glm52_vq_adapter

assert (
    sys.modules["ramp.models.glm52_vq_adapter"]
    is sys.modules["mlx_vq.models.glm52_vq_adapter"]
)
assert ramp.models.glm52_vq_adapter is mlx_vq.models.glm52_vq_adapter

# The legacy module keeps its own import metadata despite being aliased.
assert mlx_vq.models.glm52_vq_adapter.__name__ == "mlx_vq.models.glm52_vq_adapter"
assert mlx_vq.models.glm52_vq_adapter.__spec__.name == "mlx_vq.models.glm52_vq_adapter"

# Attribute access alone still resolves an aliased child, as it did when the
# aliases were installed eagerly.
assert ramp.models.qwen_moe_adapter is mlx_vq.models.qwen_moe_adapter

from keep.quant import rtn as keep_rtn
import mlx_vq.quant.rtn as legacy_rtn

assert keep_rtn is legacy_rtn
print("ok")
"""


def test_alias_child_modules_are_imported_lazily():
    """Aliased children resolve on demand, not when the alias package loads."""

    result = subprocess.run(
        [sys.executable, "-c", _LAZY_CHILD_PROBE],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ok"


def test_qwen_moe_adapter_exports_from_legacy_model_package():
    models = importlib.import_module("mlx_vq.models")
    qwen_adapter = importlib.import_module("mlx_vq.models.qwen_moe_adapter")

    assert models.bind_qwen_non_expert_weights is qwen_adapter.bind_qwen_non_expert_weights
    assert models.bind_qwen_moe_vq_experts is qwen_adapter.bind_qwen_moe_vq_experts
    assert (
        models.has_unbound_qwen_moe_vq_experts
        is qwen_adapter.has_unbound_qwen_moe_vq_experts
    )
    assert models.load_qwen_moe_switch_glu is qwen_adapter.load_qwen_moe_switch_glu
