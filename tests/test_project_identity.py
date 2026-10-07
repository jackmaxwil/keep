import importlib
import subprocess
import sys


def test_project_and_engine_identity():
    keep = importlib.import_module("keep")
    ramp = importlib.import_module("ramp")

    assert keep.PROJECT_NAME == "KEEP"
    assert keep.METHOD_NAME == "KL-distilled Expert Encoding and Precision"
    assert callable(keep.verify_environment)
    assert ramp.ENGINE_NAME == "RAMP"
    assert ramp.ENGINE_FULL_NAME == "Routed Accelerated MoE Pipeline"


def test_mutable_registries_are_single_instances():
    """A converter registered through one import path is visible through any other."""

    import ramp.models

    profiles = importlib.import_module("ramp.models.profiles")
    assert ramp.models.profiles is profiles
    profiles.register_converter("_identity_probe_converter")
    try:
        assert "_identity_probe_converter" in importlib.import_module("ramp.models.profiles").converter_kinds()
    finally:
        profiles.unregister_converter("_identity_probe_converter")


_HEADLESS_PROBE = """
import sys

import keep.quality
import ramp.models

# Package initializers stay lightweight: no adapter, no MLX, no prompt data.
eager = sorted(n for n in sys.modules if n.startswith("ramp.models.") and not n.endswith(("registry", "profiles")))
assert not eager, eager
assert "mlx.core" not in sys.modules
assert "keep.quality.prompts" not in sys.modules
print("ok")
"""


def test_package_initializers_import_nothing_heavy():
    result = subprocess.run([sys.executable, "-c", _HEADLESS_PROBE], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ok"


def test_qwen_moe_adapter_exports_from_model_package():
    models = importlib.import_module("ramp.models")
    qwen_adapter = importlib.import_module("ramp.models.qwen_moe_adapter")

    assert models.bind_qwen_non_expert_weights is qwen_adapter.bind_qwen_non_expert_weights
    assert models.load_qwen_moe_switch_glu is qwen_adapter.load_qwen_moe_switch_glu
