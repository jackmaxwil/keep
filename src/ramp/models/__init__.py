"""Model adapters that bind VQ layers to supported architectures.

The adapter modules import MLX. Keep this package initializer lightweight so
metadata-only modules such as ``ramp.models.profiles`` can be imported in
headless dry-run and test environments without requiring a Metal device.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "GLM4MoeRoutingConfig": "ramp.models.glm4_moe_adapter",
    "QuantizedVQGLM4MoE": "ramp.models.glm4_moe_adapter",
    "QuantizedVQSwitchGLU": "ramp.models.glm4_moe_adapter",
    "group_expert_select": "ramp.models.glm4_moe_adapter",
    "GLM45AirNonExpertBindReport": "ramp.models.glm45_air_vq_adapter",
    "GLM45AirVQModel": "ramp.models.glm45_air_vq_adapter",
    "bind_glm45_air_non_expert_precision_policy": "ramp.models.glm45_air_vq_adapter",
    "bind_glm45_air_non_expert_weights": "ramp.models.glm45_air_vq_adapter",
    "bind_glm45_air_vq_experts": "ramp.models.glm45_air_vq_adapter",
    "glm45_air_args_from_config": "ramp.models.glm45_air_vq_adapter",
    "has_dense_glm45_air_routed_expert_parameters": "ramp.models.glm45_air_vq_adapter",
    "has_unbound_glm45_air_vq_experts": "ramp.models.glm45_air_vq_adapter",
    "GLM52Precision": "ramp.models.glm52_policy",
    "classify_glm52_parameter": "ramp.models.glm52_policy",
    "installed_glm52_support": "ramp.models.glm52_policy",
    "is_vq_routed_expert": "ramp.models.glm52_policy",
    "validate_glm52_config": "ramp.models.glm52_policy",
    "GLM52NonVQBindReport": "ramp.models.glm52_vq_adapter",
    "GLM52VQModel": "ramp.models.glm52_vq_adapter",
    "GLM52VQModelArgs": "ramp.models.glm52_vq_adapter",
    "bind_glm52_non_vq_weights": "ramp.models.glm52_vq_adapter",
    "bind_glm52_vq_experts": "ramp.models.glm52_vq_adapter",
    "collect_glm52_non_vq_weights": "ramp.models.glm52_vq_adapter",
    "glm52_vq_args_from_config": "ramp.models.glm52_vq_adapter",
    "has_unbound_vq_experts": "ramp.models.glm52_vq_adapter",
    "QwenMoeBindReport": "ramp.models.qwen_moe_adapter",
    "QwenNonExpertBindReport": "ramp.models.qwen_moe_adapter",
    "bind_qwen_non_expert_weights": "ramp.models.qwen_moe_adapter",
    "bind_qwen_moe_vq_experts": "ramp.models.qwen_moe_adapter",
    "has_unbound_qwen_moe_vq_experts": "ramp.models.qwen_moe_adapter",
    "load_qwen_moe_switch_glu": "ramp.models.qwen_moe_adapter",
}

__all__ = sorted({*_EXPORT_MODULES, "registry"})


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


# The family registry is metadata only (no MLX), so it loads eagerly and is
# always reachable as ``ramp.models.registry``.
from ramp.models import registry  # noqa: E402,F401
