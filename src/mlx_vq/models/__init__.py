"""Model adapters that bind VQ layers to supported architectures.

The adapter modules import MLX. Keep this package initializer lightweight so
metadata-only modules such as ``mlx_vq.models.profiles`` can be imported in
headless dry-run and test environments without requiring a Metal device.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "GLM4MoeRoutingConfig": "mlx_vq.models.glm4_moe_adapter",
    "QuantizedVQGLM4MoE": "mlx_vq.models.glm4_moe_adapter",
    "QuantizedVQSwitchGLU": "mlx_vq.models.glm4_moe_adapter",
    "group_expert_select": "mlx_vq.models.glm4_moe_adapter",
    "GLM45AirNonExpertBindReport": "mlx_vq.models.glm45_air_vq_adapter",
    "GLM45AirVQModel": "mlx_vq.models.glm45_air_vq_adapter",
    "bind_glm45_air_non_expert_precision_policy": "mlx_vq.models.glm45_air_vq_adapter",
    "bind_glm45_air_non_expert_weights": "mlx_vq.models.glm45_air_vq_adapter",
    "bind_glm45_air_vq_experts": "mlx_vq.models.glm45_air_vq_adapter",
    "glm45_air_args_from_config": "mlx_vq.models.glm45_air_vq_adapter",
    "has_dense_glm45_air_routed_expert_parameters": "mlx_vq.models.glm45_air_vq_adapter",
    "has_unbound_glm45_air_vq_experts": "mlx_vq.models.glm45_air_vq_adapter",
    "GLM52Precision": "mlx_vq.models.glm52_policy",
    "classify_glm52_parameter": "mlx_vq.models.glm52_policy",
    "installed_glm52_support": "mlx_vq.models.glm52_policy",
    "is_vq_routed_expert": "mlx_vq.models.glm52_policy",
    "validate_glm52_config": "mlx_vq.models.glm52_policy",
    "GLM52NonVQBindReport": "mlx_vq.models.glm52_vq_adapter",
    "GLM52VQModel": "mlx_vq.models.glm52_vq_adapter",
    "GLM52VQModelArgs": "mlx_vq.models.glm52_vq_adapter",
    "bind_glm52_non_vq_weights": "mlx_vq.models.glm52_vq_adapter",
    "bind_glm52_vq_experts": "mlx_vq.models.glm52_vq_adapter",
    "collect_glm52_non_vq_weights": "mlx_vq.models.glm52_vq_adapter",
    "glm52_vq_args_from_config": "mlx_vq.models.glm52_vq_adapter",
    "has_unbound_vq_experts": "mlx_vq.models.glm52_vq_adapter",
    "QwenMoeBindReport": "mlx_vq.models.qwen_moe_adapter",
    "QwenNonExpertBindReport": "mlx_vq.models.qwen_moe_adapter",
    "bind_qwen_non_expert_weights": "mlx_vq.models.qwen_moe_adapter",
    "bind_qwen_moe_vq_experts": "mlx_vq.models.qwen_moe_adapter",
    "has_unbound_qwen_moe_vq_experts": "mlx_vq.models.qwen_moe_adapter",
    "load_qwen_moe_switch_glu": "mlx_vq.models.qwen_moe_adapter",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
