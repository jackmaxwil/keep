"""Registry mapping model families to their adapter surfaces.

Adding a family is a registration, not an edit to shared modules. Adapter
modules import MLX, so resolution is lazy: a binding stores dotted module
paths and symbol names; ``resolve_family`` imports on first use.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable


class FamilyRegistryError(ValueError):
    """Raised for unknown families or invalid registrations."""


@dataclass(frozen=True)
class FamilyBinding:
    family: str
    architecture: str
    adapter_module: str
    # ``None`` when the adapter module has no args/config class of its own
    # (a pure binder, e.g. ``qwen_moe_adapter``).
    model_args_symbol: str | None
    bind_vq_experts_symbol: str
    bind_non_vq_weights_symbol: str | None
    has_unbound_symbol: str
    profile_name: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "family",
            "architecture",
            "adapter_module",
            "bind_vq_experts_symbol",
            "has_unbound_symbol",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise FamilyRegistryError(f"{field_name} must be a non-empty string")

        for field_name in (
            "model_args_symbol",
            "bind_non_vq_weights_symbol",
            "profile_name",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            if not isinstance(value, str) or not value:
                raise FamilyRegistryError(
                    f"{field_name} must be a non-empty string or None"
                )


@dataclass(frozen=True)
class ResolvedFamily:
    binding: FamilyBinding
    model_args: type | None
    bind_vq_experts: Callable[..., Any]
    bind_non_vq_weights: Callable[..., Any] | None
    has_unbound: Callable[..., Any]


_REGISTRY: dict[str, FamilyBinding] = {}


def register_family(binding: FamilyBinding) -> None:
    if binding.family in _REGISTRY:
        raise FamilyRegistryError(f"family {binding.family!r} is already registered")
    _REGISTRY[binding.family] = binding


def unregister_family(family: str) -> None:
    _REGISTRY.pop(family, None)


def list_families() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get_family(family: str) -> FamilyBinding:
    try:
        return _REGISTRY[family]
    except KeyError as error:
        choices = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise FamilyRegistryError(
            f"unknown family {family!r}; choices: {choices}"
        ) from error


def resolve_family(family: str) -> ResolvedFamily:
    binding = get_family(family)
    module = import_module(binding.adapter_module)
    bind_non_vq = (
        getattr(module, binding.bind_non_vq_weights_symbol)
        if binding.bind_non_vq_weights_symbol is not None
        else None
    )
    model_args = (
        getattr(module, binding.model_args_symbol)
        if binding.model_args_symbol is not None
        else None
    )
    return ResolvedFamily(
        binding=binding,
        model_args=model_args,
        bind_vq_experts=getattr(module, binding.bind_vq_experts_symbol),
        bind_non_vq_weights=bind_non_vq,
        has_unbound=getattr(module, binding.has_unbound_symbol),
    )


register_family(
    FamilyBinding(
        family="glm52",
        architecture="glm_moe_dsa",
        adapter_module="mlx_vq.models.glm52_vq_adapter",
        model_args_symbol="GLM52VQModelArgs",
        bind_vq_experts_symbol="bind_glm52_vq_experts",
        bind_non_vq_weights_symbol="bind_glm52_non_vq_weights",
        has_unbound_symbol="has_unbound_vq_experts",
        profile_name="glm52-reap-504b-v2",
    )
)
register_family(
    FamilyBinding(
        family="glm45_air",
        architecture="glm4_moe",
        adapter_module="mlx_vq.models.glm45_air_vq_adapter",
        model_args_symbol="ModelArgs",
        bind_vq_experts_symbol="bind_glm45_air_vq_experts",
        bind_non_vq_weights_symbol="bind_glm45_air_non_expert_weights",
        has_unbound_symbol="has_unbound_glm45_air_vq_experts",
        profile_name="glm45-air",
    )
)
register_family(
    FamilyBinding(
        family="qwen_moe",
        # Matches ``QWEN36_35B_A3B_MODEL_TYPE`` and ``models/qwen36-35b-a3b.yaml``.
        architecture="qwen3_5_moe",
        adapter_module="mlx_vq.models.qwen_moe_adapter",
        # The adapter is a pure binder: it defines no args/config class.
        model_args_symbol=None,
        bind_vq_experts_symbol="bind_qwen_moe_vq_experts",
        bind_non_vq_weights_symbol="bind_qwen_non_expert_weights",
        has_unbound_symbol="has_unbound_qwen_moe_vq_experts",
        profile_name="qwen36-35b-a3b",
    )
)
register_family(
    FamilyBinding(
        family="deepseek_v4_flash",
        architecture="deepseek_v4",
        # Wave 2 increment 1: the real adapter replaces the NotImplementedError
        # stubs that stood in while only the precision policy existed. The
        # policy module keeps its ``classify_deepseek_v4_parameter`` exports --
        # it is imported headlessly by conversion tooling and must not start
        # pulling MLX in through the adapter.
        adapter_module="ramp.models.deepseek_v4_flash_adapter",
        model_args_symbol="DeepseekV4FlashVQModelArgs",
        bind_vq_experts_symbol="bind_deepseek_v4_flash_vq_experts",
        bind_non_vq_weights_symbol="bind_deepseek_v4_flash_non_vq_weights",
        has_unbound_symbol="has_unbound_deepseek_v4_flash_vq_experts",
        profile_name="deepseek-v4-flash-0731",
    )
)
register_family(
    FamilyBinding(
        family="glm5_next",
        architecture="glm5_next",
        adapter_module="ramp.models.glm5_next_adapter",
        model_args_symbol="Glm5NextVQModelArgs",
        bind_vq_experts_symbol="bind_glm5_next_vq_experts",
        bind_non_vq_weights_symbol="bind_glm5_next_non_vq_weights",
        has_unbound_symbol="has_unbound_glm5_next_vq_experts",
        profile_name="glm-5.3-flash",
    )
)
