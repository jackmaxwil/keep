"""YAML-backed model profiles for KEEP recipe and CLI targeting."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


class ProfileError(ValueError):
    """Raised when a model profile is missing or invalid."""


ConverterKind = str

_CONVERTER_KINDS: set[str] = {
    "glm_stream",
    "qwen_moe_groups",
    "glm52_vq_groups",
    "deepseek_v4_vq_groups",
}


def register_converter(kind: str) -> None:
    if not isinstance(kind, str) or not kind:
        raise ProfileError("converter kind must be a non-empty string")
    _CONVERTER_KINDS.add(kind)


def unregister_converter(kind: str) -> None:
    _CONVERTER_KINDS.discard(kind)


def converter_kinds() -> frozenset[str]:
    return frozenset(_CONVERTER_KINDS)


PROFILE_ROOT = Path(__file__).resolve().parents[3] / "models"


@dataclass(frozen=True)
class ModelProfile:
    name: str
    hf_model_id: str
    revision: str | None
    architecture: str
    num_layers: int
    num_sparse_layers: int | None
    first_sparse_layer: int | None
    hidden_size: int
    moe_intermediate_size: int
    num_experts: int
    experts_per_tok: int
    vocab_size: int
    shared_experts: int | None
    converter: ConverterKind
    fused_gate_up: bool
    default_code_bits: int
    group_size_policy: dict[str, int]
    default_engine: str
    recovery_layer: int | None = None
    recovery_projections: tuple[str, ...] = ()
    hard_layers: tuple[int, ...] = ()
    imatrix_layers: tuple[int, ...] = ()
    prompt_sets: dict[str, str] | None = None
    lane_s_scenario: str | None = None
    approx_bf16_gb: float | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        for field in (
            "num_layers",
            "hidden_size",
            "moe_intermediate_size",
            "num_experts",
            "experts_per_tok",
            "vocab_size",
            "default_code_bits",
        ):
            value = getattr(self, field)
            if not isinstance(value, int) or value <= 0:
                raise ProfileError(f"{field} must be positive")

        for field in ("num_sparse_layers", "first_sparse_layer", "shared_experts"):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ProfileError(f"{field} must be non-negative when set")

        if self.converter not in _CONVERTER_KINDS:
            choices = ", ".join(sorted(_CONVERTER_KINDS))
            raise ProfileError(f"unknown converter {self.converter!r}; choices: {choices}")

        if not self.group_size_policy:
            raise ProfileError("group_size_policy must not be empty")
        for key, value in self.group_size_policy.items():
            if not isinstance(key, str) or not key:
                raise ProfileError("group_size_policy keys must be non-empty strings")
            if not isinstance(value, int) or value <= 0:
                raise ProfileError(f"group_size_policy[{key!r}] must be positive")

        if self.num_sparse_layers is not None and self.num_sparse_layers > self.num_layers:
            raise ProfileError("num_sparse_layers must be <= num_layers")
        if self.first_sparse_layer is not None and self.first_sparse_layer >= self.num_layers:
            raise ProfileError("first_sparse_layer must be less than num_layers")

        object.__setattr__(self, "recovery_projections", _str_tuple(self.recovery_projections))
        object.__setattr__(self, "hard_layers", _int_tuple(self.hard_layers))
        imatrix_layers = _int_tuple(self.imatrix_layers)
        if not imatrix_layers and self.num_sparse_layers is not None:
            first = self.first_sparse_layer or 0
            imatrix_layers = tuple(range(first, first + self.num_sparse_layers))
        object.__setattr__(self, "imatrix_layers", imatrix_layers)

        recovery_layer = self.recovery_layer
        if recovery_layer is None:
            recovery_layer = self.num_layers - 1
        if recovery_layer < 0 or recovery_layer >= self.num_layers:
            raise ProfileError("recovery_layer must be within model layers")
        object.__setattr__(self, "recovery_layer", recovery_layer)

        if self.prompt_sets is not None:
            object.__setattr__(
                self,
                "prompt_sets",
                {str(key): str(value) for key, value in self.prompt_sets.items()},
            )


def _str_tuple(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        raise ProfileError("expected a sequence of strings, not a string")
    return tuple(str(value) for value in values)


def _int_tuple(values: Any) -> tuple[int, ...]:
    if values is None:
        return ()
    return tuple(int(value) for value in values)


def _profile_paths(root: Path = PROFILE_ROOT) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.glob("*.yaml"))


def load_profile(path: str | Path) -> ModelProfile:
    profile_path = Path(path)
    try:
        raw = yaml.safe_load(profile_path.read_text())
    except OSError as error:
        raise ProfileError(f"could not read model profile {profile_path}: {error}") from error
    if not isinstance(raw, dict):
        raise ProfileError(f"{profile_path} must contain a mapping")
    try:
        return ModelProfile(**raw)
    except TypeError as error:
        raise ProfileError(f"invalid model profile {profile_path}: {error}") from error


def _registry(root: Path = PROFILE_ROOT) -> dict[str, ModelProfile]:
    profiles: dict[str, ModelProfile] = {}
    for path in _profile_paths(root):
        profile = load_profile(path)
        if profile.name in profiles:
            raise ProfileError(f"duplicate model profile {profile.name!r}")
        profiles[profile.name] = profile
    return dict(sorted(profiles.items()))


def list_profiles(root: Path = PROFILE_ROOT) -> tuple[str, ...]:
    return tuple(_registry(root).keys())


def get_profile(name: str, root: Path = PROFILE_ROOT) -> ModelProfile:
    profiles = _registry(root)
    try:
        return profiles[name]
    except KeyError as error:
        choices = ", ".join(profiles) or "(none)"
        raise ProfileError(f"unknown model profile {name!r}; choices: {choices}") from error


def profile_to_dict(profile: ModelProfile) -> dict[str, Any]:
    raw = asdict(profile)
    for key in ("recovery_projections", "hard_layers", "imatrix_layers"):
        raw[key] = list(raw[key])
    return raw


def profile_to_json(profile: ModelProfile) -> str:
    return json.dumps(profile_to_dict(profile), indent=2, sort_keys=False) + "\n"


def validate_profile_against_hf_config(
    profile: ModelProfile,
    config_json_path: str | Path,
) -> list[str]:
    try:
        config = json.loads(Path(config_json_path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ProfileError(f"could not read HF config {config_json_path}: {error}") from error

    if not isinstance(config, Mapping):
        raise ProfileError(f"HF config {config_json_path} must contain an object")
    return validate_profile_against_hf_config_data(profile, config)


def validate_profile_against_hf_config_data(
    profile: ModelProfile,
    config: Mapping[str, Any],
) -> list[str]:
    """Compare a profile with already-loaded Hugging Face config data."""

    dims_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else config

    dense_layers = None
    sparse_layers = None
    if isinstance(dims_config.get("mlp_layer_types"), list):
        layer_types = list(dims_config["mlp_layer_types"])
        dense_layers = layer_types.count("dense")
        sparse_layers = layer_types.count("sparse")
    elif "first_k_dense_replace" in dims_config:
        dense_layers = int(dims_config["first_k_dense_replace"])
        sparse_layers = int(dims_config["num_hidden_layers"]) - dense_layers

    comparisons: dict[str, tuple[Any, Any]] = {
        "architecture": (profile.architecture, config.get("model_type")),
        "num_layers": (profile.num_layers, dims_config.get("num_hidden_layers")),
        "hidden_size": (profile.hidden_size, dims_config.get("hidden_size")),
        "moe_intermediate_size": (
            profile.moe_intermediate_size,
            dims_config.get("moe_intermediate_size"),
        ),
        "num_experts": (
            profile.num_experts,
            dims_config.get("n_routed_experts", dims_config.get("num_experts")),
        ),
        "experts_per_tok": (
            profile.experts_per_tok,
            dims_config.get("num_experts_per_tok"),
        ),
        "vocab_size": (profile.vocab_size, dims_config.get("vocab_size")),
    }
    if profile.num_sparse_layers is not None and sparse_layers is not None:
        comparisons["num_sparse_layers"] = (profile.num_sparse_layers, sparse_layers)
    if profile.first_sparse_layer is not None and dense_layers is not None:
        comparisons["first_sparse_layer"] = (profile.first_sparse_layer, dense_layers)
    if profile.shared_experts is not None and dims_config.get("n_shared_experts") is not None:
        comparisons["shared_experts"] = (
            profile.shared_experts,
            dims_config.get("n_shared_experts"),
        )

    mismatches: list[str] = []
    for field, (profile_value, config_value) in comparisons.items():
        if config_value is None:
            continue
        if profile_value != config_value:
            mismatches.append(f"{field}: profile={profile_value} config={config_value}")
    return mismatches
