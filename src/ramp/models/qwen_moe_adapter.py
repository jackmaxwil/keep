from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from mlx.utils import tree_flatten

from keep.io.source_safetensors import read_indexed_safetensors_tensor_mlx
from keep.io.load import load_quantized_vq_switch_linear
from ramp.models.glm4_moe_adapter import QuantizedVQSwitchGLU


_QWEN_LANGUAGE_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(?P<layer>\d+)\.")


@dataclass(frozen=True)
class QwenMoeBindReport:
    bound_layers: tuple[int, ...]
    skipped_layers: tuple[int, ...]
    missing_layers: tuple[int, ...]


@dataclass(frozen=True)
class QwenNonExpertBindReport:
    loaded_model_parameters: tuple[str, ...]
    skipped_routed_expert_tensors: tuple[str, ...]
    skipped_mtp_tensors: tuple[str, ...]
    skipped_visual_tensors: tuple[str, ...]
    skipped_unmatched_tensors: tuple[str, ...]
    missing_model_parameters: tuple[str, ...]
    requested_layers: tuple[int, ...] | None
    skipped_by_layer_tensors: tuple[str, ...]


@dataclass(frozen=True)
class _QwenNonExpertBindEntry:
    source_name: str
    target_name: str


def _qwen_moe_switch_prefix(*, layer: int, projection: str) -> str:
    return f"model.language_model.layers.{layer}.mlp.switch_mlp.{projection}"


def _load_materialization_manifest(artifact_dir: Path) -> Mapping[str, object]:
    manifest_path = artifact_dir / "qwen-moe-materialization-manifest.json"
    payload = json.loads(manifest_path.read_text())
    if not isinstance(payload, Mapping):
        raise ValueError("Qwen materialization manifest must be a JSON object")
    return payload


def _projection_file_from_manifest(
    artifact_dir: Path,
    *,
    layer: int,
    projection: str,
) -> Path:
    payload = _load_materialization_manifest(artifact_dir)
    groups = payload.get("groups")
    if not isinstance(groups, list):
        raise ValueError("Qwen materialization manifest must contain a groups list")
    for record in groups:
        if not isinstance(record, Mapping):
            raise ValueError("Qwen materialization manifest group entries must be objects")
        if int(record.get("layer", -1)) != layer:
            continue
        target_projections = tuple(str(item) for item in record.get("target_projections", ()))
        if projection not in target_projections:
            continue
        output_path = Path(str(record.get("output_path")))
        if output_path.is_absolute() or output_path.exists():
            return output_path
        return artifact_dir / output_path
    raise FileNotFoundError(
        f"Qwen materialization manifest has no layer {layer} {projection} projection"
    )


def load_qwen_moe_switch_glu(
    artifact_dir: str | Path,
    *,
    layer: int,
    prefill_engine: Literal["auto", "vq_metal", "nax_e8", "nax_e8p"] = "auto",
) -> QuantizedVQSwitchGLU:
    artifact_root = Path(artifact_dir)
    return QuantizedVQSwitchGLU(
        gate_proj=load_quantized_vq_switch_linear(
            _projection_file_from_manifest(artifact_root, layer=layer, projection="gate_proj"),
            _qwen_moe_switch_prefix(layer=layer, projection="gate_proj"),
        ),
        up_proj=load_quantized_vq_switch_linear(
            _projection_file_from_manifest(artifact_root, layer=layer, projection="up_proj"),
            _qwen_moe_switch_prefix(layer=layer, projection="up_proj"),
        ),
        down_proj=load_quantized_vq_switch_linear(
            _projection_file_from_manifest(artifact_root, layer=layer, projection="down_proj"),
            _qwen_moe_switch_prefix(layer=layer, projection="down_proj"),
        ),
        prefill_engine=prefill_engine,
    )


def _qwen_language_layers(model: Any) -> list[Any]:
    layers = getattr(model, "layers", None)
    if layers is not None:
        return list(layers)
    language_model = getattr(model, "language_model", None)
    layers = getattr(language_model, "layers", None)
    if layers is not None:
        return list(layers)
    nested_model = getattr(language_model, "model", None)
    layers = getattr(nested_model, "layers", None)
    if layers is not None:
        return list(layers)
    raise ValueError("Qwen model must expose layers or language_model(.model).layers")


def _is_qwen_routed_expert_tensor(name: str) -> bool:
    return ".mlp.experts." in name


def _qwen_source_target_candidates(name: str) -> tuple[str, ...]:
    if name == "lm_head.weight" or name.startswith("lm_head."):
        return (
            name,
            f"language_model.{name}",
        )
    if name.startswith("model.language_model."):
        suffix = name.removeprefix("model.language_model.")
        return (
            f"language_model.{suffix}",
            f"language_model.model.{suffix}",
            f"model.{suffix}",
        )
    if name.startswith("language_model."):
        return (
            name,
            f"language_model.model.{name.removeprefix('language_model.')}",
        )
    return ()


def _qwen_source_layer(name: str) -> int | None:
    match = _QWEN_LANGUAGE_LAYER_RE.match(name)
    if match is None:
        return None
    return int(match.group("layer"))


def _normalize_qwen_layers(layers: tuple[int, ...] | list[int] | None) -> tuple[int, ...] | None:
    if layers is None:
        return None
    return tuple(sorted({int(layer) for layer in layers}))


def _target_qwen_non_expert_params(
    model: Any,
    *,
    layers: tuple[int, ...] | list[int] | None = None,
) -> dict[str, Any]:
    selected = _normalize_qwen_layers(layers)
    target_params: dict[str, Any] = {}
    for key, value in tree_flatten(model.parameters()):
        if ".mlp.experts." in key or ".switch_mlp." in key:
            continue
        if selected is not None:
            match = re.match(
                r"^(?:language_model\.layers|language_model\.model\.layers|model\.layers)\.(?P<layer>\d+)\.",
                key,
            )
            if match is not None and int(match.group("layer")) not in selected:
                continue
        target_params[key] = value
    return target_params


def _plan_qwen_non_expert_bind(
    model: Any,
    index,
    *,
    layers: tuple[int, ...] | list[int] | None = None,
) -> tuple[list[_QwenNonExpertBindEntry], QwenNonExpertBindReport]:
    selected = _normalize_qwen_layers(layers)
    target_params = _target_qwen_non_expert_params(model, layers=selected)
    entries: list[_QwenNonExpertBindEntry] = []
    loaded: set[str] = set()
    skipped_routed_experts: list[str] = []
    skipped_mtp: list[str] = []
    skipped_visual: list[str] = []
    skipped_unmatched: list[str] = []
    skipped_by_layer: list[str] = []

    for source_name in sorted(index.weight_map):
        if source_name.startswith("mtp."):
            skipped_mtp.append(source_name)
            continue
        if source_name.startswith("model.visual."):
            skipped_visual.append(source_name)
            continue
        if _is_qwen_routed_expert_tensor(source_name):
            skipped_routed_experts.append(source_name)
            continue

        layer = _qwen_source_layer(source_name)
        if selected is not None and layer is not None and layer not in selected:
            skipped_by_layer.append(source_name)
            continue

        target_name = next(
            (
                candidate
                for candidate in _qwen_source_target_candidates(source_name)
                if candidate in target_params
            ),
            None,
        )
        if target_name is None:
            skipped_unmatched.append(source_name)
            continue

        entries.append(_QwenNonExpertBindEntry(source_name, target_name))
        loaded.add(target_name)

    missing = tuple(sorted(set(target_params) - loaded))
    return entries, QwenNonExpertBindReport(
        loaded_model_parameters=tuple(sorted(loaded)),
        skipped_routed_expert_tensors=tuple(skipped_routed_experts),
        skipped_mtp_tensors=tuple(skipped_mtp),
        skipped_visual_tensors=tuple(skipped_visual),
        skipped_unmatched_tensors=tuple(skipped_unmatched),
        missing_model_parameters=missing,
        requested_layers=selected,
        skipped_by_layer_tensors=tuple(skipped_by_layer),
    )


def _sanitize_qwen_source_weight(
    model: Any,
    *,
    source_name: str,
    target_name: str,
    value: Any,
) -> Any | None:
    sanitizer = getattr(model, "sanitize", None)
    if sanitizer is None:
        return None
    sanitized = sanitizer({source_name: value})
    if not isinstance(sanitized, Mapping):
        return None
    return sanitized.get(target_name)


def bind_qwen_non_expert_weights(
    model: Any,
    source_dir: str | Path,
    index,
    *,
    layers: tuple[int, ...] | list[int] | None = None,
    strict: bool = True,
) -> QwenNonExpertBindReport:
    entries, report = _plan_qwen_non_expert_bind(model, index, layers=layers)
    target_params = _target_qwen_non_expert_params(model, layers=layers)
    if strict and report.missing_model_parameters:
        missing = ",\n".join(report.missing_model_parameters)
        raise ValueError(f"Missing {len(report.missing_model_parameters)} Qwen non-expert parameters: \n{missing}.")

    weights = []
    for entry in entries:
        value = read_indexed_safetensors_tensor_mlx(source_dir, index, entry.source_name)
        expected = target_params.get(entry.target_name)
        if expected is None:
            raise ValueError(f"Received Qwen non-expert parameter not in model: {entry.target_name}")
        if value.shape != expected.shape:
            sanitized_value = _sanitize_qwen_source_weight(
                model,
                source_name=entry.source_name,
                target_name=entry.target_name,
                value=value,
            )
            if sanitized_value is not None:
                value = sanitized_value
        if value.shape != expected.shape:
            raise ValueError(
                f"Expected shape {expected.shape} but received shape {value.shape} for parameter {entry.target_name}"
            )
        weights.append((entry.target_name, value))

    if weights:
        model.load_weights(weights, strict=False)
    return report


def bind_qwen_moe_vq_experts(
    model: Any,
    artifact_dir: str | Path,
    *,
    layers: tuple[int, ...] | None = None,
    prefill_engine: Literal["auto", "vq_metal", "nax_e8", "nax_e8p"] = "auto",
) -> QwenMoeBindReport:
    model_layers = _qwen_language_layers(model)
    selected = set(range(len(model_layers))) if layers is None else set(layers)
    bound: list[int] = []
    skipped: list[int] = []
    missing: list[int] = []
    for layer_idx, layer in enumerate(model_layers):
        if layer_idx not in selected:
            skipped.append(layer_idx)
            continue
        mlp = getattr(layer, "mlp", None)
        binder = getattr(mlp, "bind_switch_mlp", None)
        has_switch_slot = hasattr(mlp, "switch_mlp")
        if binder is None and not has_switch_slot:
            missing.append(layer_idx)
            continue
        try:
            switch_mlp = load_qwen_moe_switch_glu(
                artifact_dir,
                layer=layer_idx,
                prefill_engine=prefill_engine,
            )
        except FileNotFoundError:
            missing.append(layer_idx)
            continue
        if binder is not None:
            binder(switch_mlp)
        else:
            setattr(mlp, "switch_mlp", switch_mlp)
        bound.append(layer_idx)
    for layer_idx in sorted(selected - set(range(len(model_layers)))):
        missing.append(layer_idx)
    return QwenMoeBindReport(
        bound_layers=tuple(bound),
        skipped_layers=tuple(skipped),
        missing_layers=tuple(missing),
    )


def has_unbound_qwen_moe_vq_experts(
    model: Any,
    *,
    layers: tuple[int, ...] | None = None,
) -> bool:
    model_layers = _qwen_language_layers(model)
    selected = set(range(len(model_layers))) if layers is None else set(layers)
    for layer_idx in selected:
        if layer_idx < 0 or layer_idx >= len(model_layers):
            return True
        mlp = getattr(model_layers[layer_idx], "mlp", None)
        if getattr(mlp, "switch_mlp", None) is None:
            return True
    return False
