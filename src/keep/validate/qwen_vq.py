from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from keep.vq.e8 import cosine_similarity, decode_weight_matrix
from keep.io.load import (
    infer_vq_switch_linear_dims,
    inspect_safetensors,
    load_quantized_vq_switch_linear,
)
from ramp.nn.linear import QuantizedVQLinear
from ramp.nn.switch_linear import QuantizedVQSwitchLinear


@dataclass(frozen=True)
class ConvertedLinearInfo:
    path: str
    input_dims: int
    output_dims: int
    group_size: int
    code_bits: int


@dataclass(frozen=True)
class QwenVQValidationResult:
    model: str
    prompt: str
    prompt_tokens: int
    generated_tokens: list[int]
    generated_text: str
    logits_cosine: float
    logits_max_abs: float
    converted: list[ConvertedLinearInfo]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["converted"] = [asdict(item) for item in self.converted]
        return data


@dataclass(frozen=True)
class QwenMoeSwitchValidationResult:
    path: str
    layer: int
    projection: str
    input_dims: int
    output_dims: int
    num_experts: int
    group_size: int
    code_bits: int
    output_shape: tuple[int, ...]
    max_abs_diff: float
    cosine: float

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["output_shape"] = list(self.output_shape)
        return data


@dataclass(frozen=True)
class QwenMoeMaterializationAudit:
    artifact_dir: str
    manifest_path: str
    model_id: str | None
    revision: str | None
    materialization_status: str | None
    source_projection_groups: int
    target_projection_groups: int
    loadable_projection_groups: int
    expected_source_projection_groups: int | None
    expected_target_projection_groups: int | None
    projection_files: tuple[str, ...]
    missing_projection_files: tuple[str, ...]
    missing_projection_tensors: tuple[str, ...]
    load_errors: tuple[str, ...]
    dense_routed_experts: bool
    unbound_vq_experts: bool
    effective_routed_bpw: float | None
    checks: dict[str, bool]

    @property
    def audit_pass(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["audit_pass"] = self.audit_pass
        data["projection_files"] = list(self.projection_files)
        data["missing_projection_files"] = list(self.missing_projection_files)
        data["missing_projection_tensors"] = list(self.missing_projection_tensors)
        data["load_errors"] = list(self.load_errors)
        return data


class DenseDequantLinear(nn.Module):
    """Dense reference layer materialized from VQ codes for validation only."""

    def __init__(
        self,
        *,
        weight: np.ndarray,
        bias: mx.array | None = None,
    ):
        super().__init__()
        self.weight = mx.array(weight.astype(np.float32, copy=False))
        if bias is not None:
            self.bias = bias
        self.freeze()

    def __call__(self, x: mx.array) -> mx.array:
        y = x @ self.weight.T
        bias = self.get("bias")
        if bias is not None:
            y = y + bias
        return y


def _get_child(obj, part: str):
    if isinstance(obj, (list, tuple)):
        return obj[int(part)]
    if isinstance(obj, dict):
        return obj[part]
    return getattr(obj, part)


def get_module_by_path(root, path: str):
    value = root
    for part in path.split("."):
        value = _get_child(value, part)
    return value


def set_module_by_path(root, path: str, value) -> None:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = _get_child(parent, part)
    last = parts[-1]
    if isinstance(parent, list):
        parent[int(last)] = value
    elif isinstance(parent, dict):
        parent[last] = value
    else:
        setattr(parent, last, value)


def linear_module_paths(model) -> list[str]:
    return [path for path, module in model.named_modules() if isinstance(module, nn.Linear)]


def qwen_moe_switch_prefix(*, layer: int, projection: str) -> str:
    if layer < 0:
        raise ValueError("layer must be non-negative")
    if projection not in {"gate_proj", "up_proj", "down_proj"}:
        raise ValueError("projection must be gate_proj, up_proj, or down_proj")
    return f"model.language_model.layers.{layer}.mlp.switch_mlp.{projection}"


def load_qwen_moe_switch_projection(
    path: str | Path,
    *,
    layer: int,
    projection: str,
) -> QuantizedVQSwitchLinear:
    return load_quantized_vq_switch_linear(
        path,
        qwen_moe_switch_prefix(layer=layer, projection=projection),
    )


def validate_qwen_moe_switch_projection(
    path: str | Path,
    *,
    layer: int,
    projection: str,
    seed: int = 20260702,
) -> QwenMoeSwitchValidationResult:
    loaded = load_qwen_moe_switch_projection(path, layer=layer, projection=projection)
    reference = QuantizedVQSwitchLinear(
        input_dims=loaded.input_dims,
        output_dims=loaded.output_dims,
        num_experts=loaded.num_experts,
        codes=loaded.codes,
        scales=loaded.scales,
        codebook=loaded.codebook,
        bias=loaded.get("bias"),
        group_size=loaded.group_size,
        code_bits=loaded.code_bits,
        use_gather_vqmm=False,
        rht_signs=loaded.get("rht_signs"),
    )
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(2, loaded.input_dims)).astype(np.float32)
    indices = np.stack(
        [
            np.arange(2, dtype=np.int32) % loaded.num_experts,
            (np.arange(2, dtype=np.int32) + 1) % loaded.num_experts,
        ],
        axis=0,
    )
    actual = loaded(mx.array(x), mx.array(indices))
    expected = reference(mx.array(x), mx.array(indices))
    mx.eval(actual, expected)
    actual_np = np.array(actual.astype(mx.float32), copy=False)
    expected_np = np.array(expected.astype(mx.float32), copy=False)
    return QwenMoeSwitchValidationResult(
        path=str(path),
        layer=layer,
        projection=projection,
        input_dims=loaded.input_dims,
        output_dims=loaded.output_dims,
        num_experts=loaded.num_experts,
        group_size=loaded.group_size,
        code_bits=loaded.code_bits,
        output_shape=tuple(int(dim) for dim in actual.shape),
        max_abs_diff=float(np.max(np.abs(actual_np - expected_np))),
        cosine=cosine_similarity(actual_np, expected_np),
    )


def _resolve_manifest_path(path: str | Path) -> tuple[Path, Path]:
    root = Path(path)
    if root.is_dir():
        return root, root / "qwen-moe-materialization-manifest.json"
    return root.parent, root


def _resolve_projection_path(artifact_dir: Path, value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute() or path.exists():
        return path
    return artifact_dir / path


def audit_qwen_moe_materialization_manifest(
    path: str | Path,
    *,
    expected_source_projection_groups: int | None = None,
    expected_target_projection_groups: int | None = None,
) -> QwenMoeMaterializationAudit:
    artifact_dir, manifest_path = _resolve_manifest_path(path)
    payload = json.loads(manifest_path.read_text())
    groups = payload.get("groups")
    if not isinstance(groups, list):
        raise ValueError("Qwen materialization manifest must contain a groups list")

    projection_files: list[str] = []
    missing_files: list[str] = []
    missing_tensors: list[str] = []
    load_errors: list[str] = []
    target_projection_groups = 0
    loadable_projection_groups = 0
    stored_projection_bytes = 0
    dense_weight_count = 0
    inspections = {}

    for record in groups:
        if not isinstance(record, dict):
            raise ValueError("Qwen materialization manifest group entries must be objects")
        output_path = _resolve_projection_path(artifact_dir, record.get("output_path"))
        output_text = str(output_path)
        if output_text not in projection_files:
            projection_files.append(output_text)
        target_projections = tuple(str(item) for item in record.get("target_projections", ()))
        layer = int(record.get("layer", -1))
        if not output_path.exists():
            if output_text not in missing_files:
                missing_files.append(output_text)
            target_projection_groups += len(target_projections)
            continue
        if output_path not in inspections:
            inspections[output_path] = inspect_safetensors(output_path)
        inspection = inspections[output_path]

        for projection in target_projections:
            target_projection_groups += 1
            prefix = qwen_moe_switch_prefix(layer=layer, projection=projection)
            codes_name = f"{prefix}.codes"
            scales_name = f"{prefix}.scales"
            try:
                in_dim, out_dim, experts, _group_size, _code_bits = (
                    infer_vq_switch_linear_dims(inspection, prefix)
                )
            except KeyError as exc:
                missing_tensors.append(str(exc))
                continue
            except Exception as exc:  # noqa: BLE001 - audit records exact loader failure.
                load_errors.append(f"{output_path}:{prefix}: {exc}")
                continue
            loadable_projection_groups += 1
            stored_projection_bytes += (
                inspection.tensors[codes_name].nbytes
                + inspection.tensors[scales_name].nbytes
            )
            dense_weight_count += experts * out_dim * in_dim

    effective_bpw = (
        None
        if dense_weight_count == 0
        else stored_projection_bytes * 8.0 / dense_weight_count
    )
    checks = {
        "materialization_status": payload.get("materialization_status")
        in {"qwen_moe_groups_materialized", "qwen_moe_groups_all_existing"},
        "expected_source_projection_groups": (
            True
            if expected_source_projection_groups is None
            else len(groups) == expected_source_projection_groups
        ),
        "expected_target_projection_groups": (
            True
            if expected_target_projection_groups is None
            else target_projection_groups == expected_target_projection_groups
        ),
        "projection_files_present": not missing_files,
        "projection_tensors_present": not missing_tensors,
        "projection_headers_loadable": not load_errors,
        "all_target_projections_loadable": target_projection_groups == loadable_projection_groups,
        "dense_routed_experts_false": True,
        "unbound_vq_experts_false": not missing_files and not missing_tensors and not load_errors,
    }
    return QwenMoeMaterializationAudit(
        artifact_dir=str(artifact_dir),
        manifest_path=str(manifest_path),
        model_id=None if payload.get("model_id") is None else str(payload.get("model_id")),
        revision=None if payload.get("revision") is None else str(payload.get("revision")),
        materialization_status=(
            None
            if payload.get("materialization_status") is None
            else str(payload.get("materialization_status"))
        ),
        source_projection_groups=len(groups),
        target_projection_groups=target_projection_groups,
        loadable_projection_groups=loadable_projection_groups,
        expected_source_projection_groups=expected_source_projection_groups,
        expected_target_projection_groups=expected_target_projection_groups,
        projection_files=tuple(projection_files),
        missing_projection_files=tuple(missing_files),
        missing_projection_tensors=tuple(missing_tensors),
        load_errors=tuple(load_errors),
        dense_routed_experts=False,
        unbound_vq_experts=not checks["unbound_vq_experts_false"],
        effective_routed_bpw=effective_bpw,
        checks=checks,
    )


def select_linear_paths(
    model,
    *,
    target: str | None,
    max_linears: int,
    require_group_size: int,
) -> list[str]:
    paths = linear_module_paths(model)
    if target:
        paths = [path for path in paths if target in path]
    selected: list[str] = []
    for path in paths:
        module = get_module_by_path(model, path)
        if module.weight.shape[1] % require_group_size == 0:
            selected.append(path)
        if len(selected) >= max_linears:
            break
    if not selected:
        raise ValueError(f"no linear modules matched target={target!r} and group_size={require_group_size}")
    return selected


def make_vq_and_dense_reference(
    module: nn.Linear,
    *,
    group_size: int,
) -> tuple[QuantizedVQLinear, DenseDequantLinear, ConvertedLinearInfo]:
    bias = module.get("bias")
    vq_layer = QuantizedVQLinear.from_weights(module.weight, bias, group_size=group_size)
    dense_weight = decode_weight_matrix(
        np.array(vq_layer.codes, copy=False),
        np.array(vq_layer.scales, copy=False),
        code_bits=vq_layer.code_bits,
        codebook=np.array(vq_layer.codebook, copy=False),
    )
    dense_layer = DenseDequantLinear(weight=dense_weight, bias=bias)
    info = ConvertedLinearInfo(
        path="",
        input_dims=vq_layer.input_dims,
        output_dims=vq_layer.output_dims,
        group_size=vq_layer.group_size,
        code_bits=vq_layer.code_bits,
    )
    return vq_layer, dense_layer, info


def _logits_for_tokens(model, tokens: Iterable[int]) -> np.ndarray:
    token_array = mx.array([list(tokens)])
    logits = model(token_array)[:, -1, :]
    mx.eval(logits)
    return np.array(logits[0].astype(mx.float32), copy=False)


def _greedy_generate(model, tokenizer, tokens: list[int], max_new_tokens: int) -> list[int]:
    generated = list(tokens)
    for _ in range(max_new_tokens):
        logits = _logits_for_tokens(model, generated)
        generated.append(int(np.argmax(logits)))
        if generated[-1] in tokenizer.eos_token_ids:
            break
    return generated


def validate_qwen_vq(
    model_id_or_path: str,
    *,
    prompt: str,
    target: str = "model.layers.0.self_attn.q_proj",
    max_linears: int = 1,
    group_size: int = 512,
    max_new_tokens: int = 4,
) -> QwenVQValidationResult:
    from mlx_lm.utils import load

    model, tokenizer = load(model_id_or_path, lazy=False)
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    paths = select_linear_paths(
        model,
        target=target,
        max_linears=max_linears,
        require_group_size=group_size,
    )

    replacements: list[tuple[str, object, QuantizedVQLinear, DenseDequantLinear, ConvertedLinearInfo]] = []
    for path in paths:
        original = get_module_by_path(model, path)
        vq_layer, dense_layer, info = make_vq_and_dense_reference(original, group_size=group_size)
        replacements.append((
            path,
            original,
            vq_layer,
            dense_layer,
            ConvertedLinearInfo(
                path=path,
                input_dims=info.input_dims,
                output_dims=info.output_dims,
                group_size=info.group_size,
                code_bits=info.code_bits,
            ),
        ))

    for path, _, _, dense_layer, _ in replacements:
        set_module_by_path(model, path, dense_layer)
    dense_logits = _logits_for_tokens(model, token_ids)

    for path, _, vq_layer, _, _ in replacements:
        set_module_by_path(model, path, vq_layer)
    vq_logits = _logits_for_tokens(model, token_ids)
    generated_tokens = _greedy_generate(model, tokenizer, token_ids, max_new_tokens)

    for path, original, _, _, _ in replacements:
        set_module_by_path(model, path, original)

    return QwenVQValidationResult(
        model=model_id_or_path,
        prompt=prompt,
        prompt_tokens=len(token_ids),
        generated_tokens=generated_tokens,
        generated_text=tokenizer.decode(generated_tokens),
        logits_cosine=cosine_similarity(vq_logits, dense_logits),
        logits_max_abs=float(np.max(np.abs(vq_logits - dense_logits))),
        converted=[info for *_, info in replacements],
    )
