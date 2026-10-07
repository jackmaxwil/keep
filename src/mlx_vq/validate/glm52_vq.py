from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_vq.convert.inspect_hf import GLM52_MODEL_ID, summarize_config, validate_glm52
from mlx_vq.convert.stream_convert import load_safetensors_index
from mlx_vq.io.load import current_rss_bytes, infer_vq_switch_linear_dims, inspect_safetensors
from mlx_vq.models.glm4_moe_adapter import GLM4MoEGate
from mlx_vq.validate.glm45_air_vq import (
    ArrayComparisonMetrics,
    _array_metrics,
    _compute_oracles,
    _load_switch_glu,
    _read_named_tensor,
    _routing_config,
)
from mlx_vq.validate.glm52_artifact import (
    GLM52ReapMaterializationAudit,
    GLM52ReapSourceAccounting,
    audit_glm52_reap_materialization_manifest,
    audit_glm52_reap_source_accounting,
)


@dataclass(frozen=True)
class VQProjectionArtifactMetadata:
    projection: str
    path: str
    input_dims: int
    output_dims: int
    num_experts: int
    group_size: int
    code_bits: int
    codebook_name: str | None
    codebook_sha256: str | None
    scale_estimator: str | None
    rht: str | None
    rht_seed: str | None
    codes_dtype: str
    scales_dtype: str


@dataclass(frozen=True)
class GLM52VQValidationResult:
    model_id: str
    layer: int
    tokens: int
    hidden_size: int
    moe_intermediate_size: int
    top_k: int
    selected_experts: tuple[int, ...]
    routes: int
    source_tensors_read: int
    peak_source_tensor_bytes: int
    rss_start_bytes: int
    rss_after_vq_load_bytes: int
    rss_after_runtime_bytes: int
    rss_after_oracles_bytes: int
    source_dir: str
    artifact_dir: str
    routed_indices: tuple[tuple[int, ...], ...]
    router_scores: tuple[tuple[float, ...], ...]
    metrics: dict[str, ArrayComparisonMetrics]
    source_artifact_metrics: dict[str, ArrayComparisonMetrics]
    artifact_quantization: dict[str, VQProjectionArtifactMetadata]
    source_artifact_diagnosis: str

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "layer": self.layer,
            "tokens": self.tokens,
            "hidden_size": self.hidden_size,
            "moe_intermediate_size": self.moe_intermediate_size,
            "top_k": self.top_k,
            "selected_experts": list(self.selected_experts),
            "routes": self.routes,
            "source_tensors_read": self.source_tensors_read,
            "peak_source_tensor_bytes": self.peak_source_tensor_bytes,
            "rss_start_bytes": self.rss_start_bytes,
            "rss_after_vq_load_bytes": self.rss_after_vq_load_bytes,
            "rss_after_runtime_bytes": self.rss_after_runtime_bytes,
            "rss_after_oracles_bytes": self.rss_after_oracles_bytes,
            "source_dir": self.source_dir,
            "artifact_dir": self.artifact_dir,
            "routed_indices": [list(row) for row in self.routed_indices],
            "router_scores": [list(row) for row in self.router_scores],
            "metrics": {name: asdict(metrics) for name, metrics in self.metrics.items()},
            "source_artifact_metrics": {
                name: asdict(metrics) for name, metrics in self.source_artifact_metrics.items()
            },
            "artifact_quantization": {
                name: asdict(metadata) for name, metadata in self.artifact_quantization.items()
            },
            "source_artifact_diagnosis": self.source_artifact_diagnosis,
        }


def _is_sparse_layer(config: dict[str, Any], layer: int, *, dense_layers: int, num_hidden_layers: int) -> bool:
    if layer < 0 or layer >= num_hidden_layers:
        return False
    layer_types = config.get("mlp_layer_types")
    if layer_types is not None:
        return list(layer_types)[layer] == "sparse"
    return layer >= dense_layers


def _switch_prefix(layer: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.switch_mlp.{projection}"


def _artifact_projection_path(artifact_dir: Path, layer: int, projection: str) -> Path:
    return artifact_dir / f"layer-{layer:05d}-{projection}.safetensors"


def _artifact_quantization_metadata(
    artifact_dir: Path,
    *,
    layer: int,
) -> dict[str, VQProjectionArtifactMetadata]:
    records: dict[str, VQProjectionArtifactMetadata] = {}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        path = _artifact_projection_path(artifact_dir, layer, projection)
        prefix = _switch_prefix(layer, projection)
        inspection = inspect_safetensors(path)
        input_dims, output_dims, num_experts, group_size, code_bits = infer_vq_switch_linear_dims(
            inspection,
            prefix,
        )
        raw_config = inspection.metadata.get("quantization_config")
        quantization_config = json.loads(raw_config) if raw_config is not None else {}
        codebook = quantization_config.get("codebook")
        policy = quantization_config.get("policy")
        codes = inspection.tensors[f"{prefix}.codes"]
        scales = inspection.tensors[f"{prefix}.scales"]
        records[projection] = VQProjectionArtifactMetadata(
            projection=projection,
            path=str(path),
            input_dims=input_dims,
            output_dims=output_dims,
            num_experts=num_experts,
            group_size=group_size,
            code_bits=code_bits,
            codebook_name=str(codebook.get("name")) if isinstance(codebook, dict) and codebook.get("name") else None,
            codebook_sha256=str(codebook.get("sha256"))
            if isinstance(codebook, dict) and codebook.get("sha256")
            else None,
            scale_estimator=str(policy.get("scale_estimator"))
            if isinstance(policy, dict) and policy.get("scale_estimator")
            else None,
            rht=str(policy.get("rht")) if isinstance(policy, dict) and policy.get("rht") else None,
            rht_seed=str(policy.get("rht_seed")) if isinstance(policy, dict) and policy.get("rht_seed") else None,
            codes_dtype=codes.dtype,
            scales_dtype=scales.dtype,
        )
    return records


def _source_artifact_diagnosis(
    *,
    metrics: dict[str, ArrayComparisonMetrics],
    source_artifact_metrics: dict[str, ArrayComparisonMetrics],
    artifact_quantization: dict[str, VQProjectionArtifactMetadata],
) -> str:
    min_artifact_cosine = min(
        metrics[name].cosine
        for name in (
            "artifact_gate_proj",
            "artifact_up_proj",
            "artifact_routed_glu",
            "artifact_weighted_routed",
        )
    )
    source_weighted_cosine = metrics["source_weighted_routed"].cosine
    source_artifact_weighted_cosine = source_artifact_metrics["weighted_routed"].cosine
    code_bits = {metadata.code_bits for metadata in artifact_quantization.values()}
    if (
        min_artifact_cosine >= 0.999
        and abs(source_weighted_cosine - source_artifact_weighted_cosine) <= 1.0e-4
        and code_bits == {8}
    ):
        return "vq1_artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
    if min_artifact_cosine >= 0.999 and abs(source_weighted_cosine - source_artifact_weighted_cosine) <= 1.0e-4:
        return "artifact_reconstruction_gap_not_runtime_or_source_binding_failure"
    if min_artifact_cosine < 0.999:
        return "runtime_artifact_parity_gap"
    return "source_oracle_gap_requires_followup"


def validate_glm52_vq(
    config: dict[str, Any],
    *,
    source_dir: str | Path,
    index_path: str | Path,
    artifact_dir: str | Path,
    model_id: str = GLM52_MODEL_ID,
    layer: int = 3,
    tokens: int = 1,
    seed: int = 20260624,
    input_scale: float = 1.0,
    strict_config: bool = True,
) -> GLM52VQValidationResult:
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if input_scale <= 0:
        raise ValueError("input_scale must be positive")
    rss_start = current_rss_bytes()

    summary = summarize_config(config, model_id=model_id)
    if strict_config:
        failed_checks = validate_glm52(summary)
        if failed_checks:
            raise ValueError(f"config does not match GLM-5.2: {', '.join(failed_checks)}")
    if summary.model_type != "glm_moe_dsa":
        raise ValueError(f"expected model_type='glm_moe_dsa', found {summary.model_type!r}")
    if not _is_sparse_layer(
        config,
        layer,
        dense_layers=summary.dense_layers,
        num_hidden_layers=summary.num_hidden_layers,
    ):
        raise ValueError(f"layer {layer} is not a sparse MoE layer")

    source_root = Path(source_dir)
    artifact_root = Path(artifact_dir)
    index = load_safetensors_index(index_path)
    routing = _routing_config(config, model_id=model_id)

    gate_weight_name = f"model.layers.{layer}.mlp.gate.weight"
    correction_name = f"model.layers.{layer}.mlp.gate.e_score_correction_bias"
    gate_weight = _read_named_tensor(source_root, index, gate_weight_name)
    source_tensors_read = 1
    peak_source_bytes = gate_weight.nbytes
    if gate_weight.shape != (summary.n_routed_experts, summary.hidden_size):
        raise ValueError(
            f"{gate_weight_name} must have shape "
            f"({summary.n_routed_experts}, {summary.hidden_size}), found {gate_weight.shape}"
        )

    if correction_name in index.weight_map:
        correction = _read_named_tensor(source_root, index, correction_name)
        source_tensors_read += 1
        peak_source_bytes = max(peak_source_bytes, correction.nbytes)
    else:
        correction = np.zeros((summary.n_routed_experts,), dtype=np.float32)
    if correction.shape != (summary.n_routed_experts,):
        raise ValueError(f"{correction_name} must have shape ({summary.n_routed_experts},), found {correction.shape}")

    rng = np.random.default_rng(seed)
    x = rng.normal(loc=0.0, scale=input_scale, size=(tokens, summary.hidden_size)).astype(np.float32)
    x_mx = mx.array(x)

    gate = GLM4MoEGate(
        routing,
        weight=mx.array(gate_weight),
        e_score_correction_bias=mx.array(correction),
    )
    indices_mx, scores_mx = gate(x_mx)
    mx.eval(indices_mx, scores_mx)
    indices = np.asarray(indices_mx, dtype=np.int64)
    scores = np.asarray(scores_mx.astype(mx.float32), dtype=np.float32)
    if indices.ndim != 2 or indices.shape[0] != tokens:
        raise ValueError(f"router indices must have shape [tokens, top_k], found {indices.shape}")

    switch_glu = _load_switch_glu(artifact_root, layer)
    mx.eval(
        switch_glu.gate_proj.codes,
        switch_glu.gate_proj.scales,
        switch_glu.gate_proj.codebook,
        switch_glu.up_proj.codes,
        switch_glu.up_proj.scales,
        switch_glu.up_proj.codebook,
        switch_glu.down_proj.codes,
        switch_glu.down_proj.scales,
        switch_glu.down_proj.codebook,
    )
    rss_after_vq_load = current_rss_bytes()
    route_indices_mx = mx.array(indices.astype(np.int32, copy=False))
    vq_gate_mx = switch_glu.gate_proj(x_mx, route_indices_mx)
    vq_up_mx = switch_glu.up_proj(x_mx, route_indices_mx)
    vq_routes_mx = switch_glu.down_proj(nn.silu(vq_gate_mx) * vq_up_mx, route_indices_mx)
    vq_weighted_mx = (vq_routes_mx * mx.array(scores)[..., None]).sum(axis=-2)
    mx.eval(vq_gate_mx, vq_up_mx, vq_routes_mx, vq_weighted_mx)

    vq_gate = np.asarray(vq_gate_mx.astype(mx.float32), dtype=np.float32)
    vq_up = np.asarray(vq_up_mx.astype(mx.float32), dtype=np.float32)
    vq_routes = np.asarray(vq_routes_mx.astype(mx.float32), dtype=np.float32)
    vq_weighted = np.asarray(vq_weighted_mx.astype(mx.float32), dtype=np.float32)
    rss_after_runtime = current_rss_bytes()

    oracles = _compute_oracles(
        source_dir=source_root,
        index=index,
        switch_glu=switch_glu,
        layer=layer,
        x=x,
        indices=indices,
        source_tensors_already_read=source_tensors_read,
        peak_source_tensor_bytes=peak_source_bytes,
    )
    artifact_weighted = (oracles.artifact_routes * scores[..., None]).sum(axis=-2)
    source_weighted = (oracles.source_routes * scores[..., None]).sum(axis=-2)
    rss_after_oracles = current_rss_bytes()

    source_artifact_metrics = {
        "gate_proj": _array_metrics(oracles.artifact_gate, oracles.source_gate),
        "up_proj": _array_metrics(oracles.artifact_up, oracles.source_up),
        "routed_glu": _array_metrics(oracles.artifact_routes, oracles.source_routes),
        "weighted_routed": _array_metrics(artifact_weighted, source_weighted),
    }

    metrics = {
        "artifact_gate_proj": _array_metrics(vq_gate, oracles.artifact_gate),
        "artifact_up_proj": _array_metrics(vq_up, oracles.artifact_up),
        "artifact_routed_glu": _array_metrics(vq_routes, oracles.artifact_routes),
        "artifact_weighted_routed": _array_metrics(vq_weighted, artifact_weighted),
        "source_gate_proj": _array_metrics(vq_gate, oracles.source_gate),
        "source_up_proj": _array_metrics(vq_up, oracles.source_up),
        "source_routed_glu": _array_metrics(vq_routes, oracles.source_routes),
        "source_weighted_routed": _array_metrics(vq_weighted, source_weighted),
    }
    artifact_quantization = _artifact_quantization_metadata(artifact_root, layer=layer)
    source_artifact_diagnosis = _source_artifact_diagnosis(
        metrics=metrics,
        source_artifact_metrics=source_artifact_metrics,
        artifact_quantization=artifact_quantization,
    )

    return GLM52VQValidationResult(
        model_id=model_id,
        layer=layer,
        tokens=tokens,
        hidden_size=summary.hidden_size,
        moe_intermediate_size=summary.moe_intermediate_size,
        top_k=indices.shape[1],
        selected_experts=tuple(sorted(int(value) for value in np.unique(indices))),
        routes=int(indices.size),
        source_tensors_read=oracles.source_tensors_read,
        peak_source_tensor_bytes=oracles.peak_source_tensor_bytes,
        rss_start_bytes=rss_start,
        rss_after_vq_load_bytes=rss_after_vq_load,
        rss_after_runtime_bytes=rss_after_runtime,
        rss_after_oracles_bytes=rss_after_oracles,
        source_dir=str(source_root),
        artifact_dir=str(artifact_root),
        routed_indices=tuple(tuple(int(value) for value in row) for row in indices),
        router_scores=tuple(tuple(float(value) for value in row) for row in scores),
        metrics=metrics,
        source_artifact_metrics=source_artifact_metrics,
        artifact_quantization=artifact_quantization,
        source_artifact_diagnosis=source_artifact_diagnosis,
    )
