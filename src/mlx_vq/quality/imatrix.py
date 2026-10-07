from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import mlx.core as mx
import numpy as np


ROUTED_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
AIR_IMATRIX_DEFAULT_MODEL_ID = "zai-org/GLM-4.5-Air"
AIR_IMATRIX_DEFAULT_ARTIFACT_DIR = "artifacts/glm-4.5-air-vq"
AIR_IMATRIX_COLLECTION_SOURCES = ("resident_vq_model", "streamed_source_weights")


@dataclass(frozen=True)
class ProjectionImatrixEntry:
    layer: int
    projection: str
    expert: int
    importance_sum: np.ndarray
    mean_importance: np.ndarray
    routing_weighted_importance: np.ndarray
    route_count: int
    total_route_count: int
    route_frequency: float
    prompt_ids: tuple[str, ...] = ()
    affinity_weighted_importance: np.ndarray | None = None
    affinity_score_sum: float | None = None

    @property
    def input_dim(self) -> int:
        return int(self.importance_sum.shape[0])

    @property
    def affinity_weighted_mean_importance(self) -> np.ndarray | None:
        if self.affinity_weighted_importance is None:
            return None
        score_sum = float(self.affinity_score_sum or 0.0)
        if score_sum <= 0.0:
            return np.zeros(self.input_dim, dtype=np.float32)
        return (self.affinity_weighted_importance.astype(np.float64) / score_sum).astype(np.float32)

    def manifest_entry(self, *, path: str) -> dict[str, object]:
        payload: dict[str, object] = {
            "layer": self.layer,
            "projection": self.projection,
            "expert": self.expert,
            "path": path,
            "input_dim": self.input_dim,
            "route_count": self.route_count,
            "total_route_count": self.total_route_count,
            "route_frequency": self.route_frequency,
            "prompt_ids": list(self.prompt_ids),
            "tensors": {
                "importance_sum": "sum_j(x_j^2) over routed activation rows for this expert",
                "mean_importance": "importance_sum / route_count",
                "routing_weighted_importance": "importance_sum / total_route_count",
            },
        }
        if self.affinity_weighted_importance is not None:
            tensors = payload["tensors"]
            if isinstance(tensors, dict):
                tensors["affinity_weighted_importance"] = (
                    "sum_j(router_score_j * x_j^2) over routed activation rows for this expert"
                )
                tensors["affinity_weighted_mean_importance"] = (
                    "affinity_weighted_importance / sum_j(router_score_j) for this expert"
                )
                payload["affinity_score_sum"] = float(self.affinity_score_sum or 0.0)
        return payload


def _validate_projection(projection: str) -> None:
    if projection not in ROUTED_PROJECTIONS:
        raise ValueError(f"unsupported projection {projection!r}; expected one of {ROUTED_PROJECTIONS}")


def _routed_rows(inputs: np.ndarray, route_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(inputs, dtype=np.float32)
    routes = np.asarray(route_indices, dtype=np.int64)
    if routes.ndim != 2:
        raise ValueError(f"route_indices must have shape [tokens, top_k], found {routes.shape}")
    if values.ndim == 2:
        if values.shape[0] != routes.shape[0]:
            raise ValueError(
                f"2D inputs must have one row per token; found inputs {values.shape} and routes {routes.shape}"
            )
        expanded = np.repeat(values[:, None, :], routes.shape[1], axis=1)
    elif values.ndim == 3:
        if values.shape[:2] != routes.shape:
            raise ValueError(
                f"3D inputs must have shape [tokens, top_k, dims]; found inputs {values.shape} "
                f"and routes {routes.shape}"
            )
        expanded = values
    else:
        raise ValueError(f"inputs must be rank 2 or 3, found {values.shape}")
    if expanded.shape[-1] <= 0:
        raise ValueError("inputs must have a non-empty trailing dimension")
    if not np.isfinite(expanded).all():
        raise ValueError("inputs must be finite")
    return expanded.reshape(-1, expanded.shape[-1]), routes.reshape(-1)


def _routed_scores(router_scores: np.ndarray | None, route_indices: np.ndarray) -> np.ndarray | None:
    if router_scores is None:
        return None
    scores = np.asarray(router_scores, dtype=np.float32)
    routes = np.asarray(route_indices, dtype=np.int64)
    if scores.shape != routes.shape:
        raise ValueError(f"router_scores must have shape {routes.shape}, found {scores.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("router_scores must be finite")
    if np.any(scores < 0.0):
        raise ValueError("router_scores must be non-negative")
    return scores.reshape(-1).astype(np.float64, copy=False)


def accumulate_routed_projection_imatrix(
    *,
    layer: int,
    projection: str,
    inputs: np.ndarray,
    route_indices: np.ndarray,
    router_scores: np.ndarray | None = None,
    num_experts: int,
    prompt_ids: Iterable[str] = (),
) -> tuple[ProjectionImatrixEntry, ...]:
    """Accumulate llama.cpp-style routed imatrix vectors for one projection.

    For gate/up projections, pass 2D inputs shaped `[tokens, input_dim]`; each
    selected route reuses the token input. For down projections, pass 3D inputs
    shaped `[tokens, top_k, input_dim]` so each routed expert gets its own GLU
    activation row.
    """

    _validate_projection(projection)
    if layer < 0:
        raise ValueError("layer must be non-negative")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    routed_inputs, routed_experts = _routed_rows(inputs, route_indices)
    if routed_experts.size == 0:
        raise ValueError("route_indices must not be empty")
    if int(np.min(routed_experts)) < 0 or int(np.max(routed_experts)) >= num_experts:
        raise ValueError(f"route_indices contain expert outside [0, {num_experts})")

    total_route_count = int(routed_experts.size)
    input_dim = int(routed_inputs.shape[-1])
    sums = np.zeros((num_experts, input_dim), dtype=np.float64)
    flat_scores = _routed_scores(router_scores, route_indices)
    affinity_sums = (
        np.zeros((num_experts, input_dim), dtype=np.float64)
        if flat_scores is not None
        else None
    )
    affinity_score_sums = np.zeros(num_experts, dtype=np.float64) if flat_scores is not None else None
    squared_inputs = routed_inputs.astype(np.float64) * routed_inputs.astype(np.float64)
    route_order = np.argsort(routed_experts, kind="stable")
    sorted_experts = routed_experts[route_order]
    sorted_squared_inputs = squared_inputs[route_order]
    sorted_scores = flat_scores[route_order] if flat_scores is not None else None
    counts = np.bincount(sorted_experts, minlength=num_experts).astype(np.int64, copy=False)
    segment_ends = np.cumsum(counts, dtype=np.int64)
    segment_starts = segment_ends - counts
    for expert in np.flatnonzero(counts):
        start = int(segment_starts[expert])
        end = int(segment_ends[expert])
        expert_squared_inputs = sorted_squared_inputs[start:end]
        sums[expert] = np.sum(expert_squared_inputs, axis=0)
        if affinity_sums is not None and sorted_scores is not None:
            expert_scores = sorted_scores[start:end]
            affinity_sums[expert] = np.sum(
                expert_squared_inputs * expert_scores[:, None], axis=0
            )
            if affinity_score_sums is not None:
                affinity_score_sums[expert] = float(np.sum(expert_scores))

    prompt_tuple = tuple(str(prompt_id) for prompt_id in prompt_ids)
    entries: list[ProjectionImatrixEntry] = []
    for expert in range(num_experts):
        route_count = int(counts[expert])
        importance_sum = sums[expert].astype(np.float32)
        mean_importance = (
            (sums[expert] / route_count).astype(np.float32)
            if route_count
            else np.zeros(input_dim, dtype=np.float32)
        )
        routing_weighted_importance = (sums[expert] / total_route_count).astype(np.float32)
        entries.append(
            ProjectionImatrixEntry(
                layer=layer,
                projection=projection,
                expert=expert,
                importance_sum=importance_sum,
                mean_importance=mean_importance,
                routing_weighted_importance=routing_weighted_importance,
                route_count=route_count,
                total_route_count=total_route_count,
                route_frequency=route_count / total_route_count,
                prompt_ids=prompt_tuple,
                affinity_weighted_importance=(
                    affinity_sums[expert].astype(np.float32)
                    if affinity_sums is not None
                    else None
                ),
                affinity_score_sum=(
                    float(affinity_score_sums[expert])
                    if affinity_score_sums is not None
                    else None
                ),
            )
        )
    return tuple(entries)


def merge_projection_imatrix_entries(
    entries: Iterable[ProjectionImatrixEntry],
) -> tuple[ProjectionImatrixEntry, ...]:
    """Merge per-record imatrix entries by layer, projection, and expert."""

    grouped: dict[tuple[int, str, int], list[ProjectionImatrixEntry]] = {}
    for entry in entries:
        _validate_projection(entry.projection)
        grouped.setdefault((entry.layer, entry.projection, entry.expert), []).append(entry)
    if not grouped:
        return ()

    projection_order = {projection: index for index, projection in enumerate(ROUTED_PROJECTIONS)}
    merged: list[ProjectionImatrixEntry] = []
    for (layer, projection, expert), group in grouped.items():
        input_dim = group[0].input_dim
        if any(entry.input_dim != input_dim for entry in group):
            raise ValueError(
                f"cannot merge imatrix entries with different input dims for "
                f"layer {layer} {projection} expert {expert}"
            )
        total_route_count = sum(int(entry.total_route_count) for entry in group)
        route_count = sum(int(entry.route_count) for entry in group)
        importance_sum = np.sum(
            [entry.importance_sum.astype(np.float64, copy=False) for entry in group],
            axis=0,
        )
        mean_importance = (
            importance_sum / route_count
            if route_count
            else np.zeros(input_dim, dtype=np.float64)
        )
        routing_weighted_importance = (
            importance_sum / total_route_count
            if total_route_count
            else np.zeros(input_dim, dtype=np.float64)
        )
        affinity_arrays = [
            entry.affinity_weighted_importance.astype(np.float64, copy=False)
            for entry in group
            if entry.affinity_weighted_importance is not None
        ]
        affinity_weighted_importance = (
            np.sum(affinity_arrays, axis=0).astype(np.float32)
            if affinity_arrays
            else None
        )
        affinity_score_sum = (
            sum(float(entry.affinity_score_sum or 0.0) for entry in group)
            if affinity_arrays
            else None
        )
        prompt_ids: list[str] = []
        seen_prompt_ids: set[str] = set()
        for entry in group:
            for prompt_id in entry.prompt_ids:
                if prompt_id not in seen_prompt_ids:
                    seen_prompt_ids.add(prompt_id)
                    prompt_ids.append(prompt_id)
        merged.append(
            ProjectionImatrixEntry(
                layer=layer,
                projection=projection,
                expert=expert,
                importance_sum=importance_sum.astype(np.float32),
                mean_importance=mean_importance.astype(np.float32),
                routing_weighted_importance=routing_weighted_importance.astype(np.float32),
                route_count=route_count,
                total_route_count=total_route_count,
                route_frequency=route_count / total_route_count if total_route_count else 0.0,
                prompt_ids=tuple(prompt_ids),
                affinity_weighted_importance=affinity_weighted_importance,
                affinity_score_sum=affinity_score_sum,
            )
        )
    merged.sort(
        key=lambda entry: (
            entry.layer,
            projection_order.get(entry.projection, len(projection_order)),
            entry.expert,
        )
    )
    return tuple(merged)


def _require_activation_rows(record: Mapping[str, Any], *, projection: str) -> np.ndarray:
    if projection in ("gate_proj", "up_proj"):
        key = "moe_input_rows"
        expected_rank = 2
    elif projection == "down_proj":
        key = "down_input_rows"
        expected_rank = 3
    else:
        _validate_projection(projection)
        raise AssertionError("unreachable")

    value = record.get(key)
    if value is None:
        layer = record.get("layer", "?")
        prompt_id = record.get("prompt_id", "?")
        raise ValueError(
            f"activation record {prompt_id!r} layer {layer} is missing {key}; "
            "rerun calibration with include_activation_rows=True"
        )
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != expected_rank:
        raise ValueError(
            f"{key} for projection {projection} must be rank {expected_rank}, found {array.shape}"
        )
    return array


def imatrix_entries_from_activation_records(
    records: Iterable[Mapping[str, Any]],
    *,
    projections: Sequence[str] = ROUTED_PROJECTIONS,
) -> tuple[ProjectionImatrixEntry, ...]:
    """Build merged routed projection imatrix entries from activation records.

    This is a non-materializing bridge: it consumes activation records captured
    with `include_activation_rows=True`, but does not write sidecars or mutate
    model artifacts.
    """

    projection_tuple = tuple(projections)
    if not projection_tuple:
        raise ValueError("at least one projection is required")
    for projection in projection_tuple:
        _validate_projection(projection)

    raw_entries: list[ProjectionImatrixEntry] = []
    record_count = 0
    for record in records:
        record_count += 1
        route_indices = np.asarray(record.get("route_indices"), dtype=np.int64)
        router_scores = record.get("router_scores")
        if router_scores is None:
            router_scores = record.get("scores")
        layer = int(record["layer"])
        num_experts = int(record["num_experts"])
        prompt_id = str(record.get("prompt_id", ""))
        for projection in projection_tuple:
            raw_entries.extend(
                accumulate_routed_projection_imatrix(
                    layer=layer,
                    projection=projection,
                    inputs=_require_activation_rows(record, projection=projection),
                    route_indices=route_indices,
                    router_scores=np.asarray(router_scores, dtype=np.float32) if router_scores is not None else None,
                    num_experts=num_experts,
                    prompt_ids=(prompt_id,) if prompt_id else (),
                )
            )
    if record_count == 0:
        raise ValueError("at least one activation record is required")
    return merge_projection_imatrix_entries(raw_entries)


def _entry_filename(entry: ProjectionImatrixEntry) -> str:
    return f"layer-{entry.layer:05d}-{entry.projection}-expert-{entry.expert:05d}.safetensors"


def build_air_imatrix_collection_plan(
    *,
    output_dir: str | Path,
    layers: Iterable[int],
    projections: Sequence[str] = ROUTED_PROJECTIONS,
    num_experts: int = 128,
    prompt_set: str = "air_imatrix_calib_v1",
    model_id: str = AIR_IMATRIX_DEFAULT_MODEL_ID,
    artifact_dir: str | Path = AIR_IMATRIX_DEFAULT_ARTIFACT_DIR,
    collection_source: str = "resident_vq_model",
    require_output_absent: bool = True,
) -> dict[str, Any]:
    """Plan real Air imatrix collection without touching the filesystem."""

    from mlx_vq.quality.prompts import validate_imatrix_calibration_prompt_set

    root = Path(output_dir)
    imatrix_dir = root / "imatrix"
    manifest_path = root / "imatrix-manifest.json"
    prompt_validation = validate_imatrix_calibration_prompt_set()
    errors: list[str] = list(prompt_validation.get("errors", []))

    layer_list: list[int] = []
    seen_layers: set[int] = set()
    for layer in layers:
        layer_int = int(layer)
        if layer_int < 0:
            errors.append("layers must be non-negative")
            continue
        if layer_int in seen_layers:
            errors.append(f"duplicate layer {layer_int}")
            continue
        seen_layers.add(layer_int)
        layer_list.append(layer_int)
    layer_list.sort()
    if not layer_list:
        errors.append("at least one layer is required")

    projection_list: list[str] = []
    seen_projections: set[str] = set()
    for projection in projections:
        projection_str = str(projection)
        if projection_str not in ROUTED_PROJECTIONS:
            errors.append(f"unsupported projection {projection_str!r}")
            continue
        if projection_str in seen_projections:
            errors.append(f"duplicate projection {projection_str}")
            continue
        seen_projections.add(projection_str)
        projection_list.append(projection_str)
    if not projection_list:
        errors.append("at least one projection is required")

    if num_experts <= 0:
        errors.append("num_experts must be positive")
    if prompt_set != "air_imatrix_calib_v1":
        errors.append("prompt_set must be air_imatrix_calib_v1")
    if collection_source not in AIR_IMATRIX_COLLECTION_SOURCES:
        errors.append(
            f"collection_source must be one of {list(AIR_IMATRIX_COLLECTION_SOURCES)}"
        )
    if require_output_absent and root.exists():
        errors.append("output_dir already exists")

    planned_sidecar_count = len(layer_list) * len(projection_list) * max(int(num_experts), 0)
    ok = not errors and bool(prompt_validation.get("ok"))
    return {
        "schema": "air_projection_imatrix_collection_plan",
        "schema_version": 1,
        "ok": ok,
        "errors": errors,
        "collection_allowed": False,
        "sidecar_writes_allowed": False,
        "prompt_validation": prompt_validation,
        "collection": {
            "model_id": str(model_id),
            "artifact_dir": str(artifact_dir),
            "collection_source": collection_source,
            "prompt_set": prompt_set,
            "prompt_count": int(prompt_validation.get("prompt_count", 0) or 0),
            "include_activation_rows": True,
            "use_chat_template": True,
            "layers": layer_list,
            "projections": projection_list,
            "num_experts": int(num_experts),
        },
        "filesystem": {
            "writes_performed": False,
            "output_dir": str(root),
            "output_dir_exists": root.exists(),
            "sidecar_root": str(imatrix_dir),
            "sidecar_root_created": False,
            "manifest_path": str(manifest_path),
            "manifest_written": False,
        },
        "targets": {
            "manifest_path": str(manifest_path),
            "sidecar_root": str(imatrix_dir),
            "sidecar_path_template": "imatrix/layer-{layer:05d}-{projection}-expert-{expert:05d}.safetensors",
            "manifest_record_type": "air_projection_imatrix_manifest",
        },
        "summary": {
            "planned_layer_count": len(layer_list),
            "planned_projection_count": len(projection_list),
            "planned_expert_count": int(num_experts),
            "planned_sidecar_count": planned_sidecar_count,
            "approval_required": True,
        },
        "approval_request": {
            "approval_required": True,
            "approval_gate": "real_air_imatrix_calib_v1_sidecar_collection",
            "ready": ok,
            "requested_operations": [
                "run_air_imatrix_calib_v1_activation_capture",
                "accumulate_routed_projection_imatrix_entries",
                "write_projection_imatrix_sidecars",
                "validate_air_projection_imatrix_manifest",
            ],
        },
        "blocked_operations": [
            "real_activation_capture",
            "real_imatrix_sidecar_writes",
            "imatrix_manifest_write",
        ],
    }


def write_projection_imatrix_sidecars(
    entries: Iterable[ProjectionImatrixEntry],
    *,
    output_dir: str | Path,
    prompt_set: str,
    manifest_name: str = "imatrix-manifest.json",
) -> dict[str, object]:
    """Write synthetic or approved imatrix sidecars plus a JSON manifest."""

    entry_list = list(entries)
    if not entry_list:
        raise ValueError("at least one imatrix entry is required")
    root = Path(output_dir)
    imatrix_dir = root / "imatrix"
    imatrix_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, object]] = []
    for entry in entry_list:
        filename = _entry_filename(entry)
        path = imatrix_dir / filename
        arrays = {
            "importance_sum": mx.array(entry.importance_sum.astype(np.float32, copy=False)),
            "mean_importance": mx.array(entry.mean_importance.astype(np.float32, copy=False)),
            "routing_weighted_importance": mx.array(
                entry.routing_weighted_importance.astype(np.float32, copy=False)
            ),
        }
        if entry.affinity_weighted_importance is not None:
            arrays["affinity_weighted_importance"] = mx.array(
                entry.affinity_weighted_importance.astype(np.float32, copy=False)
            )
            affinity_mean = entry.affinity_weighted_mean_importance
            if affinity_mean is not None:
                arrays["affinity_weighted_mean_importance"] = mx.array(
                    affinity_mean.astype(np.float32, copy=False)
                )
        metadata = {
            "layer": str(entry.layer),
            "projection": entry.projection,
            "expert": str(entry.expert),
            "route_count": str(entry.route_count),
            "total_route_count": str(entry.total_route_count),
            "route_frequency": repr(entry.route_frequency),
            "prompt_set": prompt_set,
            "affinity_weighted": "true" if entry.affinity_weighted_importance is not None else "false",
            "affinity_score_sum": repr(float(entry.affinity_score_sum or 0.0)),
        }
        mx.save_safetensors(str(path), arrays, metadata=metadata)
        manifest_entries.append(entry.manifest_entry(path=f"imatrix/{filename}"))

    manifest = {
        "schema_version": 1,
        "record_type": "air_projection_imatrix_manifest",
        "prompt_set": prompt_set,
        "entry_count": len(manifest_entries),
        "entries": manifest_entries,
        "notes": [
            "importance_sum follows llama.cpp imatrix sum(activation^2) semantics.",
            "routing_weighted_importance is importance_sum divided by total routed rows for global allocation.",
            "affinity_weighted_importance, when present, is sum(router_score * activation^2) for AGQ.",
            "affinity_weighted_mean_importance, when present, normalizes AGQ affinity by router-score mass.",
        ],
    }
    (root / manifest_name).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_projection_imatrix_manifest(manifest_path: str | Path) -> dict[str, object]:
    """Load and lightly validate an imatrix sidecar manifest JSON file."""

    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("imatrix manifest must be a JSON object")
    if manifest.get("record_type") != "air_projection_imatrix_manifest":
        raise ValueError("imatrix manifest record_type must be air_projection_imatrix_manifest")
    if manifest.get("schema_version") != 1:
        raise ValueError("imatrix manifest schema_version must be 1")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("imatrix manifest entries must be a list")
    if manifest.get("entry_count") != len(entries):
        raise ValueError("imatrix manifest entry_count must match entries length")
    return manifest


__all__ = [
    "AIR_IMATRIX_COLLECTION_SOURCES",
    "AIR_IMATRIX_DEFAULT_ARTIFACT_DIR",
    "AIR_IMATRIX_DEFAULT_MODEL_ID",
    "ProjectionImatrixEntry",
    "ROUTED_PROJECTIONS",
    "accumulate_routed_projection_imatrix",
    "build_air_imatrix_collection_plan",
    "imatrix_entries_from_activation_records",
    "load_projection_imatrix_manifest",
    "merge_projection_imatrix_entries",
    "write_projection_imatrix_sidecars",
]
