from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_METRIC_NAMES = (
    "source_gate_proj",
    "source_up_proj",
    "source_routed_glu",
    "source_weighted_routed",
)
ROUTE_SOURCE_METRIC_NAMES = (
    "source_weighted_route_contribution",
    "source_routed_glu",
    "source_up_proj",
    "source_gate_proj",
)
STATE_BUNDLE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LayerProbeTarget:
    prompt_id: str
    position: int

    @property
    def key(self) -> str:
        return f"{self.prompt_id}:{self.position}"


def parse_layer_spec(value: str) -> tuple[int, ...]:
    layers: list[int] = []
    seen: set[int] = set()
    for chunk in value.split(","):
        stripped = chunk.strip()
        if not stripped:
            continue
        if "-" in stripped:
            start_text, end_text = stripped.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"invalid descending layer range {stripped!r}")
            candidates = range(start, end + 1)
        else:
            candidates = (int(stripped),)
        for layer in candidates:
            if layer < 0:
                raise ValueError("layers must be non-negative")
            if layer not in seen:
                seen.add(layer)
                layers.append(layer)
    if not layers:
        raise ValueError("at least one layer is required")
    return tuple(layers)


def parse_probe_target(value: str) -> LayerProbeTarget:
    if ":" not in value:
        raise ValueError("target must use PROMPT_ID:POSITION")
    prompt_id, position_text = value.split(":", 1)
    if not prompt_id:
        raise ValueError("target prompt id must be non-empty")
    position = int(position_text)
    if position < 0:
        raise ValueError("target position must be non-negative")
    return LayerProbeTarget(prompt_id=prompt_id, position=position)


def write_layer_probe_state_bundle(
    path: str | Path,
    *,
    prompt_id: str,
    input_token_ids: list[int] | tuple[int, ...],
    captures: dict[int, np.ndarray],
) -> None:
    if not prompt_id:
        raise ValueError("prompt_id must be non-empty")
    token_ids = [int(token_id) for token_id in input_token_ids]
    if not token_ids:
        raise ValueError("input_token_ids must not be empty")
    if not captures:
        raise ValueError("captures must not be empty")

    arrays: dict[str, np.ndarray] = {}
    layer_metadata = []
    for layer in sorted(captures):
        layer_id = int(layer)
        if layer_id < 0:
            raise ValueError("capture layers must be non-negative")
        states = np.asarray(captures[layer], dtype=np.float32)
        if states.ndim != 2:
            raise ValueError(f"capture for layer {layer_id} must have shape [tokens, hidden]")
        if states.shape[0] != len(token_ids):
            raise ValueError(
                f"capture for layer {layer_id} has {states.shape[0]} token rows; "
                f"expected {len(token_ids)}"
            )
        key = _state_bundle_layer_key(layer_id)
        arrays[key] = states
        layer_metadata.append(
            {
                "layer": layer_id,
                "key": key,
                "shape": list(states.shape),
                "dtype": "float32",
            }
        )

    metadata = {
        "schema_version": STATE_BUNDLE_SCHEMA_VERSION,
        "capture_state_source": "resident_vq_hidden_states",
        "prompt_id": prompt_id,
        "input_token_ids": token_ids,
        "layers": layer_metadata,
    }
    arrays["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


def load_layer_probe_state_bundle(
    path: str | Path,
    *,
    prompt_id: str,
    input_token_ids: list[int] | tuple[int, ...],
    layers: tuple[int, ...],
) -> dict[int, np.ndarray]:
    requested_layers = tuple(int(layer) for layer in layers)
    if not requested_layers:
        raise ValueError("at least one layer is required")
    token_ids = [int(token_id) for token_id in input_token_ids]
    with np.load(Path(path), allow_pickle=False) as bundle:
        if "metadata_json" not in bundle:
            raise ValueError("state bundle is missing metadata_json")
        metadata = json.loads(str(bundle["metadata_json"].item()))
        if int(metadata.get("schema_version", -1)) != STATE_BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported state bundle schema_version {metadata.get('schema_version')!r}"
            )
        if metadata.get("prompt_id") != prompt_id:
            raise ValueError(
                f"state bundle prompt_id {metadata.get('prompt_id')!r} does not match "
                f"{prompt_id!r}"
            )
        if [int(token_id) for token_id in metadata.get("input_token_ids", [])] != token_ids:
            raise ValueError("state bundle input_token_ids do not match requested prompt")

        layer_keys = {
            int(row["layer"]): str(row["key"])
            for row in metadata.get("layers", [])
        }
        captures: dict[int, np.ndarray] = {}
        for layer in requested_layers:
            key = layer_keys.get(layer)
            if key is None or key not in bundle:
                raise ValueError(f"state bundle is missing requested layer {layer}")
            states = np.asarray(bundle[key], dtype=np.float32)
            if states.ndim != 2:
                raise ValueError(f"state bundle layer {layer} must have shape [tokens, hidden]")
            if states.shape[0] != len(token_ids):
                raise ValueError(
                    f"state bundle layer {layer} has {states.shape[0]} token rows; "
                    f"expected {len(token_ids)}"
                )
            captures[layer] = states
    return captures


def summarize_layer_probe_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    target_count = len({
        record["probe_target"]["key"]
        for record in records
    })
    layer_records: dict[int, list[dict[str, Any]]] = {}
    projection_records: dict[tuple[int, str], list[dict[str, Any]]] = {}
    target_records: dict[str, list[dict[str, Any]]] = {}
    enriched_records = []

    for record in records:
        layer = int(record["layer"])
        target_key = str(record["probe_target"]["key"])
        derived = _derived_record(record)
        enriched = dict(record)
        enriched["derived"] = derived
        enriched_records.append(enriched)
        layer_records.setdefault(layer, []).append(enriched)
        target_records.setdefault(target_key, []).append(enriched)
        for metric_name in SOURCE_METRIC_NAMES:
            projection_records.setdefault((layer, metric_name), []).append(enriched)

    layer_rankings = [
        _layer_summary(layer, rows)
        for layer, rows in layer_records.items()
    ]
    layer_rankings.sort(
        key=lambda row: (
            -row["mean_source_weighted_rel_l2"],
            -row["max_source_weighted_rel_l2"],
            row["layer"],
        )
    )
    for rank, row in enumerate(layer_rankings, start=1):
        row["rank"] = rank

    layer_projection_rankings = [
        _projection_summary(layer, metric_name, rows)
        for (layer, metric_name), rows in projection_records.items()
    ]
    layer_projection_rankings.sort(
        key=lambda row: (
            -row["mean_rel_l2"],
            -row["max_rel_l2"],
            row["layer"],
            row["metric"],
        )
    )
    for rank, row in enumerate(layer_projection_rankings, start=1):
        row["rank"] = rank

    target_rankings = [
        _target_summary(target_key, rows)
        for target_key, rows in target_records.items()
    ]
    target_rankings.sort(
        key=lambda row: (
            -row["max_source_weighted_rel_l2"],
            -row["mean_source_weighted_rel_l2"],
            row["target_key"],
        )
    )

    return {
        "record_count": len(records),
        "target_count": target_count,
        "layer_count": len(layer_records),
        "layer_rankings": layer_rankings,
        "layer_projection_rankings": layer_projection_rankings,
        "target_rankings": target_rankings,
        "route_source_rankings": _route_source_rankings(enriched_records),
        "route_source_residual_rankings": _route_source_residual_rankings(enriched_records),
        "route_source_sparse_residual_plan_rankings": _route_source_sparse_residual_plan_rankings(enriched_records),
        "records": enriched_records,
    }


def build_sparse_residual_rows_from_plan_report(
    report: dict[str, Any],
    *,
    target_key: str,
    expert: int,
    route_rank: int,
    projection: str = "down_proj",
    max_rows: int | None = None,
) -> dict[str, Any]:
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when provided")
    records = ((report.get("summary") or {}).get("records") or [])
    for record in records:
        if str((record.get("probe_target") or {}).get("key")) != target_key:
            continue
        for plan in record.get("route_source_sparse_residual_plans", []) or []:
            if int(plan.get("expert", -1)) != int(expert):
                continue
            if int(plan.get("route_rank", -1)) != int(route_rank):
                continue
            if str(plan.get("projection")) != projection:
                continue
            rows = list(plan.get("rows", []) or [])
            rows.sort(
                key=lambda row: (
                    -abs(float(row["desired_weighted_correction"])),
                    int(row["output_index"]),
                )
            )
            if max_rows is not None:
                rows = rows[:max_rows]
            if not rows:
                raise ValueError("matched sparse residual plan does not include any rows")
            values = np.asarray([row["residual_values"] for row in rows], dtype=np.float32)
            if values.ndim != 2 or values.shape[0] != len(rows):
                raise ValueError("sparse residual plan values must form a 2D array")
            output_indices = np.asarray([int(row["output_index"]) for row in rows], dtype=np.int32)
            expert_indices = np.full((len(rows),), int(expert), dtype=np.int32)
            return {
                "layer": int(record["layer"]),
                "target_key": target_key,
                "token_index": int(plan["token_index"]),
                "route_rank": int(route_rank),
                "expert": int(expert),
                "projection": projection,
                "source_metric": str(plan["source_metric"]),
                "expert_indices": expert_indices,
                "output_indices": output_indices,
                "values": values,
                "selected_rows": rows,
            }
    raise ValueError(
        f"no sparse residual plan found for target={target_key!r} expert={expert} "
        f"route_rank={route_rank} projection={projection!r}"
    )


def _state_bundle_layer_key(layer: int) -> str:
    return f"layer_{layer}"


def _route_source_rankings(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        layer = int(record["layer"])
        target_key = str(record["probe_target"]["key"])
        for route in record.get("route_source_metrics", []) or []:
            route_metrics = route.get("metrics", {})
            for metric_name in ROUTE_SOURCE_METRIC_NAMES:
                metric = route_metrics.get(metric_name)
                if not isinstance(metric, dict):
                    continue
                rows.append(
                    {
                        "layer": layer,
                        "target_key": target_key,
                        "token_index": int(route["token_index"]),
                        "route_rank": int(route["route_rank"]),
                        "expert": int(route["expert"]),
                        "router_score": float(route["router_score"]),
                        "metric": metric_name,
                        "rel_l2": float(metric["rel_l2"]),
                        "cosine": float(metric["cosine"]),
                        "max_abs": float(metric["max_abs"]),
                        "mean_abs": float(metric["mean_abs"]),
                    }
                )

    def sort_key(row: dict[str, Any]) -> tuple[float, float, int, int]:
        metric_priority = ROUTE_SOURCE_METRIC_NAMES.index(str(row["metric"]))
        if row["metric"] == "source_weighted_route_contribution":
            primary = float(row["mean_abs"])
            secondary = float(row["rel_l2"])
        else:
            primary = float(row["rel_l2"])
            secondary = float(row["mean_abs"])
        return (metric_priority, -primary, -secondary, int(row["route_rank"]), int(row["expert"]))

    rows.sort(key=sort_key)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _route_source_residual_rankings(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        layer = int(record["layer"])
        target_key = str(record["probe_target"]["key"])
        for route in record.get("route_source_residual_topk", []) or []:
            residuals = route.get("top_abs_residuals", {})
            for metric_name in ROUTE_SOURCE_METRIC_NAMES:
                entries = residuals.get(metric_name)
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    rows.append(
                        {
                            "layer": layer,
                            "target_key": target_key,
                            "token_index": int(route["token_index"]),
                            "route_rank": int(route["route_rank"]),
                            "expert": int(route["expert"]),
                            "router_score": float(route["router_score"]),
                            "metric": metric_name,
                            "coordinate_index": int(entry["index"]),
                            "actual": float(entry["actual"]),
                            "expected": float(entry["expected"]),
                            "source_minus_actual": float(entry["source_minus_actual"]),
                            "abs_source_minus_actual": float(entry["abs_source_minus_actual"]),
                        }
                    )

    def sort_key(row: dict[str, Any]) -> tuple[float, float, int, int, int]:
        metric_priority = ROUTE_SOURCE_METRIC_NAMES.index(str(row["metric"]))
        return (
            metric_priority,
            -float(row["abs_source_minus_actual"]),
            int(row["route_rank"]),
            int(row["expert"]),
            int(row["coordinate_index"]),
        )

    rows.sort(key=sort_key)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _route_source_sparse_residual_plan_rankings(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        layer = int(record["layer"])
        target_key = str(record["probe_target"]["key"])
        for plan in record.get("route_source_sparse_residual_plans", []) or []:
            for row in plan.get("rows", []) or []:
                if not isinstance(row, dict):
                    continue
                rows.append(
                    {
                        "layer": layer,
                        "target_key": target_key,
                        "token_index": int(plan["token_index"]),
                        "route_rank": int(plan["route_rank"]),
                        "expert": int(plan["expert"]),
                        "router_score": float(plan["router_score"]),
                        "projection": str(plan["projection"]),
                        "source_metric": str(plan["source_metric"]),
                        "output_index": int(row["output_index"]),
                        "desired_weighted_correction": float(row["desired_weighted_correction"]),
                        "desired_unweighted_correction": float(row["desired_unweighted_correction"]),
                        "input_norm_sq": float(row["input_norm_sq"]),
                        "residual_value_norm": float(row["residual_value_norm"]),
                        "reconstruction_abs_error": float(row["reconstruction_abs_error"]),
                    }
                )

    rows.sort(
        key=lambda row: (
            -abs(float(row["desired_weighted_correction"])),
            int(row["route_rank"]),
            int(row["expert"]),
            int(row["output_index"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _derived_record(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record["metrics"]
    projection_rel_l2 = {
        metric_name: float(metrics[metric_name]["rel_l2"])
        for metric_name in SOURCE_METRIC_NAMES
    }
    projection_cosine_loss = {
        metric_name: 1.0 - float(metrics[metric_name]["cosine"])
        for metric_name in SOURCE_METRIC_NAMES
    }
    projection_candidates = [
        name for name in SOURCE_METRIC_NAMES
        if name != "source_weighted_routed"
    ]
    dominant_projection = max(
        projection_candidates,
        key=lambda name: projection_rel_l2[name],
    )
    return {
        "projection_rel_l2": projection_rel_l2,
        "projection_cosine_loss": projection_cosine_loss,
        "source_weighted_rel_l2": projection_rel_l2["source_weighted_routed"],
        "source_weighted_cosine_loss": projection_cosine_loss["source_weighted_routed"],
        "dominant_projection": dominant_projection,
        "dominant_projection_rel_l2": projection_rel_l2[dominant_projection],
    }


def _layer_summary(layer: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    weighted_values = [
        float(row["derived"]["source_weighted_rel_l2"])
        for row in rows
    ]
    dominant_counts: dict[str, int] = {}
    selected_experts: set[int] = set()
    for row in rows:
        dominant = str(row["derived"]["dominant_projection"])
        dominant_counts[dominant] = dominant_counts.get(dominant, 0) + 1
        selected_experts.update(int(expert) for expert in row.get("selected_experts", []))
    return {
        "layer": layer,
        "record_count": len(rows),
        "mean_source_weighted_rel_l2": _mean(weighted_values),
        "max_source_weighted_rel_l2": max(weighted_values) if weighted_values else 0.0,
        "dominant_projection_counts": dominant_counts,
        "selected_experts": sorted(selected_experts),
    }


def _projection_summary(
    layer: int,
    metric_name: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    values = [
        float(row["derived"]["projection_rel_l2"][metric_name])
        for row in rows
    ]
    return {
        "layer": layer,
        "metric": metric_name,
        "record_count": len(rows),
        "mean_rel_l2": _mean(values),
        "max_rel_l2": max(values) if values else 0.0,
    }


def _target_summary(target_key: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    weighted_values = [
        float(row["derived"]["source_weighted_rel_l2"])
        for row in rows
    ]
    worst = max(
        rows,
        key=lambda row: float(row["derived"]["source_weighted_rel_l2"]),
    )
    return {
        "target_key": target_key,
        "record_count": len(rows),
        "mean_source_weighted_rel_l2": _mean(weighted_values),
        "max_source_weighted_rel_l2": max(weighted_values) if weighted_values else 0.0,
        "worst_layer": int(worst["layer"]),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
