from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from keep.quality.imatrix import (
    ProjectionImatrixEntry,
    accumulate_routed_projection_imatrix,
    build_air_imatrix_collection_plan,
    imatrix_entries_from_activation_records,
    merge_projection_imatrix_entries,
    write_projection_imatrix_sidecars,
)


GLU_METRIC = "source_routed_glu"
SOURCE_METRIC_TO_PROJECTIONS = {
    "source_gate_proj": ("gate_proj",),
    "source_up_proj": ("up_proj",),
    "source_down_proj": ("down_proj",),
    GLU_METRIC: ("gate_proj", "up_proj"),
}


@dataclass(frozen=True)
class LayerCalibrationSummary:
    layer: int
    record_count: int
    context_tokens: int
    route_count: int
    num_experts: int
    selected_experts: tuple[int, ...]
    moe_input_rms: float
    down_input_rms: float

    @property
    def coverage_fraction(self) -> float:
        if self.num_experts <= 0:
            return 0.0
        return len(self.selected_experts) / self.num_experts

    def activation_rms_for_metric(self, metric: str) -> float:
        if metric == "source_down_proj":
            return self.down_input_rms
        return self.moe_input_rms

    def to_dict(self, *, metric: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "layer": self.layer,
            "record_count": self.record_count,
            "context_tokens": self.context_tokens,
            "route_count": self.route_count,
            "num_experts": self.num_experts,
            "selected_expert_count": len(self.selected_experts),
            "selected_experts": list(self.selected_experts),
            "coverage_fraction": self.coverage_fraction,
            "moe_input_rms": self.moe_input_rms,
            "down_input_rms": self.down_input_rms,
        }
        if metric is not None:
            payload["activation_rms"] = self.activation_rms_for_metric(metric)
        return payload


def read_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _weighted_rms(records: Iterable[Mapping[str, Any]], field: str) -> float:
    weighted_square = 0.0
    weight_sum = 0
    for record in records:
        summary = record.get(field)
        if not isinstance(summary, Mapping):
            continue
        rms = float(summary.get("rms", 0.0) or 0.0)
        weight = int(record.get("context_tokens", 0) or 0)
        if weight <= 0:
            continue
        weighted_square += rms * rms * weight
        weight_sum += weight
    if weight_sum <= 0:
        return 0.0
    return (weighted_square / weight_sum) ** 0.5


def summarize_calibration_by_layer(
    records: Iterable[Mapping[str, Any]],
) -> dict[int, LayerCalibrationSummary]:
    by_layer: dict[int, list[Mapping[str, Any]]] = {}
    for record in records:
        by_layer.setdefault(int(record["layer"]), []).append(record)

    summaries: dict[int, LayerCalibrationSummary] = {}
    for layer, layer_records in by_layer.items():
        selected_experts: set[int] = set()
        num_experts = 0
        for record in layer_records:
            selected_experts.update(int(expert) for expert in record.get("selected_experts", []))
            num_experts = max(num_experts, int(record.get("num_experts", 0) or 0))
        summaries[layer] = LayerCalibrationSummary(
            layer=layer,
            record_count=len(layer_records),
            context_tokens=sum(int(record.get("context_tokens", 0) or 0) for record in layer_records),
            route_count=sum(int(record.get("route_count", 0) or 0) for record in layer_records),
            num_experts=num_experts,
            selected_experts=tuple(sorted(selected_experts)),
            moe_input_rms=_weighted_rms(layer_records, "moe_input"),
            down_input_rms=_weighted_rms(layer_records, "down_input"),
        )
    return summaries


def _empty_layer_summary(layer: int) -> LayerCalibrationSummary:
    return LayerCalibrationSummary(
        layer=layer,
        record_count=0,
        context_tokens=0,
        route_count=0,
        num_experts=0,
        selected_experts=(),
        moe_input_rms=0.0,
        down_input_rms=0.0,
    )


def _precision_groups_for_metric(layer: int, metric: str) -> tuple[str, ...]:
    projections = SOURCE_METRIC_TO_PROJECTIONS[metric]
    return tuple(f"{layer}:{projection}" for projection in projections)


def _candidate_from_groups(
    *,
    name: str,
    description: str,
    groups: Iterable[str],
    rationale: Iterable[str],
) -> dict[str, Any]:
    policy = {group: 16 for group in dict.fromkeys(groups)}
    return {
        "name": name,
        "description": description,
        "code_bits_policy": policy,
        "rationale": list(rationale),
    }


def _candidate_layers_from_rankings(
    rankings: list[dict[str, Any]],
    *,
    top_candidate_layers: int,
) -> tuple[int, ...]:
    layers: list[int] = []
    for row in rankings:
        if row["metric"] != GLU_METRIC or row["score"] <= 0.0:
            continue
        layer = int(row["layer"])
        if layer not in layers:
            layers.append(layer)
        if len(layers) >= top_candidate_layers:
            break
    return tuple(layers)


def build_calibration_importance_report(
    *,
    source_probe_report: Mapping[str, Any],
    calibration_records: Iterable[Mapping[str, Any]],
    top_candidate_layers: int = 3,
) -> dict[str, Any]:
    if top_candidate_layers <= 0:
        raise ValueError("top_candidate_layers must be positive")

    records = list(calibration_records)
    calibration_by_layer = summarize_calibration_by_layer(records)
    source_rows = list(source_probe_report.get("summary", {}).get("layer_projection_rankings", []))
    skipped_metrics = sorted({
        str(row.get("metric", ""))
        for row in source_rows
        if str(row.get("metric", "")) not in SOURCE_METRIC_TO_PROJECTIONS
    })

    group_rankings: list[dict[str, Any]] = []
    for row in source_rows:
        metric = str(row.get("metric", ""))
        if metric not in SOURCE_METRIC_TO_PROJECTIONS:
            continue
        layer = int(row["layer"])
        calibration = calibration_by_layer.get(layer, _empty_layer_summary(layer))
        mean_rel_l2 = float(row.get("mean_rel_l2", 0.0) or 0.0)
        activation_rms = calibration.activation_rms_for_metric(metric)
        coverage = calibration.coverage_fraction
        score = mean_rel_l2 * activation_rms * coverage
        group_rankings.append(
            {
                "layer": layer,
                "metric": metric,
                "mean_rel_l2": mean_rel_l2,
                "max_rel_l2": float(row.get("max_rel_l2", 0.0) or 0.0),
                "source_record_count": int(row.get("record_count", 0) or 0),
                "precision_groups": list(_precision_groups_for_metric(layer, metric)),
                "calibration": calibration.to_dict(metric=metric),
                "score": score,
                "score_formula": "mean_rel_l2 * activation_rms * coverage_fraction",
            }
        )
    group_rankings.sort(
        key=lambda row: (
            -float(row["score"]),
            -float(row["mean_rel_l2"]),
            int(row["layer"]),
            str(row["metric"]),
        )
    )
    for rank, row in enumerate(group_rankings, start=1):
        row["rank"] = rank

    candidate_layers = _candidate_layers_from_rankings(
        group_rankings,
        top_candidate_layers=top_candidate_layers,
    )
    candidates: list[dict[str, Any]] = []
    if candidate_layers:
        groups = [
            group
            for layer in candidate_layers
            for group in _precision_groups_for_metric(layer, GLU_METRIC)
        ]
        candidates.append(
            _candidate_from_groups(
                name=f"lane3-calib-top{len(candidate_layers)}-glu",
                description=(
                    "Calibration-importance GLU candidate: protect the highest-scored "
                    "source-error layers after weighting by activation RMS and expert coverage."
                ),
                groups=groups,
                rationale=(
                    f"Selected GLU layers: {','.join(str(layer) for layer in candidate_layers)}.",
                    "Routed-GLU source-error rows map to gate/up projection protection.",
                    "Scores are diagnostic n=5 allocation evidence until Lane 0 widened authority exists.",
                ),
            )
        )

    return {
        "schema_version": 1,
        "evidence_scope": "calibration_importance_selective_precision_planning",
        "calibration_record_count": len(records),
        "calibration_layer_count": len(calibration_by_layer),
        "calibration_layers": [
            summary.to_dict()
            for _, summary in sorted(calibration_by_layer.items())
        ],
        "source_projection_row_count": len(source_rows),
        "skipped_source_metrics": skipped_metrics,
        "score_formula": "mean_rel_l2 * activation_rms * coverage_fraction",
        "projection_metric_mapping": {
            metric: list(projections)
            for metric, projections in SOURCE_METRIC_TO_PROJECTIONS.items()
        },
        "group_rankings": group_rankings,
        "candidates": candidates,
        "notes": [
            "This report is allocation evidence only; it does not accept or rank a final artifact.",
            "Activation RMS is pooled across calibration prompts by context-token count.",
            "Expert coverage is the union of selected experts across calibration records for the layer.",
        ],
    }


__all__ = [
    "GLU_METRIC",
    "LayerCalibrationSummary",
    "ProjectionImatrixEntry",
    "SOURCE_METRIC_TO_PROJECTIONS",
    "accumulate_routed_projection_imatrix",
    "build_air_imatrix_collection_plan",
    "build_calibration_importance_report",
    "imatrix_entries_from_activation_records",
    "merge_projection_imatrix_entries",
    "read_jsonl_records",
    "summarize_calibration_by_layer",
    "write_projection_imatrix_sidecars",
]
