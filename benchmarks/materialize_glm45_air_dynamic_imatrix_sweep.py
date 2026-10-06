from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open

from mlx_vq.convert.inspect_hf import GLM45_AIR_MODEL_ID
from mlx_vq.convert.stream_convert import load_safetensors_index
from mlx_vq.io.load import (
    infer_vq_switch_linear_dims,
    inspect_safetensors,
    load_quantized_vq_switch_linear,
)
from mlx_vq.quality.dynamic_precision import (
    DynamicTensorProfile,
    allocate_dynamic_precision_tiers,
    build_dynamic_precision_tier_map_report,
)
from mlx_vq.quality.hessian_rounding import imatrix_weighted_reassign_codes
from mlx_vq.quality.imatrix import load_projection_imatrix_manifest
from mlx_vq.validate.glm45_air_vq import _expert_weight_name, _read_named_tensor


ROUTED_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
DEFAULT_BASELINE_DIR = "artifacts/glm-4.5-air-vq"
DEFAULT_HIGH_BIT_DIR = "artifacts/glm-4.5-air-vq2-e8p-rtn-uniform-parallel8"
IMPORTANCE_KEYS = (
    "routing_weighted_importance",
    "affinity_weighted_importance",
    "affinity_weighted_mean_importance",
)


def _switch_prefix(layer: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.switch_mlp.{projection}"


def _group_filename(layer: int, projection: str) -> str:
    return f"layer-{layer:05d}-{projection}.safetensors"


def _group_key(layer: int, projection: str) -> str:
    return f"{int(layer)}:{projection}"


def _relative_symlink_target(source_path: Path, target_parent: Path) -> str:
    return os.path.relpath(source_path.resolve(), start=target_parent.resolve())


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_metadata(path: Path) -> dict[str, str]:
    with safe_open(path, framework="np") as handle:
        return dict(handle.metadata() or {})


def _parse_budgets(values: list[str] | None) -> tuple[float, ...]:
    if not values:
        return (2.0, 2.4, 3.0)
    budgets = tuple(float(value) for value in values)
    if any(value <= 0 for value in budgets):
        raise argparse.ArgumentTypeError("budgets must be positive")
    return budgets


def _load_sidecar_sum(manifest_root: Path, entry: Mapping[str, Any], *, importance_key: str) -> float:
    sidecar_path = manifest_root / str(entry["path"])
    arrays = mx.load(str(sidecar_path))
    if importance_key not in arrays:
        raise ValueError(f"{sidecar_path} is missing requested importance tensor {importance_key!r}")
    importance = np.asarray(arrays[importance_key], dtype=np.float64)
    if not np.isfinite(importance).all() or np.any(importance < 0.0):
        raise ValueError(f"invalid imatrix importance values in {sidecar_path}")
    return float(np.sum(importance))


def _summarize_imatrix_importance(
    manifest: Mapping[str, Any],
    *,
    manifest_root: Path,
    importance_key: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if importance_key not in IMPORTANCE_KEYS:
        raise ValueError(f"importance_key must be one of {IMPORTANCE_KEYS}, got {importance_key!r}")
    group_stats: dict[str, dict[str, Any]] = {}
    expert_scores: list[dict[str, Any]] = []
    for entry in manifest.get("entries", []):
        layer = int(entry["layer"])
        projection = str(entry["projection"])
        expert = int(entry["expert"])
        if projection not in ROUTED_PROJECTIONS:
            raise ValueError(f"unsupported projection in imatrix manifest: {projection!r}")
        score = _load_sidecar_sum(manifest_root, entry, importance_key=importance_key)
        key = _group_key(layer, projection)
        group = group_stats.setdefault(
            key,
            {
                "layer": layer,
                "projection": projection,
                "importance_score": 0.0,
                "route_count": 0,
                "total_route_count": 0,
                "expert_count": 0,
            },
        )
        group["importance_score"] += score
        group["route_count"] += int(entry["route_count"])
        group["total_route_count"] += int(entry["total_route_count"])
        group["expert_count"] += 1
        expert_scores.append(
            {
                "key": key,
                "layer": layer,
                "projection": projection,
                "expert": expert,
                "importance_score": score,
                "importance_key": importance_key,
                "sidecar_path": str(entry["path"]),
                "route_count": int(entry["route_count"]),
            }
        )
    if not group_stats:
        raise ValueError("imatrix manifest had no entries")
    expert_scores.sort(
        key=lambda row: (-float(row["importance_score"]), int(row["layer"]), str(row["projection"]), int(row["expert"]))
    )
    return group_stats, expert_scores


def _group_weight_count(artifact_dir: Path, *, layer: int, projection: str) -> int:
    path = artifact_dir / _group_filename(layer, projection)
    inspection = inspect_safetensors(path)
    in_dim, out_dim, num_experts, _group_size, _code_bits = infer_vq_switch_linear_dims(
        inspection,
        _switch_prefix(layer, projection),
    )
    return int(in_dim * out_dim * num_experts)


def _build_profiles(
    group_stats: Mapping[str, Mapping[str, Any]],
    *,
    baseline_dir: Path,
    mid_error_factor: float,
) -> tuple[DynamicTensorProfile, ...]:
    if not (0.0 <= mid_error_factor <= 1.0):
        raise ValueError("--mid-error-factor must be in [0, 1]")
    profiles: list[DynamicTensorProfile] = []
    for key in sorted(group_stats, key=lambda item: (int(item.split(":", 1)[0]), item.split(":", 1)[1])):
        stat = group_stats[key]
        layer = int(stat["layer"])
        projection = str(stat["projection"])
        weight_count = _group_weight_count(baseline_dir, layer=layer, projection=projection)
        low_error = max(float(stat["importance_score"]), 0.0)
        profiles.append(
            DynamicTensorProfile(
                key=key,
                layer=layer,
                projection=projection,
                weight_count=weight_count,
                weighted_errors={
                    "low": low_error,
                    "mid": low_error * mid_error_factor,
                    "high": low_error * mid_error_factor,
                },
            )
        )
    return tuple(profiles)


def _materialize_links(
    *,
    tier_map: Mapping[str, str],
    baseline_dir: Path,
    high_bit_dir: Path,
    output_dir: Path,
    allow_existing: bool,
) -> dict[str, Any]:
    if output_dir.exists() and not allow_existing:
        raise FileExistsError(f"{output_dir} already exists")
    output_dir.mkdir(parents=True, exist_ok=allow_existing)
    links: list[dict[str, Any]] = []
    tier_counts = {"low": 0, "mid": 0, "high": 0}
    for baseline_path in sorted(baseline_dir.glob("layer-*.safetensors")):
        stem = baseline_path.name.removesuffix(".safetensors")
        try:
            _prefix, layer_text, projection = stem.split("-", 2)
        except ValueError:
            continue
        if projection not in ROUTED_PROJECTIONS:
            continue
        layer = int(layer_text)
        key = _group_key(layer, projection)
        tier = str(tier_map.get(key, "low"))
        if tier == "high":
            raise ValueError("projection-level high tier materialization is not supported by this sweep")
        source_dir = high_bit_dir if tier == "mid" else baseline_dir
        source_path = source_dir / baseline_path.name
        if not source_path.exists():
            raise FileNotFoundError(f"missing source group {source_path}")
        target_path = output_dir / baseline_path.name
        if target_path.exists() or target_path.is_symlink():
            if not allow_existing:
                raise FileExistsError(f"{target_path} already exists")
            target_path.unlink()
        os.symlink(_relative_symlink_target(source_path, target_path.parent), target_path)
        tier_counts[tier] += 1
        links.append(
            {
                "key": key,
                "tier": tier,
                "source": "high_bit_e8p" if tier == "mid" else "baseline_e8",
                "path": baseline_path.name,
            }
        )
    return {
        "linked_group_count": len(links),
        "tier_counts": tier_counts,
        "links": links,
    }


def _resolve_source_root(
    *,
    model_id: str,
    revision: str,
    source_dir: str | None,
) -> Path:
    if source_dir is not None:
        return Path(source_dir)
    return Path(snapshot_download(repo_id=model_id, revision=revision, local_files_only=True))


def _load_imatrix_importance(manifest_root: Path, sidecar_path: str, *, importance_key: str) -> np.ndarray:
    arrays = mx.load(str(manifest_root / sidecar_path))
    if importance_key not in arrays:
        raise ValueError(f"{manifest_root / sidecar_path} is missing requested importance tensor {importance_key!r}")
    return np.asarray(arrays[importance_key], dtype=np.float32)


def _apply_imatrix_rerounds(
    *,
    selected_experts: list[dict[str, Any]],
    output_dir: Path,
    baseline_dir: Path,
    manifest_root: Path,
    source_root: Path,
    index_path: Path,
    model_id: str,
    codeword_chunk_size: int,
    importance_key: str,
) -> dict[str, Any]:
    if not selected_experts:
        return {
            "reround_action_count": 0,
            "rewritten_group_count": 0,
            "source_tensors_read": 0,
            "changed_code_count": 0,
            "code_count": 0,
            "actions": [],
        }
    index = load_safetensors_index(index_path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in selected_experts:
        grouped.setdefault(str(row["key"]), []).append(row)

    actions: list[dict[str, Any]] = []
    source_tensors_read = 0
    changed_code_count = 0
    code_count = 0
    for key, rows in sorted(grouped.items()):
        layer_text, projection = key.split(":", 1)
        layer = int(layer_text)
        prefix = _switch_prefix(layer, projection)
        group_name = _group_filename(layer, projection)
        baseline_path = baseline_dir / group_name
        target_path = output_dir / group_name
        projection_layer = load_quantized_vq_switch_linear(baseline_path, prefix)
        if projection_layer.code_bits not in (8, 16):
            raise ValueError(f"{group_name} has unsupported code_bits={projection_layer.code_bits}")
        arrays = dict(mx.load(str(baseline_path)))
        metadata = _read_metadata(baseline_path)
        codes_name = f"{prefix}.codes"
        scales_name = f"{prefix}.scales"
        codes = np.asarray(arrays[codes_name]).copy()
        scales = np.asarray(arrays[scales_name], dtype=np.float32)
        codebook = np.asarray(projection_layer.codebook)

        for row in rows:
            expert = int(row["expert"])
            source_weight = _read_named_tensor(
                source_root,
                index,
                _expert_weight_name(layer, expert, projection),
            ).astype(np.float32, copy=False)
            source_tensors_read += 1
            reassigned = imatrix_weighted_reassign_codes(
                source_weight,
                codes[expert],
                scales[expert],
                _load_imatrix_importance(
                    manifest_root,
                    str(row["sidecar_path"]),
                    importance_key=importance_key,
                ),
                codebook=codebook,
                group_size=projection_layer.group_size,
                code_bits=projection_layer.code_bits,
                codeword_chunk_size=codeword_chunk_size,
                layer=layer,
                projection=projection,
                expert=expert,
            )
            codes[expert] = reassigned.codes
            changed_code_count += reassigned.stats.changed_code_count
            code_count += reassigned.stats.code_count
            actions.append(
                {
                    "key": key,
                    "layer": layer,
                    "projection": projection,
                    "expert": expert,
                    "sidecar_path": str(row["sidecar_path"]),
                    "importance_score": float(row["importance_score"]),
                    "importance_key": importance_key,
                    "current_imatrix_weighted_error": reassigned.stats.current_hessian_weighted_error,
                    "imatrix_weighted_error": reassigned.stats.hessian_weighted_error,
                    "changed_code_count": reassigned.stats.changed_code_count,
                    "code_count": reassigned.stats.code_count,
                }
            )

        arrays[codes_name] = mx.array(codes)
        if target_path.exists() or target_path.is_symlink():
            target_path.unlink()
        mx.save_safetensors(str(target_path), arrays, metadata=metadata)

    return {
        "reround_action_count": len(actions),
        "rewritten_group_count": len(grouped),
        "source_tensors_read": source_tensors_read,
        "changed_code_count": int(changed_code_count),
        "code_count": int(code_count),
        "changed_code_fraction": float(changed_code_count / code_count) if code_count else 0.0,
        "actions": actions,
        "model_id": model_id,
        "source_root": str(source_root),
        "index_path": str(index_path),
    }


def _select_low_reround_experts(
    expert_scores: list[dict[str, Any]],
    *,
    tier_map: Mapping[str, str],
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    selected: list[dict[str, Any]] = []
    used_groups: set[str] = set()
    for row in expert_scores:
        key = str(row["key"])
        if str(tier_map.get(key, "low")) != "low":
            continue
        if key in used_groups:
            continue
        selected.append(dict(row))
        used_groups.add(key)
        if len(selected) >= limit:
            break
    return selected


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize a GLM-4.5-Air dynamic imatrix budget sweep."
    )
    parser.add_argument("--imatrix-manifest", required=True)
    parser.add_argument("--baseline-artifact-dir", default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--high-bit-artifact-dir", default=DEFAULT_HIGH_BIT_DIR)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--candidate-prefix", default="air-dynamic-imatrix")
    parser.add_argument("--budget", action="append", help="Effective bits/weight budget. Repeatable.")
    parser.add_argument("--mid-error-factor", type=float, default=0.5)
    parser.add_argument("--reround-top-low-experts", type=int, default=1)
    parser.add_argument("--codeword-chunk-size", type=int, default=4096)
    parser.add_argument("--importance-key", choices=IMPORTANCE_KEYS, default="routing_weighted_importance")
    parser.add_argument("--model-id", default=GLM45_AIR_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--source-dir")
    parser.add_argument("--index-path")
    parser.add_argument("--allow-existing", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.reround_top_low_experts < 0:
        parser.error("--reround-top-low-experts must be non-negative")
    if args.codeword_chunk_size <= 0:
        parser.error("--codeword-chunk-size must be positive")
    try:
        budgets = _parse_budgets(args.budget)
    except ValueError as exc:
        parser.error(str(exc))

    manifest_path = Path(args.imatrix_manifest)
    manifest_root = manifest_path.parent
    manifest = load_projection_imatrix_manifest(manifest_path)
    baseline_dir = Path(args.baseline_artifact_dir)
    high_bit_dir = Path(args.high_bit_artifact_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=args.allow_existing)

    group_stats, expert_scores = _summarize_imatrix_importance(
        manifest,
        manifest_root=manifest_root,
        importance_key=args.importance_key,
    )
    profiles = _build_profiles(
        group_stats,
        baseline_dir=baseline_dir,
        mid_error_factor=args.mid_error_factor,
    )

    source_root = None
    index_path = None
    if args.reround_top_low_experts:
        source_root = _resolve_source_root(
            model_id=args.model_id,
            revision=args.revision,
            source_dir=args.source_dir,
        )
        index_path = Path(args.index_path) if args.index_path is not None else source_root / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"source index does not exist: {index_path}")

    candidates: list[dict[str, Any]] = []
    for budget in budgets:
        plan = allocate_dynamic_precision_tiers(
            profiles,
            budget_bits_per_weight=budget,
            prior_floors=(),
        )
        candidate_name = f"{args.candidate_prefix}-bpw-{budget:.1f}".replace(".", "p")
        candidate_dir = output_root / candidate_name
        tier_report = build_dynamic_precision_tier_map_report(
            profiles,
            plan,
            prior_floors=(),
            source=f"real_imatrix_manifest:{manifest_path}",
        )
        link_summary = _materialize_links(
            tier_map=plan.tier_by_key,
            baseline_dir=baseline_dir,
            high_bit_dir=high_bit_dir,
            output_dir=candidate_dir,
            allow_existing=args.allow_existing,
        )
        selected_rerounds = _select_low_reround_experts(
            expert_scores,
            tier_map=plan.tier_by_key,
            limit=args.reround_top_low_experts,
        )
        reround_summary = (
            _apply_imatrix_rerounds(
                selected_experts=selected_rerounds,
                output_dir=candidate_dir,
                baseline_dir=baseline_dir,
                manifest_root=manifest_root,
                source_root=source_root,
                index_path=index_path,
                model_id=args.model_id,
                codeword_chunk_size=args.codeword_chunk_size,
                importance_key=args.importance_key,
            )
            if source_root is not None and index_path is not None
            else {
                "reround_action_count": 0,
                "rewritten_group_count": 0,
                "source_tensors_read": 0,
                "changed_code_count": 0,
                "code_count": 0,
                "actions": [],
            }
        )
        tier_report_path = candidate_dir / "dynamic-tier-map-report.json"
        summary_path = candidate_dir / "dynamic-materialization-summary.json"
        candidate_summary = {
            "schema_version": 1,
            "record_type": "air_dynamic_imatrix_candidate_materialization",
            "candidate_name": candidate_name,
            "candidate_dir": str(candidate_dir),
            "requested_budget_bits_per_weight": budget,
            "effective_bits_per_weight": plan.effective_bits_per_weight,
            "weighted_error_proxy": plan.weighted_error,
            "baseline_artifact_dir": str(baseline_dir),
            "high_bit_artifact_dir": str(high_bit_dir),
            "imatrix_manifest": str(manifest_path),
            "importance_key": args.importance_key,
            "materialization_kind": "projection_level_e8_e8p_links_with_bounded_low_tier_imatrix_reround",
            "seed_artifacts_mutated": False,
            "link_summary": link_summary,
            "reround_summary": reround_summary,
            "tier_report_path": str(tier_report_path),
        }
        _write_json(tier_report_path, tier_report)
        _write_json(summary_path, candidate_summary)
        candidates.append(candidate_summary)

    sweep_summary = {
        "schema_version": 1,
        "record_type": "air_dynamic_imatrix_budget_sweep_materialization",
        "imatrix_manifest": str(manifest_path),
        "baseline_artifact_dir": str(baseline_dir),
        "high_bit_artifact_dir": str(high_bit_dir),
        "output_root": str(output_root),
        "budget_count": len(budgets),
        "profile_count": len(profiles),
        "expert_score_count": len(expert_scores),
        "mid_error_factor": args.mid_error_factor,
        "reround_top_low_experts": args.reround_top_low_experts,
        "importance_key": args.importance_key,
        "candidates": candidates,
    }
    summary_path = output_root / "dynamic-imatrix-sweep-summary.json"
    _write_json(summary_path, sweep_summary)
    print(json.dumps(sweep_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
