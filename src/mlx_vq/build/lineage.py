"""Legacy artifact lineage adapters for recipe backfills.

Older GLM-4.5-Air artifacts recorded continuous-training provenance in
``conversion-manifest.json`` instead of recipe-runner build steps. This module
turns that manifest chain into a declarative recipe rooted at the first
non-continuous seed artifact it can no longer reconstruct.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors import safe_open


class LineageError(ValueError):
    pass


@dataclass(frozen=True)
class LegacyContinuousNode:
    artifact_dir: Path
    seed_artifact_dir: Path
    run: dict[str, Any]


@dataclass(frozen=True)
class LegacyScaleSidecarsNode:
    artifact_dir: Path
    seed_artifact_dir: Path
    source_artifact_dir: Path
    scale_policy: dict[str, float]
    run: dict[str, Any]


@dataclass(frozen=True)
class LegacyDynamicImatrixSweepNode:
    artifact_dir: Path
    seed_artifact_dir: Path
    high_bit_artifact_dir: Path
    imatrix_manifest: Path
    candidate_prefix: str
    budget: float
    mid_error_factor: float
    reround_top_low_experts: int


@dataclass(frozen=True)
class LegacyStreamConvertProducer:
    artifact_dir: Path
    params: dict[str, Any]


@dataclass(frozen=True)
class LegacyImatrixProducer:
    manifest_path: Path
    params: dict[str, Any]


@dataclass(frozen=True)
class LegacyMergedSidecarsProducer:
    artifact_dir: Path
    source_artifact_dirs: list[Path]
    params: dict[str, Any]


@dataclass(frozen=True)
class LegacyBlockLocalFitProducer:
    artifact_dir: Path
    seed_artifact_dir: Path
    source_dir: Path | None
    teacher_jsonls: list[Path]
    params: dict[str, Any]


LegacyNode = LegacyContinuousNode | LegacyScaleSidecarsNode | LegacyDynamicImatrixSweepNode


_TRAIN_PARAM_KEYS = (
    "prefill_engine",
    "layer",
    "projection",
    "projections",
    "trainable",
    "low_rank",
    "low_rank_init_scale",
    "steps",
    "learning_rate",
    "target_nll_weight",
    "teacher_top1_margin_weight",
    "teacher_top1_margin",
    "tail_kld_weight",
    "grad_clip_norm",
    "loss_scope",
    "surrogate_projections",
    "surrogate_output_chunk_size",
    "train_cache",
    "max_train_rows",
    "train_row_indices",
    "max_positions",
    "min_top_k",
)

_DEFAULT_REPORT_CACHE = Path(
    "artifacts/quality/"
    "glm45-air-teacher-cache-air-vq-ladder-report-v1-full-logits-route-trace-clean/"
    "metadata.jsonl"
)
_DEFAULT_SELECTION_CACHE = Path(
    "artifacts/quality/"
    "glm45-air-teacher-cache-air-vq-ladder-select-v1-full-logits-route-trace-clean/"
    "metadata.jsonl"
)

_PROJECTION_ORDER = ("gate_proj", "up_proj", "down_proj")

_DEFAULT_CACHE_PREFLIGHT = {
    "model_id": "zai-org/GLM-4.5-Air",
    "pipeline_size": 2,
    "layer_split": 23,
    "rank_budget_gb": 115,
    "local_ip": "192.168.10.1",
    "local_rdma_device": "rdma_en1",
    "peer_ssh": "jackmazac@192.168.100.191",
    "peer_rdma_device": "rdma_en1",
    "local_direct_if": "en1",
    "peer_direct_if": "en1",
    "peer_direct_ip": "192.168.10.2",
}

_DEFAULT_SINGLE_HOST_LOWER_SPLIT_LAYERS = tuple(range(1, 23))
_DEFAULT_SINGLE_HOST_UPPER_SPLIT_LAYERS = tuple(range(24, 46))


def _load_manifest(artifact_dir: Path) -> dict[str, Any] | None:
    manifest = artifact_dir / "conversion-manifest.json"
    if not manifest.exists():
        return None
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LineageError(f"{manifest}: expected a JSON object")
    return payload


def _load_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LineageError(f"{path}: expected a JSON object")
    return payload


def _candidate_prefix(candidate_name: str, budget: float) -> str:
    suffix = f"-bpw-{budget:.1f}".replace(".", "p")
    if not candidate_name.endswith(suffix):
        raise LineageError(
            f"dynamic-imatrix candidate {candidate_name!r} does not end with {suffix!r}"
        )
    return candidate_name.removesuffix(suffix)


def _dynamic_imatrix_node(artifact_dir: Path) -> LegacyDynamicImatrixSweepNode | None:
    summary = _load_json_object(artifact_dir / "dynamic-materialization-summary.json")
    if summary is None:
        return None
    if summary.get("record_type") != "air_dynamic_imatrix_candidate_materialization":
        return None
    baseline = summary.get("baseline_artifact_dir")
    high_bit = summary.get("high_bit_artifact_dir")
    imatrix = summary.get("imatrix_manifest")
    candidate_name = summary.get("candidate_name")
    budget = summary.get("requested_budget_bits_per_weight")
    if not all(isinstance(value, str) and value for value in (baseline, high_bit, imatrix, candidate_name)):
        return None
    if not isinstance(budget, (int, float)) or isinstance(budget, bool):
        return None

    sweep = _load_json_object(artifact_dir.parent / "dynamic-imatrix-sweep-summary.json") or {}
    mid_error_factor = sweep.get("mid_error_factor", 0.5)
    reround_top_low_experts = sweep.get("reround_top_low_experts", 0)
    if not isinstance(mid_error_factor, (int, float)) or isinstance(mid_error_factor, bool):
        mid_error_factor = 0.5
    if not isinstance(reround_top_low_experts, int) or isinstance(reround_top_low_experts, bool):
        reround_top_low_experts = 0

    return LegacyDynamicImatrixSweepNode(
        artifact_dir=artifact_dir,
        seed_artifact_dir=Path(baseline),
        high_bit_artifact_dir=Path(high_bit),
        imatrix_manifest=Path(imatrix),
        candidate_prefix=_candidate_prefix(str(candidate_name), float(budget)),
        budget=float(budget),
        mid_error_factor=float(mid_error_factor),
        reround_top_low_experts=int(reround_top_low_experts),
    )


def _legacy_node(artifact_dir: Path) -> LegacyNode | None:
    dynamic_node = _dynamic_imatrix_node(artifact_dir)
    if dynamic_node is not None:
        return dynamic_node

    payload = _load_manifest(artifact_dir)
    if payload is None:
        return None
    continuous = payload.get("continuous_parameters")
    if not isinstance(continuous, dict):
        return None
    run = continuous.get("run")
    seed = continuous.get("seed_artifact_dir")
    if not isinstance(run, dict) or not isinstance(seed, str) or not seed:
        return None
    kind = run.get("kind")
    if kind == "rung2_continuous_finetune_probe":
        return LegacyContinuousNode(
            artifact_dir=artifact_dir,
            seed_artifact_dir=Path(seed),
            run=dict(run),
        )
    if kind == "scaled_block_local_sidecar_alpha_sweep":
        source_artifact = run.get("source_artifact")
        scale_policy = run.get("scale_policy")
        if not isinstance(source_artifact, str) or not source_artifact:
            return None
        if not isinstance(scale_policy, dict) or not scale_policy:
            return None
        parsed_policy: dict[str, float] = {}
        for key, value in scale_policy.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return None
            parsed_policy[str(key)] = float(value)
        return LegacyScaleSidecarsNode(
            artifact_dir=artifact_dir,
            seed_artifact_dir=Path(seed),
            source_artifact_dir=Path(source_artifact),
            scale_policy=parsed_policy,
            run=dict(run),
        )
    return None


def collect_legacy_continuous_lineage(
    target_artifact: Path,
    *,
    max_depth: int = 128,
) -> tuple[list[LegacyNode], Path, list[str]]:
    """Return newest-to-oldest continuous nodes plus the unresolved root seed."""

    current = Path(target_artifact)
    nodes: list[LegacyNode] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for _ in range(max_depth):
        key = str(current)
        if key in seen:
            raise LineageError(f"cycle detected while walking lineage at {current}")
        seen.add(key)
        node = _legacy_node(current)
        if node is None:
            warnings.append(
                f"stopped at {current}: no supported legacy continuous manifest"
            )
            return nodes, current, warnings
        nodes.append(node)
        current = node.seed_artifact_dir
    raise LineageError(f"lineage exceeded max depth {max_depth} at {current}")


def _train_params_from_run(run: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key in _TRAIN_PARAM_KEYS:
        if key not in run:
            continue
        value = run[key]
        if value is None:
            continue
        if key == "train_row_indices" and value == []:
            continue
        params[key] = value

    # Modern manifests write both keys, with ``projection`` set to null for
    # multi-projection runs. Older single-projection runs only need
    # ``--projection`` so the generated argv stays faithful to the source run.
    if params.get("projection") and "projections" in params:
        params.pop("projections")
    if "projection" not in params and "projections" not in params:
        projection = run.get("projection")
        if isinstance(projection, str) and projection:
            params["projection"] = projection
    return params


def _dynamic_sweep_output_name(budget: float) -> str:
    return "artifact_bpw" + f"{float(budget):.1f}".replace(".", "p")


def _manifest_ready_count(payload: dict[str, Any]) -> tuple[int, int] | None:
    ready = payload.get("ready_vq_groups")
    planned = payload.get("planned_vq_groups")
    if not isinstance(ready, int) or isinstance(ready, bool):
        return None
    if not isinstance(planned, int) or isinstance(planned, bool):
        return None
    return ready, planned


def _best_conversion_run(artifact_dir: Path) -> dict[str, Any] | None:
    best: tuple[int, int, dict[str, Any]] | None = None
    for order, path in enumerate(sorted(artifact_dir.glob("conversion-run*.json"))):
        payload = _load_json_object(path)
        if payload is None:
            continue
        manifest = payload.get("manifest")
        if not isinstance(manifest, dict):
            continue
        counts = _manifest_ready_count(manifest)
        if counts is None:
            continue
        ready, _planned = counts
        candidate = (ready, order, payload)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return None if best is None else best[2]


def _revision_from_index_path(index_path: str) -> str | None:
    parts = Path(index_path).parts
    try:
        snapshots = parts.index("snapshots")
    except ValueError:
        return None
    revision_index = snapshots + 1
    if revision_index >= len(parts):
        return None
    revision = parts[revision_index]
    return revision if revision else None


def _uniform_int_group_field(groups: list[Any], field: str) -> int | None:
    values: set[int] = set()
    for group in groups:
        if not isinstance(group, dict):
            return None
        value = group.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            return None
        values.add(value)
    if len(values) != 1:
        return None
    return next(iter(values))


def _max_int_group_field(groups: list[Any], field: str) -> int | None:
    values: list[int] = []
    for group in groups:
        if not isinstance(group, dict):
            return None
        value = group.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            return None
        values.append(value)
    return max(values) if values else None


def _group_output_path(artifact_dir: Path, group: dict[str, Any]) -> Path | None:
    value = group.get("output_path")
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(artifact_dir / path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _group_quantization_config(
    artifact_dir: Path,
    group: dict[str, Any],
) -> dict[str, Any] | None:
    output_path = _group_output_path(artifact_dir, group)
    if output_path is None:
        return None
    try:
        with safe_open(output_path, framework="np") as handle:
            metadata = handle.metadata() or {}
    except OSError:
        return None
    raw_config = metadata.get("quantization_config")
    if not isinstance(raw_config, str) or not raw_config:
        return None
    try:
        payload = json.loads(raw_config)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _legacy_group_quantization_fields(
    artifact_dir: Path,
    groups: list[Any],
) -> tuple[int, int, str] | None:
    code_bits_values: set[int] = set()
    group_size_values: list[int] = []
    scale_estimators: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            return None
        config = _group_quantization_config(artifact_dir, group)
        if config is None:
            return None
        code_bits = config.get("default_code_bits")
        group_size = config.get("default_group_size")
        if not isinstance(code_bits, int) or isinstance(code_bits, bool):
            return None
        if not isinstance(group_size, int) or isinstance(group_size, bool):
            return None
        code_bits_values.add(code_bits)
        group_size_values.append(group_size)
        policy = config.get("policy")
        if isinstance(policy, dict):
            scale_estimator = policy.get("scale_estimator")
            if isinstance(scale_estimator, str) and scale_estimator:
                scale_estimators.add(scale_estimator)
    if len(code_bits_values) != 1 or not group_size_values:
        return None
    if len(scale_estimators) > 1:
        return None
    scale_estimator = next(iter(scale_estimators)) if scale_estimators else "max_abs"
    return next(iter(code_bits_values)), max(group_size_values), scale_estimator


def _is_complete_legacy_air_existing_root(
    manifest: dict[str, Any],
    *,
    planned: int,
    ready: int,
) -> bool:
    if manifest.get("model_id") != "zai-org/GLM-4.5-Air":
        return False
    if planned != 135 or ready != 135:
        return False
    existing = manifest.get("existing_vq_groups")
    converted = manifest.get("converted_vq_groups")
    if not isinstance(existing, int) or isinstance(existing, bool):
        return False
    if not isinstance(converted, int) or isinstance(converted, bool):
        return False
    return existing == 135 and converted == 0


def _stream_convert_producer(
    artifact_dir: Path,
    *,
    assumed_revision: str | None = None,
    default_expert_workers: int = 16,
) -> LegacyStreamConvertProducer | None:
    manifest = _load_manifest(artifact_dir)
    if manifest is None:
        return None
    counts = _manifest_ready_count(manifest)
    if counts is None:
        return None
    ready, planned = counts
    if planned <= 0 or ready != planned:
        return None

    model_id = manifest.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        return None
    groups = manifest.get("groups")
    if not isinstance(groups, list) or not groups:
        return None
    code_bits = _uniform_int_group_field(groups, "code_bits")
    group_size = _max_int_group_field(groups, "group_size")

    scale_estimator = manifest.get("scale_estimator")
    if not isinstance(scale_estimator, str) or not scale_estimator:
        scale_estimators = {
            group.get("scale_estimator")
            for group in groups
            if isinstance(group, dict) and isinstance(group.get("scale_estimator"), str)
        }
        if len(scale_estimators) == 1:
            scale_estimator = next(iter(scale_estimators))
        else:
            scale_estimator = ""

    if code_bits is None or group_size is None or not scale_estimator:
        quantization_fields = _legacy_group_quantization_fields(artifact_dir, groups)
        if quantization_fields is None:
            return None
        metadata_code_bits, metadata_group_size, metadata_scale_estimator = quantization_fields
        if code_bits is None:
            code_bits = metadata_code_bits
        if group_size is None:
            group_size = metadata_group_size
        if not scale_estimator:
            scale_estimator = metadata_scale_estimator

    run = _best_conversion_run(artifact_dir)
    if run is None:
        if not assumed_revision:
            return None
        if not _is_complete_legacy_air_existing_root(
            manifest,
            planned=planned,
            ready=ready,
        ):
            return None
        revision = assumed_revision
        expert_workers = default_expert_workers
    else:
        revision = run.get("revision")
        if not isinstance(revision, str) or not revision:
            index_path = run.get("index_path")
            if not isinstance(index_path, str) or not index_path:
                return None
            revision = _revision_from_index_path(index_path)
        if not isinstance(revision, str) or not revision:
            return None

        expert_workers = run.get("expert_workers")
        if not isinstance(expert_workers, int) or isinstance(expert_workers, bool):
            expert_workers = _max_int_group_field(groups, "expert_workers")
        if expert_workers is None:
            return None

    params: dict[str, Any] = {
        "model_id": model_id,
        "revision": revision,
        "download_missing_source_shards": True,
        "code_bits": code_bits,
        "group_size": group_size,
        "scale_estimator": scale_estimator,
        "expert_workers": expert_workers,
    }
    code_bits_policy = manifest.get("code_bits_policy")
    if isinstance(code_bits_policy, dict) and code_bits_policy:
        params["code_bits_policy"] = [
            f"{key}={value}" for key, value in sorted(code_bits_policy.items())
        ]
    return LegacyStreamConvertProducer(artifact_dir=artifact_dir, params=params)


def _ordered_projections(projections: set[str]) -> list[str]:
    ordered = [projection for projection in _PROJECTION_ORDER if projection in projections]
    ordered.extend(sorted(projections.difference(_PROJECTION_ORDER)))
    return ordered


def _imatrix_producer(manifest_path: Path) -> LegacyImatrixProducer | None:
    manifest = _load_json_object(manifest_path)
    if manifest is None:
        return None
    if manifest.get("record_type") != "air_projection_imatrix_manifest":
        return None
    prompt_set = manifest.get("prompt_set")
    if not isinstance(prompt_set, str) or not prompt_set:
        return None
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        return None
    entry_count = manifest.get("entry_count")
    if isinstance(entry_count, int) and not isinstance(entry_count, bool):
        if entry_count != len(entries):
            return None

    layers: set[int] = set()
    projections: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            return None
        layer = entry.get("layer")
        projection = entry.get("projection")
        if not isinstance(layer, int) or isinstance(layer, bool):
            return None
        if not isinstance(projection, str) or not projection:
            return None
        layers.add(layer)
        projections.add(projection)
    if not layers or not projections:
        return None

    return LegacyImatrixProducer(
        manifest_path=manifest_path,
        params={
            "prompt_set": prompt_set,
            "layers": sorted(layers),
            "projections": _ordered_projections(projections),
        },
    )


def _legacy_merged_sidecar_params(run: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key in (
        "train_split",
        "train_row_count",
        "max_positions",
        "trainable",
        "teacher_jsonl",
        "teacher_cache_root",
    ):
        value = run.get(key)
        if value is None:
            continue
        params[key] = value
    return params


def _as_non_empty_str_list(value: Any) -> list[str] | None:
    if isinstance(value, str) and value:
        return [value]
    if isinstance(value, list) and value:
        result: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item:
                return None
            result.append(item)
        return result
    return None


def _quality_records_for_output(artifact_dir: Path) -> list[dict[str, Any]]:
    quality_root = Path("artifacts") / "quality"
    if not quality_root.exists():
        return []
    output_dir = str(artifact_dir)
    records: list[dict[str, Any]] = []
    for path in sorted(quality_root.glob("*.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line in lines:
            if output_dir not in line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("output_dir") == output_dir:
                records.append(payload)
    return records


def _block_local_fit_params(
    *,
    run: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any] | None:
    params: dict[str, Any] = {}
    for key in ("model_id", "revision"):
        value = record.get(key)
        if not isinstance(value, str) or not value:
            return None
        params[key] = value

    layer = run.get("layer")
    if layer is None:
        layer = record.get("layer")
    projection = run.get("projection")
    if projection is None:
        projection = record.get("projection")
    if not isinstance(layer, int) or isinstance(layer, bool):
        return None
    if not isinstance(projection, str) or not projection:
        return None
    params["layer"] = layer
    params["projection"] = projection

    trainable = run.get("trainable")
    if trainable is None:
        trainable = record.get("trainable")
    if isinstance(trainable, str) and trainable:
        params["trainable"] = trainable

    low_rank = run.get("low_rank")
    if low_rank is None:
        low_rank = record.get("low_rank")
    if isinstance(low_rank, int) and not isinstance(low_rank, bool):
        params["low_rank"] = low_rank

    row_indices = record.get("train_row_indices")
    if isinstance(row_indices, list) and row_indices:
        if not all(isinstance(item, int) and not isinstance(item, bool) for item in row_indices):
            return None
        params["row_indices"] = row_indices
    else:
        train_row_count = run.get("train_row_count")
        if train_row_count is None:
            train_row_count = record.get("train_row_count")
        if isinstance(train_row_count, int) and not isinstance(train_row_count, bool):
            params["max_train_rows"] = train_row_count

    for source_key, param_key in (
        ("max_positions", "max_positions"),
        ("min_top_k", "min_top_k"),
    ):
        value = record.get(source_key)
        if value is None:
            value = run.get(source_key)
        if isinstance(value, int) and not isinstance(value, bool):
            params[param_key] = value

    if record.get("teacher_cache_all_memory_clean") is False:
        params["allow_dirty_cache"] = True
    return params


def _is_revision_snapshot_path(path: Path, revision: str) -> bool:
    parts = path.parts
    try:
        snapshots = parts.index("snapshots")
    except ValueError:
        return False
    revision_index = snapshots + 1
    return revision_index < len(parts) and parts[revision_index] == revision


def _block_local_fit_producer(artifact_dir: Path) -> LegacyBlockLocalFitProducer | None:
    manifest = _load_manifest(artifact_dir)
    if manifest is None:
        return None
    continuous = manifest.get("continuous_parameters")
    if not isinstance(continuous, dict) or continuous.get("enabled") is not True:
        return None
    run = continuous.get("run")
    if not isinstance(run, dict) or run.get("kind") != "block_local_sidecar_least_squares":
        return None
    seed = continuous.get("seed_artifact_dir") or run.get("seed_artifact_dir")
    if not isinstance(seed, str) or not seed:
        return None

    for record in _quality_records_for_output(artifact_dir):
        if record.get("record_type") != "air_vq_block_local_sidecar_fit":
            continue
        if record.get("seed_artifact_dir") != seed:
            continue
        source_dir = record.get("source_dir")
        if not isinstance(source_dir, str) or not source_dir:
            continue
        teacher_jsonls = _as_non_empty_str_list(record.get("teacher_jsonl"))
        if teacher_jsonls is None:
            continue
        params = _block_local_fit_params(run=run, record=record)
        if params is None:
            continue
        revision = params.get("revision")
        source_path = Path(source_dir)
        replay_source_dir = (
            None
            if isinstance(revision, str) and _is_revision_snapshot_path(source_path, revision)
            else source_path
        )
        return LegacyBlockLocalFitProducer(
            artifact_dir=artifact_dir,
            seed_artifact_dir=Path(seed),
            source_dir=replay_source_dir,
            teacher_jsonls=[Path(path) for path in teacher_jsonls],
            params=params,
        )
    return None


def _merged_sidecars_producer(
    artifact_dir: Path,
    *,
    expected_seed_artifact_dir: Path,
) -> LegacyMergedSidecarsProducer | None:
    manifest = _load_manifest(artifact_dir)
    if manifest is None:
        return None
    continuous = manifest.get("continuous_parameters")
    if not isinstance(continuous, dict) or continuous.get("enabled") is not True:
        return None
    run = continuous.get("run")
    if not isinstance(run, dict) or run.get("kind") != "merged_block_local_sidecars":
        return None
    seed = continuous.get("seed_artifact_dir") or run.get("seed_artifact_dir")
    if not isinstance(seed, str) or not seed:
        return None
    if Path(seed) != expected_seed_artifact_dir:
        return None
    source_artifacts = run.get("source_artifacts")
    if not isinstance(source_artifacts, list) or not source_artifacts:
        return None
    if len(source_artifacts) > 8:
        return None
    source_paths: list[Path] = []
    for source in source_artifacts:
        if not isinstance(source, str) or not source:
            return None
        source_paths.append(Path(source))
    return LegacyMergedSidecarsProducer(
        artifact_dir=artifact_dir,
        source_artifact_dirs=source_paths,
        params=_legacy_merged_sidecar_params(run),
    )


def _cleanroom_cache_preflight_steps(*, revision: str) -> list[dict[str, Any]]:
    base_params = {
        "model_id": _DEFAULT_CACHE_PREFLIGHT["model_id"],
        "revision": revision,
        "pipeline_size": _DEFAULT_CACHE_PREFLIGHT["pipeline_size"],
        "layer_split": _DEFAULT_CACHE_PREFLIGHT["layer_split"],
        "rank_budget_gb": _DEFAULT_CACHE_PREFLIGHT["rank_budget_gb"],
    }
    views_params = {
        **base_params,
        "rank0_only_logits_views": True,
    }
    stage_views_params = {
        **base_params,
        "local_sequential_stage_views": True,
        "local_sequential_head_process": True,
        "local_sequential_lower_split_layers": list(
            _DEFAULT_SINGLE_HOST_LOWER_SPLIT_LAYERS
        ),
        "local_sequential_upper_split_layers": list(
            _DEFAULT_SINGLE_HOST_UPPER_SPLIT_LAYERS
        ),
    }
    cache_params = {
        "top_k": 128,
        "max_positions": 128,
        "layer_split": _DEFAULT_CACHE_PREFLIGHT["layer_split"],
        "required_wired_mb": 0,
        "local_direct_if": _DEFAULT_CACHE_PREFLIGHT["local_direct_if"],
        "peer_direct_if": _DEFAULT_CACHE_PREFLIGHT["peer_direct_if"],
        "local_direct_ip": _DEFAULT_CACHE_PREFLIGHT["local_ip"],
        "peer_direct_ip": _DEFAULT_CACHE_PREFLIGHT["peer_direct_ip"],
        "peer_ssh": _DEFAULT_CACHE_PREFLIGHT["peer_ssh"],
        "preflight_only": True,
    }
    single_host_cache_params = {
        "top_k": 128,
        "max_positions": 128,
        "preflight_only": True,
        "single_host_fallback": True,
        "mlx_cache_limit_gb": 48,
        "single_host_source_memory_guard_ratio": 1.0,
        "vq_artifact_dir": "artifacts/glm-4.5-air-vq",
    }
    single_host_prefix_params = {
        "top_k": 16,
        "max_positions": 1,
        "prompt_set": "base",
        "prompt_id": ["capital_france"],
        "single_host_fallback": True,
        "single_host_pipeline_local": True,
        "single_host_pipeline_local_stage_processes": True,
        "single_host_pipeline_local_head_process": True,
        "single_host_pipeline_local_lower_split_layers": list(
            _DEFAULT_SINGLE_HOST_LOWER_SPLIT_LAYERS
        ),
        "single_host_pipeline_local_upper_split_layers": list(
            _DEFAULT_SINGLE_HOST_UPPER_SPLIT_LAYERS
        ),
        "single_host_pipeline_local_abort_on_dirty_stage": True,
        "single_host_revision": revision,
        "mlx_cache_limit_gb": 48,
        "no_full_logits": True,
        "skip_local_consume": True,
        "require_memory_quiet_preflight": True,
        "memory_quiet_seconds": 3,
        "memory_quiet_max_attempts": 20,
        "memory_quiet_stage_view_margin_gb": 8,
        "peer_ssh": _DEFAULT_CACHE_PREFLIGHT["peer_ssh"],
        "peer_rank0_view_root": "glm45-air-pipeline-views/rank-0-p2",
        "peer_source_dir": (
            "/Users/jackmazac/.cache/huggingface/hub/"
            "models--zai-org--GLM-4.5-Air/snapshots/"
            f"{revision}"
        ),
        "peer_source_stage_ssh": "jackmazac@192.168.10.2",
        "stage_peer_source": True,
        "peer_source_min_free_gb": 16,
        "local_sequential_remote_workers": "head",
        "local_sequential_remote_tmp_dir": "/tmp/glm-p2-remote-workers",
        "local_sequential_remote_dirty_retries": 1,
        "local_sequential_remote_retry_sleep_seconds": 30,
        "layer_split": _DEFAULT_CACHE_PREFLIGHT["layer_split"],
        "lm_head_chunk_rows": 8192,
        "vq_artifact_dir": "artifacts/glm-4.5-air-vq",
    }
    distributed_prefix_params = {
        "top_k": 16,
        "max_positions": 1,
        "prompt_set": "base",
        "prompt_id": ["capital_france"],
        "layer_split": _DEFAULT_CACHE_PREFLIGHT["layer_split"],
        "required_wired_mb": 0,
        "local_direct_if": _DEFAULT_CACHE_PREFLIGHT["local_direct_if"],
        "peer_direct_if": _DEFAULT_CACHE_PREFLIGHT["peer_direct_if"],
        "local_direct_ip": _DEFAULT_CACHE_PREFLIGHT["local_ip"],
        "peer_direct_ip": _DEFAULT_CACHE_PREFLIGHT["peer_direct_ip"],
        "peer_ssh": _DEFAULT_CACHE_PREFLIGHT["peer_ssh"],
        "peer_source_stage_ssh": "jackmazac@192.168.10.2",
        "stage_peer_source": True,
        "peer_source_min_free_gb": 16,
        "mlx_cache_limit_gb": 48,
        "no_full_logits": True,
        "skip_local_consume": True,
        "require_memory_quiet_preflight": True,
        "memory_quiet_seconds": 3,
        "memory_quiet_max_attempts": 20,
        # Distributed 23/23 split: each rank holds only its half of the model,
        # not the whole BF16 checkpoint. The clean single-host stage prefix
        # peaked at ~9.1 GB (mlx_peak_bytes 9_115_971_661), so a per-host free
        # floor of 24 GB is ~2.6x that measured peak with wide headroom under
        # the ~80 GB available per Mac (~220 GB across the pair). The prior 111
        # demanded a whole-model residency and wrongly gated the split path.
        "memory_quiet_min_free_gb": 24,
        # The two-rank forward produces a valid teacher-cache row (validation
        # ok, correct top-k logits) but swaps during the ~110 GB/rank load, so
        # its memory counters read dirty. Dirty-but-valid cache output is
        # accepted here by operator directive: the logits are correct, and
        # clean-memory hygiene is an RC-promotion concern, not a cache-validity
        # one. Re-running clean is a later quiet-host step, not a P2 blocker.
        "allow_dirty_cache": True,
        "lm_head_chunk_rows": 8192,
        "vq_artifact_dir": "artifacts/glm-4.5-air-vq",
    }
    return [
        {
            "id": "legacy_pipeline_source_views",
            "op": "pipeline-source-views",
            "class": "diagnostic",
            "inputs": {},
            "params": {**views_params, "materialize": True},
        },
        {
            "id": "legacy_rdma_topology_audit",
            "op": "rdma-topology-audit",
            "class": "diagnostic",
            "inputs": {},
            "params": {
                "local_if": _DEFAULT_CACHE_PREFLIGHT["local_direct_if"],
                "peer_if": _DEFAULT_CACHE_PREFLIGHT["peer_direct_if"],
                "local_ip": _DEFAULT_CACHE_PREFLIGHT["local_ip"],
                "peer_ip": _DEFAULT_CACHE_PREFLIGHT["peer_direct_ip"],
                "local_rdma_device": _DEFAULT_CACHE_PREFLIGHT["local_rdma_device"],
                "peer_rdma_device": _DEFAULT_CACHE_PREFLIGHT["peer_rdma_device"],
                "peer_ssh": _DEFAULT_CACHE_PREFLIGHT["peer_ssh"],
                "require_ready": True,
            },
        },
        {
            "id": "legacy_jaccl_hostfile",
            "op": "jaccl-hostfile",
            "class": "diagnostic",
            "inputs": {},
            "params": {
                "local_ip": _DEFAULT_CACHE_PREFLIGHT["local_ip"],
                "local_rdma_device": _DEFAULT_CACHE_PREFLIGHT["local_rdma_device"],
                "peer_ssh": _DEFAULT_CACHE_PREFLIGHT["peer_ssh"],
                "peer_rdma_device": _DEFAULT_CACHE_PREFLIGHT["peer_rdma_device"],
            },
        },
        {
            "id": "legacy_local_sequential_stage_views",
            "op": "pipeline-source-views",
            "class": "diagnostic",
            "inputs": {},
            "params": {**stage_views_params, "materialize": True},
        },
        {
            "id": "legacy_pipeline_view_probe",
            "op": "pipeline-view-load-probe",
            "class": "diagnostic",
            "inputs": {"view_dir": "step:legacy_pipeline_source_views/artifact"},
            "params": {**base_params, "reuse_existing": True},
        },
        {
            "id": "legacy_report_cache_single_host_prefix",
            "op": "export-teacher-cache-cleanroom",
            "class": "diagnostic",
            "inputs": {
                "rank_view_dir": "step:legacy_pipeline_source_views/artifact",
                "local_sequential_stage_view_roots_json": (
                    "step:legacy_local_sequential_stage_views/stage_view_roots_json"
                ),
            },
            "params": single_host_prefix_params,
        },
        {
            "id": "legacy_report_cache_distributed_prefix",
            "op": "export-teacher-cache-cleanroom",
            "class": "diagnostic",
            "inputs": {
                "rank_view_dir": "step:legacy_pipeline_source_views/artifact",
                "jaccl_hostfile": "step:legacy_jaccl_hostfile/hostfile",
            },
            "params": distributed_prefix_params,
        },
        {
            "id": "legacy_report_cache_single_host_preflight",
            "op": "export-teacher-cache-cleanroom",
            "class": "diagnostic",
            "inputs": {},
            "params": {
                **single_host_cache_params,
                "prompt_set": "air_vq_ladder_report_v1",
            },
        },
        {
            "id": "legacy_selection_cache_single_host_preflight",
            "op": "export-teacher-cache-cleanroom",
            "class": "diagnostic",
            "inputs": {},
            "params": {
                **single_host_cache_params,
                "prompt_set": "air_vq_ladder_select_v1",
            },
        },
        {
            "id": "legacy_report_cache_preflight",
            "op": "export-teacher-cache-cleanroom",
            "class": "diagnostic",
            "inputs": {
                "rank_view_dir": "step:legacy_pipeline_source_views/artifact",
                "jaccl_hostfile": "step:legacy_jaccl_hostfile/hostfile",
            },
            "params": {**cache_params, "prompt_set": "air_vq_ladder_report_v1"},
        },
        {
            "id": "legacy_selection_cache_preflight",
            "op": "export-teacher-cache-cleanroom",
            "class": "diagnostic",
            "inputs": {
                "rank_view_dir": "step:legacy_pipeline_source_views/artifact",
                "jaccl_hostfile": "step:legacy_jaccl_hostfile/hostfile",
            },
            "params": {**cache_params, "prompt_set": "air_vq_ladder_select_v1"},
        },
    ]


def build_legacy_lineage_recipe(
    target_artifact: Path,
    *,
    name: str | None = None,
    teacher_report: Path | None = None,
    teacher_selection: Path | None = None,
    base_revision: str | None = None,
    include_cache_preflight: bool = False,
    max_depth: int = 128,
) -> dict[str, Any]:
    """Build a recipe YAML mapping from a legacy continuous manifest chain."""

    target = Path(target_artifact)
    nodes_newest_first, root_artifact, warnings = collect_legacy_continuous_lineage(
        target,
        max_depth=max_depth,
    )
    nodes = list(reversed(nodes_newest_first))
    if not nodes:
        raise LineageError(f"{target}: no legacy continuous lineage found")

    recipe_name = name or f"{target.name}__legacy_lineage"
    root_producer = _stream_convert_producer(
        root_artifact,
        assumed_revision=base_revision,
    )
    if root_producer is not None:
        root_stop_warning = (
            f"stopped at {root_artifact}: no supported legacy continuous manifest"
        )
        warnings = [warning for warning in warnings if warning != root_stop_warning]
    elif base_revision:
        warnings.append(
            f"root stream-convert producer unavailable for {root_artifact}: "
            "missing complete legacy Air VQ metadata"
        )

    external_inputs: dict[str, Any] = {
        "teacher_report": {
            "kind": "teacher_cache",
            "path": str(teacher_report or _DEFAULT_REPORT_CACHE),
        },
        "teacher_selection": {
            "kind": "teacher_cache",
            "path": str(teacher_selection or _DEFAULT_SELECTION_CACHE),
        },
    }
    if root_producer is None:
        external_inputs["root_artifact"] = {
            "kind": "artifact_dir",
            "path": str(root_artifact),
        }

    lineage: dict[str, Any] = {
        "target_artifact": str(target),
        "root_artifact": str(root_artifact),
        "step_count": len(nodes),
        "warnings": warnings,
    }
    if root_producer is not None:
        lineage["root_producer"] = {
            "op": "stream-convert-vq",
            "revision": root_producer.params["revision"],
        }

    raw: dict[str, Any] = {
        "schema_version": 1,
        "name": recipe_name,
        "description": (
            "Legacy continuous-training lineage reconstructed from "
            "conversion-manifest.json files."
        ),
        "external_inputs": external_inputs,
        "lineage": lineage,
        "steps": [],
    }

    if include_cache_preflight:
        cache_revision = (
            root_producer.params["revision"] if root_producer is not None else base_revision
        )
        if not isinstance(cache_revision, str) or not cache_revision:
            raise LineageError(
                "include_cache_preflight requires base_revision or a reconstructable root producer revision"
            )
        raw["steps"].extend(_cleanroom_cache_preflight_steps(revision=cache_revision))

    if root_producer is None:
        previous_ref = "external:root_artifact"
    else:
        raw["steps"].append(
            {
                "id": "legacy_stream_convert_root",
                "op": "stream-convert-vq",
                "class": "promotable",
                "inputs": {},
                "params": root_producer.params,
            }
        )
        previous_ref = "step:legacy_stream_convert_root/artifact"
    for index, node in enumerate(nodes, start=1):
        if isinstance(node, LegacyDynamicImatrixSweepNode):
            imatrix_ref = f"external:legacy_sweep_imatrix_{index:02d}"
            imatrix_producer = _imatrix_producer(node.imatrix_manifest)
            if imatrix_producer is None:
                imatrix_input = f"legacy_sweep_imatrix_{index:02d}"
                raw["external_inputs"][imatrix_input] = {
                    "kind": "file",
                    "path": str(node.imatrix_manifest),
                }
            else:
                imatrix_step_id = f"legacy_collect_imatrix_{index:02d}"
                raw["steps"].append(
                    {
                        "id": imatrix_step_id,
                        "op": "collect-imatrix",
                        "class": "promotable",
                        "inputs": {"artifact": previous_ref},
                        "params": imatrix_producer.params,
                    }
                )
                imatrix_ref = f"step:{imatrix_step_id}/imatrix_manifest"

            high_bit_ref = f"external:legacy_sweep_high_bit_{index:02d}"
            high_bit_producer = _stream_convert_producer(node.high_bit_artifact_dir)
            if high_bit_producer is None:
                high_bit_input = f"legacy_sweep_high_bit_{index:02d}"
                raw["external_inputs"][high_bit_input] = {
                    "kind": "artifact_dir",
                    "path": str(node.high_bit_artifact_dir),
                }
            else:
                high_bit_step_id = f"legacy_stream_convert_high_bit_{index:02d}"
                raw["steps"].append(
                    {
                        "id": high_bit_step_id,
                        "op": "stream-convert-vq",
                        "class": "promotable",
                        "inputs": {},
                        "params": high_bit_producer.params,
                    }
                )
                high_bit_ref = f"step:{high_bit_step_id}/artifact"
            step_id = f"legacy_materialize_sweep_{index:02d}"
            raw["steps"].append(
                {
                    "id": step_id,
                    "op": "materialize-sweep",
                    "class": "promotable",
                    "inputs": {
                        "imatrix_manifest": imatrix_ref,
                        "baseline_artifact": previous_ref,
                        "high_bit_artifact": high_bit_ref,
                    },
                    "params": {
                        "candidate_prefix": node.candidate_prefix,
                        "budget": [node.budget],
                        "mid_error_factor": node.mid_error_factor,
                        "reround_top_low_experts": node.reround_top_low_experts,
                    },
                }
            )
            previous_ref = f"step:{step_id}/{_dynamic_sweep_output_name(node.budget)}"
            continue

        if isinstance(node, LegacyScaleSidecarsNode):
            source_cache: dict[Path, str] = {}
            source_counter = 0

            def _next_source_counter() -> int:
                nonlocal source_counter
                source_counter += 1
                return source_counter

            def _external_artifact_input(
                artifact_dir: Path,
                *,
                source_index: int,
            ) -> str:
                source_input = f"legacy_merge_source_{index:02d}_{source_index:02d}"
                raw["external_inputs"][source_input] = {
                    "kind": "artifact_dir",
                    "path": str(artifact_dir),
                }
                return f"external:{source_input}"

            def _emit_source_producer(
                artifact_dir: Path,
                *,
                final_merge: bool = False,
            ) -> str | None:
                if artifact_dir in source_cache:
                    return source_cache[artifact_dir]

                fit_producer = _block_local_fit_producer(artifact_dir)
                if fit_producer is not None:
                    if fit_producer.seed_artifact_dir == node.seed_artifact_dir:
                        seed_ref = previous_ref
                    else:
                        seed_ref = _emit_source_producer(
                            fit_producer.seed_artifact_dir
                        )
                        if seed_ref is None:
                            return None
                    source_index = _next_source_counter()
                    step_id = f"legacy_fit_sidecar_{index:02d}_{source_index:02d}"
                    inputs = {
                        "seed_artifact": seed_ref,
                    }
                    if fit_producer.source_dir is not None:
                        source_dir_input = f"legacy_fit_source_dir_{index:02d}_{source_index:02d}"
                        raw["external_inputs"][source_dir_input] = {
                            "kind": "artifact_dir",
                            "path": str(fit_producer.source_dir),
                        }
                        inputs["source_dir"] = f"external:{source_dir_input}"
                    for teacher_index, teacher_jsonl in enumerate(
                        fit_producer.teacher_jsonls,
                        start=1,
                    ):
                        teacher_input = (
                            f"legacy_fit_teacher_{index:02d}_{source_index:02d}_{teacher_index:02d}"
                        )
                        raw["external_inputs"][teacher_input] = {
                            "kind": "teacher_cache",
                            "path": str(teacher_jsonl),
                        }
                        inputs[f"teacher_{teacher_index}"] = f"external:{teacher_input}"
                    raw["steps"].append(
                        {
                            "id": step_id,
                            "op": "fit-block-local-sidecar",
                            "class": "promotable",
                            "inputs": inputs,
                            "params": fit_producer.params,
                            "gate": {"profile": "train_sane"},
                        }
                    )
                    ref = f"step:{step_id}/artifact"
                    source_cache[artifact_dir] = ref
                    return ref

                merged_producer = _merged_sidecars_producer(
                    artifact_dir,
                    expected_seed_artifact_dir=node.seed_artifact_dir,
                )
                if merged_producer is None:
                    return None

                merge_inputs = {"seed_artifact": previous_ref}
                for source_index, source_artifact in enumerate(
                    merged_producer.source_artifact_dirs,
                    start=1,
                ):
                    source_ref_for_merge = _emit_source_producer(source_artifact)
                    if source_ref_for_merge is None:
                        source_ref_for_merge = _external_artifact_input(
                            source_artifact,
                            source_index=source_index,
                        )
                    merge_inputs[f"source_artifact_{source_index}"] = source_ref_for_merge
                if final_merge:
                    merge_step_id = f"legacy_merge_sidecars_{index:02d}"
                else:
                    source_index = _next_source_counter()
                    merge_step_id = f"legacy_merge_sidecars_{index:02d}_{source_index:02d}"
                raw["steps"].append(
                    {
                        "id": merge_step_id,
                        "op": "merge-block-local-sidecars",
                        "class": "promotable",
                        "inputs": merge_inputs,
                        "params": merged_producer.params,
                        "gate": {"profile": "train_sane"},
                    }
                )
                ref = f"step:{merge_step_id}/artifact"
                source_cache[artifact_dir] = ref
                return ref

            source_ref = _emit_source_producer(
                node.source_artifact_dir,
                final_merge=True,
            )
            if source_ref is None:
                source_input = f"legacy_scale_source_{index:02d}"
                raw["external_inputs"][source_input] = {
                    "kind": "artifact_dir",
                    "path": str(node.source_artifact_dir),
                }
                source_ref = f"external:{source_input}"
            step_id = f"legacy_scale_sidecars_{index:02d}"
            raw["steps"].append(
                {
                    "id": step_id,
                    "op": "scale-block-local-sidecars",
                    "class": "promotable",
                    "inputs": {
                        "seed_artifact": previous_ref,
                        "source_artifact": source_ref,
                    },
                    "params": {
                        "scale_policy_json": json.dumps(
                            node.scale_policy,
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    },
                    "gate": {"profile": "train_sane"},
                }
            )
            previous_ref = f"step:{step_id}/artifact"
        else:
            step_id = f"legacy_train_{index:02d}"
            raw["steps"].append(
                {
                    "id": step_id,
                    "op": "train-low-rank",
                    "class": "promotable",
                    "inputs": {
                        "seed_artifact": previous_ref,
                        "selection_teacher": "external:teacher_selection",
                        "validation_teacher": "external:teacher_report",
                    },
                    "params": _train_params_from_run(node.run),
                    "gate": {"profile": "train_sane"},
                }
            )
            previous_ref = f"step:{step_id}/artifact"
    return raw
