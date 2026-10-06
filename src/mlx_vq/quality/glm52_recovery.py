"""Frozen-selection activation statistics for GLM-5.2 wave-1 recovery.

This collector reuses the authenticated GLM-5.2 source-teacher construction
and its layer-major streaming forward.  Observation hooks only copy routed
inputs, route IDs, and route scores to host memory; they never alter values fed
back into the source logits path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np

from mlx_vq.io.authenticated_artifacts import AuthenticatedFile


SPARSE_LAYERS = tuple(range(3, 78))
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
EXPECTED_SELECTION_PROMPTS = 22
EXPECTED_SELECTION_POSITIONS = 255


def authenticated_recovery_candidate_group_paths(
    recovery_audit: Any,
) -> dict[str, Path]:
    """Resolve an audited mixed view to exact files for routed binding."""

    recovery_audit.verify_current_identity()
    recovery_root = Path(recovery_audit.recovery_dir).resolve(strict=True)
    artifact_root = recovery_root / "artifact"
    if not artifact_root.is_dir() or artifact_root.is_symlink():
        raise ValueError("recovery candidate artifact root must be a real directory")

    groups_by_filename: dict[str, Any] = {}
    for group in recovery_audit.groups:
        filename = group.filename
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
            or filename in groups_by_filename
        ):
            raise ValueError("recovery audit has an invalid or duplicate group filename")
        groups_by_filename[filename] = group
    if {entry.name for entry in artifact_root.iterdir()} != set(groups_by_filename):
        raise ValueError("recovery candidate artifact inventory changed after audit")

    resolved_paths: dict[str, Path] = {}
    for filename, group in groups_by_filename.items():
        link = artifact_root / filename
        if not link.is_symlink():
            raise ValueError(f"recovery candidate group must remain a link: {filename}")
        try:
            resolved = link.resolve(strict=True)
        except (FileNotFoundError, RuntimeError) as error:
            raise ValueError(
                f"recovery candidate group link is missing or cyclic: {filename}"
            ) from error
        authenticated = AuthenticatedFile.open(
            resolved,
            label=f"recovery candidate bind group {group.group_key}",
        )
        try:
            if (
                authenticated.size != group.artifact_bytes
                or authenticated.sha256 != group.artifact_sha256
            ):
                raise ValueError(
                    f"recovery candidate group identity changed after audit: {filename}"
                )
        finally:
            authenticated.close()
        resolved_paths[filename] = resolved

    # The recovery auditor owns the authenticated-root policy, including the
    # parent aliases for recovery-seeded candidates. Re-run it after resolving
    # links so the exact paths above cannot widen that authority.
    recovery_audit.verify_current_identity()
    return resolved_paths


def bind_optional_glm52_adapter_sidecars(
    model: Any,
    *,
    adapter_sidecar_dir: str | Path | None,
    expected_adapter_manifest_sha256: str | None,
    expected_parent_candidate_identity_sha256: str,
    expected_num_experts: int,
) -> Any | None:
    """Bind an optional adapter only when its complete authority pair is present."""

    if (adapter_sidecar_dir is None) != (expected_adapter_manifest_sha256 is None):
        raise ValueError(
            "adapter binding requires both adapter sidecar directory and expected manifest SHA-256"
        )
    if adapter_sidecar_dir is None:
        return None
    binding = importlib.import_module("mlx_vq.quality.glm52_adapter_binding")
    return binding.bind_authenticated_glm52_adapter_sidecars(
        model,
        adapter_sidecar_dir,
        expected_adapter_manifest_sha256=expected_adapter_manifest_sha256,
        expected_parent_candidate_identity_sha256=(
            expected_parent_candidate_identity_sha256
        ),
        expected_num_experts=expected_num_experts,
    )


def add_glm52_adapter_provenance(
    payload: dict[str, Any],
    validated_adapter: Any | None,
) -> None:
    """Add adapter provenance without changing adapter-free reevaluation output."""

    if validated_adapter is None:
        return
    payload["adapter_provenance"] = {
        "adapter_sidecar_dir": str(validated_adapter.adapter_dir),
        "parent_candidate_identity_sha256": (
            validated_adapter.parent_candidate_identity_sha256
        ),
        "teacher_manifest_body_sha256": (
            validated_adapter.teacher_manifest_body_sha256
        ),
        "adapter_manifest_body_sha256": validated_adapter.manifest_body_sha256,
        "adapter_candidate_identity_sha256": (
            validated_adapter.candidate_identity_sha256
        ),
        "release_eligible": validated_adapter.release_eligible,
    }


def _prompt_value(prompt: object, name: str) -> Any:
    if isinstance(prompt, Mapping):
        return prompt.get(name)
    return getattr(prompt, name)


def validate_selection_split(prompts: Iterable[object]) -> dict[str, object]:
    """Reject every calibration authority except the frozen selection split."""

    rows = tuple(prompts)
    if len(rows) != EXPECTED_SELECTION_PROMPTS:
        raise ValueError(f"selection recovery requires exactly {EXPECTED_SELECTION_PROMPTS} prompts")
    prompt_ids: list[str] = []
    position_count = 0
    for row in rows:
        if _prompt_value(row, "split") != "selection":
            raise ValueError("recovery calibration accepts the selection split only")
        if _prompt_value(row, "tuning_eligible") is not True:
            raise ValueError("every selection recovery prompt must be tuning eligible")
        prompt_id = _prompt_value(row, "prompt_id")
        token_ids = _prompt_value(row, "encoded_token_ids")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError("selection prompt IDs must be non-empty strings")
        if not isinstance(token_ids, (list, tuple)) or len(token_ids) < 2:
            raise ValueError("selection prompts must contain at least two encoded token IDs")
        prompt_ids.append(prompt_id)
        position_count += len(token_ids) - 1
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("selection prompt IDs must be unique")
    if position_count != EXPECTED_SELECTION_POSITIONS:
        raise ValueError(
            f"selection recovery requires exactly {EXPECTED_SELECTION_POSITIONS} positions, found {position_count}"
        )
    return {
        "split": "selection",
        "prompt_count": EXPECTED_SELECTION_PROMPTS,
        "position_count": EXPECTED_SELECTION_POSITIONS,
        "holdout_used_for_tuning": False,
        "report_used_for_tuning": False,
    }


def parse_layer_selection(*, layers: int | None, layer_list: str | None) -> tuple[int, ...]:
    if (layers is None) == (layer_list is None):
        raise ValueError("select exactly one of --layers N or --layer-list")
    if layers is not None:
        if layers <= 0 or layers > len(SPARSE_LAYERS):
            raise ValueError(f"--layers must be between 1 and {len(SPARSE_LAYERS)}")
        return SPARSE_LAYERS[:layers]
    assert layer_list is not None
    try:
        selected = tuple(int(item.strip()) for item in layer_list.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("--layer-list must contain comma-separated integers") from error
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("--layer-list must be non-empty and contain no duplicates")
    invalid = sorted(set(selected) - set(SPARSE_LAYERS))
    if invalid:
        raise ValueError(f"--layer-list contains non-sparse GLM-5.2 layers: {invalid}")
    return tuple(sorted(selected))


@dataclass
class _Moment:
    sum_x2: np.ndarray
    score_weighted_sum_x2: np.ndarray
    route_count: int = 0
    score_sum: float = 0.0


@dataclass
class RecoveryStatsAccumulator:
    selected_layers: tuple[int, ...]
    num_experts: int
    total_route_count_by_layer: dict[int, int] = field(default_factory=dict)
    input_dims: dict[tuple[int, str], int] = field(default_factory=dict)
    moments: dict[tuple[int, str, int], _Moment] = field(default_factory=dict)

    def begin_layer(self, *, layer: int, route_ids: np.ndarray) -> None:
        ids = np.asarray(route_ids)
        if ids.ndim != 2 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("route IDs must have integer shape [positions, top_k]")
        if np.any(ids < 0) or np.any(ids >= self.num_experts):
            raise ValueError("route IDs exceed the expert inventory")
        self.total_route_count_by_layer[layer] = int(ids.size)

    def add(
        self,
        *,
        layer: int,
        projection: str,
        expert: int,
        inputs: np.ndarray,
        route_scores: np.ndarray,
    ) -> None:
        if layer not in self.selected_layers or projection not in PROJECTIONS:
            return
        values = np.asarray(inputs, dtype=np.float32)
        scores = np.asarray(route_scores, dtype=np.float32).reshape(-1)
        if values.ndim != 2 or values.shape[0] != scores.size:
            raise ValueError("captured projection inputs and route scores must align")
        if not np.isfinite(values).all() or not np.isfinite(scores).all() or np.any(scores < 0):
            raise ValueError("captured inputs and route scores must be finite and scores non-negative")
        squared = values.astype(np.float64) ** 2
        dimension_key = (layer, projection)
        prior_dim = self.input_dims.setdefault(dimension_key, values.shape[1])
        if prior_dim != values.shape[1]:
            raise ValueError("captured input dimension changed within one layer/projection")
        key = (layer, projection, expert)
        moment = self.moments.get(key)
        if moment is None:
            moment = _Moment(
                sum_x2=np.zeros(values.shape[1], dtype=np.float64),
                score_weighted_sum_x2=np.zeros(values.shape[1], dtype=np.float64),
            )
            self.moments[key] = moment
        if moment.sum_x2.shape != (values.shape[1],):
            raise ValueError("captured input dimension changed within one projection/expert")
        moment.sum_x2 += squared.sum(axis=0)
        moment.score_weighted_sum_x2 += (squared * scores[:, None]).sum(axis=0)
        moment.route_count += values.shape[0]
        moment.score_sum += float(scores.sum(dtype=np.float64))

    def records(self, *, prompt_ids: Sequence[str]) -> tuple[dict[str, object], ...]:
        records: list[dict[str, object]] = []
        for layer in self.selected_layers:
            total = self.total_route_count_by_layer[layer]
            for projection in PROJECTIONS:
                input_dim = self.input_dims.get((layer, projection))
                if input_dim is None:
                    raise ValueError(f"layer {layer} projection {projection} captured no input dimension")
                for expert in range(self.num_experts):
                    moment = self.moments.get((layer, projection, expert))
                    if moment is None:
                        moment = _Moment(
                            sum_x2=np.zeros(input_dim, dtype=np.float64),
                            score_weighted_sum_x2=np.zeros(input_dim, dtype=np.float64),
                        )
                    mean = moment.sum_x2 / moment.route_count if moment.route_count else np.zeros_like(moment.sum_x2)
                    records.append(
                        {
                            "layer": layer,
                            "projection": projection,
                            "expert": expert,
                            "input_dim": int(moment.sum_x2.size),
                            "sum_x2": moment.sum_x2.astype(np.float32),
                            "mean_second_moment": mean.astype(np.float32),
                            "routing_weighted_importance": (moment.sum_x2 / total).astype(np.float32),
                            "router_score_weighted_importance": moment.score_weighted_sum_x2.astype(np.float32),
                            "route_count": moment.route_count,
                            "total_route_count": total,
                            "route_frequency": moment.route_count / total,
                            "router_score_sum": moment.score_sum,
                            "prompt_ids": tuple(prompt_ids),
                            "sample_count": EXPECTED_SELECTION_PROMPTS,
                            "prompt_count": EXPECTED_SELECTION_PROMPTS,
                            "position_count": EXPECTED_SELECTION_POSITIONS,
                        }
                    )
        return tuple(records)


def _capture_projection_stats(
    *,
    accumulator: RecoveryStatsAccumulator,
    layer: int,
    projection: str,
    expert: int,
    values: Any,
    route_scores: np.ndarray,
) -> None:
    """Copy BF16 MLX projection inputs to host FP32 before accumulation."""

    mx = importlib.import_module("mlx.core")
    inputs = np.array(values.astype(mx.float32), copy=False)
    accumulator.add(
        layer=layer,
        projection=projection,
        expert=expert,
        inputs=inputs,
        route_scores=route_scores,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{uuid4().hex}")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_recovery_stats(
    *,
    output_dir: str | Path,
    accumulator: RecoveryStatsAccumulator,
    prompts: Sequence[object],
    source_authority: Mapping[str, object],
) -> dict[str, object]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    split_evidence = validate_selection_split(prompts)
    prompt_ids = tuple(str(_prompt_value(prompt, "prompt_id")) for prompt in prompts)
    entries: list[dict[str, object]] = []
    for record in accumulator.records(prompt_ids=prompt_ids):
        filename = (
            f"layer-{record['layer']:05d}-{record['projection']}-"
            f"expert-{record['expert']:03d}.npz"
        )
        path = root / filename
        _write_npz_atomic(
            path,
            sum_x2=record["sum_x2"],
            mean_second_moment=record["mean_second_moment"],
            routing_weighted_importance=record["routing_weighted_importance"],
            router_score_weighted_importance=record["router_score_weighted_importance"],
        )
        entries.append(
            {
                key: value
                for key, value in record.items()
                if not isinstance(value, np.ndarray)
            }
            | {"path": filename, "sha256": _sha256_file(path)}
        )
    manifest = {
        "schema_version": 1,
        "record_type": "glm52_recovery_stats_manifest",
        "status": "complete",
        "method": "selection_only_diagonal_hessian_second_moments_v1",
        "selected_layers": list(accumulator.selected_layers),
        "projections": list(PROJECTIONS),
        "split_evidence": split_evidence,
        "source_authority": dict(source_authority),
        "sample_count": EXPECTED_SELECTION_PROMPTS,
        "prompt_count": EXPECTED_SELECTION_PROMPTS,
        "position_count": EXPECTED_SELECTION_POSITIONS,
        "route_count_by_layer": {
            str(layer): count for layer, count in sorted(accumulator.total_route_count_by_layer.items())
        },
        "entry_count": len(entries),
        "entries": entries,
    }
    manifest_path = root / "glm52-recovery-stats-manifest.json"
    temporary = manifest_path.with_name(f".{manifest_path.name}.partial-{uuid4().hex}")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    return manifest


def collect_authenticated_recovery_stats(
    *,
    contract: object,
    snapshot_dir: str | Path,
    non_vq_package_dir: str | Path,
    profile_path: str | Path,
    output_dir: str | Path,
    selected_layers: tuple[int, ...],
    heavy_lock_path: str | Path = ".keep-heavy-job.lock",
) -> dict[str, object]:
    """Run the authenticated layer-major source teacher with read-only hooks."""

    producer = importlib.import_module("mlx_vq.quality.glm52_teacher_cache_producer")
    source_api = importlib.import_module("mlx_vq.models.glm52_source_teacher")
    selection = tuple(prompt for prompt in contract.prompts if prompt.split == "selection")
    split_evidence = validate_selection_split(selection)
    output_root = Path(output_dir).resolve()
    checkpoint_dir = output_root / "checkpoints"
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise ValueError("recovery collection does not resume hidden-only checkpoints; use a fresh output directory")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    lock_provider = producer.FileProducerLockProvider()

    with lock_provider.heavy_job_lock(Path(heavy_lock_path)), lock_provider.run_lock(
        output_root / ".glm52-recovery.lock"
    ):
        workload = producer.load_glm52_source_teacher_workload(
            snapshot_dir=snapshot_dir,
            non_vq_package_dir=non_vq_package_dir,
            profile_path=profile_path,
            contract=contract,
        )
        accumulator = RecoveryStatsAccumulator(
            selected_layers=selected_layers,
            num_experts=int(workload.model.args.n_routed_experts),
        )
        resolver_layers: dict[int, int] = {}
        original_resolver_for_layer = workload.expert_resolver_for_layer

        def resolver_for_layer(layer: int) -> Any:
            resolver = original_resolver_for_layer(layer)
            resolver_layers[id(resolver)] = layer
            return resolver

        workload = producer.GLM52SourceTeacherWorkload(
            model=workload.model,
            expert_resolver_for_layer=resolver_for_layer,
            expected_num_layers=workload.expected_num_layers,
            source_blob_inventory=workload.source_blob_inventory,
        )
        original_stream = source_api.stream_selected_expert_moe
        original_projection = source_api._bf16_projection

        def observed_stream(hidden: Any, **kwargs: Any) -> Any:
            resolver = kwargs["expert_weight_resolver"]
            layer = resolver_layers[id(resolver)]
            if layer not in selected_layers:
                return original_stream(hidden, **kwargs)
            state: dict[str, Any] = {"projection_calls": 0}
            original_gate = kwargs["gate"]

            def observed_gate(values: Any) -> tuple[Any, Any]:
                ids, scores = original_gate(values)
                ids_np = np.array(ids).astype(np.int32, copy=False)
                scores_np = np.array(scores).astype(np.float32, copy=False)
                state["ids"] = ids_np
                state["scores"] = scores_np
                state["experts"] = sorted(int(value) for value in np.unique(ids_np))
                accumulator.begin_layer(layer=layer, route_ids=ids_np)
                return ids, scores

            def observed_projection(values: Any, source: Any, *, projection: str, **projection_kwargs: Any) -> Any:
                call = int(state["projection_calls"])
                expert = state["experts"][call // 3]
                expected_projection = PROJECTIONS[call % 3]
                if projection != expected_projection:
                    raise AssertionError("source streamed projection order changed")
                ids_np = state["ids"]
                scores_np = state["scores"]
                route_scores = scores_np.reshape(-1)[ids_np.reshape(-1) == expert]
                _capture_projection_stats(
                    accumulator=accumulator,
                    layer=layer,
                    projection=projection,
                    expert=expert,
                    values=values,
                    route_scores=route_scores,
                )
                state["projection_calls"] = call + 1
                return original_projection(values, source, projection=projection, **projection_kwargs)

            kwargs["gate"] = observed_gate
            source_api._bf16_projection = observed_projection
            try:
                return original_stream(hidden, **kwargs)
            finally:
                source_api._bf16_projection = original_projection

        source_api.stream_selected_expert_moe = observed_stream
        try:
            producer.run_glm52_source_teacher_to_sink(
                workload,
                selection,
                pending_prompt_indices=tuple(range(len(selection))),
                checkpoint_dir=checkpoint_dir,
                checkpoint_identities=producer.checkpoint_identities_for_contract(contract),
                phase_boundary=lambda: None,
                sink=lambda _index, _logits: None,
                capture_route_trace=False,
            )
        finally:
            source_api.stream_selected_expert_moe = original_stream
            source_api._bf16_projection = original_projection
            inventory = workload.source_blob_inventory
            if inventory is not None:
                inventory.close()

    expected_routes = EXPECTED_SELECTION_POSITIONS * 8
    for layer in selected_layers:
        if accumulator.total_route_count_by_layer.get(layer) != expected_routes:
            raise ValueError(f"layer {layer} did not capture exactly {expected_routes} selection routes")
    return write_recovery_stats(
        output_dir=output_root,
        accumulator=accumulator,
        prompts=selection,
        source_authority={
            "model_id": contract.source["model_id"],
            "revision": contract.source["revision"],
            "profile": contract.source["profile"],
            "config_sha256": contract.source["config_sha256"],
            "index_sha256": contract.source["index_sha256"],
            "authenticated_source_teacher": True,
            "layer_major_streaming": True,
            "split_evidence": split_evidence,
        },
    )


def _load_contract(prompt_pack_path: Path, identities_path: Path) -> object:
    cache_api = importlib.import_module("mlx_vq.quality.glm52_teacher_cache")
    identities = json.loads(identities_path.read_text())
    fields = identities.get("teacher_cache_contract", identities)
    return cache_api.GLM52TeacherCacheContract.from_frozen_prompt_pack(
        prompt_pack_path,
        producer=fields["producer"],
        source_evidence=fields["source_evidence"],
        non_vq_package=fields["non_vq_package"],
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--prompt-pack-json", required=True)
    parser.add_argument("--artifact-identities-json", required=True)
    parser.add_argument("--non-vq-package-dir", required=True)
    parser.add_argument("--profile-path", required=True)
    parser.add_argument("--output-dir", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--layers", type=int)
    selection.add_argument("--layer-list")
    parser.add_argument("--heavy-lock-path", default=".keep-heavy-job.lock")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    contract = _load_contract(Path(args.prompt_pack_json), Path(args.artifact_identities_json))
    manifest = collect_authenticated_recovery_stats(
        contract=contract,
        snapshot_dir=args.snapshot_dir,
        non_vq_package_dir=args.non_vq_package_dir,
        profile_path=args.profile_path,
        output_dir=args.output_dir,
        selected_layers=parse_layer_selection(layers=args.layers, layer_list=args.layer_list),
        heavy_lock_path=args.heavy_lock_path,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
