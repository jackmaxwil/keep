#!/usr/bin/env python3
"""Run GLM-5.2 Lane-B wave-1 selection-only recovery."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import importlib.util
import json
import os
import stat
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

import numpy as np
from safetensors import safe_open

from mlx_vq.codebook.e8 import decode_weight_matrix
from mlx_vq.io.authenticated_artifacts import (
    AuthenticatedFile,
    clone_or_copy_authenticated,
)
from mlx_vq.convert.glm52_recovery_materialize import (
    LDLQ_LEVER,
    RECOVERY_LEVER,
    RecoveryPolicy,
    _SeedFamilyArgumentParser,
    _authenticate_source_inputs,
    _load_authenticated_stats,
    _load_json_bytes,
    _read_stable_bytes,
    _snapshot_seed_groups,
    _source_lineage,
    _validate_seed_manifest,
    materialize_groups_from_source,
)


def _load_recovery_api() -> Any:
    name = "mlx_vq.quality.glm52_recovery"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = Path(__file__).resolve().parents[1] / "src/mlx_vq/quality/glm52_recovery.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the GLM-5.2 recovery collector")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_recovery_audit_api() -> Any:
    name = "mlx_vq.validate.glm52_recovery_artifact"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = (
        Path(__file__).resolve().parents[1]
        / "src/mlx_vq/validate/glm52_recovery_artifact.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the GLM-5.2 recovery artifact audit")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_recovery_profile(path: str | Path) -> Any:
    profiles = importlib.import_module("mlx_vq.models.profiles")
    return profiles.load_profile(path)


recovery_api = _load_recovery_api()


class _Wave1ArgumentParser(_SeedFamilyArgumentParser):
    def parse_args(
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        parsed = super().parse_args(args, namespace)
        adapter_dir = getattr(parsed, "adapter_sidecar_dir", None)
        adapter_sha256 = getattr(parsed, "expected_adapter_manifest_sha256", None)
        if (adapter_dir is None) != (adapter_sha256 is None):
            self.error(
                "--adapter-sidecar-dir and --expected-adapter-manifest-sha256 "
                "must be provided together"
            )
        return parsed


@contextmanager
def _cooperating_file_lock(path: str | Path):
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"lock path must be a regular file: {lock_path}")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _authenticate_stats_manifest(
    stats_dir: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    path = Path(stats_dir) / "glm52-recovery-stats-manifest.json"
    authenticated = _read_stable_bytes(path, label="recovery stats manifest")
    if authenticated.sha256 != expected_sha256:
        raise ValueError(
            "stats manifest SHA-256 does not match external accepted authority"
        )
    return _load_json_bytes(authenticated.payload, label="recovery stats manifest")


def _load_bound_attribution(
    path: str | Path,
    *,
    expected_attribution_sha256: str,
    expected_stats_manifest_sha256: str,
) -> dict[str, Any]:
    authenticated = _read_stable_bytes(Path(path), label="layer attribution")
    if authenticated.sha256 != expected_attribution_sha256:
        raise ValueError(
            "attribution SHA-256 does not match external accepted authority"
        )
    attribution = _load_json_bytes(authenticated.payload, label="layer attribution")
    if attribution.get("stats_manifest_sha256") != expected_stats_manifest_sha256:
        raise ValueError(
            "attribution stats manifest SHA-256 does not match external accepted authority"
        )
    return attribution


def _prepare_authenticated_stats_root(
    stats_dir: str | Path,
    *,
    expected_sha256: str,
    collector: Callable[[], object],
    event_sink: Callable[[str], object] | None = None,
) -> dict[str, Any]:
    root = Path(stats_dir)
    manifest_path = root / "glm52-recovery-stats-manifest.json"
    if not manifest_path.exists():
        if root.exists():
            raise ValueError(
                "stats root exists without an accepted complete stats manifest"
            )
        collector()
        if not manifest_path.exists():
            raise ValueError("stats collector did not produce the recovery stats manifest")
    if event_sink is not None:
        event_sink("authenticate")
    return _authenticate_stats_manifest(root, expected_sha256=expected_sha256)


def _write_json_atomic(path: str | Path, payload: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial-{uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


class _RecoveredGroupSnapshots:
    def __init__(
        self,
        paths: Mapping[tuple[int, str], Path],
        temporary_dir: tempfile.TemporaryDirectory[str],
    ) -> None:
        self.paths = dict(paths)
        self._temporary_dir = temporary_dir

    def close(self) -> None:
        self._temporary_dir.cleanup()


def _require_complete_recovery_audit(audit: object) -> None:
    if getattr(audit, "audit_pass", None) is not True:
        raise ValueError("recovery artifact audit did not pass")
    if getattr(audit, "group_count", None) != 225:
        raise ValueError(
            "recovery artifact audit did not prove the complete 225-group tree"
        )
    replacement_count = getattr(audit, "replacement_group_count", None)
    inherited_count = getattr(audit, "inherited_group_count", None)
    groups = getattr(audit, "groups", None)
    if (
        type(replacement_count) is not int
        or type(inherited_count) is not int
        or replacement_count + inherited_count != 225
        or not isinstance(groups, Sequence)
        or isinstance(groups, (str, bytes, bytearray))
        or len(groups) != 225
    ):
        raise ValueError(
            "recovery artifact audit did not prove a complete candidate"
        )


def _load_authenticated_recovery_policy(path: str | Path) -> dict[str, Any]:
    authenticated = _read_stable_bytes(
        Path(path), label="expected recovery policy"
    )
    return _load_json_bytes(
        authenticated.payload, label="expected recovery policy"
    )


def _authenticate_raw_recovery_manifest(
    recovery_conversion_dir: str | Path,
    *,
    expected_sha256: str,
) -> str:
    authenticated = _read_stable_bytes(
        Path(recovery_conversion_dir) / "conversion-manifest.json",
        label="raw recovery manifest",
    )
    if authenticated.sha256 != expected_sha256:
        raise ValueError(
            "raw recovery manifest SHA-256 does not match external authority"
        )
    return authenticated.sha256


def _run_recovery_audit(
    *,
    recovery_conversion_dir: str | Path,
    profile_path: str | Path,
    expected_seed_manifest_sha256: str,
    expected_stats_manifest_sha256: str,
    expected_full_source_blob_inventory_sha256: str,
    expected_routed_source_blob_inventory_sha256: str,
    expected_recovery_lever: str,
    expected_recovery_policy_json: str | Path,
    accepted_composite_audit_json: str | Path,
    expected_composite_audit_sha256: str,
) -> object:
    policy = _load_authenticated_recovery_policy(
        expected_recovery_policy_json
    )
    profile = _load_recovery_profile(profile_path)
    audit = _load_recovery_audit_api().audit_glm52_recovery_mixed_artifact(
        recovery_conversion_dir,
        profile=profile,
        expected_seed_manifest_sha256=expected_seed_manifest_sha256,
        expected_stats_manifest_sha256=expected_stats_manifest_sha256,
        expected_full_source_blob_inventory_sha256=(
            expected_full_source_blob_inventory_sha256
        ),
        expected_routed_source_blob_inventory_sha256=(
            expected_routed_source_blob_inventory_sha256
        ),
        expected_recovery_lever=expected_recovery_lever,
        expected_recovery_policy=policy,
        accepted_composite_audit_json=accepted_composite_audit_json,
        expected_composite_audit_sha256=expected_composite_audit_sha256,
    )
    _require_complete_recovery_audit(audit)
    return audit


def _snapshot_recovered_groups(
    *,
    candidate_root: Path,
    audit: object,
    groups: Sequence[tuple[int, str]],
) -> tuple[
    _RecoveredGroupSnapshots,
    dict[tuple[int, str], object],
]:
    records: dict[tuple[int, str], object] = {}
    for record in getattr(audit, "groups"):
        raw_key = getattr(record, "group_key", "")
        if not isinstance(raw_key, str) or ":" not in raw_key:
            continue
        layer_text, projection = raw_key.split(":", 1)
        try:
            key = (int(layer_text), projection)
        except ValueError:
            continue
        records[key] = record
    missing = [key for key in groups if key not in records]
    if missing:
        raise ValueError(
            f"recovery audit is missing selected candidate groups: {missing}"
        )
    temporary_dir = tempfile.TemporaryDirectory(
        prefix="glm52-recovery-attribution-"
    )
    snapshot_root = Path(temporary_dir.name).resolve(strict=True)
    paths: dict[tuple[int, str], Path] = {}
    remaining_bytes = sum(
        int(getattr(records[key], "artifact_bytes")) for key in groups
    )
    try:
        for key in groups:
            record = records[key]
            filename = str(getattr(record, "filename"))
            source = candidate_root / filename
            authenticated = AuthenticatedFile.open(
                source,
                label=f"candidate group {key}",
                allow_resolved_symlink=True,
            )
            try:
                if authenticated.size != getattr(record, "artifact_bytes"):
                    raise ValueError(
                        f"candidate group {key} size does not match recovery audit"
                    )
                if authenticated.sha256 != getattr(record, "artifact_sha256"):
                    raise ValueError(
                        f"candidate group {key} SHA-256 does not match recovery audit"
                    )
                snapshot = snapshot_root / filename
                clone_or_copy_authenticated(
                    authenticated,
                    snapshot,
                    allow_copy_fallback=True,
                    required_free_bytes=remaining_bytes,
                )
                remaining_bytes -= authenticated.size
                paths[key] = snapshot
            finally:
                authenticated.close()
    except BaseException:
        temporary_dir.cleanup()
        raise
    return _RecoveredGroupSnapshots(paths, temporary_dir), records


def audit_recovery_artifact(
    *,
    recovery_conversion_dir: str | Path,
    output_json: str | Path,
    profile_path: str | Path,
    expected_recovery_manifest_sha256: str,
    expected_seed_manifest_sha256: str,
    expected_stats_manifest_sha256: str,
    expected_full_source_blob_inventory_sha256: str,
    expected_routed_source_blob_inventory_sha256: str,
    expected_recovery_lever: str,
    expected_recovery_policy_json: str | Path,
    accepted_composite_audit_json: str | Path,
    expected_composite_audit_sha256: str,
    heavy_lock_path: str | Path = ".keep-heavy-job.lock",
) -> dict[str, object]:
    """Publish one identity-revalidated full recovery audit."""

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _cooperating_file_lock(heavy_lock_path), _cooperating_file_lock(
        output_path.with_suffix(".lock")
    ):
        raw_manifest_sha256 = _authenticate_raw_recovery_manifest(
            recovery_conversion_dir,
            expected_sha256=expected_recovery_manifest_sha256,
        )
        audit = _run_recovery_audit(
            recovery_conversion_dir=recovery_conversion_dir,
            profile_path=profile_path,
            expected_seed_manifest_sha256=expected_seed_manifest_sha256,
            expected_stats_manifest_sha256=expected_stats_manifest_sha256,
            expected_full_source_blob_inventory_sha256=(
                expected_full_source_blob_inventory_sha256
            ),
            expected_routed_source_blob_inventory_sha256=(
                expected_routed_source_blob_inventory_sha256
            ),
            expected_recovery_lever=expected_recovery_lever,
            expected_recovery_policy_json=expected_recovery_policy_json,
            accepted_composite_audit_json=accepted_composite_audit_json,
            expected_composite_audit_sha256=expected_composite_audit_sha256,
        )
        _authenticate_raw_recovery_manifest(
            recovery_conversion_dir,
            expected_sha256=raw_manifest_sha256,
        )
        audit.verify_current_identity()
        payload = {
            **audit.to_dict(),
            "raw_recovery_manifest_sha256": raw_manifest_sha256,
        }
        _write_json_atomic(output_path, payload)
        audit.verify_current_identity()
        _authenticate_raw_recovery_manifest(
            recovery_conversion_dir,
            expected_sha256=raw_manifest_sha256,
        )
        return payload


def attribute_recovery_layers(
    *,
    source_dir: str | Path,
    index_path: str | Path,
    seed_artifact_dir: str | Path | None = None,
    recovery_conversion_dir: str | Path | None = None,
    stats_dir: str | Path,
    output_json: str | Path,
    expected_stats_manifest_sha256: str,
    expected_seed_manifest_sha256: str,
    expected_full_source_blob_inventory_sha256: str,
    expected_routed_source_blob_inventory_sha256: str,
    profile_path: str | Path | None = None,
    expected_recovery_manifest_sha256: str | None = None,
    expected_recovery_lever: str | None = None,
    expected_recovery_policy_json: str | Path | None = None,
    accepted_composite_audit_json: str | Path | None = None,
    expected_composite_audit_sha256: str | None = None,
    importance_kind: str = "routing_weighted_importance",
) -> dict[str, object]:
    """Rank selected layers by selection-weighted artifact reconstruction error."""

    if importance_kind not in {
        "routing_weighted_importance",
        "router_score_weighted_importance",
    }:
        raise ValueError("unsupported importance_kind")
    if (seed_artifact_dir is None) == (recovery_conversion_dir is None):
        raise ValueError(
            "attribution requires exactly one seed or recovery artifact"
        )
    recovery_audit: object | None = None
    raw_recovery_manifest_sha256: str | None = None
    if recovery_conversion_dir is not None:
        recovered_authorities = {
            "profile_path": profile_path,
            "expected_recovery_manifest_sha256": (
                expected_recovery_manifest_sha256
            ),
            "expected_recovery_lever": expected_recovery_lever,
            "expected_recovery_policy_json": expected_recovery_policy_json,
            "accepted_composite_audit_json": accepted_composite_audit_json,
            "expected_composite_audit_sha256": expected_composite_audit_sha256,
        }
        missing = [
            name for name, value in recovered_authorities.items() if value is None
        ]
        if missing:
            raise ValueError(
                "recovered attribution requires complete audit authority: "
                + ", ".join(missing)
            )
        raw_recovery_manifest_sha256 = _authenticate_raw_recovery_manifest(
            recovery_conversion_dir,
            expected_sha256=expected_recovery_manifest_sha256,
        )
        recovery_audit = _run_recovery_audit(
            recovery_conversion_dir=recovery_conversion_dir,
            profile_path=profile_path,  # type: ignore[arg-type]
            expected_seed_manifest_sha256=expected_seed_manifest_sha256,
            expected_stats_manifest_sha256=expected_stats_manifest_sha256,
            expected_full_source_blob_inventory_sha256=(
                expected_full_source_blob_inventory_sha256
            ),
            expected_routed_source_blob_inventory_sha256=(
                expected_routed_source_blob_inventory_sha256
            ),
            expected_recovery_lever=expected_recovery_lever,  # type: ignore[arg-type]
            expected_recovery_policy_json=(
                expected_recovery_policy_json  # type: ignore[arg-type]
            ),
            accepted_composite_audit_json=(
                accepted_composite_audit_json  # type: ignore[arg-type]
            ),
            expected_composite_audit_sha256=(
                expected_composite_audit_sha256  # type: ignore[arg-type]
            ),
        )
        _authenticate_raw_recovery_manifest(
            recovery_conversion_dir,
            expected_sha256=raw_recovery_manifest_sha256,
        )
    stats_root = Path(stats_dir)
    stats_manifest_path = stats_root / "glm52-recovery-stats-manifest.json"
    stats_manifest = _authenticate_stats_manifest(
        stats_root,
        expected_sha256=expected_stats_manifest_sha256,
    )
    expected_evidence = {
        "split": "selection",
        "prompt_count": 22,
        "position_count": 255,
        "holdout_used_for_tuning": False,
        "report_used_for_tuning": False,
    }
    if stats_manifest.get("split_evidence") != expected_evidence:
        raise ValueError("attribution requires frozen selection-only recovery stats")
    selected_layers = tuple(int(layer) for layer in stats_manifest.get("selected_layers", ()))
    if not selected_layers:
        raise ValueError("stats manifest has no selected layers")
    source_root = Path(source_dir)
    authority = stats_manifest.get("source_authority")
    if not isinstance(authority, Mapping):
        raise ValueError("recovery stats source authority must be an object")
    source_lineage = _source_lineage(authority)
    selected_groups = tuple(
        (layer, projection)
        for layer in selected_layers
        for projection in recovery_api.PROJECTIONS
    )
    recovered_records: dict[tuple[int, str], object] | None = None
    if recovery_audit is None:
        seed_root = Path(seed_artifact_dir)  # type: ignore[arg-type]
        authenticated_seed_manifest = _read_stable_bytes(
            seed_root / "conversion-manifest.json",
            label="accepted seed manifest",
        )
        if authenticated_seed_manifest.sha256 != expected_seed_manifest_sha256:
            raise ValueError(
                "seed manifest SHA-256 does not match external accepted authority"
            )
        seed_manifest = _load_json_bytes(
            authenticated_seed_manifest.payload,
            label="accepted seed manifest",
        )
        seed_records = _validate_seed_manifest(
            seed_manifest,
            expected_source_lineage=source_lineage,
        )
        seed_snapshots = _snapshot_seed_groups(
            seed_root=seed_root,
            records=seed_records,
            groups=selected_groups,
        )
    else:
        seed_root = Path(getattr(recovery_audit, "recovery_dir")) / "artifact"
        seed_snapshots, recovered_records = _snapshot_recovered_groups(
            candidate_root=seed_root,
            audit=recovery_audit,
            groups=selected_groups,
        )
    group_specs: dict[tuple[int, str], tuple[int, int]] = {}
    source_inventory = None
    try:
        for layer, projection in selected_groups:
            snapshot_path = seed_snapshots.paths[(layer, projection)]
            prefix = f"model.layers.{layer}.mlp.switch_mlp.{projection}"
            with safe_open(snapshot_path, framework="np") as handle:
                codes_shape = handle.get_tensor(f"{prefix}.codes").shape
            if len(codes_shape) != 3:
                raise ValueError(f"seed group {(layer, projection)} codes must have rank 3")
            experts, _out_dim, codeword_count = codes_shape
            group_specs[(layer, projection)] = (experts, codeword_count * 8)
        authenticated_stats = _load_authenticated_stats(
            stats_dir=stats_root,
            stats_manifest=stats_manifest,
            group_specs=group_specs,
        )
        _index, source_inventory, _lineage, _verification = _authenticate_source_inputs(
            source_root=source_root,
            index_path=Path(index_path),
            authority=authority,
            expected_full_source_blob_inventory_sha256=(
                expected_full_source_blob_inventory_sha256
            ),
            expected_routed_source_blob_inventory_sha256=(
                expected_routed_source_blob_inventory_sha256
            ),
        )
        if recovery_audit is not None:
            recovery_audit.verify_current_identity()
    except BaseException:
        if source_inventory is not None:
            source_inventory.close()
        seed_snapshots.close()
        raise
    try:
        from mlx_vq.convert.nvfp4 import (
            read_modelopt_nvfp4_weight,
            resolve_modelopt_nvfp4_weight_bundle,
        )
        entries = {
            (int(item["layer"]), str(item["projection"]), int(item["expert"])): item
            for item in stats_manifest["entries"]
        }
    except BaseException:
        source_inventory.close()
        seed_snapshots.close()
        raise
    group_records: list[dict[str, object]] = []
    try:
        with source_inventory.modelopt_reader_guard():
            for layer, projection in selected_groups:
                filename = f"layer-{layer:05d}-{projection}.safetensors"
                seed_path = seed_snapshots.paths[(layer, projection)]
                prefix = f"model.layers.{layer}.mlp.switch_mlp.{projection}"
                with safe_open(seed_path, framework="np") as handle:
                    codes = handle.get_tensor(f"{prefix}.codes")
                    scales = handle.get_tensor(f"{prefix}.scales")
                    codebook = handle.get_tensor("model.vq_codebook.e8")
                importance = authenticated_stats[(layer, projection)]
                expert_errors: list[float] = []
                expert_routes: list[int] = []
                for expert in range(codes.shape[0]):
                    source_name = (
                        f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
                    )
                    bundle = resolve_modelopt_nvfp4_weight_bundle(
                        source_inventory.weight_map,
                        source_name,
                    )
                    source = read_modelopt_nvfp4_weight(source_inventory.blob_root, bundle)
                    reconstructed = decode_weight_matrix(
                        codes[expert], scales[expert], code_bits=8, codebook=codebook
                    )
                    error = float(
                        np.sum(
                            (source.astype(np.float64) - reconstructed.astype(np.float64)) ** 2
                            * importance[expert].astype(np.float64)[None, :],
                            dtype=np.float64,
                        )
                    )
                    expert_errors.append(error)
                    expert_routes.append(int(entries[(layer, projection, expert)]["route_count"]))
                    del source
                    del reconstructed
                group_record = {
                    "layer": layer,
                    "projection": projection,
                    "weighted_reconstruction_error": float(sum(expert_errors)),
                    "routed_expert_count": sum(count > 0 for count in expert_routes),
                    "route_count": sum(expert_routes),
                }
                if recovered_records is None:
                    group_record.update(
                        {
                            "seed_group_path": str(seed_root / filename),
                            "seed_group_sha256": seed_records[(layer, projection)][
                                "artifact_sha256"
                            ],
                        }
                    )
                else:
                    recovered_record = recovered_records[(layer, projection)]
                    group_record.update(
                        {
                            "recovery_group_path": str(seed_root / filename),
                            "recovery_group_sha256": getattr(
                                recovered_record, "artifact_sha256"
                            ),
                        }
                    )
                group_records.append(group_record)
        source_inventory.verify_after_forward()
    finally:
        source_inventory.close()
        seed_snapshots.close()

    layer_errors = {
        layer: sum(
            float(record["weighted_reconstruction_error"])
            for record in group_records
            if record["layer"] == layer
        )
        for layer in selected_layers
    }
    ranked_layers = sorted(layer_errors, key=lambda layer: (-layer_errors[layer], layer))
    payload = {
        "schema_version": 1,
        "record_type": "glm52_recovery_layer_attribution",
        "status": "complete",
        "importance_kind": importance_kind,
        "split_evidence": expected_evidence,
        "stats_manifest_path": str(stats_manifest_path),
        "stats_manifest_sha256": expected_stats_manifest_sha256,
        "selected_layers": list(selected_layers),
        "ranked_layers": [
            {"rank": rank, "layer": layer, "weighted_reconstruction_error": layer_errors[layer]}
            for rank, layer in enumerate(ranked_layers, start=1)
        ],
        "groups": group_records,
    }
    if recovery_audit is not None:
        payload.update(
            {
                "source_artifact_kind": "recovery_mixed",
                "raw_recovery_manifest_sha256": (
                    raw_recovery_manifest_sha256
                ),
                "candidate_identity_sha256": getattr(
                    recovery_audit, "candidate_identity_sha256"
                ),
                "baseline_composite_identity_sha256": getattr(
                    recovery_audit,
                    "accepted_baseline_composite_identity_sha256",
                ),
            }
        )
    _write_json_atomic(output_json, payload)
    if recovery_audit is not None:
        recovery_audit.verify_current_identity()
        _authenticate_raw_recovery_manifest(
            recovery_conversion_dir,  # type: ignore[arg-type]
            expected_sha256=raw_recovery_manifest_sha256,  # type: ignore[arg-type]
        )
    return payload


def _worst_layer_groups(
    attribution: Mapping[str, object], *, count: int
) -> tuple[tuple[int, str], ...]:
    ranked = attribution.get("ranked_layers")
    if not isinstance(ranked, list) or count <= 0 or count > len(ranked):
        raise ValueError("worst-layer-count is outside the attributed layer inventory")
    layers = tuple(int(record["layer"]) for record in ranked[:count])
    return tuple((layer, projection) for layer in layers for projection in recovery_api.PROJECTIONS)


def _e8p_worst_layer_code_bits(
    attribution: Mapping[str, object], *, count: int
) -> dict[int, int]:
    if count == 0:
        return {}
    ranked = attribution.get("ranked_layers")
    if not isinstance(ranked, list) or count < 0 or count > len(ranked):
        raise ValueError("e8p worst-layer count is outside the attributed layer inventory")
    return {int(record["layer"]): 16 for record in ranked[:count]}


def _validate_e8p_campaign_count(
    count: int, *, worst_layer_count: int | None = None
) -> None:
    if count < 0 or count > 16:
        raise ValueError("--e8p-worst-layer-count must be between 0 and at most 16")
    if worst_layer_count is not None and count > worst_layer_count:
        raise ValueError(
            "--e8p-worst-layer-count cannot exceed the selected worst-layer count"
        )


def build_wave1_evidence(
    *,
    stats_dir: str | Path,
    attribution_json: str | Path,
    conversion_dir: str | Path,
    output_json: str | Path,
) -> dict[str, object]:
    stats_manifest_path = Path(stats_dir) / "glm52-recovery-stats-manifest.json"
    conversion_manifest_path = Path(conversion_dir) / "conversion-manifest.json"
    stats = _load_json(stats_manifest_path, label="recovery stats manifest")
    attribution = _load_json(attribution_json, label="layer attribution")
    conversion = _load_json(conversion_manifest_path, label="conversion manifest")
    payload = {
        "schema_version": 1,
        "record_type": "glm52_recovery_wave1_evidence",
        "status": "candidate_ready_for_re_evaluation",
        "split_evidence": stats["split_evidence"],
        "stats_manifest": {"path": str(stats_manifest_path), "sha256": _sha256_file(stats_manifest_path)},
        "attribution": {"path": str(attribution_json), "sha256": _sha256_file(attribution_json)},
        "conversion_manifest": {
            "path": str(conversion_manifest_path),
            "sha256": _sha256_file(conversion_manifest_path),
        },
        "selected_groups": conversion["selected_groups"],
        "recovered_artifact_dir": conversion["mixed_artifact"]["output_dir"],
        "candidate_reevaluation_required": True,
        "holdout_used_for_tuning": False,
        "report_used_for_tuning": False,
    }
    _write_json_atomic(output_json, payload)
    return payload


def reevaluate_recovery_candidate(
    *,
    profile_path: str | Path,
    config_path: str | Path,
    source_index_path: str | Path,
    tokenizer_dir: str | Path,
    tokenizer_readiness_json: str | Path,
    family_policy_json: str | Path,
    prompt_pack_json: str | Path,
    teacher_cache_root: str | Path,
    non_vq_artifact_dir: str | Path,
    non_vq_evidence_json: str | Path,
    accepted_routed_artifact_dir: str | Path,
    accepted_composite_audit_json: str | Path,
    accepted_materialization_runs_jsonl: str | Path,
    accepted_full_bind_preflight_json: str | Path,
    recovery_conversion_dir: str | Path,
    output_json: str | Path,
    expected_seed_manifest_sha256: str,
    expected_stats_manifest_sha256: str,
    expected_full_source_blob_inventory_sha256: str,
    expected_routed_source_blob_inventory_sha256: str,
    expected_recovery_lever: str,
    expected_recovery_policy_json: str | Path,
    expected_composite_audit_sha256: str,
    adapter_sidecar_dir: str | Path | None = None,
    expected_adapter_manifest_sha256: str | None = None,
    heavy_lock_path: str | Path = ".keep-heavy-job.lock",
) -> dict[str, object]:
    """Evaluate a mixed recovery artifact without weakening accepted identity gates."""

    if (adapter_sidecar_dir is None) != (expected_adapter_manifest_sha256 is None):
        raise ValueError(
            "adapter reevaluation requires both adapter sidecar directory and expected manifest SHA-256"
        )
    expected_recovery_policy = _load_json(
        expected_recovery_policy_json,
        label="expected recovery policy",
    )
    profile = _load_recovery_profile(profile_path)
    recovery_audit_api = _load_recovery_audit_api()
    rows: list[dict[str, Any]] = []
    validated_adapter = None
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with _cooperating_file_lock(heavy_lock_path), _cooperating_file_lock(
        Path(output_json).with_suffix(".lock")
    ):
        recovery_audit = recovery_audit_api.audit_glm52_recovery_mixed_artifact(
            recovery_conversion_dir,
            profile=profile,
            expected_seed_manifest_sha256=expected_seed_manifest_sha256,
            expected_stats_manifest_sha256=expected_stats_manifest_sha256,
            expected_full_source_blob_inventory_sha256=(
                expected_full_source_blob_inventory_sha256
            ),
            expected_routed_source_blob_inventory_sha256=(
                expected_routed_source_blob_inventory_sha256
            ),
            expected_recovery_lever=expected_recovery_lever,
            expected_recovery_policy=expected_recovery_policy,
            accepted_composite_audit_json=accepted_composite_audit_json,
            expected_composite_audit_sha256=expected_composite_audit_sha256,
        )
        if recovery_audit.audit_pass is not True:
            raise ValueError("recovery artifact audit did not pass")
        if recovery_audit.group_count != 225:
            raise ValueError(
                "recovery artifact audit did not prove the complete 225-group tree"
            )
        recovered_layers = tuple(recovery_audit.complete_replacement_layer_ids)
        recovered_groups = tuple(
            group.group_key
            for group in recovery_audit.groups
            if group.classification == "replacement"
        )
        expected_groups = {
            f"{layer}:{projection}"
            for layer in recovered_layers
            for projection in recovery_api.PROJECTIONS
        }
        if not recovered_layers or set(recovered_groups) != expected_groups:
            raise ValueError(
                "recovery artifact audit did not prove complete replacement groups per layer"
            )
        recovered_artifact_dir = Path(recovery_audit.recovery_dir) / "artifact"
        candidate = importlib.import_module("mlx_vq.quality.glm52_candidate_eval")
        composite = importlib.import_module("mlx_vq.models.glm52_composite_loader")
        adapter = importlib.import_module("mlx_vq.models.glm52_vq_adapter")
        readiness = _load_json(tokenizer_readiness_json, label="tokenizer readiness")
        prompt = readiness.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("tokenizer readiness has no authenticated prompt")
        teacher_contract = candidate._contract_from_cache_manifest(
            teacher_cache_root,
            prompt_pack_path=prompt_pack_json,
            candidate=False,
        )
        candidate._strict_audit(
            teacher_cache_root,
            contract=teacher_contract,
            label="teacher",
            allow_non_release=True,
        )
        validated = composite.validate_glm52_production_inputs(
            profile_path=profile_path,
            config_path=config_path,
            source_index_path=source_index_path,
            tokenizer_dir=tokenizer_dir,
            tokenizer_readiness_json=tokenizer_readiness_json,
            family_policy_json=family_policy_json,
            non_vq_artifact_dir=non_vq_artifact_dir,
            non_vq_evidence_json=non_vq_evidence_json,
            routed_artifact_dir=accepted_routed_artifact_dir,
            composite_audit_json=accepted_composite_audit_json,
            materialization_runs_jsonl=accepted_materialization_runs_jsonl,
            full_bind_preflight_json=accepted_full_bind_preflight_json,
            model_id=teacher_contract.source["model_id"],
            revision=teacher_contract.source["revision"],
            prompt=prompt,
        )
        model, load_report = composite.load_authenticated_glm52_composite(validated)
        candidate_group_paths = (
            recovery_api.authenticated_recovery_candidate_group_paths(recovery_audit)
        )
        candidate_layers = tuple(
            sorted(
                {
                    int(group.group_key.split(":", 1)[0])
                    for group in recovery_audit.groups
                }
            )
        )
        rebound = adapter.bind_glm52_vq_experts_from_paths(
            model,
            candidate_group_paths,
            layers=candidate_layers,
            profile=validated.profile,
            strict=True,
        )
        if rebound != candidate_layers:
            raise ValueError("recovery candidate did not bind every audited routed layer")
        recovery_audit.verify_current_identity()
        if adapter_sidecar_dir is not None:
            validated_adapter = recovery_api.bind_optional_glm52_adapter_sidecars(
                model,
                adapter_sidecar_dir=adapter_sidecar_dir,
                expected_adapter_manifest_sha256=(
                    expected_adapter_manifest_sha256
                ),
                expected_parent_candidate_identity_sha256=(
                    recovery_audit.candidate_identity_sha256
                ),
                expected_num_experts=validated.profile.num_experts,
            )
            recovery_audit.verify_current_identity()
        workload = candidate.GLM52CandidateWorkload(
            model=model,
            validated_inputs=validated,
            load_report=load_report,
        )

        def compare_sink(prompt_index: int, candidate_logits: np.ndarray) -> None:
            prompt_contract = teacher_contract.prompts[prompt_index]
            teacher_path = (
                Path(teacher_cache_root)
                / candidate.cache_api.SHARD_DIRECTORY
                / f"{prompt_contract.prompt_id}.safetensors"
            )
            teacher_logits = candidate._read_f32_logits(
                teacher_path,
                expected_shape=(
                    prompt_contract.token_count - 1,
                    teacher_contract.vocab_size,
                ),
            )
            rows.append(
                candidate._compare_prompt_logits(
                    prompt_contract,
                    teacher_logits=teacher_logits,
                    candidate_logits=candidate_logits,
                )
            )

        candidate.run_glm52_candidate_to_sink(
            workload,
            teacher_contract.prompts,
            pending_prompt_indices=tuple(range(len(teacher_contract.prompts))),
            checkpoint_dir=Path(output_json).parent / "unused-recovery-checkpoints",
            checkpoint_identities=None,
            phase_boundary=lambda: None,
            sink=compare_sink,
        )
        recovery_audit.verify_current_identity()

    frozen = candidate._load_frozen_gate(family_policy_json)
    required_splits = frozen["required_splits"]
    required_domains = frozen["required_domains"]
    metrics = candidate._summarize_rows(rows)
    split_metrics = {
        split: candidate._summarize_rows([row for row in rows if row["split"] == split])
        for split in required_splits
    }
    domain_metrics = {
        domain: candidate._summarize_rows([row for row in rows if row["domain"] == domain])
        for domain in required_domains
    }
    thresholds = frozen["thresholds"]
    checks = {
        "domain_top1": all(
            domain_metrics[domain]["top1_agreement"] >= thresholds["domain_top1_min"]
            for domain in required_domains
        ),
        "mean_kld": metrics["mean_kld"] <= thresholds["mean_kld_max"],
        "mean_ppl_ratio": metrics["mean_ppl_ratio"] <= thresholds["mean_ppl_ratio_max"],
        "p999_kld": metrics["p999_kld"] <= thresholds["p999_kld_max"],
        "top1": metrics["top1_agreement"] >= thresholds["top1_min"],
    }
    payload = {
        "schema_version": 1,
        "record_type": "glm52_recovery_candidate_evaluation",
        "status": "diagnostic_complete",
        "quality_gate_pass": all(checks.values()),
        "evidence_class": "diagnostic_only",
        "recovered_layers": list(recovered_layers),
        "recovered_groups": list(recovered_groups),
        "recovery_artifact_audit": {
            "audit_pass": recovery_audit.audit_pass,
            "candidate_artifact_root": str(recovered_artifact_dir),
            "candidate_identity_sha256": recovery_audit.candidate_identity_sha256,
            "manifest_path": recovery_audit.manifest_path,
            "manifest_body_sha256": recovery_audit.manifest_body_sha256,
            "seed_manifest_sha256": recovery_audit.seed_manifest_sha256,
            "accepted_composite_audit_sha256": (
                recovery_audit.accepted_composite_audit_sha256
            ),
            "group_count": recovery_audit.group_count,
            "replacement_group_count": recovery_audit.replacement_group_count,
            "inherited_group_count": recovery_audit.inherited_group_count,
            "complete_replacement_layer_ids": list(
                recovery_audit.complete_replacement_layer_ids
            ),
            "routed_tensor_payload_bytes": recovery_audit.routed_tensor_payload_bytes,
            "non_routed_tensor_payload_bytes": (
                recovery_audit.non_routed_tensor_payload_bytes
            ),
            "logical_whole_model_tensor_payload_bytes": (
                recovery_audit.logical_whole_model_tensor_payload_bytes
            ),
            "logical_payload_limit_bytes": recovery_audit.logical_payload_limit_bytes,
            "incremental_disk_bytes": recovery_audit.incremental_disk_bytes,
        },
        "accepted_composite_authenticated_before_rebind": True,
        "metrics": metrics,
        "split_metrics": split_metrics,
        "domain_metrics": domain_metrics,
        "checks": checks,
        "thresholds": thresholds,
        "selection_used_for_tuning": True,
        "holdout_used_for_tuning": False,
        "report_used_for_tuning": False,
        "row_count": len(rows),
    }
    recovery_api.add_glm52_adapter_provenance(payload, validated_adapter)
    _write_json_atomic(output_json, payload)
    return payload


def _add_shared_collection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--prompt-pack-json", required=True)
    parser.add_argument("--artifact-identities-json", required=True)
    parser.add_argument("--non-vq-package-dir", required=True)
    parser.add_argument("--profile-path", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--layers", type=int)
    selection.add_argument("--layer-list")
    parser.add_argument("--heavy-lock-path", default=".keep-heavy-job.lock")


def _add_materialization_authority_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-stats-manifest-sha256", required=True)
    parser.add_argument(
        "--expected-attribution-sha256",
        required=True,
        help="externally accepted SHA-256 of the complete layer-attribution JSON",
    )
    parser.add_argument("--expected-seed-manifest-sha256")
    parser.add_argument("--expected-seed-recovery-manifest-sha256")
    parser.add_argument("--expected-seed-recovery-audit-sha256")
    parser.add_argument("--expected-full-source-blob-inventory-sha256", required=True)
    parser.add_argument("--expected-routed-source-blob-inventory-sha256", required=True)
    parser.add_argument("--accepted-composite-audit-json", required=True)
    parser.add_argument("--expected-composite-audit-sha256", required=True)


def _add_materialization_seed_args(parser: argparse.ArgumentParser) -> None:
    seed = parser.add_mutually_exclusive_group(required=True)
    seed.add_argument("--seed-artifact-dir")
    seed.add_argument("--seed-recovery-artifact-dir")
    parser.add_argument("--seed-recovery-audit-json")


def _add_recovery_policy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--recovery-lever",
        choices=(RECOVERY_LEVER, LDLQ_LEVER),
        default=RECOVERY_LEVER,
    )
    parser.add_argument("--scale-search-multipliers")
    parser.add_argument("--rotation-rht-seed")


def _build_parser() -> argparse.ArgumentParser:
    parser = _Wave1ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    collect = commands.add_parser("collect")
    _add_shared_collection_args(collect)
    collect.add_argument("--stats-dir", required=True)

    attribute = commands.add_parser("attribute")
    attribute.add_argument("--source-dir", required=True)
    attribute.add_argument("--index-path", required=True)
    attribute_source = attribute.add_mutually_exclusive_group(required=True)
    attribute_source.add_argument("--seed-artifact-dir")
    attribute_source.add_argument("--recovery-conversion-dir")
    attribute.add_argument("--stats-dir", required=True)
    attribute.add_argument("--output-json", required=True)
    attribute.add_argument("--expected-stats-manifest-sha256", required=True)
    attribute.add_argument("--expected-seed-manifest-sha256", required=True)
    attribute.add_argument("--expected-full-source-blob-inventory-sha256", required=True)
    attribute.add_argument("--expected-routed-source-blob-inventory-sha256", required=True)
    attribute.add_argument("--profile-path")
    attribute.add_argument("--expected-recovery-manifest-sha256")
    attribute.add_argument("--expected-recovery-lever")
    attribute.add_argument("--expected-recovery-policy-json")
    attribute.add_argument("--accepted-composite-audit-json")
    attribute.add_argument("--expected-composite-audit-sha256")
    attribute.add_argument("--heavy-lock-path", default=".keep-heavy-job.lock")
    attribute.add_argument(
        "--importance-kind",
        choices=("routing_weighted_importance", "router_score_weighted_importance"),
        default="routing_weighted_importance",
    )

    audit = commands.add_parser("audit")
    audit.add_argument("--recovery-conversion-dir", required=True)
    audit.add_argument("--output-json", required=True)
    audit.add_argument("--profile-path", required=True)
    audit.add_argument("--expected-recovery-manifest-sha256", required=True)
    audit.add_argument("--expected-seed-manifest-sha256", required=True)
    audit.add_argument("--expected-stats-manifest-sha256", required=True)
    audit.add_argument(
        "--expected-full-source-blob-inventory-sha256", required=True
    )
    audit.add_argument(
        "--expected-routed-source-blob-inventory-sha256", required=True
    )
    audit.add_argument("--expected-recovery-lever", required=True)
    audit.add_argument("--expected-recovery-policy-json", required=True)
    audit.add_argument("--accepted-composite-audit-json", required=True)
    audit.add_argument("--expected-composite-audit-sha256", required=True)
    audit.add_argument("--heavy-lock-path", default=".keep-heavy-job.lock")

    materialize = commands.add_parser("rematerialize")
    materialize.add_argument("--source-dir", required=True)
    materialize.add_argument("--index-path", required=True)
    _add_materialization_seed_args(materialize)
    materialize.add_argument("--stats-dir", required=True)
    materialize.add_argument("--attribution-json", required=True)
    materialize.add_argument("--output-dir", required=True)
    materialize.add_argument("--worst-layer-count", type=int, default=2)
    materialize.add_argument("--e8p-worst-layer-count", type=int, default=0)
    materialize.add_argument("--resume", action="store_true")
    materialize.add_argument("--heavy-lock-path", default=".keep-heavy-job.lock")
    _add_materialization_authority_args(materialize)
    _add_recovery_policy_args(materialize)

    evidence = commands.add_parser("evidence")
    evidence.add_argument("--stats-dir", required=True)
    evidence.add_argument("--attribution-json", required=True)
    evidence.add_argument("--conversion-dir", required=True)
    evidence.add_argument("--output-json", required=True)

    reevaluate = commands.add_parser("reevaluate")
    reevaluate.add_argument("--profile-path", required=True)
    reevaluate.add_argument("--config-path", required=True)
    reevaluate.add_argument("--source-index-path", required=True)
    reevaluate.add_argument("--tokenizer-dir", required=True)
    reevaluate.add_argument("--tokenizer-readiness-json", required=True)
    reevaluate.add_argument("--family-policy-json", required=True)
    reevaluate.add_argument("--prompt-pack-json", required=True)
    reevaluate.add_argument("--teacher-cache-root", required=True)
    reevaluate.add_argument("--non-vq-artifact-dir", required=True)
    reevaluate.add_argument("--non-vq-evidence-json", required=True)
    reevaluate.add_argument("--accepted-routed-artifact-dir", required=True)
    reevaluate.add_argument("--accepted-composite-audit-json", required=True)
    reevaluate.add_argument("--accepted-materialization-runs-jsonl", required=True)
    reevaluate.add_argument("--accepted-full-bind-preflight-json", required=True)
    reevaluate.add_argument("--recovery-conversion-dir", required=True)
    reevaluate.add_argument("--output-json", required=True)
    reevaluate.add_argument("--expected-seed-manifest-sha256", required=True)
    reevaluate.add_argument("--expected-stats-manifest-sha256", required=True)
    reevaluate.add_argument("--expected-full-source-blob-inventory-sha256", required=True)
    reevaluate.add_argument("--expected-routed-source-blob-inventory-sha256", required=True)
    reevaluate.add_argument("--expected-recovery-lever", required=True)
    reevaluate.add_argument("--expected-recovery-policy-json", required=True)
    reevaluate.add_argument("--expected-composite-audit-sha256", required=True)
    reevaluate.add_argument("--adapter-sidecar-dir")
    reevaluate.add_argument("--expected-adapter-manifest-sha256")
    reevaluate.add_argument("--heavy-lock-path", default=".keep-heavy-job.lock")

    run = commands.add_parser("run")
    _add_shared_collection_args(run)
    run.add_argument("--source-dir", required=True)
    run.add_argument("--index-path", required=True)
    _add_materialization_seed_args(run)
    run.add_argument("--output-root", required=True)
    run.add_argument("--attribution-json", required=True)
    run.add_argument("--worst-layer-count", type=int, default=2)
    run.add_argument("--e8p-worst-layer-count", type=int, default=0)
    run.add_argument("--resume-materialization", action="store_true")
    _add_materialization_authority_args(run)
    _add_recovery_policy_args(run)
    return parser


def _collect(args: argparse.Namespace, stats_dir: str | Path) -> dict[str, object]:
    contract = recovery_api._load_contract(
        Path(args.prompt_pack_json), Path(args.artifact_identities_json)
    )
    return recovery_api.collect_authenticated_recovery_stats(
        contract=contract,
        snapshot_dir=args.snapshot_dir,
        non_vq_package_dir=args.non_vq_package_dir,
        profile_path=args.profile_path,
        output_dir=stats_dir,
        selected_layers=recovery_api.parse_layer_selection(
            layers=args.layers, layer_list=args.layer_list
        ),
        heavy_lock_path=args.heavy_lock_path,
    )


def run_cli(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "collect":
        payload = _collect(args, args.stats_dir)
    elif args.command == "attribute":
        if args.recovery_conversion_dir is not None:
            names = (
                "profile_path",
                "expected_recovery_manifest_sha256",
                "expected_recovery_lever",
                "expected_recovery_policy_json",
                "accepted_composite_audit_json",
                "expected_composite_audit_sha256",
            )
            missing = [name for name in names if getattr(args, name) is None]
            if missing:
                raise ValueError(
                    "recovered attribution requires complete audit authority: "
                    + ", ".join(missing)
                )
        with _cooperating_file_lock(args.heavy_lock_path):
            payload = attribute_recovery_layers(
                source_dir=args.source_dir,
                index_path=args.index_path,
                seed_artifact_dir=args.seed_artifact_dir,
                recovery_conversion_dir=args.recovery_conversion_dir,
                stats_dir=args.stats_dir,
                output_json=args.output_json,
                expected_stats_manifest_sha256=(
                    args.expected_stats_manifest_sha256
                ),
                expected_seed_manifest_sha256=(
                    args.expected_seed_manifest_sha256
                ),
                expected_full_source_blob_inventory_sha256=(
                    args.expected_full_source_blob_inventory_sha256
                ),
                expected_routed_source_blob_inventory_sha256=(
                    args.expected_routed_source_blob_inventory_sha256
                ),
                profile_path=args.profile_path,
                expected_recovery_manifest_sha256=(
                    args.expected_recovery_manifest_sha256
                ),
                expected_recovery_lever=args.expected_recovery_lever,
                expected_recovery_policy_json=(
                    args.expected_recovery_policy_json
                ),
                accepted_composite_audit_json=(
                    args.accepted_composite_audit_json
                ),
                expected_composite_audit_sha256=(
                    args.expected_composite_audit_sha256
                ),
                importance_kind=args.importance_kind,
            )
    elif args.command == "audit":
        payload = audit_recovery_artifact(
            recovery_conversion_dir=args.recovery_conversion_dir,
            output_json=args.output_json,
            profile_path=args.profile_path,
            expected_recovery_manifest_sha256=(
                args.expected_recovery_manifest_sha256
            ),
            expected_seed_manifest_sha256=args.expected_seed_manifest_sha256,
            expected_stats_manifest_sha256=args.expected_stats_manifest_sha256,
            expected_full_source_blob_inventory_sha256=(
                args.expected_full_source_blob_inventory_sha256
            ),
            expected_routed_source_blob_inventory_sha256=(
                args.expected_routed_source_blob_inventory_sha256
            ),
            expected_recovery_lever=args.expected_recovery_lever,
            expected_recovery_policy_json=args.expected_recovery_policy_json,
            accepted_composite_audit_json=args.accepted_composite_audit_json,
            expected_composite_audit_sha256=(
                args.expected_composite_audit_sha256
            ),
            heavy_lock_path=args.heavy_lock_path,
        )
    elif args.command == "rematerialize":
        _validate_e8p_campaign_count(
            args.e8p_worst_layer_count,
            worst_layer_count=args.worst_layer_count,
        )
        _authenticate_stats_manifest(
            args.stats_dir,
            expected_sha256=args.expected_stats_manifest_sha256,
        )
        attribution = _load_bound_attribution(
            args.attribution_json,
            expected_attribution_sha256=args.expected_attribution_sha256,
            expected_stats_manifest_sha256=args.expected_stats_manifest_sha256,
        )
        payload = materialize_groups_from_source(
            source_dir=args.source_dir,
            index_path=args.index_path,
            seed_artifact_dir=args.seed_artifact_dir,
            seed_recovery_artifact_dir=args.seed_recovery_artifact_dir,
            seed_recovery_audit_json=args.seed_recovery_audit_json,
            stats_dir=args.stats_dir,
            output_dir=args.output_dir,
            groups=_worst_layer_groups(attribution, count=args.worst_layer_count),
            layer_code_bits=_e8p_worst_layer_code_bits(
                attribution, count=args.e8p_worst_layer_count
            ),
            resume=args.resume,
            expected_stats_manifest_sha256=args.expected_stats_manifest_sha256,
            expected_seed_manifest_sha256=args.expected_seed_manifest_sha256,
            expected_seed_recovery_manifest_sha256=(
                args.expected_seed_recovery_manifest_sha256
            ),
            expected_seed_recovery_audit_sha256=(
                args.expected_seed_recovery_audit_sha256
            ),
            expected_full_source_blob_inventory_sha256=(
                args.expected_full_source_blob_inventory_sha256
            ),
            expected_routed_source_blob_inventory_sha256=(
                args.expected_routed_source_blob_inventory_sha256
            ),
            accepted_composite_audit_json=args.accepted_composite_audit_json,
            expected_composite_audit_sha256=args.expected_composite_audit_sha256,
            heavy_lock_path=args.heavy_lock_path,
            recovery_policy=RecoveryPolicy(
                recovery_lever=args.recovery_lever,
                scale_search_multipliers=args.scale_search_multipliers,
                rotation_rht_seed=args.rotation_rht_seed,
            ),
        )
    elif args.command == "evidence":
        payload = build_wave1_evidence(
            stats_dir=args.stats_dir,
            attribution_json=args.attribution_json,
            conversion_dir=args.conversion_dir,
            output_json=args.output_json,
        )
    elif args.command == "reevaluate":
        payload = reevaluate_recovery_candidate(
            profile_path=args.profile_path,
            config_path=args.config_path,
            source_index_path=args.source_index_path,
            tokenizer_dir=args.tokenizer_dir,
            tokenizer_readiness_json=args.tokenizer_readiness_json,
            family_policy_json=args.family_policy_json,
            prompt_pack_json=args.prompt_pack_json,
            teacher_cache_root=args.teacher_cache_root,
            non_vq_artifact_dir=args.non_vq_artifact_dir,
            non_vq_evidence_json=args.non_vq_evidence_json,
            accepted_routed_artifact_dir=args.accepted_routed_artifact_dir,
            accepted_composite_audit_json=args.accepted_composite_audit_json,
            accepted_materialization_runs_jsonl=args.accepted_materialization_runs_jsonl,
            accepted_full_bind_preflight_json=args.accepted_full_bind_preflight_json,
            recovery_conversion_dir=args.recovery_conversion_dir,
            output_json=args.output_json,
            expected_seed_manifest_sha256=args.expected_seed_manifest_sha256,
            expected_stats_manifest_sha256=args.expected_stats_manifest_sha256,
            expected_full_source_blob_inventory_sha256=(
                args.expected_full_source_blob_inventory_sha256
            ),
            expected_routed_source_blob_inventory_sha256=(
                args.expected_routed_source_blob_inventory_sha256
            ),
            expected_recovery_lever=args.expected_recovery_lever,
            expected_recovery_policy_json=args.expected_recovery_policy_json,
            expected_composite_audit_sha256=args.expected_composite_audit_sha256,
            adapter_sidecar_dir=args.adapter_sidecar_dir,
            expected_adapter_manifest_sha256=(
                args.expected_adapter_manifest_sha256
            ),
            heavy_lock_path=args.heavy_lock_path,
        )
    elif args.command == "run":
        _validate_e8p_campaign_count(
            args.e8p_worst_layer_count,
            worst_layer_count=args.worst_layer_count,
        )
        root = Path(args.output_root)
        stats_dir = root / "stats"
        attribution_path = Path(args.attribution_json)
        conversion_dir = root / "candidate"
        _prepare_authenticated_stats_root(
            stats_dir,
            expected_sha256=args.expected_stats_manifest_sha256,
            collector=lambda: _collect(args, stats_dir),
        )
        attribution = _load_bound_attribution(
            attribution_path,
            expected_attribution_sha256=args.expected_attribution_sha256,
            expected_stats_manifest_sha256=args.expected_stats_manifest_sha256,
        )
        materialize_groups_from_source(
            source_dir=args.source_dir,
            index_path=args.index_path,
            seed_artifact_dir=args.seed_artifact_dir,
            seed_recovery_artifact_dir=args.seed_recovery_artifact_dir,
            seed_recovery_audit_json=args.seed_recovery_audit_json,
            stats_dir=stats_dir,
            output_dir=conversion_dir,
            groups=_worst_layer_groups(attribution, count=args.worst_layer_count),
            layer_code_bits=_e8p_worst_layer_code_bits(
                attribution, count=args.e8p_worst_layer_count
            ),
            resume=args.resume_materialization,
            expected_stats_manifest_sha256=args.expected_stats_manifest_sha256,
            expected_seed_manifest_sha256=args.expected_seed_manifest_sha256,
            expected_seed_recovery_manifest_sha256=(
                args.expected_seed_recovery_manifest_sha256
            ),
            expected_seed_recovery_audit_sha256=(
                args.expected_seed_recovery_audit_sha256
            ),
            expected_full_source_blob_inventory_sha256=(
                args.expected_full_source_blob_inventory_sha256
            ),
            expected_routed_source_blob_inventory_sha256=(
                args.expected_routed_source_blob_inventory_sha256
            ),
            accepted_composite_audit_json=args.accepted_composite_audit_json,
            expected_composite_audit_sha256=args.expected_composite_audit_sha256,
            heavy_lock_path=args.heavy_lock_path,
            recovery_policy=RecoveryPolicy(
                recovery_lever=args.recovery_lever,
                scale_search_multipliers=args.scale_search_multipliers,
                rotation_rht_seed=args.rotation_rht_seed,
            ),
        )
        payload = build_wave1_evidence(
            stats_dir=stats_dir,
            attribution_json=attribution_path,
            conversion_dir=conversion_dir,
            output_json=root / "wave1-evidence.json",
        )
    else:
        raise AssertionError(f"unhandled command {args.command}")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run_cli(argv)
    except Exception as error:
        print(
            json.dumps({"completed": False, "error": f"{type(error).__name__}: {error}"}, indent=2),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
