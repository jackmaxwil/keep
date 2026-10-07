from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from ramp.benchmark.metrics import (
    collect_metric_snapshot,
    collect_vm_stat_counts,
    reset_mlx_peak_memory,
)
from keep.convert.stream_convert import load_safetensors_index
from ramp.models.glm52_vq_adapter import (
    Glm52VQDecoderLayer,
    Glm52VQMoE,
    bind_glm52_decoder_layer_non_vq_weights,
    bind_glm52_decoder_layer_vq_experts,
    glm52_vq_args_from_config,
    is_glm52_sparse_layer,
)
from ramp.models.profiles import (
    load_profile,
    validate_profile_against_hf_config_data,
)


NON_VQ_RECORD_TYPE = "glm52_non_vq_package_manifest"
NON_VQ_READY_STATUS = "glm52_non_vq_package_ready"
NON_VQ_SOURCE_AUTHORITY = "pinned_huggingface_lfs_v1"
ROUTED_AUDIT_RECORD_TYPE = "glm52_modelopt_nvfp4_artifact_audit"
ROUTED_AUDIT_READY_STATUS = "glm52_modelopt_nvfp4_artifact_audit_ready"
ROUTED_MANIFEST_RECORD_TYPE = "glm52_modelopt_nvfp4_materialization_manifest"
ROUTED_MATERIALIZATION_READY_STATUS = "glm52_modelopt_nvfp4_groups_ready"
SOURCE_DECODER = "modelopt_nvfp4_v1"
SOURCE_WEIGHT_ENCODING = "modelopt_nvfp4"
PROBE_RECORD_TYPE = "glm52_layer_forward_probe"
PROBE_READY_STATUS = "glm52_layer_forward_probe_ready"
PROBE_FAILED_STATUS = "glm52_layer_forward_probe_failed"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    input_path = Path(path)
    try:
        payload = json.loads(input_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label} {input_path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} {input_path} must contain a JSON object")
    return payload


def _canonical(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _require_path(path: str | Path, *, label: str, directory: bool) -> Path:
    resolved = _canonical(path)
    if directory:
        if not resolved.is_dir():
            raise ValueError(f"{label} is not a directory: {path}")
    elif not resolved.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    return resolved


def _recorded_path(payload: Mapping[str, Any], key: str, *, label: str) -> Path:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}.{key} must be a non-empty path string")
    return _canonical(value)


def _expect_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, found {actual!r}")


def _expect_true(label: str, value: object) -> None:
    if value is not True:
        raise ValueError(f"{label} must be true")


def _expect_false(label: str, value: object) -> None:
    if value is not False:
        raise ValueError(f"{label} must be false")


def _expect_sha256(label: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _require_green_checks(
    payload: Mapping[str, Any],
    *,
    label: str,
    expected: tuple[str, ...],
) -> None:
    checks = payload.get("checks")
    if not isinstance(checks, dict):
        raise ValueError(f"{label}.checks must be an object")
    failed = [name for name in expected if checks.get(name) is not True]
    if failed:
        raise ValueError(f"{label} is missing green checks: {failed}")


def _validate_non_vq_evidence(
    *,
    artifact_root: Path,
    evidence: Mapping[str, Any],
    profile_name: str,
    model_id: str,
    revision: str,
    config_sha256: str,
    layer: int,
) -> dict[str, object]:
    _expect_equal("non-VQ record_type", evidence.get("record_type"), NON_VQ_RECORD_TYPE)
    _expect_equal("non-VQ pack_status", evidence.get("pack_status"), NON_VQ_READY_STATUS)
    _expect_true("non-VQ production_ready", evidence.get("production_ready"))
    _expect_equal(
        "non-VQ source_authority",
        evidence.get("source_authority"),
        NON_VQ_SOURCE_AUTHORITY,
    )
    _expect_true("non-VQ package_audit_pass", evidence.get("package_audit_pass"))
    _expect_equal("non-VQ profile", evidence.get("profile"), profile_name)
    _expect_equal("non-VQ model_id", evidence.get("model_id"), model_id)
    _expect_equal("non-VQ source_revision", evidence.get("source_revision"), revision)
    _expect_equal("non-VQ config_sha256", evidence.get("config_sha256"), config_sha256)
    source_index_sha256 = _expect_sha256(
        "non-VQ index_sha256",
        evidence.get("index_sha256"),
    )
    source_blob_inventory_sha256 = _expect_sha256(
        "non-VQ source_blob_inventory_sha256",
        evidence.get("source_blob_inventory_sha256"),
    )
    source_inventory_sha256 = _expect_sha256(
        "non-VQ source_inventory_sha256",
        evidence.get("source_inventory_sha256"),
    )
    manifest_sha256 = _expect_sha256(
        "non-VQ manifest_sha256",
        evidence.get("manifest_sha256"),
    )
    package_index_sha256 = _expect_sha256(
        "non-VQ package_index_sha256",
        evidence.get("package_index_sha256"),
    )
    package_set_sha256 = _expect_sha256(
        "non-VQ package_set_sha256",
        evidence.get("package_set_sha256"),
    )

    audit = evidence.get("package_audit")
    if not isinstance(audit, dict):
        raise ValueError("non-VQ package_audit must be an object")
    _expect_true("non-VQ package_audit.audit_pass", audit.get("audit_pass"))
    _require_green_checks(
        audit,
        label="non-VQ package_audit",
        expected=(
            "lineage",
            "strict_tree",
            "index",
            "tensor_inventory",
            "physical_extents",
            "payload_hashes",
            "shard_hashes",
            "accounting",
        ),
    )
    recorded_artifact = _recorded_path(audit, "artifact_dir", label="non-VQ package_audit")
    if recorded_artifact != artifact_root:
        raise ValueError(
            "non-VQ artifact path mismatch: "
            f"evidence records {recorded_artifact}, input resolves to {artifact_root}"
        )

    manifest_path = _recorded_path(audit, "manifest_path", label="non-VQ package_audit")
    expected_manifest_path = artifact_root / "non-vq-manifest.json"
    if manifest_path != expected_manifest_path:
        raise ValueError(
            "non-VQ manifest path mismatch: "
            f"evidence records {manifest_path}, expected {expected_manifest_path}"
        )
    _require_path(manifest_path, label="non-VQ manifest", directory=False)
    actual_manifest_sha256 = _sha256_file(manifest_path)
    _expect_equal("non-VQ manifest SHA-256", actual_manifest_sha256, manifest_sha256)
    _expect_equal(
        "non-VQ package_audit.manifest_sha256",
        audit.get("manifest_sha256"),
        manifest_sha256,
    )
    _expect_equal(
        "non-VQ package_audit.package_set_sha256",
        audit.get("package_set_sha256"),
        package_set_sha256,
    )

    package_index_path = artifact_root / "model.safetensors.index.json"
    _require_path(package_index_path, label="non-VQ package index", directory=False)
    _expect_equal(
        "non-VQ package index SHA-256",
        _sha256_file(package_index_path),
        package_index_sha256,
    )
    manifest = _load_json_object(manifest_path, label="non-VQ manifest")
    for field, expected in (
        ("record_type", NON_VQ_RECORD_TYPE),
        ("pack_status", NON_VQ_READY_STATUS),
        ("production_ready", True),
        ("source_authority", NON_VQ_SOURCE_AUTHORITY),
        ("profile", profile_name),
        ("model_id", model_id),
        ("source_revision", revision),
        ("config_sha256", config_sha256),
        ("index_sha256", source_index_sha256),
        ("source_blob_inventory_sha256", source_blob_inventory_sha256),
        ("source_inventory_sha256", source_inventory_sha256),
        ("package_index_sha256", package_index_sha256),
        ("package_set_sha256", package_set_sha256),
    ):
        _expect_equal(f"non-VQ manifest {field}", manifest.get(field), expected)

    package_index = load_safetensors_index(package_index_path)
    layer_prefix = f"model.layers.{layer}."
    selected_shard_names = sorted(
        {
            shard
            for name, shard in package_index.weight_map.items()
            if name.startswith(layer_prefix)
        }
    )
    if not selected_shard_names:
        raise ValueError(f"non-VQ package index has no tensors for layer {layer}")
    shard_rows = manifest.get("shards")
    if not isinstance(shard_rows, list):
        raise ValueError("non-VQ manifest shards must be an array")
    shard_records: dict[str, Mapping[str, Any]] = {}
    for record in shard_rows:
        if not isinstance(record, dict):
            raise ValueError("non-VQ manifest shard records must be objects")
        filename = record.get("filename")
        if not isinstance(filename, str) or not filename:
            raise ValueError("non-VQ manifest shard filename must be a non-empty string")
        if Path(filename).name != filename:
            raise ValueError(f"non-VQ manifest shard filename is not local: {filename!r}")
        if filename in shard_records:
            raise ValueError(f"non-VQ manifest repeats shard record {filename!r}")
        shard_records[filename] = record
    selected_shards: list[dict[str, object]] = []
    for filename in selected_shard_names:
        record = shard_records.get(filename)
        if record is None:
            raise ValueError(
                f"non-VQ manifest has no shard identity for selected layer shard {filename}"
            )
        shard_path = artifact_root / filename
        _require_path(
            shard_path,
            label=f"non-VQ selected shard {filename}",
            directory=False,
        )
        _expect_equal(
            f"non-VQ selected shard {filename} file_bytes",
            record.get("file_bytes"),
            shard_path.stat().st_size,
        )
        expected_shard_sha256 = _expect_sha256(
            f"non-VQ selected shard {filename} file_sha256",
            record.get("file_sha256"),
        )
        _expect_equal(
            f"non-VQ selected shard {filename} SHA-256",
            _sha256_file(shard_path),
            expected_shard_sha256,
        )
        selected_shards.append(
            {
                "filename": filename,
                "file_bytes": shard_path.stat().st_size,
                "file_sha256": expected_shard_sha256,
            }
        )

    return {
        "source_index_sha256": source_index_sha256,
        "source_blob_inventory_sha256": source_blob_inventory_sha256,
        "source_inventory_sha256": source_inventory_sha256,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "package_index_path": str(package_index_path),
        "package_index_sha256": package_index_sha256,
        "package_set_sha256": package_set_sha256,
        "selected_shards": selected_shards,
    }


def _validate_routed_evidence(
    *,
    artifact_root: Path,
    audit: Mapping[str, Any],
    profile_name: str,
    model_id: str,
    revision: str,
    config_sha256: str,
    source_index_sha256: str,
    layer: int,
) -> dict[str, object]:
    _expect_equal(
        "routed audit record_type",
        audit.get("record_type"),
        ROUTED_AUDIT_RECORD_TYPE,
    )
    _expect_equal(
        "routed audit_status",
        audit.get("audit_status"),
        ROUTED_AUDIT_READY_STATUS,
    )
    _expect_true("routed audit_pass", audit.get("audit_pass"))
    _expect_true("routed artifact_integrity_pass", audit.get("artifact_integrity_pass"))
    if audit.get("audit_blockers") != []:
        raise ValueError(f"routed audit_blockers must be empty: {audit.get('audit_blockers')!r}")
    _expect_false("routed dense_routed_experts", audit.get("dense_routed_experts"))
    _expect_equal("routed model_id", audit.get("model_id"), model_id)
    _expect_equal("routed source_revision", audit.get("source_revision"), revision)
    _expect_equal("routed config_sha256", audit.get("config_sha256"), config_sha256)
    _expect_equal("routed index_sha256", audit.get("index_sha256"), source_index_sha256)
    _expect_equal("routed source_decoder", audit.get("source_decoder"), SOURCE_DECODER)
    _expect_equal(
        "routed source_weight_encoding",
        audit.get("source_weight_encoding"),
        SOURCE_WEIGHT_ENCODING,
    )
    group_set_sha256 = _expect_sha256(
        "routed group_set_sha256",
        audit.get("group_set_sha256"),
    )
    _require_green_checks(
        audit,
        label="routed audit",
        expected=(
            "exact_manifest_contract",
            "exact_group_selection",
            "exact_artifact_files",
            "exact_tensor_contract",
            "pinned_lineage",
            "no_layer_78",
            "no_dense_routed_experts",
            "artifact_hashes_and_sizes",
            "bounded_whole_model_claims_suppressed",
        ),
    )

    recorded_artifact = _recorded_path(audit, "artifact_dir", label="routed audit")
    if recorded_artifact != artifact_root:
        raise ValueError(
            "routed artifact path mismatch: "
            f"audit records {recorded_artifact}, input resolves to {artifact_root}"
        )
    required_group_keys = tuple(
        f"{layer}:{projection}"
        for projection in ("gate_proj", "up_proj", "down_proj")
    )
    ready_group_keys = audit.get("ready_group_keys")
    if not isinstance(ready_group_keys, list):
        raise ValueError("routed ready_group_keys must be an array")
    if not all(isinstance(key, str) for key in ready_group_keys):
        raise ValueError("routed ready_group_keys entries must be strings")
    if len(set(ready_group_keys)) != len(ready_group_keys):
        raise ValueError("routed ready_group_keys must not contain duplicates")
    missing_group_keys = [key for key in required_group_keys if key not in ready_group_keys]
    if missing_group_keys:
        raise ValueError(
            f"routed audit does not contain a complete layer {layer}: {missing_group_keys}"
        )
    complete_layer_ids = audit.get("complete_layer_ids")
    if not isinstance(complete_layer_ids, list) or layer not in complete_layer_ids:
        raise ValueError(f"routed audit complete_layer_ids does not include layer {layer}")

    manifest_path = _recorded_path(audit, "manifest_path", label="routed audit")
    expected_manifest_path = artifact_root / "conversion-manifest.json"
    if manifest_path != expected_manifest_path:
        raise ValueError(
            "routed manifest path mismatch: "
            f"audit records {manifest_path}, expected {expected_manifest_path}"
        )
    _require_path(manifest_path, label="routed manifest", directory=False)
    manifest_sha256 = _expect_sha256(
        "routed manifest_sha256",
        audit.get("manifest_sha256"),
    )
    _expect_equal(
        "routed manifest SHA-256",
        _sha256_file(manifest_path),
        manifest_sha256,
    )
    manifest = _load_json_object(manifest_path, label="routed manifest")
    for field, expected in (
        ("record_type", ROUTED_MANIFEST_RECORD_TYPE),
        ("materialization_status", ROUTED_MATERIALIZATION_READY_STATUS),
        ("model_id", model_id),
        ("source_revision", revision),
        ("config_sha256", config_sha256),
        ("index_sha256", source_index_sha256),
        ("profile", profile_name),
        ("source_decoder", SOURCE_DECODER),
        ("source_weight_encoding", SOURCE_WEIGHT_ENCODING),
        ("dense_checkpoint_written", False),
    ):
        _expect_equal(f"routed manifest {field}", manifest.get(field), expected)
    selected_group_keys = manifest.get("selected_group_keys")
    if not isinstance(selected_group_keys, list):
        raise ValueError("routed manifest selected_group_keys must be an array")
    if not all(isinstance(key, str) for key in selected_group_keys):
        raise ValueError("routed manifest selected_group_keys entries must be strings")
    if len(set(selected_group_keys)) != len(selected_group_keys):
        raise ValueError("routed manifest selected_group_keys must not contain duplicates")
    if set(ready_group_keys) != set(selected_group_keys):
        raise ValueError(
            "routed audit ready_group_keys do not exactly match manifest "
            "selected_group_keys"
        )
    missing_manifest_groups = [
        key for key in required_group_keys if key not in selected_group_keys
    ]
    if missing_manifest_groups:
        raise ValueError(
            "routed manifest is missing selected layer groups: "
            f"{missing_manifest_groups}"
        )
    audit_scope = audit.get("materialization_scope")
    manifest_scope = manifest.get("materialization_scope")
    if audit_scope not in {"bounded", "full"}:
        raise ValueError(
            f"routed audit materialization_scope is unsupported: {audit_scope!r}"
        )
    _expect_equal(
        "routed audit/manifest materialization_scope",
        audit_scope,
        manifest_scope,
    )
    if audit_scope != "bounded":
        raise ValueError(
            "layer-local GLM52 probe accepts only bounded routed artifacts; "
            "full scope is unsupported"
        )
    full_ready_claim = audit.get("full_routed_artifact_ready")
    if type(full_ready_claim) is not bool:
        raise ValueError("routed full_routed_artifact_ready must be a boolean")
    _expect_false("routed full_routed_artifact_ready", full_ready_claim)
    for audit_field, manifest_field in (
        ("requested_code_bits", "code_bits"),
        ("requested_group_size", "group_size"),
        ("requested_scale_estimator", "scale_estimator"),
    ):
        _expect_equal(
            f"routed {audit_field}",
            audit.get(audit_field),
            manifest.get(manifest_field),
        )

    groups = manifest.get("groups")
    if not isinstance(groups, list):
        raise ValueError("routed manifest groups must be an array")
    group_records: dict[str, Mapping[str, Any]] = {}
    for record in groups:
        if not isinstance(record, dict):
            raise ValueError("routed manifest group records must be objects")
        key = f"{record.get('layer')}:{record.get('projection')}"
        if key in group_records:
            raise ValueError(f"routed manifest repeats group record {key}")
        group_records[key] = record
    if set(group_records) != set(selected_group_keys):
        raise ValueError(
            "routed manifest group records do not exactly match selected_group_keys"
        )
    for group_key in required_group_keys:
        record = group_records.get(group_key)
        if record is None:
            raise ValueError(f"routed manifest has no group record for {group_key}")
        _expect_equal(f"routed group {group_key} status", record.get("status"), "ready")
        projection = group_key.split(":", maxsplit=1)[1]
        expected_path = artifact_root / f"layer-{layer:05d}-{projection}.safetensors"
        recorded_path = _recorded_path(record, "artifact_path", label=f"routed group {group_key}")
        if recorded_path != expected_path:
            raise ValueError(
                f"routed group {group_key} artifact path mismatch: "
                f"manifest records {recorded_path}, expected {expected_path}"
            )
        _require_path(expected_path, label=f"routed group {group_key}", directory=False)
        _expect_equal(
            f"routed group {group_key} artifact_bytes",
            record.get("artifact_bytes"),
            expected_path.stat().st_size,
        )
        artifact_sha256 = _expect_sha256(
            f"routed group {group_key} artifact_sha256",
            record.get("artifact_sha256"),
        )
        _expect_equal(
            f"routed group {group_key} artifact SHA-256",
            _sha256_file(expected_path),
            artifact_sha256,
        )

    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "group_set_sha256": group_set_sha256,
        "materialization_scope": audit_scope,
        "full_routed_artifact_ready": False,
        "code_bits": manifest.get("code_bits"),
        "group_size": manifest.get("group_size"),
        "scale_estimator": manifest.get("scale_estimator"),
        "ready_group_keys": list(ready_group_keys),
    }


def _validate_probe_inputs(
    *,
    non_vq_artifact: str | Path,
    non_vq_evidence_json: str | Path,
    routed_artifact: str | Path,
    routed_audit_json: str | Path,
    profile_path: str | Path,
    config_path: str | Path,
    model_id: str,
    revision: str,
    layer: int,
    tokens: int,
    input_scale: float,
) -> dict[str, Any]:
    if tokens != 1:
        raise ValueError("standalone bounded GLM52 layer-forward probe requires --tokens 1")
    if input_scale <= 0.0:
        raise ValueError("input_scale must be positive")
    if layer < 0:
        raise ValueError("layer must be non-negative")

    profile_input_path = _require_path(profile_path, label="model profile", directory=False)
    config_input_path = _require_path(config_path, label="HF config", directory=False)
    non_vq_evidence_path = _require_path(
        non_vq_evidence_json,
        label="non-VQ evidence",
        directory=False,
    )
    routed_audit_path = _require_path(
        routed_audit_json,
        label="routed audit",
        directory=False,
    )
    non_vq_root = _require_path(
        non_vq_artifact,
        label="non-VQ artifact",
        directory=True,
    )
    routed_root = _require_path(
        routed_artifact,
        label="routed artifact",
        directory=True,
    )

    profile = load_profile(profile_input_path)
    _expect_equal("requested model_id/profile hf_model_id", model_id, profile.hf_model_id)
    _expect_equal("requested revision/profile revision", revision, profile.revision)
    config = _load_json_object(config_input_path, label="HF config")
    config_mismatches = validate_profile_against_hf_config_data(profile, config)
    if config_mismatches:
        raise ValueError(f"model profile/config mismatch: {config_mismatches}")
    config_sha256 = _sha256_file(config_input_path)
    model_args = glm52_vq_args_from_config(config)
    if not 0 <= layer < model_args.num_hidden_layers:
        raise ValueError(
            f"layer {layer} is outside config num_hidden_layers={model_args.num_hidden_layers}"
        )
    if not is_glm52_sparse_layer(model_args, layer):
        raise ValueError(f"layer {layer} is not a sparse GLM52 MoE layer")

    non_vq_evidence = _load_json_object(
        non_vq_evidence_path,
        label="non-VQ evidence",
    )
    non_vq_lineage = _validate_non_vq_evidence(
        artifact_root=non_vq_root,
        evidence=non_vq_evidence,
        profile_name=profile.name,
        model_id=model_id,
        revision=revision,
        config_sha256=config_sha256,
        layer=layer,
    )
    routed_audit = _load_json_object(routed_audit_path, label="routed audit")
    routed_lineage = _validate_routed_evidence(
        artifact_root=routed_root,
        audit=routed_audit,
        profile_name=profile.name,
        model_id=model_id,
        revision=revision,
        config_sha256=config_sha256,
        source_index_sha256=non_vq_lineage["source_index_sha256"],
        layer=layer,
    )
    return {
        "profile": profile,
        "profile_path": profile_input_path,
        "profile_sha256": _sha256_file(profile_input_path),
        "config": config,
        "config_path": config_input_path,
        "config_sha256": config_sha256,
        "model_args": model_args,
        "non_vq_root": non_vq_root,
        "non_vq_evidence_path": non_vq_evidence_path,
        "non_vq_lineage": non_vq_lineage,
        "routed_root": routed_root,
        "routed_audit_path": routed_audit_path,
        "routed_lineage": routed_lineage,
    }


def _vq_projection_metadata(layer: Glm52VQDecoderLayer) -> dict[str, dict[str, object]]:
    if not isinstance(layer.mlp, Glm52VQMoE) or layer.mlp.switch_mlp is None:
        raise RuntimeError("VQ switch_mlp is not bound")
    metadata: dict[str, dict[str, object]] = {}
    for name in ("gate_proj", "up_proj", "down_proj"):
        projection = getattr(layer.mlp.switch_mlp, name)
        metadata[name] = {
            "input_dims": int(projection.input_dims),
            "output_dims": int(projection.output_dims),
            "num_experts": int(projection.num_experts),
            "group_size": int(projection.group_size),
            "code_bits": int(projection.code_bits),
            "codes_shape": list(projection.codes.shape),
            "codes_dtype": str(projection.codes.dtype),
            "scales_shape": list(projection.scales.shape),
            "scales_dtype": str(projection.scales.dtype),
        }
    return metadata


def _dense_routed_parameter_names(layer: Glm52VQDecoderLayer) -> list[str]:
    names = []
    for name, _value in tree_flatten(layer.parameters()):
        routed_expert = name.startswith("mlp.experts.") or ".mlp.experts." in name
        dense_switch = (
            (name.startswith("mlp.switch_mlp.") or ".mlp.switch_mlp." in name)
            and name.endswith(".weight")
        )
        if routed_expert or dense_switch:
            names.append(name)
    return sorted(names)


def _memory_clean_diagnostic(
    metrics: Mapping[str, object],
) -> tuple[bool, bool | None]:
    pageouts_delta = metrics.get("pageouts_delta")
    swapouts_delta = metrics.get("swapouts_delta")
    available = type(pageouts_delta) is int and type(swapouts_delta) is int
    if not available:
        return False, None
    return True, pageouts_delta == 0 and swapouts_delta == 0


def probe_glm52_layer_forward(
    *,
    non_vq_artifact: str | Path,
    non_vq_evidence_json: str | Path,
    routed_artifact: str | Path,
    routed_audit_json: str | Path,
    profile_path: str | Path,
    config_path: str | Path,
    model_id: str,
    revision: str,
    layer: int,
    tokens: int = 1,
    seed: int = 5202,
    input_scale: float = 0.02,
) -> dict[str, Any]:
    validation_started = time.perf_counter()
    validated = _validate_probe_inputs(
        non_vq_artifact=non_vq_artifact,
        non_vq_evidence_json=non_vq_evidence_json,
        routed_artifact=routed_artifact,
        routed_audit_json=routed_audit_json,
        profile_path=profile_path,
        config_path=config_path,
        model_id=model_id,
        revision=revision,
        layer=layer,
        tokens=tokens,
        input_scale=input_scale,
    )
    validation_seconds = time.perf_counter() - validation_started
    model_args = validated["model_args"]
    profile = validated["profile"]

    before_vm = collect_vm_stat_counts()
    reset_mlx_peak_memory()
    runtime_started = time.perf_counter()
    instantiate_started = time.perf_counter()
    decoder_layer = Glm52VQDecoderLayer(model_args, layer)
    instantiate_seconds = time.perf_counter() - instantiate_started

    non_vq_bind_started = time.perf_counter()
    non_vq_index_path = validated["non_vq_root"] / "model.safetensors.index.json"
    non_vq_index = load_safetensors_index(non_vq_index_path)
    non_vq_report = bind_glm52_decoder_layer_non_vq_weights(
        decoder_layer,
        validated["non_vq_root"],
        non_vq_index,
        model_args=model_args,
        strict=True,
    )
    non_vq_bind_seconds = time.perf_counter() - non_vq_bind_started

    vq_bind_started = time.perf_counter()
    bound_layer = bind_glm52_decoder_layer_vq_experts(
        decoder_layer,
        validated["routed_root"],
        profile=profile,
    )
    vq_bind_seconds = time.perf_counter() - vq_bind_started
    _expect_equal("bound sparse layer", bound_layer, layer)
    decoder_layer.eval()
    mx.eval(decoder_layer.parameters())

    rng = np.random.default_rng(seed)
    input_values = rng.normal(
        loc=0.0,
        scale=input_scale,
        size=(1, tokens, model_args.hidden_size),
    ).astype(np.float32)
    inputs = mx.array(input_values).astype(mx.bfloat16)
    mx.eval(inputs)
    forward_started = time.perf_counter()
    output, topk_indices = decoder_layer(inputs)
    mx.eval(output)
    forward_seconds = time.perf_counter() - forward_started
    runtime_seconds = time.perf_counter() - runtime_started
    metrics = collect_metric_snapshot(previous_vm_stat_counts=before_vm)

    finite_value = mx.all(mx.isfinite(output))
    abs_max_value = mx.max(mx.abs(output.astype(mx.float32)))
    mx.eval(finite_value, abs_max_value)
    output_finite = bool(finite_value.item())
    output_abs_max = float(abs_max_value.item())
    dense_routed_parameter_names = _dense_routed_parameter_names(decoder_layer)
    dense_routed_parameter_present = bool(dense_routed_parameter_names)
    output_shape = list(output.shape)
    output_dtype = str(output.dtype)
    expected_output_shape = [1, tokens, model_args.hidden_size]
    indexer_type = model_args.indexer_types[layer]
    indexshare_sparse_selection_exercised = bool(
        indexer_type == "shared" and topk_indices is not None
    )
    vq_metadata = _vq_projection_metadata(decoder_layer)
    projection_contract_consistent = all(
        metadata["num_experts"] == profile.num_experts
        and metadata["code_bits"] == validated["routed_lineage"]["code_bits"]
        and metadata["group_size"] == validated["routed_lineage"]["group_size"]
        for metadata in vq_metadata.values()
    )
    probe_pass = bool(
        output_finite
        and output_shape == expected_output_shape
        and output.dtype == mx.bfloat16
        and not dense_routed_parameter_present
        and non_vq_report.missing_model_parameters == ()
        and non_vq_report.skipped_unmatched_tensors == ()
        and projection_contract_consistent
    )
    vm_stat_available, memory_clean = _memory_clean_diagnostic(metrics)

    return {
        "schema_version": 1,
        "record_type": PROBE_RECORD_TYPE,
        "probe_status": PROBE_READY_STATUS if probe_pass else PROBE_FAILED_STATUS,
        "probe_pass": probe_pass,
        "scope": "single_sparse_decoder_layer",
        "model_id": model_id,
        "source_revision": revision,
        "profile": profile.name,
        "profile_path": str(validated["profile_path"]),
        "profile_sha256": validated["profile_sha256"],
        "config_path": str(validated["config_path"]),
        "config_sha256": validated["config_sha256"],
        "source_index_sha256": validated["non_vq_lineage"]["source_index_sha256"],
        "source_blob_inventory_sha256": validated["non_vq_lineage"][
            "source_blob_inventory_sha256"
        ],
        "source_inventory_sha256": validated["non_vq_lineage"][
            "source_inventory_sha256"
        ],
        "non_vq_source_authority": NON_VQ_SOURCE_AUTHORITY,
        "production_non_vq_evidence_validated": True,
        "non_vq_artifact_dir": str(validated["non_vq_root"]),
        "non_vq_evidence_json": str(validated["non_vq_evidence_path"]),
        "non_vq_manifest_path": validated["non_vq_lineage"]["manifest_path"],
        "non_vq_manifest_sha256": validated["non_vq_lineage"]["manifest_sha256"],
        "non_vq_package_index_sha256": validated["non_vq_lineage"][
            "package_index_sha256"
        ],
        "non_vq_package_set_sha256": validated["non_vq_lineage"][
            "package_set_sha256"
        ],
        "non_vq_selected_shards": validated["non_vq_lineage"]["selected_shards"],
        "routed_artifact_dir": str(validated["routed_root"]),
        "routed_audit_json": str(validated["routed_audit_path"]),
        "routed_manifest_path": validated["routed_lineage"]["manifest_path"],
        "routed_manifest_sha256": validated["routed_lineage"]["manifest_sha256"],
        "routed_group_set_sha256": validated["routed_lineage"]["group_set_sha256"],
        "routed_artifact_audit_validated": True,
        "routed_source_decoder": SOURCE_DECODER,
        "routed_source_weight_encoding": SOURCE_WEIGHT_ENCODING,
        "routed_materialization_scope": validated["routed_lineage"][
            "materialization_scope"
        ],
        "full_routed_artifact_ready": validated["routed_lineage"][
            "full_routed_artifact_ready"
        ],
        "layer": layer,
        "tokens": tokens,
        "seed": seed,
        "input_scale": input_scale,
        "input_shape": list(inputs.shape),
        "input_dtype": str(inputs.dtype),
        "output_shape": output_shape,
        "output_dtype": output_dtype,
        "output_finite": output_finite,
        "output_abs_max": output_abs_max,
        "num_experts": profile.num_experts,
        "top_k": profile.experts_per_tok,
        "indexer_type": indexer_type,
        "previous_topk_indices_supplied": False,
        "attention_topk_indices_returned": topk_indices is not None,
        "indexshare_sparse_selection_exercised": indexshare_sparse_selection_exercised,
        "non_vq_bind_report": non_vq_report.to_dict(),
        "vq_projection_metadata": vq_metadata,
        "vq_projection_contract_consistent": projection_contract_consistent,
        "parameter_tensor_count": len(tree_flatten(decoder_layer.parameters())),
        "dense_routed_parameter_present": dense_routed_parameter_present,
        "dense_routed_parameter_names": dense_routed_parameter_names,
        "validation_seconds": validation_seconds,
        "instantiate_seconds": instantiate_seconds,
        "non_vq_bind_seconds": non_vq_bind_seconds,
        "vq_bind_seconds": vq_bind_seconds,
        "forward_seconds": forward_seconds,
        "runtime_seconds": runtime_seconds,
        "vm_stat_available": vm_stat_available,
        "memory_clean_diagnostic": memory_clean,
        **metrics,
        "whole_model_runtime_proven": False,
        "generation_proven": False,
        "tokenizer_proven": False,
        "long_context_indexshare_proven": False,
        "speed_claim": False,
        "notes": [
            "This record proves one strictly bound sparse decoder-layer forward only.",
            "The one-token shared-indexer path receives no previous layer top-k indices.",
            "Memory counters are diagnostics for this process and do not prove whole-model residency.",
        ],
    }


def _write_json_atomic(path: str | Path, payload: Mapping[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp"
    )
    try:
        temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a production-lineage-validated, one-token GLM-5.2 sparse "
            "decoder-layer forward without dense routed weights."
        )
    )
    parser.add_argument("--non-vq-artifact", required=True)
    parser.add_argument("--non-vq-evidence-json", required=True)
    parser.add_argument("--routed-artifact", required=True)
    parser.add_argument("--routed-audit-json", required=True)
    parser.add_argument("--profile-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=5202)
    parser.add_argument("--input-scale", type=float, default=0.02)
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = probe_glm52_layer_forward(
            non_vq_artifact=args.non_vq_artifact,
            non_vq_evidence_json=args.non_vq_evidence_json,
            routed_artifact=args.routed_artifact,
            routed_audit_json=args.routed_audit_json,
            profile_path=args.profile_path,
            config_path=args.config_path,
            model_id=args.model_id,
            revision=args.revision,
            layer=args.layer,
            tokens=args.tokens,
            seed=args.seed,
            input_scale=args.input_scale,
        )
    except Exception as error:
        payload = {
            "schema_version": 1,
            "record_type": PROBE_RECORD_TYPE,
            "probe_status": PROBE_FAILED_STATUS,
            "probe_pass": False,
            "scope": "single_sparse_decoder_layer",
            "model_id": args.model_id,
            "source_revision": args.revision,
            "layer": args.layer,
            "tokens": args.tokens,
            "non_vq_artifact_dir": args.non_vq_artifact,
            "non_vq_evidence_json": args.non_vq_evidence_json,
            "routed_artifact_dir": args.routed_artifact,
            "routed_audit_json": args.routed_audit_json,
            "input_error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "whole_model_runtime_proven": False,
            "generation_proven": False,
            "tokenizer_proven": False,
            "long_context_indexshare_proven": False,
            "speed_claim": False,
        }
        _write_json_atomic(args.output_json, payload)
        return 1
    _write_json_atomic(args.output_json, payload)
    return 0 if payload["probe_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
