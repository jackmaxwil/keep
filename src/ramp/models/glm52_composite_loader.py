"""Authenticated production composite loading for GLM-5.2 REAP KEEP."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from keep.convert.glm52_non_vq import audit_glm52_non_vq_package
from keep.convert.glm52_reap import (
    GLM52_REAP_CONFIG_SHA256,
    GLM52_REAP_EXPECTED_GROUPS,
    GLM52_REAP_EXPECTED_LAYER_IDS,
    GLM52_REAP_INDEX_SHA256,
    GLM52_REAP_PROFILE_NAME,
)
from keep.convert.stream_convert import SafetensorsIndex, load_safetensors_index
from ramp.models.glm52_vq_adapter import (
    GLM52NonVQBindReport,
    GLM52VQModel,
    bind_glm52_non_vq_weights,
    bind_glm52_vq_experts,
    bind_glm52_vq_experts_from_paths,
    dense_glm52_routed_parameter_names,
    glm52_vq_args_from_config,
    has_unbound_vq_experts,
)
from ramp.models.profiles import (
    ModelProfile,
    validate_profile_against_hf_config_data,
)
from keep.io.authenticated_artifacts import (
    AuthenticatedFile,
    DescriptorSnapshot,
    clone_or_copy_authenticated,
)
from keep.quality.glm52_family import (
    GLM52_PINNED_TOKENIZER_FILES,
    PINNED_GLM52_MODEL_ID,
    PINNED_GLM52_REVISION,
    validate_glm52_family_gate_policy,
    validate_glm52_tokenizer_readiness,
)
from keep.validate.glm52_artifact import (
    audit_glm52_reap_materialization_manifest,
    audit_glm52_reap_source_accounting,
)
from keep.validate.glm52_recovery_artifact import (
    GLM52RecoveryArtifactAudit,
    audit_glm52_recovery_mixed_artifact,
)
from keep.validate.glm52_runtime import (
    expected_glm52_routed_group_keys,
    preflight_glm52_full_bind,
)


GLM52_MAIN_MODEL_PARAMETER_COUNT = 494_194_805_304
GLM52_ROUTED_PARAMETER_COUNT = 475_634_073_600
GLM52_NON_VQ_PARAMETER_COUNT = 18_560_731_704
GLM52_NON_VQ_PAYLOAD_BYTES = 37_121_488_608
GLM52_ROUTED_PAYLOAD_BYTES = 61_312_204_800
GLM52_ROUTED_CODEBOOK_BYTES = 230_400
GLM52_ACCEPTED_TENSOR_PAYLOAD_BYTES = 98_433_923_808
GLM52_ACCEPTED_TENSOR_PAYLOAD_BPW = (
    GLM52_ACCEPTED_TENSOR_PAYLOAD_BYTES
    * 8
    / GLM52_MAIN_MODEL_PARAMETER_COUNT
)
GLM52_EXPECTED_SPARSE_LAYERS = tuple(range(3, 78))
GLM52_EXPECTED_NON_VQ_RUNTIME_TARGETS = 1_272


@dataclass(frozen=True, slots=True)
class GLM52CompositeArtifactIdentity:
    body: Mapping[str, object]
    sha256: str


@dataclass(frozen=True, slots=True)
class GLM52ProductionInputFingerprint:
    authority_files: tuple[str, ...]
    artifact_roots: tuple[str, ...]
    authority_file_sha256: Mapping[str, str]
    authority_file_state: Mapping[str, tuple[int, int, int, int, int, int, int]]
    authority_parent_state: Mapping[str, tuple[int, int, bool, bool]]
    artifact_tree_state: Mapping[str, tuple[int, int, int, int, int, int, int]]


@dataclass(frozen=True, slots=True)
class GLM52ValidatedProductionInputs:
    profile: ModelProfile
    config: Mapping[str, Any]
    non_vq_index: SafetensorsIndex
    tokenizer_dir: Path
    tokenizer_readiness: Mapping[str, Any]
    non_vq_artifact_dir: Path
    routed_artifact_dir: Path
    artifact_identity: GLM52CompositeArtifactIdentity
    input_evidence_file_sha256: Mapping[str, str]
    input_fingerprint: GLM52ProductionInputFingerprint
    prompt: str


@dataclass(frozen=True, slots=True)
class GLM52ValidatedRecoveryCandidate:
    baseline: GLM52ValidatedProductionInputs
    recovery_dir: Path
    recovery_artifact_dir: Path
    recovery_audit: GLM52RecoveryArtifactAudit
    recovery_candidate_identity_sha256: str
    recovery_manifest_body_sha256: str


@dataclass(frozen=True, slots=True)
class GLM52CompositeLoadReport:
    artifact_identity_sha256: str
    non_vq_bind_report: GLM52NonVQBindReport
    bound_sparse_layer_ids: tuple[int, ...]
    dense_routed_parameter_names: tuple[str, ...]
    unbound_vq_experts: bool


@dataclass(frozen=True, slots=True)
class GLM52RecoveryCandidateLoadReport:
    baseline_composite_identity_sha256: str
    recovery_candidate_identity_sha256: str
    recovery_manifest_body_sha256: str
    non_vq_bind_report: GLM52NonVQBindReport
    bound_sparse_layer_ids: tuple[int, ...]
    dense_routed_parameter_names: tuple[str, ...]
    unbound_vq_experts: bool


_GLM52RecoveryBindSnapshot = DescriptorSnapshot


def _clone_or_copy_open_file(
    source: AuthenticatedFile,
    destination: Path,
    *,
    allow_copy_fallback: bool,
    required_free_bytes: int,
) -> None:
    """Compatibility wrapper around the shared authenticated snapshot copy."""

    clone_or_copy_authenticated(
        source,
        destination,
        allow_copy_fallback=allow_copy_fallback,
        required_free_bytes=required_free_bytes,
    )


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_tree_state(
    roots: Sequence[str | Path],
) -> dict[str, tuple[int, int, int, int, int, int, int]]:
    state: dict[str, tuple[int, int, int, int, int, int, int]] = {}
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"artifact fingerprint root is not a directory: {root}")
        entries = (root, *sorted(root.rglob("*")))
        for path in entries:
            stat = path.lstat()
            state[str(path)] = (
                stat.st_mode,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
                stat.st_ino,
                stat.st_dev,
                stat.st_nlink,
            )
    return state


def _filesystem_state(path: Path) -> tuple[int, int, int, int, int, int, int]:
    stat = path.lstat()
    return (
        stat.st_mode,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
        stat.st_ino,
        stat.st_dev,
        stat.st_nlink,
    )


def _directory_identity(path: Path) -> tuple[int, int, bool, bool]:
    path_stat = path.lstat()
    return (
        path_stat.st_dev,
        path_stat.st_ino,
        stat.S_ISDIR(path_stat.st_mode),
        not stat.S_ISLNK(path_stat.st_mode),
    )


def snapshot_glm52_production_inputs(
    *,
    authority_files: Sequence[str | Path],
    artifact_roots: Sequence[str | Path],
) -> GLM52ProductionInputFingerprint:
    authority_paths = tuple(
        dict.fromkeys(
            str(Path(path).expanduser().resolve()) for path in authority_files
        )
    )
    root_paths = tuple(
        dict.fromkeys(
            str(Path(path).expanduser().resolve()) for path in artifact_roots
        )
    )
    authority_hashes: dict[str, str] = {}
    authority_states: dict[str, tuple[int, int, int, int, int, int, int]] = {}
    parent_paths = tuple(sorted({str(Path(path).parent) for path in authority_paths}))
    parent_states = {
        path: _directory_identity(Path(path)) for path in parent_paths
    }
    for path in authority_paths:
        if not Path(path).is_file():
            raise ValueError(f"authority fingerprint input is not a file: {path}")
        before_state = _filesystem_state(Path(path))
        authority_hashes[path] = _sha256_file(path)
        after_state = _filesystem_state(Path(path))
        if after_state != before_state:
            raise ValueError(
                f"authority fingerprint input changed while hashing: {path}"
            )
        authority_states[path] = after_state
    for path, expected_state in parent_states.items():
        if _directory_identity(Path(path)) != expected_state:
            raise ValueError(
                f"authority parent identity changed while hashing inputs: {path}"
            )
    return GLM52ProductionInputFingerprint(
        authority_files=authority_paths,
        artifact_roots=root_paths,
        authority_file_sha256=authority_hashes,
        authority_file_state=authority_states,
        authority_parent_state=parent_states,
        artifact_tree_state=_artifact_tree_state(root_paths),
    )


def assert_glm52_production_inputs_unchanged(
    fingerprint: GLM52ProductionInputFingerprint,
) -> None:
    for path, expected_sha256 in fingerprint.authority_file_sha256.items():
        input_path = Path(path)
        before_state = (
            _filesystem_state(input_path) if input_path.is_file() else None
        )
        actual_sha256 = _sha256_file(path) if before_state is not None else None
        after_state = (
            _filesystem_state(input_path) if input_path.is_file() else None
        )
        if (
            before_state != fingerprint.authority_file_state[path]
            or after_state != before_state
            or actual_sha256 != expected_sha256
        ):
            raise ValueError(f"authenticated authority file changed: {path}")
    for path, expected_state in fingerprint.authority_parent_state.items():
        try:
            actual_identity = _directory_identity(Path(path))
        except OSError:
            actual_identity = None
        if actual_identity != expected_state:
            raise ValueError(
                f"authenticated authority parent identity changed: {path}"
            )
    actual_tree_state = _artifact_tree_state(fingerprint.artifact_roots)
    if actual_tree_state != fingerprint.artifact_tree_state:
        raise ValueError("authenticated artifact tree changed after fresh audit")


def _load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    input_path = Path(path)
    try:
        payload = json.loads(input_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label} {input_path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _read_stable_authority_bytes(path: Path, *, label: str) -> bytes:
    parent = path.parent
    before_file = _filesystem_state(path)
    before_parent = _directory_identity(parent)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"could not read {label} {path}: {error}") from error
    if (
        _filesystem_state(path) != before_file
        or _directory_identity(parent) != before_parent
    ):
        raise ValueError(f"{label} changed while it was being authenticated")
    return raw


def _load_json_object_bytes(
    raw: bytes,
    *,
    path: Path,
    label: str,
) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not parse {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _load_profile_bytes(raw: bytes, *, path: Path) -> ModelProfile:
    try:
        payload = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"could not parse profile {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"profile {path} must contain a mapping")
    try:
        return ModelProfile(**payload)
    except TypeError as error:
        raise ValueError(f"invalid model profile {path}: {error}") from error


def _load_safetensors_index_bytes(
    raw: bytes,
    *,
    path: Path,
) -> SafetensorsIndex:
    payload = _load_json_object_bytes(raw, path=path, label="source index")
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("safetensors index must contain a weight_map object")
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("safetensors index metadata must be an object when present")
    return SafetensorsIndex(
        metadata=metadata,
        weight_map={str(key): str(value) for key, value in weight_map.items()},
    )


def _required_path(path: str | Path, *, label: str, directory: bool) -> Path:
    resolved = Path(path).expanduser().resolve()
    valid = resolved.is_dir() if directory else resolved.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise ValueError(f"{label} is not a {kind}: {path}")
    return resolved


def _expect(label: str, actual: object, expected: object) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{label} must be {expected!r}, found {actual!r}")


def _expect_sha256(label: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _resolved_recorded_path(
    payload: Mapping[str, Any],
    field: str,
    *,
    label: str,
) -> Path:
    raw = payload.get(field)
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label}.{field} must be a path string")
    return Path(raw).expanduser().resolve()


def validate_glm52_composite_claims(
    payload: Mapping[str, Any],
    *,
    expected_group_keys: Sequence[str],
    routed_artifact_dir: str | Path,
    non_vq_artifact_dir: str | Path,
    source_dir: str | Path,
    non_vq_evidence_json: str | Path,
    non_vq_evidence_sha256: str,
    model_id: str = PINNED_GLM52_MODEL_ID,
    revision: str = PINNED_GLM52_REVISION,
    config_sha256: str | None = None,
    index_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate submitted composite claims before any expensive payload work."""

    expected_keys = tuple(expected_group_keys)
    if len(expected_keys) != GLM52_REAP_EXPECTED_GROUPS:
        raise ValueError("expected routed inventory must contain exactly 225 groups")
    expected_layers = list(GLM52_REAP_EXPECTED_LAYER_IDS)
    expected = {
        "schema_version": 1,
        "record_type": "glm52_modelopt_nvfp4_artifact_audit",
        "audit_status": "glm52_modelopt_nvfp4_artifact_audit_ready",
        "audit_pass": True,
        "artifact_integrity_pass": True,
        "audit_blockers": [],
        "materialization_scope": "full",
        "model_id": model_id,
        "source_revision": revision,
        "expected_group_keys": list(expected_keys),
        "ready_group_keys": list(expected_keys),
        "complete_layer_ids": expected_layers,
        "partial_layer_ids": [],
        "full_routed_artifact_ready": True,
        "dense_routed_experts": False,
        "resume_verified": True,
        "byte_identity_verified": True,
        "requested_code_bits": 8,
        "requested_group_size": 512,
        "requested_scale_estimator": "max_abs",
        "accounting_scope": "full_composite_actual",
        "actual_routed_weight_count": GLM52_ROUTED_PARAMETER_COUNT,
        "actual_routed_payload_bytes": GLM52_ROUTED_PAYLOAD_BYTES,
        "actual_routed_codebook_bytes": GLM52_ROUTED_CODEBOOK_BYTES,
        "main_non_routed_parameter_count": GLM52_NON_VQ_PARAMETER_COUNT,
        "main_non_routed_tensor_payload_bytes": GLM52_NON_VQ_PAYLOAD_BYTES,
        "main_model_parameter_count_excluding_mtp": (
            GLM52_MAIN_MODEL_PARAMETER_COUNT
        ),
        "actual_whole_model_tensor_payload_bytes": (
            GLM52_ACCEPTED_TENSOR_PAYLOAD_BYTES
        ),
        "whole_model_tensor_payload_values_actual": True,
        "whole_model_physical_values_actual": True,
        "non_vq_package_evidence_authenticated": True,
        "non_vq_evidence_sha256": non_vq_evidence_sha256,
    }
    labels = {
        "full_routed_artifact_ready": "full routed artifact readiness",
        "non_vq_package_evidence_authenticated": (
            "non-VQ package evidence authentication"
        ),
        "dense_routed_experts": "dense routed experts",
    }
    for field, value in expected.items():
        _expect(
            f"composite audit {labels.get(field, field)}",
            payload.get(field),
            value,
        )
    if config_sha256 is not None:
        _expect("composite audit config_sha256", payload.get("config_sha256"), config_sha256)
    if index_sha256 is not None:
        _expect("composite audit index_sha256", payload.get("index_sha256"), index_sha256)

    routed_root = Path(routed_artifact_dir).expanduser().resolve()
    non_vq_root = Path(non_vq_artifact_dir).expanduser().resolve()
    source_root = Path(source_dir).expanduser().resolve()
    non_vq_evidence_path = Path(non_vq_evidence_json).expanduser().resolve()
    if _resolved_recorded_path(
        payload,
        "artifact_dir",
        label="composite audit",
    ) != routed_root:
        raise ValueError("composite audit routed artifact path is stale or mismatched")
    if _resolved_recorded_path(
        payload,
        "manifest_path",
        label="composite audit",
    ) != routed_root / "conversion-manifest.json":
        raise ValueError("composite audit routed manifest path is stale or mismatched")
    if _resolved_recorded_path(
        payload,
        "non_vq_artifact_dir",
        label="composite audit",
    ) != non_vq_root:
        raise ValueError("composite audit non-VQ artifact path is stale or mismatched")
    if _resolved_recorded_path(
        payload,
        "non_vq_evidence_json",
        label="composite audit",
    ) != non_vq_evidence_path:
        raise ValueError("composite audit non-VQ evidence path is stale or mismatched")
    if _resolved_recorded_path(
        payload,
        "source_dir",
        label="composite audit",
    ) != source_root:
        raise ValueError("composite audit source path is stale or mismatched")

    raw_bpw = payload.get("actual_whole_model_tensor_payload_bpw")
    if not isinstance(raw_bpw, (int, float)) or isinstance(raw_bpw, bool):
        raise ValueError("composite audit actual tensor payload bpw must be numeric")
    if float(raw_bpw) != GLM52_ACCEPTED_TENSOR_PAYLOAD_BPW:
        raise ValueError("composite audit actual tensor payload bpw is not accepted")

    non_vq = payload.get("non_vq_package_audit")
    if not isinstance(non_vq, Mapping):
        raise ValueError("composite audit non-VQ package audit must be an object")
    for field, value in {
        "audit_pass": True,
        "parameter_count": GLM52_NON_VQ_PARAMETER_COUNT,
        "tensor_payload_bytes": GLM52_NON_VQ_PAYLOAD_BYTES,
    }.items():
        _expect(f"composite non-VQ {field}", non_vq.get(field), value)
    if _resolved_recorded_path(
        non_vq,
        "artifact_dir",
        label="composite non-VQ audit",
    ) != non_vq_root:
        raise ValueError("composite non-VQ audit artifact path is stale or mismatched")
    if _resolved_recorded_path(
        non_vq,
        "manifest_path",
        label="composite non-VQ audit",
    ) != non_vq_root / "non-vq-manifest.json":
        raise ValueError("composite non-VQ manifest path is stale or mismatched")
    return dict(payload)


def validate_glm52_full_bind_claims(
    payload: Mapping[str, Any],
    *,
    expected_group_keys: Sequence[str],
    routed_artifact_dir: str | Path,
    non_vq_artifact_dir: str | Path,
    model_id: str = PINNED_GLM52_MODEL_ID,
    revision: str = PINNED_GLM52_REVISION,
    profile_sha256: str | None = None,
    config_sha256: str | None = None,
    index_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate complete header-only evidence without crediting runtime proof."""

    expected_keys = tuple(expected_group_keys)
    expected = {
        "schema_version": 1,
        "record_type": "glm52_full_bind_preflight",
        "preflight_status": "glm52_full_bind_preflight_ready",
        "preflight_pass": True,
        "blockers": [],
        "profile": GLM52_REAP_PROFILE_NAME,
        "model_id": model_id,
        "source_revision": revision,
        "expected_routed_group_keys": list(expected_keys),
        "present_routed_group_keys": list(expected_keys),
        "missing_routed_group_keys": [],
        "expected_routed_group_count": GLM52_REAP_EXPECTED_GROUPS,
        "present_routed_group_count": GLM52_REAP_EXPECTED_GROUPS,
        "missing_routed_group_count": 0,
        "non_vq_tensor_count": 1_194,
        "non_vq_runtime_target_count": GLM52_EXPECTED_NON_VQ_RUNTIME_TARGETS,
        "non_vq_header_file_count": 9,
        "kv_b_source_tensor_count": 78,
        "kv_b_runtime_target_count": 156,
        "routed_header_file_count": GLM52_REAP_EXPECTED_GROUPS,
        "header_only": True,
        "tensor_payloads_read": False,
        "payload_hashes_verified": False,
        "full_model_constructed": False,
        "dense_routed_experts": False,
        "production_binding_proven": False,
        "production_generation_proven": False,
    }
    for field, value in expected.items():
        _expect(f"full-bind {field}", payload.get(field), value)
    for field in ("profile_sha256", "profile_contract_sha256"):
        _expect_sha256(f"full-bind {field}", payload.get(field))
    if profile_sha256 is not None:
        _expect("full-bind profile_sha256", payload.get("profile_sha256"), profile_sha256)
    if config_sha256 is not None:
        _expect("full-bind config_sha256", payload.get("config_sha256"), config_sha256)
    if index_sha256 is not None:
        _expect("full-bind source_index_sha256", payload.get("source_index_sha256"), index_sha256)
    if _resolved_recorded_path(
        payload,
        "routed_artifact_dir",
        label="full-bind",
    ) != Path(routed_artifact_dir).expanduser().resolve():
        raise ValueError("full-bind routed artifact path is stale or mismatched")
    if _resolved_recorded_path(
        payload,
        "non_vq_artifact_dir",
        label="full-bind",
    ) != Path(non_vq_artifact_dir).expanduser().resolve():
        raise ValueError("full-bind non-VQ artifact path is stale or mismatched")
    return dict(payload)


def validate_glm52_non_vq_evidence_claims(
    payload: Mapping[str, Any],
    *,
    fresh_audit: Mapping[str, Any],
    fresh_manifest: Mapping[str, Any],
    non_vq_artifact_dir: str | Path,
    config_sha256: str,
    index_sha256: str,
    model_id: str = PINNED_GLM52_MODEL_ID,
    revision: str = PINNED_GLM52_REVISION,
) -> dict[str, Any]:
    expected = {
        "schema_version": 1,
        "record_type": "glm52_non_vq_package_manifest",
        "pack_status": "glm52_non_vq_package_ready",
        "production_ready": True,
        "package_audit_pass": True,
        "profile": GLM52_REAP_PROFILE_NAME,
        "model_id": model_id,
        "source_revision": revision,
        "config_sha256": config_sha256,
        "index_sha256": index_sha256,
        "parameter_count": GLM52_NON_VQ_PARAMETER_COUNT,
        "tensor_payload_bytes": GLM52_NON_VQ_PAYLOAD_BYTES,
    }
    for field, value in expected.items():
        _expect(f"non-VQ evidence {field}", payload.get(field), value)
    for field in (
        "manifest_sha256",
        "package_index_sha256",
        "package_set_sha256",
        "source_blob_inventory_sha256",
        "source_inventory_sha256",
    ):
        _expect_sha256(f"non-VQ evidence {field}", payload.get(field))
    for field in (
        "model_id",
        "source_revision",
        "config_sha256",
        "index_sha256",
        "package_index_sha256",
        "package_set_sha256",
        "source_blob_inventory_sha256",
        "source_inventory_sha256",
        "parameter_count",
        "tensor_payload_bytes",
    ):
        _expect(
            f"non-VQ evidence {field}",
            payload.get(field),
            fresh_manifest.get(field),
        )

    root = Path(non_vq_artifact_dir).expanduser().resolve()
    normalized_fresh = _normalized_audit_report(
        fresh_audit,
        artifact_root=root,
        manifest_name="non-vq-manifest.json",
    )
    recorded = payload.get("package_audit")
    if not isinstance(recorded, Mapping):
        raise ValueError("non-VQ evidence package_audit must be an object")
    if _resolved_recorded_path(
        fresh_audit,
        "artifact_dir",
        label="fresh non-VQ audit",
    ) != root:
        raise ValueError("fresh non-VQ audit artifact_dir is stale or mismatched")
    if _resolved_recorded_path(
        fresh_audit,
        "manifest_path",
        label="fresh non-VQ audit",
    ) != root / "non-vq-manifest.json":
        raise ValueError("fresh non-VQ audit manifest_path is stale or mismatched")
    if _resolved_recorded_path(
        recorded,
        "artifact_dir",
        label="non-VQ evidence package_audit",
    ) != root:
        raise ValueError("non-VQ evidence package_audit artifact_dir is stale")
    if _resolved_recorded_path(
        recorded,
        "manifest_path",
        label="non-VQ evidence package_audit",
    ) != root / "non-vq-manifest.json":
        raise ValueError("non-VQ evidence package_audit manifest_path is stale")
    normalized_recorded = _normalized_audit_report(
        recorded,
        artifact_root=root,
        manifest_name="non-vq-manifest.json",
    )
    if normalized_recorded != normalized_fresh:
        raise ValueError("non-VQ evidence package_audit does not match fresh audit")
    for field in ("manifest_sha256", "package_set_sha256"):
        _expect(
            f"non-VQ evidence {field}",
            payload.get(field),
            normalized_fresh.get(field),
        )
    return dict(payload)


def build_glm52_composite_artifact_identity(
    *,
    profile: str,
    profile_sha256: str,
    profile_contract_sha256: str,
    composite_audit: Mapping[str, Any],
    non_vq_evidence: Mapping[str, Any],
) -> GLM52CompositeArtifactIdentity:
    """Build the path-free identity shared by runtime, eval, and benchmark evidence."""

    non_vq_audit = composite_audit.get("non_vq_package_audit")
    if not isinstance(non_vq_audit, Mapping):
        raise ValueError("composite audit is missing non-VQ package identity")
    expected_keys = composite_audit.get("expected_group_keys")
    if not isinstance(expected_keys, list) or len(expected_keys) != 225:
        raise ValueError("composite audit must identify exactly 225 routed groups")
    body: dict[str, object] = {
        "schema_version": 1,
        "identity_kind": "glm52_production_composite_v1",
        "model_id": composite_audit.get("model_id"),
        "source_revision": composite_audit.get("source_revision"),
        "profile": profile,
        "profile_sha256": profile_sha256,
        "profile_contract_sha256": profile_contract_sha256,
        "config_sha256": composite_audit.get("config_sha256"),
        "source_index_sha256": composite_audit.get("index_sha256"),
        "source_blob_inventory_sha256": non_vq_evidence.get(
            "source_blob_inventory_sha256"
        ),
        "source_inventory_sha256": non_vq_evidence.get("source_inventory_sha256"),
        "non_vq_manifest_sha256": non_vq_audit.get("manifest_sha256"),
        "non_vq_package_index_sha256": non_vq_evidence.get(
            "package_index_sha256"
        ),
        "non_vq_package_set_sha256": non_vq_audit.get("package_set_sha256"),
        "routed_manifest_sha256": composite_audit.get("manifest_sha256"),
        "routed_group_set_sha256": composite_audit.get("group_set_sha256"),
        "routed_group_inventory_sha256": _canonical_sha256(expected_keys),
        "routed_group_count": len(expected_keys),
        "code_bits": composite_audit.get("requested_code_bits"),
        "group_size": composite_audit.get("requested_group_size"),
        "scale_estimator": composite_audit.get("requested_scale_estimator"),
        "whole_model_parameter_count": composite_audit.get(
            "main_model_parameter_count_excluding_mtp"
        ),
        "whole_model_tensor_payload_bytes": composite_audit.get(
            "actual_whole_model_tensor_payload_bytes"
        ),
        "whole_model_tensor_payload_bpw": GLM52_ACCEPTED_TENSOR_PAYLOAD_BPW,
    }
    return GLM52CompositeArtifactIdentity(body=body, sha256=_canonical_sha256(body))


def validate_glm52_routed_manifest_policy(
    payload: Mapping[str, Any],
    *,
    composite_audit: Mapping[str, Any],
) -> dict[str, object]:
    """Bind the production identity to the freshly audited conversion policy."""

    policy: dict[str, object] = {
        "code_bits": payload.get("code_bits"),
        "group_size": payload.get("group_size"),
        "scale_estimator": payload.get("scale_estimator"),
    }
    for field, expected in {
        "code_bits": 8,
        "group_size": 512,
        "scale_estimator": "max_abs",
    }.items():
        label = field.replace("_", " ")
        _expect(f"routed manifest {label}", policy[field], expected)
        _expect(
            f"composite requested {label}",
            composite_audit.get(f"requested_{field}"),
            expected,
        )
    return policy


def _normalized_audit_report(
    payload: Mapping[str, Any],
    *,
    artifact_root: Path,
    manifest_name: str,
) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["artifact_dir"] = str(artifact_root)
    normalized["manifest_path"] = str(artifact_root / manifest_name)
    return normalized


def _report_dict(report: Any) -> dict[str, Any]:
    if isinstance(report, Mapping):
        return dict(report)
    to_dict = getattr(report, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return dict(value)
    if hasattr(report, "__dataclass_fields__"):
        return asdict(report)
    raise TypeError(f"audit result {type(report).__name__} has no mapping representation")


def validate_glm52_production_inputs(
    *,
    profile_path: str | Path,
    config_path: str | Path,
    source_index_path: str | Path,
    tokenizer_dir: str | Path,
    tokenizer_readiness_json: str | Path,
    family_policy_json: str | Path,
    non_vq_artifact_dir: str | Path,
    non_vq_evidence_json: str | Path,
    routed_artifact_dir: str | Path,
    composite_audit_json: str | Path,
    materialization_runs_jsonl: str | Path,
    full_bind_preflight_json: str | Path,
    model_id: str,
    revision: str,
    prompt: str,
    expected_config_sha256: str = GLM52_REAP_CONFIG_SHA256,
    expected_index_sha256: str = GLM52_REAP_INDEX_SHA256,
    non_vq_auditor: Callable[..., Any] = audit_glm52_non_vq_package,
    routed_auditor: Callable[..., Any] = audit_glm52_reap_materialization_manifest,
    full_bind_authenticator: Callable[..., Any] = preflight_glm52_full_bind,
    source_accounting_auditor: Callable[..., Any] = audit_glm52_reap_source_accounting,
    family_policy_validator: Callable[..., Any] = validate_glm52_family_gate_policy,
    tokenizer_authenticator: Callable[..., Any] = validate_glm52_tokenizer_readiness,
) -> GLM52ValidatedProductionInputs:
    """Authenticate every production authority and freshly re-audit both payloads."""

    profile_input = _required_path(profile_path, label="profile", directory=False)
    config_input = _required_path(config_path, label="config", directory=False)
    source_index_input = _required_path(
        source_index_path,
        label="source index",
        directory=False,
    )
    tokenizer_root = _required_path(tokenizer_dir, label="tokenizer", directory=True)
    non_vq_root = _required_path(
        non_vq_artifact_dir,
        label="non-VQ artifact",
        directory=True,
    )
    routed_root = _required_path(
        routed_artifact_dir,
        label="routed artifact",
        directory=True,
    )
    evidence_paths = {
        "tokenizer_readiness_json": _required_path(
            tokenizer_readiness_json,
            label="tokenizer readiness evidence",
            directory=False,
        ),
        "family_policy_json": _required_path(
            family_policy_json,
            label="family policy",
            directory=False,
        ),
        "non_vq_evidence_json": _required_path(
            non_vq_evidence_json,
            label="non-VQ evidence",
            directory=False,
        ),
        "composite_audit_json": _required_path(
            composite_audit_json,
            label="composite audit",
            directory=False,
        ),
        "materialization_runs_jsonl": _required_path(
            materialization_runs_jsonl,
            label="materialization runs",
            directory=False,
        ),
        "full_bind_preflight_json": _required_path(
            full_bind_preflight_json,
            label="full-bind preflight",
            directory=False,
        ),
    }
    pre_validation_authorities = {
        "profile_path": profile_input,
        "config_path": config_input,
        "source_index_path": source_index_input,
        **evidence_paths,
    }
    pre_validation_bytes = {
        name: _read_stable_authority_bytes(path, label=name)
        for name, path in pre_validation_authorities.items()
    }
    pre_validation_hashes = {
        name: hashlib.sha256(raw).hexdigest()
        for name, raw in pre_validation_bytes.items()
    }
    if config_input != (tokenizer_root / "config.json").resolve():
        raise ValueError("config must be the pinned tokenizer snapshot config.json")
    if source_index_input != (
        tokenizer_root / "model.safetensors.index.json"
    ).resolve():
        raise ValueError("source index must be the pinned tokenizer snapshot index")

    profile = _load_profile_bytes(
        pre_validation_bytes["profile_path"],
        path=profile_input,
    )
    config = _load_json_object_bytes(
        pre_validation_bytes["config_path"],
        path=config_input,
        label="config",
    )
    source_index = _load_safetensors_index_bytes(
        pre_validation_bytes["source_index_path"],
        path=source_index_input,
    )
    family_policy = _load_json_object_bytes(
        pre_validation_bytes["family_policy_json"],
        path=evidence_paths["family_policy_json"],
        label="family policy",
    )
    tokenizer_evidence = _load_json_object_bytes(
        pre_validation_bytes["tokenizer_readiness_json"],
        path=evidence_paths["tokenizer_readiness_json"],
        label="tokenizer readiness",
    )
    non_vq_evidence = _load_json_object_bytes(
        pre_validation_bytes["non_vq_evidence_json"],
        path=evidence_paths["non_vq_evidence_json"],
        label="non-VQ evidence",
    )
    composite = _load_json_object_bytes(
        pre_validation_bytes["composite_audit_json"],
        path=evidence_paths["composite_audit_json"],
        label="composite audit",
    )
    preflight = _load_json_object_bytes(
        pre_validation_bytes["full_bind_preflight_json"],
        path=evidence_paths["full_bind_preflight_json"],
        label="full-bind preflight",
    )
    _expect("requested model_id", model_id, PINNED_GLM52_MODEL_ID)
    _expect("requested revision", revision, PINNED_GLM52_REVISION)
    _expect("profile name", profile.name, GLM52_REAP_PROFILE_NAME)
    _expect("profile model_id", profile.hf_model_id, model_id)
    _expect("profile revision", profile.revision, revision)
    mismatches = validate_profile_against_hf_config_data(profile, config)
    if mismatches:
        raise ValueError(f"profile/config mismatch: {mismatches}")
    config_sha256 = pre_validation_hashes["config_path"]
    index_sha256 = pre_validation_hashes["source_index_path"]
    _expect("config SHA-256", config_sha256, expected_config_sha256)
    _expect("source index SHA-256", index_sha256, expected_index_sha256)
    expected_keys = expected_glm52_routed_group_keys(profile, config)

    validated_policy = family_policy_validator(family_policy)
    artifact_target = validated_policy.get("artifact_target")
    expected_target = {
        "tensor_payload_bytes": GLM52_ACCEPTED_TENSOR_PAYLOAD_BYTES,
        "whole_main_bpw": 1.5934433,
        "target_accepted_by_user": True,
        "actual_full_artifact_required": True,
    }
    _expect("family policy artifact target", artifact_target, expected_target)

    tokenizer_authenticator(tokenizer_evidence, tokenizer_dir=tokenizer_root)
    _expect("tokenizer readiness prompt", tokenizer_evidence.get("prompt"), prompt)

    non_vq_evidence_sha256 = pre_validation_hashes["non_vq_evidence_json"]
    validate_glm52_composite_claims(
        composite,
        expected_group_keys=expected_keys,
        routed_artifact_dir=routed_root,
        non_vq_artifact_dir=non_vq_root,
        source_dir=tokenizer_root,
        non_vq_evidence_json=evidence_paths["non_vq_evidence_json"],
        non_vq_evidence_sha256=non_vq_evidence_sha256,
        model_id=model_id,
        revision=revision,
        config_sha256=config_sha256,
        index_sha256=index_sha256,
    )
    validate_glm52_full_bind_claims(
        preflight,
        expected_group_keys=expected_keys,
        routed_artifact_dir=routed_root,
        non_vq_artifact_dir=non_vq_root,
        model_id=model_id,
        revision=revision,
        profile_sha256=pre_validation_hashes["profile_path"],
        config_sha256=config_sha256,
        index_sha256=index_sha256,
    )

    fingerprint = snapshot_glm52_production_inputs(
        authority_files=(
            *pre_validation_authorities.values(),
            non_vq_root / "non-vq-manifest.json",
            non_vq_root / "model.safetensors.index.json",
            routed_root / "conversion-manifest.json",
            *(
                tokenizer_root / filename
                for filename in GLM52_PINNED_TOKENIZER_FILES
            ),
        ),
        artifact_roots=(tokenizer_root, non_vq_root, routed_root),
    )
    for name, path in pre_validation_authorities.items():
        current = fingerprint.authority_file_sha256[str(path.resolve())]
        _expect(
            f"authenticated authority stability for {name}",
            current,
            pre_validation_hashes[name],
        )

    fresh_preflight = _report_dict(
        full_bind_authenticator(
            profile=profile,
            profile_path=profile_input,
            config_path=config_input,
            source_index_path=source_index_input,
            non_vq_artifact_dir=non_vq_root,
            routed_artifact_dir=routed_root,
            model_id=model_id,
            revision=revision,
        )
    )
    if fresh_preflight != preflight:
        raise ValueError("full-bind preflight evidence does not match fresh inventory")

    fresh_non_vq_report = non_vq_auditor(
        non_vq_root,
        source_dir=tokenizer_root,
        source_index=source_index,
        profile=profile,
        expected_model_id=model_id,
        expected_revision=revision,
        expected_config_sha256=config_sha256,
        expected_index_sha256=index_sha256,
        config_path=config_input,
        index_path=source_index_input,
        enforce_pinned_source=True,
    )
    fresh_non_vq = _normalized_audit_report(
        _report_dict(fresh_non_vq_report),
        artifact_root=non_vq_root,
        manifest_name="non-vq-manifest.json",
    )
    validate_glm52_non_vq_evidence_claims(
        non_vq_evidence,
        fresh_audit=fresh_non_vq,
        fresh_manifest=_load_json_object(
            non_vq_root / "non-vq-manifest.json",
            label="fresh non-VQ manifest",
        ),
        non_vq_artifact_dir=non_vq_root,
        config_sha256=config_sha256,
        index_sha256=index_sha256,
        model_id=model_id,
        revision=revision,
    )
    composite_non_vq = composite.get("non_vq_package_audit")
    if not isinstance(composite_non_vq, Mapping):
        raise ValueError("composite non-VQ package audit must be an object")
    if _normalized_audit_report(
        composite_non_vq,
        artifact_root=non_vq_root,
        manifest_name="non-vq-manifest.json",
    ) != fresh_non_vq:
        raise ValueError(
            "composite non-VQ package audit does not match the fresh payload audit"
        )

    accounting = source_accounting_auditor(
        tokenizer_root,
        index=source_index,
        profile=profile,
    )
    _expect(
        "source main-model parameter count",
        accounting.main_model_parameter_count_excluding_mtp,
        GLM52_MAIN_MODEL_PARAMETER_COUNT,
    )
    fresh_routed_report = routed_auditor(
        routed_root,
        profile=profile,
        expected_group_keys=expected_keys,
        expected_code_bits=8,
        expected_config_sha256=config_sha256,
        expected_index_sha256=index_sha256,
        evidence_paths=(evidence_paths["materialization_runs_jsonl"],),
        non_routed_artifact_bytes=fresh_non_vq["artifact_tree_bytes"],
        whole_model_parameter_count=GLM52_MAIN_MODEL_PARAMETER_COUNT,
    )
    fresh_routed = _normalized_audit_report(
        _report_dict(fresh_routed_report),
        artifact_root=routed_root,
        manifest_name="conversion-manifest.json",
    )
    validate_glm52_routed_manifest_policy(
        _load_json_object(
            routed_root / "conversion-manifest.json",
            label="fresh routed manifest",
        ),
        composite_audit=composite,
    )
    normalized_composite = _normalized_audit_report(
        composite,
        artifact_root=routed_root,
        manifest_name="conversion-manifest.json",
    )
    for field, value in fresh_routed.items():
        if normalized_composite.get(field) != value:
            raise ValueError(
                f"composite routed audit field {field} does not match fresh audit"
            )

    actual_bytes = (
        int(fresh_routed["actual_routed_payload_bytes"])
        + int(fresh_routed["actual_routed_codebook_bytes"])
        + int(fresh_non_vq["tensor_payload_bytes"])
    )
    _expect(
        "fresh composite tensor payload bytes",
        actual_bytes,
        GLM52_ACCEPTED_TENSOR_PAYLOAD_BYTES,
    )
    _expect(
        "composite actual tensor payload bytes",
        composite.get("actual_whole_model_tensor_payload_bytes"),
        actual_bytes,
    )

    profile_contract_sha256 = str(preflight.get("profile_contract_sha256"))
    identity = build_glm52_composite_artifact_identity(
        profile=profile.name,
        profile_sha256=pre_validation_hashes["profile_path"],
        profile_contract_sha256=profile_contract_sha256,
        composite_audit=composite,
        non_vq_evidence=non_vq_evidence,
    )
    evidence_hashes = {
        name: pre_validation_hashes[name] for name in evidence_paths
    }
    non_vq_index = load_safetensors_index(
        non_vq_root / "model.safetensors.index.json"
    )
    assert_glm52_production_inputs_unchanged(fingerprint)
    return GLM52ValidatedProductionInputs(
        profile=profile,
        config=dict(config),
        non_vq_index=non_vq_index,
        tokenizer_dir=tokenizer_root,
        tokenizer_readiness=dict(tokenizer_evidence),
        non_vq_artifact_dir=non_vq_root,
        routed_artifact_dir=routed_root,
        artifact_identity=identity,
        input_evidence_file_sha256=evidence_hashes,
        input_fingerprint=fingerprint,
        prompt=prompt,
    )


def validate_glm52_recovery_candidate_inputs(
    baseline: GLM52ValidatedProductionInputs,
    *,
    recovery_dir: str | Path,
    expected_seed_manifest_sha256: str,
    expected_stats_manifest_sha256: str,
    expected_full_source_blob_inventory_sha256: str,
    expected_routed_source_blob_inventory_sha256: str,
    expected_recovery_lever: str,
    expected_recovery_policy: Mapping[str, str],
    accepted_composite_audit_json: str | Path,
    expected_composite_audit_sha256: str,
    expected_recovery_candidate_identity_sha256: str,
    expected_recovery_manifest_body_sha256: str,
    recovery_auditor: Callable[..., GLM52RecoveryArtifactAudit] = (
        audit_glm52_recovery_mixed_artifact
    ),
    integrity_checker: Callable[[GLM52ProductionInputFingerprint], Any] = (
        assert_glm52_production_inputs_unchanged
    ),
) -> GLM52ValidatedRecoveryCandidate:
    """Authenticate a recovery candidate without changing its accepted baseline."""

    integrity_checker(baseline.input_fingerprint)
    root = _required_path(
        recovery_dir,
        label="recovery directory",
        directory=True,
    )
    expected_candidate_identity = _expect_sha256(
        "expected recovery candidate identity",
        expected_recovery_candidate_identity_sha256,
    )
    expected_manifest_identity = _expect_sha256(
        "expected recovery manifest body",
        expected_recovery_manifest_body_sha256,
    )
    expected_composite_identity = _expect_sha256(
        "expected accepted composite audit",
        expected_composite_audit_sha256,
    )
    audit = recovery_auditor(
        root,
        profile=baseline.profile,
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
        expected_composite_audit_sha256=expected_composite_identity,
    )
    _expect("recovery audit pass", audit.audit_pass, True)
    _expect(
        "recovery candidate identity",
        audit.candidate_identity_sha256,
        expected_candidate_identity,
    )
    _expect(
        "recovery manifest body identity",
        audit.manifest_body_sha256,
        expected_manifest_identity,
    )
    _expect(
        "recovery accepted composite audit identity",
        audit.accepted_composite_audit_sha256,
        expected_composite_identity,
    )
    _expect(
        "recovery accepted baseline composite identity",
        audit.accepted_baseline_composite_identity_sha256,
        baseline.artifact_identity.sha256,
    )
    audited_root = Path(audit.recovery_dir).expanduser().resolve()
    if audited_root != root:
        raise ValueError("recovery directory does not match the audited recovery root")
    manifest = _required_path(
        root / "conversion-manifest.json",
        label="recovery manifest",
        directory=False,
    )
    if Path(audit.manifest_path).expanduser().resolve() != manifest:
        raise ValueError(
            "recovery manifest is not rooted in the audited recovery directory"
        )
    artifact_root = _required_path(
        root / "artifact",
        label="recovery bind artifact",
        directory=True,
    )
    integrity_checker(baseline.input_fingerprint)
    audit.verify_current_identity()
    return GLM52ValidatedRecoveryCandidate(
        baseline=baseline,
        recovery_dir=root,
        recovery_artifact_dir=artifact_root,
        recovery_audit=audit,
        recovery_candidate_identity_sha256=expected_candidate_identity,
        recovery_manifest_body_sha256=expected_manifest_identity,
    )


def _snapshot_glm52_recovery_bind_artifacts(
    validated: GLM52ValidatedRecoveryCandidate,
    *,
    snapshot_scratch_dir: str | Path,
    allow_copy_fallback: bool,
) -> _GLM52RecoveryBindSnapshot:
    """Pin the exact audited inodes used by the routed binder."""

    scratch_root = _required_path(
        snapshot_scratch_dir,
        label="recovery snapshot scratch directory",
        directory=True,
    )
    seen: set[str] = set()
    snapshots: list[DescriptorSnapshot] = []
    remaining_copy_bytes = sum(
        int(group.artifact_bytes) for group in validated.recovery_audit.groups
    )
    try:
        for group in validated.recovery_audit.groups:
            filename = group.filename
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError("recovery audit group filename must be a basename")
            if filename in seen:
                raise ValueError("recovery audit group filenames must be unique")
            seen.add(filename)
            source = validated.recovery_artifact_dir / filename
            authenticated = AuthenticatedFile.open(
                source,
                label=f"recovery bind source {filename}",
                allow_resolved_symlink=True,
            )
            try:
                if authenticated.size != group.artifact_bytes:
                    raise ValueError(f"recovery bind source size drifted: {filename}")
                if authenticated.sha256 != group.artifact_sha256:
                    raise ValueError(f"recovery bind source hash drifted: {filename}")
                snapshots.append(
                    DescriptorSnapshot.create(
                        {filename: authenticated},
                        scratch_dir=scratch_root,
                        allow_copy_fallback=allow_copy_fallback,
                        prefix=".glm52-recovery-bind-",
                        clone_function=_clone_or_copy_open_file,
                        required_free_bytes=remaining_copy_bytes,
                    )
                )
                remaining_copy_bytes -= authenticated.size
            finally:
                authenticated.close()
        return DescriptorSnapshot.combine(tuple(snapshots))
    except BaseException:
        for snapshot in snapshots:
            snapshot.close()
        raise


def load_authenticated_glm52_composite(
    validated: GLM52ValidatedProductionInputs,
    *,
    model_factory: Callable[[Any], Any] = GLM52VQModel,
    args_factory: Callable[[dict[str, Any]], Any] = glm52_vq_args_from_config,
    non_vq_binder: Callable[..., Any] = bind_glm52_non_vq_weights,
    routed_binder: Callable[..., Any] = bind_glm52_vq_experts,
    unbound_checker: Callable[[Any], bool] = has_unbound_vq_experts,
    dense_parameter_checker: Callable[[Any], tuple[str, ...]] = (
        dense_glm52_routed_parameter_names
    ),
    integrity_checker: Callable[[GLM52ProductionInputFingerprint], Any] = (
        assert_glm52_production_inputs_unchanged
    ),
    phase_marker: Callable[[str], Any] | None = None,
) -> tuple[Any, GLM52CompositeLoadReport]:
    """Construct lazily, stream non-VQ first, then bind the exact routed set."""

    mark = phase_marker or (lambda _label: None)
    integrity_checker(validated.input_fingerprint)
    mark("before_model_construction")
    model = model_factory(args_factory(dict(validated.config)))
    model.eval()
    mark("after_model_construction")
    non_vq_report = non_vq_binder(
        model,
        validated.non_vq_artifact_dir,
        validated.non_vq_index,
        strict=True,
    )
    if non_vq_report.loaded_count != GLM52_EXPECTED_NON_VQ_RUNTIME_TARGETS:
        raise ValueError("strict non-VQ bind did not load exactly 1272 runtime targets")
    for field in (
        "missing_model_parameters",
        "skipped_unmatched_tensors",
        "skipped_routed_expert_tensors",
        "skipped_mtp_tensors",
    ):
        if tuple(getattr(non_vq_report, field)):
            raise ValueError(f"strict non-VQ bind report {field} must be empty")
    integrity_checker(validated.input_fingerprint)
    mark("after_non_vq_bind")
    bound_layers = tuple(
        routed_binder(
            model,
            validated.routed_artifact_dir,
            profile=validated.profile,
            strict=True,
        )
    )
    if bound_layers != GLM52_EXPECTED_SPARSE_LAYERS:
        raise ValueError("routed bind must cover exact sparse layers 3 through 77")
    integrity_checker(validated.input_fingerprint)
    mark("after_routed_bind")
    unbound = bool(unbound_checker(model))
    dense_names = tuple(dense_parameter_checker(model))
    if unbound:
        raise ValueError("production model retains unbound VQ experts")
    if dense_names:
        raise ValueError("production model retains dense routed parameters")
    report = GLM52CompositeLoadReport(
        artifact_identity_sha256=validated.artifact_identity.sha256,
        non_vq_bind_report=non_vq_report,
        bound_sparse_layer_ids=bound_layers,
        dense_routed_parameter_names=dense_names,
        unbound_vq_experts=unbound,
    )
    return model, report


def load_authenticated_glm52_recovery_candidate(
    validated: GLM52ValidatedRecoveryCandidate,
    *,
    snapshot_scratch_dir: str | Path,
    allow_snapshot_copy_fallback: bool = False,
    model_factory: Callable[[Any], Any] = GLM52VQModel,
    args_factory: Callable[[dict[str, Any]], Any] = glm52_vq_args_from_config,
    non_vq_binder: Callable[..., Any] = bind_glm52_non_vq_weights,
    routed_binder: Callable[..., Any] = bind_glm52_vq_experts_from_paths,
    unbound_checker: Callable[[Any], bool] = has_unbound_vq_experts,
    dense_parameter_checker: Callable[[Any], tuple[str, ...]] = (
        dense_glm52_routed_parameter_names
    ),
    integrity_checker: Callable[[GLM52ProductionInputFingerprint], Any] = (
        assert_glm52_production_inputs_unchanged
    ),
    phase_marker: Callable[[str], Any] | None = None,
) -> tuple[Any, GLM52RecoveryCandidateLoadReport]:
    """Bind a recovery candidate while retaining its accepted baseline authority."""

    baseline = validated.baseline
    mark = phase_marker or (lambda _label: None)

    def verify_both_identities() -> None:
        integrity_checker(baseline.input_fingerprint)
        validated.recovery_audit.verify_current_identity()

    verify_both_identities()
    snapshot = _snapshot_glm52_recovery_bind_artifacts(
        validated,
        snapshot_scratch_dir=snapshot_scratch_dir,
        allow_copy_fallback=allow_snapshot_copy_fallback,
    )
    try:
        mark("before_model_construction")
        model = model_factory(args_factory(dict(baseline.config)))
        model.eval()
        mark("after_model_construction")
        non_vq_report = non_vq_binder(
            model,
            baseline.non_vq_artifact_dir,
            baseline.non_vq_index,
            strict=True,
        )
        if non_vq_report.loaded_count != GLM52_EXPECTED_NON_VQ_RUNTIME_TARGETS:
            raise ValueError(
                "strict non-VQ bind did not load exactly 1272 runtime targets"
            )
        for field in (
            "missing_model_parameters",
            "skipped_unmatched_tensors",
            "skipped_routed_expert_tensors",
            "skipped_mtp_tensors",
        ):
            if tuple(getattr(non_vq_report, field)):
                raise ValueError(f"strict non-VQ bind report {field} must be empty")
        verify_both_identities()
        mark("after_non_vq_bind")
        bound_layers = tuple(
            routed_binder(
                model,
                snapshot.file_paths,
                profile=baseline.profile,
                strict=True,
            )
        )
        if bound_layers != GLM52_EXPECTED_SPARSE_LAYERS:
            raise ValueError("routed bind must cover exact sparse layers 3 through 77")
        verify_both_identities()
        mark("after_routed_bind")
        unbound = bool(unbound_checker(model))
        dense_names = tuple(dense_parameter_checker(model))
        if unbound:
            raise ValueError("production model retains unbound VQ experts")
        if dense_names:
            raise ValueError("production model retains dense routed parameters")
        setattr(model, "_glm52_recovery_bind_snapshot", snapshot)
        try:
            finalizer = weakref.finalize(model, snapshot.close)
        except TypeError:
            finalizer = None
        setattr(model, "_glm52_recovery_bind_finalizer", finalizer)
        report = GLM52RecoveryCandidateLoadReport(
            baseline_composite_identity_sha256=baseline.artifact_identity.sha256,
            recovery_candidate_identity_sha256=(
                validated.recovery_candidate_identity_sha256
            ),
            recovery_manifest_body_sha256=validated.recovery_manifest_body_sha256,
            non_vq_bind_report=non_vq_report,
            bound_sparse_layer_ids=bound_layers,
            dense_routed_parameter_names=dense_names,
            unbound_vq_experts=unbound,
        )
        return model, report
    except BaseException:
        snapshot.close()
        raise


def close_authenticated_glm52_recovery_candidate(model: Any) -> None:
    """Release retained recovery snapshot descriptors deterministically."""

    finalizer = getattr(model, "_glm52_recovery_bind_finalizer", None)
    if finalizer is not None and finalizer.alive:
        finalizer()
        return
    snapshot = getattr(model, "_glm52_recovery_bind_snapshot", None)
    if isinstance(snapshot, _GLM52RecoveryBindSnapshot):
        snapshot.close()


__all__ = [
    "GLM52_ACCEPTED_TENSOR_PAYLOAD_BPW",
    "GLM52_ACCEPTED_TENSOR_PAYLOAD_BYTES",
    "GLM52CompositeArtifactIdentity",
    "GLM52CompositeLoadReport",
    "GLM52ProductionInputFingerprint",
    "GLM52RecoveryCandidateLoadReport",
    "GLM52ValidatedRecoveryCandidate",
    "GLM52ValidatedProductionInputs",
    "assert_glm52_production_inputs_unchanged",
    "build_glm52_composite_artifact_identity",
    "close_authenticated_glm52_recovery_candidate",
    "load_authenticated_glm52_composite",
    "load_authenticated_glm52_recovery_candidate",
    "snapshot_glm52_production_inputs",
    "validate_glm52_composite_claims",
    "validate_glm52_full_bind_claims",
    "validate_glm52_non_vq_evidence_claims",
    "validate_glm52_production_inputs",
    "validate_glm52_recovery_candidate_inputs",
    "validate_glm52_routed_manifest_policy",
]
