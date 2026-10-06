from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mlx_vq.build.hashing import teacher_cache_artifact_hash
from mlx_vq.quality.glm52_family import (
    GLM52_FAMILY_GATE_V1_CHECK_NAMES,
    GLM52_FAMILY_GATE_V1_MISSING_REQUIREMENTS,
    GLM52_FAMILY_GATE_V1_SCHEMA_VERSION,
    GLM52_FAMILY_GATE_V2_MISSING_REQUIREMENTS,
    GLM52_FAMILY_GATE_V2_SCHEMA_VERSION,
    GLM52_FAMILY_GATE_V3_MISSING_REQUIREMENTS,
    GLM52_FAMILY_GATE_V3_SCHEMA_VERSION,
    GLM52_FAMILY_GATE_V4_MISSING_REQUIREMENTS,
    GLM52_FAMILY_GATE_V4_SCHEMA_VERSION,
    GLM52_FAMILY_GATE_V5_MISSING_REQUIREMENTS,
    GLM52_FAMILY_GATE_V5_SCHEMA_VERSION,
    GLM52_FAMILY_GATE_V6_MISSING_REQUIREMENTS,
    GLM52_FAMILY_GATE_V6_SCHEMA_VERSION,
    GLM52_MODEL_VOCAB_SIZE,
    GLM52_REAP_CONFIG_SHA256,
    GLM52_REAP_EXPECTED_GROUP_KEYS,
    GLM52_REAP_INDEX_SHA256,
    GLM52_REAP_PROFILE,
    GLM52_REAP_PROFILE_CONTRACT_SHA256,
    GLM52_REAP_PROFILE_SHA256,
    GLM52_REQUIRED_EVAL_SPLITS,
    GLM52_TEACHER_CACHE_SOURCE_EVIDENCE,
    PINNED_GLM52_MODEL_ID,
    PINNED_GLM52_REVISION,
    build_glm52_production_composite_identity,
    canonical_sha256,
    load_glm52_same_machine_benchmark_evidence,
    sha256_file,
    validate_glm52_full_artifact_audit,
    validate_glm52_family_gate_policy,
    validate_glm52_family_prompt_pack,
    validate_glm52_non_vq_evidence,
    validate_glm52_production_generation,
)
from mlx_vq.quality.glm52_teacher_cache import (
    GLM52TeacherCacheAudit,
    GLM52TeacherCacheContract,
    MANIFEST_FILENAME as TEACHER_CACHE_MANIFEST_FILENAME,
    audit_glm52_teacher_cache,
)
from mlx_vq.quality.glm52_candidate_eval import compare_glm52_candidate_caches
from mlx_vq.quality.glm52_route_diagnostics import (
    compare_route_trace_artifacts,
    validate_capture_authority,
)


GATE_RECORD_TYPE = "glm52_family_gate"
GATE_PASSED_STATUS = "glm52_family_gate_passed"
GATE_BLOCKED_STATUS = "glm52_family_gate_blocked"
EXPECTED_ROUTED_GROUP_COUNT = len(GLM52_REAP_EXPECTED_GROUP_KEYS)
TEACHER_SOURCE_KIND = "deterministically_dequantized_pinned_modelopt_nvfp4"
EXPECTED_ROUTED_GROUP_KEYS = GLM52_REAP_EXPECTED_GROUP_KEYS
EXPECTED_TEACHER_CACHE_SOURCE_EVIDENCE = GLM52_TEACHER_CACHE_SOURCE_EVIDENCE


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _reject_non_standard_json_constant(constant: str) -> Any:
    raise ValueError(f"non-standard JSON constant {constant!r}")


def _load_json_authority(
    path: str | Path,
    *,
    label: str,
) -> tuple[dict[str, Any], str]:
    try:
        raw = Path(path).read_bytes()
        value = json.loads(
            raw,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_non_standard_json_constant,
        )
    except (OSError, ValueError) as error:
        raise ValueError(f"could not read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _audit_raw_teacher_cache(
    *,
    cache_root: str | Path,
    manifest_path: str | Path,
    prompt_pack_path: str | Path,
) -> GLM52TeacherCacheAudit:
    root = Path(cache_root).expanduser().absolute()
    submitted_manifest = Path(manifest_path).expanduser().absolute()
    expected_manifest = root / TEACHER_CACHE_MANIFEST_FILENAME
    if submitted_manifest != expected_manifest:
        raise ValueError(
            "teacher cache manifest must be the canonical manifest inside the raw root"
        )
    manifest, manifest_sha256 = _load_json_authority(
        submitted_manifest,
        label="teacher cache manifest",
    )
    for field in ("producer", "source_evidence", "non_vq_package"):
        if not isinstance(manifest.get(field), Mapping):
            raise ValueError(f"teacher cache manifest {field} must be an object")
    if dict(manifest["source_evidence"]) != EXPECTED_TEACHER_CACHE_SOURCE_EVIDENCE:
        raise ValueError(
            "teacher cache source evidence does not match the frozen raw audits"
        )
    contract = GLM52TeacherCacheContract.from_frozen_prompt_pack(
        prompt_pack_path,
        producer=dict(manifest["producer"]),
        source_evidence=dict(manifest["source_evidence"]),
        non_vq_package=dict(manifest["non_vq_package"]),
    )
    audit = audit_glm52_teacher_cache(root, contract=contract)
    final_manifest, final_sha256 = _load_json_authority(
        submitted_manifest,
        label="teacher cache manifest after audit",
    )
    if final_sha256 != manifest_sha256 or final_manifest != manifest:
        raise ValueError("teacher cache manifest changed across strict audit")
    return audit


def _validate_teacher_cache_non_vq_binding(
    manifest: Mapping[str, Any],
    *,
    non_vq_evidence: Mapping[str, Any],
    non_vq_evidence_path: str | Path,
    non_vq_evidence_file_sha256: str,
) -> None:
    cache_package = manifest.get("non_vq_package")
    if not isinstance(cache_package, Mapping):
        raise ValueError("teacher cache non_vq_package must be an object")
    evidence_path = cache_package.get("evidence_path")
    if not isinstance(evidence_path, str) or not evidence_path:
        raise ValueError("teacher cache non-VQ evidence path is missing")
    submitted_path = Path(non_vq_evidence_path).expanduser().absolute()
    recorded_path = Path(evidence_path).expanduser().absolute()
    if recorded_path != submitted_path:
        raise ValueError(
            "teacher cache non-VQ evidence path does not match the raw v2 input"
        )
    expected = {
        "evidence_file_sha256": non_vq_evidence_file_sha256,
        "manifest_sha256": non_vq_evidence.get("manifest_sha256"),
        "package_set_sha256": non_vq_evidence.get("package_set_sha256"),
        "retained_tensor_count": non_vq_evidence.get("retained_tensor_count"),
        "tensor_payload_bytes": non_vq_evidence.get("tensor_payload_bytes"),
    }
    for field, value in expected.items():
        if cache_package.get(field) != value:
            raise ValueError(
                f"teacher cache non-VQ {field} does not match the raw v2 input"
            )


def _teacher_cache_audit_identity(
    audit: GLM52TeacherCacheAudit,
    *,
    manifest_file_sha256: str,
    artifact_sha256: str,
) -> dict[str, Any]:
    return {
        "artifact_sha256": artifact_sha256,
        "cache_content_sha256": audit.cache_content_sha256,
        "manifest_body_sha256": audit.manifest_body_sha256,
        "manifest_file_sha256": manifest_file_sha256,
        "prompt_count": audit.prompt_count,
        "source_token_count": audit.source_token_count,
        "predictor_position_count": audit.predictor_position_count,
        "fp32_value_count": audit.fp32_value_count,
        "raw_tensor_bytes": audit.raw_tensor_bytes,
        "split_position_counts": dict(audit.split_position_counts),
        "split_raw_tensor_bytes": dict(audit.split_raw_tensor_bytes),
        "release_eligible": audit.release_eligible,
    }


def _teacher_cache_audit_evidence(
    audit: GLM52TeacherCacheAudit,
    *,
    manifest_file_sha256: str,
    artifact_sha256: str,
) -> dict[str, Any]:
    identity = _teacher_cache_audit_identity(
        audit,
        manifest_file_sha256=manifest_file_sha256,
        artifact_sha256=artifact_sha256,
    )
    payload = {
        "schema_version": 1,
        "record_type": "glm52_teacher_cache_audit",
        "audit_status": "glm52_teacher_cache_audit_ready",
        "audit_pass": audit.valid,
        "release_eligible": audit.release_eligible,
        "audited_teacher_cache_identity": identity,
        "audited_teacher_cache_identity_sha256": canonical_sha256(identity),
    }
    payload["audit_contract_sha256"] = canonical_sha256(payload)
    return payload


def _audit_teacher_cache_paths(
    *,
    cache_root: str | Path,
    manifest_path: str | Path,
    prompt_authority_path: str | Path,
) -> tuple[GLM52TeacherCacheAudit, str, str, dict[str, str]]:
    root = Path(cache_root).expanduser().absolute()
    manifest = Path(manifest_path).expanduser().absolute()
    prompt_authority = Path(prompt_authority_path).expanduser().absolute()
    artifact_sha256 = teacher_cache_artifact_hash(root)
    _manifest, manifest_sha256 = _load_json_authority(
        manifest,
        label="teacher cache manifest",
    )
    audit = _audit_raw_teacher_cache(
        cache_root=root,
        manifest_path=manifest,
        prompt_pack_path=prompt_authority,
    )
    if audit.valid is not True:
        raise ValueError("teacher cache strict audit did not validate the cache")
    if audit.release_eligible is not True:
        raise ValueError("teacher cache is not release eligible")
    if teacher_cache_artifact_hash(root) != artifact_sha256:
        raise ValueError("teacher cache artifact changed across strict audit")
    authority = {
        "cache_root": str(root),
        "manifest_path": str(manifest),
        "prompt_authority_path": str(prompt_authority),
        "artifact_sha256": artifact_sha256,
    }
    return audit, manifest_sha256, artifact_sha256, authority


def _load_raw_authority_path(path: str | Path, *, label: str) -> dict[str, Any]:
    value, _digest = _load_json_authority(path, label=label)
    return value


def _candidate_eval_passes(
    *,
    teacher_cache_root: str | Path,
    candidate_cache_root: str | Path,
    policy_path: str | Path,
    prompt_pack_path: str | Path,
) -> bool:
    comparison = compare_glm52_candidate_caches(
        teacher_cache_root,
        candidate_cache_root,
        policy_path=policy_path,
        prompt_pack_path=prompt_pack_path,
    )
    if not isinstance(comparison, Mapping):
        raise ValueError("candidate comparison validator returned no evidence")
    return comparison.get("family_eval_gate_pass") is True


def _route_diagnostics_pass(
    *,
    source_root: str | Path,
    candidate_root: str | Path,
    policy_path: str | Path,
    source_authority_path: str | Path,
    candidate_authority_path: str | Path,
) -> bool:
    source_authority = validate_capture_authority(
        _load_raw_authority_path(source_authority_path, label="source route authority"),
        label="source route authority",
    )
    candidate_authority = validate_capture_authority(
        _load_raw_authority_path(candidate_authority_path, label="candidate route authority"),
        label="candidate route authority",
    )
    evidence = compare_route_trace_artifacts(
        source_root,
        candidate_root,
        policy_json=policy_path,
        require_frozen_batch=True,
        expected_source_authority=source_authority,
        expected_candidate_authority=candidate_authority,
    )
    if not isinstance(evidence, Mapping):
        raise ValueError("route diagnostics validator returned no evidence")
    evaluation = evidence.get("evaluation")
    checks = evidence.get("checks")
    if not isinstance(evaluation, Mapping) or not isinstance(checks, Mapping):
        raise ValueError("route diagnostics validator returned malformed evidence")
    # The frozen policy deliberately has no route/math thresholds.  A release
    # class, structurally valid diagnostic clears this blocker; no threshold is
    # invented and release_gate_pass remains diagnostic-only false upstream.
    return (
        evaluation.get("diagnostic_only") is True
        and evaluation.get("release_eligible") is True
        and checks.get("structural_math_pass") is True
    )


def _read_jsonl(path: str | Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = Path(path).read_text().splitlines()
    except OSError as error:
        raise ValueError(f"could not read {label}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{label}:{line_number} is not valid JSON"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(f"{label}:{line_number} must be a JSON object")
        rows.append(value)
    return rows


def _write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path).expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _non_bool_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _validate_identity(payload: Mapping[str, Any], *, revision_key: str) -> None:
    if payload.get("model_id") != PINNED_GLM52_MODEL_ID:
        raise ValueError("evidence model_id does not match the pinned GLM52 source")
    if payload.get(revision_key) != PINNED_GLM52_REVISION:
        raise ValueError("evidence revision does not match the pinned GLM52 source")


def _validate_teacher_metadata(
    payload: Mapping[str, Any],
    *,
    prompt_rows: Sequence[Mapping[str, Any]],
    policy_contract: str,
    prompt_contract: str,
) -> None:
    body = dict(payload)
    embedded_digest = body.pop("metadata_contract_sha256", None)
    if embedded_digest != canonical_sha256(body):
        raise ValueError("teacher metadata contract digest does not authenticate its body")
    _validate_identity(payload, revision_key="revision")
    expected = {
        "schema_version": 1,
        "record_type": "glm52_family_eval_teacher_metadata_probe",
        "metadata_status": "glm52_teacher_source_metadata_ready",
        "metadata_ready": True,
        "config_sha256": GLM52_REAP_CONFIG_SHA256,
        "index_sha256": GLM52_REAP_INDEX_SHA256,
        "source_weight_encoding": "modelopt_nvfp4",
        "source_decoder": "modelopt_nvfp4_v1",
        "teacher_source_kind": TEACHER_SOURCE_KIND,
        "family_policy_contract_sha256": policy_contract,
        "prompt_pack_contract_sha256": prompt_contract,
        "source_index_revalidated": True,
        "source_payload_headers_revalidated": True,
        "source_payload_group_count": EXPECTED_ROUTED_GROUP_COUNT,
        "source_payload_shard_count": 61,
        "header_only_source_validation": True,
        "tensor_payloads_read": False,
        "full_model_constructed": False,
        "candidate_artifact_used": False,
        "metadata_row_count": 66,
        "metadata_counts_by_split": {
            split: 22 for split in GLM52_REQUIRED_EVAL_SPLITS
        },
        "metadata_counts_by_domain": {
            "route": 24,
            "math": 21,
            "instruction": 21,
        },
        "teacher_cache_payload_present": False,
        "teacher_cache_ready": False,
        "teacher_logits_generated": False,
        "intended_logit_scope": "full_vocabulary",
        "intended_kld_mode": "exact_full_logits",
        "expected_vocab_size": GLM52_MODEL_VOCAB_SIZE,
        "activation_quantization_emulated": False,
        "exact_w4a4_runtime_parity_claimed": False,
        "bf16_teacher_claimed": False,
        "missing_requirements": ["dequantized_source_teacher_cache_payload"],
    }
    for field, expected_value in expected.items():
        if payload.get(field) != expected_value:
            if field == "source_payload_group_count":
                raise ValueError(
                    "teacher metadata source payload group count must be 225"
                )
            raise ValueError(
                f"teacher metadata {field} must be {expected_value!r}, "
                f"found {payload.get(field)!r}"
            )

    metadata_path = payload.get("metadata_jsonl")
    if not isinstance(metadata_path, str) or not metadata_path:
        raise ValueError("teacher metadata JSONL path is missing")
    recorded_sha = payload.get("metadata_jsonl_sha256")
    if not _is_sha256(recorded_sha) or sha256_file(metadata_path) != recorded_sha:
        raise ValueError("teacher metadata JSONL SHA-256 does not match its payload")
    rows = _read_jsonl(metadata_path, label="teacher metadata JSONL")
    if len(rows) != 66 or len(prompt_rows) != 66:
        raise ValueError("teacher metadata and prompt pack must each contain 66 rows")
    prompts_by_id = {str(row.get("prompt_id")): row for row in prompt_rows}
    if len(prompts_by_id) != 66:
        raise ValueError("prompt pack IDs are not unique")
    if {row.get("prompt_id") for row in rows} != set(prompts_by_id):
        raise ValueError("teacher metadata prompt IDs do not match the prompt pack")
    for row in rows:
        prompt_id = str(row["prompt_id"])
        prompt = prompts_by_id[prompt_id]
        expected_row = {
            "schema_version": 1,
            "record_type": "glm52_teacher_source_metadata",
            "prompt_id": prompt_id,
            "split": prompt.get("split"),
            "domain": prompt.get("domain"),
            "token_count": prompt.get("token_count"),
            "token_ids_sha256": prompt.get("token_ids_sha256"),
            "tuning_eligible": prompt.get("tuning_eligible"),
            "teacher_model_id": PINNED_GLM52_MODEL_ID,
            "teacher_revision": PINNED_GLM52_REVISION,
            "teacher_source_kind": TEACHER_SOURCE_KIND,
            "source_weight_encoding": "modelopt_nvfp4",
            "source_decoder": "modelopt_nvfp4_v1",
            "family_policy_contract_sha256": policy_contract,
            "prompt_pack_contract_sha256": prompt_contract,
            "teacher_cache_payload_present": False,
            "teacher_cache_ready": False,
            "full_logits_available": False,
            "intended_logit_scope": "full_vocabulary",
            "intended_kld_mode": "exact_full_logits",
            "expected_vocab_size": GLM52_MODEL_VOCAB_SIZE,
            "activation_quantization_emulated": False,
            "exact_w4a4_runtime_parity_claimed": False,
            "bf16_teacher_claimed": False,
            "holdout_tuning_forbidden": True,
        }
        for field, expected_value in expected_row.items():
            if row.get(field) != expected_value:
                raise ValueError(
                    f"teacher metadata row {prompt_id} has invalid {field}"
                )


def _validate_full_bind_preflight(payload: Mapping[str, Any]) -> bool:
    if payload.get("record_type") != "glm52_full_bind_preflight":
        raise ValueError("full-bind evidence has the wrong record type")
    _validate_identity(payload, revision_key="source_revision")
    if payload.get("profile") != GLM52_REAP_PROFILE:
        raise ValueError("full-bind evidence has the wrong profile")
    if payload.get("config_sha256") != GLM52_REAP_CONFIG_SHA256:
        raise ValueError("full-bind config SHA-256 is not pinned")
    if payload.get("source_index_sha256") != GLM52_REAP_INDEX_SHA256:
        raise ValueError("full-bind index SHA-256 is not pinned")
    if payload.get("profile_sha256") != GLM52_REAP_PROFILE_SHA256:
        raise ValueError("full-bind profile SHA-256 is not pinned")
    if (
        payload.get("profile_contract_sha256")
        != GLM52_REAP_PROFILE_CONTRACT_SHA256
    ):
        raise ValueError("full-bind profile contract is not pinned")
    expected = _non_bool_int(
        payload.get("expected_routed_group_count"),
        label="expected routed group count",
    )
    present = _non_bool_int(
        payload.get("present_routed_group_count"),
        label="present routed group count",
    )
    missing = _non_bool_int(
        payload.get("missing_routed_group_count"),
        label="missing routed group count",
    )
    if expected != EXPECTED_ROUTED_GROUP_COUNT or present < 0 or missing < 0:
        raise ValueError("full-bind routed group counts are invalid")
    if present + missing != expected:
        raise ValueError("full-bind present and missing group counts do not sum to 225")
    present_keys = payload.get("present_routed_group_keys")
    missing_keys = payload.get("missing_routed_group_keys")
    if not isinstance(present_keys, list) or len(present_keys) != present:
        raise ValueError("full-bind present group inventory does not match its count")
    if not isinstance(missing_keys, list) or len(missing_keys) != missing:
        raise ValueError("full-bind missing group inventory does not match its count")
    if payload.get("expected_routed_group_keys") != list(EXPECTED_ROUTED_GROUP_KEYS):
        raise ValueError("full-bind expected routed group inventory is not canonical")
    present_set = set(map(str, present_keys))
    missing_set = set(map(str, missing_keys))
    expected_set = set(EXPECTED_ROUTED_GROUP_KEYS)
    if present_set & missing_set or present_set | missing_set != expected_set:
        raise ValueError("full-bind routed group inventories are not exact and disjoint")
    if present_keys != [key for key in EXPECTED_ROUTED_GROUP_KEYS if key in present_set]:
        raise ValueError("full-bind present routed group inventory is out of order")
    if missing_keys != [key for key in EXPECTED_ROUTED_GROUP_KEYS if key in missing_set]:
        raise ValueError("full-bind missing routed group inventory is out of order")
    if payload.get("dense_routed_experts") is not False:
        raise ValueError("full-bind evidence must not contain dense routed experts")
    ready = present == expected and missing == 0
    claimed_pass = payload.get("preflight_pass") is True
    expected_status = (
        "glm52_full_bind_preflight_ready"
        if ready
        else "glm52_full_bind_preflight_blocked"
    )
    if claimed_pass != ready or payload.get("preflight_status") != expected_status:
        raise ValueError("full-bind pass/status claims do not match routed coverage")
    return ready


def _validate_indexshare_runtime(payload: Mapping[str, Any]) -> bool:
    if payload.get("record_type") != "glm52_indexshare_runtime_probe":
        raise ValueError("IndexShare runtime evidence has the wrong record type")
    _validate_identity(payload, revision_key="source_revision")
    if payload.get("config_sha256") != GLM52_REAP_CONFIG_SHA256:
        raise ValueError("IndexShare runtime config SHA-256 is not pinned")
    if payload.get("index_sha256") != GLM52_REAP_INDEX_SHA256:
        raise ValueError("IndexShare runtime index SHA-256 is not pinned")
    return all(
        (
            payload.get("probe_status") == "glm52_indexshare_runtime_probe_ready",
            payload.get("probe_pass") is True,
            payload.get("production_identity_contract") is True,
            payload.get("static_contract_pass") is True,
            payload.get("synthetic_runtime_contract") is True,
        )
    )


def _validate_synthetic_generation(payload: Mapping[str, Any]) -> tuple[bool, bool]:
    if payload.get("record_type") != "glm52_synthetic_generation_probe":
        raise ValueError("synthetic generation evidence has the wrong record type")
    contract_ready = all(
        (
            payload.get("probe_status") == "glm52_synthetic_generation_probe_ready",
            payload.get("probe_pass") is True,
            payload.get("synthetic_runtime_contract") is True,
            payload.get("whole_model_scope") == "tiny_fixture",
        )
    )
    return contract_ready, payload.get("dense_routed_weight_present") is False


def check_glm52_family_gate(
    *,
    family_policy: Mapping[str, Any],
    eval_prompt_pack: Mapping[str, Any],
    teacher_metadata: Mapping[str, Any],
    full_bind_preflight: Mapping[str, Any],
    indexshare_runtime: Mapping[str, Any],
    synthetic_generation: Mapping[str, Any],
    family_eval_gate: Mapping[str, Any] | None = None,
    family_benchmark_gate: Mapping[str, Any] | None = None,
    non_vq_evidence: Mapping[str, Any] | None = None,
    composite_artifact_audit: Mapping[str, Any] | None = None,
    production_generation: Mapping[str, Any] | None = None,
    teacher_cache_root: str | Path | None = None,
    teacher_cache_manifest_path: str | Path | None = None,
    teacher_cache_prompt_authority_path: str | Path | None = None,
    candidate_cache_root: str | Path | None = None,
    family_policy_path: str | Path | None = None,
    source_route_trace_root: str | Path | None = None,
    candidate_route_trace_root: str | Path | None = None,
    source_route_authority_path: str | Path | None = None,
    candidate_route_authority_path: str | Path | None = None,
    benchmark_evidence_path: str | Path | None = None,
) -> dict[str, Any]:
    if family_eval_gate is not None or family_benchmark_gate is not None:
        raise ValueError(
            "family eval and benchmark summaries are not yet accepted: raw "
            "release evidence validators are required"
        )
    raw_trio = (
        non_vq_evidence,
        composite_artifact_audit,
        production_generation,
    )
    raw_count = sum(evidence is not None for evidence in raw_trio)
    if raw_count not in (0, len(raw_trio)):
        raise ValueError(
            "schema-v2 raw evidence must be supplied as a complete trio: "
            "non-VQ package, composite artifact audit, and production generation"
        )
    schema_v2 = raw_count == len(raw_trio)
    teacher_cache_paths = (
        teacher_cache_root,
        teacher_cache_manifest_path,
        teacher_cache_prompt_authority_path,
    )
    teacher_cache_path_count = sum(path is not None for path in teacher_cache_paths)
    if teacher_cache_path_count not in (0, len(teacher_cache_paths)):
        raise ValueError(
            "teacher cache root, manifest, and prompt authority paths must be "
            "supplied together"
        )
    if teacher_cache_path_count and not schema_v2:
        raise ValueError(
            "teacher cache evidence requires the complete schema-v2 raw trio"
        )
    candidate_paths = (candidate_cache_root, family_policy_path)
    candidate_path_count = sum(path is not None for path in candidate_paths)
    if candidate_path_count not in (0, len(candidate_paths)):
        raise ValueError(
            "candidate cache root and frozen family policy path must be supplied together"
        )
    if candidate_path_count and not teacher_cache_path_count:
        raise ValueError("candidate evaluation requires the raw teacher cache")
    route_paths = (
        source_route_trace_root,
        candidate_route_trace_root,
        source_route_authority_path,
        candidate_route_authority_path,
    )
    route_path_count = sum(path is not None for path in route_paths)
    if route_path_count not in (0, len(route_paths)):
        raise ValueError("all raw route trace roots and authorities are required")
    if route_path_count and not candidate_path_count:
        raise ValueError("route diagnostics require a passing raw candidate evaluation")
    if benchmark_evidence_path is not None and not route_path_count:
        raise ValueError("benchmark evidence requires authenticated route diagnostics")
    teacher_cache_audit: GLM52TeacherCacheAudit | None = None
    teacher_cache_manifest_sha256: str | None = None
    teacher_cache_artifact_sha256: str | None = None
    teacher_cache_authority: dict[str, str] | None = None
    if teacher_cache_path_count:
        assert teacher_cache_root is not None
        assert teacher_cache_manifest_path is not None
        assert teacher_cache_prompt_authority_path is not None
        (
            teacher_cache_audit,
            teacher_cache_manifest_sha256,
            teacher_cache_artifact_sha256,
            teacher_cache_authority,
        ) = _audit_teacher_cache_paths(
            cache_root=teacher_cache_root,
            manifest_path=teacher_cache_manifest_path,
            prompt_authority_path=teacher_cache_prompt_authority_path,
        )
    validated_policy = validate_glm52_family_gate_policy(family_policy)
    prompt_rows = validate_glm52_family_prompt_pack(eval_prompt_pack)
    policy_contract = str(family_policy["policy_contract_sha256"])
    prompt_contract = str(eval_prompt_pack["prompt_pack_contract_sha256"])
    _validate_teacher_metadata(
        teacher_metadata,
        prompt_rows=prompt_rows,
        policy_contract=policy_contract,
        prompt_contract=prompt_contract,
    )
    full_coverage_ready = _validate_full_bind_preflight(full_bind_preflight)
    indexshare_ready = _validate_indexshare_runtime(indexshare_runtime)
    synthetic_ready, synthetic_dense_free = _validate_synthetic_generation(
        synthetic_generation
    )

    common_artifact_identity: dict[str, object] | None = None
    if schema_v2:
        if not full_coverage_ready:
            raise ValueError("schema-v2 evidence requires a full 225-group bind")
        artifact_target = validated_policy.get("artifact_target")
        if not isinstance(artifact_target, Mapping):
            raise ValueError("family policy artifact target must be an object")
        if (
            type(artifact_target.get("tensor_payload_bytes")) is not int
            or artifact_target.get("tensor_payload_bytes") != 98_433_923_808
        ):
            raise ValueError("family policy accepted tensor payload bytes drifted")
        if (
            type(artifact_target.get("whole_main_bpw")) is not float
            or artifact_target.get("whole_main_bpw") != 1.5934433
        ):
            raise ValueError("family policy accepted whole-model bpw drifted")
        assert non_vq_evidence is not None
        assert composite_artifact_audit is not None
        assert production_generation is not None
        validated_non_vq = validate_glm52_non_vq_evidence(non_vq_evidence)
        validated_audit = validate_glm52_full_artifact_audit(
            composite_artifact_audit,
            non_vq_evidence=validated_non_vq,
        )
        common_artifact_identity = build_glm52_production_composite_identity(
            non_vq_evidence=validated_non_vq,
            composite_artifact_audit=validated_audit,
        )
        validate_glm52_production_generation(
            production_generation,
            common_artifact_identity=common_artifact_identity,
            composite_artifact_audit=validated_audit,
        )

    eval_pass = False
    if candidate_path_count:
        assert candidate_cache_root is not None
        assert family_policy_path is not None
        assert teacher_cache_root is not None
        assert teacher_cache_prompt_authority_path is not None
        eval_pass = _candidate_eval_passes(
            teacher_cache_root=teacher_cache_root,
            candidate_cache_root=candidate_cache_root,
            policy_path=family_policy_path,
            prompt_pack_path=teacher_cache_prompt_authority_path,
        )

    route_math_ready = False
    if route_path_count:
        if not eval_pass:
            raise ValueError("route diagnostics require a passing candidate evaluation")
        assert source_route_trace_root is not None
        assert candidate_route_trace_root is not None
        assert source_route_authority_path is not None
        assert candidate_route_authority_path is not None
        assert family_policy_path is not None
        route_math_ready = _route_diagnostics_pass(
            source_root=source_route_trace_root,
            candidate_root=candidate_route_trace_root,
            policy_path=family_policy_path,
            source_authority_path=source_route_authority_path,
            candidate_authority_path=candidate_route_authority_path,
        )

    benchmark_pass = False
    benchmark_evidence: Mapping[str, Any] | None = None
    if benchmark_evidence_path is not None:
        if not route_math_ready:
            raise ValueError("benchmark evidence requires valid route diagnostics")
        benchmark_evidence = load_glm52_same_machine_benchmark_evidence(
            benchmark_evidence_path,
            policy=validated_policy,
        )
        benchmark_pass = True

    # A header-only bind preflight is coverage evidence, not the strict artifact
    # audit, production generation, raw full-vocabulary eval, or benchmark proof
    # required to turn the release gate green. Those inputs remain deliberately
    # locked until their raw payload validators and common artifact identity exist.
    teacher_cache_ready = teacher_cache_audit is not None
    production_pass = schema_v2
    production_dense_free = True

    full_bind_dense_free = full_bind_preflight.get("dense_routed_experts") is False
    checks = {
        "policy_frozen": True,
        "prompt_pack_frozen": True,
        "teacher_source_metadata_ready": True,
        "teacher_cache_full_vocabulary_ready": teacher_cache_ready,
        "indexshare_runtime_contract_ready": indexshare_ready,
        "synthetic_generation_contract_ready": synthetic_ready,
        "full_routed_coverage_ready": full_coverage_ready,
        "full_225_group_artifact_ready": schema_v2,
        "accepted_artifact_bytes_and_bpw_ready": schema_v2,
        "production_binding_and_generation_ready": production_pass,
        "full_vocabulary_source_relative_family_eval_ready": eval_pass,
        "route_math_diagnostics_ready": route_math_ready,
        "same_machine_pinned_fp4_benchmark_ready": benchmark_pass,
        "no_dense_routed_experts": (
            full_bind_dense_free and synthetic_dense_free and production_dense_free
        ),
    }
    missing_requirements: list[str] = []
    if not teacher_cache_ready:
        missing_requirements.append("dequantized_source_teacher_cache_payload")
    if not checks["full_225_group_artifact_ready"]:
        missing_requirements.append("full_225_group_artifact")
    if not production_pass:
        missing_requirements.append("production_model_bind_and_generation")
    if not eval_pass:
        missing_requirements.append("full_vocabulary_source_relative_family_eval")
    if not route_math_ready:
        missing_requirements.append("route_math_diagnostics")
    if not benchmark_pass:
        missing_requirements.append("same_machine_pinned_fp4_benchmark")
    if not checks["no_dense_routed_experts"]:
        missing_requirements.append("candidate_without_dense_routed_experts")

    family_gate_pass = all(checks.values())
    payload: dict[str, Any] = {
        "schema_version": 1,
        "record_type": GATE_RECORD_TYPE,
        "gate_schema_version": (
            GLM52_FAMILY_GATE_V6_SCHEMA_VERSION
            if benchmark_pass
            else (
                GLM52_FAMILY_GATE_V5_SCHEMA_VERSION
                if route_math_ready
                else (
                    GLM52_FAMILY_GATE_V4_SCHEMA_VERSION
                    if eval_pass
                    else (
                        GLM52_FAMILY_GATE_V3_SCHEMA_VERSION
                        if teacher_cache_ready
                        else (
                            GLM52_FAMILY_GATE_V2_SCHEMA_VERSION
                            if schema_v2
                            else GLM52_FAMILY_GATE_V1_SCHEMA_VERSION
                        )
                    )
                )
            )
        ),
        "release_pass_enabled": benchmark_pass,
        "raw_release_evidence_validators_ready": benchmark_pass,
        "gate_status": GATE_PASSED_STATUS if family_gate_pass else GATE_BLOCKED_STATUS,
        "family_gate_pass": family_gate_pass,
        "model_id": PINNED_GLM52_MODEL_ID,
        "revision": PINNED_GLM52_REVISION,
        "profile": GLM52_REAP_PROFILE,
        "family_policy_contract_sha256": policy_contract,
        "prompt_pack_contract_sha256": prompt_contract,
        "accepted_tensor_payload_bytes": validated_policy["artifact_target"][
            "tensor_payload_bytes"
        ],
        "accepted_whole_main_bpw": validated_policy["artifact_target"][
            "whole_main_bpw"
        ],
        "expected_routed_group_count": full_bind_preflight[
            "expected_routed_group_count"
        ],
        "present_routed_group_count": full_bind_preflight[
            "present_routed_group_count"
        ],
        "missing_routed_group_count": full_bind_preflight[
            "missing_routed_group_count"
        ],
        "production_binding_proven": production_pass,
        "production_generation_proven": production_pass,
        "teacher_cache_payload_present": teacher_cache_ready,
        "full_vocabulary_eval_proven": eval_pass,
        "same_machine_benchmark_proven": benchmark_pass,
        "input_evidence_contract_sha256": {
            "family_policy": canonical_sha256(dict(family_policy)),
            "eval_prompt_pack": canonical_sha256(dict(eval_prompt_pack)),
            "teacher_metadata": canonical_sha256(dict(teacher_metadata)),
            "full_bind_preflight": canonical_sha256(dict(full_bind_preflight)),
            "indexshare_runtime": canonical_sha256(dict(indexshare_runtime)),
            "synthetic_generation": canonical_sha256(dict(synthetic_generation)),
        },
        "checks": checks,
        "missing_requirements": missing_requirements,
    }
    if schema_v2:
        assert non_vq_evidence is not None
        assert composite_artifact_audit is not None
        assert production_generation is not None
        assert common_artifact_identity is not None
        payload["input_evidence_contract_sha256"].update(
            {
                "non_vq_evidence": canonical_sha256(dict(non_vq_evidence)),
                "composite_artifact_audit": canonical_sha256(
                    dict(composite_artifact_audit)
                ),
                "production_generation": canonical_sha256(
                    dict(production_generation)
                ),
            }
        )
        payload["common_artifact_identity"] = common_artifact_identity
        payload["common_artifact_identity_sha256"] = canonical_sha256(
            common_artifact_identity
        )
        payload["raw_evidence_validator_readiness"] = {
            "artifact": True,
            "production": True,
            "family_eval": False,
            "family_benchmark": False,
        }
    if teacher_cache_audit is not None:
        assert teacher_cache_manifest_sha256 is not None
        assert teacher_cache_artifact_sha256 is not None
        assert teacher_cache_authority is not None
        teacher_cache_identity = _teacher_cache_audit_identity(
            teacher_cache_audit,
            manifest_file_sha256=teacher_cache_manifest_sha256,
            artifact_sha256=teacher_cache_artifact_sha256,
        )
        teacher_cache_identity_sha256 = canonical_sha256(teacher_cache_identity)
        payload["audited_teacher_cache_identity"] = teacher_cache_identity
        payload["audited_teacher_cache_identity_sha256"] = (
            teacher_cache_identity_sha256
        )
        payload["raw_teacher_cache_authority"] = teacher_cache_authority
        payload["input_evidence_contract_sha256"]["teacher_cache_audit"] = (
            teacher_cache_identity_sha256
        )
        payload["raw_evidence_validator_readiness"] = {
            "artifact": True,
            "production": True,
            "teacher_cache": True,
            "family_eval": False,
            "family_benchmark": False,
        }
    if candidate_path_count:
        assert candidate_cache_root is not None
        assert family_policy_path is not None
        assert teacher_cache_root is not None
        assert teacher_cache_prompt_authority_path is not None
        payload["raw_candidate_evaluation_authority"] = {
            "teacher_cache_root": str(Path(teacher_cache_root).expanduser().absolute()),
            "candidate_cache_root": str(Path(candidate_cache_root).expanduser().absolute()),
            "prompt_pack_path": str(Path(teacher_cache_prompt_authority_path).expanduser().absolute()),
            "family_policy_path": str(Path(family_policy_path).expanduser().absolute()),
        }
        payload["raw_evidence_validator_readiness"]["family_eval"] = True
    if route_path_count:
        assert source_route_trace_root is not None
        assert candidate_route_trace_root is not None
        assert source_route_authority_path is not None
        assert candidate_route_authority_path is not None
        payload["raw_route_diagnostics_authority"] = {
            "source_root": str(Path(source_route_trace_root).expanduser().absolute()),
            "candidate_root": str(Path(candidate_route_trace_root).expanduser().absolute()),
            "source_authority_path": str(Path(source_route_authority_path).expanduser().absolute()),
            "candidate_authority_path": str(Path(candidate_route_authority_path).expanduser().absolute()),
        }
        payload["raw_evidence_validator_readiness"]["route_math_diagnostics"] = True
    if benchmark_evidence_path is not None:
        assert benchmark_evidence is not None
        payload["raw_benchmark_evidence_authority"] = str(
            Path(benchmark_evidence_path).expanduser().absolute()
        )
        # This is intentionally not a signature claim: GLM52 release evidence
        # is local-filesystem hash-referenced and recomputed from embedded RAW
        # repetitions by the release-time loader.
        payload["raw_benchmark_evidence_provenance"] = benchmark_evidence[
            "evidence_provenance"
        ]
        payload["raw_evidence_validator_readiness"]["family_benchmark"] = True
    if tuple(checks) != GLM52_FAMILY_GATE_V1_CHECK_NAMES:
        raise RuntimeError("GLM52 family gate check inventory drifted")
    expected_missing = (
        GLM52_FAMILY_GATE_V6_MISSING_REQUIREMENTS
        if benchmark_pass
        else (
            GLM52_FAMILY_GATE_V5_MISSING_REQUIREMENTS
            if route_math_ready
            else (
                GLM52_FAMILY_GATE_V4_MISSING_REQUIREMENTS
                if eval_pass
                else (
                    GLM52_FAMILY_GATE_V3_MISSING_REQUIREMENTS
                    if teacher_cache_ready
                    else (
                        GLM52_FAMILY_GATE_V2_MISSING_REQUIREMENTS
                        if schema_v2
                        else GLM52_FAMILY_GATE_V1_MISSING_REQUIREMENTS
                    )
                )
            )
        )
    )
    if tuple(missing_requirements) != expected_missing:
        raise RuntimeError("GLM52 family gate blocker inventory drifted")
    payload["gate_contract_sha256"] = canonical_sha256(payload)
    return payload


def _parser(*, audit_only: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compose authenticated GLM52 release evidence and exit 2 while any "
            "hard family-gate requirement remains incomplete."
        )
    )
    parser.add_argument("--audit-teacher-cache-only", action="store_true")
    parser.add_argument("--family-policy-json", required=not audit_only)
    parser.add_argument("--eval-prompt-pack-json", required=True)
    parser.add_argument("--teacher-metadata-json", required=not audit_only)
    parser.add_argument("--full-bind-preflight-json", required=not audit_only)
    parser.add_argument("--indexshare-runtime-json", required=not audit_only)
    parser.add_argument("--synthetic-generation-json", required=not audit_only)
    parser.add_argument("--family-eval-gate-json")
    parser.add_argument("--family-benchmark-gate-json")
    parser.add_argument("--non-vq-evidence-json")
    parser.add_argument("--composite-artifact-audit-json")
    parser.add_argument("--production-generation-json")
    parser.add_argument("--teacher-cache-root", required=audit_only)
    parser.add_argument("--teacher-cache-manifest-json", required=audit_only)
    parser.add_argument("--candidate-cache-root")
    parser.add_argument("--source-route-trace-root")
    parser.add_argument("--candidate-route-trace-root")
    parser.add_argument("--source-route-authority-json")
    parser.add_argument("--candidate-route-authority-json")
    parser.add_argument("--benchmark-evidence-json")
    parser.add_argument("--output-json", required=True)
    return parser


def _validate_output_path(output: str | Path, inputs: Sequence[str | Path]) -> None:
    output_path = Path(output).expanduser().absolute()
    output_identities = {output_path, output_path.resolve(strict=False)}
    input_identities = {Path(path).expanduser().absolute() for path in inputs}
    input_identities |= {
        path.resolve(strict=False) for path in tuple(input_identities)
    }
    if output_identities & input_identities:
        raise ValueError("family gate output must not alias an input file")
    for raw in inputs:
        root = Path(raw).expanduser().absolute()
        if not root.is_dir():
            continue
        for output_identity in output_identities:
            try:
                output_identity.relative_to(root.resolve(strict=False))
            except ValueError:
                continue
            raise ValueError("family gate output must not enter an input directory")


def _referenced_paths(value: object) -> set[Path]:
    paths: set[Path] = set()
    if isinstance(value, Mapping):
        for nested in value.values():
            paths.update(_referenced_paths(nested))
    elif isinstance(value, list | tuple):
        for nested in value:
            paths.update(_referenced_paths(nested))
    elif isinstance(value, str):
        candidate = Path(value).expanduser()
        if candidate.exists():
            paths.add(candidate)
    return paths


def _validate_production_input_file_hashes(
    production: Mapping[str, Any],
    *,
    captured_file_sha256: Mapping[str, str],
) -> None:
    recorded = production.get("input_evidence_file_sha256")
    if not isinstance(recorded, Mapping):
        raise ValueError(
            "production generation input evidence file hashes must be an object"
        )
    expected = {
        "family_policy_json": captured_file_sha256["family_policy"],
        "full_bind_preflight_json": captured_file_sha256[
            "full_bind_preflight"
        ],
        "non_vq_evidence_json": captured_file_sha256["non_vq_evidence"],
        "composite_audit_json": captured_file_sha256[
            "composite_artifact_audit"
        ],
    }
    for field, digest in expected.items():
        submitted = recorded.get(field)
        if not _is_sha256(submitted) or submitted != digest:
            raise ValueError(
                f"production generation {field} SHA-256 does not match its input file"
            )


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    audit_only = "--audit-teacher-cache-only" in raw_argv
    args = _parser(audit_only=audit_only).parse_args(raw_argv)
    if audit_only:
        audit_paths = (
            args.teacher_cache_root,
            args.teacher_cache_manifest_json,
            args.eval_prompt_pack_json,
        )
        try:
            _validate_output_path(args.output_json, audit_paths)
            (
                audit,
                manifest_sha256,
                artifact_sha256,
                _authority,
            ) = _audit_teacher_cache_paths(
                cache_root=args.teacher_cache_root,
                manifest_path=args.teacher_cache_manifest_json,
                prompt_authority_path=args.eval_prompt_pack_json,
            )
            payload = _teacher_cache_audit_evidence(
                audit,
                manifest_file_sha256=manifest_sha256,
                artifact_sha256=artifact_sha256,
            )
        except Exception as error:
            payload = {
                "schema_version": 1,
                "record_type": "glm52_teacher_cache_audit",
                "audit_status": "glm52_teacher_cache_audit_invalid",
                "audit_pass": False,
                "input_error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
            _write_json_atomic(args.output_json, payload)
            return 1
        _write_json_atomic(args.output_json, payload)
        return 0

    required_paths = {
        "family_policy": args.family_policy_json,
        "eval_prompt_pack": args.eval_prompt_pack_json,
        "teacher_metadata": args.teacher_metadata_json,
        "full_bind_preflight": args.full_bind_preflight_json,
        "indexshare_runtime": args.indexshare_runtime_json,
        "synthetic_generation": args.synthetic_generation_json,
    }
    optional_paths = {
        "family_eval_gate": args.family_eval_gate_json,
        "family_benchmark_gate": args.family_benchmark_gate_json,
        "non_vq_evidence": args.non_vq_evidence_json,
        "composite_artifact_audit": args.composite_artifact_audit_json,
        "production_generation": args.production_generation_json,
    }
    cache_paths = [
        path
        for path in (
            args.teacher_cache_root,
            args.teacher_cache_manifest_json,
        )
        if path
    ]
    release_paths = [
        path
        for path in (
            args.candidate_cache_root,
            args.source_route_trace_root,
            args.candidate_route_trace_root,
            args.source_route_authority_json,
            args.candidate_route_authority_json,
            args.benchmark_evidence_json,
        )
        if path
    ]
    all_paths = [
        *required_paths.values(),
        *[p for p in optional_paths.values() if p],
        *cache_paths,
        *release_paths,
    ]
    try:
        _validate_output_path(args.output_json, all_paths)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    try:
        if bool(args.teacher_cache_root) != bool(args.teacher_cache_manifest_json):
            raise ValueError(
                "teacher cache root and manifest JSON must be supplied together"
            )
        authority_paths = {
            **required_paths,
            **{
                name: path
                for name, path in optional_paths.items()
                if path is not None
            },
        }
        captured = {
            name: _load_json_authority(path, label=name)
            for name, path in authority_paths.items()
        }
        inputs = {name: value for name, (value, _digest) in captured.items()}
        captured_file_sha256 = {
            name: digest for name, (_value, digest) in captured.items()
        }
        if args.teacher_cache_root:
            cache_manifest, manifest_sha256 = _load_json_authority(
                args.teacher_cache_manifest_json,
                label="teacher cache manifest",
            )
            non_vq_input = inputs.get("non_vq_evidence")
            if not isinstance(non_vq_input, Mapping):
                raise ValueError(
                    "teacher cache requires the complete schema-v2 raw trio"
                )
            _validate_teacher_cache_non_vq_binding(
                cache_manifest,
                non_vq_evidence=non_vq_input,
                non_vq_evidence_path=args.non_vq_evidence_json,
                non_vq_evidence_file_sha256=captured_file_sha256[
                    "non_vq_evidence"
                ],
            )
            inputs["teacher_cache_root"] = args.teacher_cache_root
            inputs["teacher_cache_manifest_path"] = (
                args.teacher_cache_manifest_json
            )
            inputs["teacher_cache_prompt_authority_path"] = (
                args.eval_prompt_pack_json
            )
            captured_file_sha256["teacher_cache_manifest"] = manifest_sha256
        if args.candidate_cache_root:
            inputs["candidate_cache_root"] = args.candidate_cache_root
            inputs["family_policy_path"] = args.family_policy_json
        route_cli_paths = (
            args.source_route_trace_root,
            args.candidate_route_trace_root,
            args.source_route_authority_json,
            args.candidate_route_authority_json,
        )
        if any(route_cli_paths) and not all(route_cli_paths):
            raise ValueError("all raw route trace roots and authorities are required")
        if all(route_cli_paths):
            inputs.update(
                {
                    "source_route_trace_root": args.source_route_trace_root,
                    "candidate_route_trace_root": args.candidate_route_trace_root,
                    "source_route_authority_path": args.source_route_authority_json,
                    "candidate_route_authority_path": args.candidate_route_authority_json,
                }
            )
        if args.benchmark_evidence_json:
            inputs["benchmark_evidence_path"] = args.benchmark_evidence_json
    except Exception as error:
        payload = {
            "schema_version": 1,
            "record_type": GATE_RECORD_TYPE,
            "gate_status": "glm52_family_gate_invalid",
            "family_gate_pass": False,
            "input_error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
        _write_json_atomic(args.output_json, payload)
        return 1
    transitive_paths = sorted(
        {path for value in inputs.values() for path in _referenced_paths(value)},
        key=str,
    )
    try:
        _validate_output_path(args.output_json, [*all_paths, *transitive_paths])
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    try:
        if all(
            (
                args.non_vq_evidence_json,
                args.composite_artifact_audit_json,
                args.production_generation_json,
            )
        ):
            production_input = inputs.get("production_generation")
            if not isinstance(production_input, Mapping):
                raise ValueError("production generation evidence must be an object")
            _validate_production_input_file_hashes(
                production_input,
                captured_file_sha256=captured_file_sha256,
            )
        payload = check_glm52_family_gate(**inputs)
    except Exception as error:
        payload = {
            "schema_version": 1,
            "record_type": GATE_RECORD_TYPE,
            "gate_status": "glm52_family_gate_invalid",
            "family_gate_pass": False,
            "input_error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
        _write_json_atomic(args.output_json, payload)
        return 1
    teacher_cache_identity = payload.get("audited_teacher_cache_identity")
    if isinstance(teacher_cache_identity, Mapping):
        artifact_sha256 = teacher_cache_identity.get("artifact_sha256")
        if not _is_sha256(artifact_sha256):
            raise RuntimeError("audited teacher cache artifact SHA-256 is invalid")
        captured_file_sha256["teacher_cache_artifact"] = artifact_sha256
    payload["input_evidence_file_sha256"] = captured_file_sha256
    payload.pop("gate_contract_sha256", None)
    payload["gate_contract_sha256"] = canonical_sha256(payload)
    _write_json_atomic(args.output_json, payload)
    return 0 if payload["family_gate_pass"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
