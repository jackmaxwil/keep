"""Gate evaluation over recorded step evidence.

All gate math reuses the extracted RC helpers in
``mlx_vq.quality.rc_gates`` — a gate here must agree with the RC pipeline's
summary on the same evidence. Gates are pure functions over evidence files,
so ``--regate`` can re-evaluate a recorded build after a threshold change
without re-running any model work.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlx_vq.build.gate_profiles import resolve_gate_thresholds
from mlx_vq.build.hashing import (
    gate_key as compute_gate_key,
    teacher_cache_artifact_hash,
)
from mlx_vq.quality.rc_gates import (
    _check_ge,
    _check_le,
    _summarize_benchmark,
    _summarize_eval_split,
)
from mlx_vq.quality.glm52_family import (
    GLM52_ACCEPTED_POLICY_BPW,
    GLM52_ACCEPTED_PRODUCTION_COMPOSITE_IDENTITY_SHA256,
    GLM52_ACCEPTED_TENSOR_PAYLOAD_BYTES,
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
    GLM52_REAP_PROFILE,
    GLM52_TEACHER_CACHE_SOURCE_EVIDENCE,
    PINNED_GLM52_MODEL_ID,
    PINNED_GLM52_REVISION,
    canonical_sha256,
    load_glm52_same_machine_benchmark_evidence,
    validate_glm52_family_gate_policy,
)
from mlx_vq.quality.glm52_teacher_cache import (
    GLM52TeacherCacheAudit,
    GLM52TeacherCacheContract,
    MANIFEST_FILENAME as GLM52_TEACHER_CACHE_MANIFEST_FILENAME,
    audit_glm52_teacher_cache,
)


_GLM52_V2_FAMILY_POLICY_CONTRACT_SHA256 = (
    "57ae812222de79555774296129e7110750f492794b1fe21bac91ed5cc986f206"
)
_GLM52_V2_PROMPT_PACK_CONTRACT_SHA256 = (
    "04c1ceacba775058de7d08462fbf69b534da7fbf766f57c8ce54daaebd84d2c1"
)
_GLM52_V2_INPUT_FILE_SHA256 = {
    "composite_artifact_audit": (
        "8026322a533606ed451d13b029d83836fefbf1a938b33529de535f3fc778591a"
    ),
    "eval_prompt_pack": (
        "697677a4949f4ee7e370ac5e9a55e9631b0a1385f95a07dfd3a3f2b87d8edf31"
    ),
    "family_policy": (
        "0975f7dc1117c5fba7532e9166f4767546fd691a6cb52520a40f09874f972ce2"
    ),
    "full_bind_preflight": (
        "9078a36c683a2487ef5d99f4c8fecc72f086689b8aa1e249ffaa34639b5946a3"
    ),
    "indexshare_runtime": (
        "94785cf0884ce5ac955a09b535c0b2ef00ad3cb742da7769bf504a4c9143ad25"
    ),
    "non_vq_evidence": (
        "ccbedd87f72f032bd86fd9d5a89fb75641595f30cf2a80fd7c84636d594e80d2"
    ),
    "production_generation": (
        "758b5bbebbb365cdf69998a034cf3791a6210d2723d5bec6c022336eceadcc82"
    ),
    "synthetic_generation": (
        "baa9c8214553c6af391c6a671a5b0185b8d7eb1746c3e6c52586f512536662a9"
    ),
    "teacher_metadata": (
        "621a013eb37f617409ec568340976e769610b3b6c8cae346639762d816a7fd7a"
    ),
}
_GLM52_V2_INPUT_CONTRACT_SHA256 = {
    "composite_artifact_audit": (
        "7362ca6d6f169454a8247d92be1e3032464a00231116d7505a6a94a770662607"
    ),
    "eval_prompt_pack": (
        "b6501b548c8242d703db0b173d54142398afd8e60724a297184c99b9382f45b5"
    ),
    "family_policy": (
        "a9260990f5d1d07c468bd3613c67938ed81ed363331d91ec378dd686ff9e03b9"
    ),
    "full_bind_preflight": (
        "47db7ba5027cb6852c22235cf3d8b8b159ba01860e60c7b1b8aee806ecc22f93"
    ),
    "indexshare_runtime": (
        "406213ae99698f93ac9cb584633b3f19cf01d5e726a6289b0009eacc4ee6d22f"
    ),
    "non_vq_evidence": (
        "30884ef0bd3a8cadf20c631d9fdc2d522cbc340dd2c6f4686d042fcb5a2e1964"
    ),
    "production_generation": (
        "830fc46ff4c83922efd033c3a5b48dabd02d4a3858274d4cef4a4be2fe19e853"
    ),
    "synthetic_generation": (
        "c567a2b20b3b39654038766f47b292e63e0d88dfcaf7f7c39e04f9a098916a16"
    ),
    "teacher_metadata": (
        "283792cd16a73ca7c8c98c7e3ddcaed6c0beb2c668af2b805ca0c3dee525e0e2"
    ),
}


@dataclass
class GateResult:
    profile: str
    thresholds: dict[str, Any]
    gate_key: str
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "thresholds": self.thresholds,
            "gate_key": self.gate_key,
            "passed": self.passed,
            "checks": self.checks,
            "reasons": self.reasons,
        }


def _finish(
    profile: str,
    thresholds: dict[str, Any],
    checks: dict[str, bool],
    *,
    summary: dict[str, Any] | None = None,
) -> GateResult:
    reasons = [name for name, passed in checks.items() if not passed]
    return GateResult(
        profile=profile,
        thresholds=thresholds,
        gate_key=compute_gate_key(profile=profile, thresholds=thresholds),
        passed=not reasons,
        checks=checks,
        reasons=reasons,
        summary=summary or {},
    )


def _load_json_line(path: Path) -> dict[str, Any]:
    text = path.read_text().strip()
    if not text:
        return {}
    first_line = text.splitlines()[0]
    value = json.loads(first_line)
    return value if isinstance(value, dict) else {}


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    return value if isinstance(value, dict) else {}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(
        float(value)
    )


def evaluate_train_sane(artifact_dir: Path, thresholds: dict[str, Any]) -> GateResult:
    manifest_path = artifact_dir / "conversion-manifest.json"
    checks: dict[str, bool] = {"manifest_present": manifest_path.exists()}
    sidecars: list[Any] = []
    run: Mapping[str, Any] = {}
    non_expert_precision: Mapping[str, Any] = {}
    router_corrections: Mapping[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        continuous = manifest.get("continuous_parameters") or {}
        if isinstance(continuous, Mapping):
            raw_sidecars = continuous.get("sidecars")
            sidecars = raw_sidecars if isinstance(raw_sidecars, list) else []
            raw_run = continuous.get("run")
            run = raw_run if isinstance(raw_run, Mapping) else {}
        raw_non_expert_precision = manifest.get("non_expert_precision")
        if isinstance(raw_non_expert_precision, Mapping):
            non_expert_precision = raw_non_expert_precision
        raw_router_corrections = manifest.get("router_corrections")
        if isinstance(raw_router_corrections, Mapping):
            router_corrections = raw_router_corrections
    checks["sidecars_present"] = (
        bool(sidecars)
        or non_expert_precision.get("enabled") is True
        or router_corrections.get("enabled") is True
    )
    final_loss = run.get("final_train_loss")
    checks["final_loss_finite"] = final_loss is None or _finite(final_loss)
    return _finish("train_sane", thresholds, checks)


def evaluate_audit_ok(evidence_path: Path, thresholds: dict[str, Any]) -> GateResult:
    audit = _load_json_line(evidence_path)
    checks = {
        "audit_present": bool(audit),
        "projections_found": int(audit.get("projection_count") or 0) > 0,
        "layers_found": int(audit.get("layer_count") or 0) > 0,
        "all_layers_nax_fast_compatible": bool(
            audit.get("auto_prefill_all_layers_nax_fast_compatible")
        ),
    }
    return _finish("audit_ok", thresholds, checks, summary=audit)


def _split_checks(
    summary: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    allow_dirty_rows = bool(thresholds.get("allow_dirty_rows"))
    if "clean_rows" in thresholds:
        checks["clean_rows"] = (
            int(summary.get("clean_record_count") or 0) >= int(thresholds["clean_rows"])
            and (allow_dirty_rows or bool(summary.get("all_memory_clean")))
        )
    if "mean_ppl_ratio_max" in thresholds:
        checks["mean_ppl_ratio"] = _check_le(
            summary.get("mean_ppl_ratio"), thresholds["mean_ppl_ratio_max"]
        )
    if "mean_kld_max" in thresholds:
        checks["mean_kld"] = _check_le(summary.get("mean_kld"), thresholds["mean_kld_max"])
    if "p999_kld_max" in thresholds:
        checks["p999_kld"] = _check_le(summary.get("p999_kld"), thresholds["p999_kld_max"])
    if "top1_min" in thresholds:
        checks["top1"] = _check_ge(
            summary.get("mean_top1_agreement"), thresholds["top1_min"]
        )
    if "domain_top1_min" in thresholds:
        domains = summary.get("domains")
        domain_checks = [
            _check_ge(domain.get("mean_top1_agreement"), thresholds["domain_top1_min"])
            for domain in (domains or {}).values()
            if isinstance(domain, Mapping)
        ]
        checks["domain_top1"] = bool(domain_checks) and all(domain_checks)
    return checks


def evaluate_eval_split(
    profile: str, evidence_path: Path, thresholds: dict[str, Any]
) -> GateResult:
    summary = _summarize_eval_split(evidence_path)
    return _finish(profile, thresholds, _split_checks(summary, thresholds), summary=summary)


def evaluate_bench_clean(evidence_path: Path, thresholds: dict[str, Any]) -> GateResult:
    bench = _summarize_benchmark(evidence_path)
    allow_dirty_rows = bool(thresholds.get("allow_dirty_rows"))
    timing = bench.get("attempted_timing") if allow_dirty_rows else bench
    timing_stable = (
        bool(timing.get("stable"))
        if isinstance(timing, Mapping) and "stable" in timing
        else bool(bench.get("timing_stable"))
    )
    checks = {
        "memory_clean": allow_dirty_rows or bool(bench.get("all_memory_clean")),
        "timing_stable": timing_stable,
    }
    return _finish("bench_clean", thresholds, checks, summary=bench)


def evaluate_lane_s(
    candidate_evidence: Path,
    control_evidence: Path,
    thresholds: dict[str, Any],
) -> GateResult:
    candidate = _summarize_benchmark(candidate_evidence)
    control = _summarize_benchmark(control_evidence)
    allow_dirty_rows = bool(thresholds.get("allow_dirty_rows"))
    candidate_timing = (
        candidate.get("attempted_timing") if allow_dirty_rows else candidate
    )
    control_timing = control.get("attempted_timing") if allow_dirty_rows else control
    candidate_median = (
        candidate_timing.get("median_seconds")
        if isinstance(candidate_timing, Mapping)
        else None
    )
    control_median = (
        control_timing.get("median_seconds") if isinstance(control_timing, Mapping) else None
    )
    candidate_timing_stable = (
        bool(candidate_timing.get("stable"))
        if isinstance(candidate_timing, Mapping) and "stable" in candidate_timing
        else bool(candidate.get("timing_stable"))
    )
    control_timing_stable = (
        bool(control_timing.get("stable"))
        if isinstance(control_timing, Mapping) and "stable" in control_timing
        else bool(control.get("timing_stable"))
    )
    ratio = (
        float(candidate_median) / float(control_median)
        if candidate_median is not None and control_median not in (None, 0)
        else None
    )
    checks = {
        "candidate_clean": allow_dirty_rows or bool(candidate.get("all_memory_clean")),
        "q2_control_clean": allow_dirty_rows or bool(control.get("all_memory_clean")),
        "candidate_timing_stable": candidate_timing_stable,
        "q2_control_timing_stable": control_timing_stable,
        "ratio": _check_le(ratio, thresholds["lane_s_ratio_max"]),
        "effective_bpw": _check_le(
            candidate.get("effective_bits_per_weight"), thresholds["effective_bpw_max"]
        ),
        "no_dense_routed_experts": candidate.get("dense_expert_params") is False,
        "no_unbound_vq_experts": candidate.get("unbound_vq_experts") is False,
        "non_expert_dtype_verified": candidate.get("non_expert_dtype_verified") is True,
    }
    summary = {"candidate": candidate, "q2_control": control, "ratio": ratio}
    return _finish("lane_s", thresholds, checks, summary=summary)


def _mapping_matches_exact(
    value: object,
    expected: Mapping[str, object],
) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == set(expected)
        and all(
            type(value.get(name)) is type(expected_value)
            and value.get(name) == expected_value
            for name, expected_value in expected.items()
        )
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _teacher_cache_identity_from_audit(
    audit: GLM52TeacherCacheAudit,
    *,
    artifact_sha256: str,
    manifest_file_sha256: str,
) -> dict[str, object]:
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


def _reaudit_glm52_teacher_cache_authority(
    authority: object,
) -> dict[str, object]:
    expected_fields = {
        "cache_root",
        "manifest_path",
        "prompt_authority_path",
        "artifact_sha256",
    }
    if not isinstance(authority, Mapping) or set(authority) != expected_fields:
        raise ValueError("raw teacher cache authority inventory is invalid")
    values = {name: authority.get(name) for name in expected_fields}
    if not all(isinstance(value, str) and value for value in values.values()):
        raise ValueError("raw teacher cache authority values must be strings")
    root = Path(str(values["cache_root"])).expanduser().absolute()
    manifest_path = Path(str(values["manifest_path"])).expanduser().absolute()
    prompt_path = Path(str(values["prompt_authority_path"])).expanduser().absolute()
    if str(root) != values["cache_root"]:
        raise ValueError("raw teacher cache root must be an absolute path")
    if str(manifest_path) != values["manifest_path"]:
        raise ValueError("raw teacher cache manifest must be an absolute path")
    if str(prompt_path) != values["prompt_authority_path"]:
        raise ValueError("teacher cache prompt authority must be an absolute path")
    if manifest_path != root / GLM52_TEACHER_CACHE_MANIFEST_FILENAME:
        raise ValueError("teacher cache manifest is not canonical for the raw root")
    expected_artifact_sha256 = values["artifact_sha256"]
    if not _is_sha256(expected_artifact_sha256):
        raise ValueError("teacher cache artifact SHA-256 is invalid")
    before_artifact_sha256 = teacher_cache_artifact_hash(root)
    if before_artifact_sha256 != expected_artifact_sha256:
        raise ValueError("teacher cache artifact SHA-256 does not match raw bytes")
    manifest_raw = manifest_path.read_bytes()
    manifest_file_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    manifest = json.loads(manifest_raw)
    if not isinstance(manifest, Mapping):
        raise ValueError("teacher cache manifest must contain an object")
    for field in ("producer", "source_evidence", "non_vq_package"):
        if not isinstance(manifest.get(field), Mapping):
            raise ValueError(f"teacher cache manifest {field} must be an object")
    if dict(manifest["source_evidence"]) != GLM52_TEACHER_CACHE_SOURCE_EVIDENCE:
        raise ValueError("teacher cache source evidence is not frozen")
    contract = GLM52TeacherCacheContract.from_frozen_prompt_pack(
        prompt_path,
        producer=dict(manifest["producer"]),
        source_evidence=dict(manifest["source_evidence"]),
        non_vq_package=dict(manifest["non_vq_package"]),
    )
    audit = audit_glm52_teacher_cache(root, contract=contract)
    if audit.valid is not True or audit.release_eligible is not True:
        raise ValueError("teacher cache strict re-audit is not release eligible")
    if teacher_cache_artifact_hash(root) != before_artifact_sha256:
        raise ValueError("teacher cache artifact changed across gate re-audit")
    return _teacher_cache_identity_from_audit(
        audit,
        artifact_sha256=before_artifact_sha256,
        manifest_file_sha256=manifest_file_sha256,
    )


def _absolute_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value).expanduser().absolute()
    if str(path) != value:
        raise ValueError(f"{label} must be an absolute path")
    return path


def _strict_json_object(path: Path, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    value = json.loads(
        path.read_bytes(),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return value


def _reaudit_glm52_candidate_evaluation_authority(authority: object) -> bool:
    if not isinstance(authority, Mapping) or set(authority) != {
        "teacher_cache_root",
        "candidate_cache_root",
        "prompt_pack_path",
        "family_policy_path",
    }:
        raise ValueError("raw candidate evaluation authority inventory is invalid")
    from mlx_vq.quality.glm52_candidate_eval import compare_glm52_candidate_caches

    teacher_root = _absolute_path(authority["teacher_cache_root"], label="teacher cache root")
    candidate_root = _absolute_path(authority["candidate_cache_root"], label="candidate cache root")
    prompt_pack = _absolute_path(authority["prompt_pack_path"], label="prompt pack")
    policy_path = _absolute_path(authority["family_policy_path"], label="family policy")
    comparison = compare_glm52_candidate_caches(
        teacher_root,
        candidate_root,
        policy_path=policy_path,
        prompt_pack_path=prompt_pack,
    )
    return isinstance(comparison, Mapping) and comparison.get("family_eval_gate_pass") is True


def _reaudit_glm52_route_diagnostics_authority(
    authority: object,
    *,
    family_policy_path: Path,
) -> bool:
    if not isinstance(authority, Mapping) or set(authority) != {
        "source_root",
        "candidate_root",
        "source_authority_path",
        "candidate_authority_path",
    }:
        raise ValueError("raw route diagnostics authority inventory is invalid")
    from mlx_vq.quality.glm52_route_diagnostics import (
        compare_route_trace_artifacts,
        validate_capture_authority,
    )

    source_root = _absolute_path(authority["source_root"], label="source route root")
    candidate_root = _absolute_path(authority["candidate_root"], label="candidate route root")
    source_authority_path = _absolute_path(authority["source_authority_path"], label="source route authority")
    candidate_authority_path = _absolute_path(authority["candidate_authority_path"], label="candidate route authority")
    source_capture = validate_capture_authority(
        _strict_json_object(source_authority_path, label="source route authority"),
        label="source route authority",
    )
    candidate_capture = validate_capture_authority(
        _strict_json_object(candidate_authority_path, label="candidate route authority"),
        label="candidate route authority",
    )
    evidence = compare_route_trace_artifacts(
        source_root,
        candidate_root,
        policy_json=family_policy_path,
        require_frozen_batch=True,
        expected_source_authority=source_capture,
        expected_candidate_authority=candidate_capture,
    )
    return (
        isinstance(evidence, Mapping)
        and isinstance(evidence.get("evaluation"), Mapping)
        and isinstance(evidence.get("checks"), Mapping)
        and evidence["evaluation"].get("diagnostic_only") is True
        and evidence["evaluation"].get("release_eligible") is True
        and evidence["checks"].get("structural_math_pass") is True
    )


def _reaudit_glm52_benchmark_authority(
    authority: object,
    *,
    policy_path: Path,
) -> bool:
    evidence_path = _absolute_path(authority, label="raw benchmark evidence")
    policy = validate_glm52_family_gate_policy(
        _strict_json_object(policy_path, label="family policy")
    )
    load_glm52_same_machine_benchmark_evidence(evidence_path, policy=policy)
    return True


def evaluate_glm52_family(
    evidence_path: Path,
    thresholds: dict[str, Any],
) -> GateResult:
    payload = _load_json_object(evidence_path)
    body = dict(payload)
    embedded_digest = body.pop("gate_contract_sha256", None)
    embedded_checks = payload.get("checks")
    evidence_checks = (
        dict(embedded_checks) if isinstance(embedded_checks, Mapping) else {}
    )
    boolean_checks = bool(evidence_checks) and all(
        isinstance(value, bool) for value in evidence_checks.values()
    )
    missing = payload.get("missing_requirements")
    missing_list = missing if isinstance(missing, list) else None
    gate_schema_version = payload.get("gate_schema_version")
    exact_check_inventory = set(evidence_checks) == set(
        GLM52_FAMILY_GATE_V1_CHECK_NAMES
    )
    v1_required_ready = {
        "policy_frozen": True,
        "prompt_pack_frozen": True,
        "teacher_source_metadata_ready": True,
        "teacher_cache_full_vocabulary_ready": False,
        "indexshare_runtime_contract_ready": True,
        "synthetic_generation_contract_ready": True,
        "full_routed_coverage_ready": False,
        "full_225_group_artifact_ready": False,
        "accepted_artifact_bytes_and_bpw_ready": False,
        "production_binding_and_generation_ready": False,
        "full_vocabulary_source_relative_family_eval_ready": False,
        "route_math_diagnostics_ready": False,
        "same_machine_pinned_fp4_benchmark_ready": False,
        "no_dense_routed_experts": True,
    }
    v2_required_ready = {
        **v1_required_ready,
        "full_routed_coverage_ready": True,
        "full_225_group_artifact_ready": True,
        "accepted_artifact_bytes_and_bpw_ready": True,
        "production_binding_and_generation_ready": True,
    }
    v3_required_ready = {
        **v2_required_ready,
        "teacher_cache_full_vocabulary_ready": True,
    }
    v2_validator_readiness = {
        "artifact": True,
        "production": True,
        "family_eval": False,
        "family_benchmark": False,
    }
    submitted_validator_readiness = payload.get(
        "raw_evidence_validator_readiness"
    )
    validator_readiness_exact = (
        isinstance(submitted_validator_readiness, Mapping)
        and set(submitted_validator_readiness) == set(v2_validator_readiness)
        and all(
            submitted_validator_readiness.get(name) is expected
            for name, expected in v2_validator_readiness.items()
        )
    )
    v3_validator_readiness = {
        "artifact": True,
        "production": True,
        "teacher_cache": True,
        "family_eval": False,
        "family_benchmark": False,
    }
    v3_validator_readiness_exact = (
        isinstance(submitted_validator_readiness, Mapping)
        and set(submitted_validator_readiness) == set(v3_validator_readiness)
        and all(
            submitted_validator_readiness.get(name) is expected
            for name, expected in v3_validator_readiness.items()
        )
    )
    common_artifact_identity = payload.get("common_artifact_identity")
    common_artifact_identity_digest = (
        canonical_sha256(dict(common_artifact_identity))
        if isinstance(common_artifact_identity, Mapping)
        and bool(common_artifact_identity)
        else None
    )
    common_artifact_identity_authenticated = (
        common_artifact_identity_digest is not None
        and payload.get("common_artifact_identity_sha256")
        == common_artifact_identity_digest
        == GLM52_ACCEPTED_PRODUCTION_COMPOSITE_IDENTITY_SHA256
    )
    v2_hash_inventories_authenticated = _mapping_matches_exact(
        payload.get("input_evidence_file_sha256"),
        _GLM52_V2_INPUT_FILE_SHA256,
    ) and _mapping_matches_exact(
        payload.get("input_evidence_contract_sha256"),
        _GLM52_V2_INPUT_CONTRACT_SHA256,
    )
    identity_routed_group_count = (
        common_artifact_identity.get("routed_group_count")
        if isinstance(common_artifact_identity, Mapping)
        else None
    )
    identity_tensor_payload_bytes = (
        common_artifact_identity.get("whole_model_tensor_payload_bytes")
        if isinstance(common_artifact_identity, Mapping)
        else None
    )
    identity_tensor_payload_bpw = (
        common_artifact_identity.get("whole_model_tensor_payload_bpw")
        if isinstance(common_artifact_identity, Mapping)
        else None
    )
    raw_trio_summary_cross_bound = (
        isinstance(common_artifact_identity, Mapping)
        and payload.get("profile") == GLM52_REAP_PROFILE
        and common_artifact_identity.get("profile") == GLM52_REAP_PROFILE
        and payload.get("family_policy_contract_sha256")
        == _GLM52_V2_FAMILY_POLICY_CONTRACT_SHA256
        and payload.get("prompt_pack_contract_sha256")
        == _GLM52_V2_PROMPT_PACK_CONTRACT_SHA256
        and type(payload.get("accepted_tensor_payload_bytes")) is int
        and payload.get("accepted_tensor_payload_bytes")
        == identity_tensor_payload_bytes
        == GLM52_ACCEPTED_TENSOR_PAYLOAD_BYTES
        and type(payload.get("accepted_whole_main_bpw")) is float
        and type(identity_tensor_payload_bpw) is float
        and payload.get("accepted_whole_main_bpw") == GLM52_ACCEPTED_POLICY_BPW
        and round(identity_tensor_payload_bpw, 7)
        == payload.get("accepted_whole_main_bpw")
        and type(identity_routed_group_count) is int
        and type(payload.get("expected_routed_group_count")) is int
        and type(payload.get("present_routed_group_count")) is int
        and type(payload.get("missing_routed_group_count")) is int
        and payload.get("expected_routed_group_count")
        == identity_routed_group_count
        and payload.get("present_routed_group_count")
        == identity_routed_group_count
        and payload.get("missing_routed_group_count") == 0
        and payload.get("production_binding_proven") is True
        and payload.get("production_generation_proven") is True
        and evidence_checks.get("production_binding_and_generation_ready") is True
        and evidence_checks.get("full_routed_coverage_ready") is True
        and evidence_checks.get("full_225_group_artifact_ready") is True
        and evidence_checks.get("accepted_artifact_bytes_and_bpw_ready") is True
    )
    v2_summary_cross_bound = (
        raw_trio_summary_cross_bound
        and payload.get("teacher_cache_payload_present") is False
        and evidence_checks.get("teacher_cache_full_vocabulary_ready") is False
        and payload.get("full_vocabulary_eval_proven") is False
        and evidence_checks.get("full_vocabulary_source_relative_family_eval_ready") is False
        and payload.get("same_machine_benchmark_proven") is False
        and evidence_checks.get("same_machine_pinned_fp4_benchmark_ready") is False
    )
    teacher_cache_identity = payload.get("audited_teacher_cache_identity")
    teacher_cache_identity_digest = (
        canonical_sha256(dict(teacher_cache_identity))
        if isinstance(teacher_cache_identity, Mapping)
        else None
    )
    reaudited_teacher_cache_identity: dict[str, object] | None = None
    if gate_schema_version in (
        GLM52_FAMILY_GATE_V3_SCHEMA_VERSION,
        GLM52_FAMILY_GATE_V4_SCHEMA_VERSION,
        GLM52_FAMILY_GATE_V5_SCHEMA_VERSION,
        GLM52_FAMILY_GATE_V6_SCHEMA_VERSION,
    ):
        try:
            reaudited_teacher_cache_identity = (
                _reaudit_glm52_teacher_cache_authority(
                    payload.get("raw_teacher_cache_authority")
                )
            )
        except (OSError, TypeError, ValueError):
            reaudited_teacher_cache_identity = None
    teacher_cache_identity_exact = (
        isinstance(teacher_cache_identity, Mapping)
        and set(teacher_cache_identity)
        == {
            "artifact_sha256",
            "cache_content_sha256",
            "manifest_body_sha256",
            "manifest_file_sha256",
            "prompt_count",
            "source_token_count",
            "predictor_position_count",
            "fp32_value_count",
            "raw_tensor_bytes",
            "split_position_counts",
            "split_raw_tensor_bytes",
            "release_eligible",
        }
        and _is_sha256(teacher_cache_identity.get("artifact_sha256"))
        and _is_sha256(teacher_cache_identity.get("cache_content_sha256"))
        and _is_sha256(teacher_cache_identity.get("manifest_body_sha256"))
        and _is_sha256(teacher_cache_identity.get("manifest_file_sha256"))
        and type(teacher_cache_identity.get("prompt_count")) is int
        and teacher_cache_identity.get("prompt_count") == 66
        and type(teacher_cache_identity.get("source_token_count")) is int
        and teacher_cache_identity.get("source_token_count") == 810
        and type(teacher_cache_identity.get("predictor_position_count")) is int
        and teacher_cache_identity.get("predictor_position_count") == 744
        and type(teacher_cache_identity.get("fp32_value_count")) is int
        and teacher_cache_identity.get("fp32_value_count") == 115_230_720
        and type(teacher_cache_identity.get("raw_tensor_bytes")) is int
        and teacher_cache_identity.get("raw_tensor_bytes") == 460_922_880
        and _mapping_matches_exact(
            teacher_cache_identity.get("split_position_counts"),
            {"report": 238, "selection": 255, "holdout": 251},
        )
        and _mapping_matches_exact(
            teacher_cache_identity.get("split_raw_tensor_bytes"),
            {
                "report": 147_445_760,
                "selection": 157_977_600,
                "holdout": 155_499_520,
            },
        )
        and teacher_cache_identity.get("release_eligible") is True
        and reaudited_teacher_cache_identity is not None
        and dict(teacher_cache_identity) == reaudited_teacher_cache_identity
    )
    teacher_cache_identity_authenticated = (
        teacher_cache_identity_exact
        and teacher_cache_identity_digest is not None
        and payload.get("audited_teacher_cache_identity_sha256")
        == teacher_cache_identity_digest
    )
    submitted_file_hashes = payload.get("input_evidence_file_sha256")
    submitted_contract_hashes = payload.get("input_evidence_contract_sha256")
    v3_hash_inventories_authenticated = (
        teacher_cache_identity_authenticated
        and isinstance(submitted_file_hashes, Mapping)
        and set(submitted_file_hashes)
        == {
            *_GLM52_V2_INPUT_FILE_SHA256,
            "teacher_cache_artifact",
            "teacher_cache_manifest",
        }
        and all(
            submitted_file_hashes.get(name) == digest
            for name, digest in _GLM52_V2_INPUT_FILE_SHA256.items()
        )
        and submitted_file_hashes.get("teacher_cache_manifest")
        == teacher_cache_identity.get("manifest_file_sha256")
        and submitted_file_hashes.get("teacher_cache_artifact")
        == teacher_cache_identity.get("artifact_sha256")
        and isinstance(submitted_contract_hashes, Mapping)
        and set(submitted_contract_hashes)
        == {*_GLM52_V2_INPUT_CONTRACT_SHA256, "teacher_cache_audit"}
        and all(
            submitted_contract_hashes.get(name) == digest
            for name, digest in _GLM52_V2_INPUT_CONTRACT_SHA256.items()
        )
        and submitted_contract_hashes.get("teacher_cache_audit")
        == teacher_cache_identity_digest
    )
    v3_summary_cross_bound = (
        raw_trio_summary_cross_bound
        and payload.get("teacher_cache_payload_present") is True
        and evidence_checks.get("teacher_cache_full_vocabulary_ready") is True
        and teacher_cache_identity_authenticated
    )
    v4_validator_readiness = {
        "artifact": True,
        "production": True,
        "teacher_cache": True,
        "family_eval": True,
        "family_benchmark": False,
    }
    v5_validator_readiness = {
        **v4_validator_readiness,
        "route_math_diagnostics": True,
    }
    v6_validator_readiness = {**v5_validator_readiness, "family_benchmark": True}
    def readiness_is(expected: Mapping[str, bool]) -> bool:
        return (
            isinstance(submitted_validator_readiness, Mapping)
            and set(submitted_validator_readiness) == set(expected)
            and all(submitted_validator_readiness.get(name) is value for name, value in expected.items())
        )
    candidate_eval_authenticated = False
    route_diagnostics_authenticated = False
    benchmark_authenticated = False
    benchmark_provenance_authenticated = (
        payload.get("raw_benchmark_evidence_provenance")
        == "local_filesystem_hash_referenced_evidence_v1"
    )
    candidate_authority = payload.get("raw_candidate_evaluation_authority")
    try:
        candidate_eval_authenticated = _reaudit_glm52_candidate_evaluation_authority(
            candidate_authority
        )
    except (OSError, TypeError, ValueError):
        candidate_eval_authenticated = False
    policy_path: Path | None = None
    if isinstance(candidate_authority, Mapping):
        try:
            policy_path = _absolute_path(
                candidate_authority.get("family_policy_path"), label="family policy"
            )
        except ValueError:
            policy_path = None
    if policy_path is not None:
        try:
            route_diagnostics_authenticated = _reaudit_glm52_route_diagnostics_authority(
                payload.get("raw_route_diagnostics_authority"),
                family_policy_path=policy_path,
            )
        except (OSError, TypeError, ValueError):
            route_diagnostics_authenticated = False
        try:
            benchmark_authenticated = _reaudit_glm52_benchmark_authority(
                payload.get("raw_benchmark_evidence_authority"),
                policy_path=policy_path,
            )
        except (OSError, TypeError, ValueError):
            benchmark_authenticated = False
    v4_summary_cross_bound = (
        v3_summary_cross_bound
        and payload.get("full_vocabulary_eval_proven") is True
        and evidence_checks.get("full_vocabulary_source_relative_family_eval_ready") is True
        and payload.get("same_machine_benchmark_proven") is False
        and evidence_checks.get("same_machine_pinned_fp4_benchmark_ready") is False
        and candidate_eval_authenticated
    )
    v5_summary_cross_bound = (
        v4_summary_cross_bound
        and evidence_checks.get("route_math_diagnostics_ready") is True
        and route_diagnostics_authenticated
    )
    v6_summary_cross_bound = (
        v5_summary_cross_bound
        and payload.get("same_machine_benchmark_proven") is True
        and evidence_checks.get("same_machine_pinned_fp4_benchmark_ready") is True
        and benchmark_authenticated
        and benchmark_provenance_authenticated
    )
    common_blocked_contract = (
        type(gate_schema_version) is int
        and payload.get("release_pass_enabled") is False
        and payload.get("raw_release_evidence_validators_ready") is False
        and payload.get("family_gate_pass") is False
        and payload.get("gate_status") == "glm52_family_gate_blocked"
        and boolean_checks
        and exact_check_inventory
    )
    v1_blocked_contract_consistent = (
        common_blocked_contract
        and gate_schema_version == GLM52_FAMILY_GATE_V1_SCHEMA_VERSION
        and missing_list == list(GLM52_FAMILY_GATE_V1_MISSING_REQUIREMENTS)
        and all(
            evidence_checks.get(name) is expected
            for name, expected in v1_required_ready.items()
        )
    )
    v2_blocked_contract_consistent = (
        common_blocked_contract
        and gate_schema_version == GLM52_FAMILY_GATE_V2_SCHEMA_VERSION
        and missing_list == list(GLM52_FAMILY_GATE_V2_MISSING_REQUIREMENTS)
        and all(
            evidence_checks.get(name) is expected
            for name, expected in v2_required_ready.items()
        )
        and common_artifact_identity_authenticated
        and validator_readiness_exact
        and v2_hash_inventories_authenticated
        and v2_summary_cross_bound
    )
    v3_blocked_contract_consistent = (
        common_blocked_contract
        and gate_schema_version == GLM52_FAMILY_GATE_V3_SCHEMA_VERSION
        and missing_list == list(GLM52_FAMILY_GATE_V3_MISSING_REQUIREMENTS)
        and all(
            evidence_checks.get(name) is expected
            for name, expected in v3_required_ready.items()
        )
        and common_artifact_identity_authenticated
        and v3_validator_readiness_exact
        and v3_hash_inventories_authenticated
        and v3_summary_cross_bound
    )
    v4_blocked_contract_consistent = (
        common_blocked_contract
        and gate_schema_version == GLM52_FAMILY_GATE_V4_SCHEMA_VERSION
        and missing_list == list(GLM52_FAMILY_GATE_V4_MISSING_REQUIREMENTS)
        and all(
            evidence_checks.get(name) is expected
            for name, expected in {**v3_required_ready, "full_vocabulary_source_relative_family_eval_ready": True}.items()
        )
        and common_artifact_identity_authenticated
        and readiness_is(v4_validator_readiness)
        and v3_hash_inventories_authenticated
        and v4_summary_cross_bound
    )
    v5_blocked_contract_consistent = (
        common_blocked_contract
        and gate_schema_version == GLM52_FAMILY_GATE_V5_SCHEMA_VERSION
        and missing_list == list(GLM52_FAMILY_GATE_V5_MISSING_REQUIREMENTS)
        and all(
            evidence_checks.get(name) is expected
            for name, expected in {
                **v3_required_ready,
                "full_vocabulary_source_relative_family_eval_ready": True,
                "route_math_diagnostics_ready": True,
            }.items()
        )
        and common_artifact_identity_authenticated
        and readiness_is(v5_validator_readiness)
        and v3_hash_inventories_authenticated
        and v5_summary_cross_bound
    )
    v6_pass_contract_consistent = (
        gate_schema_version == GLM52_FAMILY_GATE_V6_SCHEMA_VERSION
        and payload.get("release_pass_enabled") is True
        and payload.get("raw_release_evidence_validators_ready") is True
        and payload.get("family_gate_pass") is True
        and payload.get("gate_status") == "glm52_family_gate_passed"
        and missing_list == list(GLM52_FAMILY_GATE_V6_MISSING_REQUIREMENTS)
        and boolean_checks
        and exact_check_inventory
        and all(evidence_checks.values())
        and common_artifact_identity_authenticated
        and readiness_is(v6_validator_readiness)
        and v3_hash_inventories_authenticated
        and v6_summary_cross_bound
    )
    blocked_contract_consistent = (
        v1_blocked_contract_consistent
        or v2_blocked_contract_consistent
        or v3_blocked_contract_consistent
        or v4_blocked_contract_consistent
        or v5_blocked_contract_consistent
        or v6_pass_contract_consistent
    )
    checks: dict[str, bool] = {
        "evidence_present": bool(payload),
        "gate_contract_authenticated": (
            embedded_digest == canonical_sha256(body)
        ),
        "record_type": payload.get("record_type") == "glm52_family_gate",
        "model_id": payload.get("model_id") == PINNED_GLM52_MODEL_ID,
        "revision": payload.get("revision") == PINNED_GLM52_REVISION,
        "embedded_checks_well_formed": boolean_checks,
        "exact_v1_check_inventory": exact_check_inventory,
        "missing_requirements_well_formed": missing_list is not None,
        **(
            {"release_pass_enabled": payload.get("release_pass_enabled") is True}
            if gate_schema_version == GLM52_FAMILY_GATE_V6_SCHEMA_VERSION
            else {"release_pass_disabled": payload.get("release_pass_enabled") is False}
        ),
        "blocked_contract_consistent": blocked_contract_consistent,
        **{str(name): value for name, value in evidence_checks.items()},
        "family_gate_pass": payload.get("family_gate_pass") is True,
        "missing_requirements_empty": missing_list == [],
    }
    if gate_schema_version in (
        GLM52_FAMILY_GATE_V2_SCHEMA_VERSION,
        GLM52_FAMILY_GATE_V3_SCHEMA_VERSION,
        GLM52_FAMILY_GATE_V4_SCHEMA_VERSION,
        GLM52_FAMILY_GATE_V5_SCHEMA_VERSION,
        GLM52_FAMILY_GATE_V6_SCHEMA_VERSION,
    ):
        checks["common_artifact_identity_authenticated"] = (
            common_artifact_identity_authenticated
        )
    if gate_schema_version == GLM52_FAMILY_GATE_V2_SCHEMA_VERSION:
        checks["v2_hash_inventories_authenticated"] = (
            v2_hash_inventories_authenticated
        )
        checks["v2_summary_cross_bound"] = v2_summary_cross_bound
    if gate_schema_version in (
        GLM52_FAMILY_GATE_V3_SCHEMA_VERSION,
        GLM52_FAMILY_GATE_V4_SCHEMA_VERSION,
        GLM52_FAMILY_GATE_V5_SCHEMA_VERSION,
        GLM52_FAMILY_GATE_V6_SCHEMA_VERSION,
    ):
        checks["audited_teacher_cache_identity_authenticated"] = (
            teacher_cache_identity_authenticated
        )
        checks["v3_hash_inventories_authenticated"] = (
            v3_hash_inventories_authenticated
        )
        checks["v3_summary_cross_bound"] = v3_summary_cross_bound
    if gate_schema_version in (
        GLM52_FAMILY_GATE_V4_SCHEMA_VERSION,
        GLM52_FAMILY_GATE_V5_SCHEMA_VERSION,
        GLM52_FAMILY_GATE_V6_SCHEMA_VERSION,
    ):
        checks["raw_candidate_evaluation_authenticated"] = candidate_eval_authenticated
    if gate_schema_version in (
        GLM52_FAMILY_GATE_V5_SCHEMA_VERSION,
        GLM52_FAMILY_GATE_V6_SCHEMA_VERSION,
    ):
        checks["raw_route_diagnostics_authenticated"] = route_diagnostics_authenticated
    if gate_schema_version == GLM52_FAMILY_GATE_V6_SCHEMA_VERSION:
        checks["raw_benchmark_evidence_authenticated"] = benchmark_authenticated
    return _finish(
        "glm52_family",
        thresholds,
        checks,
        summary=payload,
    )


def evaluate_gate(
    *,
    profile: str,
    overrides: Mapping[str, Any] | None,
    artifact_dir: Path | None,
    evidence_path: Path | None,
    control_evidence: Path | None,
) -> GateResult:
    thresholds = resolve_gate_thresholds(profile, overrides)
    if profile == "train_sane":
        if artifact_dir is None:
            raise ValueError("train_sane gate requires an artifact output")
        return evaluate_train_sane(artifact_dir, thresholds)
    if profile == "audit_ok":
        if evidence_path is None:
            raise ValueError("audit_ok gate requires audit evidence")
        return evaluate_audit_ok(evidence_path, thresholds)
    if profile in ("balanced_rc_split", "community_wow"):
        if evidence_path is None:
            raise ValueError(f"{profile} gate requires eval evidence")
        return evaluate_eval_split(profile, evidence_path, thresholds)
    if profile == "bench_clean":
        if evidence_path is None:
            raise ValueError("bench_clean gate requires benchmark evidence")
        return evaluate_bench_clean(evidence_path, thresholds)
    if profile == "lane_s":
        if evidence_path is None or control_evidence is None:
            raise ValueError("lane_s gate requires candidate and control evidence")
        return evaluate_lane_s(evidence_path, control_evidence, thresholds)
    if profile == "glm52_family":
        if evidence_path is None:
            raise ValueError("glm52_family gate requires family gate evidence")
        return evaluate_glm52_family(evidence_path, thresholds)
    raise KeyError(f"no evaluator for gate profile {profile!r}")
