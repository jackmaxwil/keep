from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlx_vq.build import gates as build_gates
from mlx_vq.build.gates import (
    _GLM52_V2_FAMILY_POLICY_CONTRACT_SHA256,
    _GLM52_V2_INPUT_CONTRACT_SHA256,
    _GLM52_V2_INPUT_FILE_SHA256,
    _GLM52_V2_PROMPT_PACK_CONTRACT_SHA256,
    evaluate_gate,
)
from mlx_vq.quality.glm52_family import (
    GLM52_FAMILY_GATE_V1_CHECK_NAMES,
    GLM52_FAMILY_GATE_V1_MISSING_REQUIREMENTS,
    GLM52_FAMILY_GATE_V1_SCHEMA_VERSION,
    GLM52_FAMILY_GATE_V2_MISSING_REQUIREMENTS,
    GLM52_FAMILY_GATE_V2_SCHEMA_VERSION,
    GLM52_FAMILY_GATE_V3_MISSING_REQUIREMENTS,
    GLM52_FAMILY_GATE_V3_SCHEMA_VERSION,
    canonical_sha256,
)


def _eval_row(
    *,
    prompt_id: str,
    top1: float = 0.9,
    mean_kld: float = 0.2,
    ppl_ratio: float = 1.0,
    token_klds: list[float] | None = None,
    pageouts: int = 0,
    swapouts: int = 0,
) -> dict:
    return {
        "prompt_id": prompt_id,
        "top1_agreement": top1,
        "mean_kld": mean_kld,
        "ppl_ratio": ppl_ratio,
        "nll_delta": 0.01,
        "token_klds": token_klds or [0.1, 0.2],
        "pageouts_delta": pageouts,
        "swapouts_delta": swapouts,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _bench_row(
    *,
    prefill_seconds: float,
    bpw: float = 2.036,
    pageouts: int = 0,
    swapouts: int = 0,
) -> dict:
    return {
        "prefill_seconds": prefill_seconds,
        "effective_bits_per_weight": bpw,
        "dense_expert_params": False,
        "unbound_vq_experts": False,
        "non_expert_dtype_verified": True,
        "pageouts_delta": pageouts,
        "swapouts_delta": swapouts,
        "invalid_memory_pressure": False,
    }


def test_balanced_rc_split_gate_pass_and_dirty_row_failure(tmp_path: Path) -> None:
    rows = [_eval_row(prompt_id=f"report_math_{i:03d}") for i in range(4)]
    evidence = _write_jsonl(tmp_path / "eval.jsonl", rows)
    result = evaluate_gate(
        profile="balanced_rc_split",
        overrides={"clean_rows": 4},
        artifact_dir=None,
        evidence_path=evidence,
        control_evidence=None,
    )
    assert result.passed and result.checks == {"clean_rows": True, "mean_ppl_ratio": True}

    dirty_rows = rows[:-1] + [_eval_row(prompt_id="report_math_003", pageouts=5)]
    dirty_evidence = _write_jsonl(tmp_path / "dirty.jsonl", dirty_rows)
    dirty_result = evaluate_gate(
        profile="balanced_rc_split",
        overrides={"clean_rows": 4},
        artifact_dir=None,
        evidence_path=dirty_evidence,
        control_evidence=None,
    )
    assert not dirty_result.passed
    assert "clean_rows" in dirty_result.reasons


def test_balanced_rc_split_gate_can_allow_dirty_rows_explicitly(tmp_path: Path) -> None:
    dirty_rows = [
        _eval_row(prompt_id=f"report_math_{i:03d}", pageouts=5) for i in range(4)
    ]
    dirty_evidence = _write_jsonl(tmp_path / "dirty.jsonl", dirty_rows)
    result = evaluate_gate(
        profile="balanced_rc_split",
        overrides={"clean_rows": 0, "allow_dirty_rows": True},
        artifact_dir=None,
        evidence_path=dirty_evidence,
        control_evidence=None,
    )
    assert result.passed
    assert result.checks["clean_rows"] is True


def test_community_wow_gate_boundary_values(tmp_path: Path) -> None:
    passing = [
        _eval_row(
            prompt_id=f"report_route_{i:03d}",
            top1=0.85,
            mean_kld=0.30,
            token_klds=[3.0, 2.0],
        )
        for i in range(4)
    ]
    evidence = _write_jsonl(tmp_path / "wow.jsonl", passing)
    result = evaluate_gate(
        profile="community_wow",
        overrides={"clean_rows": 4},
        artifact_dir=None,
        evidence_path=evidence,
        control_evidence=None,
    )
    assert result.passed, result.reasons

    missed = [
        _eval_row(prompt_id=f"report_route_{i:03d}", top1=0.79) for i in range(4)
    ]
    missed_evidence = _write_jsonl(tmp_path / "wow-miss.jsonl", missed)
    missed_result = evaluate_gate(
        profile="community_wow",
        overrides={"clean_rows": 4},
        artifact_dir=None,
        evidence_path=missed_evidence,
        control_evidence=None,
    )
    assert not missed_result.passed
    assert "top1" in missed_result.reasons
    assert "domain_top1" in missed_result.reasons


def test_lane_s_gate_ratio_and_bpw(tmp_path: Path) -> None:
    control = _write_jsonl(
        tmp_path / "q2.jsonl", [_bench_row(prefill_seconds=1.0) for _ in range(3)]
    )
    fast = _write_jsonl(
        tmp_path / "candidate.jsonl",
        [_bench_row(prefill_seconds=1.07) for _ in range(3)],
    )
    result = evaluate_gate(
        profile="lane_s",
        overrides=None,
        artifact_dir=None,
        evidence_path=fast,
        control_evidence=control,
    )
    assert result.passed
    assert result.summary["ratio"] == pytest.approx(1.07)

    slow = _write_jsonl(
        tmp_path / "slow.jsonl", [_bench_row(prefill_seconds=1.72) for _ in range(3)]
    )
    slow_result = evaluate_gate(
        profile="lane_s",
        overrides=None,
        artifact_dir=None,
        evidence_path=slow,
        control_evidence=control,
    )
    assert not slow_result.passed and "ratio" in slow_result.reasons

    heavy = _write_jsonl(
        tmp_path / "heavy.jsonl",
        [_bench_row(prefill_seconds=1.07, bpw=2.4) for _ in range(3)],
    )
    heavy_result = evaluate_gate(
        profile="lane_s",
        overrides=None,
        artifact_dir=None,
        evidence_path=heavy,
        control_evidence=control,
    )
    assert not heavy_result.passed and "effective_bpw" in heavy_result.reasons


def test_bench_clean_gate_flags_memory_dirty_rows(tmp_path: Path) -> None:
    dirty = _write_jsonl(
        tmp_path / "dirty-bench.jsonl",
        [
            _bench_row(prefill_seconds=1.0),
            _bench_row(prefill_seconds=1.0, pageouts=6),
            _bench_row(prefill_seconds=1.0),
        ],
    )
    result = evaluate_gate(
        profile="bench_clean",
        overrides=None,
        artifact_dir=None,
        evidence_path=dirty,
        control_evidence=None,
    )
    assert not result.passed and "memory_clean" in result.reasons


def test_bench_and_lane_s_can_allow_dirty_rows_explicitly(tmp_path: Path) -> None:
    dirty = _write_jsonl(
        tmp_path / "dirty-bench.jsonl",
        [
            _bench_row(prefill_seconds=1.0),
            _bench_row(prefill_seconds=1.0, pageouts=6),
            _bench_row(prefill_seconds=1.0),
        ],
    )
    bench_result = evaluate_gate(
        profile="bench_clean",
        overrides={"allow_dirty_rows": True},
        artifact_dir=None,
        evidence_path=dirty,
        control_evidence=None,
    )
    assert bench_result.passed
    assert bench_result.checks["memory_clean"] is True

    control = _write_jsonl(
        tmp_path / "q2.jsonl",
        [_bench_row(prefill_seconds=1.0, pageouts=4) for _ in range(3)],
    )
    candidate = _write_jsonl(
        tmp_path / "candidate.jsonl",
        [_bench_row(prefill_seconds=1.07, pageouts=6) for _ in range(3)],
    )
    lane_result = evaluate_gate(
        profile="lane_s",
        overrides={"allow_dirty_rows": True},
        artifact_dir=None,
        evidence_path=candidate,
        control_evidence=control,
    )
    assert lane_result.passed
    assert lane_result.checks["candidate_clean"] is True
    assert lane_result.checks["q2_control_clean"] is True


def test_train_sane_gate_reads_manifest(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    result = evaluate_gate(
        profile="train_sane",
        overrides=None,
        artifact_dir=artifact,
        evidence_path=None,
        control_evidence=None,
    )
    assert not result.passed and "manifest_present" in result.reasons

    (artifact / "conversion-manifest.json").write_text(
        json.dumps(
            {
                "continuous_parameters": {
                    "sidecars": [{"layer": 45, "projection": "gate_proj"}],
                    "run": {"final_train_loss": 4.71},
                }
            }
        )
    )
    passing = evaluate_gate(
        profile="train_sane",
        overrides=None,
        artifact_dir=artifact,
        evidence_path=None,
        control_evidence=None,
    )
    assert passing.passed


def test_audit_ok_gate(tmp_path: Path) -> None:
    evidence = tmp_path / "audit.jsonl"
    evidence.write_text(
        json.dumps(
            {
                "projection_count": 135,
                "layer_count": 45,
                "auto_prefill_all_layers_nax_fast_compatible": True,
            }
        )
        + "\n"
    )
    result = evaluate_gate(
        profile="audit_ok",
        overrides=None,
        artifact_dir=None,
        evidence_path=evidence,
        control_evidence=None,
    )
    assert result.passed

    evidence.write_text(
        json.dumps(
            {
                "projection_count": 135,
                "layer_count": 45,
                "auto_prefill_all_layers_nax_fast_compatible": False,
            }
        )
        + "\n"
    )
    failing = evaluate_gate(
        profile="audit_ok",
        overrides=None,
        artifact_dir=None,
        evidence_path=evidence,
        control_evidence=None,
    )
    assert not failing.passed and "all_layers_nax_fast_compatible" in failing.reasons


def test_gate_threshold_override_changes_gate_key(tmp_path: Path) -> None:
    rows = [_eval_row(prompt_id=f"report_math_{i:03d}") for i in range(4)]
    evidence = _write_jsonl(tmp_path / "eval.jsonl", rows)
    default = evaluate_gate(
        profile="balanced_rc_split",
        overrides=None,
        artifact_dir=None,
        evidence_path=evidence,
        control_evidence=None,
    )
    overridden = evaluate_gate(
        profile="balanced_rc_split",
        overrides={"clean_rows": 4},
        artifact_dir=None,
        evidence_path=evidence,
        control_evidence=None,
    )
    assert default.gate_key != overridden.gate_key
    # 4 clean rows miss the default 128-row requirement but pass the override.
    assert not default.passed and overridden.passed


def _glm52_family_gate_checks(*, schema_version: int) -> dict[str, bool]:
    artifact_ready = schema_version in (
        GLM52_FAMILY_GATE_V2_SCHEMA_VERSION,
        GLM52_FAMILY_GATE_V3_SCHEMA_VERSION,
    )
    teacher_cache_ready = schema_version == GLM52_FAMILY_GATE_V3_SCHEMA_VERSION
    checks = {
        "policy_frozen": True,
        "prompt_pack_frozen": True,
        "teacher_source_metadata_ready": True,
        "teacher_cache_full_vocabulary_ready": teacher_cache_ready,
        "indexshare_runtime_contract_ready": True,
        "synthetic_generation_contract_ready": True,
        "full_routed_coverage_ready": artifact_ready,
        "full_225_group_artifact_ready": artifact_ready,
        "accepted_artifact_bytes_and_bpw_ready": artifact_ready,
        "production_binding_and_generation_ready": artifact_ready,
        "full_vocabulary_source_relative_family_eval_ready": False,
        "route_math_diagnostics_ready": False,
        "same_machine_pinned_fp4_benchmark_ready": False,
        "no_dense_routed_experts": True,
    }
    assert tuple(checks) == GLM52_FAMILY_GATE_V1_CHECK_NAMES
    return checks


def _glm52_common_artifact_identity() -> dict[str, object]:
    routed_groups = [
        f"{layer}:{projection}"
        for layer in range(3, 78)
        for projection in ("gate_proj", "up_proj", "down_proj")
    ]
    return {
        "schema_version": 1,
        "identity_kind": "glm52_production_composite_v1",
        "model_id": "0xSero/glm-5.2-reap-504B-v2",
        "source_revision": "6c9241aa05fb243a0edb7c804c213ec1cf5c920d",
        "profile": "glm52-reap-504b-v2",
        "profile_sha256": (
            "ae8b852139ac26e63846b0475064e5b955a00f1ff75f628682a3d8494474114d"
        ),
        "profile_contract_sha256": (
            "28d95f2f2e411ff886c38b0c773f335c34a1b044e401573e90b399c93db3dcb8"
        ),
        "config_sha256": (
            "5fa690755d0dab25a8e0e5e0745675bdac03ba2b6f5641da2931278235c71f1b"
        ),
        "source_index_sha256": (
            "bb5b4fa9782aea5ffc66f9145d6e630f1045d385c30437c531bebe422c075f3f"
        ),
        "source_blob_inventory_sha256": (
            "ace08e87dcbce3a22249e54196a27c0045992f8d5b8ca07f342899fa7a53fc8d"
        ),
        "source_inventory_sha256": (
            "bf42d5bc79e8eb5cef55304601571f1f7edfdda7f4086957ffb44c4a8334adde"
        ),
        "non_vq_manifest_sha256": (
            "5113750fdaf009b2174a2772dd2e3bccf490cfff354f082e5e0ad8af8771f8c8"
        ),
        "non_vq_package_index_sha256": (
            "ff7def155c88006cda458344b5cb8e8a20303049d20405cd4e9b28e7885df097"
        ),
        "non_vq_package_set_sha256": (
            "2719f13a66313b5b8acc4c053914cdbfa10fdbf103a62b924495eecddbcea2ed"
        ),
        "routed_manifest_sha256": (
            "ba1d3135ef8901f1a69ead28b5f9d330ef40d015ac31fdb4d2dcfe678c41f0a4"
        ),
        "routed_group_set_sha256": (
            "bb65cf9a0d4a78310eb46c25e07b6492e6f3a5b18b303a7d68a8f0c73819d3fe"
        ),
        "routed_group_inventory_sha256": canonical_sha256(routed_groups),
        "routed_group_count": 225,
        "code_bits": 8,
        "group_size": 512,
        "scale_estimator": "max_abs",
        "whole_model_parameter_count": 494_194_805_304,
        "whole_model_tensor_payload_bytes": 98_433_923_808,
        "whole_model_tensor_payload_bpw": (
            98_433_923_808 * 8 / 494_194_805_304
        ),
    }


def _glm52_blocked_checkpoint(*, schema_version: int) -> dict[str, object]:
    blocked: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_family_gate",
        "gate_schema_version": schema_version,
        "release_pass_enabled": False,
        "raw_release_evidence_validators_ready": False,
        "gate_status": "glm52_family_gate_blocked",
        "family_gate_pass": False,
        "model_id": "0xSero/glm-5.2-reap-504B-v2",
        "revision": "6c9241aa05fb243a0edb7c804c213ec1cf5c920d",
        "checks": _glm52_family_gate_checks(schema_version=schema_version),
        "missing_requirements": list(
            GLM52_FAMILY_GATE_V3_MISSING_REQUIREMENTS
            if schema_version == GLM52_FAMILY_GATE_V3_SCHEMA_VERSION
            else (
                GLM52_FAMILY_GATE_V2_MISSING_REQUIREMENTS
                if schema_version == GLM52_FAMILY_GATE_V2_SCHEMA_VERSION
                else GLM52_FAMILY_GATE_V1_MISSING_REQUIREMENTS
            )
        ),
    }
    if schema_version in (
        GLM52_FAMILY_GATE_V2_SCHEMA_VERSION,
        GLM52_FAMILY_GATE_V3_SCHEMA_VERSION,
    ):
        identity = _glm52_common_artifact_identity()
        blocked.update(
            {
                "common_artifact_identity": identity,
                "common_artifact_identity_sha256": canonical_sha256(identity),
                "raw_evidence_validator_readiness": {
                    "artifact": True,
                    "production": True,
                    "family_eval": False,
                    "family_benchmark": False,
                },
                "profile": "glm52-reap-504b-v2",
                "family_policy_contract_sha256": (
                    _GLM52_V2_FAMILY_POLICY_CONTRACT_SHA256
                ),
                "prompt_pack_contract_sha256": (
                    _GLM52_V2_PROMPT_PACK_CONTRACT_SHA256
                ),
                "accepted_tensor_payload_bytes": 98_433_923_808,
                "accepted_whole_main_bpw": 1.5934433,
                "expected_routed_group_count": 225,
                "present_routed_group_count": 225,
                "missing_routed_group_count": 0,
                "production_binding_proven": True,
                "production_generation_proven": True,
                "teacher_cache_payload_present": (
                    schema_version == GLM52_FAMILY_GATE_V3_SCHEMA_VERSION
                ),
                "full_vocabulary_eval_proven": False,
                "same_machine_benchmark_proven": False,
                "input_evidence_file_sha256": dict(
                    _GLM52_V2_INPUT_FILE_SHA256
                ),
                "input_evidence_contract_sha256": dict(
                    _GLM52_V2_INPUT_CONTRACT_SHA256
                ),
            }
        )
    if schema_version == GLM52_FAMILY_GATE_V3_SCHEMA_VERSION:
        teacher_cache_identity = {
            "artifact_sha256": "d" * 64,
            "cache_content_sha256": "a" * 64,
            "manifest_body_sha256": "b" * 64,
            "manifest_file_sha256": "c" * 64,
            "prompt_count": 66,
            "source_token_count": 810,
            "predictor_position_count": 744,
            "fp32_value_count": 115_230_720,
            "raw_tensor_bytes": 460_922_880,
            "split_position_counts": {
                "report": 238,
                "selection": 255,
                "holdout": 251,
            },
            "split_raw_tensor_bytes": {
                "report": 147_445_760,
                "selection": 157_977_600,
                "holdout": 155_499_520,
            },
            "release_eligible": True,
        }
        teacher_cache_digest = canonical_sha256(teacher_cache_identity)
        blocked["audited_teacher_cache_identity"] = teacher_cache_identity
        blocked["audited_teacher_cache_identity_sha256"] = teacher_cache_digest
        blocked["raw_teacher_cache_authority"] = {
            "cache_root": "/tmp/glm52-teacher-cache",
            "manifest_path": (
                "/tmp/glm52-teacher-cache/"
                "glm52-teacher-cache-fp32-manifest.json"
            ),
            "prompt_authority_path": "/tmp/glm52-prompts.json",
            "artifact_sha256": "d" * 64,
        }
        readiness = blocked["raw_evidence_validator_readiness"]
        assert isinstance(readiness, dict)
        readiness["teacher_cache"] = True
        file_hashes = blocked["input_evidence_file_sha256"]
        assert isinstance(file_hashes, dict)
        file_hashes["teacher_cache_artifact"] = "d" * 64
        file_hashes["teacher_cache_manifest"] = "c" * 64
        contract_hashes = blocked["input_evidence_contract_sha256"]
        assert isinstance(contract_hashes, dict)
        contract_hashes["teacher_cache_audit"] = teacher_cache_digest
    blocked["gate_contract_sha256"] = canonical_sha256(blocked)
    return blocked


def test_glm52_v2_hash_inventory_constants_remain_exact() -> None:
    assert _GLM52_V2_FAMILY_POLICY_CONTRACT_SHA256 == (
        "57ae812222de79555774296129e7110750f492794b1fe21bac91ed5cc986f206"
    )
    assert _GLM52_V2_PROMPT_PACK_CONTRACT_SHA256 == (
        "04c1ceacba775058de7d08462fbf69b534da7fbf766f57c8ce54daaebd84d2c1"
    )
    assert canonical_sha256(_GLM52_V2_INPUT_FILE_SHA256) == (
        "4982a1227c6d7991a65d2c27ef53d2cd99b3837c892103be6f518dc52dfb448d"
    )
    assert canonical_sha256(_GLM52_V2_INPUT_CONTRACT_SHA256) == (
        "63cf2fab6a1d0c792162cda2911168ba5fc7a04fe4d1f9779995591d389fd801"
    )


def _evaluate_glm52_checkpoint(
    tmp_path: Path,
    checkpoint: dict[str, object],
):
    evidence = tmp_path / "glm52-family-gate.json"
    evidence.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n")

    return evaluate_gate(
        profile="glm52_family",
        overrides=None,
        artifact_dir=None,
        evidence_path=evidence,
        control_evidence=None,
    )


def _rehash_glm52_checkpoint(checkpoint: dict[str, object]) -> None:
    checkpoint.pop("gate_contract_sha256", None)
    checkpoint["gate_contract_sha256"] = canonical_sha256(checkpoint)


def test_glm52_family_gate_accepts_historical_v1_blocked_contract(
    tmp_path: Path,
) -> None:
    blocked = _glm52_blocked_checkpoint(
        schema_version=GLM52_FAMILY_GATE_V1_SCHEMA_VERSION
    )

    result = _evaluate_glm52_checkpoint(tmp_path, blocked)

    assert result.passed is False
    assert result.checks["gate_contract_authenticated"] is True
    assert result.checks["blocked_contract_consistent"] is True


def test_glm52_family_gate_accepts_exact_authenticated_v2_blocked_checkpoint(
    tmp_path: Path,
) -> None:
    blocked = _glm52_blocked_checkpoint(
        schema_version=GLM52_FAMILY_GATE_V2_SCHEMA_VERSION
    )

    result = _evaluate_glm52_checkpoint(tmp_path, blocked)

    assert result.passed is False
    assert result.checks["gate_contract_authenticated"] is True
    assert result.checks["common_artifact_identity_authenticated"] is True
    assert result.checks["blocked_contract_consistent"] is True
    assert {
        "teacher_cache_full_vocabulary_ready",
        "full_vocabulary_source_relative_family_eval_ready",
        "route_math_diagnostics_ready",
        "same_machine_pinned_fp4_benchmark_ready",
    } <= set(result.reasons)


def test_glm52_family_gate_accepts_exact_authenticated_v3_blocked_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked = _glm52_blocked_checkpoint(
        schema_version=GLM52_FAMILY_GATE_V3_SCHEMA_VERSION
    )
    identity = blocked["audited_teacher_cache_identity"]
    assert isinstance(identity, dict)
    monkeypatch.setattr(
        build_gates,
        "_reaudit_glm52_teacher_cache_authority",
        lambda _authority: dict(identity),
    )

    result = _evaluate_glm52_checkpoint(tmp_path, blocked)

    assert result.passed is False
    assert result.checks["gate_contract_authenticated"] is True
    assert result.checks["common_artifact_identity_authenticated"] is True
    assert result.checks["audited_teacher_cache_identity_authenticated"] is True
    assert result.checks["blocked_contract_consistent"] is True
    assert set(result.reasons) >= {
        "full_vocabulary_source_relative_family_eval_ready",
        "route_math_diagnostics_ready",
        "same_machine_pinned_fp4_benchmark_ready",
    }
    assert "teacher_cache_full_vocabulary_ready" not in result.reasons


def test_glm52_family_gate_rejects_self_consistent_v3_without_raw_cache_authority(
    tmp_path: Path,
) -> None:
    fabricated = _glm52_blocked_checkpoint(
        schema_version=GLM52_FAMILY_GATE_V3_SCHEMA_VERSION
    )
    fabricated.pop("raw_teacher_cache_authority")
    _rehash_glm52_checkpoint(fabricated)

    result = _evaluate_glm52_checkpoint(tmp_path, fabricated)

    assert result.checks["gate_contract_authenticated"] is True
    assert result.checks["audited_teacher_cache_identity_authenticated"] is False
    assert result.checks["blocked_contract_consistent"] is False


def test_glm52_family_gate_rejects_rehashed_v3_cache_identity_forgery(
    tmp_path: Path,
) -> None:
    forged = _glm52_blocked_checkpoint(
        schema_version=GLM52_FAMILY_GATE_V3_SCHEMA_VERSION
    )
    identity = forged["audited_teacher_cache_identity"]
    assert isinstance(identity, dict)
    identity["raw_tensor_bytes"] = 1
    forged["audited_teacher_cache_identity_sha256"] = canonical_sha256(identity)
    contract_hashes = forged["input_evidence_contract_sha256"]
    assert isinstance(contract_hashes, dict)
    contract_hashes["teacher_cache_audit"] = canonical_sha256(identity)
    _rehash_glm52_checkpoint(forged)

    result = _evaluate_glm52_checkpoint(tmp_path, forged)

    assert result.checks["gate_contract_authenticated"] is True
    assert result.checks["audited_teacher_cache_identity_authenticated"] is False
    assert result.checks["blocked_contract_consistent"] is False


def test_glm52_family_gate_rejects_rehashed_v2_promotion_forgery(
    tmp_path: Path,
) -> None:
    forged = _glm52_blocked_checkpoint(
        schema_version=GLM52_FAMILY_GATE_V2_SCHEMA_VERSION
    )
    forged["release_pass_enabled"] = True
    forged["raw_release_evidence_validators_ready"] = True
    forged["gate_status"] = "glm52_family_gate_passed"
    forged["family_gate_pass"] = True
    forged["checks"] = {
        name: True for name in GLM52_FAMILY_GATE_V1_CHECK_NAMES
    }
    forged["missing_requirements"] = []
    forged["raw_evidence_validator_readiness"] = {
        "artifact": True,
        "production": True,
        "family_eval": True,
        "family_benchmark": True,
    }
    _rehash_glm52_checkpoint(forged)

    result = _evaluate_glm52_checkpoint(tmp_path, forged)

    assert result.passed is False
    assert result.checks["gate_contract_authenticated"] is True
    assert result.checks["blocked_contract_consistent"] is False
    assert "blocked_contract_consistent" in result.reasons


def test_glm52_family_gate_rejects_rehashed_v2_identity_digest_drift(
    tmp_path: Path,
) -> None:
    forged = _glm52_blocked_checkpoint(
        schema_version=GLM52_FAMILY_GATE_V2_SCHEMA_VERSION
    )
    forged["common_artifact_identity_sha256"] = "0" * 64
    _rehash_glm52_checkpoint(forged)

    result = _evaluate_glm52_checkpoint(tmp_path, forged)

    assert result.passed is False
    assert result.checks["gate_contract_authenticated"] is True
    assert result.checks["common_artifact_identity_authenticated"] is False
    assert result.checks["blocked_contract_consistent"] is False


def test_glm52_family_gate_rejects_rehashed_v2_identity_body_drift(
    tmp_path: Path,
) -> None:
    forged = _glm52_blocked_checkpoint(
        schema_version=GLM52_FAMILY_GATE_V2_SCHEMA_VERSION
    )
    identity = forged["common_artifact_identity"]
    assert isinstance(identity, dict)
    identity["group_size"] = 256
    forged["common_artifact_identity_sha256"] = canonical_sha256(identity)
    _rehash_glm52_checkpoint(forged)

    result = _evaluate_glm52_checkpoint(tmp_path, forged)

    assert result.passed is False
    assert result.checks["gate_contract_authenticated"] is True
    assert result.checks["common_artifact_identity_authenticated"] is False
    assert result.checks["blocked_contract_consistent"] is False


@pytest.mark.parametrize(
    "field",
    ["input_evidence_file_sha256", "input_evidence_contract_sha256"],
)
def test_glm52_family_gate_rejects_rehashed_v2_hash_map_omission(
    tmp_path: Path,
    field: str,
) -> None:
    forged = _glm52_blocked_checkpoint(
        schema_version=GLM52_FAMILY_GATE_V2_SCHEMA_VERSION
    )
    forged.pop(field)
    _rehash_glm52_checkpoint(forged)

    result = _evaluate_glm52_checkpoint(tmp_path, forged)

    assert result.checks["gate_contract_authenticated"] is True
    assert result.checks["blocked_contract_consistent"] is False


@pytest.mark.parametrize(
    "field",
    ["input_evidence_file_sha256", "input_evidence_contract_sha256"],
)
def test_glm52_family_gate_rejects_rehashed_v2_incomplete_hash_inventory(
    tmp_path: Path,
    field: str,
) -> None:
    forged = _glm52_blocked_checkpoint(
        schema_version=GLM52_FAMILY_GATE_V2_SCHEMA_VERSION
    )
    hashes = forged[field]
    assert isinstance(hashes, dict)
    hashes.pop("family_policy")
    _rehash_glm52_checkpoint(forged)

    result = _evaluate_glm52_checkpoint(tmp_path, forged)

    assert result.checks["gate_contract_authenticated"] is True
    assert result.checks["blocked_contract_consistent"] is False


@pytest.mark.parametrize(
    "field",
    ["input_evidence_file_sha256", "input_evidence_contract_sha256"],
)
def test_glm52_family_gate_rejects_rehashed_v2_hash_digest_drift(
    tmp_path: Path,
    field: str,
) -> None:
    forged = _glm52_blocked_checkpoint(
        schema_version=GLM52_FAMILY_GATE_V2_SCHEMA_VERSION
    )
    hashes = forged[field]
    assert isinstance(hashes, dict)
    hashes["family_policy"] = "0" * 64
    _rehash_glm52_checkpoint(forged)

    result = _evaluate_glm52_checkpoint(tmp_path, forged)

    assert result.checks["gate_contract_authenticated"] is True
    assert result.checks["blocked_contract_consistent"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("profile", "glm52-reap-forged"),
        ("family_policy_contract_sha256", "0" * 64),
        ("prompt_pack_contract_sha256", "0" * 64),
        ("accepted_tensor_payload_bytes", 1),
        ("accepted_whole_main_bpw", 1.0),
        ("expected_routed_group_count", 224),
        ("present_routed_group_count", 224),
        ("missing_routed_group_count", 1),
        ("production_binding_proven", False),
        ("production_generation_proven", False),
        ("teacher_cache_payload_present", True),
        ("full_vocabulary_eval_proven", True),
        ("same_machine_benchmark_proven", True),
    ],
)
def test_glm52_family_gate_rejects_rehashed_v2_summary_contradiction(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    forged = _glm52_blocked_checkpoint(
        schema_version=GLM52_FAMILY_GATE_V2_SCHEMA_VERSION
    )
    forged[field] = value
    _rehash_glm52_checkpoint(forged)

    result = _evaluate_glm52_checkpoint(tmp_path, forged)

    assert result.checks["gate_contract_authenticated"] is True
    assert result.checks["blocked_contract_consistent"] is False


def test_glm52_family_gate_rechecks_authenticated_composite_evidence(
    tmp_path: Path,
) -> None:
    blocked: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_family_gate",
        "gate_status": "glm52_family_gate_blocked",
        "family_gate_pass": False,
        "model_id": "0xSero/glm-5.2-reap-504B-v2",
        "revision": "6c9241aa05fb243a0edb7c804c213ec1cf5c920d",
        "checks": {
            "policy_frozen": True,
            "full_225_group_artifact_ready": False,
        },
        "missing_requirements": ["full_225_group_artifact"],
    }
    blocked["gate_contract_sha256"] = canonical_sha256(blocked)

    result = _evaluate_glm52_checkpoint(tmp_path, blocked)

    assert result.passed is False
    assert "full_225_group_artifact_ready" in result.reasons
    assert result.checks["gate_contract_authenticated"] is True

    forged = dict(blocked)
    forged["gate_schema_version"] = 1
    forged["release_pass_enabled"] = False
    forged["raw_release_evidence_validators_ready"] = False
    forged["gate_status"] = "glm52_family_gate_passed"
    forged["family_gate_pass"] = True
    forged["checks"] = {
        name: True for name in GLM52_FAMILY_GATE_V1_CHECK_NAMES
    }
    forged["missing_requirements"] = []
    _rehash_glm52_checkpoint(forged)
    rejected_forgery = _evaluate_glm52_checkpoint(tmp_path, forged)
    assert rejected_forgery.passed is False
    assert "blocked_contract_consistent" in rejected_forgery.reasons

    forged["gate_contract_sha256"] = "0" * 64
    tampered = _evaluate_glm52_checkpoint(tmp_path, forged)
    assert tampered.passed is False
    assert "gate_contract_authenticated" in tampered.reasons
