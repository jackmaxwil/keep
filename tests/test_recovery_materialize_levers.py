from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

import mlx_vq.convert.glm52_recovery_materialize as materialize


def _write_json(path: Path, value: object) -> str:
    payload = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _synthetic_recovery_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, str, str, Path]:
    filename = "layer-00003-gate_proj.safetensors"
    key = (3, "gate_proj")
    monkeypatch.setattr(
        materialize,
        "_canonical_seed_groups",
        lambda: ((key[0], key[1], filename),),
    )
    root = tmp_path / "parent-recovery"
    recovered = root / "recovered-groups"
    artifact = root / "artifact"
    recovered.mkdir(parents=True)
    artifact.mkdir()
    group_path = recovered / filename
    group_path.write_bytes(b"authenticated recovery seed group")
    os.symlink(os.path.relpath(group_path, start=artifact), artifact / filename)
    group_sha = hashlib.sha256(group_path.read_bytes()).hexdigest()
    source_lineage = {
        "source_model_id": materialize.GLM52_REAP_MODEL_ID,
        "source_revision": materialize.GLM52_REAP_REVISION,
        "source_config_sha256": materialize.GLM52_REAP_CONFIG_SHA256,
        "source_index_sha256": materialize.GLM52_REAP_INDEX_SHA256,
        "source_profile": materialize.GLM52_REAP_PROFILE,
    }
    policy = {
        "recovery_lever": materialize.RECOVERY_LEVER,
        "rounding_objective": "selection_diagonal_hessian_weighted_squared_error",
        "scale_estimator": "importance_weighted_least_squares",
    }
    group_record = {
        "layer": 3,
        "projection": "gate_proj",
        "status": "materialized",
        "artifact_path": filename,
        "artifact_bytes": group_path.stat().st_size,
        "artifact_sha256": group_sha,
        "source_lineage": source_lineage,
        "recovery_policy": policy,
        "lever_provenance": {"lever": materialize.RECOVERY_LEVER},
        "code_bits": 8,
        "codes_dtype": "uint8",
        "codebook_name": "quip_e8",
        "codebook_sha256": materialize.codebook_metadata_for_bits(8)[1],
        "tensor_payload_bytes": 1,
        "zero_importance_policy": "preserve_seed_expert_bytes",
    }
    manifest = {
        "schema_version": 2,
        "record_type": "glm52_recovery_conversion_manifest",
        "status": "complete",
        "resumable": True,
        "selected_groups": ["3:gate_proj"],
        "rate_policy": {
            "default_code_bits": 8,
            "layer_code_bits": {},
            "complete_layer_rates_required": True,
        },
        "source_lineage": source_lineage,
        "source_verification": {
            "source_blob_inventory_sha256": "1" * 64,
            "routed_source_blob_inventory_sha256": "2" * 64,
            "shard_count": 1,
        },
        "expected_full_source_blob_inventory_sha256": "1" * 64,
        "expected_routed_source_blob_inventory_sha256": "2" * 64,
        "stats_manifest_sha256": "3" * 64,
        "accepted_composite_audit_sha256": "4" * 64,
        "recovery_policy": policy,
        "lever_provenance": {"lever": materialize.RECOVERY_LEVER},
        "groups": [group_record],
        "mixed_artifact": {
            "seed_artifact_dir": str(tmp_path / "accepted-seed"),
            "output_dir": str(artifact),
            "recovered_groups_dir": str(recovered),
            "replacement_count": 1,
            "inherited_group_count": 0,
            "seed_manifest_sha256": materialize.PINNED_ACCEPTED_SEED_MANIFEST_SHA256,
        },
        "accounting": {
            "routed_codes_scales_bytes": 1,
            "routed_codebook_bytes": 0,
            "main_non_routed_tensor_payload_bytes": 1,
            "logical_whole_model_tensor_payload_bytes": 2,
            "incremental_disk_bytes": group_path.stat().st_size,
            "logical_payload_limit_bytes": materialize.GLM52_LOGICAL_PAYLOAD_LIMIT_BYTES,
        },
    }
    manifest["manifest_body_sha256"] = materialize._canonical_sha256(manifest)
    manifest_sha = _write_json(root / "conversion-manifest.json", manifest)
    audit_group = {
        "group_key": "3:gate_proj",
        "filename": filename,
        "classification": "replacement",
        "artifact_sha256": group_sha,
        "artifact_bytes": group_path.stat().st_size,
        "tensor_payload_bytes": 1,
        "code_bits": 8,
        "codes_dtype": "uint8",
        "group_size": 512,
        "codebook_name": "quip_e8",
        "codebook_sha256": materialize.codebook_metadata_for_bits(8)[1],
    }
    audit = {
        "recovery_dir": str(root),
        "manifest_path": str(root / "conversion-manifest.json"),
        "manifest_body_sha256": manifest["manifest_body_sha256"],
        "seed_manifest_sha256": materialize.PINNED_ACCEPTED_SEED_MANIFEST_SHA256,
        "accepted_composite_audit_sha256": "4" * 64,
        "accepted_baseline_composite_identity_sha256": (
            materialize.PINNED_ACCEPTED_COMPOSITE_IDENTITY_SHA256
        ),
        "group_count": 1,
        "replacement_group_count": 1,
        "inherited_group_count": 0,
        "complete_replacement_layer_ids": [],
        "routed_codes_scales_bytes": 1,
        "routed_codebook_bytes": 0,
        "routed_tensor_payload_bytes": 1,
        "main_non_routed_tensor_payload_bytes": 1,
        "non_routed_tensor_payload_bytes": 1,
        "logical_whole_model_tensor_payload_bytes": 2,
        "logical_payload_limit_bytes": materialize.GLM52_LOGICAL_PAYLOAD_LIMIT_BYTES,
        "incremental_disk_bytes": group_path.stat().st_size,
        "candidate_identity_sha256": "pending",
        "groups": [audit_group],
        "checks": {"identity": True},
        "audit_pass": True,
        "raw_recovery_manifest_sha256": manifest_sha,
    }
    audit["candidate_identity_sha256"] = materialize._canonical_sha256(
        {
            "schema_version": 2,
            "source_lineage": source_lineage,
            "seed_manifest_sha256": audit["seed_manifest_sha256"],
            "accepted_baseline_composite_identity_sha256": audit[
                "accepted_baseline_composite_identity_sha256"
            ],
            "accepted_composite_audit_sha256": audit[
                "accepted_composite_audit_sha256"
            ],
            "groups": [
                {
                    "group_key": "3:gate_proj",
                    "classification": "replacement",
                    "artifact_sha256": group_sha,
                    "tensor_payload_bytes": 1,
                    "resolved_policy": {
                        "quant_method": "mlx_vq_e8",
                        "version": 1,
                        "code_bits": 8,
                        "codes_dtype": "uint8",
                        "group_size": 512,
                        "codebook_name": "quip_e8",
                        "codebook_sha256": materialize.codebook_metadata_for_bits(8)[1],
                        "tensor_payload_bytes": 1,
                        "recovery_provenance": {"lever": materialize.RECOVERY_LEVER},
                    },
                    "code_bits": 8,
                    "codes_dtype": "uint8",
                    "codebook_name": "quip_e8",
                    "codebook_sha256": materialize.codebook_metadata_for_bits(8)[1],
                }
            ],
            "logical_whole_model_tensor_payload_bytes": 2,
            "rate_policy": manifest["rate_policy"],
            "routed_codes_scales_bytes": 1,
            "routed_codebook_bytes": 0,
            "main_non_routed_tensor_payload_bytes": 1,
        }
    )
    audit_path = tmp_path / "parent-recovery-audit.json"
    audit_sha = _write_json(audit_path, audit)
    return root, audit_path, manifest_sha, audit_sha, group_path


def test_recovery_seed_authenticates_manifest_audit_and_resolved_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, audit_path, manifest_sha, audit_sha, group_path = _synthetic_recovery_seed(
        tmp_path, monkeypatch
    )

    authenticated = materialize._authenticate_recovery_seed(
        recovery_root=root,
        audit_path=audit_path,
        expected_manifest_sha256=manifest_sha,
        expected_audit_sha256=audit_sha,
    )

    assert authenticated.kind == "audited_recovery"
    assert authenticated.artifact_root == root / "artifact"
    assert authenticated.records[(3, "gate_proj")]["resolved_path"] == str(group_path)
    assert authenticated.layer_code_bits == {3: 8}
    assert authenticated.candidate_identity_sha256 == json.loads(
        audit_path.read_text()
    )["candidate_identity_sha256"]


def test_recovery_seed_rejects_audit_group_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, audit_path, manifest_sha, _audit_sha, _group_path = _synthetic_recovery_seed(
        tmp_path, monkeypatch
    )
    audit = json.loads(audit_path.read_text())
    audit["groups"][0]["artifact_sha256"] = "0" * 64
    audit_sha = _write_json(audit_path, audit)

    with pytest.raises(ValueError, match="group.*SHA-256"):
        materialize._authenticate_recovery_seed(
            recovery_root=root,
            audit_path=audit_path,
            expected_manifest_sha256=manifest_sha,
            expected_audit_sha256=audit_sha,
        )


def test_materializer_parser_rejects_mixed_seed_flag_families() -> None:
    parser = materialize._build_parser()
    common = [
        "--source-dir", "source", "--index-path", "index", "--stats-dir", "stats",
        "--output-dir", "out", "--groups", "3:gate_proj",
        "--expected-stats-manifest-sha256", "1" * 64,
        "--expected-full-source-blob-inventory-sha256", "2" * 64,
        "--expected-routed-source-blob-inventory-sha256", "3" * 64,
        "--accepted-composite-audit-json", "composite.json",
        "--expected-composite-audit-sha256", "4" * 64,
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                *common,
                "--seed-artifact-dir", "classic",
                "--expected-seed-manifest-sha256", "5" * 64,
                "--seed-recovery-artifact-dir", "recovery",
                "--seed-recovery-audit-json", "audit.json",
                "--expected-seed-recovery-manifest-sha256", "6" * 64,
                "--expected-seed-recovery-audit-sha256", "7" * 64,
            ]
        )


def test_default_policy_is_byte_identical_to_legacy_quantization() -> None:
    rng = np.random.default_rng(20260711)
    weight = rng.standard_normal((4, 16)).astype(np.float32)
    diagonal = (np.abs(rng.standard_normal(16)) + 0.1).astype(np.float32)

    legacy = materialize.quantize_weight_importance_aware(
        weight, diagonal, group_size=8, e8p_search_backend="numpy"
    )
    explicit_default = materialize.quantize_weight_importance_aware(
        weight,
        diagonal,
        group_size=8,
        e8p_search_backend="numpy",
        recovery_policy=materialize.RecoveryPolicy(),
    )

    np.testing.assert_array_equal(explicit_default.codes, legacy.codes)
    np.testing.assert_array_equal(explicit_default.scales, legacy.scales)
    assert explicit_default.stats == legacy.stats == {}


def test_ebss_search_initializes_existing_refinement_and_records_choices() -> None:
    rng = np.random.default_rng(17)
    weight = rng.standard_normal((3, 16)).astype(np.float32)
    diagonal = (np.abs(rng.standard_normal(16)) + 0.1).astype(np.float32)
    policy = materialize.RecoveryPolicy(
        scale_search_multipliers="0.85,1.0,1.1"
    )

    quantized = materialize.quantize_weight_importance_aware(
        weight,
        diagonal,
        group_size=8,
        code_bits=8,
        iterations=3,
        e8p_search_backend="numpy",
        recovery_policy=policy,
    )

    assert quantized.stats["multiplier_policy"] == "0.85,1.0,1.1"
    assert quantized.stats["chosen_multipliers"].shape == (3, 2)
    assert materialize._recovery_policy_declarations(policy) == {
        "scale_estimator": "selection_diagonal_hessian_ebss_v1",
        "recovery_lever": materialize.RECOVERY_LEVER,
        "rounding_objective": "selection_diagonal_hessian_weighted_squared_error",
        "scale_search_multipliers": "0.85,1.0,1.1",
        "scale_search_objective": "selection_diagonal_hessian_weighted_squared_error",
        "scale_refinement_iterations": "3",
    }


def test_ebss_policy_rejects_noncanonical_multiplier_string() -> None:
    with pytest.raises(ValueError, match="canonical"):
        materialize.RecoveryPolicy(scale_search_multipliers="1.00,0.85")


def test_ldlq_lever_requires_full_hessian_and_uses_one_sweep() -> None:
    rng = np.random.default_rng(19)
    weight = (rng.standard_normal((2, 8)) * 0.2).astype(np.float32)
    diagonal = np.linspace(0.5, 1.2, 8, dtype=np.float32)
    h_in = np.diag(diagonal).astype(np.float64)
    h_in[0, 1] = h_in[1, 0] = 0.05
    policy = materialize.RecoveryPolicy(
        recovery_lever="selection_hin_blockldlq_feedback_fp32_v1"
    )

    with pytest.raises(ValueError, match="full h_in"):
        materialize.quantize_weight_importance_aware(
            weight,
            diagonal,
            group_size=8,
            code_bits=16,
            e8p_search_backend="numpy",
            recovery_policy=policy,
        )

    quantized = materialize.quantize_weight_importance_aware(
        weight,
        diagonal,
        group_size=8,
        code_bits=16,
        e8p_search_backend="numpy",
        recovery_policy=policy,
        h_in=h_in,
    )
    assert quantized.stats["ldlq"]["sweeps"] == 1
    assert quantized.stats["ldlq"]["code_bits"] == 16
    assert materialize._recovery_policy_declarations(policy)["recovery_lever"] == (
        "selection_hin_blockldlq_feedback_fp32_v1"
    )


def _write_seed_group(path: Path) -> None:
    prefix = "model.layers.3.mlp.switch_mlp.gate_proj"
    quantization = {
        "default_code_bits": 8,
        "default_group_size": 8,
        "codebook_name": "quip_e8",
        "codebook_sha256": materialize.codebook_metadata_for_bits(8)[1],
        "policy": {"decoded_expert_working_set": "one_per_worker"},
    }
    save_file(
        {
            f"{prefix}.codes": np.zeros((1, 2, 1), dtype=np.uint8),
            f"{prefix}.scales": np.ones((1, 2, 1), dtype=np.float16),
            "model.vq_codebook.e8": materialize.e8_1bit_packed(),
        },
        path,
        metadata={"quantization_config": json.dumps(quantization, sort_keys=True)},
    )


def test_rotation_acceptance_writes_loader_schema_rht_signs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = tmp_path / "seed.safetensors"
    output = tmp_path / "rotated.safetensors"
    _write_seed_group(seed)
    signs = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.int8)

    def accepted(*_args, **_kwargs):
        return materialize._rotation_module().ProjectionRotationSelection(
            True, signs, 2.0, 1.0, 1.0
        )

    monkeypatch.setattr(materialize, "select_projection_rotation", accepted)
    materialize.materialize_recovery_group(
        source_weights=np.ones((1, 2, 8), dtype=np.float32),
        importance=np.ones((1, 8), dtype=np.float32),
        seed_group_path=seed,
        output_path=output,
        layer=3,
        projection="gate_proj",
        group_size=8,
        recovery_policy=materialize.RecoveryPolicy(rotation_rht_seed="7"),
        e8p_search_backend="numpy",
    )

    with safe_open(output, framework="np") as handle:
        name = "model.layers.3.mlp.switch_mlp.gate_proj.rht_signs"
        assert set(handle.keys()) == {
            "model.layers.3.mlp.switch_mlp.gate_proj.codes",
            "model.layers.3.mlp.switch_mlp.gate_proj.scales",
            name,
            "model.vq_codebook.e8",
        }
        np.testing.assert_array_equal(handle.get_tensor(name), signs)


def test_rejected_rotation_is_byte_identical_to_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = tmp_path / "seed.safetensors"
    off = tmp_path / "off.safetensors"
    rejected = tmp_path / "rejected.safetensors"
    _write_seed_group(seed)
    kwargs = dict(
        source_weights=np.ones((1, 2, 8), dtype=np.float32),
        importance=np.ones((1, 8), dtype=np.float32),
        seed_group_path=seed,
        layer=3,
        projection="gate_proj",
        group_size=8,
        e8p_search_backend="numpy",
    )
    materialize.materialize_recovery_group(output_path=off, **kwargs)

    def not_selected(*_args, **_kwargs):
        return materialize._rotation_module().ProjectionRotationSelection(
            False, None, 1.0, 1.0, 0.0
        )

    monkeypatch.setattr(materialize, "select_projection_rotation", not_selected)
    materialize.materialize_recovery_group(
        output_path=rejected,
        recovery_policy=materialize.RecoveryPolicy(rotation_rht_seed="7"),
        **kwargs,
    )
    with safe_open(off, framework="np") as off_handle, safe_open(
        rejected, framework="np"
    ) as rejected_handle:
        assert dict(rejected_handle.metadata() or {}) == dict(off_handle.metadata() or {})
        assert tuple(rejected_handle.keys()) == tuple(off_handle.keys())
        for name in off_handle.keys():
            np.testing.assert_array_equal(
                rejected_handle.get_tensor(name), off_handle.get_tensor(name)
            )
