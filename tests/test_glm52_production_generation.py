from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import weakref
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


MODEL_ID = "0xSero/glm-5.2-reap-504B-v2"
REVISION = "6c9241aa05fb243a0edb7c804c213ec1cf5c920d"
GROUP_KEYS = tuple(
    f"{layer}:{projection}"
    for layer in range(3, 78)
    for projection in ("gate_proj", "up_proj", "down_proj")
)
ACCEPTED_BYTES = 98_433_923_808
ACCEPTED_PARAMETERS = 494_194_805_304
ACCEPTED_BPW = ACCEPTED_BYTES * 8 / ACCEPTED_PARAMETERS


def _load_probe_module():
    path = Path("benchmarks/probe_glm52_production_generation.py")
    assert path.is_file(), "production generation probe is not implemented"
    spec = importlib.util.spec_from_file_location(
        "probe_glm52_production_generation_under_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_composite_loader_module():
    module_name = "glm52_composite_loader_recovery_under_test"
    dependencies: dict[str, ModuleType] = {}

    def dependency(name: str, **values: object) -> None:
        fake = ModuleType(name)
        for key, value in values.items():
            setattr(fake, key, value)
        dependencies[name] = fake

    class FakeSafetensorsIndex:
        def __init__(self, *, metadata=None, weight_map=None):
            self.metadata = metadata or {}
            self.weight_map = weight_map or {}

    class FakeModelProfile:
        pass

    class FakeBindReport:
        pass

    dependency(
        "mlx_vq.convert.glm52_non_vq",
        audit_glm52_non_vq_package=lambda *_args, **_kwargs: None,
    )
    dependency(
        "mlx_vq.convert.glm52_reap",
        GLM52_REAP_CONFIG_SHA256="1" * 64,
        GLM52_REAP_EXPECTED_GROUPS=225,
        GLM52_REAP_EXPECTED_LAYER_IDS=tuple(range(3, 78)),
        GLM52_REAP_INDEX_SHA256="2" * 64,
        GLM52_REAP_PROFILE_NAME="glm52-reap-504b-v2",
    )
    dependency(
        "mlx_vq.convert.stream_convert",
        SafetensorsIndex=FakeSafetensorsIndex,
        load_safetensors_index=lambda _path: FakeSafetensorsIndex(),
    )
    dependency(
        "mlx_vq.models.glm52_vq_adapter",
        GLM52NonVQBindReport=FakeBindReport,
        GLM52VQModel=object,
        bind_glm52_non_vq_weights=lambda *_args, **_kwargs: None,
        bind_glm52_vq_experts=lambda *_args, **_kwargs: (),
        bind_glm52_vq_experts_from_paths=lambda *_args, **_kwargs: (),
        dense_glm52_routed_parameter_names=lambda _model: (),
        glm52_vq_args_from_config=lambda config: config,
        has_unbound_vq_experts=lambda _model: False,
    )
    dependency(
        "mlx_vq.models.profiles",
        ModelProfile=FakeModelProfile,
        validate_profile_against_hf_config_data=lambda *_args, **_kwargs: [],
    )
    dependency(
        "mlx_vq.quality.glm52_family",
        GLM52_PINNED_TOKENIZER_FILES=(),
        PINNED_GLM52_MODEL_ID=MODEL_ID,
        PINNED_GLM52_REVISION=REVISION,
        validate_glm52_family_gate_policy=lambda payload: payload,
        validate_glm52_tokenizer_readiness=lambda *_args, **_kwargs: None,
    )
    dependency(
        "mlx_vq.validate.glm52_artifact",
        audit_glm52_reap_materialization_manifest=lambda *_args, **_kwargs: None,
        audit_glm52_reap_source_accounting=lambda *_args, **_kwargs: None,
    )
    dependency(
        "mlx_vq.validate.glm52_recovery_artifact",
        GLM52RecoveryArtifactAudit=object,
        audit_glm52_recovery_mixed_artifact=lambda *_args, **_kwargs: None,
    )
    dependency(
        "mlx_vq.validate.glm52_runtime",
        expected_glm52_routed_group_keys=lambda *_args, **_kwargs: GROUP_KEYS,
        preflight_glm52_full_bind=lambda *_args, **_kwargs: None,
    )
    original = {name: sys.modules.get(name) for name in dependencies}
    sys.modules.update(dependencies)
    path = Path("src/mlx_vq/models/glm52_composite_loader.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
        for name, previous in original.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


@dataclass
class _FakeRecoveryAudit:
    recovery_dir: str
    manifest_path: str
    candidate_identity_sha256: str = "a" * 64
    manifest_body_sha256: str = "b" * 64
    accepted_composite_audit_sha256: str = "c" * 64
    accepted_baseline_composite_identity_sha256: str = "d" * 64
    audit_pass: bool = True
    fail_on_verify_call: int | None = None
    verify_current_identity_calls: int = 0
    groups: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        if self.groups:
            return
        artifact = (
            Path(self.recovery_dir)
            / "artifact/layer-00003-gate_proj.safetensors"
        )
        self.groups = (
            SimpleNamespace(
                filename=artifact.name,
                artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                artifact_bytes=artifact.stat().st_size,
            ),
        )

    def verify_current_identity(self) -> None:
        self.verify_current_identity_calls += 1
        if self.verify_current_identity_calls == self.fail_on_verify_call:
            raise ValueError("recovery artifact current identity changed after audit")


def _recovery_baseline() -> SimpleNamespace:
    return SimpleNamespace(
        profile=object(),
        config={},
        non_vq_index=object(),
        tokenizer_dir=Path("/tokenizer"),
        non_vq_artifact_dir=Path("/non-vq"),
        routed_artifact_dir=Path("/accepted-routed"),
        artifact_identity=SimpleNamespace(sha256="d" * 64),
        input_fingerprint=object(),
    )


def _recovery_authority() -> dict[str, object]:
    return {
        "expected_seed_manifest_sha256": "1" * 64,
        "expected_stats_manifest_sha256": "2" * 64,
        "expected_full_source_blob_inventory_sha256": "3" * 64,
        "expected_routed_source_blob_inventory_sha256": "4" * 64,
        "expected_recovery_lever": "importance_hessian_rounding",
        "expected_recovery_policy": {
            "recovery_lever": "importance_hessian_rounding",
        },
        "accepted_composite_audit_json": "/accepted-composite.json",
        "expected_composite_audit_sha256": "c" * 64,
        "expected_recovery_candidate_identity_sha256": "a" * 64,
        "expected_recovery_manifest_body_sha256": "b" * 64,
    }


def _recovery_tree(tmp_path: Path) -> Path:
    recovery_dir = tmp_path / "recovery"
    recovery_dir.mkdir(parents=True)
    (recovery_dir / "artifact").mkdir()
    (recovery_dir / "groups").mkdir()
    group = recovery_dir / "groups/layer-00003-gate_proj.safetensors"
    group.write_bytes(b"audited recovery group")
    (recovery_dir / "artifact/layer-00003-gate_proj.safetensors").symlink_to(
        Path("../groups") / group.name
    )
    (recovery_dir / "conversion-manifest.json").write_text("{}\n")
    return recovery_dir


def _composite_claims(
    *,
    routed_root: Path,
    non_vq_root: Path,
    non_vq_evidence_sha256: str = "a" * 64,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "glm52_modelopt_nvfp4_artifact_audit",
        "audit_status": "glm52_modelopt_nvfp4_artifact_audit_ready",
        "audit_pass": True,
        "artifact_integrity_pass": True,
        "audit_blockers": [],
        "materialization_scope": "full",
        "artifact_dir": str(routed_root),
        "manifest_path": str(routed_root / "conversion-manifest.json"),
        "non_vq_artifact_dir": str(non_vq_root),
        "non_vq_evidence_json": str(non_vq_root.parent / "non-vq-evidence.json"),
        "source_dir": str(non_vq_root.parent / "snapshot"),
        "model_id": MODEL_ID,
        "source_revision": REVISION,
        "config_sha256": "b" * 64,
        "index_sha256": "c" * 64,
        "manifest_sha256": "d" * 64,
        "group_set_sha256": "e" * 64,
        "expected_group_keys": list(GROUP_KEYS),
        "ready_group_keys": list(GROUP_KEYS),
        "complete_layer_ids": list(range(3, 78)),
        "partial_layer_ids": [],
        "full_routed_artifact_ready": True,
        "dense_routed_experts": False,
        "resume_verified": True,
        "byte_identity_verified": True,
        "requested_code_bits": 8,
        "requested_group_size": 512,
        "requested_scale_estimator": "max_abs",
        "accounting_scope": "full_composite_actual",
        "actual_routed_weight_count": 475_634_073_600,
        "actual_routed_payload_bytes": 61_312_204_800,
        "actual_routed_codebook_bytes": 230_400,
        "main_non_routed_parameter_count": 18_560_731_704,
        "main_non_routed_tensor_payload_bytes": 37_121_488_608,
        "main_model_parameter_count_excluding_mtp": ACCEPTED_PARAMETERS,
        "actual_whole_model_tensor_payload_bytes": ACCEPTED_BYTES,
        "actual_whole_model_tensor_payload_bpw": ACCEPTED_BPW,
        "whole_model_tensor_payload_values_actual": True,
        "whole_model_physical_values_actual": True,
        "non_vq_package_evidence_authenticated": True,
        "non_vq_evidence_sha256": non_vq_evidence_sha256,
        "non_vq_package_audit": {
            "audit_pass": True,
            "artifact_dir": str(non_vq_root),
            "manifest_path": str(non_vq_root / "non-vq-manifest.json"),
            "manifest_sha256": "f" * 64,
            "package_set_sha256": "1" * 64,
            "parameter_count": 18_560_731_704,
            "tensor_payload_bytes": 37_121_488_608,
        },
    }


def _full_bind_claims(*, routed_root: Path, non_vq_root: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "glm52_full_bind_preflight",
        "preflight_status": "glm52_full_bind_preflight_ready",
        "preflight_pass": True,
        "blockers": [],
        "profile": "glm52-reap-504b-v2",
        "profile_sha256": "2" * 64,
        "profile_contract_sha256": "3" * 64,
        "model_id": MODEL_ID,
        "source_revision": REVISION,
        "config_sha256": "b" * 64,
        "source_index_sha256": "c" * 64,
        "non_vq_artifact_dir": str(non_vq_root),
        "routed_artifact_dir": str(routed_root),
        "expected_routed_group_keys": list(GROUP_KEYS),
        "present_routed_group_keys": list(GROUP_KEYS),
        "missing_routed_group_keys": [],
        "expected_routed_group_count": 225,
        "present_routed_group_count": 225,
        "missing_routed_group_count": 0,
        "non_vq_tensor_count": 1_194,
        "non_vq_runtime_target_count": 1_272,
        "non_vq_header_file_count": 9,
        "kv_b_source_tensor_count": 78,
        "kv_b_runtime_target_count": 156,
        "routed_header_file_count": 225,
        "header_only": True,
        "tensor_payloads_read": False,
        "payload_hashes_verified": False,
        "full_model_constructed": False,
        "dense_routed_experts": False,
        "production_binding_proven": False,
        "production_generation_proven": False,
    }


def _non_vq_evidence(*, non_vq_root: Path) -> dict[str, object]:
    package_audit = {
        "artifact_dir": str(non_vq_root),
        "manifest_path": str(non_vq_root / "non-vq-manifest.json"),
        "manifest_sha256": "f" * 64,
        "package_set_sha256": "1" * 64,
        "parameter_count": 18_560_731_704,
        "tensor_payload_bytes": 37_121_488_608,
        "artifact_tree_bytes": 37_122_403_011,
        "audit_pass": True,
        "checks": {"payload_hashes": True},
    }
    return {
        "schema_version": 1,
        "record_type": "glm52_non_vq_package_manifest",
        "pack_status": "glm52_non_vq_package_ready",
        "production_ready": True,
        "package_audit_pass": True,
        "profile": "glm52-reap-504b-v2",
        "model_id": MODEL_ID,
        "source_revision": REVISION,
        "config_sha256": "b" * 64,
        "index_sha256": "c" * 64,
        "manifest_sha256": "f" * 64,
        "package_index_sha256": "6" * 64,
        "package_set_sha256": "1" * 64,
        "source_blob_inventory_sha256": "4" * 64,
        "source_inventory_sha256": "5" * 64,
        "parameter_count": 18_560_731_704,
        "tensor_payload_bytes": 37_121_488_608,
        "package_audit": package_audit,
    }


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("materialization_scope", "bounded", "full"),
        ("full_routed_artifact_ready", False, "full routed"),
        ("whole_model_tensor_payload_values_actual", False, "actual"),
        ("non_vq_package_evidence_authenticated", False, "non-VQ"),
        ("dense_routed_experts", True, "dense routed"),
        ("resume_verified", False, "resume"),
    ],
)
def test_composite_claims_fail_closed_before_model_construction(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    module = _load_probe_module()
    routed = tmp_path / "routed"
    non_vq = tmp_path / "non-vq"
    routed.mkdir()
    non_vq.mkdir()
    claims = _composite_claims(routed_root=routed, non_vq_root=non_vq)
    claims[field] = value

    with pytest.raises(ValueError, match=match):
        module.validate_glm52_composite_claims(
            claims,
            expected_group_keys=GROUP_KEYS,
            routed_artifact_dir=routed,
            non_vq_artifact_dir=non_vq,
            source_dir=tmp_path / "snapshot",
            non_vq_evidence_json=tmp_path / "non-vq-evidence.json",
            non_vq_evidence_sha256="a" * 64,
        )


def test_composite_claims_reject_stale_manifest_paths(tmp_path: Path) -> None:
    module = _load_probe_module()
    routed = tmp_path / "routed"
    non_vq = tmp_path / "non-vq"
    routed.mkdir()
    non_vq.mkdir()
    claims = _composite_claims(routed_root=routed, non_vq_root=non_vq)
    claims["manifest_path"] = str(tmp_path / "stale-routed.json")
    with pytest.raises(ValueError, match="routed manifest path"):
        module.validate_glm52_composite_claims(
            claims,
            expected_group_keys=GROUP_KEYS,
            routed_artifact_dir=routed,
            non_vq_artifact_dir=non_vq,
            source_dir=tmp_path / "snapshot",
            non_vq_evidence_json=tmp_path / "non-vq-evidence.json",
            non_vq_evidence_sha256="a" * 64,
        )

    claims = _composite_claims(routed_root=routed, non_vq_root=non_vq)
    claims["non_vq_package_audit"] = dict(claims["non_vq_package_audit"])
    claims["non_vq_package_audit"]["manifest_path"] = str(
        tmp_path / "stale-non-vq.json"
    )
    with pytest.raises(ValueError, match="non-VQ manifest path"):
        module.validate_glm52_composite_claims(
            claims,
            expected_group_keys=GROUP_KEYS,
            routed_artifact_dir=routed,
            non_vq_artifact_dir=non_vq,
            source_dir=tmp_path / "snapshot",
            non_vq_evidence_json=tmp_path / "non-vq-evidence.json",
            non_vq_evidence_sha256="a" * 64,
        )

    claims = _composite_claims(routed_root=routed, non_vq_root=non_vq)
    claims["non_vq_evidence_json"] = str(tmp_path / "stale-evidence.json")
    with pytest.raises(ValueError, match="non-VQ evidence path"):
        module.validate_glm52_composite_claims(
            claims,
            expected_group_keys=GROUP_KEYS,
            routed_artifact_dir=routed,
            non_vq_artifact_dir=non_vq,
            source_dir=tmp_path / "snapshot",
            non_vq_evidence_json=tmp_path / "non-vq-evidence.json",
            non_vq_evidence_sha256="a" * 64,
        )

    claims = _composite_claims(routed_root=routed, non_vq_root=non_vq)
    claims["source_dir"] = str(tmp_path / "stale-snapshot")
    with pytest.raises(ValueError, match="source path"):
        module.validate_glm52_composite_claims(
            claims,
            expected_group_keys=GROUP_KEYS,
            routed_artifact_dir=routed,
            non_vq_artifact_dir=non_vq,
            source_dir=tmp_path / "snapshot",
            non_vq_evidence_json=tmp_path / "non-vq-evidence.json",
            non_vq_evidence_sha256="a" * 64,
        )


def test_complete_header_only_preflight_is_supplemental_not_production_proof(
    tmp_path: Path,
) -> None:
    module = _load_probe_module()
    routed = tmp_path / "routed"
    non_vq = tmp_path / "non-vq"
    routed.mkdir()
    non_vq.mkdir()
    claims = _full_bind_claims(routed_root=routed, non_vq_root=non_vq)

    validated = module.validate_glm52_full_bind_claims(
        claims,
        expected_group_keys=GROUP_KEYS,
        routed_artifact_dir=routed,
        non_vq_artifact_dir=non_vq,
    )

    assert validated["header_only"] is True
    assert validated["production_binding_proven"] is False
    claims["present_routed_group_count"] = 224
    with pytest.raises(ValueError, match="225"):
        module.validate_glm52_full_bind_claims(
            claims,
            expected_group_keys=GROUP_KEYS,
            routed_artifact_dir=routed,
            non_vq_artifact_dir=non_vq,
        )

    claims = _full_bind_claims(routed_root=routed, non_vq_root=non_vq)
    claims["profile_contract_sha256"] = None
    with pytest.raises(ValueError, match="profile_contract_sha256"):
        module.validate_glm52_full_bind_claims(
            claims,
            expected_group_keys=GROUP_KEYS,
            routed_artifact_dir=routed,
            non_vq_artifact_dir=non_vq,
        )


def test_composite_claims_reject_integer_booleans_and_nearby_bpw(
    tmp_path: Path,
) -> None:
    module = _load_probe_module()
    routed = tmp_path / "routed"
    non_vq = tmp_path / "non-vq"
    routed.mkdir()
    non_vq.mkdir()
    claims = _composite_claims(routed_root=routed, non_vq_root=non_vq)
    claims["audit_pass"] = 1
    with pytest.raises(ValueError, match="audit_pass"):
        module.validate_glm52_composite_claims(
            claims,
            expected_group_keys=GROUP_KEYS,
            routed_artifact_dir=routed,
            non_vq_artifact_dir=non_vq,
            source_dir=tmp_path / "snapshot",
            non_vq_evidence_json=tmp_path / "non-vq-evidence.json",
            non_vq_evidence_sha256="a" * 64,
        )

    claims = _composite_claims(routed_root=routed, non_vq_root=non_vq)
    claims["actual_whole_model_tensor_payload_bpw"] = math.nextafter(
        ACCEPTED_BPW,
        math.inf,
    )
    with pytest.raises(ValueError, match="bpw"):
        module.validate_glm52_composite_claims(
            claims,
            expected_group_keys=GROUP_KEYS,
            routed_artifact_dir=routed,
            non_vq_artifact_dir=non_vq,
            source_dir=tmp_path / "snapshot",
            non_vq_evidence_json=tmp_path / "non-vq-evidence.json",
            non_vq_evidence_sha256="a" * 64,
        )


def test_production_input_orchestrator_rejects_bounded_before_fresh_auditors(
    tmp_path: Path,
) -> None:
    module = _load_probe_module()
    tokenizer = tmp_path / "snapshot"
    non_vq = tmp_path / "non-vq"
    routed = tmp_path / "routed"
    for root in (tokenizer, non_vq, routed):
        root.mkdir()
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        json.dumps(
            {
                "name": "glm52-reap-504b-v2",
                "hf_model_id": MODEL_ID,
                "revision": REVISION,
                "architecture": "glm_moe_dsa",
                "num_layers": 78,
                "num_sparse_layers": 75,
                "first_sparse_layer": 3,
                "hidden_size": 6144,
                "moe_intermediate_size": 2048,
                "num_experts": 168,
                "experts_per_tok": 8,
                "vocab_size": 154880,
                "shared_experts": 1,
                "converter": "glm52_vq_groups",
                "fused_gate_up": False,
                "default_code_bits": 8,
                "group_size_policy": {"gate": 512, "up": 512, "down": 512},
                "default_engine": "vq_e1_routed_nax_e8",
            }
        )
    )
    config_path = tokenizer / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "num_hidden_layers": 78,
                "hidden_size": 6144,
                "moe_intermediate_size": 2048,
                "n_routed_experts": 168,
                "num_experts_per_tok": 8,
                "vocab_size": 154880,
                "n_shared_experts": 1,
                "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 75,
            }
        )
    )
    index_path = tokenizer / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"metadata": {}, "weight_map": {}}))
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        (tokenizer / filename).write_text("{}\n")
    evidence_paths = {
        "tokenizer_readiness_json": tmp_path / "tokenizer-ready.json",
        "family_policy_json": tmp_path / "policy.json",
        "non_vq_evidence_json": tmp_path / "non-vq.json",
        "composite_audit_json": tmp_path / "composite.json",
        "materialization_runs_jsonl": tmp_path / "runs.jsonl",
        "full_bind_preflight_json": tmp_path / "preflight.json",
    }
    for path in evidence_paths.values():
        path.write_text("{}\n")
    composite = _composite_claims(routed_root=routed, non_vq_root=non_vq)
    composite["materialization_scope"] = "bounded"
    evidence_paths["composite_audit_json"].write_text(json.dumps(composite))
    evidence_paths["tokenizer_readiness_json"].write_text(
        json.dumps({"prompt": "x"})
    )
    auditor_calls: list[str] = []

    with pytest.raises(ValueError, match="full"):
        module.validate_glm52_production_inputs(
            profile_path=profile_path,
            config_path=config_path,
            source_index_path=index_path,
            tokenizer_dir=tokenizer,
            non_vq_artifact_dir=non_vq,
            routed_artifact_dir=routed,
            model_id=MODEL_ID,
            revision=REVISION,
            prompt="x",
            expected_config_sha256=hashlib.sha256(
                config_path.read_bytes()
            ).hexdigest(),
            expected_index_sha256=hashlib.sha256(
                index_path.read_bytes()
            ).hexdigest(),
            non_vq_auditor=lambda *_args, **_kwargs: auditor_calls.append("non_vq"),
            routed_auditor=lambda *_args, **_kwargs: auditor_calls.append("routed"),
            full_bind_authenticator=lambda *_args, **_kwargs: auditor_calls.append(
                "preflight"
            ),
            family_policy_validator=lambda _payload: {
                "artifact_target": {
                    "tensor_payload_bytes": ACCEPTED_BYTES,
                    "whole_main_bpw": 1.5934433,
                    "target_accepted_by_user": True,
                    "actual_full_artifact_required": True,
                }
            },
            tokenizer_authenticator=lambda *_args, **_kwargs: {},
            **evidence_paths,
        )

    assert auditor_calls == []


def test_production_input_orchestrator_authenticates_complete_mocked_path(
    tmp_path: Path,
) -> None:
    module = _load_probe_module()
    tokenizer = tmp_path / "snapshot"
    non_vq = tmp_path / "non-vq"
    routed = tmp_path / "routed"
    for root in (tokenizer, non_vq, routed):
        root.mkdir()
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        json.dumps(
            {
                "name": "glm52-reap-504b-v2",
                "hf_model_id": MODEL_ID,
                "revision": REVISION,
                "architecture": "glm_moe_dsa",
                "num_layers": 78,
                "num_sparse_layers": 75,
                "first_sparse_layer": 3,
                "hidden_size": 6144,
                "moe_intermediate_size": 2048,
                "num_experts": 168,
                "experts_per_tok": 8,
                "vocab_size": 154880,
                "shared_experts": 1,
                "converter": "glm52_vq_groups",
                "fused_gate_up": False,
                "default_code_bits": 8,
                "group_size_policy": {"gate": 512, "up": 512, "down": 512},
                "default_engine": "vq_e1_routed_nax_e8",
            }
        )
    )
    config_path = tokenizer / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "num_hidden_layers": 78,
                "hidden_size": 6144,
                "moe_intermediate_size": 2048,
                "n_routed_experts": 168,
                "num_experts_per_tok": 8,
                "vocab_size": 154880,
                "n_shared_experts": 1,
                "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 75,
            }
        )
    )
    index_path = tokenizer / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"metadata": {}, "weight_map": {}}))
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        (tokenizer / filename).write_text("{}\n")
    profile_sha256 = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    index_sha256 = hashlib.sha256(index_path.read_bytes()).hexdigest()

    non_vq_evidence = _non_vq_evidence(non_vq_root=non_vq)
    non_vq_evidence["config_sha256"] = config_sha256
    non_vq_evidence["index_sha256"] = index_sha256
    non_vq_evidence_path = tmp_path / "non-vq-evidence.json"
    non_vq_evidence_path.write_text(json.dumps(non_vq_evidence))
    non_vq_manifest = {
        key: value
        for key, value in non_vq_evidence.items()
        if key not in {"manifest_sha256", "package_audit"}
    }
    (non_vq / "non-vq-manifest.json").write_text(json.dumps(non_vq_manifest))
    (non_vq / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {}})
    )
    (routed / "conversion-manifest.json").write_text(
        json.dumps(
            {"code_bits": 8, "group_size": 512, "scale_estimator": "max_abs"}
        )
    )

    composite = _composite_claims(routed_root=routed, non_vq_root=non_vq)
    composite["config_sha256"] = config_sha256
    composite["index_sha256"] = index_sha256
    composite["non_vq_evidence_sha256"] = hashlib.sha256(
        non_vq_evidence_path.read_bytes()
    ).hexdigest()
    composite["non_vq_package_audit"] = dict(non_vq_evidence["package_audit"])
    composite_path = tmp_path / "composite.json"
    composite_path.write_text(json.dumps(composite))
    preflight = _full_bind_claims(routed_root=routed, non_vq_root=non_vq)
    preflight["profile_sha256"] = profile_sha256
    preflight["config_sha256"] = config_sha256
    preflight["source_index_sha256"] = index_sha256
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight))
    tokenizer_evidence_path = tmp_path / "tokenizer-ready.json"
    tokenizer_evidence_path.write_text(json.dumps({"prompt": "x"}))
    policy_path = tmp_path / "policy.json"
    policy_path.write_text("{}\n")
    runs_path = tmp_path / "runs.jsonl"
    runs_path.write_text("{}\n")

    validated = module.validate_glm52_production_inputs(
        profile_path=profile_path,
        config_path=config_path,
        source_index_path=index_path,
        tokenizer_dir=tokenizer,
        tokenizer_readiness_json=tokenizer_evidence_path,
        family_policy_json=policy_path,
        non_vq_artifact_dir=non_vq,
        non_vq_evidence_json=non_vq_evidence_path,
        routed_artifact_dir=routed,
        composite_audit_json=composite_path,
        materialization_runs_jsonl=runs_path,
        full_bind_preflight_json=preflight_path,
        model_id=MODEL_ID,
        revision=REVISION,
        prompt="x",
        expected_config_sha256=config_sha256,
        expected_index_sha256=index_sha256,
        non_vq_auditor=lambda *_args, **_kwargs: dict(
            non_vq_evidence["package_audit"]
        ),
        routed_auditor=lambda *_args, **_kwargs: {
            "actual_routed_payload_bytes": 61_312_204_800,
            "actual_routed_codebook_bytes": 230_400,
        },
        full_bind_authenticator=lambda *_args, **_kwargs: dict(preflight),
        source_accounting_auditor=lambda *_args, **_kwargs: SimpleNamespace(
            main_model_parameter_count_excluding_mtp=ACCEPTED_PARAMETERS
        ),
        family_policy_validator=lambda _payload: {
            "artifact_target": {
                "tensor_payload_bytes": ACCEPTED_BYTES,
                "whole_main_bpw": 1.5934433,
                "target_accepted_by_user": True,
                "actual_full_artifact_required": True,
            }
        },
        tokenizer_authenticator=lambda *_args, **_kwargs: {},
    )

    assert validated.artifact_identity.body["scale_estimator"] == "max_abs"
    assert validated.input_evidence_file_sha256["composite_audit_json"] == (
        hashlib.sha256(composite_path.read_bytes()).hexdigest()
    )
    module.assert_glm52_production_inputs_unchanged(validated.input_fingerprint)


def test_non_vq_evidence_requires_complete_top_level_identity_and_fresh_audit(
    tmp_path: Path,
) -> None:
    module = _load_probe_module()
    non_vq = tmp_path / "non-vq"
    non_vq.mkdir()
    evidence = _non_vq_evidence(non_vq_root=non_vq)
    fresh = dict(evidence["package_audit"])
    fresh_manifest = {
        key: value
        for key, value in evidence.items()
        if key not in {"manifest_sha256", "package_audit"}
    }

    validated = module.validate_glm52_non_vq_evidence_claims(
        evidence,
        fresh_audit=fresh,
        fresh_manifest=fresh_manifest,
        non_vq_artifact_dir=non_vq,
        config_sha256="b" * 64,
        index_sha256="c" * 64,
    )
    assert validated["source_blob_inventory_sha256"] == "4" * 64

    stale = dict(evidence)
    stale["package_set_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="package_set_sha256"):
        module.validate_glm52_non_vq_evidence_claims(
            stale,
            fresh_audit=fresh,
            fresh_manifest=fresh_manifest,
            non_vq_artifact_dir=non_vq,
            config_sha256="b" * 64,
            index_sha256="c" * 64,
        )

    missing = dict(evidence)
    missing.pop("source_blob_inventory_sha256")
    with pytest.raises(ValueError, match="source_blob_inventory_sha256"):
        module.validate_glm52_non_vq_evidence_claims(
            missing,
            fresh_audit=fresh,
            fresh_manifest=fresh_manifest,
            non_vq_artifact_dir=non_vq,
            config_sha256="b" * 64,
            index_sha256="c" * 64,
        )

    stale_index = dict(evidence)
    stale_index["package_index_sha256"] = "7" * 64
    with pytest.raises(ValueError, match="package_index_sha256"):
        module.validate_glm52_non_vq_evidence_claims(
            stale_index,
            fresh_audit=fresh,
            fresh_manifest=fresh_manifest,
            non_vq_artifact_dir=non_vq,
            config_sha256="b" * 64,
            index_sha256="c" * 64,
        )

    stale_path = dict(evidence)
    stale_path["package_audit"] = dict(evidence["package_audit"])
    stale_path["package_audit"]["artifact_dir"] = str(tmp_path / "other")
    with pytest.raises(ValueError, match="artifact_dir"):
        module.validate_glm52_non_vq_evidence_claims(
            stale_path,
            fresh_audit=fresh,
            fresh_manifest=fresh_manifest,
            non_vq_artifact_dir=non_vq,
            config_sha256="b" * 64,
            index_sha256="c" * 64,
        )


def test_common_artifact_identity_is_path_free_and_uses_exact_accepted_accounting(
    tmp_path: Path,
) -> None:
    module = _load_probe_module()
    routed = tmp_path / "routed"
    non_vq = tmp_path / "non-vq"
    claims = _composite_claims(routed_root=routed, non_vq_root=non_vq)

    identity = module.build_glm52_composite_artifact_identity(
        profile="glm52-reap-504b-v2",
        profile_sha256="2" * 64,
        profile_contract_sha256="3" * 64,
        composite_audit=claims,
        non_vq_evidence={
            "source_blob_inventory_sha256": "4" * 64,
            "source_inventory_sha256": "5" * 64,
            "package_index_sha256": "6" * 64,
        },
    )

    assert identity.body["whole_model_tensor_payload_bytes"] == ACCEPTED_BYTES
    assert identity.body["whole_model_parameter_count"] == ACCEPTED_PARAMETERS
    assert identity.body["whole_model_tensor_payload_bpw"] == pytest.approx(
        ACCEPTED_BPW,
        rel=0,
        abs=1e-15,
    )
    serialized = json.dumps(identity.body, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert identity.sha256 == module._canonical_sha256(identity.body)


def test_routed_manifest_policy_requires_authenticated_max_abs() -> None:
    module = _load_probe_module()
    claims = {
        "requested_code_bits": 8,
        "requested_group_size": 512,
        "requested_scale_estimator": "max_abs",
    }
    manifest = {
        "code_bits": 8,
        "group_size": 512,
        "scale_estimator": "percentile_99",
    }

    with pytest.raises(ValueError, match="scale estimator"):
        module.validate_glm52_routed_manifest_policy(
            manifest,
            composite_audit=claims,
        )

    manifest["scale_estimator"] = "max_abs"
    assert module.validate_glm52_routed_manifest_policy(
        manifest,
        composite_audit=claims,
    ) == {"code_bits": 8, "group_size": 512, "scale_estimator": "max_abs"}


def test_authenticated_loader_streams_non_vq_before_exact_routed_bind() -> None:
    module = _load_composite_loader_module()
    calls: list[object] = []
    model = SimpleNamespace(eval=lambda: calls.append("eval"))
    non_vq_report = SimpleNamespace(
        loaded_count=1272,
        missing_model_parameters=(),
        skipped_unmatched_tensors=(),
        skipped_routed_expert_tensors=(),
        skipped_mtp_tensors=(),
        to_dict=lambda: {"loaded_count": 1272},
    )
    validated = SimpleNamespace(
        config={},
        profile=object(),
        non_vq_index=object(),
        non_vq_artifact_dir=Path("/non-vq"),
        routed_artifact_dir=Path("/routed"),
        artifact_identity=SimpleNamespace(sha256="7" * 64),
        input_fingerprint=object(),
    )

    def model_factory(_args):
        calls.append("construct")
        return model

    def non_vq_binder(bound_model, artifact, index, *, strict):
        calls.append(("non_vq", bound_model, artifact, index, strict))
        return non_vq_report

    def routed_binder(bound_model, artifact, *, profile, strict):
        calls.append(("routed", bound_model, artifact, profile, strict))
        return tuple(range(3, 78))

    returned_model, report = module.load_authenticated_glm52_composite(
        validated,
        model_factory=model_factory,
        args_factory=lambda _config: object(),
        non_vq_binder=non_vq_binder,
        routed_binder=routed_binder,
        unbound_checker=lambda _model: False,
        dense_parameter_checker=lambda _model: (),
        integrity_checker=lambda fingerprint: calls.append(
            ("integrity", fingerprint)
        ),
        phase_marker=lambda label: calls.append(("phase", label)),
    )

    assert returned_model is model
    assert calls[0] == ("integrity", validated.input_fingerprint)
    assert [call[0] for call in calls if isinstance(call, tuple)].count(
        "integrity"
    ) == 3
    bind_order = [
        call[0]
        for call in calls
        if isinstance(call, tuple) and call[0] in {"non_vq", "routed"}
    ]
    assert bind_order == [
        "non_vq",
        "routed",
    ]
    assert report.bound_sparse_layer_ids == tuple(range(3, 78))
    assert report.non_vq_bind_report.loaded_count == 1272
    assert report.unbound_vq_experts is False
    assert report.dense_routed_parameter_names == ()


@pytest.mark.parametrize(
    ("audit_updates", "match"),
    [
        ({"candidate_identity_sha256": "0" * 64}, "candidate identity"),
        ({"manifest_body_sha256": "0" * 64}, "manifest body"),
        ({"accepted_composite_audit_sha256": "0" * 64}, "composite audit"),
        (
            {"accepted_baseline_composite_identity_sha256": "0" * 64},
            "baseline composite identity",
        ),
        ({"audit_pass": False}, "audit pass"),
    ],
)
def test_recovery_candidate_validation_rejects_audit_identity_mismatch(
    tmp_path: Path,
    audit_updates: dict[str, object],
    match: str,
) -> None:
    module = _load_composite_loader_module()
    baseline = _recovery_baseline()
    recovery_dir = _recovery_tree(tmp_path)
    audit = _FakeRecoveryAudit(
        recovery_dir=str(recovery_dir.resolve()),
        manifest_path=str((recovery_dir / "conversion-manifest.json").resolve()),
    )
    for field, value in audit_updates.items():
        setattr(audit, field, value)

    with pytest.raises(ValueError, match=match):
        module.validate_glm52_recovery_candidate_inputs(
            baseline,
            recovery_dir=recovery_dir,
            recovery_auditor=lambda *_args, **_kwargs: audit,
            integrity_checker=lambda _fingerprint: None,
            **_recovery_authority(),
        )


def test_recovery_candidate_validation_rejects_baseline_drift_before_audit(
    tmp_path: Path,
) -> None:
    module = _load_composite_loader_module()
    baseline = _recovery_baseline()
    recovery_dir = _recovery_tree(tmp_path)
    auditor_called = False

    def recovery_auditor(*_args, **_kwargs):
        nonlocal auditor_called
        auditor_called = True
        raise AssertionError("auditor must not run after baseline drift")

    with pytest.raises(ValueError, match="baseline drift"):
        module.validate_glm52_recovery_candidate_inputs(
            baseline,
            recovery_dir=recovery_dir,
            recovery_auditor=recovery_auditor,
            integrity_checker=lambda _fingerprint: (_ for _ in ()).throw(
                ValueError("baseline drift")
            ),
            **_recovery_authority(),
        )

    assert auditor_called is False


def test_recovery_candidate_validation_requires_exact_audited_bind_root(
    tmp_path: Path,
) -> None:
    module = _load_composite_loader_module()
    baseline = _recovery_baseline()
    recovery_dir = _recovery_tree(tmp_path)
    other_dir = _recovery_tree(tmp_path / "other")
    audit = _FakeRecoveryAudit(
        recovery_dir=str(other_dir.resolve()),
        manifest_path=str((other_dir / "conversion-manifest.json").resolve()),
    )

    with pytest.raises(ValueError, match="recovery directory"):
        module.validate_glm52_recovery_candidate_inputs(
            baseline,
            recovery_dir=recovery_dir,
            recovery_auditor=lambda *_args, **_kwargs: audit,
            integrity_checker=lambda _fingerprint: None,
            **_recovery_authority(),
        )


@pytest.mark.parametrize(
    ("fail_on_verify_call", "expected_bind_calls"),
    [
        (2, []),
        (3, ["non_vq"]),
        (4, ["non_vq", "routed"]),
    ],
)
def test_recovery_loader_rejects_mutation_at_every_integrity_checkpoint(
    tmp_path: Path,
    fail_on_verify_call: int,
    expected_bind_calls: list[str],
) -> None:
    module = _load_composite_loader_module()
    baseline = _recovery_baseline()
    recovery_dir = _recovery_tree(tmp_path)
    audit = _FakeRecoveryAudit(
        recovery_dir=str(recovery_dir.resolve()),
        manifest_path=str((recovery_dir / "conversion-manifest.json").resolve()),
        fail_on_verify_call=fail_on_verify_call,
    )
    validated = module.validate_glm52_recovery_candidate_inputs(
        baseline,
        recovery_dir=recovery_dir,
        recovery_auditor=lambda *_args, **_kwargs: audit,
        integrity_checker=lambda _fingerprint: None,
        **_recovery_authority(),
    )
    bind_calls: list[str] = []
    model = SimpleNamespace(eval=lambda: None)
    non_vq_report = SimpleNamespace(
        loaded_count=1272,
        missing_model_parameters=(),
        skipped_unmatched_tensors=(),
        skipped_routed_expert_tensors=(),
        skipped_mtp_tensors=(),
    )

    with pytest.raises(ValueError, match="current identity changed"):
        module.load_authenticated_glm52_recovery_candidate(
            validated,
            snapshot_scratch_dir=tmp_path,
            allow_snapshot_copy_fallback=True,
            model_factory=lambda _args: model,
            args_factory=lambda _config: object(),
            non_vq_binder=lambda *_args, **_kwargs: (
                bind_calls.append("non_vq") or non_vq_report
            ),
            routed_binder=lambda *_args, **_kwargs: (
                bind_calls.append("routed") or tuple(range(3, 78))
            ),
            unbound_checker=lambda _model: False,
            dense_parameter_checker=lambda _model: (),
            integrity_checker=lambda _fingerprint: None,
        )

    assert bind_calls == expected_bind_calls


def test_recovery_loader_binds_audited_artifact_and_reports_both_identities(
    tmp_path: Path,
) -> None:
    module = _load_composite_loader_module()
    baseline = _recovery_baseline()
    accepted_root = baseline.routed_artifact_dir
    accepted_identity = baseline.artifact_identity.sha256
    recovery_dir = _recovery_tree(tmp_path)
    audit = _FakeRecoveryAudit(
        recovery_dir=str(recovery_dir.resolve()),
        manifest_path=str((recovery_dir / "conversion-manifest.json").resolve()),
    )
    auditor_calls: list[tuple[object, dict[str, object]]] = []

    def recovery_auditor(root, **kwargs):
        auditor_calls.append((root, kwargs))
        return audit

    integrity_calls: list[object] = []
    validated = module.validate_glm52_recovery_candidate_inputs(
        baseline,
        recovery_dir=recovery_dir,
        recovery_auditor=recovery_auditor,
        integrity_checker=integrity_calls.append,
        **_recovery_authority(),
    )
    model = SimpleNamespace(eval=lambda: None)
    non_vq_report = SimpleNamespace(
        loaded_count=1272,
        missing_model_parameters=(),
        skipped_unmatched_tensors=(),
        skipped_routed_expert_tensors=(),
        skipped_mtp_tensors=(),
    )
    routed_sources: list[dict[str, Path]] = []
    returned_model, report = module.load_authenticated_glm52_recovery_candidate(
        validated,
        snapshot_scratch_dir=tmp_path,
        allow_snapshot_copy_fallback=True,
        model_factory=lambda _args: model,
        args_factory=lambda _config: object(),
        non_vq_binder=lambda _model, root, _index, *, strict: non_vq_report,
        routed_binder=lambda _model, paths, *, profile, strict: (
            routed_sources.append(dict(paths)) or tuple(range(3, 78))
        ),
        unbound_checker=lambda _model: False,
        dense_parameter_checker=lambda _model: (),
        integrity_checker=integrity_calls.append,
    )

    assert returned_model is model
    assert auditor_calls[0][0] == recovery_dir.resolve()
    assert auditor_calls[0][1]["profile"] is baseline.profile
    assert len(routed_sources) == 1
    snapshot_path = routed_sources[0]["layer-00003-gate_proj.safetensors"]
    assert str(snapshot_path).startswith("/dev/fd/")
    assert snapshot_path.read_bytes() == b"audited recovery group"
    assert model._glm52_recovery_bind_snapshot.file_paths == routed_sources[0]
    assert validated.baseline is baseline
    assert baseline.routed_artifact_dir == accepted_root
    assert baseline.artifact_identity.sha256 == accepted_identity
    assert report.baseline_composite_identity_sha256 == "d" * 64
    assert report.recovery_candidate_identity_sha256 == "a" * 64
    assert report.recovery_manifest_body_sha256 == "b" * 64
    assert len(integrity_calls) == 5
    assert audit.verify_current_identity_calls == 4


def test_recovery_loader_binds_from_snapshot_during_symlink_aba_swap(
    tmp_path: Path,
) -> None:
    module = _load_composite_loader_module()
    baseline = _recovery_baseline()
    recovery_dir = _recovery_tree(tmp_path)
    artifact = recovery_dir / "artifact/layer-00003-gate_proj.safetensors"
    audited_bytes = artifact.read_bytes()
    substituted = recovery_dir / "groups/substituted.safetensors"
    substituted.write_bytes(b"shape-compatible unaudited group")
    audit = _FakeRecoveryAudit(
        recovery_dir=str(recovery_dir.resolve()),
        manifest_path=str((recovery_dir / "conversion-manifest.json").resolve()),
    )
    validated = module.validate_glm52_recovery_candidate_inputs(
        baseline,
        recovery_dir=recovery_dir,
        recovery_auditor=lambda *_args, **_kwargs: audit,
        integrity_checker=lambda _fingerprint: None,
        **_recovery_authority(),
    )
    model = SimpleNamespace(eval=lambda: None)
    non_vq_report = SimpleNamespace(
        loaded_count=1272,
        missing_model_parameters=(),
        skipped_unmatched_tensors=(),
        skipped_routed_expert_tensors=(),
        skipped_mtp_tensors=(),
    )
    loaded: list[bytes] = []

    def routed_binder(_model, paths, *, profile, strict):
        artifact.unlink()
        artifact.symlink_to(Path("../groups") / substituted.name)
        loaded.append(Path(paths[artifact.name]).read_bytes())
        artifact.unlink()
        artifact.symlink_to(Path("../groups/layer-00003-gate_proj.safetensors"))
        return tuple(range(3, 78))

    module.load_authenticated_glm52_recovery_candidate(
        validated,
        snapshot_scratch_dir=tmp_path,
        allow_snapshot_copy_fallback=True,
        model_factory=lambda _args: model,
        args_factory=lambda _config: object(),
        non_vq_binder=lambda *_args, **_kwargs: non_vq_report,
        routed_binder=routed_binder,
        unbound_checker=lambda _model: False,
        dense_parameter_checker=lambda _model: (),
        integrity_checker=lambda _fingerprint: None,
    )

    assert loaded == [audited_bytes]


def test_recovery_loader_snapshot_is_independent_during_inode_byte_aba(
    tmp_path: Path,
) -> None:
    module = _load_composite_loader_module()
    baseline = _recovery_baseline()
    recovery_dir = _recovery_tree(tmp_path)
    artifact = recovery_dir / "artifact/layer-00003-gate_proj.safetensors"
    source = artifact.resolve()
    audited_bytes = source.read_bytes()
    audit = _FakeRecoveryAudit(
        recovery_dir=str(recovery_dir.resolve()),
        manifest_path=str((recovery_dir / "conversion-manifest.json").resolve()),
    )
    validated = module.validate_glm52_recovery_candidate_inputs(
        baseline,
        recovery_dir=recovery_dir,
        recovery_auditor=lambda *_args, **_kwargs: audit,
        integrity_checker=lambda _fingerprint: None,
        **_recovery_authority(),
    )
    model = SimpleNamespace(eval=lambda: None)
    non_vq_report = SimpleNamespace(
        loaded_count=1272,
        missing_model_parameters=(),
        skipped_unmatched_tensors=(),
        skipped_routed_expert_tensors=(),
        skipped_mtp_tensors=(),
    )
    loaded: list[bytes] = []

    def routed_binder(_model, paths, *, profile, strict):
        source.write_bytes(b"shape-compatible unaudited group")
        loaded.append(Path(paths[artifact.name]).read_bytes())
        source.write_bytes(audited_bytes)
        return tuple(range(3, 78))

    module.load_authenticated_glm52_recovery_candidate(
        validated,
        snapshot_scratch_dir=tmp_path,
        allow_snapshot_copy_fallback=True,
        model_factory=lambda _args: model,
        args_factory=lambda _config: object(),
        non_vq_binder=lambda *_args, **_kwargs: non_vq_report,
        routed_binder=routed_binder,
        unbound_checker=lambda _model: False,
        dense_parameter_checker=lambda _model: (),
        integrity_checker=lambda _fingerprint: None,
    )

    assert loaded == [audited_bytes]


def test_recovery_loader_hashes_the_exact_retained_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlx_vq.io import authenticated_artifacts

    module = _load_composite_loader_module()
    baseline = _recovery_baseline()
    recovery_dir = _recovery_tree(tmp_path)
    substituted = recovery_dir / "groups/substituted.safetensors"
    substituted.write_bytes(b"x" * len(b"audited recovery group"))
    audit = _FakeRecoveryAudit(
        recovery_dir=str(recovery_dir.resolve()),
        manifest_path=str((recovery_dir / "conversion-manifest.json").resolve()),
    )
    validated = module.validate_glm52_recovery_candidate_inputs(
        baseline,
        recovery_dir=recovery_dir,
        recovery_auditor=lambda *_args, **_kwargs: audit,
        integrity_checker=lambda _fingerprint: None,
        **_recovery_authority(),
    )
    real_authenticated_open = authenticated_artifacts.AuthenticatedFile.open.__func__
    swapped = False

    def replace_before_retained_open(cls, path, *, label, **kwargs):
        nonlocal swapped
        candidate = Path(path)
        if (
            not swapped
            and candidate.name == "layer-00003-gate_proj.safetensors"
            and candidate.parent.name.startswith(".glm52-recovery-bind-")
        ):
            module.os.replace(substituted, candidate)
            swapped = True
        return real_authenticated_open(cls, path, label=label, **kwargs)

    monkeypatch.setattr(
        authenticated_artifacts.AuthenticatedFile,
        "open",
        classmethod(replace_before_retained_open),
    )

    with pytest.raises(ValueError, match="snapshot hash drifted"):
        module.load_authenticated_glm52_recovery_candidate(
            validated,
            snapshot_scratch_dir=tmp_path,
            allow_snapshot_copy_fallback=True,
            model_factory=lambda _args: pytest.fail("model constructed before snapshot"),
            integrity_checker=lambda _fingerprint: None,
        )

    assert swapped is True


@pytest.mark.parametrize(
    ("allow_copy_fallback", "free_bytes", "match"),
    [
        (False, 1024, "fallback was not explicitly enabled"),
        (True, 0, "requires 1 free bytes"),
    ],
)
def test_recovery_snapshot_copy_fallback_is_explicit_and_space_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allow_copy_fallback: bool,
    free_bytes: int,
    match: str,
) -> None:
    from mlx_vq.io import authenticated_artifacts

    module = _load_composite_loader_module()
    monkeypatch.setattr(
        authenticated_artifacts,
        "_try_fclonefileat",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        authenticated_artifacts.os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=free_bytes, f_frsize=1),
    )
    source = tmp_path / "source"
    source.write_bytes(b"x")
    authenticated = authenticated_artifacts.AuthenticatedFile.open(
        source, label="recovery bind source"
    )
    try:
        with pytest.raises(ValueError, match=match):
            module._clone_or_copy_open_file(
                authenticated,
                tmp_path / "snapshot",
                allow_copy_fallback=allow_copy_fallback,
                required_free_bytes=1,
            )
    finally:
        authenticated.close()


def test_recovery_snapshot_tracks_only_remaining_copy_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_composite_loader_module()
    baseline = _recovery_baseline()
    recovery_dir = _recovery_tree(tmp_path)
    second = recovery_dir / "groups/layer-00003-up_proj.safetensors"
    second.write_bytes(b"second")
    (recovery_dir / "artifact/layer-00003-up_proj.safetensors").symlink_to(
        Path("../groups") / second.name
    )

    def group_for(filename: str) -> SimpleNamespace:
        artifact = recovery_dir / "artifact" / filename
        return SimpleNamespace(
            filename=filename,
            artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            artifact_bytes=artifact.stat().st_size,
        )

    audit = _FakeRecoveryAudit(
        recovery_dir=str(recovery_dir.resolve()),
        manifest_path=str((recovery_dir / "conversion-manifest.json").resolve()),
        groups=(
            group_for("layer-00003-gate_proj.safetensors"),
            group_for("layer-00003-up_proj.safetensors"),
        ),
    )
    validated = module.validate_glm52_recovery_candidate_inputs(
        baseline,
        recovery_dir=recovery_dir,
        recovery_auditor=lambda *_args, **_kwargs: audit,
        integrity_checker=lambda _fingerprint: None,
        **_recovery_authority(),
    )
    real_clone_or_copy = module._clone_or_copy_open_file
    required: list[int] = []

    def record_required(descriptor, destination, **kwargs):
        required.append(kwargs["required_free_bytes"])
        return real_clone_or_copy(descriptor, destination, **kwargs)

    monkeypatch.setattr(module, "_clone_or_copy_open_file", record_required)
    snapshot = module._snapshot_glm52_recovery_bind_artifacts(
        validated,
        snapshot_scratch_dir=tmp_path,
        allow_copy_fallback=True,
    )
    try:
        first_size = audit.groups[0].artifact_bytes
        second_size = audit.groups[1].artifact_bytes
        assert required == [first_size + second_size, second_size]
    finally:
        snapshot.close()


def test_recovery_loader_streams_source_authority_one_group_at_a_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlx_vq.io import authenticated_artifacts

    module = _load_composite_loader_module()
    baseline = _recovery_baseline()
    recovery_dir = _recovery_tree(tmp_path)
    groups: list[SimpleNamespace] = []
    for index in range(32):
        filename = f"stream-{index:03d}.safetensors"
        source = recovery_dir / "groups" / filename
        source.write_bytes(f"group-{index}".encode())
        (recovery_dir / "artifact" / filename).symlink_to(
            Path("../groups") / filename
        )
        groups.append(
            SimpleNamespace(
                filename=filename,
                artifact_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                artifact_bytes=source.stat().st_size,
            )
        )
    audit = _FakeRecoveryAudit(
        recovery_dir=str(recovery_dir.resolve()),
        manifest_path=str((recovery_dir / "conversion-manifest.json").resolve()),
        groups=tuple(groups),
    )
    validated = module.validate_glm52_recovery_candidate_inputs(
        baseline,
        recovery_dir=recovery_dir,
        recovery_auditor=lambda *_args, **_kwargs: audit,
        integrity_checker=lambda _fingerprint: None,
        **_recovery_authority(),
    )
    real_open = authenticated_artifacts.AuthenticatedFile.open.__func__
    source_authorities: list[authenticated_artifacts.AuthenticatedFile] = []
    peak_live_sources = 0

    def track_source_authority(cls, path, *, label, **kwargs):
        nonlocal peak_live_sources
        authenticated = real_open(cls, path, label=label, **kwargs)
        if label.startswith("recovery bind source ") and not label.endswith(
            " snapshot"
        ):
            source_authorities.append(authenticated)
            peak_live_sources = max(
                peak_live_sources,
                sum(not source._closed for source in source_authorities),
            )
        return authenticated

    monkeypatch.setattr(
        authenticated_artifacts.AuthenticatedFile,
        "open",
        classmethod(track_source_authority),
    )
    snapshot = module._snapshot_glm52_recovery_bind_artifacts(
        validated,
        snapshot_scratch_dir=tmp_path,
        allow_copy_fallback=True,
    )
    try:
        assert peak_live_sources == 1
        assert len(snapshot.descriptors) == len(groups)
        assert len(snapshot.file_paths) == len(groups)
    finally:
        snapshot.close()


def test_recovery_loader_closes_retained_descriptor_when_unlink_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlx_vq.io import authenticated_artifacts

    module = _load_composite_loader_module()
    baseline = _recovery_baseline()
    recovery_dir = _recovery_tree(tmp_path)
    audit = _FakeRecoveryAudit(
        recovery_dir=str(recovery_dir.resolve()),
        manifest_path=str((recovery_dir / "conversion-manifest.json").resolve()),
    )
    validated = module.validate_glm52_recovery_candidate_inputs(
        baseline,
        recovery_dir=recovery_dir,
        recovery_auditor=lambda *_args, **_kwargs: audit,
        integrity_checker=lambda _fingerprint: None,
        **_recovery_authority(),
    )
    real_dup = authenticated_artifacts.os.dup
    real_unlink = authenticated_artifacts.os.unlink
    retained: list[int] = []
    failed = False

    def record_retained_dup(descriptor):
        descriptor = real_dup(descriptor)
        retained.append(descriptor)
        return descriptor

    def fail_first_snapshot_unlink(path, *args, **kwargs):
        nonlocal failed
        candidate = Path(path)
        if (
            not failed
            and candidate.name == "layer-00003-gate_proj.safetensors"
            and kwargs.get("dir_fd") is not None
        ):
            failed = True
            raise PermissionError("injected snapshot unlink failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(authenticated_artifacts.os, "dup", record_retained_dup)
    monkeypatch.setattr(authenticated_artifacts.os, "unlink", fail_first_snapshot_unlink)

    with pytest.raises(PermissionError, match="snapshot unlink failure"):
        module.load_authenticated_glm52_recovery_candidate(
            validated,
            snapshot_scratch_dir=tmp_path,
            allow_snapshot_copy_fallback=True,
            model_factory=lambda _args: pytest.fail("model constructed before snapshot"),
            integrity_checker=lambda _fingerprint: None,
        )

    assert failed is True
    assert len(retained) == 1
    with pytest.raises(OSError):
        module.os.fstat(retained[0])


def test_recovery_loader_explicit_close_releases_descriptors_idempotently(
    tmp_path: Path,
) -> None:
    module = _load_composite_loader_module()
    baseline = _recovery_baseline()
    recovery_dir = _recovery_tree(tmp_path)
    audit = _FakeRecoveryAudit(
        recovery_dir=str(recovery_dir.resolve()),
        manifest_path=str((recovery_dir / "conversion-manifest.json").resolve()),
    )
    validated = module.validate_glm52_recovery_candidate_inputs(
        baseline,
        recovery_dir=recovery_dir,
        recovery_auditor=lambda *_args, **_kwargs: audit,
        integrity_checker=lambda _fingerprint: None,
        **_recovery_authority(),
    )

    class Model:
        def eval(self) -> None:
            return None

    model = Model()
    non_vq_report = SimpleNamespace(
        loaded_count=1272,
        missing_model_parameters=(),
        skipped_unmatched_tensors=(),
        skipped_routed_expert_tensors=(),
        skipped_mtp_tensors=(),
    )
    loaded, _report = module.load_authenticated_glm52_recovery_candidate(
        validated,
        snapshot_scratch_dir=tmp_path,
        allow_snapshot_copy_fallback=True,
        model_factory=lambda _args: model,
        args_factory=lambda _config: object(),
        non_vq_binder=lambda *_args, **_kwargs: non_vq_report,
        routed_binder=lambda *_args, **_kwargs: tuple(range(3, 78)),
        unbound_checker=lambda _model: False,
        dense_parameter_checker=lambda _model: (),
        integrity_checker=lambda _fingerprint: None,
    )
    descriptors = loaded._glm52_recovery_bind_snapshot.descriptors

    module.close_authenticated_glm52_recovery_candidate(loaded)
    module.close_authenticated_glm52_recovery_candidate(loaded)

    assert descriptors
    for descriptor in descriptors:
        with pytest.raises(OSError):
            module.os.fstat(descriptor)


def test_recovery_loader_model_finalizer_releases_descriptors(
    tmp_path: Path,
) -> None:
    module = _load_composite_loader_module()
    baseline = _recovery_baseline()
    recovery_dir = _recovery_tree(tmp_path)
    audit = _FakeRecoveryAudit(
        recovery_dir=str(recovery_dir.resolve()),
        manifest_path=str((recovery_dir / "conversion-manifest.json").resolve()),
    )
    validated = module.validate_glm52_recovery_candidate_inputs(
        baseline,
        recovery_dir=recovery_dir,
        recovery_auditor=lambda *_args, **_kwargs: audit,
        integrity_checker=lambda _fingerprint: None,
        **_recovery_authority(),
    )

    class Model:
        def eval(self) -> None:
            return None

    non_vq_report = SimpleNamespace(
        loaded_count=1272,
        missing_model_parameters=(),
        skipped_unmatched_tensors=(),
        skipped_routed_expert_tensors=(),
        skipped_mtp_tensors=(),
    )
    loaded, _report = module.load_authenticated_glm52_recovery_candidate(
        validated,
        snapshot_scratch_dir=tmp_path,
        allow_snapshot_copy_fallback=True,
        model_factory=lambda _args: Model(),
        args_factory=lambda _config: object(),
        non_vq_binder=lambda *_args, **_kwargs: non_vq_report,
        routed_binder=lambda *_args, **_kwargs: tuple(range(3, 78)),
        unbound_checker=lambda _model: False,
        dense_parameter_checker=lambda _model: (),
        integrity_checker=lambda _fingerprint: None,
    )
    descriptors = loaded._glm52_recovery_bind_snapshot.descriptors
    model_ref = weakref.ref(loaded)

    del loaded
    gc.collect()

    assert model_ref() is None
    for descriptor in descriptors:
        with pytest.raises(OSError):
            module.os.fstat(descriptor)


def test_recovery_loader_exports_are_additive_to_baseline_api() -> None:
    module = _load_composite_loader_module()
    baseline_exports = {
        "GLM52ValidatedProductionInputs",
        "GLM52CompositeLoadReport",
        "validate_glm52_production_inputs",
        "load_authenticated_glm52_composite",
        "build_glm52_composite_artifact_identity",
    }
    recovery_exports = {
        "GLM52ValidatedRecoveryCandidate",
        "GLM52RecoveryCandidateLoadReport",
        "validate_glm52_recovery_candidate_inputs",
        "load_authenticated_glm52_recovery_candidate",
        "close_authenticated_glm52_recovery_candidate",
    }

    assert baseline_exports | recovery_exports <= set(module.__all__)


def test_generation_record_requires_one_finite_full_vocabulary_greedy_row() -> None:
    module = _load_probe_module()
    valid = {
        "generated_token_ids": [42],
        "generated_token_count": 1,
        "logits_shape": [154_880],
        "logits_finite": True,
        "greedy_sampling": True,
    }
    assert module.validate_glm52_generation_record(valid) == valid

    for field, value in (
        ("generated_token_count", 2),
        ("logits_shape", [1]),
        ("logits_finite", False),
        ("greedy_sampling", False),
        ("generated_token_ids", [154_880]),
    ):
        invalid = dict(valid)
        invalid[field] = value
        with pytest.raises(ValueError):
            module.validate_glm52_generation_record(invalid)


def test_generation_consistency_requires_matching_greedy_token_ids() -> None:
    module = _load_probe_module()
    warmup = {"generated_token_ids": [42]}
    measured = {"generated_token_ids": [42]}

    assert module.validate_glm52_generation_consistency(warmup, measured) is True

    with pytest.raises(ValueError, match="differ"):
        module.validate_glm52_generation_consistency(
            warmup,
            {"generated_token_ids": [43]},
        )


def test_measured_generation_record_requires_direct_single_decode() -> None:
    module = _load_probe_module()
    valid = {
        "generated_token_ids": [42],
        "generated_token_count": 1,
        "logits_shape": [154_880],
        "logits_finite": True,
        "greedy_sampling": True,
        "generation_method": "direct_prefill_single_decode",
        "lookahead_forward_scheduled": False,
    }

    assert module.validate_glm52_measured_generation_record(valid) == valid

    for field, value in (
        ("generation_method", "mlx_lm_generate_step"),
        ("lookahead_forward_scheduled", True),
    ):
        invalid = dict(valid)
        invalid[field] = value
        with pytest.raises(ValueError, match="direct|lookahead"):
            module.validate_glm52_measured_generation_record(invalid)


def test_one_token_runner_exhausts_generator_cleanup_after_first_yield() -> None:
    module = _load_probe_module()
    events: list[str] = []
    generation_stream = object()

    class Model:
        def make_cache(self):
            events.append("cache")
            return object()

    def generation_factory(
        _prompt,
        _model,
        *,
        max_tokens,
        sampler,
        prompt_cache,
    ):
        assert max_tokens == 1
        assert callable(sampler)
        assert prompt_cache is not None
        events.append("start")
        try:
            yield module.mx.array(42), module.mx.zeros((154_880,))
            events.append("cleanup")
        finally:
            events.append("closed")

    payload = module.run_one_greedy_token(
        Model(),
        [1, 2, 3],
        generation_factory=generation_factory,
        synchronize=lambda stream: events.append(("synchronize", stream)),
        generation_stream_value=generation_stream,
    )

    assert events == [
        "cache",
        "start",
        "cleanup",
        "closed",
        ("synchronize", generation_stream),
    ]
    assert payload["generated_token_ids"] == [42]
    assert payload["generated_token_count"] == 1
    assert payload["logits_shape"] == [154_880]
    assert payload["logits_finite"] is True


def test_one_token_runner_synchronizes_generation_stream_after_failure() -> None:
    module = _load_probe_module()
    events: list[object] = []
    generation_stream = object()

    class Model:
        def make_cache(self):
            return object()

    def generation_factory(*_args, **_kwargs):
        try:
            events.append("start")
            raise RuntimeError("generation failed")
            yield
        finally:
            events.append("closed")

    with pytest.raises(RuntimeError, match="generation failed"):
        module.run_one_greedy_token(
            Model(),
            [1],
            generation_factory=generation_factory,
            synchronize=lambda stream: events.append(("synchronize", stream)),
            generation_stream_value=generation_stream,
        )

    assert events == [
        "start",
        "closed",
        ("synchronize", generation_stream),
    ]


def test_direct_prefill_runner_avoids_decode_lookahead() -> None:
    module = _load_probe_module()
    events: list[object] = []
    generation_stream = object()
    prompt_cache = [SimpleNamespace(state=module.mx.array(0))]

    class Model:
        def make_cache(self):
            events.append("cache")
            return prompt_cache

        def __call__(self, inputs, *, cache):
            events.append(("forward", tuple(inputs.shape), cache is prompt_cache))
            return module.mx.zeros((1, inputs.shape[1], 154_880))

    def stream_context(stream):
        events.append(("stream", stream))
        return nullcontext()

    payload = module.run_one_direct_greedy_token(
        Model(),
        [1, 2, 3],
        stream_context=stream_context,
        clear_cache=lambda: events.append("clear_cache"),
        synchronize=lambda stream: events.append(("synchronize", stream)),
        generation_stream_value=generation_stream,
    )

    assert events == [
        "cache",
        ("stream", generation_stream),
        ("forward", (1, 2), True),
        "clear_cache",
        ("forward", (1, 1), True),
        ("synchronize", generation_stream),
    ]
    assert payload["generated_token_ids"] == [0]
    assert payload["generated_token_count"] == 1
    assert payload["logits_shape"] == [154_880]
    assert payload["logits_finite"] is True
    assert payload["generation_method"] == "direct_prefill_single_decode"
    assert payload["lookahead_forward_scheduled"] is False
    assert payload["prefill_token_count"] == 2
    assert payload["decode_token_count"] == 1


def test_residency_barrier_evaluates_parameters_before_quiet_measurement() -> None:
    module = _load_probe_module()
    events: list[object] = []
    parameters = {"weight": object()}

    class Model:
        def parameters(self):
            events.append("parameters")
            return parameters

    def evaluate(value):
        events.append(("evaluate", value))

    def synchronize():
        events.append("synchronize")

    def clear_cache():
        events.append("clear_cache")

    def quiet_probe(*, window_seconds, max_attempts):
        events.append(("quiet", window_seconds, max_attempts))
        return {
            "enabled": True,
            "available": True,
            "quiet": True,
            "attempts": 1,
            "window_seconds": window_seconds,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        }

    snapshots = iter(
        (
            {
                "mlx_active_bytes": ACCEPTED_BYTES,
                "mlx_peak_bytes": ACCEPTED_BYTES,
                "mlx_cache_bytes": 2_000,
                "rss_bytes": 1,
            },
            {
                "mlx_active_bytes": ACCEPTED_BYTES,
                "mlx_peak_bytes": ACCEPTED_BYTES,
                "mlx_cache_bytes": 0,
                "rss_bytes": 1,
            },
            {
                "mlx_active_bytes": ACCEPTED_BYTES,
                "mlx_peak_bytes": ACCEPTED_BYTES,
                "mlx_cache_bytes": 0,
                "rss_bytes": 1,
            },
        )
    )

    def snapshot():
        events.append("snapshot")
        return next(snapshots)

    result = module.establish_glm52_residency(
        Model(),
        evaluate_parameters=evaluate,
        synchronize=synchronize,
        clear_cache=clear_cache,
        quiet_probe=quiet_probe,
        metric_snapshot=snapshot,
        quiet_window_seconds=15.0,
        quiet_max_attempts=3,
    )

    assert events == [
        "parameters",
        ("evaluate", parameters),
        "synchronize",
        "snapshot",
        "clear_cache",
        "synchronize",
        "snapshot",
        ("quiet", 15.0, 3),
        "snapshot",
    ]
    assert result["parameter_residency_established"] is True
    assert result["memory_quiet"]["quiet"] is True
    assert result["cache_clear"]["released_bytes"] == 2_000
    assert result["resident_memory"]["mlx_active_bytes"] == ACCEPTED_BYTES


def test_residency_barrier_rejects_nonquiet_host() -> None:
    module = _load_probe_module()
    model = SimpleNamespace(parameters=lambda: {})

    with pytest.raises(ValueError, match="quiet"):
        module.establish_glm52_residency(
            model,
            evaluate_parameters=lambda _parameters: None,
            synchronize=lambda: None,
            clear_cache=lambda: None,
            quiet_probe=lambda **_kwargs: {
                "enabled": True,
                "available": True,
                "quiet": False,
                "pageouts_delta": 1,
                "swapouts_delta": 0,
            },
            metric_snapshot=lambda: {"mlx_cache_bytes": 0},
        )


def test_residency_barrier_rejects_boolean_memory_counter() -> None:
    module = _load_probe_module()
    model = SimpleNamespace(parameters=lambda: {})

    with pytest.raises(ValueError, match="quiet"):
        module.establish_glm52_residency(
            model,
            evaluate_parameters=lambda _parameters: None,
            synchronize=lambda: None,
            clear_cache=lambda: None,
            quiet_probe=lambda **_kwargs: {
                "enabled": True,
                "available": True,
                "quiet": True,
                "pageouts_delta": False,
                "swapouts_delta": 0,
            },
            metric_snapshot=lambda: {"mlx_cache_bytes": 0},
        )


def test_residency_barrier_requires_cache_clear_to_reach_zero() -> None:
    module = _load_probe_module()
    model = SimpleNamespace(parameters=lambda: {})
    snapshots = iter(
        (
            {"mlx_cache_bytes": 2},
            {"mlx_cache_bytes": 1},
        )
    )

    with pytest.raises(ValueError, match="cache"):
        module.establish_glm52_residency(
            model,
            evaluate_parameters=lambda _parameters: None,
            synchronize=lambda: None,
            clear_cache=lambda: None,
            quiet_probe=lambda **_kwargs: pytest.fail(
                "quiet probe must not run after invalid cache accounting"
            ),
            metric_snapshot=lambda: next(snapshots),
        )


def test_memory_phase_clean_requires_exact_label_and_integer_zero_deltas() -> None:
    module = _load_probe_module()
    clean = {
        "label": "after_residency_quiet",
        "available": True,
        "pageouts_delta": 0,
        "swapouts_delta": 0,
    }
    assert module.glm52_memory_phase_is_clean(
        clean,
        label="after_residency_quiet",
    ) is True

    for field, value in (
        ("label", "after_generation"),
        ("available", False),
        ("pageouts_delta", 1),
        ("pageouts_delta", False),
        ("swapouts_delta", 1),
        ("swapouts_delta", False),
    ):
        dirty = dict(clean)
        dirty[field] = value
        assert module.glm52_memory_phase_is_clean(
            dirty,
            label="after_residency_quiet",
        ) is False


def test_residency_claims_reject_dirty_pre_generation_interval() -> None:
    module = _load_probe_module()

    def phase(label, *, swapouts_delta=0):
        return {
            "label": label,
            "available": True,
            "pageouts_delta": 0,
            "swapouts_delta": swapouts_delta,
        }

    claims = module.classify_glm52_residency_evidence(
        parameter_residency_established=True,
        system_wired_default=True,
        generation_warmup_memory_clean=True,
        cold_residency_memory=phase("after_residency_quiet"),
        pre_generation_memory=phase("before_generation", swapouts_delta=1),
        steady_state_generation_memory=phase("after_generation"),
    )

    assert claims == {
        "cold_residency_memory_clean": True,
        "pre_generation_memory_clean": False,
        "steady_state_generation_memory_clean": True,
        "warm_residency_proven": False,
        "production_residency_proven": False,
    }


def test_production_residency_rejects_dirty_generation_warmup() -> None:
    module = _load_probe_module()

    def phase(label):
        return {
            "label": label,
            "available": True,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        }

    claims = module.classify_glm52_residency_evidence(
        parameter_residency_established=True,
        system_wired_default=True,
        generation_warmup_memory_clean=False,
        cold_residency_memory=phase("after_residency_quiet"),
        pre_generation_memory=phase("before_generation"),
        steady_state_generation_memory=phase("after_generation"),
    )

    assert claims["warm_residency_proven"] is True
    assert claims["production_residency_proven"] is False


def test_residency_barrier_rejects_incomplete_active_payload() -> None:
    module = _load_probe_module()
    model = SimpleNamespace(parameters=lambda: {})

    with pytest.raises(ValueError, match="active"):
        module.establish_glm52_residency(
            model,
            evaluate_parameters=lambda _parameters: None,
            synchronize=lambda: None,
            clear_cache=lambda: None,
            quiet_probe=lambda **_kwargs: {
                "enabled": True,
                "available": True,
                "quiet": True,
                "pageouts_delta": 0,
                "swapouts_delta": 0,
            },
            metric_snapshot=lambda: {
                "mlx_cache_bytes": 0,
                "mlx_active_bytes": ACCEPTED_BYTES - 1,
            },
        )


def test_output_path_rejects_authority_roots_and_hardlink_aliases(
    tmp_path: Path,
) -> None:
    module = _load_probe_module()
    routed = tmp_path / "routed"
    non_vq = tmp_path / "non-vq"
    tokenizer = tmp_path / "tokenizer"
    for root in (routed, non_vq, tokenizer):
        root.mkdir()
    evidence = tmp_path / "composite.json"
    evidence.write_text("{}\n")

    with pytest.raises(ValueError, match="artifact|authority"):
        module._validated_output_path(
            routed / "production.json",
            authority_roots=(tokenizer, non_vq, routed),
            protected_files=(evidence,),
        )

    alias = tmp_path / "production.json"
    alias.hardlink_to(evidence)
    original = evidence.read_bytes()
    with pytest.raises(ValueError, match="aliases"):
        module._validated_output_path(
            alias,
            authority_roots=(tokenizer, non_vq, routed),
            protected_files=(evidence,),
        )
    assert evidence.read_bytes() == original

    routed_payload = routed / "layer-00003-gate_proj.safetensors"
    routed_payload.write_bytes(b"payload")
    artifact_alias = tmp_path / "artifact-alias.json"
    artifact_alias.hardlink_to(routed_payload)
    with pytest.raises(ValueError, match="aliases"):
        module._validated_output_path(
            artifact_alias,
            authority_roots=(tokenizer, non_vq, routed),
            protected_files=(evidence,),
        )


def test_atomic_writer_preserves_old_output_and_removes_partial_on_replace_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_probe_module()
    output = tmp_path / "production.json"
    output.write_text("old\n")
    monkeypatch.setattr(
        module.os,
        "replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(OSError, match="replace failed"):
        module._write_json_atomic(output, {"probe_pass": True})

    assert output.read_text() == "old\n"
    assert list(tmp_path.glob(".*.partial-*")) == []


def test_atomic_writer_rejects_replaced_output_parent(tmp_path: Path) -> None:
    module = _load_probe_module()
    parent = tmp_path / "evidence"
    parent.mkdir()
    output = parent / "production.json"
    identity = module._directory_identity(parent)
    original_parent = tmp_path / "original-evidence"
    parent.rename(original_parent)
    parent.mkdir()

    with pytest.raises(ValueError, match="output parent changed"):
        module._write_json_atomic(
            output,
            {"probe_pass": True},
            expected_parent_identity=identity,
        )

    assert not output.exists()
    assert not (original_parent / output.name).exists()


def test_main_writes_atomic_failure_evidence_and_returns_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_probe_module()
    roots = [tmp_path / name for name in ("tokenizer", "non-vq", "routed")]
    for root in roots:
        root.mkdir()
    output = tmp_path / "production.json"

    def fail_probe(**_kwargs):
        raise module.GLM52ProductionProbeError(
            "input_authentication",
            ValueError("bounded evidence is forbidden"),
            [{"label": "before_input_validation", "rss_bytes": 1}],
        )

    monkeypatch.setattr(module, "probe_glm52_production_generation", fail_probe)
    exit_code = module.main(
        [
            "--profile-path", str(tmp_path / "profile.yaml"),
            "--config-path", str(tmp_path / "config.json"),
            "--source-index-path", str(tmp_path / "index.json"),
            "--tokenizer-dir", str(roots[0]),
            "--tokenizer-readiness-json", str(tmp_path / "tokenizer.json"),
            "--family-policy-json", str(tmp_path / "policy.json"),
            "--non-vq-artifact-dir", str(roots[1]),
            "--non-vq-evidence-json", str(tmp_path / "non-vq.json"),
            "--routed-artifact-dir", str(roots[2]),
            "--composite-audit-json", str(tmp_path / "composite.json"),
            "--materialization-runs-jsonl", str(tmp_path / "runs.jsonl"),
            "--full-bind-preflight-json", str(tmp_path / "preflight.json"),
            "--model-id", MODEL_ID,
            "--revision", REVISION,
            "--prompt", "x",
            "--output-json", str(output),
        ]
    )

    assert exit_code == 1
    failure = json.loads(output.read_text())
    assert failure["probe_pass"] is False
    assert failure["failure"]["stage"] == "input_authentication"
    assert failure["production_generation_proven"] is False
    assert failure["parameter_residency_established"] is False
    assert "mlx_peak_reset_before_generation" not in failure
    assert failure["steady_state_generation_memory_clean"] is False
    assert failure["cold_residency_memory_clean"] is False
    assert failure["warm_residency_proven"] is False
    assert failure["production_residency_proven"] is False
    assert list(tmp_path.glob(".*.partial-*")) == []


def test_probe_success_is_one_greedy_token_with_honest_negative_claims() -> None:
    module = _load_probe_module()
    order: list[str] = []
    peak_state = {"generation_baseline_reset": False}
    validated = SimpleNamespace(
        tokenizer_dir=Path("/tokenizer"),
        tokenizer_readiness={
            "prompt": "x",
            "thinking_disabled_render": "<user>x<|assistant|><think></think>",
        },
        artifact_identity=SimpleNamespace(
            body={"whole_model_tensor_payload_bytes": ACCEPTED_BYTES},
            sha256="8" * 64,
        ),
        input_evidence_file_sha256={"composite_audit_json": "9" * 64},
        input_fingerprint=object(),
    )
    load_report = SimpleNamespace(
        non_vq_bind_report=SimpleNamespace(to_dict=lambda: {"loaded_count": 1272}),
        bound_sparse_layer_ids=tuple(range(3, 78)),
        dense_routed_parameter_names=(),
        unbound_vq_experts=False,
    )

    class Tokenizer:
        class Inner:
            vocab_size = 154_820
            pad_token_id = 154_820
            eos_token_id = 154_820

            def __len__(self):
                return 154_856

        _tokenizer = Inner()
        eos_token_ids = [154_820, 154_827, 154_829]

        def apply_chat_template(self, _messages, *, tokenize, **kwargs):
            assert kwargs == {
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            return [1, 2, 3] if tokenize else "<user>x<|assistant|><think></think>"

        def decode(self, token_ids):
            assert token_ids == [42]
            return " answer"

    def input_validator(**_kwargs):
        order.append("validate")
        return validated

    def tokenizer_factory(*_args, **_kwargs):
        order.append("tokenizer")
        return Tokenizer()

    def composite_loader(_validated, **_kwargs):
        order.append("load")
        return object(), load_report

    def residency_preparer(_model):
        order.append("resident")
        return {
            "parameter_residency_established": True,
            "cache_clear": {
                "before_memory": {"mlx_cache_bytes": 2_000},
                "after_memory": {"mlx_cache_bytes": 0},
                "released_bytes": 2_000,
            },
            "memory_quiet": {
                "enabled": True,
                "available": True,
                "quiet": True,
                "pageouts_delta": 0,
                "swapouts_delta": 0,
            },
            "resident_memory": {
                "mlx_active_bytes": ACCEPTED_BYTES,
                "mlx_peak_bytes": 0,
            },
        }

    def generation_record(token_ids, *, method, lookahead):
        assert token_ids == [1, 2, 3]
        return {
            "generated_token_ids": [42],
            "generated_token_count": 1,
            "logits_shape": [154_880],
            "logits_finite": True,
            "greedy_sampling": True,
            "generation_method": method,
            "lookahead_forward_scheduled": lookahead,
        }

    def generation_warmup_runner(_model, token_ids):
        order.append("warmup")
        return generation_record(
            token_ids,
            method="mlx_lm_generate_step",
            lookahead=True,
        )

    def generation_runner(_model, token_ids):
        order.append("generate")
        return generation_record(
            token_ids,
            method="direct_prefill_single_decode",
            lookahead=False,
        )

    def generation_warmup_cache_clearer():
        order.append("warmup_cache_clear")
        return {
            "before_memory": {"mlx_cache_bytes": 4_000},
            "after_memory": {"mlx_cache_bytes": 0},
            "released_bytes": 4_000,
        }

    def generation_warmup_quiet_preparer():
        order.append("warmup_quiet")
        return {
            "enabled": True,
            "available": True,
            "quiet": True,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        }

    def reset_generation_peak():
        order.append("reset_generation_peak")
        peak_state["generation_baseline_reset"] = True

    def memory_snapshot(label):
        return {
            "label": label,
            "available": True,
            "rss_bytes": 1,
            "mlx_peak_bytes": (
                7 if peak_state["generation_baseline_reset"] else 123
            ),
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        }

    payload = module.probe_glm52_production_generation(
        profile_path="profile.yaml",
        config_path="config.json",
        source_index_path="index.json",
        tokenizer_dir="tokenizer",
        tokenizer_readiness_json="tokenizer-ready.json",
        family_policy_json="policy.json",
        non_vq_artifact_dir="non-vq",
        non_vq_evidence_json="non-vq.json",
        routed_artifact_dir="routed",
        composite_audit_json="composite.json",
        materialization_runs_jsonl="runs.jsonl",
        full_bind_preflight_json="preflight.json",
        model_id=MODEL_ID,
        revision=REVISION,
        prompt="x",
        input_validator=input_validator,
        tokenizer_factory=tokenizer_factory,
        tokenizer_reauthenticator=lambda *_args, **_kwargs: order.append("reauth"),
        input_integrity_checker=lambda _fingerprint: order.append("integrity"),
        composite_loader=composite_loader,
        residency_preparer=residency_preparer,
        generation_runner=generation_runner,
        generation_warmup_runner=generation_warmup_runner,
        generation_warmup_cache_clearer=generation_warmup_cache_clearer,
        generation_warmup_quiet_preparer=(
            generation_warmup_quiet_preparer
        ),
        generation_peak_resetter=reset_generation_peak,
        memory_snapshot=memory_snapshot,
        wired_policy_probe=lambda: {
            "system_wired_default": True,
            "environment_overrides": {},
            "iogpu_wired_limit_mb": 0,
        },
    )

    assert order == [
        "validate",
        "integrity",
        "reauth",
        "tokenizer",
        "reauth",
        "integrity",
        "load",
        "integrity",
        "resident",
        "integrity",
        "warmup",
        "integrity",
        "warmup_cache_clear",
        "warmup_quiet",
        "integrity",
        "reset_generation_peak",
        "generate",
        "integrity",
    ]
    assert payload["probe_pass"] is True
    assert payload["production_binding_proven"] is True
    assert payload["production_generation_proven"] is True
    assert payload["generation_warmup_proven"] is True
    assert payload["generation_warmup_memory_clean"] is True
    assert payload["measured_generation_after_warmup"] is True
    assert payload["warmup_measured_token_match"] is True
    assert payload["generation_warmup"]["generated_token_ids"] == [42]
    assert (
        payload["generation_warmup"]["generation_method"]
        == "mlx_lm_generate_step"
    )
    assert payload["generation_method"] == "direct_prefill_single_decode"
    assert payload["lookahead_forward_scheduled"] is False
    assert payload["measured_generation_avoids_lookahead"] is True
    assert payload["generation_warmup_cache_clear"]["released_bytes"] == 4_000
    assert payload["generation_warmup_memory_quiet"]["quiet"] is True
    assert payload["generated_token_count"] == 1
    assert payload["accepted_whole_model_tensor_payload_bytes"] == ACCEPTED_BYTES
    assert payload["common_artifact_identity_sha256"] == "8" * 64
    assert payload["common_artifact_identity"] == validated.artifact_identity.body
    assert payload["input_fingerprint_reverified_after_generation"] is True
    assert payload["parameter_residency_established"] is True
    assert payload["residency_cache_clear"]["released_bytes"] == 2_000
    assert payload["resident_memory"]["mlx_active_bytes"] == ACCEPTED_BYTES
    assert payload["resident_memory_quiet"]["quiet"] is True
    assert payload["mlx_peak_reset_before_generation"] is True
    assert payload["cold_residency_mlx_peak_scope"] == "since_probe_start"
    assert (
        payload["steady_state_generation_mlx_peak_scope"]
        == "since_pre_generation_reset"
    )
    assert payload["cold_residency_memory"]["mlx_peak_bytes"] == 123
    assert payload["cold_residency_memory_clean"] is True
    assert payload["pre_generation_memory"]["mlx_peak_bytes"] == 7
    assert payload["pre_generation_memory_clean"] is True
    assert payload["steady_state_generation_memory"]["mlx_peak_bytes"] == 7
    assert payload["steady_state_generation_memory_clean"] is True
    assert payload["warm_residency_proven"] is True
    assert payload["production_residency_proven"] is True
    assert payload["long_context_indexshare_proven"] is False
    assert payload["quality_claim"] is False
    assert payload["speed_claim"] is False
    assert payload["phase_memory"]


def test_live_tokenizer_contract_rejects_runtime_identity_drift() -> None:
    module = _load_probe_module()

    class Inner:
        vocab_size = 154_819
        pad_token_id = 154_820
        eos_token_id = 154_820

        def __len__(self):
            return 154_856

    tokenizer = SimpleNamespace(
        _tokenizer=Inner(),
        eos_token_ids=[154_820, 154_827, 154_829],
    )
    readiness = {
        "prompt": "x",
        "thinking_disabled_render": "<user>x<|assistant|><think></think>",
    }

    with pytest.raises(ValueError, match="base vocabulary"):
        module.validate_glm52_live_tokenizer(
            tokenizer,
            authenticated_readiness=readiness,
            prompt="x",
            rendered_prompt="<user>x<|assistant|><think></think>",
        )


def test_input_fingerprint_rejects_authority_and_artifact_drift(
    tmp_path: Path,
) -> None:
    module = _load_probe_module()
    authority = tmp_path / "authority.json"
    authority.write_text('{"value": 1}\n')
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    payload = artifact / "group.safetensors"
    payload.write_bytes(b"abcd")

    fingerprint = module.snapshot_glm52_production_inputs(
        authority_files=(authority,),
        artifact_roots=(artifact,),
    )
    module.assert_glm52_production_inputs_unchanged(fingerprint)

    authority.write_text('{"value": 2}\n')
    with pytest.raises(ValueError, match="authority file changed"):
        module.assert_glm52_production_inputs_unchanged(fingerprint)

    authority.write_text('{"value": 1}\n')
    fingerprint = module.snapshot_glm52_production_inputs(
        authority_files=(authority,),
        artifact_roots=(artifact,),
    )
    original_stat = payload.stat()
    payload.write_bytes(b"wxyz")
    os.utime(
        payload,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    with pytest.raises(ValueError, match="artifact tree changed"):
        module.assert_glm52_production_inputs_unchanged(fingerprint)


def test_input_fingerprint_allows_authority_parent_sibling_creation(
    tmp_path: Path,
) -> None:
    module = _load_probe_module()
    authority = tmp_path / "authority.json"
    authority.write_text('{"value": 1}\n')
    artifact = tmp_path / "artifact"
    artifact.mkdir()

    fingerprint = module.snapshot_glm52_production_inputs(
        authority_files=(authority,),
        artifact_roots=(artifact,),
    )
    (tmp_path / "candidate-eval-cache.json").write_text("{}\n")

    module.assert_glm52_production_inputs_unchanged(fingerprint)


def test_input_fingerprint_rejects_same_size_authority_replacement(
    tmp_path: Path,
) -> None:
    module = _load_probe_module()
    authority = tmp_path / "authority.json"
    authority.write_text('{"value": 1}\n')
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    fingerprint = module.snapshot_glm52_production_inputs(
        authority_files=(authority,),
        artifact_roots=(artifact,),
    )

    replacement = tmp_path / "replacement.json"
    replacement.write_text('{"value": 2}\n')
    replacement.replace(authority)

    with pytest.raises(ValueError, match="authority file changed"):
        module.assert_glm52_production_inputs_unchanged(fingerprint)


def test_input_fingerprint_rejects_authority_parent_replacement(
    tmp_path: Path,
) -> None:
    module = _load_probe_module()
    parent = tmp_path / "authority"
    parent.mkdir()
    authority = parent / "authority.json"
    authority.write_text('{"value": 1}\n')
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    fingerprint = module.snapshot_glm52_production_inputs(
        authority_files=(authority,),
        artifact_roots=(artifact,),
    )
    parent_only_fingerprint = replace(
        fingerprint,
        authority_file_sha256={},
        authority_file_state={},
    )

    old_parent = tmp_path / "old-authority"
    parent.rename(old_parent)
    parent.mkdir()
    (parent / authority.name).write_text('{"value": 1}\n')

    with pytest.raises(ValueError, match="authority parent identity changed"):
        module.assert_glm52_production_inputs_unchanged(parent_only_fingerprint)


def test_default_wired_policy_recomputes_environment_and_sysctl(
    monkeypatch,
) -> None:
    module = _load_probe_module()
    monkeypatch.setattr(module.platform, "system", lambda: "Darwin")
    monkeypatch.delenv("GLM_MLX_WIRED_LIMIT_GB", raising=False)
    monkeypatch.delenv("GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB", raising=False)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="0\n"),
    )

    assert module.audit_default_wired_policy() == {
        "platform": "Darwin",
        "environment_overrides": {},
        "iogpu_wired_limit_mb": 0,
        "iogpu_wired_limit_available": True,
        "system_wired_default": True,
    }

    monkeypatch.setenv("GLM_MLX_WIRED_LIMIT_GB", "120")
    assert module.audit_default_wired_policy()["system_wired_default"] is False
    monkeypatch.delenv("GLM_MLX_WIRED_LIMIT_GB")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="122880\n"),
    )
    assert module.audit_default_wired_policy()["system_wired_default"] is False


def test_probe_rejects_custom_wired_policy_before_input_or_model_work() -> None:
    module = _load_probe_module()
    calls: list[str] = []

    with pytest.raises(module.GLM52ProductionProbeError, match="wired") as error:
        module.probe_glm52_production_generation(
            profile_path="profile.yaml",
            config_path="config.json",
            source_index_path="index.json",
            tokenizer_dir="tokenizer",
            tokenizer_readiness_json="tokenizer-ready.json",
            family_policy_json="policy.json",
            non_vq_artifact_dir="non-vq",
            non_vq_evidence_json="non-vq.json",
            routed_artifact_dir="routed",
            composite_audit_json="composite.json",
            materialization_runs_jsonl="runs.jsonl",
            full_bind_preflight_json="preflight.json",
            model_id=MODEL_ID,
            revision=REVISION,
            prompt="x",
            input_validator=lambda **_kwargs: calls.append("validate"),
            wired_policy_probe=lambda: {
                "system_wired_default": False,
                "environment_overrides": {"GLM_MLX_WIRED_LIMIT_GB": "120"},
                "iogpu_wired_limit_mb": 122_880,
            },
            memory_snapshot=lambda label: {"label": label, "rss_bytes": 1},
        )

    assert error.value.stage == "memory_policy"
    assert error.value.phase_memory[0]["label"] == "before_memory_policy"
    assert calls == []
