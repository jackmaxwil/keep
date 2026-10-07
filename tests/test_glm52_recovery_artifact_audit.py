from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

from keep.vq.e8 import e8_1bit_packed, e8p_packed_abs_grid
import keep.convert.glm52_recovery_materialize as recovery_materialize
from keep.io.schema import codebook_metadata_for_bits
from ramp.models.profiles import ModelProfile, get_profile


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
MODEL_ID = "0xSero/glm-5.2-reap-504B-v2"
REVISION = "6c9241aa05fb243a0edb7c804c213ec1cf5c920d"
PROFILE = "glm52-reap-504b-v2"
CONFIG_SHA256 = "c" * 64
INDEX_SHA256 = "d" * 64
STATS_MANIFEST_SHA256 = "e" * 64
RECOVERY_LEVER = "selection_diagonal_hessian_importance_weighted_reround_v1"
SOURCE_VERIFICATION = {
    "source_blob_inventory_sha256": "7" * 64,
    "routed_source_blob_inventory_sha256": "8" * 64,
    "shard_count": 1,
}
NON_ROUTED_PAYLOAD = 37_121_488_608
CANONICAL_PAYLOAD_LIMIT = 112_000_000_000
MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/keep/validate/glm52_recovery_artifact.py"
)


@dataclass(frozen=True)
class _Fixture:
    root: Path
    seed: Path
    profile: ModelProfile
    seed_manifest_sha256: str
    manifest_path: Path
    replacement_names: tuple[str, ...]
    accepted_composite_audit_path: Path
    accepted_composite_audit_sha256: str


def _profile() -> ModelProfile:
    return replace(
        get_profile(PROFILE),
        name=PROFILE,
        hf_model_id=MODEL_ID,
        revision=REVISION,
        hidden_size=16,
        moe_intermediate_size=8,
        num_experts=2,
        group_size_policy={"gate": 8, "up": 8, "down": 8},
    )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _filename(layer: int, projection: str) -> str:
    return f"layer-{layer:05d}-{projection}.safetensors"


def _prefix(layer: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.switch_mlp.{projection}"


def _dims(profile: ModelProfile, projection: str) -> tuple[int, int]:
    if projection in {"gate_proj", "up_proj"}:
        return profile.hidden_size, profile.moe_intermediate_size
    return profile.moe_intermediate_size, profile.hidden_size


def _lineage(profile: ModelProfile) -> dict[str, str]:
    return {
        "source_model_id": profile.hf_model_id,
        "source_revision": profile.revision,
        "source_config_sha256": CONFIG_SHA256,
        "source_index_sha256": INDEX_SHA256,
        "source_profile": profile.name,
    }


def _lever_provenance() -> dict[str, object]:
    return {
        "lever": RECOVERY_LEVER,
        "scale_estimator": "importance_weighted_least_squares",
        "rounding_objective": "selection_diagonal_hessian_weighted_squared_error",
        "stats_manifest_sha256": STATS_MANIFEST_SHA256,
        "holdout_used_for_tuning": False,
        "report_used_for_tuning": False,
        "source_lineage": _lineage(_profile()),
        "source_verification": SOURCE_VERIFICATION,
    }


def _recovery_policy(profile: ModelProfile) -> dict[str, str]:
    return {
        **_lineage(profile),
        "source_weight_encoding": "modelopt_nvfp4",
        "source_decoder": "modelopt_nvfp4_v1",
        "decoded_expert_working_set": "one_per_worker",
        "scale_estimator": "importance_weighted_least_squares",
        "recovery_lever": RECOVERY_LEVER,
        "rounding_objective": "selection_diagonal_hessian_weighted_squared_error",
    }


def _write_group(
    path: Path,
    *,
    profile: ModelProfile,
    layer: int,
    projection: str,
    code_bits: int,
    replacement: bool,
    wrong_codebook: bool = False,
    metadata_code_bits: int | None = None,
) -> None:
    input_dim, output_dim = _dims(profile, projection)
    prefix = _prefix(layer, projection)
    codebook = (
        e8_1bit_packed().copy()
        if code_bits == 8
        else e8p_packed_abs_grid().copy()
    )
    if wrong_codebook:
        codebook[0] ^= np.uint32(1)
    metadata_bits = code_bits if metadata_code_bits is None else metadata_code_bits
    codebook_name, codebook_sha256 = codebook_metadata_for_bits(metadata_bits)
    policy: dict[str, str] = {
        **_lineage(profile),
        "source_weight_encoding": "modelopt_nvfp4",
        "source_decoder": "modelopt_nvfp4_v1",
        "decoded_expert_working_set": "one_per_worker",
        "scale_estimator": "max_abs",
    }
    if replacement:
        policy = _recovery_policy(profile)
    metadata: dict[str, str] = {
        "quantization_config": json.dumps(
            {
                "quant_method": "mlx_vq_e8",
                "version": 1,
                "default_code_bits": metadata_bits,
                "default_group_size": 8,
                "codebook": {
                    "name": codebook_name,
                    "dtype": "uint32",
                    "entries": 256,
                    "sha256": codebook_sha256,
                },
                "policy": policy,
            },
            sort_keys=True,
        )
    }
    if replacement:
        provenance = _lever_provenance()
        metadata["glm52_recovery_provenance"] = json.dumps(
            provenance, sort_keys=True
        )
    save_file(
        {
            f"{prefix}.codes": np.zeros(
                (profile.num_experts, output_dim, input_dim // 8),
                dtype=np.uint8 if code_bits == 8 else np.uint16,
            ),
            f"{prefix}.scales": np.ones(
                (profile.num_experts, output_dim, input_dim // 8),
                dtype=np.float16,
            ),
            "model.vq_codebook.e8": codebook,
        },
        path,
        metadata=metadata,
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_accepted_composite_audit(
    path: Path,
    *,
    profile: ModelProfile,
    audit_pass: bool = True,
    artifact_integrity_pass: bool = True,
    whole_payload: int = 98_433_923_808,
    routed_payload: int = 61_312_204_800,
    model_id: str | None = None,
    seed_manifest_sha256: str | None = None,
) -> str:
    _write_json(
        path,
        {
            "audit_pass": audit_pass,
            "artifact_integrity_pass": artifact_integrity_pass,
            "model_id": model_id or profile.hf_model_id,
            "source_revision": profile.revision,
            "config_sha256": CONFIG_SHA256,
            "index_sha256": INDEX_SHA256,
            "actual_whole_model_tensor_payload_bytes": whole_payload,
            "actual_routed_payload_bytes": routed_payload,
            "main_non_routed_tensor_payload_bytes": 37_121_488_608,
            "manifest_sha256": (
                seed_manifest_sha256
                or recovery_materialize.PINNED_ACCEPTED_SEED_MANIFEST_SHA256
            ),
            "group_set_sha256": recovery_materialize.PINNED_ACCEPTED_ROUTED_GROUP_SET_SHA256,
            "non_vq_evidence_sha256": recovery_materialize.PINNED_ACCEPTED_NON_VQ_EVIDENCE_SHA256,
            "expected_group_keys": [
                f"{layer}:{projection}"
                for layer in range(3, 78)
                for projection in PROJECTIONS
            ],
            "ready_group_keys": [
                f"{layer}:{projection}"
                for layer in range(3, 78)
                for projection in PROJECTIONS
            ],
            "checks": recovery_materialize.PINNED_ACCEPTED_AUDIT_CHECKS,
            "byte_identity_verified": True,
            "full_routed_artifact_ready": True,
            "main_routed_tensor_count": 151_200,
            "main_non_routed_tensor_count": 1_194,
            "non_vq_package_audit": {
                "audit_pass": True,
                "checks": recovery_materialize.PINNED_ACCEPTED_NON_VQ_AUDIT_CHECKS,
                "manifest_sha256": recovery_materialize.PINNED_ACCEPTED_NON_VQ_MANIFEST_SHA256,
                "package_set_sha256": recovery_materialize.PINNED_ACCEPTED_NON_VQ_PACKAGE_SET_SHA256,
                "retained_tensor_count": 1_194,
                "shard_count": 9,
                "parameter_count": 18_560_731_704,
                "tensor_payload_bytes": 37_121_488_608,
            },
        },
    )
    return _sha256(path)


def _reattest(fixture: _Fixture) -> None:
    manifest = json.loads(fixture.manifest_path.read_text())
    manifest.pop("manifest_body_sha256", None)
    manifest["manifest_body_sha256"] = _canonical_sha256(manifest)
    _write_json(fixture.manifest_path, manifest)


def _build_fixture(
    tmp_path: Path,
    *,
    upgraded_layers: tuple[int, ...] = (77,),
    root_name: str = "candidate",
    schema_version: int = 2,
    replacement_code_bits: int | None = None,
) -> _Fixture:
    profile = _profile()
    resolved_replacement_code_bits = (
        replacement_code_bits
        if replacement_code_bits is not None
        else (16 if schema_version == 2 else 8)
    )
    seed = tmp_path / f"{root_name}-seed"
    root = tmp_path / root_name
    recovered = root / "recovered-groups"
    artifact = root / "artifact"
    seed.mkdir()
    recovered.mkdir(parents=True)
    artifact.mkdir()

    seed_records: list[dict[str, object]] = []
    replacements: list[dict[str, object]] = []
    replacement_names: list[str] = []
    for layer in range(3, 78):
        for projection in PROJECTIONS:
            filename = _filename(layer, projection)
            seed_path = seed / filename
            _write_group(
                seed_path,
                profile=profile,
                layer=layer,
                projection=projection,
                code_bits=8,
                replacement=False,
            )
            seed_records.append(
                {
                    "layer": layer,
                    "projection": projection,
                    "artifact_path": filename,
                    "artifact_bytes": seed_path.stat().st_size,
                    "artifact_sha256": _sha256(seed_path),
                    "codes_dtype": "uint8",
                    "codes_name": f"model.layers.{layer}.mlp.switch_mlp.{projection}.codes",
                    "codes_shape": [profile.num_experts, 4, 1],
                    "cross_shard_bundle_count": 0,
                    "decoded_expert_bytes": 128,
                    "expert_count": profile.num_experts,
                    "scales_dtype": "float16",
                    "scales_name": f"model.layers.{layer}.mlp.switch_mlp.{projection}.scales",
                    "scales_shape": [profile.num_experts, 4, 1],
                    "source_bundle_members": profile.num_experts,
                    "source_bundles": profile.num_experts,
                    "source_shards": ["model-00001.safetensors"],
                    "status": "ready",
                }
            )
            link_source = seed_path
            if layer in upgraded_layers:
                code_bits = resolved_replacement_code_bits
                replacement_path = recovered / filename
                _write_group(
                    replacement_path,
                    profile=profile,
                    layer=layer,
                    projection=projection,
                    code_bits=code_bits,
                    replacement=True,
                )
                codebook_name, codebook_sha256 = codebook_metadata_for_bits(code_bits)
                with safe_open(replacement_path, framework="np") as handle:
                    tensor_payload_bytes = sum(
                        handle.get_tensor(name).nbytes for name in handle.keys()
                    )
                replacements.append(
                    {
                        "layer": layer,
                        "projection": projection,
                        "status": "materialized",
                        "artifact_path": filename,
                        "artifact_bytes": replacement_path.stat().st_size,
                        "artifact_sha256": _sha256(replacement_path),
                        "source_lineage": _lineage(profile),
                        "recovery_policy": _recovery_policy(profile),
                        "lever_provenance": _lever_provenance(),
                        **(
                            {
                                "code_bits": code_bits,
                                "codes_dtype": "uint8" if code_bits == 8 else "uint16",
                                "codebook_name": codebook_name,
                                "codebook_sha256": codebook_sha256,
                                "tensor_payload_bytes": tensor_payload_bytes,
                                "zero_importance_policy": (
                                    "preserve_seed_expert_bytes"
                                    if code_bits == 8
                                    else "source_rtn_e8p"
                                ),
                            }
                            if schema_version == 2
                            else {}
                        ),
                    }
                )
                replacement_names.append(filename)
                link_source = replacement_path
            os.symlink(os.path.relpath(link_source, artifact), artifact / filename)

    selected_group_keys = [
        f"{layer}:{projection}"
        for layer in range(3, 78)
        for projection in PROJECTIONS
    ]
    seed_manifest: dict[str, object] = {
        "artifact_total_bytes": sum(int(record["artifact_bytes"]) for record in seed_records),
        "code_bits": 8,
        "codebook_name": "quip_e8",
        "codebook_sha256": "1" * 64,
        "schema_version": 1,
        "record_type": "glm52_modelopt_nvfp4_materialization_manifest",
        "profile": profile.name,
        "model_id": profile.hf_model_id,
        "source_revision": profile.revision,
        "config_sha256": CONFIG_SHA256,
        "dense_checkpoint_written": False,
        "full_group_coverage": True,
        "group_size": 512,
        "index_sha256": INDEX_SHA256,
        "materialization_blocked": False,
        "materialization_blockers": [],
        "materialization_scope": "full",
        "materialization_status": "glm52_modelopt_nvfp4_groups_ready",
        "planned_vq_groups": 225,
        "peak_decoded_expert_bytes": 128,
        "ready_vq_groups": 225,
        "scale_estimator": "max_abs",
        "selected_group_keys": selected_group_keys,
        "selected_vq_groups": 225,
        "skipped_vq_groups": 0,
        "source_bundle_members": 450,
        "source_bundles": 450,
        "source_decoder": "modelopt_nvfp4_v1",
        "source_weight_encoding": "modelopt_nvfp4",
        "working_set_policy": "one_decoded_expert_per_worker_v1",
        "groups": seed_records,
    }
    seed_manifest_path = seed / "conversion-manifest.json"
    _write_json(seed_manifest_path, seed_manifest)
    seed_manifest_sha256 = _sha256(seed_manifest_path)
    recovery_materialize.PINNED_ACCEPTED_SEED_MANIFEST_SHA256 = seed_manifest_sha256
    seed_routed_payload = 0
    for record in seed_records:
        with safe_open(seed / str(record["artifact_path"]), framework="np") as handle:
            seed_routed_payload += sum(
                handle.get_tensor(name).nbytes for name in handle.keys()
            )
    accepted_composite_audit_path = tmp_path / f"{root_name}-accepted-composite.json"
    accepted_composite_audit_sha256 = _write_accepted_composite_audit(
        accepted_composite_audit_path,
        profile=profile,
        seed_manifest_sha256=seed_manifest_sha256,
    )

    routed_payload = 0
    for link in artifact.iterdir():
        # The audit must derive this value itself.  This fixture claim is only
        # present so laundering it can be pinned by a negative test.
        with safe_open(link, framework="np") as handle:
            routed_payload += sum(handle.get_tensor(name).nbytes for name in handle.keys())
    routed_codes_scales_bytes = routed_payload - 225 * 256 * np.dtype(np.uint32).itemsize
    routed_codebook_bytes = 225 * 256 * np.dtype(np.uint32).itemsize
    manifest: dict[str, object] = {
        "schema_version": schema_version,
        "record_type": "glm52_recovery_conversion_manifest",
        "status": "complete",
        "resumable": True,
        "selected_groups": [
            f"{record['layer']}:{record['projection']}" for record in replacements
        ],
        "source_lineage": _lineage(profile),
        "source_verification": SOURCE_VERIFICATION,
        "expected_full_source_blob_inventory_sha256": SOURCE_VERIFICATION[
            "source_blob_inventory_sha256"
        ],
        "expected_routed_source_blob_inventory_sha256": SOURCE_VERIFICATION[
            "routed_source_blob_inventory_sha256"
        ],
        "stats_manifest_sha256": STATS_MANIFEST_SHA256,
        "accepted_composite_audit_sha256": accepted_composite_audit_sha256,
        "recovery_policy": _recovery_policy(profile),
        "lever_provenance": _lever_provenance(),
        "groups": replacements,
        **(
            {
                "rate_policy": {
                    "default_code_bits": 8,
                    "layer_code_bits": {
                        str(layer): 16
                        for layer in sorted(upgraded_layers)
                        if resolved_replacement_code_bits == 16
                    },
                    "complete_layer_rates_required": True,
                }
            }
            if schema_version == 2
            else {}
        ),
        "mixed_artifact": {
            "seed_artifact_dir": str(seed.resolve()),
            "seed_manifest_sha256": seed_manifest_sha256,
            "output_dir": str(artifact.resolve()),
            "recovered_groups_dir": str(recovered.resolve()),
            "replacement_count": len(replacements),
            "inherited_group_count": 225 - len(replacements),
        },
        "accounting": (
            {
                "routed_codes_scales_bytes": routed_codes_scales_bytes,
                "routed_codebook_bytes": routed_codebook_bytes,
                "main_non_routed_tensor_payload_bytes": NON_ROUTED_PAYLOAD,
                "logical_whole_model_tensor_payload_bytes": (
                    NON_ROUTED_PAYLOAD
                    + routed_codes_scales_bytes
                    + routed_codebook_bytes
                ),
                "incremental_disk_bytes": sum(
                    (recovered / name).stat().st_size for name in replacement_names
                ),
                "logical_payload_limit_bytes": 112_000_000_000,
            }
            if schema_version == 2
            else {
                "non_routed_tensor_payload_bytes": NON_ROUTED_PAYLOAD,
                "logical_whole_model_tensor_payload_bytes": NON_ROUTED_PAYLOAD
                + routed_payload,
                "incremental_disk_bytes": sum(
                    (recovered / name).stat().st_size for name in replacement_names
                ),
                "logical_payload_limit_bytes": 112_000_000_000,
            }
        ),
    }
    manifest["manifest_body_sha256"] = _canonical_sha256(manifest)
    manifest_path = root / "conversion-manifest.json"
    _write_json(manifest_path, manifest)
    return _Fixture(
        root=root,
        seed=seed,
        profile=profile,
        seed_manifest_sha256=seed_manifest_sha256,
        manifest_path=manifest_path,
        replacement_names=tuple(replacement_names),
        accepted_composite_audit_path=accepted_composite_audit_path,
        accepted_composite_audit_sha256=accepted_composite_audit_sha256,
    )


def _audit(
    fixture: _Fixture,
):
    name = "glm52_recovery_artifact_under_test"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module.audit_glm52_recovery_mixed_artifact(
        fixture.root,
        profile=fixture.profile,
        expected_seed_manifest_sha256=fixture.seed_manifest_sha256,
        expected_stats_manifest_sha256=STATS_MANIFEST_SHA256,
        expected_full_source_blob_inventory_sha256=SOURCE_VERIFICATION[
            "source_blob_inventory_sha256"
        ],
        expected_routed_source_blob_inventory_sha256=SOURCE_VERIFICATION[
            "routed_source_blob_inventory_sha256"
        ],
        expected_recovery_lever=RECOVERY_LEVER,
        expected_recovery_policy=_recovery_policy(fixture.profile),
        accepted_composite_audit_json=fixture.accepted_composite_audit_path,
        expected_composite_audit_sha256=fixture.accepted_composite_audit_sha256,
    )


def _reseed_from_recovery_artifact(
    fixture: _Fixture,
    *,
    parent: _Fixture,
    parent_audit_path: Path,
    link_to_parent_resolved_groups: bool = False,
) -> None:
    parent_audit = _audit(parent)
    parent_audit_payload = {
        **parent_audit.to_dict(),
        "raw_recovery_manifest_sha256": _sha256(parent.manifest_path),
    }
    _write_json(parent_audit_path, parent_audit_payload)

    parent_groups = {
        group.group_key: group for group in parent_audit.groups
    }
    replacement_names = set(fixture.replacement_names)
    for link in (fixture.root / "artifact").iterdir():
        if link.name in replacement_names:
            continue
        link.unlink()
        layer = int(link.name.split("-")[1])
        projection = link.name.removeprefix(
            f"layer-{layer:05d}-"
        ).removesuffix(".safetensors")
        parent_target = parent.root / "artifact" / link.name
        if link_to_parent_resolved_groups:
            parent_group = parent_groups[f"{layer}:{projection}"]
            parent_target = parent_target.resolve(strict=True)
            assert _sha256(parent_target) == parent_group.artifact_sha256
        os.symlink(os.path.relpath(parent_target, link.parent), link)

    manifest = json.loads(fixture.manifest_path.read_text())
    parent_artifact_root = str((parent.root / "artifact").resolve())
    manifest["mixed_artifact"] = {
        "seed_artifact_dir": parent_artifact_root,
        "output_dir": str((fixture.root / "artifact").resolve()),
        "recovered_groups_dir": str(
            (fixture.root / "recovered-groups").resolve()
        ),
        "replacement_count": len(fixture.replacement_names),
        "inherited_group_count": 225 - len(fixture.replacement_names),
        "seed_kind": "audited_recovery",
        "seed_recovery_manifest_sha256": _sha256(parent.manifest_path),
        "seed_recovery_audit_sha256": _sha256(parent_audit_path),
        "seed_candidate_identity_sha256": parent_audit.candidate_identity_sha256,
        "accepted_baseline_seed_manifest_sha256": fixture.seed_manifest_sha256,
        "parent_artifact_root": parent_artifact_root,
    }
    manifest.pop("manifest_body_sha256")
    manifest["manifest_body_sha256"] = _canonical_sha256(manifest)
    _write_json(fixture.manifest_path, manifest)


@pytest.mark.parametrize("layer_count", (8, 16))
def test_schema_v2_accepts_synthetic_worst_layer_rate_sets(
    tmp_path: Path,
    layer_count: int,
) -> None:
    upgraded = tuple(range(78 - layer_count, 78))
    fixture = _build_fixture(
        tmp_path,
        upgraded_layers=upgraded,
        root_name=f"worst-{layer_count}",
    )

    audit = _audit(fixture)

    assert audit.complete_replacement_layer_ids == upgraded
    assert audit.replacement_group_count == layer_count * 3
    assert audit.routed_codes_scales_bytes + audit.routed_codebook_bytes == (
        audit.routed_tensor_payload_bytes
    )
    assert audit.logical_whole_model_tensor_payload_bytes == (
        audit.main_non_routed_tensor_payload_bytes
        + audit.routed_codes_scales_bytes
        + audit.routed_codebook_bytes
    )


def test_accepted_e8_baseline_accounting_reconciles_without_double_count() -> None:
    assert 61_312_204_800 + 230_400 + NON_ROUTED_PAYLOAD == 98_433_923_808


@pytest.mark.parametrize("code_bits", (8, 16))
def test_schema_v2_allows_authenticated_e8_and_e8p_replacements(
    tmp_path: Path,
    code_bits: int,
) -> None:
    fixture = _build_fixture(
        tmp_path,
        root_name=f"replacement-e{code_bits}",
        replacement_code_bits=code_bits,
    )

    audit = _audit(fixture)

    replacement_rates = {
        group.code_bits
        for group in audit.groups
        if group.classification == "replacement"
    }
    assert replacement_rates == {code_bits}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("code_bits", 8, "code bits|rate"),
        ("codes_dtype", "uint8", "codes dtype|code bits"),
        ("codebook_name", "quip_e8", "codebook"),
        ("codebook_sha256", "0" * 64, "codebook"),
        ("tensor_payload_bytes", 1, "tensor payload"),
        ("zero_importance_policy", "reround_without_evidence", "zero importance"),
    ),
)
def test_schema_v2_rejects_replacement_manifest_claim_mismatch(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    fixture = _build_fixture(tmp_path)
    manifest = json.loads(fixture.manifest_path.read_text())
    manifest["groups"][0][field] = value
    _write_json(fixture.manifest_path, manifest)
    _reattest(fixture)

    with pytest.raises(ValueError, match=message):
        _audit(fixture)


def test_schema_v2_rejects_rate_policy_not_derived_from_authenticated_groups(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    manifest = json.loads(fixture.manifest_path.read_text())
    manifest["rate_policy"]["layer_code_bits"] = {}
    _write_json(fixture.manifest_path, manifest)
    _reattest(fixture)

    with pytest.raises(ValueError, match="rate policy|rate_policy|authenticated tensors"):
        _audit(fixture)


def test_schema_v2_rejects_noncanonical_embedded_codebook_bytes(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    name = fixture.replacement_names[0]
    layer = int(name.split("-")[1])
    projection = name.removeprefix(f"layer-{layer:05d}-").removesuffix(
        ".safetensors"
    )
    replacement = fixture.root / "recovered-groups" / name
    _write_group(
        replacement,
        profile=fixture.profile,
        layer=layer,
        projection=projection,
        code_bits=16,
        replacement=True,
        wrong_codebook=True,
    )
    manifest = json.loads(fixture.manifest_path.read_text())
    record = next(
        item for item in manifest["groups"] if item["artifact_path"] == name
    )
    record["artifact_bytes"] = replacement.stat().st_size
    record["artifact_sha256"] = _sha256(replacement)
    _write_json(fixture.manifest_path, manifest)
    _reattest(fixture)

    with pytest.raises(ValueError, match="codebook.*canonical|canonical.*codebook"):
        _audit(fixture)


def test_candidate_identity_changes_with_authenticated_raw_group_bytes(
    tmp_path: Path,
) -> None:
    original = _build_fixture(tmp_path, root_name="identity-original")
    mutated = _build_fixture(tmp_path, root_name="identity-mutated")
    target = mutated.root / "recovered-groups" / mutated.replacement_names[0]
    with safe_open(target, framework="np") as handle:
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
        metadata = dict(handle.metadata() or {})
    codes_name = next(name for name in tensors if name.endswith(".codes"))
    tensors[codes_name] = np.ones_like(tensors[codes_name])
    save_file(tensors, target, metadata=metadata)
    manifest = json.loads(mutated.manifest_path.read_text())
    record = next(
        item
        for item in manifest["groups"]
        if item["artifact_path"] == target.name
    )
    record["artifact_bytes"] = target.stat().st_size
    record["artifact_sha256"] = _sha256(target)
    _write_json(mutated.manifest_path, manifest)
    _reattest(mutated)

    assert _audit(original).candidate_identity_sha256 != _audit(
        mutated
    ).candidate_identity_sha256


def test_logical_payload_budget_accepts_boundary_and_rejects_one_byte_over() -> None:
    name = "glm52_recovery_budget_boundary_under_test"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    module._validate_logical_payload_budget(CANONICAL_PAYLOAD_LIMIT)
    with pytest.raises(ValueError, match="112|budget|limit"):
        module._validate_logical_payload_budget(CANONICAL_PAYLOAD_LIMIT + 1)


def test_schema_v1_accepts_only_uniform_e8_recovery_candidates(
    tmp_path: Path,
) -> None:
    legacy_e8 = _build_fixture(
        tmp_path,
        root_name="legacy-e8",
        schema_version=1,
        replacement_code_bits=8,
    )
    assert _audit(legacy_e8).audit_pass is True

    legacy_e8p = _build_fixture(
        tmp_path,
        root_name="legacy-e8p",
        schema_version=1,
        replacement_code_bits=16,
    )
    with pytest.raises(ValueError, match="schema.?v?1|uniform.*E8|E8P|code bits"):
        _audit(legacy_e8p)


def test_audit_accepts_exact_225_group_mixed_artifact_and_reports_identity(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    manifest_bytes = fixture.manifest_path.read_bytes()

    audit = _audit(fixture)

    assert fixture.manifest_path.read_bytes() == manifest_bytes
    assert audit.audit_pass is True
    assert audit.group_count == 225
    assert audit.replacement_group_count == 3
    assert audit.inherited_group_count == 222
    assert audit.complete_replacement_layer_ids == (77,)
    assert audit.logical_whole_model_tensor_payload_bytes > NON_ROUTED_PAYLOAD
    assert audit.incremental_disk_bytes > 0
    assert audit.logical_whole_model_tensor_payload_bytes != audit.incremental_disk_bytes
    assert len(audit.candidate_identity_sha256) == 64
    assert audit.to_dict()["candidate_identity_sha256"] == audit.candidate_identity_sha256


def test_audit_accepts_two_hop_recovery_seeded_parent_links_with_correct_hashes(
    tmp_path: Path,
) -> None:
    parent = _build_fixture(
        tmp_path,
        root_name="parent-recovery",
        upgraded_layers=(77,),
    )
    fixture = _build_fixture(tmp_path, root_name="recovery-seeded")
    parent_audit_path = tmp_path / "parent-recovery-audit.json"
    _reseed_from_recovery_artifact(
        fixture,
        parent=parent,
        parent_audit_path=parent_audit_path,
        link_to_parent_resolved_groups=True,
    )

    audit = _audit(fixture)

    assert audit.audit_pass is True
    assert audit.seed_manifest_sha256 == fixture.seed_manifest_sha256
    audit.verify_current_identity()


def test_recovery_seeded_audit_rejects_link_through_unauthenticated_directory(
    tmp_path: Path,
) -> None:
    parent = _build_fixture(tmp_path, root_name="unauth-parent")
    fixture = _build_fixture(tmp_path, root_name="unauth-child")
    parent_audit_path = tmp_path / "unauth-parent-audit.json"
    _reseed_from_recovery_artifact(
        fixture,
        parent=parent,
        parent_audit_path=parent_audit_path,
        link_to_parent_resolved_groups=True,
    )
    name = _filename(3, "gate_proj")
    link = fixture.root / "artifact" / name
    authenticated_target = link.resolve(strict=True)
    unauthenticated = tmp_path / "unauthenticated"
    unauthenticated.mkdir()
    os.symlink(authenticated_target, unauthenticated / name)
    link.unlink()
    os.symlink(os.path.relpath(unauthenticated / name, link.parent), link)

    with pytest.raises(ValueError, match="alias|authenticated roots"):
        _audit(fixture)


def test_parent_alias_roots_are_not_accepted_for_classic_manifest_shape(
    tmp_path: Path,
) -> None:
    parent = _build_fixture(tmp_path, root_name="classic-parent")
    fixture = _build_fixture(tmp_path, root_name="classic-child")
    name = _filename(3, "gate_proj")
    link = fixture.root / "artifact" / name
    link.unlink()
    parent_target = (parent.root / "artifact" / name).resolve(strict=True)
    os.symlink(os.path.relpath(parent_target, link.parent), link)

    with pytest.raises(ValueError, match="alias|authenticated roots"):
        _audit(fixture)


@pytest.mark.parametrize("mutation", ("mixed", "unknown"))
def test_audit_rejects_nonexact_recovery_seeded_field_families(
    tmp_path: Path,
    mutation: str,
) -> None:
    parent = _build_fixture(tmp_path, root_name=f"parent-{mutation}")
    fixture = _build_fixture(tmp_path, root_name=f"child-{mutation}")
    parent_audit_path = tmp_path / f"parent-{mutation}-audit.json"
    _reseed_from_recovery_artifact(
        fixture,
        parent=parent,
        parent_audit_path=parent_audit_path,
    )
    manifest = json.loads(fixture.manifest_path.read_text())
    if mutation == "mixed":
        manifest["mixed_artifact"]["seed_manifest_sha256"] = (
            fixture.seed_manifest_sha256
        )
    else:
        manifest["mixed_artifact"]["unknown_parent_claim"] = True
    manifest.pop("manifest_body_sha256")
    manifest["manifest_body_sha256"] = _canonical_sha256(manifest)
    _write_json(fixture.manifest_path, manifest)

    with pytest.raises(ValueError, match="mixed_artifact.*exact fields"):
        _audit(fixture)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("parent_hash", "parent recovery manifest.*SHA-256"),
        ("audit_pass", "recovery seed audit did not pass"),
        ("candidate_identity", "candidate identity"),
    ),
)
def test_audit_rejects_recovery_seed_chain_authentication_failures(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    parent = _build_fixture(tmp_path, root_name=f"chain-parent-{mutation}")
    fixture = _build_fixture(tmp_path, root_name=f"chain-child-{mutation}")
    parent_audit_path = tmp_path / f"chain-parent-{mutation}-audit.json"
    _reseed_from_recovery_artifact(
        fixture,
        parent=parent,
        parent_audit_path=parent_audit_path,
    )
    manifest = json.loads(fixture.manifest_path.read_text())
    if mutation == "parent_hash":
        manifest["mixed_artifact"]["seed_recovery_manifest_sha256"] = "0" * 64
    elif mutation == "audit_pass":
        parent_audit = json.loads(parent_audit_path.read_text())
        parent_audit["audit_pass"] = False
        _write_json(parent_audit_path, parent_audit)
        manifest["mixed_artifact"]["seed_recovery_audit_sha256"] = _sha256(
            parent_audit_path
        )
    else:
        manifest["mixed_artifact"]["seed_candidate_identity_sha256"] = "0" * 64
    manifest.pop("manifest_body_sha256")
    manifest["manifest_body_sha256"] = _canonical_sha256(manifest)
    _write_json(fixture.manifest_path, manifest)

    with pytest.raises(ValueError, match=message):
        _audit(fixture)


def test_real_audit_rechecks_exact_loaded_tree_identity(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    audit = _audit(fixture)
    replacement = fixture.root / "recovered-groups" / fixture.replacement_names[0]
    replacement.write_bytes(replacement.read_bytes() + b"mutated")

    with pytest.raises(ValueError, match="identity|hash|changed|retarget|artifact bytes"):
        audit.verify_current_identity()


def test_current_identity_recheck_does_not_copy_group_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(tmp_path)
    audit = _audit(fixture)
    module = sys.modules[audit.__class__.__module__]
    monkeypatch.setattr(
        module.tempfile,
        "mkdtemp",
        lambda *_args, **_kwargs: pytest.fail(
            "current identity recheck copied group payloads"
        ),
    )

    audit.verify_current_identity()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("hash", "SHA-256"),
        ("audit_pass", "audit_pass"),
        ("integrity", "artifact_integrity_pass"),
        ("lineage", "lineage|model"),
        ("negative", "payload"),
        ("over_budget", "112000000000|budget"),
    ),
)
def test_accepted_composite_audit_is_pinned_and_derives_accounting(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    profile = _profile()
    path = tmp_path / "accepted-composite-audit.json"
    expected_sha256 = _write_accepted_composite_audit(path, profile=profile)
    if mutation == "hash":
        expected_sha256 = "f" * 64
    elif mutation == "audit_pass":
        expected_sha256 = _write_accepted_composite_audit(
            path, profile=profile, audit_pass=False
        )
    elif mutation == "integrity":
        expected_sha256 = _write_accepted_composite_audit(
            path, profile=profile, artifact_integrity_pass=False
        )
    elif mutation == "lineage":
        expected_sha256 = _write_accepted_composite_audit(
            path, profile=profile, model_id="wrong/model"
        )
    elif mutation == "negative":
        expected_sha256 = _write_accepted_composite_audit(
            path, profile=profile, whole_payload=49_999, routed_payload=50_000
        )
    elif mutation == "over_budget":
        expected_sha256 = _write_accepted_composite_audit(
            path,
            profile=profile,
            whole_payload=CANONICAL_PAYLOAD_LIMIT + 1,
            routed_payload=50_000,
        )

    with pytest.raises(ValueError, match=message):
        recovery_materialize._authenticate_accepted_composite_audit(
            path,
            expected_sha256=expected_sha256,
            expected_source_lineage=_lineage(profile),
        )


@pytest.mark.parametrize("inventory", ("full", "routed"))
def test_audit_rejects_source_inventory_not_matching_external_authority(
    tmp_path: Path,
    inventory: str,
) -> None:
    fixture = _build_fixture(tmp_path)
    name = (
        "expected_full_source_blob_inventory_sha256"
        if inventory == "full"
        else "expected_routed_source_blob_inventory_sha256"
    )
    authorities = {
        "expected_full_source_blob_inventory_sha256": SOURCE_VERIFICATION[
            "source_blob_inventory_sha256"
        ],
        "expected_routed_source_blob_inventory_sha256": SOURCE_VERIFICATION[
            "routed_source_blob_inventory_sha256"
        ],
    }
    authorities[name] = "9" * 64
    spec = importlib.util.spec_from_file_location(
        f"glm52_recovery_inventory_{inventory}_under_test", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with pytest.raises(ValueError, match=f"{inventory}.*source blob inventory"):
        module.audit_glm52_recovery_mixed_artifact(
            fixture.root,
            profile=fixture.profile,
            expected_seed_manifest_sha256=fixture.seed_manifest_sha256,
            expected_stats_manifest_sha256=STATS_MANIFEST_SHA256,
            **authorities,
            expected_recovery_lever=RECOVERY_LEVER,
            expected_recovery_policy=_recovery_policy(fixture.profile),
            accepted_composite_audit_json=fixture.accepted_composite_audit_path,
            expected_composite_audit_sha256=fixture.accepted_composite_audit_sha256,
        )


def test_materialize_fixture_manifest_passes_audit_without_schema_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GLM52_E8P_BACKEND", "numpy")
    source_root = tmp_path / "hf/models--0xSero--glm-5.2-reap-504B-v2/snapshots" / REVISION
    source_root.mkdir(parents=True)
    config_path = source_root / "config.json"
    index_path = source_root / "model.safetensors.index.json"
    config_path.write_text('{"model_type":"glm_moe_dsa"}\n')
    index_path.write_text('{"metadata":{},"weight_map":{}}\n')
    config_sha256 = _sha256(config_path)
    index_sha256 = _sha256(index_path)
    monkeypatch.setattr(sys.modules[__name__], "CONFIG_SHA256", config_sha256)
    monkeypatch.setattr(sys.modules[__name__], "INDEX_SHA256", index_sha256)
    fixture = _build_fixture(tmp_path, root_name="producer-seed")

    stats_root = tmp_path / "stats"
    stats_root.mkdir()
    entries: list[dict[str, object]] = []
    for projection in PROJECTIONS:
        input_dim = 16 if projection in {"gate_proj", "up_proj"} else 8
        for expert in range(2):
            filename = (
                f"layer-00077-{projection}-expert-{expert:03d}.npz"
            )
            path = stats_root / filename
            np.savez(
                path,
                sum_x2=np.ones(input_dim, dtype=np.float32),
                mean_second_moment=np.ones(input_dim, dtype=np.float32),
                routing_weighted_importance=np.ones(input_dim, dtype=np.float32),
                router_score_weighted_importance=np.ones(input_dim, dtype=np.float32),
            )
            entries.append(
                {
                    "layer": 77,
                    "projection": projection,
                    "expert": expert,
                    "input_dim": input_dim,
                    "route_count": 1,
                    "total_route_count": 2040,
                    "route_frequency": 1 / 2040,
                    "router_score_sum": 0.5,
                    "prompt_ids": [f"selection-{index:02d}" for index in range(22)],
                    "sample_count": 22,
                    "prompt_count": 22,
                    "position_count": 255,
                    "path": filename,
                    "sha256": _sha256(path),
                }
            )
    split = {
        "split": "selection",
        "prompt_count": 22,
        "position_count": 255,
        "holdout_used_for_tuning": False,
        "report_used_for_tuning": False,
    }
    stats_manifest = {
        "schema_version": 1,
        "record_type": "glm52_recovery_stats_manifest",
        "status": "complete",
        "method": "selection_only_diagonal_hessian_second_moments_v1",
        "selected_layers": [77],
        "projections": list(PROJECTIONS),
        "split_evidence": split,
        "source_authority": {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "profile": PROFILE,
            "config_sha256": config_sha256,
            "index_sha256": index_sha256,
            "authenticated_source_teacher": True,
            "layer_major_streaming": True,
            "split_evidence": split,
        },
        "sample_count": 22,
        "prompt_count": 22,
        "position_count": 255,
        "route_count_by_layer": {"77": 2040},
        "entry_count": len(entries),
        "entries": entries,
    }
    stats_manifest_path = stats_root / "glm52-recovery-stats-manifest.json"
    _write_json(stats_manifest_path, stats_manifest)
    stats_sha256 = _sha256(stats_manifest_path)

    class _Inventory:
        blob_root = Path("/authenticated/blob-root")
        weight_map: dict[str, str] = {}

        def report(self) -> dict[str, object]:
            return {
                "source_blob_inventory_sha256": "7" * 64,
                "routed_source_blob_inventory_sha256": "8" * 64,
                "shard_count": 1,
            }

        def modelopt_reader_guard(self):
            return nullcontext()

        def verify_after_forward(self) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(recovery_materialize, "GLM52_REAP_CONFIG_SHA256", config_sha256, raising=False)
    monkeypatch.setattr(recovery_materialize, "GLM52_REAP_INDEX_SHA256", index_sha256, raising=False)
    monkeypatch.setattr(
        recovery_materialize,
        "_load_source_auth_module",
        lambda: SimpleNamespace(
            _open_authenticated_source_blob_inventory=lambda _source, _index: _Inventory()
        ),
        raising=False,
    )
    import keep.convert.nvfp4 as nvfp4

    monkeypatch.setattr(nvfp4, "resolve_modelopt_nvfp4_weight_bundle", lambda _map, name: name)
    monkeypatch.setattr(
        nvfp4,
        "read_modelopt_nvfp4_weight",
        lambda _root, name: np.full(
            (16, 8) if ".down_proj." in name else (8, 16),
            0.25 if ".experts.0." in name else -0.25,
            dtype=np.float32,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "keep.convert.stream_convert",
        SimpleNamespace(load_safetensors_index=lambda _path: SimpleNamespace(weight_map={})),
    )
    output_root = tmp_path / "producer-output"
    manifest = recovery_materialize.materialize_groups_from_source(
        heavy_lock_path=tmp_path / "materializer-test.lock",
        source_dir=source_root,
        index_path=index_path,
        seed_artifact_dir=fixture.seed,
        stats_dir=stats_root,
        output_dir=output_root,
        groups=tuple((77, projection) for projection in PROJECTIONS),
        group_size=8,
        expected_stats_manifest_sha256=stats_sha256,
        expected_seed_manifest_sha256=fixture.seed_manifest_sha256,
        expected_full_source_blob_inventory_sha256=SOURCE_VERIFICATION[
            "source_blob_inventory_sha256"
        ],
        expected_routed_source_blob_inventory_sha256=SOURCE_VERIFICATION[
            "routed_source_blob_inventory_sha256"
        ],
        accepted_composite_audit_json=fixture.accepted_composite_audit_path,
        expected_composite_audit_sha256=fixture.accepted_composite_audit_sha256,
    )
    assert json.loads((output_root / "conversion-manifest.json").read_text()) == manifest
    assert manifest["source_verification"] == SOURCE_VERIFICATION
    assert manifest["groups"][0]["lever_provenance"]["source_lineage"] == _lineage(
        fixture.profile
    )

    name = "glm52_recovery_direct_integration_audit"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    audit = module.audit_glm52_recovery_mixed_artifact(
        output_root,
        profile=fixture.profile,
        expected_seed_manifest_sha256=fixture.seed_manifest_sha256,
        expected_stats_manifest_sha256=stats_sha256,
        expected_full_source_blob_inventory_sha256=SOURCE_VERIFICATION[
            "source_blob_inventory_sha256"
        ],
        expected_routed_source_blob_inventory_sha256=SOURCE_VERIFICATION[
            "routed_source_blob_inventory_sha256"
        ],
        expected_recovery_lever=RECOVERY_LEVER,
        expected_recovery_policy=_recovery_policy(fixture.profile),
        accepted_composite_audit_json=fixture.accepted_composite_audit_path,
        expected_composite_audit_sha256=fixture.accepted_composite_audit_sha256,
    )
    assert audit.audit_pass is True


def test_seed_manifest_hash_and_parse_reject_deterministic_path_retarget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(tmp_path)
    seed_manifest = fixture.seed / "conversion-manifest.json"
    retarget = tmp_path / "retargeted-seed-manifest.json"
    shutil.copyfile(seed_manifest, retarget)
    name = "glm52_recovery_seed_manifest_toctou_under_test"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    real_loads = json.loads
    swapped = False

    def swap_after_snapshot(payload: object, *args: Any, **kwargs: Any):
        nonlocal swapped
        value = real_loads(payload, *args, **kwargs)
        if (
            not swapped
            and isinstance(payload, bytes)
            and isinstance(value, dict)
            and value.get("record_type") == "glm52_modelopt_nvfp4_materialization_manifest"
        ):
            os.replace(retarget, seed_manifest)
            swapped = True
        return value

    monkeypatch.setattr(module.json, "loads", swap_after_snapshot)
    with pytest.raises(ValueError, match="seed manifest.*retarget|identity.*seed manifest|seed manifest.*identity"):
        module.audit_glm52_recovery_mixed_artifact(
            fixture.root,
            profile=fixture.profile,
            expected_seed_manifest_sha256=fixture.seed_manifest_sha256,
            expected_stats_manifest_sha256=STATS_MANIFEST_SHA256,
            expected_full_source_blob_inventory_sha256=SOURCE_VERIFICATION[
                "source_blob_inventory_sha256"
            ],
            expected_routed_source_blob_inventory_sha256=SOURCE_VERIFICATION[
                "routed_source_blob_inventory_sha256"
            ],
            expected_recovery_lever=RECOVERY_LEVER,
            expected_recovery_policy=_recovery_policy(fixture.profile),
            accepted_composite_audit_json=fixture.accepted_composite_audit_path,
            expected_composite_audit_sha256=fixture.accepted_composite_audit_sha256,
        )
    assert swapped is True


def test_candidate_identity_is_path_independent(tmp_path: Path) -> None:
    first = _build_fixture(tmp_path, root_name="first")
    second = _build_fixture(tmp_path, root_name="second")
    second_manifest = json.loads(second.manifest_path.read_text())
    for name in first.replacement_names:
        shutil.copyfile(
            first.root / "recovered-groups" / name,
            second.root / "recovered-groups" / name,
        )
        record = next(
            item for item in second_manifest["groups"] if item["artifact_path"] == name
        )
        record["artifact_bytes"] = (second.root / "recovered-groups" / name).stat().st_size
        record["artifact_sha256"] = _sha256(second.root / "recovered-groups" / name)
    _write_json(second.manifest_path, second_manifest)
    _reattest(second)

    assert _audit(first).candidate_identity_sha256 == _audit(second).candidate_identity_sha256


def test_rejects_recovered_group_masquerading_as_inherited(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    name = fixture.replacement_names[0]
    link = fixture.root / "artifact" / name
    link.unlink()
    os.symlink(os.path.relpath(fixture.seed / name, link.parent), link)

    with pytest.raises(ValueError, match="masquerad|classification|replacement"):
        _audit(fixture)


def test_rejects_rate_and_embedded_codebook_mismatch(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    name = fixture.replacement_names[0]
    layer = int(name.split("-")[1])
    projection = name.removeprefix(f"layer-{layer:05d}-").removesuffix(".safetensors")
    replacement = fixture.root / "recovered-groups" / name
    _write_group(
        replacement,
        profile=fixture.profile,
        layer=layer,
        projection=projection,
        code_bits=8,
        metadata_code_bits=16,
        replacement=True,
    )
    manifest = json.loads(fixture.manifest_path.read_text())
    record = next(item for item in manifest["groups"] if item["artifact_path"] == name)
    record["artifact_bytes"] = replacement.stat().st_size
    record["artifact_sha256"] = _sha256(replacement)
    _write_json(fixture.manifest_path, manifest)
    _reattest(fixture)

    with pytest.raises(ValueError, match="codebook|code bits|dtype"):
        _audit(fixture)


def test_rejects_partial_layer_rate_upgrade(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    manifest = json.loads(fixture.manifest_path.read_text())
    removed = manifest["groups"].pop()
    manifest["selected_groups"].remove(
        f"{removed['layer']}:{removed['projection']}"
    )
    manifest["mixed_artifact"]["replacement_count"] -= 1
    manifest["mixed_artifact"]["inherited_group_count"] += 1
    name = removed["artifact_path"]
    link = fixture.root / "artifact" / name
    link.unlink()
    os.symlink(os.path.relpath(fixture.seed / name, link.parent), link)
    (fixture.root / "recovered-groups" / name).unlink()
    _write_json(fixture.manifest_path, manifest)
    _reattest(fixture)

    with pytest.raises(ValueError, match="complete layer|rate consistency|gate.*up.*down"):
        _audit(fixture)


def test_rejects_symlink_escape(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    name = _filename(3, "gate_proj")
    outside = tmp_path / "outside.safetensors"
    shutil.copyfile(fixture.seed / name, outside)
    link = fixture.root / "artifact" / name
    link.unlink()
    os.symlink(os.path.relpath(outside, link.parent), link)

    with pytest.raises(ValueError, match="outside|escape|root"):
        _audit(fixture)


def test_rejects_absolute_symlink_even_when_target_is_allowed(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    name = _filename(3, "gate_proj")
    link = fixture.root / "artifact" / name
    link.unlink()
    os.symlink(fixture.seed / name, link)

    with pytest.raises(ValueError, match="absolute"):
        _audit(fixture)


def test_rejects_manifest_replay_in_another_candidate_root(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    replay = tmp_path / "replay"
    shutil.copytree(fixture.root, replay, symlinks=True)

    replayed = replace(fixture, root=replay, manifest_path=replay / "conversion-manifest.json")
    with pytest.raises(ValueError, match="replay|output.*root|output_dir"):
        _audit(replayed)


def test_rejects_budget_laundering_from_manifest_claim(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    manifest = json.loads(fixture.manifest_path.read_text())
    manifest["accounting"]["logical_whole_model_tensor_payload_bytes"] = 1
    manifest["accounting"]["logical_payload_limit_bytes"] = 112_000_000_000
    _write_json(fixture.manifest_path, manifest)
    _reattest(fixture)

    with pytest.raises(ValueError, match="logical.*payload|accounting"):
        _audit(fixture)

    honest = _build_fixture(tmp_path, root_name="over-budget")
    over_budget_sha256 = _write_accepted_composite_audit(
        honest.accepted_composite_audit_path,
        profile=honest.profile,
        whole_payload=CANONICAL_PAYLOAD_LIMIT + 1,
        routed_payload=1,
    )
    with pytest.raises(ValueError, match="112|limit|budget"):
        _audit(replace(honest, accepted_composite_audit_sha256=over_budget_sha256))


def test_rejects_tampered_manifest_body(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    manifest = json.loads(fixture.manifest_path.read_text())
    manifest["status"] = "tampered"
    _write_json(fixture.manifest_path, manifest)

    with pytest.raises(ValueError, match="manifest body"):
        _audit(fixture)


@pytest.mark.parametrize(
    ("scope", "mutation"),
    (
        ("top", "arbitrary_lever"),
        ("group", "wrong_stats"),
        ("embedded", "extra_field"),
        ("top", "missing_policy_field"),
    ),
)
def test_external_recovery_authority_is_exact_at_every_scope(
    tmp_path: Path,
    scope: str,
    mutation: str,
) -> None:
    fixture = _build_fixture(tmp_path)
    manifest = json.loads(fixture.manifest_path.read_text())
    replacement = fixture.root / "recovered-groups" / fixture.replacement_names[0]
    if scope == "top" and mutation == "arbitrary_lever":
        manifest["lever_provenance"]["lever"] = "fixture_selected_lever"
    elif scope == "group":
        manifest["groups"][0]["lever_provenance"]["stats_manifest_sha256"] = "f" * 64
    elif scope == "top":
        manifest["recovery_policy"].pop("rounding_objective")
    else:
        from safetensors import safe_open

        with safe_open(replacement, framework="np") as handle:
            tensors = {name: handle.get_tensor(name) for name in handle.keys()}
            metadata = dict(handle.metadata() or {})
        provenance = json.loads(metadata["glm52_recovery_provenance"])
        provenance["fixture_only"] = True
        metadata["glm52_recovery_provenance"] = json.dumps(provenance, sort_keys=True)
        save_file(tensors, replacement, metadata=metadata)
        manifest["groups"][0]["artifact_bytes"] = replacement.stat().st_size
        manifest["groups"][0]["artifact_sha256"] = _sha256(replacement)
    _write_json(fixture.manifest_path, manifest)
    _reattest(fixture)

    with pytest.raises(ValueError, match="lever|stats|provenance|policy|exact"):
        _audit(fixture)


def test_rejects_artifact_retarget_between_hash_and_tensor_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(tmp_path)
    target = fixture.root / "recovered-groups" / fixture.replacement_names[0]
    replacement = tmp_path / "retargeted.safetensors"
    shutil.copyfile(target, replacement)
    from safetensors import safe_open as real_safe_open

    with real_safe_open(replacement, framework="np") as handle:
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
        metadata = dict(handle.metadata() or {})
    codes_name = next(name for name in tensors if name.endswith(".codes"))
    tensors[codes_name] = np.ones_like(tensors[codes_name])
    save_file(tensors, replacement, metadata=metadata)

    name = "glm52_recovery_artifact_toctou_under_test"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    safe_open_calls = 0

    def swap_then_open(path: Path, *args: Any, **kwargs: Any):
        nonlocal safe_open_calls
        safe_open_calls += 1
        if safe_open_calls == 223:
            os.replace(replacement, target)
        return real_safe_open(path, *args, **kwargs)

    monkeypatch.setattr(module, "safe_open", swap_then_open)
    with pytest.raises(ValueError, match="changed|identity|snapshot|retarget"):
        module.audit_glm52_recovery_mixed_artifact(
            fixture.root,
            profile=fixture.profile,
            expected_seed_manifest_sha256=fixture.seed_manifest_sha256,
            expected_stats_manifest_sha256=STATS_MANIFEST_SHA256,
            expected_full_source_blob_inventory_sha256=SOURCE_VERIFICATION[
                "source_blob_inventory_sha256"
            ],
            expected_routed_source_blob_inventory_sha256=SOURCE_VERIFICATION[
                "routed_source_blob_inventory_sha256"
            ],
            expected_recovery_lever=RECOVERY_LEVER,
            expected_recovery_policy=_recovery_policy(fixture.profile),
            accepted_composite_audit_json=fixture.accepted_composite_audit_path,
            expected_composite_audit_sha256=fixture.accepted_composite_audit_sha256,
        )


def test_rejects_unexpected_file_layer_78_and_retargeting(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    artifact = fixture.root / "artifact"
    (artifact / "notes.txt").write_text("not allowed")
    with pytest.raises(ValueError, match="unexpected"):
        _audit(fixture)
    (artifact / "notes.txt").unlink()

    layer_78 = artifact / _filename(78, "gate_proj")
    os.symlink(os.path.relpath(fixture.seed / _filename(3, "gate_proj"), artifact), layer_78)
    with pytest.raises(ValueError, match="layer 78|unexpected"):
        _audit(fixture)
    layer_78.unlink()

    link = artifact / _filename(3, "gate_proj")
    link.unlink()
    os.symlink(os.path.relpath(fixture.seed / _filename(4, "gate_proj"), artifact), link)
    with pytest.raises(ValueError, match="retarget"):
        _audit(fixture)


def test_rejects_missing_target_cycle_and_alias_outside_roots(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    artifact = fixture.root / "artifact"
    name = _filename(3, "gate_proj")
    link = artifact / name
    link.unlink()
    os.symlink("../missing/" + name, link)
    with pytest.raises(ValueError, match="missing|target"):
        _audit(fixture)

    link.unlink()
    os.symlink(name, link)
    with pytest.raises(ValueError, match="cycle|loop"):
        _audit(fixture)

    link.unlink()
    alias = tmp_path / "seed-alias"
    os.symlink(fixture.seed, alias)
    os.symlink(os.path.relpath(alias / name, artifact), link)
    with pytest.raises(ValueError, match="alias|root"):
        _audit(fixture)
