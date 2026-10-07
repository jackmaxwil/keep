from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors.numpy import save_file

from mlx_vq.codebook.e8 import e8_1bit_packed, e8p_packed_abs_grid
from mlx_vq.convert.stream_convert import SafetensorsIndex
from mlx_vq.io.schema import codebook_metadata_for_bits
from mlx_vq.models.profiles import ModelProfile, get_profile
from mlx_vq.validate import glm52_vq


MODEL_ID = "fixture/glm52-reap"
REVISION = "fixture-revision"
CONFIG_SHA256 = "c" * 64
INDEX_SHA256 = "d" * 64
SOURCE_DECODER = "modelopt_nvfp4_v1"
ALL_GROUP_KEYS = (
    "3:gate_proj",
    "3:up_proj",
    "3:down_proj",
    "4:gate_proj",
    "4:up_proj",
    "4:down_proj",
)


@dataclass(frozen=True)
class _FixtureArtifact:
    root: Path
    profile: ModelProfile
    group_keys: tuple[str, ...]
    manifest_path: Path
    evidence_paths: tuple[Path, ...]


def _fixture_profile() -> ModelProfile:
    return replace(
        get_profile("glm52-reap-504b-v2"),
        name="glm52-reap-fixture",
        hf_model_id=MODEL_ID,
        revision=REVISION,
        num_layers=5,
        num_sparse_layers=2,
        first_sparse_layer=3,
        hidden_size=16,
        moe_intermediate_size=8,
        num_experts=2,
        experts_per_tok=1,
        group_size_policy={"gate": 8, "up": 8, "down": 8},
        recovery_layer=4,
        recovery_projections=("gate", "up", "down"),
        hard_layers=(),
        imatrix_layers=(3, 4),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _group_identity(group_key: str) -> tuple[int, str]:
    layer_raw, projection = group_key.split(":", maxsplit=1)
    return int(layer_raw), projection


def _group_dims(profile: ModelProfile, projection: str) -> tuple[int, int]:
    if projection in {"gate_proj", "up_proj"}:
        return profile.hidden_size, profile.moe_intermediate_size
    if projection == "down_proj":
        return profile.moe_intermediate_size, profile.hidden_size
    raise AssertionError(projection)


def _prefix(layer: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.switch_mlp.{projection}"


def _quantization_metadata(
    *,
    profile: ModelProfile,
    code_bits: int,
    group_size: int,
    policy_overrides: dict[str, object] | None = None,
    codebook_sha256: str | None = None,
) -> dict[str, str]:
    codebook_name, expected_sha256 = codebook_metadata_for_bits(code_bits)
    policy: dict[str, object] = {
        "scale_estimator": "max_abs",
        "source_weight_encoding": "modelopt_nvfp4",
        "source_decoder": SOURCE_DECODER,
        "source_model_id": MODEL_ID,
        "source_revision": REVISION,
        "source_config_sha256": CONFIG_SHA256,
        "source_index_sha256": INDEX_SHA256,
        "source_profile": profile.name,
        "decoded_expert_working_set": "one_per_worker",
    }
    policy.update(policy_overrides or {})
    payload = {
        "quant_method": "mlx_vq_e8",
        "version": 1,
        "default_code_bits": code_bits,
        "default_group_size": group_size,
        "codebook": {
            "name": codebook_name,
            "dtype": "uint32",
            "entries": 256,
            "sha256": codebook_sha256 or expected_sha256,
        },
        "policy": policy,
    }
    return {"quantization_config": json.dumps(payload, sort_keys=True)}


def _write_group(
    root: Path,
    *,
    profile: ModelProfile,
    group_key: str,
    code_bits: int,
    group_size: int,
    filename: str | None = None,
    omit: str | None = None,
    add_tensor: str | None = None,
    codes_dtype: np.dtype[Any] | None = None,
    scales_dtype: np.dtype[Any] = np.dtype(np.float16),
    codes_shape: tuple[int, ...] | None = None,
    wrong_codebook_tensor: bool = False,
    codebook_sha256: str | None = None,
    policy_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    layer, projection = _group_identity(group_key)
    input_dims, output_dims = _group_dims(profile, projection)
    prefix = _prefix(layer, projection)
    codebook = (
        e8_1bit_packed().copy()
        if code_bits == 8
        else e8p_packed_abs_grid().copy()
    )
    if wrong_codebook_tensor:
        codebook[0] ^= np.uint32(1)
    codes_dtype = codes_dtype or np.dtype(np.uint8 if code_bits == 8 else np.uint16)
    expected_codes_shape = (
        profile.num_experts,
        output_dims,
        input_dims // 8,
    )
    arrays: dict[str, np.ndarray] = {
        f"{prefix}.codes": np.zeros(codes_shape or expected_codes_shape, dtype=codes_dtype),
        f"{prefix}.scales": np.ones(
            (profile.num_experts, output_dims, input_dims // group_size),
            dtype=scales_dtype,
        ),
        "model.vq_codebook.e8": codebook,
    }
    if omit == "codes":
        arrays.pop(f"{prefix}.codes")
    elif omit == "scales":
        arrays.pop(f"{prefix}.scales")
    elif omit == "codebook":
        arrays.pop("model.vq_codebook.e8")
    if add_tensor is not None:
        arrays[add_tensor] = np.zeros((1,), dtype=np.float16)

    output = root / (filename or f"layer-{layer:05d}-{projection}.safetensors")
    save_file(
        arrays,
        output,
        metadata=_quantization_metadata(
            profile=profile,
            code_bits=code_bits,
            group_size=group_size,
            policy_overrides=policy_overrides,
            codebook_sha256=codebook_sha256,
        ),
    )
    codes_name = f"{prefix}.codes"
    scales_name = f"{prefix}.scales"
    return {
        "status": "ready",
        "layer": layer,
        "projection": projection,
        "expert_count": profile.num_experts,
        "artifact_path": output.name,
        "artifact_bytes": output.stat().st_size,
        "artifact_sha256": _sha256(output),
        "codes_name": codes_name,
        "codes_shape": list(expected_codes_shape),
        "codes_dtype": "uint8" if code_bits == 8 else "uint16",
        "scales_name": scales_name,
        "scales_shape": [
            profile.num_experts,
            output_dims,
            input_dims // group_size,
        ],
        "scales_dtype": "float16",
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _build_artifact(
    tmp_path: Path,
    *,
    group_keys: tuple[str, ...] = ("3:gate_proj",),
    code_bits: int = 8,
    group_size: int = 8,
    manifest_overrides: dict[str, object] | None = None,
    group_kwargs: dict[str, object] | None = None,
) -> _FixtureArtifact:
    profile = _fixture_profile()
    root = tmp_path / "artifact"
    root.mkdir()
    groups = [
        _write_group(
            root,
            profile=profile,
            group_key=key,
            code_bits=code_bits,
            group_size=group_size,
            **(group_kwargs or {}),
        )
        for key in group_keys
    ]
    full = group_keys == ALL_GROUP_KEYS
    codebook_name, codebook_sha256 = codebook_metadata_for_bits(code_bits)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_modelopt_nvfp4_materialization_manifest",
        "materialization_status": "glm52_modelopt_nvfp4_groups_ready",
        "materialization_scope": "full" if full else "bounded",
        "full_group_coverage": full,
        "materialization_blocked": False,
        "materialization_blockers": [],
        "profile": profile.name,
        "model_id": MODEL_ID,
        "source_revision": REVISION,
        "config_sha256": CONFIG_SHA256,
        "index_sha256": INDEX_SHA256,
        "source_weight_encoding": "modelopt_nvfp4",
        "source_decoder": SOURCE_DECODER,
        "planned_vq_groups": len(ALL_GROUP_KEYS),
        "selected_vq_groups": len(group_keys),
        "ready_vq_groups": len(group_keys),
        "skipped_vq_groups": len(ALL_GROUP_KEYS) - len(group_keys),
        "selected_group_keys": list(group_keys),
        "code_bits": code_bits,
        "group_size": group_size,
        "scale_estimator": "max_abs",
        "codebook_name": codebook_name,
        "codebook_sha256": codebook_sha256,
        "artifact_total_bytes": sum(int(group["artifact_bytes"]) for group in groups),
        "dense_checkpoint_written": False,
        "groups": groups,
    }
    manifest.update(manifest_overrides or {})
    manifest_path = root / "conversion-manifest.json"
    _write_json(manifest_path, manifest)
    manifest_sha256 = _sha256(manifest_path)

    base_run = {
        **manifest,
        "record_type": "glm52_modelopt_nvfp4_materialization_run",
        "artifact_manifest_path": str(manifest_path),
        "artifact_manifest_sha256": manifest_sha256,
        "converted_vq_groups": len(group_keys),
        "existing_vq_groups": 0,
        "run_groups": [
            {
                "layer": group["layer"],
                "projection": group["projection"],
                "run_status": "converted",
            }
            for group in groups
        ],
    }
    resume_run = {
        **base_run,
        "converted_vq_groups": 0,
        "existing_vq_groups": len(group_keys),
        "run_groups": [
            {**record, "run_status": "existing"}
            for record in base_run["run_groups"]
        ],
    }
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    first_path = evidence_root / "first-run.json"
    resume_path = evidence_root / "resume-run.json"
    _write_json(first_path, base_run)
    _write_json(resume_path, resume_run)
    return _FixtureArtifact(
        root=root,
        profile=profile,
        group_keys=group_keys,
        manifest_path=manifest_path,
        evidence_paths=(first_path, resume_path),
    )


def _audit(
    fixture: _FixtureArtifact,
    *,
    expected_group_keys: tuple[str, ...] | None = None,
    expected_code_bits: int = 8,
    evidence_paths: tuple[Path, ...] = (),
    non_routed_artifact_bytes: int | None = None,
    whole_model_parameter_count: int | None = None,
):
    return glm52_vq.audit_glm52_reap_materialization_manifest(
        fixture.root,
        profile=fixture.profile,
        expected_group_keys=expected_group_keys or fixture.group_keys,
        expected_code_bits=expected_code_bits,
        expected_config_sha256=CONFIG_SHA256,
        expected_index_sha256=INDEX_SHA256,
        evidence_paths=evidence_paths,
        non_routed_artifact_bytes=non_routed_artifact_bytes,
        whole_model_parameter_count=whole_model_parameter_count,
    )


def test_bounded_artifact_audit_reports_only_actual_bounded_values_and_resume_identity(
    tmp_path: Path,
) -> None:
    fixture = _build_artifact(tmp_path)

    payload = _audit(fixture, evidence_paths=fixture.evidence_paths).to_dict()

    artifact_path = fixture.root / "layer-00003-gate_proj.safetensors"
    assert payload["audit_pass"] is True
    assert payload["materialization_scope"] == "bounded"
    assert payload["expected_group_keys"] == ["3:gate_proj"]
    assert payload["ready_group_keys"] == ["3:gate_proj"]
    assert payload["actual_routed_weight_count"] == 2 * 8 * 16
    assert payload["actual_routed_payload_bytes"] == 96
    assert payload["actual_routed_bpw"] == pytest.approx(3.0)
    assert payload["actual_artifact_bytes"] == artifact_path.stat().st_size
    assert payload["actual_whole_model_artifact_bytes"] is None
    assert payload["actual_whole_model_bpw"] is None
    assert payload["whole_model_values_actual"] is False
    assert payload["dense_routed_experts"] is False
    assert payload["resume_verified"] is True
    assert payload["byte_identity_verified"] is True


def test_full_artifact_audit_may_report_actual_whole_model_values(tmp_path: Path) -> None:
    fixture = _build_artifact(tmp_path, group_keys=ALL_GROUP_KEYS)
    routed_artifact_bytes = sum(
        path.stat().st_size for path in fixture.root.glob("layer-*.safetensors")
    )

    payload = _audit(
        fixture,
        evidence_paths=fixture.evidence_paths,
        non_routed_artifact_bytes=424,
        whole_model_parameter_count=2_000,
    ).to_dict()

    assert payload["audit_pass"] is True
    assert payload["materialization_scope"] == "full"
    assert payload["actual_routed_weight_count"] == 1_536
    assert payload["actual_routed_payload_bytes"] == 576
    assert payload["actual_routed_bpw"] == pytest.approx(3.0)
    assert payload["actual_whole_model_artifact_bytes"] == routed_artifact_bytes + 424
    assert payload["actual_whole_model_bpw"] == pytest.approx(
        (routed_artifact_bytes + 424) * 8 / 2_000
    )
    assert payload["whole_model_values_actual"] is True


@pytest.mark.parametrize("failure", ["missing", "extra", "wrong_name"])
def test_artifact_audit_fails_loud_on_missing_extra_or_wrong_named_files(
    tmp_path: Path,
    failure: str,
) -> None:
    kwargs = {"filename": "renamed-group.safetensors"} if failure == "wrong_name" else {}
    fixture = _build_artifact(tmp_path, group_kwargs=kwargs)
    expected = fixture.root / "layer-00003-gate_proj.safetensors"
    if failure == "missing":
        expected.unlink()
    elif failure == "extra":
        shutil.copyfile(expected, fixture.root / "unexpected.safetensors")

    with pytest.raises(ValueError, match="missing|unexpected|file name|filename"):
        _audit(fixture)


@pytest.mark.parametrize(
    ("group_kwargs", "message"),
    [
        ({"omit": "scales"}, "missing tensor|scales"),
        ({"add_tensor": "model.unexpected"}, "unexpected tensor"),
        (
            {"add_tensor": "model.layers.3.mlp.switch_mlp.gate_proj.weight"},
            "dense routed|weight",
        ),
    ],
)
def test_artifact_audit_requires_exact_tensor_set_and_no_dense_routed_payload(
    tmp_path: Path,
    group_kwargs: dict[str, object],
    message: str,
) -> None:
    fixture = _build_artifact(tmp_path, group_kwargs=group_kwargs)

    with pytest.raises(ValueError, match=message):
        _audit(fixture)


@pytest.mark.parametrize(
    ("build_kwargs", "expected_code_bits", "message"),
    [
        ({"group_kwargs": {"codes_dtype": np.dtype(np.int32)}}, 8, "codes.*dtype"),
        ({"group_kwargs": {"scales_dtype": np.dtype(np.float32)}}, 8, "scales.*dtype"),
        ({"group_kwargs": {"codes_shape": (2, 8, 1)}}, 8, "codes.*shape"),
        ({"code_bits": 16}, 8, "code bits|code_bits|budget"),
        ({"group_size": 16}, 8, "group size|group_size"),
        (
            {"group_kwargs": {"wrong_codebook_tensor": True}},
            8,
            "codebook",
        ),
        (
            {"group_kwargs": {"codebook_sha256": "0" * 64}},
            8,
            "codebook",
        ),
    ],
)
def test_artifact_audit_enforces_dtype_shape_budget_group_size_and_codebook(
    tmp_path: Path,
    build_kwargs: dict[str, object],
    expected_code_bits: int,
    message: str,
) -> None:
    fixture = _build_artifact(tmp_path, **build_kwargs)

    with pytest.raises(ValueError, match=message):
        _audit(fixture, expected_code_bits=expected_code_bits)


@pytest.mark.parametrize(
    ("build_kwargs", "message"),
    [
        ({"manifest_overrides": {"source_revision": "wrong"}}, "revision"),
        (
            {"group_kwargs": {"policy_overrides": {"source_decoder": "wrong"}}},
            "decoder",
        ),
        ({"manifest_overrides": {"config_sha256": "0" * 64}}, "config"),
        ({"manifest_overrides": {"index_sha256": "0" * 64}}, "index"),
    ],
)
def test_artifact_audit_rejects_wrong_manifest_or_tensor_lineage(
    tmp_path: Path,
    build_kwargs: dict[str, object],
    message: str,
) -> None:
    fixture = _build_artifact(tmp_path, **build_kwargs)

    with pytest.raises(ValueError, match=message):
        _audit(fixture)


def test_artifact_audit_rejects_any_layer_78_artifact(tmp_path: Path) -> None:
    fixture = _build_artifact(tmp_path)
    shutil.copyfile(
        fixture.root / "layer-00003-gate_proj.safetensors",
        fixture.root / "layer-00078-gate_proj.safetensors",
    )

    with pytest.raises(ValueError, match="layer.?78|00078"):
        _audit(fixture)


def test_artifact_audit_rejects_any_unapproved_non_safetensors_payload(
    tmp_path: Path,
) -> None:
    fixture = _build_artifact(tmp_path)
    (fixture.root / "dense.bin").write_bytes(b"unapproved-payload")

    with pytest.raises(ValueError, match=r"unexpected artifact tree|dense\.bin"):
        _audit(fixture)


@pytest.mark.parametrize("field", ["artifact_bytes", "artifact_sha256"])
def test_artifact_audit_recomputes_manifest_size_and_hash(
    tmp_path: Path,
    field: str,
) -> None:
    fixture = _build_artifact(tmp_path)
    manifest = json.loads(fixture.manifest_path.read_text())
    manifest["groups"][0][field] = 0 if field == "artifact_bytes" else "0" * 64
    _write_json(fixture.manifest_path, manifest)

    with pytest.raises(ValueError, match="byte size|bytes|SHA-256|sha256|hash"):
        _audit(fixture)


def test_artifact_audit_requires_an_exact_explicit_expected_selection(
    tmp_path: Path,
) -> None:
    fixture = _build_artifact(tmp_path)
    audit = glm52_vq.audit_glm52_reap_materialization_manifest

    with pytest.raises(TypeError, match="expected_group_keys"):
        audit(
            fixture.root,
            profile=fixture.profile,
            expected_code_bits=8,
            expected_config_sha256=CONFIG_SHA256,
            expected_index_sha256=INDEX_SHA256,
        )
    with pytest.raises(ValueError, match="selection|expected group|group keys"):
        _audit(fixture, expected_group_keys=("3:up_proj",))


def test_artifact_audit_rejects_resume_evidence_with_byte_identity_drift(
    tmp_path: Path,
) -> None:
    fixture = _build_artifact(tmp_path)
    resume = json.loads(fixture.evidence_paths[1].read_text())
    resume["groups"][0]["artifact_sha256"] = "0" * 64
    _write_json(fixture.evidence_paths[1], resume)

    with pytest.raises(ValueError, match="resume|byte identity|SHA-256|sha256|hash"):
        _audit(fixture, evidence_paths=fixture.evidence_paths)


def test_duplicate_run_evidence_does_not_prove_byte_identity(tmp_path: Path) -> None:
    fixture = _build_artifact(tmp_path)

    payload = _audit(
        fixture,
        evidence_paths=(fixture.evidence_paths[0], fixture.evidence_paths[0]),
    ).to_dict()

    assert payload["resume_verified"] is False
    assert payload["byte_identity_verified"] is False


def test_source_accounting_excludes_routed_source_metadata_and_mtp(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    shard = source_dir / "model.safetensors"
    tensors = {
        "model.embed_tokens.weight": np.zeros((2, 2), dtype=np.float16),
        "model.layers.3.mlp.experts.0.gate_proj.weight": np.zeros(
            (2, 1), dtype=np.uint8
        ),
        "model.layers.3.mlp.experts.0.gate_proj.weight_scale": np.zeros(
            (2, 1), dtype=np.uint8
        ),
        "model.layers.3.mlp.experts.0.gate_proj.weight_scale_2": np.zeros(
            (), dtype=np.float32
        ),
        "model.layers.3.mlp.experts.0.gate_proj.input_scale": np.zeros(
            (), dtype=np.float32
        ),
        "model.layers.5.mlp.experts.0.gate_proj.weight": np.zeros(
            (2, 1), dtype=np.uint8
        ),
    }
    save_file(tensors, shard)
    index = SafetensorsIndex(
        metadata={},
        weight_map={name: shard.name for name in tensors},
    )

    accounting = glm52_vq.audit_glm52_reap_source_accounting(
        source_dir,
        index=index,
        profile=_fixture_profile(),
    ).to_dict()

    assert accounting["main_non_routed_tensor_count"] == 1
    assert accounting["main_non_routed_parameter_count"] == 4
    assert accounting["main_non_routed_tensor_payload_bytes"] == 8
    assert accounting["main_routed_parameter_count"] == 4
    assert accounting["main_routed_source_payload_bytes"] == 12
    assert accounting["excluded_mtp_tensor_count"] == 1
    assert accounting["excluded_mtp_parameter_count"] == 2
    assert accounting["excluded_mtp_tensor_payload_bytes"] == 2
    assert accounting["main_model_parameter_count_excluding_mtp"] == 8


def test_source_accounting_rejects_unindexed_header_tensors(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    shard = source_dir / "model.safetensors"
    save_file(
        {
            "model.embed_tokens.weight": np.zeros((2, 2), dtype=np.float16),
            "unindexed.extra": np.zeros((1,), dtype=np.float16),
        },
        shard,
    )
    index = SafetensorsIndex(
        metadata={},
        weight_map={"model.embed_tokens.weight": shard.name},
    )

    with pytest.raises(ValueError, match="unindexed|header tensor"):
        glm52_vq.audit_glm52_reap_source_accounting(
            source_dir,
            index=index,
            profile=_fixture_profile(),
        )


def test_source_accounting_rejects_truncated_shard_payload(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    shard = source_dir / "model.safetensors"
    save_file(
        {"model.embed_tokens.weight": np.zeros((2, 2), dtype=np.float16)},
        shard,
    )
    shard.write_bytes(shard.read_bytes()[:-1])
    index = SafetensorsIndex(
        metadata={},
        weight_map={"model.embed_tokens.weight": shard.name},
    )

    with pytest.raises(ValueError, match="physical|truncated|file size|extent"):
        glm52_vq.audit_glm52_reap_source_accounting(
            source_dir,
            index=index,
            profile=_fixture_profile(),
        )


def test_artifact_audit_cli_writes_failure_evidence_before_nonzero_exit(
    tmp_path: Path,
) -> None:
    output_json = tmp_path / "failed-audit.json"
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/audit_glm52_reap_materialization.py",
            "--artifact-dir",
            str(tmp_path / "missing-artifact"),
            "--source-dir",
            str(tmp_path / "missing-source"),
            "--profile-path",
            str(tmp_path / "missing-profile.yaml"),
            "--config-path",
            str(tmp_path / "missing-config.json"),
            "--index-path",
            str(tmp_path / "missing-index.json"),
            "--model-id",
            MODEL_ID,
            "--revision",
            REVISION,
            "--groups",
            "3:gate_proj",
            "--code-bits",
            "8",
            "--group-size",
            "8",
            "--scale-estimator",
            "max_abs",
            "--output-json",
            str(output_json),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    payload = json.loads(output_json.read_text())
    assert payload["audit_pass"] is False
    assert payload["artifact_integrity_pass"] is False
    assert payload["actual_whole_model_artifact_bytes"] is None
    assert payload["whole_model_values_actual"] is False
    assert payload["input_error"]["type"] == "FileNotFoundError"
