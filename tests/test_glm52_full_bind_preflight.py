from __future__ import annotations

import hashlib
import json
import re
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from mlx_vq.io.schema import codebook_metadata_for_bits
from mlx_vq.io.source_safetensors import read_safetensors_file_header
from mlx_vq.models.profiles import ModelProfile
from mlx_vq.models.profiles import get_profile
from mlx_vq.validate import glm52_runtime
from mlx_vq.validate.glm52_runtime import (
    expected_glm52_non_vq_runtime_schema,
    expected_glm52_non_vq_source_schema,
    expected_glm52_routed_group_keys,
)


PINNED_REVISION = "6c9241aa05fb243a0edb7c804c213ec1cf5c920d"
PINNED_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--0xSero--glm-5.2-reap-504B-v2"
    / "snapshots"
    / PINNED_REVISION
)
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "benchmarks/probe_glm52_full_bind_preflight.py"
PRODUCTION_PROFILE_PATH = REPO_ROOT / "models/glm52-reap-504b-v2.yaml"
PRODUCTION_NON_VQ_DIR = (
    REPO_ROOT
    / "artifacts/build/glm52_reap_504b_materialization_probe_20260709"
    / "steps/non_vq_pack-a4346e38/out"
)
PRODUCTION_ROUTED_DIR = (
    REPO_ROOT
    / "artifacts/quality/glm52-wave2-materialization-20260709/e8-layers3-4-w1"
)


@dataclass(frozen=True)
class _Fixture:
    profile: ModelProfile
    config_path: Path
    source_index_path: Path
    non_vq_dir: Path
    routed_dir: Path


_DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "U8": 1,
    "U16": 2,
    "U32": 4,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _element_count(shape: tuple[int, ...]) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count


def _write_header_fixture(
    path: Path,
    tensors: dict[str, tuple[str, tuple[int, ...]]],
    *,
    metadata: dict[str, str] | None = None,
) -> None:
    cursor = 0
    header: dict[str, object] = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    for name, (dtype, shape) in sorted(tensors.items()):
        end = cursor + _element_count(shape) * _DTYPE_BYTES[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [cursor, end],
        }
        cursor = end
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(cursor))


def _tiny_profile() -> ModelProfile:
    return ModelProfile(
        name="glm52-full-bind-fixture",
        hf_model_id="fixture/glm52-full-bind",
        revision="fixture-revision",
        architecture="glm_moe_dsa",
        num_layers=2,
        num_sparse_layers=1,
        first_sparse_layer=1,
        hidden_size=16,
        moe_intermediate_size=8,
        num_experts=2,
        experts_per_tok=1,
        vocab_size=32,
        shared_experts=1,
        converter="glm52_vq_groups",
        fused_gate_up=False,
        default_code_bits=8,
        group_size_policy={"gate": 8, "up": 8, "down": 8},
        default_engine="vq_e1_routed_nax_e8",
    )


def _tiny_config() -> dict[str, object]:
    return {
        "model_type": "glm_moe_dsa",
        "attention_bias": False,
        "num_hidden_layers": 2,
        "hidden_size": 16,
        "vocab_size": 32,
        "q_lora_rank": 8,
        "kv_lora_rank": 4,
        "qk_rope_head_dim": 2,
        "qk_nope_head_dim": 4,
        "v_head_dim": 4,
        "num_attention_heads": 2,
        "index_n_heads": 2,
        "index_head_dim": 4,
        "intermediate_size": 32,
        "moe_intermediate_size": 8,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "mlp_layer_types": ["dense", "sparse"],
        "indexer_types": ["full", "shared"],
    }


def _write_non_vq_package(
    root: Path,
    *,
    profile: ModelProfile,
    config: dict[str, object],
    config_sha256: str,
    source_index_sha256: str,
    wrong_dtype_name: str | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    schema = expected_glm52_non_vq_source_schema(config)
    shard_name = "model-00001-of-00001.safetensors"
    tensors = {
        name: (
            "F16" if name == wrong_dtype_name else contract.dtype,
            contract.shape,
        )
        for name, contract in schema.items()
    }
    shard_path = root / shard_name
    _write_header_fixture(shard_path, tensors)
    header = read_safetensors_file_header(shard_path)
    package_index = {
        "metadata": {
            "total_size": sum(
                tensor.data_offsets[1] - tensor.data_offsets[0]
                for tensor in header.tensors.values()
            )
        },
        "weight_map": {name: shard_name for name in sorted(tensors)},
    }
    package_index_path = root / "model.safetensors.index.json"
    _write_json(package_index_path, package_index)

    records: list[dict[str, object]] = []
    dtype_counts: dict[str, int] = {}
    parameter_count = 0
    payload_bytes = 0
    for name in sorted(tensors):
        tensor = header.tensors[name]
        manifest_dtype = schema[name].dtype
        count = _element_count(tensor.shape)
        size = tensor.data_offsets[1] - tensor.data_offsets[0]
        parameter_count += count
        payload_bytes += size
        dtype_counts[manifest_dtype] = dtype_counts.get(manifest_dtype, 0) + 1
        records.append(
            {
                "name": name,
                "dtype": manifest_dtype,
                "shape": list(tensor.shape),
                "parameter_count": count,
                "payload_bytes": size,
                "payload_sha256": "b" * 64,
                "source_shard": "source-00001.safetensors",
                "source_offsets": list(tensor.data_offsets),
                "output_shard": shard_name,
                "output_offsets": list(tensor.data_offsets),
            }
        )
    manifest = {
        "schema_version": 1,
        "record_type": "glm52_non_vq_package_manifest",
        "pack_status": "glm52_non_vq_package_ready",
        "production_ready": True,
        "source_authority": "pinned_huggingface_lfs_v1",
        "selection_policy": "main_model_non_routed_bf16_f32_v1",
        "copy_mode": "raw_safetensors_byte_ranges_v1",
        "profile": profile.name,
        "model_id": profile.hf_model_id,
        "source_revision": profile.revision,
        "config_sha256": config_sha256,
        "index_sha256": source_index_sha256,
        "index_path": "model.safetensors.index.json",
        "source_blob_inventory_sha256": "c" * 64,
        "source_inventory_sha256": "d" * 64,
        "plan_sha256": "e" * 64,
        "package_index_sha256": _sha256(package_index_path),
        "package_set_sha256": "f" * 64,
        "retained_tensor_count": len(records),
        "parameter_count": parameter_count,
        "tensor_payload_bytes": payload_bytes,
        "dtype_tensor_counts": dict(sorted(dtype_counts.items())),
        "shard_count": 1,
        "max_shard_payload_bytes": 1 << 30,
        "excluded_routed_tensor_count": 1,
        "excluded_mtp_tensor_count": 1,
        "excluded_runtime_vq_tensor_count": 0,
        "shards": [
            {
                "filename": shard_name,
                "file_bytes": shard_path.stat().st_size,
                "file_sha256": "a" * 64,
                "tensor_count": len(records),
                "tensor_payload_bytes": payload_bytes,
                "tensor_inventory_sha256": "1" * 64,
            }
        ],
        "tensors": records,
    }
    _write_json(root / "non-vq-manifest.json", manifest)


def _projection_dims(profile: ModelProfile, projection: str) -> tuple[int, int]:
    if projection in {"gate_proj", "up_proj"}:
        return profile.hidden_size, profile.moe_intermediate_size
    return profile.moe_intermediate_size, profile.hidden_size


def _write_routed_package(
    root: Path,
    *,
    profile: ModelProfile,
    config: dict[str, object],
    config_sha256: str,
    source_index_sha256: str,
    selected_keys: tuple[str, ...] | None = None,
    wrong_header_key: str | None = None,
    wrong_codes_dtype_key: str | None = None,
    metadata_mutation: str | None = None,
    lineage_index_sha256: str | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    all_keys = expected_glm52_routed_group_keys(profile, config)
    selected = all_keys if selected_keys is None else selected_keys
    codebook_name, codebook_sha256 = codebook_metadata_for_bits(8)
    group_records: list[dict[str, object]] = []
    artifact_total_bytes = 0
    peak_decoded_expert_bytes = 0
    for group_key in selected:
        layer_raw, projection = group_key.split(":", maxsplit=1)
        layer = int(layer_raw)
        input_dims, output_dims = _projection_dims(profile, projection)
        prefix = f"model.layers.{layer}.mlp.switch_mlp.{projection}"
        codes_shape = (profile.num_experts, output_dims, input_dims // 8)
        header_codes_shape = (
            (profile.num_experts, output_dims // 2, input_dims // 4)
            if group_key == wrong_header_key
            else codes_shape
        )
        scales_shape = (profile.num_experts, output_dims, input_dims // 8)
        quantization_config = {
            "quant_method": "mlx_vq_e8",
            "version": 1,
            "default_code_bits": 8,
            "default_group_size": 8,
            "codebook": {
                "name": codebook_name,
                "dtype": "uint32",
                "entries": 256,
                "sha256": codebook_sha256,
            },
            "policy": {
                "scale_estimator": "max_abs",
                "source_weight_encoding": "modelopt_nvfp4",
                "source_decoder": "modelopt_nvfp4_v1",
                "source_model_id": profile.hf_model_id,
                "source_revision": profile.revision,
                "source_config_sha256": config_sha256,
                "source_index_sha256": source_index_sha256,
                "source_profile": profile.name,
                "decoded_expert_working_set": "one_per_worker",
            },
        }
        if group_key == all_keys[0]:
            if metadata_mutation == "code_bits":
                quantization_config["default_code_bits"] = 16
            elif metadata_mutation == "group_size":
                quantization_config["default_group_size"] = 4
            elif metadata_mutation == "codebook_name":
                quantization_config["codebook"]["name"] = "wrong_codebook"
        filename = f"layer-{layer:05d}-{projection}.safetensors"
        path = root / filename
        _write_header_fixture(
            path,
            {
                f"{prefix}.codes": (
                    "U16" if group_key == wrong_codes_dtype_key else "U8",
                    header_codes_shape,
                ),
                f"{prefix}.scales": ("F16", scales_shape),
                "model.vq_codebook.e8": ("U32", (256,)),
            },
            metadata={"quantization_config": json.dumps(quantization_config)},
        )
        artifact_total_bytes += path.stat().st_size
        decoded_bytes = input_dims * output_dims * 4
        peak_decoded_expert_bytes = max(peak_decoded_expert_bytes, decoded_bytes)
        group_records.append(
            {
                "layer": layer,
                "projection": projection,
                "status": "ready",
                "artifact_path": str(path),
                "artifact_bytes": path.stat().st_size,
                "artifact_sha256": "2" * 64,
                "expert_count": profile.num_experts,
                "codes_name": f"{prefix}.codes",
                "codes_dtype": "uint8",
                "codes_shape": list(codes_shape),
                "scales_name": f"{prefix}.scales",
                "scales_dtype": "float16",
                "scales_shape": list(scales_shape),
                "decoded_expert_bytes": decoded_bytes,
                "source_bundles": profile.num_experts,
                "source_bundle_members": profile.num_experts * 3,
                "source_shards": ["source-00001.safetensors"],
                "cross_shard_bundle_count": 0,
            }
        )
    missing = len(all_keys) - len(selected)
    manifest = {
        "schema_version": 1,
        "record_type": "glm52_modelopt_nvfp4_materialization_manifest",
        "materialization_status": "glm52_modelopt_nvfp4_groups_ready",
        "materialization_scope": "full" if not missing else "bounded",
        "materialization_blocked": False,
        "materialization_blockers": [],
        "full_group_coverage": not missing,
        "profile": profile.name,
        "model_id": profile.hf_model_id,
        "source_revision": profile.revision,
        "config_sha256": config_sha256,
        "index_sha256": lineage_index_sha256 or source_index_sha256,
        "source_weight_encoding": "modelopt_nvfp4",
        "source_decoder": "modelopt_nvfp4_v1",
        "working_set_policy": "one_decoded_expert_per_worker_v1",
        "planned_vq_groups": len(all_keys),
        "selected_vq_groups": len(selected),
        "ready_vq_groups": len(selected),
        "skipped_vq_groups": missing,
        "selected_group_keys": list(selected),
        "code_bits": 8,
        "group_size": 8,
        "codebook_name": codebook_name,
        "codebook_sha256": codebook_sha256,
        "scale_estimator": "max_abs",
        "dense_checkpoint_written": False,
        "artifact_total_bytes": artifact_total_bytes,
        "peak_decoded_expert_bytes": peak_decoded_expert_bytes,
        "source_bundles": len(selected) * profile.num_experts,
        "source_bundle_members": len(selected) * profile.num_experts * 3,
        "groups": group_records,
    }
    _write_json(root / "conversion-manifest.json", manifest)


def _write_fixture(
    tmp_path: Path,
    *,
    selected_keys: tuple[str, ...] | None = None,
    wrong_non_vq_dtype: str | None = None,
    wrong_routed_header: str | None = None,
    wrong_routed_codes_dtype: str | None = None,
    routed_metadata_mutation: str | None = None,
    lineage_index_sha256: str | None = None,
    source_index_shard: str = "source-00001.safetensors",
) -> _Fixture:
    profile = _tiny_profile()
    config = _tiny_config()
    config_path = tmp_path / "config.json"
    source_index_path = tmp_path / "source-index.json"
    _write_json(config_path, config)
    _write_json(
        source_index_path,
        {
            "metadata": {},
            "weight_map": {
                name: source_index_shard
                for name in expected_glm52_non_vq_source_schema(config)
            },
        },
    )
    config_sha256 = _sha256(config_path)
    source_index_sha256 = _sha256(source_index_path)
    non_vq_dir = tmp_path / "non-vq"
    routed_dir = tmp_path / "routed"
    _write_non_vq_package(
        non_vq_dir,
        profile=profile,
        config=config,
        config_sha256=config_sha256,
        source_index_sha256=source_index_sha256,
        wrong_dtype_name=wrong_non_vq_dtype,
    )
    _write_routed_package(
        routed_dir,
        profile=profile,
        config=config,
        config_sha256=config_sha256,
        source_index_sha256=source_index_sha256,
        selected_keys=selected_keys,
        wrong_header_key=wrong_routed_header,
        wrong_codes_dtype_key=wrong_routed_codes_dtype,
        metadata_mutation=routed_metadata_mutation,
        lineage_index_sha256=lineage_index_sha256,
    )
    return _Fixture(
        profile=profile,
        config_path=config_path,
        source_index_path=source_index_path,
        non_vq_dir=non_vq_dir,
        routed_dir=routed_dir,
    )


def _run_preflight(fixture: _Fixture):
    return glm52_runtime.preflight_glm52_full_bind(
        profile=fixture.profile,
        config_path=fixture.config_path,
        source_index_path=fixture.source_index_path,
        non_vq_artifact_dir=fixture.non_vq_dir,
        routed_artifact_dir=fixture.routed_dir,
        model_id=fixture.profile.hf_model_id,
        revision=str(fixture.profile.revision),
    )


def test_pinned_full_bind_schema_is_exact_without_constructing_the_model() -> None:
    config_path = PINNED_SNAPSHOT / "config.json"
    if not config_path.is_file():
        pytest.skip("pinned GLM52 config is not cached")
    config = json.loads(config_path.read_text())
    profile = get_profile("glm52-reap-504b-v2")

    schema = expected_glm52_non_vq_source_schema(config)
    runtime_schema = expected_glm52_non_vq_runtime_schema(config)
    group_keys = expected_glm52_routed_group_keys(profile, config)

    assert len(schema) == 1_194
    assert len(runtime_schema) == 1_272
    assert sum(name.endswith(".self_attn.kv_b_proj.weight") for name in schema) == 78
    assert not any(name.endswith(".self_attn.kv_b_proj.weight") for name in runtime_schema)
    assert runtime_schema["model.layers.0.self_attn.embed_q.weight"].shape == (
        64,
        512,
        192,
    )
    assert runtime_schema["model.layers.0.self_attn.unembed_out.weight"].shape == (
        64,
        256,
        512,
    )
    assert len(group_keys) == 225
    assert group_keys[:3] == (
        "3:gate_proj",
        "3:up_proj",
        "3:down_proj",
    )
    assert group_keys[-3:] == (
        "77:gate_proj",
        "77:up_proj",
        "77:down_proj",
    )
    assert not any(key.startswith("78:") for key in group_keys)


def test_complete_tiny_full_bind_preflight_is_header_only_and_ready(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)

    report = _run_preflight(fixture)

    assert report.preflight_pass
    assert report.preflight_status == "glm52_full_bind_preflight_ready"
    assert report.blockers == ()
    assert report.profile_path is None
    assert report.profile_sha256 is None
    assert len(report.profile_contract_sha256) == 64
    assert report.expected_routed_group_count == 3
    assert report.present_routed_group_keys == (
        "1:gate_proj",
        "1:up_proj",
        "1:down_proj",
    )
    assert report.missing_routed_group_keys == ()
    assert report.non_vq_tensor_count == len(
        expected_glm52_non_vq_source_schema(_tiny_config())
    )
    assert report.non_vq_runtime_target_count == report.non_vq_tensor_count + 2
    assert report.kv_b_source_tensor_count == 2
    assert report.kv_b_runtime_target_count == 4
    assert report.routed_header_file_count == 3
    assert report.tensor_payloads_read is False
    assert report.payload_hashes_verified is False
    assert report.full_model_constructed is False
    assert report.dense_routed_experts is False
    assert report.production_binding_proven is False
    assert report.production_generation_proven is False


def test_bounded_tiny_preflight_blocks_with_exact_missing_group(tmp_path: Path) -> None:
    fixture = _write_fixture(
        tmp_path,
        selected_keys=("1:gate_proj", "1:up_proj"),
    )

    report = _run_preflight(fixture)

    assert not report.preflight_pass
    assert report.preflight_status == "glm52_full_bind_preflight_blocked"
    assert report.blockers == ("missing_routed_groups=1",)
    assert report.missing_routed_group_keys == ("1:down_proj",)
    assert report.present_routed_group_keys == ("1:gate_proj", "1:up_proj")
    assert report.full_model_constructed is False


def test_preflight_rejects_wrong_non_vq_header_dtype(tmp_path: Path) -> None:
    wrong_name = "model.layers.0.input_layernorm.weight"
    fixture = _write_fixture(tmp_path, wrong_non_vq_dtype=wrong_name)

    with pytest.raises(
        ValueError,
        match=r"non-VQ header tensor .*input_layernorm\.weight dtype must be BF16, found F16",
    ):
        _run_preflight(fixture)


def test_preflight_rejects_unexpected_routed_mtp_tensor_in_non_vq_index(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    package_index_path = fixture.non_vq_dir / "model.safetensors.index.json"
    manifest_path = fixture.non_vq_dir / "non-vq-manifest.json"
    package_index = json.loads(package_index_path.read_text())
    unexpected = "model.layers.2.mlp.experts.0.gate_proj.weight"
    package_index["weight_map"][unexpected] = "model-00001-of-00001.safetensors"
    _write_json(package_index_path, package_index)
    manifest = json.loads(manifest_path.read_text())
    manifest["package_index_sha256"] = _sha256(package_index_path)
    _write_json(manifest_path, manifest)

    with pytest.raises(
        ValueError,
        match=r"unexpected_non_vq_tensors=.*model\.layers\.2\.mlp\.experts\.0\.gate_proj\.weight",
    ):
        _run_preflight(fixture)


def test_preflight_rejects_wrong_present_vq_header_shape(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path, wrong_routed_header="1:gate_proj")

    with pytest.raises(
        ValueError,
        match=r"layer-00001-gate_proj\.safetensors codes shape must be \(2, 8, 2\), found \(2, 4, 4\)",
    ):
        _run_preflight(fixture)


@pytest.mark.parametrize(
    ("fixture_kwargs", "expected_message"),
    [
        (
            {"wrong_routed_codes_dtype": "1:gate_proj"},
            "codes dtype must be U8, found U16",
        ),
        (
            {"routed_metadata_mutation": "code_bits"},
            "code bits must be 8, found 16",
        ),
        (
            {"routed_metadata_mutation": "group_size"},
            "group size must be 8, found 4",
        ),
        (
            {"routed_metadata_mutation": "codebook_name"},
            "codebook name must be 'quip_e8', found 'wrong_codebook'",
        ),
    ],
    ids=("codes-dtype", "code-bits", "group-size", "codebook"),
)
def test_preflight_rejects_present_vq_header_and_metadata_contract_drift(
    tmp_path: Path,
    fixture_kwargs: dict[str, object],
    expected_message: str,
) -> None:
    fixture = _write_fixture(tmp_path, **fixture_kwargs)

    with pytest.raises(ValueError, match=re.escape(expected_message)):
        _run_preflight(fixture)


def test_preflight_rejects_routed_source_index_lineage_drift(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path, lineage_index_sha256="9" * 64)

    with pytest.raises(
        ValueError,
        match="routed manifest source index SHA-256",
    ):
        _run_preflight(fixture)


def test_preflight_rejects_non_vq_source_shard_drift_from_pinned_index(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(
        tmp_path,
        source_index_shard="different-source.safetensors",
    )

    with pytest.raises(
        ValueError,
        match=r"non-VQ tensor .* source_shard must match pinned source index",
    ):
        _run_preflight(fixture)


def test_preflight_rejects_unmanifested_layer2_mtp_routed_file(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    shutil.copyfile(
        fixture.routed_dir / "layer-00001-gate_proj.safetensors",
        fixture.routed_dir / "layer-00002-gate_proj.safetensors",
    )

    with pytest.raises(
        ValueError,
        match=r"unexpected_routed_safetensors_files=.*layer-00002-gate_proj\.safetensors",
    ):
        _run_preflight(fixture)


def test_preflight_rejects_unindexed_non_vq_safetensors_file(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    shutil.copyfile(
        fixture.non_vq_dir / "model-00001-of-00001.safetensors",
        fixture.non_vq_dir / "unindexed.safetensors",
    )

    with pytest.raises(
        ValueError,
        match=r"unexpected_non_vq_safetensors_files=.*unindexed\.safetensors",
    ):
        _run_preflight(fixture)


def test_preflight_rejects_non_vq_shard_through_symlinked_parent(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    shard_name = "model-00001-of-00001.safetensors"
    outside = tmp_path / "outside"
    outside.mkdir()
    shutil.move(
        fixture.non_vq_dir / shard_name,
        outside / shard_name,
    )
    (fixture.non_vq_dir / "link").symlink_to(outside, target_is_directory=True)
    linked_name = f"link/{shard_name}"

    package_index_path = fixture.non_vq_dir / "model.safetensors.index.json"
    package_index = json.loads(package_index_path.read_text())
    package_index["weight_map"] = {
        name: linked_name for name in package_index["weight_map"]
    }
    _write_json(package_index_path, package_index)
    manifest_path = fixture.non_vq_dir / "non-vq-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for record in manifest["tensors"]:
        record["output_shard"] = linked_name
    manifest["shards"][0]["filename"] = linked_name
    manifest["package_index_sha256"] = _sha256(package_index_path)
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="symlink|escapes artifact root"):
        _run_preflight(fixture)


@pytest.mark.skipif(
    not (PINNED_SNAPSHOT / "config.json").is_file()
    or not (PINNED_SNAPSHOT / "model.safetensors.index.json").is_file()
    or not PRODUCTION_PROFILE_PATH.is_file()
    or not PRODUCTION_NON_VQ_DIR.is_dir()
    or not PRODUCTION_ROUTED_DIR.is_dir(),
    reason="pinned GLM52 production preflight inputs are not resident",
)
def test_real_bounded_cli_writes_exact_219_group_blocker_without_model_load(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "glm52-full-bind-preflight.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--profile-path",
            str(PRODUCTION_PROFILE_PATH),
            "--config-path",
            str(PINNED_SNAPSHOT / "config.json"),
            "--source-index-path",
            str(PINNED_SNAPSHOT / "model.safetensors.index.json"),
            "--non-vq-artifact-dir",
            str(PRODUCTION_NON_VQ_DIR),
            "--routed-artifact-dir",
            str(PRODUCTION_ROUTED_DIR),
            "--model-id",
            "0xSero/glm-5.2-reap-504B-v2",
            "--revision",
            PINNED_REVISION,
            "--output-json",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2, result.stderr
    payload = json.loads(output_path.read_text())
    assert payload["preflight_status"] == "glm52_full_bind_preflight_blocked"
    assert payload["preflight_pass"] is False
    assert payload["blockers"] == ["missing_routed_groups=219"]
    assert payload["profile_path"] == str(PRODUCTION_PROFILE_PATH.resolve())
    assert payload["profile_sha256"] == _sha256(PRODUCTION_PROFILE_PATH)
    assert len(payload["profile_contract_sha256"]) == 64
    assert payload["expected_routed_group_count"] == 225
    assert payload["present_routed_group_count"] == 6
    assert len(payload["missing_routed_group_keys"]) == 219
    assert payload["missing_routed_group_keys"][:3] == [
        "5:gate_proj",
        "5:up_proj",
        "5:down_proj",
    ]
    assert payload["missing_routed_group_keys"][-3:] == [
        "77:gate_proj",
        "77:up_proj",
        "77:down_proj",
    ]
    assert payload["non_vq_tensor_count"] == 1_194
    assert payload["non_vq_runtime_target_count"] == 1_272
    assert payload["kv_b_source_tensor_count"] == 78
    assert payload["kv_b_runtime_target_count"] == 156
    assert payload["routed_header_file_count"] == 6
    assert payload["tensor_payloads_read"] is False
    assert payload["payload_hashes_verified"] is False
    assert payload["full_model_constructed"] is False
    assert payload["dense_routed_experts"] is False
    assert list(tmp_path.glob(".glm52-full-bind-preflight.json.*.tmp")) == []

    evidence_only_output = tmp_path / "glm52-full-bind-evidence.json"
    evidence_only = subprocess.run(
        [
            *result.args[:-1],
            str(evidence_only_output),
            "--evidence-only",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert evidence_only.returncode == 0, evidence_only.stderr
    assert json.loads(evidence_only_output.read_text())["preflight_pass"] is False


def test_cli_rejects_output_aliasing_config_without_mutation(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    before = fixture.config_path.read_bytes()

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--profile-path",
            str(PRODUCTION_PROFILE_PATH),
            "--config-path",
            str(fixture.config_path),
            "--source-index-path",
            str(fixture.source_index_path),
            "--non-vq-artifact-dir",
            str(fixture.non_vq_dir),
            "--routed-artifact-dir",
            str(fixture.routed_dir),
            "--model-id",
            fixture.profile.hf_model_id,
            "--revision",
            str(fixture.profile.revision),
            "--output-json",
            str(fixture.config_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert fixture.config_path.read_bytes() == before
    assert "output JSON" in result.stderr
