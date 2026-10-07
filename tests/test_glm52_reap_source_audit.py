from __future__ import annotations

import json
import subprocess
import struct
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from benchmarks.materialize_glm52_reap_groups import (
    _parse_group_keys,
    _select_groups,
    _sha256_file,
    _validate_prior_artifact_identity,
)
from keep.convert.glm52_reap import (
    GLM52_REAP_CONFIG_SHA256,
    GLM52_REAP_INDEX_SHA256,
    GLM52_REAP_PROFILE_NAME,
    audit_glm52_reap_source_index,
    audit_glm52_reap_source_payloads,
)
from keep.convert.stream_convert import (
    SafetensorsIndex,
    SourceWeightEncoding,
    VQExpertGroup,
    load_safetensors_index,
    vq_group_output_filename,
)
from ramp.models.profiles import get_profile


def _fixture_profile(*, experts: int = 1):
    return replace(
        get_profile("glm52-reap-504b-v2"),
        name="glm52-reap-fixture",
        hf_model_id="fixture/glm52-reap",
        revision="fixture-revision",
        num_experts=experts,
    )


def _fixture_config(*, experts: int = 1) -> dict[str, object]:
    return {
        "model_type": "glm_moe_dsa",
        "vocab_size": 154880,
        "hidden_size": 6144,
        "index_head_dim": 128,
        "index_n_heads": 32,
        "index_topk": 2048,
        "intermediate_size": 12288,
        "moe_intermediate_size": 2048,
        "num_hidden_layers": 78,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "n_shared_experts": 1,
        "n_routed_experts": experts,
        "routed_scaling_factor": 2.5,
        "kv_lora_rank": 512,
        "q_lora_rank": 2048,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "qk_nope_head_dim": 192,
        "topk_method": "noaux_tc",
        "scoring_func": "sigmoid",
        "norm_topk_prob": True,
        "n_group": 1,
        "topk_group": 1,
        "num_experts_per_tok": 8,
        "moe_layer_freq": 1,
        "first_k_dense_replace": 3,
        "max_position_embeddings": 1048576,
        "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"},
        "index_topk_pattern": None,
        "index_topk_freq": 4,
        "index_skip_topk_offset": 3,
        "attention_bias": False,
        "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 75,
        "indexer_types": ["full", "full", "full"]
        + [
            "full"
            if layer
            in {6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74}
            else "shared"
            for layer in range(3, 78)
        ],
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "producer": {"name": "modelopt", "version": "fixture-v1"},
            "config_groups": {
                "group_0": {
                    "weights": {
                        "dynamic": False,
                        "num_bits": 4,
                        "type": "float",
                        "group_size": 16,
                    },
                    "input_activations": {
                        "dynamic": False,
                        "num_bits": 4,
                        "type": "float",
                        "group_size": 16,
                    },
                    "targets": ["Linear"],
                }
            },
        },
    }


def _bundle_names(layer: int, expert: int, projection: str) -> tuple[str, str, str]:
    weight = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
    return weight, f"{weight}_scale", f"{weight}_scale_2"


def _fixture_index(
    *,
    experts: int = 1,
    selected_shard: str = "main.safetensors",
    other_shard: str | None = None,
    include_mtp: bool = True,
) -> SafetensorsIndex:
    weight_map: dict[str, str] = {}
    for layer in range(3, 78):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for expert in range(experts):
                shard = (
                    selected_shard
                    if layer == 3 and projection == "gate_proj"
                    else (other_shard or selected_shard)
                )
                for name in _bundle_names(layer, expert, projection):
                    weight_map[name] = shard
    if include_mtp:
        weight_map["model.layers.78.mlp.experts.0.gate_proj.weight"] = "mtp.safetensors"
    return SafetensorsIndex(metadata={}, weight_map=weight_map)


def _write_raw_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
    *,
    truncate_payload: bool = False,
) -> None:
    header: dict[str, object] = {}
    payload = bytearray()
    for name, (dtype, shape, raw) in tensors.items():
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    raw_payload = b"" if truncate_payload else bytes(payload)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw_payload)


def _write_overlapping_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
) -> None:
    header = {
        name: {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [0, len(raw)],
        }
        for name, (dtype, shape, raw) in tensors.items()
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    payload = b"".join(raw for _, _, raw in tensors.values())
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _selected_gate_tensors(
    *,
    wrong_weight_shape: bool = False,
    wrong_weight_dtype: bool = False,
    wrong_weight_name: bool = False,
):
    weight, block_scale, global_scale = _bundle_names(3, 0, "gate_proj")
    weight_shape = (2048, 1) if wrong_weight_shape else (2048, 3072)
    written_weight = f"{weight}.wrong" if wrong_weight_name else weight
    return {
        written_weight: (
            "I8" if wrong_weight_dtype else "U8",
            weight_shape,
            bytes(np.prod(weight_shape, dtype=np.int64)),
        ),
        block_scale: ("F8_E4M3", (2048, 384), bytes(2048 * 384)),
        global_scale: ("F32", (), np.float32(1.0).tobytes()),
    }


def _source_audit(*, index: SafetensorsIndex | None = None):
    profile = _fixture_profile()
    return audit_glm52_reap_source_index(
        config=_fixture_config(),
        index=index or _fixture_index(),
        model_id=profile.hf_model_id,
        revision=profile.revision,
        profile=profile,
        enforce_pinned_profile=False,
    )


def test_glm52_reap_source_audit_reports_every_main_group_and_excludes_mtp() -> None:
    audit = _source_audit()
    payload = audit.to_json_dict()

    assert audit.source_checks_pass is True
    assert payload["record_type"] == "glm52_modelopt_nvfp4_source_audit"
    assert payload["source_weight_encoding"] == "modelopt_nvfp4"
    assert payload["source_decoder"] == "modelopt_nvfp4_v1"
    assert payload["expected_groups"] == payload["planned_groups"] == 225
    assert payload["expected_bundles"] == payload["resolved_bundles"] == 225
    assert payload["required_bundle_members"] == 675
    assert payload["main_layer_ids"] == list(range(3, 78))
    assert payload["excluded_mtp_layer_ids"] == [78]
    assert payload["excluded_mtp_tensors"] == 1
    assert all(group["layer"] != 78 for group in payload["groups"])
    assert len(payload["groups"]) == 225
    assert payload["checks"]["profile_config"] is True
    assert payload["checks"]["complete_bundle_index"] is True


@pytest.mark.parametrize("member", ["weight", "block_scale", "global_scale"])
def test_glm52_reap_source_audit_fails_on_each_missing_bundle_member(member: str) -> None:
    index = _fixture_index()
    names = dict(zip(("weight", "block_scale", "global_scale"), _bundle_names(3, 0, "gate_proj")))
    del index.weight_map[names[member]]

    audit = _source_audit(index=index)

    assert audit.source_checks_pass is False
    assert names[member] in audit.missing_companion_tensors
    assert audit.to_json_dict()["checks"]["complete_bundle_index"] is False


def test_glm52_reap_source_audit_resolves_cross_shard_companions() -> None:
    index = _fixture_index()
    weight, block_scale, global_scale = _bundle_names(3, 0, "gate_proj")
    index.weight_map[weight] = "packed.safetensors"
    index.weight_map[block_scale] = "blocks.safetensors"
    index.weight_map[global_scale] = "globals.safetensors"

    audit = _source_audit(index=index)
    first_group = audit.to_json_dict()["groups"][0]

    assert audit.source_checks_pass is True
    assert audit.cross_shard_bundles == 1
    assert first_group["cross_shard_bundles"] == 1
    assert first_group["required_shards"] == [
        "blocks.safetensors",
        "globals.safetensors",
        "packed.safetensors",
    ]


def test_glm52_reap_source_audit_rejects_orphan_routed_companions() -> None:
    index = _fixture_index()
    orphan = "model.layers.2.mlp.experts.0.gate_proj.weight_scale"
    index.weight_map[orphan] = "main.safetensors"

    audit = _source_audit(index=index)

    assert audit.source_checks_pass is False
    assert orphan in audit.unexpected_routed_tensors


@pytest.mark.parametrize("unsafe_shard", ["/tmp/outside.safetensors", "../outside.safetensors"])
def test_glm52_reap_source_audit_rejects_escaping_shard_names(unsafe_shard: str) -> None:
    index = _fixture_index()
    index.weight_map[_bundle_names(3, 0, "gate_proj")[0]] = unsafe_shard

    audit = _source_audit(index=index)

    assert audit.source_checks_pass is False
    assert unsafe_shard in audit.invalid_source_shards


def test_glm52_reap_source_audit_fails_wrong_revision_and_encoding() -> None:
    profile = _fixture_profile()
    wrong_revision = audit_glm52_reap_source_index(
        config=_fixture_config(),
        index=_fixture_index(),
        model_id=profile.hf_model_id,
        revision="wrong",
        profile=profile,
        enforce_pinned_profile=False,
    )
    config = _fixture_config()
    config["quantization_config"] = {"quant_method": "bitsandbytes"}
    wrong_encoding = audit_glm52_reap_source_index(
        config=config,
        index=_fixture_index(),
        model_id=profile.hf_model_id,
        revision=profile.revision,
        profile=profile,
        enforce_pinned_profile=False,
    )

    assert wrong_revision.source_checks_pass is False
    assert wrong_revision.to_json_dict()["checks"]["pinned_revision"] is False
    assert wrong_encoding.source_checks_pass is False
    assert wrong_encoding.to_json_dict()["checks"]["native_modelopt_nvfp4"] is False
    assert "quantization_config" in wrong_encoding.source_encoding_error


def test_pinned_profile_name_cannot_redefine_the_168_expert_contract() -> None:
    profile = replace(_fixture_profile(), name=GLM52_REAP_PROFILE_NAME)

    audit = audit_glm52_reap_source_index(
        config=_fixture_config(),
        index=_fixture_index(),
        model_id=profile.hf_model_id,
        revision=profile.revision,
        profile=profile,
        config_sha256=GLM52_REAP_CONFIG_SHA256,
        index_sha256=GLM52_REAP_INDEX_SHA256,
    )

    assert audit.source_checks_pass is False
    assert audit.to_json_dict()["checks"]["pinned_profile_definition"] is False
    assert audit.to_json_dict()["checks"]["pinned_expert_count"] is True
    assert audit.to_json_dict()["checks"]["pinned_bundle_count"] is True


def test_pinned_audit_requires_explicit_raw_file_hashes() -> None:
    profile = get_profile(GLM52_REAP_PROFILE_NAME)

    with pytest.raises(ValueError, match="requires raw config_sha256 and index_sha256"):
        audit_glm52_reap_source_index(
            config=_fixture_config(),
            index=_fixture_index(),
            model_id=profile.hf_model_id,
            revision=profile.revision,
            profile=profile,
        )


def test_pinned_revision_rejects_structurally_valid_wrong_file_hash() -> None:
    profile = get_profile(GLM52_REAP_PROFILE_NAME)
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--0xSero--glm-5.2-reap-504B-v2/snapshots"
        / profile.revision
    )
    config_path = snapshot / "config.json"
    index_path = snapshot / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        pytest.skip("pinned GLM52 REAP config/index are not cached")
    config = json.loads(config_path.read_text())
    config["structurally_irrelevant_mutation"] = True

    audit = audit_glm52_reap_source_index(
        config=config,
        index=load_safetensors_index(index_path),
        model_id=profile.hf_model_id,
        revision=profile.revision,
        profile=profile,
        config_sha256="0" * 64,
        index_sha256=GLM52_REAP_INDEX_SHA256,
    )

    assert audit.source_checks_pass is False
    assert audit.to_json_dict()["checks"]["pinned_config_sha256"] is False
    assert audit.to_json_dict()["checks"]["pinned_index_sha256"] is True


def test_glm52_reap_payload_audit_validates_headers_without_decoding(tmp_path: Path) -> None:
    index = _fixture_index(selected_shard="selected.safetensors", other_shard="other.safetensors")
    source_audit = _source_audit(index=index)
    _write_raw_safetensors(tmp_path / "selected.safetensors", _selected_gate_tensors())

    payload = audit_glm52_reap_source_payloads(
        source_dir=tmp_path,
        source_audit=source_audit,
        max_groups=1,
    )

    assert payload["payload_status"] == "glm52_modelopt_nvfp4_source_payloads_bounded_ready"
    assert payload["payload_scope"] == "bounded"
    assert payload["full_payload_ready"] is False
    assert payload["materialization_blocked"] is False
    assert payload["selected_groups"] == 1
    assert payload["selected_bundles"] == 1
    assert payload["header_checked_tensors"] == 3
    assert payload["required_shards"] == ["selected.safetensors"]
    assert payload["invalid_tensor_headers"] == []


def test_glm52_reap_payload_audit_selects_exact_nonprefix_group(tmp_path: Path) -> None:
    source_audit = _source_audit(
        index=_fixture_index(
            selected_shard="first.safetensors",
            other_shard="later.safetensors",
        )
    )

    payload = audit_glm52_reap_source_payloads(
        source_dir=tmp_path,
        source_audit=source_audit,
        group_keys=((29, "gate_proj"),),
    )

    assert payload["selected_groups"] == 1
    assert payload["requested_groups"] == ["29:gate_proj"]
    assert payload["payload_scope"] == "bounded"
    assert payload["full_payload_ready"] is False
    assert payload["missing_shards"] == ["later.safetensors"]


def test_glm52_reap_payload_audit_rejects_duplicate_or_unknown_exact_groups(
    tmp_path: Path,
) -> None:
    source_audit = _source_audit()

    with pytest.raises(ValueError, match="duplicate"):
        audit_glm52_reap_source_payloads(
            source_dir=tmp_path,
            source_audit=source_audit,
            group_keys=((3, "gate_proj"), (3, "gate_proj")),
        )
    with pytest.raises(ValueError, match="unplanned"):
        audit_glm52_reap_source_payloads(
            source_dir=tmp_path,
            source_audit=source_audit,
            group_keys=((78, "gate_proj"),),
        )


def test_glm52_materializer_requires_explicit_selection_and_rejects_layer_78() -> None:
    group = VQExpertGroup(
        layer=10,
        projection="gate_proj",
        experts=(0,),
        source_shards=("source.safetensors",),
        input_dims=16,
        output_dims=16,
        code_bits=8,
        group_size=8,
        source_weight_encoding=SourceWeightEncoding.MODELOPT_NVFP4,
    )

    with pytest.raises(ValueError, match="select exactly one"):
        _select_groups(
            (group,),
            group_keys=None,
            max_groups=None,
            all_groups=False,
        )
    assert _select_groups(
        (group,),
        group_keys=None,
        max_groups=None,
        all_groups=True,
    ) == (group,)
    with pytest.raises(ValueError, match="3..77"):
        _parse_group_keys("78:gate_proj")


def test_glm52_materializer_prior_manifest_detects_group_byte_drift(
    tmp_path: Path,
) -> None:
    group = VQExpertGroup(
        layer=10,
        projection="gate_proj",
        experts=(0,),
        source_shards=("source.safetensors",),
        input_dims=16,
        output_dims=16,
        code_bits=8,
        group_size=8,
        source_weight_encoding=SourceWeightEncoding.MODELOPT_NVFP4,
    )
    artifact = tmp_path / vq_group_output_filename(group)
    artifact.write_bytes(b"deterministic-group")
    prior = {
        "record_type": "glm52_modelopt_nvfp4_materialization_manifest",
        "profile": "glm52-reap-504b-v2",
        "model_id": "0xSero/glm-5.2-reap-504B-v2",
        "source_revision": "revision",
        "config_sha256": "a" * 64,
        "index_sha256": "b" * 64,
        "source_weight_encoding": "modelopt_nvfp4",
        "source_decoder": "modelopt_nvfp4_v1",
        "code_bits": 8,
        "group_size": 512,
        "scale_estimator": "max_abs",
        "selected_group_keys": ["10:gate_proj"],
        "groups": [
            {
                "layer": 10,
                "projection": "gate_proj",
                "artifact_bytes": artifact.stat().st_size,
                "artifact_sha256": _sha256_file(artifact),
            }
        ],
    }
    kwargs = {
        "output_dir": tmp_path,
        "prior": prior,
        "selected": (group,),
        "profile": "glm52-reap-504b-v2",
        "model_id": "0xSero/glm-5.2-reap-504B-v2",
        "revision": "revision",
        "config_sha256": "a" * 64,
        "index_sha256": "b" * 64,
        "code_bits": 8,
        "group_size": 512,
        "scale_estimator": "max_abs",
    }

    assert _validate_prior_artifact_identity(**kwargs) == {
        (10, "gate_proj"): _sha256_file(artifact)
    }
    artifact.write_bytes(b"X" * artifact.stat().st_size)
    with pytest.raises(ValueError, match="SHA-256 drifted"):
        _validate_prior_artifact_identity(**kwargs)


def test_glm52_reap_payload_audit_rejects_overlapping_shard_extents(tmp_path: Path) -> None:
    index = _fixture_index(selected_shard="selected.safetensors", other_shard="other.safetensors")
    source_audit = _source_audit(index=index)
    _write_overlapping_safetensors(
        tmp_path / "selected.safetensors",
        _selected_gate_tensors(),
    )

    payload = audit_glm52_reap_source_payloads(
        source_dir=tmp_path,
        source_audit=source_audit,
        max_groups=1,
    )

    assert payload["payload_status"] == "glm52_modelopt_nvfp4_source_payloads_invalid"
    assert payload["materialization_blocked"] is True
    assert any("overlapping tensor extent" in error for error in payload["invalid_shards"])


def test_glm52_reap_payload_audit_never_readies_an_incomplete_source(tmp_path: Path) -> None:
    index = _fixture_index()
    del index.weight_map[_bundle_names(3, 0, "gate_proj")[1]]
    source_audit = _source_audit(index=index)

    payload = audit_glm52_reap_source_payloads(
        source_dir=tmp_path,
        source_audit=source_audit,
        max_groups=1,
    )

    assert payload["payload_status"] == "glm52_modelopt_nvfp4_source_audit_failed"
    assert payload["materialization_blocked"] is True
    assert "repair_source_index_or_nvfp4_companions" in payload["materialization_blockers"]


@pytest.mark.parametrize(
    ("write_kind", "expected_fragment"),
    [
        ("missing", "selected.safetensors"),
        ("directory", "selected.safetensors"),
        ("truncated", "declared file extent"),
        ("wrong_dtype", "expected dtype"),
        ("wrong_name", "missing tensor header"),
        ("wrong_shape", "expected shape"),
    ],
)
def test_glm52_reap_payload_audit_blocks_missing_or_invalid_payloads(
    tmp_path: Path,
    write_kind: str,
    expected_fragment: str,
) -> None:
    index = _fixture_index(selected_shard="selected.safetensors", other_shard="other.safetensors")
    source_audit = _source_audit(index=index)
    if write_kind == "directory":
        (tmp_path / "selected.safetensors").mkdir()
    elif write_kind == "truncated":
        _write_raw_safetensors(
            tmp_path / "selected.safetensors",
            _selected_gate_tensors(),
            truncate_payload=True,
        )
    elif write_kind == "wrong_shape":
        _write_raw_safetensors(
            tmp_path / "selected.safetensors",
            _selected_gate_tensors(wrong_weight_shape=True),
        )
    elif write_kind == "wrong_dtype":
        _write_raw_safetensors(
            tmp_path / "selected.safetensors",
            _selected_gate_tensors(wrong_weight_dtype=True),
        )
    elif write_kind == "wrong_name":
        _write_raw_safetensors(
            tmp_path / "selected.safetensors",
            _selected_gate_tensors(wrong_weight_name=True),
        )

    payload = audit_glm52_reap_source_payloads(
        source_dir=tmp_path,
        source_audit=source_audit,
        max_groups=1,
    )

    assert payload["materialization_blocked"] is True
    assert payload["payload_status"] in {
        "glm52_modelopt_nvfp4_source_payloads_missing",
        "glm52_modelopt_nvfp4_source_payloads_invalid",
    }
    evidence = json.dumps(
        payload["missing_shards"]
        + payload["invalid_shards"]
        + payload["invalid_tensor_headers"]
    )
    assert expected_fragment in evidence


def test_glm52_reap_payload_audit_rejects_oversized_group_selection(tmp_path: Path) -> None:
    source_audit = _source_audit()

    payload = audit_glm52_reap_source_payloads(
        source_dir=tmp_path,
        source_audit=source_audit,
        max_groups=226,
    )

    assert payload["materialization_blocked"] is True
    assert payload["checks"]["requested_group_coverage"] is False


def test_glm52_reap_source_audit_cli_writes_evidence_before_exit_1(tmp_path: Path) -> None:
    profile = get_profile(GLM52_REAP_PROFILE_NAME)
    profile_path = Path(__file__).resolve().parents[1] / "models/glm52-reap-504b-v2.yaml"
    config_path = tmp_path / "config.json"
    index_path = tmp_path / "model.safetensors.index.json"
    config_path.write_text("{}")
    index_path.write_text(json.dumps({"metadata": {}, "weight_map": {}}))
    output_json = tmp_path / "source-audit.json"
    evidence_jsonl = tmp_path / "source-audit.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/inspect_glm52_reap_source.py",
            "--config-path",
            str(config_path),
            "--index-path",
            str(index_path),
            "--profile-path",
            str(profile_path),
            "--model-id",
            profile.hf_model_id,
            "--revision",
            profile.revision,
            "--output-json",
            str(output_json),
            "--append-jsonl",
            str(evidence_jsonl),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(output_json.read_text())
    assert payload["audit_status"] == "glm52_modelopt_nvfp4_source_incompatible"
    assert payload["source_checks_pass"] is False
    assert payload["audit_blockers"] == ["input_load_or_validation_error"]
    assert payload["input_error"]["type"] in {"KeyError", "ValueError"}
    assert json.loads(evidence_jsonl.read_text()) == payload


def test_glm52_reap_payload_audit_cli_writes_evidence_before_exit_2(tmp_path: Path) -> None:
    profile = get_profile(GLM52_REAP_PROFILE_NAME)
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--0xSero--glm-5.2-reap-504B-v2/snapshots"
        / profile.revision
    )
    config_path = snapshot / "config.json"
    index_path = snapshot / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        pytest.skip("pinned GLM52 REAP config/index are not cached")
    profile_path = Path(__file__).resolve().parents[1] / "models/glm52-reap-504b-v2.yaml"
    output_json = tmp_path / "payload-audit.json"
    evidence_jsonl = tmp_path / "payload-audit.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/audit_glm52_reap_source_payloads.py",
            "--source-dir",
            str(tmp_path),
            "--config-path",
            str(config_path),
            "--index-path",
            str(index_path),
            "--profile-path",
            str(profile_path),
            "--model-id",
            profile.hf_model_id,
            "--revision",
            profile.revision,
            "--max-groups",
            "1",
            "--output-json",
            str(output_json),
            "--append-jsonl",
            str(evidence_jsonl),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "glm52_modelopt_nvfp4_source_payloads_missing" in result.stderr
    payload = json.loads(output_json.read_text())
    assert payload["payload_status"] == "glm52_modelopt_nvfp4_source_payloads_missing"
    assert payload["missing_shards"] == ["model-00017.safetensors"]
    assert json.loads(evidence_jsonl.read_text()) == payload


def test_glm52_reap_materializer_writes_evidence_before_input_exit_1(
    tmp_path: Path,
) -> None:
    profile = get_profile(GLM52_REAP_PROFILE_NAME)
    profile_path = Path(__file__).resolve().parents[1] / "models/glm52-reap-504b-v2.yaml"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    config_path = source_dir / "config.json"
    index_path = source_dir / "model.safetensors.index.json"
    config_path.write_text("{}")
    index_path.write_text(json.dumps({"metadata": {}, "weight_map": {}}))
    output_json = tmp_path / "materialize.json"
    evidence_jsonl = tmp_path / "materialize.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/materialize_glm52_reap_groups.py",
            "--source-dir",
            str(source_dir),
            "--profile-path",
            str(profile_path),
            "--config-path",
            str(config_path),
            "--index-path",
            str(index_path),
            "--model-id",
            profile.hf_model_id,
            "--revision",
            profile.revision,
            "--groups",
            "10:gate_proj",
            "--output-dir",
            str(tmp_path / "artifact"),
            "--output-json",
            str(output_json),
            "--append-jsonl",
            str(evidence_jsonl),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(output_json.read_text())
    assert payload["materialization_status"] in {
        "glm52_modelopt_nvfp4_source_incompatible",
        "glm52_modelopt_nvfp4_materialization_input_failed",
    }
    assert payload["materialization_blocked"] is True
    assert json.loads(evidence_jsonl.read_text()) == payload


def test_glm52_reap_materializer_writes_evidence_before_payload_exit_2(
    tmp_path: Path,
) -> None:
    profile = get_profile(GLM52_REAP_PROFILE_NAME)
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--0xSero--glm-5.2-reap-504B-v2/snapshots"
        / profile.revision
    )
    config_path = snapshot / "config.json"
    index_path = snapshot / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        pytest.skip("pinned GLM52 REAP config/index are not cached")
    profile_path = Path(__file__).resolve().parents[1] / "models/glm52-reap-504b-v2.yaml"
    empty_source = tmp_path / "empty-source"
    empty_source.mkdir()
    output_json = tmp_path / "materialize.json"
    evidence_jsonl = tmp_path / "materialize.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/materialize_glm52_reap_groups.py",
            "--source-dir",
            str(empty_source),
            "--profile-path",
            str(profile_path),
            "--config-path",
            str(config_path),
            "--index-path",
            str(index_path),
            "--model-id",
            profile.hf_model_id,
            "--revision",
            profile.revision,
            "--groups",
            "10:gate_proj",
            "--output-dir",
            str(tmp_path / "artifact"),
            "--output-json",
            str(output_json),
            "--append-jsonl",
            str(evidence_jsonl),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    payload = json.loads(output_json.read_text())
    assert payload["materialization_status"] == (
        "glm52_modelopt_nvfp4_source_payloads_missing"
    )
    assert payload["materialization_blocked"] is True
    assert payload["payload_audit"]["requested_groups"] == ["10:gate_proj"]
    assert json.loads(evidence_jsonl.read_text()) == payload
