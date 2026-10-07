from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import yaml
from mlx.utils import tree_flatten

from mlx_vq.models.glm4_moe_adapter import QuantizedVQSwitchGLU
from mlx_vq.models.glm52_vq_adapter import GLM52VQModelArgs, Glm52VQDecoderLayer
from mlx_vq.models.profiles import ModelProfile, profile_to_dict


SCRIPT_PATH = Path(__file__).parents[1] / "benchmarks" / "probe_glm52_layer_forward.py"
MODEL_ID = "fixture/glm52-layer-forward"
REVISION = "fixture-revision"
SOURCE_INDEX_SHA256 = "d" * 64
SOURCE_BLOB_INVENTORY_SHA256 = "b" * 64
SOURCE_INVENTORY_SHA256 = "a" * 64


def _api():
    assert SCRIPT_PATH.is_file(), "the durable GLM52 layer-forward probe CLI is missing"
    spec = importlib.util.spec_from_file_location("probe_glm52_layer_forward_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _tiny_args() -> GLM52VQModelArgs:
    return GLM52VQModelArgs(
        model_type="glm_moe_dsa",
        vocab_size=32,
        hidden_size=16,
        index_head_dim=4,
        index_n_heads=2,
        index_topk=4,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=2,
        routed_scaling_factor=1.0,
        kv_lora_rank=4,
        q_lora_rank=8,
        qk_rope_head_dim=2,
        v_head_dim=4,
        qk_nope_head_dim=4,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=1,
        moe_layer_freq=1,
        first_k_dense_replace=1,
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        attention_bias=False,
        indexer_types=["full", "full", "shared"],
        index_topk_pattern=None,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        mlp_layer_types=["dense", "sparse", "sparse"],
    )


def _tiny_profile() -> ModelProfile:
    return ModelProfile(
        name="glm52-layer-forward-fixture",
        hf_model_id=MODEL_ID,
        revision=REVISION,
        architecture="glm_moe_dsa",
        num_layers=3,
        num_sparse_layers=2,
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


def _kv_b_from_split_weights(embed_q: mx.array, unembed_out: mx.array) -> mx.array:
    nope = embed_q.swapaxes(-1, -2)
    combined = mx.concatenate([nope, unembed_out], axis=1)
    return combined.reshape(-1, combined.shape[-1])


def _write_non_vq_artifact(path: Path, *, layer_idx: int) -> tuple[Path, str]:
    args = _tiny_args()
    layer = Glm52VQDecoderLayer(args, layer_idx)
    arrays: dict[str, mx.array] = {}
    layer_prefix = f"model.layers.{layer_idx}"
    params = dict(tree_flatten(layer.parameters()))
    embed_q = params.pop("self_attn.embed_q.weight")
    unembed_out = params.pop("self_attn.unembed_out.weight")
    params = {
        name: value
        for name, value in params.items()
        if not name.startswith("mlp.switch_mlp.")
    }
    for ordinal, (name, value) in enumerate(sorted(params.items()), start=1):
        arrays[f"{layer_prefix}.{name}"] = mx.full(
            value.shape,
            ordinal / 64.0,
            dtype=mx.bfloat16,
        )
    arrays[f"{layer_prefix}.self_attn.kv_b_proj.weight"] = _kv_b_from_split_weights(
        mx.full(embed_q.shape, 0.25, dtype=mx.bfloat16),
        mx.full(unembed_out.shape, 0.125, dtype=mx.bfloat16),
    )

    path.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(path / shard_name), arrays)
    index_path = path / "model.safetensors.index.json"
    _write_json(
        index_path,
        {
            "metadata": {},
            "weight_map": {name: shard_name for name in sorted(arrays)},
        },
    )
    return index_path, _sha256(index_path)


def _write_vq_artifact(path: Path, *, layer_idx: int) -> None:
    rng = np.random.default_rng(5202)
    path.mkdir(exist_ok=True)
    switch = QuantizedVQSwitchGLU.from_weights(
        gate_weight=mx.array(rng.normal(scale=0.04, size=(2, 8, 16)).astype(np.float32)),
        up_weight=mx.array(rng.normal(scale=0.04, size=(2, 8, 16)).astype(np.float32)),
        down_weight=mx.array(rng.normal(scale=0.04, size=(2, 16, 8)).astype(np.float32)),
        group_size=8,
    )
    prefix = f"model.layers.{layer_idx}.mlp.switch_mlp"
    for projection in ("gate_proj", "up_proj", "down_proj"):
        linear = getattr(switch, projection)
        mx.save_safetensors(
            str(path / f"layer-{layer_idx:05d}-{projection}.safetensors"),
            {
                f"{prefix}.{projection}.codes": linear.codes,
                f"{prefix}.{projection}.scales": linear.scales,
                "model.vq_codebook.e8": linear.codebook,
            },
        )


def _fixture(tmp_path: Path) -> dict[str, Path | str | int]:
    layer = 2
    profile = _tiny_profile()
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(yaml.safe_dump(profile_to_dict(profile), sort_keys=False))

    config_path = tmp_path / "config.json"
    _write_json(config_path, dict(_tiny_args().__dict__))
    config_sha256 = _sha256(config_path)

    non_vq_artifact = tmp_path / "non-vq"
    package_index_path, package_index_sha256 = _write_non_vq_artifact(
        non_vq_artifact,
        layer_idx=layer,
    )
    package_set_sha256 = hashlib.sha256(b"fixture-package-set").hexdigest()
    non_vq_manifest_path = non_vq_artifact / "non-vq-manifest.json"
    non_vq_manifest = {
        "schema_version": 1,
        "record_type": "glm52_non_vq_package_manifest",
        "pack_status": "glm52_non_vq_package_ready",
        "production_ready": True,
        "source_authority": "pinned_huggingface_lfs_v1",
        "profile": profile.name,
        "model_id": MODEL_ID,
        "source_revision": REVISION,
        "config_sha256": config_sha256,
        "index_sha256": SOURCE_INDEX_SHA256,
        "source_blob_inventory_sha256": SOURCE_BLOB_INVENTORY_SHA256,
        "source_inventory_sha256": SOURCE_INVENTORY_SHA256,
        "package_index_sha256": package_index_sha256,
        "package_set_sha256": package_set_sha256,
        "shards": [
            {
                "filename": "model-00001-of-00001.safetensors",
                "file_bytes": (
                    non_vq_artifact / "model-00001-of-00001.safetensors"
                ).stat().st_size,
                "file_sha256": _sha256(
                    non_vq_artifact / "model-00001-of-00001.safetensors"
                ),
            }
        ],
    }
    _write_json(non_vq_manifest_path, non_vq_manifest)
    non_vq_manifest_sha256 = _sha256(non_vq_manifest_path)
    non_vq_evidence_path = tmp_path / "non-vq-evidence.json"
    _write_json(
        non_vq_evidence_path,
        {
            **non_vq_manifest,
            "manifest_sha256": non_vq_manifest_sha256,
            "package_audit_pass": True,
            "package_audit": {
                "artifact_dir": str(non_vq_artifact),
                "manifest_path": str(non_vq_manifest_path),
                "manifest_sha256": non_vq_manifest_sha256,
                "package_set_sha256": package_set_sha256,
                "audit_pass": True,
                "checks": {
                    "lineage": True,
                    "strict_tree": True,
                    "index": True,
                    "tensor_inventory": True,
                    "physical_extents": True,
                    "payload_hashes": True,
                    "shard_hashes": True,
                    "accounting": True,
                },
            },
        },
    )

    routed_artifact = tmp_path / "routed"
    _write_vq_artifact(routed_artifact, layer_idx=layer)
    selected_group_keys = [
        f"{layer}:gate_proj",
        f"{layer}:up_proj",
        f"{layer}:down_proj",
    ]
    group_records = []
    for projection in ("gate_proj", "up_proj", "down_proj"):
        artifact_path = routed_artifact / f"layer-{layer:05d}-{projection}.safetensors"
        group_records.append(
            {
                "status": "ready",
                "layer": layer,
                "projection": projection,
                "artifact_path": str(artifact_path),
                "artifact_bytes": artifact_path.stat().st_size,
                "artifact_sha256": _sha256(artifact_path),
            }
        )
    routed_manifest_path = routed_artifact / "conversion-manifest.json"
    _write_json(
        routed_manifest_path,
        {
            "schema_version": 1,
            "record_type": "glm52_modelopt_nvfp4_materialization_manifest",
            "materialization_status": "glm52_modelopt_nvfp4_groups_ready",
            "materialization_scope": "bounded",
            "model_id": MODEL_ID,
            "source_revision": REVISION,
            "config_sha256": config_sha256,
            "index_sha256": SOURCE_INDEX_SHA256,
            "profile": profile.name,
            "source_decoder": "modelopt_nvfp4_v1",
            "source_weight_encoding": "modelopt_nvfp4",
            "code_bits": 8,
            "group_size": 8,
            "scale_estimator": "max_abs",
            "selected_group_keys": selected_group_keys,
            "groups": group_records,
            "dense_checkpoint_written": False,
        },
    )
    routed_manifest_sha256 = _sha256(routed_manifest_path)
    routed_audit_path = tmp_path / "routed-audit.json"
    _write_json(
        routed_audit_path,
        {
            "schema_version": 1,
            "record_type": "glm52_modelopt_nvfp4_artifact_audit",
            "audit_status": "glm52_modelopt_nvfp4_artifact_audit_ready",
            "audit_pass": True,
            "artifact_integrity_pass": True,
            "audit_blockers": [],
            "model_id": MODEL_ID,
            "source_revision": REVISION,
            "config_sha256": config_sha256,
            "index_sha256": SOURCE_INDEX_SHA256,
            "artifact_dir": str(routed_artifact),
            "manifest_path": str(routed_manifest_path),
            "manifest_sha256": routed_manifest_sha256,
            "group_set_sha256": hashlib.sha256(b"fixture-group-set").hexdigest(),
            "dense_routed_experts": False,
            "full_routed_artifact_ready": False,
            "materialization_scope": "bounded",
            "ready_group_keys": selected_group_keys,
            "complete_layer_ids": [layer],
            "requested_code_bits": 8,
            "requested_group_size": 8,
            "requested_scale_estimator": "max_abs",
            "source_decoder": "modelopt_nvfp4_v1",
            "source_weight_encoding": "modelopt_nvfp4",
            "checks": {
                "exact_manifest_contract": True,
                "exact_group_selection": True,
                "exact_artifact_files": True,
                "exact_tensor_contract": True,
                "pinned_lineage": True,
                "no_layer_78": True,
                "no_dense_routed_experts": True,
                "artifact_hashes_and_sizes": True,
                "bounded_whole_model_claims_suppressed": True,
            },
        },
    )
    return {
        "layer": layer,
        "profile_path": profile_path,
        "config_path": config_path,
        "non_vq_artifact": non_vq_artifact,
        "non_vq_evidence_path": non_vq_evidence_path,
        "routed_artifact": routed_artifact,
        "routed_audit_path": routed_audit_path,
    }


def _probe_kwargs(fixture: dict[str, Path | str | int]) -> dict[str, object]:
    return {
        "non_vq_artifact": fixture["non_vq_artifact"],
        "non_vq_evidence_json": fixture["non_vq_evidence_path"],
        "routed_artifact": fixture["routed_artifact"],
        "routed_audit_json": fixture["routed_audit_path"],
        "profile_path": fixture["profile_path"],
        "config_path": fixture["config_path"],
        "model_id": MODEL_ID,
        "revision": REVISION,
        "layer": fixture["layer"],
        "tokens": 1,
        "seed": 5202,
        "input_scale": 0.02,
    }


def test_probe_runs_real_bound_bf16_layer_and_keeps_scope_bounded(tmp_path: Path) -> None:
    api = _api()
    fixture = _fixture(tmp_path)

    report = api.probe_glm52_layer_forward(**_probe_kwargs(fixture))

    assert report["record_type"] == "glm52_layer_forward_probe"
    assert report["probe_status"] == "glm52_layer_forward_probe_ready"
    assert report["probe_pass"] is True
    assert report["non_vq_source_authority"] == "pinned_huggingface_lfs_v1"
    assert report["production_non_vq_evidence_validated"] is True
    assert report["routed_artifact_audit_validated"] is True
    assert report["routed_source_decoder"] == "modelopt_nvfp4_v1"
    assert report["routed_source_weight_encoding"] == "modelopt_nvfp4"
    assert report["output_shape"] == [1, 1, 16]
    assert report["output_dtype"] == "mlx.core.bfloat16"
    assert report["output_finite"] is True
    assert report["dense_routed_parameter_present"] is False
    assert report["non_vq_bind_report"]["missing_model_parameters"] == []
    assert report["non_vq_bind_report"]["skipped_unmatched_tensors"] == []
    assert report["vq_projection_metadata"]["gate_proj"]["code_bits"] == 8
    assert report["vq_projection_metadata"]["gate_proj"]["group_size"] == 8
    assert report["indexer_type"] == "shared"
    assert report["indexshare_sparse_selection_exercised"] is False
    assert report["whole_model_runtime_proven"] is False
    assert report["generation_proven"] is False
    assert report["long_context_indexshare_proven"] is False
    assert report["full_routed_artifact_ready"] is False
    assert report["mlx_peak_bytes"] is not None
    assert report["rss_bytes"] > 0


def test_probe_rejects_non_production_non_vq_evidence(tmp_path: Path) -> None:
    api = _api()
    fixture = _fixture(tmp_path)
    evidence_path = fixture["non_vq_evidence_path"]
    assert isinstance(evidence_path, Path)
    evidence = json.loads(evidence_path.read_text())
    evidence["production_ready"] = False
    _write_json(evidence_path, evidence)

    with pytest.raises(ValueError, match="production_ready"):
        api.probe_glm52_layer_forward(**_probe_kwargs(fixture))


def test_probe_rejects_production_evidence_without_pinned_blob_identity(
    tmp_path: Path,
) -> None:
    api = _api()
    fixture = _fixture(tmp_path)
    evidence_path = fixture["non_vq_evidence_path"]
    assert isinstance(evidence_path, Path)
    evidence = json.loads(evidence_path.read_text())
    evidence.pop("source_blob_inventory_sha256")
    _write_json(evidence_path, evidence)

    with pytest.raises(ValueError, match="source_blob_inventory_sha256"):
        api.probe_glm52_layer_forward(**_probe_kwargs(fixture))


def test_probe_rejects_tampered_selected_non_vq_shard(tmp_path: Path) -> None:
    api = _api()
    fixture = _fixture(tmp_path)
    artifact = fixture["non_vq_artifact"]
    assert isinstance(artifact, Path)
    shard = artifact / "model-00001-of-00001.safetensors"
    payload = bytearray(shard.read_bytes())
    payload[-1] ^= 1
    shard.write_bytes(payload)

    with pytest.raises(ValueError, match="non-VQ selected shard.*SHA-256"):
        api.probe_glm52_layer_forward(**_probe_kwargs(fixture))


def test_memory_clean_diagnostic_is_unavailable_without_vm_stat_deltas() -> None:
    api = _api()

    assert api._memory_clean_diagnostic(
        {"pageouts_delta": None, "swapouts_delta": None}
    ) == (False, None)
    assert api._memory_clean_diagnostic(
        {"pageouts_delta": 0, "swapouts_delta": 0}
    ) == (True, True)
    assert api._memory_clean_diagnostic(
        {"pageouts_delta": 1, "swapouts_delta": 0}
    ) == (True, False)


def test_probe_rejects_bounded_routed_audit_relabelled_as_full(tmp_path: Path) -> None:
    api = _api()
    fixture = _fixture(tmp_path)
    audit_path = fixture["routed_audit_path"]
    assert isinstance(audit_path, Path)
    audit = json.loads(audit_path.read_text())
    audit["materialization_scope"] = "full"
    audit["full_routed_artifact_ready"] = True
    _write_json(audit_path, audit)

    with pytest.raises(ValueError, match="full_routed_artifact_ready|materialization_scope"):
        api.probe_glm52_layer_forward(**_probe_kwargs(fixture))


def test_layer_local_probe_rejects_full_evidence_with_missing_non_probed_group(
    tmp_path: Path,
) -> None:
    api = _api()
    fixture = _fixture(tmp_path)
    routed_artifact = fixture["routed_artifact"]
    routed_audit_path = fixture["routed_audit_path"]
    assert isinstance(routed_artifact, Path)
    assert isinstance(routed_audit_path, Path)
    _write_vq_artifact(routed_artifact, layer_idx=1)

    manifest_path = routed_artifact / "conversion-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    extra_keys = ["1:gate_proj", "1:up_proj", "1:down_proj"]
    extra_records = []
    for projection in ("gate_proj", "up_proj", "down_proj"):
        path = routed_artifact / f"layer-00001-{projection}.safetensors"
        extra_records.append(
            {
                "status": "ready",
                "layer": 1,
                "projection": projection,
                "artifact_path": str(path),
                "artifact_bytes": path.stat().st_size,
                "artifact_sha256": _sha256(path),
            }
        )
    manifest["materialization_scope"] = "full"
    manifest["selected_group_keys"] = extra_keys + manifest["selected_group_keys"]
    manifest["groups"] = extra_records + manifest["groups"]
    _write_json(manifest_path, manifest)

    audit = json.loads(routed_audit_path.read_text())
    audit["materialization_scope"] = "full"
    audit["full_routed_artifact_ready"] = True
    audit["ready_group_keys"] = extra_keys + audit["ready_group_keys"]
    audit["complete_layer_ids"] = [1, 2]
    audit["manifest_sha256"] = _sha256(manifest_path)
    _write_json(routed_audit_path, audit)

    (routed_artifact / "layer-00001-gate_proj.safetensors").unlink()

    with pytest.raises(ValueError, match="layer-local.*bounded|full.*unsupported"):
        api.probe_glm52_layer_forward(**_probe_kwargs(fixture))


def test_cli_writes_durable_failure_json_for_routed_artifact_identity_mismatch(
    tmp_path: Path,
) -> None:
    api = _api()
    fixture = _fixture(tmp_path)
    audit_path = fixture["routed_audit_path"]
    assert isinstance(audit_path, Path)
    audit = json.loads(audit_path.read_text())
    audit["artifact_dir"] = str(tmp_path / "wrong-routed-artifact")
    _write_json(audit_path, audit)
    output_path = tmp_path / "probe.json"

    rc = api.main(
        [
            "--non-vq-artifact",
            str(fixture["non_vq_artifact"]),
            "--non-vq-evidence-json",
            str(fixture["non_vq_evidence_path"]),
            "--routed-artifact",
            str(fixture["routed_artifact"]),
            "--routed-audit-json",
            str(fixture["routed_audit_path"]),
            "--profile-path",
            str(fixture["profile_path"]),
            "--config-path",
            str(fixture["config_path"]),
            "--model-id",
            MODEL_ID,
            "--revision",
            REVISION,
            "--layer",
            str(fixture["layer"]),
            "--tokens",
            "1",
            "--seed",
            "5202",
            "--input-scale",
            "0.02",
            "--output-json",
            str(output_path),
        ]
    )

    failure = json.loads(output_path.read_text())
    assert rc == 1
    assert failure["record_type"] == "glm52_layer_forward_probe"
    assert failure["probe_status"] == "glm52_layer_forward_probe_failed"
    assert failure["probe_pass"] is False
    assert failure["input_error"]["type"] == "ValueError"
    assert "routed artifact path" in failure["input_error"]["message"]
    assert failure["whole_model_runtime_proven"] is False
    assert failure["generation_proven"] is False
    assert failure["long_context_indexshare_proven"] is False
