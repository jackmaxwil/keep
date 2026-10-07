from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open

from mlx_vq.convert.qwen_moe import (
    QWEN36_35B_A3B_MODEL_ID,
    QwenMoeConversionGroup,
    QwenMoeConversionPlan,
    audit_qwen_moe_source_index,
    convert_qwen_moe_group_from_safetensors,
    convert_qwen_moe_groups_from_safetensors,
    audit_qwen_moe_source_payloads,
    plan_qwen_moe_conversion_from_index,
    qwen_moe_group_output_filename,
)
from mlx_vq.quant.rtn import quantize_weight_rtn


def _qwen36_index() -> dict[str, object]:
    weight_map: dict[str, str] = {
        "model.language_model.embed_tokens.weight": "model-00001-of-00003.safetensors",
        "model.language_model.layers.0.mlp.experts.gate_up_proj": "model-00001-of-00003.safetensors",
        "model.language_model.layers.0.mlp.experts.down_proj": "model-00002-of-00003.safetensors",
        "model.language_model.layers.1.mlp.experts.gate_up_proj": "model-00002-of-00003.safetensors",
        "model.language_model.layers.1.mlp.experts.down_proj": "model-00003-of-00003.safetensors",
        "mtp.layers.0.mlp.experts.gate_up_proj": "model-00003-of-00003.safetensors",
        "mtp.layers.0.mlp.experts.down_proj": "model-00003-of-00003.safetensors",
    }
    return {"metadata": {"total_size": 123}, "weight_map": weight_map}


def test_qwen36_source_audit_captures_fused_language_expert_mapping() -> None:
    audit = audit_qwen_moe_source_index(
        _qwen36_index(),
        config={"model_type": "qwen3_5_moe"},
        model_id=QWEN36_35B_A3B_MODEL_ID,
        revision="fixture",
        expected_language_layers=2,
    )

    assert audit.model_id == QWEN36_35B_A3B_MODEL_ID
    assert audit.model_type == "qwen3_5_moe"
    assert audit.language_sparse_layers == 2
    assert audit.language_layer_range == (0, 1)
    assert audit.language_expert_tensors == 4
    assert audit.auxiliary_expert_tensors == 2
    assert audit.projections_per_language_layer == (("down_proj", "gate_up_proj"),)
    assert audit.has_fused_gate_up_proj is True
    assert audit.has_unfused_gate_or_up_proj is False
    assert audit.missing_language_projection_pairs == ()
    assert audit.checks["expected_language_layers"] is True
    assert audit.conversion_status == "source_mapping_audited__materializer_available"
    assert "qwen3_5_moe_runtime_adapter" in audit.conversion_blockers


def test_qwen36_source_audit_inventories_non_expert_source_tensors() -> None:
    index = _qwen36_index()
    weight_map = dict(index["weight_map"])
    weight_map.update(
        {
            "lm_head.weight": "model-00003-of-00003.safetensors",
            "model.language_model.layers.0.input_layernorm.weight": (
                "model-00001-of-00003.safetensors"
            ),
            "model.language_model.layers.0.linear_attn.q_proj.weight": (
                "model-00001-of-00003.safetensors"
            ),
            "model.language_model.layers.0.mlp.gate.weight": (
                "model-00001-of-00003.safetensors"
            ),
            "model.language_model.layers.1.self_attn.q_proj.weight": (
                "model-00002-of-00003.safetensors"
            ),
            "model.language_model.norm.weight": "model-00003-of-00003.safetensors",
            "model.visual.blocks.0.attn.proj.weight": (
                "model-00003-of-00003.safetensors"
            ),
            "mtp.layers.0.input_layernorm.weight": "model-00003-of-00003.safetensors",
        }
    )

    audit = audit_qwen_moe_source_index(
        {"weight_map": weight_map},
        config={"model_type": "qwen3_5_moe"},
        model_id=QWEN36_35B_A3B_MODEL_ID,
        revision="fixture",
        expected_language_layers=2,
    )

    payload = audit.to_json_dict()
    assert payload["non_expert_tensors_total"] == 9
    assert payload["non_expert_tensors_by_family"] == {
        "language_model": 6,
        "lm_head": 1,
        "mtp": 1,
        "visual": 1,
    }
    assert payload["language_non_expert_tensors_by_module"] == {
        "embed_tokens": 1,
        "input_layernorm": 1,
        "linear_attn": 1,
        "mlp": 1,
        "norm": 1,
        "self_attn": 1,
    }
    assert payload["language_non_expert_layers_by_module"] == {
        "input_layernorm": [0],
        "linear_attn": [0],
        "mlp": [0],
        "self_attn": [1],
    }


def test_qwen36_source_audit_reports_missing_projection_pairs() -> None:
    index = _qwen36_index()
    weight_map = dict(index["weight_map"])
    del weight_map["model.language_model.layers.1.mlp.experts.down_proj"]
    audit = audit_qwen_moe_source_index(
        {"weight_map": weight_map},
        config={"model_type": "qwen3_5_moe"},
        model_id=QWEN36_35B_A3B_MODEL_ID,
        revision="fixture",
        expected_language_layers=2,
    )

    assert audit.checks["complete_language_projection_pairs"] is False
    assert audit.missing_language_projection_pairs == ("1:down_proj",)


def test_qwen36_source_audit_json_round_trip(tmp_path: Path) -> None:
    audit = audit_qwen_moe_source_index(
        _qwen36_index(),
        config={"model_type": "qwen3_5_moe"},
        model_id=QWEN36_35B_A3B_MODEL_ID,
        revision="fixture",
        expected_language_layers=2,
    )
    output = tmp_path / "audit.json"
    output.write_text(json.dumps(audit.to_json_dict(), sort_keys=True))

    payload = json.loads(output.read_text())
    assert payload["architecture"]["total_parameters"] == 35_000_000_000
    assert payload["expert_tensor_shape"]["gate_up_proj"] == "fused"
    assert payload["checks"]["model_type"] is True


def test_qwen36_conversion_plan_maps_fused_gate_up_to_logical_projections() -> None:
    plan = plan_qwen_moe_conversion_from_index(
        _qwen36_index(),
        config={"model_type": "qwen3_5_moe"},
        model_id=QWEN36_35B_A3B_MODEL_ID,
        revision="fixture",
        expected_language_layers=2,
    )

    assert plan.source_projection_groups == 4
    assert plan.target_projection_groups == 6
    assert plan.missing_language_projection_pairs == ()
    assert plan.conversion_status == "qwen_moe_plan_ready__materializer_available"
    assert "qwen3_5_moe_runtime_adapter" in plan.conversion_blockers

    gate_up = next(
        group
        for group in plan.groups
        if group.layer == 0 and group.source_projection == "gate_up_proj"
    )
    assert gate_up.source_tensor == "model.language_model.layers.0.mlp.experts.gate_up_proj"
    assert gate_up.target_projections == ("gate_proj", "up_proj")
    assert gate_up.experts == 256
    assert gate_up.input_dims == 2048
    assert gate_up.output_dims == 1024
    assert gate_up.logical_output_dims == (512, 512)
    assert gate_up.code_bits == 8
    assert gate_up.group_size == 512
    assert gate_up.code_bytes == 256 * 1024 * (2048 // 8)

    down = next(
        group for group in plan.groups if group.layer == 0 and group.source_projection == "down_proj"
    )
    assert down.target_projections == ("down_proj",)
    assert down.input_dims == 512
    assert down.output_dims == 2048


def test_qwen36_conversion_plan_verifies_source_tensor_shapes() -> None:
    metadata = {
        "model.language_model.layers.0.mlp.experts.gate_up_proj": {
            "dtype": "BF16",
            "shape": [256, 1024, 2048],
            "parameter_count": 536_870_912,
        },
        "model.language_model.layers.0.mlp.experts.down_proj": {
            "dtype": "BF16",
            "shape": [256, 2048, 512],
            "parameter_count": 268_435_456,
        },
    }

    plan = plan_qwen_moe_conversion_from_index(
        _qwen36_index(),
        config={"model_type": "qwen3_5_moe"},
        model_id=QWEN36_35B_A3B_MODEL_ID,
        revision="fixture",
        expected_language_layers=1,
        tensor_metadata=metadata,
    )

    assert plan.source_shape_checks_present is True
    assert plan.source_shapes_verified is True
    assert plan.mismatched_source_shapes == ()
    assert plan.missing_source_shapes == ()
    gate_up = next(group for group in plan.groups if group.source_projection == "gate_up_proj")
    assert gate_up.expected_source_shape == (256, 1024, 2048)
    assert gate_up.source_shape == (256, 1024, 2048)
    assert gate_up.source_dtype == "BF16"
    assert gate_up.source_parameter_count == 536_870_912
    assert gate_up.source_shape_matches is True


def test_qwen36_conversion_plan_reports_shape_mismatch() -> None:
    metadata = {
        "model.language_model.layers.0.mlp.experts.gate_up_proj": {
            "dtype": "BF16",
            "shape": [256, 2048, 1024],
            "parameter_count": 536_870_912,
        },
        "model.language_model.layers.0.mlp.experts.down_proj": {
            "dtype": "BF16",
            "shape": [256, 2048, 512],
            "parameter_count": 268_435_456,
        },
    }

    plan = plan_qwen_moe_conversion_from_index(
        _qwen36_index(),
        config={"model_type": "qwen3_5_moe"},
        model_id=QWEN36_35B_A3B_MODEL_ID,
        revision="fixture",
        expected_language_layers=1,
        tensor_metadata=metadata,
    )

    assert plan.source_shapes_verified is False
    assert plan.mismatched_source_shapes == (
        "model.language_model.layers.0.mlp.experts.gate_up_proj: expected [256, 1024, 2048], found [256, 2048, 1024]",
    )


def test_qwen36_conversion_plan_is_written_in_source_audit_json() -> None:
    audit = audit_qwen_moe_source_index(
        _qwen36_index(),
        config={"model_type": "qwen3_5_moe"},
        model_id=QWEN36_35B_A3B_MODEL_ID,
        revision="fixture",
        expected_language_layers=2,
    )

    payload = audit.to_json_dict()
    assert payload["conversion_plan"]["source_projection_groups"] == 4
    assert payload["conversion_plan"]["target_projection_groups"] == 6
    assert payload["conversion_plan"]["groups"][0]["target_projections"] == [
        "gate_proj",
        "up_proj",
    ]


def test_qwen_moe_source_payload_audit_blocks_metadata_only_snapshot(
    tmp_path: Path,
) -> None:
    plan = plan_qwen_moe_conversion_from_index(
        _qwen36_index(),
        config={"model_type": "qwen3_5_moe"},
        model_id=QWEN36_35B_A3B_MODEL_ID,
        revision="fixture",
        expected_language_layers=1,
    )

    audit = audit_qwen_moe_source_payloads(
        source_dir=tmp_path,
        plan=plan,
        max_groups=2,
    )

    assert audit["record_type"] == "qwen_moe_source_payload_audit"
    assert audit["payload_status"] == "qwen_moe_source_payloads_missing"
    assert audit["planned_groups"] == 2
    assert audit["missing_shards"] == [
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
    ]
    assert audit["present_shards"] == []
    assert audit["materialization_blocked"] is True
    assert audit["materialization_blockers"] == [
        "download_payload_shards",
        "rerun_with_payload_bearing_source_dir",
    ]
    assert audit["groups"][0]["payload_ready"] is False


def test_qwen_moe_materializer_splits_fused_gate_up_projection(tmp_path: Path) -> None:
    source_tensor = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    shard_name = "model-00001-of-00001.safetensors"
    fused = (np.arange(2 * 16 * 8, dtype=np.float32).reshape(2, 16, 8) - 64.0) / 32.0
    mx.save_safetensors(str(tmp_path / shard_name), {source_tensor: mx.array(fused)})
    group = QwenMoeConversionGroup(
        layer=0,
        source_projection="gate_up_proj",
        source_tensor=source_tensor,
        source_shards=(shard_name,),
        target_projections=("gate_proj", "up_proj"),
        experts=2,
        input_dims=8,
        output_dims=16,
        logical_output_dims=(8, 8),
        expected_source_shape=(2, 16, 8),
        source_shape=(2, 16, 8),
        source_dtype="F32",
        source_parameter_count=256,
        source_shape_matches=True,
        code_bits=8,
        group_size=8,
    )
    output = tmp_path / "qwen-layer0-gate-up.safetensors"

    converted = convert_qwen_moe_group_from_safetensors(
        source_dir=tmp_path,
        group=group,
        output_path=output,
    )

    assert converted.output_path == output
    assert converted.source_tensors_read == 1
    assert converted.peak_source_tensor_bytes == fused.nbytes
    assert converted.target_projections == ("gate_proj", "up_proj")
    assert converted.codes_shapes == {
        "gate_proj": (2, 8, 1),
        "up_proj": (2, 8, 1),
    }
    assert converted.scales_shapes == {
        "gate_proj": (2, 8, 1),
        "up_proj": (2, 8, 1),
    }

    gate_prefix = "model.language_model.layers.0.mlp.switch_mlp.gate_proj"
    up_prefix = "model.language_model.layers.0.mlp.switch_mlp.up_proj"
    with safe_open(output, framework="np") as handle:
        metadata = handle.metadata()
        gate_codes = handle.get_tensor(f"{gate_prefix}.codes")
        gate_scales = handle.get_tensor(f"{gate_prefix}.scales")
        up_codes = handle.get_tensor(f"{up_prefix}.codes")
        up_scales = handle.get_tensor(f"{up_prefix}.scales")
        assert "quantization_config" in metadata

    assert gate_codes.shape == (2, 8, 1)
    assert gate_scales.shape == (2, 8, 1)
    assert up_codes.shape == (2, 8, 1)
    assert up_scales.shape == (2, 8, 1)
    expected_gate0 = quantize_weight_rtn(fused[0, :8, :], group_size=8)
    expected_up1 = quantize_weight_rtn(fused[1, 8:, :], group_size=8)
    np.testing.assert_array_equal(gate_codes[0], expected_gate0.codes)
    np.testing.assert_array_equal(up_codes[1], expected_up1.codes)


def test_qwen_moe_batch_materializer_writes_manifest_for_multiple_groups(
    tmp_path: Path,
) -> None:
    gate_up_tensor = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    down_tensor = "model.language_model.layers.0.mlp.experts.down_proj"
    shard_name = "model-00001-of-00001.safetensors"
    gate_up = (
        np.arange(2 * 16 * 8, dtype=np.float32).reshape(2, 16, 8) - 64.0
    ) / 32.0
    down = (np.arange(2 * 8 * 8, dtype=np.float32).reshape(2, 8, 8) - 16.0) / 16.0
    mx.save_safetensors(
        str(tmp_path / shard_name),
        {gate_up_tensor: mx.array(gate_up), down_tensor: mx.array(down)},
    )
    plan = QwenMoeConversionPlan(
        model_id=QWEN36_35B_A3B_MODEL_ID,
        revision="fixture",
        groups=(
            QwenMoeConversionGroup(
                layer=0,
                source_projection="gate_up_proj",
                source_tensor=gate_up_tensor,
                source_shards=(shard_name,),
                target_projections=("gate_proj", "up_proj"),
                experts=2,
                input_dims=8,
                output_dims=16,
                logical_output_dims=(8, 8),
                expected_source_shape=(2, 16, 8),
                source_shape=(2, 16, 8),
                source_dtype="F32",
                source_parameter_count=256,
                source_shape_matches=True,
                code_bits=8,
                group_size=8,
            ),
            QwenMoeConversionGroup(
                layer=0,
                source_projection="down_proj",
                source_tensor=down_tensor,
                source_shards=(shard_name,),
                target_projections=("down_proj",),
                experts=2,
                input_dims=8,
                output_dims=8,
                logical_output_dims=(8,),
                expected_source_shape=(2, 8, 8),
                source_shape=(2, 8, 8),
                source_dtype="F32",
                source_parameter_count=128,
                source_shape_matches=True,
                code_bits=8,
                group_size=8,
            ),
        ),
        missing_language_projection_pairs=(),
        source_shape_checks_present=True,
        conversion_status="qwen_moe_plan_ready__materializer_available",
        conversion_blockers=("qwen3_5_moe_runtime_adapter",),
    )
    output_dir = tmp_path / "qwen-artifact"

    manifest = convert_qwen_moe_groups_from_safetensors(
        source_dir=tmp_path,
        plan=plan,
        output_dir=output_dir,
        max_groups=2,
    )

    assert manifest.output_dir == output_dir
    assert manifest.manifest_path == output_dir / "qwen-moe-materialization-manifest.json"
    assert manifest.planned_groups == 2
    assert manifest.converted_groups == 2
    assert manifest.skipped_existing_groups == 0
    assert manifest.source_tensors_read == 2
    assert manifest.peak_source_tensor_bytes == gate_up.nbytes
    assert manifest.materialization_status == "qwen_moe_groups_materialized"
    manifest_payload = json.loads(manifest.manifest_path.read_text())
    assert [group["output_path"] for group in manifest_payload["groups"]] == [
        str(output_dir / qwen_moe_group_output_filename(plan.groups[0])),
        str(output_dir / qwen_moe_group_output_filename(plan.groups[1])),
    ]

    down_path = output_dir / qwen_moe_group_output_filename(plan.groups[1])
    with safe_open(down_path, framework="np") as handle:
        down_codes = handle.get_tensor(
            "model.language_model.layers.0.mlp.switch_mlp.down_proj.codes"
        )
    assert down_codes.shape == (2, 8, 1)


def test_qwen_moe_materializer_reports_missing_source_payloads(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "metadata-only-qwen"
    source_dir.mkdir()
    index_path = source_dir / "model.safetensors.index.json"
    config_path = source_dir / "config.json"
    output_dir = tmp_path / "artifact"
    output_json = tmp_path / "blocked.json"
    evidence_jsonl = tmp_path / "blocked.jsonl"
    index_path.write_text(json.dumps(_qwen36_index()))
    config_path.write_text(json.dumps({"model_type": "qwen3_5_moe"}))

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/materialize_qwen_moe_group.py",
            "--source-dir",
            str(source_dir),
            "--index-path",
            str(index_path),
            "--config-path",
            str(config_path),
            "--model-id",
            QWEN36_35B_A3B_MODEL_ID,
            "--revision",
            "fixture",
            "--expected-language-layers",
            "1",
            "--all-groups",
            "--max-groups",
            "2",
            "--output-dir",
            str(output_dir),
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
    assert "qwen_moe_source_payloads_missing" in result.stderr
    payload = json.loads(output_json.read_text())
    assert payload["payload_status"] == "qwen_moe_source_payloads_missing"
    assert payload["missing_shards"] == [
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
    ]
    assert payload["materialization_blockers"] == [
        "download_payload_shards",
        "rerun_with_payload_bearing_source_dir",
    ]
    assert json.loads(evidence_jsonl.read_text()) == payload


def test_qwen_moe_source_payload_audit_cli_reports_missing_source_payloads(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "metadata-only-qwen"
    source_dir.mkdir()
    index_path = source_dir / "model.safetensors.index.json"
    config_path = source_dir / "config.json"
    output_json = tmp_path / "payload-audit.json"
    evidence_jsonl = tmp_path / "payload-audit.jsonl"
    index_path.write_text(json.dumps(_qwen36_index()))
    config_path.write_text(json.dumps({"model_type": "qwen3_5_moe"}))

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/audit_qwen_moe_source_payloads.py",
            "--source-dir",
            str(source_dir),
            "--index-path",
            str(index_path),
            "--config-path",
            str(config_path),
            "--model-id",
            QWEN36_35B_A3B_MODEL_ID,
            "--revision",
            "fixture",
            "--expected-language-layers",
            "1",
            "--max-groups",
            "2",
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
    assert "qwen_moe_source_payloads_missing" in result.stderr
    payload = json.loads(output_json.read_text())
    assert payload["payload_status"] == "qwen_moe_source_payloads_missing"
    assert payload["missing_shards"] == [
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
    ]
    assert json.loads(evidence_jsonl.read_text()) == payload
