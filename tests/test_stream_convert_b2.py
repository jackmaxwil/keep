from __future__ import annotations

import json
import struct
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from safetensors import safe_open

import mlx_vq.convert.stream_convert as stream_convert
from mlx_vq.codebook.e8 import E8P_PACKED_ABS_SHA256
from mlx_vq.convert.stream_convert import (
    TensorAction,
    _read_tensor_from_shard,
    convert_vq_group_from_safetensors,
    convert_vq_groups_from_safetensors,
    effective_group_size,
    load_safetensors_index,
    parse_expert_projection,
    plan_from_config,
    plan_streaming_conversion_from_index,
    source_shard_inventory,
)
from mlx_vq.io.load import load_quantized_vq_switch_linear
from mlx_vq.quant.rotation import apply_rotation_np
from mlx_vq.quant.rht import apply_rht_np, deterministic_rht_signs
from mlx_vq.quant.rtn import quantize_weight_rtn

import mlx.core as mx
import numpy as np


GLM52_CONFIG = {
    "model_type": "glm_moe_dsa",
    "num_hidden_layers": 78,
    "first_k_dense_replace": 3,
    "mlp_layer_types": ["dense", "dense", "dense"] + ["sparse"] * 75,
    "indexer_types": ["full", "full", "full"]
    + [
        "full" if layer in {6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74} else "shared"
        for layer in range(3, 78)
    ],
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "hidden_size": 6144,
    "moe_intermediate_size": 2048,
    "intermediate_size": 12288,
    "max_position_embeddings": 1048576,
}


GLM45_AIR_CONFIG = {
    "model_type": "glm4_moe",
    "num_hidden_layers": 46,
    "first_k_dense_replace": 1,
    "n_routed_experts": 128,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "hidden_size": 4096,
    "moe_intermediate_size": 1408,
    "intermediate_size": 10944,
    "max_position_embeddings": 131072,
}


def _index_for_config(config: dict, *, shard_count: int = 4) -> dict:
    weight_map = {
        "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
        "lm_head.weight": "model-00004-of-00004.safetensors",
        "model.layers.0.self_attn.indexer.wk.weight": "model-00001-of-00004.safetensors",
        "model.layers.0.mlp.shared_experts.up_proj.weight": "model-00001-of-00004.safetensors",
    }
    dense_layers = config.get("first_k_dense_replace", 0)
    sparse_layers = range(dense_layers, config["num_hidden_layers"])
    for layer in sparse_layers:
        for expert in range(config["n_routed_experts"]):
            shard = f"model-{1 + (layer + expert) % shard_count:05d}-of-{shard_count:05d}.safetensors"
            for proj in ("gate_proj", "up_proj", "down_proj"):
                weight_map[f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"] = shard
    return {"metadata": {"total_size": 123}, "weight_map": weight_map}


def test_load_safetensors_index_and_parse_expert_projection(tmp_path) -> None:
    path = tmp_path / "model.safetensors.index.json"
    path.write_text(json.dumps(_index_for_config(GLM45_AIR_CONFIG)))

    index = load_safetensors_index(path)
    parsed = parse_expert_projection("model.layers.7.mlp.experts.12.down_proj.weight")

    assert len(index.shards) == 4
    assert parsed is not None
    assert (parsed.layer, parsed.expert, parsed.projection) == (7, 12, "down_proj")
    assert parse_expert_projection("model.layers.7.mlp.shared_experts.down_proj.weight") is None


def test_glm52_metadata_dry_run_groups_all_routed_experts_without_weights(tmp_path) -> None:
    path = tmp_path / "glm52.index.json"
    path.write_text(json.dumps(_index_for_config(GLM52_CONFIG, shard_count=8)))
    index = load_safetensors_index(path)

    dry_run = plan_streaming_conversion_from_index(GLM52_CONFIG, index, model_id="zai-org/GLM-5.2")
    budget = plan_from_config(GLM52_CONFIG, model_id="zai-org/GLM-5.2")

    assert dry_run.missing_vq_groups == ()
    assert len(dry_run.vq_groups) == 75 * 3
    assert dry_run.vq_code_bytes == budget.vq1_storage.code_bytes
    assert dry_run.vq_scale_bytes == budget.vq1_storage.scale_bytes
    assert dry_run.peak_source_projection_bytes == 6144 * 2048 * 2
    assert any(t.action == TensorAction.COPY for t in dry_run.tensors)
    assert sum(t.action == TensorAction.VQ for t in dry_run.tensors) == 75 * 256 * 3


def test_glm45_air_metadata_plan_matches_phase_a_vq_budget(tmp_path) -> None:
    path = tmp_path / "glm45.index.json"
    path.write_text(json.dumps(_index_for_config(GLM45_AIR_CONFIG, shard_count=4)))
    index = load_safetensors_index(path)

    dry_run = plan_streaming_conversion_from_index(GLM45_AIR_CONFIG, index, model_id="zai-org/GLM-4.5-Air")
    budget = plan_from_config(GLM45_AIR_CONFIG, model_id="zai-org/GLM-4.5-Air")

    assert dry_run.missing_vq_groups == ()
    assert len(dry_run.vq_groups) == 45 * 3
    assert dry_run.vq_code_bytes == budget.vq1_storage.code_bytes
    assert dry_run.vq_scale_bytes == budget.vq1_storage.scale_bytes
    down_group = next(group for group in dry_run.vq_groups if group.projection == "down_proj")
    assert down_group.group_size == 352


def test_dense_streaming_plan_records_dense_source_without_a_decoder(tmp_path) -> None:
    path = tmp_path / "glm45.index.json"
    path.write_text(json.dumps(_index_for_config(GLM45_AIR_CONFIG, shard_count=4)))

    plan = plan_streaming_conversion_from_index(
        GLM45_AIR_CONFIG,
        load_safetensors_index(path),
        model_id="zai-org/GLM-4.5-Air",
    )

    assert plan.source_weight_encoding.value == "dense"
    assert plan.source_decoder is None
    assert {group.source_weight_encoding.value for group in plan.vq_groups} == {"dense"}


def test_streaming_plan_rejects_present_unsupported_quantization_config(tmp_path) -> None:
    config = {**GLM45_AIR_CONFIG, "quantization_config": {"quant_method": "gptq"}}
    path = tmp_path / "glm45.index.json"
    path.write_text(json.dumps(_index_for_config(config, shard_count=4)))

    with pytest.raises(ValueError, match="unsupported quantization_config"):
        plan_streaming_conversion_from_index(
            config,
            load_safetensors_index(path),
            model_id="unsupported-quantized-glm",
        )


def test_streaming_plan_accepts_mixed_code_bits_policy_without_changing_default(tmp_path) -> None:
    path = tmp_path / "glm45.index.json"
    path.write_text(json.dumps(_index_for_config(GLM45_AIR_CONFIG, shard_count=4)))
    index = load_safetensors_index(path)

    default_plan = plan_streaming_conversion_from_index(
        GLM45_AIR_CONFIG,
        index,
        model_id="zai-org/GLM-4.5-Air",
    )
    mixed_plan = plan_streaming_conversion_from_index(
        GLM45_AIR_CONFIG,
        index,
        model_id="zai-org/GLM-4.5-Air",
        code_bits_policy={"1:gate_proj": 16, "*:down_proj": 16},
    )

    default_gate = next(group for group in default_plan.vq_groups if group.layer == 1 and group.projection == "gate_proj")
    mixed_gate = next(group for group in mixed_plan.vq_groups if group.layer == 1 and group.projection == "gate_proj")
    mixed_up = next(group for group in mixed_plan.vq_groups if group.layer == 1 and group.projection == "up_proj")
    mixed_down = next(group for group in mixed_plan.vq_groups if group.layer == 2 and group.projection == "down_proj")

    assert default_gate.code_bits == 8
    assert mixed_gate.code_bits == 16
    assert mixed_up.code_bits == 8
    assert mixed_down.code_bits == 16
    assert mixed_plan.code_bits_policy == (("*:down_proj", 16), ("1:gate_proj", 16))
    assert mixed_plan.vq_code_bytes > default_plan.vq_code_bytes


def test_effective_group_size_uses_largest_8d_aligned_divisor() -> None:
    assert effective_group_size(4096, 512) == 512
    assert effective_group_size(6144, 512) == 512
    assert effective_group_size(2048, 512) == 512
    assert effective_group_size(1408, 512) == 352


def test_streaming_plan_reports_incomplete_expert_groups(tmp_path) -> None:
    payload = _index_for_config(GLM45_AIR_CONFIG, shard_count=2)
    payload["weight_map"].pop("model.layers.1.mlp.experts.0.gate_proj.weight")
    path = tmp_path / "broken.index.json"
    path.write_text(json.dumps(payload))

    dry_run = plan_streaming_conversion_from_index(
        GLM45_AIR_CONFIG,
        load_safetensors_index(path),
        model_id="zai-org/GLM-4.5-Air",
    )

    assert dry_run.missing_vq_groups == ("layer 1 gate_proj: missing 1 experts",)


def test_modelopt_nvfp4_plan_resolves_cross_shard_companions_for_preflight(tmp_path) -> None:
    config_path, index_path, _ = _write_tiny_modelopt_nvfp4_checkpoint(
        tmp_path,
        projections=("gate_proj", "up_proj", "down_proj"),
    )
    config = json.loads(config_path.read_text())
    index = load_safetensors_index(index_path)

    plan = plan_streaming_conversion_from_index(config, index, model_id="tiny-modelopt-nvfp4", group_size=8)
    group = next(group for group in plan.vq_groups if group.projection == "gate_proj")
    inventory = source_shard_inventory(tmp_path, (group,))

    assert plan.source_weight_encoding.value == "modelopt_nvfp4"
    assert plan.source_decoder == "modelopt_nvfp4_v1"
    assert group.source_weight_encoding.value == "modelopt_nvfp4"
    assert group.source_shards == (
        "block-scales.safetensors",
        "global-scales.safetensors",
        "packed-weights.safetensors",
    )
    assert inventory.required_shards == 3
    assert inventory.present_shards == 3
    assert plan.peak_source_projection_bytes == 16 * 16 * 4


def test_modelopt_nvfp4_plan_reports_the_missing_companion_tensor(tmp_path) -> None:
    config_path, index_path, _ = _write_tiny_modelopt_nvfp4_checkpoint(tmp_path)
    index_payload = json.loads(index_path.read_text())
    missing_name = "model.layers.1.mlp.experts.0.gate_proj.weight_scale_2"
    index_payload["weight_map"].pop(missing_name)
    index_path.write_text(json.dumps(index_payload))

    plan = plan_streaming_conversion_from_index(
        json.loads(config_path.read_text()),
        load_safetensors_index(index_path),
        model_id="tiny-modelopt-nvfp4",
        group_size=8,
    )

    assert not any(group.projection == "gate_proj" for group in plan.vq_groups)
    assert len(plan.missing_vq_groups) == 3
    assert "missing ModelOpt NVFP4 companion" in plan.missing_vq_groups[0]
    assert missing_name in plan.missing_vq_groups[0]


def test_modelopt_nvfp4_plan_excludes_reap_layer_78_mtp_weights(tmp_path) -> None:
    config = _tiny_modelopt_nvfp4_config(num_hidden_layers=78, dense_layers=77)
    weight_map = {}
    for layer in (77, 78):
        for expert in range(config["n_routed_experts"]):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
                weight_map[name] = f"layer-{layer}-weights.safetensors"
                weight_map[f"{name}_scale"] = f"layer-{layer}-block-scales.safetensors"
                weight_map[f"{name}_scale_2"] = f"layer-{layer}-global-scales.safetensors"
    index_path = _write_index(tmp_path / "index.json", {"metadata": {}, "weight_map": weight_map})

    plan = plan_streaming_conversion_from_index(
        config,
        load_safetensors_index(index_path),
        model_id="tiny-reap-modelopt-nvfp4",
        group_size=8,
    )

    assert {group.layer for group in plan.vq_groups} == {77}
    assert len(plan.vq_groups) == 3


@pytest.mark.parametrize("code_bits", [8, 16])
def test_modelopt_nvfp4_conversion_matches_hard_coded_dense_oracle_serial_and_parallel(
    tmp_path,
    code_bits,
) -> None:
    config_path, index_path, dense_oracle = _write_tiny_modelopt_nvfp4_checkpoint(tmp_path)
    config = json.loads(config_path.read_text())
    index = load_safetensors_index(index_path)
    plan = plan_streaming_conversion_from_index(
        config,
        index,
        model_id="tiny-modelopt-nvfp4",
        group_size=8,
        code_bits=code_bits,
    )
    group = next(group for group in plan.vq_groups if group.projection == "gate_proj")

    serial = convert_vq_group_from_safetensors(
        source_dir=tmp_path,
        index=index,
        group=group,
        output_path=tmp_path / f"serial-{code_bits}.safetensors",
    )
    parallel = convert_vq_group_from_safetensors(
        source_dir=tmp_path,
        index=index,
        group=group,
        output_path=tmp_path / f"parallel-{code_bits}.safetensors",
        expert_workers=2,
    )
    expected = [
        quantize_weight_rtn(dense_oracle[expert], group_size=8, code_bits=code_bits)
        for expert in group.experts
    ]
    expected_codes = np.stack([item.codes for item in expected])
    expected_scales = np.stack([item.scales for item in expected])

    assert serial.source_tensors_read == len(group.experts)
    assert serial.peak_source_tensor_bytes == 16 * 16 * 4
    assert serial.source_weight_encoding.value == "modelopt_nvfp4"
    assert serial.source_decoder == "modelopt_nvfp4_v1"
    assert parallel.expert_workers == 2
    with safe_open(serial.output_path, framework="np") as handle:
        metadata = handle.metadata()
        serial_codes = handle.get_tensor(group.codes_name)
        serial_scales = handle.get_tensor(group.scales_name)
    with safe_open(parallel.output_path, framework="np") as handle:
        parallel_codes = handle.get_tensor(group.codes_name)
        parallel_scales = handle.get_tensor(group.scales_name)
    policy = json.loads(metadata["quantization_config"])["policy"]
    assert policy["source_weight_encoding"] == "modelopt_nvfp4"
    assert policy["source_decoder"] == "modelopt_nvfp4_v1"
    np.testing.assert_array_equal(serial_codes, expected_codes)
    np.testing.assert_array_equal(serial_scales, expected_scales)
    np.testing.assert_array_equal(parallel_codes, expected_codes)
    np.testing.assert_array_equal(parallel_scales, expected_scales)
    assert serial.output_path.read_bytes() == parallel.output_path.read_bytes()


def test_convert_vq_group_streams_raw_expert_tensors_to_stacked_safetensors(tmp_path) -> None:
    config = {
        "model_type": "glm4_moe",
        "num_hidden_layers": 2,
        "first_k_dense_replace": 1,
        "n_routed_experts": 3,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "hidden_size": 16,
        "moe_intermediate_size": 8,
        "intermediate_size": 32,
        "max_position_embeddings": 128,
    }
    rng = np.random.default_rng(2026)
    weight_map = {}
    arrays = {}
    for expert in range(3):
        name = f"model.layers.1.mlp.experts.{expert}.gate_proj.weight"
        weight = rng.normal(scale=0.04, size=(8, 16)).astype(np.float32)
        arrays[name] = mx.array(weight)
        weight_map[name] = "model-00001-of-00001.safetensors"
    shard = tmp_path / "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(shard), arrays)
    index_path = _write_index(tmp_path / "index.json", {"metadata": {}, "weight_map": weight_map})
    index = load_safetensors_index(index_path)
    plan = plan_streaming_conversion_from_index(config, index, model_id="tiny-glm4", group_size=8)
    group = next(group for group in plan.vq_groups if group.projection == "gate_proj")
    output = tmp_path / "converted-gate.safetensors"

    converted = convert_vq_group_from_safetensors(
        source_dir=tmp_path,
        index=index,
        group=group,
        output_path=output,
    )
    parallel_output = tmp_path / "converted-gate-parallel.safetensors"
    parallel = convert_vq_group_from_safetensors(
        source_dir=tmp_path,
        index=index,
        group=group,
        output_path=parallel_output,
        expert_workers=2,
    )

    assert converted.source_tensors_read == 3
    assert converted.peak_source_tensor_bytes == 8 * 16 * 4
    assert converted.codes_shape == (3, 8, 2)
    assert converted.scales_shape == (3, 8, 2)
    assert converted.scale_estimator == "max_abs"
    assert converted.elapsed_seconds is not None
    assert converted.elapsed_seconds >= 0
    assert converted.expert_workers == 1
    assert parallel.expert_workers == 2

    layer = load_quantized_vq_switch_linear(output, "model.layers.1.mlp.switch_mlp.gate_proj")
    parallel_layer = load_quantized_vq_switch_linear(parallel_output, "model.layers.1.mlp.switch_mlp.gate_proj")
    assert layer.codes.shape == (3, 8, 2)
    np.testing.assert_array_equal(np.array(parallel_layer.codes), np.array(layer.codes))
    np.testing.assert_array_equal(np.array(parallel_layer.scales), np.array(layer.scales))
    expected = quantize_weight_rtn(np.array(arrays["model.layers.1.mlp.experts.0.gate_proj.weight"]), group_size=8)
    np.testing.assert_array_equal(np.array(layer.codes[0]), expected.codes)


def test_convert_vq_group_records_explicit_scale_estimator_variant(tmp_path) -> None:
    config_path, index_path = _write_complete_tiny_sparse_checkpoint(tmp_path)
    config = json.loads(config_path.read_text())
    index = load_safetensors_index(index_path)
    plan = plan_streaming_conversion_from_index(config, index, model_id="tiny-glm4", group_size=8)
    group = next(group for group in plan.vq_groups if group.projection == "gate_proj")
    output = tmp_path / "converted-gate-percentile.safetensors"

    converted = convert_vq_group_from_safetensors(
        source_dir=tmp_path,
        index=index,
        group=group,
        output_path=output,
        scale_estimator="percentile_99",
    )

    assert converted.scale_estimator == "percentile_99"
    with safe_open(output, framework="np") as handle:
        metadata = handle.metadata()
    quantization_config = json.loads(metadata["quantization_config"])
    assert quantization_config["policy"]["scale_estimator"] == "percentile_99"


def test_convert_vq_group_with_rht_seed_writes_signs_and_quantizes_transformed_weights(tmp_path) -> None:
    config_path, index_path = _write_complete_tiny_sparse_checkpoint(tmp_path)
    config = json.loads(config_path.read_text())
    index = load_safetensors_index(index_path)
    plan = plan_streaming_conversion_from_index(config, index, model_id="tiny-glm4", group_size=8)
    group = next(group for group in plan.vq_groups if group.projection == "gate_proj")
    output = tmp_path / "converted-gate-rht.safetensors"
    rht_seed = "tiny-gate-rht-v1"

    converted = convert_vq_group_from_safetensors(
        source_dir=tmp_path,
        index=index,
        group=group,
        output_path=output,
        rht_seed=rht_seed,
    )

    assert converted.rht_seed == rht_seed
    signs = deterministic_rht_signs(group.input_dims, seed=rht_seed)
    with safe_open(output, framework="np") as handle:
        metadata = handle.metadata()
        np.testing.assert_array_equal(handle.get_tensor(group.rht_signs_name), signs)
    quantization_config = json.loads(metadata["quantization_config"])
    assert quantization_config["policy"]["rht"] == "hadamard_signs_v1"
    assert quantization_config["policy"]["rht_seed"] == rht_seed

    layer = load_quantized_vq_switch_linear(output, "model.layers.1.mlp.switch_mlp.gate_proj")
    assert layer.get("rht_signs") is not None
    np.testing.assert_array_equal(np.array(layer.rht_signs), signs.astype(np.float32))

    source_weight = _read_tensor_from_shard(
        tmp_path,
        "model-00001-of-00001.safetensors",
        "model.layers.1.mlp.experts.0.gate_proj.weight",
    )
    expected = quantize_weight_rtn(apply_rht_np(source_weight, signs), group_size=8)
    np.testing.assert_array_equal(np.array(layer.codes[0]), expected.codes)


def test_convert_vq_group_with_learned_rht_signs_writes_signs_and_metadata(tmp_path) -> None:
    config_path, index_path = _write_complete_tiny_sparse_checkpoint(tmp_path)
    config = json.loads(config_path.read_text())
    index = load_safetensors_index(index_path)
    plan = plan_streaming_conversion_from_index(config, index, model_id="tiny-glm4", group_size=8)
    group = next(group for group in plan.vq_groups if group.projection == "gate_proj")
    output = tmp_path / "converted-gate-learned-rht.safetensors"
    signs = np.where(np.arange(group.input_dims) % 2 == 0, 1, -1).astype(np.int8)

    converted = convert_vq_group_from_safetensors(
        source_dir=tmp_path,
        index=index,
        group=group,
        output_path=output,
        rht_signs=signs,
        rht_id="tiny-learned-rht-v1",
    )

    assert converted.rht_id == "tiny-learned-rht-v1"
    with safe_open(output, framework="np") as handle:
        metadata = handle.metadata()
        np.testing.assert_array_equal(handle.get_tensor(group.rht_signs_name), signs)
    quantization_config = json.loads(metadata["quantization_config"])
    assert quantization_config["policy"]["rht"] == "learned_hadamard_signs_v1"
    assert quantization_config["policy"]["rht_id"] == "tiny-learned-rht-v1"

    layer = load_quantized_vq_switch_linear(output, "model.layers.1.mlp.switch_mlp.gate_proj")
    np.testing.assert_array_equal(np.array(layer.rht_signs), signs.astype(np.float32))


def test_convert_vq_group_with_learned_rotation_writes_matrix_and_quantizes_transformed_weights(tmp_path) -> None:
    config_path, index_path = _write_complete_tiny_sparse_checkpoint(tmp_path)
    config = json.loads(config_path.read_text())
    index = load_safetensors_index(index_path)
    plan = plan_streaming_conversion_from_index(config, index, model_id="tiny-glm4", group_size=8)
    group = next(group for group in plan.vq_groups if group.projection == "gate_proj")
    output = tmp_path / "converted-gate-rotation.safetensors"
    rotation = np.eye(group.input_dims, dtype=np.float32)
    rotation[[0, 1]] = rotation[[1, 0]]

    converted = convert_vq_group_from_safetensors(
        source_dir=tmp_path,
        index=index,
        group=group,
        output_path=output,
        rotation_matrix=rotation,
        rotation_id="tiny-learned-rotation-v1",
    )

    assert converted.rotation_id == "tiny-learned-rotation-v1"
    with safe_open(output, framework="np") as handle:
        metadata = handle.metadata()
        np.testing.assert_array_equal(handle.get_tensor(group.rotation_matrix_name), rotation)
    quantization_config = json.loads(metadata["quantization_config"])
    assert quantization_config["policy"]["rotation"] == "learned_orthogonal_v1"
    assert quantization_config["policy"]["rotation_id"] == "tiny-learned-rotation-v1"

    layer = load_quantized_vq_switch_linear(output, "model.layers.1.mlp.switch_mlp.gate_proj")
    assert layer.get("rotation_matrix") is not None
    np.testing.assert_array_equal(np.array(layer.rotation_matrix), rotation)

    source_weight = _read_tensor_from_shard(
        tmp_path,
        "model-00001-of-00001.safetensors",
        "model.layers.1.mlp.experts.0.gate_proj.weight",
    )
    expected = quantize_weight_rtn(apply_rotation_np(source_weight, rotation), group_size=8)
    np.testing.assert_array_equal(np.array(layer.codes[0]), expected.codes)


def test_convert_vq_group_writes_e8p_metadata_for_code_bits_16(tmp_path) -> None:
    config_path, index_path = _write_complete_tiny_sparse_checkpoint(tmp_path)
    config = json.loads(config_path.read_text())
    index = load_safetensors_index(index_path)
    plan = plan_streaming_conversion_from_index(
        config,
        index,
        model_id="tiny-glm4",
        group_size=8,
        code_bits=16,
    )
    group = next(group for group in plan.vq_groups if group.projection == "gate_proj")
    output = tmp_path / "converted-gate-e8p.safetensors"

    converted = convert_vq_group_from_safetensors(
        source_dir=tmp_path,
        index=index,
        group=group,
        output_path=output,
    )

    assert converted.code_bits == 16
    with safe_open(output, framework="np") as handle:
        metadata = handle.metadata()
        assert handle.get_slice(group.codes_name).get_dtype().upper() in {"U16", "UINT16"}
    quantization_config = json.loads(metadata["quantization_config"])
    assert quantization_config["default_code_bits"] == 16
    assert quantization_config["codebook"]["name"] == "quip_e8p"
    assert quantization_config["codebook"]["sha256"] == E8P_PACKED_ABS_SHA256

    layer = load_quantized_vq_switch_linear(output, "model.layers.1.mlp.switch_mlp.gate_proj")
    assert layer.codes.shape == (2, 8, 2)
    assert layer.codes.dtype == mx.uint16
    assert layer.code_bits == 16


@pytest.mark.parametrize(
    "source_dtype",
    [mx.float32, mx.float16, mx.bfloat16],
    ids=["float32", "float16", "bfloat16"],
)
def test_streaming_tensor_reader_preserves_dense_float_sources(tmp_path, source_dtype) -> None:
    name = "model.layers.1.mlp.experts.0.gate_proj.weight"
    expected = np.arange(128, dtype=np.float32).reshape(8, 16) / 32
    mx.save_safetensors(
        str(tmp_path / "model-00001-of-00001.safetensors"),
        {name: mx.array(expected).astype(source_dtype)},
    )

    actual = _read_tensor_from_shard(tmp_path, "model-00001-of-00001.safetensors", name)

    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-2)


def test_convert_vq_groups_writes_manifest_and_multiple_group_shards(tmp_path) -> None:
    config_path, index_path = _write_complete_tiny_sparse_checkpoint(tmp_path)
    index = load_safetensors_index(index_path)
    config = json.loads(config_path.read_text())
    plan = plan_streaming_conversion_from_index(config, index, model_id="tiny-glm4", group_size=8)

    manifest = convert_vq_groups_from_safetensors(
        source_dir=tmp_path,
        index=index,
        plan=plan,
        output_dir=tmp_path / "converted",
        max_groups=2,
    )

    assert manifest.planned_vq_groups == 3
    assert len(manifest.converted_groups) == 2
    assert manifest.skipped_vq_groups == 1
    assert manifest.total_source_tensors_read == 4
    assert manifest.peak_source_tensor_bytes == 8 * 16 * 4
    assert manifest.source_weight_encoding.value == "dense"
    assert manifest.source_decoder is None
    assert manifest.manifest_path.exists()

    payload = json.loads(manifest.manifest_path.read_text())
    assert payload["converted_vq_groups"] == 2
    assert payload["scale_estimator"] == "max_abs"
    assert payload["source_weight_encoding"] == "dense"
    assert payload["source_decoder"] is None
    assert payload["code_bits_policy"] == {}
    assert payload["ready_vq_groups"] == 2
    assert payload["existing_vq_groups"] == 0
    assert payload["skipped_vq_groups"] == 1
    assert payload["groups"][0]["codes_shape"] == [2, 8, 2]
    assert payload["groups"][0]["code_bits"] == 8
    assert payload["groups"][0]["group_size"] == 8
    assert payload["groups"][0]["scale_estimator"] == "max_abs"
    assert payload["groups"][0]["elapsed_seconds"] >= 0
    assert payload["groups"][0]["expert_workers"] == 1
    assert payload["groups"][0]["source_weight_encoding"] == "dense"
    assert payload["groups"][0]["source_decoder"] is None
    assert payload["groups"][0]["status"] == "converted"
    assert Path(payload["groups"][0]["output_path"]).exists()

    layer = load_quantized_vq_switch_linear(
        manifest.converted_groups[0].output_path,
        "model.layers.1.mlp.switch_mlp.gate_proj",
    )
    assert layer.codes.shape == (2, 8, 2)


def test_convert_vq_groups_reuses_expert_pool_across_groups(tmp_path, monkeypatch) -> None:
    config_path, index_path = _write_complete_tiny_sparse_checkpoint(tmp_path)
    index = load_safetensors_index(index_path)
    config = json.loads(config_path.read_text())
    plan = plan_streaming_conversion_from_index(config, index, model_id="tiny-glm4", group_size=8)
    created_executors = []

    class ImmediateExecutor:
        def __init__(self, *, max_workers):
            self.max_workers = max_workers
            created_executors.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            from concurrent.futures import Future

            future = Future()
            try:
                future.set_result(fn(*args, **kwargs))
            except BaseException as exc:  # pragma: no cover - mirrors executor Future behavior
                future.set_exception(exc)
            return future

    monkeypatch.setattr(stream_convert, "ProcessPoolExecutor", ImmediateExecutor)

    manifest = convert_vq_groups_from_safetensors(
        source_dir=tmp_path,
        index=index,
        plan=plan,
        output_dir=tmp_path / "converted",
        max_groups=2,
        expert_workers=2,
    )

    assert len(created_executors) == 1
    assert created_executors[0].max_workers == 2
    assert len(manifest.converted_groups) == 2
    assert all(group.expert_workers == 2 for group in manifest.converted_groups)


def test_convert_vq_groups_preflights_missing_source_shards(tmp_path) -> None:
    config_path, index_path = _write_complete_tiny_sparse_checkpoint(tmp_path)
    (tmp_path / "model-00001-of-00001.safetensors").unlink()
    index = load_safetensors_index(index_path)
    config = json.loads(config_path.read_text())
    plan = plan_streaming_conversion_from_index(config, index, model_id="tiny-glm4", group_size=8)

    with pytest.raises(FileNotFoundError, match="missing source shard"):
        convert_vq_groups_from_safetensors(
            source_dir=tmp_path,
            index=index,
            plan=plan,
            output_dir=tmp_path / "converted",
            max_groups=1,
        )


def test_source_shard_inventory_reports_present_and_missing_shards(tmp_path) -> None:
    config_path, index_path = _write_complete_tiny_sparse_checkpoint(tmp_path)
    index = load_safetensors_index(index_path)
    config = json.loads(config_path.read_text())
    plan = plan_streaming_conversion_from_index(config, index, model_id="tiny-glm4", group_size=8)

    inventory = source_shard_inventory(tmp_path, plan.vq_groups)
    assert inventory.required_shards == 1
    assert inventory.present_shards == 1
    assert inventory.missing_shard_count == 0
    assert inventory.present_bytes > 0

    (tmp_path / "model-00001-of-00001.safetensors").unlink()
    missing = source_shard_inventory(tmp_path, plan.vq_groups)
    assert missing.required_shards == 1
    assert missing.present_shards == 0
    assert missing.missing_shard_count == 1
    assert missing.to_json_dict()["missing_shards"] == 1


def test_convert_vq_groups_skip_existing_resumes_without_overwrite(tmp_path) -> None:
    config_path, index_path = _write_complete_tiny_sparse_checkpoint(tmp_path)
    index = load_safetensors_index(index_path)
    config = json.loads(config_path.read_text())
    plan = plan_streaming_conversion_from_index(config, index, model_id="tiny-glm4", group_size=8)
    output_dir = tmp_path / "converted"

    convert_vq_groups_from_safetensors(
        source_dir=tmp_path,
        index=index,
        plan=plan,
        output_dir=output_dir,
        max_groups=1,
    )
    with pytest.raises(FileExistsError, match="already exists"):
        convert_vq_groups_from_safetensors(
            source_dir=tmp_path,
            index=index,
            plan=plan,
            output_dir=output_dir,
            max_groups=1,
        )

    manifest = convert_vq_groups_from_safetensors(
        source_dir=tmp_path,
        index=index,
        plan=plan,
        output_dir=output_dir,
        max_groups=2,
        skip_existing=True,
    )

    assert len(manifest.converted_groups) == 1
    assert len(manifest.existing_groups) == 1
    assert manifest.skipped_existing_outputs == 1
    assert manifest.skipped_vq_groups == 1
    assert (output_dir / "layer-00001-gate_proj.safetensors").exists()
    assert (output_dir / "layer-00001-up_proj.safetensors").exists()


def test_modelopt_nvfp4_manifest_preserves_source_lineage_when_skipping_existing(tmp_path) -> None:
    config_path, index_path, _ = _write_tiny_modelopt_nvfp4_checkpoint(tmp_path)
    config = json.loads(config_path.read_text())
    index = load_safetensors_index(index_path)
    plan = plan_streaming_conversion_from_index(
        config,
        index,
        model_id="tiny-modelopt-nvfp4",
        group_size=8,
    )
    output_dir = tmp_path / "converted"

    first = convert_vq_groups_from_safetensors(
        source_dir=tmp_path,
        index=index,
        plan=plan,
        output_dir=output_dir,
        max_groups=1,
    )
    first_payload = json.loads(first.manifest_path.read_text())
    assert first.source_weight_encoding.value == "modelopt_nvfp4"
    assert first.source_decoder == "modelopt_nvfp4_v1"
    assert first_payload["source_weight_encoding"] == "modelopt_nvfp4"
    assert first_payload["source_decoder"] == "modelopt_nvfp4_v1"

    resumed = convert_vq_groups_from_safetensors(
        source_dir=tmp_path,
        index=index,
        plan=plan,
        output_dir=output_dir,
        max_groups=1,
        skip_existing=True,
    )
    existing = resumed.existing_groups[0]
    resumed_payload = json.loads(resumed.manifest_path.read_text())

    assert existing.source_weight_encoding.value == "modelopt_nvfp4"
    assert existing.source_decoder == "modelopt_nvfp4_v1"
    assert resumed_payload["groups"][0]["source_weight_encoding"] == "modelopt_nvfp4"
    assert resumed_payload["groups"][0]["source_decoder"] == "modelopt_nvfp4_v1"


@pytest.mark.parametrize("code_bits", [8, 16])
def test_modelopt_nvfp4_selected_group_preserves_revision_lineage_and_resumes(
    tmp_path: Path,
    code_bits: int,
) -> None:
    config_path, index_path, _ = _write_tiny_modelopt_nvfp4_checkpoint(
        tmp_path,
        projections=("gate_proj", "up_proj", "down_proj"),
    )
    config = json.loads(config_path.read_text())
    index = load_safetensors_index(index_path)
    revision = "0123456789abcdef"
    plan = plan_streaming_conversion_from_index(
        config,
        index,
        model_id="tiny-modelopt-nvfp4",
        revision=revision,
        config_sha256="a" * 64,
        index_sha256="b" * 64,
        source_profile="tiny-modelopt-profile",
        group_size=8,
        code_bits=code_bits,
    )
    selected = next(group for group in plan.vq_groups if group.projection == "gate_proj")
    output_dir = tmp_path / f"selected-{code_bits}"

    first = convert_vq_groups_from_safetensors(
        source_dir=tmp_path,
        index=index,
        plan=plan,
        output_dir=output_dir,
        selected_groups=(selected,),
    )
    output_path = first.converted_groups[0].output_path
    first_bytes = output_path.read_bytes()
    with safe_open(output_path, framework="np") as handle:
        policy = json.loads(handle.metadata()["quantization_config"])["policy"]
    first_payload = json.loads(first.manifest_path.read_text())

    assert policy["source_model_id"] == "tiny-modelopt-nvfp4"
    assert policy["source_revision"] == revision
    assert policy["source_config_sha256"] == "a" * 64
    assert policy["source_index_sha256"] == "b" * 64
    assert policy["source_profile"] == "tiny-modelopt-profile"
    assert policy["decoded_expert_working_set"] == "one_per_worker"
    assert first.source_revision == revision
    assert first_payload["source_revision"] == revision
    assert first_payload["selected_vq_groups"] == 1
    assert first_payload["planned_vq_groups"] == 3
    assert not (output_dir / "layer-00001-up_proj.safetensors").exists()

    resumed = convert_vq_groups_from_safetensors(
        source_dir=tmp_path,
        index=index,
        plan=plan,
        output_dir=output_dir,
        selected_groups=(selected,),
        skip_existing=True,
    )
    assert len(resumed.existing_groups) == 1
    assert output_path.read_bytes() == first_bytes

    with pytest.raises(ValueError, match="exact members of plan.vq_groups"):
        convert_vq_groups_from_safetensors(
            source_dir=tmp_path,
            index=index,
            plan=plan,
            output_dir=tmp_path / f"injected-{code_bits}",
            selected_groups=(replace(selected, experts=(0,)),),
        )


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("source_model_id", "wrong/model"),
        ("source_revision", "wrong-revision"),
        ("source_config_sha256", "0" * 64),
        ("source_index_sha256", "1" * 64),
        ("source_profile", "wrong-profile"),
    ],
)
def test_modelopt_nvfp4_resume_rejects_pinned_source_lineage_drift(
    tmp_path: Path,
    field: str,
    wrong_value: str,
) -> None:
    config_path, index_path, _ = _write_tiny_modelopt_nvfp4_checkpoint(tmp_path)
    index = load_safetensors_index(index_path)
    plan = plan_streaming_conversion_from_index(
        json.loads(config_path.read_text()),
        index,
        model_id="tiny-modelopt-nvfp4",
        revision="0123456789abcdef",
        config_sha256="a" * 64,
        index_sha256="b" * 64,
        source_profile="tiny-modelopt-profile",
        group_size=8,
    )
    output_dir = tmp_path / f"lineage-{field}"
    first = convert_vq_groups_from_safetensors(
        source_dir=tmp_path,
        index=index,
        plan=plan,
        output_dir=output_dir,
        max_groups=1,
    )
    output_path = first.converted_groups[0].output_path
    with safe_open(output_path, framework="np") as handle:
        arrays = {name: handle.get_tensor(name) for name in handle.keys()}
        metadata = dict(handle.metadata() or {})
    quantization_config = json.loads(metadata["quantization_config"])
    quantization_config["policy"][field] = wrong_value
    metadata["quantization_config"] = json.dumps(quantization_config)
    mx.save_safetensors(
        str(output_path),
        {name: mx.array(value) for name, value in arrays.items()},
        metadata=metadata,
    )

    with pytest.raises(ValueError, match=field):
        convert_vq_groups_from_safetensors(
            source_dir=tmp_path,
            index=index,
            plan=plan,
            output_dir=output_dir,
            max_groups=1,
            skip_existing=True,
        )


def test_convert_vq_group_does_not_publish_partial_output_on_save_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path, index_path, _ = _write_tiny_modelopt_nvfp4_checkpoint(tmp_path)
    plan = plan_streaming_conversion_from_index(
        json.loads(config_path.read_text()),
        load_safetensors_index(index_path),
        model_id="tiny-modelopt-nvfp4",
        revision="0123456789abcdef",
        group_size=8,
    )
    group = next(group for group in plan.vq_groups if group.projection == "gate_proj")
    output_path = tmp_path / "atomic-output.safetensors"

    def fail_after_partial_write(path, arrays, *, metadata):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("injected save failure")

    monkeypatch.setattr(stream_convert.mx, "save_safetensors", fail_after_partial_write)

    with pytest.raises(RuntimeError, match="injected save failure"):
        convert_vq_group_from_safetensors(
            source_dir=tmp_path,
            index=load_safetensors_index(index_path),
            group=group,
            output_path=output_path,
        )

    assert not output_path.exists()
    assert list(tmp_path.glob("*.partial-*")) == []


def test_modelopt_nvfp4_resume_rejects_existing_artifact_with_wrong_decoder_lineage(tmp_path) -> None:
    config_path, index_path, _ = _write_tiny_modelopt_nvfp4_checkpoint(tmp_path)
    config = json.loads(config_path.read_text())
    index = load_safetensors_index(index_path)
    plan = plan_streaming_conversion_from_index(
        config,
        index,
        model_id="tiny-modelopt-nvfp4",
        group_size=8,
    )
    output_dir = tmp_path / "converted"
    first = convert_vq_groups_from_safetensors(
        source_dir=tmp_path,
        index=index,
        plan=plan,
        output_dir=output_dir,
        max_groups=1,
    )
    output_path = first.converted_groups[0].output_path
    with safe_open(output_path, framework="np") as handle:
        arrays = {name: handle.get_tensor(name) for name in handle.keys()}
        metadata = dict(handle.metadata() or {})
    quantization_config = json.loads(metadata["quantization_config"])
    quantization_config["policy"]["source_weight_encoding"] = "dense"
    quantization_config["policy"].pop("source_decoder")
    metadata["quantization_config"] = json.dumps(quantization_config)
    mx.save_safetensors(
        str(output_path),
        {name: mx.array(value) for name, value in arrays.items()},
        metadata=metadata,
    )

    with pytest.raises(ValueError, match="source_weight_encoding|source_decoder"):
        convert_vq_groups_from_safetensors(
            source_dir=tmp_path,
            index=index,
            plan=plan,
            output_dir=output_dir,
            max_groups=1,
            skip_existing=True,
        )


def test_modelopt_nvfp4_resume_rejects_extra_dense_routed_tensor(tmp_path) -> None:
    config_path, index_path, _ = _write_tiny_modelopt_nvfp4_checkpoint(tmp_path)
    index = load_safetensors_index(index_path)
    plan = plan_streaming_conversion_from_index(
        json.loads(config_path.read_text()),
        index,
        model_id="tiny-modelopt-nvfp4",
        revision="0123456789abcdef",
        config_sha256="a" * 64,
        index_sha256="b" * 64,
        source_profile="tiny-modelopt-profile",
        group_size=8,
    )
    output_dir = tmp_path / "extra-dense"
    first = convert_vq_groups_from_safetensors(
        source_dir=tmp_path,
        index=index,
        plan=plan,
        output_dir=output_dir,
        max_groups=1,
    )
    output_path = first.converted_groups[0].output_path
    with safe_open(output_path, framework="np") as handle:
        arrays = {name: handle.get_tensor(name) for name in handle.keys()}
        metadata = dict(handle.metadata() or {})
    arrays["model.layers.1.mlp.experts.0.gate_proj.weight"] = np.zeros(
        (16, 16),
        dtype=np.float16,
    )
    mx.save_safetensors(
        str(output_path),
        {name: mx.array(value) for name, value in arrays.items()},
        metadata=metadata,
    )

    with pytest.raises(ValueError, match="unexpected tensor"):
        convert_vq_groups_from_safetensors(
            source_dir=tmp_path,
            index=index,
            plan=plan,
            output_dir=output_dir,
            max_groups=1,
            skip_existing=True,
        )


@pytest.mark.parametrize("tamper", ["missing", "wrong"], ids=["missing", "wrong"])
def test_modelopt_nvfp4_resume_rejects_incompatible_embedded_codebook(tmp_path, tamper) -> None:
    config_path, index_path, _ = _write_tiny_modelopt_nvfp4_checkpoint(tmp_path)
    config = json.loads(config_path.read_text())
    index = load_safetensors_index(index_path)
    plan = plan_streaming_conversion_from_index(
        config,
        index,
        model_id="tiny-modelopt-nvfp4",
        group_size=8,
    )
    output_dir = tmp_path / "converted"
    first = convert_vq_groups_from_safetensors(
        source_dir=tmp_path,
        index=index,
        plan=plan,
        output_dir=output_dir,
        max_groups=1,
    )
    output_path = first.converted_groups[0].output_path
    with safe_open(output_path, framework="np") as handle:
        arrays = {name: handle.get_tensor(name) for name in handle.keys()}
        metadata = dict(handle.metadata() or {})
    if tamper == "missing":
        arrays.pop("model.vq_codebook.e8")
    else:
        wrong_codebook = arrays["model.vq_codebook.e8"].copy()
        wrong_codebook[0] ^= np.uint32(1)
        arrays["model.vq_codebook.e8"] = wrong_codebook
    mx.save_safetensors(
        str(output_path),
        {name: mx.array(value) for name, value in arrays.items()},
        metadata=metadata,
    )

    with pytest.raises(ValueError, match="model.vq_codebook.e8"):
        convert_vq_groups_from_safetensors(
            source_dir=tmp_path,
            index=index,
            plan=plan,
            output_dir=output_dir,
            max_groups=1,
            skip_existing=True,
        )


def test_convert_vq_groups_skip_existing_avoids_expert_pool(tmp_path, monkeypatch) -> None:
    config_path, index_path = _write_complete_tiny_sparse_checkpoint(tmp_path)
    index = load_safetensors_index(index_path)
    config = json.loads(config_path.read_text())
    plan = plan_streaming_conversion_from_index(config, index, model_id="tiny-glm4", group_size=8)
    output_dir = tmp_path / "converted"

    convert_vq_groups_from_safetensors(
        source_dir=tmp_path,
        index=index,
        plan=plan,
        output_dir=output_dir,
        max_groups=2,
    )
    created_executors = []

    class RecordingExecutor:
        def __init__(self, *, max_workers):
            self.max_workers = max_workers
            created_executors.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(stream_convert, "ProcessPoolExecutor", RecordingExecutor)

    manifest = convert_vq_groups_from_safetensors(
        source_dir=tmp_path,
        index=index,
        plan=plan,
        output_dir=output_dir,
        max_groups=2,
        skip_existing=True,
        expert_workers=2,
    )

    assert created_executors == []
    assert len(manifest.converted_groups) == 0
    assert len(manifest.existing_groups) == 2
    assert manifest.skipped_existing_outputs == 2


def _write_index(path, payload):
    path.write_text(json.dumps(payload))
    return path


def _modelopt_nvfp4_quantization_config() -> dict:
    return {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "producer": {"name": "modelopt", "version": "test-v1"},
        "config_groups": {
            "group_0": {
                "weights": {
                    "dynamic": False,
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                }
            }
        },
    }


def _tiny_modelopt_nvfp4_config(*, num_hidden_layers: int = 2, dense_layers: int = 1) -> dict:
    return {
        "model_type": "glm4_moe",
        "num_hidden_layers": num_hidden_layers,
        "first_k_dense_replace": dense_layers,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "hidden_size": 16,
        "moe_intermediate_size": 16,
        "intermediate_size": 32,
        "max_position_embeddings": 128,
        "quantization_config": _modelopt_nvfp4_quantization_config(),
    }


def _write_raw_safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    header = {}
    payload = bytearray()
    for name, (dtype, shape, raw) in tensors.items():
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    encoded_header = json.dumps(header, separators=(",", ":")).encode()
    encoded_header += b" " * (-len(encoded_header) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded_header)) + encoded_header + payload)


def _write_tiny_modelopt_nvfp4_checkpoint(
    tmp_path,
    *,
    projections=("gate_proj",),
):
    config = _tiny_modelopt_nvfp4_config()
    weight_tensors = {}
    block_scale_tensors = {}
    global_scale_tensors = {}
    weight_map = {}
    dense_oracle = {}
    for expert, packed_code in ((0, 0x11), (1, 0x99)):
        for projection in projections:
            weight_name = (
                f"model.layers.1.mlp.experts.{expert}.{projection}.weight"
            )
            block_scale_name = f"{weight_name}_scale"
            global_scale_name = f"{weight_name}_scale_2"
            weight_tensors[weight_name] = (
                "U8",
                (16, 8),
                bytes([packed_code]) * (16 * 8),
            )
            block_scale_tensors[block_scale_name] = (
                "F8_E4M3",
                (16, 1),
                bytes([0x38]) * 16,
            )
            global_scale_tensors[global_scale_name] = (
                "F32",
                (),
                np.float32(2.0).tobytes(),
            )
            weight_map[weight_name] = "packed-weights.safetensors"
            weight_map[block_scale_name] = "block-scales.safetensors"
            weight_map[global_scale_name] = "global-scales.safetensors"
        dense_oracle[expert] = np.full(
            (16, 16),
            1.0 if expert == 0 else -1.0,
            dtype=np.float32,
        )

    _write_raw_safetensors(tmp_path / "packed-weights.safetensors", weight_tensors)
    _write_raw_safetensors(tmp_path / "block-scales.safetensors", block_scale_tensors)
    _write_raw_safetensors(tmp_path / "global-scales.safetensors", global_scale_tensors)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    index_path = _write_index(tmp_path / "index.json", {"metadata": {}, "weight_map": weight_map})
    return config_path, index_path, dense_oracle


def _write_complete_tiny_sparse_checkpoint(tmp_path):
    config = {
        "model_type": "glm4_moe",
        "num_hidden_layers": 2,
        "first_k_dense_replace": 1,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "hidden_size": 16,
        "moe_intermediate_size": 8,
        "intermediate_size": 32,
        "max_position_embeddings": 128,
    }
    rng = np.random.default_rng(2027)
    weight_map = {}
    arrays = {}
    for expert in range(2):
        for projection, shape in {
            "gate_proj": (8, 16),
            "up_proj": (8, 16),
            "down_proj": (16, 8),
        }.items():
            name = f"model.layers.1.mlp.experts.{expert}.{projection}.weight"
            arrays[name] = mx.array(rng.normal(scale=0.04, size=shape).astype(np.float32))
            weight_map[name] = "model-00001-of-00001.safetensors"

    mx.save_safetensors(str(tmp_path / "model-00001-of-00001.safetensors"), arrays)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    index_path = _write_index(tmp_path / "index.json", {"metadata": {}, "weight_map": weight_map})
    return config_path, index_path


def test_plan_stream_convert_cli_dry_run_uses_local_index_and_config(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    index_path = tmp_path / "index.json"
    config_path.write_text(json.dumps(GLM45_AIR_CONFIG))
    _write_index(index_path, _index_for_config(GLM45_AIR_CONFIG, shard_count=3))

    result = subprocess.run(
        [
            sys.executable,
            "scripts/plan_stream_convert.py",
            "--model-id",
            "zai-org/GLM-4.5-Air",
            "--config-path",
            str(config_path),
            "--index-path",
            str(index_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["vq_groups"] == 45 * 3
    assert payload["missing_vq_groups"] == []
    assert payload["source_shards"] == 5
    assert payload["code_bits_policy"] == {}
    assert payload["source_weight_encoding"] == "dense"
    assert payload["source_decoder"] is None


def test_plan_stream_convert_cli_reports_mixed_code_bits_policy(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    index_path = tmp_path / "index.json"
    config_path.write_text(json.dumps(GLM45_AIR_CONFIG))
    _write_index(index_path, _index_for_config(GLM45_AIR_CONFIG, shard_count=3))

    result = subprocess.run(
        [
            sys.executable,
            "scripts/plan_stream_convert.py",
            "--model-id",
            "zai-org/GLM-4.5-Air",
            "--config-path",
            str(config_path),
            "--index-path",
            str(index_path),
            "--code-bits-policy",
            "1:gate_proj=16",
            "--code-bits-policy",
            "*:down_proj=16",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["code_bits_policy"] == {"*:down_proj": 16, "1:gate_proj": 16}
    assert payload["vq_code_bytes"] > plan_streaming_conversion_from_index(
        GLM45_AIR_CONFIG,
        load_safetensors_index(index_path),
        model_id="zai-org/GLM-4.5-Air",
    ).vq_code_bytes


def test_plan_stream_convert_cli_reports_local_source_shard_inventory(tmp_path) -> None:
    config_path, index_path = _write_complete_tiny_sparse_checkpoint(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/plan_stream_convert.py",
            "--model-id",
            "tiny-glm4",
            "--config-path",
            str(config_path),
            "--index-path",
            str(index_path),
            "--source-dir",
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["source_shards_local"]["required_shards"] == 1
    assert payload["source_shards_local"]["present_shards"] == 1
    assert payload["source_shards_local"]["missing_shards"] == 0
    assert payload["source_shards_local"]["present_bytes"] > 0


def test_plan_stream_convert_cli_reports_missing_local_source_shards_without_conversion(tmp_path) -> None:
    config_path, index_path = _write_complete_tiny_sparse_checkpoint(tmp_path)
    empty_source_dir = tmp_path / "empty"
    empty_source_dir.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "scripts/plan_stream_convert.py",
            "--model-id",
            "tiny-glm4",
            "--config-path",
            str(config_path),
            "--index-path",
            str(index_path),
            "--source-dir",
            str(empty_source_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["vq_groups"] == 3
    assert payload["source_shards_local"]["required_shards"] == 1
    assert payload["source_shards_local"]["present_shards"] == 0
    assert payload["source_shards_local"]["missing_shards"] == 1
    assert len(payload["source_shards_local"]["missing_examples"]) == 1


def test_plan_stream_convert_cli_download_missing_source_shards_dry_run(tmp_path) -> None:
    config_path, index_path = _write_complete_tiny_sparse_checkpoint(tmp_path)
    empty_source_dir = tmp_path / "empty"
    empty_source_dir.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "scripts/plan_stream_convert.py",
            "--model-id",
            "tiny-glm4",
            "--config-path",
            str(config_path),
            "--index-path",
            str(index_path),
            "--source-dir",
            str(empty_source_dir),
            "--download-missing-source-shards",
            "--download-dry-run",
            "--max-download-shards",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["source_shard_download"]["dry_run"] is True
    assert payload["source_shard_download"]["requested_shards"] == 1
    assert payload["source_shard_download"]["downloaded_shards"] == 0
    assert payload["source_shard_download"]["requested_filenames"] == ["model-00001-of-00001.safetensors"]
    assert payload["source_shard_download"]["before"]["missing_shards"] == 1
    assert payload["source_shard_download"]["after"]["missing_shards"] == 1


def test_plan_stream_convert_cli_convert_all_writes_manifest(tmp_path) -> None:
    config_path, index_path = _write_complete_tiny_sparse_checkpoint(tmp_path)
    output_dir = tmp_path / "cli-converted"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/plan_stream_convert.py",
            "--model-id",
            "tiny-glm4",
            "--config-path",
            str(config_path),
            "--index-path",
            str(index_path),
            "--source-dir",
            str(tmp_path),
            "--convert-all",
            "--output-dir",
            str(output_dir),
            "--max-groups",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["vq_groups"] == 3
    assert payload["manifest"]["converted_vq_groups"] == 1
    assert payload["manifest"]["existing_vq_groups"] == 0
    assert payload["manifest"]["ready_vq_groups"] == 1
    assert payload["manifest"]["skipped_vq_groups"] == 2
    assert payload["manifest"]["skipped_existing_outputs"] == 0
    assert Path(payload["manifest"]["path"]).exists()
    assert (output_dir / "layer-00001-gate_proj.safetensors").exists()


def test_plan_stream_convert_cli_reports_modelopt_nvfp4_decoder_and_manifest(tmp_path) -> None:
    config_path, index_path, _ = _write_tiny_modelopt_nvfp4_checkpoint(tmp_path)
    output_dir = tmp_path / "cli-modelopt-nvfp4"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/plan_stream_convert.py",
            "--model-id",
            "tiny-modelopt-nvfp4",
            "--config-path",
            str(config_path),
            "--index-path",
            str(index_path),
            "--source-dir",
            str(tmp_path),
            "--group-size",
            "8",
            "--convert-all",
            "--output-dir",
            str(output_dir),
            "--max-groups",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    manifest_payload = json.loads(Path(payload["manifest"]["path"]).read_text())

    assert payload["source_weight_encoding"] == "modelopt_nvfp4"
    assert payload["source_decoder"] == "modelopt_nvfp4_v1"
    assert payload["source_shards_local"]["required_shards"] == 3
    assert payload["manifest"]["source_weight_encoding"] == "modelopt_nvfp4"
    assert payload["manifest"]["source_decoder"] == "modelopt_nvfp4_v1"
    assert manifest_payload["source_weight_encoding"] == "modelopt_nvfp4"
    assert manifest_payload["source_decoder"] == "modelopt_nvfp4_v1"


def test_plan_stream_convert_cli_convert_all_defaults_source_dir_to_index_parent(tmp_path) -> None:
    config_path, index_path = _write_complete_tiny_sparse_checkpoint(tmp_path)
    output_dir = tmp_path / "cli-converted-index-parent"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/plan_stream_convert.py",
            "--model-id",
            "tiny-glm4",
            "--config-path",
            str(config_path),
            "--index-path",
            str(index_path),
            "--convert-all",
            "--output-dir",
            str(output_dir),
            "--max-groups",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["source_dir"] == str(index_path.parent)
    assert payload["source_shards_local"]["present_shards"] == 1
    assert payload["manifest"]["converted_vq_groups"] == 1
    assert (output_dir / "layer-00001-gate_proj.safetensors").exists()


def test_plan_stream_convert_cli_convert_group_reports_elapsed_seconds(tmp_path) -> None:
    config_path, index_path = _write_complete_tiny_sparse_checkpoint(tmp_path)
    output_path = tmp_path / "cli-converted-gate.safetensors"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/plan_stream_convert.py",
            "--model-id",
            "tiny-glm4",
            "--config-path",
            str(config_path),
            "--index-path",
            str(index_path),
            "--source-dir",
            str(tmp_path),
            "--convert-group",
            "1:gate_proj",
            "--output-path",
            str(output_path),
            "--expert-workers",
            "2",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["converted"]["output_path"] == str(output_path)
    assert payload["converted"]["elapsed_seconds"] >= 0
    assert payload["converted"]["expert_workers"] == 2
    assert payload["expert_workers"] == 2
    assert output_path.exists()


def test_plan_stream_convert_cli_accepts_scale_estimator_variant(tmp_path) -> None:
    config_path, index_path = _write_complete_tiny_sparse_checkpoint(tmp_path)
    output_dir = tmp_path / "cli-converted-percentile"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/plan_stream_convert.py",
            "--model-id",
            "tiny-glm4",
            "--config-path",
            str(config_path),
            "--index-path",
            str(index_path),
            "--source-dir",
            str(tmp_path),
            "--convert-all",
            "--output-dir",
            str(output_dir),
            "--max-groups",
            "1",
            "--scale-estimator",
            "percentile_99",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    manifest = json.loads(Path(payload["manifest"]["path"]).read_text())

    assert payload["scale_estimator"] == "percentile_99"
    assert payload["manifest"]["scale_estimator"] == "percentile_99"
    assert payload["manifest"]["code_bits_policy"] == {}
    assert manifest["scale_estimator"] == "percentile_99"
    assert manifest["groups"][0]["scale_estimator"] == "percentile_99"
