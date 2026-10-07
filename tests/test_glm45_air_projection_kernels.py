from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import mlx_vq.benchmark.projection_kernels as projection_kernels
from mlx_vq.benchmark.projection_kernels import (
    _call_dense_or_mlx,
    _call_mlx_gather_mm_bf16_routed,
    _call_mlx_q2_routed,
    _call_nax_e8_fp16_routed,
    _call_nax_e8_fp16_sorted_steel_routed,
    _call_nax_e8_fp16_steel_routed,
    _call_nax_e8p_fp16_sorted_steel_routed,
    _call_nax_e8p_fp16_sorted_steel_raw,
    _call_nax_e8_int8_routed,
    _call_nax_predecoded_fp16_routed,
    _call_vq_decode_to_scratch_gather_mm_routed,
    _compact_nonempty_sorted_tile_descriptors,
    _dense_switch_linear,
    _prepare_sorted_steel_inputs,
    _vq_switch_linear,
    build_projection_fixture_from_vq_artifact,
    build_projection_fixture,
    decode_bandwidth_floor_report,
    projection_variants_for_cli,
    run_nax_int8_raw_ceiling,
    run_loaded_projection_variant,
    run_projection_variant,
)
from mlx_vq.codebook.e8 import e8p_packed_abs_grid
from mlx_vq.kernels import nax


def _write_e8p_projection_artifact(
    root: Path,
    *,
    layer_index: int = 45,
    projection: str = "gate_proj",
    num_experts: int = 4,
    input_dims: int = 64,
    output_dims: int = 32,
    group_size: int = 64,
    seed: int = 9201,
) -> Path:
    prefix = f"model.layers.{layer_index}.mlp.switch_mlp.{projection}"
    rng = np.random.default_rng(seed)
    codes = rng.integers(
        0,
        65536,
        size=(num_experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales = rng.uniform(
        0.01,
        0.05,
        size=(num_experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    shard = root / f"layer-{layer_index:05d}-{projection}.safetensors"
    mx.save_safetensors(
        str(shard),
        {
            f"{prefix}.codes": mx.array(codes),
            f"{prefix}.scales": mx.array(scales),
            "model.vq_codebook.e8": mx.array(e8p_packed_abs_grid()),
        },
    )
    return shard


def test_projection_fixture_builds_dense_mlx_and_vq_variants() -> None:
    fixture = build_projection_fixture(
        projection="gate_up",
        tokens=3,
        top_k=2,
        experts=4,
        input_dims=64,
        output_dims=32,
        mlx_group_size=32,
        vq_preferred_group_size=64,
        seed=123,
    )

    assert fixture.dense_weight.shape == (4, 32, 64)
    assert fixture.indices.shape == (3, 2)
    assert fixture.vq_group_size == 64
    assert fixture.mlx_group_size == 32
    assert fixture.vq_code_bits == 8


def test_projection_fixture_can_build_16bit_vq_probe() -> None:
    fixture = build_projection_fixture(
        projection="gate_up",
        tokens=2,
        top_k=2,
        experts=4,
        input_dims=64,
        output_dims=32,
        mlx_group_size=32,
        vq_preferred_group_size=64,
        vq_code_bits=16,
        seed=124,
    )

    layer = _vq_switch_linear(fixture)
    record = run_projection_variant(fixture, variant="vq_e1", iterations=1, warmup=0)

    assert fixture.vq_code_bits == 16
    assert layer.code_bits == 16
    assert record["vq_code_bits"] == 16
    assert record["vq_group_size"] == 64
    assert record["finite_output"] is True


def test_artifact_projection_fixture_loads_prequantized_e8p_layer(tmp_path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    shard = _write_e8p_projection_artifact(artifact)

    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="gate_proj",
        tokens=3,
        top_k=2,
        mlx_group_size=32,
        seed=141,
    )
    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="vq_e1",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert fixture.dense_weight is None
    assert fixture.projection == "gate_up"
    assert fixture.input_dims == 64
    assert fixture.output_dims == 32
    assert fixture.experts == 4
    assert fixture.vq_code_bits == 16
    assert fixture.vq_group_size == 64
    assert layer.codes.dtype == mx.uint16
    assert metadata["artifact_shard"] == str(shard)
    assert record["fixture_source"] == "vq_artifact"
    assert record["artifact_projection"] == "gate_proj"
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [3, 2, 32]
    assert record["finite_output"] is True


def test_artifact_nax_e8p_sorted_steel_variant_uses_loaded_16bit_layer(tmp_path) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(artifact)
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="gate_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=32,
        seed=142,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_fp16_sorted_steel",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["variant"] == "nax_e8p_fp16_sorted_steel"
    assert record["diagnostic_phase"] == "nax_e8p_fp16_sorted_steel"
    assert "excludes RTN/E8P encode cost" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [6, 2, 32]
    assert record["finite_output"] is True


def test_artifact_nax_e8p_raw_sorted_variant_times_prepared_native_call(tmp_path) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(artifact)
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="gate_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=32,
        seed=143,
    )
    sorted_inputs = _prepare_sorted_steel_inputs(fixture)
    raw_output = _call_nax_e8p_fp16_sorted_steel_raw(layer, fixture, sorted_inputs)
    full_output = _call_nax_e8p_fp16_sorted_steel_routed(layer, fixture)
    mx.eval(raw_output, full_output)

    expected_sorted = full_output.reshape((-1, fixture.output_dims))[sorted_inputs.order]
    mx.eval(expected_sorted)
    raw_np = np.array(raw_output.astype(mx.float32))
    expected_np = np.array(expected_sorted.astype(mx.float32))
    dot = float(np.sum(raw_np * expected_np))
    norm = float(np.sqrt(np.sum(raw_np * raw_np) * np.sum(expected_np * expected_np)))
    assert raw_output.shape == (fixture.tokens * fixture.top_k, fixture.output_dims)
    assert dot / norm >= 0.9999

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_fp16_sorted_steel_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["variant"] == "nax_e8p_fp16_sorted_steel_raw"
    assert record["diagnostic_phase"] == "nax_e8p_fp16_sorted_steel_raw"
    assert "route sort/scatter overhead" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 32]
    assert record["finite_output"] is True


def test_artifact_nax_e8p_direct_reduce_raw_variant_times_prepared_native_call(tmp_path) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(artifact)
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="gate_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=32,
        seed=145,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_fp16_sorted_direct_reduce_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["variant"] == "nax_e8p_fp16_sorted_direct_reduce_raw"
    assert record["diagnostic_phase"] == "nax_e8p_fp16_sorted_direct_reduce_raw"
    assert "no decoded-B threadgroup staging" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 32]
    assert record["finite_output"] is True


def test_artifact_nax_e8p_inline_b_raw_variant_times_prepared_native_call(tmp_path) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(artifact)
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="gate_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=32,
        seed=146,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_fp16_sorted_inline_b_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["variant"] == "nax_e8p_fp16_sorted_inline_b_raw"
    assert record["diagnostic_phase"] == "nax_e8p_fp16_sorted_inline_b_raw"
    assert "inline B-tile decode" in record["diagnostic_note"]
    assert "no decoded-B threadgroup staging" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 32]
    assert record["finite_output"] is True


def test_artifact_nax_e8p_packed_rhs_tiled_raw_variant_times_prepared_native_call(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(artifact)
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="gate_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=32,
        seed=148,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_packed_rhs_sorted_tiled_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["variant"] == "nax_e8p_packed_rhs_sorted_tiled_raw"
    assert record["diagnostic_phase"] == "nax_e8p_packed_rhs_sorted_tiled_raw"
    assert "packed RHS" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 32]
    assert record["finite_output"] is True


def test_artifact_nax_e8p_split_byte_rhs_tiled_raw_variant_times_prepared_native_call(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(artifact)
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="gate_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=32,
        seed=151,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_split_byte_rhs_sorted_tiled_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["variant"] == "nax_e8p_split_byte_rhs_sorted_tiled_raw"
    assert record["diagnostic_phase"] == "nax_e8p_split_byte_rhs_sorted_tiled_raw"
    assert "split-byte RHS" in record["diagnostic_note"]
    assert "sign/abs/parity" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 32]
    assert record["finite_output"] is True


def test_artifact_nax_e8p_split_byte_factor_reuse_rhs_tiled_raw_variant_times_prepared_native_call(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(artifact)
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="gate_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=32,
        seed=152,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_split_byte_factor_reuse_rhs_sorted_tiled_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["variant"] == "nax_e8p_split_byte_factor_reuse_rhs_sorted_tiled_raw"
    assert record["diagnostic_phase"] == "nax_e8p_split_byte_factor_reuse_rhs_sorted_tiled_raw"
    assert "factor-reuse RHS" in record["diagnostic_note"]
    assert "sign/abs LUT" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 32]
    assert record["finite_output"] is True


def test_artifact_nax_e8p_split_byte_factor_reuse_rhs_sorted_native_raw_variant_times_prepared_native_call(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(artifact)
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="gate_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=32,
        seed=153,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_split_byte_factor_reuse_rhs_sorted_native_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["variant"] == "nax_e8p_split_byte_factor_reuse_rhs_sorted_native_raw"
    assert record["diagnostic_phase"] == "nax_e8p_split_byte_factor_reuse_rhs_sorted_native_raw"
    assert "factor-reuse RHS" in record["diagnostic_note"]
    assert "non-staged" in record["diagnostic_note"]
    assert "scalar/direct" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 32]
    assert record["finite_output"] is True


def test_artifact_nax_e8p_expert_kblock_factor_reuse_rhs_native_raw_variant_times_prepared_native_call(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(artifact)
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="gate_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=32,
        seed=160,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_expert_kblock_factor_reuse_rhs_sorted_native_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["variant"] == "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_native_raw"
    assert record["diagnostic_phase"] == "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_native_raw"
    assert "expert/K-block factor-reuse RHS" in record["diagnostic_note"]
    assert "scalar/direct" in record["diagnostic_note"]
    assert "dense RHS is never materialized" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 32]
    assert record["finite_output"] is True


def test_artifact_nax_e8p_expert_kblock_factor_reuse_rhs_tensorops_raw_variant_times_prepared_native_call(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(artifact)
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="gate_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=32,
        seed=161,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["variant"] == "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_raw"
    assert record["diagnostic_phase"] == "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_raw"
    assert "expert/K-block factor-reuse RHS" in record["diagnostic_note"]
    assert "non-staged TensorOps" in record["diagnostic_note"]
    assert "dense RHS is never materialized" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 32]
    assert record["finite_output"] is True


def test_artifact_nax_e8p_component_stream_rhs_sorted_scalar_raw_variant_uses_air_down_shape(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(
        artifact,
        projection="down_proj",
        input_dims=1408,
        output_dims=65,
        group_size=352,
    )
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="down_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=128,
        seed=167,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_component_stream_rhs_sorted_scalar_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["projection"] == "down"
    assert record["artifact_projection"] == "down_proj"
    assert record["variant"] == "nax_e8p_component_stream_rhs_sorted_scalar_raw"
    assert record["diagnostic_phase"] == "nax_e8p_component_stream_rhs_sorted_scalar_raw"
    assert "component-stream RHS" in record["diagnostic_note"]
    assert "scalar/source oracle" in record["diagnostic_note"]
    assert "dense RHS is never materialized" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_group_size"] == 352
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 65]
    assert record["finite_output"] is True
    assert record["artifact_reference_variant"] == "decoded_fp16_sorted_direct"
    assert record["artifact_reference_cosine"] >= 0.999999
    assert record["artifact_reference_max_abs_diff"] < 6e-3


def test_artifact_nax_e8p_route_slot_codeword_stream_raw_variant_uses_air_down_shape(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(
        artifact,
        projection="down_proj",
        input_dims=1408,
        output_dims=65,
        group_size=352,
    )
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="down_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=128,
        seed=191,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_route_slot_codeword_stream_rhs_sorted_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["projection"] == "down"
    assert record["artifact_projection"] == "down_proj"
    assert record["variant"] == "nax_e8p_route_slot_codeword_stream_rhs_sorted_raw"
    assert record["diagnostic_phase"] == "nax_e8p_route_slot_codeword_stream_rhs_sorted_raw"
    assert "route-slot codeword-stream RHS" in record["diagnostic_note"]
    assert "same-window q2" in record["diagnostic_note"]
    assert "dense RHS is never materialized" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_group_size"] == 352
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 65]
    assert record["finite_output"] is True
    assert record["artifact_reference_variant"] == "decoded_fp16_sorted_direct"
    assert record["artifact_reference_cosine"] >= 0.999999
    assert record["artifact_reference_max_abs_diff"] < 6e-3


def test_artifact_nax_e8p_route_slot_mma_codeword_tile_raw_variant_uses_air_down_shape(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(
        artifact,
        projection="down_proj",
        input_dims=1408,
        output_dims=65,
        group_size=352,
    )
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="down_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=128,
        seed=192,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_route_slot_mma_codeword_tile_rhs_sorted_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["projection"] == "down"
    assert record["artifact_projection"] == "down_proj"
    assert record["variant"] == "nax_e8p_route_slot_mma_codeword_tile_rhs_sorted_raw"
    assert (
        record["diagnostic_phase"]
        == "nax_e8p_route_slot_mma_codeword_tile_rhs_sorted_raw"
    )
    assert "route-slot MMA codeword-tile RHS" in record["diagnostic_note"]
    assert "same-window q2" in record["diagnostic_note"]
    assert "dense RHS is never materialized" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_group_size"] == 352
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 65]
    assert record["finite_output"] is True
    assert record["artifact_reference_variant"] == "decoded_fp16_sorted_direct"
    assert record["artifact_reference_cosine"] >= 0.999999
    assert record["artifact_reference_max_abs_diff"] < 6e-3


def test_artifact_nax_e8p_token_cohort_mma_codeword_tile_raw_variant_uses_air_down_shape(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(
        artifact,
        projection="down_proj",
        input_dims=1408,
        output_dims=65,
        group_size=352,
    )
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="down_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=128,
        seed=192,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_token_cohort_mma_codeword_tile_rhs_sorted_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["projection"] == "down"
    assert record["artifact_projection"] == "down_proj"
    assert (
        record["variant"]
        == "nax_e8p_token_cohort_mma_codeword_tile_rhs_sorted_raw"
    )
    assert (
        record["diagnostic_phase"]
        == "nax_e8p_token_cohort_mma_codeword_tile_rhs_sorted_raw"
    )
    assert "token-cohort MMA codeword-tile RHS" in record["diagnostic_note"]
    assert "same-window q2" in record["diagnostic_note"]
    assert "dense RHS is never materialized" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_group_size"] == 352
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 65]
    assert record["finite_output"] is True
    assert record["artifact_reference_variant"] == "decoded_fp16_sorted_direct"
    assert record["artifact_reference_cosine"] >= 0.999999
    assert record["artifact_reference_max_abs_diff"] < 6e-3


def test_artifact_nax_e8p_output_stationary_codeword_tile_raw_variant_uses_air_down_shape(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(
        artifact,
        projection="down_proj",
        input_dims=1408,
        output_dims=65,
        group_size=352,
    )
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="down_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=128,
        seed=192,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_output_stationary_codeword_tile_rhs_sorted_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["projection"] == "down"
    assert record["artifact_projection"] == "down_proj"
    assert (
        record["variant"]
        == "nax_e8p_output_stationary_codeword_tile_rhs_sorted_raw"
    )
    assert (
        record["diagnostic_phase"]
        == "nax_e8p_output_stationary_codeword_tile_rhs_sorted_raw"
    )
    assert "output-stationary codeword-tile RHS" in record["diagnostic_note"]
    assert "same-window q2" in record["diagnostic_note"]
    assert "dense RHS is never materialized" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_group_size"] == 352
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 65]
    assert record["finite_output"] is True
    assert record["artifact_reference_variant"] == "decoded_fp16_sorted_direct"
    assert record["artifact_reference_cosine"] >= 0.999999
    assert record["artifact_reference_max_abs_diff"] < 6e-3


def test_artifact_nax_e8p_expert_kblock_codeword_factor_reuse_raw_variant_uses_air_down_shape(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(
        artifact,
        projection="down_proj",
        input_dims=1408,
        output_dims=65,
        group_size=352,
    )
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="down_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=128,
        seed=193,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_expert_kblock_codeword_factor_reuse_rhs_sorted_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["projection"] == "down"
    assert record["artifact_projection"] == "down_proj"
    assert record["variant"] == (
        "nax_e8p_expert_kblock_codeword_factor_reuse_rhs_sorted_raw"
    )
    assert record["diagnostic_phase"] == (
        "nax_e8p_expert_kblock_codeword_factor_reuse_rhs_sorted_raw"
    )
    assert "expert/K-block codeword factor-reuse RHS" in record["diagnostic_note"]
    assert "same-window q2" in record["diagnostic_note"]
    assert "dense RHS is never materialized" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_group_size"] == 352
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 65]
    assert record["finite_output"] is True
    assert record["artifact_reference_variant"] == "decoded_fp16_sorted_direct"
    assert record["artifact_reference_cosine"] >= 0.999999
    assert record["artifact_reference_max_abs_diff"] < 6e-3


def test_artifact_nax_e8p_expert_kblock_scale_slot_stream_raw_variant_uses_air_down_shape(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(
        artifact,
        projection="down_proj",
        input_dims=1408,
        output_dims=65,
        group_size=352,
    )
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="down_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=128,
        seed=197,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_expert_kblock_scale_slot_stream_rhs_sorted_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["projection"] == "down"
    assert record["artifact_projection"] == "down_proj"
    assert record["variant"] == (
        "nax_e8p_expert_kblock_scale_slot_stream_rhs_sorted_raw"
    )
    assert record["diagnostic_phase"] == (
        "nax_e8p_expert_kblock_scale_slot_stream_rhs_sorted_raw"
    )
    assert "expert/K-block scale-slot stream RHS" in record["diagnostic_note"]
    assert "same-window q2" in record["diagnostic_note"]
    assert "dense RHS is never materialized" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_group_size"] == 352
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 65]
    assert record["finite_output"] is True
    assert record["artifact_reference_variant"] == "decoded_fp16_sorted_direct"
    assert record["artifact_reference_cosine"] >= 0.999999
    assert record["artifact_reference_max_abs_diff"] < 6e-3


def test_artifact_nax_e8p_rowwise_codeword_tile_accumulate_raw_variant_uses_air_down_shape(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(
        artifact,
        projection="down_proj",
        input_dims=1408,
        output_dims=65,
        group_size=352,
    )
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="down_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=128,
        seed=194,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_rowwise_codeword_tile_accumulate_rhs_sorted_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["projection"] == "down"
    assert record["artifact_projection"] == "down_proj"
    assert record["variant"] == (
        "nax_e8p_rowwise_codeword_tile_accumulate_rhs_sorted_raw"
    )
    assert record["diagnostic_phase"] == (
        "nax_e8p_rowwise_codeword_tile_accumulate_rhs_sorted_raw"
    )
    assert "rowwise codeword-tile accumulate RHS" in record["diagnostic_note"]
    assert "same-window q2" in record["diagnostic_note"]
    assert "dense RHS is never materialized" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_group_size"] == 352
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 65]
    assert record["finite_output"] is True
    assert record["artifact_reference_variant"] == "decoded_fp16_sorted_direct"
    assert record["artifact_reference_cosine"] >= 0.999999
    assert record["artifact_reference_max_abs_diff"] < 6e-3


def test_artifact_nax_e8p_output_tile_local_codeword_lut_raw_variant_uses_air_down_shape(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(
        artifact,
        projection="down_proj",
        input_dims=1408,
        output_dims=65,
        group_size=352,
    )
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="down_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=128,
        seed=194,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_output_tile_local_codeword_lut_rhs_sorted_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["projection"] == "down"
    assert record["artifact_projection"] == "down_proj"
    assert record["variant"] == (
        "nax_e8p_output_tile_local_codeword_lut_rhs_sorted_raw"
    )
    assert record["diagnostic_phase"] == (
        "nax_e8p_output_tile_local_codeword_lut_rhs_sorted_raw"
    )
    assert "output-tile-local codeword LUT RHS" in record["diagnostic_note"]
    assert "same-window q2" in record["diagnostic_note"]
    assert "dense RHS is never materialized" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_group_size"] == 352
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 65]
    assert record["finite_output"] is True
    assert record["artifact_reference_variant"] == "decoded_fp16_sorted_direct"
    assert record["artifact_reference_cosine"] >= 0.999999
    assert record["artifact_reference_max_abs_diff"] < 6e-3


def test_artifact_nax_e8p_route_microtile_codeword_block_reduce_raw_variant_uses_air_down_shape(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(
        artifact,
        projection="down_proj",
        input_dims=1408,
        output_dims=65,
        group_size=352,
    )
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="down_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=128,
        seed=194,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_route_microtile_codeword_block_reduce_rhs_sorted_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["projection"] == "down"
    assert record["artifact_projection"] == "down_proj"
    assert record["variant"] == (
        "nax_e8p_route_microtile_codeword_block_reduce_rhs_sorted_raw"
    )
    assert record["diagnostic_phase"] == (
        "nax_e8p_route_microtile_codeword_block_reduce_rhs_sorted_raw"
    )
    assert "route-microtile codeword block-reduce RHS" in record["diagnostic_note"]
    assert "same-window q2" in record["diagnostic_note"]
    assert "dense RHS is never materialized" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_group_size"] == 352
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 65]
    assert record["finite_output"] is True
    assert record["artifact_reference_variant"] == "decoded_fp16_sorted_direct"
    assert record["artifact_reference_cosine"] >= 0.999999
    assert record["artifact_reference_max_abs_diff"] < 6e-3


def test_artifact_nax_e8p_active_route_tile_raw_variant_uses_air_down_shape(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(
        artifact,
        projection="down_proj",
        input_dims=1408,
        output_dims=65,
        group_size=352,
    )
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="down_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=128,
        seed=193,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_active_route_tile_codeword_outer_product_rhs_sorted_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["projection"] == "down"
    assert record["artifact_projection"] == "down_proj"
    assert (
        record["variant"]
        == "nax_e8p_active_route_tile_codeword_outer_product_rhs_sorted_raw"
    )
    assert (
        record["diagnostic_phase"]
        == "nax_e8p_active_route_tile_codeword_outer_product_rhs_sorted_raw"
    )
    assert "active-route-tile codeword outer-product RHS" in record["diagnostic_note"]
    assert "same-window q2" in record["diagnostic_note"]
    assert "dense RHS is never materialized" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_group_size"] == 352
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 65]
    assert record["finite_output"] is True
    assert record["artifact_reference_variant"] == "decoded_fp16_sorted_direct"
    assert record["artifact_reference_cosine"] >= 0.999999
    assert record["artifact_reference_max_abs_diff"] < 6e-3


def test_artifact_nax_e8p_route_batch_segmented_raw_variant_uses_air_down_shape(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(
        artifact,
        projection="down_proj",
        input_dims=1408,
        output_dims=65,
        group_size=352,
    )
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="down_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=128,
        seed=194,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_route_batch_segmented_codeword_reduce_rhs_sorted_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["projection"] == "down"
    assert record["artifact_projection"] == "down_proj"
    assert (
        record["variant"]
        == "nax_e8p_route_batch_segmented_codeword_reduce_rhs_sorted_raw"
    )
    assert (
        record["diagnostic_phase"]
        == "nax_e8p_route_batch_segmented_codeword_reduce_rhs_sorted_raw"
    )
    assert "route-batch segmented codeword-reduce RHS" in record["diagnostic_note"]
    assert "same-window q2" in record["diagnostic_note"]
    assert "dense RHS is never materialized" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_group_size"] == 352
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 65]
    assert record["finite_output"] is True
    assert record["artifact_reference_variant"] == "decoded_fp16_sorted_direct"
    assert record["artifact_reference_cosine"] >= 0.999999
    assert record["artifact_reference_max_abs_diff"] < 6e-3


def test_component_stream_partial_compacts_empty_route_tiles() -> None:
    fixture = build_projection_fixture(
        projection="down",
        tokens=6,
        top_k=2,
        experts=128,
        input_dims=1408,
        output_dims=65,
        mlx_group_size=128,
        seed=173,
    )
    sorted_inputs = _prepare_sorted_steel_inputs(fixture)

    compacted = _compact_nonempty_sorted_tile_descriptors(sorted_inputs)

    assert compacted.sorted_x is sorted_inputs.sorted_x
    assert compacted.tile_descriptors[0].shape[0] < sorted_inputs.tile_descriptors[0].shape[0]
    assert compacted.tile_descriptors[0].shape == compacted.tile_descriptors[1].shape
    assert compacted.tile_descriptors[1].shape == compacted.tile_descriptors[2].shape
    assert int(mx.sum(compacted.tile_descriptors[2]).item()) == fixture.tokens * fixture.top_k
    assert bool(mx.all(compacted.tile_descriptors[2] > 0).item())


def test_artifact_nax_e8p_component_stream_rhs_sorted_partial_raw_variant_uses_air_down_shape(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(
        artifact,
        projection="down_proj",
        input_dims=1408,
        output_dims=65,
        group_size=352,
    )
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="down_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=128,
        seed=173,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_component_stream_rhs_sorted_partial_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["projection"] == "down"
    assert record["artifact_projection"] == "down_proj"
    assert record["variant"] == "nax_e8p_component_stream_rhs_sorted_partial_raw"
    assert record["diagnostic_phase"] == "nax_e8p_component_stream_rhs_sorted_partial_raw"
    assert "component-stream RHS" in record["diagnostic_note"]
    assert "partial-reduction scaffold" in record["diagnostic_note"]
    assert "speed path" not in record["diagnostic_note"]
    assert "dense RHS is never materialized" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_group_size"] == 352
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 65]
    assert record["finite_output"] is True
    assert record["artifact_reference_variant"] == "decoded_fp16_sorted_direct"
    assert record["artifact_reference_cosine"] >= 0.999999
    assert record["artifact_reference_max_abs_diff"] < 6e-3


def test_artifact_nax_e8p_component_stream_rhs_sorted_tensorops_raw_variant_uses_air_down_shape(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(
        artifact,
        projection="down_proj",
        input_dims=1408,
        output_dims=65,
        group_size=352,
    )
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="down_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=128,
        seed=179,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_component_stream_rhs_sorted_tensorops_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["projection"] == "down"
    assert record["artifact_projection"] == "down_proj"
    assert record["variant"] == "nax_e8p_component_stream_rhs_sorted_tensorops_raw"
    assert record["diagnostic_phase"] == "nax_e8p_component_stream_rhs_sorted_tensorops_raw"
    assert "component-stream RHS" in record["diagnostic_note"]
    assert "TensorOps parity candidate" in record["diagnostic_note"]
    assert "same-window q2" in record["diagnostic_note"]
    assert "dense RHS is never materialized" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_group_size"] == 352
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 65]
    assert record["finite_output"] is True
    assert record["artifact_reference_variant"] == "decoded_fp16_sorted_direct"
    assert record["artifact_reference_cosine"] >= 0.999999
    assert record["artifact_reference_max_abs_diff"] < 6e-3


def test_artifact_nax_e8p_component_stream_rhs_sorted_shared_decode_raw_variant_uses_air_down_shape(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(
        artifact,
        projection="down_proj",
        input_dims=1408,
        output_dims=65,
        group_size=352,
    )
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="down_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=128,
        seed=183,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_component_stream_rhs_sorted_shared_decode_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["projection"] == "down"
    assert record["artifact_projection"] == "down_proj"
    assert record["variant"] == "nax_e8p_component_stream_rhs_sorted_shared_decode_raw"
    assert record["diagnostic_phase"] == "nax_e8p_component_stream_rhs_sorted_shared_decode_raw"
    assert "component-stream RHS" in record["diagnostic_note"]
    assert "shared-decode" in record["diagnostic_note"]
    assert "same-window q2" in record["diagnostic_note"]
    assert "dense RHS is never materialized" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_group_size"] == 352
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 65]
    assert record["finite_output"] is True
    assert record["artifact_reference_variant"] == "decoded_fp16_sorted_direct"
    assert record["artifact_reference_cosine"] >= 0.999999
    assert record["artifact_reference_max_abs_diff"] < 6e-3


def test_artifact_component_stream_tensorops_raw_can_skip_reference_for_speed_rows(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(
        artifact,
        projection="down_proj",
        input_dims=1408,
        output_dims=65,
        group_size=352,
    )
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="down_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=128,
        seed=181,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_component_stream_rhs_sorted_tensorops_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
        artifact_reference_check=False,
    )

    assert record["variant"] == "nax_e8p_component_stream_rhs_sorted_tensorops_raw"
    assert record["finite_output"] is True
    assert record["artifact_reference_check"] is False
    assert "artifact_reference_variant" not in record
    assert "artifact_reference_cosine" not in record
    assert "artifact_reference_max_abs_diff" not in record


def test_artifact_nax_e8p_split_byte_factor_reuse_rhs_shared_decode_raw_variant_times_prepared_native_call(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(artifact)
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="gate_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=32,
        seed=154,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["variant"] == "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_raw"
    assert record["diagnostic_phase"] == "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_raw"
    assert "factor-reuse RHS" in record["diagnostic_note"]
    assert "shared-decode" in record["diagnostic_note"]
    assert "non-staged TensorOps" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 32]
    assert record["finite_output"] is True


def test_artifact_nax_e8p_split_byte_factor_reuse_rhs_shared_n_decode_raw_variant_times_prepared_native_call(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(artifact)
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="gate_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=32,
        seed=155,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["variant"] == "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_raw"
    assert record["diagnostic_phase"] == "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_raw"
    assert "factor-reuse RHS" in record["diagnostic_note"]
    assert "shared-n-decode" in record["diagnostic_note"]
    assert "lane-local" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 32]
    assert record["finite_output"] is True


def test_artifact_nax_e8p_sign_nibble_rhs_tensorops_raw_variant_times_prepared_native_call(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(artifact)
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="gate_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=32,
        seed=156,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["variant"] == "nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_raw"
    assert record["diagnostic_phase"] == "nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_raw"
    assert "sign-nibble RHS" in record["diagnostic_note"]
    assert "non-staged TensorOps" in record["diagnostic_note"]
    assert "dense RHS is never materialized" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 32]
    assert record["finite_output"] is True


def test_artifact_nax_e8p_sign_plane_rhs_tensorops_raw_variant_times_prepared_native_call(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(artifact)
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="gate_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=32,
        seed=159,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["variant"] == "nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_raw"
    assert record["diagnostic_phase"] == "nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_raw"
    assert "sign-plane RHS" in record["diagnostic_note"]
    assert "non-staged TensorOps" in record["diagnostic_note"]
    assert "dense RHS is never materialized" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 32]
    assert record["finite_output"] is True


def test_artifact_nax_e8p_sign_plane_rhs_sorted_native_raw_variant_times_prepared_native_call(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(artifact)
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="gate_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=32,
        seed=160,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_sign_plane_abs_index_rhs_sorted_native_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["variant"] == "nax_e8p_sign_plane_abs_index_rhs_sorted_native_raw"
    assert record["diagnostic_phase"] == "nax_e8p_sign_plane_abs_index_rhs_sorted_native_raw"
    assert "sign-plane RHS" in record["diagnostic_note"]
    assert "scalar/direct" in record["diagnostic_note"]
    assert "dense RHS is never materialized" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 32]
    assert record["finite_output"] is True


def test_artifact_nax_e8p_sign_nibble_micro_lut_rhs_tensorops_raw_variant_times_prepared_native_call(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(artifact)
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="gate_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=32,
        seed=157,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["variant"] == "nax_e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_raw"
    assert record["diagnostic_phase"] == "nax_e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_raw"
    assert "sign-nibble micro-LUT RHS" in record["diagnostic_note"]
    assert "non-staged TensorOps" in record["diagnostic_note"]
    assert "dense RHS is never materialized" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 32]
    assert record["finite_output"] is True


def test_artifact_nax_e8p_sign_nibble_micro_lut_rhs_sorted_native_raw_variant_times_prepared_native_call(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(artifact)
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="gate_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=32,
        seed=158,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_sign_nibble_micro_lut_rhs_sorted_native_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["variant"] == "nax_e8p_sign_nibble_micro_lut_rhs_sorted_native_raw"
    assert record["diagnostic_phase"] == "nax_e8p_sign_nibble_micro_lut_rhs_sorted_native_raw"
    assert "sign-nibble micro-LUT RHS" in record["diagnostic_note"]
    assert "scalar/direct" in record["diagnostic_note"]
    assert "dense RHS is never materialized" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 32]
    assert record["finite_output"] is True


def test_artifact_nax_e8p_packed_rhs_tiled_m128_raw_variant_times_prepared_native_call(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(artifact)
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="gate_proj",
        tokens=70,
        top_k=2,
        mlx_group_size=32,
        seed=149,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_packed_rhs_sorted_tiled_m128_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["variant"] == "nax_e8p_packed_rhs_sorted_tiled_m128_raw"
    assert record["diagnostic_phase"] == "nax_e8p_packed_rhs_sorted_tiled_m128_raw"
    assert "128-route" in record["diagnostic_note"]
    assert "packed RHS" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [140, 32]
    assert record["finite_output"] is True


def test_artifact_nax_e8p_packed_rhs_tiled_k128_raw_variant_times_prepared_native_call(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(artifact)
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="gate_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=32,
        seed=150,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_packed_rhs_sorted_tiled_k128_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["variant"] == "nax_e8p_packed_rhs_sorted_tiled_k128_raw"
    assert record["diagnostic_phase"] == "nax_e8p_packed_rhs_sorted_tiled_k128_raw"
    assert "two bk64" in record["diagnostic_note"]
    assert "packed RHS" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 32]
    assert record["finite_output"] is True


def test_artifact_nax_e8p_predecoded_fp16_raw_variant_times_prepared_upper_bound(
    tmp_path,
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(artifact)
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="gate_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=32,
        seed=147,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_predecoded_fp16_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["variant"] == "nax_e8p_predecoded_fp16_raw"
    assert record["diagnostic_phase"] == "nax_e8p_predecoded_fp16_raw"
    assert "predecoded FP16 upper-bound" in record["diagnostic_note"]
    assert record["diagnostic_predecoded_dense_weight_bytes"] == 4 * 32 * 64 * 2
    assert record["mlx_sorted_indices"] is True
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 32]
    assert record["finite_output"] is True


def test_artifact_nax_e8p_gs352_raw_variant_uses_air_down_shape(tmp_path) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(
        artifact,
        projection="down_proj",
        input_dims=1408,
        output_dims=65,
        group_size=352,
    )
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact,
        layer_index=45,
        artifact_projection="down_proj",
        tokens=6,
        top_k=2,
        mlx_group_size=128,
        seed=144,
    )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant="nax_e8p_fp16_sorted_steel_gs352_raw",
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )

    assert record["fixture_source"] == "vq_artifact"
    assert record["projection"] == "down"
    assert record["artifact_projection"] == "down_proj"
    assert record["variant"] == "nax_e8p_fp16_sorted_steel_gs352_raw"
    assert record["diagnostic_phase"] == "nax_e8p_fp16_sorted_steel_gs352_raw"
    assert "group_size=352" in record["diagnostic_note"]
    assert record["mlx_sorted_indices"] is True
    assert record["vq_group_size"] == 352
    assert record["vq_code_bits"] == 16
    assert record["output_shape"] == [12, 65]
    assert record["finite_output"] is True


def test_projection_variants_produce_finite_outputs_and_metric_schema() -> None:
    fixture = build_projection_fixture(
        projection="down",
        tokens=2,
        top_k=2,
        experts=4,
        input_dims=64,
        output_dims=32,
        mlx_group_size=32,
        vq_preferred_group_size=64,
        seed=456,
    )

    records = [
        run_projection_variant(fixture, variant=variant, iterations=1, warmup=1)
        for variant in ("dense_bf16", "mlx_q2", "vq_e1")
    ]

    assert [record["variant"] for record in records] == ["dense_bf16", "mlx_q2", "vq_e1"]
    for record in records:
        assert record["projection"] == "down"
        assert record["output_shape"] == [2, 2, 32]
        assert record["finite_output"] is True
        assert record["iterations"] == 1
        assert record["warmup"] == 1
        assert record["ms_per_iter"] >= 0
        assert record["mlx_peak_bytes"] is not None
        assert json.dumps(record, sort_keys=True)


def test_mlx_q2_projection_variant_uses_resident_sorted_route_flow() -> None:
    fixture = build_projection_fixture(
        projection="gate_up",
        tokens=32,
        top_k=2,
        experts=4,
        input_dims=64,
        output_dims=32,
        mlx_group_size=32,
        vq_preferred_group_size=64,
        seed=457,
    )
    dense = _dense_switch_linear(fixture)
    layer = dense.to_quantized(group_size=fixture.mlx_group_size, bits=2, mode="affine")
    sorted_output = _call_mlx_q2_routed(layer, fixture)
    unsorted_output = _call_dense_or_mlx(layer, fixture)
    mx.eval(sorted_output, unsorted_output)

    assert sorted_output.shape == (32, 2, 32)
    sorted_np = np.array(sorted_output.astype(mx.float32))
    unsorted_np = np.array(unsorted_output.astype(mx.float32))
    dot = float(np.sum(sorted_np * unsorted_np))
    norm = float(np.sqrt(np.sum(sorted_np * sorted_np) * np.sum(unsorted_np * unsorted_np)))
    assert float(np.max(np.abs(sorted_np - unsorted_np))) < 5e-3
    assert dot / norm >= 0.99995

    record = run_projection_variant(fixture, variant="mlx_q2", iterations=1, warmup=0)
    assert record["mlx_sorted_indices"] is True


def test_mlx_gather_mm_bf16_variant_uses_sorted_route_flow() -> None:
    fixture = build_projection_fixture(
        projection="gate_up",
        tokens=32,
        top_k=2,
        experts=4,
        input_dims=64,
        output_dims=32,
        mlx_group_size=32,
        vq_preferred_group_size=64,
        seed=458,
    )
    dense = _dense_switch_linear(fixture)
    sorted_output = _call_mlx_gather_mm_bf16_routed(dense.weight, fixture)
    dense_output = _call_dense_or_mlx(dense, fixture)
    mx.eval(sorted_output, dense_output)

    assert sorted_output.shape == (32, 2, 32)
    sorted_np = np.array(sorted_output.astype(mx.float32))
    dense_np = np.array(dense_output.astype(mx.float32))
    dot = float(np.sum(sorted_np * dense_np))
    norm = float(np.sqrt(np.sum(sorted_np * sorted_np) * np.sum(dense_np * dense_np)))
    assert dot / norm >= 0.99995

    record = run_projection_variant(fixture, variant="mlx_gather_mm_bf16", iterations=1, warmup=0)
    assert record["variant"] == "mlx_gather_mm_bf16"
    assert record["mlx_sorted_indices"] is True
    assert record["output_shape"] == [32, 2, 32]


def test_nax_predecoded_fp16_variant_uses_native_sorted_route_flow() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    fixture = build_projection_fixture(
        projection="gate_up",
        tokens=6,
        top_k=2,
        experts=4,
        input_dims=64,
        output_dims=32,
        mlx_group_size=32,
        vq_preferred_group_size=64,
        seed=4581,
    )
    dense = _dense_switch_linear(fixture)
    weight_t = mx.swapaxes(dense.weight.astype(mx.float16), -1, -2)
    observed = _call_nax_predecoded_fp16_routed(weight_t, fixture)
    expected = _call_dense_or_mlx(dense, fixture)
    mx.eval(observed, expected)

    assert observed.shape == (6, 2, 32)
    observed_np = np.array(observed.astype(mx.float32))
    expected_np = np.array(expected.astype(mx.float32))
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert dot / norm >= 0.9999

    record = run_projection_variant(fixture, variant="nax_predecoded_fp16", iterations=1, warmup=0)
    assert record["variant"] == "nax_predecoded_fp16"
    assert record["diagnostic_phase"] == "nax_predecoded_fp16"
    assert record["mlx_sorted_indices"] is True
    assert record["output_shape"] == [6, 2, 32]


def test_nax_e8_fp16_variant_uses_native_sorted_route_blocks() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    fixture = build_projection_fixture(
        projection="gate_up",
        tokens=6,
        top_k=2,
        experts=4,
        input_dims=64,
        output_dims=32,
        mlx_group_size=32,
        vq_preferred_group_size=64,
        seed=4582,
    )
    layer = _vq_switch_linear(fixture)
    observed = _call_nax_e8_fp16_routed(layer, fixture)
    expected = layer(fixture.x, fixture.indices)
    mx.eval(observed, expected)

    assert observed.shape == (6, 2, 32)
    observed_np = np.array(observed.astype(mx.float32))
    expected_np = np.array(expected.astype(mx.float32))
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert dot / norm >= 0.9999

    record = run_projection_variant(fixture, variant="nax_e8_fp16", iterations=1, warmup=0)
    assert record["variant"] == "nax_e8_fp16"
    assert record["diagnostic_phase"] == "nax_e8_fp16"
    assert record["mlx_sorted_indices"] is True
    assert record["output_shape"] == [6, 2, 32]


def test_nax_e8_fp16_steel_variant_uses_native_sorted_route_blocks() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    fixture = build_projection_fixture(
        projection="gate_up",
        tokens=6,
        top_k=2,
        experts=4,
        input_dims=64,
        output_dims=32,
        mlx_group_size=32,
        vq_preferred_group_size=64,
        seed=4592,
    )
    layer = _vq_switch_linear(fixture)
    observed = _call_nax_e8_fp16_steel_routed(layer, fixture)
    expected = layer(fixture.x, fixture.indices)
    mx.eval(observed, expected)

    assert observed.shape == (6, 2, 32)
    observed_np = np.array(observed.astype(mx.float32))
    expected_np = np.array(expected.astype(mx.float32))
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert dot / norm >= 0.9999

    record = run_projection_variant(fixture, variant="nax_e8_fp16_steel", iterations=1, warmup=0)
    assert record["variant"] == "nax_e8_fp16_steel"
    assert record["diagnostic_phase"] == "nax_e8_fp16_steel"
    assert record["mlx_sorted_indices"] is True
    assert record["vq_group_size"] == 64
    assert record["output_shape"] == [6, 2, 32]


def test_nax_e8_fp16_sorted_steel_variant_uses_contiguous_sorted_activations() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    fixture = build_projection_fixture(
        projection="gate_up",
        tokens=6,
        top_k=2,
        experts=4,
        input_dims=64,
        output_dims=32,
        mlx_group_size=32,
        vq_preferred_group_size=64,
        seed=4602,
    )
    layer = _vq_switch_linear(fixture)
    observed = _call_nax_e8_fp16_sorted_steel_routed(layer, fixture)
    expected = layer(fixture.x, fixture.indices)
    mx.eval(observed, expected)

    assert observed.shape == (6, 2, 32)
    observed_np = np.array(observed.astype(mx.float32))
    expected_np = np.array(expected.astype(mx.float32))
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert dot / norm >= 0.9999

    record = run_projection_variant(
        fixture,
        variant="nax_e8_fp16_sorted_steel",
        iterations=1,
        warmup=0,
    )
    assert record["variant"] == "nax_e8_fp16_sorted_steel"
    assert record["diagnostic_phase"] == "nax_e8_fp16_sorted_steel"
    assert record["mlx_sorted_indices"] is True
    assert record["vq_group_size"] == 64
    assert record["output_shape"] == [6, 2, 32]


def test_nax_e8p_fp16_sorted_steel_variant_uses_16bit_contiguous_sorted_activations() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    fixture = build_projection_fixture(
        projection="gate_up",
        tokens=6,
        top_k=2,
        experts=4,
        input_dims=64,
        output_dims=32,
        mlx_group_size=32,
        vq_preferred_group_size=64,
        vq_code_bits=16,
        seed=4603,
    )
    layer = _vq_switch_linear(fixture)
    observed = _call_nax_e8p_fp16_sorted_steel_routed(layer, fixture)
    expected = layer(fixture.x, fixture.indices)
    mx.eval(observed, expected)

    assert observed.shape == (6, 2, 32)
    observed_np = np.array(observed.astype(mx.float32))
    expected_np = np.array(expected.astype(mx.float32))
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert dot / norm >= 0.9999

    record = run_projection_variant(
        fixture,
        variant="nax_e8p_fp16_sorted_steel",
        iterations=1,
        warmup=0,
    )
    assert record["variant"] == "nax_e8p_fp16_sorted_steel"
    assert record["diagnostic_phase"] == "nax_e8p_fp16_sorted_steel"
    assert record["mlx_sorted_indices"] is True
    assert record["vq_code_bits"] == 16
    assert record["vq_group_size"] == 64
    assert record["output_shape"] == [6, 2, 32]


def test_nax_e8p_fp16_sorted_steel_routed_variant_uses_switch_selector(
    monkeypatch,
) -> None:
    fixture = build_projection_fixture(
        projection="down",
        tokens=4,
        top_k=2,
        experts=4,
        input_dims=64,
        output_dims=128,
        mlx_group_size=32,
        vq_preferred_group_size=64,
        vq_code_bits=16,
        seed=4604,
    )
    layer = _vq_switch_linear(fixture)
    observed: dict[str, object] = {}

    def fake_gather(
        x,
        codes,
        scales,
        codebook,
        sorted_rhs,
        sorted_lhs,
        *,
        input_dims: int,
        output_dims: int,
        group_size: int,
        code_bits: int,
        implementation: str,
        projection: str,
        **kwargs,
    ):
        del x, codes, scales, codebook, sorted_lhs, kwargs
        observed.update(
            {
                "input_dims": input_dims,
                "output_dims": output_dims,
                "group_size": group_size,
                "code_bits": code_bits,
                "implementation": implementation,
                "projection": projection,
            }
        )
        return mx.zeros((sorted_rhs.shape[0], output_dims), dtype=mx.float16)

    def fail_direct_native(*args, **kwargs):
        del args, kwargs
        raise AssertionError("routed benchmark bypassed gather_vqmm_sorted_routes")

    monkeypatch.setattr(
        projection_kernels.vq_switch,
        "gather_vqmm_sorted_routes",
        fake_gather,
    )
    monkeypatch.setattr(
        projection_kernels.nax,
        "nax_e8p_fp16_sorted_steel_matmul",
        fail_direct_native,
    )

    out = _call_nax_e8p_fp16_sorted_steel_routed(layer, fixture)
    mx.eval(out)

    assert out.shape == (4, 2, 128)
    assert observed == {
        "input_dims": 64,
        "output_dims": 128,
        "group_size": 64,
        "code_bits": 16,
        "implementation": "nax_e8p",
        "projection": "down",
    }


def test_nax_e8_int8_variant_uses_native_sorted_route_blocks() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    fixture = build_projection_fixture(
        projection="gate_up",
        tokens=6,
        top_k=2,
        experts=4,
        input_dims=64,
        output_dims=32,
        mlx_group_size=32,
        vq_preferred_group_size=64,
        seed=4583,
    )
    layer = _vq_switch_linear(fixture)
    observed = _call_nax_e8_int8_routed(layer, fixture)
    expected = layer(fixture.x, fixture.indices)
    mx.eval(observed, expected)

    assert observed.shape == (6, 2, 32)
    observed_np = np.array(observed.astype(mx.float32))
    expected_np = np.array(expected.astype(mx.float32))
    assert np.all(np.isfinite(observed_np))
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    assert dot / norm >= 0.999

    record = run_projection_variant(fixture, variant="nax_e8_int8", iterations=1, warmup=0)
    assert record["variant"] == "nax_e8_int8"
    assert record["diagnostic_phase"] == "nax_e8_int8"
    assert record["mlx_sorted_indices"] is True
    assert record["output_shape"] == [6, 2, 32]


def test_nax_int8_raw_ceiling_probe_reports_gate_schema() -> None:
    record = run_nax_int8_raw_ceiling(iterations=1, warmup=0, seed=4584)

    assert record["benchmark"] == "nax_int8_raw_ceiling"
    if not record["available"]:
        assert "native VQ NAX extension" in record["reason"]
        return
    assert record["tile_geometry"] == "64x64x64"
    assert record["fp16_ms_per_iter"] >= 0
    assert record["int8_prequantized_ms_per_iter"] >= 0
    assert record["raw_ceiling_gate"] in {"pass", "stop"}
    assert record["finite_fp16"] is True
    assert record["finite_int8"] is True


def test_vq_decode_to_scratch_gather_mm_variant_matches_vq_output() -> None:
    fixture = build_projection_fixture(
        projection="gate_up",
        tokens=6,
        top_k=2,
        experts=4,
        input_dims=64,
        output_dims=32,
        mlx_group_size=32,
        vq_preferred_group_size=64,
        seed=459,
    )
    layer = _vq_switch_linear(fixture)
    scratch_output = _call_vq_decode_to_scratch_gather_mm_routed(layer, fixture)
    vq_output = layer(fixture.x, fixture.indices)
    mx.eval(scratch_output, vq_output)

    assert scratch_output.shape == (6, 2, 32)
    scratch_np = np.array(scratch_output.astype(mx.float32))
    vq_np = np.array(vq_output.astype(mx.float32))
    dot = float(np.sum(scratch_np * vq_np))
    norm = float(np.sqrt(np.sum(scratch_np * scratch_np) * np.sum(vq_np * vq_np)))
    assert dot / norm >= 0.99999

    record = run_projection_variant(
        fixture,
        variant="vq_decode_to_scratch_gather_mm",
        iterations=1,
        warmup=0,
    )
    assert record["variant"] == "vq_decode_to_scratch_gather_mm"
    assert record["diagnostic_phase"] == "decode_to_scratch_gather_mm"
    assert record["output_shape"] == [6, 2, 32]


def test_decomposed_vq_projection_variants_emit_current_and_diagnostics() -> None:
    assert projection_variants_for_cli("vq_e1", decompose_vq=True) == [
        "vq_current",
        "vq_decode_only",
        "vq_matmul_only",
        "vq_decode_direct_candidates",
        "vq_decode_to_scratch_gather_mm",
    ]
    assert projection_variants_for_cli("all", decompose_vq=True) == [
        "dense_bf16",
        "mlx_q2",
        "vq_current",
        "vq_decode_only",
        "vq_matmul_only",
        "vq_decode_direct_candidates",
        "vq_decode_to_scratch_gather_mm",
    ]
    assert projection_variants_for_cli("m1_decode_gate", decompose_vq=False) == [
        "mlx_q2",
        "vq_decode_direct_candidates",
    ]
    assert projection_variants_for_cli("nax_audit", decompose_vq=False) == [
        "mlx_q2",
        "mlx_gather_mm_bf16",
        "vq_e1",
    ]


def test_decomposed_vq_projection_variants_run_on_small_fixture() -> None:
    fixture = build_projection_fixture(
        projection="gate_up",
        tokens=1,
        top_k=2,
        experts=4,
        input_dims=64,
        output_dims=32,
        mlx_group_size=32,
        vq_preferred_group_size=64,
        seed=789,
    )

    records = [
        run_projection_variant(fixture, variant=variant, iterations=1, warmup=0)
        for variant in (
            "vq_current",
            "vq_decode_only",
            "vq_matmul_only",
            "vq_decode_direct_candidates",
        )
    ]

    assert [record["variant"] for record in records] == [
        "vq_current",
        "vq_decode_only",
        "vq_matmul_only",
        "vq_decode_direct_candidates",
    ]
    for record in records:
        assert record["finite_output"] is True
        assert record["checksum"] is not None
        assert record["vq_group_size"] == 64
    diagnostics = {record["variant"]: record for record in records}
    assert diagnostics["vq_decode_only"]["diagnostic_phase"] == "decode_only"
    assert "not additive" in diagnostics["vq_decode_only"]["diagnostic_note"]
    assert diagnostics["vq_matmul_only"]["diagnostic_phase"] == "matmul_only"
    assert "pseudo weights" in diagnostics["vq_matmul_only"]["diagnostic_note"]
    assert diagnostics["vq_decode_direct_candidates"]["diagnostic_phase"] == "decode_direct_candidates"
    assert "decoded threadgroup codebook" in diagnostics["vq_decode_direct_candidates"]["diagnostic_note"]


def test_decode_bandwidth_floor_report_uses_m1_read_bytes_and_measured_bandwidth() -> None:
    report = decode_bandwidth_floor_report(
        input_dims=4096,
        output_dims=1408,
        top_k=8,
        code_bits=8,
        group_size=512,
        measured_gb_s=400.0,
    )

    expected_bytes = 8 * (1408 * (4096 // 8) + 1408 * (4096 // 512) * 2)
    assert report["read_bytes_m1"] == expected_bytes
    assert report["bandwidth_floor_ms"] == expected_bytes / (400.0 * 1_000_000_000) * 1000.0


def test_diagnostic_metal_source_is_packaged_with_kernel_sources() -> None:
    path = Path(__file__).resolve().parents[1] / "src" / "mlx_vq" / "kernels" / "gather_vqmm_diagnostics.metal"

    assert path.exists()
    assert "DIAGNOSTIC_MODE" in path.read_text()
