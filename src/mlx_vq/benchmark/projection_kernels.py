from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mlx.core as mx
import numpy as np

from mlx_lm.models.switch_layers import SwitchLinear, _gather_sort, _scatter_unsort
import mlx_vq.ops.vq_switch as vq_switch

from mlx_vq.benchmark.metrics import (
    collect_metric_snapshot,
    collect_vm_stat_counts,
    reset_mlx_peak_memory,
)
from mlx_vq.convert.stream_convert import effective_group_size
from mlx_vq.codebook.e8 import (
    decode_e8p,
    decode_weight_matrix,
    e8_1bit_packed,
    e8p_packed_abs_grid,
)
from mlx_vq.kernels.gather_vqmm import (
    gather_vqmm_m1_kernel_unchecked,
    gather_vqmm_m1_rowpair_kernel_unchecked,
    m1_rows_per_threadgroup,
    m1_use_decoded_codebook,
    m1_use_threadgroup_codebook,
)
from mlx_vq.kernels import nax
from mlx_vq.kernels.e8p_rhs_layout import (
    pack_e8p_component_stream_rhs_tiles,
    pack_e8p_expert_kblock_factor_reuse_rhs_tiles,
    pack_e8p_rhs_tiles,
    pack_e8p_sign_plane_abs_index_rhs_tiles,
    pack_e8p_sign_nibble_abs_index_rhs_tiles,
    pack_e8p_sign_nibble_micro_lut_rhs_tiles,
    pack_e8p_split_byte_factor_reuse_rhs_tiles,
    pack_e8p_split_byte_rhs_tiles,
)
from mlx_vq.kernels.vq_qmv import _qmv_output_dtype
from mlx_vq.io.load import load_quantized_vq_switch_linear
from mlx_vq.nn.switch_linear import QuantizedVQSwitchLinear

ProjectionKind = Literal["gate_up", "down"]
ProjectionVariant = Literal[
    "dense_bf16",
    "mlx_gather_mm_bf16",
    "mlx_q2",
    "nax_e8_fp16",
    "nax_e8_fp16_sorted_steel",
    "nax_e8_fp16_sorted_steel_raw",
    "nax_e8p_fp16_sorted_steel",
    "nax_e8p_fp16_sorted_steel_raw",
    "nax_e8p_fp16_sorted_direct_reduce_raw",
    "nax_e8p_fp16_sorted_inline_b_raw",
    "nax_e8p_packed_rhs_sorted_tiled_raw",
    "nax_e8p_route_slot_codeword_stream_rhs_sorted_raw",
    "nax_e8p_route_slot_mma_codeword_tile_rhs_sorted_raw",
    "nax_e8p_active_route_tile_codeword_outer_product_rhs_sorted_raw",
    "nax_e8p_expert_cohort_codeword_broadcast_rhs_sorted_raw",
    "nax_e8p_route_batch_segmented_codeword_reduce_rhs_sorted_raw",
    "nax_e8p_token_cohort_codeword_stream_rhs_sorted_raw",
    "nax_e8p_token_cohort_mma_codeword_tile_rhs_sorted_raw",
    "nax_e8p_output_stationary_codeword_tile_rhs_sorted_raw",
    "nax_e8p_input_stationary_codeword_tile_rhs_sorted_raw",
    "nax_e8p_expert_kblock_codeword_factor_reuse_rhs_sorted_raw",
    "nax_e8p_route_codeword_lut_accumulate_rhs_sorted_raw",
    "nax_e8p_rowwise_codeword_tile_accumulate_rhs_sorted_raw",
    "nax_e8p_output_tile_local_codeword_lut_rhs_sorted_raw",
    "nax_e8p_route_microtile_codeword_block_reduce_rhs_sorted_raw",
    "nax_e8p_kblock_wavefront_codeword_scan_rhs_sorted_raw",
    "nax_e8p_token_route_output_stripe_pipeline_rhs_sorted_raw",
    "nax_e8p_expert_kblock_scale_slot_stream_rhs_sorted_raw",
    "nax_e8p_scale_group_route_block_reduce_rhs_sorted_raw",
    "nax_e8p_route_block_output_group_stream_rhs_sorted_raw",
    "nax_e8p_output_group_pretransposed_codeword_stream_rhs_sorted_raw",
    "nax_e8p_kblock_output_group_route_fused_stream_rhs_sorted_raw",
    "nax_e8p_route_tile_output_swizzle_stream_rhs_sorted_raw",
    "nax_e8p_split_byte_rhs_sorted_tiled_raw",
    "nax_e8p_split_byte_factor_reuse_rhs_sorted_tiled_raw",
    "nax_e8p_split_byte_factor_reuse_rhs_sorted_native_raw",
    "nax_e8p_component_stream_rhs_sorted_scalar_raw",
    "nax_e8p_component_stream_rhs_sorted_partial_raw",
    "nax_e8p_component_stream_rhs_sorted_tensorops_raw",
    "nax_e8p_component_stream_rhs_sorted_shared_decode_raw",
    "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_native_raw",
    "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_raw",
    "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_raw",
    "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_raw",
    "nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_raw",
    "nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_raw",
    "nax_e8p_sign_plane_abs_index_rhs_sorted_native_raw",
    "nax_e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_raw",
    "nax_e8p_sign_nibble_micro_lut_rhs_sorted_native_raw",
    "nax_e8p_packed_rhs_sorted_tiled_m128_raw",
    "nax_e8p_packed_rhs_sorted_tiled_k128_raw",
    "nax_e8p_predecoded_fp16_raw",
    "nax_e8p_fp16_sorted_steel_gs352_raw",
    "nax_e8p_fp16_sorted_steel_lut_raw",
    "nax_e8p_fp16_sorted_steel_tgcb_raw",
    "nax_e8p_fp16_sorted_steel_tgscale_raw",
    "nax_e8p_fp16_sorted_steel_tgcb_tgscale_raw",
    "nax_e8p_fp16_sorted_steel_tgcb_hoist_raw",
    "nax_e8p_fp16_sorted_steel_bk128_raw",
    "nax_e8p_fp16_sorted_steel_m128n32_raw",
    "nax_e8p_fp16_sorted_steel_m64n128_raw",
    "nax_e8p_fp16_sorted_steel_m32n64_raw",
    "nax_e8p_fp16_sorted_steel_m64n64t64_raw",
    "nax_e8p_fp16_sorted_steel_m32n64t128_raw",
    "nax_e8p_fp16_sorted_steel_m32n128_raw",
    "nax_e8_fp16_steel",
    "nax_e8_int8",
    "nax_predecoded_fp16",
    "vq_e1",
    "vq_current",
    "vq_decode_only",
    "vq_matmul_only",
    "vq_decode_direct_candidates",
    "vq_decode_to_scratch_gather_mm",
]

_KERNEL_DIR = Path(__file__).resolve().parents[1] / "kernels"
_GATHER_VQMM_DIAGNOSTIC_KERNEL = mx.fast.metal_kernel(
    name="mlx_vq_benchmark_gather_vqmm_diagnostic",
    input_names=["x", "codes", "scales", "codebook", "rhs_indices"],
    output_names=["out"],
    source=(_KERNEL_DIR / "gather_vqmm_diagnostics.metal").read_text(),
    header=(_KERNEL_DIR / "common_e8.metal").read_text(),
)


@dataclass(frozen=True)
class ProjectionFixture:
    projection: ProjectionKind
    tokens: int
    top_k: int
    experts: int
    input_dims: int
    output_dims: int
    mlx_group_size: int
    vq_group_size: int
    vq_code_bits: Literal[8, 16]
    dense_weight: mx.array | None
    x: mx.array
    indices: mx.array


@dataclass(frozen=True)
class SortedSteelInputs:
    sorted_x: mx.array
    sorted_rhs: mx.array
    order: mx.array
    inverse_order: mx.array
    tile_descriptors: tuple[mx.array, mx.array, mx.array]


def build_projection_fixture(
    *,
    projection: ProjectionKind,
    tokens: int,
    top_k: int,
    experts: int = 128,
    input_dims: int,
    output_dims: int,
    mlx_group_size: int = 128,
    vq_preferred_group_size: int = 512,
    vq_code_bits: Literal[8, 16] = 8,
    seed: int = 20260624,
) -> ProjectionFixture:
    if input_dims % mlx_group_size != 0:
        raise ValueError(f"input_dims={input_dims} must be divisible by mlx_group_size={mlx_group_size}")
    if vq_code_bits not in (8, 16):
        raise ValueError("vq_code_bits must be 8 or 16")
    vq_group_size = effective_group_size(input_dims, vq_preferred_group_size)
    rng = np.random.default_rng(seed)
    dense_weight = mx.array(
        rng.normal(scale=0.02, size=(experts, output_dims, input_dims)).astype(np.float32)
    ).astype(mx.bfloat16)
    x = mx.array(rng.normal(size=(tokens, input_dims)).astype(np.float32)).astype(mx.bfloat16)
    indices = mx.array(
        rng.integers(0, experts, size=(tokens, top_k), dtype=np.int32)
    )
    mx.eval(dense_weight, x, indices)
    return ProjectionFixture(
        projection=projection,
        tokens=tokens,
        top_k=top_k,
        experts=experts,
        input_dims=input_dims,
        output_dims=output_dims,
        mlx_group_size=mlx_group_size,
        vq_group_size=vq_group_size,
        vq_code_bits=vq_code_bits,
        dense_weight=dense_weight,
        x=x,
        indices=indices,
    )


def _dense_switch_linear(fixture: ProjectionFixture) -> SwitchLinear:
    if fixture.dense_weight is None:
        raise ValueError("dense_weight is required for dense/MLX projection variants")
    layer = SwitchLinear(
        fixture.input_dims,
        fixture.output_dims,
        fixture.experts,
        bias=False,
    )
    layer.weight = fixture.dense_weight
    mx.eval(layer.parameters())
    return layer


def build_projection_fixture_from_vq_artifact(
    *,
    artifact_dir: str | Path,
    layer_index: int,
    artifact_projection: Literal["gate_proj", "up_proj", "down_proj"],
    tokens: int,
    top_k: int,
    mlx_group_size: int = 128,
    seed: int = 20260624,
    block_module: str = "mlp",
) -> tuple[ProjectionFixture, QuantizedVQSwitchLinear, dict[str, Any]]:
    """Load a prequantized switch projection and build routed benchmark inputs.

    This path is intentionally separate from build_projection_fixture so Air-scale
    E8P timing does not include synthetic RTN/E8P encode cost.

    ``block_module`` is the decoder-block attribute that owns the routed experts.
    GLM-4.5-Air and GLM-5.2 name it ``mlp``; DeepSeek-V4-Flash names it ``ffn``
    (see the materialized prefixes in ``dsv4_vq_materialize``), so the family
    under test has to say which one rather than inheriting Air's.
    """

    artifact_root = Path(artifact_dir)
    shard = artifact_root / f"layer-{layer_index:05d}-{artifact_projection}.safetensors"
    prefix = f"model.layers.{layer_index}.{block_module}.switch_mlp.{artifact_projection}"
    layer = load_quantized_vq_switch_linear(shard, prefix)
    if layer.input_dims % mlx_group_size != 0:
        raise ValueError(
            f"artifact input_dims={layer.input_dims} must be divisible by mlx_group_size={mlx_group_size}"
        )
    rng = np.random.default_rng(seed)
    x = mx.array(rng.normal(size=(tokens, layer.input_dims)).astype(np.float32)).astype(mx.bfloat16)
    indices = mx.array(rng.integers(0, layer.num_experts, size=(tokens, top_k), dtype=np.int32))
    mx.eval(x, indices, layer.parameters())
    projection: ProjectionKind = "down" if artifact_projection == "down_proj" else "gate_up"
    fixture = ProjectionFixture(
        projection=projection,
        tokens=tokens,
        top_k=top_k,
        experts=layer.num_experts,
        input_dims=layer.input_dims,
        output_dims=layer.output_dims,
        mlx_group_size=mlx_group_size,
        vq_group_size=layer.group_size,
        vq_code_bits=layer.code_bits,
        dense_weight=None,
        x=x,
        indices=indices,
    )
    metadata = {
        "artifact_dir": str(artifact_root),
        "artifact_shard": str(shard),
        "artifact_layer": layer_index,
        "artifact_projection": artifact_projection,
        "artifact_prefix": prefix,
    }
    return fixture, layer, metadata


def _call_dense_or_mlx(layer, fixture: ProjectionFixture) -> mx.array:
    expanded = mx.expand_dims(fixture.x, (-2, -3))
    return layer(expanded, fixture.indices).squeeze(-2)


def _call_mlx_q2_routed(layer, fixture: ProjectionFixture) -> mx.array:
    expanded = mx.expand_dims(fixture.x, (-2, -3))
    if fixture.indices.size < 64:
        return layer(expanded, fixture.indices).squeeze(-2)
    sorted_x, sorted_indices, inverse_order = _gather_sort(expanded, fixture.indices)
    routed = layer(sorted_x, sorted_indices, sorted_indices=True)
    return _scatter_unsort(routed, inverse_order, fixture.indices.shape).squeeze(-2)


def _call_mlx_gather_mm_bf16_routed(weight: mx.array, fixture: ProjectionFixture) -> mx.array:
    expanded = mx.expand_dims(fixture.x.astype(mx.bfloat16), (-2, -3))
    weight_t = mx.swapaxes(weight.astype(mx.bfloat16), -1, -2)
    sorted_x, sorted_indices, inverse_order = _gather_sort(expanded, fixture.indices)
    routed = mx.gather_mm(
        sorted_x,
        weight_t,
        rhs_indices=sorted_indices,
        sorted_indices=True,
    )
    return _scatter_unsort(routed, inverse_order, fixture.indices.shape).squeeze(-2)


def _call_nax_predecoded_fp16_routed(weight_t: mx.array, fixture: ProjectionFixture) -> mx.array:
    expanded = mx.expand_dims(fixture.x.astype(mx.float16), (-2, -3))
    sorted_x, sorted_indices, inverse_order = _gather_sort(expanded, fixture.indices)
    routed = nax.predecoded_fp16_gather_mm(sorted_x, weight_t, sorted_indices)
    return _scatter_unsort(routed, inverse_order, fixture.indices.shape).squeeze(-2)


def _call_nax_e8_fp16_routed(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
) -> mx.array:
    flat_rhs = fixture.indices.reshape((-1,)).astype(mx.int32)
    flat_lhs = (mx.arange(flat_rhs.shape[0], dtype=mx.int32) // fixture.indices.shape[1]).astype(mx.int32)
    order = mx.argsort(flat_rhs)
    inverse_order = vq_switch._inverse_permutation(order)
    sorted_rhs = flat_rhs[order]
    sorted_lhs = flat_lhs[order]
    tile_descriptors = vq_switch._expert_block_tile_descriptors(
        sorted_rhs,
        num_experts=fixture.experts,
        route_tile=64,
    )
    routed = nax.nax_e8_fp16_routed_matmul(
        fixture.x.astype(mx.float16),
        layer.codes,
        layer.scales,
        sorted_lhs,
        *tile_descriptors,
        layer.codebook,
        group_size=fixture.vq_group_size,
    )
    return routed[inverse_order].reshape((*fixture.indices.shape, fixture.output_dims))


def _call_nax_e8_fp16_steel_routed(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
) -> mx.array:
    flat_rhs = fixture.indices.reshape((-1,)).astype(mx.int32)
    flat_lhs = (mx.arange(flat_rhs.shape[0], dtype=mx.int32) // fixture.indices.shape[1]).astype(mx.int32)
    order = mx.argsort(flat_rhs)
    inverse_order = vq_switch._inverse_permutation(order)
    sorted_rhs = flat_rhs[order]
    sorted_lhs = flat_lhs[order]
    tile_descriptors = vq_switch._expert_block_tile_descriptors(
        sorted_rhs,
        num_experts=fixture.experts,
        route_tile=64,
    )
    routed = nax.nax_e8_fp16_routed_steel_matmul(
        fixture.x.astype(mx.float16),
        layer.codes,
        layer.scales,
        sorted_lhs,
        *tile_descriptors,
        layer.codebook,
        group_size=fixture.vq_group_size,
    )
    return routed[inverse_order].reshape((*fixture.indices.shape, fixture.output_dims))


def _call_nax_e8_fp16_sorted_steel_routed(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
) -> mx.array:
    route_count = fixture.indices.size
    if fixture.input_dims >= 4096 and route_count >= 32768:
        expanded = mx.expand_dims(fixture.x.astype(mx.float16), (-2, -3))
        sorted_x, sorted_rhs, inverse_order = _gather_sort(expanded, fixture.indices)
        sorted_x = mx.squeeze(sorted_x, axis=-2)
    else:
        flat_rhs = fixture.indices.reshape((-1,)).astype(mx.int32)
        flat_lhs = (
            mx.arange(flat_rhs.shape[0], dtype=mx.int32) // fixture.indices.shape[1]
        ).astype(mx.int32)
        order = mx.argsort(flat_rhs)
        inverse_order = vq_switch._inverse_permutation(order)
        sorted_rhs = flat_rhs[order]
        sorted_lhs = flat_lhs[order]
        sorted_x = fixture.x[sorted_lhs].astype(mx.float16)
    tile_descriptors = vq_switch._expert_block_tile_descriptors(
        sorted_rhs,
        num_experts=fixture.experts,
        route_tile=64,
    )
    routed = nax.nax_e8_fp16_sorted_steel_matmul(
        sorted_x,
        layer.codes,
        layer.scales,
        *tile_descriptors,
        layer.codebook,
        group_size=fixture.vq_group_size,
    )
    return routed[inverse_order].reshape((*fixture.indices.shape, fixture.output_dims))


def _prepare_sorted_steel_inputs(
    fixture: ProjectionFixture,
    *,
    route_tile: int = 64,
) -> SortedSteelInputs:
    flat_rhs = fixture.indices.reshape((-1,)).astype(mx.int32)
    flat_lhs = (
        mx.arange(flat_rhs.shape[0], dtype=mx.int32) // fixture.indices.shape[1]
    ).astype(mx.int32)
    order = mx.argsort(flat_rhs)
    inverse_order = vq_switch._inverse_permutation(order)
    sorted_rhs = flat_rhs[order]
    sorted_lhs = flat_lhs[order]
    sorted_x = fixture.x[sorted_lhs].astype(mx.float16)
    tile_descriptors = vq_switch._expert_block_tile_descriptors(
        sorted_rhs,
        num_experts=fixture.experts,
        route_tile=route_tile,
    )
    mx.eval(sorted_x, sorted_rhs, order, inverse_order, *tile_descriptors)
    return SortedSteelInputs(
        sorted_x=sorted_x,
        sorted_rhs=sorted_rhs,
        order=order,
        inverse_order=inverse_order,
        tile_descriptors=tile_descriptors,
    )


def _compact_nonempty_sorted_tile_descriptors(
    sorted_inputs: SortedSteelInputs,
) -> SortedSteelInputs:
    tile_experts, tile_offsets, tile_counts = sorted_inputs.tile_descriptors
    counts_np = np.array(tile_counts, dtype=np.int32)
    keep = counts_np > 0
    if bool(np.all(keep)):
        return sorted_inputs
    compacted = (
        mx.array(np.array(tile_experts, dtype=np.int32)[keep], dtype=mx.int32),
        mx.array(np.array(tile_offsets, dtype=np.int32)[keep], dtype=mx.int32),
        mx.array(counts_np[keep], dtype=mx.int32),
    )
    mx.eval(*compacted)
    return SortedSteelInputs(
        sorted_x=sorted_inputs.sorted_x,
        sorted_rhs=sorted_inputs.sorted_rhs,
        order=sorted_inputs.order,
        inverse_order=sorted_inputs.inverse_order,
        tile_descriptors=compacted,
    )


def _call_nax_e8p_fp16_sorted_steel_routed(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
) -> mx.array:
    flat_rhs = fixture.indices.reshape((-1,)).astype(mx.int32)
    flat_lhs = (
        mx.arange(flat_rhs.shape[0], dtype=mx.int32) // fixture.indices.shape[1]
    ).astype(mx.int32)
    order = mx.argsort(flat_rhs)
    inverse_order = vq_switch._inverse_permutation(order)
    routed = vq_switch.gather_vqmm_sorted_routes(
        fixture.x,
        layer.codes,
        layer.scales,
        layer.codebook,
        flat_rhs[order],
        flat_lhs[order],
        input_dims=fixture.input_dims,
        output_dims=fixture.output_dims,
        group_size=fixture.vq_group_size,
        code_bits=16,
        implementation="nax_e8p",
        projection=fixture.projection,
    )
    return routed[inverse_order].reshape((*fixture.indices.shape, fixture.output_dims))


def _call_nax_e8_fp16_sorted_steel_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
) -> mx.array:
    del fixture
    return nax.nax_e8_fp16_sorted_steel_matmul(
        sorted_inputs.sorted_x,
        layer.codes,
        layer.scales,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        group_size=layer.group_size,
    )


def _call_nax_e8p_fp16_sorted_steel_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
) -> mx.array:
    del fixture
    return nax.nax_e8p_fp16_sorted_steel_matmul(
        sorted_inputs.sorted_x,
        layer.codes,
        layer.scales,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        group_size=layer.group_size,
    )


def _call_nax_e8p_fp16_sorted_direct_reduce_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
) -> mx.array:
    del fixture
    return nax.nax_e8p_fp16_sorted_direct_reduce_matmul(
        sorted_inputs.sorted_x,
        layer.codes,
        layer.scales,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        group_size=layer.group_size,
    )


def _call_nax_e8p_fp16_sorted_inline_b_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
) -> mx.array:
    del fixture
    return nax.nax_e8p_fp16_sorted_inline_b_matmul(
        sorted_inputs.sorted_x,
        layer.codes,
        layer.scales,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        group_size=layer.group_size,
    )


def _prepare_e8p_packed_rhs_tiles(
    layer: QuantizedVQSwitchLinear,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    packed = pack_e8p_rhs_tiles(
        np.array(layer.codes),
        np.array(layer.scales),
        group_size=layer.group_size,
        bn=64,
        bk=64,
    )
    code_tiles = mx.array(packed.code_tiles)
    scale_tiles = mx.array(packed.scale_tiles)
    scale_group_indices = mx.array(packed.scale_group_indices, dtype=mx.int32)
    codeword_scale_slots = mx.array(packed.codeword_scale_slots, dtype=mx.int32)
    mx.eval(code_tiles, scale_tiles, scale_group_indices, codeword_scale_slots)
    return code_tiles, scale_tiles, scale_group_indices, codeword_scale_slots


def _prepare_e8p_split_byte_rhs_tiles(
    layer: QuantizedVQSwitchLinear,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    packed = pack_e8p_split_byte_rhs_tiles(
        np.array(layer.codes),
        np.array(layer.scales),
        group_size=layer.group_size,
        bn=64,
        bk=64,
    )
    sign_tiles = mx.array(packed.sign_tiles)
    abs_index_tiles = mx.array(packed.abs_index_tiles)
    parity_tiles = mx.array(packed.parity_tiles)
    scale_tiles = mx.array(packed.scale_tiles)
    scale_group_indices = mx.array(packed.scale_group_indices, dtype=mx.int32)
    codeword_scale_slots = mx.array(packed.codeword_scale_slots, dtype=mx.int32)
    mx.eval(
        sign_tiles,
        abs_index_tiles,
        parity_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
    )
    return (
        sign_tiles,
        abs_index_tiles,
        parity_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
    )


def _prepare_e8p_split_byte_factor_reuse_rhs_tiles(
    layer: QuantizedVQSwitchLinear,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    packed = pack_e8p_split_byte_factor_reuse_rhs_tiles(
        np.array(layer.codes),
        np.array(layer.scales),
        group_size=layer.group_size,
        bn=64,
        bk=64,
    )
    sign_byte_lut = mx.array(packed.sign_byte_lut)
    sign_byte_slots = mx.array(packed.sign_byte_slots)
    abs_index_lut = mx.array(packed.abs_index_lut)
    abs_index_slots = mx.array(packed.abs_index_slots)
    scale_tiles = mx.array(packed.scale_tiles)
    scale_group_indices = mx.array(packed.scale_group_indices, dtype=mx.int32)
    codeword_scale_slots = mx.array(packed.codeword_scale_slots, dtype=mx.int32)
    mx.eval(
        sign_byte_lut,
        sign_byte_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
    )
    return (
        sign_byte_lut,
        sign_byte_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
    )


def _prepare_e8p_expert_kblock_factor_reuse_rhs_tiles(
    layer: QuantizedVQSwitchLinear,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    packed = pack_e8p_expert_kblock_factor_reuse_rhs_tiles(
        np.array(layer.codes),
        np.array(layer.scales),
        group_size=layer.group_size,
        bn=64,
        bk=64,
    )
    sign_byte_lut = mx.array(packed.sign_byte_lut)
    sign_byte_slots = mx.array(packed.sign_byte_slots)
    abs_index_lut = mx.array(packed.abs_index_lut)
    abs_index_slots = mx.array(packed.abs_index_slots)
    scale_tiles = mx.array(packed.scale_tiles)
    scale_group_indices = mx.array(packed.scale_group_indices, dtype=mx.int32)
    codeword_scale_slots = mx.array(packed.codeword_scale_slots, dtype=mx.int32)
    mx.eval(
        sign_byte_lut,
        sign_byte_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
    )
    return (
        sign_byte_lut,
        sign_byte_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
    )


def _prepare_e8p_component_stream_rhs_tiles(
    layer: QuantizedVQSwitchLinear,
) -> tuple[
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
]:
    packed = pack_e8p_component_stream_rhs_tiles(
        np.array(layer.codes),
        np.array(layer.scales),
        group_size=layer.group_size,
        bn=64,
        bk=64,
    )
    sign_component_bits = mx.array(packed.sign_component_bits)
    abs_index_tiles = mx.array(packed.abs_index_tiles)
    scale_tiles = mx.array(packed.scale_tiles)
    scale_group_indices = mx.array(packed.scale_group_indices, dtype=mx.int32)
    codeword_scale_slots = mx.array(packed.codeword_scale_slots, dtype=mx.int32)
    component_scale_slots = mx.array(packed.component_scale_slots, dtype=mx.int32)
    component_codeword_indices = mx.array(
        packed.component_codeword_indices,
        dtype=mx.int32,
    )
    component_offsets = mx.array(packed.component_offsets, dtype=mx.int32)
    mx.eval(
        sign_component_bits,
        abs_index_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        component_scale_slots,
        component_codeword_indices,
        component_offsets,
    )
    return (
        sign_component_bits,
        abs_index_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        component_scale_slots,
        component_codeword_indices,
        component_offsets,
    )


def _prepare_e8p_sign_nibble_abs_index_rhs_tiles(
    layer: QuantizedVQSwitchLinear,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    packed = pack_e8p_sign_nibble_abs_index_rhs_tiles(
        np.array(layer.codes),
        np.array(layer.scales),
        group_size=layer.group_size,
        bn=64,
        bk=64,
    )
    sign_low_nibble_tiles = mx.array(packed.sign_low_nibble_tiles)
    sign_high_nibble_tiles = mx.array(packed.sign_high_nibble_tiles)
    abs_index_tiles = mx.array(packed.abs_index_tiles)
    parity_tiles = mx.array(packed.parity_tiles)
    scale_tiles = mx.array(packed.scale_tiles)
    scale_group_indices = mx.array(packed.scale_group_indices, dtype=mx.int32)
    codeword_scale_slots = mx.array(packed.codeword_scale_slots, dtype=mx.int32)
    mx.eval(
        sign_low_nibble_tiles,
        sign_high_nibble_tiles,
        abs_index_tiles,
        parity_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
    )
    return (
        sign_low_nibble_tiles,
        sign_high_nibble_tiles,
        abs_index_tiles,
        parity_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
    )


def _prepare_e8p_sign_plane_abs_index_rhs_tiles(
    layer: QuantizedVQSwitchLinear,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    packed = pack_e8p_sign_plane_abs_index_rhs_tiles(
        np.array(layer.codes),
        np.array(layer.scales),
        group_size=layer.group_size,
        bn=64,
        bk=64,
    )
    sign_bit_planes = mx.array(packed.sign_bit_planes, dtype=mx.uint64)
    abs_index_tiles = mx.array(packed.abs_index_tiles)
    scale_tiles = mx.array(packed.scale_tiles)
    scale_group_indices = mx.array(packed.scale_group_indices, dtype=mx.int32)
    codeword_scale_slots = mx.array(packed.codeword_scale_slots, dtype=mx.int32)
    mx.eval(
        sign_bit_planes,
        abs_index_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
    )
    return (
        sign_bit_planes,
        abs_index_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
    )


def _prepare_e8p_sign_nibble_micro_lut_rhs_tiles(
    layer: QuantizedVQSwitchLinear,
) -> tuple[
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
]:
    packed = pack_e8p_sign_nibble_micro_lut_rhs_tiles(
        np.array(layer.codes),
        np.array(layer.scales),
        group_size=layer.group_size,
        bn=64,
        bk=64,
    )
    sign_low_nibble_lut = mx.array(packed.sign_low_nibble_lut)
    sign_low_nibble_slots = mx.array(packed.sign_low_nibble_slots)
    sign_high_nibble_lut = mx.array(packed.sign_high_nibble_lut)
    sign_high_nibble_slots = mx.array(packed.sign_high_nibble_slots)
    abs_index_lut = mx.array(packed.abs_index_lut)
    abs_index_slots = mx.array(packed.abs_index_slots)
    scale_tiles = mx.array(packed.scale_tiles)
    scale_group_indices = mx.array(packed.scale_group_indices, dtype=mx.int32)
    codeword_scale_slots = mx.array(packed.codeword_scale_slots, dtype=mx.int32)
    mx.eval(
        sign_low_nibble_lut,
        sign_low_nibble_slots,
        sign_high_nibble_lut,
        sign_high_nibble_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
    )
    return (
        sign_low_nibble_lut,
        sign_low_nibble_slots,
        sign_high_nibble_lut,
        sign_high_nibble_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
    )


def _call_nax_e8p_packed_rhs_sorted_tiled_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array],
) -> mx.array:
    code_tiles, scale_tiles, scale_group_indices, codeword_scale_slots = packed_rhs
    return nax.nax_e8p_packed_rhs_sorted_tiled_matmul(
        sorted_inputs.sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_route_slot_codeword_stream_rhs_sorted_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array],
) -> mx.array:
    code_tiles, scale_tiles, scale_group_indices, codeword_scale_slots = packed_rhs
    return nax.nax_e8p_route_slot_codeword_stream_rhs_sorted_matmul(
        sorted_inputs.sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_route_slot_mma_codeword_tile_rhs_sorted_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array],
) -> mx.array:
    code_tiles, scale_tiles, scale_group_indices, codeword_scale_slots = packed_rhs
    return nax.nax_e8p_route_slot_mma_codeword_tile_rhs_sorted_matmul(
        sorted_inputs.sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_active_route_tile_codeword_outer_product_rhs_sorted_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array],
) -> mx.array:
    code_tiles, scale_tiles, scale_group_indices, codeword_scale_slots = packed_rhs
    tile_experts, _tile_offsets, _tile_counts = sorted_inputs.tile_descriptors
    active_route_tiles = mx.arange(tile_experts.shape[0], dtype=mx.int32)
    return nax.nax_e8p_active_route_tile_codeword_outer_product_rhs_sorted_matmul(
        sorted_inputs.sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        active_route_tiles,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _expert_cohort_descriptors_from_route_tiles(
    *,
    tile_experts: mx.array,
    tile_counts: mx.array,
    experts: int,
) -> tuple[mx.array, mx.array, mx.array]:
    tile_experts_np = np.array(tile_experts, dtype=np.int32)
    tile_counts_np = np.array(tile_counts, dtype=np.int32)
    route_cohort_offsets: list[int] = []
    expert_cohort_offsets = np.full((experts,), -1, dtype=np.int32)
    expert_cohort_counts = np.zeros((experts,), dtype=np.int32)
    for expert in range(experts):
        expert_cohort_offsets[expert] = len(route_cohort_offsets)
        for route_tile, (tile_expert, count) in enumerate(
            zip(tile_experts_np, tile_counts_np, strict=True)
        ):
            if int(count) > 0 and int(tile_expert) == expert:
                route_cohort_offsets.append(route_tile)
        expert_cohort_counts[expert] = len(route_cohort_offsets) - int(
            expert_cohort_offsets[expert]
        )
    return (
        mx.array(expert_cohort_offsets, dtype=mx.int32),
        mx.array(expert_cohort_counts, dtype=mx.int32),
        mx.array(np.array(route_cohort_offsets, dtype=np.int32), dtype=mx.int32),
    )


def _call_nax_e8p_expert_cohort_codeword_broadcast_rhs_sorted_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array],
) -> mx.array:
    code_tiles, scale_tiles, scale_group_indices, codeword_scale_slots = packed_rhs
    tile_experts, _tile_offsets, tile_counts = sorted_inputs.tile_descriptors
    expert_cohort_offsets, expert_cohort_counts, route_cohort_offsets = (
        _expert_cohort_descriptors_from_route_tiles(
            tile_experts=tile_experts,
            tile_counts=tile_counts,
            experts=fixture.experts,
        )
    )
    return nax.nax_e8p_expert_cohort_codeword_broadcast_rhs_sorted_matmul(
        sorted_inputs.sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        expert_cohort_offsets,
        expert_cohort_counts,
        route_cohort_offsets,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _route_batch_segment_descriptors(
    *, routes: int, route_batch_size: int
) -> tuple[mx.array, mx.array, mx.array]:
    if route_batch_size <= 0:
        raise ValueError("route_batch_size must be positive")
    offsets: list[int] = []
    counts: list[int] = []
    route_ids: list[int] = []
    for start in range(0, routes, route_batch_size):
        end = min(start + route_batch_size, routes)
        offsets.append(len(route_ids))
        batch_route_ids = list(range(start, end))
        route_ids.extend(reversed(batch_route_ids))
        counts.append(end - start)
    descriptors = (
        mx.array(np.array(offsets, dtype=np.int32), dtype=mx.int32),
        mx.array(np.array(counts, dtype=np.int32), dtype=mx.int32),
        mx.array(np.array(route_ids, dtype=np.int32), dtype=mx.int32),
    )
    mx.eval(*descriptors)
    return descriptors


def _call_nax_e8p_route_batch_segmented_codeword_reduce_rhs_sorted_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array],
    *,
    route_batch_size: int = 8,
) -> mx.array:
    code_tiles, scale_tiles, scale_group_indices, codeword_scale_slots = packed_rhs
    route_batch_segment_offsets, route_batch_segment_counts, route_batch_route_ids = (
        _route_batch_segment_descriptors(
            routes=int(sorted_inputs.sorted_x.shape[0]),
            route_batch_size=route_batch_size,
        )
    )
    return nax.nax_e8p_route_batch_segmented_codeword_reduce_rhs_sorted_matmul(
        sorted_inputs.sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        route_batch_segment_offsets,
        route_batch_segment_counts,
        route_batch_route_ids,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _token_cohort_descriptors(
    *, routes: int, token_cohort_size: int, experts: int
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    if token_cohort_size <= 0:
        raise ValueError("token_cohort_size must be positive")
    offsets: list[int] = []
    counts: list[int] = []
    route_ids: list[int] = []
    for start in range(0, routes, token_cohort_size):
        end = min(start + token_cohort_size, routes)
        offsets.append(len(route_ids))
        route_ids.extend(range(start, end))
        counts.append(end - start)
    descriptors = (
        mx.array(np.array(offsets, dtype=np.int32), dtype=mx.int32),
        mx.array(np.array(counts, dtype=np.int32), dtype=mx.int32),
        mx.array(np.arange(experts, dtype=np.int32), dtype=mx.int32),
        mx.array(np.array(route_ids, dtype=np.int32), dtype=mx.int32),
    )
    mx.eval(*descriptors)
    return descriptors


def _token_route_output_stripe_descriptors(
    *,
    sorted_order: mx.array,
    tokens: int,
    top_k: int,
    max_routes_per_token: int = 8,
) -> tuple[mx.array, mx.array, mx.array]:
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if max_routes_per_token <= 0:
        raise ValueError("max_routes_per_token must be positive")

    buckets: list[list[int]] = [[] for _ in range(tokens)]
    for sorted_route_slot, original_route_slot in enumerate(
        np.array(sorted_order, dtype=np.int32)
    ):
        token = int(original_route_slot) // top_k
        if token < 0 or token >= tokens:
            raise ValueError("sorted route order contains a token outside fixture bounds")
        buckets[token].append(sorted_route_slot)

    offsets: list[int] = []
    counts: list[int] = []
    route_ids: list[int] = []
    for bucket in buckets:
        if len(bucket) > max_routes_per_token:
            raise ValueError("token-route bucket exceeds native route-slot capacity")
        offsets.append(len(route_ids))
        route_ids.extend(bucket)
        counts.append(len(bucket))

    descriptors = (
        mx.array(np.array(offsets, dtype=np.int32), dtype=mx.int32),
        mx.array(np.array(counts, dtype=np.int32), dtype=mx.int32),
        mx.array(np.array(route_ids, dtype=np.int32), dtype=mx.int32),
    )
    mx.eval(*descriptors)
    return descriptors


def _route_block_descriptors(
    *, routes: int, route_block_size: int
) -> tuple[mx.array, mx.array, mx.array]:
    if routes <= 0:
        raise ValueError("routes must be positive")
    if route_block_size <= 0:
        raise ValueError("route_block_size must be positive")

    offsets: list[int] = []
    counts: list[int] = []
    route_slot_ids: list[int] = []
    for start in range(0, routes, route_block_size):
        offsets.append(len(route_slot_ids))
        count = min(route_block_size, routes - start)
        counts.append(count)
        route_slot_ids.extend(range(start, start + count))

    descriptors = (
        mx.array(np.array(offsets, dtype=np.int32), dtype=mx.int32),
        mx.array(np.array(counts, dtype=np.int32), dtype=mx.int32),
        mx.array(np.array(route_slot_ids, dtype=np.int32), dtype=mx.int32),
    )
    mx.eval(*descriptors)
    return descriptors


def _route_tile_output_swizzle_descriptors(
    *, tile_offsets: mx.array, tile_counts: mx.array
) -> tuple[mx.array, mx.array, mx.array]:
    tile_offsets_np = np.array(tile_offsets, dtype=np.int32)
    tile_counts_np = np.array(tile_counts, dtype=np.int32)
    if tile_offsets_np.ndim != 1 or tile_counts_np.ndim != 1:
        raise ValueError("tile descriptor arrays must be one-dimensional")
    if tile_offsets_np.shape != tile_counts_np.shape:
        raise ValueError("tile offsets and counts must have the same shape")

    offsets: list[int] = []
    counts: list[int] = []
    route_slot_ids: list[int] = []
    for offset, count in zip(tile_offsets_np.tolist(), tile_counts_np.tolist()):
        offsets.append(len(route_slot_ids))
        count_i = int(count)
        counts.append(count_i)
        route_slot_ids.extend(range(int(offset), int(offset) + count_i))

    descriptors = (
        mx.array(np.array(offsets, dtype=np.int32), dtype=mx.int32),
        mx.array(np.array(counts, dtype=np.int32), dtype=mx.int32),
        mx.array(np.array(route_slot_ids, dtype=np.int32), dtype=mx.int32),
    )
    mx.eval(*descriptors)
    return descriptors


def _call_nax_e8p_token_cohort_codeword_stream_rhs_sorted_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array],
    *,
    token_cohort_size: int = 8,
) -> mx.array:
    code_tiles, scale_tiles, scale_group_indices, codeword_scale_slots = packed_rhs
    (
        token_cohort_offsets,
        token_cohort_counts,
        token_cohort_active_expert_ids,
        token_cohort_route_slot_ids,
    ) = _token_cohort_descriptors(
        routes=int(sorted_inputs.sorted_x.shape[0]),
        token_cohort_size=token_cohort_size,
        experts=fixture.experts,
    )
    return nax.nax_e8p_token_cohort_codeword_stream_rhs_sorted_matmul(
        sorted_inputs.sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        token_cohort_offsets,
        token_cohort_counts,
        token_cohort_active_expert_ids,
        token_cohort_route_slot_ids,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_token_cohort_mma_codeword_tile_rhs_sorted_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array],
    *,
    token_cohort_size: int = 8,
) -> mx.array:
    code_tiles, scale_tiles, scale_group_indices, codeword_scale_slots = packed_rhs
    (
        token_cohort_offsets,
        token_cohort_counts,
        token_cohort_active_expert_ids,
        token_cohort_route_slot_ids,
    ) = _token_cohort_descriptors(
        routes=int(sorted_inputs.sorted_x.shape[0]),
        token_cohort_size=token_cohort_size,
        experts=fixture.experts,
    )
    return nax.nax_e8p_token_cohort_mma_codeword_tile_rhs_sorted_matmul(
        sorted_inputs.sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        token_cohort_offsets,
        token_cohort_counts,
        token_cohort_active_expert_ids,
        token_cohort_route_slot_ids,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_output_stationary_codeword_tile_rhs_sorted_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array],
    *,
    route_batch_size: int = 8,
) -> mx.array:
    code_tiles, scale_tiles, scale_group_indices, codeword_scale_slots = packed_rhs
    (
        route_batch_offsets,
        route_batch_counts,
        route_batch_active_expert_ids,
        route_batch_route_slot_ids,
    ) = _token_cohort_descriptors(
        routes=int(sorted_inputs.sorted_x.shape[0]),
        token_cohort_size=route_batch_size,
        experts=fixture.experts,
    )
    return nax.nax_e8p_output_stationary_codeword_tile_rhs_sorted_matmul(
        sorted_inputs.sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        route_batch_offsets,
        route_batch_counts,
        route_batch_active_expert_ids,
        route_batch_route_slot_ids,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_input_stationary_codeword_tile_rhs_sorted_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array],
    *,
    route_batch_size: int = 8,
) -> mx.array:
    code_tiles, scale_tiles, scale_group_indices, codeword_scale_slots = packed_rhs
    (
        route_batch_offsets,
        route_batch_counts,
        route_batch_active_expert_ids,
        route_batch_route_slot_ids,
    ) = _token_cohort_descriptors(
        routes=int(sorted_inputs.sorted_x.shape[0]),
        token_cohort_size=route_batch_size,
        experts=fixture.experts,
    )
    return nax.nax_e8p_input_stationary_codeword_tile_rhs_sorted_matmul(
        sorted_inputs.sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        route_batch_offsets,
        route_batch_counts,
        route_batch_active_expert_ids,
        route_batch_route_slot_ids,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _route_codeword_lut_ids_from_code_tiles(
    code_tiles: np.ndarray,
    *,
    output_dims: int,
) -> np.ndarray:
    if code_tiles.ndim != 5:
        raise ValueError("code_tiles must have shape [E,n_tiles,k_blocks,bn,codewords]")
    _, n_tiles, _, bn, codeword_count = code_tiles.shape
    encoded: list[np.ndarray] = []
    for n_tile in range(n_tiles):
        valid_n = min(bn, max(output_dims - n_tile * bn, 0))
        if valid_n <= 0:
            continue
        for codeword_axis in range(codeword_count):
            codewords = code_tiles[:, n_tile, :, :valid_n, codeword_axis].reshape(-1)
            encoded.append(
                (np.uint32(codeword_axis + 1) << np.uint32(16))
                | codewords.astype(np.uint32)
            )
    if not encoded:
        raise ValueError("no route-codeword LUT ids found for output_dims")
    ids = np.unique(np.concatenate(encoded).astype(np.uint32))
    if np.any(ids > np.iinfo(np.int32).max):
        raise ValueError("route-codeword LUT ids must fit int32")
    return ids.astype(np.int32)


def _route_codeword_dot_lut_from_sorted_inputs(
    *,
    sorted_x: mx.array,
    route_codeword_lut_codeword_ids: np.ndarray,
    k_block_count: int,
    bk: int,
    codebook: mx.array,
) -> mx.array:
    sorted_x_np = np.array(sorted_x, dtype=np.float32)
    codebook_np = np.array(codebook, dtype=np.uint32)
    encoded_ids = route_codeword_lut_codeword_ids.astype(np.uint32, copy=False)
    codeword_axes = ((encoded_ids >> np.uint32(16)) - np.uint32(1)).astype(np.int32)
    compressed_codewords = (encoded_ids & np.uint32(0xFFFF)).astype(np.uint16)
    decoded_codewords = decode_e8p(compressed_codewords, codebook_np).astype(np.float32)
    route_count = sorted_x_np.shape[0]
    lut_id_count = encoded_ids.shape[0]
    lut = np.empty((route_count, k_block_count, lut_id_count), dtype=np.float16)
    for k_block in range(k_block_count):
        for codeword_axis in range(bk // 8):
            id_indices = np.flatnonzero(codeword_axes == codeword_axis)
            if id_indices.size == 0:
                continue
            k_start = k_block * bk + codeword_axis * 8
            k_stop = k_start + 8
            lut[:, k_block, id_indices] = (
                sorted_x_np[:, k_start:k_stop] @ decoded_codewords[id_indices].T
            ).astype(np.float16)
    out = mx.array(lut, dtype=mx.float16)
    mx.eval(out)
    return out


def _call_nax_e8p_route_codeword_lut_accumulate_rhs_sorted_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array],
) -> mx.array:
    code_tiles, scale_tiles, scale_group_indices, codeword_scale_slots = packed_rhs
    code_tiles_np = np.array(code_tiles)
    route_codeword_lut_codeword_ids_np = _route_codeword_lut_ids_from_code_tiles(
        code_tiles_np,
        output_dims=fixture.output_dims,
    )
    route_count = int(sorted_inputs.sorted_x.shape[0])
    k_block_count = int(code_tiles.shape[2])
    bk = int(code_tiles.shape[4]) * 8
    route_local_codeword_dot_lut = _route_codeword_dot_lut_from_sorted_inputs(
        sorted_x=sorted_inputs.sorted_x,
        route_codeword_lut_codeword_ids=route_codeword_lut_codeword_ids_np,
        k_block_count=k_block_count,
        bk=bk,
        codebook=layer.codebook,
    )
    route_codeword_lut_route_slots = mx.arange(route_count, dtype=mx.int32)
    route_codeword_lut_offsets = mx.zeros((route_count,), dtype=mx.int32)
    route_codeword_lut_counts = mx.full(
        (route_count,),
        route_codeword_lut_codeword_ids_np.shape[0],
        dtype=mx.int32,
    )
    route_codeword_lut_codeword_ids = mx.array(
        route_codeword_lut_codeword_ids_np,
        dtype=mx.int32,
    )
    mx.eval(
        route_codeword_lut_route_slots,
        route_codeword_lut_offsets,
        route_codeword_lut_counts,
        route_codeword_lut_codeword_ids,
    )
    return nax.nax_e8p_route_codeword_lut_accumulate_rhs_sorted_matmul(
        route_local_codeword_dot_lut,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        route_codeword_lut_route_slots,
        route_codeword_lut_offsets,
        route_codeword_lut_counts,
        route_codeword_lut_codeword_ids,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_rowwise_codeword_tile_accumulate_rhs_sorted_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array],
    *,
    route_microtile_size: int = 8,
) -> mx.array:
    code_tiles, scale_tiles, scale_group_indices, codeword_scale_slots = packed_rhs
    (
        rowwise_offsets,
        rowwise_counts,
        _rowwise_active_expert_ids,
        rowwise_route_slot_ids,
    ) = _token_cohort_descriptors(
        routes=int(sorted_inputs.sorted_x.shape[0]),
        token_cohort_size=route_microtile_size,
        experts=fixture.experts,
    )
    return nax.nax_e8p_rowwise_codeword_tile_accumulate_rhs_sorted_matmul(
        sorted_inputs.sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        rowwise_offsets,
        rowwise_counts,
        rowwise_route_slot_ids,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_output_tile_local_codeword_lut_rhs_sorted_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array],
    *,
    route_microtile_size: int = 8,
) -> mx.array:
    code_tiles, scale_tiles, scale_group_indices, codeword_scale_slots = packed_rhs
    (
        route_microtile_offsets,
        route_microtile_counts,
        _active_expert_ids,
        route_microtile_route_slot_ids,
    ) = _token_cohort_descriptors(
        routes=int(sorted_inputs.sorted_x.shape[0]),
        token_cohort_size=route_microtile_size,
        experts=fixture.experts,
    )
    return nax.nax_e8p_output_tile_local_codeword_lut_rhs_sorted_matmul(
        sorted_inputs.sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        route_microtile_offsets,
        route_microtile_counts,
        route_microtile_route_slot_ids,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_route_microtile_codeword_block_reduce_rhs_sorted_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array],
    *,
    route_microtile_size: int = 8,
) -> mx.array:
    code_tiles, scale_tiles, scale_group_indices, codeword_scale_slots = packed_rhs
    (
        route_microtile_offsets,
        route_microtile_counts,
        _active_expert_ids,
        route_microtile_route_slot_ids,
    ) = _token_cohort_descriptors(
        routes=int(sorted_inputs.sorted_x.shape[0]),
        token_cohort_size=route_microtile_size,
        experts=fixture.experts,
    )
    return nax.nax_e8p_route_microtile_codeword_block_reduce_rhs_sorted_matmul(
        sorted_inputs.sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        route_microtile_offsets,
        route_microtile_counts,
        route_microtile_route_slot_ids,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_kblock_wavefront_codeword_scan_rhs_sorted_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array],
    *,
    route_microtile_size: int = 8,
) -> mx.array:
    code_tiles, scale_tiles, scale_group_indices, codeword_scale_slots = packed_rhs
    (
        kblock_wavefront_offsets,
        kblock_wavefront_counts,
        _active_expert_ids,
        kblock_wavefront_route_slot_ids,
    ) = _token_cohort_descriptors(
        routes=int(sorted_inputs.sorted_x.shape[0]),
        token_cohort_size=route_microtile_size,
        experts=fixture.experts,
    )
    return nax.nax_e8p_kblock_wavefront_codeword_scan_rhs_sorted_matmul(
        sorted_inputs.sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        kblock_wavefront_offsets,
        kblock_wavefront_counts,
        kblock_wavefront_route_slot_ids,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_token_route_output_stripe_pipeline_rhs_sorted_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array],
    *,
    max_routes_per_token: int = 8,
) -> mx.array:
    code_tiles, scale_tiles, scale_group_indices, codeword_scale_slots = packed_rhs
    (
        token_route_output_stripe_offsets,
        token_route_output_stripe_counts,
        token_route_output_stripe_route_slot_ids,
    ) = _token_route_output_stripe_descriptors(
        sorted_order=sorted_inputs.order,
        tokens=fixture.tokens,
        top_k=fixture.top_k,
        max_routes_per_token=max_routes_per_token,
    )
    return nax.nax_e8p_token_route_output_stripe_pipeline_rhs_sorted_matmul(
        sorted_inputs.sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        token_route_output_stripe_offsets,
        token_route_output_stripe_counts,
        token_route_output_stripe_route_slot_ids,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_expert_kblock_scale_slot_stream_rhs_sorted_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array],
) -> mx.array:
    code_tiles, scale_tiles, scale_group_indices, codeword_scale_slots = packed_rhs
    return nax.nax_e8p_expert_kblock_scale_slot_stream_rhs_sorted_matmul(
        sorted_inputs.sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_scale_group_route_block_reduce_rhs_sorted_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array],
    *,
    route_block_size: int = 8,
) -> mx.array:
    code_tiles, scale_tiles, scale_group_indices, codeword_scale_slots = packed_rhs
    (
        route_block_offsets,
        route_block_counts,
        route_block_route_slot_ids,
    ) = _route_block_descriptors(
        routes=int(sorted_inputs.sorted_x.shape[0]),
        route_block_size=route_block_size,
    )
    return nax.nax_e8p_scale_group_route_block_reduce_rhs_sorted_matmul(
        sorted_inputs.sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        route_block_offsets,
        route_block_counts,
        route_block_route_slot_ids,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_route_block_output_group_stream_rhs_sorted_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array],
    *,
    route_block_size: int = 8,
) -> mx.array:
    code_tiles, scale_tiles, scale_group_indices, codeword_scale_slots = packed_rhs
    (
        route_block_offsets,
        route_block_counts,
        route_block_route_slot_ids,
    ) = _route_block_descriptors(
        routes=int(sorted_inputs.sorted_x.shape[0]),
        route_block_size=route_block_size,
    )
    return nax.nax_e8p_route_block_output_group_stream_rhs_sorted_matmul(
        sorted_inputs.sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        route_block_offsets,
        route_block_counts,
        route_block_route_slot_ids,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_output_group_pretransposed_codeword_stream_rhs_sorted_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array],
    *,
    route_block_size: int = 8,
) -> mx.array:
    code_tiles, scale_tiles, scale_group_indices, codeword_scale_slots = packed_rhs
    (
        route_block_offsets,
        route_block_counts,
        route_block_route_slot_ids,
    ) = _route_block_descriptors(
        routes=int(sorted_inputs.sorted_x.shape[0]),
        route_block_size=route_block_size,
    )
    return nax.nax_e8p_output_group_pretransposed_codeword_stream_rhs_sorted_matmul(
        sorted_inputs.sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        route_block_offsets,
        route_block_counts,
        route_block_route_slot_ids,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_kblock_output_group_route_fused_stream_rhs_sorted_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array],
    *,
    route_block_size: int = 8,
) -> mx.array:
    code_tiles, scale_tiles, scale_group_indices, codeword_scale_slots = packed_rhs
    (
        route_block_offsets,
        route_block_counts,
        route_block_route_slot_ids,
    ) = _route_block_descriptors(
        routes=int(sorted_inputs.sorted_x.shape[0]),
        route_block_size=route_block_size,
    )
    return nax.nax_e8p_kblock_output_group_route_fused_stream_rhs_sorted_matmul(
        sorted_inputs.sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        route_block_offsets,
        route_block_counts,
        route_block_route_slot_ids,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_route_tile_output_swizzle_stream_rhs_sorted_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array],
) -> mx.array:
    code_tiles, scale_tiles, scale_group_indices, codeword_scale_slots = packed_rhs
    _tile_experts, tile_offsets, tile_counts = sorted_inputs.tile_descriptors
    (
        route_tile_output_swizzle_offsets,
        route_tile_output_swizzle_counts,
        route_tile_output_swizzle_route_slot_ids,
    ) = _route_tile_output_swizzle_descriptors(
        tile_offsets=tile_offsets,
        tile_counts=tile_counts,
    )
    return nax.nax_e8p_route_tile_output_swizzle_stream_rhs_sorted_matmul(
        sorted_inputs.sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        route_tile_output_swizzle_offsets,
        route_tile_output_swizzle_counts,
        route_tile_output_swizzle_route_slot_ids,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_split_byte_rhs_sorted_tiled_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array],
) -> mx.array:
    (
        sign_tiles,
        abs_index_tiles,
        parity_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
    ) = packed_rhs
    return nax.nax_e8p_split_byte_rhs_sorted_tiled_matmul(
        sorted_inputs.sorted_x,
        sign_tiles,
        abs_index_tiles,
        parity_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_split_byte_factor_reuse_rhs_sorted_tiled_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array],
) -> mx.array:
    (
        sign_byte_lut,
        sign_byte_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
    ) = packed_rhs
    return nax.nax_e8p_split_byte_factor_reuse_rhs_sorted_tiled_matmul(
        sorted_inputs.sorted_x,
        sign_byte_lut,
        sign_byte_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_split_byte_factor_reuse_rhs_sorted_native_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array],
) -> mx.array:
    (
        sign_byte_lut,
        sign_byte_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
    ) = packed_rhs
    return nax.nax_e8p_split_byte_factor_reuse_rhs_sorted_native_matmul(
        sorted_inputs.sorted_x,
        sign_byte_lut,
        sign_byte_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array],
) -> mx.array:
    (
        sign_byte_lut,
        sign_byte_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
    ) = packed_rhs
    return nax.nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul(
        sorted_inputs.sorted_x,
        sign_byte_lut,
        sign_byte_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_expert_kblock_factor_reuse_rhs_sorted_native_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array],
) -> mx.array:
    (
        sign_byte_lut,
        sign_byte_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
    ) = packed_rhs
    return nax.nax_e8p_expert_kblock_factor_reuse_rhs_sorted_native_matmul(
        sorted_inputs.sorted_x,
        sign_byte_lut,
        sign_byte_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_component_stream_rhs_sorted_scalar_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
    ],
) -> mx.array:
    (
        sign_component_bits,
        abs_index_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        component_scale_slots,
        component_codeword_indices,
        component_offsets,
    ) = packed_rhs
    return nax.nax_e8p_component_stream_rhs_sorted_scalar_matmul(
        sorted_inputs.sorted_x,
        sign_component_bits,
        abs_index_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        component_scale_slots,
        component_codeword_indices,
        component_offsets,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_component_stream_rhs_sorted_partial_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
    ],
) -> mx.array:
    (
        sign_component_bits,
        abs_index_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        component_scale_slots,
        component_codeword_indices,
        component_offsets,
    ) = packed_rhs
    return nax.nax_e8p_component_stream_rhs_sorted_partial_matmul(
        sorted_inputs.sorted_x,
        sign_component_bits,
        abs_index_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        component_scale_slots,
        component_codeword_indices,
        component_offsets,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_component_stream_rhs_sorted_tensorops_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
    ],
) -> mx.array:
    (
        sign_component_bits,
        abs_index_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        component_scale_slots,
        component_codeword_indices,
        component_offsets,
    ) = packed_rhs
    return nax.nax_e8p_component_stream_rhs_sorted_tensorops_matmul(
        sorted_inputs.sorted_x,
        sign_component_bits,
        abs_index_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        component_scale_slots,
        component_codeword_indices,
        component_offsets,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_component_stream_rhs_sorted_shared_decode_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
    ],
) -> mx.array:
    (
        sign_component_bits,
        abs_index_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        component_scale_slots,
        component_codeword_indices,
        component_offsets,
    ) = packed_rhs
    return nax.nax_e8p_component_stream_rhs_sorted_shared_decode_matmul(
        sorted_inputs.sorted_x,
        sign_component_bits,
        abs_index_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        component_scale_slots,
        component_codeword_indices,
        component_offsets,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array],
) -> mx.array:
    (
        sign_byte_lut,
        sign_byte_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
    ) = packed_rhs
    return nax.nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_matmul(
        sorted_inputs.sorted_x,
        sign_byte_lut,
        sign_byte_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array],
) -> mx.array:
    (
        sign_byte_lut,
        sign_byte_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
    ) = packed_rhs
    return nax.nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_matmul(
        sorted_inputs.sorted_x,
        sign_byte_lut,
        sign_byte_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array],
) -> mx.array:
    (
        sign_low_nibble_tiles,
        sign_high_nibble_tiles,
        abs_index_tiles,
        parity_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
    ) = packed_rhs
    return nax.nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_matmul(
        sorted_inputs.sorted_x,
        sign_low_nibble_tiles,
        sign_high_nibble_tiles,
        abs_index_tiles,
        parity_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array, mx.array],
) -> mx.array:
    (
        sign_bit_planes,
        abs_index_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
    ) = packed_rhs
    return nax.nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_matmul(
        sorted_inputs.sorted_x,
        sign_bit_planes,
        abs_index_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_sign_plane_abs_index_rhs_sorted_native_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array, mx.array],
) -> mx.array:
    (
        sign_bit_planes,
        abs_index_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
    ) = packed_rhs
    return nax.nax_e8p_sign_plane_abs_index_rhs_sorted_native_matmul(
        sorted_inputs.sorted_x,
        sign_bit_planes,
        abs_index_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
    ],
) -> mx.array:
    (
        sign_low_nibble_lut,
        sign_low_nibble_slots,
        sign_high_nibble_lut,
        sign_high_nibble_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
    ) = packed_rhs
    return nax.nax_e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_matmul(
        sorted_inputs.sorted_x,
        sign_low_nibble_lut,
        sign_low_nibble_slots,
        sign_high_nibble_lut,
        sign_high_nibble_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_sign_nibble_micro_lut_rhs_sorted_native_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
    ],
) -> mx.array:
    (
        sign_low_nibble_lut,
        sign_low_nibble_slots,
        sign_high_nibble_lut,
        sign_high_nibble_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
    ) = packed_rhs
    return nax.nax_e8p_sign_nibble_micro_lut_rhs_sorted_native_matmul(
        sorted_inputs.sorted_x,
        sign_low_nibble_lut,
        sign_low_nibble_slots,
        sign_high_nibble_lut,
        sign_high_nibble_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_packed_rhs_sorted_tiled_m128_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array],
) -> mx.array:
    code_tiles, scale_tiles, scale_group_indices, codeword_scale_slots = packed_rhs
    return nax.nax_e8p_packed_rhs_sorted_tiled_m128_matmul(
        sorted_inputs.sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _call_nax_e8p_packed_rhs_sorted_tiled_k128_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
    packed_rhs: tuple[mx.array, mx.array, mx.array, mx.array],
) -> mx.array:
    code_tiles, scale_tiles, scale_group_indices, codeword_scale_slots = packed_rhs
    return nax.nax_e8p_packed_rhs_sorted_tiled_k128_matmul(
        sorted_inputs.sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        output_dims=fixture.output_dims,
    )


def _predecode_vq_layer_weight_t_fp16(layer: QuantizedVQSwitchLinear) -> mx.array:
    codes_np = np.array(layer.codes)
    scales_np = np.array(layer.scales)
    codebook_np = np.array(layer.codebook)
    decoded_np = np.empty(
        (layer.num_experts, layer.output_dims, layer.input_dims),
        dtype=np.float16,
    )
    for expert in range(layer.num_experts):
        decoded_np[expert] = decode_weight_matrix(
            codes_np[expert],
            scales_np[expert],
            code_bits=layer.code_bits,
            codebook=codebook_np,
        ).astype(np.float16, copy=False)
    weight_t = mx.swapaxes(mx.array(decoded_np, dtype=mx.float16), -1, -2)
    mx.eval(weight_t)
    return weight_t


def _call_nax_e8p_predecoded_fp16_raw(
    weight_t: mx.array,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
) -> mx.array:
    del fixture
    sorted_x = mx.expand_dims(sorted_inputs.sorted_x, axis=-2)
    routed = nax.predecoded_fp16_gather_mm(
        sorted_x,
        weight_t,
        sorted_inputs.sorted_rhs,
    )
    return mx.squeeze(routed, axis=-2)


def _call_decoded_fp16_sorted_direct_reference(
    weight_t: mx.array,
    sorted_inputs: SortedSteelInputs,
) -> mx.array:
    sorted_x = mx.expand_dims(sorted_inputs.sorted_x, axis=-2)
    gathered_weight_t = weight_t[sorted_inputs.sorted_rhs]
    routed = mx.matmul(sorted_x, gathered_weight_t)
    return mx.squeeze(routed, axis=-2)


def _call_nax_e8p_fp16_sorted_steel_gs352_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
) -> mx.array:
    del fixture
    return nax.nax_e8p_fp16_sorted_steel_gs352_matmul(
        sorted_inputs.sorted_x,
        layer.codes,
        layer.scales,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        group_size=layer.group_size,
    )


def _call_nax_e8p_fp16_sorted_steel_lut_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
) -> mx.array:
    del fixture
    return nax.nax_e8p_fp16_sorted_steel_lut_matmul(
        sorted_inputs.sorted_x,
        layer.codes,
        layer.scales,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        group_size=layer.group_size,
    )


def _call_nax_e8p_fp16_sorted_steel_tgcb_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
) -> mx.array:
    del fixture
    return nax.nax_e8p_fp16_sorted_steel_tgcb_matmul(
        sorted_inputs.sorted_x,
        layer.codes,
        layer.scales,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        group_size=layer.group_size,
    )


def _call_nax_e8p_fp16_sorted_steel_tgscale_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
) -> mx.array:
    del fixture
    return nax.nax_e8p_fp16_sorted_steel_tgscale_matmul(
        sorted_inputs.sorted_x,
        layer.codes,
        layer.scales,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        group_size=layer.group_size,
    )


def _call_nax_e8p_fp16_sorted_steel_tgcb_tgscale_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
) -> mx.array:
    del fixture
    return nax.nax_e8p_fp16_sorted_steel_tgcb_tgscale_matmul(
        sorted_inputs.sorted_x,
        layer.codes,
        layer.scales,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        group_size=layer.group_size,
    )


def _call_nax_e8p_fp16_sorted_steel_tgcb_hoist_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
) -> mx.array:
    del fixture
    return nax.nax_e8p_fp16_sorted_steel_tgcb_hoist_matmul(
        sorted_inputs.sorted_x,
        layer.codes,
        layer.scales,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        group_size=layer.group_size,
    )


def _call_nax_e8p_fp16_sorted_steel_bk128_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
) -> mx.array:
    del fixture
    return nax.nax_e8p_fp16_sorted_steel_bk128_matmul(
        sorted_inputs.sorted_x,
        layer.codes,
        layer.scales,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        group_size=layer.group_size,
    )


def _call_nax_e8p_fp16_sorted_steel_m128n32_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
) -> mx.array:
    del fixture
    return nax.nax_e8p_fp16_sorted_steel_m128n32_matmul(
        sorted_inputs.sorted_x,
        layer.codes,
        layer.scales,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        group_size=layer.group_size,
    )


def _call_nax_e8p_fp16_sorted_steel_m64n128_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
) -> mx.array:
    del fixture
    return nax.nax_e8p_fp16_sorted_steel_m64n128_matmul(
        sorted_inputs.sorted_x,
        layer.codes,
        layer.scales,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        group_size=layer.group_size,
    )


def _call_nax_e8p_fp16_sorted_steel_m64n64t64_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
) -> mx.array:
    del fixture
    return nax.nax_e8p_fp16_sorted_steel_m64n64t64_matmul(
        sorted_inputs.sorted_x,
        layer.codes,
        layer.scales,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        group_size=layer.group_size,
    )


def _call_nax_e8p_fp16_sorted_steel_m32n64_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
) -> mx.array:
    del fixture
    return nax.nax_e8p_fp16_sorted_steel_m32n64_matmul(
        sorted_inputs.sorted_x,
        layer.codes,
        layer.scales,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        group_size=layer.group_size,
    )


def _call_nax_e8p_fp16_sorted_steel_m32n64t128_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
) -> mx.array:
    del fixture
    return nax.nax_e8p_fp16_sorted_steel_m32n64t128_matmul(
        sorted_inputs.sorted_x,
        layer.codes,
        layer.scales,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        group_size=layer.group_size,
    )


def _call_nax_e8p_fp16_sorted_steel_m32n128_raw(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
    sorted_inputs: SortedSteelInputs,
) -> mx.array:
    del fixture
    return nax.nax_e8p_fp16_sorted_steel_m32n128_matmul(
        sorted_inputs.sorted_x,
        layer.codes,
        layer.scales,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        group_size=layer.group_size,
    )


def _quantize_a8_per_group(x: mx.array, group_size: int) -> tuple[mx.array, mx.array]:
    tokens, input_dims = x.shape
    if input_dims % group_size != 0:
        raise ValueError("A8 quantization requires input_dims divisible by group_size")
    groups = input_dims // group_size
    grouped = x.astype(mx.float32).reshape((tokens, groups, group_size))
    max_abs = mx.max(mx.abs(grouped), axis=-1)
    scales = mx.maximum(max_abs / 127.0, mx.array(1e-6, dtype=mx.float32)).astype(mx.float16)
    quantized = mx.clip(mx.round(grouped / scales.astype(mx.float32)[..., None]), -127, 127).astype(
        mx.int8
    )
    return quantized.reshape((tokens, input_dims)), scales


def _call_nax_e8_int8_routed(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
) -> mx.array:
    flat_rhs = fixture.indices.reshape((-1,)).astype(mx.int32)
    flat_lhs = (mx.arange(flat_rhs.shape[0], dtype=mx.int32) // fixture.indices.shape[1]).astype(mx.int32)
    order = mx.argsort(flat_rhs)
    inverse_order = vq_switch._inverse_permutation(order)
    sorted_rhs = flat_rhs[order]
    sorted_lhs = flat_lhs[order]
    tile_descriptors = vq_switch._expert_block_tile_descriptors(
        sorted_rhs,
        num_experts=fixture.experts,
        route_tile=64,
    )
    x_q, x_scales = _quantize_a8_per_group(fixture.x, fixture.vq_group_size)
    routed = nax.nax_e8_int8_routed_matmul(
        x_q,
        x_scales,
        layer.codes,
        layer.scales,
        sorted_lhs,
        *tile_descriptors,
        layer.codebook,
        group_size=fixture.vq_group_size,
    )
    return routed[inverse_order].reshape((*fixture.indices.shape, fixture.output_dims))


def _sorted_route_metadata(fixture: ProjectionFixture) -> tuple[mx.array, mx.array, tuple[mx.array, mx.array, mx.array]]:
    flat_rhs = fixture.indices.reshape((-1,)).astype(mx.int32)
    flat_lhs = (mx.arange(flat_rhs.shape[0], dtype=mx.int32) // fixture.indices.shape[1]).astype(mx.int32)
    order = mx.argsort(flat_rhs)
    sorted_rhs = flat_rhs[order]
    sorted_lhs = flat_lhs[order]
    tile_descriptors = vq_switch._expert_block_tile_descriptors(
        sorted_rhs,
        num_experts=fixture.experts,
        route_tile=64,
    )
    return sorted_lhs, sorted_rhs, tile_descriptors


def _time_projection_call(run_once, *, iterations: int, warmup: int) -> tuple[float, mx.array]:
    for _ in range(warmup):
        mx.eval(run_once())
    start = time.perf_counter()
    output = None
    for _ in range(iterations):
        output = run_once()
        mx.eval(output)
    if output is None:
        raise AssertionError("projection benchmark did not run")
    return time.perf_counter() - start, output


def run_nax_int8_raw_ceiling(
    *,
    iterations: int = 50,
    warmup: int = 10,
    seed: int = 20260624,
) -> dict[str, Any]:
    """Compare the native E8 INT8 path against FP16 on one 64x64x64 route tile.

    This keeps activation quantization outside the prequantized INT8 timing so
    N6 can decide whether the native TensorOps ceiling is worth pursuing before
    optimizing the resident wrapper.
    """

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if warmup < 0:
        raise ValueError("warmup cannot be negative")
    if not nax.is_available():
        return {
            "benchmark": "nax_int8_raw_ceiling",
            "available": False,
            "reason": "native VQ NAX extension is not built",
        }

    fixture = build_projection_fixture(
        projection="gate_up",
        tokens=64,
        top_k=1,
        experts=1,
        input_dims=64,
        output_dims=64,
        mlx_group_size=64,
        vq_preferred_group_size=64,
        seed=seed,
    )
    layer = _vq_switch_linear(fixture)
    sorted_lhs, _sorted_rhs, tile_descriptors = _sorted_route_metadata(fixture)
    sorted_x = fixture.x[sorted_lhs].astype(mx.float16)
    x_q, x_scales = _quantize_a8_per_group(fixture.x, fixture.vq_group_size)
    mx.eval(sorted_x, x_q, x_scales, layer.codes, layer.scales, layer.codebook, *tile_descriptors)

    def run_fp16() -> mx.array:
        return nax.nax_e8_fp16_sorted_steel_matmul(
            sorted_x,
            layer.codes,
            layer.scales,
            *tile_descriptors,
            layer.codebook,
            group_size=fixture.vq_group_size,
        )

    def run_int8_prequantized() -> mx.array:
        return nax.nax_e8_int8_routed_matmul(
            x_q,
            x_scales,
            layer.codes,
            layer.scales,
            sorted_lhs,
            *tile_descriptors,
            layer.codebook,
            group_size=fixture.vq_group_size,
        )

    def run_int8_with_quant() -> mx.array:
        return _call_nax_e8_int8_routed(layer, fixture).reshape((-1, fixture.output_dims))

    before_vm = collect_vm_stat_counts()
    reset_mlx_peak_memory()
    fp16_elapsed, fp16_output = _time_projection_call(run_fp16, iterations=iterations, warmup=warmup)
    int8_elapsed, int8_output = _time_projection_call(
        run_int8_prequantized,
        iterations=iterations,
        warmup=warmup,
    )
    int8_full_elapsed, int8_full_output = _time_projection_call(
        run_int8_with_quant,
        iterations=iterations,
        warmup=warmup,
    )
    fp16_ms = fp16_elapsed * 1000.0 / iterations
    int8_ms = int8_elapsed * 1000.0 / iterations
    int8_full_ms = int8_full_elapsed * 1000.0 / iterations
    fp16_float = fp16_output.astype(mx.float32)
    int8_float = int8_output.astype(mx.float32)
    dot = float(mx.sum(fp16_float * int8_float).item())
    norm = float(mx.sqrt(mx.sum(fp16_float * fp16_float) * mx.sum(int8_float * int8_float)).item())
    metrics = collect_metric_snapshot(previous_vm_stat_counts=before_vm)
    return {
        "benchmark": "nax_int8_raw_ceiling",
        "available": True,
        "tile_geometry": "64x64x64",
        "tokens": fixture.tokens,
        "top_k": fixture.top_k,
        "input_dims": fixture.input_dims,
        "output_dims": fixture.output_dims,
        "experts": fixture.experts,
        "group_size": fixture.vq_group_size,
        "iterations": iterations,
        "warmup": warmup,
        "fp16_ms_per_iter": fp16_ms,
        "int8_prequantized_ms_per_iter": int8_ms,
        "int8_with_quant_ms_per_iter": int8_full_ms,
        "int8_prequantized_speedup_vs_fp16": None if int8_ms == 0 else fp16_ms / int8_ms,
        "int8_with_quant_speedup_vs_fp16": None if int8_full_ms == 0 else fp16_ms / int8_full_ms,
        "raw_ceiling_gate": "pass" if int8_ms > 0 and fp16_ms / int8_ms >= 1.7 else "stop",
        "cosine_vs_fp16": None if norm == 0.0 else dot / norm,
        "finite_fp16": bool(mx.all(mx.isfinite(fp16_float)).item()),
        "finite_int8": bool(mx.all(mx.isfinite(int8_float)).item()),
        "int8_full_checksum": float(mx.sum(int8_full_output.astype(mx.float32)).item()),
        "diagnostic_note": (
            "N6 raw-ceiling proxy: native fused E8 INT8 path over a single 64x64x64 route tile "
            "with A8 quantization excluded from the prequantized timing and measured separately."
        ),
        **metrics,
    }


def _call_vq_decode_to_scratch_gather_mm_routed(
    layer: QuantizedVQSwitchLinear,
    fixture: ProjectionFixture,
) -> mx.array:
    expanded = mx.expand_dims(fixture.x.astype(mx.bfloat16), (-2, -3))
    sorted_x, sorted_indices, inverse_order = _gather_sort(expanded, fixture.indices)
    active_experts = np.unique(np.array(sorted_indices, copy=False).astype(np.int32))
    remapped_rhs = np.searchsorted(active_experts, np.array(sorted_indices, copy=False)).astype(np.int32)
    codes_np = np.array(layer.codes, copy=False)
    scales_np = np.array(layer.scales, copy=False)
    dense_scratch = decode_weight_matrix(
        codes_np[active_experts],
        scales_np[active_experts],
        code_bits=layer.code_bits,
        codebook=np.array(layer.codebook, copy=False),
    )
    scratch_weight_t = mx.swapaxes(mx.array(dense_scratch).astype(mx.bfloat16), -1, -2)
    routed = mx.gather_mm(
        sorted_x,
        scratch_weight_t,
        rhs_indices=mx.array(remapped_rhs),
        sorted_indices=True,
    )
    return _scatter_unsort(routed, inverse_order, fixture.indices.shape).squeeze(-2)


def projection_variants_for_cli(
    variant: Literal[
        "dense_bf16",
        "mlx_gather_mm_bf16",
        "mlx_q2",
        "nax_e8_fp16",
        "nax_e8_fp16_sorted_steel",
        "nax_e8_fp16_sorted_steel_raw",
        "nax_e8p_fp16_sorted_steel",
        "nax_e8p_fp16_sorted_steel_raw",
        "nax_e8p_fp16_sorted_direct_reduce_raw",
        "nax_e8p_fp16_sorted_inline_b_raw",
        "nax_e8p_packed_rhs_sorted_tiled_raw",
        "nax_e8p_route_slot_codeword_stream_rhs_sorted_raw",
        "nax_e8p_route_slot_mma_codeword_tile_rhs_sorted_raw",
        "nax_e8p_active_route_tile_codeword_outer_product_rhs_sorted_raw",
        "nax_e8p_expert_cohort_codeword_broadcast_rhs_sorted_raw",
        "nax_e8p_route_batch_segmented_codeword_reduce_rhs_sorted_raw",
        "nax_e8p_token_cohort_codeword_stream_rhs_sorted_raw",
        "nax_e8p_token_cohort_mma_codeword_tile_rhs_sorted_raw",
        "nax_e8p_output_stationary_codeword_tile_rhs_sorted_raw",
        "nax_e8p_input_stationary_codeword_tile_rhs_sorted_raw",
        "nax_e8p_expert_kblock_codeword_factor_reuse_rhs_sorted_raw",
        "nax_e8p_route_codeword_lut_accumulate_rhs_sorted_raw",
        "nax_e8p_rowwise_codeword_tile_accumulate_rhs_sorted_raw",
        "nax_e8p_output_tile_local_codeword_lut_rhs_sorted_raw",
        "nax_e8p_route_microtile_codeword_block_reduce_rhs_sorted_raw",
        "nax_e8p_kblock_wavefront_codeword_scan_rhs_sorted_raw",
        "nax_e8p_token_route_output_stripe_pipeline_rhs_sorted_raw",
        "nax_e8p_expert_kblock_scale_slot_stream_rhs_sorted_raw",
        "nax_e8p_scale_group_route_block_reduce_rhs_sorted_raw",
        "nax_e8p_route_block_output_group_stream_rhs_sorted_raw",
        "nax_e8p_output_group_pretransposed_codeword_stream_rhs_sorted_raw",
        "nax_e8p_kblock_output_group_route_fused_stream_rhs_sorted_raw",
        "nax_e8p_route_tile_output_swizzle_stream_rhs_sorted_raw",
        "nax_e8p_split_byte_rhs_sorted_tiled_raw",
        "nax_e8p_split_byte_factor_reuse_rhs_sorted_tiled_raw",
        "nax_e8p_split_byte_factor_reuse_rhs_sorted_native_raw",
        "nax_e8p_component_stream_rhs_sorted_scalar_raw",
        "nax_e8p_component_stream_rhs_sorted_partial_raw",
        "nax_e8p_component_stream_rhs_sorted_tensorops_raw",
        "nax_e8p_component_stream_rhs_sorted_shared_decode_raw",
        "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_native_raw",
        "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_raw",
        "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_raw",
        "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_raw",
        "nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_raw",
        "nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_raw",
        "nax_e8p_sign_plane_abs_index_rhs_sorted_native_raw",
        "nax_e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_raw",
        "nax_e8p_sign_nibble_micro_lut_rhs_sorted_native_raw",
        "nax_e8p_packed_rhs_sorted_tiled_m128_raw",
        "nax_e8p_packed_rhs_sorted_tiled_k128_raw",
        "nax_e8p_predecoded_fp16_raw",
        "nax_e8p_fp16_sorted_steel_gs352_raw",
        "nax_e8p_fp16_sorted_steel_lut_raw",
        "nax_e8p_fp16_sorted_steel_tgcb_raw",
        "nax_e8p_fp16_sorted_steel_tgscale_raw",
        "nax_e8p_fp16_sorted_steel_tgcb_tgscale_raw",
        "nax_e8p_fp16_sorted_steel_tgcb_hoist_raw",
        "nax_e8p_fp16_sorted_steel_bk128_raw",
        "nax_e8p_fp16_sorted_steel_m128n32_raw",
        "nax_e8p_fp16_sorted_steel_m64n128_raw",
        "nax_e8p_fp16_sorted_steel_m32n64_raw",
        "nax_e8p_fp16_sorted_steel_m64n64t64_raw",
        "nax_e8p_fp16_sorted_steel_m32n64t128_raw",
        "nax_e8p_fp16_sorted_steel_m32n128_raw",
        "nax_e8_fp16_steel",
        "nax_e8_int8",
        "nax_predecoded_fp16",
        "vq_e1",
        "vq_decode_direct_candidates",
        "m1_decode_gate",
        "nax_audit",
        "all",
    ],
    *,
    decompose_vq: bool,
) -> list[ProjectionVariant]:
    if variant == "all":
        requested: list[ProjectionVariant] = ["dense_bf16", "mlx_q2", "vq_e1"]
    elif variant == "m1_decode_gate":
        requested = ["mlx_q2", "vq_decode_direct_candidates"]
    elif variant == "nax_audit":
        requested = ["mlx_q2", "mlx_gather_mm_bf16", "vq_e1"]
    else:
        requested = [variant]
    if not decompose_vq:
        return requested

    expanded: list[ProjectionVariant] = []
    for name in requested:
        if name == "vq_e1":
            expanded.extend(
                [
                    "vq_current",
                    "vq_decode_only",
                    "vq_matmul_only",
                    "vq_decode_direct_candidates",
                    "vq_decode_to_scratch_gather_mm",
                ]
            )
        else:
            expanded.append(name)
    return expanded


def _as_mlx(array: mx.array | np.ndarray) -> mx.array:
    if isinstance(array, mx.array):
        return array
    return mx.array(array)


def _default_codebook(code_bits: Literal[8, 16]) -> np.ndarray:
    if code_bits == 8:
        return e8_1bit_packed()
    return e8p_packed_abs_grid()


def gather_vqmm_diagnostic_kernel(
    x: mx.array | np.ndarray,
    codes: mx.array | np.ndarray,
    scales: mx.array | np.ndarray,
    codebook: mx.array | np.ndarray | None,
    rhs_indices: mx.array | np.ndarray,
    *,
    input_dims: int,
    output_dims: int,
    group_size: int = 512,
    code_bits: Literal[8, 16] = 8,
    codebook_duplication: Literal[1, 4, 8] = 1,
    mode: Literal["decode_only", "matmul_only"] = "decode_only",
) -> mx.array:
    x_mx = _as_mlx(x)
    codes_mx = _as_mlx(codes)
    scales_mx = _as_mlx(scales)
    rhs_mx = _as_mlx(rhs_indices)
    codebook_mx = mx.array(_default_codebook(code_bits)) if codebook is None else _as_mlx(codebook)

    diagnostic_mode = {"decode_only": 1, "matmul_only": 2}[mode]
    output_dtype = _qmv_output_dtype(x_mx.dtype)
    tokens, top_k = rhs_mx.shape
    outputs = _GATHER_VQMM_DIAGNOSTIC_KERNEL(
        inputs=[x_mx, codes_mx, scales_mx, codebook_mx, rhs_mx],
        template=[
            ("CODE_BITS", code_bits),
            ("CODEBOOK_DUP", codebook_duplication),
            ("DIAGNOSTIC_MODE", diagnostic_mode),
            ("OUT_T", output_dtype),
        ],
        grid=(256, output_dims, tokens * top_k),
        threadgroup=(256, 1, 1),
        output_shapes=[(tokens, top_k, output_dims)],
        output_dtypes=[output_dtype],
    )
    return outputs[0]


def decode_bandwidth_floor_report(
    *,
    input_dims: int,
    output_dims: int,
    top_k: int,
    measured_gb_s: float,
    code_bits: Literal[8, 16] = 8,
    group_size: int = 512,
    scale_bytes: int = 2,
) -> dict[str, float | int]:
    if input_dims <= 0 or output_dims <= 0 or top_k <= 0:
        raise ValueError("input_dims, output_dims, and top_k must be positive")
    if input_dims % 8 != 0:
        raise ValueError("input_dims must be divisible by 8")
    if input_dims % group_size != 0:
        raise ValueError("input_dims must be divisible by group_size")
    if code_bits not in (8, 16):
        raise ValueError("code_bits must be 8 or 16")
    if measured_gb_s <= 0:
        raise ValueError("measured_gb_s must be positive")

    code_bytes = output_dims * (input_dims // 8) * (code_bits // 8)
    scales_bytes = output_dims * (input_dims // group_size) * scale_bytes
    read_bytes = top_k * (code_bytes + scales_bytes)
    return {
        "read_bytes_m1": read_bytes,
        "measured_gb_s": measured_gb_s,
        "bandwidth_floor_ms": read_bytes / (measured_gb_s * 1_000_000_000) * 1000.0,
        "top_k": top_k,
        "input_dims": input_dims,
        "output_dims": output_dims,
        "group_size": group_size,
        "code_bits": code_bits,
    }


def _vq_switch_linear(fixture: ProjectionFixture) -> QuantizedVQSwitchLinear:
    layer = QuantizedVQSwitchLinear.from_weights(
        fixture.dense_weight.astype(mx.float32),
        group_size=fixture.vq_group_size,
        code_bits=fixture.vq_code_bits,
    )
    mx.eval(layer.parameters())
    return layer


def _vq_decode_direct_candidates(
    fixture: ProjectionFixture,
    layer: QuantizedVQSwitchLinear,
) -> mx.array:
    if fixture.tokens != 1:
        raise ValueError("vq_decode_direct_candidates is an M=1-only benchmark variant")

    if fixture.input_dims > fixture.output_dims:
        return gather_vqmm_m1_rowpair_kernel_unchecked(
            fixture.x,
            layer.codes,
            layer.scales,
            layer.codebook,
            fixture.indices,
            output_dims=fixture.output_dims,
        )

    return gather_vqmm_m1_kernel_unchecked(
        fixture.x,
        layer.codes,
        layer.scales,
        layer.codebook,
        fixture.indices,
        output_dims=fixture.output_dims,
        code_bits=layer.code_bits,
        rows_per_threadgroup=m1_rows_per_threadgroup(fixture.input_dims, fixture.output_dims),
        use_threadgroup_codebook=m1_use_threadgroup_codebook(fixture.input_dims, fixture.output_dims),
        use_decoded_codebook=m1_use_decoded_codebook(fixture.input_dims, fixture.output_dims),
    )


def _variant_callable(fixture: ProjectionFixture, variant: ProjectionVariant):
    if variant == "dense_bf16":
        layer = _dense_switch_linear(fixture)
        return lambda: _call_dense_or_mlx(layer, fixture)
    if variant == "mlx_gather_mm_bf16":
        layer = _dense_switch_linear(fixture)
        return lambda: _call_mlx_gather_mm_bf16_routed(layer.weight, fixture)
    if variant == "nax_predecoded_fp16":
        layer = _dense_switch_linear(fixture)
        weight_t = mx.swapaxes(layer.weight.astype(mx.float16), -1, -2)
        mx.eval(weight_t)
        return lambda: _call_nax_predecoded_fp16_routed(weight_t, fixture)
    if variant == "nax_e8_fp16":
        layer = _vq_switch_linear(fixture)
        return lambda: _call_nax_e8_fp16_routed(layer, fixture)
    if variant == "nax_e8_fp16_sorted_steel":
        layer = _vq_switch_linear(fixture)
        return lambda: _call_nax_e8_fp16_sorted_steel_routed(layer, fixture)
    if variant == "nax_e8p_fp16_sorted_steel":
        layer = _vq_switch_linear(fixture)
        return lambda: _call_nax_e8p_fp16_sorted_steel_routed(layer, fixture)
    if variant == "nax_e8_fp16_steel":
        layer = _vq_switch_linear(fixture)
        return lambda: _call_nax_e8_fp16_steel_routed(layer, fixture)
    if variant == "nax_e8_int8":
        layer = _vq_switch_linear(fixture)
        return lambda: _call_nax_e8_int8_routed(layer, fixture)
    if variant == "mlx_q2":
        dense = _dense_switch_linear(fixture)
        layer = dense.to_quantized(
            group_size=fixture.mlx_group_size,
            bits=2,
            mode="affine",
        )
        mx.eval(layer.parameters())
        return lambda: _call_mlx_q2_routed(layer, fixture)
    if variant in ("vq_e1", "vq_current"):
        layer = _vq_switch_linear(fixture)
        return lambda: layer(fixture.x, fixture.indices)
    if variant == "vq_decode_only":
        layer = _vq_switch_linear(fixture)
        return lambda: gather_vqmm_diagnostic_kernel(
            fixture.x,
            layer.codes,
            layer.scales,
            layer.codebook,
            fixture.indices,
            input_dims=fixture.input_dims,
            output_dims=fixture.output_dims,
            group_size=fixture.vq_group_size,
            code_bits=layer.code_bits,
            codebook_duplication=layer.gather_codebook_duplication,
            mode="decode_only",
        )
    if variant == "vq_matmul_only":
        layer = _vq_switch_linear(fixture)
        return lambda: gather_vqmm_diagnostic_kernel(
            fixture.x,
            layer.codes,
            layer.scales,
            layer.codebook,
            fixture.indices,
            input_dims=fixture.input_dims,
            output_dims=fixture.output_dims,
            group_size=fixture.vq_group_size,
            code_bits=layer.code_bits,
            codebook_duplication=layer.gather_codebook_duplication,
            mode="matmul_only",
        )
    if variant == "vq_decode_direct_candidates":
        layer = _vq_switch_linear(fixture)
        return lambda: _vq_decode_direct_candidates(fixture, layer)
    if variant == "vq_decode_to_scratch_gather_mm":
        layer = _vq_switch_linear(fixture)
        return lambda: _call_vq_decode_to_scratch_gather_mm_routed(layer, fixture)
    raise ValueError(f"unknown projection variant {variant!r}")


def run_projection_variant(
    fixture: ProjectionFixture,
    *,
    variant: ProjectionVariant,
    iterations: int = 5,
    warmup: int = 2,
) -> dict[str, Any]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if warmup < 0:
        raise ValueError("warmup cannot be negative")
    run_once = _variant_callable(fixture, variant)
    for _ in range(warmup):
        mx.eval(run_once())

    before_vm = collect_vm_stat_counts()
    reset_mlx_peak_memory()
    start = time.perf_counter()
    output = None
    for _ in range(iterations):
        output = run_once()
        mx.eval(output)
    elapsed = time.perf_counter() - start
    if output is None:
        raise AssertionError("projection benchmark did not run")
    output_float = output.astype(mx.float32)
    finite = bool(mx.all(mx.isfinite(output_float)).item())
    checksum = float(mx.sum(output_float).item())
    metrics = collect_metric_snapshot(previous_vm_stat_counts=before_vm)
    diagnostic_notes = {
        "vq_decode_only": (
            "benchmark-only proxy preserving current route/output-row/codeword decode loop, "
            "codebook staging, scale reads, reduction, and checksum output; not additive with matmul_only"
        ),
        "vq_matmul_only": (
            "benchmark-only proxy preserving current route/output-row/codeword, scale, activation, "
            "reduction, and output work with deterministic pseudo weights instead of codebook decode; "
            "not additive with decode_only"
        ),
        "vq_decode_direct_candidates": (
            "M=1 benchmark-only direct-route candidate using shape-selected row packing: Air gate/up uses "
            "row-pair shared-activation packing, while down uses rows_per_threadgroup=32 with vectorized "
            "half4 decoded threadgroup codebook; mirrors the production auto M=1 direct path"
        ),
        "vq_decode_to_scratch_gather_mm": (
            "benchmark-only active-expert probe that decodes VQ weights into bf16 scratch and calls "
            "mx.gather_mm on resident-style sorted routes; diagnostic only, not a final storage or "
            "runtime contract"
        ),
        "nax_predecoded_fp16": (
            "benchmark-only native-extension scaffold path over predecoded fp16 weights and "
            "resident-style sorted routes; diagnostic N3 proof, not the final fused E8 primitive"
        ),
        "nax_e8_fp16": (
            "benchmark-only native fused E8 FP16 TensorOps path over resident-style sorted route "
            "blocks; diagnostic N4 proof, not a resident default"
        ),
        "nax_e8_fp16_sorted_steel": (
            "benchmark-only native fused E8 FP16 TensorOps path using contiguous sorted route "
            "activations plus MLX Steel NAXTile/tile_matmad helpers; structural N4 probe, not a resident default"
        ),
        "nax_e8p_fp16_sorted_steel": (
            "benchmark-only native fused E8P FP16 TensorOps path using contiguous sorted route "
            "activations plus MLX Steel NAXTile/tile_matmad helpers; Track B 16-bit prefill probe"
        ),
        "nax_e8_fp16_steel": (
            "benchmark-only native fused E8 FP16 TensorOps path using MLX Steel NAXTile/tile_matmad "
            "helpers over resident-style sorted route blocks; structural N4 probe, not a resident default"
        ),
        "nax_e8_int8": (
            "benchmark-only native fused E8 INT8 TensorOps path with per-group A8 activation "
            "quantization; diagnostic N6 proof, not a resident default"
        ),
    }
    extra: dict[str, object] = {}
    if variant in diagnostic_notes:
        extra["diagnostic_phase"] = variant.removeprefix("vq_")
        extra["diagnostic_note"] = diagnostic_notes[variant]
    return {
        "benchmark": "glm45_air_hot_projection_kernel",
        "projection": fixture.projection,
        "variant": variant,
        "tokens": fixture.tokens,
        "top_k": fixture.top_k,
        "experts": fixture.experts,
        "input_dims": fixture.input_dims,
        "output_dims": fixture.output_dims,
        "mlx_group_size": fixture.mlx_group_size if variant == "mlx_q2" else None,
        "mlx_sorted_indices": (variant == "mlx_q2" and fixture.indices.size >= 64)
        or variant
        in (
            "mlx_gather_mm_bf16",
            "nax_predecoded_fp16",
            "nax_e8_fp16",
            "nax_e8_fp16_sorted_steel",
            "nax_e8p_fp16_sorted_steel",
            "nax_e8_fp16_steel",
            "nax_e8_int8",
        ),
        "vq_group_size": (
            fixture.vq_group_size
            if str(variant).startswith("vq_")
            or variant
            in (
                "nax_e8_fp16",
                "nax_e8_fp16_sorted_steel",
                "nax_e8p_fp16_sorted_steel",
                "nax_e8_fp16_steel",
                "nax_e8_int8",
            )
            else None
        ),
        "vq_code_bits": (
            fixture.vq_code_bits
            if str(variant).startswith("vq_")
            or variant
            in (
                "nax_e8_fp16",
                "nax_e8_fp16_sorted_steel",
                "nax_e8p_fp16_sorted_steel",
                "nax_e8_fp16_steel",
                "nax_e8_int8",
            )
            else None
        ),
        "iterations": iterations,
        "warmup": warmup,
        "elapsed_seconds": elapsed,
        "ms_per_iter": elapsed * 1000.0 / iterations,
        "output_shape": list(output.shape),
        "finite_output": finite,
        "checksum": checksum,
        **metrics,
        **extra,
    }


def _loaded_artifact_variant_callable(
    fixture: ProjectionFixture,
    layer: QuantizedVQSwitchLinear,
    variant: ProjectionVariant,
):
    if variant in ("vq_e1", "vq_current"):
        return lambda: layer(fixture.x, fixture.indices)
    if variant == "nax_e8_fp16_sorted_steel":
        if layer.code_bits != 8:
            raise ValueError("nax_e8_fp16_sorted_steel artifact benchmark requires code_bits=8")
        return lambda: _call_nax_e8_fp16_sorted_steel_routed(layer, fixture)
    if variant == "nax_e8_fp16_sorted_steel_raw":
        if layer.code_bits != 8:
            raise ValueError("nax_e8_fp16_sorted_steel_raw artifact benchmark requires code_bits=8")
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        return lambda: _call_nax_e8_fp16_sorted_steel_raw(layer, fixture, sorted_inputs)
    if variant == "nax_e8p_fp16_sorted_steel":
        if layer.code_bits != 16:
            raise ValueError("nax_e8p_fp16_sorted_steel artifact benchmark requires code_bits=16")
        return lambda: _call_nax_e8p_fp16_sorted_steel_routed(layer, fixture)
    if variant == "nax_e8p_fp16_sorted_steel_raw":
        if layer.code_bits != 16:
            raise ValueError("nax_e8p_fp16_sorted_steel_raw artifact benchmark requires code_bits=16")
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        return lambda: _call_nax_e8p_fp16_sorted_steel_raw(layer, fixture, sorted_inputs)
    if variant == "nax_e8p_fp16_sorted_direct_reduce_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_fp16_sorted_direct_reduce_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        return lambda: _call_nax_e8p_fp16_sorted_direct_reduce_raw(
            layer, fixture, sorted_inputs
        )
    if variant == "nax_e8p_fp16_sorted_inline_b_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_fp16_sorted_inline_b_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        return lambda: _call_nax_e8p_fp16_sorted_inline_b_raw(layer, fixture, sorted_inputs)
    if variant == "nax_e8p_packed_rhs_sorted_tiled_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_packed_rhs_sorted_tiled_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        packed_rhs = _prepare_e8p_packed_rhs_tiles(layer)
        return lambda: _call_nax_e8p_packed_rhs_sorted_tiled_raw(
            layer, fixture, sorted_inputs, packed_rhs
        )
    if variant == "nax_e8p_route_slot_codeword_stream_rhs_sorted_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_route_slot_codeword_stream_rhs_sorted_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        packed_rhs = _prepare_e8p_packed_rhs_tiles(layer)
        return lambda: _call_nax_e8p_route_slot_codeword_stream_rhs_sorted_raw(
            layer, fixture, sorted_inputs, packed_rhs
        )
    if variant == "nax_e8p_route_slot_mma_codeword_tile_rhs_sorted_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_route_slot_mma_codeword_tile_rhs_sorted_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        packed_rhs = _prepare_e8p_packed_rhs_tiles(layer)
        return lambda: _call_nax_e8p_route_slot_mma_codeword_tile_rhs_sorted_raw(
            layer, fixture, sorted_inputs, packed_rhs
        )
    if variant == "nax_e8p_active_route_tile_codeword_outer_product_rhs_sorted_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_active_route_tile_codeword_outer_product_rhs_sorted_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        packed_rhs = _prepare_e8p_packed_rhs_tiles(layer)
        return lambda: _call_nax_e8p_active_route_tile_codeword_outer_product_rhs_sorted_raw(
            layer, fixture, sorted_inputs, packed_rhs
        )
    if variant == "nax_e8p_expert_cohort_codeword_broadcast_rhs_sorted_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_expert_cohort_codeword_broadcast_rhs_sorted_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        packed_rhs = _prepare_e8p_packed_rhs_tiles(layer)
        return lambda: _call_nax_e8p_expert_cohort_codeword_broadcast_rhs_sorted_raw(
            layer, fixture, sorted_inputs, packed_rhs
        )
    if variant == "nax_e8p_route_batch_segmented_codeword_reduce_rhs_sorted_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_route_batch_segmented_codeword_reduce_rhs_sorted_raw artifact benchmark requires code_bits=16"
            )
        route_batch_size = 8
        sorted_inputs = _prepare_sorted_steel_inputs(fixture, route_tile=route_batch_size)
        packed_rhs = _prepare_e8p_packed_rhs_tiles(layer)
        return lambda: _call_nax_e8p_route_batch_segmented_codeword_reduce_rhs_sorted_raw(
            layer,
            fixture,
            sorted_inputs,
            packed_rhs,
            route_batch_size=route_batch_size,
        )
    if variant == "nax_e8p_token_cohort_codeword_stream_rhs_sorted_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_token_cohort_codeword_stream_rhs_sorted_raw artifact benchmark requires code_bits=16"
            )
        token_cohort_size = 8
        sorted_inputs = _prepare_sorted_steel_inputs(fixture, route_tile=token_cohort_size)
        packed_rhs = _prepare_e8p_packed_rhs_tiles(layer)
        return lambda: _call_nax_e8p_token_cohort_codeword_stream_rhs_sorted_raw(
            layer,
            fixture,
            sorted_inputs,
            packed_rhs,
            token_cohort_size=token_cohort_size,
        )
    if variant == "nax_e8p_token_cohort_mma_codeword_tile_rhs_sorted_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_token_cohort_mma_codeword_tile_rhs_sorted_raw artifact benchmark requires code_bits=16"
            )
        token_cohort_size = 8
        sorted_inputs = _prepare_sorted_steel_inputs(fixture, route_tile=token_cohort_size)
        packed_rhs = _prepare_e8p_packed_rhs_tiles(layer)
        return lambda: _call_nax_e8p_token_cohort_mma_codeword_tile_rhs_sorted_raw(
            layer,
            fixture,
            sorted_inputs,
            packed_rhs,
            token_cohort_size=token_cohort_size,
        )
    if variant == "nax_e8p_output_stationary_codeword_tile_rhs_sorted_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_output_stationary_codeword_tile_rhs_sorted_raw artifact benchmark requires code_bits=16"
            )
        route_batch_size = 8
        sorted_inputs = _prepare_sorted_steel_inputs(fixture, route_tile=route_batch_size)
        packed_rhs = _prepare_e8p_packed_rhs_tiles(layer)
        return lambda: _call_nax_e8p_output_stationary_codeword_tile_rhs_sorted_raw(
            layer,
            fixture,
            sorted_inputs,
            packed_rhs,
            route_batch_size=route_batch_size,
        )
    if variant == "nax_e8p_input_stationary_codeword_tile_rhs_sorted_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_input_stationary_codeword_tile_rhs_sorted_raw artifact benchmark requires code_bits=16"
            )
        route_batch_size = 8
        sorted_inputs = _prepare_sorted_steel_inputs(fixture, route_tile=route_batch_size)
        packed_rhs = _prepare_e8p_packed_rhs_tiles(layer)
        return lambda: _call_nax_e8p_input_stationary_codeword_tile_rhs_sorted_raw(
            layer,
            fixture,
            sorted_inputs,
            packed_rhs,
            route_batch_size=route_batch_size,
        )
    if variant == "nax_e8p_expert_kblock_codeword_factor_reuse_rhs_sorted_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_expert_kblock_codeword_factor_reuse_rhs_sorted_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        packed_rhs = _prepare_e8p_expert_kblock_factor_reuse_rhs_tiles(layer)
        return lambda: _call_nax_e8p_expert_kblock_factor_reuse_rhs_sorted_native_raw(
            layer, fixture, sorted_inputs, packed_rhs
        )
    if variant == "nax_e8p_route_codeword_lut_accumulate_rhs_sorted_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_route_codeword_lut_accumulate_rhs_sorted_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        packed_rhs = _prepare_e8p_packed_rhs_tiles(layer)
        return lambda: _call_nax_e8p_route_codeword_lut_accumulate_rhs_sorted_raw(
            layer, fixture, sorted_inputs, packed_rhs
        )
    if variant == "nax_e8p_rowwise_codeword_tile_accumulate_rhs_sorted_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_rowwise_codeword_tile_accumulate_rhs_sorted_raw artifact benchmark requires code_bits=16"
            )
        route_microtile_size = 8
        sorted_inputs = _prepare_sorted_steel_inputs(
            fixture,
            route_tile=route_microtile_size,
        )
        packed_rhs = _prepare_e8p_packed_rhs_tiles(layer)
        return lambda: _call_nax_e8p_rowwise_codeword_tile_accumulate_rhs_sorted_raw(
            layer,
            fixture,
            sorted_inputs,
            packed_rhs,
            route_microtile_size=route_microtile_size,
        )
    if variant == "nax_e8p_output_tile_local_codeword_lut_rhs_sorted_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_output_tile_local_codeword_lut_rhs_sorted_raw artifact benchmark requires code_bits=16"
            )
        route_microtile_size = 8
        sorted_inputs = _prepare_sorted_steel_inputs(
            fixture,
            route_tile=route_microtile_size,
        )
        packed_rhs = _prepare_e8p_packed_rhs_tiles(layer)
        return lambda: _call_nax_e8p_output_tile_local_codeword_lut_rhs_sorted_raw(
            layer,
            fixture,
            sorted_inputs,
            packed_rhs,
            route_microtile_size=route_microtile_size,
        )
    if variant == "nax_e8p_route_microtile_codeword_block_reduce_rhs_sorted_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_route_microtile_codeword_block_reduce_rhs_sorted_raw artifact benchmark requires code_bits=16"
            )
        route_microtile_size = 8
        sorted_inputs = _prepare_sorted_steel_inputs(
            fixture,
            route_tile=route_microtile_size,
        )
        packed_rhs = _prepare_e8p_packed_rhs_tiles(layer)
        return lambda: _call_nax_e8p_route_microtile_codeword_block_reduce_rhs_sorted_raw(
            layer,
            fixture,
            sorted_inputs,
            packed_rhs,
            route_microtile_size=route_microtile_size,
        )
    if variant == "nax_e8p_kblock_wavefront_codeword_scan_rhs_sorted_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_kblock_wavefront_codeword_scan_rhs_sorted_raw artifact benchmark requires code_bits=16"
            )
        route_microtile_size = 8
        sorted_inputs = _prepare_sorted_steel_inputs(
            fixture,
            route_tile=route_microtile_size,
        )
        packed_rhs = _prepare_e8p_packed_rhs_tiles(layer)
        return lambda: _call_nax_e8p_kblock_wavefront_codeword_scan_rhs_sorted_raw(
            layer,
            fixture,
            sorted_inputs,
            packed_rhs,
            route_microtile_size=route_microtile_size,
        )
    if variant == "nax_e8p_token_route_output_stripe_pipeline_rhs_sorted_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_token_route_output_stripe_pipeline_rhs_sorted_raw artifact benchmark requires code_bits=16"
            )
        max_routes_per_token = 8
        sorted_inputs = _prepare_sorted_steel_inputs(
            fixture,
            route_tile=max_routes_per_token,
        )
        packed_rhs = _prepare_e8p_packed_rhs_tiles(layer)
        return lambda: _call_nax_e8p_token_route_output_stripe_pipeline_rhs_sorted_raw(
            layer,
            fixture,
            sorted_inputs,
            packed_rhs,
            max_routes_per_token=max_routes_per_token,
        )
    if variant == "nax_e8p_expert_kblock_scale_slot_stream_rhs_sorted_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_expert_kblock_scale_slot_stream_rhs_sorted_raw artifact benchmark requires code_bits=16"
            )
        route_tile_size = 8
        sorted_inputs = _prepare_sorted_steel_inputs(
            fixture,
            route_tile=route_tile_size,
        )
        packed_rhs = _prepare_e8p_packed_rhs_tiles(layer)
        return lambda: _call_nax_e8p_expert_kblock_scale_slot_stream_rhs_sorted_raw(
            layer,
            fixture,
            sorted_inputs,
            packed_rhs,
        )
    if variant == "nax_e8p_scale_group_route_block_reduce_rhs_sorted_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_scale_group_route_block_reduce_rhs_sorted_raw artifact benchmark requires code_bits=16"
            )
        route_tile_size = 8
        route_block_size = 8
        sorted_inputs = _prepare_sorted_steel_inputs(
            fixture,
            route_tile=route_tile_size,
        )
        packed_rhs = _prepare_e8p_packed_rhs_tiles(layer)
        return lambda: _call_nax_e8p_scale_group_route_block_reduce_rhs_sorted_raw(
            layer,
            fixture,
            sorted_inputs,
            packed_rhs,
            route_block_size=route_block_size,
        )
    if variant == "nax_e8p_route_block_output_group_stream_rhs_sorted_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_route_block_output_group_stream_rhs_sorted_raw artifact benchmark requires code_bits=16"
            )
        route_tile_size = 8
        route_block_size = 8
        sorted_inputs = _prepare_sorted_steel_inputs(
            fixture,
            route_tile=route_tile_size,
        )
        packed_rhs = _prepare_e8p_packed_rhs_tiles(layer)
        return lambda: _call_nax_e8p_route_block_output_group_stream_rhs_sorted_raw(
            layer,
            fixture,
            sorted_inputs,
            packed_rhs,
            route_block_size=route_block_size,
        )
    if variant == "nax_e8p_output_group_pretransposed_codeword_stream_rhs_sorted_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_output_group_pretransposed_codeword_stream_rhs_sorted_raw artifact benchmark requires code_bits=16"
            )
        route_tile_size = 8
        route_block_size = 8
        sorted_inputs = _prepare_sorted_steel_inputs(
            fixture,
            route_tile=route_tile_size,
        )
        packed_rhs = _prepare_e8p_packed_rhs_tiles(layer)
        return lambda: _call_nax_e8p_output_group_pretransposed_codeword_stream_rhs_sorted_raw(
            layer,
            fixture,
            sorted_inputs,
            packed_rhs,
            route_block_size=route_block_size,
        )
    if variant == "nax_e8p_kblock_output_group_route_fused_stream_rhs_sorted_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_kblock_output_group_route_fused_stream_rhs_sorted_raw artifact benchmark requires code_bits=16"
            )
        route_tile_size = 8
        route_block_size = 8
        sorted_inputs = _prepare_sorted_steel_inputs(
            fixture,
            route_tile=route_tile_size,
        )
        packed_rhs = _prepare_e8p_packed_rhs_tiles(layer)
        return lambda: _call_nax_e8p_kblock_output_group_route_fused_stream_rhs_sorted_raw(
            layer,
            fixture,
            sorted_inputs,
            packed_rhs,
            route_block_size=route_block_size,
        )
    if variant == "nax_e8p_route_tile_output_swizzle_stream_rhs_sorted_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_route_tile_output_swizzle_stream_rhs_sorted_raw artifact benchmark requires code_bits=16"
            )
        route_tile_size = 8
        sorted_inputs = _prepare_sorted_steel_inputs(
            fixture,
            route_tile=route_tile_size,
        )
        packed_rhs = _prepare_e8p_packed_rhs_tiles(layer)
        return lambda: _call_nax_e8p_route_tile_output_swizzle_stream_rhs_sorted_raw(
            layer,
            fixture,
            sorted_inputs,
            packed_rhs,
        )
    if variant == "nax_e8p_split_byte_rhs_sorted_tiled_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_split_byte_rhs_sorted_tiled_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        packed_rhs = _prepare_e8p_split_byte_rhs_tiles(layer)
        return lambda: _call_nax_e8p_split_byte_rhs_sorted_tiled_raw(
            layer, fixture, sorted_inputs, packed_rhs
        )
    if variant == "nax_e8p_split_byte_factor_reuse_rhs_sorted_tiled_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_split_byte_factor_reuse_rhs_sorted_tiled_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        packed_rhs = _prepare_e8p_split_byte_factor_reuse_rhs_tiles(layer)
        return lambda: _call_nax_e8p_split_byte_factor_reuse_rhs_sorted_tiled_raw(
            layer, fixture, sorted_inputs, packed_rhs
        )
    if variant == "nax_e8p_split_byte_factor_reuse_rhs_sorted_native_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_split_byte_factor_reuse_rhs_sorted_native_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        packed_rhs = _prepare_e8p_split_byte_factor_reuse_rhs_tiles(layer)
        return lambda: _call_nax_e8p_split_byte_factor_reuse_rhs_sorted_native_raw(
            layer, fixture, sorted_inputs, packed_rhs
        )
    if variant == "nax_e8p_component_stream_rhs_sorted_scalar_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_component_stream_rhs_sorted_scalar_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        packed_rhs = _prepare_e8p_component_stream_rhs_tiles(layer)
        return lambda: _call_nax_e8p_component_stream_rhs_sorted_scalar_raw(
            layer, fixture, sorted_inputs, packed_rhs
        )
    if variant == "nax_e8p_component_stream_rhs_sorted_partial_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_component_stream_rhs_sorted_partial_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        packed_rhs = _prepare_e8p_component_stream_rhs_tiles(layer)
        return lambda: _call_nax_e8p_component_stream_rhs_sorted_partial_raw(
            layer,
            fixture,
            _compact_nonempty_sorted_tile_descriptors(sorted_inputs),
            packed_rhs,
        )
    if variant == "nax_e8p_component_stream_rhs_sorted_tensorops_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_component_stream_rhs_sorted_tensorops_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        packed_rhs = _prepare_e8p_component_stream_rhs_tiles(layer)
        return lambda: _call_nax_e8p_component_stream_rhs_sorted_tensorops_raw(
            layer,
            fixture,
            _compact_nonempty_sorted_tile_descriptors(sorted_inputs),
            packed_rhs,
        )
    if variant == "nax_e8p_component_stream_rhs_sorted_shared_decode_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_component_stream_rhs_sorted_shared_decode_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        packed_rhs = _prepare_e8p_component_stream_rhs_tiles(layer)
        return lambda: _call_nax_e8p_component_stream_rhs_sorted_shared_decode_raw(
            layer,
            fixture,
            _compact_nonempty_sorted_tile_descriptors(sorted_inputs),
            packed_rhs,
        )
    if variant == "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_native_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_native_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        packed_rhs = _prepare_e8p_expert_kblock_factor_reuse_rhs_tiles(layer)
        return lambda: _call_nax_e8p_expert_kblock_factor_reuse_rhs_sorted_native_raw(
            layer, fixture, sorted_inputs, packed_rhs
        )
    if variant == "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        packed_rhs = _prepare_e8p_expert_kblock_factor_reuse_rhs_tiles(layer)
        return lambda: _call_nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_raw(
            layer, fixture, sorted_inputs, packed_rhs
        )
    if variant == "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        packed_rhs = _prepare_e8p_split_byte_factor_reuse_rhs_tiles(layer)
        return lambda: _call_nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_raw(
            layer, fixture, sorted_inputs, packed_rhs
        )
    if variant == "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        packed_rhs = _prepare_e8p_split_byte_factor_reuse_rhs_tiles(layer)
        return lambda: _call_nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_raw(
            layer, fixture, sorted_inputs, packed_rhs
        )
    if variant == "nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        packed_rhs = _prepare_e8p_sign_nibble_abs_index_rhs_tiles(layer)
        return lambda: _call_nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_raw(
            layer, fixture, sorted_inputs, packed_rhs
        )
    if variant == "nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        packed_rhs = _prepare_e8p_sign_plane_abs_index_rhs_tiles(layer)
        return lambda: _call_nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_raw(
            layer, fixture, sorted_inputs, packed_rhs
        )
    if variant == "nax_e8p_sign_plane_abs_index_rhs_sorted_native_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_sign_plane_abs_index_rhs_sorted_native_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        packed_rhs = _prepare_e8p_sign_plane_abs_index_rhs_tiles(layer)
        return lambda: _call_nax_e8p_sign_plane_abs_index_rhs_sorted_native_raw(
            layer, fixture, sorted_inputs, packed_rhs
        )
    if variant == "nax_e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        packed_rhs = _prepare_e8p_sign_nibble_micro_lut_rhs_tiles(layer)
        return lambda: _call_nax_e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_raw(
            layer, fixture, sorted_inputs, packed_rhs
        )
    if variant == "nax_e8p_sign_nibble_micro_lut_rhs_sorted_native_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_sign_nibble_micro_lut_rhs_sorted_native_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        packed_rhs = _prepare_e8p_sign_nibble_micro_lut_rhs_tiles(layer)
        return lambda: _call_nax_e8p_sign_nibble_micro_lut_rhs_sorted_native_raw(
            layer, fixture, sorted_inputs, packed_rhs
        )
    if variant == "nax_e8p_packed_rhs_sorted_tiled_m128_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_packed_rhs_sorted_tiled_m128_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture, route_tile=128)
        packed_rhs = _prepare_e8p_packed_rhs_tiles(layer)
        return lambda: _call_nax_e8p_packed_rhs_sorted_tiled_m128_raw(
            layer, fixture, sorted_inputs, packed_rhs
        )
    if variant == "nax_e8p_packed_rhs_sorted_tiled_k128_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_packed_rhs_sorted_tiled_k128_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        packed_rhs = _prepare_e8p_packed_rhs_tiles(layer)
        return lambda: _call_nax_e8p_packed_rhs_sorted_tiled_k128_raw(
            layer, fixture, sorted_inputs, packed_rhs
        )
    if variant == "nax_e8p_predecoded_fp16_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_predecoded_fp16_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        weight_t = _predecode_vq_layer_weight_t_fp16(layer)
        return lambda: _call_nax_e8p_predecoded_fp16_raw(weight_t, fixture, sorted_inputs)
    if variant == "nax_e8p_fp16_sorted_steel_gs352_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_fp16_sorted_steel_gs352_raw artifact benchmark requires code_bits=16"
            )
        if layer.group_size != 352 or fixture.input_dims != 1408 or layer.scales.shape[2] != 4:
            raise ValueError(
                "nax_e8p_fp16_sorted_steel_gs352_raw requires Air down K=1408 group_size=352"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        return lambda: _call_nax_e8p_fp16_sorted_steel_gs352_raw(
            layer, fixture, sorted_inputs
        )
    if variant == "nax_e8p_fp16_sorted_steel_lut_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_fp16_sorted_steel_lut_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        return lambda: _call_nax_e8p_fp16_sorted_steel_lut_raw(
            layer, fixture, sorted_inputs
        )
    if variant == "nax_e8p_fp16_sorted_steel_tgcb_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_fp16_sorted_steel_tgcb_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        return lambda: _call_nax_e8p_fp16_sorted_steel_tgcb_raw(
            layer, fixture, sorted_inputs
        )
    if variant == "nax_e8p_fp16_sorted_steel_tgcb_hoist_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_fp16_sorted_steel_tgcb_hoist_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        return lambda: _call_nax_e8p_fp16_sorted_steel_tgcb_hoist_raw(
            layer, fixture, sorted_inputs
        )
    if variant == "nax_e8p_fp16_sorted_steel_tgscale_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_fp16_sorted_steel_tgscale_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        return lambda: _call_nax_e8p_fp16_sorted_steel_tgscale_raw(
            layer, fixture, sorted_inputs
        )
    if variant == "nax_e8p_fp16_sorted_steel_tgcb_tgscale_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_fp16_sorted_steel_tgcb_tgscale_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        return lambda: _call_nax_e8p_fp16_sorted_steel_tgcb_tgscale_raw(
            layer, fixture, sorted_inputs
        )
    if variant == "nax_e8p_fp16_sorted_steel_bk128_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_fp16_sorted_steel_bk128_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        return lambda: _call_nax_e8p_fp16_sorted_steel_bk128_raw(
            layer, fixture, sorted_inputs
        )
    if variant == "nax_e8p_fp16_sorted_steel_m128n32_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_fp16_sorted_steel_m128n32_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture, route_tile=128)
        return lambda: _call_nax_e8p_fp16_sorted_steel_m128n32_raw(
            layer, fixture, sorted_inputs
        )
    if variant == "nax_e8p_fp16_sorted_steel_m64n128_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_fp16_sorted_steel_m64n128_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        return lambda: _call_nax_e8p_fp16_sorted_steel_m64n128_raw(
            layer, fixture, sorted_inputs
        )
    if variant == "nax_e8p_fp16_sorted_steel_m64n64t64_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_fp16_sorted_steel_m64n64t64_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        return lambda: _call_nax_e8p_fp16_sorted_steel_m64n64t64_raw(
            layer, fixture, sorted_inputs
        )
    if variant == "nax_e8p_fp16_sorted_steel_m32n64_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_fp16_sorted_steel_m32n64_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture, route_tile=32)
        return lambda: _call_nax_e8p_fp16_sorted_steel_m32n64_raw(
            layer, fixture, sorted_inputs
        )
    if variant == "nax_e8p_fp16_sorted_steel_m32n64t128_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_fp16_sorted_steel_m32n64t128_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture, route_tile=32)
        return lambda: _call_nax_e8p_fp16_sorted_steel_m32n64t128_raw(
            layer, fixture, sorted_inputs
        )
    if variant == "nax_e8p_fp16_sorted_steel_m32n128_raw":
        if layer.code_bits != 16:
            raise ValueError(
                "nax_e8p_fp16_sorted_steel_m32n128_raw artifact benchmark requires code_bits=16"
            )
        sorted_inputs = _prepare_sorted_steel_inputs(fixture, route_tile=32)
        return lambda: _call_nax_e8p_fp16_sorted_steel_m32n128_raw(
            layer, fixture, sorted_inputs
        )
    raise ValueError(
        "artifact-backed projection benchmarks currently support "
        "vq_e1, vq_current, nax_e8_fp16_sorted_steel, nax_e8_fp16_sorted_steel_raw, "
        "nax_e8p_fp16_sorted_steel, nax_e8p_fp16_sorted_steel_raw, "
        "nax_e8p_fp16_sorted_direct_reduce_raw, "
        "nax_e8p_fp16_sorted_inline_b_raw, "
        "nax_e8p_packed_rhs_sorted_tiled_raw, "
        "nax_e8p_route_slot_codeword_stream_rhs_sorted_raw, "
        "nax_e8p_route_slot_mma_codeword_tile_rhs_sorted_raw, "
        "nax_e8p_active_route_tile_codeword_outer_product_rhs_sorted_raw, "
        "nax_e8p_expert_cohort_codeword_broadcast_rhs_sorted_raw, "
        "nax_e8p_route_batch_segmented_codeword_reduce_rhs_sorted_raw, "
        "nax_e8p_token_cohort_codeword_stream_rhs_sorted_raw, "
        "nax_e8p_token_cohort_mma_codeword_tile_rhs_sorted_raw, "
        "nax_e8p_output_stationary_codeword_tile_rhs_sorted_raw, "
        "nax_e8p_input_stationary_codeword_tile_rhs_sorted_raw, "
        "nax_e8p_expert_kblock_codeword_factor_reuse_rhs_sorted_raw, "
        "nax_e8p_rowwise_codeword_tile_accumulate_rhs_sorted_raw, "
        "nax_e8p_output_tile_local_codeword_lut_rhs_sorted_raw, "
        "nax_e8p_route_microtile_codeword_block_reduce_rhs_sorted_raw, "
        "nax_e8p_kblock_wavefront_codeword_scan_rhs_sorted_raw, "
        "nax_e8p_token_route_output_stripe_pipeline_rhs_sorted_raw, "
        "nax_e8p_expert_kblock_scale_slot_stream_rhs_sorted_raw, "
        "nax_e8p_scale_group_route_block_reduce_rhs_sorted_raw, "
        "nax_e8p_split_byte_rhs_sorted_tiled_raw, "
        "nax_e8p_split_byte_factor_reuse_rhs_sorted_tiled_raw, "
        "nax_e8p_split_byte_factor_reuse_rhs_sorted_native_raw, "
        "nax_e8p_component_stream_rhs_sorted_scalar_raw, "
        "nax_e8p_component_stream_rhs_sorted_partial_raw, "
        "nax_e8p_component_stream_rhs_sorted_tensorops_raw, "
        "nax_e8p_component_stream_rhs_sorted_shared_decode_raw, "
        "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_native_raw, "
        "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_raw, "
        "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_raw, "
        "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_raw, "
        "nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_raw, "
        "nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_raw, "
        "nax_e8p_sign_plane_abs_index_rhs_sorted_native_raw, "
        "nax_e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_raw, "
        "nax_e8p_sign_nibble_micro_lut_rhs_sorted_native_raw, "
        "nax_e8p_packed_rhs_sorted_tiled_m128_raw, "
        "nax_e8p_packed_rhs_sorted_tiled_k128_raw, "
        "nax_e8p_predecoded_fp16_raw, "
        "nax_e8p_fp16_sorted_steel_gs352_raw, "
        "nax_e8p_fp16_sorted_steel_lut_raw, "
        "nax_e8p_fp16_sorted_steel_tgcb_raw, "
        "nax_e8p_fp16_sorted_steel_tgscale_raw, "
        "nax_e8p_fp16_sorted_steel_tgcb_tgscale_raw, "
        "nax_e8p_fp16_sorted_steel_tgcb_hoist_raw, "
        "nax_e8p_fp16_sorted_steel_bk128_raw, "
        "nax_e8p_fp16_sorted_steel_m128n32_raw, "
        "nax_e8p_fp16_sorted_steel_m64n128_raw, "
        "nax_e8p_fp16_sorted_steel_m32n64_raw, "
        "nax_e8p_fp16_sorted_steel_m64n64t64_raw, "
        "nax_e8p_fp16_sorted_steel_m32n64t128_raw, and "
        "nax_e8p_fp16_sorted_steel_m32n128_raw"
    )


def run_loaded_projection_variant(
    fixture: ProjectionFixture,
    layer: QuantizedVQSwitchLinear,
    *,
    variant: ProjectionVariant,
    iterations: int = 5,
    warmup: int = 2,
    artifact_metadata: dict[str, Any] | None = None,
    artifact_reference_check: bool = True,
) -> dict[str, Any]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if warmup < 0:
        raise ValueError("warmup cannot be negative")
    if layer.input_dims != fixture.input_dims or layer.output_dims != fixture.output_dims:
        raise ValueError("loaded layer dimensions must match the benchmark fixture")
    if layer.num_experts != fixture.experts:
        raise ValueError("loaded layer expert count must match the benchmark fixture")
    if layer.group_size != fixture.vq_group_size or layer.code_bits != fixture.vq_code_bits:
        raise ValueError("loaded layer quantization metadata must match the benchmark fixture")

    run_once = _loaded_artifact_variant_callable(fixture, layer, variant)
    for _ in range(warmup):
        mx.eval(run_once())

    before_vm = collect_vm_stat_counts()
    reset_mlx_peak_memory()
    start = time.perf_counter()
    output = None
    for _ in range(iterations):
        output = run_once()
        mx.eval(output)
    elapsed = time.perf_counter() - start
    if output is None:
        raise AssertionError("projection benchmark did not run")
    output_float = output.astype(mx.float32)
    finite = bool(mx.all(mx.isfinite(output_float)).item())
    checksum = float(mx.sum(output_float).item())
    metrics = collect_metric_snapshot(previous_vm_stat_counts=before_vm)
    extra = dict(artifact_metadata or {})
    extra["artifact_reference_check"] = artifact_reference_check
    if variant in (
        "nax_e8p_component_stream_rhs_sorted_scalar_raw",
        "nax_e8p_component_stream_rhs_sorted_partial_raw",
        "nax_e8p_component_stream_rhs_sorted_tensorops_raw",
        "nax_e8p_component_stream_rhs_sorted_shared_decode_raw",
        "nax_e8p_route_slot_codeword_stream_rhs_sorted_raw",
        "nax_e8p_route_slot_mma_codeword_tile_rhs_sorted_raw",
        "nax_e8p_active_route_tile_codeword_outer_product_rhs_sorted_raw",
        "nax_e8p_expert_cohort_codeword_broadcast_rhs_sorted_raw",
        "nax_e8p_route_batch_segmented_codeword_reduce_rhs_sorted_raw",
        "nax_e8p_token_cohort_codeword_stream_rhs_sorted_raw",
        "nax_e8p_token_cohort_mma_codeword_tile_rhs_sorted_raw",
        "nax_e8p_output_stationary_codeword_tile_rhs_sorted_raw",
        "nax_e8p_input_stationary_codeword_tile_rhs_sorted_raw",
        "nax_e8p_expert_kblock_codeword_factor_reuse_rhs_sorted_raw",
        "nax_e8p_route_codeword_lut_accumulate_rhs_sorted_raw",
        "nax_e8p_rowwise_codeword_tile_accumulate_rhs_sorted_raw",
        "nax_e8p_output_tile_local_codeword_lut_rhs_sorted_raw",
        "nax_e8p_route_microtile_codeword_block_reduce_rhs_sorted_raw",
        "nax_e8p_kblock_wavefront_codeword_scan_rhs_sorted_raw",
        "nax_e8p_token_route_output_stripe_pipeline_rhs_sorted_raw",
        "nax_e8p_expert_kblock_scale_slot_stream_rhs_sorted_raw",
        "nax_e8p_scale_group_route_block_reduce_rhs_sorted_raw",
        "nax_e8p_route_block_output_group_stream_rhs_sorted_raw",
        "nax_e8p_output_group_pretransposed_codeword_stream_rhs_sorted_raw",
        "nax_e8p_kblock_output_group_route_fused_stream_rhs_sorted_raw",
        "nax_e8p_route_tile_output_swizzle_stream_rhs_sorted_raw",
    ) and artifact_reference_check:
        sorted_inputs = _prepare_sorted_steel_inputs(fixture)
        weight_t = _predecode_vq_layer_weight_t_fp16(layer)
        reference_output = _call_decoded_fp16_sorted_direct_reference(
            weight_t,
            sorted_inputs,
        )
        reference_float = reference_output.astype(mx.float32)
        mx.eval(output_float, reference_float)
        dot = mx.sum(output_float * reference_float)
        norm = mx.sqrt(
            mx.sum(output_float * output_float) * mx.sum(reference_float * reference_float)
        )
        max_abs_diff = mx.max(mx.abs(output_float - reference_float))
        mx.eval(dot, norm, max_abs_diff)
        norm_value = float(norm.item())
        extra["artifact_reference_variant"] = "decoded_fp16_sorted_direct"
        extra["artifact_reference_cosine"] = (
            float((dot / norm).item()) if norm_value > 0.0 else 0.0
        )
        extra["artifact_reference_max_abs_diff"] = float(max_abs_diff.item())
    if variant == "nax_e8p_fp16_sorted_steel":
        extra["diagnostic_phase"] = "nax_e8p_fp16_sorted_steel"
        extra["diagnostic_note"] = (
            "artifact-backed native fused E8P FP16 TensorOps timing over prequantized "
            "codes/scales/codebook; excludes RTN/E8P encode cost"
        )
    elif variant == "nax_e8p_fp16_sorted_steel_raw":
        extra["diagnostic_phase"] = "nax_e8p_fp16_sorted_steel_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native fused E8P FP16 TensorOps timing over prequantized "
            "codes/scales/codebook and pre-sorted contiguous route activations; excludes "
            "RTN/E8P encode cost plus route sort/scatter overhead"
        )
    elif variant == "nax_e8p_fp16_sorted_direct_reduce_raw":
        extra["diagnostic_phase"] = "nax_e8p_fp16_sorted_direct_reduce_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native fused E8P FP16 direct-reduce timing over prequantized "
            "codes/scales/codebook and pre-sorted contiguous route activations; no decoded-B "
            "threadgroup staging; excludes RTN/E8P encode cost plus route sort/scatter overhead"
        )
    elif variant == "nax_e8p_fp16_sorted_inline_b_raw":
        extra["diagnostic_phase"] = "nax_e8p_fp16_sorted_inline_b_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native fused E8P FP16 TensorOps timing with inline B-tile "
            "decode over prequantized codes/scales/codebook and pre-sorted contiguous "
            "route activations; no decoded-B threadgroup staging; excludes RTN/E8P "
            "encode cost plus route sort/scatter overhead"
        )
    elif variant == "nax_e8p_packed_rhs_sorted_tiled_raw":
        extra["diagnostic_phase"] = "nax_e8p_packed_rhs_sorted_tiled_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native packed RHS E8P FP16 TensorOps timing over prepacked "
            "bn64/bk64 code/scale tiles and pre-sorted contiguous route activations; "
            "packing happens before the timed loop and dense RHS is never materialized"
        )
    elif variant == "nax_e8p_route_slot_codeword_stream_rhs_sorted_raw":
        extra["diagnostic_phase"] = "nax_e8p_route_slot_codeword_stream_rhs_sorted_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native route-slot codeword-stream RHS E8P FP16 parity "
            "candidate over prepacked bn64/bk64 code/scale tiles and pre-sorted "
            "contiguous route activations; route-slot outputs are streamed from "
            "compressed uint16 codewords, packing happens before the timed loop, "
            "dense RHS is never materialized, and same-window q2 timing is still "
            "required before any speed claim"
        )
    elif variant == "nax_e8p_route_slot_mma_codeword_tile_rhs_sorted_raw":
        extra["diagnostic_phase"] = (
            "nax_e8p_route_slot_mma_codeword_tile_rhs_sorted_raw"
        )
        extra["diagnostic_note"] = (
            "artifact-backed native route-slot MMA codeword-tile RHS E8P FP16 "
            "parity candidate over prepacked bn64/bk64 code/scale tiles and "
            "pre-sorted contiguous route activations; route-slot outputs are "
            "accumulated from compressed uint16 codewords, packing happens before "
            "the timed loop, dense RHS is never materialized, and same-window q2 "
            "timing is still required before any speed claim"
        )
    elif variant == "nax_e8p_active_route_tile_codeword_outer_product_rhs_sorted_raw":
        extra["diagnostic_phase"] = (
            "nax_e8p_active_route_tile_codeword_outer_product_rhs_sorted_raw"
        )
        extra["diagnostic_note"] = (
            "artifact-backed native active-route-tile codeword outer-product RHS "
            "E8P FP16 parity candidate over prepacked bn64/bk64 code/scale tiles "
            "and pre-sorted contiguous route activations; active-route-tile outputs "
            "are accumulated from compressed uint16 codewords, packing happens "
            "before the timed loop, dense RHS is never materialized, and same-window "
            "q2 timing is still required before any speed claim"
        )
    elif variant == "nax_e8p_expert_cohort_codeword_broadcast_rhs_sorted_raw":
        extra["diagnostic_phase"] = (
            "nax_e8p_expert_cohort_codeword_broadcast_rhs_sorted_raw"
        )
        extra["diagnostic_note"] = (
            "artifact-backed native expert-cohort codeword-broadcast RHS E8P FP16 "
            "parity candidate over prepacked bn64/bk64 code/scale tiles and "
            "pre-sorted contiguous route activations; expert-cohort descriptors "
            "group route tiles by expert, packing happens before the timed loop, "
            "dense RHS is never materialized, and same-window q2 timing is still "
            "required before any speed claim"
        )
    elif variant == "nax_e8p_route_batch_segmented_codeword_reduce_rhs_sorted_raw":
        extra["diagnostic_phase"] = (
            "nax_e8p_route_batch_segmented_codeword_reduce_rhs_sorted_raw"
        )
        extra["diagnostic_note"] = (
            "artifact-backed native route-batch segmented codeword-reduce RHS "
            "E8P FP16 parity candidate over prepacked bn64/bk64 code/scale tiles "
            "and pre-sorted contiguous route activations; route-batch segment "
            "descriptors preserve the compressed route IDs, packing happens "
            "before the timed loop, dense RHS is never materialized, and "
            "same-window q2 timing is still required before any speed claim"
        )
    elif variant == "nax_e8p_token_cohort_codeword_stream_rhs_sorted_raw":
        extra["diagnostic_phase"] = (
            "nax_e8p_token_cohort_codeword_stream_rhs_sorted_raw"
        )
        extra["diagnostic_note"] = (
            "artifact-backed native token-cohort codeword-stream RHS E8P FP16 "
            "parity candidate over prepacked bn64/bk64 code/scale tiles and "
            "pre-sorted contiguous route activations; token-cohort descriptors "
            "preserve exact route-slot indirection, packing happens before the "
            "timed loop, dense RHS is never materialized, and same-window q2 "
            "timing is still required before any speed claim"
        )
    elif variant == "nax_e8p_token_cohort_mma_codeword_tile_rhs_sorted_raw":
        extra["diagnostic_phase"] = (
            "nax_e8p_token_cohort_mma_codeword_tile_rhs_sorted_raw"
        )
        extra["diagnostic_note"] = (
            "artifact-backed native token-cohort MMA codeword-tile RHS E8P FP16 "
            "parity candidate over prepacked bn64/bk64 code/scale tiles and "
            "pre-sorted contiguous route activations; token-cohort descriptors "
            "preserve exact route-slot indirection, packing happens before the "
            "timed loop, dense RHS is never materialized, and same-window q2 "
            "timing is still required before any speed claim"
        )
    elif variant == "nax_e8p_output_stationary_codeword_tile_rhs_sorted_raw":
        extra["diagnostic_phase"] = (
            "nax_e8p_output_stationary_codeword_tile_rhs_sorted_raw"
        )
        extra["diagnostic_note"] = (
            "artifact-backed native output-stationary codeword-tile RHS E8P "
            "FP16 parity candidate over prepacked bn64/bk64 code/scale tiles "
            "and pre-sorted contiguous route activations; output-stationary "
            "route-batch descriptors preserve exact route-slot indirection, "
            "packing happens before the timed loop, dense RHS is never "
            "materialized, and same-window q2 timing is still required before "
            "any speed claim"
        )
    elif variant == "nax_e8p_input_stationary_codeword_tile_rhs_sorted_raw":
        extra["diagnostic_phase"] = (
            "nax_e8p_input_stationary_codeword_tile_rhs_sorted_raw"
        )
        extra["diagnostic_note"] = (
            "artifact-backed native input-stationary codeword-tile RHS E8P "
            "FP16 parity candidate over prepacked bn64/bk64 code/scale tiles "
            "and pre-sorted contiguous route activations; input-stationary "
            "route-batch descriptors preserve exact route-slot indirection, "
            "packing happens before the timed loop, dense RHS is never "
            "materialized, and same-window q2 timing is still required before "
            "any speed claim"
        )
    elif variant == "nax_e8p_expert_kblock_codeword_factor_reuse_rhs_sorted_raw":
        extra["diagnostic_phase"] = (
            "nax_e8p_expert_kblock_codeword_factor_reuse_rhs_sorted_raw"
        )
        extra["diagnostic_note"] = (
            "artifact-backed native expert/K-block codeword factor-reuse RHS "
            "E8P FP16 parity candidate over sign/abs LUTs shared across all "
            "output tiles for each expert and K block, plus slot maps and "
            "pre-sorted contiguous route activations; packing happens before "
            "the timed loop, dense RHS is never materialized, and same-window "
            "q2 timing is still required before any speed claim"
        )
    elif variant == "nax_e8p_route_codeword_lut_accumulate_rhs_sorted_raw":
        extra["diagnostic_phase"] = (
            "nax_e8p_route_codeword_lut_accumulate_rhs_sorted_raw"
        )
        extra["diagnostic_note"] = (
            "artifact-backed native route-codeword LUT accumulate E8P FP16 "
            "parity candidate over sparse encoded axis/codeword descriptors, "
            "prepacked bn64/bk64 compressed RHS tiles, and pre-sorted "
            "contiguous route activations; route-local codeword dot LUT "
            "construction happens before the timed loop, dense RHS is never "
            "materialized, and same-window q2 timing is still required before "
            "any speed claim"
        )
    elif variant == "nax_e8p_rowwise_codeword_tile_accumulate_rhs_sorted_raw":
        extra["diagnostic_phase"] = (
            "nax_e8p_rowwise_codeword_tile_accumulate_rhs_sorted_raw"
        )
        extra["diagnostic_note"] = (
            "artifact-backed native rowwise codeword-tile accumulate RHS E8P "
            "FP16 parity candidate over prepacked bn64/bk64 code/scale tiles "
            "and pre-sorted contiguous route activations; route-microtile "
            "descriptors preserve exact route-slot indirection, packing "
            "happens before the timed loop, dense RHS is never materialized, "
            "and same-window q2 timing is still required before any speed claim"
        )
    elif variant == "nax_e8p_output_tile_local_codeword_lut_rhs_sorted_raw":
        extra["diagnostic_phase"] = (
            "nax_e8p_output_tile_local_codeword_lut_rhs_sorted_raw"
        )
        extra["diagnostic_note"] = (
            "artifact-backed native output-tile-local codeword LUT RHS E8P "
            "FP16 parity candidate over prepacked bn64/bk64 code/scale tiles "
            "and pre-sorted contiguous route activations; route-microtile "
            "descriptors preserve exact route-slot indirection, output-tile-local "
            "activation-codeword dot reuse happens inside the native wrapper, "
            "dense RHS is never materialized, and same-window q2 timing is still "
            "required before any speed claim"
        )
    elif variant == "nax_e8p_route_microtile_codeword_block_reduce_rhs_sorted_raw":
        extra["diagnostic_phase"] = (
            "nax_e8p_route_microtile_codeword_block_reduce_rhs_sorted_raw"
        )
        extra["diagnostic_note"] = (
            "artifact-backed native route-microtile codeword block-reduce RHS "
            "E8P FP16 parity candidate over prepacked bn64/bk64 code/scale "
            "tiles and pre-sorted contiguous route activations; route-microtile "
            "descriptors preserve exact route-slot indirection, codeword-block "
            "partials are reduced before output-tile writeback inside the "
            "native wrapper, dense RHS is never materialized, and same-window "
            "q2 timing is still required before any speed claim"
        )
    elif variant == "nax_e8p_kblock_wavefront_codeword_scan_rhs_sorted_raw":
        extra["diagnostic_phase"] = (
            "nax_e8p_kblock_wavefront_codeword_scan_rhs_sorted_raw"
        )
        extra["diagnostic_note"] = (
            "artifact-backed native k-block wavefront codeword-scan RHS E8P "
            "FP16 parity candidate over prepacked bn64/bk64 code/scale tiles "
            "and pre-sorted contiguous route activations; route-microtile "
            "descriptors preserve exact route-slot indirection, K-block "
            "wavefronts stream compressed codeword groups through the native "
            "wrapper, dense RHS is never materialized, and same-window q2 "
            "timing is still required before any speed claim"
        )
    elif variant == "nax_e8p_token_route_output_stripe_pipeline_rhs_sorted_raw":
        extra["diagnostic_phase"] = (
            "nax_e8p_token_route_output_stripe_pipeline_rhs_sorted_raw"
        )
        extra["diagnostic_note"] = (
            "artifact-backed native token-route output-stripe pipeline RHS E8P "
            "FP16 parity candidate over prepacked bn64/bk64 code/scale tiles "
            "and pre-sorted contiguous route activations; token descriptors "
            "preserve route-slot indirection after expert sorting, K-block "
            "stages stream compressed codewords through the native wrapper, "
            "dense RHS is never materialized, and same-window q2 timing is "
            "still required before any speed claim"
        )
    elif variant == "nax_e8p_expert_kblock_scale_slot_stream_rhs_sorted_raw":
        extra["diagnostic_phase"] = (
            "nax_e8p_expert_kblock_scale_slot_stream_rhs_sorted_raw"
        )
        extra["diagnostic_note"] = (
            "artifact-backed native expert/K-block scale-slot stream RHS E8P "
            "FP16 parity candidate over prepacked bn64/bk64 code/scale tiles "
            "and pre-sorted contiguous route activations; expert, K-block, "
            "scale-group, route-tile, and output-tile axes are preserved, "
            "compressed codeword tiles stream through the native wrapper, dense "
            "RHS is never materialized, and same-window q2 timing is still "
            "required before any speed claim"
        )
    elif variant == "nax_e8p_scale_group_route_block_reduce_rhs_sorted_raw":
        extra["diagnostic_phase"] = (
            "nax_e8p_scale_group_route_block_reduce_rhs_sorted_raw"
        )
        extra["diagnostic_note"] = (
            "artifact-backed native scale-group route-block reduce RHS E8P "
            "FP16 parity candidate over prepacked bn64/bk64 code/scale tiles "
            "and pre-sorted contiguous route activations; scale-group, "
            "route-block, K-block, and output-tile axes are preserved, "
            "route-block partials reduce before output-tile writeback, dense "
            "RHS is never materialized, and same-window q2 timing is still "
            "required before any speed claim"
        )
    elif variant == "nax_e8p_route_block_output_group_stream_rhs_sorted_raw":
        extra["diagnostic_phase"] = (
            "nax_e8p_route_block_output_group_stream_rhs_sorted_raw"
        )
        extra["diagnostic_note"] = (
            "artifact-backed native route-block output-group stream RHS E8P "
            "FP16 parity candidate over prepacked bn64/bk64 code/scale tiles "
            "and pre-sorted contiguous route activations; route-block, "
            "output-group, K-block, and codeword-group axes are preserved, "
            "output-group accumulators write directly to route slots, dense "
            "RHS is never materialized, and same-window q2 timing is still "
            "required before any speed claim"
        )
    elif variant == "nax_e8p_output_group_pretransposed_codeword_stream_rhs_sorted_raw":
        extra["diagnostic_phase"] = (
            "nax_e8p_output_group_pretransposed_codeword_stream_rhs_sorted_raw"
        )
        extra["diagnostic_note"] = (
            "artifact-backed native output-group-pretransposed codeword stream "
            "RHS E8P FP16 parity candidate over prepacked bn64/bk64 "
            "code/scale tiles and pre-sorted contiguous route activations; "
            "output-group, route-block, K-block, and codeword-group axes are "
            "preserved, output-group-pretransposed streams feed route-block "
            "accumulators, dense RHS is never materialized, and same-window q2 "
            "timing is still required before any speed claim"
        )
    elif variant == "nax_e8p_kblock_output_group_route_fused_stream_rhs_sorted_raw":
        extra["diagnostic_phase"] = (
            "nax_e8p_kblock_output_group_route_fused_stream_rhs_sorted_raw"
        )
        extra["diagnostic_note"] = (
            "artifact-backed native K-block output-group route-fused stream "
            "RHS E8P FP16 parity candidate over prepacked bn64/bk64 "
            "code/scale tiles and pre-sorted contiguous route activations; "
            "K-block, output-group, route-block, and codeword-group axes are "
            "preserved, route slots write after K-block output-group route "
            "fusion, dense RHS is never materialized, and same-window q2 "
            "timing is still required before any speed claim"
        )
    elif variant == "nax_e8p_route_tile_output_swizzle_stream_rhs_sorted_raw":
        extra["diagnostic_phase"] = (
            "nax_e8p_route_tile_output_swizzle_stream_rhs_sorted_raw"
        )
        extra["diagnostic_note"] = (
            "artifact-backed native route-tile output-swizzle stream RHS E8P "
            "FP16 parity candidate over prepacked bn64/bk64 code/scale tiles "
            "and pre-sorted contiguous route activations; route-tile, "
            "output-swizzle, K-block, and codeword-group axes are preserved, "
            "route slots write after output-swizzle accumulation, dense RHS "
            "is never materialized, and same-window q2 timing is still "
            "required before any speed claim"
        )
    elif variant == "nax_e8p_split_byte_rhs_sorted_tiled_raw":
        extra["diagnostic_phase"] = "nax_e8p_split_byte_rhs_sorted_tiled_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native split-byte RHS E8P FP16 TensorOps timing over "
            "prepacked bn64/bk64 sign/abs/parity factor tiles and pre-sorted "
            "contiguous route activations; packing happens before the timed loop "
            "and dense RHS is never materialized"
        )
    elif variant == "nax_e8p_split_byte_factor_reuse_rhs_sorted_tiled_raw":
        extra["diagnostic_phase"] = "nax_e8p_split_byte_factor_reuse_rhs_sorted_tiled_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native split-byte factor-reuse RHS E8P FP16 TensorOps "
            "timing over prepacked bn64/bk64 sign/abs LUTs plus slot maps and "
            "pre-sorted contiguous route activations; packing happens before the "
            "timed loop and dense RHS is never materialized"
        )
    elif variant == "nax_e8p_split_byte_factor_reuse_rhs_sorted_native_raw":
        extra["diagnostic_phase"] = "nax_e8p_split_byte_factor_reuse_rhs_sorted_native_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native split-byte factor-reuse RHS E8P FP16 scalar/direct "
            "non-staged timing over prepacked bn64/bk64 sign/abs LUTs plus slot maps "
            "and pre-sorted contiguous route activations; packing happens before the "
            "timed loop and dense RHS is never materialized"
        )
    elif variant == "nax_e8p_component_stream_rhs_sorted_scalar_raw":
        extra["diagnostic_phase"] = "nax_e8p_component_stream_rhs_sorted_scalar_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native component-stream RHS E8P FP16 scalar/source oracle "
            "timing over prepacked component sign bits, abs-index rows, component scale "
            "slots, codeword scale slots, compact scales, and pre-sorted contiguous route "
            "activations; packing happens before the timed loop and dense RHS is never "
            "materialized"
        )
    elif variant == "nax_e8p_component_stream_rhs_sorted_partial_raw":
        extra["diagnostic_phase"] = "nax_e8p_component_stream_rhs_sorted_partial_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native component-stream RHS E8P FP16 partial-reduction scaffold "
            "over prepacked component sign bits, abs-index rows, component scale slots, "
            "codeword scale slots, compact scales, and pre-sorted contiguous route "
            "activations; packing happens before the timed loop and dense RHS is never "
            "materialized"
        )
    elif variant == "nax_e8p_component_stream_rhs_sorted_tensorops_raw":
        extra["diagnostic_phase"] = "nax_e8p_component_stream_rhs_sorted_tensorops_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native component-stream RHS E8P FP16 TensorOps parity candidate "
            "over prepacked component sign bits, abs-index rows, component scale slots, "
            "codeword scale slots, compact scales, and pre-sorted contiguous route "
            "activations; same-window q2 timing is still required before any speed claim, "
            "packing happens before the timed loop, and dense RHS is never materialized"
        )
    elif variant == "nax_e8p_component_stream_rhs_sorted_shared_decode_raw":
        extra["diagnostic_phase"] = "nax_e8p_component_stream_rhs_sorted_shared_decode_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native component-stream RHS E8P FP16 shared-decode parity "
            "candidate over prepacked component sign bits, abs-index rows, component "
            "scale slots, codeword scale slots, compact scales, and pre-sorted "
            "contiguous route activations; same-window q2 timing is still required "
            "before any speed claim, packing happens before the timed loop, and dense "
            "RHS is never materialized"
        )
    elif variant == "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_native_raw":
        extra["diagnostic_phase"] = "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_native_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native expert/K-block factor-reuse RHS E8P FP16 scalar/direct "
            "timing over sign/abs LUTs shared across all output tiles for each expert and "
            "K block, plus slot maps and pre-sorted contiguous route activations; packing "
            "happens before the timed loop and dense RHS is never materialized"
        )
    elif variant == "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_raw":
        extra["diagnostic_phase"] = "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native expert/K-block factor-reuse RHS E8P FP16 non-staged "
            "TensorOps timing over sign/abs LUTs shared across all output tiles for each "
            "expert and K block, plus slot maps and pre-sorted contiguous route activations; "
            "packing happens before the timed loop and dense RHS is never materialized"
        )
    elif variant == "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_raw":
        extra["diagnostic_phase"] = "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native split-byte factor-reuse RHS E8P FP16 non-staged "
            "TensorOps shared-decode timing over prepacked bn64/bk64 sign/abs LUTs "
            "plus slot maps and pre-sorted contiguous route activations; packing "
            "happens before the timed loop, dense RHS is never materialized, and no "
            "decoded-B threadgroup Ws tile is staged"
        )
    elif variant == "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_raw":
        extra["diagnostic_phase"] = "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native split-byte factor-reuse RHS E8P FP16 shared-n-decode "
            "TensorOps timing over prepacked bn64/bk64 sign/abs LUTs plus slot maps and "
            "pre-sorted contiguous route activations; packing happens before the timed loop, "
            "dense RHS is never materialized, and lane-local decoded B fragments are shared "
            "across the paired route-group simdgroups"
        )
    elif variant == "nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_raw":
        extra["diagnostic_phase"] = "nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native sign-nibble RHS E8P FP16 non-staged TensorOps timing "
            "over prepacked bn64/bk64 low/high sign nibbles, abs-index bytes, parity bytes, "
            "and compact scale-slot maps with pre-sorted contiguous route activations; "
            "packing happens before the timed loop, dense RHS is never materialized, and no "
            "decoded-B threadgroup tile is staged"
        )
    elif variant == "nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_raw":
        extra["diagnostic_phase"] = "nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native sign-plane RHS E8P FP16 non-staged TensorOps timing "
            "over prepacked bn64/bk64 uint64 sign-bit planes, abs-index bytes, and compact "
            "scale-slot maps with pre-sorted contiguous route activations; packing happens "
            "before the timed loop, dense RHS is never materialized, and no decoded-B "
            "threadgroup tile is staged"
        )
    elif variant == "nax_e8p_sign_plane_abs_index_rhs_sorted_native_raw":
        extra["diagnostic_phase"] = "nax_e8p_sign_plane_abs_index_rhs_sorted_native_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native sign-plane RHS E8P FP16 scalar/direct timing over "
            "prepacked bn64/bk64 uint64 sign-bit planes, abs-index bytes, and compact "
            "scale-slot maps with pre-sorted contiguous route activations; packing happens "
            "before the timed loop and dense RHS is never materialized"
        )
    elif variant == "nax_e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_raw":
        extra["diagnostic_phase"] = "nax_e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native sign-nibble micro-LUT RHS E8P FP16 non-staged TensorOps "
            "timing over prepacked bn64/bk64 fixed low/high nibble LUTs, slot maps, per-tile "
            "abs-index LUTs, and compact scale-slot maps with pre-sorted contiguous route "
            "activations; packing happens before the timed loop, dense RHS is never "
            "materialized, and no decoded-B threadgroup tile is staged"
        )
    elif variant == "nax_e8p_sign_nibble_micro_lut_rhs_sorted_native_raw":
        extra["diagnostic_phase"] = "nax_e8p_sign_nibble_micro_lut_rhs_sorted_native_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native sign-nibble micro-LUT RHS E8P FP16 scalar/direct timing "
            "over prepacked bn64/bk64 fixed low/high nibble LUTs, slot maps, per-tile "
            "abs-index LUTs, and compact scale-slot maps with pre-sorted contiguous route "
            "activations; packing happens before the timed loop and dense RHS is never "
            "materialized"
        )
    elif variant == "nax_e8p_packed_rhs_sorted_tiled_m128_raw":
        extra["diagnostic_phase"] = "nax_e8p_packed_rhs_sorted_tiled_m128_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native packed RHS E8P FP16 TensorOps timing over prepacked "
            "bn64/bk64 code/scale tiles and 128-route descriptor tiles; packing happens "
            "before the timed loop and dense RHS is never materialized"
        )
    elif variant == "nax_e8p_packed_rhs_sorted_tiled_k128_raw":
        extra["diagnostic_phase"] = "nax_e8p_packed_rhs_sorted_tiled_k128_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native packed RHS E8P FP16 TensorOps timing over prepacked "
            "bn64/bk64 code/scale tiles, staging two bk64 blocks per barrier; packing "
            "happens before the timed loop and dense RHS is never materialized"
        )
    elif variant == "nax_e8p_predecoded_fp16_raw":
        extra["diagnostic_phase"] = "nax_e8p_predecoded_fp16_raw"
        extra["diagnostic_note"] = (
            "artifact-backed predecoded FP16 upper-bound timing: E8P codes/scales are "
            "decoded to dense FP16 expert weights before the timed loop, then native "
            "sorted gather_mm is timed over pre-sorted contiguous route activations; "
            "violates compressed-size goals and is diagnostic only"
        )
        extra["diagnostic_predecoded_dense_weight_bytes"] = (
            fixture.experts * fixture.output_dims * fixture.input_dims * 2
        )
    elif variant == "nax_e8p_fp16_sorted_steel_gs352_raw":
        extra["diagnostic_phase"] = "nax_e8p_fp16_sorted_steel_gs352_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native fused E8P FP16 TensorOps timing using a benchmark-only "
            "Air down group_size=352 scale-group lookup over prequantized codes/scales/"
            "codebook and pre-sorted contiguous route activations"
        )
    elif variant == "nax_e8p_fp16_sorted_steel_lut_raw":
        extra["diagnostic_phase"] = "nax_e8p_fp16_sorted_steel_lut_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native fused E8P FP16 TensorOps timing using a benchmark-only "
            "predecoded full-grid LUT over prequantized codes/scales and pre-sorted "
            "contiguous route activations"
        )
    elif variant == "nax_e8p_fp16_sorted_steel_tgcb_raw":
        extra["diagnostic_phase"] = "nax_e8p_fp16_sorted_steel_tgcb_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native fused E8P FP16 TensorOps timing using a benchmark-only "
            "threadgroup-staged packed-abs codebook over prequantized codes/scales and "
            "pre-sorted contiguous route activations"
        )
    elif variant == "nax_e8p_fp16_sorted_steel_tgcb_hoist_raw":
        extra["diagnostic_phase"] = "nax_e8p_fp16_sorted_steel_tgcb_hoist_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native fused E8P FP16 TensorOps timing using a benchmark-only "
            "threadgroup-staged packed-abs codebook with per-codeword hoisted E8P decode "
            "over prequantized codes/scales and pre-sorted contiguous route activations"
        )
    elif variant == "nax_e8p_fp16_sorted_steel_tgscale_raw":
        extra["diagnostic_phase"] = "nax_e8p_fp16_sorted_steel_tgscale_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native fused E8P FP16 TensorOps timing using a benchmark-only "
            "threadgroup scale cache over prequantized codes/codebook and pre-sorted "
            "contiguous route activations"
        )
    elif variant == "nax_e8p_fp16_sorted_steel_tgcb_tgscale_raw":
        extra["diagnostic_phase"] = "nax_e8p_fp16_sorted_steel_tgcb_tgscale_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native fused E8P FP16 TensorOps timing using a benchmark-only "
            "threadgroup packed-abs codebook plus scale cache over prequantized codes and "
            "pre-sorted contiguous route activations"
        )
    elif variant == "nax_e8p_fp16_sorted_steel_bk128_raw":
        extra["diagnostic_phase"] = "nax_e8p_fp16_sorted_steel_bk128_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native fused E8P FP16 TensorOps timing using a benchmark-only "
            "BK128 staging tile over prequantized codes/scales/codebook and pre-sorted "
            "contiguous route activations"
        )
    elif variant == "nax_e8p_fp16_sorted_steel_m128n32_raw":
        extra["diagnostic_phase"] = "nax_e8p_fp16_sorted_steel_m128n32_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native fused E8P FP16 TensorOps timing using a benchmark-only "
            "M128/N32 output tile over prequantized codes/scales/codebook and pre-sorted "
            "contiguous route activations"
        )
    elif variant == "nax_e8p_fp16_sorted_steel_m64n128_raw":
        extra["diagnostic_phase"] = "nax_e8p_fp16_sorted_steel_m64n128_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native fused E8P FP16 TensorOps timing using a benchmark-only "
            "M64/N128 output tile over prequantized codes/scales/codebook and pre-sorted "
            "contiguous route activations"
        )
    elif variant == "nax_e8p_fp16_sorted_steel_m32n64_raw":
        extra["diagnostic_phase"] = "nax_e8p_fp16_sorted_steel_m32n64_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native fused E8P FP16 TensorOps timing using a benchmark-only "
            "M32/N64 output tile over prequantized codes/scales/codebook and pre-sorted "
            "contiguous route activations"
        )
    elif variant == "nax_e8p_fp16_sorted_steel_m64n64t64_raw":
        extra["diagnostic_phase"] = "nax_e8p_fp16_sorted_steel_m64n64t64_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native fused E8P FP16 TensorOps timing using a benchmark-only "
            "M64/N64 output tile with two compute simdgroups over prequantized "
            "codes/scales/codebook and pre-sorted contiguous route activations"
        )
    elif variant == "nax_e8p_fp16_sorted_steel_m32n64t128_raw":
        extra["diagnostic_phase"] = "nax_e8p_fp16_sorted_steel_m32n64t128_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native fused E8P FP16 TensorOps timing using a benchmark-only "
            "M32/N64 output tile with 128 staging threads over prequantized "
            "codes/scales/codebook and pre-sorted contiguous route activations"
        )
    elif variant == "nax_e8p_fp16_sorted_steel_m32n128_raw":
        extra["diagnostic_phase"] = "nax_e8p_fp16_sorted_steel_m32n128_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native fused E8P FP16 TensorOps timing using a benchmark-only "
            "M32/N128 output tile over prequantized codes/scales/codebook and pre-sorted "
            "contiguous route activations"
        )
    elif variant == "nax_e8_fp16_sorted_steel":
        extra["diagnostic_phase"] = "nax_e8_fp16_sorted_steel"
        extra["diagnostic_note"] = (
            "artifact-backed native fused E8 FP16 TensorOps timing over prequantized "
            "codes/scales/codebook; excludes RTN/E8 encode cost"
        )
    elif variant == "nax_e8_fp16_sorted_steel_raw":
        extra["diagnostic_phase"] = "nax_e8_fp16_sorted_steel_raw"
        extra["diagnostic_note"] = (
            "artifact-backed native fused E8 FP16 TensorOps timing over prequantized "
            "codes/scales/codebook and pre-sorted contiguous route activations; excludes "
            "RTN/E8 encode cost plus route sort/scatter overhead"
        )
    return {
        "benchmark": "glm45_air_hot_projection_kernel",
        "fixture_source": "vq_artifact",
        "projection": fixture.projection,
        "variant": variant,
        "tokens": fixture.tokens,
        "top_k": fixture.top_k,
        "experts": fixture.experts,
        "input_dims": fixture.input_dims,
        "output_dims": fixture.output_dims,
        "mlx_group_size": None,
        "mlx_sorted_indices": variant
        in (
            "nax_e8_fp16_sorted_steel",
            "nax_e8_fp16_sorted_steel_raw",
            "nax_e8p_fp16_sorted_steel",
            "nax_e8p_fp16_sorted_steel_raw",
            "nax_e8p_fp16_sorted_direct_reduce_raw",
            "nax_e8p_fp16_sorted_inline_b_raw",
            "nax_e8p_packed_rhs_sorted_tiled_raw",
            "nax_e8p_route_slot_codeword_stream_rhs_sorted_raw",
            "nax_e8p_route_slot_mma_codeword_tile_rhs_sorted_raw",
            "nax_e8p_active_route_tile_codeword_outer_product_rhs_sorted_raw",
            "nax_e8p_expert_cohort_codeword_broadcast_rhs_sorted_raw",
            "nax_e8p_route_batch_segmented_codeword_reduce_rhs_sorted_raw",
            "nax_e8p_token_cohort_codeword_stream_rhs_sorted_raw",
            "nax_e8p_token_cohort_mma_codeword_tile_rhs_sorted_raw",
            "nax_e8p_output_stationary_codeword_tile_rhs_sorted_raw",
            "nax_e8p_input_stationary_codeword_tile_rhs_sorted_raw",
            "nax_e8p_expert_kblock_codeword_factor_reuse_rhs_sorted_raw",
            "nax_e8p_rowwise_codeword_tile_accumulate_rhs_sorted_raw",
            "nax_e8p_output_tile_local_codeword_lut_rhs_sorted_raw",
            "nax_e8p_route_microtile_codeword_block_reduce_rhs_sorted_raw",
            "nax_e8p_kblock_wavefront_codeword_scan_rhs_sorted_raw",
            "nax_e8p_token_route_output_stripe_pipeline_rhs_sorted_raw",
            "nax_e8p_expert_kblock_scale_slot_stream_rhs_sorted_raw",
            "nax_e8p_scale_group_route_block_reduce_rhs_sorted_raw",
            "nax_e8p_route_block_output_group_stream_rhs_sorted_raw",
            "nax_e8p_output_group_pretransposed_codeword_stream_rhs_sorted_raw",
            "nax_e8p_kblock_output_group_route_fused_stream_rhs_sorted_raw",
            "nax_e8p_route_tile_output_swizzle_stream_rhs_sorted_raw",
            "nax_e8p_split_byte_rhs_sorted_tiled_raw",
            "nax_e8p_split_byte_factor_reuse_rhs_sorted_tiled_raw",
            "nax_e8p_split_byte_factor_reuse_rhs_sorted_native_raw",
            "nax_e8p_component_stream_rhs_sorted_scalar_raw",
            "nax_e8p_component_stream_rhs_sorted_partial_raw",
            "nax_e8p_component_stream_rhs_sorted_tensorops_raw",
            "nax_e8p_component_stream_rhs_sorted_shared_decode_raw",
            "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_native_raw",
            "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_raw",
            "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_raw",
            "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_raw",
            "nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_raw",
            "nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_raw",
            "nax_e8p_sign_plane_abs_index_rhs_sorted_native_raw",
            "nax_e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_raw",
            "nax_e8p_sign_nibble_micro_lut_rhs_sorted_native_raw",
            "nax_e8p_packed_rhs_sorted_tiled_m128_raw",
            "nax_e8p_packed_rhs_sorted_tiled_k128_raw",
            "nax_e8p_predecoded_fp16_raw",
            "nax_e8p_fp16_sorted_steel_gs352_raw",
            "nax_e8p_fp16_sorted_steel_lut_raw",
            "nax_e8p_fp16_sorted_steel_tgcb_raw",
            "nax_e8p_fp16_sorted_steel_tgscale_raw",
            "nax_e8p_fp16_sorted_steel_tgcb_tgscale_raw",
            "nax_e8p_fp16_sorted_steel_tgcb_hoist_raw",
            "nax_e8p_fp16_sorted_steel_bk128_raw",
            "nax_e8p_fp16_sorted_steel_m128n32_raw",
            "nax_e8p_fp16_sorted_steel_m64n128_raw",
            "nax_e8p_fp16_sorted_steel_m32n64_raw",
            "nax_e8p_fp16_sorted_steel_m64n64t64_raw",
            "nax_e8p_fp16_sorted_steel_m32n64t128_raw",
            "nax_e8p_fp16_sorted_steel_m32n128_raw",
        ),
        "vq_group_size": fixture.vq_group_size,
        "vq_code_bits": fixture.vq_code_bits,
        "iterations": iterations,
        "warmup": warmup,
        "elapsed_seconds": elapsed,
        "ms_per_iter": elapsed * 1000.0 / iterations,
        "output_shape": list(output.shape),
        "finite_output": finite,
        "checksum": checksum,
        **metrics,
        **extra,
    }
