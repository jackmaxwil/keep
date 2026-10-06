from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


_Q2_NAX_RE = re.compile(
    r"_gs_(?P<group_size>\d+)_b_(?P<bits>\d+)_bm_(?P<bm>\d+)_bn_(?P<bn>\d+)"
    r"_bk_(?P<bk>\d+)_wm_(?P<wm>\d+)_wn_(?P<wn>\d+)"
)

_PACKED_RHS_TILED_KERNELS = (
    "nax_e8p_packed_rhs_sorted_tiled_matmul",
    "nax_e8p_packed_rhs_sorted_tiled_m128_matmul",
    "nax_e8p_packed_rhs_sorted_tiled_k128_matmul",
)

_SIGN_NIBBLE_SCALAR_SORTED_KERNEL = "nax_e8p_sign_nibble_abs_index_rhs_sorted_matmul"

_SIGN_NIBBLE_TENSOROPS_KERNELS = (
    "nax_e8p_sign_nibble_abs_index_rhs_sorted_tiled_matmul",
    "nax_e8p_sign_nibble_abs_index_rhs_sorted_shared_decode_matmul",
    "nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_matmul",
    "nax_e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_matmul",
)

_SIGN_PLANE_SCALAR_SORTED_KERNEL = "nax_e8p_sign_plane_abs_index_rhs_sorted_matmul"

_SIGN_PLANE_TENSOROPS_KERNELS = (
    "nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_matmul",
)

_EXPERT_KBLOCK_FACTOR_DECODE_KERNELS = (
    "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul",
    "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_matmul",
    "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul",
)

_EXPERT_KBLOCK_TENSOROPS_KERNEL = (
    "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul"
)

_EXPERT_KBLOCK_SCALAR_KERNEL = (
    "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_matmul"
)

_EXPERT_KBLOCK_TENSOROPS_V2_KERNEL = (
    "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul"
)

_EXPERT_KBLOCK_REQUIRED_FEATURES = [
    "reuse_factor_decode_across_output_tiles_per_expert_kblock",
    "preserve_compressed_rhs_storage",
    "preserve_codeword_scale_slots",
    "account_for_output_column_scale_groups",
    "avoid_per_fragment_factor_reconstruction",
    "avoid_lane_local_barrier_fragment_buffer",
]

_COMPONENT_STREAM_SCALAR_KERNEL = "nax_e8p_component_stream_rhs_sorted_scalar_matmul"
_COMPONENT_STREAM_TENSOROPS_KERNEL = (
    "nax_e8p_component_stream_rhs_sorted_tensorops_matmul"
)
_COMPONENT_STREAM_SHARED_DECODE_KERNEL = (
    "nax_e8p_component_stream_rhs_sorted_shared_decode_matmul"
)
_COMPONENT_STREAM_PARTIAL_KERNEL = "nax_e8p_component_stream_rhs_sorted_partial_matmul"
_COMPONENT_STREAM_PARTIAL_REDUCE_KERNEL = (
    "nax_e8p_component_stream_rhs_sorted_partial_reduce"
)
_ROUTE_SLOT_CODEWORD_STREAM_KERNEL = (
    "nax_e8p_route_slot_codeword_stream_rhs_sorted_matmul"
)
_ROUTE_SLOT_MMA_CODEWORD_TILE_KERNEL = (
    "nax_e8p_route_slot_mma_codeword_tile_rhs_sorted_matmul"
)
_ACTIVE_ROUTE_TILE_CODEWORD_OUTER_PRODUCT_KERNEL = (
    "nax_e8p_active_route_tile_codeword_outer_product_rhs_sorted_matmul"
)
_EXPERT_COHORT_CODEWORD_BROADCAST_KERNEL = (
    "nax_e8p_expert_cohort_codeword_broadcast_rhs_sorted_matmul"
)
_ROUTE_BATCH_SEGMENTED_CODEWORD_REDUCE_KERNEL = (
    "nax_e8p_route_batch_segmented_codeword_reduce_rhs_sorted_matmul"
)
_TOKEN_COHORT_CODEWORD_STREAM_KERNEL = (
    "nax_e8p_token_cohort_codeword_stream_rhs_sorted_matmul"
)
_TOKEN_COHORT_MMA_CODEWORD_TILE_KERNEL = (
    "nax_e8p_token_cohort_mma_codeword_tile_rhs_sorted_matmul"
)
_OUTPUT_STATIONARY_CODEWORD_TILE_KERNEL = (
    "nax_e8p_output_stationary_codeword_tile_rhs_sorted_matmul"
)
_INPUT_STATIONARY_CODEWORD_TILE_KERNEL = (
    "nax_e8p_input_stationary_codeword_tile_rhs_sorted_matmul"
)
_EXPERT_KBLOCK_CODEWORD_FACTOR_REUSE_KERNEL = (
    "nax_e8p_expert_kblock_codeword_factor_reuse_rhs_sorted_matmul"
)
_ROUTE_CODEWORD_LUT_ACCUMULATE_KERNEL = (
    "nax_e8p_route_codeword_lut_accumulate_rhs_sorted_matmul"
)
_ROWWISE_CODEWORD_TILE_ACCUMULATE_KERNEL = (
    "nax_e8p_rowwise_codeword_tile_accumulate_rhs_sorted_matmul"
)
_OUTPUT_TILE_LOCAL_CODEWORD_LUT_KERNEL = (
    "nax_e8p_output_tile_local_codeword_lut_rhs_sorted_matmul"
)
_ROUTE_MICROTILE_CODEWORD_BLOCK_REDUCE_KERNEL = (
    "nax_e8p_route_microtile_codeword_block_reduce_rhs_sorted_matmul"
)
_KBLOCK_WAVEFRONT_CODEWORD_SCAN_KERNEL = (
    "nax_e8p_kblock_wavefront_codeword_scan_rhs_sorted_matmul"
)
_TOKEN_ROUTE_OUTPUT_STRIPE_PIPELINE_KERNEL = (
    "nax_e8p_token_route_output_stripe_pipeline_rhs_sorted_matmul"
)
_EXPERT_KBLOCK_SCALE_SLOT_STREAM_KERNEL = (
    "nax_e8p_expert_kblock_scale_slot_stream_rhs_sorted_matmul"
)
_SCALE_GROUP_ROUTE_BLOCK_REDUCE_KERNEL = (
    "nax_e8p_scale_group_route_block_reduce_rhs_sorted_matmul"
)
_ROUTE_BLOCK_OUTPUT_GROUP_STREAM_KERNEL = (
    "nax_e8p_route_block_output_group_stream_rhs_sorted_matmul"
)
_OUTPUT_GROUP_PRETRANSPOSED_CODEWORD_STREAM_KERNEL = (
    "nax_e8p_output_group_pretransposed_codeword_stream_rhs_sorted_matmul"
)
_KBLOCK_OUTPUT_GROUP_ROUTE_FUSED_STREAM_KERNEL = (
    "nax_e8p_kblock_output_group_route_fused_stream_rhs_sorted_matmul"
)
_ROUTE_TILE_OUTPUT_SWIZZLE_STREAM_KERNEL = (
    "nax_e8p_route_tile_output_swizzle_stream_rhs_sorted_matmul"
)
_TOKEN_TOPK_OUTPUT_TILE_STREAM_KERNEL = (
    "nax_e8p_token_topk_output_tile_stream_rhs_sorted_matmul"
)
_TOKEN_BLOCK_OUTPUT_GROUP_STREAM_KERNEL = (
    "nax_e8p_token_block_output_group_stream_rhs_sorted_matmul"
)
_TOKEN_OUTPUT_STRIPE_GROUP_STREAM_KERNEL = (
    "nax_e8p_token_output_stripe_group_stream_rhs_sorted_matmul"
)
_TOKEN_EXPERT_OUTPUT_BLOCK_STREAM_KERNEL = (
    "nax_e8p_token_expert_output_block_stream_rhs_sorted_matmul"
)
_TOKEN_PAIR_KBLOCK_ACCUMULATOR_STREAM_KERNEL = (
    "nax_e8p_token_pair_kblock_accumulator_stream_rhs_sorted_matmul"
)
_TOKEN_PAIR_OUTPUT_GROUP_STREAM_KERNEL = (
    "nax_e8p_token_pair_output_group_stream_rhs_sorted_matmul"
)
_TOKEN_PAIR_SLOT_TOPK_OUTPUT_GROUP_STREAM_KERNEL = (
    "nax_e8p_token_pair_slot_topk_output_group_stream_rhs_sorted_matmul"
)
_TOKEN_PAIR_SLOT_TOPK_CODEWORD_GROUP_PIPELINE_KERNEL = (
    "nax_e8p_token_pair_slot_topk_codeword_group_pipeline_rhs_sorted_matmul"
)
_TOKEN_PAIR_SLOT_TOPK_SCALE_SLOT_BROADCAST_STREAM_KERNEL = (
    "nax_e8p_token_pair_slot_topk_scale_slot_broadcast_stream_rhs_sorted_matmul"
)
_TOKEN_PAIR_SLOT_TOPK_ROUTE_BUCKET_CODEWORD_REDUCE_KERNEL = (
    "nax_e8p_token_pair_slot_topk_route_bucket_codeword_reduce_rhs_sorted_matmul"
)
_TOKEN_PAIR_SLOT_TOPK_KBLOCK_MICROTILE_STREAM_KERNEL = (
    "nax_e8p_token_pair_slot_topk_kblock_microtile_stream_rhs_sorted_matmul"
)
_TOKEN_PAIR_SLOT_TOPK_OUTPUT_TILE_FUSED_STREAM_KERNEL = (
    "nax_e8p_token_pair_slot_topk_output_tile_fused_stream_rhs_sorted_matmul"
)


def parse_q2_nax_kernel_name(name: str) -> dict[str, int]:
    match = _Q2_NAX_RE.search(name)
    if match is None:
        raise ValueError(f"not a recognized q2 NAX kernel name: {name!r}")
    return {key: int(value) for key, value in match.groupdict().items()}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            rows.append(row)
    return rows


def _extract_kernel_source(source: str, kernel_name: str) -> str:
    marker = f"void {kernel_name}"
    start = source.find(marker)
    if start < 0:
        raise ValueError(f"kernel {kernel_name!r} not found")
    next_kernel = source.find("[[kernel]]", start + len(marker))
    if next_kernel < 0:
        return source[start:]
    return source[start:next_kernel]


def _extract_cpp_function_source(source: str, function_name: str) -> str:
    definition = re.search(
        rf"(?m)^(?:[\w:<>]+\s+)+{re.escape(function_name)}\s*\(",
        source,
    )
    if definition is not None:
        start = definition.start()
    else:
        start = source.find(function_name)
    if start < 0:
        raise ValueError(f"function {function_name!r} not found")
    signature_start = source.rfind("\n", 0, start)
    if signature_start < 0:
        signature_start = 0
    else:
        signature_start += 1
    body_start = source.find("{", start)
    if body_start < 0:
        raise ValueError(f"function {function_name!r} has no body")
    depth = 0
    for position in range(body_start, len(source)):
        char = source[position]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[signature_start : position + 1]
    raise ValueError(f"function {function_name!r} body is unterminated")


def audit_e8p_kernel_source(
    source: str,
    kernel_name: str = "nax_e8p_fp16_sorted_matmul_steel",
) -> dict[str, Any]:
    kernel = _extract_kernel_source(source, kernel_name)
    decoded_b_staging = "threadgroup half Ws[" in kernel
    barrier_count = kernel.count("threadgroup_barrier(")
    has_k_block_decode_loop = "for (uint k_block" in kernel and "k_block += mlx_vq_nax_bk" in kernel
    loads_btile_from_threadgroup_ws = "Btile.template load" in kernel and "Ws +" in kernel
    decodes_e8p_to_ws = "mlx_vq_decode_e8p" in kernel and "Ws[base" in kernel
    structural_risks: list[str] = []
    if decoded_b_staging and decodes_e8p_to_ws:
        structural_risks.append(
            "decoded-B threadgroup staging materializes E8P weight fragments before TensorOps"
        )
    if barrier_count:
        structural_risks.append(f"{barrier_count} threadgroup barriers in the kernel body")
    if loads_btile_from_threadgroup_ws:
        structural_risks.append("B tiles are loaded from staged Ws rather than source codes/scales")
    return {
        "kernel_name": kernel_name,
        "decoded_b_threadgroup_staging": decoded_b_staging,
        "decodes_e8p_to_ws": decodes_e8p_to_ws,
        "threadgroup_barrier_count": barrier_count,
        "has_k_block_decode_loop": has_k_block_decode_loop,
        "loads_btile_from_threadgroup_ws": loads_btile_from_threadgroup_ws,
        "structural_risk": "; ".join(structural_risks) if structural_risks else "none",
    }


def audit_e8p_shared_decode_kernel_source(
    source: str,
    kernel_name: str = "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_matmul",
) -> dict[str, Any]:
    kernel = _extract_kernel_source(source, kernel_name)
    has_threadgroup_half_staging = "threadgroup half" in kernel
    barrier_count = kernel.count("threadgroup_barrier(")
    has_tensorops_matmul = "mpp::tensor_ops::matmul2d" in kernel and "matmul_op.run" in kernel
    factor_lut_load_count = kernel.count("sign_byte_lut[") + kernel.count("abs_index_lut[")
    reconstructs_b_fragments = (
        "b_t[" in kernel
        and "mlx_vq_decode_e8p_split_value" in kernel
        and "sign_byte_lut[" in kernel
        and "abs_index_lut[" in kernel
        and "scale_tiles[" in kernel
    )
    rejects_current_schedule = (
        has_tensorops_matmul
        and not has_threadgroup_half_staging
        and barrier_count == 0
        and reconstructs_b_fragments
    )
    required_next_features = [
        "broader_expert_output_factor_decode_reuse",
        "avoid_per_fragment_factor_reconstruction",
        "preserve_compressed_rhs_storage",
        "change_rhs_layout_or_kernel_family",
    ]
    structural_risks: list[str] = []
    if reconstructs_b_fragments:
        structural_risks.append(
            "B cooperative tensor fragments are reconstructed from split-byte factor LUTs per k/n fragment"
        )
    if has_tensorops_matmul and not has_threadgroup_half_staging:
        structural_risks.append(
            "TensorOps are reached without decoded-B staging, so remaining cost is factor decode schedule reuse"
        )
    return {
        "kernel_name": kernel_name,
        "has_tensorops_matmul": has_tensorops_matmul,
        "has_threadgroup_half_staging": has_threadgroup_half_staging,
        "threadgroup_barrier_count": barrier_count,
        "factor_lut_load_count": factor_lut_load_count,
        "reconstructs_b_fragments_from_factor_luts": reconstructs_b_fragments,
        "decision": (
            "reject_per_fragment_factor_reconstruction_shared_decode"
            if rejects_current_schedule
            else "needs_manual_review"
        ),
        "required_next_features": required_next_features if rejects_current_schedule else [],
        "structural_risk": "; ".join(structural_risks) if structural_risks else "none",
    }


def audit_e8p_shared_n_decode_kernel_source(
    source: str,
    kernel_name: str = "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_matmul",
) -> dict[str, Any]:
    kernel = _extract_kernel_source(source, kernel_name)
    has_lane_local_shared_b = "threadgroup half shared_b" in kernel
    barrier_count = kernel.count("threadgroup_barrier(")
    has_tensorops_matmul = "mpp::tensor_ops::matmul2d" in kernel and "matmul_op.run" in kernel
    uses_factor_luts = "sign_byte_lut[" in kernel and "abs_index_lut[" in kernel
    reconstructs_shared_b = (
        "shared_b[" in kernel
        and "mlx_vq_decode_e8p_split_value" in kernel
        and uses_factor_luts
        and "scale_tiles[" in kernel
    )
    loads_b_fragments_from_shared_b = (
        ("b_t.thread_elements()[" in kernel or "b_t[" in kernel)
        and "shared_b[" in kernel
    )
    rejects_current_schedule = (
        has_tensorops_matmul
        and has_lane_local_shared_b
        and barrier_count > 0
        and reconstructs_shared_b
        and loads_b_fragments_from_shared_b
    )
    required_next_features = [
        "avoid_lane_local_barrier_fragment_buffer",
        "broader_expert_output_factor_decode_reuse",
        "avoid_per_fragment_factor_reconstruction",
        "preserve_compressed_rhs_storage",
        "change_rhs_layout_or_kernel_family",
    ]
    structural_risks: list[str] = []
    if reconstructs_shared_b:
        structural_risks.append(
            "split-byte factor values are decoded into lane-local shared_b before TensorOps"
        )
    if loads_b_fragments_from_shared_b:
        structural_risks.append(
            "right-hand TensorOps fragments are populated from lane-local shared_b storage"
        )
    if barrier_count:
        structural_risks.append(f"{barrier_count} threadgroup barriers in the kernel body")
    return {
        "kernel_name": kernel_name,
        "has_tensorops_matmul": has_tensorops_matmul,
        "has_lane_local_shared_b": has_lane_local_shared_b,
        "threadgroup_barrier_count": barrier_count,
        "uses_factor_luts": uses_factor_luts,
        "reconstructs_shared_b_from_factor_luts": reconstructs_shared_b,
        "loads_b_fragments_from_shared_b": loads_b_fragments_from_shared_b,
        "decision": (
            "reject_lane_local_shared_n_decode"
            if rejects_current_schedule
            else "needs_manual_review"
        ),
        "required_next_features": required_next_features if rejects_current_schedule else [],
        "structural_risk": "; ".join(structural_risks) if structural_risks else "none",
    }


def audit_e8p_packed_rhs_tiled_family_source(
    source: str,
    kernel_names: tuple[str, ...] = _PACKED_RHS_TILED_KERNELS,
) -> dict[str, Any]:
    kernel_reports: list[dict[str, Any]] = []
    rejected_kernels: list[str] = []
    for kernel_name in kernel_names:
        try:
            kernel = _extract_kernel_source(source, kernel_name)
        except ValueError:
            kernel_reports.append(
                {
                    "kernel_name": kernel_name,
                    "present": False,
                    "decision": "not_present",
                }
            )
            continue
        has_threadgroup_staged_b = "threadgroup half staged_b[" in kernel
        decodes_e8p_to_staged_b = (
            "staged_b[" in kernel and "mlx_vq_decode_e8p_value(" in kernel
        )
        barrier_count = kernel.count("threadgroup_barrier(")
        has_tensorops_matmul = "matmul_op.run" in kernel and (
            "mpp::tensor_ops::matmul2d" in kernel
            or "get_right_input_cooperative_tensor" in kernel
        )
        loads_b_fragments_from_staged_b = (
            ("b_t.thread_elements()[" in kernel or "b_t[" in kernel)
            and "staged_b[" in kernel
        )
        rejects_kernel = (
            has_threadgroup_staged_b
            and decodes_e8p_to_staged_b
            and barrier_count > 0
            and has_tensorops_matmul
            and loads_b_fragments_from_staged_b
        )
        if rejects_kernel:
            rejected_kernels.append(kernel_name)
        structural_risks: list[str] = []
        if has_threadgroup_staged_b and decodes_e8p_to_staged_b:
            structural_risks.append(
                "packed E8P RHS codes are decoded into threadgroup staged_b before TensorOps"
            )
        if loads_b_fragments_from_staged_b:
            structural_risks.append(
                "right-hand TensorOps fragments are populated from staged decoded B storage"
            )
        if barrier_count:
            structural_risks.append(f"{barrier_count} threadgroup barriers in the kernel body")
        kernel_reports.append(
            {
                "kernel_name": kernel_name,
                "present": True,
                "has_threadgroup_staged_b": has_threadgroup_staged_b,
                "decodes_e8p_to_staged_b": decodes_e8p_to_staged_b,
                "threadgroup_barrier_count": barrier_count,
                "has_tensorops_matmul": has_tensorops_matmul,
                "loads_b_fragments_from_staged_b": loads_b_fragments_from_staged_b,
                "decision": (
                    "reject_decoded_b_staged_packed_rhs_kernel"
                    if rejects_kernel
                    else "needs_manual_review"
                ),
                "structural_risk": "; ".join(structural_risks) if structural_risks else "none",
            }
        )
    present_kernel_count = sum(1 for report in kernel_reports if report.get("present"))
    rejects_family = bool(rejected_kernels) and len(rejected_kernels) == present_kernel_count
    required_next_features = [
        "avoid_decoded_b_threadgroup_staging",
        "reuse_decode_outside_threadgroup_b_tile",
        "preserve_compressed_rhs_storage",
        "change_rhs_layout_or_kernel_family",
    ]
    return {
        "kernel_names": list(kernel_names),
        "present_kernel_count": present_kernel_count,
        "rejected_kernel_count": len(rejected_kernels),
        "rejected_kernels": rejected_kernels,
        "kernel_reports": kernel_reports,
        "decision": (
            "reject_decoded_b_staged_packed_rhs_family"
            if rejects_family
            else "needs_manual_review"
        ),
        "required_next_features": required_next_features if rejects_family else [],
    }


def audit_e8p_sign_nibble_schedule_source(
    source: str,
    *,
    benchmark_context: dict[str, Any] | None = None,
    scalar_kernel_name: str = _SIGN_NIBBLE_SCALAR_SORTED_KERNEL,
    kernel_names: tuple[str, ...] = _SIGN_NIBBLE_TENSOROPS_KERNELS,
) -> dict[str, Any]:
    benchmark_context = benchmark_context or {}
    try:
        scalar_kernel = _extract_kernel_source(source, scalar_kernel_name)
        scalar_sorted_route_kernel_present = True
    except ValueError:
        scalar_kernel = ""
        scalar_sorted_route_kernel_present = False

    scalar_consumes_sign_nibbles = (
        "sign_low_nibble_tiles" in scalar_kernel
        and "sign_high_nibble_tiles" in scalar_kernel
        and "abs_index_tiles" in scalar_kernel
        and "codeword_scale_slots" in scalar_kernel
    )
    scalar_reconstructs_sign_byte = (
        "<<" in scalar_kernel
        and "4u" in scalar_kernel
        and "sign_low_nibble_tiles" in scalar_kernel
        and "sign_high_nibble_tiles" in scalar_kernel
    )

    kernel_reports: list[dict[str, Any]] = []
    rejected_kernels: list[str] = []
    per_fragment_rejected_kernels: list[str] = []
    micro_lut_rejected_kernels: list[str] = []
    benchmark_rejects_sign_nibble_tensorops = (
        benchmark_context.get("all_lane_s_pass") is False
        or benchmark_context.get("all_parity_pass") is False
    ) and (
        benchmark_context.get("sign_nibble_tensorops_comparison_count", 0) > 0
        or benchmark_context.get("sign_nibble_tensorops_worst_ratio_to_q2") is not None
    )
    benchmark_rejects_micro_lut_tensorops = (
        benchmark_context.get("all_lane_s_pass") is False
        or benchmark_context.get("all_parity_pass") is False
    ) and (
        benchmark_context.get("sign_nibble_micro_lut_tensorops_comparison_count", 0) > 0
        or benchmark_context.get("sign_nibble_micro_lut_tensorops_worst_ratio_to_q2")
        is not None
    )
    for kernel_name in kernel_names:
        try:
            kernel = _extract_kernel_source(source, kernel_name)
        except ValueError:
            kernel_reports.append(
                {
                    "kernel_name": kernel_name,
                    "present": False,
                    "decision": "not_present",
                }
            )
            continue
        has_tensorops_matmul = "matmul_op.run" in kernel and (
            "mpp::tensor_ops::matmul2d" in kernel
            or "get_right_input_cooperative_tensor" in kernel
        )
        has_threadgroup_staged_b = "threadgroup half staged_b[" in kernel
        has_lane_local_fragment_buffer = (
            "threadgroup half lane" in kernel
            or "threadgroup half shared" in kernel
            or "threadgroup half b_fragment" in kernel
        )
        barrier_count = kernel.count("threadgroup_barrier(")
        reconstructs_sign_byte = (
            "sign_low_nibble_tiles" in kernel
            and "sign_high_nibble_tiles" in kernel
            and "<<" in kernel
            and "4u" in kernel
        )
        uses_sign_nibble_tiles = (
            "sign_low_nibble_tiles" in kernel and "sign_high_nibble_tiles" in kernel
        )
        uses_abs_index_tiles = "abs_index_tiles" in kernel
        uses_parity_tiles = "parity_tiles" in kernel
        uses_scale_tiles = "scale_tiles" in kernel
        uses_codeword_scale_slots = "codeword_scale_slots" in kernel
        uses_micro_lut_slot_maps = (
            "sign_low_nibble_slots" in kernel and "sign_high_nibble_slots" in kernel
        )
        uses_micro_lut_luts = (
            "sign_low_nibble_lut[" in kernel and "sign_high_nibble_lut[" in kernel
        )
        uses_abs_index_lut = "abs_index_lut[" in kernel and "abs_index_slots" in kernel
        decodes_nibble_values = "mlx_vq_decode_e8p_nibble_value" in kernel
        populates_b_fragments = (
            ("b_t[" in kernel or "b_t.thread_elements()[" in kernel)
            and decodes_nibble_values
        )
        decodes_nibble_values_per_b_fragment = (
            has_tensorops_matmul
            and uses_sign_nibble_tiles
            and uses_abs_index_tiles
            and uses_parity_tiles
            and uses_scale_tiles
            and decodes_nibble_values
            and populates_b_fragments
        )
        populates_micro_lut_b_fragments = (
            "b_t[" in kernel or "b_t.thread_elements()[" in kernel
        ) and "mlx_vq_decode_e8p_split_value" in kernel
        decodes_micro_lut_values_per_b_fragment = (
            has_tensorops_matmul
            and uses_micro_lut_slot_maps
            and uses_micro_lut_luts
            and uses_abs_index_lut
            and uses_scale_tiles
            and populates_micro_lut_b_fragments
        )
        decodes_from_nibbles = reconstructs_sign_byte and "mlx_vq_decode_e8p_split_value" in kernel
        loads_b_fragments_from_staged_b = (
            ("b_t.thread_elements()[" in kernel or "b_t[" in kernel)
            and "staged_b[" in kernel
        )
        rejects_decoded_b_staged_kernel = (
            has_tensorops_matmul
            and decodes_from_nibbles
            and (
                has_threadgroup_staged_b
                or has_lane_local_fragment_buffer
                or loads_b_fragments_from_staged_b
            )
        )
        rejects_per_fragment_nibble_kernel = (
            decodes_nibble_values_per_b_fragment
            and not has_threadgroup_staged_b
            and not has_lane_local_fragment_buffer
            and barrier_count == 0
            and benchmark_rejects_sign_nibble_tensorops
        )
        rejects_micro_lut_kernel = (
            decodes_micro_lut_values_per_b_fragment
            and not has_threadgroup_staged_b
            and not has_lane_local_fragment_buffer
            and barrier_count == 0
            and benchmark_rejects_micro_lut_tensorops
        )
        rejects_kernel = (
            rejects_decoded_b_staged_kernel
            or rejects_per_fragment_nibble_kernel
            or rejects_micro_lut_kernel
        )
        if rejects_kernel:
            rejected_kernels.append(kernel_name)
        if rejects_per_fragment_nibble_kernel:
            per_fragment_rejected_kernels.append(kernel_name)
        if rejects_micro_lut_kernel:
            per_fragment_rejected_kernels.append(kernel_name)
            micro_lut_rejected_kernels.append(kernel_name)
        structural_risks: list[str] = []
        if has_threadgroup_staged_b:
            structural_risks.append(
                "sign-nibble E8P values are decoded into threadgroup staged_b before TensorOps"
            )
        if has_lane_local_fragment_buffer:
            structural_risks.append(
                "lane-local threadgroup fragment buffering repeats the rejected shared-n shape"
            )
        if loads_b_fragments_from_staged_b:
            structural_risks.append(
                "right-hand TensorOps fragments are populated from staged decoded B storage"
            )
        if reconstructs_sign_byte:
            structural_risks.append(
                "sign bytes are reconstructed from nibbles inside the TensorOps schedule"
            )
        if decodes_nibble_values_per_b_fragment:
            structural_risks.append(
                "B cooperative tensor fragments reconstruct nibble signs, abs indices, parity, and scales per fragment"
            )
        if decodes_micro_lut_values_per_b_fragment:
            structural_risks.append(
                "B cooperative tensor fragments reconstruct micro-LUT sign slots, abs-index LUT entries, parity, and scales per fragment"
            )
        decision = "needs_manual_review"
        if rejects_decoded_b_staged_kernel:
            decision = "reject_decoded_b_staged_sign_nibble_kernel"
        elif rejects_micro_lut_kernel:
            decision = "reject_per_fragment_micro_lut_tensorops_kernel"
        elif rejects_per_fragment_nibble_kernel:
            decision = "reject_per_fragment_nibble_tensorops_kernel"
        kernel_reports.append(
            {
                "kernel_name": kernel_name,
                "present": True,
                "has_tensorops_matmul": has_tensorops_matmul,
                "has_threadgroup_staged_b": has_threadgroup_staged_b,
                "has_lane_local_fragment_buffer": has_lane_local_fragment_buffer,
                "threadgroup_barrier_count": barrier_count,
                "uses_sign_nibble_tiles": uses_sign_nibble_tiles,
                "uses_abs_index_tiles": uses_abs_index_tiles,
                "uses_parity_tiles": uses_parity_tiles,
                "uses_scale_tiles": uses_scale_tiles,
                "uses_codeword_scale_slots": uses_codeword_scale_slots,
                "uses_micro_lut_slot_maps": uses_micro_lut_slot_maps,
                "uses_micro_lut_luts": uses_micro_lut_luts,
                "uses_abs_index_lut": uses_abs_index_lut,
                "reconstructs_sign_byte_from_nibbles": reconstructs_sign_byte,
                "decodes_from_sign_nibbles": decodes_from_nibbles,
                "decodes_nibble_values_per_b_fragment": decodes_nibble_values_per_b_fragment,
                "decodes_micro_lut_values_per_b_fragment": (
                    decodes_micro_lut_values_per_b_fragment
                ),
                "loads_b_fragments_from_staged_b": loads_b_fragments_from_staged_b,
                "benchmark_rejects_sign_nibble_tensorops": benchmark_rejects_sign_nibble_tensorops,
                "benchmark_rejects_micro_lut_tensorops": benchmark_rejects_micro_lut_tensorops,
                "decision": decision,
                "structural_risk": "; ".join(structural_risks) if structural_risks else "none",
            }
        )

    present_tensorops_count = sum(1 for report in kernel_reports if report.get("present"))
    q2_shaped_tensorops_kernel_present = present_tensorops_count > 0
    required_next_features = [
        "match_q2_bk64_bn64_tensorops_geometry",
        "decode_nibble_sign_masks_without_per_fragment_sign_byte_reconstruction",
        "avoid_decoded_b_threadgroup_staging",
        "avoid_lane_local_barrier_fragment_buffer",
        "preserve_compressed_rhs_storage",
        "preserve_codeword_scale_slots",
    ]
    if micro_lut_rejected_kernels:
        decision = "reject_per_fragment_micro_lut_tensorops_schedule"
        required_next_features = [
            "broader_decode_reuse_scope",
            "avoid_per_fragment_micro_lut_slot_reconstruction",
            "avoid_per_fragment_abs_index_lut_reconstruction",
            "preserve_compressed_rhs_storage",
            "preserve_codeword_scale_slots",
            "change_rhs_layout_or_kernel_family",
        ]
    elif per_fragment_rejected_kernels:
        decision = "reject_per_fragment_nibble_tensorops_schedule"
        required_next_features = [
            "broader_decode_reuse_scope",
            "avoid_per_fragment_nibble_abs_scale_reconstruction",
            "preserve_compressed_rhs_storage",
            "preserve_codeword_scale_slots",
            "change_rhs_layout_or_kernel_family",
        ]
    elif rejected_kernels:
        decision = "reject_decoded_b_staged_sign_nibble_schedule"
    elif scalar_sorted_route_kernel_present and not q2_shaped_tensorops_kernel_present:
        decision = "missing_q2_shaped_sign_nibble_tensorops_schedule"
    elif not scalar_sorted_route_kernel_present:
        decision = "missing_sign_nibble_sorted_route_correctness_kernel"
    else:
        decision = "needs_manual_review"
    return {
        "scalar_kernel_name": scalar_kernel_name,
        "scalar_sorted_route_kernel_present": scalar_sorted_route_kernel_present,
        "scalar_consumes_sign_nibbles": scalar_consumes_sign_nibbles,
        "scalar_reconstructs_sign_byte_from_nibbles": scalar_reconstructs_sign_byte,
        "kernel_names": list(kernel_names),
        "q2_shaped_tensorops_kernel_present": q2_shaped_tensorops_kernel_present,
        "present_tensorops_kernel_count": present_tensorops_count,
        "rejected_kernel_count": len(rejected_kernels),
        "rejected_kernels": rejected_kernels,
        "per_fragment_rejected_kernels": per_fragment_rejected_kernels,
        "micro_lut_rejected_kernels": micro_lut_rejected_kernels,
        "kernel_reports": kernel_reports,
        "benchmark_rejects_sign_nibble_tensorops": benchmark_rejects_sign_nibble_tensorops,
        "benchmark_rejects_micro_lut_tensorops": benchmark_rejects_micro_lut_tensorops,
        "benchmark_context": benchmark_context,
        "decision": decision,
        "required_next_features": required_next_features,
    }


def audit_e8p_sign_plane_schedule_source(
    source: str,
    *,
    benchmark_context: dict[str, Any] | None = None,
    scalar_kernel_name: str = _SIGN_PLANE_SCALAR_SORTED_KERNEL,
    kernel_names: tuple[str, ...] = _SIGN_PLANE_TENSOROPS_KERNELS,
) -> dict[str, Any]:
    benchmark_context = benchmark_context or {}
    try:
        scalar_kernel = _extract_kernel_source(source, scalar_kernel_name)
        scalar_sorted_route_kernel_present = True
    except ValueError:
        scalar_kernel = ""
        scalar_sorted_route_kernel_present = False

    scalar_consumes_sign_planes = (
        "sign_bit_planes" in scalar_kernel
        and "abs_index_tiles" in scalar_kernel
        and "codeword_scale_slots" in scalar_kernel
    )
    scalar_reconstructs_signs_from_bit_planes = (
        "sign_bit_planes" in scalar_kernel
        and ">> ulong(" in scalar_kernel
        and "bit < 8" in scalar_kernel
    )
    benchmark_rejects_sign_plane_tensorops = (
        benchmark_context.get("all_lane_s_pass") is False
        or benchmark_context.get("all_parity_pass") is False
    ) and (
        benchmark_context.get("sign_plane_tensorops_comparison_count", 0) > 0
        or benchmark_context.get("sign_plane_tensorops_worst_ratio_to_q2") is not None
    )

    kernel_reports: list[dict[str, Any]] = []
    rejected_kernels: list[str] = []
    per_fragment_rejected_kernels: list[str] = []
    for kernel_name in kernel_names:
        try:
            kernel = _extract_kernel_source(source, kernel_name)
        except ValueError:
            kernel_reports.append(
                {
                    "kernel_name": kernel_name,
                    "present": False,
                    "decision": "not_present",
                }
            )
            continue
        has_tensorops_matmul = "matmul_op.run" in kernel and (
            "mpp::tensor_ops::matmul2d" in kernel
            or "get_right_input_cooperative_tensor" in kernel
        )
        has_threadgroup_staged_b = "threadgroup half staged_b[" in kernel
        has_lane_local_fragment_buffer = (
            "threadgroup half lane" in kernel
            or "threadgroup half shared" in kernel
            or "threadgroup half b_fragment" in kernel
        )
        barrier_count = kernel.count("threadgroup_barrier(")
        uses_sign_bit_planes = "sign_bit_planes" in kernel
        uses_abs_index_tiles = "abs_index_tiles" in kernel
        uses_scale_tiles = "scale_tiles" in kernel
        uses_codeword_scale_slots = "codeword_scale_slots" in kernel
        reconstructs_signs_from_bit_planes = (
            uses_sign_bit_planes
            and "bit < 8" in kernel
            and ">> ulong(" in kernel
            and ("n_local" in kernel or "n_in_tile" in kernel)
        )
        populates_b_fragments = (
            "b_t[" in kernel or "b_t.thread_elements()[" in kernel
        ) and "mlx_vq_decode_e8p_split_value" in kernel
        decodes_sign_plane_values_per_b_fragment = (
            has_tensorops_matmul
            and uses_sign_bit_planes
            and uses_abs_index_tiles
            and uses_scale_tiles
            and reconstructs_signs_from_bit_planes
            and populates_b_fragments
        )
        rejects_per_fragment_kernel = (
            decodes_sign_plane_values_per_b_fragment
            and not has_threadgroup_staged_b
            and not has_lane_local_fragment_buffer
            and barrier_count == 0
            and benchmark_rejects_sign_plane_tensorops
        )
        if rejects_per_fragment_kernel:
            rejected_kernels.append(kernel_name)
            per_fragment_rejected_kernels.append(kernel_name)

        structural_risks: list[str] = []
        if has_threadgroup_staged_b:
            structural_risks.append(
                "sign-plane E8P values are decoded into threadgroup staged_b before TensorOps"
            )
        if has_lane_local_fragment_buffer:
            structural_risks.append(
                "lane-local threadgroup fragment buffering repeats an already rejected shape"
            )
        if reconstructs_signs_from_bit_planes:
            structural_risks.append(
                "sign bytes are reconstructed from output-column bit planes inside the TensorOps schedule"
            )
        if decodes_sign_plane_values_per_b_fragment:
            structural_risks.append(
                "B cooperative tensor fragments reconstruct sign planes, abs indices, parity, and scales per fragment"
            )
        kernel_reports.append(
            {
                "kernel_name": kernel_name,
                "present": True,
                "has_tensorops_matmul": has_tensorops_matmul,
                "has_threadgroup_staged_b": has_threadgroup_staged_b,
                "has_lane_local_fragment_buffer": has_lane_local_fragment_buffer,
                "threadgroup_barrier_count": barrier_count,
                "uses_sign_bit_planes": uses_sign_bit_planes,
                "uses_abs_index_tiles": uses_abs_index_tiles,
                "uses_scale_tiles": uses_scale_tiles,
                "uses_codeword_scale_slots": uses_codeword_scale_slots,
                "reconstructs_signs_from_bit_planes": reconstructs_signs_from_bit_planes,
                "decodes_sign_plane_values_per_b_fragment": (
                    decodes_sign_plane_values_per_b_fragment
                ),
                "benchmark_rejects_sign_plane_tensorops": (
                    benchmark_rejects_sign_plane_tensorops
                ),
                "decision": (
                    "reject_per_fragment_sign_plane_tensorops_kernel"
                    if rejects_per_fragment_kernel
                    else "needs_manual_review"
                ),
                "structural_risk": "; ".join(structural_risks)
                if structural_risks
                else "none",
            }
        )

    present_tensorops_count = sum(1 for report in kernel_reports if report.get("present"))
    q2_shaped_tensorops_kernel_present = present_tensorops_count > 0
    if per_fragment_rejected_kernels:
        decision = "reject_per_fragment_sign_plane_tensorops_schedule"
        required_next_features = [
            "broader_decode_reuse_scope",
            "avoid_per_fragment_sign_plane_abs_scale_reconstruction",
            "preserve_compressed_rhs_storage",
            "preserve_codeword_scale_slots",
            "change_rhs_layout_or_kernel_family",
        ]
    elif scalar_sorted_route_kernel_present and not q2_shaped_tensorops_kernel_present:
        decision = "missing_q2_shaped_sign_plane_tensorops_schedule"
        required_next_features = [
            "match_q2_bk64_bn64_tensorops_geometry",
            "decode_sign_plane_masks_without_per_fragment_sign_byte_reconstruction",
            "avoid_decoded_b_threadgroup_staging",
            "avoid_lane_local_barrier_fragment_buffer",
            "preserve_compressed_rhs_storage",
            "preserve_codeword_scale_slots",
        ]
    elif not scalar_sorted_route_kernel_present:
        decision = "missing_sign_plane_sorted_route_correctness_kernel"
        required_next_features = [
            "native_sign_plane_sorted_route_correctness_oracle",
            "preserve_compressed_rhs_storage",
            "preserve_codeword_scale_slots",
        ]
    else:
        decision = "needs_manual_review"
        required_next_features = []
    return {
        "scalar_kernel_name": scalar_kernel_name,
        "scalar_sorted_route_kernel_present": scalar_sorted_route_kernel_present,
        "scalar_consumes_sign_planes": scalar_consumes_sign_planes,
        "scalar_reconstructs_signs_from_bit_planes": (
            scalar_reconstructs_signs_from_bit_planes
        ),
        "kernel_names": list(kernel_names),
        "q2_shaped_tensorops_kernel_present": q2_shaped_tensorops_kernel_present,
        "present_tensorops_kernel_count": present_tensorops_count,
        "rejected_kernel_count": len(rejected_kernels),
        "rejected_kernels": rejected_kernels,
        "per_fragment_rejected_kernels": per_fragment_rejected_kernels,
        "kernel_reports": kernel_reports,
        "benchmark_rejects_sign_plane_tensorops": benchmark_rejects_sign_plane_tensorops,
        "benchmark_context": benchmark_context,
        "decision": decision,
        "required_next_features": required_next_features,
    }


def audit_e8p_expert_kblock_schedule_source(
    source: str,
    *,
    benchmark_context: dict[str, Any] | None = None,
    kernel_names: tuple[str, ...] = _EXPERT_KBLOCK_FACTOR_DECODE_KERNELS,
) -> dict[str, Any]:
    benchmark_context = benchmark_context or {}
    schedule_decision = benchmark_context.get("expert_kblock_schedule_decision")
    factor_signal = benchmark_context.get("expert_kblock_factor_signal")
    requested_features = benchmark_context.get("expert_kblock_required_kernel_features")
    if not isinstance(requested_features, list):
        requested_features = []
    tensorops_benchmark_count = int(
        benchmark_context.get("expert_kblock_tensorops_comparison_count") or 0
    )
    tensorops_worst_ratio = benchmark_context.get(
        "expert_kblock_tensorops_worst_ratio_to_q2"
    )
    current_tensorops_benchmark_rejected = (
        tensorops_benchmark_count > 0
        and isinstance(tensorops_worst_ratio, (int, float))
        and float(tensorops_worst_ratio) > 1.25
    )
    schedule_requested = (
        schedule_decision == "probe_expert_kblock_factor_decode_reuse"
        or factor_signal == "expert_kblock_factor_reuse_present"
        or tensorops_benchmark_count > 0
        or int(benchmark_context.get("expert_kblock_native_comparison_count") or 0) > 0
    )

    kernel_reports: list[dict[str, Any]] = []
    present_kernels: list[str] = []
    rejected_v2_kernels: list[str] = []
    invalid_v2_kernels: list[str] = []
    kblock_reduction_missing_v2_kernels: list[str] = []
    for kernel_name in kernel_names:
        try:
            kernel = _extract_kernel_source(source, kernel_name)
        except ValueError:
            kernel_reports.append(
                {
                    "kernel_name": kernel_name,
                    "present": False,
                    "decision": "not_present",
                }
            )
            continue
        present_kernels.append(kernel_name)
        has_tensorops_matmul = "matmul_op.run" in kernel and (
            "mpp::tensor_ops::matmul2d" in kernel
            or "get_right_input_cooperative_tensor" in kernel
        )
        uses_factor_luts = (
            "sign_byte_lut[" in kernel
            or "sign_low_nibble_lut[" in kernel
            or "sign_bit_planes" in kernel
        ) and ("abs_index_lut[" in kernel or "abs_index_tiles" in kernel)
        has_threadgroup_staged_b = "threadgroup half staged_b[" in kernel or (
            "threadgroup half Ws[" in kernel
        )
        has_lane_local_fragment_buffer = (
            "threadgroup half lane" in kernel
            or "threadgroup half shared" in kernel
            or "threadgroup half b_fragment" in kernel
        )
        populates_b_fragments = "b_t[" in kernel or "b_t.thread_elements()[" in kernel
        decodes_factor_values_per_b_fragment = (
            has_tensorops_matmul
            and uses_factor_luts
            and populates_b_fragments
            and (
                "scale_tiles[" in kernel
                or "mlx_vq_decode_e8p_split_value" in kernel
                or "mlx_vq_decode_e8p_value" in kernel
            )
        )
        single_kblock_completion = (
            kernel_name == _EXPERT_KBLOCK_TENSOROPS_V2_KERNEL
            and "contract_k_block != 0u" in kernel
            and "return;" in kernel
            and "routes_in_tile * output_dims" in kernel
            and "out[" in kernel
        )
        structural_risks: list[str] = []
        if not has_tensorops_matmul:
            structural_risks.append("candidate does not reach TensorOps")
        if single_kblock_completion:
            structural_risks.append(
                "candidate completes all K-block work from contract_k_block zero"
            )
        if has_threadgroup_staged_b:
            structural_risks.append("candidate still stages decoded B in threadgroup memory")
        if has_lane_local_fragment_buffer:
            structural_risks.append("candidate still uses lane-local fragment buffering")
        if not uses_factor_luts:
            structural_risks.append("candidate does not visibly consume compressed factor inputs")
        if decodes_factor_values_per_b_fragment:
            structural_risks.append(
                "candidate still reconstructs factor-derived B values per TensorOps fragment"
            )
        report_decision = "needs_manual_review"
        if schedule_requested and single_kblock_completion:
            report_decision = "reject_v2_single_kblock_completion"
            kblock_reduction_missing_v2_kernels.append(kernel_name)
        if (
            schedule_requested
            and current_tensorops_benchmark_rejected
            and kernel_name == _EXPERT_KBLOCK_TENSOROPS_V2_KERNEL
            and (not has_tensorops_matmul or not uses_factor_luts)
            and not single_kblock_completion
        ):
            if not has_tensorops_matmul:
                report_decision = "reject_v2_tensorops_missing"
            else:
                report_decision = "reject_v2_factor_inputs_missing"
            invalid_v2_kernels.append(kernel_name)
        if (
            schedule_requested
            and current_tensorops_benchmark_rejected
            and kernel_name == _EXPERT_KBLOCK_TENSOROPS_V2_KERNEL
            and has_tensorops_matmul
            and (
                has_threadgroup_staged_b
                or has_lane_local_fragment_buffer
                or decodes_factor_values_per_b_fragment
            )
        ):
            if has_threadgroup_staged_b:
                report_decision = "reject_v2_decoded_b_staging"
            elif has_lane_local_fragment_buffer:
                report_decision = "reject_v2_lane_local_fragment_buffer"
            else:
                report_decision = "reject_v2_per_fragment_factor_reconstruction"
            rejected_v2_kernels.append(kernel_name)
        if (
            schedule_requested
            and kernel_name == _EXPERT_KBLOCK_TENSOROPS_KERNEL
            and has_tensorops_matmul
            and uses_factor_luts
        ):
            report_decision = "tensorops_candidate_present_needs_benchmark"
        if (
            schedule_requested
            and kernel_name == _EXPERT_KBLOCK_SCALAR_KERNEL
            and not has_tensorops_matmul
        ):
            report_decision = "scalar_consumer_present_tensorops_missing"
        kernel_reports.append(
            {
                "kernel_name": kernel_name,
                "present": True,
                "has_tensorops_matmul": has_tensorops_matmul,
                "uses_factor_luts": uses_factor_luts,
                "has_threadgroup_staged_b": has_threadgroup_staged_b,
                "has_lane_local_fragment_buffer": has_lane_local_fragment_buffer,
                "populates_b_fragments": populates_b_fragments,
                "decodes_factor_values_per_b_fragment": (
                    decodes_factor_values_per_b_fragment
                ),
                "single_kblock_completion": single_kblock_completion,
                "decision": report_decision,
                "structural_risk": "; ".join(structural_risks)
                if structural_risks
                else "none",
            }
        )

    tensorops_kernel_present = _EXPERT_KBLOCK_TENSOROPS_KERNEL in present_kernels
    scalar_kernel_present = _EXPERT_KBLOCK_SCALAR_KERNEL in present_kernels
    v2_kernel_present = _EXPERT_KBLOCK_TENSOROPS_V2_KERNEL in present_kernels
    v2_kernel_kblock_reduction_missing = bool(kblock_reduction_missing_v2_kernels)
    v2_kernel_rejected = bool(rejected_v2_kernels)
    v2_kernel_invalid = bool(invalid_v2_kernels)
    if schedule_requested and not present_kernels:
        decision = "missing_expert_kblock_factor_decode_kernel"
        required_next_features = list(_EXPERT_KBLOCK_REQUIRED_FEATURES)
        rejected_next_steps = [
            "fixed_codeword_position_output_column_grouping",
            "current_shared_decode_factor_reuse",
            "current_shared_n_factor_reuse",
            "current_sign_plane_tensorops",
            "current_micro_lut_tensorops",
        ]
    elif schedule_requested and v2_kernel_kblock_reduction_missing:
        decision = "reject_expert_kblock_tensorops_v2_kblock_reduction_missing"
        required_next_features = [
            "add_kblock_partial_accumulation_or_two_pass_reduction",
            "make_each_kblock_workgroup_contribute_only_its_k_slice",
            "build_tensorops_v2_before_benchmark",
            *_EXPERT_KBLOCK_REQUIRED_FEATURES,
        ]
        rejected_next_steps = [
            "do_not_benchmark_single_kblock_completion_v2",
            "do_not_retime_current_expert_kblock_tensorops",
            "do_not_call_dispatch_contract_skeleton_tensorops_speed_evidence",
        ]
    elif schedule_requested and v2_kernel_invalid:
        decision = "reject_expert_kblock_tensorops_v2_invalid_source"
        required_next_features = [
            "build_tensorops_v2_before_benchmark",
            "consume_compressed_factor_inputs_in_v2",
            *_EXPERT_KBLOCK_REQUIRED_FEATURES,
        ]
        rejected_next_steps = [
            "do_not_benchmark_non_tensorops_expert_kblock_v2",
            "do_not_benchmark_factorless_expert_kblock_v2",
            "do_not_retime_current_expert_kblock_tensorops",
        ]
    elif schedule_requested and v2_kernel_rejected:
        decision = "reject_expert_kblock_tensorops_v2_near_duplicate"
        required_next_features = [
            "materially_different_rhs_layout_or_kernel_family",
            "broaden_decode_reuse_scope_before_tensorops_fragment_fill",
            *_EXPERT_KBLOCK_REQUIRED_FEATURES,
        ]
        rejected_next_steps = [
            "do_not_benchmark_near_duplicate_expert_kblock_v2",
            "do_not_retime_current_expert_kblock_tensorops",
            "do_not_lightly_tweak_current_expert_kblock_tensorops",
        ]
    elif (
        schedule_requested
        and tensorops_kernel_present
        and current_tensorops_benchmark_rejected
        and not v2_kernel_present
    ):
        decision = "current_expert_kblock_tensorops_speed_rejected_v2_missing"
        required_next_features = [
            "materially_different_rhs_layout_or_kernel_family",
            *_EXPERT_KBLOCK_REQUIRED_FEATURES,
        ]
        rejected_next_steps = [
            "do_not_retime_current_expert_kblock_tensorops",
            "do_not_air_shape_sweep_current_expert_kblock_tensorops",
            "do_not_lightly_tweak_current_expert_kblock_tensorops",
        ]
    elif schedule_requested and tensorops_kernel_present:
        decision = "expert_kblock_tensorops_candidate_present_needs_benchmark"
        required_next_features = list(_EXPERT_KBLOCK_REQUIRED_FEATURES)
        rejected_next_steps = [
            "benchmark_expert_kblock_tensorops_factor_decode_reuse_kernel",
        ]
    elif schedule_requested and scalar_kernel_present and not tensorops_kernel_present:
        decision = "scalar_expert_kblock_consumer_present_tensorops_missing"
        required_next_features = list(_EXPERT_KBLOCK_REQUIRED_FEATURES)
        rejected_next_steps = [
            "benchmark_scalar_expert_kblock_consumer_as_speed_path",
            "build_expert_kblock_tensorops_factor_decode_reuse_kernel",
        ]
    elif schedule_requested:
        decision = "needs_manual_review"
        required_next_features = list(_EXPERT_KBLOCK_REQUIRED_FEATURES)
        rejected_next_steps = []
    else:
        decision = "expert_kblock_factor_decode_not_requested"
        required_next_features = []
        rejected_next_steps = []

    return {
        "kernel_names": list(kernel_names),
        "present_kernel_count": len(present_kernels),
        "present_kernels": present_kernels,
        "kernel_reports": kernel_reports,
        "schedule_requested": schedule_requested,
        "benchmark_decision": schedule_decision,
        "factor_signal": factor_signal,
        "requested_kernel_features": requested_features,
        "current_tensorops_benchmark_rejected": current_tensorops_benchmark_rejected,
        "v2_kernel_present": v2_kernel_present,
        "v2_kernel_kblock_reduction_missing": v2_kernel_kblock_reduction_missing,
        "kblock_reduction_missing_v2_kernels": kblock_reduction_missing_v2_kernels,
        "v2_kernel_invalid": v2_kernel_invalid,
        "invalid_v2_kernels": invalid_v2_kernels,
        "v2_kernel_rejected": v2_kernel_rejected,
        "rejected_v2_kernels": rejected_v2_kernels,
        "decision": decision,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_expert_kblock_primitive_shape_source(
    source: str,
    *,
    benchmark_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    benchmark_context = benchmark_context or {}
    tensorops_benchmark_count = int(
        benchmark_context.get("expert_kblock_tensorops_comparison_count") or 0
    )
    tensorops_worst_ratio = benchmark_context.get(
        "expert_kblock_tensorops_worst_ratio_to_q2"
    )
    current_tensorops_benchmark_rejected = (
        tensorops_benchmark_count > 0
        and isinstance(tensorops_worst_ratio, (int, float))
        and float(tensorops_worst_ratio) > 1.25
    )
    dispatch_grid_locked = "MTL::Size::Make(n_tiles, num_route_tiles, 1)" in source
    fixed_threadgroup_threads = 128 if "MTL::Size::Make(128, 1, 1)" in source else None
    uses_current_tensorops_pso = (
        'cache.get("nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul")'
        in source
    )
    enforces_bn64_codewords8 = "bn != 64 || codewords != 8" in source
    slot_layout_rank = (
        5
        if "sign_byte_slots.ndim() != 5" in source and "scale_tiles.ndim() != 5" in source
        else None
    )
    shape_contract_locked = (
        dispatch_grid_locked
        and fixed_threadgroup_threads == 128
        and uses_current_tensorops_pso
        and enforces_bn64_codewords8
        and slot_layout_rank == 5
    )
    if current_tensorops_benchmark_rejected and shape_contract_locked:
        decision = "current_primitive_shape_blocks_v2_reuse_scope"
        required_next_features = [
            "redesign_expert_kblock_primitive_dispatch_contract",
            "dispatch_over_reuse_scope_before_output_tile",
            "broaden_decode_reuse_scope_before_tensorops_fragment_fill",
            "preserve_compressed_rhs_storage",
            "preserve_codeword_scale_slots",
        ]
        rejected_next_steps = [
            "do_not_add_v2_wrapper_on_current_n_tiles_x_route_tiles_dispatch",
            "do_not_retime_current_expert_kblock_tensorops",
            "do_not_lightly_tweak_current_expert_kblock_tensorops",
        ]
        structural_risk = (
            "current primitive dispatches one threadgroup grid over n_tiles x route_tiles, "
            "so factor decode reuse cannot be broadened across output tiles without a new "
            "dispatch/storage contract"
        )
    else:
        decision = "needs_manual_review"
        required_next_features = []
        rejected_next_steps = []
        structural_risk = "none"
    return {
        "decision": decision,
        "current_tensorops_benchmark_rejected": current_tensorops_benchmark_rejected,
        "dispatch_grid": "n_tiles_x_num_route_tiles" if dispatch_grid_locked else "unknown",
        "fixed_threadgroup_threads": fixed_threadgroup_threads,
        "uses_current_tensorops_pso": uses_current_tensorops_pso,
        "enforces_bn64_codewords8": enforces_bn64_codewords8,
        "slot_layout_rank": slot_layout_rank,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
        "structural_risk": structural_risk,
    }


def audit_e8p_expert_kblock_dispatch_contract_source(source: str) -> dict[str, Any]:
    normalized = re.sub(r"\s+", " ", source)
    v2_kernel_name = "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul"
    v2_primitive_present = (
        "NaxE8PExpertKBlockFactorReuseRHSSortedTensorOpsV2Matmul::eval_gpu" in source
        or v2_kernel_name in source
    )
    uses_v2_pso = f'cache.get("{v2_kernel_name}")' in source
    dispatches_reuse_scope = (
        "MTL::Size::Make(experts, k_blocks, num_route_tiles)" in normalized
    )
    dispatches_current_grid = "MTL::Size::Make(n_tiles, num_route_tiles, 1)" in normalized
    carries_output_tiles_inside_scope = "set_bytes(n_tiles" in source or "uint32_t n_tiles" in source
    fixed_threadgroup_threads = 128 if "MTL::Size::Make(128, 1, 1)" in source else None
    enforces_bn64_codewords8 = "bn != 64 || codewords != 8" in source
    slot_layout_rank = (
        5
        if "sign_byte_slots.ndim() != 5" in source and "scale_tiles.ndim() != 5" in source
        else None
    )
    passes_contract = (
        v2_primitive_present
        and uses_v2_pso
        and dispatches_reuse_scope
        and carries_output_tiles_inside_scope
        and fixed_threadgroup_threads == 128
        and enforces_bn64_codewords8
        and slot_layout_rank == 5
    )
    if passes_contract:
        decision = "expert_kblock_v2_dispatch_contract_present"
        required_next_features: list[str] = []
        structural_risk = "none"
    else:
        decision = "missing_expert_kblock_v2_dispatch_contract"
        required_next_features = [
            "add_expert_kblock_v2_primitive_dispatch",
            "dispatch_experts_x_k_blocks_x_route_tiles",
            "carry_output_tiles_inside_reuse_scope",
            "preserve_compressed_rhs_storage",
            "preserve_codeword_scale_slots",
        ]
        structural_risk = (
            "current primitive dispatch grid is incompatible with expert/K-block "
            "decode reuse scope"
            if dispatches_current_grid
            else "expert/K-block v2 primitive dispatch contract is absent"
        )
    if dispatches_reuse_scope:
        dispatch_grid = "experts_x_k_blocks_x_route_tiles"
    elif dispatches_current_grid:
        dispatch_grid = "n_tiles_x_num_route_tiles"
    else:
        dispatch_grid = "unknown"
    return {
        "decision": decision,
        "passes_contract": passes_contract,
        "v2_primitive_present": v2_primitive_present,
        "uses_v2_pso": uses_v2_pso,
        "dispatch_grid": dispatch_grid,
        "incompatible_primitive_grid_present": dispatches_current_grid,
        "carries_output_tiles_inside_reuse_scope": carries_output_tiles_inside_scope,
        "fixed_threadgroup_threads": fixed_threadgroup_threads,
        "enforces_bn64_codewords8": enforces_bn64_codewords8,
        "slot_layout_rank": slot_layout_rank,
        "required_next_features": required_next_features,
        "structural_risk": structural_risk,
    }


def audit_e8p_expert_kblock_partial_reduction_source(
    *,
    metal_source: str,
    primitive_source: str,
) -> dict[str, Any]:
    normalized_primitive = re.sub(r"\s+", " ", primitive_source)
    partial_kernel = "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul"
    reduce_kernel = "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_reduce"
    try:
        partial_kernel_source = _extract_kernel_source(metal_source, partial_kernel)
    except ValueError:
        partial_kernel_source = ""
    try:
        reduce_kernel_source = _extract_kernel_source(metal_source, reduce_kernel)
    except ValueError:
        reduce_kernel_source = ""
    partial_write_kernel_present = (
        f"void {partial_kernel}" in metal_source
        and "device float* kblock_partials" in metal_source
        and "kblock_partials[partial_offset]" in metal_source
    )
    partial_accumulation_scratch_present = (
        "array kblock_partials" in primitive_source
        and "kblock_partials.set_data(allocator::malloc(kblock_partials.nbytes()))"
        in primitive_source
        and (
            "{num_route_tiles, k_blocks, n_tiles, route_tile_size, 64}"
            in normalized_primitive
            or "{num_route_tiles, k_blocks, n_tiles, route_tile_size, bn}"
            in normalized_primitive
            or (
                "{num_route_tiles_shape, k_blocks_shape, n_tiles_shape,"
                " route_tile_size_shape, bn_shape}"
            )
            in normalized_primitive
        )
    )
    final_reduction_kernel_present = (
        f"void {reduce_kernel}" in metal_source
        and f'cache.get("{reduce_kernel}")' in primitive_source
        and "for (uint k_block = 0; k_block < k_blocks; ++k_block)"
        in reduce_kernel_source
        and "out[route * output_dims + n]" in reduce_kernel_source
    )
    partial_body_tensorops_present = (
        "matmul_op.run" in partial_kernel_source
        and (
            "mpp::tensor_ops::" in partial_kernel_source
            or "get_right_input_cooperative_tensor" in partial_kernel_source
        )
    )
    partial_body_parallel_schedule_present = partial_body_tensorops_present
    partial_grid_ok = "MTL::Size::Make(experts, k_blocks, num_route_tiles)" in normalized_primitive
    reduction_grid_ok = "MTL::Size::Make(num_route_tiles, n_tiles, 1)" in normalized_primitive
    scratch_dtype = (
        "float32"
        if "float32" in primitive_source and "device float* kblock_partials" in metal_source
        else "unknown"
    )
    passes_contract = (
        partial_write_kernel_present
        and partial_accumulation_scratch_present
        and final_reduction_kernel_present
        and partial_grid_ok
        and reduction_grid_ok
        and scratch_dtype == "float32"
    )
    if passes_contract:
        decision = "expert_kblock_v2_partial_reduction_scaffold_present"
        if partial_body_parallel_schedule_present:
            required_next_features = [
                "prove_native_v2_partial_reduction_parity",
                "resolve_v2_schedule_guardrail_before_speed",
                "run_same_window_q2_speed_packet_after_guardrail_and_parity",
            ]
            rejected_next_steps = [
                "do_not_benchmark_before_native_v2_parity",
                "do_not_promote_without_tensorops_speed_packet",
            ]
            structural_risk = (
                "partial accumulation, TensorOps writer body, and final reduction "
                "are source-visible, but schedule guardrails and native parity must "
                "pass before this becomes speed evidence"
            )
        else:
            required_next_features = [
                "prove_native_v2_partial_reduction_parity",
                "replace_scalar_partial_body_with_valid_tensorops_schedule_before_speed",
            ]
            rejected_next_steps = [
                "do_not_benchmark_partial_reduction_scaffold_as_speed_path",
                "do_not_promote_without_tensorops_speed_packet",
            ]
            structural_risk = (
                "partial accumulation and final reduction are source-visible, but this "
                "only proves the reduction scaffold; TensorOps speed validity remains open"
            )
    else:
        decision = "missing_expert_kblock_v2_partial_reduction_scaffold"
        required_next_features = [
            "allocate_partial_accumulation_scratch",
            "write_one_partial_per_kblock_workgroup",
            "run_final_kblock_reduction_kernel",
        ]
        rejected_next_steps = [
            "do_not_benchmark_single_kblock_completion_v2",
            "do_not_write_complete_output_from_one_kblock",
        ]
        structural_risk = (
            "v2 source does not yet prove per-K-block partial accumulation plus "
            "final reduction"
        )
    return {
        "decision": decision,
        "passes_contract": passes_contract,
        "partial_accumulation_scratch_present": partial_accumulation_scratch_present,
        "partial_write_kernel_present": partial_write_kernel_present,
        "final_reduction_kernel_present": final_reduction_kernel_present,
        "partial_body_parallel_schedule_present": partial_body_parallel_schedule_present,
        "partial_body_tensorops_present": partial_body_tensorops_present,
        "scratch_dtype": scratch_dtype,
        "speed_claim": False,
        "partial_grid": "experts_x_k_blocks_x_route_tiles" if partial_grid_ok else "unknown",
        "reduction_grid": "route_tiles_x_n_tiles" if reduction_grid_ok else "unknown",
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
        "structural_risk": structural_risk,
    }


def audit_e8p_component_stream_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _COMPONENT_STREAM_SCALAR_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        component_stream_kernel_present = True
    except ValueError:
        kernel = ""
        component_stream_kernel_present = False

    primitive_binding_present = (
        "e8p_component_stream_rhs_sorted_scalar_matmul" in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    route_tile_dispatch_ok = "MTL::Size::Make(num_route_tiles, experts, k_blocks)" in primitive_source
    route_output_scalar_dispatch_ok = (
        "MTL::Size::Make((output_dims + 15u) / 16u, (route_count + 15u) / 16u, 1)"
        in primitive_source
    )
    dispatch_grid_ok = route_tile_dispatch_ok or route_output_scalar_dispatch_ok
    uses_sign_component_bits = "sign_component_bits" in kernel
    uses_abs_index_tiles = "abs_index_tiles" in kernel
    uses_component_scale_slots = "component_scale_slots" in kernel
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    accumulates_sorted_routes = "sorted_x" in kernel and "out[route * output_dims + n]" in kernel
    materializes_decoded_dense_rhs = (
        "decoded_dense_rhs" in kernel
        or "decoded_rhs" in kernel
        or "dense_rhs" in kernel
        or "array decoded" in primitive_source
    )
    fills_tensorops_b_fragment = (
        "b_t[" in kernel
        or "b_t.thread_elements()" in kernel
        or "get_right_input_cooperative_tensor" in kernel
        or "matmul_op.run" in kernel
    )
    passes_contract = (
        component_stream_kernel_present
        and primitive_binding_present
        and dispatch_grid_ok
        and uses_sign_component_bits
        and uses_abs_index_tiles
        and uses_component_scale_slots
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and accumulates_sorted_routes
        and not materializes_decoded_dense_rhs
        and not fills_tensorops_b_fragment
    )
    if passes_contract:
        decision = "component_stream_native_scalar_oracle_present"
        required_next_features = [
            "prove_native_component_stream_parity",
            "prove_air_down_group_size_352_parity",
            "add_artifact_projection_parity_gate",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_scalar_oracle",
            "do_not_route_resident_auto_before_artifact_speed_gate",
        ]
        structural_risk = (
            "component-stream scalar source is present, but it is only a parity "
            "oracle before native/artifact projection gates"
        )
    else:
        decision = "missing_component_stream_native_source_oracle"
        required_next_features = [
            "add_component_stream_native_scalar_accumulator",
            "bind_component_stream_sorted_route_primitive",
            "preserve_component_scale_slots",
            "prove_air_down_group_size_352_parity",
        ]
        rejected_next_steps = [
            "do_not_benchmark_component_stream_without_native_parity",
            "do_not_materialize_decoded_dense_rhs",
        ]
        structural_risk = (
            "component-stream layout has a Python oracle, but native source does "
            "not yet prove a sorted-route scalar accumulator over compressed components"
        )
    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "component_stream_kernel_present": component_stream_kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "route_tiles_x_experts_x_k_blocks"
            if route_tile_dispatch_ok
            else "route_output_scalar"
            if route_output_scalar_dispatch_ok
            else "unknown"
        ),
        "uses_sign_component_bits": uses_sign_component_bits,
        "uses_abs_index_tiles": uses_abs_index_tiles,
        "uses_component_scale_slots": uses_component_scale_slots,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "accumulates_sorted_routes": accumulates_sorted_routes,
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "fills_tensorops_b_fragment": fills_tensorops_b_fragment,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
        "structural_risk": structural_risk,
    }


def audit_e8p_component_stream_speed_path_source(
    *,
    metal_source: str,
    primitive_source: str,
    benchmark_context: dict[str, Any] | None = None,
    scalar_kernel_name: str = _COMPONENT_STREAM_SCALAR_KERNEL,
    speed_kernel_name: str = _COMPONENT_STREAM_TENSOROPS_KERNEL,
) -> dict[str, Any]:
    benchmark_context = benchmark_context or {}
    scalar_guard = audit_e8p_component_stream_source(
        metal_source=metal_source,
        primitive_source=primitive_source,
        kernel_name=scalar_kernel_name,
    )
    try:
        speed_kernel = _extract_kernel_source(metal_source, speed_kernel_name)
        tensorops_speed_kernel_present = True
    except ValueError:
        speed_kernel = ""
        tensorops_speed_kernel_present = False

    primitive_binding_present = (
        "e8p_component_stream_rhs_sorted_tensorops_matmul" in primitive_source
        and f'cache.get("{speed_kernel_name}")' in primitive_source
    )
    has_tensorops_matmul = (
        "matmul_op.run" in speed_kernel
        or "mpp::tensor_ops" in speed_kernel
        or "get_right_input_cooperative_tensor" in speed_kernel
    )
    uses_component_stream_storage = all(
        token in speed_kernel
        for token in (
            "sign_component_bits",
            "abs_index_tiles",
            "component_scale_slots",
            "scale_tiles",
            "sorted_x",
        )
    )
    materializes_decoded_dense_rhs = (
        "decoded_dense_rhs" in speed_kernel
        or "decoded_rhs" in speed_kernel
        or "dense_rhs" in speed_kernel
        or "array decoded" in primitive_source
    )
    stages_decoded_b = (
        "threadgroup half" in speed_kernel
        and (
            "get_right_input_cooperative_tensor" in speed_kernel
            or "Btile.template load" in speed_kernel
        )
    )
    passes_contract = (
        tensorops_speed_kernel_present
        and primitive_binding_present
        and has_tensorops_matmul
        and uses_component_stream_storage
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
    )
    component_stream_comparison_count = int(
        benchmark_context.get("component_stream_tensorops_comparison_count") or 0
    )
    component_stream_best_ratio = benchmark_context.get(
        "component_stream_tensorops_best_ratio_to_q2"
    )
    component_stream_worst_ratio = benchmark_context.get(
        "component_stream_tensorops_worst_ratio_to_q2"
    )
    component_stream_best_ms = benchmark_context.get(
        "component_stream_tensorops_best_ms_per_iter"
    )
    component_stream_worst_ms = benchmark_context.get(
        "component_stream_tensorops_worst_ms_per_iter"
    )
    parity_ratio = float(benchmark_context.get("parity_ratio") or 1.25)
    component_stream_speed_rejected = (
        passes_contract
        and component_stream_comparison_count > 0
        and isinstance(component_stream_best_ratio, (int, float))
        and float(component_stream_best_ratio) > parity_ratio
    )

    if component_stream_speed_rejected:
        decision = "reject_component_stream_tensorops_speed_path"
        required_next_features = [
            "broaden_component_stream_decode_reuse_scope",
            "avoid_per_fragment_component_reconstruction",
            "preserve_compressed_component_stream_storage",
            "change_rhs_layout_or_kernel_family",
        ]
        rejected_next_steps = [
            "do_not_retime_component_stream_tensorops_unchanged",
            "do_not_route_resident_auto_from_rejected_component_stream_tensorops",
            "do_not_claim_speed_without_new_kernel_family",
        ]
        structural_risk = (
            "component-stream TensorOps reaches compressed RHS parity, but clean "
            "same-window q2 rows reject the current per-fragment reconstruction schedule"
        )
    elif passes_contract:
        decision = "component_stream_tensorops_speed_path_candidate_present_needs_parity"
        required_next_features = [
            "prove_native_component_stream_tensorops_parity",
            "prove_artifact_component_stream_tensorops_parity",
            "run_same_window_q2_speed_packet",
            "verify_clean_rows_no_pageouts_or_swapouts",
        ]
        rejected_next_steps = [
            "do_not_benchmark_before_native_and_artifact_parity",
            "do_not_route_resident_auto_before_same_window_q2_speed_gate",
            "do_not_claim_speed_without_q2_control",
        ]
        structural_risk = (
            "component-stream TensorOps candidate is present, but speed is "
            "unproven until native parity, artifact parity, and same-window q2 rows pass"
        )
    elif scalar_guard.get("passes_contract"):
        decision = "component_stream_scalar_oracle_speed_path_missing"
        required_next_features = [
            "design_component_stream_tensorops_or_parallel_speed_path",
            "preserve_compressed_component_stream_storage",
            "prove_speed_path_against_decoded_sorted_reference",
            "run_same_window_q2_speed_packet",
        ]
        rejected_next_steps = [
            "do_not_benchmark_component_stream_scalar_oracle_as_speed_path",
            "do_not_route_resident_auto_before_same_window_q2_speed_gate",
        ]
        structural_risk = (
            "component-stream scalar oracle and artifact parity exist, but no "
            "TensorOps or parallel speed path is present"
        )
    else:
        decision = "missing_component_stream_speed_path"
        required_next_features = [
            "add_component_stream_native_scalar_accumulator",
            "prove_artifact_down_parity",
            "design_component_stream_tensorops_or_parallel_speed_path",
        ]
        rejected_next_steps = [
            "do_not_benchmark_component_stream_without_native_parity",
            "do_not_route_resident_auto_before_artifact_speed_gate",
        ]
        structural_risk = (
            "component-stream speed work cannot start before the scalar source "
            "oracle and artifact parity gates exist"
        )

    return {
        "scalar_kernel_name": scalar_kernel_name,
        "speed_kernel_name": speed_kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "scalar_oracle_present": bool(scalar_guard.get("passes_contract")),
        "tensorops_speed_kernel_present": tensorops_speed_kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "has_tensorops_matmul": has_tensorops_matmul,
        "uses_component_stream_storage": uses_component_stream_storage,
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "speed_claim": False,
        "component_stream_tensorops_comparison_count": component_stream_comparison_count,
        "component_stream_tensorops_best_ratio_to_q2": component_stream_best_ratio,
        "component_stream_tensorops_worst_ratio_to_q2": component_stream_worst_ratio,
        "component_stream_tensorops_best_ms_per_iter": component_stream_best_ms,
        "component_stream_tensorops_worst_ms_per_iter": component_stream_worst_ms,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
        "structural_risk": structural_risk,
    }


def audit_e8p_route_slot_codeword_stream_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _ROUTE_SLOT_CODEWORD_STREAM_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_binding_present = (
        "e8p_route_slot_codeword_stream_rhs_sorted_matmul" in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    if primitive_binding_present:
        try:
            scoped_primitive_source = _extract_cpp_function_source(
                primitive_source,
                "e8p_route_slot_codeword_stream_rhs_sorted_matmul",
            )
        except ValueError:
            scoped_primitive_source = primitive_source
    else:
        scoped_primitive_source = ""
    scoped_source = f"{kernel}\n{scoped_primitive_source}"
    route_slot_dispatch_ok = (
        "route_tiles_x_route_slots_x_k_blocks_x_codewords" in primitive_source
        or "MTL::Size::Make(num_route_tiles, route_tile_size, k_blocks)" in primitive_source
    )
    preserves_route_slot_axis = "route_slot" in kernel and "routes_in_tile" in kernel
    streams_compressed_codewords = "code_tiles" in kernel and (
        "uint16_t" in kernel or "ushort" in kernel or "uint16" in kernel
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    uses_codebook = "codebook" in kernel and "decode_e8p" in kernel
    accumulates_sorted_routes = "sorted_x" in kernel and "out[route * output_dims + n]" in kernel
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in (
            "decoded_dense_rhs",
            "decoded_rhs",
            "dense_rhs",
            "array decoded",
        )
    )
    uses_cross_ntile_decode_cache = any(
        token in scoped_source
        for token in (
            "shared_decode",
            "decode_cache",
            "component_decode_cache",
        )
    )
    fills_tensorops_b_fragment = (
        "b_t[" in kernel
        or "b_t.thread_elements()" in kernel
        or "get_right_input_cooperative_tensor" in kernel
        or "matmul_op.run" in kernel
    )
    uses_lane_local_fragment_buffer = (
        "threadgroup half shared_b" in kernel or "lane_local" in kernel
    )
    passes_contract = (
        kernel_present
        and primitive_binding_present
        and route_slot_dispatch_ok
        and preserves_route_slot_axis
        and streams_compressed_codewords
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and uses_codebook
        and accumulates_sorted_routes
        and not materializes_decoded_dense_rhs
        and not uses_cross_ntile_decode_cache
        and not fills_tensorops_b_fragment
        and not uses_lane_local_fragment_buffer
    )

    if passes_contract:
        decision = "route_slot_codeword_stream_source_guardrail_present"
        required_next_features = [
            "prove_native_route_slot_codeword_stream_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_artifact_speed_gate",
        ]
        structural_risk = (
            "route-slot codeword stream source is visible and preserves the "
            "compressed route-slot contract, but parity and q2 timing are still required"
        )
    else:
        decision = "missing_route_slot_codeword_stream_source"
        required_next_features = [
            "add_route_slot_codeword_stream_native_scalar_source",
            "bind_route_slot_codeword_stream_primitive",
            "preserve_route_slot_axis",
            "stream_compressed_uint16_codewords_and_scale_slots",
            "prove_air_down_group_size_352_artifact_parity",
        ]
        rejected_next_steps = [
            "do_not_benchmark_route_slot_codeword_stream_before_source_guardrail",
            "do_not_reopen_expert_kblock_v2_or_route_abs_component_cache_families",
            "do_not_materialize_decoded_dense_rhs",
        ]
        structural_risk = (
            "route-slot codeword stream is the selected next-family design, "
            "but native source does not yet prove the route-slot/compressed-codeword contract"
        )

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "route_tiles_x_route_slots_x_k_blocks_x_codewords"
            if route_slot_dispatch_ok
            else "unknown"
        ),
        "preserves_route_slot_axis": preserves_route_slot_axis,
        "streams_compressed_codewords": streams_compressed_codewords,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "uses_codebook": uses_codebook,
        "accumulates_sorted_routes": accumulates_sorted_routes,
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "uses_cross_ntile_decode_cache": uses_cross_ntile_decode_cache,
        "fills_tensorops_b_fragment": fills_tensorops_b_fragment,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
        "structural_risk": structural_risk,
    }


def audit_e8p_route_slot_mma_codeword_tile_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _ROUTE_SLOT_MMA_CODEWORD_TILE_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_binding_present = (
        "e8p_route_slot_mma_codeword_tile_rhs_sorted_matmul" in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(
                    primitive_source,
                    "e8p_route_slot_mma_codeword_tile_rhs_sorted_matmul",
                )
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    route_slot_axis = "route_slot" in kernel and "routes_in_tile" in kernel
    output_tile_dispatch = (
        "route_tiles_x_output_tiles_x_k_blocks_x_codeword_tiles" in primitive_source
        or "num_output_tiles" in primitive_source
    )
    streams_compressed_codewords = "code_tiles" in kernel and (
        "uint16_t" in kernel or "ushort" in kernel or "uint16" in kernel
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel
    passes_contract = (
        kernel_present
        and primitive_binding_present
        and route_slot_axis
        and output_tile_dispatch
        and streams_compressed_codewords
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
    )

    if passes_contract:
        decision = "route_slot_mma_codeword_tile_source_guardrail_present"
        required_next_features = [
            "prove_native_route_slot_mma_codeword_tile_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_route_slot_mma_codeword_tile_source"
        required_next_features = [
            "add_route_slot_mma_codeword_tile_native_source_guardrail",
            "bind_route_slot_mma_codeword_tile_primitive",
            "preserve_route_slot_axis",
            "batch_output_tiles_in_dispatch",
            "stream_compressed_uint16_codewords_and_scale_slots",
        ]
        rejected_next_steps = [
            "do_not_benchmark_route_slot_mma_codeword_tile_before_source_guardrail",
            "do_not_retime_scalar_route_slot_codeword_stream",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "route_tiles_x_output_tiles_x_k_blocks_x_codeword_tiles"
            if output_tile_dispatch
            else "unknown"
        ),
        "preserves_route_slot_axis": route_slot_axis,
        "batches_output_tiles": output_tile_dispatch,
        "streams_compressed_codewords": streams_compressed_codewords,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_active_route_tile_codeword_outer_product_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _ACTIVE_ROUTE_TILE_CODEWORD_OUTER_PRODUCT_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_binding_present = (
        "e8p_active_route_tile_codeword_outer_product_rhs_sorted_matmul"
        in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(
                    primitive_source,
                    "e8p_active_route_tile_codeword_outer_product_rhs_sorted_matmul",
                )
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    active_route_tile_indirection = (
        "active_route_tile" in kernel
        and "active_route_tiles" in kernel
        and ("route_tile" in kernel or "routes_in_tile" in kernel)
    )
    dispatch_grid_ok = (
        "active_route_tiles_x_k_blocks_x_codewords_x_output_microtiles"
        in primitive_source
        or "active_route_tile_count" in primitive_source
    )
    streams_compressed_codewords = "code_tiles" in kernel and (
        "uint16_t" in kernel
        or "ushort" in kernel
        or "uint16" in kernel
        or "uint code" in kernel
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel
    uses_route_slot_mma_schedule = (
        "route_tiles_x_output_tiles_x_k_blocks_x_codeword_tiles" in scoped_source
        or "route_slot_mma_codeword_tile" in scoped_source
        or "num_output_tiles" in scoped_source
    )
    passes_contract = (
        kernel_present
        and primitive_binding_present
        and active_route_tile_indirection
        and dispatch_grid_ok
        and streams_compressed_codewords
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
        and not uses_route_slot_mma_schedule
    )

    if passes_contract:
        decision = "active_route_tile_codeword_outer_product_source_guardrail_present"
        required_next_features = [
            "prove_native_active_route_tile_codeword_outer_product_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_active_route_tile_codeword_outer_product_source"
        required_next_features = [
            "add_active_route_tile_codeword_outer_product_native_source_guardrail",
            "bind_active_route_tile_codeword_outer_product_primitive",
            "preserve_active_route_tile_indirection",
            "stream_compressed_uint16_codewords_and_scale_slots",
            "avoid_route_slot_mma_codeword_tile_schedule",
        ]
        rejected_next_steps = [
            "do_not_benchmark_active_route_tile_codeword_outer_product_before_source_guardrail",
            "do_not_retime_route_slot_mma_codeword_tile",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "active_route_tiles_x_k_blocks_x_codewords_x_output_microtiles"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_active_route_tile_indirection": active_route_tile_indirection,
        "streams_compressed_codewords": streams_compressed_codewords,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "uses_route_slot_mma_schedule": uses_route_slot_mma_schedule,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_expert_cohort_codeword_broadcast_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _EXPERT_COHORT_CODEWORD_BROADCAST_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_binding_present = (
        "e8p_expert_cohort_codeword_broadcast_rhs_sorted_matmul" in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(
                    primitive_source,
                    "e8p_expert_cohort_codeword_broadcast_rhs_sorted_matmul",
                )
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    preserves_expert_cohort_route_descriptors = all(
        token in kernel
        for token in (
            "expert_cohort_offsets",
            "expert_cohort_counts",
            "route_cohort_offsets",
        )
    )
    dispatch_grid_ok = (
        "experts_x_route_cohorts_x_output_microtiles_x_k_blocks" in primitive_source
        or "expert_cohort_count" in primitive_source
    )
    streams_compressed_codewords = "code_tiles" in kernel and (
        "uint16_t" in kernel
        or "ushort" in kernel
        or "uint16" in kernel
        or "compressed_codeword" in kernel
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel
    uses_active_route_tile_schedule = (
        "active_route_tiles_x_k_blocks_x_codewords_x_output_microtiles"
        in scoped_source
        or "active_route_tile_codeword_outer_product" in scoped_source
        or "active_route_tile_count" in scoped_source
    )
    passes_contract = (
        kernel_present
        and primitive_binding_present
        and preserves_expert_cohort_route_descriptors
        and dispatch_grid_ok
        and streams_compressed_codewords
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
        and not uses_active_route_tile_schedule
    )

    if passes_contract:
        decision = "expert_cohort_codeword_broadcast_source_guardrail_present"
        required_next_features = [
            "prove_expert_cohort_codeword_broadcast_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_expert_cohort_codeword_broadcast_source"
        required_next_features = [
            "add_expert_cohort_codeword_broadcast_native_source_guardrail",
            "bind_expert_cohort_codeword_broadcast_primitive",
            "preserve_expert_cohort_route_descriptors",
            "preserve_compressed_uint16_codeword_broadcast",
            "avoid_decoded_dense_rhs",
        ]
        rejected_next_steps = [
            "do_not_benchmark_expert_cohort_codeword_broadcast_before_source_guardrail",
            "do_not_retime_active_route_tile_codeword_outer_product",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "experts_x_route_cohorts_x_output_microtiles_x_k_blocks"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_expert_cohort_route_descriptors": (
            preserves_expert_cohort_route_descriptors
        ),
        "streams_compressed_codewords": streams_compressed_codewords,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "uses_active_route_tile_schedule": uses_active_route_tile_schedule,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_route_batch_segmented_codeword_reduce_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _ROUTE_BATCH_SEGMENTED_CODEWORD_REDUCE_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_binding_present = (
        "e8p_route_batch_segmented_codeword_reduce_rhs_sorted_matmul"
        in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(
                    primitive_source,
                    "e8p_route_batch_segmented_codeword_reduce_rhs_sorted_matmul",
                )
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    preserves_route_batch_segment_descriptors = all(
        token in kernel
        for token in (
            "route_batch_segment_offsets",
            "route_batch_segment_counts",
            "route_batch_route_ids",
        )
    )
    dispatch_grid_ok = (
        "route_batches_x_k_blocks_x_output_microtiles_x_codewords" in primitive_source
        or "route_batch_count" in primitive_source
    )
    streams_compressed_codewords = "code_tiles" in kernel and (
        "uint16_t" in kernel
        or "ushort" in kernel
        or "uint16" in kernel
        or "compressed_codeword" in kernel
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel
    uses_expert_cohort_broadcast_schedule = (
        "experts_x_route_cohorts_x_output_microtiles_x_k_blocks" in scoped_source
        or "expert_cohort_codeword_broadcast" in scoped_source
        or "expert_cohort_count" in scoped_source
    )
    passes_contract = (
        kernel_present
        and primitive_binding_present
        and preserves_route_batch_segment_descriptors
        and dispatch_grid_ok
        and streams_compressed_codewords
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
        and not uses_expert_cohort_broadcast_schedule
    )

    if passes_contract:
        decision = "route_batch_segmented_codeword_reduce_source_guardrail_present"
        required_next_features = [
            "prove_route_batch_segmented_codeword_reduce_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_route_batch_segmented_codeword_reduce_source"
        required_next_features = [
            "add_route_batch_segmented_codeword_reduce_native_source_guardrail",
            "bind_route_batch_segmented_codeword_reduce_primitive",
            "preserve_route_batch_segment_descriptors",
            "stream_compressed_uint16_codewords_and_scale_slots",
            "avoid_expert_major_codeword_broadcast_schedule",
        ]
        rejected_next_steps = [
            "do_not_benchmark_route_batch_segmented_codeword_reduce_before_source_guardrail",
            "do_not_retime_expert_cohort_codeword_broadcast",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "route_batches_x_k_blocks_x_output_microtiles_x_codewords"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_route_batch_segment_descriptors": (
            preserves_route_batch_segment_descriptors
        ),
        "streams_compressed_codewords": streams_compressed_codewords,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "uses_expert_cohort_broadcast_schedule": uses_expert_cohort_broadcast_schedule,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_token_cohort_codeword_stream_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _TOKEN_COHORT_CODEWORD_STREAM_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_binding_present = (
        "e8p_token_cohort_codeword_stream_rhs_sorted_matmul" in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(
                    primitive_source,
                    "e8p_token_cohort_codeword_stream_rhs_sorted_matmul",
                )
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    preserves_token_cohort_route_descriptors = all(
        token in kernel
        for token in (
            "token_cohort_offsets",
            "token_cohort_counts",
            "token_cohort_active_expert_ids",
        )
    )
    preserves_exact_route_slot_indirection = "token_cohort_route_slot_ids" in kernel
    dispatch_grid_ok = (
        "token_cohorts_x_active_experts_x_k_blocks_x_codewords" in primitive_source
        or "token_cohort_count" in primitive_source
    )
    streams_compressed_codewords = "code_tiles" in kernel and (
        "uint16_t" in kernel
        or "ushort" in kernel
        or "uint16" in kernel
        or "compressed_codeword" in kernel
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel
    uses_component_stream_partial_schedule = (
        "component_stream_partial" in scoped_source
        or "component_stream_partials" in scoped_source
        or "route_tiles_x_component_pairs_x_k_blocks" in scoped_source
    )
    passes_contract = (
        kernel_present
        and primitive_binding_present
        and preserves_token_cohort_route_descriptors
        and preserves_exact_route_slot_indirection
        and dispatch_grid_ok
        and streams_compressed_codewords
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
        and not uses_component_stream_partial_schedule
    )

    if passes_contract:
        decision = "token_cohort_codeword_stream_source_guardrail_present"
        required_next_features = [
            "prove_token_cohort_codeword_stream_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_token_cohort_codeword_stream_source"
        required_next_features = [
            "add_token_cohort_codeword_stream_native_source_guardrail",
            "bind_token_cohort_codeword_stream_primitive",
            "preserve_token_cohort_route_descriptors",
            "preserve_exact_route_slot_indirection",
            "stream_compressed_uint16_codewords_and_scale_slots",
            "avoid_decoded_dense_rhs",
        ]
        rejected_next_steps = [
            "do_not_benchmark_token_cohort_codeword_stream_before_source_guardrail",
            "do_not_retime_component_stream_tensorops",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "token_cohorts_x_active_experts_x_k_blocks_x_codewords"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_token_cohort_route_descriptors": (
            preserves_token_cohort_route_descriptors
        ),
        "preserves_exact_route_slot_indirection": (
            preserves_exact_route_slot_indirection
        ),
        "streams_compressed_codewords": streams_compressed_codewords,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "uses_component_stream_partial_schedule": uses_component_stream_partial_schedule,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_token_cohort_mma_codeword_tile_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _TOKEN_COHORT_MMA_CODEWORD_TILE_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_name = "e8p_token_cohort_mma_codeword_tile_rhs_sorted_matmul"
    primitive_binding_present = (
        primitive_name in primitive_source and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(primitive_source, primitive_name)
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    preserves_token_cohort_route_descriptors = all(
        token in kernel
        for token in (
            "token_cohort_offsets",
            "token_cohort_counts",
            "token_cohort_active_expert_ids",
        )
    )
    preserves_exact_route_slot_indirection = "token_cohort_route_slot_ids" in kernel
    dispatch_grid_ok = (
        "token_cohorts_x_output_tiles_x_active_experts_x_k_blocks_x_codeword_tiles"
        in primitive_source
        or (
            "token_cohort_count" in primitive_source
            and "output_tile_count" in primitive_source
            and "active_expert_count" in primitive_source
            and "k_blocks" in primitive_source
            and "codeword_tile_count" in primitive_source
        )
    )
    uses_output_tile_axis = "output_tile" in scoped_source
    uses_codeword_tile_axis = "codeword_tile" in scoped_source
    streams_compressed_codeword_tiles = (
        ("codeword_tiles" in kernel or ("code_tiles" in kernel and "codeword_tile" in kernel))
        and (
            "uint16_t" in kernel
            or "ushort" in kernel
            or "uint16" in kernel
            or "compressed_codeword" in kernel
        )
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    uses_mma_accumulation = (
        "mpp::tensor_ops" in kernel
        or "matmul2d" in kernel
        or "simdgroup_matrix" in kernel
        or "mma_accum" in kernel
    )
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel
    uses_rejected_codeword_stream_schedule = (
        "token_cohort_codeword_stream" in scoped_source
        or "token_cohorts_x_active_experts_x_k_blocks_x_codewords" in scoped_source
    )
    passes_contract = (
        kernel_present
        and primitive_binding_present
        and preserves_token_cohort_route_descriptors
        and preserves_exact_route_slot_indirection
        and dispatch_grid_ok
        and uses_output_tile_axis
        and uses_codeword_tile_axis
        and streams_compressed_codeword_tiles
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and uses_mma_accumulation
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
        and not uses_rejected_codeword_stream_schedule
    )

    if passes_contract:
        decision = "token_cohort_mma_codeword_tile_source_guardrail_present"
        required_next_features = [
            "prove_token_cohort_mma_codeword_tile_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_token_cohort_mma_codeword_tile_source"
        required_next_features = [
            "add_token_cohort_mma_codeword_tile_native_source_guardrail",
            "bind_token_cohort_mma_codeword_tile_primitive",
            "preserve_token_cohort_route_descriptors",
            "preserve_exact_route_slot_indirection",
            "use_output_tile_and_codeword_tile_axes",
            "stream_compressed_uint16_codeword_tiles_and_scale_slots",
            "use_mma_codeword_tile_accumulation",
            "avoid_decoded_dense_rhs",
        ]
        rejected_next_steps = [
            "do_not_benchmark_token_cohort_mma_codeword_tile_before_source_guardrail",
            "do_not_retime_token_cohort_codeword_stream",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "token_cohorts_x_output_tiles_x_active_experts_x_k_blocks_x_codeword_tiles"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_token_cohort_route_descriptors": (
            preserves_token_cohort_route_descriptors
        ),
        "preserves_exact_route_slot_indirection": (
            preserves_exact_route_slot_indirection
        ),
        "uses_output_tile_axis": uses_output_tile_axis,
        "uses_codeword_tile_axis": uses_codeword_tile_axis,
        "streams_compressed_codeword_tiles": streams_compressed_codeword_tiles,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "uses_mma_accumulation": uses_mma_accumulation,
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "uses_rejected_codeword_stream_schedule": uses_rejected_codeword_stream_schedule,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_output_stationary_codeword_tile_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _OUTPUT_STATIONARY_CODEWORD_TILE_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_name = "e8p_output_stationary_codeword_tile_rhs_sorted_matmul"
    primitive_binding_present = (
        primitive_name in primitive_source and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(primitive_source, primitive_name)
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    preserves_output_stationary_route_batches = all(
        token in kernel
        for token in (
            "output_stationary_route_batch_offsets",
            "output_stationary_route_batch_counts",
            "output_stationary_route_batch_active_expert_ids",
        )
    )
    preserves_exact_route_slot_indirection = (
        "output_stationary_route_batch_route_slot_ids" in kernel
    )
    dispatch_grid_ok = (
        "output_tiles_x_route_batches_x_active_experts_x_k_blocks_x_codeword_tiles"
        in primitive_source
        or (
            "output_tile_count" in primitive_source
            and "route_batch_count" in primitive_source
            and "active_expert_count" in primitive_source
            and "k_blocks" in primitive_source
            and "codeword_tile_count" in primitive_source
        )
    )
    uses_output_tile_axis = "output_tile" in scoped_source
    uses_route_batch_axis = "route_batch" in scoped_source
    uses_codeword_tile_axis = "codeword_tile" in scoped_source
    keeps_output_stationary_accumulation = (
        "output_stationary" in scoped_source and "accum" in scoped_source
    )
    streams_compressed_codeword_tiles = (
        ("codeword_tiles" in kernel or ("code_tiles" in kernel and "codeword_tile" in kernel))
        and (
            "uint16_t" in kernel
            or "ushort" in kernel
            or "uint16" in kernel
            or "compressed_codeword" in kernel
        )
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel
    uses_rejected_token_cohort_mma_schedule = (
        "token_cohort_mma" in scoped_source
        or "token_cohorts_x_output_tiles_x_active_experts_x_k_blocks_x_codeword_tiles"
        in scoped_source
    )
    passes_contract = (
        kernel_present
        and primitive_binding_present
        and preserves_output_stationary_route_batches
        and preserves_exact_route_slot_indirection
        and dispatch_grid_ok
        and uses_output_tile_axis
        and uses_route_batch_axis
        and uses_codeword_tile_axis
        and keeps_output_stationary_accumulation
        and streams_compressed_codeword_tiles
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
        and not uses_rejected_token_cohort_mma_schedule
    )

    if passes_contract:
        decision = "output_stationary_codeword_tile_source_guardrail_present"
        required_next_features = [
            "prove_output_stationary_codeword_tile_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_output_stationary_codeword_tile_source"
        required_next_features = [
            "add_output_stationary_codeword_tile_native_source_guardrail",
            "bind_output_stationary_codeword_tile_primitive",
            "preserve_output_stationary_route_batches",
            "preserve_exact_route_slot_indirection",
            "use_output_tile_route_batch_and_codeword_tile_axes",
            "stream_compressed_uint16_codeword_tiles_and_scale_slots",
            "avoid_decoded_dense_rhs",
        ]
        rejected_next_steps = [
            "do_not_benchmark_output_stationary_codeword_tile_before_source_guardrail",
            "do_not_retime_token_cohort_mma_codeword_tile",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "output_tiles_x_route_batches_x_active_experts_x_k_blocks_x_codeword_tiles"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_output_stationary_route_batches": (
            preserves_output_stationary_route_batches
        ),
        "preserves_exact_route_slot_indirection": (
            preserves_exact_route_slot_indirection
        ),
        "uses_output_tile_axis": uses_output_tile_axis,
        "uses_route_batch_axis": uses_route_batch_axis,
        "uses_codeword_tile_axis": uses_codeword_tile_axis,
        "keeps_output_stationary_accumulation": keeps_output_stationary_accumulation,
        "streams_compressed_codeword_tiles": streams_compressed_codeword_tiles,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "uses_rejected_token_cohort_mma_schedule": (
            uses_rejected_token_cohort_mma_schedule
        ),
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_input_stationary_codeword_tile_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _INPUT_STATIONARY_CODEWORD_TILE_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_name = "e8p_input_stationary_codeword_tile_rhs_sorted_matmul"
    primitive_binding_present = (
        primitive_name in primitive_source and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(primitive_source, primitive_name)
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    preserves_input_stationary_route_batches = all(
        token in kernel
        for token in (
            "input_stationary_route_batch_offsets",
            "input_stationary_route_batch_counts",
            "input_stationary_route_batch_active_expert_ids",
        )
    )
    preserves_exact_route_slot_indirection = (
        "input_stationary_route_batch_route_slot_ids" in kernel
    )
    dispatch_grid_ok = (
        "input_tiles_x_route_batches_x_active_experts_x_output_tiles_x_codeword_tiles"
        in primitive_source
        or (
            "input_tile_count" in primitive_source
            and "route_batch_count" in primitive_source
            and "active_expert_count" in primitive_source
            and "output_tile_count" in primitive_source
            and "codeword_tile_count" in primitive_source
        )
    )
    uses_input_tile_axis = "input_tile" in scoped_source
    uses_route_batch_axis = "route_batch" in scoped_source
    uses_output_tile_axis = "output_tile" in scoped_source
    uses_codeword_tile_axis = "codeword_tile" in scoped_source
    keeps_input_stationary_accumulation = (
        "input_stationary" in scoped_source and "accum" in scoped_source
    )
    reuses_kblock_across_output_tiles = (
        "k_block" in scoped_source and "output_tile" in scoped_source
    )
    streams_compressed_codeword_tiles = (
        ("codeword_tiles" in kernel or ("code_tiles" in kernel and "codeword_tile" in kernel))
        and (
            "uint16_t" in kernel
            or "ushort" in kernel
            or "uint16" in kernel
            or "compressed_codeword" in kernel
        )
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel
    uses_rejected_output_stationary_schedule = (
        "output_stationary" in scoped_source
        or "output_tiles_x_route_batches_x_active_experts_x_k_blocks_x_codeword_tiles"
        in scoped_source
    )
    passes_contract = (
        kernel_present
        and primitive_binding_present
        and preserves_input_stationary_route_batches
        and preserves_exact_route_slot_indirection
        and dispatch_grid_ok
        and uses_input_tile_axis
        and uses_route_batch_axis
        and uses_output_tile_axis
        and uses_codeword_tile_axis
        and keeps_input_stationary_accumulation
        and reuses_kblock_across_output_tiles
        and streams_compressed_codeword_tiles
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
        and not uses_rejected_output_stationary_schedule
    )

    if passes_contract:
        decision = "input_stationary_codeword_tile_source_guardrail_present"
        required_next_features = [
            "prove_input_stationary_codeword_tile_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_input_stationary_codeword_tile_source"
        required_next_features = [
            "add_input_stationary_codeword_tile_native_source_guardrail",
            "bind_input_stationary_codeword_tile_primitive",
            "preserve_input_stationary_route_batches",
            "preserve_exact_route_slot_indirection",
            "use_input_tile_route_batch_output_tile_and_codeword_tile_axes",
            "reuse_kblock_codeword_tiles_across_output_tiles",
            "stream_compressed_uint16_codeword_tiles_and_scale_slots",
            "avoid_decoded_dense_rhs",
        ]
        rejected_next_steps = [
            "do_not_benchmark_input_stationary_codeword_tile_before_source_guardrail",
            "do_not_retime_output_stationary_codeword_tile",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "input_tiles_x_route_batches_x_active_experts_x_output_tiles_x_codeword_tiles"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_input_stationary_route_batches": (
            preserves_input_stationary_route_batches
        ),
        "preserves_exact_route_slot_indirection": (
            preserves_exact_route_slot_indirection
        ),
        "uses_input_tile_axis": uses_input_tile_axis,
        "uses_route_batch_axis": uses_route_batch_axis,
        "uses_output_tile_axis": uses_output_tile_axis,
        "uses_codeword_tile_axis": uses_codeword_tile_axis,
        "keeps_input_stationary_accumulation": keeps_input_stationary_accumulation,
        "reuses_kblock_across_output_tiles": reuses_kblock_across_output_tiles,
        "streams_compressed_codeword_tiles": streams_compressed_codeword_tiles,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "uses_rejected_output_stationary_schedule": (
            uses_rejected_output_stationary_schedule
        ),
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_expert_kblock_codeword_factor_reuse_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _EXPERT_KBLOCK_CODEWORD_FACTOR_REUSE_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_name = "e8p_expert_kblock_codeword_factor_reuse_rhs_sorted_matmul"
    primitive_binding_present = (
        primitive_name in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(primitive_source, primitive_name)
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    dispatch_grid_ok = (
        "experts_x_k_blocks_x_output_tiles_x_route_tiles_x_codeword_tiles"
        in primitive_source
        or (
            "expert_count" in primitive_source
            and "k_block_count" in primitive_source
            and "output_tile_count" in primitive_source
            and "route_tile_count" in primitive_source
            and "codeword_tile_count" in primitive_source
        )
    )
    uses_expert_axis = "expert" in scoped_source
    uses_kblock_axis = "k_block" in scoped_source or "kblock" in scoped_source
    uses_output_tile_axis = "output_tile" in scoped_source
    uses_route_tile_axis = "route_tile" in scoped_source
    uses_codeword_tile_axis = "codeword_tile" in scoped_source
    reuses_factor_decode_across_output_tiles = (
        "factor" in scoped_source and "output_tile" in scoped_source
    )
    streams_compressed_codeword_factor_tiles = (
        (
            "codeword_factor_tiles" in kernel
            or "codeword_tiles" in kernel
            or ("code_tiles" in kernel and "factor" in kernel)
        )
        and (
            "uint16_t" in kernel
            or "ushort" in kernel
            or "uint16" in kernel
            or "compressed_codeword" in kernel
        )
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    reconstructs_per_fragment = any(
        token in scoped_source
        for token in ("b_t[", "b_t.thread_elements()", "per_fragment", "fragment_decode")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel
    uses_rejected_input_stationary_schedule = (
        "input_stationary" in scoped_source
        or "input_tiles_x_route_batches_x_active_experts_x_output_tiles_x_codeword_tiles"
        in scoped_source
    )
    uses_historical_v2_schedule = "tensorops_v2" in scoped_source
    passes_contract = (
        kernel_present
        and primitive_binding_present
        and dispatch_grid_ok
        and uses_expert_axis
        and uses_kblock_axis
        and uses_output_tile_axis
        and uses_route_tile_axis
        and uses_codeword_tile_axis
        and reuses_factor_decode_across_output_tiles
        and streams_compressed_codeword_factor_tiles
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and not materializes_decoded_dense_rhs
        and not reconstructs_per_fragment
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
        and not uses_rejected_input_stationary_schedule
        and not uses_historical_v2_schedule
    )

    if passes_contract:
        decision = "expert_kblock_codeword_factor_reuse_source_guardrail_present"
        required_next_features = [
            "prove_expert_kblock_codeword_factor_reuse_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_expert_kblock_codeword_factor_reuse_source"
        required_next_features = [
            "add_expert_kblock_codeword_factor_reuse_native_source_guardrail",
            "bind_expert_kblock_codeword_factor_reuse_primitive",
            "use_expert_kblock_output_tile_route_tile_and_codeword_tile_axes",
            "reuse_factor_decode_across_output_tiles",
            "stream_compressed_uint16_codeword_factor_tiles_and_scale_slots",
            "avoid_decoded_dense_rhs",
            "avoid_per_fragment_factor_reconstruction",
        ]
        rejected_next_steps = [
            "do_not_benchmark_expert_kblock_codeword_factor_reuse_before_source_guardrail",
            "do_not_retime_input_stationary_codeword_tile",
            "do_not_reopen_expert_kblock_tensorops_v2",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "experts_x_k_blocks_x_output_tiles_x_route_tiles_x_codeword_tiles"
            if dispatch_grid_ok
            else "unknown"
        ),
        "uses_expert_axis": uses_expert_axis,
        "uses_kblock_axis": uses_kblock_axis,
        "uses_output_tile_axis": uses_output_tile_axis,
        "uses_route_tile_axis": uses_route_tile_axis,
        "uses_codeword_tile_axis": uses_codeword_tile_axis,
        "reuses_factor_decode_across_output_tiles": (
            reuses_factor_decode_across_output_tiles
        ),
        "streams_compressed_codeword_factor_tiles": (
            streams_compressed_codeword_factor_tiles
        ),
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "reconstructs_per_fragment": reconstructs_per_fragment,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "uses_rejected_input_stationary_schedule": (
            uses_rejected_input_stationary_schedule
        ),
        "uses_historical_v2_schedule": uses_historical_v2_schedule,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_route_codeword_lut_accumulate_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _ROUTE_CODEWORD_LUT_ACCUMULATE_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_name = "e8p_route_codeword_lut_accumulate_rhs_sorted_matmul"
    primitive_binding_present = (
        primitive_name in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(primitive_source, primitive_name)
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    preserves_route_codeword_lut_descriptors = all(
        token in kernel
        for token in (
            "route_codeword_lut_route_slots",
            "route_codeword_lut_offsets",
            "route_codeword_lut_counts",
            "route_codeword_lut_codeword_ids",
        )
    )
    dispatch_grid_ok = (
        "route_slots_x_k_blocks_x_codewords_then_output_tiles" in primitive_source
        or (
            "route_slot_count" in primitive_source
            and "k_block_count" in primitive_source
            and "codeword_count" in primitive_source
        )
    )
    streams_compressed_codewords = "code_tiles" in kernel and (
        "uint16_t" in kernel
        or "ushort" in kernel
        or "uint16" in kernel
        or "compressed_codeword" in kernel
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    accumulates_route_local_codeword_dot_lut = (
        "route_local_codeword_dot_lut" in kernel
        and "route_codeword_lut_accumulators" in kernel
        and "accum" in kernel
    )
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel
    uses_expert_kblock_reuse_schedule = (
        "expert_kblock_codeword_factor_reuse" in scoped_source
        or "experts_x_k_blocks_x_output_tiles_x_route_tiles_x_codeword_tiles"
        in scoped_source
    )
    passes_contract = (
        kernel_present
        and primitive_binding_present
        and preserves_route_codeword_lut_descriptors
        and dispatch_grid_ok
        and streams_compressed_codewords
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and accumulates_route_local_codeword_dot_lut
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
        and not uses_expert_kblock_reuse_schedule
    )

    if passes_contract:
        decision = "route_codeword_lut_accumulate_source_guardrail_present"
        required_next_features = [
            "prove_route_codeword_lut_accumulate_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_route_codeword_lut_accumulate_source"
        required_next_features = [
            "add_route_codeword_lut_accumulate_native_source_guardrail",
            "bind_route_codeword_lut_accumulate_primitive",
            "preserve_route_codeword_lut_descriptors",
            "stream_compressed_uint16_codewords_and_scale_slots",
            "accumulate_route_local_codeword_dot_lut",
            "avoid_decoded_dense_rhs",
        ]
        rejected_next_steps = [
            "do_not_benchmark_route_codeword_lut_accumulate_before_source_guardrail",
            "do_not_retime_expert_kblock_codeword_factor_reuse",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "route_slots_x_k_blocks_x_codewords_then_output_tiles"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_route_codeword_lut_descriptors": (
            preserves_route_codeword_lut_descriptors
        ),
        "streams_compressed_codewords": streams_compressed_codewords,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "accumulates_route_local_codeword_dot_lut": (
            accumulates_route_local_codeword_dot_lut
        ),
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "uses_expert_kblock_reuse_schedule": uses_expert_kblock_reuse_schedule,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_rowwise_codeword_tile_accumulate_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _ROWWISE_CODEWORD_TILE_ACCUMULATE_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_name = "e8p_rowwise_codeword_tile_accumulate_rhs_sorted_matmul"
    primitive_binding_present = (
        primitive_name in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(primitive_source, primitive_name)
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    dispatch_grid_ok = (
        "route_microtiles_x_output_tiles_x_k_blocks_x_codeword_tiles"
        in primitive_source
        or all(
            token in primitive_source
            for token in (
                "route_microtile",
                "output_tile",
                "k_block",
                "codeword_tile",
            )
        )
    )
    preserves_route_microtile_axis = "route_microtile" in scoped_source
    preserves_output_tile_axis = "output_tile" in scoped_source
    preserves_k_block_axis = "k_block" in scoped_source
    preserves_codeword_tile_axis = "codeword_tile" in scoped_source
    streams_compressed_codeword_tiles = (
        ("codeword_tiles" in kernel or "code_tiles" in kernel)
        and (
            "uint16_t" in kernel
            or "ushort" in kernel
            or "uint16" in kernel
            or "compressed_codeword" in kernel
        )
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    accumulates_online_activation_codeword_dots = any(
        token in kernel
        for token in (
            "online_activation_codeword_dot",
            "activation_codeword_dot",
            "rowwise_codeword_accum",
        )
    ) and "accum" in kernel
    materializes_route_local_full_lut = any(
        token in scoped_source
        for token in (
            "route_local_codeword_dot_lut",
            "route_codeword_lut_accumulators",
            "route_codeword_lut_codeword_ids",
        )
    )
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel

    passes_contract = (
        kernel_present
        and primitive_binding_present
        and dispatch_grid_ok
        and preserves_route_microtile_axis
        and preserves_output_tile_axis
        and preserves_k_block_axis
        and preserves_codeword_tile_axis
        and streams_compressed_codeword_tiles
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and accumulates_online_activation_codeword_dots
        and not materializes_route_local_full_lut
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
    )

    if passes_contract:
        decision = "rowwise_codeword_tile_accumulate_source_guardrail_present"
        required_next_features = [
            "prove_rowwise_codeword_tile_accumulate_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_rowwise_codeword_tile_accumulate_source"
        required_next_features = [
            "add_rowwise_codeword_tile_accumulate_native_source_guardrail",
            "bind_rowwise_codeword_tile_accumulate_primitive",
            "preserve_route_microtile_and_output_tile_axes",
            "stream_compressed_uint16_codeword_tiles_and_scale_slots",
            "accumulate_online_activation_codeword_dots",
            "avoid_route_local_full_codeword_lut",
            "avoid_decoded_dense_rhs",
        ]
        rejected_next_steps = [
            "do_not_benchmark_rowwise_codeword_tile_accumulate_before_source_guardrail",
            "do_not_retime_route_codeword_lut_accumulate",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "route_microtiles_x_output_tiles_x_k_blocks_x_codeword_tiles"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_route_microtile_axis": preserves_route_microtile_axis,
        "preserves_output_tile_axis": preserves_output_tile_axis,
        "preserves_k_block_axis": preserves_k_block_axis,
        "preserves_codeword_tile_axis": preserves_codeword_tile_axis,
        "streams_compressed_codeword_tiles": streams_compressed_codeword_tiles,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "accumulates_online_activation_codeword_dots": (
            accumulates_online_activation_codeword_dots
        ),
        "materializes_route_local_full_lut": materializes_route_local_full_lut,
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_output_tile_local_codeword_lut_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _OUTPUT_TILE_LOCAL_CODEWORD_LUT_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_name = "e8p_output_tile_local_codeword_lut_rhs_sorted_matmul"
    primitive_binding_present = (
        primitive_name in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(primitive_source, primitive_name)
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    dispatch_grid_ok = (
        "route_microtiles_x_output_tiles_x_k_blocks_x_unique_codewords_then_output_rows"
        in primitive_source
        or all(
            token in primitive_source
            for token in (
                "route_microtile",
                "output_tile",
                "k_block",
                "unique_codeword",
                "output_row",
            )
        )
    )
    preserves_route_microtile_axis = "route_microtile" in scoped_source
    preserves_output_tile_axis = "output_tile" in scoped_source
    preserves_k_block_axis = "k_block" in scoped_source
    builds_output_tile_local_unique_codeword_lut = all(
        token in scoped_source
        for token in (
            "output_tile_local",
            "unique_codeword",
            "codeword_lut",
        )
    )
    reuses_activation_codeword_dots_across_output_rows = any(
        token in scoped_source
        for token in (
            "reuse_activation_codeword_dot",
            "output_tile_local_activation_dot_lut",
            "activation_codeword_dot_lut",
        )
    )
    streams_compressed_codeword_tiles = (
        ("codeword_tiles" in kernel or "code_tiles" in kernel)
        and (
            "uint16_t" in kernel
            or "ushort" in kernel
            or "uint16" in kernel
            or "compressed_codeword" in kernel
        )
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    recomputes_per_output_row_codeword_dots = any(
        token in scoped_source
        for token in (
            "rowwise_codeword_accum",
            "online_activation_codeword_dot",
            "per_output_row_codeword_dot",
        )
    )
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel

    passes_contract = (
        kernel_present
        and primitive_binding_present
        and dispatch_grid_ok
        and preserves_route_microtile_axis
        and preserves_output_tile_axis
        and preserves_k_block_axis
        and builds_output_tile_local_unique_codeword_lut
        and reuses_activation_codeword_dots_across_output_rows
        and streams_compressed_codeword_tiles
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and not recomputes_per_output_row_codeword_dots
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
    )

    if passes_contract:
        decision = "output_tile_local_codeword_lut_source_guardrail_present"
        required_next_features = [
            "prove_output_tile_local_codeword_lut_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_output_tile_local_codeword_lut_source"
        required_next_features = [
            "add_output_tile_local_codeword_lut_native_source_guardrail",
            "bind_output_tile_local_codeword_lut_primitive",
            "preserve_route_microtile_and_output_tile_axes",
            "build_output_tile_local_unique_codeword_lut",
            "reuse_activation_codeword_dots_across_output_rows",
            "stream_compressed_uint16_codeword_tiles_and_scale_slots",
            "avoid_rowwise_per_output_row_codeword_dot_recompute",
            "avoid_decoded_dense_rhs",
        ]
        rejected_next_steps = [
            "do_not_benchmark_output_tile_local_codeword_lut_before_source_guardrail",
            "do_not_retime_rowwise_codeword_tile_accumulate",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "route_microtiles_x_output_tiles_x_k_blocks_x_unique_codewords_then_output_rows"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_route_microtile_axis": preserves_route_microtile_axis,
        "preserves_output_tile_axis": preserves_output_tile_axis,
        "preserves_k_block_axis": preserves_k_block_axis,
        "builds_output_tile_local_unique_codeword_lut": (
            builds_output_tile_local_unique_codeword_lut
        ),
        "reuses_activation_codeword_dots_across_output_rows": (
            reuses_activation_codeword_dots_across_output_rows
        ),
        "streams_compressed_codeword_tiles": streams_compressed_codeword_tiles,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "recomputes_per_output_row_codeword_dots": (
            recomputes_per_output_row_codeword_dots
        ),
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_route_microtile_codeword_block_reduce_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _ROUTE_MICROTILE_CODEWORD_BLOCK_REDUCE_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_name = "e8p_route_microtile_codeword_block_reduce_rhs_sorted_matmul"
    primitive_binding_present = (
        primitive_name in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(primitive_source, primitive_name)
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    dispatch_grid_ok = (
        "route_microtiles_x_k_blocks_x_codeword_blocks_then_output_tiles"
        in primitive_source
        or all(
            token in primitive_source
            for token in (
                "route_microtile",
                "k_block",
                "codeword_block",
                "output_tile",
            )
        )
    )
    preserves_route_microtile_axis = "route_microtile" in scoped_source
    preserves_k_block_axis = "k_block" in scoped_source
    preserves_codeword_block_axis = "codeword_block" in scoped_source
    reduces_route_microtile_codeword_block_partials = all(
        token in scoped_source
        for token in ("route_microtile", "codeword_block", "partial")
    )
    writes_output_tiles_after_block_reduction = (
        "output_tile" in scoped_source
        and any(token in scoped_source for token in ("writeback", "write_back"))
    )
    streams_compressed_codeword_tiles = (
        ("codeword_tiles" in kernel or "code_tiles" in kernel)
        and (
            "uint16_t" in kernel
            or "ushort" in kernel
            or "uint16" in kernel
            or "compressed_codeword" in kernel
        )
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    materializes_output_tile_local_full_lut = any(
        token in scoped_source
        for token in (
            "output_tile_local_unique_codeword_lut",
            "output_tile_local_activation_dot_lut",
            "unique_codeword_lut",
        )
    )
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel

    passes_contract = (
        kernel_present
        and primitive_binding_present
        and dispatch_grid_ok
        and preserves_route_microtile_axis
        and preserves_k_block_axis
        and preserves_codeword_block_axis
        and reduces_route_microtile_codeword_block_partials
        and writes_output_tiles_after_block_reduction
        and streams_compressed_codeword_tiles
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and not materializes_output_tile_local_full_lut
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
    )

    if passes_contract:
        decision = "route_microtile_codeword_block_reduce_source_guardrail_present"
        required_next_features = [
            "prove_route_microtile_codeword_block_reduce_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_route_microtile_codeword_block_reduce_source"
        required_next_features = [
            "add_route_microtile_codeword_block_reduce_native_source_guardrail",
            "bind_route_microtile_codeword_block_reduce_primitive",
            "preserve_route_microtile_and_k_block_axes",
            "reduce_route_microtile_codeword_block_partials",
            "write_back_output_tiles_after_block_reduction",
            "stream_compressed_uint16_codeword_tiles_and_scale_slots",
            "avoid_output_tile_local_full_codeword_pass",
            "avoid_decoded_dense_rhs",
        ]
        rejected_next_steps = [
            "do_not_benchmark_route_microtile_codeword_block_reduce_before_source_guardrail",
            "do_not_retime_output_tile_local_codeword_lut",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "route_microtiles_x_k_blocks_x_codeword_blocks_then_output_tiles"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_route_microtile_axis": preserves_route_microtile_axis,
        "preserves_k_block_axis": preserves_k_block_axis,
        "preserves_codeword_block_axis": preserves_codeword_block_axis,
        "reduces_route_microtile_codeword_block_partials": (
            reduces_route_microtile_codeword_block_partials
        ),
        "writes_output_tiles_after_block_reduction": (
            writes_output_tiles_after_block_reduction
        ),
        "streams_compressed_codeword_tiles": streams_compressed_codeword_tiles,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "materializes_output_tile_local_full_lut": (
            materializes_output_tile_local_full_lut
        ),
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_kblock_wavefront_codeword_scan_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _KBLOCK_WAVEFRONT_CODEWORD_SCAN_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_name = "e8p_kblock_wavefront_codeword_scan_rhs_sorted_matmul"
    primitive_binding_present = (
        primitive_name in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(primitive_source, primitive_name)
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    dispatch_grid_ok = (
        "k_blocks_x_route_microtiles_x_output_stripes_x_codeword_groups"
        in primitive_source
        or all(
            token in primitive_source
            for token in (
                "k_block",
                "route_microtile",
                "output_stripe",
                "codeword_group",
            )
        )
    )
    preserves_k_block_axis = "k_block" in scoped_source
    preserves_route_microtile_axis = "route_microtile" in scoped_source
    preserves_output_stripe_axis = "output_stripe" in scoped_source
    preserves_codeword_group_axis = "codeword_group" in scoped_source
    streams_codeword_groups_as_wavefronts = all(
        token in scoped_source
        for token in ("kblock_wavefront", "codeword_group", "wavefront")
    )
    streams_compressed_codeword_tiles = (
        (
            "codeword_tiles" in kernel
            or "codeword_groups" in kernel
            or "code_tiles" in kernel
        )
        and (
            "uint16_t" in kernel
            or "ushort" in kernel
            or "uint16" in kernel
            or "compressed_codeword" in kernel
        )
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    materializes_route_microtile_partial_cache = any(
        token in scoped_source
        for token in (
            "route_microtile_codeword_block_partial",
            "codeword_block_partial_cache",
            "route_microtile_partial_cache",
        )
    )
    materializes_output_tile_local_full_lut = any(
        token in scoped_source
        for token in (
            "output_tile_local_unique_codeword_lut",
            "output_tile_local_activation_dot_lut",
            "unique_codeword_lut",
        )
    )
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel

    passes_contract = (
        kernel_present
        and primitive_binding_present
        and dispatch_grid_ok
        and preserves_k_block_axis
        and preserves_route_microtile_axis
        and preserves_output_stripe_axis
        and preserves_codeword_group_axis
        and streams_codeword_groups_as_wavefronts
        and streams_compressed_codeword_tiles
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and not materializes_route_microtile_partial_cache
        and not materializes_output_tile_local_full_lut
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
    )

    if passes_contract:
        decision = "kblock_wavefront_codeword_scan_source_guardrail_present"
        required_next_features = [
            "prove_kblock_wavefront_codeword_scan_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_kblock_wavefront_codeword_scan_source"
        required_next_features = [
            "add_kblock_wavefront_codeword_scan_native_source_guardrail",
            "bind_kblock_wavefront_codeword_scan_primitive",
            "preserve_k_block_route_microtile_output_stripe_codeword_group_axes",
            "stream_compressed_codeword_groups_as_kblock_wavefronts",
            "stream_compressed_uint16_codeword_tiles_and_scale_slots",
            "avoid_route_microtile_codeword_block_partial_cache",
            "avoid_output_tile_local_full_codeword_pass",
            "avoid_decoded_dense_rhs",
        ]
        rejected_next_steps = [
            "do_not_benchmark_kblock_wavefront_codeword_scan_before_source_guardrail",
            "do_not_retime_route_microtile_codeword_block_reduce",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "k_blocks_x_route_microtiles_x_output_stripes_x_codeword_groups"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_k_block_axis": preserves_k_block_axis,
        "preserves_route_microtile_axis": preserves_route_microtile_axis,
        "preserves_output_stripe_axis": preserves_output_stripe_axis,
        "preserves_codeword_group_axis": preserves_codeword_group_axis,
        "streams_codeword_groups_as_wavefronts": streams_codeword_groups_as_wavefronts,
        "streams_compressed_codeword_tiles": streams_compressed_codeword_tiles,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "materializes_route_microtile_partial_cache": (
            materializes_route_microtile_partial_cache
        ),
        "materializes_output_tile_local_full_lut": (
            materializes_output_tile_local_full_lut
        ),
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_token_route_output_stripe_pipeline_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _TOKEN_ROUTE_OUTPUT_STRIPE_PIPELINE_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_name = "e8p_token_route_output_stripe_pipeline_rhs_sorted_matmul"
    primitive_binding_present = (
        primitive_name in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(primitive_source, primitive_name)
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    dispatch_grid_ok = (
        "tokens_x_route_slots_x_output_stripes_x_kblock_stages" in primitive_source
        or all(
            token in primitive_source
            for token in (
                "token",
                "route_slot",
                "output_stripe",
                "kblock_stage",
            )
        )
    )
    preserves_token_axis = "token" in scoped_source
    preserves_route_slot_axis = "route_slot" in scoped_source
    preserves_output_stripe_axis = "output_stripe" in scoped_source
    preserves_kblock_stage_axis = (
        "kblock_stage" in scoped_source or "k_block" in scoped_source
    )
    accumulates_full_output_without_route_expansion = (
        "token_route_output_stripe" in scoped_source
        and "output_stripe" in scoped_source
        and "accum" in scoped_source
    )
    streams_kblock_stages_inside_token_route_output_stripes = all(
        token in scoped_source
        for token in ("token_route_output_stripe", "kblock_stage", "codeword")
    )
    streams_compressed_codeword_tiles = (
        (
            "codeword_tiles" in kernel
            or "code_tiles" in kernel
            or "codeword_stage" in kernel
        )
        and (
            "uint16_t" in kernel
            or "ushort" in kernel
            or "uint16" in kernel
            or "compressed_codeword" in kernel
        )
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    materializes_route_microtile_partial_cache = any(
        token in scoped_source
        for token in (
            "route_microtile_codeword_block_partial",
            "codeword_block_partial_cache",
            "route_microtile_partial_cache",
        )
    )
    materializes_output_tile_local_full_lut = any(
        token in scoped_source
        for token in (
            "output_tile_local_unique_codeword_lut",
            "output_tile_local_activation_dot_lut",
            "unique_codeword_lut",
        )
    )
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel
    uses_rejected_kblock_wavefront_schedule = (
        "kblock_wavefront" in scoped_source
        or "k_blocks_x_route_microtiles_x_output_stripes_x_codeword_groups"
        in scoped_source
    )
    expands_route_microtiles = (
        "route_microtile" in scoped_source or "route_microtiles" in scoped_source
    )

    passes_contract = (
        kernel_present
        and primitive_binding_present
        and dispatch_grid_ok
        and preserves_token_axis
        and preserves_route_slot_axis
        and preserves_output_stripe_axis
        and preserves_kblock_stage_axis
        and accumulates_full_output_without_route_expansion
        and streams_kblock_stages_inside_token_route_output_stripes
        and streams_compressed_codeword_tiles
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and not materializes_route_microtile_partial_cache
        and not materializes_output_tile_local_full_lut
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
        and not uses_rejected_kblock_wavefront_schedule
        and not expands_route_microtiles
    )

    if passes_contract:
        decision = "token_route_output_stripe_pipeline_source_guardrail_present"
        required_next_features = [
            "prove_token_route_output_stripe_pipeline_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_token_route_output_stripe_pipeline_source"
        required_next_features = [
            "add_token_route_output_stripe_pipeline_native_source_guardrail",
            "bind_token_route_output_stripe_pipeline_primitive",
            "preserve_token_route_slot_output_stripe_kblock_stage_axes",
            "accumulate_full_output_inside_token_route_output_stripes",
            "stream_kblock_stages_inside_each_token_route_output_stripe",
            "stream_compressed_uint16_codeword_tiles_and_scale_slots",
            "avoid_route_microtile_output_expansion",
            "avoid_decoded_dense_rhs",
        ]
        rejected_next_steps = [
            "do_not_benchmark_token_route_output_stripe_pipeline_before_source_guardrail",
            "do_not_retime_kblock_wavefront_codeword_scan",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "tokens_x_route_slots_x_output_stripes_x_kblock_stages"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_token_axis": preserves_token_axis,
        "preserves_route_slot_axis": preserves_route_slot_axis,
        "preserves_output_stripe_axis": preserves_output_stripe_axis,
        "preserves_kblock_stage_axis": preserves_kblock_stage_axis,
        "accumulates_full_output_without_route_expansion": (
            accumulates_full_output_without_route_expansion
        ),
        "streams_kblock_stages_inside_token_route_output_stripes": (
            streams_kblock_stages_inside_token_route_output_stripes
        ),
        "streams_compressed_codeword_tiles": streams_compressed_codeword_tiles,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "materializes_route_microtile_partial_cache": (
            materializes_route_microtile_partial_cache
        ),
        "materializes_output_tile_local_full_lut": (
            materializes_output_tile_local_full_lut
        ),
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "uses_rejected_kblock_wavefront_schedule": (
            uses_rejected_kblock_wavefront_schedule
        ),
        "expands_route_microtiles": expands_route_microtiles,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_expert_kblock_scale_slot_stream_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _EXPERT_KBLOCK_SCALE_SLOT_STREAM_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_name = "e8p_expert_kblock_scale_slot_stream_rhs_sorted_matmul"
    primitive_binding_present = (
        primitive_name in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(primitive_source, primitive_name)
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    dispatch_grid_ok = (
        "experts_x_k_blocks_x_scale_groups_x_route_tiles_x_output_tiles"
        in primitive_source
        or all(
            token in primitive_source
            for token in (
                "expert",
                "k_block",
                "scale_group",
                "route_tile",
                "output_tile",
            )
        )
    )
    preserves_expert_axis = "expert" in scoped_source
    preserves_k_block_axis = "k_block" in scoped_source or "kblock" in scoped_source
    preserves_scale_group_axis = "scale_group" in scoped_source
    preserves_route_tile_axis = "route_tile" in scoped_source
    preserves_output_tile_axis = "output_tile" in scoped_source
    streams_scale_slot_groups = (
        "scale_slot_stream" in scoped_source
        or "scale_group" in scoped_source
        and "codeword_scale_slots" in scoped_source
    )
    reuses_codeword_scale_slots_across_output_tiles = (
        "codeword_scale_slots" in scoped_source
        and "output_tile" in scoped_source
        and "scale_group" in scoped_source
    )
    streams_compressed_codeword_tiles = (
        (
            "codeword_tiles" in kernel
            or "code_tiles" in kernel
            or "compressed_codeword" in kernel
        )
        and (
            "uint16_t" in kernel
            or "ushort" in kernel
            or "uint16" in kernel
            or "compressed_codeword" in kernel
        )
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_group_indices = "scale_group_indices" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    uses_rejected_token_route_schedule = (
        "token_route_output_stripe" in scoped_source
        or "tokens_x_route_slots_x_output_stripes_x_kblock_stages" in scoped_source
    )
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel
    reconstructs_e8p_per_fragment = (
        "mlx_vq_decode_e8p" in kernel and "fragment" in kernel
    )

    passes_contract = (
        kernel_present
        and primitive_binding_present
        and dispatch_grid_ok
        and preserves_expert_axis
        and preserves_k_block_axis
        and preserves_scale_group_axis
        and preserves_route_tile_axis
        and preserves_output_tile_axis
        and streams_scale_slot_groups
        and reuses_codeword_scale_slots_across_output_tiles
        and streams_compressed_codeword_tiles
        and uses_codeword_scale_slots
        and uses_scale_group_indices
        and uses_scale_tiles
        and not uses_rejected_token_route_schedule
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
        and not reconstructs_e8p_per_fragment
    )

    if passes_contract:
        decision = "expert_kblock_scale_slot_stream_source_guardrail_present"
        required_next_features = [
            "prove_expert_kblock_scale_slot_stream_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_expert_kblock_scale_slot_stream_source"
        required_next_features = [
            "add_expert_kblock_scale_slot_stream_native_source_guardrail",
            "bind_expert_kblock_scale_slot_stream_primitive",
            "preserve_expert_kblock_scale_group_route_tile_output_tile_axes",
            "stream_compressed_uint16_codeword_tiles_and_scale_slots",
            "preserve_scale_group_indices_for_group_size_352",
            "reuse_codeword_scale_slots_across_output_tiles_without_decoded_rhs_cache",
            "avoid_token_route_output_stripe_pipeline_schedule",
            "avoid_decoded_dense_rhs",
        ]
        rejected_next_steps = [
            "do_not_benchmark_expert_kblock_scale_slot_stream_before_source_guardrail",
            "do_not_retime_token_route_output_stripe_pipeline",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "experts_x_k_blocks_x_scale_groups_x_route_tiles_x_output_tiles"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_expert_axis": preserves_expert_axis,
        "preserves_k_block_axis": preserves_k_block_axis,
        "preserves_scale_group_axis": preserves_scale_group_axis,
        "preserves_route_tile_axis": preserves_route_tile_axis,
        "preserves_output_tile_axis": preserves_output_tile_axis,
        "streams_scale_slot_groups": streams_scale_slot_groups,
        "reuses_codeword_scale_slots_across_output_tiles": (
            reuses_codeword_scale_slots_across_output_tiles
        ),
        "streams_compressed_codeword_tiles": streams_compressed_codeword_tiles,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_group_indices": uses_scale_group_indices,
        "uses_scale_tiles": uses_scale_tiles,
        "uses_rejected_token_route_schedule": uses_rejected_token_route_schedule,
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "reconstructs_e8p_per_fragment": reconstructs_e8p_per_fragment,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_scale_group_route_block_reduce_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _SCALE_GROUP_ROUTE_BLOCK_REDUCE_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_name = "e8p_scale_group_route_block_reduce_rhs_sorted_matmul"
    primitive_binding_present = (
        primitive_name in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(primitive_source, primitive_name)
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    dispatch_grid_ok = (
        "scale_groups_x_route_blocks_x_k_blocks_x_output_tiles" in primitive_source
        or all(
            token in primitive_source
            for token in ("scale_group", "route_block", "k_block", "output_tile")
        )
    )
    preserves_scale_group_axis = "scale_group" in scoped_source
    preserves_route_block_axis = "route_block" in scoped_source
    preserves_k_block_axis = "k_block" in scoped_source or "kblock" in scoped_source
    preserves_output_tile_axis = "output_tile" in scoped_source
    reduces_route_block_partials_before_output_writeback = (
        "route_block_partial" in scoped_source
        or (
            "route_block" in scoped_source
            and "partial" in scoped_source
            and "output_tile" in scoped_source
        )
    )
    streams_compressed_codeword_tiles = (
        (
            "codeword_tiles" in kernel
            or "code_tiles" in kernel
            or "compressed_codeword" in kernel
        )
        and (
            "uint16_t" in kernel
            or "ushort" in kernel
            or "uint16" in kernel
            or "compressed_codeword" in kernel
        )
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_group_indices = "scale_group_indices" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    uses_rejected_expert_kblock_scale_slot_schedule = (
        "expert_kblock_scale_slot_stream" in scoped_source
        or "experts_x_k_blocks_x_scale_groups_x_route_tiles_x_output_tiles"
        in scoped_source
    )
    uses_rejected_token_route_schedule = (
        "token_route_output_stripe" in scoped_source
        or "tokens_x_route_slots_x_output_stripes_x_kblock_stages" in scoped_source
    )
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel
    reconstructs_e8p_per_fragment = (
        "mlx_vq_decode_e8p" in kernel and "fragment" in kernel
    )

    passes_contract = (
        kernel_present
        and primitive_binding_present
        and dispatch_grid_ok
        and preserves_scale_group_axis
        and preserves_route_block_axis
        and preserves_k_block_axis
        and preserves_output_tile_axis
        and reduces_route_block_partials_before_output_writeback
        and streams_compressed_codeword_tiles
        and uses_codeword_scale_slots
        and uses_scale_group_indices
        and uses_scale_tiles
        and not uses_rejected_expert_kblock_scale_slot_schedule
        and not uses_rejected_token_route_schedule
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
        and not reconstructs_e8p_per_fragment
    )

    if passes_contract:
        decision = "scale_group_route_block_reduce_source_guardrail_present"
        required_next_features = [
            "prove_scale_group_route_block_reduce_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_scale_group_route_block_reduce_source"
        required_next_features = [
            "add_scale_group_route_block_reduce_native_source_guardrail",
            "bind_scale_group_route_block_reduce_primitive",
            "preserve_scale_group_route_block_k_block_output_tile_axes",
            "reduce_route_block_partials_before_output_tile_writeback",
            "stream_compressed_uint16_codeword_tiles_and_scale_slots",
            "preserve_scale_group_indices_for_group_size_352",
            "avoid_expert_kblock_scale_slot_stream_schedule",
            "avoid_token_route_output_stripe_pipeline_schedule",
            "avoid_decoded_dense_rhs",
        ]
        rejected_next_steps = [
            "do_not_benchmark_scale_group_route_block_reduce_before_source_guardrail",
            "do_not_retime_expert_kblock_scale_slot_stream",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "scale_groups_x_route_blocks_x_k_blocks_x_output_tiles"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_scale_group_axis": preserves_scale_group_axis,
        "preserves_route_block_axis": preserves_route_block_axis,
        "preserves_k_block_axis": preserves_k_block_axis,
        "preserves_output_tile_axis": preserves_output_tile_axis,
        "reduces_route_block_partials_before_output_writeback": (
            reduces_route_block_partials_before_output_writeback
        ),
        "streams_compressed_codeword_tiles": streams_compressed_codeword_tiles,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_group_indices": uses_scale_group_indices,
        "uses_scale_tiles": uses_scale_tiles,
        "uses_rejected_expert_kblock_scale_slot_schedule": (
            uses_rejected_expert_kblock_scale_slot_schedule
        ),
        "uses_rejected_token_route_schedule": uses_rejected_token_route_schedule,
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "reconstructs_e8p_per_fragment": reconstructs_e8p_per_fragment,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_route_block_output_group_stream_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _ROUTE_BLOCK_OUTPUT_GROUP_STREAM_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_name = "e8p_route_block_output_group_stream_rhs_sorted_matmul"
    primitive_binding_present = (
        primitive_name in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(primitive_source, primitive_name)
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    dispatch_grid_ok = (
        "route_blocks_x_output_groups_x_k_blocks_x_codeword_groups"
        in primitive_source
        or all(
            token in primitive_source
            for token in ("route_block", "output_group", "k_block", "codeword_group")
        )
    )
    preserves_route_block_axis = "route_block" in scoped_source
    preserves_output_group_axis = "output_group" in scoped_source
    preserves_k_block_axis = "k_block" in scoped_source or "kblock" in scoped_source
    preserves_codeword_group_axis = "codeword_group" in scoped_source
    writes_output_group_accumulators_to_route_slots = (
        "output_group_accumulator" in scoped_source
        and ("route_slot" in scoped_source or "route_slots" in scoped_source)
    )
    streams_codeword_groups_inside_route_block_output_groups = all(
        token in scoped_source
        for token in ("route_block_output_group", "codeword_group", "codeword")
    )
    streams_compressed_codeword_tiles = (
        (
            "codeword_tiles" in kernel
            or "code_tiles" in kernel
            or "compressed_codeword" in kernel
        )
        and (
            "uint16_t" in kernel
            or "ushort" in kernel
            or "uint16" in kernel
            or "compressed_codeword" in kernel
        )
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    uses_rejected_scale_group_route_block_schedule = (
        "scale_group_route_block_reduce" in scoped_source
        or "scale_groups_x_route_blocks_x_k_blocks_x_output_tiles" in scoped_source
        or "route_block_partial" in scoped_source
    )
    uses_rejected_expert_kblock_scale_slot_schedule = (
        "expert_kblock_scale_slot_stream" in scoped_source
        or "experts_x_k_blocks_x_scale_groups_x_route_tiles_x_output_tiles"
        in scoped_source
    )
    uses_rejected_token_route_schedule = (
        "token_route_output_stripe" in scoped_source
        or "tokens_x_route_slots_x_output_stripes_x_kblock_stages" in scoped_source
    )
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel
    reconstructs_e8p_per_fragment = (
        "mlx_vq_decode_e8p" in kernel and "fragment" in kernel
    )

    passes_contract = (
        kernel_present
        and primitive_binding_present
        and dispatch_grid_ok
        and preserves_route_block_axis
        and preserves_output_group_axis
        and preserves_k_block_axis
        and preserves_codeword_group_axis
        and writes_output_group_accumulators_to_route_slots
        and streams_codeword_groups_inside_route_block_output_groups
        and streams_compressed_codeword_tiles
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and not uses_rejected_scale_group_route_block_schedule
        and not uses_rejected_expert_kblock_scale_slot_schedule
        and not uses_rejected_token_route_schedule
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
        and not reconstructs_e8p_per_fragment
    )

    if passes_contract:
        decision = "route_block_output_group_stream_source_guardrail_present"
        required_next_features = [
            "prove_route_block_output_group_stream_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_route_block_output_group_stream_source"
        required_next_features = [
            "add_route_block_output_group_stream_native_source_guardrail",
            "bind_route_block_output_group_stream_primitive",
            "preserve_route_block_output_group_k_block_codeword_group_axes",
            "write_output_group_accumulators_directly_to_route_slots",
            "stream_compressed_uint16_codeword_tiles_and_scale_slots",
            "avoid_scale_group_route_block_reduce_schedule",
            "avoid_decoded_dense_rhs",
        ]
        rejected_next_steps = [
            "do_not_benchmark_route_block_output_group_stream_before_source_guardrail",
            "do_not_retime_scale_group_route_block_reduce",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "route_blocks_x_output_groups_x_k_blocks_x_codeword_groups"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_route_block_axis": preserves_route_block_axis,
        "preserves_output_group_axis": preserves_output_group_axis,
        "preserves_k_block_axis": preserves_k_block_axis,
        "preserves_codeword_group_axis": preserves_codeword_group_axis,
        "writes_output_group_accumulators_to_route_slots": (
            writes_output_group_accumulators_to_route_slots
        ),
        "streams_codeword_groups_inside_route_block_output_groups": (
            streams_codeword_groups_inside_route_block_output_groups
        ),
        "streams_compressed_codeword_tiles": streams_compressed_codeword_tiles,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "uses_rejected_scale_group_route_block_schedule": (
            uses_rejected_scale_group_route_block_schedule
        ),
        "uses_rejected_expert_kblock_scale_slot_schedule": (
            uses_rejected_expert_kblock_scale_slot_schedule
        ),
        "uses_rejected_token_route_schedule": uses_rejected_token_route_schedule,
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "reconstructs_e8p_per_fragment": reconstructs_e8p_per_fragment,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_output_group_pretransposed_codeword_stream_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _OUTPUT_GROUP_PRETRANSPOSED_CODEWORD_STREAM_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_name = "e8p_output_group_pretransposed_codeword_stream_rhs_sorted_matmul"
    primitive_binding_present = (
        primitive_name in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(primitive_source, primitive_name)
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    dispatch_grid_ok = (
        "output_groups_x_route_blocks_x_k_blocks_x_codeword_groups"
        in primitive_source
        or all(
            token in primitive_source
            for token in ("output_group", "route_block", "k_block", "codeword_group")
        )
    )
    preserves_output_group_axis = "output_group" in scoped_source
    preserves_route_block_axis = "route_block" in scoped_source
    preserves_k_block_axis = "k_block" in scoped_source or "kblock" in scoped_source
    preserves_codeword_group_axis = "codeword_group" in scoped_source
    uses_output_group_pretransposed_streams = (
        "output_group_pretransposed" in scoped_source
        or "pretransposed_codeword_stream" in scoped_source
    )
    feeds_route_block_accumulators = (
        "route_block_accumulator" in scoped_source
        or "route_block_accumulators" in scoped_source
    )
    streams_compressed_codeword_tiles = (
        (
            "codeword_tiles" in kernel
            or "code_tiles" in kernel
            or "compressed_codeword" in kernel
        )
        and (
            "uint16_t" in kernel
            or "ushort" in kernel
            or "uint16" in kernel
            or "compressed_codeword" in kernel
        )
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    uses_rejected_route_block_output_group_schedule = (
        "route_block_output_group_stream" in scoped_source
        or "route_blocks_x_output_groups_x_k_blocks_x_codeword_groups"
        in scoped_source
        or "route_block_output_group" in scoped_source
    )
    uses_rejected_scale_group_route_block_schedule = (
        "scale_group_route_block_reduce" in scoped_source
        or "scale_groups_x_route_blocks_x_k_blocks_x_output_tiles" in scoped_source
        or "route_block_partial" in scoped_source
    )
    uses_rejected_expert_kblock_scale_slot_schedule = (
        "expert_kblock_scale_slot_stream" in scoped_source
        or "experts_x_k_blocks_x_scale_groups_x_route_tiles_x_output_tiles"
        in scoped_source
    )
    uses_rejected_token_route_schedule = (
        "token_route_output_stripe" in scoped_source
        or "tokens_x_route_slots_x_output_stripes_x_kblock_stages" in scoped_source
    )
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel
    reconstructs_e8p_per_fragment = (
        "mlx_vq_decode_e8p" in kernel and "fragment" in kernel
    )

    passes_contract = (
        kernel_present
        and primitive_binding_present
        and dispatch_grid_ok
        and preserves_output_group_axis
        and preserves_route_block_axis
        and preserves_k_block_axis
        and preserves_codeword_group_axis
        and uses_output_group_pretransposed_streams
        and feeds_route_block_accumulators
        and streams_compressed_codeword_tiles
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and not uses_rejected_route_block_output_group_schedule
        and not uses_rejected_scale_group_route_block_schedule
        and not uses_rejected_expert_kblock_scale_slot_schedule
        and not uses_rejected_token_route_schedule
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
        and not reconstructs_e8p_per_fragment
    )

    if passes_contract:
        decision = "output_group_pretransposed_codeword_stream_source_guardrail_present"
        required_next_features = [
            "prove_output_group_pretransposed_codeword_stream_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_output_group_pretransposed_codeword_stream_source"
        required_next_features = [
            "add_output_group_pretransposed_codeword_stream_native_source_guardrail",
            "bind_output_group_pretransposed_codeword_stream_primitive",
            "preserve_output_group_route_block_k_block_codeword_group_axes",
            "make_output_group_pretransposed_streams_outer_schedule",
            "stream_compressed_uint16_codeword_tiles_and_scale_slots",
            "avoid_route_block_output_group_stream_schedule",
            "avoid_decoded_dense_rhs",
        ]
        rejected_next_steps = [
            "do_not_benchmark_output_group_pretransposed_codeword_stream_before_source_guardrail",
            "do_not_retime_route_block_output_group_stream",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "output_groups_x_route_blocks_x_k_blocks_x_codeword_groups"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_output_group_axis": preserves_output_group_axis,
        "preserves_route_block_axis": preserves_route_block_axis,
        "preserves_k_block_axis": preserves_k_block_axis,
        "preserves_codeword_group_axis": preserves_codeword_group_axis,
        "uses_output_group_pretransposed_streams": (
            uses_output_group_pretransposed_streams
        ),
        "feeds_route_block_accumulators": feeds_route_block_accumulators,
        "streams_compressed_codeword_tiles": streams_compressed_codeword_tiles,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "uses_rejected_route_block_output_group_schedule": (
            uses_rejected_route_block_output_group_schedule
        ),
        "uses_rejected_scale_group_route_block_schedule": (
            uses_rejected_scale_group_route_block_schedule
        ),
        "uses_rejected_expert_kblock_scale_slot_schedule": (
            uses_rejected_expert_kblock_scale_slot_schedule
        ),
        "uses_rejected_token_route_schedule": uses_rejected_token_route_schedule,
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "reconstructs_e8p_per_fragment": reconstructs_e8p_per_fragment,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_kblock_output_group_route_fused_stream_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _KBLOCK_OUTPUT_GROUP_ROUTE_FUSED_STREAM_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_name = "e8p_kblock_output_group_route_fused_stream_rhs_sorted_matmul"
    primitive_binding_present = (
        primitive_name in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(primitive_source, primitive_name)
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    dispatch_grid_ok = (
        "k_blocks_x_output_groups_x_route_blocks_x_codeword_groups"
        in primitive_source
        or all(
            token in primitive_source
            for token in ("k_block", "output_group", "route_block", "codeword_group")
        )
    )
    preserves_k_block_axis = "k_block" in scoped_source or "kblock" in scoped_source
    preserves_output_group_axis = "output_group" in scoped_source
    preserves_route_block_axis = "route_block" in scoped_source
    preserves_codeword_group_axis = "codeword_group" in scoped_source
    uses_kblock_output_group_route_fusion = (
        "kblock_output_group_route_fused" in scoped_source
        or "k_block_output_group_route_fused" in scoped_source
        or "kblock_output_group_route" in scoped_source
    )
    writes_route_slots_after_fusion = (
        "route_slot" in scoped_source
        and (
            "route_fused_accumulator" in scoped_source
            or "kblock_output_group_route_accumulator" in scoped_source
            or "route_fused" in scoped_source
        )
    )
    streams_compressed_codeword_tiles = (
        (
            "codeword_tiles" in kernel
            or "code_tiles" in kernel
            or "compressed_codeword" in kernel
        )
        and (
            "uint16_t" in kernel
            or "ushort" in kernel
            or "uint16" in kernel
            or "compressed_codeword" in kernel
        )
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    uses_rejected_output_group_pretransposed_schedule = (
        "output_group_pretransposed" in scoped_source
        or "pretransposed_codeword_stream" in scoped_source
        or "output_groups_x_route_blocks_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    uses_rejected_route_block_output_group_schedule = (
        "route_block_output_group_stream" in scoped_source
        or "route_blocks_x_output_groups_x_k_blocks_x_codeword_groups"
        in scoped_source
        or "route_block_output_group" in scoped_source
    )
    uses_rejected_scale_group_route_block_schedule = (
        "scale_group_route_block_reduce" in scoped_source
        or "scale_groups_x_route_blocks_x_k_blocks_x_output_tiles" in scoped_source
        or "route_block_partial" in scoped_source
    )
    uses_rejected_expert_kblock_scale_slot_schedule = (
        "expert_kblock_scale_slot_stream" in scoped_source
        or "experts_x_k_blocks_x_scale_groups_x_route_tiles_x_output_tiles"
        in scoped_source
    )
    uses_rejected_token_route_schedule = (
        "token_route_output_stripe" in scoped_source
        or "tokens_x_route_slots_x_output_stripes_x_kblock_stages" in scoped_source
    )
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel
    reconstructs_e8p_per_fragment = (
        "mlx_vq_decode_e8p" in kernel and "fragment" in kernel
    )

    passes_contract = (
        kernel_present
        and primitive_binding_present
        and dispatch_grid_ok
        and preserves_k_block_axis
        and preserves_output_group_axis
        and preserves_route_block_axis
        and preserves_codeword_group_axis
        and uses_kblock_output_group_route_fusion
        and writes_route_slots_after_fusion
        and streams_compressed_codeword_tiles
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and not uses_rejected_output_group_pretransposed_schedule
        and not uses_rejected_route_block_output_group_schedule
        and not uses_rejected_scale_group_route_block_schedule
        and not uses_rejected_expert_kblock_scale_slot_schedule
        and not uses_rejected_token_route_schedule
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
        and not reconstructs_e8p_per_fragment
    )

    if passes_contract:
        decision = "kblock_output_group_route_fused_stream_source_guardrail_present"
        required_next_features = [
            "prove_kblock_output_group_route_fused_stream_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_kblock_output_group_route_fused_stream_source"
        required_next_features = [
            "add_kblock_output_group_route_fused_stream_native_source_guardrail",
            "bind_kblock_output_group_route_fused_stream_primitive",
            "preserve_kblock_output_group_route_block_codeword_group_axes",
            "fuse_route_blocks_under_each_kblock_output_group_unit",
            "write_route_slots_after_kblock_output_group_route_fusion",
            "stream_compressed_uint16_codeword_tiles_and_scale_slots",
            "avoid_output_group_pretransposed_codeword_stream_schedule",
            "avoid_route_block_output_group_stream_schedule",
            "avoid_decoded_dense_rhs",
        ]
        rejected_next_steps = [
            "do_not_benchmark_kblock_output_group_route_fused_stream_before_source_guardrail",
            "do_not_retime_output_group_pretransposed_codeword_stream",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "k_blocks_x_output_groups_x_route_blocks_x_codeword_groups"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_k_block_axis": preserves_k_block_axis,
        "preserves_output_group_axis": preserves_output_group_axis,
        "preserves_route_block_axis": preserves_route_block_axis,
        "preserves_codeword_group_axis": preserves_codeword_group_axis,
        "uses_kblock_output_group_route_fusion": (
            uses_kblock_output_group_route_fusion
        ),
        "writes_route_slots_after_fusion": writes_route_slots_after_fusion,
        "streams_compressed_codeword_tiles": streams_compressed_codeword_tiles,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "uses_rejected_output_group_pretransposed_schedule": (
            uses_rejected_output_group_pretransposed_schedule
        ),
        "uses_rejected_route_block_output_group_schedule": (
            uses_rejected_route_block_output_group_schedule
        ),
        "uses_rejected_scale_group_route_block_schedule": (
            uses_rejected_scale_group_route_block_schedule
        ),
        "uses_rejected_expert_kblock_scale_slot_schedule": (
            uses_rejected_expert_kblock_scale_slot_schedule
        ),
        "uses_rejected_token_route_schedule": uses_rejected_token_route_schedule,
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "reconstructs_e8p_per_fragment": reconstructs_e8p_per_fragment,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_route_tile_output_swizzle_stream_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _ROUTE_TILE_OUTPUT_SWIZZLE_STREAM_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_name = "e8p_route_tile_output_swizzle_stream_rhs_sorted_matmul"
    primitive_binding_present = (
        primitive_name in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(primitive_source, primitive_name)
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    dispatch_grid_ok = (
        "route_tiles_x_output_swizzles_x_k_blocks_x_codeword_groups"
        in primitive_source
        or all(
            token in primitive_source
            for token in ("route_tile", "output_swizzle", "k_block", "codeword_group")
        )
    )
    preserves_route_tile_axis = "route_tile" in scoped_source
    preserves_output_swizzle_axis = "output_swizzle" in scoped_source
    preserves_k_block_axis = "k_block" in scoped_source or "kblock" in scoped_source
    preserves_codeword_group_axis = "codeword_group" in scoped_source
    uses_route_tile_output_swizzle_streams = (
        "route_tile_output_swizzle_stream" in scoped_source
        or "route_tile_output_swizzle" in scoped_source
    )
    writes_route_slots_after_output_swizzle = (
        "route_slot" in scoped_source
        and (
            "output_swizzle_accumulator" in scoped_source
            or "route_tile_output_swizzle_accumulator" in scoped_source
            or "after_output_swizzle" in scoped_source
        )
    )
    streams_compressed_codeword_tiles = (
        (
            "codeword_tiles" in kernel
            or "code_tiles" in kernel
            or "compressed_codeword" in kernel
        )
        and (
            "uint16_t" in kernel
            or "ushort" in kernel
            or "uint16" in kernel
            or "compressed_codeword" in kernel
        )
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    uses_rejected_kblock_output_group_route_fused_schedule = (
        "kblock_output_group_route_fused" in scoped_source
        or "k_blocks_x_output_groups_x_route_blocks_x_codeword_groups"
        in scoped_source
        or "kblock_output_group_route" in scoped_source
    )
    uses_rejected_output_group_pretransposed_schedule = (
        "output_group_pretransposed" in scoped_source
        or "pretransposed_codeword_stream" in scoped_source
        or "output_groups_x_route_blocks_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    uses_rejected_route_block_output_group_schedule = (
        "route_block_output_group_stream" in scoped_source
        or "route_blocks_x_output_groups_x_k_blocks_x_codeword_groups"
        in scoped_source
        or "route_block_output_group" in scoped_source
    )
    uses_rejected_scale_group_route_block_schedule = (
        "scale_group_route_block_reduce" in scoped_source
        or "scale_groups_x_route_blocks_x_k_blocks_x_output_tiles" in scoped_source
        or "route_block_partial" in scoped_source
    )
    uses_rejected_token_route_schedule = (
        "token_route_output_stripe" in scoped_source
        or "tokens_x_route_slots_x_output_stripes_x_kblock_stages" in scoped_source
    )
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel
    reconstructs_e8p_per_fragment = (
        "mlx_vq_decode_e8p" in kernel and "fragment" in kernel
    )

    passes_contract = (
        kernel_present
        and primitive_binding_present
        and dispatch_grid_ok
        and preserves_route_tile_axis
        and preserves_output_swizzle_axis
        and preserves_k_block_axis
        and preserves_codeword_group_axis
        and uses_route_tile_output_swizzle_streams
        and writes_route_slots_after_output_swizzle
        and streams_compressed_codeword_tiles
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and not uses_rejected_kblock_output_group_route_fused_schedule
        and not uses_rejected_output_group_pretransposed_schedule
        and not uses_rejected_route_block_output_group_schedule
        and not uses_rejected_scale_group_route_block_schedule
        and not uses_rejected_token_route_schedule
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
        and not reconstructs_e8p_per_fragment
    )

    if passes_contract:
        decision = "route_tile_output_swizzle_stream_source_guardrail_present"
        required_next_features = [
            "prove_route_tile_output_swizzle_stream_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_route_tile_output_swizzle_stream_source"
        required_next_features = [
            "add_route_tile_output_swizzle_stream_native_source_guardrail",
            "bind_route_tile_output_swizzle_stream_primitive",
            "preserve_route_tile_output_swizzle_kblock_codeword_group_axes",
            "write_route_slots_after_route_tile_output_swizzle_accumulation",
            "stream_compressed_uint16_codeword_tiles_and_scale_slots",
            "avoid_kblock_output_group_route_fused_stream_schedule",
            "avoid_decoded_dense_rhs",
        ]
        rejected_next_steps = [
            "do_not_benchmark_route_tile_output_swizzle_stream_before_source_guardrail",
            "do_not_retime_kblock_output_group_route_fused_stream",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "route_tiles_x_output_swizzles_x_k_blocks_x_codeword_groups"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_route_tile_axis": preserves_route_tile_axis,
        "preserves_output_swizzle_axis": preserves_output_swizzle_axis,
        "preserves_k_block_axis": preserves_k_block_axis,
        "preserves_codeword_group_axis": preserves_codeword_group_axis,
        "uses_route_tile_output_swizzle_streams": (
            uses_route_tile_output_swizzle_streams
        ),
        "writes_route_slots_after_output_swizzle": (
            writes_route_slots_after_output_swizzle
        ),
        "streams_compressed_codeword_tiles": streams_compressed_codeword_tiles,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "uses_rejected_kblock_output_group_route_fused_schedule": (
            uses_rejected_kblock_output_group_route_fused_schedule
        ),
        "uses_rejected_output_group_pretransposed_schedule": (
            uses_rejected_output_group_pretransposed_schedule
        ),
        "uses_rejected_route_block_output_group_schedule": (
            uses_rejected_route_block_output_group_schedule
        ),
        "uses_rejected_scale_group_route_block_schedule": (
            uses_rejected_scale_group_route_block_schedule
        ),
        "uses_rejected_token_route_schedule": uses_rejected_token_route_schedule,
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "reconstructs_e8p_per_fragment": reconstructs_e8p_per_fragment,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_token_topk_output_tile_stream_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _TOKEN_TOPK_OUTPUT_TILE_STREAM_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_name = "e8p_token_topk_output_tile_stream_rhs_sorted_matmul"
    primitive_binding_present = (
        primitive_name in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(primitive_source, primitive_name)
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    dispatch_grid_ok = (
        "tokens_x_topk_slots_x_output_tiles_x_k_blocks_x_codeword_groups"
        in primitive_source
        or all(
            token in primitive_source
            for token in (
                "token",
                "topk_slot",
                "output_tile",
                "k_block",
                "codeword_group",
            )
        )
    )
    preserves_token_axis = "token" in scoped_source
    preserves_topk_slot_axis = "topk_slot" in scoped_source or "top_k_slot" in scoped_source
    preserves_output_tile_axis = "output_tile" in scoped_source
    preserves_k_block_axis = "k_block" in scoped_source or "kblock" in scoped_source
    preserves_codeword_group_axis = "codeword_group" in scoped_source
    uses_token_topk_output_tile_streams = (
        "token_topk_output_tile_stream" in scoped_source
        or "token_topk_output_tile" in scoped_source
    )
    preserves_q2_token_topk_output_shape = (
        "tokens_x_topk_slots_x_output_tiles" in scoped_source
        or "q2_token_topk_output" in scoped_source
        or "token_topk_output_accumulator" in scoped_source
    )
    streams_compressed_codeword_tiles = (
        (
            "codeword_tiles" in kernel
            or "code_tiles" in kernel
            or "compressed_codeword" in kernel
        )
        and (
            "uint16_t" in kernel
            or "ushort" in kernel
            or "uint16" in kernel
            or "compressed_codeword" in kernel
        )
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    uses_rejected_route_tile_output_swizzle_schedule = (
        "route_tile_output_swizzle" in scoped_source
        or "route_tiles_x_output_swizzles_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    uses_rejected_kblock_output_group_route_fused_schedule = (
        "kblock_output_group_route_fused" in scoped_source
        or "k_blocks_x_output_groups_x_route_blocks_x_codeword_groups"
        in scoped_source
        or "kblock_output_group_route" in scoped_source
    )
    uses_rejected_output_group_pretransposed_schedule = (
        "output_group_pretransposed" in scoped_source
        or "pretransposed_codeword_stream" in scoped_source
        or "output_groups_x_route_blocks_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    uses_rejected_route_block_output_group_schedule = (
        "route_block_output_group_stream" in scoped_source
        or "route_blocks_x_output_groups_x_k_blocks_x_codeword_groups"
        in scoped_source
        or "route_block_output_group" in scoped_source
    )
    uses_rejected_scale_group_route_block_schedule = (
        "scale_group_route_block_reduce" in scoped_source
        or "scale_groups_x_route_blocks_x_k_blocks_x_output_tiles" in scoped_source
        or "route_block_partial" in scoped_source
    )
    uses_rejected_token_route_schedule = (
        "token_route_output_stripe" in scoped_source
        or "tokens_x_route_slots_x_output_stripes_x_kblock_stages" in scoped_source
    )
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel
    reconstructs_e8p_per_fragment = (
        "mlx_vq_decode_e8p" in kernel and "fragment" in kernel
    )

    passes_contract = (
        kernel_present
        and primitive_binding_present
        and dispatch_grid_ok
        and preserves_token_axis
        and preserves_topk_slot_axis
        and preserves_output_tile_axis
        and preserves_k_block_axis
        and preserves_codeword_group_axis
        and uses_token_topk_output_tile_streams
        and preserves_q2_token_topk_output_shape
        and streams_compressed_codeword_tiles
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and not uses_rejected_route_tile_output_swizzle_schedule
        and not uses_rejected_kblock_output_group_route_fused_schedule
        and not uses_rejected_output_group_pretransposed_schedule
        and not uses_rejected_route_block_output_group_schedule
        and not uses_rejected_scale_group_route_block_schedule
        and not uses_rejected_token_route_schedule
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
        and not reconstructs_e8p_per_fragment
    )

    if passes_contract:
        decision = "token_topk_output_tile_stream_source_guardrail_present"
        required_next_features = [
            "prove_token_topk_output_tile_stream_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_token_topk_output_tile_stream_source"
        required_next_features = [
            "add_token_topk_output_tile_stream_native_source_guardrail",
            "bind_token_topk_output_tile_stream_primitive",
            "preserve_tokens_topk_slots_output_tiles_kblocks_codeword_group_axes",
            "preserve_q2_tokens_x_topk_slots_x_output_tiles_writeback",
            "stream_compressed_uint16_codeword_tiles_and_scale_slots",
            "avoid_route_tile_output_swizzle_stream_schedule",
            "avoid_decoded_dense_rhs",
        ]
        rejected_next_steps = [
            "do_not_benchmark_token_topk_output_tile_stream_before_source_guardrail",
            "do_not_retime_route_tile_output_swizzle_stream",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "tokens_x_topk_slots_x_output_tiles_x_k_blocks_x_codeword_groups"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_token_axis": preserves_token_axis,
        "preserves_topk_slot_axis": preserves_topk_slot_axis,
        "preserves_output_tile_axis": preserves_output_tile_axis,
        "preserves_k_block_axis": preserves_k_block_axis,
        "preserves_codeword_group_axis": preserves_codeword_group_axis,
        "uses_token_topk_output_tile_streams": uses_token_topk_output_tile_streams,
        "preserves_q2_token_topk_output_shape": preserves_q2_token_topk_output_shape,
        "streams_compressed_codeword_tiles": streams_compressed_codeword_tiles,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "uses_rejected_route_tile_output_swizzle_schedule": (
            uses_rejected_route_tile_output_swizzle_schedule
        ),
        "uses_rejected_kblock_output_group_route_fused_schedule": (
            uses_rejected_kblock_output_group_route_fused_schedule
        ),
        "uses_rejected_output_group_pretransposed_schedule": (
            uses_rejected_output_group_pretransposed_schedule
        ),
        "uses_rejected_route_block_output_group_schedule": (
            uses_rejected_route_block_output_group_schedule
        ),
        "uses_rejected_scale_group_route_block_schedule": (
            uses_rejected_scale_group_route_block_schedule
        ),
        "uses_rejected_token_route_schedule": uses_rejected_token_route_schedule,
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "reconstructs_e8p_per_fragment": reconstructs_e8p_per_fragment,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_token_block_output_group_stream_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _TOKEN_BLOCK_OUTPUT_GROUP_STREAM_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_name = "e8p_token_block_output_group_stream_rhs_sorted_matmul"
    primitive_binding_present = (
        primitive_name in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(primitive_source, primitive_name)
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    dispatch_grid_ok = (
        "token_blocks_x_output_groups_x_topk_slots_x_k_blocks_x_codeword_groups"
        in primitive_source
        or all(
            token in primitive_source
            for token in (
                "token_block",
                "output_group",
                "topk_slot",
                "k_block",
                "codeword_group",
            )
        )
    )
    preserves_token_block_axis = "token_block" in scoped_source
    preserves_output_group_axis = "output_group" in scoped_source
    preserves_topk_slot_axis = "topk_slot" in scoped_source or "top_k_slot" in scoped_source
    preserves_k_block_axis = "k_block" in scoped_source or "kblock" in scoped_source
    preserves_codeword_group_axis = "codeword_group" in scoped_source
    uses_token_block_output_group_streams = (
        "token_block_output_group_stream" in scoped_source
        or "token_block_output_group" in scoped_source
    )
    preserves_token_topk_output_writeback = (
        "token_block_topk_output_writeback" in scoped_source
        or "token_topk_output_writeback" in scoped_source
        or "token_block_output_group_accumulator" in scoped_source
    )
    streams_compressed_codeword_tiles = (
        (
            "codeword_tiles" in kernel
            or "code_tiles" in kernel
            or "compressed_codeword" in kernel
        )
        and (
            "uint16_t" in kernel
            or "ushort" in kernel
            or "uint16" in kernel
            or "compressed_codeword" in kernel
        )
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    uses_rejected_token_topk_output_tile_schedule = (
        "token_topk_output_tile_stream" in scoped_source
        or "tokens_x_topk_slots_x_output_tiles_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    uses_rejected_route_tile_output_swizzle_schedule = (
        "route_tile_output_swizzle" in scoped_source
        or "route_tiles_x_output_swizzles_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    uses_rejected_kblock_output_group_route_fused_schedule = (
        "kblock_output_group_route_fused" in scoped_source
        or "k_blocks_x_output_groups_x_route_blocks_x_codeword_groups"
        in scoped_source
        or "kblock_output_group_route" in scoped_source
    )
    uses_rejected_output_group_pretransposed_schedule = (
        "output_group_pretransposed" in scoped_source
        or "pretransposed_codeword_stream" in scoped_source
        or "output_groups_x_route_blocks_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    uses_rejected_route_block_output_group_schedule = (
        "route_block_output_group_stream" in scoped_source
        or "route_blocks_x_output_groups_x_k_blocks_x_codeword_groups"
        in scoped_source
        or "route_block_output_group" in scoped_source
    )
    uses_rejected_scale_group_route_block_schedule = (
        "scale_group_route_block_reduce" in scoped_source
        or "scale_groups_x_route_blocks_x_k_blocks_x_output_tiles" in scoped_source
        or "route_block_partial" in scoped_source
    )
    uses_rejected_token_route_schedule = (
        "token_route_output_stripe" in scoped_source
        or "tokens_x_route_slots_x_output_stripes_x_kblock_stages" in scoped_source
    )
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel
    reconstructs_e8p_per_fragment = (
        "mlx_vq_decode_e8p" in kernel and "fragment" in kernel
    )

    passes_contract = (
        kernel_present
        and primitive_binding_present
        and dispatch_grid_ok
        and preserves_token_block_axis
        and preserves_output_group_axis
        and preserves_topk_slot_axis
        and preserves_k_block_axis
        and preserves_codeword_group_axis
        and uses_token_block_output_group_streams
        and preserves_token_topk_output_writeback
        and streams_compressed_codeword_tiles
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and not uses_rejected_token_topk_output_tile_schedule
        and not uses_rejected_route_tile_output_swizzle_schedule
        and not uses_rejected_kblock_output_group_route_fused_schedule
        and not uses_rejected_output_group_pretransposed_schedule
        and not uses_rejected_route_block_output_group_schedule
        and not uses_rejected_scale_group_route_block_schedule
        and not uses_rejected_token_route_schedule
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
        and not reconstructs_e8p_per_fragment
    )

    if passes_contract:
        decision = "token_block_output_group_stream_source_guardrail_present"
        required_next_features = [
            "prove_token_block_output_group_stream_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_token_block_output_group_stream_source"
        required_next_features = [
            "add_token_block_output_group_stream_native_source_guardrail",
            "bind_token_block_output_group_stream_primitive",
            "preserve_token_blocks_output_groups_topk_slots_kblocks_codeword_group_axes",
            "preserve_token_topk_output_writeback",
            "stream_compressed_uint16_codeword_tiles_and_scale_slots",
            "avoid_token_topk_output_tile_stream_schedule",
            "avoid_decoded_dense_rhs",
        ]
        rejected_next_steps = [
            "do_not_benchmark_token_block_output_group_stream_before_source_guardrail",
            "do_not_retime_token_topk_output_tile_stream",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "token_blocks_x_output_groups_x_topk_slots_x_k_blocks_x_codeword_groups"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_token_block_axis": preserves_token_block_axis,
        "preserves_output_group_axis": preserves_output_group_axis,
        "preserves_topk_slot_axis": preserves_topk_slot_axis,
        "preserves_k_block_axis": preserves_k_block_axis,
        "preserves_codeword_group_axis": preserves_codeword_group_axis,
        "uses_token_block_output_group_streams": (
            uses_token_block_output_group_streams
        ),
        "preserves_token_topk_output_writeback": (
            preserves_token_topk_output_writeback
        ),
        "streams_compressed_codeword_tiles": streams_compressed_codeword_tiles,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "uses_rejected_token_topk_output_tile_schedule": (
            uses_rejected_token_topk_output_tile_schedule
        ),
        "uses_rejected_route_tile_output_swizzle_schedule": (
            uses_rejected_route_tile_output_swizzle_schedule
        ),
        "uses_rejected_kblock_output_group_route_fused_schedule": (
            uses_rejected_kblock_output_group_route_fused_schedule
        ),
        "uses_rejected_output_group_pretransposed_schedule": (
            uses_rejected_output_group_pretransposed_schedule
        ),
        "uses_rejected_route_block_output_group_schedule": (
            uses_rejected_route_block_output_group_schedule
        ),
        "uses_rejected_scale_group_route_block_schedule": (
            uses_rejected_scale_group_route_block_schedule
        ),
        "uses_rejected_token_route_schedule": uses_rejected_token_route_schedule,
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "reconstructs_e8p_per_fragment": reconstructs_e8p_per_fragment,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_token_output_stripe_group_stream_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _TOKEN_OUTPUT_STRIPE_GROUP_STREAM_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_name = "e8p_token_output_stripe_group_stream_rhs_sorted_matmul"
    primitive_binding_present = (
        primitive_name in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(primitive_source, primitive_name)
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    dispatch_grid_ok = (
        "tokens_x_output_stripes_x_topk_groups_x_k_blocks_x_codeword_groups"
        in primitive_source
        or all(
            token in primitive_source
            for token in (
                "token",
                "output_stripe",
                "topk_group",
                "k_block",
                "codeword_group",
            )
        )
    )
    preserves_token_axis = "token" in scoped_source
    preserves_output_stripe_axis = "output_stripe" in scoped_source
    preserves_topk_group_axis = (
        "topk_group" in scoped_source or "top_k_group" in scoped_source
    )
    preserves_k_block_axis = "k_block" in scoped_source or "kblock" in scoped_source
    preserves_codeword_group_axis = "codeword_group" in scoped_source
    uses_token_output_stripe_group_streams = (
        "token_output_stripe_group_stream" in scoped_source
        or "token_output_stripe_group" in scoped_source
    )
    preserves_q2_token_topk_output_shape = (
        "tokens_x_output_stripes_x_topk_groups" in scoped_source
        or "q2_token_topk_output" in scoped_source
        or "token_output_stripe_accumulator" in scoped_source
        or "token_output_stripe_topk_writeback" in scoped_source
    )
    streams_compressed_codeword_tiles = (
        (
            "codeword_tiles" in kernel
            or "code_tiles" in kernel
            or "compressed_codeword" in kernel
        )
        and (
            "uint16_t" in kernel
            or "ushort" in kernel
            or "uint16" in kernel
            or "compressed_codeword" in kernel
        )
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    uses_rejected_token_block_output_group_schedule = (
        "token_block_output_group_stream" in scoped_source
        or "token_blocks_x_output_groups_x_topk_slots_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    uses_rejected_token_topk_output_tile_schedule = (
        "token_topk_output_tile_stream" in scoped_source
        or "tokens_x_topk_slots_x_output_tiles_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    uses_rejected_route_tile_output_swizzle_schedule = (
        "route_tile_output_swizzle" in scoped_source
        or "route_tiles_x_output_swizzles_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    uses_rejected_kblock_output_group_route_fused_schedule = (
        "kblock_output_group_route_fused" in scoped_source
        or "k_blocks_x_output_groups_x_route_blocks_x_codeword_groups"
        in scoped_source
        or "kblock_output_group_route" in scoped_source
    )
    uses_rejected_output_group_pretransposed_schedule = (
        "output_group_pretransposed" in scoped_source
        or "pretransposed_codeword_stream" in scoped_source
        or "output_groups_x_route_blocks_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    uses_rejected_route_block_output_group_schedule = (
        "route_block_output_group_stream" in scoped_source
        or "route_blocks_x_output_groups_x_k_blocks_x_codeword_groups"
        in scoped_source
        or "route_block_output_group" in scoped_source
    )
    uses_rejected_scale_group_route_block_schedule = (
        "scale_group_route_block_reduce" in scoped_source
        or "scale_groups_x_route_blocks_x_k_blocks_x_output_tiles" in scoped_source
        or "route_block_partial" in scoped_source
    )
    uses_rejected_token_route_schedule = (
        "token_route_output_stripe" in scoped_source
        or "tokens_x_route_slots_x_output_stripes_x_kblock_stages" in scoped_source
    )
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel
    reconstructs_e8p_per_fragment = (
        "mlx_vq_decode_e8p" in kernel and "fragment" in kernel
    )

    passes_contract = (
        kernel_present
        and primitive_binding_present
        and dispatch_grid_ok
        and preserves_token_axis
        and preserves_output_stripe_axis
        and preserves_topk_group_axis
        and preserves_k_block_axis
        and preserves_codeword_group_axis
        and uses_token_output_stripe_group_streams
        and preserves_q2_token_topk_output_shape
        and streams_compressed_codeword_tiles
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and not uses_rejected_token_block_output_group_schedule
        and not uses_rejected_token_topk_output_tile_schedule
        and not uses_rejected_route_tile_output_swizzle_schedule
        and not uses_rejected_kblock_output_group_route_fused_schedule
        and not uses_rejected_output_group_pretransposed_schedule
        and not uses_rejected_route_block_output_group_schedule
        and not uses_rejected_scale_group_route_block_schedule
        and not uses_rejected_token_route_schedule
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
        and not reconstructs_e8p_per_fragment
    )

    if passes_contract:
        decision = "token_output_stripe_group_stream_source_guardrail_present"
        required_next_features = [
            "prove_token_output_stripe_group_stream_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_token_output_stripe_group_stream_source"
        required_next_features = [
            "add_token_output_stripe_group_stream_native_source_guardrail",
            "bind_token_output_stripe_group_stream_primitive",
            "preserve_tokens_output_stripes_topk_groups_kblocks_codeword_group_axes",
            "preserve_q2_tokens_x_topk_groups_x_output_stripes_writeback",
            "stream_compressed_uint16_codeword_tiles_and_scale_slots",
            "avoid_token_block_output_group_stream_schedule",
            "avoid_decoded_dense_rhs",
        ]
        rejected_next_steps = [
            "do_not_benchmark_token_output_stripe_group_stream_before_source_guardrail",
            "do_not_retime_token_block_output_group_stream",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "tokens_x_output_stripes_x_topk_groups_x_k_blocks_x_codeword_groups"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_token_axis": preserves_token_axis,
        "preserves_output_stripe_axis": preserves_output_stripe_axis,
        "preserves_topk_group_axis": preserves_topk_group_axis,
        "preserves_k_block_axis": preserves_k_block_axis,
        "preserves_codeword_group_axis": preserves_codeword_group_axis,
        "uses_token_output_stripe_group_streams": (
            uses_token_output_stripe_group_streams
        ),
        "preserves_q2_token_topk_output_shape": (
            preserves_q2_token_topk_output_shape
        ),
        "streams_compressed_codeword_tiles": streams_compressed_codeword_tiles,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "uses_rejected_token_block_output_group_schedule": (
            uses_rejected_token_block_output_group_schedule
        ),
        "uses_rejected_token_topk_output_tile_schedule": (
            uses_rejected_token_topk_output_tile_schedule
        ),
        "uses_rejected_route_tile_output_swizzle_schedule": (
            uses_rejected_route_tile_output_swizzle_schedule
        ),
        "uses_rejected_kblock_output_group_route_fused_schedule": (
            uses_rejected_kblock_output_group_route_fused_schedule
        ),
        "uses_rejected_output_group_pretransposed_schedule": (
            uses_rejected_output_group_pretransposed_schedule
        ),
        "uses_rejected_route_block_output_group_schedule": (
            uses_rejected_route_block_output_group_schedule
        ),
        "uses_rejected_scale_group_route_block_schedule": (
            uses_rejected_scale_group_route_block_schedule
        ),
        "uses_rejected_token_route_schedule": uses_rejected_token_route_schedule,
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "reconstructs_e8p_per_fragment": reconstructs_e8p_per_fragment,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_token_expert_output_block_stream_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _TOKEN_EXPERT_OUTPUT_BLOCK_STREAM_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_name = "e8p_token_expert_output_block_stream_rhs_sorted_matmul"
    primitive_binding_present = (
        primitive_name in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(primitive_source, primitive_name)
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    dispatch_grid_ok = (
        "tokens_x_active_experts_x_output_blocks_x_k_blocks_x_codeword_groups"
        in primitive_source
        or all(
            token in primitive_source
            for token in (
                "token",
                "active_expert",
                "output_block",
                "k_block",
                "codeword_group",
            )
        )
    )
    preserves_token_axis = "token" in scoped_source
    preserves_active_expert_axis = (
        "active_expert" in scoped_source or "expert_id" in scoped_source
    )
    preserves_output_block_axis = "output_block" in scoped_source
    preserves_k_block_axis = "k_block" in scoped_source or "kblock" in scoped_source
    preserves_codeword_group_axis = "codeword_group" in scoped_source
    uses_token_expert_output_block_streams = (
        "token_expert_output_block_stream" in scoped_source
        or "token_expert_output_block" in scoped_source
    )
    reduces_routes_before_q2_writeback = (
        "token_expert_output_block_accumulator" in scoped_source
        or "reduce_routes_before_q2_writeback" in scoped_source
        or "route_reduction_before_q2_writeback" in scoped_source
        or "token_expert_output_block_topk_writeback" in scoped_source
    )
    streams_compressed_codeword_tiles = (
        (
            "codeword_tiles" in kernel
            or "code_tiles" in kernel
            or "compressed_codeword" in kernel
        )
        and (
            "uint16_t" in kernel
            or "ushort" in kernel
            or "uint16" in kernel
            or "compressed_codeword" in kernel
        )
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    uses_rejected_token_output_stripe_group_schedule = (
        "token_output_stripe_group_stream" in scoped_source
        or "tokens_x_output_stripes_x_topk_groups_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    uses_rejected_token_block_output_group_schedule = (
        "token_block_output_group_stream" in scoped_source
        or "token_blocks_x_output_groups_x_topk_slots_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    uses_rejected_token_topk_output_tile_schedule = (
        "token_topk_output_tile_stream" in scoped_source
        or "tokens_x_topk_slots_x_output_tiles_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel
    reconstructs_e8p_per_fragment = (
        "mlx_vq_decode_e8p" in kernel and "fragment" in kernel
    )

    passes_contract = (
        kernel_present
        and primitive_binding_present
        and dispatch_grid_ok
        and preserves_token_axis
        and preserves_active_expert_axis
        and preserves_output_block_axis
        and preserves_k_block_axis
        and preserves_codeword_group_axis
        and uses_token_expert_output_block_streams
        and reduces_routes_before_q2_writeback
        and streams_compressed_codeword_tiles
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and not uses_rejected_token_output_stripe_group_schedule
        and not uses_rejected_token_block_output_group_schedule
        and not uses_rejected_token_topk_output_tile_schedule
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
        and not reconstructs_e8p_per_fragment
    )

    if passes_contract:
        decision = "token_expert_output_block_stream_source_guardrail_present"
        required_next_features = [
            "prove_token_expert_output_block_stream_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_token_expert_output_block_stream_source"
        required_next_features = [
            "add_token_expert_output_block_stream_native_source_guardrail",
            "bind_token_expert_output_block_stream_primitive",
            "preserve_tokens_active_experts_output_blocks_kblocks_codeword_group_axes",
            "reduce_routes_before_q2_token_topk_output_writeback",
            "stream_compressed_uint16_codeword_tiles_and_scale_slots",
            "avoid_token_output_stripe_group_stream_schedule",
            "avoid_decoded_dense_rhs",
        ]
        rejected_next_steps = [
            "do_not_benchmark_token_expert_output_block_stream_before_source_guardrail",
            "do_not_retime_token_output_stripe_group_stream",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "tokens_x_active_experts_x_output_blocks_x_k_blocks_x_codeword_groups"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_token_axis": preserves_token_axis,
        "preserves_active_expert_axis": preserves_active_expert_axis,
        "preserves_output_block_axis": preserves_output_block_axis,
        "preserves_k_block_axis": preserves_k_block_axis,
        "preserves_codeword_group_axis": preserves_codeword_group_axis,
        "uses_token_expert_output_block_streams": (
            uses_token_expert_output_block_streams
        ),
        "reduces_routes_before_q2_writeback": reduces_routes_before_q2_writeback,
        "streams_compressed_codeword_tiles": streams_compressed_codeword_tiles,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "uses_rejected_token_output_stripe_group_schedule": (
            uses_rejected_token_output_stripe_group_schedule
        ),
        "uses_rejected_token_block_output_group_schedule": (
            uses_rejected_token_block_output_group_schedule
        ),
        "uses_rejected_token_topk_output_tile_schedule": (
            uses_rejected_token_topk_output_tile_schedule
        ),
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "reconstructs_e8p_per_fragment": reconstructs_e8p_per_fragment,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_token_pair_kblock_accumulator_stream_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _TOKEN_PAIR_KBLOCK_ACCUMULATOR_STREAM_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_name = "e8p_token_pair_kblock_accumulator_stream_rhs_sorted_matmul"
    primitive_binding_present = (
        primitive_name in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(primitive_source, primitive_name)
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    dispatch_grid_ok = (
        "token_pairs_x_active_experts_x_k_blocks_x_output_blocks_x_codeword_groups"
        in primitive_source
        or all(
            token in primitive_source
            for token in (
                "token_pair",
                "active_expert",
                "k_block",
                "output_block",
                "codeword_group",
            )
        )
    )
    preserves_token_pair_axis = (
        "token_pair" in scoped_source or "token_pairs" in scoped_source
    )
    preserves_active_expert_axis = (
        "active_expert" in scoped_source or "expert_id" in scoped_source
    )
    preserves_k_block_axis = "k_block" in scoped_source or "kblock" in scoped_source
    preserves_output_block_axis = "output_block" in scoped_source
    preserves_codeword_group_axis = "codeword_group" in scoped_source
    uses_token_pair_kblock_accumulator_streams = (
        "token_pair_kblock_accumulator_stream" in scoped_source
        or "token_pair_kblock_accumulator" in scoped_source
        or "token_pair_kblock" in scoped_source
    )
    reuses_lhs_across_token_pairs = (
        "reuse_lhs_across_adjacent_tokens" in scoped_source
        or "token_pair_lhs" in scoped_source
        or "adjacent_token" in scoped_source
    )
    preserves_exact_token_topk_scatter = (
        "token_pair_topk_scatter" in scoped_source
        or "pair_scatter" in scoped_source
        or "exact_token_topk" in scoped_source
        or "exact_token_topk_scatter" in scoped_source
    )
    streams_compressed_codeword_tiles = (
        (
            "codeword_tiles" in kernel
            or "code_tiles" in kernel
            or "compressed_codeword" in kernel
        )
        and (
            "uint16_t" in kernel
            or "ushort" in kernel
            or "uint16" in kernel
            or "compressed_codeword" in kernel
        )
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    uses_rejected_token_expert_output_block_schedule = (
        "token_expert_output_block_stream" in scoped_source
        or "tokens_x_active_experts_x_output_blocks_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    uses_rejected_token_output_stripe_group_schedule = (
        "token_output_stripe_group_stream" in scoped_source
        or "tokens_x_output_stripes_x_topk_groups_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    uses_rejected_token_block_output_group_schedule = (
        "token_block_output_group_stream" in scoped_source
        or "token_blocks_x_output_groups_x_topk_slots_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    uses_rejected_token_topk_output_tile_schedule = (
        "token_topk_output_tile_stream" in scoped_source
        or "tokens_x_topk_slots_x_output_tiles_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel
    reconstructs_e8p_per_fragment = (
        "mlx_vq_decode_e8p" in kernel and "fragment" in kernel
    )

    passes_contract = (
        kernel_present
        and primitive_binding_present
        and dispatch_grid_ok
        and preserves_token_pair_axis
        and preserves_active_expert_axis
        and preserves_k_block_axis
        and preserves_output_block_axis
        and preserves_codeword_group_axis
        and uses_token_pair_kblock_accumulator_streams
        and reuses_lhs_across_token_pairs
        and preserves_exact_token_topk_scatter
        and streams_compressed_codeword_tiles
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and not uses_rejected_token_expert_output_block_schedule
        and not uses_rejected_token_output_stripe_group_schedule
        and not uses_rejected_token_block_output_group_schedule
        and not uses_rejected_token_topk_output_tile_schedule
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
        and not reconstructs_e8p_per_fragment
    )

    if passes_contract:
        decision = "token_pair_kblock_accumulator_stream_source_guardrail_present"
        required_next_features = [
            "prove_token_pair_kblock_accumulator_stream_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_token_pair_kblock_accumulator_stream_source"
        required_next_features = [
            "add_token_pair_kblock_accumulator_stream_native_source_guardrail",
            "bind_token_pair_kblock_accumulator_stream_primitive",
            "preserve_token_pairs_active_experts_kblocks_output_blocks_codeword_group_axes",
            "reuse_lhs_across_adjacent_tokens_before_expert_output_scatter",
            "preserve_exact_token_topk_scatter_after_pair_accumulation",
            "stream_compressed_uint16_codeword_tiles_and_scale_slots",
            "avoid_token_expert_output_block_stream_schedule",
            "avoid_decoded_dense_rhs",
        ]
        rejected_next_steps = [
            "do_not_benchmark_token_pair_kblock_accumulator_stream_before_source_guardrail",
            "do_not_retime_token_expert_output_block_stream",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "token_pairs_x_active_experts_x_k_blocks_x_output_blocks_x_codeword_groups"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_token_pair_axis": preserves_token_pair_axis,
        "preserves_active_expert_axis": preserves_active_expert_axis,
        "preserves_k_block_axis": preserves_k_block_axis,
        "preserves_output_block_axis": preserves_output_block_axis,
        "preserves_codeword_group_axis": preserves_codeword_group_axis,
        "uses_token_pair_kblock_accumulator_streams": (
            uses_token_pair_kblock_accumulator_streams
        ),
        "reuses_lhs_across_token_pairs": reuses_lhs_across_token_pairs,
        "preserves_exact_token_topk_scatter": preserves_exact_token_topk_scatter,
        "streams_compressed_codeword_tiles": streams_compressed_codeword_tiles,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "uses_rejected_token_expert_output_block_schedule": (
            uses_rejected_token_expert_output_block_schedule
        ),
        "uses_rejected_token_output_stripe_group_schedule": (
            uses_rejected_token_output_stripe_group_schedule
        ),
        "uses_rejected_token_block_output_group_schedule": (
            uses_rejected_token_block_output_group_schedule
        ),
        "uses_rejected_token_topk_output_tile_schedule": (
            uses_rejected_token_topk_output_tile_schedule
        ),
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "reconstructs_e8p_per_fragment": reconstructs_e8p_per_fragment,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_token_pair_output_group_stream_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _TOKEN_PAIR_OUTPUT_GROUP_STREAM_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_name = "e8p_token_pair_output_group_stream_rhs_sorted_matmul"
    primitive_binding_present = (
        primitive_name in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(primitive_source, primitive_name)
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    dispatch_grid_ok = (
        "token_pairs_x_output_groups_x_active_experts_x_k_blocks_x_codeword_groups"
        in primitive_source
        or all(
            token in primitive_source
            for token in (
                "token_pair",
                "output_group",
                "active_expert",
                "k_block",
                "codeword_group",
            )
        )
    )
    preserves_token_pair_axis = (
        "token_pair" in scoped_source or "token_pairs" in scoped_source
    )
    preserves_output_group_axis = "output_group" in scoped_source
    preserves_active_expert_axis = (
        "active_expert" in scoped_source or "expert_id" in scoped_source
    )
    preserves_k_block_axis = "k_block" in scoped_source or "kblock" in scoped_source
    preserves_codeword_group_axis = "codeword_group" in scoped_source
    uses_token_pair_output_group_streams = (
        "token_pair_output_group_stream" in scoped_source
        or "token_pair_output_group" in scoped_source
    )
    preserves_q2_token_topk_scatter = (
        "token_pair_output_group_topk_scatter" in scoped_source
        or "q2_scatter" in scoped_source
        or "exact_token_topk" in scoped_source
        or "exact_token_topk_scatter" in scoped_source
    )
    materializes_full_pair_expert_matrix = any(
        token in scoped_source
        for token in (
            "full_pair_expert_matrix",
            "pair_expert_matrix",
            "token_pair_expert_output_matrix",
        )
    )
    avoids_full_pair_expert_matrix = not materializes_full_pair_expert_matrix
    streams_compressed_codeword_tiles = (
        (
            "codeword_tiles" in kernel
            or "code_tiles" in kernel
            or "compressed_codeword" in kernel
        )
        and (
            "uint16_t" in kernel
            or "ushort" in kernel
            or "uint16" in kernel
            or "compressed_codeword" in kernel
        )
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    uses_rejected_token_pair_kblock_schedule = (
        "token_pair_kblock_accumulator_stream" in scoped_source
        or "token_pairs_x_active_experts_x_k_blocks_x_output_blocks_x_codeword_groups"
        in scoped_source
    )
    uses_rejected_token_expert_output_block_schedule = (
        "token_expert_output_block_stream" in scoped_source
        or "tokens_x_active_experts_x_output_blocks_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    uses_rejected_token_output_stripe_group_schedule = (
        "token_output_stripe_group_stream" in scoped_source
        or "tokens_x_output_stripes_x_topk_groups_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel
    reconstructs_e8p_per_fragment = (
        "mlx_vq_decode_e8p" in kernel and "fragment" in kernel
    )

    passes_contract = (
        kernel_present
        and primitive_binding_present
        and dispatch_grid_ok
        and preserves_token_pair_axis
        and preserves_output_group_axis
        and preserves_active_expert_axis
        and preserves_k_block_axis
        and preserves_codeword_group_axis
        and uses_token_pair_output_group_streams
        and preserves_q2_token_topk_scatter
        and avoids_full_pair_expert_matrix
        and streams_compressed_codeword_tiles
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and not uses_rejected_token_pair_kblock_schedule
        and not uses_rejected_token_expert_output_block_schedule
        and not uses_rejected_token_output_stripe_group_schedule
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
        and not reconstructs_e8p_per_fragment
    )

    if passes_contract:
        decision = "token_pair_output_group_stream_source_guardrail_present"
        required_next_features = [
            "prove_token_pair_output_group_stream_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_token_pair_output_group_stream_source"
        required_next_features = [
            "add_token_pair_output_group_stream_native_source_guardrail",
            "bind_token_pair_output_group_stream_primitive",
            "preserve_token_pairs_output_groups_active_experts_kblocks_codeword_group_axes",
            "preserve_q2_token_topk_scatter_without_full_pair_expert_matrix",
            "stream_compressed_uint16_codeword_tiles_and_scale_slots",
            "avoid_token_pair_kblock_accumulator_stream_schedule",
            "avoid_decoded_dense_rhs",
        ]
        rejected_next_steps = [
            "do_not_benchmark_token_pair_output_group_stream_before_source_guardrail",
            "do_not_retime_token_pair_kblock_accumulator_stream",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "token_pairs_x_output_groups_x_active_experts_x_k_blocks_x_codeword_groups"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_token_pair_axis": preserves_token_pair_axis,
        "preserves_output_group_axis": preserves_output_group_axis,
        "preserves_active_expert_axis": preserves_active_expert_axis,
        "preserves_k_block_axis": preserves_k_block_axis,
        "preserves_codeword_group_axis": preserves_codeword_group_axis,
        "uses_token_pair_output_group_streams": uses_token_pair_output_group_streams,
        "preserves_q2_token_topk_scatter": preserves_q2_token_topk_scatter,
        "avoids_full_pair_expert_matrix": avoids_full_pair_expert_matrix,
        "streams_compressed_codeword_tiles": streams_compressed_codeword_tiles,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "uses_rejected_token_pair_kblock_schedule": (
            uses_rejected_token_pair_kblock_schedule
        ),
        "uses_rejected_token_expert_output_block_schedule": (
            uses_rejected_token_expert_output_block_schedule
        ),
        "uses_rejected_token_output_stripe_group_schedule": (
            uses_rejected_token_output_stripe_group_schedule
        ),
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "reconstructs_e8p_per_fragment": reconstructs_e8p_per_fragment,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_token_pair_slot_topk_output_group_stream_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _TOKEN_PAIR_SLOT_TOPK_OUTPUT_GROUP_STREAM_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_name = "e8p_token_pair_slot_topk_output_group_stream_rhs_sorted_matmul"
    primitive_binding_present = (
        primitive_name in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(primitive_source, primitive_name)
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    dispatch_grid_ok = (
        "token_pairs_x_pair_slots_x_topk_slots_x_output_groups_x_k_blocks_x_codeword_groups"
        in primitive_source
        or all(
            token in primitive_source
            for token in (
                "token_pair",
                "pair_slot",
                "topk_slot",
                "output_group",
                "k_block",
                "codeword_group",
            )
        )
    )
    preserves_token_pair_axis = (
        "token_pair" in scoped_source or "token_pairs" in scoped_source
    )
    preserves_pair_slot_axis = "pair_slot" in scoped_source or "pair_slots" in scoped_source
    preserves_topk_slot_axis = "topk_slot" in scoped_source or "topk_slots" in scoped_source
    preserves_output_group_axis = "output_group" in scoped_source
    preserves_k_block_axis = "k_block" in scoped_source or "kblock" in scoped_source
    preserves_codeword_group_axis = "codeword_group" in scoped_source
    uses_token_pair_slot_topk_output_group_streams = (
        "token_pair_slot_topk_output_group_stream" in scoped_source
        or "token_pair_slot_topk_output_group" in scoped_source
    )
    preserves_q2_token_topk_scatter = (
        "token_pair_slot_topk_output_group_topk_scatter" in scoped_source
        or "pair_slot_topk_scatter" in scoped_source
        or "q2_scatter" in scoped_source
        or "exact_token_topk" in scoped_source
        or "exact_token_topk_scatter" in scoped_source
    )
    materializes_full_expert_axis = any(
        token in scoped_source
        for token in (
            "full_expert_axis",
            "full_pair_expert_matrix",
            "pair_expert_matrix",
            "token_pair_expert_output_matrix",
            "active_experts_x_output_groups",
        )
    )
    avoids_full_expert_axis = not materializes_full_expert_axis
    streams_compressed_codeword_tiles = (
        (
            "codeword_tiles" in kernel
            or "code_tiles" in kernel
            or "compressed_codeword" in kernel
        )
        and (
            "uint16_t" in kernel
            or "ushort" in kernel
            or "uint16" in kernel
            or "compressed_codeword" in kernel
        )
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    uses_rejected_token_pair_output_group_schedule = (
        "token_pair_output_group_stream" in scoped_source
        or "token_pairs_x_output_groups_x_active_experts_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    uses_rejected_token_pair_kblock_schedule = (
        "token_pair_kblock_accumulator_stream" in scoped_source
        or "token_pairs_x_active_experts_x_k_blocks_x_output_blocks_x_codeword_groups"
        in scoped_source
    )
    uses_rejected_token_expert_output_block_schedule = (
        "token_expert_output_block_stream" in scoped_source
        or "tokens_x_active_experts_x_output_blocks_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    uses_rejected_token_output_stripe_group_schedule = (
        "token_output_stripe_group_stream" in scoped_source
        or "tokens_x_output_stripes_x_topk_groups_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel
    reconstructs_e8p_per_fragment = (
        "mlx_vq_decode_e8p" in kernel and "fragment" in kernel
    )

    passes_contract = (
        kernel_present
        and primitive_binding_present
        and dispatch_grid_ok
        and preserves_token_pair_axis
        and preserves_pair_slot_axis
        and preserves_topk_slot_axis
        and preserves_output_group_axis
        and preserves_k_block_axis
        and preserves_codeword_group_axis
        and uses_token_pair_slot_topk_output_group_streams
        and preserves_q2_token_topk_scatter
        and avoids_full_expert_axis
        and streams_compressed_codeword_tiles
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and not uses_rejected_token_pair_output_group_schedule
        and not uses_rejected_token_pair_kblock_schedule
        and not uses_rejected_token_expert_output_block_schedule
        and not uses_rejected_token_output_stripe_group_schedule
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
        and not reconstructs_e8p_per_fragment
    )

    if passes_contract:
        decision = (
            "token_pair_slot_topk_output_group_stream_source_guardrail_present"
        )
        required_next_features = [
            "prove_token_pair_slot_topk_output_group_stream_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_token_pair_slot_topk_output_group_stream_source"
        required_next_features = [
            "add_token_pair_slot_topk_output_group_stream_native_source_guardrail",
            "bind_token_pair_slot_topk_output_group_stream_primitive",
            "preserve_token_pairs_pair_slots_topk_slots_output_groups_kblocks_codeword_group_axes",
            "preserve_q2_token_topk_scatter_without_full_expert_axis",
            "stream_compressed_uint16_codeword_tiles_and_scale_slots",
            "avoid_token_pair_output_group_stream_schedule",
            "avoid_token_pair_kblock_accumulator_stream_schedule",
            "avoid_decoded_dense_rhs",
        ]
        rejected_next_steps = [
            "do_not_benchmark_token_pair_slot_topk_output_group_stream_before_source_guardrail",
            "do_not_retime_token_pair_output_group_stream",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "token_pairs_x_pair_slots_x_topk_slots_x_output_groups_x_k_blocks_x_codeword_groups"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_token_pair_axis": preserves_token_pair_axis,
        "preserves_pair_slot_axis": preserves_pair_slot_axis,
        "preserves_topk_slot_axis": preserves_topk_slot_axis,
        "preserves_output_group_axis": preserves_output_group_axis,
        "preserves_k_block_axis": preserves_k_block_axis,
        "preserves_codeword_group_axis": preserves_codeword_group_axis,
        "uses_token_pair_slot_topk_output_group_streams": (
            uses_token_pair_slot_topk_output_group_streams
        ),
        "preserves_q2_token_topk_scatter": preserves_q2_token_topk_scatter,
        "avoids_full_expert_axis": avoids_full_expert_axis,
        "streams_compressed_codeword_tiles": streams_compressed_codeword_tiles,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "uses_rejected_token_pair_output_group_schedule": (
            uses_rejected_token_pair_output_group_schedule
        ),
        "uses_rejected_token_pair_kblock_schedule": (
            uses_rejected_token_pair_kblock_schedule
        ),
        "uses_rejected_token_expert_output_block_schedule": (
            uses_rejected_token_expert_output_block_schedule
        ),
        "uses_rejected_token_output_stripe_group_schedule": (
            uses_rejected_token_output_stripe_group_schedule
        ),
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "reconstructs_e8p_per_fragment": reconstructs_e8p_per_fragment,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_token_pair_slot_topk_codeword_group_pipeline_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _TOKEN_PAIR_SLOT_TOPK_CODEWORD_GROUP_PIPELINE_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_name = (
        "e8p_token_pair_slot_topk_codeword_group_pipeline_rhs_sorted_matmul"
    )
    primitive_binding_present = (
        primitive_name in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(primitive_source, primitive_name)
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    dispatch_grid_ok = primitive_binding_present and (
        (
            "token_pairs_x_pair_slots_x_topk_slots_x_codeword_groups_x_output_stripes_x_k_blocks"
            in primitive_source
        )
        or all(
            token in scoped_source
            for token in (
                "token_pair",
                "pair_slot",
                "topk_slot",
                "codeword_group",
                "output_stripe",
                "k_block",
            )
        )
    )
    preserves_token_pair_axis = (
        "token_pair" in scoped_source or "token_pairs" in scoped_source
    )
    preserves_pair_slot_axis = "pair_slot" in scoped_source or "pair_slots" in scoped_source
    preserves_topk_slot_axis = "topk_slot" in scoped_source or "topk_slots" in scoped_source
    preserves_codeword_group_axis = "codeword_group" in scoped_source
    preserves_output_stripe_axis = "output_stripe" in scoped_source
    preserves_k_block_axis = "k_block" in scoped_source or "kblock" in scoped_source
    uses_token_pair_slot_topk_codeword_group_pipeline = (
        "token_pair_slot_topk_codeword_group_pipeline" in scoped_source
    )
    preserves_q2_token_topk_scatter = (
        "token_pair_slot_topk_codeword_group_topk_scatter" in scoped_source
        or "pair_slot_topk_scatter" in scoped_source
        or "q2_scatter" in scoped_source
        or "exact_token_topk" in scoped_source
        or "exact_token_topk_scatter" in scoped_source
    )
    codeword_group_index = scoped_source.find("codeword_group")
    output_stripe_index = scoped_source.find("output_stripe")
    streams_codeword_groups_before_output_stripe = (
        "codeword_groups_before_output_stripe" in scoped_source
        or (
            codeword_group_index >= 0
            and output_stripe_index >= 0
            and codeword_group_index < output_stripe_index
        )
    )
    streams_compressed_codeword_tiles = (
        (
            "codeword_tiles" in kernel
            or "code_tiles" in kernel
            or "compressed_codeword" in kernel
        )
        and (
            "uint16_t" in kernel
            or "ushort" in kernel
            or "uint16" in kernel
            or "compressed_codeword" in kernel
        )
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    uses_rejected_token_pair_slot_topk_output_group_schedule = (
        "token_pair_slot_topk_output_group_stream" in scoped_source
        or "token_pairs_x_pair_slots_x_topk_slots_x_output_groups_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    uses_rejected_token_pair_output_group_schedule = (
        "token_pair_output_group_stream" in scoped_source
        or "token_pairs_x_output_groups_x_active_experts_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    uses_rejected_token_pair_kblock_schedule = (
        "token_pair_kblock_accumulator_stream" in scoped_source
        or "token_pairs_x_active_experts_x_k_blocks_x_output_blocks_x_codeword_groups"
        in scoped_source
    )
    uses_rejected_token_expert_output_block_schedule = (
        "token_expert_output_block_stream" in scoped_source
        or "tokens_x_active_experts_x_output_blocks_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    uses_rejected_token_output_stripe_group_schedule = (
        "token_output_stripe_group_stream" in scoped_source
        or "tokens_x_output_stripes_x_topk_groups_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel
    reconstructs_e8p_per_fragment = (
        "mlx_vq_decode_e8p" in kernel and "fragment" in kernel
    )

    passes_contract = (
        kernel_present
        and primitive_binding_present
        and dispatch_grid_ok
        and preserves_token_pair_axis
        and preserves_pair_slot_axis
        and preserves_topk_slot_axis
        and preserves_codeword_group_axis
        and preserves_output_stripe_axis
        and preserves_k_block_axis
        and uses_token_pair_slot_topk_codeword_group_pipeline
        and preserves_q2_token_topk_scatter
        and streams_codeword_groups_before_output_stripe
        and streams_compressed_codeword_tiles
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and not uses_rejected_token_pair_slot_topk_output_group_schedule
        and not uses_rejected_token_pair_output_group_schedule
        and not uses_rejected_token_pair_kblock_schedule
        and not uses_rejected_token_expert_output_block_schedule
        and not uses_rejected_token_output_stripe_group_schedule
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
        and not reconstructs_e8p_per_fragment
    )

    if passes_contract:
        decision = (
            "token_pair_slot_topk_codeword_group_pipeline_source_guardrail_present"
        )
        required_next_features = [
            "prove_token_pair_slot_topk_codeword_group_pipeline_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_token_pair_slot_topk_codeword_group_pipeline_source"
        required_next_features = [
            "add_token_pair_slot_topk_codeword_group_pipeline_native_source_guardrail",
            "bind_token_pair_slot_topk_codeword_group_pipeline_primitive",
            "preserve_token_pairs_pair_slots_topk_slots_codeword_groups_output_stripes_kblocks_axes",
            "stream_codeword_groups_before_output_stripe_writeback",
            "preserve_q2_token_topk_scatter_without_output_group_inner_loop",
            "stream_compressed_uint16_codeword_tiles_and_scale_slots",
            "avoid_token_pair_slot_topk_output_group_stream_schedule",
            "avoid_decoded_dense_rhs",
        ]
        rejected_next_steps = [
            "do_not_benchmark_token_pair_slot_topk_codeword_group_pipeline_before_source_guardrail",
            "do_not_retime_token_pair_slot_topk_output_group_stream",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "token_pairs_x_pair_slots_x_topk_slots_x_codeword_groups_x_output_stripes_x_k_blocks"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_token_pair_axis": preserves_token_pair_axis,
        "preserves_pair_slot_axis": preserves_pair_slot_axis,
        "preserves_topk_slot_axis": preserves_topk_slot_axis,
        "preserves_codeword_group_axis": preserves_codeword_group_axis,
        "preserves_output_stripe_axis": preserves_output_stripe_axis,
        "preserves_k_block_axis": preserves_k_block_axis,
        "uses_token_pair_slot_topk_codeword_group_pipeline": (
            uses_token_pair_slot_topk_codeword_group_pipeline
        ),
        "preserves_q2_token_topk_scatter": preserves_q2_token_topk_scatter,
        "streams_codeword_groups_before_output_stripe": (
            streams_codeword_groups_before_output_stripe
        ),
        "streams_compressed_codeword_tiles": streams_compressed_codeword_tiles,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "uses_rejected_token_pair_slot_topk_output_group_schedule": (
            uses_rejected_token_pair_slot_topk_output_group_schedule
        ),
        "uses_rejected_token_pair_output_group_schedule": (
            uses_rejected_token_pair_output_group_schedule
        ),
        "uses_rejected_token_pair_kblock_schedule": (
            uses_rejected_token_pair_kblock_schedule
        ),
        "uses_rejected_token_expert_output_block_schedule": (
            uses_rejected_token_expert_output_block_schedule
        ),
        "uses_rejected_token_output_stripe_group_schedule": (
            uses_rejected_token_output_stripe_group_schedule
        ),
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "reconstructs_e8p_per_fragment": reconstructs_e8p_per_fragment,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_token_pair_slot_topk_scale_slot_broadcast_stream_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _TOKEN_PAIR_SLOT_TOPK_SCALE_SLOT_BROADCAST_STREAM_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_name = (
        "e8p_token_pair_slot_topk_scale_slot_broadcast_stream_rhs_sorted_matmul"
    )
    primitive_binding_present = (
        primitive_name in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(primitive_source, primitive_name)
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    dispatch_grid_ok = primitive_binding_present and (
        (
            "token_pairs_x_pair_slots_x_topk_slots_x_scale_slots_x_output_stripes_x_k_blocks_x_codewords"
            in primitive_source
        )
        or all(
            token in scoped_source
            for token in (
                "token_pair",
                "pair_slot",
                "topk_slot",
                "scale_slot",
                "output_stripe",
                "k_block",
                "codeword",
            )
        )
    )
    preserves_token_pair_axis = (
        "token_pair" in scoped_source or "token_pairs" in scoped_source
    )
    preserves_pair_slot_axis = "pair_slot" in scoped_source or "pair_slots" in scoped_source
    preserves_topk_slot_axis = "topk_slot" in scoped_source or "topk_slots" in scoped_source
    preserves_scale_slot_axis = (
        "scale_slot" in scoped_source or "scale_slots" in scoped_source
    )
    preserves_output_stripe_axis = "output_stripe" in scoped_source
    preserves_k_block_axis = "k_block" in scoped_source or "kblock" in scoped_source
    preserves_codeword_axis = "codeword" in scoped_source
    uses_token_pair_slot_topk_scale_slot_broadcast_stream = (
        "token_pair_slot_topk_scale_slot_broadcast_stream" in scoped_source
    )
    preserves_q2_token_topk_scatter = (
        "token_pair_slot_topk_scale_slot_broadcast_topk_scatter" in scoped_source
        or "pair_slot_topk_scatter" in scoped_source
        or "q2_scatter" in scoped_source
        or "exact_token_topk" in scoped_source
        or "exact_token_topk_scatter" in scoped_source
    )
    scale_slot_index = scoped_source.find("scale_slot")
    codeword_index = scoped_source.find("codeword")
    broadcasts_scale_slots_before_codewords = (
        "scale_slot_broadcast_before_codeword" in scoped_source
        or "broadcast_scale_slots_before_codeword" in scoped_source
        or (
            scale_slot_index >= 0
            and codeword_index >= 0
            and scale_slot_index < codeword_index
        )
    )
    streams_compressed_codeword_tiles = (
        (
            "codeword_tiles" in kernel
            or "code_tiles" in kernel
            or "compressed_codeword" in kernel
        )
        and (
            "uint16_t" in kernel
            or "ushort" in kernel
            or "uint16" in kernel
            or "compressed_codeword" in kernel
        )
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    uses_rejected_token_pair_slot_topk_codeword_group_pipeline = (
        "token_pair_slot_topk_codeword_group_pipeline" in scoped_source
        or "token_pairs_x_pair_slots_x_topk_slots_x_codeword_groups_x_output_stripes_x_k_blocks"
        in scoped_source
    )
    uses_rejected_token_pair_slot_topk_output_group_schedule = (
        "token_pair_slot_topk_output_group_stream" in scoped_source
        or "token_pairs_x_pair_slots_x_topk_slots_x_output_groups_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    uses_rejected_token_pair_output_group_schedule = (
        "token_pair_output_group_stream" in scoped_source
        or "token_pairs_x_output_groups_x_active_experts_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    uses_rejected_token_pair_kblock_schedule = (
        "token_pair_kblock_accumulator_stream" in scoped_source
        or "token_pairs_x_active_experts_x_k_blocks_x_output_blocks_x_codeword_groups"
        in scoped_source
    )
    uses_rejected_token_expert_output_block_schedule = (
        "token_expert_output_block_stream" in scoped_source
        or "tokens_x_active_experts_x_output_blocks_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    uses_rejected_token_output_stripe_group_schedule = (
        "token_output_stripe_group_stream" in scoped_source
        or "tokens_x_output_stripes_x_topk_groups_x_k_blocks_x_codeword_groups"
        in scoped_source
    )
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel
    reconstructs_e8p_per_fragment = (
        "mlx_vq_decode_e8p" in kernel and "fragment" in kernel
    )

    passes_contract = (
        kernel_present
        and primitive_binding_present
        and dispatch_grid_ok
        and preserves_token_pair_axis
        and preserves_pair_slot_axis
        and preserves_topk_slot_axis
        and preserves_scale_slot_axis
        and preserves_output_stripe_axis
        and preserves_k_block_axis
        and preserves_codeword_axis
        and uses_token_pair_slot_topk_scale_slot_broadcast_stream
        and preserves_q2_token_topk_scatter
        and broadcasts_scale_slots_before_codewords
        and streams_compressed_codeword_tiles
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and not uses_rejected_token_pair_slot_topk_codeword_group_pipeline
        and not uses_rejected_token_pair_slot_topk_output_group_schedule
        and not uses_rejected_token_pair_output_group_schedule
        and not uses_rejected_token_pair_kblock_schedule
        and not uses_rejected_token_expert_output_block_schedule
        and not uses_rejected_token_output_stripe_group_schedule
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
        and not reconstructs_e8p_per_fragment
    )

    if passes_contract:
        decision = (
            "token_pair_slot_topk_scale_slot_broadcast_stream_source_guardrail_present"
        )
        required_next_features = [
            "prove_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_token_pair_slot_topk_scale_slot_broadcast_stream_source"
        required_next_features = [
            "add_token_pair_slot_topk_scale_slot_broadcast_stream_native_source_guardrail",
            "bind_token_pair_slot_topk_scale_slot_broadcast_stream_primitive",
            "preserve_token_pairs_pair_slots_topk_slots_scale_slots_output_stripes_kblocks_codeword_axes",
            "broadcast_scale_slots_before_codeword_streaming",
            "preserve_q2_token_topk_scatter_without_codeword_group_pipeline",
            "stream_compressed_uint16_codeword_tiles_and_scale_slots",
            "avoid_token_pair_slot_topk_codeword_group_pipeline_schedule",
            "avoid_decoded_dense_rhs",
        ]
        rejected_next_steps = [
            "do_not_benchmark_token_pair_slot_topk_scale_slot_broadcast_stream_before_source_guardrail",
            "do_not_retime_token_pair_slot_topk_codeword_group_pipeline",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "token_pairs_x_pair_slots_x_topk_slots_x_scale_slots_x_output_stripes_x_k_blocks_x_codewords"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_token_pair_axis": preserves_token_pair_axis,
        "preserves_pair_slot_axis": preserves_pair_slot_axis,
        "preserves_topk_slot_axis": preserves_topk_slot_axis,
        "preserves_scale_slot_axis": preserves_scale_slot_axis,
        "preserves_output_stripe_axis": preserves_output_stripe_axis,
        "preserves_k_block_axis": preserves_k_block_axis,
        "preserves_codeword_axis": preserves_codeword_axis,
        "uses_token_pair_slot_topk_scale_slot_broadcast_stream": (
            uses_token_pair_slot_topk_scale_slot_broadcast_stream
        ),
        "preserves_q2_token_topk_scatter": preserves_q2_token_topk_scatter,
        "broadcasts_scale_slots_before_codewords": (
            broadcasts_scale_slots_before_codewords
        ),
        "streams_compressed_codeword_tiles": streams_compressed_codeword_tiles,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "uses_rejected_token_pair_slot_topk_codeword_group_pipeline": (
            uses_rejected_token_pair_slot_topk_codeword_group_pipeline
        ),
        "uses_rejected_token_pair_slot_topk_output_group_schedule": (
            uses_rejected_token_pair_slot_topk_output_group_schedule
        ),
        "uses_rejected_token_pair_output_group_schedule": (
            uses_rejected_token_pair_output_group_schedule
        ),
        "uses_rejected_token_pair_kblock_schedule": (
            uses_rejected_token_pair_kblock_schedule
        ),
        "uses_rejected_token_expert_output_block_schedule": (
            uses_rejected_token_expert_output_block_schedule
        ),
        "uses_rejected_token_output_stripe_group_schedule": (
            uses_rejected_token_output_stripe_group_schedule
        ),
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "reconstructs_e8p_per_fragment": reconstructs_e8p_per_fragment,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_token_pair_slot_topk_route_bucket_codeword_reduce_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _TOKEN_PAIR_SLOT_TOPK_ROUTE_BUCKET_CODEWORD_REDUCE_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_name = (
        "e8p_token_pair_slot_topk_route_bucket_codeword_reduce_rhs_sorted_matmul"
    )
    primitive_binding_present = (
        primitive_name in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(primitive_source, primitive_name)
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    dispatch_grid_ok = primitive_binding_present and (
        (
            "route_buckets_x_token_pairs_x_pair_slots_x_topk_slots_x_k_blocks_x_output_microtiles_x_codeword_tiles"
            in primitive_source
        )
        or all(
            token in scoped_source
            for token in (
                "route_bucket",
                "token_pair",
                "pair_slot",
                "topk_slot",
                "k_block",
                "output_microtile",
                "codeword_tile",
            )
        )
    )
    preserves_route_bucket_axis = (
        "route_bucket" in scoped_source or "route_buckets" in scoped_source
    )
    preserves_token_pair_axis = (
        "token_pair" in scoped_source or "token_pairs" in scoped_source
    )
    preserves_pair_slot_axis = "pair_slot" in scoped_source or "pair_slots" in scoped_source
    preserves_topk_slot_axis = "topk_slot" in scoped_source or "topk_slots" in scoped_source
    preserves_k_block_axis = "k_block" in scoped_source or "kblock" in scoped_source
    preserves_output_microtile_axis = (
        "output_microtile" in scoped_source or "output_microtiles" in scoped_source
    )
    preserves_codeword_tile_axis = (
        "codeword_tile" in scoped_source
        or "codeword_tiles" in scoped_source
        or "code_tiles" in scoped_source
    )
    uses_token_pair_slot_topk_route_bucket_codeword_reduce = (
        "token_pair_slot_topk_route_bucket_codeword_reduce" in scoped_source
    )
    preserves_route_bucket_offsets = any(
        token in scoped_source
        for token in (
            "route_bucket_offsets",
            "route_bucket_offset",
            "route_slot_exactness",
            "exact_route_slot",
        )
    )
    preserves_q2_token_topk_scatter = (
        "token_pair_slot_topk_route_bucket_codeword_reduce_topk_scatter"
        in scoped_source
        or "pair_slot_topk_scatter" in scoped_source
        or "q2_scatter" in scoped_source
        or "exact_token_topk" in scoped_source
        or "exact_token_topk_scatter" in scoped_source
    )
    streams_compressed_codeword_tiles = (
        (
            "codeword_tiles" in kernel
            or "code_tiles" in kernel
            or "compressed_codeword" in kernel
        )
        and (
            "uint16_t" in kernel
            or "ushort" in kernel
            or "uint16" in kernel
            or "compressed_codeword" in kernel
        )
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    uses_rejected_token_pair_slot_topk_scale_slot_broadcast_stream = (
        "token_pair_slot_topk_scale_slot_broadcast_stream" in scoped_source
        or "token_pairs_x_pair_slots_x_topk_slots_x_scale_slots_x_output_stripes_x_k_blocks_x_codewords"
        in scoped_source
    )
    uses_rejected_token_pair_slot_topk_codeword_group_pipeline = (
        "token_pair_slot_topk_codeword_group_pipeline" in scoped_source
        or "token_pairs_x_pair_slots_x_topk_slots_x_codeword_groups_x_output_stripes_x_k_blocks"
        in scoped_source
    )
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel
    reconstructs_e8p_per_fragment = (
        "mlx_vq_decode_e8p" in kernel and "fragment" in kernel
    )

    passes_contract = (
        kernel_present
        and primitive_binding_present
        and dispatch_grid_ok
        and preserves_route_bucket_axis
        and preserves_token_pair_axis
        and preserves_pair_slot_axis
        and preserves_topk_slot_axis
        and preserves_k_block_axis
        and preserves_output_microtile_axis
        and preserves_codeword_tile_axis
        and uses_token_pair_slot_topk_route_bucket_codeword_reduce
        and preserves_route_bucket_offsets
        and preserves_q2_token_topk_scatter
        and streams_compressed_codeword_tiles
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and not uses_rejected_token_pair_slot_topk_scale_slot_broadcast_stream
        and not uses_rejected_token_pair_slot_topk_codeword_group_pipeline
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
        and not reconstructs_e8p_per_fragment
    )

    if passes_contract:
        decision = (
            "token_pair_slot_topk_route_bucket_codeword_reduce_source_guardrail_present"
        )
        required_next_features = [
            "prove_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_token_pair_slot_topk_route_bucket_codeword_reduce_source"
        required_next_features = [
            "add_token_pair_slot_topk_route_bucket_codeword_reduce_native_source_guardrail",
            "bind_token_pair_slot_topk_route_bucket_codeword_reduce_primitive",
            "preserve_route_buckets_token_pairs_pair_slots_topk_slots_kblocks_output_microtiles_codeword_tiles_axes",
            "preserve_route_bucket_offsets_for_route_slot_exactness",
            "preserve_q2_token_topk_scatter_without_scale_slot_broadcast_stream",
            "stream_compressed_uint16_codeword_tiles_and_scale_slots",
            "avoid_token_pair_slot_topk_scale_slot_broadcast_stream_schedule",
            "avoid_decoded_dense_rhs",
        ]
        rejected_next_steps = [
            "do_not_benchmark_token_pair_slot_topk_route_bucket_codeword_reduce_before_source_guardrail",
            "do_not_retime_token_pair_slot_topk_scale_slot_broadcast_stream",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "route_buckets_x_token_pairs_x_pair_slots_x_topk_slots_x_k_blocks_x_output_microtiles_x_codeword_tiles"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_route_bucket_axis": preserves_route_bucket_axis,
        "preserves_token_pair_axis": preserves_token_pair_axis,
        "preserves_pair_slot_axis": preserves_pair_slot_axis,
        "preserves_topk_slot_axis": preserves_topk_slot_axis,
        "preserves_k_block_axis": preserves_k_block_axis,
        "preserves_output_microtile_axis": preserves_output_microtile_axis,
        "preserves_codeword_tile_axis": preserves_codeword_tile_axis,
        "uses_token_pair_slot_topk_route_bucket_codeword_reduce": (
            uses_token_pair_slot_topk_route_bucket_codeword_reduce
        ),
        "preserves_route_bucket_offsets": preserves_route_bucket_offsets,
        "preserves_q2_token_topk_scatter": preserves_q2_token_topk_scatter,
        "streams_compressed_codeword_tiles": streams_compressed_codeword_tiles,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "uses_rejected_token_pair_slot_topk_scale_slot_broadcast_stream": (
            uses_rejected_token_pair_slot_topk_scale_slot_broadcast_stream
        ),
        "uses_rejected_token_pair_slot_topk_codeword_group_pipeline": (
            uses_rejected_token_pair_slot_topk_codeword_group_pipeline
        ),
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "reconstructs_e8p_per_fragment": reconstructs_e8p_per_fragment,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_token_pair_slot_topk_kblock_microtile_stream_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _TOKEN_PAIR_SLOT_TOPK_KBLOCK_MICROTILE_STREAM_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_name = (
        "e8p_token_pair_slot_topk_kblock_microtile_stream_rhs_sorted_matmul"
    )
    primitive_binding_present = (
        primitive_name in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(primitive_source, primitive_name)
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    dispatch_grid_ok = primitive_binding_present and (
        (
            "token_pairs_x_pair_slots_x_topk_slots_x_k_blocks_x_output_microtiles"
            in primitive_source
        )
        or all(
            token in scoped_source
            for token in (
                "token_pair",
                "pair_slot",
                "topk_slot",
                "k_block",
                "output_microtile",
            )
        )
    )
    preserves_token_pair_axis = (
        "token_pair" in scoped_source or "token_pairs" in scoped_source
    )
    preserves_pair_slot_axis = "pair_slot" in scoped_source or "pair_slots" in scoped_source
    preserves_topk_slot_axis = "topk_slot" in scoped_source or "topk_slots" in scoped_source
    preserves_k_block_axis = "k_block" in scoped_source or "kblock" in scoped_source
    preserves_output_microtile_axis = (
        "output_microtile" in scoped_source or "output_microtiles" in scoped_source
    )
    uses_token_pair_slot_topk_kblock_microtile_stream = (
        "token_pair_slot_topk_kblock_microtile_stream" in scoped_source
    )
    preserves_q2_token_topk_scatter = (
        "token_pair_slot_topk_kblock_microtile_stream_topk_scatter"
        in scoped_source
        or "pair_slot_topk_scatter" in scoped_source
        or "q2_scatter" in scoped_source
        or "exact_token_topk" in scoped_source
        or "exact_token_topk_scatter" in scoped_source
    )
    streams_compressed_codeword_tiles = (
        (
            "codeword_tiles" in kernel
            or "code_tiles" in kernel
            or "compressed_codeword" in kernel
        )
        and (
            "uint16_t" in kernel
            or "ushort" in kernel
            or "uint16" in kernel
            or "compressed_codeword" in kernel
        )
    )
    streams_codeword_tiles_inside_threadgroup = (
        streams_compressed_codeword_tiles
        and (
            "codeword_tiles_inside_threadgroup" in scoped_source
            or "kblock_microtile_stream" in scoped_source
            or "codeword_tiles_inside_kblock_microtile" in scoped_source
        )
    )
    avoids_route_slot_expansion = (
        "no_route_slot_expansion" in scoped_source
        or "without_route_slot_expansion" in scoped_source
        or "avoid_route_slot_expansion" in scoped_source
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    uses_rejected_token_pair_slot_topk_route_bucket_codeword_reduce = (
        "token_pair_slot_topk_route_bucket_codeword_reduce" in scoped_source
        or "route_buckets_x_token_pairs_x_pair_slots_x_topk_slots_x_k_blocks_x_output_microtiles_x_codeword_tiles"
        in scoped_source
    )
    uses_rejected_token_pair_slot_topk_scale_slot_broadcast_stream = (
        "token_pair_slot_topk_scale_slot_broadcast_stream" in scoped_source
        or "token_pairs_x_pair_slots_x_topk_slots_x_scale_slots_x_output_stripes_x_k_blocks_x_codewords"
        in scoped_source
    )
    uses_rejected_token_pair_slot_topk_codeword_group_pipeline = (
        "token_pair_slot_topk_codeword_group_pipeline" in scoped_source
        or "token_pairs_x_pair_slots_x_topk_slots_x_codeword_groups_x_output_stripes_x_k_blocks"
        in scoped_source
    )
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel
    reconstructs_e8p_per_fragment = (
        "mlx_vq_decode_e8p" in kernel and "fragment" in kernel
    )

    passes_contract = (
        kernel_present
        and primitive_binding_present
        and dispatch_grid_ok
        and preserves_token_pair_axis
        and preserves_pair_slot_axis
        and preserves_topk_slot_axis
        and preserves_k_block_axis
        and preserves_output_microtile_axis
        and uses_token_pair_slot_topk_kblock_microtile_stream
        and preserves_q2_token_topk_scatter
        and streams_compressed_codeword_tiles
        and streams_codeword_tiles_inside_threadgroup
        and avoids_route_slot_expansion
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and not uses_rejected_token_pair_slot_topk_route_bucket_codeword_reduce
        and not uses_rejected_token_pair_slot_topk_scale_slot_broadcast_stream
        and not uses_rejected_token_pair_slot_topk_codeword_group_pipeline
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
        and not reconstructs_e8p_per_fragment
    )

    if passes_contract:
        decision = (
            "token_pair_slot_topk_kblock_microtile_stream_source_guardrail_present"
        )
        required_next_features = [
            "prove_token_pair_slot_topk_kblock_microtile_stream_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_token_pair_slot_topk_kblock_microtile_stream_source"
        required_next_features = [
            "add_token_pair_slot_topk_kblock_microtile_stream_native_source_guardrail",
            "bind_token_pair_slot_topk_kblock_microtile_stream_primitive",
            "preserve_token_pairs_pair_slots_topk_slots_kblocks_output_microtiles_axes",
            "avoid_route_bucket_route_slot_expansion",
            "stream_compressed_uint16_codeword_tiles_inside_kblock_microtile",
            "preserve_q2_token_topk_scatter_without_route_bucket_codeword_reduce",
            "avoid_token_pair_slot_topk_route_bucket_codeword_reduce_schedule",
            "avoid_decoded_dense_rhs",
        ]
        rejected_next_steps = [
            "do_not_benchmark_token_pair_slot_topk_kblock_microtile_stream_before_source_guardrail",
            "do_not_retime_token_pair_slot_topk_route_bucket_codeword_reduce",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "token_pairs_x_pair_slots_x_topk_slots_x_k_blocks_x_output_microtiles"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_token_pair_axis": preserves_token_pair_axis,
        "preserves_pair_slot_axis": preserves_pair_slot_axis,
        "preserves_topk_slot_axis": preserves_topk_slot_axis,
        "preserves_k_block_axis": preserves_k_block_axis,
        "preserves_output_microtile_axis": preserves_output_microtile_axis,
        "uses_token_pair_slot_topk_kblock_microtile_stream": (
            uses_token_pair_slot_topk_kblock_microtile_stream
        ),
        "preserves_q2_token_topk_scatter": preserves_q2_token_topk_scatter,
        "streams_compressed_codeword_tiles": streams_compressed_codeword_tiles,
        "streams_codeword_tiles_inside_threadgroup": (
            streams_codeword_tiles_inside_threadgroup
        ),
        "avoids_route_slot_expansion": avoids_route_slot_expansion,
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "uses_rejected_token_pair_slot_topk_route_bucket_codeword_reduce": (
            uses_rejected_token_pair_slot_topk_route_bucket_codeword_reduce
        ),
        "uses_rejected_token_pair_slot_topk_scale_slot_broadcast_stream": (
            uses_rejected_token_pair_slot_topk_scale_slot_broadcast_stream
        ),
        "uses_rejected_token_pair_slot_topk_codeword_group_pipeline": (
            uses_rejected_token_pair_slot_topk_codeword_group_pipeline
        ),
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "reconstructs_e8p_per_fragment": reconstructs_e8p_per_fragment,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_token_pair_slot_topk_output_tile_fused_stream_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _TOKEN_PAIR_SLOT_TOPK_OUTPUT_TILE_FUSED_STREAM_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_name = (
        "e8p_token_pair_slot_topk_output_tile_fused_stream_rhs_sorted_matmul"
    )
    primitive_binding_present = (
        primitive_name in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    scoped_source = kernel
    if primitive_binding_present:
        try:
            scoped_source = (
                kernel
                + "\n"
                + _extract_cpp_function_source(primitive_source, primitive_name)
            )
        except ValueError:
            scoped_source = f"{kernel}\n{primitive_source}"

    dispatch_grid_ok = primitive_binding_present and (
        "token_pairs_x_pair_slots_x_topk_slots_x_output_tiles" in primitive_source
        or all(
            token in scoped_source
            for token in ("token_pair", "pair_slot", "topk_slot", "output_tile")
        )
    )
    preserves_token_pair_axis = (
        "token_pair" in scoped_source or "token_pairs" in scoped_source
    )
    preserves_pair_slot_axis = "pair_slot" in scoped_source or "pair_slots" in scoped_source
    preserves_topk_slot_axis = "topk_slot" in scoped_source or "topk_slots" in scoped_source
    preserves_output_tile_axis = (
        "output_tile" in scoped_source or "output_tiles" in scoped_source
    )
    uses_token_pair_slot_topk_output_tile_fused_stream = (
        "token_pair_slot_topk_output_tile_fused_stream" in scoped_source
    )
    preserves_q2_token_topk_scatter = (
        "pair_slot_topk_scatter" in scoped_source
        or "q2_scatter" in scoped_source
        or "exact_token_topk" in scoped_source
        or "exact_token_topk_scatter" in scoped_source
        or "token_pair_slot_topk_output_tile_fused_stream" in scoped_source
    )
    streams_compressed_codeword_tiles = (
        (
            "codeword_tiles" in kernel
            or "code_tiles" in kernel
            or "compressed_codeword" in kernel
        )
        and (
            "uint16_t" in kernel
            or "ushort" in kernel
            or "uint16" in kernel
            or "compressed_codeword" in kernel
        )
    )
    streams_kblocks_and_codewords_inside_output_tile = (
        streams_compressed_codeword_tiles
        and (
            "streams_kblocks_and_codewords_inside_output_tile_fused_workgroup"
            in scoped_source
            or "output_tile_fused_workgroup" in scoped_source
            or "codeword_tiles_streamed_inside_output_tile" in scoped_source
        )
    )
    uses_codeword_scale_slots = "codeword_scale_slots" in kernel
    uses_scale_tiles = "scale_tiles" in kernel
    uses_rejected_token_pair_slot_topk_kblock_microtile_stream = (
        "token_pair_slot_topk_kblock_microtile_stream" in scoped_source
        or "token_pairs_x_pair_slots_x_topk_slots_x_k_blocks_x_output_microtiles"
        in scoped_source
    )
    uses_rejected_token_pair_slot_topk_route_bucket_codeword_reduce = (
        "token_pair_slot_topk_route_bucket_codeword_reduce" in scoped_source
        or "route_buckets_x_token_pairs_x_pair_slots_x_topk_slots_x_k_blocks_x_output_microtiles_x_codeword_tiles"
        in scoped_source
    )
    uses_rejected_token_pair_slot_topk_scale_slot_broadcast_stream = (
        "token_pair_slot_topk_scale_slot_broadcast_stream" in scoped_source
        or "token_pairs_x_pair_slots_x_topk_slots_x_scale_slots_x_output_stripes_x_k_blocks_x_codewords"
        in scoped_source
    )
    uses_rejected_token_pair_slot_topk_codeword_group_pipeline = (
        "token_pair_slot_topk_codeword_group_pipeline" in scoped_source
        or "token_pairs_x_pair_slots_x_topk_slots_x_codeword_groups_x_output_stripes_x_k_blocks"
        in scoped_source
    )
    materializes_decoded_dense_rhs = any(
        token in scoped_source
        for token in ("decoded_dense_rhs", "decoded_rhs", "dense_rhs", "array decoded")
    )
    stages_decoded_b = "threadgroup half shared_b" in kernel or "decoded_b" in kernel
    uses_lane_local_fragment_buffer = "lane_local" in kernel
    reconstructs_e8p_per_fragment = (
        "mlx_vq_decode_e8p" in kernel and "fragment" in kernel
    )

    passes_contract = (
        kernel_present
        and primitive_binding_present
        and dispatch_grid_ok
        and preserves_token_pair_axis
        and preserves_pair_slot_axis
        and preserves_topk_slot_axis
        and preserves_output_tile_axis
        and uses_token_pair_slot_topk_output_tile_fused_stream
        and preserves_q2_token_topk_scatter
        and streams_compressed_codeword_tiles
        and streams_kblocks_and_codewords_inside_output_tile
        and uses_codeword_scale_slots
        and uses_scale_tiles
        and not uses_rejected_token_pair_slot_topk_kblock_microtile_stream
        and not uses_rejected_token_pair_slot_topk_route_bucket_codeword_reduce
        and not uses_rejected_token_pair_slot_topk_scale_slot_broadcast_stream
        and not uses_rejected_token_pair_slot_topk_codeword_group_pipeline
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
        and not uses_lane_local_fragment_buffer
        and not reconstructs_e8p_per_fragment
    )

    if passes_contract:
        decision = (
            "token_pair_slot_topk_output_tile_fused_stream_source_guardrail_present"
        )
        required_next_features = [
            "prove_token_pair_slot_topk_output_tile_fused_stream_native_parity",
            "prove_air_down_group_size_352_artifact_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_claim_speed_from_source_guardrail",
            "do_not_route_resident_auto_before_lane_s_speed_gate",
        ]
    else:
        decision = "missing_token_pair_slot_topk_output_tile_fused_stream_source"
        required_next_features = [
            "add_token_pair_slot_topk_output_tile_fused_stream_native_source_guardrail",
            "bind_token_pair_slot_topk_output_tile_fused_stream_primitive",
            "preserve_token_pairs_pair_slots_topk_slots_output_tiles_axes",
            "stream_compressed_uint16_codeword_tiles_inside_output_tile_fused_workgroup",
            "preserve_q2_token_topk_scatter_without_kblock_microtile_route_slot_expansion",
            "avoid_token_pair_slot_topk_kblock_microtile_stream_schedule",
            "avoid_decoded_dense_rhs",
        ]
        rejected_next_steps = [
            "do_not_benchmark_token_pair_slot_topk_output_tile_fused_stream_before_source_guardrail",
            "do_not_retime_token_pair_slot_topk_kblock_microtile_stream",
            "do_not_materialize_decoded_dense_rhs",
        ]

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "dispatch_grid": (
            "token_pairs_x_pair_slots_x_topk_slots_x_output_tiles"
            if dispatch_grid_ok
            else "unknown"
        ),
        "preserves_token_pair_axis": preserves_token_pair_axis,
        "preserves_pair_slot_axis": preserves_pair_slot_axis,
        "preserves_topk_slot_axis": preserves_topk_slot_axis,
        "preserves_output_tile_axis": preserves_output_tile_axis,
        "uses_token_pair_slot_topk_output_tile_fused_stream": (
            uses_token_pair_slot_topk_output_tile_fused_stream
        ),
        "preserves_q2_token_topk_scatter": preserves_q2_token_topk_scatter,
        "streams_compressed_codeword_tiles": streams_compressed_codeword_tiles,
        "streams_kblocks_and_codewords_inside_output_tile": (
            streams_kblocks_and_codewords_inside_output_tile
        ),
        "uses_codeword_scale_slots": uses_codeword_scale_slots,
        "uses_scale_tiles": uses_scale_tiles,
        "uses_rejected_token_pair_slot_topk_kblock_microtile_stream": (
            uses_rejected_token_pair_slot_topk_kblock_microtile_stream
        ),
        "uses_rejected_token_pair_slot_topk_route_bucket_codeword_reduce": (
            uses_rejected_token_pair_slot_topk_route_bucket_codeword_reduce
        ),
        "uses_rejected_token_pair_slot_topk_scale_slot_broadcast_stream": (
            uses_rejected_token_pair_slot_topk_scale_slot_broadcast_stream
        ),
        "uses_rejected_token_pair_slot_topk_codeword_group_pipeline": (
            uses_rejected_token_pair_slot_topk_codeword_group_pipeline
        ),
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "uses_lane_local_fragment_buffer": uses_lane_local_fragment_buffer,
        "reconstructs_e8p_per_fragment": reconstructs_e8p_per_fragment,
        "speed_claim": False,
        "native_parity_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
    }


def audit_e8p_component_stream_shared_decode_source(
    *,
    metal_source: str,
    primitive_source: str,
    kernel_name: str = _COMPONENT_STREAM_SHARED_DECODE_KERNEL,
) -> dict[str, Any]:
    try:
        kernel = _extract_kernel_source(metal_source, kernel_name)
        kernel_present = True
    except ValueError:
        kernel = ""
        kernel_present = False

    primitive_binding_present = (
        "e8p_component_stream_rhs_sorted_shared_decode_matmul" in primitive_source
        and f'cache.get("{kernel_name}")' in primitive_source
    )
    has_tensorops_matmul = (
        "matmul_op.run" in kernel
        or "mpp::tensor_ops" in kernel
        or "get_right_input_cooperative_tensor" in kernel
    )
    uses_component_stream_storage = all(
        token in kernel
        for token in (
            "sign_component_bits",
            "abs_index_tiles",
            "component_scale_slots",
            "scale_tiles",
        )
    )
    shared_decode_cache_present = any(
        token in f"{kernel}\n{primitive_source}"
        for token in (
            "component_shared_decode",
            "shared_component_decode",
            "component_decode_cache",
        )
    )
    reconstructs_components_per_fragment = (
        "b_t" in kernel
        and "mlx_vq_decode_e8p_component" in kernel
        and uses_component_stream_storage
        and not shared_decode_cache_present
    )
    materializes_decoded_dense_rhs = any(
        token in f"{kernel}\n{primitive_source}"
        for token in (
            "decoded_dense_rhs",
            "decoded_rhs",
            "dense_rhs",
            "array decoded",
        )
    )
    stages_decoded_b = (
        "threadgroup half" in kernel
        and (
            "get_right_input_cooperative_tensor" in kernel
            or "Btile.template load" in kernel
        )
    )

    if not kernel_present:
        decision = "missing_component_stream_shared_decode_source"
        required_next_features = [
            "add_component_stream_shared_decode_cache_source",
            "bind_component_stream_shared_decode_primitive",
            "reuse_component_pair_kblock_decode_across_n_tiles",
            "prove_shared_decode_cache_matches_component_stream_oracle",
        ]
        rejected_next_steps = [
            "do_not_retime_component_stream_tensorops_unchanged",
            "do_not_materialize_decoded_dense_rhs",
        ]
        structural_risk = (
            "component-stream TensorOps was speed-rejected; the next shared "
            "decode family is still missing from native source"
        )
    elif (
        has_tensorops_matmul
        and uses_component_stream_storage
        and reconstructs_components_per_fragment
    ):
        decision = "reject_component_stream_shared_decode_per_fragment_reconstruction"
        required_next_features = [
            "reuse_component_pair_kblock_decode_across_n_tiles",
            "avoid_per_fragment_component_reconstruction",
            "preserve_compressed_component_stream_storage",
            "prove_native_shared_decode_cache_parity",
        ]
        rejected_next_steps = [
            "do_not_retime_component_stream_tensorops_unchanged",
            "do_not_fill_tensorops_b_fragment_from_e8p_values",
            "do_not_claim_shared_decode_without_shared_cache_source",
        ]
        structural_risk = (
            "shared-decode kernel name reaches TensorOps but still rebuilds "
            "component values inside B fragments instead of using a shared "
            "component-pair/K-block decode cache"
        )
    elif (
        primitive_binding_present
        and uses_component_stream_storage
        and shared_decode_cache_present
        and not has_tensorops_matmul
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
    ):
        decision = "component_stream_shared_decode_scalar_cache_scaffold_present"
        required_next_features = [
            "prove_native_shared_decode_cache_parity",
            "replace_scalar_shared_decode_body_with_valid_parallel_schedule",
            "prove_artifact_shared_decode_cache_parity_after_schedule_change",
        ]
        rejected_next_steps = [
            "do_not_benchmark_scalar_shared_decode_cache_scaffold",
            "do_not_route_resident_auto_before_parallel_schedule",
            "do_not_claim_speed_without_same_window_q2_packet",
        ]
        structural_risk = (
            "component-stream shared decode cache source is present as a "
            "scalar parity scaffold; it must be replaced by a valid parallel "
            "schedule before q2 timing"
        )
    elif (
        primitive_binding_present
        and has_tensorops_matmul
        and uses_component_stream_storage
        and shared_decode_cache_present
        and not materializes_decoded_dense_rhs
        and not stages_decoded_b
    ):
        decision = "component_stream_shared_decode_candidate_present_needs_parity"
        required_next_features = [
            "prove_native_shared_decode_cache_parity",
            "prove_artifact_shared_decode_cache_parity",
            "run_same_window_q2_speed_packet_after_parity",
        ]
        rejected_next_steps = [
            "do_not_benchmark_before_native_and_artifact_parity",
            "do_not_route_resident_auto_before_same_window_q2_speed_gate",
        ]
        structural_risk = (
            "component-stream shared decode source is present, but parity and "
            "same-window q2 timing are still required before any speed claim"
        )
    else:
        decision = "component_stream_shared_decode_source_needs_manual_review"
        required_next_features = [
            "make_shared_decode_cache_and_binding_source_visible",
            "preserve_compressed_component_stream_storage",
            "avoid_decoded_dense_rhs",
        ]
        rejected_next_steps = [
            "do_not_claim_shared_decode_from_ambiguous_source",
        ]
        structural_risk = (
            "component-stream shared decode source is present but does not yet "
            "prove the shared-cache contract"
        )

    return {
        "kernel_name": kernel_name,
        "decision": decision,
        "kernel_present": kernel_present,
        "primitive_binding_present": primitive_binding_present,
        "has_tensorops_matmul": has_tensorops_matmul,
        "uses_component_stream_storage": uses_component_stream_storage,
        "shared_decode_cache_present": shared_decode_cache_present,
        "reconstructs_components_per_fragment": reconstructs_components_per_fragment,
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "stages_decoded_b": stages_decoded_b,
        "speed_claim": False,
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
        "structural_risk": structural_risk,
    }


def audit_e8p_component_stream_partial_reduction_source(
    *,
    metal_source: str,
    primitive_source: str,
    partial_kernel_name: str = _COMPONENT_STREAM_PARTIAL_KERNEL,
    reduce_kernel_name: str = _COMPONENT_STREAM_PARTIAL_REDUCE_KERNEL,
) -> dict[str, Any]:
    normalized_primitive = re.sub(r"\s+", " ", primitive_source)
    try:
        partial_kernel = _extract_kernel_source(metal_source, partial_kernel_name)
        partial_kernel_present = True
    except ValueError:
        partial_kernel = ""
        partial_kernel_present = False
    try:
        reduce_kernel = _extract_kernel_source(metal_source, reduce_kernel_name)
        reduce_kernel_present = True
    except ValueError:
        reduce_kernel = ""
        reduce_kernel_present = False

    partial_accumulation_scratch_present = (
        "component_stream_partials = array" in primitive_source
        and "component_stream_partials.set_data(allocator::malloc(component_stream_partials.nbytes()))"
        in primitive_source
        and (
            "{num_route_tiles, k_blocks, component_pairs, n_tiles, route_tile_size, 64}"
            in normalized_primitive
            or (
                "{num_route_tiles_shape, k_blocks_shape, component_pairs_shape,"
                " n_tiles_shape, route_tile_size_shape, component_width_shape}"
            )
            in normalized_primitive
        )
        and "float32" in primitive_source
    )
    partial_write_kernel_present = (
        partial_kernel_present
        and "device float* component_stream_partials" in partial_kernel
        and "component_stream_partials[partial_offset]" in partial_kernel
    )
    final_reduction_kernel_present = (
        reduce_kernel_present
        and f'cache.get("{reduce_kernel_name}")' in primitive_source
        and "device const float* component_stream_partials" in reduce_kernel
        and "for (uint k_block = 0; k_block < k_blocks; ++k_block)" in reduce_kernel
        and "for (uint component_pair = 0; component_pair < component_pairs; ++component_pair)"
        in reduce_kernel
        and "out[route * output_dims + n]" in reduce_kernel
    )
    partial_binding_present = f'cache.get("{partial_kernel_name}")' in primitive_source
    partial_grid_ok = (
        "MTL::Size::Make(num_route_tiles, component_pairs, k_blocks)"
        in normalized_primitive
    )
    reduction_grid_ok = "MTL::Size::Make(num_route_tiles, n_tiles, 1)" in normalized_primitive
    uses_component_stream_storage = all(
        token in partial_kernel
        for token in (
            "sign_component_bits",
            "abs_index_tiles",
            "component_scale_slots",
            "codeword_scale_slots",
            "scale_tiles",
            "sorted_x",
        )
    )
    materializes_decoded_dense_rhs = (
        "decoded_dense_rhs" in partial_kernel
        or "decoded_rhs" in partial_kernel
        or "dense_rhs" in partial_kernel
        or "array decoded" in primitive_source
    )
    partial_body_tensorops_present = (
        "matmul_op.run" in partial_kernel
        and (
            "mpp::tensor_ops::" in partial_kernel
            or "simdgroup_matrix_storage" in partial_kernel
        )
    )
    component_pair_parallel_reduction_present = (
        "threadgroup float component_pair_sums" in partial_kernel
        and "component_lane" in partial_kernel
        and "output_lane" in partial_kernel
        and "threadgroup_barrier(mem_flags::mem_threadgroup)" in partial_kernel
        and "component_pair_sums[lane]" in partial_kernel
        and "component_pair_sums[lane + 1u]" in partial_kernel
        and "component_pair_sums[lane + 2u]" in partial_kernel
        and "component_pair_sums[lane + 3u]" in partial_kernel
    )
    partial_body_parallel_schedule_present = (
        partial_body_tensorops_present or component_pair_parallel_reduction_present
    )
    scratch_dtype = (
        "float32"
        if partial_accumulation_scratch_present
        and "device float* component_stream_partials" in partial_kernel
        else "unknown"
    )
    passes_contract = (
        partial_accumulation_scratch_present
        and partial_write_kernel_present
        and final_reduction_kernel_present
        and partial_binding_present
        and partial_grid_ok
        and reduction_grid_ok
        and uses_component_stream_storage
        and scratch_dtype == "float32"
        and not materializes_decoded_dense_rhs
    )
    if passes_contract and partial_body_parallel_schedule_present:
        decision = (
            "component_stream_partial_reduction_tensorops_body_present"
            if partial_body_tensorops_present
            else "component_stream_partial_reduction_parallel_body_present"
        )
        required_next_features = (
            [
                "prove_native_component_stream_tensorops_partial_parity",
                "prove_artifact_component_stream_tensorops_partial_parity",
                "run_same_window_q2_speed_packet_after_parity",
            ]
            if partial_body_tensorops_present
            else [
                "prove_native_component_stream_partial_reduction_parity",
                "prove_artifact_component_stream_partial_reduction_parity",
                "run_same_window_q2_speed_packet_after_parity",
            ]
        )
        rejected_next_steps = [
            "do_not_benchmark_before_native_and_artifact_parity",
            "do_not_promote_without_same_window_q2_speed_packet",
        ]
        structural_risk = (
            "component-stream partial accumulation, parallel partial writer body, "
            "and final reduction are source-visible, but this is still a "
            "parity-first candidate before same-window q2 speed proof"
        )
    elif passes_contract:
        decision = "component_stream_partial_reduction_scaffold_scalar_body"
        required_next_features = [
            "replace_scalar_component_stream_partial_body_with_valid_parallel_schedule",
            "preserve_component_stream_partial_reduction_scaffold",
            "preserve_compressed_component_stream_storage",
            "prove_native_component_stream_partial_reduction_parity_after_schedule_change",
        ]
        rejected_next_steps = [
            "do_not_benchmark_scalar_component_stream_partial_scaffold",
            "do_not_promote_without_same_window_q2_speed_packet",
        ]
        structural_risk = (
            "component-stream partial accumulation and final reduction are "
            "source-visible, but the partial writer still computes one scalar "
            "component contribution per lane instead of a valid parallel speed "
            "schedule"
        )
    else:
        decision = "missing_component_stream_partial_reduction_scaffold"
        required_next_features = [
            "add_component_stream_partial_accumulation_scratch",
            "dispatch_component_pair_partial_writer",
            "dispatch_component_stream_partial_reduce",
            "preserve_compressed_component_stream_storage",
        ]
        rejected_next_steps = [
            "do_not_benchmark_component_stream_scalar_oracle_as_speed_path",
            "do_not_materialize_decoded_dense_rhs",
        ]
        structural_risk = (
            "component-stream has scalar parity, but source does not yet prove "
            "the split partial-writer plus final-reducer scaffold required by "
            "the partial-reduction contract"
        )
    return {
        "partial_kernel_name": partial_kernel_name,
        "reduce_kernel_name": reduce_kernel_name,
        "decision": decision,
        "passes_contract": passes_contract,
        "partial_accumulation_scratch_present": partial_accumulation_scratch_present,
        "partial_write_kernel_present": partial_write_kernel_present,
        "final_reduction_kernel_present": final_reduction_kernel_present,
        "partial_binding_present": partial_binding_present,
        "uses_component_stream_storage": uses_component_stream_storage,
        "materializes_decoded_dense_rhs": materializes_decoded_dense_rhs,
        "partial_body_parallel_schedule_present": partial_body_parallel_schedule_present,
        "partial_body_tensorops_present": partial_body_tensorops_present,
        "component_pair_parallel_reduction_present": (
            component_pair_parallel_reduction_present
        ),
        "scratch_dtype": scratch_dtype,
        "speed_claim": False,
        "partial_grid": (
            "route_tiles_x_component_pairs_x_k_blocks" if partial_grid_ok else "unknown"
        ),
        "reduction_grid": "route_tiles_x_n_tiles" if reduction_grid_ok else "unknown",
        "required_next_features": required_next_features,
        "rejected_next_steps": rejected_next_steps,
        "structural_risk": structural_risk,
    }


def _q2_geometry_by_projection(q2_jsonl: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(q2_jsonl)
    by_projection: dict[str, dict[str, Any]] = {}
    for row in rows:
        projection = str(row.get("projection", "unknown"))
        names = row.get("kernel_names") or row.get("trace_nax_symbol_samples") or []
        if not isinstance(names, list):
            continue
        for name in names:
            if not isinstance(name, str) or "affine_gather_qmm" not in name:
                continue
            try:
                geometry = parse_q2_nax_kernel_name(name)
            except ValueError:
                continue
            current = by_projection.setdefault(projection, dict(geometry))
            current["observed_rows"] = int(current.get("observed_rows", 0)) + 1
            tokens = row.get("tokens")
            if isinstance(tokens, int):
                current.setdefault("tokens_observed", [])
                if tokens not in current["tokens_observed"]:
                    current["tokens_observed"].append(tokens)
            route_count = row.get("route_count")
            if isinstance(route_count, int):
                current["max_route_count"] = max(int(current.get("max_route_count", 0)), route_count)
            current["requests_bk32"] = bool(current.get("requests_bk32", False)) or bool(
                row.get("requests_bk32")
            )
            current["passes_bk64_rhs_nax"] = bool(
                current.get("passes_bk64_rhs_nax", False)
            ) or bool(row.get("passes_bk64_rhs_nax", row.get("has_allowed_gather_qmm_nax_bk64_symbol")))
            break
    return by_projection


def _common_q2_geometry(q2_geometry: dict[str, dict[str, Any]]) -> dict[str, int]:
    fields = ("group_size", "bits", "bm", "bn", "bk", "wm", "wn")
    common: dict[str, int] = {}
    for field in fields:
        values = {
            row.get(field)
            for row in q2_geometry.values()
            if isinstance(row.get(field), int)
        }
        if len(values) == 1:
            common[field] = int(next(iter(values)))
    return common


def _packed_rhs_requirements(
    *,
    q2_geometry: dict[str, dict[str, Any]],
    e8p_kernel: dict[str, Any],
    benchmark_context: dict[str, Any],
) -> dict[str, Any]:
    geometry = _common_q2_geometry(q2_geometry)
    direct_decode_gap = bool(e8p_kernel.get("decoded_b_threadgroup_staging")) and bool(
        e8p_kernel.get("loads_btile_from_threadgroup_ws")
    )
    benchmark_rejected = benchmark_context.get("all_parity_pass") is False or benchmark_context.get(
        "all_lane_s_pass"
    ) is False
    return {
        "target_kernel_family": "sorted_gather_qmm_rhs_nax",
        "storage_constraint": "compressed_e8p_codes_scales",
        "must_match_geometry": geometry,
        "requires_bk64_rhs_nax": geometry.get("bk") == 64,
        "requires_sorted_route_gather": True,
        "must_avoid_decoded_b_threadgroup_staging": direct_decode_gap,
        "benchmark_rejects_current_steel_family": benchmark_rejected,
        "reject_next_paths": [
            "direct_reduce_scalar_accumulation",
            "inline_b_decode_per_output_tile",
            "dense_or_predecoded_fp16_rhs",
        ],
        "next_kernel_requirement": (
            "Build an E8P-packed RHS tile layout that preserves compressed codes/scales, "
            "feeds a bk64 sorted gather-qmm RHS NAX TensorOps path, and avoids per-output-tile "
            "E8P decode or dense/predecoded FP16 RHS materialization."
        ),
    }


def _track_b_current_family_verdict(
    *,
    e8p_kernel: dict[str, Any],
    packed_rhs_tiled_family: dict[str, Any],
    shared_decode_kernel: dict[str, Any],
    shared_n_decode_kernel: dict[str, Any],
    sign_nibble_schedule_guardrail: dict[str, Any],
    sign_plane_schedule_guardrail: dict[str, Any],
    expert_kblock_schedule_guardrail: dict[str, Any],
    component_stream_speed_path_guardrail: dict[str, Any],
    component_stream_partial_reduction_guardrail: dict[str, Any],
    component_stream_shared_decode_guardrail: dict[str, Any],
    benchmark_context: dict[str, Any],
) -> dict[str, Any]:
    blocked_families: list[str] = []
    open_review_families: list[str] = []
    benchmarkable_families: list[str] = []
    family_decisions: dict[str, str] = {}

    def classify(
        family: str,
        decision: str,
        *,
        blocked: bool = False,
        benchmarkable: bool = False,
    ) -> None:
        family_decisions[family] = decision
        if benchmarkable:
            benchmarkable_families.append(family)
        elif blocked:
            blocked_families.append(family)
        else:
            open_review_families.append(family)

    steel_rejected = (
        bool(e8p_kernel.get("decoded_b_threadgroup_staging"))
        and bool(e8p_kernel.get("loads_btile_from_threadgroup_ws"))
        and (
            benchmark_context.get("all_parity_pass") is False
            or benchmark_context.get("all_lane_s_pass") is False
        )
    )
    classify(
        "steel",
        "reject_decoded_b_threadgroup_staging" if steel_rejected else "needs_manual_review",
        blocked=steel_rejected,
    )
    packed_decision = str(packed_rhs_tiled_family.get("decision", "unknown"))
    classify(
        "packed_rhs_tiled",
        packed_decision,
        blocked=packed_decision == "reject_decoded_b_staged_packed_rhs_family",
    )
    shared_decode_decision = str(shared_decode_kernel.get("decision", "unknown"))
    classify(
        "shared_decode",
        shared_decode_decision,
        blocked=shared_decode_decision
        == "reject_per_fragment_factor_reconstruction_shared_decode",
    )
    shared_n_decision = str(shared_n_decode_kernel.get("decision", "unknown"))
    classify(
        "shared_n_decode",
        shared_n_decision,
        blocked=shared_n_decision == "reject_lane_local_shared_n_decode",
    )

    sign_nibble_decision = str(sign_nibble_schedule_guardrail.get("decision", "unknown"))
    sign_nibble_blocked = sign_nibble_decision in {
        "missing_q2_shaped_sign_nibble_tensorops_schedule",
        "reject_decoded_b_staged_sign_nibble_schedule",
        "reject_per_fragment_nibble_tensorops_schedule",
        "reject_per_fragment_micro_lut_tensorops_schedule",
    }
    classify("sign_nibble", sign_nibble_decision, blocked=sign_nibble_blocked)

    sign_plane_decision = str(sign_plane_schedule_guardrail.get("decision", "unknown"))
    sign_plane_blocked = sign_plane_decision in {
        "reject_per_fragment_sign_plane_tensorops_schedule",
        "missing_sign_plane_tensorops_schedule",
    }
    classify("sign_plane", sign_plane_decision, blocked=sign_plane_blocked)

    expert_decision = str(expert_kblock_schedule_guardrail.get("decision", "unknown"))
    expert_blocked = expert_decision in {
        "current_expert_kblock_tensorops_speed_rejected_v2_missing",
        "reject_expert_kblock_tensorops_v2_near_duplicate",
        "reject_expert_kblock_tensorops_v2_invalid_source",
        "reject_expert_kblock_tensorops_v2_kblock_reduction_missing",
        "expert_kblock_tensorops_v2_partial_reduction_scaffold_scalar_body",
        "scalar_expert_kblock_consumer_present_tensorops_missing",
    }
    expert_benchmarkable = expert_decision in {
        "expert_kblock_tensorops_candidate_present_needs_benchmark",
        "expert_kblock_tensorops_v2_partial_reduction_scaffold_present",
    }
    classify(
        "expert_kblock_v2",
        expert_decision,
        blocked=expert_blocked,
        benchmarkable=expert_benchmarkable,
    )

    component_speed_decision = str(
        component_stream_speed_path_guardrail.get("decision", "unknown")
    )
    component_speed_blocked = component_speed_decision in {
        "reject_component_stream_tensorops_speed_path",
        "component_stream_scalar_oracle_speed_path_missing",
        "missing_component_stream_speed_path",
    }
    component_speed_benchmarkable = component_speed_decision in {
        "component_stream_tensorops_speed_path_candidate_present_needs_parity",
    }
    classify(
        "component_stream_tensorops",
        component_speed_decision,
        blocked=component_speed_blocked,
        benchmarkable=component_speed_benchmarkable,
    )

    component_partial_decision = str(
        component_stream_partial_reduction_guardrail.get("decision", "unknown")
    )
    component_partial_blocked = component_partial_decision in {
        "missing_component_stream_partial_reduction_scaffold",
        "component_stream_partial_reduction_scaffold_scalar_body",
    }
    component_partial_benchmarkable = component_partial_decision in {
        "component_stream_partial_reduction_tensorops_body_present",
    }
    classify(
        "component_stream_partial_reduction",
        component_partial_decision,
        blocked=component_partial_blocked,
        benchmarkable=component_partial_benchmarkable,
    )

    component_shared_decision = str(
        component_stream_shared_decode_guardrail.get("decision", "unknown")
    )
    component_shared_blocked = component_shared_decision in {
        "missing_component_stream_shared_decode_source",
        "reject_component_stream_shared_decode_per_fragment_reconstruction",
        "component_stream_shared_decode_scalar_cache_scaffold_present",
        "component_stream_shared_decode_source_needs_manual_review",
    }
    component_shared_benchmarkable = component_shared_decision in {
        "component_stream_shared_decode_candidate_present_needs_parity",
    }
    classify(
        "component_stream_shared_decode",
        component_shared_decision,
        blocked=component_shared_blocked,
        benchmarkable=component_shared_benchmarkable,
    )

    all_current_families_blocked = (
        len(blocked_families) == len(family_decisions)
        and not open_review_families
        and not benchmarkable_families
    )
    required_next_features = [
        "materially_different_rhs_layout_or_kernel_family",
        "avoid_decoded_b_threadgroup_staging",
        "avoid_per_fragment_factor_reconstruction",
        "avoid_lane_local_fragment_buffer",
        "preserve_compressed_rhs_storage",
        "preserve_codeword_scale_slots",
    ]
    rejected_next_steps = [
        "do_not_time_current_e8p16_families",
        "do_not_benchmark_scalar_v2_partial_scaffold",
        "do_not_copy_old_expert_kblock_tensorops_body_into_v2",
        "do_not_reopen_staged_or_lane_local_tensorops_variants",
        "do_not_retime_rejected_component_stream_tensorops",
        "do_not_benchmark_scalar_component_stream_scaffolds",
    ]
    return {
        "decision": (
            "current_track_b_family_not_benchmarkable"
            if all_current_families_blocked
            else "current_track_b_family_has_open_review_or_benchmark_candidate"
        ),
        "benchmarkable_candidate_present": bool(benchmarkable_families),
        "benchmarkable_families": benchmarkable_families,
        "open_review_family_count": len(open_review_families),
        "open_review_families": open_review_families,
        "blocked_families": blocked_families,
        "family_decisions": family_decisions,
        "required_next_features": required_next_features
        if all_current_families_blocked
        else [],
        "rejected_next_steps": rejected_next_steps if all_current_families_blocked else [],
    }


def _benchmark_context(e8p_analysis_json: Path) -> dict[str, Any]:
    data = json.loads(e8p_analysis_json.read_text(encoding="utf-8"))
    down_summary = (data.get("projection_summaries") or {}).get("down") or {}
    expert_kblock_requirements = data.get("expert_kblock_schedule_requirements") or {}
    if not isinstance(expert_kblock_requirements, dict):
        expert_kblock_requirements = {}

    def variant_rows(variant: str) -> list[dict[str, Any]]:
        return [
            row
            for row in data.get("comparisons", [])
            if isinstance(row, dict)
            and (
                row.get("candidate_variant") == variant
                or row.get("candidate_diagnostic_phase") == variant
            )
        ]

    def row_stats(prefix: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        ratios = [
            row.get("ratio_to_q2")
            for row in rows
            if isinstance(row.get("ratio_to_q2"), (int, float))
        ]
        ms_per_iter = [
            row.get("candidate_ms_per_iter")
            for row in rows
            if isinstance(row.get("candidate_ms_per_iter"), (int, float))
        ]
        return {
            f"{prefix}_comparison_count": len(rows),
            f"{prefix}_best_ratio_to_q2": min(ratios) if ratios else None,
            f"{prefix}_worst_ratio_to_q2": max(ratios) if ratios else None,
            f"{prefix}_best_ms_per_iter": min(ms_per_iter) if ms_per_iter else None,
            f"{prefix}_worst_ms_per_iter": max(ms_per_iter) if ms_per_iter else None,
        }

    sign_nibble_tensorops_rows = variant_rows(
        "nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_raw"
    )
    sign_nibble_micro_lut_tensorops_rows = variant_rows(
        "nax_e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_raw"
    )
    sign_plane_tensorops_rows = variant_rows(
        "nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_raw"
    )
    expert_kblock_tensorops_rows = variant_rows(
        "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_raw"
    )
    expert_kblock_native_rows = variant_rows(
        "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_native_raw"
    )
    component_stream_tensorops_rows = variant_rows(
        "nax_e8p_component_stream_rhs_sorted_tensorops_raw"
    )
    context = {
        "analysis_json": str(e8p_analysis_json),
        "down_best_ratio_to_q2": down_summary.get("best_ratio_to_q2"),
        "down_best_variant": down_summary.get("best_variant"),
        "all_parity_pass": data.get("all_parity_pass"),
        "all_lane_s_pass": data.get("all_lane_s_pass"),
        "lane_s_ratio": data.get("lane_s_ratio"),
        "parity_ratio": data.get("parity_ratio"),
        "expert_kblock_schedule_decision": expert_kblock_requirements.get("decision"),
        "expert_kblock_factor_signal": expert_kblock_requirements.get("factor_signal"),
        "expert_kblock_required_kernel_features": expert_kblock_requirements.get(
            "required_kernel_features"
        )
        or [],
    }
    context.update(row_stats("sign_nibble_tensorops", sign_nibble_tensorops_rows))
    context.update(
        row_stats(
            "sign_nibble_micro_lut_tensorops",
            sign_nibble_micro_lut_tensorops_rows,
        )
    )
    context.update(row_stats("sign_plane_tensorops", sign_plane_tensorops_rows))
    context.update(row_stats("expert_kblock_tensorops", expert_kblock_tensorops_rows))
    context.update(row_stats("expert_kblock_native", expert_kblock_native_rows))
    context.update(row_stats("component_stream_tensorops", component_stream_tensorops_rows))
    return context


def build_structure_report(
    *,
    metal_source: Path,
    primitive_source: Path | None = None,
    q2_jsonl: Path,
    e8p_analysis_json: Path,
    q2_geometry_jsonl: Path | None = None,
    active_route_tile_design_json: Path | None = None,
    expert_cohort_design_json: Path | None = None,
    route_batch_segmented_design_json: Path | None = None,
    component_stream_partial_reduction_design_json: Path | None = None,
    token_cohort_design_json: Path | None = None,
    token_cohort_mma_codeword_tile_design_json: Path | None = None,
    output_stationary_codeword_tile_design_json: Path | None = None,
    input_stationary_codeword_tile_design_json: Path | None = None,
    expert_kblock_codeword_factor_reuse_design_json: Path | None = None,
    route_codeword_lut_accumulate_design_json: Path | None = None,
    rowwise_codeword_tile_accumulate_design_json: Path | None = None,
    output_tile_local_codeword_lut_design_json: Path | None = None,
    route_microtile_codeword_block_reduce_design_json: Path | None = None,
    kblock_wavefront_codeword_scan_design_json: Path | None = None,
    token_route_output_stripe_pipeline_design_json: Path | None = None,
    expert_kblock_scale_slot_stream_design_json: Path | None = None,
    scale_group_route_block_reduce_design_json: Path | None = None,
    route_block_output_group_stream_design_json: Path | None = None,
    output_group_pretransposed_codeword_stream_design_json: Path | None = None,
    kblock_output_group_route_fused_stream_design_json: Path | None = None,
    route_tile_output_swizzle_stream_design_json: Path | None = None,
    token_topk_output_tile_stream_design_json: Path | None = None,
    token_block_output_group_stream_design_json: Path | None = None,
    token_output_stripe_group_stream_design_json: Path | None = None,
    token_expert_output_block_stream_design_json: Path | None = None,
    token_pair_kblock_accumulator_stream_design_json: Path | None = None,
    token_pair_output_group_stream_design_json: Path | None = None,
    token_pair_slot_topk_output_group_stream_design_json: Path | None = None,
    token_pair_slot_topk_codeword_group_pipeline_design_json: Path | None = None,
    token_pair_slot_topk_scale_slot_broadcast_stream_design_json: Path | None = None,
    token_pair_slot_topk_route_bucket_codeword_reduce_design_json: (
        Path | None
    ) = None,
    token_pair_slot_topk_kblock_microtile_stream_design_json: Path | None = None,
    token_pair_slot_topk_output_tile_fused_stream_design_json: Path | None = None,
    kernel_name: str = "nax_e8p_fp16_sorted_matmul_steel",
) -> dict[str, Any]:
    source = metal_source.read_text(encoding="utf-8")
    e8p_kernel = audit_e8p_kernel_source(
        source,
        kernel_name=kernel_name,
    )
    try:
        shared_decode_kernel = audit_e8p_shared_decode_kernel_source(source)
    except ValueError:
        shared_decode_kernel = {
            "kernel_name": "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_matmul",
            "present": False,
        }
    try:
        shared_n_decode_kernel = audit_e8p_shared_n_decode_kernel_source(source)
    except ValueError:
        shared_n_decode_kernel = {
            "kernel_name": "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_matmul",
            "present": False,
        }
    packed_rhs_tiled_family = audit_e8p_packed_rhs_tiled_family_source(source)
    q2_geometry_source = q2_geometry_jsonl or q2_jsonl
    q2_geometry = _q2_geometry_by_projection(q2_geometry_source)
    context = _benchmark_context(e8p_analysis_json)
    active_route_tile_design: dict[str, Any] | None = None
    if active_route_tile_design_json is not None and active_route_tile_design_json.is_file():
        active_route_tile_design = json.loads(
            active_route_tile_design_json.read_text(encoding="utf-8")
        )
        if not isinstance(active_route_tile_design, dict):
            raise ValueError(f"{active_route_tile_design_json} must contain a JSON object")
    active_route_tile_design_ready = (
        active_route_tile_design is not None
        and active_route_tile_design.get("decision")
        == "active_route_tile_codeword_outer_product_ready_for_source_structure_probe"
        and active_route_tile_design.get("selector_verdict", {}).get(
            "candidate_ready_for_source_probe"
        )
        is True
    )
    expert_cohort_design: dict[str, Any] | None = None
    if expert_cohort_design_json is not None and expert_cohort_design_json.is_file():
        expert_cohort_design = json.loads(
            expert_cohort_design_json.read_text(encoding="utf-8")
        )
        if not isinstance(expert_cohort_design, dict):
            raise ValueError(f"{expert_cohort_design_json} must contain a JSON object")
    expert_cohort_design_ready = (
        expert_cohort_design is not None
        and expert_cohort_design.get("decision")
        == "expert_cohort_codeword_broadcast_ready_for_source_structure_probe"
        and expert_cohort_design.get("selector_verdict", {}).get(
            "candidate_ready_for_source_probe"
        )
        is True
    )
    route_batch_segmented_design: dict[str, Any] | None = None
    if (
        route_batch_segmented_design_json is not None
        and route_batch_segmented_design_json.is_file()
    ):
        route_batch_segmented_design = json.loads(
            route_batch_segmented_design_json.read_text(encoding="utf-8")
        )
        if not isinstance(route_batch_segmented_design, dict):
            raise ValueError(
                f"{route_batch_segmented_design_json} must contain a JSON object"
            )
    route_batch_segmented_design_ready = (
        route_batch_segmented_design is not None
        and route_batch_segmented_design.get("decision")
        == "route_batch_segmented_codeword_reduce_ready_for_source_structure_probe"
        and route_batch_segmented_design.get("selector_verdict", {}).get(
            "candidate_ready_for_source_probe"
        )
        is True
    )
    component_stream_partial_reduction_design: dict[str, Any] | None = None
    if (
        component_stream_partial_reduction_design_json is not None
        and component_stream_partial_reduction_design_json.is_file()
    ):
        component_stream_partial_reduction_design = json.loads(
            component_stream_partial_reduction_design_json.read_text(encoding="utf-8")
        )
        if not isinstance(component_stream_partial_reduction_design, dict):
            raise ValueError(
                f"{component_stream_partial_reduction_design_json} must contain a JSON object"
            )
    component_stream_partial_reduction_design_ready = (
        component_stream_partial_reduction_design is not None
        and component_stream_partial_reduction_design.get("decision")
        == "component_stream_partial_reduction_ready_for_source_structure_probe"
        and component_stream_partial_reduction_design.get("selector_verdict", {}).get(
            "candidate_ready_for_source_probe"
        )
        is True
    )
    token_cohort_design: dict[str, Any] | None = None
    if token_cohort_design_json is not None and token_cohort_design_json.is_file():
        token_cohort_design = json.loads(
            token_cohort_design_json.read_text(encoding="utf-8")
        )
        if not isinstance(token_cohort_design, dict):
            raise ValueError(f"{token_cohort_design_json} must contain a JSON object")
    token_cohort_design_ready = (
        token_cohort_design is not None
        and token_cohort_design.get("decision")
        == "token_cohort_codeword_stream_ready_for_source_structure_probe"
        and token_cohort_design.get("selector_verdict", {}).get(
            "candidate_ready_for_source_probe"
        )
        is True
    )
    token_cohort_mma_codeword_tile_design: dict[str, Any] | None = None
    if (
        token_cohort_mma_codeword_tile_design_json is not None
        and token_cohort_mma_codeword_tile_design_json.is_file()
    ):
        token_cohort_mma_codeword_tile_design = json.loads(
            token_cohort_mma_codeword_tile_design_json.read_text(encoding="utf-8")
        )
        if not isinstance(token_cohort_mma_codeword_tile_design, dict):
            raise ValueError(
                f"{token_cohort_mma_codeword_tile_design_json} must contain a JSON object"
            )
    token_cohort_mma_codeword_tile_design_ready = (
        token_cohort_mma_codeword_tile_design is not None
        and token_cohort_mma_codeword_tile_design.get("decision")
        == "token_cohort_mma_codeword_tile_ready_for_source_structure_probe"
        and token_cohort_mma_codeword_tile_design.get("selector_verdict", {}).get(
            "candidate_ready_for_source_probe"
        )
        is True
    )
    output_stationary_codeword_tile_design: dict[str, Any] | None = None
    if (
        output_stationary_codeword_tile_design_json is not None
        and output_stationary_codeword_tile_design_json.is_file()
    ):
        output_stationary_codeword_tile_design = json.loads(
            output_stationary_codeword_tile_design_json.read_text(encoding="utf-8")
        )
        if not isinstance(output_stationary_codeword_tile_design, dict):
            raise ValueError(
                f"{output_stationary_codeword_tile_design_json} must contain a JSON object"
            )
    output_stationary_codeword_tile_design_ready = (
        output_stationary_codeword_tile_design is not None
        and output_stationary_codeword_tile_design.get("decision")
        == "output_stationary_codeword_tile_ready_for_source_structure_probe"
        and output_stationary_codeword_tile_design.get("selector_verdict", {}).get(
            "candidate_ready_for_source_probe"
        )
        is True
    )
    input_stationary_codeword_tile_design: dict[str, Any] | None = None
    if (
        input_stationary_codeword_tile_design_json is not None
        and input_stationary_codeword_tile_design_json.is_file()
    ):
        input_stationary_codeword_tile_design = json.loads(
            input_stationary_codeword_tile_design_json.read_text(encoding="utf-8")
        )
        if not isinstance(input_stationary_codeword_tile_design, dict):
            raise ValueError(
                f"{input_stationary_codeword_tile_design_json} must contain a JSON object"
            )
    input_stationary_codeword_tile_design_ready = (
        input_stationary_codeword_tile_design is not None
        and input_stationary_codeword_tile_design.get("decision")
        == "input_stationary_codeword_tile_ready_for_source_structure_probe"
        and input_stationary_codeword_tile_design.get("selector_verdict", {}).get(
            "candidate_ready_for_source_probe"
        )
        is True
    )
    expert_kblock_codeword_factor_reuse_design: dict[str, Any] | None = None
    if (
        expert_kblock_codeword_factor_reuse_design_json is not None
        and expert_kblock_codeword_factor_reuse_design_json.is_file()
    ):
        expert_kblock_codeword_factor_reuse_design = json.loads(
            expert_kblock_codeword_factor_reuse_design_json.read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(expert_kblock_codeword_factor_reuse_design, dict):
            raise ValueError(
                f"{expert_kblock_codeword_factor_reuse_design_json} must contain a JSON object"
            )
    expert_kblock_codeword_factor_reuse_design_ready = (
        expert_kblock_codeword_factor_reuse_design is not None
        and expert_kblock_codeword_factor_reuse_design.get("decision")
        == "expert_kblock_codeword_factor_reuse_ready_for_source_structure_probe"
        and expert_kblock_codeword_factor_reuse_design.get("selector_verdict", {}).get(
            "candidate_ready_for_source_probe"
        )
        is True
    )
    route_codeword_lut_accumulate_design: dict[str, Any] | None = None
    if (
        route_codeword_lut_accumulate_design_json is not None
        and route_codeword_lut_accumulate_design_json.is_file()
    ):
        route_codeword_lut_accumulate_design = json.loads(
            route_codeword_lut_accumulate_design_json.read_text(encoding="utf-8")
        )
        if not isinstance(route_codeword_lut_accumulate_design, dict):
            raise ValueError(
                f"{route_codeword_lut_accumulate_design_json} must contain a JSON object"
            )
    route_codeword_lut_accumulate_design_ready = (
        route_codeword_lut_accumulate_design is not None
        and route_codeword_lut_accumulate_design.get("decision")
        == "route_codeword_lut_accumulate_ready_for_source_structure_probe"
        and route_codeword_lut_accumulate_design.get("selector_verdict", {}).get(
            "candidate_ready_for_source_probe"
        )
        is True
    )
    rowwise_codeword_tile_accumulate_design: dict[str, Any] | None = None
    if (
        rowwise_codeword_tile_accumulate_design_json is not None
        and rowwise_codeword_tile_accumulate_design_json.is_file()
    ):
        rowwise_codeword_tile_accumulate_design = json.loads(
            rowwise_codeword_tile_accumulate_design_json.read_text(encoding="utf-8")
        )
        if not isinstance(rowwise_codeword_tile_accumulate_design, dict):
            raise ValueError(
                f"{rowwise_codeword_tile_accumulate_design_json} must contain a JSON object"
            )
    rowwise_codeword_tile_accumulate_design_ready = (
        rowwise_codeword_tile_accumulate_design is not None
        and rowwise_codeword_tile_accumulate_design.get("decision")
        == "rowwise_codeword_tile_accumulate_ready_for_source_structure_probe"
        and rowwise_codeword_tile_accumulate_design.get("selector_verdict", {}).get(
            "candidate_ready_for_source_probe"
        )
        is True
    )
    output_tile_local_codeword_lut_design: dict[str, Any] | None = None
    if (
        output_tile_local_codeword_lut_design_json is not None
        and output_tile_local_codeword_lut_design_json.is_file()
    ):
        output_tile_local_codeword_lut_design = json.loads(
            output_tile_local_codeword_lut_design_json.read_text(encoding="utf-8")
        )
        if not isinstance(output_tile_local_codeword_lut_design, dict):
            raise ValueError(
                f"{output_tile_local_codeword_lut_design_json} must contain a JSON object"
            )
    output_tile_local_codeword_lut_design_ready = (
        output_tile_local_codeword_lut_design is not None
        and output_tile_local_codeword_lut_design.get("decision")
        == "output_tile_local_codeword_lut_ready_for_source_structure_probe"
        and output_tile_local_codeword_lut_design.get("selector_verdict", {}).get(
            "candidate_ready_for_source_probe"
        )
        is True
    )
    route_microtile_codeword_block_reduce_design: dict[str, Any] | None = None
    if (
        route_microtile_codeword_block_reduce_design_json is not None
        and route_microtile_codeword_block_reduce_design_json.is_file()
    ):
        route_microtile_codeword_block_reduce_design = json.loads(
            route_microtile_codeword_block_reduce_design_json.read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(route_microtile_codeword_block_reduce_design, dict):
            raise ValueError(
                f"{route_microtile_codeword_block_reduce_design_json} must contain a JSON object"
            )
    route_microtile_codeword_block_reduce_design_ready = (
        route_microtile_codeword_block_reduce_design is not None
        and route_microtile_codeword_block_reduce_design.get("decision")
        == "route_microtile_codeword_block_reduce_ready_for_source_structure_probe"
        and route_microtile_codeword_block_reduce_design.get(
            "selector_verdict", {}
        ).get("candidate_ready_for_source_probe")
        is True
    )
    kblock_wavefront_codeword_scan_design: dict[str, Any] | None = None
    if (
        kblock_wavefront_codeword_scan_design_json is not None
        and kblock_wavefront_codeword_scan_design_json.is_file()
    ):
        kblock_wavefront_codeword_scan_design = json.loads(
            kblock_wavefront_codeword_scan_design_json.read_text(encoding="utf-8")
        )
        if not isinstance(kblock_wavefront_codeword_scan_design, dict):
            raise ValueError(
                f"{kblock_wavefront_codeword_scan_design_json} must contain a JSON object"
            )
    kblock_wavefront_codeword_scan_design_ready = (
        kblock_wavefront_codeword_scan_design is not None
        and kblock_wavefront_codeword_scan_design.get("decision")
        == "kblock_wavefront_codeword_scan_ready_for_source_structure_probe"
        and kblock_wavefront_codeword_scan_design.get("selector_verdict", {}).get(
            "candidate_ready_for_source_probe"
        )
        is True
    )
    token_route_output_stripe_pipeline_design: dict[str, Any] | None = None
    if (
        token_route_output_stripe_pipeline_design_json is not None
        and token_route_output_stripe_pipeline_design_json.is_file()
    ):
        token_route_output_stripe_pipeline_design = json.loads(
            token_route_output_stripe_pipeline_design_json.read_text(encoding="utf-8")
        )
        if not isinstance(token_route_output_stripe_pipeline_design, dict):
            raise ValueError(
                f"{token_route_output_stripe_pipeline_design_json} must contain a JSON object"
            )
    token_route_output_stripe_pipeline_design_ready = (
        token_route_output_stripe_pipeline_design is not None
        and token_route_output_stripe_pipeline_design.get("decision")
        == "token_route_output_stripe_pipeline_ready_for_source_structure_probe"
        and token_route_output_stripe_pipeline_design.get("selector_verdict", {}).get(
            "candidate_ready_for_source_probe"
        )
        is True
    )
    expert_kblock_scale_slot_stream_design: dict[str, Any] | None = None
    if (
        expert_kblock_scale_slot_stream_design_json is not None
        and expert_kblock_scale_slot_stream_design_json.is_file()
    ):
        expert_kblock_scale_slot_stream_design = json.loads(
            expert_kblock_scale_slot_stream_design_json.read_text(encoding="utf-8")
        )
        if not isinstance(expert_kblock_scale_slot_stream_design, dict):
            raise ValueError(
                f"{expert_kblock_scale_slot_stream_design_json} must contain a JSON object"
            )
    expert_kblock_scale_slot_stream_design_ready = (
        expert_kblock_scale_slot_stream_design is not None
        and expert_kblock_scale_slot_stream_design.get("decision")
        == "expert_kblock_scale_slot_stream_ready_for_source_structure_probe"
        and expert_kblock_scale_slot_stream_design.get("selector_verdict", {}).get(
            "candidate_ready_for_source_probe"
        )
        is True
    )
    scale_group_route_block_reduce_design: dict[str, Any] | None = None
    if (
        scale_group_route_block_reduce_design_json is not None
        and scale_group_route_block_reduce_design_json.is_file()
    ):
        scale_group_route_block_reduce_design = json.loads(
            scale_group_route_block_reduce_design_json.read_text(encoding="utf-8")
        )
        if not isinstance(scale_group_route_block_reduce_design, dict):
            raise ValueError(
                f"{scale_group_route_block_reduce_design_json} must contain a JSON object"
            )
    scale_group_route_block_reduce_design_ready = (
        scale_group_route_block_reduce_design is not None
        and scale_group_route_block_reduce_design.get("decision")
        == "scale_group_route_block_reduce_ready_for_source_structure_probe"
        and scale_group_route_block_reduce_design.get("selector_verdict", {}).get(
            "candidate_ready_for_source_probe"
        )
        is True
    )
    route_block_output_group_stream_design: dict[str, Any] | None = None
    if (
        route_block_output_group_stream_design_json is not None
        and route_block_output_group_stream_design_json.is_file()
    ):
        route_block_output_group_stream_design = json.loads(
            route_block_output_group_stream_design_json.read_text(encoding="utf-8")
        )
        if not isinstance(route_block_output_group_stream_design, dict):
            raise ValueError(
                f"{route_block_output_group_stream_design_json} must contain a JSON object"
            )
    route_block_output_group_stream_design_ready = (
        route_block_output_group_stream_design is not None
        and route_block_output_group_stream_design.get("decision")
        == "route_block_output_group_stream_ready_for_source_structure_probe"
        and route_block_output_group_stream_design.get("selector_verdict", {}).get(
            "candidate_ready_for_source_probe"
        )
        is True
    )
    output_group_pretransposed_codeword_stream_design: dict[str, Any] | None = None
    if (
        output_group_pretransposed_codeword_stream_design_json is not None
        and output_group_pretransposed_codeword_stream_design_json.is_file()
    ):
        output_group_pretransposed_codeword_stream_design = json.loads(
            output_group_pretransposed_codeword_stream_design_json.read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(output_group_pretransposed_codeword_stream_design, dict):
            raise ValueError(
                f"{output_group_pretransposed_codeword_stream_design_json} must contain a JSON object"
            )
    output_group_pretransposed_codeword_stream_design_ready = (
        output_group_pretransposed_codeword_stream_design is not None
        and output_group_pretransposed_codeword_stream_design.get("decision")
        == "output_group_pretransposed_codeword_stream_ready_for_source_structure_probe"
        and output_group_pretransposed_codeword_stream_design.get(
            "selector_verdict", {}
        ).get("candidate_ready_for_source_probe")
        is True
    )
    kblock_output_group_route_fused_stream_design: dict[str, Any] | None = None
    if (
        kblock_output_group_route_fused_stream_design_json is not None
        and kblock_output_group_route_fused_stream_design_json.is_file()
    ):
        kblock_output_group_route_fused_stream_design = json.loads(
            kblock_output_group_route_fused_stream_design_json.read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(kblock_output_group_route_fused_stream_design, dict):
            raise ValueError(
                f"{kblock_output_group_route_fused_stream_design_json} must contain a JSON object"
            )
    kblock_output_group_route_fused_stream_design_ready = (
        kblock_output_group_route_fused_stream_design is not None
        and kblock_output_group_route_fused_stream_design.get("decision")
        == "kblock_output_group_route_fused_stream_ready_for_source_structure_probe"
        and kblock_output_group_route_fused_stream_design.get(
            "selector_verdict", {}
        ).get("candidate_ready_for_source_probe")
        is True
    )
    route_tile_output_swizzle_stream_design: dict[str, Any] | None = None
    if (
        route_tile_output_swizzle_stream_design_json is not None
        and route_tile_output_swizzle_stream_design_json.is_file()
    ):
        route_tile_output_swizzle_stream_design = json.loads(
            route_tile_output_swizzle_stream_design_json.read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(route_tile_output_swizzle_stream_design, dict):
            raise ValueError(
                f"{route_tile_output_swizzle_stream_design_json} must contain a JSON object"
            )
    route_tile_output_swizzle_stream_design_ready = (
        route_tile_output_swizzle_stream_design is not None
        and route_tile_output_swizzle_stream_design.get("decision")
        == "route_tile_output_swizzle_stream_ready_for_source_structure_probe"
        and route_tile_output_swizzle_stream_design.get(
            "selector_verdict", {}
        ).get("candidate_ready_for_source_probe")
        is True
    )
    token_topk_output_tile_stream_design: dict[str, Any] | None = None
    if (
        token_topk_output_tile_stream_design_json is not None
        and token_topk_output_tile_stream_design_json.is_file()
    ):
        token_topk_output_tile_stream_design = json.loads(
            token_topk_output_tile_stream_design_json.read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(token_topk_output_tile_stream_design, dict):
            raise ValueError(
                f"{token_topk_output_tile_stream_design_json} must contain a JSON object"
            )
    token_topk_output_tile_stream_design_ready = (
        token_topk_output_tile_stream_design is not None
        and token_topk_output_tile_stream_design.get("decision")
        == "token_topk_output_tile_stream_ready_for_source_structure_probe"
        and token_topk_output_tile_stream_design.get("selector_verdict", {}).get(
            "candidate_ready_for_source_probe"
        )
        is True
    )
    token_block_output_group_stream_design: dict[str, Any] | None = None
    if (
        token_block_output_group_stream_design_json is not None
        and token_block_output_group_stream_design_json.is_file()
    ):
        token_block_output_group_stream_design = json.loads(
            token_block_output_group_stream_design_json.read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(token_block_output_group_stream_design, dict):
            raise ValueError(
                f"{token_block_output_group_stream_design_json} must contain a JSON object"
            )
    token_block_output_group_stream_design_ready = (
        token_block_output_group_stream_design is not None
        and token_block_output_group_stream_design.get("decision")
        == "token_block_output_group_stream_ready_for_source_structure_probe"
        and token_block_output_group_stream_design.get("selector_verdict", {}).get(
            "candidate_ready_for_source_probe"
        )
        is True
    )
    token_output_stripe_group_stream_design: dict[str, Any] | None = None
    if (
        token_output_stripe_group_stream_design_json is not None
        and token_output_stripe_group_stream_design_json.is_file()
    ):
        token_output_stripe_group_stream_design = json.loads(
            token_output_stripe_group_stream_design_json.read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(token_output_stripe_group_stream_design, dict):
            raise ValueError(
                f"{token_output_stripe_group_stream_design_json} must contain a JSON object"
            )
    token_output_stripe_group_stream_design_ready = (
        token_output_stripe_group_stream_design is not None
        and token_output_stripe_group_stream_design.get("decision")
        == "token_output_stripe_group_stream_ready_for_source_structure_probe"
        and token_output_stripe_group_stream_design.get(
            "selector_verdict", {}
        ).get("candidate_ready_for_source_probe")
        is True
    )
    token_expert_output_block_stream_design: dict[str, Any] | None = None
    if (
        token_expert_output_block_stream_design_json is not None
        and token_expert_output_block_stream_design_json.is_file()
    ):
        token_expert_output_block_stream_design = json.loads(
            token_expert_output_block_stream_design_json.read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(token_expert_output_block_stream_design, dict):
            raise ValueError(
                f"{token_expert_output_block_stream_design_json} must contain a JSON object"
            )
    token_expert_output_block_stream_design_ready = (
        token_expert_output_block_stream_design is not None
        and token_expert_output_block_stream_design.get("decision")
        == "token_expert_output_block_stream_ready_for_source_structure_probe"
        and token_expert_output_block_stream_design.get(
            "selector_verdict", {}
        ).get("candidate_ready_for_source_probe")
        is True
    )
    token_pair_kblock_accumulator_stream_design: dict[str, Any] | None = None
    if (
        token_pair_kblock_accumulator_stream_design_json is not None
        and token_pair_kblock_accumulator_stream_design_json.is_file()
    ):
        token_pair_kblock_accumulator_stream_design = json.loads(
            token_pair_kblock_accumulator_stream_design_json.read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(token_pair_kblock_accumulator_stream_design, dict):
            raise ValueError(
                f"{token_pair_kblock_accumulator_stream_design_json} must contain a JSON object"
            )
    token_pair_kblock_accumulator_stream_design_ready = (
        token_pair_kblock_accumulator_stream_design is not None
        and token_pair_kblock_accumulator_stream_design.get("decision")
        == "token_pair_kblock_accumulator_stream_ready_for_source_structure_probe"
        and token_pair_kblock_accumulator_stream_design.get(
            "selector_verdict", {}
        ).get("candidate_ready_for_source_probe")
        is True
    )
    token_pair_output_group_stream_design: dict[str, Any] | None = None
    if (
        token_pair_output_group_stream_design_json is not None
        and token_pair_output_group_stream_design_json.is_file()
    ):
        token_pair_output_group_stream_design = json.loads(
            token_pair_output_group_stream_design_json.read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(token_pair_output_group_stream_design, dict):
            raise ValueError(
                f"{token_pair_output_group_stream_design_json} must contain a JSON object"
            )
    token_pair_output_group_stream_design_ready = (
        token_pair_output_group_stream_design is not None
        and token_pair_output_group_stream_design.get("decision")
        == "token_pair_output_group_stream_ready_for_source_structure_probe"
        and token_pair_output_group_stream_design.get("selector_verdict", {}).get(
            "candidate_ready_for_source_probe"
        )
        is True
    )
    token_pair_slot_topk_output_group_stream_design: dict[str, Any] | None = None
    if (
        token_pair_slot_topk_output_group_stream_design_json is not None
        and token_pair_slot_topk_output_group_stream_design_json.is_file()
    ):
        token_pair_slot_topk_output_group_stream_design = json.loads(
            token_pair_slot_topk_output_group_stream_design_json.read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(token_pair_slot_topk_output_group_stream_design, dict):
            raise ValueError(
                f"{token_pair_slot_topk_output_group_stream_design_json} must contain a JSON object"
            )
    token_pair_slot_topk_output_group_stream_design_ready = (
        token_pair_slot_topk_output_group_stream_design is not None
        and token_pair_slot_topk_output_group_stream_design.get("decision")
        == "token_pair_slot_topk_output_group_stream_ready_for_source_structure_probe"
        and token_pair_slot_topk_output_group_stream_design.get(
            "selector_verdict", {}
        ).get("candidate_ready_for_source_probe")
        is True
    )
    token_pair_slot_topk_codeword_group_pipeline_design: dict[str, Any] | None = None
    if (
        token_pair_slot_topk_codeword_group_pipeline_design_json is not None
        and token_pair_slot_topk_codeword_group_pipeline_design_json.is_file()
    ):
        token_pair_slot_topk_codeword_group_pipeline_design = json.loads(
            token_pair_slot_topk_codeword_group_pipeline_design_json.read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(
            token_pair_slot_topk_codeword_group_pipeline_design, dict
        ):
            raise ValueError(
                f"{token_pair_slot_topk_codeword_group_pipeline_design_json} must contain a JSON object"
            )
    token_pair_slot_topk_codeword_group_pipeline_design_ready = (
        token_pair_slot_topk_codeword_group_pipeline_design is not None
        and token_pair_slot_topk_codeword_group_pipeline_design.get("decision")
        == "token_pair_slot_topk_codeword_group_pipeline_ready_for_source_structure_probe"
        and token_pair_slot_topk_codeword_group_pipeline_design.get(
            "selector_verdict", {}
        ).get("candidate_ready_for_source_probe")
        is True
    )
    token_pair_slot_topk_scale_slot_broadcast_stream_design: (
        dict[str, Any] | None
    ) = None
    if (
        token_pair_slot_topk_scale_slot_broadcast_stream_design_json is not None
        and token_pair_slot_topk_scale_slot_broadcast_stream_design_json.is_file()
    ):
        token_pair_slot_topk_scale_slot_broadcast_stream_design = json.loads(
            token_pair_slot_topk_scale_slot_broadcast_stream_design_json.read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(
            token_pair_slot_topk_scale_slot_broadcast_stream_design, dict
        ):
            raise ValueError(
                f"{token_pair_slot_topk_scale_slot_broadcast_stream_design_json} must contain a JSON object"
            )
    token_pair_slot_topk_scale_slot_broadcast_stream_design_ready = (
        token_pair_slot_topk_scale_slot_broadcast_stream_design is not None
        and token_pair_slot_topk_scale_slot_broadcast_stream_design.get("decision")
        == "token_pair_slot_topk_scale_slot_broadcast_stream_ready_for_source_structure_probe"
        and token_pair_slot_topk_scale_slot_broadcast_stream_design.get(
            "selector_verdict", {}
        ).get("candidate_ready_for_source_probe")
        is True
    )
    token_pair_slot_topk_route_bucket_codeword_reduce_design: (
        dict[str, Any] | None
    ) = None
    if (
        token_pair_slot_topk_route_bucket_codeword_reduce_design_json is not None
        and token_pair_slot_topk_route_bucket_codeword_reduce_design_json.is_file()
    ):
        token_pair_slot_topk_route_bucket_codeword_reduce_design = json.loads(
            token_pair_slot_topk_route_bucket_codeword_reduce_design_json.read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(
            token_pair_slot_topk_route_bucket_codeword_reduce_design, dict
        ):
            raise ValueError(
                f"{token_pair_slot_topk_route_bucket_codeword_reduce_design_json} must contain a JSON object"
            )
    token_pair_slot_topk_route_bucket_codeword_reduce_design_ready = (
        token_pair_slot_topk_route_bucket_codeword_reduce_design is not None
        and token_pair_slot_topk_route_bucket_codeword_reduce_design.get("decision")
        == "token_pair_slot_topk_route_bucket_codeword_reduce_ready_for_source_structure_probe"
        and token_pair_slot_topk_route_bucket_codeword_reduce_design.get(
            "selector_verdict", {}
        ).get("candidate_ready_for_source_probe")
        is True
    )
    token_pair_slot_topk_kblock_microtile_stream_design: dict[str, Any] | None = None
    if (
        token_pair_slot_topk_kblock_microtile_stream_design_json is not None
        and token_pair_slot_topk_kblock_microtile_stream_design_json.is_file()
    ):
        token_pair_slot_topk_kblock_microtile_stream_design = json.loads(
            token_pair_slot_topk_kblock_microtile_stream_design_json.read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(token_pair_slot_topk_kblock_microtile_stream_design, dict):
            raise ValueError(
                f"{token_pair_slot_topk_kblock_microtile_stream_design_json} must contain a JSON object"
            )
    token_pair_slot_topk_kblock_microtile_stream_design_ready = (
        token_pair_slot_topk_kblock_microtile_stream_design is not None
        and token_pair_slot_topk_kblock_microtile_stream_design.get("decision")
        == "token_pair_slot_topk_kblock_microtile_stream_ready_for_source_structure_probe"
        and token_pair_slot_topk_kblock_microtile_stream_design.get(
            "selector_verdict", {}
        ).get("candidate_ready_for_source_probe")
        is True
    )
    token_pair_slot_topk_output_tile_fused_stream_design: dict[str, Any] | None = None
    if (
        token_pair_slot_topk_output_tile_fused_stream_design_json is not None
        and token_pair_slot_topk_output_tile_fused_stream_design_json.is_file()
    ):
        token_pair_slot_topk_output_tile_fused_stream_design = json.loads(
            token_pair_slot_topk_output_tile_fused_stream_design_json.read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(
            token_pair_slot_topk_output_tile_fused_stream_design, dict
        ):
            raise ValueError(
                f"{token_pair_slot_topk_output_tile_fused_stream_design_json} must contain a JSON object"
            )
    token_pair_slot_topk_output_tile_fused_stream_design_ready = (
        token_pair_slot_topk_output_tile_fused_stream_design is not None
        and token_pair_slot_topk_output_tile_fused_stream_design.get("decision")
        == "token_pair_slot_topk_output_tile_fused_stream_ready_for_source_structure_probe"
        and token_pair_slot_topk_output_tile_fused_stream_design.get(
            "selector_verdict", {}
        ).get("candidate_ready_for_source_probe")
        is True
    )
    sign_nibble_schedule_guardrail = audit_e8p_sign_nibble_schedule_source(
        source,
        benchmark_context=context,
    )
    sign_plane_schedule_guardrail = audit_e8p_sign_plane_schedule_source(
        source,
        benchmark_context=context,
    )
    expert_kblock_schedule_guardrail = audit_e8p_expert_kblock_schedule_source(
        source,
        benchmark_context=context,
    )
    if primitive_source is not None:
        primitive_source_text = primitive_source.read_text(encoding="utf-8")
        primitive_shape_guardrail = audit_e8p_expert_kblock_primitive_shape_source(
            primitive_source_text,
            benchmark_context=context,
        )
        dispatch_contract_guardrail = audit_e8p_expert_kblock_dispatch_contract_source(
            primitive_source_text
        )
        partial_reduction_guardrail = audit_e8p_expert_kblock_partial_reduction_source(
            metal_source=source,
            primitive_source=primitive_source_text,
        )
        component_stream_source_guardrail = audit_e8p_component_stream_source(
            metal_source=source,
            primitive_source=primitive_source_text,
        )
        component_stream_speed_path_guardrail = audit_e8p_component_stream_speed_path_source(
            metal_source=source,
            primitive_source=primitive_source_text,
            benchmark_context=context,
        )
        component_stream_shared_decode_guardrail = (
            audit_e8p_component_stream_shared_decode_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        component_stream_partial_reduction_guardrail = (
            audit_e8p_component_stream_partial_reduction_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        route_slot_codeword_stream_guardrail = (
            audit_e8p_route_slot_codeword_stream_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        route_slot_mma_codeword_tile_guardrail = (
            audit_e8p_route_slot_mma_codeword_tile_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        active_route_tile_codeword_outer_product_guardrail = (
            audit_e8p_active_route_tile_codeword_outer_product_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        expert_cohort_codeword_broadcast_guardrail = (
            audit_e8p_expert_cohort_codeword_broadcast_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        route_batch_segmented_codeword_reduce_guardrail = (
            audit_e8p_route_batch_segmented_codeword_reduce_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        token_cohort_codeword_stream_guardrail = (
            audit_e8p_token_cohort_codeword_stream_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        token_cohort_mma_codeword_tile_guardrail = (
            audit_e8p_token_cohort_mma_codeword_tile_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        output_stationary_codeword_tile_guardrail = (
            audit_e8p_output_stationary_codeword_tile_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        input_stationary_codeword_tile_guardrail = (
            audit_e8p_input_stationary_codeword_tile_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        expert_kblock_codeword_factor_reuse_guardrail = (
            audit_e8p_expert_kblock_codeword_factor_reuse_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        route_codeword_lut_accumulate_guardrail = (
            audit_e8p_route_codeword_lut_accumulate_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        rowwise_codeword_tile_accumulate_guardrail = (
            audit_e8p_rowwise_codeword_tile_accumulate_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        output_tile_local_codeword_lut_guardrail = (
            audit_e8p_output_tile_local_codeword_lut_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        route_microtile_codeword_block_reduce_guardrail = (
            audit_e8p_route_microtile_codeword_block_reduce_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        kblock_wavefront_codeword_scan_guardrail = (
            audit_e8p_kblock_wavefront_codeword_scan_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        token_route_output_stripe_pipeline_guardrail = (
            audit_e8p_token_route_output_stripe_pipeline_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        expert_kblock_scale_slot_stream_guardrail = (
            audit_e8p_expert_kblock_scale_slot_stream_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        scale_group_route_block_reduce_guardrail = (
            audit_e8p_scale_group_route_block_reduce_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        route_block_output_group_stream_guardrail = (
            audit_e8p_route_block_output_group_stream_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        output_group_pretransposed_codeword_stream_guardrail = (
            audit_e8p_output_group_pretransposed_codeword_stream_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        kblock_output_group_route_fused_stream_guardrail = (
            audit_e8p_kblock_output_group_route_fused_stream_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        route_tile_output_swizzle_stream_guardrail = (
            audit_e8p_route_tile_output_swizzle_stream_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        token_topk_output_tile_stream_guardrail = (
            audit_e8p_token_topk_output_tile_stream_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        token_block_output_group_stream_guardrail = (
            audit_e8p_token_block_output_group_stream_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        token_output_stripe_group_stream_guardrail = (
            audit_e8p_token_output_stripe_group_stream_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        token_expert_output_block_stream_guardrail = (
            audit_e8p_token_expert_output_block_stream_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        token_pair_kblock_accumulator_stream_guardrail = (
            audit_e8p_token_pair_kblock_accumulator_stream_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        token_pair_output_group_stream_guardrail = (
            audit_e8p_token_pair_output_group_stream_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        token_pair_slot_topk_output_group_stream_guardrail = (
            audit_e8p_token_pair_slot_topk_output_group_stream_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        token_pair_slot_topk_codeword_group_pipeline_guardrail = (
            audit_e8p_token_pair_slot_topk_codeword_group_pipeline_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        token_pair_slot_topk_scale_slot_broadcast_stream_guardrail = (
            audit_e8p_token_pair_slot_topk_scale_slot_broadcast_stream_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        token_pair_slot_topk_route_bucket_codeword_reduce_guardrail = (
            audit_e8p_token_pair_slot_topk_route_bucket_codeword_reduce_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        token_pair_slot_topk_kblock_microtile_stream_guardrail = (
            audit_e8p_token_pair_slot_topk_kblock_microtile_stream_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
        token_pair_slot_topk_output_tile_fused_stream_guardrail = (
            audit_e8p_token_pair_slot_topk_output_tile_fused_stream_source(
                metal_source=source,
                primitive_source=primitive_source_text,
            )
        )
    else:
        primitive_shape_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        dispatch_contract_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        partial_reduction_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        component_stream_source_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        component_stream_speed_path_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        component_stream_shared_decode_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        component_stream_partial_reduction_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        route_slot_codeword_stream_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        route_slot_mma_codeword_tile_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        active_route_tile_codeword_outer_product_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        expert_cohort_codeword_broadcast_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        route_batch_segmented_codeword_reduce_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        token_cohort_codeword_stream_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        token_cohort_mma_codeword_tile_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        output_stationary_codeword_tile_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        input_stationary_codeword_tile_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        expert_kblock_codeword_factor_reuse_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        route_codeword_lut_accumulate_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        rowwise_codeword_tile_accumulate_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        output_tile_local_codeword_lut_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        route_microtile_codeword_block_reduce_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        kblock_wavefront_codeword_scan_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        token_route_output_stripe_pipeline_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        expert_kblock_scale_slot_stream_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        scale_group_route_block_reduce_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        route_block_output_group_stream_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        output_group_pretransposed_codeword_stream_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        kblock_output_group_route_fused_stream_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        route_tile_output_swizzle_stream_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        token_topk_output_tile_stream_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        token_block_output_group_stream_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        token_output_stripe_group_stream_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        token_expert_output_block_stream_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        token_pair_kblock_accumulator_stream_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        token_pair_output_group_stream_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        token_pair_slot_topk_output_group_stream_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        token_pair_slot_topk_codeword_group_pipeline_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        token_pair_slot_topk_scale_slot_broadcast_stream_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        token_pair_slot_topk_route_bucket_codeword_reduce_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        token_pair_slot_topk_kblock_microtile_stream_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
        token_pair_slot_topk_output_tile_fused_stream_guardrail = {
            "decision": "primitive_source_not_supplied",
            "required_next_features": [],
            "rejected_next_steps": [],
        }
    v2_kernel_report = next(
        (
            report
            for report in expert_kblock_schedule_guardrail.get("kernel_reports", [])
            if report.get("kernel_name") == _EXPERT_KBLOCK_TENSOROPS_V2_KERNEL
        ),
        {},
    )
    v2_partial_body_tensorops_present = bool(v2_kernel_report.get("has_tensorops_matmul"))
    v2_partial_body_rejected = str(v2_kernel_report.get("decision", "")).startswith(
        "reject_v2_"
    )
    if (
        partial_reduction_guardrail.get("decision")
        == "expert_kblock_v2_partial_reduction_scaffold_present"
        and not v2_partial_body_tensorops_present
    ):
        expert_kblock_schedule_guardrail = {
            **expert_kblock_schedule_guardrail,
            "decision": "expert_kblock_tensorops_v2_partial_reduction_scaffold_scalar_body",
            "v2_partial_body_tensorops_present": False,
            "required_next_features": [
                "replace_scalar_partial_body_with_valid_tensorops_schedule",
                "preserve_kblock_partial_reduction_scaffold",
                "preserve_compressed_rhs_storage",
                "preserve_codeword_scale_slots",
            ],
            "rejected_next_steps": [
                "do_not_benchmark_scalar_partial_reduction_scaffold",
                "do_not_promote_without_tensorops_speed_packet",
            ],
        }
    elif (
        partial_reduction_guardrail.get("decision")
        == "expert_kblock_v2_partial_reduction_scaffold_present"
        and v2_partial_body_rejected
    ):
        expert_kblock_schedule_guardrail = {
            **expert_kblock_schedule_guardrail,
            "v2_partial_body_tensorops_present": v2_partial_body_tensorops_present,
            "v2_partial_reduction_scaffold_present": True,
        }
    elif (
        partial_reduction_guardrail.get("decision")
        == "expert_kblock_v2_partial_reduction_scaffold_present"
    ):
        expert_kblock_schedule_guardrail = {
            **expert_kblock_schedule_guardrail,
            "decision": "expert_kblock_tensorops_v2_partial_reduction_scaffold_present",
            "v2_partial_body_tensorops_present": True,
            "required_next_features": partial_reduction_guardrail.get(
                "required_next_features", []
            ),
            "rejected_next_steps": partial_reduction_guardrail.get(
                "rejected_next_steps", []
            ),
        }
    track_b_current_family_verdict = _track_b_current_family_verdict(
        e8p_kernel=e8p_kernel,
        packed_rhs_tiled_family=packed_rhs_tiled_family,
        shared_decode_kernel=shared_decode_kernel,
        shared_n_decode_kernel=shared_n_decode_kernel,
        sign_nibble_schedule_guardrail=sign_nibble_schedule_guardrail,
        sign_plane_schedule_guardrail=sign_plane_schedule_guardrail,
        expert_kblock_schedule_guardrail=expert_kblock_schedule_guardrail,
        component_stream_speed_path_guardrail=component_stream_speed_path_guardrail,
        component_stream_partial_reduction_guardrail=component_stream_partial_reduction_guardrail,
        component_stream_shared_decode_guardrail=component_stream_shared_decode_guardrail,
        benchmark_context=context,
    )
    structural_gap = (
        "Current E8P Steel decodes into threadgroup Ws and reloads B tiles from that staging, "
        "while the audited q2 path is a sorted gather-qmm RHS NAX kernel with bk64 geometry."
    )
    expert_kblock_v2_rejection_next: str | None = None
    expert_kblock_v2_contract_ready = (
        dispatch_contract_guardrail.get("passes_contract") is True
        and partial_reduction_guardrail.get("passes_contract") is True
    )
    if (
        expert_kblock_v2_contract_ready
        and expert_kblock_schedule_guardrail.get("decision")
        == "reject_expert_kblock_tensorops_v2_near_duplicate"
    ):
        expert_kblock_v2_rejection_next = (
            "redesign_expert_kblock_v2_decode_reuse_or_change_kernel_family"
        )

    next_track_b_hypothesis = "replace_decoded_b_staging_or_change_kernel_family"
    if (
        shared_decode_kernel.get("decision")
        == "reject_per_fragment_factor_reconstruction_shared_decode"
    ):
        next_track_b_hypothesis = "share_factor_decode_across_expert_output_or_change_kernel_family"
    if (
        packed_rhs_tiled_family.get("decision")
        == "reject_decoded_b_staged_packed_rhs_family"
    ):
        next_track_b_hypothesis = "replace_decoded_b_staged_packed_rhs_family"
    if shared_n_decode_kernel.get("decision") == "reject_lane_local_shared_n_decode":
        next_track_b_hypothesis = "avoid_lane_local_shared_n_decode_or_change_kernel_family"
    if sign_nibble_schedule_guardrail.get("decision") in {
        "missing_q2_shaped_sign_nibble_tensorops_schedule",
        "reject_decoded_b_staged_sign_nibble_schedule",
    }:
        next_track_b_hypothesis = "build_q2_shaped_non_staged_sign_nibble_schedule"
    if (
        sign_nibble_schedule_guardrail.get("decision")
        == "reject_per_fragment_nibble_tensorops_schedule"
    ):
        next_track_b_hypothesis = "broaden_sign_nibble_decode_reuse_or_change_kernel_family"
    if (
        sign_nibble_schedule_guardrail.get("decision")
        == "reject_per_fragment_micro_lut_tensorops_schedule"
    ):
        next_track_b_hypothesis = "replace_micro_lut_tensorops_layout_or_kernel_family"
    if (
        sign_plane_schedule_guardrail.get("decision")
        == "reject_per_fragment_sign_plane_tensorops_schedule"
    ):
        next_track_b_hypothesis = "replace_sign_plane_tensorops_layout_or_kernel_family"
    if (
        expert_kblock_schedule_guardrail.get("decision")
        == "missing_expert_kblock_factor_decode_kernel"
    ):
        next_track_b_hypothesis = "build_expert_kblock_factor_decode_reuse_kernel"
    if (
        expert_kblock_schedule_guardrail.get("decision")
        == "scalar_expert_kblock_consumer_present_tensorops_missing"
    ):
        next_track_b_hypothesis = (
            "build_expert_kblock_tensorops_factor_decode_reuse_kernel"
        )
    if (
        expert_kblock_schedule_guardrail.get("decision")
        == "expert_kblock_tensorops_candidate_present_needs_benchmark"
    ):
        next_track_b_hypothesis = (
            "benchmark_expert_kblock_tensorops_factor_decode_reuse_kernel"
        )
    if (
        expert_kblock_schedule_guardrail.get("decision")
        == "current_expert_kblock_tensorops_speed_rejected_v2_missing"
    ):
        next_track_b_hypothesis = "design_expert_kblock_v2_or_change_kernel_family"
    if (
        expert_kblock_schedule_guardrail.get("decision")
        == "reject_expert_kblock_tensorops_v2_near_duplicate"
    ):
        next_track_b_hypothesis = (
            "redesign_expert_kblock_v2_decode_reuse_or_change_kernel_family"
        )
    if (
        expert_kblock_schedule_guardrail.get("decision")
        == "reject_expert_kblock_tensorops_v2_invalid_source"
    ):
        next_track_b_hypothesis = (
            "build_valid_expert_kblock_tensorops_v2_or_change_kernel_family"
        )
    if (
        expert_kblock_schedule_guardrail.get("decision")
        == "reject_expert_kblock_tensorops_v2_kblock_reduction_missing"
    ):
        next_track_b_hypothesis = (
            "design_expert_kblock_v2_partial_accumulation_or_change_kernel_family"
        )
    if (
        primitive_shape_guardrail.get("decision")
        == "current_primitive_shape_blocks_v2_reuse_scope"
    ):
        next_track_b_hypothesis = (
            "redesign_expert_kblock_primitive_dispatch_or_change_kernel_family"
        )
    if (
        dispatch_contract_guardrail.get("passes_contract") is True
        and expert_kblock_schedule_guardrail.get("decision")
        == "reject_expert_kblock_tensorops_v2_kblock_reduction_missing"
    ):
        next_track_b_hypothesis = (
            "design_expert_kblock_v2_partial_accumulation_or_change_kernel_family"
        )
    if (
        partial_reduction_guardrail.get("decision")
        == "expert_kblock_v2_partial_reduction_scaffold_present"
        and expert_kblock_schedule_guardrail.get("decision")
        == "expert_kblock_tensorops_v2_partial_reduction_scaffold_present"
    ):
        next_track_b_hypothesis = (
            "implement_native_v2_partial_reduction_parity_or_tensorops_schedule"
        )
    if (
        expert_kblock_schedule_guardrail.get("decision")
        == "expert_kblock_tensorops_v2_partial_reduction_scaffold_scalar_body"
    ):
        next_track_b_hypothesis = (
            "replace_scalar_v2_partial_body_with_valid_tensorops_schedule"
        )
    if (
        track_b_current_family_verdict.get("decision")
        == "current_track_b_family_not_benchmarkable"
    ):
        next_track_b_hypothesis = "change_rhs_layout_or_kernel_family"
    if (
        track_b_current_family_verdict.get("decision")
        == "current_track_b_family_not_benchmarkable"
        and route_slot_codeword_stream_guardrail.get("decision")
        == "missing_route_slot_codeword_stream_source"
    ):
        next_track_b_hypothesis = "implement_route_slot_codeword_stream_source_guardrail"
    if (
        route_slot_codeword_stream_guardrail.get("decision")
        == "route_slot_codeword_stream_source_guardrail_present"
    ):
        next_track_b_hypothesis = "prove_route_slot_codeword_stream_native_parity"
    route_slot_mma_probe_ready = (
        next_track_b_hypothesis == "implement_route_slot_codeword_stream_source_guardrail"
    )
    if (
        route_slot_mma_probe_ready
        and route_slot_mma_codeword_tile_guardrail.get("decision")
        == "missing_route_slot_mma_codeword_tile_source"
    ):
        next_track_b_hypothesis = "implement_route_slot_mma_codeword_tile_source_guardrail"
    if (
        route_slot_mma_probe_ready
        and route_slot_mma_codeword_tile_guardrail.get("decision")
        == "route_slot_mma_codeword_tile_source_guardrail_present"
    ):
        next_track_b_hypothesis = "prove_route_slot_mma_codeword_tile_native_parity"
    if (
        component_stream_speed_path_guardrail.get("decision")
        == "component_stream_scalar_oracle_speed_path_missing"
    ):
        next_track_b_hypothesis = (
            "design_component_stream_speed_path_or_change_kernel_family"
        )
    if (
        component_stream_source_guardrail.get("passes_contract") is True
        and component_stream_partial_reduction_guardrail.get("decision")
        == "missing_component_stream_partial_reduction_scaffold"
    ):
        next_track_b_hypothesis = (
            "implement_component_stream_partial_reduction_scaffold_or_change_kernel_family"
        )
    if (
        component_stream_partial_reduction_guardrail.get("decision")
        == "component_stream_partial_reduction_scaffold_present"
    ):
        next_track_b_hypothesis = (
            "prove_component_stream_partial_reduction_parity_or_parallel_schedule"
        )
    if (
        component_stream_partial_reduction_guardrail.get("decision")
        == "component_stream_partial_reduction_tensorops_body_present"
    ):
        next_track_b_hypothesis = (
            "prove_component_stream_tensorops_partial_parity_before_speed"
        )
    if (
        component_stream_partial_reduction_guardrail.get("decision")
        == "component_stream_partial_reduction_parallel_body_present"
    ):
        next_track_b_hypothesis = (
            "prove_component_stream_partial_reduction_native_parity"
        )
    if (
        component_stream_partial_reduction_guardrail.get("decision")
        == "component_stream_partial_reduction_scaffold_scalar_body"
    ):
        next_track_b_hypothesis = (
            "replace_scalar_component_stream_partial_body_with_valid_parallel_schedule"
        )
    if (
        component_stream_speed_path_guardrail.get("decision")
        == "component_stream_tensorops_speed_path_candidate_present_needs_parity"
    ):
        next_track_b_hypothesis = "prove_component_stream_tensorops_artifact_parity"
    if (
        component_stream_speed_path_guardrail.get("decision")
        == "reject_component_stream_tensorops_speed_path"
    ):
        next_track_b_hypothesis = "change_component_stream_tensorops_layout_or_kernel_family"
    if (
        component_stream_speed_path_guardrail.get("decision")
        == "reject_component_stream_tensorops_speed_path"
        and component_stream_shared_decode_guardrail.get("decision")
        == "missing_component_stream_shared_decode_source"
    ):
        next_track_b_hypothesis = (
            "implement_component_stream_shared_decode_source_guardrail_then_parity"
        )
    if (
        component_stream_shared_decode_guardrail.get("decision")
        == "reject_component_stream_shared_decode_per_fragment_reconstruction"
    ):
        next_track_b_hypothesis = (
            "redesign_component_stream_shared_decode_cache_or_change_kernel_family"
        )
    if (
        component_stream_speed_path_guardrail.get("decision")
        == "reject_component_stream_tensorops_speed_path"
        and component_stream_shared_decode_guardrail.get("decision")
        == "component_stream_shared_decode_scalar_cache_scaffold_present"
    ):
        next_track_b_hypothesis = (
            "replace_scalar_component_stream_shared_decode_body_with_valid_parallel_schedule"
        )
    if (
        component_stream_shared_decode_guardrail.get("decision")
        == "component_stream_shared_decode_candidate_present_needs_parity"
    ):
        next_track_b_hypothesis = "prove_component_stream_shared_decode_parity"
    if expert_kblock_v2_rejection_next is not None:
        next_track_b_hypothesis = expert_kblock_v2_rejection_next
    if (
        active_route_tile_design_ready
        and active_route_tile_codeword_outer_product_guardrail.get("decision")
        == "missing_active_route_tile_codeword_outer_product_source"
    ):
        next_track_b_hypothesis = (
            "implement_active_route_tile_codeword_outer_product_source_guardrail"
        )
    if (
        active_route_tile_design_ready
        and active_route_tile_codeword_outer_product_guardrail.get("decision")
        == "active_route_tile_codeword_outer_product_source_guardrail_present"
    ):
        next_track_b_hypothesis = (
            "prove_active_route_tile_codeword_outer_product_native_parity"
        )
    if (
        expert_cohort_design_ready
        and expert_cohort_codeword_broadcast_guardrail.get("decision")
        == "missing_expert_cohort_codeword_broadcast_source"
    ):
        next_track_b_hypothesis = (
            "implement_expert_cohort_codeword_broadcast_source_guardrail"
        )
    if (
        expert_cohort_design_ready
        and expert_cohort_codeword_broadcast_guardrail.get("decision")
        == "expert_cohort_codeword_broadcast_source_guardrail_present"
    ):
        next_track_b_hypothesis = (
            "prove_expert_cohort_codeword_broadcast_native_parity"
        )
    if (
        route_batch_segmented_design_ready
        and route_batch_segmented_codeword_reduce_guardrail.get("decision")
        == "missing_route_batch_segmented_codeword_reduce_source"
    ):
        next_track_b_hypothesis = (
            "implement_route_batch_segmented_codeword_reduce_source_guardrail"
        )
    if (
        route_batch_segmented_design_ready
        and route_batch_segmented_codeword_reduce_guardrail.get("decision")
        == "route_batch_segmented_codeword_reduce_source_guardrail_present"
    ):
        next_track_b_hypothesis = (
            "prove_route_batch_segmented_codeword_reduce_native_parity"
        )
    if (
        component_stream_partial_reduction_design_ready
        and component_stream_partial_reduction_guardrail.get("decision")
        == "missing_component_stream_partial_reduction_scaffold"
    ):
        next_track_b_hypothesis = (
            "implement_component_stream_partial_reduction_source_guardrail"
        )
    if (
        component_stream_partial_reduction_design_ready
        and component_stream_partial_reduction_guardrail.get("decision")
        == "component_stream_partial_reduction_scaffold_scalar_body"
    ):
        next_track_b_hypothesis = (
            "replace_scalar_component_stream_partial_body_with_valid_parallel_schedule"
        )
    if (
        component_stream_partial_reduction_design_ready
        and component_stream_partial_reduction_guardrail.get("decision")
        == "component_stream_partial_reduction_tensorops_body_present"
    ):
        next_track_b_hypothesis = (
            "prove_component_stream_tensorops_partial_parity_before_speed"
        )
    if (
        component_stream_partial_reduction_design_ready
        and component_stream_partial_reduction_guardrail.get("decision")
        == "component_stream_partial_reduction_parallel_body_present"
    ):
        next_track_b_hypothesis = (
            "prove_component_stream_partial_reduction_native_parity"
        )
    if (
        token_cohort_design_ready
        and token_cohort_codeword_stream_guardrail.get("decision")
        == "missing_token_cohort_codeword_stream_source"
    ):
        next_track_b_hypothesis = (
            "implement_token_cohort_codeword_stream_source_guardrail"
        )
    if (
        token_cohort_design_ready
        and token_cohort_codeword_stream_guardrail.get("decision")
        == "token_cohort_codeword_stream_source_guardrail_present"
    ):
        next_track_b_hypothesis = "prove_token_cohort_codeword_stream_native_parity"
    if (
        token_cohort_mma_codeword_tile_design_ready
        and token_cohort_mma_codeword_tile_guardrail.get("decision")
        == "missing_token_cohort_mma_codeword_tile_source"
    ):
        next_track_b_hypothesis = (
            "implement_token_cohort_mma_codeword_tile_source_guardrail"
        )
    if (
        token_cohort_mma_codeword_tile_design_ready
        and token_cohort_mma_codeword_tile_guardrail.get("decision")
        == "token_cohort_mma_codeword_tile_source_guardrail_present"
    ):
        next_track_b_hypothesis = "prove_token_cohort_mma_codeword_tile_native_parity"
    if (
        output_stationary_codeword_tile_design_ready
        and output_stationary_codeword_tile_guardrail.get("decision")
        == "missing_output_stationary_codeword_tile_source"
    ):
        next_track_b_hypothesis = (
            "implement_output_stationary_codeword_tile_source_guardrail"
        )
    if (
        output_stationary_codeword_tile_design_ready
        and output_stationary_codeword_tile_guardrail.get("decision")
        == "output_stationary_codeword_tile_source_guardrail_present"
    ):
        next_track_b_hypothesis = (
            "prove_output_stationary_codeword_tile_native_parity"
        )
    if (
        input_stationary_codeword_tile_design_ready
        and input_stationary_codeword_tile_guardrail.get("decision")
        == "missing_input_stationary_codeword_tile_source"
    ):
        next_track_b_hypothesis = (
            "implement_input_stationary_codeword_tile_source_guardrail"
        )
    if (
        input_stationary_codeword_tile_design_ready
        and input_stationary_codeword_tile_guardrail.get("decision")
        == "input_stationary_codeword_tile_source_guardrail_present"
    ):
        next_track_b_hypothesis = (
            "prove_input_stationary_codeword_tile_native_parity"
        )
    if (
        expert_kblock_codeword_factor_reuse_design_ready
        and expert_kblock_codeword_factor_reuse_guardrail.get("decision")
        == "missing_expert_kblock_codeword_factor_reuse_source"
    ):
        next_track_b_hypothesis = (
            "implement_expert_kblock_codeword_factor_reuse_source_guardrail"
        )
    if (
        expert_kblock_codeword_factor_reuse_design_ready
        and expert_kblock_codeword_factor_reuse_guardrail.get("decision")
        == "expert_kblock_codeword_factor_reuse_source_guardrail_present"
    ):
        next_track_b_hypothesis = (
            "prove_expert_kblock_codeword_factor_reuse_native_parity"
        )
    if (
        route_codeword_lut_accumulate_design_ready
        and route_codeword_lut_accumulate_guardrail.get("decision")
        == "missing_route_codeword_lut_accumulate_source"
    ):
        next_track_b_hypothesis = (
            "implement_route_codeword_lut_accumulate_source_guardrail"
        )
    if (
        route_codeword_lut_accumulate_design_ready
        and route_codeword_lut_accumulate_guardrail.get("decision")
        == "route_codeword_lut_accumulate_source_guardrail_present"
    ):
        next_track_b_hypothesis = (
            "prove_route_codeword_lut_accumulate_native_parity"
        )
    if (
        rowwise_codeword_tile_accumulate_design_ready
        and rowwise_codeword_tile_accumulate_guardrail.get("decision")
        == "missing_rowwise_codeword_tile_accumulate_source"
    ):
        next_track_b_hypothesis = (
            "implement_rowwise_codeword_tile_accumulate_source_guardrail"
        )
    if (
        rowwise_codeword_tile_accumulate_design_ready
        and rowwise_codeword_tile_accumulate_guardrail.get("decision")
        == "rowwise_codeword_tile_accumulate_source_guardrail_present"
    ):
        next_track_b_hypothesis = (
            "prove_rowwise_codeword_tile_accumulate_native_parity"
        )
    if (
        output_tile_local_codeword_lut_design_ready
        and output_tile_local_codeword_lut_guardrail.get("decision")
        == "missing_output_tile_local_codeword_lut_source"
    ):
        next_track_b_hypothesis = (
            "implement_output_tile_local_codeword_lut_source_guardrail"
        )
    if (
        output_tile_local_codeword_lut_design_ready
        and output_tile_local_codeword_lut_guardrail.get("decision")
        == "output_tile_local_codeword_lut_source_guardrail_present"
    ):
        next_track_b_hypothesis = (
            "prove_output_tile_local_codeword_lut_native_parity"
        )
    if (
        route_microtile_codeword_block_reduce_design_ready
        and route_microtile_codeword_block_reduce_guardrail.get("decision")
        == "missing_route_microtile_codeword_block_reduce_source"
    ):
        next_track_b_hypothesis = (
            "implement_route_microtile_codeword_block_reduce_source_guardrail"
        )
    if (
        route_microtile_codeword_block_reduce_design_ready
        and route_microtile_codeword_block_reduce_guardrail.get("decision")
        == "route_microtile_codeword_block_reduce_source_guardrail_present"
    ):
        next_track_b_hypothesis = (
            "prove_route_microtile_codeword_block_reduce_native_parity"
        )
    if (
        kblock_wavefront_codeword_scan_design_ready
        and kblock_wavefront_codeword_scan_guardrail.get("decision")
        == "missing_kblock_wavefront_codeword_scan_source"
    ):
        next_track_b_hypothesis = (
            "implement_kblock_wavefront_codeword_scan_source_guardrail"
        )
    if (
        kblock_wavefront_codeword_scan_design_ready
        and kblock_wavefront_codeword_scan_guardrail.get("decision")
        == "kblock_wavefront_codeword_scan_source_guardrail_present"
    ):
        next_track_b_hypothesis = "prove_kblock_wavefront_codeword_scan_native_parity"
    if (
        token_route_output_stripe_pipeline_design_ready
        and token_route_output_stripe_pipeline_guardrail.get("decision")
        == "missing_token_route_output_stripe_pipeline_source"
    ):
        next_track_b_hypothesis = (
            "implement_token_route_output_stripe_pipeline_source_guardrail"
        )
    if (
        token_route_output_stripe_pipeline_design_ready
        and token_route_output_stripe_pipeline_guardrail.get("decision")
        == "token_route_output_stripe_pipeline_source_guardrail_present"
    ):
        next_track_b_hypothesis = (
            "prove_token_route_output_stripe_pipeline_native_parity"
        )
    if (
        expert_kblock_scale_slot_stream_design_ready
        and expert_kblock_scale_slot_stream_guardrail.get("decision")
        == "missing_expert_kblock_scale_slot_stream_source"
    ):
        next_track_b_hypothesis = (
            "implement_expert_kblock_scale_slot_stream_source_guardrail"
        )
    if (
        expert_kblock_scale_slot_stream_design_ready
        and expert_kblock_scale_slot_stream_guardrail.get("decision")
        == "expert_kblock_scale_slot_stream_source_guardrail_present"
    ):
        next_track_b_hypothesis = (
            "prove_expert_kblock_scale_slot_stream_native_parity"
        )
    if (
        scale_group_route_block_reduce_design_ready
        and scale_group_route_block_reduce_guardrail.get("decision")
        == "missing_scale_group_route_block_reduce_source"
    ):
        next_track_b_hypothesis = (
            "implement_scale_group_route_block_reduce_source_guardrail"
        )
    if (
        scale_group_route_block_reduce_design_ready
        and scale_group_route_block_reduce_guardrail.get("decision")
        == "scale_group_route_block_reduce_source_guardrail_present"
    ):
        next_track_b_hypothesis = (
            "prove_scale_group_route_block_reduce_native_parity"
        )
    if (
        route_block_output_group_stream_design_ready
        and route_block_output_group_stream_guardrail.get("decision")
        == "missing_route_block_output_group_stream_source"
    ):
        next_track_b_hypothesis = (
            "implement_route_block_output_group_stream_source_guardrail"
        )
    if (
        route_block_output_group_stream_design_ready
        and route_block_output_group_stream_guardrail.get("decision")
        == "route_block_output_group_stream_source_guardrail_present"
    ):
        next_track_b_hypothesis = (
            "prove_route_block_output_group_stream_native_parity"
        )
    if (
        output_group_pretransposed_codeword_stream_design_ready
        and output_group_pretransposed_codeword_stream_guardrail.get("decision")
        == "missing_output_group_pretransposed_codeword_stream_source"
    ):
        next_track_b_hypothesis = (
            "implement_output_group_pretransposed_codeword_stream_source_guardrail"
        )
    if (
        output_group_pretransposed_codeword_stream_design_ready
        and output_group_pretransposed_codeword_stream_guardrail.get("decision")
        == "output_group_pretransposed_codeword_stream_source_guardrail_present"
    ):
        next_track_b_hypothesis = (
            "prove_output_group_pretransposed_codeword_stream_native_parity"
        )
    if (
        kblock_output_group_route_fused_stream_design_ready
        and kblock_output_group_route_fused_stream_guardrail.get("decision")
        == "missing_kblock_output_group_route_fused_stream_source"
    ):
        next_track_b_hypothesis = (
            "implement_kblock_output_group_route_fused_stream_source_guardrail"
        )
    if (
        kblock_output_group_route_fused_stream_design_ready
        and kblock_output_group_route_fused_stream_guardrail.get("decision")
        == "kblock_output_group_route_fused_stream_source_guardrail_present"
    ):
        next_track_b_hypothesis = (
            "prove_kblock_output_group_route_fused_stream_native_parity"
        )
    if (
        route_tile_output_swizzle_stream_design_ready
        and route_tile_output_swizzle_stream_guardrail.get("decision")
        == "missing_route_tile_output_swizzle_stream_source"
    ):
        next_track_b_hypothesis = (
            "implement_route_tile_output_swizzle_stream_source_guardrail"
        )
    if (
        route_tile_output_swizzle_stream_design_ready
        and route_tile_output_swizzle_stream_guardrail.get("decision")
        == "route_tile_output_swizzle_stream_source_guardrail_present"
    ):
        next_track_b_hypothesis = (
            "prove_route_tile_output_swizzle_stream_native_parity"
        )
    if (
        token_topk_output_tile_stream_design_ready
        and token_topk_output_tile_stream_guardrail.get("decision")
        == "missing_token_topk_output_tile_stream_source"
    ):
        next_track_b_hypothesis = (
            "implement_token_topk_output_tile_stream_source_guardrail"
        )
    if (
        token_topk_output_tile_stream_design_ready
        and token_topk_output_tile_stream_guardrail.get("decision")
        == "token_topk_output_tile_stream_source_guardrail_present"
    ):
        next_track_b_hypothesis = (
            "prove_token_topk_output_tile_stream_native_parity"
        )
    if (
        token_block_output_group_stream_design_ready
        and token_block_output_group_stream_guardrail.get("decision")
        == "missing_token_block_output_group_stream_source"
    ):
        next_track_b_hypothesis = (
            "implement_token_block_output_group_stream_source_guardrail"
        )
    if (
        token_block_output_group_stream_design_ready
        and token_block_output_group_stream_guardrail.get("decision")
        == "token_block_output_group_stream_source_guardrail_present"
    ):
        next_track_b_hypothesis = (
            "prove_token_block_output_group_stream_native_parity"
        )
    if (
        token_output_stripe_group_stream_design_ready
        and token_output_stripe_group_stream_guardrail.get("decision")
        == "missing_token_output_stripe_group_stream_source"
    ):
        next_track_b_hypothesis = (
            "implement_token_output_stripe_group_stream_source_guardrail"
        )
    if (
        token_output_stripe_group_stream_design_ready
        and token_output_stripe_group_stream_guardrail.get("decision")
        == "token_output_stripe_group_stream_source_guardrail_present"
    ):
        next_track_b_hypothesis = (
            "prove_token_output_stripe_group_stream_native_parity"
        )
    if (
        token_expert_output_block_stream_design_ready
        and token_expert_output_block_stream_guardrail.get("decision")
        == "missing_token_expert_output_block_stream_source"
    ):
        next_track_b_hypothesis = (
            "implement_token_expert_output_block_stream_source_guardrail"
        )
    if (
        token_expert_output_block_stream_design_ready
        and token_expert_output_block_stream_guardrail.get("decision")
        == "token_expert_output_block_stream_source_guardrail_present"
    ):
        next_track_b_hypothesis = (
            "prove_token_expert_output_block_stream_native_parity"
        )
    if (
        token_pair_kblock_accumulator_stream_design_ready
        and token_pair_kblock_accumulator_stream_guardrail.get("decision")
        == "missing_token_pair_kblock_accumulator_stream_source"
    ):
        next_track_b_hypothesis = (
            "implement_token_pair_kblock_accumulator_stream_source_guardrail"
        )
    if (
        token_pair_kblock_accumulator_stream_design_ready
        and token_pair_kblock_accumulator_stream_guardrail.get("decision")
        == "token_pair_kblock_accumulator_stream_source_guardrail_present"
    ):
        next_track_b_hypothesis = (
            "prove_token_pair_kblock_accumulator_stream_native_parity"
        )
    if (
        token_pair_output_group_stream_design_ready
        and token_pair_output_group_stream_guardrail.get("decision")
        == "missing_token_pair_output_group_stream_source"
    ):
        next_track_b_hypothesis = (
            "implement_token_pair_output_group_stream_source_guardrail"
        )
    if (
        token_pair_output_group_stream_design_ready
        and token_pair_output_group_stream_guardrail.get("decision")
        == "token_pair_output_group_stream_source_guardrail_present"
    ):
        next_track_b_hypothesis = (
            "prove_token_pair_output_group_stream_native_parity"
        )
    if (
        token_pair_slot_topk_output_group_stream_design_ready
        and token_pair_slot_topk_output_group_stream_guardrail.get("decision")
        == "missing_token_pair_slot_topk_output_group_stream_source"
    ):
        next_track_b_hypothesis = (
            "implement_token_pair_slot_topk_output_group_stream_source_guardrail"
        )
    if (
        token_pair_slot_topk_output_group_stream_design_ready
        and token_pair_slot_topk_output_group_stream_guardrail.get("decision")
        == "token_pair_slot_topk_output_group_stream_source_guardrail_present"
    ):
        next_track_b_hypothesis = (
            "prove_token_pair_slot_topk_output_group_stream_native_parity"
        )
    if (
        token_pair_slot_topk_codeword_group_pipeline_design_ready
        and token_pair_slot_topk_codeword_group_pipeline_guardrail.get("decision")
        == "missing_token_pair_slot_topk_codeword_group_pipeline_source"
    ):
        next_track_b_hypothesis = (
            "implement_token_pair_slot_topk_codeword_group_pipeline_source_guardrail"
        )
    if (
        token_pair_slot_topk_codeword_group_pipeline_design_ready
        and token_pair_slot_topk_codeword_group_pipeline_guardrail.get("decision")
        == "token_pair_slot_topk_codeword_group_pipeline_source_guardrail_present"
    ):
        next_track_b_hypothesis = (
            "prove_token_pair_slot_topk_codeword_group_pipeline_native_parity"
        )
    if (
        token_pair_slot_topk_scale_slot_broadcast_stream_design_ready
        and token_pair_slot_topk_scale_slot_broadcast_stream_guardrail.get("decision")
        == "missing_token_pair_slot_topk_scale_slot_broadcast_stream_source"
    ):
        next_track_b_hypothesis = (
            "implement_token_pair_slot_topk_scale_slot_broadcast_stream_source_guardrail"
        )
    if (
        token_pair_slot_topk_scale_slot_broadcast_stream_design_ready
        and token_pair_slot_topk_scale_slot_broadcast_stream_guardrail.get("decision")
        == "token_pair_slot_topk_scale_slot_broadcast_stream_source_guardrail_present"
    ):
        next_track_b_hypothesis = (
            "prove_token_pair_slot_topk_scale_slot_broadcast_stream_native_parity"
        )
    if (
        token_pair_slot_topk_route_bucket_codeword_reduce_design_ready
        and token_pair_slot_topk_route_bucket_codeword_reduce_guardrail.get("decision")
        == "missing_token_pair_slot_topk_route_bucket_codeword_reduce_source"
    ):
        next_track_b_hypothesis = (
            "implement_token_pair_slot_topk_route_bucket_codeword_reduce_source_guardrail"
        )
    if (
        token_pair_slot_topk_route_bucket_codeword_reduce_design_ready
        and token_pair_slot_topk_route_bucket_codeword_reduce_guardrail.get("decision")
        == "token_pair_slot_topk_route_bucket_codeword_reduce_source_guardrail_present"
    ):
        next_track_b_hypothesis = (
            "prove_token_pair_slot_topk_route_bucket_codeword_reduce_native_parity"
        )
    if (
        token_pair_slot_topk_kblock_microtile_stream_design_ready
        and token_pair_slot_topk_kblock_microtile_stream_guardrail.get("decision")
        == "missing_token_pair_slot_topk_kblock_microtile_stream_source"
    ):
        next_track_b_hypothesis = (
            "implement_token_pair_slot_topk_kblock_microtile_stream_source_guardrail"
        )
    if (
        token_pair_slot_topk_kblock_microtile_stream_design_ready
        and token_pair_slot_topk_kblock_microtile_stream_guardrail.get("decision")
        == "token_pair_slot_topk_kblock_microtile_stream_source_guardrail_present"
    ):
        next_track_b_hypothesis = (
            "prove_token_pair_slot_topk_kblock_microtile_stream_native_parity"
        )
    if (
        token_pair_slot_topk_output_tile_fused_stream_design_ready
        and token_pair_slot_topk_output_tile_fused_stream_guardrail.get("decision")
        == "missing_token_pair_slot_topk_output_tile_fused_stream_source"
    ):
        next_track_b_hypothesis = (
            "implement_token_pair_slot_topk_output_tile_fused_stream_source_guardrail"
        )
    if (
        token_pair_slot_topk_output_tile_fused_stream_design_ready
        and token_pair_slot_topk_output_tile_fused_stream_guardrail.get("decision")
        == "token_pair_slot_topk_output_tile_fused_stream_source_guardrail_present"
    ):
        next_track_b_hypothesis = (
            "prove_token_pair_slot_topk_output_tile_fused_stream_native_parity"
        )
    return {
        "schema_version": 1,
        "record_type": "glm45_air_e8p_kernel_structure_analysis",
        "peer2_used": False,
        "rdma_jaccl_touched": False,
        "speed_claim": False,
        "native_parity_claim": False,
        "metal_source": str(metal_source),
        "primitive_source": None if primitive_source is None else str(primitive_source),
        "q2_jsonl": str(q2_jsonl),
        "q2_geometry_jsonl": str(q2_geometry_source),
        "active_route_tile_design_json": (
            None
            if active_route_tile_design_json is None
            else str(active_route_tile_design_json)
        ),
        "expert_cohort_design_json": (
            None if expert_cohort_design_json is None else str(expert_cohort_design_json)
        ),
        "route_batch_segmented_design_json": (
            None
            if route_batch_segmented_design_json is None
            else str(route_batch_segmented_design_json)
        ),
        "component_stream_partial_reduction_design_json": (
            None
            if component_stream_partial_reduction_design_json is None
            else str(component_stream_partial_reduction_design_json)
        ),
        "token_cohort_design_json": (
            None if token_cohort_design_json is None else str(token_cohort_design_json)
        ),
        "token_cohort_mma_codeword_tile_design_json": (
            None
            if token_cohort_mma_codeword_tile_design_json is None
            else str(token_cohort_mma_codeword_tile_design_json)
        ),
        "output_stationary_codeword_tile_design_json": (
            None
            if output_stationary_codeword_tile_design_json is None
            else str(output_stationary_codeword_tile_design_json)
        ),
        "input_stationary_codeword_tile_design_json": (
            None
            if input_stationary_codeword_tile_design_json is None
            else str(input_stationary_codeword_tile_design_json)
        ),
        "expert_kblock_codeword_factor_reuse_design_json": (
            None
            if expert_kblock_codeword_factor_reuse_design_json is None
            else str(expert_kblock_codeword_factor_reuse_design_json)
        ),
        "route_codeword_lut_accumulate_design_json": (
            None
            if route_codeword_lut_accumulate_design_json is None
            else str(route_codeword_lut_accumulate_design_json)
        ),
        "rowwise_codeword_tile_accumulate_design_json": (
            None
            if rowwise_codeword_tile_accumulate_design_json is None
            else str(rowwise_codeword_tile_accumulate_design_json)
        ),
        "output_tile_local_codeword_lut_design_json": (
            None
            if output_tile_local_codeword_lut_design_json is None
            else str(output_tile_local_codeword_lut_design_json)
        ),
        "route_microtile_codeword_block_reduce_design_json": (
            None
            if route_microtile_codeword_block_reduce_design_json is None
            else str(route_microtile_codeword_block_reduce_design_json)
        ),
        "kblock_wavefront_codeword_scan_design_json": (
            None
            if kblock_wavefront_codeword_scan_design_json is None
            else str(kblock_wavefront_codeword_scan_design_json)
        ),
        "token_route_output_stripe_pipeline_design_json": (
            None
            if token_route_output_stripe_pipeline_design_json is None
            else str(token_route_output_stripe_pipeline_design_json)
        ),
        "expert_kblock_scale_slot_stream_design_json": (
            None
            if expert_kblock_scale_slot_stream_design_json is None
            else str(expert_kblock_scale_slot_stream_design_json)
        ),
        "scale_group_route_block_reduce_design_json": (
            None
            if scale_group_route_block_reduce_design_json is None
            else str(scale_group_route_block_reduce_design_json)
        ),
        "route_block_output_group_stream_design_json": (
            None
            if route_block_output_group_stream_design_json is None
            else str(route_block_output_group_stream_design_json)
        ),
        "output_group_pretransposed_codeword_stream_design_json": (
            None
            if output_group_pretransposed_codeword_stream_design_json is None
            else str(output_group_pretransposed_codeword_stream_design_json)
        ),
        "kblock_output_group_route_fused_stream_design_json": (
            None
            if kblock_output_group_route_fused_stream_design_json is None
            else str(kblock_output_group_route_fused_stream_design_json)
        ),
        "route_tile_output_swizzle_stream_design_json": (
            None
            if route_tile_output_swizzle_stream_design_json is None
            else str(route_tile_output_swizzle_stream_design_json)
        ),
        "token_topk_output_tile_stream_design_json": (
            None
            if token_topk_output_tile_stream_design_json is None
            else str(token_topk_output_tile_stream_design_json)
        ),
        "token_block_output_group_stream_design_json": (
            None
            if token_block_output_group_stream_design_json is None
            else str(token_block_output_group_stream_design_json)
        ),
        "token_output_stripe_group_stream_design_json": (
            None
            if token_output_stripe_group_stream_design_json is None
            else str(token_output_stripe_group_stream_design_json)
        ),
        "token_expert_output_block_stream_design_json": (
            None
            if token_expert_output_block_stream_design_json is None
            else str(token_expert_output_block_stream_design_json)
        ),
        "token_pair_kblock_accumulator_stream_design_json": (
            None
            if token_pair_kblock_accumulator_stream_design_json is None
            else str(token_pair_kblock_accumulator_stream_design_json)
        ),
        "token_pair_output_group_stream_design_json": (
            None
            if token_pair_output_group_stream_design_json is None
            else str(token_pair_output_group_stream_design_json)
        ),
        "token_pair_slot_topk_output_group_stream_design_json": (
            None
            if token_pair_slot_topk_output_group_stream_design_json is None
            else str(token_pair_slot_topk_output_group_stream_design_json)
        ),
        "token_pair_slot_topk_codeword_group_pipeline_design_json": (
            None
            if token_pair_slot_topk_codeword_group_pipeline_design_json is None
            else str(token_pair_slot_topk_codeword_group_pipeline_design_json)
        ),
        "token_pair_slot_topk_scale_slot_broadcast_stream_design_json": (
            None
            if token_pair_slot_topk_scale_slot_broadcast_stream_design_json is None
            else str(token_pair_slot_topk_scale_slot_broadcast_stream_design_json)
        ),
        "token_pair_slot_topk_route_bucket_codeword_reduce_design_json": (
            None
            if token_pair_slot_topk_route_bucket_codeword_reduce_design_json is None
            else str(token_pair_slot_topk_route_bucket_codeword_reduce_design_json)
        ),
        "token_pair_slot_topk_kblock_microtile_stream_design_json": (
            None
            if token_pair_slot_topk_kblock_microtile_stream_design_json is None
            else str(token_pair_slot_topk_kblock_microtile_stream_design_json)
        ),
        "token_pair_slot_topk_output_tile_fused_stream_design_json": (
            None
            if token_pair_slot_topk_output_tile_fused_stream_design_json is None
            else str(token_pair_slot_topk_output_tile_fused_stream_design_json)
        ),
        "e8p_kernel": e8p_kernel,
        "shared_decode_kernel": shared_decode_kernel,
        "shared_n_decode_kernel": shared_n_decode_kernel,
        "packed_rhs_tiled_family": packed_rhs_tiled_family,
        "sign_nibble_schedule_guardrail": sign_nibble_schedule_guardrail,
        "sign_plane_schedule_guardrail": sign_plane_schedule_guardrail,
        "expert_kblock_schedule_guardrail": expert_kblock_schedule_guardrail,
        "expert_kblock_primitive_shape_guardrail": primitive_shape_guardrail,
        "expert_kblock_dispatch_contract_guardrail": dispatch_contract_guardrail,
        "expert_kblock_partial_reduction_guardrail": partial_reduction_guardrail,
        "component_stream_source_guardrail": component_stream_source_guardrail,
        "component_stream_speed_path_guardrail": component_stream_speed_path_guardrail,
        "component_stream_shared_decode_guardrail": (
            component_stream_shared_decode_guardrail
        ),
        "component_stream_partial_reduction_guardrail": (
            component_stream_partial_reduction_guardrail
        ),
        "route_slot_codeword_stream_guardrail": route_slot_codeword_stream_guardrail,
        "route_slot_mma_codeword_tile_guardrail": route_slot_mma_codeword_tile_guardrail,
        "active_route_tile_codeword_outer_product_guardrail": (
            active_route_tile_codeword_outer_product_guardrail
        ),
        "expert_cohort_codeword_broadcast_guardrail": (
            expert_cohort_codeword_broadcast_guardrail
        ),
        "route_batch_segmented_codeword_reduce_guardrail": (
            route_batch_segmented_codeword_reduce_guardrail
        ),
        "token_cohort_codeword_stream_guardrail": (
            token_cohort_codeword_stream_guardrail
        ),
        "token_cohort_mma_codeword_tile_guardrail": (
            token_cohort_mma_codeword_tile_guardrail
        ),
        "output_stationary_codeword_tile_guardrail": (
            output_stationary_codeword_tile_guardrail
        ),
        "input_stationary_codeword_tile_guardrail": (
            input_stationary_codeword_tile_guardrail
        ),
        "expert_kblock_codeword_factor_reuse_guardrail": (
            expert_kblock_codeword_factor_reuse_guardrail
        ),
        "route_codeword_lut_accumulate_guardrail": (
            route_codeword_lut_accumulate_guardrail
        ),
        "rowwise_codeword_tile_accumulate_guardrail": (
            rowwise_codeword_tile_accumulate_guardrail
        ),
        "output_tile_local_codeword_lut_guardrail": (
            output_tile_local_codeword_lut_guardrail
        ),
        "route_microtile_codeword_block_reduce_guardrail": (
            route_microtile_codeword_block_reduce_guardrail
        ),
        "kblock_wavefront_codeword_scan_guardrail": (
            kblock_wavefront_codeword_scan_guardrail
        ),
        "token_route_output_stripe_pipeline_guardrail": (
            token_route_output_stripe_pipeline_guardrail
        ),
        "expert_kblock_scale_slot_stream_guardrail": (
            expert_kblock_scale_slot_stream_guardrail
        ),
        "scale_group_route_block_reduce_guardrail": (
            scale_group_route_block_reduce_guardrail
        ),
        "route_block_output_group_stream_guardrail": (
            route_block_output_group_stream_guardrail
        ),
        "output_group_pretransposed_codeword_stream_guardrail": (
            output_group_pretransposed_codeword_stream_guardrail
        ),
        "kblock_output_group_route_fused_stream_guardrail": (
            kblock_output_group_route_fused_stream_guardrail
        ),
        "route_tile_output_swizzle_stream_guardrail": (
            route_tile_output_swizzle_stream_guardrail
        ),
        "token_topk_output_tile_stream_guardrail": (
            token_topk_output_tile_stream_guardrail
        ),
        "token_block_output_group_stream_guardrail": (
            token_block_output_group_stream_guardrail
        ),
        "token_output_stripe_group_stream_guardrail": (
            token_output_stripe_group_stream_guardrail
        ),
        "token_expert_output_block_stream_guardrail": (
            token_expert_output_block_stream_guardrail
        ),
        "token_pair_kblock_accumulator_stream_guardrail": (
            token_pair_kblock_accumulator_stream_guardrail
        ),
        "token_pair_output_group_stream_guardrail": (
            token_pair_output_group_stream_guardrail
        ),
        "token_pair_slot_topk_output_group_stream_guardrail": (
            token_pair_slot_topk_output_group_stream_guardrail
        ),
        "token_pair_slot_topk_codeword_group_pipeline_guardrail": (
            token_pair_slot_topk_codeword_group_pipeline_guardrail
        ),
        "token_pair_slot_topk_scale_slot_broadcast_stream_guardrail": (
            token_pair_slot_topk_scale_slot_broadcast_stream_guardrail
        ),
        "token_pair_slot_topk_route_bucket_codeword_reduce_guardrail": (
            token_pair_slot_topk_route_bucket_codeword_reduce_guardrail
        ),
        "token_pair_slot_topk_kblock_microtile_stream_guardrail": (
            token_pair_slot_topk_kblock_microtile_stream_guardrail
        ),
        "token_pair_slot_topk_output_tile_fused_stream_guardrail": (
            token_pair_slot_topk_output_tile_fused_stream_guardrail
        ),
        "track_b_current_family_verdict": track_b_current_family_verdict,
        "q2_geometry": q2_geometry,
        "benchmark_context": context,
        "e8p_packed_rhs_requirements": _packed_rhs_requirements(
            q2_geometry=q2_geometry,
            e8p_kernel=e8p_kernel,
            benchmark_context=context,
        ),
        "structural_gap": structural_gap,
        "next_track_b_hypothesis": next_track_b_hypothesis,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit GLM-4.5-Air E8P NAX kernel structure against audited q2 NAX geometry."
    )
    parser.add_argument(
        "--metal-source",
        default="native/vq_nax_ext/kernels/nax_fp16_matmul.metal",
    )
    parser.add_argument(
        "--primitive-source",
        default="native/vq_nax_ext/csrc/nax_fp16_primitive.mm",
    )
    parser.add_argument(
        "--q2-jsonl",
        default="artifacts/benchmarks/glm45-air-nax-q2-lldb-summary.jsonl",
    )
    parser.add_argument(
        "--q2-geometry-jsonl",
        help=(
            "Optional LLDB/kernel-symbol q2 geometry JSONL. Use this when --q2-jsonl "
            "is a same-window timing/control file without kernel_names."
        ),
    )
    parser.add_argument(
        "--e8p-analysis-json",
        default="artifacts/quality/glm45-air-e8p-gs352-benchmark-analysis-20260701.json",
    )
    parser.add_argument("--active-route-tile-design-json")
    parser.add_argument("--expert-cohort-design-json")
    parser.add_argument("--route-batch-segmented-design-json")
    parser.add_argument("--component-stream-partial-reduction-design-json")
    parser.add_argument("--token-cohort-design-json")
    parser.add_argument("--token-cohort-mma-codeword-tile-design-json")
    parser.add_argument("--output-stationary-codeword-tile-design-json")
    parser.add_argument("--input-stationary-codeword-tile-design-json")
    parser.add_argument("--expert-kblock-codeword-factor-reuse-design-json")
    parser.add_argument("--route-codeword-lut-accumulate-design-json")
    parser.add_argument("--rowwise-codeword-tile-accumulate-design-json")
    parser.add_argument("--output-tile-local-codeword-lut-design-json")
    parser.add_argument("--route-microtile-codeword-block-reduce-design-json")
    parser.add_argument("--kblock-wavefront-codeword-scan-design-json")
    parser.add_argument("--token-route-output-stripe-pipeline-design-json")
    parser.add_argument("--expert-kblock-scale-slot-stream-design-json")
    parser.add_argument("--scale-group-route-block-reduce-design-json")
    parser.add_argument("--route-block-output-group-stream-design-json")
    parser.add_argument("--output-group-pretransposed-codeword-stream-design-json")
    parser.add_argument("--kblock-output-group-route-fused-stream-design-json")
    parser.add_argument("--route-tile-output-swizzle-stream-design-json")
    parser.add_argument("--token-topk-output-tile-stream-design-json")
    parser.add_argument("--token-block-output-group-stream-design-json")
    parser.add_argument("--token-output-stripe-group-stream-design-json")
    parser.add_argument("--token-expert-output-block-stream-design-json")
    parser.add_argument("--token-pair-kblock-accumulator-stream-design-json")
    parser.add_argument("--token-pair-output-group-stream-design-json")
    parser.add_argument("--token-pair-slot-topk-output-group-stream-design-json")
    parser.add_argument("--token-pair-slot-topk-codeword-group-pipeline-design-json")
    parser.add_argument(
        "--token-pair-slot-topk-scale-slot-broadcast-stream-design-json"
    )
    parser.add_argument(
        "--token-pair-slot-topk-route-bucket-codeword-reduce-design-json"
    )
    parser.add_argument("--token-pair-slot-topk-kblock-microtile-stream-design-json")
    parser.add_argument("--token-pair-slot-topk-output-tile-fused-stream-design-json")
    parser.add_argument("--kernel-name", default="nax_e8p_fp16_sorted_matmul_steel")
    parser.add_argument("--output-json")
    args = parser.parse_args()
    report = build_structure_report(
        metal_source=Path(args.metal_source),
        primitive_source=Path(args.primitive_source) if args.primitive_source else None,
        q2_jsonl=Path(args.q2_jsonl),
        q2_geometry_jsonl=Path(args.q2_geometry_jsonl)
        if args.q2_geometry_jsonl
        else None,
        e8p_analysis_json=Path(args.e8p_analysis_json),
        active_route_tile_design_json=Path(args.active_route_tile_design_json)
        if args.active_route_tile_design_json
        else None,
        expert_cohort_design_json=Path(args.expert_cohort_design_json)
        if args.expert_cohort_design_json
        else None,
        route_batch_segmented_design_json=Path(args.route_batch_segmented_design_json)
        if args.route_batch_segmented_design_json
        else None,
        component_stream_partial_reduction_design_json=Path(
            args.component_stream_partial_reduction_design_json
        )
        if args.component_stream_partial_reduction_design_json
        else None,
        token_cohort_design_json=Path(args.token_cohort_design_json)
        if args.token_cohort_design_json
        else None,
        token_cohort_mma_codeword_tile_design_json=Path(
            args.token_cohort_mma_codeword_tile_design_json
        )
        if args.token_cohort_mma_codeword_tile_design_json
        else None,
        output_stationary_codeword_tile_design_json=Path(
            args.output_stationary_codeword_tile_design_json
        )
        if args.output_stationary_codeword_tile_design_json
        else None,
        input_stationary_codeword_tile_design_json=Path(
            args.input_stationary_codeword_tile_design_json
        )
        if args.input_stationary_codeword_tile_design_json
        else None,
        expert_kblock_codeword_factor_reuse_design_json=Path(
            args.expert_kblock_codeword_factor_reuse_design_json
        )
        if args.expert_kblock_codeword_factor_reuse_design_json
        else None,
        route_codeword_lut_accumulate_design_json=Path(
            args.route_codeword_lut_accumulate_design_json
        )
        if args.route_codeword_lut_accumulate_design_json
        else None,
        rowwise_codeword_tile_accumulate_design_json=Path(
            args.rowwise_codeword_tile_accumulate_design_json
        )
        if args.rowwise_codeword_tile_accumulate_design_json
        else None,
        output_tile_local_codeword_lut_design_json=Path(
            args.output_tile_local_codeword_lut_design_json
        )
        if args.output_tile_local_codeword_lut_design_json
        else None,
        route_microtile_codeword_block_reduce_design_json=Path(
            args.route_microtile_codeword_block_reduce_design_json
        )
        if args.route_microtile_codeword_block_reduce_design_json
        else None,
        kblock_wavefront_codeword_scan_design_json=Path(
            args.kblock_wavefront_codeword_scan_design_json
        )
        if args.kblock_wavefront_codeword_scan_design_json
        else None,
        token_route_output_stripe_pipeline_design_json=Path(
            args.token_route_output_stripe_pipeline_design_json
        )
        if args.token_route_output_stripe_pipeline_design_json
        else None,
        expert_kblock_scale_slot_stream_design_json=Path(
            args.expert_kblock_scale_slot_stream_design_json
        )
        if args.expert_kblock_scale_slot_stream_design_json
        else None,
        scale_group_route_block_reduce_design_json=Path(
            args.scale_group_route_block_reduce_design_json
        )
        if args.scale_group_route_block_reduce_design_json
        else None,
        route_block_output_group_stream_design_json=Path(
            args.route_block_output_group_stream_design_json
        )
        if args.route_block_output_group_stream_design_json
        else None,
        output_group_pretransposed_codeword_stream_design_json=Path(
            args.output_group_pretransposed_codeword_stream_design_json
        )
        if args.output_group_pretransposed_codeword_stream_design_json
        else None,
        kblock_output_group_route_fused_stream_design_json=Path(
            args.kblock_output_group_route_fused_stream_design_json
        )
        if args.kblock_output_group_route_fused_stream_design_json
        else None,
        route_tile_output_swizzle_stream_design_json=Path(
            args.route_tile_output_swizzle_stream_design_json
        )
        if args.route_tile_output_swizzle_stream_design_json
        else None,
        token_topk_output_tile_stream_design_json=Path(
            args.token_topk_output_tile_stream_design_json
        )
        if args.token_topk_output_tile_stream_design_json
        else None,
        token_block_output_group_stream_design_json=Path(
            args.token_block_output_group_stream_design_json
        )
        if args.token_block_output_group_stream_design_json
        else None,
        token_output_stripe_group_stream_design_json=Path(
            args.token_output_stripe_group_stream_design_json
        )
        if args.token_output_stripe_group_stream_design_json
        else None,
        token_expert_output_block_stream_design_json=Path(
            args.token_expert_output_block_stream_design_json
        )
        if args.token_expert_output_block_stream_design_json
        else None,
        token_pair_kblock_accumulator_stream_design_json=Path(
            args.token_pair_kblock_accumulator_stream_design_json
        )
        if args.token_pair_kblock_accumulator_stream_design_json
        else None,
        token_pair_output_group_stream_design_json=Path(
            args.token_pair_output_group_stream_design_json
        )
        if args.token_pair_output_group_stream_design_json
        else None,
        token_pair_slot_topk_output_group_stream_design_json=Path(
            args.token_pair_slot_topk_output_group_stream_design_json
        )
        if args.token_pair_slot_topk_output_group_stream_design_json
        else None,
        token_pair_slot_topk_codeword_group_pipeline_design_json=Path(
            args.token_pair_slot_topk_codeword_group_pipeline_design_json
        )
        if args.token_pair_slot_topk_codeword_group_pipeline_design_json
        else None,
        token_pair_slot_topk_scale_slot_broadcast_stream_design_json=Path(
            args.token_pair_slot_topk_scale_slot_broadcast_stream_design_json
        )
        if args.token_pair_slot_topk_scale_slot_broadcast_stream_design_json
        else None,
        token_pair_slot_topk_route_bucket_codeword_reduce_design_json=Path(
            args.token_pair_slot_topk_route_bucket_codeword_reduce_design_json
        )
        if args.token_pair_slot_topk_route_bucket_codeword_reduce_design_json
        else None,
        token_pair_slot_topk_kblock_microtile_stream_design_json=Path(
            args.token_pair_slot_topk_kblock_microtile_stream_design_json
        )
        if args.token_pair_slot_topk_kblock_microtile_stream_design_json
        else None,
        token_pair_slot_topk_output_tile_fused_stream_design_json=Path(
            args.token_pair_slot_topk_output_tile_fused_stream_design_json
        )
        if args.token_pair_slot_topk_output_tile_fused_stream_design_json
        else None,
        kernel_name=args.kernel_name,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        Path(args.output_json).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
