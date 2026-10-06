from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_analyzer():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "analyze_glm45_air_e8p_kernel_structure.py"
    )
    spec = importlib.util.spec_from_file_location(
        "analyze_glm45_air_e8p_kernel_structure", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_q2_nax_kernel_name_extracts_geometry() -> None:
    analyzer = _load_analyzer()

    geometry = analyzer.parse_q2_nax_kernel_name(
        "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2_align_M_t"
    )

    assert geometry == {
        "group_size": 128,
        "bits": 2,
        "bm": 64,
        "bn": 64,
        "bk": 64,
        "wm": 2,
        "wn": 2,
    }


def test_audit_e8p_steel_kernel_reports_decoded_b_staging_and_barriers() -> None:
    analyzer = _load_analyzer()
    source = """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  for (uint k_block = 0; k_block < K; k_block += mlx_vq_nax_bk_tile) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    Ws[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(Ws + kk);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
[[kernel]] void other_kernel() {}
"""

    report = analyzer.audit_e8p_kernel_source(source, "nax_e8p_fp16_sorted_matmul_steel")

    assert report["kernel_name"] == "nax_e8p_fp16_sorted_matmul_steel"
    assert report["decoded_b_threadgroup_staging"] is True
    assert report["threadgroup_barrier_count"] == 3
    assert report["has_k_block_decode_loop"] is True
    assert report["loads_btile_from_threadgroup_ws"] is True
    assert "decoded-B threadgroup staging" in report["structural_risk"]


def test_audit_e8p_shared_decode_kernel_rejects_per_fragment_factor_reconstruction() -> None:
    analyzer = _load_analyzer()
    source = """
[[kernel]] void nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_matmul() {
  simdgroup_matrix_storage<half, 8, 8> b_t[4];
  for (uint k_block = 0; k_block < K; k_block += mlx_vq_nax_bk_tile) {
    for (uint frag_n = 0; frag_n < 4; ++frag_n) {
      for (uint kk = 0; kk < 8; ++kk) {
        uint sign_byte = sign_byte_lut[packed_code & 15u];
        uint abs_index = abs_index_lut[packed_code & 15u];
        half decoded = half(mlx_vq_decode_e8p_split_value(abs_index, sign_byte, codebook));
        b_t[frag_n].thread_elements()[kk] = decoded * scale_tiles[frag_n];
      }
    }
    mpp::tensor_ops::matmul2d<8, 8, 8, half, float> matmul_op;
    matmul_op.run(a_t, b_t[0], acc);
  }
}
[[kernel]] void other_kernel() {}
"""

    report = analyzer.audit_e8p_shared_decode_kernel_source(
        source,
        "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_matmul",
    )

    assert report["kernel_name"] == "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_matmul"
    assert report["has_tensorops_matmul"] is True
    assert report["has_threadgroup_half_staging"] is False
    assert report["threadgroup_barrier_count"] == 0
    assert report["factor_lut_load_count"] == 2
    assert report["reconstructs_b_fragments_from_factor_luts"] is True
    assert report["decision"] == "reject_per_fragment_factor_reconstruction_shared_decode"
    assert report["required_next_features"] == [
        "broader_expert_output_factor_decode_reuse",
        "avoid_per_fragment_factor_reconstruction",
        "preserve_compressed_rhs_storage",
        "change_rhs_layout_or_kernel_family",
    ]


def test_audit_component_stream_shared_decode_rejects_renamed_tensorops_retime() -> None:
    analyzer = _load_analyzer()
    metal_source = """
[[kernel]] void nax_e8p_component_stream_rhs_sorted_shared_decode_matmul() {
  simdgroup_matrix_storage<half, 8, 8> b_t;
  uint sign_bit = uint(sign_component_bits[component_base + component_offset]);
  uint abs_index = uint(abs_index_tiles[codeword_base + codeword_index]);
  int scale_slot = component_scale_slots[k_block * component_count + component_offset];
  float scale_f = float(scale_tiles[scale_base + scale_slot]);
  half decoded = half(mlx_vq_decode_e8p_component(sign_bit, abs_index, codebook, component_offset));
  b_t.thread_elements()[0] = decoded * half(scale_f);
  mpp::tensor_ops::matmul2d<8, 8, 8, half, float> matmul_op;
  matmul_op.run(a_t, b_t, acc);
}
[[kernel]] void other_kernel() {}
"""
    primitive_source = """
array e8p_component_stream_rhs_sorted_shared_decode_matmul() {
  auto* pso = cache.get("nax_e8p_component_stream_rhs_sorted_shared_decode_matmul");
  encoder.dispatch_threadgroups(MTL::Size::Make(num_route_tiles, n_tiles, k_blocks),
                                MTL::Size::Make(64, 1, 1));
}
"""

    guard = analyzer.audit_e8p_component_stream_shared_decode_source(
        metal_source=metal_source,
        primitive_source=primitive_source,
    )

    assert guard["decision"] == (
        "reject_component_stream_shared_decode_per_fragment_reconstruction"
    )
    assert guard["kernel_present"] is True
    assert guard["has_tensorops_matmul"] is True
    assert guard["uses_component_stream_storage"] is True
    assert guard["shared_decode_cache_present"] is False
    assert guard["reconstructs_components_per_fragment"] is True
    assert guard["speed_claim"] is False
    assert "do_not_retime_component_stream_tensorops_unchanged" in guard[
        "rejected_next_steps"
    ]


def test_audit_component_stream_shared_decode_accepts_scalar_cache_scaffold_before_parallel_schedule() -> None:
    analyzer = _load_analyzer()
    metal_source = """
[[kernel]] void nax_e8p_component_stream_rhs_sorted_shared_decode_matmul() {
  float component_decode_cache[4];
  uint sign_bit = uint(sign_component_bits[component_base + component_offset]);
  uint abs_index = uint(abs_index_tiles[codeword_base + codeword_index]);
  int scale_slot = component_scale_slots[k_block * component_count + component_offset];
  float scale_f = float(scale_tiles[scale_base + scale_slot]);
  component_decode_cache[component_lane] =
      float(mlx_vq_decode_e8p_component(sign_bit, abs_index, codebook, component_offset));
  accum += float(sorted_x[route * K + k]) * component_decode_cache[component_lane] * scale_f;
}
[[kernel]] void other_kernel() {}
"""
    primitive_source = """
array e8p_component_stream_rhs_sorted_shared_decode_matmul() {
  auto* pso = cache.get("nax_e8p_component_stream_rhs_sorted_shared_decode_matmul");
  encoder.dispatch_threadgroups(MTL::Size::Make((output_dims + 15u) / 16u,
                                                (route_count + 15u) / 16u,
                                                1),
                                MTL::Size::Make(16, 16, 1));
}
"""

    guard = analyzer.audit_e8p_component_stream_shared_decode_source(
        metal_source=metal_source,
        primitive_source=primitive_source,
    )

    assert guard["decision"] == (
        "component_stream_shared_decode_scalar_cache_scaffold_present"
    )
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["shared_decode_cache_present"] is True
    assert guard["has_tensorops_matmul"] is False
    assert guard["speed_claim"] is False
    assert guard["required_next_features"] == [
        "prove_native_shared_decode_cache_parity",
        "replace_scalar_shared_decode_body_with_valid_parallel_schedule",
        "prove_artifact_shared_decode_cache_parity_after_schedule_change",
    ]
    assert "do_not_benchmark_scalar_shared_decode_cache_scaffold" in guard[
        "rejected_next_steps"
    ]


def test_audit_token_cohort_mma_codeword_tile_requires_distinct_source_guardrail() -> None:
    analyzer = _load_analyzer()
    metal_source = """
[[kernel]] void nax_e8p_token_cohort_codeword_stream_rhs_sorted_matmul() {
  uint token_cohort_offsets = 0;
  uint token_cohort_counts = 0;
  uint token_cohort_active_expert_ids = 0;
  uint token_cohort_route_slot_ids = 0;
  device const uint16_t* code_tiles;
  device const half* codeword_scale_slots;
  device const half* scale_tiles;
}
[[kernel]] void other_kernel() {}
"""
    primitive_source = """
array e8p_token_cohort_codeword_stream_rhs_sorted_matmul() {
  auto* pso = cache.get("nax_e8p_token_cohort_codeword_stream_rhs_sorted_matmul");
  encoder.dispatch_threadgroups(MTL::Size::Make(token_cohort_count, active_expert_count, k_blocks),
                                MTL::Size::Make(64, 1, 1));
}
"""

    guard = analyzer.audit_e8p_token_cohort_mma_codeword_tile_source(
        metal_source=metal_source,
        primitive_source=primitive_source,
    )

    assert guard["decision"] == "missing_token_cohort_mma_codeword_tile_source"
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["passes_contract"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert "add_token_cohort_mma_codeword_tile_native_source_guardrail" in guard[
        "required_next_features"
    ]
    assert "do_not_retime_token_cohort_codeword_stream" in guard[
        "rejected_next_steps"
    ]


def test_audit_token_cohort_mma_codeword_tile_accepts_mma_tile_source_guardrail() -> None:
    analyzer = _load_analyzer()
    metal_source = """
[[kernel]] void nax_e8p_token_cohort_mma_codeword_tile_rhs_sorted_matmul() {
  uint token_cohort_offsets = 0;
  uint token_cohort_counts = 0;
  uint token_cohort_active_expert_ids = 0;
  uint token_cohort_route_slot_ids = 0;
  uint output_tile_id = threadgroup_position_in_grid.y;
  uint codeword_tile_id = threadgroup_position_in_grid.w;
  device const uint16_t* codeword_tiles;
  device const half* codeword_scale_slots;
  device const half* scale_tiles;
  float mma_accum = 0.0f;
  mpp::tensor_ops::matmul2d<8, 8, 8, half, float> matmul_op;
  (void)matmul_op;
  (void)mma_accum;
}
[[kernel]] void other_kernel() {}
"""
    primitive_source = """
array e8p_token_cohort_mma_codeword_tile_rhs_sorted_matmul() {
  auto* pso = cache.get("nax_e8p_token_cohort_mma_codeword_tile_rhs_sorted_matmul");
  // token_cohorts_x_output_tiles_x_active_experts_x_k_blocks_x_codeword_tiles
  encoder.dispatch_threadgroups(MTL::Size::Make(token_cohort_count,
                                                output_tile_count,
                                                active_expert_count * k_blocks * codeword_tile_count),
                                MTL::Size::Make(64, 1, 1));
}
"""

    guard = analyzer.audit_e8p_token_cohort_mma_codeword_tile_source(
        metal_source=metal_source,
        primitive_source=primitive_source,
    )

    assert guard["decision"] == (
        "token_cohort_mma_codeword_tile_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "token_cohorts_x_output_tiles_x_active_experts_x_k_blocks_x_codeword_tiles"
    )
    assert guard["preserves_token_cohort_route_descriptors"] is True
    assert guard["preserves_exact_route_slot_indirection"] is True
    assert guard["uses_output_tile_axis"] is True
    assert guard["uses_codeword_tile_axis"] is True
    assert guard["uses_mma_accumulation"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["uses_rejected_codeword_stream_schedule"] is False
    assert guard["required_next_features"] == [
        "prove_token_cohort_mma_codeword_tile_native_parity",
        "prove_air_down_group_size_352_artifact_parity",
        "run_same_window_q2_speed_packet_after_parity",
    ]


def test_build_structure_report_routes_token_cohort_mma_source_present_on_repo_source() -> None:
    analyzer = _load_analyzer()
    root = Path(__file__).resolve().parents[1]

    report = analyzer.build_structure_report(
        metal_source=root / "native/vq_nax_ext/kernels/nax_fp16_matmul.metal",
        primitive_source=root / "native/vq_nax_ext/csrc/nax_fp16_primitive.mm",
        q2_jsonl=root / "artifacts/benchmarks/glm45-air-nax-q2-lldb-summary.jsonl",
        e8p_analysis_json=(
            root
            / "artifacts/quality/glm45-air-e8p-gs352-benchmark-analysis-20260701.json"
        ),
        token_cohort_design_json=(
            root
            / "artifacts/quality/glm45-air-e8p-token-cohort-codeword-stream-design-20260703.json"
        ),
        token_cohort_mma_codeword_tile_design_json=(
            root
            / "artifacts/quality/glm45-air-e8p-token-cohort-mma-codeword-tile-design-20260703.json"
        ),
    )

    guard = report["token_cohort_mma_codeword_tile_guardrail"]
    assert guard["decision"] == (
        "token_cohort_mma_codeword_tile_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["uses_output_tile_axis"] is True
    assert guard["uses_codeword_tile_axis"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["uses_mma_accumulation"] is True
    assert guard["uses_rejected_codeword_stream_schedule"] is False
    assert report["next_track_b_hypothesis"] == (
        "prove_token_cohort_mma_codeword_tile_native_parity"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_audit_output_stationary_codeword_tile_requires_distinct_source_guardrail() -> None:
    analyzer = _load_analyzer()
    metal_source = """
[[kernel]] void nax_e8p_token_cohort_mma_codeword_tile_rhs_sorted_matmul() {
  uint token_cohort_offsets = 0;
  uint token_cohort_counts = 0;
  uint token_cohort_active_expert_ids = 0;
  uint token_cohort_route_slot_ids = 0;
  uint output_tile_id = 0;
  uint codeword_tile_id = 0;
  device const uint16_t* codeword_tiles;
  device const half* codeword_scale_slots;
  device const half* scale_tiles;
  float mma_accum = 0.0f;
}
[[kernel]] void other_kernel() {}
"""
    primitive_source = """
array e8p_token_cohort_mma_codeword_tile_rhs_sorted_matmul() {
  auto* pso = cache.get("nax_e8p_token_cohort_mma_codeword_tile_rhs_sorted_matmul");
}
"""

    guard = analyzer.audit_e8p_output_stationary_codeword_tile_source(
        metal_source=metal_source,
        primitive_source=primitive_source,
    )

    assert guard["decision"] == "missing_output_stationary_codeword_tile_source"
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["passes_contract"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert "add_output_stationary_codeword_tile_native_source_guardrail" in guard[
        "required_next_features"
    ]
    assert "do_not_retime_token_cohort_mma_codeword_tile" in guard[
        "rejected_next_steps"
    ]


def test_build_structure_report_routes_output_stationary_source_present_on_repo_source() -> None:
    analyzer = _load_analyzer()
    root = Path(__file__).resolve().parents[1]

    report = analyzer.build_structure_report(
        metal_source=root / "native/vq_nax_ext/kernels/nax_fp16_matmul.metal",
        primitive_source=root / "native/vq_nax_ext/csrc/nax_fp16_primitive.mm",
        q2_jsonl=root / "artifacts/benchmarks/glm45-air-nax-q2-lldb-summary.jsonl",
        e8p_analysis_json=(
            root
            / "artifacts/quality/glm45-air-e8p-gs352-benchmark-analysis-20260701.json"
        ),
        token_cohort_mma_codeword_tile_design_json=(
            root
            / "artifacts/quality/glm45-air-e8p-token-cohort-mma-codeword-tile-design-20260703.json"
        ),
        output_stationary_codeword_tile_design_json=(
            root
            / "artifacts/quality/glm45-air-e8p-output-stationary-codeword-tile-design-20260703.json"
        ),
    )

    guard = report["output_stationary_codeword_tile_guardrail"]
    assert guard["decision"] == (
        "output_stationary_codeword_tile_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert report["next_track_b_hypothesis"] == (
        "prove_output_stationary_codeword_tile_native_parity"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_real_current_output_stationary_codeword_tile_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()
    root = Path(__file__).resolve().parents[1]

    report = analyzer.build_structure_report(
        metal_source=root / "native/vq_nax_ext/kernels/nax_fp16_matmul.metal",
        primitive_source=root / "native/vq_nax_ext/csrc/nax_fp16_primitive.mm",
        q2_jsonl=root / "artifacts/benchmarks/glm45-air-nax-q2-lldb-summary.jsonl",
        e8p_analysis_json=(
            root
            / "artifacts/quality/glm45-air-e8p-gs352-benchmark-analysis-20260701.json"
        ),
        output_stationary_codeword_tile_design_json=(
            root
            / "artifacts/quality/glm45-air-e8p-output-stationary-codeword-tile-design-20260703.json"
        ),
    )

    guard = report["output_stationary_codeword_tile_guardrail"]
    assert guard["decision"] == (
        "output_stationary_codeword_tile_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "output_tiles_x_route_batches_x_active_experts_x_k_blocks_x_codeword_tiles"
    )
    assert guard["preserves_output_stationary_route_batches"] is True
    assert guard["preserves_exact_route_slot_indirection"] is True
    assert guard["uses_output_tile_axis"] is True
    assert guard["uses_route_batch_axis"] is True
    assert guard["uses_codeword_tile_axis"] is True
    assert guard["keeps_output_stationary_accumulation"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["uses_rejected_token_cohort_mma_schedule"] is False
    assert report["next_track_b_hypothesis"] == (
        "prove_output_stationary_codeword_tile_native_parity"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_audit_input_stationary_codeword_tile_requires_distinct_source_guardrail() -> None:
    analyzer = _load_analyzer()
    metal_source = """
[[kernel]] void nax_e8p_output_stationary_codeword_tile_rhs_sorted_matmul() {
  uint output_stationary_route_batch_offsets = 0;
  uint output_stationary_route_batch_counts = 0;
  uint output_stationary_route_batch_active_expert_ids = 0;
  uint output_stationary_route_batch_route_slot_ids = 0;
  uint output_tile_id = 0;
  uint route_batch_id = 0;
  uint codeword_tile_id = 0;
  device const uint16_t* codeword_tiles;
  device const half* codeword_scale_slots;
  device const half* scale_tiles;
  float output_stationary_accum = 0.0f;
}
[[kernel]] void other_kernel() {}
"""
    primitive_source = """
array e8p_output_stationary_codeword_tile_rhs_sorted_matmul() {
  auto* pso = cache.get("nax_e8p_output_stationary_codeword_tile_rhs_sorted_matmul");
}
"""

    guard = analyzer.audit_e8p_input_stationary_codeword_tile_source(
        metal_source=metal_source,
        primitive_source=primitive_source,
    )

    assert guard["decision"] == "missing_input_stationary_codeword_tile_source"
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["passes_contract"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert "add_input_stationary_codeword_tile_native_source_guardrail" in guard[
        "required_next_features"
    ]
    assert "do_not_retime_output_stationary_codeword_tile" in guard[
        "rejected_next_steps"
    ]


def test_build_structure_report_surfaces_missing_input_stationary_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    root = Path(__file__).resolve().parents[1]
    metal_source = tmp_path / "nax_fp16_matmul.metal"
    metal_source.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[64];
  for (uint k_block = 0; k_block < K; k_block += mlx_vq_nax_bk_tile) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    Ws[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u));
    Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(Ws + kk);
  }
}
[[kernel]] void nax_e8p_output_stationary_codeword_tile_rhs_sorted_matmul() {
  uint output_stationary_route_batch_offsets = 0;
  uint output_stationary_route_batch_counts = 0;
  uint output_stationary_route_batch_active_expert_ids = 0;
  uint output_stationary_route_batch_route_slot_ids = 0;
}
[[kernel]] void other_kernel() {}
""",
        encoding="utf-8",
    )
    primitive_source = tmp_path / "nax_fp16_primitive.mm"
    primitive_source.write_text(
        """
array e8p_output_stationary_codeword_tile_rhs_sorted_matmul() {
  auto* pso = cache.get("nax_e8p_output_stationary_codeword_tile_rhs_sorted_matmul");
}
""",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal_source,
        primitive_source=primitive_source,
        q2_jsonl=root / "artifacts/benchmarks/glm45-air-nax-q2-lldb-summary.jsonl",
        e8p_analysis_json=(
            root
            / "artifacts/quality/glm45-air-e8p-gs352-benchmark-analysis-20260701.json"
        ),
        input_stationary_codeword_tile_design_json=(
            root
            / "artifacts/quality/glm45-air-e8p-input-stationary-codeword-tile-design-20260703.json"
        ),
    )

    guard = report["input_stationary_codeword_tile_guardrail"]
    assert guard["decision"] == "missing_input_stationary_codeword_tile_source"
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert report["next_track_b_hypothesis"] == (
        "implement_input_stationary_codeword_tile_source_guardrail"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_real_current_input_stationary_codeword_tile_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()
    root = Path(__file__).resolve().parents[1]

    report = analyzer.build_structure_report(
        metal_source=root / "native/vq_nax_ext/kernels/nax_fp16_matmul.metal",
        primitive_source=root / "native/vq_nax_ext/csrc/nax_fp16_primitive.mm",
        q2_jsonl=root / "artifacts/benchmarks/glm45-air-nax-q2-lldb-summary.jsonl",
        e8p_analysis_json=(
            root
            / "artifacts/quality/glm45-air-e8p-gs352-benchmark-analysis-20260701.json"
        ),
        input_stationary_codeword_tile_design_json=(
            root
            / "artifacts/quality/glm45-air-e8p-input-stationary-codeword-tile-design-20260703.json"
        ),
    )

    guard = report["input_stationary_codeword_tile_guardrail"]
    assert (
        guard["decision"]
        == "input_stationary_codeword_tile_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "input_tiles_x_route_batches_x_active_experts_x_output_tiles_x_codeword_tiles"
    )
    assert guard["preserves_input_stationary_route_batches"] is True
    assert guard["preserves_exact_route_slot_indirection"] is True
    assert guard["uses_input_tile_axis"] is True
    assert guard["uses_output_tile_axis"] is True
    assert guard["uses_codeword_tile_axis"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert report["next_track_b_hypothesis"] == (
        "prove_input_stationary_codeword_tile_native_parity"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_audit_expert_kblock_codeword_factor_reuse_requires_distinct_source_guardrail() -> None:
    analyzer = _load_analyzer()
    metal_source = """
[[kernel]] void nax_e8p_input_stationary_codeword_tile_rhs_sorted_matmul() {
  uint input_stationary_route_batch_offsets = 0;
  uint input_stationary_route_batch_counts = 0;
  uint input_stationary_route_batch_active_expert_ids = 0;
  uint input_stationary_route_batch_route_slot_ids = 0;
}
"""
    primitive_source = """
array e8p_input_stationary_codeword_tile_rhs_sorted_matmul() {
  auto* pso = cache.get("nax_e8p_input_stationary_codeword_tile_rhs_sorted_matmul");
}
"""

    guard = analyzer.audit_e8p_expert_kblock_codeword_factor_reuse_source(
        metal_source=metal_source,
        primitive_source=primitive_source,
    )

    assert guard["decision"] == "missing_expert_kblock_codeword_factor_reuse_source"
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert "add_expert_kblock_codeword_factor_reuse_native_source_guardrail" in guard[
        "required_next_features"
    ]
    assert "do_not_benchmark_expert_kblock_codeword_factor_reuse_before_source_guardrail" in guard[
        "rejected_next_steps"
    ]
    assert "do_not_retime_input_stationary_codeword_tile" in guard[
        "rejected_next_steps"
    ]
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_build_structure_report_surfaces_missing_expert_kblock_codeword_factor_reuse_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    root = Path(__file__).resolve().parents[1]
    metal_source = tmp_path / "nax_fp16_matmul.metal"
    metal_source.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[64];
  Ws[0] = half(0.0);
}
[[kernel]] void nax_e8p_input_stationary_codeword_tile_rhs_sorted_matmul() {
  uint input_stationary_route_batch_offsets = 0;
  uint input_stationary_route_batch_counts = 0;
  uint input_stationary_route_batch_active_expert_ids = 0;
  uint input_stationary_route_batch_route_slot_ids = 0;
}
""",
        encoding="utf-8",
    )
    primitive_source = tmp_path / "nax_fp16_primitive.mm"
    primitive_source.write_text(
        """
array e8p_input_stationary_codeword_tile_rhs_sorted_matmul() {
  auto* pso = cache.get("nax_e8p_input_stationary_codeword_tile_rhs_sorted_matmul");
}
""",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal_source,
        primitive_source=primitive_source,
        q2_jsonl=root / "artifacts/benchmarks/glm45-air-nax-q2-lldb-summary.jsonl",
        e8p_analysis_json=(
            root
            / "artifacts/quality/glm45-air-e8p-gs352-benchmark-analysis-20260701.json"
        ),
        expert_kblock_codeword_factor_reuse_design_json=(
            root
            / "artifacts/quality/glm45-air-e8p-expert-kblock-codeword-factor-reuse-design-20260703.json"
        ),
    )

    guard = report["expert_kblock_codeword_factor_reuse_guardrail"]
    assert guard["decision"] == "missing_expert_kblock_codeword_factor_reuse_source"
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert report["next_track_b_hypothesis"] == (
        "implement_expert_kblock_codeword_factor_reuse_source_guardrail"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_real_current_expert_kblock_codeword_factor_reuse_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()
    root = Path(__file__).resolve().parents[1]

    report = analyzer.build_structure_report(
        metal_source=root / "native/vq_nax_ext/kernels/nax_fp16_matmul.metal",
        primitive_source=root / "native/vq_nax_ext/csrc/nax_fp16_primitive.mm",
        q2_jsonl=root / "artifacts/benchmarks/glm45-air-nax-q2-lldb-summary.jsonl",
        e8p_analysis_json=(
            root
            / "artifacts/quality/glm45-air-e8p-gs352-benchmark-analysis-20260701.json"
        ),
        expert_kblock_codeword_factor_reuse_design_json=(
            root
            / "artifacts/quality/glm45-air-e8p-expert-kblock-codeword-factor-reuse-design-20260703.json"
        ),
    )

    guard = report["expert_kblock_codeword_factor_reuse_guardrail"]
    assert guard["decision"] == (
        "expert_kblock_codeword_factor_reuse_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "experts_x_k_blocks_x_output_tiles_x_route_tiles_x_codeword_tiles"
    )
    assert guard["uses_expert_axis"] is True
    assert guard["uses_kblock_axis"] is True
    assert guard["uses_output_tile_axis"] is True
    assert guard["uses_route_tile_axis"] is True
    assert guard["uses_codeword_tile_axis"] is True
    assert guard["reuses_factor_decode_across_output_tiles"] is True
    assert guard["streams_compressed_codeword_factor_tiles"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["reconstructs_per_fragment"] is False
    assert guard["uses_rejected_input_stationary_schedule"] is False
    assert guard["uses_historical_v2_schedule"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert report["next_track_b_hypothesis"] == (
        "prove_expert_kblock_codeword_factor_reuse_native_parity"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_audit_e8p_shared_n_decode_kernel_rejects_lane_local_buffering() -> None:
    analyzer = _load_analyzer()
    source = """
[[kernel]] void nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_matmul() {
  threadgroup half shared_b[2 * mlx_vq_nax_shared_lanes * 2 * mlx_vq_nax_elems_per_frag];
  if (simdgroup_id < 2u) {
    uint sign_slot = uint(sign_byte_slots[factor_index]);
    uint abs_slot = uint(abs_index_slots[factor_index]);
    uint signs = uint(sign_byte_lut[lut_base + sign_slot]);
    uint abs_idx = uint(abs_index_lut[lut_base + abs_slot]);
    float scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
    shared_b[shared_group_base + elem] = half(
        mlx_vq_decode_e8p_split_value(signs, abs_idx, codebook, dim) * scale_f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto b_t = matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  b_t[n_frag * mlx_vq_nax_elems_per_frag + elem] = shared_b[shared_group_base + elem];
  mpp::tensor_ops::matmul2d<8, 8, 8, half, float> matmul_op;
  matmul_op.run(a_t, b_t, c_t);
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
[[kernel]] void other_kernel() {}
"""

    report = analyzer.audit_e8p_shared_n_decode_kernel_source(
        source,
        "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_matmul",
    )

    assert report["kernel_name"] == (
        "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_matmul"
    )
    assert report["has_tensorops_matmul"] is True
    assert report["has_lane_local_shared_b"] is True
    assert report["threadgroup_barrier_count"] == 2
    assert report["uses_factor_luts"] is True
    assert report["loads_b_fragments_from_shared_b"] is True
    assert report["decision"] == "reject_lane_local_shared_n_decode"
    assert report["required_next_features"] == [
        "avoid_lane_local_barrier_fragment_buffer",
        "broader_expert_output_factor_decode_reuse",
        "avoid_per_fragment_factor_reconstruction",
        "preserve_compressed_rhs_storage",
        "change_rhs_layout_or_kernel_family",
    ]


def test_audit_e8p_packed_rhs_tiled_family_rejects_decoded_b_staging() -> None:
    analyzer = _load_analyzer()
    source = """
[[kernel]] void nax_e8p_packed_rhs_sorted_tiled_matmul() {
  threadgroup half staged_b[mlx_vq_nax_bk_tile * mlx_vq_nax_bn_tile];
  staged_b[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto b_t = matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  b_t.thread_elements()[elem] = staged_b[kk * mlx_vq_nax_bn_tile + n_local];
  matmul_op.run(a_t, b_t, c_t);
}
[[kernel]] void nax_e8p_packed_rhs_sorted_tiled_k128_matmul() {
  threadgroup half staged_b[2u * mlx_vq_nax_bk_tile * mlx_vq_nax_bn_tile];
  staged_b[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto b_t = matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  b_t.thread_elements()[elem] = staged_b[stage_offset + kk * mlx_vq_nax_bn_tile + n_local];
  matmul_op.run(a_t, b_t, c_t);
}
[[kernel]] void other_kernel() {}
"""

    report = analyzer.audit_e8p_packed_rhs_tiled_family_source(source)

    assert report["decision"] == "reject_decoded_b_staged_packed_rhs_family"
    assert report["rejected_kernels"] == [
        "nax_e8p_packed_rhs_sorted_tiled_matmul",
        "nax_e8p_packed_rhs_sorted_tiled_k128_matmul",
    ]
    assert report["present_kernel_count"] == 2
    assert report["rejected_kernel_count"] == 2
    assert report["required_next_features"] == [
        "avoid_decoded_b_threadgroup_staging",
        "reuse_decode_outside_threadgroup_b_tile",
        "preserve_compressed_rhs_storage",
        "change_rhs_layout_or_kernel_family",
    ]


def test_audit_e8p_sign_nibble_schedule_marks_scalar_route_path_correctness_only() -> None:
    analyzer = _load_analyzer()
    source = """
[[kernel]] void nax_e8p_sign_nibble_abs_index_rhs_sorted_matmul() {
  uint low = sign_low_nibble_tiles[tile_offset];
  uint high = sign_high_nibble_tiles[tile_offset];
  uint signs = (low & 15u) | ((high & 15u) << 4u);
  half decoded = mlx_vq_decode_e8p_split_value(abs_index_tiles[tile_offset], signs, codebook);
  result += float(decoded * scale_tiles[scale_base + codeword_scale_slots[kw]]);
}
[[kernel]] void other_kernel() {}
"""

    report = analyzer.audit_e8p_sign_nibble_schedule_source(source)

    assert report["scalar_sorted_route_kernel_present"] is True
    assert report["q2_shaped_tensorops_kernel_present"] is False
    assert report["decision"] == "missing_q2_shaped_sign_nibble_tensorops_schedule"
    assert report["required_next_features"] == [
        "match_q2_bk64_bn64_tensorops_geometry",
        "decode_nibble_sign_masks_without_per_fragment_sign_byte_reconstruction",
        "avoid_decoded_b_threadgroup_staging",
        "avoid_lane_local_barrier_fragment_buffer",
        "preserve_compressed_rhs_storage",
        "preserve_codeword_scale_slots",
    ]


def test_audit_e8p_sign_nibble_schedule_rejects_decoded_b_staged_tensorops() -> None:
    analyzer = _load_analyzer()
    source = """
[[kernel]] void nax_e8p_sign_nibble_abs_index_rhs_sorted_tiled_matmul() {
  threadgroup half staged_b[mlx_vq_nax_bk_tile * mlx_vq_nax_bn_tile];
  uint low = sign_low_nibble_tiles[tile_offset];
  uint high = sign_high_nibble_tiles[tile_offset];
  uint signs = (low & 15u) | ((high & 15u) << 4u);
  staged_b[base] = half(mlx_vq_decode_e8p_split_value(abs_index_tiles[tile_offset], signs, codebook));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto b_t = matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  b_t.thread_elements()[elem] = staged_b[kk * mlx_vq_nax_bn_tile + n_local];
  matmul_op.run(a_t, b_t, c_t);
}
[[kernel]] void other_kernel() {}
"""

    report = analyzer.audit_e8p_sign_nibble_schedule_source(source)

    assert report["q2_shaped_tensorops_kernel_present"] is True
    assert report["rejected_kernel_count"] == 1
    assert report["kernel_reports"][0]["reconstructs_sign_byte_from_nibbles"] is True
    assert report["kernel_reports"][0]["has_threadgroup_staged_b"] is True
    assert report["kernel_reports"][0]["loads_b_fragments_from_staged_b"] is True
    assert report["decision"] == "reject_decoded_b_staged_sign_nibble_schedule"


def test_audit_e8p_sign_nibble_schedule_rejects_per_fragment_tensorops_decode() -> None:
    analyzer = _load_analyzer()
    source = """
[[kernel]] void nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_matmul() {
  auto b_t = matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += 16u) {
      uint low = uint(sign_low_nibble_tiles[factor_index]) & 0xFu;
      uint high = uint(sign_high_nibble_tiles[factor_index]) & 0xFu;
      uint abs_idx = uint(abs_index_tiles[factor_index]);
      uint parity = uint(parity_tiles[factor_index]) & 1u;
      float scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
      b_t[n_frag * mlx_vq_nax_elems_per_frag + elem] = half(
          mlx_vq_decode_e8p_nibble_value(
              low, high, abs_idx, parity, codebook, dim) * scale_f);
    }
    mpp::tensor_ops::matmul2d<8, 8, 8, half, float> matmul_op;
    matmul_op.run(a_t, b_t, c_t);
  }
}
[[kernel]] void other_kernel() {}
"""

    report = analyzer.audit_e8p_sign_nibble_schedule_source(
        source,
        benchmark_context={
            "all_lane_s_pass": False,
            "all_parity_pass": False,
            "sign_nibble_tensorops_worst_ratio_to_q2": 14.71,
        },
    )

    kernel_report = report["kernel_reports"][2]
    assert report["q2_shaped_tensorops_kernel_present"] is True
    assert report["rejected_kernel_count"] == 1
    assert report["rejected_kernels"] == [
        "nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_matmul"
    ]
    assert kernel_report["has_tensorops_matmul"] is True
    assert kernel_report["has_threadgroup_staged_b"] is False
    assert kernel_report["threadgroup_barrier_count"] == 0
    assert kernel_report["decodes_nibble_values_per_b_fragment"] is True
    assert kernel_report["uses_codeword_scale_slots"] is False
    assert kernel_report["decision"] == "reject_per_fragment_nibble_tensorops_kernel"
    assert report["decision"] == "reject_per_fragment_nibble_tensorops_schedule"
    assert report["required_next_features"] == [
        "broader_decode_reuse_scope",
        "avoid_per_fragment_nibble_abs_scale_reconstruction",
        "preserve_compressed_rhs_storage",
        "preserve_codeword_scale_slots",
        "change_rhs_layout_or_kernel_family",
    ]


def test_audit_e8p_sign_nibble_schedule_rejects_micro_lut_tensorops_decode() -> None:
    analyzer = _load_analyzer()
    source = """
[[kernel]] void nax_e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_matmul() {
  auto b_t = matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  uint low_slot = uint(sign_low_nibble_slots[factor_index]);
  uint high_slot = uint(sign_high_nibble_slots[factor_index]);
  uint low = uint(sign_low_nibble_lut[low_slot]) & 0xFu;
  uint high = uint(sign_high_nibble_lut[high_slot]) & 0xFu;
  uint signs = low | (high << 4u);
  uint abs_slot = uint(abs_index_slots[factor_index]);
  uint abs_base = ((expert * n_tiles + n_tile) * k_blocks + k_block) * 256u;
  uint abs_idx = uint(abs_index_lut[abs_base + abs_slot]);
  uint parity = mlx_vq_sign_parity8(signs);
  float scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
  b_t[n_frag * mlx_vq_nax_elems_per_frag + elem] = half(
      mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, dim) * scale_f);
  mpp::tensor_ops::matmul2d<8, 8, 8, half, float> matmul_op;
  matmul_op.run(a_t, b_t, c_t);
}
[[kernel]] void other_kernel() {}
"""

    report = analyzer.audit_e8p_sign_nibble_schedule_source(
        source,
        benchmark_context={
            "all_lane_s_pass": False,
            "all_parity_pass": False,
            "sign_nibble_micro_lut_tensorops_worst_ratio_to_q2": 16.41,
        },
    )

    kernel_report = report["kernel_reports"][3]
    assert report["q2_shaped_tensorops_kernel_present"] is True
    assert report["rejected_kernel_count"] == 1
    assert report["rejected_kernels"] == [
        "nax_e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_matmul"
    ]
    assert report["per_fragment_rejected_kernels"] == [
        "nax_e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_matmul"
    ]
    assert kernel_report["has_tensorops_matmul"] is True
    assert kernel_report["uses_micro_lut_slot_maps"] is True
    assert kernel_report["uses_micro_lut_luts"] is True
    assert kernel_report["uses_abs_index_lut"] is True
    assert kernel_report["decodes_micro_lut_values_per_b_fragment"] is True
    assert kernel_report["decision"] == "reject_per_fragment_micro_lut_tensorops_kernel"
    assert report["decision"] == "reject_per_fragment_micro_lut_tensorops_schedule"
    assert report["required_next_features"] == [
        "broader_decode_reuse_scope",
        "avoid_per_fragment_micro_lut_slot_reconstruction",
        "avoid_per_fragment_abs_index_lut_reconstruction",
        "preserve_compressed_rhs_storage",
        "preserve_codeword_scale_slots",
        "change_rhs_layout_or_kernel_family",
    ]


def test_audit_e8p_sign_plane_schedule_rejects_per_fragment_tensorops_decode() -> None:
    analyzer = _load_analyzer()
    source = """
[[kernel]] void nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_matmul() {
  auto b_t = matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  for (uint bit = 0; bit < 8u; ++bit) {
    ulong mask = sign_bit_planes[sign_base + bit];
    signs |= uint((mask >> ulong(n_local)) & 1ul) << bit;
  }
  uint abs_idx = uint(abs_index_tiles[factor_index]);
  uint parity = mlx_vq_sign_parity8(signs);
  float scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
  int scale_slot_i = codeword_scale_slots[k_block * codewords + codeword_slot];
  b_t[n_frag * mlx_vq_nax_elems_per_frag + elem] = half(
      mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, dim) * scale_f);
  mpp::tensor_ops::matmul2d<8, 8, 8, half, float> matmul_op;
  matmul_op.run(a_t, b_t, c_t);
}
[[kernel]] void other_kernel() {}
"""

    report = analyzer.audit_e8p_sign_plane_schedule_source(
        source,
        benchmark_context={
            "all_lane_s_pass": False,
            "all_parity_pass": False,
            "sign_plane_tensorops_worst_ratio_to_q2": 22.42,
        },
    )

    kernel_report = report["kernel_reports"][0]
    assert report["q2_shaped_tensorops_kernel_present"] is True
    assert report["rejected_kernel_count"] == 1
    assert report["rejected_kernels"] == [
        "nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_matmul"
    ]
    assert kernel_report["has_tensorops_matmul"] is True
    assert kernel_report["has_threadgroup_staged_b"] is False
    assert kernel_report["threadgroup_barrier_count"] == 0
    assert kernel_report["uses_sign_bit_planes"] is True
    assert kernel_report["uses_abs_index_tiles"] is True
    assert kernel_report["uses_codeword_scale_slots"] is True
    assert kernel_report["reconstructs_signs_from_bit_planes"] is True
    assert kernel_report["decodes_sign_plane_values_per_b_fragment"] is True
    assert kernel_report["decision"] == "reject_per_fragment_sign_plane_tensorops_kernel"
    assert report["decision"] == "reject_per_fragment_sign_plane_tensorops_schedule"
    assert report["required_next_features"] == [
        "broader_decode_reuse_scope",
        "avoid_per_fragment_sign_plane_abs_scale_reconstruction",
        "preserve_compressed_rhs_storage",
        "preserve_codeword_scale_slots",
        "change_rhs_layout_or_kernel_family",
    ]


def test_build_structure_report_combines_source_q2_and_benchmark_evidence(tmp_path) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  for (uint k_block = 0; k_block < K; k_block += mlx_vq_nax_bk_tile) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    Ws[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(Ws + kk);
  }
}
""",
        encoding="utf-8",
    )
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "tokens": 4096,
                "kernel_names": [
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2"
                ],
                "passes_bk64_rhs_nax": True,
                "requests_bk32": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps(
            {
                "projection_summaries": {
                    "down": {
                        "best_ratio_to_q2": 1.48,
                        "best_variant": "nax_e8p_fp16_sorted_steel_gs352_raw",
                    }
                }
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
    )

    assert report["record_type"] == "glm45_air_e8p_kernel_structure_analysis"
    assert report["q2_geometry"]["down"]["bk"] == 64
    assert report["q2_geometry"]["down"]["bits"] == 2
    assert report["e8p_kernel"]["decoded_b_threadgroup_staging"] is True
    assert report["benchmark_context"]["down_best_ratio_to_q2"] == 1.48
    assert report["next_track_b_hypothesis"] == "replace_decoded_b_staging_or_change_kernel_family"


def test_build_structure_report_separates_q2_timing_controls_from_geometry(
    tmp_path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    q2_timing = tmp_path / "q2-timing.jsonl"
    q2_geometry = tmp_path / "q2-geometry.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  for (uint k_block = 0; k_block < K; k_block += mlx_vq_nax_bk_tile) {
    Ws[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
    Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(Ws + kk);
  }
}
""",
        encoding="utf-8",
    )
    q2_timing.write_text(
        json.dumps(
            {
                "projection": "down",
                "variant": "mlx_q2",
                "tokens": 4096,
                "ms_per_iter": 4.56,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    q2_geometry.write_text(
        json.dumps(
            {
                "projection": "down",
                "tokens": 4096,
                "route_count": 32768,
                "kernel_names": [
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2"
                ],
                "passes_bk64_rhs_nax": True,
                "requests_bk32": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps(
            {
                "projection_summaries": {
                    "down": {
                        "best_ratio_to_q2": 1.48,
                        "best_variant": "nax_e8p_fp16_sorted_steel_gs352_raw",
                    }
                }
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        q2_jsonl=q2_timing,
        q2_geometry_jsonl=q2_geometry,
        e8p_analysis_json=e8p_analysis,
    )

    assert report["q2_jsonl"] == str(q2_timing)
    assert report["q2_geometry_jsonl"] == str(q2_geometry)
    assert report["q2_geometry"]["down"]["bk"] == 64
    assert report["q2_geometry"]["down"]["bits"] == 2
    assert report["q2_geometry"]["down"]["group_size"] == 128


def test_build_structure_report_names_e8p_packed_rhs_requirements(tmp_path) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  for (uint k_block = 0; k_block < K; k_block += mlx_vq_nax_bk_tile) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    Ws[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(Ws + kk);
  }
}
""",
        encoding="utf-8",
    )
    q2.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "projection": "down",
                        "tokens": 4096,
                        "route_count": 32768,
                        "kernel_names": [
                            "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2"
                        ],
                        "passes_bk64_rhs_nax": True,
                        "requests_bk32": False,
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "projection": "gate_up",
                        "tokens": 4096,
                        "route_count": 32768,
                        "kernel_names": [
                            "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2"
                        ],
                        "passes_bk64_rhs_nax": True,
                        "requests_bk32": False,
                    },
                    sort_keys=True,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps(
            {
                "all_lane_s_pass": False,
                "all_parity_pass": False,
                "projection_summaries": {
                    "down": {
                        "best_ratio_to_q2": 1.79,
                        "best_variant": "nax_e8p_fp16_sorted_steel_raw",
                    },
                    "gate_up": {
                        "best_ratio_to_q2": 1.85,
                        "best_variant": "nax_e8p_fp16_sorted_steel_raw",
                    },
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
    )

    requirements = report["e8p_packed_rhs_requirements"]
    assert requirements["target_kernel_family"] == "sorted_gather_qmm_rhs_nax"
    assert requirements["storage_constraint"] == "compressed_e8p_codes_scales"
    assert requirements["must_match_geometry"]["bk"] == 64
    assert requirements["must_match_geometry"]["bm"] == 64
    assert requirements["must_match_geometry"]["bn"] == 64
    assert requirements["reject_next_paths"] == [
        "direct_reduce_scalar_accumulation",
        "inline_b_decode_per_output_tile",
        "dense_or_predecoded_fp16_rhs",
    ]
    assert "packed RHS tile layout" in requirements["next_kernel_requirement"]


def test_build_structure_report_rejects_sign_nibble_per_fragment_tensorops(tmp_path) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  for (uint k_block = 0; k_block < K; k_block += mlx_vq_nax_bk_tile) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    Ws[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(Ws + kk);
  }
}
[[kernel]] void nax_e8p_sign_nibble_abs_index_rhs_sorted_matmul() {
  uint low = sign_low_nibble_tiles[tile_offset];
  uint high = sign_high_nibble_tiles[tile_offset];
  uint signs = (low & 15u) | ((high & 15u) << 4u);
  half decoded = mlx_vq_decode_e8p_split_value(abs_index_tiles[tile_offset], signs, codebook);
  result += float(decoded * scale_tiles[scale_base + codeword_scale_slots[kw]]);
}
[[kernel]] void nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_matmul() {
  auto b_t = matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  uint low = uint(sign_low_nibble_tiles[factor_index]) & 0xFu;
  uint high = uint(sign_high_nibble_tiles[factor_index]) & 0xFu;
  uint abs_idx = uint(abs_index_tiles[factor_index]);
  uint parity = uint(parity_tiles[factor_index]) & 1u;
  float scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
  int scale_slot_i = codeword_scale_slots[k_block * codewords + codeword_slot];
  b_t[n_frag * mlx_vq_nax_elems_per_frag + elem] = half(
      mlx_vq_decode_e8p_nibble_value(
          low, high, abs_idx, parity, codebook, dim) * scale_f);
  mpp::tensor_ops::matmul2d<8, 8, 8, half, float> matmul_op;
  matmul_op.run(a_t, b_t, c_t);
}
""",
        encoding="utf-8",
    )
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "tokens": 4096,
                "route_count": 32768,
                "kernel_names": [
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2"
                ],
                "passes_bk64_rhs_nax": True,
                "requests_bk32": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps(
            {
                "all_lane_s_pass": False,
                "all_parity_pass": False,
                "comparisons": [
                    {
                        "candidate_variant": (
                            "nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_raw"
                        ),
                        "candidate_ms_per_iter": 17.82,
                        "ratio_to_q2": 14.71,
                    }
                ],
                "projection_summaries": {
                    "down": {
                        "best_ratio_to_q2": 2.4,
                        "best_variant": "nax_e8p_packed_rhs_sorted_tiled_raw",
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
    )

    guardrail = report["sign_nibble_schedule_guardrail"]
    assert guardrail["decision"] == "reject_per_fragment_nibble_tensorops_schedule"
    assert guardrail["per_fragment_rejected_kernels"] == [
        "nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_matmul"
    ]
    assert report["benchmark_context"]["sign_nibble_tensorops_worst_ratio_to_q2"] == 14.71
    assert report["next_track_b_hypothesis"] == (
        "broaden_sign_nibble_decode_reuse_or_change_kernel_family"
    )


def test_build_structure_report_rejects_sign_plane_per_fragment_tensorops(tmp_path) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
  Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(Ws + kk);
}
[[kernel]] void nax_e8p_sign_plane_abs_index_rhs_sorted_matmul() {
  ulong mask = sign_bit_planes[sign_base + bit];
  uint signs = uint((mask >> ulong(n_in_tile)) & 1ul);
  half decoded = mlx_vq_decode_e8p_split_value(
      signs, abs_index_tiles[tile_offset], codebook);
  result += float(decoded * scale_tiles[scale_base + codeword_scale_slots[kw]]);
}
[[kernel]] void nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_matmul() {
  auto b_t = matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  for (uint bit = 0; bit < 8u; ++bit) {
    ulong mask = sign_bit_planes[sign_base + bit];
    signs |= uint((mask >> ulong(n_local)) & 1ul) << bit;
  }
  uint abs_idx = uint(abs_index_tiles[factor_index]);
  uint parity = mlx_vq_sign_parity8(signs);
  float scale_f = float(scale_tiles[rhs_base * scale_groups + scale_slot]);
  int scale_slot_i = codeword_scale_slots[k_block * codewords + codeword_slot];
  b_t[n_frag * mlx_vq_nax_elems_per_frag + elem] = half(
      mlx_vq_decode_e8p_split_value(signs, abs_idx, parity, codebook, dim) * scale_f);
  mpp::tensor_ops::matmul2d<8, 8, 8, half, float> matmul_op;
  matmul_op.run(a_t, b_t, c_t);
}
""",
        encoding="utf-8",
    )
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "tokens": 4096,
                "route_count": 32768,
                "kernel_names": [
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2"
                ],
                "passes_bk64_rhs_nax": True,
                "requests_bk32": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps(
            {
                "all_lane_s_pass": False,
                "all_parity_pass": False,
                "comparisons": [
                    {
                        "candidate_variant": (
                            "nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_raw"
                        ),
                        "candidate_ms_per_iter": 21.98,
                        "ratio_to_q2": 22.42,
                    }
                ],
                "projection_summaries": {
                    "down": {
                        "best_ratio_to_q2": 2.4,
                        "best_variant": "nax_e8p_packed_rhs_sorted_tiled_raw",
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
    )

    guardrail = report["sign_plane_schedule_guardrail"]
    assert guardrail["decision"] == "reject_per_fragment_sign_plane_tensorops_schedule"
    assert guardrail["per_fragment_rejected_kernels"] == [
        "nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_matmul"
    ]
    assert report["benchmark_context"]["sign_plane_tensorops_worst_ratio_to_q2"] == 22.42
    assert report["next_track_b_hypothesis"] == (
        "replace_sign_plane_tensorops_layout_or_kernel_family"
    )


def test_build_structure_report_rejects_shared_n_lane_local_decode(tmp_path) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
  Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(Ws + kk);
}
[[kernel]] void nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_matmul() {
  threadgroup half shared_b[2 * mlx_vq_nax_shared_lanes * 2 * mlx_vq_nax_elems_per_frag];
  uint sign_slot = uint(sign_byte_slots[factor_index]);
  uint abs_slot = uint(abs_index_slots[factor_index]);
  uint signs = uint(sign_byte_lut[lut_base + sign_slot]);
  uint abs_idx = uint(abs_index_lut[lut_base + abs_slot]);
  shared_b[shared_group_base + elem] = half(
      mlx_vq_decode_e8p_split_value(signs, abs_idx, codebook, dim) * scale_tiles[rhs_base]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto b_t = matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  b_t[n_frag * mlx_vq_nax_elems_per_frag + elem] = shared_b[shared_group_base + elem];
  mpp::tensor_ops::matmul2d<8, 8, 8, half, float> matmul_op;
  matmul_op.run(a_t, b_t, c_t);
}
""",
        encoding="utf-8",
    )
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "tokens": 4096,
                "route_count": 32768,
                "kernel_names": [
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2"
                ],
                "passes_bk64_rhs_nax": True,
                "requests_bk32": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps(
            {
                "all_lane_s_pass": False,
                "all_parity_pass": False,
                "projection_summaries": {
                    "down": {
                        "best_ratio_to_q2": 2.81,
                        "best_variant": "nax_e8p_split_byte_factor_reuse_rhs_sorted_tiled_raw",
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
    )

    assert report["shared_n_decode_kernel"]["decision"] == "reject_lane_local_shared_n_decode"
    assert report["shared_n_decode_kernel"]["loads_b_fragments_from_shared_b"] is True
    assert report["next_track_b_hypothesis"] == (
        "avoid_lane_local_shared_n_decode_or_change_kernel_family"
    )


def test_build_structure_report_requires_expert_kblock_factor_decode_kernel(
    tmp_path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
  Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(Ws + kk);
}
""",
        encoding="utf-8",
    )
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "tokens": 4096,
                "route_count": 32768,
                "kernel_names": [
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2"
                ],
                "passes_bk64_rhs_nax": True,
                "requests_bk32": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps(
            {
                "all_lane_s_pass": False,
                "all_parity_pass": False,
                "expert_kblock_schedule_requirements": {
                    "decision": "probe_expert_kblock_factor_decode_reuse",
                    "factor_signal": "expert_kblock_factor_reuse_present",
                    "required_kernel_features": [
                        "reuse_factor_decode_across_output_tiles_per_expert_kblock",
                        "avoid_per_fragment_factor_reconstruction",
                        "avoid_lane_local_barrier_fragment_buffer",
                        "account_for_output_column_scale_groups",
                    ],
                },
                "projection_summaries": {
                    "down": {
                        "best_ratio_to_q2": 2.4,
                        "best_variant": "nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_raw",
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
    )

    guardrail = report["expert_kblock_schedule_guardrail"]
    assert guardrail["decision"] == "missing_expert_kblock_factor_decode_kernel"
    assert guardrail["required_next_features"] == [
        "reuse_factor_decode_across_output_tiles_per_expert_kblock",
        "preserve_compressed_rhs_storage",
        "preserve_codeword_scale_slots",
        "account_for_output_column_scale_groups",
        "avoid_per_fragment_factor_reconstruction",
        "avoid_lane_local_barrier_fragment_buffer",
    ]
    assert report["benchmark_context"]["expert_kblock_schedule_decision"] == (
        "probe_expert_kblock_factor_decode_reuse"
    )
    assert report["benchmark_context"]["expert_kblock_factor_signal"] == (
        "expert_kblock_factor_reuse_present"
    )
    assert report["next_track_b_hypothesis"] == (
        "build_expert_kblock_factor_decode_reuse_kernel"
    )


def test_build_structure_report_separates_scalar_expert_kblock_consumer_from_tensorops_path(
    tmp_path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
  Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(Ws + kk);
}
[[kernel]] void nax_e8p_expert_kblock_factor_reuse_rhs_sorted_matmul() {
  for (uint expert = 0; expert < E; ++expert) {
    uint lut_base = (expert * k_blocks + k_block) * 256u;
    uint sign_byte = sign_byte_lut[lut_base + packed_code];
    uint abs_index = abs_index_lut[lut_base + packed_code];
    half value = half(float(sign_byte + abs_index) * scale_tiles[scale_slot]);
    out[row] = value;
  }
}
""",
        encoding="utf-8",
    )
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "tokens": 4096,
                "route_count": 32768,
                "kernel_names": [
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2"
                ],
                "passes_bk64_rhs_nax": True,
                "requests_bk32": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps(
            {
                "all_lane_s_pass": False,
                "all_parity_pass": False,
                "expert_kblock_schedule_requirements": {
                    "decision": "probe_expert_kblock_factor_decode_reuse",
                    "factor_signal": "expert_kblock_factor_reuse_present",
                    "required_kernel_features": [
                        "reuse_factor_decode_across_output_tiles_per_expert_kblock",
                        "avoid_per_fragment_factor_reconstruction",
                        "avoid_lane_local_barrier_fragment_buffer",
                        "account_for_output_column_scale_groups",
                    ],
                },
                "projection_summaries": {
                    "down": {
                        "best_ratio_to_q2": 2.4,
                        "best_variant": "nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_raw",
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
    )

    guardrail = report["expert_kblock_schedule_guardrail"]
    assert guardrail["decision"] == (
        "scalar_expert_kblock_consumer_present_tensorops_missing"
    )
    assert guardrail["present_kernel_count"] == 1
    assert guardrail["present_kernels"] == [
        "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_matmul"
    ]
    assert guardrail["kernel_reports"][1]["decision"] == (
        "scalar_consumer_present_tensorops_missing"
    )
    assert "candidate does not reach TensorOps" in guardrail["kernel_reports"][1][
        "structural_risk"
    ]
    assert "build_expert_kblock_tensorops_factor_decode_reuse_kernel" in guardrail[
        "rejected_next_steps"
    ]
    assert report["next_track_b_hypothesis"] == (
        "build_expert_kblock_tensorops_factor_decode_reuse_kernel"
    )


def test_build_structure_report_marks_expert_kblock_tensorops_candidate_as_benchmark_pending(
    tmp_path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
  Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(Ws + kk);
}
[[kernel]] void nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul() {
  uint lut_base = (expert * k_blocks + k_block) * 256u;
  uint sign_byte = sign_byte_lut[lut_base + sign_slot];
  uint abs_index = abs_index_lut[lut_base + abs_slot];
  auto b_t = matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  b_t[0] = half(float(sign_byte + abs_index) * scale_tiles[scale_slot]);
  mpp::tensor_ops::matmul2d<8, 8, 8, half, float> matmul_op;
  matmul_op.run(a_t, b_t, c_t);
}
[[kernel]] void nax_e8p_expert_kblock_factor_reuse_rhs_sorted_matmul() {
  uint lut_base = (expert * k_blocks + k_block) * 256u;
  uint sign_byte = sign_byte_lut[lut_base + sign_slot];
  uint abs_index = abs_index_lut[lut_base + abs_slot];
  out[row] = half(float(sign_byte + abs_index) * scale_tiles[scale_slot]);
}
""",
        encoding="utf-8",
    )
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "tokens": 4096,
                "route_count": 32768,
                "kernel_names": [
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2"
                ],
                "passes_bk64_rhs_nax": True,
                "requests_bk32": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps(
            {
                "all_lane_s_pass": False,
                "all_parity_pass": False,
                "expert_kblock_schedule_requirements": {
                    "decision": "probe_expert_kblock_factor_decode_reuse",
                    "factor_signal": "expert_kblock_factor_reuse_present",
                    "required_kernel_features": [
                        "reuse_factor_decode_across_output_tiles_per_expert_kblock",
                        "avoid_per_fragment_factor_reconstruction",
                        "avoid_lane_local_barrier_fragment_buffer",
                        "account_for_output_column_scale_groups",
                    ],
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
    )

    guardrail = report["expert_kblock_schedule_guardrail"]
    assert guardrail["decision"] == (
        "expert_kblock_tensorops_candidate_present_needs_benchmark"
    )
    assert guardrail["present_kernel_count"] == 2
    assert guardrail["kernel_reports"][0]["decision"] == (
        "tensorops_candidate_present_needs_benchmark"
    )
    assert guardrail["kernel_reports"][0]["has_tensorops_matmul"] is True
    assert guardrail["kernel_reports"][0]["uses_factor_luts"] is True
    assert "benchmark_expert_kblock_tensorops_factor_decode_reuse_kernel" in guardrail[
        "rejected_next_steps"
    ]
    assert report["next_track_b_hypothesis"] == (
        "benchmark_expert_kblock_tensorops_factor_decode_reuse_kernel"
    )


def test_build_structure_report_rejects_benchmarked_expert_kblock_tensorops_without_v2(
    tmp_path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
  Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(Ws + kk);
}
[[kernel]] void nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul() {
  uint lut_base = (expert * k_blocks + k_block) * 256u;
  uint sign_byte = sign_byte_lut[lut_base + sign_slot];
  uint abs_index = abs_index_lut[lut_base + abs_slot];
  auto b_t = matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  b_t[0] = half(float(sign_byte + abs_index) * scale_tiles[scale_slot]);
  mpp::tensor_ops::matmul2d<8, 8, 8, half, float> matmul_op;
  matmul_op.run(a_t, b_t, c_t);
}
""",
        encoding="utf-8",
    )
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "tokens": 4096,
                "route_count": 32768,
                "kernel_names": [
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2"
                ],
                "passes_bk64_rhs_nax": True,
                "requests_bk32": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps(
            {
                "all_lane_s_pass": False,
                "all_parity_pass": False,
                "comparisons": [
                    {
                        "candidate_variant": (
                            "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_raw"
                        ),
                        "projection": "gate_up",
                        "ratio_to_q2": 4.057,
                        "candidate_ms_per_iter": 30.8863,
                    },
                    {
                        "candidate_variant": (
                            "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_raw"
                        ),
                        "projection": "down",
                        "ratio_to_q2": 6.255,
                        "candidate_ms_per_iter": 28.5235,
                    },
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
    )

    guardrail = report["expert_kblock_schedule_guardrail"]
    assert guardrail["decision"] == (
        "current_expert_kblock_tensorops_speed_rejected_v2_missing"
    )
    assert guardrail["current_tensorops_benchmark_rejected"] is True
    assert guardrail["v2_kernel_present"] is False
    assert guardrail["required_next_features"] == [
        "materially_different_rhs_layout_or_kernel_family",
        "reuse_factor_decode_across_output_tiles_per_expert_kblock",
        "preserve_compressed_rhs_storage",
        "preserve_codeword_scale_slots",
        "account_for_output_column_scale_groups",
        "avoid_per_fragment_factor_reconstruction",
        "avoid_lane_local_barrier_fragment_buffer",
    ]
    assert "do_not_retime_current_expert_kblock_tensorops" in guardrail[
        "rejected_next_steps"
    ]
    assert report["benchmark_context"]["expert_kblock_tensorops_worst_ratio_to_q2"] == 6.255
    assert report["next_track_b_hypothesis"] == (
        "design_expert_kblock_v2_or_change_kernel_family"
    )


def test_build_structure_report_rejects_near_duplicate_expert_kblock_v2(
    tmp_path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
  Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(Ws + kk);
}
[[kernel]] void nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul() {
  uint lut_base = (expert * k_blocks + k_block) * 256u;
  uint sign_byte = sign_byte_lut[lut_base + sign_slot];
  uint abs_index = abs_index_lut[lut_base + abs_slot];
  auto b_t = matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  b_t[0] = half(float(sign_byte + abs_index) * scale_tiles[scale_slot]);
  mpp::tensor_ops::matmul2d<8, 8, 8, half, float> matmul_op;
  matmul_op.run(a_t, b_t, c_t);
}
[[kernel]] void nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul() {
  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint kk = 0; kk < mlx_vq_nax_bk_tile; kk += 16u) {
      auto b_t = matmul_op.get_right_input_cooperative_tensor<half, half, float>();
      uint lut_base = (expert * k_blocks + k_block) * 256u;
      uint sign_byte = sign_byte_lut[lut_base + sign_slot];
      uint abs_index = abs_index_lut[lut_base + abs_slot];
      b_t[0] = half(float(sign_byte + abs_index) * scale_tiles[scale_slot]);
      mpp::tensor_ops::matmul2d<8, 8, 8, half, float> matmul_op;
      matmul_op.run(a_t, b_t, c_t);
    }
  }
}
""",
        encoding="utf-8",
    )
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "tokens": 4096,
                "route_count": 32768,
                "kernel_names": [
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2"
                ],
                "passes_bk64_rhs_nax": True,
                "requests_bk32": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps(
            {
                "all_lane_s_pass": False,
                "all_parity_pass": False,
                "comparisons": [
                    {
                        "candidate_variant": (
                            "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_raw"
                        ),
                        "projection": "down",
                        "ratio_to_q2": 6.255,
                        "candidate_ms_per_iter": 28.5235,
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
    )

    guardrail = report["expert_kblock_schedule_guardrail"]
    assert guardrail["current_tensorops_benchmark_rejected"] is True
    v2_report = next(
        item
        for item in guardrail["kernel_reports"]
        if item["kernel_name"]
        == "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul"
    )
    assert guardrail["decision"] == "reject_expert_kblock_tensorops_v2_near_duplicate"
    assert v2_report["decision"] == "reject_v2_per_fragment_factor_reconstruction"
    assert v2_report["decodes_factor_values_per_b_fragment"] is True
    assert "avoid_per_fragment_factor_reconstruction" in guardrail[
        "required_next_features"
    ]
    assert "do_not_benchmark_near_duplicate_expert_kblock_v2" in guardrail[
        "rejected_next_steps"
    ]
    assert report["next_track_b_hypothesis"] == (
        "redesign_expert_kblock_v2_decode_reuse_or_change_kernel_family"
    )


def test_build_structure_report_rejects_non_tensorops_expert_kblock_v2(
    tmp_path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
  Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(Ws + kk);
}
[[kernel]] void nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul() {
  uint lut_base = (expert * k_blocks + k_block) * 256u;
  uint sign_byte = sign_byte_lut[lut_base + sign_slot];
  uint abs_index = abs_index_lut[lut_base + abs_slot];
  auto b_t = matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  b_t[0] = half(float(sign_byte + abs_index) * scale_tiles[scale_slot]);
  mpp::tensor_ops::matmul2d<8, 8, 8, half, float> matmul_op;
  matmul_op.run(a_t, b_t, c_t);
}
[[kernel]] void nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul() {
  uint lut_base = (expert * k_blocks + k_block) * 256u;
  uint sign_byte = sign_byte_lut[lut_base + sign_slot];
  uint abs_index = abs_index_lut[lut_base + abs_slot];
  out[row] = half(float(sign_byte + abs_index) * scale_tiles[scale_slot]);
}
""",
        encoding="utf-8",
    )
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "tokens": 4096,
                "route_count": 32768,
                "kernel_names": [
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2"
                ],
                "passes_bk64_rhs_nax": True,
                "requests_bk32": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps(
            {
                "all_lane_s_pass": False,
                "all_parity_pass": False,
                "comparisons": [
                    {
                        "candidate_variant": (
                            "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_raw"
                        ),
                        "projection": "down",
                        "ratio_to_q2": 6.255,
                        "candidate_ms_per_iter": 28.5235,
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
    )

    guardrail = report["expert_kblock_schedule_guardrail"]
    assert guardrail["current_tensorops_benchmark_rejected"] is True
    v2_report = next(
        item
        for item in guardrail["kernel_reports"]
        if item["kernel_name"]
        == "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul"
    )
    assert guardrail["decision"] == "reject_expert_kblock_tensorops_v2_invalid_source"
    assert v2_report["decision"] == "reject_v2_tensorops_missing"
    assert v2_report["has_tensorops_matmul"] is False
    assert "build_tensorops_v2_before_benchmark" in guardrail["required_next_features"]
    assert "do_not_benchmark_non_tensorops_expert_kblock_v2" in guardrail[
        "rejected_next_steps"
    ]
    assert report["next_track_b_hypothesis"] == (
        "build_valid_expert_kblock_tensorops_v2_or_change_kernel_family"
    )


def test_build_structure_report_rejects_single_kblock_completion_v2(
    tmp_path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
  Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(Ws + kk);
}
[[kernel]] void nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul() {
  uint lut_base = (expert * k_blocks + k_block) * 256u;
  uint sign_byte = sign_byte_lut[lut_base + sign_slot];
  uint abs_index = abs_index_lut[lut_base + abs_slot];
  auto b_t = matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  b_t[0] = half(float(sign_byte + abs_index) * scale_tiles[scale_slot]);
  mpp::tensor_ops::matmul2d<8, 8, 8, half, float> matmul_op;
  matmul_op.run(a_t, b_t, c_t);
}
[[kernel]] void nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul() {
  uint expert = tgid.x;
  uint contract_k_block = tgid.y;
  uint tile_id = tgid.z;
  if (contract_k_block != 0u) {
    return;
  }
  uint total = routes_in_tile * output_dims;
  for (uint linear = lane; linear < total; linear += mlx_vq_nax_threads_per_tg) {
    uint route_slot = linear / output_dims;
    uint n = linear - route_slot * output_dims;
    out[(route_base + route_slot) * output_dims + n] = half(acc);
  }
}
""",
        encoding="utf-8",
    )
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "tokens": 4096,
                "route_count": 32768,
                "kernel_names": [
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2"
                ],
                "passes_bk64_rhs_nax": True,
                "requests_bk32": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps(
            {
                "all_lane_s_pass": False,
                "all_parity_pass": False,
                "expert_kblock_schedule_requirements": {
                    "decision": "probe_expert_kblock_factor_decode_reuse",
                    "factor_signal": "expert_kblock_factor_reuse_present",
                },
                "comparisons": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
    )

    guardrail = report["expert_kblock_schedule_guardrail"]
    assert guardrail["current_tensorops_benchmark_rejected"] is False
    v2_report = next(
        item
        for item in guardrail["kernel_reports"]
        if item["kernel_name"]
        == "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul"
    )
    assert guardrail["decision"] == (
        "reject_expert_kblock_tensorops_v2_kblock_reduction_missing"
    )
    assert v2_report["decision"] == "reject_v2_single_kblock_completion"
    assert v2_report["single_kblock_completion"] is True
    assert "add_kblock_partial_accumulation_or_two_pass_reduction" in guardrail[
        "required_next_features"
    ]
    assert "do_not_benchmark_single_kblock_completion_v2" in guardrail[
        "rejected_next_steps"
    ]
    assert report["next_track_b_hypothesis"] == (
        "design_expert_kblock_v2_partial_accumulation_or_change_kernel_family"
    )


def test_build_structure_report_accepts_v2_partial_reduction_scaffold(
    tmp_path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
  Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(Ws + kk);
}
[[kernel]] void nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul() {
  uint lut_base = (expert * k_blocks + k_block) * 256u;
  uint sign_byte = sign_byte_lut[lut_base + sign_slot];
  uint abs_index = abs_index_lut[lut_base + abs_slot];
  auto b_t = matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  b_t[0] = half(float(sign_byte + abs_index) * scale_tiles[scale_slot]);
  mpp::tensor_ops::matmul2d<8, 8, 8, half, float> matmul_op;
  matmul_op.run(a_t, b_t, c_t);
}
[[kernel]] void nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul(
    device float* kblock_partials [[buffer(12)]],
    constant const uint& k_blocks [[buffer(18)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
  uint contract_k_block = tgid.y;
  uint n_tile = n / bn;
  uint partial_offset =
      (((tile_id * k_blocks + contract_k_block) * n_tiles + n_tile) *
       mlx_vq_nax_route_tile + route_slot) * bn + n_in_tile;
  kblock_partials[partial_offset] = acc;
}
[[kernel]] void nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_reduce(
    const device float* kblock_partials [[buffer(0)]],
    device half* out [[buffer(1)]],
    constant const uint& k_blocks [[buffer(2)]]) {
  float accum = 0.0f;
  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    accum += kblock_partials[partial_base + k_block * stride];
  }
  out[route * output_dims + n] = half(accum);
}
""",
        encoding="utf-8",
    )
    primitive.write_text(
        """
void NaxE8PExpertKBlockFactorReuseRHSSortedTensorOpsV2Matmul::eval_gpu() {
  uint32_t experts = static_cast<uint32_t>(sign_byte_lut.shape(0));
  uint32_t k_blocks = static_cast<uint32_t>(sign_byte_lut.shape(1));
  uint32_t n_tiles = static_cast<uint32_t>(sign_byte_slots.shape(1));
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t route_tile_size = 64;
  array kblock_partials(
      {num_route_tiles, k_blocks, n_tiles, route_tile_size, 64},
      float32,
      nullptr,
      {});
  kblock_partials.set_data(allocator::malloc(kblock_partials.nbytes()));
  auto* partial_pso = cache.get("nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul");
  encoder.set_output_array(kblock_partials, 12);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(experts, k_blocks, num_route_tiles),
      MTL::Size::Make(128, 1, 1));
  auto* reduce_pso = cache.get("nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_reduce");
  encoder.set_input_array(kblock_partials, 0);
  encoder.set_output_array(out, 1);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(num_route_tiles, n_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}
""",
        encoding="utf-8",
    )
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "tokens": 4096,
                "route_count": 32768,
                "kernel_names": [
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2"
                ],
                "passes_bk64_rhs_nax": True,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps(
            {
                "all_lane_s_pass": False,
                "all_parity_pass": False,
                "expert_kblock_schedule_requirements": {
                    "decision": "probe_expert_kblock_factor_decode_reuse",
                    "factor_signal": "expert_kblock_factor_reuse_present",
                },
                "comparisons": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
    )

    guardrail = report["expert_kblock_schedule_guardrail"]
    partial_guard = report["expert_kblock_partial_reduction_guardrail"]
    v2_report = next(
        item
        for item in guardrail["kernel_reports"]
        if item["kernel_name"]
        == "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul"
    )
    assert partial_guard["decision"] == "expert_kblock_v2_partial_reduction_scaffold_present"
    assert partial_guard["partial_accumulation_scratch_present"] is True
    assert partial_guard["final_reduction_kernel_present"] is True
    assert partial_guard["partial_grid"] == "experts_x_k_blocks_x_route_tiles"
    assert partial_guard["reduction_grid"] == "route_tiles_x_n_tiles"
    assert partial_guard["partial_body_tensorops_present"] is False
    assert partial_guard["partial_body_parallel_schedule_present"] is False
    assert guardrail["decision"] == (
        "expert_kblock_tensorops_v2_partial_reduction_scaffold_scalar_body"
    )
    assert v2_report["single_kblock_completion"] is False
    assert v2_report["has_tensorops_matmul"] is False
    assert report["next_track_b_hypothesis"] == (
        "replace_scalar_v2_partial_body_with_valid_tensorops_schedule"
    )


def test_build_structure_report_keeps_v2_partial_tensorops_near_duplicate_rejection(
    tmp_path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
  Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(Ws + kk);
}
[[kernel]] void nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul() {
  uint lut_base = (expert * k_blocks + k_block) * 256u;
  uint sign_byte = sign_byte_lut[lut_base + sign_slot];
  uint abs_index = abs_index_lut[lut_base + abs_slot];
  auto b_t = matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  b_t[0] = half(float(sign_byte + abs_index) * scale_tiles[scale_slot]);
  mpp::tensor_ops::matmul2d<8, 8, 8, half, float> matmul_op;
  matmul_op.run(a_t, b_t, c_t);
}
[[kernel]] void nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul(
    device float* kblock_partials [[buffer(12)]],
    constant const uint& k_blocks [[buffer(18)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  uint contract_k_block = tgid.y;
  uint n_tile = tgid.x;
  uint lut_base = (expert * k_blocks + contract_k_block) * 256u;
  uint factor_index = rhs_base * codewords + codeword_slot;
  uint sign_slot = uint(sign_byte_slots[factor_index]);
  uint abs_slot = uint(abs_index_slots[factor_index]);
  uint signs = uint(sign_byte_lut[lut_base + sign_slot]);
  uint abs_idx = uint(abs_index_lut[lut_base + abs_slot]);
  auto b_t = matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  b_t[0] = half(mlx_vq_decode_e8p_split_value(
      signs, abs_idx, parity, codebook, dim) * scale_tiles[scale_slot]);
  mpp::tensor_ops::matmul2d<8, 8, 8, half, float> matmul_op;
  matmul_op.run(a_t, b_t, c_t);
  uint partial_offset =
      (((tile_id * k_blocks + contract_k_block) * n_tiles + n_tile) *
       mlx_vq_nax_route_tile + route_slot) * bn + n_in_tile;
  kblock_partials[partial_offset] = c_t[0];
}
[[kernel]] void nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_reduce(
    const device float* kblock_partials [[buffer(0)]],
    device half* out [[buffer(1)]],
    constant const uint& k_blocks [[buffer(2)]]) {
  float accum = 0.0f;
  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    accum += kblock_partials[partial_base + k_block * stride];
  }
  out[route * output_dims + n] = half(accum);
}
""",
        encoding="utf-8",
    )
    primitive.write_text(
        """
void NaxE8PExpertKBlockFactorReuseRHSSortedTensorOpsV2Matmul::eval_gpu() {
  uint32_t experts = static_cast<uint32_t>(sign_byte_lut.shape(0));
  uint32_t k_blocks = static_cast<uint32_t>(sign_byte_lut.shape(1));
  uint32_t n_tiles = static_cast<uint32_t>(sign_byte_slots.shape(1));
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  uint32_t route_tile_size = 64;
  array kblock_partials(
      {num_route_tiles, k_blocks, n_tiles, route_tile_size, 64},
      float32,
      nullptr,
      {});
  kblock_partials.set_data(allocator::malloc(kblock_partials.nbytes()));
  auto* partial_pso = cache.get("nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul");
  encoder.set_output_array(kblock_partials, 12);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(experts, k_blocks, num_route_tiles),
      MTL::Size::Make(128, 1, 1));
  auto* reduce_pso = cache.get("nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_reduce");
  encoder.set_input_array(kblock_partials, 0);
  encoder.set_output_array(out, 1);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(num_route_tiles, n_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}
""",
        encoding="utf-8",
    )
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "tokens": 4096,
                "route_count": 32768,
                "kernel_names": [
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2"
                ],
                "passes_bk64_rhs_nax": True,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps(
            {
                "all_lane_s_pass": False,
                "all_parity_pass": False,
                "expert_kblock_schedule_requirements": {
                    "decision": "probe_expert_kblock_factor_decode_reuse",
                    "factor_signal": "expert_kblock_factor_reuse_present",
                },
                "comparisons": [
                    {
                        "candidate_variant": (
                            "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_raw"
                        ),
                        "projection": "down",
                        "ratio_to_q2": 6.255,
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
    )

    guardrail = report["expert_kblock_schedule_guardrail"]
    partial_guard = report["expert_kblock_partial_reduction_guardrail"]
    v2_report = next(
        item
        for item in guardrail["kernel_reports"]
        if item["kernel_name"]
        == "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul"
    )
    assert partial_guard["decision"] == "expert_kblock_v2_partial_reduction_scaffold_present"
    assert partial_guard["partial_body_tensorops_present"] is True
    assert partial_guard["partial_body_parallel_schedule_present"] is True
    assert partial_guard["speed_claim"] is False
    assert v2_report["has_tensorops_matmul"] is True
    assert v2_report["decodes_factor_values_per_b_fragment"] is True
    assert v2_report["decision"] == "reject_v2_per_fragment_factor_reconstruction"
    assert guardrail["decision"] == "reject_expert_kblock_tensorops_v2_near_duplicate"
    assert guardrail["v2_partial_body_tensorops_present"] is True
    assert "do_not_benchmark_near_duplicate_expert_kblock_v2" in guardrail[
        "rejected_next_steps"
    ]
    assert report["next_track_b_hypothesis"] == (
        "redesign_expert_kblock_v2_decode_reuse_or_change_kernel_family"
    )


def test_build_structure_report_prioritizes_partial_reduction_after_v2_dispatch(
    tmp_path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
  Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(Ws + kk);
}
[[kernel]] void nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul() {
  uint lut_base = (expert * k_blocks + k_block) * 256u;
  uint sign_byte = sign_byte_lut[lut_base + sign_slot];
  uint abs_index = abs_index_lut[lut_base + abs_slot];
  auto b_t = matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  b_t[0] = half(float(sign_byte + abs_index) * scale_tiles[scale_slot]);
  mpp::tensor_ops::matmul2d<8, 8, 8, half, float> matmul_op;
  matmul_op.run(a_t, b_t, c_t);
}
[[kernel]] void nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul() {
  uint contract_k_block = tgid.y;
  if (contract_k_block != 0u) {
    return;
  }
  uint total = routes_in_tile * output_dims;
  out[route * output_dims + n] = half(acc);
}
""",
        encoding="utf-8",
    )
    primitive.write_text(
        """
void NaxE8PExpertKBlockFactorReuseRHSSortedTensorOpsMatmul::eval_gpu() {
  uint32_t n_tiles = static_cast<uint32_t>(sign_byte_slots.shape(1));
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  auto* pso = cache.get("nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul");
  encoder.dispatch_threadgroups(
      MTL::Size::Make(n_tiles, num_route_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}

array e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul() {
  if (sign_byte_slots.ndim() != 5 || scale_tiles.ndim() != 5) {
    throw std::invalid_argument("slot tiles [E,n_tiles,k_blocks,bn,*]");
  }
  if (bn != 64 || codewords != 8) {
    throw std::invalid_argument("requires q2-like bn64/bk64 expert/K-block factor-reuse RHS tiles");
  }
}

void NaxE8PExpertKBlockFactorReuseRHSSortedTensorOpsV2Matmul::eval_gpu() {
  uint32_t experts = static_cast<uint32_t>(sign_byte_lut.shape(0));
  uint32_t k_blocks = static_cast<uint32_t>(sign_byte_lut.shape(1));
  uint32_t n_tiles = static_cast<uint32_t>(sign_byte_slots.shape(1));
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  auto* pso = cache.get("nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul");
  encoder.set_bytes(n_tiles, 17);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(experts, k_blocks, num_route_tiles),
      MTL::Size::Make(128, 1, 1));
}

array e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul() {
  if (sign_byte_slots.ndim() != 5 || scale_tiles.ndim() != 5) {
    throw std::invalid_argument("slot tiles [E,n_tiles,k_blocks,bn,*]");
  }
  if (bn != 64 || codewords != 8) {
    throw std::invalid_argument("requires q2-like bn64/bk64 expert/K-block factor-reuse RHS tiles");
  }
}
""",
        encoding="utf-8",
    )
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "tokens": 4096,
                "route_count": 32768,
                "kernel_names": [
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2"
                ],
                "passes_bk64_rhs_nax": True,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps(
            {
                "all_lane_s_pass": False,
                "all_parity_pass": False,
                "expert_kblock_schedule_requirements": {
                    "decision": "probe_expert_kblock_factor_decode_reuse",
                    "factor_signal": "expert_kblock_factor_reuse_present",
                },
                "comparisons": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
    )

    assert report["expert_kblock_dispatch_contract_guardrail"]["passes_contract"] is True
    assert report["expert_kblock_partial_reduction_guardrail"]["passes_contract"] is False
    assert report["next_track_b_hypothesis"] == (
        "design_expert_kblock_v2_partial_accumulation_or_change_kernel_family"
    )


def test_real_current_v2_dispatch_routes_next_hypothesis_to_tensorops_schedule(
    tmp_path,
) -> None:
    analyzer = _load_analyzer()
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "tokens": 4096,
                "route_count": 32768,
                "kernel_names": [
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2"
                ],
                "passes_bk64_rhs_nax": True,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps(
            {
                "all_lane_s_pass": False,
                "all_parity_pass": False,
                "comparisons": [
                    {
                        "candidate_variant": (
                            "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_raw"
                        ),
                        "projection": "down",
                        "ratio_to_q2": 6.255,
                        "candidate_ms_per_iter": 28.5235,
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal"),
        primitive_source=Path("native/vq_nax_ext/csrc/nax_fp16_primitive.mm"),
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
    )

    assert report["expert_kblock_dispatch_contract_guardrail"]["passes_contract"] is True
    assert report["expert_kblock_partial_reduction_guardrail"]["passes_contract"] is True
    assert report["expert_kblock_partial_reduction_guardrail"][
        "partial_body_tensorops_present"
    ] is True
    assert report["expert_kblock_schedule_guardrail"]["decision"] == (
        "reject_expert_kblock_tensorops_v2_near_duplicate"
    )
    assert "do_not_benchmark_near_duplicate_expert_kblock_v2" in report[
        "expert_kblock_schedule_guardrail"
    ]["rejected_next_steps"]
    assert report["component_stream_speed_path_guardrail"]["decision"] == (
        "component_stream_tensorops_speed_path_candidate_present_needs_parity"
    )
    assert report["component_stream_partial_reduction_guardrail"]["decision"] == (
        "component_stream_partial_reduction_parallel_body_present"
    )
    assert report["next_track_b_hypothesis"] == (
        "redesign_expert_kblock_v2_decode_reuse_or_change_kernel_family"
    )


def test_real_current_component_stream_tensorops_candidate_is_speed_rejected_after_q2_packet(
    tmp_path,
) -> None:
    analyzer = _load_analyzer()
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "tokens": 1024,
                "route_count": 2048,
                "kernel_names": [
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2"
                ],
                "passes_bk64_rhs_nax": True,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps(
            {
                "all_lane_s_pass": False,
                "all_parity_pass": False,
                "comparisons": [
                    {
                        "candidate_variant": (
                            "nax_e8p_component_stream_rhs_sorted_tensorops_raw"
                        ),
                        "candidate_diagnostic_phase": (
                            "nax_e8p_component_stream_rhs_sorted_tensorops_raw"
                        ),
                        "projection": "gate_up",
                        "ratio_to_q2": 7.878,
                        "candidate_ms_per_iter": 24.155,
                    },
                    {
                        "candidate_variant": (
                            "nax_e8p_component_stream_rhs_sorted_tensorops_raw"
                        ),
                        "candidate_diagnostic_phase": (
                            "nax_e8p_component_stream_rhs_sorted_tensorops_raw"
                        ),
                        "projection": "down",
                        "ratio_to_q2": 8.448,
                        "candidate_ms_per_iter": 22.689,
                    },
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal"),
        primitive_source=Path("native/vq_nax_ext/csrc/nax_fp16_primitive.mm"),
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
    )

    speed_guard = report["component_stream_speed_path_guardrail"]
    assert speed_guard["decision"] == "reject_component_stream_tensorops_speed_path"
    assert speed_guard["component_stream_tensorops_comparison_count"] == 2
    assert speed_guard["component_stream_tensorops_best_ratio_to_q2"] == 7.878
    assert speed_guard["component_stream_tensorops_worst_ratio_to_q2"] == 8.448
    assert speed_guard["speed_claim"] is False
    assert "do_not_retime_component_stream_tensorops_unchanged" in speed_guard[
        "rejected_next_steps"
    ]
    assert report["component_stream_shared_decode_guardrail"]["decision"] == (
        "component_stream_shared_decode_scalar_cache_scaffold_present"
    )
    assert report["next_track_b_hypothesis"] == (
        "replace_scalar_component_stream_shared_decode_body_with_valid_parallel_schedule"
    )


def test_real_current_v2_source_has_partial_reduction_scaffold(
    tmp_path,
) -> None:
    analyzer = _load_analyzer()
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "tokens": 4096,
                "route_count": 32768,
                "kernel_names": [
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2"
                ],
                "passes_bk64_rhs_nax": True,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps(
            {
                "all_lane_s_pass": False,
                "all_parity_pass": False,
                "comparisons": [
                    {
                        "candidate_variant": (
                            "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_raw"
                        ),
                        "projection": "down",
                        "ratio_to_q2": 6.255,
                        "candidate_ms_per_iter": 28.5235,
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal"),
        primitive_source=Path("native/vq_nax_ext/csrc/nax_fp16_primitive.mm"),
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
    )

    partial_guard = report["expert_kblock_partial_reduction_guardrail"]
    assert partial_guard["decision"] == "expert_kblock_v2_partial_reduction_scaffold_present"
    assert partial_guard["partial_accumulation_scratch_present"] is True
    assert partial_guard["partial_write_kernel_present"] is True
    assert partial_guard["final_reduction_kernel_present"] is True
    assert partial_guard["partial_body_tensorops_present"] is True
    assert partial_guard["partial_body_parallel_schedule_present"] is True
    assert partial_guard["speed_claim"] is False
    assert partial_guard["scratch_dtype"] == "float32"
    assert report["component_stream_speed_path_guardrail"]["decision"] == (
        "component_stream_tensorops_speed_path_candidate_present_needs_parity"
    )
    assert report["component_stream_partial_reduction_guardrail"]["decision"] == (
        "component_stream_partial_reduction_parallel_body_present"
    )
    assert report["next_track_b_hypothesis"] == (
        "redesign_expert_kblock_v2_decode_reuse_or_change_kernel_family"
    )


def test_real_current_v2_partial_reduction_uses_tensorops_but_rejects_near_duplicate(
    tmp_path,
) -> None:
    analyzer = _load_analyzer()
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "tokens": 4096,
                "route_count": 32768,
                "kernel_names": [
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2"
                ],
                "passes_bk64_rhs_nax": True,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps(
            {
                "all_lane_s_pass": False,
                "all_parity_pass": False,
                "comparisons": [
                    {
                        "candidate_variant": (
                            "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_raw"
                        ),
                        "projection": "down",
                        "ratio_to_q2": 6.255,
                        "candidate_ms_per_iter": 28.5235,
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal"),
        primitive_source=Path("native/vq_nax_ext/csrc/nax_fp16_primitive.mm"),
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
    )

    schedule_guard = report["expert_kblock_schedule_guardrail"]
    partial_guard = report["expert_kblock_partial_reduction_guardrail"]
    v2_report = next(
        r
        for r in schedule_guard["kernel_reports"]
        if r["kernel_name"]
        == "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul"
    )
    assert partial_guard["passes_contract"] is True
    assert partial_guard["partial_body_tensorops_present"] is True
    assert partial_guard["partial_body_parallel_schedule_present"] is True
    assert v2_report["has_tensorops_matmul"] is True
    assert v2_report["decodes_factor_values_per_b_fragment"] is True
    assert v2_report["decision"] == "reject_v2_per_fragment_factor_reconstruction"
    assert schedule_guard["decision"] == "reject_expert_kblock_tensorops_v2_near_duplicate"
    assert schedule_guard["v2_partial_body_tensorops_present"] is True
    assert "do_not_benchmark_near_duplicate_expert_kblock_v2" in schedule_guard[
        "rejected_next_steps"
    ]
    assert report["component_stream_speed_path_guardrail"]["decision"] == (
        "component_stream_tensorops_speed_path_candidate_present_needs_parity"
    )
    assert report["component_stream_partial_reduction_guardrail"]["decision"] == (
        "component_stream_partial_reduction_parallel_body_present"
    )
    assert report["next_track_b_hypothesis"] == (
        "redesign_expert_kblock_v2_decode_reuse_or_change_kernel_family"
    )


def test_real_current_track_b_family_verdict_requires_new_kernel_family(
    tmp_path,
) -> None:
    analyzer = _load_analyzer()
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "tokens": 4096,
                "route_count": 32768,
                "kernel_names": [
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2"
                ],
                "passes_bk64_rhs_nax": True,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps(
            {
                "all_lane_s_pass": False,
                "all_parity_pass": False,
                "comparisons": [
                    {
                        "candidate_variant": (
                            "nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_raw"
                        ),
                        "projection": "down",
                        "ratio_to_q2": 14.71,
                    },
                    {
                        "candidate_variant": (
                            "nax_e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_raw"
                        ),
                        "projection": "down",
                        "ratio_to_q2": 16.41,
                    },
                    {
                        "candidate_variant": (
                            "nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_raw"
                        ),
                        "projection": "down",
                        "ratio_to_q2": 22.42,
                    },
                    {
                        "candidate_variant": (
                            "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_raw"
                        ),
                        "projection": "down",
                        "ratio_to_q2": 6.255,
                    },
                    {
                        "candidate_variant": (
                            "nax_e8p_component_stream_rhs_sorted_tensorops_raw"
                        ),
                        "projection": "down",
                        "ratio_to_q2": 7.878,
                        "candidate_ms_per_iter": 24.155,
                    },
                    {
                        "candidate_variant": (
                            "nax_e8p_component_stream_rhs_sorted_tensorops_raw"
                        ),
                        "projection": "gate_up",
                        "ratio_to_q2": 8.448,
                        "candidate_ms_per_iter": 37.160,
                    },
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal"),
        primitive_source=Path("native/vq_nax_ext/csrc/nax_fp16_primitive.mm"),
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
    )

    verdict = report["track_b_current_family_verdict"]
    assert verdict["decision"] == (
        "current_track_b_family_has_open_review_or_benchmark_candidate"
    )
    assert verdict["benchmarkable_candidate_present"] is False
    assert verdict["open_review_family_count"] == 1
    assert verdict["open_review_families"] == ["component_stream_partial_reduction"]
    assert verdict["blocked_families"] == [
        "steel",
        "packed_rhs_tiled",
        "shared_decode",
        "shared_n_decode",
        "sign_nibble",
        "sign_plane",
        "expert_kblock_v2",
        "component_stream_tensorops",
        "component_stream_shared_decode",
    ]
    assert verdict["rejected_next_steps"] == []
    assert verdict["required_next_features"] == []
    assert report["component_stream_speed_path_guardrail"]["decision"] == (
        "reject_component_stream_tensorops_speed_path"
    )
    assert report["component_stream_partial_reduction_guardrail"]["decision"] == (
        "component_stream_partial_reduction_parallel_body_present"
    )
    assert report["next_track_b_hypothesis"] == (
        "redesign_expert_kblock_v2_decode_reuse_or_change_kernel_family"
    )


def test_build_structure_report_flags_current_expert_kblock_primitive_shape_no_go(
    tmp_path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
  Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(Ws + kk);
}
[[kernel]] void nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul() {
  uint lut_base = (expert * k_blocks + k_block) * 256u;
  uint sign_byte = sign_byte_lut[lut_base + sign_slot];
  uint abs_index = abs_index_lut[lut_base + abs_slot];
  auto b_t = matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  b_t[0] = half(float(sign_byte + abs_index) * scale_tiles[scale_slot]);
  mpp::tensor_ops::matmul2d<8, 8, 8, half, float> matmul_op;
  matmul_op.run(a_t, b_t, c_t);
}
""",
        encoding="utf-8",
    )
    primitive.write_text(
        """
void NaxE8PExpertKBlockFactorReuseRHSSortedTensorOpsMatmul::eval_gpu() {
  uint32_t n_tiles = static_cast<uint32_t>(sign_byte_slots.shape(1));
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  auto* pso = cache.get("nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul");
  encoder.dispatch_threadgroups(
      MTL::Size::Make(n_tiles, num_route_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}

array e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul() {
  if (sign_byte_slots.ndim() != 5 || scale_tiles.ndim() != 5) {
    throw std::invalid_argument("slot tiles [E,n_tiles,k_blocks,bn,*]");
  }
  if (bn != 64 || codewords != 8) {
    throw std::invalid_argument("requires q2-like bn64/bk64 expert/K-block factor-reuse RHS tiles");
  }
}
""",
        encoding="utf-8",
    )
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "tokens": 4096,
                "route_count": 32768,
                "kernel_names": [
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2"
                ],
                "passes_bk64_rhs_nax": True,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps(
            {
                "all_lane_s_pass": False,
                "all_parity_pass": False,
                "comparisons": [
                    {
                        "candidate_variant": (
                            "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_raw"
                        ),
                        "projection": "down",
                        "ratio_to_q2": 6.255,
                        "candidate_ms_per_iter": 28.5235,
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
    )

    primitive_guard = report["expert_kblock_primitive_shape_guardrail"]
    dispatch_contract_guard = report["expert_kblock_dispatch_contract_guardrail"]
    assert primitive_guard["decision"] == "current_primitive_shape_blocks_v2_reuse_scope"
    assert primitive_guard["dispatch_grid"] == "n_tiles_x_num_route_tiles"
    assert primitive_guard["fixed_threadgroup_threads"] == 128
    assert primitive_guard["enforces_bn64_codewords8"] is True
    assert primitive_guard["slot_layout_rank"] == 5
    assert dispatch_contract_guard["decision"] == "missing_expert_kblock_v2_dispatch_contract"
    assert dispatch_contract_guard["dispatch_grid"] == "n_tiles_x_num_route_tiles"
    assert dispatch_contract_guard["incompatible_primitive_grid_present"] is True
    assert "redesign_expert_kblock_primitive_dispatch_contract" in primitive_guard[
        "required_next_features"
    ]
    assert report["next_track_b_hypothesis"] == (
        "redesign_expert_kblock_primitive_dispatch_or_change_kernel_family"
    )


def test_audit_expert_kblock_dispatch_contract_accepts_reuse_scope_v2_source() -> None:
    analyzer = _load_analyzer()
    source = """
void NaxE8PExpertKBlockFactorReuseRHSSortedTensorOpsV2Matmul::eval_gpu() {
  uint32_t experts = static_cast<uint32_t>(sign_byte_lut.shape(0));
  uint32_t k_blocks = static_cast<uint32_t>(sign_byte_lut.shape(1));
  uint32_t n_tiles = static_cast<uint32_t>(sign_byte_slots.shape(1));
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  auto* pso = cache.get("nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul");
  encoder.set_bytes(n_tiles, 17);
  encoder.dispatch_threadgroups(
      MTL::Size::Make(experts, k_blocks, num_route_tiles),
      MTL::Size::Make(128, 1, 1));
}

array e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul() {
  if (sign_byte_slots.ndim() != 5 || scale_tiles.ndim() != 5) {
    throw std::invalid_argument("slot tiles [E,n_tiles,k_blocks,bn,*]");
  }
  if (bn != 64 || codewords != 8) {
    throw std::invalid_argument("requires q2-like bn64/bk64 expert/K-block factor-reuse RHS tiles");
  }
}
"""

    guard = analyzer.audit_e8p_expert_kblock_dispatch_contract_source(source)

    assert guard["decision"] == "expert_kblock_v2_dispatch_contract_present"
    assert guard["dispatch_grid"] == "experts_x_k_blocks_x_route_tiles"
    assert guard["incompatible_primitive_grid_present"] is False
    assert guard["passes_contract"] is True
    assert guard["required_next_features"] == []


def test_audit_expert_kblock_dispatch_contract_rejects_current_grid() -> None:
    analyzer = _load_analyzer()
    source = """
void NaxE8PExpertKBlockFactorReuseRHSSortedTensorOpsMatmul::eval_gpu() {
  uint32_t n_tiles = static_cast<uint32_t>(sign_byte_slots.shape(1));
  uint32_t num_route_tiles = static_cast<uint32_t>(tile_experts.shape(0));
  auto* pso = cache.get("nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul");
  encoder.dispatch_threadgroups(
      MTL::Size::Make(n_tiles, num_route_tiles, 1),
      MTL::Size::Make(128, 1, 1));
}
"""

    guard = analyzer.audit_e8p_expert_kblock_dispatch_contract_source(source)

    assert guard["decision"] == "missing_expert_kblock_v2_dispatch_contract"
    assert guard["dispatch_grid"] == "n_tiles_x_num_route_tiles"
    assert guard["incompatible_primitive_grid_present"] is True
    assert guard["passes_contract"] is False
    assert guard["required_next_features"] == [
        "add_expert_kblock_v2_primitive_dispatch",
        "dispatch_experts_x_k_blocks_x_route_tiles",
        "carry_output_tiles_inside_reuse_scope",
        "preserve_compressed_rhs_storage",
        "preserve_codeword_scale_slots",
    ]


def test_audit_component_stream_source_requires_native_scalar_oracle() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_component_stream_source(
        metal_source="[[kernel]] void unrelated_kernel() {}",
        primitive_source="array unrelated() { return array(); }",
    )

    assert guard["decision"] == "missing_component_stream_native_source_oracle"
    assert guard["passes_contract"] is False
    assert guard["component_stream_kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["required_next_features"] == [
        "add_component_stream_native_scalar_accumulator",
        "bind_component_stream_sorted_route_primitive",
        "preserve_component_scale_slots",
        "prove_air_down_group_size_352_parity",
    ]
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_component_stream_without_native_parity",
        "do_not_materialize_decoded_dense_rhs",
    ]


def test_audit_component_stream_source_accepts_scalar_accumulator_without_b_fragment() -> None:
    analyzer = _load_analyzer()
    metal_source = """
[[kernel]] void nax_e8p_component_stream_rhs_sorted_scalar_matmul() {
  uint sign_bit = uint(sign_component_bits[component_base + component_offset]);
  uint abs_index = uint(abs_index_tiles[codeword_base + codeword_index]);
  int scale_slot = component_scale_slots[k_block * component_count + component_offset];
  int codeword_slot = codeword_scale_slots[k_block * codewords + codeword_index];
  float scale_f = float(scale_tiles[scale_base + scale_slot + codeword_slot]);
  float component = mlx_vq_decode_e8p_component(sign_bit, abs_index, codebook, component_offset);
  accum += float(sorted_x[route * input_dims + k]) * component * scale_f;
  out[route * output_dims + n] = half(accum);
}
[[kernel]] void other_kernel() {}
"""
    primitive_source = """
array e8p_component_stream_rhs_sorted_scalar_matmul() {
  if (sign_component_bits.ndim() != 6 || component_scale_slots.ndim() != 2) {
    throw std::invalid_argument("component-stream RHS tiles required");
  }
  auto* pso = cache.get("nax_e8p_component_stream_rhs_sorted_scalar_matmul");
  encoder.dispatch_threadgroups(
      MTL::Size::Make((output_dims + 15u) / 16u, (route_count + 15u) / 16u, 1),
      MTL::Size::Make(16, 16, 1));
}
"""

    guard = analyzer.audit_e8p_component_stream_source(
        metal_source=metal_source,
        primitive_source=primitive_source,
    )

    assert guard["decision"] == "component_stream_native_scalar_oracle_present"
    assert guard["passes_contract"] is True
    assert guard["component_stream_kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["uses_component_scale_slots"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_abs_index_tiles"] is True
    assert guard["uses_sign_component_bits"] is True
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["fills_tensorops_b_fragment"] is False
    assert guard["dispatch_grid"] == "route_output_scalar"
    assert guard["required_next_features"] == [
        "prove_native_component_stream_parity",
        "prove_air_down_group_size_352_parity",
        "add_artifact_projection_parity_gate",
    ]


def test_audit_route_slot_codeword_stream_source_requires_native_guardrail() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_route_slot_codeword_stream_source(
        metal_source="[[kernel]] void unrelated_kernel() {}",
        primitive_source="array unrelated() { return array(); }",
    )

    assert guard["decision"] == "missing_route_slot_codeword_stream_source"
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["required_next_features"] == [
        "add_route_slot_codeword_stream_native_scalar_source",
        "bind_route_slot_codeword_stream_primitive",
        "preserve_route_slot_axis",
        "stream_compressed_uint16_codewords_and_scale_slots",
        "prove_air_down_group_size_352_artifact_parity",
    ]
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_route_slot_codeword_stream_before_source_guardrail",
        "do_not_reopen_expert_kblock_v2_or_route_abs_component_cache_families",
        "do_not_materialize_decoded_dense_rhs",
    ]


def test_audit_route_slot_codeword_stream_source_accepts_compressed_route_slot_stream() -> None:
    analyzer = _load_analyzer()
    metal_source = """
[[kernel]] void nax_e8p_route_slot_codeword_stream_rhs_sorted_matmul() {
  uint route_slot = route_frag_base + thread_position_in_grid.y;
  uint routes_in_tile = route_tile_counts[route_tile];
  uint route = route_tile_offsets[route_tile] + route_slot;
  uint16_t code = code_tiles[code_base + codeword];
  int scale_slot = codeword_scale_slots[k_block * codewords + codeword];
  float scale_f = float(scale_tiles[scale_base + scale_slot]);
  float value = mlx_vq_decode_e8p_value(code, codebook, component) * scale_f;
  accum += float(sorted_x[route * input_dims + k]) * value;
  if (route_slot < routes_in_tile) {
    out[route * output_dims + n] = half(accum);
  }
}
[[kernel]] void other_kernel() {}
"""
    primitive_source = """
array e8p_route_slot_codeword_stream_rhs_sorted_matmul() {
  auto* pso = cache.get("nax_e8p_route_slot_codeword_stream_rhs_sorted_matmul");
  const char* contract = "route_tiles_x_route_slots_x_k_blocks_x_codewords";
  encoder.dispatch_threadgroups(
      MTL::Size::Make(num_route_tiles, route_tile_size, k_blocks),
      MTL::Size::Make(1, 64, 1));
}
"""

    guard = analyzer.audit_e8p_route_slot_codeword_stream_source(
        metal_source=metal_source,
        primitive_source=primitive_source,
    )

    assert guard["decision"] == "route_slot_codeword_stream_source_guardrail_present"
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == "route_tiles_x_route_slots_x_k_blocks_x_codewords"
    assert guard["preserves_route_slot_axis"] is True
    assert guard["streams_compressed_codewords"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["uses_codebook"] is True
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["uses_cross_ntile_decode_cache"] is False
    assert guard["fills_tensorops_b_fragment"] is False
    assert guard["required_next_features"] == [
        "prove_native_route_slot_codeword_stream_parity",
        "prove_air_down_group_size_352_artifact_parity",
        "run_same_window_q2_speed_packet_after_parity",
    ]


def test_real_current_route_slot_codeword_stream_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()
    repo_root = Path(__file__).resolve().parents[1]
    metal_source = (
        repo_root / "native" / "vq_nax_ext" / "kernels" / "nax_fp16_matmul.metal"
    ).read_text(encoding="utf-8")
    primitive_source = (
        repo_root / "native" / "vq_nax_ext" / "csrc" / "nax_fp16_primitive.mm"
    ).read_text(encoding="utf-8")

    guard = analyzer.audit_e8p_route_slot_codeword_stream_source(
        metal_source=metal_source,
        primitive_source=primitive_source,
    )

    assert guard["decision"] == "route_slot_codeword_stream_source_guardrail_present"
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_audit_route_slot_mma_codeword_tile_source_requires_native_guardrail() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_route_slot_mma_codeword_tile_source(
        metal_source="[[kernel]] void unrelated_kernel() {}",
        primitive_source="array unrelated() { return array(); }",
    )

    assert guard["decision"] == "missing_route_slot_mma_codeword_tile_source"
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert guard["required_next_features"] == [
        "add_route_slot_mma_codeword_tile_native_source_guardrail",
        "bind_route_slot_mma_codeword_tile_primitive",
        "preserve_route_slot_axis",
        "batch_output_tiles_in_dispatch",
        "stream_compressed_uint16_codewords_and_scale_slots",
    ]
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_route_slot_mma_codeword_tile_before_source_guardrail",
        "do_not_retime_scalar_route_slot_codeword_stream",
        "do_not_materialize_decoded_dense_rhs",
    ]


def test_real_current_route_slot_mma_codeword_tile_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()
    repo_root = Path(__file__).resolve().parents[1]
    metal_source = (
        repo_root / "native" / "vq_nax_ext" / "kernels" / "nax_fp16_matmul.metal"
    ).read_text(encoding="utf-8")
    primitive_source = (
        repo_root / "native" / "vq_nax_ext" / "csrc" / "nax_fp16_primitive.mm"
    ).read_text(encoding="utf-8")

    guard = analyzer.audit_e8p_route_slot_mma_codeword_tile_source(
        metal_source=metal_source,
        primitive_source=primitive_source,
    )

    assert guard["decision"] == "route_slot_mma_codeword_tile_source_guardrail_present"
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == "route_tiles_x_output_tiles_x_k_blocks_x_codeword_tiles"
    assert guard["preserves_route_slot_axis"] is True
    assert guard["streams_compressed_codewords"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_audit_active_route_tile_codeword_outer_product_source_requires_native_guardrail() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_active_route_tile_codeword_outer_product_source(
        metal_source="[[kernel]] void unrelated_kernel() {}",
        primitive_source="array unrelated() { return array(); }",
    )

    assert guard["decision"] == (
        "missing_active_route_tile_codeword_outer_product_source"
    )
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert guard["required_next_features"] == [
        "add_active_route_tile_codeword_outer_product_native_source_guardrail",
        "bind_active_route_tile_codeword_outer_product_primitive",
        "preserve_active_route_tile_indirection",
        "stream_compressed_uint16_codewords_and_scale_slots",
        "avoid_route_slot_mma_codeword_tile_schedule",
    ]
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_active_route_tile_codeword_outer_product_before_source_guardrail",
        "do_not_retime_route_slot_mma_codeword_tile",
        "do_not_materialize_decoded_dense_rhs",
    ]


def test_audit_active_route_tile_codeword_outer_product_source_accepts_contract() -> None:
    analyzer = _load_analyzer()
    metal_source = """
[[kernel]] void nax_e8p_active_route_tile_codeword_outer_product_rhs_sorted_matmul() {
  uint active_route_tile = uint(active_route_tiles[0]);
  uint route_tile = active_route_tile / routes_in_tile;
  uint code = uint(code_tiles[active_route_tile]);
  int scale_slot = codeword_scale_slots[0];
  float scale = float(scale_tiles[scale_slot]);
  float basis = float(codebook[code]);
  float accum = basis * scale;
  out[route_tile] = half(accum);
}
"""
    primitive_source = """
array e8p_active_route_tile_codeword_outer_product_rhs_sorted_matmul() {
  auto* pso = cache.get("nax_e8p_active_route_tile_codeword_outer_product_rhs_sorted_matmul");
  auto dispatch_grid = "active_route_tiles_x_k_blocks_x_codewords_x_output_microtiles";
  encoder.dispatch_threadgroups(MTL::Size::Make(active_route_tile_count, k_blocks, output_microtiles),
                                MTL::Size::Make(1, 1, 1));
}
"""

    guard = analyzer.audit_e8p_active_route_tile_codeword_outer_product_source(
        metal_source=metal_source,
        primitive_source=primitive_source,
    )

    assert guard["decision"] == (
        "active_route_tile_codeword_outer_product_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "active_route_tiles_x_k_blocks_x_codewords_x_output_microtiles"
    )
    assert guard["preserves_active_route_tile_indirection"] is True
    assert guard["streams_compressed_codewords"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["uses_route_slot_mma_schedule"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_real_current_active_route_tile_codeword_outer_product_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_active_route_tile_codeword_outer_product_source(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal").read_text(
            encoding="utf-8"
        ),
        primitive_source=Path("native/vq_nax_ext/csrc/nax_fp16_primitive.mm").read_text(
            encoding="utf-8"
        ),
    )

    assert guard["decision"] == (
        "active_route_tile_codeword_outer_product_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["preserves_active_route_tile_indirection"] is True
    assert guard["streams_compressed_codewords"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["uses_route_slot_mma_schedule"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_audit_expert_cohort_codeword_broadcast_source_requires_native_guardrail() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_expert_cohort_codeword_broadcast_source(
        metal_source="[[kernel]] void unrelated_kernel() {}",
        primitive_source="array unrelated() { return array(); }",
    )

    assert guard["decision"] == "missing_expert_cohort_codeword_broadcast_source"
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert guard["required_next_features"] == [
        "add_expert_cohort_codeword_broadcast_native_source_guardrail",
        "bind_expert_cohort_codeword_broadcast_primitive",
        "preserve_expert_cohort_route_descriptors",
        "preserve_compressed_uint16_codeword_broadcast",
        "avoid_decoded_dense_rhs",
    ]
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_expert_cohort_codeword_broadcast_before_source_guardrail",
        "do_not_retime_active_route_tile_codeword_outer_product",
        "do_not_materialize_decoded_dense_rhs",
    ]


def test_audit_expert_cohort_codeword_broadcast_source_accepts_contract() -> None:
    analyzer = _load_analyzer()
    metal_source = """
[[kernel]] void nax_e8p_expert_cohort_codeword_broadcast_rhs_sorted_matmul() {
  uint cohort = uint(expert_cohort_offsets[0]);
  uint cohort_count = uint(expert_cohort_counts[cohort]);
  uint route_offset = uint(route_cohort_offsets[cohort]);
  uint route = uint(route_ids[route_offset + cohort_count - 1]);
  uint code = uint(code_tiles[route]);
  uint16_t compressed_codeword = uint16_t(code);
  int scale_slot = codeword_scale_slots[code];
  float scale = float(scale_tiles[scale_slot]);
  out[route] = half(float(compressed_codeword) * scale);
}
"""
    primitive_source = """
array e8p_expert_cohort_codeword_broadcast_rhs_sorted_matmul() {
  auto* pso = cache.get("nax_e8p_expert_cohort_codeword_broadcast_rhs_sorted_matmul");
  auto dispatch_grid = "experts_x_route_cohorts_x_output_microtiles_x_k_blocks";
  auto wrapper = "nax_e8p_expert_cohort_codeword_broadcast_rhs_sorted_matmul_into";
  encoder.dispatch_threadgroups(MTL::Size::Make(expert_cohort_count, k_blocks, output_microtiles),
                                MTL::Size::Make(1, 1, 1));
}
array nax_e8p_expert_cohort_codeword_broadcast_rhs_sorted_matmul() {
  return e8p_expert_cohort_codeword_broadcast_rhs_sorted_matmul();
}
"""

    guard = analyzer.audit_e8p_expert_cohort_codeword_broadcast_source(
        metal_source=metal_source,
        primitive_source=primitive_source,
    )

    assert guard["decision"] == (
        "expert_cohort_codeword_broadcast_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "experts_x_route_cohorts_x_output_microtiles_x_k_blocks"
    )
    assert guard["preserves_expert_cohort_route_descriptors"] is True
    assert guard["streams_compressed_codewords"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["uses_active_route_tile_schedule"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_real_current_expert_cohort_codeword_broadcast_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_expert_cohort_codeword_broadcast_source(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal").read_text(
            encoding="utf-8"
        ),
        primitive_source=Path("native/vq_nax_ext/csrc/nax_fp16_primitive.mm").read_text(
            encoding="utf-8"
        ),
    )

    assert guard["decision"] == (
        "expert_cohort_codeword_broadcast_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "experts_x_route_cohorts_x_output_microtiles_x_k_blocks"
    )
    assert guard["preserves_expert_cohort_route_descriptors"] is True
    assert guard["streams_compressed_codewords"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["uses_active_route_tile_schedule"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_audit_route_batch_segmented_codeword_reduce_source_requires_native_guardrail() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_route_batch_segmented_codeword_reduce_source(
        metal_source="[[kernel]] void unrelated_kernel() {}",
        primitive_source="array unrelated() { return array(); }",
    )

    assert guard["decision"] == "missing_route_batch_segmented_codeword_reduce_source"
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert guard["required_next_features"] == [
        "add_route_batch_segmented_codeword_reduce_native_source_guardrail",
        "bind_route_batch_segmented_codeword_reduce_primitive",
        "preserve_route_batch_segment_descriptors",
        "stream_compressed_uint16_codewords_and_scale_slots",
        "avoid_expert_major_codeword_broadcast_schedule",
    ]
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_route_batch_segmented_codeword_reduce_before_source_guardrail",
        "do_not_retime_expert_cohort_codeword_broadcast",
        "do_not_materialize_decoded_dense_rhs",
    ]


def test_audit_route_batch_segmented_codeword_reduce_source_accepts_contract() -> None:
    analyzer = _load_analyzer()
    metal_source = """
[[kernel]] void nax_e8p_route_batch_segmented_codeword_reduce_rhs_sorted_matmul() {
  uint segment = uint(route_batch_segment_offsets[0]);
  uint segment_count = uint(route_batch_segment_counts[segment]);
  uint route = uint(route_batch_route_ids[segment + segment_count - 1]);
  uint code = uint(code_tiles[route]);
  uint16_t compressed_codeword = uint16_t(code);
  int scale_slot = codeword_scale_slots[code];
  float scale = float(scale_tiles[scale_slot]);
  out[route] = half(float(compressed_codeword) * scale);
}
"""
    primitive_source = """
array e8p_route_batch_segmented_codeword_reduce_rhs_sorted_matmul() {
  auto* pso = cache.get("nax_e8p_route_batch_segmented_codeword_reduce_rhs_sorted_matmul");
  auto dispatch_grid = "route_batches_x_k_blocks_x_output_microtiles_x_codewords";
  auto wrapper = "nax_e8p_route_batch_segmented_codeword_reduce_rhs_sorted_matmul_into";
  encoder.dispatch_threadgroups(MTL::Size::Make(route_batch_count, k_blocks, output_microtiles),
                                MTL::Size::Make(1, 1, 1));
}
array nax_e8p_route_batch_segmented_codeword_reduce_rhs_sorted_matmul() {
  return e8p_route_batch_segmented_codeword_reduce_rhs_sorted_matmul();
}
"""

    guard = analyzer.audit_e8p_route_batch_segmented_codeword_reduce_source(
        metal_source=metal_source,
        primitive_source=primitive_source,
    )

    assert guard["decision"] == (
        "route_batch_segmented_codeword_reduce_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "route_batches_x_k_blocks_x_output_microtiles_x_codewords"
    )
    assert guard["preserves_route_batch_segment_descriptors"] is True
    assert guard["streams_compressed_codewords"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["uses_expert_cohort_broadcast_schedule"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_real_current_route_batch_segmented_codeword_reduce_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_route_batch_segmented_codeword_reduce_source(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal").read_text(
            encoding="utf-8"
        ),
        primitive_source=Path("native/vq_nax_ext/csrc/nax_fp16_primitive.mm").read_text(
            encoding="utf-8"
        ),
    )

    assert guard["decision"] == (
        "route_batch_segmented_codeword_reduce_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["preserves_route_batch_segment_descriptors"] is True
    assert guard["streams_compressed_codewords"] is True
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_audit_token_cohort_codeword_stream_source_requires_native_guardrail() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_token_cohort_codeword_stream_source(
        metal_source="[[kernel]] void unrelated_kernel() {}",
        primitive_source="array unrelated() { return array(); }",
    )

    assert guard["decision"] == "missing_token_cohort_codeword_stream_source"
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert guard["required_next_features"] == [
        "add_token_cohort_codeword_stream_native_source_guardrail",
        "bind_token_cohort_codeword_stream_primitive",
        "preserve_token_cohort_route_descriptors",
        "preserve_exact_route_slot_indirection",
        "stream_compressed_uint16_codewords_and_scale_slots",
        "avoid_decoded_dense_rhs",
    ]
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_token_cohort_codeword_stream_before_source_guardrail",
        "do_not_retime_component_stream_tensorops",
        "do_not_materialize_decoded_dense_rhs",
    ]


def test_audit_token_cohort_codeword_stream_source_accepts_contract() -> None:
    analyzer = _load_analyzer()
    metal_source = """
[[kernel]] void nax_e8p_token_cohort_codeword_stream_rhs_sorted_matmul() {
  uint cohort = uint(token_cohort_offsets[0]);
  uint token_count = uint(token_cohort_counts[cohort]);
  uint route_slot = uint(token_cohort_route_slot_ids[cohort + token_count - 1]);
  uint active_expert = uint(token_cohort_active_expert_ids[route_slot]);
  uint code = uint(code_tiles[active_expert]);
  uint16_t compressed_codeword = uint16_t(code);
  int scale_slot = codeword_scale_slots[code];
  float scale = float(scale_tiles[scale_slot]);
  out[route_slot] = half(float(compressed_codeword) * scale);
}
"""
    primitive_source = """
array e8p_token_cohort_codeword_stream_rhs_sorted_matmul() {
  auto* pso = cache.get("nax_e8p_token_cohort_codeword_stream_rhs_sorted_matmul");
  auto dispatch_grid = "token_cohorts_x_active_experts_x_k_blocks_x_codewords";
  auto wrapper = "nax_e8p_token_cohort_codeword_stream_rhs_sorted_matmul_into";
  encoder.dispatch_threadgroups(MTL::Size::Make(token_cohort_count, active_expert_count, k_blocks),
                                MTL::Size::Make(1, 1, 1));
}
array nax_e8p_token_cohort_codeword_stream_rhs_sorted_matmul() {
  return e8p_token_cohort_codeword_stream_rhs_sorted_matmul();
}
"""

    guard = analyzer.audit_e8p_token_cohort_codeword_stream_source(
        metal_source=metal_source,
        primitive_source=primitive_source,
    )

    assert guard["decision"] == "token_cohort_codeword_stream_source_guardrail_present"
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "token_cohorts_x_active_experts_x_k_blocks_x_codewords"
    )
    assert guard["preserves_token_cohort_route_descriptors"] is True
    assert guard["preserves_exact_route_slot_indirection"] is True
    assert guard["streams_compressed_codewords"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["uses_component_stream_partial_schedule"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_real_current_token_cohort_codeword_stream_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_token_cohort_codeword_stream_source(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal").read_text(
            encoding="utf-8"
        ),
        primitive_source=Path("native/vq_nax_ext/csrc/nax_fp16_primitive.mm").read_text(
            encoding="utf-8"
        ),
    )

    assert guard["decision"] == "token_cohort_codeword_stream_source_guardrail_present"
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["preserves_token_cohort_route_descriptors"] is True
    assert guard["preserves_exact_route_slot_indirection"] is True
    assert guard["streams_compressed_codewords"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["uses_component_stream_partial_schedule"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_build_structure_report_surfaces_missing_route_slot_mma_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[0] = half(0.0);
}
[[kernel]] void unrelated_kernel() {}
""",
        encoding="utf-8",
    )
    primitive.write_text("array unrelated() { return array(); }\n", encoding="utf-8")
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "kernel": (
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_"
                    "bn_64_bk_64_wm_2_wn_2_align_M_t"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps({"projection_summaries": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
    )

    guard = report["route_slot_mma_codeword_tile_guardrail"]
    assert guard["decision"] == "missing_route_slot_mma_codeword_tile_source"
    assert guard["passes_contract"] is False
    assert report["next_track_b_hypothesis"] == (
        "replace_decoded_b_staging_or_change_kernel_family"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_build_structure_report_surfaces_missing_active_route_tile_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    active_design = tmp_path / "active-route-tile-design.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[0] = half(0.0);
}
[[kernel]] void unrelated_kernel() {}
""",
        encoding="utf-8",
    )
    primitive.write_text("array unrelated() { return array(); }\n", encoding="utf-8")
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "kernel": (
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_"
                    "bn_64_bk_64_wm_2_wn_2_align_M_t"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps({"projection_summaries": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    active_design.write_text(
        json.dumps(
            {
                "decision": (
                    "active_route_tile_codeword_outer_product_ready_for_source_structure_probe"
                ),
                "selector_verdict": {"candidate_ready_for_source_probe": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
        active_route_tile_design_json=active_design,
    )

    guard = report["active_route_tile_codeword_outer_product_guardrail"]
    assert guard["decision"] == (
        "missing_active_route_tile_codeword_outer_product_source"
    )
    assert guard["passes_contract"] is False
    assert report["next_track_b_hypothesis"] == (
        "implement_active_route_tile_codeword_outer_product_source_guardrail"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_build_structure_report_surfaces_missing_expert_cohort_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    expert_design = tmp_path / "expert-cohort-design.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[0] = half(0.0);
}
[[kernel]] void unrelated_kernel() {}
""",
        encoding="utf-8",
    )
    primitive.write_text("array unrelated() { return array(); }\n", encoding="utf-8")
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "kernel": (
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_"
                    "bn_64_bk_64_wm_2_wn_2_align_M_t"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps({"projection_summaries": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    expert_design.write_text(
        json.dumps(
            {
                "decision": (
                    "expert_cohort_codeword_broadcast_ready_for_source_structure_probe"
                ),
                "selector_verdict": {"candidate_ready_for_source_probe": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
        expert_cohort_design_json=expert_design,
    )

    guard = report["expert_cohort_codeword_broadcast_guardrail"]
    assert guard["decision"] == "missing_expert_cohort_codeword_broadcast_source"
    assert guard["passes_contract"] is False
    assert report["next_track_b_hypothesis"] == (
        "implement_expert_cohort_codeword_broadcast_source_guardrail"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_audit_route_codeword_lut_accumulate_source_requires_native_guardrail() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_route_codeword_lut_accumulate_source(
        metal_source="[[kernel]] void unrelated_kernel() {}",
        primitive_source="array unrelated() { return array(); }",
    )

    assert guard["decision"] == "missing_route_codeword_lut_accumulate_source"
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert guard["required_next_features"] == [
        "add_route_codeword_lut_accumulate_native_source_guardrail",
        "bind_route_codeword_lut_accumulate_primitive",
        "preserve_route_codeword_lut_descriptors",
        "stream_compressed_uint16_codewords_and_scale_slots",
        "accumulate_route_local_codeword_dot_lut",
        "avoid_decoded_dense_rhs",
    ]
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_route_codeword_lut_accumulate_before_source_guardrail",
        "do_not_retime_expert_kblock_codeword_factor_reuse",
        "do_not_materialize_decoded_dense_rhs",
    ]


def test_audit_route_codeword_lut_accumulate_source_accepts_contract() -> None:
    analyzer = _load_analyzer()
    metal_source = """
[[kernel]] void nax_e8p_route_codeword_lut_accumulate_rhs_sorted_matmul() {
  uint route_slot = uint(route_codeword_lut_route_slots[0]);
  uint lut_offset = uint(route_codeword_lut_offsets[route_slot]);
  uint lut_count = uint(route_codeword_lut_counts[route_slot]);
  uint codeword = uint(route_codeword_lut_codeword_ids[lut_offset + lut_count - 1]);
  uint code = uint(code_tiles[codeword]);
  uint16_t compressed_codeword = uint16_t(code);
  int scale_slot = codeword_scale_slots[codeword];
  float scale = float(scale_tiles[scale_slot]);
  float lut_dot = route_local_codeword_dot_lut[codeword] * scale;
  route_codeword_lut_accumulators[route_slot] += lut_dot * float(compressed_codeword);
}
"""
    primitive_source = """
array e8p_route_codeword_lut_accumulate_rhs_sorted_matmul() {
  auto* pso = cache.get("nax_e8p_route_codeword_lut_accumulate_rhs_sorted_matmul");
  auto dispatch_grid = "route_slots_x_k_blocks_x_codewords_then_output_tiles";
  auto wrapper = "nax_e8p_route_codeword_lut_accumulate_rhs_sorted_matmul_into";
  encoder.dispatch_threadgroups(MTL::Size::Make(route_slot_count, k_block_count, codeword_count),
                                MTL::Size::Make(1, 1, 1));
}
array nax_e8p_route_codeword_lut_accumulate_rhs_sorted_matmul() {
  return e8p_route_codeword_lut_accumulate_rhs_sorted_matmul();
}
"""

    guard = analyzer.audit_e8p_route_codeword_lut_accumulate_source(
        metal_source=metal_source,
        primitive_source=primitive_source,
    )

    assert guard["decision"] == (
        "route_codeword_lut_accumulate_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "route_slots_x_k_blocks_x_codewords_then_output_tiles"
    )
    assert guard["preserves_route_codeword_lut_descriptors"] is True
    assert guard["streams_compressed_codewords"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["accumulates_route_local_codeword_dot_lut"] is True
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["uses_expert_kblock_reuse_schedule"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_audit_rowwise_codeword_tile_accumulate_source_requires_native_guardrail() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_rowwise_codeword_tile_accumulate_source(
        metal_source="[[kernel]] void unrelated_kernel() {}",
        primitive_source="array unrelated() { return array(); }",
    )

    assert guard["decision"] == "missing_rowwise_codeword_tile_accumulate_source"
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["dispatch_grid"] == "unknown"
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert guard["required_next_features"] == [
        "add_rowwise_codeword_tile_accumulate_native_source_guardrail",
        "bind_rowwise_codeword_tile_accumulate_primitive",
        "preserve_route_microtile_and_output_tile_axes",
        "stream_compressed_uint16_codeword_tiles_and_scale_slots",
        "accumulate_online_activation_codeword_dots",
        "avoid_route_local_full_codeword_lut",
        "avoid_decoded_dense_rhs",
    ]
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_rowwise_codeword_tile_accumulate_before_source_guardrail",
        "do_not_retime_route_codeword_lut_accumulate",
        "do_not_materialize_decoded_dense_rhs",
    ]


def test_audit_output_tile_local_codeword_lut_source_requires_native_guardrail() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_output_tile_local_codeword_lut_source(
        metal_source="[[kernel]] void unrelated_kernel() {}",
        primitive_source="array unrelated() { return array(); }",
    )

    assert guard["decision"] == "missing_output_tile_local_codeword_lut_source"
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["dispatch_grid"] == "unknown"
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert guard["required_next_features"] == [
        "add_output_tile_local_codeword_lut_native_source_guardrail",
        "bind_output_tile_local_codeword_lut_primitive",
        "preserve_route_microtile_and_output_tile_axes",
        "build_output_tile_local_unique_codeword_lut",
        "reuse_activation_codeword_dots_across_output_rows",
        "stream_compressed_uint16_codeword_tiles_and_scale_slots",
        "avoid_rowwise_per_output_row_codeword_dot_recompute",
        "avoid_decoded_dense_rhs",
    ]
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_output_tile_local_codeword_lut_before_source_guardrail",
        "do_not_retime_rowwise_codeword_tile_accumulate",
        "do_not_materialize_decoded_dense_rhs",
    ]


def test_audit_route_microtile_codeword_block_reduce_source_requires_native_guardrail() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_route_microtile_codeword_block_reduce_source(
        metal_source="[[kernel]] void unrelated_kernel() {}",
        primitive_source="array unrelated() { return array(); }",
    )

    assert guard["decision"] == "missing_route_microtile_codeword_block_reduce_source"
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["dispatch_grid"] == "unknown"
    assert guard["preserves_route_microtile_axis"] is False
    assert guard["preserves_k_block_axis"] is False
    assert guard["preserves_codeword_block_axis"] is False
    assert guard["reduces_route_microtile_codeword_block_partials"] is False
    assert guard["writes_output_tiles_after_block_reduction"] is False
    assert guard["materializes_output_tile_local_full_lut"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert guard["required_next_features"] == [
        "add_route_microtile_codeword_block_reduce_native_source_guardrail",
        "bind_route_microtile_codeword_block_reduce_primitive",
        "preserve_route_microtile_and_k_block_axes",
        "reduce_route_microtile_codeword_block_partials",
        "write_back_output_tiles_after_block_reduction",
        "stream_compressed_uint16_codeword_tiles_and_scale_slots",
        "avoid_output_tile_local_full_codeword_pass",
        "avoid_decoded_dense_rhs",
    ]
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_route_microtile_codeword_block_reduce_before_source_guardrail",
        "do_not_retime_output_tile_local_codeword_lut",
        "do_not_materialize_decoded_dense_rhs",
    ]


def test_real_current_route_codeword_lut_accumulate_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_route_codeword_lut_accumulate_source(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal").read_text(
            encoding="utf-8"
        ),
        primitive_source=Path("native/vq_nax_ext/csrc/nax_fp16_primitive.mm").read_text(
            encoding="utf-8"
        ),
    )

    assert guard["decision"] == (
        "route_codeword_lut_accumulate_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["preserves_route_codeword_lut_descriptors"] is True
    assert guard["streams_compressed_codewords"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["accumulates_route_local_codeword_dot_lut"] is True
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["uses_expert_kblock_reuse_schedule"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_real_current_rowwise_codeword_tile_accumulate_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_rowwise_codeword_tile_accumulate_source(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal").read_text(
            encoding="utf-8"
        ),
        primitive_source=Path("native/vq_nax_ext/csrc/nax_fp16_primitive.mm").read_text(
            encoding="utf-8"
        ),
    )

    assert guard["decision"] == (
        "rowwise_codeword_tile_accumulate_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "route_microtiles_x_output_tiles_x_k_blocks_x_codeword_tiles"
    )
    assert guard["preserves_route_microtile_axis"] is True
    assert guard["preserves_output_tile_axis"] is True
    assert guard["preserves_k_block_axis"] is True
    assert guard["preserves_codeword_tile_axis"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["accumulates_online_activation_codeword_dots"] is True
    assert guard["materializes_route_local_full_lut"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_real_current_output_tile_local_codeword_lut_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_output_tile_local_codeword_lut_source(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal").read_text(
            encoding="utf-8"
        ),
        primitive_source=Path("native/vq_nax_ext/csrc/nax_fp16_primitive.mm").read_text(
            encoding="utf-8"
        ),
    )

    assert guard["decision"] == (
        "output_tile_local_codeword_lut_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "route_microtiles_x_output_tiles_x_k_blocks_x_unique_codewords_then_output_rows"
    )
    assert guard["preserves_route_microtile_axis"] is True
    assert guard["preserves_output_tile_axis"] is True
    assert guard["preserves_k_block_axis"] is True
    assert guard["builds_output_tile_local_unique_codeword_lut"] is True
    assert guard["reuses_activation_codeword_dots_across_output_rows"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["recomputes_per_output_row_codeword_dots"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_real_current_route_microtile_codeword_block_reduce_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_route_microtile_codeword_block_reduce_source(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal").read_text(
            encoding="utf-8"
        ),
        primitive_source=Path("native/vq_nax_ext/csrc/nax_fp16_primitive.mm").read_text(
            encoding="utf-8"
        ),
    )

    assert guard["decision"] == (
        "route_microtile_codeword_block_reduce_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "route_microtiles_x_k_blocks_x_codeword_blocks_then_output_tiles"
    )
    assert guard["preserves_route_microtile_axis"] is True
    assert guard["preserves_k_block_axis"] is True
    assert guard["preserves_codeword_block_axis"] is True
    assert guard["reduces_route_microtile_codeword_block_partials"] is True
    assert guard["writes_output_tiles_after_block_reduction"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["materializes_output_tile_local_full_lut"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_audit_kblock_wavefront_codeword_scan_source_requires_native_guardrail() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_kblock_wavefront_codeword_scan_source(
        metal_source="[[kernel]] void unrelated_kernel() {}",
        primitive_source="array unrelated() { return array(); }",
    )

    assert guard["decision"] == "missing_kblock_wavefront_codeword_scan_source"
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["dispatch_grid"] == "unknown"
    assert guard["preserves_k_block_axis"] is False
    assert guard["preserves_route_microtile_axis"] is False
    assert guard["preserves_output_stripe_axis"] is False
    assert guard["preserves_codeword_group_axis"] is False
    assert guard["streams_codeword_groups_as_wavefronts"] is False
    assert guard["materializes_route_microtile_partial_cache"] is False
    assert guard["materializes_output_tile_local_full_lut"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert guard["required_next_features"] == [
        "add_kblock_wavefront_codeword_scan_native_source_guardrail",
        "bind_kblock_wavefront_codeword_scan_primitive",
        "preserve_k_block_route_microtile_output_stripe_codeword_group_axes",
        "stream_compressed_codeword_groups_as_kblock_wavefronts",
        "stream_compressed_uint16_codeword_tiles_and_scale_slots",
        "avoid_route_microtile_codeword_block_partial_cache",
        "avoid_output_tile_local_full_codeword_pass",
        "avoid_decoded_dense_rhs",
    ]
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_kblock_wavefront_codeword_scan_before_source_guardrail",
        "do_not_retime_route_microtile_codeword_block_reduce",
        "do_not_materialize_decoded_dense_rhs",
    ]


def test_audit_kblock_wavefront_codeword_scan_source_accepts_contract() -> None:
    analyzer = _load_analyzer()
    metal_source = """
[[kernel]] void nax_e8p_kblock_wavefront_codeword_scan_rhs_sorted_matmul() {
  device const uint16_t* codeword_groups;
  device const half* codeword_scale_slots;
  device const half* scale_tiles;
  uint k_block = 0;
  uint route_microtile = 0;
  uint output_stripe = 0;
  uint codeword_group = 0;
  uint kblock_wavefront = k_block + codeword_group;
  uint wavefront = kblock_wavefront + route_microtile + output_stripe;
  uint codeword_tiles = uint(codeword_groups[wavefront]);
}
[[kernel]] void other_kernel() {}
"""
    primitive_source = """
array e8p_kblock_wavefront_codeword_scan_rhs_sorted_matmul() {
  auto* pso = cache.get("nax_e8p_kblock_wavefront_codeword_scan_rhs_sorted_matmul");
  // k_blocks_x_route_microtiles_x_output_stripes_x_codeword_groups
  encoder.dispatch_threadgroups(MTL::Size::Make(k_block_count, route_microtile_count, output_stripe_count),
                                MTL::Size::Make(64, 1, 1));
}
"""

    guard = analyzer.audit_e8p_kblock_wavefront_codeword_scan_source(
        metal_source=metal_source,
        primitive_source=primitive_source,
    )

    assert guard["decision"] == (
        "kblock_wavefront_codeword_scan_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "k_blocks_x_route_microtiles_x_output_stripes_x_codeword_groups"
    )
    assert guard["preserves_k_block_axis"] is True
    assert guard["preserves_route_microtile_axis"] is True
    assert guard["preserves_output_stripe_axis"] is True
    assert guard["preserves_codeword_group_axis"] is True
    assert guard["streams_codeword_groups_as_wavefronts"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["materializes_route_microtile_partial_cache"] is False
    assert guard["materializes_output_tile_local_full_lut"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_audit_token_route_output_stripe_pipeline_source_requires_native_guardrail() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_token_route_output_stripe_pipeline_source(
        metal_source="[[kernel]] void unrelated_kernel() {}",
        primitive_source="array unrelated() { return array(); }",
    )

    assert guard["decision"] == "missing_token_route_output_stripe_pipeline_source"
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["dispatch_grid"] == "unknown"
    assert guard["preserves_token_axis"] is False
    assert guard["preserves_route_slot_axis"] is False
    assert guard["preserves_output_stripe_axis"] is False
    assert guard["preserves_kblock_stage_axis"] is False
    assert guard["accumulates_full_output_without_route_expansion"] is False
    assert guard["streams_kblock_stages_inside_token_route_output_stripes"] is False
    assert guard["materializes_route_microtile_partial_cache"] is False
    assert guard["materializes_output_tile_local_full_lut"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["uses_rejected_kblock_wavefront_schedule"] is False
    assert guard["expands_route_microtiles"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert guard["required_next_features"] == [
        "add_token_route_output_stripe_pipeline_native_source_guardrail",
        "bind_token_route_output_stripe_pipeline_primitive",
        "preserve_token_route_slot_output_stripe_kblock_stage_axes",
        "accumulate_full_output_inside_token_route_output_stripes",
        "stream_kblock_stages_inside_each_token_route_output_stripe",
        "stream_compressed_uint16_codeword_tiles_and_scale_slots",
        "avoid_route_microtile_output_expansion",
        "avoid_decoded_dense_rhs",
    ]
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_token_route_output_stripe_pipeline_before_source_guardrail",
        "do_not_retime_kblock_wavefront_codeword_scan",
        "do_not_materialize_decoded_dense_rhs",
    ]


def test_audit_token_route_output_stripe_pipeline_source_accepts_contract() -> None:
    analyzer = _load_analyzer()
    metal_source = """
[[kernel]] void nax_e8p_token_route_output_stripe_pipeline_rhs_sorted_matmul() {
  device const uint16_t* codeword_tiles;
  device const half* codeword_scale_slots;
  device const half* scale_tiles;
  uint token = 0;
  uint route_slot = 0;
  uint output_stripe = 0;
  uint kblock_stage = 0;
  uint token_route_output_stripe = token + route_slot + output_stripe;
  half accum = half(codeword_tiles[token_route_output_stripe + kblock_stage]);
  accum += codeword_scale_slots[kblock_stage] + scale_tiles[output_stripe];
}
[[kernel]] void other_kernel() {}
"""
    primitive_source = """
array e8p_token_route_output_stripe_pipeline_rhs_sorted_matmul() {
  auto* pso = cache.get("nax_e8p_token_route_output_stripe_pipeline_rhs_sorted_matmul");
  // tokens_x_route_slots_x_output_stripes_x_kblock_stages
  encoder.dispatch_threadgroups(MTL::Size::Make(token_count, route_slot_count, output_stripe_count),
                                MTL::Size::Make(64, 1, 1));
}
"""

    guard = analyzer.audit_e8p_token_route_output_stripe_pipeline_source(
        metal_source=metal_source,
        primitive_source=primitive_source,
    )

    assert guard["decision"] == (
        "token_route_output_stripe_pipeline_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "tokens_x_route_slots_x_output_stripes_x_kblock_stages"
    )
    assert guard["preserves_token_axis"] is True
    assert guard["preserves_route_slot_axis"] is True
    assert guard["preserves_output_stripe_axis"] is True
    assert guard["preserves_kblock_stage_axis"] is True
    assert guard["accumulates_full_output_without_route_expansion"] is True
    assert guard["streams_kblock_stages_inside_token_route_output_stripes"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["materializes_route_microtile_partial_cache"] is False
    assert guard["materializes_output_tile_local_full_lut"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["uses_rejected_kblock_wavefront_schedule"] is False
    assert guard["expands_route_microtiles"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_audit_expert_kblock_scale_slot_stream_source_requires_native_guardrail() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_expert_kblock_scale_slot_stream_source(
        metal_source="[[kernel]] void unrelated_kernel() {}",
        primitive_source="array unrelated() { return array(); }",
    )

    assert guard["decision"] == "missing_expert_kblock_scale_slot_stream_source"
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["dispatch_grid"] == "unknown"
    assert guard["preserves_expert_axis"] is False
    assert guard["preserves_k_block_axis"] is False
    assert guard["preserves_scale_group_axis"] is False
    assert guard["preserves_route_tile_axis"] is False
    assert guard["preserves_output_tile_axis"] is False
    assert guard["streams_scale_slot_groups"] is False
    assert guard["reuses_codeword_scale_slots_across_output_tiles"] is False
    assert guard["streams_compressed_codeword_tiles"] is False
    assert guard["uses_codeword_scale_slots"] is False
    assert guard["uses_scale_group_indices"] is False
    assert guard["uses_scale_tiles"] is False
    assert guard["uses_rejected_token_route_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert guard["required_next_features"] == [
        "add_expert_kblock_scale_slot_stream_native_source_guardrail",
        "bind_expert_kblock_scale_slot_stream_primitive",
        "preserve_expert_kblock_scale_group_route_tile_output_tile_axes",
        "stream_compressed_uint16_codeword_tiles_and_scale_slots",
        "preserve_scale_group_indices_for_group_size_352",
        "reuse_codeword_scale_slots_across_output_tiles_without_decoded_rhs_cache",
        "avoid_token_route_output_stripe_pipeline_schedule",
        "avoid_decoded_dense_rhs",
    ]
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_expert_kblock_scale_slot_stream_before_source_guardrail",
        "do_not_retime_token_route_output_stripe_pipeline",
        "do_not_materialize_decoded_dense_rhs",
    ]


def test_audit_expert_kblock_scale_slot_stream_source_accepts_contract() -> None:
    analyzer = _load_analyzer()
    metal_source = """
[[kernel]] void nax_e8p_expert_kblock_scale_slot_stream_rhs_sorted_matmul() {
  device const uint16_t* codeword_tiles;
  device const half* codeword_scale_slots;
  device const uint* scale_group_indices;
  device const half* scale_tiles;
  uint expert = 0;
  uint k_block = 0;
  uint scale_group = scale_group_indices[k_block];
  uint route_tile = 0;
  uint output_tile = 0;
  uint scale_slot_stream = expert + k_block + scale_group + route_tile + output_tile;
  half accum = half(codeword_tiles[scale_slot_stream]);
  accum += codeword_scale_slots[scale_group] + scale_tiles[output_tile];
}
[[kernel]] void other_kernel() {}
"""
    primitive_source = """
array e8p_expert_kblock_scale_slot_stream_rhs_sorted_matmul() {
  auto* pso = cache.get("nax_e8p_expert_kblock_scale_slot_stream_rhs_sorted_matmul");
  // experts_x_k_blocks_x_scale_groups_x_route_tiles_x_output_tiles
  encoder.dispatch_threadgroups(MTL::Size::Make(expert_count, k_block_count, scale_group_count),
                                MTL::Size::Make(64, 1, 1));
}
"""

    guard = analyzer.audit_e8p_expert_kblock_scale_slot_stream_source(
        metal_source=metal_source,
        primitive_source=primitive_source,
    )

    assert guard["decision"] == "expert_kblock_scale_slot_stream_source_guardrail_present"
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "experts_x_k_blocks_x_scale_groups_x_route_tiles_x_output_tiles"
    )
    assert guard["preserves_expert_axis"] is True
    assert guard["preserves_k_block_axis"] is True
    assert guard["preserves_scale_group_axis"] is True
    assert guard["preserves_route_tile_axis"] is True
    assert guard["preserves_output_tile_axis"] is True
    assert guard["streams_scale_slot_groups"] is True
    assert guard["reuses_codeword_scale_slots_across_output_tiles"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_group_indices"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["uses_rejected_token_route_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_audit_scale_group_route_block_reduce_source_requires_native_guardrail() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_scale_group_route_block_reduce_source(
        metal_source="[[kernel]] void unrelated_kernel() {}",
        primitive_source="array unrelated() { return array(); }",
    )

    assert guard["decision"] == "missing_scale_group_route_block_reduce_source"
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["dispatch_grid"] == "unknown"
    assert guard["preserves_scale_group_axis"] is False
    assert guard["preserves_route_block_axis"] is False
    assert guard["preserves_k_block_axis"] is False
    assert guard["preserves_output_tile_axis"] is False
    assert guard["reduces_route_block_partials_before_output_writeback"] is False
    assert guard["streams_compressed_codeword_tiles"] is False
    assert guard["uses_codeword_scale_slots"] is False
    assert guard["uses_scale_group_indices"] is False
    assert guard["uses_scale_tiles"] is False
    assert guard["uses_rejected_expert_kblock_scale_slot_schedule"] is False
    assert guard["uses_rejected_token_route_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert guard["required_next_features"] == [
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
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_scale_group_route_block_reduce_before_source_guardrail",
        "do_not_retime_expert_kblock_scale_slot_stream",
        "do_not_materialize_decoded_dense_rhs",
    ]


def test_audit_scale_group_route_block_reduce_source_accepts_contract() -> None:
    analyzer = _load_analyzer()
    metal_source = """
[[kernel]] void nax_e8p_scale_group_route_block_reduce_rhs_sorted_matmul() {
  device const uint16_t* codeword_tiles;
  device const half* codeword_scale_slots;
  device const uint* scale_group_indices;
  device const half* scale_tiles;
  uint scale_group = scale_group_indices[0];
  uint route_block = 0;
  uint k_block = 0;
  uint output_tile = 0;
  float route_block_partial = float(codeword_tiles[route_block + k_block]);
  route_block_partial += float(codeword_scale_slots[scale_group]);
  route_block_partial += float(scale_tiles[output_tile]);
}
[[kernel]] void other_kernel() {}
"""
    primitive_source = """
array e8p_scale_group_route_block_reduce_rhs_sorted_matmul() {
  auto* pso = cache.get("nax_e8p_scale_group_route_block_reduce_rhs_sorted_matmul");
  // scale_groups_x_route_blocks_x_k_blocks_x_output_tiles
  encoder.dispatch_threadgroups(MTL::Size::Make(scale_group_count, route_block_count, k_block_count),
                                MTL::Size::Make(64, 1, 1));
}
"""

    guard = analyzer.audit_e8p_scale_group_route_block_reduce_source(
        metal_source=metal_source,
        primitive_source=primitive_source,
    )

    assert guard["decision"] == (
        "scale_group_route_block_reduce_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "scale_groups_x_route_blocks_x_k_blocks_x_output_tiles"
    )
    assert guard["preserves_scale_group_axis"] is True
    assert guard["preserves_route_block_axis"] is True
    assert guard["preserves_k_block_axis"] is True
    assert guard["preserves_output_tile_axis"] is True
    assert guard["reduces_route_block_partials_before_output_writeback"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_group_indices"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["uses_rejected_expert_kblock_scale_slot_schedule"] is False
    assert guard["uses_rejected_token_route_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_audit_route_block_output_group_stream_source_requires_native_guardrail() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_route_block_output_group_stream_source(
        metal_source="[[kernel]] void unrelated_kernel() {}",
        primitive_source="array unrelated() { return array(); }",
    )

    assert guard["decision"] == "missing_route_block_output_group_stream_source"
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["dispatch_grid"] == "unknown"
    assert guard["preserves_route_block_axis"] is False
    assert guard["preserves_output_group_axis"] is False
    assert guard["preserves_k_block_axis"] is False
    assert guard["preserves_codeword_group_axis"] is False
    assert guard["writes_output_group_accumulators_to_route_slots"] is False
    assert guard["streams_codeword_groups_inside_route_block_output_groups"] is False
    assert guard["streams_compressed_codeword_tiles"] is False
    assert guard["uses_codeword_scale_slots"] is False
    assert guard["uses_scale_tiles"] is False
    assert guard["uses_rejected_scale_group_route_block_schedule"] is False
    assert guard["uses_rejected_expert_kblock_scale_slot_schedule"] is False
    assert guard["uses_rejected_token_route_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert guard["required_next_features"] == [
        "add_route_block_output_group_stream_native_source_guardrail",
        "bind_route_block_output_group_stream_primitive",
        "preserve_route_block_output_group_k_block_codeword_group_axes",
        "write_output_group_accumulators_directly_to_route_slots",
        "stream_compressed_uint16_codeword_tiles_and_scale_slots",
        "avoid_scale_group_route_block_reduce_schedule",
        "avoid_decoded_dense_rhs",
    ]
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_route_block_output_group_stream_before_source_guardrail",
        "do_not_retime_scale_group_route_block_reduce",
        "do_not_materialize_decoded_dense_rhs",
    ]


def test_audit_route_block_output_group_stream_source_accepts_contract() -> None:
    analyzer = _load_analyzer()
    metal_source = """
[[kernel]] void nax_e8p_route_block_output_group_stream_rhs_sorted_matmul() {
  device const uint16_t* codeword_tiles;
  device const half* codeword_scale_slots;
  device const half* scale_tiles;
  uint route_block = 0;
  uint output_group = 0;
  uint k_block = 0;
  uint codeword_group = 0;
  uint route_slot = route_block;
  uint route_block_output_group = route_block + output_group;
  float output_group_accumulator = float(codeword_tiles[route_block_output_group + k_block + codeword_group]);
  output_group_accumulator += float(codeword_scale_slots[codeword_group]);
  output_group_accumulator += float(scale_tiles[output_group]);
  float route_slot_output_accumulator = output_group_accumulator + float(route_slot);
}
[[kernel]] void other_kernel() {}
"""
    primitive_source = """
array e8p_route_block_output_group_stream_rhs_sorted_matmul() {
  auto* pso = cache.get("nax_e8p_route_block_output_group_stream_rhs_sorted_matmul");
  // route_blocks_x_output_groups_x_k_blocks_x_codeword_groups
  encoder.dispatch_threadgroups(MTL::Size::Make(route_block_count, output_group_count, k_block_count),
                                MTL::Size::Make(64, 1, 1));
}
"""

    guard = analyzer.audit_e8p_route_block_output_group_stream_source(
        metal_source=metal_source,
        primitive_source=primitive_source,
    )

    assert guard["decision"] == (
        "route_block_output_group_stream_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "route_blocks_x_output_groups_x_k_blocks_x_codeword_groups"
    )
    assert guard["preserves_route_block_axis"] is True
    assert guard["preserves_output_group_axis"] is True
    assert guard["preserves_k_block_axis"] is True
    assert guard["preserves_codeword_group_axis"] is True
    assert guard["writes_output_group_accumulators_to_route_slots"] is True
    assert guard["streams_codeword_groups_inside_route_block_output_groups"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["uses_rejected_scale_group_route_block_schedule"] is False
    assert guard["uses_rejected_expert_kblock_scale_slot_schedule"] is False
    assert guard["uses_rejected_token_route_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_audit_output_group_pretransposed_codeword_stream_source_requires_native_guardrail() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_output_group_pretransposed_codeword_stream_source(
        metal_source="[[kernel]] void unrelated_kernel() {}",
        primitive_source="array unrelated() { return array(); }",
    )

    assert (
        guard["decision"]
        == "missing_output_group_pretransposed_codeword_stream_source"
    )
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["dispatch_grid"] == "unknown"
    assert guard["preserves_output_group_axis"] is False
    assert guard["preserves_route_block_axis"] is False
    assert guard["preserves_k_block_axis"] is False
    assert guard["preserves_codeword_group_axis"] is False
    assert guard["uses_output_group_pretransposed_streams"] is False
    assert guard["feeds_route_block_accumulators"] is False
    assert guard["streams_compressed_codeword_tiles"] is False
    assert guard["uses_codeword_scale_slots"] is False
    assert guard["uses_scale_tiles"] is False
    assert guard["uses_rejected_route_block_output_group_schedule"] is False
    assert guard["uses_rejected_scale_group_route_block_schedule"] is False
    assert guard["uses_rejected_expert_kblock_scale_slot_schedule"] is False
    assert guard["uses_rejected_token_route_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert guard["required_next_features"] == [
        "add_output_group_pretransposed_codeword_stream_native_source_guardrail",
        "bind_output_group_pretransposed_codeword_stream_primitive",
        "preserve_output_group_route_block_k_block_codeword_group_axes",
        "make_output_group_pretransposed_streams_outer_schedule",
        "stream_compressed_uint16_codeword_tiles_and_scale_slots",
        "avoid_route_block_output_group_stream_schedule",
        "avoid_decoded_dense_rhs",
    ]
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_output_group_pretransposed_codeword_stream_before_source_guardrail",
        "do_not_retime_route_block_output_group_stream",
        "do_not_materialize_decoded_dense_rhs",
    ]


def test_audit_output_group_pretransposed_codeword_stream_source_accepts_contract() -> None:
    analyzer = _load_analyzer()
    metal_source = """
[[kernel]] void nax_e8p_output_group_pretransposed_codeword_stream_rhs_sorted_matmul() {
  device const uint16_t* codeword_tiles;
  device const half* codeword_scale_slots;
  device const half* scale_tiles;
  uint output_group = 0;
  uint route_block = 0;
  uint k_block = 0;
  uint codeword_group = 0;
  uint output_group_pretransposed = output_group;
  float route_block_accumulator = float(codeword_tiles[output_group_pretransposed + route_block + k_block + codeword_group]);
  route_block_accumulator += float(codeword_scale_slots[codeword_group]);
  route_block_accumulator += float(scale_tiles[output_group]);
}
[[kernel]] void other_kernel() {}
"""
    primitive_source = """
array e8p_output_group_pretransposed_codeword_stream_rhs_sorted_matmul() {
  auto* pso = cache.get("nax_e8p_output_group_pretransposed_codeword_stream_rhs_sorted_matmul");
  // output_groups_x_route_blocks_x_k_blocks_x_codeword_groups
  encoder.dispatch_threadgroups(MTL::Size::Make(output_group_count, route_block_count, k_block_count),
                                MTL::Size::Make(64, 1, 1));
}
"""

    guard = analyzer.audit_e8p_output_group_pretransposed_codeword_stream_source(
        metal_source=metal_source,
        primitive_source=primitive_source,
    )

    assert guard["decision"] == (
        "output_group_pretransposed_codeword_stream_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "output_groups_x_route_blocks_x_k_blocks_x_codeword_groups"
    )
    assert guard["preserves_output_group_axis"] is True
    assert guard["preserves_route_block_axis"] is True
    assert guard["preserves_k_block_axis"] is True
    assert guard["preserves_codeword_group_axis"] is True
    assert guard["uses_output_group_pretransposed_streams"] is True
    assert guard["feeds_route_block_accumulators"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["uses_rejected_route_block_output_group_schedule"] is False
    assert guard["uses_rejected_scale_group_route_block_schedule"] is False
    assert guard["uses_rejected_expert_kblock_scale_slot_schedule"] is False
    assert guard["uses_rejected_token_route_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_audit_kblock_output_group_route_fused_stream_source_requires_native_guardrail() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_kblock_output_group_route_fused_stream_source(
        metal_source="[[kernel]] void unrelated_kernel() {}",
        primitive_source="array unrelated() { return array(); }",
    )

    assert guard["decision"] == "missing_kblock_output_group_route_fused_stream_source"
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["dispatch_grid"] == "unknown"
    assert guard["preserves_k_block_axis"] is False
    assert guard["preserves_output_group_axis"] is False
    assert guard["preserves_route_block_axis"] is False
    assert guard["preserves_codeword_group_axis"] is False
    assert guard["uses_kblock_output_group_route_fusion"] is False
    assert guard["writes_route_slots_after_fusion"] is False
    assert guard["streams_compressed_codeword_tiles"] is False
    assert guard["uses_codeword_scale_slots"] is False
    assert guard["uses_scale_tiles"] is False
    assert guard["uses_rejected_output_group_pretransposed_schedule"] is False
    assert guard["uses_rejected_route_block_output_group_schedule"] is False
    assert guard["uses_rejected_scale_group_route_block_schedule"] is False
    assert guard["uses_rejected_expert_kblock_scale_slot_schedule"] is False
    assert guard["uses_rejected_token_route_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert guard["required_next_features"] == [
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
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_kblock_output_group_route_fused_stream_before_source_guardrail",
        "do_not_retime_output_group_pretransposed_codeword_stream",
        "do_not_materialize_decoded_dense_rhs",
    ]


def test_audit_kblock_output_group_route_fused_stream_source_accepts_contract() -> None:
    analyzer = _load_analyzer()
    metal_source = """
[[kernel]] void nax_e8p_kblock_output_group_route_fused_stream_rhs_sorted_matmul() {
  device const uint16_t* codeword_tiles;
  device const half* codeword_scale_slots;
  device const half* scale_tiles;
  uint k_block = 0;
  uint output_group = 0;
  uint route_block = 0;
  uint codeword_group = 0;
  uint route_slot = route_block;
  uint kblock_output_group_route_fused = k_block + output_group + route_block;
  float route_fused_accumulator = float(codeword_tiles[kblock_output_group_route_fused + codeword_group]);
  route_fused_accumulator += float(codeword_scale_slots[codeword_group]);
  route_fused_accumulator += float(scale_tiles[output_group]);
  float route_slot_output = route_fused_accumulator + float(route_slot);
}
[[kernel]] void other_kernel() {}
"""
    primitive_source = """
array e8p_kblock_output_group_route_fused_stream_rhs_sorted_matmul() {
  auto* pso = cache.get("nax_e8p_kblock_output_group_route_fused_stream_rhs_sorted_matmul");
  // k_blocks_x_output_groups_x_route_blocks_x_codeword_groups
  encoder.dispatch_threadgroups(MTL::Size::Make(k_block_count, output_group_count, route_block_count),
                                MTL::Size::Make(64, 1, 1));
}
"""

    guard = analyzer.audit_e8p_kblock_output_group_route_fused_stream_source(
        metal_source=metal_source,
        primitive_source=primitive_source,
    )

    assert guard["decision"] == (
        "kblock_output_group_route_fused_stream_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "k_blocks_x_output_groups_x_route_blocks_x_codeword_groups"
    )
    assert guard["preserves_k_block_axis"] is True
    assert guard["preserves_output_group_axis"] is True
    assert guard["preserves_route_block_axis"] is True
    assert guard["preserves_codeword_group_axis"] is True
    assert guard["uses_kblock_output_group_route_fusion"] is True
    assert guard["writes_route_slots_after_fusion"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["uses_rejected_output_group_pretransposed_schedule"] is False
    assert guard["uses_rejected_route_block_output_group_schedule"] is False
    assert guard["uses_rejected_scale_group_route_block_schedule"] is False
    assert guard["uses_rejected_expert_kblock_scale_slot_schedule"] is False
    assert guard["uses_rejected_token_route_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_real_current_expert_kblock_scale_slot_stream_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_expert_kblock_scale_slot_stream_source(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal").read_text(
            encoding="utf-8"
        ),
        primitive_source=Path("native/vq_nax_ext/csrc/nax_fp16_primitive.mm").read_text(
            encoding="utf-8"
        ),
    )

    assert guard["decision"] == (
        "expert_kblock_scale_slot_stream_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "experts_x_k_blocks_x_scale_groups_x_route_tiles_x_output_tiles"
    )
    assert guard["preserves_expert_axis"] is True
    assert guard["preserves_k_block_axis"] is True
    assert guard["preserves_scale_group_axis"] is True
    assert guard["preserves_route_tile_axis"] is True
    assert guard["preserves_output_tile_axis"] is True
    assert guard["streams_scale_slot_groups"] is True
    assert guard["reuses_codeword_scale_slots_across_output_tiles"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_group_indices"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["uses_rejected_token_route_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_real_current_scale_group_route_block_reduce_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_scale_group_route_block_reduce_source(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal").read_text(
            encoding="utf-8"
        ),
        primitive_source=Path("native/vq_nax_ext/csrc/nax_fp16_primitive.mm").read_text(
            encoding="utf-8"
        ),
    )

    assert guard["decision"] == (
        "scale_group_route_block_reduce_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "scale_groups_x_route_blocks_x_k_blocks_x_output_tiles"
    )
    assert guard["preserves_scale_group_axis"] is True
    assert guard["preserves_route_block_axis"] is True
    assert guard["preserves_k_block_axis"] is True
    assert guard["preserves_output_tile_axis"] is True
    assert guard["reduces_route_block_partials_before_output_writeback"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_group_indices"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["uses_rejected_expert_kblock_scale_slot_schedule"] is False
    assert guard["uses_rejected_token_route_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_real_current_route_block_output_group_stream_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_route_block_output_group_stream_source(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal").read_text(
            encoding="utf-8"
        ),
        primitive_source=Path("native/vq_nax_ext/csrc/nax_fp16_primitive.mm").read_text(
            encoding="utf-8"
        ),
    )

    assert guard["decision"] == (
        "route_block_output_group_stream_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "route_blocks_x_output_groups_x_k_blocks_x_codeword_groups"
    )
    assert guard["preserves_route_block_axis"] is True
    assert guard["preserves_output_group_axis"] is True
    assert guard["preserves_k_block_axis"] is True
    assert guard["preserves_codeword_group_axis"] is True
    assert guard["writes_output_group_accumulators_to_route_slots"] is True
    assert guard["streams_codeword_groups_inside_route_block_output_groups"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["uses_rejected_scale_group_route_block_schedule"] is False
    assert guard["uses_rejected_expert_kblock_scale_slot_schedule"] is False
    assert guard["uses_rejected_token_route_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_real_current_output_group_pretransposed_codeword_stream_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_output_group_pretransposed_codeword_stream_source(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal").read_text(
            encoding="utf-8"
        ),
        primitive_source=Path("native/vq_nax_ext/csrc/nax_fp16_primitive.mm").read_text(
            encoding="utf-8"
        ),
    )

    assert guard["decision"] == (
        "output_group_pretransposed_codeword_stream_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "output_groups_x_route_blocks_x_k_blocks_x_codeword_groups"
    )
    assert guard["preserves_output_group_axis"] is True
    assert guard["preserves_route_block_axis"] is True
    assert guard["preserves_k_block_axis"] is True
    assert guard["preserves_codeword_group_axis"] is True
    assert guard["uses_output_group_pretransposed_streams"] is True
    assert guard["feeds_route_block_accumulators"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["uses_rejected_route_block_output_group_schedule"] is False
    assert guard["uses_rejected_scale_group_route_block_schedule"] is False
    assert guard["uses_rejected_expert_kblock_scale_slot_schedule"] is False
    assert guard["uses_rejected_token_route_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_real_current_kblock_output_group_route_fused_stream_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_kblock_output_group_route_fused_stream_source(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal").read_text(
            encoding="utf-8"
        ),
        primitive_source=Path("native/vq_nax_ext/csrc/nax_fp16_primitive.mm").read_text(
            encoding="utf-8"
        ),
    )

    assert guard["decision"] == (
        "kblock_output_group_route_fused_stream_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "k_blocks_x_output_groups_x_route_blocks_x_codeword_groups"
    )
    assert guard["preserves_k_block_axis"] is True
    assert guard["preserves_output_group_axis"] is True
    assert guard["preserves_route_block_axis"] is True
    assert guard["preserves_codeword_group_axis"] is True
    assert guard["uses_kblock_output_group_route_fusion"] is True
    assert guard["writes_route_slots_after_fusion"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["uses_rejected_output_group_pretransposed_schedule"] is False
    assert guard["uses_rejected_route_block_output_group_schedule"] is False
    assert guard["uses_rejected_scale_group_route_block_schedule"] is False
    assert guard["uses_rejected_expert_kblock_scale_slot_schedule"] is False
    assert guard["uses_rejected_token_route_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_real_current_route_tile_output_swizzle_stream_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_route_tile_output_swizzle_stream_source(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal").read_text(
            encoding="utf-8"
        ),
        primitive_source=Path("native/vq_nax_ext/csrc/nax_fp16_primitive.mm").read_text(
            encoding="utf-8"
        ),
    )

    assert guard["decision"] == (
        "route_tile_output_swizzle_stream_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "route_tiles_x_output_swizzles_x_k_blocks_x_codeword_groups"
    )
    assert guard["preserves_route_tile_axis"] is True
    assert guard["preserves_output_swizzle_axis"] is True
    assert guard["preserves_k_block_axis"] is True
    assert guard["preserves_codeword_group_axis"] is True
    assert guard["uses_route_tile_output_swizzle_streams"] is True
    assert guard["writes_route_slots_after_output_swizzle"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["uses_rejected_kblock_output_group_route_fused_schedule"] is False
    assert guard["uses_rejected_output_group_pretransposed_schedule"] is False
    assert guard["uses_rejected_route_block_output_group_schedule"] is False
    assert guard["uses_rejected_scale_group_route_block_schedule"] is False
    assert guard["uses_rejected_token_route_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_real_current_token_topk_output_tile_stream_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_token_topk_output_tile_stream_source(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal").read_text(
            encoding="utf-8"
        ),
        primitive_source=Path("native/vq_nax_ext/csrc/nax_fp16_primitive.mm").read_text(
            encoding="utf-8"
        ),
    )

    assert guard["decision"] == (
        "token_topk_output_tile_stream_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "tokens_x_topk_slots_x_output_tiles_x_k_blocks_x_codeword_groups"
    )
    assert guard["preserves_token_axis"] is True
    assert guard["preserves_topk_slot_axis"] is True
    assert guard["preserves_output_tile_axis"] is True
    assert guard["preserves_k_block_axis"] is True
    assert guard["preserves_codeword_group_axis"] is True
    assert guard["uses_token_topk_output_tile_streams"] is True
    assert guard["preserves_q2_token_topk_output_shape"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["uses_rejected_route_tile_output_swizzle_schedule"] is False
    assert guard["uses_rejected_kblock_output_group_route_fused_schedule"] is False
    assert guard["uses_rejected_output_group_pretransposed_schedule"] is False
    assert guard["uses_rejected_route_block_output_group_schedule"] is False
    assert guard["uses_rejected_scale_group_route_block_schedule"] is False
    assert guard["uses_rejected_token_route_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_audit_token_topk_output_tile_stream_source_requires_native_guardrail() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_token_topk_output_tile_stream_source(
        metal_source="[[kernel]] void unrelated_kernel() {}",
        primitive_source="array unrelated() { return array(); }",
    )

    assert guard["decision"] == "missing_token_topk_output_tile_stream_source"
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["dispatch_grid"] == "unknown"
    assert guard["preserves_token_axis"] is False
    assert guard["preserves_topk_slot_axis"] is False
    assert guard["preserves_output_tile_axis"] is False
    assert guard["preserves_k_block_axis"] is False
    assert guard["preserves_codeword_group_axis"] is False
    assert guard["uses_token_topk_output_tile_streams"] is False
    assert guard["preserves_q2_token_topk_output_shape"] is False
    assert guard["streams_compressed_codeword_tiles"] is False
    assert guard["uses_codeword_scale_slots"] is False
    assert guard["uses_scale_tiles"] is False
    assert guard["uses_rejected_route_tile_output_swizzle_schedule"] is False
    assert guard["uses_rejected_kblock_output_group_route_fused_schedule"] is False
    assert guard["uses_rejected_output_group_pretransposed_schedule"] is False
    assert guard["uses_rejected_route_block_output_group_schedule"] is False
    assert guard["uses_rejected_scale_group_route_block_schedule"] is False
    assert guard["uses_rejected_token_route_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert guard["required_next_features"] == [
        "add_token_topk_output_tile_stream_native_source_guardrail",
        "bind_token_topk_output_tile_stream_primitive",
        "preserve_tokens_topk_slots_output_tiles_kblocks_codeword_group_axes",
        "preserve_q2_tokens_x_topk_slots_x_output_tiles_writeback",
        "stream_compressed_uint16_codeword_tiles_and_scale_slots",
        "avoid_route_tile_output_swizzle_stream_schedule",
        "avoid_decoded_dense_rhs",
    ]
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_token_topk_output_tile_stream_before_source_guardrail",
        "do_not_retime_route_tile_output_swizzle_stream",
        "do_not_materialize_decoded_dense_rhs",
    ]


def test_audit_token_block_output_group_stream_source_requires_native_guardrail() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_token_block_output_group_stream_source(
        metal_source="[[kernel]] void unrelated_kernel() {}",
        primitive_source="array unrelated() { return array(); }",
    )

    assert guard["decision"] == "missing_token_block_output_group_stream_source"
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["dispatch_grid"] == "unknown"
    assert guard["preserves_token_block_axis"] is False
    assert guard["preserves_output_group_axis"] is False
    assert guard["preserves_topk_slot_axis"] is False
    assert guard["preserves_k_block_axis"] is False
    assert guard["preserves_codeword_group_axis"] is False
    assert guard["uses_token_block_output_group_streams"] is False
    assert guard["preserves_token_topk_output_writeback"] is False
    assert guard["streams_compressed_codeword_tiles"] is False
    assert guard["uses_codeword_scale_slots"] is False
    assert guard["uses_scale_tiles"] is False
    assert guard["uses_rejected_token_topk_output_tile_schedule"] is False
    assert guard["uses_rejected_route_tile_output_swizzle_schedule"] is False
    assert guard["uses_rejected_kblock_output_group_route_fused_schedule"] is False
    assert guard["uses_rejected_output_group_pretransposed_schedule"] is False
    assert guard["uses_rejected_route_block_output_group_schedule"] is False
    assert guard["uses_rejected_scale_group_route_block_schedule"] is False
    assert guard["uses_rejected_token_route_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert guard["required_next_features"] == [
        "add_token_block_output_group_stream_native_source_guardrail",
        "bind_token_block_output_group_stream_primitive",
        "preserve_token_blocks_output_groups_topk_slots_kblocks_codeword_group_axes",
        "preserve_token_topk_output_writeback",
        "stream_compressed_uint16_codeword_tiles_and_scale_slots",
        "avoid_token_topk_output_tile_stream_schedule",
        "avoid_decoded_dense_rhs",
    ]
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_token_block_output_group_stream_before_source_guardrail",
        "do_not_retime_token_topk_output_tile_stream",
        "do_not_materialize_decoded_dense_rhs",
    ]


def test_real_current_token_block_output_group_stream_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_token_block_output_group_stream_source(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal").read_text(
            encoding="utf-8"
        ),
        primitive_source=Path("native/vq_nax_ext/csrc/nax_fp16_primitive.mm").read_text(
            encoding="utf-8"
        ),
    )

    assert guard["decision"] == (
        "token_block_output_group_stream_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "token_blocks_x_output_groups_x_topk_slots_x_k_blocks_x_codeword_groups"
    )
    assert guard["preserves_token_block_axis"] is True
    assert guard["preserves_output_group_axis"] is True
    assert guard["preserves_topk_slot_axis"] is True
    assert guard["preserves_k_block_axis"] is True
    assert guard["preserves_codeword_group_axis"] is True
    assert guard["uses_token_block_output_group_streams"] is True
    assert guard["preserves_token_topk_output_writeback"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["uses_rejected_token_topk_output_tile_schedule"] is False
    assert guard["uses_rejected_route_tile_output_swizzle_schedule"] is False
    assert guard["uses_rejected_kblock_output_group_route_fused_schedule"] is False
    assert guard["uses_rejected_output_group_pretransposed_schedule"] is False
    assert guard["uses_rejected_route_block_output_group_schedule"] is False
    assert guard["uses_rejected_scale_group_route_block_schedule"] is False
    assert guard["uses_rejected_token_route_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_audit_token_output_stripe_group_stream_source_requires_native_guardrail() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_token_output_stripe_group_stream_source(
        metal_source="[[kernel]] void unrelated_kernel() {}",
        primitive_source="array unrelated() { return array(); }",
    )

    assert guard["decision"] == "missing_token_output_stripe_group_stream_source"
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["dispatch_grid"] == "unknown"
    assert guard["preserves_token_axis"] is False
    assert guard["preserves_output_stripe_axis"] is False
    assert guard["preserves_topk_group_axis"] is False
    assert guard["preserves_k_block_axis"] is False
    assert guard["preserves_codeword_group_axis"] is False
    assert guard["uses_token_output_stripe_group_streams"] is False
    assert guard["preserves_q2_token_topk_output_shape"] is False
    assert guard["streams_compressed_codeword_tiles"] is False
    assert guard["uses_codeword_scale_slots"] is False
    assert guard["uses_scale_tiles"] is False
    assert guard["uses_rejected_token_block_output_group_schedule"] is False
    assert guard["uses_rejected_token_topk_output_tile_schedule"] is False
    assert guard["uses_rejected_route_tile_output_swizzle_schedule"] is False
    assert guard["uses_rejected_kblock_output_group_route_fused_schedule"] is False
    assert guard["uses_rejected_output_group_pretransposed_schedule"] is False
    assert guard["uses_rejected_route_block_output_group_schedule"] is False
    assert guard["uses_rejected_scale_group_route_block_schedule"] is False
    assert guard["uses_rejected_token_route_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert guard["required_next_features"] == [
        "add_token_output_stripe_group_stream_native_source_guardrail",
        "bind_token_output_stripe_group_stream_primitive",
        "preserve_tokens_output_stripes_topk_groups_kblocks_codeword_group_axes",
        "preserve_q2_tokens_x_topk_groups_x_output_stripes_writeback",
        "stream_compressed_uint16_codeword_tiles_and_scale_slots",
        "avoid_token_block_output_group_stream_schedule",
        "avoid_decoded_dense_rhs",
    ]
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_token_output_stripe_group_stream_before_source_guardrail",
        "do_not_retime_token_block_output_group_stream",
        "do_not_materialize_decoded_dense_rhs",
    ]


def test_real_current_token_output_stripe_group_stream_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_token_output_stripe_group_stream_source(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal").read_text(
            encoding="utf-8"
        ),
        primitive_source=Path("native/vq_nax_ext/csrc/nax_fp16_primitive.mm").read_text(
            encoding="utf-8"
        ),
    )

    assert guard["decision"] == (
        "token_output_stripe_group_stream_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "tokens_x_output_stripes_x_topk_groups_x_k_blocks_x_codeword_groups"
    )
    assert guard["preserves_token_axis"] is True
    assert guard["preserves_output_stripe_axis"] is True
    assert guard["preserves_topk_group_axis"] is True
    assert guard["preserves_k_block_axis"] is True
    assert guard["preserves_codeword_group_axis"] is True
    assert guard["uses_token_output_stripe_group_streams"] is True
    assert guard["preserves_q2_token_topk_output_shape"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["uses_rejected_token_block_output_group_schedule"] is False
    assert guard["uses_rejected_token_topk_output_tile_schedule"] is False
    assert guard["uses_rejected_route_tile_output_swizzle_schedule"] is False
    assert guard["uses_rejected_kblock_output_group_route_fused_schedule"] is False
    assert guard["uses_rejected_output_group_pretransposed_schedule"] is False
    assert guard["uses_rejected_route_block_output_group_schedule"] is False
    assert guard["uses_rejected_scale_group_route_block_schedule"] is False
    assert guard["uses_rejected_token_route_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_real_current_kblock_wavefront_codeword_scan_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_kblock_wavefront_codeword_scan_source(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal").read_text(
            encoding="utf-8"
        ),
        primitive_source=Path("native/vq_nax_ext/csrc/nax_fp16_primitive.mm").read_text(
            encoding="utf-8"
        ),
    )

    assert guard["decision"] == (
        "kblock_wavefront_codeword_scan_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "k_blocks_x_route_microtiles_x_output_stripes_x_codeword_groups"
    )
    assert guard["preserves_k_block_axis"] is True
    assert guard["preserves_route_microtile_axis"] is True
    assert guard["preserves_output_stripe_axis"] is True
    assert guard["preserves_codeword_group_axis"] is True
    assert guard["streams_codeword_groups_as_wavefronts"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["materializes_route_microtile_partial_cache"] is False
    assert guard["materializes_output_tile_local_full_lut"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_build_structure_report_surfaces_missing_route_codeword_lut_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    route_lut_design = tmp_path / "route-codeword-lut-design.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[0] = half(0.0);
}
[[kernel]] void unrelated_kernel() {}
""",
        encoding="utf-8",
    )
    primitive.write_text("array unrelated() { return array(); }\n", encoding="utf-8")
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "kernel": (
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_"
                    "bn_64_bk_64_wm_2_wn_2_align_M_t"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps({"projection_summaries": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    route_lut_design.write_text(
        json.dumps(
            {
                "decision": (
                    "route_codeword_lut_accumulate_ready_for_source_structure_probe"
                ),
                "selector_verdict": {"candidate_ready_for_source_probe": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
        route_codeword_lut_accumulate_design_json=route_lut_design,
    )

    guard = report["route_codeword_lut_accumulate_guardrail"]
    assert guard["decision"] == "missing_route_codeword_lut_accumulate_source"
    assert guard["passes_contract"] is False
    assert report["next_track_b_hypothesis"] == (
        "implement_route_codeword_lut_accumulate_source_guardrail"
    )
    assert report["route_codeword_lut_accumulate_design_json"] == str(route_lut_design)
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_build_structure_report_surfaces_missing_rowwise_codeword_tile_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    rowwise_design = tmp_path / "rowwise-design.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[0] = half(0.0);
}
[[kernel]] void unrelated_kernel() {}
""",
        encoding="utf-8",
    )
    primitive.write_text("array unrelated() { return array(); }\n", encoding="utf-8")
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "kernel": (
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_"
                    "bn_64_bk_64_wm_2_wn_2_align_M_t"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps({"projection_summaries": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rowwise_design.write_text(
        json.dumps(
            {
                "decision": (
                    "rowwise_codeword_tile_accumulate_ready_for_source_structure_probe"
                ),
                "selector_verdict": {"candidate_ready_for_source_probe": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
        rowwise_codeword_tile_accumulate_design_json=rowwise_design,
    )

    guard = report["rowwise_codeword_tile_accumulate_guardrail"]
    assert guard["decision"] == "missing_rowwise_codeword_tile_accumulate_source"
    assert guard["passes_contract"] is False
    assert report["next_track_b_hypothesis"] == (
        "implement_rowwise_codeword_tile_accumulate_source_guardrail"
    )
    assert report["rowwise_codeword_tile_accumulate_design_json"] == str(
        rowwise_design
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_build_structure_report_surfaces_missing_output_tile_local_lut_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    output_tile_design = tmp_path / "output-tile-local-design.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[0] = half(0.0);
}
[[kernel]] void unrelated_kernel() {}
""",
        encoding="utf-8",
    )
    primitive.write_text("array unrelated() { return array(); }\n", encoding="utf-8")
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "kernel": (
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_"
                    "bn_64_bk_64_wm_2_wn_2_align_M_t"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps({"projection_summaries": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_tile_design.write_text(
        json.dumps(
            {
                "decision": (
                    "output_tile_local_codeword_lut_ready_for_source_structure_probe"
                ),
                "selector_verdict": {"candidate_ready_for_source_probe": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
        output_tile_local_codeword_lut_design_json=output_tile_design,
    )

    guard = report["output_tile_local_codeword_lut_guardrail"]
    assert guard["decision"] == "missing_output_tile_local_codeword_lut_source"
    assert guard["passes_contract"] is False
    assert report["next_track_b_hypothesis"] == (
        "implement_output_tile_local_codeword_lut_source_guardrail"
    )
    assert report["output_tile_local_codeword_lut_design_json"] == str(
        output_tile_design
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_build_structure_report_surfaces_missing_route_microtile_codeword_block_reduce_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    route_microtile_design = tmp_path / "route-microtile-design.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[0] = half(0.0);
}
[[kernel]] void unrelated_kernel() {}
""",
        encoding="utf-8",
    )
    primitive.write_text("array unrelated() { return array(); }\n", encoding="utf-8")
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "kernel": (
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_"
                    "bn_64_bk_64_wm_2_wn_2_align_M_t"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps({"projection_summaries": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    route_microtile_design.write_text(
        json.dumps(
            {
                "decision": (
                    "route_microtile_codeword_block_reduce_ready_for_source_structure_probe"
                ),
                "selector_verdict": {"candidate_ready_for_source_probe": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
        route_microtile_codeword_block_reduce_design_json=route_microtile_design,
    )

    guard = report["route_microtile_codeword_block_reduce_guardrail"]
    assert guard["decision"] == "missing_route_microtile_codeword_block_reduce_source"
    assert guard["passes_contract"] is False
    assert report["next_track_b_hypothesis"] == (
        "implement_route_microtile_codeword_block_reduce_source_guardrail"
    )
    assert report["route_microtile_codeword_block_reduce_design_json"] == str(
        route_microtile_design
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_build_structure_report_surfaces_missing_kblock_wavefront_codeword_scan_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    kblock_wavefront_design = tmp_path / "kblock-wavefront-design.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[0] = half(0.0);
}
[[kernel]] void unrelated_kernel() {}
""",
        encoding="utf-8",
    )
    primitive.write_text("array unrelated() { return array(); }\n", encoding="utf-8")
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "kernel": (
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_"
                    "bn_64_bk_64_wm_2_wn_2_align_M_t"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps({"projection_summaries": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    kblock_wavefront_design.write_text(
        json.dumps(
            {
                "decision": (
                    "kblock_wavefront_codeword_scan_ready_for_source_structure_probe"
                ),
                "selector_verdict": {"candidate_ready_for_source_probe": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
        kblock_wavefront_codeword_scan_design_json=kblock_wavefront_design,
    )

    guard = report["kblock_wavefront_codeword_scan_guardrail"]
    assert guard["decision"] == "missing_kblock_wavefront_codeword_scan_source"
    assert guard["passes_contract"] is False
    assert report["next_track_b_hypothesis"] == (
        "implement_kblock_wavefront_codeword_scan_source_guardrail"
    )
    assert report["kblock_wavefront_codeword_scan_design_json"] == str(
        kblock_wavefront_design
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_build_structure_report_surfaces_missing_token_route_output_stripe_pipeline_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    token_route_design = tmp_path / "token-route-design.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[0] = half(0.0);
}
[[kernel]] void unrelated_kernel() {}
""",
        encoding="utf-8",
    )
    primitive.write_text("array unrelated() { return array(); }\n", encoding="utf-8")
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "kernel": (
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_"
                    "bn_64_bk_64_wm_2_wn_2_align_M_t"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps({"projection_summaries": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    token_route_design.write_text(
        json.dumps(
            {
                "decision": (
                    "token_route_output_stripe_pipeline_ready_for_source_structure_probe"
                ),
                "selector_verdict": {"candidate_ready_for_source_probe": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
        token_route_output_stripe_pipeline_design_json=token_route_design,
    )

    guard = report["token_route_output_stripe_pipeline_guardrail"]
    assert guard["decision"] == "missing_token_route_output_stripe_pipeline_source"
    assert guard["passes_contract"] is False
    assert report["next_track_b_hypothesis"] == (
        "implement_token_route_output_stripe_pipeline_source_guardrail"
    )
    assert report["token_route_output_stripe_pipeline_design_json"] == str(
        token_route_design
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_build_structure_report_surfaces_missing_expert_kblock_scale_slot_stream_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    expert_kblock_design = tmp_path / "expert-kblock-scale-slot-design.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[0] = half(0.0);
}
[[kernel]] void unrelated_kernel() {}
""",
        encoding="utf-8",
    )
    primitive.write_text("array unrelated() { return array(); }\n", encoding="utf-8")
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "kernel": (
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_"
                    "bn_64_bk_64_wm_2_wn_2_align_M_t"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps({"projection_summaries": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    expert_kblock_design.write_text(
        json.dumps(
            {
                "decision": (
                    "expert_kblock_scale_slot_stream_ready_for_source_structure_probe"
                ),
                "selector_verdict": {"candidate_ready_for_source_probe": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
        expert_kblock_scale_slot_stream_design_json=expert_kblock_design,
    )

    guard = report["expert_kblock_scale_slot_stream_guardrail"]
    assert guard["decision"] == "missing_expert_kblock_scale_slot_stream_source"
    assert guard["passes_contract"] is False
    assert report["next_track_b_hypothesis"] == (
        "implement_expert_kblock_scale_slot_stream_source_guardrail"
    )
    assert report["expert_kblock_scale_slot_stream_design_json"] == str(
        expert_kblock_design
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_build_structure_report_surfaces_missing_scale_group_route_block_reduce_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    scale_group_design = tmp_path / "scale-group-route-block-design.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[0] = half(0.0);
}
[[kernel]] void unrelated_kernel() {}
""",
        encoding="utf-8",
    )
    primitive.write_text("array unrelated() { return array(); }\n", encoding="utf-8")
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "kernel": (
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_"
                    "bn_64_bk_64_wm_2_wn_2_align_M_t"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps({"projection_summaries": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    scale_group_design.write_text(
        json.dumps(
            {
                "decision": (
                    "scale_group_route_block_reduce_ready_for_source_structure_probe"
                ),
                "selector_verdict": {"candidate_ready_for_source_probe": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
        scale_group_route_block_reduce_design_json=scale_group_design,
    )

    guard = report["scale_group_route_block_reduce_guardrail"]
    assert guard["decision"] == "missing_scale_group_route_block_reduce_source"
    assert guard["passes_contract"] is False
    assert report["next_track_b_hypothesis"] == (
        "implement_scale_group_route_block_reduce_source_guardrail"
    )
    assert report["scale_group_route_block_reduce_design_json"] == str(
        scale_group_design
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_build_structure_report_surfaces_missing_route_block_output_group_stream_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    route_block_design = tmp_path / "route-block-output-group-design.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[0] = half(0.0);
}
[[kernel]] void unrelated_kernel() {}
""",
        encoding="utf-8",
    )
    primitive.write_text("array unrelated() { return array(); }\n", encoding="utf-8")
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "kernel": (
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_"
                    "bn_64_bk_64_wm_2_wn_2_align_M_t"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps({"projection_summaries": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    route_block_design.write_text(
        json.dumps(
            {
                "decision": (
                    "route_block_output_group_stream_ready_for_source_structure_probe"
                ),
                "selector_verdict": {"candidate_ready_for_source_probe": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
        route_block_output_group_stream_design_json=route_block_design,
    )

    guard = report["route_block_output_group_stream_guardrail"]
    assert guard["decision"] == "missing_route_block_output_group_stream_source"
    assert guard["passes_contract"] is False
    assert report["next_track_b_hypothesis"] == (
        "implement_route_block_output_group_stream_source_guardrail"
    )
    assert report["route_block_output_group_stream_design_json"] == str(
        route_block_design
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_build_structure_report_surfaces_missing_output_group_pretransposed_codeword_stream_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    output_group_design = tmp_path / "output-group-pretransposed-design.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[0] = half(0.0);
}
[[kernel]] void unrelated_kernel() {}
""",
        encoding="utf-8",
    )
    primitive.write_text("array unrelated() { return array(); }\n", encoding="utf-8")
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "kernel": (
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_"
                    "bn_64_bk_64_wm_2_wn_2_align_M_t"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps({"projection_summaries": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_group_design.write_text(
        json.dumps(
            {
                "decision": (
                    "output_group_pretransposed_codeword_stream_ready_for_source_structure_probe"
                ),
                "selector_verdict": {"candidate_ready_for_source_probe": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
        output_group_pretransposed_codeword_stream_design_json=output_group_design,
    )

    guard = report["output_group_pretransposed_codeword_stream_guardrail"]
    assert (
        guard["decision"]
        == "missing_output_group_pretransposed_codeword_stream_source"
    )
    assert guard["passes_contract"] is False
    assert report["next_track_b_hypothesis"] == (
        "implement_output_group_pretransposed_codeword_stream_source_guardrail"
    )
    assert report["output_group_pretransposed_codeword_stream_design_json"] == str(
        output_group_design
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_build_structure_report_surfaces_missing_kblock_output_group_route_fused_stream_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    kblock_route_fused_design = tmp_path / "kblock-route-fused-design.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[0] = half(0.0);
}
[[kernel]] void unrelated_kernel() {}
""",
        encoding="utf-8",
    )
    primitive.write_text("array unrelated() { return array(); }\n", encoding="utf-8")
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "kernel": (
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_"
                    "bn_64_bk_64_wm_2_wn_2_align_M_t"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps({"projection_summaries": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    kblock_route_fused_design.write_text(
        json.dumps(
            {
                "decision": (
                    "kblock_output_group_route_fused_stream_ready_for_source_structure_probe"
                ),
                "selector_verdict": {"candidate_ready_for_source_probe": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
        kblock_output_group_route_fused_stream_design_json=kblock_route_fused_design,
    )

    guard = report["kblock_output_group_route_fused_stream_guardrail"]
    assert guard["decision"] == "missing_kblock_output_group_route_fused_stream_source"
    assert guard["passes_contract"] is False
    assert report["next_track_b_hypothesis"] == (
        "implement_kblock_output_group_route_fused_stream_source_guardrail"
    )
    assert report["kblock_output_group_route_fused_stream_design_json"] == str(
        kblock_route_fused_design
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_build_structure_report_surfaces_missing_route_tile_output_swizzle_stream_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    route_tile_output_swizzle_design = tmp_path / "route-tile-output-swizzle.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[0] = half(0.0);
}
[[kernel]] void unrelated_kernel() {}
""",
        encoding="utf-8",
    )
    primitive.write_text("array unrelated() { return array(); }\n", encoding="utf-8")
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "kernel": (
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_"
                    "bn_64_bk_64_wm_2_wn_2_align_M_t"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps({"projection_summaries": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    route_tile_output_swizzle_design.write_text(
        json.dumps(
            {
                "decision": (
                    "route_tile_output_swizzle_stream_ready_for_source_structure_probe"
                ),
                "selector_verdict": {"candidate_ready_for_source_probe": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
        route_tile_output_swizzle_stream_design_json=(
            route_tile_output_swizzle_design
        ),
    )

    guard = report["route_tile_output_swizzle_stream_guardrail"]
    assert guard["decision"] == "missing_route_tile_output_swizzle_stream_source"
    assert guard["passes_contract"] is False
    assert report["next_track_b_hypothesis"] == (
        "implement_route_tile_output_swizzle_stream_source_guardrail"
    )
    assert report["route_tile_output_swizzle_stream_design_json"] == str(
        route_tile_output_swizzle_design
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_build_structure_report_surfaces_missing_token_topk_output_tile_stream_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    token_topk_output_tile_design = tmp_path / "token-topk-output-tile.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[0] = half(0.0);
}
[[kernel]] void unrelated_kernel() {}
""",
        encoding="utf-8",
    )
    primitive.write_text("array unrelated() { return array(); }\n", encoding="utf-8")
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "kernel": (
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_"
                    "bn_64_bk_64_wm_2_wn_2_align_M_t"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps({"projection_summaries": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    token_topk_output_tile_design.write_text(
        json.dumps(
            {
                "decision": (
                    "token_topk_output_tile_stream_ready_for_source_structure_probe"
                ),
                "selector_verdict": {"candidate_ready_for_source_probe": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
        token_topk_output_tile_stream_design_json=token_topk_output_tile_design,
    )

    guard = report["token_topk_output_tile_stream_guardrail"]
    assert guard["decision"] == "missing_token_topk_output_tile_stream_source"
    assert guard["passes_contract"] is False
    assert report["next_track_b_hypothesis"] == (
        "implement_token_topk_output_tile_stream_source_guardrail"
    )
    assert report["token_topk_output_tile_stream_design_json"] == str(
        token_topk_output_tile_design
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_build_structure_report_surfaces_missing_token_block_output_group_stream_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    token_block_output_group_design = tmp_path / "token-block-output-group.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[0] = half(0.0);
}
[[kernel]] void unrelated_kernel() {}
""",
        encoding="utf-8",
    )
    primitive.write_text("array unrelated() { return array(); }\n", encoding="utf-8")
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "kernel": (
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_"
                    "bn_64_bk_64_wm_2_wn_2_align_M_t"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps({"projection_summaries": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    token_block_output_group_design.write_text(
        json.dumps(
            {
                "decision": (
                    "token_block_output_group_stream_ready_for_source_structure_probe"
                ),
                "selector_verdict": {"candidate_ready_for_source_probe": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
        token_block_output_group_stream_design_json=token_block_output_group_design,
    )

    guard = report["token_block_output_group_stream_guardrail"]
    assert guard["decision"] == "missing_token_block_output_group_stream_source"
    assert guard["passes_contract"] is False
    assert report["next_track_b_hypothesis"] == (
        "implement_token_block_output_group_stream_source_guardrail"
    )
    assert report["token_block_output_group_stream_design_json"] == str(
        token_block_output_group_design
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_audit_token_expert_output_block_stream_source_requires_native_guardrail() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_token_expert_output_block_stream_source(
        metal_source="[[kernel]] void unrelated_kernel() {}",
        primitive_source="array unrelated() { return array(); }",
    )

    assert guard["decision"] == "missing_token_expert_output_block_stream_source"
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["dispatch_grid"] == "unknown"
    assert guard["preserves_token_axis"] is False
    assert guard["preserves_active_expert_axis"] is False
    assert guard["preserves_output_block_axis"] is False
    assert guard["preserves_k_block_axis"] is False
    assert guard["preserves_codeword_group_axis"] is False
    assert guard["uses_token_expert_output_block_streams"] is False
    assert guard["reduces_routes_before_q2_writeback"] is False
    assert guard["streams_compressed_codeword_tiles"] is False
    assert guard["uses_codeword_scale_slots"] is False
    assert guard["uses_scale_tiles"] is False
    assert guard["uses_rejected_token_output_stripe_group_schedule"] is False
    assert guard["uses_rejected_token_block_output_group_schedule"] is False
    assert guard["uses_rejected_token_topk_output_tile_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert guard["required_next_features"] == [
        "add_token_expert_output_block_stream_native_source_guardrail",
        "bind_token_expert_output_block_stream_primitive",
        "preserve_tokens_active_experts_output_blocks_kblocks_codeword_group_axes",
        "reduce_routes_before_q2_token_topk_output_writeback",
        "stream_compressed_uint16_codeword_tiles_and_scale_slots",
        "avoid_token_output_stripe_group_stream_schedule",
        "avoid_decoded_dense_rhs",
    ]
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_token_expert_output_block_stream_before_source_guardrail",
        "do_not_retime_token_output_stripe_group_stream",
        "do_not_materialize_decoded_dense_rhs",
    ]


def test_real_current_token_expert_output_block_stream_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_token_expert_output_block_stream_source(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal").read_text(
            encoding="utf-8"
        ),
        primitive_source=Path("native/vq_nax_ext/csrc/nax_fp16_primitive.mm").read_text(
            encoding="utf-8"
        ),
    )

    assert guard["decision"] == (
        "token_expert_output_block_stream_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "tokens_x_active_experts_x_output_blocks_x_k_blocks_x_codeword_groups"
    )
    assert guard["preserves_token_axis"] is True
    assert guard["preserves_active_expert_axis"] is True
    assert guard["preserves_output_block_axis"] is True
    assert guard["preserves_k_block_axis"] is True
    assert guard["preserves_codeword_group_axis"] is True
    assert guard["uses_token_expert_output_block_streams"] is True
    assert guard["reduces_routes_before_q2_writeback"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["uses_rejected_token_output_stripe_group_schedule"] is False
    assert guard["uses_rejected_token_block_output_group_schedule"] is False
    assert guard["uses_rejected_token_topk_output_tile_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_audit_token_pair_kblock_accumulator_stream_source_requires_native_guardrail() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_token_pair_kblock_accumulator_stream_source(
        metal_source="[[kernel]] void unrelated_kernel() {}",
        primitive_source="array unrelated() { return array(); }",
    )

    assert guard["decision"] == "missing_token_pair_kblock_accumulator_stream_source"
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["dispatch_grid"] == "unknown"
    assert guard["preserves_token_pair_axis"] is False
    assert guard["preserves_active_expert_axis"] is False
    assert guard["preserves_k_block_axis"] is False
    assert guard["preserves_output_block_axis"] is False
    assert guard["preserves_codeword_group_axis"] is False
    assert guard["uses_token_pair_kblock_accumulator_streams"] is False
    assert guard["reuses_lhs_across_token_pairs"] is False
    assert guard["preserves_exact_token_topk_scatter"] is False
    assert guard["streams_compressed_codeword_tiles"] is False
    assert guard["uses_codeword_scale_slots"] is False
    assert guard["uses_scale_tiles"] is False
    assert guard["uses_rejected_token_expert_output_block_schedule"] is False
    assert guard["uses_rejected_token_output_stripe_group_schedule"] is False
    assert guard["uses_rejected_token_block_output_group_schedule"] is False
    assert guard["uses_rejected_token_topk_output_tile_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert guard["required_next_features"] == [
        "add_token_pair_kblock_accumulator_stream_native_source_guardrail",
        "bind_token_pair_kblock_accumulator_stream_primitive",
        "preserve_token_pairs_active_experts_kblocks_output_blocks_codeword_group_axes",
        "reuse_lhs_across_adjacent_tokens_before_expert_output_scatter",
        "preserve_exact_token_topk_scatter_after_pair_accumulation",
        "stream_compressed_uint16_codeword_tiles_and_scale_slots",
        "avoid_token_expert_output_block_stream_schedule",
        "avoid_decoded_dense_rhs",
    ]
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_token_pair_kblock_accumulator_stream_before_source_guardrail",
        "do_not_retime_token_expert_output_block_stream",
        "do_not_materialize_decoded_dense_rhs",
    ]


def test_real_current_token_pair_kblock_accumulator_stream_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_token_pair_kblock_accumulator_stream_source(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal").read_text(
            encoding="utf-8"
        ),
        primitive_source=Path("native/vq_nax_ext/csrc/nax_fp16_primitive.mm").read_text(
            encoding="utf-8"
        ),
    )

    assert guard["decision"] == (
        "token_pair_kblock_accumulator_stream_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "token_pairs_x_active_experts_x_k_blocks_x_output_blocks_x_codeword_groups"
    )
    assert guard["preserves_token_pair_axis"] is True
    assert guard["preserves_active_expert_axis"] is True
    assert guard["preserves_k_block_axis"] is True
    assert guard["preserves_output_block_axis"] is True
    assert guard["preserves_codeword_group_axis"] is True
    assert guard["uses_token_pair_kblock_accumulator_streams"] is True
    assert guard["reuses_lhs_across_token_pairs"] is True
    assert guard["preserves_exact_token_topk_scatter"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["uses_rejected_token_expert_output_block_schedule"] is False
    assert guard["uses_rejected_token_output_stripe_group_schedule"] is False
    assert guard["uses_rejected_token_block_output_group_schedule"] is False
    assert guard["uses_rejected_token_topk_output_tile_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_audit_token_pair_output_group_stream_source_requires_native_guardrail() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_token_pair_output_group_stream_source(
        metal_source="[[kernel]] void unrelated_kernel() {}",
        primitive_source="array unrelated() { return array(); }",
    )

    assert guard["decision"] == "missing_token_pair_output_group_stream_source"
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["dispatch_grid"] == "unknown"
    assert guard["preserves_token_pair_axis"] is False
    assert guard["preserves_output_group_axis"] is False
    assert guard["preserves_active_expert_axis"] is False
    assert guard["preserves_k_block_axis"] is False
    assert guard["preserves_codeword_group_axis"] is False
    assert guard["uses_token_pair_output_group_streams"] is False
    assert guard["preserves_q2_token_topk_scatter"] is False
    assert guard["avoids_full_pair_expert_matrix"] is True
    assert guard["streams_compressed_codeword_tiles"] is False
    assert guard["uses_codeword_scale_slots"] is False
    assert guard["uses_scale_tiles"] is False
    assert guard["uses_rejected_token_pair_kblock_schedule"] is False
    assert guard["uses_rejected_token_expert_output_block_schedule"] is False
    assert guard["uses_rejected_token_output_stripe_group_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert guard["required_next_features"] == [
        "add_token_pair_output_group_stream_native_source_guardrail",
        "bind_token_pair_output_group_stream_primitive",
        "preserve_token_pairs_output_groups_active_experts_kblocks_codeword_group_axes",
        "preserve_q2_token_topk_scatter_without_full_pair_expert_matrix",
        "stream_compressed_uint16_codeword_tiles_and_scale_slots",
        "avoid_token_pair_kblock_accumulator_stream_schedule",
        "avoid_decoded_dense_rhs",
    ]
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_token_pair_output_group_stream_before_source_guardrail",
        "do_not_retime_token_pair_kblock_accumulator_stream",
        "do_not_materialize_decoded_dense_rhs",
    ]


def test_real_current_token_pair_output_group_stream_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_token_pair_output_group_stream_source(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal").read_text(
            encoding="utf-8"
        ),
        primitive_source=Path("native/vq_nax_ext/csrc/nax_fp16_primitive.mm").read_text(
            encoding="utf-8"
        ),
    )

    assert guard["decision"] == (
        "token_pair_output_group_stream_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "token_pairs_x_output_groups_x_active_experts_x_k_blocks_x_codeword_groups"
    )
    assert guard["preserves_token_pair_axis"] is True
    assert guard["preserves_output_group_axis"] is True
    assert guard["preserves_active_expert_axis"] is True
    assert guard["preserves_k_block_axis"] is True
    assert guard["preserves_codeword_group_axis"] is True
    assert guard["uses_token_pair_output_group_streams"] is True
    assert guard["preserves_q2_token_topk_scatter"] is True
    assert guard["avoids_full_pair_expert_matrix"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["uses_rejected_token_pair_kblock_schedule"] is False
    assert guard["uses_rejected_token_expert_output_block_schedule"] is False
    assert guard["uses_rejected_token_output_stripe_group_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_audit_token_pair_slot_topk_output_group_stream_source_requires_native_guardrail() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_token_pair_slot_topk_output_group_stream_source(
        metal_source="[[kernel]] void unrelated_kernel() {}",
        primitive_source="array unrelated() { return array(); }",
    )

    assert (
        guard["decision"]
        == "missing_token_pair_slot_topk_output_group_stream_source"
    )
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["dispatch_grid"] == "unknown"
    assert guard["preserves_token_pair_axis"] is False
    assert guard["preserves_pair_slot_axis"] is False
    assert guard["preserves_topk_slot_axis"] is False
    assert guard["preserves_output_group_axis"] is False
    assert guard["preserves_k_block_axis"] is False
    assert guard["preserves_codeword_group_axis"] is False
    assert guard["uses_token_pair_slot_topk_output_group_streams"] is False
    assert guard["preserves_q2_token_topk_scatter"] is False
    assert guard["avoids_full_expert_axis"] is True
    assert guard["streams_compressed_codeword_tiles"] is False
    assert guard["uses_codeword_scale_slots"] is False
    assert guard["uses_scale_tiles"] is False
    assert guard["uses_rejected_token_pair_output_group_schedule"] is False
    assert guard["uses_rejected_token_pair_kblock_schedule"] is False
    assert guard["uses_rejected_token_expert_output_block_schedule"] is False
    assert guard["uses_rejected_token_output_stripe_group_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert guard["required_next_features"] == [
        "add_token_pair_slot_topk_output_group_stream_native_source_guardrail",
        "bind_token_pair_slot_topk_output_group_stream_primitive",
        "preserve_token_pairs_pair_slots_topk_slots_output_groups_kblocks_codeword_group_axes",
        "preserve_q2_token_topk_scatter_without_full_expert_axis",
        "stream_compressed_uint16_codeword_tiles_and_scale_slots",
        "avoid_token_pair_output_group_stream_schedule",
        "avoid_token_pair_kblock_accumulator_stream_schedule",
        "avoid_decoded_dense_rhs",
    ]
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_token_pair_slot_topk_output_group_stream_before_source_guardrail",
        "do_not_retime_token_pair_output_group_stream",
        "do_not_materialize_decoded_dense_rhs",
    ]


def test_real_current_token_pair_slot_topk_output_group_stream_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_token_pair_slot_topk_output_group_stream_source(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal").read_text(
            encoding="utf-8"
        ),
        primitive_source=Path("native/vq_nax_ext/csrc/nax_fp16_primitive.mm").read_text(
            encoding="utf-8"
        ),
    )

    assert guard["decision"] == (
        "token_pair_slot_topk_output_group_stream_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "token_pairs_x_pair_slots_x_topk_slots_x_output_groups_x_k_blocks_x_codeword_groups"
    )
    assert guard["preserves_token_pair_axis"] is True
    assert guard["preserves_pair_slot_axis"] is True
    assert guard["preserves_topk_slot_axis"] is True
    assert guard["preserves_output_group_axis"] is True
    assert guard["preserves_k_block_axis"] is True
    assert guard["preserves_codeword_group_axis"] is True
    assert guard["uses_token_pair_slot_topk_output_group_streams"] is True
    assert guard["preserves_q2_token_topk_scatter"] is True
    assert guard["avoids_full_expert_axis"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["uses_rejected_token_pair_output_group_schedule"] is False
    assert guard["uses_rejected_token_pair_kblock_schedule"] is False
    assert guard["uses_rejected_token_expert_output_block_schedule"] is False
    assert guard["uses_rejected_token_output_stripe_group_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_real_current_token_pair_slot_topk_codeword_group_pipeline_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_token_pair_slot_topk_codeword_group_pipeline_source(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal").read_text(
            encoding="utf-8"
        ),
        primitive_source=Path("native/vq_nax_ext/csrc/nax_fp16_primitive.mm").read_text(
            encoding="utf-8"
        ),
    )

    assert guard["decision"] == (
        "token_pair_slot_topk_codeword_group_pipeline_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "token_pairs_x_pair_slots_x_topk_slots_x_codeword_groups_x_output_stripes_x_k_blocks"
    )
    assert guard["preserves_token_pair_axis"] is True
    assert guard["preserves_pair_slot_axis"] is True
    assert guard["preserves_topk_slot_axis"] is True
    assert guard["preserves_codeword_group_axis"] is True
    assert guard["preserves_output_stripe_axis"] is True
    assert guard["preserves_k_block_axis"] is True
    assert guard["uses_token_pair_slot_topk_codeword_group_pipeline"] is True
    assert guard["preserves_q2_token_topk_scatter"] is True
    assert guard["streams_codeword_groups_before_output_stripe"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["uses_rejected_token_pair_slot_topk_output_group_schedule"] is False
    assert guard["uses_rejected_token_pair_output_group_schedule"] is False
    assert guard["uses_rejected_token_pair_kblock_schedule"] is False
    assert guard["uses_rejected_token_expert_output_block_schedule"] is False
    assert guard["uses_rejected_token_output_stripe_group_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_real_current_token_pair_slot_topk_scale_slot_broadcast_stream_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_token_pair_slot_topk_scale_slot_broadcast_stream_source(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal").read_text(
            encoding="utf-8"
        ),
        primitive_source=Path("native/vq_nax_ext/csrc/nax_fp16_primitive.mm").read_text(
            encoding="utf-8"
        ),
    )

    assert guard["decision"] == (
        "token_pair_slot_topk_scale_slot_broadcast_stream_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "token_pairs_x_pair_slots_x_topk_slots_x_scale_slots_x_output_stripes_x_k_blocks_x_codewords"
    )
    assert guard["preserves_token_pair_axis"] is True
    assert guard["preserves_pair_slot_axis"] is True
    assert guard["preserves_topk_slot_axis"] is True
    assert guard["preserves_scale_slot_axis"] is True
    assert guard["preserves_output_stripe_axis"] is True
    assert guard["preserves_k_block_axis"] is True
    assert guard["preserves_codeword_axis"] is True
    assert guard["uses_token_pair_slot_topk_scale_slot_broadcast_stream"] is True
    assert guard["preserves_q2_token_topk_scatter"] is True
    assert guard["broadcasts_scale_slots_before_codewords"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["uses_rejected_token_pair_slot_topk_codeword_group_pipeline"] is False
    assert guard["uses_rejected_token_pair_slot_topk_output_group_schedule"] is False
    assert guard["uses_rejected_token_pair_output_group_schedule"] is False
    assert guard["uses_rejected_token_pair_kblock_schedule"] is False
    assert guard["uses_rejected_token_expert_output_block_schedule"] is False
    assert guard["uses_rejected_token_output_stripe_group_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_real_current_token_pair_slot_topk_route_bucket_codeword_reduce_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()

    guard = (
        analyzer.audit_e8p_token_pair_slot_topk_route_bucket_codeword_reduce_source(
            metal_source=Path(
                "native/vq_nax_ext/kernels/nax_fp16_matmul.metal"
            ).read_text(encoding="utf-8"),
            primitive_source=Path(
                "native/vq_nax_ext/csrc/nax_fp16_primitive.mm"
            ).read_text(encoding="utf-8"),
        )
    )

    assert guard["decision"] == (
        "token_pair_slot_topk_route_bucket_codeword_reduce_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "route_buckets_x_token_pairs_x_pair_slots_x_topk_slots_x_k_blocks_x_output_microtiles_x_codeword_tiles"
    )
    assert guard["preserves_route_bucket_axis"] is True
    assert guard["preserves_token_pair_axis"] is True
    assert guard["preserves_pair_slot_axis"] is True
    assert guard["preserves_topk_slot_axis"] is True
    assert guard["preserves_k_block_axis"] is True
    assert guard["preserves_output_microtile_axis"] is True
    assert guard["preserves_codeword_tile_axis"] is True
    assert guard["uses_token_pair_slot_topk_route_bucket_codeword_reduce"] is True
    assert guard["preserves_route_bucket_offsets"] is True
    assert guard["preserves_q2_token_topk_scatter"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["uses_rejected_token_pair_slot_topk_scale_slot_broadcast_stream"] is False
    assert guard["uses_rejected_token_pair_slot_topk_codeword_group_pipeline"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_real_current_token_pair_slot_topk_kblock_microtile_stream_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_token_pair_slot_topk_kblock_microtile_stream_source(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal").read_text(
            encoding="utf-8"
        ),
        primitive_source=Path(
            "native/vq_nax_ext/csrc/nax_fp16_primitive.mm"
        ).read_text(encoding="utf-8"),
    )

    assert guard["decision"] == (
        "token_pair_slot_topk_kblock_microtile_stream_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "token_pairs_x_pair_slots_x_topk_slots_x_k_blocks_x_output_microtiles"
    )
    assert guard["preserves_token_pair_axis"] is True
    assert guard["preserves_pair_slot_axis"] is True
    assert guard["preserves_topk_slot_axis"] is True
    assert guard["preserves_k_block_axis"] is True
    assert guard["preserves_output_microtile_axis"] is True
    assert guard["uses_token_pair_slot_topk_kblock_microtile_stream"] is True
    assert guard["preserves_q2_token_topk_scatter"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["streams_codeword_tiles_inside_threadgroup"] is True
    assert guard["avoids_route_slot_expansion"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["uses_rejected_token_pair_slot_topk_route_bucket_codeword_reduce"] is False
    assert guard["uses_rejected_token_pair_slot_topk_scale_slot_broadcast_stream"] is False
    assert guard["uses_rejected_token_pair_slot_topk_codeword_group_pipeline"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_audit_token_pair_slot_topk_codeword_group_pipeline_source_requires_native_guardrail() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_token_pair_slot_topk_codeword_group_pipeline_source(
        metal_source="[[kernel]] void unrelated_kernel() {}",
        primitive_source="array unrelated() { return array(); }",
    )

    assert (
        guard["decision"]
        == "missing_token_pair_slot_topk_codeword_group_pipeline_source"
    )
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["dispatch_grid"] == "unknown"
    assert guard["preserves_token_pair_axis"] is False
    assert guard["preserves_pair_slot_axis"] is False
    assert guard["preserves_topk_slot_axis"] is False
    assert guard["preserves_codeword_group_axis"] is False
    assert guard["preserves_output_stripe_axis"] is False
    assert guard["preserves_k_block_axis"] is False
    assert guard["uses_token_pair_slot_topk_codeword_group_pipeline"] is False
    assert guard["preserves_q2_token_topk_scatter"] is False
    assert guard["streams_codeword_groups_before_output_stripe"] is False
    assert guard["streams_compressed_codeword_tiles"] is False
    assert guard["uses_codeword_scale_slots"] is False
    assert guard["uses_scale_tiles"] is False
    assert guard["uses_rejected_token_pair_slot_topk_output_group_schedule"] is False
    assert guard["uses_rejected_token_pair_output_group_schedule"] is False
    assert guard["uses_rejected_token_pair_kblock_schedule"] is False
    assert guard["uses_rejected_token_expert_output_block_schedule"] is False
    assert guard["uses_rejected_token_output_stripe_group_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert guard["required_next_features"] == [
        "add_token_pair_slot_topk_codeword_group_pipeline_native_source_guardrail",
        "bind_token_pair_slot_topk_codeword_group_pipeline_primitive",
        "preserve_token_pairs_pair_slots_topk_slots_codeword_groups_output_stripes_kblocks_axes",
        "stream_codeword_groups_before_output_stripe_writeback",
        "preserve_q2_token_topk_scatter_without_output_group_inner_loop",
        "stream_compressed_uint16_codeword_tiles_and_scale_slots",
        "avoid_token_pair_slot_topk_output_group_stream_schedule",
        "avoid_decoded_dense_rhs",
    ]
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_token_pair_slot_topk_codeword_group_pipeline_before_source_guardrail",
        "do_not_retime_token_pair_slot_topk_output_group_stream",
        "do_not_materialize_decoded_dense_rhs",
    ]


def test_audit_token_pair_slot_topk_scale_slot_broadcast_stream_source_requires_native_guardrail() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_token_pair_slot_topk_scale_slot_broadcast_stream_source(
        metal_source="[[kernel]] void unrelated_kernel() {}",
        primitive_source="array unrelated() { return array(); }",
    )

    assert (
        guard["decision"]
        == "missing_token_pair_slot_topk_scale_slot_broadcast_stream_source"
    )
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["dispatch_grid"] == "unknown"
    assert guard["preserves_token_pair_axis"] is False
    assert guard["preserves_pair_slot_axis"] is False
    assert guard["preserves_topk_slot_axis"] is False
    assert guard["preserves_scale_slot_axis"] is False
    assert guard["preserves_output_stripe_axis"] is False
    assert guard["preserves_k_block_axis"] is False
    assert guard["uses_token_pair_slot_topk_scale_slot_broadcast_stream"] is False
    assert guard["preserves_q2_token_topk_scatter"] is False
    assert guard["broadcasts_scale_slots_before_codewords"] is False
    assert guard["streams_compressed_codeword_tiles"] is False
    assert guard["uses_codeword_scale_slots"] is False
    assert guard["uses_scale_tiles"] is False
    assert guard["uses_rejected_token_pair_slot_topk_codeword_group_pipeline"] is False
    assert guard["uses_rejected_token_pair_slot_topk_output_group_schedule"] is False
    assert guard["uses_rejected_token_pair_output_group_schedule"] is False
    assert guard["uses_rejected_token_pair_kblock_schedule"] is False
    assert guard["uses_rejected_token_expert_output_block_schedule"] is False
    assert guard["uses_rejected_token_output_stripe_group_schedule"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert guard["required_next_features"] == [
        "add_token_pair_slot_topk_scale_slot_broadcast_stream_native_source_guardrail",
        "bind_token_pair_slot_topk_scale_slot_broadcast_stream_primitive",
        "preserve_token_pairs_pair_slots_topk_slots_scale_slots_output_stripes_kblocks_codeword_axes",
        "broadcast_scale_slots_before_codeword_streaming",
        "preserve_q2_token_topk_scatter_without_codeword_group_pipeline",
        "stream_compressed_uint16_codeword_tiles_and_scale_slots",
        "avoid_token_pair_slot_topk_codeword_group_pipeline_schedule",
        "avoid_decoded_dense_rhs",
    ]
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_token_pair_slot_topk_scale_slot_broadcast_stream_before_source_guardrail",
        "do_not_retime_token_pair_slot_topk_codeword_group_pipeline",
        "do_not_materialize_decoded_dense_rhs",
    ]


def test_audit_token_pair_slot_topk_route_bucket_codeword_reduce_source_requires_native_guardrail() -> None:
    analyzer = _load_analyzer()

    guard = (
        analyzer.audit_e8p_token_pair_slot_topk_route_bucket_codeword_reduce_source(
            metal_source="[[kernel]] void unrelated_kernel() {}",
            primitive_source="array unrelated() { return array(); }",
        )
    )

    assert (
        guard["decision"]
        == "missing_token_pair_slot_topk_route_bucket_codeword_reduce_source"
    )
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["dispatch_grid"] == "unknown"
    assert guard["preserves_route_bucket_axis"] is False
    assert guard["preserves_token_pair_axis"] is False
    assert guard["preserves_pair_slot_axis"] is False
    assert guard["preserves_topk_slot_axis"] is False
    assert guard["preserves_k_block_axis"] is False
    assert guard["preserves_output_microtile_axis"] is False
    assert guard["preserves_codeword_tile_axis"] is False
    assert guard["uses_token_pair_slot_topk_route_bucket_codeword_reduce"] is False
    assert guard["preserves_route_bucket_offsets"] is False
    assert guard["preserves_q2_token_topk_scatter"] is False
    assert guard["streams_compressed_codeword_tiles"] is False
    assert guard["uses_codeword_scale_slots"] is False
    assert guard["uses_scale_tiles"] is False
    assert guard["uses_rejected_token_pair_slot_topk_scale_slot_broadcast_stream"] is False
    assert guard["uses_rejected_token_pair_slot_topk_codeword_group_pipeline"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert guard["required_next_features"] == [
        "add_token_pair_slot_topk_route_bucket_codeword_reduce_native_source_guardrail",
        "bind_token_pair_slot_topk_route_bucket_codeword_reduce_primitive",
        "preserve_route_buckets_token_pairs_pair_slots_topk_slots_kblocks_output_microtiles_codeword_tiles_axes",
        "preserve_route_bucket_offsets_for_route_slot_exactness",
        "preserve_q2_token_topk_scatter_without_scale_slot_broadcast_stream",
        "stream_compressed_uint16_codeword_tiles_and_scale_slots",
        "avoid_token_pair_slot_topk_scale_slot_broadcast_stream_schedule",
        "avoid_decoded_dense_rhs",
    ]
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_token_pair_slot_topk_route_bucket_codeword_reduce_before_source_guardrail",
        "do_not_retime_token_pair_slot_topk_scale_slot_broadcast_stream",
        "do_not_materialize_decoded_dense_rhs",
    ]


def test_audit_token_pair_slot_topk_kblock_microtile_stream_source_requires_native_guardrail() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_token_pair_slot_topk_kblock_microtile_stream_source(
        metal_source="[[kernel]] void unrelated_kernel() {}",
        primitive_source="array unrelated() { return array(); }",
    )

    assert (
        guard["decision"]
        == "missing_token_pair_slot_topk_kblock_microtile_stream_source"
    )
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["dispatch_grid"] == "unknown"
    assert guard["preserves_token_pair_axis"] is False
    assert guard["preserves_pair_slot_axis"] is False
    assert guard["preserves_topk_slot_axis"] is False
    assert guard["preserves_k_block_axis"] is False
    assert guard["preserves_output_microtile_axis"] is False
    assert guard["uses_token_pair_slot_topk_kblock_microtile_stream"] is False
    assert guard["preserves_q2_token_topk_scatter"] is False
    assert guard["streams_compressed_codeword_tiles"] is False
    assert guard["streams_codeword_tiles_inside_threadgroup"] is False
    assert guard["avoids_route_slot_expansion"] is False
    assert guard["uses_codeword_scale_slots"] is False
    assert guard["uses_scale_tiles"] is False
    assert guard["uses_rejected_token_pair_slot_topk_route_bucket_codeword_reduce"] is False
    assert guard["uses_rejected_token_pair_slot_topk_scale_slot_broadcast_stream"] is False
    assert guard["uses_rejected_token_pair_slot_topk_codeword_group_pipeline"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert guard["required_next_features"] == [
        "add_token_pair_slot_topk_kblock_microtile_stream_native_source_guardrail",
        "bind_token_pair_slot_topk_kblock_microtile_stream_primitive",
        "preserve_token_pairs_pair_slots_topk_slots_kblocks_output_microtiles_axes",
        "avoid_route_bucket_route_slot_expansion",
        "stream_compressed_uint16_codeword_tiles_inside_kblock_microtile",
        "preserve_q2_token_topk_scatter_without_route_bucket_codeword_reduce",
        "avoid_token_pair_slot_topk_route_bucket_codeword_reduce_schedule",
        "avoid_decoded_dense_rhs",
    ]
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_token_pair_slot_topk_kblock_microtile_stream_before_source_guardrail",
        "do_not_retime_token_pair_slot_topk_route_bucket_codeword_reduce",
        "do_not_materialize_decoded_dense_rhs",
    ]


def test_audit_token_pair_slot_topk_kblock_microtile_stream_source_accepts_contract() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_token_pair_slot_topk_kblock_microtile_stream_source(
        metal_source="""
[[kernel]] void nax_e8p_token_pair_slot_topk_kblock_microtile_stream_rhs_sorted_matmul() {
  device const uint16_t* codeword_tiles;
  device const half* codeword_scale_slots;
  device const half* scale_tiles;
  uint token_pair = 0;
  uint pair_slot = 0;
  uint topk_slot = 0;
  uint k_block = 0;
  uint output_microtile = 0;
  uint pair_slot_topk_scatter = token_pair + pair_slot + topk_slot;
  uint no_route_slot_expansion = pair_slot_topk_scatter + k_block;
  uint kblock_microtile_stream = no_route_slot_expansion + output_microtile;
  float codeword_tiles_inside_threadgroup = float(
      codeword_tiles[kblock_microtile_stream]);
  codeword_tiles_inside_threadgroup += float(codeword_scale_slots[k_block]);
  codeword_tiles_inside_threadgroup += float(scale_tiles[output_microtile]);
}
""",
        primitive_source="""
array e8p_token_pair_slot_topk_kblock_microtile_stream_rhs_sorted_matmul() {
  auto* pso = cache.get("nax_e8p_token_pair_slot_topk_kblock_microtile_stream_rhs_sorted_matmul");
  // token_pairs_x_pair_slots_x_topk_slots_x_k_blocks_x_output_microtiles
  encoder.dispatch_threadgroups(
      MTL::Size::Make(token_pairs, pair_slots * topk_slots, k_blocks),
      MTL::Size::Make(1, 1, 1));
  return array();
}
""",
    )

    assert guard["decision"] == (
        "token_pair_slot_topk_kblock_microtile_stream_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "token_pairs_x_pair_slots_x_topk_slots_x_k_blocks_x_output_microtiles"
    )
    assert guard["preserves_token_pair_axis"] is True
    assert guard["preserves_pair_slot_axis"] is True
    assert guard["preserves_topk_slot_axis"] is True
    assert guard["preserves_k_block_axis"] is True
    assert guard["preserves_output_microtile_axis"] is True
    assert guard["uses_token_pair_slot_topk_kblock_microtile_stream"] is True
    assert guard["preserves_q2_token_topk_scatter"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["streams_codeword_tiles_inside_threadgroup"] is True
    assert guard["avoids_route_slot_expansion"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["uses_rejected_token_pair_slot_topk_route_bucket_codeword_reduce"] is False
    assert guard["uses_rejected_token_pair_slot_topk_scale_slot_broadcast_stream"] is False
    assert guard["uses_rejected_token_pair_slot_topk_codeword_group_pipeline"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_build_structure_report_surfaces_missing_token_output_stripe_group_stream_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    token_output_stripe_design = tmp_path / "token-output-stripe-group.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[0] = half(0.0);
}
[[kernel]] void unrelated_kernel() {}
""",
        encoding="utf-8",
    )
    primitive.write_text("array unrelated() { return array(); }\n", encoding="utf-8")
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "kernel": (
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_"
                    "bn_64_bk_64_wm_2_wn_2_align_M_t"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps({"projection_summaries": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    token_output_stripe_design.write_text(
        json.dumps(
            {
                "decision": (
                    "token_output_stripe_group_stream_ready_for_source_structure_probe"
                ),
                "selector_verdict": {"candidate_ready_for_source_probe": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
        token_output_stripe_group_stream_design_json=token_output_stripe_design,
    )

    guard = report["token_output_stripe_group_stream_guardrail"]
    assert guard["decision"] == "missing_token_output_stripe_group_stream_source"
    assert guard["passes_contract"] is False
    assert report["next_track_b_hypothesis"] == (
        "implement_token_output_stripe_group_stream_source_guardrail"
    )
    assert report["token_output_stripe_group_stream_design_json"] == str(
        token_output_stripe_design
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_build_structure_report_surfaces_missing_token_expert_output_block_stream_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    token_expert_design = tmp_path / "token-expert-output-block.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[0] = half(0.0);
}
[[kernel]] void unrelated_kernel() {}
""",
        encoding="utf-8",
    )
    primitive.write_text("array unrelated() { return array(); }\n", encoding="utf-8")
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "kernel": (
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_"
                    "bn_64_bk_64_wm_2_wn_2_align_M_t"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps({"projection_summaries": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    token_expert_design.write_text(
        json.dumps(
            {
                "decision": (
                    "token_expert_output_block_stream_ready_for_source_structure_probe"
                ),
                "selector_verdict": {"candidate_ready_for_source_probe": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
        token_expert_output_block_stream_design_json=token_expert_design,
    )

    guard = report["token_expert_output_block_stream_guardrail"]
    assert guard["decision"] == "missing_token_expert_output_block_stream_source"
    assert guard["passes_contract"] is False
    assert report["next_track_b_hypothesis"] == (
        "implement_token_expert_output_block_stream_source_guardrail"
    )
    assert report["token_expert_output_block_stream_design_json"] == str(
        token_expert_design
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_build_structure_report_surfaces_missing_token_pair_kblock_accumulator_stream_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    token_pair_design = tmp_path / "token-pair-kblock-accumulator.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[0] = half(0.0);
}
[[kernel]] void unrelated_kernel() {}
""",
        encoding="utf-8",
    )
    primitive.write_text("array unrelated() { return array(); }\n", encoding="utf-8")
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "kernel": (
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_"
                    "bn_64_bk_64_wm_2_wn_2_align_M_t"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps({"projection_summaries": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    token_pair_design.write_text(
        json.dumps(
            {
                "decision": (
                    "token_pair_kblock_accumulator_stream_ready_for_source_structure_probe"
                ),
                "selector_verdict": {"candidate_ready_for_source_probe": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
        token_pair_kblock_accumulator_stream_design_json=token_pair_design,
    )

    guard = report["token_pair_kblock_accumulator_stream_guardrail"]
    assert guard["decision"] == "missing_token_pair_kblock_accumulator_stream_source"
    assert guard["passes_contract"] is False
    assert report["next_track_b_hypothesis"] == (
        "implement_token_pair_kblock_accumulator_stream_source_guardrail"
    )
    assert report["token_pair_kblock_accumulator_stream_design_json"] == str(
        token_pair_design
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_build_structure_report_surfaces_missing_token_pair_output_group_stream_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    token_pair_output_group_design = tmp_path / "token-pair-output-group.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[0] = half(0.0);
}
[[kernel]] void unrelated_kernel() {}
""",
        encoding="utf-8",
    )
    primitive.write_text("array unrelated() { return array(); }\n", encoding="utf-8")
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "kernel": (
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_"
                    "bn_64_bk_64_wm_2_wn_2_align_M_t"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps({"projection_summaries": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    token_pair_output_group_design.write_text(
        json.dumps(
            {
                "decision": (
                    "token_pair_output_group_stream_ready_for_source_structure_probe"
                ),
                "selector_verdict": {"candidate_ready_for_source_probe": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
        token_pair_output_group_stream_design_json=token_pair_output_group_design,
    )

    guard = report["token_pair_output_group_stream_guardrail"]
    assert guard["decision"] == "missing_token_pair_output_group_stream_source"
    assert guard["passes_contract"] is False
    assert report["next_track_b_hypothesis"] == (
        "implement_token_pair_output_group_stream_source_guardrail"
    )
    assert report["token_pair_output_group_stream_design_json"] == str(
        token_pair_output_group_design
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_build_structure_report_surfaces_missing_token_pair_slot_topk_output_group_stream_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    slot_topk_design = tmp_path / "token-pair-slot-topk-output-group.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[0] = half(0.0);
}
[[kernel]] void unrelated_kernel() {}
""",
        encoding="utf-8",
    )
    primitive.write_text("array unrelated() { return array(); }\n", encoding="utf-8")
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "kernel": (
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_"
                    "bn_64_bk_64_wm_2_wn_2_align_M_t"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps({"projection_summaries": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    slot_topk_design.write_text(
        json.dumps(
            {
                "decision": (
                    "token_pair_slot_topk_output_group_stream_ready_for_source_structure_probe"
                ),
                "selector_verdict": {"candidate_ready_for_source_probe": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
        token_pair_slot_topk_output_group_stream_design_json=slot_topk_design,
    )

    guard = report["token_pair_slot_topk_output_group_stream_guardrail"]
    assert (
        guard["decision"]
        == "missing_token_pair_slot_topk_output_group_stream_source"
    )
    assert guard["passes_contract"] is False
    assert report["next_track_b_hypothesis"] == (
        "implement_token_pair_slot_topk_output_group_stream_source_guardrail"
    )
    assert report["token_pair_slot_topk_output_group_stream_design_json"] == str(
        slot_topk_design
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_build_structure_report_surfaces_missing_token_pair_slot_topk_codeword_group_pipeline_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    codeword_group_design = (
        tmp_path / "token-pair-slot-topk-codeword-group-pipeline.json"
    )
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[0] = half(0.0);
}
[[kernel]] void unrelated_kernel() {}
""",
        encoding="utf-8",
    )
    primitive.write_text("array unrelated() { return array(); }\n", encoding="utf-8")
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "kernel": (
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_"
                    "bn_64_bk_64_wm_2_wn_2_align_M_t"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps({"projection_summaries": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    codeword_group_design.write_text(
        json.dumps(
            {
                "decision": (
                    "token_pair_slot_topk_codeword_group_pipeline_ready_for_source_structure_probe"
                ),
                "selector_verdict": {"candidate_ready_for_source_probe": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
        token_pair_slot_topk_codeword_group_pipeline_design_json=(
            codeword_group_design
        ),
    )

    guard = report["token_pair_slot_topk_codeword_group_pipeline_guardrail"]
    assert guard["decision"] == (
        "missing_token_pair_slot_topk_codeword_group_pipeline_source"
    )
    assert guard["passes_contract"] is False
    assert report["next_track_b_hypothesis"] == (
        "implement_token_pair_slot_topk_codeword_group_pipeline_source_guardrail"
    )
    assert report[
        "token_pair_slot_topk_codeword_group_pipeline_design_json"
    ] == str(codeword_group_design)
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_build_structure_report_surfaces_missing_token_pair_slot_topk_scale_slot_broadcast_stream_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    scale_slot_design = tmp_path / "token-pair-slot-topk-scale-slot-broadcast.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[0] = half(0.0);
}
[[kernel]] void unrelated_kernel() {}
""",
        encoding="utf-8",
    )
    primitive.write_text("array unrelated() { return array(); }\n", encoding="utf-8")
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "kernel": (
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_"
                    "bn_64_bk_64_wm_2_wn_2_align_M_t"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps({"projection_summaries": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    scale_slot_design.write_text(
        json.dumps(
            {
                "decision": (
                    "token_pair_slot_topk_scale_slot_broadcast_stream_ready_for_source_structure_probe"
                ),
                "selector_verdict": {"candidate_ready_for_source_probe": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
        token_pair_slot_topk_scale_slot_broadcast_stream_design_json=(
            scale_slot_design
        ),
    )

    guard = report["token_pair_slot_topk_scale_slot_broadcast_stream_guardrail"]
    assert guard["decision"] == (
        "missing_token_pair_slot_topk_scale_slot_broadcast_stream_source"
    )
    assert guard["passes_contract"] is False
    assert report["next_track_b_hypothesis"] == (
        "implement_token_pair_slot_topk_scale_slot_broadcast_stream_source_guardrail"
    )
    assert report[
        "token_pair_slot_topk_scale_slot_broadcast_stream_design_json"
    ] == str(scale_slot_design)
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_build_structure_report_surfaces_missing_token_pair_slot_topk_route_bucket_codeword_reduce_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    route_bucket_design = (
        tmp_path / "token-pair-slot-topk-route-bucket-codeword-reduce.json"
    )
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[0] = half(0.0);
}
[[kernel]] void unrelated_kernel() {}
""",
        encoding="utf-8",
    )
    primitive.write_text("array unrelated() { return array(); }\n", encoding="utf-8")
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "kernel": (
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_"
                    "bn_64_bk_64_wm_2_wn_2_align_M_t"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps({"projection_summaries": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    route_bucket_design.write_text(
        json.dumps(
            {
                "decision": (
                    "token_pair_slot_topk_route_bucket_codeword_reduce_ready_for_source_structure_probe"
                ),
                "selector_verdict": {"candidate_ready_for_source_probe": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
        token_pair_slot_topk_route_bucket_codeword_reduce_design_json=(
            route_bucket_design
        ),
    )

    guard = report["token_pair_slot_topk_route_bucket_codeword_reduce_guardrail"]
    assert guard["decision"] == (
        "missing_token_pair_slot_topk_route_bucket_codeword_reduce_source"
    )
    assert guard["passes_contract"] is False
    assert report["next_track_b_hypothesis"] == (
        "implement_token_pair_slot_topk_route_bucket_codeword_reduce_source_guardrail"
    )
    assert report[
        "token_pair_slot_topk_route_bucket_codeword_reduce_design_json"
    ] == str(route_bucket_design)
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_build_structure_report_surfaces_missing_token_pair_slot_topk_kblock_microtile_stream_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    kblock_microtile_design = (
        tmp_path / "token-pair-slot-topk-kblock-microtile-stream.json"
    )
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[0] = half(0.0);
}
[[kernel]] void unrelated_kernel() {}
""",
        encoding="utf-8",
    )
    primitive.write_text("array unrelated() { return array(); }\n", encoding="utf-8")
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "kernel": (
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_"
                    "bn_64_bk_64_wm_2_wn_2_align_M_t"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps({"projection_summaries": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    kblock_microtile_design.write_text(
        json.dumps(
            {
                "decision": (
                    "token_pair_slot_topk_kblock_microtile_stream_ready_for_source_structure_probe"
                ),
                "selector_verdict": {"candidate_ready_for_source_probe": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
        token_pair_slot_topk_kblock_microtile_stream_design_json=(
            kblock_microtile_design
        ),
    )

    guard = report["token_pair_slot_topk_kblock_microtile_stream_guardrail"]
    assert guard["decision"] == (
        "missing_token_pair_slot_topk_kblock_microtile_stream_source"
    )
    assert guard["passes_contract"] is False
    assert report["next_track_b_hypothesis"] == (
        "implement_token_pair_slot_topk_kblock_microtile_stream_source_guardrail"
    )
    assert report[
        "token_pair_slot_topk_kblock_microtile_stream_design_json"
    ] == str(kblock_microtile_design)
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_audit_token_pair_slot_topk_output_tile_fused_stream_source_requires_native_guardrail() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_token_pair_slot_topk_output_tile_fused_stream_source(
        metal_source="""
[[kernel]] void unrelated_kernel() {
  uint16_t codeword_tiles = 0;
}
""",
        primitive_source="array unrelated() { return array(); }\n",
    )

    assert guard["decision"] == (
        "missing_token_pair_slot_topk_output_tile_fused_stream_source"
    )
    assert guard["passes_contract"] is False
    assert guard["kernel_present"] is False
    assert guard["primitive_binding_present"] is False
    assert guard["dispatch_grid"] == "unknown"
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False
    assert (
        "add_token_pair_slot_topk_output_tile_fused_stream_native_source_guardrail"
        in guard["required_next_features"]
    )
    assert (
        "do_not_benchmark_token_pair_slot_topk_output_tile_fused_stream_before_source_guardrail"
        in guard["rejected_next_steps"]
    )


def test_audit_token_pair_slot_topk_output_tile_fused_stream_source_accepts_contract() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_token_pair_slot_topk_output_tile_fused_stream_source(
        metal_source="""
[[kernel]] void nax_e8p_token_pair_slot_topk_output_tile_fused_stream_rhs_sorted_matmul() {
  uint token_pair = 0;
  uint pair_slot = 0;
  uint topk_slot = 0;
  uint output_tile = 0;
  uint16_t codeword_tiles = 0;
  uint codeword_scale_slots = 0;
  uint scale_tiles = 0;
  // token_pair_slot_topk_output_tile_fused_stream q2_scatter
  // output_tile_fused_workgroup streams_kblocks_and_codewords_inside_output_tile_fused_workgroup
}
""",
        primitive_source="""
array e8p_token_pair_slot_topk_output_tile_fused_stream_rhs_sorted_matmul() {
  auto* pso = cache.get("nax_e8p_token_pair_slot_topk_output_tile_fused_stream_rhs_sorted_matmul");
  // token_pairs_x_pair_slots_x_topk_slots_x_output_tiles
  return array();
}
""",
    )

    assert guard["decision"] == (
        "token_pair_slot_topk_output_tile_fused_stream_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "token_pairs_x_pair_slots_x_topk_slots_x_output_tiles"
    )
    assert guard["preserves_token_pair_axis"] is True
    assert guard["preserves_pair_slot_axis"] is True
    assert guard["preserves_topk_slot_axis"] is True
    assert guard["preserves_output_tile_axis"] is True
    assert guard["uses_token_pair_slot_topk_output_tile_fused_stream"] is True
    assert guard["preserves_q2_token_topk_scatter"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["streams_kblocks_and_codewords_inside_output_tile"] is True
    assert guard["uses_rejected_token_pair_slot_topk_kblock_microtile_stream"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_real_current_token_pair_slot_topk_output_tile_fused_stream_source_guardrail_is_present() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_token_pair_slot_topk_output_tile_fused_stream_source(
        metal_source=Path("native/vq_nax_ext/kernels/nax_fp16_matmul.metal").read_text(
            encoding="utf-8"
        ),
        primitive_source=Path(
            "native/vq_nax_ext/csrc/nax_fp16_primitive.mm"
        ).read_text(encoding="utf-8"),
    )

    assert guard["decision"] == (
        "token_pair_slot_topk_output_tile_fused_stream_source_guardrail_present"
    )
    assert guard["passes_contract"] is True
    assert guard["kernel_present"] is True
    assert guard["primitive_binding_present"] is True
    assert guard["dispatch_grid"] == (
        "token_pairs_x_pair_slots_x_topk_slots_x_output_tiles"
    )
    assert guard["preserves_token_pair_axis"] is True
    assert guard["preserves_pair_slot_axis"] is True
    assert guard["preserves_topk_slot_axis"] is True
    assert guard["preserves_output_tile_axis"] is True
    assert guard["uses_token_pair_slot_topk_output_tile_fused_stream"] is True
    assert guard["preserves_q2_token_topk_scatter"] is True
    assert guard["streams_compressed_codeword_tiles"] is True
    assert guard["streams_kblocks_and_codewords_inside_output_tile"] is True
    assert guard["uses_codeword_scale_slots"] is True
    assert guard["uses_scale_tiles"] is True
    assert guard["uses_rejected_token_pair_slot_topk_kblock_microtile_stream"] is False
    assert guard["uses_rejected_token_pair_slot_topk_route_bucket_codeword_reduce"] is False
    assert guard["uses_rejected_token_pair_slot_topk_scale_slot_broadcast_stream"] is False
    assert guard["uses_rejected_token_pair_slot_topk_codeword_group_pipeline"] is False
    assert guard["materializes_decoded_dense_rhs"] is False
    assert guard["stages_decoded_b"] is False
    assert guard["uses_lane_local_fragment_buffer"] is False
    assert guard["reconstructs_e8p_per_fragment"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_build_structure_report_surfaces_missing_token_pair_slot_topk_output_tile_fused_stream_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    output_tile_fused_design = (
        tmp_path / "token-pair-slot-topk-output-tile-fused-stream.json"
    )
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[0] = half(0.0);
}
[[kernel]] void unrelated_kernel() {}
""",
        encoding="utf-8",
    )
    primitive.write_text("array unrelated() { return array(); }\n", encoding="utf-8")
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "kernel": (
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_"
                    "bn_64_bk_64_wm_2_wn_2_align_M_t"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps({"projection_summaries": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_tile_fused_design.write_text(
        json.dumps(
            {
                "decision": (
                    "token_pair_slot_topk_output_tile_fused_stream_ready_for_source_structure_probe"
                ),
                "selector_verdict": {"candidate_ready_for_source_probe": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
        token_pair_slot_topk_output_tile_fused_stream_design_json=(
            output_tile_fused_design
        ),
    )

    guard = report["token_pair_slot_topk_output_tile_fused_stream_guardrail"]
    assert guard["decision"] == (
        "missing_token_pair_slot_topk_output_tile_fused_stream_source"
    )
    assert guard["passes_contract"] is False
    assert report["next_track_b_hypothesis"] == (
        "implement_token_pair_slot_topk_output_tile_fused_stream_source_guardrail"
    )
    assert report[
        "token_pair_slot_topk_output_tile_fused_stream_design_json"
    ] == str(output_tile_fused_design)
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False
    assert guard["speed_claim"] is False
    assert guard["native_parity_claim"] is False


def test_build_structure_report_surfaces_missing_route_batch_segmented_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    route_batch_design = tmp_path / "route-batch-design.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[0] = half(0.0);
}
[[kernel]] void unrelated_kernel() {}
""",
        encoding="utf-8",
    )
    primitive.write_text("array unrelated() { return array(); }\n", encoding="utf-8")
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "kernel": (
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_"
                    "bn_64_bk_64_wm_2_wn_2_align_M_t"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps({"projection_summaries": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    route_batch_design.write_text(
        json.dumps(
            {
                "decision": (
                    "route_batch_segmented_codeword_reduce_ready_for_source_structure_probe"
                ),
                "selector_verdict": {"candidate_ready_for_source_probe": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
        route_batch_segmented_design_json=route_batch_design,
    )

    guard = report["route_batch_segmented_codeword_reduce_guardrail"]
    assert guard["decision"] == "missing_route_batch_segmented_codeword_reduce_source"
    assert guard["passes_contract"] is False
    assert report["next_track_b_hypothesis"] == (
        "implement_route_batch_segmented_codeword_reduce_source_guardrail"
    )
    assert report["route_batch_segmented_design_json"] == str(route_batch_design)
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_build_structure_report_surfaces_missing_token_cohort_guardrail(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    token_cohort_design = tmp_path / "token-cohort-design.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[0] = half(0.0);
}
[[kernel]] void unrelated_kernel() {}
""",
        encoding="utf-8",
    )
    primitive.write_text("array unrelated() { return array(); }\n", encoding="utf-8")
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "kernel": (
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_"
                    "bn_64_bk_64_wm_2_wn_2_align_M_t"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps({"projection_summaries": {}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    token_cohort_design.write_text(
        json.dumps(
            {
                "decision": (
                    "token_cohort_codeword_stream_ready_for_source_structure_probe"
                ),
                "selector_verdict": {"candidate_ready_for_source_probe": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
        token_cohort_design_json=token_cohort_design,
    )

    guard = report["token_cohort_codeword_stream_guardrail"]
    assert guard["decision"] == "missing_token_cohort_codeword_stream_source"
    assert guard["passes_contract"] is False
    assert report["next_track_b_hypothesis"] == (
        "implement_token_cohort_codeword_stream_source_guardrail"
    )
    assert report["token_cohort_design_json"] == str(token_cohort_design)
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_build_structure_report_prioritizes_component_stream_partial_design(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "route-batch-speed.json"
    component_design = tmp_path / "component-stream-design.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(Ws + kk);
}
[[kernel]] void nax_e8p_component_stream_rhs_sorted_partial_matmul() {
  device float* component_stream_partials = partials;
  uint sign_bit = uint(sign_component_bits[component_base + component_offset]);
  uint abs_index = uint(abs_index_tiles[codeword_base + codeword_index]);
  int scale_slot = component_scale_slots[k_block * component_count + component_offset];
  int codeword_slot = codeword_scale_slots[k_block * codewords + codeword_index];
  float scale_f = float(scale_tiles[scale_base + scale_slot + codeword_slot]);
  float component = mlx_vq_decode_e8p_component(sign_bit, abs_index, codebook, component_offset);
  float partial = float(sorted_x[route * input_dims + k]) * component * scale_f;
  component_stream_partials[partial_offset] = partial;
}
[[kernel]] void nax_e8p_component_stream_rhs_sorted_partial_reduce() {
  device const float* component_stream_partials = partials;
  float accum = 0.0f;
  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint component_pair = 0; component_pair < component_pairs; ++component_pair) {
      accum += component_stream_partials[partial_offset];
    }
  }
  out[route * output_dims + n] = half(accum);
}
""",
        encoding="utf-8",
    )
    primitive.write_text(
        """
array e8p_component_stream_rhs_sorted_partial_matmul() {
  auto component_stream_partials = array(
      {num_route_tiles, k_blocks, component_pairs, n_tiles, route_tile_size, 64},
      float32);
  component_stream_partials.set_data(allocator::malloc(component_stream_partials.nbytes()));
  auto* partial_pso = cache.get("nax_e8p_component_stream_rhs_sorted_partial_matmul");
  encoder.dispatch_threadgroups(
      MTL::Size::Make(num_route_tiles, component_pairs, k_blocks),
      MTL::Size::Make(64, 1, 1));
  auto* reduce_pso = cache.get("nax_e8p_component_stream_rhs_sorted_partial_reduce");
  encoder.dispatch_threadgroups(
      MTL::Size::Make(num_route_tiles, n_tiles, 1),
      MTL::Size::Make(64, 1, 1));
}
""",
        encoding="utf-8",
    )
    q2.write_text(json.dumps({"projection": "down"}) + "\n", encoding="utf-8")
    e8p_analysis.write_text(
        json.dumps(
            {
                "decision": "reject_route_batch_segmented_codeword_reduce_speed_path",
                "same_window_q2_speed_packet": True,
                "all_lane_s_pass": False,
                "all_parity_pass": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    component_design.write_text(
        json.dumps(
            {
                "decision": (
                    "component_stream_partial_reduction_ready_for_source_structure_probe"
                ),
                "selector_verdict": {"candidate_ready_for_source_probe": True},
                "peer2_used": False,
                "rdma_jaccl_touched": False,
                "speed_claim": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
        component_stream_partial_reduction_design_json=component_design,
    )

    guard = report["component_stream_partial_reduction_guardrail"]
    assert report["component_stream_partial_reduction_design_json"] == str(
        component_design
    )
    assert guard["decision"] == "component_stream_partial_reduction_scaffold_scalar_body"
    assert guard["passes_contract"] is True
    assert report["next_track_b_hypothesis"] == (
        "replace_scalar_component_stream_partial_body_with_valid_parallel_schedule"
    )
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False


def test_audit_component_stream_speed_path_rejects_scalar_oracle_as_benchmarkable() -> None:
    analyzer = _load_analyzer()
    metal_source = """
[[kernel]] void nax_e8p_component_stream_rhs_sorted_scalar_matmul() {
  uint sign_bit = uint(sign_component_bits[component_base + component_offset]);
  uint abs_index = uint(abs_index_tiles[codeword_base + codeword_index]);
  int scale_slot = component_scale_slots[k_block * component_count + component_offset];
  int codeword_slot = codeword_scale_slots[k_block * codewords + codeword_index];
  float scale_f = float(scale_tiles[scale_base + scale_slot + codeword_slot]);
  float component = mlx_vq_decode_e8p_component(sign_bit, abs_index, codebook, component_offset);
  accum += float(sorted_x[route * input_dims + k]) * component * scale_f;
  out[route * output_dims + n] = half(accum);
}
[[kernel]] void other_kernel() {}
"""
    primitive_source = """
array e8p_component_stream_rhs_sorted_scalar_matmul() {
  auto* pso = cache.get("nax_e8p_component_stream_rhs_sorted_scalar_matmul");
  encoder.dispatch_threadgroups(
      MTL::Size::Make((output_dims + 15u) / 16u, (route_count + 15u) / 16u, 1),
      MTL::Size::Make(16, 16, 1));
}
"""

    guard = analyzer.audit_e8p_component_stream_speed_path_source(
        metal_source=metal_source,
        primitive_source=primitive_source,
    )

    assert guard["decision"] == "component_stream_scalar_oracle_speed_path_missing"
    assert guard["passes_contract"] is False
    assert guard["scalar_oracle_present"] is True
    assert guard["tensorops_speed_kernel_present"] is False
    assert guard["speed_claim"] is False
    assert guard["required_next_features"] == [
        "design_component_stream_tensorops_or_parallel_speed_path",
        "preserve_compressed_component_stream_storage",
        "prove_speed_path_against_decoded_sorted_reference",
        "run_same_window_q2_speed_packet",
    ]
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_component_stream_scalar_oracle_as_speed_path",
        "do_not_route_resident_auto_before_same_window_q2_speed_gate",
    ]


def test_audit_component_stream_partial_reduction_requires_two_pass_scaffold() -> None:
    analyzer = _load_analyzer()

    guard = analyzer.audit_e8p_component_stream_partial_reduction_source(
        metal_source="[[kernel]] void nax_e8p_component_stream_rhs_sorted_scalar_matmul() {}",
        primitive_source="array e8p_component_stream_rhs_sorted_scalar_matmul() { return array(); }",
    )

    assert guard["decision"] == "missing_component_stream_partial_reduction_scaffold"
    assert guard["passes_contract"] is False
    assert guard["partial_accumulation_scratch_present"] is False
    assert guard["partial_write_kernel_present"] is False
    assert guard["final_reduction_kernel_present"] is False
    assert guard["required_next_features"] == [
        "add_component_stream_partial_accumulation_scratch",
        "dispatch_component_pair_partial_writer",
        "dispatch_component_stream_partial_reduce",
        "preserve_compressed_component_stream_storage",
    ]
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_component_stream_scalar_oracle_as_speed_path",
        "do_not_materialize_decoded_dense_rhs",
    ]


def test_audit_component_stream_partial_reduction_accepts_two_pass_scaffold() -> None:
    analyzer = _load_analyzer()
    metal_source = """
[[kernel]] void nax_e8p_component_stream_rhs_sorted_partial_matmul() {
  device float* component_stream_partials = partials;
  uint sign_bit = uint(sign_component_bits[component_base + component_offset]);
  uint abs_index = uint(abs_index_tiles[codeword_base + codeword_index]);
  int scale_slot = component_scale_slots[k_block * component_count + component_offset];
  int codeword_slot = codeword_scale_slots[k_block * codewords + codeword_index];
  float scale_f = float(scale_tiles[scale_base + scale_slot + codeword_slot]);
  float component = mlx_vq_decode_e8p_component(sign_bit, abs_index, codebook, component_offset);
  float partial = float(sorted_x[route * input_dims + k]) * component * scale_f;
  component_stream_partials[partial_offset] = partial;
}
[[kernel]] void nax_e8p_component_stream_rhs_sorted_partial_reduce() {
  device const float* component_stream_partials = partials;
  float accum = 0.0f;
  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint component_pair = 0; component_pair < component_pairs; ++component_pair) {
      accum += component_stream_partials[partial_offset];
    }
  }
  out[route * output_dims + n] = half(accum);
}
"""
    primitive_source = """
array e8p_component_stream_rhs_sorted_partial_matmul() {
  auto component_stream_partials = array(
      {num_route_tiles, k_blocks, component_pairs, n_tiles, route_tile_size, 64},
      float32);
  component_stream_partials.set_data(allocator::malloc(component_stream_partials.nbytes()));
  auto* partial_pso = cache.get("nax_e8p_component_stream_rhs_sorted_partial_matmul");
  encoder.dispatch_threadgroups(
      MTL::Size::Make(num_route_tiles, component_pairs, k_blocks),
      MTL::Size::Make(64, 1, 1));
  auto* reduce_pso = cache.get("nax_e8p_component_stream_rhs_sorted_partial_reduce");
  encoder.dispatch_threadgroups(
      MTL::Size::Make(num_route_tiles, n_tiles, 1),
      MTL::Size::Make(64, 1, 1));
}
"""

    guard = analyzer.audit_e8p_component_stream_partial_reduction_source(
        metal_source=metal_source,
        primitive_source=primitive_source,
    )

    assert guard["decision"] == "component_stream_partial_reduction_scaffold_scalar_body"
    assert guard["passes_contract"] is True
    assert guard["partial_accumulation_scratch_present"] is True
    assert guard["partial_write_kernel_present"] is True
    assert guard["final_reduction_kernel_present"] is True
    assert guard["partial_body_parallel_schedule_present"] is False
    assert guard["scratch_dtype"] == "float32"
    assert guard["partial_grid"] == "route_tiles_x_component_pairs_x_k_blocks"
    assert guard["reduction_grid"] == "route_tiles_x_n_tiles"
    assert guard["required_next_features"] == [
        "replace_scalar_component_stream_partial_body_with_valid_parallel_schedule",
        "preserve_component_stream_partial_reduction_scaffold",
        "preserve_compressed_component_stream_storage",
        "prove_native_component_stream_partial_reduction_parity_after_schedule_change",
    ]


def test_audit_component_stream_partial_reduction_rejects_scalar_body_for_speed() -> None:
    analyzer = _load_analyzer()
    metal_source = """
[[kernel]] void nax_e8p_component_stream_rhs_sorted_partial_matmul() {
  device float* component_stream_partials = partials;
  uint sign_bit = uint(sign_component_bits[component_base + component_offset]);
  uint abs_index = uint(abs_index_tiles[codeword_base + codeword_index]);
  int scale_slot = component_scale_slots[k_block * component_count + component_offset];
  int codeword_slot = codeword_scale_slots[k_block * codewords + codeword_index];
  float scale_f = float(scale_tiles[scale_base + scale_slot + codeword_slot]);
  float component = mlx_vq_decode_e8p_component(sign_bit, abs_index, codebook, component_offset);
  float partial = float(sorted_x[route * input_dims + k]) * component * scale_f;
  component_stream_partials[partial_offset] = partial;
}
[[kernel]] void nax_e8p_component_stream_rhs_sorted_partial_reduce() {
  device const float* component_stream_partials = partials;
  float accum = 0.0f;
  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint component_pair = 0; component_pair < component_pairs; ++component_pair) {
      accum += component_stream_partials[partial_offset];
    }
  }
  out[route * output_dims + n] = half(accum);
}
"""
    primitive_source = """
array e8p_component_stream_rhs_sorted_partial_matmul() {
  auto component_stream_partials = array(
      {num_route_tiles, k_blocks, component_pairs, n_tiles, route_tile_size, 64},
      float32);
  component_stream_partials.set_data(allocator::malloc(component_stream_partials.nbytes()));
  auto* partial_pso = cache.get("nax_e8p_component_stream_rhs_sorted_partial_matmul");
  encoder.dispatch_threadgroups(
      MTL::Size::Make(num_route_tiles, component_pairs, k_blocks),
      MTL::Size::Make(64, 1, 1));
  auto* reduce_pso = cache.get("nax_e8p_component_stream_rhs_sorted_partial_reduce");
  encoder.dispatch_threadgroups(
      MTL::Size::Make(num_route_tiles, n_tiles, 1),
      MTL::Size::Make(64, 1, 1));
}
"""

    guard = analyzer.audit_e8p_component_stream_partial_reduction_source(
        metal_source=metal_source,
        primitive_source=primitive_source,
    )

    assert guard["decision"] == (
        "component_stream_partial_reduction_scaffold_scalar_body"
    )
    assert guard["passes_contract"] is True
    assert guard["partial_body_parallel_schedule_present"] is False
    assert guard["speed_claim"] is False
    assert "do_not_benchmark_scalar_component_stream_partial_scaffold" in guard[
        "rejected_next_steps"
    ]


def test_audit_component_stream_partial_reduction_recognizes_tensorops_body_before_benchmark() -> None:
    analyzer = _load_analyzer()
    metal_source = """
[[kernel]] void nax_e8p_component_stream_rhs_sorted_partial_matmul() {
  device float* component_stream_partials = partials;
  uint sign_bit = uint(sign_component_bits[component_base + component_offset]);
  uint abs_index = uint(abs_index_tiles[codeword_base + codeword_index]);
  int scale_slot = component_scale_slots[k_block * component_count + component_offset];
  int codeword_slot = codeword_scale_slots[k_block * codewords + codeword_index];
  int scale_group = scale_group_indices[k_block * scale_groups + scale_slot];
  float scale_f = float(scale_tiles[scale_base + scale_slot + codeword_slot]);
  half x = sorted_x[route * K + k + scale_group];
  auto a_t = matmul_op.get_left_input_cooperative_tensor<half, half, float>();
  auto b_t = matmul_op.get_right_input_cooperative_tensor<half, half, float>();
  simdgroup_matrix_storage<float, 8, 8> acc;
  mpp::tensor_ops::matmul2d<8, 8, 8, half, float> matmul_op;
  matmul_op.run(a_t, b_t, acc);
  component_stream_partials[partial_offset] =
      float(acc.thread_elements()[0]) * float(x) * scale_f;
}
[[kernel]] void nax_e8p_component_stream_rhs_sorted_partial_reduce() {
  device const float* component_stream_partials = partials;
  float accum = 0.0f;
  for (uint k_block = 0; k_block < k_blocks; ++k_block) {
    for (uint component_pair = 0; component_pair < component_pairs; ++component_pair) {
      accum += component_stream_partials[partial_offset];
    }
  }
  out[route * output_dims + n] = half(accum);
}
"""
    primitive_source = """
array e8p_component_stream_rhs_sorted_partial_matmul() {
  auto component_stream_partials = array(
      {num_route_tiles, k_blocks, component_pairs, n_tiles, route_tile_size, 64},
      float32);
  component_stream_partials.set_data(allocator::malloc(component_stream_partials.nbytes()));
  auto* partial_pso = cache.get("nax_e8p_component_stream_rhs_sorted_partial_matmul");
  encoder.dispatch_threadgroups(
      MTL::Size::Make(num_route_tiles, component_pairs, k_blocks),
      MTL::Size::Make(64, 1, 1));
  auto* reduce_pso = cache.get("nax_e8p_component_stream_rhs_sorted_partial_reduce");
  encoder.dispatch_threadgroups(
      MTL::Size::Make(num_route_tiles, n_tiles, 1),
      MTL::Size::Make(64, 1, 1));
}
"""

    guard = analyzer.audit_e8p_component_stream_partial_reduction_source(
        metal_source=metal_source,
        primitive_source=primitive_source,
    )

    assert guard["decision"] == (
        "component_stream_partial_reduction_tensorops_body_present"
    )
    assert guard["passes_contract"] is True
    assert guard["partial_body_parallel_schedule_present"] is True
    assert guard["partial_body_tensorops_present"] is True
    assert guard["speed_claim"] is False
    assert guard["required_next_features"] == [
        "prove_native_component_stream_tensorops_partial_parity",
        "prove_artifact_component_stream_tensorops_partial_parity",
        "run_same_window_q2_speed_packet_after_parity",
    ]
    assert guard["rejected_next_steps"] == [
        "do_not_benchmark_before_native_and_artifact_parity",
        "do_not_promote_without_same_window_q2_speed_packet",
    ]


def test_real_current_component_stream_partial_reduction_uses_parallel_body() -> None:
    analyzer = _load_analyzer()
    repo = Path(__file__).resolve().parents[1]
    metal = repo / "native" / "vq_nax_ext" / "kernels" / "nax_fp16_matmul.metal"
    primitive = repo / "native" / "vq_nax_ext" / "csrc" / "nax_fp16_primitive.mm"

    guard = analyzer.audit_e8p_component_stream_partial_reduction_source(
        metal_source=metal.read_text(encoding="utf-8"),
        primitive_source=primitive.read_text(encoding="utf-8"),
    )

    assert guard["decision"] == (
        "component_stream_partial_reduction_parallel_body_present"
    )
    assert guard["passes_contract"] is True
    assert guard["partial_body_parallel_schedule_present"] is True
    assert guard["partial_body_tensorops_present"] is False
    assert guard["component_pair_parallel_reduction_present"] is True
    assert guard["speed_claim"] is False


def test_build_structure_report_carries_component_stream_source_guardrail(tmp_path) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
  Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(Ws + kk);
}
""",
        encoding="utf-8",
    )
    primitive.write_text("array unrelated() { return array(); }\n", encoding="utf-8")
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "tokens": 4096,
                "route_count": 32768,
                "kernel_names": [
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2"
                ],
                "passes_bk64_rhs_nax": True,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text(
        json.dumps({"all_lane_s_pass": False, "all_parity_pass": False}),
        encoding="utf-8",
    )

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
    )

    guard = report["component_stream_source_guardrail"]
    assert guard["decision"] == "missing_component_stream_native_source_oracle"
    assert guard["passes_contract"] is False
    assert guard["required_next_features"] == [
        "add_component_stream_native_scalar_accumulator",
        "bind_component_stream_sorted_route_primitive",
        "preserve_component_scale_slots",
        "prove_air_down_group_size_352_parity",
    ]


def test_build_structure_report_carries_component_stream_speed_path_guardrail(
    tmp_path,
) -> None:
    analyzer = _load_analyzer()
    metal = tmp_path / "nax_fp16_matmul.metal"
    primitive = tmp_path / "nax_fp16_primitive.mm"
    q2 = tmp_path / "q2.jsonl"
    e8p_analysis = tmp_path / "e8p-analysis.json"
    metal.write_text(
        """
[[kernel]] void nax_e8p_fp16_sorted_matmul_steel() {
  threadgroup half Ws[mlx_vq_nax_bn_tile * mlx_vq_nax_bk_padded];
  Ws[base] = half(mlx_vq_decode_e8p_value(code, codebook, 0u) * scale_f);
  Btile.template load<half, int(mlx_vq_nax_bk_padded), 1>(Ws + kk);
}
[[kernel]] void nax_e8p_component_stream_rhs_sorted_scalar_matmul() {
  uint sign_bit = uint(sign_component_bits[component_base + component_offset]);
  uint abs_index = uint(abs_index_tiles[codeword_base + codeword_index]);
  int scale_slot = component_scale_slots[k_block * component_count + component_offset];
  int codeword_slot = codeword_scale_slots[k_block * codewords + codeword_index];
  float scale_f = float(scale_tiles[scale_base + scale_slot + codeword_slot]);
  float component = mlx_vq_decode_e8p_component(sign_bit, abs_index, codebook, component_offset);
  accum += float(sorted_x[route * input_dims + k]) * component * scale_f;
  out[route * output_dims + n] = half(accum);
}
""",
        encoding="utf-8",
    )
    primitive.write_text(
        """
array e8p_component_stream_rhs_sorted_scalar_matmul() {
  auto* pso = cache.get("nax_e8p_component_stream_rhs_sorted_scalar_matmul");
  encoder.dispatch_threadgroups(
      MTL::Size::Make((output_dims + 15u) / 16u, (route_count + 15u) / 16u, 1),
      MTL::Size::Make(16, 16, 1));
}
""",
        encoding="utf-8",
    )
    q2.write_text(
        json.dumps(
            {
                "projection": "down",
                "tokens": 4096,
                "route_count": 32768,
                "kernel_names": [
                    "affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2"
                ],
                "passes_bk64_rhs_nax": True,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    e8p_analysis.write_text("{}", encoding="utf-8")

    report = analyzer.build_structure_report(
        metal_source=metal,
        primitive_source=primitive,
        q2_jsonl=q2,
        e8p_analysis_json=e8p_analysis,
    )

    guard = report["component_stream_speed_path_guardrail"]
    assert guard["decision"] == "component_stream_scalar_oracle_speed_path_missing"
    assert guard["passes_contract"] is False
    assert guard["scalar_oracle_present"] is True
    assert report["component_stream_partial_reduction_guardrail"]["decision"] == (
        "missing_component_stream_partial_reduction_scaffold"
    )
    assert report["next_track_b_hypothesis"] == (
        "implement_component_stream_partial_reduction_scaffold_or_change_kernel_family"
    )
