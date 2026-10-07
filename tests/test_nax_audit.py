from __future__ import annotations

import json
import plistlib

from mlx_vq.benchmark.nax_audit import (
    Q2NaxRuntimeCase,
    q2_nax_capture_plan_records,
    q2_nax_runtime_cases,
    run_sorted_q2_case,
    summarize_gputrace,
    summarize_xctrace,
    mlx_metallib_nax_inventory,
)
import mlx_vq.benchmark.nax_audit as nax_audit


def test_mlx_metallib_nax_inventory_reports_expected_kernel_families() -> None:
    inventory = mlx_metallib_nax_inventory()

    assert inventory["mlx_metallib_path"].endswith("mlx.metallib")
    assert inventory["gather_qmm_nax_count"] > 0
    assert inventory["qmm_nax_count"] > 0
    assert inventory["steel_gemm_fused_nax_count"] > 0
    assert 128 in inventory["gather_qmm_nax_group_sizes"]
    assert inventory["has_gather_qmm_nax_bk32"] is False
    assert inventory["has_gather_qmm_nax_bk64"] is True
    assert json.dumps(inventory, sort_keys=True)


def test_q2_nax_runtime_cases_cover_air_prefill_targets() -> None:
    cases = q2_nax_runtime_cases()

    assert {(case.projection, case.tokens) for case in cases} == {
        ("gate_up", 1024),
        ("gate_up", 2048),
        ("gate_up", 4096),
        ("down", 1024),
        ("down", 2048),
        ("down", 4096),
    }
    gate = next(case for case in cases if case.projection == "gate_up" and case.tokens == 1024)
    down = next(case for case in cases if case.projection == "down" and case.tokens == 4096)
    assert (gate.input_dims, gate.output_dims) == (4096, 1408)
    assert (down.input_dims, down.output_dims) == (1408, 4096)
    assert gate.route_count == 8192
    assert down.route_count == 32768
    assert any("gather_qmm_rhs_nax" in pattern for pattern in gate.allowed_kernel_regexes)
    assert any("gather_qmm_t_nax" in pattern for pattern in gate.allowed_kernel_regexes)
    assert all("bk_?64" in pattern for pattern in gate.allowed_kernel_regexes)


def test_q2_nax_capture_plan_records_include_capture_env_and_commands() -> None:
    records = q2_nax_capture_plan_records(tokens=(1024,), projections=("gate_up",), trace_dir="tmp/traces")

    assert len(records) == 1
    record = records[0]
    assert record["trace_path"].endswith("tmp/traces/mlx-q2-nax-gate_up-m1024.gputrace")
    assert record["capture_env"] == {"MTL_CAPTURE_ENABLED": "1"}
    assert "MTL_CAPTURE_ENABLED=1" in record["capture_command"]
    assert "--capture --projection gate_up --tokens 1024" in record["capture_command"]
    assert "bk64" in record["manual_xcode_check"]


def test_sorted_q2_runtime_case_uses_packed_quantized_tensors() -> None:
    record = run_sorted_q2_case(
        Q2NaxRuntimeCase(
            projection="gate_up",
            tokens=4,
            input_dims=4096,
            output_dims=1408,
        )
    )

    assert record["runtime_call"] == "mx.gather_qmm(sorted_indices=True)"
    assert record["finite_output"] is True
    assert record["packed_weight_shape"] == [128, 1408, 256]
    assert record["scales_shape"] == [128, 1408, 32]
    assert record["biases_shape"] == [128, 1408, 32]
    assert record["output_shape"] == [32, 1, 1408]


def test_summarize_gputrace_reports_allowed_nax_symbols(tmp_path) -> None:
    trace = tmp_path / "sample.gputrace"
    trace.mkdir()
    (trace / "metadata").write_bytes(
        plistlib.dumps({"DYCaptureEngine.captured_frames_count": 1, "DYCaptureSession.unusedComputePipelineStateCount": 0})
    )
    (trace / "MTLBuffer-0-0").write_bytes(b"skip me")
    (trace / "metallib-resource").write_text(
        "\n".join(
            [
                "affine_gather_qmm_rhs_nax_nt_bfloat16_t_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2",
                "affine_gather_qmm_rhs_nax_nt_bfloat16_t_gs_128_b_2_bm_64_bn_64_bk_32_wm_2_wn_2",
            ]
        )
    )

    summary = summarize_gputrace(
        trace,
        allowed_kernel_regexes=[r"affine_gather_qmm_rhs_nax_.*_gs_128_b_2_.*bk_?64"],
    )

    assert summary["metadata"]["DYCaptureEngine.captured_frames_count"] == 1
    assert summary["trace_nax_symbol_count"] == 2
    assert summary["has_allowed_gather_qmm_nax_bk64_symbol"] is True
    assert summary["has_gather_qmm_nax_bk64_symbol"] is True
    assert summary["has_gather_qmm_nax_bk32_symbol"] is True
    assert any(file["skip_reason"] == "buffer" for file in summary["scanned_files"] if not file["scanned"])


def test_summarize_xctrace_reports_shader_list_symbols(tmp_path, monkeypatch) -> None:
    trace = tmp_path / "sample.trace"
    trace.mkdir()

    class Completed:
        returncode = 0
        stdout = (
            '<row><metal-object-label fmt="'
            'affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2_align_M_t">'
            "kernel</metal-object-label></row>"
        )
        stderr = ""

    monkeypatch.setattr(nax_audit.subprocess, "run", lambda *args, **kwargs: Completed())

    summary = summarize_xctrace(
        trace,
        allowed_kernel_regexes=[r"affine_gather_qmm_rhs_nax_.*_gs_128_b_2_.*bk_?64"],
    )

    assert summary["trace_summary_source"] == "xctrace_export_shader_list"
    assert summary["xctrace_export_returncode"] == 0
    assert summary["has_allowed_gather_qmm_nax_bk64_symbol"] is True
    assert summary["has_gather_qmm_nax_bk64_symbol"] is True
    assert summary["has_gather_qmm_nax_bk32_symbol"] is False
