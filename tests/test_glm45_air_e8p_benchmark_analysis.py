from __future__ import annotations

import json
import importlib.util
from pathlib import Path


def _load_analyzer():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "analyze_glm45_air_e8p_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_glm45_air_e8p_benchmark", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_analyze_e8p_benchmarks_reports_projection_ratios_and_bottleneck(tmp_path) -> None:
    analyzer = _load_analyzer()
    e8p_path = tmp_path / "e8p.jsonl"
    q2_path = tmp_path / "q2.jsonl"
    _write_jsonl(
        e8p_path,
        [
            {
                "variant": "nax_e8p_fp16_sorted_steel_raw",
                "tokens": 4096,
                "projection": "gate_up",
                "ms_per_iter": 13.0,
                "pageouts_delta": 0,
                "swapouts_delta": 0,
                "finite_output": True,
            },
            {
                "variant": "nax_e8p_fp16_sorted_steel_raw",
                "tokens": 4096,
                "projection": "down",
                "ms_per_iter": 16.0,
                "pageouts_delta": 0,
                "swapouts_delta": 0,
                "finite_output": True,
            },
        ],
    )
    _write_jsonl(
        q2_path,
        [
            {
                "variant": "mlx_q2",
                "tokens": 4096,
                "projection": "gate_up",
                "ms_per_iter": 10.0,
                "pageouts_delta": 0,
                "swapouts_delta": 0,
                "finite_output": True,
            },
            {
                "variant": "mlx_q2",
                "tokens": 4096,
                "projection": "down",
                "ms_per_iter": 12.0,
                "pageouts_delta": 0,
                "swapouts_delta": 0,
                "finite_output": True,
            },
        ],
    )

    summary = analyzer.analyze_benchmark_files(
        e8p_files=[e8p_path],
        q2_files=[q2_path],
        parity_ratio=1.25,
        lane_s_ratio=1.15,
    )

    assert summary["record_type"] == "glm45_air_e8p_benchmark_analysis"
    assert summary["candidate_row_count"] == 2
    assert summary["q2_row_count"] == 2
    assert summary["clean_candidate_row_count"] == 2
    assert summary["worst_ratio"]["projection"] == "down"
    assert summary["worst_ratio"]["ratio_to_q2"] == 16.0 / 12.0
    assert summary["worst_ratio"]["parity_pass"] is False
    assert summary["closest_ratio"]["projection"] == "gate_up"
    assert summary["projection_summaries"]["gate_up"]["best_ratio_to_q2"] == 1.3
    assert summary["projection_summaries"]["down"]["best_ratio_to_q2"] == 16.0 / 12.0


def test_analyze_e8p_benchmarks_compares_candidate_to_artifact_baseline(tmp_path) -> None:
    analyzer = _load_analyzer()
    e8p_path = tmp_path / "e8p.jsonl"
    q2_path = tmp_path / "q2.jsonl"
    _write_jsonl(
        e8p_path,
        [
            {
                "variant": "nax_e8p_fp16_sorted_steel_raw",
                "artifact_projection": "gate_proj",
                "tokens": 4096,
                "projection": "gate_up",
                "ms_per_iter": 6.0,
                "pageouts_delta": 0,
                "swapouts_delta": 0,
                "finite_output": True,
            },
            {
                "variant": "nax_e8p_packed_rhs_sorted_tiled_raw",
                "artifact_projection": "gate_proj",
                "tokens": 4096,
                "projection": "gate_up",
                "ms_per_iter": 7.5,
                "pageouts_delta": 0,
                "swapouts_delta": 0,
                "finite_output": True,
            },
            {
                "variant": "nax_e8p_fp16_sorted_steel_raw",
                "artifact_projection": "up_proj",
                "tokens": 4096,
                "projection": "gate_up",
                "ms_per_iter": 8.0,
                "pageouts_delta": 0,
                "swapouts_delta": 0,
                "finite_output": True,
            },
            {
                "variant": "nax_e8p_packed_rhs_sorted_tiled_raw",
                "artifact_projection": "up_proj",
                "tokens": 4096,
                "projection": "gate_up",
                "ms_per_iter": 7.0,
                "pageouts_delta": 0,
                "swapouts_delta": 0,
                "finite_output": True,
            },
        ],
    )
    _write_jsonl(
        q2_path,
        [
            {
                "variant": "mlx_q2",
                "tokens": 4096,
                "projection": "gate_up",
                "ms_per_iter": 4.0,
                "pageouts_delta": 0,
                "swapouts_delta": 0,
                "finite_output": True,
            },
        ],
    )

    summary = analyzer.analyze_benchmark_files(
        e8p_files=[e8p_path],
        q2_files=[q2_path],
        baseline_variant="nax_e8p_fp16_sorted_steel_raw",
    )

    comparisons = {
        row["artifact_projection"]: row for row in summary["baseline_comparisons"]
    }
    assert comparisons["gate_proj"]["candidate_variant"] == "nax_e8p_packed_rhs_sorted_tiled_raw"
    assert comparisons["gate_proj"]["ratio_to_baseline"] == 7.5 / 6.0
    assert comparisons["gate_proj"]["beats_baseline"] is False
    assert comparisons["up_proj"]["ratio_to_baseline"] == 7.0 / 8.0
    assert comparisons["up_proj"]["beats_baseline"] is True


def test_analyze_e8p_benchmarks_excludes_q2_rows_from_candidate_file(tmp_path) -> None:
    analyzer = _load_analyzer()
    combined_path = tmp_path / "combined.jsonl"
    _write_jsonl(
        combined_path,
        [
            {
                "variant": "nax_e8p_packed_rhs_sorted_tiled_raw",
                "tokens": 1024,
                "projection": "down",
                "ms_per_iter": 7.0,
                "pageouts_delta": 0,
                "swapouts_delta": 0,
                "finite_output": True,
            },
            {
                "variant": "mlx_q2",
                "tokens": 1024,
                "projection": "down",
                "ms_per_iter": 3.5,
                "pageouts_delta": 0,
                "swapouts_delta": 0,
                "finite_output": True,
            },
        ],
    )

    summary = analyzer.analyze_benchmark_files(
        e8p_files=[combined_path],
        q2_files=[combined_path],
    )

    assert summary["candidate_row_count"] == 1
    assert summary["q2_row_count"] == 1
    assert summary["comparison_count"] == 1
    assert summary["comparisons"][0]["candidate_variant"] == "nax_e8p_packed_rhs_sorted_tiled_raw"
