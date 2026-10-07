from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from ramp.benchmark.quant_compare import (
    add_comparison_metadata,
    aggregate_repetition_records,
    audit_vq_artifact_prefill_compatibility,
    compute_effective_bits_per_weight,
    official_conversion_memory_preflight,
    mlx_affine_bit_support,
    quantized_parameter_bytes,
    validate_mlx_routed_group_size,
)
from keep.vq.e8 import e8_1bit_packed, e8p_packed_abs_grid

_QUANT_COMPARE_SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_glm45_air_quant_compare.py"
_QUANT_COMPARE_SPEC = importlib.util.spec_from_file_location(
    "bench_glm45_air_quant_compare",
    _QUANT_COMPARE_SCRIPT,
)
assert _QUANT_COMPARE_SPEC is not None
bench_glm45_air_quant_compare = importlib.util.module_from_spec(_QUANT_COMPARE_SPEC)
assert _QUANT_COMPARE_SPEC.loader is not None
_QUANT_COMPARE_SPEC.loader.exec_module(bench_glm45_air_quant_compare)


def test_mlx_affine_support_records_one_bit_rejection_and_low_bit_support() -> None:
    support = mlx_affine_bit_support(bits=(1, 2, 3, 4), group_size=64)

    assert support[1]["supported"] is False
    assert "not supported" in str(support[1]["error"])
    assert support[2]["supported"] is True
    assert support[3]["supported"] is True
    assert support[4]["supported"] is True


def test_validate_mlx_routed_group_size_accepts_g128_and_rejects_g256() -> None:
    assert validate_mlx_routed_group_size(128, hidden_size=4096, moe_intermediate_size=1408) == {
        "group_size": 128,
        "valid": True,
        "invalid_dimensions": [],
    }

    invalid = validate_mlx_routed_group_size(256, hidden_size=4096, moe_intermediate_size=1408)

    assert invalid["valid"] is False
    assert invalid["invalid_dimensions"] == [1408]


def test_quantized_parameter_bytes_counts_weight_scale_and_bias_arrays() -> None:
    params = {
        "switch.weight": mx.zeros((2, 4), dtype=mx.uint32),
        "switch.scales": mx.zeros((2, 1), dtype=mx.float32),
        "switch.biases": mx.zeros((2, 1), dtype=mx.float32),
        "switch.bias": mx.zeros((2,), dtype=mx.float32),
    }

    counts = quantized_parameter_bytes(params)

    assert counts == {
        "code_bytes": 32,
        "scale_bytes": 8,
        "bias_bytes": 16,
        "total_quantized_bytes": 56,
    }
    assert compute_effective_bits_per_weight(total_bytes=56, weights=128) == 3.5


def test_add_comparison_metadata_schema_is_json_serializable() -> None:
    base = {
        "scenario": "decode_128",
        "elapsed_seconds": 2.0,
        "prefill_seconds": 0.25,
        "decode_seconds": 1.75,
        "pageouts_delta": 0,
        "swapouts_delta": 0,
    }

    record = add_comparison_metadata(
        base,
        engine="mlx_quant_routed_only",
        quant_family="mlx_affine",
        quantized_scope="routed_experts",
        artifact_dir="artifacts/glm-4.5-air-mlx-q2-routed-g128",
        code_bytes=32,
        scale_bytes=8,
        bias_bytes=16,
        quantized_weights=128,
        mlx_bits=2,
        mlx_group_size=128,
        mlx_mode="affine",
        non_expert_dtype_policy="source_bfloat16",
        non_expert_dtype_verified=True,
        run_index=1,
        repetition_count=3,
    )

    assert record["engine"] == "mlx_quant_routed_only"
    assert record["fresh_process"] is True
    assert record["effective_bits_per_weight"] == 3.5
    assert record["invalid_memory_pressure"] is False
    assert json.dumps(record, sort_keys=True)


def test_aggregate_repetition_records_invalidates_memory_pressure_and_reports_spread() -> None:
    records = [
        {"elapsed_seconds": 3.0, "pageouts_delta": 0, "swapouts_delta": 0},
        {"elapsed_seconds": 2.0, "pageouts_delta": 0, "swapouts_delta": 0},
        {"elapsed_seconds": 9.0, "pageouts_delta": 1, "swapouts_delta": 0},
    ]

    summary = aggregate_repetition_records(records, timing_key="elapsed_seconds")

    assert summary["invalid_memory_pressure"] is True
    assert summary["valid_repetition_count"] == 2
    assert summary["invalid_repetition_count"] == 1
    assert summary["invalid_repetition_reasons"] == [
        {
            "run_index": None,
            "pageouts_delta": 1,
            "swapouts_delta": 0,
            "reason": "memory_pressure",
        }
    ]
    assert summary["timing_median_seconds"] == 2.5
    assert summary["timing_min_seconds"] == 2.0
    assert summary["timing_max_seconds"] == 3.0
    assert summary["timing_p90_seconds"] == 3.0
    assert summary["timing_stddev_seconds"] > 0.7
    assert summary["timing_relative_spread"] == 0.4
    assert summary["timing_stable"] is True
    assert summary["invalid_timing_instability"] is False


def test_aggregate_repetition_records_flags_unstable_clean_timing() -> None:
    records = [
        {"elapsed_seconds": 1.6, "pageouts_delta": 0, "swapouts_delta": 0},
        {"elapsed_seconds": 4.0, "pageouts_delta": 0, "swapouts_delta": 0},
    ]

    summary = aggregate_repetition_records(records, timing_key="elapsed_seconds")

    assert summary["invalid_memory_pressure"] is False
    assert summary["timing_relative_spread"] > 0.5
    assert summary["timing_stable"] is False
    assert summary["invalid_timing_instability"] is True


def test_official_conversion_memory_preflight_applies_safety_factor() -> None:
    rejected = official_conversion_memory_preflight(
        dense_index_total_bytes=110,
        system_memory_bytes=128,
        safety_factor=2.0,
    )
    accepted = official_conversion_memory_preflight(
        dense_index_total_bytes=50,
        system_memory_bytes=128,
        safety_factor=2.0,
    )

    assert rejected["allowed"] is False
    assert rejected["required_bytes"] == 220
    assert accepted["allowed"] is True


def test_quant_compare_parser_and_worker_command_forward_profile_components() -> None:
    parser = bench_glm45_air_quant_compare.build_arg_parser()
    parsed = parser.parse_args(["--engine", "mlx_q2_routed_g128", "--profile-components"])

    command = bench_glm45_air_quant_compare._worker_command(
        Namespace(
            engine=parsed.engine,
            scenario=parsed.scenario,
            model_id=parsed.model_id,
            revision=parsed.revision,
            prompt=parsed.prompt,
            repetitions=parsed.repetitions,
            artifact_dir=None,
            source_dir=None,
            config_path=None,
            index_path=None,
            context_tokens=None,
            max_new_tokens=None,
            profile_components=parsed.profile_components,
        ),
        run_index=2,
    )

    assert parsed.profile_components is True
    assert parsed.timing_stability_max_relative_spread == 0.5
    assert parsed.warmup_repetitions == 0
    assert parsed.parent_vm_stat_diagnostics is False
    assert parsed.mlx_cache_limit_gb is None
    assert parsed.mlx_clear_cache_before_run is False
    assert "--profile-components" in command
    assert command[command.index("--run-index") + 1] == "2"


def test_quant_compare_parser_accepts_timing_stability_limit() -> None:
    parser = bench_glm45_air_quant_compare.build_arg_parser()
    parsed = parser.parse_args(["--timing-stability-max-relative-spread", "0.2"])

    assert parsed.timing_stability_max_relative_spread == 0.2


def test_quant_compare_parser_accepts_warmup_repetitions() -> None:
    parser = bench_glm45_air_quant_compare.build_arg_parser()
    parsed = parser.parse_args(["--warmup-repetitions", "1"])

    assert parsed.warmup_repetitions == 1


def test_quant_compare_parser_accepts_clean_repetition_max_attempts() -> None:
    parser = bench_glm45_air_quant_compare.build_arg_parser()
    parsed = parser.parse_args(["--clean-repetition-max-attempts", "4"])

    assert parsed.clean_repetition_max_attempts == 4


def test_quant_compare_parser_accepts_parent_vm_stat_diagnostics() -> None:
    parser = bench_glm45_air_quant_compare.build_arg_parser()
    parsed = parser.parse_args(["--parent-vm-stat-diagnostics"])

    assert parsed.parent_vm_stat_diagnostics is True


def test_quant_compare_parser_accepts_memory_phase_trace() -> None:
    parser = bench_glm45_air_quant_compare.build_arg_parser()
    parsed = parser.parse_args(["--memory-phase-trace"])

    assert parsed.memory_phase_trace is True


def test_quant_compare_parser_accepts_memory_quiet_gate_flags() -> None:
    parser = bench_glm45_air_quant_compare.build_arg_parser()
    parsed = parser.parse_args(["--memory-quiet-preflight", "--require-memory-quiet-preflight"])

    assert parsed.memory_quiet_preflight is True
    assert parsed.require_memory_quiet_preflight is True


def test_quant_compare_parser_accepts_prefill_memory_quiet_flags() -> None:
    parser = bench_glm45_air_quant_compare.build_arg_parser()
    parsed = parser.parse_args(
        [
            "--prefill-memory-quiet-window-seconds",
            "2.5",
            "--prefill-memory-quiet-max-attempts",
            "3",
        ]
    )

    assert parsed.prefill_memory_quiet_window_seconds == 2.5
    assert parsed.prefill_memory_quiet_max_attempts == 3


def test_quant_compare_parser_accepts_prefill_chunk_size() -> None:
    parser = bench_glm45_air_quant_compare.build_arg_parser()
    parsed = parser.parse_args(["--prefill-chunk-size", "256"])

    assert parsed.prefill_chunk_size == 256


def test_quant_compare_parser_accepts_mlx_memory_policy_flags() -> None:
    parser = bench_glm45_air_quant_compare.build_arg_parser()
    parsed = parser.parse_args(
        [
            "--mlx-cache-limit-gb",
            "1.5",
            "--mlx-memory-limit-gb",
            "64",
            "--mlx-wired-limit-gb",
            "96",
            "--mlx-clear-cache-before-run",
        ]
    )

    assert parsed.mlx_cache_limit_gb == 1.5
    assert parsed.mlx_memory_limit_gb == 64
    assert parsed.mlx_wired_limit_gb == 96
    assert parsed.mlx_clear_cache_before_run is True


def test_quant_compare_worker_command_forwards_mlx_memory_policy_flags() -> None:
    parser = bench_glm45_air_quant_compare.build_arg_parser()
    parsed = parser.parse_args(
        [
            "--mlx-cache-limit-gb",
            "1",
            "--mlx-memory-limit-gb",
            "64",
            "--mlx-wired-limit-gb",
            "96",
            "--mlx-clear-cache-before-run",
        ]
    )

    command = bench_glm45_air_quant_compare._worker_command(parsed, run_index=1)

    assert command[command.index("--mlx-cache-limit-gb") + 1] == "1.0"
    assert command[command.index("--mlx-memory-limit-gb") + 1] == "64.0"
    assert command[command.index("--mlx-wired-limit-gb") + 1] == "96.0"
    assert "--mlx-clear-cache-before-run" in command


def test_quant_compare_worker_command_forwards_prefill_memory_quiet_flags() -> None:
    parser = bench_glm45_air_quant_compare.build_arg_parser()
    parsed = parser.parse_args(
        [
            "--prefill-memory-quiet-window-seconds",
            "3",
            "--prefill-memory-quiet-max-attempts",
            "2",
        ]
    )

    command = bench_glm45_air_quant_compare._worker_command(parsed, run_index=1)

    assert command[command.index("--prefill-memory-quiet-window-seconds") + 1] == "3.0"
    assert command[command.index("--prefill-memory-quiet-max-attempts") + 1] == "2"


def test_quant_compare_worker_command_forwards_memory_phase_trace() -> None:
    parser = bench_glm45_air_quant_compare.build_arg_parser()
    parsed = parser.parse_args(["--memory-phase-trace"])

    command = bench_glm45_air_quant_compare._worker_command(parsed, run_index=1)

    assert "--memory-phase-trace" in command


def test_quant_compare_worker_command_forwards_prefill_chunk_size() -> None:
    parser = bench_glm45_air_quant_compare.build_arg_parser()
    parsed = parser.parse_args(["--prefill-chunk-size", "256"])

    command = bench_glm45_air_quant_compare._worker_command(parsed, run_index=1)

    assert command[command.index("--prefill-chunk-size") + 1] == "256"


def test_quant_compare_worker_passes_prefill_memory_quiet_to_generation(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_generation_benchmark(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "prefill_seconds": 1.0,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
            "non_expert_dtype_verified": True,
        }

    monkeypatch.setattr(
        bench_glm45_air_quant_compare,
        "run_generation_benchmark",
        fake_run_generation_benchmark,
    )
    monkeypatch.setattr(
        bench_glm45_air_quant_compare,
        "_apply_mlx_memory_policy",
        lambda args: {"enabled": False},
    )
    monkeypatch.setattr(
        bench_glm45_air_quant_compare,
        "_load_config",
        lambda *args, **kwargs: {
            "num_hidden_layers": 2,
            "first_k_dense_replace": 1,
            "n_routed_experts": 2,
            "hidden_size": 4,
            "moe_intermediate_size": 3,
        },
    )
    monkeypatch.setattr(
        bench_glm45_air_quant_compare,
        "artifact_quantized_parameter_bytes",
        lambda artifact_dir: {"code_bytes": 1, "scale_bytes": 2, "bias_bytes": 3},
    )
    monkeypatch.setattr(
        bench_glm45_air_quant_compare,
        "audit_vq_artifact_prefill_compatibility",
        lambda artifact_dir: {
            "code_bits_counts": {"16": 135},
            "group_size_counts": {"512": 90, "352": 45},
            "nax_e8p_compatible_layer_count": 45,
            "nax_fast_compatible_layer_count": 45,
        },
    )
    parser = bench_glm45_air_quant_compare.build_arg_parser()
    parsed = parser.parse_args(
        [
            "--engine",
            "vq_e1_routed_nax_e8p",
            "--prefill-memory-quiet-window-seconds",
            "3",
            "--prefill-memory-quiet-max-attempts",
            "2",
        ]
    )

    bench_glm45_air_quant_compare._run_worker(parsed)

    assert captured["prefill_memory_quiet_window_seconds"] == 3.0
    assert captured["prefill_memory_quiet_max_attempts"] == 2


def test_quant_compare_worker_passes_memory_phase_trace_to_generation(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_generation_benchmark(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "prefill_seconds": 1.0,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
            "non_expert_dtype_verified": True,
        }

    monkeypatch.setattr(
        bench_glm45_air_quant_compare,
        "run_generation_benchmark",
        fake_run_generation_benchmark,
    )
    monkeypatch.setattr(
        bench_glm45_air_quant_compare,
        "_apply_mlx_memory_policy",
        lambda args: {"enabled": False},
    )
    monkeypatch.setattr(
        bench_glm45_air_quant_compare,
        "_load_config",
        lambda *args, **kwargs: {
            "num_hidden_layers": 2,
            "first_k_dense_replace": 1,
            "n_routed_experts": 2,
            "hidden_size": 4,
            "moe_intermediate_size": 3,
        },
    )
    monkeypatch.setattr(
        bench_glm45_air_quant_compare,
        "artifact_quantized_parameter_bytes",
        lambda artifact_dir: {"code_bytes": 1, "scale_bytes": 2, "bias_bytes": 3},
    )
    monkeypatch.setattr(
        bench_glm45_air_quant_compare,
        "audit_vq_artifact_prefill_compatibility",
        lambda artifact_dir: {
            "code_bits_counts": {"16": 135},
            "group_size_counts": {"512": 90, "352": 45},
            "nax_e8p_compatible_layer_count": 45,
            "nax_fast_compatible_layer_count": 45,
        },
    )
    parser = bench_glm45_air_quant_compare.build_arg_parser()
    parsed = parser.parse_args(["--engine", "vq_e1_routed_nax_e8p", "--memory-phase-trace"])

    bench_glm45_air_quant_compare._run_worker(parsed)

    assert captured["memory_phase_trace"] is True


def test_quant_compare_worker_passes_prefill_chunk_size_to_generation(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_generation_benchmark(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "prefill_seconds": 1.0,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
            "non_expert_dtype_verified": True,
        }

    monkeypatch.setattr(
        bench_glm45_air_quant_compare,
        "run_generation_benchmark",
        fake_run_generation_benchmark,
    )
    monkeypatch.setattr(
        bench_glm45_air_quant_compare,
        "_apply_mlx_memory_policy",
        lambda args: {"enabled": False},
    )
    monkeypatch.setattr(
        bench_glm45_air_quant_compare,
        "_load_config",
        lambda *args, **kwargs: {
            "num_hidden_layers": 2,
            "first_k_dense_replace": 1,
            "n_routed_experts": 2,
            "hidden_size": 4,
            "moe_intermediate_size": 3,
        },
    )
    monkeypatch.setattr(
        bench_glm45_air_quant_compare,
        "artifact_quantized_parameter_bytes",
        lambda artifact_dir: {"code_bytes": 1, "scale_bytes": 2, "bias_bytes": 3},
    )
    monkeypatch.setattr(
        bench_glm45_air_quant_compare,
        "audit_vq_artifact_prefill_compatibility",
        lambda artifact_dir: {
            "code_bits_counts": {"16": 135},
            "group_size_counts": {"512": 90, "352": 45},
            "nax_e8p_compatible_layer_count": 45,
            "nax_fast_compatible_layer_count": 45,
        },
    )
    parser = bench_glm45_air_quant_compare.build_arg_parser()
    parsed = parser.parse_args(["--engine", "vq_e1_routed_nax_e8p", "--prefill-chunk-size", "256"])

    bench_glm45_air_quant_compare._run_worker(parsed)

    assert captured["prefill_chunk_size"] == 256


def test_quant_compare_applies_mlx_memory_policy(monkeypatch) -> None:
    calls: list[tuple[str, int | None]] = []

    class FakeMx:
        def __init__(self) -> None:
            self.cache_bytes = 123

        def set_cache_limit(self, value: int) -> int:
            calls.append(("cache", value))
            return 999

        def set_memory_limit(self, value: int) -> int:
            calls.append(("memory", value))
            return 888

        def set_wired_limit(self, value: int) -> int:
            calls.append(("wired", value))
            return 777

        def get_cache_memory(self) -> int:
            calls.append(("get_cache", None))
            return self.cache_bytes

        def clear_cache(self) -> None:
            calls.append(("clear", None))
            self.cache_bytes = 0

    monkeypatch.setattr(bench_glm45_air_quant_compare, "mx", FakeMx())
    args = Namespace(
        mlx_cache_limit_gb=1,
        mlx_memory_limit_gb=2,
        mlx_wired_limit_gb=3,
        mlx_clear_cache_before_run=True,
    )

    policy = bench_glm45_air_quant_compare._apply_mlx_memory_policy(args)

    assert policy["enabled"] is True
    assert policy["cache_limit_bytes"] == 1024**3
    assert policy["memory_limit_bytes"] == 2 * 1024**3
    assert policy["wired_limit_bytes"] == 3 * 1024**3
    assert policy["previous_cache_limit_bytes"] == 999
    assert policy["previous_memory_limit_bytes"] == 888
    assert policy["previous_wired_limit_bytes"] == 777
    assert policy["cache_bytes_before_clear"] == 123
    assert policy["cache_bytes_after_clear"] == 0
    assert calls == [
        ("cache", 1024**3),
        ("memory", 2 * 1024**3),
        ("wired", 3 * 1024**3),
        ("get_cache", None),
        ("clear", None),
        ("get_cache", None),
    ]


def test_quant_compare_allows_zero_mlx_cache_limit(monkeypatch) -> None:
    calls: list[int] = []

    class FakeMx:
        def set_cache_limit(self, value: int) -> int:
            calls.append(value)
            return 999

    monkeypatch.setattr(bench_glm45_air_quant_compare, "mx", FakeMx())
    args = Namespace(
        mlx_cache_limit_gb=0,
        mlx_memory_limit_gb=None,
        mlx_wired_limit_gb=None,
        mlx_clear_cache_before_run=False,
    )

    policy = bench_glm45_air_quant_compare._apply_mlx_memory_policy(args)

    assert policy["enabled"] is True
    assert policy["cache_limit_bytes"] == 0
    assert policy["previous_cache_limit_bytes"] == 999
    assert calls == [0]


def test_quant_compare_repetition_series_offsets_from_requested_run_index(monkeypatch) -> None:
    calls: list[int] = []

    def fake_run_repetition(args: Namespace, *, run_index: int) -> dict[str, object]:
        calls.append(run_index)
        return {
            "prefill_seconds": float(run_index),
            "pageouts_delta": 0,
            "swapouts_delta": 0,
            "run_index": run_index,
        }

    monkeypatch.setattr(bench_glm45_air_quant_compare, "_run_repetition", fake_run_repetition)
    args = Namespace(
        engine="vq_e1_routed_nax_e8p",
        scenario="prefill_1k",
        repetitions=2,
        run_index=5,
        append_jsonl=None,
    )

    summary = bench_glm45_air_quant_compare._run_repetition_series(args)

    assert calls == [5, 6]
    assert [record["run_index"] for record in summary["records"]] == [5, 6]
    assert [record["benchmark_phase"] for record in summary["records"]] == ["measured", "measured"]


def test_quant_compare_repetition_series_discards_dirty_attempts_until_clean(monkeypatch) -> None:
    calls: list[int] = []
    rows = iter(
        [
            {"prefill_seconds": 9.0, "pageouts_delta": 1, "swapouts_delta": 0, "run_index": 5},
            {"prefill_seconds": 8.0, "pageouts_delta": 0, "swapouts_delta": 1, "run_index": 5},
            {"prefill_seconds": 1.5, "pageouts_delta": 0, "swapouts_delta": 0, "run_index": 5},
        ]
    )

    def fake_run_repetition(args: Namespace, *, run_index: int) -> dict[str, object]:
        calls.append(run_index)
        return dict(next(rows))

    monkeypatch.setattr(bench_glm45_air_quant_compare, "_run_repetition", fake_run_repetition)
    args = Namespace(
        engine="vq_e1_routed_nax_e8p",
        scenario="prefill_1k",
        repetitions=1,
        run_index=5,
        append_jsonl=None,
        clean_repetition_max_attempts=3,
    )

    summary = bench_glm45_air_quant_compare._run_repetition_series(args)

    assert calls == [5, 5, 5]
    assert len(summary["records"]) == 1
    assert summary["records"][0]["prefill_seconds"] == 1.5
    assert summary["records"][0]["clean_repetition_attempt_count"] == 3
    assert summary["records"][0]["discarded_memory_pressure_attempts"] == 2
    assert summary["valid_repetition_count"] == 1
    assert summary["invalid_memory_pressure"] is False


def test_quant_compare_repetition_series_warms_up_without_appending_to_evidence(monkeypatch) -> None:
    calls: list[int] = []
    appended: list[dict[str, object]] = []

    def fake_run_repetition(args: Namespace, *, run_index: int) -> dict[str, object]:
        calls.append(run_index)
        return {
            "prefill_seconds": float(run_index),
            "pageouts_delta": 0,
            "swapouts_delta": 0,
            "run_index": run_index,
        }

    def fake_append_jsonl(path: str, record: dict[str, object]) -> None:
        appended.append(record)

    monkeypatch.setattr(bench_glm45_air_quant_compare, "_run_repetition", fake_run_repetition)
    monkeypatch.setattr(bench_glm45_air_quant_compare, "append_jsonl", fake_append_jsonl)
    args = Namespace(
        engine="vq_e1_routed_nax_e8p",
        scenario="prefill_1k",
        repetitions=2,
        warmup_repetitions=2,
        run_index=5,
        append_jsonl="evidence.jsonl",
    )

    summary = bench_glm45_air_quant_compare._run_repetition_series(args)

    assert calls == [3, 4, 5, 6]
    assert summary["warmup_repetition_count"] == 2
    assert [record["run_index"] for record in summary["warmup_records"]] == [3, 4]
    assert [record["benchmark_phase"] for record in summary["warmup_records"]] == ["warmup", "warmup"]
    assert [record["run_index"] for record in summary["records"]] == [5, 6]
    assert [record["benchmark_phase"] for record in appended] == ["measured", "measured"]


def test_quant_compare_artifact_summary_omits_raw_measured_and_warmup_rows() -> None:
    summary = {
        "engine_name": "vq_e1_routed_nax_e8p",
        "records": [{"run_index": 1}],
        "warmup_records": [{"run_index": 0}],
        "warmup_repetition_count": 1,
    }

    artifact_summary = bench_glm45_air_quant_compare._artifact_summary(summary)

    assert artifact_summary == {
        "engine_name": "vq_e1_routed_nax_e8p",
        "warmup_repetition_count": 1,
    }


def test_quant_compare_repetition_series_records_parent_vm_stat_diagnostics(monkeypatch) -> None:
    samples = iter(
        [
            {"pageouts": 10, "swapouts": 20, "pages_free": 100},
            {"pageouts": 11, "swapouts": 20, "pages_free": 90},
        ]
    )

    def fake_collect_vm_stat_counts() -> dict[str, int]:
        return next(samples)

    def fake_run_repetition(args: Namespace, *, run_index: int) -> dict[str, object]:
        return {
            "prefill_seconds": 1.0,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
            "run_index": run_index,
        }

    monkeypatch.setattr(bench_glm45_air_quant_compare, "collect_vm_stat_counts", fake_collect_vm_stat_counts)
    monkeypatch.setattr(bench_glm45_air_quant_compare, "_run_repetition", fake_run_repetition)
    args = Namespace(
        engine="vq_e1_routed_nax_e8p",
        scenario="prefill_1k",
        repetitions=1,
        run_index=1,
        append_jsonl=None,
        parent_vm_stat_diagnostics=True,
    )

    summary = bench_glm45_air_quant_compare._run_repetition_series(args)
    diagnostic = summary["records"][0]["parent_vm_stat"]

    assert diagnostic["available"] is True
    assert diagnostic["before"]["pages_free"] == 100
    assert diagnostic["after"]["pages_free"] == 90
    assert diagnostic["deltas"]["pages_free"] == -10
    assert diagnostic["deltas"]["pageouts"] == 1


def test_quant_compare_repetition_series_records_memory_quiet_preflight(monkeypatch) -> None:
    def fake_run_repetition(args: Namespace, *, run_index: int) -> dict[str, object]:
        return {
            "prefill_seconds": 1.0,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
            "run_index": run_index,
        }

    def fake_wait_for_memory_quiet(*, window_seconds: float, max_attempts: int) -> dict[str, object]:
        return {
            "enabled": True,
            "available": True,
            "quiet": True,
            "attempts": max_attempts,
            "window_seconds": window_seconds,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        }

    monkeypatch.setattr(
        bench_glm45_air_quant_compare,
        "wait_for_memory_quiet",
        fake_wait_for_memory_quiet,
    )
    monkeypatch.setattr(bench_glm45_air_quant_compare, "_run_repetition", fake_run_repetition)
    args = Namespace(
        engine="vq_e1_routed_nax_e8p",
        scenario="prefill_1k",
        repetitions=1,
        run_index=1,
        append_jsonl=None,
        memory_quiet_window_seconds=0.1,
        memory_quiet_max_attempts=2,
    )

    summary = bench_glm45_air_quant_compare._run_repetition_series(args)

    assert summary["records"][0]["memory_quiet_preflight"] == {
        "enabled": True,
        "available": True,
        "quiet": True,
        "attempts": 2,
        "window_seconds": 0.1,
        "pageouts_delta": 0,
        "swapouts_delta": 0,
    }


def test_quant_compare_required_memory_quiet_preflight_blocks_worker(monkeypatch) -> None:
    launched = False

    def fake_run_repetition(args: Namespace, *, run_index: int) -> dict[str, object]:
        nonlocal launched
        launched = True
        return {"prefill_seconds": 1.0, "pageouts_delta": 0, "swapouts_delta": 0}

    monkeypatch.setattr(
        bench_glm45_air_quant_compare,
        "wait_for_memory_quiet",
        lambda *, window_seconds, max_attempts: {
            "enabled": True,
            "available": True,
            "quiet": False,
            "attempts": 1,
            "window_seconds": window_seconds,
            "pageouts_delta": 1,
            "swapouts_delta": 0,
        },
    )
    monkeypatch.setattr(bench_glm45_air_quant_compare, "_run_repetition", fake_run_repetition)
    args = Namespace(
        engine="vq_e1_routed_nax_e8p",
        scenario="prefill_1k",
        repetitions=1,
        run_index=1,
        append_jsonl=None,
        memory_quiet_window_seconds=0.1,
        memory_quiet_max_attempts=1,
        require_memory_quiet_preflight=True,
        parent_vm_stat_diagnostics=False,
    )

    with pytest.raises(bench_glm45_air_quant_compare.MemoryQuietPreflightError) as error:
        bench_glm45_air_quant_compare._run_repetition_series(args)

    assert launched is False
    assert error.value.preflight["quiet"] is False
    assert error.value.preflight["pageouts_delta"] == 1


def test_quant_compare_publication_matrix_defaults_are_the_headline_cells() -> None:
    assert bench_glm45_air_quant_compare.PUBLICATION_ENGINES == (
        "vq_e1_routed",
        "vq_e1_routed_nax_e8",
        "vq_e1_routed_nax_e8p",
        "vq_e1_routed_vq_metal",
        "mlx_q2_routed_g128",
    )
    assert bench_glm45_air_quant_compare.PUBLICATION_SCENARIOS == (
        "decode_128",
        "prefill_1k",
        "prefill_4k",
    )


def test_quant_compare_vq_engine_names_separate_default_auto_from_legacy_metal() -> None:
    engines = bench_glm45_air_quant_compare.ENGINE_CONFIG

    assert engines["vq_e1_routed"]["prefill_engine"] == "auto"
    assert engines["vq_e1_routed"]["engine"] == "vq_resident"
    assert engines["vq_e1_routed_vq_metal"]["prefill_engine"] == "vq_metal"
    assert engines["vq_e1_routed_nax_e8"]["prefill_engine"] == "nax_e8"
    assert engines["vq_e1_routed_nax_e8p"]["prefill_engine"] == "nax_e8p"


def _write_switch_projection(
    path: Path,
    *,
    layer: int,
    projection: str,
    code_bits: int = 8,
    group_size: int = 8,
) -> None:
    prefix = f"model.layers.{layer}.mlp.switch_mlp.{projection}"
    dtype = np.uint8 if code_bits == 8 else np.uint16
    rng = np.random.default_rng(layer * 100 + len(projection) + code_bits)
    codes = rng.integers(0, 256, size=(2, 4, 2), dtype=dtype)
    scales = rng.uniform(0.8, 1.2, size=(2, 4, 16 // group_size)).astype(np.float16)
    codebook = e8_1bit_packed() if code_bits == 8 else e8p_packed_abs_grid()
    mx.save_safetensors(
        str(path),
        {
            f"{prefix}.codes": mx.array(codes),
            f"{prefix}.scales": mx.array(scales),
            "model.vq_codebook.e8": mx.array(codebook),
        },
    )


def test_audit_vq_artifact_prefill_compatibility_reports_mixed_bit_fallback(tmp_path) -> None:
    seed = tmp_path / "seed"
    seed.mkdir()
    artifact = tmp_path / "artifact"
    artifact.mkdir()

    for projection in ("gate_proj", "up_proj", "down_proj"):
        _write_switch_projection(artifact / f"layer-00001-{projection}.safetensors", layer=1, projection=projection)
    _write_switch_projection(seed / "layer-00002-gate_proj.safetensors", layer=2, projection="gate_proj", code_bits=16)
    (artifact / "layer-00002-gate_proj.safetensors").symlink_to(seed / "layer-00002-gate_proj.safetensors")
    for projection in ("up_proj", "down_proj"):
        _write_switch_projection(artifact / f"layer-00002-{projection}.safetensors", layer=2, projection=projection)
    (artifact / "conversion-manifest.json").write_text(
        json.dumps({"continuous_parameters": {"sidecars": [{"layer": 45, "projection": "gate_proj"}]}})
    )

    audit = audit_vq_artifact_prefill_compatibility(artifact)

    assert audit["projection_count"] == 6
    assert audit["layer_count"] == 2
    assert audit["code_bits_counts"] == {"16": 1, "8": 5}
    assert audit["symlink_projection_count"] == 1
    assert audit["continuous_sidecar_count"] == 1
    assert audit["nax_e8_compatible_layer_count"] == 1
    assert audit["nax_e8p_compatible_layer_count"] == 0
    assert audit["nax_fast_compatible_layer_count"] == 1
    assert audit["metal_fallback_layer_count"] == 1
    assert audit["auto_prefill_all_layers_nax_e8_compatible"] is False
    assert audit["auto_prefill_all_layers_nax_e8p_compatible"] is False
    assert audit["auto_prefill_all_layers_nax_fast_compatible"] is False
    assert audit["pinned_nax_e8_will_fail"] is True
    assert audit["pinned_nax_e8p_will_fail"] is True
    layer2 = next(row for row in audit["layer_rows"] if row["layer"] == 2)
    assert layer2["auto_prefill_implementation"] == "metal"
    assert layer2["blockers"] == ["gate_proj:code_bits=16"]
    assert layer2["nax_fast_blockers"] == ["mixed_prefill_implementations:nax_e8,nax_e8p"]


def test_audit_vq_artifact_prefill_compatibility_reports_all_e8p_fast_path(tmp_path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()

    for projection in ("gate_proj", "up_proj", "down_proj"):
        _write_switch_projection(
            artifact / f"layer-00045-{projection}.safetensors",
            layer=45,
            projection=projection,
            code_bits=16,
        )

    audit = audit_vq_artifact_prefill_compatibility(artifact)

    assert audit["projection_count"] == 3
    assert audit["layer_count"] == 1
    assert audit["code_bits_counts"] == {"16": 3}
    assert audit["nax_e8_compatible_layer_count"] == 0
    assert audit["nax_e8p_compatible_layer_count"] == 1
    assert audit["nax_fast_compatible_layer_count"] == 1
    assert audit["metal_fallback_layer_count"] == 0
    assert audit["auto_prefill_all_layers_nax_e8_compatible"] is False
    assert audit["auto_prefill_all_layers_nax_e8p_compatible"] is True
    assert audit["auto_prefill_all_layers_nax_fast_compatible"] is True
    assert audit["pinned_nax_e8_will_fail"] is True
    assert audit["pinned_nax_e8p_will_fail"] is False
    layer45 = audit["layer_rows"][0]
    assert layer45["auto_prefill_implementation"] == "nax_e8p"
    assert layer45["nax_fast_blockers"] == []


def test_quant_compare_parser_exposes_prefill_compatibility_audit() -> None:
    parser = bench_glm45_air_quant_compare.build_arg_parser()
    parsed = parser.parse_args(["--audit-prefill-compatibility", "--artifact-dir", "candidate"])

    assert parsed.audit_prefill_compatibility is True
    assert parsed.artifact_dir == "candidate"


def test_quant_compare_vq_metadata_uses_actual_artifact_audit_counts() -> None:
    single = bench_glm45_air_quant_compare._vq_metadata_from_prefill_audit(
        {
            "code_bits_counts": {"16": 135},
            "group_size_counts": {"352": 45, "512": 90},
            "nax_e8_compatible_layer_count": 0,
            "nax_e8p_compatible_layer_count": 45,
            "nax_fast_compatible_layer_count": 45,
            "metal_fallback_layer_count": 0,
            "auto_prefill_all_layers_nax_e8_compatible": False,
            "auto_prefill_all_layers_nax_e8p_compatible": True,
            "auto_prefill_all_layers_nax_fast_compatible": True,
            "pinned_nax_e8_will_fail": True,
            "pinned_nax_e8p_will_fail": False,
            "continuous_sidecar_count": 6,
            "symlink_projection_count": 135,
        }
    )
    mixed = bench_glm45_air_quant_compare._vq_metadata_from_prefill_audit(
        {
            "code_bits_counts": {"8": 90, "16": 45},
            "group_size_counts": {"352": 45, "512": 90},
        }
    )

    assert single["vq_code_bits"] == 16
    assert single["vq_group_size"] is None
    assert single["prefill_compatibility"]["pinned_nax_e8_will_fail"] is True
    assert single["prefill_compatibility"]["pinned_nax_e8p_will_fail"] is False
    assert single["prefill_compatibility"]["metal_fallback_layer_count"] == 0
    assert single["prefill_compatibility"]["auto_prefill_all_layers_nax_e8p_compatible"] is True
    assert single["prefill_compatibility"]["auto_prefill_all_layers_nax_fast_compatible"] is True
    assert mixed["vq_code_bits"] is None
