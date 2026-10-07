from __future__ import annotations

import json
from pathlib import Path

import pytest

import ramp.benchmark.glm45_air as glm45_air_benchmark
from ramp.benchmark.glm45_air import (
    append_jsonl,
    build_benchmark_record,
    ComponentTimer,
    enforce_context_caps,
    format_markdown_summary,
    run_sparse_residual_microbenchmark,
    scenario_defaults,
    summarize_component_timings,
    validate_context_gate,
)


def _metrics() -> dict[str, int | None]:
    return {
        "mlx_active_bytes": 100,
        "mlx_peak_bytes": 200,
        "mlx_cache_bytes": 50,
        "rss_bytes": 300,
        "pageouts_delta": 0,
        "swapouts_delta": 0,
    }


def test_build_benchmark_record_includes_timing_memory_and_throughput() -> None:
    record = build_benchmark_record(
        model_id="zai-org/GLM-4.5-Air",
        artifact_dir="artifacts/glm-4.5-air-vq",
        scenario="prefill_1k",
        prompt_id="synthetic_repeat_1k",
        context_tokens=1024,
        max_new_tokens=1,
        elapsed_seconds=4.0,
        prefill_seconds=2.0,
        decode_seconds=1.0,
        generated_text="ok",
        metrics=_metrics(),
        extra={"dense_expert_params": False},
    )

    assert record["prefill_tokens_per_second"] == 512.0
    assert record["decode_tokens_per_second"] == 1.0
    assert record["mlx_peak_bytes"] == 200
    assert record["dense_expert_params"] is False
    assert json.dumps(record, sort_keys=True)


def test_append_jsonl_writes_one_record_per_line(tmp_path) -> None:
    record = build_benchmark_record(
        model_id="zai-org/GLM-4.5-Air",
        artifact_dir="artifacts/glm-4.5-air-vq",
        scenario="short_decode",
        prompt_id="capital_france",
        context_tokens=5,
        max_new_tokens=8,
        elapsed_seconds=2.0,
        prefill_seconds=1.0,
        decode_seconds=1.0,
        generated_text="The capital of France is Paris.",
        metrics=_metrics(),
    )
    output = tmp_path / "bench.jsonl"

    append_jsonl(output, record)
    append_jsonl(output, record)

    lines = output.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["scenario"] == "short_decode"


def test_format_markdown_summary_is_work_log_ready() -> None:
    record = build_benchmark_record(
        model_id="zai-org/GLM-4.5-Air",
        artifact_dir="artifacts/glm-4.5-air-vq",
        scenario="moe_kernel",
        prompt_id="synthetic_moe",
        context_tokens=8,
        max_new_tokens=0,
        elapsed_seconds=0.1,
        prefill_seconds=0.1,
        decode_seconds=0.0,
        generated_text="",
        metrics=_metrics(),
    )

    summary = format_markdown_summary(record)

    assert "- Scenario: `moe_kernel`" in summary
    assert "- MLX peak bytes: `200`" in summary
    assert "- Pageouts/swapouts delta: `0` / `0`" in summary


def test_component_timer_records_named_sections() -> None:
    timer = ComponentTimer(eval_outputs=False)

    with timer.time("layer.1.moe.gate_proj"):
        pass
    with timer.time("layer.1.moe.gate_proj"):
        pass

    records = timer.to_records()

    assert records == [
        {
            "name": "layer.1.moe.gate_proj",
            "calls": 2,
            "total_seconds": records[0]["total_seconds"],
            "seconds_per_call": records[0]["seconds_per_call"],
        }
    ]
    assert records[0]["total_seconds"] >= 0
    assert records[0]["seconds_per_call"] >= 0


def test_summarize_component_timings_groups_by_component_kind() -> None:
    summary = summarize_component_timings(
        [
            {"name": "layer.1.moe.gate_proj", "calls": 2, "total_seconds": 0.5},
            {"name": "layer.2.moe.gate_proj", "calls": 2, "total_seconds": 0.25},
            {"name": "layer.1.attention", "calls": 2, "total_seconds": 1.0},
        ]
    )

    assert summary == {
        "attention": {
            "calls": 2,
            "sections": 1,
            "total_seconds": 1.0,
            "seconds_per_call": 0.5,
        },
        "moe.gate_proj": {
            "calls": 4,
            "sections": 2,
            "total_seconds": 0.75,
            "seconds_per_call": 0.1875,
        },
    }


def test_scenario_defaults_cover_short_decode_and_prefill() -> None:
    assert scenario_defaults("short_decode")["context_tokens"] is None
    assert scenario_defaults("short_decode")["max_new_tokens"] == 8
    assert scenario_defaults("decode_128")["context_tokens"] is None
    assert scenario_defaults("decode_128")["max_new_tokens"] == 128
    assert scenario_defaults("prefill_1k")["context_tokens"] == 1024
    assert scenario_defaults("prefill_4k")["context_tokens"] == 4096


def test_mlx_quantized_generation_can_chunk_prefill(monkeypatch) -> None:
    call_lengths: list[int] = []

    class FakeTokenizer:
        eos_token_ids: list[int] = []

        def encode(self, text, *, add_special_tokens=False):
            assert add_special_tokens is False
            return [1, 2, 3]

        def decode(self, tokens):
            return "decoded"

    class FakeReport:
        loaded_count = 7

    class FakeModel:
        def set_profile_recorder(self, recorder):
            pass

        def make_cache(self):
            return ["cache"]

        def __call__(self, tokens, *, cache):
            assert cache == ["cache"]
            call_lengths.append(int(tokens.shape[1]))
            return glm45_air_benchmark.mx.zeros((1, int(tokens.shape[1]), 2))

    monkeypatch.setattr(
        glm45_air_benchmark,
        "load_mlx_quantized_routed_air",
        lambda **_: (FakeModel(), FakeTokenizer(), FakeReport(), ["layer0"]),
    )
    monkeypatch.setattr(
        glm45_air_benchmark,
        "collect_metric_snapshot",
        lambda previous_vm_stat_counts=None: _metrics(),
    )
    monkeypatch.setattr(glm45_air_benchmark, "collect_vm_stat_counts", lambda: {})
    monkeypatch.setattr(glm45_air_benchmark, "reset_mlx_peak_memory", lambda: None)
    monkeypatch.setattr(
        glm45_air_benchmark,
        "glm45_air_non_expert_dtype_report",
        lambda model: {"verified": True},
    )
    monkeypatch.setattr(
        glm45_air_benchmark,
        "has_dense_glm45_air_routed_expert_parameters",
        lambda model: False,
    )
    monkeypatch.setattr(glm45_air_benchmark, "has_unbound_glm45_air_vq_experts", lambda model: False)

    record = glm45_air_benchmark.run_mlx_quantized_routed_generation_benchmark(
        scenario="prefill_1k",
        context_tokens=5,
        max_new_tokens=1,
        prefill_chunk_size=2,
    )

    assert call_lengths == [2, 2, 1]
    assert record["prefill_chunk_size"] == 2
    assert record["prefill_chunk_count"] == 3


def test_sparse_residual_microbenchmark_reports_residual_timing() -> None:
    record = run_sparse_residual_microbenchmark(
        artifact_dir="artifacts/test-sparse",
        tokens=2,
        top_k=2,
        experts=4,
        input_dims=16,
        output_dims=8,
        group_size=8,
        sparse_rows=3,
        iterations=1,
        warmup=0,
    )

    assert record["scenario"] == "sparse_residual_micro"
    assert record["sparse_rows"] == 3
    assert record["routed_rows"] == 4
    assert record["residual_ms_per_iter"] >= 0
    assert record["noop_ms_per_iter"] >= 0
    assert record["residual_over_noop_ratio"] is None or record["residual_over_noop_ratio"] >= 0


def test_larger_context_guard_requires_explicit_caps() -> None:
    validate_context_gate(
        context_tokens=4096,
        allow_larger_context=False,
        max_mlx_peak_bytes=None,
        max_rss_bytes=None,
        max_prefill_seconds=None,
    )

    with pytest.raises(ValueError, match="larger than the 4K baseline"):
        validate_context_gate(
            context_tokens=8192,
            allow_larger_context=False,
            max_mlx_peak_bytes=None,
            max_rss_bytes=None,
            max_prefill_seconds=None,
        )

    with pytest.raises(ValueError, match="requires all larger-context caps"):
        validate_context_gate(
            context_tokens=8192,
            allow_larger_context=True,
            max_mlx_peak_bytes=42_000_000_000,
            max_rss_bytes=None,
            max_prefill_seconds=900.0,
        )


def test_enforce_context_caps_checks_post_run_metrics() -> None:
    record = build_benchmark_record(
        model_id="zai-org/GLM-4.5-Air",
        artifact_dir="artifacts/glm-4.5-air-vq",
        scenario="prefill",
        prompt_id="synthetic_repeat",
        context_tokens=8192,
        max_new_tokens=1,
        elapsed_seconds=100.0,
        prefill_seconds=99.0,
        decode_seconds=0.0,
        generated_text="",
        metrics=_metrics(),
    )

    enforce_context_caps(
        record,
        max_mlx_peak_bytes=250,
        max_rss_bytes=350,
        max_prefill_seconds=100.0,
    )

    with pytest.raises(ValueError, match="prefill_seconds"):
        enforce_context_caps(
            record,
            max_mlx_peak_bytes=250,
            max_rss_bytes=350,
            max_prefill_seconds=98.0,
        )
    with pytest.raises(ValueError, match="mlx_peak_bytes"):
        enforce_context_caps(
            record,
            max_mlx_peak_bytes=199,
            max_rss_bytes=350,
            max_prefill_seconds=100.0,
        )
    with pytest.raises(ValueError, match="rss_bytes"):
        enforce_context_caps(
            record,
            max_mlx_peak_bytes=250,
            max_rss_bytes=299,
            max_prefill_seconds=100.0,
        )


def test_load_tokenizer_for_source_passes_path_object(monkeypatch, tmp_path) -> None:
    observed: dict[str, object] = {}
    tokenizer = object()

    def fake_load(path):
        observed["path"] = path
        return tokenizer

    monkeypatch.setattr("mlx_lm.tokenizer_utils.load", fake_load)

    actual = glm45_air_benchmark.load_tokenizer_for_source(tmp_path)

    assert actual is tokenizer
    assert isinstance(observed["path"], Path)


def test_mlx_quantized_generation_profile_components_emit_vq_schema(monkeypatch) -> None:
    class FakeTokenizer:
        eos_token_ids: list[int] = []

        def encode(self, text, *, add_special_tokens=False):
            assert add_special_tokens is False
            return [1, 2, 3]

        def decode(self, tokens):
            return "decoded"

    class FakeReport:
        loaded_count = 7

    class FakeModel:
        def __init__(self):
            self.recorder = None

        def set_profile_recorder(self, recorder):
            self.recorder = recorder

        def make_cache(self):
            return []

        def __call__(self, tokens, *, cache):
            assert self.recorder is not None
            with self.recorder.time("layer.0.moe.gate_proj"):
                pass
            return glm45_air_benchmark.mx.array([[[0.0, 1.0]]])

    fake_model = FakeModel()
    monkeypatch.setattr(
        glm45_air_benchmark,
        "load_mlx_quantized_routed_air",
        lambda **_: (fake_model, FakeTokenizer(), FakeReport(), ["layer0"]),
    )
    monkeypatch.setattr(
        glm45_air_benchmark,
        "collect_metric_snapshot",
        lambda previous_vm_stat_counts=None: _metrics(),
    )
    monkeypatch.setattr(glm45_air_benchmark, "collect_vm_stat_counts", lambda: {})
    monkeypatch.setattr(glm45_air_benchmark, "reset_mlx_peak_memory", lambda: None)
    monkeypatch.setattr(
        glm45_air_benchmark,
        "glm45_air_non_expert_dtype_report",
        lambda model: {"verified": True},
    )
    monkeypatch.setattr(
        glm45_air_benchmark,
        "has_dense_glm45_air_routed_expert_parameters",
        lambda model: False,
    )
    monkeypatch.setattr(glm45_air_benchmark, "has_unbound_glm45_air_vq_experts", lambda model: False)

    record = glm45_air_benchmark.run_mlx_quantized_routed_generation_benchmark(
        scenario="short_decode",
        max_new_tokens=1,
        profile_components=True,
    )

    assert record["component_timings"][0]["name"] == "layer.0.moe.gate_proj"
    assert record["component_summary"]["moe.gate_proj"]["calls"] == 1
    assert record["bound_mlx_quant_layers"] == 1
    assert record["profile_components"] is True
    assert record["timing_instrumented"] is True
    assert "not the top-level timing fields" in record["timing_instrumentation_note"]
