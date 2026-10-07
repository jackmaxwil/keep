from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_vq.quality.prompts import QualityPrompt


def _load_export_cli():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "export_glm45_air_distributed_teacher_cache.py"
    )
    spec = importlib.util.spec_from_file_location("distributed_teacher_cache_export_test", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert text
        assert add_special_tokens is False
        return [7, 8, 9]


def test_prompt_tokens_repeat_to_requested_context() -> None:
    cli = _load_export_cli()
    prompt = QualityPrompt(
        prompt_id="synthetic_context",
        text="repeat",
        max_new_tokens=1,
        context_tokens=8,
    )

    tokens = cli._prompt_tokens(_FakeTokenizer(), prompt)

    assert tokens == [7, 8, 9, 7, 8, 9, 7, 8]


def test_selected_prompts_use_named_lane0_set_or_explicit_ids() -> None:
    cli = _load_export_cli()

    widened = cli._selected_prompts(None, prompt_set="lane0_widened")
    widened_ids = [prompt.prompt_id for prompt in widened]
    explicit = cli._selected_prompts(["short_math_multi_step"], prompt_set="base")

    assert widened_ids[:5] == [
        "capital_france",
        "short_math",
        "code_completion",
        "instruction_following",
        "long_recall_1k",
    ]
    assert len(widened_ids) == 17
    assert "short_math_multi_step" in widened_ids
    assert "code_completion_branch" in widened_ids
    assert "instruction_following_json" in widened_ids
    assert [prompt.prompt_id for prompt in explicit] == ["short_math_multi_step"]


def test_aggregate_rank_metrics_uses_max_known_values() -> None:
    cli = _load_export_cli()

    aggregate = cli._aggregate_rank_metrics(
        [
            {
                "rank": 0,
                "elapsed_seconds": 10.0,
                "mlx_peak_bytes": 120,
                "pageouts_delta": None,
                "swapouts_delta": 3,
            },
            {
                "rank": 1,
                "elapsed_seconds": 9.0,
                "mlx_peak_bytes": 90,
                "pageouts_delta": 7,
                "swapouts_delta": None,
            },
        ]
    )

    assert aggregate == {
        "elapsed_seconds": 10.0,
        "mlx_peak_bytes": 120,
        "pageouts_delta": 7,
        "swapouts_delta": 3,
    }


def test_stream_lm_head_logits_reads_bf16_rows_in_chunks(tmp_path) -> None:
    cli = _load_export_cli()
    rank_dir = tmp_path / "rank-0"
    rank_dir.mkdir()
    shard = rank_dir / "model-00001-of-00001.safetensors"
    weight = mx.array(
        np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            dtype=np.float32,
        )
    ).astype(mx.bfloat16)
    mx.save_safetensors(str(shard), {"lm_head.weight": weight})
    (rank_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {"lm_head.weight": shard.name}}),
        encoding="utf-8",
    )
    hidden = mx.array(np.array([[[2.0, 3.0], [7.0, 11.0]]], dtype=np.float32)).astype(
        mx.bfloat16
    )

    logits = cli._stream_lm_head_logits(
        hidden,
        rank_dir=rank_dir,
        input_token_ids=[10, 20, 30],
        max_positions=1,
        chunk_rows=2,
    )

    np.testing.assert_allclose(logits, np.array([[8.0, 18.0, 28.0]], dtype=np.float32))


def test_stream_lm_head_topk_payload_matches_full_log_softmax(tmp_path) -> None:
    cli = _load_export_cli()
    rank_dir = tmp_path / "rank-0"
    rank_dir.mkdir()
    shard = rank_dir / "model-00001-of-00001.safetensors"
    weight = mx.array(
        np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ],
            dtype=np.float32,
        )
    ).astype(mx.bfloat16)
    mx.save_safetensors(str(shard), {"lm_head.weight": weight})
    (rank_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {"lm_head.weight": shard.name}}),
        encoding="utf-8",
    )
    hidden = mx.array(np.array([[[2.0, 3.0], [7.0, 11.0]]], dtype=np.float32)).astype(
        mx.bfloat16
    )

    payload = cli._stream_lm_head_topk_payload(
        hidden,
        rank_dir=rank_dir,
        input_token_ids=[10, 1, 2],
        max_positions=2,
        chunk_rows=2,
        top_k=2,
    )

    full_logits = np.array([[8.0, 18.0, 28.0, 38.0], [29.0, 65.0, 101.0, 137.0]])
    log_probs = full_logits - np.log(np.exp(full_logits).sum(axis=-1, keepdims=True))
    expected_topk_ids = np.array([[3, 2], [3, 2]], dtype=np.int32)
    np.testing.assert_array_equal(payload.topk_ids, expected_topk_ids)
    np.testing.assert_allclose(
        payload.topk_logprobs,
        np.take_along_axis(log_probs, expected_topk_ids, axis=-1),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        payload.target_logprobs,
        log_probs[np.arange(2), np.array([1, 2])],
        rtol=1e-5,
    )
    np.testing.assert_array_equal(payload.teacher_top1_ids, np.array([3, 3], dtype=np.int32))


def test_stream_lm_head_topk_payload_reports_nonfinite_source(tmp_path) -> None:
    cli = _load_export_cli()
    rank_dir = tmp_path / "rank-0"
    rank_dir.mkdir()
    shard = rank_dir / "model-00001-of-00001.safetensors"
    weight = mx.array(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)).astype(mx.bfloat16)
    mx.save_safetensors(str(shard), {"lm_head.weight": weight})
    (rank_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {"lm_head.weight": shard.name}}),
        encoding="utf-8",
    )
    hidden = mx.array(np.array([[[np.nan, 3.0]]], dtype=np.float32)).astype(mx.bfloat16)

    try:
        cli._stream_lm_head_topk_payload(
            hidden,
            rank_dir=rank_dir,
            input_token_ids=[10, 1],
            max_positions=1,
            chunk_rows=2,
            top_k=1,
        )
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected non-finite top-k payload diagnostic")

    assert "streamed lm_head top-k payload contains non-finite values" in message
    assert '"selected_hidden":' in message
    assert '"finite": false' in message
    assert '"first_nonfinite_chunk": {"end": 2, "start": 0}' in message
    assert '"topk_logprobs":' in message


def test_custom_layer_bounds_keep_rank0_as_late_partition() -> None:
    cli = _load_export_cli()

    assert cli._custom_layer_bounds(
        num_hidden_layers=46,
        rank=0,
        pipeline_size=2,
        layer_split=24,
    ) == (24, 46)
    assert cli._custom_layer_bounds(
        num_hidden_layers=46,
        rank=1,
        pipeline_size=2,
        layer_split=24,
    ) == (0, 24)


def test_apply_pipeline_split_can_prune_rank0_after_route_trace_stop_layer() -> None:
    cli = _load_export_cli()

    class FakeLanguageModel:
        def __init__(self) -> None:
            self.layers = list(range(6))

    model = FakeLanguageModel()
    cli._apply_pipeline_split(
        model,
        rank=0,
        pipeline_size=2,
        layer_split=3,
        rank0_stop_after_layer=4,
    )

    assert model.start_idx == 3
    assert model.end_idx == 5
    assert model.layers == [None, None, None, 3, 4]


def test_mlx_memory_limit_gb_conversion_allows_zero_cache_limit() -> None:
    cli = _load_export_cli()

    assert cli._gb_to_bytes(1.5, name="--mlx-memory-limit-gb") == 1610612736
    assert cli._gb_to_bytes(0.0, name="--mlx-cache-limit-gb", allow_zero=True) == 0


def test_route_coverage_summary_counts_experts_and_scores() -> None:
    cli = _load_export_cli()

    summary = cli._summarize_route_coverage_events(
        {
            31: [
                (
                    np.array([[[7, 3], [7, 4]]], dtype=np.int32),
                    np.array([[[0.8, 0.2], [0.7, 0.3]]], dtype=np.float32),
                )
            ],
            41: [
                (
                    np.array([[[9, 7]]], dtype=np.int32),
                    np.array([[[0.6, 0.4]]], dtype=np.float32),
                )
            ],
        },
        row_intent="layer41_glu_code_route",
    )

    assert summary["31"]["token_count"] == 2
    assert summary["31"]["route_count"] == 4
    assert summary["31"]["unique_experts"] == [3, 4, 7]
    assert summary["31"]["top_experts"][:2] == [
        {"expert": 7, "count": 2},
        {"expert": 3, "count": 1},
    ]
    assert summary["31"]["router_score_max"] == 0.800000011920929
    assert summary["41"]["glu_code_route_count"] == 2


def test_route_coverage_summary_can_include_token_route_trace() -> None:
    cli = _load_export_cli()

    summary = cli._summarize_route_coverage_events(
        {
            36: [
                (
                    np.array([[[7, 3], [7, 4]]], dtype=np.int32),
                    np.array([[[0.8, 0.2], [0.7, 0.3]]], dtype=np.float32),
                )
            ]
        },
        row_intent=None,
        include_trace=True,
    )

    assert summary["36"]["route_trace"] == {
        "top_k": 2,
        "token_expert_indices": [[7, 3], [7, 4]],
        "token_router_scores": [[0.800000011920929, 0.20000000298023224], [0.699999988079071, 0.30000001192092896]],
    }


def test_route_trace_only_row_is_not_a_teacher_cache_row() -> None:
    cli = _load_export_cli()
    prompt = QualityPrompt(
        prompt_id="report_route_000",
        text="trace",
        max_new_tokens=1,
        row_intent="route",
        eval_split="report",
    )
    route_trace = {
        "36": {
            "top_k": 2,
            "token_expert_indices": [[7, 3], [4, 5]],
            "token_router_scores": [[0.8, 0.2], [0.6, 0.4]],
        }
    }

    row = cli._build_route_trace_only_row(
        prompt=prompt,
        prompt_set="air_vq_ladder_report_v1",
        input_token_ids=[11, 22, 33, 44],
        max_positions=2,
        model_id="zai-org/GLM-4.5-Air",
        revision="rev",
        teacher_kind="bf16_source",
        route_coverage={"36": {"token_count": 2}},
        route_trace=route_trace,
        metadata={
            "elapsed_seconds": 1.5,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        },
    )

    assert row["record_type"] == "air_route_trace_only"
    assert row["prompt_id"] == "report_route_000"
    assert row["positions"] == [0, 1]
    assert row["target_token_ids"] == [22, 33]
    assert row["route_trace"] == route_trace
    assert row["route_trace_layers"] == ["36"]
    assert row["prompt_set"] == "air_vq_ladder_report_v1"
    assert row["teacher_cache_compatible"] is False
    assert "shard" not in row
    assert "full_logits_available" not in row


def test_parse_route_trace_layers_accepts_comma_separated_values() -> None:
    cli = _load_export_cli()

    assert cli._parse_route_trace_layers(["36, 41", "31"]) == (31, 36, 41)


def test_route_trace_only_forward_can_stop_after_last_traced_layer() -> None:
    cli = _load_export_cli()
    calls: list[str] = []

    class FakeLayer:
        def __init__(self, index: int) -> None:
            self.index = index

        def __call__(self, h, _mask, cache=None):
            calls.append(f"layer-{self.index}")
            return h + 1

    class FakeLanguageModel:
        def __init__(self) -> None:
            self.pipeline_layers = [FakeLayer(index) for index in range(4)]
            self.start_idx = 0

        def embed_tokens(self, inputs):
            calls.append("embed")
            return mx.zeros((inputs.shape[0], inputs.shape[1], 2), dtype=mx.float32)

        def norm(self, h):
            calls.append("norm")
            return h

    class FakeModel:
        def __init__(self) -> None:
            self.model = FakeLanguageModel()
            self.args = type("Args", (), {"hidden_size": 2})()

        def lm_head(self, h):
            calls.append("lm_head")
            return h

    result = cli._forward_pipeline_rank0_output(
        FakeModel(),
        [1, 2, 3],
        rank=0,
        pipeline_size=1,
        return_hidden=True,
        stop_after_layer=1,
        normalize_output=False,
    )

    mx.eval(result)
    assert calls == ["embed", "layer-0", "layer-1"]


def test_route_trace_only_forward_can_stop_after_traced_gate_before_experts() -> None:
    cli = _load_export_cli()
    calls: list[str] = []

    class FakeGate:
        def __init__(self, index: int) -> None:
            self.index = index

        def __call__(self, _x):
            calls.append(f"gate-{self.index}")
            return (
                mx.array([[[self.index, self.index + 1]]], dtype=mx.int32),
                mx.array([[[0.75, 0.25]]], dtype=mx.float32),
            )

    class FakeMLP:
        def __init__(self, index: int) -> None:
            self.gate = FakeGate(index)

    class FakeLayer:
        def __init__(self, index: int) -> None:
            self.index = index
            self.mlp = FakeMLP(index)

        def __call__(self, h, _mask, cache=None):
            calls.append(f"layer-{self.index}-pre")
            self.mlp.gate(h)
            calls.append(f"layer-{self.index}-expert")
            return h + 1

    class FakeLanguageModel:
        def __init__(self) -> None:
            self.pipeline_layers = [FakeLayer(index) for index in range(3)]
            self.layers = self.pipeline_layers
            self.start_idx = 0

        def embed_tokens(self, inputs):
            calls.append("embed")
            return mx.zeros((inputs.shape[0], inputs.shape[1], 2), dtype=mx.float32)

        def norm(self, h):
            calls.append("norm")
            return h

    class FakeModel:
        def __init__(self) -> None:
            self.model = FakeLanguageModel()
            self.args = type("Args", (), {"hidden_size": 2})()

        def lm_head(self, h):
            calls.append("lm_head")
            return h

    fake_model = FakeModel()
    tracer = cli._RouteCoverageTracer(
        layers=(1,),
        include_trace=True,
        stop_after_gate_layer=1,
    )
    tracer.install(fake_model.model)
    result = cli._forward_pipeline_rank0_output(
        fake_model,
        [1, 2, 3],
        rank=0,
        pipeline_size=1,
        return_hidden=True,
        stop_after_layer=1,
        stop_after_route_gate=1,
        normalize_output=False,
    )

    mx.eval(result)
    assert calls == [
        "embed",
        "layer-0-pre",
        "gate-0",
        "layer-0-expert",
        "layer-1-pre",
        "gate-1",
    ]
    assert tracer.summary(row_intent="route")["1"]["token_count"] == 1


def test_rank0_forward_reports_first_nonfinite_hidden_stage() -> None:
    cli = _load_export_cli()

    class FakeLayer:
        def __init__(self, index: int, *, returns_nan: bool = False) -> None:
            self.index = int(index)
            self.returns_nan = bool(returns_nan)

        def __call__(self, h, _mask, cache=None):
            if self.returns_nan:
                return mx.full(h.shape, np.nan, dtype=mx.float32)
            return h + 1

    class FakeLanguageModel:
        def __init__(self) -> None:
            self.pipeline_layers = [
                FakeLayer(24),
                FakeLayer(25, returns_nan=True),
            ]
            self.start_idx = 24

        def embed_tokens(self, inputs):
            return mx.ones((inputs.shape[0], inputs.shape[1], 2), dtype=mx.float32)

        def norm(self, h):
            return h

    class FakeModel:
        def __init__(self) -> None:
            self.model = FakeLanguageModel()
            self.args = type("Args", (), {"hidden_size": 2})()

        def lm_head(self, h):
            return h

    try:
        cli._forward_pipeline_rank0_output(
            FakeModel(),
            [1, 2, 3],
            rank=0,
            pipeline_size=1,
            return_hidden=True,
            diagnose_hidden_stages=True,
        )
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected non-finite hidden stage diagnostic")

    assert "rank0 hidden contains non-finite values" in message
    assert '"stage": "after_layer"' in message
    assert '"layer_index": 25' in message
    assert '"first_nonfinite_stage": {"layer_index": 25, "stage": "after_layer"}' in message


def test_nonzero_rank_does_not_send_nonfinite_hidden() -> None:
    cli = _load_export_cli()

    class FakeLayer:
        def __call__(self, h, _mask, cache=None):
            return mx.full(h.shape, np.nan, dtype=mx.float32)

    class FakeLanguageModel:
        def __init__(self) -> None:
            self.pipeline_layers = [FakeLayer()]
            self.start_idx = 0

        def embed_tokens(self, inputs):
            return mx.ones((inputs.shape[0], inputs.shape[1], 2), dtype=mx.float32)

        def norm(self, h):
            return h

    class FakeModel:
        def __init__(self) -> None:
            self.model = FakeLanguageModel()
            self.args = type("Args", (), {"hidden_size": 2})()

    original_send = cli.mx.distributed.send

    def fail_send(*_args, **_kwargs):
        raise AssertionError("send should not receive non-finite hidden")

    cli.mx.distributed.send = fail_send
    try:
        try:
            cli._forward_pipeline_rank0_output(
                FakeModel(),
                [1, 2],
                rank=1,
                pipeline_size=2,
                return_hidden=True,
                diagnose_hidden_stages=True,
            )
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError("expected non-finite hidden diagnostic")
    finally:
        cli.mx.distributed.send = original_send

    assert "rank0 hidden contains non-finite values" in message
    assert '"first_nonfinite_stage": {"layer_index": 0, "stage": "after_layer"}' in message


def test_route_gate_proxy_records_mlx_gate_outputs() -> None:
    cli = _load_export_cli()

    class FakeGate:
        def __call__(self, _x):
            return (
                mx.array([[[2, 5], [5, 8]]], dtype=mx.int32),
                mx.array([[[0.9, 0.1], [0.55, 0.45]]], dtype=mx.float32),
            )

    tracer = cli._RouteCoverageTracer(layers=(41,))
    proxy = cli._RouteGateProxy(FakeGate(), layer_index=41, tracer=tracer)

    inds, scores = proxy(mx.array([1], dtype=mx.float32))
    mx.eval(inds, scores)
    summary = tracer.summary(row_intent="code_debug")

    assert summary["41"]["token_count"] == 2
    assert summary["41"]["route_count"] == 4
    assert summary["41"]["unique_expert_count"] == 3
    assert summary["41"]["glu_code_route_count"] == 0


def test_route_tracer_records_corrected_route_method_outputs() -> None:
    cli = _load_export_cli()

    class FakeGate:
        def __call__(self, _x):
            raise AssertionError("corrected route bypasses gate.__call__")

    class FakeMLP:
        def __init__(self) -> None:
            self.gate = FakeGate()

        def route(self, _x):
            return (
                mx.array([[[7], [3]]], dtype=mx.int32),
                mx.array([[[0.8], [0.7]]], dtype=mx.float32),
            )

    class FakeLayer:
        def __init__(self) -> None:
            self.mlp = FakeMLP()

    class FakeLanguageModel:
        def __init__(self) -> None:
            self.layers = [FakeLayer()]

    tracer = cli._RouteCoverageTracer(layers=(0,), include_trace=True)
    model = FakeLanguageModel()
    tracer.install(model)

    inds, scores = model.layers[0].mlp.route(mx.zeros((1, 2, 4), dtype=mx.float32))
    mx.eval(inds, scores)
    summary = tracer.summary(row_intent="router_kd")
    route_trace = summary["0"]["route_trace"]

    assert route_trace["top_k"] == 1
    assert route_trace["token_expert_indices"] == [[7], [3]]
    np.testing.assert_allclose(route_trace["token_router_scores"], [[0.8], [0.7]])


def test_distributed_teacher_cache_export_rejects_bad_roots_json_before_init(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/export_glm45_air_distributed_teacher_cache.py",
            "--rank-view-roots-json",
            "[]",
            "--output-dir",
            str(tmp_path / "cache"),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "--rank-view-roots-json must be a JSON object" in completed.stderr


def test_distributed_teacher_cache_export_rejects_invalid_limits_before_init(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/export_glm45_air_distributed_teacher_cache.py",
            "--rank-view-roots-json",
            json.dumps({"0": str(tmp_path)}),
            "--output-dir",
            str(tmp_path / "cache"),
            "--top-k",
            "0",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "--top-k must be positive" in completed.stderr


def test_distributed_teacher_cache_export_rejects_invalid_prompt_set_before_init(
    tmp_path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/export_glm45_air_distributed_teacher_cache.py",
            "--rank-view-roots-json",
            json.dumps({"0": str(tmp_path)}),
            "--output-dir",
            str(tmp_path / "cache"),
            "--prompt-set",
            "missing",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "invalid choice: 'missing'" in completed.stderr


def test_distributed_teacher_cache_export_rejects_invalid_layer_split_before_init(
    tmp_path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/export_glm45_air_distributed_teacher_cache.py",
            "--rank-view-roots-json",
            json.dumps({"0": str(tmp_path)}),
            "--output-dir",
            str(tmp_path / "cache"),
            "--layer-split",
            "0",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "--layer-split must be positive" in completed.stderr


def test_distributed_teacher_cache_export_rejects_invalid_mlx_cache_limit_before_init(
    tmp_path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/export_glm45_air_distributed_teacher_cache.py",
            "--rank-view-roots-json",
            json.dumps({"0": str(tmp_path)}),
            "--output-dir",
            str(tmp_path / "cache"),
            "--mlx-cache-limit-gb",
            "-1",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "--mlx-cache-limit-gb must be non-negative" in completed.stderr


def test_distributed_teacher_cache_export_rejects_stream_without_rank0_only(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/export_glm45_air_distributed_teacher_cache.py",
            "--rank-view-roots-json",
            json.dumps({"0": str(tmp_path)}),
            "--output-dir",
            str(tmp_path / "cache"),
            "--stream-lm-head",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "--stream-lm-head requires --rank0-only-logits" in completed.stderr
