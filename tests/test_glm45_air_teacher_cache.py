from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_vq.quality.teacher_cache import (
    build_teacher_cache_payload,
    build_teacher_cache_topk_payload,
    evaluate_teacher_cache_row,
    read_teacher_cache_rows,
    summarize_teacher_cache_records,
    validate_teacher_cache_metadata,
    validate_teacher_cache_row,
)
from mlx_vq.io.logit_bias import write_logit_bias_artifact_manifest, write_logit_bias_sidecar


def _load_eval_teacher_cache_cli():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "eval_glm45_air_teacher_cache.py"
    spec = importlib.util.spec_from_file_location("eval_glm45_air_teacher_cache_test", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_select_teacher_cache_rows_cli():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "select_glm45_air_teacher_cache_rows.py"
    spec = importlib.util.spec_from_file_location("select_glm45_air_teacher_cache_rows_test", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_replay_saved_vq_logits_cli():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "replay_glm45_air_saved_vq_logits.py"
    spec = importlib.util.spec_from_file_location("replay_glm45_air_saved_vq_logits_test", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_export_teacher_cache_cli():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "export_glm45_air_teacher_cache.py"
    spec = importlib.util.spec_from_file_location("export_glm45_air_teacher_cache_test", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_distributed_teacher_cache_cli():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "export_glm45_air_distributed_teacher_cache.py"
    )
    spec = importlib.util.spec_from_file_location(
        "export_glm45_air_distributed_teacher_cache_test",
        path,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _log_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def _with_memory_counters(row: dict) -> dict:
    row.update(
        {
            "elapsed_seconds": 0.01,
            "mlx_active_bytes": 1,
            "mlx_peak_bytes": 1,
            "mlx_cache_bytes": 0,
            "rss_bytes": 1,
            "pageouts_total": 0,
            "swapouts_total": 0,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        }
    )
    return row


def test_teacher_cache_full_logits_reports_exact_metrics(tmp_path) -> None:
    teacher_logits = np.array(
        [
            [3.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
        ],
        dtype=np.float32,
    )
    vq_logits = np.array(
        [
            [2.5, 1.0, 0.0],
            [0.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    mx.save_safetensors(str(tmp_path / "teacher.safetensors"), {"logits": mx.array(teacher_logits)})
    row = _with_memory_counters({
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "bf16_source",
        "prompt_id": "synthetic",
        "input_token_ids": [10, 11, 12],
        "target_token_ids": [0, 1],
        "positions": [0, 1],
        "logit_shard": "teacher.safetensors",
        "logit_tensor": "logits",
        "full_logits_available": True,
    })

    record = evaluate_teacher_cache_row(row, vq_logits=mx.array(vq_logits), cache_root=tmp_path)

    teacher_log_probs = _log_softmax(teacher_logits)
    vq_log_probs = _log_softmax(vq_logits)
    expected_teacher_nll = -float(np.mean([teacher_log_probs[0, 0], teacher_log_probs[1, 1]]))
    expected_vq_nll = -float(np.mean([vq_log_probs[0, 0], vq_log_probs[1, 1]]))
    expected_kld = np.sum(
        np.exp(teacher_log_probs) * (teacher_log_probs - vq_log_probs),
        axis=-1,
    )

    assert record["kld_mode"] == "exact_full_logits"
    assert record["full_logits_available"] is True
    assert record["top1_agreement"] == 1.0
    assert np.isclose(record["teacher_nll"], expected_teacher_nll)
    assert np.isclose(record["vq_nll"], expected_vq_nll)
    assert np.isclose(record["mean_kld"], float(np.mean(expected_kld)))
    assert np.isclose(record["ppl_ratio"], float(np.exp(expected_vq_nll - expected_teacher_nll)))
    json.dumps(record, sort_keys=True)


def test_teacher_cache_topk_fallback_and_summary(tmp_path) -> None:
    vq_logits = np.array(
        [
            [2.0, 1.0, 0.0, -1.0],
            [0.0, 1.0, 2.0, -1.0],
        ],
        dtype=np.float32,
    )
    topk_probs = np.array(
        [
            [0.7, 0.2],
            [0.6, 0.3],
        ],
        dtype=np.float32,
    )
    row = _with_memory_counters({
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "q8",
        "prompt_id": "synthetic_topk",
        "input_token_ids": [10, 11, 12],
        "target_token_ids": [0, 2],
        "positions": [0, 1],
        "target_logprobs": [float(np.log(0.7)), float(np.log(0.6))],
        "teacher_top1_ids": [0, 2],
        "topk_ids": [[0, 1], [2, 1]],
        "topk_logprobs": np.log(topk_probs).tolist(),
        "full_logits_available": False,
    })

    record = evaluate_teacher_cache_row(
        row,
        vq_logits=mx.array(vq_logits),
        cache_root=tmp_path,
        extra={"pageouts_delta": 0, "swapouts_delta": 0},
    )
    summary = summarize_teacher_cache_records([record])

    assert record["kld_mode"] == "teacher_topk_lower_bound"
    assert record["full_logits_available"] is False
    assert np.isclose(
        record["teacher_topk_probability_mass"],
        float(np.mean(np.sum(topk_probs, axis=-1))),
    )
    assert record["top1_agreement"] == 1.0
    assert summary["record_count"] == 1
    assert summary["clean_record_count"] == 1
    assert summary["all_memory_clean"] is True
    assert np.isclose(summary["mean_kld"], record["mean_kld"])


def test_teacher_cache_summary_p999_uses_token_level_klds() -> None:
    records = [
        {
            "nll_delta": 0.1,
            "ppl_ratio": 1.1,
            "mean_kld": 0.2,
            "token_klds": [0.1, 0.2, 0.3],
            "top1_agreement": 1.0,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        },
        {
            "nll_delta": 0.2,
            "ppl_ratio": 1.2,
            "mean_kld": 5.0,
            "token_klds": [5.0],
            "top1_agreement": 0.0,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        },
    ]

    summary = summarize_teacher_cache_records(records)

    expected_token_p999 = float(np.quantile(np.array([0.1, 0.2, 0.3, 5.0]), 0.999))
    row_mean_p999 = float(np.quantile(np.array([0.2, 5.0]), 0.999))
    assert np.isclose(summary["p999_kld"], expected_token_p999)
    assert not np.isclose(summary["p999_kld"], row_mean_p999)


def test_teacher_cache_summary_requires_explicit_clean_memory_counters() -> None:
    records = [
        {
            "nll_delta": 0.1,
            "ppl_ratio": 1.1,
            "mean_kld": 0.2,
            "token_klds": [0.2],
            "top1_agreement": 1.0,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        },
        {
            "nll_delta": 0.2,
            "ppl_ratio": 1.2,
            "mean_kld": 0.3,
            "token_klds": [0.3],
            "top1_agreement": 1.0,
        },
    ]

    summary = summarize_teacher_cache_records(records)

    assert summary["record_count"] == 2
    assert summary["clean_record_count"] == 1
    assert summary["all_memory_clean"] is False


def test_teacher_cache_payload_round_trips_through_consumer(tmp_path) -> None:
    teacher_logits = np.array(
        [
            [2.0, 0.5, -1.0, 0.0],
            [0.0, 1.0, 3.0, -1.0],
            [1.5, 0.0, 0.25, 2.5],
        ],
        dtype=np.float32,
    )
    input_token_ids = [99, 0, 2, 3]
    row, tensors = build_teacher_cache_payload(
        logits=mx.array(teacher_logits),
        input_token_ids=input_token_ids,
        prompt_id="payload",
        model_id="zai-org/GLM-4.5-Air",
        revision="test",
        teacher_kind="bf16_source",
        shard_path="teacher.safetensors",
        top_k=2,
        max_positions=2,
        save_full_logits=True,
    )
    mx.save_safetensors(str(tmp_path / "teacher.safetensors"), tensors)

    record = evaluate_teacher_cache_row(row, vq_logits=mx.array(teacher_logits[:2]), cache_root=tmp_path)

    assert row["target_token_ids"] == [0, 2]
    assert row["positions"] == [0, 1]
    assert set(tensors) == {
        "logits",
        "topk_ids",
        "topk_logprobs",
        "target_logprobs",
        "teacher_top1_ids",
    }
    assert record["kld_mode"] == "exact_full_logits"
    assert np.isclose(record["mean_kld"], 0.0, atol=1e-6)
    assert np.isclose(record["ppl_ratio"], 1.0, atol=1e-6)


def test_teacher_cache_topk_payload_round_trips_through_consumer(tmp_path) -> None:
    input_token_ids = [99, 0, 2]
    topk_ids = np.array([[0, 1], [2, 1]], dtype=np.int32)
    topk_logprobs = np.log(np.array([[0.7, 0.2], [0.8, 0.1]], dtype=np.float32))
    target_logprobs = np.array([topk_logprobs[0, 0], topk_logprobs[1, 0]], dtype=np.float32)
    teacher_top1_ids = np.array([0, 2], dtype=np.int32)

    row, tensors = build_teacher_cache_topk_payload(
        topk_ids=topk_ids,
        topk_logprobs=topk_logprobs,
        target_logprobs=target_logprobs,
        teacher_top1_ids=teacher_top1_ids,
        input_token_ids=input_token_ids,
        prompt_id="payload_topk",
        model_id="zai-org/GLM-4.5-Air",
        revision="test",
        teacher_kind="bf16_source",
        shard_path="teacher.safetensors",
        max_positions=2,
    )
    mx.save_safetensors(str(tmp_path / "teacher.safetensors"), tensors)

    vq_logits = np.array([[4.0, 1.0, 0.0], [0.0, 1.0, 4.0]], dtype=np.float32)
    record = evaluate_teacher_cache_row(row, vq_logits=mx.array(vq_logits), cache_root=tmp_path)

    assert row["full_logits_available"] is False
    assert row["target_token_ids"] == [0, 2]
    assert set(tensors) == {
        "topk_ids",
        "topk_logprobs",
        "target_logprobs",
        "teacher_top1_ids",
    }
    assert record["kld_mode"] == "teacher_topk_lower_bound"
    assert record["full_logits_available"] is False
    assert record["mean_kld"] is not None


def test_teacher_cache_topk_payload_rejects_nonfinite_logprobs() -> None:
    with pytest.raises(ValueError, match="topk_logprobs values must be finite"):
        build_teacher_cache_topk_payload(
            topk_ids=np.array([[0, 1]], dtype=np.int32),
            topk_logprobs=np.array([[np.nan, -0.2]], dtype=np.float32),
            target_logprobs=np.array([-0.1], dtype=np.float32),
            teacher_top1_ids=np.array([0], dtype=np.int32),
            input_token_ids=[99, 0],
            prompt_id="payload_topk_nan",
            model_id="zai-org/GLM-4.5-Air",
            revision="test",
            teacher_kind="bf16_source",
            shard_path="teacher.safetensors",
            max_positions=1,
        )

    with pytest.raises(ValueError, match="target_logprobs values must be finite"):
        build_teacher_cache_topk_payload(
            topk_ids=np.array([[0, 1]], dtype=np.int32),
            topk_logprobs=np.array([[-0.1, -0.2]], dtype=np.float32),
            target_logprobs=np.array([np.inf], dtype=np.float32),
            teacher_top1_ids=np.array([0], dtype=np.int32),
            input_token_ids=[99, 0],
            prompt_id="payload_topk_inf",
            model_id="zai-org/GLM-4.5-Air",
            revision="test",
            teacher_kind="bf16_source",
            shard_path="teacher.safetensors",
            max_positions=1,
        )


def test_local_sequential_pipeline_forward_runs_lower_then_upper_stage() -> None:
    cli = _load_distributed_teacher_cache_cli()

    class FakeLayer:
        def __init__(self, delta: float) -> None:
            self.delta = delta

        def __call__(self, h, mask, cache=None):
            return h + self.delta

    class FakeLanguageModel:
        def __init__(self, layers: list[FakeLayer], *, input_stage: bool = False) -> None:
            self.pipeline_layers = layers
            self.input_stage = input_stage

        def embed_tokens(self, inputs):
            if not self.input_stage:
                raise AssertionError("upper stage must not embed input ids")
            return inputs.astype(mx.float32)[..., None]

        def norm(self, h):
            return h + 10.0

    class FakeModel:
        def __init__(self, language_model: FakeLanguageModel) -> None:
            self.model = language_model

    lower = FakeModel(
        FakeLanguageModel(
            [FakeLayer(1.0), FakeLayer(2.0)],
            input_stage=True,
        )
    )
    upper = FakeModel(FakeLanguageModel([FakeLayer(3.0)]))

    lower_hidden = cli._forward_local_lower_stage(lower, [4, 5])
    hidden = cli._forward_local_upper_stage(upper, lower_hidden, return_hidden=True)
    mx.eval(hidden)

    np.testing.assert_allclose(
        np.asarray(hidden),
        np.array([[[20.0], [21.0]]], dtype=np.float32),
        rtol=1e-6,
    )


def test_local_sequential_export_collects_lower_stage_before_upper_load(
    tmp_path,
    monkeypatch,
) -> None:
    cli = _load_distributed_teacher_cache_cli()
    import gc as python_gc

    events: list[str] = []
    prompt = cli.QualityPrompt(
        prompt_id="synthetic_gc",
        text="synthetic",
        max_new_tokens=1,
    )

    rank0_dir = tmp_path / "rank0"
    rank1_dir = tmp_path / "rank1"
    rank0_dir.mkdir()
    rank1_dir.mkdir()

    def fake_load_rank_model_and_tokenizer(
        rank_dir,
        *,
        rank,
        pipeline_size,
        layer_split,
        rank0_stop_after_layer=None,
    ):
        del rank_dir, pipeline_size, layer_split, rank0_stop_after_layer
        events.append(f"load_rank{rank}")
        return object(), object()

    monkeypatch.setattr(
        cli,
        "_selected_prompts",
        lambda prompt_ids, *, prompt_set: [prompt],
    )
    monkeypatch.setattr(cli, "_prompt_tokens", lambda tokenizer, prompt: [3, 1])
    monkeypatch.setattr(cli, "_load_rank_model_and_tokenizer", fake_load_rank_model_and_tokenizer)
    monkeypatch.setattr(
        cli,
        "_forward_local_lower_stage",
        lambda model, token_ids: events.append("forward_lower") or mx.array([[[1.0]]]),
    )
    monkeypatch.setattr(
        cli,
        "_forward_local_upper_stage",
        lambda model, lower_hidden, *, return_hidden: events.append("forward_upper")
        or mx.array([[[1.0]]]),
    )
    monkeypatch.setattr(
        cli,
        "_stream_lm_head_topk_payload",
        lambda hidden, **kwargs: cli._TopKOnlyPayload(
            topk_ids=np.array([[1]], dtype=np.int32),
            topk_logprobs=np.array([[-0.1]], dtype=np.float32),
            target_logprobs=np.array([-0.1], dtype=np.float32),
            teacher_top1_ids=np.array([1], dtype=np.int32),
        ),
    )
    monkeypatch.setattr(cli, "_collect_vm_stat_counts", lambda: {"pageouts": 0, "swapouts": 0})
    monkeypatch.setattr(
        cli,
        "_collect_metric_snapshot",
        lambda previous_vm_stat_counts: {
            "mlx_active_bytes": 1,
            "mlx_peak_bytes": 1,
            "mlx_cache_bytes": 0,
            "rss_bytes": 1,
            "pageouts_total": 0,
            "swapouts_total": 0,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        },
    )
    monkeypatch.setattr(cli.mx, "reset_peak_memory", lambda: events.append("reset_peak_memory"))
    monkeypatch.setattr(cli.mx, "clear_cache", lambda: events.append("clear_cache"))
    monkeypatch.setattr(python_gc, "collect", lambda: events.append("gc_collect") or 0)

    args = cli.argparse.Namespace(
        rank_view_roots_json={0: rank0_dir, 1: rank1_dir},
        output_dir=str(tmp_path / "out"),
        prompt_id=["synthetic_gc"],
        prompt_set="base",
        layer_split=23,
        stream_lm_head=True,
        no_full_logits=True,
        lm_head_chunk_rows=1,
        top_k=1,
        max_positions=1,
        model_id="zai-org/GLM-4.5-Air",
        revision="test",
        teacher_kind="bf16_source",
        rank0_only_logits=True,
    )

    cli._run_local_sequential_export(args, mlx_memory_settings={
        "mlx_memory_limit_bytes": None,
        "mlx_cache_limit_bytes": 0,
        "mlx_wired_limit_bytes": None,
        "previous_mlx_memory_limit_bytes": None,
        "previous_mlx_cache_limit_bytes": None,
        "previous_mlx_wired_limit_bytes": None,
    })

    forward_lower_index = events.index("forward_lower")
    load_upper_index = events.index("load_rank0")
    lower_stage_collects = [
        index
        for index, event in enumerate(events)
        if event == "gc_collect" and forward_lower_index < index < load_upper_index
    ]
    assert lower_stage_collects, events
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["lm_head_chunk_rows"] == 1


def test_local_sequential_export_spills_lower_hidden_to_host_before_upper_load(
    tmp_path,
    monkeypatch,
) -> None:
    cli = _load_distributed_teacher_cache_cli()

    events: list[str] = []
    prompt = cli.QualityPrompt(
        prompt_id="synthetic_host_spill",
        text="synthetic",
        max_new_tokens=1,
    )

    rank0_dir = tmp_path / "rank0"
    rank1_dir = tmp_path / "rank1"
    rank0_dir.mkdir()
    rank1_dir.mkdir()

    def fake_load_rank_model_and_tokenizer(
        rank_dir,
        *,
        rank,
        pipeline_size,
        layer_split,
        rank0_stop_after_layer=None,
    ):
        del rank_dir, pipeline_size, layer_split, rank0_stop_after_layer
        events.append(f"load_rank{rank}")
        return object(), object()

    def fake_spill(lower_hidden):
        events.append("spill_lower_hidden")
        assert lower_hidden == "lower_hidden"
        return {
            "array": "host_hidden",
            "mlx_dtype": "bfloat16",
            "host_shape": [1, 1, 1],
            "host_nbytes": 4,
        }

    def fake_restore(spill):
        events.append("restore_lower_hidden")
        assert spill["array"] == "host_hidden"
        return "restored_hidden"

    def fake_forward_upper(model, lower_hidden, *, return_hidden):
        del model, return_hidden
        events.append("forward_upper")
        assert lower_hidden == "restored_hidden"
        return mx.array([[[1.0]]])

    monkeypatch.setattr(cli, "_selected_prompts", lambda prompt_ids, *, prompt_set: [prompt])
    monkeypatch.setattr(cli, "_prompt_tokens", lambda tokenizer, prompt: [3, 1])
    monkeypatch.setattr(cli, "_load_rank_model_and_tokenizer", fake_load_rank_model_and_tokenizer)
    monkeypatch.setattr(
        cli,
        "_forward_local_lower_stage",
        lambda model, token_ids: events.append("forward_lower") or "lower_hidden",
    )
    monkeypatch.setattr(cli, "_spill_local_hidden_to_host", fake_spill)
    monkeypatch.setattr(cli, "_restore_local_hidden_from_host", fake_restore)
    monkeypatch.setattr(cli, "_forward_local_upper_stage", fake_forward_upper)
    monkeypatch.setattr(
        cli,
        "_stream_lm_head_topk_payload",
        lambda hidden, **kwargs: cli._TopKOnlyPayload(
            topk_ids=np.array([[1]], dtype=np.int32),
            topk_logprobs=np.array([[-0.1]], dtype=np.float32),
            target_logprobs=np.array([-0.1], dtype=np.float32),
            teacher_top1_ids=np.array([1], dtype=np.int32),
        ),
    )
    monkeypatch.setattr(cli, "_collect_vm_stat_counts", lambda: {"pageouts": 0, "swapouts": 0})
    monkeypatch.setattr(
        cli,
        "_collect_metric_snapshot",
        lambda previous_vm_stat_counts: {
            "mlx_active_bytes": 1,
            "mlx_peak_bytes": 1,
            "mlx_cache_bytes": 0,
            "rss_bytes": 1,
            "pageouts_total": 0,
            "swapouts_total": 0,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        },
    )
    monkeypatch.setattr(cli.mx, "reset_peak_memory", lambda: events.append("reset_peak_memory"))
    monkeypatch.setattr(cli, "_release_local_stage_memory", lambda: events.append("release"))

    args = cli.argparse.Namespace(
        rank_view_roots_json={0: rank0_dir, 1: rank1_dir},
        output_dir=str(tmp_path / "out"),
        prompt_id=["synthetic_host_spill"],
        prompt_set="base",
        layer_split=23,
        stream_lm_head=True,
        no_full_logits=True,
        lm_head_chunk_rows=1,
        top_k=1,
        max_positions=1,
        model_id="zai-org/GLM-4.5-Air",
        revision="test",
        teacher_kind="bf16_source",
        rank0_only_logits=True,
    )

    cli._run_local_sequential_export(
        args,
        mlx_memory_settings={
            "mlx_memory_limit_bytes": None,
            "mlx_cache_limit_bytes": 0,
            "mlx_wired_limit_bytes": None,
            "previous_mlx_memory_limit_bytes": None,
            "previous_mlx_cache_limit_bytes": None,
            "previous_mlx_wired_limit_bytes": None,
        },
    )

    assert events.index("spill_lower_hidden") < events.index("load_rank0")
    assert events.index("release") < events.index("load_rank0")
    assert events.index("load_rank0") < events.index("restore_lower_hidden")
    assert events.index("restore_lower_hidden") < events.index("forward_upper")

    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["local_sequential_lower_hidden_host_spill"] is True
    assert rows[0]["local_sequential_lower_hidden_host_nbytes"] == 4
    assert rows[0]["local_sequential_lower_hidden_host_dtype"] == "bfloat16"


def test_local_sequential_stage_processes_spawn_lower_then_upper_workers(
    tmp_path,
    monkeypatch,
) -> None:
    cli = _load_distributed_teacher_cache_cli()

    prompt = cli.QualityPrompt(
        prompt_id="synthetic_stage_process",
        text="synthetic",
        max_new_tokens=1,
    )
    rank0_dir = tmp_path / "rank0"
    rank1_dir = tmp_path / "rank1"
    rank0_dir.mkdir()
    rank1_dir.mkdir()
    commands: list[list[str]] = []

    def arg_value(command: list[str], name: str) -> str:
        return command[command.index(name) + 1]

    def fake_subprocess_run(command, *, check=False, text=False, capture_output=False):
        del check, text, capture_output
        commands.append(list(command))
        worker = arg_value(list(command), "--local-sequential-stage-worker")
        info_path = Path(arg_value(list(command), "--local-sequential-worker-json"))
        if worker == "lower":
            hidden_path = Path(arg_value(list(command), "--local-sequential-hidden-path"))
            np.savez(hidden_path, hidden=np.array([[[1.0]]], dtype=np.float32))
            info_path.write_text(
                json.dumps(
                    {
                        "input_token_ids": [3, 1],
                        "local_sequential_lower_hidden_host_dtype": "bfloat16",
                        "local_sequential_lower_hidden_host_shape": [1, 1, 1],
                        "local_sequential_lower_hidden_host_nbytes": 4,
                        "elapsed_seconds": 1.25,
                        "metrics": {
                            "mlx_active_bytes": 10,
                            "mlx_peak_bytes": 100,
                            "mlx_cache_bytes": 0,
                            "rss_bytes": 1000,
                            "pageouts_total": 1,
                            "swapouts_total": 2,
                            "pageouts_delta": 0,
                            "swapouts_delta": 0,
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        else:
            output_path = Path(arg_value(list(command), "--local-sequential-output-path"))
            np.savez(
                output_path,
                topk_ids=np.array([[1]], dtype=np.int32),
                topk_logprobs=np.array([[-0.1]], dtype=np.float32),
                target_logprobs=np.array([-0.1], dtype=np.float32),
                teacher_top1_ids=np.array([1], dtype=np.int32),
            )
            info_path.write_text(
                json.dumps(
                    {
                        "output_kind": "topk",
                        "elapsed_seconds": 2.5,
                        "metrics": {
                            "mlx_active_bytes": 20,
                            "mlx_peak_bytes": 200,
                            "mlx_cache_bytes": 0,
                            "rss_bytes": 2000,
                            "pageouts_total": 1,
                            "swapouts_total": 2,
                            "pageouts_delta": 0,
                            "swapouts_delta": 0,
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        return cli.subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(cli, "_selected_prompts", lambda prompt_ids, *, prompt_set: [prompt])
    monkeypatch.setattr(
        cli,
        "_load_rank_model_and_tokenizer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("stage-process mode must not load rank models in parent")
        ),
    )
    monkeypatch.setattr(cli.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(cli, "_collect_vm_stat_counts", lambda: {"pageouts": 1, "swapouts": 2})
    monkeypatch.setattr(
        cli,
        "_collect_metric_snapshot",
        lambda previous_vm_stat_counts: {
            "mlx_active_bytes": 0,
            "mlx_peak_bytes": 0,
            "mlx_cache_bytes": 0,
            "rss_bytes": 0,
            "pageouts_total": 1,
            "swapouts_total": 2,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        },
    )
    monkeypatch.setattr(cli.mx, "reset_peak_memory", lambda: None)

    args = cli.argparse.Namespace(
        rank_view_roots_json={0: rank0_dir, 1: rank1_dir},
        output_dir=str(tmp_path / "out"),
        prompt_id=["synthetic_stage_process"],
        prompt_set="base",
        layer_split=23,
        stream_lm_head=True,
        no_full_logits=True,
        lm_head_chunk_rows=1,
        top_k=1,
        max_positions=1,
        model_id="zai-org/GLM-4.5-Air",
        revision="test",
        teacher_kind="bf16_source",
        rank0_only_logits=True,
        local_sequential_stage_processes=True,
        mlx_memory_limit_gb=None,
        mlx_cache_limit_gb=0,
        mlx_wired_limit_gb=None,
    )

    cli._run_local_sequential_export(
        args,
        mlx_memory_settings={
            "mlx_memory_limit_bytes": None,
            "mlx_cache_limit_bytes": 0,
            "mlx_wired_limit_bytes": None,
            "previous_mlx_memory_limit_bytes": None,
            "previous_mlx_cache_limit_bytes": None,
            "previous_mlx_wired_limit_bytes": None,
        },
    )

    worker_names = [
        arg_value(command, "--local-sequential-stage-worker")
        for command in commands
    ]
    assert worker_names == ["lower", "upper"]
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["local_sequential_stage_processes"] is True
    assert rows[0]["local_sequential_lower_hidden_host_spill"] is True
    assert rows[0]["local_sequential_lower_hidden_host_nbytes"] == 4
    assert rows[0]["mlx_peak_bytes"] == 200
    assert [entry["stage"] for entry in rows[0]["rank_metrics"]] == [
        "local_sequential_lower_process",
        "local_sequential_upper_process",
    ]


def test_local_sequential_head_process_spawns_head_worker_after_upper_hidden(
    tmp_path,
    monkeypatch,
) -> None:
    cli = _load_distributed_teacher_cache_cli()

    prompt = cli.QualityPrompt(
        prompt_id="synthetic_head_process",
        text="synthetic",
        max_new_tokens=1,
    )
    rank0_dir = tmp_path / "rank0"
    rank1_dir = tmp_path / "rank1"
    rank0_dir.mkdir()
    rank1_dir.mkdir()
    commands: list[list[str]] = []

    def arg_value(command: list[str], name: str) -> str:
        return command[command.index(name) + 1]

    def fake_subprocess_run(command, *, check=False, text=False, capture_output=False):
        del check, text, capture_output
        commands.append(list(command))
        assert "--local-sequential-head-process" in command
        worker = arg_value(list(command), "--local-sequential-stage-worker")
        info_path = Path(arg_value(list(command), "--local-sequential-worker-json"))
        if worker == "lower":
            hidden_path = Path(arg_value(list(command), "--local-sequential-hidden-path"))
            np.savez(hidden_path, hidden=np.array([[[1.0]]], dtype=np.float32))
            info_path.write_text(
                json.dumps(
                    {
                        "input_token_ids": [3, 1],
                        "local_sequential_lower_hidden_host_dtype": "bfloat16",
                        "local_sequential_lower_hidden_host_shape": [1, 1, 1],
                        "local_sequential_lower_hidden_host_nbytes": 4,
                        "elapsed_seconds": 1.0,
                        "metrics": {
                            "mlx_active_bytes": 10,
                            "mlx_peak_bytes": 100,
                            "mlx_cache_bytes": 0,
                            "rss_bytes": 1000,
                            "pageouts_total": 1,
                            "swapouts_total": 2,
                            "pageouts_delta": 0,
                            "swapouts_delta": 0,
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        elif worker == "upper":
            output_path = Path(arg_value(list(command), "--local-sequential-output-path"))
            assert output_path.name.endswith("-upper-hidden.npz")
            np.savez(output_path, hidden=np.array([[[2.0]]], dtype=np.float32))
            info_path.write_text(
                json.dumps(
                    {
                        "input_token_ids": [3, 1],
                        "output_kind": "hidden",
                        "local_sequential_upper_hidden_host_dtype": "bfloat16",
                        "local_sequential_upper_hidden_host_shape": [1, 1, 1],
                        "local_sequential_upper_hidden_host_nbytes": 4,
                        "elapsed_seconds": 2.0,
                        "metrics": {
                            "mlx_active_bytes": 15,
                            "mlx_peak_bytes": 150,
                            "mlx_cache_bytes": 0,
                            "rss_bytes": 1500,
                            "pageouts_total": 1,
                            "swapouts_total": 2,
                            "pageouts_delta": 0,
                            "swapouts_delta": 0,
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        else:
            assert worker == "head"
            assert Path(arg_value(list(command), "--local-sequential-hidden-path")).name.endswith(
                "-upper-hidden.npz"
            )
            output_path = Path(arg_value(list(command), "--local-sequential-output-path"))
            np.savez(
                output_path,
                topk_ids=np.array([[1]], dtype=np.int32),
                topk_logprobs=np.array([[-0.1]], dtype=np.float32),
                target_logprobs=np.array([-0.1], dtype=np.float32),
                teacher_top1_ids=np.array([1], dtype=np.int32),
            )
            info_path.write_text(
                json.dumps(
                    {
                        "output_kind": "topk",
                        "elapsed_seconds": 0.5,
                        "metrics": {
                            "mlx_active_bytes": 5,
                            "mlx_peak_bytes": 50,
                            "mlx_cache_bytes": 0,
                            "rss_bytes": 500,
                            "pageouts_total": 1,
                            "swapouts_total": 2,
                            "pageouts_delta": 0,
                            "swapouts_delta": 0,
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        return cli.subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(cli, "_selected_prompts", lambda prompt_ids, *, prompt_set: [prompt])
    monkeypatch.setattr(
        cli,
        "_load_rank_model_and_tokenizer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("head-process mode must not load rank models in parent")
        ),
    )
    monkeypatch.setattr(cli.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(cli, "_collect_vm_stat_counts", lambda: {"pageouts": 1, "swapouts": 2})
    monkeypatch.setattr(
        cli,
        "_collect_metric_snapshot",
        lambda previous_vm_stat_counts: {
            "mlx_active_bytes": 0,
            "mlx_peak_bytes": 0,
            "mlx_cache_bytes": 0,
            "rss_bytes": 0,
            "pageouts_total": 1,
            "swapouts_total": 2,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        },
    )
    monkeypatch.setattr(cli.mx, "reset_peak_memory", lambda: None)

    args = cli.argparse.Namespace(
        rank_view_roots_json={0: rank0_dir, 1: rank1_dir},
        output_dir=str(tmp_path / "out"),
        prompt_id=["synthetic_head_process"],
        prompt_set="base",
        layer_split=23,
        stream_lm_head=True,
        no_full_logits=True,
        lm_head_chunk_rows=1,
        top_k=1,
        max_positions=1,
        model_id="zai-org/GLM-4.5-Air",
        revision="test",
        teacher_kind="bf16_source",
        rank0_only_logits=True,
        local_sequential_stage_processes=True,
        local_sequential_head_process=True,
        mlx_memory_limit_gb=None,
        mlx_cache_limit_gb=0,
        mlx_wired_limit_gb=None,
    )

    cli._run_local_sequential_export(
        args,
        mlx_memory_settings={
            "mlx_memory_limit_bytes": None,
            "mlx_cache_limit_bytes": 0,
            "mlx_wired_limit_bytes": None,
            "previous_mlx_memory_limit_bytes": None,
            "previous_mlx_cache_limit_bytes": None,
            "previous_mlx_wired_limit_bytes": None,
        },
    )

    worker_names = [
        arg_value(command, "--local-sequential-stage-worker")
        for command in commands
    ]
    assert worker_names == ["lower", "upper", "head"]
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["local_sequential_stage_processes"] is True
    assert rows[0]["local_sequential_head_process"] is True
    assert rows[0]["local_sequential_upper_hidden_host_spill"] is True
    assert rows[0]["local_sequential_upper_hidden_host_dtype"] == "bfloat16"
    assert rows[0]["local_sequential_upper_hidden_host_shape"] == [1, 1, 1]
    assert rows[0]["local_sequential_upper_hidden_host_nbytes"] == 4
    assert rows[0]["mlx_peak_bytes"] == 150
    assert [entry["stage"] for entry in rows[0]["rank_metrics"]] == [
        "local_sequential_lower_process",
        "local_sequential_upper_process",
        "local_sequential_head_process",
    ]


def test_local_sequential_upper_split_process_spawns_pre_final_and_head_workers(
    tmp_path,
    monkeypatch,
) -> None:
    cli = _load_distributed_teacher_cache_cli()

    prompt = cli.QualityPrompt(
        prompt_id="synthetic_upper_split_process",
        text="synthetic",
        max_new_tokens=1,
    )
    rank0_dir = tmp_path / "rank0"
    rank1_dir = tmp_path / "rank1"
    rank0_dir.mkdir()
    rank1_dir.mkdir()
    commands: list[list[str]] = []

    def arg_value(command: list[str], name: str) -> str:
        return command[command.index(name) + 1]

    def write_metrics(
        info_path: Path,
        *,
        output_kind: str | None = None,
        peak: int,
        extra: dict[str, object] | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "input_token_ids": [3, 1],
            "elapsed_seconds": peak / 100.0,
            "metrics": {
                "mlx_active_bytes": peak // 10,
                "mlx_peak_bytes": peak,
                "mlx_cache_bytes": 0,
                "rss_bytes": peak * 10,
                "pageouts_total": 1,
                "swapouts_total": 2,
                "pageouts_delta": 0,
                "swapouts_delta": 0,
            },
        }
        if output_kind is not None:
            payload["output_kind"] = output_kind
        if extra:
            payload.update(extra)
        info_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    def fake_subprocess_run(command, *, check=False, text=False, capture_output=False):
        del check, text, capture_output
        commands.append(list(command))
        assert "--local-sequential-head-process" in command
        assert arg_value(list(command), "--local-sequential-upper-split-layer") == "34"
        worker = arg_value(list(command), "--local-sequential-stage-worker")
        info_path = Path(arg_value(list(command), "--local-sequential-worker-json"))
        if worker == "lower":
            hidden_path = Path(arg_value(list(command), "--local-sequential-hidden-path"))
            np.savez(hidden_path, hidden=np.array([[[1.0]]], dtype=np.float32))
            write_metrics(
                info_path,
                peak=100,
                extra={
                    "local_sequential_lower_hidden_host_dtype": "bfloat16",
                    "local_sequential_lower_hidden_host_shape": [1, 1, 1],
                    "local_sequential_lower_hidden_host_nbytes": 4,
                },
            )
        elif worker == "upper-pre":
            assert Path(arg_value(list(command), "--local-sequential-hidden-path")).name.endswith(
                "-hidden.npz"
            )
            output_path = Path(arg_value(list(command), "--local-sequential-output-path"))
            assert output_path.name.endswith("-upper-pre-hidden.npz")
            np.savez(output_path, hidden=np.array([[[2.0]]], dtype=np.float32))
            write_metrics(
                info_path,
                output_kind="hidden",
                peak=125,
                extra={
                    "local_sequential_upper_pre_hidden_host_dtype": "bfloat16",
                    "local_sequential_upper_pre_hidden_host_shape": [1, 1, 1],
                    "local_sequential_upper_pre_hidden_host_nbytes": 4,
                },
            )
        elif worker == "upper-final":
            assert Path(arg_value(list(command), "--local-sequential-hidden-path")).name.endswith(
                "-upper-pre-hidden.npz"
            )
            output_path = Path(arg_value(list(command), "--local-sequential-output-path"))
            assert output_path.name.endswith("-upper-hidden.npz")
            np.savez(output_path, hidden=np.array([[[3.0]]], dtype=np.float32))
            write_metrics(
                info_path,
                output_kind="hidden",
                peak=175,
                extra={
                    "local_sequential_upper_hidden_host_dtype": "bfloat16",
                    "local_sequential_upper_hidden_host_shape": [1, 1, 1],
                    "local_sequential_upper_hidden_host_nbytes": 4,
                },
            )
        else:
            assert worker == "head"
            assert Path(arg_value(list(command), "--local-sequential-hidden-path")).name.endswith(
                "-upper-hidden.npz"
            )
            output_path = Path(arg_value(list(command), "--local-sequential-output-path"))
            np.savez(
                output_path,
                topk_ids=np.array([[1]], dtype=np.int32),
                topk_logprobs=np.array([[-0.1]], dtype=np.float32),
                target_logprobs=np.array([-0.1], dtype=np.float32),
                teacher_top1_ids=np.array([1], dtype=np.int32),
            )
            write_metrics(info_path, output_kind="topk", peak=50)
        return cli.subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(cli, "_selected_prompts", lambda prompt_ids, *, prompt_set: [prompt])
    monkeypatch.setattr(
        cli,
        "_load_rank_model_and_tokenizer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("upper-split stage-process mode must not load rank models in parent")
        ),
    )
    monkeypatch.setattr(cli.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(cli, "_collect_vm_stat_counts", lambda: {"pageouts": 1, "swapouts": 2})
    monkeypatch.setattr(
        cli,
        "_collect_metric_snapshot",
        lambda previous_vm_stat_counts: {
            "mlx_active_bytes": 0,
            "mlx_peak_bytes": 0,
            "mlx_cache_bytes": 0,
            "rss_bytes": 0,
            "pageouts_total": 1,
            "swapouts_total": 2,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        },
    )
    monkeypatch.setattr(cli.mx, "reset_peak_memory", lambda: None)

    args = cli.argparse.Namespace(
        rank_view_roots_json={0: rank0_dir, 1: rank1_dir},
        output_dir=str(tmp_path / "out"),
        prompt_id=["synthetic_upper_split_process"],
        prompt_set="base",
        layer_split=23,
        stream_lm_head=True,
        no_full_logits=True,
        lm_head_chunk_rows=1,
        top_k=1,
        max_positions=1,
        model_id="zai-org/GLM-4.5-Air",
        revision="test",
        teacher_kind="bf16_source",
        rank0_only_logits=True,
        local_sequential_stage_processes=True,
        local_sequential_head_process=True,
        local_sequential_upper_split_layer=34,
        mlx_memory_limit_gb=None,
        mlx_cache_limit_gb=0,
        mlx_wired_limit_gb=None,
    )

    cli._run_local_sequential_export(
        args,
        mlx_memory_settings={
            "mlx_memory_limit_bytes": None,
            "mlx_cache_limit_bytes": 0,
            "mlx_wired_limit_bytes": None,
            "previous_mlx_memory_limit_bytes": None,
            "previous_mlx_cache_limit_bytes": None,
            "previous_mlx_wired_limit_bytes": None,
        },
    )

    worker_names = [
        arg_value(command, "--local-sequential-stage-worker")
        for command in commands
    ]
    assert worker_names == ["lower", "upper-pre", "upper-final", "head"]
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["local_sequential_stage_processes"] is True
    assert rows[0]["local_sequential_head_process"] is True
    assert rows[0]["local_sequential_upper_split_layer"] == 34
    assert rows[0]["local_sequential_upper_pre_hidden_host_spill"] is True
    assert rows[0]["local_sequential_upper_pre_hidden_host_dtype"] == "bfloat16"
    assert rows[0]["local_sequential_upper_pre_hidden_host_shape"] == [1, 1, 1]
    assert rows[0]["local_sequential_upper_pre_hidden_host_nbytes"] == 4
    assert rows[0]["local_sequential_upper_hidden_host_spill"] is True
    assert rows[0]["local_sequential_upper_hidden_host_dtype"] == "bfloat16"
    assert rows[0]["local_sequential_upper_hidden_host_shape"] == [1, 1, 1]
    assert rows[0]["local_sequential_upper_hidden_host_nbytes"] == 4
    assert rows[0]["mlx_peak_bytes"] == 175
    assert [entry["stage"] for entry in rows[0]["rank_metrics"]] == [
        "local_sequential_lower_process",
        "local_sequential_upper_pre_process",
        "local_sequential_upper_final_process",
        "local_sequential_head_process",
    ]


def test_local_sequential_lower_and_upper_split_process_spawns_all_split_workers(
    tmp_path,
    monkeypatch,
) -> None:
    cli = _load_distributed_teacher_cache_cli()

    prompt = cli.QualityPrompt(
        prompt_id="synthetic_lower_upper_split_process",
        text="synthetic",
        max_new_tokens=1,
    )
    rank0_dir = tmp_path / "rank0"
    rank1_dir = tmp_path / "rank1"
    rank0_dir.mkdir()
    rank1_dir.mkdir()
    stage_dirs = {
        "lower-pre": {1: tmp_path / "stage-lower-pre-rank1"},
        "lower-final": {1: tmp_path / "stage-lower-final-rank1"},
        "upper-pre": {0: tmp_path / "stage-upper-pre-rank0"},
        "upper-final": {0: tmp_path / "stage-upper-final-rank0"},
        "head": {0: tmp_path / "stage-head-rank0"},
    }
    for roots in stage_dirs.values():
        for root in roots.values():
            root.mkdir()
    commands: list[list[str]] = []

    def arg_value(command: list[str], name: str) -> str:
        return command[command.index(name) + 1]

    def write_metrics(
        info_path: Path,
        *,
        peak: int,
        output_kind: str | None = None,
        extra: dict[str, object] | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "input_token_ids": [3, 1],
            "elapsed_seconds": peak / 100.0,
            "metrics": {
                "mlx_active_bytes": peak // 10,
                "mlx_peak_bytes": peak,
                "mlx_cache_bytes": 0,
                "rss_bytes": peak * 10,
                "pageouts_total": 1,
                "swapouts_total": 2,
                "pageouts_delta": 0,
                "swapouts_delta": 0,
            },
        }
        if output_kind is not None:
            payload["output_kind"] = output_kind
        if extra:
            payload.update(extra)
        info_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    def fake_subprocess_run(command, *, check=False, text=False, capture_output=False):
        del check, text, capture_output
        commands.append(list(command))
        assert "--local-sequential-head-process" in command
        assert arg_value(list(command), "--local-sequential-lower-split-layer") == "12"
        assert arg_value(list(command), "--local-sequential-upper-split-layer") == "34"
        worker = arg_value(list(command), "--local-sequential-stage-worker")
        roots = {
            int(rank): Path(path)
            for rank, path in json.loads(
                arg_value(list(command), "--rank-view-roots-json")
            ).items()
        }
        if worker in {"lower-pre", "lower-final"}:
            assert roots[1] == stage_dirs[worker][1]
            assert roots[0] == rank0_dir
        elif worker == "head":
            assert roots[0] == stage_dirs["head"][0]
            assert roots[1] == rank1_dir
        else:
            assert roots[0] == stage_dirs[worker][0]
            assert roots[1] == rank1_dir
        info_path = Path(arg_value(list(command), "--local-sequential-worker-json"))
        if worker == "lower-pre":
            hidden_path = Path(arg_value(list(command), "--local-sequential-hidden-path"))
            assert hidden_path.name.endswith("-lower-pre-hidden.npz")
            np.savez(hidden_path, hidden=np.array([[[1.0]]], dtype=np.float32))
            write_metrics(
                info_path,
                peak=90,
                output_kind="hidden",
                extra={
                    "local_sequential_lower_pre_hidden_host_dtype": "bfloat16",
                    "local_sequential_lower_pre_hidden_host_shape": [1, 1, 1],
                    "local_sequential_lower_pre_hidden_host_nbytes": 4,
                },
            )
        elif worker == "lower-final":
            assert Path(arg_value(list(command), "--local-sequential-hidden-path")).name.endswith(
                "-lower-pre-hidden.npz"
            )
            output_path = Path(arg_value(list(command), "--local-sequential-output-path"))
            assert output_path.name.endswith("-hidden.npz")
            np.savez(output_path, hidden=np.array([[[2.0]]], dtype=np.float32))
            write_metrics(
                info_path,
                peak=95,
                output_kind="hidden",
                extra={
                    "local_sequential_lower_hidden_host_dtype": "bfloat16",
                    "local_sequential_lower_hidden_host_shape": [1, 1, 1],
                    "local_sequential_lower_hidden_host_nbytes": 4,
                },
            )
        elif worker == "upper-pre":
            assert Path(arg_value(list(command), "--local-sequential-hidden-path")).name.endswith(
                "-hidden.npz"
            )
            output_path = Path(arg_value(list(command), "--local-sequential-output-path"))
            assert output_path.name.endswith("-upper-pre-hidden.npz")
            np.savez(output_path, hidden=np.array([[[3.0]]], dtype=np.float32))
            write_metrics(
                info_path,
                peak=125,
                output_kind="hidden",
                extra={
                    "local_sequential_upper_pre_hidden_host_dtype": "bfloat16",
                    "local_sequential_upper_pre_hidden_host_shape": [1, 1, 1],
                    "local_sequential_upper_pre_hidden_host_nbytes": 4,
                },
            )
        elif worker == "upper-final":
            assert Path(arg_value(list(command), "--local-sequential-hidden-path")).name.endswith(
                "-upper-pre-hidden.npz"
            )
            output_path = Path(arg_value(list(command), "--local-sequential-output-path"))
            assert output_path.name.endswith("-upper-hidden.npz")
            np.savez(output_path, hidden=np.array([[[4.0]]], dtype=np.float32))
            write_metrics(
                info_path,
                peak=175,
                output_kind="hidden",
                extra={
                    "local_sequential_upper_hidden_host_dtype": "bfloat16",
                    "local_sequential_upper_hidden_host_shape": [1, 1, 1],
                    "local_sequential_upper_hidden_host_nbytes": 4,
                },
            )
        else:
            assert worker == "head"
            assert Path(arg_value(list(command), "--local-sequential-hidden-path")).name.endswith(
                "-upper-hidden.npz"
            )
            output_path = Path(arg_value(list(command), "--local-sequential-output-path"))
            np.savez(
                output_path,
                topk_ids=np.array([[1]], dtype=np.int32),
                topk_logprobs=np.array([[-0.1]], dtype=np.float32),
                target_logprobs=np.array([-0.1], dtype=np.float32),
                teacher_top1_ids=np.array([1], dtype=np.int32),
            )
            write_metrics(info_path, peak=50, output_kind="topk")
        return cli.subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(cli, "_selected_prompts", lambda prompt_ids, *, prompt_set: [prompt])
    monkeypatch.setattr(
        cli,
        "_load_rank_model_and_tokenizer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("split stage-process mode must not load rank models in parent")
        ),
    )
    monkeypatch.setattr(cli.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(cli, "_collect_vm_stat_counts", lambda: {"pageouts": 1, "swapouts": 2})
    monkeypatch.setattr(
        cli,
        "_collect_metric_snapshot",
        lambda previous_vm_stat_counts: {
            "mlx_active_bytes": 0,
            "mlx_peak_bytes": 0,
            "mlx_cache_bytes": 0,
            "rss_bytes": 0,
            "pageouts_total": 1,
            "swapouts_total": 2,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        },
    )
    monkeypatch.setattr(cli.mx, "reset_peak_memory", lambda: None)

    args = cli.argparse.Namespace(
        rank_view_roots_json={0: rank0_dir, 1: rank1_dir},
        output_dir=str(tmp_path / "out"),
        prompt_id=["synthetic_lower_upper_split_process"],
        prompt_set="base",
        layer_split=23,
        stream_lm_head=True,
        no_full_logits=True,
        lm_head_chunk_rows=1,
        top_k=1,
        max_positions=1,
        model_id="zai-org/GLM-4.5-Air",
        revision="test",
        teacher_kind="bf16_source",
        rank0_only_logits=True,
        local_sequential_stage_processes=True,
        local_sequential_head_process=True,
        local_sequential_lower_split_layer=12,
        local_sequential_upper_split_layer=34,
        local_sequential_stage_view_roots_json=stage_dirs,
        mlx_memory_limit_gb=None,
        mlx_cache_limit_gb=0,
        mlx_wired_limit_gb=None,
    )

    cli._run_local_sequential_export(
        args,
        mlx_memory_settings={
            "mlx_memory_limit_bytes": None,
            "mlx_cache_limit_bytes": 0,
            "mlx_wired_limit_bytes": None,
            "previous_mlx_memory_limit_bytes": None,
            "previous_mlx_cache_limit_bytes": None,
            "previous_mlx_wired_limit_bytes": None,
        },
    )

    worker_names = [
        arg_value(command, "--local-sequential-stage-worker")
        for command in commands
    ]
    assert worker_names == ["lower-pre", "lower-final", "upper-pre", "upper-final", "head"]
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["local_sequential_lower_split_layer"] == 12
    assert rows[0]["local_sequential_upper_split_layer"] == 34
    assert rows[0]["local_sequential_lower_pre_hidden_host_spill"] is True
    assert rows[0]["local_sequential_lower_pre_hidden_host_dtype"] == "bfloat16"
    assert rows[0]["local_sequential_lower_pre_hidden_host_shape"] == [1, 1, 1]
    assert rows[0]["local_sequential_lower_pre_hidden_host_nbytes"] == 4
    assert rows[0]["local_sequential_lower_hidden_host_spill"] is True
    assert rows[0]["local_sequential_upper_pre_hidden_host_spill"] is True
    assert rows[0]["local_sequential_upper_hidden_host_spill"] is True
    assert rows[0]["local_sequential_stage_view_roots"] == {
        "head": {"0": str(stage_dirs["head"][0])},
        "lower-final": {"1": str(stage_dirs["lower-final"][1])},
        "lower-pre": {"1": str(stage_dirs["lower-pre"][1])},
        "upper-final": {"0": str(stage_dirs["upper-final"][0])},
        "upper-pre": {"0": str(stage_dirs["upper-pre"][0])},
    }
    assert rows[0]["mlx_peak_bytes"] == 175
    assert [entry["stage"] for entry in rows[0]["rank_metrics"]] == [
        "local_sequential_lower_pre_process",
        "local_sequential_lower_final_process",
        "local_sequential_upper_pre_process",
        "local_sequential_upper_final_process",
        "local_sequential_head_process",
    ]


def test_local_sequential_multi_window_process_spawns_window_workers(
    tmp_path,
    monkeypatch,
) -> None:
    cli = _load_distributed_teacher_cache_cli()

    prompt = cli.QualityPrompt(
        prompt_id="synthetic_multi_window_process",
        text="synthetic",
        max_new_tokens=1,
    )
    rank0_dir = tmp_path / "rank0"
    rank1_dir = tmp_path / "rank1"
    rank0_dir.mkdir()
    rank1_dir.mkdir()
    stage_dirs = {
        "lower-embed": {1: tmp_path / "stage-lower-embed-rank1"},
        "lower-window-0": {1: tmp_path / "stage-lower-window-0-rank1"},
        "lower-window-1": {1: tmp_path / "stage-lower-window-1-rank1"},
        "lower-window-2": {1: tmp_path / "stage-lower-window-2-rank1"},
        "upper-window-0": {0: tmp_path / "stage-upper-window-0-rank0"},
        "upper-window-1": {0: tmp_path / "stage-upper-window-1-rank0"},
        "upper-window-2": {0: tmp_path / "stage-upper-window-2-rank0"},
        "head": {0: tmp_path / "stage-head-rank0"},
    }
    for roots in stage_dirs.values():
        for root in roots.values():
            root.mkdir()
    commands: list[list[str]] = []

    def arg_value(command: list[str], name: str) -> str:
        return command[command.index(name) + 1]

    def write_metrics(
        info_path: Path,
        *,
        peak: int,
        output_kind: str = "hidden",
        extra: dict[str, object] | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "input_token_ids": [3, 1],
            "output_kind": output_kind,
            "elapsed_seconds": peak / 100.0,
            "metrics": {
                "mlx_active_bytes": peak // 10,
                "mlx_peak_bytes": peak,
                "mlx_cache_bytes": 0,
                "rss_bytes": peak * 10,
                "pageouts_total": 1,
                "swapouts_total": 2,
                "pageouts_delta": 0,
                "swapouts_delta": 0,
            },
        }
        if extra:
            payload.update(extra)
        info_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    worker_output_prefixes = {
        "lower-embed": "local_sequential_lower_embed_hidden_host",
        "lower-window-0": "local_sequential_lower_window_0_hidden_host",
        "lower-window-1": "local_sequential_lower_window_1_hidden_host",
        "lower-window-2": "local_sequential_lower_hidden_host",
        "upper-window-0": "local_sequential_upper_window_0_hidden_host",
        "upper-window-1": "local_sequential_upper_window_1_hidden_host",
        "upper-window-2": "local_sequential_upper_hidden_host",
    }

    def hidden_extra(prefix: str) -> dict[str, object]:
        return {
            f"{prefix}_dtype": "bfloat16",
            f"{prefix}_shape": [1, 1, 1],
            f"{prefix}_nbytes": 4,
        }

    def fake_subprocess_run(command, *, check=False, text=False, capture_output=False):
        del check, text, capture_output
        command = list(command)
        commands.append(command)
        assert "--local-sequential-head-process" in command
        assert arg_value(command, "--local-sequential-lower-split-layers") == "1,2"
        assert arg_value(command, "--local-sequential-upper-split-layers") == "4,5"
        worker = arg_value(command, "--local-sequential-stage-worker")
        roots = {
            int(rank): Path(path)
            for rank, path in json.loads(arg_value(command, "--rank-view-roots-json")).items()
        }
        if worker == "lower-embed":
            assert roots[1] == stage_dirs[worker][1]
            assert roots[0] == rank0_dir
            assert "--local-sequential-worker-input-json" not in command
        elif worker.startswith("lower-window"):
            assert roots[1] == stage_dirs[worker][1]
            assert roots[0] == rank0_dir
            assert "--local-sequential-worker-input-json" in command
        elif worker.startswith("upper-window"):
            assert roots[0] == stage_dirs[worker][0]
            assert roots[1] == rank1_dir
        else:
            assert worker == "head"
            assert roots[0] == stage_dirs["head"][0]
            assert roots[1] == rank1_dir
        info_path = Path(arg_value(command, "--local-sequential-worker-json"))
        if worker == "head":
            assert (
                Path(arg_value(command, "--local-sequential-hidden-path")).name
                == "upper-window-2-hidden.npz"
            )
            output_path = Path(arg_value(command, "--local-sequential-output-path"))
            np.savez(
                output_path,
                topk_ids=np.array([[1]], dtype=np.int32),
                topk_logprobs=np.array([[-0.1]], dtype=np.float32),
                target_logprobs=np.array([-0.1], dtype=np.float32),
                teacher_top1_ids=np.array([1], dtype=np.int32),
            )
            write_metrics(info_path, peak=50, output_kind="topk")
            return cli.subprocess.CompletedProcess(command, 0, "", "")

        output_path = Path(arg_value(command, "--local-sequential-output-path"))
        np.savez(output_path, hidden=np.array([[[1.0]]], dtype=np.float32))
        write_metrics(
            info_path,
            peak={
                "lower-embed": 70,
                "lower-window-0": 80,
                "lower-window-1": 85,
                "lower-window-2": 90,
                "upper-window-0": 95,
                "upper-window-1": 100,
                "upper-window-2": 105,
            }[worker],
            extra=hidden_extra(worker_output_prefixes[worker]),
        )
        return cli.subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(cli, "_selected_prompts", lambda prompt_ids, *, prompt_set: [prompt])
    monkeypatch.setattr(
        cli,
        "_load_rank_model_and_tokenizer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("multi-window stage-process mode must not load rank models in parent")
        ),
    )
    monkeypatch.setattr(cli.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(cli, "_collect_vm_stat_counts", lambda: {"pageouts": 1, "swapouts": 2})
    monkeypatch.setattr(
        cli,
        "_collect_metric_snapshot",
        lambda previous_vm_stat_counts: {
            "mlx_active_bytes": 0,
            "mlx_peak_bytes": 0,
            "mlx_cache_bytes": 0,
            "rss_bytes": 0,
            "pageouts_total": 1,
            "swapouts_total": 2,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        },
    )
    monkeypatch.setattr(cli.mx, "reset_peak_memory", lambda: None)

    args = cli.argparse.Namespace(
        rank_view_roots_json={0: rank0_dir, 1: rank1_dir},
        output_dir=str(tmp_path / "out"),
        prompt_id=["synthetic_multi_window_process"],
        prompt_set="base",
        layer_split=3,
        stream_lm_head=True,
        no_full_logits=True,
        lm_head_chunk_rows=1,
        top_k=1,
        max_positions=1,
        model_id="zai-org/GLM-4.5-Air",
        revision="test",
        teacher_kind="bf16_source",
        rank0_only_logits=True,
        local_sequential_stage_processes=True,
        local_sequential_head_process=True,
        local_sequential_lower_split_layer=None,
        local_sequential_upper_split_layer=None,
        local_sequential_lower_split_layers=[1, 2],
        local_sequential_upper_split_layers=[4, 5],
        local_sequential_stage_view_roots_json=stage_dirs,
        mlx_memory_limit_gb=None,
        mlx_cache_limit_gb=0,
        mlx_wired_limit_gb=None,
    )

    cli._run_local_sequential_export(
        args,
        mlx_memory_settings={
            "mlx_memory_limit_bytes": None,
            "mlx_cache_limit_bytes": 0,
            "mlx_wired_limit_bytes": None,
            "previous_mlx_memory_limit_bytes": None,
            "previous_mlx_cache_limit_bytes": None,
            "previous_mlx_wired_limit_bytes": None,
        },
    )

    worker_names = [
        arg_value(command, "--local-sequential-stage-worker")
        for command in commands
    ]
    assert worker_names == [
        "lower-embed",
        "lower-window-0",
        "lower-window-1",
        "lower-window-2",
        "upper-window-0",
        "upper-window-1",
        "upper-window-2",
        "head",
    ]
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["local_sequential_lower_split_layers"] == [1, 2]
    assert rows[0]["local_sequential_upper_split_layers"] == [4, 5]
    assert rows[0]["local_sequential_lower_hidden_host_spill"] is True
    assert rows[0]["local_sequential_upper_hidden_host_spill"] is True
    assert rows[0]["local_sequential_stage_view_roots"]["lower-embed"] == {
        "1": str(stage_dirs["lower-embed"][1])
    }
    assert rows[0]["local_sequential_stage_view_roots"]["lower-window-2"] == {
        "1": str(stage_dirs["lower-window-2"][1])
    }
    assert rows[0]["mlx_peak_bytes"] == 105
    assert [entry["stage"] for entry in rows[0]["rank_metrics"]] == [
        "local_sequential_lower_embed_process",
        "local_sequential_lower_window_0_process",
        "local_sequential_lower_window_1_process",
        "local_sequential_lower_window_2_process",
        "local_sequential_upper_window_0_process",
        "local_sequential_upper_window_1_process",
        "local_sequential_upper_window_2_process",
        "local_sequential_head_process",
    ]


def test_local_sequential_dirty_stage_abort_reports_worker() -> None:
    cli = _load_distributed_teacher_cache_cli()

    with pytest.raises(SystemExit, match="worker=upper-window-4"):
        cli._raise_if_local_sequential_stage_dirty(
            "upper-window-4",
            {"pageouts_delta": 0, "swapouts_delta": 5},
        )


def test_local_sequential_remote_worker_rewrites_paths_and_stage_roots(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cli = _load_distributed_teacher_cache_cli()
    local_rank0 = tmp_path / "local-rank0"
    local_rank1 = tmp_path / "local-rank1"
    remote_rank0 = "/remote/views/upper-window-0-rank0"
    remote_rank1 = "/remote/views/rank1"
    local_rank0.mkdir()
    local_rank1.mkdir()
    hidden_path = tmp_path / "upper-window-0-input.npz"
    input_json = tmp_path / "lower-window-2.json"
    worker_json = tmp_path / "upper-window-0.json"
    output_path = tmp_path / "upper-window-0-hidden.npz"
    hidden_path.write_bytes(b"npz")
    input_json.write_text("{}", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_checked(command, *, description):
        del description
        calls.append(list(command))
        return cli.subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(cli, "_run_checked_subprocess", fake_checked)
    args = cli.argparse.Namespace(
        rank_view_roots_json={0: local_rank0, 1: local_rank1},
        local_sequential_remote_worker=["upper-window-*", "head"],
        local_sequential_remote_ssh="jackmazac@203.0.113.191",
        local_sequential_remote_ssh_option=["-o", "BatchMode=yes"],
        local_sequential_remote_repo="/Users/jackmazac/Development/mlx",
        local_sequential_remote_python="/Users/jackmazac/.local/bin/uv run python",
        local_sequential_remote_tmp_dir="/tmp/glm-remote-workers",
        local_sequential_remote_dirty_retries=0,
        local_sequential_remote_retry_sleep_seconds=0,
        local_sequential_remote_stage_view_roots_json={
            "upper-window-0": {0: remote_rank0, 1: remote_rank1}
        },
    )
    command = [
        "python",
        "/local/exporter.py",
        "--rank-view-roots-json",
        json.dumps({"0": str(local_rank0), "1": str(local_rank1)}),
        "--output-dir",
        str(tmp_path / "out"),
        "--layer-split",
        "3",
        "--rank0-only-logits",
        "--stream-lm-head",
        "--prompt-set",
        "base",
        "--top-k",
        "1",
        "--model-id",
        "zai-org/GLM-4.5-Air",
        "--revision",
        "test",
        "--teacher-kind",
        "bf16_source",
        "--lm-head-chunk-rows",
        "1",
        "--local-sequential-stage-worker",
        "upper-window-0",
        "--local-sequential-worker-prompt-id",
        "capital_france",
        "--local-sequential-hidden-path",
        str(hidden_path),
        "--local-sequential-worker-json",
        str(worker_json),
        "--local-sequential-window-rank",
        "0",
        "--local-sequential-window-start-layer",
        "23",
        "--local-sequential-window-output-prefix",
        "local_sequential_upper_window_0_hidden_host",
        "--local-sequential-output-path",
        str(output_path),
        "--local-sequential-worker-input-json",
        str(input_json),
        "--local-sequential-window-input-prefix",
        "local_sequential_lower_hidden_host",
    ]

    assert cli._local_sequential_worker_is_remote(args, "upper-window-0")
    assert cli._local_sequential_worker_is_remote(args, "head")
    assert not cli._local_sequential_worker_is_remote(args, "lower-window-0")

    cli._run_local_sequential_worker_for_args(
        args,
        worker="upper-window-0",
        command=command,
    )

    ssh_runs = [call for call in calls if call[0] == "ssh"]
    scp_runs = [call for call in calls if call[0] == "scp"]
    rsync_runs = [call for call in calls if call[0] == "rsync"]
    assert len(scp_runs) == 5
    assert len(rsync_runs) == 1
    assert ssh_runs[0][1:3] == ["-o", "BatchMode=yes"]
    assert scp_runs[0][1:3] == ["-o", "BatchMode=yes"]
    assert rsync_runs[0][3] == "ssh -o BatchMode=yes"
    remote_run = ssh_runs[-1]
    remote_shell = remote_run[-1]
    assert "cd /Users/jackmazac/Development/mlx" in remote_shell
    assert (
        "PYTHONPATH=/tmp/glm-remote-workers/capital_france/upper-window-0/src:"
        "/Users/jackmazac/Development/mlx/src:/Users/jackmazac/Development/mlx"
        in remote_shell
    )
    assert "/Users/jackmazac/.local/bin/uv run python" in remote_shell
    assert "/tmp/glm-remote-workers/capital_france/upper-window-0" in remote_shell
    assert str(local_rank0) not in remote_shell
    assert remote_rank0 in remote_shell
    assert remote_rank1 in remote_shell
    assert str(hidden_path) not in remote_shell
    assert "upper-window-0-input.npz" in remote_shell
    assert "upper-window-0-hidden.npz" in remote_shell


def test_local_sequential_remote_worker_retries_dirty_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cli = _load_distributed_teacher_cache_cli()
    rank0 = tmp_path / "rank0"
    rank1 = tmp_path / "rank1"
    rank0.mkdir()
    rank1.mkdir()
    hidden_path = tmp_path / "upper-window-19-input.npz"
    input_json = tmp_path / "upper-window-18.json"
    worker_json = tmp_path / "upper-window-19.json"
    output_path = tmp_path / "upper-window-19-hidden.npz"
    hidden_path.write_bytes(b"npz")
    input_json.write_text("{}", encoding="utf-8")
    worker_runs = 0
    cleanup_runs = 0

    def fake_checked(command, *, description):
        nonlocal worker_runs, cleanup_runs
        command = list(command)
        if description.startswith("remote local-sequential stage worker"):
            worker_runs += 1
        if description.startswith("remote dirty worker cleanup"):
            cleanup_runs += 1
        if description == "remote worker output fetch --local-sequential-worker-json":
            metrics = {
                "pageouts_delta": 5 if worker_runs == 1 else 0,
                "swapouts_delta": 0,
            }
            Path(command[-1]).write_text(
                json.dumps({"metrics": metrics}, sort_keys=True),
                encoding="utf-8",
            )
        if description == "remote worker output fetch --local-sequential-output-path":
            Path(command[-1]).write_bytes(b"npz")
        return cli.subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(cli, "_run_checked_subprocess", fake_checked)
    args = cli.argparse.Namespace(
        rank_view_roots_json={0: rank0, 1: rank1},
        local_sequential_remote_worker=["upper-window-*"],
        local_sequential_remote_ssh="jackmazac@203.0.113.191",
        local_sequential_remote_ssh_option=[],
        local_sequential_remote_repo="/Users/jackmazac/Development/mlx",
        local_sequential_remote_python="/Users/jackmazac/.local/bin/uv run python",
        local_sequential_remote_tmp_dir="/tmp/glm-remote-workers",
        local_sequential_remote_dirty_retries=1,
        local_sequential_remote_retry_sleep_seconds=0,
        local_sequential_remote_stage_view_roots_json={
            "upper-window-19": {0: "/remote/rank0"}
        },
        local_sequential_stage_memory_quiet_preflight=False,
    )
    command = [
        "python",
        "/local/exporter.py",
        "--rank-view-roots-json",
        json.dumps({"0": str(rank0), "1": str(rank1)}),
        "--output-dir",
        str(tmp_path / "out"),
        "--layer-split",
        "23",
        "--rank0-only-logits",
        "--stream-lm-head",
        "--prompt-set",
        "base",
        "--top-k",
        "16",
        "--model-id",
        "zai-org/GLM-4.5-Air",
        "--revision",
        "test",
        "--teacher-kind",
        "bf16_source",
        "--lm-head-chunk-rows",
        "8192",
        "--local-sequential-stage-worker",
        "upper-window-19",
        "--local-sequential-worker-prompt-id",
        "capital_france",
        "--local-sequential-hidden-path",
        str(hidden_path),
        "--local-sequential-worker-json",
        str(worker_json),
        "--local-sequential-window-rank",
        "0",
        "--local-sequential-window-start-layer",
        "42",
        "--local-sequential-window-output-prefix",
        "local_sequential_upper_window_19_hidden_host",
        "--local-sequential-output-path",
        str(output_path),
        "--local-sequential-worker-input-json",
        str(input_json),
        "--local-sequential-window-input-prefix",
        "local_sequential_upper_window_18_hidden_host",
    ]

    cli._run_local_sequential_worker_for_args(
        args,
        worker="upper-window-19",
        command=command,
    )

    assert worker_runs == 2
    assert cleanup_runs == 1
    final_info = json.loads(worker_json.read_text(encoding="utf-8"))
    assert final_info["metrics"] == {"pageouts_delta": 0, "swapouts_delta": 0}


def test_local_sequential_stage_memory_threshold_uses_worker_view_plan(
    tmp_path: Path,
) -> None:
    cli = _load_distributed_teacher_cache_cli()
    worker_root = tmp_path / "lower-window-2"
    worker_root.mkdir()
    (worker_root / "pipeline_stage_view_plan.json").write_text(
        json.dumps(
            {
                "visible_shard_file_bytes": 3 * 1024**3,
                "required_present_tensor_bytes": 2 * 1024**3,
            }
        ),
        encoding="utf-8",
    )
    args = cli.argparse.Namespace(
        local_sequential_stage_view_roots_json={"lower-window-2": {1: worker_root}},
        local_sequential_stage_memory_quiet_min_free_gb=0,
        local_sequential_stage_memory_quiet_stage_view_margin_gb=8,
    )

    effective_min, metadata = cli._local_sequential_stage_min_free_gb(
        args,
        "lower-window-2",
    )

    assert effective_min == 11
    assert metadata["stage_view_plan_count"] == 1
    assert metadata["stage_view_max_visible_shard_gb"] == 3
    assert metadata["stage_view_max_required_tensor_gb"] == 2
    assert metadata["stage_view_worker_found"] is True

    fallback_min, fallback_metadata = cli._local_sequential_stage_min_free_gb(
        args,
        "missing-worker",
    )

    assert fallback_min == 11
    assert fallback_metadata["stage_view_plan_count"] == 1
    assert fallback_metadata["stage_view_worker_found"] is False


def test_local_sequential_stage_available_memory_counts_inactive_pages() -> None:
    cli = _load_distributed_teacher_cache_cli()
    counts = {
        "page_size": 1024**3,
        "pages_free": 2,
        "pages_speculative": 1,
        "pages_inactive": 9,
        "pages_purgeable": 4,
    }

    assert cli._free_gb_from_vm_counts(counts) == 3
    assert cli._available_gb_from_vm_counts(counts) == 12
    assert cli._vm_page_gb(counts, "pages_purgeable") == 4


def test_local_sequential_checkpoint_reuses_only_clean_worker_outputs(
    tmp_path: Path,
) -> None:
    cli = _load_distributed_teacher_cache_cli()
    worker_json = tmp_path / "worker.json"
    output_path = tmp_path / "worker-output.npz"
    np.savez(output_path, hidden=np.array([1], dtype=np.float32))
    worker_json.write_text(
        json.dumps(
            {
                "checkpoint_schema_version": 2,
                "worker": "lower-window-4",
                "metrics": {"pageouts_delta": 0, "swapouts_delta": 0},
            }
        ),
        encoding="utf-8",
    )

    clean = cli._load_reusable_local_sequential_checkpoint(
        worker="lower-window-4",
        worker_json_path=worker_json,
        output_path=output_path,
    )

    assert clean is not None
    worker_json.write_text(
        json.dumps(
            {
                "checkpoint_schema_version": 2,
                "worker": "lower-window-4",
                "metrics": {"pageouts_delta": 1, "swapouts_delta": 0},
            }
        ),
        encoding="utf-8",
    )
    np.savez(output_path, hidden=np.array([1], dtype=np.float32))

    dirty = cli._load_reusable_local_sequential_checkpoint(
        worker="lower-window-4",
        worker_json_path=worker_json,
        output_path=output_path,
    )

    assert dirty is None
    assert not worker_json.exists()
    assert not output_path.exists()


def test_local_sequential_checkpoint_root_reuses_legacy_step_key_root(
    tmp_path: Path,
) -> None:
    cli = _load_distributed_teacher_cache_cli()
    steps_dir = tmp_path / "steps"
    legacy_root = steps_dir / ".local-sequential-checkpoints" / "prefix-step-aaaaaaaa"
    legacy_root.mkdir(parents=True)
    output_dir = steps_dir / "prefix-step-bbbbbbbb" / "out"
    output_dir.mkdir(parents=True)

    assert cli._local_sequential_checkpoint_root(output_dir) == legacy_root

    stable_root = steps_dir / ".local-sequential-checkpoints" / "prefix-step"
    stable_root.mkdir()

    assert cli._local_sequential_checkpoint_root(output_dir) == stable_root


def test_evaluate_teacher_cache_row_records_vq_teacher_top1_margin(tmp_path) -> None:
    row = {
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "bf16_source",
        "prompt_id": "margin",
        "input_token_ids": [99, 0, 1],
        "target_token_ids": [0, 1],
        "positions": [0, 1],
        "target_logprobs": [-0.2, -0.3],
        "teacher_top1_ids": [2, 1],
        "full_logits_available": False,
    }
    vq_logits = np.array(
        [
            [0.0, 1.0, 2.0, 4.0],
            [0.0, 5.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    vq_log_probs = _log_softmax(vq_logits)

    record = evaluate_teacher_cache_row(row, vq_logits=mx.array(vq_logits), cache_root=tmp_path)

    assert record["vq_top1_ids"] == [3, 1]
    np.testing.assert_allclose(
        record["vq_teacher_top1_logprobs"],
        [vq_log_probs[0, 2], vq_log_probs[1, 1]],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        record["vq_top1_logprobs"],
        [vq_log_probs[0, 3], vq_log_probs[1, 1]],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        record["vq_teacher_top1_margin_vs_vq_top1"],
        [vq_log_probs[0, 2] - vq_log_probs[0, 3], 0.0],
        rtol=1e-6,
    )


def test_evaluate_teacher_cache_row_records_watched_token_margins(tmp_path) -> None:
    row = {
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "bf16_source",
        "prompt_id": "watched_tokens",
        "input_token_ids": [99, 0, 1],
        "target_token_ids": [0, 1],
        "positions": [0, 1],
        "target_logprobs": [-0.2, -0.3],
        "teacher_top1_ids": [2, 1],
        "full_logits_available": False,
    }
    vq_logits = np.array(
        [
            [0.0, 1.0, 2.0, 4.0],
            [0.0, 5.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    vq_log_probs = _log_softmax(vq_logits)

    record = evaluate_teacher_cache_row(
        row,
        vq_logits=mx.array(vq_logits),
        cache_root=tmp_path,
        watch_token_ids=[2, 0],
    )

    assert record["vq_watch_token_ids"] == [2, 0]
    np.testing.assert_allclose(
        record["vq_watch_token_logprobs"]["2"],
        [vq_log_probs[0, 2], vq_log_probs[1, 2]],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        record["vq_watch_token_margin_vs_vq_top1"]["2"],
        [vq_log_probs[0, 2] - vq_log_probs[0, 3], vq_log_probs[1, 2] - vq_log_probs[1, 1]],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        record["vq_watch_token_required_bias_to_vq_top1"]["2"],
        [vq_log_probs[0, 3] - vq_log_probs[0, 2], vq_log_probs[1, 1] - vq_log_probs[1, 2]],
        rtol=1e-6,
    )


def test_teacher_cache_payload_rejects_invalid_limits() -> None:
    logits = np.ones((2, 3), dtype=np.float32)
    common = {
        "logits": logits,
        "input_token_ids": [0, 1, 2],
        "prompt_id": "invalid_limits",
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "bf16_source",
        "shard_path": "teacher_logits/invalid_limits.safetensors",
    }

    with pytest.raises(ValueError, match="top_k must be positive"):
        build_teacher_cache_payload(**common, top_k=0)
    with pytest.raises(ValueError, match="max_positions must be positive"):
        build_teacher_cache_payload(**common, max_positions=0)


def test_teacher_cache_export_cli_rejects_invalid_limits_before_load(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cases = [
        (["--top-k", "0"], "--top-k must be positive"),
        (["--max-positions", "0"], "--max-positions must be positive"),
    ]
    for extra_args, expected_error in cases:
        completed = subprocess.run(
            [
                sys.executable,
                "benchmarks/export_glm45_air_teacher_cache.py",
                "--output-dir",
                str(tmp_path / "cache"),
                *extra_args,
            ],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )

        assert completed.returncode == 2
        assert expected_error in completed.stderr


def test_teacher_cache_export_cli_source_memory_guard_blocks_before_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_export_teacher_cache_cli()
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 300}, "weight_map": {}}),
        encoding="utf-8",
    )

    def fail_load(*args, **kwargs):
        raise AssertionError("source memory guard should run before mlx_lm.load")

    monkeypatch.setattr(cli, "load", fail_load)
    monkeypatch.setattr(cli, "_local_physical_memory_bytes", lambda: 100)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_glm45_air_teacher_cache.py",
            "--model-path",
            str(source_dir),
            "--output-dir",
            str(tmp_path / "cache"),
            "--prompt-id",
            "capital_france",
            "--source-memory-guard-ratio",
            "1.0",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 23
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"] == "single_host_source_memory_guard_blocked"
    assert payload["source_memory_guard_pass"] is False
    assert payload["expected_source_to_physical_memory_ratio"] == 3.0
    assert not (tmp_path / "cache" / "metadata.jsonl").exists()


def test_teacher_cache_export_source_memory_guard_uses_present_shard_bytes(tmp_path: Path) -> None:
    cli = _load_export_teacher_cache_cli()
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "model-00001-of-00002.safetensors").write_bytes(b"x" * 120)
    (source_dir / "model-00002-of-00002.safetensors").write_bytes(b"y" * 90)
    (source_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 100},
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "lm_head.weight": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    record = cli.source_memory_guard_record(
        model_path=str(source_dir),
        source_memory_guard_ratio=1.0,
        physical_memory_bytes=200,
    )

    assert record["decision"] == "single_host_source_memory_guard_blocked"
    assert record["source_index_total_size"] == 100
    assert record["present_shard_bytes"] == 210
    assert record["guard_source_bytes"] == 210
    assert record["expected_source_to_physical_memory_ratio"] == 1.05


def test_teacher_cache_export_cli_can_apply_mlx_memory_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    cli = _load_export_teacher_cache_cli()
    calls: list[tuple[str, int]] = []

    class FakeMx:
        @staticmethod
        def set_cache_limit(value: int) -> int:
            calls.append(("cache", value))
            return 11

        @staticmethod
        def set_wired_limit(value: int) -> int:
            calls.append(("wired", value))
            return 22

    monkeypatch.setattr(cli, "mx", FakeMx)

    record = cli.configure_mlx_memory_limits(
        mlx_cache_limit_gb=0.5,
        mlx_wired_limit_gb=1.25,
    )

    assert calls == [
        ("cache", 536870912),
        ("wired", 1342177280),
    ]
    assert record["mlx_cache_limit_bytes"] == 536870912
    assert record["previous_mlx_cache_limit_bytes"] == 11
    assert record["mlx_wired_limit_bytes"] == 1342177280
    assert record["previous_mlx_wired_limit_bytes"] == 22


def test_read_teacher_cache_rows_rejects_empty_file(tmp_path) -> None:
    metadata = tmp_path / "teacher.jsonl"
    metadata.write_text("\n", encoding="utf-8")

    try:
        read_teacher_cache_rows(metadata)
    except ValueError as error:
        assert "contained no rows" in str(error)
    else:
        raise AssertionError("empty teacher cache metadata should fail")


def test_read_teacher_cache_rows_reports_invalid_json_line_number(tmp_path) -> None:
    metadata = tmp_path / "teacher.jsonl"
    metadata.write_text('{"prompt_id": "a"}\n{"prompt_id":\n', encoding="utf-8")

    with pytest.raises(ValueError, match="teacher cache row 2 is invalid JSON"):
        read_teacher_cache_rows(metadata)


def test_read_teacher_cache_rows_parses_jsonl(tmp_path) -> None:
    metadata = tmp_path / "teacher.jsonl"
    metadata.write_text('{"prompt_id": "a"}\n{"prompt_id": "b"}\n', encoding="utf-8")

    rows = read_teacher_cache_rows(metadata)

    assert [row["prompt_id"] for row in rows] == ["a", "b"]


def test_validate_teacher_cache_row_accepts_full_logits_without_loading_vq(tmp_path) -> None:
    mx.save_safetensors(
        str(tmp_path / "teacher.safetensors"),
        {
            "logits": mx.array(np.ones((2, 4), dtype=np.float16)),
            "topk_ids": mx.array(np.array([[0, 1], [2, 3]], dtype=np.int32)),
            "topk_logprobs": mx.array(np.array([[-0.1, -0.2], [-0.3, -0.4]], dtype=np.float32)),
            "target_logprobs": mx.array(np.array([-0.1, -0.3], dtype=np.float32)),
            "teacher_top1_ids": mx.array(np.array([0, 2], dtype=np.int32)),
        },
    )
    row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "bf16_source",
        "prompt_id": "full",
        "input_token_ids": [0, 1, 2],
        "target_token_ids": [1, 2],
        "positions": [0, 1],
        "logit_shard": "teacher.safetensors",
        "logit_tensor": "logits",
        "topk_tensor": "topk_ids",
        "topk_logprob_tensor": "topk_logprobs",
        "target_logprob_tensor": "target_logprobs",
        "teacher_top1_tensor": "teacher_top1_ids",
        "full_logits_available": True,
    })

    report = validate_teacher_cache_row(row, cache_root=tmp_path, row_index=0, min_top_k=2)

    assert report["ok"] is True
    assert report["errors"] == []
    assert report["kld_mode"] == "exact_full_logits"
    assert report["full_logits_available"] is True
    assert report["position_count"] == 2
    assert report["tensors"]["logits"]["shape"] == [2, 4]
    assert report["tensors"]["topk_ids"]["shape"] == [2, 2]


def test_validate_teacher_cache_row_accepts_topk_only_fallback(tmp_path) -> None:
    mx.save_safetensors(
        str(tmp_path / "teacher.safetensors"),
        {
            "topk_ids": mx.array(np.array([[0, 1], [2, 3]], dtype=np.int32)),
            "topk_logprobs": mx.array(np.array([[-0.1, -0.2], [-0.3, -0.4]], dtype=np.float32)),
            "target_logprobs": mx.array(np.array([-0.1, -0.3], dtype=np.float32)),
            "teacher_top1_ids": mx.array(np.array([0, 2], dtype=np.int32)),
        },
    )
    row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "q8",
        "prompt_id": "topk",
        "input_token_ids": [10, 11, 12],
        "target_token_ids": [11, 12],
        "positions": [0, 1],
        "logit_shard": "teacher.safetensors",
        "topk_tensor": "topk_ids",
        "topk_logprob_tensor": "topk_logprobs",
        "target_logprob_tensor": "target_logprobs",
        "teacher_top1_tensor": "teacher_top1_ids",
        "full_logits_available": False,
    })

    report = validate_teacher_cache_row(row, cache_root=tmp_path, row_index=0, min_top_k=2)

    assert report["ok"] is True
    assert report["errors"] == []
    assert report["kld_mode"] == "teacher_topk_lower_bound"
    assert report["full_logits_available"] is False


def test_validate_teacher_cache_row_reports_missing_tensor(tmp_path) -> None:
    mx.save_safetensors(
        str(tmp_path / "teacher.safetensors"),
        {"target_logprobs": mx.array(np.array([-0.1, -0.3], dtype=np.float32))},
    )
    row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "q8",
        "prompt_id": "missing",
        "input_token_ids": [10, 11, 12],
        "target_token_ids": [11, 12],
        "positions": [0, 1],
        "logit_shard": "teacher.safetensors",
        "topk_tensor": "topk_ids",
        "topk_logprob_tensor": "topk_logprobs",
        "target_logprob_tensor": "target_logprobs",
        "full_logits_available": False,
    })

    report = validate_teacher_cache_row(row, cache_root=tmp_path, row_index=0, min_top_k=2)

    assert report["ok"] is False
    assert any("topk_ids" in error for error in report["errors"])
    assert any("topk_logprobs" in error for error in report["errors"])


def test_validate_teacher_cache_metadata_cli_appends_summary(tmp_path) -> None:
    mx.save_safetensors(
        str(tmp_path / "teacher.safetensors"),
        {
            "logits": mx.array(np.ones((1, 3), dtype=np.float16)),
            "topk_ids": mx.array(np.array([[0, 1, 2]], dtype=np.int32)),
            "topk_logprobs": mx.array(
                np.log(np.array([[0.5, 0.3, 0.2]], dtype=np.float32))
            ),
            "target_logprobs": mx.array(np.log(np.array([0.3], dtype=np.float32))),
            "teacher_top1_ids": mx.array(np.array([0], dtype=np.int32)),
        },
    )
    row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "bf16_source",
        "prompt_id": "cli",
        "input_token_ids": [0, 1],
        "target_token_ids": [1],
        "positions": [0],
        "logit_shard": "teacher.safetensors",
        "logit_tensor": "logits",
        "topk_tensor": "topk_ids",
        "topk_logprob_tensor": "topk_logprobs",
        "target_logprob_tensor": "target_logprobs",
        "teacher_top1_tensor": "teacher_top1_ids",
        "full_logits_available": True,
    })
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(json.dumps(row) + "\n", encoding="utf-8")
    append_jsonl = tmp_path / "summary.jsonl"

    summary = validate_teacher_cache_metadata(metadata, cache_root=tmp_path, check_values=True)
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/validate_glm45_air_teacher_cache.py",
            "--teacher-jsonl",
            str(metadata),
            "--cache-root",
            str(tmp_path),
            "--check-values",
            "--append-jsonl",
            str(append_jsonl),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    cli_summary = json.loads(completed.stdout)
    appended = json.loads(append_jsonl.read_text(encoding="utf-8"))
    assert summary["ok"] is True
    assert summary["check_values"] is True
    assert cli_summary["ok"] is True
    assert cli_summary["check_values"] is True
    assert appended["row_count"] == 1
    assert appended["ok_count"] == 1
    assert appended["memory_counter_row_count"] == 1
    assert appended["memory_clean_row_count"] == 1


def test_validate_teacher_cache_metadata_can_check_only_selected_rows(tmp_path) -> None:
    mx.save_safetensors(
        str(tmp_path / "teacher.safetensors"),
        {
            "logits": mx.array(np.ones((1, 3), dtype=np.float16)),
            "topk_ids": mx.array(np.array([[0, 1, 2]], dtype=np.int32)),
            "topk_logprobs": mx.array(
                np.log(np.array([[0.5, 0.3, 0.2]], dtype=np.float32))
            ),
            "target_logprobs": mx.array(np.log(np.array([0.3], dtype=np.float32))),
            "teacher_top1_ids": mx.array(np.array([0], dtype=np.int32)),
        },
    )
    valid_row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "bf16_source",
        "prompt_id": "valid",
        "input_token_ids": [0, 1],
        "target_token_ids": [1],
        "positions": [0],
        "logit_shard": "teacher.safetensors",
        "logit_tensor": "logits",
        "topk_tensor": "topk_ids",
        "topk_logprob_tensor": "topk_logprobs",
        "target_logprob_tensor": "target_logprobs",
        "teacher_top1_tensor": "teacher_top1_ids",
        "full_logits_available": True,
    })
    invalid_row = dict(valid_row, prompt_id="invalid", logit_shard="missing.safetensors")
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(
        json.dumps(valid_row) + "\n" + json.dumps(invalid_row) + "\n",
        encoding="utf-8",
    )

    summary = validate_teacher_cache_metadata(
        metadata,
        cache_root=tmp_path,
        check_values=True,
        row_indices=(0,),
    )

    assert summary["ok"] is True
    assert summary["row_count"] == 1
    assert summary["source_row_count"] == 2
    assert summary["selected_row_indices"] == [0]
    assert summary["rows"][0]["row_index"] == 0


def test_eval_teacher_cache_cli_prevalidates_before_loading_resident_vq(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    eval_teacher_cache_cli = _load_eval_teacher_cache_cli()
    row = {
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "bf16_source",
        "prompt_id": "invalid_prevalidation",
        "input_token_ids": [0, 1],
        "target_token_ids": [1],
        "positions": [0],
        "logit_shard": "teacher.safetensors",
        "topk_ids": [[0, 1, 2]],
        "topk_logprobs": [[-0.1, -0.2, -0.3]],
        "target_logprobs": [-0.1],
        "teacher_top1_ids": [0],
        "full_logits_available": False,
    }
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(json.dumps(row) + "\n", encoding="utf-8")

    def fail_load(*args, **kwargs):
        raise AssertionError("resident VQ should not load before cache validation")

    monkeypatch.setattr(eval_teacher_cache_cli, "load_resident_air", fail_load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_glm45_air_teacher_cache.py",
            "--teacher-jsonl",
            str(metadata),
            "--cache-root",
            str(tmp_path),
            "--min-top-k",
            "2",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        eval_teacher_cache_cli.main()

    captured = capsys.readouterr()
    assert exit_info.value.code == 1
    assert "teacher_cache_validation_summary" in captured.out
    assert "elapsed_seconds must be recorded" in captured.out


@pytest.mark.parametrize("max_rows", ["0", "-1"])
def test_eval_teacher_cache_cli_rejects_non_positive_max_rows_before_work(
    tmp_path,
    monkeypatch,
    capsys,
    max_rows,
) -> None:
    eval_teacher_cache_cli = _load_eval_teacher_cache_cli()

    def fail_validation(*args, **kwargs):
        raise AssertionError("cache validation should not run for invalid --max-rows")

    def fail_load(*args, **kwargs):
        raise AssertionError("resident VQ should not load for invalid --max-rows")

    monkeypatch.setattr(
        eval_teacher_cache_cli,
        "validate_teacher_cache_metadata",
        fail_validation,
    )
    monkeypatch.setattr(eval_teacher_cache_cli, "load_resident_air", fail_load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_glm45_air_teacher_cache.py",
            "--teacher-jsonl",
            str(tmp_path / "missing.jsonl"),
            "--max-rows",
            max_rows,
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        eval_teacher_cache_cli.main()

    captured = capsys.readouterr()
    assert exit_info.value.code == 2
    assert "--max-rows must be positive when provided" in captured.err


def test_eval_teacher_cache_cli_selects_original_row_index() -> None:
    eval_teacher_cache_cli = _load_eval_teacher_cache_cli()
    rows = [
        {"prompt_id": "first"},
        {"prompt_id": "second"},
        {"prompt_id": "third"},
    ]

    selected = eval_teacher_cache_cli._select_teacher_rows(
        rows,
        max_rows=None,
        row_index=1,
        row_indices=None,
    )

    assert selected == [(1, rows[1])]


def test_eval_teacher_cache_cli_selects_original_row_indices() -> None:
    eval_teacher_cache_cli = _load_eval_teacher_cache_cli()
    rows = [
        {"prompt_id": "first"},
        {"prompt_id": "second"},
        {"prompt_id": "third"},
    ]

    selected = eval_teacher_cache_cli._select_teacher_rows(
        rows,
        max_rows=None,
        row_index=None,
        row_indices=(2, 0),
    )

    assert selected == [(2, rows[2]), (0, rows[0])]


def test_eval_teacher_cache_cli_parses_watch_token_ids() -> None:
    eval_teacher_cache_cli = _load_eval_teacher_cache_cli()

    assert eval_teacher_cache_cli._parse_token_ids("565, 220,82") == (565, 220, 82)

    with pytest.raises(ValueError, match="must include at least one token"):
        eval_teacher_cache_cli._parse_token_ids("")
    with pytest.raises(ValueError, match="zero or greater"):
        eval_teacher_cache_cli._parse_token_ids("565,-1")


def test_eval_teacher_cache_cli_parses_and_applies_logit_biases() -> None:
    eval_teacher_cache_cli = _load_eval_teacher_cache_cli()

    biases = eval_teacher_cache_cli._parse_logit_biases(["565:2.589", "220:-0.25,82:0.5"])
    assert biases == ((565, 2.589), (220, -0.25), (82, 0.5))

    logits = mx.array(np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32))
    biased = eval_teacher_cache_cli._apply_logit_biases(logits, ((1, 0.5), (2, -1.0)))

    np.testing.assert_allclose(
        np.array(biased),
        np.array([[0.0, 1.5, 1.0], [3.0, 4.5, 4.0]], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="TOKEN:BIAS"):
        eval_teacher_cache_cli._parse_logit_biases(["565"])
    with pytest.raises(ValueError, match="outside logits vocab size"):
        eval_teacher_cache_cli._apply_logit_biases(logits, ((3, 1.0),))


def test_eval_teacher_cache_cli_prefers_mlx_clear_cache_between_rows(monkeypatch) -> None:
    eval_teacher_cache_cli = _load_eval_teacher_cache_cli()
    calls: list[str] = []

    class FakeMetal:
        @staticmethod
        def clear_cache() -> None:
            calls.append("metal")

    monkeypatch.setattr(
        eval_teacher_cache_cli,
        "mx",
        type(
            "FakeMX",
            (),
            {
                "clear_cache": staticmethod(lambda: calls.append("mx")),
                "metal": FakeMetal,
            },
        ),
    )

    eval_teacher_cache_cli._clear_mlx_caches()

    assert calls == ["mx"]


def test_eval_teacher_cache_cli_falls_back_to_metal_clear_cache(monkeypatch) -> None:
    eval_teacher_cache_cli = _load_eval_teacher_cache_cli()
    calls: list[str] = []

    class FakeMetal:
        @staticmethod
        def clear_cache() -> None:
            calls.append("metal")

    monkeypatch.setattr(
        eval_teacher_cache_cli,
        "mx",
        type("FakeMX", (), {"metal": FakeMetal}),
    )

    eval_teacher_cache_cli._clear_mlx_caches()

    assert calls == ["metal"]


def test_eval_teacher_cache_cli_applies_mlx_memory_policy(monkeypatch) -> None:
    eval_teacher_cache_cli = _load_eval_teacher_cache_cli()
    calls: list[tuple[str, int | None]] = []

    class FakeMX:
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

    monkeypatch.setattr(eval_teacher_cache_cli, "mx", FakeMX())
    args = type(
        "Args",
        (),
        {
            "mlx_cache_limit_gb": 0.0,
            "mlx_memory_limit_gb": 2.0,
            "mlx_wired_limit_gb": 3.0,
            "mlx_clear_cache_before_load": True,
        },
    )()

    policy = eval_teacher_cache_cli._apply_mlx_memory_policy(args)

    assert policy["enabled"] is True
    assert policy["cache_limit_bytes"] == 0
    assert policy["memory_limit_bytes"] == 2 * 1024**3
    assert policy["wired_limit_bytes"] == 3 * 1024**3
    assert policy["previous_cache_limit_bytes"] == 999
    assert policy["previous_memory_limit_bytes"] == 888
    assert policy["previous_wired_limit_bytes"] == 777
    assert policy["clear_cache_before_load"] is True
    assert policy["cache_bytes_before_clear"] == 123
    assert policy["cache_bytes_after_clear"] == 0
    assert calls == [
        ("cache", 0),
        ("memory", 2 * 1024**3),
        ("wired", 3 * 1024**3),
        ("get_cache", None),
        ("clear", None),
        ("get_cache", None),
    ]


def test_eval_teacher_cache_cli_appends_memory_trace_records(tmp_path, monkeypatch) -> None:
    eval_teacher_cache_cli = _load_eval_teacher_cache_cli()
    trace_path = tmp_path / "trace.jsonl"

    monkeypatch.setattr(
        eval_teacher_cache_cli,
        "collect_metric_snapshot",
        lambda previous_vm_stat_counts=None: {
            "mlx_active_bytes": 10,
            "mlx_peak_bytes": 20,
            "mlx_cache_bytes": 3,
            "rss_bytes": 40,
            "pageouts_total": 7,
            "swapouts_total": 8,
            "pageouts_delta": 2,
            "swapouts_delta": 0,
        },
    )

    eval_teacher_cache_cli._append_memory_trace(
        str(trace_path),
        stage="after_row_record",
        previous_vm_stat_counts={"pageouts": 5, "swapouts": 8},
        row_index=12,
        rows_completed=3,
        extra={"prompt_id": "report_route_000"},
    )

    record = json.loads(trace_path.read_text(encoding="utf-8"))
    assert record["stage"] == "after_row_record"
    assert record["row_index"] == 12
    assert record["rows_completed"] == 3
    assert record["prompt_id"] == "report_route_000"
    assert record["pageouts_delta"] == 2
    assert record["mlx_active_bytes"] == 10


def test_eval_teacher_cache_cli_summarizes_process_memory_cleanliness(monkeypatch) -> None:
    eval_teacher_cache_cli = _load_eval_teacher_cache_cli()

    monkeypatch.setattr(
        eval_teacher_cache_cli,
        "collect_metric_snapshot",
        lambda previous_vm_stat_counts=None: {
            "mlx_active_bytes": 10,
            "mlx_peak_bytes": 20,
            "mlx_cache_bytes": 3,
            "rss_bytes": 40,
            "pageouts_total": 7,
            "swapouts_total": 8,
            "pageouts_delta": 2,
            "swapouts_delta": 0,
        },
    )

    summary = eval_teacher_cache_cli._summarize_process_memory(
        previous_vm_stat_counts={"pageouts": 5, "swapouts": 8}
    )

    assert summary["process_memory_clean"] is False
    assert summary["process_pageouts_delta"] == 2
    assert summary["process_swapouts_delta"] == 0
    assert summary["process_mlx_active_bytes"] == 10
    assert summary["process_mlx_peak_bytes"] == 20
    assert summary["process_mlx_cache_bytes"] == 3
    assert summary["process_rss_bytes"] == 40


def test_eval_teacher_cache_cli_can_truncate_input_to_selected_position_prefix() -> None:
    eval_teacher_cache_cli = _load_eval_teacher_cache_cli()
    input_token_ids = list(range(1024))
    positions = list(range(128))

    truncated = eval_teacher_cache_cli._truncate_input_to_selected_position_prefix(
        input_token_ids,
        positions,
    )

    assert truncated == list(range(129))


def test_eval_teacher_cache_cli_can_save_selected_vq_logits(tmp_path) -> None:
    eval_teacher_cache_cli = _load_eval_teacher_cache_cli()
    logits = mx.array(np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]], dtype=np.float32))

    metadata = eval_teacher_cache_cli._save_vq_logits_shard(
        output_dir=tmp_path,
        row_index=7,
        row={"prompt_id": "route/000:pos0"},
        selected_logits=logits,
    )

    assert metadata["vq_logit_shard"] == "row-00007-route_000_pos0.safetensors"
    assert metadata["vq_logit_tensor"] == "vq_logits"
    assert metadata["vq_logit_shape"] == [2, 3]
    arrays = mx.load(str(tmp_path / metadata["vq_logit_shard"]))
    np.testing.assert_allclose(np.array(arrays["vq_logits"]), np.array(logits, dtype=np.float16))


def test_replay_saved_vq_logits_applies_position_scoped_logit_bias(
    tmp_path,
    monkeypatch,
) -> None:
    replay_cli = _load_replay_saved_vq_logits_cli()

    teacher_logits = np.array(
        [
            [0.0, 4.0, 1.0],
            [0.0, 1.0, 4.0],
        ],
        dtype=np.float32,
    )
    vq_logits = np.array(
        [
            [0.0, 2.0, 3.0],
            [0.0, 1.0, 4.0],
        ],
        dtype=np.float32,
    )
    mx.save_safetensors(str(tmp_path / "teacher.safetensors"), {"logits": mx.array(teacher_logits)})
    teacher_log_probs = _log_softmax(teacher_logits)
    teacher_row = _with_memory_counters(
        {
            "schema_version": 1,
            "model_id": "zai-org/GLM-4.5-Air",
            "revision": "test",
            "teacher_kind": "bf16_source",
            "prompt_id": "route_000",
            "input_token_ids": [0, 1, 2],
            "target_token_ids": [1, 2],
            "positions": [0, 1],
            "logit_shard": "teacher.safetensors",
            "logit_tensor": "logits",
            "topk_ids": [[1, 2], [2, 1]],
            "topk_logprobs": [
                [float(teacher_log_probs[0, 1]), float(teacher_log_probs[0, 2])],
                [float(teacher_log_probs[1, 2]), float(teacher_log_probs[1, 1])],
            ],
            "target_logprobs": [
                float(teacher_log_probs[0, 1]),
                float(teacher_log_probs[1, 2]),
            ],
            "teacher_top1_ids": [1, 2],
            "full_logits_available": True,
        }
    )
    teacher_jsonl = tmp_path / "teacher.jsonl"
    teacher_jsonl.write_text(json.dumps(teacher_row) + "\n", encoding="utf-8")

    logits_root = tmp_path / "student-logits"
    logits_root.mkdir()
    mx.save_safetensors(str(logits_root / "route_000.safetensors"), {"vq_logits": mx.array(vq_logits)})
    student_jsonl = tmp_path / "student.jsonl"
    student_jsonl.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "prompt_id": "route_000",
                "row_index": 0,
                "vq_logit_shard": "route_000.safetensors",
                "vq_logit_tensor": "vq_logits",
                "pageouts_delta": 0,
                "swapouts_delta": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with student_jsonl.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "record_type": "air_vq_teacher_cache_eval_summary",
                    "process_memory_clean": True,
                    "acceptance_memory_clean": True,
                }
            )
            + "\n"
        )

    artifact_dir = tmp_path / "artifact"
    sidecar = write_logit_bias_sidecar(
        output_dir=artifact_dir,
        token_biases={1: 2.0},
        position_indices=(0,),
    )
    write_logit_bias_artifact_manifest(
        seed_artifact_dir=tmp_path / "seed",
        output_dir=artifact_dir,
        sidecar=sidecar,
        run_manifest={"kind": "test"},
    )

    monkeypatch.setattr(replay_cli, "collect_vm_stat_counts", lambda: {"pageouts": 0, "swapouts": 0})
    monkeypatch.setattr(
        replay_cli,
        "collect_metric_snapshot",
        lambda previous_vm_stat_counts: {
            "elapsed_seconds": 0.01,
            "mlx_active_bytes": 1,
            "mlx_peak_bytes": 1,
            "mlx_cache_bytes": 0,
            "rss_bytes": 1,
            "pageouts_total": 0,
            "swapouts_total": 0,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        },
    )
    append_jsonl = tmp_path / "replay.jsonl"

    summary = replay_cli.replay_saved_vq_logits(
        teacher_jsonl=teacher_jsonl,
        student_jsonl=student_jsonl,
        append_jsonl_path=append_jsonl,
        cache_root=tmp_path,
        student_logits_root=logits_root,
        artifact_dir=artifact_dir,
        watch_token_ids=(1,),
        min_top_k=2,
    )

    rows = [json.loads(line) for line in append_jsonl.read_text(encoding="utf-8").splitlines()]
    record = rows[0]
    assert record["vq_top1_ids"] == [1, 2]
    assert record["top1_agreement"] == 1.0
    assert record["logit_bias_sidecar"]["position_indices"] == [0]
    assert record["logit_bias_sidecar"]["applied_selected_position_count"] == 1
    assert record["source_student_row_memory_clean"] is True
    assert summary["source_vq_logits_memory_clean"] is True
    assert summary["acceptance_memory_clean"] is True
    assert rows[-1]["engine"] == "saved_vq_logits_replay"


def test_replay_saved_vq_logits_rejects_dirty_source_for_acceptance(
    tmp_path,
    monkeypatch,
) -> None:
    replay_cli = _load_replay_saved_vq_logits_cli()

    teacher_logits = np.array([[0.0, 4.0, 1.0]], dtype=np.float32)
    vq_logits = np.array([[0.0, 4.0, 1.0]], dtype=np.float32)
    mx.save_safetensors(str(tmp_path / "teacher.safetensors"), {"logits": mx.array(teacher_logits)})
    teacher_log_probs = _log_softmax(teacher_logits)
    teacher_row = _with_memory_counters(
        {
            "schema_version": 1,
            "model_id": "zai-org/GLM-4.5-Air",
            "revision": "test",
            "teacher_kind": "bf16_source",
            "prompt_id": "route_000",
            "input_token_ids": [0, 1],
            "target_token_ids": [1],
            "positions": [0],
            "logit_shard": "teacher.safetensors",
            "logit_tensor": "logits",
            "topk_ids": [[1, 2]],
            "topk_logprobs": [[float(teacher_log_probs[0, 1]), float(teacher_log_probs[0, 2])]],
            "target_logprobs": [float(teacher_log_probs[0, 1])],
            "teacher_top1_ids": [1],
            "full_logits_available": True,
        }
    )
    teacher_jsonl = tmp_path / "teacher.jsonl"
    teacher_jsonl.write_text(json.dumps(teacher_row) + "\n", encoding="utf-8")

    logits_root = tmp_path / "student-logits"
    logits_root.mkdir()
    mx.save_safetensors(str(logits_root / "route_000.safetensors"), {"vq_logits": mx.array(vq_logits)})
    student_jsonl = tmp_path / "student.jsonl"
    student_jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "schema_version": 1,
                        "prompt_id": "route_000",
                        "row_index": 0,
                        "vq_logit_shard": "route_000.safetensors",
                        "vq_logit_tensor": "vq_logits",
                        "pageouts_delta": 7,
                        "swapouts_delta": 0,
                    }
                ),
                json.dumps(
                    {
                        "record_type": "air_vq_teacher_cache_eval_summary",
                        "process_memory_clean": False,
                        "acceptance_memory_clean": False,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(replay_cli, "collect_vm_stat_counts", lambda: {"pageouts": 0, "swapouts": 0})
    monkeypatch.setattr(
        replay_cli,
        "collect_metric_snapshot",
        lambda previous_vm_stat_counts: {
            "elapsed_seconds": 0.01,
            "mlx_active_bytes": 1,
            "mlx_peak_bytes": 1,
            "mlx_cache_bytes": 0,
            "rss_bytes": 1,
            "pageouts_total": 0,
            "swapouts_total": 0,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        },
    )

    summary = replay_cli.replay_saved_vq_logits(
        teacher_jsonl=teacher_jsonl,
        student_jsonl=student_jsonl,
        append_jsonl_path=tmp_path / "replay.jsonl",
        cache_root=tmp_path,
        student_logits_root=logits_root,
        min_top_k=2,
    )

    assert summary["row_memory_clean"] is True
    assert summary["process_memory_clean"] is True
    assert summary["source_student_row_memory_clean"] is False
    assert summary["source_student_clean_record_count"] == 0
    assert summary["source_student_dirty_record_count"] == 1
    assert summary["source_student_acceptance_memory_clean"] is False
    assert summary["source_vq_logits_memory_clean"] is False
    assert summary["acceptance_memory_clean"] is False


def test_eval_teacher_cache_cli_can_append_student_route_trace(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    eval_teacher_cache_cli = _load_eval_teacher_cache_cli()

    class FakeGate:
        def __call__(self, x):
            token_count = int(x.shape[1])
            return (
                mx.array([[[7, 3], [4, 5]][:token_count]], dtype=mx.int32),
                mx.array([[[0.75, 0.25], [0.6, 0.4]][:token_count]], dtype=mx.float32),
            )

    class FakeMLP:
        def __init__(self):
            self.gate = FakeGate()

    class FakeLayer:
        def __init__(self):
            self.mlp = FakeMLP()

    class FakeInnerModel:
        def __init__(self):
            self.layers = [None] * 37
            self.layers[36] = FakeLayer()

    class FakeResidentModel:
        def __init__(self):
            self.model = FakeInnerModel()

        def make_cache(self):
            return []

        def __call__(self, input_ids, *, cache):
            gate = self.model.layers[36].mlp.gate
            gate(mx.zeros((1, int(input_ids.shape[1]), 1), dtype=mx.float32))
            return mx.array([[[3.0, 0.0, -1.0], [0.0, 3.0, -1.0]]], dtype=mx.float32)

    row = _with_memory_counters(
        {
            "schema_version": 1,
            "model_id": "zai-org/GLM-4.5-Air",
            "revision": "test",
            "teacher_kind": "bf16_source",
            "prompt_id": "trace_student",
            "input_token_ids": [0, 1, 2],
            "target_token_ids": [1, 2],
            "positions": [0, 1],
            "logit_shard": "inline-topk",
            "topk_ids": [[0, 1, 2], [1, 0, 2]],
            "topk_logprobs": [[-0.1, -2.0, -3.0], [-0.1, -2.0, -3.0]],
            "target_logprobs": [-0.1, -0.1],
            "teacher_top1_ids": [0, 1],
            "full_logits_available": False,
        }
    )
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(json.dumps(row) + "\n", encoding="utf-8")
    append_jsonl = tmp_path / "student.jsonl"

    monkeypatch.setattr(
        eval_teacher_cache_cli,
        "load_resident_air",
        lambda **kwargs: (FakeResidentModel(), None, None, None),
    )
    monkeypatch.setattr(
        eval_teacher_cache_cli,
        "collect_vm_stat_counts",
        lambda: {"pageouts": 0, "swapouts": 0},
    )
    monkeypatch.setattr(eval_teacher_cache_cli, "reset_mlx_peak_memory", lambda: None)
    monkeypatch.setattr(
        eval_teacher_cache_cli,
        "collect_metric_snapshot",
        lambda previous_vm_stat_counts: {
            "elapsed_seconds": 0.01,
            "mlx_active_bytes": 1,
            "mlx_peak_bytes": 1,
            "mlx_cache_bytes": 0,
            "rss_bytes": 1,
            "pageouts_total": 0,
            "swapouts_total": 0,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        },
    )
    monkeypatch.setattr(eval_teacher_cache_cli, "has_dense_glm45_air_routed_expert_parameters", lambda model: False)
    monkeypatch.setattr(eval_teacher_cache_cli, "has_unbound_glm45_air_vq_experts", lambda model: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_glm45_air_teacher_cache.py",
            "--teacher-jsonl",
            str(metadata),
            "--cache-root",
            str(tmp_path),
            "--min-top-k",
            "2",
            "--include-route-trace",
            "--route-trace-layer",
            "36",
            "--append-jsonl",
            str(append_jsonl),
        ],
    )

    eval_teacher_cache_cli.main()

    summary = json.loads(capsys.readouterr().out)
    assert summary["row_memory_clean"] is True
    assert summary["process_memory_clean"] is True
    assert summary["acceptance_memory_clean"] is True
    assert summary["process_pageouts_delta"] == 0
    assert summary["process_swapouts_delta"] == 0

    record = json.loads(append_jsonl.read_text(encoding="utf-8"))
    route_trace = record["route_trace"]["36"]
    assert route_trace["top_k"] == 2
    assert route_trace["token_expert_indices"] == [[7, 3], [4, 5]]
    np.testing.assert_allclose(route_trace["token_router_scores"], [[0.75, 0.25], [0.6, 0.4]])
    assert record["route_trace_layers"] == ["36"]


def test_cleanroom_runner_can_forward_route_trace_collection() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_glm45_air_distributed_cleanroom_cache.sh")
    text = script.read_text(encoding="utf-8")

    assert "INCLUDE_ROUTE_TRACE=\"${GLM_INCLUDE_ROUTE_TRACE:-0}\"" in text
    assert "TMP_SRC_ARCHIVE=" in text
    assert "| tar -xf - -C \"$TMP_SRC_ROOT\"" not in text
    assert "GLM_INCLUDE_ROUTE_TRACE=1" in text
    assert 'exporter will include token-level route traces' in text
    assert 'local eval will include student token-level route traces' in text


def test_cleanroom_runner_can_skip_cache_validation_for_route_trace_only() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_glm45_air_distributed_cleanroom_cache.sh")
    text = script.read_text(encoding="utf-8")

    assert "ROUTE_TRACE_ONLY=\"${GLM_ROUTE_TRACE_ONLY:-0}\"" in text
    assert 'args+=("--route-trace-only")' in text
    assert "route-trace-only export requested; skipping teacher-cache validation and local consumption" in text


def test_cleanroom_runner_can_forward_route_trace_layers() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_glm45_air_distributed_cleanroom_cache.sh")
    text = script.read_text(encoding="utf-8")

    assert "ROUTE_TRACE_LAYERS=\"${GLM_ROUTE_TRACE_LAYERS:-}\"" in text
    assert 'printf \'%s\\0%s\\0\' "--route-trace-layer" "$layer"' in text


def test_cleanroom_runner_can_forward_layer_split_for_pruned_views() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_glm45_air_distributed_cleanroom_cache.sh")
    text = script.read_text(encoding="utf-8")

    assert "LAYER_SPLIT=\"${GLM_LAYER_SPLIT:-}\"" in text
    assert 'args+=("--layer-split" "$LAYER_SPLIT")' in text


def test_cleanroom_runner_recovery_hint_uses_current_rdma_netmask() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_glm45_air_distributed_cleanroom_cache.sh")
    text = script.read_text(encoding="utf-8")

    assert "sudo ifconfig bridge0 down" in text
    assert "netmask 255.255.255.252 up" in text
    assert "route -n add -host" not in text


def test_cleanroom_runner_rank_view_json_default_avoids_zsh_extra_brace() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_glm45_air_distributed_cleanroom_cache.sh")
    text = script.read_text(encoding="utf-8")

    assert "DEFAULT_RANK_VIEW_ROOTS_JSON=" in text
    assert 'RANK_VIEW_ROOTS_JSON="${GLM_RANK_VIEW_ROOTS_JSON:-$DEFAULT_RANK_VIEW_ROOTS_JSON}"' in text
    assert 'RANK_VIEW_ROOTS_JSON="${GLM_RANK_VIEW_ROOTS_JSON:-{\\"0\\"' not in text


def test_cleanroom_runner_uses_active_python_for_mlx_launcher() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_glm45_air_distributed_cleanroom_cache.sh")
    text = script.read_text(encoding="utf-8")

    assert "mlx_launch" in text
    assert "from mlx._distributed_utils.launch import main; main()" in text
    assert "uv run mlx.launch" not in text


def test_cleanroom_runner_promotes_rank_launch_failures_to_nonzero() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_glm45_air_distributed_cleanroom_cache.sh")
    text = script.read_text(encoding="utf-8")

    assert "mlx_launch_checked" in text
    assert 'grep -Eq "Node with rank .*exited with code -?[1-9][0-9]*"' in text
    assert "Traceback (most recent call last)" in text
    assert "PIPESTATUS[0]" in text
    assert "run_jaccl_probe()" in text
    assert "mlx_launch_checked \\" in text
    assert "local launch_cmd=(" in text
    assert "mlx_launch_checked" in text[text.index("local launch_cmd=(") : text.index("--rank-view-roots-json")]


def test_cleanroom_runner_can_persist_launcher_logs() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_glm45_air_distributed_cleanroom_cache.sh")
    text = script.read_text(encoding="utf-8")

    assert "LAUNCH_LOG_DIR=\"${GLM_LAUNCH_LOG_DIR:-}\"" in text
    assert "mkdir -p \"$LAUNCH_LOG_DIR\"" in text
    assert "cp \"$output_file\" \"$LAUNCH_LOG_DIR/" in text


def test_cleanroom_runner_has_explicit_single_host_fallback_without_peer_preflight() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_glm45_air_distributed_cleanroom_cache.sh")
    text = script.read_text(encoding="utf-8")

    assert "SINGLE_HOST_FALLBACK=\"${GLM_SINGLE_HOST_FALLBACK:-0}\"" in text
    assert "run_single_host_exporter()" in text
    assert "benchmarks/export_glm45_air_teacher_cache.py" in text
    assert "GLM_SINGLE_HOST_FALLBACK=1; skipping peer, direct-link, and JACCL preflight" in text
    assert "GLM_PREFLIGHT_ONLY=1; single-host fallback prerequisites are present" in text
    fallback_branch = text[
        text.index('if [[ "$SINGLE_HOST_FALLBACK" == "1" ]]; then') :
        text.index('require_file "$HOSTFILE"')
    ]
    assert "require_direct_link" not in fallback_branch
    assert "require_wired_caps" not in fallback_branch
    assert "write_wrappers_and_probes" not in fallback_branch
    assert "run_jaccl_probe" not in fallback_branch
    assert "ssh " not in fallback_branch


def test_cleanroom_runner_can_skip_local_consume_after_validation() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_glm45_air_distributed_cleanroom_cache.sh")
    text = script.read_text(encoding="utf-8")

    assert "SKIP_LOCAL_CONSUME=\"${GLM_SKIP_LOCAL_CONSUME:-0}\"" in text
    assert '[[ "$ROUTE_TRACE_ONLY" != "1" && "$SKIP_LOCAL_CONSUME" != "1" ]]' in text
    assert "validate_teacher_cache()" in text
    assert "consume_teacher_cache()" in text
    assert "GLM_SKIP_LOCAL_CONSUME=1; skipping local VQ consumption after validation" in text
    validate_and_consume = text[
        text.index("validate_and_consume()") :
        text.index("\nmain()")
    ]
    assert "validate_teacher_cache" in validate_and_consume
    assert "consume_teacher_cache" in validate_and_consume
    assert "consume_teacher_cache" not in validate_and_consume[
        validate_and_consume.index('if [[ "$SKIP_LOCAL_CONSUME" == "1" ]]; then') :
        validate_and_consume.index("  consume_teacher_cache")
    ]


def test_cleanroom_runner_forwards_single_host_memory_bounds() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_glm45_air_distributed_cleanroom_cache.sh")
    text = script.read_text(encoding="utf-8")

    assert "SINGLE_HOST_SOURCE_MEMORY_GUARD_RATIO=" in text
    single_host_exporter = text[
        text.index("run_single_host_exporter()") :
        text.index("\nvalidate_teacher_cache()")
    ]
    assert 'args+=("--mlx-cache-limit-gb" "$MLX_CACHE_LIMIT_GB")' in single_host_exporter
    assert 'args+=("--mlx-wired-limit-gb" "$SINGLE_HOST_MLX_WIRED_LIMIT_GB")' in single_host_exporter
    assert 'args+=("--source-memory-guard-ratio" "$SINGLE_HOST_SOURCE_MEMORY_GUARD_RATIO")' in single_host_exporter


def test_cleanroom_runner_defaults_to_system_wired_memory_limits() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_glm45_air_distributed_cleanroom_cache.sh")
    text = script.read_text(encoding="utf-8")

    assert 'REQUIRED_WIRED_MB="${GLM_REQUIRED_WIRED_MB:-0}"' in text
    assert 'MLX_WIRED_LIMIT_GB="${GLM_MLX_WIRED_LIMIT_GB:-}"' in text
    assert 'MLX_WIRED_LIMIT_GB="${GLM_MLX_WIRED_LIMIT_GB:-120}"' not in text

    run_exporter = text[text.index("run_exporter()") : text.index("\nrun_single_host_exporter()")]
    assert 'args+=("--mlx-wired-limit-gb" "$MLX_WIRED_LIMIT_GB")' in run_exporter
    assert '--mlx-wired-limit-gb "$MLX_WIRED_LIMIT_GB"' not in run_exporter

    check_wired = text[text.index("check_mlx_wired_limit()") : text.index("\nrun_jaccl_probe()")]
    assert 'if [[ -z "$MLX_WIRED_LIMIT_GB" ]]; then' in check_wired
    assert "using system MLX wired memory limit" in check_wired


def test_cleanroom_runner_can_use_local_sequential_pipeline_views() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_glm45_air_distributed_cleanroom_cache.sh")
    text = script.read_text(encoding="utf-8")

    single_host_exporter = text[
        text.index("run_single_host_exporter()") :
        text.index("\nvalidate_teacher_cache()")
    ]
    assert "SINGLE_HOST_PIPELINE_LOCAL=" in text
    assert "SINGLE_HOST_PIPELINE_LOCAL_STAGE_PROCESSES=" in text
    assert 'if [[ "$SINGLE_HOST_PIPELINE_LOCAL" == "1" ]]; then' in single_host_exporter
    assert "benchmarks/export_glm45_air_distributed_teacher_cache.py" in single_host_exporter
    assert "--local-sequential" in single_host_exporter
    assert "--local-sequential-stage-processes" in single_host_exporter
    assert "--rank-view-roots-json" in single_host_exporter
    assert "--rank0-only-logits" in single_host_exporter
    assert "--stream-lm-head" in single_host_exporter
    assert "benchmarks/export_glm45_air_teacher_cache.py" in single_host_exporter


def test_cleanroom_runner_can_explicitly_allow_dirty_cache_validation() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_glm45_air_distributed_cleanroom_cache.sh")
    text = script.read_text(encoding="utf-8")

    validation = text[
        text.index("validate_teacher_cache()") :
        text.index("\nconsume_teacher_cache()")
    ]
    assert "ALLOW_DIRTY_CACHE=" in text
    assert 'os.environ.get("GLM_ALLOW_DIRTY_CACHE") == "1"' in validation
    assert "and not allow_dirty" in validation
    assert "GLM_ALLOW_DIRTY_CACHE=1" in validation


def test_cleanroom_runner_can_require_memory_quiet_preflight_before_export() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_glm45_air_distributed_cleanroom_cache.sh")
    text = script.read_text(encoding="utf-8")

    assert 'MEMORY_QUIET_PREFLIGHT="${GLM_MEMORY_QUIET_PREFLIGHT:-0}"' in text
    assert (
        'REQUIRE_MEMORY_QUIET_PREFLIGHT="${GLM_REQUIRE_MEMORY_QUIET_PREFLIGHT:-0}"'
        in text
    )
    assert 'MEMORY_QUIET_SECONDS="${GLM_MEMORY_QUIET_SECONDS:-3}"' in text
    assert 'MEMORY_QUIET_MAX_ATTEMPTS="${GLM_MEMORY_QUIET_MAX_ATTEMPTS:-1}"' in text
    assert 'MEMORY_QUIET_MIN_FREE_GB="${GLM_MEMORY_QUIET_MIN_FREE_GB:-0}"' in text
    assert (
        'MEMORY_QUIET_STAGE_VIEW_ROOTS_JSON="${GLM_MEMORY_QUIET_STAGE_VIEW_ROOTS_JSON:-$LOCAL_SEQUENTIAL_STAGE_VIEW_ROOTS_JSON}"'
        in text
    )
    assert (
        'MEMORY_QUIET_STAGE_VIEW_MARGIN_GB="${GLM_MEMORY_QUIET_STAGE_VIEW_MARGIN_GB:-0}"'
        in text
    )
    assert 'MEMORY_QUIET_STAGE_VIEW_WORKER="${GLM_MEMORY_QUIET_STAGE_VIEW_WORKER:-}"' in text
    assert 'MEMORY_QUIET_STAGE_VIEW_WORKER="lower-embed"' in text
    assert "stage_view_max_visible_shard_gb" in text
    assert "stage_view_worker_found" in text
    assert "effective_min_free_gb" in text
    assert "available_gb" in text
    assert '"availability_basis": "free+speculative+inactive"' in text
    assert "--local-sequential-stage-memory-quiet-preflight" in text
    assert "--local-sequential-stage-memory-quiet-stage-view-margin-gb" in text
    assert "run_memory_quiet_preflight()" in text
    assert "required memory quiet preflight failed before model/exporter work" in text

    main = text[text.index("main() {") :]
    fallback_start = main.index('if [[ "$SINGLE_HOST_FALLBACK" == "1" ]]; then')
    fallback_end = main.index("run_single_host_exporter", fallback_start)
    distributed_start = main.index('run_jaccl_probe')
    distributed_end = main.index("run_exporter", distributed_start)
    fallback_branch = main[fallback_start:fallback_end]
    distributed_branch = main[distributed_start:distributed_end]
    assert "run_memory_quiet_preflight" in fallback_branch
    assert "run_memory_quiet_preflight" in distributed_branch


def test_cleanroom_runner_checks_uc_pingpong_before_jaccl() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_glm45_air_distributed_cleanroom_cache.sh")
    text = script.read_text(encoding="utf-8")

    assert 'LOCAL_RDMA_DEVICE="${GLM_LOCAL_RDMA_DEVICE:-rdma_en1}"' in text
    assert 'PEER_RDMA_DEVICE="${GLM_PEER_RDMA_DEVICE:-rdma_en1}"' in text
    assert "run_uc_pingpong()" in text
    assert "/usr/bin/ibv_uc_pingpong -d ${peer_device_quoted} -n 100 -g 1" in text
    assert "/usr/bin/ibv_uc_pingpong" in text
    assert "UC pingpong passed" in text

    main = text[text.index("main() {") :]
    uc_index = main.index("run_uc_pingpong")
    jaccl_index = main.index("run_jaccl_probe")
    exporter_index = main.index("run_exporter")
    assert uc_index < jaccl_index < exporter_index


def test_cleanroom_runner_checks_local_and_peer_rank_views_before_export() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_glm45_air_distributed_cleanroom_cache.sh")
    text = script.read_text(encoding="utf-8")

    assert 'PEER_RANK1_VIEW_ROOT="${GLM_PEER_RANK1_VIEW_ROOT:-}"' in text
    assert "rank_view_roots_json()" in text
    assert "require_rank_view_roots()" in text
    assert "local rank0 view root" in text
    assert "peer rank1 view root" in text
    assert "model.safetensors.index.json" in text
    assert "pipeline_view_plan.json" in text

    main = text[text.index("main() {") :]
    distributed_start = main.index("run_jaccl_probe")
    exporter_index = main.index("run_exporter", distributed_start)
    distributed_branch = main[distributed_start:exporter_index]
    assert (
        distributed_branch.index("run_jaccl_probe")
        < distributed_branch.index("stage_peer_rank1_view_root")
        < distributed_branch.index("require_rank_view_roots")
        < distributed_branch.index("run_memory_quiet_preflight")
    )


def test_cleanroom_runner_can_stage_peer_rank1_view_before_export() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_glm45_air_distributed_cleanroom_cache.sh")
    text = script.read_text(encoding="utf-8")

    assert 'PEER_SOURCE_DIR="${GLM_PEER_SOURCE_DIR:-}"' in text
    assert 'PEER_SOURCE_STAGE_SSH="${GLM_PEER_SOURCE_STAGE_SSH:-$PEER_SSH}"' in text
    assert 'STAGE_PEER_SOURCE="${GLM_STAGE_PEER_SOURCE:-0}"' in text
    assert 'PEER_SOURCE_MIN_FREE_GB="${GLM_PEER_SOURCE_MIN_FREE_GB:-16}"' in text
    assert "local_rank_view_root()" in text
    assert "peer_source_dir_for_rank1()" in text
    assert "stage_peer_rank1_source_files()" in text
    assert "stage_peer_rank1_view_root()" in text
    assert "missing peer source snapshot" in text
    assert "insufficient peer disk for rank1 source stage" in text
    assert "rsync -aL --partial" in text
    assert "$PEER_SOURCE_STAGE_SSH:$peer_source_dir" in text
    assert "tar -cf" in text
    assert "rewiring peer rank1 view symlinks" in text
    assert "missing peer source files" in text
    assert "target = source / child.name" in text
    assert "child.symlink_to(target)" in text


def test_cleanroom_runner_forwards_local_sequential_remote_workers() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_glm45_air_distributed_cleanroom_cache.sh")
    text = script.read_text(encoding="utf-8")

    assert 'PEER_RANK0_VIEW_ROOT="${GLM_PEER_RANK0_VIEW_ROOT:-}"' in text
    assert 'LOCAL_SEQUENTIAL_REMOTE_WORKERS="${GLM_LOCAL_SEQUENTIAL_REMOTE_WORKERS:-}"' in text
    assert 'LOCAL_SEQUENTIAL_REMOTE_SSH="${GLM_LOCAL_SEQUENTIAL_REMOTE_SSH:-$PEER_SSH}"' in text
    assert 'LOCAL_SEQUENTIAL_REMOTE_REPO="${GLM_LOCAL_SEQUENTIAL_REMOTE_REPO:-$PEER_REPO}"' in text
    assert (
        'LOCAL_SEQUENTIAL_REMOTE_PYTHON="${GLM_LOCAL_SEQUENTIAL_REMOTE_PYTHON:-$PEER_UV run python}"'
        in text
    )
    assert 'LOCAL_SEQUENTIAL_REMOTE_DIRTY_RETRIES="${GLM_LOCAL_SEQUENTIAL_REMOTE_DIRTY_RETRIES:-0}"' in text
    assert (
        'LOCAL_SEQUENTIAL_REMOTE_RETRY_SLEEP_SECONDS="${GLM_LOCAL_SEQUENTIAL_REMOTE_RETRY_SLEEP_SECONDS:-0}"'
        in text
    )
    assert "stage_peer_rank0_view_root()" in text
    assert "peer_rank0_view_root()" in text
    assert "local_sequential_remote_stage_roots_json()" in text
    assert 'roots[f"upper-window-{index}"] = {"0": peer_rank0_root}' in text
    assert "--local-sequential-remote-worker" in text
    assert "--local-sequential-remote-ssh" in text
    assert "--local-sequential-remote-repo" in text
    assert "--local-sequential-remote-python" in text
    assert "--local-sequential-remote-ssh-option" in text
    assert "--local-sequential-remote-tmp-dir" in text
    assert "--local-sequential-remote-dirty-retries" in text
    assert "--local-sequential-remote-retry-sleep-seconds" in text
    assert "--local-sequential-remote-stage-view-roots-json" in text


def test_eval_teacher_cache_cli_rejects_out_of_range_row_index() -> None:
    eval_teacher_cache_cli = _load_eval_teacher_cache_cli()

    with pytest.raises(IndexError, match="--row-index 3 is out of range for 2 rows"):
        eval_teacher_cache_cli._select_teacher_rows(
            [{"prompt_id": "first"}, {"prompt_id": "second"}],
            max_rows=None,
            row_index=3,
            row_indices=None,
        )


def test_eval_teacher_cache_cli_rejects_out_of_range_row_indices() -> None:
    eval_teacher_cache_cli = _load_eval_teacher_cache_cli()

    with pytest.raises(IndexError, match="--row-indices includes 3, out of range for 2 rows"):
        eval_teacher_cache_cli._select_teacher_rows(
            [{"prompt_id": "first"}, {"prompt_id": "second"}],
            max_rows=None,
            row_index=None,
            row_indices=(1, 3),
        )


def test_eval_teacher_cache_cli_parse_row_indices() -> None:
    eval_teacher_cache_cli = _load_eval_teacher_cache_cli()

    assert eval_teacher_cache_cli._parse_row_indices("43, 67,71") == (43, 67, 71)


def test_select_teacher_cache_rows_skips_summary_and_avoids_trained_rows(tmp_path) -> None:
    selector_cli = _load_select_teacher_cache_rows_cli()
    report_eval = tmp_path / "report.jsonl"
    selection_eval = tmp_path / "selection.jsonl"
    report_eval.write_text(
        "\n".join(
            [
                json.dumps({"row_index": 0, "prompt_id": "report_0", "top1_agreement": 0.9, "mean_kld": 0.1}),
                json.dumps({"row_index": 1, "prompt_id": "report_1", "top1_agreement": 0.2, "mean_kld": 0.3}),
                json.dumps({"row_index": 2, "prompt_id": "report_2", "top1_agreement": 0.4, "mean_kld": 1.1}),
                json.dumps({"record_type": "air_vq_teacher_cache_eval_summary", "mean_top1_agreement": 0.5}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    selection_eval.write_text(
        "\n".join(
            [
                json.dumps({"row_index": 0, "prompt_id": "select_0", "top1_agreement": 0.8, "mean_kld": 0.2}),
                json.dumps({"row_index": 1, "prompt_id": "select_1", "top1_agreement": 0.3, "mean_kld": 0.4}),
                json.dumps({"row_index": 2, "prompt_id": "select_2", "top1_agreement": 0.7, "mean_kld": 1.2}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = selector_cli.select_target_rows(
        selector_cli._read_eval_rows(report_eval),
        selector_cli._read_eval_rows(selection_eval),
        count=2,
        pool_size=3,
        avoided_row_indices=(1,),
    )

    assert result["row_indices"] == [2, 0]
    assert result["row_indices_csv"] == "2,0"
    assert result["avoided_row_indices"] == [1]
    assert result["rows"][0]["report"]["prompt_id"] == "report_2"
    assert result["rows"][0]["selection"]["prompt_id"] == "select_2"


def test_select_teacher_cache_rows_combines_low_top1_and_high_kld_ranks() -> None:
    selector_cli = _load_select_teacher_cache_rows_cli()
    report_rows = [
        {"row_index": 0, "prompt_id": "r0", "top1_agreement": 0.1, "mean_kld": 0.2, "ppl_ratio": 1.0, "memory_clean": True},
        {"row_index": 1, "prompt_id": "r1", "top1_agreement": 0.8, "mean_kld": 2.0, "ppl_ratio": 2.0, "memory_clean": True},
        {"row_index": 2, "prompt_id": "r2", "top1_agreement": 0.3, "mean_kld": 1.0, "ppl_ratio": 1.5, "memory_clean": True},
    ]
    selection_rows = [
        {"row_index": 0, "prompt_id": "s0", "top1_agreement": 0.9, "mean_kld": 0.1, "ppl_ratio": 1.0, "memory_clean": True},
        {"row_index": 1, "prompt_id": "s1", "top1_agreement": 0.2, "mean_kld": 1.9, "ppl_ratio": 2.0, "memory_clean": True},
        {"row_index": 2, "prompt_id": "s2", "top1_agreement": 0.4, "mean_kld": 0.5, "ppl_ratio": 1.5, "memory_clean": True},
    ]

    result = selector_cli.select_target_rows(
        report_rows,
        selection_rows,
        count=2,
        pool_size=2,
        top1_weight=1.0,
        kld_weight=1.0,
    )

    assert result["row_indices"][0] == 1
    assert {source["metric"] for source in result["rows"][0]["sources"]} == {"low_top1", "high_kld"}


def test_select_teacher_cache_rows_can_require_shared_rank_signal() -> None:
    selector_cli = _load_select_teacher_cache_rows_cli()
    report_rows = [
        {"row_index": 0, "prompt_id": "r0", "top1_agreement": 0.1, "mean_kld": 0.1, "ppl_ratio": 1.0, "memory_clean": True},
        {"row_index": 1, "prompt_id": "r1", "top1_agreement": 0.2, "mean_kld": 0.2, "ppl_ratio": 1.0, "memory_clean": True},
        {"row_index": 2, "prompt_id": "r2", "top1_agreement": 0.9, "mean_kld": 2.0, "ppl_ratio": 1.0, "memory_clean": True},
    ]
    selection_rows = [
        {"row_index": 0, "prompt_id": "s0", "top1_agreement": 0.9, "mean_kld": 0.1, "ppl_ratio": 1.0, "memory_clean": True},
        {"row_index": 1, "prompt_id": "s1", "top1_agreement": 0.3, "mean_kld": 0.3, "ppl_ratio": 1.0, "memory_clean": True},
        {"row_index": 2, "prompt_id": "s2", "top1_agreement": 0.8, "mean_kld": 1.9, "ppl_ratio": 1.0, "memory_clean": True},
    ]

    result = selector_cli.select_target_rows(
        report_rows,
        selection_rows,
        count=3,
        pool_size=2,
        top1_weight=1.0,
        kld_weight=1.0,
        shared_signal_only=True,
    )

    assert result["row_indices"] == [1, 2]
    assert result["shared_signal_only"] is True
    assert all(
        {source["split"] for source in row["sources"]} == {"report", "selection"}
        for row in result["rows"]
    )


@pytest.mark.parametrize("row_indices", ["", " ", "1,-2", "1,nope"])
def test_eval_teacher_cache_cli_rejects_invalid_row_indices_before_work(
    tmp_path,
    monkeypatch,
    capsys,
    row_indices,
) -> None:
    eval_teacher_cache_cli = _load_eval_teacher_cache_cli()

    def fail_validation(*args, **kwargs):
        raise AssertionError("cache validation should not run for invalid --row-indices")

    def fail_load(*args, **kwargs):
        raise AssertionError("resident VQ should not load for invalid --row-indices")

    monkeypatch.setattr(
        eval_teacher_cache_cli,
        "validate_teacher_cache_metadata",
        fail_validation,
    )
    monkeypatch.setattr(eval_teacher_cache_cli, "load_resident_air", fail_load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_glm45_air_teacher_cache.py",
            "--teacher-jsonl",
            str(tmp_path / "missing.jsonl"),
            "--row-indices",
            row_indices,
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        eval_teacher_cache_cli.main()

    captured = capsys.readouterr()
    assert exit_info.value.code == 2
    assert "--row-indices" in captured.err


@pytest.mark.parametrize("row_index", ["-1", "-2"])
def test_eval_teacher_cache_cli_rejects_negative_row_index_before_work(
    tmp_path,
    monkeypatch,
    capsys,
    row_index,
) -> None:
    eval_teacher_cache_cli = _load_eval_teacher_cache_cli()

    def fail_validation(*args, **kwargs):
        raise AssertionError("cache validation should not run for invalid --row-index")

    def fail_load(*args, **kwargs):
        raise AssertionError("resident VQ should not load for invalid --row-index")

    monkeypatch.setattr(
        eval_teacher_cache_cli,
        "validate_teacher_cache_metadata",
        fail_validation,
    )
    monkeypatch.setattr(eval_teacher_cache_cli, "load_resident_air", fail_load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_glm45_air_teacher_cache.py",
            "--teacher-jsonl",
            str(tmp_path / "missing.jsonl"),
            "--row-index",
            row_index,
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        eval_teacher_cache_cli.main()

    captured = capsys.readouterr()
    assert exit_info.value.code == 2
    assert "--row-index must be zero or greater" in captured.err


def test_eval_teacher_cache_cli_rejects_row_index_with_max_rows_before_work(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    eval_teacher_cache_cli = _load_eval_teacher_cache_cli()

    def fail_validation(*args, **kwargs):
        raise AssertionError("cache validation should not run for invalid row selection")

    def fail_load(*args, **kwargs):
        raise AssertionError("resident VQ should not load for invalid row selection")

    monkeypatch.setattr(
        eval_teacher_cache_cli,
        "validate_teacher_cache_metadata",
        fail_validation,
    )
    monkeypatch.setattr(eval_teacher_cache_cli, "load_resident_air", fail_load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_glm45_air_teacher_cache.py",
            "--teacher-jsonl",
            str(tmp_path / "missing.jsonl"),
            "--max-rows",
            "1",
            "--row-index",
            "1",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        eval_teacher_cache_cli.main()

    captured = capsys.readouterr()
    assert exit_info.value.code == 2
    assert "--row-index cannot be combined with --max-rows" in captured.err


def test_eval_teacher_cache_cli_rejects_row_indices_with_max_rows_before_work(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    eval_teacher_cache_cli = _load_eval_teacher_cache_cli()

    def fail_validation(*args, **kwargs):
        raise AssertionError("cache validation should not run for invalid row selection")

    def fail_load(*args, **kwargs):
        raise AssertionError("resident VQ should not load for invalid row selection")

    monkeypatch.setattr(
        eval_teacher_cache_cli,
        "validate_teacher_cache_metadata",
        fail_validation,
    )
    monkeypatch.setattr(eval_teacher_cache_cli, "load_resident_air", fail_load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_glm45_air_teacher_cache.py",
            "--teacher-jsonl",
            str(tmp_path / "missing.jsonl"),
            "--max-rows",
            "1",
            "--row-indices",
            "1,2",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        eval_teacher_cache_cli.main()

    captured = capsys.readouterr()
    assert exit_info.value.code == 2
    assert "--row-indices cannot be combined with --max-rows" in captured.err


@pytest.mark.parametrize(
    "script",
    [
        "benchmarks/validate_glm45_air_teacher_cache.py",
        "benchmarks/eval_glm45_air_teacher_cache.py",
    ],
)
@pytest.mark.parametrize("min_top_k", ["0", "-1"])
def test_teacher_cache_clis_reject_non_positive_min_top_k_before_file_load(
    tmp_path,
    script,
    min_top_k,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            script,
            "--teacher-jsonl",
            str(tmp_path / "missing.jsonl"),
            "--min-top-k",
            min_top_k,
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--min-top-k must be positive" in result.stderr
    assert "No such file" not in result.stderr


def test_validate_teacher_cache_row_requires_counters_and_minimum_topk(tmp_path) -> None:
    mx.save_safetensors(
        str(tmp_path / "teacher.safetensors"),
        {
            "topk_ids": mx.array(np.array([[0, 1]], dtype=np.int32)),
            "topk_logprobs": mx.array(np.array([[-0.1, -0.2]], dtype=np.float32)),
            "target_logprobs": mx.array(np.array([-0.1], dtype=np.float32)),
            "teacher_top1_ids": mx.array(np.array([0], dtype=np.int32)),
        },
    )
    row = {
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "q8",
        "prompt_id": "strict",
        "input_token_ids": [10, 11],
        "target_token_ids": [11],
        "positions": [0],
        "logit_shard": "teacher.safetensors",
        "topk_tensor": "topk_ids",
        "topk_logprob_tensor": "topk_logprobs",
        "target_logprob_tensor": "target_logprobs",
        "teacher_top1_tensor": "teacher_top1_ids",
        "full_logits_available": False,
    }

    report = validate_teacher_cache_row(row, cache_root=tmp_path, row_index=0)

    assert report["ok"] is False
    assert report["memory_counters_known"] is False
    assert any("pageouts_delta must be recorded" in error for error in report["errors"])
    assert any("swapouts_delta must be recorded" in error for error in report["errors"])
    assert any("elapsed_seconds must be recorded" in error for error in report["errors"])
    assert any("mlx_active_bytes must be recorded" in error for error in report["errors"])
    assert any("pageouts_total must be recorded" in error for error in report["errors"])
    assert any("topk width 2 must be at least 128" in error for error in report["errors"])


def test_validate_teacher_cache_row_allows_null_vm_totals(tmp_path) -> None:
    row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "q8",
        "prompt_id": "null_vm_totals",
        "input_token_ids": [10, 11],
        "target_token_ids": [11],
        "positions": [0],
        "logit_shard": "teacher.safetensors",
        "topk_ids": [[0, 1, 2]],
        "topk_logprobs": [[-0.1, -0.2, -0.3]],
        "target_logprobs": [-0.1],
        "teacher_top1_ids": [0],
        "full_logits_available": False,
    })
    row["pageouts_total"] = None
    row["swapouts_total"] = None
    row["pageouts_delta"] = None
    row["swapouts_delta"] = None

    report = validate_teacher_cache_row(row, cache_root=tmp_path, row_index=0, min_top_k=2)

    assert report["ok"] is True
    assert report["pageouts_total"] is None
    assert report["swapouts_total"] is None
    assert report["memory_counters_known"] is False
    assert any("pageouts_total is null" in warning for warning in report["warnings"])
    assert any("swapouts_total is null" in warning for warning in report["warnings"])


def test_validate_teacher_cache_row_rejects_bad_runtime_counters(tmp_path) -> None:
    row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "q8",
        "prompt_id": "bad_runtime_counters",
        "input_token_ids": [10, 11],
        "target_token_ids": [11],
        "positions": [0],
        "logit_shard": "teacher.safetensors",
        "topk_ids": [[0, 1, 2]],
        "topk_logprobs": [[-0.1, -0.2, -0.3]],
        "target_logprobs": [-0.1],
        "teacher_top1_ids": [0],
        "full_logits_available": False,
    })
    row.update(
        {
            "elapsed_seconds": -0.1,
            "mlx_active_bytes": -1,
            "mlx_peak_bytes": 1.5,
            "mlx_cache_bytes": None,
            "rss_bytes": True,
            "pageouts_total": -1,
            "swapouts_total": "0",
        }
    )

    report = validate_teacher_cache_row(row, cache_root=tmp_path, row_index=0, min_top_k=2)

    assert report["ok"] is False
    assert any("elapsed_seconds must be a non-negative finite number" in error for error in report["errors"])
    assert any("mlx_active_bytes must be a non-negative integer" in error for error in report["errors"])
    assert any("mlx_peak_bytes must be a non-negative integer" in error for error in report["errors"])
    assert any("mlx_cache_bytes must be a non-negative integer" in error for error in report["errors"])
    assert any("rss_bytes must be a non-negative integer" in error for error in report["errors"])
    assert any("pageouts_total must be a non-negative integer or null" in error for error in report["errors"])
    assert any("swapouts_total must be a non-negative integer or null" in error for error in report["errors"])


def test_validate_teacher_cache_row_rejects_negative_memory_counters(tmp_path) -> None:
    row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "q8",
        "prompt_id": "negative_counters",
        "input_token_ids": [10, 11],
        "target_token_ids": [11],
        "positions": [0],
        "logit_shard": "teacher.safetensors",
        "topk_ids": [[0, 1, 2]],
        "topk_logprobs": [[-0.1, -0.2, -0.3]],
        "target_logprobs": [-0.1],
        "teacher_top1_ids": [0],
        "full_logits_available": False,
    })
    row["pageouts_delta"] = -1
    row["swapouts_delta"] = -2

    report = validate_teacher_cache_row(row, cache_root=tmp_path, row_index=0, min_top_k=2)

    assert report["ok"] is False
    assert report["memory_clean"] is False
    assert any("pageouts_delta must be non-negative or null" in error for error in report["errors"])
    assert any("swapouts_delta must be non-negative or null" in error for error in report["errors"])


def test_validate_teacher_cache_row_requires_boolean_full_logits_flag(tmp_path) -> None:
    row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "q8",
        "prompt_id": "string_full_logits_flag",
        "input_token_ids": [10, 11],
        "target_token_ids": [11],
        "positions": [0],
        "logit_shard": "teacher.safetensors",
        "topk_ids": [[0, 1, 2]],
        "topk_logprobs": [[-0.1, -0.2, -0.3]],
        "target_logprobs": [-0.1],
        "teacher_top1_ids": [0],
        "full_logits_available": "false",
    })

    report = validate_teacher_cache_row(row, cache_root=tmp_path, row_index=0, min_top_k=2)

    assert report["ok"] is False
    assert report["full_logits_available"] is False
    assert any("full_logits_available must be a boolean" in error for error in report["errors"])


def test_validate_teacher_cache_row_rejects_false_full_logits_flag_with_logits(
    tmp_path,
) -> None:
    mx.save_safetensors(
        str(tmp_path / "teacher.safetensors"),
        {"logits": mx.array(np.ones((1, 4), dtype=np.float16))},
    )
    row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "q8",
        "prompt_id": "false_flag_with_logits",
        "input_token_ids": [10, 11],
        "target_token_ids": [11],
        "positions": [0],
        "logit_shard": "teacher.safetensors",
        "logit_tensor": "logits",
        "topk_ids": [[0, 1, 2]],
        "topk_logprobs": [[-1.0, -1.2, -1.5]],
        "target_logprobs": [-1.2],
        "teacher_top1_ids": [0],
        "full_logits_available": False,
    })

    report = validate_teacher_cache_row(row, cache_root=tmp_path, row_index=0, min_top_k=2)

    assert report["ok"] is False
    assert report["full_logits_available"] is True
    assert any(
        "full_logits_available was false but logits were usable" in error
        for error in report["errors"]
    )


def test_validate_teacher_cache_row_rejects_bad_tensor_dtypes(tmp_path) -> None:
    mx.save_safetensors(
        str(tmp_path / "teacher.safetensors"),
        {
            "logits": mx.array(np.ones((1, 3), dtype=np.int32)),
            "topk_ids": mx.array(np.array([[0.0, 1.0, 2.0]], dtype=np.float32)),
            "topk_logprobs": mx.array(np.array([[0, 1, 2]], dtype=np.int32)),
            "target_logprobs": mx.array(np.array([0], dtype=np.int32)),
            "teacher_top1_ids": mx.array(np.array([0.0], dtype=np.float32)),
        },
    )
    row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "bf16_source",
        "prompt_id": "bad_tensor_dtypes",
        "input_token_ids": [0, 1],
        "target_token_ids": [1],
        "positions": [0],
        "logit_shard": "teacher.safetensors",
        "logit_tensor": "logits",
        "topk_tensor": "topk_ids",
        "topk_logprob_tensor": "topk_logprobs",
        "target_logprob_tensor": "target_logprobs",
        "teacher_top1_tensor": "teacher_top1_ids",
        "full_logits_available": True,
    })

    report = validate_teacher_cache_row(row, cache_root=tmp_path, row_index=0, min_top_k=2)

    assert report["ok"] is False
    assert any("logits dtype I32 must be floating point" in error for error in report["errors"])
    assert any("topk_ids dtype F32 must be integer token ids" in error for error in report["errors"])
    assert any(
        "topk_logprobs dtype I32 must be floating point log-probabilities" in error
        for error in report["errors"]
    )
    assert any(
        "target_logprobs dtype I32 must be floating point log-probabilities" in error
        for error in report["errors"]
    )
    assert any(
        "teacher_top1_ids dtype F32 must be integer token ids" in error
        for error in report["errors"]
    )


def test_validate_teacher_cache_row_rejects_bad_tensor_names(tmp_path) -> None:
    mx.save_safetensors(
        str(tmp_path / "teacher.safetensors"),
        {
            "logits": mx.array(np.ones((1, 3), dtype=np.float16)),
            "topk_ids": mx.array(np.array([[0, 1, 2]], dtype=np.int32)),
            "topk_logprobs": mx.array(np.array([[-0.1, -0.2, -0.3]], dtype=np.float32)),
            "target_logprobs": mx.array(np.array([-0.1], dtype=np.float32)),
            "teacher_top1_ids": mx.array(np.array([0], dtype=np.int32)),
        },
    )
    row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "bf16_source",
        "prompt_id": "bad_tensor_names",
        "input_token_ids": [0, 1],
        "target_token_ids": [1],
        "positions": [0],
        "logit_shard": "teacher.safetensors",
        "logit_tensor": 123,
        "topk_tensor": "",
        "topk_logprob_tensor": ["topk_logprobs"],
        "target_logprob_tensor": False,
        "teacher_top1_tensor": 0,
        "full_logits_available": True,
    })

    report = validate_teacher_cache_row(row, cache_root=tmp_path, row_index=0, min_top_k=2)

    assert report["ok"] is False
    assert any("logits tensor name must be a non-empty string" in error for error in report["errors"])
    assert any("topk_ids tensor name must be a non-empty string" in error for error in report["errors"])
    assert any("topk_logprobs tensor name must be a non-empty string" in error for error in report["errors"])
    assert any("target_logprobs tensor name must be a non-empty string" in error for error in report["errors"])
    assert any("teacher_top1_ids tensor name must be a non-empty string" in error for error in report["errors"])


def test_validate_teacher_cache_row_check_values_rejects_bad_tensor_values(tmp_path) -> None:
    mx.save_safetensors(
        str(tmp_path / "teacher.safetensors"),
        {
            "logits": mx.array(np.array([[0.0, np.nan, 2.0, 3.0]], dtype=np.float32)),
            "topk_ids": mx.array(np.array([[0, 4, -1]], dtype=np.int32)),
            "topk_logprobs": mx.array(np.array([[-0.1, np.nan, -0.3]], dtype=np.float32)),
            "target_logprobs": mx.array(np.array([np.inf], dtype=np.float32)),
            "teacher_top1_ids": mx.array(np.array([5], dtype=np.int32)),
        },
    )
    row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "bf16_source",
        "prompt_id": "bad_tensor_values",
        "input_token_ids": [0, 1],
        "target_token_ids": [1],
        "positions": [0],
        "logit_shard": "teacher.safetensors",
        "logit_tensor": "logits",
        "topk_tensor": "topk_ids",
        "topk_logprob_tensor": "topk_logprobs",
        "target_logprob_tensor": "target_logprobs",
        "teacher_top1_tensor": "teacher_top1_ids",
        "full_logits_available": True,
    })

    report = validate_teacher_cache_row(
        row,
        cache_root=tmp_path,
        row_index=0,
        min_top_k=2,
        check_values=True,
    )

    assert report["ok"] is False
    assert report["value_checks_enabled"] is True
    assert any("logits values must be finite" in error for error in report["errors"])
    assert any("topk_ids values must be non-negative" in error for error in report["errors"])
    assert any(
        "topk_ids values must be less than logits vocab size 4" in error
        for error in report["errors"]
    )
    assert any("topk_logprobs values must be finite" in error for error in report["errors"])
    assert any("target_logprobs values must be finite" in error for error in report["errors"])
    assert any(
        "teacher_top1_ids values must be less than logits vocab size 4" in error
        for error in report["errors"]
    )


def test_validate_teacher_cache_row_check_values_rejects_positive_logprobs(tmp_path) -> None:
    mx.save_safetensors(
        str(tmp_path / "teacher.safetensors"),
        {
            "topk_ids": mx.array(np.array([[0, 1, 2]], dtype=np.int32)),
            "topk_logprobs": mx.array(np.array([[0.1, -0.2, 0.0]], dtype=np.float32)),
            "target_logprobs": mx.array(np.array([0.01], dtype=np.float32)),
            "teacher_top1_ids": mx.array(np.array([0], dtype=np.int32)),
        },
    )
    row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "q8",
        "prompt_id": "positive_logprobs",
        "input_token_ids": [10, 11],
        "target_token_ids": [11],
        "positions": [0],
        "logit_shard": "teacher.safetensors",
        "topk_tensor": "topk_ids",
        "topk_logprob_tensor": "topk_logprobs",
        "target_logprob_tensor": "target_logprobs",
        "teacher_top1_tensor": "teacher_top1_ids",
        "full_logits_available": False,
    })

    report = validate_teacher_cache_row(
        row,
        cache_root=tmp_path,
        row_index=0,
        min_top_k=2,
        check_values=True,
    )

    assert report["ok"] is False
    assert any(
        "topk_logprobs values must be log-probabilities <= 0" in error
        for error in report["errors"]
    )
    assert any(
        "target_logprobs values must be log-probabilities <= 0" in error
        for error in report["errors"]
    )


def test_validate_teacher_cache_row_check_values_rejects_topk_probability_mass(
    tmp_path,
) -> None:
    row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "q8",
        "prompt_id": "topk_probability_mass",
        "input_token_ids": [10, 11],
        "target_token_ids": [11],
        "positions": [0],
        "logit_shard": "teacher.safetensors",
        "topk_ids": [[11, 12, 13]],
        "topk_logprobs": [[-0.1, -0.1, -0.1]],
        "target_logprobs": [-0.1],
        "teacher_top1_ids": [11],
        "full_logits_available": False,
    })

    report = validate_teacher_cache_row(
        row,
        cache_root=tmp_path,
        row_index=0,
        min_top_k=2,
        check_values=True,
    )

    assert report["ok"] is False
    assert any(
        "topk_logprobs[0] probability mass must be <= 1.0" in error
        for error in report["errors"]
    )


def test_validate_teacher_cache_row_check_values_rejects_unsorted_topk_logprobs(
    tmp_path,
) -> None:
    row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "q8",
        "prompt_id": "unsorted_topk_logprobs",
        "input_token_ids": [10, 11],
        "target_token_ids": [11],
        "positions": [0],
        "logit_shard": "teacher.safetensors",
        "topk_ids": [[11, 12, 13]],
        "topk_logprobs": [[-2.0, -0.5, -2.5]],
        "target_logprobs": [-2.0],
        "teacher_top1_ids": [12],
        "full_logits_available": False,
    })

    report = validate_teacher_cache_row(
        row,
        cache_root=tmp_path,
        row_index=0,
        min_top_k=2,
        check_values=True,
    )

    assert report["ok"] is False
    assert any(
        "topk_logprobs[0] must be sorted from highest to lowest log-probability"
        in error
        for error in report["errors"]
    )


def test_validate_teacher_cache_row_check_values_rejects_duplicate_topk_ids(tmp_path) -> None:
    row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "q8",
        "prompt_id": "duplicate_topk_ids",
        "input_token_ids": [10, 11],
        "target_token_ids": [11],
        "positions": [0],
        "logit_shard": "teacher.safetensors",
        "topk_ids": [[0, 1, 1]],
        "topk_logprobs": [[-0.1, -0.2, -0.3]],
        "target_logprobs": [-0.1],
        "teacher_top1_ids": [0],
        "full_logits_available": False,
    })

    report = validate_teacher_cache_row(
        row,
        cache_root=tmp_path,
        row_index=0,
        min_top_k=2,
        check_values=True,
    )

    assert report["ok"] is False
    assert any(
        "topk_ids[0] contains duplicate token id 1" in error
        for error in report["errors"]
    )


def test_validate_teacher_cache_row_check_values_rejects_target_topk_logprob_mismatch(
    tmp_path,
) -> None:
    row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "q8",
        "prompt_id": "target_topk_logprob_mismatch",
        "input_token_ids": [10, 11],
        "target_token_ids": [11],
        "positions": [0],
        "logit_shard": "teacher.safetensors",
        "topk_ids": [[11, 12, 13]],
        "topk_logprobs": [[-0.2, -0.3, -0.4]],
        "target_logprobs": [-1.0],
        "teacher_top1_ids": [11],
        "full_logits_available": False,
    })

    report = validate_teacher_cache_row(
        row,
        cache_root=tmp_path,
        row_index=0,
        min_top_k=2,
        check_values=True,
    )

    assert report["ok"] is False
    assert any(
        "target_logprobs[0]=-1.0 must match topk_logprobs[0,0]=" in error
        and "for target token id 11" in error
        for error in report["errors"]
    )


def test_validate_teacher_cache_row_check_values_rejects_absent_target_above_topk_cutoff(
    tmp_path,
) -> None:
    row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "q8",
        "prompt_id": "absent_target_above_topk_cutoff",
        "input_token_ids": [10, 99],
        "target_token_ids": [99],
        "positions": [0],
        "logit_shard": "teacher.safetensors",
        "topk_ids": [[11, 12, 13]],
        "topk_logprobs": [[-1.0, -2.0, -3.0]],
        "target_logprobs": [-0.5],
        "teacher_top1_ids": [11],
        "full_logits_available": False,
    })

    report = validate_teacher_cache_row(
        row,
        cache_root=tmp_path,
        row_index=0,
        min_top_k=2,
        check_values=True,
    )

    assert report["ok"] is False
    assert any(
        "target_logprobs[0]=-0.5 for target token id 99 is absent from topk_ids[0]"
        in error
        for error in report["errors"]
    )


def test_validate_teacher_cache_row_check_values_allows_absent_target_below_topk_cutoff(
    tmp_path,
) -> None:
    row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "q8",
        "prompt_id": "absent_target_below_topk_cutoff",
        "input_token_ids": [10, 99],
        "target_token_ids": [99],
        "positions": [0],
        "logit_shard": "teacher.safetensors",
        "topk_ids": [[11, 12, 13]],
        "topk_logprobs": [[-1.0, -2.0, -3.0]],
        "target_logprobs": [-4.0],
        "teacher_top1_ids": [11],
        "full_logits_available": False,
    })

    report = validate_teacher_cache_row(
        row,
        cache_root=tmp_path,
        row_index=0,
        min_top_k=2,
        check_values=True,
    )

    assert report["ok"] is True
    assert report["errors"] == []


def test_validate_teacher_cache_row_check_values_rejects_teacher_top1_mismatch(tmp_path) -> None:
    mx.save_safetensors(
        str(tmp_path / "teacher.safetensors"),
        {"logits": mx.array(np.array([[0.0, 5.0, 4.0, 1.0]], dtype=np.float32))},
    )
    row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "bf16_source",
        "prompt_id": "teacher_top1_mismatch",
        "input_token_ids": [10, 11],
        "target_token_ids": [11],
        "positions": [0],
        "logit_shard": "teacher.safetensors",
        "logit_tensor": "logits",
        "topk_ids": [[1, 3, 0]],
        "topk_logprobs": [[-0.2, -0.3, -0.4]],
        "target_logprobs": [-0.2],
        "teacher_top1_ids": [2],
        "full_logits_available": True,
    })

    report = validate_teacher_cache_row(
        row,
        cache_root=tmp_path,
        row_index=0,
        min_top_k=2,
        check_values=True,
    )

    assert report["ok"] is False
    assert any(
        "teacher_top1_ids[0]=2 must match argmax(logits[0])=1" in error
        for error in report["errors"]
    )
    assert any(
        "teacher_top1_ids[0]=2 must appear in topk_ids[0]" in error
        for error in report["errors"]
    )


def test_validate_teacher_cache_row_check_values_rejects_teacher_top1_not_first_topk(
    tmp_path,
) -> None:
    row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "q8",
        "prompt_id": "teacher_top1_not_first_topk",
        "input_token_ids": [10, 11],
        "target_token_ids": [11],
        "positions": [0],
        "logit_shard": "teacher.safetensors",
        "topk_ids": [[11, 12, 13]],
        "topk_logprobs": [[-1.0, -2.0, -3.0]],
        "target_logprobs": [-1.0],
        "teacher_top1_ids": [12],
        "full_logits_available": False,
    })

    report = validate_teacher_cache_row(
        row,
        cache_root=tmp_path,
        row_index=0,
        min_top_k=2,
        check_values=True,
    )

    assert report["ok"] is False
    assert any(
        "teacher_top1_ids[0]=12 must match first-ranked topk_ids[0,0]=11"
        in error
        for error in report["errors"]
    )


def test_validate_teacher_cache_row_check_values_allows_tied_teacher_top1(
    tmp_path,
) -> None:
    row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "q8",
        "prompt_id": "tied_teacher_top1",
        "input_token_ids": [10, 11],
        "target_token_ids": [11],
        "positions": [0],
        "logit_shard": "teacher.safetensors",
        "topk_ids": [[11, 12, 13]],
        "topk_logprobs": [[-2.0, -2.0, -3.0]],
        "target_logprobs": [-2.0],
        "teacher_top1_ids": [12],
        "full_logits_available": False,
    })

    report = validate_teacher_cache_row(
        row,
        cache_root=tmp_path,
        row_index=0,
        min_top_k=2,
        check_values=True,
    )

    assert report["ok"] is True
    assert report["errors"] == []


def test_validate_teacher_cache_row_rejects_routed_q2_teacher_kind(tmp_path) -> None:
    mx.save_safetensors(
        str(tmp_path / "teacher.safetensors"),
        {
            "logits": mx.array(np.ones((1, 3), dtype=np.float16)),
            "topk_ids": mx.array(np.array([[0, 1, 2]], dtype=np.int32)),
            "topk_logprobs": mx.array(np.array([[-0.1, -0.2, -0.3]], dtype=np.float32)),
            "target_logprobs": mx.array(np.array([-0.1], dtype=np.float32)),
            "teacher_top1_ids": mx.array(np.array([0], dtype=np.int32)),
        },
    )
    row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "mlx_q2_routed_g128",
        "prompt_id": "q2_not_teacher",
        "input_token_ids": [10, 11],
        "target_token_ids": [11],
        "positions": [0],
        "logit_shard": "teacher.safetensors",
        "logit_tensor": "logits",
        "topk_tensor": "topk_ids",
        "topk_logprob_tensor": "topk_logprobs",
        "target_logprob_tensor": "target_logprobs",
        "teacher_top1_tensor": "teacher_top1_ids",
        "full_logits_available": True,
    })

    report = validate_teacher_cache_row(row, cache_root=tmp_path, row_index=0)

    assert report["ok"] is False
    assert any("not an allowed high-bit authority" in error for error in report["errors"])


def test_validate_teacher_cache_row_rejects_target_token_mismatch(tmp_path) -> None:
    mx.save_safetensors(
        str(tmp_path / "teacher.safetensors"),
        {
            "logits": mx.array(np.ones((2, 4), dtype=np.float16)),
            "topk_ids": mx.array(np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int32)),
            "topk_logprobs": mx.array(
                np.array([[-0.1, -0.2, -0.3, -0.4], [-0.1, -0.2, -0.3, -0.4]], dtype=np.float32)
            ),
            "target_logprobs": mx.array(np.array([-0.1, -0.2], dtype=np.float32)),
            "teacher_top1_ids": mx.array(np.array([0, 1], dtype=np.int32)),
        },
    )
    row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "bf16_source",
        "prompt_id": "target_mismatch",
        "input_token_ids": [10, 11, 12],
        "target_token_ids": [11, 99],
        "positions": [0, 1],
        "logit_shard": "teacher.safetensors",
        "logit_tensor": "logits",
        "topk_tensor": "topk_ids",
        "topk_logprob_tensor": "topk_logprobs",
        "target_logprob_tensor": "target_logprobs",
        "teacher_top1_tensor": "teacher_top1_ids",
        "full_logits_available": True,
    })

    report = validate_teacher_cache_row(row, cache_root=tmp_path, row_index=0)

    assert report["ok"] is False
    assert any("must match input_token_ids[2]=12" in error for error in report["errors"])


def test_validate_teacher_cache_row_rejects_negative_positions(tmp_path) -> None:
    mx.save_safetensors(
        str(tmp_path / "teacher.safetensors"),
        {
            "logits": mx.array(np.ones((1, 4), dtype=np.float16)),
            "topk_ids": mx.array(np.array([[0, 1, 2, 3]], dtype=np.int32)),
            "topk_logprobs": mx.array(np.array([[-0.1, -0.2, -0.3, -0.4]], dtype=np.float32)),
            "target_logprobs": mx.array(np.array([-0.1], dtype=np.float32)),
            "teacher_top1_ids": mx.array(np.array([0], dtype=np.int32)),
        },
    )
    row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "bf16_source",
        "prompt_id": "negative_position",
        "input_token_ids": [10, 11],
        "target_token_ids": [11],
        "positions": [-1],
        "logit_shard": "teacher.safetensors",
        "logit_tensor": "logits",
        "topk_tensor": "topk_ids",
        "topk_logprob_tensor": "topk_logprobs",
        "target_logprob_tensor": "target_logprobs",
        "teacher_top1_tensor": "teacher_top1_ids",
        "full_logits_available": True,
    })

    report = validate_teacher_cache_row(row, cache_root=tmp_path, row_index=0)

    assert report["ok"] is False
    assert any("positions[0] must be non-negative" in error for error in report["errors"])


def test_validate_teacher_cache_row_rejects_duplicate_positions(tmp_path) -> None:
    row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "q8",
        "prompt_id": "duplicate_positions",
        "input_token_ids": [10, 11, 12],
        "target_token_ids": [11, 11],
        "positions": [0, 0],
        "logit_shard": "teacher.safetensors",
        "topk_ids": [[0, 1, 2], [0, 1, 2]],
        "topk_logprobs": [[-0.1, -0.2, -0.3], [-0.1, -0.2, -0.3]],
        "target_logprobs": [-0.1, -0.1],
        "teacher_top1_ids": [0, 0],
        "full_logits_available": False,
    })

    report = validate_teacher_cache_row(row, cache_root=tmp_path, row_index=0, min_top_k=2)

    assert report["ok"] is False
    assert any("positions[1] duplicates earlier value 0" in error for error in report["errors"])


def test_validate_teacher_cache_row_rejects_boolean_integer_list_values(tmp_path) -> None:
    row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "q8",
        "prompt_id": "boolean_int_lists",
        "input_token_ids": [10, True],
        "target_token_ids": [False],
        "positions": [True],
        "logit_shard": "teacher.safetensors",
        "topk_ids": [[0, 1, 2]],
        "topk_logprobs": [[-0.1, -0.2, -0.3]],
        "target_logprobs": [-0.1],
        "teacher_top1_ids": [0],
        "full_logits_available": False,
    })

    report = validate_teacher_cache_row(row, cache_root=tmp_path, row_index=0, min_top_k=2)

    assert report["ok"] is False
    assert any("input_token_ids[1] must be an integer" in error for error in report["errors"])
    assert any("target_token_ids[0] must be an integer" in error for error in report["errors"])
    assert any("positions[0] must be an integer" in error for error in report["errors"])


def test_validate_teacher_cache_row_rejects_negative_token_ids(tmp_path) -> None:
    row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "bf16_source",
        "prompt_id": "negative_token_id",
        "input_token_ids": [-10, -1],
        "target_token_ids": [-1],
        "positions": [0],
        "logit_shard": "teacher.safetensors",
        "logit_tensor": "logits",
        "topk_ids": [[0, 1, 2]],
        "topk_logprobs": [[-0.1, -0.2, -0.3]],
        "target_logprobs": [-0.1],
        "teacher_top1_ids": [0],
        "full_logits_available": True,
    })

    report = validate_teacher_cache_row(row, cache_root=tmp_path, row_index=0)

    assert report["ok"] is False
    assert any("input_token_ids[0] must be non-negative" in error for error in report["errors"])
    assert any("input_token_ids[1] must be non-negative" in error for error in report["errors"])
    assert any("target_token_ids[0] must be non-negative" in error for error in report["errors"])


def test_validate_teacher_cache_row_rejects_target_outside_full_logits_vocab(tmp_path) -> None:
    mx.save_safetensors(
        str(tmp_path / "teacher.safetensors"),
        {
            "logits": mx.array(np.ones((1, 4), dtype=np.float16)),
            "topk_ids": mx.array(np.array([[0, 1, 2, 3]], dtype=np.int32)),
            "topk_logprobs": mx.array(np.array([[-0.1, -0.2, -0.3, -0.4]], dtype=np.float32)),
            "target_logprobs": mx.array(np.array([-0.1], dtype=np.float32)),
            "teacher_top1_ids": mx.array(np.array([0], dtype=np.int32)),
        },
    )
    row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "bf16_source",
        "prompt_id": "target_outside_vocab",
        "input_token_ids": [10, 99],
        "target_token_ids": [99],
        "positions": [0],
        "logit_shard": "teacher.safetensors",
        "logit_tensor": "logits",
        "topk_tensor": "topk_ids",
        "topk_logprob_tensor": "topk_logprobs",
        "target_logprob_tensor": "target_logprobs",
        "teacher_top1_tensor": "teacher_top1_ids",
        "full_logits_available": True,
    })

    report = validate_teacher_cache_row(row, cache_root=tmp_path, row_index=0)

    assert report["ok"] is False
    assert any(
        "target_token_ids[0]=99 must be less than logits vocab size 4" in error
        for error in report["errors"]
    )


def test_validate_teacher_cache_row_rejects_shards_outside_cache_root(tmp_path) -> None:
    base_row = _with_memory_counters({
        "schema_version": 1,
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "test",
        "teacher_kind": "bf16_source",
        "prompt_id": "path_escape",
        "input_token_ids": [10, 11],
        "target_token_ids": [11],
        "positions": [0],
        "logit_tensor": "logits",
        "topk_ids": [[0, 1, 2]],
        "topk_logprobs": [[-0.1, -0.2, -0.3]],
        "target_logprobs": [-0.1],
        "teacher_top1_ids": [0],
        "full_logits_available": True,
    })

    traversal_report = validate_teacher_cache_row(
        {**base_row, "logit_shard": "../teacher.safetensors"},
        cache_root=tmp_path,
        row_index=0,
    )
    absolute_report = validate_teacher_cache_row(
        {**base_row, "logit_shard": str(tmp_path.parent / "teacher.safetensors")},
        cache_root=tmp_path,
        row_index=1,
    )

    assert traversal_report["ok"] is False
    assert any("escapes cache_root" in error for error in traversal_report["errors"])
    assert absolute_report["ok"] is False
    assert any("must be relative to cache_root" in error for error in absolute_report["errors"])
