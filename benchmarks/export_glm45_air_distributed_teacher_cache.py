from __future__ import annotations

import argparse
import gc
import json
import platform
import posixpath
import re
import resource
import shlex
import socket
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm.models.base import create_attention_mask
from mlx_lm.utils import load_model, load_tokenizer

from keep.quality.prompts import (
    QualityPrompt,
    get_quality_prompt_by_id,
    get_quality_prompt_ids,
    get_quality_prompt_set_names,
    get_quality_prompts,
    quality_prompt_metadata,
)


GLM45_AIR_MODEL_ID = "zai-org/GLM-4.5-Air"
GLM45_AIR_REVISION = "a24ceef6ce4f3536971efe9b778bdaa1bab18daa"
DEFAULT_ROUTE_COVERAGE_LAYERS = (31, 36, 41)
_ROUTE_TOP_EXPERT_LIMIT = 16


@dataclass(frozen=True)
class _TopKOnlyPayload:
    topk_ids: np.ndarray
    topk_logprobs: np.ndarray
    target_logprobs: np.ndarray
    teacher_top1_ids: np.ndarray


class _PipelineGroup:
    def __init__(self, rank: int, size: int) -> None:
        self._rank = int(rank)
        self._size = int(size)

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


class _RouteTraceStopAfterGate(RuntimeError):
    def __init__(self, layer_index: int) -> None:
        super().__init__(f"route trace stopped after layer {layer_index} gate")
        self.layer_index = int(layer_index)


class _RouteGateProxy:
    def __init__(self, gate: Any, *, layer_index: int, tracer: "_RouteCoverageTracer") -> None:
        self.gate = gate
        self.layer_index = int(layer_index)
        self.tracer = tracer
        self._glm_route_trace_original_gate = gate

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        inds, scores = self.gate(x)
        self.tracer.record(self.layer_index, inds, scores)
        if self.tracer.should_stop_after_gate(self.layer_index):
            raise _RouteTraceStopAfterGate(self.layer_index)
        return inds, scores

    def __getattr__(self, name: str) -> Any:
        return getattr(self.gate, name)


class _RouteMethodProxy:
    def __init__(self, route: Any, *, layer_index: int, tracer: "_RouteCoverageTracer") -> None:
        self.route = route
        self.layer_index = int(layer_index)
        self.tracer = tracer
        self._glm_route_trace_original_route = route

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        inds, scores = self.route(x)
        self.tracer.record(self.layer_index, inds, scores)
        if self.tracer.should_stop_after_gate(self.layer_index):
            raise _RouteTraceStopAfterGate(self.layer_index)
        return inds, scores

    def __getattr__(self, name: str) -> Any:
        return getattr(self.route, name)


class _RouteCoverageTracer:
    def __init__(
        self,
        layers: tuple[int, ...] = DEFAULT_ROUTE_COVERAGE_LAYERS,
        *,
        include_trace: bool = False,
        stop_after_gate_layer: int | None = None,
    ) -> None:
        self.layers = {int(layer) for layer in layers}
        self.include_trace = bool(include_trace)
        self.stop_after_gate_layer = (
            int(stop_after_gate_layer) if stop_after_gate_layer is not None else None
        )
        self._events: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        self._installed_layers: set[int] = set()

    def install(self, language_model: Any) -> None:
        for layer_index, layer in enumerate(getattr(language_model, "layers", ())):
            if layer_index not in self.layers or layer is None:
                continue
            mlp = getattr(layer, "mlp", None)
            route = getattr(mlp, "route", None)
            if route is not None:
                original_route = getattr(route, "_glm_route_trace_original_route", route)
                mlp.route = _RouteMethodProxy(original_route, layer_index=layer_index, tracer=self)
                self._installed_layers.add(layer_index)
                continue
            gate = getattr(mlp, "gate", None)
            if gate is None:
                continue
            original_gate = getattr(gate, "_glm_route_trace_original_gate", gate)
            mlp.gate = _RouteGateProxy(original_gate, layer_index=layer_index, tracer=self)
            self._installed_layers.add(layer_index)

    def reset(self) -> None:
        self._events = {}

    def record(self, layer_index: int, inds: mx.array, scores: mx.array) -> None:
        if int(layer_index) not in self.layers:
            return
        mx.eval(inds, scores)
        inds_np = np.asarray(inds, dtype=np.int32).copy()
        scores_np = np.asarray(scores.astype(mx.float32), dtype=np.float32).copy()
        self._events.setdefault(int(layer_index), []).append((inds_np, scores_np))

    def should_stop_after_gate(self, layer_index: int) -> bool:
        return self.stop_after_gate_layer is not None and int(layer_index) == self.stop_after_gate_layer

    def summary(self, *, row_intent: str | None) -> dict[str, dict[str, Any]]:
        return _summarize_route_coverage_events(
            self._events,
            row_intent=row_intent,
            installed_layers=self._installed_layers,
            include_trace=self.include_trace,
        )


def _summarize_route_coverage_events(
    events: dict[int, list[tuple[np.ndarray, np.ndarray]]],
    *,
    row_intent: str | None,
    installed_layers: set[int] | None = None,
    include_trace: bool = False,
) -> dict[str, dict[str, Any]]:
    coverage: dict[str, dict[str, Any]] = {}
    present_layers = sorted(set(events) | set(installed_layers or set()))
    for layer_index in present_layers:
        layer_events = events.get(layer_index, [])
        if not layer_events:
            continue
        token_count = 0
        route_count = 0
        ids_chunks: list[np.ndarray] = []
        score_chunks: list[np.ndarray] = []
        for inds, scores in layer_events:
            if inds.size == 0:
                continue
            token_count += int(np.prod(inds.shape[:-1])) if inds.ndim > 1 else int(inds.size)
            route_count += int(inds.size)
            ids_chunks.append(inds.reshape(-1).astype(np.int64, copy=False))
            if scores.size:
                score_chunks.append(scores.reshape(-1).astype(np.float64, copy=False))
        if not ids_chunks:
            continue
        ids_flat = np.concatenate(ids_chunks)
        unique, counts = np.unique(ids_flat, return_counts=True)
        ordered = sorted(
            zip(unique.tolist(), counts.tolist(), strict=True),
            key=lambda item: (-int(item[1]), int(item[0])),
        )
        top_experts = [
            {"expert": int(expert), "count": int(count)}
            for expert, count in ordered[:_ROUTE_TOP_EXPERT_LIMIT]
        ]
        layer_summary: dict[str, Any] = {
            "token_count": int(token_count),
            "route_count": int(route_count),
            "unique_expert_count": int(len(unique)),
            "unique_experts": [int(expert) for expert in sorted(unique.tolist())],
            "top_experts": top_experts,
            "glu_code_route_count": 0,
        }
        if score_chunks:
            scores_flat = np.concatenate(score_chunks)
            layer_summary.update(
                {
                    "router_score_mean": float(np.mean(scores_flat)),
                    "router_score_min": float(np.min(scores_flat)),
                    "router_score_max": float(np.max(scores_flat)),
                }
            )
        if include_trace:
            layer_summary["route_trace"] = _route_trace_payload(layer_events)
        if int(layer_index) == 41 and row_intent == "layer41_glu_code_route":
            layer_summary["glu_code_route_count"] = int(route_count)
        coverage[str(layer_index)] = layer_summary
    return coverage


def _route_trace_payload(layer_events: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    token_expert_indices: list[list[int]] = []
    token_router_scores: list[list[float]] = []
    top_k: int | None = None
    for inds, scores in layer_events:
        if inds.size == 0:
            continue
        if inds.ndim == 0:
            token_indices = inds.reshape(1, 1)
        elif inds.ndim == 1:
            token_indices = inds.reshape(-1, 1)
        else:
            token_indices = inds.reshape(-1, inds.shape[-1])
        if top_k is None:
            top_k = int(token_indices.shape[-1])
        elif int(token_indices.shape[-1]) != top_k:
            raise ValueError(f"route trace top_k changed within one layer: {top_k} vs {token_indices.shape[-1]}")
        token_expert_indices.extend(
            [int(value) for value in token_row]
            for token_row in token_indices
        )
        if scores.size:
            if scores.ndim == 0:
                token_scores = scores.reshape(1, 1)
            elif scores.ndim == 1:
                token_scores = scores.reshape(-1, 1)
            else:
                token_scores = scores.reshape(-1, scores.shape[-1])
            if token_scores.shape != token_indices.shape:
                raise ValueError(
                    f"route trace scores shape {token_scores.shape} does not match indices {token_indices.shape}"
                )
            token_router_scores.extend(
                [float(value) for value in score_row]
                for score_row in token_scores
            )
    payload: dict[str, Any] = {
        "top_k": int(top_k or 0),
        "token_expert_indices": token_expert_indices,
    }
    if token_router_scores:
        payload["token_router_scores"] = token_router_scores
    return payload


def _custom_layer_bounds(
    *,
    num_hidden_layers: int,
    rank: int,
    pipeline_size: int,
    layer_split: int | None,
) -> tuple[int, int] | None:
    if layer_split is None:
        return None
    if pipeline_size != 2:
        raise ValueError("--layer-split is currently supported only with two distributed ranks")
    if rank not in {0, 1}:
        raise ValueError("--layer-split expects ranks 0 and 1")
    if layer_split <= 0 or layer_split >= num_hidden_layers:
        raise ValueError(
            f"--layer-split must be between 1 and {num_hidden_layers - 1}, got {layer_split}"
        )
    if rank == 0:
        return layer_split, num_hidden_layers
    return 0, layer_split


def _apply_pipeline_split(
    language_model: Any,
    *,
    rank: int,
    pipeline_size: int,
    layer_split: int | None,
    rank_start_layer: int | None = None,
    rank_stop_after_layer: int | None = None,
    rank0_start_layer: int | None = None,
    rank0_stop_after_layer: int | None = None,
) -> None:
    if layer_split is None:
        language_model.pipeline(_PipelineGroup(rank, pipeline_size))
        return

    start_idx, end_idx = _custom_layer_bounds(
        num_hidden_layers=len(language_model.layers),
        rank=rank,
        pipeline_size=pipeline_size,
        layer_split=layer_split,
    )
    if rank0_start_layer is not None:
        if rank != 0:
            raise ValueError("--rank0-start-layer only applies to rank 0")
        if rank_start_layer is not None:
            raise ValueError("rank start layer was specified twice")
        rank_start_layer = rank0_start_layer
    if rank0_stop_after_layer is not None:
        if rank != 0:
            raise ValueError("--rank0-stop-after-layer only applies to rank 0")
        if rank_stop_after_layer is not None:
            raise ValueError("rank stop layer was specified twice")
        rank_stop_after_layer = rank0_stop_after_layer
    if rank_start_layer is not None:
        if rank_start_layer < start_idx or rank_start_layer >= end_idx:
            raise ValueError(
                "--rank-start-layer must be within rank layer bounds "
                f"{start_idx}..{end_idx - 1}, got {rank_start_layer}"
            )
        start_idx = rank_start_layer
    if rank_stop_after_layer is not None:
        if rank_stop_after_layer < start_idx or rank_stop_after_layer >= end_idx:
            raise ValueError(
                "--rank-stop-after-layer must be within rank layer bounds "
                f"{start_idx}..{end_idx - 1}, got {rank_stop_after_layer}"
            )
        end_idx = rank_stop_after_layer + 1
    language_model.pipeline_rank = rank
    language_model.pipeline_size = pipeline_size
    language_model.start_idx = start_idx
    language_model.end_idx = end_idx
    language_model.layers = language_model.layers[:end_idx]
    language_model.layers[:start_idx] = [None] * start_idx


_VM_STAT_RE = re.compile(r"^\s*(?P<name>[^:]+):\s+(?P<value>[0-9]+)\.?\s*$")
_VM_STAT_PAGE_SIZE_RE = re.compile(r"page size of (?P<page_size>[0-9]+) bytes")
_DTYPE_BYTES = {"BF16": 2, "F16": 2}


def _parse_rank_view_roots(raw: str) -> dict[int, Path]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("--rank-view-roots-json must be a JSON object")
    roots: dict[int, Path] = {}
    for key, value in payload.items():
        try:
            rank = int(key)
        except (TypeError, ValueError) as error:
            raise argparse.ArgumentTypeError(f"rank key {key!r} is not an integer") from error
        if rank < 0:
            raise argparse.ArgumentTypeError(f"rank key {key!r} must be non-negative")
        if not isinstance(value, str) or not value:
            raise argparse.ArgumentTypeError(f"rank {rank} root must be a non-empty string")
        roots[rank] = Path(value)
    if not roots:
        raise argparse.ArgumentTypeError("--rank-view-roots-json must contain at least one rank")
    return roots


def _parse_local_sequential_stage_view_roots(raw: str) -> dict[str, dict[int, Path]]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError(
            "--local-sequential-stage-view-roots-json must be a JSON object"
        )
    stage_roots: dict[str, dict[int, Path]] = {}
    for stage, roots_payload in payload.items():
        if not isinstance(stage, str) or not stage:
            raise argparse.ArgumentTypeError("stage view root keys must be non-empty strings")
        if not isinstance(roots_payload, dict):
            raise argparse.ArgumentTypeError(f"stage {stage!r} roots must be a JSON object")
        roots: dict[int, Path] = {}
        for key, value in roots_payload.items():
            try:
                rank = int(key)
            except (TypeError, ValueError) as error:
                raise argparse.ArgumentTypeError(
                    f"stage {stage!r} rank key {key!r} is not an integer"
                ) from error
            if rank < 0:
                raise argparse.ArgumentTypeError(
                    f"stage {stage!r} rank key {key!r} must be non-negative"
                )
            if not isinstance(value, str) or not value:
                raise argparse.ArgumentTypeError(
                    f"stage {stage!r} rank {rank} root must be a non-empty string"
                )
            roots[rank] = Path(value)
        if not roots:
            raise argparse.ArgumentTypeError(f"stage {stage!r} must contain at least one rank root")
        stage_roots[stage] = roots
    if not stage_roots:
        raise argparse.ArgumentTypeError(
            "--local-sequential-stage-view-roots-json must contain at least one stage"
        )
    return stage_roots


def _gb_to_bytes(value: float | None, *, name: str, allow_zero: bool = False) -> int | None:
    if value is None:
        return None
    if value < 0 or (value == 0 and not allow_zero):
        requirement = "non-negative" if allow_zero else "positive"
        raise argparse.ArgumentTypeError(f"{name} must be {requirement}")
    return int(value * 1024**3)


def _apply_mlx_memory_settings(
    *,
    memory_limit_bytes: int | None,
    cache_limit_bytes: int | None,
    wired_limit_bytes: int | None,
) -> dict[str, int | None]:
    settings: dict[str, int | None] = {
        "mlx_memory_limit_bytes": memory_limit_bytes,
        "mlx_cache_limit_bytes": cache_limit_bytes,
        "mlx_wired_limit_bytes": wired_limit_bytes,
        "previous_mlx_memory_limit_bytes": None,
        "previous_mlx_cache_limit_bytes": None,
        "previous_mlx_wired_limit_bytes": None,
    }
    if memory_limit_bytes is not None:
        settings["previous_mlx_memory_limit_bytes"] = int(mx.set_memory_limit(memory_limit_bytes))
    if cache_limit_bytes is not None:
        settings["previous_mlx_cache_limit_bytes"] = int(mx.set_cache_limit(cache_limit_bytes))
    if wired_limit_bytes is not None:
        settings["previous_mlx_wired_limit_bytes"] = int(mx.set_wired_limit(wired_limit_bytes))
    if cache_limit_bytes == 0:
        mx.clear_cache()
    return settings


def _release_local_stage_memory() -> None:
    gc.collect()
    mx.clear_cache()


def _local_hidden_dtype_name(hidden: mx.array) -> str:
    dtype = hidden.dtype
    if dtype == mx.bfloat16:
        return "bfloat16"
    if dtype == mx.float16:
        return "float16"
    if dtype == mx.float32:
        return "float32"
    return str(dtype)


def _spill_local_hidden_to_host(hidden: mx.array) -> dict[str, Any]:
    dtype_name = _local_hidden_dtype_name(hidden)
    if dtype_name == "bfloat16":
        host_array = np.asarray(hidden.astype(mx.float32), dtype=np.float32)
    elif dtype_name == "float16":
        host_array = np.asarray(hidden, dtype=np.float16)
    elif dtype_name == "float32":
        host_array = np.asarray(hidden, dtype=np.float32)
    else:
        raise ValueError(f"unsupported local hidden dtype for host spill: {dtype_name}")
    return {
        "array": host_array.copy(),
        "mlx_dtype": dtype_name,
        "host_shape": [int(dim) for dim in host_array.shape],
        "host_nbytes": int(host_array.nbytes),
    }


def _restore_local_hidden_from_host(spill: dict[str, Any]) -> mx.array:
    hidden = mx.array(spill["array"])
    dtype_name = str(spill["mlx_dtype"])
    if dtype_name == "bfloat16":
        return hidden.astype(mx.bfloat16)
    if dtype_name == "float16":
        return hidden.astype(mx.float16)
    if dtype_name == "float32":
        return hidden.astype(mx.float32)
    raise ValueError(f"unsupported local hidden dtype for host restore: {dtype_name}")


def _prompt_tokens(tokenizer: Any, prompt: QualityPrompt) -> list[int]:
    tokens = list(tokenizer.encode(prompt.text, add_special_tokens=False))
    if prompt.context_tokens is None:
        return tokens
    if not tokens:
        raise ValueError(f"quality prompt {prompt.prompt_id!r} tokenized to zero tokens")
    expanded = list(tokens)
    while len(expanded) < prompt.context_tokens:
        expanded.extend(tokens)
    return expanded[: prompt.context_tokens]


def _collect_vm_stat_counts() -> dict[str, int] | None:
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["vm_stat"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    counts: dict[str, int] = {"page_size": 16384}
    for line in result.stdout.splitlines():
        page_size_match = _VM_STAT_PAGE_SIZE_RE.search(line)
        if page_size_match is not None:
            counts["page_size"] = int(page_size_match.group("page_size"))
            continue
        match = _VM_STAT_RE.match(line)
        if match is None:
            continue
        key = match.group("name").strip().lower().replace(" ", "_")
        if key in {"pageouts", "swapouts", "pages_free", "pages_speculative"}:
            counts[key] = int(match.group("value"))
    if "pageouts" not in counts or "swapouts" not in counts:
        return None
    return counts


def _current_rss_bytes() -> int:
    ru_maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return ru_maxrss
    return ru_maxrss * 1024


def _delta(
    counts: dict[str, int] | None,
    previous: dict[str, int] | None,
    key: str,
) -> int | None:
    if counts is None or previous is None:
        return None
    if key not in counts or key not in previous:
        return None
    return int(counts[key]) - int(previous[key])


def _collect_metric_snapshot(previous_vm_stat_counts: dict[str, int] | None) -> dict[str, int | None]:
    vm_counts = _collect_vm_stat_counts()
    return {
        "mlx_active_bytes": int(mx.get_active_memory()),
        "mlx_peak_bytes": int(mx.get_peak_memory()),
        "mlx_cache_bytes": int(mx.get_cache_memory()),
        "rss_bytes": _current_rss_bytes(),
        "pageouts_total": None if vm_counts is None else vm_counts.get("pageouts"),
        "swapouts_total": None if vm_counts is None else vm_counts.get("swapouts"),
        "pageouts_delta": _delta(vm_counts, previous_vm_stat_counts, "pageouts"),
        "swapouts_delta": _delta(vm_counts, previous_vm_stat_counts, "swapouts"),
    }


def _all_max_optional_int(value: int | None) -> int | None:
    encoded = -1 if value is None else int(value)
    result = mx.distributed.all_max(mx.array([encoded], dtype=mx.int64))
    mx.eval(result)
    decoded = int(result[0].item())
    return None if decoded < 0 else decoded


def _all_max_float(value: float) -> float:
    result = mx.distributed.all_max(mx.array([float(value)], dtype=mx.float32))
    mx.eval(result)
    return float(result[0].item())


def _all_rank_float(value: float, *, rank: int, distributed_size: int) -> list[float]:
    values = [0.0] * distributed_size
    values[rank] = float(value)
    result = mx.distributed.all_sum(mx.array(values, dtype=mx.float32))
    mx.eval(result)
    return [float(item) for item in result.tolist()]


def _all_rank_optional_int(
    value: int | None,
    *,
    rank: int,
    distributed_size: int,
) -> list[int | None]:
    values = [0] * distributed_size
    known = [0] * distributed_size
    if value is not None:
        values[rank] = int(value)
        known[rank] = 1
    result_values = mx.distributed.all_sum(mx.array(values, dtype=mx.int64))
    result_known = mx.distributed.all_sum(mx.array(known, dtype=mx.int32))
    mx.eval(result_values, result_known)
    decoded_values = [int(item) for item in result_values.tolist()]
    decoded_known = [int(item) for item in result_known.tolist()]
    return [decoded_values[i] if decoded_known[i] else None for i in range(distributed_size)]


def _aggregate_rank_metrics(rank_metrics: list[dict[str, int | float | None]]) -> dict[str, int | float | None]:
    if not rank_metrics:
        return {}
    aggregate: dict[str, int | float | None] = {}
    for key in rank_metrics[0]:
        if key == "rank":
            continue
        known_values = [row[key] for row in rank_metrics if row.get(key) is not None]
        aggregate[key] = max(known_values) if known_values else None
    return aggregate


def _collect_rank_metrics(
    metrics: dict[str, int | None],
    *,
    elapsed_seconds: float,
    rank: int,
    distributed_size: int,
) -> list[dict[str, int | float | None]]:
    elapsed_by_rank = _all_rank_float(
        elapsed_seconds,
        rank=rank,
        distributed_size=distributed_size,
    )
    metric_fields = [
        "mlx_active_bytes",
        "mlx_peak_bytes",
        "mlx_cache_bytes",
        "rss_bytes",
        "pageouts_total",
        "swapouts_total",
        "pageouts_delta",
        "swapouts_delta",
    ]
    values_by_field = {
        field: _all_rank_optional_int(
            metrics[field],
            rank=rank,
            distributed_size=distributed_size,
        )
        for field in metric_fields
    }
    return [
        {
            "rank": metric_rank,
            "elapsed_seconds": elapsed_by_rank[metric_rank],
            **{field: values_by_field[field][metric_rank] for field in metric_fields},
        }
        for metric_rank in range(distributed_size)
    ]


def _aggregate_metrics(metrics: dict[str, int | None], *, elapsed_seconds: float) -> dict[str, int | float | None]:
    return {
        "elapsed_seconds": _all_max_float(elapsed_seconds),
        "mlx_active_bytes": _all_max_optional_int(metrics["mlx_active_bytes"]),
        "mlx_peak_bytes": _all_max_optional_int(metrics["mlx_peak_bytes"]),
        "mlx_cache_bytes": _all_max_optional_int(metrics["mlx_cache_bytes"]),
        "rss_bytes": _all_max_optional_int(metrics["rss_bytes"]),
        "pageouts_total": _all_max_optional_int(metrics["pageouts_total"]),
        "swapouts_total": _all_max_optional_int(metrics["swapouts_total"]),
        "pageouts_delta": _all_max_optional_int(metrics["pageouts_delta"]),
        "swapouts_delta": _all_max_optional_int(metrics["swapouts_delta"]),
    }


def _load_rank_model_and_tokenizer(
    rank_dir: Path,
    *,
    rank: int,
    pipeline_size: int,
    layer_split: int | None,
    rank_start_layer: int | None = None,
    rank_stop_after_layer: int | None = None,
    rank0_start_layer: int | None = None,
    rank0_stop_after_layer: int | None = None,
) -> tuple[Any, Any]:
    if not rank_dir.exists():
        raise FileNotFoundError(f"rank {rank} view root does not exist: {rank_dir}")
    model, config = load_model(rank_dir, lazy=True, strict=False)
    _apply_pipeline_split(
        model.model,
        rank=rank,
        pipeline_size=pipeline_size,
        layer_split=layer_split,
        rank_start_layer=rank_start_layer,
        rank_stop_after_layer=rank_stop_after_layer,
        rank0_start_layer=rank0_start_layer,
        rank0_stop_after_layer=rank0_stop_after_layer,
    )
    tokenizer = load_tokenizer(rank_dir, eos_token_ids=config.get("eos_token_id", None))
    return model, tokenizer


def _forward_pipeline_rank0_output(
    model: Any,
    input_token_ids: list[int],
    *,
    rank: int,
    pipeline_size: int,
    return_hidden: bool,
    stop_after_layer: int | None = None,
    stop_after_route_gate: int | None = None,
    normalize_output: bool = True,
    diagnose_hidden_stages: bool = False,
) -> mx.array:
    inputs = mx.array([input_token_ids], dtype=mx.int32)
    language_model = model.model
    cache = [None] * len(language_model.pipeline_layers)
    hidden_stage_reports: list[dict[str, Any]] = []

    def record_hidden_stage(stage: str, values: mx.array, *, layer_index: int | None = None) -> None:
        if not diagnose_hidden_stages:
            return
        report = _array_finite_summary(
            "hidden",
            np.asarray(values.astype(mx.float32), dtype=np.float32),
        )
        report["stage"] = stage
        if layer_index is not None:
            report["layer_index"] = int(layer_index)
        hidden_stage_reports.append(report)

    if rank < pipeline_size - 1:
        h = mx.zeros((inputs.shape[0], inputs.shape[1], model.args.hidden_size), dtype=mx.bfloat16)
    else:
        h = language_model.embed_tokens(inputs)
        record_hidden_stage("after_embed", h)
    mask = create_attention_mask(h, cache[0])

    if rank < pipeline_size - 1:
        h = mx.distributed.recv_like(h, (rank + 1))
        record_hidden_stage("after_recv", h)

    start_idx = int(getattr(language_model, "start_idx", 0) or 0)
    for local_layer_index, (layer, layer_cache) in enumerate(zip(language_model.pipeline_layers, cache)):
        layer_index = start_idx + local_layer_index
        try:
            h = layer(h, mask, cache=layer_cache)
        except _RouteTraceStopAfterGate as stop:
            if stop_after_route_gate is None or stop.layer_index != stop_after_route_gate:
                raise
            break
        record_hidden_stage("after_layer", h, layer_index=layer_index)
        if stop_after_layer is not None and layer_index >= stop_after_layer:
            break

    if rank != 0:
        _raise_for_nonfinite_hidden_stages(hidden_stage_reports)
        h = mx.distributed.send(h, (rank - 1) % pipeline_size)
        if cache[-1] is not None:
            cache[-1].keys = mx.depends(cache[-1].keys, h)

    if not normalize_output:
        _raise_for_nonfinite_hidden_stages(hidden_stage_reports)
        return h

    if pipeline_size > 1:
        record_hidden_stage("before_gather", h)
        h = mx.distributed.all_gather(h)[: h.shape[0]]
        record_hidden_stage("after_gather", h)

    hidden = language_model.norm(h)
    record_hidden_stage("after_norm", hidden)
    _raise_for_nonfinite_hidden_stages(hidden_stage_reports)
    if rank == 0 and not return_hidden:
        return model.lm_head(hidden)
    return hidden


def _forward_local_lower_stage(model: Any, input_token_ids: list[int]) -> mx.array:
    inputs = mx.array([input_token_ids], dtype=mx.int32)
    language_model = model.model
    cache = [None] * len(language_model.pipeline_layers)
    h = language_model.embed_tokens(inputs)
    mask = create_attention_mask(h, cache[0] if cache else None)
    for layer, layer_cache in zip(language_model.pipeline_layers, cache):
        h = layer(h, mask, cache=layer_cache)
    mx.eval(h)
    return h


def _forward_local_embed_stage(model: Any, input_token_ids: list[int]) -> mx.array:
    inputs = mx.array([input_token_ids], dtype=mx.int32)
    h = model.model.embed_tokens(inputs)
    mx.eval(h)
    return h


def _forward_local_hidden_stage(model: Any, hidden: mx.array) -> mx.array:
    language_model = model.model
    cache = [None] * len(language_model.pipeline_layers)
    h = hidden
    mask = create_attention_mask(h, cache[0] if cache else None)
    for layer, layer_cache in zip(language_model.pipeline_layers, cache):
        h = layer(h, mask, cache=layer_cache)
    mx.eval(h)
    return h


def _forward_local_upper_stage(
    model: Any,
    lower_hidden: mx.array,
    *,
    return_hidden: bool,
    normalize_output: bool = True,
) -> mx.array:
    language_model = model.model
    cache = [None] * len(language_model.pipeline_layers)
    h = lower_hidden
    mask = create_attention_mask(h, cache[0] if cache else None)
    for layer, layer_cache in zip(language_model.pipeline_layers, cache):
        h = layer(h, mask, cache=layer_cache)
    if not normalize_output:
        mx.eval(h)
        return h
    hidden = language_model.norm(h)
    mx.eval(hidden)
    if return_hidden:
        return hidden
    return model.lm_head(hidden)


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        header_size_raw = handle.read(8)
        if len(header_size_raw) != 8:
            raise ValueError(f"{path} is not a valid safetensors file")
        header_size = struct.unpack("<Q", header_size_raw)[0]
        return header_size, json.loads(handle.read(header_size))


def _lm_head_index_entry(rank_dir: Path) -> tuple[Path, str]:
    index_path = rank_dir / "model.safetensors.index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index_path} does not contain a weight_map object")
    shard = weight_map.get("lm_head.weight")
    if not isinstance(shard, str) or not shard:
        raise KeyError(f"{index_path} does not map lm_head.weight")
    return rank_dir / shard, "lm_head.weight"


def _read_matrix_rows(path: Path, tensor_name: str, *, start: int, end: int) -> mx.array:
    header_size, header = _read_safetensors_header(path)
    tensor_info = header.get(tensor_name)
    if tensor_info is None:
        raise KeyError(f"{tensor_name!r} not found in {path}")
    dtype = str(tensor_info["dtype"])
    if dtype not in _DTYPE_BYTES:
        raise ValueError(f"{tensor_name} must be BF16 or F16 for streamed reads, got {dtype}")
    shape = tuple(int(dim) for dim in tensor_info["shape"])
    if len(shape) != 2:
        raise ValueError(f"{tensor_name} must be a matrix, got shape {shape}")
    rows, cols = shape
    if start < 0 or end <= start or end > rows:
        raise ValueError(f"invalid row slice {start}:{end} for {tensor_name} with {rows} rows")
    tensor_start, _tensor_end = (int(offset) for offset in tensor_info["data_offsets"])
    row_bytes = cols * _DTYPE_BYTES[dtype]
    byte_start = 8 + header_size + tensor_start + start * row_bytes
    byte_count = (end - start) * row_bytes
    with path.open("rb") as handle:
        handle.seek(byte_start)
        raw = handle.read(byte_count)
    if len(raw) != byte_count:
        raise ValueError(f"short read for {tensor_name} rows {start}:{end} from {path}")
    if dtype == "BF16":
        words = np.frombuffer(raw, dtype="<u2").copy()
        return mx.array(words).view(mx.bfloat16).reshape((end - start, cols))
    values = np.frombuffer(raw, dtype="<f2").reshape((end - start, cols)).copy()
    return mx.array(values)


def _stream_lm_head_logits(
    hidden: mx.array,
    *,
    rank_dir: Path,
    input_token_ids: list[int],
    max_positions: int | None,
    chunk_rows: int,
) -> np.ndarray:
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if len(input_token_ids) < 2:
        raise ValueError("teacher cache prompts need at least two input tokens")
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError(f"hidden states must have shape [1, tokens, hidden], got {hidden.shape}")
    position_count = min(len(input_token_ids) - 1, hidden.shape[1])
    if max_positions is not None:
        position_count = min(position_count, max_positions)
    if position_count <= 0:
        raise ValueError("no teacher positions selected")

    selected_hidden = hidden[0, :position_count, :]
    shard_path, tensor_name = _lm_head_index_entry(rank_dir)
    _header_size, header = _read_safetensors_header(shard_path)
    tensor_info = header[tensor_name]
    vocab_size, hidden_size = (int(dim) for dim in tensor_info["shape"])
    if hidden_size != selected_hidden.shape[-1]:
        raise ValueError(
            f"lm_head hidden size {hidden_size} does not match model hidden {selected_hidden.shape[-1]}"
        )

    chunks: list[np.ndarray] = []
    for start in range(0, vocab_size, chunk_rows):
        end = min(start + chunk_rows, vocab_size)
        weight = _read_matrix_rows(shard_path, tensor_name, start=start, end=end)
        logits_chunk = selected_hidden @ weight.T
        mx.eval(logits_chunk)
        chunks.append(np.asarray(logits_chunk.astype(mx.float32)))
        del weight, logits_chunk
        mx.clear_cache()
    return np.concatenate(chunks, axis=-1)


def _array_finite_summary(name: str, values: Any) -> dict[str, Any]:
    array = np.asarray(values)
    summary: dict[str, Any] = {
        "name": name,
        "shape": [int(dim) for dim in array.shape],
        "dtype": str(array.dtype),
        "size": int(array.size),
    }
    if array.size == 0:
        summary.update({"finite": True, "finite_count": 0, "nonfinite_count": 0})
        return summary
    if not np.issubdtype(array.dtype, np.number):
        summary.update({"finite": True, "finite_count": int(array.size), "nonfinite_count": 0})
        return summary

    finite_mask = np.isfinite(array)
    finite_count = int(np.count_nonzero(finite_mask))
    nonfinite_count = int(array.size - finite_count)
    summary.update(
        {
            "finite": nonfinite_count == 0,
            "finite_count": finite_count,
            "nonfinite_count": nonfinite_count,
            "nan_count": int(np.count_nonzero(np.isnan(array))),
            "posinf_count": int(np.count_nonzero(np.isposinf(array))),
            "neginf_count": int(np.count_nonzero(np.isneginf(array))),
        }
    )
    if finite_count:
        finite_values = array[finite_mask].astype(np.float64, copy=False)
        summary["finite_min"] = float(np.min(finite_values))
        summary["finite_max"] = float(np.max(finite_values))
    return summary


def _raise_for_nonfinite_hidden_stages(stage_reports: list[dict[str, Any]]) -> None:
    if not stage_reports:
        return
    first_nonfinite = next(
        (report for report in stage_reports if not bool(report.get("finite"))),
        None,
    )
    if first_nonfinite is None:
        return
    first_stage = {"stage": first_nonfinite["stage"]}
    if "layer_index" in first_nonfinite:
        first_stage["layer_index"] = int(first_nonfinite["layer_index"])
    raise ValueError(
        "rank0 hidden contains non-finite values: "
        + json.dumps(
            {
                "first_nonfinite_stage": first_stage,
                "hidden_stages": stage_reports,
            },
            sort_keys=True,
        )
    )


def _raise_for_nonfinite_streamed_topk_payload(
    *,
    selected_hidden: Any,
    topk_logits: np.ndarray,
    target_logits: np.ndarray,
    running_max: np.ndarray,
    running_exp_sum: np.ndarray,
    log_normalizer: np.ndarray,
    topk_logprobs: np.ndarray,
    target_logprobs: np.ndarray,
    first_nonfinite_chunk: dict[str, int] | None,
) -> None:
    report = {
        "selected_hidden": _array_finite_summary("selected_hidden", selected_hidden),
        "topk_logits": _array_finite_summary("topk_logits", topk_logits),
        "target_logits": _array_finite_summary("target_logits", target_logits),
        "running_max": _array_finite_summary("running_max", running_max),
        "running_exp_sum": _array_finite_summary("running_exp_sum", running_exp_sum),
        "log_normalizer": _array_finite_summary("log_normalizer", log_normalizer),
        "topk_logprobs": _array_finite_summary("topk_logprobs", topk_logprobs),
        "target_logprobs": _array_finite_summary("target_logprobs", target_logprobs),
        "first_nonfinite_chunk": first_nonfinite_chunk,
    }
    if all(
        bool(report[field]["finite"])
        for field in (
            "selected_hidden",
            "topk_logits",
            "target_logits",
            "running_max",
            "running_exp_sum",
            "log_normalizer",
            "topk_logprobs",
            "target_logprobs",
        )
    ):
        return
    raise ValueError(
        "streamed lm_head top-k payload contains non-finite values: "
        + json.dumps(report, sort_keys=True)
    )


def _chunk_topk(logits: np.ndarray, *, ids_offset: int, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    count = min(top_k, logits.shape[-1])
    partition = np.argpartition(-logits, count - 1, axis=-1)[..., :count]
    partition_values = np.take_along_axis(logits, partition, axis=-1)
    order = np.argsort(-partition_values, axis=-1)
    ordered = np.take_along_axis(partition, order, axis=-1)
    return ordered + ids_offset, np.take_along_axis(logits, ordered, axis=-1)


def _merge_topk(
    current_ids: np.ndarray,
    current_logits: np.ndarray,
    new_ids: np.ndarray,
    new_logits: np.ndarray,
    *,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    if current_ids.size == 0:
        merged_ids = new_ids
        merged_logits = new_logits
    else:
        merged_ids = np.concatenate([current_ids, new_ids], axis=-1)
        merged_logits = np.concatenate([current_logits, new_logits], axis=-1)
    keep = min(top_k, merged_logits.shape[-1])
    keep_local, _keep_logits = _chunk_topk(merged_logits, ids_offset=0, top_k=keep)
    return (
        np.take_along_axis(merged_ids, keep_local, axis=-1).astype(np.int32),
        np.take_along_axis(merged_logits, keep_local, axis=-1),
    )


def _stream_lm_head_topk_payload(
    hidden: mx.array,
    *,
    rank_dir: Path,
    input_token_ids: list[int],
    max_positions: int | None,
    chunk_rows: int,
    top_k: int,
) -> _TopKOnlyPayload:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if len(input_token_ids) < 2:
        raise ValueError("teacher cache prompts need at least two input tokens")
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError(f"hidden states must have shape [1, tokens, hidden], got {hidden.shape}")
    position_count = min(len(input_token_ids) - 1, hidden.shape[1])
    if max_positions is not None:
        position_count = min(position_count, max_positions)
    if position_count <= 0:
        raise ValueError("no teacher positions selected")

    selected_hidden = hidden[0, :position_count, :]
    shard_path, tensor_name = _lm_head_index_entry(rank_dir)
    _header_size, header = _read_safetensors_header(shard_path)
    tensor_info = header[tensor_name]
    vocab_size, hidden_size = (int(dim) for dim in tensor_info["shape"])
    if hidden_size != selected_hidden.shape[-1]:
        raise ValueError(
            f"lm_head hidden size {hidden_size} does not match model hidden {selected_hidden.shape[-1]}"
        )
    target_ids = np.asarray(input_token_ids[1 : position_count + 1], dtype=np.int64)
    if np.any(target_ids < 0) or np.any(target_ids >= vocab_size):
        raise ValueError("target token ids must be within the lm_head vocab for top-k streaming")

    top_count = min(top_k, vocab_size)
    topk_ids = np.empty((position_count, 0), dtype=np.int32)
    topk_logits = np.empty((position_count, 0), dtype=np.float64)
    target_logits = np.full((position_count,), -np.inf, dtype=np.float64)
    running_max = np.full((position_count,), -np.inf, dtype=np.float64)
    running_exp_sum = np.zeros((position_count,), dtype=np.float64)
    first_nonfinite_chunk: dict[str, int] | None = None

    for start in range(0, vocab_size, chunk_rows):
        end = min(start + chunk_rows, vocab_size)
        weight = _read_matrix_rows(shard_path, tensor_name, start=start, end=end)
        logits_chunk_mx = selected_hidden @ weight.T
        mx.eval(logits_chunk_mx)
        logits_chunk = np.asarray(logits_chunk_mx.astype(mx.float32), dtype=np.float64)
        if first_nonfinite_chunk is None and not bool(np.all(np.isfinite(logits_chunk))):
            first_nonfinite_chunk = {"start": int(start), "end": int(end)}

        chunk_max = np.max(logits_chunk, axis=-1)
        new_max = np.maximum(running_max, chunk_max)
        running_exp_sum = (
            running_exp_sum * np.exp(running_max - new_max)
            + np.sum(np.exp(logits_chunk - new_max[:, None]), axis=-1)
        )
        running_max = new_max

        local_ids, local_logits = _chunk_topk(logits_chunk, ids_offset=start, top_k=top_count)
        topk_ids, topk_logits = _merge_topk(
            topk_ids,
            topk_logits,
            local_ids.astype(np.int32),
            local_logits,
            top_k=top_count,
        )

        in_chunk = (target_ids >= start) & (target_ids < end)
        if np.any(in_chunk):
            row_indices = np.flatnonzero(in_chunk)
            target_logits[row_indices] = logits_chunk[row_indices, target_ids[row_indices] - start]

        del weight, logits_chunk_mx
        mx.clear_cache()

    log_normalizer = running_max + np.log(running_exp_sum)
    topk_logprobs = (topk_logits - log_normalizer[:, None]).astype(np.float32)
    target_logprobs = (target_logits - log_normalizer).astype(np.float32)
    _raise_for_nonfinite_streamed_topk_payload(
        selected_hidden=np.asarray(selected_hidden.astype(mx.float32), dtype=np.float32),
        topk_logits=topk_logits,
        target_logits=target_logits,
        running_max=running_max,
        running_exp_sum=running_exp_sum,
        log_normalizer=log_normalizer,
        topk_logprobs=topk_logprobs,
        target_logprobs=target_logprobs,
        first_nonfinite_chunk=first_nonfinite_chunk,
    )
    return _TopKOnlyPayload(
        topk_ids=topk_ids.astype(np.int32, copy=False),
        topk_logprobs=topk_logprobs,
        target_logprobs=target_logprobs,
        teacher_top1_ids=topk_ids[:, 0].astype(np.int32, copy=False),
    )


def _selected_prompts(prompt_ids: list[str] | None, *, prompt_set: str) -> list[QualityPrompt]:
    if not prompt_ids:
        return list(get_quality_prompts(prompt_set=prompt_set))
    return [get_quality_prompt_by_id(prompt_id) for prompt_id in prompt_ids]


def _parse_route_trace_layers(values: list[str] | None) -> tuple[int, ...]:
    if not values:
        return DEFAULT_ROUTE_COVERAGE_LAYERS
    parsed: set[int] = set()
    for value in values:
        for part in value.split(","):
            stripped = part.strip()
            if stripped:
                parsed.add(int(stripped))
    if not parsed:
        raise argparse.ArgumentTypeError("--route-trace-layer must include at least one layer")
    return tuple(sorted(parsed))


def _selected_position_count(input_token_ids: list[int], max_positions: int | None) -> int:
    if len(input_token_ids) < 2:
        raise ValueError("teacher cache prompts need at least two input tokens")
    position_count = len(input_token_ids) - 1
    if max_positions is not None:
        position_count = min(position_count, max_positions)
    if position_count <= 0:
        raise ValueError("no teacher positions selected")
    return position_count


def _build_route_trace_only_row(
    *,
    prompt: QualityPrompt,
    prompt_set: str,
    input_token_ids: list[int],
    max_positions: int | None,
    model_id: str,
    revision: str,
    teacher_kind: str,
    route_coverage: dict[str, Any],
    route_trace: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    position_count = _selected_position_count(input_token_ids, max_positions)
    return {
        "schema_version": 1,
        "record_type": "air_route_trace_only",
        "teacher_cache_compatible": False,
        "model_id": model_id,
        "revision": revision,
        "teacher_kind": teacher_kind,
        "prompt_id": prompt.prompt_id,
        "input_token_ids": [int(token_id) for token_id in input_token_ids],
        "positions": list(range(position_count)),
        "target_token_ids": [int(token_id) for token_id in input_token_ids[1 : position_count + 1]],
        "route_coverage": route_coverage,
        "route_trace": route_trace,
        "route_trace_layers": sorted(route_trace, key=int),
        **quality_prompt_metadata(prompt, prompt_set=prompt_set),
        **metadata,
    }


def _rank_view_roots_json(roots: dict[int, Path]) -> str:
    return json.dumps({str(rank): str(path) for rank, path in sorted(roots.items())})


def _local_sequential_stage_view_roots_json(
    roots: dict[str, dict[int, Path]],
) -> dict[str, dict[str, str]]:
    return {
        stage: {str(rank): str(path) for rank, path in sorted(stage_roots.items())}
        for stage, stage_roots in sorted(roots.items())
    }


def _local_sequential_worker_rank_view_roots(
    args: argparse.Namespace,
    *,
    worker: str,
) -> dict[int, Path]:
    roots = dict(args.rank_view_roots_json)
    stage_view_roots = getattr(args, "local_sequential_stage_view_roots_json", None) or {}
    worker_roots = stage_view_roots.get(worker)
    if worker_roots:
        roots.update(worker_roots)
    return roots


def _parse_int_list(raw: str) -> list[int]:
    values = [int(item) for item in raw.split(",") if item]
    if values != sorted(set(values)):
        raise argparse.ArgumentTypeError("split layers must be sorted and unique")
    return values


def _format_int_list(values: list[int] | None) -> str | None:
    if values is None:
        return None
    return ",".join(str(value) for value in values)


def _parse_string_list(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list must contain at least one non-empty value")
    return values


def _parse_remote_stage_view_roots(raw: str) -> dict[str, dict[int, str]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError("remote stage view roots must be a JSON object")
    parsed: dict[str, dict[int, str]] = {}
    for worker, roots in data.items():
        if not isinstance(worker, str) or not worker:
            raise argparse.ArgumentTypeError("remote stage worker names must be non-empty strings")
        if not isinstance(roots, dict):
            raise argparse.ArgumentTypeError(f"remote stage roots for {worker!r} must be an object")
        worker_roots: dict[int, str] = {}
        for rank, path in roots.items():
            try:
                rank_int = int(rank)
            except (TypeError, ValueError) as error:
                raise argparse.ArgumentTypeError(
                    f"remote stage root rank for {worker!r} must be an integer"
                ) from error
            if not isinstance(path, str) or not path:
                raise argparse.ArgumentTypeError(
                    f"remote stage root path for {worker!r} rank {rank_int} must be a string"
                )
            worker_roots[rank_int] = path
        parsed[worker] = worker_roots
    return parsed


def _window_prefix(kind: str, index: int, *, is_final: bool) -> str:
    if is_final:
        return f"local_sequential_{kind}_hidden_host"
    return f"local_sequential_{kind}_window_{index}_hidden_host"


def _window_metric_stage(kind: str, index: int) -> str:
    return f"local_sequential_{kind}_window_{index}_process"


def _append_optional_float_arg(command: list[str], flag: str, value: float | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _local_sequential_worker_command(
    args: argparse.Namespace,
    *,
    worker: str,
    prompt_id: str,
    hidden_path: Path,
    worker_json_path: Path,
    output_path: Path | None = None,
    lower_json_path: Path | None = None,
    window_rank: int | None = None,
    window_start_layer: int | None = None,
    window_stop_after_layer: int | None = None,
    window_input_prefix: str | None = None,
    window_output_prefix: str | None = None,
    window_include_embed: bool = False,
    window_include_norm: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--rank-view-roots-json",
        _rank_view_roots_json(
            _local_sequential_worker_rank_view_roots(args, worker=worker)
        ),
        "--output-dir",
        str(args.output_dir),
        "--layer-split",
        str(args.layer_split),
        "--rank0-only-logits",
        "--stream-lm-head",
        "--prompt-set",
        str(args.prompt_set),
        "--top-k",
        str(args.top_k),
        "--model-id",
        str(args.model_id),
        "--revision",
        str(args.revision),
        "--teacher-kind",
        str(args.teacher_kind),
        "--lm-head-chunk-rows",
        str(args.lm_head_chunk_rows),
        "--local-sequential-stage-worker",
        worker,
        "--local-sequential-worker-prompt-id",
        prompt_id,
        "--local-sequential-hidden-path",
        str(hidden_path),
        "--local-sequential-worker-json",
        str(worker_json_path),
    ]
    if bool(getattr(args, "local_sequential_head_process", False)):
        command.append("--local-sequential-head-process")
    lower_split_layer = getattr(args, "local_sequential_lower_split_layer", None)
    if lower_split_layer is not None:
        command.extend(["--local-sequential-lower-split-layer", str(lower_split_layer)])
    upper_split_layer = getattr(args, "local_sequential_upper_split_layer", None)
    if upper_split_layer is not None:
        command.extend(["--local-sequential-upper-split-layer", str(upper_split_layer)])
    lower_split_layers = _format_int_list(getattr(args, "local_sequential_lower_split_layers", None))
    if lower_split_layers is not None:
        command.extend(["--local-sequential-lower-split-layers", lower_split_layers])
    upper_split_layers = _format_int_list(getattr(args, "local_sequential_upper_split_layers", None))
    if upper_split_layers is not None:
        command.extend(["--local-sequential-upper-split-layers", upper_split_layers])
    if args.no_full_logits:
        command.append("--no-full-logits")
    if args.max_positions is not None:
        command.extend(["--max-positions", str(args.max_positions)])
    _append_optional_float_arg(command, "--mlx-memory-limit-gb", args.mlx_memory_limit_gb)
    _append_optional_float_arg(command, "--mlx-cache-limit-gb", args.mlx_cache_limit_gb)
    _append_optional_float_arg(command, "--mlx-wired-limit-gb", args.mlx_wired_limit_gb)
    if worker == "lower-embed":
        if output_path is None:
            raise ValueError("lower-embed stage worker requires output path")
        command.extend(
            [
                "--local-sequential-output-path",
                str(output_path),
            ]
        )
        return command
    if worker.startswith(("lower-window-", "upper-window-")):
        if window_rank is None or window_start_layer is None or window_output_prefix is None:
            raise ValueError(f"{worker} window worker requires rank, start layer, and output prefix")
        if output_path is None:
            raise ValueError(f"{worker} window worker requires output path")
        command.extend(
            [
                "--local-sequential-window-rank",
                str(window_rank),
                "--local-sequential-window-start-layer",
                str(window_start_layer),
                "--local-sequential-window-output-prefix",
                window_output_prefix,
                "--local-sequential-output-path",
                str(output_path),
            ]
        )
        if window_stop_after_layer is not None:
            command.extend(
                [
                    "--local-sequential-window-stop-after-layer",
                    str(window_stop_after_layer),
                ]
            )
        if window_include_embed:
            command.append("--local-sequential-window-include-embed")
        if window_include_norm:
            command.append("--local-sequential-window-include-norm")
        if lower_json_path is not None:
            if window_input_prefix is None:
                raise ValueError(f"{worker} window worker requires an input prefix")
            command.extend(
                [
                    "--local-sequential-worker-input-json",
                    str(lower_json_path),
                    "--local-sequential-window-input-prefix",
                    window_input_prefix,
                ]
            )
        return command
    if worker in {"lower-final", "upper", "upper-pre", "upper-final", "head"}:
        if output_path is None or lower_json_path is None:
            raise ValueError(f"{worker} stage worker requires output and input JSON paths")
        command.extend(
            [
                "--local-sequential-output-path",
                str(output_path),
                "--local-sequential-worker-input-json",
                str(lower_json_path),
            ]
        )
    return command


def _command_arg(command: list[str], flag: str) -> str | None:
    try:
        index = command.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    return command[index + 1]


def _replace_command_arg(command: list[str], flag: str, value: str) -> None:
    try:
        index = command.index(flag)
    except ValueError as error:
        raise ValueError(f"command does not contain {flag}") from error
    if index + 1 >= len(command):
        raise ValueError(f"command flag {flag} has no value")
    command[index + 1] = value


def _worker_matches_remote_patterns(worker: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if pattern == worker:
            return True
        if pattern.endswith("*") and worker.startswith(pattern[:-1]):
            return True
        if worker.startswith(pattern):
            return True
    return False


def _local_sequential_remote_worker_patterns(args: argparse.Namespace) -> list[str]:
    patterns: list[str] = []
    for raw in getattr(args, "local_sequential_remote_worker", None) or []:
        patterns.extend(_parse_string_list(str(raw)))
    return patterns


def _local_sequential_worker_is_remote(args: argparse.Namespace, worker: str) -> bool:
    remote_ssh = getattr(args, "local_sequential_remote_ssh", None)
    if not remote_ssh:
        return False
    return _worker_matches_remote_patterns(
        worker,
        _local_sequential_remote_worker_patterns(args),
    )


def _safe_remote_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "worker"


def _remote_stage_roots_for_worker(args: argparse.Namespace, worker: str) -> dict[int, str] | None:
    roots = getattr(args, "local_sequential_remote_stage_view_roots_json", None) or {}
    worker_roots = roots.get(worker)
    if worker_roots:
        base_roots = {rank: str(path) for rank, path in args.rank_view_roots_json.items()}
        base_roots.update(worker_roots)
        return base_roots
    return None


def _local_sequential_remote_ssh_options(args: argparse.Namespace) -> list[str]:
    options: list[str] = []
    for option in getattr(args, "local_sequential_remote_ssh_option", None) or []:
        options.append(str(option))
    return options


def _remote_worker_path(job_dir: str, local_path: str | None) -> str | None:
    if not local_path:
        return None
    return posixpath.join(job_dir, Path(local_path).name)


def _run_checked_subprocess(command: list[str], *, description: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode == 0:
        return result
    raise RuntimeError(
        f"{description} failed with exit {result.returncode}: {' '.join(command)}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def _rsync_remote_shell(ssh_options: list[str]) -> str:
    return " ".join(shlex.quote(item) for item in ["ssh", *ssh_options])


_REMOTE_MEMORY_QUIET_PREFLIGHT_SCRIPT = r"""
import json
import re
import subprocess
import sys
import time

worker = sys.argv[1]
window_seconds = float(sys.argv[2])
max_attempts = max(1, int(sys.argv[3]))
effective_min = float(sys.argv[4])
metadata = json.loads(sys.argv[5])


def vm_counts():
    output = subprocess.check_output(["vm_stat"], text=True)
    page_size = 4096
    header = output.splitlines()[0] if output.splitlines() else ""
    match = re.search(r"page size of (\d+) bytes", header)
    if match:
        page_size = int(match.group(1))
    counts = {"page_size": page_size}
    for line in output.splitlines()[1:]:
        if ":" not in line:
            continue
        label, raw_value = line.split(":", 1)
        value_match = re.search(r"([0-9.]+)", raw_value)
        if not value_match:
            continue
        value = int(value_match.group(1).replace(".", ""))
        normalized = label.strip().lower().strip('"').replace(" ", "_").replace("-", "_")
        if normalized.startswith("pages_"):
            counts[normalized] = value
        elif normalized in {"pageouts", "swapouts"}:
            counts[normalized] = value
    return counts


def page_gb(counts, *keys):
    pages = sum(int(counts.get(key, 0)) for key in keys)
    return pages * int(counts.get("page_size", 4096)) / (1024**3)


last = {
    "available": False,
    "quiet": False,
    "attempts": 0,
    "worker": worker,
    "window_seconds": window_seconds,
    "free_gb": None,
    "inactive_gb": None,
    "purgeable_gb": None,
    "available_gb": None,
    "availability_basis": "free+speculative+inactive",
    "pageouts_delta": None,
    "swapouts_delta": None,
    **metadata,
}
for attempt in range(1, max_attempts + 1):
    before = vm_counts()
    time.sleep(window_seconds)
    after = vm_counts()
    pageouts_delta = int(after.get("pageouts", 0)) - int(before.get("pageouts", 0))
    swapouts_delta = int(after.get("swapouts", 0)) - int(before.get("swapouts", 0))
    free_gb = page_gb(after, "pages_free", "pages_speculative")
    inactive_gb = page_gb(after, "pages_inactive")
    purgeable_gb = page_gb(after, "pages_purgeable")
    available_gb = page_gb(after, "pages_free", "pages_speculative", "pages_inactive")
    last = {
        **last,
        "available": True,
        "quiet": (
            pageouts_delta == 0
            and swapouts_delta == 0
            and available_gb >= effective_min
        ),
        "attempts": attempt,
        "free_gb": free_gb,
        "inactive_gb": inactive_gb,
        "purgeable_gb": purgeable_gb,
        "available_gb": available_gb,
        "pageouts_delta": pageouts_delta,
        "swapouts_delta": swapouts_delta,
    }
    if last["quiet"]:
        print(json.dumps(last, sort_keys=True))
        raise SystemExit(0)
print(json.dumps(last, sort_keys=True), file=sys.stderr)
raise SystemExit(2)
"""


def _run_remote_stage_memory_quiet_preflight(
    args: argparse.Namespace,
    *,
    worker: str,
    remote_ssh: str,
    ssh_options: list[str],
) -> None:
    if not bool(getattr(args, "local_sequential_stage_memory_quiet_preflight", False)):
        return
    window_seconds = float(
        getattr(args, "local_sequential_stage_memory_quiet_seconds", 0.0) or 0.0
    )
    max_attempts = max(
        1,
        int(getattr(args, "local_sequential_stage_memory_quiet_max_attempts", 1) or 1),
    )
    effective_min, metadata = _local_sequential_stage_min_free_gb(args, worker)
    remote_script = "python3 - " + " ".join(
        shlex.quote(item)
        for item in [
            worker,
            str(window_seconds),
            str(max_attempts),
            str(effective_min),
            json.dumps(metadata, sort_keys=True),
        ]
    ) + " <<'PY'\n" + _REMOTE_MEMORY_QUIET_PREFLIGHT_SCRIPT + "\nPY"
    _run_checked_subprocess(
        ["ssh", *ssh_options, remote_ssh, remote_script],
        description=f"remote local-sequential memory quiet preflight {worker}",
    )


def _run_remote_local_sequential_worker(
    args: argparse.Namespace,
    *,
    worker: str,
    command: list[str],
) -> None:
    remote_ssh = str(getattr(args, "local_sequential_remote_ssh"))
    remote_repo = str(getattr(args, "local_sequential_remote_repo", "") or "")
    remote_python = str(getattr(args, "local_sequential_remote_python", "python") or "python")
    remote_tmp_root = str(
        getattr(args, "local_sequential_remote_tmp_dir", "/tmp/glm-local-sequential-workers")
        or "/tmp/glm-local-sequential-workers"
    )
    prompt_id = _command_arg(command, "--local-sequential-worker-prompt-id") or "prompt"
    ssh_options = _local_sequential_remote_ssh_options(args)
    job_dir = posixpath.join(
        remote_tmp_root.rstrip("/"),
        _safe_remote_component(prompt_id),
        _safe_remote_component(worker),
    )
    remote_command = list(command)
    script_remote_path = posixpath.join(job_dir, "export_glm45_air_distributed_teacher_cache.py")
    remote_command[0:2] = [*shlex.split(remote_python), script_remote_path]
    if remote_repo:
        _replace_command_arg(remote_command, "--output-dir", posixpath.join(job_dir, "out"))
    else:
        _replace_command_arg(remote_command, "--output-dir", posixpath.join(job_dir, "out"))

    remote_roots = _remote_stage_roots_for_worker(args, worker)
    if remote_roots is not None:
        _replace_command_arg(
            remote_command,
            "--rank-view-roots-json",
            json.dumps({str(rank): path for rank, path in sorted(remote_roots.items())}),
        )

    path_flags = [
        "--local-sequential-hidden-path",
        "--local-sequential-output-path",
        "--local-sequential-worker-json",
        "--local-sequential-worker-input-json",
    ]
    local_to_remote: dict[str, str] = {}
    for flag in path_flags:
        local_value = _command_arg(command, flag)
        remote_value = _remote_worker_path(job_dir, local_value)
        if local_value and remote_value:
            local_to_remote[local_value] = remote_value
            _replace_command_arg(remote_command, flag, remote_value)

    mkdir_command = (
        f"mkdir -p {shlex.quote(job_dir)} {shlex.quote(posixpath.join(job_dir, 'out'))}"
    )
    _run_checked_subprocess(
        ["ssh", *ssh_options, remote_ssh, mkdir_command],
        description="remote worker mkdir",
    )
    _run_checked_subprocess(
        [
            "scp",
            *ssh_options,
            str(Path(__file__).resolve()),
            f"{remote_ssh}:{script_remote_path}",
        ],
        description="remote worker exporter stage",
    )
    local_src_dir = Path(__file__).resolve().parents[1] / "src"
    if local_src_dir.exists():
        _run_checked_subprocess(
            [
                "rsync",
                "-a",
                "-e",
                _rsync_remote_shell(ssh_options),
                f"{local_src_dir}/",
                f"{remote_ssh}:{posixpath.join(job_dir, 'src')}/",
            ],
            description="remote worker source package stage",
        )
    for flag in ["--local-sequential-hidden-path", "--local-sequential-worker-input-json"]:
        local_value = _command_arg(command, flag)
        if local_value and Path(local_value).exists():
            _run_checked_subprocess(
                [
                    "scp",
                    *ssh_options,
                    local_value,
                    f"{remote_ssh}:{local_to_remote[local_value]}",
                ],
                description=f"remote worker input stage {flag}",
            )

    remote_shell_parts: list[str] = []
    remote_env_prefix = ""
    if remote_repo:
        remote_shell_parts.append(f"cd {shlex.quote(remote_repo)}")
        pythonpath = f"{job_dir}/src:{remote_repo}/src:{remote_repo}"
        remote_env_prefix = f"PYTHONPATH={shlex.quote(pythonpath)} "
    remote_shell_parts.append(
        remote_env_prefix + " ".join(shlex.quote(item) for item in remote_command)
    )
    dirty_retries = max(
        0,
        int(getattr(args, "local_sequential_remote_dirty_retries", 0) or 0),
    )
    retry_sleep_seconds = max(
        0.0,
        float(getattr(args, "local_sequential_remote_retry_sleep_seconds", 0.0) or 0.0),
    )
    worker_json_local = _command_arg(command, "--local-sequential-worker-json")
    output_local = _command_arg(command, "--local-sequential-output-path")
    for attempt in range(1, dirty_retries + 2):
        _run_remote_stage_memory_quiet_preflight(
            args,
            worker=worker,
            remote_ssh=remote_ssh,
            ssh_options=ssh_options,
        )
        _run_checked_subprocess(
            ["ssh", *ssh_options, remote_ssh, " && ".join(remote_shell_parts)],
            description=f"remote local-sequential stage worker {worker}",
        )

        for flag in ["--local-sequential-worker-json", "--local-sequential-output-path"]:
            local_value = _command_arg(command, flag)
            if local_value:
                Path(local_value).parent.mkdir(parents=True, exist_ok=True)
                _run_checked_subprocess(
                    [
                        "scp",
                        *ssh_options,
                        f"{remote_ssh}:{local_to_remote[local_value]}",
                        local_value,
                    ],
                    description=f"remote worker output fetch {flag}",
                )
        if not worker_json_local or not Path(worker_json_local).exists():
            return
        try:
            worker_info = json.loads(Path(worker_json_local).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if _local_sequential_checkpoint_is_clean(worker_info):
            return
        if attempt > dirty_retries:
            return
        for local_value in [worker_json_local, output_local]:
            if local_value:
                try:
                    Path(local_value).unlink()
                except OSError:
                    pass
        remote_cleanup_paths = [
            local_to_remote[local_value]
            for local_value in [worker_json_local, output_local]
            if local_value and local_value in local_to_remote
        ]
        if remote_cleanup_paths:
            _run_checked_subprocess(
                [
                    "ssh",
                    *ssh_options,
                    remote_ssh,
                    "rm -f " + " ".join(shlex.quote(path) for path in remote_cleanup_paths),
                ],
                description=f"remote dirty worker cleanup {worker}",
            )
        if retry_sleep_seconds:
            time.sleep(retry_sleep_seconds)


def _run_local_sequential_worker(command: list[str]) -> None:
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode == 0:
        return
    raise RuntimeError(
        "local-sequential stage worker failed with exit "
        f"{result.returncode}: {' '.join(command)}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def _run_local_sequential_worker_for_args(
    args: argparse.Namespace,
    *,
    worker: str,
    command: list[str],
) -> None:
    if _local_sequential_worker_is_remote(args, worker):
        _run_remote_local_sequential_worker(args, worker=worker, command=command)
        return
    _run_local_sequential_worker(command)


def _worker_metric_row(
    *,
    rank: int,
    stage: str,
    info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "rank": int(rank),
        "stage": stage,
        "elapsed_seconds": float(info.get("elapsed_seconds", 0.0)),
        **dict(info.get("metrics") or {}),
    }


def _max_known_int(*values: Any) -> int | None:
    known = [int(value) for value in values if value is not None]
    return max(known) if known else None


def _merge_process_metrics(
    parent_metrics: dict[str, int | None],
    rank_metrics: list[dict[str, Any]],
) -> dict[str, int | None]:
    merged = dict(parent_metrics)
    for key in ("mlx_active_bytes", "mlx_peak_bytes", "mlx_cache_bytes", "rss_bytes"):
        merged[key] = _max_known_int(
            parent_metrics.get(key),
            *[metric.get(key) for metric in rank_metrics],
        )
    return merged


def _run_local_sequential_lower_stage_worker(
    args: argparse.Namespace,
    *,
    mlx_memory_settings: dict[str, int | None],
) -> None:
    del mlx_memory_settings
    previous_vm = _collect_vm_stat_counts()
    mx.reset_peak_memory()
    start = time.perf_counter()
    prompt = get_quality_prompt_by_id(args.local_sequential_worker_prompt_id)
    lower_model, tokenizer = _load_rank_model_and_tokenizer(
        args.rank_view_roots_json[1],
        rank=1,
        pipeline_size=2,
        layer_split=args.layer_split,
    )
    token_ids = _prompt_tokens(tokenizer, prompt)
    if len(token_ids) < 2:
        raise ValueError(f"quality prompt {prompt.prompt_id!r} tokenized to fewer than two tokens")
    lower_hidden = _forward_local_lower_stage(lower_model, token_ids)
    spill = _spill_local_hidden_to_host(lower_hidden)
    Path(args.local_sequential_hidden_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(Path(args.local_sequential_hidden_path), hidden=spill["array"])
    del lower_hidden, lower_model, tokenizer
    _release_local_stage_memory()
    metrics = _collect_metric_snapshot(previous_vm)
    info = {
        "input_token_ids": [int(token_id) for token_id in token_ids],
        "elapsed_seconds": time.perf_counter() - start,
        "metrics": metrics,
        "local_sequential_lower_hidden_host_dtype": spill["mlx_dtype"],
        "local_sequential_lower_hidden_host_shape": spill["host_shape"],
        "local_sequential_lower_hidden_host_nbytes": spill["host_nbytes"],
    }
    Path(args.local_sequential_worker_json).write_text(
        json.dumps(info, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_npz_array(path: Path, name: str) -> np.ndarray:
    with np.load(path) as payload:
        return np.asarray(payload[name]).copy()


def _run_local_sequential_upper_stage_worker(
    args: argparse.Namespace,
    *,
    mlx_memory_settings: dict[str, int | None],
) -> None:
    del mlx_memory_settings
    previous_vm = _collect_vm_stat_counts()
    mx.reset_peak_memory()
    start = time.perf_counter()
    lower_info = json.loads(Path(args.local_sequential_worker_input_json).read_text(encoding="utf-8"))
    hidden_array = _load_npz_array(Path(args.local_sequential_hidden_path), "hidden")
    lower_hidden = _restore_local_hidden_from_host(
        {
            "array": hidden_array,
            "mlx_dtype": lower_info["local_sequential_lower_hidden_host_dtype"],
            "host_shape": lower_info["local_sequential_lower_hidden_host_shape"],
            "host_nbytes": lower_info["local_sequential_lower_hidden_host_nbytes"],
        }
    )
    token_ids = [int(token_id) for token_id in lower_info["input_token_ids"]]
    upper_model, upper_tokenizer = _load_rank_model_and_tokenizer(
        args.rank_view_roots_json[0],
        rank=0,
        pipeline_size=2,
        layer_split=args.layer_split,
    )
    output = _forward_local_upper_stage(
        upper_model,
        lower_hidden,
        return_hidden=args.stream_lm_head,
    )
    output_path = Path(args.local_sequential_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_hidden_spill: dict[str, Any] | None = None
    if bool(getattr(args, "local_sequential_head_process", False)):
        if not args.stream_lm_head:
            raise ValueError("head-process split requires streamed lm_head output")
        output_hidden_spill = _spill_local_hidden_to_host(output)
        np.savez(output_path, hidden=output_hidden_spill["array"])
        output_kind = "hidden"
    elif args.stream_lm_head and args.no_full_logits:
        topk_payload = _stream_lm_head_topk_payload(
            output,
            rank_dir=args.rank_view_roots_json[0],
            input_token_ids=token_ids,
            max_positions=args.max_positions,
            chunk_rows=args.lm_head_chunk_rows,
            top_k=args.top_k,
        )
        np.savez(
            output_path,
            topk_ids=topk_payload.topk_ids,
            topk_logprobs=topk_payload.topk_logprobs,
            target_logprobs=topk_payload.target_logprobs,
            teacher_top1_ids=topk_payload.teacher_top1_ids,
        )
        output_kind = "topk"
    elif args.stream_lm_head:
        logits = _stream_lm_head_logits(
            output,
            rank_dir=args.rank_view_roots_json[0],
            input_token_ids=token_ids,
            max_positions=args.max_positions,
            chunk_rows=args.lm_head_chunk_rows,
        )
        np.savez(output_path, logits=logits)
        output_kind = "logits"
    else:
        mx.eval(output)
        np.savez(output_path, logits=np.asarray(output.astype(mx.float32), dtype=np.float32))
        output_kind = "logits"
    del output, lower_hidden, upper_model, upper_tokenizer
    _release_local_stage_memory()
    info = {
        "input_token_ids": [int(token_id) for token_id in token_ids],
        "output_kind": output_kind,
        "elapsed_seconds": time.perf_counter() - start,
        "metrics": _collect_metric_snapshot(previous_vm),
    }
    if output_hidden_spill is not None:
        info.update(
            {
                "local_sequential_upper_hidden_host_dtype": output_hidden_spill["mlx_dtype"],
                "local_sequential_upper_hidden_host_shape": output_hidden_spill["host_shape"],
                "local_sequential_upper_hidden_host_nbytes": output_hidden_spill["host_nbytes"],
            }
        )
    Path(args.local_sequential_worker_json).write_text(
        json.dumps(info, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _restore_local_hidden_from_info(
    *,
    hidden_path: Path,
    info: dict[str, Any],
    prefix: str,
) -> mx.array:
    hidden_array = _load_npz_array(hidden_path, "hidden")
    return _restore_local_hidden_from_host(
        {
            "array": hidden_array,
            "mlx_dtype": info[f"{prefix}_dtype"],
            "host_shape": info[f"{prefix}_shape"],
            "host_nbytes": info[f"{prefix}_nbytes"],
        }
    )


def _run_local_sequential_window_stage_worker(
    args: argparse.Namespace,
    *,
    mlx_memory_settings: dict[str, int | None],
) -> None:
    del mlx_memory_settings
    previous_vm = _collect_vm_stat_counts()
    mx.reset_peak_memory()
    start = time.perf_counter()
    rank = int(args.local_sequential_window_rank)
    start_layer = int(args.local_sequential_window_start_layer)
    stop_after_layer = args.local_sequential_window_stop_after_layer
    if stop_after_layer is not None:
        stop_after_layer = int(stop_after_layer)
    output_prefix = str(args.local_sequential_window_output_prefix)
    if bool(args.local_sequential_window_include_embed):
        prompt = get_quality_prompt_by_id(args.local_sequential_worker_prompt_id)
        model, tokenizer = _load_rank_model_and_tokenizer(
            args.rank_view_roots_json[rank],
            rank=rank,
            pipeline_size=2,
            layer_split=args.layer_split,
            rank_start_layer=start_layer,
            rank_stop_after_layer=stop_after_layer,
        )
        token_ids = _prompt_tokens(tokenizer, prompt)
        if len(token_ids) < 2:
            raise ValueError(
                f"quality prompt {prompt.prompt_id!r} tokenized to fewer than two tokens"
            )
        hidden = _forward_local_lower_stage(model, token_ids)
        del model, tokenizer
    else:
        if args.local_sequential_worker_input_json is None:
            raise ValueError("window stage without embeddings requires input JSON")
        input_prefix = args.local_sequential_window_input_prefix
        if not input_prefix:
            raise ValueError("window stage without embeddings requires input prefix")
        previous_info = json.loads(
            Path(args.local_sequential_worker_input_json).read_text(encoding="utf-8")
        )
        hidden_in = _restore_local_hidden_from_info(
            hidden_path=Path(args.local_sequential_hidden_path),
            info=previous_info,
            prefix=str(input_prefix),
        )
        token_ids = [int(token_id) for token_id in previous_info["input_token_ids"]]
        model, tokenizer = _load_rank_model_and_tokenizer(
            args.rank_view_roots_json[rank],
            rank=rank,
            pipeline_size=2,
            layer_split=args.layer_split,
            rank_start_layer=start_layer,
            rank_stop_after_layer=stop_after_layer,
        )
        if bool(args.local_sequential_window_include_norm):
            hidden = _forward_local_upper_stage(model, hidden_in, return_hidden=True)
        else:
            hidden = _forward_local_hidden_stage(model, hidden_in)
        del hidden_in, model, tokenizer
    spill = _spill_local_hidden_to_host(hidden)
    output_path = Path(args.local_sequential_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, hidden=spill["array"])
    del hidden
    _release_local_stage_memory()
    info = {
        "input_token_ids": [int(token_id) for token_id in token_ids],
        "output_kind": "hidden",
        "elapsed_seconds": time.perf_counter() - start,
        "metrics": _collect_metric_snapshot(previous_vm),
        f"{output_prefix}_dtype": spill["mlx_dtype"],
        f"{output_prefix}_shape": spill["host_shape"],
        f"{output_prefix}_nbytes": spill["host_nbytes"],
    }
    Path(args.local_sequential_worker_json).write_text(
        json.dumps(info, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_local_sequential_lower_embed_stage_worker(
    args: argparse.Namespace,
    *,
    mlx_memory_settings: dict[str, int | None],
) -> None:
    del mlx_memory_settings
    previous_vm = _collect_vm_stat_counts()
    mx.reset_peak_memory()
    start = time.perf_counter()
    prompt = get_quality_prompt_by_id(args.local_sequential_worker_prompt_id)
    model, tokenizer = _load_rank_model_and_tokenizer(
        args.rank_view_roots_json[1],
        rank=1,
        pipeline_size=2,
        layer_split=args.layer_split,
    )
    token_ids = _prompt_tokens(tokenizer, prompt)
    if len(token_ids) < 2:
        raise ValueError(
            f"quality prompt {prompt.prompt_id!r} tokenized to fewer than two tokens"
        )
    hidden = _forward_local_embed_stage(model, token_ids)
    del model, tokenizer
    spill = _spill_local_hidden_to_host(hidden)
    output_path = Path(args.local_sequential_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, hidden=spill["array"])
    del hidden
    _release_local_stage_memory()
    info = {
        "input_token_ids": [int(token_id) for token_id in token_ids],
        "output_kind": "hidden",
        "elapsed_seconds": time.perf_counter() - start,
        "metrics": _collect_metric_snapshot(previous_vm),
        "local_sequential_lower_embed_hidden_host_dtype": spill["mlx_dtype"],
        "local_sequential_lower_embed_hidden_host_shape": spill["host_shape"],
        "local_sequential_lower_embed_hidden_host_nbytes": spill["host_nbytes"],
    }
    Path(args.local_sequential_worker_json).write_text(
        json.dumps(info, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_local_sequential_lower_pre_stage_worker(
    args: argparse.Namespace,
    *,
    mlx_memory_settings: dict[str, int | None],
) -> None:
    del mlx_memory_settings
    previous_vm = _collect_vm_stat_counts()
    mx.reset_peak_memory()
    start = time.perf_counter()
    prompt = get_quality_prompt_by_id(args.local_sequential_worker_prompt_id)
    lower_model, tokenizer = _load_rank_model_and_tokenizer(
        args.rank_view_roots_json[1],
        rank=1,
        pipeline_size=2,
        layer_split=args.layer_split,
        rank_stop_after_layer=args.local_sequential_lower_split_layer - 1,
    )
    token_ids = _prompt_tokens(tokenizer, prompt)
    if len(token_ids) < 2:
        raise ValueError(f"quality prompt {prompt.prompt_id!r} tokenized to fewer than two tokens")
    lower_pre_hidden = _forward_local_lower_stage(lower_model, token_ids)
    spill = _spill_local_hidden_to_host(lower_pre_hidden)
    Path(args.local_sequential_hidden_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(Path(args.local_sequential_hidden_path), hidden=spill["array"])
    del lower_pre_hidden, lower_model, tokenizer
    _release_local_stage_memory()
    info = {
        "input_token_ids": [int(token_id) for token_id in token_ids],
        "output_kind": "hidden",
        "elapsed_seconds": time.perf_counter() - start,
        "metrics": _collect_metric_snapshot(previous_vm),
        "local_sequential_lower_pre_hidden_host_dtype": spill["mlx_dtype"],
        "local_sequential_lower_pre_hidden_host_shape": spill["host_shape"],
        "local_sequential_lower_pre_hidden_host_nbytes": spill["host_nbytes"],
    }
    Path(args.local_sequential_worker_json).write_text(
        json.dumps(info, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_local_sequential_lower_final_stage_worker(
    args: argparse.Namespace,
    *,
    mlx_memory_settings: dict[str, int | None],
) -> None:
    del mlx_memory_settings
    previous_vm = _collect_vm_stat_counts()
    mx.reset_peak_memory()
    start = time.perf_counter()
    lower_pre_info = json.loads(
        Path(args.local_sequential_worker_input_json).read_text(encoding="utf-8")
    )
    lower_pre_hidden = _restore_local_hidden_from_info(
        hidden_path=Path(args.local_sequential_hidden_path),
        info=lower_pre_info,
        prefix="local_sequential_lower_pre_hidden_host",
    )
    token_ids = [int(token_id) for token_id in lower_pre_info["input_token_ids"]]
    lower_model, lower_tokenizer = _load_rank_model_and_tokenizer(
        args.rank_view_roots_json[1],
        rank=1,
        pipeline_size=2,
        layer_split=args.layer_split,
        rank_start_layer=args.local_sequential_lower_split_layer,
    )
    lower_hidden = _forward_local_hidden_stage(lower_model, lower_pre_hidden)
    spill = _spill_local_hidden_to_host(lower_hidden)
    output_path = Path(args.local_sequential_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, hidden=spill["array"])
    del lower_hidden, lower_pre_hidden, lower_model, lower_tokenizer
    _release_local_stage_memory()
    info = {
        "input_token_ids": [int(token_id) for token_id in token_ids],
        "output_kind": "hidden",
        "elapsed_seconds": time.perf_counter() - start,
        "metrics": _collect_metric_snapshot(previous_vm),
        "local_sequential_lower_hidden_host_dtype": spill["mlx_dtype"],
        "local_sequential_lower_hidden_host_shape": spill["host_shape"],
        "local_sequential_lower_hidden_host_nbytes": spill["host_nbytes"],
    }
    Path(args.local_sequential_worker_json).write_text(
        json.dumps(info, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_local_sequential_upper_pre_stage_worker(
    args: argparse.Namespace,
    *,
    mlx_memory_settings: dict[str, int | None],
) -> None:
    del mlx_memory_settings
    previous_vm = _collect_vm_stat_counts()
    mx.reset_peak_memory()
    start = time.perf_counter()
    lower_info = json.loads(Path(args.local_sequential_worker_input_json).read_text(encoding="utf-8"))
    lower_hidden = _restore_local_hidden_from_info(
        hidden_path=Path(args.local_sequential_hidden_path),
        info=lower_info,
        prefix="local_sequential_lower_hidden_host",
    )
    token_ids = [int(token_id) for token_id in lower_info["input_token_ids"]]
    upper_model, upper_tokenizer = _load_rank_model_and_tokenizer(
        args.rank_view_roots_json[0],
        rank=0,
        pipeline_size=2,
        layer_split=args.layer_split,
        rank0_start_layer=args.layer_split,
        rank0_stop_after_layer=args.local_sequential_upper_split_layer - 1,
    )
    output_hidden = _forward_local_upper_stage(
        upper_model,
        lower_hidden,
        return_hidden=True,
        normalize_output=False,
    )
    output_hidden_spill = _spill_local_hidden_to_host(output_hidden)
    output_path = Path(args.local_sequential_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, hidden=output_hidden_spill["array"])
    del output_hidden, lower_hidden, upper_model, upper_tokenizer
    _release_local_stage_memory()
    info = {
        "input_token_ids": [int(token_id) for token_id in token_ids],
        "output_kind": "hidden",
        "elapsed_seconds": time.perf_counter() - start,
        "metrics": _collect_metric_snapshot(previous_vm),
        "local_sequential_upper_pre_hidden_host_dtype": output_hidden_spill["mlx_dtype"],
        "local_sequential_upper_pre_hidden_host_shape": output_hidden_spill["host_shape"],
        "local_sequential_upper_pre_hidden_host_nbytes": output_hidden_spill["host_nbytes"],
    }
    Path(args.local_sequential_worker_json).write_text(
        json.dumps(info, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_local_sequential_upper_final_stage_worker(
    args: argparse.Namespace,
    *,
    mlx_memory_settings: dict[str, int | None],
) -> None:
    del mlx_memory_settings
    previous_vm = _collect_vm_stat_counts()
    mx.reset_peak_memory()
    start = time.perf_counter()
    upper_pre_info = json.loads(
        Path(args.local_sequential_worker_input_json).read_text(encoding="utf-8")
    )
    upper_pre_hidden = _restore_local_hidden_from_info(
        hidden_path=Path(args.local_sequential_hidden_path),
        info=upper_pre_info,
        prefix="local_sequential_upper_pre_hidden_host",
    )
    token_ids = [int(token_id) for token_id in upper_pre_info["input_token_ids"]]
    upper_model, upper_tokenizer = _load_rank_model_and_tokenizer(
        args.rank_view_roots_json[0],
        rank=0,
        pipeline_size=2,
        layer_split=args.layer_split,
        rank0_start_layer=args.local_sequential_upper_split_layer,
    )
    output = _forward_local_upper_stage(
        upper_model,
        upper_pre_hidden,
        return_hidden=args.stream_lm_head,
    )
    output_path = Path(args.local_sequential_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_hidden_spill: dict[str, Any] | None = None
    if bool(getattr(args, "local_sequential_head_process", False)):
        if not args.stream_lm_head:
            raise ValueError("head-process split requires streamed lm_head output")
        output_hidden_spill = _spill_local_hidden_to_host(output)
        np.savez(output_path, hidden=output_hidden_spill["array"])
        output_kind = "hidden"
    elif args.stream_lm_head and args.no_full_logits:
        topk_payload = _stream_lm_head_topk_payload(
            output,
            rank_dir=args.rank_view_roots_json[0],
            input_token_ids=token_ids,
            max_positions=args.max_positions,
            chunk_rows=args.lm_head_chunk_rows,
            top_k=args.top_k,
        )
        np.savez(
            output_path,
            topk_ids=topk_payload.topk_ids,
            topk_logprobs=topk_payload.topk_logprobs,
            target_logprobs=topk_payload.target_logprobs,
            teacher_top1_ids=topk_payload.teacher_top1_ids,
        )
        output_kind = "topk"
    elif args.stream_lm_head:
        logits = _stream_lm_head_logits(
            output,
            rank_dir=args.rank_view_roots_json[0],
            input_token_ids=token_ids,
            max_positions=args.max_positions,
            chunk_rows=args.lm_head_chunk_rows,
        )
        np.savez(output_path, logits=logits)
        output_kind = "logits"
    else:
        mx.eval(output)
        np.savez(output_path, logits=np.asarray(output.astype(mx.float32), dtype=np.float32))
        output_kind = "logits"
    del output, upper_pre_hidden, upper_model, upper_tokenizer
    _release_local_stage_memory()
    info = {
        "input_token_ids": [int(token_id) for token_id in token_ids],
        "output_kind": output_kind,
        "elapsed_seconds": time.perf_counter() - start,
        "metrics": _collect_metric_snapshot(previous_vm),
    }
    if output_hidden_spill is not None:
        info.update(
            {
                "local_sequential_upper_hidden_host_dtype": output_hidden_spill["mlx_dtype"],
                "local_sequential_upper_hidden_host_shape": output_hidden_spill["host_shape"],
                "local_sequential_upper_hidden_host_nbytes": output_hidden_spill["host_nbytes"],
            }
        )
    Path(args.local_sequential_worker_json).write_text(
        json.dumps(info, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_local_sequential_head_stage_worker(
    args: argparse.Namespace,
    *,
    mlx_memory_settings: dict[str, int | None],
) -> None:
    del mlx_memory_settings
    previous_vm = _collect_vm_stat_counts()
    mx.reset_peak_memory()
    start = time.perf_counter()
    upper_info = json.loads(Path(args.local_sequential_worker_input_json).read_text(encoding="utf-8"))
    hidden_array = _load_npz_array(Path(args.local_sequential_hidden_path), "hidden")
    hidden = _restore_local_hidden_from_host(
        {
            "array": hidden_array,
            "mlx_dtype": upper_info["local_sequential_upper_hidden_host_dtype"],
            "host_shape": upper_info["local_sequential_upper_hidden_host_shape"],
            "host_nbytes": upper_info["local_sequential_upper_hidden_host_nbytes"],
        }
    )
    token_ids = [int(token_id) for token_id in upper_info["input_token_ids"]]
    output_path = Path(args.local_sequential_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.no_full_logits:
        topk_payload = _stream_lm_head_topk_payload(
            hidden,
            rank_dir=args.rank_view_roots_json[0],
            input_token_ids=token_ids,
            max_positions=args.max_positions,
            chunk_rows=args.lm_head_chunk_rows,
            top_k=args.top_k,
        )
        np.savez(
            output_path,
            topk_ids=topk_payload.topk_ids,
            topk_logprobs=topk_payload.topk_logprobs,
            target_logprobs=topk_payload.target_logprobs,
            teacher_top1_ids=topk_payload.teacher_top1_ids,
        )
        output_kind = "topk"
    else:
        logits = _stream_lm_head_logits(
            hidden,
            rank_dir=args.rank_view_roots_json[0],
            input_token_ids=token_ids,
            max_positions=args.max_positions,
            chunk_rows=args.lm_head_chunk_rows,
        )
        np.savez(output_path, logits=logits)
        output_kind = "logits"
    del hidden
    _release_local_stage_memory()
    info = {
        "output_kind": output_kind,
        "elapsed_seconds": time.perf_counter() - start,
        "metrics": _collect_metric_snapshot(previous_vm),
    }
    Path(args.local_sequential_worker_json).write_text(
        json.dumps(info, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_stage_process_output(output_path: Path, output_kind: str) -> _TopKOnlyPayload | np.ndarray:
    if output_kind == "topk":
        with np.load(output_path) as payload:
            return _TopKOnlyPayload(
                topk_ids=np.asarray(payload["topk_ids"], dtype=np.int32).copy(),
                topk_logprobs=np.asarray(payload["topk_logprobs"], dtype=np.float32).copy(),
                target_logprobs=np.asarray(payload["target_logprobs"], dtype=np.float32).copy(),
                teacher_top1_ids=np.asarray(payload["teacher_top1_ids"], dtype=np.int32).copy(),
            )
    if output_kind == "logits":
        return _load_npz_array(output_path, "logits")
    raise ValueError(f"unsupported local-sequential stage output kind: {output_kind}")


def _multi_window_specs(args: argparse.Namespace, *, kind: str) -> list[dict[str, Any]]:
    if kind == "lower":
        split_layers = list(getattr(args, "local_sequential_lower_split_layers", None) or [])
        starts = [0, *split_layers]
        stops = [layer - 1 for layer in split_layers] + [int(args.layer_split) - 1]
        rank = 1
    elif kind == "upper":
        split_layers = list(getattr(args, "local_sequential_upper_split_layers", None) or [])
        starts = [int(args.layer_split), *split_layers]
        stops = [layer - 1 for layer in split_layers] + [None]
        rank = 0
    else:
        raise ValueError(f"unsupported window kind: {kind}")
    specs: list[dict[str, Any]] = []
    for index, (start_layer, stop_after_layer) in enumerate(zip(starts, stops)):
        is_final = index == len(starts) - 1
        specs.append(
            {
                "kind": kind,
                "index": index,
                "worker": f"{kind}-window-{index}",
                "rank": rank,
                "start_layer": int(start_layer),
                "stop_after_layer": None if stop_after_layer is None else int(stop_after_layer),
                "include_embed": False,
                "include_norm": bool(kind == "upper" and is_final),
                "output_prefix": _window_prefix(kind, index, is_final=is_final),
                "metric_stage": _window_metric_stage(kind, index),
            }
        )
    return specs


def _spill_metadata_from_info(info: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_spill": True,
        f"{prefix}_dtype": info[f"{prefix}_dtype"],
        f"{prefix}_shape": info[f"{prefix}_shape"],
        f"{prefix}_nbytes": info[f"{prefix}_nbytes"],
    }


def _memory_delta(value: object) -> int:
    if value is None:
        return 0
    return int(value)


def _free_gb_from_vm_counts(counts: dict[str, int] | None) -> float | None:
    if counts is None:
        return None
    free_pages = int(counts.get("pages_free", 0)) + int(
        counts.get("pages_speculative", 0)
    )
    return free_pages * int(counts.get("page_size", 16384)) / (1024**3)


def _vm_page_gb(counts: dict[str, int] | None, *keys: str) -> float | None:
    if counts is None:
        return None
    pages = sum(int(counts.get(key, 0)) for key in keys)
    return pages * int(counts.get("page_size", 16384)) / (1024**3)


def _available_gb_from_vm_counts(counts: dict[str, int] | None) -> float | None:
    # macOS can hold a large inactive page pool that is not counted by
    # pages_free+pages_speculative but is still reclaimable without paging out.
    return _vm_page_gb(counts, "pages_free", "pages_speculative", "pages_inactive")


def _raise_if_local_sequential_stage_dirty(worker: str, metric: dict[str, Any]) -> None:
    pageouts_delta = _memory_delta(metric.get("pageouts_delta"))
    swapouts_delta = _memory_delta(metric.get("swapouts_delta"))
    if pageouts_delta > 0 or swapouts_delta > 0:
        raise SystemExit(
            "local-sequential worker dirtied memory: "
            f"worker={worker} "
            f"pageouts_delta={pageouts_delta} "
            f"swapouts_delta={swapouts_delta}"
        )


_LOCAL_SEQUENTIAL_CHECKPOINT_VERSION = 2


def _local_sequential_checkpoint_root(output_dir: Path) -> Path:
    step_dir = output_dir.parent
    steps_dir = step_dir.parent
    checkpoint_dir = steps_dir / ".local-sequential-checkpoints"
    step_namespace = step_dir.name.rsplit("-", 1)[0]
    stable_root = checkpoint_dir / step_namespace
    if stable_root.exists():
        return stable_root
    legacy_roots = [
        path
        for path in checkpoint_dir.glob(f"{step_namespace}-*")
        if path.is_dir()
    ]
    if legacy_roots:
        return max(legacy_roots, key=lambda path: path.stat().st_mtime)
    return stable_root


def _local_sequential_checkpoint_is_clean(info: dict[str, Any]) -> bool:
    metrics = dict(info.get("metrics") or {})
    return (
        _memory_delta(metrics.get("pageouts_delta")) == 0
        and _memory_delta(metrics.get("swapouts_delta")) == 0
    )


def _load_reusable_local_sequential_checkpoint(
    *,
    worker: str,
    worker_json_path: Path,
    output_path: Path,
) -> dict[str, Any] | None:
    if not worker_json_path.exists() or not output_path.exists():
        return None
    try:
        info = json.loads(worker_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if info.get("checkpoint_schema_version") != _LOCAL_SEQUENTIAL_CHECKPOINT_VERSION:
        return None
    if info.get("worker") != worker:
        return None
    if not _local_sequential_checkpoint_is_clean(info):
        try:
            worker_json_path.unlink()
            output_path.unlink()
        except OSError:
            pass
        return None
    return info


def _stage_view_plan_for_worker(
    args: argparse.Namespace,
    worker: str,
) -> tuple[list[dict[str, Any]], bool | None]:
    stage_view_roots = getattr(args, "local_sequential_stage_view_roots_json", None) or {}
    roots = stage_view_roots.get(worker, {})
    worker_found: bool | None = None
    root_values = list(roots.values())
    if stage_view_roots:
        worker_found = bool(roots)
        if not roots:
            root_values = [
                root
                for stage_roots in stage_view_roots.values()
                for root in stage_roots.values()
            ]
    plans: list[dict[str, Any]] = []
    for root in root_values:
        try:
            plans.append(
                json.loads(
                    (Path(root) / "pipeline_stage_view_plan.json").read_text(
                        encoding="utf-8"
                    )
                )
            )
        except (OSError, json.JSONDecodeError):
            continue
    return plans, worker_found


def _local_sequential_stage_min_free_gb(
    args: argparse.Namespace,
    worker: str,
) -> tuple[float, dict[str, Any]]:
    configured_min = float(
        getattr(args, "local_sequential_stage_memory_quiet_min_free_gb", 0.0) or 0.0
    )
    margin = float(
        getattr(
            args,
            "local_sequential_stage_memory_quiet_stage_view_margin_gb",
            0.0,
        )
        or 0.0
    )
    plans, worker_found = _stage_view_plan_for_worker(args, worker)
    max_visible_bytes = max(
        (int(plan.get("visible_shard_file_bytes") or 0) for plan in plans),
        default=0,
    )
    max_required_bytes = max(
        (int(plan.get("required_present_tensor_bytes") or 0) for plan in plans),
        default=0,
    )
    max_visible_gb = max_visible_bytes / (1024**3)
    max_required_gb = max_required_bytes / (1024**3)
    stage_min_free_gb = max_visible_gb + margin if plans else 0.0
    effective_min = max(configured_min, stage_min_free_gb)
    return effective_min, {
        "configured_min_free_gb": configured_min,
        "effective_min_free_gb": effective_min,
        "stage_view_margin_gb": margin,
        "stage_view_max_required_tensor_gb": max_required_gb,
        "stage_view_max_visible_shard_gb": max_visible_gb,
        "stage_view_min_free_gb": stage_min_free_gb,
        "stage_view_plan_count": len(plans),
        "stage_view_worker_found": worker_found,
    }


def _run_local_sequential_stage_memory_quiet_preflight(
    args: argparse.Namespace,
    *,
    worker: str,
) -> None:
    if not bool(getattr(args, "local_sequential_stage_memory_quiet_preflight", False)):
        return
    window_seconds = float(
        getattr(args, "local_sequential_stage_memory_quiet_seconds", 0.0) or 0.0
    )
    max_attempts = max(
        1,
        int(getattr(args, "local_sequential_stage_memory_quiet_max_attempts", 1) or 1),
    )
    effective_min, metadata = _local_sequential_stage_min_free_gb(args, worker)
    last: dict[str, Any] = {
        "available": False,
        "quiet": False,
        "attempts": 0,
        "worker": worker,
        "window_seconds": window_seconds,
        "free_gb": None,
        "inactive_gb": None,
        "purgeable_gb": None,
        "available_gb": None,
        "availability_basis": "free+speculative+inactive",
        "pageouts_delta": None,
        "swapouts_delta": None,
        **metadata,
    }
    for attempt in range(1, max_attempts + 1):
        before = _collect_vm_stat_counts()
        time.sleep(window_seconds)
        after = _collect_vm_stat_counts()
        if before is None or after is None:
            last = {**last, "available": False, "attempts": attempt}
            break
        pageouts_delta = int(after.get("pageouts", 0)) - int(before.get("pageouts", 0))
        swapouts_delta = int(after.get("swapouts", 0)) - int(before.get("swapouts", 0))
        free_gb = _free_gb_from_vm_counts(after)
        inactive_gb = _vm_page_gb(after, "pages_inactive")
        purgeable_gb = _vm_page_gb(after, "pages_purgeable")
        available_gb = _available_gb_from_vm_counts(after)
        last = {
            **last,
            "available": True,
            "quiet": (
                pageouts_delta == 0
                and swapouts_delta == 0
                and available_gb is not None
                and available_gb >= effective_min
            ),
            "attempts": attempt,
            "free_gb": free_gb,
            "inactive_gb": inactive_gb,
            "purgeable_gb": purgeable_gb,
            "available_gb": available_gb,
            "pageouts_delta": pageouts_delta,
            "swapouts_delta": swapouts_delta,
        }
        if last["quiet"]:
            return
    raise SystemExit(
        "local-sequential worker memory quiet preflight failed: "
        + json.dumps(last, sort_keys=True)
    )


def _run_local_sequential_export_multi_window_stage_processes(
    args: argparse.Namespace,
    *,
    mlx_memory_settings: dict[str, int | None],
    output_dir: Path,
    metadata_path: Path,
) -> None:
    if not bool(getattr(args, "local_sequential_head_process", False)):
        raise ValueError("multi-window local-sequential export requires head-process mode")
    rank0_dir = args.rank_view_roots_json[0]
    rank1_dir = args.rank_view_roots_json[1]
    lower_specs = _multi_window_specs(args, kind="lower")
    upper_specs = _multi_window_specs(args, kind="upper")
    stage_view_roots = getattr(args, "local_sequential_stage_view_roots_json", None) or {}
    summaries: list[dict[str, Any]] = []

    checkpoint_root = _local_sequential_checkpoint_root(output_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="local-sequential-window-", dir=output_dir) as tmp_name:
        tmp_dir = Path(tmp_name)
        for prompt_index, prompt in enumerate(
            _selected_prompts(args.prompt_id, prompt_set=args.prompt_set)
        ):
            previous_vm = _collect_vm_stat_counts()
            mx.reset_peak_memory()
            start = time.perf_counter()
            current_hidden_path: Path | None = None
            current_json_path: Path | None = None
            current_prefix: str | None = None
            window_infos: dict[str, dict[str, Any]] = {}
            rank_metrics: list[dict[str, int | float | None]] = []

            if getattr(args, "local_sequential_lower_split_layers", None) is not None:
                worker = "lower-embed"
                output_path = checkpoint_root / prompt.prompt_id / f"{worker}-hidden.npz"
                worker_json_path = checkpoint_root / prompt.prompt_id / f"{worker}.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cached_info = _load_reusable_local_sequential_checkpoint(
                    worker=worker,
                    worker_json_path=worker_json_path,
                    output_path=output_path,
                )
                command = _local_sequential_worker_command(
                    args,
                    worker=worker,
                    prompt_id=prompt.prompt_id,
                    hidden_path=checkpoint_root / prompt.prompt_id / f"{worker}-input.npz",
                    worker_json_path=worker_json_path,
                    output_path=output_path,
                )
                if cached_info is None:
                    _release_local_stage_memory()
                    _run_local_sequential_stage_memory_quiet_preflight(args, worker=worker)
                    _run_local_sequential_worker_for_args(
                        args,
                        worker=worker,
                        command=command,
                    )
                    info = json.loads(worker_json_path.read_text(encoding="utf-8"))
                    info["checkpoint_schema_version"] = _LOCAL_SEQUENTIAL_CHECKPOINT_VERSION
                    info["worker"] = worker
                    worker_json_path.write_text(
                        json.dumps(info, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                else:
                    info = cached_info
                window_infos[worker] = info
                rank_metrics.append(
                    _worker_metric_row(
                        rank=1,
                        stage="local_sequential_lower_embed_process",
                        info=info,
                    )
                )
                if bool(getattr(args, "local_sequential_abort_on_dirty_stage", False)):
                    _raise_if_local_sequential_stage_dirty(worker, rank_metrics[-1])
                current_hidden_path = output_path
                current_json_path = worker_json_path
                current_prefix = "local_sequential_lower_embed_hidden_host"

            for spec in lower_specs + upper_specs:
                worker = str(spec["worker"])
                hidden_path = current_hidden_path or (
                    checkpoint_root / prompt.prompt_id / f"{worker}-input.npz"
                )
                output_path = checkpoint_root / prompt.prompt_id / f"{worker}-hidden.npz"
                worker_json_path = checkpoint_root / prompt.prompt_id / f"{worker}.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cached_info = _load_reusable_local_sequential_checkpoint(
                    worker=worker,
                    worker_json_path=worker_json_path,
                    output_path=output_path,
                )
                command = _local_sequential_worker_command(
                    args,
                    worker=worker,
                    prompt_id=prompt.prompt_id,
                    hidden_path=hidden_path,
                    worker_json_path=worker_json_path,
                    output_path=output_path,
                    lower_json_path=current_json_path,
                    window_rank=int(spec["rank"]),
                    window_start_layer=int(spec["start_layer"]),
                    window_stop_after_layer=spec["stop_after_layer"],
                    window_input_prefix=current_prefix,
                    window_output_prefix=str(spec["output_prefix"]),
                    window_include_embed=bool(spec["include_embed"]),
                    window_include_norm=bool(spec["include_norm"]),
                )
                if cached_info is None:
                    _release_local_stage_memory()
                    _run_local_sequential_stage_memory_quiet_preflight(args, worker=worker)
                    _run_local_sequential_worker_for_args(
                        args,
                        worker=worker,
                        command=command,
                    )
                    info = json.loads(worker_json_path.read_text(encoding="utf-8"))
                    info["checkpoint_schema_version"] = _LOCAL_SEQUENTIAL_CHECKPOINT_VERSION
                    info["worker"] = worker
                    worker_json_path.write_text(
                        json.dumps(info, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                else:
                    info = cached_info
                window_infos[worker] = info
                rank_metrics.append(
                    _worker_metric_row(
                        rank=int(spec["rank"]),
                        stage=str(spec["metric_stage"]),
                        info=info,
                    )
                )
                if bool(getattr(args, "local_sequential_abort_on_dirty_stage", False)):
                    _raise_if_local_sequential_stage_dirty(worker, rank_metrics[-1])
                current_hidden_path = output_path
                current_json_path = worker_json_path
                current_prefix = str(spec["output_prefix"])

            if current_hidden_path is None or current_json_path is None:
                raise RuntimeError("multi-window local-sequential export produced no hidden state")
            head_json_path = checkpoint_root / prompt.prompt_id / "head.json"
            output_path = checkpoint_root / prompt.prompt_id / "head-output.npz"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            head_command = _local_sequential_worker_command(
                args,
                worker="head",
                prompt_id=prompt.prompt_id,
                hidden_path=current_hidden_path,
                worker_json_path=head_json_path,
                output_path=output_path,
                lower_json_path=current_json_path,
            )
            output_info = _load_reusable_local_sequential_checkpoint(
                worker="head",
                worker_json_path=head_json_path,
                output_path=output_path,
            )
            if output_info is None:
                _release_local_stage_memory()
                _run_local_sequential_stage_memory_quiet_preflight(args, worker="head")
                _run_local_sequential_worker_for_args(
                    args,
                    worker="head",
                    command=head_command,
                )
                output_info = json.loads(head_json_path.read_text(encoding="utf-8"))
                output_info["checkpoint_schema_version"] = _LOCAL_SEQUENTIAL_CHECKPOINT_VERSION
                output_info["worker"] = "head"
                head_json_path.write_text(
                    json.dumps(output_info, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            rank_metrics.append(
                _worker_metric_row(
                    rank=0,
                    stage="local_sequential_head_process",
                    info=output_info,
                )
            )

            lower_info = window_infos[str(lower_specs[-1]["worker"])]
            upper_info = window_infos[str(upper_specs[-1]["worker"])]
            token_ids = [int(token_id) for token_id in lower_info["input_token_ids"]]
            output = _load_stage_process_output(output_path, str(output_info["output_kind"]))
            elapsed_seconds = time.perf_counter() - start
            metric_snapshot = _merge_process_metrics(
                _collect_metric_snapshot(previous_vm),
                rank_metrics,
            )
            metrics: dict[str, int | float | None] = {
                "elapsed_seconds": elapsed_seconds,
                **metric_snapshot,
            }
            shard_relative = f"teacher_logits/{prompt.prompt_id}.safetensors"
            metadata_update = {
                **metrics,
                "distributed_backend": "local_sequential",
                "distributed_size": 1,
                "pipeline_size": 2,
                "layer_split": args.layer_split,
                "rank0_host": socket.gethostname(),
                "rank0_view": str(rank0_dir),
                "rank1_view": str(rank1_dir),
                "rank_view_roots": {"0": str(rank0_dir), "1": str(rank1_dir)},
                "metric_aggregation": "single_host_local_sequential_multi_window_stage_processes",
                "rank_metrics": rank_metrics,
                "rank0_only_logits": bool(args.rank0_only_logits),
                "stream_lm_head": bool(args.stream_lm_head),
                "lm_head_chunk_rows": args.lm_head_chunk_rows if args.stream_lm_head else None,
                "local_sequential_stage_processes": True,
                "local_sequential_head_process": True,
                "local_sequential_lower_split_layer": None,
                "local_sequential_upper_split_layer": None,
                "local_sequential_lower_split_layers": list(
                    getattr(args, "local_sequential_lower_split_layers", None) or []
                ),
                "local_sequential_upper_split_layers": list(
                    getattr(args, "local_sequential_upper_split_layers", None) or []
                ),
                "local_sequential_lower_window_count": len(lower_specs),
                "local_sequential_upper_window_count": len(upper_specs),
                "route_trace_only": False,
                "route_coverage": {},
                **_spill_metadata_from_info(lower_info, "local_sequential_lower_hidden_host"),
                **_spill_metadata_from_info(upper_info, "local_sequential_upper_hidden_host"),
                **quality_prompt_metadata(prompt, prompt_set=args.prompt_set),
                **mlx_memory_settings,
            }
            for spec in lower_specs[:-1] + upper_specs[:-1]:
                prefix = str(spec["output_prefix"])
                metadata_update.update(
                    _spill_metadata_from_info(window_infos[str(spec["worker"])], prefix)
                )
            if stage_view_roots:
                metadata_update["local_sequential_stage_view_roots"] = (
                    _local_sequential_stage_view_roots_json(stage_view_roots)
                )

            from keep.quality.teacher_cache import (
                build_teacher_cache_payload,
                build_teacher_cache_topk_payload,
            )

            if isinstance(output, _TopKOnlyPayload):
                row, tensors = build_teacher_cache_topk_payload(
                    topk_ids=output.topk_ids,
                    topk_logprobs=output.topk_logprobs,
                    target_logprobs=output.target_logprobs,
                    teacher_top1_ids=output.teacher_top1_ids,
                    input_token_ids=token_ids,
                    prompt_id=prompt.prompt_id,
                    model_id=args.model_id,
                    revision=args.revision,
                    teacher_kind=args.teacher_kind,
                    shard_path=shard_relative,
                    max_positions=args.max_positions,
                )
            else:
                row, tensors = build_teacher_cache_payload(
                    logits=output,
                    input_token_ids=token_ids,
                    prompt_id=prompt.prompt_id,
                    model_id=args.model_id,
                    revision=args.revision,
                    teacher_kind=args.teacher_kind,
                    shard_path=shard_relative,
                    top_k=args.top_k,
                    max_positions=args.max_positions,
                    save_full_logits=not args.no_full_logits,
                )
            row.update(metadata_update)
            mx.save_safetensors(str(output_dir / shard_relative), tensors)
            with metadata_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            summaries.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "prompt_index": prompt_index,
                    "input_token_count": len(token_ids),
                    "selected_position_count": len(row["positions"]),
                    "elapsed_seconds": metrics["elapsed_seconds"],
                    "mlx_peak_bytes": metrics["mlx_peak_bytes"],
                    "pageouts_delta": metrics["pageouts_delta"],
                    "swapouts_delta": metrics["swapouts_delta"],
                    "record_type": row.get("record_type", "teacher_cache"),
                    "full_logits_available": row["full_logits_available"],
                    "shard": shard_relative,
                    "mlx_cache_limit_bytes": mlx_memory_settings["mlx_cache_limit_bytes"],
                    "lm_head_chunk_rows": args.lm_head_chunk_rows if args.stream_lm_head else None,
                    "local_sequential_stage_processes": True,
                    "local_sequential_head_process": True,
                    "local_sequential_lower_window_count": len(lower_specs),
                    "local_sequential_upper_window_count": len(upper_specs),
                }
            )

    summary = {
        "backend": "local_sequential",
        "distributed_size": 1,
        "pipeline_size": 2,
        "layer_split": args.layer_split,
        "metadata_jsonl": str(metadata_path),
        "output_dir": str(output_dir),
        "prompt_set": args.prompt_set,
        "record_count": len(summaries),
        "records": summaries,
        "top_k": args.top_k,
        "max_positions": args.max_positions,
        "full_logits_available": not args.no_full_logits,
        "route_trace_only": False,
        "route_trace_layers": [],
        "rank_view_roots": {"0": str(rank0_dir), "1": str(rank1_dir)},
        "lm_head_chunk_rows": args.lm_head_chunk_rows if args.stream_lm_head else None,
        "local_sequential_stage_processes": True,
        "local_sequential_head_process": True,
        "local_sequential_lower_split_layers": list(
            getattr(args, "local_sequential_lower_split_layers", None) or []
        ),
        "local_sequential_upper_split_layers": list(
            getattr(args, "local_sequential_upper_split_layers", None) or []
        ),
        "local_sequential_lower_window_count": len(lower_specs),
        "local_sequential_upper_window_count": len(upper_specs),
        "mlx_memory_settings": mlx_memory_settings,
    }
    if stage_view_roots:
        summary["local_sequential_stage_view_roots"] = (
            _local_sequential_stage_view_roots_json(stage_view_roots)
        )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _run_local_sequential_export_stage_processes(
    args: argparse.Namespace,
    *,
    mlx_memory_settings: dict[str, int | None],
    output_dir: Path,
    metadata_path: Path,
) -> None:
    if (
        getattr(args, "local_sequential_lower_split_layers", None) is not None
        or getattr(args, "local_sequential_upper_split_layers", None) is not None
    ):
        _run_local_sequential_export_multi_window_stage_processes(
            args,
            mlx_memory_settings=mlx_memory_settings,
            output_dir=output_dir,
            metadata_path=metadata_path,
        )
        return
    rank0_dir = args.rank_view_roots_json[0]
    rank1_dir = args.rank_view_roots_json[1]
    selected_prompts: list[QualityPrompt] | None = None
    summaries: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="local-sequential-stage-", dir=output_dir) as tmp_name:
        tmp_dir = Path(tmp_name)
        head_process_enabled = bool(getattr(args, "local_sequential_head_process", False))
        lower_split_layer = getattr(args, "local_sequential_lower_split_layer", None)
        upper_split_layer = getattr(args, "local_sequential_upper_split_layer", None)
        stage_view_roots = getattr(args, "local_sequential_stage_view_roots_json", None) or {}
        for prompt_index, prompt in enumerate(
            selected_prompts or _selected_prompts(args.prompt_id, prompt_set=args.prompt_set)
        ):
            if selected_prompts is None:
                selected_prompts = _selected_prompts(args.prompt_id, prompt_set=args.prompt_set)
            previous_vm = _collect_vm_stat_counts()
            mx.reset_peak_memory()
            start = time.perf_counter()

            hidden_path = tmp_dir / f"{prompt.prompt_id}-hidden.npz"
            lower_json_path = tmp_dir / f"{prompt.prompt_id}-lower.json"
            output_path = tmp_dir / f"{prompt.prompt_id}-output.npz"
            upper_json_path = tmp_dir / f"{prompt.prompt_id}-upper.json"
            head_json_path = tmp_dir / f"{prompt.prompt_id}-head.json"
            upper_hidden_path = tmp_dir / f"{prompt.prompt_id}-upper-hidden.npz"
            upper_pre_json_path = tmp_dir / f"{prompt.prompt_id}-upper-pre.json"
            upper_final_json_path = tmp_dir / f"{prompt.prompt_id}-upper-final.json"
            upper_pre_hidden_path = tmp_dir / f"{prompt.prompt_id}-upper-pre-hidden.npz"
            lower_pre_json_path = tmp_dir / f"{prompt.prompt_id}-lower-pre.json"
            lower_pre_hidden_path = tmp_dir / f"{prompt.prompt_id}-lower-pre-hidden.npz"
            if lower_split_layer is None:
                lower_command = _local_sequential_worker_command(
                    args,
                    worker="lower",
                    prompt_id=prompt.prompt_id,
                    hidden_path=hidden_path,
                    worker_json_path=lower_json_path,
                )
                _run_local_sequential_worker_for_args(
                    args,
                    worker="lower",
                    command=lower_command,
                )
            else:
                lower_pre_command = _local_sequential_worker_command(
                    args,
                    worker="lower-pre",
                    prompt_id=prompt.prompt_id,
                    hidden_path=lower_pre_hidden_path,
                    worker_json_path=lower_pre_json_path,
                )
                _run_local_sequential_worker_for_args(
                    args,
                    worker="lower-pre",
                    command=lower_pre_command,
                )
                lower_final_command = _local_sequential_worker_command(
                    args,
                    worker="lower-final",
                    prompt_id=prompt.prompt_id,
                    hidden_path=lower_pre_hidden_path,
                    worker_json_path=lower_json_path,
                    output_path=hidden_path,
                    lower_json_path=lower_pre_json_path,
                )
                _run_local_sequential_worker_for_args(
                    args,
                    worker="lower-final",
                    command=lower_final_command,
                )
            if upper_split_layer is None:
                upper_command = _local_sequential_worker_command(
                    args,
                    worker="upper",
                    prompt_id=prompt.prompt_id,
                    hidden_path=hidden_path,
                    worker_json_path=upper_json_path,
                    output_path=upper_hidden_path if head_process_enabled else output_path,
                    lower_json_path=lower_json_path,
                )
                _run_local_sequential_worker_for_args(
                    args,
                    worker="upper",
                    command=upper_command,
                )
                upper_info_path = upper_json_path
            else:
                upper_pre_command = _local_sequential_worker_command(
                    args,
                    worker="upper-pre",
                    prompt_id=prompt.prompt_id,
                    hidden_path=hidden_path,
                    worker_json_path=upper_pre_json_path,
                    output_path=upper_pre_hidden_path,
                    lower_json_path=lower_json_path,
                )
                _run_local_sequential_worker_for_args(
                    args,
                    worker="upper-pre",
                    command=upper_pre_command,
                )
                upper_final_command = _local_sequential_worker_command(
                    args,
                    worker="upper-final",
                    prompt_id=prompt.prompt_id,
                    hidden_path=upper_pre_hidden_path,
                    worker_json_path=upper_final_json_path,
                    output_path=upper_hidden_path if head_process_enabled else output_path,
                    lower_json_path=upper_pre_json_path,
                )
                _run_local_sequential_worker_for_args(
                    args,
                    worker="upper-final",
                    command=upper_final_command,
                )
                upper_info_path = upper_final_json_path
            if head_process_enabled:
                head_command = _local_sequential_worker_command(
                    args,
                    worker="head",
                    prompt_id=prompt.prompt_id,
                    hidden_path=upper_hidden_path,
                    worker_json_path=head_json_path,
                    output_path=output_path,
                    lower_json_path=upper_info_path,
                )
                _run_local_sequential_worker_for_args(
                    args,
                    worker="head",
                    command=head_command,
                )

            lower_info = json.loads(lower_json_path.read_text(encoding="utf-8"))
            lower_pre_info = (
                json.loads(lower_pre_json_path.read_text(encoding="utf-8"))
                if lower_split_layer is not None
                else None
            )
            upper_pre_info = (
                json.loads(upper_pre_json_path.read_text(encoding="utf-8"))
                if upper_split_layer is not None
                else None
            )
            upper_info = json.loads(upper_info_path.read_text(encoding="utf-8"))
            output_info = (
                json.loads(head_json_path.read_text(encoding="utf-8"))
                if head_process_enabled
                else upper_info
            )
            token_ids = [int(token_id) for token_id in lower_info["input_token_ids"]]
            output = _load_stage_process_output(output_path, str(output_info["output_kind"]))
            elapsed_seconds = time.perf_counter() - start
            rank_metrics = []
            if lower_split_layer is None:
                rank_metrics.append(
                    _worker_metric_row(
                        rank=1,
                        stage="local_sequential_lower_process",
                        info=lower_info,
                    )
                )
            else:
                rank_metrics.extend(
                    [
                        _worker_metric_row(
                            rank=1,
                            stage="local_sequential_lower_pre_process",
                            info=lower_pre_info,
                        ),
                        _worker_metric_row(
                            rank=1,
                            stage="local_sequential_lower_final_process",
                            info=lower_info,
                        ),
                    ]
                )
            if upper_split_layer is None:
                rank_metrics.append(
                    _worker_metric_row(
                        rank=0,
                        stage="local_sequential_upper_process",
                        info=upper_info,
                    )
                )
            else:
                rank_metrics.extend(
                    [
                        _worker_metric_row(
                            rank=0,
                            stage="local_sequential_upper_pre_process",
                            info=upper_pre_info,
                        ),
                        _worker_metric_row(
                            rank=0,
                            stage="local_sequential_upper_final_process",
                            info=upper_info,
                        ),
                    ]
                )
            if head_process_enabled:
                rank_metrics.append(
                    _worker_metric_row(
                        rank=0,
                        stage="local_sequential_head_process",
                        info=output_info,
                    )
                )
            metric_snapshot = _merge_process_metrics(
                _collect_metric_snapshot(previous_vm),
                rank_metrics,
            )
            metrics: dict[str, int | float | None] = {
                "elapsed_seconds": elapsed_seconds,
                **metric_snapshot,
            }
            shard_relative = f"teacher_logits/{prompt.prompt_id}.safetensors"
            metadata_update = {
                **metrics,
                "distributed_backend": "local_sequential",
                "distributed_size": 1,
                "pipeline_size": 2,
                "layer_split": args.layer_split,
                "rank0_host": socket.gethostname(),
                "rank0_view": str(rank0_dir),
                "rank1_view": str(rank1_dir),
                "rank_view_roots": {"0": str(rank0_dir), "1": str(rank1_dir)},
                "metric_aggregation": "single_host_local_sequential_stage_processes",
                "rank_metrics": rank_metrics,
                "rank0_only_logits": bool(args.rank0_only_logits),
                "stream_lm_head": bool(args.stream_lm_head),
                "lm_head_chunk_rows": args.lm_head_chunk_rows if args.stream_lm_head else None,
                "local_sequential_stage_processes": True,
                "local_sequential_head_process": head_process_enabled,
                "local_sequential_lower_split_layer": lower_split_layer,
                "local_sequential_upper_split_layer": upper_split_layer,
                "local_sequential_lower_hidden_host_spill": True,
                "local_sequential_lower_hidden_host_dtype": lower_info[
                    "local_sequential_lower_hidden_host_dtype"
                ],
                "local_sequential_lower_hidden_host_shape": lower_info[
                    "local_sequential_lower_hidden_host_shape"
                ],
                "local_sequential_lower_hidden_host_nbytes": lower_info[
                    "local_sequential_lower_hidden_host_nbytes"
                ],
                "local_sequential_upper_hidden_host_spill": head_process_enabled,
                "route_trace_only": False,
                "route_coverage": {},
                **quality_prompt_metadata(prompt, prompt_set=args.prompt_set),
                **mlx_memory_settings,
            }
            if stage_view_roots:
                metadata_update["local_sequential_stage_view_roots"] = (
                    _local_sequential_stage_view_roots_json(stage_view_roots)
                )
            if lower_split_layer is not None:
                metadata_update.update(
                    {
                        "local_sequential_lower_pre_hidden_host_spill": True,
                        "local_sequential_lower_pre_hidden_host_dtype": lower_pre_info[
                            "local_sequential_lower_pre_hidden_host_dtype"
                        ],
                        "local_sequential_lower_pre_hidden_host_shape": lower_pre_info[
                            "local_sequential_lower_pre_hidden_host_shape"
                        ],
                        "local_sequential_lower_pre_hidden_host_nbytes": lower_pre_info[
                            "local_sequential_lower_pre_hidden_host_nbytes"
                        ],
                    }
                )
            if upper_split_layer is not None:
                metadata_update.update(
                    {
                        "local_sequential_upper_pre_hidden_host_spill": True,
                        "local_sequential_upper_pre_hidden_host_dtype": upper_pre_info[
                            "local_sequential_upper_pre_hidden_host_dtype"
                        ],
                        "local_sequential_upper_pre_hidden_host_shape": upper_pre_info[
                            "local_sequential_upper_pre_hidden_host_shape"
                        ],
                        "local_sequential_upper_pre_hidden_host_nbytes": upper_pre_info[
                            "local_sequential_upper_pre_hidden_host_nbytes"
                        ],
                    }
                )
            if head_process_enabled:
                metadata_update.update(
                    {
                        "local_sequential_upper_hidden_host_dtype": upper_info[
                            "local_sequential_upper_hidden_host_dtype"
                        ],
                        "local_sequential_upper_hidden_host_shape": upper_info[
                            "local_sequential_upper_hidden_host_shape"
                        ],
                        "local_sequential_upper_hidden_host_nbytes": upper_info[
                            "local_sequential_upper_hidden_host_nbytes"
                        ],
                    }
                )

            from keep.quality.teacher_cache import (
                build_teacher_cache_payload,
                build_teacher_cache_topk_payload,
            )

            if isinstance(output, _TopKOnlyPayload):
                row, tensors = build_teacher_cache_topk_payload(
                    topk_ids=output.topk_ids,
                    topk_logprobs=output.topk_logprobs,
                    target_logprobs=output.target_logprobs,
                    teacher_top1_ids=output.teacher_top1_ids,
                    input_token_ids=token_ids,
                    prompt_id=prompt.prompt_id,
                    model_id=args.model_id,
                    revision=args.revision,
                    teacher_kind=args.teacher_kind,
                    shard_path=shard_relative,
                    max_positions=args.max_positions,
                )
            else:
                row, tensors = build_teacher_cache_payload(
                    logits=output,
                    input_token_ids=token_ids,
                    prompt_id=prompt.prompt_id,
                    model_id=args.model_id,
                    revision=args.revision,
                    teacher_kind=args.teacher_kind,
                    shard_path=shard_relative,
                    top_k=args.top_k,
                    max_positions=args.max_positions,
                    save_full_logits=not args.no_full_logits,
                )
            row.update(metadata_update)
            mx.save_safetensors(str(output_dir / shard_relative), tensors)
            with metadata_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            summaries.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "prompt_index": prompt_index,
                    "input_token_count": len(token_ids),
                    "selected_position_count": len(row["positions"]),
                    "elapsed_seconds": metrics["elapsed_seconds"],
                    "mlx_peak_bytes": metrics["mlx_peak_bytes"],
                    "pageouts_delta": metrics["pageouts_delta"],
                    "swapouts_delta": metrics["swapouts_delta"],
                    "record_type": row.get("record_type", "teacher_cache"),
                    "full_logits_available": row["full_logits_available"],
                    "shard": shard_relative,
                    "mlx_cache_limit_bytes": mlx_memory_settings["mlx_cache_limit_bytes"],
                    "lm_head_chunk_rows": args.lm_head_chunk_rows if args.stream_lm_head else None,
                    "local_sequential_stage_processes": True,
                    "local_sequential_head_process": head_process_enabled,
                }
            )

    summary = {
        "backend": "local_sequential",
        "distributed_size": 1,
        "pipeline_size": 2,
        "layer_split": args.layer_split,
        "metadata_jsonl": str(metadata_path),
        "output_dir": str(output_dir),
        "prompt_set": args.prompt_set,
        "record_count": len(summaries),
        "records": summaries,
        "top_k": args.top_k,
        "max_positions": args.max_positions,
        "full_logits_available": not args.no_full_logits,
        "route_trace_only": False,
        "route_trace_layers": [],
        "rank_view_roots": {"0": str(rank0_dir), "1": str(rank1_dir)},
        "lm_head_chunk_rows": args.lm_head_chunk_rows if args.stream_lm_head else None,
        "local_sequential_stage_processes": True,
        "local_sequential_head_process": head_process_enabled,
        "mlx_memory_settings": mlx_memory_settings,
    }
    if stage_view_roots:
        summary["local_sequential_stage_view_roots"] = (
            _local_sequential_stage_view_roots_json(stage_view_roots)
        )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _run_local_sequential_export(
    args: argparse.Namespace,
    *,
    mlx_memory_settings: dict[str, int | None],
) -> None:
    rank0_dir = args.rank_view_roots_json[0]
    rank1_dir = args.rank_view_roots_json[1]
    output_dir = Path(args.output_dir)
    shard_dir = output_dir / "teacher_logits"
    shard_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    if metadata_path.exists():
        metadata_path.unlink()
    if bool(getattr(args, "local_sequential_stage_processes", False)):
        _run_local_sequential_export_stage_processes(
            args,
            mlx_memory_settings=mlx_memory_settings,
            output_dir=output_dir,
            metadata_path=metadata_path,
        )
        return

    selected_prompts: list[QualityPrompt] | None = None
    summaries: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(
        selected_prompts or _selected_prompts(args.prompt_id, prompt_set=args.prompt_set)
    ):
        previous_vm = _collect_vm_stat_counts()
        mx.reset_peak_memory()
        start = time.perf_counter()

        lower_model, tokenizer = _load_rank_model_and_tokenizer(
            rank1_dir,
            rank=1,
            pipeline_size=2,
            layer_split=args.layer_split,
        )
        if selected_prompts is None:
            selected_prompts = _selected_prompts(args.prompt_id, prompt_set=args.prompt_set)
        token_ids = _prompt_tokens(tokenizer, prompt)
        if len(token_ids) < 2:
            raise ValueError(f"quality prompt {prompt.prompt_id!r} tokenized to fewer than two tokens")
        lower_hidden = _forward_local_lower_stage(lower_model, token_ids)
        lower_hidden_spill = _spill_local_hidden_to_host(lower_hidden)
        del lower_hidden, lower_model, tokenizer
        _release_local_stage_memory()

        upper_model, _upper_tokenizer = _load_rank_model_and_tokenizer(
            rank0_dir,
            rank=0,
            pipeline_size=2,
            layer_split=args.layer_split,
        )
        lower_hidden = _restore_local_hidden_from_host(lower_hidden_spill)
        output = _forward_local_upper_stage(
            upper_model,
            lower_hidden,
            return_hidden=args.stream_lm_head,
        )
        if args.stream_lm_head:
            if args.no_full_logits:
                output = _stream_lm_head_topk_payload(
                    output,
                    rank_dir=rank0_dir,
                    input_token_ids=token_ids,
                    max_positions=args.max_positions,
                    chunk_rows=args.lm_head_chunk_rows,
                    top_k=args.top_k,
                )
            else:
                output = _stream_lm_head_logits(
                    output,
                    rank_dir=rank0_dir,
                    input_token_ids=token_ids,
                    max_positions=args.max_positions,
                    chunk_rows=args.lm_head_chunk_rows,
                )
        elif isinstance(output, mx.array):
            mx.eval(output)
        del upper_model, _upper_tokenizer, lower_hidden
        _release_local_stage_memory()

        elapsed_seconds = time.perf_counter() - start
        metric_snapshot = _collect_metric_snapshot(previous_vm)
        metrics: dict[str, int | float | None] = {
            "elapsed_seconds": elapsed_seconds,
            **metric_snapshot,
        }
        rank_metrics = [
            {
                "rank": 0,
                "stage": "local_sequential_two_stage",
                **metrics,
            }
        ]
        shard_relative = f"teacher_logits/{prompt.prompt_id}.safetensors"
        metadata_update = {
            **metrics,
            "distributed_backend": "local_sequential",
            "distributed_size": 1,
            "pipeline_size": 2,
            "layer_split": args.layer_split,
            "rank0_host": socket.gethostname(),
            "rank0_view": str(rank0_dir),
            "rank1_view": str(rank1_dir),
            "rank_view_roots": {"0": str(rank0_dir), "1": str(rank1_dir)},
            "metric_aggregation": "single_host_local_sequential",
            "rank_metrics": rank_metrics,
            "rank0_only_logits": bool(args.rank0_only_logits),
            "stream_lm_head": bool(args.stream_lm_head),
            "lm_head_chunk_rows": args.lm_head_chunk_rows if args.stream_lm_head else None,
            "local_sequential_lower_hidden_host_spill": True,
            "local_sequential_lower_hidden_host_dtype": lower_hidden_spill["mlx_dtype"],
            "local_sequential_lower_hidden_host_shape": lower_hidden_spill["host_shape"],
            "local_sequential_lower_hidden_host_nbytes": lower_hidden_spill["host_nbytes"],
            "route_trace_only": False,
            "route_coverage": {},
            **quality_prompt_metadata(prompt, prompt_set=args.prompt_set),
            **mlx_memory_settings,
        }

        from keep.quality.teacher_cache import (
            build_teacher_cache_payload,
            build_teacher_cache_topk_payload,
        )

        if isinstance(output, _TopKOnlyPayload):
            row, tensors = build_teacher_cache_topk_payload(
                topk_ids=output.topk_ids,
                topk_logprobs=output.topk_logprobs,
                target_logprobs=output.target_logprobs,
                teacher_top1_ids=output.teacher_top1_ids,
                input_token_ids=token_ids,
                prompt_id=prompt.prompt_id,
                model_id=args.model_id,
                revision=args.revision,
                teacher_kind=args.teacher_kind,
                shard_path=shard_relative,
                max_positions=args.max_positions,
            )
        else:
            row, tensors = build_teacher_cache_payload(
                logits=output,
                input_token_ids=token_ids,
                prompt_id=prompt.prompt_id,
                model_id=args.model_id,
                revision=args.revision,
                teacher_kind=args.teacher_kind,
                shard_path=shard_relative,
                top_k=args.top_k,
                max_positions=args.max_positions,
                save_full_logits=not args.no_full_logits,
            )
        row.update(metadata_update)
        mx.save_safetensors(str(output_dir / shard_relative), tensors)
        with metadata_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        summaries.append(
            {
                "prompt_id": prompt.prompt_id,
                "prompt_index": prompt_index,
                "input_token_count": len(token_ids),
                "selected_position_count": len(row["positions"]),
                "elapsed_seconds": metrics["elapsed_seconds"],
                "mlx_peak_bytes": metrics["mlx_peak_bytes"],
                "pageouts_delta": metrics["pageouts_delta"],
                "swapouts_delta": metrics["swapouts_delta"],
                "record_type": row.get("record_type", "teacher_cache"),
                "full_logits_available": row["full_logits_available"],
                "shard": shard_relative,
                "mlx_cache_limit_bytes": mlx_memory_settings["mlx_cache_limit_bytes"],
                "lm_head_chunk_rows": args.lm_head_chunk_rows if args.stream_lm_head else None,
            }
        )
        del lower_hidden_spill

    print(
        json.dumps(
            {
                "backend": "local_sequential",
                "distributed_size": 1,
                "pipeline_size": 2,
                "layer_split": args.layer_split,
                "metadata_jsonl": str(metadata_path),
                "output_dir": str(output_dir),
                "prompt_set": args.prompt_set,
                "record_count": len(summaries),
                "records": summaries,
                "top_k": args.top_k,
                "max_positions": args.max_positions,
                "full_logits_available": not args.no_full_logits,
                "route_trace_only": False,
                "route_trace_layers": [],
                "rank_view_roots": {"0": str(rank0_dir), "1": str(rank1_dir)},
                "lm_head_chunk_rows": args.lm_head_chunk_rows if args.stream_lm_head else None,
                "mlx_memory_settings": mlx_memory_settings,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    prompt_choices = list(get_quality_prompt_ids(prompt_set=None))
    parser = argparse.ArgumentParser(
        description="Export a GLM-4.5-Air high-bit teacher cache via two-rank MLX pipeline JACCL."
    )
    parser.add_argument("--backend", default="jaccl", choices=["any", "ring", "jaccl"])
    parser.add_argument("--rank-view-roots-json", required=True, type=_parse_rank_view_roots)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--local-sequential",
        action="store_true",
        help=(
            "Run two materialized pipeline rank views sequentially in one process. "
            "This avoids JACCL and keeps only one rank view loaded at a time."
        ),
    )
    parser.add_argument(
        "--local-sequential-stage-processes",
        action="store_true",
        help=(
            "When used with --local-sequential, run the lower and upper stages in "
            "separate child processes so stage-local MLX allocations exit before "
            "the next stage loads."
        ),
    )
    parser.add_argument(
        "--local-sequential-head-process",
        action="store_true",
        help=(
            "With --local-sequential-stage-processes and streamed lm_head output, "
            "run lm_head streaming in a third child after the upper-stage model exits."
        ),
    )
    parser.add_argument(
        "--local-sequential-lower-split-layer",
        type=int,
        help=(
            "With --local-sequential-stage-processes, split rank 1 lower-stage "
            "work at this global layer index. The first lower child embeds and "
            "runs 0..N-1; the final lower child runs N..layer_split-1."
        ),
    )
    parser.add_argument(
        "--local-sequential-upper-split-layer",
        type=int,
        help=(
            "With --local-sequential-stage-processes, split rank 0 upper-stage "
            "work at this global layer index. The first upper child runs "
            "layer_split..N-1 without norm; the final upper child runs N..end."
        ),
    )
    parser.add_argument(
        "--local-sequential-lower-split-layers",
        type=_parse_int_list,
        help=(
            "Comma-separated lower-rank layer split points for multi-window "
            "local-sequential stage processes."
        ),
    )
    parser.add_argument(
        "--local-sequential-upper-split-layers",
        type=_parse_int_list,
        help=(
            "Comma-separated upper-rank layer split points for multi-window "
            "local-sequential stage processes."
        ),
    )
    parser.add_argument(
        "--local-sequential-stage-view-roots-json",
        type=_parse_local_sequential_stage_view_roots,
        help=(
            "Optional JSON mapping local-sequential worker names to rank view roots. "
            "Worker roots overlay --rank-view-roots-json for that child process."
        ),
    )
    parser.add_argument(
        "--local-sequential-remote-worker",
        action="append",
        help=(
            "Worker name, prefix, wildcard suffix, or comma-separated list to run "
            "over SSH during local-sequential stage-process export."
        ),
    )
    parser.add_argument(
        "--local-sequential-remote-ssh",
        help="SSH destination for --local-sequential-remote-worker.",
    )
    parser.add_argument(
        "--local-sequential-remote-ssh-option",
        action="append",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--local-sequential-remote-repo",
        help="Repository path on the remote host used as the command working directory.",
    )
    parser.add_argument(
        "--local-sequential-remote-python",
        default="python",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--local-sequential-remote-tmp-dir",
        default="/tmp/glm-local-sequential-workers",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--local-sequential-remote-dirty-retries",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--local-sequential-remote-retry-sleep-seconds",
        type=float,
        default=0.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--local-sequential-remote-stage-view-roots-json",
        type=_parse_remote_stage_view_roots,
        help=(
            "Optional JSON mapping remote local-sequential worker names to "
            "peer-local rank view roots. Worker roots overlay --rank-view-roots-json "
            "only for the remote child command."
        ),
    )
    parser.add_argument(
        "--local-sequential-abort-on-dirty-stage",
        action="store_true",
        help=(
            "With multi-window local-sequential stage processes, abort after the "
            "first worker that records pageouts or swapouts."
        ),
    )
    parser.add_argument(
        "--local-sequential-stage-memory-quiet-preflight",
        action="store_true",
        help=(
            "With local-sequential stage processes, check memory headroom and "
            "pageout quietness before each worker."
        ),
    )
    parser.add_argument(
        "--local-sequential-stage-memory-quiet-seconds",
        type=float,
        default=0.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--local-sequential-stage-memory-quiet-max-attempts",
        type=int,
        default=1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--local-sequential-stage-memory-quiet-min-free-gb",
        type=float,
        default=0.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--local-sequential-stage-memory-quiet-stage-view-margin-gb",
        type=float,
        default=0.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--layer-split",
        type=int,
        help=(
            "Optional two-rank split boundary. Rank 1 owns layers below this "
            "index and rank 0 owns this layer through the final layer."
        ),
    )
    parser.add_argument(
        "--prompt-set",
        choices=get_quality_prompt_set_names(),
        default="base",
        help="Named prompt battery to export when --prompt-id is not supplied.",
    )
    parser.add_argument("--prompt-id", action="append", choices=prompt_choices)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--max-positions", type=int, default=128)
    parser.add_argument("--model-id", default=GLM45_AIR_MODEL_ID)
    parser.add_argument("--revision", default=GLM45_AIR_REVISION)
    parser.add_argument(
        "--teacher-kind",
        choices=["bf16_source", "q8", "other_high_bit"],
        default="bf16_source",
    )
    parser.add_argument(
        "--no-full-logits",
        action="store_true",
        help="Store top-k/target tensors only. This disables exact KLD in the consumer.",
    )
    parser.add_argument(
        "--require-two-ranks",
        action="store_true",
        help="Exit nonzero unless the launcher created at least two ranks.",
    )
    parser.add_argument(
        "--rank0-only-logits",
        action="store_true",
        help=(
            "Use the pipeline graph directly so only rank 0 applies lm_head and rank 0 "
            "uses a BF16 receive placeholder instead of evaluating input embeddings."
        ),
    )
    parser.add_argument(
        "--stream-lm-head",
        action="store_true",
        help=(
            "With --rank0-only-logits, stream rank 0 lm_head.weight from safetensors "
            "in row chunks instead of evaluating the full lm_head parameter."
        ),
    )
    parser.add_argument(
        "--include-route-trace",
        action="store_true",
        help="Include token-level expert indices and router scores for traced layers in metadata rows.",
    )
    parser.add_argument(
        "--route-trace-layer",
        action="append",
        help="Layer index or comma-separated layer indices to trace. Defaults to 31,36,41.",
    )
    parser.add_argument(
        "--route-trace-only",
        action="store_true",
        help="Collect route traces without running lm_head or writing teacher-cache tensor shards.",
    )
    parser.add_argument("--lm-head-chunk-rows", type=int, default=8192)
    parser.add_argument(
        "--mlx-memory-limit-gb",
        type=float,
        help="Optional MLX graph-evaluation memory limit in GiB.",
    )
    parser.add_argument(
        "--mlx-cache-limit-gb",
        type=float,
        help="Optional MLX free-cache limit in GiB. Use 0 to disable the cache.",
    )
    parser.add_argument(
        "--mlx-wired-limit-gb",
        type=float,
        help="Optional MLX wired-memory limit in GiB.",
    )
    parser.add_argument(
        "--local-sequential-stage-worker",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--local-sequential-worker-prompt-id", help=argparse.SUPPRESS)
    parser.add_argument("--local-sequential-hidden-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--local-sequential-output-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--local-sequential-worker-json", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--local-sequential-worker-input-json", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--local-sequential-window-rank", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--local-sequential-window-start-layer", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--local-sequential-window-stop-after-layer", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--local-sequential-window-input-prefix", help=argparse.SUPPRESS)
    parser.add_argument("--local-sequential-window-output-prefix", help=argparse.SUPPRESS)
    parser.add_argument(
        "--local-sequential-window-include-embed",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--local-sequential-window-include-norm",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.max_positions is not None and args.max_positions <= 0:
        parser.error("--max-positions must be positive")
    if args.layer_split is not None and args.layer_split <= 0:
        parser.error("--layer-split must be positive")
    try:
        memory_limit_bytes = _gb_to_bytes(args.mlx_memory_limit_gb, name="--mlx-memory-limit-gb")
        cache_limit_bytes = _gb_to_bytes(
            args.mlx_cache_limit_gb,
            name="--mlx-cache-limit-gb",
            allow_zero=True,
        )
        wired_limit_bytes = _gb_to_bytes(args.mlx_wired_limit_gb, name="--mlx-wired-limit-gb")
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if args.stream_lm_head and not args.rank0_only_logits:
        parser.error("--stream-lm-head requires --rank0-only-logits")
    if args.local_sequential:
        if args.require_two_ranks:
            parser.error("--local-sequential cannot be combined with --require-two-ranks")
        if args.layer_split is None:
            parser.error("--local-sequential requires --layer-split")
        if 0 not in args.rank_view_roots_json or 1 not in args.rank_view_roots_json:
            parser.error("--local-sequential requires rank view roots for ranks 0 and 1")
        if args.route_trace_only or args.include_route_trace:
            parser.error("--local-sequential does not support route-trace export")
        if not args.rank0_only_logits:
            parser.error("--local-sequential requires --rank0-only-logits")
        if not args.stream_lm_head:
            parser.error("--local-sequential requires --stream-lm-head")
    if args.route_trace_only:
        args.include_route_trace = True
    try:
        route_trace_layers = _parse_route_trace_layers(args.route_trace_layer)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if args.lm_head_chunk_rows <= 0:
        parser.error("--lm-head-chunk-rows must be positive")
    if args.local_sequential_stage_processes and not args.local_sequential:
        parser.error("--local-sequential-stage-processes requires --local-sequential")
    if (
        args.local_sequential_stage_view_roots_json
        and not args.local_sequential_stage_processes
        and args.local_sequential_stage_worker is None
    ):
        parser.error(
            "--local-sequential-stage-view-roots-json requires "
            "--local-sequential-stage-processes"
        )
    if args.local_sequential_remote_worker and args.local_sequential_stage_worker is not None:
        parser.error("--local-sequential-remote-worker is only for the parent stage-process export")
    if args.local_sequential_remote_worker:
        if not args.local_sequential:
            parser.error("--local-sequential-remote-worker requires --local-sequential")
        if not args.local_sequential_stage_processes:
            parser.error(
                "--local-sequential-remote-worker requires --local-sequential-stage-processes"
            )
        if not args.local_sequential_remote_ssh:
            parser.error("--local-sequential-remote-worker requires --local-sequential-remote-ssh")
    if args.local_sequential_remote_ssh and not args.local_sequential_remote_worker:
        parser.error("--local-sequential-remote-ssh requires --local-sequential-remote-worker")
    if args.local_sequential_remote_stage_view_roots_json and not args.local_sequential_remote_worker:
        parser.error(
            "--local-sequential-remote-stage-view-roots-json requires "
            "--local-sequential-remote-worker"
        )
    if args.local_sequential_remote_dirty_retries < 0:
        parser.error("--local-sequential-remote-dirty-retries must be non-negative")
    if args.local_sequential_remote_retry_sleep_seconds < 0:
        parser.error("--local-sequential-remote-retry-sleep-seconds must be non-negative")
    if args.local_sequential_lower_split_layer is not None:
        if args.layer_split is None:
            parser.error("--local-sequential-lower-split-layer requires --layer-split")
        if args.local_sequential_lower_split_layer <= 0:
            parser.error("--local-sequential-lower-split-layer must be positive")
        if args.local_sequential_lower_split_layer >= args.layer_split:
            parser.error("--local-sequential-lower-split-layer must be less than --layer-split")
        if (
            not args.local_sequential_stage_processes
            and args.local_sequential_stage_worker is None
        ):
            parser.error(
                "--local-sequential-lower-split-layer requires "
                "--local-sequential-stage-processes"
            )
    if args.local_sequential_lower_split_layers is not None:
        if args.layer_split is None:
            parser.error("--local-sequential-lower-split-layers requires --layer-split")
        if args.local_sequential_lower_split_layer is not None:
            parser.error(
                "--local-sequential-lower-split-layers cannot be combined with "
                "--local-sequential-lower-split-layer"
            )
        for split_layer in args.local_sequential_lower_split_layers:
            if split_layer <= 0 or split_layer >= args.layer_split:
                parser.error(
                    "--local-sequential-lower-split-layers entries must be greater than 0 "
                    "and less than --layer-split"
                )
        if (
            not args.local_sequential_stage_processes
            and args.local_sequential_stage_worker is None
        ):
            parser.error(
                "--local-sequential-lower-split-layers requires "
                "--local-sequential-stage-processes"
            )
    if args.local_sequential_upper_split_layer is not None:
        if args.layer_split is None:
            parser.error("--local-sequential-upper-split-layer requires --layer-split")
        if args.local_sequential_upper_split_layer <= args.layer_split:
            parser.error("--local-sequential-upper-split-layer must be greater than --layer-split")
        if (
            not args.local_sequential_stage_processes
            and args.local_sequential_stage_worker is None
        ):
            parser.error(
                "--local-sequential-upper-split-layer requires "
                "--local-sequential-stage-processes"
            )
    if args.local_sequential_upper_split_layers is not None:
        if args.layer_split is None:
            parser.error("--local-sequential-upper-split-layers requires --layer-split")
        if args.local_sequential_upper_split_layer is not None:
            parser.error(
                "--local-sequential-upper-split-layers cannot be combined with "
                "--local-sequential-upper-split-layer"
            )
        for split_layer in args.local_sequential_upper_split_layers:
            if split_layer <= args.layer_split:
                parser.error(
                    "--local-sequential-upper-split-layers entries must be greater than "
                    "--layer-split"
                )
        if (
            not args.local_sequential_stage_processes
            and args.local_sequential_stage_worker is None
        ):
            parser.error(
                "--local-sequential-upper-split-layers requires "
                "--local-sequential-stage-processes"
            )
    if (
        (
            args.local_sequential_lower_split_layers is not None
            or args.local_sequential_upper_split_layers is not None
        )
        and not args.local_sequential_head_process
        and args.local_sequential_stage_worker is None
    ):
        parser.error("multi-window local-sequential split layers require head-process mode")
    if (
        args.local_sequential_head_process
        and not args.local_sequential_stage_processes
        and args.local_sequential_stage_worker is None
    ):
        parser.error("--local-sequential-head-process requires --local-sequential-stage-processes")
    if args.local_sequential_head_process and not args.stream_lm_head:
        parser.error("--local-sequential-head-process requires --stream-lm-head")
    if args.local_sequential_stage_worker is not None:
        if args.layer_split is None:
            parser.error("--local-sequential-stage-worker requires --layer-split")
        if 0 not in args.rank_view_roots_json or 1 not in args.rank_view_roots_json:
            parser.error("--local-sequential-stage-worker requires rank view roots for ranks 0 and 1")
        if not args.local_sequential_worker_prompt_id:
            parser.error("--local-sequential-stage-worker requires --local-sequential-worker-prompt-id")
        if args.local_sequential_hidden_path is None:
            parser.error("--local-sequential-stage-worker requires --local-sequential-hidden-path")
        if args.local_sequential_worker_json is None:
            parser.error("--local-sequential-stage-worker requires --local-sequential-worker-json")
        mlx_memory_settings = _apply_mlx_memory_settings(
            memory_limit_bytes=memory_limit_bytes,
            cache_limit_bytes=cache_limit_bytes,
            wired_limit_bytes=wired_limit_bytes,
        )
        if args.local_sequential_stage_worker == "lower":
            _run_local_sequential_lower_stage_worker(
                args,
                mlx_memory_settings=mlx_memory_settings,
            )
            return
        if args.local_sequential_stage_worker == "lower-embed":
            if args.local_sequential_output_path is None:
                parser.error("lower-embed stage worker requires --local-sequential-output-path")
            _run_local_sequential_lower_embed_stage_worker(
                args,
                mlx_memory_settings=mlx_memory_settings,
            )
            return
        if args.local_sequential_stage_worker == "lower-pre":
            if args.local_sequential_lower_split_layer is None:
                parser.error("lower-pre stage worker requires --local-sequential-lower-split-layer")
            _run_local_sequential_lower_pre_stage_worker(
                args,
                mlx_memory_settings=mlx_memory_settings,
            )
            return
        if args.local_sequential_stage_worker.startswith(("lower-window-", "upper-window-")):
            if args.local_sequential_window_rank is None:
                parser.error("window stage worker requires --local-sequential-window-rank")
            if args.local_sequential_window_start_layer is None:
                parser.error("window stage worker requires --local-sequential-window-start-layer")
            if args.local_sequential_window_output_prefix is None:
                parser.error("window stage worker requires --local-sequential-window-output-prefix")
            if args.local_sequential_output_path is None:
                parser.error("window stage worker requires --local-sequential-output-path")
            if not args.local_sequential_window_include_embed:
                if args.local_sequential_worker_input_json is None:
                    parser.error(
                        "window stage worker without embeddings requires "
                        "--local-sequential-worker-input-json"
                    )
                if args.local_sequential_window_input_prefix is None:
                    parser.error(
                        "window stage worker without embeddings requires "
                        "--local-sequential-window-input-prefix"
                    )
            _run_local_sequential_window_stage_worker(
                args,
                mlx_memory_settings=mlx_memory_settings,
            )
            return
        if args.local_sequential_output_path is None:
            parser.error(
                f"{args.local_sequential_stage_worker} stage worker requires "
                "--local-sequential-output-path"
            )
        if args.local_sequential_worker_input_json is None:
            parser.error(
                f"{args.local_sequential_stage_worker} stage worker requires "
                "--local-sequential-worker-input-json"
            )
        if args.local_sequential_stage_worker == "lower-final":
            if args.local_sequential_lower_split_layer is None:
                parser.error(
                    "lower-final stage worker requires --local-sequential-lower-split-layer"
                )
            _run_local_sequential_lower_final_stage_worker(
                args,
                mlx_memory_settings=mlx_memory_settings,
            )
            return
        if args.local_sequential_stage_worker == "upper":
            _run_local_sequential_upper_stage_worker(
                args,
                mlx_memory_settings=mlx_memory_settings,
            )
            return
        if args.local_sequential_stage_worker == "upper-pre":
            if args.local_sequential_upper_split_layer is None:
                parser.error("upper-pre stage worker requires --local-sequential-upper-split-layer")
            _run_local_sequential_upper_pre_stage_worker(
                args,
                mlx_memory_settings=mlx_memory_settings,
            )
            return
        if args.local_sequential_stage_worker == "upper-final":
            if args.local_sequential_upper_split_layer is None:
                parser.error("upper-final stage worker requires --local-sequential-upper-split-layer")
            _run_local_sequential_upper_final_stage_worker(
                args,
                mlx_memory_settings=mlx_memory_settings,
            )
            return
        _run_local_sequential_head_stage_worker(
            args,
            mlx_memory_settings=mlx_memory_settings,
        )
        return

    if args.local_sequential:
        mlx_memory_settings = _apply_mlx_memory_settings(
            memory_limit_bytes=memory_limit_bytes,
            cache_limit_bytes=cache_limit_bytes,
            wired_limit_bytes=wired_limit_bytes,
        )
        _run_local_sequential_export(args, mlx_memory_settings=mlx_memory_settings)
        return

    group = mx.distributed.init(strict=args.require_two_ranks, backend=args.backend)
    rank = int(group.rank())
    distributed_size = int(group.size())
    if args.require_two_ranks and distributed_size < 2:
        raise SystemExit("--require-two-ranks needs at least two distributed ranks")
    if args.layer_split is not None and distributed_size != 2:
        parser.error("--layer-split is currently supported only with two distributed ranks")
    mlx_memory_settings = _apply_mlx_memory_settings(
        memory_limit_bytes=memory_limit_bytes,
        cache_limit_bytes=cache_limit_bytes,
        wired_limit_bytes=wired_limit_bytes,
    )
    if rank not in args.rank_view_roots_json:
        available = ", ".join(str(key) for key in sorted(args.rank_view_roots_json))
        raise SystemExit(f"no rank view root for rank {rank}; available ranks: {available}")

    rank_dir = args.rank_view_roots_json[rank]
    rank0_stop_after_layer = max(route_trace_layers) if args.route_trace_only and args.layer_split is not None else None
    model, tokenizer = _load_rank_model_and_tokenizer(
        rank_dir,
        rank=rank,
        pipeline_size=distributed_size,
        layer_split=args.layer_split,
        rank0_stop_after_layer=rank0_stop_after_layer,
    )
    route_trace_stop_after_gate = max(route_trace_layers) if args.route_trace_only else None
    route_tracer = _RouteCoverageTracer(
        layers=route_trace_layers,
        include_trace=args.include_route_trace,
        stop_after_gate_layer=route_trace_stop_after_gate,
    )
    route_tracer.install(model.model)

    output_dir = Path(args.output_dir)
    shard_dir = output_dir / "teacher_logits"
    metadata_path = output_dir / "metadata.jsonl"
    if rank == 0:
        shard_dir.mkdir(parents=True, exist_ok=True)
        if metadata_path.exists():
            metadata_path.unlink()

    selected_prompts = _selected_prompts(args.prompt_id, prompt_set=args.prompt_set)
    summaries: list[dict[str, Any]] = []
    for prompt in selected_prompts:
        token_ids = _prompt_tokens(tokenizer, prompt)
        if len(token_ids) < 2:
            raise ValueError(f"quality prompt {prompt.prompt_id!r} tokenized to fewer than two tokens")

        route_tracer.reset()
        previous_vm = _collect_vm_stat_counts()
        mx.reset_peak_memory()
        start = time.perf_counter()
        if args.route_trace_only:
            output = _forward_pipeline_rank0_output(
                model,
                token_ids,
                rank=rank,
                pipeline_size=distributed_size,
                return_hidden=True,
                stop_after_layer=max(route_trace_layers),
                stop_after_route_gate=route_trace_stop_after_gate,
                normalize_output=False,
            )
        elif args.rank0_only_logits:
            output = _forward_pipeline_rank0_output(
                model,
                token_ids,
                rank=rank,
                pipeline_size=distributed_size,
                return_hidden=args.stream_lm_head,
                diagnose_hidden_stages=args.stream_lm_head,
            )
            if args.stream_lm_head and rank == 0:
                if args.no_full_logits:
                    output = _stream_lm_head_topk_payload(
                        output,
                        rank_dir=rank_dir,
                        input_token_ids=token_ids,
                        max_positions=args.max_positions,
                        chunk_rows=args.lm_head_chunk_rows,
                        top_k=args.top_k,
                    )
                else:
                    output = _stream_lm_head_logits(
                        output,
                        rank_dir=rank_dir,
                        input_token_ids=token_ids,
                        max_positions=args.max_positions,
                        chunk_rows=args.lm_head_chunk_rows,
                    )
        else:
            output = model(mx.array([token_ids], dtype=mx.int32))
        if isinstance(output, mx.array):
            mx.eval(output)
        elapsed_seconds = time.perf_counter() - start
        metric_snapshot = _collect_metric_snapshot(previous_vm)
        rank_metrics = _collect_rank_metrics(
            metric_snapshot,
            elapsed_seconds=elapsed_seconds,
            rank=rank,
            distributed_size=distributed_size,
        )
        metrics = _aggregate_rank_metrics(rank_metrics)
        shard_relative = f"teacher_logits/{prompt.prompt_id}.safetensors"
        route_coverage = route_tracer.summary(row_intent=prompt.row_intent)
        route_trace = {
            layer: coverage["route_trace"]
            for layer, coverage in route_coverage.items()
            if isinstance(coverage, dict) and "route_trace" in coverage
        }

        if rank == 0:
            metadata_update = {
                **metrics,
                "distributed_backend": args.backend,
                "distributed_size": distributed_size,
                "pipeline_size": distributed_size,
                "layer_split": args.layer_split,
                "rank0_host": socket.gethostname(),
                "rank0_view": str(rank_dir),
                "rank_view_roots": {
                    str(view_rank): str(view_root)
                    for view_rank, view_root in sorted(args.rank_view_roots_json.items())
                },
                "metric_aggregation": "max_across_distributed_ranks",
                "rank_metrics": rank_metrics,
                "rank0_only_logits": bool(args.rank0_only_logits),
                "stream_lm_head": bool(args.stream_lm_head),
                "route_trace_only": bool(args.route_trace_only),
                "route_coverage": route_coverage,
                **quality_prompt_metadata(prompt, prompt_set=args.prompt_set),
                **mlx_memory_settings,
            }
            if args.route_trace_only:
                row = _build_route_trace_only_row(
                    prompt=prompt,
                    prompt_set=args.prompt_set,
                    input_token_ids=token_ids,
                    max_positions=args.max_positions,
                    model_id=args.model_id,
                    revision=args.revision,
                    teacher_kind=args.teacher_kind,
                    route_coverage=route_coverage,
                    route_trace=route_trace,
                    metadata=metadata_update,
                )
            else:
                from keep.quality.teacher_cache import (
                    build_teacher_cache_payload,
                    build_teacher_cache_topk_payload,
                )

                if isinstance(output, _TopKOnlyPayload):
                    row, tensors = build_teacher_cache_topk_payload(
                        topk_ids=output.topk_ids,
                        topk_logprobs=output.topk_logprobs,
                        target_logprobs=output.target_logprobs,
                        teacher_top1_ids=output.teacher_top1_ids,
                        input_token_ids=token_ids,
                        prompt_id=prompt.prompt_id,
                        model_id=args.model_id,
                        revision=args.revision,
                        teacher_kind=args.teacher_kind,
                        shard_path=shard_relative,
                        max_positions=args.max_positions,
                    )
                else:
                    row, tensors = build_teacher_cache_payload(
                        logits=output,
                        input_token_ids=token_ids,
                        prompt_id=prompt.prompt_id,
                        model_id=args.model_id,
                        revision=args.revision,
                        teacher_kind=args.teacher_kind,
                        shard_path=shard_relative,
                        top_k=args.top_k,
                        max_positions=args.max_positions,
                        save_full_logits=not args.no_full_logits,
                    )
                if args.include_route_trace:
                    metadata_update["route_trace"] = route_trace
                row.update(metadata_update)
                mx.save_safetensors(str(output_dir / shard_relative), tensors)
            with metadata_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            summary_record = {
                "prompt_id": prompt.prompt_id,
                "input_token_count": len(token_ids),
                "selected_position_count": len(row["positions"]),
                "elapsed_seconds": metrics["elapsed_seconds"],
                "mlx_peak_bytes": metrics["mlx_peak_bytes"],
                "pageouts_delta": metrics["pageouts_delta"],
                "swapouts_delta": metrics["swapouts_delta"],
                "record_type": row.get("record_type", "teacher_cache"),
                "route_coverage_layers": sorted(route_coverage, key=int),
                "mlx_cache_limit_bytes": mlx_memory_settings["mlx_cache_limit_bytes"],
            }
            if not args.route_trace_only:
                summary_record["full_logits_available"] = row["full_logits_available"]
                summary_record["shard"] = shard_relative
            if args.include_route_trace:
                summary_record["route_trace_layers"] = sorted(route_trace, key=int)
            summaries.append(summary_record)

        sync = mx.distributed.all_sum(mx.array([float(rank + 1)], dtype=mx.float32))
        mx.eval(sync)

    if rank == 0:
        print(
            json.dumps(
                {
                    "backend": args.backend,
                    "distributed_size": distributed_size,
                    "layer_split": args.layer_split,
                    "metadata_jsonl": str(metadata_path),
                    "output_dir": str(output_dir),
                    "prompt_set": args.prompt_set,
                    "record_count": len(summaries),
                    "records": summaries,
                    "top_k": args.top_k,
                    "max_positions": args.max_positions,
                    "full_logits_available": None if args.route_trace_only else not args.no_full_logits,
                    "route_trace_only": bool(args.route_trace_only),
                    "route_trace_layers": [int(layer) for layer in route_trace_layers],
                    "mlx_memory_settings": mlx_memory_settings,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        print(
            json.dumps(
                {
                    "backend": args.backend,
                    "distributed_size": distributed_size,
                    "layer_split": args.layer_split,
                    "host": socket.gethostname(),
                    "rank": rank,
                    "prompt_set": args.prompt_set,
                    "record_count": len(selected_prompts),
                    "status": "participated",
                    "mlx_memory_settings": mlx_memory_settings,
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
