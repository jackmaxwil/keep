from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

try:
    from benchmarks.bench_qwen_family_candidate_latency import (
        _inputs_for_switch,
        _minimum_repetitions,
        _required_scenarios,
        _scenario_token_count,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path.
    from bench_qwen_family_candidate_latency import (
        _inputs_for_switch,
        _minimum_repetitions,
        _required_scenarios,
        _scenario_token_count,
    )
from mlx_vq.benchmark.metrics import (
    collect_metric_snapshot,
    collect_vm_stat_counts,
    reset_mlx_peak_memory,
)
from mlx_vq.convert.stream_convert import load_safetensors_index
from mlx_vq.io.source_safetensors import read_indexed_safetensors_tensor_mlx


@dataclass(frozen=True)
class QwenSourceSwitchControl:
    gate_weight: mx.array
    up_weight: mx.array
    down_weight: mx.array

    @property
    def input_dims(self) -> int:
        return int(self.gate_weight.shape[2])

    @property
    def num_experts(self) -> int:
        return int(self.gate_weight.shape[0])

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        route_outputs: list[mx.array] = []
        for route_idx in range(int(indices.shape[1])):
            route_experts = indices[:, route_idx]
            expert_id = int(np.array(route_experts[0]).item())
            if not bool(np.all(np.array(route_experts) == expert_id)):
                raise ValueError("control benchmark expects one stable expert per route")
            x_f32 = x.astype(mx.float32)
            gate = x_f32 @ self.gate_weight[expert_id].T.astype(mx.float32)
            up = x_f32 @ self.up_weight[expert_id].T.astype(mx.float32)
            hidden = nn.silu(gate) * up
            route_outputs.append(hidden @ self.down_weight[expert_id].T.astype(mx.float32))
        return mx.stack(route_outputs, axis=1)


ControlLoader = Callable[[str | Path, str | Path, int], QwenSourceSwitchControl]


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def load_qwen_source_switch_control(
    source_dir: str | Path,
    index_path: str | Path,
    layer: int,
) -> QwenSourceSwitchControl:
    index = load_safetensors_index(index_path)
    prefix = f"model.language_model.layers.{layer}.mlp.experts"
    gate_up = read_indexed_safetensors_tensor_mlx(
        source_dir,
        index,
        f"{prefix}.gate_up_proj",
    )
    down = read_indexed_safetensors_tensor_mlx(
        source_dir,
        index,
        f"{prefix}.down_proj",
    )
    if gate_up.ndim != 3 or down.ndim != 3:
        raise ValueError("Qwen source expert tensors must be 3D")
    if int(gate_up.shape[1]) % 2 != 0:
        raise ValueError("Qwen gate_up source tensor must have even fused output dim")
    split = int(gate_up.shape[1]) // 2
    return QwenSourceSwitchControl(
        gate_weight=gate_up[:, :split, :],
        up_weight=gate_up[:, split:, :],
        down_weight=down,
    )


def _run_control_once(
    control: QwenSourceSwitchControl,
    *,
    scenario: str,
    token_count: int,
    top_k: int,
    input_scale: float,
) -> dict[str, Any]:
    x, indices = _inputs_for_switch(
        token_count=token_count,
        input_dims=control.input_dims,
        top_k=top_k,
        num_experts=control.num_experts,
        input_scale=input_scale,
    )
    reset_mlx_peak_memory()
    before_vm = collect_vm_stat_counts()
    start = time.perf_counter()
    output = control(x, indices)
    mx.eval(output)
    elapsed_seconds = time.perf_counter() - start
    metrics = collect_metric_snapshot(previous_vm_stat_counts=before_vm)
    return {
        "scenario": scenario,
        "token_count": token_count,
        "top_k": top_k,
        "latency_ms": elapsed_seconds * 1000.0,
        "output_shape": list(output.shape),
        "finite": bool(np.isfinite(np.array(output)).all()),
        **metrics,
    }


def run_qwen_control_latency_benchmark(
    *,
    family_policy: dict[str, Any],
    source_dir: str | Path,
    index_path: str | Path,
    output_jsonl: str | Path,
    layer: int = 0,
    repetitions: int | None = None,
    warmup_repetitions: int = 1,
    top_k: int = 8,
    input_scale: float = 0.125,
    control_loader: ControlLoader | None = None,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if warmup_repetitions < 0:
        raise ValueError("warmup_repetitions must be non-negative")
    repetition_count = (
        _minimum_repetitions(family_policy) if repetitions is None else repetitions
    )
    if repetition_count <= 0:
        raise ValueError("repetitions must be positive")
    loader = control_loader or load_qwen_source_switch_control
    control = loader(source_dir, index_path, layer)
    scenarios = _required_scenarios(family_policy)
    rows: list[dict[str, Any]] = []
    scenario_counts = {scenario: 0 for scenario in scenarios}
    discarded_attempt_counts = {scenario: 0 for scenario in scenarios}
    row_errors: list[dict[str, Any]] = []

    for scenario in scenarios:
        token_count = _scenario_token_count(scenario)
        for _ in range(warmup_repetitions):
            warmup = _run_control_once(
                control,
                scenario=scenario,
                token_count=token_count,
                top_k=top_k,
                input_scale=input_scale,
            )
            if not warmup["finite"]:
                row_errors.append(
                    {"scenario": scenario, "error": "warmup_output_not_finite"}
                )
        run_index = 0
        max_attempts = repetition_count + max(warmup_repetitions, 1)
        while run_index < repetition_count and (
            scenario_counts[scenario] + discarded_attempt_counts[scenario]
        ) < max_attempts:
            result = _run_control_once(
                control,
                scenario=scenario,
                token_count=token_count,
                top_k=top_k,
                input_scale=input_scale,
            )
            if (
                not result["finite"]
                or result["pageouts_delta"] != 0
                or result["swapouts_delta"] != 0
            ):
                discarded_attempt_counts[scenario] += 1
                continue
            row = {
                "record_type": "qwen_benchmark_latency_row",
                "model_id": family_policy.get("model_id"),
                "revision": family_policy.get("revision"),
                "benchmark_scope": "qwen_source_switch_projection_control_latency",
                "scenario": scenario,
                "role": "control",
                "run_index": run_index,
                "layer": layer,
                "token_count": result["token_count"],
                "top_k": result["top_k"],
                "latency_ms": result["latency_ms"],
                "pageouts_delta": result["pageouts_delta"],
                "swapouts_delta": result["swapouts_delta"],
                "mlx_peak_bytes": result["mlx_peak_bytes"],
                "mlx_active_bytes": result["mlx_active_bytes"],
                "mlx_cache_bytes": result["mlx_cache_bytes"],
                "rss_bytes": result["rss_bytes"],
                "output_shape": result["output_shape"],
                "finite": result["finite"],
            }
            rows.append(row)
            scenario_counts[scenario] += 1
            run_index += 1
        if scenario_counts[scenario] < repetition_count:
            row_errors.append(
                {
                    "scenario": scenario,
                    "error": "insufficient_clean_control_latency_rows",
                    "clean_rows": scenario_counts[scenario],
                    "required_rows": repetition_count,
                    "discarded_attempts": discarded_attempt_counts[scenario],
                }
            )

    _write_jsonl(output_jsonl, rows)
    missing_requirements: list[str] = []
    if row_errors:
        missing_requirements.append("qwen_control_latency_rows_finite")
    if any(count < repetition_count for count in scenario_counts.values()):
        missing_requirements.append("qwen_control_latency_repetitions")

    return {
        "record_type": "qwen_control_benchmark_latency_probe",
        "model_id": family_policy.get("model_id"),
        "revision": family_policy.get("revision"),
        "source_dir": str(Path(source_dir)),
        "index_path": str(Path(index_path)),
        "output_jsonl": str(Path(output_jsonl)),
        "benchmark_scope": "qwen_source_switch_projection_control_latency",
        "role": "control",
        "layer": layer,
        "required_scenarios": list(scenarios),
        "minimum_repetitions_per_scenario": repetition_count,
        "control_latency_row_count": len(rows),
        "scenario_counts": scenario_counts,
        "discarded_attempt_counts": discarded_attempt_counts,
        "row_error_count": len(row_errors),
        "row_errors": row_errors[:20],
        "missing_requirements": missing_requirements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark local Qwen source control switch latency rows."
    )
    parser.add_argument("--family-policy-json", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--input-scale", type=float, default=0.125)
    parser.add_argument("--output-json")
    parser.add_argument("--append-jsonl")
    args = parser.parse_args()

    output_jsonl = Path(args.output_dir) / "qwen_control_benchmark_latency_rows.jsonl"
    payload = run_qwen_control_latency_benchmark(
        family_policy=_load_json(args.family_policy_json),
        source_dir=args.source_dir,
        index_path=args.index_path,
        output_jsonl=output_jsonl,
        layer=args.layer,
        repetitions=args.repetitions,
        warmup_repetitions=args.warmup_repetitions,
        top_k=args.top_k,
        input_scale=args.input_scale,
    )
    if args.output_json is not None:
        _write_json(args.output_json, payload)
    if args.append_jsonl is not None:
        _append_jsonl(args.append_jsonl, payload)
    if args.output_json is None and args.append_jsonl is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["missing_requirements"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
