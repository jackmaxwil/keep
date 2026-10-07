from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from ramp.benchmark.metrics import (
    collect_metric_snapshot,
    collect_vm_stat_counts,
    reset_mlx_peak_memory,
)
from ramp.models.qwen_moe_adapter import load_qwen_moe_switch_glu


SwitchLoader = Callable[[str | Path, int], Any]


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


def _scenario_token_count(scenario: str) -> int:
    if scenario == "prefill_1k":
        return 1024
    if scenario == "decode_128":
        return 128
    raise ValueError(f"unsupported Qwen benchmark scenario {scenario!r}")


def _required_scenarios(family_policy: dict[str, Any]) -> tuple[str, ...]:
    benchmark_gate = family_policy.get("benchmark_gate") or {}
    return tuple(
        scenario
        for scenario in benchmark_gate.get("required_scenarios") or ()
        if isinstance(scenario, str)
    )


def _minimum_repetitions(family_policy: dict[str, Any]) -> int:
    benchmark_gate = family_policy.get("benchmark_gate") or {}
    return int(benchmark_gate.get("minimum_repetitions_per_scenario") or 1)


def _inputs_for_switch(
    *,
    token_count: int,
    input_dims: int,
    top_k: int,
    num_experts: int,
    input_scale: float,
) -> tuple[mx.array, mx.array]:
    values = np.linspace(
        -float(input_scale),
        float(input_scale),
        token_count * input_dims,
        dtype=np.float32,
    ).reshape(token_count, input_dims)
    indices = np.tile(np.arange(top_k, dtype=np.int32), (token_count, 1)) % num_experts
    return mx.array(values), mx.array(indices)


def _run_candidate_once(
    switch_mlp: Any,
    *,
    scenario: str,
    token_count: int,
    top_k: int,
    input_scale: float,
) -> dict[str, Any]:
    x, indices = _inputs_for_switch(
        token_count=token_count,
        input_dims=int(switch_mlp.input_dims),
        top_k=top_k,
        num_experts=int(switch_mlp.num_experts),
        input_scale=input_scale,
    )
    reset_mlx_peak_memory()
    before_vm = collect_vm_stat_counts()
    start = time.perf_counter()
    output = switch_mlp(x, indices)
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


def run_qwen_candidate_latency_benchmark(
    *,
    family_policy: dict[str, Any],
    artifact_dir: str | Path,
    output_jsonl: str | Path,
    layer: int = 0,
    repetitions: int | None = None,
    warmup_repetitions: int = 1,
    top_k: int = 8,
    input_scale: float = 0.125,
    switch_loader: SwitchLoader | None = None,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if warmup_repetitions < 0:
        raise ValueError("warmup_repetitions must be non-negative")
    repetition_count = _minimum_repetitions(family_policy) if repetitions is None else repetitions
    if repetition_count <= 0:
        raise ValueError("repetitions must be positive")
    loader = switch_loader or (
        lambda root, layer_idx: load_qwen_moe_switch_glu(root, layer=layer_idx)
    )
    switch_mlp = loader(artifact_dir, layer)
    scenarios = _required_scenarios(family_policy)
    rows: list[dict[str, Any]] = []
    scenario_counts = {scenario: 0 for scenario in scenarios}
    row_errors: list[dict[str, Any]] = []

    for scenario in scenarios:
        token_count = _scenario_token_count(scenario)
        for _ in range(warmup_repetitions):
            warmup = _run_candidate_once(
                switch_mlp,
                scenario=scenario,
                token_count=token_count,
                top_k=top_k,
                input_scale=input_scale,
            )
            if not warmup["finite"]:
                row_errors.append({"scenario": scenario, "error": "warmup_output_not_finite"})
        for run_index in range(repetition_count):
            result = _run_candidate_once(
                switch_mlp,
                scenario=scenario,
                token_count=token_count,
                top_k=top_k,
                input_scale=input_scale,
            )
            row = {
                "record_type": "qwen_benchmark_latency_row",
                "model_id": family_policy.get("model_id"),
                "revision": family_policy.get("revision"),
                "benchmark_scope": "qwen_vq_switch_projection_candidate_latency",
                "scenario": scenario,
                "role": "candidate",
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
            if not row["finite"]:
                row_errors.append(
                    {
                        "scenario": scenario,
                        "run_index": run_index,
                        "error": "candidate_output_not_finite",
                    }
                )
            rows.append(row)
            scenario_counts[scenario] += 1

    _write_jsonl(output_jsonl, rows)
    missing_requirements: list[str] = []
    if row_errors:
        missing_requirements.append("qwen_candidate_latency_rows_finite")
    if any(count < repetition_count for count in scenario_counts.values()):
        missing_requirements.append("qwen_candidate_latency_repetitions")

    return {
        "record_type": "qwen_candidate_benchmark_latency_probe",
        "model_id": family_policy.get("model_id"),
        "revision": family_policy.get("revision"),
        "artifact_dir": str(Path(artifact_dir)),
        "output_jsonl": str(Path(output_jsonl)),
        "benchmark_scope": "qwen_vq_switch_projection_candidate_latency",
        "role": "candidate",
        "layer": layer,
        "required_scenarios": list(scenarios),
        "minimum_repetitions_per_scenario": repetition_count,
        "candidate_latency_row_count": len(rows),
        "scenario_counts": scenario_counts,
        "row_error_count": len(row_errors),
        "row_errors": row_errors[:20],
        "missing_requirements": missing_requirements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark local Qwen VQ candidate switch latency rows."
    )
    parser.add_argument("--family-policy-json", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--input-scale", type=float, default=0.125)
    parser.add_argument("--output-json")
    parser.add_argument("--append-jsonl")
    args = parser.parse_args()

    output_jsonl = Path(args.output_dir) / "qwen_candidate_benchmark_latency_rows.jsonl"
    payload = run_qwen_candidate_latency_benchmark(
        family_policy=_load_json(args.family_policy_json),
        artifact_dir=args.artifact_dir,
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
