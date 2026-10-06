from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

import mlx_vq.models.glm4_moe_adapter as glm4_moe_adapter
from mlx_vq.benchmark.metrics import (
    collect_metric_snapshot,
    collect_vm_stat_counts,
    reset_mlx_peak_memory,
)
from mlx_vq.io.load import load_quantized_vq_switch_linear
from mlx_vq.models.glm4_moe_adapter import QuantizedVQSwitchGLU

DEFAULT_ARTIFACT_DIR = Path("artifacts/glm-4.5-air-vq2-e8p-rtn-uniform-parallel8")


def _load_switch_glu_from_artifact(
    artifact_dir: Path,
    *,
    layer_index: int,
    prefill_engine: str,
) -> tuple[QuantizedVQSwitchGLU, dict[str, Any]]:
    prefix_base = f"model.layers.{layer_index}.mlp.switch_mlp"
    projections = {}
    shards: dict[str, str] = {}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        shard = artifact_dir / f"layer-{layer_index:05d}-{projection}.safetensors"
        projections[projection] = load_quantized_vq_switch_linear(
            shard,
            f"{prefix_base}.{projection}",
        )
        shards[projection] = str(shard)

    switch_mlp = QuantizedVQSwitchGLU(
        gate_proj=projections["gate_proj"],
        up_proj=projections["up_proj"],
        down_proj=projections["down_proj"],
        prefill_engine=prefill_engine,  # type: ignore[arg-type]
    )
    metadata = {
        "artifact_dir": str(artifact_dir),
        "artifact_layer": layer_index,
        "artifact_shards": shards,
        "projection_metadata": {
            name: {
                "input_dims": int(projection.input_dims),
                "output_dims": int(projection.output_dims),
                "experts": int(projection.num_experts),
                "group_size": int(projection.group_size),
                "code_bits": int(projection.code_bits),
            }
            for name, projection in projections.items()
        },
    }
    return switch_mlp, metadata


def _recorded_native_forward(
    switch_mlp: QuantizedVQSwitchGLU,
    x: mx.array,
    indices: mx.array,
) -> tuple[mx.array, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    real_helper = glm4_moe_adapter.gather_vqmm_sorted_routes

    def recording_helper(x_arg, codes, scales, codebook, sorted_rhs, sorted_lhs, **kwargs):
        calls.append(
            {
                "implementation": kwargs.get("implementation"),
                "projection": kwargs.get("projection"),
                "input_dims": int(kwargs["input_dims"]),
                "output_dims": int(kwargs["output_dims"]),
                "route_count": int(sorted_rhs.shape[0]),
                "x_shape": list(x_arg.shape),
                "lhs_shape": list(sorted_lhs.shape),
            }
        )
        return real_helper(x_arg, codes, scales, codebook, sorted_rhs, sorted_lhs, **kwargs)

    glm4_moe_adapter.gather_vqmm_sorted_routes = recording_helper
    try:
        output = switch_mlp(x, indices)
        mx.eval(output)
    finally:
        glm4_moe_adapter.gather_vqmm_sorted_routes = real_helper
    return output, calls


def _cosine_and_max_abs(lhs: mx.array, rhs: mx.array) -> tuple[float, float, bool]:
    lhs_float = lhs.astype(mx.float32)
    rhs_float = rhs.astype(mx.float32)
    mx.eval(lhs_float, rhs_float)
    dot = mx.sum(lhs_float * rhs_float)
    norm = mx.sqrt(mx.sum(lhs_float * lhs_float) * mx.sum(rhs_float * rhs_float))
    max_abs = mx.max(mx.abs(lhs_float - rhs_float))
    finite = mx.all(mx.isfinite(lhs_float))
    mx.eval(dot, norm, max_abs, finite)
    norm_value = float(norm.item())
    cosine = float((dot / norm).item()) if norm_value > 0.0 else 0.0
    return cosine, float(max_abs.item()), bool(finite.item())


def run_sorted_prefill_dispatch_parity(
    *,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    layer_index: int = 1,
    tokens: int = 8,
    top_k: int = 8,
    seed: int = 20260706,
    min_cosine: float = 0.99998,
    max_abs_diff: float = 2e-2,
) -> dict[str, Any]:
    artifact_root = Path(artifact_dir)
    switch_mlp, source_metadata = _load_switch_glu_from_artifact(
        artifact_root,
        layer_index=layer_index,
        prefill_engine="auto",
    )
    if not all(
        projection["code_bits"] == 16
        for projection in source_metadata["projection_metadata"].values()
    ):
        raise ValueError("sorted prefill E8P dispatch proof requires all projections to be 16-bit")

    rng = np.random.default_rng(seed)
    x = mx.array(
        rng.normal(size=(tokens, switch_mlp.input_dims)).astype(np.float32)
    ).astype(mx.bfloat16)
    indices = mx.array(
        rng.integers(0, switch_mlp.num_experts, size=(tokens, top_k), dtype=np.int32)
    )
    mx.eval(x, indices, switch_mlp.parameters())
    can_use_shared_sorted_prefill = bool(switch_mlp._can_use_shared_sorted_route_prefill(x, indices))
    selected_implementation = switch_mlp._sorted_prefill_implementation(
        route_count=tokens * top_k,
        activation_dtype=x.dtype,
    )

    before_vm = collect_vm_stat_counts()
    reset_mlx_peak_memory()
    native_start = time.perf_counter()
    native_output, dispatch_calls = _recorded_native_forward(switch_mlp, x, indices)
    native_elapsed_seconds = time.perf_counter() - native_start
    metrics = collect_metric_snapshot(previous_vm_stat_counts=before_vm)

    switch_mlp.prefill_engine = "vq_metal"
    reference_start = time.perf_counter()
    reference_output = switch_mlp(x, indices)
    mx.eval(reference_output)
    reference_elapsed_seconds = time.perf_counter() - reference_start

    cosine, max_abs_value, finite_output = _cosine_and_max_abs(native_output, reference_output)
    implementations = [str(call["implementation"]) for call in dispatch_calls]
    projections = [str(call["projection"]) for call in dispatch_calls]
    dispatch_claim = (
        can_use_shared_sorted_prefill
        and selected_implementation == "nax_e8p_m32n64"
        and implementations == ["nax_e8p_m32n64"] * 3
        and projections == ["gate_up", "gate_up", "down"]
    )
    passes_routed_forward_parity = bool(
        dispatch_claim
        and finite_output
        and cosine >= min_cosine
        and max_abs_value <= max_abs_diff
    )
    memory_clean = (
        int(metrics.get("pageouts_delta", 0) or 0) == 0
        and int(metrics.get("swapouts_delta", 0) or 0) == 0
    )
    decision = (
        "sorted_prefill_e8p_dispatch_parity_pass"
        if passes_routed_forward_parity
        else "sorted_prefill_e8p_dispatch_parity_failed"
    )

    return {
        "record_type": "glm45_air_e8p_sorted_prefill_dispatch_parity",
        "schema_version": 1,
        "decision": decision,
        "passes_routed_forward_parity": passes_routed_forward_parity,
        "resident_auto_claim": dispatch_claim,
        "decode_parity_claim": passes_routed_forward_parity,
        "speed_claim": False,
        "peer2_used": False,
        "rdma_jaccl_touched": False,
        "artifact_dir": str(artifact_root),
        "layer_index": layer_index,
        "tokens": tokens,
        "top_k": top_k,
        "seed": seed,
        "route_count": int(tokens * top_k),
        "variant": "nax_e8p_m32n64_sorted_prefill_dispatch",
        "comparison": "auto_nax_e8p_m32n64_sorted_prefill_vs_vq_metal_sorted_prefill",
        "can_use_shared_sorted_prefill": can_use_shared_sorted_prefill,
        "selected_implementation": selected_implementation,
        "dispatch_calls": dispatch_calls,
        "call_implementations": implementations,
        "call_projections": projections,
        "finite_output": finite_output,
        "cosine": cosine,
        "min_cosine": min_cosine,
        "max_abs_diff": max_abs_value,
        "max_abs_diff_threshold": max_abs_diff,
        "output_shape": list(native_output.shape),
        "reference_shape": list(reference_output.shape),
        "native_elapsed_seconds_diagnostic": native_elapsed_seconds,
        "reference_elapsed_seconds_diagnostic": reference_elapsed_seconds,
        "memory_clean": memory_clean,
        **metrics,
        "source_metadata": source_metadata,
        "next_track_b_hypothesis": "run_first_device_benchmark_vs_q2",
        "rejected_next_steps": [
            "do_not_claim_speed_from_dispatch_parity",
            "do_not_add_a_new_kernel_family_for_this_milestone",
            "do_not_use_peer2_for_this_local_trackb_slice",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prove GLM-4.5-Air E8P sorted prefill dispatch hits native NAX and matches Metal."
    )
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--layer-index", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--min-cosine", type=float, default=0.99998)
    parser.add_argument("--max-abs-diff", type=float, default=2e-2)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    report = run_sorted_prefill_dispatch_parity(
        artifact_dir=args.artifact_dir,
        layer_index=args.layer_index,
        tokens=args.tokens,
        top_k=args.top_k,
        seed=args.seed,
        min_cosine=args.min_cosine,
        max_abs_diff=args.max_abs_diff,
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passes_routed_forward_parity"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
