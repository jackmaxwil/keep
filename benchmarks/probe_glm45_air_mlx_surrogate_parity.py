from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_vq.benchmark.glm45_air import append_jsonl
from mlx_vq.codebook.e8 import decode_weight_matrix
from mlx_vq.io.load import load_quantized_vq_switch_linear
from mlx_vq.quant.rht import apply_rht_np
from mlx_vq.quality.mlx_surrogate import (
    SwitchLinearSidecar,
    SwitchLinearSurrogate,
    call_switch_linear_with_optional_sidecar,
)


def _prefix(layer: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.switch_mlp.{projection}"


def _artifact_shard(artifact_dir: Path, layer: int, projection: str) -> Path:
    return artifact_dir / f"layer-{layer:05d}-{projection}.safetensors"


def _reference_output(layer, x: np.ndarray, indices: np.ndarray) -> np.ndarray:
    rht_signs = layer.get("rht_signs")
    if rht_signs is not None:
        x = apply_rht_np(x, np.array(rht_signs, dtype=np.int8))
    weights = np.stack(
        [
            decode_weight_matrix(
                np.array(layer.codes[expert]),
                np.array(layer.scales[expert], dtype=np.float32),
                code_bits=layer.code_bits,
                codebook=np.array(layer.codebook),
            )
            for expert in range(layer.num_experts)
        ],
        axis=0,
    )
    bias = np.array(layer.get("bias"), dtype=np.float32) if layer.get("bias") is not None else None
    out = np.empty((*indices.shape, layer.output_dims), dtype=np.float32)
    for token_idx in range(indices.shape[0]):
        for route_idx in range(indices.shape[1]):
            expert = int(indices[token_idx, route_idx])
            out[token_idx, route_idx] = x[token_idx] @ weights[expert].T
            if bias is not None:
                out[token_idx, route_idx] += bias[expert]
    return out


def _error_stats(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    abs_err = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    rel = abs_err / np.maximum(np.abs(expected.astype(np.float64)), 1e-8)
    return {
        "max_abs": float(np.max(abs_err)),
        "mean_abs": float(np.mean(abs_err)),
        "max_rel": float(np.max(rel)),
        "mean_rel": float(np.mean(rel)),
    }


def _within_tolerance(actual: np.ndarray, expected: np.ndarray, *, atol: float, rtol: float) -> bool:
    abs_err = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    allowed = atol + rtol * np.abs(expected.astype(np.float64))
    return bool(np.all(abs_err <= allowed))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe one GLM-4.5-Air VQ shard with the MLX surrogate parity path."
    )
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument(
        "--projection",
        choices=("gate_proj", "up_proj", "down_proj"),
        required=True,
    )
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument(
        "--append-jsonl",
        default="artifacts/quality/glm45-air-mlx-surrogate-parity.jsonl",
    )
    args = parser.parse_args()
    if args.tokens <= 0:
        parser.error("--tokens must be positive")
    if args.expert < 0:
        parser.error("--expert must be zero or greater")

    artifact_dir = Path(args.artifact_dir)
    shard = _artifact_shard(artifact_dir, args.layer, args.projection)
    layer = load_quantized_vq_switch_linear(shard, _prefix(args.layer, args.projection))
    if args.expert >= layer.num_experts:
        parser.error(f"--expert {args.expert} outside num_experts={layer.num_experts}")

    rng = np.random.default_rng(args.seed)
    x_np = rng.normal(size=(args.tokens, layer.input_dims)).astype(np.float32)
    indices_np = np.full((args.tokens, 1), args.expert, dtype=np.int32)
    x = mx.array(x_np)
    indices = mx.array(indices_np)

    start = time.perf_counter()
    reference = _reference_output(layer, x_np, indices_np)
    surrogate = SwitchLinearSurrogate.from_layer(layer)(x, indices)
    runtime = layer(x, indices)
    zero_sidecar = call_switch_linear_with_optional_sidecar(
        layer,
        x,
        indices,
        sidecar=SwitchLinearSidecar(scale_delta=mx.zeros(layer.scales.shape, dtype=mx.float32)),
    )
    mx.eval(surrogate, runtime, zero_sidecar)
    elapsed = time.perf_counter() - start

    surrogate_np = np.array(surrogate)
    runtime_np = np.array(runtime)
    zero_sidecar_np = np.array(zero_sidecar)
    surrogate_reference = _error_stats(surrogate_np, reference)
    runtime_reference = _error_stats(runtime_np, reference)
    sidecar_surrogate = _error_stats(zero_sidecar_np, surrogate_np)
    passed = (
        _within_tolerance(surrogate_np, reference, atol=1e-5, rtol=1e-4)
        and _within_tolerance(runtime_np, reference, atol=args.atol, rtol=args.rtol)
        and _within_tolerance(zero_sidecar_np, surrogate_np, atol=1e-6, rtol=0.0)
    )
    record = {
        "schema_version": 1,
        "record_type": "glm45_air_mlx_surrogate_parity_probe",
        "artifact_dir": str(artifact_dir),
        "shard": str(shard),
        "layer": args.layer,
        "projection": args.projection,
        "expert": args.expert,
        "tokens": args.tokens,
        "elapsed_seconds": elapsed,
        "surrogate_vs_decoded_reference": surrogate_reference,
        "runtime_vs_decoded_reference": runtime_reference,
        "zero_sidecar_vs_surrogate": sidecar_surrogate,
        "surrogate_reference_atol": 1e-5,
        "surrogate_reference_rtol": 1e-4,
        "runtime_atol": args.atol,
        "runtime_rtol": args.rtol,
        "passed": passed,
    }
    append_jsonl(args.append_jsonl, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
