from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import mlx.core as mx
import numpy as np

from mlx_vq.benchmark.projection_kernels import (
    _prepare_e8p_expert_kblock_factor_reuse_rhs_tiles,
    _prepare_sorted_steel_inputs,
    build_projection_fixture_from_vq_artifact,
)
from mlx_vq.kernels import nax

ArtifactProjection = Literal["gate_proj", "up_proj", "down_proj"]


def _cosine_and_max_abs(observed: mx.array, expected: mx.array) -> tuple[float, float]:
    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np)))
    cosine = dot / norm if norm else 1.0
    max_abs = float(np.max(np.abs(observed_np - expected_np))) if observed_np.size else 0.0
    return cosine, max_abs


def _check_projection(
    *,
    artifact_dir: Path,
    layer_index: int,
    projection: ArtifactProjection,
    tokens: int,
    top_k: int,
    seed: int,
    mlx_group_size: int,
    min_cosine: float,
    max_abs_diff: float,
) -> dict[str, Any]:
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact_dir,
        layer_index=layer_index,
        artifact_projection=projection,
        tokens=tokens,
        top_k=top_k,
        mlx_group_size=mlx_group_size,
        seed=seed,
    )
    if layer.code_bits != 16:
        raise ValueError("expert-kblock v2 native smoke requires a 16-bit E8P artifact")

    sorted_inputs = _prepare_sorted_steel_inputs(fixture)
    packed_rhs = _prepare_e8p_expert_kblock_factor_reuse_rhs_tiles(layer)
    (
        sign_byte_lut,
        sign_byte_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
    ) = packed_rhs

    observed = nax.nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul(
        sorted_inputs.sorted_x,
        sign_byte_lut,
        sign_byte_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        output_dims=fixture.output_dims,
    )
    expected = nax.nax_e8p_expert_kblock_factor_reuse_rhs_sorted_native_matmul(
        sorted_inputs.sorted_x,
        sign_byte_lut,
        sign_byte_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        *sorted_inputs.tile_descriptors,
        layer.codebook,
        output_dims=fixture.output_dims,
    )
    mx.eval(observed, expected)
    cosine, max_abs = _cosine_and_max_abs(observed, expected)
    parity_pass = bool(cosine >= min_cosine and max_abs <= max_abs_diff)
    return {
        **metadata,
        "projection": projection,
        "tokens": tokens,
        "top_k": top_k,
        "route_count": int(tokens * top_k),
        "experts": int(fixture.experts),
        "input_dims": int(fixture.input_dims),
        "output_dims": int(fixture.output_dims),
        "group_size": int(layer.group_size),
        "code_bits": int(layer.code_bits),
        "v2_dispatch_contract": "experts_x_k_blocks_x_route_tiles",
        "comparison": "v2_partial_tensorops_vs_scalar_native",
        "cosine": cosine,
        "max_abs_diff": max_abs,
        "min_cosine": min_cosine,
        "max_abs_diff_threshold": max_abs_diff,
        "parity_pass": parity_pass,
    }


def run_expert_kblock_v2_native_smoke(
    *,
    artifact_dir: str | Path,
    layer_index: int,
    projections: tuple[ArtifactProjection, ...] = ("gate_proj", "up_proj", "down_proj"),
    tokens: int = 17,
    top_k: int = 8,
    seed: int = 20260714,
    mlx_group_size: int = 64,
    min_cosine: float = 0.999999,
    max_abs_diff: float = 8e-3,
) -> dict[str, Any]:
    artifact_root = Path(artifact_dir)
    checks = [
        _check_projection(
            artifact_dir=artifact_root,
            layer_index=layer_index,
            projection=projection,
            tokens=tokens,
            top_k=top_k,
            seed=seed,
            mlx_group_size=mlx_group_size,
            min_cosine=min_cosine,
            max_abs_diff=max_abs_diff,
        )
        for projection in projections
    ]
    all_pass = all(bool(check["parity_pass"]) for check in checks)
    return {
        "record_type": "glm45_air_e8p_expert_kblock_v2_native_smoke",
        "schema_version": 1,
        "artifact_dir": str(artifact_root),
        "layer_index": layer_index,
        "projections": list(projections),
        "checks": checks,
        "all_v2_checks_pass": all_pass,
        "decision": "parity_pass_not_tensorops_speed" if all_pass else "parity_failed",
        "speed_claim": False,
        "tensorops_speed_claim": False,
        "note": (
            "This smoke proves native callable parity for the v2 partial TensorOps body "
            "and dispatch contract only; source guardrails must pass and same-window q2 "
            "speed evidence must exist before any benchmark promotion."
        ),
    }


def _parse_projection(value: str) -> ArtifactProjection:
    if value not in {"gate_proj", "up_proj", "down_proj"}:
        raise argparse.ArgumentTypeError("projection must be gate_proj, up_proj, or down_proj")
    return value  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--layer-index", type=int, default=1)
    parser.add_argument(
        "--projection",
        action="append",
        type=_parse_projection,
        help="projection to smoke; may be repeated; defaults to gate/up/down",
    )
    parser.add_argument("--tokens", type=int, default=17)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--mlx-group-size", type=int, default=64)
    parser.add_argument("--min-cosine", type=float, default=0.999999)
    parser.add_argument("--max-abs-diff", type=float, default=8e-3)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-jsonl", type=Path)
    args = parser.parse_args(argv)

    projections: tuple[ArtifactProjection, ...] = tuple(args.projection or ("gate_proj", "up_proj", "down_proj"))
    report = run_expert_kblock_v2_native_smoke(
        artifact_dir=args.artifact_dir,
        layer_index=args.layer_index,
        projections=projections,
        tokens=args.tokens,
        top_k=args.top_k,
        seed=args.seed,
        mlx_group_size=args.mlx_group_size,
        min_cosine=args.min_cosine,
        max_abs_diff=args.max_abs_diff,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w") as fh:
            for check in report["checks"]:
                fh.write(json.dumps(check, sort_keys=True) + "\n")
    return 0 if report["all_v2_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
