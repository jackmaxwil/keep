from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_vq.codebook.e8 import e8p_packed_abs_grid
from mlx_vq.kernels import nax
from mlx_vq.kernels.e8p_rhs_layout import decode_e8p_rhs_tile, pack_e8p_rhs_tiles
import mlx_vq.ops.vq_switch as vq_switch


def _cosine_and_max_abs(observed: mx.array, expected: mx.array) -> tuple[float, float]:
    observed_np = np.array(observed, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    dot = float(np.sum(observed_np * expected_np))
    norm = float(
        np.sqrt(np.sum(observed_np * observed_np) * np.sum(expected_np * expected_np))
    )
    cosine = dot / norm if norm else 1.0
    max_abs = float(np.max(np.abs(observed_np - expected_np))) if observed_np.size else 0.0
    return cosine, max_abs


def _token_route_output_stripe_descriptors(
    *, route_tokens: np.ndarray, tokens: int, max_routes_per_token: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if max_routes_per_token <= 0:
        raise ValueError("max_routes_per_token must be positive")

    buckets: list[list[int]] = [[] for _ in range(tokens)]
    for route, token in enumerate(route_tokens):
        token_i = int(token)
        if token_i < 0 or token_i >= tokens:
            raise ValueError("route token id is out of range")
        buckets[token_i].append(route)

    offsets: list[int] = []
    counts: list[int] = []
    route_ids: list[int] = []
    for bucket in buckets:
        if len(bucket) > max_routes_per_token:
            raise ValueError("token-route bucket exceeds native route-slot capacity")
        offsets.append(len(route_ids))
        route_ids.extend(bucket)
        counts.append(len(bucket))

    return (
        np.array(offsets, dtype=np.int32),
        np.array(counts, dtype=np.int32),
        np.array(route_ids, dtype=np.int32),
    )


def build_report(
    *,
    experts: int,
    tokens: int,
    routes: int,
    output_dims: int,
    input_dims: int,
    group_size: int,
    bn: int,
    bk: int,
    max_routes_per_token: int,
    seed: int,
    min_cosine: float,
    max_abs_diff: float,
) -> dict[str, Any]:
    if not nax.is_available():
        raise RuntimeError("native VQ NAX extension is not built")
    if input_dims % 8 != 0:
        raise ValueError("input_dims must be divisible by 8 for E8P codewords")
    if input_dims % group_size != 0:
        raise ValueError("input_dims must be divisible by group_size")

    rng = np.random.default_rng(seed)
    x_np = rng.normal(size=(tokens, input_dims)).astype(np.float16)
    codes_np = rng.integers(
        0,
        np.iinfo(np.uint16).max,
        size=(experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales_np = rng.uniform(
        0.01,
        0.08,
        size=(experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    sorted_experts_np = np.sort(
        rng.integers(0, experts, size=(routes,), dtype=np.int32),
        kind="stable",
    )
    route_tokens_np = (np.arange(routes, dtype=np.int32) % tokens).astype(np.int32)
    sorted_x_np = x_np[route_tokens_np]
    packed = pack_e8p_rhs_tiles(
        codes_np,
        scales_np,
        group_size=group_size,
        bn=bn,
        bk=bk,
    )
    codebook_np = e8p_packed_abs_grid()
    tile_experts, tile_offsets, tile_counts = vq_switch._expert_block_tile_descriptors(
        mx.array(sorted_experts_np),
        num_experts=experts,
        route_tile=max_routes_per_token,
    )
    (
        token_route_output_stripe_offsets_np,
        token_route_output_stripe_counts_np,
        token_route_output_stripe_route_slot_ids_np,
    ) = _token_route_output_stripe_descriptors(
        route_tokens=route_tokens_np,
        tokens=tokens,
        max_routes_per_token=max_routes_per_token,
    )

    observed = nax.nax_e8p_token_route_output_stripe_pipeline_rhs_sorted_matmul(
        mx.array(sorted_x_np, dtype=mx.float16),
        mx.array(packed.code_tiles),
        mx.array(packed.scale_tiles),
        mx.array(packed.scale_group_indices, dtype=mx.int32),
        mx.array(packed.codeword_scale_slots, dtype=mx.int32),
        tile_experts,
        tile_offsets,
        tile_counts,
        mx.array(token_route_output_stripe_offsets_np, dtype=mx.int32),
        mx.array(token_route_output_stripe_counts_np, dtype=mx.int32),
        mx.array(token_route_output_stripe_route_slot_ids_np, dtype=mx.int32),
        mx.array(codebook_np, dtype=mx.uint32),
        output_dims=output_dims,
    )

    expert_weights: list[np.ndarray] = []
    for expert in range(experts):
        decoded_tiles: list[np.ndarray] = []
        for n_tile in range(packed.layout.n_tiles):
            k_tiles = [
                decode_e8p_rhs_tile(
                    packed,
                    expert=expert,
                    n_tile=n_tile,
                    k_block=k_block,
                    codebook=codebook_np,
                )
                for k_block in range(packed.layout.k_blocks)
            ]
            decoded_tiles.append(np.concatenate(k_tiles, axis=1))
        expert_weights.append(np.concatenate(decoded_tiles, axis=0))

    expected_np = np.empty((routes, output_dims), dtype=np.float32)
    for route, expert in enumerate(sorted_experts_np):
        expected_np[route] = sorted_x_np[route].astype(np.float32) @ expert_weights[
            int(expert)
        ].T.astype(np.float32)
    expected = mx.array(expected_np, dtype=mx.float16)
    mx.eval(observed, expected)
    cosine, max_abs = _cosine_and_max_abs(observed, expected)
    parity_pass = bool(cosine >= min_cosine and max_abs <= max_abs_diff)

    return {
        "schema_version": 1,
        "record_type": (
            "glm45_air_e8p_token_route_output_stripe_pipeline_native_parity"
        ),
        "peer2_used": False,
        "rdma_jaccl_touched": False,
        "speed_claim": False,
        "native_parity_claim": parity_pass,
        "decision": (
            "token_route_output_stripe_pipeline_native_parity_pass"
            if parity_pass
            else "token_route_output_stripe_pipeline_native_parity_failed"
        ),
        "passes_native_parity": parity_pass,
        "comparison": (
            "token_route_output_stripe_pipeline_native_vs_decoded_e8p_rhs_oracle"
        ),
        "cosine": cosine,
        "max_abs_diff": max_abs,
        "min_cosine": min_cosine,
        "max_abs_diff_threshold": max_abs_diff,
        "shape": {
            "experts": experts,
            "tokens": tokens,
            "routes": routes,
            "output_dims": output_dims,
            "input_dims": input_dims,
            "group_size": group_size,
            "bn": bn,
            "bk": bk,
            "max_routes_per_token": max_routes_per_token,
            "n_tiles": packed.layout.n_tiles,
            "k_blocks": packed.layout.k_blocks,
            "codeword_stages_per_bk": packed.layout.codewords_per_bk,
            "token_route_output_stripe_count": int(
                token_route_output_stripe_offsets_np.size
            ),
            "max_routes_in_token": int(token_route_output_stripe_counts_np.max())
            if token_route_output_stripe_counts_np.size
            else 0,
        },
        "contract": {
            "storage_constraint": "compressed_e8p_token_route_output_stripe_pipelines",
            "dispatch_grid": "tokens_x_route_slots_x_output_stripes_x_kblock_stages",
            "preserves_token_axis": True,
            "preserves_route_slot_axis": True,
            "preserves_output_stripe_axis": True,
            "preserves_kblock_stage_axis": True,
            "streams_compressed_codeword_tiles": True,
            "streams_kblock_stages_inside_token_route_output_stripes": True,
            "accumulates_full_output_without_route_expansion": True,
            "expands_route_microtiles": False,
            "materializes_output_tile_local_full_lut": False,
            "materializes_decoded_dense_rhs": False,
        },
        "next_track_b_hypothesis": (
            "prove_air_down_group_size_352_token_route_output_stripe_pipeline_artifact_parity"
            if parity_pass
            else "fix_token_route_output_stripe_pipeline_native_parity"
        ),
        "rejected_next_steps": [
            "do_not_claim_speed_from_native_parity",
            "do_not_route_resident_auto_before_artifact_speed_gate",
            "do_not_run_q2_speed_packet_before_air_artifact_parity",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prove native token-route output-stripe pipeline parity against "
            "decoded E8P RHS."
        )
    )
    parser.add_argument("--experts", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=13)
    parser.add_argument("--routes", type=int, default=41)
    parser.add_argument("--output-dims", type=int, default=70)
    parser.add_argument("--input-dims", type=int, default=704)
    parser.add_argument("--group-size", type=int, default=352)
    parser.add_argument("--bn", type=int, default=64)
    parser.add_argument("--bk", type=int, default=64)
    parser.add_argument("--max-routes-per-token", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--min-cosine", type=float, default=0.999999)
    parser.add_argument("--max-abs-diff", type=float, default=6e-3)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(
        experts=args.experts,
        tokens=args.tokens,
        routes=args.routes,
        output_dims=args.output_dims,
        input_dims=args.input_dims,
        group_size=args.group_size,
        bn=args.bn,
        bk=args.bk,
        max_routes_per_token=args.max_routes_per_token,
        seed=args.seed,
        min_cosine=args.min_cosine,
        max_abs_diff=args.max_abs_diff,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
