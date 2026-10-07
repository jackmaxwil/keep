from __future__ import annotations

import argparse
import json

from mlx_vq.benchmark.glm45_air import append_jsonl
from mlx_vq.benchmark.projection_kernels import (
    build_projection_fixture_from_vq_artifact,
    build_projection_fixture,
    decode_bandwidth_floor_report,
    projection_variants_for_cli,
    run_loaded_projection_variant,
    run_projection_variant,
)


AIR_PROJECTIONS = {
    "gate_up": (4096, 1408),
    "down": (1408, 4096),
}
PUBLICATION_TOKENS = (1024, 2048, 4096)
PUBLICATION_VARIANTS = (
    "mlx_q2",
    "nax_e8_fp16_sorted_steel",
    "vq_e1",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark dense, MLX q2, and VQ routed projection kernels at GLM-4.5-Air shapes."
    )
    parser.add_argument("--projection", choices=["gate_up", "down", "all"], default="all")
    parser.add_argument(
        "--variant",
        choices=[
            "dense_bf16",
            "mlx_gather_mm_bf16",
            "mlx_q2",
            "nax_e8_fp16",
            "nax_e8_fp16_sorted_steel",
            "nax_e8_fp16_sorted_steel_raw",
            "nax_e8p_fp16_sorted_steel",
            "nax_e8p_fp16_sorted_steel_raw",
            "nax_e8p_fp16_sorted_direct_reduce_raw",
            "nax_e8p_fp16_sorted_inline_b_raw",
            "nax_e8p_packed_rhs_sorted_tiled_raw",
            "nax_e8p_route_slot_codeword_stream_rhs_sorted_raw",
            "nax_e8p_route_slot_mma_codeword_tile_rhs_sorted_raw",
            "nax_e8p_active_route_tile_codeword_outer_product_rhs_sorted_raw",
            "nax_e8p_split_byte_rhs_sorted_tiled_raw",
            "nax_e8p_split_byte_factor_reuse_rhs_sorted_tiled_raw",
            "nax_e8p_split_byte_factor_reuse_rhs_sorted_native_raw",
            "nax_e8p_component_stream_rhs_sorted_scalar_raw",
            "nax_e8p_component_stream_rhs_sorted_partial_raw",
            "nax_e8p_component_stream_rhs_sorted_tensorops_raw",
            "nax_e8p_component_stream_rhs_sorted_shared_decode_raw",
            "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_native_raw",
            "nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_raw",
            "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_raw",
            "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_raw",
            "nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_raw",
            "nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_raw",
            "nax_e8p_sign_plane_abs_index_rhs_sorted_native_raw",
            "nax_e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_raw",
            "nax_e8p_sign_nibble_micro_lut_rhs_sorted_native_raw",
            "nax_e8p_packed_rhs_sorted_tiled_m128_raw",
            "nax_e8p_packed_rhs_sorted_tiled_k128_raw",
            "nax_e8p_predecoded_fp16_raw",
            "nax_e8p_fp16_sorted_steel_gs352_raw",
            "nax_e8p_fp16_sorted_steel_lut_raw",
            "nax_e8p_fp16_sorted_steel_tgcb_raw",
            "nax_e8p_fp16_sorted_steel_tgscale_raw",
            "nax_e8p_fp16_sorted_steel_tgcb_tgscale_raw",
            "nax_e8p_fp16_sorted_steel_tgcb_hoist_raw",
            "nax_e8p_fp16_sorted_steel_bk128_raw",
            "nax_e8p_fp16_sorted_steel_m128n32_raw",
            "nax_e8p_fp16_sorted_steel_m64n128_raw",
            "nax_e8p_fp16_sorted_steel_m32n64_raw",
            "nax_e8p_fp16_sorted_steel_m64n64t64_raw",
            "nax_e8p_fp16_sorted_steel_m32n64t128_raw",
            "nax_e8p_fp16_sorted_steel_m32n128_raw",
            "nax_e8_fp16_steel",
            "nax_e8_int8",
            "nax_predecoded_fp16",
            "vq_e1",
            "vq_decode_to_scratch_gather_mm",
            "vq_decode_direct_candidates",
            "m1_decode_gate",
            "nax_audit",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--tokens", type=int, action="append", default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--mlx-group-size", type=int, default=128)
    parser.add_argument("--vq-preferred-group-size", type=int, default=512)
    parser.add_argument("--vq-code-bits", type=int, choices=[8, 16], default=8)
    parser.add_argument(
        "--artifact-dir",
        help="Load prequantized projection shards from this artifact directory instead of synthetic RTN fixtures.",
    )
    parser.add_argument("--artifact-layer", type=int, default=45)
    parser.add_argument(
        "--artifact-projection",
        choices=["gate_proj", "up_proj", "down_proj", "all"],
        default="all",
    )
    parser.add_argument(
        "--artifact-block-module",
        choices=["mlp", "ffn"],
        default="mlp",
        help=(
            "decoder-block attribute owning the routed experts: 'mlp' for the "
            "GLM families, 'ffn' for DeepSeek-V4-Flash"
        ),
    )
    parser.add_argument("--decompose-vq", action="store_true")
    parser.add_argument("--decode-measured-gb-s", type=float)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--append-jsonl")
    parser.add_argument(
        "--skip-artifact-reference",
        action="store_true",
        help=(
            "Skip untimed decoded-reference checks for artifact component-stream rows. "
            "Use only after a separate parity row exists."
        ),
    )
    parser.add_argument(
        "--publication-matrix",
        action="store_true",
        help="Run the publication projection matrix at M=1024/2048/4096.",
    )
    parser.add_argument(
        "--include-int8-diagnostic",
        action="store_true",
        help="Include the diagnostic INT8 NAX projection rows in publication-matrix mode.",
    )
    args = parser.parse_args()

    projections = list(AIR_PROJECTIONS) if args.projection == "all" else [args.projection]
    if args.publication_matrix:
        variants = list(PUBLICATION_VARIANTS)
        if args.include_int8_diagnostic:
            variants.append("nax_e8_int8")
        token_counts = list(PUBLICATION_TOKENS)
    else:
        variants = projection_variants_for_cli(args.variant, decompose_vq=args.decompose_vq)
        token_counts = args.tokens or [1, 1024, 4096]

    records = []
    if args.artifact_dir:
        if args.publication_matrix:
            raise SystemExit("--publication-matrix cannot be combined with --artifact-dir")
        artifact_projections = (
            ["gate_proj", "up_proj", "down_proj"]
            if args.artifact_projection == "all"
            else [args.artifact_projection]
        )
        for artifact_projection in artifact_projections:
            for tokens in token_counts:
                fixture, layer, artifact_metadata = build_projection_fixture_from_vq_artifact(
                    artifact_dir=args.artifact_dir,
                    layer_index=args.artifact_layer,
                    artifact_projection=artifact_projection,
                    tokens=tokens,
                    top_k=args.top_k,
                    mlx_group_size=args.mlx_group_size,
                    block_module=args.artifact_block_module,
                    seed=args.seed
                    + tokens
                    + {
                        "gate_proj": 0,
                        "up_proj": 20_000,
                        "down_proj": 10_000,
                    }[artifact_projection],
                )
                for variant in variants:
                    record = run_loaded_projection_variant(
                        fixture,
                        layer,
                        variant=variant,
                        iterations=args.iterations,
                        warmup=args.warmup,
                        artifact_metadata=artifact_metadata,
                        artifact_reference_check=not args.skip_artifact_reference,
                    )
                    records.append(record)
                    if args.append_jsonl:
                        append_jsonl(args.append_jsonl, record)
                    print(json.dumps(record, indent=2, sort_keys=True))
        print(json.dumps({"records": len(records)}, sort_keys=True))
        return

    for projection in projections:
        input_dims, output_dims = AIR_PROJECTIONS[projection]
        for tokens in token_counts:
            fixture = build_projection_fixture(
                projection=projection,
                tokens=tokens,
                top_k=args.top_k,
                experts=args.experts,
                input_dims=input_dims,
                output_dims=output_dims,
                mlx_group_size=args.mlx_group_size,
                vq_preferred_group_size=args.vq_preferred_group_size,
                vq_code_bits=args.vq_code_bits,
                seed=args.seed + tokens + (0 if projection == "gate_up" else 10_000),
            )
            for variant in variants:
                if variant == "vq_decode_direct_candidates" and tokens != 1:
                    continue
                record = run_projection_variant(
                    fixture,
                    variant=variant,
                    iterations=args.iterations,
                    warmup=args.warmup,
                )
                if args.decode_measured_gb_s is not None and str(variant).startswith("vq_") and tokens == 1:
                    record["decode_bandwidth_floor"] = decode_bandwidth_floor_report(
                        input_dims=input_dims,
                        output_dims=output_dims,
                        top_k=args.top_k,
                        group_size=fixture.vq_group_size,
                        measured_gb_s=args.decode_measured_gb_s,
                    )
                if args.publication_matrix:
                    record["record_type"] = "publication_projection"
                    record["publication_matrix"] = True
                records.append(record)
                if args.append_jsonl:
                    append_jsonl(args.append_jsonl, record)
                print(json.dumps(record, indent=2, sort_keys=True))

    print(json.dumps({"records": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
