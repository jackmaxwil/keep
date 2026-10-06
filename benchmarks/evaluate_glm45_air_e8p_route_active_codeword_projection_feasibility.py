from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np

from mlx_vq.benchmark.projection_kernels import build_projection_fixture_from_vq_artifact
from mlx_vq.kernels.e8p_rhs_layout import (
    E8PNextKernelFamilyCandidate,
    evaluate_e8p_next_kernel_family_candidate,
    evaluate_e8p_route_active_codeword_projection_feasibility,
)


def build_report(
    *,
    artifact_dir: Path,
    layer_index: int,
    artifact_projection: Literal["gate_proj", "up_proj", "down_proj"],
    tokens: int,
    top_k: int,
    mlx_group_size: int,
    bn: int,
    bk: int,
    route_tile_size: int,
    unique_codeword_cap: int,
    projection_cache_byte_cap: int,
    projection_cache_dtype: str,
    seed: int,
) -> dict[str, Any]:
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact_dir,
        layer_index=layer_index,
        artifact_projection=artifact_projection,
        tokens=tokens,
        top_k=top_k,
        mlx_group_size=mlx_group_size,
        seed=seed,
    )
    route_experts = np.array(fixture.indices, dtype=np.int32).reshape(-1)
    codes = np.array(layer.codes)
    scales = np.array(layer.scales)

    feasibility = evaluate_e8p_route_active_codeword_projection_feasibility(
        codes,
        scales,
        route_experts,
        group_size=layer.group_size,
        bn=bn,
        bk=bk,
        route_tile_size=route_tile_size,
        unique_codeword_cap=unique_codeword_cap,
        projection_cache_dtype=projection_cache_dtype,
        projection_cache_byte_cap=projection_cache_byte_cap,
    )
    selector_verdict = evaluate_e8p_next_kernel_family_candidate(
        E8PNextKernelFamilyCandidate(
            target_kernel_family=feasibility.target_kernel_family,
            storage_constraint=feasibility.storage_constraint,
            dispatch_grid=feasibility.dispatch_grid,
            decode_reuse_scope="active_route_tile_codeword_dot_cache",
            tensorops_rhs_source="activation_side_active_codeword_dot_lookup_no_b_fragment",
            preserves_compressed_rhs_storage=feasibility.preserves_compressed_rhs_storage,
            preserves_codeword_scale_slots=feasibility.preserves_codeword_scale_slots,
            decoded_dense_weight_bytes=feasibility.decoded_dense_weight_bytes,
        )
    )

    return {
        "schema_version": 1,
        "record_type": "glm45_air_e8p_route_active_codeword_projection_feasibility",
        "peer2_used": False,
        "rdma_jaccl_touched": False,
        "speed_claim": False,
        "native_parity_claim": False,
        "artifact": metadata,
        "shape": {
            "projection": fixture.projection,
            "tokens": fixture.tokens,
            "top_k": fixture.top_k,
            "route_count": fixture.tokens * fixture.top_k,
            "route_tile_size": route_tile_size,
            "active_route_tile_count": feasibility.active_route_tile_count,
            "active_expert_count": feasibility.active_expert_count,
            "experts": fixture.experts,
            "input_dims": fixture.input_dims,
            "output_dims": fixture.output_dims,
            "mlx_group_size": fixture.mlx_group_size,
            "vq_group_size": fixture.vq_group_size,
            "vq_code_bits": fixture.vq_code_bits,
            "bn": bn,
            "bk": bk,
        },
        "feasibility": asdict(feasibility),
        "next_family_selector_verdict": selector_verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/glm-4.5-air-vq2-e8p-rtn-uniform-parallel8"),
    )
    parser.add_argument("--layer-index", type=int, default=45)
    parser.add_argument(
        "--artifact-projection",
        choices=("gate_proj", "up_proj", "down_proj"),
        default="down_proj",
    )
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--mlx-group-size", type=int, default=128)
    parser.add_argument("--bn", type=int, default=64)
    parser.add_argument("--bk", type=int, default=64)
    parser.add_argument("--route-tile-size", type=int, default=64)
    parser.add_argument("--unique-codeword-cap", type=int, default=4096)
    parser.add_argument("--projection-cache-byte-cap", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--projection-cache-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(
        artifact_dir=args.artifact_dir,
        layer_index=args.layer_index,
        artifact_projection=args.artifact_projection,
        tokens=args.tokens,
        top_k=args.top_k,
        mlx_group_size=args.mlx_group_size,
        bn=args.bn,
        bk=args.bk,
        route_tile_size=args.route_tile_size,
        unique_codeword_cap=args.unique_codeword_cap,
        projection_cache_byte_cap=args.projection_cache_byte_cap,
        projection_cache_dtype=args.projection_cache_dtype,
        seed=args.seed,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
