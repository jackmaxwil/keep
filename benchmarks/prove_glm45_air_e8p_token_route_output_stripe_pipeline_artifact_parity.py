from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

from mlx_vq.benchmark.projection_kernels import (
    build_projection_fixture_from_vq_artifact,
    run_loaded_projection_variant,
)

ArtifactProjection = Literal["gate_proj", "up_proj", "down_proj"]

DEFAULT_ARTIFACT_DIR = Path("artifacts/glm-4.5-air-vq2-e8p-rtn-uniform-parallel8")
TOKEN_ROUTE_OUTPUT_STRIPE_PIPELINE_VARIANT = (
    "nax_e8p_token_route_output_stripe_pipeline_rhs_sorted_raw"
)


def _parse_projection(value: str) -> ArtifactProjection:
    if value not in {"gate_proj", "up_proj", "down_proj"}:
        raise argparse.ArgumentTypeError(
            "projection must be gate_proj, up_proj, or down_proj"
        )
    return value  # type: ignore[return-value]


def run_token_route_output_stripe_pipeline_artifact_parity(
    *,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    layer_index: int = 1,
    projection: ArtifactProjection = "down_proj",
    tokens: int = 2,
    top_k: int = 1,
    seed: int = 20261008,
    mlx_group_size: int = 128,
    min_cosine: float = 0.999999,
    max_abs_diff: float = 6e-3,
) -> dict[str, Any]:
    artifact_root = Path(artifact_dir)
    fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
        artifact_dir=artifact_root,
        layer_index=layer_index,
        artifact_projection=projection,
        tokens=tokens,
        top_k=top_k,
        mlx_group_size=mlx_group_size,
        seed=seed,
    )
    if layer.code_bits != 16:
        raise ValueError(
            "token-route output-stripe pipeline artifact parity requires a 16-bit E8P artifact"
        )
    if projection == "down_proj" and layer.group_size != 352:
        raise ValueError(
            "Air down token-route output-stripe pipeline artifact parity requires group_size=352"
        )

    record = run_loaded_projection_variant(
        fixture,
        layer,
        variant=TOKEN_ROUTE_OUTPUT_STRIPE_PIPELINE_VARIANT,
        iterations=1,
        warmup=0,
        artifact_metadata=metadata,
    )
    cosine = float(record.get("artifact_reference_cosine", 0.0))
    observed_max_abs = float(record.get("artifact_reference_max_abs_diff", float("inf")))
    finite_output = bool(record.get("finite_output"))
    memory_clean = (
        int(record.get("pageouts_delta", 0)) == 0
        and int(record.get("swapouts_delta", 0)) == 0
    )
    passes_artifact_parity = bool(
        finite_output
        and memory_clean
        and cosine >= min_cosine
        and observed_max_abs <= max_abs_diff
    )
    decision = (
        "token_route_output_stripe_pipeline_air_artifact_parity_pass"
        if passes_artifact_parity
        else "token_route_output_stripe_pipeline_air_artifact_parity_failed"
    )
    max_routes_per_token = 8
    return {
        "record_type": (
            "glm45_air_e8p_token_route_output_stripe_pipeline_artifact_parity"
        ),
        "schema_version": 1,
        "decision": decision,
        "passes_artifact_parity": passes_artifact_parity,
        "artifact_parity_claim": passes_artifact_parity,
        "native_parity_claim": False,
        "speed_claim": False,
        "peer2_used": False,
        "rdma_jaccl_touched": False,
        "artifact_dir": str(artifact_root),
        "layer_index": layer_index,
        "projection": projection,
        "tokens": tokens,
        "top_k": top_k,
        "seed": seed,
        "variant": TOKEN_ROUTE_OUTPUT_STRIPE_PIPELINE_VARIANT,
        "comparison": (
            "token_route_output_stripe_pipeline_artifact_vs_decoded_fp16_sorted_direct"
        ),
        "input_dims": int(fixture.input_dims),
        "output_dims": int(fixture.output_dims),
        "experts": int(fixture.experts),
        "group_size": int(layer.group_size),
        "code_bits": int(layer.code_bits),
        "route_count": int(tokens * top_k),
        "max_routes_per_token": max_routes_per_token,
        "finite_output": finite_output,
        "memory_clean": memory_clean,
        "pageouts_delta": int(record.get("pageouts_delta", 0)),
        "swapouts_delta": int(record.get("swapouts_delta", 0)),
        "artifact_reference_variant": record.get("artifact_reference_variant"),
        "cosine": cosine,
        "max_abs_diff": observed_max_abs,
        "min_cosine": min_cosine,
        "max_abs_diff_threshold": max_abs_diff,
        "output_shape": record.get("output_shape"),
        "ms_per_iter_diagnostic": record.get("ms_per_iter"),
        "diagnostic_phase": record.get("diagnostic_phase"),
        "diagnostic_note": record.get("diagnostic_note"),
        "source_metadata": metadata,
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
            "run_token_route_output_stripe_pipeline_same_window_q2_speed_packet"
            if passes_artifact_parity
            else "fix_token_route_output_stripe_pipeline_air_artifact_parity"
        ),
        "rejected_next_steps": [
            "do_not_claim_speed_from_artifact_parity_smoke",
            "do_not_route_resident_auto_before_same_window_q2_speed_packet",
            "do_not_use_peer2_for_this_local_trackb_slice",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--layer-index", type=int, default=1)
    parser.add_argument("--projection", type=_parse_projection, default="down_proj")
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--mlx-group-size", type=int, default=128)
    parser.add_argument("--min-cosine", type=float, default=0.999999)
    parser.add_argument("--max-abs-diff", type=float, default=6e-3)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)

    report = run_token_route_output_stripe_pipeline_artifact_parity(
        artifact_dir=args.artifact_dir,
        layer_index=args.layer_index,
        projection=args.projection,
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
        args.output_json.write_text(text + "\n", encoding="utf-8")
    return 0 if report["passes_artifact_parity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
