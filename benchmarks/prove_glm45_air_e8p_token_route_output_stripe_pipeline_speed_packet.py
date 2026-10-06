from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import mlx.core as mx
import numpy as np

from mlx_vq.benchmark.glm45_air import append_jsonl
from mlx_vq.benchmark.projection_kernels import (
    ProjectionFixture,
    build_projection_fixture_from_vq_artifact,
    run_loaded_projection_variant,
    run_projection_variant,
)

ArtifactProjection = Literal["gate_proj", "up_proj", "down_proj"]

DEFAULT_ARTIFACT_DIR = Path("artifacts/glm-4.5-air-vq2-e8p-rtn-uniform-parallel8")
DEFAULT_CANDIDATE_JSONL = Path(
    "artifacts/benchmarks/glm45-air-e8p-token-route-output-stripe-pipeline-m1024-candidate-20260704.jsonl"
)
DEFAULT_Q2_JSONL = Path(
    "artifacts/benchmarks/glm45-air-e8p-token-route-output-stripe-pipeline-m1024-q2-20260704.jsonl"
)
TOKEN_ROUTE_OUTPUT_STRIPE_PIPELINE_VARIANT = (
    "nax_e8p_token_route_output_stripe_pipeline_rhs_sorted_raw"
)


def _parse_projection(value: str) -> ArtifactProjection:
    if value not in {"gate_proj", "up_proj", "down_proj"}:
        raise argparse.ArgumentTypeError(
            "projection must be gate_proj, up_proj, or down_proj"
        )
    return value  # type: ignore[return-value]


def _q2_fixture_from_candidate(
    fixture: ProjectionFixture,
    *,
    seed: int,
) -> ProjectionFixture:
    rng = np.random.default_rng(seed)
    dense_weight = mx.array(
        rng.normal(
            scale=0.02,
            size=(fixture.experts, fixture.output_dims, fixture.input_dims),
        ).astype(np.float32)
    ).astype(mx.bfloat16)
    mx.eval(dense_weight)
    return ProjectionFixture(
        projection=fixture.projection,
        tokens=fixture.tokens,
        top_k=fixture.top_k,
        experts=fixture.experts,
        input_dims=fixture.input_dims,
        output_dims=fixture.output_dims,
        mlx_group_size=fixture.mlx_group_size,
        vq_group_size=fixture.vq_group_size,
        vq_code_bits=fixture.vq_code_bits,
        dense_weight=dense_weight,
        x=fixture.x,
        indices=fixture.indices,
    )


def _clean(row: dict[str, Any]) -> bool:
    return (
        bool(row.get("finite_output"))
        and int(row.get("pageouts_delta") or 0) == 0
        and int(row.get("swapouts_delta") or 0) == 0
    )


def run_token_route_output_stripe_pipeline_speed_packet(
    *,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    layer_index: int = 1,
    projection: ArtifactProjection = "down_proj",
    tokens: tuple[int, ...] = (1024,),
    top_k: int = 8,
    seed: int = 20261009,
    mlx_group_size: int = 128,
    candidate_iterations: int = 1,
    candidate_warmup: int = 0,
    q2_iterations: int = 3,
    q2_warmup: int = 1,
    parity_ratio: float = 1.25,
    lane_s_ratio: float = 1.15,
    candidate_timeout_observed_seconds: float | None = None,
    candidate_jsonl: str | Path | None = DEFAULT_CANDIDATE_JSONL,
    q2_jsonl: str | Path | None = DEFAULT_Q2_JSONL,
) -> dict[str, Any]:
    artifact_root = Path(artifact_dir)
    comparisons: list[dict[str, Any]] = []
    if candidate_jsonl is not None:
        Path(candidate_jsonl).unlink(missing_ok=True)
    if q2_jsonl is not None:
        Path(q2_jsonl).unlink(missing_ok=True)

    for token_count in tokens:
        fixture, layer, metadata = build_projection_fixture_from_vq_artifact(
            artifact_dir=artifact_root,
            layer_index=layer_index,
            artifact_projection=projection,
            tokens=token_count,
            top_k=top_k,
            mlx_group_size=mlx_group_size,
            seed=seed + token_count,
        )
        if layer.code_bits != 16:
            raise ValueError(
                "token-route output-stripe speed packet requires a 16-bit E8P artifact"
            )
        if projection == "down_proj" and layer.group_size != 352:
            raise ValueError(
                "Air down token-route output-stripe speed packet requires group_size=352"
            )

        if candidate_timeout_observed_seconds is None:
            candidate = run_loaded_projection_variant(
                fixture,
                layer,
                variant=TOKEN_ROUTE_OUTPUT_STRIPE_PIPELINE_VARIANT,
                iterations=candidate_iterations,
                warmup=candidate_warmup,
                artifact_metadata=metadata,
                artifact_reference_check=False,
            )
        else:
            candidate = {
                **metadata,
                "fixture_source": "vq_artifact",
                "projection": fixture.projection,
                "artifact_projection": projection,
                "variant": TOKEN_ROUTE_OUTPUT_STRIPE_PIPELINE_VARIANT,
                "timed_out": True,
                "timeout_observed_seconds": float(candidate_timeout_observed_seconds),
                "ms_per_iter": None,
                "finite_output": False,
                "output_shape": None,
                "pageouts_delta": 0,
                "swapouts_delta": 0,
            }
        q2 = run_projection_variant(
            _q2_fixture_from_candidate(fixture, seed=seed + token_count + 1_000_000),
            variant="mlx_q2",
            iterations=q2_iterations,
            warmup=q2_warmup,
        )
        candidate["record_type"] = "token_route_output_stripe_pipeline_candidate_speed_row"
        candidate["same_window_q2_speed_packet"] = True
        candidate["speed_claim"] = False
        candidate["peer2_used"] = False
        candidate["rdma_jaccl_touched"] = False
        q2["record_type"] = "token_route_output_stripe_pipeline_q2_control_speed_row"
        q2["same_window_q2_speed_packet"] = True
        q2["peer2_used"] = False
        q2["rdma_jaccl_touched"] = False
        if candidate_jsonl is not None:
            append_jsonl(str(candidate_jsonl), candidate)
        if q2_jsonl is not None:
            append_jsonl(str(q2_jsonl), q2)

        ratio = (
            float(candidate["timeout_observed_seconds"]) * 1000.0
            / float(q2["ms_per_iter"])
            if candidate.get("timed_out")
            else float(candidate["ms_per_iter"]) / float(q2["ms_per_iter"])
        )
        comparisons.append(
            {
                "tokens": token_count,
                "projection": fixture.projection,
                "artifact_projection": projection,
                "candidate_variant": candidate["variant"],
                "q2_variant": q2["variant"],
                "candidate_ms_per_iter": (
                    None
                    if candidate.get("timed_out")
                    else float(candidate["ms_per_iter"])
                ),
                "q2_ms_per_iter": float(q2["ms_per_iter"]),
                "ratio_to_q2": ratio,
                "candidate_timed_out": bool(candidate.get("timed_out")),
                "ratio_is_timeout_lower_bound": bool(candidate.get("timed_out")),
                "timeout_observed_seconds": candidate.get("timeout_observed_seconds"),
                "parity_ratio": parity_ratio,
                "lane_s_ratio": lane_s_ratio,
                "parity_pass": ratio <= parity_ratio,
                "lane_s_pass": ratio <= lane_s_ratio,
                "candidate_memory_clean": _clean(candidate),
                "q2_memory_clean": _clean(q2),
                "candidate_output_shape": candidate.get("output_shape"),
                "q2_output_shape": q2.get("output_shape"),
            }
        )

    all_clean = all(
        row["candidate_memory_clean"] and row["q2_memory_clean"]
        for row in comparisons
    )
    all_parity_pass = bool(comparisons) and all(
        row["parity_pass"] for row in comparisons
    )
    all_lane_s_pass = bool(comparisons) and all(
        row["lane_s_pass"] for row in comparisons
    )
    worst = (
        max(comparisons, key=lambda row: float(row["ratio_to_q2"]))
        if comparisons
        else None
    )
    closest = (
        min(comparisons, key=lambda row: float(row["ratio_to_q2"]))
        if comparisons
        else None
    )
    decision = (
        "token_route_output_stripe_pipeline_speed_packet_lane_s_pass"
        if all_clean and all_lane_s_pass
        else "token_route_output_stripe_pipeline_speed_packet_parity_pass"
        if all_clean and all_parity_pass
        else "reject_token_route_output_stripe_pipeline_speed_path"
    )
    return {
        "record_type": "glm45_air_e8p_token_route_output_stripe_pipeline_speed_packet",
        "schema_version": 1,
        "decision": decision,
        "same_window_q2_speed_packet": True,
        "speed_claim": False,
        "lane_s_claim": False,
        "resident_auto_claim": False,
        "peer2_used": False,
        "rdma_jaccl_touched": False,
        "artifact_dir": str(artifact_root),
        "layer_index": layer_index,
        "projection": projection,
        "tokens": list(tokens),
        "top_k": top_k,
        "candidate_iterations": candidate_iterations,
        "candidate_warmup": candidate_warmup,
        "q2_iterations": q2_iterations,
        "q2_warmup": q2_warmup,
        "parity_ratio": parity_ratio,
        "lane_s_ratio": lane_s_ratio,
        "candidate_timeout_observed_seconds": candidate_timeout_observed_seconds,
        "candidate_jsonl": str(candidate_jsonl) if candidate_jsonl is not None else None,
        "q2_jsonl": str(q2_jsonl) if q2_jsonl is not None else None,
        "comparison_count": len(comparisons),
        "all_memory_clean": all_clean,
        "all_parity_pass": all_parity_pass,
        "all_lane_s_pass": all_lane_s_pass,
        "closest_ratio": closest,
        "worst_ratio": worst,
        "comparisons": comparisons,
        "next_track_b_hypothesis": (
            "route_resident_auto_after_token_route_output_stripe_pipeline_lane_s_gate"
            if all_clean and all_lane_s_pass
            else "run_missing_token_route_output_stripe_pipeline_speed_shapes"
            if all_clean and all_parity_pass
            else "change_token_route_output_stripe_pipeline_layout_or_kernel_family"
        ),
        "rejected_next_steps": [
            "do_not_claim_lane_s_from_rejected_speed_packet",
            "do_not_route_resident_auto_from_rejected_speed_packet",
            "do_not_use_peer2_for_this_local_trackb_slice",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--layer-index", type=int, default=1)
    parser.add_argument("--projection", type=_parse_projection, default="down_proj")
    parser.add_argument("--tokens", type=int, action="append", default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--mlx-group-size", type=int, default=128)
    parser.add_argument("--candidate-iterations", type=int, default=1)
    parser.add_argument("--candidate-warmup", type=int, default=0)
    parser.add_argument("--q2-iterations", type=int, default=3)
    parser.add_argument("--q2-warmup", type=int, default=1)
    parser.add_argument("--parity-ratio", type=float, default=1.25)
    parser.add_argument("--lane-s-ratio", type=float, default=1.15)
    parser.add_argument("--candidate-timeout-observed-seconds", type=float)
    parser.add_argument("--candidate-jsonl", type=Path, default=DEFAULT_CANDIDATE_JSONL)
    parser.add_argument("--q2-jsonl", type=Path, default=DEFAULT_Q2_JSONL)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)

    report = run_token_route_output_stripe_pipeline_speed_packet(
        artifact_dir=args.artifact_dir,
        layer_index=args.layer_index,
        projection=args.projection,
        tokens=tuple(args.tokens or (1024,)),
        top_k=args.top_k,
        seed=args.seed,
        mlx_group_size=args.mlx_group_size,
        candidate_iterations=args.candidate_iterations,
        candidate_warmup=args.candidate_warmup,
        q2_iterations=args.q2_iterations,
        q2_warmup=args.q2_warmup,
        parity_ratio=args.parity_ratio,
        lane_s_ratio=args.lane_s_ratio,
        candidate_timeout_observed_seconds=args.candidate_timeout_observed_seconds,
        candidate_jsonl=args.candidate_jsonl,
        q2_jsonl=args.q2_jsonl,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
