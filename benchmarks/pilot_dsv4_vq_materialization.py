#!/usr/bin/env python3
"""Wave 5 pilot: measure the DeepSeek-V4-Flash VQ fit before the 43-layer sweep.

Three subcommands, in the order the campaign needs them:

``sweep``
    Re-derive the group-size policy. Fits a stratified sample of experts from
    one or more layers across the whole (code_bits, group_size) rate ladder and
    reports imatrix-weighted reconstruction error at each point, route-count
    weighted to the layer. Optionally evaluates the RHT rotation lever.

``layer-fit``
    Fit a whole layer -- 256 experts x 3 projections -- at one policy, for
    wall-clock and peak memory. This is the term the schedule extrapolates.

``schedule``
    Turn measured layer timings into the full-materialization projection:
    hours, peak memory, artifact bytes, and whether it fits the campaign's
    local-deliverable envelope.

``monitor``
    Print the current state of a run directory (house CLI convention).

Heavy subcommands take ``.keep-heavy-job.lock`` unless ``--no-lock``.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import platform
import resource
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mlx_vq.convert.dsv4_vq_pilot import (
    PROJECTION_SHAPE,
    RATE_LADDER,
    GroupSizePolicy,
    LayerFitTiming,
    aggregate_calibration_importance,
    decode_expert,
    estimate_source_fp4_step,
    extrapolate_schedule,
    fit_projection,
    load_profile_geometry,
    policy_bpw,
    projection_importance,
    rate_bpw,
    read_layer_experts,
    route_weighted_mean,
    stratified_experts,
    write_json,
)

DEFAULT_CHECKPOINT = Path.home() / "models" / "DeepSeek-V4-Flash-0731"
DEFAULT_CALIBRATION = Path.home() / "keep-artifacts" / "dsv4-teacher-calibration"
DEFAULT_PROFILE = REPO_ROOT / "models" / "deepseek-v4-flash-0731.yaml"
HEAVY_JOB_LOCK = REPO_ROOT / ".keep-heavy-job.lock"
PROJECTIONS = ("gate", "up", "down")


@contextlib.contextmanager
def heavy_job_lock(path: Path, *, enabled: bool = True) -> Iterator[None]:
    """The shared KEEP heavy-job lock: cooperative, released on process exit."""

    if not enabled:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Open append-only so the wait does not truncate the current holder's
    # identity, then rewrite the file to name *this* holder once the lock is
    # actually ours. A plain append grows the file for the life of the machine.
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(f"{os.getpid()}\ndsv4-vq-pilot\n")
            handle.flush()
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def peak_rss_gb() -> float:
    """Peak resident set of this process, in GB (Darwin reports bytes)."""

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if platform.system() == "Darwin" else peak / 1e6


def _environment() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "wired_limit_env": {
            name: os.environ.get(name)
            for name in ("GLM_MLX_WIRED_LIMIT_GB", "MLX_WIRED_LIMIT_GB")
        },
        "e8p_backend_env": os.environ.get("GLM52_E8P_BACKEND"),
    }


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------


def _rate_points(max_bpw: float | None) -> list[tuple[int, int]]:
    points = [point for point in RATE_LADDER if point[1] >= 16]
    if max_bpw is not None:
        points = [point for point in points if rate_bpw(*point) <= max_bpw + 1e-9]
    return points


def command_sweep(args: argparse.Namespace) -> dict[str, object]:
    from mlx_vq.codebook.e8 import e8_1bit_grid, e8p_full_grid
    from mlx_vq.quality.rotation_search import (
        evaluate_rotation_gain,
        generate_rht_signs,
    )

    layers = [int(value) for value in args.layers]
    stats = aggregate_calibration_importance(
        args.calibration, layers=layers, session_limit=args.session_limit
    )
    points = _rate_points(args.max_bpw)
    results: list[dict[str, object]] = []
    fp4_reference: dict[str, object] = {}

    for layer in layers:
        row = stats["layers"].index(layer)
        # route_count is per input space; hidden and down agree by construction
        # (a token routed to an expert enters both its gate/up and its down).
        counts = np.asarray(stats["route_count"]["hidden"][row], dtype=np.int64)
        experts = stratified_experts(counts, args.experts)
        weights, scales, read_seconds = read_layer_experts(
            args.checkpoint, layer, experts=experts
        )
        print(
            f"[layer {layer}] read {read_seconds:.2f}s; "
            f"experts {experts[:4]}... route counts "
            f"{[int(counts[index]) for index in experts[:4]]}...",
            flush=True,
        )
        for projection in PROJECTIONS:
            per_point: dict[tuple[int, int], list[float]] = {
                point: [] for point in points
            }
            per_point_cos: dict[tuple[int, int], list[float]] = {
                point: [] for point in points
            }
            rht_rows: list[dict[str, object]] = []
            for expert in experts:
                dense = decode_expert(
                    weights, scales, projection=projection, expert=expert
                )
                importance = projection_importance(
                    stats, layer=layer, projection=projection, expert=expert
                )
                if not fp4_reference:
                    fp4_reference = {
                        "layer": layer,
                        "projection": projection,
                        "expert": expert,
                        **estimate_source_fp4_step(dense),
                    }
                for point in points:
                    code_bits, group_size = point
                    _, metrics, _ = fit_projection(
                        dense,
                        importance,
                        group_size=group_size,
                        code_bits=code_bits,
                        iterations=args.iterations,
                        backend=args.backend,
                    )
                    per_point[point].append(metrics.weighted_relative_mse)
                    per_point_cos[point].append(metrics.expected_cosine)
                if args.rht:
                    signs = generate_rht_signs(
                        dense.shape[1], f"{args.rht_seed}:{layer}:{projection}"
                    )
                    for code_bits, group_size in args.rht_points:
                        gain = evaluate_rotation_gain(
                            dense,
                            importance.astype(np.float32),
                            e8_1bit_grid() if code_bits == 8 else e8p_full_grid(),
                            group_size,
                            code_bits,
                            signs,
                        )
                        rht_rows.append(
                            {
                                "expert": expert,
                                "code_bits": code_bits,
                                "group_size": group_size,
                                "unrotated_objective": gain.unrotated_objective,
                                "rotated_objective": gain.rotated_objective,
                                "relative_gain": (
                                    gain.objective_gain / gain.unrotated_objective
                                    if gain.unrotated_objective > 0
                                    else 0.0
                                ),
                            }
                        )
            expert_counts = [int(counts[index]) for index in experts]
            for point in points:
                code_bits, group_size = point
                results.append(
                    {
                        "layer": layer,
                        "projection": projection,
                        "code_bits": code_bits,
                        "group_size": group_size,
                        "bpw": rate_bpw(code_bits, group_size),
                        "weighted_relative_mse_route_weighted": route_weighted_mean(
                            per_point[point], expert_counts
                        ),
                        "weighted_relative_mse_mean": float(
                            np.mean(per_point[point])
                        ),
                        "weighted_relative_mse_max": float(np.max(per_point[point])),
                        "expected_cosine_route_weighted": route_weighted_mean(
                            per_point_cos[point], expert_counts
                        ),
                        "experts_sampled": len(experts),
                    }
                )
                print(
                    f"  L{layer} {projection:>4} bits={code_bits:>2} g={group_size:>3} "
                    f"bpw={rate_bpw(code_bits, group_size):.4f} "
                    f"rel={results[-1]['weighted_relative_mse_route_weighted']:.5e} "
                    f"cos={results[-1]['expected_cosine_route_weighted']:.6f}",
                    flush=True,
                )
            if rht_rows:
                results[-1]["rht"] = rht_rows
        del weights, scales

    payload = {
        "record_type": "dsv4_vq_pilot_group_size_sweep_v1",
        "environment": _environment(),
        "checkpoint": str(args.checkpoint),
        "calibration": str(args.calibration),
        "calibration_sessions": stats["sessions"],
        "iterations": args.iterations,
        "backend": args.backend,
        "layers": layers,
        "experts_per_layer": args.experts,
        "fp4_source_step_estimate": fp4_reference,
        "rows": results,
        "peak_rss_gb": peak_rss_gb(),
    }
    write_json(args.out, payload)
    print(f"wrote {args.out}")
    return payload


# ---------------------------------------------------------------------------
# layer-fit
# ---------------------------------------------------------------------------


def command_layer_fit(args: argparse.Namespace) -> dict[str, object]:
    layer = int(args.layer)
    policy = GroupSizePolicy(
        gate=(args.gate_code_bits, args.gate_group_size),
        up=(args.up_code_bits, args.up_group_size),
        down=(args.down_code_bits, args.down_group_size),
    )
    stats = aggregate_calibration_importance(
        args.calibration, layers=[layer], session_limit=args.session_limit
    )
    row = stats["layers"].index(layer)
    counts = np.asarray(stats["route_count"]["hidden"][row], dtype=np.int64)
    experts = (
        list(range(int(stats["num_experts"])))
        if args.experts is None
        else stratified_experts(counts, args.experts)
    )

    weights, scales, read_seconds = read_layer_experts(args.checkpoint, layer)
    decode_seconds = 0.0
    fit_seconds = 0.0
    fitted = 0
    per_projection: dict[str, list[float]] = {name: [] for name in PROJECTIONS}
    per_projection_cos: dict[str, list[float]] = {name: [] for name in PROJECTIONS}
    started = time.perf_counter()
    for position, expert in enumerate(experts):
        for projection in PROJECTIONS:
            code_bits, group_size = policy.for_projection(projection)
            mark = time.perf_counter()
            dense = decode_expert(
                weights, scales, projection=projection, expert=expert
            )
            decode_seconds += time.perf_counter() - mark
            importance = projection_importance(
                stats, layer=layer, projection=projection, expert=expert
            )
            _, metrics, elapsed = fit_projection(
                dense,
                importance,
                group_size=group_size,
                code_bits=code_bits,
                iterations=args.iterations,
                backend=args.backend,
            )
            fit_seconds += elapsed
            fitted += 1
            per_projection[projection].append(metrics.weighted_relative_mse)
            per_projection_cos[projection].append(metrics.expected_cosine)
        if position and position % 32 == 0:
            done = time.perf_counter() - started
            print(
                f"  expert {position}/{len(experts)} "
                f"{done:.1f}s elapsed, {done / position:.3f}s/expert",
                flush=True,
            )
    wall_seconds = time.perf_counter() - started
    del weights, scales

    expert_counts = [int(counts[index]) for index in experts]
    timing = LayerFitTiming(
        layer=layer,
        projections_fitted=fitted,
        read_seconds=read_seconds,
        decode_seconds=decode_seconds,
        fit_seconds=fit_seconds,
        peak_rss_gb=peak_rss_gb(),
    )
    payload = {
        "record_type": "dsv4_vq_pilot_layer_fit_v1",
        "environment": _environment(),
        "checkpoint": str(args.checkpoint),
        "layer": layer,
        "policy": policy.as_profile_block(),
        "policy_bpw": policy_bpw(policy),
        "experts_fitted": len(experts),
        "projections_fitted": fitted,
        "read_seconds": read_seconds,
        "decode_seconds": decode_seconds,
        "fit_seconds": fit_seconds,
        "wall_seconds": wall_seconds,
        "unaccounted_seconds": wall_seconds - decode_seconds - fit_seconds,
        "peak_rss_gb": timing.peak_rss_gb,
        "error": {
            projection: {
                "weighted_relative_mse_route_weighted": route_weighted_mean(
                    per_projection[projection], expert_counts
                ),
                "expected_cosine_route_weighted": route_weighted_mean(
                    per_projection_cos[projection], expert_counts
                ),
                "weighted_relative_mse_max": float(
                    np.max(per_projection[projection])
                ),
            }
            for projection in PROJECTIONS
        },
    }
    write_json(args.out, payload)
    print(
        f"layer {layer}: read {read_seconds:.2f}s decode {decode_seconds:.1f}s "
        f"fit {fit_seconds:.1f}s wall {wall_seconds:.1f}s peak {timing.peak_rss_gb:.2f} GB"
    )
    print(f"wrote {args.out}")
    return payload


# ---------------------------------------------------------------------------
# control
# ---------------------------------------------------------------------------


def command_control(args: argparse.Namespace) -> dict[str, object]:
    """Affine group quantization at matched bit rates, as the VQ control.

    The campaign names a same-machine control artifact (streamed mxfp4 or
    resident affine-q3). This measures the affine family on the *same* experts
    and the *same* objective, so "VQ is worth it" is a measured claim at each
    rate rather than an inherited assumption. Affine stores an fp16 scale and
    an fp16 bias per group, hence ``bits + 32 / group_size``.
    """

    import mlx.core as mx

    from mlx_vq.convert.dsv4_vq_pilot import projection_error

    layers = [int(value) for value in args.layers]
    stats = aggregate_calibration_importance(
        args.calibration, layers=layers, session_limit=args.session_limit
    )
    rows: list[dict[str, object]] = []
    for layer in layers:
        row = stats["layers"].index(layer)
        counts = np.asarray(stats["route_count"]["hidden"][row], dtype=np.int64)
        experts = stratified_experts(counts, args.experts)
        weights, scales, _ = read_layer_experts(args.checkpoint, layer, experts=experts)
        for projection in PROJECTIONS:
            per_point: dict[tuple[int, int], list[float]] = {}
            per_point_cos: dict[tuple[int, int], list[float]] = {}
            for expert in experts:
                dense = decode_expert(
                    weights, scales, projection=projection, expert=expert
                )
                importance = projection_importance(
                    stats, layer=layer, projection=projection, expert=expert
                )
                for bits, group_size in args.points:
                    array = mx.array(dense)
                    packed, scale, bias = mx.quantize(
                        array, group_size=group_size, bits=bits
                    )
                    restored = mx.dequantize(
                        packed, scale, bias, group_size=group_size, bits=bits
                    )
                    mx.eval(restored)
                    metrics = projection_error(
                        dense, np.asarray(restored, dtype=np.float32), importance
                    )
                    per_point.setdefault((bits, group_size), []).append(
                        metrics.weighted_relative_mse
                    )
                    per_point_cos.setdefault((bits, group_size), []).append(
                        metrics.expected_cosine
                    )
            expert_counts = [int(counts[index]) for index in experts]
            for (bits, group_size), values in per_point.items():
                rows.append(
                    {
                        "layer": layer,
                        "projection": projection,
                        "scheme": f"affine-q{bits}",
                        "bits": bits,
                        "group_size": group_size,
                        # fp16 scale + fp16 bias per group.
                        "bpw": bits + 32.0 / group_size,
                        "weighted_relative_mse_route_weighted": route_weighted_mean(
                            values, expert_counts
                        ),
                        "expected_cosine_route_weighted": route_weighted_mean(
                            per_point_cos[(bits, group_size)], expert_counts
                        ),
                        "experts_sampled": len(experts),
                    }
                )
                print(
                    f"  L{layer} {projection:>4} affine-q{bits} g={group_size:>3} "
                    f"bpw={rows[-1]['bpw']:.4f} "
                    f"rel={rows[-1]['weighted_relative_mse_route_weighted']:.5e} "
                    f"cos={rows[-1]['expected_cosine_route_weighted']:.6f}",
                    flush=True,
                )
        del weights, scales

    payload = {
        "record_type": "dsv4_vq_pilot_affine_control_v1",
        "environment": _environment(),
        "note": "affine stores an fp16 scale and an fp16 bias per group",
        "rows": rows,
        "peak_rss_gb": peak_rss_gb(),
    }
    write_json(args.out, payload)
    print(f"wrote {args.out}")
    return payload


# ---------------------------------------------------------------------------
# block-probe
# ---------------------------------------------------------------------------


def command_block_probe(args: argparse.Namespace) -> dict[str, object]:
    """Whole-expert output error through the SwiGLU, per layer and rate point."""

    from mlx_vq.convert.dsv4_vq_pilot import expert_block_proxy, reconstruct_quantized
    from mlx_vq.convert.glm52_recovery_materialize import (
        quantize_weight_importance_aware,
    )

    layers = [int(value) for value in args.layers]
    stats = aggregate_calibration_importance(
        args.calibration, layers=layers, session_limit=args.session_limit
    )
    points = args.points
    rows: list[dict[str, object]] = []

    for layer in layers:
        row = stats["layers"].index(layer)
        counts = np.asarray(stats["route_count"]["hidden"][row], dtype=np.int64)
        experts = stratified_experts(counts, args.experts)
        weights, scales, _ = read_layer_experts(args.checkpoint, layer, experts=experts)
        for code_bits, group_size in points:
            per_expert_mse: list[float] = []
            per_expert_cos: list[float] = []
            used_counts: list[int] = []
            for expert in experts:
                route = int(counts[expert])
                if route <= 0:
                    continue
                dense = {
                    projection: decode_expert(
                        weights, scales, projection=projection, expert=expert
                    )
                    for projection in PROJECTIONS
                }
                rebuilt = {}
                for projection, values in dense.items():
                    importance = projection_importance(
                        stats, layer=layer, projection=projection, expert=expert
                    ).astype(np.float32)
                    quantized = quantize_weight_importance_aware(
                        values,
                        importance,
                        group_size=group_size,
                        code_bits=code_bits,
                        iterations=args.iterations,
                        e8p_search_backend=args.backend,
                    )
                    rebuilt[projection] = reconstruct_quantized(quantized)
                # sqrt(sum(x^2) / count) is the measured per-column activation RMS.
                sigma = np.sqrt(
                    projection_importance(
                        stats, layer=layer, projection="gate", expert=expert
                    )
                    / route
                )
                probe = expert_block_proxy(
                    dense,
                    rebuilt,
                    column_sigma=sigma,
                    tokens=args.tokens,
                    swiglu_limit=args.swiglu_limit,
                    seed=args.seed,
                )
                per_expert_mse.append(probe["block_relative_mse"])
                per_expert_cos.append(probe["block_cosine"])
                used_counts.append(route)
            rows.append(
                {
                    "layer": layer,
                    "code_bits": code_bits,
                    "group_size": group_size,
                    "bpw": rate_bpw(code_bits, group_size),
                    "block_relative_mse_route_weighted": route_weighted_mean(
                        per_expert_mse, used_counts
                    ),
                    "block_cosine_route_weighted": route_weighted_mean(
                        per_expert_cos, used_counts
                    ),
                    "block_relative_mse_max": float(np.max(per_expert_mse)),
                    "experts_sampled": len(used_counts),
                }
            )
            print(
                f"  L{layer} bits={code_bits} g={group_size:>3} "
                f"bpw={rows[-1]['bpw']:.4f} "
                f"block_rel={rows[-1]['block_relative_mse_route_weighted']:.5e} "
                f"block_cos={rows[-1]['block_cosine_route_weighted']:.6f}",
                flush=True,
            )
        del weights, scales

    payload = {
        "record_type": "dsv4_vq_pilot_block_probe_v1",
        "environment": _environment(),
        "proxy_caveat": (
            "synthetic diagonal-Gaussian hidden states at the measured per-column "
            "activation RMS; real hidden states are heavy-tailed and correlated, "
            "and no router, shared expert, or residual stream is modelled"
        ),
        "tokens": args.tokens,
        "swiglu_limit": args.swiglu_limit,
        "iterations": args.iterations,
        "rows": rows,
        "peak_rss_gb": peak_rss_gb(),
    }
    write_json(args.out, payload)
    print(f"wrote {args.out}")
    return payload


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------


def command_schedule(args: argparse.Namespace) -> dict[str, object]:
    geometry = load_profile_geometry(args.profile)
    measurements = []
    for path in args.layer_fit:
        payload = json.loads(Path(path).read_text())
        if payload.get("record_type") != "dsv4_vq_pilot_layer_fit_v1":
            raise ValueError(f"{path} is not a layer-fit artifact")
        measurements.append(
            LayerFitTiming(
                layer=int(payload["layer"]),
                projections_fitted=int(payload["projections_fitted"]),
                read_seconds=float(payload["read_seconds"]),
                decode_seconds=float(payload["decode_seconds"]),
                fit_seconds=float(payload["fit_seconds"]),
                peak_rss_gb=float(payload["peak_rss_gb"]),
            )
        )
    if not measurements:
        raise ValueError("at least one --layer-fit artifact is required")

    experts = geometry["num_experts"]
    sparse_layers = geometry["num_sparse_layers"]
    per_expert = sum(rows * cols for rows, cols in PROJECTION_SHAPE.values())
    routed_per_layer = per_expert * experts
    moe_blocks = sparse_layers + args.mtp_blocks
    routed_weight_count = routed_per_layer * moe_blocks

    estimates = []
    for label, bpw in args.ladder:
        estimates.append(
            extrapolate_schedule(
                measurements,
                label=label,
                bpw=bpw,
                layers=moe_blocks,
                projections_per_layer=experts * len(PROJECTIONS),
                routed_weight_count=routed_weight_count,
                resident_weight_count=args.resident_weights,
                resident_bpw=args.resident_bpw,
                codebook_bytes=args.codebook_bytes,
                assumptions=(
                    (
                        f"per-layer cost is the mean of {len(measurements)} "
                        "measured layer(s), assumed uniform across layers"
                    ),
                    (
                        f"{args.mtp_blocks} MTP drafter block(s) counted as full "
                        "MoE layers at the same cost and rate"
                    ),
                    (
                        f"resident tensors assumed "
                        f"{args.resident_weights / 1e9:.2f} G weights at "
                        f"{args.resident_bpw} bpw"
                    ),
                    "no thermal derating, no batching speed-up, single process",
                ),
            )
        )
    payload = {
        "record_type": "dsv4_vq_pilot_schedule_v1",
        "environment": _environment(),
        "geometry": geometry,
        "moe_blocks": moe_blocks,
        "routed_weight_count": routed_weight_count,
        "measured_layers": [timing.layer for timing in measurements],
        "envelope_gb": list(args.envelope_gb),
        "estimates": [
            {
                **estimate.as_dict(),
                "fits_envelope": estimate.artifact_bytes / 1e9 <= args.envelope_gb[1],
            }
            for estimate in estimates
        ],
    }
    write_json(args.out, payload)
    for estimate in payload["estimates"]:
        print(
            f"{estimate['label']:>28}  {estimate['bpw']:.3f} bpw  "
            f"{estimate['total_hours']:6.2f} h  "
            f"{estimate['artifact_gb']:7.2f} GB  "
            f"fits={estimate['fits_envelope']}"
        )
    print(f"wrote {args.out}")
    return payload


def command_monitor(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.run_dir)
    found = sorted(root.glob("*.json")) if root.is_dir() else []
    summary = []
    for path in found:
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            summary.append({"path": str(path), "state": "unreadable"})
            continue
        summary.append(
            {
                "path": str(path),
                "record_type": payload.get("record_type"),
                "layer": payload.get("layer"),
                "wall_seconds": payload.get("wall_seconds"),
            }
        )
        print(json.dumps(summary[-1]))
    if not summary:
        print(f"no pilot artifacts under {root}")
    return {"artifacts": summary}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _rate_point(text: str) -> tuple[int, int]:
    code_bits, _, group_size = text.partition(":")
    return int(code_bits), int(group_size)


def _ladder_entry(text: str) -> tuple[str, float]:
    label, _, bpw = text.partition("=")
    return label, float(bpw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-lock", action="store_true", help="skip the heavy-job lock")
    sub = parser.add_subparsers(dest="command", required=True)

    sweep = sub.add_parser("sweep", help="group-size / rate ladder error sweep")
    sweep.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    sweep.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    sweep.add_argument("--layers", nargs="+", default=["1", "20"])
    sweep.add_argument("--experts", type=int, default=16)
    sweep.add_argument("--iterations", type=int, default=3)
    sweep.add_argument("--backend", default="metal", choices=["metal", "numpy"])
    sweep.add_argument("--session-limit", type=int, default=None)
    sweep.add_argument("--max-bpw", type=float, default=None)
    sweep.add_argument("--rht", action="store_true", help="evaluate the RHT lever")
    sweep.add_argument("--rht-seed", default="dsv4-wave5-pilot")
    sweep.add_argument(
        "--rht-points", nargs="*", type=_rate_point, default=[(16, 512), (16, 32)]
    )
    sweep.add_argument("--out", type=Path, required=True)
    sweep.set_defaults(func=command_sweep, heavy=True)

    layer_fit = sub.add_parser("layer-fit", help="whole-layer fit for wall-clock")
    layer_fit.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    layer_fit.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    layer_fit.add_argument("--layer", type=int, required=True)
    layer_fit.add_argument("--experts", type=int, default=None)
    layer_fit.add_argument("--iterations", type=int, default=3)
    layer_fit.add_argument("--backend", default="metal", choices=["metal", "numpy"])
    layer_fit.add_argument("--session-limit", type=int, default=None)
    layer_fit.add_argument("--gate-code-bits", type=int, default=16)
    layer_fit.add_argument("--gate-group-size", type=int, default=32)
    layer_fit.add_argument("--up-code-bits", type=int, default=16)
    layer_fit.add_argument("--up-group-size", type=int, default=32)
    layer_fit.add_argument("--down-code-bits", type=int, default=16)
    layer_fit.add_argument("--down-group-size", type=int, default=32)
    layer_fit.add_argument("--out", type=Path, required=True)
    layer_fit.set_defaults(func=command_layer_fit, heavy=True)

    control = sub.add_parser("control", help="affine-quantizer control at matched rates")
    control.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    control.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    control.add_argument("--layers", nargs="+", default=["1", "20"])
    control.add_argument("--experts", type=int, default=16)
    control.add_argument("--session-limit", type=int, default=None)
    control.add_argument(
        "--points",
        nargs="*",
        type=_rate_point,
        default=[(2, 32), (3, 128), (3, 64), (3, 32), (4, 64), (4, 32)],
    )
    control.add_argument("--out", type=Path, required=True)
    control.set_defaults(func=command_control, heavy=True)

    block = sub.add_parser("block-probe", help="whole-expert SwiGLU output error")
    block.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    block.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    block.add_argument("--layers", nargs="+", default=["1", "20"])
    block.add_argument("--experts", type=int, default=8)
    block.add_argument("--tokens", type=int, default=512)
    block.add_argument("--swiglu-limit", type=float, default=10.0)
    block.add_argument("--iterations", type=int, default=8)
    block.add_argument("--backend", default="metal", choices=["metal", "numpy"])
    block.add_argument("--session-limit", type=int, default=None)
    block.add_argument("--seed", type=int, default=0)
    block.add_argument(
        "--points",
        nargs="*",
        type=_rate_point,
        default=[(8, 512), (16, 512), (16, 256), (16, 128), (16, 32), (16, 16)],
    )
    block.add_argument("--out", type=Path, required=True)
    block.set_defaults(func=command_block_probe, heavy=True)

    schedule = sub.add_parser("schedule", help="extrapolate the full sweep")
    schedule.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    schedule.add_argument("--layer-fit", nargs="+", required=True)
    schedule.add_argument(
        "--ladder",
        nargs="+",
        type=_ladder_entry,
        default=[("e8p-g512", 2.03125), ("e8p-g32", 2.5), ("e8p-g16", 3.0)],
    )
    schedule.add_argument("--mtp-blocks", type=int, default=3)
    schedule.add_argument("--resident-weights", type=float, default=7.83e9)
    schedule.add_argument("--resident-bpw", type=float, default=8.5)
    schedule.add_argument("--codebook-bytes", type=int, default=0)
    schedule.add_argument("--envelope-gb", nargs=2, type=float, default=[98.0, 105.0])
    schedule.add_argument("--out", type=Path, required=True)
    schedule.set_defaults(func=command_schedule, heavy=False)

    monitor = sub.add_parser("monitor", help="show pilot artifacts in a run dir")
    monitor.add_argument("--run-dir", type=Path, required=True)
    monitor.set_defaults(func=command_monitor, heavy=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with heavy_job_lock(
        HEAVY_JOB_LOCK, enabled=getattr(args, "heavy", False) and not args.no_lock
    ):
        args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
