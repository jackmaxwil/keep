"""Kill test: replay recorded routing through an in-RAM hot set of experts.

Reads the ``routes__*`` arrays that DSV4 calibration sessions record (the
first tokens of each session, which are also its decode routing) and, if
given, ``reap-stats.npz`` to choose the initial hot set in REAP order. Replays
every hot-set size and policy asked for and applies the plan's rule: at a
given hot-set size, if the best policy that fetches every miss still misses
more than 5% of routed slots, keeping all experts behind an SSD tier at that
size is too slow to be worth building.

Example (V4-Flash, half the experts resident)::

    uv run python benchmarks/replay_dsv4_expert_residency.py \\
        --calibration-dir ~/keep-artifacts/dsv4-teacher-calibration \\
        --reap-stats ~/keep-artifacts/dsv4-teacher-calibration/reap-stats.npz \\
        --hot 96,128,160 --expert-mb 6.39 --output residency.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from keep.quality.expert_residency import (
    ROUTE_ARRAY_PREFIX,
    decode_tokens_per_second,
    residency_report,
    simulate_sessions,
)


def load_route_sessions(calibration_dir: Path) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    sessions: list[tuple[np.ndarray, np.ndarray]] = []
    layers: np.ndarray | None = None
    missing: list[str] = []
    for path in sorted((calibration_dir / "sessions").glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            if f"{ROUTE_ARRAY_PREFIX}expert_ids" not in archive.files:
                missing.append(path.name)
                continue
            current = archive[f"{ROUTE_ARRAY_PREFIX}layers"].astype(np.int64)
            if layers is None:
                layers = current
            elif not np.array_equal(layers, current):
                raise ValueError(f"{path.name} records a different layer set")
            sessions.append(
                (
                    archive[f"{ROUTE_ARRAY_PREFIX}expert_ids"].astype(np.int64),
                    archive[f"{ROUTE_ARRAY_PREFIX}gate_weights"].astype(np.float32),
                )
            )
    if missing:
        raise ValueError(f"{len(missing)} sessions have no recorded routes: {missing[:5]}")
    if not sessions or layers is None:
        raise ValueError(f"no session routes under {calibration_dir / 'sessions'}")
    return sessions, layers


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--reap-stats", type=Path, help="initial hot set in REAP order")
    parser.add_argument("--num-experts", type=int, help="default: from --reap-stats")
    parser.add_argument("--hot", default="128", help="comma-separated hot experts per layer")
    parser.add_argument("--policies", default="static,lru,decayed")
    parser.add_argument("--half-life", type=float, default=64.0)
    parser.add_argument("--fetch-threshold", type=float, help="drop misses below this gate weight")
    parser.add_argument("--warmup-tokens", type=int, default=0)
    parser.add_argument("--expert-mb", type=float, default=8.9856, help="bytes per expert, MB (V4.1 at 2.031 bpw)")
    parser.add_argument("--resident-ms", type=float, help="resident-only decode ms/token, for a tok/s estimate")
    parser.add_argument("--ssd-gb-s", type=float, default=5.0)
    parser.add_argument("--kill-miss-rate", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    sessions, layers = load_route_sessions(args.calibration_dir.expanduser())
    initial = None
    num_experts = args.num_experts
    if args.reap_stats is not None:
        from keep.quality.reap import ReapStats, expert_saliency

        stats = ReapStats.load(args.reap_stats.expanduser())
        if not np.array_equal(stats.layers, layers):
            parser.error("--reap-stats covers different layers than the recorded routes")
        initial = expert_saliency(stats, "reap")
        num_experts = num_experts or stats.num_experts
    if num_experts is None:
        parser.error("pass --num-experts or --reap-stats")
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    if "static" in policies and initial is None:
        policies.remove("static")  # a static set needs an order to choose it
    expert_bytes = args.expert_mb * 1e6
    results = []
    for hot in (int(h) for h in args.hot.split(",")):
        for policy in policies:
            results.append(
                simulate_sessions(
                    sessions,
                    num_experts=num_experts,
                    hot_per_layer=hot,
                    policy=policy,
                    half_life=args.half_life,
                    initial_order=initial,
                    expert_bytes=expert_bytes,
                    warmup_tokens=args.warmup_tokens,
                )
            )
            if args.fetch_threshold is not None:
                results.append(
                    simulate_sessions(
                        sessions,
                        num_experts=num_experts,
                        hot_per_layer=hot,
                        policy=policy,
                        half_life=args.half_life,
                        initial_order=initial,
                        expert_bytes=expert_bytes,
                        warmup_tokens=args.warmup_tokens,
                        fetch_threshold=args.fetch_threshold,
                    )
                )
    report = dict(residency_report(results, kill_miss_rate=args.kill_miss_rate))
    report["sessions"] = len(sessions)
    report["layers"] = layers.tolist()
    report["expert_mb"] = args.expert_mb
    if args.resident_ms is not None:
        report["tokens_per_second_estimate"] = {
            f"{r.policy}@{r.hot_per_layer}"
            + ("" if r.dropped == 0 else f"+drop<{args.fetch_threshold}"): round(
                decode_tokens_per_second(
                    resident_ms_per_token=args.resident_ms,
                    misses_per_token=r.misses_per_token,
                    expert_bytes=expert_bytes,
                    ssd_bytes_per_second=args.ssd_gb_s * 1e9,
                ),
                2,
            )
            for r in results
        }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
