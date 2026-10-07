#!/usr/bin/env python3
"""Wave 5 production materializer for DeepSeek-V4-Flash-0731 routed experts.

The CLI behind the registered converter kind ``deepseek_v4_vq_groups``. It
turns the release's FP4-in-I8 routed experts into KEEP-format per-block VQ
artifacts at the ladder the Wave 5 pilot settled -- E8P ``code_bits=16``,
``group_size=512`` uniform, 8 iterations, 46 MoE blocks (43 backbone layers plus
the three DSpark drafter blocks at ``mtp.{0,1,2}``), 2.031 bpw routed, ~83.6 GB.

Subcommands, in the order a run uses them::

    imatrix           aggregate the Wave 3 calibration captures into one cache
    plan              print the block plan and what is already done
    parity            NumPy vs MLX fit: speed and byte-parity, on real weights
    block             materialize one block (the unit the sweep resumes on)
    verify-roundtrip  bind a materialized block through the adapter and prove
                      the w1->gate / w3->up / w2->down contract semantically
    sweep             materialize every block, resumable, with a status file
    manifest          (re)build the manifest from the per-block records
    monitor           print the current status of a run directory

Heavy subcommands take ``.keep-heavy-job.lock``, the shared KEEP cooperative
lock, unless ``--no-lock``.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from keep.convert.dsv4_vq_materialize import (
    PROJECTIONS,
    BlockSpec,
    MaterializePolicy,
    build_imatrix_cache,
    build_manifest,
    environment,
    load_imatrix_cache,
    materialize_block,
    peak_rss_gb,
    plan_blocks,
    record_path,
    verified_block_record,
    verify_bind_roundtrip,
    write_block_record,
)
from keep.convert.dsv4_vq_pilot import write_json

DEFAULT_CHECKPOINT = Path.home() / "models" / "DeepSeek-V4-Flash-0731"
DEFAULT_CALIBRATION = Path.home() / "keep-artifacts" / "dsv4-teacher-calibration"
DEFAULT_OUTPUT = Path.home() / "keep-artifacts" / "dsv4-vq-e8p-g512"
HEAVY_JOB_LOCK = REPO_ROOT / ".keep-heavy-job.lock"
PILOT_REPORT = "docs/deepseek-v4-flash/research/wave5-pilot-report.md"
PILOT_EVIDENCE = "artifacts/quality/dsv4-vq-pilot-20260813"


@contextlib.contextmanager
def heavy_job_lock(path: Path, *, enabled: bool = True, holder: str) -> Iterator[None]:
    """The shared KEEP heavy-job lock: cooperative, released on process exit."""

    if not enabled:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(f"{os.getpid()}\n{holder}\n")
            handle.flush()
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _policy(args: argparse.Namespace) -> MaterializePolicy:
    return MaterializePolicy(
        code_bits=args.code_bits,
        group_size=args.group_size,
        iterations=args.iterations,
        fit_backend=args.fit_backend,
        mtp_importance=args.mtp_importance,
        search_backend=args.search_backend,
    )


def _specs(args: argparse.Namespace) -> tuple[BlockSpec, ...]:
    return plan_blocks(
        num_layers=args.num_layers,
        mtp_blocks=args.mtp_blocks,
        only=args.blocks,
    )


def _cache_path(args: argparse.Namespace) -> Path:
    return Path(args.imatrix_cache or Path(args.output) / "imatrix-cache.npz")


def _ensure_cache(args: argparse.Namespace) -> tuple[dict, dict]:
    """Load the imatrix cache, building it first if this is a fresh run."""

    path = _cache_path(args)
    meta_path = path.with_suffix(".json")
    if not path.is_file():
        print(f"[imatrix] building cache at {path} ...", flush=True)
        started = time.perf_counter()
        meta = build_imatrix_cache(
            args.calibration,
            path,
            layers=list(range(args.num_layers)),
            session_limit=args.session_limit,
        )
        meta["build_seconds"] = round(time.perf_counter() - started, 1)
        meta["calibration_dir"] = str(args.calibration)
        write_json(meta_path, meta)
        print(f"[imatrix] {meta['sessions']} sessions in {meta['build_seconds']}s", flush=True)
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {"path": str(path)}
    return load_imatrix_cache(path), meta


# ---------------------------------------------------------------------------
# imatrix / plan
# ---------------------------------------------------------------------------


def command_imatrix(args: argparse.Namespace) -> dict:
    _, meta = _ensure_cache(args)
    print(json.dumps(meta, indent=2, sort_keys=True))
    return meta


def command_plan(args: argparse.Namespace) -> dict:
    policy = _policy(args)
    specs = _specs(args)
    done, pending = [], []
    for spec in specs:
        complete = verified_block_record(
            args.output, spec, policy=policy, expected_experts=args.experts_per_block
        )
        (done if complete else pending).append(spec.key)
    payload = {
        "record_type": "dsv4_vq_materialize_plan_v1",
        "output_dir": str(args.output),
        "policy": policy.as_dict(),
        "blocks": [spec.as_dict() for spec in specs],
        "complete": done,
        "pending": pending,
    }
    print(
        f"{len(specs)} MoE blocks at {policy.bpw:.4f} bpw "
        f"({len(done)} complete, {len(pending)} pending)"
    )
    for spec in specs:
        state = "done" if spec.key in done else "pending"
        print(f"  {spec.key:<12} {spec.kind:<9} {spec.file_stem:<12} {state}")
    if args.out:
        write_json(args.out, payload)
    return payload


# ---------------------------------------------------------------------------
# parity
# ---------------------------------------------------------------------------


def command_parity(args: argparse.Namespace) -> dict:
    """NumPy vs MLX on a real projection: speed and byte-parity together."""

    from keep.convert.dsv4_vq_fit_mlx import compare_fit_backends
    from keep.convert.dsv4_vq_materialize import (
        block_importance,
        decode_block_expert,
        read_block_experts,
    )

    policy = _policy(args)
    cache, cache_meta = _ensure_cache(args)
    spec = plan_blocks(num_layers=args.num_layers, mtp_blocks=args.mtp_blocks, only=[args.block])[0]
    weights, scales, read_seconds, span = read_block_experts(args.checkpoint, spec.key)
    rows = []
    for expert in args.experts:
        for projection in PROJECTIONS:
            dense = decode_block_expert(
                weights, scales, projection=projection, expert=int(expert)
            )
            importance, source = block_importance(
                cache,
                spec,
                projection=projection,
                expert=int(expert),
                policy=policy,
                mtp_reference_layer=args.mtp_reference_layer,
            )
            comparison = compare_fit_backends(
                dense,
                importance,
                group_size=policy.group_size,
                code_bits=policy.code_bits,
                iterations=policy.iterations,
            )
            comparison["block"] = spec.key
            comparison["expert"] = int(expert)
            comparison["projection"] = projection
            comparison["importance_source"] = source
            rows.append(dict(comparison))
            backends = comparison["backends"]
            print(
                f"  {spec.key} e{expert} {projection:<10} "
                f"numpy={backends['numpy']['seconds']:.3f}s "
                f"mlx-exact={backends['mlx-exact']['seconds']:.3f}s "
                f"({backends['mlx-exact']['speedup_vs_numpy']:.2f}x, "
                f"identical={backends['mlx-exact']['codes_byte_identical']}) "
                f"mlx-fp32={backends['mlx-fp32']['seconds']:.3f}s "
                f"({backends['mlx-fp32']['speedup_vs_numpy']:.2f}x, "
                f"identical={backends['mlx-fp32']['codes_byte_identical']}, "
                f"disagree={backends['mlx-fp32']['code_disagreement_fraction']:.3e})",
                flush=True,
            )
    del weights, scales

    def _summarise(backend: str) -> dict:
        return {
            "mean_seconds": float(
                np.mean([row["backends"][backend]["seconds"] for row in rows])
            ),
            "mean_speedup_vs_numpy": float(
                np.mean(
                    [row["backends"][backend].get("speedup_vs_numpy", 1.0) for row in rows]
                )
            ),
            "all_codes_byte_identical": all(
                row["backends"][backend].get("codes_byte_identical", True) for row in rows
            ),
            "all_scales_byte_identical": all(
                row["backends"][backend].get("scales_byte_identical", True) for row in rows
            ),
            "max_code_disagreement_fraction": float(
                max(
                    row["backends"][backend].get("code_disagreement_fraction", 0.0)
                    for row in rows
                )
            ),
            "max_abs_relative_mse_delta": float(
                max(
                    abs(row["backends"][backend].get("relative_mse_delta", 0.0))
                    for row in rows
                )
            ),
        }

    payload = {
        "record_type": "dsv4_vq_fit_parity_v1",
        "environment": environment(),
        "block": spec.key,
        "policy": policy.as_dict(),
        "calibration": cache_meta,
        "read_seconds": read_seconds,
        "source_span_bytes": span.span_bytes,
        "rows": rows,
        "summary": {name: _summarise(name) for name in ("numpy", "mlx-exact", "mlx-fp32")},
        "peak_rss_gb": peak_rss_gb(),
    }
    write_json(args.out, payload)
    print(f"wrote {args.out}")
    return payload


# ---------------------------------------------------------------------------
# block / sweep
# ---------------------------------------------------------------------------


def _progress_printer(prefix: str = ""):
    def report(*, block, projection, expert, experts, elapsed, fitted):
        print(
            f"{prefix}  [{block}] {projection:<10} expert {expert}/{experts} "
            f"{elapsed:.1f}s elapsed ({elapsed / max(fitted, 1):.3f}s/projection)",
            flush=True,
        )

    return report


def _materialize_one(
    spec: BlockSpec,
    args: argparse.Namespace,
    policy: MaterializePolicy,
    cache,
    progress=None,
) -> dict:
    record = materialize_block(
        spec,
        checkpoint_dir=args.checkpoint,
        output_dir=args.output,
        cache=cache,
        policy=policy,
        mtp_reference_layer=args.mtp_reference_layer,
        experts=args.experts if getattr(args, "experts", None) else None,
        progress=progress or _progress_printer(),
    )
    write_block_record(args.output, record)
    return record.as_dict()


def command_block(args: argparse.Namespace) -> dict:
    policy = _policy(args)
    cache, _ = _ensure_cache(args)
    spec = plan_blocks(num_layers=args.num_layers, mtp_blocks=args.mtp_blocks, only=[args.block])[0]
    payload = _materialize_one(spec, args, policy, cache)
    timing = payload["timing"]
    print(
        f"{spec.key}: read {timing['read_seconds']:.2f}s decode "
        f"{timing['decode_seconds']:.1f}s fit {timing['fit_seconds']:.1f}s "
        f"write {timing['write_seconds']:.1f}s wall {timing['wall_seconds']:.1f}s "
        f"peak {timing['peak_rss_gb']:.2f} GB"
    )
    print(f"wrote {record_path(args.output, spec)}")
    return payload


class SweepHeartbeat:
    """Atomic ``status.json`` plus an append-only ``status.log``.

    Same shape as the Wave 3 teacher runner's heartbeat so one monitor habit
    covers both: the JSON is what a poller reads, the log keeps one line per
    completed block so a rate trend (thermal derate, contention) is visible
    rather than a surprise at the end.
    """

    def __init__(self, path: Path, *, total_blocks: int, policy: MaterializePolicy, extra: dict):
        self.path = Path(path)
        self.log_path = self.path.with_suffix(".log")
        self.started = time.time()
        self.total_blocks = int(total_blocks)
        self.policy = policy
        self.extra = dict(extra)
        self.blocks_done = 0
        self.blocks_skipped = 0
        self.block_seconds: list[float] = []
        self.mtp_seconds: list[float] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _eta(self) -> tuple[float | None, str]:
        remaining = self.total_blocks - self.blocks_done - self.blocks_skipped
        if not self.block_seconds:
            return None, "no completed block yet"
        rate = float(np.mean(self.block_seconds))
        basis = f"mean of {len(self.block_seconds)} measured block(s) at {rate:.1f}s"
        if self.mtp_seconds:
            basis += f"; MTP measured at {float(np.mean(self.mtp_seconds)):.1f}s/block"
        else:
            basis += "; MTP blocks NOT yet measured, counted at the backbone rate"
        return remaining * rate, basis

    def payload(self, **fields) -> dict:
        eta, basis = self._eta()
        elapsed = max(time.time() - self.started, 1e-9)
        return {
            "record_type": "dsv4_vq_materialize_status_v1",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pid": os.getpid(),
            "elapsed_s": round(elapsed, 1),
            "blocks_done": self.blocks_done,
            "blocks_skipped": self.blocks_skipped,
            "total_blocks": self.total_blocks,
            "seconds_per_block": round(float(np.mean(self.block_seconds)), 1)
            if self.block_seconds
            else None,
            "mtp_seconds_per_block": round(float(np.mean(self.mtp_seconds)), 1)
            if self.mtp_seconds
            else None,
            "eta_s": round(eta, 0) if eta else None,
            "eta_h": round(eta / 3600.0, 2) if eta else None,
            "eta_basis": basis,
            "peak_rss_gb": round(peak_rss_gb(), 2),
            "policy": self.policy.as_dict(),
            **self.extra,
            **fields,
        }

    def update(self, **fields) -> dict:
        payload = self.payload(**fields)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, self.path)
        return payload

    def log(self, **fields) -> dict:
        payload = self.update(**fields)
        with self.log_path.open("a") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return payload


def command_sweep(args: argparse.Namespace) -> dict:
    policy = _policy(args)
    specs = _specs(args)
    cache, cache_meta = _ensure_cache(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    heartbeat = SweepHeartbeat(
        output / "status.json",
        total_blocks=len(specs),
        policy=policy,
        extra={
            "output_dir": str(output),
            "checkpoint_dir": str(args.checkpoint),
            "calibration_dir": str(args.calibration),
            "environment": environment(),
        },
    )
    heartbeat.update(phase="starting", block=None)
    _printer = _progress_printer()

    records: list[dict] = []
    for spec in specs:
        existing = (
            None
            if args.force
            else verified_block_record(
                output, spec, policy=policy, expected_experts=args.experts_per_block
            )
        )
        if existing is not None:
            records.append(existing)
            heartbeat.blocks_skipped += 1
            heartbeat.update(phase="skipped", block=spec.key)
            print(f"[{spec.key}] complete and verified, skipping", flush=True)
            continue
        heartbeat.update(phase="running", block=spec.key)
        started = time.perf_counter()
        def progress(**fields):
            # An intra-block heartbeat: a block is ~100 s, and a status file that
            # only moved between blocks would look stalled for most of that.
            _printer(**fields)
            heartbeat.update(
                phase="running",
                block=fields["block"],
                block_projection=fields["projection"],
                block_expert=fields["expert"],
                block_experts=fields["experts"],
                block_elapsed_s=round(fields["elapsed"], 1),
            )

        payload = _materialize_one(spec, args, policy, cache, progress=progress)
        elapsed = time.perf_counter() - started
        records.append(payload)
        heartbeat.blocks_done += 1
        heartbeat.block_seconds.append(elapsed)
        if spec.kind == "mtp":
            heartbeat.mtp_seconds.append(elapsed)
        heartbeat.log(
            phase="block_complete",
            block=spec.key,
            block_kind=spec.kind,
            block_seconds=round(elapsed, 1),
            block_timing=payload["timing"],
            block_metrics=payload["fit_metrics"],
        )
        print(
            f"[{spec.key}] done in {elapsed:.1f}s "
            f"({heartbeat.blocks_done + heartbeat.blocks_skipped}/{len(specs)})",
            flush=True,
        )
        _write_manifest(args, policy, specs, records, cache_meta)

    manifest = _write_manifest(args, policy, specs, records, cache_meta)
    heartbeat.update(phase="complete", block=None)
    print(f"sweep complete: {len(records)}/{len(specs)} blocks")
    return manifest


def _bind_proof_from_evidence(output: Path) -> dict | None:
    path = output / "roundtrip.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return payload.get("bind_proof")


def _write_manifest(args, policy, specs, records, cache_meta) -> dict:
    output = Path(args.output)
    manifest = build_manifest(
        output_dir=output,
        checkpoint_dir=args.checkpoint,
        specs=specs,
        policy=policy,
        records=records,
        calibration=cache_meta,
        bind_proof=_bind_proof_from_evidence(output),
        evidence={
            "pilot_report": PILOT_REPORT,
            "pilot_evidence_dir": PILOT_EVIDENCE,
            "calibration_dir": str(args.calibration),
            "status": "status.json",
            "status_log": "status.log",
            "block_records": "records/",
            "roundtrip": "roundtrip.json",
            "fit_parity": "fit-parity.json",
        },
    )
    write_json(output / "manifest.json", manifest)
    return manifest


def command_manifest(args: argparse.Namespace) -> dict:
    policy = _policy(args)
    specs = _specs(args)
    output = Path(args.output)
    records = []
    for spec in specs:
        payload = verified_block_record(
            output, spec, policy=policy, expected_experts=args.experts_per_block
        )
        if payload is not None:
            records.append(payload)
    meta_path = _cache_path(args).with_suffix(".json")
    cache_meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    manifest = _write_manifest(args, policy, specs, records, cache_meta)
    print(json.dumps(manifest["audit"], indent=2, sort_keys=True))
    print(f"wrote {output / 'manifest.json'}")
    return manifest


# ---------------------------------------------------------------------------
# verify-roundtrip -- the naming-contract proof
# ---------------------------------------------------------------------------


def command_verify_roundtrip(args: argparse.Namespace) -> dict:
    """Bind one materialized block and prove the projection naming contract."""

    policy = _policy(args)
    output = Path(args.output)
    spec = plan_blocks(num_layers=args.num_layers, mtp_blocks=args.mtp_blocks, only=[args.block])[0]
    record = verified_block_record(
        output, spec, policy=policy, expected_experts=args.experts_per_block
    )
    if record is None:
        raise ValueError(
            f"{spec.key} has no verified block record under {output}; materialize it first"
        )
    payload = verify_bind_roundtrip(
        output,
        spec,
        policy=policy,
        experts=args.experts,
        tokens=args.tokens,
        input_scale=args.input_scale,
        tolerance=args.tolerance,
        seed=args.seed,
        bind_path=args.bind_path,
    )
    write_json(args.out or (output / "roundtrip.json"), payload)
    print(json.dumps(payload["bind_proof"], indent=2, sort_keys=True))
    print(json.dumps(payload["forward_agreement"], indent=2, sort_keys=True))
    print(f"roundtrip passed={payload['passed']}")
    if not payload["passed"]:
        raise SystemExit(2)
    return payload


# ---------------------------------------------------------------------------
# monitor
# ---------------------------------------------------------------------------


def command_monitor(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    status_path = output / "status.json"
    if not status_path.is_file():
        print(f"no status.json under {output}")
        return {}
    payload = json.loads(status_path.read_text())
    done = payload.get("blocks_done", 0) + payload.get("blocks_skipped", 0)
    total = payload.get("total_blocks", 0)
    print(
        f"phase={payload.get('phase')} block={payload.get('block')} "
        f"{done}/{total} blocks  elapsed={payload.get('elapsed_s')}s  "
        f"eta={payload.get('eta_h')}h  peak={payload.get('peak_rss_gb')} GB"
    )
    print(f"eta basis: {payload.get('eta_basis')}")
    if args.verbose:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imatrix-cache", type=Path, default=None)
    parser.add_argument("--session-limit", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=43)
    parser.add_argument("--mtp-blocks", type=int, default=3)
    parser.add_argument("--code-bits", type=int, default=16, choices=[8, 16])
    parser.add_argument("--group-size", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument(
        "--fit-backend", default="mlx-exact", choices=["numpy", "mlx-exact", "mlx-fp32"]
    )
    parser.add_argument("--search-backend", default="metal", choices=["metal", "numpy"])
    parser.add_argument(
        "--mtp-importance", default="backbone-mean", choices=["backbone-mean", "uniform"]
    )
    parser.add_argument("--mtp-reference-layer", type=int, default=42)
    parser.add_argument(
        "--experts-per-block",
        type=int,
        default=256,
        help="expert count a block record must cover to count as complete",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-lock", action="store_true", help="skip the heavy-job lock")
    sub = parser.add_subparsers(dest="command", required=True)

    imatrix = sub.add_parser("imatrix", help="build the aggregated imatrix cache")
    _add_common(imatrix)
    imatrix.set_defaults(func=command_imatrix, heavy=True, blocks=None)

    plan = sub.add_parser("plan", help="show the block plan and what is complete")
    _add_common(plan)
    plan.add_argument("--blocks", nargs="*", default=None)
    plan.add_argument("--out", type=Path, default=None)
    plan.set_defaults(func=command_plan, heavy=False)

    parity = sub.add_parser("parity", help="NumPy vs MLX fit speed and byte parity")
    _add_common(parity)
    parity.add_argument("--block", default="layers.20")
    parity.add_argument("--experts", nargs="*", type=int, default=[0, 128, 255])
    parity.add_argument("--out", type=Path, required=True)
    parity.set_defaults(func=command_parity, heavy=True, blocks=None)

    block = sub.add_parser("block", help="materialize one MoE block")
    _add_common(block)
    block.add_argument("--block", required=True)
    block.add_argument("--experts", nargs="*", type=int, default=None)
    block.set_defaults(func=command_block, heavy=True, blocks=None)

    sweep = sub.add_parser("sweep", help="materialize every MoE block, resumably")
    _add_common(sweep)
    sweep.add_argument("--blocks", nargs="*", default=None)
    sweep.add_argument("--force", action="store_true", help="redo complete blocks")
    sweep.set_defaults(func=command_sweep, heavy=True, experts=None)

    verify = sub.add_parser("verify-roundtrip", help="bind a block and prove the mapping")
    _add_common(verify)
    verify.add_argument("--block", default="layers.0")
    verify.add_argument("--experts", nargs="*", type=int, default=[0, 1, 2, 3, 4, 5])
    verify.add_argument("--tokens", type=int, default=4)
    verify.add_argument("--input-scale", type=float, default=0.02)
    verify.add_argument("--tolerance", type=float, default=1e-3)
    verify.add_argument("--seed", type=int, default=0)
    verify.add_argument(
        "--bind-path",
        default="auto",
        choices=["auto", "binder", "loader"],
        help="'binder' exercises file discovery too but only reaches layer 0",
    )
    verify.add_argument("--out", type=Path, default=None)
    verify.set_defaults(func=command_verify_roundtrip, heavy=False, blocks=None)

    manifest = sub.add_parser("manifest", help="rebuild manifest.json from block records")
    _add_common(manifest)
    manifest.add_argument("--blocks", nargs="*", default=None)
    manifest.set_defaults(func=command_manifest, heavy=False)

    monitor = sub.add_parser("monitor", help="print a run directory's status")
    monitor.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    monitor.add_argument("--verbose", action="store_true")
    monitor.set_defaults(func=command_monitor, heavy=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with heavy_job_lock(
        HEAVY_JOB_LOCK,
        enabled=getattr(args, "heavy", False) and not args.no_lock,
        holder=f"dsv4-vq-materialize:{args.command}",
    ):
        args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
