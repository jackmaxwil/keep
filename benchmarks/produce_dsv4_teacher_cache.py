"""CLI: layer-sequential streaming teacher cache for DeepSeek-V4-Flash-0731.

Streams the 163 GB released checkpoint through KEEP's MLX adapter one layer at a
time and writes per-session teacher artifacts. Design, measurements and the two
pinned behaviours live in :mod:`mlx_vq.quality.dsv4_teacher_runner` and in
``.superpowers/sdd/wave2-increment2-report.md``.

Verbs
-----
``run``       produce artifacts for a split (resumable; skips validated sessions)
``monitor``   print, or follow, a run's status file
``finalize``  merge calibration sessions into imatrix sidecars
``plan``      print the streaming plan and the byte budget, touching no weights

Launch the calibration split (40 sessions, ~5-8 h) detached::

    nohup uv run --group dev python benchmarks/produce_dsv4_teacher_cache.py run \\
        --mode calibration --split calibration \\
        --out-dir ~/keep-artifacts/dsv4-teacher-calibration \\
        > ~/keep-artifacts/dsv4-teacher-calibration/nohup.out 2>&1 &
    disown

Watch it from anywhere::

    uv run python benchmarks/produce_dsv4_teacher_cache.py monitor \\
        --out-dir ~/keep-artifacts/dsv4-teacher-calibration --follow

Resume after any interruption: re-run the identical ``run`` command. Completed
sessions are re-validated and skipped; an interrupted session costs itself and
nothing else.

The heavy-job lock at the repo root is held for the whole ``run``, and
``--chunk`` is frozen into every artifact because chunked prefill is not
bit-identical across chunk sizes.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

HEAVY_JOB_LOCK = REPO_ROOT / ".keep-heavy-job.lock"
DEFAULT_CHECKPOINT = Path.home() / "models/DeepSeek-V4-Flash-0731"
DEFAULT_PACK = Path.home() / "models/teich/dsv4-coding-agent-v1-20260811.json"

#: Env vars whose presence would silently change the memory behaviour a run's
#: peak numbers are reported against. Increment 2 measured with them unset.
_WIRED_LIMIT_ENV = (
    "GLM_MLX_WIRED_LIMIT_GB",
    "GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB",
    "MLX_METAL_DEBUG",
    "IOGPUWiredLimitMB",
    "MLX_WIRED_LIMIT_MB",
    "MLX_RELAXED_WIRED_LIMIT",
)


def _refuse_wired_limit_env() -> None:
    present = sorted(name for name in _WIRED_LIMIT_ENV if name in os.environ)
    if present:
        raise SystemExit(
            "refusing to run with wired-limit environment overrides set: "
            f"{present}. Increment 2's throughput and peak-memory numbers were "
            "measured with these unset; a run under different limits is not "
            "comparable to them. unset and retry."
        )


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _cmd_run(opts: argparse.Namespace) -> int:
    from mlx_vq.quality.dsv4_teacher_runner import (
        DEFAULT_PREFILL_CHUNK_TOKENS,
        MtpDrafterUnavailable,
        run_dsv4_teacher_production,
    )

    _refuse_wired_limit_env()
    out_dir = Path(opts.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    lock = HEAVY_JOB_LOCK.open("w")
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            if not opts.wait_for_lock:
                raise SystemExit(
                    f"another heavy job holds {HEAVY_JOB_LOCK}; pass "
                    "--wait-for-lock to queue behind it"
                )
            print(f"waiting for {HEAVY_JOB_LOCK}...", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        lock.write(f"{os.getpid()}\ndsv4-teacher-{opts.mode}\n")
        lock.flush()

        try:
            summary = run_dsv4_teacher_production(
                mode=opts.mode,
                pack_path=Path(opts.pack).expanduser(),
                out_dir=out_dir,
                checkpoint_dir=Path(opts.checkpoint).expanduser(),
                splits=opts.split or None,
                prompt_ids=opts.session_id or None,
                max_session_tokens=opts.max_session_tokens,
                max_sessions=opts.max_sessions,
                chunk=opts.chunk or DEFAULT_PREFILL_CHUNK_TOKENS,
                top_k=opts.top_k,
                lm_head_slice=opts.lm_head_slice,
                mtp_draft_width=opts.mtp_draft_width,
                io_threads=opts.io_threads,
                nocache=not opts.page_cache,
                compress_outputs=opts.compress,
                status_path=out_dir / "status.json",
                stop_file=Path(opts.stop_file).expanduser() if opts.stop_file else None,
            )
        except MtpDrafterUnavailable as error:
            print(json.dumps({"status": "refused", "reason": str(error)}, indent=2))
            return 2
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------
# monitor
# ---------------------------------------------------------------------------


def _format_status(payload: dict) -> str:
    done = payload.get("sessions_done", 0)
    skipped = payload.get("sessions_skipped", 0)
    total = payload.get("total_sessions", 0)
    tokens = payload.get("tokens_done", 0)
    total_tokens = payload.get("total_tokens", 0) or 1
    rate = payload.get("tokens_per_s")
    eta = payload.get("eta_s")
    bar_width = 32
    filled = int(bar_width * min(tokens / total_tokens, 1.0))
    bar = "#" * filled + "." * (bar_width - filled)
    lines = [
        (
            f"mode={payload.get('mode')} chunk={payload.get('prefill_chunk_tokens')} "
            f"pid={payload.get('pid')} phase={payload.get('phase')}"
        ),
        (
            f"[{bar}] {100.0 * tokens / total_tokens:5.1f}%  "
            f"sessions {done + skipped}/{total} (done {done}, skipped {skipped})"
        ),
        (
            f"tokens {tokens:,}/{total_tokens:,}  "
            f"supervised {payload.get('supervised_done', 0):,}/"
            f"{payload.get('total_supervised', 0):,}"
        ),
        (
            f"tok/s {rate if rate is not None else '-'}  "
            f"last-session tok/s {payload.get('last_session_tokens_per_s', '-')}  "
            f"peak_mlx_gb {payload.get('peak_gb', '-')}  "
            f"peak_host_rss_gb {payload.get('peak_host_rss_gb', '-')}"
        ),
        (
            f"elapsed {payload.get('elapsed_s', 0) / 3600:.2f} h  "
            f"eta {(eta / 3600) if eta else float('nan'):.2f} h  "
            f"last={payload.get('session', '-')}  ts={payload.get('ts')}"
        ),
    ]
    stream = payload.get("stream")
    if isinstance(stream, dict):
        # read_wait is the only stream number that costs wall-clock; io_gb_per_s
        # is the device rate. A span-derived "read rate" is deliberately not
        # shown -- it reads as a slow disk when the overlap is working.
        lines.append(
            f"stream: {stream.get('layer_reads')} layer reads, "
            f"{stream.get('gb_read')} GB at "
            f"{stream.get('io_gb_per_s', stream.get('read_gb_per_s'))} GB/s device, "
            f"blocked {stream.get('read_wait_seconds')} s, "
            f"convert {stream.get('convert_seconds')} s"
        )
    return "\n".join(lines)


def _cmd_monitor(opts: argparse.Namespace) -> int:
    status = (
        Path(opts.status).expanduser()
        if opts.status
        else Path(opts.out_dir).expanduser() / "status.json"
    )
    while True:
        if not status.is_file():
            print(f"no status file at {status}")
            if not opts.follow:
                return 1
        else:
            payload = json.loads(status.read_text())
            if opts.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(_format_status(payload))
            if payload.get("phase") in {"complete", "stopped"} or not opts.follow:
                return 0
        if not opts.follow:
            return 0
        time.sleep(opts.interval)
        if not opts.json:
            print("-" * 72)


# ---------------------------------------------------------------------------
# finalize / plan
# ---------------------------------------------------------------------------


def _cmd_finalize(opts: argparse.Namespace) -> int:
    from mlx_vq.quality.dsv4_teacher_runner import finalize_dsv4_calibration_imatrix

    report = finalize_dsv4_calibration_imatrix(
        Path(opts.out_dir).expanduser(),
        sidecar_dir=Path(opts.sidecar_dir).expanduser() if opts.sidecar_dir else None,
        prompt_set=opts.prompt_set,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _cmd_plan(opts: argparse.Namespace) -> int:
    from mlx_vq.quality.dsv4_teacher_runner import (
        build_dsv4_expert_span_index,
        load_dsv4_teich_pack,
    )

    spans = build_dsv4_expert_span_index(Path(opts.checkpoint).expanduser())
    sessions = load_dsv4_teich_pack(
        Path(opts.pack).expanduser(), splits=opts.split or None
    )
    tensor_bytes = sum(span.tensor_bytes for span in spans.values())
    span_bytes = sum(span.span_bytes for span in spans.values())
    per_layer = next(iter(spans.values()))
    payload = {
        "record_type": "dsv4_teacher_stream_plan",
        "layers": len(spans),
        "one_shard_per_layer": True,
        "expert_tensor_gb_per_layer": round(per_layer.tensor_bytes / 1e9, 4),
        "expert_span_gb_per_layer": round(per_layer.span_bytes / 1e9, 4),
        "span_density": round(per_layer.tensor_bytes / per_layer.span_bytes, 4),
        "coalesced_runs_per_layer": len(per_layer.runs),
        "reads_per_layer": len(per_layer.reads),
        "expert_tensor_gb_all_layers": round(tensor_bytes / 1e9, 3),
        "expert_span_gb_all_layers": round(span_bytes / 1e9, 3),
        "one_pass_seconds_at_9_9_gb_s": round(tensor_bytes / 9.9e9, 1),
        "sessions": len(sessions),
        "session_tokens_total": sum(s.token_count for s in sessions),
        "session_tokens_min": min(s.token_count for s in sessions),
        "session_tokens_max": max(s.token_count for s in sessions),
        "supervised_total": sum(s.supervised_count for s in sessions),
        "stream_gb_whole_split": round(len(sessions) * tensor_bytes / 1e9, 1),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    verbs = parser.add_subparsers(dest="verb", required=True)

    run = verbs.add_parser("run", help="produce teacher artifacts (resumable)")
    run.add_argument("--mode", required=True, choices=("calibration", "logits", "mtp-targets"))
    run.add_argument("--out-dir", required=True)
    run.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    run.add_argument("--pack", default=str(DEFAULT_PACK))
    run.add_argument(
        "--split",
        action="append",
        help="campaign_split to include; repeatable. Omit for the whole pack.",
    )
    run.add_argument("--session-id", action="append", help="only these prompt_ids")
    run.add_argument("--max-session-tokens", type=int)
    run.add_argument("--max-sessions", type=int)
    run.add_argument(
        "--chunk",
        type=int,
        default=None,
        help=(
            "prefill chunk in tokens (default 1024, the increment-2 working "
            "point). RECORDED in every artifact: chunked prefill is not "
            "bit-identical across chunk sizes, so a resumed run refuses "
            "sessions written at a different chunk."
        ),
    )
    run.add_argument("--top-k", type=int, default=2048)
    run.add_argument("--lm-head-slice", type=int, default=2048)
    run.add_argument(
        "--mtp-draft-width",
        type=int,
        default=None,
        help=(
            "mtp-targets only: draft-block width, capped at dspark_block_size "
            "(the default). The artifact scales linearly with it."
        ),
    )
    run.add_argument("--io-threads", type=int, default=8)
    run.add_argument(
        "--page-cache",
        action="store_true",
        help="do NOT set F_NOCACHE; let the ~150 GB/session stream fill the "
        "unified buffer cache (measured slower and evicts what MLX wants)",
    )
    run.add_argument("--compress", action="store_true", help="gzip the npz outputs")
    run.add_argument("--stop-file", help="graceful stop after the current session")
    run.add_argument("--wait-for-lock", action="store_true")
    run.set_defaults(func=_cmd_run)

    monitor = verbs.add_parser("monitor", help="print or follow a run's status")
    monitor.add_argument("--out-dir")
    monitor.add_argument("--status")
    monitor.add_argument("--follow", action="store_true")
    monitor.add_argument("--interval", type=float, default=30.0)
    monitor.add_argument("--json", action="store_true")
    monitor.set_defaults(func=_cmd_monitor)

    finalize = verbs.add_parser(
        "finalize", help="merge calibration sessions into imatrix sidecars"
    )
    finalize.add_argument("--out-dir", required=True)
    finalize.add_argument("--sidecar-dir")
    finalize.add_argument("--prompt-set", default="dsv4_teich_calibration_v1")
    finalize.set_defaults(func=_cmd_finalize)

    plan = verbs.add_parser("plan", help="print the streaming plan and byte budget")
    plan.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    plan.add_argument("--pack", default=str(DEFAULT_PACK))
    plan.add_argument("--split", action="append")
    plan.set_defaults(func=_cmd_plan)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    opts = parser.parse_args(argv)
    if opts.verb == "monitor" and not (opts.out_dir or opts.status):
        parser.error("monitor needs --out-dir or --status")
    return int(opts.func(opts))


if __name__ == "__main__":
    raise SystemExit(main())
