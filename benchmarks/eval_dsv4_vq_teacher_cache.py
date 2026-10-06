"""Teacher-agreement eval for the DeepSeek-V4-Flash VQ artifact.

The Wave 5 quality gate. Streams the 2.031 bpw VQ artifact layer-sequentially
through the real 43-layer model, forwards the teacher's own token sequences at
the teacher's own prefill chunk size, and compares the resulting logits against
the captured top-2048 teacher logits at every supervised position.

Verbs::

    plan        what would run, and what it should cost, before committing
    run         the eval; resumable, one JSONL row per session
    control     the same eval with the unmodified source experts (no VQ)
    monitor     print a run directory's status the house way
    summarize   recompute the split summary and the verdict from the JSONL

Heavy verbs (``run``, ``control``) take ``.keep-heavy-job.lock``.

The control is not optional garnish. "Mean KLD 0.2" says nothing until the
source model's own KLD against the same teacher cache is on the page next to it:
the teacher was produced by this same code at this same chunk size, so the
control measures the floor contributed by float non-determinism, the fp16 logit
storage and the harness itself, and the artifact's number is only interpretable
as the excess over that floor.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

from mlx_vq.quality.dsv4_teacher_agreement import (  # noqa: E402
    DEFAULT_POSITION_SLICE,
    agreement_verdict,
    heavy_job_lock,
    read_agreement_rows,
    run_dsv4_teacher_agreement,
    select_teacher_sessions,
    summarize_agreement_records,
    vq_artifact_identity,
)
from mlx_vq.quality.dsv4_teacher_runner import (  # noqa: E402
    DEFAULT_IO_THREADS,
    DEFAULT_PREFILL_CHUNK_TOKENS,
)

DEFAULT_CHECKPOINT = Path.home() / "models" / "DeepSeek-V4-Flash-0731"
DEFAULT_ARTIFACT = Path.home() / "keep-artifacts" / "dsv4-vq-e8p-g512"
DEFAULT_TEACHER = Path.home() / "keep-artifacts" / "dsv4-teacher-logits-eval"
DEFAULT_PACK = Path.home() / "models" / "teich" / "dsv4-coding-agent-v1-20260811.json"
DEFAULT_OUTPUT = Path.home() / "keep-artifacts" / "dsv4-vq-quality-gate"
HEAVY_JOB_LOCK = REPO_ROOT / ".keep-heavy-job.lock"

VQ_ENGINE = "vq_e8p_streamed"
SOURCE_ENGINE = "source_mxfp4_streamed"


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def command_plan(args: argparse.Namespace) -> dict[str, Any]:
    sessions = select_teacher_sessions(
        args.teacher_dir,
        split=args.split,
        prompt_ids=args.prompt_id or None,
        stratify=args.stratify,
        max_sessions=args.max_sessions,
    )
    identity = vq_artifact_identity(args.artifact_dir)
    tokens = sum(session.token_count for session in sessions)
    positions = sum(session.supervised_count for session in sessions)
    # The teacher's own measured rate over the same sessions is the only honest
    # basis for an estimate here; a throughput number from a different split or a
    # different expert mode is a guess dressed as a measurement.
    rate = float(args.assumed_tokens_per_s)
    payload = {
        "record_type": "dsv4_teacher_agreement_plan_v1",
        "split": args.split,
        "sessions": len(sessions),
        "tokens": tokens,
        "supervised_positions": positions,
        "assumed_tokens_per_s": rate,
        "estimated_forward_hours": round(tokens / rate / 3600.0, 2),
        "prefill_chunk_tokens": int(args.chunk),
        "teacher_prefill_chunks": sorted(
            {session.prefill_chunk_tokens for session in sessions}
        ),
        "teacher_generation_config_sha256": sorted(
            {session.generation_config_sha256 for session in sessions}
        ),
        "artifact": {
            key: value for key, value in identity.items() if key != "manifest"
        },
        "rows": [
            {
                "prompt_id": session.prompt_id,
                "split": session.campaign_split,
                "tokens": session.token_count,
                "supervised": session.supervised_count,
                "teacher_top_k": session.top_k,
                "mean_teacher_tail_mass": round(float(session.tail_mass.mean()), 8),
            }
            for session in sessions
        ],
    }
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2, sort_keys=True))
    print(f"{len(sessions)} session(s), {tokens} tokens, {positions} supervised positions")
    print(f"estimated forward wall at {rate:.1f} tok/s: {payload['estimated_forward_hours']} h")
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    return payload


# ---------------------------------------------------------------------------
# run / control
# ---------------------------------------------------------------------------


def _run(args: argparse.Namespace, engine: str) -> dict[str, Any]:
    summary = run_dsv4_teacher_agreement(
        teacher_dir=args.teacher_dir,
        pack_path=args.pack,
        out_dir=args.output,
        artifact_dir=args.artifact_dir if engine == VQ_ENGINE else None,
        checkpoint_dir=args.checkpoint,
        engine=engine,
        split=args.split,
        prompt_ids=args.prompt_id or None,
        stratify=args.stratify,
        max_sessions=args.max_sessions,
        chunk=args.chunk,
        position_slice=args.position_slice,
        io_threads=args.io_threads,
        nocache=not args.no_nocache,
        append_jsonl=args.append_jsonl,
        status_path=args.status_path,
        save_token_arrays=not args.no_token_arrays,
        stop_file=args.stop_file,
    )
    verdict = agreement_verdict(
        summary,
        mean_kld_max=args.mean_kld_max,
        p999_kld_max=args.p999_kld_max,
        top1_min=args.top1_min,
    )
    summary["verdict"] = verdict
    print(json.dumps(_headline(summary), indent=2, sort_keys=True))
    return summary


def command_run(args: argparse.Namespace) -> dict[str, Any]:
    return _run(args, VQ_ENGINE)


def command_control(args: argparse.Namespace) -> dict[str, Any]:
    return _run(args, SOURCE_ENGINE)


def _headline(summary: dict[str, Any]) -> dict[str, Any]:
    weighted = summary.get("position_weighted", {})
    return {
        "engine": summary.get("engine"),
        "row_count": summary.get("row_count"),
        "clean_row_count": summary.get("clean_row_count"),
        "total_supervised_positions": summary.get("total_supervised_positions"),
        "position_weighted_mean_kld_lower": weighted.get("mean_kld_lower"),
        "position_weighted_mean_kld_head": weighted.get("mean_kld_head"),
        "position_weighted_mean_kld_upper": weighted.get("mean_kld_upper"),
        "position_weighted_top1_agreement": weighted.get("top1_agreement"),
        "position_weighted_top5_agreement": weighted.get("top5_agreement"),
        "position_weighted_top10_agreement": weighted.get("top10_agreement"),
        "position_weighted_vq_nll": weighted.get("vq_nll"),
        "position_weighted_teacher_nll": weighted.get("teacher_nll"),
        "position_weighted_ppl_ratio": weighted.get("position_weighted_ppl_ratio")
        or weighted.get("ppl_ratio"),
        "token_tail": summary.get("token_tail"),
        "verdict": summary.get("verdict"),
    }


# ---------------------------------------------------------------------------
# monitor / summarize
# ---------------------------------------------------------------------------


def command_monitor(args: argparse.Namespace) -> dict[str, Any]:
    status_path = Path(args.status_path) if args.status_path else Path(args.output) / "status.json"
    if not status_path.is_file():
        print(f"no status.json at {status_path}")
        return {}
    payload = json.loads(status_path.read_text())
    done = payload.get("sessions_done", 0) + payload.get("sessions_skipped", 0)
    print(
        f"engine={payload.get('engine')} phase={payload.get('phase')} "
        f"{done}/{payload.get('total_sessions')} sessions  "
        f"{payload.get('tokens_done')}/{payload.get('total_tokens')} tokens  "
        f"elapsed={payload.get('elapsed_s')}s  rate={payload.get('tokens_per_s')} tok/s  "
        f"eta={payload.get('eta_h')}h"
    )
    print(f"eta basis: {payload.get('eta_basis')}")
    if payload.get("prompt_id"):
        print(f"current: {payload.get('prompt_id')}")
    if args.verbose:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def command_summarize(args: argparse.Namespace) -> dict[str, Any]:
    rows_path = Path(args.append_jsonl) if args.append_jsonl else Path(args.output) / "rows.jsonl"
    rows = read_agreement_rows(rows_path)
    engines = sorted({str(row.get("engine")) for row in rows})
    out: dict[str, Any] = {
        "record_type": "dsv4_teacher_agreement_report_v1",
        "rows_jsonl": str(rows_path),
        "engines": engines,
        "by_engine": {},
    }
    token_dir = Path(args.output) / "token-arrays"
    for engine in engines:
        engine_rows = [row for row in rows if row.get("engine") == engine]
        summary = summarize_agreement_records(
            engine_rows, token_array_dir=token_dir if token_dir.is_dir() else None
        )
        summary["engine"] = engine
        summary["verdict"] = agreement_verdict(
            summary,
            mean_kld_max=args.mean_kld_max,
            p999_kld_max=args.p999_kld_max,
            top1_min=args.top1_min,
        )
        out["by_engine"][engine] = summary
    if len(engines) == 2 and VQ_ENGINE in engines and SOURCE_ENGINE in engines:
        vq = out["by_engine"][VQ_ENGINE]["position_weighted"]
        source = out["by_engine"][SOURCE_ENGINE]["position_weighted"]
        shared = sorted(
            {row["prompt_id"] for row in rows if row.get("engine") == VQ_ENGINE}
            & {row["prompt_id"] for row in rows if row.get("engine") == SOURCE_ENGINE}
        )
        paired: dict[str, Any] = {"shared_prompt_ids": shared}
        if shared:
            for engine, label in ((VQ_ENGINE, "vq"), (SOURCE_ENGINE, "source")):
                subset = [
                    row
                    for row in rows
                    if row.get("engine") == engine and row["prompt_id"] in shared
                ]
                paired[label] = summarize_agreement_records(
                    subset, token_array_dir=token_dir if token_dir.is_dir() else None
                )["position_weighted"]
            paired["excess_mean_kld_lower"] = (
                paired["vq"]["mean_kld_lower"] - paired["source"]["mean_kld_lower"]
            )
            paired["top1_agreement_gap"] = (
                paired["source"]["top1_agreement"] - paired["vq"]["top1_agreement"]
            )
        out["vq_vs_source_on_shared_sessions"] = paired
        out["whole_split_note"] = (
            "vq and source were not necessarily run on the same session set; the "
            "paired block above is the only comparison that controls for that"
        )
    print(json.dumps(out, indent=2, sort_keys=True))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--teacher-dir", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", default="report")
    parser.add_argument("--prompt-id", action="append")
    parser.add_argument(
        "--stratify",
        type=int,
        default=None,
        help="evenly spaced subset over the length-sorted split",
    )
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=DEFAULT_PREFILL_CHUNK_TOKENS)
    parser.add_argument("--position-slice", type=int, default=DEFAULT_POSITION_SLICE)
    parser.add_argument("--io-threads", type=int, default=DEFAULT_IO_THREADS)
    parser.add_argument("--no-nocache", action="store_true")
    parser.add_argument("--append-jsonl", type=Path, default=None)
    parser.add_argument("--status-path", type=Path, default=None)
    parser.add_argument("--stop-file", type=Path, default=None)
    parser.add_argument("--no-token-arrays", action="store_true")


def _add_thresholds(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mean-kld-max", type=float, default=0.30)
    parser.add_argument("--p999-kld-max", type=float, default=3.0)
    parser.add_argument("--top1-min", type=float, default=0.85)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-lock", action="store_true", help="skip the heavy-job lock")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="what would run and what it should cost")
    _add_common(plan)
    plan.add_argument("--assumed-tokens-per-s", type=float, default=78.2)
    plan.add_argument("--output-json", type=Path, default=None)
    plan.set_defaults(func=command_plan, heavy=False)

    run = sub.add_parser("run", help="eval the VQ artifact against the teacher cache")
    _add_common(run)
    _add_thresholds(run)
    run.set_defaults(func=command_run, heavy=True)

    control = sub.add_parser(
        "control", help="eval the unmodified source experts against the same cache"
    )
    _add_common(control)
    _add_thresholds(control)
    control.set_defaults(func=command_control, heavy=True)

    monitor = sub.add_parser("monitor", help="print a run directory's status")
    monitor.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    monitor.add_argument("--status-path", type=Path, default=None)
    monitor.add_argument("--verbose", action="store_true")
    monitor.set_defaults(func=command_monitor, heavy=False)

    summarize = sub.add_parser(
        "summarize", help="recompute the summary and verdict from the JSONL rows"
    )
    summarize.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    summarize.add_argument("--append-jsonl", type=Path, default=None)
    summarize.add_argument("--output-json", type=Path, default=None)
    _add_thresholds(summarize)
    summarize.set_defaults(func=command_summarize, heavy=False)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    holder = f"dsv4-vq-quality-gate:{args.command}:{os.getpid()}"
    with heavy_job_lock(
        HEAVY_JOB_LOCK,
        enabled=getattr(args, "heavy", False) and not args.no_lock,
        holder=holder,
    ):
        args.func(args)


if __name__ == "__main__":
    main()
