"""CLI: produce top-K teacher captures for the teich coding-agent corpus.

Personal-model production path (contract-bypass): see
:mod:`mlx_vq.quality.glm52_teich_teacher_producer`.

Example (GPU node):
    python benchmarks/produce_glm52_teich_teacher_cache.py \
        --snapshot-dir /mnt/nvme/source-snapshot \
        --non-vq-package-dir /mnt/nvme/non-vq-package \
        --teich-pack /mnt/nvme/teich-pack/glm52-coding-agent-initial-v2-20260713.json \
        --out-dir /mnt/nvme/teacher-cache-out \
        --heartbeat /mnt/nvme/teacher-cache-out/heartbeat.json

Monitor from anywhere:
    watch -n 30 cat /mnt/nvme/teacher-cache-out/heartbeat.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--non-vq-package-dir", required=True)
    parser.add_argument("--teich-pack", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--heartbeat", required=True)
    parser.add_argument("--top-k", type=int, default=2048)
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=131_072,
        help="padded-batch token budget per forward pass (batch * max_len)",
    )
    parser.add_argument("--lm-head-slice", type=int, default=4096)
    parser.add_argument(
        "--query-chunk-size",
        type=int,
        default=512,
        help=(
            "query-axis chunk size for DSA attention on long sequences. "
            "Sequences long enough to engage DSA use a memory-bounded, "
            "gather-based path chunked over queries (dense attention's "
            "O(seq^2) score matrices crash past ~37K tokens); short "
            "sequences are unaffected (byte-identical dense path). Lower "
            "this if a run still hits an out-of-memory error."
        ),
    )
    parser.add_argument(
        "--max-session-tokens",
        type=int,
        default=None,
        help="only produce sessions with token_count <= this (validation runs)",
    )
    parser.add_argument(
        "--session-ids",
        nargs="*",
        default=None,
        help="only produce these prompt_ids (validation runs)",
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-s3-prefix")
    parser.add_argument("--checkpoint-run-id")
    parser.add_argument("--code-tar-sha256")
    parser.add_argument("--stop-file", type=Path)
    deadline_group = parser.add_mutually_exclusive_group()
    deadline_group.add_argument(
        "--capacity-block-end",
        help="timezone-aware ISO-8601 Capacity Block end; T-60m starts drain",
    )
    deadline_group.add_argument(
        "--execution-deadline",
        help="timezone-aware ISO-8601 orchestrator deadline; T-60m starts drain",
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    from mlx_vq.quality.glm52_teich_teacher_producer import (
        TeacherStopRequested,
        run_teich_teacher_production,
    )
    from mlx_vq.quality.glm52_teich_checkpoint import (
        TeacherCheckpointConfig,
        TeacherRunIdentity,
    )

    session_filter = None
    if args.session_ids or args.max_session_tokens is not None:
        wanted = set(args.session_ids or [])

        def session_filter(session):  # noqa: ANN001
            if wanted and session.prompt_id not in wanted:
                return False
            if (
                args.max_session_tokens is not None
                and session.token_count > args.max_session_tokens
            ):
                return False
            return True

    checkpoint_config = None
    if args.checkpoint_dir is not None:
        if not args.checkpoint_run_id or not args.code_tar_sha256:
            parser.error(
                "--checkpoint-dir requires --checkpoint-run-id and --code-tar-sha256"
            )

        def sha(raw: bytes) -> str:
            return hashlib.sha256(raw).hexdigest()

        snapshot = Path(args.snapshot_dir)
        non_vq = Path(args.non_vq_package_dir)
        model_sha = sha(
            json.dumps(
                {
                    "config_sha256": sha((snapshot / "config.json").read_bytes()),
                    "index_sha256": sha(
                        (snapshot / "model.safetensors.index.json").read_bytes()
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        generation = {
            "record_type": "glm52_teich_full_v2_generation_config_v1",
            "top_k": args.top_k,
            "lm_head_slice": args.lm_head_slice,
            "query_chunk_size": args.query_chunk_size,
            "max_batch_tokens": args.max_batch_tokens,
            "router_layers": list(range(70, 78)),
            "cka_probe_layers": [77],
        }
        generation_sha = sha(
            json.dumps(
                generation, sort_keys=True, separators=(",", ":")
            ).encode()
        )
        capacity_block_deadline = None
        execution_deadline = None
        if args.capacity_block_end:
            deadline = datetime.fromisoformat(
                args.capacity_block_end.replace("Z", "+00:00")
            )
            if deadline.tzinfo is None:
                parser.error("--capacity-block-end must include a timezone")
            capacity_block_deadline = deadline
        elif args.execution_deadline:
            deadline = datetime.fromisoformat(
                args.execution_deadline.replace("Z", "+00:00")
            )
            if deadline.tzinfo is None:
                parser.error("--execution-deadline must include a timezone")
            execution_deadline = deadline
        identity = TeacherRunIdentity(
            run_id=args.checkpoint_run_id,
            model_sha256=model_sha,
            non_vq_package_sha256=sha(
                (non_vq / "non-vq-manifest.json").read_bytes()
            ),
            prompt_pack_sha256=sha(Path(args.teich_pack).read_bytes()),
            code_sha256=args.code_tar_sha256,
            generation_config_sha256=generation_sha,
        )
        checkpoint_config = TeacherCheckpointConfig(
            local_checkpoint_dir=args.checkpoint_dir,
            s3_checkpoint_prefix=args.checkpoint_s3_prefix,
            stop_file=args.stop_file,
            capacity_block_deadline=capacity_block_deadline,
            execution_deadline=execution_deadline,
            resume=not args.no_resume,
            identity=identity,
        )

        if args.stop_file is not None:
            def request_stop(_signum, _frame):  # noqa: ANN001
                args.stop_file.parent.mkdir(parents=True, exist_ok=True)
                args.stop_file.touch()

            signal.signal(signal.SIGTERM, request_stop)
            signal.signal(signal.SIGINT, request_stop)

    try:
        result = run_teich_teacher_production(
            snapshot_dir=args.snapshot_dir,
            non_vq_package_dir=args.non_vq_package_dir,
            pack_path=args.teich_pack,
            out_dir=args.out_dir,
            heartbeat_path=args.heartbeat,
            top_k=args.top_k,
            max_batch_tokens=args.max_batch_tokens,
            lm_head_slice=args.lm_head_slice,
            query_chunk_size=args.query_chunk_size,
            session_filter=session_filter,
            checkpoint_config=checkpoint_config,
        )
    except TeacherStopRequested as error:
        print(json.dumps({"status": "checkpointed-stop", "reason": str(error)}))
        return 75
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
