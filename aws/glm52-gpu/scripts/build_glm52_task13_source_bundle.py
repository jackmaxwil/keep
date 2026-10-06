#!/usr/bin/env python3
"""Build the deterministic local source bundle for reviewed Task 13 inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from glm52_enforcement.task13_source_bundle import (  # noqa: E402
    PinnedFileSource,
    PinnedJsonSource,
    Task13SourceBundleError,
    build_task13_source_bundle,
)

_LIVE_FLAGS = {
    "TASK11_REVIEW_APPROVAL": "task11-review-approval",
    "TASK12_REVIEW_APPROVAL": "task12-review-approval",
    "GPU_SPEND_APPROVAL": "gpu-spend-approval",
    "CAMPAIGN_DESCRIPTOR": "campaign-descriptor",
    "TASK10_WORKER_DESCRIPTOR": "task10-worker-descriptor",
    "TASK10_TASK_INPUTS": "task10-task-inputs",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--owner-approval-source", required=True, type=Path)
    parser.add_argument("--owner-approval-file-sha256", required=True)
    for flag in _LIVE_FLAGS.values():
        parser.add_argument(f"--{flag}-source", required=True, type=Path)
        parser.add_argument(f"--{flag}-file-sha256", required=True)
        parser.add_argument(f"--{flag}-body-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    values = vars(args)
    live_sources = {
        kind: PinnedJsonSource(
            path=values[flag.replace("-", "_") + "_source"],
            expected_file_sha256=values[flag.replace("-", "_") + "_file_sha256"],
            expected_body_sha256=values[flag.replace("-", "_") + "_body_sha256"],
        )
        for kind, flag in _LIVE_FLAGS.items()
    }
    try:
        build_task13_source_bundle(
            output_directory=args.output_directory,
            owner_approval_source=PinnedFileSource(
                path=args.owner_approval_source,
                expected_file_sha256=args.owner_approval_file_sha256,
            ),
            live_sources=live_sources,
        )
    except (OSError, Task13SourceBundleError, TypeError, ValueError) as error:
        print(
            f"Task 13 source-bundle build refused: {error}",
            file=sys.stderr,
        )
        return 64
    print(str(args.output_directory / "source-bundle-manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
