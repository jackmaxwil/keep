#!/usr/bin/env python3
"""Build or check the canonical H.1g Task 9 launch-custody contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from glm52_enforcement.task9_contract import build_task9_contract  # noqa: E402


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "aws/glm52-gpu/cfn/h1g/task9-contract-v1.json",
    )
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    expected = _canonical_bytes(build_task9_contract(REPO_ROOT))
    output = arguments.output.resolve()
    if arguments.check:
        if not output.is_file() or output.read_bytes() != expected:
            raise SystemExit("Task 9 contract artifact is stale")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
