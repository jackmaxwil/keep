#!/usr/bin/env python3
"""Emit the fixed post-stop observation for the parameterless SSM document."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import NoReturn


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[2]
SRC_ROOT = REPO_ROOT / "src"
for root in (SCRIPT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from glm52_enforcement.canonical import canonical_json_bytes  # noqa: E402
from materialize_task10_graceful_stop import _build_evidence  # noqa: E402


def _fail(message: str) -> NoReturn:
    print("Task 10 SSM graceful observation: " + message, file=sys.stderr)
    raise SystemExit(70)


def main(argv: list[str]) -> int:
    if argv:
        _fail("this collector accepts no arguments")
    try:
        evidence = _build_evidence(
            authority="LOCAL_TIMER",
            ssm_command_id=None,
        )
        sys.stdout.buffer.write(canonical_json_bytes(evidence) + b"\n")
        sys.stdout.buffer.flush()
    except (OSError, TypeError, ValueError) as exc:
        _fail(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
