#!/usr/bin/env python3
"""Render one canonical mount-free Task 10 SkyPilot YAML."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from glm52_enforcement.canonical import canonical_json_bytes  # noqa: E402
from glm52_enforcement.task10_worker import (  # noqa: E402
    MountFreeTaskInputs,
    render_mount_free_task,
    validate_task_inputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    raw = args.inputs.read_bytes()
    value = json.loads(raw)
    if (
        type(value) is not dict
        or raw != canonical_json_bytes(value) + b"\n"
        or set(value) != set(MountFreeTaskInputs.__dataclass_fields__)
    ):
        raise SystemExit("Task 10 input JSON is not exact and canonical")
    task = render_mount_free_task(
        validate_task_inputs(MountFreeTaskInputs(**value))
    ).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        args.output,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(task)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            args.output.unlink()
        except FileNotFoundError:
            pass
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
