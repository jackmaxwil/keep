#!/usr/bin/env python3
"""Normalize SkyPilot 0.13's empty managed-jobs queue into JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EMPTY_QUEUE_ERROR = (
    "sky.exceptions.ClusterNotUpError: No in-progress managed jobs."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    stdout = args.stdout.read_text()
    stderr = args.stderr.read_text()
    if args.exit_code != 0:
        if EMPTY_QUEUE_ERROR in f"{stdout}\n{stderr}":
            args.output.write_text("[]\n")
            return 0
        sys.stderr.write(stderr or stdout)
        return args.exit_code if 0 < args.exit_code < 256 else 1

    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise SystemExit("SkyPilot jobs queue did not return JSON") from error
    if not isinstance(value, (dict, list)):
        raise SystemExit("SkyPilot jobs queue JSON must be an object or list")
    args.output.write_text(json.dumps(value, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
