#!/usr/bin/env python3
"""Observe, verify, and exclusively create both Task 13 transport gates."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from glm52_enforcement.canonical import canonical_json_bytes  # noqa: E402
from glm52_enforcement.task13_transport_gates import (  # noqa: E402
    TransportGateError,
    build_t01_t25_gate,
    build_transport_mutant_gate,
    make_pytest_observer,
    validate_t01_t25_gate,
    validate_transport_mutant_gate,
)


def _write_new(path: Path, raw: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "task13/gates",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter that owns the pytest installation",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    t01_path = output_dir / "t01-t25.json"
    mutant_path = output_dir / "transport-22-mutants.json"
    try:
        if (
            not output_dir.is_dir()
            or output_dir.is_symlink()
            or t01_path.exists()
            or mutant_path.exists()
        ):
            raise TransportGateError(
                "output directory must exist and both gate paths must be absent"
            )
        first_observer = make_pytest_observer(
            repo_root=REPO_ROOT,
            python_executable=args.python,
        )
        t01_gate = build_t01_t25_gate(
            repo_root=REPO_ROOT,
            observe=first_observer,
        )
        mutant_gate = build_transport_mutant_gate(
            repo_root=REPO_ROOT,
            observe=first_observer,
        )

        second_observer = make_pytest_observer(
            repo_root=REPO_ROOT,
            python_executable=args.python,
        )
        validate_t01_t25_gate(
            t01_gate,
            repo_root=REPO_ROOT,
            observe=second_observer,
        )
        validate_transport_mutant_gate(
            mutant_gate,
            repo_root=REPO_ROOT,
            observe=second_observer,
        )

        _write_new(t01_path, canonical_json_bytes(t01_gate) + b"\n")
        try:
            _write_new(
                mutant_path,
                canonical_json_bytes(mutant_gate) + b"\n",
            )
        except BaseException:
            t01_path.unlink()
            raise
    except (OSError, TransportGateError) as error:
        print(f"Task 13 transport gates refused: {error}", file=sys.stderr)
        return 64
    print(str(t01_path))
    print(str(mutant_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
