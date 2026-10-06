#!/usr/bin/env python3
"""Originate one H100 recovery job through the guarded submitter."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from glm52_enforcement.h100_live_route import (  # noqa: E402
    GuardedQualificationError,
    QualificationAuthorities,
    resolve_qualification_authorities,
    run_guarded_qualification,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--sky-bin", required=True, type=Path)
    parser.add_argument("--submitter", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--profile", required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    resolver: Callable[..., QualificationAuthorities] = (
        resolve_qualification_authorities
    ),
    route: Callable[..., dict[str, object]] = run_guarded_qualification,
) -> int:
    args = _parser().parse_args(argv)
    try:
        authorities = resolver(
            descriptor=args.descriptor,
            config=args.config,
            repo_root=REPO_ROOT,
            workspace=args.workspace,
            materialization_dir=args.work_root / "authorities",
        )
        outcome = route(
            authorities=authorities,
            submitter=args.submitter,
            sky_bin=args.sky_bin,
            profile=args.profile,
            work_dir=args.work_root / "submission",
        )
    except (GuardedQualificationError, OSError) as error:
        print(f"H100 guarded route refused: {error}", file=sys.stderr)
        return 70
    print(
        json.dumps(
            outcome,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
