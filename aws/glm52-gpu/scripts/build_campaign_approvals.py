#!/usr/bin/env python3
"""Build the two authenticated GLM-5.2 campaign approval records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from glm52_enforcement.approvals import (  # noqa: E402
    SOURCE_AUTHORITY_PATH,
    build_gpu_residual_liability_approval,
    build_production_support_plane_approval,
    validate_gpu_residual_liability_approval,
    validate_production_support_plane_approval,
)
from glm52_enforcement.canonical import canonical_json_bytes  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-output", type=Path, required=True)
    parser.add_argument("--residual-output", type=Path, required=True)
    return parser


def _write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as output:
        output.write(payload)


def main(argv: List[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_paths = (args.support_output, args.residual_output)
    if output_paths[0] == output_paths[1]:
        raise SystemExit("approval output paths must be distinct")
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite existing output: {existing}")

    source_bytes = (REPO_ROOT / SOURCE_AUTHORITY_PATH).read_bytes()
    support = build_production_support_plane_approval(source_bytes)
    residual = build_gpu_residual_liability_approval(source_bytes)
    validate_production_support_plane_approval(support, source_bytes)
    validate_gpu_residual_liability_approval(residual, source_bytes)

    payloads = (
        canonical_json_bytes(support) + b"\n",
        canonical_json_bytes(residual) + b"\n",
    )
    for path, payload in zip(output_paths, payloads):
        _write_new(path, payload)

    validate_production_support_plane_approval(
        json.loads(output_paths[0].read_bytes()), source_bytes
    )
    validate_gpu_residual_liability_approval(
        json.loads(output_paths[1].read_bytes()), source_bytes
    )

    outputs: List[Dict[str, object]] = []
    for path in output_paths:
        raw = path.read_bytes()
        outputs.append(
            {
                "path": str(path),
                "byte_size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    print(json.dumps({"outputs": outputs}, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
