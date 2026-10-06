#!/usr/bin/env python3
"""Build the exact Task 13 prequalification request and reviewed-artifact list."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from glm52_enforcement.canonical import canonical_json_bytes  # noqa: E402
from glm52_enforcement.task13_campaign_package import (  # noqa: E402
    CampaignPackageError,
    build_campaign_package,
)
from glm52_enforcement.task13_transport_gates import (  # noqa: E402
    SEMANTIC_GATE_PINS,
)


def _read(path: Path, label: str) -> object:
    if not path.is_file() or path.is_symlink():
        raise CampaignPackageError(label + " must be a regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CampaignPackageError(label + " is not JSON") from error
    if raw != canonical_json_bytes(value) + b"\n":
        raise CampaignPackageError(label + " is not canonical JSON plus LF")
    return value


def _write_new(path: Path, value: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, canonical_json_bytes(value) + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-request", required=True, type=Path)
    parser.add_argument("--artifact-coordinates", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--reviewed-artifacts", required=True, type=Path)
    args = parser.parse_args()
    try:
        base = _read(args.base_request, "base request")
        coordinates = _read(args.artifact_coordinates, "artifact coordinates")
        if type(base) is not dict or "artifacts" in base:
            raise CampaignPackageError("base request must omit the reviewed artifacts")
        if type(coordinates) is not list:
            raise CampaignPackageError("artifact coordinates must be a list")
        request = {**base, "artifacts": coordinates}
        package = build_campaign_package(request)
        if (
            package["package_phase"] != "PREQUALIFICATION"
            or package["semantic_transport_gates"] != SEMANTIC_GATE_PINS
        ):
            raise CampaignPackageError("prequalification semantic gate closure drifted")
        reviewed = package["reviewed_artifacts"]
        if args.request.exists() or args.reviewed_artifacts.exists():
            raise FileExistsError("output already exists")
        _write_new(args.request, request)
        try:
            _write_new(args.reviewed_artifacts, reviewed)
        except BaseException:
            args.request.unlink()
            raise
    except (CampaignPackageError, FileExistsError, OSError) as error:
        print(f"Task 13 prequalification inputs refused: {error}", file=sys.stderr)
        return 64
    print(str(args.request))
    print(str(args.reviewed_artifacts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
