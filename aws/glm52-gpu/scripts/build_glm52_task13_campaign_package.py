#!/usr/bin/env python3
"""Build one deterministic, disabled, no-execute Task 13 package."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from glm52_enforcement.canonical import canonical_json_bytes  # noqa: E402
from glm52_enforcement.task13_campaign_package import (  # noqa: E402
    CampaignPackageError,
    build_campaign_package,
    canonical_campaign_package_bytes,
)


def _read_canonical_request(path: Path) -> object:
    if not path.is_file() or path.is_symlink():
        raise CampaignPackageError("request must be a regular non-symlink file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignPackageError("request is not JSON") from error
    if raw != canonical_json_bytes(value) + b"\n":
        raise CampaignPackageError("request is not canonical JSON plus LF")
    return value


def _write_new(path: Path, raw: bytes) -> None:
    if not path.is_absolute():
        raise CampaignPackageError("output must be an absolute path")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise CampaignPackageError("output parent must be an existing directory")
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
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--predecessor-package", type=Path)
    parser.add_argument("--predecessor-reviewed-artifacts", type=Path)
    parser.add_argument("--clean-rehearsal-evidence", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        request = _read_canonical_request(args.request)
        if (
            (args.predecessor_package is None)
            != (args.predecessor_reviewed_artifacts is None)
        ):
            raise CampaignPackageError(
                "both predecessor inputs are required together"
            )
        predecessor_package = (
            _read_canonical_request(args.predecessor_package)
            if args.predecessor_package is not None
            else None
        )
        predecessor_artifacts = (
            _read_canonical_request(args.predecessor_reviewed_artifacts)
            if args.predecessor_reviewed_artifacts is not None
            else None
        )
        clean_rehearsal_evidence = (
            _read_canonical_request(args.clean_rehearsal_evidence)
            if args.clean_rehearsal_evidence is not None
            else None
        )
        package = build_campaign_package(
            request,
            predecessor_package=predecessor_package,
            predecessor_reviewed_artifacts=predecessor_artifacts,
            clean_rehearsal_evidence=clean_rehearsal_evidence,
        )
        _write_new(args.output, canonical_campaign_package_bytes(package))
    except (CampaignPackageError, FileExistsError, OSError) as error:
        print("Task 13 package refused: %s" % error, file=sys.stderr)
        return 64
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
