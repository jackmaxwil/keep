#!/usr/bin/env python3
"""Build one exact local Task 10 to Task 13 launch bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER_RELATIVE = "aws/glm52-gpu/scripts/run_glm52_task13_campaign.py"
COORDINATOR_RELATIVE = "aws/glm52-gpu/scripts/glm52_task13_production_coordinator.py"
sys.path.insert(0, str(REPO_ROOT / "src"))

from glm52_enforcement.canonical import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from glm52_enforcement.task10_task13_bridge import (  # noqa: E402
    Task13BridgeError,
    task13_launch_bridge_from_mapping,
)
from glm52_enforcement.task13_authority_materialization import (  # noqa: E402
    AuthorityMaterializationError,
    validate_public_task10_inputs_manifest,
)


def _read(path: Path, label: str) -> tuple[object, bytes]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise Task13BridgeError(label + " must be an absolute regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Task13BridgeError(label + " is not JSON") from error
    if raw != canonical_json_bytes(value) + b"\n":
        raise Task13BridgeError(label + " is not canonical JSON plus LF")
    return value, raw


def _write_new(path: Path, value: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, canonical_json_bytes(value) + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _public_inputs(
    path: Path,
) -> tuple[dict[str, object], str, dict[str, object]]:
    value, _raw = _read(path, "public Task10 inputs")
    try:
        manifest = validate_public_task10_inputs_manifest(value)
    except AuthorityMaterializationError as error:
        raise Task13BridgeError(str(error)) from error
    h100, _h100_raw = _read(
        Path(str(manifest["h100_resume_ready_coordinate_path"])),
        "H100 resume-ready coordinate",
    )
    if type(h100) is not dict:
        raise Task13BridgeError(
            "H100 resume-ready coordinate is not an object"
        )
    return (
        h100,
        str(manifest["task13_route_binding_identity_sha256"]),
        manifest,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--reviewed-artifacts", required=True, type=Path)
    parser.add_argument("--controller-authority", required=True, type=Path)
    parser.add_argument("--public-task10-inputs", type=Path)
    parser.add_argument("--h100-resume-ready-coordinate", type=Path)
    parser.add_argument("--task13-route-binding-identity")
    parser.add_argument("--fence-journal", required=True, type=Path)
    parser.add_argument("--support-journal", required=True, type=Path)
    parser.add_argument("--campaign-journal", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        package, package_raw = _read(args.package, "package")
        reviewed, reviewed_raw = _read(
            args.reviewed_artifacts,
            "reviewed artifacts",
        )
        controller, controller_raw = _read(
            args.controller_authority,
            "controller authority",
        )
        public_manifest = None
        if args.public_task10_inputs is not None:
            if (
                args.h100_resume_ready_coordinate is not None
                or args.task13_route_binding_identity is not None
            ):
                raise Task13BridgeError(
                    "public Task10 inputs cannot be mixed with legacy inputs"
                )
            h100, route_identity, public_manifest = _public_inputs(
                args.public_task10_inputs
            )
        else:
            if (
                args.h100_resume_ready_coordinate is None
                or args.task13_route_binding_identity is None
            ):
                raise Task13BridgeError(
                    "one public Task10 manifest or both legacy inputs "
                    "are required"
                )
            h100, _h100_raw = _read(
                args.h100_resume_ready_coordinate,
                "H100 resume-ready coordinate",
            )
            route_identity = args.task13_route_binding_identity
        if (
            type(package) is not dict
            or type(reviewed) is not list
            or package.get("reviewed_artifacts") != reviewed
            or type(controller) is not dict
            or type(h100) is not dict
            or controller.get("activation_id") != package.get("activation_id")
        ):
            raise Task13BridgeError("Task 13 launch inputs disagree")
        if public_manifest is not None:
            task10_coordinates = [
                item
                for item in reviewed
                if type(item) is dict
                and item.get("artifact_kind")
                == "TASK10_PRODUCTION_AUTHORITY"
            ]
            if task10_coordinates != [
                public_manifest["task10_authority_coordinate"]
            ]:
                raise Task13BridgeError(
                    "public Task10 authority coordinate disagrees"
                )
        journals = (
            args.fence_journal.resolve(strict=False),
            args.support_journal.resolve(strict=False),
            args.campaign_journal.resolve(strict=False),
        )
        if (
            len(set(journals)) != 3
            or any(not path.is_absolute() for path in journals)
            or any(not path.parent.is_dir() for path in journals)
        ):
            raise Task13BridgeError("journal paths are not exact")
        runner = REPO_ROOT / RUNNER_RELATIVE
        coordinator = REPO_ROOT / COORDINATOR_RELATIVE
        body = {
            "schema_version": 1,
            "record_type": "glm52_task10_task13_launch_bridge_v1",
            "account_id": "246813579024",
            "region": "us-west-2",
            "profile": "keep-gpu",
            "run_id": "glm52-sky-20260724",
            "activation_id": package["activation_id"],
            "package_path": str(args.package),
            "package_file_sha256": hashlib.sha256(package_raw).hexdigest(),
            "package_body_sha256": canonical_sha256(package),
            "reviewed_artifacts_path": str(args.reviewed_artifacts),
            "reviewed_artifacts_file_sha256": hashlib.sha256(reviewed_raw).hexdigest(),
            "reviewed_artifacts_body_sha256": canonical_sha256(reviewed),
            "controller_authority_path": str(args.controller_authority),
            "controller_authority_file_sha256": hashlib.sha256(
                controller_raw
            ).hexdigest(),
            "controller_authority_body_sha256": canonical_sha256(controller),
            "fence_journal_path": str(journals[0]),
            "support_journal_path": str(journals[1]),
            "campaign_journal_path": str(journals[2]),
            "runner_relative_path": RUNNER_RELATIVE,
            "runner_file_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
            "coordinator_relative_path": COORDINATOR_RELATIVE,
            "coordinator_file_sha256": hashlib.sha256(
                coordinator.read_bytes()
            ).hexdigest(),
            "h100_resume_ready_bucket": h100["bucket"],
            "h100_resume_ready_key": h100["key"],
            "h100_resume_ready_version_id": h100["version_id"],
            "h100_resume_ready_file_sha256": h100["file_sha256"],
            "h100_resume_ready_body_sha256": h100["body_sha256"],
            "task13_route_binding_identity_sha256": (
                route_identity
            ),
        }
        value = {
            **body,
            "canonical_identity_sha256": canonical_sha256(body),
        }
        task13_launch_bridge_from_mapping(value)
        _write_new(args.output, value)
    except (KeyError, OSError, Task13BridgeError) as error:
        print(f"Task 13 launch bridge refused: {error}", file=sys.stderr)
        return 64
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
