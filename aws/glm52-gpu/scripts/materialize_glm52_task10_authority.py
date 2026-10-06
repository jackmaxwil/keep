#!/usr/bin/env python3
"""Materialize the acyclic Task10 authority before the final Task13 package."""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Mapping


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from glm52_enforcement.canonical import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from glm52_enforcement.task10_worker import (  # noqa: E402
    MountFreeTaskInputs,
    worker_bootstrap_descriptor_from_mapping,
)
from glm52_enforcement.glm52_h100_qualification import (  # noqa: E402
    validate_h100_resume_ready,
)
from glm52_enforcement.task11_boundary import (  # noqa: E402
    task11_boundary_from_bytes,
)
from glm52_enforcement.task13_authority_materialization import (  # noqa: E402
    AuthorityMaterializationError,
    attach_task10_authority_coordinate,
    build_public_task10_inputs_manifest,
    build_task10_authority_from_sources,
    build_task13_route_binding_identity,
    publish_task10_authority,
    validate_public_task10_inputs_manifest,
)
from glm52_enforcement.task13_live_inputs import (  # noqa: E402
    validate_task10_materialization_manifest,
)


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
PROFILE = "keep-gpu"
OPERATION_ID = "prepackage:task10-production-authority"


def _canonical_file(path: Path, label: str) -> object:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AuthorityMaterializationError(label + " is not JSON") from error
    if raw != canonical_json_bytes(value) + b"\n":
        raise AuthorityMaterializationError(
            label + " is not canonical JSON plus LF"
        )
    return value


def _read_exact_s3(client: object, coordinate: Mapping[str, object]) -> bytes:
    required = {"bucket", "key", "version_id", "file_sha256", "body_sha256"}
    if not required.issubset(coordinate):
        raise AuthorityMaterializationError("S3 coordinate is incomplete")
    response = client.get_object(
        Bucket=coordinate["bucket"],
        Key=coordinate["key"],
        VersionId=coordinate["version_id"],
        ExpectedBucketOwner=ACCOUNT_ID,
        ChecksumMode="ENABLED",
    )
    metadata = response.get("ResponseMetadata", {})
    if (
        response.get("VersionId") != coordinate["version_id"]
        or metadata.get("HTTPStatusCode") != 200
        or not metadata.get("RequestId")
        or metadata.get("RetryAttempts") != 0
    ):
        raise AuthorityMaterializationError(
            "S3 exact VersionId readback is ambiguous"
        )
    body = response.get("Body")
    if getattr(body, "read", None) is None:
        raise AuthorityMaterializationError("S3 exact body is absent")
    raw = body.read()
    expected_checksum = base64.b64encode(
        hashlib.sha256(raw).digest()
    ).decode("ascii")
    if response.get("ChecksumSHA256") != expected_checksum:
        raise AuthorityMaterializationError("S3 checksum identity drifted")
    if hashlib.sha256(raw).hexdigest() != coordinate["file_sha256"]:
        raise AuthorityMaterializationError("S3 file identity drifted")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AuthorityMaterializationError(
            "S3 authority source is not JSON"
        ) from error
    unsigned = dict(value) if type(value) is dict else None
    if unsigned is None:
        raise AuthorityMaterializationError(
            "S3 authority source is not an object"
        )
    supplied = unsigned.pop("canonical_identity_sha256", None)
    candidates = {
        hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest(),
        supplied,
        value.get("ready_body_sha256"),
    }
    if coordinate["body_sha256"] not in candidates:
        raise AuthorityMaterializationError("S3 body identity drifted")
    return raw


def _read_validated_h100_resume_ready(
    client: object,
    *,
    coordinate: Mapping[str, object],
    campaign_identity_sha256: str,
    repository_archive_file_sha256: str,
    qualification_cache_manifest_sha256: str,
) -> dict[str, object]:
    raw = _read_exact_s3(client, coordinate)
    try:
        value = json.loads(raw)
        marker = validate_h100_resume_ready(value)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise AuthorityMaterializationError(
            "H100 resume readiness schema drifted"
        ) from error
    if marker["run_id"] != "glm52-sky-20260724":
        raise AuthorityMaterializationError(
            "H100 resume readiness run identity drifted"
        )
    if marker["campaign_identity_sha256"] != campaign_identity_sha256:
        raise AuthorityMaterializationError(
            "H100 resume readiness campaign identity drifted"
        )
    if marker["repo_tar_sha256"] != repository_archive_file_sha256:
        raise AuthorityMaterializationError(
            "H100 resume readiness archive identity drifted"
        )
    if (
        marker["qualification_cache_manifest_sha256"]
        != qualification_cache_manifest_sha256
    ):
        raise AuthorityMaterializationError(
            "H100 resume readiness qualification cache identity drifted"
        )
    if marker["ready_body_sha256"] != coordinate.get("body_sha256"):
        raise AuthorityMaterializationError(
            "H100 resume readiness body identity drifted"
        )
    return marker


def _unique_source_field(
    source_documents: Mapping[str, Mapping[str, object]],
    field: str,
) -> object:
    found = []

    def visit(value: object) -> None:
        if type(value) is dict:
            if field in value:
                found.append(value[field])
            for child in value.values():
                visit(child)
        elif type(value) is list:
            for child in value:
                visit(child)

    visit(source_documents)
    unique = {
        canonical_json_bytes(value): value
        for value in found
    }
    if len(unique) != 1:
        raise AuthorityMaterializationError(
            "authenticated source field is absent or ambiguous: " + field
        )
    return next(iter(unique.values()))


def _append_journal(path: Path, state: str, identity: str | None) -> None:
    if state not in {"PREPARED", "POSSIBLY_SENT", "COMMITTED"}:
        raise AuthorityMaterializationError("journal state drifted")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        info = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(info.st_mode):
            raise AuthorityMaterializationError("journal path drifted")
    row = {
        "schema_version": 1,
        "record_type": (
            "glm52_task13_authority_materialization_journal_v1"
        ),
        "operation_id": OPERATION_ID,
        "state": state,
        "result_identity_sha256": identity,
    }
    descriptor = os.open(
        path,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o600,
    )
    try:
        os.write(descriptor, canonical_json_bytes(row) + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new(path: Path, value: object) -> None:
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )
    try:
        raw = canonical_json_bytes(value) + b"\n"
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_public_outputs(
    output_directory: Path,
    *,
    authority: object,
    task_inputs: MountFreeTaskInputs,
    worker_descriptor: object,
    h100_resume_ready_coordinate: Mapping[str, object],
    task10_authority_coordinate: Mapping[str, object],
    task13_route_binding_identity_sha256: str,
) -> dict[str, object]:
    directory = Path(output_directory)
    if (
        not directory.is_absolute()
        or not directory.is_dir()
        or directory.is_symlink()
        or directory.resolve(strict=True) != directory
        or any(directory.iterdir())
    ):
        raise AuthorityMaterializationError(
            "public output must be a new empty directory"
        )
    paths = {
        "authority": directory / "task10-production-authority.json",
        "task_inputs": directory / "task10-task-inputs.json",
        "descriptor": directory / "task10-worker-descriptor.json",
        "h100": directory / "h100-resume-ready-coordinate.json",
        "route": directory / "task13-route-binding-identity.json",
        "manifest": directory / "public-task10-inputs.json",
    }
    manifest = build_public_task10_inputs_manifest(
        production_authority_path=paths["authority"],
        authority=authority,
        production_task_inputs_path=paths["task_inputs"],
        task_inputs=task_inputs,
        production_descriptor_path=paths["descriptor"],
        worker_descriptor=worker_descriptor,
        h100_resume_ready_coordinate_path=paths["h100"],
        h100_resume_ready_coordinate=h100_resume_ready_coordinate,
        task13_route_binding_identity_path=paths["route"],
        task10_authority_coordinate=task10_authority_coordinate,
        task13_route_binding_identity_sha256=(
            task13_route_binding_identity_sha256
        ),
    )
    route_body = {
        "schema_version": 1,
        "record_type": "glm52_task13_route_binding_identity_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "profile": PROFILE,
        "run_id": "glm52-sky-20260724",
        "task13_route_binding_identity_sha256": (
            task13_route_binding_identity_sha256
        ),
    }
    route = {
        **route_body,
        "canonical_identity_sha256": canonical_sha256(route_body),
    }
    outputs = (
        (paths["authority"], asdict(authority)),
        (paths["task_inputs"], asdict(task_inputs)),
        (paths["descriptor"], asdict(worker_descriptor)),
        (paths["h100"], dict(h100_resume_ready_coordinate)),
        (paths["route"], route),
        (paths["manifest"], manifest),
    )
    created = []
    try:
        for path, value in outputs:
            _write_new(path, value)
            created.append(path)
        validate_public_task10_inputs_manifest(manifest)
    except BaseException:
        for path in created:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    return manifest


def _preflight_outputs(
    *,
    output_coordinate: Path,
    output_reviewed_artifacts: Path,
    public_output_dir: Path,
) -> None:
    files = (output_coordinate, output_reviewed_artifacts)
    if (
        len(set(files)) != 2
        or any(
            not path.is_absolute()
            or path.resolve(strict=False) != path
            or not path.parent.is_dir()
            or path.parent.is_symlink()
            or path.exists()
            or path.is_symlink()
            for path in files
        )
        or not public_output_dir.is_absolute()
        or not public_output_dir.is_dir()
        or public_output_dir.is_symlink()
        or public_output_dir.resolve(strict=True) != public_output_dir
        or any(public_output_dir.iterdir())
        or any(
            public_output_dir == path.parent
            for path in files
        )
    ):
        raise AuthorityMaterializationError(
            "all outputs must be distinct new files and one new empty directory"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--journal", required=True, type=Path)
    parser.add_argument("--output-coordinate", required=True, type=Path)
    parser.add_argument(
        "--output-reviewed-artifacts",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--public-output-dir",
        required=True,
        type=Path,
    )
    args = parser.parse_args(argv)
    manifest = _canonical_file(args.manifest, "materialization manifest")
    if type(manifest) is not dict:
        raise AuthorityMaterializationError("manifest must be an object")
    manifest = validate_task10_materialization_manifest(
        manifest,
        repository_root=ROOT,
    )
    _preflight_outputs(
        output_coordinate=args.output_coordinate,
        output_reviewed_artifacts=args.output_reviewed_artifacts,
        public_output_dir=args.public_output_dir,
    )
    import boto3

    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    sts = session.client("sts")
    caller = sts.get_caller_identity()
    metadata = caller.get("ResponseMetadata", {})
    if (
        caller.get("Account") != ACCOUNT_ID
        or metadata.get("HTTPStatusCode") != 200
        or not metadata.get("RequestId")
        or metadata.get("RetryAttempts") != 0
    ):
        raise AuthorityMaterializationError("caller identity drifted")
    s3 = session.client("s3")
    boundary_raw = _read_exact_s3(s3, manifest["task11_boundary"])
    boundary = task11_boundary_from_bytes(boundary_raw)
    source_documents = {}
    for coordinate in boundary.inputs:
        raw = _read_exact_s3(s3, asdict(coordinate))
        source_documents[coordinate.input_kind] = json.loads(raw)
    approvals = manifest["approval_coordinates"]
    for coordinate in approvals.values():
        _read_exact_s3(s3, coordinate)
    task_inputs = MountFreeTaskInputs(
        **json.loads(
            _read_exact_s3(s3, manifest["task10_task_inputs"])
        )
    )
    h100_resume_ready = _read_validated_h100_resume_ready(
        s3,
        coordinate=manifest["h100_resume_ready"],
        campaign_identity_sha256=boundary.campaign_identity_sha256,
        repository_archive_file_sha256=(
            task_inputs.repository_archive_file_sha256
        ),
        qualification_cache_manifest_sha256=str(
            _unique_source_field(
                source_documents,
                "qualification_cache_manifest_sha256",
            )
        ),
    )
    descriptor = worker_bootstrap_descriptor_from_mapping(
        json.loads(
            _read_exact_s3(s3, manifest["task10_worker_descriptor"])
        )
    )
    runner = ROOT / manifest["runner_relative_path"]
    coordinator = ROOT / manifest["coordinator_relative_path"]
    runner_sha = hashlib.sha256(runner.read_bytes()).hexdigest()
    coordinator_sha = hashlib.sha256(coordinator.read_bytes()).hexdigest()
    if (
        runner_sha != manifest["runner_file_sha256"]
        or coordinator_sha != manifest["coordinator_file_sha256"]
    ):
        raise AuthorityMaterializationError(
            "Task13 executable identity drifted"
        )
    route_identity = build_task13_route_binding_identity(
        task11_boundary_coordinate=manifest["task11_boundary"],
        task11_input_coordinates=[
            asdict(value) for value in boundary.inputs
        ],
        approval_coordinates=approvals,
        workflow_inventory=manifest["workflow_inventory"],
        runner_file_sha256=runner_sha,
        coordinator_file_sha256=coordinator_sha,
    )
    states = session.client("stepfunctions")
    described = states.describe_state_machine(
        stateMachineArn=boundary.state_machine_version_arn
    )
    definition = json.loads(described["definition"])
    definition_text = json.dumps(definition, sort_keys=True)
    metadata = described.get("ResponseMetadata", {})
    if (
        described.get("stateMachineArn")
        != boundary.state_machine_version_arn
        or described.get("status") != "ACTIVE"
        or "RunInternalSixAzSoleSender" not in definition.get(
            "States",
            {},
        )
        or "keep-glm52-h1g-task10-capacity-reconciliation"
        not in definition_text
        or metadata.get("HTTPStatusCode") != 200
        or not metadata.get("RequestId")
        or metadata.get("RetryAttempts") != 0
    ):
        raise AuthorityMaterializationError(
            "retained Task10 workflow version readback drifted"
        )
    authority = build_task10_authority_from_sources(
        boundary=boundary,
        source_documents=source_documents,
        task_inputs=task_inputs,
        worker_descriptor=descriptor,
        task11_boundary_coordinate=manifest["task11_boundary"],
        h100_resume_ready_coordinate=manifest["h100_resume_ready"],
        h100_resume_ready=h100_resume_ready,
        workflow_readback={
            "state_machine_version_arn": boundary.state_machine_version_arn,
            "state_machine_name": "keep-glm52-h1g-production",
            "status": "ACTIVE",
            "terminal_writer_state": "RunInternalSixAzSoleSender",
            "reconciliation_function_name": (
                "keep-glm52-h1g-task10-capacity-reconciliation"
            ),
        },
        task13_route_binding_identity_sha256=route_identity,
    )
    _append_journal(args.journal, "PREPARED", None)
    _append_journal(args.journal, "POSSIBLY_SENT", None)
    coordinate = publish_task10_authority(s3, authority=authority)
    _append_journal(
        args.journal,
        "COMMITTED",
        canonical_sha256(coordinate),
    )
    reviewed = attach_task10_authority_coordinate(
        manifest["reviewed_artifacts_without_task10_authority"],
        coordinate,
    )
    public_created = False
    coordinate_created = False
    try:
        _write_public_outputs(
            args.public_output_dir,
            authority=authority,
            task_inputs=task_inputs,
            worker_descriptor=descriptor,
            h100_resume_ready_coordinate=manifest["h100_resume_ready"],
            task10_authority_coordinate=coordinate,
            task13_route_binding_identity_sha256=route_identity,
        )
        public_created = True
        _write_new(args.output_coordinate, coordinate)
        coordinate_created = True
        _write_new(args.output_reviewed_artifacts, reviewed)
    except BaseException:
        if coordinate_created:
            args.output_coordinate.unlink(missing_ok=True)
        if public_created:
            for name in (
                "task10-production-authority.json",
                "task10-task-inputs.json",
                "task10-worker-descriptor.json",
                "h100-resume-ready-coordinate.json",
                "task13-route-binding-identity.json",
                "public-task10-inputs.json",
            ):
                (args.public_output_dir / name).unlink(missing_ok=True)
        raise
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        AuthorityMaterializationError,
        KeyError,
        OSError,
        ValueError,
    ) as error:
        print("Task10 authority materialization refused: " + str(error), file=sys.stderr)
        raise SystemExit(64)
