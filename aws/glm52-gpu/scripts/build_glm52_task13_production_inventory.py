#!/usr/bin/env python3
"""Merge exact Task 13 S3 truth into a versioned production inventory."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Mapping

import boto3
from botocore.config import Config


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from glm52_enforcement.canonical import canonical_json_bytes  # noqa: E402
from glm52_enforcement.task13_campaign_package import (  # noqa: E402
    ACCOUNT_ID,
    PROFILE,
    REGION,
    RUN_ID,
    canonical_campaign_package_bytes,
    validate_campaign_artifact_coordinate,
)
from glm52_enforcement.task13_fixed_artifacts import (  # noqa: E402
    validate_repository_archive_manifest,
)
from mlx_vq.quality.glm52_s3_artifact_audit import (  # noqa: E402
    build_s3_artifact_inventory,
    validate_s3_artifact_inventory,
)


_STAGED_COORDINATE_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "activation_id",
        "package_identity_sha256",
        "bucket",
        "key",
        "version_id",
        "size_bytes",
        "file_sha256",
    }
)
_SUPPORT_ARCHIVE_FIELDS = frozenset(
    {
        "bucket",
        "key",
        "version_id",
        "size_bytes",
        "file_sha256",
    }
)
_SHA256 = frozenset("0123456789abcdef")
_ARTIFACT_INVENTORY_KINDS = {
    "ACCEPTED_BASELINE": "training_baseline",
    "BOOTSTRAP_TEMPLATE": "watchdog_code",
    "CLEAN_REHEARSAL": "approval",
    "FENCE_TEMPLATE": "watchdog_code",
    "GPU_SPEND_APPROVAL": "approval",
    "H100_QUALIFICATION_INPUT": "qualification_cache",
    "PRODUCTION_DESCRIPTOR": "campaign_descriptor",
    "PROMPT_PACK": "prompt_pack",
    "QUALIFICATION_CACHE_SEED_INPUT": "qualification_cache",
    "REPOSITORY_ARCHIVE": "campaign_descriptor",
    "RESIDUAL_LIABILITY_APPROVAL": "approval",
    "RETAINED_FOUNDATION_TEMPLATE": "watchdog_code",
    "RETAINED_PRE_SUPPORT_TEMPLATE": "watchdog_code",
    "RETAINED_TEMPLATE": "watchdog_code",
    "SUPPORT_APPROVAL": "approval",
    "SUPPORT_INPUTS": "training_configuration",
    "SUPPORT_TEMPLATE": "watchdog_code",
    "T01_T25_GATE": "approval",
    "TASK10_PRODUCTION_AUTHORITY": "approval",
    "TASK10_TASK_INPUTS": "campaign_descriptor",
    "TASK10_WORKER_DESCRIPTOR": "campaign_descriptor",
    "TASK11_REVIEW_APPROVAL": "approval",
    "TASK12_REVIEW_APPROVAL": "approval",
    "TRAINING_CONFIGURATION": "training_configuration",
    "TRANSPORT_22_MUTANT_GATE": "approval",
}


class Task13ProductionInventoryError(ValueError):
    """Task 13 truth cannot be projected into production inventory."""


@dataclass(frozen=True)
class Task13ProductionInventoryServices:
    """Read-only, single-attempt S3 transport."""

    s3: object
    total_max_attempts: int


def _fail(message: str) -> None:
    raise Task13ProductionInventoryError(message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256 for character in value)
    ):
        _fail(label + " must be one lowercase SHA-256")
    return value


def _require_version_id(value: object, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value in {"null", "None"}
        or value != value.strip()
        or not value.isascii()
    ):
        _fail(label + " must be one opaque non-null VersionId")
    return value


def _response(
    value: object,
    *,
    operation: str,
) -> Mapping[str, object]:
    if type(value) is not dict:
        _fail(operation + " returned no exact response object")
    metadata = value.get("ResponseMetadata")
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or type(metadata.get("RetryAttempts")) is not int
        or metadata["RetryAttempts"] != 0
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
    ):
        _fail(operation + " lacks authenticated zero-retry success")
    return value


def _read_exact_version(
    *,
    services: Task13ProductionInventoryServices,
    bucket: str,
    key: str,
    version_id: str,
    expected_sha256: str,
    expected_size: int | None = None,
) -> bytes:
    if (
        type(services) is not Task13ProductionInventoryServices
        or services.total_max_attempts != 1
    ):
        _fail("Task 13 production inventory S3 transport is not zero-retry")
    method = getattr(services.s3, "get_object", None)
    if not callable(method):
        _fail("read-only S3 GetObject method is absent")
    response = _response(
        method(
            Bucket=bucket,
            Key=key,
            VersionId=_require_version_id(
                version_id,
                key + " VersionId",
            ),
            ExpectedBucketOwner=ACCOUNT_ID,
            ChecksumMode="ENABLED",
        ),
        operation="GetObject",
    )
    if response.get("VersionId") != version_id:
        _fail(key + " exact-read VersionId drifted")
    content_length = response.get("ContentLength")
    if type(content_length) is not int or content_length < 1:
        _fail(key + " exact-read size is missing or empty")
    if expected_size is not None and content_length != expected_size:
        _fail(key + " exact-read size drifted")
    body = response.get("Body")
    reader = getattr(body, "read", None)
    if not callable(reader):
        _fail(key + " exact-read body is absent")
    chunks: list[bytes] = []
    length = 0
    digest = hashlib.sha256()
    while True:
        block = reader(8 * 1024 * 1024)
        if not block:
            break
        if type(block) is not bytes:
            _fail(key + " exact-read body yielded non-bytes")
        chunks.append(block)
        length += len(block)
        digest.update(block)
    if length != content_length:
        _fail(key + " exact-read byte length drifted")
    expected_digest = _require_sha256(
        expected_sha256,
        key + " file SHA-256",
    )
    if digest.hexdigest() != expected_digest:
        _fail(key + " exact-read file SHA-256 drifted")
    return b"".join(chunks)


def _canonical_json(raw: bytes, label: str) -> object:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Task13ProductionInventoryError(
            label + " is not UTF-8 JSON"
        ) from error
    if raw != canonical_json_bytes(value) + b"\n":
        _fail(label + " is not one canonical JSON value plus LF")
    return value


def _inventory_record(
    *,
    key: str,
    size: int,
    sha256: str,
    kind: str,
    run_scope: str,
    version_id: str,
) -> dict[str, object]:
    return {
        "key": key,
        "size": size,
        "sha256": sha256,
        "kind": kind,
        "safetensors": False,
        "run_scope": run_scope,
        "version_id": version_id,
    }


def _validate_staged_coordinate(
    value: object,
    *,
    package: Mapping[str, object],
    bucket: str,
) -> dict[str, object]:
    if (
        type(value) is not dict
        or frozenset(value) != _STAGED_COORDINATE_FIELDS
    ):
        _fail("staged evidence coordinate field set drifted")
    identity = _require_sha256(
        package.get("canonical_identity_sha256"),
        "campaign package identity",
    )
    activation_id = package.get("activation_id")
    key = value.get("key")
    expected_prefix = (
        f"campaigns/{RUN_ID}/task13/{identity}/"
    )
    if (
        value.get("schema_version") != 1
        or value.get("record_type")
        != "glm52_task13_staged_evidence_coordinate_v1"
        or value.get("account_id") != ACCOUNT_ID
        or value.get("region") != REGION
        or value.get("run_id") != RUN_ID
        or value.get("activation_id") != activation_id
        or value.get("package_identity_sha256") != identity
        or value.get("bucket") != bucket
        or type(key) is not str
        or not key.startswith(expected_prefix)
        or key != (
            expected_prefix
            + "staged-infrastructure-evidence-v1.json"
        )
        or PurePosixPath(key).is_absolute()
        or ".." in PurePosixPath(key).parts
    ):
        _fail("staged evidence coordinate identity drifted")
    _require_version_id(
        value.get("version_id"),
        "staged evidence VersionId",
    )
    size = value.get("size_bytes")
    if type(size) is not int or size < 1:
        _fail("staged evidence size is missing or empty")
    _require_sha256(
        value.get("file_sha256"),
        "staged evidence file SHA-256",
    )
    return dict(value)


def _staged_support_archives(
    value: object,
    *,
    bucket: str,
) -> list[dict[str, object]]:
    if type(value) is not dict:
        _fail("staged infrastructure evidence is not an object")
    support = value.get("support_stack")
    staged_request = value.get("staged_request")
    production_request = (
        staged_request.get("production_request")
        if type(staged_request) is dict
        else None
    )
    publication = (
        production_request.get("support_artifact_publication")
        if type(production_request) is dict
        else None
    )
    if type(support) is not dict or type(publication) is not dict:
        _fail("staged support artifact authority is missing")
    expected_prefixes = {
        "support_lambda_archive": "task13/artifacts/support-lambda/",
        "cryptography_layer_archive": (
            "task13/artifacts/cryptography-layer-python312-x86_64/"
        ),
    }
    result: list[dict[str, object]] = []
    for field, prefix in expected_prefixes.items():
        coordinate = support.get(field)
        if (
            type(coordinate) is not dict
            or frozenset(coordinate) != _SUPPORT_ARCHIVE_FIELDS
            or publication.get(field) != coordinate
        ):
            _fail(field + " staged coordinate binding drifted")
        key = coordinate.get("key")
        file_sha256 = _require_sha256(
            coordinate.get("file_sha256"),
            field + " file SHA-256",
        )
        size = coordinate.get("size_bytes")
        if (
            coordinate.get("bucket") != bucket
            or type(key) is not str
            or key != prefix + file_sha256 + ".zip"
            or PurePosixPath(key).is_absolute()
            or ".." in PurePosixPath(key).parts
            or type(size) is not int
            or size < 1
        ):
            _fail(field + " staged coordinate identity drifted")
        _require_version_id(
            coordinate.get("version_id"),
            field + " VersionId",
        )
        result.append(dict(coordinate))
    return result


def _merge_records(
    base: list[dict[str, object]],
    added: list[dict[str, object]],
) -> list[dict[str, object]]:
    merged = {str(row["key"]): dict(row) for row in base}
    for row in added:
        key = str(row["key"])
        prior = merged.get(key)
        if prior is None:
            merged[key] = row
        elif prior != row:
            _fail(
                key
                + " duplicate inventory key has version or identity drift"
            )
    return [merged[key] for key in sorted(merged)]


def build_task13_production_inventory(
    *,
    base_inventory: object,
    campaign_package: object,
    reviewed_artifacts: object,
    staged_evidence_coordinate: object,
    services: Task13ProductionInventoryServices,
) -> dict[str, object]:
    """Build the exact versioned inventory without making an AWS mutation."""

    try:
        base = validate_s3_artifact_inventory(base_inventory)
        canonical_campaign_package_bytes(campaign_package)
    except (TypeError, ValueError) as error:
        raise Task13ProductionInventoryError(
            "base inventory or campaign package is not exact"
        ) from error
    if (
        type(campaign_package) is not dict
        or campaign_package.get("package_phase")
        not in {"PREQUALIFICATION", "PRODUCTION"}
        or campaign_package.get("account_id") != ACCOUNT_ID
        or campaign_package.get("region") != REGION
        or campaign_package.get("profile") != PROFILE
        or campaign_package.get("run_id") != RUN_ID
        or base.get("run_id") != RUN_ID
    ):
        _fail("campaign package or base inventory identity is foreign")
    activation_id = campaign_package.get("activation_id")
    if type(activation_id) is not str:
        _fail("campaign package activation identity is missing")
    bucket = base.get("bucket")
    if type(bucket) is not str or not bucket:
        _fail("base inventory bucket is missing")
    base_rows = base.get("objects")
    if type(base_rows) is not list:
        _fail("base production inventory object set is missing")
    for row in base_rows:
        if type(row) is not dict or "version_id" not in row:
            _fail("base production inventory requires exact VersionIds")
        _require_version_id(
            row.get("version_id"),
            str(row.get("key", "base object")) + " VersionId",
        )
    if not base_rows:
        _fail("base production inventory requires exact VersionIds")
    package_rows = campaign_package.get("reviewed_artifacts")
    if (
        type(reviewed_artifacts) is not list
        or reviewed_artifacts != package_rows
    ):
        _fail("reviewed artifacts do not match the campaign package")

    coordinates: list[dict[str, object]] = []
    kinds: set[str] = set()
    for raw_coordinate in reviewed_artifacts:
        try:
            coordinate = validate_campaign_artifact_coordinate(
                raw_coordinate
            )
        except ValueError as error:
            raise Task13ProductionInventoryError(
                "reviewed artifact coordinate is not exact"
            ) from error
        kind = str(coordinate["artifact_kind"])
        if (
            kind in kinds
            or kind not in _ARTIFACT_INVENTORY_KINDS
            or coordinate["bucket"] != bucket
        ):
            _fail("reviewed artifact kind or bucket drifted")
        kinds.add(kind)
        coordinates.append(coordinate)

    added: list[dict[str, object]] = []
    repository_manifest: dict[str, object] | None = None
    for coordinate in coordinates:
        key = str(coordinate["key"])
        version_id = str(coordinate["version_id"])
        raw = _read_exact_version(
            services=services,
            bucket=bucket,
            key=key,
            version_id=version_id,
            expected_sha256=str(coordinate["file_sha256"]),
        )
        added.append(
            _inventory_record(
                key=key,
                size=len(raw),
                sha256=str(coordinate["file_sha256"]),
                kind=_ARTIFACT_INVENTORY_KINDS[
                    str(coordinate["artifact_kind"])
                ],
                run_scope="shared",
                version_id=version_id,
            )
        )
        if coordinate["artifact_kind"] == "REPOSITORY_ARCHIVE":
            value = _canonical_json(
                raw,
                "REPOSITORY_ARCHIVE reviewed artifact",
            )
            if type(value) is not dict:
                _fail("repository archive authority is not an object")
            if (
                value.get("canonical_identity_sha256")
                != coordinate["body_sha256"]
            ):
                _fail("repository archive manifest body identity drifted")
            archive_sha = value.get("archive_file_sha256")
            archive_size = value.get("archive_size_bytes")
            if type(archive_size) is not int or archive_size < 1:
                _fail("repository archive payload size is missing")
            try:
                repository_manifest = validate_repository_archive_manifest(
                    value,
                    expected_activation_id=activation_id,
                    expected_archive_sha256=_require_sha256(
                        archive_sha,
                        "repository archive payload SHA-256",
                    ),
                    expected_archive_size=archive_size,
                    expected_bucket=bucket,
                )
            except ValueError as error:
                raise Task13ProductionInventoryError(
                    "repository archive manifest is not exact"
                ) from error

    if repository_manifest is None:
        _fail("repository archive reviewed authority is missing")
    archive = repository_manifest["archive"]
    if type(archive) is not dict:
        _fail("repository archive payload coordinate is missing")
    archive_key = str(archive.get("key"))
    archive_version = _require_version_id(
        archive.get("version_id"),
        "repository archive payload VersionId",
    )
    archive_sha = _require_sha256(
        archive.get("file_sha256"),
        "repository archive payload file SHA-256",
    )
    archive_size = repository_manifest.get("archive_size_bytes")
    if type(archive_size) is not int or archive_size < 1:
        _fail("repository archive payload size is missing")
    _read_exact_version(
        services=services,
        bucket=bucket,
        key=archive_key,
        version_id=archive_version,
        expected_sha256=archive_sha,
        expected_size=archive_size,
    )
    added.append(
        _inventory_record(
            key=archive_key,
            size=archive_size,
            sha256=archive_sha,
            kind="repository_tar",
            run_scope="shared",
            version_id=archive_version,
        )
    )

    staged = _validate_staged_coordinate(
        staged_evidence_coordinate,
        package=campaign_package,
        bucket=bucket,
    )
    staged_raw = _read_exact_version(
        services=services,
        bucket=bucket,
        key=str(staged["key"]),
        version_id=str(staged["version_id"]),
        expected_sha256=str(staged["file_sha256"]),
        expected_size=int(staged["size_bytes"]),
    )
    staged_value = _canonical_json(
        staged_raw,
        "published staged infrastructure evidence",
    )
    deployment = campaign_package.get("disabled_deployment")
    embedded = (
        deployment.get("staged_infrastructure_evidence")
        if type(deployment) is dict
        else None
    )
    if staged_value != embedded:
        _fail("published staged evidence differs from the campaign package")
    added.append(
        _inventory_record(
            key=str(staged["key"]),
            size=int(staged["size_bytes"]),
            sha256=str(staged["file_sha256"]),
            kind="approval",
            run_scope="shared",
            version_id=str(staged["version_id"]),
        )
    )
    for coordinate in _staged_support_archives(
        staged_value,
        bucket=bucket,
    ):
        raw = _read_exact_version(
            services=services,
            bucket=bucket,
            key=str(coordinate["key"]),
            version_id=str(coordinate["version_id"]),
            expected_sha256=str(coordinate["file_sha256"]),
            expected_size=int(coordinate["size_bytes"]),
        )
        added.append(
            _inventory_record(
                key=str(coordinate["key"]),
                size=len(raw),
                sha256=str(coordinate["file_sha256"]),
                kind="watchdog_code",
                run_scope="shared",
                version_id=str(coordinate["version_id"]),
            )
        )

    merged = _merge_records(
        [dict(row) for row in base_rows],
        added,
    )
    try:
        return build_s3_artifact_inventory(
            run_id=RUN_ID,
            bucket=bucket,
            objects=merged,
        )
    except ValueError as error:
        raise Task13ProductionInventoryError(
            "merged production inventory is not exact"
        ) from error


def _read_regular_file(path: Path, label: str) -> bytes:
    source = Path(path)
    if (
        not source.is_absolute()
        or not source.is_file()
        or source.is_symlink()
    ):
        _fail(label + " must be one absolute regular non-symlink file")
    return source.read_bytes()


def _parse_canonical(path: Path, label: str) -> object:
    return _canonical_json(_read_regular_file(path, label), label)


def _parse_package(path: Path) -> dict[str, object]:
    raw = _read_regular_file(path, "campaign package")
    value = _canonical_json(raw, "campaign package")
    try:
        expected = canonical_campaign_package_bytes(value)
    except ValueError as error:
        raise Task13ProductionInventoryError(
            "campaign package failed validation"
        ) from error
    if raw != expected:
        _fail("campaign package bytes are noncanonical")
    assert type(value) is dict
    return value


def _parse_inventory(path: Path) -> dict[str, object]:
    raw = _read_regular_file(path, "base production inventory")
    value = _canonical_json(raw, "base production inventory")
    try:
        checked = validate_s3_artifact_inventory(value)
    except ValueError as error:
        raise Task13ProductionInventoryError(
            "base production inventory failed validation"
        ) from error
    expected = (
        json.dumps(
            checked,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    if raw != expected:
        _fail("base production inventory bytes are noncanonical")
    return checked


def write_inventory_once(
    path: Path,
    inventory: Mapping[str, object],
) -> None:
    """Durably create one private inventory without replacement."""

    target = Path(path)
    if (
        not target.is_absolute()
        or not target.parent.is_dir()
        or target.parent.is_symlink()
        or target.exists()
        or target.is_symlink()
    ):
        _fail("production inventory output must be one new absolute file")
    checked = validate_s3_artifact_inventory(dict(inventory))
    raw = (
        json.dumps(
            checked,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    descriptor_number = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    descriptor = os.fdopen(descriptor_number, "wb")
    try:
        descriptor.write(raw)
        descriptor.flush()
        os.fsync(descriptor.fileno())
    except BaseException:
        descriptor.close()
        target.unlink(missing_ok=True)
        raise
    descriptor.close()
    parent = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--region", default=REGION)
    parser.add_argument(
        "--base-inventory",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--campaign-package",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--reviewed-artifacts",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--staged-evidence-coordinate",
        required=True,
        type=Path,
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.profile != PROFILE or args.region != REGION:
            _fail("Task 13 production inventory profile or region is foreign")
        subprocess.run(
            [
                str(
                    ROOT
                    / "aws/glm52-gpu/scripts/assert_rnd_aws_account.sh"
                )
            ],
            check=True,
            env={**os.environ, "AWS_PROFILE": args.profile},
        )
        package = _parse_package(args.campaign_package)
        reviewed = _parse_canonical(
            args.reviewed_artifacts,
            "reviewed artifact set",
        )
        staged = _parse_canonical(
            args.staged_evidence_coordinate,
            "staged evidence coordinate",
        )
        session = boto3.Session(
            profile_name=args.profile,
            region_name=args.region,
        )
        inventory = build_task13_production_inventory(
            base_inventory=_parse_inventory(args.base_inventory),
            campaign_package=package,
            reviewed_artifacts=reviewed,
            staged_evidence_coordinate=staged,
            services=Task13ProductionInventoryServices(
                s3=session.client(
                    "s3",
                    config=Config(
                        retries={
                            "mode": "standard",
                            "total_max_attempts": 1,
                        }
                    ),
                ),
                total_max_attempts=1,
            ),
        )
        write_inventory_once(args.output, inventory)
    except (
        OSError,
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "Task 13 production-inventory build refused: " + str(error),
            file=sys.stderr,
        )
        return 64
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
