#!/usr/bin/env python3
"""Stage one immutable Sky bundle, audit all S3 inputs, and publish readiness last."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIT_SCRIPT_RELATIVE = "aws/glm52-gpu/scripts/audit_s3_campaign_artifacts.py"
_READY_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "run_id",
        "descriptor_key",
        "descriptor_sha256",
        "descriptor_body_sha256",
        "campaign_identity_sha256",
        "bundle_manifest_key",
        "bundle_manifest_file_sha256",
        "bundle_manifest_body_sha256",
        "bundle_manifest_version_id",
        "staged_object_version_ids",
        "artifact_audit_key",
        "artifact_audit_sha256",
        "staged_at",
        "ready_body_sha256",
    }
)
_STAGED_OBJECT_VERSION_ROLES = frozenset(
    {
        "repository_tar",
        "approval",
        "training_config",
        "watchdog",
        "artifact_inventory",
        "artifact_audit",
        "descriptor",
    }
)
_AUDIT_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "audit_pass",
        "run_id",
        "bucket",
        "inventory_body_sha256",
        "object_count",
        "object_bytes",
        "safetensors_object_count",
        "safetensors_tensor_count",
    }
)
_RECEIPT_VERSION_ROLES = frozenset(
    {
        "repository_tar",
        "approval",
        "training_config",
        "watchdog",
        "artifact_inventory",
        "artifact_audit",
        "bundle_manifest",
        "descriptor",
        "ready",
    }
)
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_BUCKET = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]")
_REPOSITORY_ARCHIVE_RECORD_TYPE = (
    "glm52_task13_repository_archive_payload_v1"
)


def _load(name: str, relative: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SKY = _load("_glm52_stage_sky", "src/mlx_vq/quality/glm52_sky_campaign.py")
AUDIT = _load("_glm52_stage_audit", "src/mlx_vq/quality/glm52_s3_artifact_audit.py")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class _StagedObject:
    disposition: str
    version_id: str


def _command_error(command: subprocess.CompletedProcess[str]) -> str:
    return command.stderr.strip() or command.stdout.strip() or "AWS CLI failed"


def _version_id(value: object, *, context: str) -> str:
    if not isinstance(value, str) or value in {"", "null"}:
        raise ValueError(f"{context} requires a non-null S3 VersionId")
    return value


class _AwsCliS3:
    def __init__(
        self,
        *,
        profile: str,
        region: str,
        bucket: str,
    ) -> None:
        self.profile = profile
        self.region = region
        self.bucket = bucket

    def head_object(self, key: str) -> Mapping[str, object] | None:
        command = subprocess.run(
            [
                "aws",
                "s3api",
                "head-object",
                "--bucket",
                self.bucket,
                "--key",
                key,
                "--profile",
                self.profile,
                "--region",
                self.region,
                "--output",
                "json",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if command.returncode == 0:
            value = json.loads(command.stdout)
            if not isinstance(value, dict):
                raise ValueError(f"S3 HEAD returned a non-object for {key}")
            return value
        combined = command.stderr + command.stdout
        if any(
            token in combined for token in ("404", "Not Found", "NotFound", "NoSuchKey")
        ):
            return None
        raise RuntimeError(_command_error(command))

    def get_object(
        self,
        *,
        key: str,
        version_id: str,
        destination: Path,
    ) -> Mapping[str, object]:
        command = subprocess.run(
            [
                "aws",
                "s3api",
                "get-object",
                "--bucket",
                self.bucket,
                "--key",
                key,
                "--version-id",
                version_id,
                "--profile",
                self.profile,
                "--region",
                self.region,
                "--output",
                "json",
                str(destination),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if command.returncode != 0:
            raise RuntimeError(_command_error(command))
        value = json.loads(command.stdout)
        if not isinstance(value, dict):
            raise ValueError(f"S3 GET returned a non-object for {key}")
        returned_version = _version_id(
            value.get("VersionId"),
            context=f"S3 GET s3://{self.bucket}/{key}",
        )
        if returned_version != version_id:
            raise ValueError(
                f"S3 GET returned the wrong VersionId for s3://{self.bucket}/{key}"
            )
        return value

    def put_object(
        self,
        *,
        key: str,
        path: Path,
        run_id: str,
        expected_sha: str,
        metadata: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        object_metadata = (
            {"glm52-run-id": run_id}
            if metadata is None
            else dict(metadata)
        )
        metadata_argument = ",".join(
            f"{name}={value}"
            for name, value in sorted(object_metadata.items())
        )
        checksum = base64.b64encode(bytes.fromhex(expected_sha)).decode()
        return subprocess.run(
            [
                "aws",
                "s3api",
                "put-object",
                "--bucket",
                self.bucket,
                "--key",
                key,
                "--body",
                str(path),
                "--checksum-algorithm",
                "SHA256",
                "--checksum-sha256",
                checksum,
                "--if-none-match",
                "*",
                "--metadata",
                metadata_argument,
                "--profile",
                self.profile,
                "--region",
                self.region,
                "--output",
                "json",
            ],
            text=True,
            capture_output=True,
            check=False,
        )


def _authenticate_version(
    *,
    s3: object,
    bucket: str,
    key: str,
    version_id: str,
    expected_size: int,
    expected_sha: str,
    expected_metadata: Mapping[str, str] | None = None,
) -> None:
    with tempfile.TemporaryDirectory(prefix="glm52-stage-readback-") as directory:
        destination = Path(directory) / "object"
        response = s3.get_object(
            key=key,
            version_id=version_id,
            destination=destination,
        )
        if destination.stat().st_size != expected_size or _sha(destination) != (
            expected_sha
        ):
            raise ValueError(f"incompatible exact S3 version s3://{bucket}/{key}")
        if (
            expected_metadata is not None
            and response.get("Metadata") != dict(expected_metadata)
        ):
            raise ValueError(
                f"incompatible exact S3 metadata s3://{bucket}/{key}"
            )


def _authenticate_current(
    *,
    s3: object,
    bucket: str,
    key: str,
    expected_size: int,
    expected_sha: str,
    expected_metadata: Mapping[str, str] | None = None,
) -> str | None:
    head = s3.head_object(key)
    if head is None:
        return None
    version = _version_id(
        head.get("VersionId"),
        context=f"S3 HEAD s3://{bucket}/{key}",
    )
    size = head.get("ContentLength")
    if type(size) is not int or size != expected_size:
        raise ValueError(f"refusing to overwrite incompatible s3://{bucket}/{key}")
    _authenticate_version(
        s3=s3,
        bucket=bucket,
        key=key,
        version_id=version,
        expected_size=expected_size,
        expected_sha=expected_sha,
        expected_metadata=expected_metadata,
    )
    return version


def _put_immutable(
    *,
    profile: str,
    region: str,
    bucket: str,
    key: str,
    path: Path,
    run_id: str,
    record_type: str | None = None,
    s3: object | None = None,
) -> _StagedObject:
    boundary = s3 or _AwsCliS3(
        profile=profile,
        region=region,
        bucket=bucket,
    )
    expected_sha = _sha(path)
    expected_size = path.stat().st_size
    expected_metadata = (
        None
        if record_type is None
        else {
            "glm52-run-id": run_id,
            "record-type": record_type,
            "file-sha256": expected_sha,
        }
    )
    existing_version = _authenticate_current(
        s3=boundary,
        bucket=bucket,
        key=key,
        expected_size=expected_size,
        expected_sha=expected_sha,
        expected_metadata=expected_metadata,
    )
    if existing_version is not None:
        return _StagedObject("reused", existing_version)
    put_arguments: dict[str, object] = {
        "key": key,
        "path": path,
        "run_id": run_id,
        "expected_sha": expected_sha,
    }
    if expected_metadata is not None:
        put_arguments["metadata"] = expected_metadata
    put = boundary.put_object(**put_arguments)
    if put.returncode == 0:
        value = json.loads(put.stdout)
        if not isinstance(value, dict):
            raise ValueError(f"S3 PUT returned a non-object for {key}")
        version = _version_id(
            value.get("VersionId"),
            context=f"S3 PUT s3://{bucket}/{key}",
        )
        _authenticate_version(
            s3=boundary,
            bucket=bucket,
            key=key,
            version_id=version,
            expected_size=expected_size,
            expected_sha=expected_sha,
            expected_metadata=expected_metadata,
        )
        return _StagedObject("uploaded", version)

    original_failure = _command_error(put)
    try:
        recovered_version = _authenticate_current(
            s3=boundary,
            bucket=bucket,
            key=key,
            expected_size=expected_size,
            expected_sha=expected_sha,
            expected_metadata=expected_metadata,
        )
    except (RuntimeError, TypeError, ValueError) as recovery_error:
        raise RuntimeError(
            f"{original_failure}; ambiguous PUT recovery failed: {recovery_error}"
        ) from recovery_error
    if recovered_version is None:
        raise RuntimeError(
            f"{original_failure}; ambiguous PUT recovery found no object"
        )
    return _StagedObject(
        "reused-after-ambiguous-put",
        recovered_version,
    )


def _validate_manifest_bytes(
    raw: bytes,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    value = _finite_json(raw, label="Sky campaign bundle manifest")
    required = {
        "schema_version",
        "record_type",
        "run_id",
        "bucket",
        "files",
        "descriptor_body_sha256",
        "bundle_manifest_body_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("schema_version") != 1
        or value.get("record_type") != "glm52_sky_campaign_bundle_v1"
    ):
        raise ValueError("Sky campaign bundle manifest schema mismatch")
    if raw != _canonical(value) + b"\n":
        raise ValueError(
            "Sky campaign bundle manifest bytes are not canonical JSON plus LF"
        )
    run_id = value.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("Sky campaign bundle manifest run_id is invalid")
    bucket = value.get("bucket")
    if (
        not isinstance(bucket, str)
        or _BUCKET.fullmatch(bucket) is None
        or ".." in bucket
        or bucket.startswith(("xn--", "sthree-", "amzn-s3-demo-"))
        or bucket.endswith(
            ("-s3alias", "--ol-s3", ".mrap", "--x-s3", "--table-s3")
        )
        or (
            len(bucket.split(".")) == 4
            and all(part.isdigit() for part in bucket.split("."))
        )
    ):
        raise ValueError("Sky campaign bundle manifest bucket is invalid")
    body = dict(value)
    digest = _require_sha256(
        body.pop("bundle_manifest_body_sha256"),
        field="Sky campaign bundle manifest body SHA-256",
    )
    if digest != hashlib.sha256(_canonical(body)).hexdigest():
        raise ValueError("Sky campaign bundle manifest SHA-256 mismatch")
    files = value.get("files")
    if not isinstance(files, list) or len(files) != 6:
        raise ValueError("Sky campaign bundle files are missing")
    fields = {"local_name", "key", "role", "stage_order", "size", "sha256"}
    if any(not isinstance(item, dict) or set(item) != fields for item in files):
        raise ValueError("Sky campaign bundle file schema mismatch")
    roles = [item["role"] for item in files]
    expected_roles = {
        "repository_tar",
        "approval",
        "training_config",
        "watchdog",
        "artifact_inventory",
        "descriptor",
    }
    if set(roles) != expected_roles or len(roles) != len(expected_roles):
        raise ValueError("Sky campaign bundle roles are incomplete or duplicated")

    local_names: list[str] = []
    keys: list[str] = []
    stage_orders: list[int] = []
    for item in files:
        local_name = item["local_name"]
        if (
            not isinstance(local_name, str)
            or not local_name
            or local_name in {".", ".."}
            or Path(local_name).name != local_name
            or "/" in local_name
            or "\\" in local_name
        ):
            raise ValueError("Sky campaign bundle local name is unsafe")
        try:
            local_name.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError(
                "Sky campaign bundle local name is not ASCII"
            ) from error
        local_names.append(local_name)
        keys.append(_safe_key(item["key"], field="Sky campaign bundle file key"))
        size = item["size"]
        if type(size) is not int or size < 0:
            raise ValueError("Sky campaign bundle file size is invalid")
        _require_sha256(
            item["sha256"],
            field="Sky campaign bundle file SHA-256",
        )
        stage_order = item["stage_order"]
        if type(stage_order) is not int or stage_order < 0:
            raise ValueError("Sky campaign bundle stage order is invalid")
        stage_orders.append(stage_order)
    if len(set(local_names)) != len(local_names):
        raise ValueError("Sky campaign bundle local names are duplicated")
    if len(set(keys)) != len(keys):
        raise ValueError("Sky campaign bundle S3 keys are duplicated")
    if len(set(stage_orders)) != len(stage_orders):
        raise ValueError("Sky campaign bundle stage orders are duplicated")

    ordered = sorted(files, key=lambda item: item["stage_order"])
    if ordered[-1]["role"] != "descriptor":
        raise ValueError("Sky campaign descriptor must stage last")
    return dict(value), [dict(item) for item in ordered]


def _validate_manifest(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    return _validate_manifest_bytes(path.read_bytes())


@dataclass(frozen=True)
class _BundleContext:
    manifest: dict[str, object]
    manifest_raw: bytes
    manifest_path: Path
    manifest_key: str
    manifest_file_sha256: str
    files: tuple[dict[str, object], ...]
    local: dict[str, Path]
    descriptor: dict[str, object]
    inventory: dict[str, object]
    run_id: str
    bucket: str
    descriptor_item: dict[str, object]
    ready_key: str


@dataclass(frozen=True)
class _RemoteBytes:
    raw: bytes
    version_id: str


@dataclass(frozen=True)
class _ReadinessWinner:
    value: dict[str, object]
    raw: bytes
    version_id: str
    s3_version_ids: dict[str, str]


def _finite_json(raw: bytes, *, label: str) -> object:
    try:
        return json.loads(
            raw,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not finite JSON: {error}") from error


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} is not a canonical SHA-256")
    return value


def _safe_key(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("/")
        or value.endswith("/")
        or any(not 0x21 <= ord(character) <= 0x7E for character in value)
        or "\\" in value
        or any(character in value for character in "*?[]")
    ):
        raise ValueError(f"{field} is not a safe S3 key")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{field} is not a safe S3 key")
    return value


def _canonical_staged_at(value: object) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("readiness staged_at is not canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("readiness staged_at is not canonical UTC") from error
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != timezone.utc.utcoffset(parsed)
        or parsed.microsecond != 0
        or parsed.isoformat().replace("+00:00", "Z") != value
    ):
        raise ValueError("readiness staged_at is not canonical UTC")
    return value


def _prepare_bundle(bundle: Path) -> _BundleContext:
    manifest_path = bundle / "bundle-manifest-v1.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Sky campaign bundle manifest is missing")
    manifest_raw = manifest_path.read_bytes()
    manifest, files = _validate_manifest_bytes(manifest_raw)
    run_id = manifest["run_id"]
    bucket = manifest["bucket"]
    assert isinstance(run_id, str)
    assert isinstance(bucket, str)
    local: dict[str, Path] = {}
    for item in files:
        path = bundle / str(item["local_name"])
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"bundle file is missing: {path}")
        if path.stat().st_size != item["size"] or _sha(path) != item["sha256"]:
            raise ValueError(f"bundle file identity mismatch: {path.name}")
        local[str(item["role"])] = path
    descriptor = SKY.validate_sky_campaign_descriptor(
        _finite_json(local["descriptor"].read_bytes(), label="descriptor")
    )
    inventory = AUDIT.validate_s3_artifact_inventory(
        _finite_json(
            local["artifact_inventory"].read_bytes(),
            label="artifact inventory",
        )
    )
    if (
        descriptor["run_id"] != run_id
        or descriptor["bucket"] != bucket
        or inventory["run_id"] != run_id
        or inventory["bucket"] != bucket
        or descriptor["descriptor_body_sha256"] != manifest["descriptor_body_sha256"]
    ):
        raise ValueError("bundle authorities do not agree")
    descriptor_item = next(item for item in files if item["role"] == "descriptor")
    descriptor_key = _safe_key(
        descriptor_item["key"],
        field="descriptor key",
    )
    expected_suffix = "/campaign-descriptor-v2.json"
    prefix = f"campaigns/{run_id}/submissions/"
    if not descriptor_key.startswith(prefix) or not descriptor_key.endswith(
        expected_suffix
    ):
        raise ValueError("descriptor key is outside the exact submission prefix")
    submission_id = descriptor_key[len(prefix) : -len(expected_suffix)]
    if "/" in submission_id or submission_id in {"", ".", ".."}:
        raise ValueError("descriptor submission ID is unsafe")
    manifest_key = (
        f"{prefix}{submission_id}/bundle-manifests/"
        f"{manifest['bundle_manifest_body_sha256']}/bundle-manifest-v1.json"
    )
    ready_key = f"{prefix}{submission_id}/STAGED_CONTROL_PLANE_READY.json"
    return _BundleContext(
        manifest=dict(manifest),
        manifest_raw=manifest_raw,
        manifest_path=manifest_path,
        manifest_key=manifest_key,
        manifest_file_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        files=tuple(dict(item) for item in files),
        local=local,
        descriptor=dict(descriptor),
        inventory=dict(inventory),
        run_id=run_id,
        bucket=bucket,
        descriptor_item=dict(descriptor_item),
        ready_key=ready_key,
    )


def _read_current_bytes(
    *,
    s3: object,
    bucket: str,
    key: str,
) -> _RemoteBytes | None:
    head = s3.head_object(key)
    if head is None:
        return None
    version = _version_id(
        head.get("VersionId"),
        context=f"S3 HEAD s3://{bucket}/{key}",
    )
    size = head.get("ContentLength")
    if type(size) is not int or size < 0:
        raise ValueError(f"S3 HEAD returned an invalid size for {key}")
    with tempfile.TemporaryDirectory(prefix="glm52-stage-read-") as directory:
        destination = Path(directory) / "object"
        s3.get_object(
            key=key,
            version_id=version,
            destination=destination,
        )
        if destination.stat().st_size != size:
            raise ValueError(f"S3 version size drift for {key}")
        return _RemoteBytes(destination.read_bytes(), version)


def _read_exact_bytes(
    *,
    s3: object,
    bucket: str,
    key: str,
    version_id: object,
    label: str,
) -> _RemoteBytes:
    version = _version_id(version_id, context=f"{label} VersionId")
    with tempfile.TemporaryDirectory(prefix="glm52-stage-exact-read-") as directory:
        destination = Path(directory) / "object"
        s3.get_object(
            key=key,
            version_id=version,
            destination=destination,
        )
        return _RemoteBytes(destination.read_bytes(), version)


def _validate_readiness(
    raw: bytes,
    *,
    context: _BundleContext,
) -> dict[str, object]:
    value = _finite_json(raw, label="staged readiness")
    if not isinstance(value, dict) or set(value) != _READY_FIELDS:
        raise ValueError("staged readiness schema mismatch")
    if raw != _canonical(value) + b"\n":
        raise ValueError("staged readiness bytes are not canonical JSON plus LF")
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 2
        or value.get("record_type") != "glm52_staged_control_plane_ready_v2"
    ):
        raise ValueError("staged readiness schema mismatch")
    body = dict(value)
    digest = _require_sha256(
        body.pop("ready_body_sha256"),
        field="staged readiness body SHA-256",
    )
    if hashlib.sha256(_canonical(body)).hexdigest() != digest:
        raise ValueError("staged readiness body SHA-256 mismatch")
    _canonical_staged_at(value.get("staged_at"))
    descriptor_key = _safe_key(
        value.get("descriptor_key"),
        field="readiness descriptor key",
    )
    manifest_key = _safe_key(
        value.get("bundle_manifest_key"),
        field="readiness bundle manifest key",
    )
    manifest_file_sha = _require_sha256(
        value.get("bundle_manifest_file_sha256"),
        field="readiness bundle manifest file SHA-256",
    )
    manifest_body_sha = _require_sha256(
        value.get("bundle_manifest_body_sha256"),
        field="readiness bundle manifest body SHA-256",
    )
    manifest_version = _version_id(
        value.get("bundle_manifest_version_id"),
        context="readiness bundle manifest",
    )
    staged_versions = value.get("staged_object_version_ids")
    if (
        not isinstance(staged_versions, dict)
        or set(staged_versions) != _STAGED_OBJECT_VERSION_ROLES
    ):
        raise ValueError("staged readiness VersionId role map mismatch")
    validated_versions = {
        role: _version_id(
            staged_versions[role],
            context=f"staged readiness {role}",
        )
        for role in _STAGED_OBJECT_VERSION_ROLES
    }
    audit_key = _safe_key(
        value.get("artifact_audit_key"),
        field="readiness artifact audit key",
    )
    audit_sha = _require_sha256(
        value.get("artifact_audit_sha256"),
        field="readiness artifact audit SHA-256",
    )
    expected_audit_key = (
        f"campaigns/{context.run_id}/audits/artifact-audit-{audit_sha}.json"
    )
    if audit_key != expected_audit_key:
        raise ValueError("readiness artifact audit binding mismatch")
    bindings = {
        "run_id": context.run_id,
        "descriptor_key": context.descriptor_item["key"],
        "descriptor_sha256": context.descriptor_item["sha256"],
        "descriptor_body_sha256": context.descriptor["descriptor_body_sha256"],
        "campaign_identity_sha256": context.descriptor["campaign_identity_sha256"],
        "bundle_manifest_key": context.manifest_key,
        "bundle_manifest_file_sha256": context.manifest_file_sha256,
        "bundle_manifest_body_sha256": context.manifest["bundle_manifest_body_sha256"],
    }
    for field, expected in bindings.items():
        if value.get(field) != expected:
            raise ValueError(f"staged readiness {field} binding mismatch")
    if descriptor_key != context.descriptor_item["key"]:
        raise ValueError("staged readiness descriptor binding mismatch")
    if (
        manifest_key != context.manifest_key
        or manifest_file_sha != context.manifest_file_sha256
        or manifest_body_sha != context.manifest["bundle_manifest_body_sha256"]
    ):
        raise ValueError("staged readiness bundle manifest binding mismatch")
    value["bundle_manifest_version_id"] = manifest_version
    value["staged_object_version_ids"] = validated_versions
    return dict(value)


def _validate_artifact_audit(
    raw: bytes,
    *,
    context: _BundleContext,
    label: str,
) -> dict[str, object]:
    value = _finite_json(raw, label=label)
    if not isinstance(value, dict) or set(value) != _AUDIT_FIELDS:
        raise ValueError(f"{label} schema mismatch")
    if raw != _canonical(value) + b"\n":
        raise ValueError(f"{label} bytes are not canonical JSON plus LF")
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("record_type") != "glm52_s3_artifact_audit_v1"
        or value.get("audit_pass") is not True
    ):
        raise ValueError(f"{label} schema or pass status mismatch")
    bindings = {
        "run_id": context.run_id,
        "bucket": context.bucket,
        "inventory_body_sha256": context.inventory["inventory_body_sha256"],
    }
    for field, expected in bindings.items():
        if value.get(field) != expected:
            raise ValueError(f"{label} {field} binding mismatch")

    objects = context.inventory["objects"]
    if not isinstance(objects, list):
        raise ValueError("validated artifact inventory objects are unavailable")
    expected_counts = {
        "object_count": len(objects),
        "object_bytes": sum(int(item["size"]) for item in objects),
        "safetensors_object_count": sum(
            item["safetensors"] is True for item in objects
        ),
    }
    for field in (
        "object_count",
        "object_bytes",
        "safetensors_object_count",
        "safetensors_tensor_count",
    ):
        if type(value.get(field)) is not int or int(value[field]) < 0:
            raise ValueError(f"{label} {field} is not a nonnegative integer")
    for field, expected in expected_counts.items():
        if value[field] != expected:
            raise ValueError(f"{label} {field} does not match the inventory")
    tensor_count = int(value["safetensors_tensor_count"])
    safetensors_count = expected_counts["safetensors_object_count"]
    if (safetensors_count == 0 and tensor_count != 0) or (
        safetensors_count > 0 and tensor_count < safetensors_count
    ):
        raise ValueError(
            f"{label} safetensors_tensor_count is inconsistent with the inventory"
        )
    return dict(value)


def _authenticate_referenced_audit(
    *,
    s3: object,
    context: _BundleContext,
    ready: Mapping[str, object],
) -> tuple[str, str]:
    audit_key = str(ready["artifact_audit_key"])
    versions = ready["staged_object_version_ids"]
    if not isinstance(versions, Mapping):
        raise ValueError("readiness VersionId map is unavailable")
    remote = _read_exact_bytes(
        s3=s3,
        bucket=context.bucket,
        key=audit_key,
        version_id=versions["artifact_audit"],
        label="referenced artifact audit",
    )
    if hashlib.sha256(remote.raw).hexdigest() != ready["artifact_audit_sha256"]:
        raise ValueError("referenced artifact audit SHA-256 drift")
    _validate_artifact_audit(
        remote.raw,
        context=context,
        label="referenced artifact audit",
    )
    return audit_key, remote.version_id


def _authenticate_remote_manifest(
    *,
    s3: object,
    context: _BundleContext,
    ready: Mapping[str, object],
) -> tuple[dict[str, object], tuple[dict[str, object], ...], str]:
    manifest_key = str(ready["bundle_manifest_key"])
    remote = _read_exact_bytes(
        s3=s3,
        bucket=context.bucket,
        key=manifest_key,
        version_id=ready["bundle_manifest_version_id"],
        label="remote bundle manifest",
    )
    if hashlib.sha256(remote.raw).hexdigest() != ready[
        "bundle_manifest_file_sha256"
    ]:
        raise ValueError("remote bundle manifest file SHA-256 mismatch")
    manifest, files = _validate_manifest_bytes(remote.raw)
    if (
        manifest_key != context.manifest_key
        or manifest["bundle_manifest_body_sha256"]
        != ready["bundle_manifest_body_sha256"]
        or manifest["run_id"] != context.run_id
        or manifest["bucket"] != context.bucket
        or manifest["descriptor_body_sha256"]
        != context.descriptor["descriptor_body_sha256"]
        or remote.raw != context.manifest_raw
    ):
        raise ValueError("remote bundle manifest authority mismatch")
    descriptor_items = [
        item for item in files if item["role"] == "descriptor"
    ]
    if len(descriptor_items) != 1 or descriptor_items[0] != context.descriptor_item:
        raise ValueError("remote bundle manifest descriptor binding mismatch")
    return manifest, tuple(files), remote.version_id


def _load_readiness_winner(
    *,
    s3: object,
    context: _BundleContext,
) -> _ReadinessWinner | None:
    remote = _read_current_bytes(
        s3=s3,
        bucket=context.bucket,
        key=context.ready_key,
    )
    if remote is None:
        return None
    ready = _validate_readiness(remote.raw, context=context)
    _manifest, remote_files, manifest_version = _authenticate_remote_manifest(
        s3=s3,
        context=context,
        ready=ready,
    )
    staged_versions = ready["staged_object_version_ids"]
    if not isinstance(staged_versions, Mapping):
        raise ValueError("readiness VersionId map is unavailable")
    versions: dict[str, str] = {"bundle_manifest": manifest_version}
    for item in remote_files:
        role = str(item["role"])
        exact = _read_exact_bytes(
            s3=s3,
            bucket=context.bucket,
            key=str(item["key"]),
            version_id=staged_versions[role],
            label=f"readiness bundle role {role}",
        )
        if (
            len(exact.raw) != int(item["size"])
            or hashlib.sha256(exact.raw).hexdigest() != item["sha256"]
        ):
            raise ValueError(
                f"readiness bundle role exact-version drift: {role}"
            )
        versions[role] = exact.version_id
    _audit_key, audit_version = _authenticate_referenced_audit(
        s3=s3,
        context=context,
        ready=ready,
    )
    versions["artifact_audit"] = audit_version
    versions["ready"] = remote.version_id
    if set(versions) != _RECEIPT_VERSION_ROLES:
        raise ValueError("readiness bundle role authentication is incomplete")
    return _ReadinessWinner(
        value=ready,
        raw=remote.raw,
        version_id=remote.version_id,
        s3_version_ids=versions,
    )


def _receipt(
    *,
    context: _BundleContext,
    winner: _ReadinessWinner,
    staged: Mapping[str, str],
) -> dict[str, object]:
    return {
        "run_id": context.run_id,
        "bucket": context.bucket,
        "descriptor_key": context.descriptor_item["key"],
        "ready_key": context.ready_key,
        "artifact_audit_key": winner.value["artifact_audit_key"],
        "staged": dict(staged),
        "staged_readiness_file_sha256": hashlib.sha256(winner.raw).hexdigest(),
        "staged_readiness_body_sha256": winner.value["ready_body_sha256"],
        "staged_readiness_version_id": winner.version_id,
        "s3_version_ids": dict(winner.s3_version_ids),
    }


def _stage_bundle(
    *,
    bundle: Path,
    s3: object,
    audit_output: Path,
    run_audit: object,
    clock: object,
    prepared_context: _BundleContext | None = None,
) -> dict[str, object]:
    context = prepared_context or _prepare_bundle(bundle)
    existing = _load_readiness_winner(s3=s3, context=context)
    if existing is not None:
        return _receipt(
            context=context,
            winner=existing,
            staged={role: "reused" for role in _RECEIPT_VERSION_ROLES},
        )

    staged: dict[str, str] = {}
    version_ids: dict[str, str] = {}
    for item in context.files:
        role = str(item["role"])
        if role == "descriptor":
            continue
        result = _put_immutable(
            profile="",
            region="",
            bucket=context.bucket,
            key=str(item["key"]),
            path=context.local[role],
            run_id=context.run_id,
            s3=s3,
            record_type=(
                _REPOSITORY_ARCHIVE_RECORD_TYPE
                if role == "repository_tar"
                else None
            ),
        )
        staged[role] = result.disposition
        version_ids[role] = result.version_id

    run_audit(context.local["artifact_inventory"], audit_output)
    _validate_artifact_audit(
        audit_output.read_bytes(),
        context=context,
        label="fresh artifact audit",
    )
    audit_sha = _sha(audit_output)
    audit_key = f"campaigns/{context.run_id}/audits/artifact-audit-{audit_sha}.json"
    audit_result = _put_immutable(
        profile="",
        region="",
        bucket=context.bucket,
        key=audit_key,
        path=audit_output,
        run_id=context.run_id,
        s3=s3,
    )
    staged["artifact_audit"] = audit_result.disposition
    version_ids["artifact_audit"] = audit_result.version_id

    manifest_result = _put_immutable(
        profile="",
        region="",
        bucket=context.bucket,
        key=context.manifest_key,
        path=context.manifest_path,
        run_id=context.run_id,
        s3=s3,
    )
    staged["bundle_manifest"] = manifest_result.disposition
    version_ids["bundle_manifest"] = manifest_result.version_id

    descriptor_result = _put_immutable(
        profile="",
        region="",
        bucket=context.bucket,
        key=str(context.descriptor_item["key"]),
        path=context.local["descriptor"],
        run_id=context.run_id,
        s3=s3,
    )
    staged["descriptor"] = descriptor_result.disposition
    version_ids["descriptor"] = descriptor_result.version_id

    observed = clock()
    if not isinstance(observed, datetime) or observed.tzinfo is None:
        raise ValueError("staging clock must return an aware datetime")
    staged_at = (
        observed.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    ready_body = {
        "schema_version": 2,
        "record_type": "glm52_staged_control_plane_ready_v2",
        "run_id": context.run_id,
        "descriptor_key": context.descriptor_item["key"],
        "descriptor_sha256": context.descriptor_item["sha256"],
        "descriptor_body_sha256": context.descriptor["descriptor_body_sha256"],
        "campaign_identity_sha256": context.descriptor["campaign_identity_sha256"],
        "bundle_manifest_key": context.manifest_key,
        "bundle_manifest_file_sha256": context.manifest_file_sha256,
        "bundle_manifest_body_sha256": context.manifest["bundle_manifest_body_sha256"],
        "bundle_manifest_version_id": manifest_result.version_id,
        "staged_object_version_ids": {
            role: version_ids[role]
            for role in _STAGED_OBJECT_VERSION_ROLES
        },
        "artifact_audit_key": audit_key,
        "artifact_audit_sha256": audit_sha,
        "staged_at": staged_at,
    }
    ready = {
        **ready_body,
        "ready_body_sha256": hashlib.sha256(_canonical(ready_body)).hexdigest(),
    }
    with tempfile.TemporaryDirectory(prefix="glm52-stage-ready-") as directory:
        ready_path = Path(directory) / "STAGED_CONTROL_PLANE_READY.json"
        ready_path.write_bytes(_canonical(ready) + b"\n")
        ready_sha = _sha(ready_path)
        put = s3.put_object(
            key=context.ready_key,
            path=ready_path,
            run_id=context.run_id,
            expected_sha=ready_sha,
        )
        if put.returncode == 0:
            response = json.loads(put.stdout)
            if not isinstance(response, dict):
                raise ValueError("readiness S3 PUT returned a non-object")
            ready_version = _version_id(
                response.get("VersionId"),
                context=(f"readiness S3 PUT s3://{context.bucket}/{context.ready_key}"),
            )
            _authenticate_version(
                s3=s3,
                bucket=context.bucket,
                key=context.ready_key,
                version_id=ready_version,
                expected_size=ready_path.stat().st_size,
                expected_sha=ready_sha,
            )
            version_ids["ready"] = ready_version
            staged["ready"] = "uploaded"
            winner = _ReadinessWinner(
                value=ready,
                raw=ready_path.read_bytes(),
                version_id=ready_version,
                s3_version_ids=version_ids,
            )
        else:
            original_failure = _command_error(put)
            try:
                winner = _load_readiness_winner(
                    s3=s3,
                    context=context,
                )
            except (RuntimeError, TypeError, ValueError) as recovery_error:
                raise RuntimeError(
                    f"{original_failure}; readiness winner recovery failed: "
                    f"{recovery_error}"
                ) from recovery_error
            if winner is None:
                raise RuntimeError(f"{original_failure}; readiness winner is absent")
            staged = {
                role: "reused-after-race"
                for role in _RECEIPT_VERSION_ROLES
            }
    if set(staged) != _RECEIPT_VERSION_ROLES:
        raise ValueError("staging receipt disposition roles are incomplete")
    if set(winner.s3_version_ids) != _RECEIPT_VERSION_ROLES:
        raise ValueError("staging receipt VersionId roles are incomplete")
    return _receipt(
        context=context,
        winner=winner,
        staged=staged,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--checksum-authority", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    if args.profile == "default":
        parser.error("the default AWS profile is forbidden")
    try:
        context = _prepare_bundle(args.bundle)
    except (RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    subprocess.run(
        [str(REPO_ROOT / "aws/glm52-gpu/scripts/assert_rnd_aws_account.sh")],
        check=True,
        env={**os.environ, "AWS_PROFILE": args.profile},
    )
    s3 = _AwsCliS3(
        profile=args.profile,
        region=args.region,
        bucket=context.bucket,
    )
    audit_output = args.audit_output or args.bundle / "artifact-audit-v1.json"

    def run_audit(inventory_path: Path, output: Path) -> None:
        subprocess.run(
            [
                str(REPO_ROOT / AUDIT_SCRIPT_RELATIVE),
                "--inventory",
                str(inventory_path),
                "--output",
                str(output),
                "--profile",
                args.profile,
                "--region",
                args.region,
                "--checksum-authority",
                str(args.checksum_authority),
            ],
            check=True,
        )

    try:
        receipt = _stage_bundle(
            bundle=args.bundle,
            s3=s3,
            audit_output=audit_output,
            run_audit=run_audit,
            clock=lambda: datetime.now(timezone.utc),
            prepared_context=context,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
