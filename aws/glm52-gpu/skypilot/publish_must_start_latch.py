#!/usr/bin/env python3
"""Authenticate a SkyPilot worker and gate it on exact must-start acceptance.

This file intentionally uses only the Python standard library.  It runs before
the repository archive or campaign Python environment exists.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Mapping, Sequence
from urllib.parse import urlparse

APPROVED_ACCOUNT_ID = "246813579024"
APPROVED_REGION = "us-west-2"
_HEX = set("0123456789abcdef")
_DESCRIPTOR_BODY_FIELDS = {
    "schema_version",
    "record_type",
    "run_id",
    "campaign_identity_sha256",
    "account_id",
    "provider",
    "region",
    "instance_type",
    "instance_count",
    "use_spot",
    "max_hourly_cost_usd",
    "approved_gpu_runtime_seconds",
    "approved_gpu_cost_usd",
    "must_start_by",
    "skypilot_version",
    "task_name",
    "controller_identity",
    "worker_identity",
    "vpc_name",
    "image_id",
    "bucket",
    "jobs_bucket",
    "repo_tar_key",
    "repo_tar_sha256",
    "campaign_descriptor_key",
    "approval_key",
    "approval_sha256",
    "artifacts",
}
_SUBMISSION_BODY_FIELDS = {
    "schema_version",
    "record_type",
    "run_id",
    "descriptor_body_sha256",
    "submitted_at",
    "must_start_by",
    "sky_job_name",
}
_AUTHORITY_FIELDS = (
    "run_id",
    "managed_mode",
    "account_id",
    "region",
    "bucket",
    "descriptor_body_sha256",
    "submission_body_sha256",
    "sky_job_name",
    "must_start_by",
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_hex(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{field} is invalid")
    return value


def _parse_time(value: object, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _load_json_bytes(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return value


def _parse_s3_uri(uri: str, *, field: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.startswith("/")
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{field} is invalid")
    return parsed.netloc, parsed.path[1:]


def _load_policy(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("_glm52_must_start_policy", path)
    if spec is None or spec.loader is None:
        raise ValueError("must-start policy module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_descriptor(
    raw: bytes,
    *,
    file_sha256: str,
    s3_uri: str,
) -> dict[str, object]:
    if _sha256_bytes(raw) != _require_hex(
        file_sha256, field="descriptor file SHA-256"
    ):
        raise ValueError("descriptor file SHA-256 mismatch")
    value = _load_json_bytes(raw, label="descriptor")
    if set(value) != _DESCRIPTOR_BODY_FIELDS | {"descriptor_body_sha256"}:
        raise ValueError("descriptor schema mismatch")
    body = dict(value)
    actual = body.pop("descriptor_body_sha256")
    if actual != _sha256_bytes(_canonical_bytes(body)):
        raise ValueError("descriptor body SHA-256 mismatch")
    if (
        value.get("schema_version") != 2
        or value.get("record_type") != "glm52_sky_campaign_descriptor_v2"
        or value.get("account_id") != APPROVED_ACCOUNT_ID
        or value.get("provider") != "aws"
        or value.get("region") != APPROVED_REGION
        or value.get("instance_type") != "p5.48xlarge"
        or value.get("instance_count") != 1
        or value.get("use_spot") is not False
        or value.get("skypilot_version") != "0.13.0"
    ):
        raise ValueError("descriptor AWS worker authority mismatch")
    bucket, key = _parse_s3_uri(s3_uri, field="descriptor S3 URI")
    if (
        bucket != value.get("bucket")
        or key != value.get("campaign_descriptor_key")
    ):
        raise ValueError("descriptor S3 URI authority mismatch")
    for field in (
        "descriptor_body_sha256",
        "repo_tar_sha256",
        "approval_sha256",
    ):
        _require_hex(value.get(field), field=field)
    _parse_time(value.get("must_start_by"), field="descriptor must_start_by")
    return value


def _validate_submission(
    raw: bytes,
    *,
    expected_body_sha256: str,
    s3_uri: str,
    descriptor: Mapping[str, object],
    descriptor_file_sha256: str,
    managed_mode: str,
    policy: ModuleType,
) -> dict[str, object]:
    value = _load_json_bytes(raw, label="immutable submission")
    if set(value) != _SUBMISSION_BODY_FIELDS | {"submission_body_sha256"}:
        raise ValueError("immutable submission schema mismatch")
    body = dict(value)
    actual = body.pop("submission_body_sha256")
    expected = _sha256_bytes(_canonical_bytes(body))
    if actual != expected or actual != _require_hex(
        expected_body_sha256, field="submission body SHA-256"
    ):
        raise ValueError("immutable submission body SHA-256 mismatch")
    if (
        value.get("schema_version") != 1
        or value.get("record_type") != "glm52_skypilot_submission_v1"
        or value.get("run_id") != descriptor.get("run_id")
        or value.get("descriptor_body_sha256")
        != descriptor.get("descriptor_body_sha256")
        or value.get("must_start_by") != descriptor.get("must_start_by")
        or value.get("sky_job_name")
        != policy.expected_sky_job_name(str(value.get("run_id")), managed_mode)
    ):
        raise ValueError("immutable submission authority mismatch")
    submitted = _parse_time(
        value.get("submitted_at"), field="submission submitted_at"
    )
    deadline = _parse_time(
        value.get("must_start_by"), field="submission must_start_by"
    )
    if submitted >= deadline:
        raise ValueError("immutable submission deadline is invalid")
    bucket, key = _parse_s3_uri(s3_uri, field="submission S3 URI")
    expected_key = (
        f"campaigns/{value['run_id']}/monitor/submission-locks/"
        f"{descriptor_file_sha256}-{managed_mode}.json"
    )
    if bucket != descriptor.get("bucket") or key != expected_key:
        raise ValueError("immutable submission key authority mismatch")
    return value


def _validate_sts(
    identity: Mapping[str, object],
    *,
    worker_role_arn: str,
) -> None:
    if identity.get("Account") != APPROVED_ACCOUNT_ID:
        raise ValueError("STS account authority mismatch")
    role_prefix = f"arn:aws:iam::{APPROVED_ACCOUNT_ID}:role/"
    if not worker_role_arn.startswith(role_prefix):
        raise ValueError("descriptor worker role ARN is invalid")
    role_name = worker_role_arn.rsplit("/", 1)[-1]
    expected_prefix = (
        f"arn:aws:sts::{APPROVED_ACCOUNT_ID}:assumed-role/{role_name}/"
    )
    if not str(identity.get("Arn", "")).startswith(expected_prefix):
        raise ValueError("STS assumed role authority mismatch")


def build_authenticated_worker_latch(
    *,
    policy_path: Path,
    descriptor_bytes: bytes,
    descriptor_s3_uri: str,
    descriptor_file_sha256: str,
    submission_bytes: bytes,
    submission_s3_uri: str,
    submission_body_sha256: str,
    managed_mode: str,
    sts_identity: Mapping[str, object],
    identity_document_bytes: bytes,
    entrypoint_observed_at: datetime,
) -> tuple[dict[str, object], str]:
    policy = _load_policy(policy_path)
    descriptor = _validate_descriptor(
        descriptor_bytes,
        file_sha256=descriptor_file_sha256,
        s3_uri=descriptor_s3_uri,
    )
    submission = _validate_submission(
        submission_bytes,
        expected_body_sha256=submission_body_sha256,
        s3_uri=submission_s3_uri,
        descriptor=descriptor,
        descriptor_file_sha256=descriptor_file_sha256,
        managed_mode=managed_mode,
        policy=policy,
    )
    worker_role_arn = str(descriptor["worker_identity"])
    _validate_sts(sts_identity, worker_role_arn=worker_role_arn)
    identity = _load_json_bytes(
        identity_document_bytes,
        label="IMDS instance identity document",
    )
    expected_identity = {
        "accountId": APPROVED_ACCOUNT_ID,
        "region": APPROVED_REGION,
        "instanceType": descriptor["instance_type"],
        "imageId": descriptor["image_id"],
    }
    for field, expected in expected_identity.items():
        if identity.get(field) != expected:
            label = {
                "accountId": "account",
                "instanceType": "instance type",
                "imageId": "AMI",
            }.get(field, field)
            raise ValueError(f"IMDS {label} authority mismatch")
    instance_id = identity.get("instanceId")
    if not isinstance(instance_id, str) or not instance_id.startswith("i-"):
        raise ValueError("IMDS instance ID is invalid")
    pending_time = _parse_time(
        identity.get("pendingTime"), field="IMDS pendingTime"
    )
    if entrypoint_observed_at.tzinfo is None:
        raise ValueError("worker entrypoint observation must be timezone-aware")
    descriptor_bucket, descriptor_key = _parse_s3_uri(
        descriptor_s3_uri, field="descriptor S3 URI"
    )
    submission_bucket, submission_key = _parse_s3_uri(
        submission_s3_uri, field="submission S3 URI"
    )
    if descriptor_bucket != submission_bucket:
        raise ValueError("descriptor and submission buckets differ")
    latch = policy.build_worker_start_latch(
        run_id=str(descriptor["run_id"]),
        managed_mode=managed_mode,
        account_id=APPROVED_ACCOUNT_ID,
        region=APPROVED_REGION,
        bucket=descriptor_bucket,
        descriptor_body_sha256=str(descriptor["descriptor_body_sha256"]),
        submission_body_sha256=str(submission["submission_body_sha256"]),
        sky_job_name=str(submission["sky_job_name"]),
        must_start_by=str(submission["must_start_by"]),
        descriptor_key=descriptor_key,
        descriptor_file_sha256=descriptor_file_sha256,
        submission_key=submission_key,
        submission_submitted_at=str(submission["submitted_at"]),
        repo_tar_sha256=str(descriptor["repo_tar_sha256"]),
        instance_id=instance_id,
        instance_type=str(identity["instanceType"]),
        image_id=str(identity["imageId"]),
        worker_role_arn=worker_role_arn,
        instance_identity_document_sha256=_sha256_bytes(
            identity_document_bytes
        ),
        ec2_pending_time=pending_time,
        entrypoint_observed_at=entrypoint_observed_at,
    )
    key = (
        f"campaigns/{descriptor['run_id']}/monitor/must-start/"
        f"{managed_mode}/{submission['submission_body_sha256']}/worker-latches/"
        f"{instance_id}/{latch['worker_latch_body_sha256']}.json"
    )
    return latch, key


def validate_accepted_for_worker(
    *,
    policy_path: Path,
    accepted: Mapping[str, object],
    latch: Mapping[str, object],
    latch_key: str | None,
    allow_same_submission_recovery: bool = False,
) -> dict[str, object]:
    for field in _AUTHORITY_FIELDS:
        if accepted.get(field) != latch.get(field):
            raise ValueError(f"accepted start {field} authority mismatch")
    policy = _load_policy(policy_path)
    authenticated_latch = policy.validate_worker_start_latch(latch)
    authenticated = policy.validate_timely_start_accepted(accepted)
    if authenticated["source_kind"] != "worker-latch":
        raise ValueError("accepted start does not bind a worker latch")
    if allow_same_submission_recovery and latch_key is None:
        return authenticated
    if latch_key is None:
        raise ValueError("current worker latch key is required")
    if authenticated["source_key"] != latch_key:
        raise ValueError("accepted start source key mismatch")
    if authenticated["source_body_sha256"] != policy.canonical_sha256(
        authenticated_latch
    ):
        raise ValueError("accepted start source body SHA-256 mismatch")
    if authenticated["started_at"] != authenticated_latch[
        "entrypoint_observed_at"
    ]:
        raise ValueError("accepted start timestamp mismatch")
    return authenticated


def validate_binding_for_worker(
    *,
    policy_path: Path,
    binding: Mapping[str, object],
    accepted: Mapping[str, object],
    latch: Mapping[str, object],
) -> dict[str, object]:
    """Authenticate the controller's numeric job binding for this worker."""

    for field in _AUTHORITY_FIELDS:
        if binding.get(field) != latch.get(field):
            raise ValueError(f"job binding {field} authority mismatch")
        if accepted.get(field) != latch.get(field):
            raise ValueError(f"accepted start {field} authority mismatch")
    policy = _load_policy(policy_path)
    authenticated_latch = policy.validate_worker_start_latch(latch)
    authenticated_accepted = policy.validate_timely_start_accepted(accepted)
    authenticated_binding = policy.validate_must_start_job_binding(binding)
    exact_fields = (
        "descriptor_key",
        "descriptor_file_sha256",
        "submission_key",
        "submission_submitted_at",
    )
    for field in exact_fields:
        if authenticated_binding[field] != authenticated_latch[field]:
            raise ValueError(f"job binding {field} authority mismatch")
    if authenticated_binding["sky_job_name"] != authenticated_accepted[
        "sky_job_name"
    ]:
        raise ValueError("job binding sky job name authority mismatch")
    return authenticated_binding


def _run_json(command: Sequence[str]) -> dict[str, object]:
    completed = subprocess.run(
        list(command),
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {completed.stderr.strip()}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("command did not return JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError("command JSON must be an object")
    return value


def _read_imds_identity(base_url: str) -> bytes:
    token_request = urllib.request.Request(
        f"{base_url.rstrip('/')}/latest/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "300"},
    )
    try:
        with urllib.request.urlopen(token_request, timeout=5) as response:
            token = response.read().decode("utf-8")
        identity_request = urllib.request.Request(
            (
                f"{base_url.rstrip('/')}/latest/dynamic/"
                "instance-identity/document"
            ),
            headers={"X-aws-ec2-metadata-token": token},
        )
        with urllib.request.urlopen(identity_request, timeout=5) as response:
            return response.read()
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError("IMDSv2 identity lookup failed") from error


def _aws_read_json(
    aws_bin: str,
    *,
    region: str,
    service_args: Sequence[str],
) -> dict[str, object]:
    return _run_json(
        [
            aws_bin,
            *service_args,
            "--region",
            region,
            "--output",
            "json",
            "--no-cli-pager",
        ]
    )


def _aws_s3_stdout(
    aws_bin: str,
    *,
    region: str,
    uri: str,
) -> bytes | None:
    completed = subprocess.run(
        [
            aws_bin,
            "s3",
            "cp",
            uri,
            "-",
            "--region",
            region,
            "--only-show-errors",
            "--no-cli-pager",
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode == 0:
        return completed.stdout
    stderr = completed.stderr.decode("utf-8", errors="replace")
    if "NoSuchKey" in stderr or "404" in stderr or "Not Found" in stderr:
        return None
    raise RuntimeError(f"S3 read failed: {stderr.strip()}")


def _put_latch(
    aws_bin: str,
    *,
    region: str,
    bucket: str,
    key: str,
    latch: Mapping[str, object],
    policy_path: Path,
) -> None:
    body = _canonical_bytes(latch)
    with tempfile.NamedTemporaryFile(prefix="glm52-worker-latch-", suffix=".json") as file:
        file.write(body)
        file.flush()
        completed = subprocess.run(
            [
                aws_bin,
                "s3api",
                "put-object",
                "--bucket",
                bucket,
                "--key",
                key,
                "--body",
                file.name,
                "--if-none-match",
                "*",
                "--content-type",
                "application/json",
                "--metadata",
                (
                    f"glm52-run-id={latch['run_id']},"
                    f"glm52-body-sha256={latch['worker_latch_body_sha256']}"
                ),
                "--region",
                region,
                "--no-cli-pager",
            ],
            check=False,
            text=True,
            capture_output=True,
        )
    if completed.returncode == 0:
        return
    if (
        "PreconditionFailed" not in completed.stderr
        and "412" not in completed.stderr
    ):
        raise RuntimeError(f"worker latch publish failed: {completed.stderr.strip()}")
    winner = _aws_s3_stdout(
        aws_bin,
        region=region,
        uri=f"s3://{bucket}/{key}",
    )
    if winner is None:
        raise RuntimeError("worker latch conditional winner disappeared")
    value = _load_json_bytes(winner, label="worker latch winner")
    policy = _load_policy(policy_path)
    if policy.validate_worker_start_latch(value) != dict(latch):
        raise ValueError("worker latch conditional winner is foreign")


def _accepted_uri(latch: Mapping[str, object]) -> str:
    return (
        f"s3://{latch['bucket']}/campaigns/{latch['run_id']}/monitor/"
        f"must-start/{latch['managed_mode']}/{latch['submission_body_sha256']}/"
        "TIMELY_START_ACCEPTED.json"
    )


def _binding_uri(latch: Mapping[str, object]) -> str:
    return (
        f"s3://{latch['bucket']}/campaigns/{latch['run_id']}/monitor/"
        f"must-start/{latch['managed_mode']}/{latch['submission_body_sha256']}/"
        "JOB_BINDING.json"
    )


def _read_accepted(
    aws_bin: str,
    *,
    region: str,
    latch: Mapping[str, object],
) -> dict[str, object] | None:
    raw = _aws_s3_stdout(
        aws_bin,
        region=region,
        uri=_accepted_uri(latch),
    )
    return (
        None
        if raw is None
        else _load_json_bytes(raw, label="accepted start marker")
    )


def _read_binding(
    aws_bin: str,
    *,
    region: str,
    latch: Mapping[str, object],
) -> dict[str, object] | None:
    raw = _aws_s3_stdout(
        aws_bin,
        region=region,
        uri=_binding_uri(latch),
    )
    return (
        None
        if raw is None
        else _load_json_bytes(raw, label="must-start job binding")
    )


def _write_accepted(path: Path, accepted: Mapping[str, object]) -> None:
    path.write_bytes(_canonical_bytes(accepted) + b"\n")


def _build_runtime_latch(args: argparse.Namespace) -> tuple[dict[str, object], str]:
    descriptor_path = Path(args.descriptor)
    submission_path = Path(args.submission)
    sts = _aws_read_json(
        args.aws_bin,
        region=APPROVED_REGION,
        service_args=("sts", "get-caller-identity"),
    )
    identity_document = _read_imds_identity(args.imds_base_url)
    return build_authenticated_worker_latch(
        policy_path=Path(args.policy),
        descriptor_bytes=descriptor_path.read_bytes(),
        descriptor_s3_uri=args.descriptor_s3_uri,
        descriptor_file_sha256=args.descriptor_file_sha256,
        submission_bytes=submission_path.read_bytes(),
        submission_s3_uri=args.submission_s3_uri,
        submission_body_sha256=args.submission_body_sha256,
        managed_mode=args.managed_mode,
        sts_identity=sts,
        identity_document_bytes=identity_document,
        entrypoint_observed_at=datetime.now(timezone.utc),
    )


def _publish(args: argparse.Namespace) -> int:
    latch, key = _build_runtime_latch(args)
    deadline = _parse_time(latch["must_start_by"], field="must_start_by")
    now = datetime.now(timezone.utc)
    existing = _read_accepted(
        args.aws_bin,
        region=APPROVED_REGION,
        latch=latch,
    )
    if now > deadline:
        if existing is None:
            raise RuntimeError(
                "must-start deadline expired without accepted first start"
            )
        validate_accepted_for_worker(
            policy_path=Path(args.policy),
            accepted=existing,
            latch=latch,
            latch_key=None,
            allow_same_submission_recovery=True,
        )
        _put_latch(
            args.aws_bin,
            region=APPROVED_REGION,
            bucket=str(latch["bucket"]),
            key=key,
            latch=latch,
            policy_path=Path(args.policy),
        )
        _write_accepted(Path(args.accepted_output), existing)
        print(json.dumps({"status": "accepted-recovery", "latch_key": key}))
        return 0
    _put_latch(
        args.aws_bin,
        region=APPROVED_REGION,
        bucket=str(latch["bucket"]),
        key=key,
        latch=latch,
        policy_path=Path(args.policy),
    )
    stop_at = min(
        deadline.timestamp(),
        time.time() + args.wait_seconds,
    )
    while True:
        accepted = _read_accepted(
            args.aws_bin,
            region=APPROVED_REGION,
            latch=latch,
        )
        if accepted is not None:
            validate_accepted_for_worker(
                policy_path=Path(args.policy),
                accepted=accepted,
                latch=latch,
                latch_key=key,
            )
            _write_accepted(Path(args.accepted_output), accepted)
            print(json.dumps({"status": "accepted", "latch_key": key}))
            return 0
        if time.time() >= stop_at:
            raise RuntimeError(
                "worker latch was not accepted before the bounded deadline"
            )
        time.sleep(min(args.poll_seconds, max(0.0, stop_at - time.time())))


def _verify(args: argparse.Namespace) -> int:
    latch, _key = _build_runtime_latch(args)
    stop_at = time.time() + args.wait_seconds
    while True:
        accepted = _read_accepted(
            args.aws_bin,
            region=APPROVED_REGION,
            latch=latch,
        )
        if accepted is None:
            raise RuntimeError("accepted start marker is missing before spend")
        validate_accepted_for_worker(
            policy_path=Path(args.policy),
            accepted=accepted,
            latch=latch,
            latch_key=None,
            allow_same_submission_recovery=True,
        )
        binding = _read_binding(
            args.aws_bin,
            region=APPROVED_REGION,
            latch=latch,
        )
        if binding is not None:
            authenticated = validate_binding_for_worker(
                policy_path=Path(args.policy),
                binding=binding,
                accepted=accepted,
                latch=latch,
            )
            _write_accepted(Path(args.accepted_output), accepted)
            _write_accepted(Path(args.binding_output), authenticated)
            print(
                json.dumps(
                    {
                        "status": "accepted-before-spend",
                        "controller_job_id": authenticated["target_job_id"],
                    }
                )
            )
            return 0
        if time.time() >= stop_at:
            raise RuntimeError(
                "authenticated controller job binding is missing before spend"
            )
        time.sleep(min(args.poll_seconds, max(0.0, stop_at - time.time())))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("publish", "verify-accepted"):
        child = subparsers.add_parser(command)
        child.add_argument("--policy", required=True)
        child.add_argument("--descriptor", required=True)
        child.add_argument("--descriptor-s3-uri", required=True)
        child.add_argument("--descriptor-file-sha256", required=True)
        child.add_argument("--submission", required=True)
        child.add_argument("--submission-s3-uri", required=True)
        child.add_argument("--submission-body-sha256", required=True)
        child.add_argument(
            "--managed-mode",
            required=True,
            choices=("production", "qualification", "cache-seed"),
        )
        child.add_argument("--accepted-output", required=True)
        child.add_argument("--aws-bin", default="aws")
        child.add_argument(
            "--imds-base-url",
            default="http://169.254.169.254",
        )
        if command == "publish":
            child.add_argument("--wait-seconds", type=int, default=90)
            child.add_argument("--poll-seconds", type=float, default=5.0)
        else:
            child.add_argument("--binding-output", required=True)
            child.add_argument("--wait-seconds", type=int, default=120)
            child.add_argument("--poll-seconds", type=float, default=5.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "publish":
        if args.wait_seconds <= 0 or args.poll_seconds <= 0:
            raise ValueError("worker acceptance wait must be positive")
        return _publish(args)
    if args.wait_seconds <= 0 or args.poll_seconds <= 0:
        raise ValueError("worker binding wait must be positive")
    return _verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
