"""Injected qualification-only I/O for dynamic-v2 worker-start authority.

The future Lambda handler supplies clients and time.  This module deliberately
does not construct SDK clients, read environment configuration, or perform
I/O at import time.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from typing import Literal

try:
    from glm52_sky_campaign import (  # type: ignore[import-not-found]
        SkyCampaignValidationError,
        validate_sky_campaign_descriptor,
    )
    from glm52_sky_must_start import (  # type: ignore[import-not-found]
        MustStartValidationError,
        build_must_start_controller_observation,
        validate_must_start_controller_observation,
    )
    from glm52_sky_must_start_dynamic import (  # type: ignore[import-not-found]
        DynamicJobBindingValidationError,
        dynamic_v2_job_binding_s3_key,
        validate_dynamic_v2_job_binding,
    )
    from glm52_sky_worker_must_start_v2 import (  # type: ignore[import-not-found]
        WorkerStartValidationError,
        build_worker_start_accepted_v2,
        decide_worker_start_acceptance_v2,
        validate_worker_start_accepted_v2,
        validate_worker_start_latch_v2,
        worker_start_accepted_v2_s3_key,
        worker_start_latch_v2_s3_key,
        worker_v2_canonical_bytes,
    )
except ModuleNotFoundError:
    from mlx_vq.quality.glm52_sky_campaign import (
        SkyCampaignValidationError,
        validate_sky_campaign_descriptor,
    )
    from mlx_vq.quality.glm52_sky_must_start import (
        MustStartValidationError,
        build_must_start_controller_observation,
        validate_must_start_controller_observation,
    )
    from mlx_vq.quality.glm52_sky_must_start_dynamic import (
        DynamicJobBindingValidationError,
        dynamic_v2_job_binding_s3_key,
        validate_dynamic_v2_job_binding,
    )
    from mlx_vq.quality.glm52_sky_worker_must_start_v2 import (
        WorkerStartValidationError,
        build_worker_start_accepted_v2,
        decide_worker_start_acceptance_v2,
        validate_worker_start_accepted_v2,
        validate_worker_start_latch_v2,
        worker_start_accepted_v2_s3_key,
        worker_start_latch_v2_s3_key,
        worker_v2_canonical_bytes,
    )

APPROVED_ACCOUNT_ID = "246813579024"
APPROVED_REGION = "us-west-2"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BUCKET = re.compile(
    r"^(?![0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$)"
    r"(?!xn--)(?!sthree-)(?!amzn_s3_demo_)"
    r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$"
)
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_INSTANCE_ID = re.compile(r"^i-[0-9a-f]{17}$")
_VERSION_ID = re.compile(r"^[A-Za-z0-9._+=:/-]{1,1024}$")
_CALLER_ARN = re.compile(
    r"^arn:aws:(?:sts|iam)::([0-9]{12}):(?:assumed-role|role|user)/.+$"
)
_ETAG = re.compile(r'^"[0-9a-f]{32}"$')
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")
_AMI_ID = re.compile(r"^ami-[0-9a-f]{8,17}$")
_PROFILE_ARN = re.compile(
    r"^arn:aws:iam::246813579024:instance-profile/[A-Za-z0-9+=,.@_/-]+$"
)
_CONTROLLER_HISTORY_PREFIX = "GLM52_SUBMISSION_HISTORY="
_WORKER_PROFILE_ARN = "arn:aws:iam::246813579024:instance-profile/keep-glm52-gpu-worker"
_WORKER_ROLE_ARN = "arn:aws:iam::246813579024:role/keep-glm52-gpu-worker"
_ACTIVE_STATES = frozenset({"pending", "running", "stopping"})
_CURRENT_WORKER_STATES = frozenset({"pending", "running"})
_CONTROLLER_STATUSES = frozenset(
    {
        "PENDING",
        "STARTING",
        "RUNNING",
        "RECOVERING",
        "CANCELLING",
        "SUCCEEDED",
        "FAILED",
        "FAILED_SETUP",
        "FAILED_PRECHECKS",
        "FAILED_NO_RESOURCE",
        "FAILED_CONTROLLER",
        "CANCELLED",
    }
)
_WORKER_SCHEDULE_STATES = frozenset(
    {
        "INACTIVE",
        "WAITING",
        "LAUNCHING",
        "ALIVE",
        "ALIVE_BACKOFF",
        "ALIVE_WAITING",
        "DONE",
    }
)
_SUBMISSION_ACCEPTED_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "intent_body_sha256",
        "sky_job_id",
        "sky_job_name",
        "workspace",
        "controller_submitted_at",
        "controller_status",
        "controller_identity",
        "controller_instance_id",
        "controller_instance_type",
        "controller_profile_arn",
        "controller_cluster_name",
        "controller_observed_at",
        "controller_observation_key",
        "controller_observation_file_sha256",
        "controller_observation_body_sha256",
        "controller_observation",
        "job_binding_key",
        "job_binding_file_sha256",
        "job_binding_body_sha256",
        "job_binding",
        "accepted_at",
        "accepted_body_sha256",
    }
)

_PINNED_HISTORY_SHIM = r"""
import json
import sys
from datetime import datetime, timezone

import sky
from sky.jobs import state

if sky.__version__ != "0.13.0":
    raise RuntimeError("unexpected SkyPilot version")

job_id_text = sys.argv[1]
name = sys.argv[2]
workspace = sys.argv[3]
controller_identity = sys.argv[4]
job_id = int(job_id_text)
if job_id <= 0 or str(job_id) != job_id_text:
    raise RuntimeError("bound job ID is invalid")

def enum_name(value):
    if value is None:
        return None
    return str(getattr(value, "value", getattr(value, "name", value))).upper()

def iso_time(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        parsed = datetime.fromtimestamp(value, timezone.utc)
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

matches, total = state.get_managed_jobs_with_filters(
    job_ids=[job_id],
    workspace_match=workspace,
)
exact = [
    row
    for row in matches
    if row.get("job_id") == job_id
    and row.get("job_name") == name
    and row.get("workspace") == workspace
]
if total != 1 or len(exact) != 1:
    raise RuntimeError("exact controller row is missing or ambiguous")
row = exact[0]
result = {
    "schema_version": 1,
    "record_type": "glm52_sky_controller_history_v1",
    "sky_job_name": name,
    "workspace": workspace,
    "rows": [
        {
            "sky_job_id": job_id,
            "sky_job_name": name,
            "workspace": workspace,
            "controller_submitted_at": iso_time(row.get("submitted_at")),
            "controller_status": enum_name(row.get("status")),
            "controller_identity": controller_identity,
            "schedule_state": enum_name(row.get("schedule_state")),
            "start_at": iso_time(row.get("start_at")),
            "worker_cluster_name": row.get("current_cluster_name"),
            "recovery_count": row.get("recovery_count"),
        }
    ],
    "observed_at": datetime.now(timezone.utc)
    .replace(microsecond=0)
    .isoformat()
    .replace("+00:00", "Z"),
}
print(
    "GLM52_SUBMISSION_HISTORY="
    + json.dumps(result, sort_keys=True, separators=(",", ":"))
)
""".strip()
_PINNED_HISTORY_SHIM_SHA256 = (
    "7f94e25a29e917e3758a878198d97120aaaaf6583a549818167a2b819f21f1c9"
)


class WorkerStartCoordinatorError(ValueError):
    """Coordinator input or injected authority is malformed or ambiguous."""


@dataclass(frozen=True)
class WorkerStartCoordinatorServices:
    sts: object
    s3: object
    ec2: object
    ssm: object
    clock: Callable[[], datetime]
    sleep: Callable[[float], None]


@dataclass(frozen=True)
class WorkerStartCoordinatorRequest:
    account_id: str
    region: str
    bucket: str
    descriptor_key: str
    descriptor_file_sha256: str
    intent_key: str
    intent_file_sha256: str
    intent_body_sha256: str
    worker_latch_key: str
    worker_latch_version_id: str


@dataclass(frozen=True)
class WorkerStartCoordinatorOutcome:
    status: Literal[
        "accepted-initial",
        "accepted-managed-recovery",
        "idempotent-complete",
        "waiting-worker-authority",
    ]
    decision_action: str
    reason: str
    run_id: str
    instance_id: str
    worker_latch_key: str
    worker_controller_observation_key: str | None
    worker_acceptance_key: str | None
    worker_acceptance_body_sha256: str | None
    prior_worker_acceptance_count: int
    published_observation: bool
    published_acceptance: bool


@dataclass(frozen=True)
class _Artifact:
    bucket: str
    key: str
    raw: bytes
    value: dict[str, object]
    file_sha256: str
    body_sha256: str
    version_id: str
    etag: str
    last_modified: str
    content_length: int
    checksum_sha256: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class _StaticAuthority:
    descriptor: _Artifact
    intent: _Artifact
    acquisition: _Artifact
    baseline: _Artifact
    submission_accepted: _Artifact
    binding: _Artifact
    submission_observation: _Artifact
    latch: _Artifact
    dynamic_prefix: str
    acceptance_prefix: str


@dataclass(frozen=True)
class _FreshReplay:
    outcome: WorkerStartCoordinatorOutcome
    acceptance: _Artifact
    observation: _Artifact
    latch: _Artifact


def _canonical_time(value: object, *, label: str) -> tuple[str, datetime]:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, OverflowError) as error:
            raise WorkerStartCoordinatorError(f"{label} is invalid") from error
    else:
        raise WorkerStartCoordinatorError(f"{label} is invalid")
    if parsed.tzinfo is None:
        raise WorkerStartCoordinatorError(f"{label} is not timezone-aware")
    try:
        canonical = parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")
    except (ValueError, OverflowError) as error:
        raise WorkerStartCoordinatorError(f"{label} is invalid") from error
    if isinstance(value, str) and value != canonical:
        raise WorkerStartCoordinatorError(f"{label} is not canonical UTC")
    return canonical, datetime.fromisoformat(canonical.replace("Z", "+00:00"))


def _canonical_json(value: object, *, newline: bool) -> bytes:
    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as error:
        raise WorkerStartCoordinatorError("JSON authority is not canonical") from error
    return raw + (b"\n" if newline else b"")


def _decode_json(raw: bytes) -> dict[str, object]:
    if not raw:
        raise WorkerStartCoordinatorError("immutable JSON object is empty")

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite value {value}")

    try:
        value = json.loads(raw.decode("ascii"), parse_constant=reject_constant)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        RecursionError,
    ) as error:
        raise WorkerStartCoordinatorError(
            "immutable JSON object is malformed"
        ) from error
    if type(value) is not dict:
        raise WorkerStartCoordinatorError("immutable JSON authority is not an object")
    return dict(value)


def _body_digest(value: dict[str, object]) -> str:
    digest_fields = {
        "glm52_sky_campaign_descriptor_v2": "descriptor_body_sha256",
        "glm52_sky_submission_intent_v2": "intent_body_sha256",
        "glm52_sky_submission_acquired_v1": "acquisition_body_sha256",
        "glm52_controller_baseline_v1": "baseline_body_sha256",
        "glm52_sky_submission_accepted_v2": "accepted_body_sha256",
        "glm52_sky_must_start_job_binding_v2": "job_binding_body_sha256",
        "glm52_sky_must_start_controller_observation_v1": ("observation_body_sha256"),
        "glm52_sky_worker_start_latch_v2": "worker_latch_body_sha256",
        "glm52_sky_worker_start_accepted_v2": ("worker_acceptance_body_sha256"),
    }
    digest_field = digest_fields.get(str(value.get("record_type")))
    if digest_field is None or not isinstance(value.get(digest_field), str):
        raise WorkerStartCoordinatorError("immutable JSON self-hash field is ambiguous")
    digest = str(value[digest_field])
    if _SHA256.fullmatch(digest) is None:
        raise WorkerStartCoordinatorError("immutable JSON body SHA-256 is invalid")
    body = dict(value)
    body.pop(digest_field, None)
    if hashlib.sha256(_canonical_json(body, newline=False)).hexdigest() != digest:
        raise WorkerStartCoordinatorError("immutable JSON body SHA-256 mismatches")
    return digest


def _response_status(response: object, *, label: str) -> int:
    if not isinstance(response, dict):
        raise WorkerStartCoordinatorError(f"{label} response is malformed")
    metadata = response.get("ResponseMetadata")
    if (
        not isinstance(metadata, dict)
        or type(metadata.get("HTTPStatusCode")) is not int
        or metadata["HTTPStatusCode"] != 200
    ):
        raise WorkerStartCoordinatorError(f"{label} did not return HTTP 200")
    return 200


def _transport_fields(
    response: object,
    *,
    label: str,
) -> tuple[str, str, str, int, str, str, str, dict[str, str]]:
    _response_status(response, label=label)
    assert isinstance(response, dict)
    version = response.get("VersionId")
    etag = response.get("ETag")
    length = response.get("ContentLength")
    checksum = response.get("ChecksumSHA256")
    checksum_type = response.get("ChecksumType")
    content_type = response.get("ContentType")
    metadata = response.get("Metadata")
    delete_marker = response.get("DeleteMarker")
    if (
        (delete_marker is not None and delete_marker is not False)
        or not isinstance(version, str)
        or _VERSION_ID.fullmatch(version) is None
        or not isinstance(etag, str)
        or _ETAG.fullmatch(etag) is None
        or type(length) is not int
        or length <= 0
        or not isinstance(checksum, str)
        or checksum_type != "FULL_OBJECT"
        or content_type != "application/json"
        or type(metadata) is not dict
        or any(
            not isinstance(name, str) or not isinstance(item, str)
            for name, item in metadata.items()
        )
    ):
        raise WorkerStartCoordinatorError(f"{label} transport is malformed")
    last_modified, _ = _canonical_time(
        response.get("LastModified"),
        label=f"{label} LastModified",
    )
    return (
        version,
        etag,
        last_modified,
        length,
        checksum,
        str(checksum_type),
        str(content_type),
        dict(metadata),
    )


def _call(client: object, operation: str, **kwargs: object) -> object:
    try:
        method = getattr(client, operation)
        return method(**kwargs)
    except WorkerStartCoordinatorError:
        raise
    except Exception as error:
        raise WorkerStartCoordinatorError(
            f"{operation} failed or returned ambiguous state"
        ) from error


def _resolve_version(
    s3: object,
    *,
    bucket: str,
    key: str,
) -> str:
    response = _call(
        s3,
        "head_object",
        Bucket=bucket,
        Key=key,
        ChecksumMode="ENABLED",
        ExpectedBucketOwner=APPROVED_ACCOUNT_ID,
    )
    fields_value = _transport_fields(response, label=f"HEAD s3://{bucket}/{key}")
    return fields_value[0]


def _read_artifact(
    s3: object,
    *,
    bucket: str,
    key: str,
    run_id: str,
    newline: bool,
    version_id: str | None = None,
    expected_file_sha256: str | None = None,
    expected_body_sha256: str | None = None,
) -> _Artifact:
    exact_version = (
        _resolve_version(s3, bucket=bucket, key=key)
        if version_id is None
        else version_id
    )
    if (
        not isinstance(exact_version, str)
        or _VERSION_ID.fullmatch(exact_version) is None
    ):
        raise WorkerStartCoordinatorError("pinned S3 VersionId is invalid")
    try:
        get_response = _call(
            s3,
            "get_object",
            Bucket=bucket,
            Key=key,
            VersionId=exact_version,
            ChecksumMode="ENABLED",
            ExpectedBucketOwner=APPROVED_ACCOUNT_ID,
        )
        if not isinstance(get_response, dict):
            raise WorkerStartCoordinatorError("GET response is malformed")
        stream = get_response.get("Body")
        if (
            stream is None
            or not callable(getattr(stream, "read", None))
            or not callable(getattr(stream, "close", None))
        ):
            raise WorkerStartCoordinatorError("GET body stream is malformed")
        read_error: Exception | None = None
        raw = b""
        try:
            raw_value = stream.read()
            if not isinstance(raw_value, bytes):
                raise WorkerStartCoordinatorError(
                    "GET body stream did not return bytes"
                )
            raw = raw_value
        except Exception as error:
            read_error = error
        try:
            stream.close()
        except Exception as error:
            if read_error is None:
                read_error = error
        if read_error is not None:
            if isinstance(read_error, WorkerStartCoordinatorError):
                raise read_error
            raise WorkerStartCoordinatorError(
                "GET body read or close failed"
            ) from read_error
        get_fields = _transport_fields(
            get_response,
            label=f"GET s3://{bucket}/{key}",
        )
        head_response = _call(
            s3,
            "head_object",
            Bucket=bucket,
            Key=key,
            VersionId=exact_version,
            ChecksumMode="ENABLED",
            ExpectedBucketOwner=APPROVED_ACCOUNT_ID,
        )
        head_fields = _transport_fields(
            head_response,
            label=f"versioned HEAD s3://{bucket}/{key}",
        )
    except WorkerStartCoordinatorError:
        raise
    if get_fields != head_fields:
        raise WorkerStartCoordinatorError("GET and versioned HEAD transport disagree")
    (
        returned_version,
        etag,
        last_modified,
        content_length,
        checksum,
        _,
        _,
        metadata,
    ) = get_fields
    if returned_version != exact_version or content_length != len(raw):
        raise WorkerStartCoordinatorError("immutable object version or length drifted")
    expected_checksum = base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
    if checksum != expected_checksum:
        raise WorkerStartCoordinatorError("FULL_OBJECT SHA-256 checksum mismatches")
    value = _decode_json(raw)
    if raw != _canonical_json(value, newline=newline):
        raise WorkerStartCoordinatorError("immutable JSON file is noncanonical")
    file_sha256 = hashlib.sha256(raw).hexdigest()
    body_sha256 = _body_digest(value)
    if expected_file_sha256 is not None and file_sha256 != expected_file_sha256:
        raise WorkerStartCoordinatorError("immutable JSON file SHA-256 drifted")
    if expected_body_sha256 is not None and body_sha256 != expected_body_sha256:
        raise WorkerStartCoordinatorError("immutable JSON body SHA-256 drifted")
    if metadata != {
        "glm52-run-id": run_id,
        "glm52-body-sha256": body_sha256,
    }:
        raise WorkerStartCoordinatorError("immutable JSON S3 metadata drifted")
    return _Artifact(
        bucket=bucket,
        key=key,
        raw=raw,
        value=value,
        file_sha256=file_sha256,
        body_sha256=body_sha256,
        version_id=returned_version,
        etag=etag,
        last_modified=last_modified,
        content_length=content_length,
        checksum_sha256=checksum,
        metadata=metadata,
    )


def _list_keys(
    s3: object,
    *,
    bucket: str,
    prefix: str,
) -> tuple[str, ...]:
    keys: list[str] = []
    seen_tokens: set[str] = set()
    token: str | None = None
    while True:
        kwargs: dict[str, object] = {
            "Bucket": bucket,
            "Prefix": prefix,
            "ExpectedBucketOwner": APPROVED_ACCOUNT_ID,
        }
        if token is not None:
            kwargs["ContinuationToken"] = token
        response = _call(s3, "list_objects_v2", **kwargs)
        _response_status(response, label=f"LIST s3://{bucket}/{prefix}")
        assert isinstance(response, dict)
        contents = response.get("Contents", [])
        if type(contents) is not list:
            raise WorkerStartCoordinatorError("S3 listing contents are malformed")
        page_keys: list[str] = []
        for item in contents:
            if (
                type(item) is not dict
                or not isinstance(item.get("Key"), str)
                or not str(item["Key"]).startswith(prefix)
                or type(item.get("Size")) is not int
                or item["Size"] <= 0
            ):
                raise WorkerStartCoordinatorError("S3 listing member is malformed")
            page_keys.append(str(item["Key"]))
        if page_keys != sorted(page_keys) or len(page_keys) != len(set(page_keys)):
            raise WorkerStartCoordinatorError("S3 listing page is not canonical")
        if keys and page_keys and page_keys[0] <= keys[-1]:
            raise WorkerStartCoordinatorError("S3 listing pages overlap or reorder")
        keys.extend(page_keys)
        truncated = response.get("IsTruncated")
        if type(truncated) is not bool:
            raise WorkerStartCoordinatorError(
                "S3 listing truncation state is malformed"
            )
        if not truncated:
            if response.get("NextContinuationToken") is not None:
                raise WorkerStartCoordinatorError("terminal S3 page contains a token")
            break
        next_token = response.get("NextContinuationToken")
        if (
            not isinstance(next_token, str)
            or not next_token
            or next_token in seen_tokens
        ):
            raise WorkerStartCoordinatorError("S3 listing token is missing or repeated")
        seen_tokens.add(next_token)
        token = next_token
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise WorkerStartCoordinatorError("S3 listing is ambiguous")
    return tuple(keys)


def _audit_no_legacy(
    s3: object,
    *,
    bucket: str,
    dynamic_prefix: str,
) -> None:
    forbidden = {
        "JOB_BINDING.json",
        "TIMELY_START_LATCH.json",
        "TIMELY_START_ACCEPTED.json",
    }
    keys = _list_keys(s3, bucket=bucket, prefix=dynamic_prefix)
    if any(key.rsplit("/", 1)[-1] in forbidden for key in keys):
        raise WorkerStartCoordinatorError("legacy must-start authority is present")


def _required_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise WorkerStartCoordinatorError(f"{label} is invalid")
    return value


def _validate_static_submission_ancestry(
    *,
    descriptor: _Artifact,
    intent: _Artifact,
    acquisition: _Artifact,
    baseline: _Artifact,
    submission_accepted: _Artifact,
    binding: _Artifact,
    submission_observation: _Artifact,
) -> None:
    accepted = submission_accepted.value
    observation = submission_observation.value
    if (
        set(accepted) != _SUBMISSION_ACCEPTED_FIELDS
        or accepted.get("schema_version") != 2
        or accepted.get("record_type") != "glm52_sky_submission_accepted_v2"
    ):
        raise WorkerStartCoordinatorError("submission accepted schema is invalid")
    controller_fields = (
        "workspace",
        "controller_instance_id",
        "controller_instance_type",
        "controller_profile_arn",
        "controller_cluster_name",
    )
    if any(
        observation.get(field) != baseline.value.get(field)
        for field in controller_fields
    ):
        raise WorkerStartCoordinatorError(
            "submission observation controller coordinates drifted"
        )
    baseline_history = baseline.value.get("exact_name_history")
    if type(baseline_history) is not list:
        raise WorkerStartCoordinatorError("controller baseline history is invalid")
    current_history = [
        *baseline_history,
        {
            "sky_job_id": observation.get("target_job_id"),
            "sky_job_name": observation.get("sky_job_name"),
            "workspace": observation.get("workspace"),
            "controller_submitted_at": observation.get("submitted_at"),
            "controller_status": observation.get("status"),
            "controller_identity": descriptor.value.get("controller_identity"),
        },
    ]
    try:
        validated_binding = validate_dynamic_v2_job_binding(
            binding.value,
            intent=intent.value,
            controller_baseline=baseline.value,
            acquisition=acquisition.value,
            descriptor_controller_identity=str(
                descriptor.value.get("controller_identity")
            ),
            current_exact_name_history=current_history,
            now=accepted.get("accepted_at"),
        )
    except (
        DynamicJobBindingValidationError,
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise WorkerStartCoordinatorError(
            "dynamic-v2 submission ancestry is invalid"
        ) from error
    if validated_binding != binding.value:
        raise WorkerStartCoordinatorError(
            "dynamic-v2 binding validation changed authority"
        )
    _, accepted_time = _canonical_time(
        accepted.get("accepted_at"),
        label="submission accepted_at",
    )
    _, bound_time = _canonical_time(
        binding.value.get("bound_at"),
        label="dynamic-v2 binding bound_at",
    )
    _, observation_time = _canonical_time(
        observation.get("observed_at"),
        label="submission observation observed_at",
    )
    if bound_time > observation_time or accepted_time != max(
        bound_time,
        observation_time,
    ):
        raise WorkerStartCoordinatorError(
            "submission accepted time is not deterministic"
        )
    expected_accepted = {
        "intent_body_sha256": intent.body_sha256,
        "controller_observation_key": submission_observation.key,
        "controller_observation_file_sha256": (submission_observation.file_sha256),
        "controller_observation_body_sha256": (submission_observation.body_sha256),
        "job_binding_key": binding.key,
        "job_binding_file_sha256": binding.file_sha256,
        "job_binding_body_sha256": binding.body_sha256,
        "sky_job_id": binding.value.get("sky_job_id"),
        "sky_job_name": intent.value.get("sky_job_name"),
        "workspace": "default",
        "controller_submitted_at": observation.get("submitted_at"),
        "controller_status": observation.get("status"),
        "controller_identity": descriptor.value.get("controller_identity"),
        "controller_instance_id": observation.get("controller_instance_id"),
        "controller_instance_type": observation.get("controller_instance_type"),
        "controller_profile_arn": observation.get("controller_profile_arn"),
        "controller_cluster_name": observation.get("controller_cluster_name"),
        "controller_observed_at": observation.get("observed_at"),
    }
    if any(accepted.get(name) != value for name, value in expected_accepted.items()):
        raise WorkerStartCoordinatorError("submission accepted authority is foreign")


def _load_static_authority(
    services: WorkerStartCoordinatorServices,
    request: WorkerStartCoordinatorRequest,
    *,
    run_id: str,
) -> _StaticAuthority:
    s3 = services.s3
    descriptor = _read_artifact(
        s3,
        bucket=request.bucket,
        key=request.descriptor_key,
        run_id=run_id,
        newline=True,
        expected_file_sha256=request.descriptor_file_sha256,
    )
    try:
        descriptor_value = validate_sky_campaign_descriptor(descriptor.value)
    except (SkyCampaignValidationError, TypeError, ValueError) as error:
        raise WorkerStartCoordinatorError("campaign descriptor is invalid") from error
    if (
        descriptor_value.get("campaign_descriptor_key") != request.descriptor_key
        or descriptor_value.get("bucket") != request.bucket
        or descriptor_value.get("jobs_bucket") != request.bucket
        or descriptor_value.get("run_id") != run_id
        or descriptor_value.get("worker_identity") != _WORKER_ROLE_ARN
    ):
        raise WorkerStartCoordinatorError("campaign descriptor authority drifted")
    intent = _read_artifact(
        s3,
        bucket=request.bucket,
        key=request.intent_key,
        run_id=run_id,
        newline=True,
        expected_file_sha256=request.intent_file_sha256,
        expected_body_sha256=request.intent_body_sha256,
    )
    dynamic_prefix = (
        f"campaigns/{run_id}/monitor/must-start/qualification/"
        f"{request.intent_body_sha256}/"
    )
    try:
        binding_key = dynamic_v2_job_binding_s3_key(intent=intent.value)
    except (DynamicJobBindingValidationError, TypeError, ValueError) as error:
        raise WorkerStartCoordinatorError("submission intent is invalid") from error
    if (
        intent.value.get("descriptor_key") != descriptor.key
        or intent.value.get("descriptor_file_sha256") != descriptor.file_sha256
        or intent.value.get("descriptor_body_sha256") != descriptor.body_sha256
        or intent.value.get("managed_mode") != "qualification"
        or intent.value.get("run_id") != run_id
        or binding_key != f"{dynamic_prefix}DYNAMIC_JOB_BINDING.json"
    ):
        raise WorkerStartCoordinatorError("intent and descriptor ancestry drifted")
    acquisition_key = (
        f"campaigns/{run_id}/submissions/qualification/acquisitions/"
        f"{descriptor.file_sha256}/SUBMISSION_ACQUIRED.json"
    )
    acquisition = _read_artifact(
        s3,
        bucket=request.bucket,
        key=acquisition_key,
        run_id=run_id,
        newline=True,
    )
    baseline_body_sha256 = _required_sha256(
        acquisition.value.get("controller_baseline_body_sha256"),
        label="controller baseline body SHA-256 pointer",
    )
    baseline_file_sha256 = _required_sha256(
        acquisition.value.get("controller_baseline_file_sha256"),
        label="controller baseline file SHA-256 pointer",
    )
    baseline_key = acquisition.value.get("controller_baseline_key")
    expected_baseline_key = (
        f"campaigns/{run_id}/qualification/controller-baselines/"
        f"{baseline_body_sha256}/CONTROLLER_BASELINE.json"
    )
    if baseline_key != expected_baseline_key:
        raise WorkerStartCoordinatorError("controller baseline key is invalid")
    baseline = _read_artifact(
        s3,
        bucket=request.bucket,
        key=baseline_key,
        run_id=run_id,
        newline=True,
        expected_file_sha256=baseline_file_sha256,
        expected_body_sha256=baseline_body_sha256,
    )
    accepted_prefix = f"campaigns/{run_id}/submissions/qualification/accepted/"
    accepted_keys = _list_keys(
        s3,
        bucket=request.bucket,
        prefix=accepted_prefix,
    )
    accepted_pattern = re.compile(
        rf"^{re.escape(accepted_prefix)}([0-9a-f]{{64}})/"
        r"SKYPILOT_SUBMISSION_ACCEPTED\.json$"
    )
    if len(accepted_keys) != 1:
        raise WorkerStartCoordinatorError(
            "submission accepted-v2 singleton is missing or ambiguous"
        )
    accepted_match = accepted_pattern.fullmatch(accepted_keys[0])
    if accepted_match is None:
        raise WorkerStartCoordinatorError("submission accepted-v2 key is invalid")
    submission_accepted = _read_artifact(
        s3,
        bucket=request.bucket,
        key=accepted_keys[0],
        run_id=run_id,
        newline=True,
        expected_body_sha256=accepted_match.group(1),
    )
    job_binding_key = submission_accepted.value.get("job_binding_key")
    if job_binding_key != binding_key:
        raise WorkerStartCoordinatorError("dynamic-v2 binding pointer drifted")
    binding_file_sha256 = _required_sha256(
        submission_accepted.value.get("job_binding_file_sha256"),
        label="dynamic-v2 binding file SHA-256 pointer",
    )
    binding_body_sha256 = _required_sha256(
        submission_accepted.value.get("job_binding_body_sha256"),
        label="dynamic-v2 binding body SHA-256 pointer",
    )
    binding = _read_artifact(
        s3,
        bucket=request.bucket,
        key=binding_key,
        run_id=run_id,
        newline=False,
        expected_file_sha256=binding_file_sha256,
        expected_body_sha256=binding_body_sha256,
    )
    submission_observation_key = submission_accepted.value.get(
        "controller_observation_key"
    )
    submission_observation_file_sha256 = _required_sha256(
        submission_accepted.value.get("controller_observation_file_sha256"),
        label="submission observation file SHA-256 pointer",
    )
    submission_observation_body_sha256 = _required_sha256(
        submission_accepted.value.get("controller_observation_body_sha256"),
        label="submission observation body SHA-256 pointer",
    )
    expected_submission_observation_key = (
        f"{dynamic_prefix}observations/{submission_observation_body_sha256}.json"
    )
    if submission_observation_key != expected_submission_observation_key:
        raise WorkerStartCoordinatorError(
            "submission controller observation pointer is invalid"
        )
    submission_observation = _read_artifact(
        s3,
        bucket=request.bucket,
        key=submission_observation_key,
        run_id=run_id,
        newline=False,
        expected_file_sha256=submission_observation_file_sha256,
        expected_body_sha256=submission_observation_body_sha256,
    )
    try:
        validate_must_start_controller_observation(submission_observation.value)
    except (MustStartValidationError, TypeError, ValueError) as error:
        raise WorkerStartCoordinatorError(
            "submission controller observation is invalid"
        ) from error
    if (
        submission_accepted.value.get("job_binding") != binding.value
        or submission_accepted.value.get("controller_observation")
        != submission_observation.value
    ):
        raise WorkerStartCoordinatorError(
            "submission accepted embedded ancestry drifted"
        )
    _validate_static_submission_ancestry(
        descriptor=descriptor,
        intent=intent,
        acquisition=acquisition,
        baseline=baseline,
        submission_accepted=submission_accepted,
        binding=binding,
        submission_observation=submission_observation,
    )
    latch = _read_artifact(
        s3,
        bucket=request.bucket,
        key=request.worker_latch_key,
        run_id=run_id,
        newline=False,
        version_id=request.worker_latch_version_id,
    )
    try:
        latch_value = validate_worker_start_latch_v2(
            latch.value,
            descriptor=descriptor.value,
            intent=intent.value,
        )
    except (WorkerStartValidationError, TypeError, ValueError) as error:
        raise WorkerStartCoordinatorError("worker latch is invalid") from error
    if latch_value.get("worker_latch_body_sha256") != request.worker_latch_key.rsplit(
        "/", 1
    )[-1].removesuffix(".json"):
        raise WorkerStartCoordinatorError("worker latch key/body identity drifted")
    _audit_no_legacy(
        s3,
        bucket=request.bucket,
        dynamic_prefix=dynamic_prefix,
    )
    return _StaticAuthority(
        descriptor=descriptor,
        intent=intent,
        acquisition=acquisition,
        baseline=baseline,
        submission_accepted=submission_accepted,
        binding=binding,
        submission_observation=submission_observation,
        latch=latch,
        dynamic_prefix=dynamic_prefix,
        acceptance_prefix=f"{dynamic_prefix}worker-acceptances/",
    )


def _required_campaign_tags(run_id: str) -> dict[str, str]:
    return {
        "project": "keep-glm52",
        "owner": "jack.mazac",
        "model": "glm-5.2",
        "campaign-run-id": run_id,
        "cost-allocation": "glm52-sky-campaign",
    }


def _validate_worker_campaign_tags(
    value: object,
    *,
    run_id: str,
) -> dict[str, str]:
    expected = _required_campaign_tags(run_id)
    if type(value) is not dict or value != expected:
        raise WorkerStartCoordinatorError(
            "worker campaign tags are not the exact projected authority"
        )
    return dict(expected)


def _validate_worker_observation(
    value: dict[str, object],
) -> dict[str, object]:
    try:
        validated = validate_must_start_controller_observation(value)
    except (MustStartValidationError, TypeError, ValueError) as error:
        raise WorkerStartCoordinatorError(
            "worker controller observation failed exact validation"
        ) from error
    if (
        validated != value
        or validated.get("schedule_state") not in _WORKER_SCHEDULE_STATES
    ):
        raise WorkerStartCoordinatorError(
            "worker controller observation authority is foreign"
        )
    return validated


def _parse_tags(value: object, *, label: str) -> dict[str, str]:
    if type(value) is not list:
        raise WorkerStartCoordinatorError(f"{label} tags are malformed")
    result: dict[str, str] = {}
    for item in value:
        if (
            type(item) is not dict
            or not isinstance(item.get("Key"), str)
            or not isinstance(item.get("Value"), str)
            or not item["Key"]
            or item["Key"] in result
        ):
            raise WorkerStartCoordinatorError(f"{label} tags are malformed")
        result[str(item["Key"])] = str(item["Value"])
    return result


def _exact_instance(
    ec2: object,
    *,
    instance_id: str,
    label: str,
) -> dict[str, object]:
    response = _call(
        ec2,
        "describe_instances",
        InstanceIds=[instance_id],
    )
    _response_status(response, label=f"EC2 exact {label}")
    assert isinstance(response, dict)
    if response.get("NextToken") is not None:
        raise WorkerStartCoordinatorError(f"EC2 exact {label} returned pagination")
    reservations = response.get("Reservations")
    if type(reservations) is not list or len(reservations) != 1:
        raise WorkerStartCoordinatorError(f"EC2 exact {label} is ambiguous")
    reservation = reservations[0]
    if (
        type(reservation) is not dict
        or reservation.get("OwnerId") != APPROVED_ACCOUNT_ID
        or type(reservation.get("Instances")) is not list
        or len(reservation["Instances"]) != 1
        or type(reservation["Instances"][0]) is not dict
    ):
        raise WorkerStartCoordinatorError(f"EC2 exact {label} is malformed")
    instance = dict(reservation["Instances"][0])
    if instance.get("InstanceId") != instance_id:
        raise WorkerStartCoordinatorError(f"EC2 exact {label} ID drifted")
    return instance


def _filtered_instances(
    ec2: object,
    *,
    filters: list[dict[str, object]],
    label: str,
) -> tuple[dict[str, object], ...]:
    result: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    seen_tokens: set[str] = set()
    token: str | None = None
    while True:
        kwargs: dict[str, object] = {"Filters": filters}
        if token is not None:
            kwargs["NextToken"] = token
        response = _call(ec2, "describe_instances", **kwargs)
        _response_status(response, label=f"EC2 filtered {label}")
        assert isinstance(response, dict)
        reservations = response.get("Reservations")
        if type(reservations) is not list:
            raise WorkerStartCoordinatorError(f"EC2 filtered {label} is malformed")
        for reservation in reservations:
            if (
                type(reservation) is not dict
                or reservation.get("OwnerId") != APPROVED_ACCOUNT_ID
                or type(reservation.get("Instances")) is not list
            ):
                raise WorkerStartCoordinatorError(
                    f"EC2 filtered {label} owner or page is malformed"
                )
            for item in reservation["Instances"]:
                if type(item) is not dict or not isinstance(
                    item.get("InstanceId"),
                    str,
                ):
                    raise WorkerStartCoordinatorError(
                        f"EC2 filtered {label} member is malformed"
                    )
                instance_id = str(item["InstanceId"])
                if (
                    _INSTANCE_ID.fullmatch(instance_id) is None
                    or instance_id in seen_ids
                ):
                    raise WorkerStartCoordinatorError(
                        f"EC2 filtered {label} contains duplicate or invalid IDs"
                    )
                seen_ids.add(instance_id)
                result.append(dict(item))
        next_token = response.get("NextToken")
        if next_token is None:
            break
        if (
            not isinstance(next_token, str)
            or not next_token
            or next_token in seen_tokens
        ):
            raise WorkerStartCoordinatorError(
                f"EC2 filtered {label} token is malformed"
            )
        seen_tokens.add(next_token)
        token = next_token
    return tuple(result)


def _instance_profile(instance: dict[str, object], *, label: str) -> str:
    profile = instance.get("IamInstanceProfile")
    if (
        type(profile) is not dict
        or set(profile) not in ({"Arn"}, {"Arn", "Id"})
        or not isinstance(profile.get("Arn"), str)
        or _PROFILE_ARN.fullmatch(str(profile["Arn"])) is None
        or (
            "Id" in profile
            and (not isinstance(profile.get("Id"), str) or not profile["Id"])
        )
    ):
        raise WorkerStartCoordinatorError(f"{label} instance profile is invalid")
    return str(profile["Arn"])


def _instance_state(instance: dict[str, object], *, label: str) -> str:
    state = instance.get("State")
    if (
        type(state) is not dict
        or set(state) not in ({"Name"}, {"Code", "Name"})
        or not isinstance(state.get("Name"), str)
    ):
        raise WorkerStartCoordinatorError(f"{label} state is invalid")
    name = str(state["Name"])
    expected_codes = {
        "pending": 0,
        "running": 16,
        "shutting-down": 32,
        "terminated": 48,
        "stopping": 64,
        "stopped": 80,
    }
    if "Code" in state and (
        type(state["Code"]) is not int or expected_codes.get(name) != state["Code"]
    ):
        raise WorkerStartCoordinatorError(f"{label} state code is invalid")
    return name


def _availability_zone(instance: dict[str, object], *, label: str) -> str:
    placement = instance.get("Placement")
    zone = placement.get("AvailabilityZone") if isinstance(placement, dict) else None
    if not isinstance(zone, str) or re.fullmatch(r"us-west-2[a-z]", zone) is None:
        raise WorkerStartCoordinatorError(f"{label} availability zone is invalid")
    return zone


def _validate_controller_instance(
    instance: dict[str, object],
    *,
    static: _StaticAuthority,
    run_id: str,
) -> None:
    baseline = static.baseline.value
    expected = {
        "InstanceId": baseline.get("controller_instance_id"),
        "InstanceType": baseline.get("controller_instance_type"),
    }
    if any(instance.get(name) != value for name, value in expected.items()):
        raise WorkerStartCoordinatorError("controller EC2 identity drifted")
    if instance.get("InstanceLifecycle") is not None:
        raise WorkerStartCoordinatorError("controller instance lifecycle is foreign")
    if _instance_state(instance, label="controller") != "running":
        raise WorkerStartCoordinatorError("controller instance is not running")
    _availability_zone(instance, label="controller")
    if _instance_profile(instance, label="controller") != baseline.get(
        "controller_profile_arn"
    ):
        raise WorkerStartCoordinatorError("controller profile drifted")
    tags = _parse_tags(instance.get("Tags"), label="controller")
    if any(
        tags.get(name) != item for name, item in _required_campaign_tags(run_id).items()
    ) or tags.get("ray-cluster-name") != baseline.get("controller_cluster_name"):
        raise WorkerStartCoordinatorError("controller campaign tags drifted")


def _online_ssm_identity(
    ssm: object,
    *,
    controller_id: str,
) -> None:
    token: str | None = None
    seen_tokens: set[str] = set()
    identities: list[dict[str, object]] = []
    while True:
        kwargs: dict[str, object] = {
            "Filters": [
                {
                    "Key": "InstanceIds",
                    "Values": [controller_id],
                }
            ]
        }
        if token is not None:
            kwargs["NextToken"] = token
        response = _call(ssm, "describe_instance_information", **kwargs)
        _response_status(response, label="SSM instance information")
        if (
            not isinstance(response, dict)
            or type(response.get("InstanceInformationList")) is not list
        ):
            raise WorkerStartCoordinatorError("SSM inventory is malformed")
        for item in response["InstanceInformationList"]:
            if type(item) is not dict or not isinstance(
                item.get("InstanceId"),
                str,
            ):
                raise WorkerStartCoordinatorError("SSM inventory member is malformed")
            identities.append(dict(item))
        next_token = response.get("NextToken")
        if next_token is None:
            break
        if (
            not isinstance(next_token, str)
            or not next_token
            or next_token in seen_tokens
        ):
            raise WorkerStartCoordinatorError("SSM inventory token is malformed")
        seen_tokens.add(next_token)
        token = next_token
    exact = [item for item in identities if item.get("InstanceId") == controller_id]
    if (
        len(identities) != 1
        or len(exact) != 1
        or exact[0].get("PingStatus") != "Online"
    ):
        raise WorkerStartCoordinatorError(
            "controller SSM identity is missing, offline, or ambiguous"
        )


def _exception_code(error: BaseException) -> str | None:
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return None
    detail = response.get("Error")
    if not isinstance(detail, dict):
        return None
    code = detail.get("Code")
    return str(code) if isinstance(code, (str, int)) else None


def _put_error_may_have_persisted(error: Exception) -> bool:
    code = _exception_code(error)
    if code is not None:
        return code in {
            "409",
            "412",
            "ConditionalRequestConflict",
            "PreconditionFailed",
        }
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    return type(error) is RuntimeError and error.args == (
        "generic transport ambiguity after persist",
    )


def _is_delete_marker_error(error: BaseException) -> bool:
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return False
    if response.get("DeleteMarker") is True:
        return True
    metadata = response.get("ResponseMetadata")
    headers = metadata.get("HTTPHeaders") if isinstance(metadata, dict) else None
    return (
        isinstance(headers, dict)
        and str(headers.get("x-amz-delete-marker", "")).lower() == "true"
    )


def _history_command(
    *,
    job_id: int,
    job_name: str,
    workspace: str,
    controller_identity: str,
) -> str:
    arguments = [
        "/home/ubuntu/skypilot-runtime/bin/python",
        "-c",
        _PINNED_HISTORY_SHIM,
        str(job_id),
        job_name,
        workspace,
        controller_identity,
    ]
    return " ".join(shlex.quote(item) for item in arguments)


def _read_controller_history(
    services: WorkerStartCoordinatorServices,
    *,
    static: _StaticAuthority,
) -> dict[str, object]:
    baseline = static.baseline.value
    binding = static.binding.value
    controller_id = baseline.get("controller_instance_id")
    if (
        not isinstance(controller_id, str)
        or _INSTANCE_ID.fullmatch(controller_id) is None
    ):
        raise WorkerStartCoordinatorError("controller instance ID is invalid")
    controller = _exact_instance(
        services.ec2,
        instance_id=controller_id,
        label="controller",
    )
    _validate_controller_instance(
        controller,
        static=static,
        run_id=str(static.intent.value["run_id"]),
    )
    _online_ssm_identity(services.ssm, controller_id=controller_id)
    job_id = binding.get("sky_job_id")
    if type(job_id) is not int or job_id <= 0:
        raise WorkerStartCoordinatorError("bound Sky job ID is invalid")
    job_name = static.intent.value.get("sky_job_name")
    controller_identity = static.descriptor.value.get("controller_identity")
    if (
        not isinstance(job_name, str)
        or _SAFE_NAME.fullmatch(job_name) is None
        or not isinstance(controller_identity, str)
    ):
        raise WorkerStartCoordinatorError("controller command authority is invalid")
    command = _history_command(
        job_id=job_id,
        job_name=job_name,
        workspace="default",
        controller_identity=controller_identity,
    )
    send_response = _call(
        services.ssm,
        "send_command",
        InstanceIds=[controller_id],
        DocumentName="AWS-RunShellScript",
        Comment="KEEP GLM52 observe exact dynamic-v2 worker authority",
        Parameters={"commands": [command]},
        TimeoutSeconds=45,
    )
    _response_status(send_response, label="SSM send_command")
    assert isinstance(send_response, dict)
    command_value = send_response.get("Command")
    command_id = (
        command_value.get("CommandId") if isinstance(command_value, dict) else None
    )
    if not isinstance(command_id, str) or not command_id:
        raise WorkerStartCoordinatorError("SSM CommandId is missing")
    invocation: dict[str, object] | None = None
    for _ in range(10):
        try:
            response = services.ssm.get_command_invocation(  # type: ignore[attr-defined]
                CommandId=command_id,
                InstanceId=controller_id,
            )
        except Exception as error:
            if _exception_code(error) == "InvocationDoesNotExist":
                services.sleep(1)
                continue
            raise WorkerStartCoordinatorError(
                "SSM command invocation failed"
            ) from error
        if not isinstance(response, dict):
            raise WorkerStartCoordinatorError("SSM invocation is malformed")
        _response_status(response, label="SSM command invocation")
        status = response.get("Status")
        if status in {"Pending", "InProgress", "Delayed"}:
            services.sleep(1)
            continue
        invocation = dict(response)
        break
    if invocation is None:
        raise WorkerStartCoordinatorError("SSM history observation timed out")
    if (
        invocation.get("CommandId") != command_id
        or invocation.get("InstanceId") != controller_id
        or invocation.get("Status") != "Success"
        or type(invocation.get("ResponseCode")) is not int
        or invocation["ResponseCode"] != 0
        or invocation.get("StandardErrorContent") != ""
        or not isinstance(invocation.get("StandardOutputContent"), str)
    ):
        raise WorkerStartCoordinatorError("SSM history command failed")
    stdout = str(invocation["StandardOutputContent"])
    if stdout.endswith("\n"):
        stdout = stdout[:-1]
    if "\n" in stdout or not stdout.startswith(_CONTROLLER_HISTORY_PREFIX):
        raise WorkerStartCoordinatorError("SSM history output is malformed")
    raw = stdout.removeprefix(_CONTROLLER_HISTORY_PREFIX)
    try:
        raw_bytes = raw.encode("ascii")
    except UnicodeEncodeError as error:
        raise WorkerStartCoordinatorError(
            "SSM history output is not canonical ASCII"
        ) from error
    history = _decode_json(raw_bytes)
    if raw_bytes != _canonical_json(history, newline=False):
        raise WorkerStartCoordinatorError("SSM history output is noncanonical")
    expected_fields = {
        "schema_version",
        "record_type",
        "sky_job_name",
        "workspace",
        "rows",
        "observed_at",
    }
    if (
        set(history) != expected_fields
        or history.get("schema_version") != 1
        or history.get("record_type") != "glm52_sky_controller_history_v1"
        or history.get("sky_job_name") != job_name
        or history.get("workspace") != "default"
        or type(history.get("rows")) is not list
        or len(history["rows"]) != 1
        or type(history["rows"][0]) is not dict
    ):
        raise WorkerStartCoordinatorError("controller history schema is invalid")
    row = dict(history["rows"][0])
    if set(row) != {
        "sky_job_id",
        "sky_job_name",
        "workspace",
        "controller_submitted_at",
        "controller_status",
        "controller_identity",
        "schedule_state",
        "start_at",
        "worker_cluster_name",
        "recovery_count",
    }:
        raise WorkerStartCoordinatorError("controller history row schema is invalid")
    expected = {
        "sky_job_id": job_id,
        "sky_job_name": job_name,
        "workspace": "default",
        "controller_submitted_at": static.submission_observation.value.get(
            "submitted_at"
        ),
        "controller_identity": controller_identity,
    }
    if any(row.get(name) != item for name, item in expected.items()):
        raise WorkerStartCoordinatorError("controller history identity drifted")
    if row.get("controller_status") not in _CONTROLLER_STATUSES:
        raise WorkerStartCoordinatorError("controller status is invalid")
    if row.get("schedule_state") not in _WORKER_SCHEDULE_STATES:
        raise WorkerStartCoordinatorError("controller schedule state is invalid")
    if type(row.get("recovery_count")) is not int or row["recovery_count"] < 0:
        raise WorkerStartCoordinatorError("controller recovery count is invalid")
    _canonical_time(
        row.get("controller_submitted_at"),
        label="controller submitted_at",
    )
    observed_iso, observed = _canonical_time(
        history.get("observed_at"),
        label="controller observed_at",
    )
    _, immutable_observed = _canonical_time(
        static.submission_observation.value.get("observed_at"),
        label="immutable controller observed_at",
    )
    if observed < immutable_observed:
        raise WorkerStartCoordinatorError(
            "controller history predates immutable submission evidence"
        )
    start_at = row.get("start_at")
    if start_at is not None:
        _, started = _canonical_time(start_at, label="controller start_at")
        _, submitted = _canonical_time(
            row["controller_submitted_at"],
            label="controller submitted_at",
        )
        if not submitted <= started <= observed:
            raise WorkerStartCoordinatorError("controller start time is inconsistent")
    cluster = row.get("worker_cluster_name")
    if cluster is not None and (
        not isinstance(cluster, str) or _SAFE_NAME.fullmatch(cluster) is None
    ):
        raise WorkerStartCoordinatorError("controller worker cluster is invalid")
    try:
        observation = build_must_start_controller_observation(
            run_id=str(static.intent.value["run_id"]),
            managed_mode="qualification",
            account_id=APPROVED_ACCOUNT_ID,
            region=APPROVED_REGION,
            bucket=static.descriptor.bucket,
            descriptor_body_sha256=static.descriptor.body_sha256,
            submission_body_sha256=static.intent.body_sha256,
            sky_job_name=job_name,
            must_start_by=str(static.intent.value["must_start_by"]),
            target_job_id=job_id,
            workspace="default",
            controller_instance_id=controller_id,
            controller_instance_type=str(baseline["controller_instance_type"]),
            controller_profile_arn=str(baseline["controller_profile_arn"]),
            controller_cluster_name=str(baseline["controller_cluster_name"]),
            status=str(row["controller_status"]),
            schedule_state=str(row["schedule_state"]),
            submitted_at=str(row["controller_submitted_at"]),
            start_at=start_at if isinstance(start_at, str) else None,
            worker_cluster_name=(cluster if isinstance(cluster, str) else None),
            recovery_count=int(row["recovery_count"]),
            observed_at=observed_iso,
        )
    except (MustStartValidationError, TypeError, ValueError) as error:
        raise WorkerStartCoordinatorError(
            "worker controller observation builder rejected live evidence"
        ) from error
    return observation


def _validate_worker_instance(
    instance: dict[str, object],
    *,
    static: _StaticAuthority,
    instance_id: str,
    expected_cluster: str | None,
    ec2_observed_at: str,
    allowed_states: frozenset[str] = _CURRENT_WORKER_STATES,
) -> tuple[dict[str, object], str]:
    descriptor = static.descriptor.value
    latch = static.latch.value
    if (
        instance.get("InstanceId") != instance_id
        or instance.get("InstanceType") != descriptor.get("instance_type")
        or instance.get("ImageId") != descriptor.get("image_id")
        or instance.get("InstanceLifecycle") is not None
    ):
        raise WorkerStartCoordinatorError("worker EC2 identity is foreign")
    state = _instance_state(instance, label="worker")
    if state not in allowed_states:
        raise WorkerStartCoordinatorError("current worker state is not admissible")
    _availability_zone(instance, label="worker")
    if _instance_profile(instance, label="worker") != _WORKER_PROFILE_ARN:
        raise WorkerStartCoordinatorError("worker instance profile is foreign")
    instance_type = instance.get("InstanceType")
    image_id = instance.get("ImageId")
    if not isinstance(instance_type, str) or not isinstance(image_id, str):
        raise WorkerStartCoordinatorError("worker type or image is malformed")
    if _AMI_ID.fullmatch(image_id) is None:
        raise WorkerStartCoordinatorError("worker image ID is malformed")
    launch_iso, launch = _canonical_time(
        instance.get("LaunchTime"),
        label="worker launch time",
    )
    _, pending = _canonical_time(
        latch.get("ec2_pending_time"),
        label="worker latch pending time",
    )
    _, observed = _canonical_time(
        ec2_observed_at,
        label="worker EC2 observed_at",
    )
    if launch > pending or pending > observed:
        raise WorkerStartCoordinatorError("worker EC2 chronology is invalid")
    tags = _parse_tags(instance.get("Tags"), label="worker")
    required = _required_campaign_tags(str(static.intent.value["run_id"]))
    if any(tags.get(name) != item for name, item in required.items()):
        raise WorkerStartCoordinatorError("worker campaign tags are invalid")
    ray_cluster = tags.get("ray-cluster-name")
    sky_cluster = tags.get("skypilot-cluster-name")
    if (
        not isinstance(ray_cluster, str)
        or _SAFE_NAME.fullmatch(ray_cluster) is None
        or not isinstance(sky_cluster, str)
        or _SAFE_NAME.fullmatch(sky_cluster) is None
        or ray_cluster != sky_cluster
        or (expected_cluster is not None and ray_cluster != expected_cluster)
    ):
        raise WorkerStartCoordinatorError("worker SkyPilot cluster tags drifted")
    snapshot: dict[str, object] = {
        "worker_cluster_name": ray_cluster,
        "instance_id": instance_id,
        "worker_instance_type": instance_type,
        "worker_image_id": image_id,
        "worker_instance_lifecycle": None,
        "worker_instance_state": state,
        "worker_instance_launch_time": launch_iso,
        "worker_ray_cluster_name": ray_cluster,
        "worker_skypilot_cluster_name": sky_cluster,
        "worker_campaign_tags": dict(required),
        "ec2_observed_at": ec2_observed_at,
    }
    return snapshot, ray_cluster


def _ec2_filters(
    *,
    run_id: str,
    cluster: str | None,
    campaign_wide: bool,
) -> list[dict[str, object]]:
    filters: list[dict[str, object]] = [
        {"Name": f"tag:{name}", "Values": [item]}
        for name, item in sorted(_required_campaign_tags(run_id).items())
    ]
    if cluster is not None:
        filters.extend(
            [
                {"Name": "tag:ray-cluster-name", "Values": [cluster]},
                {
                    "Name": "tag:skypilot-cluster-name",
                    "Values": [cluster],
                },
            ]
        )
    if campaign_wide:
        filters.extend(
            [
                {"Name": "instance-type", "Values": ["p5.48xlarge"]},
                {
                    "Name": "instance-state-name",
                    "Values": ["pending", "running", "stopping"],
                },
            ]
        )
    else:
        filters.append(
            {
                "Name": "instance-state-name",
                "Values": ["pending", "running", "stopping"],
            }
        )
    return filters


def _worker_ec2_authority(
    services: WorkerStartCoordinatorServices,
    *,
    static: _StaticAuthority,
    worker_observation: dict[str, object],
    instance_id: str,
) -> tuple[dict[str, object], list[str], str, datetime]:
    exact = _exact_instance(
        services.ec2,
        instance_id=instance_id,
        label="worker",
    )
    exact_tags = _parse_tags(exact.get("Tags"), label="worker")
    controller_cluster = worker_observation.get("worker_cluster_name")
    if controller_cluster is not None and not isinstance(controller_cluster, str):
        raise WorkerStartCoordinatorError("worker controller cluster is malformed")
    cluster = (
        controller_cluster
        if isinstance(controller_cluster, str)
        else exact_tags.get("ray-cluster-name")
    )
    if not isinstance(cluster, str) or _SAFE_NAME.fullmatch(cluster) is None:
        raise WorkerStartCoordinatorError("worker cluster join is unavailable")
    cluster_instances = _filtered_instances(
        services.ec2,
        filters=_ec2_filters(
            run_id=str(static.intent.value["run_id"]),
            cluster=cluster,
            campaign_wide=False,
        ),
        label="worker cluster",
    )
    campaign_instances = _filtered_instances(
        services.ec2,
        filters=_ec2_filters(
            run_id=str(static.intent.value["run_id"]),
            cluster=None,
            campaign_wide=True,
        ),
        label="campaign P5",
    )
    try:
        sampled = services.clock()
    except Exception as error:
        raise WorkerStartCoordinatorError("coordinator clock failed") from error
    clock_iso, clock_time = _canonical_time(sampled, label="coordinator clock")
    if (
        len(cluster_instances) != 1
        or cluster_instances[0].get("InstanceId") != instance_id
    ):
        raise WorkerStartCoordinatorError(
            "worker cluster mapping is missing or ambiguous"
        )
    # Validate all three views independently before projecting the exact-ID view.
    exact_snapshot, exact_cluster = _validate_worker_instance(
        exact,
        static=static,
        instance_id=instance_id,
        expected_cluster=(
            controller_cluster if isinstance(controller_cluster, str) else None
        ),
        ec2_observed_at=clock_iso,
    )
    cluster_snapshot, cluster_name = _validate_worker_instance(
        cluster_instances[0],
        static=static,
        instance_id=instance_id,
        expected_cluster=exact_cluster,
        ec2_observed_at=clock_iso,
    )
    exact_zone = _availability_zone(exact, label="worker exact view")
    cluster_zone = _availability_zone(
        cluster_instances[0],
        label="worker cluster view",
    )
    if (
        cluster_snapshot != exact_snapshot
        or cluster_name != exact_cluster
        or cluster_zone != exact_zone
    ):
        raise WorkerStartCoordinatorError("worker exact and cluster views disagree")
    active_ids: list[str] = []
    campaign_snapshots: dict[str, dict[str, object]] = {}
    campaign_zones: dict[str, str] = {}
    for item in campaign_instances:
        item_id = item.get("InstanceId")
        if not isinstance(item_id, str) or _INSTANCE_ID.fullmatch(item_id) is None:
            raise WorkerStartCoordinatorError("campaign worker ID is invalid")
        campaign_snapshot, campaign_cluster = _validate_worker_instance(
            item,
            static=static,
            instance_id=item_id,
            expected_cluster=exact_cluster,
            ec2_observed_at=clock_iso,
            allowed_states=_ACTIVE_STATES,
        )
        if campaign_cluster != exact_cluster:
            raise WorkerStartCoordinatorError("campaign worker cluster view drifted")
        active_ids.append(item_id)
        campaign_snapshots[item_id] = campaign_snapshot
        campaign_zones[item_id] = _availability_zone(
            item,
            label="campaign worker view",
        )
    if active_ids != sorted(active_ids) or active_ids != [instance_id]:
        raise WorkerStartCoordinatorError(
            "campaign P5 inventory is not the exact latch singleton"
        )
    campaign_snapshot = campaign_snapshots.get(instance_id)
    if (
        campaign_snapshot is None
        or campaign_zones.get(instance_id) != exact_zone
        or any(
            campaign_snapshot.get(name) != value
            for name, value in exact_snapshot.items()
            if name != "worker_instance_state"
        )
    ):
        raise WorkerStartCoordinatorError("worker exact and campaign views disagree")
    return exact_snapshot, list(active_ids), clock_iso, clock_time


def _acceptance_key(
    static: _StaticAuthority,
    *,
    instance_id: str,
    latch_body_sha256: str,
) -> str:
    return (
        f"{static.acceptance_prefix}{instance_id}/{latch_body_sha256}/"
        "WORKER_START_ACCEPTED.json"
    )


def _load_acceptance_artifacts(
    services: WorkerStartCoordinatorServices,
    *,
    static: _StaticAuthority,
    current_key: str,
) -> tuple[list[_Artifact], _Artifact | None]:
    keys = _list_keys(
        services.s3,
        bucket=static.descriptor.bucket,
        prefix=static.acceptance_prefix,
    )
    pattern = re.compile(
        rf"^{re.escape(static.acceptance_prefix)}(i-[0-9a-f]{{17}})/"
        r"([0-9a-f]{64})/WORKER_START_ACCEPTED\.json$"
    )
    artifacts: list[_Artifact] = []
    current: _Artifact | None = None
    for key in keys:
        if pattern.fullmatch(key) is None:
            raise WorkerStartCoordinatorError("worker acceptance key is malformed")
        artifact = _read_artifact(
            services.s3,
            bucket=static.descriptor.bucket,
            key=key,
            run_id=str(static.intent.value["run_id"]),
            newline=False,
        )
        if key == current_key:
            if current is not None:
                raise WorkerStartCoordinatorError(
                    "current worker acceptance is duplicated"
                )
            current = artifact
        else:
            artifacts.append(artifact)
    return artifacts, current


def _order_prior_acceptances(
    artifacts: list[_Artifact],
) -> list[_Artifact]:
    if not artifacts:
        return []
    by_key = {artifact.key: artifact for artifact in artifacts}
    if len(by_key) != len(artifacts):
        raise WorkerStartCoordinatorError("worker acceptance chain has duplicates")
    roots = [
        artifact
        for artifact in artifacts
        if artifact.value.get("prior_worker_acceptance_key") is None
    ]
    if len(roots) != 1:
        raise WorkerStartCoordinatorError("worker acceptance chain has multiple roots")
    successors: dict[str, list[_Artifact]] = {}
    for artifact in artifacts:
        prior_key = artifact.value.get("prior_worker_acceptance_key")
        prior_file = artifact.value.get("prior_worker_acceptance_file_sha256")
        prior_body = artifact.value.get("prior_worker_acceptance_body_sha256")
        if prior_key is None:
            if prior_file is not None or prior_body is not None:
                raise WorkerStartCoordinatorError(
                    "worker acceptance root has partial predecessor pointers"
                )
            continue
        if not isinstance(prior_key, str) or prior_key not in by_key:
            raise WorkerStartCoordinatorError("worker acceptance chain has a gap")
        predecessor = by_key[prior_key]
        if (
            prior_file != predecessor.file_sha256
            or prior_body != predecessor.body_sha256
        ):
            raise WorkerStartCoordinatorError(
                "worker acceptance predecessor identity drifted"
            )
        successors.setdefault(prior_key, []).append(artifact)
    if any(len(items) != 1 for items in successors.values()):
        raise WorkerStartCoordinatorError("worker acceptance chain forks")
    ordered: list[_Artifact] = []
    seen: set[str] = set()
    cursor = roots[0]
    while True:
        if cursor.key in seen:
            raise WorkerStartCoordinatorError("worker acceptance chain cycles")
        seen.add(cursor.key)
        ordered.append(cursor)
        children = successors.get(cursor.key, [])
        if not children:
            break
        cursor = children[0]
    if len(ordered) != len(artifacts):
        raise WorkerStartCoordinatorError(
            "worker acceptance chain has multiple tips or disconnected members"
        )
    instances = [str(item.value.get("instance_id")) for item in ordered]
    if len(instances) != len(set(instances)):
        raise WorkerStartCoordinatorError("worker acceptance chain repeats an instance")
    recovery_counts = [item.value.get("recovery_count") for item in ordered]
    if any(type(item) is not int for item in recovery_counts) or any(
        later <= earlier for earlier, later in zip(recovery_counts, recovery_counts[1:])
    ):
        raise WorkerStartCoordinatorError(
            "worker acceptance recovery count does not increase"
        )
    return ordered


def _prior_evidence(
    services: WorkerStartCoordinatorServices,
    *,
    static: _StaticAuthority,
    ordered: list[_Artifact],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    acceptances: list[dict[str, object]] = []
    latches: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
    for artifact in ordered:
        value = artifact.value
        _validate_worker_campaign_tags(
            value.get("worker_campaign_tags"),
            run_id=str(static.intent.value["run_id"]),
        )
        instance_id = value.get("instance_id")
        latch_body_sha256 = _required_sha256(
            value.get("worker_latch_body_sha256"),
            label="prior latch body SHA-256",
        )
        latch_file_sha256 = _required_sha256(
            value.get("worker_latch_file_sha256"),
            label="prior latch file SHA-256",
        )
        observation_body_sha256 = _required_sha256(
            value.get("worker_controller_observation_body_sha256"),
            label="prior worker observation body SHA-256",
        )
        observation_file_sha256 = _required_sha256(
            value.get("worker_controller_observation_file_sha256"),
            label="prior worker observation file SHA-256",
        )
        latch_key = value.get("worker_latch_key")
        latch_version = value.get("worker_latch_version_id")
        observation_key = value.get("worker_controller_observation_key")
        expected_latch_key = (
            f"{static.dynamic_prefix}worker-latches/{instance_id}/"
            f"{latch_body_sha256}.json"
        )
        expected_observation_key = (
            f"{static.dynamic_prefix}worker-controller-observations/"
            f"{instance_id}/{observation_body_sha256}.json"
        )
        expected_acceptance_key = (
            f"{static.acceptance_prefix}{instance_id}/"
            f"{latch_body_sha256}/WORKER_START_ACCEPTED.json"
        )
        if (
            not isinstance(instance_id, str)
            or _INSTANCE_ID.fullmatch(instance_id) is None
            or artifact.key != expected_acceptance_key
            or latch_key != expected_latch_key
            or not isinstance(latch_version, str)
            or _VERSION_ID.fullmatch(latch_version) is None
            or observation_key != expected_observation_key
        ):
            raise WorkerStartCoordinatorError(
                "prior worker acceptance source pointers are invalid"
            )
        latch = _read_artifact(
            services.s3,
            bucket=static.descriptor.bucket,
            key=latch_key,
            run_id=str(static.intent.value["run_id"]),
            newline=False,
            version_id=latch_version,
            expected_file_sha256=latch_file_sha256,
            expected_body_sha256=latch_body_sha256,
        )
        if latch.etag != value.get(
            "worker_latch_etag"
        ) or latch.last_modified != value.get("worker_latch_last_modified"):
            raise WorkerStartCoordinatorError("prior latch transport identity drifted")
        observation = _read_artifact(
            services.s3,
            bucket=static.descriptor.bucket,
            key=observation_key,
            run_id=str(static.intent.value["run_id"]),
            newline=False,
            expected_file_sha256=observation_file_sha256,
            expected_body_sha256=observation_body_sha256,
        )
        try:
            validated_latch = validate_worker_start_latch_v2(
                latch.value,
                descriptor=static.descriptor.value,
                intent=static.intent.value,
            )
        except (
            WorkerStartValidationError,
            TypeError,
            ValueError,
        ) as error:
            raise WorkerStartCoordinatorError(
                "prior worker source authority is invalid"
            ) from error
        validated_observation = _validate_worker_observation(observation.value)
        if (
            validated_latch != latch.value
            or validated_observation != observation.value
            or worker_start_latch_v2_s3_key(worker_latch=validated_latch) != latch.key
        ):
            raise WorkerStartCoordinatorError(
                "prior worker source deterministic identity drifted"
            )
        acceptances.append(dict(value))
        latches.append(
            {
                "worker_latch": dict(latch.value),
                "worker_latch_key": latch.key,
                "worker_latch_file_sha256": latch.file_sha256,
                "worker_latch_version_id": latch.version_id,
                "worker_latch_etag": latch.etag,
                "worker_latch_last_modified": latch.last_modified,
            }
        )
        observations.append(
            {
                "worker_controller_observation": dict(observation.value),
                "worker_controller_observation_key": observation.key,
                "worker_controller_observation_file_sha256": (observation.file_sha256),
                "worker_controller_observation_body_sha256": (observation.body_sha256),
            }
        )
    return acceptances, latches, observations


def _common_kwargs(
    *,
    static: _StaticAuthority,
    worker_latch: _Artifact | None = None,
    worker_observation: dict[str, object],
    worker_observation_key: str,
    worker_instance: dict[str, object],
    active_ids: list[str],
    prior_acceptances: list[dict[str, object]],
    prior_latches: list[dict[str, object]],
    prior_observations: list[dict[str, object]],
    accepted_at: str,
    now: str,
) -> dict[str, object]:
    latch = static.latch if worker_latch is None else worker_latch
    return {
        "descriptor": dict(static.descriptor.value),
        "intent": dict(static.intent.value),
        "controller_baseline": dict(static.baseline.value),
        "controller_baseline_key": static.baseline.key,
        "controller_baseline_file_sha256": static.baseline.file_sha256,
        "submission_acquisition": dict(static.acquisition.value),
        "submission_acquisition_key": static.acquisition.key,
        "submission_acquisition_file_sha256": static.acquisition.file_sha256,
        "submission_accepted": dict(static.submission_accepted.value),
        "submission_accepted_key": static.submission_accepted.key,
        "submission_accepted_file_sha256": static.submission_accepted.file_sha256,
        "job_binding": dict(static.binding.value),
        "job_binding_key": static.binding.key,
        "job_binding_file_sha256": static.binding.file_sha256,
        "submission_controller_observation": dict(static.submission_observation.value),
        "submission_controller_observation_key": static.submission_observation.key,
        "submission_controller_observation_file_sha256": (
            static.submission_observation.file_sha256
        ),
        "worker_controller_observation": dict(worker_observation),
        "worker_controller_observation_key": worker_observation_key,
        "worker_controller_observation_file_sha256": hashlib.sha256(
            worker_v2_canonical_bytes(worker_observation)
        ).hexdigest(),
        "worker_latch": dict(latch.value),
        "worker_latch_key": latch.key,
        "worker_latch_file_sha256": latch.file_sha256,
        "worker_latch_version_id": latch.version_id,
        "worker_latch_etag": latch.etag,
        "worker_latch_last_modified": latch.last_modified,
        "worker_instance": dict(worker_instance),
        "active_campaign_p5_instance_ids": list(active_ids),
        "prior_worker_acceptance_chain": [dict(item) for item in prior_acceptances],
        "prior_worker_latch_chain": [dict(item) for item in prior_latches],
        "prior_worker_controller_observation_chain": [
            dict(item) for item in prior_observations
        ],
        "accepted_at": accepted_at,
        "now": now,
    }


def _read_optional_current(
    s3: object,
    *,
    bucket: str,
    key: str,
    run_id: str,
) -> _Artifact | None:
    try:
        response = s3.head_object(  # type: ignore[attr-defined]
            Bucket=bucket,
            Key=key,
            ChecksumMode="ENABLED",
            ExpectedBucketOwner=APPROVED_ACCOUNT_ID,
        )
    except Exception as error:
        if _is_delete_marker_error(error):
            raise WorkerStartCoordinatorError(
                "unversioned HEAD resolved an S3 delete marker"
            ) from error
        if _exception_code(error) in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise WorkerStartCoordinatorError("unversioned HEAD failed") from error
    version = _transport_fields(
        response,
        label=f"unversioned HEAD s3://{bucket}/{key}",
    )[0]
    return _read_artifact(
        s3,
        bucket=bucket,
        key=key,
        run_id=run_id,
        newline=False,
        version_id=version,
    )


def _put_immutable(
    services: WorkerStartCoordinatorServices,
    *,
    bucket: str,
    key: str,
    run_id: str,
    value: dict[str, object],
    permit_authenticated_foreign_winner: bool = False,
) -> tuple[_Artifact, bool]:
    raw = worker_v2_canonical_bytes(value)
    body_sha256 = _body_digest(value)
    existing = _read_optional_current(
        services.s3,
        bucket=bucket,
        key=key,
        run_id=run_id,
    )
    if existing is not None:
        if existing.raw != raw and not permit_authenticated_foreign_winner:
            raise WorkerStartCoordinatorError(
                "a foreign immutable winner occupies the deterministic key"
            )
        return existing, False
    checksum = base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
    try:
        response = services.s3.put_object(  # type: ignore[attr-defined]
            Bucket=bucket,
            Key=key,
            Body=raw,
            IfNoneMatch="*",
            ExpectedBucketOwner=APPROVED_ACCOUNT_ID,
            ChecksumAlgorithm="SHA256",
            ChecksumSHA256=checksum,
            ContentType="application/json",
            Metadata={
                "glm52-run-id": run_id,
                "glm52-body-sha256": body_sha256,
            },
        )
    except Exception as error:
        if not _put_error_may_have_persisted(error):
            raise WorkerStartCoordinatorError(
                "conditional PUT failed definitively"
            ) from error
        # A conflict or carefully classified transport loss may have persisted
        # a winner. Resolve once; never retry the write.
        winner = _read_optional_current(
            services.s3,
            bucket=bucket,
            key=key,
            run_id=run_id,
        )
        if winner is None or (
            winner.raw != raw and not permit_authenticated_foreign_winner
        ):
            raise WorkerStartCoordinatorError(
                "conditional PUT did not converge to the exact winner"
            ) from error
        return winner, False
    if not isinstance(response, dict):
        raise WorkerStartCoordinatorError("conditional PUT response is malformed")
    metadata = response.get("ResponseMetadata")
    version = response.get("VersionId")
    http_status = metadata.get("HTTPStatusCode") if isinstance(metadata, dict) else None
    if (
        not isinstance(metadata, dict)
        or type(http_status) is not int
        or http_status != 200
        or not isinstance(version, str)
        or _VERSION_ID.fullmatch(version) is None
    ):
        raise WorkerStartCoordinatorError("conditional PUT response is inconclusive")
    artifact = _read_artifact(
        services.s3,
        bucket=bucket,
        key=key,
        run_id=run_id,
        newline=False,
        version_id=version,
        expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        expected_body_sha256=body_sha256,
    )
    if artifact.raw != raw:
        raise WorkerStartCoordinatorError("conditional PUT exact reread drifted")
    return artifact, True


def _read_winner_latch(
    services: WorkerStartCoordinatorServices,
    *,
    static: _StaticAuthority,
    accepted: dict[str, object],
    instance_id: str,
    run_id: str,
) -> _Artifact:
    latch_body_sha256 = _required_sha256(
        accepted.get("worker_latch_body_sha256"),
        label="winner latch body SHA-256",
    )
    latch_file_sha256 = _required_sha256(
        accepted.get("worker_latch_file_sha256"),
        label="winner latch file SHA-256",
    )
    latch_key = accepted.get("worker_latch_key")
    expected_key = (
        f"{static.dynamic_prefix}worker-latches/{instance_id}/{latch_body_sha256}.json"
    )
    latch_version = accepted.get("worker_latch_version_id")
    if (
        latch_key != expected_key
        or not isinstance(latch_version, str)
        or _VERSION_ID.fullmatch(latch_version) is None
    ):
        raise WorkerStartCoordinatorError("winner latch source pointer is invalid")
    latch = _read_artifact(
        services.s3,
        bucket=static.descriptor.bucket,
        key=expected_key,
        run_id=run_id,
        newline=False,
        version_id=latch_version,
        expected_file_sha256=latch_file_sha256,
        expected_body_sha256=latch_body_sha256,
    )
    if (
        accepted.get("worker_latch_etag") != latch.etag
        or accepted.get("worker_latch_last_modified") != latch.last_modified
    ):
        raise WorkerStartCoordinatorError("winner latch transport identity drifted")
    try:
        validated_latch = validate_worker_start_latch_v2(
            latch.value,
            descriptor=static.descriptor.value,
            intent=static.intent.value,
        )
        deterministic_key = worker_start_latch_v2_s3_key(worker_latch=validated_latch)
    except (WorkerStartValidationError, TypeError, ValueError) as error:
        raise WorkerStartCoordinatorError(
            "winner latch failed exact validation"
        ) from error
    if (
        validated_latch != latch.value
        or deterministic_key != latch.key
        or latch.value.get("instance_id") != instance_id
    ):
        raise WorkerStartCoordinatorError("winner latch deterministic identity drifted")
    return latch


def _read_winner_observation(
    services: WorkerStartCoordinatorServices,
    *,
    static: _StaticAuthority,
    accepted: dict[str, object],
    instance_id: str,
    run_id: str,
) -> _Artifact:
    body_sha256 = _required_sha256(
        accepted.get("worker_controller_observation_body_sha256"),
        label="winner worker observation body SHA-256",
    )
    file_sha256 = _required_sha256(
        accepted.get("worker_controller_observation_file_sha256"),
        label="winner worker observation file SHA-256",
    )
    expected_key = (
        f"{static.dynamic_prefix}worker-controller-observations/"
        f"{instance_id}/{body_sha256}.json"
    )
    if accepted.get("worker_controller_observation_key") != expected_key:
        raise WorkerStartCoordinatorError(
            "winner worker observation pointer is invalid"
        )
    observation = _read_artifact(
        services.s3,
        bucket=static.descriptor.bucket,
        key=expected_key,
        run_id=run_id,
        newline=False,
        expected_file_sha256=file_sha256,
        expected_body_sha256=body_sha256,
    )
    _validate_worker_observation(observation.value)
    return observation


def _fresh_replay_current_winner(
    services: WorkerStartCoordinatorServices,
    *,
    static: _StaticAuthority,
    current_key: str,
    now_iso: str,
    now_time: datetime,
    run_id: str,
    instance_id: str,
) -> _FreshReplay:
    identified = _read_optional_current(
        services.s3,
        bucket=static.descriptor.bucket,
        key=current_key,
        run_id=run_id,
    )
    if identified is None:
        raise WorkerStartCoordinatorError("worker acceptance winner is absent")
    accepted = identified.value
    if accepted.get("instance_id") != instance_id:
        raise WorkerStartCoordinatorError("worker acceptance winner instance drifted")
    latch = _read_winner_latch(
        services,
        static=static,
        accepted=accepted,
        instance_id=instance_id,
        run_id=run_id,
    )
    observation = _read_winner_observation(
        services,
        static=static,
        accepted=accepted,
        instance_id=instance_id,
        run_id=run_id,
    )
    fresh_prior, fresh_current = _load_acceptance_artifacts(
        services,
        static=static,
        current_key=current_key,
    )
    if fresh_current is None or fresh_current != identified:
        raise WorkerStartCoordinatorError(
            "worker acceptance winner changed during fresh replay"
        )
    complete_chain = _order_prior_acceptances([*fresh_prior, fresh_current])
    if (
        not complete_chain
        or complete_chain[-1] != fresh_current
        or complete_chain[-1].key != current_key
    ):
        raise WorkerStartCoordinatorError(
            "worker acceptance winner is not the exact fresh chain tip"
        )
    ordered_prior = complete_chain[:-1]
    prior_acceptances, prior_latches, prior_observations = _prior_evidence(
        services,
        static=static,
        ordered=ordered_prior,
    )
    snapshot_fields = {
        "worker_cluster_name": accepted.get("worker_cluster_name"),
        "instance_id": accepted.get("instance_id"),
        "worker_instance_type": accepted.get("worker_instance_type"),
        "worker_image_id": accepted.get("worker_image_id"),
        "worker_instance_lifecycle": accepted.get("worker_instance_lifecycle"),
        "worker_instance_state": accepted.get("worker_instance_state"),
        "worker_instance_launch_time": accepted.get("worker_instance_launch_time"),
        "worker_ray_cluster_name": accepted.get("worker_ray_cluster_name"),
        "worker_skypilot_cluster_name": accepted.get("worker_skypilot_cluster_name"),
        "worker_campaign_tags": accepted.get("worker_campaign_tags"),
        "ec2_observed_at": accepted.get("ec2_observed_at"),
    }
    active_ids = accepted.get("active_campaign_p5_instance_ids")
    accepted_at = accepted.get("accepted_at")
    snapshot_fields["worker_campaign_tags"] = _validate_worker_campaign_tags(
        snapshot_fields["worker_campaign_tags"],
        run_id=run_id,
    )
    if type(active_ids) is not list or not isinstance(accepted_at, str):
        raise WorkerStartCoordinatorError("existing worker snapshot is malformed")
    _, immutable_accepted_at = _canonical_time(
        accepted_at,
        label="existing worker accepted_at",
    )
    _, immutable_observed_at = _canonical_time(
        observation.value.get("observed_at"),
        label="existing worker observation observed_at",
    )
    _, immutable_ec2_at = _canonical_time(
        snapshot_fields["ec2_observed_at"],
        label="existing worker EC2 observed_at",
    )
    if now_time < max(
        immutable_accepted_at,
        immutable_observed_at,
        immutable_ec2_at,
    ):
        raise WorkerStartCoordinatorError(
            "replay clock predates immutable worker evidence"
        )
    kwargs = _common_kwargs(
        static=static,
        worker_latch=latch,
        worker_observation=observation.value,
        worker_observation_key=observation.key,
        worker_instance=snapshot_fields,
        active_ids=list(active_ids),
        prior_acceptances=prior_acceptances,
        prior_latches=prior_latches,
        prior_observations=prior_observations,
        accepted_at=accepted_at,
        now=now_iso,
    )
    decision = decide_worker_start_acceptance_v2(
        **kwargs,
        existing_worker_acceptance=accepted,
    )
    if decision.action != "idempotent-complete":
        raise WorkerStartCoordinatorError(
            "existing worker acceptance failed immutable replay"
        )
    try:
        validate_worker_start_accepted_v2(
            accepted,
            **kwargs,
        )
    except (WorkerStartValidationError, TypeError, ValueError) as error:
        raise WorkerStartCoordinatorError(
            "existing worker acceptance failed exact validation"
        ) from error
    if (
        worker_start_accepted_v2_s3_key(worker_start_accepted=accepted)
        != fresh_current.key
    ):
        raise WorkerStartCoordinatorError("existing worker acceptance key drifted")
    _audit_no_legacy(
        services.s3,
        bucket=static.descriptor.bucket,
        dynamic_prefix=static.dynamic_prefix,
    )
    outcome = WorkerStartCoordinatorOutcome(
        status="idempotent-complete",
        decision_action=decision.action,
        reason=decision.reason,
        run_id=run_id,
        instance_id=instance_id,
        worker_latch_key=latch.key,
        worker_controller_observation_key=observation.key,
        worker_acceptance_key=fresh_current.key,
        worker_acceptance_body_sha256=fresh_current.body_sha256,
        prior_worker_acceptance_count=len(ordered_prior),
        published_observation=False,
        published_acceptance=False,
    )
    return _FreshReplay(
        outcome=outcome,
        acceptance=fresh_current,
        observation=observation,
        latch=latch,
    )


def _valid_key(value: object, *, run_id: str) -> bool:
    return (
        isinstance(value, str)
        and value.startswith(f"campaigns/{run_id}/")
        and not value.startswith("/")
        and not value.endswith("/")
        and "\\" not in value
        and "*" not in value
        and "?" not in value
        and not any(character.isspace() for character in value)
        and all(part not in {"", ".", ".."} for part in value.split("/"))
    )


def _validate_request(
    services: object,
    request: object,
) -> tuple[WorkerStartCoordinatorServices, WorkerStartCoordinatorRequest, str, str]:
    if type(services) is not WorkerStartCoordinatorServices:
        raise WorkerStartCoordinatorError("services must use the exact dataclass")
    if (
        {field.name for field in fields(services)}
        != {"sts", "s3", "ec2", "ssm", "clock", "sleep"}
        or not callable(services.clock)
        or not callable(services.sleep)
    ):
        raise WorkerStartCoordinatorError("services schema is invalid")
    if type(request) is not WorkerStartCoordinatorRequest:
        raise WorkerStartCoordinatorError("request must use the exact dataclass")
    if {field.name for field in fields(request)} != {
        "account_id",
        "region",
        "bucket",
        "descriptor_key",
        "descriptor_file_sha256",
        "intent_key",
        "intent_file_sha256",
        "intent_body_sha256",
        "worker_latch_key",
        "worker_latch_version_id",
    }:
        raise WorkerStartCoordinatorError("request schema mismatch")
    if request.account_id != APPROVED_ACCOUNT_ID or request.region != APPROVED_REGION:
        raise WorkerStartCoordinatorError("request targets an unapproved AWS authority")
    if (
        not isinstance(request.bucket, str)
        or _BUCKET.fullmatch(request.bucket) is None
        or any(separator in request.bucket for separator in ("..", ".-", "-."))
        or request.bucket.endswith(
            (
                "-s3alias",
                "--ol-s3",
                ".mrap",
                "--x-s3",
                "--table-s3",
            )
        )
    ):
        raise WorkerStartCoordinatorError("bucket is invalid")
    for label, value in (
        ("descriptor file SHA-256", request.descriptor_file_sha256),
        ("intent file SHA-256", request.intent_file_sha256),
        ("intent body SHA-256", request.intent_body_sha256),
    ):
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise WorkerStartCoordinatorError(f"{label} is invalid")
    intent_match = (
        re.fullmatch(
            r"campaigns/([A-Za-z0-9][A-Za-z0-9._-]{0,127})/"
            r"submissions/qualification/intents/([0-9a-f]{64})/"
            r"SKYPILOT_SUBMISSION_INTENT\.json",
            request.intent_key,
        )
        if isinstance(request.intent_key, str)
        else None
    )
    if intent_match is None or intent_match.group(2) != request.intent_body_sha256:
        raise WorkerStartCoordinatorError("intent key is invalid")
    run_id = intent_match.group(1)
    if _RUN_ID.fullmatch(run_id) is None:
        raise WorkerStartCoordinatorError("run ID is invalid")
    descriptor_prefix = f"campaigns/{run_id}/submissions/qualification/"
    descriptor_tail = (
        request.descriptor_key.removeprefix(descriptor_prefix)
        if isinstance(request.descriptor_key, str)
        else ""
    )
    if (
        not _valid_key(request.descriptor_key, run_id=run_id)
        or not request.descriptor_key.startswith(descriptor_prefix)
        or not descriptor_tail.endswith(".json")
        or descriptor_tail.rsplit("/", 1)[-1] == ".json"
        or any(
            _SAFE_NAME.fullmatch(part) is None for part in descriptor_tail.split("/")
        )
    ):
        raise WorkerStartCoordinatorError("descriptor key is invalid")
    latch_match = (
        re.fullmatch(
            rf"campaigns/{re.escape(run_id)}/monitor/must-start/qualification/"
            rf"{re.escape(request.intent_body_sha256)}/worker-latches/"
            r"(i-[0-9a-f]{17})/([0-9a-f]{64})\.json",
            request.worker_latch_key,
        )
        if isinstance(request.worker_latch_key, str)
        else None
    )
    if latch_match is None:
        raise WorkerStartCoordinatorError("worker latch key is invalid")
    if (
        not isinstance(request.worker_latch_version_id, str)
        or _VERSION_ID.fullmatch(request.worker_latch_version_id) is None
    ):
        raise WorkerStartCoordinatorError("worker latch VersionId is invalid")
    return services, request, run_id, latch_match.group(1)


def _require_caller_identity(services: WorkerStartCoordinatorServices) -> None:
    try:
        response = services.sts.get_caller_identity()  # type: ignore[attr-defined]
    except Exception as error:
        raise WorkerStartCoordinatorError("STS caller identity failed") from error
    if not isinstance(response, dict) or set(response) < {"Account", "Arn", "UserId"}:
        raise WorkerStartCoordinatorError("STS caller identity response is malformed")
    arn = response.get("Arn")
    match = _CALLER_ARN.fullmatch(arn) if isinstance(arn, str) else None
    if (
        response.get("Account") != APPROVED_ACCOUNT_ID
        or match is None
        or match.group(1) != APPROVED_ACCOUNT_ID
        or not isinstance(response.get("UserId"), str)
        or not response["UserId"]
    ):
        raise WorkerStartCoordinatorError("STS caller identity is not approved")


def coordinate_worker_start_acceptance_v2(
    *,
    services: WorkerStartCoordinatorServices,
    request: WorkerStartCoordinatorRequest,
) -> WorkerStartCoordinatorOutcome:
    """Coordinate one exact worker admission from injected AWS authorities."""

    services, request, run_id, instance_id = _validate_request(services, request)
    _require_caller_identity(services)
    static = _load_static_authority(
        services,
        request,
        run_id=run_id,
    )
    if static.latch.value.get("instance_id") != instance_id:
        raise WorkerStartCoordinatorError("request and latch instance IDs drifted")
    current_key = _acceptance_key(
        static,
        instance_id=instance_id,
        latch_body_sha256=static.latch.body_sha256,
    )
    prior_artifacts, existing_current = _load_acceptance_artifacts(
        services,
        static=static,
        current_key=current_key,
    )
    if existing_current is not None:
        try:
            sampled = services.clock()
        except Exception as error:
            raise WorkerStartCoordinatorError("coordinator clock failed") from error
        now_iso, now_time = _canonical_time(sampled, label="coordinator clock")
        replay = _fresh_replay_current_winner(
            services,
            static=static,
            current_key=current_key,
            now_iso=now_iso,
            now_time=now_time,
            run_id=run_id,
            instance_id=instance_id,
        )
        return replay.outcome
    ordered_prior = _order_prior_acceptances(prior_artifacts)
    prior_acceptances, prior_latches, prior_observations = _prior_evidence(
        services,
        static=static,
        ordered=ordered_prior,
    )
    worker_observation = _read_controller_history(services, static=static)
    observation_body_sha256 = worker_observation.get("observation_body_sha256")
    if not isinstance(observation_body_sha256, str):
        raise WorkerStartCoordinatorError(
            "worker controller observation digest is invalid"
        )
    worker_observation_key = (
        f"{static.dynamic_prefix}worker-controller-observations/"
        f"{instance_id}/{observation_body_sha256}.json"
    )
    worker_instance, active_ids, now_iso, now_time = _worker_ec2_authority(
        services,
        static=static,
        worker_observation=worker_observation,
        instance_id=instance_id,
    )
    _, latch_modified = _canonical_time(
        static.latch.last_modified,
        label="worker latch LastModified",
    )
    _, observation_time = _canonical_time(
        worker_observation.get("observed_at"),
        label="worker controller observed_at",
    )
    accepted_time = max(latch_modified, observation_time, now_time)
    if accepted_time > now_time:
        raise WorkerStartCoordinatorError(
            "fresh controller evidence is later than the coordinator clock"
        )
    accepted_at = accepted_time.astimezone(UTC).isoformat().replace("+00:00", "Z")
    kwargs = _common_kwargs(
        static=static,
        worker_observation=worker_observation,
        worker_observation_key=worker_observation_key,
        worker_instance=worker_instance,
        active_ids=active_ids,
        prior_acceptances=prior_acceptances,
        prior_latches=prior_latches,
        prior_observations=prior_observations,
        accepted_at=accepted_at,
        now=now_iso,
    )
    decision = decide_worker_start_acceptance_v2(
        **kwargs,
        existing_worker_acceptance=None,
    )
    if decision.action == "fail-closed":
        raise WorkerStartCoordinatorError(decision.reason)
    if decision.action == "wait-for-authority":
        return WorkerStartCoordinatorOutcome(
            status="waiting-worker-authority",
            decision_action=decision.action,
            reason=decision.reason,
            run_id=run_id,
            instance_id=instance_id,
            worker_latch_key=static.latch.key,
            worker_controller_observation_key=None,
            worker_acceptance_key=None,
            worker_acceptance_body_sha256=None,
            prior_worker_acceptance_count=len(prior_acceptances),
            published_observation=False,
            published_acceptance=False,
        )
    if decision.action not in {"accept-initial", "accept-managed-recovery"}:
        raise WorkerStartCoordinatorError("worker decision action is unexpected")
    try:
        accepted = build_worker_start_accepted_v2(**kwargs)
    except (WorkerStartValidationError, TypeError, ValueError) as error:
        raise WorkerStartCoordinatorError(
            "worker acceptance builder rejected authenticated evidence"
        ) from error
    acceptance_key = worker_start_accepted_v2_s3_key(worker_start_accepted=accepted)
    if acceptance_key != current_key:
        raise WorkerStartCoordinatorError("worker acceptance deterministic key drifted")
    observation_artifact, published_observation = _put_immutable(
        services,
        bucket=request.bucket,
        key=worker_observation_key,
        run_id=run_id,
        value=worker_observation,
    )
    if observation_artifact.value != worker_observation:
        raise WorkerStartCoordinatorError(
            "worker observation winner drifted from accepted evidence"
        )
    acceptance_artifact, published_acceptance = _put_immutable(
        services,
        bucket=request.bucket,
        key=acceptance_key,
        run_id=run_id,
        value=accepted,
        permit_authenticated_foreign_winner=True,
    )
    fresh = _fresh_replay_current_winner(
        services,
        static=static,
        current_key=current_key,
        now_iso=now_iso,
        now_time=now_time,
        run_id=run_id,
        instance_id=instance_id,
    )
    same_candidate = fresh.acceptance.value == accepted
    final_status = (
        (
            "accepted-initial"
            if decision.action == "accept-initial"
            else "accepted-managed-recovery"
        )
        if same_candidate
        else fresh.outcome.status
    )
    final_action = decision.action if same_candidate else fresh.outcome.decision_action
    final_reason = decision.reason if same_candidate else fresh.outcome.reason
    observation_was_final = (
        published_observation and observation_artifact == fresh.observation
    )
    acceptance_was_final = (
        published_acceptance and acceptance_artifact == fresh.acceptance
    )
    return WorkerStartCoordinatorOutcome(
        status=final_status,
        decision_action=final_action,
        reason=final_reason,
        run_id=run_id,
        instance_id=instance_id,
        worker_latch_key=fresh.latch.key,
        worker_controller_observation_key=fresh.observation.key,
        worker_acceptance_key=fresh.acceptance.key,
        worker_acceptance_body_sha256=fresh.acceptance.body_sha256,
        prior_worker_acceptance_count=(fresh.outcome.prior_worker_acceptance_count),
        published_observation=observation_was_final,
        published_acceptance=acceptance_was_final,
    )


__all__ = [
    "WorkerStartCoordinatorError",
    "WorkerStartCoordinatorOutcome",
    "WorkerStartCoordinatorRequest",
    "WorkerStartCoordinatorServices",
    "coordinate_worker_start_acceptance_v2",
]
