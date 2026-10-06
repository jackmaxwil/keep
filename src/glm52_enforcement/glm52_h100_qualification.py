"""Enforcement-native real-H100 replacement and training-smoke gate."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import datetime, timezone

_SHA = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_INSTANCE_ID = re.compile(r"^i-(?:[0-9a-f]{8}|[0-9a-f]{17})$")
_SOURCE_BODY_FIELDS = {
    "schema_version",
    "record_type",
    "run_id",
    "campaign_identity_sha256",
    "repo_tar_sha256",
    "descriptor_body_sha256",
    "instance_id",
    "allocation_record_sha256",
    "allocation_body_sha256",
    "checkpoint_marker_sha256",
    "published_at",
}
_SOURCE_FIELDS = _SOURCE_BODY_FIELDS | {"source_body_sha256"}
_TERMINATION_BODY_FIELDS = {
    "schema_version",
    "record_type",
    "run_id",
    "campaign_identity_sha256",
    "repo_tar_sha256",
    "descriptor_body_sha256",
    "instance_id",
    "allocation_record_sha256",
    "allocation_body_sha256",
    "checkpoint_marker_sha256",
    "source_body_sha256",
    "requested_at",
}
_TERMINATION_FIELDS = _TERMINATION_BODY_FIELDS | {"request_body_sha256"}
_ALLOCATION_BODY_FIELDS = {
    "record_type",
    "run_id",
    "job_id",
    "instance_id",
    "launched_at",
    "observed_at",
    "execution_deadline",
    "approval_sha256",
    "gpu_spend_authority_sha256",
    "gpu_spend_record_sha256",
    "gpu_spend_ledger_sha256",
    "remaining_gpu_seconds",
    "estimated_gpu_cost_usd",
}
_ALLOCATION_FIELDS = _ALLOCATION_BODY_FIELDS | {"allocation_body_sha256"}
_BODY_FIELDS = {
    "schema_version",
    "record_type",
    "run_id",
    "campaign_identity_sha256",
    "repo_tar_sha256",
    "qualification_cache_manifest_sha256",
    "first_instance_id",
    "replacement_instance_id",
    "first_allocation_record_sha256",
    "replacement_allocation_record_sha256",
    "source_checkpoint_marker_sha256",
    "parity_report_sha256",
    "training_smoke_sha256",
    "resumed_capture_sha256",
    "training_sample_steps",
    "peak_gpu_gib",
    "cross_node_resume",
    "completed_at",
}
_FIELDS = _BODY_FIELDS | {"ready_body_sha256"}


class H100QualificationValidationError(ValueError):
    """Raised when an H100 qualification authority is malformed or foreign."""


def _from_iso8601(value: str) -> datetime:
    """Parse canonical ``Z`` timestamps on Python 3.9 through current."""

    if value.endswith("Z"):
        value = f"{value[:-1]}+00:00"
    return datetime.fromisoformat(value)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _canonical_time(value: datetime | str, *, field: str) -> str:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = _from_iso8601(value)
        except ValueError as error:
            raise H100QualificationValidationError(
                f"H100 qualification {field} must be ISO-8601"
            ) from error
    else:
        raise H100QualificationValidationError(
            f"H100 qualification {field} must be ISO-8601"
        )
    if parsed.tzinfo is None:
        raise H100QualificationValidationError(
            f"H100 qualification {field} must be timezone-aware"
        )
    # datetime.UTC is Python 3.11-only; worker/operator shells may use 3.9.
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")  # noqa: UP017


def _parse_time(value: datetime | str, *, field: str) -> datetime:
    return _from_iso8601(_canonical_time(value, field=field))


def _validate_exact_digest(
    value: Mapping[str, object],
    *,
    fields: set[str],
    digest_field: str,
    record_type: str,
) -> dict[str, object]:
    if set(value) != fields:
        raise H100QualificationValidationError(
            f"H100 qualification {record_type} schema mismatch"
        )
    schema_version = value.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 1
        or value.get("record_type") != record_type
    ):
        raise H100QualificationValidationError(
            f"H100 qualification {record_type} schema mismatch"
        )
    body = dict(value)
    digest = body.pop(digest_field)
    if (
        not isinstance(digest, str)
        or digest != hashlib.sha256(_canonical(body)).hexdigest()
    ):
        raise H100QualificationValidationError(
            f"H100 qualification {record_type} body SHA-256 mismatch"
        )
    return dict(value)


def _validate_common_source_identity(
    value: Mapping[str, object],
    *,
    record_label: str,
) -> None:
    run_id = value.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise H100QualificationValidationError(
            f"H100 qualification {record_label} run_id is invalid"
        )
    for field in (
        "campaign_identity_sha256",
        "repo_tar_sha256",
        "descriptor_body_sha256",
        "allocation_record_sha256",
        "allocation_body_sha256",
        "checkpoint_marker_sha256",
    ):
        digest = value.get(field)
        if not isinstance(digest, str) or _SHA.fullmatch(digest) is None:
            raise H100QualificationValidationError(
                f"H100 qualification {record_label} {field} is invalid"
            )
    instance_id = value.get("instance_id")
    if not isinstance(instance_id, str) or _INSTANCE_ID.fullmatch(instance_id) is None:
        raise H100QualificationValidationError(
            f"H100 qualification {record_label} instance ID is invalid"
        )


def validate_h100_runtime_allocation(
    value: Mapping[str, object],
    *,
    expected_run_id: str | None = None,
    expected_job_id: str | None = None,
    expected_instance_id: str | None = None,
    expected_approval_sha256: str | None = None,
    expected_gpu_spend_authority_sha256: str | None = None,
    expected_gpu_spend_record_sha256: str | None = None,
    expected_allocation_body_sha256: str | None = None,
    max_remaining_gpu_seconds: int | None = None,
    max_estimated_gpu_cost_usd: float | None = None,
) -> dict[str, object]:
    """Authenticate one current allocation and its caller-supplied authority."""

    if (
        set(value) != _ALLOCATION_FIELDS
        or value.get("record_type") != "glm52_gpu_runtime_allocation_v1"
    ):
        raise H100QualificationValidationError(
            "H100 qualification runtime allocation schema mismatch"
        )
    body = dict(value)
    digest = body.pop("allocation_body_sha256")
    try:
        expected_digest = hashlib.sha256(_canonical(body)).hexdigest()
    except (TypeError, ValueError) as error:
        raise H100QualificationValidationError(
            "H100 qualification runtime allocation body is not canonical JSON"
        ) from error
    if not isinstance(digest, str) or digest != expected_digest:
        raise H100QualificationValidationError(
            "H100 qualification runtime allocation body SHA-256 mismatch"
        )

    run_id = value.get("run_id")
    job_id = value.get("job_id")
    instance_id = value.get("instance_id")
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise H100QualificationValidationError(
            "H100 qualification runtime allocation run_id is invalid"
        )
    if not isinstance(job_id, str) or _RUN_ID.fullmatch(job_id) is None:
        raise H100QualificationValidationError(
            "H100 qualification runtime allocation job_id is invalid"
        )
    if not isinstance(instance_id, str) or _INSTANCE_ID.fullmatch(instance_id) is None:
        raise H100QualificationValidationError(
            "H100 qualification runtime allocation instance ID is invalid"
        )
    for field in (
        "approval_sha256",
        "gpu_spend_authority_sha256",
        "gpu_spend_record_sha256",
        "gpu_spend_ledger_sha256",
    ):
        field_value = value.get(field)
        if not isinstance(field_value, str) or _SHA.fullmatch(field_value) is None:
            raise H100QualificationValidationError(
                f"H100 qualification runtime allocation {field} is invalid"
            )

    times = {
        field: _canonical_time(value[field], field=field)
        for field in ("launched_at", "observed_at", "execution_deadline")
    }
    for field, canonical in times.items():
        if value[field] != canonical:
            raise H100QualificationValidationError(
                f"H100 qualification runtime allocation {field} is not canonical"
            )
    launched_at = _parse_time(times["launched_at"], field="launched_at")
    observed_at = _parse_time(times["observed_at"], field="observed_at")
    execution_deadline = _parse_time(
        times["execution_deadline"],
        field="execution_deadline",
    )
    if launched_at > observed_at:
        raise H100QualificationValidationError(
            "H100 qualification allocation launch follows observation"
        )
    if execution_deadline <= observed_at:
        raise H100QualificationValidationError(
            "H100 qualification allocation deadline must follow observation"
        )

    remaining = value.get("remaining_gpu_seconds")
    if not isinstance(remaining, int) or isinstance(remaining, bool) or remaining < 0:
        raise H100QualificationValidationError(
            "H100 qualification allocation remaining_gpu_seconds is invalid"
        )
    if int((execution_deadline - observed_at).total_seconds()) != remaining:
        raise H100QualificationValidationError(
            "H100 qualification allocation deadline does not match remaining time"
        )
    cost = value.get("estimated_gpu_cost_usd")
    if (
        not isinstance(cost, (int, float))
        or isinstance(cost, bool)
        or not math.isfinite(float(cost))
        or float(cost) < 0
    ):
        raise H100QualificationValidationError(
            "H100 qualification allocation estimated cost is invalid"
        )

    expected = {
        "run_id": expected_run_id,
        "job_id": expected_job_id,
        "instance_id": expected_instance_id,
        "approval_sha256": expected_approval_sha256,
        "gpu_spend_authority_sha256": expected_gpu_spend_authority_sha256,
        "gpu_spend_record_sha256": expected_gpu_spend_record_sha256,
        "allocation_body_sha256": expected_allocation_body_sha256,
    }
    mismatched = [
        field
        for field, expected_value in expected.items()
        if expected_value is not None and value[field] != expected_value
    ]
    if mismatched:
        raise H100QualificationValidationError(
            "H100 qualification runtime allocation identity mismatch: "
            + ", ".join(sorted(mismatched))
        )
    if max_remaining_gpu_seconds is not None and remaining > max_remaining_gpu_seconds:
        raise H100QualificationValidationError(
            "H100 qualification allocation remaining time exceeds authority"
        )
    if max_estimated_gpu_cost_usd is not None and float(cost) > float(
        max_estimated_gpu_cost_usd
    ):
        raise H100QualificationValidationError(
            "H100 qualification allocation estimated cost exceeds authority"
        )
    return dict(value)


def build_h100_source_node_ready(
    *,
    run_id: str,
    campaign_identity_sha256: str,
    repo_tar_sha256: str,
    descriptor_body_sha256: str,
    instance_id: str,
    allocation_record_sha256: str,
    allocation_body_sha256: str,
    checkpoint_marker_sha256: str,
    published_at: datetime | str,
) -> dict[str, object]:
    """Build the immutable qualification source-node recovery authority."""

    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_h100_qualification_source_v1",
        "run_id": run_id,
        "campaign_identity_sha256": campaign_identity_sha256,
        "repo_tar_sha256": repo_tar_sha256,
        "descriptor_body_sha256": descriptor_body_sha256,
        "instance_id": instance_id,
        "allocation_record_sha256": allocation_record_sha256,
        "allocation_body_sha256": allocation_body_sha256,
        "checkpoint_marker_sha256": checkpoint_marker_sha256,
        "published_at": _canonical_time(published_at, field="published_at"),
    }
    marker = {
        **body,
        "source_body_sha256": hashlib.sha256(_canonical(body)).hexdigest(),
    }
    return validate_h100_source_node_ready(marker)


def validate_h100_source_node_ready(
    value: Mapping[str, object],
    *,
    expected_run_id: str | None = None,
    expected_campaign_identity_sha256: str | None = None,
    expected_repo_tar_sha256: str | None = None,
    expected_descriptor_body_sha256: str | None = None,
    expected_instance_id: str | None = None,
    expected_allocation_record_sha256: str | None = None,
    expected_allocation_body_sha256: str | None = None,
    expected_checkpoint_marker_sha256: str | None = None,
    expected_published_at: datetime | str | None = None,
    expected_source_body_sha256: str | None = None,
) -> dict[str, object]:
    """Validate a source marker and any caller-supplied immutable authority."""

    marker = _validate_exact_digest(
        value,
        fields=_SOURCE_FIELDS,
        digest_field="source_body_sha256",
        record_type="glm52_h100_qualification_source_v1",
    )
    _validate_common_source_identity(marker, record_label="source marker")
    published_at = _canonical_time(marker["published_at"], field="published_at")
    if marker["published_at"] != published_at:
        raise H100QualificationValidationError(
            "H100 qualification source marker published_at is not canonical"
        )
    expected_publication = (
        None
        if expected_published_at is None
        else _canonical_time(expected_published_at, field="expected_published_at")
    )
    expected = {
        "run_id": expected_run_id,
        "campaign_identity_sha256": expected_campaign_identity_sha256,
        "repo_tar_sha256": expected_repo_tar_sha256,
        "descriptor_body_sha256": expected_descriptor_body_sha256,
        "instance_id": expected_instance_id,
        "allocation_record_sha256": expected_allocation_record_sha256,
        "allocation_body_sha256": expected_allocation_body_sha256,
        "checkpoint_marker_sha256": expected_checkpoint_marker_sha256,
        "published_at": expected_publication,
        "source_body_sha256": expected_source_body_sha256,
    }
    mismatched = [
        field
        for field, expected_value in expected.items()
        if expected_value is not None and marker[field] != expected_value
    ]
    if mismatched:
        raise H100QualificationValidationError(
            "H100 qualification source identity mismatch: "
            + ", ".join(sorted(mismatched))
        )
    return marker


def build_h100_termination_requested(
    *,
    source_marker: Mapping[str, object],
    requested_at: datetime | str,
) -> dict[str, object]:
    """Build the AWS-owned request that authorizes one source termination."""

    source = validate_h100_source_node_ready(source_marker)
    requested = _canonical_time(requested_at, field="requested_at")
    if _parse_time(requested, field="requested_at") < _parse_time(
        str(source["published_at"]), field="published_at"
    ):
        raise H100QualificationValidationError(
            "H100 qualification requested_at precedes source publication"
        )
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_h100_qualification_termination_request_v1",
        "run_id": source["run_id"],
        "campaign_identity_sha256": source["campaign_identity_sha256"],
        "repo_tar_sha256": source["repo_tar_sha256"],
        "descriptor_body_sha256": source["descriptor_body_sha256"],
        "instance_id": source["instance_id"],
        "allocation_record_sha256": source["allocation_record_sha256"],
        "allocation_body_sha256": source["allocation_body_sha256"],
        "checkpoint_marker_sha256": source["checkpoint_marker_sha256"],
        "source_body_sha256": source["source_body_sha256"],
        "requested_at": requested,
    }
    marker = {
        **body,
        "request_body_sha256": hashlib.sha256(_canonical(body)).hexdigest(),
    }
    return validate_h100_termination_requested(marker, source_marker=source)


def validate_h100_termination_requested(
    value: Mapping[str, object],
    *,
    source_marker: Mapping[str, object],
) -> dict[str, object]:
    """Validate an AWS termination request against its exact source marker."""

    marker = _validate_exact_digest(
        value,
        fields=_TERMINATION_FIELDS,
        digest_field="request_body_sha256",
        record_type="glm52_h100_qualification_termination_request_v1",
    )
    _validate_common_source_identity(marker, record_label="termination request")
    source_digest = marker.get("source_body_sha256")
    if not isinstance(source_digest, str) or _SHA.fullmatch(source_digest) is None:
        raise H100QualificationValidationError(
            "H100 qualification termination request source_body_sha256 is invalid"
        )
    requested_at = _canonical_time(marker["requested_at"], field="requested_at")
    if marker["requested_at"] != requested_at:
        raise H100QualificationValidationError(
            "H100 qualification termination request requested_at is not canonical"
        )
    source = validate_h100_source_node_ready(source_marker)
    bound_fields = (
        "run_id",
        "campaign_identity_sha256",
        "repo_tar_sha256",
        "descriptor_body_sha256",
        "instance_id",
        "allocation_record_sha256",
        "allocation_body_sha256",
        "checkpoint_marker_sha256",
        "source_body_sha256",
    )
    if any(marker[field] != source[field] for field in bound_fields):
        raise H100QualificationValidationError(
            "H100 qualification termination request source identity mismatch"
        )
    if _parse_time(requested_at, field="requested_at") < _parse_time(
        str(source["published_at"]), field="published_at"
    ):
        raise H100QualificationValidationError(
            "H100 qualification requested_at precedes source publication"
        )
    return marker


def build_h100_resume_ready(
    *,
    run_id: str,
    campaign_identity_sha256: str,
    repo_tar_sha256: str,
    qualification_cache_manifest_sha256: str,
    first_instance_id: str,
    replacement_instance_id: str,
    first_allocation_record_sha256: str,
    replacement_allocation_record_sha256: str,
    source_checkpoint_marker_sha256: str,
    parity_report_sha256: str,
    training_smoke_sha256: str,
    resumed_capture_sha256: str,
    peak_gpu_gib: float,
    completed_at: datetime,
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_h100_resume_ready_v1",
        "run_id": run_id,
        "campaign_identity_sha256": campaign_identity_sha256,
        "repo_tar_sha256": repo_tar_sha256,
        "qualification_cache_manifest_sha256": (qualification_cache_manifest_sha256),
        "first_instance_id": first_instance_id,
        "replacement_instance_id": replacement_instance_id,
        "first_allocation_record_sha256": first_allocation_record_sha256,
        "replacement_allocation_record_sha256": replacement_allocation_record_sha256,
        "source_checkpoint_marker_sha256": source_checkpoint_marker_sha256,
        "parity_report_sha256": parity_report_sha256,
        "training_smoke_sha256": training_smoke_sha256,
        "resumed_capture_sha256": resumed_capture_sha256,
        "training_sample_steps": 2,
        "peak_gpu_gib": peak_gpu_gib,
        "cross_node_resume": True,
        "completed_at": _canonical_time(completed_at, field="completed_at"),
    }
    return validate_h100_resume_ready(
        {**body, "ready_body_sha256": hashlib.sha256(_canonical(body)).hexdigest()}
    )


def validate_h100_resume_ready(
    value: Mapping[str, object],
    *,
    verify_body_sha: bool = True,
) -> dict[str, object]:
    expected = _FIELDS if verify_body_sha else _BODY_FIELDS
    if set(value) != expected:
        raise ValueError("H100 resume readiness schema mismatch")
    body = dict(value)
    if verify_body_sha:
        digest = body.pop("ready_body_sha256")
        if digest != hashlib.sha256(_canonical(body)).hexdigest():
            raise ValueError("H100 resume readiness body SHA-256 mismatch")
    if (
        value.get("schema_version") != 1
        or value.get("record_type") != "glm52_h100_resume_ready_v1"
    ):
        raise ValueError("H100 resume readiness schema mismatch")
    for field in (
        "campaign_identity_sha256",
        "repo_tar_sha256",
        "qualification_cache_manifest_sha256",
        "first_allocation_record_sha256",
        "replacement_allocation_record_sha256",
        "source_checkpoint_marker_sha256",
        "parity_report_sha256",
        "training_smoke_sha256",
        "resumed_capture_sha256",
    ):
        if (
            not isinstance(value.get(field), str)
            or _SHA.fullmatch(value[field]) is None
        ):
            raise ValueError(f"H100 resume readiness {field} is invalid")
    if not isinstance(value.get("run_id"), str) or not value["run_id"]:
        raise ValueError("H100 resume readiness run_id is invalid")
    first = value.get("first_instance_id")
    replacement = value.get("replacement_instance_id")
    if (
        not isinstance(first, str)
        or not isinstance(replacement, str)
        or (first == replacement)
    ):
        raise ValueError("H100 qualification instance identities must be distinct")
    if value.get("training_sample_steps") != 2:
        raise ValueError("H100 qualification must run exactly two training steps")
    peak = value.get("peak_gpu_gib")
    if not isinstance(peak, (int, float)) or not 0 <= float(peak) < 70:
        raise ValueError("H100 qualification peak must remain below 70 GiB")
    if value.get("cross_node_resume") is not True:
        raise ValueError("H100 qualification must prove cross-node resume")
    parsed = _from_iso8601(str(value["completed_at"]))
    if parsed.tzinfo is None:
        raise ValueError("H100 qualification completion must be timezone-aware")
    return dict(value)


__all__ = [
    "H100QualificationValidationError",
    "build_h100_resume_ready",
    "build_h100_source_node_ready",
    "build_h100_termination_requested",
    "validate_h100_resume_ready",
    "validate_h100_runtime_allocation",
    "validate_h100_source_node_ready",
    "validate_h100_termination_requested",
]
