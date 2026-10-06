"""Pure production prelaunch acquisition authority tests."""

from __future__ import annotations

import builtins
import copy
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import replace
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any

import pytest

from mlx_vq.quality.glm52_sky_production_submission import (
    VersionedJsonArtifact,
    build_production_submission_intent,
    production_submission_intent_file_bytes,
    production_submission_intent_s3_key,
)


ROOT = Path(__file__).resolve().parents[1]
ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
MODE = "production"
WORKSPACE = "default"
CONTROLLER_INSTANCE_ID = "i-0511af4e31aa5406a"
CONTROLLER_INSTANCE_TYPE = "c6a.xlarge"
CONTROLLER_CLUSTER_NAME = "sky-jobs-controller-production-a1"
BASELINE_DIGEST_FIELD = "baseline_body_sha256"
READY_DIGEST_FIELD = "control_plane_ready_body_sha256"
ACQUISITION_DIGEST_FIELD = "acquisition_body_sha256"
TERMINAL_STATUS = "CANCELLED"

BASELINE_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "bucket",
    "run_id",
    "managed_mode",
    "campaign_identity_sha256",
    "descriptor_key",
    "descriptor_file_sha256",
    "descriptor_body_sha256",
    "descriptor_version_id",
    "intent_key",
    "intent_file_sha256",
    "intent_body_sha256",
    "intent_version_id",
    "sky_job_name",
    "workspace",
    "controller_instance_id",
    "controller_instance_type",
    "controller_identity",
    "controller_profile_arn",
    "controller_cluster_name",
    "ssm_ping_status",
    "exact_name_history",
    "active_exact_name_job_ids",
    "active_tagged_p5_instance_ids",
    "observed_at",
    BASELINE_DIGEST_FIELD,
}
READY_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "bucket",
    "run_id",
    "managed_mode",
    "campaign_identity_sha256",
    "descriptor_key",
    "descriptor_file_sha256",
    "descriptor_body_sha256",
    "descriptor_version_id",
    "intent_key",
    "intent_file_sha256",
    "intent_body_sha256",
    "intent_version_id",
    "controller_baseline_key",
    "controller_baseline_file_sha256",
    "controller_baseline_body_sha256",
    "controller_baseline_version_id",
    "stack_id",
    "template_sha256",
    "lambda_function_arn",
    "lambda_code_s3_key",
    "lambda_code_sha256",
    "lambda_code_version_id",
    "iam_policy_sha256",
    "reconciliation_rule_arn",
    "reconciliation_rule_state",
    "deadline_schedule_arn",
    "deadline_schedule_state",
    "dlq_arn",
    "coordinator_mode",
    "activation_capabilities",
    "observed_at",
    READY_DIGEST_FIELD,
}
ACQUISITION_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "bucket",
    "run_id",
    "managed_mode",
    "campaign_identity_sha256",
    "descriptor_key",
    "descriptor_file_sha256",
    "descriptor_body_sha256",
    "descriptor_version_id",
    "intent_key",
    "intent_file_sha256",
    "intent_body_sha256",
    "intent_version_id",
    "controller_baseline_key",
    "controller_baseline_file_sha256",
    "controller_baseline_body_sha256",
    "controller_baseline_version_id",
    "must_start_control_plane_ready_key",
    "must_start_control_plane_ready_file_sha256",
    "must_start_control_plane_ready_body_sha256",
    "must_start_control_plane_ready_version_id",
    "sky_job_name",
    "must_start_by",
    "spend_snapshot_observed_at",
    "controller_baseline_observed_at",
    "control_plane_ready_observed_at",
    "acquired_at",
    ACQUISITION_DIGEST_FIELD,
}
DEPLOYMENT_IDENTITY_FIELDS = {
    "stack_id",
    "template_sha256",
    "lambda_function_arn",
    "lambda_code_s3_key",
    "lambda_code_sha256",
    "lambda_code_version_id",
    "iam_policy_sha256",
    "reconciliation_rule_arn",
    "deadline_schedule_arn",
    "dlq_arn",
    "activation_capabilities",
}
ACTIVATION_CAPABILITY_FIELDS = {
    "stack_status",
    "stack_operation_in_progress",
    "foundation_revision",
    "managed_mode",
    "run_id",
    "bucket",
    "descriptor_key",
    "descriptor_file_sha256",
    "descriptor_body_sha256",
    "descriptor_version_id",
    "intent_key",
    "intent_file_sha256",
    "intent_body_sha256",
    "intent_version_id",
    "sky_job_name",
    "must_start_by",
    "primary_wake_at",
    "lambda_state",
    "lambda_last_update_status",
    "lambda_qualified_arn",
    "lambda_execution_role_arn",
    "lambda_code_sha256",
    "lambda_code_version_id",
    "lambda_environment",
    "reconciliation_schedule_expression",
    "reconciliation_target_arn",
    "reconciliation_target_id",
    "reconciliation_input",
    "reconciliation_retry_max_event_age_seconds",
    "reconciliation_retry_max_attempts",
    "reconciliation_dlq_arn",
    "deadline_schedule_expression",
    "deadline_schedule_timezone",
    "deadline_flexible_window_mode",
    "deadline_target_arn",
    "deadline_target_role_arn",
    "deadline_input",
    "deadline_retry_max_event_age_seconds",
    "deadline_retry_max_attempts",
    "deadline_dlq_arn",
}
ENVIRONMENT_FIELDS = {
    "EXPECTED_ACCOUNT_ID",
    "CAMPAIGN_BUCKET",
    "CAMPAIGN_RUN_ID",
    "MANAGED_MODE",
    "CAMPAIGN_DESCRIPTOR_KEY",
    "CAMPAIGN_DESCRIPTOR_VERSION_ID",
    "IMMUTABLE_SUBMISSION_KEY",
    "IMMUTABLE_SUBMISSION_VERSION_ID",
    "SUBMISSION_BODY_SHA256",
    "SKY_JOB_NAME",
    "MUST_START_BY",
    "COORDINATOR_MODE",
}
TRIGGER_FIELDS = {
    "campaign_run_id",
    "managed_mode",
    "sky_job_name",
    "must_start_by",
    "descriptor_key",
    "descriptor_version_id",
    "intent_key",
    "intent_version_id",
    "trigger",
}

EXPECTED_PUBLIC_INTERFACE = [
    "ProductionPrelaunchAuthorityError",
    "build_production_controller_baseline",
    "production_controller_baseline_file_bytes",
    "production_controller_baseline_file_sha256",
    "production_controller_baseline_s3_key",
    "validate_production_controller_baseline",
    "build_production_must_start_control_plane_ready",
    "production_must_start_control_plane_ready_file_bytes",
    "production_must_start_control_plane_ready_file_sha256",
    "production_must_start_control_plane_ready_s3_key",
    "validate_production_must_start_control_plane_ready",
    "build_production_submission_acquired",
    "production_submission_acquired_file_bytes",
    "production_submission_acquired_file_sha256",
    "production_submission_acquired_s3_key",
    "validate_production_submission_acquired",
]


def _module() -> Any:
    try:
        return importlib.import_module(
            "mlx_vq.quality.glm52_sky_production_acquisition"
        )
    except ModuleNotFoundError:
        pytest.fail("production prelaunch authority module is not implemented")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _file_bytes(value: object) -> bytes:
    return _canonical(value) + b"\n"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _rehash(
    value: dict[str, object],
    digest_field: str,
) -> dict[str, object]:
    body = copy.deepcopy(value)
    body.pop(digest_field, None)
    return {**body, digest_field: _sha(_canonical(body))}


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_time(value: object) -> datetime:
    assert type(value) is str and value.endswith("Z")
    return datetime.fromisoformat(f"{value[:-1]}+00:00")


def _load_production_fixture_module() -> Any:
    path = ROOT / "tests/test_glm52_sky_production_submission.py"
    specification = importlib.util.spec_from_file_location(
        "_task3ph1d_production_fixture",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def source_authorities(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, object]:
    production_tests = _load_production_fixture_module()
    source_chain = production_tests.source_chain.__wrapped__(tmp_path_factory)
    intent = build_production_submission_intent(
        **production_tests._kwargs(source_chain)
    )
    descriptor_artifact = source_chain["artifacts"]["descriptor"]
    assert type(descriptor_artifact) is VersionedJsonArtifact
    intent_artifact = VersionedJsonArtifact(
        key=production_submission_intent_s3_key(
            run_id=str(intent["run_id"]),
            intent_body_sha256=str(intent["intent_body_sha256"]),
        ),
        raw=production_submission_intent_file_bytes(intent),
        version_id="production-intent-version-1",
    )
    return {
        "descriptor": descriptor_artifact,
        "intent": intent_artifact,
        "descriptor_record": json.loads(descriptor_artifact.raw),
        "intent_record": intent,
    }


def _profile_arn(descriptor: dict[str, object]) -> str:
    role = str(descriptor["controller_identity"])
    return role.replace(":role/", ":instance-profile/", 1)


def _baseline_kwargs(
    authorities: dict[str, object],
    *,
    observed_at: datetime | None = None,
) -> dict[str, object]:
    descriptor = authorities["descriptor_record"]
    intent = authorities["intent_record"]
    intent_at = _parse_time(intent["intent_at"])
    baseline_time = observed_at or intent_at + timedelta(seconds=10)
    return {
        "descriptor": authorities["descriptor"],
        "intent": authorities["intent"],
        "controller_instance_id": CONTROLLER_INSTANCE_ID,
        "controller_instance_type": CONTROLLER_INSTANCE_TYPE,
        "controller_profile_arn": _profile_arn(descriptor),
        "controller_cluster_name": CONTROLLER_CLUSTER_NAME,
        "ssm_ping_status": "Online",
        "exact_name_history": [
            {
                "sky_job_id": 7,
                "sky_job_name": intent["sky_job_name"],
                "workspace": WORKSPACE,
                "controller_submitted_at": intent["intent_at"],
                "controller_status": TERMINAL_STATUS,
                "controller_identity": descriptor["controller_identity"],
            }
        ],
        "active_exact_name_job_ids": [],
        "active_tagged_p5_instance_ids": [],
        "observed_at": baseline_time,
    }


def _artifact(
    *,
    key: str,
    value: dict[str, object],
    version_id: str,
) -> VersionedJsonArtifact:
    return VersionedJsonArtifact(
        key=key,
        raw=_file_bytes(value),
        version_id=version_id,
    )


def _baseline_artifact(module: Any, baseline: dict[str, object]) -> VersionedJsonArtifact:
    return _artifact(
        key=module.production_controller_baseline_s3_key(
            run_id=str(baseline["run_id"]),
            baseline_body_sha256=str(baseline[BASELINE_DIGEST_FIELD]),
        ),
        value=baseline,
        version_id="production-baseline-version-1",
    )


def _deployment_identity(
    authorities: dict[str, object],
) -> dict[str, object]:
    descriptor_artifact = authorities["descriptor"]
    intent_artifact = authorities["intent"]
    descriptor = authorities["descriptor_record"]
    intent = authorities["intent_record"]
    run_id = str(intent["run_id"])
    bucket = str(intent["bucket"])
    descriptor_key = str(descriptor_artifact.key)
    descriptor_file_sha = _sha(descriptor_artifact.raw)
    intent_key = str(intent_artifact.key)
    intent_file_sha = _sha(intent_artifact.raw)
    template_sha = "d" * 64
    lambda_code_sha = "e" * 64
    lambda_code_version = "lambda-code-version-1"
    lambda_arn = (
        "arn:aws:lambda:us-west-2:246813579024:function:"
        "keep-glm52-sky-production-must-start"
    )
    lambda_qualified_arn = f"{lambda_arn}:7"
    dlq_arn = (
        "arn:aws:sqs:us-west-2:246813579024:"
        "keep-glm52-sky-production-must-start-dlq"
    )
    shared_input = {
        "campaign_run_id": run_id,
        "managed_mode": MODE,
        "sky_job_name": intent["sky_job_name"],
        "must_start_by": intent["must_start_by"],
        "descriptor_key": descriptor_key,
        "descriptor_version_id": descriptor_artifact.version_id,
        "intent_key": intent_key,
        "intent_version_id": intent_artifact.version_id,
    }
    activation_capabilities: dict[str, object] = {
        "stack_status": "UPDATE_COMPLETE",
        "stack_operation_in_progress": False,
        "foundation_revision": template_sha,
        "managed_mode": MODE,
        "run_id": run_id,
        "bucket": bucket,
        "descriptor_key": descriptor_key,
        "descriptor_file_sha256": descriptor_file_sha,
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "descriptor_version_id": descriptor_artifact.version_id,
        "intent_key": intent_key,
        "intent_file_sha256": intent_file_sha,
        "intent_body_sha256": intent["intent_body_sha256"],
        "intent_version_id": intent_artifact.version_id,
        "sky_job_name": intent["sky_job_name"],
        "must_start_by": intent["must_start_by"],
        "primary_wake_at": intent["must_start_by"],
        "lambda_state": "Active",
        "lambda_last_update_status": "Successful",
        "lambda_qualified_arn": lambda_qualified_arn,
        "lambda_execution_role_arn": (
            "arn:aws:iam::246813579024:role/"
            "keep-glm52-sky-production-must-start"
        ),
        "lambda_code_sha256": lambda_code_sha,
        "lambda_code_version_id": lambda_code_version,
        "lambda_environment": {
            "EXPECTED_ACCOUNT_ID": ACCOUNT_ID,
            "CAMPAIGN_BUCKET": bucket,
            "CAMPAIGN_RUN_ID": run_id,
            "MANAGED_MODE": MODE,
            "CAMPAIGN_DESCRIPTOR_KEY": descriptor_key,
            "CAMPAIGN_DESCRIPTOR_VERSION_ID": descriptor_artifact.version_id,
            "IMMUTABLE_SUBMISSION_KEY": intent_key,
            "IMMUTABLE_SUBMISSION_VERSION_ID": intent_artifact.version_id,
            "SUBMISSION_BODY_SHA256": intent["intent_body_sha256"],
            "SKY_JOB_NAME": intent["sky_job_name"],
            "MUST_START_BY": intent["must_start_by"],
            "COORDINATOR_MODE": "production-dynamic-job-binding-active",
        },
        "reconciliation_schedule_expression": "rate(1 minute)",
        "reconciliation_target_arn": lambda_qualified_arn,
        "reconciliation_target_id": "sky-production-must-start-reconcile",
        "reconciliation_input": {
            **shared_input,
            "trigger": "reconcile",
        },
        "reconciliation_retry_max_event_age_seconds": 300,
        "reconciliation_retry_max_attempts": 2,
        "reconciliation_dlq_arn": dlq_arn,
        "deadline_schedule_expression": (
            f"at({str(intent['must_start_by'])[:-1]})"
        ),
        "deadline_schedule_timezone": "UTC",
        "deadline_flexible_window_mode": "OFF",
        "deadline_target_arn": lambda_qualified_arn,
        "deadline_target_role_arn": (
            "arn:aws:iam::246813579024:role/"
            "keep-glm52-sky-production-deadline-scheduler"
        ),
        "deadline_input": {
            **shared_input,
            "trigger": "primary-deadline",
        },
        "deadline_retry_max_event_age_seconds": 300,
        "deadline_retry_max_attempts": 2,
        "deadline_dlq_arn": dlq_arn,
    }
    assert set(activation_capabilities) == ACTIVATION_CAPABILITY_FIELDS
    assert set(activation_capabilities["lambda_environment"]) == ENVIRONMENT_FIELDS
    assert set(activation_capabilities["reconciliation_input"]) == TRIGGER_FIELDS
    assert set(activation_capabilities["deadline_input"]) == TRIGGER_FIELDS
    deployment = {
        "stack_id": (
            "arn:aws:cloudformation:us-west-2:246813579024:stack/"
            "keep-glm52-sky-production-control-plane/"
            "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        ),
        "template_sha256": template_sha,
        "lambda_function_arn": lambda_arn,
        "lambda_code_s3_key": (
            f"campaigns/{run_id}/control-plane/production/lambda-code/"
            f"{lambda_code_sha}/sky-must-start.zip"
        ),
        "lambda_code_sha256": lambda_code_sha,
        "lambda_code_version_id": lambda_code_version,
        "iam_policy_sha256": "f" * 64,
        "reconciliation_rule_arn": (
            "arn:aws:events:us-west-2:246813579024:rule/"
            "keep-glm52-sky-production-reconcile"
        ),
        "deadline_schedule_arn": (
            "arn:aws:scheduler:us-west-2:246813579024:schedule/default/"
            "keep-glm52-sky-production-deadline"
        ),
        "dlq_arn": dlq_arn,
        "activation_capabilities": activation_capabilities,
    }
    assert set(deployment) == DEPLOYMENT_IDENTITY_FIELDS
    return deployment


def _ready_kwargs(
    module: Any,
    authorities: dict[str, object],
    baseline: dict[str, object],
    *,
    observed_at: datetime | None = None,
) -> dict[str, object]:
    baseline_time = _parse_time(baseline["observed_at"])
    deployment = _deployment_identity(authorities)
    return {
        "descriptor": authorities["descriptor"],
        "intent": authorities["intent"],
        "controller_baseline": _baseline_artifact(module, baseline),
        "reviewed_deployment_identity": deployment,
        "observed_deployment_identity": copy.deepcopy(deployment),
        "reconciliation_rule_state": "ENABLED",
        "deadline_schedule_state": "ENABLED",
        "coordinator_mode": "production-dynamic-job-binding-active",
        "observed_at": observed_at or baseline_time + timedelta(seconds=10),
    }


def _ready_artifact(module: Any, ready: dict[str, object]) -> VersionedJsonArtifact:
    return _artifact(
        key=module.production_must_start_control_plane_ready_s3_key(
            run_id=str(ready["run_id"]),
            intent_body_sha256=str(ready["intent_body_sha256"]),
            control_plane_ready_body_sha256=str(ready[READY_DIGEST_FIELD]),
        ),
        value=ready,
        version_id="production-control-plane-ready-version-1",
    )


def _full_chain(
    module: Any,
    authorities: dict[str, object],
    *,
    baseline_at: datetime | None = None,
    ready_at: datetime | None = None,
    acquired_at: datetime | None = None,
) -> dict[str, object]:
    baseline = module.build_production_controller_baseline(
        **_baseline_kwargs(authorities, observed_at=baseline_at)
    )
    ready = module.build_production_must_start_control_plane_ready(
        **_ready_kwargs(
            module,
            authorities,
            baseline,
            observed_at=ready_at,
        )
    )
    acquired_time = acquired_at or _parse_time(ready["observed_at"]) + timedelta(
        seconds=10
    )
    acquisition = module.build_production_submission_acquired(
        descriptor=authorities["descriptor"],
        intent=authorities["intent"],
        controller_baseline=_baseline_artifact(module, baseline),
        must_start_control_plane_ready=_ready_artifact(module, ready),
        acquired_at=acquired_time,
    )
    return {
        "baseline": baseline,
        "ready": ready,
        "acquisition": acquisition,
        "baseline_artifact": _baseline_artifact(module, baseline),
        "ready_artifact": _ready_artifact(module, ready),
        "acquired_at": acquired_time,
    }


def _artifact_with_raw(
    artifact: VersionedJsonArtifact,
    *,
    raw: bytes,
    key: str | None = None,
    version_id: str | None = None,
) -> VersionedJsonArtifact:
    return VersionedJsonArtifact(
        key=artifact.key if key is None else key,
        raw=raw,
        version_id=artifact.version_id if version_id is None else version_id,
    )


def _descriptor_artifact_from_record(
    artifact: VersionedJsonArtifact,
    record: dict[str, object],
) -> VersionedJsonArtifact:
    mutated = copy.deepcopy(record)
    identity = {
        key: value
        for key, value in mutated.items()
        if key
        not in {
            "campaign_identity_sha256",
            "descriptor_body_sha256",
            "must_start_by",
            "campaign_descriptor_key",
        }
    }
    mutated["campaign_identity_sha256"] = _sha(_canonical(identity))
    mutated = _rehash(mutated, "descriptor_body_sha256")
    return VersionedJsonArtifact(
        key=str(mutated["campaign_descriptor_key"]),
        raw=_file_bytes(mutated),
        version_id=artifact.version_id,
    )


def _intent_artifact_from_record(
    record: dict[str, object],
    *,
    version_id: str = "production-intent-version-mutated",
) -> VersionedJsonArtifact:
    mutated = _rehash(record, "intent_body_sha256")
    return VersionedJsonArtifact(
        key=production_submission_intent_s3_key(
            run_id=str(mutated["run_id"]),
            intent_body_sha256=str(mutated["intent_body_sha256"]),
        ),
        raw=production_submission_intent_file_bytes(mutated),
        version_id=version_id,
    )


def _authorities_with_intent(
    authorities: dict[str, object],
    intent_artifact: VersionedJsonArtifact,
) -> dict[str, object]:
    return {
        **authorities,
        "intent": intent_artifact,
        "intent_record": json.loads(intent_artifact.raw),
    }


class StringSubclass(str):
    """A JSON-compatible string subclass that must fail exact-type gates."""


class BytesSubclass(bytes):
    """A bytes subclass that must fail exact raw-byte gates."""


class DictSubclass(dict[str, object]):
    """A mapping subclass that must fail exact durable-record gates."""


class ArtifactSubclass(VersionedJsonArtifact):
    """A provenance subclass that must not inherit source authority."""


class PseudoAwareTimezone(tzinfo):
    """A tzinfo marker that is not actually offset-aware."""

    def utcoffset(self, value: datetime | None) -> None:
        return None

    def dst(self, value: datetime | None) -> None:
        return None


class ExplodingOffsetTimezone(tzinfo):
    """A tzinfo whose offset lookup must fail through the owned error."""

    def utcoffset(self, value: datetime | None) -> timedelta | None:
        raise RuntimeError("utcoffset lookup exploded")

    def dst(self, value: datetime | None) -> timedelta | None:
        return timedelta(0)


class SingleUseOffsetTimezone(tzinfo):
    """A stateful zone that grants exactly one authoritative offset lookup."""

    def __init__(self, offset: timedelta) -> None:
        self.offset = offset
        self.utcoffset_calls = 0

    def utcoffset(self, value: datetime | None) -> timedelta | None:
        self.utcoffset_calls += 1
        if self.utcoffset_calls != 1:
            raise RuntimeError("UTC offset was queried more than once")
        return self.offset

    def dst(self, value: datetime | None) -> timedelta | None:
        return timedelta(0)


class FractionalOffsetTimezone(tzinfo):
    """A valid offset that maps a whole wall second to a fractional UTC instant."""

    def __init__(self) -> None:
        self.utcoffset_calls = 0

    def utcoffset(self, value: datetime | None) -> timedelta | None:
        self.utcoffset_calls += 1
        return timedelta(hours=2, microseconds=1)

    def dst(self, value: datetime | None) -> timedelta | None:
        return timedelta(0)


def _datetime_path_value(
    chain: dict[str, object],
    public_path: str,
) -> datetime:
    if public_path == "baseline-observed-at":
        return _parse_time(chain["baseline"]["observed_at"])
    if public_path == "readiness-observed-at":
        return _parse_time(chain["ready"]["observed_at"])
    if public_path in {"acquisition-acquired-at", "acquisition-validation-now"}:
        return _parse_time(chain["acquisition"]["acquired_at"])
    raise AssertionError(f"unknown datetime public path: {public_path}")


def _call_datetime_public_path(
    module: Any,
    authorities: dict[str, object],
    chain: dict[str, object],
    *,
    public_path: str,
    value: datetime,
) -> dict[str, object]:
    if public_path == "baseline-observed-at":
        kwargs = _baseline_kwargs(authorities)
        kwargs["observed_at"] = value
        return module.build_production_controller_baseline(**kwargs)
    if public_path == "readiness-observed-at":
        kwargs = _ready_kwargs(module, authorities, chain["baseline"])
        kwargs["observed_at"] = value
        return module.build_production_must_start_control_plane_ready(**kwargs)
    if public_path == "acquisition-acquired-at":
        return module.build_production_submission_acquired(
            descriptor=authorities["descriptor"],
            intent=authorities["intent"],
            controller_baseline=chain["baseline_artifact"],
            must_start_control_plane_ready=chain["ready_artifact"],
            acquired_at=value,
        )
    if public_path == "acquisition-validation-now":
        return module.validate_production_submission_acquired(
            chain["acquisition"],
            descriptor=authorities["descriptor"],
            intent=authorities["intent"],
            controller_baseline=chain["baseline_artifact"],
            must_start_control_plane_ready=chain["ready_artifact"],
            now=value,
        )
    raise AssertionError(f"unknown datetime public path: {public_path}")


def test_public_interface_is_exact_and_exposes_no_io_or_launch_mutant() -> None:
    module = _module()
    assert module.__all__ == EXPECTED_PUBLIC_INTERFACE
    forbidden = {
        "acquire",
        "conditional_put",
        "get_object",
        "launch",
        "publish",
        "put_object",
        "submit",
    }
    assert forbidden.isdisjoint(module.__all__)
    assert all(not hasattr(module, name) for name in forbidden)


def test_every_module_defined_unprefixed_class_or_function_is_in_frozen_all() -> None:
    module = _module()
    defined_unprefixed = {
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and (inspect.isclass(value) or inspect.isfunction(value))
        and getattr(value, "__module__", None) == module.__name__
    }
    assert defined_unprefixed == set(EXPECTED_PUBLIC_INTERFACE)


def test_exact_raw_bytes_are_hashed_before_canonical_json_is_parsed(
    source_authorities: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    descriptor = source_authorities["descriptor"]
    intent = source_authorities["intent"]
    events: list[tuple[str, str]] = []
    original_sha = module._sha
    original_parse = module._parse_raw

    def label_for(raw: bytes) -> str:
        if raw == descriptor.raw:
            return "descriptor"
        if raw == intent.raw:
            return "intent"
        return "other"

    def observed_sha(raw: bytes) -> str:
        events.append(("hash", label_for(raw)))
        return original_sha(raw)

    def observed_parse(raw: bytes, *, label: str) -> dict[str, object]:
        events.append(("parse", label_for(raw)))
        return original_parse(raw, label=label)

    monkeypatch.setattr(module, "_sha", observed_sha)
    monkeypatch.setattr(module, "_parse_raw", observed_parse)
    module.build_production_controller_baseline(
        **_baseline_kwargs(source_authorities)
    )
    for source_name in ("descriptor", "intent"):
        assert events.index(("hash", source_name)) < events.index(
            ("parse", source_name)
        )


@pytest.mark.parametrize(
    "public_path",
    [
        "baseline-observed-at",
        "readiness-observed-at",
        "acquisition-acquired-at",
        "acquisition-validation-now",
    ],
)
@pytest.mark.parametrize("ambient_timezone", ["UTC", "America/Los_Angeles"])
def test_public_datetime_paths_reject_pseudo_aware_values_without_ambient_conversion(
    source_authorities: dict[str, object],
    public_path: str,
    ambient_timezone: str,
) -> None:
    module = _module()
    chain = _full_chain(module, source_authorities)
    pseudo_aware = _datetime_path_value(chain, public_path).replace(
        tzinfo=PseudoAwareTimezone()
    )
    previous_timezone = os.environ.get("TZ")
    try:
        os.environ["TZ"] = ambient_timezone
        time.tzset()
        with pytest.raises(module.ProductionPrelaunchAuthorityError):
            _call_datetime_public_path(
                module,
                source_authorities,
                chain,
                public_path=public_path,
                value=pseudo_aware,
            )
    finally:
        if previous_timezone is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous_timezone
        time.tzset()


@pytest.mark.parametrize(
    "public_path",
    [
        "baseline-observed-at",
        "readiness-observed-at",
        "acquisition-acquired-at",
        "acquisition-validation-now",
    ],
)
def test_public_datetime_paths_translate_utcoffset_exceptions(
    source_authorities: dict[str, object],
    public_path: str,
) -> None:
    module = _module()
    chain = _full_chain(module, source_authorities)
    exploding = _datetime_path_value(chain, public_path).replace(
        tzinfo=ExplodingOffsetTimezone()
    )
    with pytest.raises(module.ProductionPrelaunchAuthorityError) as captured:
        _call_datetime_public_path(
            module,
            source_authorities,
            chain,
            public_path=public_path,
            value=exploding,
        )
    assert isinstance(captured.value.__cause__, RuntimeError)


@pytest.mark.parametrize(
    "public_path",
    [
        "baseline-observed-at",
        "readiness-observed-at",
        "acquisition-acquired-at",
        "acquisition-validation-now",
    ],
)
@pytest.mark.parametrize("ambient_timezone", ["UTC", "America/Los_Angeles"])
def test_public_datetime_paths_query_stateful_offset_once_and_normalize_deterministically(
    source_authorities: dict[str, object],
    public_path: str,
    ambient_timezone: str,
) -> None:
    module = _module()
    chain = _full_chain(module, source_authorities)
    expected_utc = _datetime_path_value(chain, public_path)
    offset = timedelta(hours=2)
    stateful_timezone = SingleUseOffsetTimezone(offset)
    wall_time = (expected_utc.replace(tzinfo=None) + offset).replace(
        tzinfo=stateful_timezone
    )
    previous_timezone = os.environ.get("TZ")
    try:
        os.environ["TZ"] = ambient_timezone
        time.tzset()
        result = _call_datetime_public_path(
            module,
            source_authorities,
            chain,
            public_path=public_path,
            value=wall_time,
        )
    finally:
        if previous_timezone is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous_timezone
        time.tzset()

    assert stateful_timezone.utcoffset_calls == 1
    if public_path == "baseline-observed-at":
        assert result["observed_at"] == chain["baseline"]["observed_at"]
    elif public_path == "readiness-observed-at":
        assert result["observed_at"] == chain["ready"]["observed_at"]
    else:
        assert result["acquired_at"] == chain["acquisition"]["acquired_at"]


@pytest.mark.parametrize(
    "public_path",
    [
        "baseline-observed-at",
        "readiness-observed-at",
        "acquisition-acquired-at",
        "acquisition-validation-now",
    ],
)
@pytest.mark.parametrize("ambient_timezone", ["UTC", "America/Los_Angeles"])
def test_public_datetime_paths_reject_fractional_normalized_utc_at_exact_guard(
    source_authorities: dict[str, object],
    public_path: str,
    ambient_timezone: str,
) -> None:
    module = _module()
    chain = _full_chain(module, source_authorities)
    expected_utc = _datetime_path_value(chain, public_path)
    fractional_timezone = FractionalOffsetTimezone()
    wall_time = (
        expected_utc.replace(tzinfo=None) + timedelta(hours=2)
    ).replace(tzinfo=fractional_timezone)
    previous_timezone = os.environ.get("TZ")
    try:
        os.environ["TZ"] = ambient_timezone
        time.tzset()
        with pytest.raises(
            module.ProductionPrelaunchAuthorityError,
            match="normalized UTC instant must be whole-second",
        ):
            _call_datetime_public_path(
                module,
                source_authorities,
                chain,
                public_path=public_path,
                value=wall_time,
            )
    finally:
        if previous_timezone is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous_timezone
        time.tzset()

    assert fractional_timezone.utcoffset_calls == 1


def test_all_three_records_round_trip_exact_schemas_keys_and_bytes(
    source_authorities: dict[str, object],
) -> None:
    module = _module()
    chain = _full_chain(module, source_authorities)
    baseline = chain["baseline"]
    ready = chain["ready"]
    acquisition = chain["acquisition"]

    assert set(baseline) == BASELINE_FIELDS
    assert set(ready) == READY_FIELDS
    assert set(acquisition) == ACQUISITION_FIELDS
    assert set(ready["activation_capabilities"]) == ACTIVATION_CAPABILITY_FIELDS
    assert baseline["record_type"] == "glm52_production_controller_baseline_v1"
    assert (
        ready["record_type"]
        == "glm52_production_must_start_control_plane_ready_v1"
    )
    assert acquisition["record_type"] == "glm52_sky_production_submission_acquired_v1"
    assert baseline["managed_mode"] == ready["managed_mode"] == MODE
    assert acquisition["managed_mode"] == MODE

    assert (
        module.validate_production_controller_baseline(
            baseline,
            descriptor=source_authorities["descriptor"],
            intent=source_authorities["intent"],
        )
        == baseline
    )
    assert (
        module.validate_production_must_start_control_plane_ready(
            ready,
            descriptor=source_authorities["descriptor"],
            intent=source_authorities["intent"],
            controller_baseline=chain["baseline_artifact"],
        )
        == ready
    )
    assert (
        module.validate_production_submission_acquired(
            acquisition,
            descriptor=source_authorities["descriptor"],
            intent=source_authorities["intent"],
            controller_baseline=chain["baseline_artifact"],
            must_start_control_plane_ready=chain["ready_artifact"],
            now=acquisition["acquired_at"],
        )
        == acquisition
    )

    for value, bytes_helper, sha_helper in (
        (
            baseline,
            module.production_controller_baseline_file_bytes,
            module.production_controller_baseline_file_sha256,
        ),
        (
            ready,
            module.production_must_start_control_plane_ready_file_bytes,
            module.production_must_start_control_plane_ready_file_sha256,
        ),
        (
            acquisition,
            module.production_submission_acquired_file_bytes,
            module.production_submission_acquired_file_sha256,
        ),
    ):
        assert bytes_helper(value) == _file_bytes(value)
        assert sha_helper(value) == _sha(_file_bytes(value))
        assert bytes_helper(value).endswith(b"\n")
        assert not bytes_helper(value).endswith(b"\n\n")

    assert module.production_controller_baseline_s3_key(
        run_id=str(baseline["run_id"]),
        baseline_body_sha256=str(baseline[BASELINE_DIGEST_FIELD]),
    ) == (
        f"campaigns/{baseline['run_id']}/production/controller-baselines/"
        f"{baseline[BASELINE_DIGEST_FIELD]}/CONTROLLER_BASELINE.json"
    )
    assert module.production_must_start_control_plane_ready_s3_key(
        run_id=str(ready["run_id"]),
        intent_body_sha256=str(ready["intent_body_sha256"]),
        control_plane_ready_body_sha256=str(ready[READY_DIGEST_FIELD]),
    ) == (
        f"campaigns/{ready['run_id']}/monitor/must-start/production/"
        f"{ready['intent_body_sha256']}/control-plane-ready/"
        f"{ready[READY_DIGEST_FIELD]}/CONTROL_PLANE_READY.json"
    )
    assert module.production_submission_acquired_s3_key(
        run_id=str(acquisition["run_id"]),
        descriptor_file_sha256=str(acquisition["descriptor_file_sha256"]),
    ) == (
        f"campaigns/{acquisition['run_id']}/submissions/production/acquisitions/"
        f"{acquisition['descriptor_file_sha256']}/SUBMISSION_ACQUIRED.json"
    )


@pytest.mark.parametrize("record_kind", ["baseline", "readiness", "acquisition"])
def test_public_validators_return_deep_json_copies_in_both_mutation_directions(
    source_authorities: dict[str, object],
    record_kind: str,
) -> None:
    module = _module()
    chain = _full_chain(module, source_authorities)
    original = copy.deepcopy(chain[record_kind if record_kind != "readiness" else "ready"])

    if record_kind == "baseline":
        def validator(candidate: dict[str, object]) -> dict[str, object]:
            return module.validate_production_controller_baseline(
                candidate,
                descriptor=source_authorities["descriptor"],
                intent=source_authorities["intent"],
            )

        file_bytes = module.production_controller_baseline_file_bytes

        def mutate(record: dict[str, object]) -> None:
            record["exact_name_history"][0]["controller_status"] = "FAILED"

    elif record_kind == "readiness":
        def validator(candidate: dict[str, object]) -> dict[str, object]:
            return module.validate_production_must_start_control_plane_ready(
                candidate,
                descriptor=source_authorities["descriptor"],
                intent=source_authorities["intent"],
                controller_baseline=chain["baseline_artifact"],
            )

        file_bytes = module.production_must_start_control_plane_ready_file_bytes

        def mutate(record: dict[str, object]) -> None:
            record["activation_capabilities"]["lambda_environment"][
                "EXPECTED_ACCOUNT_ID"
            ] = "135792468013"

    else:
        def validator(candidate: dict[str, object]) -> dict[str, object]:
            return module.validate_production_submission_acquired(
                candidate,
                descriptor=source_authorities["descriptor"],
                intent=source_authorities["intent"],
                controller_baseline=chain["baseline_artifact"],
                must_start_control_plane_ready=chain["ready_artifact"],
                now=candidate["acquired_at"],
            )

        file_bytes = module.production_submission_acquired_file_bytes

        def mutate(record: dict[str, object]) -> None:
            record["sky_job_name"] = f"{record['sky_job_name']}-mutated"

    caller_candidate = copy.deepcopy(original)
    validated = validator(caller_candidate)
    validated_snapshot = copy.deepcopy(validated)
    assert validated == original
    assert validated is not caller_candidate
    assert file_bytes(validated) == _file_bytes(original)

    mutate(caller_candidate)
    assert validated == validated_snapshot
    assert file_bytes(validated) == _file_bytes(validated_snapshot)

    second_candidate = copy.deepcopy(original)
    second_candidate_snapshot = copy.deepcopy(second_candidate)
    second_validated = validator(second_candidate)
    mutate(second_validated)
    assert second_candidate == second_candidate_snapshot
    assert file_bytes(second_candidate) == _file_bytes(second_candidate_snapshot)


def test_builders_are_deterministic_and_acquisition_is_descriptor_singleton_only(
    source_authorities: dict[str, object],
) -> None:
    module = _module()
    first = _full_chain(module, source_authorities)
    second = _full_chain(module, source_authorities)
    assert first["baseline"] == second["baseline"]
    assert first["ready"] == second["ready"]
    assert first["acquisition"] == second["acquisition"]

    altered_baseline_kwargs = _baseline_kwargs(source_authorities)
    altered_baseline_kwargs["exact_name_history"] = [
        {
            **altered_baseline_kwargs["exact_name_history"][0],
            "sky_job_id": 8,
        }
    ]
    altered_baseline = module.build_production_controller_baseline(
        **altered_baseline_kwargs
    )
    altered_ready = module.build_production_must_start_control_plane_ready(
        **_ready_kwargs(module, source_authorities, altered_baseline)
    )
    altered = module.build_production_submission_acquired(
        descriptor=source_authorities["descriptor"],
        intent=source_authorities["intent"],
        controller_baseline=_baseline_artifact(module, altered_baseline),
        must_start_control_plane_ready=_ready_artifact(module, altered_ready),
        acquired_at=first["acquired_at"],
    )
    first_key = module.production_submission_acquired_s3_key(
        run_id=str(first["acquisition"]["run_id"]),
        descriptor_file_sha256=str(
            first["acquisition"]["descriptor_file_sha256"]
        ),
    )
    altered_key = module.production_submission_acquired_s3_key(
        run_id=str(altered["run_id"]),
        descriptor_file_sha256=str(altered["descriptor_file_sha256"]),
    )
    assert altered != first["acquisition"]
    assert altered_key == first_key

    other_descriptor_key = module.production_submission_acquired_s3_key(
        run_id=str(altered["run_id"]),
        descriptor_file_sha256="a" * 64,
    )
    assert other_descriptor_key != first_key
    assert (
        other_descriptor_key
        == f"campaigns/{altered['run_id']}/submissions/production/acquisitions/"
        f"{'a' * 64}/SUBMISSION_ACQUIRED.json"
    )


def test_standalone_readiness_observation_is_strictly_before_deadline(
    source_authorities: dict[str, object],
) -> None:
    module = _module()
    ready = copy.deepcopy(_full_chain(module, source_authorities)["ready"])
    deadline = _parse_time(
        ready["activation_capabilities"]["must_start_by"]
    )

    just_before = copy.deepcopy(ready)
    just_before["observed_at"] = _iso(deadline - timedelta(seconds=1))
    just_before = _rehash(just_before, READY_DIGEST_FIELD)
    assert (
        module.production_must_start_control_plane_ready_file_bytes(just_before)
        == _file_bytes(just_before)
    )

    for observed_at in (deadline, deadline + timedelta(seconds=1)):
        expired = copy.deepcopy(ready)
        expired["observed_at"] = _iso(observed_at)
        expired = _rehash(expired, READY_DIGEST_FIELD)
        with pytest.raises(module.ProductionPrelaunchAuthorityError):
            module.production_must_start_control_plane_ready_file_bytes(expired)


@pytest.mark.parametrize(
    "last_stale_field",
    [
        "spend_snapshot_observed_at",
        "controller_baseline_observed_at",
        "control_plane_ready_observed_at",
    ],
)
def test_standalone_acquisition_intrinsic_freshness_is_exactly_sixty_seconds(
    source_authorities: dict[str, object],
    last_stale_field: str,
) -> None:
    module = _module()
    acquisition = copy.deepcopy(
        _full_chain(module, source_authorities)["acquisition"]
    )
    acquired_at = _parse_time(acquisition["acquired_at"])
    ordered_fields = [
        "spend_snapshot_observed_at",
        "controller_baseline_observed_at",
        "control_plane_ready_observed_at",
    ]
    last_index = ordered_fields.index(last_stale_field)

    exact_boundary = copy.deepcopy(acquisition)
    for field in ordered_fields[: last_index + 1]:
        exact_boundary[field] = _iso(acquired_at - timedelta(seconds=60))
    exact_boundary = _rehash(exact_boundary, ACQUISITION_DIGEST_FIELD)
    assert (
        module.production_submission_acquired_file_bytes(exact_boundary)
        == _file_bytes(exact_boundary)
    )

    stale = copy.deepcopy(acquisition)
    for field in ordered_fields[: last_index + 1]:
        stale[field] = _iso(acquired_at - timedelta(seconds=61))
    stale = _rehash(stale, ACQUISITION_DIGEST_FIELD)
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        module.production_submission_acquired_file_bytes(stale)


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "record_type",
        "account_id",
        "provider",
        "region",
        "instance_type",
        "instance_count",
        "use_spot",
        "max_hourly_cost_usd",
        "approved_gpu_runtime_seconds",
        "approved_gpu_cost_usd",
        "skypilot_version",
        "task_name",
    ],
)
def test_descriptor_exact_production_field_guard_kills_each_coherent_mutant(
    source_authorities: dict[str, object],
    field: str,
) -> None:
    module = _module()
    descriptor = copy.deepcopy(source_authorities["descriptor_record"])
    replacements: dict[str, object] = {
        "schema_version": True,
        "record_type": "glm52_sky_campaign_descriptor_v1",
        "account_id": "135792468013",
        "provider": "gcp",
        "region": "us-east-1",
        "instance_type": "p5.4xlarge",
        "instance_count": True,
        "use_spot": 0,
        "max_hourly_cost_usd": 55,
        "approved_gpu_runtime_seconds": 86_400.0,
        "approved_gpu_cost_usd": 1320.96,
        "skypilot_version": "0.13.0+drift",
        "task_name": "glm52-qualification",
    }
    if field == "approved_gpu_cost_usd":
        descriptor[field] = 1_321.0
    else:
        descriptor[field] = replacements[field]
    artifact = _descriptor_artifact_from_record(
        source_authorities["descriptor"],
        descriptor,
    )
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        module.build_production_controller_baseline(
            **_baseline_kwargs(
                {
                    **source_authorities,
                    "descriptor": artifact,
                    "descriptor_record": json.loads(artifact.raw),
                }
            )
        )


@pytest.mark.parametrize(
    "bad_artifact",
    [
        None,
        {},
        "artifact",
        1,
    ],
)
def test_sources_must_be_exact_versioned_artifacts(
    source_authorities: dict[str, object],
    bad_artifact: object,
) -> None:
    module = _module()
    kwargs = _baseline_kwargs(source_authorities)
    kwargs["descriptor"] = bad_artifact
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        module.build_production_controller_baseline(**kwargs)


def test_artifact_subclasses_and_same_bytes_wrong_versions_have_no_authority(
    source_authorities: dict[str, object],
) -> None:
    module = _module()
    descriptor = source_authorities["descriptor"]
    assert type(descriptor) is VersionedJsonArtifact
    subclass = ArtifactSubclass(
        key=descriptor.key,
        raw=descriptor.raw,
        version_id=descriptor.version_id,
    )
    kwargs = _baseline_kwargs(source_authorities)
    kwargs["descriptor"] = subclass
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        module.build_production_controller_baseline(**kwargs)

    wrong_descriptor_version = replace(descriptor, version_id="other-version")
    kwargs = _baseline_kwargs(source_authorities)
    kwargs["descriptor"] = wrong_descriptor_version
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        module.build_production_controller_baseline(**kwargs)

    intent_record = copy.deepcopy(source_authorities["intent_record"])
    intent_record["descriptor_version_id"] = "other-version"
    drifted_intent = _intent_artifact_from_record(intent_record)
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        module.build_production_controller_baseline(
            **_baseline_kwargs(
                _authorities_with_intent(source_authorities, drifted_intent)
            )
        )


@pytest.mark.parametrize(
    "version_id",
    [
        "",
        "null",
        " ",
        "\t",
        "\n",
        "has space",
        "\x1f",
        "\x7f",
        "caf\u00e9",
        StringSubclass("version-1"),
    ],
)
def test_every_invalid_or_subclass_version_id_fails_closed(
    source_authorities: dict[str, object],
    version_id: str,
) -> None:
    module = _module()
    descriptor = source_authorities["descriptor"]
    kwargs = _baseline_kwargs(source_authorities)
    kwargs["descriptor"] = replace(descriptor, version_id=version_id)
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        module.build_production_controller_baseline(**kwargs)


def test_coherently_rehashed_foreign_controller_role_cannot_relabel_intent(
    source_authorities: dict[str, object],
) -> None:
    module = _module()
    descriptor = copy.deepcopy(source_authorities["descriptor_record"])
    descriptor["controller_identity"] = (
        "arn:aws:iam::246813579024:role/foreign-controller"
    )
    artifact = _descriptor_artifact_from_record(
        source_authorities["descriptor"],
        descriptor,
    )
    mutated = {
        **source_authorities,
        "descriptor": artifact,
        "descriptor_record": json.loads(artifact.raw),
    }
    kwargs = _baseline_kwargs(mutated)
    kwargs["controller_profile_arn"] = (
        "arn:aws:iam::246813579024:instance-profile/foreign-controller"
    )
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        module.build_production_controller_baseline(**kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("controller_instance_id", "i-invalid"),
        ("controller_instance_type", "P5.48XLARGE"),
        (
            "controller_profile_arn",
            "arn:aws:iam::246813579024:instance-profile/foreign",
        ),
        ("controller_cluster_name", "sky-jobs-controller-bad/name"),
        ("controller_cluster_name", " sky-jobs-controller-bad"),
        ("controller_cluster_name", "sky-jobs-controller-*"),
        ("ssm_ping_status", "Offline"),
        ("active_exact_name_job_ids", [7]),
        ("active_tagged_p5_instance_ids", [CONTROLLER_INSTANCE_ID]),
    ],
)
def test_controller_coordinate_or_nonempty_inventory_mutant_fails(
    source_authorities: dict[str, object],
    field: str,
    value: object,
) -> None:
    module = _module()
    kwargs = _baseline_kwargs(source_authorities)
    kwargs[field] = value
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        module.build_production_controller_baseline(**kwargs)


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        ("active-row", "active"),
        ("foreign-name", "foreign"),
        ("foreign-workspace", "foreign"),
        ("foreign-identity", "foreign"),
        ("duplicate-id", "duplicate"),
        ("unsorted", "sorted"),
        ("invalid-status", "status"),
        ("future-row", "observation"),
        ("boolean-id", "integer"),
    ],
)
def test_each_controller_history_mutant_fails_closed(
    source_authorities: dict[str, object],
    mutation: str,
    expected_message: str,
) -> None:
    module = _module()
    kwargs = _baseline_kwargs(source_authorities)
    history = copy.deepcopy(kwargs["exact_name_history"])
    if mutation == "active-row":
        history[0]["controller_status"] = "RUNNING"
    elif mutation == "foreign-name":
        history[0]["sky_job_name"] = "foreign"
    elif mutation == "foreign-workspace":
        history[0]["workspace"] = "other"
    elif mutation == "foreign-identity":
        history[0]["controller_identity"] = (
            "arn:aws:iam::246813579024:role/foreign"
        )
    elif mutation == "duplicate-id":
        history.append(copy.deepcopy(history[0]))
    elif mutation == "unsorted":
        older = {**copy.deepcopy(history[0]), "sky_job_id": 8}
        history = [older, history[0]]
    elif mutation == "invalid-status":
        history[0]["controller_status"] = "STOPPED"
    elif mutation == "future-row":
        history[0]["controller_submitted_at"] = _iso(
            kwargs["observed_at"] + timedelta(seconds=1)
        )
    elif mutation == "boolean-id":
        history[0]["sky_job_id"] = True
    kwargs["exact_name_history"] = history
    with pytest.raises(
        module.ProductionPrelaunchAuthorityError,
        match=expected_message,
    ):
        module.build_production_controller_baseline(**kwargs)


@pytest.mark.parametrize(
    "field",
    sorted(DEPLOYMENT_IDENTITY_FIELDS - {"activation_capabilities"}),
)
def test_reviewed_observed_deployment_drift_in_each_top_field_fails(
    source_authorities: dict[str, object],
    field: str,
) -> None:
    module = _module()
    baseline = module.build_production_controller_baseline(
        **_baseline_kwargs(source_authorities)
    )
    kwargs = _ready_kwargs(module, source_authorities, baseline)
    observed = copy.deepcopy(kwargs["observed_deployment_identity"])
    current = observed[field]
    observed[field] = (
        f"{current}-drift"
        if type(current) is str
        else {"unexpected": "deployment-drift"}
    )
    kwargs["observed_deployment_identity"] = observed
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        module.build_production_must_start_control_plane_ready(**kwargs)


def _invalid_activation_value(
    field: str,
    activation: dict[str, object],
) -> object:
    explicit: dict[str, object] = {
        "stack_status": "UPDATE_IN_PROGRESS",
        "stack_operation_in_progress": True,
        "foundation_revision": "a" * 64,
        "managed_mode": "qualification",
        "run_id": "foreign-run",
        "bucket": "foreign-valid-bucket",
        "descriptor_key": "campaigns/foreign/descriptor.json",
        "descriptor_file_sha256": "a" * 64,
        "descriptor_body_sha256": "a" * 64,
        "descriptor_version_id": "foreign-descriptor-version",
        "intent_key": "campaigns/foreign/intent.json",
        "intent_file_sha256": "a" * 64,
        "intent_body_sha256": "a" * 64,
        "intent_version_id": "foreign-intent-version",
        "sky_job_name": "foreign-job",
        "must_start_by": "2026-07-26T17:59:59Z",
        "primary_wake_at": "2026-07-26T17:59:59Z",
        "lambda_state": "Pending",
        "lambda_last_update_status": "Failed",
        "lambda_qualified_arn": str(activation["lambda_qualified_arn"]) + ":8",
        "lambda_execution_role_arn": (
            "arn:aws:iam::246813579024:role/foreign"
        ),
        "lambda_code_sha256": "a" * 64,
        "lambda_code_version_id": "foreign-code-version",
        "lambda_environment": {
            **copy.deepcopy(activation["lambda_environment"]),
            "MANAGED_MODE": "qualification",
        },
        "reconciliation_schedule_expression": "rate(5 minutes)",
        "reconciliation_target_arn": (
            "arn:aws:lambda:us-west-2:246813579024:function:foreign:1"
        ),
        "reconciliation_target_id": "foreign-target",
        "reconciliation_input": {
            **copy.deepcopy(activation["reconciliation_input"]),
            "trigger": "foreign",
        },
        "reconciliation_retry_max_event_age_seconds": 301,
        "reconciliation_retry_max_attempts": 3,
        "reconciliation_dlq_arn": (
            "arn:aws:sqs:us-west-2:246813579024:foreign-dlq"
        ),
        "deadline_schedule_expression": "at(2026-07-26T17:59:59)",
        "deadline_schedule_timezone": "America/Los_Angeles",
        "deadline_flexible_window_mode": "FLEXIBLE",
        "deadline_target_arn": (
            "arn:aws:lambda:us-west-2:246813579024:function:foreign:1"
        ),
        "deadline_target_role_arn": (
            "arn:aws:iam::246813579024:role/foreign"
        ),
        "deadline_input": {
            **copy.deepcopy(activation["deadline_input"]),
            "trigger": "foreign",
        },
        "deadline_retry_max_event_age_seconds": 301,
        "deadline_retry_max_attempts": 3,
        "deadline_dlq_arn": (
            "arn:aws:sqs:us-west-2:246813579024:foreign-dlq"
        ),
    }
    return explicit[field]


@pytest.mark.parametrize("field", sorted(ACTIVATION_CAPABILITY_FIELDS))
def test_each_activation_capability_guard_kills_coherent_reviewed_observed_mutant(
    source_authorities: dict[str, object],
    field: str,
) -> None:
    module = _module()
    baseline = module.build_production_controller_baseline(
        **_baseline_kwargs(source_authorities)
    )
    kwargs = _ready_kwargs(module, source_authorities, baseline)
    reviewed = copy.deepcopy(kwargs["reviewed_deployment_identity"])
    activation = reviewed["activation_capabilities"]
    activation[field] = _invalid_activation_value(field, activation)
    kwargs["reviewed_deployment_identity"] = reviewed
    kwargs["observed_deployment_identity"] = copy.deepcopy(reviewed)
    with pytest.raises(
        module.ProductionPrelaunchAuthorityError,
        match="activation|deployment|Lambda|stack|schedule|retry|deadline|source",
    ):
        module.build_production_must_start_control_plane_ready(**kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "stack_id",
            "arn:aws:cloudformation:us-east-1:246813579024:stack/"
            "keep-glm52-sky-production-control-plane/id",
        ),
        ("template_sha256", "A" * 64),
        (
            "lambda_function_arn",
            "arn:aws:lambda:us-west-2:135792468013:function:"
            "keep-glm52-sky-production-must-start",
        ),
        ("lambda_code_s3_key", "campaigns/foreign/sky-must-start.zip"),
        ("lambda_code_sha256", "0" * 63),
        ("lambda_code_version_id", "null"),
        ("iam_policy_sha256", "not-a-policy-digest"),
        (
            "reconciliation_rule_arn",
            "arn:aws:events:us-west-2:246813579024:rule/foreign",
        ),
        (
            "deadline_schedule_arn",
            "arn:aws:scheduler:us-west-2:246813579024:schedule/default/foreign",
        ),
        (
            "dlq_arn",
            "arn:aws:sqs:us-west-2:246813579024:foreign",
        ),
    ],
)
def test_each_invalid_top_level_deployment_identity_fails_even_when_observed_equal(
    source_authorities: dict[str, object],
    field: str,
    value: object,
) -> None:
    module = _module()
    baseline = module.build_production_controller_baseline(
        **_baseline_kwargs(source_authorities)
    )
    kwargs = _ready_kwargs(module, source_authorities, baseline)
    reviewed = copy.deepcopy(kwargs["reviewed_deployment_identity"])
    reviewed[field] = value
    kwargs["reviewed_deployment_identity"] = reviewed
    kwargs["observed_deployment_identity"] = copy.deepcopy(reviewed)
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        module.build_production_must_start_control_plane_ready(**kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reconciliation_rule_state", "DISABLED"),
        ("deadline_schedule_state", "DISABLED"),
        ("coordinator_mode", "dynamic-job-binding-active"),
    ],
)
def test_disabled_control_plane_or_wrong_coordinator_mode_fails(
    source_authorities: dict[str, object],
    field: str,
    value: str,
) -> None:
    module = _module()
    baseline = module.build_production_controller_baseline(
        **_baseline_kwargs(source_authorities)
    )
    kwargs = _ready_kwargs(module, source_authorities, baseline)
    kwargs[field] = value
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        module.build_production_must_start_control_plane_ready(**kwargs)


def test_unknown_fields_in_deployment_activation_environment_or_inputs_fail(
    source_authorities: dict[str, object],
) -> None:
    module = _module()
    baseline = module.build_production_controller_baseline(
        **_baseline_kwargs(source_authorities)
    )
    for target in ("deployment", "activation", "environment", "reconciliation_input"):
        kwargs = _ready_kwargs(module, source_authorities, baseline)
        reviewed = copy.deepcopy(kwargs["reviewed_deployment_identity"])
        if target == "deployment":
            reviewed["unknown"] = True
        elif target == "activation":
            reviewed["activation_capabilities"]["unknown"] = True
        elif target == "environment":
            reviewed["activation_capabilities"]["lambda_environment"]["unknown"] = (
                "x"
            )
        else:
            reviewed["activation_capabilities"]["reconciliation_input"][
                "unknown"
            ] = "x"
        kwargs["reviewed_deployment_identity"] = reviewed
        kwargs["observed_deployment_identity"] = copy.deepcopy(reviewed)
        with pytest.raises(module.ProductionPrelaunchAuthorityError):
            module.build_production_must_start_control_plane_ready(**kwargs)


def test_baseline_and_readiness_version_ids_are_direct_edge_authority(
    source_authorities: dict[str, object],
) -> None:
    module = _module()
    chain = _full_chain(module, source_authorities)
    wrong_baseline_version = replace(
        chain["baseline_artifact"],
        version_id="wrong-baseline-version",
    )
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        module.validate_production_must_start_control_plane_ready(
            chain["ready"],
            descriptor=source_authorities["descriptor"],
            intent=source_authorities["intent"],
            controller_baseline=wrong_baseline_version,
        )

    swapped_baseline = replace(
        chain["baseline_artifact"],
        version_id=chain["ready_artifact"].version_id,
    )
    swapped_ready = replace(
        chain["ready_artifact"],
        version_id=chain["baseline_artifact"].version_id,
    )
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        module.validate_production_submission_acquired(
            chain["acquisition"],
            descriptor=source_authorities["descriptor"],
            intent=source_authorities["intent"],
            controller_baseline=swapped_baseline,
            must_start_control_plane_ready=swapped_ready,
            now=chain["acquisition"]["acquired_at"],
        )


@pytest.mark.parametrize(
    ("source_name", "variant"),
    [
        ("descriptor", "no-lf"),
        ("descriptor", "double-lf"),
        ("descriptor", "pretty"),
        ("descriptor", "duplicate-key"),
        ("descriptor", "nonfinite"),
        ("descriptor", "bytes-subclass"),
        ("intent", "no-lf"),
        ("intent", "double-lf"),
        ("intent", "pretty"),
        ("intent", "duplicate-key"),
        ("intent", "nonfinite"),
        ("intent", "bytes-subclass"),
    ],
)
def test_noncanonical_duplicate_nonfinite_or_subclass_source_bytes_fail(
    source_authorities: dict[str, object],
    source_name: str,
    variant: str,
) -> None:
    module = _module()
    artifact = source_authorities[source_name]
    record = json.loads(artifact.raw)
    if variant == "no-lf":
        raw: bytes = artifact.raw[:-1]
    elif variant == "double-lf":
        raw = artifact.raw + b"\n"
    elif variant == "pretty":
        raw = json.dumps(record, indent=2).encode("ascii") + b"\n"
    elif variant == "duplicate-key":
        raw = b'{"schema_version":1,' + artifact.raw[1:]
    elif variant == "nonfinite":
        raw = b'{"bad":NaN}\n'
    else:
        raw = BytesSubclass(artifact.raw)
    mutated = _artifact_with_raw(artifact, raw=raw)
    kwargs = _baseline_kwargs(
        {
            **source_authorities,
            source_name: mutated,
        }
    )
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        module.build_production_controller_baseline(**kwargs)


@pytest.mark.parametrize(
    "unsafe_key",
    [
        "/leading",
        "trailing/",
        "campaigns//empty",
        "campaigns/./dot",
        "campaigns/../parent",
        "campaigns/back\\slash",
        "campaigns/star*",
        "campaigns/question?",
        "campaigns/bracket[",
        "campaigns/control\x1f",
        "campaigns/nonascii-\u00e9",
        StringSubclass("campaigns/run/key.json"),
    ],
)
def test_each_unsafe_or_subclass_source_key_fails(
    source_authorities: dict[str, object],
    unsafe_key: str,
) -> None:
    module = _module()
    descriptor = source_authorities["descriptor"]
    mutated = _artifact_with_raw(descriptor, raw=descriptor.raw, key=unsafe_key)
    kwargs = _baseline_kwargs(
        {
            **source_authorities,
            "descriptor": mutated,
        }
    )
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        module.build_production_controller_baseline(**kwargs)


def test_coherently_readdressed_descriptor_and_intent_aliases_fail(
    source_authorities: dict[str, object],
) -> None:
    module = _module()
    descriptor = source_authorities["descriptor"]
    descriptor_alias = replace(
        descriptor,
        key=f"{descriptor.key}.alias",
    )
    kwargs = _baseline_kwargs(
        {
            **source_authorities,
            "descriptor": descriptor_alias,
        }
    )
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        module.build_production_controller_baseline(**kwargs)

    intent = source_authorities["intent"]
    intent_alias = replace(intent, key=f"{intent.key}.alias")
    kwargs = _baseline_kwargs(
        {
            **source_authorities,
            "intent": intent_alias,
        }
    )
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        module.build_production_controller_baseline(**kwargs)


def _fresh_intent_authorities(
    source_authorities: dict[str, object],
) -> dict[str, object]:
    intent_record = copy.deepcopy(source_authorities["intent_record"])
    intent_record["spend_snapshot_observed_at"] = intent_record["intent_at"]
    artifact = _intent_artifact_from_record(intent_record)
    return _authorities_with_intent(source_authorities, artifact)


def test_all_time_equalities_and_exact_sixty_second_boundaries_are_accepted(
    source_authorities: dict[str, object],
) -> None:
    module = _module()
    authorities = _fresh_intent_authorities(source_authorities)
    instant = _parse_time(authorities["intent_record"]["intent_at"])
    equal_chain = _full_chain(
        module,
        authorities,
        baseline_at=instant,
        ready_at=instant,
        acquired_at=instant,
    )
    assert (
        module.validate_production_submission_acquired(
            equal_chain["acquisition"],
            descriptor=authorities["descriptor"],
            intent=authorities["intent"],
            controller_baseline=equal_chain["baseline_artifact"],
            must_start_control_plane_ready=equal_chain["ready_artifact"],
            now=instant,
        )
        == equal_chain["acquisition"]
    )

    sixty_chain = _full_chain(
        module,
        authorities,
        baseline_at=instant,
        ready_at=instant,
        acquired_at=instant + timedelta(seconds=60),
    )
    assert (
        module.validate_production_submission_acquired(
            sixty_chain["acquisition"],
            descriptor=authorities["descriptor"],
            intent=authorities["intent"],
            controller_baseline=sixty_chain["baseline_artifact"],
            must_start_control_plane_ready=sixty_chain["ready_artifact"],
            now=instant + timedelta(seconds=60),
        )
        == sixty_chain["acquisition"]
    )

    current_chain = _full_chain(
        module,
        authorities,
        baseline_at=instant,
        ready_at=instant,
        acquired_at=instant,
    )
    assert (
        module.validate_production_submission_acquired(
            current_chain["acquisition"],
            descriptor=authorities["descriptor"],
            intent=authorities["intent"],
            controller_baseline=current_chain["baseline_artifact"],
            must_start_control_plane_ready=current_chain["ready_artifact"],
            now=instant + timedelta(seconds=60),
        )
        == current_chain["acquisition"]
    )


def test_sixty_one_second_staleness_fails_at_build_and_current_validation(
    source_authorities: dict[str, object],
) -> None:
    module = _module()
    authorities = _fresh_intent_authorities(source_authorities)
    instant = _parse_time(authorities["intent_record"]["intent_at"])
    baseline = module.build_production_controller_baseline(
        **_baseline_kwargs(authorities, observed_at=instant)
    )
    ready = module.build_production_must_start_control_plane_ready(
        **_ready_kwargs(
            module,
            authorities,
            baseline,
            observed_at=instant,
        )
    )
    with pytest.raises(module.ProductionPrelaunchAuthorityError, match="60"):
        module.build_production_submission_acquired(
            descriptor=authorities["descriptor"],
            intent=authorities["intent"],
            controller_baseline=_baseline_artifact(module, baseline),
            must_start_control_plane_ready=_ready_artifact(module, ready),
            acquired_at=instant + timedelta(seconds=61),
        )

    chain = _full_chain(
        module,
        authorities,
        baseline_at=instant,
        ready_at=instant,
        acquired_at=instant,
    )
    with pytest.raises(module.ProductionPrelaunchAuthorityError, match="60"):
        module.validate_production_submission_acquired(
            chain["acquisition"],
            descriptor=authorities["descriptor"],
            intent=authorities["intent"],
            controller_baseline=chain["baseline_artifact"],
            must_start_control_plane_ready=chain["ready_artifact"],
            now=instant + timedelta(seconds=61),
        )


def test_each_time_order_reversal_and_deadline_boundary_fails(
    source_authorities: dict[str, object],
) -> None:
    module = _module()
    intent_at = _parse_time(source_authorities["intent_record"]["intent_at"])
    deadline = _parse_time(source_authorities["intent_record"]["must_start_by"])
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        module.build_production_controller_baseline(
            **_baseline_kwargs(
                source_authorities,
                observed_at=intent_at - timedelta(seconds=1),
            )
        )

    baseline = module.build_production_controller_baseline(
        **_baseline_kwargs(source_authorities, observed_at=intent_at)
    )
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        module.build_production_must_start_control_plane_ready(
            **_ready_kwargs(
                module,
                source_authorities,
                baseline,
                observed_at=intent_at - timedelta(seconds=1),
            )
        )

    ready = module.build_production_must_start_control_plane_ready(
        **_ready_kwargs(
            module,
            source_authorities,
            baseline,
            observed_at=intent_at,
        )
    )
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        module.build_production_submission_acquired(
            descriptor=source_authorities["descriptor"],
            intent=source_authorities["intent"],
            controller_baseline=_baseline_artifact(module, baseline),
            must_start_control_plane_ready=_ready_artifact(module, ready),
            acquired_at=intent_at - timedelta(seconds=1),
        )
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        module.build_production_submission_acquired(
            descriptor=source_authorities["descriptor"],
            intent=source_authorities["intent"],
            controller_baseline=_baseline_artifact(module, baseline),
            must_start_control_plane_ready=_ready_artifact(module, ready),
            acquired_at=deadline,
        )


def test_now_must_not_precede_acquisition_or_reach_deadline(
    source_authorities: dict[str, object],
) -> None:
    module = _module()
    chain = _full_chain(module, source_authorities)
    acquired = _parse_time(chain["acquisition"]["acquired_at"])
    deadline = _parse_time(chain["acquisition"]["must_start_by"])
    for now in (acquired - timedelta(seconds=1), deadline):
        with pytest.raises(module.ProductionPrelaunchAuthorityError):
            module.validate_production_submission_acquired(
                chain["acquisition"],
                descriptor=source_authorities["descriptor"],
                intent=source_authorities["intent"],
                controller_baseline=chain["baseline_artifact"],
                must_start_control_plane_ready=chain["ready_artifact"],
                now=now,
            )


@pytest.mark.parametrize(
    ("record_name", "digest_field", "mutation"),
    [
        ("baseline", BASELINE_DIGEST_FIELD, "unknown"),
        ("baseline", BASELINE_DIGEST_FIELD, "schema-bool"),
        ("baseline", BASELINE_DIGEST_FIELD, "cross-mode"),
        ("baseline", BASELINE_DIGEST_FIELD, "self-hash"),
        ("ready", READY_DIGEST_FIELD, "unknown"),
        ("ready", READY_DIGEST_FIELD, "schema-bool"),
        ("ready", READY_DIGEST_FIELD, "cross-mode"),
        ("ready", READY_DIGEST_FIELD, "self-hash"),
        ("acquisition", ACQUISITION_DIGEST_FIELD, "unknown"),
        ("acquisition", ACQUISITION_DIGEST_FIELD, "schema-bool"),
        ("acquisition", ACQUISITION_DIGEST_FIELD, "cross-mode"),
        ("acquisition", ACQUISITION_DIGEST_FIELD, "self-hash"),
    ],
)
def test_standalone_file_helpers_reject_schema_mode_and_self_hash_mutants(
    source_authorities: dict[str, object],
    record_name: str,
    digest_field: str,
    mutation: str,
) -> None:
    module = _module()
    chain = _full_chain(module, source_authorities)
    value = copy.deepcopy(chain[record_name])
    if mutation == "unknown":
        value["unknown"] = True
        value = _rehash(value, digest_field)
    elif mutation == "schema-bool":
        value["schema_version"] = True
        value = _rehash(value, digest_field)
    elif mutation == "cross-mode":
        value["managed_mode"] = "qualification"
        value = _rehash(value, digest_field)
    else:
        value[digest_field] = "0" * 64
    helper = {
        "baseline": module.production_controller_baseline_file_bytes,
        "ready": module.production_must_start_control_plane_ready_file_bytes,
        "acquisition": module.production_submission_acquired_file_bytes,
    }[record_name]
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        helper(value)


def test_readiness_file_helper_rejects_coherent_foreign_job_or_noncanonical_deadline(
    source_authorities: dict[str, object],
) -> None:
    module = _module()
    ready = copy.deepcopy(_full_chain(module, source_authorities)["ready"])
    activation = ready["activation_capabilities"]
    activation["sky_job_name"] = "foreign-production-job"
    activation["lambda_environment"]["SKY_JOB_NAME"] = "foreign-production-job"
    activation["reconciliation_input"]["sky_job_name"] = "foreign-production-job"
    activation["deadline_input"]["sky_job_name"] = "foreign-production-job"
    ready = _rehash(ready, READY_DIGEST_FIELD)
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        module.production_must_start_control_plane_ready_file_bytes(ready)

    ready = copy.deepcopy(_full_chain(module, source_authorities)["ready"])
    activation = ready["activation_capabilities"]
    noncanonical = str(activation["must_start_by"]).replace("Z", "+00:00")
    activation["must_start_by"] = noncanonical
    activation["primary_wake_at"] = noncanonical
    activation["lambda_environment"]["MUST_START_BY"] = noncanonical
    activation["reconciliation_input"]["must_start_by"] = noncanonical
    activation["deadline_input"]["must_start_by"] = noncanonical
    activation["deadline_schedule_expression"] = f"at({noncanonical[:-1]})"
    ready = _rehash(ready, READY_DIGEST_FIELD)
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        module.production_must_start_control_plane_ready_file_bytes(ready)


@pytest.mark.parametrize(
    "record_name",
    ["baseline", "ready", "acquisition"],
)
def test_file_helpers_require_exact_dict_and_recursive_exact_json_types(
    source_authorities: dict[str, object],
    record_name: str,
) -> None:
    module = _module()
    chain = _full_chain(module, source_authorities)
    helper = {
        "baseline": module.production_controller_baseline_file_bytes,
        "ready": module.production_must_start_control_plane_ready_file_bytes,
        "acquisition": module.production_submission_acquired_file_bytes,
    }[record_name]
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        helper(DictSubclass(chain[record_name]))

    value = copy.deepcopy(chain[record_name])
    value["run_id"] = StringSubclass(str(value["run_id"]))
    digest_field = {
        "baseline": BASELINE_DIGEST_FIELD,
        "ready": READY_DIGEST_FIELD,
        "acquisition": ACQUISITION_DIGEST_FIELD,
    }[record_name]
    value = _rehash(value, digest_field)
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        helper(value)


def test_nested_mapping_and_scalar_subclasses_or_nonfinite_values_fail(
    source_authorities: dict[str, object],
) -> None:
    module = _module()
    kwargs = _baseline_kwargs(source_authorities)
    kwargs["exact_name_history"] = [
        DictSubclass(kwargs["exact_name_history"][0])
    ]
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        module.build_production_controller_baseline(**kwargs)

    baseline = module.build_production_controller_baseline(
        **_baseline_kwargs(source_authorities)
    )
    kwargs = _ready_kwargs(module, source_authorities, baseline)
    deployment = copy.deepcopy(kwargs["reviewed_deployment_identity"])
    deployment["activation_capabilities"]["lambda_environment"] = DictSubclass(
        deployment["activation_capabilities"]["lambda_environment"]
    )
    kwargs["reviewed_deployment_identity"] = deployment
    kwargs["observed_deployment_identity"] = copy.deepcopy(deployment)
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        module.build_production_must_start_control_plane_ready(**kwargs)

    chain = _full_chain(module, source_authorities)
    acquisition = copy.deepcopy(chain["acquisition"])
    acquisition["controller_baseline_observed_at"] = float("nan")
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        module.production_submission_acquired_file_bytes(acquisition)


@pytest.mark.parametrize(
    ("helper_name", "kwargs"),
    [
        (
            "production_controller_baseline_s3_key",
            {"run_id": "../run", "baseline_body_sha256": "a" * 64},
        ),
        (
            "production_controller_baseline_s3_key",
            {"run_id": "run", "baseline_body_sha256": "A" * 64},
        ),
        (
            "production_must_start_control_plane_ready_s3_key",
            {
                "run_id": "run/child",
                "intent_body_sha256": "a" * 64,
                "control_plane_ready_body_sha256": "b" * 64,
            },
        ),
        (
            "production_must_start_control_plane_ready_s3_key",
            {
                "run_id": "run",
                "intent_body_sha256": "a" * 63,
                "control_plane_ready_body_sha256": "b" * 64,
            },
        ),
        (
            "production_submission_acquired_s3_key",
            {
                "run_id": StringSubclass("run"),
                "descriptor_file_sha256": "a" * 64,
            },
        ),
        (
            "production_submission_acquired_s3_key",
            {
                "run_id": "run",
                "descriptor_file_sha256": StringSubclass("a" * 64),
            },
        ),
    ],
)
def test_key_helpers_reject_unsafe_or_noncanonical_components(
    helper_name: str,
    kwargs: dict[str, object],
) -> None:
    module = _module()
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        getattr(module, helper_name)(**kwargs)


def test_coherently_readdressed_baseline_or_readiness_record_fails_source_binding(
    source_authorities: dict[str, object],
) -> None:
    module = _module()
    chain = _full_chain(module, source_authorities)
    baseline = copy.deepcopy(chain["baseline"])
    baseline["controller_cluster_name"] = "sky-jobs-controller-production-b2"
    baseline = _rehash(baseline, BASELINE_DIGEST_FIELD)
    readdressed_baseline = _baseline_artifact(module, baseline)
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        module.validate_production_must_start_control_plane_ready(
            chain["ready"],
            descriptor=source_authorities["descriptor"],
            intent=source_authorities["intent"],
            controller_baseline=readdressed_baseline,
        )

    ready = copy.deepcopy(chain["ready"])
    ready["reconciliation_rule_state"] = "DISABLED"
    ready = _rehash(ready, READY_DIGEST_FIELD)
    readdressed_ready = _ready_artifact(module, ready)
    with pytest.raises(module.ProductionPrelaunchAuthorityError):
        module.validate_production_submission_acquired(
            chain["acquisition"],
            descriptor=source_authorities["descriptor"],
            intent=source_authorities["intent"],
            controller_baseline=chain["baseline_artifact"],
            must_start_control_plane_ready=readdressed_ready,
            now=chain["acquisition"]["acquired_at"],
        )


def test_every_public_failure_is_translated_to_owned_error(
    source_authorities: dict[str, object],
) -> None:
    module = _module()
    operations = [
        lambda: module.production_controller_baseline_s3_key(
            run_id=None,
            baseline_body_sha256="x",
        ),
        lambda: module.production_must_start_control_plane_ready_s3_key(
            run_id="run",
            intent_body_sha256=None,
            control_plane_ready_body_sha256="x",
        ),
        lambda: module.production_submission_acquired_s3_key(
            run_id="run",
            descriptor_file_sha256=None,
        ),
        lambda: module.production_controller_baseline_file_bytes(None),
        lambda: module.production_must_start_control_plane_ready_file_bytes(None),
        lambda: module.production_submission_acquired_file_bytes(None),
        lambda: module.build_production_controller_baseline(
            **{
                **_baseline_kwargs(source_authorities),
                "descriptor": object(),
            }
        ),
    ]
    for operation in operations:
        with pytest.raises(module.ProductionPrelaunchAuthorityError):
            operation()


def test_readiness_candidate_and_callers_do_not_share_nested_mutable_state(
    source_authorities: dict[str, object],
) -> None:
    module = _module()
    baseline = module.build_production_controller_baseline(
        **_baseline_kwargs(source_authorities)
    )
    kwargs = _ready_kwargs(module, source_authorities, baseline)
    reviewed = kwargs["reviewed_deployment_identity"]
    observed = kwargs["observed_deployment_identity"]
    ready = module.build_production_must_start_control_plane_ready(**kwargs)
    ready_before = copy.deepcopy(ready)

    reviewed["activation_capabilities"]["lambda_environment"][
        "MANAGED_MODE"
    ] = "caller-mutated"
    observed["activation_capabilities"]["deadline_input"]["trigger"] = (
        "caller-mutated"
    )
    assert ready == ready_before
    ready_body = dict(ready)
    digest = ready_body.pop(READY_DIGEST_FIELD)
    assert digest == _sha(_canonical(ready_body))

    kwargs = _ready_kwargs(module, source_authorities, baseline)
    reviewed = kwargs["reviewed_deployment_identity"]
    observed = kwargs["observed_deployment_identity"]
    reviewed_before = copy.deepcopy(reviewed)
    observed_before = copy.deepcopy(observed)
    ready = module.build_production_must_start_control_plane_ready(**kwargs)
    ready["activation_capabilities"]["lambda_environment"][
        "MANAGED_MODE"
    ] = "output-mutated"
    assert reviewed == reviewed_before
    assert observed == observed_before


def test_all_public_builders_helpers_and_validators_run_under_preinstalled_traps(
    source_authorities: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    baseline_kwargs = _baseline_kwargs(source_authorities)
    baseline_kwargs["observed_at"] = _iso(baseline_kwargs["observed_at"])
    deployment = _deployment_identity(source_authorities)
    intent_at = _parse_time(source_authorities["intent_record"]["intent_at"])
    ready_at = _iso(intent_at + timedelta(seconds=20))
    acquired_at = _iso(intent_at + timedelta(seconds=30))

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("pure authority attempted forbidden I/O")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(os, "getenv", forbidden)

    class ClockTrap:
        @staticmethod
        def now(*_args: object, **_kwargs: object) -> datetime:
            raise AssertionError("pure authority read the ambient clock")

        @staticmethod
        def fromisoformat(value: str) -> datetime:
            return datetime.fromisoformat(value)

    monkeypatch.setattr(module, "datetime", ClockTrap)
    baseline = module.build_production_controller_baseline(**baseline_kwargs)
    baseline_bytes = module.production_controller_baseline_file_bytes(baseline)
    baseline_sha = module.production_controller_baseline_file_sha256(baseline)
    baseline_key = module.production_controller_baseline_s3_key(
        run_id=str(baseline["run_id"]),
        baseline_body_sha256=str(baseline[BASELINE_DIGEST_FIELD]),
    )
    baseline_artifact = VersionedJsonArtifact(
        key=baseline_key,
        raw=baseline_bytes,
        version_id="production-baseline-trapped-version",
    )
    assert (
        module.validate_production_controller_baseline(
            baseline,
            descriptor=source_authorities["descriptor"],
            intent=source_authorities["intent"],
        )
        == baseline
    )
    assert baseline_sha == _sha(baseline_bytes)

    ready = module.build_production_must_start_control_plane_ready(
        descriptor=source_authorities["descriptor"],
        intent=source_authorities["intent"],
        controller_baseline=baseline_artifact,
        reviewed_deployment_identity=deployment,
        observed_deployment_identity=copy.deepcopy(deployment),
        reconciliation_rule_state="ENABLED",
        deadline_schedule_state="ENABLED",
        coordinator_mode="production-dynamic-job-binding-active",
        observed_at=ready_at,
    )
    ready_bytes = module.production_must_start_control_plane_ready_file_bytes(
        ready
    )
    ready_sha = module.production_must_start_control_plane_ready_file_sha256(
        ready
    )
    ready_key = module.production_must_start_control_plane_ready_s3_key(
        run_id=str(ready["run_id"]),
        intent_body_sha256=str(ready["intent_body_sha256"]),
        control_plane_ready_body_sha256=str(ready[READY_DIGEST_FIELD]),
    )
    ready_artifact = VersionedJsonArtifact(
        key=ready_key,
        raw=ready_bytes,
        version_id="production-ready-trapped-version",
    )
    assert (
        module.validate_production_must_start_control_plane_ready(
            ready,
            descriptor=source_authorities["descriptor"],
            intent=source_authorities["intent"],
            controller_baseline=baseline_artifact,
        )
        == ready
    )
    assert ready_sha == _sha(ready_bytes)

    acquisition = module.build_production_submission_acquired(
        descriptor=source_authorities["descriptor"],
        intent=source_authorities["intent"],
        controller_baseline=baseline_artifact,
        must_start_control_plane_ready=ready_artifact,
        acquired_at=acquired_at,
    )
    acquisition_bytes = module.production_submission_acquired_file_bytes(
        acquisition
    )
    acquisition_sha = module.production_submission_acquired_file_sha256(
        acquisition
    )
    module.production_submission_acquired_s3_key(
        run_id=str(acquisition["run_id"]),
        descriptor_file_sha256=str(acquisition["descriptor_file_sha256"]),
    )
    assert (
        module.validate_production_submission_acquired(
            acquisition,
            descriptor=source_authorities["descriptor"],
            intent=source_authorities["intent"],
            controller_baseline=baseline_artifact,
            must_start_control_plane_ready=ready_artifact,
            now=acquired_at,
        )
        == acquisition
    )
    assert acquisition_sha == _sha(acquisition_bytes)


def test_flat_import_fallback_does_not_mask_transitive_module_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = ROOT / "src/mlx_vq/quality/glm52_sky_production_acquisition.py"
    isolated = tmp_path / "isolated_production_acquisition.py"
    isolated.write_bytes(source.read_bytes())
    real_import_module = importlib.import_module

    def import_module(name: str) -> Any:
        if name == "glm52_sky_campaign":
            raise ModuleNotFoundError(
                "missing transitive dependency",
                name="transitive_dependency",
            )
        return real_import_module(name)

    monkeypatch.setattr(importlib, "import_module", import_module)
    specification = importlib.util.spec_from_file_location(
        "_task3ph1d_transitive_import",
        isolated,
    )
    assert specification is not None and specification.loader is not None
    imported = importlib.util.module_from_spec(specification)
    with pytest.raises(ModuleNotFoundError) as captured:
        specification.loader.exec_module(imported)
    assert captured.value.name == "transitive_dependency"


def test_clean_isolated_import_blocks_forbidden_imports_and_environment_access(
    tmp_path: Path,
) -> None:
    source = ROOT / "src/mlx_vq/quality/glm52_sky_production_acquisition.py"
    (tmp_path / source.name).write_bytes(source.read_bytes())
    (tmp_path / "glm52_sky_campaign.py").write_text(
        "def validate_sky_campaign_descriptor(value): return value\n",
        encoding="utf-8",
    )
    (tmp_path / "glm52_sky_production_submission.py").write_text(
        "from dataclasses import dataclass\n"
        "@dataclass(frozen=True)\n"
        "class VersionedJsonArtifact:\n"
        "    key: str\n"
        "    raw: bytes\n"
        "    version_id: str\n"
        "def validate_production_submission_intent(value): return value\n"
        "def production_submission_intent_file_bytes(value): return b''\n"
        "def production_submission_intent_file_sha256(value): return '0' * 64\n"
        "def production_submission_intent_s3_key(**kwargs): return 'key'\n",
        encoding="utf-8",
    )
    (tmp_path / "glm52_sky_submission_modes.py").write_text(
        "class SubmissionModeContractError(ValueError): pass\n"
        "def expected_sky_job_name(**kwargs): return kwargs['run_id']\n"
        "def record_contract(**kwargs):\n"
        "    return type('C', (), {'schema_version': 1, "
        "'record_type': 'x', 'digest_field': 'x', "
        "'address_kind': 'content-addressed-body'})()\n"
        "def require_opaque_version_id(value, **kwargs): return value\n",
        encoding="utf-8",
    )
    code = """
import builtins
import importlib
import os
import sys

forbidden = {
    "boto3", "botocore", "sky", "skypilot", "mlx", "cuda", "torch",
    "socket", "subprocess", "urllib", "http", "requests",
}
real_import = builtins.__import__
real_import_module = importlib.import_module

def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in forbidden:
        raise AssertionError(f"forbidden import: {name}")
    return real_import(name, *args, **kwargs)

def guarded_import_module(name, *args, **kwargs):
    if name.split(".", 1)[0] in forbidden:
        raise AssertionError(f"forbidden dynamic import: {name}")
    return real_import_module(name, *args, **kwargs)

class EnvironmentTrap:
    def __getitem__(self, key):
        raise AssertionError(f"environment read: {key}")
    def get(self, key, default=None):
        raise AssertionError(f"environment read: {key}")
    def __iter__(self):
        raise AssertionError("environment iteration")

builtins.__import__ = guarded_import
importlib.import_module = guarded_import_module
os.environ = EnvironmentTrap()
os.getenv = lambda *args, **kwargs: (_ for _ in ()).throw(
    AssertionError("getenv read")
)
sys.path.insert(0, sys.argv[1])
import glm52_sky_production_acquisition as module
assert len(module.__all__) == 16
print("ISOLATED_IMPORT_CLEAN")
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code, str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ISOLATED_IMPORT_CLEAN"


def test_actual_system_python39_compiles_and_imports_against_flat_public_stubs(
    tmp_path: Path,
) -> None:
    source = ROOT / "src/mlx_vq/quality/glm52_sky_production_acquisition.py"
    isolated = tmp_path / source.name
    isolated.write_bytes(source.read_bytes())
    (tmp_path / "glm52_sky_campaign.py").write_text(
        "def validate_sky_campaign_descriptor(value): return value\n",
        encoding="utf-8",
    )
    (tmp_path / "glm52_sky_production_submission.py").write_text(
        "from dataclasses import dataclass\n"
        "@dataclass(frozen=True)\n"
        "class VersionedJsonArtifact:\n"
        "    key: str\n"
        "    raw: bytes\n"
        "    version_id: str\n"
        "def validate_production_submission_intent(value): return value\n"
        "def production_submission_intent_file_bytes(value): return b''\n"
        "def production_submission_intent_file_sha256(value): return '0' * 64\n"
        "def production_submission_intent_s3_key(**kwargs): return 'key'\n",
        encoding="utf-8",
    )
    (tmp_path / "glm52_sky_submission_modes.py").write_text(
        "class SubmissionModeContractError(ValueError): pass\n"
        "def expected_sky_job_name(**kwargs): return kwargs['run_id']\n"
        "def record_contract(**kwargs):\n"
        "    return type('C', (), {'schema_version': 1, "
        "'record_type': 'x', 'digest_field': 'x', "
        "'address_kind': 'content-addressed-body'})()\n"
        "def require_opaque_version_id(value, **kwargs): return value\n",
        encoding="utf-8",
    )
    code = """
import json
import pathlib
import sys
sys.path.insert(0, sys.argv[1])
import glm52_sky_production_acquisition as module
assert len(module.__all__) == 16
print(json.dumps({"python": list(sys.version_info[:2]), "ok": True}))
"""
    completed = subprocess.run(
        ["/usr/bin/python3", "-I", "-c", code, str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["ok"] is True
    assert tuple(result["python"]) >= (3, 9)
