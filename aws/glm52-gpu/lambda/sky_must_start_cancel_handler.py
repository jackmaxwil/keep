"""One-minute AWS coordinator for an immutable SkyPilot must-start deadline."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from mlx_vq.quality.glm52_campaign_watchdog import (
    validate_skypilot_submission_marker,
)
from mlx_vq.quality.glm52_sky_campaign import (
    validate_sky_campaign_descriptor,
)
from mlx_vq.quality.glm52_sky_must_start import (
    APPROVED_ACCOUNT_ID,
    APPROVED_REGION,
    build_must_start_alert_delivered,
    build_must_start_cancel_completed,
    build_must_start_cancel_requested,
    build_must_start_cancel_superseded,
    build_must_start_controller_observation,
    build_must_start_job_binding,
    build_must_start_target_binding,
    build_timely_start_accepted,
    canonical_bytes,
    decide_controller_start_authority,
    decide_must_start_action,
    expected_sky_job_name,
    validate_must_start_alert_delivered,
    validate_must_start_cancel_completed,
    validate_must_start_cancel_requested,
    validate_must_start_cancel_superseded,
    validate_must_start_controller_observation,
    validate_must_start_job_binding,
    validate_must_start_target_binding,
    validate_timely_start_accepted,
    validate_timely_start_latch,
)

_MISSING_CODES = {"404", "NoSuchKey", "NotFound"}
_PRECONDITION_CODES = {"412", "PreconditionFailed"}
_TERMINAL_STATUSES = {
    "CANCELLED",
    "SUCCEEDED",
    "FAILED",
    "FAILED_SETUP",
    "FAILED_PRECHECKS",
    "FAILED_NO_RESOURCE",
    "FAILED_CONTROLLER",
}
_RESULT_FIELDS = {
    "schema_version",
    "record_type",
    "sky_job_name",
    "outcome",
    "before_nonterminal_job_ids",
    "after_nonterminal_job_ids",
    "all_job_ids",
    "target_job_id",
    "target_status",
    "observed_at",
}
_STRICT_RESULT_FIELDS = {
    "schema_version",
    "record_type",
    "operation",
    "sky_job_name",
    "workspace",
    "all_job_ids",
    "target_job_id",
    "target_job_name",
    "target_workspace",
    "status",
    "schedule_state",
    "submitted_at",
    "start_at",
    "worker_cluster_name",
    "recovery_count",
    "cancel_called",
    "post_status",
    "observed_at",
}
_RUNTIME_SESSION: Any | None = None
_RUNTIME_CLIENTS: dict[str, Any] = {}
_RUNTIME_CLIENT_CONFIG = Config(
    connect_timeout=1,
    read_timeout=2,
    retries={"mode": "standard", "total_max_attempts": 1},
    tcp_keepalive=True,
)


def _runtime_client(*, region: str, service: str) -> tuple[Any, Any]:
    global _RUNTIME_SESSION
    if _RUNTIME_SESSION is None:
        _RUNTIME_SESSION = boto3.session.Session(region_name=region)
    elif str(_RUNTIME_SESSION.region_name) != region:
        raise ValueError("cached runtime AWS region mismatch")
    client = _RUNTIME_CLIENTS.get(service)
    if client is None:
        client = _RUNTIME_SESSION.client(
            service,
            config=_RUNTIME_CLIENT_CONFIG,
        )
        _RUNTIME_CLIENTS[service] = client
    return _RUNTIME_SESSION, client


def _log_stage(stage: str) -> None:
    print(
        json.dumps(
            {
                "component": "glm52-sky-must-start",
                "stage": stage,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _parse_time(value: object, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class MustStartCoordinatorConfig:
    account_id: str
    region: str
    bucket: str
    descriptor_key: str
    submission_key: str
    submission_body_sha256: str
    run_id: str
    managed_mode: str
    sky_job_name: str
    must_start_by: str
    alert_topic_arn: str
    expected_target_job_id: int | None = None
    observe_only: bool = False
    expected_workspace: str | None = None
    expected_descriptor_file_sha256: str | None = None
    expected_submission_submitted_at: str | None = None
    expected_controller_instance_id: str | None = None
    expected_controller_instance_type: str | None = None
    expected_controller_profile_arn: str | None = None
    expected_controller_cluster_name: str | None = None
    starting_grace_seconds: int = 20
    reconciliation_rule_name: str | None = None
    primary_wake_max_wait_seconds: int = 120
    expected_activation_job_binding_sha256: str | None = None
    expected_activation_observation_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.account_id != APPROVED_ACCOUNT_ID:
            raise ValueError("configured AWS account is not approved")
        if self.region != APPROVED_REGION:
            raise ValueError("configured AWS region is not approved")
        if self.sky_job_name != expected_sky_job_name(
            self.run_id, self.managed_mode
        ):
            raise ValueError("configured SkyPilot job name authority mismatch")
        _parse_time(self.must_start_by, field="configured must_start_by")
        prefix = f"campaigns/{self.run_id}/"
        if not self.descriptor_key.startswith(prefix + "submissions/"):
            raise ValueError("configured descriptor key is not run scoped")
        if not self.submission_key.startswith(
            prefix + "monitor/submission-locks/"
        ):
            raise ValueError("configured submission key is not immutable")
        if (
            len(self.submission_body_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.submission_body_sha256
            )
        ):
            raise ValueError("configured submission body SHA-256 is invalid")
        if self.expected_target_job_id is not None and (
            type(self.expected_target_job_id) is not int
            or self.expected_target_job_id <= 0
        ):
            raise ValueError("configured target job ID is invalid")
        if type(self.observe_only) is not bool:
            raise ValueError("observe_only must be a boolean")
        if type(self.starting_grace_seconds) is not int or not (
            15 <= self.starting_grace_seconds <= 30
        ):
            raise ValueError("starting grace must be between 15 and 30 seconds")
        if type(self.primary_wake_max_wait_seconds) is not int or not (
            60 <= self.primary_wake_max_wait_seconds <= 120
        ):
            raise ValueError(
                "primary wake maximum wait must be between 60 and 120 seconds"
            )
        if self.reconciliation_rule_name is not None and (
            not self.reconciliation_rule_name
            or len(self.reconciliation_rule_name) > 64
            or any(
                character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "abcdefghijklmnopqrstuvwxyz0123456789-_."
                for character in self.reconciliation_rule_name
            )
        ):
            raise ValueError("reconciliation rule name is invalid")
        activation_hashes = (
            self.expected_activation_job_binding_sha256,
            self.expected_activation_observation_sha256,
        )
        if any(value is not None for value in activation_hashes):
            if any(value is None for value in activation_hashes):
                raise ValueError("active observation proof authority is incomplete")
            for field, value in (
                (
                    "activation JOB_BINDING.json SHA-256",
                    self.expected_activation_job_binding_sha256,
                ),
                (
                    "activation observation SHA-256",
                    self.expected_activation_observation_sha256,
                ),
            ):
                if len(str(value)) != 64 or any(
                    character not in "0123456789abcdef"
                    for character in str(value)
                ):
                    raise ValueError(f"{field} is invalid")
        strict_values = (
            self.expected_workspace,
            self.expected_descriptor_file_sha256,
            self.expected_submission_submitted_at,
            self.expected_controller_instance_id,
            self.expected_controller_instance_type,
            self.expected_controller_profile_arn,
            self.expected_controller_cluster_name,
        )
        if any(value is not None for value in strict_values):
            if any(value is None for value in strict_values):
                raise ValueError("exact controller authority is incomplete")
            if self.expected_target_job_id is None:
                raise ValueError("exact controller authority requires a target job ID")
            if self.expected_workspace != "default":
                raise ValueError("exact controller workspace must be default")
            descriptor_file_sha = str(self.expected_descriptor_file_sha256)
            if len(descriptor_file_sha) != 64 or any(
                character not in "0123456789abcdef"
                for character in descriptor_file_sha
            ):
                raise ValueError("configured descriptor file SHA-256 is invalid")
            _parse_time(
                self.expected_submission_submitted_at,
                field="configured submission submitted_at",
            )
            instance_id = str(self.expected_controller_instance_id)
            if not instance_id.startswith("i-"):
                raise ValueError("configured controller instance ID is invalid")
            profile_prefix = (
                f"arn:aws:iam::{self.account_id}:instance-profile/"
            )
            if not str(self.expected_controller_profile_arn).startswith(
                profile_prefix
            ):
                raise ValueError("configured controller profile is invalid")
            if not str(self.expected_controller_cluster_name).startswith(
                "sky-jobs-controller-"
            ):
                raise ValueError("configured controller cluster is invalid")

    @property
    def strict_controller_authority(self) -> bool:
        return self.expected_controller_instance_id is not None

    @property
    def marker_prefix(self) -> str:
        return (
            f"campaigns/{self.run_id}/monitor/must-start/"
            f"{self.managed_mode}/{self.submission_body_sha256}"
        )

    @property
    def latch_key(self) -> str:
        return f"{self.marker_prefix}/TIMELY_START_LATCH.json"

    @property
    def request_key(self) -> str:
        return f"{self.marker_prefix}/CANCEL_REQUESTED.json"

    @property
    def completion_key(self) -> str:
        return f"{self.marker_prefix}/CANCEL_COMPLETED.json"

    @property
    def target_binding_key(self) -> str:
        return f"{self.marker_prefix}/TARGET_BOUND.json"

    @property
    def request_alerted_key(self) -> str:
        return f"{self.marker_prefix}/REQUEST_ALERTED.json"

    @property
    def completion_alerted_key(self) -> str:
        return f"{self.marker_prefix}/COMPLETION_ALERTED.json"

    @property
    def job_binding_key(self) -> str:
        return f"{self.marker_prefix}/JOB_BINDING.json"

    @property
    def accepted_key(self) -> str:
        return f"{self.marker_prefix}/TIMELY_START_ACCEPTED.json"

    @property
    def superseded_key(self) -> str:
        return f"{self.marker_prefix}/CANCEL_SUPERSEDED.json"

    def observation_key(self, body_sha256: str) -> str:
        return f"{self.marker_prefix}/observations/{body_sha256}.json"


def _read_json(s3, *, bucket: str, key: str) -> tuple[dict[str, Any], bytes] | None:  # noqa: ANN001
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"]
        try:
            raw = body.read()
        finally:
            body.close()
    except ClientError as error:
        if str(error.response.get("Error", {}).get("Code")) in _MISSING_CODES:
            return None
        raise
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"s3://{bucket}/{key} is not JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"s3://{bucket}/{key} must contain an object")
    return value, raw


def _conditional_put(
    s3,  # noqa: ANN001
    *,
    bucket: str,
    key: str,
    marker: Mapping[str, object],
    validator: Callable[[Mapping[str, object]], dict[str, object]],
    winner_matches: Callable[[Mapping[str, object]], bool] | None = None,
) -> bool:
    body = json.dumps(
        marker,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
            IfNoneMatch="*",
            Metadata={
                "glm52-run-id": str(marker["run_id"]),
                "glm52-body-sha256": str(
                    marker.get("request_body_sha256")
                    or marker.get("completion_body_sha256")
                    or marker.get("target_binding_body_sha256")
                    or marker.get("alert_delivery_body_sha256")
                    or marker.get("observation_body_sha256")
                    or marker.get("job_binding_body_sha256")
                    or marker.get("accepted_body_sha256")
                    or marker.get("superseded_body_sha256")
                ),
            },
        )
        return True
    except ClientError as error:
        if str(error.response.get("Error", {}).get("Code")) not in (
            _PRECONDITION_CODES
        ):
            raise
    winner_value = _read_json(s3, bucket=bucket, key=key)
    if winner_value is None:
        raise ValueError("conditional marker winner disappeared")
    winner = validator(winner_value[0])
    if winner_matches is not None:
        matches = winner_matches(winner)
    else:
        matches = winner == dict(marker)
    if not matches:
        raise ValueError("conditional marker winner is foreign")
    return False


def _assert_exact_authority(
    marker: Mapping[str, object],
    authority: Mapping[str, object],
    *,
    label: str,
) -> None:
    for field, expected in authority.items():
        if marker.get(field) != expected:
            raise ValueError(f"{label} {field} authority mismatch")


def _find_controller(ec2, *, run_id: str) -> str:  # noqa: ANN001
    response = ec2.describe_instances(
        Filters=[
            {"Name": "tag:project", "Values": ["keep-glm52"]},
            {"Name": "tag:owner", "Values": ["jack.mazac"]},
            {"Name": "tag:model", "Values": ["glm-5.2"]},
            {"Name": "tag:campaign-run-id", "Values": [run_id]},
            {
                "Name": "tag:cost-allocation",
                "Values": ["glm52-sky-campaign"],
            },
            {"Name": "tag:ray-cluster-name", "Values": ["sky-jobs-controller-*"]},
            {"Name": "instance-state-name", "Values": ["running"]},
        ]
    )
    instances = [
        instance
        for reservation in response.get("Reservations", [])
        for instance in reservation.get("Instances", [])
    ]
    if len(instances) != 1:
        raise ValueError("must resolve exactly one tagged SkyPilot jobs controller")
    instance_id = instances[0].get("InstanceId")
    if not isinstance(instance_id, str) or not instance_id.startswith("i-"):
        raise ValueError("SkyPilot jobs controller instance ID is invalid")
    return instance_id


def _resolve_exact_controller(
    ec2,  # noqa: ANN001
    ssm,  # noqa: ANN001
    *,
    config: MustStartCoordinatorConfig,
) -> dict[str, str]:
    if not config.strict_controller_authority:
        raise ValueError("exact controller authority is not configured")
    response = ec2.describe_instances(
        InstanceIds=[str(config.expected_controller_instance_id)]
    )
    instances = [
        instance
        for reservation in response.get("Reservations", [])
        for instance in reservation.get("Instances", [])
    ]
    if len(instances) != 1:
        raise ValueError("exact controller instance is missing or ambiguous")
    instance = instances[0]
    if instance.get("InstanceId") != config.expected_controller_instance_id:
        raise ValueError("controller instance ID mismatch")
    if instance.get("InstanceType") != config.expected_controller_instance_type:
        raise ValueError("controller instance type mismatch")
    if instance.get("State", {}).get("Name") != "running":
        raise ValueError("controller instance is not running")
    if (
        instance.get("IamInstanceProfile", {}).get("Arn")
        != config.expected_controller_profile_arn
    ):
        raise ValueError("controller instance profile mismatch")
    availability_zone = str(instance.get("Placement", {}).get("AvailabilityZone", ""))
    if not availability_zone.startswith(config.region):
        raise ValueError("controller instance region mismatch")
    tags = {
        str(tag.get("Key")): str(tag.get("Value"))
        for tag in instance.get("Tags", [])
        if isinstance(tag, dict)
    }
    if tags.get("campaign-run-id") != config.run_id:
        raise ValueError("controller campaign tag mismatch")
    if tags.get("ray-cluster-name") != config.expected_controller_cluster_name:
        raise ValueError("controller cluster tag mismatch")
    information = ssm.describe_instance_information(
        Filters=[
            {
                "Key": "InstanceIds",
                "Values": [str(config.expected_controller_instance_id)],
            }
        ]
    ).get("InstanceInformationList", [])
    if len(information) != 1:
        raise ValueError("controller SSM identity is missing or ambiguous")
    if (
        information[0].get("InstanceId")
        != config.expected_controller_instance_id
        or information[0].get("PingStatus") != "Online"
    ):
        raise ValueError("controller is not SSM Online")
    return {
        "controller_instance_id": str(config.expected_controller_instance_id),
        "controller_instance_type": str(config.expected_controller_instance_type),
        "controller_profile_arn": str(config.expected_controller_profile_arn),
        "controller_cluster_name": str(config.expected_controller_cluster_name),
    }


def _strict_controller_shim() -> str:
    # Pinned SkyPilot 0.13 state is the authority.  The cancel operation
    # performs its own exact row read immediately before the exact-ID call.
    return r'''
import json
import sys
import time
from datetime import datetime, timezone

from sky.jobs import state, utils

name = sys.argv[1]
workspace = sys.argv[2]
target_job_id = int(sys.argv[3])
operation = sys.argv[4]
deadline_epoch = float(sys.argv[5])
invoked_epoch = float(sys.argv[6])
terminal = {
    "CANCELLED", "SUCCEEDED", "FAILED", "FAILED_SETUP",
    "FAILED_PRECHECKS", "FAILED_NO_RESOURCE", "FAILED_CONTROLLER",
}

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

def exact_row():
    rows, total = state.get_managed_jobs_with_filters(
        job_ids=[target_job_id],
        workspace_match=workspace,
    )
    matching = [
        row for row in rows
        if row.get("job_id") == target_job_id
        and row.get("job_name") == name
        and row.get("workspace") == workspace
    ]
    if total != 1 or len(matching) != 1:
        raise RuntimeError("exact managed-job row is missing or ambiguous")
    return matching[0]

all_job_ids = sorted(set(state.get_all_job_ids_by_name(name)))
row = exact_row()
status = enum_name(row.get("status"))
schedule_state = enum_name(row.get("schedule_state"))
start_epoch = row.get("start_at")
start_at = iso_time(start_epoch)
worker_cluster_name = row.get("current_cluster_name")
if worker_cluster_name is None:
    task_name = row.get("task_name")
    if task_name:
        worker_cluster_name = utils.generate_managed_job_cluster_name(
            task_name, target_job_id
        )
cancel_called = False
post_status = None
if operation == "cancel":
    timely = (
        status in terminal | {"RUNNING", "RECOVERING"}
        and start_epoch is not None
        and float(start_epoch) <= deadline_epoch
    )
    late_started = (
        status in terminal | {"RUNNING", "RECOVERING"}
        and start_epoch is not None
        and float(start_epoch) > deadline_epoch
    )
    pending_expired = (
        status in {"PENDING", "STARTING"}
        and invoked_epoch >= deadline_epoch
    )
    if not timely and (late_started or pending_expired):
        # The row above is the required immediate name/workspace/ID reread.
        utils.cancel_jobs_by_id(
            [target_job_id],
            current_workspace=workspace,
            graceful=False,
        )
        cancel_called = True
        for _ in range(10):
            post_status = enum_name(state.get_status(target_job_id))
            if post_status in terminal:
                break
            time.sleep(1)
elif operation != "observe":
    raise RuntimeError("unsupported exact controller operation")

result = {
    "schema_version": 1,
    "record_type": "glm52_sky_must_start_controller_result_v1",
    "operation": operation,
    "sky_job_name": name,
    "workspace": workspace,
    "all_job_ids": all_job_ids,
    "target_job_id": target_job_id,
    "target_job_name": row.get("job_name"),
    "target_workspace": row.get("workspace"),
    "status": status,
    "schedule_state": schedule_state,
    "submitted_at": iso_time(row.get("submitted_at")),
    "start_at": start_at,
    "worker_cluster_name": worker_cluster_name,
    "recovery_count": row.get("recovery_count"),
    "cancel_called": cancel_called,
    "post_status": post_status,
    "observed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
}
print("GLM52_MUST_START_RESULT=" + json.dumps(
    result, sort_keys=True, separators=(",", ":")
))
'''.strip()


def _validate_strict_result(
    value: object,
    *,
    config: MustStartCoordinatorConfig,
    operation: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _STRICT_RESULT_FIELDS:
        raise ValueError("exact controller result schema mismatch")
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("record_type")
        != "glm52_sky_must_start_controller_result_v1"
        or value.get("operation") != operation
        or value.get("sky_job_name") != config.sky_job_name
        or value.get("workspace") != config.expected_workspace
        or value.get("target_job_id") != config.expected_target_job_id
        or value.get("target_job_name") != config.sky_job_name
        or value.get("target_workspace") != config.expected_workspace
    ):
        raise ValueError("exact controller result authority mismatch")
    all_ids = value.get("all_job_ids")
    if (
        not isinstance(all_ids, list)
        or any(type(item) is not int or item <= 0 for item in all_ids)
        or all_ids != sorted(set(all_ids))
        or config.expected_target_job_id not in all_ids
    ):
        raise ValueError("exact controller result history is invalid")
    status = value.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError("exact controller result status is invalid")
    schedule_state = value.get("schedule_state")
    if not isinstance(schedule_state, str) or not schedule_state:
        raise ValueError("exact controller result schedule state is invalid")
    _parse_time(value.get("submitted_at"), field="controller submitted_at")
    if value.get("start_at") is not None:
        _parse_time(value.get("start_at"), field="controller start_at")
    if value.get("worker_cluster_name") is not None and not isinstance(
        value.get("worker_cluster_name"), str
    ):
        raise ValueError("exact controller worker cluster is invalid")
    if type(value.get("recovery_count")) is not int or value.get(
        "recovery_count"
    ) < 0:
        raise ValueError("exact controller recovery count is invalid")
    if type(value.get("cancel_called")) is not bool:
        raise ValueError("exact controller cancel flag is invalid")
    post_status = value.get("post_status")
    if post_status is not None and not isinstance(post_status, str):
        raise ValueError("exact controller post status is invalid")
    if operation == "observe" and (
        value.get("cancel_called") or post_status is not None
    ):
        raise ValueError("observe-only controller result attempted cancellation")
    _parse_time(value.get("observed_at"), field="controller observed_at")
    return dict(value)


def _run_exact_controller(
    ssm,  # noqa: ANN001
    *,
    config: MustStartCoordinatorConfig,
    operation: str,
    now: datetime,
    sleep: Callable[[float], None],
) -> dict[str, object]:
    command = (
        "sudo -u ubuntu -- /home/ubuntu/skypilot-runtime/bin/python -c "
        f"{shlex.quote(_strict_controller_shim())} "
        f"{shlex.quote(config.sky_job_name)} "
        f"{shlex.quote(str(config.expected_workspace))} "
        f"{config.expected_target_job_id} {operation} "
        f"{_parse_time(config.must_start_by, field='must_start_by').timestamp()} "
        f"{now.astimezone(timezone.utc).timestamp()}"
    )
    sent = ssm.send_command(
        InstanceIds=[str(config.expected_controller_instance_id)],
        DocumentName="AWS-RunShellScript",
        Comment=f"KEEP GLM52 exact must-start {operation}",
        Parameters={"commands": [command]},
        TimeoutSeconds=45,
    )
    command_id = sent["Command"]["CommandId"]
    for _ in range(10):
        try:
            invocation = ssm.get_command_invocation(
                CommandId=command_id,
                InstanceId=str(config.expected_controller_instance_id),
            )
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code"))
            if code == "InvocationDoesNotExist":
                sleep(1)
                continue
            raise
        status = invocation.get("Status")
        if status in {"Pending", "InProgress", "Delayed"}:
            sleep(1)
            continue
        if status != "Success":
            raise ValueError(f"exact controller SSM command failed: {status}")
        prefix = "GLM52_MUST_START_RESULT="
        lines = [
            line
            for line in str(invocation.get("StandardOutputContent", "")).splitlines()
            if line.startswith(prefix)
        ]
        if len(lines) != 1:
            raise ValueError("exact controller result is missing or ambiguous")
        try:
            value = json.loads(lines[0][len(prefix) :])
        except json.JSONDecodeError as error:
            raise ValueError("exact controller result is not JSON") from error
        return _validate_strict_result(
            value,
            config=config,
            operation=operation,
        )
    raise TimeoutError("exact controller SSM command did not finish")


def _controller_shim() -> str:
    # This runs inside the pinned SkyPilot 0.13 control environment. Public
    # queue() can omit controller-local jobs, so bind directly to its state DB.
    return r'''
import json
import sys
from datetime import datetime, timezone

from sky.jobs import state, utils

name = sys.argv[1]
bound_target = None if sys.argv[2] == "-" else int(sys.argv[2])
terminal = {
    "CANCELLED", "SUCCEEDED", "FAILED", "FAILED_SETUP",
    "FAILED_PRECHECKS", "FAILED_NO_RESOURCE", "FAILED_CONTROLLER",
}

def status_name(value):
    if value is None:
        return None
    return str(getattr(value, "value", getattr(value, "name", value))).upper()

all_job_ids = sorted(set(state.get_all_job_ids_by_name(name)))
target_job_id = None
target_status = None
if bound_target is not None:
    target_job_id = bound_target
    if target_job_id not in all_job_ids:
        before_nonterminal = []
        after_nonterminal = []
        outcome = "ambiguous"
    else:
        target_status = status_name(state.get_status(target_job_id))
        before_nonterminal = (
            [] if target_status in terminal else [target_job_id]
        )
        if target_status in terminal:
            after_nonterminal = []
            outcome = "terminal"
        else:
            utils.cancel_jobs_by_id(
                [target_job_id],
                current_workspace="default",
                graceful=False,
            )
            target_status = status_name(state.get_status(target_job_id))
            after_nonterminal = (
                [] if target_status in terminal else [target_job_id]
            )
            if target_status in terminal:
                outcome = "terminal"
            elif target_status is not None:
                outcome = "nonterminal"
            else:
                outcome = "ambiguous"
else:
    matched = sorted(set(
        state.get_nonterminal_job_ids_by_name(name, all_users=False)
    ))
    if not matched:
        before_nonterminal = []
        after_nonterminal = []
        outcome = "not_found"
    elif len(matched) != 1:
        before_nonterminal = matched
        after_nonterminal = matched
        outcome = "ambiguous"
    else:
        target_job_id = matched[0]
        target_status = status_name(state.get_status(target_job_id))
        before_nonterminal = (
            [] if target_status in terminal else [target_job_id]
        )
        after_nonterminal = before_nonterminal
        outcome = "terminal" if target_status in terminal else "nonterminal"
result = {
    "schema_version": 1,
    "record_type": "glm52_sky_must_start_cancel_result_v1",
    "sky_job_name": name,
    "outcome": outcome,
    "before_nonterminal_job_ids": before_nonterminal,
    "after_nonterminal_job_ids": after_nonterminal,
    "all_job_ids": all_job_ids,
    "target_job_id": target_job_id,
    "target_status": target_status,
    "observed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
}
print("GLM52_MUST_START_RESULT=" + json.dumps(
    result, sort_keys=True, separators=(",", ":")
))
'''.strip()


def _validate_result(
    value: object,
    *,
    sky_job_name: str,
    expected_target_job_id: int | None = None,
    discovered_target: bool = False,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _RESULT_FIELDS:
        raise ValueError("SkyPilot cancellation result schema mismatch")
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("record_type")
        != "glm52_sky_must_start_cancel_result_v1"
        or value.get("sky_job_name") != sky_job_name
    ):
        raise ValueError("SkyPilot cancellation result authority mismatch")
    outcome = value.get("outcome")
    if outcome not in {"not_found", "nonterminal", "terminal", "ambiguous"}:
        raise ValueError("SkyPilot cancellation result outcome is invalid")
    for field in (
        "before_nonterminal_job_ids",
        "after_nonterminal_job_ids",
        "all_job_ids",
    ):
        items = value.get(field)
        if (
            not isinstance(items, list)
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or item <= 0
                for item in items
            )
            or items != sorted(set(items))
        ):
            raise ValueError(f"SkyPilot cancellation result {field} is invalid")
    target = value.get("target_job_id")
    if target is not None and (
        isinstance(target, bool)
        or not isinstance(target, int)
        or target <= 0
    ):
        raise ValueError("SkyPilot cancellation target job ID is invalid")
    if expected_target_job_id is not None:
        if (
            isinstance(expected_target_job_id, bool)
            or not isinstance(expected_target_job_id, int)
            or expected_target_job_id <= 0
        ):
            raise ValueError("expected SkyPilot target job ID is invalid")
        if target != expected_target_job_id:
            raise ValueError("SkyPilot cancellation target job ID mismatch")
    status = value.get("target_status")
    if status is not None and not isinstance(status, str):
        raise ValueError("SkyPilot cancellation target status is invalid")
    _parse_time(value.get("observed_at"), field="cancellation observed_at")
    before = value["before_nonterminal_job_ids"]
    after = value["after_nonterminal_job_ids"]
    all_ids = value["all_job_ids"]
    if not set(after).issubset(before) or not set(before).issubset(all_ids):
        raise ValueError("SkyPilot cancellation result job sets are inconsistent")
    if outcome == "not_found" and (
        all_ids or before or after or target is not None or status is not None
    ):
        raise ValueError("not_found cancellation result contains job state")
    if outcome == "nonterminal" and not (
        isinstance(target, int)
        and before == [target]
        and after == [target]
        and target in all_ids
        and status not in _TERMINAL_STATUSES
    ):
        raise ValueError("nonterminal cancellation result is not exact")
    terminal_has_authority = (
        expected_target_job_id == target
        if expected_target_job_id is not None
        else discovered_target
    )
    if outcome == "terminal" and not (
        isinstance(target, int)
        and before in ([], [target])
        and after == []
        and target in all_ids
        and status in _TERMINAL_STATUSES
        and terminal_has_authority
    ):
        raise ValueError("terminal cancellation result is not exact")
    return dict(value)


def _run_controller_cancel(
    ssm,  # noqa: ANN001
    *,
    instance_id: str,
    sky_job_name: str,
    sleep: Callable[[float], None],
    target_job_id: int | None = None,
) -> dict[str, object]:
    target_arg = "-" if target_job_id is None else str(target_job_id)
    command = (
        "sudo -u ubuntu -- /home/ubuntu/skypilot-runtime/bin/python -c "
        f"{shlex.quote(_controller_shim())} {shlex.quote(sky_job_name)}"
        f" {target_arg}"
    )
    sent = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Comment="KEEP GLM52 exact must-start cancellation reconciliation",
        Parameters={"commands": [command]},
        TimeoutSeconds=45,
    )
    command_id = sent["Command"]["CommandId"]
    for _ in range(10):
        try:
            invocation = ssm.get_command_invocation(
                CommandId=command_id,
                InstanceId=instance_id,
            )
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code"))
            if code == "InvocationDoesNotExist":
                sleep(1)
                continue
            raise
        status = invocation.get("Status")
        if status in {"Pending", "InProgress", "Delayed"}:
            sleep(1)
            continue
        if status != "Success":
            raise ValueError(f"SkyPilot cancellation SSM command failed: {status}")
        prefix = "GLM52_MUST_START_RESULT="
        lines = [
            line
            for line in str(invocation.get("StandardOutputContent", "")).splitlines()
            if line.startswith(prefix)
        ]
        if len(lines) != 1:
            raise ValueError("SkyPilot cancellation result is missing or ambiguous")
        try:
            value = json.loads(lines[0][len(prefix) :])
        except json.JSONDecodeError as error:
            raise ValueError("SkyPilot cancellation result is not JSON") from error
        return _validate_result(
            value,
            sky_job_name=sky_job_name,
            expected_target_job_id=target_job_id,
            discovered_target=target_job_id is None,
        )
    raise TimeoutError("SkyPilot cancellation SSM command did not finish")


def _alert(sns, *, topic_arn: str, subject: str, message: Mapping[str, object]) -> None:  # noqa: ANN001
    sns.publish(
        TopicArn=topic_arn,
        Subject=subject[:100],
        Message=json.dumps(
            message,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ),
    )


def _load_exact_latch(
    s3,  # noqa: ANN001
    *,
    config: MustStartCoordinatorConfig,
    authority: Mapping[str, object],
) -> dict[str, object] | None:
    latch_value = _read_json(s3, bucket=config.bucket, key=config.latch_key)
    if latch_value is None:
        return None
    latch = validate_timely_start_latch(latch_value[0])
    try:
        _assert_exact_authority(latch, authority, label="timely-start latch")
    except ValueError:
        return None
    deadline = _parse_time(latch["must_start_by"], field="latch must_start_by")
    started = _parse_time(latch["started_at"], field="latch started_at")
    published = _parse_time(latch["published_at"], field="latch published_at")
    return latch if started <= deadline and published <= deadline else None


def _ensure_alert(
    *,
    s3,  # noqa: ANN001
    sns,  # noqa: ANN001
    config: MustStartCoordinatorConfig,
    authority: Mapping[str, object],
    alert_kind: str,
    lifecycle_body_sha256: str,
    now: datetime,
    subject: str,
    message: Mapping[str, object],
) -> None:
    key = (
        config.request_alerted_key
        if alert_kind == "requested"
        else config.completion_alerted_key
    )
    existing_value = _read_json(s3, bucket=config.bucket, key=key)
    if existing_value is not None:
        existing = validate_must_start_alert_delivered(existing_value[0])
        _assert_exact_authority(existing, authority, label="alert delivery marker")
        if (
            existing["alert_kind"] != alert_kind
            or existing["lifecycle_body_sha256"] != lifecycle_body_sha256
        ):
            raise ValueError("alert delivery marker lifecycle authority mismatch")
        return
    _alert(sns, topic_arn=config.alert_topic_arn, subject=subject, message=message)
    delivered = build_must_start_alert_delivered(
        **authority,
        alert_kind=alert_kind,
        lifecycle_body_sha256=lifecycle_body_sha256,
        delivered_at=now,
    )
    _conditional_put(
        s3,
        bucket=config.bucket,
        key=key,
        marker=delivered,
        validator=validate_must_start_alert_delivered,
    )


def _object_metadata(s3, *, bucket: str, key: str) -> dict[str, object]:  # noqa: ANN001
    value = s3.head_object(Bucket=bucket, Key=key)
    modified = value.get("LastModified")
    if not isinstance(modified, datetime) or modified.tzinfo is None:
        raise ValueError("S3 object LastModified is missing or invalid")
    etag = value.get("ETag")
    if not isinstance(etag, str) or not etag:
        raise ValueError("S3 object ETag is missing or invalid")
    version_id = value.get("VersionId")
    if version_id is not None and not isinstance(version_id, str):
        raise ValueError("S3 object VersionId is invalid")
    return {
        "last_modified": modified.astimezone(timezone.utc),
        "etag": etag,
        "version_id": version_id,
    }


def _publish_exact_observation(
    *,
    s3,  # noqa: ANN001
    config: MustStartCoordinatorConfig,
    authority: Mapping[str, object],
    controller: Mapping[str, str],
    result: Mapping[str, object],
) -> tuple[dict[str, object], str, dict[str, object]]:
    observation = build_must_start_controller_observation(
        **authority,
        target_job_id=int(result["target_job_id"]),
        workspace=str(result["target_workspace"]),
        **controller,
        status=str(result["status"]),
        schedule_state=str(result["schedule_state"]),
        submitted_at=str(result["submitted_at"]),
        start_at=(
            None if result["start_at"] is None else str(result["start_at"])
        ),
        worker_cluster_name=(
            None
            if result["worker_cluster_name"] is None
            else str(result["worker_cluster_name"])
        ),
        recovery_count=int(result["recovery_count"]),
        observed_at=str(result["observed_at"]),
    )
    key = config.observation_key(str(observation["observation_body_sha256"]))
    _conditional_put(
        s3,
        bucket=config.bucket,
        key=key,
        marker=observation,
        validator=validate_must_start_controller_observation,
    )
    return observation, key, _object_metadata(s3, bucket=config.bucket, key=key)


def _ensure_exact_job_binding(
    *,
    s3,  # noqa: ANN001
    config: MustStartCoordinatorConfig,
    authority: Mapping[str, object],
    controller: Mapping[str, str],
    observation: Mapping[str, object],
    submission: Mapping[str, object],
) -> dict[str, object]:
    expected_static = {
        **authority,
        "descriptor_key": config.descriptor_key,
        "descriptor_file_sha256": config.expected_descriptor_file_sha256,
        "submission_key": config.submission_key,
        "submission_submitted_at": _parse_time(
            submission["submitted_at"], field="submission submitted_at"
        )
        .isoformat()
        .replace("+00:00", "Z"),
        "target_job_id": config.expected_target_job_id,
        "workspace": config.expected_workspace,
        **controller,
    }
    existing_value = _read_json(
        s3,
        bucket=config.bucket,
        key=config.job_binding_key,
    )
    if existing_value is not None:
        existing = validate_must_start_job_binding(existing_value[0])
        _assert_exact_authority(existing, expected_static, label="job binding")
        return existing
    binding = build_must_start_job_binding(
        **authority,
        descriptor_key=config.descriptor_key,
        descriptor_file_sha256=str(config.expected_descriptor_file_sha256),
        submission_key=config.submission_key,
        submission_submitted_at=submission["submitted_at"],
        target_job_id=int(config.expected_target_job_id),
        workspace=str(config.expected_workspace),
        **controller,
        observation_body_sha256=str(
            observation["observation_body_sha256"]
        ),
        bound_at=str(observation["observed_at"]),
    )
    _conditional_put(
        s3,
        bucket=config.bucket,
        key=config.job_binding_key,
        marker=binding,
        validator=validate_must_start_job_binding,
        winner_matches=lambda winner: all(
            winner.get(field) == expected
            for field, expected in expected_static.items()
        ),
    )
    winner_value = _read_json(
        s3,
        bucket=config.bucket,
        key=config.job_binding_key,
    )
    if winner_value is None:
        raise ValueError("job binding disappeared after conditional write")
    winner = validate_must_start_job_binding(winner_value[0])
    _assert_exact_authority(winner, expected_static, label="job binding")
    return winner


def _load_exact_accepted(
    *,
    s3,  # noqa: ANN001
    config: MustStartCoordinatorConfig,
    authority: Mapping[str, object],
) -> dict[str, object] | None:
    value = _read_json(s3, bucket=config.bucket, key=config.accepted_key)
    if value is None:
        return None
    accepted = validate_timely_start_accepted(value[0])
    _assert_exact_authority(accepted, authority, label="accepted start")
    if accepted["source_kind"] != "controller-observation":
        raise ValueError(
            "current-job accepted start source must be a controller observation"
        )
    source_key = str(accepted["source_key"])
    source_value = _read_json(
        s3,
        bucket=config.bucket,
        key=source_key,
    )
    if source_value is None:
        raise ValueError("accepted start source observation is missing")
    observation = validate_must_start_controller_observation(source_value[0])
    _assert_exact_authority(
        observation,
        authority,
        label="accepted source observation",
    )
    expected_controller = {
        "target_job_id": config.expected_target_job_id,
        "workspace": config.expected_workspace,
        "controller_instance_id": config.expected_controller_instance_id,
        "controller_instance_type": config.expected_controller_instance_type,
        "controller_profile_arn": config.expected_controller_profile_arn,
        "controller_cluster_name": config.expected_controller_cluster_name,
    }
    _assert_exact_authority(
        observation,
        expected_controller,
        label="accepted source controller",
    )
    expected_source_key = config.observation_key(
        str(observation["observation_body_sha256"])
    )
    if source_key != expected_source_key:
        raise ValueError("accepted start source key is not content addressed")
    if observation["start_at"] != accepted["started_at"]:
        raise ValueError("accepted start does not bind source start_at")
    source_decision = decide_controller_start_authority(
        now=_parse_time(accepted["accepted_at"], field="accepted_at"),
        must_start_by=_parse_time(config.must_start_by, field="must_start_by"),
        status=str(observation["status"]),
        start_at=_parse_time(observation["start_at"], field="source start_at"),
        starting_grace_seconds=config.starting_grace_seconds,
    )
    if source_decision.action != "accept":
        raise ValueError("accepted start source does not authorize acceptance")
    source_metadata = _object_metadata(
        s3,
        bucket=config.bucket,
        key=source_key,
    )
    source_last_modified = source_metadata["last_modified"]
    if not isinstance(source_last_modified, datetime):
        raise ValueError("accepted start source LastModified is invalid")
    expected_metadata = {
        "source_version_id": source_metadata.get("version_id"),
        "source_etag": source_metadata["etag"],
        "source_body_sha256": hashlib.sha256(
            canonical_bytes(observation)
        ).hexdigest(),
        "source_last_modified": (
            source_last_modified.astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        ),
    }
    for field, expected in expected_metadata.items():
        if accepted.get(field) != expected:
            raise ValueError(f"accepted start {field} source mismatch")
    return accepted


def _disable_exact_reconciliation(
    events,  # noqa: ANN001
    *,
    config: MustStartCoordinatorConfig,
) -> None:
    if events is None or config.reconciliation_rule_name is None:
        return
    events.disable_rule(Name=config.reconciliation_rule_name)


def _publish_exact_accepted(
    *,
    s3,  # noqa: ANN001
    config: MustStartCoordinatorConfig,
    authority: Mapping[str, object],
    observation: Mapping[str, object],
    observation_key: str,
    observation_metadata: Mapping[str, object],
    now: datetime,
) -> dict[str, object]:
    existing = _load_exact_accepted(
        s3=s3,
        config=config,
        authority=authority,
    )
    if existing is not None:
        return existing
    raw_observation = json.dumps(
        observation,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    last_modified = observation_metadata["last_modified"]
    if not isinstance(last_modified, datetime):
        raise ValueError("observation LastModified is invalid")
    accepted_at = max(
        now.astimezone(timezone.utc),
        last_modified.astimezone(timezone.utc),
    )
    accepted = build_timely_start_accepted(
        **authority,
        source_kind="controller-observation",
        source_key=observation_key,
        source_version_id=observation_metadata.get("version_id"),  # type: ignore[arg-type]
        source_etag=str(observation_metadata["etag"]),
        source_body_sha256=hashlib.sha256(raw_observation).hexdigest(),
        source_last_modified=last_modified,
        started_at=str(observation["start_at"]),
        accepted_at=accepted_at,
    )
    _conditional_put(
        s3,
        bucket=config.bucket,
        key=config.accepted_key,
        marker=accepted,
        validator=validate_timely_start_accepted,
        winner_matches=lambda winner: all(
            winner.get(field) == expected
            for field, expected in authority.items()
        ),
    )
    winner = _load_exact_accepted(
        s3=s3,
        config=config,
        authority=authority,
    )
    if winner is None:
        raise ValueError("accepted start disappeared after conditional write")
    return winner


def _publish_superseded_if_requested(
    *,
    s3,  # noqa: ANN001
    config: MustStartCoordinatorConfig,
    authority: Mapping[str, object],
    request: Mapping[str, object] | None,
    accepted: Mapping[str, object],
    now: datetime,
) -> None:
    if request is None:
        return
    marker = build_must_start_cancel_superseded(
        **authority,
        request_body_sha256=str(request["request_body_sha256"]),
        accepted_body_sha256=str(accepted["accepted_body_sha256"]),
        superseded_at=now,
    )
    _conditional_put(
        s3,
        bucket=config.bucket,
        key=config.superseded_key,
        marker=marker,
        validator=validate_must_start_cancel_superseded,
        winner_matches=lambda winner: (
            winner.get("request_body_sha256")
            == request["request_body_sha256"]
            and winner.get("accepted_body_sha256")
            == accepted["accepted_body_sha256"]
        ),
    )


def _load_exact_request(
    *,
    s3,  # noqa: ANN001
    config: MustStartCoordinatorConfig,
    authority: Mapping[str, object],
) -> dict[str, object] | None:
    value = _read_json(s3, bucket=config.bucket, key=config.request_key)
    if value is None:
        return None
    request = validate_must_start_cancel_requested(value[0])
    _assert_exact_authority(request, authority, label="request marker")
    return request


def _load_exact_completion(
    *,
    s3,  # noqa: ANN001
    config: MustStartCoordinatorConfig,
    authority: Mapping[str, object],
    request: Mapping[str, object] | None,
) -> dict[str, object] | None:
    value = _read_json(s3, bucket=config.bucket, key=config.completion_key)
    if value is None:
        return None
    completion = validate_must_start_cancel_completed(value[0])
    _assert_exact_authority(completion, authority, label="completion marker")
    if (
        request is None
        or completion["request_body_sha256"] != request["request_body_sha256"]
    ):
        raise ValueError("completion marker does not bind exact request")
    return completion


def _ensure_exact_request(
    *,
    s3,  # noqa: ANN001
    config: MustStartCoordinatorConfig,
    authority: Mapping[str, object],
    now: datetime,
) -> dict[str, object]:
    existing = _load_exact_request(
        s3=s3,
        config=config,
        authority=authority,
    )
    if existing is not None:
        return existing
    request = build_must_start_cancel_requested(
        **authority,
        requested_at=now,
    )
    _conditional_put(
        s3,
        bucket=config.bucket,
        key=config.request_key,
        marker=request,
        validator=validate_must_start_cancel_requested,
        winner_matches=lambda winner: all(
            winner.get(field) == expected
            for field, expected in authority.items()
        ),
    )
    winner = _load_exact_request(
        s3=s3,
        config=config,
        authority=authority,
    )
    if winner is None:
        raise ValueError("request marker disappeared after conditional write")
    return winner


def _load_exact_activation_proof(
    *,
    s3,  # noqa: ANN001
    config: MustStartCoordinatorConfig,
    authority: Mapping[str, object],
    submission: Mapping[str, object],
) -> None:
    expected_binding_sha = config.expected_activation_job_binding_sha256
    expected_observation_sha = config.expected_activation_observation_sha256
    if expected_binding_sha is None and expected_observation_sha is None:
        return
    binding_value = _read_json(
        s3,
        bucket=config.bucket,
        key=config.job_binding_key,
    )
    if binding_value is None:
        raise ValueError("activation JOB_BINDING.json is missing")
    binding = validate_must_start_job_binding(binding_value[0])
    expected_binding = {
        **authority,
        "descriptor_key": config.descriptor_key,
        "descriptor_file_sha256": config.expected_descriptor_file_sha256,
        "submission_key": config.submission_key,
        "submission_submitted_at": _parse_time(
            submission["submitted_at"], field="submission submitted_at"
        )
        .isoformat()
        .replace("+00:00", "Z"),
        "target_job_id": config.expected_target_job_id,
        "workspace": config.expected_workspace,
        "controller_instance_id": config.expected_controller_instance_id,
        "controller_instance_type": config.expected_controller_instance_type,
        "controller_profile_arn": config.expected_controller_profile_arn,
        "controller_cluster_name": config.expected_controller_cluster_name,
        "observation_body_sha256": expected_observation_sha,
        "job_binding_body_sha256": expected_binding_sha,
    }
    _assert_exact_authority(
        binding,
        expected_binding,
        label="activation job binding",
    )
    observation_value = _read_json(
        s3,
        bucket=config.bucket,
        key=config.observation_key(str(expected_observation_sha)),
    )
    if observation_value is None:
        raise ValueError("content-addressed activation observation is missing")
    observation = validate_must_start_controller_observation(
        observation_value[0]
    )
    for field in (
        "run_id",
        "managed_mode",
        "account_id",
        "region",
        "bucket",
        "descriptor_body_sha256",
        "submission_body_sha256",
        "sky_job_name",
        "must_start_by",
        "target_job_id",
        "workspace",
        "controller_instance_id",
        "controller_instance_type",
        "controller_profile_arn",
        "controller_cluster_name",
    ):
        if observation.get(field) != binding.get(field):
            raise ValueError(f"activation observation {field} mismatch")
    if observation.get("observation_body_sha256") != expected_observation_sha:
        raise ValueError("activation observation SHA-256 mismatch")
    if observation.get("observed_at") != binding.get("bound_at"):
        raise ValueError("activation observation time does not bind job binding")


def _coordinate_exact_must_start(
    *,
    config: MustStartCoordinatorConfig,
    now: datetime,
    descriptor_file_sha: str,
    submission: Mapping[str, object],
    authority: Mapping[str, object],
    s3,  # noqa: ANN001
    ec2,  # noqa: ANN001
    ssm,  # noqa: ANN001
    sns,  # noqa: ANN001
    events,  # noqa: ANN001
    sleep: Callable[[float], None],
) -> dict[str, object]:
    if descriptor_file_sha != config.expected_descriptor_file_sha256:
        raise ValueError("descriptor file SHA-256 authority mismatch")
    submitted_at = (
        _parse_time(submission["submitted_at"], field="submission submitted_at")
        .isoformat()
        .replace("+00:00", "Z")
    )
    configured_submitted_at = (
        _parse_time(
            config.expected_submission_submitted_at,
            field="configured submission submitted_at",
        )
        .isoformat()
        .replace("+00:00", "Z")
    )
    if submitted_at != configured_submitted_at:
        raise ValueError("submission submitted_at authority mismatch")
    deadline = _parse_time(config.must_start_by, field="must_start_by")

    def decide(candidate: Mapping[str, object], clock: datetime):
        return decide_controller_start_authority(
            now=clock,
            must_start_by=deadline,
            status=str(candidate["status"]),
            start_at=(
                None
                if candidate["start_at"] is None
                else _parse_time(candidate["start_at"], field="controller start_at")
            ),
            starting_grace_seconds=config.starting_grace_seconds,
        )

    if not config.observe_only:
        _load_exact_activation_proof(
            s3=s3,
            config=config,
            authority=authority,
            submission=submission,
        )
    accepted = _load_exact_accepted(
        s3=s3,
        config=config,
        authority=authority,
    )
    if accepted is not None:
        _resolve_exact_controller(ec2, ssm, config=config)
        fresh_accepted_source = _run_exact_controller(
            ssm,
            config=config,
            operation="observe",
            now=now,
            sleep=sleep,
        )
        fresh_observed_at = _parse_time(
            fresh_accepted_source["observed_at"],
            field="controller observed_at",
        )
        fresh_decision = decide(
            fresh_accepted_source,
            max(now.astimezone(timezone.utc), fresh_observed_at),
        )
        if (
            fresh_decision.action != "accept"
            or fresh_accepted_source["start_at"] != accepted["started_at"]
        ):
            raise ValueError(
                "accepted start fresh controller evidence mismatch"
            )
        _disable_exact_reconciliation(events, config=config)
        return {
            "status": "timely-started",
            "run_id": config.run_id,
            "accepted_body_sha256": accepted["accepted_body_sha256"],
        }
    request = _load_exact_request(
        s3=s3,
        config=config,
        authority=authority,
    )
    completion = _load_exact_completion(
        s3=s3,
        config=config,
        authority=authority,
        request=request,
    )
    if completion is not None:
        _resolve_exact_controller(ec2, ssm, config=config)
        fresh_completion_source = _run_exact_controller(
            ssm,
            config=config,
            operation="observe",
            now=now,
            sleep=sleep,
        )
        if (
            fresh_completion_source["status"] not in _TERMINAL_STATUSES
            or fresh_completion_source["status"]
            != completion["terminal_status"]
        ):
            raise ValueError(
                "completion terminal controller evidence mismatch"
            )
        if config.observe_only:
            return {
                "status": "observed",
                "run_id": config.run_id,
                "target_job_id": fresh_completion_source["target_job_id"],
                "controller_status": fresh_completion_source["status"],
                "decision": "completed",
            }
        _ensure_alert(
            s3=s3,
            sns=sns,
            config=config,
            authority=authority,
            alert_kind="completed",
            lifecycle_body_sha256=str(completion["completion_body_sha256"]),
            now=now,
            subject="KEEP GLM52 must-start cancellation completed",
            message={
                "outcome": "completed",
                "run_id": config.run_id,
                "managed_mode": config.managed_mode,
                "sky_job_name": config.sky_job_name,
                "target_job_id": config.expected_target_job_id,
                "terminal_status": completion["terminal_status"],
                "completion_body_sha256": completion[
                    "completion_body_sha256"
                ],
            },
        )
        _disable_exact_reconciliation(events, config=config)
        return {
            "status": "completed",
            "run_id": config.run_id,
            "completion_body_sha256": completion["completion_body_sha256"],
        }
    controller = _resolve_exact_controller(ec2, ssm, config=config)
    result = _run_exact_controller(
        ssm,
        config=config,
        operation="observe",
        now=now,
        sleep=sleep,
    )
    observation, observation_key, observation_metadata = (
        _publish_exact_observation(
            s3=s3,
            config=config,
            authority=authority,
            controller=controller,
            result=result,
        )
    )
    _log_stage("post-observation")
    _log_stage("before-binding")
    _ensure_exact_job_binding(
        s3=s3,
        config=config,
        authority=authority,
        controller=controller,
        observation=observation,
        submission=submission,
    )
    _log_stage("after-binding")

    observed_at = _parse_time(result["observed_at"], field="controller observed_at")
    decision = decide(result, max(now.astimezone(timezone.utc), observed_at))
    if decision.action == "accept":
        accepted = _publish_exact_accepted(
            s3=s3,
            config=config,
            authority=authority,
            observation=observation,
            observation_key=observation_key,
            observation_metadata=observation_metadata,
            now=now,
        )
        _publish_superseded_if_requested(
            s3=s3,
            config=config,
            authority=authority,
            request=request,
            accepted=accepted,
            now=now,
        )
        _disable_exact_reconciliation(events, config=config)
        return {
            "status": "timely-started",
            "run_id": config.run_id,
            "accepted_body_sha256": accepted["accepted_body_sha256"],
        }
    if config.observe_only:
        return {
            "status": "observed",
            "run_id": config.run_id,
            "target_job_id": result["target_job_id"],
            "controller_status": result["status"],
            "decision": decision.action,
        }
    if decision.action == "grace":
        grace_deadline = deadline + timedelta(
            seconds=config.starting_grace_seconds
        )
        remaining = max(0.0, (grace_deadline - observed_at).total_seconds())
        sleep(remaining)
        recheck_clock = grace_deadline + timedelta(microseconds=1)
        result = _run_exact_controller(
            ssm,
            config=config,
            operation="observe",
            now=recheck_clock,
            sleep=sleep,
        )
        observation, observation_key, observation_metadata = (
            _publish_exact_observation(
                s3=s3,
                config=config,
                authority=authority,
                controller=controller,
                result=result,
            )
        )
        _log_stage("post-observation")
        _log_stage("before-binding")
        _ensure_exact_job_binding(
            s3=s3,
            config=config,
            authority=authority,
            controller=controller,
            observation=observation,
            submission=submission,
        )
        _log_stage("after-binding")
        decision = decide(result, recheck_clock)
        if decision.action == "accept":
            accepted = _publish_exact_accepted(
                s3=s3,
                config=config,
                authority=authority,
                observation=observation,
                observation_key=observation_key,
                observation_metadata=observation_metadata,
                now=recheck_clock,
            )
            _publish_superseded_if_requested(
                s3=s3,
                config=config,
                authority=authority,
                request=request,
                accepted=accepted,
                now=recheck_clock,
            )
            _disable_exact_reconciliation(events, config=config)
            return {
                "status": "timely-started",
                "run_id": config.run_id,
                "accepted_body_sha256": accepted["accepted_body_sha256"],
            }
    if decision.action == "observe":
        return {"status": "waiting", "run_id": config.run_id}
    if decision.action not in {"cancel"}:
        raise ValueError(f"exact controller evidence failed closed: {decision.reason}")
    request_time = max(now.astimezone(timezone.utc), deadline)
    request = _ensure_exact_request(
        s3=s3,
        config=config,
        authority=authority,
        now=request_time,
    )
    _ensure_alert(
        s3=s3,
        sns=sns,
        config=config,
        authority=authority,
        alert_kind="requested",
        lifecycle_body_sha256=str(request["request_body_sha256"]),
        now=request_time,
        subject="KEEP GLM52 must-start cancellation requested",
        message={
            "outcome": "requested",
            "run_id": config.run_id,
            "managed_mode": config.managed_mode,
            "sky_job_name": config.sky_job_name,
            "target_job_id": config.expected_target_job_id,
            "request_body_sha256": request["request_body_sha256"],
        },
    )
    cancel_result = _run_exact_controller(
        ssm,
        config=config,
        operation="cancel",
        now=max(request_time, deadline),
        sleep=sleep,
    )
    cancel_observation, cancel_key, cancel_metadata = (
        _publish_exact_observation(
            s3=s3,
            config=config,
            authority=authority,
            controller=controller,
            result=cancel_result,
        )
    )
    if not cancel_result["cancel_called"]:
        reread_decision = decide(
            cancel_result,
            max(request_time, deadline),
        )
        if reread_decision.action == "accept":
            accepted = _publish_exact_accepted(
                s3=s3,
                config=config,
                authority=authority,
                observation=cancel_observation,
                observation_key=cancel_key,
                observation_metadata=cancel_metadata,
                now=request_time,
            )
            _publish_superseded_if_requested(
                s3=s3,
                config=config,
                authority=authority,
                request=request,
                accepted=accepted,
                now=request_time,
            )
            _disable_exact_reconciliation(events, config=config)
            return {
                "status": "timely-started",
                "run_id": config.run_id,
                "accepted_body_sha256": accepted["accepted_body_sha256"],
            }
        raise ValueError("exact cancellation reread did not authorize cancellation")
    post_status = cancel_result.get("post_status")
    if post_status not in _TERMINAL_STATUSES:
        return {
            "status": "requested",
            "run_id": config.run_id,
            "controller_outcome": "nonterminal",
            "request_body_sha256": request["request_body_sha256"],
        }
    completed = build_must_start_cancel_completed(
        **authority,
        request_body_sha256=str(request["request_body_sha256"]),
        completed_at=str(cancel_result["observed_at"]),
        terminal_status=str(post_status),
    )
    _conditional_put(
        s3,
        bucket=config.bucket,
        key=config.completion_key,
        marker=completed,
        validator=validate_must_start_cancel_completed,
    )
    _ensure_alert(
        s3=s3,
        sns=sns,
        config=config,
        authority=authority,
        alert_kind="completed",
        lifecycle_body_sha256=str(completed["completion_body_sha256"]),
        now=request_time,
        subject="KEEP GLM52 must-start cancellation completed",
        message={
            "outcome": "completed",
            "run_id": config.run_id,
            "managed_mode": config.managed_mode,
            "sky_job_name": config.sky_job_name,
            "target_job_id": config.expected_target_job_id,
            "terminal_status": post_status,
            "completion_body_sha256": completed["completion_body_sha256"],
        },
    )
    _disable_exact_reconciliation(events, config=config)
    return {
        "status": "completed",
        "run_id": config.run_id,
        "completion_body_sha256": completed["completion_body_sha256"],
    }


def coordinate_must_start(
    *,
    config: MustStartCoordinatorConfig,
    runtime_account_id: str,
    runtime_region: str,
    now: datetime,
    trigger: str = "reconcile",
    s3,  # noqa: ANN001
    ec2,  # noqa: ANN001
    ssm,  # noqa: ANN001
    sns,  # noqa: ANN001
    events=None,  # noqa: ANN001
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    if runtime_account_id != config.account_id:
        raise ValueError("runtime AWS account authority mismatch")
    if runtime_region != config.region:
        raise ValueError("runtime AWS region authority mismatch")
    if now.tzinfo is None:
        raise ValueError("coordinator clock must be timezone-aware")
    if trigger not in {"reconcile", "primary-deadline"}:
        raise ValueError("must-start trigger is invalid")
    if config.observe_only and not config.strict_controller_authority:
        raise ValueError(
            "observe-only requires complete exact controller authority"
        )

    descriptor_value = _read_json(
        s3, bucket=config.bucket, key=config.descriptor_key
    )
    if descriptor_value is None:
        raise ValueError("configured campaign descriptor is missing")
    descriptor = validate_sky_campaign_descriptor(descriptor_value[0])
    descriptor_file_sha = hashlib.sha256(descriptor_value[1]).hexdigest()
    expected_submission_key = (
        f"campaigns/{config.run_id}/monitor/submission-locks/"
        f"{descriptor_file_sha}-{config.managed_mode}.json"
    )
    if config.submission_key != expected_submission_key:
        raise ValueError("immutable submission key does not bind descriptor bytes")
    submission_value = _read_json(
        s3, bucket=config.bucket, key=config.submission_key
    )
    if submission_value is None:
        raise ValueError("configured immutable submission is missing")
    submission = validate_skypilot_submission_marker(submission_value[0])
    if submission["submission_body_sha256"] != config.submission_body_sha256:
        raise ValueError("configured submission body SHA-256 authority mismatch")
    deadline = _parse_time(config.must_start_by, field="configured must_start_by")
    descriptor_deadline = _parse_time(
        descriptor["must_start_by"], field="descriptor must_start_by"
    )
    submission_deadline = _parse_time(
        submission["must_start_by"], field="submission must_start_by"
    )
    if not (
        descriptor["account_id"] == config.account_id
        and descriptor["region"] == config.region
        and descriptor["bucket"] == config.bucket
        and descriptor["campaign_descriptor_key"] == config.descriptor_key
        and descriptor["run_id"] == config.run_id
        and submission["run_id"] == config.run_id
        and submission["descriptor_body_sha256"]
        == descriptor["descriptor_body_sha256"]
        and submission["sky_job_name"] == config.sky_job_name
        and config.sky_job_name
        == expected_sky_job_name(config.run_id, config.managed_mode)
        and descriptor_deadline == deadline
        and submission_deadline == deadline
    ):
        raise ValueError("descriptor/submission/runtime authority mismatch")

    authority: dict[str, object] = {
        "run_id": config.run_id,
        "managed_mode": config.managed_mode,
        "account_id": config.account_id,
        "region": config.region,
        "bucket": config.bucket,
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "submission_body_sha256": submission["submission_body_sha256"],
        "sky_job_name": config.sky_job_name,
        "must_start_by": deadline.isoformat().replace("+00:00", "Z"),
    }
    effective_now = now.astimezone(timezone.utc)
    if trigger == "primary-deadline" and effective_now < deadline:
        remaining = (deadline - effective_now).total_seconds()
        if remaining > config.primary_wake_max_wait_seconds:
            raise ValueError("primary deadline wake is too early")
        sleep(remaining)
        effective_now = deadline

    if config.strict_controller_authority:
        return _coordinate_exact_must_start(
            config=config,
            now=effective_now,
            descriptor_file_sha=descriptor_file_sha,
            submission=submission,
            authority=authority,
            s3=s3,
            ec2=ec2,
            ssm=ssm,
            sns=sns,
            events=events,
            sleep=sleep,
        )

    latch = _load_exact_latch(s3, config=config, authority=authority)

    request_value = _read_json(s3, bucket=config.bucket, key=config.request_key)
    request: dict[str, object] | None = None
    if request_value is not None:
        request = validate_must_start_cancel_requested(request_value[0])
        _assert_exact_authority(request, authority, label="request marker")

    completion_value = _read_json(
        s3, bucket=config.bucket, key=config.completion_key
    )
    completion: dict[str, object] | None = None
    if completion_value is not None:
        completion = validate_must_start_cancel_completed(completion_value[0])
        _assert_exact_authority(completion, authority, label="completion marker")
        if request is None or completion["request_body_sha256"] != request[
            "request_body_sha256"
        ]:
            raise ValueError("completion marker does not bind exact request")

    initial = decide_must_start_action(
        now=effective_now,
        must_start_by=deadline,
        expected_authority=authority,
        timely_start_latch=latch,
        request_marker=request,
        completion_marker=completion,
        controller_outcome=None,
    )
    if initial.action == "wait":
        return {"status": "waiting", "run_id": config.run_id}
    if initial.action == "timely-started":
        return {"status": "timely-started", "run_id": config.run_id}
    if initial.action == "completed":
        _ensure_alert(
            s3=s3,
            sns=sns,
            config=config,
            authority=authority,
            alert_kind="completed",
            lifecycle_body_sha256=str(completion["completion_body_sha256"]),
            now=now,
            subject="KEEP GLM52 must-start cancellation completed",
            message={
                "outcome": "completed",
                "run_id": config.run_id,
                "managed_mode": config.managed_mode,
                "sky_job_name": config.sky_job_name,
                "terminal_status": completion["terminal_status"],
                "completion_body_sha256": completion["completion_body_sha256"],
            },
        )
        return {
            "status": "completed",
            "run_id": config.run_id,
            "completion_body_sha256": completion["completion_body_sha256"],
        }

    if request is None:
        request = build_must_start_cancel_requested(
            **authority,
            requested_at=now,
        )
        _conditional_put(
            s3,
            bucket=config.bucket,
            key=config.request_key,
            marker=request,
            validator=validate_must_start_cancel_requested,
            winner_matches=lambda winner: all(
                winner.get(field) == expected
                for field, expected in authority.items()
            ),
        )
        request_value = _read_json(s3, bucket=config.bucket, key=config.request_key)
        if request_value is None:
            raise ValueError("request marker disappeared after conditional write")
        request = validate_must_start_cancel_requested(request_value[0])
        _assert_exact_authority(request, authority, label="request marker")

    _ensure_alert(
        s3=s3,
        sns=sns,
        config=config,
        authority=authority,
        alert_kind="requested",
        lifecycle_body_sha256=str(request["request_body_sha256"]),
        now=now,
        subject="KEEP GLM52 must-start cancellation requested",
        message={
            "outcome": "requested",
            "run_id": config.run_id,
            "managed_mode": config.managed_mode,
            "sky_job_name": config.sky_job_name,
            "request_body_sha256": request["request_body_sha256"],
        },
    )
    if _load_exact_latch(s3, config=config, authority=authority) is not None:
        return {"status": "timely-started", "run_id": config.run_id}

    binding_value = _read_json(
        s3, bucket=config.bucket, key=config.target_binding_key
    )
    binding: dict[str, object] | None = None
    if binding_value is not None:
        binding = validate_must_start_target_binding(binding_value[0])
        _assert_exact_authority(binding, authority, label="target binding")
        if binding["request_body_sha256"] != request["request_body_sha256"]:
            raise ValueError("target binding does not bind exact request")
    elif config.expected_target_job_id is not None:
        binding = build_must_start_target_binding(
            **authority,
            request_body_sha256=str(request["request_body_sha256"]),
            target_job_id=config.expected_target_job_id,
                bound_at=now,
        )
        _conditional_put(
            s3,
            bucket=config.bucket,
            key=config.target_binding_key,
            marker=binding,
            validator=validate_must_start_target_binding,
        )

    try:
        controller_id = _find_controller(ec2, run_id=config.run_id)
        result = _run_controller_cancel(
            ssm,
            instance_id=controller_id,
            sky_job_name=config.sky_job_name,
            sleep=sleep,
            target_job_id=(
                int(binding["target_job_id"]) if binding is not None else None
            ),
        )
        if binding is None and result["target_job_id"] is not None:
            binding = build_must_start_target_binding(
                **authority,
                request_body_sha256=str(request["request_body_sha256"]),
                target_job_id=int(result["target_job_id"]),
                bound_at=str(result["observed_at"]),
            )
            _conditional_put(
                s3,
                bucket=config.bucket,
                key=config.target_binding_key,
                marker=binding,
                validator=validate_must_start_target_binding,
            )
        if _load_exact_latch(s3, config=config, authority=authority) is not None:
            return {"status": "timely-started", "run_id": config.run_id}
    except Exception as error:
        try:
            _alert(
                sns,
                topic_arn=config.alert_topic_arn,
                subject="KEEP GLM52 must-start cancellation failed closed",
                message={
                    "outcome": "fail-closed",
                    "run_id": config.run_id,
                    "managed_mode": config.managed_mode,
                    "sky_job_name": config.sky_job_name,
                    "error": str(error),
                },
            )
            setattr(error, "_glm52_must_start_alerted", True)
        except Exception:
            pass
        raise

    if result["outcome"] == "not_found":
        return {
            "status": "requested",
            "run_id": config.run_id,
            "controller_outcome": "not_found",
            "request_body_sha256": request["request_body_sha256"],
        }
    if result["outcome"] == "nonterminal":
        return {
            "status": "requested",
            "run_id": config.run_id,
            "controller_outcome": "nonterminal",
            "request_body_sha256": request["request_body_sha256"],
        }

    if result["outcome"] == "ambiguous":
        raise ValueError("SkyPilot exact-name job state is ambiguous")
    terminal_status = str(result["target_status"])
    completed = build_must_start_cancel_completed(
        **authority,
        request_body_sha256=str(request["request_body_sha256"]),
        completed_at=str(result["observed_at"]),
        terminal_status=terminal_status,
    )
    _conditional_put(
        s3,
        bucket=config.bucket,
        key=config.completion_key,
        marker=completed,
        validator=validate_must_start_cancel_completed,
    )
    _ensure_alert(
        s3=s3,
        sns=sns,
        config=config,
        authority=authority,
        alert_kind="completed",
        lifecycle_body_sha256=str(completed["completion_body_sha256"]),
            now=now,
        subject="KEEP GLM52 must-start cancellation completed",
        message={
            "outcome": "completed",
            "run_id": config.run_id,
            "managed_mode": config.managed_mode,
            "sky_job_name": config.sky_job_name,
            "terminal_status": terminal_status,
            "completion_body_sha256": completed["completion_body_sha256"],
        },
    )
    return {
        "status": "completed",
        "run_id": config.run_id,
        "completion_body_sha256": completed["completion_body_sha256"],
    }


def _config_from_environment() -> MustStartCoordinatorConfig:
    observe_only_value = os.environ.get("OBSERVE_ONLY", "true")
    if observe_only_value not in {"true", "false"}:
        raise ValueError("OBSERVE_ONLY must be true or false")
    controller_id = os.environ.get("EXPECTED_CONTROLLER_INSTANCE_ID")
    exact_controller_values = {
        "EXPECTED_TARGET_JOB_ID": os.environ.get("EXPECTED_TARGET_JOB_ID"),
        "EXPECTED_WORKSPACE": os.environ.get("EXPECTED_WORKSPACE"),
        "EXPECTED_DESCRIPTOR_FILE_SHA256": os.environ.get(
            "EXPECTED_DESCRIPTOR_FILE_SHA256"
        ),
        "EXPECTED_SUBMISSION_SUBMITTED_AT": os.environ.get(
            "EXPECTED_SUBMISSION_SUBMITTED_AT"
        ),
        "EXPECTED_CONTROLLER_INSTANCE_ID": controller_id,
        "EXPECTED_CONTROLLER_INSTANCE_TYPE": os.environ.get(
            "EXPECTED_CONTROLLER_INSTANCE_TYPE"
        ),
        "EXPECTED_CONTROLLER_PROFILE_ARN": os.environ.get(
            "EXPECTED_CONTROLLER_PROFILE_ARN"
        ),
        "EXPECTED_CONTROLLER_CLUSTER_NAME": os.environ.get(
            "EXPECTED_CONTROLLER_CLUSTER_NAME"
        ),
    }
    if any(
        value in {None, "", "disabled", "-1", "1970-01-01T00:00:00Z"}
        for value in exact_controller_values.values()
    ):
        raise ValueError(
            "coordinator requires complete exact controller authority"
        )
    strict_enabled = True
    activation_binding_sha = os.environ.get(
        "EXPECTED_ACTIVATION_JOB_BINDING_SHA256",
        "disabled",
    )
    activation_observation_sha = os.environ.get(
        "EXPECTED_ACTIVATION_OBSERVATION_SHA256",
        "disabled",
    )
    if strict_enabled and observe_only_value == "false" and (
        activation_binding_sha == "disabled"
        or activation_observation_sha == "disabled"
    ):
        raise ValueError(
            "active mode requires exact prior observation proof hashes"
        )
    return MustStartCoordinatorConfig(
        account_id=os.environ["EXPECTED_ACCOUNT_ID"],
        region=os.environ["AWS_REGION"],
        bucket=os.environ["CAMPAIGN_BUCKET"],
        descriptor_key=os.environ["CAMPAIGN_DESCRIPTOR_KEY"],
        submission_key=os.environ["IMMUTABLE_SUBMISSION_KEY"],
        submission_body_sha256=os.environ["SUBMISSION_BODY_SHA256"],
        run_id=os.environ["CAMPAIGN_RUN_ID"],
        managed_mode=os.environ["MANAGED_MODE"],
        sky_job_name=os.environ["SKY_JOB_NAME"],
        must_start_by=os.environ["MUST_START_BY"],
        alert_topic_arn=os.environ["ALERT_TOPIC_ARN"],
        expected_target_job_id=(
            None
            if os.environ.get("EXPECTED_TARGET_JOB_ID", "-1") == "-1"
            else int(os.environ["EXPECTED_TARGET_JOB_ID"])
        ),
        observe_only=observe_only_value == "true",
        expected_workspace=(
            os.environ["EXPECTED_WORKSPACE"] if strict_enabled else None
        ),
        expected_descriptor_file_sha256=(
            os.environ["EXPECTED_DESCRIPTOR_FILE_SHA256"]
            if strict_enabled
            else None
        ),
        expected_submission_submitted_at=(
            os.environ["EXPECTED_SUBMISSION_SUBMITTED_AT"]
            if strict_enabled
            else None
        ),
        expected_controller_instance_id=controller_id if strict_enabled else None,
        expected_controller_instance_type=(
            os.environ["EXPECTED_CONTROLLER_INSTANCE_TYPE"]
            if strict_enabled
            else None
        ),
        expected_controller_profile_arn=(
            os.environ["EXPECTED_CONTROLLER_PROFILE_ARN"]
            if strict_enabled
            else None
        ),
        expected_controller_cluster_name=(
            os.environ["EXPECTED_CONTROLLER_CLUSTER_NAME"]
            if strict_enabled
            else None
        ),
        starting_grace_seconds=int(
            os.environ.get("STARTING_GRACE_SECONDS", "20")
        ),
        reconciliation_rule_name=(
            os.environ["RECONCILIATION_RULE_NAME"]
            if strict_enabled
            else os.environ.get("RECONCILIATION_RULE_NAME")
        ),
        primary_wake_max_wait_seconds=int(
            os.environ.get("PRIMARY_WAKE_MAX_WAIT_SECONDS", "120")
        ),
        expected_activation_job_binding_sha256=(
            None
            if activation_binding_sha == "disabled"
            else activation_binding_sha
        ),
        expected_activation_observation_sha256=(
            None
            if activation_observation_sha == "disabled"
            else activation_observation_sha
        ),
    )


def lambda_handler(event, _context):  # noqa: ANN001
    config = _config_from_environment()
    if not isinstance(event, dict):
        raise ValueError("EventBridge must-start target input must be an object")
    trigger = event.get("trigger")
    expected_event = {
        "campaign_run_id": config.run_id,
        "managed_mode": config.managed_mode,
        "sky_job_name": config.sky_job_name,
        "must_start_by": config.must_start_by,
        "trigger": trigger,
    }
    session, sts = _runtime_client(region=config.region, service="sts")
    identity = sts.get_caller_identity()
    runtime_account_id = str(identity.get("Account"))
    runtime_region = str(session.region_name)
    if runtime_account_id != config.account_id or runtime_region != config.region:
        raise ValueError("runtime AWS authority mismatch")
    _, sns = _runtime_client(region=config.region, service="sns")
    try:
        if trigger not in {"reconcile", "primary-deadline"} or event != expected_event:
            raise ValueError(
                "EventBridge must-start target input authority mismatch"
            )
        _, s3 = _runtime_client(region=config.region, service="s3")
        _, ec2 = _runtime_client(region=config.region, service="ec2")
        _, ssm = _runtime_client(region=config.region, service="ssm")
        _, events = _runtime_client(region=config.region, service="events")
        return coordinate_must_start(
            config=config,
            runtime_account_id=runtime_account_id,
            runtime_region=runtime_region,
            now=datetime.now(timezone.utc),
            trigger=str(trigger),
            s3=s3,
            ec2=ec2,
            ssm=ssm,
            sns=sns,
            events=events,
        )
    except Exception as error:
        if not getattr(error, "_glm52_must_start_alerted", False):
            try:
                _alert(
                    sns,
                    topic_arn=config.alert_topic_arn,
                    subject="KEEP GLM52 must-start cancellation failed closed",
                    message={
                        "outcome": "fail-closed",
                        "run_id": config.run_id,
                        "managed_mode": config.managed_mode,
                        "sky_job_name": config.sky_job_name,
                        "error": str(error),
                    },
                )
            except Exception:
                pass
        raise


__all__ = [
    "MustStartCoordinatorConfig",
    "coordinate_must_start",
    "lambda_handler",
]
