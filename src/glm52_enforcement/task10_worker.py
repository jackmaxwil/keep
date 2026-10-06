"""Mount-free Sky task and reboot-safe worker contracts for H.1g Task 10.

This module is deliberately pure and import-light.  It renders immutable
artifacts and evaluates injected observations; it never invokes SkyPilot,
systemd, Systems Manager, AWS, or a subprocess.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import re
from typing import Mapping, Optional, Sequence, Tuple

from .canonical import canonical_json_bytes, canonical_sha256


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
SKYPILOT_VERSION = "0.13.0"
SKYPILOT_API_VERSION = 56
SKY_ENDPOINT = "/jobs/launch"
SKY_USER_ROLE = "user"
CAMPAIGN_UNIT = "keep-glm52-campaign.service"
DEADLINE_UNIT = "keep-glm52-deadline.service"
DEADLINE_TIMER = "keep-glm52-deadline.timer"
UNIT_NAMES = (CAMPAIGN_UNIT, DEADLINE_UNIT, DEADLINE_TIMER)
GRACEFUL_SCRIPT_NAMES = (
    "collect_task10_ssm_graceful_stop.py",
    "glm52_deadline_guard.py",
    "materialize_task10_graceful_stop.py",
    "run_campaign.sh",
    "run_task10_production_campaign.py",
)
CAMPAIGN_ROOT = "/mnt/nvme/glm52-campaign"
REPOSITORY_ROOT = "/opt/keep-campaign/repo"
DESCRIPTOR_PATH = "/etc/keep-glm52/campaign.json"
DEADLINE_DROPIN_PATH = (
    "/etc/systemd/system/keep-glm52-deadline.timer.d/"
    "10-immutable-deadline.conf"
)
CAMPAIGN_MAIN_ARGV = (
    "/usr/bin/python3",
    "/opt/keep-campaign/repo/benchmarks/run_glm52_campaign.py",
    "--descriptor",
    DESCRIPTOR_PATH,
    "--root",
    CAMPAIGN_ROOT,
    "--repo-root",
    REPOSITORY_ROOT,
)
WORKER_BOOTSTRAP_DESCRIPTOR_PATH = (
    "/etc/keep-glm52/worker-bootstrap.json"
)
WORKER_INSTANCE_OBSERVATION_PATH = (
    "/var/lib/keep-glm52/worker-instance-observation.json"
)
DEADLINE_STATE_PATH = "/var/lib/keep-glm52/deadline-state.json"
STOP_PATH = "/run/keep-glm52/STOP"
GRACEFUL_STOP_DOCUMENT = "KeepGlm52GracefulStopV1"
WORKER_DRAIN_ROLE = "keep-glm52-h1g-worker-drain-signal"
WORKER_DRAIN_ROLE_ARN = (
    "arn:aws:iam::246813579024:role/"
    "keep-glm52-h1g-worker-drain-signal"
)
H100_READY_KEY = (
    "campaigns/glm52-sky-20260724/qualification/H100_RESUME_READY.json"
)
JOBS_LAUNCH_BODY_FIELDS = frozenset(
    {
        "env_vars",
        "entrypoint",
        "entrypoint_command",
        "using_remote_api_server",
        "override_skypilot_config",
        "override_skypilot_config_path",
        "file_mounts_blob_id",
        "client_api_version",
        "task",
        "name",
        "pool",
        "num_jobs",
    }
)
_SHA = re.compile(r"^[0-9a-f]{64}$")
_S3_URI = re.compile(r"^s3://[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]/[^?]+$")
_JOB_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_VERSION = re.compile(r"^[A-Za-z0-9._~+/=-]{1,1024}$")
_INSTANCE = re.compile(r"^i-[0-9a-f]{17}$")
_SSM_COMMAND = re.compile(r"^[0-9a-f]{8}-[0-9a-f-]{27}$")
_SSM_DOCUMENT_VERSION = re.compile(r"^[1-9][0-9]*$")


class Task10WorkerError(ValueError):
    """An immutable worker, deadline, or drain contract drifted."""


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise Task10WorkerError(label + " must be a lowercase SHA-256")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise Task10WorkerError(label + " must be an exact nonempty string")
    return value


def _version(value: object, label: str) -> str:
    value = _text(value, label)
    if value == "null" or _VERSION.fullmatch(value) is None:
        raise Task10WorkerError(label + " must be an opaque VersionId")
    return value


def _utc_text(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise Task10WorkerError("deadline clock must be timezone-aware")
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc(value: object, label: str) -> datetime:
    if type(value) is not str:
        raise Task10WorkerError(label + " must be canonical UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise Task10WorkerError(label + " must be canonical UTC") from exc
    return parsed.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class MountFreeTaskInputs:
    """Only values allowed in the immutable production task environment."""

    job_name: str
    descriptor_s3_uri: str
    descriptor_version_id: str
    descriptor_file_sha256: str
    approval_s3_uri: str
    approval_version_id: str
    approval_file_sha256: str
    intent_s3_uri: str
    intent_version_id: str
    intent_file_sha256: str
    intent_body_sha256: str
    repository_archive_s3_uri: str
    repository_archive_version_id: str
    repository_archive_file_sha256: str


def validate_task_inputs(value: object) -> MountFreeTaskInputs:
    if not isinstance(value, MountFreeTaskInputs):
        raise Task10WorkerError("mount-free task inputs must be typed")
    if value.job_name != RUN_ID:
        raise Task10WorkerError("production Sky job name is not exact")
    for field in (
        "descriptor_s3_uri",
        "approval_s3_uri",
        "intent_s3_uri",
        "repository_archive_s3_uri",
    ):
        item = getattr(value, field)
        if type(item) is not str or _S3_URI.fullmatch(item) is None:
            raise Task10WorkerError(field + " must be an exact S3 URI")
    for field in (
        "descriptor_file_sha256",
        "approval_file_sha256",
        "intent_file_sha256",
        "intent_body_sha256",
        "repository_archive_file_sha256",
    ):
        _sha(getattr(value, field), field)
    _version(
        value.descriptor_version_id,
        "descriptor_version_id",
    )
    _version(
        value.approval_version_id,
        "approval_version_id",
    )
    _version(
        value.intent_version_id,
        "intent_version_id",
    )
    _version(
        value.repository_archive_version_id,
        "repository_archive_version_id",
    )
    return value


def task_environment(value: MountFreeTaskInputs) -> Mapping[str, str]:
    value = validate_task_inputs(value)
    return {
        "GLM52_APPROVAL_FILE_SHA256": value.approval_file_sha256,
        "GLM52_APPROVAL_S3_URI": value.approval_s3_uri,
        "GLM52_APPROVAL_VERSION_ID": value.approval_version_id,
        "GLM52_DESCRIPTOR_FILE_SHA256": value.descriptor_file_sha256,
        "GLM52_DESCRIPTOR_S3_URI": value.descriptor_s3_uri,
        "GLM52_DESCRIPTOR_VERSION_ID": value.descriptor_version_id,
        "GLM52_EXPECTED_SKY_JOB_NAME": value.job_name,
        "GLM52_MANAGED_MODE": "production",
        "GLM52_REPOSITORY_ARCHIVE_FILE_SHA256": (
            value.repository_archive_file_sha256
        ),
        "GLM52_REPOSITORY_ARCHIVE_S3_URI": (
            value.repository_archive_s3_uri
        ),
        "GLM52_REPOSITORY_ARCHIVE_VERSION_ID": (
            value.repository_archive_version_id
        ),
        "GLM52_SUBMISSION_INTENT_BODY_SHA256": value.intent_body_sha256,
        "GLM52_SUBMISSION_INTENT_FILE_SHA256": value.intent_file_sha256,
        "GLM52_SUBMISSION_INTENT_S3_URI": value.intent_s3_uri,
        "GLM52_SUBMISSION_INTENT_VERSION_ID": value.intent_version_id,
    }


_SETUP = """set -euo pipefail
sudo install -d -m 0755 /opt/keep-campaign /opt/keep-campaign/repo
sudo install -d -m 0700 /opt/keep-campaign/download
verify_exact_get() {
  /usr/bin/python3 - "$1" "$2" "$3" <<'PY'
import base64
import hashlib
import json
from pathlib import Path
import sys

metadata = json.loads(Path(sys.argv[1]).read_bytes())
raw = Path(sys.argv[2]).read_bytes()
checksum = base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
if (
    type(metadata) is not dict
    or metadata.get("VersionId") != sys.argv[3]
    or metadata.get("ChecksumSHA256") != checksum
):
    raise SystemExit("exact S3 get metadata/checksum drifted")
PY
}
descriptor_ref=${GLM52_DESCRIPTOR_S3_URI#s3://}
descriptor_bucket=${descriptor_ref%%/*}
descriptor_key=${descriptor_ref#*/}
test ! -e /opt/keep-campaign/download/campaign.json
AWS_MAX_ATTEMPTS=1 AWS_RETRY_MODE=standard aws s3api get-object --bucket "$descriptor_bucket" --key "$descriptor_key" --version-id "$GLM52_DESCRIPTOR_VERSION_ID" --expected-bucket-owner 246813579024 --checksum-mode ENABLED /opt/keep-campaign/download/campaign.json >/opt/keep-campaign/download/campaign.get.json
verify_exact_get /opt/keep-campaign/download/campaign.get.json /opt/keep-campaign/download/campaign.json "$GLM52_DESCRIPTOR_VERSION_ID"
printf '%s  %s\\n' "$GLM52_DESCRIPTOR_FILE_SHA256" /opt/keep-campaign/download/campaign.json | sha256sum -c -
archive_ref=${GLM52_REPOSITORY_ARCHIVE_S3_URI#s3://}
archive_bucket=${archive_ref%%/*}
archive_key=${archive_ref#*/}
test ! -e /opt/keep-campaign/download/repo.tar.gz
AWS_MAX_ATTEMPTS=1 AWS_RETRY_MODE=standard aws s3api get-object --bucket "$archive_bucket" --key "$archive_key" --version-id "$GLM52_REPOSITORY_ARCHIVE_VERSION_ID" --expected-bucket-owner 246813579024 --checksum-mode ENABLED /opt/keep-campaign/download/repo.tar.gz >/opt/keep-campaign/download/archive.get.json
verify_exact_get /opt/keep-campaign/download/archive.get.json /opt/keep-campaign/download/repo.tar.gz "$GLM52_REPOSITORY_ARCHIVE_VERSION_ID"
printf '%s  %s\\n' "$GLM52_REPOSITORY_ARCHIVE_FILE_SHA256" /opt/keep-campaign/download/repo.tar.gz | sha256sum -c -
sudo tar -xzf /opt/keep-campaign/download/repo.tar.gz -C /opt/keep-campaign/repo
sudo env \\
  GLM52_APPROVAL_FILE_SHA256="$GLM52_APPROVAL_FILE_SHA256" \\
  GLM52_APPROVAL_S3_URI="$GLM52_APPROVAL_S3_URI" \\
  GLM52_APPROVAL_VERSION_ID="$GLM52_APPROVAL_VERSION_ID" \\
  GLM52_DESCRIPTOR_FILE_SHA256="$GLM52_DESCRIPTOR_FILE_SHA256" \\
  GLM52_DESCRIPTOR_S3_URI="$GLM52_DESCRIPTOR_S3_URI" \\
  GLM52_DESCRIPTOR_VERSION_ID="$GLM52_DESCRIPTOR_VERSION_ID" \\
  GLM52_EXPECTED_SKY_JOB_NAME="$GLM52_EXPECTED_SKY_JOB_NAME" \\
  GLM52_MANAGED_MODE="$GLM52_MANAGED_MODE" \\
  GLM52_REPOSITORY_ARCHIVE_FILE_SHA256="$GLM52_REPOSITORY_ARCHIVE_FILE_SHA256" \\
  GLM52_REPOSITORY_ARCHIVE_S3_URI="$GLM52_REPOSITORY_ARCHIVE_S3_URI" \\
  GLM52_REPOSITORY_ARCHIVE_VERSION_ID="$GLM52_REPOSITORY_ARCHIVE_VERSION_ID" \\
  GLM52_SUBMISSION_INTENT_BODY_SHA256="$GLM52_SUBMISSION_INTENT_BODY_SHA256" \\
  GLM52_SUBMISSION_INTENT_FILE_SHA256="$GLM52_SUBMISSION_INTENT_FILE_SHA256" \\
  GLM52_SUBMISSION_INTENT_S3_URI="$GLM52_SUBMISSION_INTENT_S3_URI" \\
  GLM52_SUBMISSION_INTENT_VERSION_ID="$GLM52_SUBMISSION_INTENT_VERSION_ID" \\
  /opt/keep-campaign/repo/aws/glm52-gpu/skypilot/bootstrap_production_campaign.sh
"""

_RUN = (
    "sudo systemctl is-active --quiet keep-glm52-campaign.service && "
    "sudo systemctl is-active --quiet keep-glm52-deadline.timer"
)


def _yaml_scalar(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def render_mount_free_task(value: MountFreeTaskInputs) -> str:
    """Render the one canonical SkyPilot 0.13.0 task YAML."""

    value = validate_task_inputs(value)
    lines = [
        "name: " + _yaml_scalar(value.job_name),
        "num_nodes: 1",
        "api_server_access: false",
        "resources:",
        "  infra: aws/us-west-2",
        "  instance_type: p5.48xlarge",
        "  use_spot: false",
        "  max_hourly_cost: 55.04",
        "  job_recovery:",
        "    strategy: FAILOVER",
        "    max_restarts_on_errors: 0",
        "envs:",
    ]
    for key, item in sorted(task_environment(value).items()):
        lines.append("  " + key + ": " + _yaml_scalar(item))
    lines.extend(["setup: |"])
    lines.extend("  " + item for item in _SETUP.rstrip("\n").splitlines())
    lines.extend(["run: |", "  " + _RUN, ""])
    rendered = "\n".join(lines)
    forbidden = ("file_mounts:", "workdir:", "sky jobs launch")
    if any(token in rendered for token in forbidden):
        raise Task10WorkerError("mount-free task renderer exposed a forbidden path")
    return rendered


def build_jobs_launch_body(
    *,
    task_yaml: str,
    job_name: str,
) -> Mapping[str, object]:
    if type(task_yaml) is not str or not task_yaml:
        raise Task10WorkerError("task YAML must be a nonempty exact string")
    if _JOB_NAME.fullmatch(job_name) is None:
        raise Task10WorkerError("Sky job name is invalid")
    body = {
        "env_vars": {},
        "entrypoint": "",
        "entrypoint_command": "",
        "using_remote_api_server": True,
        "override_skypilot_config": {},
        "override_skypilot_config_path": None,
        "file_mounts_blob_id": None,
        "client_api_version": None,
        "task": task_yaml,
        "name": job_name,
        "pool": None,
        "num_jobs": None,
    }
    return validate_jobs_launch_body(
        body,
        expected_task_yaml=task_yaml,
        expected_job_name=job_name,
    )


def validate_jobs_launch_body(
    value: object,
    *,
    expected_task_yaml: str,
    expected_job_name: str,
) -> Mapping[str, object]:
    if type(value) is not dict or set(value) != JOBS_LAUNCH_BODY_FIELDS:
        raise Task10WorkerError("JobsLaunchBody field set drifted")
    expected = {
        "env_vars": {},
        "entrypoint": "",
        "entrypoint_command": "",
        "using_remote_api_server": True,
        "override_skypilot_config": {},
        "override_skypilot_config_path": None,
        "file_mounts_blob_id": None,
        "client_api_version": None,
        "task": expected_task_yaml,
        "name": expected_job_name,
        "pool": None,
        "num_jobs": None,
    }
    if value != expected:
        raise Task10WorkerError("JobsLaunchBody value or nullability drifted")
    return value


@dataclass(frozen=True)
class ClosedWireRequest:
    endpoint: str
    headers: Mapping[str, str]
    service_account_user_id: str
    service_account_role: str
    body: Mapping[str, object]
    task_yaml_sha256: str
    request_body_sha256: str
    relay_envelope_sha256: str
    client_certificate_sha256: str


def build_closed_wire_request(
    *,
    body: Mapping[str, object],
    task_yaml: str,
    job_name: str,
    bearer_token: str,
    service_account_user_id: str,
    relay_envelope_sha256: str,
    client_certificate_sha256: str,
) -> ClosedWireRequest:
    validate_jobs_launch_body(
        body,
        expected_task_yaml=task_yaml,
        expected_job_name=job_name,
    )
    if (
        type(bearer_token) is not str
        or not bearer_token.startswith("sky_")
        or len(bearer_token) <= 4
    ):
        raise Task10WorkerError("Sky bearer token format is invalid")
    _text(service_account_user_id, "service-account user")
    relay = _sha(relay_envelope_sha256, "relay envelope")
    certificate = _sha(client_certificate_sha256, "client certificate")
    return ClosedWireRequest(
        endpoint=SKY_ENDPOINT,
        headers={
            "Authorization": "Bearer " + bearer_token,
            "Content-Type": "application/json",
            "X-SkyPilot-API-Version": str(SKYPILOT_API_VERSION),
            "X-SkyPilot-Version": SKYPILOT_VERSION,
        },
        service_account_user_id=service_account_user_id,
        service_account_role=SKY_USER_ROLE,
        body=body,
        task_yaml_sha256=hashlib.sha256(task_yaml.encode("utf-8")).hexdigest(),
        request_body_sha256=canonical_sha256(body),
        relay_envelope_sha256=relay,
        client_certificate_sha256=certificate,
    )


def validate_closed_wire_request(
    value: object,
    *,
    expected_body: Mapping[str, object],
    expected_task_yaml: str,
    expected_job_name: str,
    expected_user_id: str,
    expected_relay_envelope_sha256: str,
    expected_client_certificate_sha256: str,
) -> ClosedWireRequest:
    if not isinstance(value, ClosedWireRequest):
        raise Task10WorkerError("closed wire request must be typed")
    validate_jobs_launch_body(
        value.body,
        expected_task_yaml=expected_task_yaml,
        expected_job_name=expected_job_name,
    )
    expected_headers = {
        "Authorization": value.headers.get("Authorization"),
        "Content-Type": "application/json",
        "X-SkyPilot-API-Version": "56",
        "X-SkyPilot-Version": "0.13.0",
    }
    authorization = value.headers.get("Authorization")
    if (
        type(authorization) is not str
        or not authorization.startswith("Bearer sky_")
        or value.endpoint != SKY_ENDPOINT
        or value.headers != expected_headers
        or value.service_account_user_id != expected_user_id
        or value.service_account_role != SKY_USER_ROLE
        or value.body != expected_body
        or value.task_yaml_sha256
        != hashlib.sha256(expected_task_yaml.encode("utf-8")).hexdigest()
        or value.request_body_sha256 != canonical_sha256(expected_body)
        or value.relay_envelope_sha256
        != expected_relay_envelope_sha256
        or value.client_certificate_sha256
        != expected_client_certificate_sha256
    ):
        raise Task10WorkerError("closed Sky wire identity drifted")
    return value


_CAMPAIGN_SERVICE = """[Unit]
Description=Immutable GLM-5.2 production campaign
After=network-online.target keep-glm52-deadline.timer
Wants=network-online.target keep-glm52-deadline.timer

[Service]
Type=notify
NotifyAccess=main
TimeoutStartSec=300
WorkingDirectory=/opt/keep-campaign/repo
Environment=CAMPAIGN_DESCRIPTOR=/etc/keep-glm52/campaign.json
Environment=ROOT=/mnt/nvme/glm52-campaign
Environment=KEEP_REPO_DIR=/opt/keep-campaign/repo
Environment=GLM52_DEADLINE_STATE=/var/lib/keep-glm52/deadline-state.json
Environment=GLM52_HEAVY_JOB_LOCK=/mnt/nvme/glm52-campaign/runtime/heavy-job.lock
Environment=GLM52_ENV_PREPARED=1
Environment=GLM52_MANAGED_MODE=production
ExecStartPre=/opt/keep-campaign/repo/aws/glm52-gpu/scripts/glm52_deadline_guard.py pre-start
ExecStart=/opt/keep-campaign/repo/aws/glm52-gpu/scripts/run_campaign.sh --descriptor /etc/keep-glm52/campaign.json --deadline-state /var/lib/keep-glm52/deadline-state.json
KillSignal=SIGTERM
KillMode=control-group
TimeoutStopSec=1200
SendSIGKILL=no
Restart=no

[Install]
WantedBy=multi-user.target
"""

_DEADLINE_SERVICE = """[Unit]
Description=Immutable GLM-5.2 T-50 graceful-stop edge
Requires=keep-glm52-deadline.timer

[Service]
Type=oneshot
WorkingDirectory=/opt/keep-campaign/repo
ExecStart=/opt/keep-campaign/repo/aws/glm52-gpu/scripts/glm52_deadline_guard.py timer
"""

_DEADLINE_TIMER = """[Unit]
Description=Persistent immutable GLM-5.2 deadline timer

[Timer]
Persistent=true
Unit=keep-glm52-deadline.service

[Install]
WantedBy=timers.target
"""


def render_worker_units() -> Mapping[str, bytes]:
    units = {
        CAMPAIGN_UNIT: _CAMPAIGN_SERVICE.encode("utf-8"),
        DEADLINE_UNIT: _DEADLINE_SERVICE.encode("utf-8"),
        DEADLINE_TIMER: _DEADLINE_TIMER.encode("utf-8"),
    }
    validate_worker_units(units)
    return units


def worker_unit_hashes(units: Mapping[str, bytes]) -> Mapping[str, str]:
    validate_worker_units(units)
    return {
        name: hashlib.sha256(units[name]).hexdigest()
        for name in UNIT_NAMES
    }


def validate_worker_units(value: object) -> Mapping[str, bytes]:
    if type(value) is not dict or set(value) != set(UNIT_NAMES):
        raise Task10WorkerError("the worker must install exactly three units")
    if any(type(item) is not bytes for item in value.values()):
        raise Task10WorkerError("unit content must be immutable bytes")
    campaign = value[CAMPAIGN_UNIT].decode("utf-8")
    required = (
        "Type=notify",
        "NotifyAccess=main",
        "TimeoutStartSec=300",
        "WorkingDirectory=/opt/keep-campaign/repo",
        "Restart=no",
        "KillMode=control-group",
        "TimeoutStopSec=1200",
        "SendSIGKILL=no",
        "GLM52_HEAVY_JOB_LOCK=/mnt/nvme/glm52-campaign/runtime/heavy-job.lock",
        "ExecStartPre=/opt/keep-campaign/repo/aws/glm52-gpu/scripts/"
        "glm52_deadline_guard.py pre-start",
        "ExecStart=/opt/keep-campaign/repo/aws/glm52-gpu/scripts/"
        "run_campaign.sh --descriptor /etc/keep-glm52/campaign.json "
        "--deadline-state /var/lib/keep-glm52/deadline-state.json",
    )
    if any(item not in campaign for item in required):
        raise Task10WorkerError("campaign unit contract drifted")
    forbidden = ("bash -c", "sh -c", "EnvironmentFile=", "SIGKILL")
    if any(item in campaign for item in forbidden if item != "SIGKILL"):
        raise Task10WorkerError("campaign unit exposes caller authority")
    if "SendSIGKILL=no" not in campaign:
        raise Task10WorkerError("campaign unit may send SIGKILL")
    deadline = value[DEADLINE_UNIT].decode("utf-8")
    timer = value[DEADLINE_TIMER].decode("utf-8")
    if (
        "glm52_deadline_guard.py timer" not in deadline
        or "Persistent=true" not in timer
        or "OnCalendar=" in timer
    ):
        raise Task10WorkerError("deadline unit or base timer drifted")
    return value


@dataclass(frozen=True)
class DeadlineState:
    schema_version: int
    record_type: str
    descriptor_file_sha256: str
    instance_id: str
    allocation_ordinal: int
    execution_deadline: str
    stop_assignment_at: str
    graceful_stop_at: str
    retained_force_not_before: str
    last_completed_edge: str
    state_body_sha256: str
    state_signature_sha256: str


@dataclass(frozen=True)
class WorkerBootstrapDescriptor:
    """Task 8/9-derived runtime authority, distinct from H.1c descriptor v2."""

    schema_version: int
    record_type: str
    account_id: str
    region: str
    run_id: str
    campaign_identity_sha256: str
    activation_id: str
    activation_ordinal: int
    generation: int
    generation_text: str
    action_key: str
    sky_job_name: str
    execution_deadline: str
    gpu_allocation_sha256: str
    base_descriptor_s3_uri: str
    base_descriptor_version_id: str
    base_descriptor_file_sha256: str
    base_descriptor_body_sha256: str
    archive_identity_sha256: str
    repository_archive_version_id: str
    approval_identity_sha256: str
    approval_version_id: str
    intent_identity_sha256: str
    intent_version_id: str
    task8_live_h1d_identity_sha256: str
    task8_spend_authority_identity_sha256: str
    task9_launch_identity_sha256: str
    task9_admission_identity_sha256: str
    task9_custody_identity_sha256: str
    descriptor_body_sha256: str


def _bootstrap_descriptor_body(
    value: WorkerBootstrapDescriptor,
) -> Mapping[str, object]:
    body = asdict(value)
    body.pop("descriptor_body_sha256")
    return body


def build_worker_bootstrap_descriptor(
    **values: object,
) -> WorkerBootstrapDescriptor:
    provisional = WorkerBootstrapDescriptor(
        **{**values, "descriptor_body_sha256": ""}
    )
    return validate_worker_bootstrap_descriptor(
        WorkerBootstrapDescriptor(
            **{
                **asdict(provisional),
                "descriptor_body_sha256": canonical_sha256(
                    _bootstrap_descriptor_body(provisional)
                ),
            }
        )
    )


def validate_worker_bootstrap_descriptor(
    value: object,
) -> WorkerBootstrapDescriptor:
    if not isinstance(value, WorkerBootstrapDescriptor):
        raise Task10WorkerError("worker bootstrap descriptor must be typed")
    if (
        value.schema_version != 1
        or value.record_type != "glm52_task10_worker_bootstrap_descriptor_v1"
        or value.account_id != ACCOUNT_ID
        or value.region != REGION
        or value.run_id != RUN_ID
        or type(value.activation_id) is not str
        or not value.activation_id
        or type(value.activation_ordinal) is not int
        or value.activation_ordinal <= 0
        or type(value.generation) is not int
        or value.generation <= 0
        or value.generation_text != "%08d" % value.generation
        or type(value.action_key) is not str
        or "SKY_POST" not in value.action_key
        or type(value.sky_job_name) is not str
        or value.sky_job_name != RUN_ID
        or type(value.base_descriptor_s3_uri) is not str
        or _S3_URI.fullmatch(value.base_descriptor_s3_uri) is None
    ):
        raise Task10WorkerError("worker bootstrap descriptor identity drifted")
    _utc(value.execution_deadline, "execution_deadline")
    _version(value.base_descriptor_version_id, "base descriptor VersionId")
    _version(
        value.repository_archive_version_id,
        "repository archive VersionId",
    )
    _version(value.approval_version_id, "approval VersionId")
    _version(value.intent_version_id, "intent VersionId")
    for field in (
        "campaign_identity_sha256",
        "gpu_allocation_sha256",
        "base_descriptor_file_sha256",
        "base_descriptor_body_sha256",
        "archive_identity_sha256",
        "approval_identity_sha256",
        "intent_identity_sha256",
        "task8_live_h1d_identity_sha256",
        "task8_spend_authority_identity_sha256",
        "task9_launch_identity_sha256",
        "task9_admission_identity_sha256",
        "task9_custody_identity_sha256",
    ):
        _sha(getattr(value, field), field)
    if value.descriptor_body_sha256 != canonical_sha256(
        _bootstrap_descriptor_body(value)
    ):
        raise Task10WorkerError("worker bootstrap descriptor self-hash drifted")
    return value


def worker_bootstrap_descriptor_from_mapping(
    value: object,
) -> WorkerBootstrapDescriptor:
    if (
        type(value) is not dict
        or set(value) != set(WorkerBootstrapDescriptor.__dataclass_fields__)
    ):
        raise Task10WorkerError("worker bootstrap descriptor schema drifted")
    try:
        result = WorkerBootstrapDescriptor(**value)
    except TypeError as exc:
        raise Task10WorkerError(
            "worker bootstrap descriptor is malformed"
        ) from exc
    return validate_worker_bootstrap_descriptor(result)


@dataclass(frozen=True)
class WorkerInstanceObservation:
    schema_version: int
    record_type: str
    account_id: str
    region: str
    run_id: str
    campaign_identity_sha256: str
    activation_id: str
    activation_ordinal: int
    activation_ordinal_text: str
    generation: int
    generation_text: str
    allocation_ordinal: int
    allocation_ordinal_text: str
    instance_id: str
    action_key: str
    sky_job_name: str
    task_yaml_sha256: str
    request_body_sha256: str
    task9_launch_identity_sha256: str
    task9_custody_identity_sha256: str
    observation_body_sha256: str


def _instance_observation_body(
    value: WorkerInstanceObservation,
) -> Mapping[str, object]:
    body = asdict(value)
    body.pop("observation_body_sha256")
    return body


def build_worker_instance_observation(
    **values: object,
) -> WorkerInstanceObservation:
    provisional = WorkerInstanceObservation(
        **{**values, "observation_body_sha256": ""}
    )
    return validate_worker_instance_observation(
        WorkerInstanceObservation(
            **{
                **asdict(provisional),
                "observation_body_sha256": canonical_sha256(
                    _instance_observation_body(provisional)
                ),
            }
        )
    )


def validate_worker_instance_observation(
    value: object,
) -> WorkerInstanceObservation:
    if not isinstance(value, WorkerInstanceObservation):
        raise Task10WorkerError("worker instance observation must be typed")
    if (
        value.schema_version != 1
        or value.record_type != "glm52_task10_worker_instance_observation_v1"
        or value.account_id != ACCOUNT_ID
        or value.region != REGION
        or value.run_id != RUN_ID
        or type(value.activation_id) is not str
        or not value.activation_id
        or type(value.activation_ordinal) is not int
        or value.activation_ordinal <= 0
        or value.activation_ordinal_text
        != "%08d" % value.activation_ordinal
        or type(value.generation) is not int
        or value.generation <= 0
        or value.generation_text != "%08d" % value.generation
        or type(value.allocation_ordinal) is not int
        or value.allocation_ordinal <= 0
        or value.allocation_ordinal_text
        != "%08d" % value.allocation_ordinal
        or _INSTANCE.fullmatch(value.instance_id) is None
        or type(value.action_key) is not str
        or "SKY_POST" not in value.action_key
        or type(value.sky_job_name) is not str
        or value.sky_job_name != RUN_ID
    ):
        raise Task10WorkerError("worker instance observation drifted")
    for field in (
        "campaign_identity_sha256",
        "task_yaml_sha256",
        "request_body_sha256",
        "task9_launch_identity_sha256",
        "task9_custody_identity_sha256",
    ):
        _sha(getattr(value, field), field)
    if value.observation_body_sha256 != canonical_sha256(
        _instance_observation_body(value)
    ):
        raise Task10WorkerError("worker instance observation self-hash drifted")
    return value


def worker_instance_observation_from_mapping(
    value: object,
) -> WorkerInstanceObservation:
    if (
        type(value) is not dict
        or set(value) != set(WorkerInstanceObservation.__dataclass_fields__)
    ):
        raise Task10WorkerError("worker instance observation schema drifted")
    try:
        result = WorkerInstanceObservation(**value)
    except TypeError as exc:
        raise Task10WorkerError(
            "worker instance observation is malformed"
        ) from exc
    return validate_worker_instance_observation(result)


@dataclass(frozen=True)
class WorkerRuntimeAuthority:
    descriptor: WorkerBootstrapDescriptor
    observation: WorkerInstanceObservation
    runtime_authority_sha256: str


@dataclass(frozen=True)
class Task12WorkerDrainAuthority:
    """Durable Task 12 authority for one exact retained SSM drain."""

    schema_version: int
    record_type: str
    account_id: str
    region: str
    run_id: str
    campaign_identity_sha256: str
    activation_id: str
    activation_ordinal: int
    generation: int
    generation_text: str
    allocation_ordinal: int
    allocation_ordinal_text: str
    instance_id: str
    command_id: str
    document_name: str
    document_version: str
    worker_drain_role_arn: str
    worker_descriptor_body_sha256: str
    worker_observation: Mapping[str, object]
    expected_unit_hashes: Mapping[str, str]
    expected_script_hashes: Mapping[str, str]
    task9_launch_identity_sha256: str
    task9_custody_identity_sha256: str
    task12_action_identity_sha256: str
    task12_audit_identity_sha256: str
    authority_body_sha256: str


def _task12_worker_drain_authority_body(
    value: Task12WorkerDrainAuthority,
) -> Mapping[str, object]:
    body = asdict(value)
    body.pop("authority_body_sha256")
    return body


def build_task12_worker_drain_authority(
    *,
    worker_descriptor: WorkerBootstrapDescriptor,
    **values: object,
) -> Task12WorkerDrainAuthority:
    provisional = Task12WorkerDrainAuthority(
        **{**values, "authority_body_sha256": ""}
    )
    return validate_task12_worker_drain_authority(
        Task12WorkerDrainAuthority(
            **{
                **asdict(provisional),
                "authority_body_sha256": canonical_sha256(
                    _task12_worker_drain_authority_body(provisional)
                ),
            }
        ),
        worker_descriptor=worker_descriptor,
    )


def validate_task12_worker_drain_authority(
    value: object,
    *,
    worker_descriptor: WorkerBootstrapDescriptor,
) -> Task12WorkerDrainAuthority:
    if not isinstance(value, Task12WorkerDrainAuthority):
        raise Task10WorkerError("Task 12 worker-drain authority must be typed")
    descriptor = validate_worker_bootstrap_descriptor(worker_descriptor)
    if type(value.worker_observation) is not dict:
        raise Task10WorkerError("Task 12 worker observation is not closed")
    observation = worker_instance_observation_from_mapping(
        value.worker_observation
    )
    build_worker_runtime_authority(descriptor, observation)
    if (
        type(value.expected_unit_hashes) is not dict
        or set(value.expected_unit_hashes) != set(UNIT_NAMES)
        or type(value.expected_script_hashes) is not dict
        or set(value.expected_script_hashes) != set(GRACEFUL_SCRIPT_NAMES)
    ):
        raise Task10WorkerError("Task 12 worker artifact hashes are not closed")
    for name, artifact_sha256 in (
        tuple(value.expected_unit_hashes.items())
        + tuple(value.expected_script_hashes.items())
    ):
        if type(name) is not str:
            raise Task10WorkerError("Task 12 worker artifact name is invalid")
        _sha(artifact_sha256, "Task 12 worker artifact")
    if (
        value.schema_version != 1
        or value.record_type
        != "glm52_task12_retained_worker_drain_authority_v1"
        or value.account_id != ACCOUNT_ID
        or value.region != REGION
        or value.run_id != RUN_ID
        or value.campaign_identity_sha256
        != descriptor.campaign_identity_sha256
        or value.activation_id != descriptor.activation_id
        or value.activation_ordinal != descriptor.activation_ordinal
        or value.generation != descriptor.generation
        or value.generation_text != descriptor.generation_text
        or value.allocation_ordinal != observation.allocation_ordinal
        or value.allocation_ordinal_text
        != observation.allocation_ordinal_text
        or value.instance_id != observation.instance_id
        or _SSM_COMMAND.fullmatch(value.command_id) is None
        or value.document_name != GRACEFUL_STOP_DOCUMENT
        or _SSM_DOCUMENT_VERSION.fullmatch(value.document_version) is None
        or value.worker_drain_role_arn != WORKER_DRAIN_ROLE_ARN
        or value.worker_descriptor_body_sha256
        != descriptor.descriptor_body_sha256
        or value.task9_launch_identity_sha256
        != descriptor.task9_launch_identity_sha256
        or value.task9_launch_identity_sha256
        != observation.task9_launch_identity_sha256
        or value.task9_custody_identity_sha256
        != descriptor.task9_custody_identity_sha256
        or value.task9_custody_identity_sha256
        != observation.task9_custody_identity_sha256
    ):
        raise Task10WorkerError("Task 12 worker-drain authority drifted")
    _sha(value.task12_action_identity_sha256, "Task 12 action identity")
    _sha(value.task12_audit_identity_sha256, "Task 12 audit identity")
    if value.authority_body_sha256 != canonical_sha256(
        _task12_worker_drain_authority_body(value)
    ):
        raise Task10WorkerError("Task 12 worker-drain authority self-hash drifted")
    return value


def task12_worker_drain_authority_from_mapping(
    value: object,
    *,
    worker_descriptor: WorkerBootstrapDescriptor,
) -> Task12WorkerDrainAuthority:
    if (
        type(value) is not dict
        or set(value) != set(Task12WorkerDrainAuthority.__dataclass_fields__)
    ):
        raise Task10WorkerError("Task 12 worker-drain authority schema drifted")
    try:
        result = Task12WorkerDrainAuthority(**value)
    except TypeError as exc:
        raise Task10WorkerError(
            "Task 12 worker-drain authority is malformed"
        ) from exc
    return validate_task12_worker_drain_authority(
        result,
        worker_descriptor=worker_descriptor,
    )


@dataclass(frozen=True)
class RetainedSsmGracefulStopHandoff:
    schema_version: int
    record_type: str
    account_id: str
    region: str
    run_id: str
    command_id: str
    document_name: str
    document_version: str
    instance_id: str
    allocation_ordinal: int
    worker_descriptor: Mapping[str, object]
    campaign_descriptor: Mapping[str, object]
    task12_authority_key: str
    task12_authority_version_id: str
    task12_authority_file_sha256: str
    task12_authority: Mapping[str, object]
    handoff_body_sha256: str


def _retained_ssm_handoff_body(
    value: RetainedSsmGracefulStopHandoff,
) -> Mapping[str, object]:
    body = asdict(value)
    body.pop("handoff_body_sha256")
    return body


def build_retained_ssm_graceful_stop_handoff(
    **values: object,
) -> RetainedSsmGracefulStopHandoff:
    provisional = RetainedSsmGracefulStopHandoff(
        **{**values, "handoff_body_sha256": ""}
    )
    return validate_retained_ssm_graceful_stop_handoff(
        RetainedSsmGracefulStopHandoff(
            **{
                **asdict(provisional),
                "handoff_body_sha256": canonical_sha256(
                    _retained_ssm_handoff_body(provisional)
                ),
            }
        )
    )


def validate_retained_ssm_graceful_stop_handoff(
    value: object,
) -> RetainedSsmGracefulStopHandoff:
    if not isinstance(value, RetainedSsmGracefulStopHandoff):
        raise Task10WorkerError("retained SSM handoff must be typed")
    if (
        value.schema_version != 1
        or value.record_type
        != "glm52_task10_retained_ssm_graceful_stop_handoff_v1"
        or value.account_id != ACCOUNT_ID
        or value.region != REGION
        or value.run_id != RUN_ID
        or _SSM_COMMAND.fullmatch(value.command_id) is None
        or value.document_name != GRACEFUL_STOP_DOCUMENT
        or _SSM_DOCUMENT_VERSION.fullmatch(value.document_version) is None
        or _INSTANCE.fullmatch(value.instance_id) is None
        or type(value.allocation_ordinal) is not int
        or value.allocation_ordinal <= 0
        or type(value.worker_descriptor) is not dict
        or type(value.campaign_descriptor) is not dict
        or type(value.task12_authority_key) is not str
        or type(value.task12_authority) is not dict
        or value.handoff_body_sha256
        != canonical_sha256(_retained_ssm_handoff_body(value))
    ):
        raise Task10WorkerError("retained SSM handoff identity drifted")
    descriptor = worker_bootstrap_descriptor_from_mapping(
        value.worker_descriptor
    )
    authority = task12_worker_drain_authority_from_mapping(
        value.task12_authority,
        worker_descriptor=descriptor,
    )
    expected_key = (
        f"campaigns/{RUN_ID}/submissions/production/generations/"
        f"{descriptor.generation:08d}/allocations/"
        f"{authority.allocation_ordinal:08d}/"
        "TASK12_WORKER_DRAIN_AUTHORITY.json"
    )
    authority_raw = canonical_json_bytes(asdict(authority)) + b"\n"
    _version(
        value.task12_authority_version_id,
        "Task 12 authority VersionId",
    )
    _sha(
        value.task12_authority_file_sha256,
        "Task 12 authority file",
    )
    if (
        value.task12_authority_key != expected_key
        or value.task12_authority_file_sha256
        != hashlib.sha256(authority_raw).hexdigest()
        or value.command_id != authority.command_id
        or value.document_name != authority.document_name
        or value.document_version != authority.document_version
        or value.instance_id != authority.instance_id
        or value.allocation_ordinal != authority.allocation_ordinal
    ):
        raise Task10WorkerError(
            "retained SSM handoff lacks closed Task 12 provenance"
        )
    return value


def retained_ssm_graceful_stop_handoff_from_mapping(
    value: object,
) -> RetainedSsmGracefulStopHandoff:
    if (
        type(value) is not dict
        or set(value)
        != set(RetainedSsmGracefulStopHandoff.__dataclass_fields__)
    ):
        raise Task10WorkerError("retained SSM handoff schema drifted")
    return validate_retained_ssm_graceful_stop_handoff(
        RetainedSsmGracefulStopHandoff(**value)
    )


def build_worker_runtime_authority(
    descriptor: WorkerBootstrapDescriptor,
    observation: WorkerInstanceObservation,
) -> WorkerRuntimeAuthority:
    descriptor = validate_worker_bootstrap_descriptor(descriptor)
    observation = validate_worker_instance_observation(observation)
    if (
        observation.campaign_identity_sha256
        != descriptor.campaign_identity_sha256
        or observation.activation_id != descriptor.activation_id
        or observation.activation_ordinal
        != descriptor.activation_ordinal
        or observation.generation != descriptor.generation
        or observation.generation_text != descriptor.generation_text
        or observation.action_key != descriptor.action_key
        or observation.sky_job_name != descriptor.sky_job_name
        or observation.task9_launch_identity_sha256
        != descriptor.task9_launch_identity_sha256
        or observation.task9_custody_identity_sha256
        != descriptor.task9_custody_identity_sha256
    ):
        raise Task10WorkerError(
            "Task 8/9 worker runtime authority did not cross-bind"
        )
    body = {
        "descriptor_body_sha256": descriptor.descriptor_body_sha256,
        "observation_body_sha256": observation.observation_body_sha256,
    }
    return WorkerRuntimeAuthority(
        descriptor=descriptor,
        observation=observation,
        runtime_authority_sha256=canonical_sha256(body),
    )


def _deadline_body(value: DeadlineState) -> Mapping[str, object]:
    body = asdict(value)
    body.pop("state_body_sha256")
    body.pop("state_signature_sha256")
    return body


def _sign(body_sha256: str, signing_key: bytes) -> str:
    if type(signing_key) is not bytes or len(signing_key) < 32:
        raise Task10WorkerError("deadline signing key is invalid")
    return hmac.new(
        signing_key,
        body_sha256.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()


def build_deadline_state(
    *,
    descriptor_file_sha256: str,
    instance_id: str,
    allocation_ordinal: int,
    execution_deadline: datetime,
    signing_key: bytes,
    last_completed_edge: str = "NONE",
) -> DeadlineState:
    _sha(descriptor_file_sha256, "descriptor_file_sha256")
    if type(instance_id) is not str or _INSTANCE.fullmatch(instance_id) is None:
        raise Task10WorkerError("deadline instance identity is invalid")
    if type(allocation_ordinal) is not int or allocation_ordinal <= 0:
        raise Task10WorkerError("deadline allocation ordinal is invalid")
    if last_completed_edge not in {"NONE", "T_MINUS_60", "T_MINUS_50"}:
        raise Task10WorkerError("deadline edge is invalid")
    deadline = _utc(_utc_text(execution_deadline), "execution_deadline")
    provisional = DeadlineState(
        schema_version=1,
        record_type="glm52_worker_deadline_state_v1",
        descriptor_file_sha256=descriptor_file_sha256,
        instance_id=instance_id,
        allocation_ordinal=allocation_ordinal,
        execution_deadline=_utc_text(deadline),
        stop_assignment_at=_utc_text(deadline - timedelta(minutes=60)),
        graceful_stop_at=_utc_text(deadline - timedelta(minutes=50)),
        retained_force_not_before=_utc_text(deadline - timedelta(minutes=30)),
        last_completed_edge=last_completed_edge,
        state_body_sha256="",
        state_signature_sha256="",
    )
    body_sha = canonical_sha256(_deadline_body(provisional))
    return validate_deadline_state(
        DeadlineState(
            **{
                **asdict(provisional),
                "state_body_sha256": body_sha,
                "state_signature_sha256": _sign(body_sha, signing_key),
            }
        ),
        descriptor_file_sha256=descriptor_file_sha256,
        signing_key=signing_key,
    )


def validate_deadline_state(
    value: object,
    *,
    descriptor_file_sha256: str,
    signing_key: bytes,
) -> DeadlineState:
    if not isinstance(value, DeadlineState):
        raise Task10WorkerError("persistent deadline state is absent")
    if (
        value.schema_version != 1
        or value.record_type != "glm52_worker_deadline_state_v1"
        or value.descriptor_file_sha256 != descriptor_file_sha256
        or _INSTANCE.fullmatch(value.instance_id) is None
        or type(value.allocation_ordinal) is not int
        or value.allocation_ordinal <= 0
        or value.last_completed_edge
        not in {"NONE", "T_MINUS_60", "T_MINUS_50"}
    ):
        raise Task10WorkerError("persistent deadline identity drifted")
    deadline = _utc(value.execution_deadline, "execution_deadline")
    if (
        _utc(value.stop_assignment_at, "stop_assignment_at")
        != deadline - timedelta(minutes=60)
        or _utc(value.graceful_stop_at, "graceful_stop_at")
        != deadline - timedelta(minutes=50)
        or _utc(value.retained_force_not_before, "retained_force_not_before")
        != deadline - timedelta(minutes=30)
        or value.state_body_sha256 != canonical_sha256(_deadline_body(value))
        or value.state_signature_sha256
        != _sign(value.state_body_sha256, signing_key)
    ):
        raise Task10WorkerError("persistent deadline state is corrupt or stale")
    return value


@dataclass(frozen=True)
class DeadlineDecision:
    phase: str
    allow_campaign_start: bool
    allow_new_assignments: bool
    recreate_stop_marker: bool
    persist_edge: Optional[str]
    systemctl_commands: Tuple[Tuple[str, ...], ...]
    checkpoint_sync_required: bool
    allow_restart: bool
    allow_force_kill: bool


def evaluate_deadline(
    *,
    now: datetime,
    state: DeadlineState,
    descriptor_file_sha256: str,
    signing_key: bytes,
    stop_marker_present: bool,
) -> DeadlineDecision:
    state = validate_deadline_state(
        state,
        descriptor_file_sha256=descriptor_file_sha256,
        signing_key=signing_key,
    )
    current = _utc(_utc_text(now), "now")
    t60 = _utc(state.stop_assignment_at, "stop_assignment_at")
    t50 = _utc(state.graceful_stop_at, "graceful_stop_at")
    t30 = _utc(state.retained_force_not_before, "retained_force_not_before")
    edge_rank = {"NONE": 0, "T_MINUS_60": 1, "T_MINUS_50": 2}
    wall_edge = (
        "NONE"
        if current < t60
        else ("T_MINUS_60" if current < t50 else "T_MINUS_50")
    )
    effective_edge = max(
        (state.last_completed_edge, wall_edge),
        key=edge_rank.__getitem__,
    )
    if effective_edge == "NONE":
        if stop_marker_present:
            raise Task10WorkerError("premature stop marker forbids campaign start")
        return DeadlineDecision(
            "BEFORE_T_MINUS_60",
            True,
            True,
            False,
            None,
            (),
            False,
            False,
            False,
        )
    if effective_edge == "T_MINUS_60":
        return DeadlineDecision(
            "T_MINUS_60",
            False,
            False,
            True,
            (
                "T_MINUS_60"
                if state.last_completed_edge == "NONE"
                else None
            ),
            (),
            False,
            False,
            False,
        )
    phase = "T_MINUS_50" if current < t30 else "T_MINUS_30"
    return DeadlineDecision(
        phase,
        False,
        False,
        True,
        (
            "T_MINUS_50"
            if state.last_completed_edge != "T_MINUS_50"
            else None
        ),
        (("systemctl", "stop", CAMPAIGN_UNIT),),
        True,
        False,
        False,
    )


def render_deadline_timer_dropin(state: DeadlineState) -> bytes:
    """Derive the persistent timer's two immutable wall-clock deliveries."""

    if not isinstance(state, DeadlineState):
        raise Task10WorkerError("deadline timer state is absent")
    _utc(state.stop_assignment_at, "stop_assignment_at")
    _utc(state.graceful_stop_at, "graceful_stop_at")
    return (
        "[Timer]\n"
        "OnCalendar=\n"
        "OnCalendar="
        + state.stop_assignment_at
        + "\n"
        "OnCalendar="
        + state.graceful_stop_at
        + "\n"
        "Persistent=true\n"
    ).encode("utf-8")


@dataclass(frozen=True)
class SystemdStopObservation:
    term_signals: Tuple[int, ...]
    kill_signals: Tuple[int, ...]
    remaining_control_group: Tuple[int, ...]
    active_state: str
    sub_state: str
    result: str
    exec_main_code: int
    exec_main_status: int


def simulate_systemd_stop(
    *,
    control_group_pids: Sequence[int],
    exited_after_sigterm: Sequence[int],
    elapsed_seconds: int,
) -> SystemdStopObservation:
    pids = tuple(control_group_pids)
    exited = tuple(exited_after_sigterm)
    if (
        not pids
        or any(type(item) is not int or item <= 0 for item in pids)
        or len(set(pids)) != len(pids)
        or any(item not in pids for item in exited)
        or type(elapsed_seconds) is not int
        or elapsed_seconds < 0
    ):
        raise Task10WorkerError("systemd stop simulation input is invalid")
    survivors = tuple(item for item in pids if item not in set(exited))
    timed_out = elapsed_seconds >= 1200 and bool(survivors)
    return SystemdStopObservation(
        term_signals=pids,
        kill_signals=(),
        remaining_control_group=survivors,
        active_state="deactivating" if survivors else "inactive",
        sub_state="stop-sigterm" if survivors else "dead",
        result="timeout" if timed_out else ("success" if not survivors else "stop-sigterm"),
        exec_main_code=1 if timed_out else 0,
        exec_main_status=15 if timed_out else 0,
    )


def bootstrap_systemd_sequence() -> Tuple[Tuple[str, ...], ...]:
    """The sole accepted unit activation/readback command sequence."""

    return (
        ("systemctl", "daemon-reload"),
        ("systemctl", "enable", DEADLINE_TIMER),
        ("systemctl", "start", DEADLINE_TIMER),
        ("systemctl", "start", CAMPAIGN_UNIT),
        ("systemctl", "is-enabled", DEADLINE_TIMER),
        ("systemctl", "show", DEADLINE_TIMER),
        ("systemctl", "show", CAMPAIGN_UNIT),
    )


def build_bootstrap_ready(
    *,
    unit_hashes: Mapping[str, str],
    deadline_state: DeadlineState,
    systemd_readback: Mapping[str, object],
    descriptor_file_sha256: str,
    descriptor_version_id: str,
    archive_file_sha256: str,
    archive_version_id: str,
    instance_id: str,
    allocation_ordinal: int,
    marker_publish_sequence: int,
) -> Mapping[str, object]:
    if type(unit_hashes) is not dict or set(unit_hashes) != set(UNIT_NAMES):
        raise Task10WorkerError("bootstrap unit identity set drifted")
    for name, value in unit_hashes.items():
        _sha(value, name + " hash")
    _sha(descriptor_file_sha256, "descriptor hash")
    _sha(archive_file_sha256, "archive hash")
    _version(descriptor_version_id, "descriptor VersionId")
    _version(archive_version_id, "archive VersionId")
    expected_readback_fields = {
        "campaign_active_state",
        "campaign_sub_state",
        "campaign_type",
        "campaign_notify_access",
        "campaign_fragment_path",
        "campaign_main_pid",
        "campaign_exec_main_pid",
        "campaign_main_argv",
        "deadline_timer_active_state",
        "deadline_timer_sub_state",
        "deadline_timer_enabled_state",
        "deadline_timer_next_elapse_usec_realtime",
        "deadline_timer_dropin_path",
        "deadline_timer_dropin_sha256",
        "deadline_timer_on_calendar",
    }
    expected_dropin_sha256 = hashlib.sha256(
        render_deadline_timer_dropin(deadline_state)
    ).hexdigest()
    main_pid = systemd_readback.get("campaign_main_pid")
    next_elapse = systemd_readback.get(
        "deadline_timer_next_elapse_usec_realtime"
    )
    if (
        type(systemd_readback) is not dict
        or set(systemd_readback) != expected_readback_fields
        or systemd_readback["campaign_active_state"] != "active"
        or systemd_readback["campaign_sub_state"] != "running"
        or systemd_readback["campaign_type"] != "notify"
        or systemd_readback["campaign_notify_access"] != "main"
        or systemd_readback["campaign_fragment_path"]
        != "/etc/systemd/system/" + CAMPAIGN_UNIT
        or type(main_pid) is not int
        or main_pid <= 0
        or systemd_readback["campaign_exec_main_pid"] != main_pid
        or systemd_readback["campaign_main_argv"]
        != list(CAMPAIGN_MAIN_ARGV)
        or systemd_readback["deadline_timer_active_state"] != "active"
        or systemd_readback["deadline_timer_sub_state"] != "waiting"
        or systemd_readback["deadline_timer_enabled_state"] != "enabled"
        or type(next_elapse) is not str
        or next_elapse != deadline_state.stop_assignment_at
        or systemd_readback["deadline_timer_dropin_path"]
        != DEADLINE_DROPIN_PATH
        or systemd_readback["deadline_timer_dropin_sha256"]
        != expected_dropin_sha256
        or systemd_readback["deadline_timer_on_calendar"]
        != [
            deadline_state.stop_assignment_at,
            deadline_state.graceful_stop_at,
        ]
        or marker_publish_sequence <= 0
        or deadline_state.instance_id != instance_id
        or deadline_state.allocation_ordinal != allocation_ordinal
    ):
        raise Task10WorkerError(
            "bootstrap start/readback or marker-last ordering failed"
        )
    body = {
        "schema_version": 1,
        "record_type": "glm52_bootstrap_ready_v1",
        "run_id": RUN_ID,
        "unit_file_sha256": dict(unit_hashes),
        "deadline_state_body_sha256": deadline_state.state_body_sha256,
        "deadline_state_signature_sha256": (
            deadline_state.state_signature_sha256
        ),
        "deadline_state_descriptor_file_sha256": (
            deadline_state.descriptor_file_sha256
        ),
        "systemd_readback": dict(systemd_readback),
        "descriptor_path": DESCRIPTOR_PATH,
        "descriptor_file_sha256": descriptor_file_sha256,
        "descriptor_version_id": descriptor_version_id,
        "execution_deadline": deadline_state.execution_deadline,
        "instance_id": instance_id,
        "allocation_ordinal": allocation_ordinal,
        "archive_file_sha256": archive_file_sha256,
        "archive_version_id": archive_version_id,
        "campaign_root": CAMPAIGN_ROOT,
        "resume_directories": [
            CAMPAIGN_ROOT + "/teacher-cache",
            CAMPAIGN_ROOT + "/spike-resume",
            CAMPAIGN_ROOT + "/training",
        ],
        "marker_publish_sequence": marker_publish_sequence,
    }
    return {**body, "bootstrap_body_sha256": canonical_sha256(body)}


def build_bootstrap_ledger_receipt(
    *,
    marker: Mapping[str, object],
    marker_file_sha256: str,
    ledger_record: Mapping[str, object],
    ledger_file_sha256: str,
    worker_descriptor: WorkerBootstrapDescriptor,
    instance_id: str,
    allocation_ordinal: int,
    ledger_record_s3_key: str,
    ledger_record_s3_version_id: str,
) -> Mapping[str, object]:
    """Authenticate the sole BOOTSTRAP append after marker release."""

    worker_descriptor = validate_worker_bootstrap_descriptor(
        worker_descriptor
    )
    marker_file_sha256 = _sha(
        marker_file_sha256,
        "bootstrap marker file hash",
    )
    ledger_file_sha256 = _sha(ledger_file_sha256, "ledger file hash")
    if type(marker) is not dict or type(ledger_record) is not dict:
        raise Task10WorkerError("bootstrap ledger evidence must be objects")
    marker_body = dict(marker)
    marker_body_sha = marker_body.pop("bootstrap_body_sha256", None)
    required_ledger_fields = {
        "record_type",
        "run_id",
        "phase",
        "input_identities",
        "output_identities",
        "timestamp",
        "execution_deadline",
        "gpu_spend_authority_sha256",
        "gpu_allocation_sha256",
        "prior_record_sha256",
        "record_sha256",
    }
    ledger_body = dict(ledger_record)
    ledger_record_sha = ledger_body.pop("record_sha256", None)
    ledger_record_raw = canonical_json_bytes(ledger_record) + b"\n"
    expected_record_key = (
        f"campaigns/{RUN_ID}/submissions/production/generations/"
        f"{worker_descriptor.generation:08d}/allocations/"
        f"{allocation_ordinal:08d}/"
        f"BOOTSTRAP_LEDGER_RECORD-{ledger_record_sha}.json"
    )
    expected_genesis = canonical_sha256(
        {
            "record_type": "glm52_sky_campaign_ledger_genesis_v2",
            "run_id": RUN_ID,
            "gpu_spend_authority_sha256": (
                worker_descriptor.task8_spend_authority_identity_sha256
            ),
        }
    )
    if (
        marker.get("record_type") != "glm52_bootstrap_ready_v1"
        or marker.get("run_id") != RUN_ID
        or marker_body_sha != canonical_sha256(marker_body)
        or marker.get("descriptor_file_sha256")
        != worker_descriptor.base_descriptor_file_sha256
        or marker.get("descriptor_version_id")
        != worker_descriptor.base_descriptor_version_id
        or marker.get("archive_file_sha256")
        != worker_descriptor.archive_identity_sha256
        or marker.get("archive_version_id")
        != worker_descriptor.repository_archive_version_id
        or marker.get("execution_deadline")
        != worker_descriptor.execution_deadline
        or marker.get("instance_id") != instance_id
        or marker.get("allocation_ordinal") != allocation_ordinal
        or type(marker.get("marker_publish_sequence")) is not int
        or marker["marker_publish_sequence"] <= 0
        or set(ledger_record) != required_ledger_fields
        or ledger_record_sha != canonical_sha256(ledger_body)
        or ledger_record.get("record_type")
        != "glm52_sky_campaign_transition_v2"
        or ledger_record.get("run_id") != RUN_ID
        or ledger_record.get("phase") != "BOOTSTRAP"
        or ledger_record.get("input_identities")
        != {
            "descriptor": worker_descriptor.campaign_identity_sha256,
            "repo_tar": worker_descriptor.archive_identity_sha256,
        }
        or ledger_record.get("output_identities")
        != {"bootstrap_report": marker_file_sha256}
        or ledger_record.get("execution_deadline")
        != worker_descriptor.execution_deadline
        or ledger_record.get("gpu_spend_authority_sha256")
        != worker_descriptor.task8_spend_authority_identity_sha256
        or ledger_record.get("gpu_allocation_sha256")
        != worker_descriptor.gpu_allocation_sha256
        or ledger_record.get("prior_record_sha256") != expected_genesis
        or ledger_file_sha256
        != hashlib.sha256(ledger_record_raw).hexdigest()
        or ledger_record_s3_key != expected_record_key
    ):
        raise Task10WorkerError(
            "BOOTSTRAP ledger append or marker ancestry drifted"
        )
    body = {
        "schema_version": 1,
        "record_type": "glm52_task10_bootstrap_ledger_receipt_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "campaign_identity_sha256": (
            worker_descriptor.campaign_identity_sha256
        ),
        "activation_id": worker_descriptor.activation_id,
        "generation": worker_descriptor.generation,
        "generation_text": worker_descriptor.generation_text,
        "instance_id": instance_id,
        "allocation_ordinal": allocation_ordinal,
        "marker_file_sha256": marker_file_sha256,
        "marker_body_sha256": marker_body_sha,
        "marker_publish_sequence": marker["marker_publish_sequence"],
        "ledger_append_sequence": marker["marker_publish_sequence"] + 1,
        "ledger_record_index": 0,
        "ledger_record_sha256": ledger_record_sha,
        "ledger_prior_record_sha256": expected_genesis,
        "ledger_file_sha256": ledger_file_sha256,
        "ledger_record_s3_key": ledger_record_s3_key,
        "ledger_record_s3_version_id": _version(
            ledger_record_s3_version_id,
            "BOOTSTRAP ledger record VersionId",
        ),
        "ledger_record_file_sha256": hashlib.sha256(
            ledger_record_raw
        ).hexdigest(),
        "ledger_record_checksum_sha256_base64": base64.b64encode(
            hashlib.sha256(ledger_record_raw).digest()
        ).decode("ascii"),
        "execution_deadline": worker_descriptor.execution_deadline,
        "gpu_allocation_sha256": (
            worker_descriptor.gpu_allocation_sha256
        ),
        "gpu_spend_authority_sha256": (
            worker_descriptor.task8_spend_authority_identity_sha256
        ),
        "worker_descriptor_body_sha256": (
            worker_descriptor.descriptor_body_sha256
        ),
    }
    return {
        **body,
        "receipt_body_sha256": canonical_sha256(body),
    }


def render_graceful_stop_document() -> Mapping[str, object]:
    """Render the exact parameterless worker-drain Systems Manager document."""

    return {
        "schemaVersion": "2.2",
        "description": GRACEFUL_STOP_DOCUMENT,
        "parameters": {},
        "mainSteps": [
            {
                "action": "aws:runShellScript",
                "name": "fixedGracefulStop",
                "inputs": {
                    "runCommand": [
                        "set -eu",
                        "install -d -m 0755 /run/keep-glm52",
                        "umask 022",
                        ": > /run/keep-glm52/STOP.tmp",
                        "mv -f /run/keep-glm52/STOP.tmp /run/keep-glm52/STOP",
                        "systemctl stop keep-glm52-campaign.service",
                        (
                            "/opt/keep-campaign/repo/aws/glm52-gpu/scripts/"
                            "collect_task10_ssm_graceful_stop.py"
                        ),
                    ]
                },
            }
        ],
    }


def render_worker_drain_iam(
    *, activation_id: str
) -> Mapping[str, object]:
    activation_id = _text(activation_id, "activation_id")
    document_arn = (
        "arn:aws:ssm:us-west-2:246813579024:document/"
        + GRACEFUL_STOP_DOCUMENT
    )
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "OnlyExactParameterlessWorkerDrain",
                "Effect": "Allow",
                "Action": "ssm:SendCommand",
                "Resource": [
                    document_arn,
                    "arn:aws:ec2:us-west-2:246813579024:instance/*",
                ],
                "Condition": {
                    "StringEquals": {
                        "ssm:resourceTag/RunId": RUN_ID,
                        "ssm:resourceTag/activation-id": activation_id,
                    }
                },
            },
            {
                "Sid": "DenyEverySession",
                "Effect": "Deny",
                "Action": [
                    "ssm:StartSession",
                    "ssm:ResumeSession",
                    "ssm:TerminateSession",
                ],
                "Resource": "*",
            },
        ],
    }


_GRACEFUL_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "run_id",
        "unit_file_sha256",
        "script_file_sha256",
        "ActiveState",
        "SubState",
        "Result",
        "ExecMainCode",
        "ExecMainStatus",
        "control_group_pids",
        "stop_file_identity_sha256",
        "checkpoint_identity_sha256",
        "latest_marker_identity_sha256",
        "campaign_terminal_marker_identity_sha256",
        "ssm_command_id",
        "authority",
        "graceful_stop_body_sha256",
    }
)


def build_graceful_stop_evidence(
    *,
    unit_hashes: Mapping[str, str],
    script_hashes: Mapping[str, str],
    active_state: str,
    sub_state: str,
    result: str,
    exec_main_code: int,
    exec_main_status: int,
    control_group_pids: Sequence[int],
    stop_file_identity_sha256: str,
    checkpoint_identity_sha256: str,
    latest_marker_identity_sha256: str,
    campaign_terminal_marker_identity_sha256: Optional[str],
    ssm_command_id: Optional[str],
    authority: str,
) -> Mapping[str, object]:
    body = {
        "schema_version": 1,
        "record_type": "glm52_worker_graceful_stop_v1",
        "run_id": RUN_ID,
        "unit_file_sha256": dict(unit_hashes),
        "script_file_sha256": dict(script_hashes),
        "ActiveState": active_state,
        "SubState": sub_state,
        "Result": result,
        "ExecMainCode": exec_main_code,
        "ExecMainStatus": exec_main_status,
        "control_group_pids": list(control_group_pids),
        "stop_file_identity_sha256": stop_file_identity_sha256,
        "checkpoint_identity_sha256": checkpoint_identity_sha256,
        "latest_marker_identity_sha256": latest_marker_identity_sha256,
        "campaign_terminal_marker_identity_sha256": (
            campaign_terminal_marker_identity_sha256
        ),
        "ssm_command_id": ssm_command_id,
        "authority": authority,
    }
    value = {
        **body,
        "graceful_stop_body_sha256": canonical_sha256(body),
    }
    return validate_graceful_stop_evidence(
        value,
        expected_unit_hashes=unit_hashes,
        expected_script_hashes=script_hashes,
    )


def validate_graceful_stop_evidence(
    value: object,
    *,
    expected_unit_hashes: Mapping[str, str],
    expected_script_hashes: Mapping[str, str],
) -> Mapping[str, object]:
    if (
        type(value) is not dict
        or set(value) != _GRACEFUL_FIELDS
        or set(expected_unit_hashes) != set(UNIT_NAMES)
        or set(expected_script_hashes) != set(GRACEFUL_SCRIPT_NAMES)
    ):
        raise Task10WorkerError("graceful-stop evidence field set drifted")
    body = dict(value)
    observed_hash = body.pop("graceful_stop_body_sha256")
    if (
        value["schema_version"] != 1
        or value["record_type"] != "glm52_worker_graceful_stop_v1"
        or value["run_id"] != RUN_ID
        or value["unit_file_sha256"] != dict(expected_unit_hashes)
        or value["script_file_sha256"] != dict(expected_script_hashes)
        or value["ActiveState"] != "inactive"
        or value["SubState"] != "dead"
        or value["Result"] != "success"
        or value["ExecMainCode"] != 1
        or value["ExecMainStatus"] != 0
        or value["control_group_pids"] != []
        or value["authority"] not in {"LOCAL_TIMER", "SSM"}
        or observed_hash != canonical_sha256(body)
    ):
        raise Task10WorkerError("graceful-stop systemd evidence failed")
    for field in (
        "stop_file_identity_sha256",
        "checkpoint_identity_sha256",
        "latest_marker_identity_sha256",
    ):
        _sha(value[field], field)
    terminal = value["campaign_terminal_marker_identity_sha256"]
    if terminal is not None:
        _sha(terminal, "campaign terminal marker")
    command = value["ssm_command_id"]
    if value["authority"] == "LOCAL_TIMER":
        if command is not None:
            raise Task10WorkerError("timer evidence must carry exact null command")
    elif type(command) is not str or _SSM_COMMAND.fullmatch(command) is None:
        raise Task10WorkerError("SSM drain command identity is invalid")
    return value


def promote_graceful_stop_evidence_to_ssm(
    value: object,
    *,
    ssm_command_id: str,
    expected_unit_hashes: Mapping[str, str],
    expected_script_hashes: Mapping[str, str],
) -> Mapping[str, object]:
    """Bind one authenticated post-stop observation to its SSM command."""

    value = validate_graceful_stop_evidence(
        value,
        expected_unit_hashes=expected_unit_hashes,
        expected_script_hashes=expected_script_hashes,
    )
    if value["authority"] != "LOCAL_TIMER" or value["ssm_command_id"] is not None:
        raise Task10WorkerError(
            "SSM observation source must carry null local authority"
        )
    return build_graceful_stop_evidence(
        unit_hashes=expected_unit_hashes,
        script_hashes=expected_script_hashes,
        active_state=str(value["ActiveState"]),
        sub_state=str(value["SubState"]),
        result=str(value["Result"]),
        exec_main_code=int(value["ExecMainCode"]),
        exec_main_status=int(value["ExecMainStatus"]),
        control_group_pids=value["control_group_pids"],
        stop_file_identity_sha256=str(
            value["stop_file_identity_sha256"]
        ),
        checkpoint_identity_sha256=str(
            value["checkpoint_identity_sha256"]
        ),
        latest_marker_identity_sha256=str(
            value["latest_marker_identity_sha256"]
        ),
        campaign_terminal_marker_identity_sha256=value[
            "campaign_terminal_marker_identity_sha256"
        ],
        ssm_command_id=ssm_command_id,
        authority="SSM",
    )


__all__ = [
    "ACCOUNT_ID",
    "CAMPAIGN_MAIN_ARGV",
    "CAMPAIGN_ROOT",
    "CAMPAIGN_UNIT",
    "ClosedWireRequest",
    "DEADLINE_STATE_PATH",
    "DEADLINE_DROPIN_PATH",
    "DEADLINE_TIMER",
    "DEADLINE_UNIT",
    "DESCRIPTOR_PATH",
    "DeadlineDecision",
    "DeadlineState",
    "GRACEFUL_STOP_DOCUMENT",
    "GRACEFUL_SCRIPT_NAMES",
    "H100_READY_KEY",
    "JOBS_LAUNCH_BODY_FIELDS",
    "MountFreeTaskInputs",
    "RetainedSsmGracefulStopHandoff",
    "RUN_ID",
    "SKY_ENDPOINT",
    "SKYPILOT_API_VERSION",
    "SKYPILOT_VERSION",
    "STOP_PATH",
    "SystemdStopObservation",
    "Task10WorkerError",
    "Task12WorkerDrainAuthority",
    "UNIT_NAMES",
    "WorkerBootstrapDescriptor",
    "WorkerInstanceObservation",
    "WorkerRuntimeAuthority",
    "WORKER_DRAIN_ROLE",
    "WORKER_DRAIN_ROLE_ARN",
    "WORKER_BOOTSTRAP_DESCRIPTOR_PATH",
    "WORKER_INSTANCE_OBSERVATION_PATH",
    "bootstrap_systemd_sequence",
    "build_bootstrap_ready",
    "build_bootstrap_ledger_receipt",
    "build_closed_wire_request",
    "build_deadline_state",
    "build_graceful_stop_evidence",
    "build_jobs_launch_body",
    "build_retained_ssm_graceful_stop_handoff",
    "build_task12_worker_drain_authority",
    "build_worker_bootstrap_descriptor",
    "build_worker_instance_observation",
    "build_worker_runtime_authority",
    "evaluate_deadline",
    "render_graceful_stop_document",
    "render_deadline_timer_dropin",
    "render_mount_free_task",
    "render_worker_drain_iam",
    "render_worker_units",
    "retained_ssm_graceful_stop_handoff_from_mapping",
    "simulate_systemd_stop",
    "task_environment",
    "task12_worker_drain_authority_from_mapping",
    "validate_closed_wire_request",
    "validate_deadline_state",
    "validate_graceful_stop_evidence",
    "promote_graceful_stop_evidence_to_ssm",
    "validate_jobs_launch_body",
    "validate_retained_ssm_graceful_stop_handoff",
    "validate_task_inputs",
    "validate_task12_worker_drain_authority",
    "validate_worker_units",
    "validate_worker_bootstrap_descriptor",
    "validate_worker_instance_observation",
    "worker_bootstrap_descriptor_from_mapping",
    "worker_instance_observation_from_mapping",
    "worker_unit_hashes",
]
