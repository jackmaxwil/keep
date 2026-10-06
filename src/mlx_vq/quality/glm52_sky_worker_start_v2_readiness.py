"""Pure source-readiness authority for the GLM-5.2 worker-start-v2 path."""

from __future__ import annotations

import datetime as datetime_module
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath
from typing import Mapping

if not hasattr(datetime_module, "UTC"):
    datetime_module.UTC = timezone.utc

from mlx_vq.quality.glm52_qualification_submission_ready import (  # noqa: E402
    qualification_submission_ready_s3_key,
    rehearsal_evidence_s3_key,
)
from mlx_vq.quality.glm52_sky_campaign import (  # noqa: E402
    validate_sky_campaign_descriptor,
)
from mlx_vq.quality.glm52_sky_must_start_dynamic import (  # noqa: E402
    dynamic_v2_job_binding_s3_key,
)

_SCHEMA_VERSION = 1
_RECORD_TYPE = "glm52_sky_worker_start_v2_readiness_v1"
_DIGEST_FIELD = "worker_start_v2_readiness_body_sha256"
_ACCOUNT_ID = "246813579024"
_REGION = "us-west-2"
_MANAGED_MODE = "qualification"
_SKYPILOT_VERSION = "0.13.0"
_SKY_TASK_NAME = "glm52-campaign"
_INSTANCE_TYPE = "p5.48xlarge"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_AMI_ID = re.compile(r"^ami-[0-9a-f]{8,17}$")
_IAM_ROLE_COMPONENT = r"[A-Za-z0-9+=,.@_-]+"
_IAM_ROLE_ARN = re.compile(
    rf"^arn:aws:iam::{_ACCOUNT_ID}:role/"
    rf"{_IAM_ROLE_COMPONENT}(?:/{_IAM_ROLE_COMPONENT})*$"
)
_BUCKET = re.compile(
    r"^(?!xn--)(?!sthree-)(?!amzn_s3_demo_)(?!.*\.\.)"
    r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$"
)

_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "bucket",
        "run_id",
        "managed_mode",
        "descriptor_key",
        "descriptor_file_sha256",
        "descriptor_body_sha256",
        "campaign_identity_sha256",
        "repo_tar_key",
        "repo_tar_sha256",
        "qualification_submission_ready_key",
        "qualification_submission_ready_file_sha256",
        "qualification_submission_ready_body_sha256",
        "rehearsal_evidence_key",
        "rehearsal_evidence_file_sha256",
        "rehearsal_evidence_body_sha256",
        "intent_key",
        "intent_file_sha256",
        "intent_body_sha256",
        "skypilot_version",
        "sky_task_name",
        "sky_task_file_sha256",
        "sky_job_name",
        "must_start_by",
        "instance_type",
        "instance_count",
        "use_spot",
        "image_id",
        "controller_identity",
        "worker_identity",
        "source_contract_sha256",
        "source_closure_sha256",
        "source_files",
        _DIGEST_FIELD,
    }
)

_READINESS_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "managed_mode",
        "descriptor_key",
        "descriptor_file_sha256",
        "descriptor_body_sha256",
        "campaign_identity_sha256",
        "approval_sha256",
        "approval_body_sha256",
        "repo_tar_sha256",
        "seed_descriptor_key",
        "seed_descriptor_file_sha256",
        "seed_descriptor_body_sha256",
        "seed_campaign_identity_sha256",
        "seed_repo_tar_sha256",
        "seed_descriptor",
        "staged_readiness_key",
        "staged_readiness_file_sha256",
        "staged_readiness_body_sha256",
        "cache_seed_acceptance_key",
        "cache_seed_acceptance_file_sha256",
        "cache_seed_acceptance_body_sha256",
        "gpu_spend_snapshot_key",
        "gpu_spend_snapshot_sha256",
        "gpu_spend_snapshot_body_sha256",
        "gpu_spend_ledger_tip_record_sha256",
        "remaining_gpu_seconds",
        "remaining_gpu_cost_usd",
        "qualification_allowance_seconds",
        "qualification_allowance_cost_usd",
        "rehearsal_evidence_key",
        "rehearsal_evidence_sha256",
        "rehearsal_evidence_body_sha256",
        "sky_task_name",
        "sky_job_name",
        "must_start_by",
        "allowed_gpu_seconds",
        "allowed_gpu_cost_usd",
        "built_at",
        "readiness_body_sha256",
    }
)

_REHEARSAL_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "status",
        "run_id",
        "campaign_identity_sha256",
        "descriptor_key",
        "descriptor_file_sha256",
        "descriptor_body_sha256",
        "repo_tar_sha256",
        "bootstrap_receipt_file_sha256",
        "staged_readiness_key",
        "staged_readiness_file_sha256",
        "staged_readiness_body_sha256",
        "artifact_inventory_key",
        "artifact_inventory_file_sha256",
        "artifact_inventory_body_sha256",
        "artifact_audit_key",
        "artifact_audit_file_sha256",
        "extracted_repo_path",
        "production_repo_path",
        "production_resume_root",
        "skypilot_task_path",
        "skypilot_task_file_sha256",
        "skypilot_config_file_sha256",
        "skypilot_validation_file_sha256",
        "skypilot_version",
        "skypilot_task_name",
        "completed_at",
        "rehearsal_body_sha256",
    }
)

_READINESS_SHA_FIELDS = frozenset(
    field for field in _READINESS_FIELDS if field.endswith("_sha256")
)
_REHEARSAL_SHA_FIELDS = frozenset(
    field for field in _REHEARSAL_FIELDS if field.endswith("_sha256")
)
_READINESS_KEY_FIELDS = frozenset(
    {
        "descriptor_key",
        "seed_descriptor_key",
        "staged_readiness_key",
        "cache_seed_acceptance_key",
        "gpu_spend_snapshot_key",
        "rehearsal_evidence_key",
    }
)
_REHEARSAL_KEY_FIELDS = frozenset(
    {
        "descriptor_key",
        "staged_readiness_key",
        "artifact_inventory_key",
        "artifact_audit_key",
    }
)

_DESCRIPTOR_READINESS_BINDINGS = (
    ("account_id", "account_id"),
    ("region", "region"),
    ("run_id", "run_id"),
    ("campaign_descriptor_key", "descriptor_key"),
    ("descriptor_body_sha256", "descriptor_body_sha256"),
    ("campaign_identity_sha256", "campaign_identity_sha256"),
    ("approval_sha256", "approval_sha256"),
    ("repo_tar_sha256", "repo_tar_sha256"),
    ("task_name", "sky_task_name"),
    ("must_start_by", "must_start_by"),
)
_DESCRIPTOR_REHEARSAL_BINDINGS = (
    ("run_id", "run_id"),
    ("campaign_identity_sha256", "campaign_identity_sha256"),
    ("campaign_descriptor_key", "descriptor_key"),
    ("descriptor_body_sha256", "descriptor_body_sha256"),
    ("repo_tar_sha256", "repo_tar_sha256"),
    ("skypilot_version", "skypilot_version"),
    ("task_name", "skypilot_task_name"),
)
_INTENT_READINESS_FIELDS = (
    "account_id",
    "region",
    "run_id",
    "managed_mode",
    "descriptor_key",
    "descriptor_file_sha256",
    "descriptor_body_sha256",
    "campaign_identity_sha256",
    "approval_sha256",
    "approval_body_sha256",
    "repo_tar_sha256",
    "cache_seed_acceptance_key",
    "cache_seed_acceptance_file_sha256",
    "cache_seed_acceptance_body_sha256",
    "gpu_spend_snapshot_key",
    "gpu_spend_snapshot_sha256",
    "gpu_spend_snapshot_body_sha256",
    "gpu_spend_ledger_tip_record_sha256",
    "remaining_gpu_seconds",
    "remaining_gpu_cost_usd",
    "qualification_allowance_seconds",
    "qualification_allowance_cost_usd",
    "sky_job_name",
    "must_start_by",
)

_SOURCE_MEMBERS = (
    (
        "aws/glm52-gpu/lambda/sky_worker_start_v2_coordinator.py",
        "coordinator_raw",
        "efa7c48ad259e10c6ce0e4cb1db9ff0a6e6a9fa203274329a7c2eb5203366406",
    ),
    (
        "aws/glm52-gpu/skypilot/bootstrap_campaign.sh",
        "bootstrap_raw",
        "8073ec6e55184073eece202d1726cbb1e7fe83c6f95abee927114b713f4366fc",
    ),
    (
        "aws/glm52-gpu/skypilot/glm52-campaign.yaml",
        "sky_task_raw",
        "cc0a56a34684d297e55827cf017802f4e8b16bd8dd0bb18fa1e1c0a197d787e5",
    ),
    (
        "aws/glm52-gpu/skypilot/publish_worker_start_v2.py",
        "publisher_raw",
        "4d1d47aef6dd211c20b52f05ede9bd6d32c35433e305f2b1bd56b48c83b9051e",
    ),
    (
        "aws/glm52-gpu/skypilot/run_h100_qualification.sh",
        "h100_qualification_worker_raw",
        "870d85585ea8037b9e6a92647da585ce537bd15768fa298c330e6894ddd87686",
    ),
    (
        "aws/glm52-gpu/skypilot/run_managed_campaign.sh",
        "managed_entrypoint_raw",
        "ac2400441f2b6ecac6bf6ad9cc46fba5abb0386e1b2cf6046527f8c3c3ba9a9c",
    ),
    (
        "src/glm52_enforcement/glm52_h100_qualification.py",
        "h100_qualification_native_raw",
        "f1720b675a2ae84b23be6a34a8bd6b84e0e65cabab260c5a1c79707096c11ff5",
    ),
    (
        "src/glm52_enforcement/glm52_sky_campaign.py",
        "campaign_policy_native_raw",
        "f023eaeddf8fac73fa6546e3c4a0b60dd32e5266f06e1519fe21e0444d445d03",
    ),
    (
        "src/glm52_enforcement/glm52_sky_must_start.py",
        "must_start_policy_native_raw",
        "e910d3d03f7b30b7b9b668e214b61ecc7cbd641a53dedabd620b8e14573d33f3",
    ),
    (
        "src/mlx_vq/quality/glm52_h100_qualification.py",
        "h100_qualification_raw",
        "b9418985afdd3f9ef44a3dbf44731cb149a2bc8e17194c0bf9fab9d6a88b329a",
    ),
    (
        "src/mlx_vq/quality/glm52_sky_campaign.py",
        "campaign_policy_raw",
        "77412115f6b8ce325177864418d838e0335d536329e8e7565d7811bcc806cd94",
    ),
    (
        "src/mlx_vq/quality/glm52_sky_must_start.py",
        "must_start_policy_raw",
        "478ac3a16ce8a77a4e844c1e0a3dca99454d50e7233fd7cd67dfa184b486860a",
    ),
    (
        "src/mlx_vq/quality/glm52_sky_must_start_dynamic.py",
        "dynamic_policy_raw",
        "527019a41b54f810e9f98734c2ca99acc5a7b370344adf48c80f46f925396e18",
    ),
    (
        "src/mlx_vq/quality/glm52_sky_worker_must_start_v2.py",
        "worker_policy_raw",
        "8dfad6682d8afc9774dc31385cfc9ecc8c8e2f80d9a535e3c8d4cf6cf0fa97df",
    ),
)
_SOURCE_CLOSURE_SHA256 = (
    "6ec127962854b61002eec7f04bef935edb35ef6acd84e2b1c814beb0de822b2c"
)

_FORBIDDEN_TOKENS = (
    "publish_must_start_latch.py",
    "GLM52_IMMUTABLE_SUBMISSION_",
    "IMMUTABLE_SUBMISSION.json",
    "TIMELY_START_ACCEPTED.json",
    "TIMELY_START_LATCH.json",
    "JOB_BINDING.json",
    "verify-accepted",
)

_PYTHON_REQUIRED_TOKENS = {
    "publisher_raw": (
        "def publish_worker_start_latch_v2(",
        "def replay_worker_start_accepted_v2(",
        "def verify_worker_start_receipt_v2(",
        'commands.add_parser("publish-latch-v2")',
        'commands.add_parser("replay-accepted-v2")',
        'commands.add_parser("verify-receipt-v2")',
    ),
    "campaign_policy_raw": (
        "def build_sky_campaign_descriptor(",
        "def validate_sky_campaign_descriptor(",
        "def require_approved_aws_identity(",
    ),
    "must_start_policy_raw": (
        "def build_must_start_controller_observation(",
        "def validate_must_start_controller_observation(",
    ),
    "dynamic_policy_raw": (
        "def resolve_dynamic_v2_job_binding(",
        "def dynamic_v2_job_binding_s3_key(",
        "def build_dynamic_v2_job_binding(",
        "def validate_dynamic_v2_job_binding(",
    ),
    "worker_policy_raw": (
        "def build_worker_start_latch_v2(",
        "def validate_worker_start_latch_v2(",
        "def worker_start_latch_v2_s3_key(",
        "def decide_worker_start_acceptance_v2(",
        "def build_worker_start_accepted_v2(",
        "def validate_worker_start_accepted_v2(",
    ),
    "coordinator_raw": ("def coordinate_worker_start_acceptance_v2(",),
}

_TASK_MOUNTS = (
    (
        "/tmp/glm52-worker-start-v2/publish_worker_start_v2.py: "
        "aws/glm52-gpu/skypilot/publish_worker_start_v2.py"
    ),
    (
        "/tmp/glm52-worker-start-v2/glm52_sky_campaign.py: "
        "src/mlx_vq/quality/glm52_sky_campaign.py"
    ),
    (
        "/tmp/glm52-worker-start-v2/glm52_sky_campaign_native.py: "
        "src/glm52_enforcement/glm52_sky_campaign.py"
    ),
    (
        "/tmp/glm52-worker-start-v2/glm52_sky_must_start.py: "
        "src/mlx_vq/quality/glm52_sky_must_start.py"
    ),
    (
        "/tmp/glm52-worker-start-v2/glm52_sky_must_start_native.py: "
        "src/glm52_enforcement/glm52_sky_must_start.py"
    ),
    (
        "/tmp/glm52-worker-start-v2/glm52_sky_must_start_dynamic.py: "
        "src/mlx_vq/quality/glm52_sky_must_start_dynamic.py"
    ),
    (
        "/tmp/glm52-worker-start-v2/glm52_sky_worker_must_start_v2.py: "
        "src/mlx_vq/quality/glm52_sky_worker_must_start_v2.py"
    ),
    (
        "/tmp/glm52-worker-start-v2/sky_worker_start_v2_coordinator.py: "
        "aws/glm52-gpu/lambda/sky_worker_start_v2_coordinator.py"
    ),
    (
        "/tmp/glm52-worker-start-v2/run_h100_qualification.sh: "
        "aws/glm52-gpu/skypilot/run_h100_qualification.sh"
    ),
    (
        "/tmp/glm52-worker-start-v2/glm52_h100_qualification.py: "
        "src/mlx_vq/quality/glm52_h100_qualification.py"
    ),
    (
        "/tmp/glm52-worker-start-v2/glm52_h100_qualification_native.py: "
        "src/glm52_enforcement/glm52_h100_qualification.py"
    ),
)

_TASK_HASH_ARGUMENTS = (
    (
        "--publisher-file-sha256 "
        "4d1d47aef6dd211c20b52f05ede9bd6d32c35433e305f2b1bd56b48c83b9051e"
    ),
    (
        "--campaign-policy-file-sha256 "
        "77412115f6b8ce325177864418d838e0335d536329e8e7565d7811bcc806cd94"
    ),
    (
        "--campaign-policy-native-file-sha256 "
        "f023eaeddf8fac73fa6546e3c4a0b60dd32e5266f06e1519fe21e0444d445d03"
    ),
    (
        "--must-start-policy-file-sha256 "
        "478ac3a16ce8a77a4e844c1e0a3dca99454d50e7233fd7cd67dfa184b486860a"
    ),
    (
        "--must-start-policy-native-file-sha256 "
        "e910d3d03f7b30b7b9b668e214b61ecc7cbd641a53dedabd620b8e14573d33f3"
    ),
    (
        "--dynamic-policy-file-sha256 "
        "527019a41b54f810e9f98734c2ca99acc5a7b370344adf48c80f46f925396e18"
    ),
    (
        "--worker-policy-file-sha256 "
        "8dfad6682d8afc9774dc31385cfc9ecc8c8e2f80d9a535e3c8d4cf6cf0fa97df"
    ),
    (
        "--coordinator-file-sha256 "
        "efa7c48ad259e10c6ce0e4cb1db9ff0a6e6a9fa203274329a7c2eb5203366406"
    ),
)

_ORDERED_LANDMARKS = {
    "sky_task_raw": (
        "publish-latch-v2",
        "replay-accepted-v2",
        'aws s3 cp "$GLM52_GPU_SPEND_APPROVAL_S3_URI"',
        'aws s3 cp "$REPO_URI"',
        ("cmp -s /tmp/glm52-worker-start-v2/publish_worker_start_v2.py"),
        "cmp -s /tmp/glm52-worker-start-v2/glm52_sky_campaign.py",
        "cmp -s /tmp/glm52-worker-start-v2/glm52_sky_campaign_native.py",
        "cmp -s /tmp/glm52-worker-start-v2/glm52_sky_must_start.py",
        "cmp -s /tmp/glm52-worker-start-v2/glm52_sky_must_start_native.py",
        "cmp -s /tmp/glm52-worker-start-v2/glm52_sky_must_start_dynamic.py",
        "cmp -s /tmp/glm52-worker-start-v2/glm52_sky_worker_must_start_v2.py",
        "cmp -s /tmp/glm52-worker-start-v2/sky_worker_start_v2_coordinator.py",
        ("/opt/keep-campaign/repo/aws/glm52-gpu/skypilot/bootstrap_campaign.sh"),
    ),
    "bootstrap_raw": (
        "frozen_sources = {",
        "verify-receipt-v2",
        "prepare_nvme_storage.sh",
        "gpu_env_setup.sh",
    ),
    "managed_entrypoint_raw": (
        "RECEIPT_FINGERPRINT=",
        "INTENT_FINGERPRINT=",
        "verify-receipt-v2",
        "GLM52_SKY_CONTROLLER_JOB_ID=",
        "restore_gpu_spend_ledger.sh",
        "http://169.254.169.254/latest/api/token",
        'manage_gpu_spend.py" start',
    ),
}

_EXACT_SOURCE_CONSTANTS = {
    "sky_task_raw": (
        "infra: aws/us-west-2",
        "instance_type: p5.48xlarge",
        "use_spot: false",
        "GLM52_MANAGED_MODE: qualification",
        'test "$GLM52_MANAGED_MODE" = qualification',
        'test "$ACCOUNT" = 246813579024',
    ),
    "bootstrap_raw": (
        'test "$MANAGED_MODE" = qualification',
        "assert_rnd_aws_account.py",
        "/opt/keep-campaign/repo",
        "/mnt/nvme/glm52-campaign",
    ),
    "managed_entrypoint_raw": (
        "REGION=us-west-2",
        'test "$MANAGED_MODE" = qualification',
        "Name=instance-type,Values=p5.48xlarge",
    ),
    "campaign_policy_raw": (
        'APPROVED_ACCOUNT_ID = "246813579024"',
        'APPROVED_REGION = "us-west-2"',
        'APPROVED_INSTANCE_TYPE = "p5.48xlarge"',
    ),
}

_SOURCE_CONTRACT = {
    "schema_version": 1,
    "python_required_tokens": _PYTHON_REQUIRED_TOKENS,
    "task_file_mounts": _TASK_MOUNTS,
    "task_source_hash_arguments": _TASK_HASH_ARGUMENTS,
    "ordered_landmarks": _ORDERED_LANDMARKS,
    "exact_source_constants": _EXACT_SOURCE_CONSTANTS,
    "forbidden_tokens": _FORBIDDEN_TOKENS,
}


class WorkerStartV2ReadinessError(ValueError):
    """The worker-start-v2 source closure is malformed, foreign, or drifted."""


@dataclass(frozen=True)
class WorkerStartV2ReadinessArtifact:
    key: str
    raw: bytes


@dataclass(frozen=True)
class WorkerStartV2ReadinessAuthorities:
    descriptor: WorkerStartV2ReadinessArtifact
    qualification_submission_ready: WorkerStartV2ReadinessArtifact
    rehearsal_evidence: WorkerStartV2ReadinessArtifact
    intent: WorkerStartV2ReadinessArtifact


@dataclass(frozen=True)
class WorkerStartV2SourceSnapshot:
    publisher_raw: bytes
    h100_qualification_worker_raw: bytes
    campaign_policy_raw: bytes
    campaign_policy_native_raw: bytes
    h100_qualification_raw: bytes
    h100_qualification_native_raw: bytes
    must_start_policy_raw: bytes
    must_start_policy_native_raw: bytes
    dynamic_policy_raw: bytes
    worker_policy_raw: bytes
    coordinator_raw: bytes
    sky_task_raw: bytes
    bootstrap_raw: bytes
    managed_entrypoint_raw: bytes


_ARTIFACT_INSTANCE_FIELDS = frozenset({"key", "raw"})
_AUTHORITIES_INSTANCE_FIELDS = frozenset(
    {
        "descriptor",
        "qualification_submission_ready",
        "rehearsal_evidence",
        "intent",
    }
)
_SOURCE_SNAPSHOT_INSTANCE_FIELDS = (
    "publisher_raw",
    "h100_qualification_worker_raw",
    "campaign_policy_raw",
    "campaign_policy_native_raw",
    "h100_qualification_raw",
    "h100_qualification_native_raw",
    "must_start_policy_raw",
    "must_start_policy_native_raw",
    "dynamic_policy_raw",
    "worker_policy_raw",
    "coordinator_raw",
    "sky_task_raw",
    "bootstrap_raw",
    "managed_entrypoint_raw",
)


def _require_exact_instance_state(
    value: object,
    *,
    exact_type: type,
    expected_fields: frozenset[str],
    label: str,
) -> dict[str, object]:
    if type(value) is not exact_type:
        raise WorkerStartV2ReadinessError(f"{label} class is not exact")
    try:
        state = vars(value)
    except TypeError as error:
        raise WorkerStartV2ReadinessError(
            f"{label} instance state is unavailable"
        ) from error
    if type(state) is not dict or set(state) != expected_fields:
        missing = sorted(expected_fields - set(state))
        unknown = sorted(set(state) - expected_fields)
        raise WorkerStartV2ReadinessError(
            f"{label} instance schema mismatch: missing={missing}, unknown={unknown}"
        )
    return state


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as error:
        raise WorkerStartV2ReadinessError(
            "value is not canonical finite ASCII JSON"
        ) from error


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _body_sha(value: Mapping[str, object]) -> str:
    return _sha(_canonical(value))


def _file_sha(value: Mapping[str, object]) -> str:
    return _sha(_canonical(value) + b"\n")


def _reject_constant(value: str) -> object:
    raise WorkerStartV2ReadinessError(f"non-finite JSON constant is forbidden: {value}")


def _object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise WorkerStartV2ReadinessError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _plain_json(value: object, *, label: str) -> None:
    active_containers: set[int] = set()
    stack: list[tuple[object, bool]] = [(value, False)]
    while stack:
        item, leaving = stack.pop()
        if leaving:
            active_containers.remove(id(item))
            continue
        if type(item) is dict:
            identity = id(item)
            if identity in active_containers:
                raise WorkerStartV2ReadinessError(f"{label} is recursive")
            active_containers.add(identity)
            stack.append((item, True))
            children: list[object] = []
            for key, child in item.items():
                if type(key) is not str:
                    raise WorkerStartV2ReadinessError(f"{label} has a non-string key")
                children.append(child)
            stack.extend((child, False) for child in reversed(children))
            continue
        if type(item) is list:
            identity = id(item)
            if identity in active_containers:
                raise WorkerStartV2ReadinessError(f"{label} is recursive")
            active_containers.add(identity)
            stack.append((item, True))
            stack.extend((child, False) for child in reversed(item))
            continue
        if item is None or type(item) in {str, bool, int}:
            continue
        if type(item) is float and math.isfinite(item):
            continue
        raise WorkerStartV2ReadinessError(f"{label} is not plain finite JSON")


def _decode_artifact(
    artifact: object,
    *,
    label: str,
) -> tuple[WorkerStartV2ReadinessArtifact, dict[str, object]]:
    _require_exact_instance_state(
        artifact,
        exact_type=WorkerStartV2ReadinessArtifact,
        expected_fields=_ARTIFACT_INSTANCE_FIELDS,
        label=f"{label} artifact",
    )
    assert isinstance(artifact, WorkerStartV2ReadinessArtifact)
    if type(artifact.raw) is not bytes:
        raise WorkerStartV2ReadinessError(f"{label} artifact bytes are not immutable")
    _safe_key(artifact.key, label=f"{label} key")
    if not artifact.raw.endswith(b"\n") or artifact.raw.endswith(b"\n\n"):
        raise WorkerStartV2ReadinessError(
            f"{label} artifact is not one-LF canonical JSON"
        )
    try:
        value = json.loads(
            artifact.raw[:-1].decode("ascii"),
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_constant,
        )
    except WorkerStartV2ReadinessError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise WorkerStartV2ReadinessError(
            f"{label} artifact is not canonical ASCII JSON"
        ) from error
    if type(value) is not dict or not value:
        raise WorkerStartV2ReadinessError(f"{label} artifact must be a nonempty object")
    _plain_json(value, label=label)
    if _canonical(value) + b"\n" != artifact.raw:
        raise WorkerStartV2ReadinessError(
            f"{label} artifact bytes are not compact canonical JSON"
        )
    return artifact, value


def _sha_field(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise WorkerStartV2ReadinessError(f"{label} must be lowercase SHA-256")
    return value


def _run_id(value: object) -> str:
    if type(value) is not str or _RUN_ID.fullmatch(value) is None:
        raise WorkerStartV2ReadinessError("run_id is invalid")
    return value


def _safe_key(value: object, *, label: str, run_id: str = "") -> str:
    if (
        type(value) is not str
        or not value
        or value.startswith("/")
        or value.endswith("/")
        or "\\" in value
        or "*" in value
        or "?" in value
        or "[" in value
        or "]" in value
        or any(character.isspace() for character in value)
        or "//" in value
        or any(segment in {"", ".", ".."} for segment in value.split("/"))
        or (run_id and not value.startswith(f"campaigns/{run_id}/"))
    ):
        raise WorkerStartV2ReadinessError(
            f"{label} is not a safe exact campaign-scoped key"
        )
    return value


def _time(value: object, *, label: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise WorkerStartV2ReadinessError(f"{label} is not canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
        normalized = parsed.astimezone(timezone.utc)
        canonical = normalized.isoformat().replace("+00:00", "Z")
    except (ValueError, OverflowError) as error:
        raise WorkerStartV2ReadinessError(f"{label} is invalid") from error
    if parsed.utcoffset() != timedelta(0) or canonical != value:
        raise WorkerStartV2ReadinessError(f"{label} is not canonical UTC")
    return normalized


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise WorkerStartV2ReadinessError(f"{label} must be a positive integer")
    return value


def _nonnegative_float(value: object, *, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0:
        raise WorkerStartV2ReadinessError(f"{label} must be finite nonnegative money")
    return value


def _require_fields(
    value: Mapping[str, object],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        unknown = sorted(set(value) - expected)
        raise WorkerStartV2ReadinessError(
            f"{label} schema mismatch: missing={missing}, unknown={unknown}"
        )


def _validate_self_hash(
    value: dict[str, object],
    *,
    digest_field: str,
    label: str,
) -> None:
    digest = _sha_field(value.get(digest_field), label=f"{label} {digest_field}")
    body = dict(value)
    body.pop(digest_field)
    if digest != _body_sha(body):
        raise WorkerStartV2ReadinessError(f"{label} body SHA-256 mismatch")


def _validate_descriptor(
    artifact: object,
) -> tuple[WorkerStartV2ReadinessArtifact, dict[str, object], str]:
    exact, parsed = _decode_artifact(artifact, label="descriptor")
    try:
        descriptor = validate_sky_campaign_descriptor(parsed, verify_body_sha=True)
    except (TypeError, ValueError) as error:
        raise WorkerStartV2ReadinessError(
            "descriptor failed public validation"
        ) from error
    if (
        type(descriptor.get("instance_count")) is not int
        or descriptor["instance_count"] != 1
        or type(descriptor.get("use_spot")) is not bool
        or descriptor["use_spot"] is not False
    ):
        raise WorkerStartV2ReadinessError("descriptor market shape is not exact")
    run_id = _run_id(descriptor.get("run_id"))
    if descriptor.get("account_id") != _ACCOUNT_ID:
        raise WorkerStartV2ReadinessError("descriptor account is foreign")
    if descriptor.get("region") != _REGION:
        raise WorkerStartV2ReadinessError("descriptor region is foreign")
    bucket = descriptor.get("bucket")
    if type(bucket) is not str or _BUCKET.fullmatch(bucket) is None:
        raise WorkerStartV2ReadinessError("descriptor bucket is invalid")
    for field in (
        "campaign_descriptor_key",
        "repo_tar_key",
        "approval_key",
    ):
        _safe_key(descriptor.get(field), label=f"descriptor {field}", run_id=run_id)
    if exact.key != descriptor["campaign_descriptor_key"]:
        raise WorkerStartV2ReadinessError("descriptor artifact key mismatch")
    return exact, descriptor, _sha(exact.raw)


def _validate_readiness(
    artifact: object,
) -> tuple[WorkerStartV2ReadinessArtifact, dict[str, object], str]:
    exact, ready = _decode_artifact(artifact, label="qualification readiness")
    _require_fields(ready, _READINESS_FIELDS, label="qualification readiness")
    if (
        type(ready.get("schema_version")) is not int
        or ready.get("schema_version") != 1
        or ready.get("record_type") != "glm52_qualification_submission_ready_v1"
        or ready.get("account_id") != _ACCOUNT_ID
        or ready.get("region") != _REGION
        or ready.get("managed_mode") != _MANAGED_MODE
    ):
        raise WorkerStartV2ReadinessError(
            "qualification readiness schema or identity mismatch"
        )
    run_id = _run_id(ready.get("run_id"))
    for field in _READINESS_SHA_FIELDS:
        _sha_field(ready.get(field), label=f"qualification readiness {field}")
    for field in _READINESS_KEY_FIELDS:
        _safe_key(
            ready.get(field),
            label=f"qualification readiness {field}",
            run_id=run_id,
        )
    for field in (
        "remaining_gpu_seconds",
        "qualification_allowance_seconds",
        "allowed_gpu_seconds",
    ):
        _positive_int(ready.get(field), label=f"qualification readiness {field}")
    for field in (
        "remaining_gpu_cost_usd",
        "qualification_allowance_cost_usd",
        "allowed_gpu_cost_usd",
    ):
        _nonnegative_float(ready.get(field), label=f"qualification readiness {field}")
    if (
        ready.get("sky_task_name") != _SKY_TASK_NAME
        or ready.get("sky_job_name") != f"{run_id}-qualification"
        or ready.get("allowed_gpu_seconds")
        != ready.get("qualification_allowance_seconds")
        or ready.get("allowed_gpu_cost_usd")
        != ready.get("qualification_allowance_cost_usd")
    ):
        raise WorkerStartV2ReadinessError(
            "qualification readiness job or allowance mismatch"
        )
    built_at = _time(ready.get("built_at"), label="qualification readiness built_at")
    must_start_by = _time(
        ready.get("must_start_by"),
        label="qualification readiness must_start_by",
    )
    if built_at >= must_start_by:
        raise WorkerStartV2ReadinessError(
            "qualification readiness time ordering is invalid"
        )
    seed = ready.get("seed_descriptor")
    if type(seed) is not dict:
        raise WorkerStartV2ReadinessError(
            "qualification readiness seed_descriptor is invalid"
        )
    try:
        seed_descriptor = validate_sky_campaign_descriptor(seed, verify_body_sha=True)
    except (TypeError, ValueError) as error:
        raise WorkerStartV2ReadinessError(
            "qualification readiness seed descriptor failed validation"
        ) from error
    if (
        type(seed_descriptor.get("instance_count")) is not int
        or type(seed_descriptor.get("use_spot")) is not bool
    ):
        raise WorkerStartV2ReadinessError(
            "qualification readiness seed descriptor aliases boolean and integer"
        )
    seed_bindings = {
        "seed_descriptor_key": seed_descriptor["campaign_descriptor_key"],
        "seed_descriptor_file_sha256": _file_sha(seed_descriptor),
        "seed_descriptor_body_sha256": seed_descriptor["descriptor_body_sha256"],
        "seed_campaign_identity_sha256": seed_descriptor["campaign_identity_sha256"],
        "seed_repo_tar_sha256": seed_descriptor["repo_tar_sha256"],
    }
    for field, expected in seed_bindings.items():
        if ready.get(field) != expected:
            raise WorkerStartV2ReadinessError(
                f"qualification readiness {field} drifts from seed descriptor"
            )
    _validate_self_hash(
        ready,
        digest_field="readiness_body_sha256",
        label="qualification readiness",
    )
    try:
        expected_key = qualification_submission_ready_s3_key(ready)
    except (TypeError, ValueError) as error:
        raise WorkerStartV2ReadinessError(
            "qualification readiness key derivation failed"
        ) from error
    if exact.key != expected_key:
        raise WorkerStartV2ReadinessError(
            "qualification readiness artifact key mismatch"
        )
    return exact, ready, _sha(exact.raw)


def _validate_rehearsal(
    artifact: object,
) -> tuple[WorkerStartV2ReadinessArtifact, dict[str, object], str]:
    exact, rehearsal = _decode_artifact(artifact, label="rehearsal evidence")
    _require_fields(rehearsal, _REHEARSAL_FIELDS, label="rehearsal evidence")
    if (
        type(rehearsal.get("schema_version")) is not int
        or rehearsal.get("schema_version") != 2
        or rehearsal.get("record_type") != "glm52_staged_control_plane_rehearsal_v2"
        or rehearsal.get("status") != "passed_before_cuda_h100_boundary"
    ):
        raise WorkerStartV2ReadinessError(
            "rehearsal evidence schema or status mismatch"
        )
    run_id = _run_id(rehearsal.get("run_id"))
    for field in _REHEARSAL_SHA_FIELDS:
        _sha_field(rehearsal.get(field), label=f"rehearsal evidence {field}")
    for field in _REHEARSAL_KEY_FIELDS:
        _safe_key(
            rehearsal.get(field),
            label=f"rehearsal evidence {field}",
            run_id=run_id,
        )
    extracted = rehearsal.get("extracted_repo_path")
    if (
        type(extracted) is not str
        or not extracted
        or "\x00" in extracted
        or not PurePosixPath(extracted).is_absolute()
        or ".." in PurePosixPath(extracted).parts
    ):
        raise WorkerStartV2ReadinessError(
            "rehearsal extracted repository path is unsafe"
        )
    if (
        rehearsal.get("production_repo_path") != "/opt/keep-campaign/repo"
        or rehearsal.get("production_resume_root") != "/mnt/nvme/glm52-campaign"
        or rehearsal.get("skypilot_task_path")
        != "/opt/keep-campaign/repo/aws/glm52-gpu/skypilot/glm52-campaign.yaml"
        or rehearsal.get("skypilot_version") != _SKYPILOT_VERSION
        or rehearsal.get("skypilot_task_name") != _SKY_TASK_NAME
    ):
        raise WorkerStartV2ReadinessError(
            "rehearsal production or SkyPilot identity mismatch"
        )
    _time(rehearsal.get("completed_at"), label="rehearsal completed_at")
    _validate_self_hash(
        rehearsal,
        digest_field="rehearsal_body_sha256",
        label="rehearsal evidence",
    )
    try:
        expected_key = rehearsal_evidence_s3_key(
            run_id=run_id,
            rehearsal_body_sha256=str(rehearsal["rehearsal_body_sha256"]),
        )
    except (TypeError, ValueError) as error:
        raise WorkerStartV2ReadinessError(
            "rehearsal evidence key derivation failed"
        ) from error
    if exact.key != expected_key:
        raise WorkerStartV2ReadinessError("rehearsal evidence artifact key mismatch")
    return exact, rehearsal, _sha(exact.raw)


def _validate_intent(
    artifact: object,
) -> tuple[
    WorkerStartV2ReadinessArtifact,
    dict[str, object],
    str,
    datetime,
    datetime,
]:
    exact, intent = _decode_artifact(artifact, label="submission intent")
    try:
        dynamic_v2_job_binding_s3_key(intent=intent)
    except (TypeError, ValueError) as error:
        raise WorkerStartV2ReadinessError(
            "submission intent failed public dynamic-v2 validation"
        ) from error
    run_id = _run_id(intent.get("run_id"))
    expected_key = (
        f"campaigns/{run_id}/submissions/qualification/intents/"
        f"{intent['intent_body_sha256']}/SKYPILOT_SUBMISSION_INTENT.json"
    )
    if exact.key != expected_key:
        raise WorkerStartV2ReadinessError("submission intent artifact key mismatch")
    intent_at = _time(intent.get("intent_at"), label="submission intent intent_at")
    deadline = _time(
        intent.get("must_start_by"), label="submission intent must_start_by"
    )
    if intent_at >= deadline:
        raise WorkerStartV2ReadinessError("submission intent deadline is invalid")
    return exact, intent, _sha(exact.raw), intent_at, deadline


def _require_exact(
    left: object,
    right: object,
    *,
    label: str,
) -> None:
    if type(left) is not type(right) or left != right:
        raise WorkerStartV2ReadinessError(f"{label} authority drift")


def _validate_cross_bindings(
    *,
    descriptor: dict[str, object],
    descriptor_file_sha256: str,
    readiness_artifact: WorkerStartV2ReadinessArtifact,
    readiness: dict[str, object],
    readiness_file_sha256: str,
    rehearsal_artifact: WorkerStartV2ReadinessArtifact,
    rehearsal: dict[str, object],
    rehearsal_file_sha256: str,
    intent_artifact: WorkerStartV2ReadinessArtifact,
    intent: dict[str, object],
    intent_at: datetime,
    intent_deadline: datetime,
) -> None:
    for descriptor_field, ready_field in _DESCRIPTOR_READINESS_BINDINGS:
        _require_exact(
            descriptor[descriptor_field],
            readiness[ready_field],
            label=f"descriptor/readiness {ready_field}",
        )
    _require_exact(
        descriptor_file_sha256,
        readiness["descriptor_file_sha256"],
        label="descriptor/readiness descriptor_file_sha256",
    )
    _require_exact(
        readiness["sky_job_name"],
        f"{descriptor['run_id']}-qualification",
        label="descriptor/readiness sky_job_name",
    )
    _require_exact(
        readiness["managed_mode"],
        _MANAGED_MODE,
        label="descriptor/readiness managed_mode",
    )

    for descriptor_field, rehearsal_field in _DESCRIPTOR_REHEARSAL_BINDINGS:
        _require_exact(
            descriptor[descriptor_field],
            rehearsal[rehearsal_field],
            label=f"descriptor/rehearsal {rehearsal_field}",
        )
    _require_exact(
        descriptor_file_sha256,
        rehearsal["descriptor_file_sha256"],
        label="descriptor/rehearsal descriptor_file_sha256",
    )
    artifacts = descriptor.get("artifacts")
    if type(artifacts) is not dict:
        raise WorkerStartV2ReadinessError("descriptor artifacts are invalid")
    _require_exact(
        artifacts["artifact_inventory_key"],
        rehearsal["artifact_inventory_key"],
        label="descriptor/rehearsal artifact_inventory_key",
    )
    _require_exact(
        artifacts["artifact_inventory_sha256"],
        rehearsal["artifact_inventory_file_sha256"],
        label="descriptor/rehearsal artifact_inventory_file_sha256",
    )

    readiness_rehearsal = {
        "rehearsal_evidence_key": rehearsal_artifact.key,
        "rehearsal_evidence_sha256": rehearsal_file_sha256,
        "rehearsal_evidence_body_sha256": rehearsal["rehearsal_body_sha256"],
        "staged_readiness_key": rehearsal["staged_readiness_key"],
        "staged_readiness_file_sha256": rehearsal["staged_readiness_file_sha256"],
        "staged_readiness_body_sha256": rehearsal["staged_readiness_body_sha256"],
    }
    for field, expected in readiness_rehearsal.items():
        _require_exact(
            readiness[field],
            expected,
            label=f"readiness/rehearsal {field}",
        )

    for field in _INTENT_READINESS_FIELDS:
        _require_exact(
            intent[field],
            readiness[field],
            label=f"intent/readiness {field}",
        )
    intent_readiness = {
        "qualification_submission_ready_key": readiness_artifact.key,
        "qualification_submission_ready_sha256": readiness_file_sha256,
        "qualification_submission_ready_body_sha256": readiness[
            "readiness_body_sha256"
        ],
    }
    for field, expected in intent_readiness.items():
        _require_exact(intent[field], expected, label=f"intent/readiness {field}")
    if intent.get("open_allocation_count") != 0:
        raise WorkerStartV2ReadinessError(
            "submission intent has an open GPU allocation"
        )
    ready_at = _time(
        readiness.get("built_at"), label="qualification readiness built_at"
    )
    ready_deadline = _time(
        readiness.get("must_start_by"),
        label="qualification readiness must_start_by",
    )
    if not ready_at <= intent_at < intent_deadline or ready_deadline != intent_deadline:
        raise WorkerStartV2ReadinessError(
            "readiness, intent, and must-start time ordering drifted"
        )
    _require_exact(
        intent_artifact.key,
        (
            f"campaigns/{intent['run_id']}/submissions/qualification/intents/"
            f"{intent['intent_body_sha256']}/SKYPILOT_SUBMISSION_INTENT.json"
        ),
        label="intent content-addressed key",
    )


def _source_text(raw: object, *, label: str) -> str:
    if type(raw) is not bytes:
        raise WorkerStartV2ReadinessError(
            f"{label} source is not immutable exact bytes"
        )
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise WorkerStartV2ReadinessError(f"{label} source is not UTF-8") from error


def _require_tokens(
    text: str,
    tokens: tuple[str, ...],
    *,
    label: str,
    exact_count: int = 1,
) -> None:
    for token in tokens:
        count = text.count(token)
        if (exact_count < 0 and count < 1) or (
            exact_count >= 0 and count != exact_count
        ):
            raise WorkerStartV2ReadinessError(
                f"{label} source contract token count drift: {token}"
            )


def _require_order(text: str, tokens: tuple[str, ...], *, label: str) -> None:
    cursor = -1
    for token in tokens:
        location = text.find(token, cursor + 1)
        if location < 0 or location <= cursor:
            raise WorkerStartV2ReadinessError(
                f"{label} source contract ordering drift: {token}"
            )
        cursor = location


def _validate_source_contract(snapshot: WorkerStartV2SourceSnapshot) -> str:
    texts = {
        field: _source_text(getattr(snapshot, field), label=field)
        for field in _SOURCE_SNAPSHOT_INSTANCE_FIELDS
    }
    for field, tokens in _PYTHON_REQUIRED_TOKENS.items():
        _require_tokens(texts[field], tokens, label=field)
    _require_tokens(texts["sky_task_raw"], _TASK_MOUNTS, label="sky task")
    _require_tokens(
        texts["sky_task_raw"],
        _TASK_HASH_ARGUMENTS,
        label="sky task",
        exact_count=2,
    )
    for field, tokens in _ORDERED_LANDMARKS.items():
        _require_order(texts[field], tokens, label=field)
    for field, tokens in _EXACT_SOURCE_CONSTANTS.items():
        _require_tokens(texts[field], tokens, label=field, exact_count=-1)
    for field in ("sky_task_raw", "bootstrap_raw", "managed_entrypoint_raw"):
        for token in _FORBIDDEN_TOKENS:
            if token in texts[field]:
                raise WorkerStartV2ReadinessError(
                    f"{field} source contract forbids legacy token: {token}"
                )
    return _body_sha(_SOURCE_CONTRACT)


def _validate_sources(
    source_snapshot: object,
    *,
    rehearsal: dict[str, object],
) -> tuple[list[dict[str, str]], str, str]:
    _require_exact_instance_state(
        source_snapshot,
        exact_type=WorkerStartV2SourceSnapshot,
        expected_fields=frozenset(_SOURCE_SNAPSHOT_INSTANCE_FIELDS),
        label="source snapshot",
    )
    assert isinstance(source_snapshot, WorkerStartV2SourceSnapshot)
    for field in _SOURCE_SNAPSHOT_INSTANCE_FIELDS:
        if type(getattr(source_snapshot, field)) is not bytes:
            raise WorkerStartV2ReadinessError(
                f"source snapshot {field} is not immutable exact bytes"
            )
    source_contract_sha256 = _validate_source_contract(source_snapshot)
    source_files: list[dict[str, str]] = []
    for relative_path, field, expected_sha256 in _SOURCE_MEMBERS:
        actual = _sha(getattr(source_snapshot, field))
        if actual != expected_sha256:
            raise WorkerStartV2ReadinessError(
                f"frozen source hash drift: {relative_path}"
            )
        source_files.append(
            {
                "relative_path": relative_path,
                "file_sha256": actual,
            }
        )
    if source_files != sorted(
        source_files, key=lambda item: str(item["relative_path"])
    ):
        raise WorkerStartV2ReadinessError("frozen source closure is reordered")
    direct_source_files_sha256 = _sha(_canonical(source_files))
    if direct_source_files_sha256 != _SOURCE_CLOSURE_SHA256:
        raise WorkerStartV2ReadinessError(
            "frozen source-files canonical SHA-256 drifted"
        )
    if _sha(source_snapshot.sky_task_raw) != rehearsal["skypilot_task_file_sha256"]:
        raise WorkerStartV2ReadinessError(
            "SkyPilot task source SHA-256 differs from rehearsal"
        )
    return source_files, source_contract_sha256, direct_source_files_sha256


def _derive(
    *,
    authorities: object,
    source_snapshot: object,
) -> dict[str, object]:
    _require_exact_instance_state(
        authorities,
        exact_type=WorkerStartV2ReadinessAuthorities,
        expected_fields=_AUTHORITIES_INSTANCE_FIELDS,
        label="authorities",
    )
    assert isinstance(authorities, WorkerStartV2ReadinessAuthorities)
    descriptor_artifact, descriptor, descriptor_file_sha256 = _validate_descriptor(
        authorities.descriptor
    )
    readiness_artifact, readiness, readiness_file_sha256 = _validate_readiness(
        authorities.qualification_submission_ready
    )
    rehearsal_artifact, rehearsal, rehearsal_file_sha256 = _validate_rehearsal(
        authorities.rehearsal_evidence
    )
    intent_artifact, intent, intent_file_sha256, intent_at, intent_deadline = (
        _validate_intent(authorities.intent)
    )
    _validate_cross_bindings(
        descriptor=descriptor,
        descriptor_file_sha256=descriptor_file_sha256,
        readiness_artifact=readiness_artifact,
        readiness=readiness,
        readiness_file_sha256=readiness_file_sha256,
        rehearsal_artifact=rehearsal_artifact,
        rehearsal=rehearsal,
        rehearsal_file_sha256=rehearsal_file_sha256,
        intent_artifact=intent_artifact,
        intent=intent,
        intent_at=intent_at,
        intent_deadline=intent_deadline,
    )
    source_files, source_contract_sha256, source_closure_sha256 = _validate_sources(
        source_snapshot, rehearsal=rehearsal
    )
    body: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "record_type": _RECORD_TYPE,
        "account_id": descriptor["account_id"],
        "region": descriptor["region"],
        "bucket": descriptor["bucket"],
        "run_id": descriptor["run_id"],
        "managed_mode": _MANAGED_MODE,
        "descriptor_key": descriptor_artifact.key,
        "descriptor_file_sha256": descriptor_file_sha256,
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
        "repo_tar_key": descriptor["repo_tar_key"],
        "repo_tar_sha256": descriptor["repo_tar_sha256"],
        "qualification_submission_ready_key": readiness_artifact.key,
        "qualification_submission_ready_file_sha256": readiness_file_sha256,
        "qualification_submission_ready_body_sha256": readiness[
            "readiness_body_sha256"
        ],
        "rehearsal_evidence_key": rehearsal_artifact.key,
        "rehearsal_evidence_file_sha256": rehearsal_file_sha256,
        "rehearsal_evidence_body_sha256": rehearsal["rehearsal_body_sha256"],
        "intent_key": intent_artifact.key,
        "intent_file_sha256": intent_file_sha256,
        "intent_body_sha256": intent["intent_body_sha256"],
        "skypilot_version": descriptor["skypilot_version"],
        "sky_task_name": descriptor["task_name"],
        "sky_task_file_sha256": rehearsal["skypilot_task_file_sha256"],
        "sky_job_name": intent["sky_job_name"],
        "must_start_by": descriptor["must_start_by"],
        "instance_type": descriptor["instance_type"],
        "instance_count": descriptor["instance_count"],
        "use_spot": descriptor["use_spot"],
        "image_id": descriptor["image_id"],
        "controller_identity": descriptor["controller_identity"],
        "worker_identity": descriptor["worker_identity"],
        "source_contract_sha256": source_contract_sha256,
        "source_closure_sha256": source_closure_sha256,
        "source_files": source_files,
    }
    return {**body, _DIGEST_FIELD: _body_sha(body)}


def _validate_record_shape(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise WorkerStartV2ReadinessError("source-readiness record must be an object")
    _plain_json(value, label="source-readiness record")
    _require_fields(value, _RECORD_FIELDS, label="source-readiness record")
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != _SCHEMA_VERSION
        or value.get("record_type") != _RECORD_TYPE
        or value.get("account_id") != _ACCOUNT_ID
        or value.get("region") != _REGION
        or value.get("managed_mode") != _MANAGED_MODE
        or value.get("skypilot_version") != _SKYPILOT_VERSION
        or value.get("sky_task_name") != _SKY_TASK_NAME
        or value.get("instance_type") != _INSTANCE_TYPE
        or type(value.get("instance_count")) is not int
        or value.get("instance_count") != 1
        or type(value.get("use_spot")) is not bool
        or value.get("use_spot") is not False
    ):
        raise WorkerStartV2ReadinessError(
            "source-readiness schema or immutable identity mismatch"
        )
    run_id = _run_id(value.get("run_id"))
    bucket = value.get("bucket")
    if type(bucket) is not str or _BUCKET.fullmatch(bucket) is None:
        raise WorkerStartV2ReadinessError("source-readiness bucket is invalid")
    for field in (
        "descriptor_key",
        "repo_tar_key",
        "qualification_submission_ready_key",
        "rehearsal_evidence_key",
        "intent_key",
    ):
        _safe_key(value.get(field), label=field, run_id=run_id)
    for field, item in value.items():
        if field.endswith("_sha256"):
            _sha_field(item, label=field)
    _time(value.get("must_start_by"), label="must_start_by")
    if (
        type(value.get("sky_job_name")) is not str
        or value["sky_job_name"] != f"{run_id}-qualification"
    ):
        raise WorkerStartV2ReadinessError("source-readiness Sky job name is not exact")
    image_id = value.get("image_id")
    if type(image_id) is not str or _AMI_ID.fullmatch(image_id) is None:
        raise WorkerStartV2ReadinessError("source-readiness image_id is invalid")
    for field in ("controller_identity", "worker_identity"):
        identity = value.get(field)
        if (
            type(identity) is not str
            or _IAM_ROLE_ARN.fullmatch(identity) is None
            or any(
                component in {".", ".."}
                for component in identity.split(":role/", 1)[1].split("/")
            )
        ):
            raise WorkerStartV2ReadinessError(
                f"source-readiness {field} is not an approved-account IAM role"
            )
    descriptor_key = str(value["descriptor_key"])
    if not descriptor_key.startswith(
        f"campaigns/{run_id}/submissions/"
    ) or not descriptor_key.endswith("/campaign-descriptor-v2.json"):
        raise WorkerStartV2ReadinessError(
            "source-readiness descriptor key is not submission-scoped"
        )
    expected_repo_tar_key = (
        f"campaigns/{run_id}/repository/keep-{value['repo_tar_sha256']}.tar.gz"
    )
    if value["repo_tar_key"] != expected_repo_tar_key:
        raise WorkerStartV2ReadinessError(
            "source-readiness repository key is not exact"
        )
    expected_content_addressed_keys = {
        "qualification_submission_ready_key": (
            f"campaigns/{run_id}/qualification/submission-ready/"
            f"{value['qualification_submission_ready_body_sha256']}/"
            "QUALIFICATION_SUBMISSION_READY.json"
        ),
        "rehearsal_evidence_key": (
            f"campaigns/{run_id}/qualification/rehearsals/"
            f"{value['rehearsal_evidence_body_sha256']}/"
            "GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
        ),
        "intent_key": (
            f"campaigns/{run_id}/submissions/qualification/intents/"
            f"{value['intent_body_sha256']}/SKYPILOT_SUBMISSION_INTENT.json"
        ),
    }
    for field, expected in expected_content_addressed_keys.items():
        if value[field] != expected:
            raise WorkerStartV2ReadinessError(
                f"source-readiness {field} is not content-addressed"
            )
    for file_field, body_field in (
        ("descriptor_file_sha256", "descriptor_body_sha256"),
        (
            "qualification_submission_ready_file_sha256",
            "qualification_submission_ready_body_sha256",
        ),
        ("rehearsal_evidence_file_sha256", "rehearsal_evidence_body_sha256"),
        ("intent_file_sha256", "intent_body_sha256"),
    ):
        if value[file_field] == value[body_field]:
            raise WorkerStartV2ReadinessError(
                f"source-readiness {file_field} aliases {body_field}"
            )
    source_files = value.get("source_files")
    if type(source_files) is not list or source_files != sorted(
        source_files,
        key=lambda item: str(item.get("relative_path")) if type(item) is dict else "",
    ):
        raise WorkerStartV2ReadinessError(
            "source-readiness source_files are not a sorted array"
        )
    seen: set[str] = set()
    for item in source_files:
        if type(item) is not dict or set(item) != {"relative_path", "file_sha256"}:
            raise WorkerStartV2ReadinessError("source file entry schema mismatch")
        path = item.get("relative_path")
        if (
            type(path) is not str
            or not path
            or path.startswith("/")
            or "\\" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))
            or path in seen
        ):
            raise WorkerStartV2ReadinessError("source file path is unsafe or duplicate")
        seen.add(path)
        _sha_field(item.get("file_sha256"), label=f"source file {path}")
    expected_source_files = [
        {
            "relative_path": relative_path,
            "file_sha256": expected_sha256,
        }
        for relative_path, _field, expected_sha256 in _SOURCE_MEMBERS
    ]
    if source_files != expected_source_files:
        raise WorkerStartV2ReadinessError(
            "source-readiness source_files differ from the frozen closure"
        )
    task_entry = next(
        item
        for item in expected_source_files
        if item["relative_path"] == "aws/glm52-gpu/skypilot/glm52-campaign.yaml"
    )
    if value["sky_task_file_sha256"] != task_entry["file_sha256"]:
        raise WorkerStartV2ReadinessError(
            "source-readiness Sky task SHA-256 differs from frozen source"
        )
    if value["source_contract_sha256"] != _body_sha(_SOURCE_CONTRACT):
        raise WorkerStartV2ReadinessError(
            "source-readiness source contract SHA-256 mismatch"
        )
    if (
        value["source_closure_sha256"] != _SOURCE_CLOSURE_SHA256
        or _sha(_canonical(source_files)) != value["source_closure_sha256"]
    ):
        raise WorkerStartV2ReadinessError("source closure SHA-256 mismatch")
    _validate_self_hash(
        value,
        digest_field=_DIGEST_FIELD,
        label="source-readiness record",
    )
    return dict(value)


def build_worker_start_v2_readiness(
    *,
    authorities: WorkerStartV2ReadinessAuthorities,
    source_snapshot: WorkerStartV2SourceSnapshot,
) -> dict[str, object]:
    """Build one deterministic local worker-start-v2 source-readiness record."""

    return _validate_record_shape(
        _derive(authorities=authorities, source_snapshot=source_snapshot)
    )


def validate_worker_start_v2_readiness(
    value: object,
    *,
    authorities: WorkerStartV2ReadinessAuthorities,
    source_snapshot: WorkerStartV2SourceSnapshot,
) -> dict[str, object]:
    """Authenticate a record against exact supplied authorities and source bytes."""

    record = _validate_record_shape(value)
    expected = _derive(authorities=authorities, source_snapshot=source_snapshot)
    if record != expected:
        raise WorkerStartV2ReadinessError(
            "source-readiness record differs from exact derived authority"
        )
    return record


def worker_start_v2_readiness_file_bytes(value: object) -> bytes:
    """Return compact, sorted, finite ASCII JSON plus exactly one LF."""

    record = _validate_record_shape(value)
    return _canonical(record) + b"\n"


def worker_start_v2_readiness_file_sha256(value: object) -> str:
    """Return the SHA-256 of the exact canonical one-LF file transport."""

    return _sha(worker_start_v2_readiness_file_bytes(value))


def worker_start_v2_readiness_s3_key(value: object) -> str:
    """Return the future content-addressed key without publishing an object."""

    record = _validate_record_shape(value)
    return (
        f"campaigns/{record['run_id']}/submissions/qualification/intents/"
        f"{record['intent_body_sha256']}/worker-start-v2-readiness/"
        f"{record[_DIGEST_FIELD]}/WORKER_START_V2_READINESS.json"
    )


__all__ = [
    "WorkerStartV2ReadinessArtifact",
    "WorkerStartV2ReadinessAuthorities",
    "WorkerStartV2ReadinessError",
    "WorkerStartV2SourceSnapshot",
    "build_worker_start_v2_readiness",
    "validate_worker_start_v2_readiness",
    "worker_start_v2_readiness_file_bytes",
    "worker_start_v2_readiness_file_sha256",
    "worker_start_v2_readiness_s3_key",
]
