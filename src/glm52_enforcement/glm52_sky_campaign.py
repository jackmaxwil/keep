"""Enforcement-native SkyPilot campaign authority and GPU spend ledger."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from .glm52_teich_campaign import CampaignPhase


APPROVED_ACCOUNT_ID = "246813579024"
APPROVED_REGION = "us-west-2"
APPROVED_INSTANCE_TYPE = "p5.48xlarge"
APPROVED_HOURLY_COST_USD = 55.04
APPROVED_GPU_HOURS = 24
APPROVED_GPU_RUNTIME_SECONDS = APPROVED_GPU_HOURS * 60 * 60
APPROVED_GPU_COST_USD = 1_320.96
PINNED_SKYPILOT_VERSION = "0.13.0"

GPU_APPROVAL_REQUEST = """Hey Kon — following your SkyPilot suggestion, I’m planning to run the GLM-5.2 teacher-cache and adapter-training campaign as an AWS-only SkyPilot Managed Job instead of buying another Capacity Block.

Current verified `p5.48xlarge` on-demand pricing in Oregon is **$55.04/hour**:
- 24-hour hard cap: **$1,320.96**
- 36-hour hard cap: **$1,981.44**
- 48-hour hard cap: **$2,641.92**

My recommendation is the **24-hour cap**. The campaign will checkpoint to S3 and stop cleanly at the cap; if it is incomplete, we can resume under a separately approved budget extension. There will be no paid GPU reservation while AWS has no capacity, though the small SkyPilot controller and S3 charges can continue while queued. Spot will remain disabled until interruption/resume has passed on a real H100.

Are you okay with me proceeding with the 24-hour / $1,320.96 GPU cap, or would you approve a different cap?"""

_HEX64 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_AMI_ID = re.compile(r"ami-[0-9a-f]{8,17}")
_IAM_ROLE_ARN = re.compile(r"arn:aws:iam::([0-9]{12}):role/[A-Za-z0-9+=,.@_/-]+")


class SkyCampaignValidationError(ValueError):
    """Raised when SkyPilot campaign authority is incomplete or inconsistent."""


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _iso_utc(value: datetime | str, *, field: str) -> str:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise SkyCampaignValidationError(f"{field} must be ISO-8601") from error
    else:
        raise SkyCampaignValidationError(f"{field} must be ISO-8601")
    if parsed.tzinfo is None:
        raise SkyCampaignValidationError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parsed_time(value: object, *, field: str) -> datetime:
    return datetime.fromisoformat(_iso_utc(value, field=field).replace("Z", "+00:00"))


_APPROVAL_BODY_FIELDS = {
    "schema_version",
    "record_type",
    "approved_by",
    "approval_request",
    "approval_response",
    "approval_displayed_time",
    "approval_ingested_at",
    "evidence_source",
    "slack_permalink",
    "approved_hourly_usd",
    "approved_gpu_hours",
    "approved_gpu_cost_usd",
    "instance_type",
    "region",
    "includes_qualification",
    "includes_recovery_instances",
}
_APPROVAL_FIELDS = _APPROVAL_BODY_FIELDS | {"approval_body_sha256"}


def build_gpu_spend_approval(
    *,
    ingested_at: datetime,
    slack_permalink: str | None,
) -> dict[str, object]:
    """Build the exact immutable authority for Kon's approved spend envelope."""

    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_gpu_spend_approval_v1",
        "approved_by": "Alex Approver",
        "approval_request": GPU_APPROVAL_REQUEST,
        "approval_response": "yep",
        "approval_displayed_time": "5:59 PM",
        "approval_ingested_at": _iso_utc(
            ingested_at, field="approval_ingested_at"
        ),
        "evidence_source": "user-supplied-slack-exchange",
        "slack_permalink": slack_permalink,
        "approved_hourly_usd": APPROVED_HOURLY_COST_USD,
        "approved_gpu_hours": APPROVED_GPU_HOURS,
        "approved_gpu_cost_usd": APPROVED_GPU_COST_USD,
        "instance_type": APPROVED_INSTANCE_TYPE,
        "region": APPROVED_REGION,
        "includes_qualification": True,
        "includes_recovery_instances": True,
    }
    return {**body, "approval_body_sha256": _sha256(body)}


def validate_gpu_spend_approval(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Authenticate the exact approval text and cost envelope."""

    if set(value) != _APPROVAL_FIELDS:
        raise SkyCampaignValidationError("GPU spend approval schema mismatch")
    body = dict(value)
    digest = body.pop("approval_body_sha256")
    if digest != _sha256(body):
        raise SkyCampaignValidationError("GPU spend approval body SHA-256 mismatch")
    expected: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_gpu_spend_approval_v1",
        "approved_by": "Alex Approver",
        "approval_request": GPU_APPROVAL_REQUEST,
        "approval_response": "yep",
        "approval_displayed_time": "5:59 PM",
        "evidence_source": "user-supplied-slack-exchange",
        "approved_hourly_usd": APPROVED_HOURLY_COST_USD,
        "approved_gpu_hours": APPROVED_GPU_HOURS,
        "approved_gpu_cost_usd": APPROVED_GPU_COST_USD,
        "instance_type": APPROVED_INSTANCE_TYPE,
        "region": APPROVED_REGION,
        "includes_qualification": True,
        "includes_recovery_instances": True,
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise SkyCampaignValidationError(
                f"GPU spend approval {field} does not match approved authority"
            )
    _iso_utc(value["approval_ingested_at"], field="approval_ingested_at")
    permalink = value.get("slack_permalink")
    if permalink is not None and (
        not isinstance(permalink, str)
        or not permalink.startswith("https://")
        or "slack.com/" not in permalink
    ):
        raise SkyCampaignValidationError("GPU spend approval Slack permalink is invalid")
    return dict(value)


_ARTIFACT_FIELDS = {
    "source_snapshot_prefix",
    "source_snapshot_sha256",
    "non_vq_prefix",
    "non_vq_package_sha256",
    "teich_pack_key",
    "teich_pack_sha256",
    "frozen_prompt_pack_key",
    "frozen_prompt_pack_sha256",
    "training_baseline_prefix",
    "training_baseline_sha256",
    "training_config_key",
    "training_config_sha256",
    "artifact_inventory_key",
    "artifact_inventory_sha256",
    "qualification_cache_prefix",
    "qualification_cache_manifest_sha256",
}
_ARTIFACT_SHA_FIELDS = {
    "source_snapshot_sha256",
    "non_vq_package_sha256",
    "teich_pack_sha256",
    "frozen_prompt_pack_sha256",
    "training_baseline_sha256",
    "training_config_sha256",
    "artifact_inventory_sha256",
    "qualification_cache_manifest_sha256",
}
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
_DESCRIPTOR_FIELDS = _DESCRIPTOR_BODY_FIELDS | {"descriptor_body_sha256"}


def _campaign_identity_sha256(
    descriptor_body: Mapping[str, object],
) -> str:
    """Hash campaign science/infrastructure authority, not submission timing."""

    identity = {
        name: value
        for name, value in descriptor_body.items()
        if name
        not in {
            "campaign_identity_sha256",
            "descriptor_body_sha256",
            "must_start_by",
            "campaign_descriptor_key",
        }
    }
    return _sha256(identity)


def build_sky_campaign_descriptor(
    *,
    run_id: str,
    must_start_by: datetime,
    controller_identity: str,
    worker_identity: str,
    vpc_name: str,
    image_id: str,
    bucket: str,
    jobs_bucket: str,
    repo_tar_key: str,
    repo_tar_sha256: str,
    campaign_descriptor_key: str,
    approval_key: str,
    approval_sha256: str,
    artifacts: Mapping[str, str],
) -> dict[str, object]:
    """Build the only descriptor accepted by the SkyPilot campaign path."""

    body: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_sky_campaign_descriptor_v2",
        "run_id": run_id,
        "account_id": APPROVED_ACCOUNT_ID,
        "provider": "aws",
        "region": APPROVED_REGION,
        "instance_type": APPROVED_INSTANCE_TYPE,
        "instance_count": 1,
        "use_spot": False,
        "max_hourly_cost_usd": APPROVED_HOURLY_COST_USD,
        "approved_gpu_runtime_seconds": APPROVED_GPU_RUNTIME_SECONDS,
        "approved_gpu_cost_usd": APPROVED_GPU_COST_USD,
        "must_start_by": _iso_utc(must_start_by, field="must_start_by"),
        "skypilot_version": PINNED_SKYPILOT_VERSION,
        "task_name": "glm52-campaign",
        "controller_identity": controller_identity,
        "worker_identity": worker_identity,
        "vpc_name": vpc_name,
        "image_id": image_id,
        "bucket": bucket,
        "jobs_bucket": jobs_bucket,
        "repo_tar_key": repo_tar_key,
        "repo_tar_sha256": repo_tar_sha256,
        "campaign_descriptor_key": campaign_descriptor_key,
        "approval_key": approval_key,
        "approval_sha256": approval_sha256,
        "artifacts": dict(artifacts),
    }
    body["campaign_identity_sha256"] = _campaign_identity_sha256(body)
    descriptor = {**body, "descriptor_body_sha256": _sha256(body)}
    return validate_sky_campaign_descriptor(descriptor)


def validate_sky_campaign_descriptor(
    value: Mapping[str, object],
    *,
    verify_body_sha: bool = True,
) -> dict[str, object]:
    """Validate an AWS-only, on-demand, fixed-budget SkyPilot campaign."""

    expected_fields = _DESCRIPTOR_FIELDS if verify_body_sha else _DESCRIPTOR_BODY_FIELDS
    if set(value) != expected_fields:
        raise SkyCampaignValidationError("SkyPilot campaign descriptor schema mismatch")
    body = dict(value)
    if verify_body_sha:
        digest = body.pop("descriptor_body_sha256")
        if digest != _sha256(body):
            raise SkyCampaignValidationError(
                "SkyPilot campaign descriptor body SHA-256 mismatch"
            )
    if value.get("schema_version") != 2 or value.get(
        "record_type"
    ) != "glm52_sky_campaign_descriptor_v2":
        raise SkyCampaignValidationError("SkyPilot campaign descriptor schema mismatch")
    if value.get("account_id") != APPROVED_ACCOUNT_ID:
        raise SkyCampaignValidationError(
            "SkyPilot campaign must use the approved AWS account"
        )
    if value.get("provider") != "aws":
        raise SkyCampaignValidationError("SkyPilot campaign provider must be aws")
    if value.get("region") != APPROVED_REGION:
        raise SkyCampaignValidationError("SkyPilot campaign region must be us-west-2")
    if value.get("instance_type") != APPROVED_INSTANCE_TYPE or value.get(
        "instance_count"
    ) != 1:
        raise SkyCampaignValidationError(
            "SkyPilot campaign permits one p5.48xlarge instance"
        )
    if value.get("use_spot") is not False:
        raise SkyCampaignValidationError("SkyPilot campaign must remain on-demand")
    if value.get("max_hourly_cost_usd") != APPROVED_HOURLY_COST_USD:
        raise SkyCampaignValidationError(
            "SkyPilot campaign hourly price exceeds approved authority"
        )
    if (
        value.get("approved_gpu_runtime_seconds") != APPROVED_GPU_RUNTIME_SECONDS
        or value.get("approved_gpu_cost_usd") != APPROVED_GPU_COST_USD
    ):
        raise SkyCampaignValidationError(
            "SkyPilot campaign GPU budget does not match approved authority"
        )
    if value.get("skypilot_version") != PINNED_SKYPILOT_VERSION:
        raise SkyCampaignValidationError("SkyPilot version is not pinned to 0.13.0")
    if value.get("task_name") != "glm52-campaign":
        raise SkyCampaignValidationError("SkyPilot task name is invalid")
    run_id = value.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise SkyCampaignValidationError("SkyPilot run_id is invalid")
    _iso_utc(value["must_start_by"], field="must_start_by")
    for field in ("controller_identity", "worker_identity"):
        identity = value.get(field)
        match = _IAM_ROLE_ARN.fullmatch(str(identity))
        if match is None or match.group(1) != APPROVED_ACCOUNT_ID:
            raise SkyCampaignValidationError(
                f"SkyPilot {field} must be an approved-account IAM role ARN"
            )
    if _AMI_ID.fullmatch(str(value.get("image_id"))) is None:
        raise SkyCampaignValidationError("SkyPilot image_id is invalid")
    for field in (
        "vpc_name",
        "bucket",
        "jobs_bucket",
        "repo_tar_key",
        "campaign_descriptor_key",
        "approval_key",
    ):
        if not isinstance(value.get(field), str) or not value[field]:
            raise SkyCampaignValidationError(
                f"SkyPilot campaign descriptor {field} is invalid"
            )
    campaign_prefix = f"campaigns/{run_id}/"
    if not str(value["repo_tar_key"]).startswith(
        campaign_prefix + "repository/"
    ):
        raise SkyCampaignValidationError(
            "SkyPilot repository tar must be campaign-scoped"
        )
    if not str(value["campaign_descriptor_key"]).startswith(
        campaign_prefix + "submissions/"
    ):
        raise SkyCampaignValidationError(
            "SkyPilot descriptor key must be submission-scoped"
        )
    if not str(value["approval_key"]).startswith(
        campaign_prefix + "authorities/"
    ):
        raise SkyCampaignValidationError(
            "SkyPilot approval key must be campaign-scoped"
        )
    if value["jobs_bucket"] != value["bucket"]:
        raise SkyCampaignValidationError(
            "SkyPilot jobs bucket must match the regional campaign bucket"
        )
    for field in ("repo_tar_sha256", "approval_sha256"):
        if _HEX64.fullmatch(str(value.get(field))) is None:
            raise SkyCampaignValidationError(
                f"SkyPilot campaign descriptor {field} is invalid"
            )
    artifacts = value.get("artifacts")
    if (
        not isinstance(artifacts, dict)
        or set(artifacts) != _ARTIFACT_FIELDS
        or any(
            not isinstance(artifacts.get(field), str) or not artifacts[field]
            for field in _ARTIFACT_FIELDS
        )
    ):
        raise SkyCampaignValidationError(
            "SkyPilot campaign artifact inventory is invalid"
        )
    if any(_HEX64.fullmatch(artifacts[field]) is None for field in _ARTIFACT_SHA_FIELDS):
        raise SkyCampaignValidationError(
            "SkyPilot campaign artifact SHA-256 is invalid"
        )
    exact_prefixes = {
        "source_snapshot_prefix": "source-snapshot/",
        "non_vq_prefix": "non-vq-package/",
        "training_baseline_prefix": "training-baseline/",
    }
    if any(artifacts[field] != expected for field, expected in exact_prefixes.items()):
        raise SkyCampaignValidationError(
            "SkyPilot shared artifact prefix is outside the approved inventory"
        )
    qualification_cache_sha = artifacts["qualification_cache_manifest_sha256"]
    empty_qualification_cache_sha = "0" * 64
    expected_qualification_cache_prefix = (
        "qualification-cache/"
        if qualification_cache_sha == empty_qualification_cache_sha
        else (
            f"qualification-cache/seeds/{run_id}/"
            f"{qualification_cache_sha}/"
        )
    )
    if (
        artifacts["qualification_cache_prefix"]
        != expected_qualification_cache_prefix
    ):
        raise SkyCampaignValidationError(
            "SkyPilot qualification cache prefix is not bound to its manifest"
        )
    if not artifacts["teich_pack_key"].startswith("teich-pack/") or not artifacts[
        "frozen_prompt_pack_key"
    ].startswith("quality/"):
        raise SkyCampaignValidationError(
            "SkyPilot prompt-pack key is outside the approved inventory"
        )
    expected_inventory_key = (
        campaign_prefix
        + "inventories/artifact-inventory-"
        + artifacts["artifact_inventory_sha256"]
        + ".json"
    )
    if not artifacts["training_config_key"].startswith(
        campaign_prefix + "authorities/"
    ) or artifacts["artifact_inventory_key"] != expected_inventory_key:
        raise SkyCampaignValidationError(
            "SkyPilot campaign artifact key is outside the approved inventory"
        )
    if value.get("campaign_identity_sha256") != _campaign_identity_sha256(value):
        raise SkyCampaignValidationError(
            "SkyPilot campaign identity SHA-256 mismatch"
        )
    return dict(value)


def require_approved_aws_identity(
    identity: Mapping[str, object],
) -> dict[str, object]:
    """Refuse every AWS operation outside the approved R&D account."""

    account = identity.get("Account")
    arn = identity.get("Arn")
    if account != APPROVED_ACCOUNT_ID or not isinstance(arn, str) or (
        f":{APPROVED_ACCOUNT_ID}:" not in arn
    ):
        raise SkyCampaignValidationError(
            "refusing AWS operation: active identity is not approved account "
            f"{APPROVED_ACCOUNT_ID}"
        )
    return dict(identity)


def execution_window_action(
    *,
    now: datetime,
    execution_deadline: datetime,
) -> str:
    """Return the deterministic boundary action for an approved GPU allocation."""

    if now.tzinfo is None or execution_deadline.tzinfo is None:
        raise ValueError("execution window times must be timezone-aware")
    remaining = execution_deadline - now
    if remaining <= timedelta(0):
        return "CANCEL"
    if remaining <= timedelta(minutes=30):
        return "REQUIRE_DRAINED"
    if remaining <= timedelta(minutes=35):
        return "FINAL_SYNC"
    if remaining <= timedelta(minutes=50):
        return "SIGTERM_WORKERS"
    if remaining <= timedelta(minutes=60):
        return "STOP_ASSIGNING"
    return "RUN"


def _validate_identity_map(values: Mapping[str, str], *, label: str) -> None:
    if any(not isinstance(name, str) or not name for name in values):
        raise SkyCampaignValidationError(
            f"Sky campaign {label} identity names must be non-empty strings"
        )
    for name, digest in values.items():
        if _HEX64.fullmatch(digest) is None:
            raise SkyCampaignValidationError(
                f"Sky campaign {label} identity {name!r} must be SHA-256"
            )


@dataclass(frozen=True)
class SkyCampaignRecord:
    phase: "CampaignPhase"
    input_identities: dict[str, str]
    output_identities: dict[str, str]
    timestamp: str
    execution_deadline: str
    gpu_spend_authority_sha256: str
    gpu_allocation_sha256: str
    prior_record_sha256: str
    record_sha256: str


class SkyCampaignLedger:
    """V2 phase ledger bound to a generic deadline and cumulative spend authority."""

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str,
        execution_deadline: str,
        gpu_spend_authority_sha256: str,
    ):
        self.path = Path(path)
        self.run_id = run_id
        self.execution_deadline = _iso_utc(
            execution_deadline, field="execution_deadline"
        )
        self.gpu_spend_authority_sha256 = gpu_spend_authority_sha256
        if _RUN_ID.fullmatch(run_id) is None:
            raise SkyCampaignValidationError("Sky campaign ledger run_id is invalid")
        if _HEX64.fullmatch(gpu_spend_authority_sha256) is None:
            raise SkyCampaignValidationError(
                "Sky campaign spend authority SHA-256 is invalid"
            )
        from .glm52_teich_campaign import CampaignPhase

        self._phase_type = CampaignPhase
        self._phases = tuple(CampaignPhase)
        self.records = self._read_and_validate()

    @property
    def _genesis(self) -> str:
        return _sha256(
            {
                "record_type": "glm52_sky_campaign_ledger_genesis_v2",
                "run_id": self.run_id,
                "gpu_spend_authority_sha256": self.gpu_spend_authority_sha256,
            }
        )

    @property
    def genesis_sha256(self) -> str:
        """Stable authority hash shared by all allocation records for this approval."""

        return self._genesis

    @property
    def current_phase(self) -> CampaignPhase | None:
        return self.records[-1].phase if self.records else None

    def _read_and_validate(self) -> list[SkyCampaignRecord]:
        if not self.path.exists():
            return []
        if self.path.is_symlink() or not self.path.is_file():
            raise SkyCampaignValidationError(
                "Sky campaign ledger must be a regular non-symlink file"
            )
        records: list[SkyCampaignRecord] = []
        prior_sha = self._genesis
        allocation_deadlines: dict[str, str] = {}
        required_fields = {
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
        for line_number, line in enumerate(self.path.read_text().splitlines(), start=1):
            if records and records[-1].phase is self._phase_type.DRAINED:
                raise SkyCampaignValidationError(
                    "Sky campaign ledger contains records after terminal drain"
                )
            if not line:
                raise SkyCampaignValidationError(
                    f"Sky campaign ledger line {line_number} is empty"
                )
            try:
                value = json.loads(line)
            except Exception as error:
                raise SkyCampaignValidationError(
                    f"Sky campaign ledger line {line_number} is malformed"
                ) from error
            if not isinstance(value, dict) or set(value) != required_fields:
                raise SkyCampaignValidationError("Sky campaign ledger schema mismatch")
            body = dict(value)
            record_sha = body.pop("record_sha256")
            if record_sha != _sha256(body):
                raise SkyCampaignValidationError(
                    "Sky campaign ledger record SHA-256 mismatch"
                )
            if body.get("record_type") != "glm52_sky_campaign_transition_v2":
                raise SkyCampaignValidationError("Sky campaign ledger schema mismatch")
            if (
                body.get("run_id") != self.run_id
                or body.get("gpu_spend_authority_sha256")
                != self.gpu_spend_authority_sha256
            ):
                raise SkyCampaignValidationError(
                    "Sky campaign ledger contains a foreign run or authority"
                )
            if body.get("prior_record_sha256") != prior_sha:
                raise SkyCampaignValidationError(
                    "Sky campaign ledger prior-record SHA-256 mismatch"
                )
            try:
                phase = self._phase_type(str(body.get("phase")))
            except ValueError as error:
                raise SkyCampaignValidationError(
                    "Sky campaign ledger phase is invalid"
                ) from error
            expected_phase = (
                self._phases[len(records)]
                if len(records) < len(self._phases)
                else None
            )
            if phase != expected_phase and phase is not self._phase_type.DRAINED:
                raise SkyCampaignValidationError(
                    "Sky campaign ledger phases are not contiguous"
                )
            inputs = body.get("input_identities")
            outputs = body.get("output_identities")
            if not isinstance(inputs, dict) or not isinstance(outputs, dict):
                raise SkyCampaignValidationError(
                    "Sky campaign ledger identities must be objects"
                )
            _validate_identity_map(inputs, label="input")
            _validate_identity_map(outputs, label="output")
            allocation_sha = str(body.get("gpu_allocation_sha256"))
            if _HEX64.fullmatch(allocation_sha) is None:
                raise SkyCampaignValidationError(
                    "Sky campaign GPU allocation SHA-256 is invalid"
                )
            timestamp = _iso_utc(body.get("timestamp"), field="timestamp")
            record_deadline = _iso_utc(
                body.get("execution_deadline"),
                field="execution_deadline",
            )
            if _parsed_time(record_deadline, field="execution_deadline") <= (
                _parsed_time(timestamp, field="timestamp")
            ):
                raise SkyCampaignValidationError(
                    "Sky campaign execution deadline must follow transition"
                )
            prior_deadline = allocation_deadlines.setdefault(
                allocation_sha,
                record_deadline,
            )
            if prior_deadline != record_deadline:
                raise SkyCampaignValidationError(
                    "Sky campaign allocation deadline changed within one allocation"
                )
            records.append(
                SkyCampaignRecord(
                    phase=phase,
                    input_identities=dict(inputs),
                    output_identities=dict(outputs),
                    timestamp=timestamp,
                    execution_deadline=record_deadline,
                    gpu_spend_authority_sha256=self.gpu_spend_authority_sha256,
                    gpu_allocation_sha256=allocation_sha,
                    prior_record_sha256=prior_sha,
                    record_sha256=str(record_sha),
                )
            )
            prior_sha = str(record_sha)
        return records

    def transition(
        self,
        phase: "CampaignPhase",
        *,
        input_identities: Mapping[str, str],
        output_identities: Mapping[str, str],
        gpu_allocation_sha256: str,
        timestamp: datetime | None = None,
    ) -> SkyCampaignRecord:
        expected = (
            self._phases[len(self.records)]
            if len(self.records) < len(self._phases)
            else None
        )
        allow_terminal_drain = (
            phase is self._phase_type.DRAINED
            and bool(self.records)
            and self.current_phase is not self._phase_type.DRAINED
        )
        if phase != expected and not allow_terminal_drain:
            raise SkyCampaignValidationError(
                f"Sky campaign next phase must be "
                f"{expected.value if expected else 'none'}"
            )
        _validate_identity_map(input_identities, label="input")
        _validate_identity_map(output_identities, label="output")
        if _HEX64.fullmatch(gpu_allocation_sha256) is None:
            raise SkyCampaignValidationError(
                "Sky campaign GPU allocation SHA-256 is invalid"
            )
        now = timestamp or datetime.now(timezone.utc)
        timestamp_value = _iso_utc(now, field="timestamp")
        prior_sha = (
            self.records[-1].record_sha256 if self.records else self._genesis
        )
        body: dict[str, object] = {
            "record_type": "glm52_sky_campaign_transition_v2",
            "run_id": self.run_id,
            "phase": phase.value,
            "input_identities": dict(sorted(input_identities.items())),
            "output_identities": dict(sorted(output_identities.items())),
            "timestamp": timestamp_value,
            "execution_deadline": self.execution_deadline,
            "gpu_spend_authority_sha256": self.gpu_spend_authority_sha256,
            "gpu_allocation_sha256": gpu_allocation_sha256,
            "prior_record_sha256": prior_sha,
        }
        record = {**body, "record_sha256": _sha256(body)}
        encoded = _canonical_bytes(record) + b"\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self.path,
            os.O_CREAT | os.O_APPEND | os.O_WRONLY,
            0o600,
        )
        try:
            if os.write(descriptor, encoded) != len(encoded):
                raise OSError("short Sky campaign ledger append")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        self.records = self._read_and_validate()
        return self.records[-1]


class GpuSpendLedger:
    """Append-only allocation ledger whose budget survives node replacement."""

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str,
        approval_sha256: str,
        approved_gpu_runtime_seconds: int,
        approved_gpu_cost_usd: float,
        hourly_cost_usd: float,
    ):
        self.path = Path(path)
        self.run_id = run_id
        self.approval_sha256 = approval_sha256
        self.approved_gpu_runtime_seconds = approved_gpu_runtime_seconds
        self.approved_gpu_cost_usd = approved_gpu_cost_usd
        self.hourly_cost_usd = hourly_cost_usd
        if _RUN_ID.fullmatch(run_id) is None:
            raise SkyCampaignValidationError("GPU spend ledger run_id is invalid")
        if _HEX64.fullmatch(approval_sha256) is None:
            raise SkyCampaignValidationError(
                "GPU spend ledger approval SHA-256 is invalid"
            )
        if approved_gpu_runtime_seconds != APPROVED_GPU_RUNTIME_SECONDS:
            raise SkyCampaignValidationError(
                "GPU spend ledger runtime does not match approved authority"
            )
        if approved_gpu_cost_usd != APPROVED_GPU_COST_USD:
            raise SkyCampaignValidationError(
                "GPU spend ledger cost does not match approved authority"
            )
        if hourly_cost_usd != APPROVED_HOURLY_COST_USD:
            raise SkyCampaignValidationError(
                "GPU spend ledger hourly price does not match approved authority"
            )
        self.records = self._read_and_validate()

    @property
    def _genesis(self) -> str:
        return _sha256(
            {
                "record_type": "glm52_gpu_spend_ledger_genesis_v1",
                "run_id": self.run_id,
                "approval_sha256": self.approval_sha256,
                "approved_gpu_runtime_seconds": self.approved_gpu_runtime_seconds,
                "approved_gpu_cost_usd": self.approved_gpu_cost_usd,
                "hourly_cost_usd": self.hourly_cost_usd,
            }
        )

    @property
    def genesis_sha256(self) -> str:
        """Stable authority hash for the cumulative approved-spend stream."""

        return self._genesis

    def _read_and_validate(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        if self.path.is_symlink() or not self.path.is_file():
            raise SkyCampaignValidationError(
                "GPU spend ledger must be a regular non-symlink file"
            )
        records: list[dict[str, object]] = []
        prior_sha = self._genesis
        active: dict[str, object] | None = None
        for line_number, line in enumerate(self.path.read_text().splitlines(), start=1):
            if not line:
                raise SkyCampaignValidationError(
                    f"GPU spend ledger line {line_number} is empty"
                )
            try:
                value = json.loads(line)
            except Exception as error:
                raise SkyCampaignValidationError(
                    f"GPU spend ledger line {line_number} is malformed"
                ) from error
            if not isinstance(value, dict):
                raise SkyCampaignValidationError(
                    "GPU spend ledger records must be objects"
                )
            body = dict(value)
            record_sha = body.pop("record_sha256", None)
            if record_sha != _sha256(body):
                raise SkyCampaignValidationError(
                    "GPU spend ledger record SHA-256 mismatch"
                )
            if body.get("record_type") != "glm52_gpu_spend_event_v1":
                raise SkyCampaignValidationError("GPU spend ledger schema mismatch")
            if body.get("run_id") != self.run_id:
                raise SkyCampaignValidationError("GPU spend ledger contains a foreign run")
            if body.get("approval_sha256") != self.approval_sha256:
                raise SkyCampaignValidationError(
                    "GPU spend ledger contains foreign approval authority"
                )
            if body.get("prior_record_sha256") != prior_sha:
                raise SkyCampaignValidationError(
                    "GPU spend ledger prior-record SHA-256 mismatch"
                )
            event = body.get("event")
            instance_id = body.get("instance_id")
            if not isinstance(instance_id, str) or not instance_id:
                raise SkyCampaignValidationError(
                    "GPU spend ledger instance_id is invalid"
                )
            timestamp = _parsed_time(body.get("timestamp"), field="timestamp")
            if event == "allocation_started":
                if active is not None:
                    raise SkyCampaignValidationError(
                        "GPU spend ledger contains parallel active allocation"
                    )
                if not isinstance(body.get("job_id"), str) or not body["job_id"]:
                    raise SkyCampaignValidationError(
                        "GPU spend ledger job_id is invalid"
                    )
                active = {
                    "instance_id": instance_id,
                    "job_id": body["job_id"],
                    "launched_at": timestamp,
                }
            elif event == "allocation_ended":
                if active is None or active["instance_id"] != instance_id:
                    raise SkyCampaignValidationError(
                        "GPU spend ledger allocation end is noncontiguous"
                    )
                if timestamp < active["launched_at"]:
                    raise SkyCampaignValidationError(
                        "GPU spend ledger allocation end precedes launch"
                    )
                active = None
            else:
                raise SkyCampaignValidationError(
                    "GPU spend ledger event is invalid"
                )
            records.append(dict(value))
            prior_sha = str(record_sha)
        return records

    def _append(self, body: dict[str, object]) -> dict[str, object]:
        prior_sha = (
            str(self.records[-1]["record_sha256"]) if self.records else self._genesis
        )
        complete_body = {
            "record_type": "glm52_gpu_spend_event_v1",
            "run_id": self.run_id,
            "approval_sha256": self.approval_sha256,
            **body,
            "prior_record_sha256": prior_sha,
        }
        record = {
            **complete_body,
            "record_sha256": _sha256(complete_body),
        }
        encoded = _canonical_bytes(record) + b"\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self.path,
            os.O_CREAT | os.O_APPEND | os.O_WRONLY,
            0o600,
        )
        try:
            if os.write(descriptor, encoded) != len(encoded):
                raise OSError("short GPU spend ledger append")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        self.records = self._read_and_validate()
        return self.records[-1]

    def _allocations(
        self, *, now: datetime | None = None
    ) -> list[tuple[str, str, datetime, datetime | None]]:
        allocations: list[tuple[str, str, datetime, datetime | None]] = []
        active: tuple[str, str, datetime] | None = None
        for record in self.records:
            timestamp = _parsed_time(record["timestamp"], field="timestamp")
            if record["event"] == "allocation_started":
                active = (
                    str(record["job_id"]),
                    str(record["instance_id"]),
                    timestamp,
                )
            else:
                assert active is not None
                allocations.append((*active, timestamp))
                active = None
        if active is not None:
            allocations.append((*active, None))
        if now is not None and now.tzinfo is None:
            raise ValueError("GPU spend calculation time must be timezone-aware")
        return allocations

    def start_allocation(
        self,
        *,
        job_id: str,
        instance_id: str,
        launched_at: datetime,
    ) -> dict[str, object]:
        launched = _iso_utc(launched_at, field="launched_at")
        allocations = self._allocations()
        if allocations and allocations[-1][3] is None:
            active = allocations[-1]
            if (
                active[0] == job_id
                and active[1] == instance_id
                and _iso_utc(active[2], field="launched_at") == launched
            ):
                return self.records[-1]
            raise SkyCampaignValidationError(
                "GPU spend ledger already has an active allocation"
            )
        if not job_id or not instance_id:
            raise SkyCampaignValidationError(
                "GPU spend allocation identity must be non-empty"
            )
        return self._append(
            {
                "event": "allocation_started",
                "job_id": job_id,
                "instance_id": instance_id,
                "timestamp": launched,
            }
        )

    def end_allocation(
        self,
        *,
        instance_id: str,
        ended_at: datetime,
    ) -> dict[str, object]:
        ended = _iso_utc(ended_at, field="ended_at")
        allocations = self._allocations()
        if not allocations or allocations[-1][3] is not None:
            if (
                self.records
                and self.records[-1]["event"] == "allocation_ended"
                and self.records[-1]["instance_id"] == instance_id
                and self.records[-1]["timestamp"] == ended
            ):
                return self.records[-1]
            raise SkyCampaignValidationError(
                "GPU spend ledger has no active allocation"
            )
        active = allocations[-1]
        if active[1] != instance_id:
            raise SkyCampaignValidationError(
                "GPU spend allocation end instance does not match active allocation"
            )
        if _parsed_time(ended, field="ended_at") < active[2]:
            raise SkyCampaignValidationError(
                "GPU spend allocation end precedes launch"
            )
        return self._append(
            {
                "event": "allocation_ended",
                "instance_id": instance_id,
                "timestamp": ended,
            }
        )

    def consumed_gpu_seconds(self, *, now: datetime) -> float:
        if now.tzinfo is None:
            raise ValueError("GPU spend calculation time must be timezone-aware")
        consumed = 0.0
        for _job_id, _instance_id, launched_at, ended_at in self._allocations():
            effective_end = ended_at or now.astimezone(timezone.utc)
            if effective_end < launched_at:
                raise SkyCampaignValidationError(
                    "GPU spend calculation precedes active allocation"
                )
            consumed += (effective_end - launched_at).total_seconds()
        return consumed

    def remaining_gpu_seconds(self, *, now: datetime) -> float:
        return max(
            0.0,
            self.approved_gpu_runtime_seconds - self.consumed_gpu_seconds(now=now),
        )

    def estimated_gpu_cost_usd(self, *, now: datetime) -> float:
        return round(
            self.consumed_gpu_seconds(now=now) / 3600 * self.hourly_cost_usd,
            2,
        )

    def execution_deadline(self, *, now: datetime) -> datetime:
        if now.tzinfo is None:
            raise ValueError("execution deadline time must be timezone-aware")
        return now + timedelta(seconds=self.remaining_gpu_seconds(now=now))


__all__ = [
    "APPROVED_ACCOUNT_ID",
    "APPROVED_GPU_COST_USD",
    "APPROVED_GPU_HOURS",
    "APPROVED_GPU_RUNTIME_SECONDS",
    "APPROVED_HOURLY_COST_USD",
    "APPROVED_INSTANCE_TYPE",
    "APPROVED_REGION",
    "GPU_APPROVAL_REQUEST",
    "GpuSpendLedger",
    "PINNED_SKYPILOT_VERSION",
    "SkyCampaignValidationError",
    "SkyCampaignLedger",
    "SkyCampaignRecord",
    "build_gpu_spend_approval",
    "build_sky_campaign_descriptor",
    "execution_window_action",
    "require_approved_aws_identity",
    "validate_gpu_spend_approval",
    "validate_sky_campaign_descriptor",
]
