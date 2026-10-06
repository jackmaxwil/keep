"""Pure cache-seed intent and durable launch-claim authority."""

# ruff: noqa: UP017

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import datetime, timezone

try:
    from glm52_gpu_launch_allowance import (
        GpuLaunchAllowanceError,
        gpu_launch_allowance_s3_key,
        validate_gpu_launch_allowance,
    )
    from glm52_sky_campaign import (
        APPROVED_ACCOUNT_ID,
        APPROVED_GPU_COST_USD,
        APPROVED_GPU_RUNTIME_SECONDS,
        APPROVED_REGION,
        SkyCampaignValidationError,
        validate_gpu_spend_approval,
        validate_sky_campaign_descriptor,
    )
except ModuleNotFoundError as error:
    if error.name not in {
        "glm52_gpu_launch_allowance",
        "glm52_sky_campaign",
    }:
        raise
    from mlx_vq.quality.glm52_gpu_launch_allowance import (
        GpuLaunchAllowanceError,
        gpu_launch_allowance_s3_key,
        validate_gpu_launch_allowance,
    )
    from mlx_vq.quality.glm52_sky_campaign import (
        APPROVED_ACCOUNT_ID,
        APPROVED_GPU_COST_USD,
        APPROVED_GPU_RUNTIME_SECONDS,
        APPROVED_REGION,
        SkyCampaignValidationError,
        validate_gpu_spend_approval,
        validate_sky_campaign_descriptor,
    )


INTENT_SCHEMA_VERSION = 1
INTENT_RECORD_TYPE = "glm52_sky_cache_seed_submission_intent_v1"
INTENT_DIGEST_FIELD = "intent_body_sha256"

CLAIM_SCHEMA_VERSION = 1
CLAIM_RECORD_TYPE = "glm52_sky_cache_seed_launch_claim_v1"
CLAIM_DIGEST_FIELD = "launch_claim_body_sha256"

MANAGED_MODE = "cache-seed"
MAX_LIVE_AGE_SECONDS = 60

_HEX64 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_STAGED_FIELDS = {
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
_STAGED_ROLES = {
    "repository_tar",
    "approval",
    "training_config",
    "watchdog",
    "artifact_inventory",
    "artifact_audit",
    "descriptor",
}
_MANIFEST_FIELDS = {
    "schema_version",
    "record_type",
    "run_id",
    "bucket",
    "files",
    "descriptor_body_sha256",
    "bundle_manifest_body_sha256",
}
_MANIFEST_FILE_FIELDS = {
    "local_name",
    "key",
    "role",
    "stage_order",
    "size",
    "sha256",
}
_MANIFEST_ROLES = {
    "repository_tar",
    "approval",
    "training_config",
    "watchdog",
    "artifact_inventory",
    "descriptor",
}
_REHEARSAL_FIELDS = {
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
_INTENT_BODY_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "run_id",
    "managed_mode",
    "sky_job_name",
    "descriptor_key",
    "descriptor_file_sha256",
    "descriptor_body_sha256",
    "campaign_identity_sha256",
    "approval_key",
    "approval_sha256",
    "approval_body_sha256",
    "repo_tar_key",
    "repo_tar_sha256",
    "staged_readiness_key",
    "staged_readiness_file_sha256",
    "staged_readiness_body_sha256",
    "staged_readiness_version_id",
    "bundle_manifest_key",
    "bundle_manifest_file_sha256",
    "bundle_manifest_body_sha256",
    "bundle_manifest_version_id",
    "rehearsal_evidence_key",
    "rehearsal_evidence_file_sha256",
    "rehearsal_evidence_body_sha256",
    "s3_spend_ledger_absence_observation_key",
    "s3_spend_ledger_absence_observation_sha256",
    "ec2_tagged_p5_zero_inventory_observation_key",
    "ec2_tagged_p5_zero_inventory_observation_sha256",
    "launch_allowance_key",
    "launch_allowance_file_sha256",
    "launch_allowance_body_sha256",
    "zero_cache_prefix",
    "zero_cache_manifest_sha256",
    "remaining_gpu_seconds",
    "remaining_gpu_cost_usd",
    "cache_seed_allowance_seconds",
    "cache_seed_allowance_cost_usd",
    "open_allocation_count",
    "must_start_by",
    "not_after",
    "intent_at",
}
_INTENT_FIELDS = _INTENT_BODY_FIELDS | {INTENT_DIGEST_FIELD}
_CLAIM_BODY_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "bucket",
    "run_id",
    "managed_mode",
    "sky_job_name",
    "descriptor_key",
    "descriptor_file_sha256",
    "descriptor_body_sha256",
    "campaign_identity_sha256",
    "approval_key",
    "approval_sha256",
    "approval_body_sha256",
    "staged_readiness_key",
    "staged_readiness_file_sha256",
    "staged_readiness_body_sha256",
    "staged_readiness_version_id",
    "bundle_manifest_key",
    "bundle_manifest_file_sha256",
    "bundle_manifest_body_sha256",
    "bundle_manifest_version_id",
    "rehearsal_evidence_key",
    "rehearsal_evidence_file_sha256",
    "rehearsal_evidence_body_sha256",
    "s3_spend_ledger_absence_observation_key",
    "s3_spend_ledger_absence_observation_sha256",
    "ec2_tagged_p5_zero_inventory_observation_key",
    "ec2_tagged_p5_zero_inventory_observation_sha256",
    "launch_allowance_key",
    "launch_allowance_file_sha256",
    "launch_allowance_body_sha256",
    "submission_intent_key",
    "submission_intent_file_sha256",
    "submission_intent_body_sha256",
    "claimed_at",
    "must_start_by",
    "not_after",
}
_CLAIM_FIELDS = _CLAIM_BODY_FIELDS | {CLAIM_DIGEST_FIELD}


class CacheSeedLaunchError(ValueError):
    """An incomplete, stale, foreign, or ambiguous cache-seed authority."""


def _require_exact_json(value: object, *, field: str = "value") -> None:
    value_type = type(value)
    if value is None or value_type in {bool, int, str}:
        return
    if value_type is float:
        if not math.isfinite(value):
            raise CacheSeedLaunchError(f"{field} is not finite exact JSON")
        return
    if value_type is list:
        for index, member in enumerate(value):
            _require_exact_json(member, field=f"{field}[{index}]")
        return
    if value_type is dict:
        for key, member in value.items():
            if type(key) is not str:
                raise CacheSeedLaunchError(
                    f"{field} contains a non-string JSON object key"
                )
            _require_exact_json(member, field=f"{field}.{key}")
        return
    raise CacheSeedLaunchError(f"{field} contains a non-JSON Python type")


def _canonical(value: object) -> bytes:
    _require_exact_json(value)
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CacheSeedLaunchError("value is not finite canonical JSON") from error


def canonical_file_bytes(value: Mapping[str, object]) -> bytes:
    """Return canonical durable object bytes with exactly one trailing LF."""

    if type(value) is not dict:
        raise CacheSeedLaunchError("canonical file value must be an exact dict")
    return _canonical(value) + b"\n"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise CacheSeedLaunchError(f"{field} must be a lowercase SHA-256")
    return value


def _version_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value == "null":
        raise CacheSeedLaunchError(f"{field} must be an opaque VersionId")
    if any(ord(character) < 0x21 or ord(character) > 0x7E for character in value):
        raise CacheSeedLaunchError(f"{field} must be an opaque VersionId")
    return value


def _run_id(value: object) -> str:
    if not isinstance(value, str) or _RUN_ID.fullmatch(value) is None:
        raise CacheSeedLaunchError("run_id is invalid")
    return value


def _safe_key(value: object, *, field: str, prefix: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise CacheSeedLaunchError(f"{field} S3 key is invalid")
    candidate = value
    if prefix:
        if not candidate.endswith("/") or candidate.endswith("//"):
            raise CacheSeedLaunchError(f"{field} S3 prefix is invalid")
        candidate = candidate[:-1]
    elif candidate.startswith("/") or candidate.endswith("/"):
        raise CacheSeedLaunchError(f"{field} S3 key is invalid")
    if (
        not candidate
        or "\\" in candidate
        or any(character in candidate for character in "*?[]")
        or any(
            ord(character) < 0x21 or ord(character) > 0x7E for character in candidate
        )
        or any(segment in {"", ".", ".."} for segment in candidate.split("/"))
    ):
        raise CacheSeedLaunchError(f"{field} S3 key is invalid")
    return value


def _content_key(
    *,
    run_id: str,
    digest: str,
    relative: str,
    field: str,
) -> str:
    run = _run_id(run_id)
    body_digest = _digest(digest, field=field)
    return _safe_key(
        f"campaigns/{run}/submissions/{MANAGED_MODE}/{relative.format(digest=body_digest)}",
        field=field,
    )


def s3_spend_ledger_absence_observation_s3_key(
    *,
    run_id: str,
    observation_file_sha256: str,
) -> str:
    return _content_key(
        run_id=run_id,
        digest=observation_file_sha256,
        relative=(
            "launch-observations/s3-spend-ledger-absence/{digest}/"
            "S3_SPEND_LEDGER_ABSENCE_OBSERVATION.json"
        ),
        field="S3 spend-ledger absence observation",
    )


def ec2_tagged_p5_zero_inventory_observation_s3_key(
    *,
    run_id: str,
    observation_file_sha256: str,
) -> str:
    return _content_key(
        run_id=run_id,
        digest=observation_file_sha256,
        relative=(
            "launch-observations/ec2-tagged-p5-zero-inventory/{digest}/"
            "EC2_TAGGED_P5_ZERO_INVENTORY_OBSERVATION.json"
        ),
        field="EC2 tagged-P5 zero-inventory observation",
    )


def cache_seed_submission_intent_s3_key(
    *,
    run_id: str,
    intent_body_sha256: str,
) -> str:
    return _content_key(
        run_id=run_id,
        digest=intent_body_sha256,
        relative="intents/{digest}/SKYPILOT_SUBMISSION_INTENT.json",
        field="cache-seed submission intent",
    )


def cache_seed_launch_claim_s3_key(
    *,
    run_id: str,
    descriptor_file_sha256: str,
) -> str:
    return _content_key(
        run_id=run_id,
        digest=descriptor_file_sha256,
        relative="launch-claims/{digest}/LAUNCH_CLAIM.json",
        field="cache-seed launch claim",
    )


def _canonical_time(value: datetime | str, *, field: str) -> tuple[datetime, str]:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise CacheSeedLaunchError(f"{field} must be timezone-aware")
        parsed = value.astimezone(timezone.utc)
    elif isinstance(value, str):
        try:
            parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError as error:
            raise CacheSeedLaunchError(
                f"{field} must be canonical whole-second UTC"
            ) from error
    else:
        raise CacheSeedLaunchError(f"{field} must be canonical whole-second UTC")
    if parsed.microsecond:
        raise CacheSeedLaunchError(f"{field} must be canonical whole-second UTC")
    canonical = parsed.strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(value, str) and value != canonical:
        raise CacheSeedLaunchError(f"{field} must be canonical whole-second UTC")
    return parsed, canonical


def _file_digest(
    value: Mapping[str, object],
    expected: object,
    *,
    field: str,
) -> str:
    claimed = _digest(expected, field=field)
    if _sha(canonical_file_bytes(value)) != claimed:
        raise CacheSeedLaunchError(f"{field} exact canonical file SHA-256 mismatch")
    return claimed


def _self_hash(
    value: Mapping[str, object],
    *,
    fields: set[str],
    digest_field: str,
    label: str,
) -> None:
    if set(value) != fields:
        raise CacheSeedLaunchError(f"{label} schema mismatch")
    body = dict(value)
    digest = _digest(body.pop(digest_field), field=digest_field)
    if _sha(_canonical(body)) != digest:
        raise CacheSeedLaunchError(f"{label} body SHA-256 mismatch")


def _descriptor_approval(
    descriptor: Mapping[str, object],
    descriptor_file_sha256: str,
    approval: Mapping[str, object],
    approval_file_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    try:
        descriptor_value = validate_sky_campaign_descriptor(descriptor)
        approval_value = validate_gpu_spend_approval(approval)
    except (SkyCampaignValidationError, ValueError) as error:
        raise CacheSeedLaunchError(str(error)) from error
    descriptor_sha = _file_digest(
        descriptor_value,
        descriptor_file_sha256,
        field="descriptor_file_sha256",
    )
    approval_sha = _file_digest(
        approval_value,
        approval_file_sha256,
        field="approval_file_sha256",
    )
    exact = {
        "schema_version": 2,
        "account_id": APPROVED_ACCOUNT_ID,
        "region": APPROVED_REGION,
        "instance_type": "p5.48xlarge",
        "instance_count": 1,
        "use_spot": False,
        "task_name": "glm52-campaign",
    }
    exact_types = {
        "schema_version": int,
        "account_id": str,
        "region": str,
        "instance_type": str,
        "instance_count": int,
        "use_spot": bool,
        "task_name": str,
    }
    for field, expected in exact.items():
        if (
            type(descriptor_value.get(field)) is not exact_types[field]
            or descriptor_value.get(field) != expected
        ):
            raise CacheSeedLaunchError(f"descriptor {field} is foreign")
    if descriptor_value.get("approval_sha256") != approval_sha:
        raise CacheSeedLaunchError("descriptor approval binding mismatch")
    artifacts = descriptor_value.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise CacheSeedLaunchError("descriptor artifacts are malformed")
    if (
        artifacts.get("qualification_cache_prefix") != "qualification-cache/"
        or artifacts.get("qualification_cache_manifest_sha256") != "0" * 64
    ):
        raise CacheSeedLaunchError("descriptor is not the zero-cache authority")
    _run_id(descriptor_value.get("run_id"))
    descriptor_key = _safe_key(
        descriptor_value.get("campaign_descriptor_key"),
        field="descriptor",
    )
    expected_prefix = f"campaigns/{descriptor_value['run_id']}/submissions/"
    suffix = "/campaign-descriptor-v2.json"
    if not descriptor_key.startswith(expected_prefix) or not descriptor_key.endswith(
        suffix
    ):
        raise CacheSeedLaunchError("descriptor submission coordinate is invalid")
    submission_id = descriptor_key[len(expected_prefix) : -len(suffix)]
    if not submission_id or "/" in submission_id or submission_id in {".", ".."}:
        raise CacheSeedLaunchError("descriptor submission coordinate is invalid")
    del descriptor_sha
    return dict(descriptor_value), dict(approval_value)


def _validate_staged_chain(
    *,
    descriptor: Mapping[str, object],
    descriptor_file_sha256: str,
    staged_readiness: Mapping[str, object],
    staged_readiness_file_sha256: str,
    staged_readiness_version_id: str,
    bundle_manifest: Mapping[str, object],
    bundle_manifest_file_sha256: str,
    rehearsal_evidence: Mapping[str, object],
    rehearsal_evidence_file_sha256: str,
) -> tuple[str, str]:
    _version_id(
        staged_readiness_version_id,
        field="staged_readiness_version_id",
    )
    _self_hash(
        staged_readiness,
        fields=_STAGED_FIELDS,
        digest_field="ready_body_sha256",
        label="staged readiness",
    )
    if (
        type(staged_readiness.get("schema_version")) is not int
        or staged_readiness.get("schema_version") != 2
        or staged_readiness.get("record_type") != "glm52_staged_control_plane_ready_v2"
    ):
        raise CacheSeedLaunchError("staged readiness is not exact v2")
    run_id = str(descriptor["run_id"])
    descriptor_key = str(descriptor["campaign_descriptor_key"])
    staged_key = (
        descriptor_key.removesuffix("campaign-descriptor-v2.json")
        + "STAGED_CONTROL_PLANE_READY.json"
    )
    exact_staged = {
        "run_id": run_id,
        "descriptor_key": descriptor_key,
        "descriptor_sha256": descriptor_file_sha256,
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
    }
    if any(staged_readiness.get(k) != v for k, v in exact_staged.items()):
        raise CacheSeedLaunchError("staged readiness descriptor binding mismatch")
    _file_digest(
        staged_readiness,
        staged_readiness_file_sha256,
        field="staged_readiness_file_sha256",
    )
    versions = staged_readiness.get("staged_object_version_ids")
    if not isinstance(versions, Mapping) or set(versions) != _STAGED_ROLES:
        raise CacheSeedLaunchError("staged readiness VersionId roles mismatch")
    for role in sorted(_STAGED_ROLES):
        _version_id(versions[role], field=f"{role} VersionId")
    manifest_version = _version_id(
        staged_readiness.get("bundle_manifest_version_id"),
        field="bundle manifest VersionId",
    )
    _self_hash(
        bundle_manifest,
        fields=_MANIFEST_FIELDS,
        digest_field="bundle_manifest_body_sha256",
        label="bundle manifest",
    )
    if (
        type(bundle_manifest.get("schema_version")) is not int
        or bundle_manifest.get("schema_version") != 1
        or bundle_manifest.get("record_type") != "glm52_sky_campaign_bundle_v1"
        or bundle_manifest.get("run_id") != run_id
        or bundle_manifest.get("bucket") != descriptor["bucket"]
        or bundle_manifest.get("descriptor_body_sha256")
        != descriptor["descriptor_body_sha256"]
    ):
        raise CacheSeedLaunchError("bundle manifest authority mismatch")
    manifest_file_sha = _file_digest(
        bundle_manifest,
        bundle_manifest_file_sha256,
        field="bundle_manifest_file_sha256",
    )
    manifest_body_sha = str(bundle_manifest["bundle_manifest_body_sha256"])
    expected_manifest_key = (
        descriptor_key.removesuffix("campaign-descriptor-v2.json")
        + f"bundle-manifests/{manifest_body_sha}/bundle-manifest-v1.json"
    )
    if (
        staged_readiness.get("bundle_manifest_key") != expected_manifest_key
        or staged_readiness.get("bundle_manifest_file_sha256") != manifest_file_sha
        or staged_readiness.get("bundle_manifest_body_sha256") != manifest_body_sha
    ):
        raise CacheSeedLaunchError("staged readiness manifest binding mismatch")
    files = bundle_manifest.get("files")
    if not isinstance(files, list) or len(files) != len(_MANIFEST_ROLES):
        raise CacheSeedLaunchError("bundle manifest roles are incomplete")
    if any(not isinstance(item, Mapping) for item in files):
        raise CacheSeedLaunchError("bundle manifest file is malformed")
    copied = [dict(item) for item in files]
    if any(set(item) != _MANIFEST_FILE_FIELDS for item in copied):
        raise CacheSeedLaunchError("bundle manifest file schema mismatch")
    roles = [item.get("role") for item in copied]
    if set(roles) != _MANIFEST_ROLES or len(roles) != len(set(roles)):
        raise CacheSeedLaunchError("bundle manifest roles mismatch")
    orders: list[int] = []
    for item in copied:
        _safe_key(item.get("key"), field="bundle manifest object")
        _digest(item.get("sha256"), field="bundle manifest object SHA-256")
        if (
            not isinstance(item.get("local_name"), str)
            or not item["local_name"]
            or "/" in str(item["local_name"])
            or "\\" in str(item["local_name"])
            or type(item.get("size")) is not int
            or int(item["size"]) < 0
            or type(item.get("stage_order")) is not int
            or int(item["stage_order"]) < 0
        ):
            raise CacheSeedLaunchError("bundle manifest file coordinate is invalid")
        orders.append(int(item["stage_order"]))
    if len(orders) != len(set(orders)):
        raise CacheSeedLaunchError("bundle manifest stage order is duplicated")
    ordered = sorted(copied, key=lambda item: int(item["stage_order"]))
    if ordered[-1].get("role") != "descriptor":
        raise CacheSeedLaunchError("bundle manifest descriptor is not staged last")
    descriptor_item = next(item for item in copied if item.get("role") == "descriptor")
    if (
        descriptor_item.get("key") != descriptor_key
        or descriptor_item.get("sha256") != descriptor_file_sha256
        or descriptor_item.get("size") != len(canonical_file_bytes(descriptor))
    ):
        raise CacheSeedLaunchError("bundle manifest descriptor binding mismatch")
    _self_hash(
        rehearsal_evidence,
        fields=_REHEARSAL_FIELDS,
        digest_field="rehearsal_body_sha256",
        label="rehearsal evidence",
    )
    if (
        type(rehearsal_evidence.get("schema_version")) is not int
        or rehearsal_evidence.get("schema_version") != 2
        or rehearsal_evidence.get("record_type")
        != "glm52_staged_control_plane_rehearsal_v2"
        or rehearsal_evidence.get("status") != "passed_before_cuda_h100_boundary"
        or rehearsal_evidence.get("skypilot_version") != "0.13.0"
        or rehearsal_evidence.get("skypilot_task_name") != "glm52-campaign"
        or rehearsal_evidence.get("production_repo_path") != "/opt/keep-campaign/repo"
        or rehearsal_evidence.get("production_resume_root")
        != "/mnt/nvme/glm52-campaign"
        or rehearsal_evidence.get("skypilot_task_path")
        != ("/opt/keep-campaign/repo/aws/glm52-gpu/skypilot/glm52-campaign.yaml")
    ):
        raise CacheSeedLaunchError("rehearsal evidence authority mismatch")
    exact_rehearsal = {
        "run_id": run_id,
        "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
        "descriptor_key": descriptor_key,
        "descriptor_file_sha256": descriptor_file_sha256,
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "repo_tar_sha256": descriptor["repo_tar_sha256"],
        "staged_readiness_key": staged_key,
        "staged_readiness_file_sha256": staged_readiness_file_sha256,
        "staged_readiness_body_sha256": staged_readiness["ready_body_sha256"],
        "artifact_audit_key": staged_readiness["artifact_audit_key"],
        "artifact_audit_file_sha256": staged_readiness["artifact_audit_sha256"],
    }
    artifacts = descriptor["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise CacheSeedLaunchError("descriptor artifacts are malformed")
    exact_rehearsal.update(
        {
            "artifact_inventory_key": artifacts["artifact_inventory_key"],
            "artifact_inventory_file_sha256": artifacts["artifact_inventory_sha256"],
        }
    )
    if any(rehearsal_evidence.get(k) != v for k, v in exact_rehearsal.items()):
        raise CacheSeedLaunchError("rehearsal evidence binding mismatch")
    for field in (
        "bootstrap_receipt_file_sha256",
        "artifact_inventory_body_sha256",
        "skypilot_task_file_sha256",
        "skypilot_config_file_sha256",
        "skypilot_validation_file_sha256",
        "rehearsal_body_sha256",
    ):
        _digest(rehearsal_evidence.get(field), field=field)
    _file_digest(
        rehearsal_evidence,
        rehearsal_evidence_file_sha256,
        field="rehearsal_evidence_file_sha256",
    )
    return staged_key, manifest_version


def _validate_allowance(
    *,
    descriptor: Mapping[str, object],
    descriptor_file_sha256: str,
    approval: Mapping[str, object],
    approval_file_sha256: str,
    launch_allowance: Mapping[str, object],
    launch_allowance_file_sha256: str,
    s3_observation_file_sha256: str,
    ec2_observation_file_sha256: str,
) -> dict[str, object]:
    try:
        allowance = validate_gpu_launch_allowance(launch_allowance)
    except (GpuLaunchAllowanceError, ValueError) as error:
        raise CacheSeedLaunchError(str(error)) from error
    _file_digest(
        allowance,
        launch_allowance_file_sha256,
        field="launch_allowance_file_sha256",
    )
    expected = {
        "run_id": descriptor["run_id"],
        "managed_mode": MANAGED_MODE,
        "sky_job_name": f"{descriptor['run_id']}-{MANAGED_MODE}",
        "account_id": APPROVED_ACCOUNT_ID,
        "region": APPROVED_REGION,
        "bucket": descriptor["bucket"],
        "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
        "descriptor_sha256": descriptor_file_sha256,
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "descriptor_key": descriptor["campaign_descriptor_key"],
        "approval_sha256": approval_file_sha256,
        "approval_body_sha256": approval["approval_body_sha256"],
        "approval_key": descriptor["approval_key"],
        "remaining_gpu_seconds": APPROVED_GPU_RUNTIME_SECONDS,
        "remaining_gpu_cost_usd": float(APPROVED_GPU_COST_USD),
        "cache_seed_max_seconds": 21_600,
        "cache_seed_max_cost_usd": 330.24,
        "open_allocation_count": 0,
        "s3_spend_ledger_absence_observation_sha256": (s3_observation_file_sha256),
        "ec2_tagged_p5_zero_inventory_observation_sha256": (
            ec2_observation_file_sha256
        ),
    }
    if any(allowance.get(k) != v for k, v in expected.items()):
        raise CacheSeedLaunchError("launch allowance binding mismatch")
    return dict(allowance)


def build_cache_seed_submission_intent(
    *,
    descriptor: Mapping[str, object],
    descriptor_file_sha256: str,
    approval: Mapping[str, object],
    approval_file_sha256: str,
    staged_readiness: Mapping[str, object],
    staged_readiness_file_sha256: str,
    staged_readiness_version_id: str,
    bundle_manifest: Mapping[str, object],
    bundle_manifest_file_sha256: str,
    rehearsal_evidence: Mapping[str, object],
    rehearsal_evidence_file_sha256: str,
    s3_spend_ledger_absence_observation_file_sha256: str,
    ec2_tagged_p5_zero_inventory_observation_file_sha256: str,
    launch_allowance: Mapping[str, object],
    launch_allowance_file_sha256: str,
    intent_at: datetime | str,
) -> dict[str, object]:
    descriptor_value, approval_value = _descriptor_approval(
        descriptor,
        descriptor_file_sha256,
        approval,
        approval_file_sha256,
    )
    staged_key, manifest_version = _validate_staged_chain(
        descriptor=descriptor_value,
        descriptor_file_sha256=descriptor_file_sha256,
        staged_readiness=staged_readiness,
        staged_readiness_file_sha256=staged_readiness_file_sha256,
        staged_readiness_version_id=staged_readiness_version_id,
        bundle_manifest=bundle_manifest,
        bundle_manifest_file_sha256=bundle_manifest_file_sha256,
        rehearsal_evidence=rehearsal_evidence,
        rehearsal_evidence_file_sha256=rehearsal_evidence_file_sha256,
    )
    s3_sha = _digest(
        s3_spend_ledger_absence_observation_file_sha256,
        field="S3 observation file SHA-256",
    )
    ec2_sha = _digest(
        ec2_tagged_p5_zero_inventory_observation_file_sha256,
        field="EC2 observation file SHA-256",
    )
    allowance = _validate_allowance(
        descriptor=descriptor_value,
        descriptor_file_sha256=descriptor_file_sha256,
        approval=approval_value,
        approval_file_sha256=approval_file_sha256,
        launch_allowance=launch_allowance,
        launch_allowance_file_sha256=launch_allowance_file_sha256,
        s3_observation_file_sha256=s3_sha,
        ec2_observation_file_sha256=ec2_sha,
    )
    intent_time, intent_iso = _canonical_time(intent_at, field="intent_at")
    allowance_time, allowance_iso = _canonical_time(
        allowance["observed_at"],
        field="allowance observed_at",
    )
    staged_time, _ = _canonical_time(
        staged_readiness["staged_at"],
        field="staged_at",
    )
    rehearsal_time, _ = _canonical_time(
        rehearsal_evidence["completed_at"],
        field="rehearsal completed_at",
    )
    not_after_time, _ = _canonical_time(
        allowance["not_after"],
        field="not_after",
    )
    must_start_time, _ = _canonical_time(
        allowance["must_start_by"],
        field="must_start_by",
    )
    if (
        intent_iso != allowance_iso
        or intent_time != allowance_time
        or not (staged_time <= rehearsal_time <= intent_time)
        or not (intent_time < not_after_time <= must_start_time)
    ):
        raise CacheSeedLaunchError("cache-seed intent time authority is inconsistent")
    run_id = str(descriptor_value["run_id"])
    rehearsal_body_sha = str(rehearsal_evidence["rehearsal_body_sha256"])
    rehearsal_key = (
        f"campaigns/{run_id}/qualification/rehearsals/"
        f"{rehearsal_body_sha}/GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
    )
    _safe_key(rehearsal_key, field="rehearsal evidence")
    body: dict[str, object] = {
        "schema_version": INTENT_SCHEMA_VERSION,
        "record_type": INTENT_RECORD_TYPE,
        "account_id": APPROVED_ACCOUNT_ID,
        "region": APPROVED_REGION,
        "run_id": run_id,
        "managed_mode": MANAGED_MODE,
        "sky_job_name": f"{run_id}-{MANAGED_MODE}",
        "descriptor_key": descriptor_value["campaign_descriptor_key"],
        "descriptor_file_sha256": descriptor_file_sha256,
        "descriptor_body_sha256": descriptor_value["descriptor_body_sha256"],
        "campaign_identity_sha256": descriptor_value["campaign_identity_sha256"],
        "approval_key": descriptor_value["approval_key"],
        "approval_sha256": approval_file_sha256,
        "approval_body_sha256": approval_value["approval_body_sha256"],
        "repo_tar_key": descriptor_value["repo_tar_key"],
        "repo_tar_sha256": descriptor_value["repo_tar_sha256"],
        "staged_readiness_key": staged_key,
        "staged_readiness_file_sha256": staged_readiness_file_sha256,
        "staged_readiness_body_sha256": staged_readiness["ready_body_sha256"],
        "staged_readiness_version_id": staged_readiness_version_id,
        "bundle_manifest_key": staged_readiness["bundle_manifest_key"],
        "bundle_manifest_file_sha256": bundle_manifest_file_sha256,
        "bundle_manifest_body_sha256": bundle_manifest["bundle_manifest_body_sha256"],
        "bundle_manifest_version_id": manifest_version,
        "rehearsal_evidence_key": rehearsal_key,
        "rehearsal_evidence_file_sha256": rehearsal_evidence_file_sha256,
        "rehearsal_evidence_body_sha256": rehearsal_body_sha,
        "s3_spend_ledger_absence_observation_key": (
            s3_spend_ledger_absence_observation_s3_key(
                run_id=run_id,
                observation_file_sha256=s3_sha,
            )
        ),
        "s3_spend_ledger_absence_observation_sha256": s3_sha,
        "ec2_tagged_p5_zero_inventory_observation_key": (
            ec2_tagged_p5_zero_inventory_observation_s3_key(
                run_id=run_id,
                observation_file_sha256=ec2_sha,
            )
        ),
        "ec2_tagged_p5_zero_inventory_observation_sha256": ec2_sha,
        "launch_allowance_key": gpu_launch_allowance_s3_key(
            run_id=run_id,
            allowance_body_sha256=str(allowance["allowance_body_sha256"]),
        ),
        "launch_allowance_file_sha256": launch_allowance_file_sha256,
        "launch_allowance_body_sha256": allowance["allowance_body_sha256"],
        "zero_cache_prefix": "qualification-cache/",
        "zero_cache_manifest_sha256": "0" * 64,
        "remaining_gpu_seconds": 86_400,
        "remaining_gpu_cost_usd": 1320.96,
        "cache_seed_allowance_seconds": 21_600,
        "cache_seed_allowance_cost_usd": 330.24,
        "open_allocation_count": 0,
        "must_start_by": allowance["must_start_by"],
        "not_after": allowance["not_after"],
        "intent_at": intent_iso,
    }
    return {
        **body,
        INTENT_DIGEST_FIELD: _sha(_canonical(body)),
    }


def validate_cache_seed_submission_intent(
    value: Mapping[str, object],
    *,
    descriptor: Mapping[str, object],
    descriptor_file_sha256: str,
    approval: Mapping[str, object],
    approval_file_sha256: str,
    staged_readiness: Mapping[str, object],
    staged_readiness_file_sha256: str,
    staged_readiness_version_id: str,
    bundle_manifest: Mapping[str, object],
    bundle_manifest_file_sha256: str,
    rehearsal_evidence: Mapping[str, object],
    rehearsal_evidence_file_sha256: str,
    s3_spend_ledger_absence_observation_file_sha256: str,
    ec2_tagged_p5_zero_inventory_observation_file_sha256: str,
    launch_allowance: Mapping[str, object],
    launch_allowance_file_sha256: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise CacheSeedLaunchError("cache-seed intent must be a mapping")
    _self_hash(
        value,
        fields=_INTENT_FIELDS,
        digest_field=INTENT_DIGEST_FIELD,
        label="cache-seed submission intent",
    )
    expected = build_cache_seed_submission_intent(
        descriptor=descriptor,
        descriptor_file_sha256=descriptor_file_sha256,
        approval=approval,
        approval_file_sha256=approval_file_sha256,
        staged_readiness=staged_readiness,
        staged_readiness_file_sha256=staged_readiness_file_sha256,
        staged_readiness_version_id=staged_readiness_version_id,
        bundle_manifest=bundle_manifest,
        bundle_manifest_file_sha256=bundle_manifest_file_sha256,
        rehearsal_evidence=rehearsal_evidence,
        rehearsal_evidence_file_sha256=rehearsal_evidence_file_sha256,
        s3_spend_ledger_absence_observation_file_sha256=(
            s3_spend_ledger_absence_observation_file_sha256
        ),
        ec2_tagged_p5_zero_inventory_observation_file_sha256=(
            ec2_tagged_p5_zero_inventory_observation_file_sha256
        ),
        launch_allowance=launch_allowance,
        launch_allowance_file_sha256=launch_allowance_file_sha256,
        intent_at=value.get("intent_at"),
    )
    if dict(value) != expected:
        raise CacheSeedLaunchError("cache-seed submission intent authority drift")
    return dict(value)


def _validate_intent_for_claim(
    *,
    descriptor: Mapping[str, object],
    descriptor_file_sha256: str,
    intent: Mapping[str, object],
    intent_file_sha256: str,
    launch_allowance: Mapping[str, object],
    launch_allowance_file_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    try:
        descriptor_value = validate_sky_campaign_descriptor(descriptor)
        allowance = validate_gpu_launch_allowance(launch_allowance)
    except (SkyCampaignValidationError, GpuLaunchAllowanceError, ValueError) as error:
        raise CacheSeedLaunchError(str(error)) from error
    _file_digest(
        descriptor_value,
        descriptor_file_sha256,
        field="descriptor_file_sha256",
    )
    _file_digest(intent, intent_file_sha256, field="intent_file_sha256")
    _file_digest(
        allowance,
        launch_allowance_file_sha256,
        field="launch_allowance_file_sha256",
    )
    _self_hash(
        intent,
        fields=_INTENT_FIELDS,
        digest_field=INTENT_DIGEST_FIELD,
        label="cache-seed submission intent",
    )
    run_id = _run_id(descriptor_value.get("run_id"))
    descriptor_key = _safe_key(
        descriptor_value.get("campaign_descriptor_key"),
        field="descriptor",
    )
    descriptor_prefix = f"campaigns/{run_id}/submissions/"
    descriptor_suffix = "/campaign-descriptor-v2.json"
    if not descriptor_key.startswith(descriptor_prefix) or not descriptor_key.endswith(
        descriptor_suffix
    ):
        raise CacheSeedLaunchError("descriptor submission coordinate is invalid")
    submission_id = descriptor_key[len(descriptor_prefix) : -len(descriptor_suffix)]
    if not submission_id or "/" in submission_id or submission_id in {".", ".."}:
        raise CacheSeedLaunchError("descriptor submission coordinate is invalid")
    staged_key = (
        f"campaigns/{run_id}/submissions/{submission_id}/"
        "STAGED_CONTROL_PLANE_READY.json"
    )
    manifest_body_sha = _digest(
        intent.get("bundle_manifest_body_sha256"),
        field="bundle_manifest_body_sha256",
    )
    manifest_key = (
        f"campaigns/{run_id}/submissions/{submission_id}/bundle-manifests/"
        f"{manifest_body_sha}/bundle-manifest-v1.json"
    )
    rehearsal_body_sha = _digest(
        intent.get("rehearsal_evidence_body_sha256"),
        field="rehearsal_evidence_body_sha256",
    )
    rehearsal_key = (
        f"campaigns/{run_id}/qualification/rehearsals/"
        f"{rehearsal_body_sha}/GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
    )
    s3_observation_sha = _digest(
        intent.get("s3_spend_ledger_absence_observation_sha256"),
        field="s3_spend_ledger_absence_observation_sha256",
    )
    ec2_observation_sha = _digest(
        intent.get("ec2_tagged_p5_zero_inventory_observation_sha256"),
        field="ec2_tagged_p5_zero_inventory_observation_sha256",
    )
    allowance_expected = {
        "account_id": APPROVED_ACCOUNT_ID,
        "region": APPROVED_REGION,
        "bucket": descriptor_value["bucket"],
        "run_id": run_id,
        "managed_mode": MANAGED_MODE,
        "sky_job_name": f"{run_id}-{MANAGED_MODE}",
        "descriptor_key": descriptor_key,
        "descriptor_sha256": descriptor_file_sha256,
        "descriptor_body_sha256": descriptor_value["descriptor_body_sha256"],
        "campaign_identity_sha256": descriptor_value["campaign_identity_sha256"],
        "approval_key": descriptor_value["approval_key"],
        "approval_sha256": descriptor_value["approval_sha256"],
        "remaining_gpu_seconds": APPROVED_GPU_RUNTIME_SECONDS,
        "remaining_gpu_cost_usd": float(APPROVED_GPU_COST_USD),
        "cache_seed_max_seconds": 21_600,
        "cache_seed_max_cost_usd": 330.24,
        "open_allocation_count": 0,
        "s3_spend_ledger_absence_observation_sha256": s3_observation_sha,
        "ec2_tagged_p5_zero_inventory_observation_sha256": ec2_observation_sha,
    }
    if any(
        allowance.get(field) != expected
        for field, expected in allowance_expected.items()
    ):
        raise CacheSeedLaunchError("allowance claim binding mismatch")
    approval_body_sha = _digest(
        allowance.get("approval_body_sha256"),
        field="allowance approval_body_sha256",
    )
    exact_intent = {
        "schema_version": INTENT_SCHEMA_VERSION,
        "record_type": INTENT_RECORD_TYPE,
        "account_id": APPROVED_ACCOUNT_ID,
        "region": APPROVED_REGION,
        "run_id": run_id,
        "managed_mode": MANAGED_MODE,
        "sky_job_name": f"{run_id}-{MANAGED_MODE}",
        "descriptor_key": descriptor_key,
        "descriptor_file_sha256": descriptor_file_sha256,
        "descriptor_body_sha256": descriptor_value["descriptor_body_sha256"],
        "campaign_identity_sha256": descriptor_value["campaign_identity_sha256"],
        "approval_key": descriptor_value["approval_key"],
        "approval_sha256": descriptor_value["approval_sha256"],
        "approval_body_sha256": approval_body_sha,
        "repo_tar_key": descriptor_value["repo_tar_key"],
        "repo_tar_sha256": descriptor_value["repo_tar_sha256"],
        "staged_readiness_key": staged_key,
        "bundle_manifest_key": manifest_key,
        "rehearsal_evidence_key": rehearsal_key,
        "s3_spend_ledger_absence_observation_key": (
            s3_spend_ledger_absence_observation_s3_key(
                run_id=run_id,
                observation_file_sha256=s3_observation_sha,
            )
        ),
        "ec2_tagged_p5_zero_inventory_observation_key": (
            ec2_tagged_p5_zero_inventory_observation_s3_key(
                run_id=run_id,
                observation_file_sha256=ec2_observation_sha,
            )
        ),
        "s3_spend_ledger_absence_observation_sha256": s3_observation_sha,
        "ec2_tagged_p5_zero_inventory_observation_sha256": ec2_observation_sha,
        "launch_allowance_file_sha256": launch_allowance_file_sha256,
        "launch_allowance_body_sha256": allowance["allowance_body_sha256"],
        "must_start_by": allowance["must_start_by"],
        "not_after": allowance["not_after"],
        "intent_at": allowance["observed_at"],
    }
    if any(
        type(intent.get(field)) is not type(expected) or intent.get(field) != expected
        for field, expected in exact_intent.items()
    ):
        raise CacheSeedLaunchError("intent claim binding mismatch")
    expected_allowance_key = gpu_launch_allowance_s3_key(
        run_id=run_id,
        allowance_body_sha256=str(allowance["allowance_body_sha256"]),
    )
    if intent.get("launch_allowance_key") != expected_allowance_key:
        raise CacheSeedLaunchError("intent allowance key binding mismatch")
    for field in _INTENT_FIELDS:
        if field.endswith("_sha256"):
            _digest(intent.get(field), field=field)
    for field in (
        "descriptor_key",
        "approval_key",
        "repo_tar_key",
        "staged_readiness_key",
        "bundle_manifest_key",
        "rehearsal_evidence_key",
        "s3_spend_ledger_absence_observation_key",
        "ec2_tagged_p5_zero_inventory_observation_key",
        "launch_allowance_key",
    ):
        _safe_key(intent.get(field), field=field)
    _version_id(
        intent.get("staged_readiness_version_id"),
        field="intent staged readiness VersionId",
    )
    _version_id(
        intent.get("bundle_manifest_version_id"),
        field="intent bundle manifest VersionId",
    )
    if (
        type(intent.get("remaining_gpu_seconds")) is not int
        or intent.get("remaining_gpu_seconds") != 86_400
        or type(intent.get("remaining_gpu_cost_usd")) is not float
        or not math.isfinite(float(intent["remaining_gpu_cost_usd"]))
        or intent.get("remaining_gpu_cost_usd") != 1320.96
        or type(intent.get("cache_seed_allowance_seconds")) is not int
        or intent.get("cache_seed_allowance_seconds") != 21_600
        or type(intent.get("cache_seed_allowance_cost_usd")) is not float
        or intent.get("cache_seed_allowance_cost_usd") != 330.24
        or type(intent.get("open_allocation_count")) is not int
        or intent.get("open_allocation_count") != 0
        or intent.get("zero_cache_prefix") != "qualification-cache/"
        or intent.get("zero_cache_manifest_sha256") != "0" * 64
    ):
        raise CacheSeedLaunchError("intent exact allocation authority mismatch")
    return dict(descriptor_value), dict(allowance)


def build_cache_seed_launch_claim(
    *,
    descriptor: Mapping[str, object],
    descriptor_file_sha256: str,
    intent: Mapping[str, object],
    intent_file_sha256: str,
    launch_allowance: Mapping[str, object],
    launch_allowance_file_sha256: str,
    claimed_at: datetime | str,
) -> dict[str, object]:
    descriptor_value, allowance = _validate_intent_for_claim(
        descriptor=descriptor,
        descriptor_file_sha256=descriptor_file_sha256,
        intent=intent,
        intent_file_sha256=intent_file_sha256,
        launch_allowance=launch_allowance,
        launch_allowance_file_sha256=launch_allowance_file_sha256,
    )
    claimed_time, claimed_iso = _canonical_time(claimed_at, field="claimed_at")
    observed_time, _ = _canonical_time(
        allowance["observed_at"],
        field="allowance observed_at",
    )
    not_after_time, _ = _canonical_time(
        allowance["not_after"],
        field="allowance not_after",
    )
    must_start_time, _ = _canonical_time(
        allowance["must_start_by"],
        field="allowance must_start_by",
    )
    if not observed_time <= claimed_time < not_after_time <= must_start_time:
        raise CacheSeedLaunchError("launch claim time authority is inconsistent")
    run_id = str(descriptor_value["run_id"])
    copied_fields = (
        "managed_mode",
        "sky_job_name",
        "descriptor_key",
        "descriptor_file_sha256",
        "descriptor_body_sha256",
        "campaign_identity_sha256",
        "approval_key",
        "approval_sha256",
        "approval_body_sha256",
        "staged_readiness_key",
        "staged_readiness_file_sha256",
        "staged_readiness_body_sha256",
        "staged_readiness_version_id",
        "bundle_manifest_key",
        "bundle_manifest_file_sha256",
        "bundle_manifest_body_sha256",
        "bundle_manifest_version_id",
        "rehearsal_evidence_key",
        "rehearsal_evidence_file_sha256",
        "rehearsal_evidence_body_sha256",
        "s3_spend_ledger_absence_observation_key",
        "s3_spend_ledger_absence_observation_sha256",
        "ec2_tagged_p5_zero_inventory_observation_key",
        "ec2_tagged_p5_zero_inventory_observation_sha256",
        "launch_allowance_key",
        "launch_allowance_file_sha256",
        "launch_allowance_body_sha256",
    )
    body: dict[str, object] = {
        "schema_version": CLAIM_SCHEMA_VERSION,
        "record_type": CLAIM_RECORD_TYPE,
        "account_id": APPROVED_ACCOUNT_ID,
        "region": APPROVED_REGION,
        "bucket": descriptor_value["bucket"],
        "run_id": run_id,
        **{field: intent[field] for field in copied_fields},
        "submission_intent_key": cache_seed_submission_intent_s3_key(
            run_id=run_id,
            intent_body_sha256=str(intent[INTENT_DIGEST_FIELD]),
        ),
        "submission_intent_file_sha256": intent_file_sha256,
        "submission_intent_body_sha256": intent[INTENT_DIGEST_FIELD],
        "claimed_at": claimed_iso,
        "must_start_by": allowance["must_start_by"],
        "not_after": allowance["not_after"],
    }
    return {
        **body,
        CLAIM_DIGEST_FIELD: _sha(_canonical(body)),
    }


def validate_cache_seed_launch_claim(
    value: Mapping[str, object],
    *,
    descriptor: Mapping[str, object],
    descriptor_file_sha256: str,
    intent: Mapping[str, object],
    intent_file_sha256: str,
    launch_allowance: Mapping[str, object],
    launch_allowance_file_sha256: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise CacheSeedLaunchError("cache-seed launch claim must be a mapping")
    _self_hash(
        value,
        fields=_CLAIM_FIELDS,
        digest_field=CLAIM_DIGEST_FIELD,
        label="cache-seed launch claim",
    )
    expected = build_cache_seed_launch_claim(
        descriptor=descriptor,
        descriptor_file_sha256=descriptor_file_sha256,
        intent=intent,
        intent_file_sha256=intent_file_sha256,
        launch_allowance=launch_allowance,
        launch_allowance_file_sha256=launch_allowance_file_sha256,
        claimed_at=value.get("claimed_at"),
    )
    if dict(value) != expected:
        raise CacheSeedLaunchError("cache-seed launch claim authority drift")
    return dict(value)


def _prevalidate_cache_seed_launch_claim_references(
    value: Mapping[str, object],
    *,
    descriptor: Mapping[str, object],
    descriptor_file_sha256: str,
    claim_key: str,
) -> dict[str, str]:
    """Authenticate a durable claim and every derived key before reference I/O."""

    if not isinstance(value, Mapping):
        raise CacheSeedLaunchError("cache-seed launch claim must be a mapping")
    _self_hash(
        value,
        fields=_CLAIM_FIELDS,
        digest_field=CLAIM_DIGEST_FIELD,
        label="cache-seed launch claim",
    )
    try:
        descriptor_value = validate_sky_campaign_descriptor(descriptor)
    except (SkyCampaignValidationError, ValueError) as error:
        raise CacheSeedLaunchError(str(error)) from error
    _file_digest(
        descriptor_value,
        descriptor_file_sha256,
        field="descriptor_file_sha256",
    )
    run_id = _run_id(descriptor_value.get("run_id"))
    expected_claim_key = cache_seed_launch_claim_s3_key(
        run_id=run_id,
        descriptor_file_sha256=descriptor_file_sha256,
    )
    if claim_key != expected_claim_key:
        raise CacheSeedLaunchError("launch claim key is not descriptor-addressed")
    exact = {
        "schema_version": CLAIM_SCHEMA_VERSION,
        "record_type": CLAIM_RECORD_TYPE,
        "account_id": APPROVED_ACCOUNT_ID,
        "region": APPROVED_REGION,
        "bucket": descriptor_value["bucket"],
        "run_id": run_id,
        "managed_mode": MANAGED_MODE,
        "sky_job_name": f"{run_id}-{MANAGED_MODE}",
        "descriptor_key": descriptor_value["campaign_descriptor_key"],
        "descriptor_file_sha256": descriptor_file_sha256,
        "descriptor_body_sha256": descriptor_value["descriptor_body_sha256"],
        "campaign_identity_sha256": descriptor_value["campaign_identity_sha256"],
        "approval_key": descriptor_value["approval_key"],
        "approval_sha256": descriptor_value["approval_sha256"],
    }
    if any(
        type(value.get(field)) is not type(expected) or value.get(field) != expected
        for field, expected in exact.items()
    ):
        raise CacheSeedLaunchError("launch claim descriptor authority drift")
    for field in _CLAIM_FIELDS:
        if field.endswith("_sha256"):
            _digest(value.get(field), field=field)
    _version_id(
        value.get("staged_readiness_version_id"),
        field="claim staged readiness VersionId",
    )
    _version_id(
        value.get("bundle_manifest_version_id"),
        field="claim bundle manifest VersionId",
    )
    descriptor_key = str(descriptor_value["campaign_descriptor_key"])
    submission_prefix = descriptor_key.removesuffix("campaign-descriptor-v2.json")
    references = {
        "submission_intent_key": cache_seed_submission_intent_s3_key(
            run_id=run_id,
            intent_body_sha256=str(value["submission_intent_body_sha256"]),
        ),
        "launch_allowance_key": gpu_launch_allowance_s3_key(
            run_id=run_id,
            allowance_body_sha256=str(value["launch_allowance_body_sha256"]),
        ),
        "s3_spend_ledger_absence_observation_key": (
            s3_spend_ledger_absence_observation_s3_key(
                run_id=run_id,
                observation_file_sha256=str(
                    value["s3_spend_ledger_absence_observation_sha256"]
                ),
            )
        ),
        "ec2_tagged_p5_zero_inventory_observation_key": (
            ec2_tagged_p5_zero_inventory_observation_s3_key(
                run_id=run_id,
                observation_file_sha256=str(
                    value["ec2_tagged_p5_zero_inventory_observation_sha256"]
                ),
            )
        ),
        "rehearsal_evidence_key": (
            f"campaigns/{run_id}/qualification/rehearsals/"
            f"{value['rehearsal_evidence_body_sha256']}/"
            "GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
        ),
        "staged_readiness_key": (f"{submission_prefix}STAGED_CONTROL_PLANE_READY.json"),
        "bundle_manifest_key": (
            f"{submission_prefix}bundle-manifests/"
            f"{value['bundle_manifest_body_sha256']}/bundle-manifest-v1.json"
        ),
    }
    for field, expected in references.items():
        _safe_key(expected, field=field)
        if value.get(field) != expected:
            raise CacheSeedLaunchError(f"launch claim {field} is not derived")
    claimed, _ = _canonical_time(value.get("claimed_at"), field="claimed_at")
    not_after, _ = _canonical_time(value.get("not_after"), field="not_after")
    must_start, _ = _canonical_time(
        value.get("must_start_by"),
        field="must_start_by",
    )
    if not claimed < not_after <= must_start:
        raise CacheSeedLaunchError("launch claim time authority is inconsistent")
    return {
        field: expected
        for field, expected in references.items()
        if field
        in {
            "submission_intent_key",
            "launch_allowance_key",
            "s3_spend_ledger_absence_observation_key",
            "ec2_tagged_p5_zero_inventory_observation_key",
            "rehearsal_evidence_key",
        }
    }


__all__ = [
    "CLAIM_DIGEST_FIELD",
    "CLAIM_RECORD_TYPE",
    "CLAIM_SCHEMA_VERSION",
    "INTENT_DIGEST_FIELD",
    "INTENT_RECORD_TYPE",
    "INTENT_SCHEMA_VERSION",
    "MANAGED_MODE",
    "MAX_LIVE_AGE_SECONDS",
    "CacheSeedLaunchError",
    "build_cache_seed_launch_claim",
    "build_cache_seed_submission_intent",
    "cache_seed_launch_claim_s3_key",
    "cache_seed_submission_intent_s3_key",
    "canonical_file_bytes",
    "ec2_tagged_p5_zero_inventory_observation_s3_key",
    "s3_spend_ledger_absence_observation_s3_key",
    "validate_cache_seed_launch_claim",
    "validate_cache_seed_submission_intent",
]
