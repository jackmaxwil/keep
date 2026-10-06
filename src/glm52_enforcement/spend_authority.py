"""Fresh authenticated GPU-spend authority and residual reserve contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
from typing import Mapping, Optional, Tuple

from .approvals import validate_gpu_residual_liability_approval


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
APPROVED_GPU_RUNTIME_SECONDS = 86_400
APPROVED_GPU_COST_USD = Decimal("1320.96")
APPROVED_HOURLY_COST_USD = Decimal("55.04")
GPU_RESERVE_SECONDS = 900
GPU_RESERVE_COST_USD = Decimal("13.76")
ROOT_VOLUME_GIB = 300
ROOT_VOLUME_TAIL_USD_MAX = Decimal("0.01")

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_INSTANCE_ID = re.compile(r"i-(?:[0-9a-f]{8}|[0-9a-f]{17})\Z")
_RECORD_NAME = re.compile(
    r"(?P<index>[0-9]{6})-"
    r"(?P<event>allocation_started|allocation_ended)-"
    r"(?P<sha>[0-9a-f]{64})[.]json\Z"
)
_APPROVAL_REQUEST = """Hey Kon — following your SkyPilot suggestion, I’m planning to run the GLM-5.2 teacher-cache and adapter-training campaign as an AWS-only SkyPilot Managed Job instead of buying another Capacity Block.

Current verified `p5.48xlarge` on-demand pricing in Oregon is **$55.04/hour**:
- 24-hour hard cap: **$1,320.96**
- 36-hour hard cap: **$1,981.44**
- 48-hour hard cap: **$2,641.92**

My recommendation is the **24-hour cap**. The campaign will checkpoint to S3 and stop cleanly at the cap; if it is incomplete, we can resume under a separately approved budget extension. There will be no paid GPU reservation while AWS has no capacity, though the small SkyPilot controller and S3 charges can continue while queued. Spot will remain disabled until interruption/resume has passed on a real H100.

Are you okay with me proceeding with the 24-hour / $1,320.96 GPU cap, or would you approve a different cap?"""


class SpendAuthorityError(ValueError):
    """Fresh spend evidence is incomplete, divergent, or stale."""


class AmbiguousReserveTransport(RuntimeError):
    """The sole conditional reserve write has an ambiguous transport result."""


class ReserveRejected(RuntimeError):
    """The sole conditional reserve write was definitely rejected."""


def _decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        raise SpendAuthorityError("non-finite decimal is forbidden")
    text = format(value, "f")
    if text.startswith("+"):
        text = text[1:]
    if text == "-0":
        text = "0"
    return text


def canonical_decimal_json_bytes(value: object) -> bytes:
    """Canonical JSON that preserves decimal arithmetic without binary floats."""

    def encode(item: object) -> str:
        if item is None:
            return "null"
        if item is True:
            return "true"
        if item is False:
            return "false"
        if type(item) is int:
            return str(item)
        if isinstance(item, Decimal):
            return _decimal_text(item)
        if type(item) is float:
            raise SpendAuthorityError("binary floating point is forbidden")
        if type(item) is str:
            return json.dumps(item, ensure_ascii=True)
        if type(item) in (list, tuple):
            return "[" + ",".join(encode(member) for member in item) + "]"
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise SpendAuthorityError("canonical JSON object keys must be strings")
            return (
                "{"
                + ",".join(
                    json.dumps(key, ensure_ascii=True) + ":" + encode(item[key])
                    for key in sorted(item)
                )
                + "}"
            )
        raise SpendAuthorityError(
            f"unsupported canonical JSON value: {type(item).__name__}"
        )

    return encode(value).encode("ascii")


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_decimal_json_bytes(value)).hexdigest()


def _sha_raw(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(value: object, *, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise SpendAuthorityError(f"{field} must be a lowercase SHA-256")
    return value


def _parse_time(value: object, *, field: str) -> datetime:
    if type(value) is not str:
        raise SpendAuthorityError(f"{field} must be canonical UTC")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, OverflowError) as exc:
        raise SpendAuthorityError(f"{field} must be canonical UTC") from exc
    if parsed.tzinfo is None:
        raise SpendAuthorityError(f"{field} must be timezone-aware")
    canonical = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if canonical != value:
        raise SpendAuthorityError(f"{field} must be canonical UTC")
    return parsed.astimezone(timezone.utc)


def _money(value: object, *, field: str) -> Decimal:
    if isinstance(value, bool) or isinstance(value, float):
        raise SpendAuthorityError(f"{field} must use exact decimal arithmetic")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise SpendAuthorityError(f"{field} must be an exact decimal") from exc
    if (
        not result.is_finite()
        or result < 0
        or result != result.quantize(Decimal("0.01"))
    ):
        raise SpendAuthorityError(
            f"{field} must be a nonnegative two-decimal amount"
        )
    return result


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SpendAuthorityError(f"{field} must be an integer >= {minimum}")
    return value


def _object(raw: bytes, *, label: str) -> dict[str, object]:
    if type(raw) is not bytes:
        raise SpendAuthorityError(f"{label} must be exact bytes")
    try:
        value = json.loads(raw, parse_float=Decimal)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpendAuthorityError(f"{label} is malformed JSON") from exc
    if type(value) is not dict:
        raise SpendAuthorityError(f"{label} must contain one object")
    if raw != canonical_decimal_json_bytes(value) + b"\n":
        raise SpendAuthorityError(f"{label} bytes are not canonical")
    return value


_APPROVAL_FIELDS = {
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
    "approval_body_sha256",
}


def build_gpu_spend_approval(
    *, ingested_at: str, slack_permalink: Optional[str]
) -> dict[str, object]:
    """Build the already accepted immutable Kon GPU approval schema."""

    _parse_time(ingested_at, field="approval_ingested_at")
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_gpu_spend_approval_v1",
        "approved_by": "Alex Approver",
        "approval_request": _APPROVAL_REQUEST,
        "approval_response": "yep",
        "approval_displayed_time": "5:59 PM",
        "approval_ingested_at": ingested_at,
        "evidence_source": "user-supplied-slack-exchange",
        "slack_permalink": slack_permalink,
        "approved_hourly_usd": APPROVED_HOURLY_COST_USD,
        "approved_gpu_hours": 24,
        "approved_gpu_cost_usd": APPROVED_GPU_COST_USD,
        "instance_type": "p5.48xlarge",
        "region": REGION,
        "includes_qualification": True,
        "includes_recovery_instances": True,
    }
    return {**body, "approval_body_sha256": _sha(body)}


def validate_gpu_spend_approval(value: object) -> dict[str, object]:
    """Authenticate the exact accepted GPU approval without ML imports."""

    if type(value) is not dict or set(value) != _APPROVAL_FIELDS:
        raise SpendAuthorityError("GPU spend approval schema mismatch")
    body = dict(value)
    identity = body.pop("approval_body_sha256")
    if identity != _sha(body):
        raise SpendAuthorityError("GPU spend approval body SHA-256 mismatch")
    expected = build_gpu_spend_approval(
        ingested_at=str(value["approval_ingested_at"]),
        slack_permalink=value["slack_permalink"],
    )
    if value != expected:
        raise SpendAuthorityError("GPU spend approval differs from accepted authority")
    permalink = value["slack_permalink"]
    if permalink is not None and (
        type(permalink) is not str
        or not permalink.startswith("https://")
        or "slack.com/" not in permalink
    ):
        raise SpendAuthorityError("GPU approval Slack permalink is invalid")
    return dict(value)


@dataclass(frozen=True)
class SpendObject:
    key: str
    raw: bytes
    version_id: str
    etag: str
    checksum_sha256: str
    request_id: str
    observed_at: str


@dataclass(frozen=True)
class SpendListPage:
    keys: Tuple[str, ...]
    next_token: Optional[str]
    request_id: str
    observed_at: str
    version_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SpendAuthorityRequest:
    account_id: str
    region: str
    run_id: str
    descriptor_key: str
    descriptor_file_sha256: str
    descriptor_version_id: str
    approval_key: str
    approval_file_sha256: str
    approval_version_id: str
    latest_key: str
    latest_file_sha256: str
    latest_version_id: str
    snapshot_key: str
    snapshot_file_sha256: str
    snapshot_version_id: str
    ledger_records_prefix: str
    expected_snapshot_body_sha256: str
    observed_at: str
    snapshot_observed_at: Optional[str] = None


@dataclass(frozen=True)
class SpendAuthorityServices:
    object_store: object
    ec2: object
    reserve_reader: object = None


@dataclass(frozen=True)
class ReserveListPage:
    records: Tuple[Mapping[str, object], ...]
    next_token: Optional[str]
    request_id: str
    observed_at: str


def spend_authority_request_from_mapping(
    value: Mapping[str, object],
) -> SpendAuthorityRequest:
    """Parse the closed production spend-read request from authenticated JSON."""

    required_fields = {
        "account_id",
        "region",
        "run_id",
        "descriptor_key",
        "descriptor_file_sha256",
        "descriptor_version_id",
        "approval_key",
        "approval_file_sha256",
        "approval_version_id",
        "latest_key",
        "latest_file_sha256",
        "latest_version_id",
        "snapshot_key",
        "snapshot_file_sha256",
        "snapshot_version_id",
        "ledger_records_prefix",
        "expected_snapshot_body_sha256",
        "observed_at",
    }
    allowed_fields = required_fields | {"snapshot_observed_at"}
    if (
        type(value) is not dict
        or not required_fields.issubset(value)
        or not set(value).issubset(allowed_fields)
    ):
        raise SpendAuthorityError("spend authority request schema mismatch")
    request = SpendAuthorityRequest(
        **{
            field: str(value[field])
            for field in required_fields
        },
        snapshot_observed_at=(
            None
            if value.get("snapshot_observed_at") is None
            else str(value["snapshot_observed_at"])
        ),
    )
    if (
        request.account_id != ACCOUNT_ID
        or request.region != REGION
        or request.run_id != RUN_ID
    ):
        raise SpendAuthorityError("spend authority request is foreign")
    for field in (
        "descriptor_file_sha256",
        "approval_file_sha256",
        "latest_file_sha256",
        "snapshot_file_sha256",
        "expected_snapshot_body_sha256",
    ):
        _digest(getattr(request, field), field=field)
    for field in (
        "descriptor_key",
        "descriptor_version_id",
        "approval_key",
        "approval_version_id",
        "latest_key",
        "latest_version_id",
        "snapshot_key",
        "snapshot_version_id",
        "ledger_records_prefix",
    ):
        if not getattr(request, field):
            raise SpendAuthorityError(f"{field} must be nonempty")
    _parse_time(request.observed_at, field="observed_at")
    if request.snapshot_observed_at is not None:
        snapshot_time = _parse_time(
            request.snapshot_observed_at,
            field="snapshot_observed_at",
        )
        observation = _parse_time(
            request.observed_at,
            field="observed_at",
        )
        if (
            snapshot_time > observation
            or observation - snapshot_time > timedelta(seconds=60)
        ):
            raise SpendAuthorityError(
                "snapshot observation is outside the live authority window"
            )
    return request


@dataclass(frozen=True)
class AllocationInterval:
    instance_id: str
    job_id: str
    started_at: str
    ended_at: Optional[str]
    charged_seconds: int
    charged_cost_usd: Decimal
    state: str


@dataclass(frozen=True)
class SpendAuthorityResult:
    account_id: str
    region: str
    run_id: str
    approval_identity_sha256: str
    descriptor_identity_sha256: str
    snapshot_identity_sha256: str
    ledger_genesis_identity_sha256: str
    ledger_tip_identity_sha256: str
    ledger_record_count: int
    used_gpu_seconds: int
    open_gpu_seconds: int
    reserved_gpu_seconds: int
    remaining_gpu_seconds: int
    refundable_gpu_seconds: int
    used_gpu_cost_usd: Decimal
    open_gpu_cost_usd: Decimal
    reserved_gpu_cost_usd: Decimal
    remaining_gpu_cost_usd: Decimal
    refundable_gpu_cost_usd: Decimal
    allocation_intervals: Tuple[AllocationInterval, ...]
    observed_at: str
    canonical_identity_sha256: str


def _result_body(result: SpendAuthorityResult) -> dict[str, object]:
    body = asdict(result)
    body.pop("canonical_identity_sha256")
    body["allocation_intervals"] = tuple(
        asdict(interval) for interval in result.allocation_intervals
    )
    return body


def validate_spend_authority_result(value: object) -> SpendAuthorityResult:
    if not isinstance(value, SpendAuthorityResult):
        raise SpendAuthorityError("spend authority result is not typed")
    if (
        value.account_id != ACCOUNT_ID
        or value.region != REGION
        or value.run_id != RUN_ID
    ):
        raise SpendAuthorityError("spend authority result is foreign")
    for field in (
        "approval_identity_sha256",
        "descriptor_identity_sha256",
        "snapshot_identity_sha256",
        "ledger_genesis_identity_sha256",
        "ledger_tip_identity_sha256",
    ):
        _digest(getattr(value, field), field=field)
    for field in (
        "used_gpu_seconds",
        "open_gpu_seconds",
        "reserved_gpu_seconds",
        "remaining_gpu_seconds",
        "refundable_gpu_seconds",
    ):
        _integer(getattr(value, field), field=field)
    for field in (
        "used_gpu_cost_usd",
        "open_gpu_cost_usd",
        "reserved_gpu_cost_usd",
        "remaining_gpu_cost_usd",
        "refundable_gpu_cost_usd",
    ):
        _money(getattr(value, field), field=field)
    if (
        value.used_gpu_seconds
        + value.open_gpu_seconds
        + value.reserved_gpu_seconds
        + value.remaining_gpu_seconds
        != APPROVED_GPU_RUNTIME_SECONDS
        or value.used_gpu_cost_usd
        + value.open_gpu_cost_usd
        + value.reserved_gpu_cost_usd
        + value.remaining_gpu_cost_usd
        != APPROVED_GPU_COST_USD
    ):
        raise SpendAuthorityError("spend authority envelope accounting diverges")
    _parse_time(value.observed_at, field="observed_at")
    if value.canonical_identity_sha256 != _sha(_result_body(value)):
        raise SpendAuthorityError("spend authority result identity mismatch")
    return value


def spend_authority_result_from_mapping(
    value: Mapping[str, object],
) -> SpendAuthorityResult:
    """Materialize a typed result at an injected authority boundary."""

    expected = {
        "account_id",
        "region",
        "run_id",
        "approval_identity_sha256",
        "descriptor_identity_sha256",
        "snapshot_identity_sha256",
        "ledger_genesis_identity_sha256",
        "ledger_tip_identity_sha256",
        "ledger_record_count",
        "used_gpu_seconds",
        "open_gpu_seconds",
        "reserved_gpu_seconds",
        "remaining_gpu_seconds",
        "refundable_gpu_seconds",
        "used_gpu_cost_usd",
        "open_gpu_cost_usd",
        "reserved_gpu_cost_usd",
        "remaining_gpu_cost_usd",
        "refundable_gpu_cost_usd",
        "allocation_intervals",
        "observed_at",
    }
    if set(value) != expected:
        raise SpendAuthorityError("spend authority result schema mismatch")
    intervals_raw = value["allocation_intervals"]
    if type(intervals_raw) not in (list, tuple):
        raise SpendAuthorityError("allocation intervals must be an array")
    intervals = tuple(
        AllocationInterval(
            instance_id=str(item["instance_id"]),
            job_id=str(item["job_id"]),
            started_at=str(item["started_at"]),
            ended_at=(
                None if item["ended_at"] is None else str(item["ended_at"])
            ),
            charged_seconds=_integer(
                item["charged_seconds"], field="charged_seconds"
            ),
            charged_cost_usd=_money(
                item["charged_cost_usd"], field="charged_cost_usd"
            ),
            state=str(item["state"]),
        )
        for item in intervals_raw
        if type(item) is dict
    )
    if len(intervals) != len(intervals_raw):
        raise SpendAuthorityError("allocation interval schema mismatch")
    provisional = SpendAuthorityResult(
        account_id=str(value["account_id"]),
        region=str(value["region"]),
        run_id=str(value["run_id"]),
        approval_identity_sha256=str(value["approval_identity_sha256"]),
        descriptor_identity_sha256=str(value["descriptor_identity_sha256"]),
        snapshot_identity_sha256=str(value["snapshot_identity_sha256"]),
        ledger_genesis_identity_sha256=str(
            value["ledger_genesis_identity_sha256"]
        ),
        ledger_tip_identity_sha256=str(value["ledger_tip_identity_sha256"]),
        ledger_record_count=_integer(
            value["ledger_record_count"], field="ledger_record_count"
        ),
        used_gpu_seconds=_integer(
            value["used_gpu_seconds"], field="used_gpu_seconds"
        ),
        open_gpu_seconds=_integer(
            value["open_gpu_seconds"], field="open_gpu_seconds"
        ),
        reserved_gpu_seconds=_integer(
            value["reserved_gpu_seconds"], field="reserved_gpu_seconds"
        ),
        remaining_gpu_seconds=_integer(
            value["remaining_gpu_seconds"], field="remaining_gpu_seconds"
        ),
        refundable_gpu_seconds=_integer(
            value["refundable_gpu_seconds"], field="refundable_gpu_seconds"
        ),
        used_gpu_cost_usd=_money(
            value["used_gpu_cost_usd"], field="used_gpu_cost_usd"
        ),
        open_gpu_cost_usd=_money(
            value["open_gpu_cost_usd"], field="open_gpu_cost_usd"
        ),
        reserved_gpu_cost_usd=_money(
            value["reserved_gpu_cost_usd"], field="reserved_gpu_cost_usd"
        ),
        remaining_gpu_cost_usd=_money(
            value["remaining_gpu_cost_usd"], field="remaining_gpu_cost_usd"
        ),
        refundable_gpu_cost_usd=_money(
            value["refundable_gpu_cost_usd"], field="refundable_gpu_cost_usd"
        ),
        allocation_intervals=intervals,
        observed_at=str(value["observed_at"]),
        canonical_identity_sha256="",
    )
    result = SpendAuthorityResult(
        **{
            **asdict(provisional),
            "allocation_intervals": intervals,
            "canonical_identity_sha256": _sha(_result_body(provisional)),
        }
    )
    return validate_spend_authority_result(result)


def _get_authenticated(
    services: SpendAuthorityServices,
    *,
    key: str,
    expected_sha256: Optional[str],
    expected_version_id: Optional[str] = None,
    label: str,
) -> tuple[dict[str, object], SpendObject]:
    method = getattr(services.object_store, "get_object", None)
    if not callable(method):
        raise SpendAuthorityError("spend object-store boundary is missing")
    observed = method(key=key)
    if not isinstance(observed, SpendObject) or observed.key != key:
        raise SpendAuthorityError(f"{label} returned the wrong object")
    if expected_sha256 is not None:
        _digest(expected_sha256, field=f"{label} expected file SHA-256")
        if _sha_raw(observed.raw) != expected_sha256:
            raise SpendAuthorityError(f"{label} exact-byte SHA-256 mismatch")
    if (
        type(observed.version_id) is not str
        or not observed.version_id
        or (
            expected_version_id is not None
            and observed.version_id != expected_version_id
        )
        or type(observed.etag) is not str
        or not observed.etag
        or observed.checksum_sha256 != _sha_raw(observed.raw)
        or type(observed.request_id) is not str
        or not observed.request_id
    ):
        raise SpendAuthorityError(f"{label} response metadata is incomplete")
    _parse_time(observed.observed_at, field=f"{label} observed_at")
    return _object(observed.raw, label=label), observed


def _list_record_keys(
    request: SpendAuthorityRequest, services: SpendAuthorityServices
) -> tuple[tuple[str, str], ...]:
    method = getattr(services.object_store, "list_namespace", None)
    if not callable(method):
        raise SpendAuthorityError("spend namespace list boundary is missing")
    token: Optional[str] = None
    seen_tokens: set[str] = set()
    seen_keys: set[str] = set()
    keys: list[tuple[str, str]] = []
    while True:
        page = method(
            prefix=request.ledger_records_prefix,
            continuation_token=token,
        )
        if not isinstance(page, SpendListPage):
            raise SpendAuthorityError("spend namespace page is not typed")
        if (
            type(page.request_id) is not str
            or not page.request_id
            or type(page.keys) is not tuple
        ):
            raise SpendAuthorityError("spend namespace page metadata is incomplete")
        _parse_time(page.observed_at, field="spend list observed_at")
        if len(page.version_ids) != len(page.keys):
            raise SpendAuthorityError(
                "spend namespace page lacks exact object versions"
            )
        for key, version_id in zip(page.keys, page.version_ids):
            if (
                type(key) is not str
                or not key.startswith(request.ledger_records_prefix)
                or key in seen_keys
                or type(version_id) is not str
                or not version_id
            ):
                raise SpendAuthorityError(
                    "spend namespace contains a duplicate or foreign key"
                )
            seen_keys.add(key)
            keys.append((key, version_id))
        if page.next_token is None:
            return tuple(keys)
        if (
            type(page.next_token) is not str
            or not page.next_token
            or page.next_token in seen_tokens
        ):
            raise SpendAuthorityError("spend namespace pagination token repeated")
        seen_tokens.add(page.next_token)
        token = page.next_token


def _validate_descriptor(
    value: dict[str, object],
    *,
    raw_sha256: str,
    approval_sha256: str,
) -> dict[str, object]:
    required = {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "campaign_identity_sha256",
        "approval_sha256",
        "approved_gpu_runtime_seconds",
        "approved_gpu_cost_usd",
        "max_hourly_cost_usd",
        "descriptor_body_sha256",
    }
    if not required.issubset(value):
        raise SpendAuthorityError("campaign descriptor lacks spend authority")
    body = dict(value)
    identity = body.pop("descriptor_body_sha256")
    if identity != _sha(body):
        raise SpendAuthorityError("campaign descriptor body SHA-256 mismatch")
    if (
        value["schema_version"] != 2
        or value["record_type"] != "glm52_sky_campaign_descriptor_v2"
        or value["account_id"] != ACCOUNT_ID
        or value["region"] != REGION
        or value["run_id"] != RUN_ID
        or value["approval_sha256"] != approval_sha256
        or value["approved_gpu_runtime_seconds"]
        != APPROVED_GPU_RUNTIME_SECONDS
        or _money(
            value["approved_gpu_cost_usd"], field="approved_gpu_cost_usd"
        )
        != APPROVED_GPU_COST_USD
        or _money(value["max_hourly_cost_usd"], field="max_hourly_cost_usd")
        != APPROVED_HOURLY_COST_USD
    ):
        raise SpendAuthorityError("campaign descriptor spend authority diverges")
    _digest(value["campaign_identity_sha256"], field="campaign identity")
    _digest(raw_sha256, field="descriptor file SHA-256")
    return value


def _ledger_genesis(approval_sha256: str) -> str:
    return _sha(
        {
            "record_type": "glm52_gpu_spend_ledger_genesis_v1",
            "run_id": RUN_ID,
            "approval_sha256": approval_sha256,
            "approved_gpu_runtime_seconds": APPROVED_GPU_RUNTIME_SECONDS,
            "approved_gpu_cost_usd": APPROVED_GPU_COST_USD,
            "hourly_cost_usd": APPROVED_HOURLY_COST_USD,
        }
    )


def _cost(seconds: int) -> Decimal:
    return (
        Decimal(seconds) * APPROVED_HOURLY_COST_USD / Decimal(3600)
    ).quantize(Decimal("0.01"))


def _inspect_ec2(
    services: SpendAuthorityServices,
    *,
    allocations: Tuple[AllocationInterval, ...],
) -> None:
    method = getattr(services.ec2, "describe_allocation_history", None)
    if not callable(method):
        raise SpendAuthorityError("EC2 allocation-history boundary is missing")
    token: Optional[str] = None
    seen_tokens: set[str] = set()
    by_id: dict[str, dict[str, object]] = {}
    while True:
        page = method(run_id=RUN_ID, next_token=token)
        if type(page) is not dict or set(page) != {
            "instances",
            "next_token",
            "request_id",
            "observed_at",
        }:
            raise SpendAuthorityError("EC2 allocation-history page schema mismatch")
        if type(page["request_id"]) is not str or not page["request_id"]:
            raise SpendAuthorityError("EC2 allocation-history request ID is absent")
        _parse_time(page["observed_at"], field="EC2 page observed_at")
        instances = page["instances"]
        if type(instances) is not list:
            raise SpendAuthorityError("EC2 allocation history is not an array")
        for instance in instances:
            if type(instance) is not dict:
                raise SpendAuthorityError("EC2 allocation item is malformed")
            instance_id = instance.get("InstanceId")
            if (
                type(instance_id) is not str
                or _INSTANCE_ID.fullmatch(instance_id) is None
                or instance_id in by_id
            ):
                raise SpendAuthorityError("EC2 allocation identity is duplicated")
            by_id[instance_id] = instance
        next_token = page["next_token"]
        if next_token is None:
            break
        if (
            type(next_token) is not str
            or not next_token
            or next_token in seen_tokens
        ):
            raise SpendAuthorityError("EC2 allocation pagination token repeated")
        seen_tokens.add(next_token)
        token = next_token
    allocation_ids = {interval.instance_id for interval in allocations}
    open_ids = {
        interval.instance_id
        for interval in allocations
        if interval.state == "OPEN"
    }
    if not set(by_id).issubset(allocation_ids) or not open_ids.issubset(by_id):
        raise SpendAuthorityError("EC2 allocation history diverges from ledger")
    fixed_tags = {
        "Project": "KEEP",
        "Campaign": "GLM-5.2",
        "RunId": RUN_ID,
        "Market": "on-demand",
    }
    required_tag_names = fixed_tags.keys() | {
        "campaign-identity-sha256", "activation-id", "activation-ordinal-text",
        "generation-text", "allocation-ordinal-text", "action-key",
        "sky-request-id", "sky-job-name", "sky-task-name", "task-yaml-sha256",
        "request-body-sha256",
    }
    for interval in allocations:
        if interval.instance_id not in by_id:
            # Terminated instances age out of DescribeInstances. The exact
            # authenticated allocation-ended ledger record is the closure
            # authority; live EC2 remains mandatory for every open interval.
            if interval.state == "CLOSED":
                continue
            raise SpendAuthorityError("open EC2 allocation is absent")
        instance = by_id[interval.instance_id]
        state = instance.get("State")
        expected_state = (
            {"running", "pending"}
            if interval.state == "OPEN"
            else {"stopped", "terminated"}
        )
        if (
            instance.get("InstanceType") != "p5.48xlarge"
            or instance.get("InstanceLifecycle") is not None
            or state not in expected_state
            or instance.get("LaunchTime") != interval.started_at
            or type(instance.get("AvailabilityZone")) is not str
            or not str(instance["AvailabilityZone"]).startswith(REGION)
            or type(instance.get("Tags")) is not dict
            or set(instance["Tags"]) != required_tag_names
            or any(instance["Tags"].get(key) != value for key, value in fixed_tags.items())
            or _SHA256.fullmatch(str(instance["Tags"].get("campaign-identity-sha256"))) is None
            or _SHA256.fullmatch(str(instance["Tags"].get("task-yaml-sha256"))) is None
            or _SHA256.fullmatch(str(instance["Tags"].get("request-body-sha256"))) is None
            or any(not isinstance(instance["Tags"].get(key), str) or not instance["Tags"][key] for key in ("activation-id", "sky-request-id", "sky-job-name", "sky-task-name"))
            or any(not isinstance(instance["Tags"].get(key), str) or re.fullmatch(r"[0-9]{8}", instance["Tags"][key]) is None or int(instance["Tags"][key]) <= 0 for key in ("activation-ordinal-text", "generation-text", "allocation-ordinal-text"))
            or instance["Tags"].get("action-key") != "ACTION#{0}#SKY_POST#{1}".format(instance["Tags"].get("activation-ordinal-text"), instance["Tags"].get("generation-text"))
        ):
            raise SpendAuthorityError("EC2 allocation history contains drift")


def _inspect_held_reserves(
    services: SpendAuthorityServices,
    *,
    ledger_chain_identities: set[str],
    observed_at: str,
) -> tuple[int, Decimal]:
    if services.reserve_reader is None:
        raise SpendAuthorityError("GPU reserve-reader boundary is required")
    method = getattr(services.reserve_reader, "list_reserves", None)
    if not callable(method):
        raise SpendAuthorityError("GPU reserve-reader boundary is incomplete")
    token: Optional[str] = None
    seen_tokens: set[str] = set()
    seen_keys: set[str] = set()
    held_seconds = 0
    held_cost = Decimal("0.00")
    record_count = 0
    while True:
        page = method(next_token=token)
        if not isinstance(page, ReserveListPage):
            raise SpendAuthorityError("GPU reserve page is not typed")
        if (
            type(page.request_id) is not str
            or not page.request_id
            or type(page.records) is not tuple
        ):
            raise SpendAuthorityError("GPU reserve page metadata is incomplete")
        _parse_time(page.observed_at, field="GPU reserve page observed_at")
        for record in page.records:
            if type(record) is not dict:
                raise SpendAuthorityError("GPU reserve record is malformed")
            fields = {
                "schema_version",
                "record_type",
                "account_id",
                "region",
                "run_id",
                "activation_id",
                "generation",
                "generation_text",
                "allocation_ordinal",
                "allocation_ordinal_text",
                "ec2_client_token",
                "request_identity_sha256",
                "ledger_predecessor_identity_sha256",
                "gpu_reserve_seconds",
                "gpu_reserve_cost_usd",
                "root_volume_gib",
                "root_volume_tail_usd_max",
                "residual_liability_approval_identity_sha256",
                "epoch",
                "revision",
                "nonce_owner_identity_sha256",
                "state",
                "observed_at",
                "reserve_key",
                "canonical_body_sha256",
            }
            if set(record) != fields:
                raise SpendAuthorityError("GPU reserve record schema mismatch")
            body = dict(record)
            identity = body.pop("canonical_body_sha256")
            reserve_key = record["reserve_key"]
            if (
                record["schema_version"] != 1
                or record["record_type"] != "glm52_gpu_liability_reserve_v1"
                or record["account_id"] != ACCOUNT_ID
                or record["region"] != REGION
                or record["run_id"] != RUN_ID
                or record["ledger_predecessor_identity_sha256"]
                not in ledger_chain_identities
                or record["gpu_reserve_seconds"] != GPU_RESERVE_SECONDS
                or _money(
                    record["gpu_reserve_cost_usd"],
                    field="gpu_reserve_cost_usd",
                )
                != GPU_RESERVE_COST_USD
                or record["root_volume_gib"] != ROOT_VOLUME_GIB
                or _money(
                    record["root_volume_tail_usd_max"],
                    field="root_volume_tail_usd_max",
                )
                != ROOT_VOLUME_TAIL_USD_MAX
                or record["state"] != "HELD"
                or type(reserve_key) is not str
                or not reserve_key
                or reserve_key in seen_keys
                or identity != _sha(body)
                or _parse_time(
                    record["observed_at"], field="reserve observed_at"
                )
                > _parse_time(observed_at, field="observed_at")
            ):
                raise SpendAuthorityError("GPU reserve record is divergent")
            for field in (
                "request_identity_sha256",
                "residual_liability_approval_identity_sha256",
                "nonce_owner_identity_sha256",
            ):
                _digest(record[field], field=field)
            seen_keys.add(reserve_key)
            record_count += 1
            held_seconds += GPU_RESERVE_SECONDS
            held_cost += GPU_RESERVE_COST_USD
        if page.next_token is None:
            break
        if (
            type(page.next_token) is not str
            or not page.next_token
            or page.next_token in seen_tokens
        ):
            raise SpendAuthorityError("GPU reserve pagination token repeated")
        seen_tokens.add(page.next_token)
        token = page.next_token
    if record_count > 1:
        raise SpendAuthorityError("more than one launch liability is held")
    return held_seconds, held_cost


def inspect_spend_authority(
    request: SpendAuthorityRequest,
    services: SpendAuthorityServices,
) -> SpendAuthorityResult:
    """Perform one complete no-retry spend namespace and EC2 reconstruction."""

    if (
        request.account_id != ACCOUNT_ID
        or request.region != REGION
        or request.run_id != RUN_ID
    ):
        raise SpendAuthorityError("spend inspection request is foreign")
    observation = _parse_time(request.observed_at, field="observed_at")
    descriptor, _descriptor_object = _get_authenticated(
        services,
        key=request.descriptor_key,
        expected_sha256=request.descriptor_file_sha256,
        expected_version_id=request.descriptor_version_id,
        label="campaign descriptor",
    )
    approval, _approval_object = _get_authenticated(
        services,
        key=request.approval_key,
        expected_sha256=request.approval_file_sha256,
        expected_version_id=request.approval_version_id,
        label="GPU spend approval",
    )
    validate_gpu_spend_approval(approval)
    _validate_descriptor(
        descriptor,
        raw_sha256=request.descriptor_file_sha256,
        approval_sha256=request.approval_file_sha256,
    )
    latest, _latest_object = _get_authenticated(
        services,
        key=request.latest_key,
        expected_sha256=request.latest_file_sha256,
        expected_version_id=request.latest_version_id,
        label="GPU spend latest marker",
    )
    latest_fields = {
        "schema_version",
        "record_type",
        "run_id",
        "gpu_spend_authority_sha256",
        "record_count",
        "record_keys",
        "latest_record_sha256",
        "ledger_sha256",
        "latest_body_sha256",
    }
    if set(latest) != latest_fields:
        raise SpendAuthorityError("GPU spend latest marker schema mismatch")
    latest_body = dict(latest)
    latest_identity = latest_body.pop("latest_body_sha256")
    genesis = _ledger_genesis(request.approval_file_sha256)
    record_names = latest["record_keys"]
    if (
        latest["schema_version"] != 1
        or latest["record_type"] != "glm52_gpu_spend_ledger_latest_v1"
        or latest["run_id"] != RUN_ID
        or latest["gpu_spend_authority_sha256"] != genesis
        or latest_identity != _sha(latest_body)
        or type(record_names) is not list
        or latest["record_count"] != len(record_names)
    ):
        raise SpendAuthorityError("GPU spend latest marker is invalid")
    listed_entries = _list_record_keys(request, services)
    expected_keys = tuple(
        request.ledger_records_prefix + str(name) for name in record_names
    )
    if tuple(key for key, _version in listed_entries) != expected_keys:
        raise SpendAuthorityError(
            "GPU spend namespace differs from immutable latest inventory"
        )
    prior = genesis
    chain_identities = {genesis}
    ledger = bytearray()
    active: Optional[tuple[str, str, str]] = None
    intervals: list[AllocationInterval] = []
    prior_time: Optional[datetime] = None
    seen_instances: set[str] = set()
    for index, (name, key) in enumerate(zip(record_names, expected_keys)):
        match = _RECORD_NAME.fullmatch(str(name))
        if match is None or int(match.group("index")) != index:
            raise SpendAuthorityError("GPU spend immutable record name is invalid")
        record, observed = _get_authenticated(
            services,
            key=key,
            expected_sha256=None,
            expected_version_id=listed_entries[index][1],
            label=f"GPU spend immutable record {index}",
        )
        event = record.get("event")
        expected_fields = {
            "record_type",
            "run_id",
            "approval_sha256",
            "event",
            "instance_id",
            "timestamp",
            "prior_record_sha256",
            "record_sha256",
        } | ({"job_id"} if event == "allocation_started" else set())
        if set(record) != expected_fields:
            raise SpendAuthorityError("GPU spend immutable record schema mismatch")
        body = dict(record)
        record_identity = body.pop("record_sha256")
        timestamp = _parse_time(record.get("timestamp"), field="record timestamp")
        instance_id = record.get("instance_id")
        if (
            record["record_type"] != "glm52_gpu_spend_event_v1"
            or record["run_id"] != RUN_ID
            or record["approval_sha256"] != request.approval_file_sha256
            or record["prior_record_sha256"] != prior
            or record_identity != _sha(body)
            or record_identity != match.group("sha")
            or event != match.group("event")
            or type(instance_id) is not str
            or _INSTANCE_ID.fullmatch(instance_id) is None
            or timestamp > observation
            or (prior_time is not None and timestamp < prior_time)
        ):
            raise SpendAuthorityError("GPU spend immutable record is divergent")
        if event == "allocation_started":
            job_id = record.get("job_id")
            if (
                active is not None
                or type(job_id) is not str
                or not job_id
                or instance_id in seen_instances
            ):
                raise SpendAuthorityError("GPU allocation start is invalid")
            active = (job_id, instance_id, str(record["timestamp"]))
            seen_instances.add(instance_id)
        elif event == "allocation_ended":
            if active is None or active[1] != instance_id:
                raise SpendAuthorityError("GPU allocation end is noncontiguous")
            started = _parse_time(active[2], field="allocation start")
            seconds_value = (timestamp - started).total_seconds()
            if seconds_value < 0 or not seconds_value.is_integer():
                raise SpendAuthorityError("GPU allocation duration is not exact")
            seconds = int(seconds_value)
            intervals.append(
                AllocationInterval(
                    instance_id=instance_id,
                    job_id=active[0],
                    started_at=active[2],
                    ended_at=str(record["timestamp"]),
                    charged_seconds=seconds,
                    charged_cost_usd=_cost(seconds),
                    state="CLOSED",
                )
            )
            active = None
        else:
            raise SpendAuthorityError("GPU spend event is invalid")
        prior = str(record_identity)
        chain_identities.add(prior)
        prior_time = timestamp
        ledger.extend(observed.raw)
    if active is not None:
        started = _parse_time(active[2], field="allocation start")
        seconds_value = (observation - started).total_seconds()
        if seconds_value < 0 or not seconds_value.is_integer():
            raise SpendAuthorityError("open GPU allocation duration is not exact")
        seconds = int(seconds_value)
        intervals.append(
            AllocationInterval(
                instance_id=active[1],
                job_id=active[0],
                started_at=active[2],
                ended_at=None,
                charged_seconds=seconds,
                charged_cost_usd=_cost(seconds),
                state="OPEN",
            )
        )
    if prior != latest["latest_record_sha256"]:
        raise SpendAuthorityError("GPU spend chain tip differs from latest")
    if _sha_raw(bytes(ledger)) != latest["ledger_sha256"]:
        raise SpendAuthorityError("GPU spend ledger byte identity differs")
    _inspect_ec2(services, allocations=tuple(intervals))
    snapshot, _snapshot_object = _get_authenticated(
        services,
        key=request.snapshot_key,
        expected_sha256=request.snapshot_file_sha256,
        expected_version_id=request.snapshot_version_id,
        label="GPU spend snapshot",
    )
    snapshot_fields = {
        "schema_version",
        "record_type",
        "run_id",
        "campaign_identity_sha256",
        "descriptor_sha256",
        "descriptor_body_sha256",
        "approval_sha256",
        "approval_body_sha256",
        "gpu_spend_ledger_latest_sha256",
        "gpu_spend_ledger_latest_body_sha256",
        "gpu_spend_ledger_genesis_sha256",
        "gpu_spend_ledger_record_count",
        "gpu_spend_ledger_tip_record_sha256",
        "gpu_spend_ledger_file_sha256",
        "ec2_allocation_history_sha256",
        "ec2_allocation_instance_ids",
        "observed_at",
        "approved_gpu_runtime_seconds",
        "approved_gpu_cost_usd",
        "hourly_cost_usd",
        "consumed_gpu_seconds",
        "remaining_gpu_seconds",
        "consumed_gpu_cost_usd",
        "remaining_gpu_cost_usd",
        "qualification_allowance_seconds",
        "qualification_allowance_cost_usd",
        "open_allocation_count",
        "snapshot_body_sha256",
    }
    if set(snapshot) != snapshot_fields:
        raise SpendAuthorityError("GPU spend snapshot schema mismatch")
    snapshot_body = dict(snapshot)
    snapshot_identity = snapshot_body.pop("snapshot_body_sha256")
    used_seconds = sum(
        interval.charged_seconds
        for interval in intervals
        if interval.state == "CLOSED"
    )
    open_seconds = sum(
        interval.charged_seconds
        for interval in intervals
        if interval.state == "OPEN"
    )
    charged_seconds = used_seconds + open_seconds
    # The accepted snapshot rounds once after summing all charged seconds.
    used_cost = _cost(used_seconds)
    open_cost = _cost(charged_seconds) - used_cost
    charged_cost = used_cost + open_cost
    remaining_seconds = APPROVED_GPU_RUNTIME_SECONDS - charged_seconds
    remaining_cost = APPROVED_GPU_COST_USD - charged_cost
    if (
        snapshot_identity != _sha(snapshot_body)
        or snapshot_identity != request.expected_snapshot_body_sha256
        or snapshot["run_id"] != RUN_ID
        or snapshot["campaign_identity_sha256"]
        != descriptor["campaign_identity_sha256"]
        or snapshot["descriptor_sha256"] != request.descriptor_file_sha256
        or snapshot["descriptor_body_sha256"]
        != descriptor["descriptor_body_sha256"]
        or snapshot["approval_sha256"] != request.approval_file_sha256
        or snapshot["approval_body_sha256"] != approval["approval_body_sha256"]
        or snapshot["gpu_spend_ledger_latest_sha256"]
        != request.latest_file_sha256
        or snapshot["gpu_spend_ledger_latest_body_sha256"] != latest_identity
        or snapshot["gpu_spend_ledger_genesis_sha256"] != genesis
        or snapshot["gpu_spend_ledger_record_count"] != len(record_names)
        or snapshot["gpu_spend_ledger_tip_record_sha256"] != prior
        or snapshot["gpu_spend_ledger_file_sha256"] != _sha_raw(bytes(ledger))
        or snapshot["ec2_allocation_instance_ids"]
        != [interval.instance_id for interval in intervals]
        or snapshot["observed_at"]
        != (request.snapshot_observed_at or request.observed_at)
        or snapshot["approved_gpu_runtime_seconds"]
        != APPROVED_GPU_RUNTIME_SECONDS
        or _money(
            snapshot["approved_gpu_cost_usd"],
            field="snapshot approved_gpu_cost_usd",
        )
        != APPROVED_GPU_COST_USD
        or _money(snapshot["hourly_cost_usd"], field="snapshot hourly_cost_usd")
        != APPROVED_HOURLY_COST_USD
        or snapshot["consumed_gpu_seconds"] != charged_seconds
        or snapshot["remaining_gpu_seconds"] != remaining_seconds
        or _money(
            snapshot["consumed_gpu_cost_usd"],
            field="snapshot consumed_gpu_cost_usd",
        )
        != charged_cost
        or _money(
            snapshot["remaining_gpu_cost_usd"],
            field="snapshot remaining_gpu_cost_usd",
        )
        != remaining_cost
        or snapshot["qualification_allowance_seconds"]
        != min(14_400, remaining_seconds)
        or _money(
            snapshot["qualification_allowance_cost_usd"],
            field="snapshot qualification_allowance_cost_usd",
        )
        != _cost(min(14_400, remaining_seconds))
        or snapshot["open_allocation_count"]
        != sum(interval.state == "OPEN" for interval in intervals)
    ):
        raise SpendAuthorityError("GPU spend snapshot diverges from live walk")
    reserved_seconds, reserved_cost = _inspect_held_reserves(
        services,
        ledger_chain_identities=chain_identities,
        observed_at=request.observed_at,
    )
    if (
        reserved_seconds > remaining_seconds
        or reserved_cost > remaining_cost
    ):
        raise SpendAuthorityError("held GPU reserve exceeds remaining authority")
    result = spend_authority_result_from_mapping(
        {
            "account_id": ACCOUNT_ID,
            "region": REGION,
            "run_id": RUN_ID,
            "approval_identity_sha256": approval["approval_body_sha256"],
            "descriptor_identity_sha256": descriptor["descriptor_body_sha256"],
            "snapshot_identity_sha256": snapshot_identity,
            "ledger_genesis_identity_sha256": genesis,
            "ledger_tip_identity_sha256": prior,
            "ledger_record_count": len(record_names),
            "used_gpu_seconds": used_seconds,
            "open_gpu_seconds": open_seconds,
            "reserved_gpu_seconds": reserved_seconds,
            "remaining_gpu_seconds": remaining_seconds - reserved_seconds,
            "refundable_gpu_seconds": 0,
            "used_gpu_cost_usd": used_cost,
            "open_gpu_cost_usd": open_cost,
            "reserved_gpu_cost_usd": reserved_cost,
            "remaining_gpu_cost_usd": remaining_cost - reserved_cost,
            "refundable_gpu_cost_usd": Decimal("0.00"),
            "allocation_intervals": tuple(asdict(interval) for interval in intervals),
            "observed_at": request.observed_at,
        }
    )
    return result


@dataclass(frozen=True)
class GpuLiabilityReserveRequest:
    account_id: str
    region: str
    run_id: str
    activation_id: str
    generation: int
    allocation_ordinal: int
    ec2_client_token: str
    request_identity_sha256: str
    ledger_predecessor_identity_sha256: str
    gpu_reserve_seconds: int
    gpu_reserve_cost_usd: Decimal
    root_volume_gib: int
    root_volume_tail_usd_max: Decimal
    residual_liability_approval: Mapping[str, object]
    residual_liability_source_bytes: bytes
    residual_liability_approval_identity_sha256: str
    epoch: int
    revision: int
    nonce_owner_identity_sha256: str
    observed_at: str


@dataclass(frozen=True)
class GpuLiabilityReserveResult:
    disposition: str
    reserve_key: str
    reserve_identity_sha256: str
    gpu_reserve_seconds: int
    gpu_reserve_cost_usd: Decimal
    root_volume_gib: int
    root_volume_tail_usd_max: Decimal
    remaining_gpu_seconds: int
    remaining_gpu_cost_usd: Decimal
    record: Mapping[str, object]


def _reserve_record(
    request: GpuLiabilityReserveRequest, current: SpendAuthorityResult
) -> tuple[str, dict[str, object]]:
    if (
        request.account_id != ACCOUNT_ID
        or request.region != REGION
        or request.run_id != RUN_ID
        or request.generation <= 0
        or request.allocation_ordinal <= 0
        or request.epoch <= 0
        or request.revision < 0
        or type(request.activation_id) is not str
        or not request.activation_id
        or type(request.ec2_client_token) is not str
        or len(request.ec2_client_token) != 64
    ):
        raise SpendAuthorityError("GPU reserve request identity is invalid")
    for field in (
        "request_identity_sha256",
        "ledger_predecessor_identity_sha256",
        "residual_liability_approval_identity_sha256",
        "nonce_owner_identity_sha256",
    ):
        _digest(getattr(request, field), field=field)
    if (
        request.gpu_reserve_seconds != GPU_RESERVE_SECONDS
        or _money(request.gpu_reserve_cost_usd, field="gpu_reserve_cost_usd")
        != GPU_RESERVE_COST_USD
        or request.root_volume_gib != ROOT_VOLUME_GIB
        or _money(
            request.root_volume_tail_usd_max,
            field="root_volume_tail_usd_max",
        )
        != ROOT_VOLUME_TAIL_USD_MAX
    ):
        raise SpendAuthorityError("GPU reserve amount or root tail is not exact")
    if request.observed_at != current.observed_at:
        raise SpendAuthorityError("GPU reserve request uses stale spend authority")
    validate_spend_authority_result(current)
    try:
        approval = validate_gpu_residual_liability_approval(
            request.residual_liability_approval,
            request.residual_liability_source_bytes,
        )
    except (TypeError, ValueError) as exc:
        raise SpendAuthorityError(
            "residual-liability approval did not authenticate"
        ) from exc
    if (
        approval["canonical_body_sha256"]
        != request.residual_liability_approval_identity_sha256
        or request.ledger_predecessor_identity_sha256
        != current.ledger_tip_identity_sha256
        or current.open_gpu_seconds != 0
    ):
        raise SpendAuthorityError("GPU reserve head is not admissible")
    reserve_key = (
        f"GPU_LIABILITY_RESERVE#{request.activation_id}#"
        f"{request.generation:08d}#{request.allocation_ordinal:08d}"
    )
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_gpu_liability_reserve_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": request.activation_id,
        "generation": request.generation,
        "generation_text": f"{request.generation:08d}",
        "allocation_ordinal": request.allocation_ordinal,
        "allocation_ordinal_text": f"{request.allocation_ordinal:08d}",
        "ec2_client_token": request.ec2_client_token,
        "request_identity_sha256": request.request_identity_sha256,
        "ledger_predecessor_identity_sha256": (
            request.ledger_predecessor_identity_sha256
        ),
        "gpu_reserve_seconds": GPU_RESERVE_SECONDS,
        "gpu_reserve_cost_usd": GPU_RESERVE_COST_USD,
        "root_volume_gib": ROOT_VOLUME_GIB,
        "root_volume_tail_usd_max": ROOT_VOLUME_TAIL_USD_MAX,
        "residual_liability_approval_identity_sha256": (
            request.residual_liability_approval_identity_sha256
        ),
        "epoch": request.epoch,
        "revision": request.revision,
        "nonce_owner_identity_sha256": request.nonce_owner_identity_sha256,
        "state": "HELD",
        "observed_at": request.observed_at,
        "reserve_key": reserve_key,
    }
    return reserve_key, {**body, "canonical_body_sha256": _sha(body)}


def reserve_gpu_liability(
    request: GpuLiabilityReserveRequest,
    spend_request: SpendAuthorityRequest,
    services: SpendAuthorityServices,
    writer: object,
) -> GpuLiabilityReserveResult:
    """Freshly reconstruct spend, then reserve once with one exact readback."""

    if not isinstance(spend_request, SpendAuthorityRequest):
        raise SpendAuthorityError(
            "GPU reserve requires a typed fresh spend-inspection request"
        )
    current = inspect_spend_authority(spend_request, services)
    reserve_key, record = _reserve_record(request, current)
    read = getattr(writer, "exact_read", None)
    persist = getattr(writer, "persist_intent", None)
    append = getattr(writer, "conditional_append", None)
    if not callable(read) or not callable(persist) or not callable(append):
        raise SpendAuthorityError("GPU reserve writer boundary is incomplete")
    existing = read(reserve_key=reserve_key)
    if existing is not None:
        if existing != record:
            raise SpendAuthorityError("existing GPU reserve differs")
        if (
            current.reserved_gpu_seconds != GPU_RESERVE_SECONDS
            or current.reserved_gpu_cost_usd != GPU_RESERVE_COST_USD
        ):
            raise SpendAuthorityError(
                "exact GPU reserve is absent from the fresh held-reserve walk"
            )
        disposition = "EXACT_DUPLICATE"
    else:
        if (
            current.reserved_gpu_seconds != 0
            or current.reserved_gpu_cost_usd != Decimal("0.00")
        ):
            raise SpendAuthorityError(
                "a foreign or unresolved GPU reserve is already held"
            )
        if (
            current.remaining_gpu_seconds < GPU_RESERVE_SECONDS
            or current.remaining_gpu_cost_usd < GPU_RESERVE_COST_USD
        ):
            raise SpendAuthorityError("GPU reserve balance is insufficient")
        intent_body: dict[str, object] = {
            "schema_version": 1,
            "record_type": "glm52_gpu_liability_reserve_intent_v1",
            "reserve_key": reserve_key,
            "reserve_identity_sha256": record["canonical_body_sha256"],
            "ledger_predecessor_identity_sha256": (
                current.ledger_tip_identity_sha256
            ),
            "nonce_owner_identity_sha256": request.nonce_owner_identity_sha256,
            "observed_at": request.observed_at,
        }
        intent = {**intent_body, "canonical_body_sha256": _sha(intent_body)}
        persist(intent=intent)
        try:
            response = append(
                record=record,
                expected_predecessor_identity_sha256=(
                    current.ledger_tip_identity_sha256
                ),
            )
        except AmbiguousReserveTransport:
            response = read(reserve_key=reserve_key)
            if response != record:
                raise SpendAuthorityError(
                    "ambiguous GPU reserve did not reconcile exactly"
                )
            disposition = "AMBIGUOUS_EXACT_READBACK"
        if "disposition" not in locals():
            if response != record:
                raise SpendAuthorityError("GPU reserve response differs")
            disposition = "CREATED"
    is_duplicate = disposition == "EXACT_DUPLICATE"
    return GpuLiabilityReserveResult(
        disposition=disposition,
        reserve_key=reserve_key,
        reserve_identity_sha256=str(record["canonical_body_sha256"]),
        gpu_reserve_seconds=GPU_RESERVE_SECONDS,
        gpu_reserve_cost_usd=GPU_RESERVE_COST_USD,
        root_volume_gib=ROOT_VOLUME_GIB,
        root_volume_tail_usd_max=ROOT_VOLUME_TAIL_USD_MAX,
        remaining_gpu_seconds=(
            current.remaining_gpu_seconds
            if is_duplicate
            else current.remaining_gpu_seconds - GPU_RESERVE_SECONDS
        ),
        remaining_gpu_cost_usd=(
            current.remaining_gpu_cost_usd
            if is_duplicate
            else current.remaining_gpu_cost_usd - GPU_RESERVE_COST_USD
        ),
        record=record,
    )


@dataclass(frozen=True)
class LiabilitySettlementEvidence:
    schema_version: int
    record_type: str
    account_id: str
    region: str
    run_id: str
    reserve_identity_sha256: str
    liability_identity_sha256: str
    allocation_close_identity_sha256: str
    instance_terminal_identity_sha256: str
    charged_gpu_seconds: int
    charged_gpu_cost_usd: Decimal
    refundable_gpu_seconds: int
    refundable_gpu_cost_usd: Decimal
    settled_at: str
    canonical_body_sha256: str


def validate_liability_settlement(
    reserve: GpuLiabilityReserveResult,
    settlement: Optional[LiabilitySettlementEvidence],
) -> LiabilitySettlementEvidence:
    """Validate Task 9's future exact settlement; this performs no release."""

    if not isinstance(settlement, LiabilitySettlementEvidence):
        raise SpendAuthorityError("no authenticated liability settlement")
    body = asdict(settlement)
    identity = body.pop("canonical_body_sha256")
    if (
        settlement.schema_version != 1
        or settlement.record_type != "glm52_gpu_liability_settlement_v1"
        or settlement.account_id != ACCOUNT_ID
        or settlement.region != REGION
        or settlement.run_id != RUN_ID
        or settlement.reserve_identity_sha256
        != reserve.reserve_identity_sha256
        or identity != _sha(body)
    ):
        raise SpendAuthorityError("liability settlement identity is invalid")
    for field in (
        "liability_identity_sha256",
        "allocation_close_identity_sha256",
        "instance_terminal_identity_sha256",
    ):
        _digest(getattr(settlement, field), field=field)
    if (
        settlement.charged_gpu_seconds < 0
        or settlement.refundable_gpu_seconds < 0
        or settlement.charged_gpu_seconds
        + settlement.refundable_gpu_seconds
        != GPU_RESERVE_SECONDS
        or _money(
            settlement.charged_gpu_cost_usd,
            field="charged_gpu_cost_usd",
        )
        + _money(
            settlement.refundable_gpu_cost_usd,
            field="refundable_gpu_cost_usd",
        )
        != GPU_RESERVE_COST_USD
    ):
        raise SpendAuthorityError("liability settlement accounting diverges")
    _parse_time(settlement.settled_at, field="settled_at")
    return settlement


__all__ = [
    "ACCOUNT_ID",
    "APPROVED_GPU_COST_USD",
    "APPROVED_GPU_RUNTIME_SECONDS",
    "APPROVED_HOURLY_COST_USD",
    "REGION",
    "RUN_ID",
    "AllocationInterval",
    "AmbiguousReserveTransport",
    "GpuLiabilityReserveRequest",
    "GpuLiabilityReserveResult",
    "LiabilitySettlementEvidence",
    "ReserveRejected",
    "ReserveListPage",
    "SpendAuthorityError",
    "SpendAuthorityRequest",
    "SpendAuthorityResult",
    "SpendAuthorityServices",
    "SpendListPage",
    "SpendObject",
    "build_gpu_spend_approval",
    "canonical_decimal_json_bytes",
    "inspect_spend_authority",
    "reserve_gpu_liability",
    "spend_authority_request_from_mapping",
    "spend_authority_result_from_mapping",
    "validate_gpu_spend_approval",
    "validate_liability_settlement",
    "validate_spend_authority_result",
]
