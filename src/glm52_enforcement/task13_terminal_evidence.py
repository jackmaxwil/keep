"""Generation-1 terminal evidence and one-shot S3 publication.

The Task 13 coordinator consumes three immutable objects at the generation-1
terminal prefix.  This module owns the two records that were previously only
read: ``TERMINAL_VERIFIED.json`` and ``TASK13_TERMINAL_PROOF.json``.
"""

from __future__ import annotations

import base64
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Mapping

from .canonical import canonical_json_bytes
from .support_price_card_authority import build_approved_support_price_card

ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
GENERATION = 1
GENERATION_TEXT = "00000001"
CAMPAIGN_BUCKET = "keep-glm52-models-246813579024-us-west-2"
MODEL_BUCKET = "keep-glm52-models-246813579024-us-west-2"
GPU_SPEND_LEDGER_KEY = f"campaigns/{RUN_ID}/runtime/GPU_SPEND_LEDGER.jsonl"
TERMINAL_PREFIX = (
    f"campaigns/{RUN_ID}/submissions/production/generations/"
    f"{GENERATION_TEXT}/terminal/"
)
CAMPAIGN_DRAINED_KEY = TERMINAL_PREFIX + "CAMPAIGN_DRAINED.json"
TERMINAL_VERIFIED_KEY = TERMINAL_PREFIX + "TERMINAL_VERIFIED.json"
TASK13_TERMINAL_PROOF_KEY = (
    TERMINAL_PREFIX + "TASK13_TERMINAL_PROOF.json"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_INSTANCE_ID = re.compile(r"^i-[0-9a-f]{17}$")
_USD = re.compile(r"^(?:0|[1-9][0-9]*)[.][0-9]{2}$")
_ACTIVATION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_VERSION = re.compile(r"^[A-Za-z0-9._~+/=-]{1,1024}$")
_MUTABLE_VERSION_WORDS = {"", "null", "latest", "$latest"}
_RETAINED_COST_FIELDS = frozenset(
    {
        "retained_s3",
        "retained_ledger",
        "retained_kms",
        "liability_watcher",
        "snapshot_cleanup",
        "snapshot_retention",
    }
)
_ALLOCATION_KINDS = frozenset(
    {
        "qualification-cache-seed",
        "source-qualification",
        "replacement-qualification",
        "production",
        "recovery",
    }
)
_MARKER_FIELDS = frozenset(
    {
        "key",
        "version_id",
        "file_sha256",
        "body_sha256",
        "canonical_identity_sha256",
    }
)
_TERMINAL_STATE_FIELDS = frozenset(
    {
        "record_type",
        "run_id",
        "phase",
        "outcome",
        "last_completed_phase",
        "terminal_state_authenticated",
        "drained_sha256",
        "campaign_ledger_sha256",
        "gpu_spend_ledger_sha256",
        "descriptor_body_sha256",
        "verification_body_sha256",
    }
)
_TERMINAL_VERIFIED_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "status",
        "account_id",
        "region",
        "run_id",
        "activation_id",
        "generation",
        "generation_text",
        "sky_state",
        "ec2_billable_instance_ids",
        "campaign_drained",
        "terminal_state",
        "verified_at",
        "canonical_identity_sha256",
    }
)
_TERMINAL_PROOF_FIELDS = frozenset(
    {
        "status",
        "account_id",
        "region",
        "run_id",
        "activation_id",
        "sky_state",
        "ec2_billable_instance_ids",
        "campaign_drained",
        "terminal_verified",
        "settlement",
        "canonical_identity_sha256",
    }
)
_SETTLEMENT_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "activation_id",
        "generation",
        "generation_text",
        "terminal_state_verification_body_sha256",
        "gpu_spend_ledger",
        "gpu_hourly_price_usd",
        "approved_gpu_seconds",
        "approved_gpu_usd",
        "residual_reserve_gpu_seconds",
        "residual_reserve_gpu_usd",
        "root_volume_tail_usd_max",
        "allocations",
        "total_gpu_seconds",
        "total_gpu_cost_usd",
        "remaining_gpu_seconds",
        "remaining_gpu_usd",
        "support_price_card",
        "retained_costs",
        "canonical_identity_sha256",
    }
)
_SPEND_LEDGER_FIELDS = frozenset(
    {
        "bucket",
        "key",
        "version_id",
        "file_sha256",
        "body_sha256",
        "head_record_sha256",
        "canonical_identity_sha256",
    }
)
_CAMPAIGN_DRAINED_FIELDS = frozenset(
    {
        "record_type",
        "run_id",
        "outcome",
        "last_completed_phase",
        "drain_reason",
        "prior_record_sha256",
        "teacher_cache_ready_sha256",
        "automatic_model_promotion",
        "model_uploaded",
        "execution_deadline",
        "gpu_allocation_sha256",
    }
)
_CAMPAIGN_PHASES = frozenset(
    {
        "BOOTSTRAP",
        "CUDA_GATE",
        "TEACHER",
        "CACHE_AUDIT",
        "TRAINING_PREFLIGHT",
        "TRAINING",
        "EVALUATION",
    }
)
_ALLOCATION_FIELDS = frozenset(
    {
        "allocation_kind",
        "job_id",
        "instance_id",
        "started_at",
        "ended_at",
        "gpu_seconds",
        "gpu_cost_usd",
        "start_record_sha256",
        "end_record_sha256",
    }
)
_GPU_HOURLY_PRICE_USD = "55.04"
_APPROVED_GPU_SECONDS = 86_400
_APPROVED_GPU_USD = "1320.96"
_RESIDUAL_RESERVE_GPU_SECONDS = 900
_RESIDUAL_RESERVE_GPU_USD = "13.76"
_ROOT_VOLUME_TAIL_USD_MAX = "0.01"


class TerminalEvidenceError(ValueError):
    """Task 13 terminal evidence is incomplete, foreign, or mutable."""


def _canonical(value: object) -> bytes:
    return canonical_json_bytes(value)


def _sha(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise TerminalEvidenceError(label + " must be a lowercase SHA-256")
    return value


def _activation(value: object) -> str:
    if type(value) is not str or _ACTIVATION.fullmatch(value) is None:
        raise TerminalEvidenceError("activation_id is invalid")
    return value


def _canonical_time(value: object, *, label: str) -> str:
    if type(value) is not str or not value.endswith("Z"):
        raise TerminalEvidenceError(label + " must be canonical UTC")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise TerminalEvidenceError(label + " must be canonical UTC") from error
    if (
        parsed.tzinfo is None
        or parsed.microsecond != 0
        or parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        != value
    ):
        raise TerminalEvidenceError(label + " must be canonical UTC")
    return value


def _self_hash(value: Mapping[str, object], *, label: str) -> None:
    unsigned = dict(value)
    identity = unsigned.pop("canonical_identity_sha256", None)
    if identity != hashlib.sha256(_canonical(unsigned)).hexdigest():
        raise TerminalEvidenceError(label + " canonical identity drifted")


def validate_marker_coordinate(
    value: object,
    *,
    expected_key: str,
) -> dict[str, object]:
    if (
        type(value) is not dict
        or set(value) != _MARKER_FIELDS
        or value.get("key") != expected_key
    ):
        raise TerminalEvidenceError(
            "immutable marker coordinate drifted: " + expected_key
        )
    version_id = value.get("version_id")
    if (
        type(version_id) is not str
        or version_id.lower() in _MUTABLE_VERSION_WORDS
        or _VERSION.fullmatch(version_id) is None
    ):
        raise TerminalEvidenceError(
            "immutable marker VersionId drifted: " + expected_key
        )
    _sha(value.get("file_sha256"), label=expected_key + " file")
    _sha(value.get("body_sha256"), label=expected_key + " body")
    _self_hash(value, label=expected_key + " coordinate")
    return dict(value)


def _validate_terminal_state(value: object) -> dict[str, object]:
    if (
        type(value) is not dict
        or set(value) != _TERMINAL_STATE_FIELDS
    ):
        raise TerminalEvidenceError(
            "legacy terminal-state verification schema drifted"
        )
    unsigned = dict(value)
    identity = unsigned.pop("verification_body_sha256")
    if (
        value.get("record_type")
        != "glm52_sky_terminal_verification_v1"
        or value.get("run_id") != RUN_ID
        or value.get("phase") != "DRAINED"
        or value.get("outcome")
        not in {"completed", "training_deferred", "resumable_deadline"}
        or type(value.get("last_completed_phase")) is not str
        or not value["last_completed_phase"]
        or value.get("terminal_state_authenticated") is not True
        or identity != hashlib.sha256(_canonical(unsigned)).hexdigest()
    ):
        raise TerminalEvidenceError(
            "legacy terminal-state verification authority drifted"
        )
    for field in (
        "drained_sha256",
        "campaign_ledger_sha256",
        "gpu_spend_ledger_sha256",
        "descriptor_body_sha256",
    ):
        _sha(value.get(field), label="terminal state " + field)
    return dict(value)


def validate_campaign_drained(value: object) -> dict[str, object]:
    """Validate the generation-scoped legacy Sky drained marker.

    ``CAMPAIGN_DRAINED`` and activation-scoped ``H1G_DRAINED`` are separate
    frozen authorities.  The former authenticates the Sky campaign ledger;
    the latter closes retained support-plane finalization and must never be
    substituted at this key.
    """

    if type(value) is not dict or set(value) != _CAMPAIGN_DRAINED_FIELDS:
        raise TerminalEvidenceError("campaign drained record schema drifted")
    exact = dict(value)
    if (
        exact.get("record_type") != "glm52_sky_campaign_drained_v2"
        or exact.get("run_id") != RUN_ID
        or exact.get("automatic_model_promotion") is not False
        or exact.get("model_uploaded") is not False
        or exact.get("last_completed_phase") not in _CAMPAIGN_PHASES
    ):
        raise TerminalEvidenceError("campaign drained authority drifted")
    _canonical_time(
        exact.get("execution_deadline"),
        label="campaign drained execution_deadline",
    )
    for field in ("prior_record_sha256", "gpu_allocation_sha256"):
        _sha(exact.get(field), label="campaign drained " + field)

    outcome = exact.get("outcome")
    last_phase = exact.get("last_completed_phase")
    drain_reason = exact.get("drain_reason")
    ready_sha = exact.get("teacher_cache_ready_sha256")
    if outcome == "completed":
        if (
            last_phase != "EVALUATION"
            or drain_reason != "campaign_complete"
        ):
            raise TerminalEvidenceError(
                "campaign drained completed authority drifted"
            )
        _sha(ready_sha, label="campaign drained teacher cache")
    elif outcome == "training_deferred":
        if (
            last_phase != "EVALUATION"
            or drain_reason != "training_deferred"
        ):
            raise TerminalEvidenceError(
                "campaign drained deferred authority drifted"
            )
        _sha(ready_sha, label="campaign drained teacher cache")
    elif outcome == "resumable_deadline":
        if drain_reason not in {
            "execution_window_prestop",
            "operator_graceful_stop",
        }:
            raise TerminalEvidenceError(
                "campaign drained resumable authority drifted"
            )
        if ready_sha is not None:
            _sha(ready_sha, label="campaign drained teacher cache")
    else:
        raise TerminalEvidenceError("campaign drained outcome drifted")
    return exact


def validate_terminal_verified(value: object) -> dict[str, object]:
    if (
        type(value) is not dict
        or set(value) != _TERMINAL_VERIFIED_FIELDS
    ):
        raise TerminalEvidenceError("terminal verified field set drifted")
    _self_hash(value, label="terminal verified")
    if (
        value.get("schema_version") != 1
        or value.get("record_type")
        != "glm52_task13_terminal_verified_v1"
        or value.get("status") != "TERMINAL_VERIFIED"
        or value.get("account_id") != ACCOUNT_ID
        or value.get("region") != REGION
        or value.get("run_id") != RUN_ID
        or _ACTIVATION.fullmatch(str(value.get("activation_id"))) is None
        or value.get("generation") != GENERATION
        or value.get("generation_text") != GENERATION_TEXT
        or value.get("sky_state") != "SUCCEEDED"
        or value.get("ec2_billable_instance_ids") != []
    ):
        raise TerminalEvidenceError("terminal verified scope drifted")
    drained_coordinate = validate_marker_coordinate(
        value["campaign_drained"],
        expected_key=CAMPAIGN_DRAINED_KEY,
    )
    terminal_state = _validate_terminal_state(value["terminal_state"])
    if (
        drained_coordinate["file_sha256"]
        != terminal_state["drained_sha256"]
    ):
        raise TerminalEvidenceError(
            "terminal verified campaign drained hash drifted"
        )
    _canonical_time(value["verified_at"], label="verified_at")
    return dict(value)


def _validate_campaign_drained_binding(
    *,
    coordinate: Mapping[str, object],
    drained_value: Mapping[str, object],
    terminal_state: Mapping[str, object],
) -> None:
    exact_coordinate = validate_marker_coordinate(
        coordinate,
        expected_key=CAMPAIGN_DRAINED_KEY,
    )
    exact_drained = validate_campaign_drained(drained_value)
    exact_terminal = _validate_terminal_state(terminal_state)
    if (
        exact_coordinate["file_sha256"]
        != exact_terminal["drained_sha256"]
        or exact_coordinate["file_sha256"]
        != hashlib.sha256(_canonical(exact_drained) + b"\n").hexdigest()
        or exact_coordinate["body_sha256"]
        != hashlib.sha256(_canonical(exact_drained)).hexdigest()
        or exact_drained["outcome"] != exact_terminal["outcome"]
        or exact_drained["last_completed_phase"]
        != exact_terminal["last_completed_phase"]
    ):
        raise TerminalEvidenceError(
            "terminal verified campaign drained source drifted"
        )


def build_terminal_verified(
    *,
    activation_id: str,
    campaign_drained: Mapping[str, object],
    campaign_drained_value: Mapping[str, object],
    terminal_state: Mapping[str, object],
    verified_at: str,
) -> dict[str, object]:
    exact_drained = validate_campaign_drained(campaign_drained_value)
    exact_coordinate = validate_marker_coordinate(
        campaign_drained,
        expected_key=CAMPAIGN_DRAINED_KEY,
    )
    exact_terminal = _validate_terminal_state(terminal_state)
    _validate_campaign_drained_binding(
        coordinate=exact_coordinate,
        drained_value=exact_drained,
        terminal_state=exact_terminal,
    )
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_task13_terminal_verified_v1",
        "status": "TERMINAL_VERIFIED",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": _activation(activation_id),
        "generation": GENERATION,
        "generation_text": GENERATION_TEXT,
        "sky_state": "SUCCEEDED",
        "ec2_billable_instance_ids": [],
        "campaign_drained": exact_coordinate,
        "terminal_state": exact_terminal,
        "verified_at": _canonical_time(verified_at, label="verified_at"),
    }
    value = {
        **body,
        "canonical_identity_sha256": hashlib.sha256(
            _canonical(body)
        ).hexdigest(),
    }
    return validate_terminal_verified(value)


def _usd_cents(value: object, *, label: str) -> int:
    if type(value) is not str or _USD.fullmatch(value) is None:
        raise TerminalEvidenceError(label + " is not canonical USD")
    whole, fractional = value.split(".")
    return int(whole) * 100 + int(fractional)


def _usd_from_seconds(seconds: int) -> str:
    return str(
        (
            Decimal(seconds)
            * Decimal(_GPU_HOURLY_PRICE_USD)
            / Decimal(3600)
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    )


def _spend_ledger_records(raw: object) -> list[dict[str, object]]:
    if type(raw) is not bytes or not raw or not raw.endswith(b"\n"):
        raise TerminalEvidenceError(
            "GPU spend ledger is not canonical nonempty JSONL"
        )
    records: list[dict[str, object]] = []
    prior: str | None = None
    approval: str | None = None
    active = False
    for line_number, line in enumerate(raw.splitlines(keepends=True), 1):
        try:
            value = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise TerminalEvidenceError(
                "GPU spend ledger contains malformed JSON"
            ) from error
        if (
            type(value) is not dict
            or line != _canonical(value) + b"\n"
        ):
            raise TerminalEvidenceError(
                "GPU spend ledger is not canonical JSONL"
            )
        body = dict(value)
        identity = body.pop("record_sha256", None)
        if (
            type(identity) is not str
            or identity != hashlib.sha256(_canonical(body)).hexdigest()
            or body.get("record_type") != "glm52_gpu_spend_event_v1"
            or body.get("run_id") != RUN_ID
            or type(body.get("approval_sha256")) is not str
            or _SHA256.fullmatch(str(body["approval_sha256"])) is None
            or type(body.get("instance_id")) is not str
            or _INSTANCE_ID.fullmatch(str(body["instance_id"])) is None
        ):
            raise TerminalEvidenceError(
                f"GPU spend ledger record {line_number} is foreign"
            )
        if approval is None:
            approval = str(body["approval_sha256"])
            genesis = {
                "record_type": "glm52_gpu_spend_ledger_genesis_v1",
                "run_id": RUN_ID,
                "approval_sha256": approval,
                "approved_gpu_runtime_seconds": _APPROVED_GPU_SECONDS,
                "approved_gpu_cost_usd": float(_APPROVED_GPU_USD),
                "hourly_cost_usd": float(_GPU_HOURLY_PRICE_USD),
            }
            prior = hashlib.sha256(_canonical(genesis)).hexdigest()
        if (
            body.get("approval_sha256") != approval
            or body.get("prior_record_sha256") != prior
        ):
            raise TerminalEvidenceError(
                "GPU spend ledger hash chain drifted"
            )
        event = body.get("event")
        if event == "allocation_started":
            if active or type(body.get("job_id")) is not str or not body["job_id"]:
                raise TerminalEvidenceError(
                    "GPU spend ledger allocation start drifted"
                )
            active = True
        elif event == "allocation_ended":
            if not active or "job_id" in body:
                raise TerminalEvidenceError(
                    "GPU spend ledger allocation end drifted"
                )
            active = False
        else:
            raise TerminalEvidenceError("GPU spend ledger event drifted")
        _canonical_time(body.get("timestamp"), label="GPU spend timestamp")
        prior = identity
        records.append(dict(value))
    if active or len(records) % 2:
        raise TerminalEvidenceError(
            "GPU spend ledger has an open allocation"
        )
    return records


def _spend_ledger_coordinate(
    value: object,
    *,
    raw: bytes,
    records: list[dict[str, object]],
) -> dict[str, object]:
    if (
        type(value) is not dict
        or set(value) != _SPEND_LEDGER_FIELDS
        or value.get("bucket") != MODEL_BUCKET
        or value.get("key") != GPU_SPEND_LEDGER_KEY
    ):
        raise TerminalEvidenceError("GPU spend ledger coordinate drifted")
    _version_id(value.get("version_id"), label="GPU spend ledger")
    _self_hash(value, label="GPU spend ledger coordinate")
    if (
        value.get("file_sha256") != hashlib.sha256(raw).hexdigest()
        or value.get("body_sha256")
        != hashlib.sha256(_canonical(records)).hexdigest()
        or value.get("head_record_sha256")
        != records[-1]["record_sha256"]
    ):
        raise TerminalEvidenceError(
            "GPU spend ledger source identity drifted"
        )
    return dict(value)


def _semantic_allocation_kinds(
    records: list[dict[str, object]],
) -> list[str]:
    """Derive allocation semantics from the authenticated ledger sequence."""

    starts = records[::2]
    job_ids = [str(record["job_id"]) for record in starts]
    cache_names = {"glm52-cache-seed", f"{RUN_ID}-cache-seed"}
    qualification_names = {
        "glm52-qualification",
        f"{RUN_ID}-qualification",
    }
    production_names = {
        "glm52-production",
        RUN_ID,
        f"{RUN_ID}-production",
    }
    recognized = cache_names | qualification_names | production_names

    if all(job_id in recognized for job_id in job_ids):
        seen_by_job: dict[str, int] = {}
        result: list[str] = []
        for job_id in job_ids:
            ordinal = seen_by_job.get(job_id, 0)
            seen_by_job[job_id] = ordinal + 1
            if job_id in cache_names:
                if ordinal:
                    raise TerminalEvidenceError(
                        "cache-seed allocation semantic sequence drifted"
                    )
                result.append("qualification-cache-seed")
            elif job_id in qualification_names:
                if ordinal > 1:
                    raise TerminalEvidenceError(
                        "qualification allocation semantic sequence drifted"
                    )
                result.append(
                    "source-qualification"
                    if ordinal == 0
                    else "replacement-qualification"
                )
            else:
                result.append("production" if ordinal == 0 else "recovery")
        ranks = {
            "qualification-cache-seed": 0,
            "source-qualification": 1,
            "replacement-qualification": 2,
            "production": 3,
            "recovery": 4,
        }
        if [ranks[kind] for kind in result] != sorted(
            ranks[kind] for kind in result
        ):
            raise TerminalEvidenceError(
                "allocation semantic stage order drifted"
            )
        if (
            result.count("qualification-cache-seed") != 1
            or result.count("production") != 1
            or result.count("source-qualification")
            != result.count("replacement-qualification")
        ):
            raise TerminalEvidenceError(
                "allocation semantic campaign cardinality drifted"
            )
        return result

    # The real Managed Jobs worker records the authenticated numeric Sky
    # controller job ID.  Its closed campaign order is one cache-seed job,
    # one two-instance qualification job, then the production job and any
    # later recovery jobs.
    if not job_ids or any(
        not job_id.isascii() or not job_id.isdigit()
        for job_id in job_ids
    ):
        raise TerminalEvidenceError(
            "allocation semantic job identity drifted"
        )
    groups: list[tuple[str, int]] = []
    for job_id in job_ids:
        if groups and groups[-1][0] == job_id:
            prior_job, count = groups[-1]
            groups[-1] = (prior_job, count + 1)
        else:
            if any(prior_job == job_id for prior_job, _count in groups):
                raise TerminalEvidenceError(
                    "allocation semantic job sequence is noncontiguous"
                )
            groups.append((job_id, 1))
    if len(groups) < 3 or groups[0][1] != 1 or groups[1][1] != 2:
        raise TerminalEvidenceError(
            "allocation semantic campaign sequence drifted"
        )
    result = ["qualification-cache-seed"]
    result.extend(["source-qualification", "replacement-qualification"])
    production_count = sum(count for _job_id, count in groups[2:])
    result.append("production")
    result.extend(["recovery"] * (production_count - 1))
    if len(result) != len(starts):
        raise TerminalEvidenceError(
            "allocation semantic campaign sequence drifted"
        )
    return result


def _allocation_rows(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    allocation_kinds = _semantic_allocation_kinds(records)
    rows: list[dict[str, object]] = []
    for index, kind in enumerate(allocation_kinds):
        start = records[index * 2]
        end = records[index * 2 + 1]
        if (
            kind not in _ALLOCATION_KINDS
            or start["event"] != "allocation_started"
            or end["event"] != "allocation_ended"
            or start["instance_id"] != end["instance_id"]
        ):
            raise TerminalEvidenceError("terminal allocation ledger drifted")
        started = datetime.fromisoformat(
            str(start["timestamp"]).replace("Z", "+00:00")
        )
        ended = datetime.fromisoformat(
            str(end["timestamp"]).replace("Z", "+00:00")
        )
        elapsed = ended - started
        seconds = int(elapsed.total_seconds())
        if seconds <= 0 or elapsed.total_seconds() != seconds:
            raise TerminalEvidenceError(
                "terminal allocation duration drifted"
            )
        rows.append(
            {
                "allocation_kind": kind,
                "job_id": start["job_id"],
                "instance_id": start["instance_id"],
                "started_at": start["timestamp"],
                "ended_at": end["timestamp"],
                "gpu_seconds": seconds,
                "gpu_cost_usd": _usd_from_seconds(seconds),
                "start_record_sha256": start["record_sha256"],
                "end_record_sha256": end["record_sha256"],
            }
        )
    return rows


def _retained_costs(price_card: Mapping[str, object]) -> dict[str, str]:
    rows = price_card.get("retained_terms")
    if type(rows) is not list:
        raise TerminalEvidenceError("support price-card retained terms drifted")
    result = {
        str(row.get("term")): str(row.get("estimated_usd"))
        for row in rows
        if type(row) is dict
    }
    if set(result) != _RETAINED_COST_FIELDS:
        raise TerminalEvidenceError("support price-card retained terms drifted")
    return result


def validate_terminal_settlement(
    value: object,
    *,
    terminal_state: Mapping[str, object],
    spend_ledger_raw: bytes,
) -> dict[str, object]:
    """Validate one authenticated final GPU/retained-cost settlement."""

    if type(value) is not dict or set(value) != _SETTLEMENT_FIELDS:
        raise TerminalEvidenceError("terminal settlement field set drifted")
    _self_hash(value, label="terminal settlement")
    terminal = _validate_terminal_state(terminal_state)
    records = _spend_ledger_records(spend_ledger_raw)
    coordinate = _spend_ledger_coordinate(
        value.get("gpu_spend_ledger"),
        raw=spend_ledger_raw,
        records=records,
    )
    allocations = value.get("allocations")
    if type(allocations) is not list or not allocations:
        raise TerminalEvidenceError("terminal allocation ledger drifted")
    expected_allocations = _allocation_rows(records)
    if (
        allocations != expected_allocations
        or any(
            type(row) is not dict or set(row) != _ALLOCATION_FIELDS
            for row in allocations
        )
    ):
        raise TerminalEvidenceError(
            "terminal allocation ledger does not match spend source"
        )
    exact_price_card = dict(build_approved_support_price_card())
    expected_retained = _retained_costs(exact_price_card)
    total_seconds = sum(
        int(row["gpu_seconds"]) for row in expected_allocations
    )
    total_cost = _usd_from_seconds(total_seconds)
    remaining_seconds = _APPROVED_GPU_SECONDS - total_seconds
    if (
        value.get("schema_version") != 1
        or value.get("record_type")
        != "glm52_task13_terminal_settlement_v1"
        or value.get("account_id") != ACCOUNT_ID
        or value.get("region") != REGION
        or value.get("run_id") != RUN_ID
        or _ACTIVATION.fullmatch(str(value.get("activation_id"))) is None
        or value.get("generation") != GENERATION
        or value.get("generation_text") != GENERATION_TEXT
        or value.get("terminal_state_verification_body_sha256")
        != terminal["verification_body_sha256"]
        or terminal["gpu_spend_ledger_sha256"]
        != coordinate["file_sha256"]
        or value.get("gpu_hourly_price_usd") != _GPU_HOURLY_PRICE_USD
        or value.get("approved_gpu_seconds") != _APPROVED_GPU_SECONDS
        or value.get("approved_gpu_usd") != _APPROVED_GPU_USD
        or value.get("residual_reserve_gpu_seconds")
        != _RESIDUAL_RESERVE_GPU_SECONDS
        or value.get("residual_reserve_gpu_usd")
        != _RESIDUAL_RESERVE_GPU_USD
        or value.get("root_volume_tail_usd_max")
        != _ROOT_VOLUME_TAIL_USD_MAX
        or value.get("total_gpu_seconds") != total_seconds
        or value.get("total_gpu_cost_usd") != total_cost
        or value.get("remaining_gpu_seconds") != remaining_seconds
        or remaining_seconds < 0
        or value.get("remaining_gpu_usd")
        != _usd_from_seconds(remaining_seconds)
        or value.get("support_price_card") != exact_price_card
        or value.get("retained_costs") != expected_retained
    ):
        raise TerminalEvidenceError(
            "terminal settlement provenance or budget drifted"
        )
    return json.loads(json.dumps(value))


def build_terminal_settlement(
    *,
    activation_id: str,
    terminal_state: Mapping[str, object],
    gpu_spend_ledger: Mapping[str, object],
    spend_ledger_raw: bytes,
    allocation_kinds: list[str],
) -> dict[str, object]:
    """Build the exact settlement only from the authenticated spend ledger."""

    terminal = _validate_terminal_state(terminal_state)
    records = _spend_ledger_records(spend_ledger_raw)
    coordinate = _spend_ledger_coordinate(
        gpu_spend_ledger,
        raw=spend_ledger_raw,
        records=records,
    )
    derived_kinds = _semantic_allocation_kinds(records)
    if list(allocation_kinds) != derived_kinds:
        raise TerminalEvidenceError(
            "caller allocation kinds disagree with authenticated semantics"
        )
    allocations = _allocation_rows(records)
    total_seconds = sum(int(row["gpu_seconds"]) for row in allocations)
    remaining_seconds = _APPROVED_GPU_SECONDS - total_seconds
    price_card = dict(build_approved_support_price_card())
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_task13_terminal_settlement_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": _activation(activation_id),
        "generation": GENERATION,
        "generation_text": GENERATION_TEXT,
        "terminal_state_verification_body_sha256": terminal[
            "verification_body_sha256"
        ],
        "gpu_spend_ledger": coordinate,
        "gpu_hourly_price_usd": _GPU_HOURLY_PRICE_USD,
        "approved_gpu_seconds": _APPROVED_GPU_SECONDS,
        "approved_gpu_usd": _APPROVED_GPU_USD,
        "residual_reserve_gpu_seconds": _RESIDUAL_RESERVE_GPU_SECONDS,
        "residual_reserve_gpu_usd": _RESIDUAL_RESERVE_GPU_USD,
        "root_volume_tail_usd_max": _ROOT_VOLUME_TAIL_USD_MAX,
        "allocations": allocations,
        "total_gpu_seconds": total_seconds,
        "total_gpu_cost_usd": _usd_from_seconds(total_seconds),
        "remaining_gpu_seconds": remaining_seconds,
        "remaining_gpu_usd": _usd_from_seconds(remaining_seconds),
        "support_price_card": price_card,
        "retained_costs": _retained_costs(price_card),
    }
    value = {
        **body,
        "canonical_identity_sha256": hashlib.sha256(
            _canonical(body)
        ).hexdigest(),
    }
    return validate_terminal_settlement(
        value,
        terminal_state=terminal,
        spend_ledger_raw=spend_ledger_raw,
    )


def validate_task13_terminal_proof(
    value: object,
    *,
    terminal_verified_value: Mapping[str, object],
    campaign_drained_value: Mapping[str, object],
    spend_ledger_raw: bytes,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != _TERMINAL_PROOF_FIELDS:
        raise TerminalEvidenceError("terminal proof field set drifted")
    _self_hash(value, label="terminal proof")
    if (
        value.get("status") != "TERMINAL_PROVEN"
        or value.get("account_id") != ACCOUNT_ID
        or value.get("region") != REGION
        or value.get("run_id") != RUN_ID
        or _ACTIVATION.fullmatch(str(value.get("activation_id"))) is None
        or value.get("sky_state") != "SUCCEEDED"
        or value.get("ec2_billable_instance_ids") != []
    ):
        raise TerminalEvidenceError("terminal proof scope drifted")
    validate_marker_coordinate(
        value["campaign_drained"],
        expected_key=CAMPAIGN_DRAINED_KEY,
    )
    validate_marker_coordinate(
        value["terminal_verified"],
        expected_key=TERMINAL_VERIFIED_KEY,
    )
    exact_verified = validate_terminal_verified(terminal_verified_value)
    if (
        exact_verified["activation_id"] != value["activation_id"]
        or value["terminal_verified"]["file_sha256"]
        != hashlib.sha256(
            _canonical(exact_verified) + b"\n"
        ).hexdigest()
        or value["terminal_verified"]["body_sha256"]
        != exact_verified["canonical_identity_sha256"]
    ):
        raise TerminalEvidenceError(
            "terminal proof terminal-verified source drifted"
        )
    exact_drained = validate_campaign_drained(campaign_drained_value)
    exact_terminal = exact_verified["terminal_state"]
    drained_body_sha256 = hashlib.sha256(
        _canonical(exact_drained)
    ).hexdigest()
    if (
        value["campaign_drained"] != exact_verified["campaign_drained"]
        or value["campaign_drained"]["file_sha256"]
        != exact_terminal["drained_sha256"]
        or value["campaign_drained"]["body_sha256"] != drained_body_sha256
        or exact_drained["outcome"] != exact_terminal["outcome"]
        or exact_drained["last_completed_phase"]
        != exact_terminal["last_completed_phase"]
    ):
        raise TerminalEvidenceError(
            "terminal proof campaign drained source drifted"
        )
    validate_terminal_settlement(
        value["settlement"],
        terminal_state=exact_terminal,
        spend_ledger_raw=spend_ledger_raw,
    )
    return dict(value)


def build_task13_terminal_proof(
    *,
    activation_id: str,
    campaign_drained: Mapping[str, object],
    campaign_drained_value: Mapping[str, object],
    terminal_verified: Mapping[str, object],
    terminal_verified_value: Mapping[str, object],
    settlement: Mapping[str, object],
    spend_ledger_raw: bytes,
) -> dict[str, object]:
    exact_settlement = validate_terminal_settlement(
        settlement,
        terminal_state=terminal_verified_value["terminal_state"],
        spend_ledger_raw=spend_ledger_raw,
    )
    body: dict[str, object] = {
        "status": "TERMINAL_PROVEN",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": _activation(activation_id),
        "sky_state": "SUCCEEDED",
        "ec2_billable_instance_ids": [],
        "campaign_drained": validate_marker_coordinate(
            campaign_drained,
            expected_key=CAMPAIGN_DRAINED_KEY,
        ),
        "terminal_verified": validate_marker_coordinate(
            terminal_verified,
            expected_key=TERMINAL_VERIFIED_KEY,
        ),
        "settlement": exact_settlement,
    }
    value = {
        **body,
        "canonical_identity_sha256": hashlib.sha256(
            _canonical(body)
        ).hexdigest(),
    }
    return validate_task13_terminal_proof(
        value,
        terminal_verified_value=terminal_verified_value,
        campaign_drained_value=campaign_drained_value,
        spend_ledger_raw=spend_ledger_raw,
    )


def _method(client: object, name: str):
    method = getattr(client, name, None)
    if not callable(method):
        raise TerminalEvidenceError("S3 boundary lacks " + name)
    return method


def _response(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise TerminalEvidenceError(label + " response is invalid")
    metadata = value.get("ResponseMetadata")
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
        or metadata.get("RetryAttempts") != 0
    ):
        raise TerminalEvidenceError(label + " response status is invalid")
    return value


def _version_id(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or value.lower() in _MUTABLE_VERSION_WORDS
        or _VERSION.fullmatch(value) is None
    ):
        raise TerminalEvidenceError(label + " VersionId is invalid")
    return value


def _exact_version_inventory(
    client: object,
    *,
    key: str,
) -> str:
    list_versions = _method(client, "list_object_versions")
    request: dict[str, object] = {
        "Bucket": CAMPAIGN_BUCKET,
        "Prefix": key,
        "ExpectedBucketOwner": ACCOUNT_ID,
    }
    versions: list[dict[str, object]] = []
    deletes: list[dict[str, object]] = []
    while True:
        page = _response(
            list_versions(**request),
            label="S3 version inventory",
        )
        versions.extend(
            row
            for row in page.get("Versions", [])
            if type(row) is dict and row.get("Key") == key
        )
        deletes.extend(
            row
            for row in page.get("DeleteMarkers", [])
            if type(row) is dict and row.get("Key") == key
        )
        if page.get("IsTruncated") is not True:
            break
        next_key = page.get("NextKeyMarker")
        next_version = page.get("NextVersionIdMarker")
        if type(next_key) is not str or type(next_version) is not str:
            raise TerminalEvidenceError(
                "S3 version inventory pagination drifted"
            )
        request["KeyMarker"] = next_key
        request["VersionIdMarker"] = next_version
    if (
        len(versions) != 1
        or deletes
        or versions[0].get("IsLatest") is not True
    ):
        raise TerminalEvidenceError(
            "S3 positive truth is not one immutable version"
        )
    return _version_id(
        versions[0].get("VersionId"),
        label="S3 immutable object",
    )


def _coordinate(
    *,
    key: str,
    version_id: str,
    raw: bytes,
    value: Mapping[str, object],
) -> dict[str, object]:
    unsigned = dict(value)
    canonical_identity = unsigned.pop("canonical_identity_sha256", None)
    canonical_body = unsigned.pop("canonical_body_sha256", None)
    if canonical_identity is not None and canonical_body is not None:
        raise TerminalEvidenceError(
            "immutable result has ambiguous canonical identity"
        )
    identity = (
        canonical_identity
        if canonical_identity is not None
        else canonical_body
    )
    body_sha256 = hashlib.sha256(_canonical(unsigned)).hexdigest()
    if identity is not None and identity != body_sha256:
        raise TerminalEvidenceError("immutable result self-hash drifted")
    body: dict[str, object] = {
        "key": key,
        "version_id": version_id,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "body_sha256": identity if identity is not None else body_sha256,
    }
    return {
        **body,
        "canonical_identity_sha256": hashlib.sha256(
            _canonical(body)
        ).hexdigest(),
    }


def read_exact_coordinate(
    client: object,
    *,
    key: str,
    validator=None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Read one key only when its complete version history is immutable."""

    version_id = _exact_version_inventory(client, key=key)
    response = _response(
        _method(client, "get_object")(
            Bucket=CAMPAIGN_BUCKET,
            Key=key,
            VersionId=version_id,
            ExpectedBucketOwner=ACCOUNT_ID,
            ChecksumMode="ENABLED",
        ),
        label="S3 exact readback",
    )
    body = response.get("Body")
    raw = body.read() if callable(getattr(body, "read", None)) else None
    if type(raw) is not bytes or response.get("VersionId") != version_id:
        raise TerminalEvidenceError("S3 exact readback body drifted")
    checksum = base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
    if response.get("ChecksumSHA256") != checksum:
        raise TerminalEvidenceError("S3 exact readback checksum drifted")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise TerminalEvidenceError("S3 exact object is not JSON") from error
    if type(value) is not dict or raw != _canonical(value) + b"\n":
        raise TerminalEvidenceError(
            "S3 exact object is not canonical JSON plus LF"
        )
    if validator is not None:
        value = validator(value)
    coordinate = _coordinate(
        key=key,
        version_id=version_id,
        raw=raw,
        value=value,
    )
    return dict(value), coordinate


def _publish(
    client: object,
    *,
    key: str,
    value: Mapping[str, object],
) -> dict[str, object]:
    raw = _canonical(value) + b"\n"
    identity = _sha(
        value.get("canonical_identity_sha256"),
        label="published canonical identity",
    )
    metadata = {
        "record-type": str(value["record_type"])
        if "record_type" in value
        else "glm52_task13_terminal_proof_v1",
        "canonical-identity-sha256": identity,
    }
    checksum = base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
    response_version: str | None = None
    try:
        response = _response(
            _method(client, "put_object")(
                Bucket=CAMPAIGN_BUCKET,
                Key=key,
                Body=raw,
                ContentType="application/json",
                Metadata=metadata,
                ChecksumAlgorithm="SHA256",
                ChecksumSHA256=checksum,
                ExpectedBucketOwner=ACCOUNT_ID,
                IfNoneMatch="*",
            ),
            label="S3 conditional terminal put",
        )
        response_version = _version_id(
            response.get("VersionId"),
            label="S3 conditional terminal put",
        )
    except Exception:  # noqa: BLE001 - exact inventory resolves ambiguity
        response_version = None
    winner_value, coordinate = read_exact_coordinate(
        client,
        key=key,
    )
    if winner_value != value:
        raise TerminalEvidenceError(
            "S3 immutable terminal winner differs from candidate"
        )
    if (
        response_version is not None
        and coordinate["version_id"] != response_version
    ):
        raise TerminalEvidenceError(
            "S3 terminal response VersionId differs from exact winner"
        )
    exact = _response(
        _method(client, "get_object")(
            Bucket=CAMPAIGN_BUCKET,
            Key=key,
            VersionId=coordinate["version_id"],
            ExpectedBucketOwner=ACCOUNT_ID,
            ChecksumMode="ENABLED",
        ),
        label="S3 terminal metadata readback",
    )
    if exact.get("Metadata") != metadata:
        raise TerminalEvidenceError("S3 terminal metadata drifted")
    return coordinate


def publish_terminal_verified(
    client: object,
    value: Mapping[str, object],
) -> dict[str, object]:
    exact = validate_terminal_verified(value)
    drained_value, drained_coordinate = read_exact_coordinate(
        client,
        key=CAMPAIGN_DRAINED_KEY,
        validator=validate_campaign_drained,
    )
    if drained_coordinate != exact["campaign_drained"]:
        raise TerminalEvidenceError(
            "terminal verified campaign drained VersionId drifted"
        )
    _validate_campaign_drained_binding(
        coordinate=drained_coordinate,
        drained_value=drained_value,
        terminal_state=exact["terminal_state"],
    )
    return _publish(client, key=TERMINAL_VERIFIED_KEY, value=exact)


def publish_task13_terminal_proof(
    client: object,
    value: Mapping[str, object],
    *,
    terminal_verified_value: Mapping[str, object],
    campaign_drained_value: Mapping[str, object],
    spend_ledger_raw: bytes,
) -> dict[str, object]:
    exact = validate_task13_terminal_proof(
        value,
        terminal_verified_value=terminal_verified_value,
        campaign_drained_value=campaign_drained_value,
        spend_ledger_raw=spend_ledger_raw,
    )
    return _publish(client, key=TASK13_TERMINAL_PROOF_KEY, value=exact)


def read_exact_spend_ledger(
    client: object,
    coordinate: Mapping[str, object],
) -> bytes:
    """Exact-read and authenticate the final versioned GPU spend ledger."""

    if (
        type(coordinate) is not dict
        or set(coordinate) != _SPEND_LEDGER_FIELDS
        or coordinate.get("bucket") != MODEL_BUCKET
        or coordinate.get("key") != GPU_SPEND_LEDGER_KEY
    ):
        raise TerminalEvidenceError("GPU spend ledger coordinate drifted")
    version_id = _version_id(
        coordinate.get("version_id"),
        label="GPU spend ledger",
    )
    response = _response(
        _method(client, "get_object")(
            Bucket=MODEL_BUCKET,
            Key=GPU_SPEND_LEDGER_KEY,
            VersionId=version_id,
            ExpectedBucketOwner=ACCOUNT_ID,
            ChecksumMode="ENABLED",
        ),
        label="GPU spend ledger exact readback",
    )
    body = response.get("Body")
    raw = body.read() if callable(getattr(body, "read", None)) else None
    if type(raw) is not bytes or response.get("VersionId") != version_id:
        raise TerminalEvidenceError(
            "GPU spend ledger exact readback body drifted"
        )
    checksum = base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
    if response.get("ChecksumSHA256") != checksum:
        raise TerminalEvidenceError(
            "GPU spend ledger exact readback checksum drifted"
        )
    records = _spend_ledger_records(raw)
    _spend_ledger_coordinate(coordinate, raw=raw, records=records)
    return raw


__all__ = [
    "CAMPAIGN_BUCKET",
    "CAMPAIGN_DRAINED_KEY",
    "GPU_SPEND_LEDGER_KEY",
    "MODEL_BUCKET",
    "TASK13_TERMINAL_PROOF_KEY",
    "TERMINAL_VERIFIED_KEY",
    "TerminalEvidenceError",
    "build_task13_terminal_proof",
    "build_terminal_settlement",
    "build_terminal_verified",
    "publish_task13_terminal_proof",
    "publish_terminal_verified",
    "read_exact_coordinate",
    "read_exact_spend_ledger",
    "validate_campaign_drained",
    "validate_marker_coordinate",
    "validate_task13_terminal_proof",
    "validate_terminal_settlement",
    "validate_terminal_verified",
]
