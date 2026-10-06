"""Closed authenticated approval records for the GLM-5.2 campaign."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Mapping

from .canonical import canonical_sha256


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
APPROVAL_PRINCIPAL_IDENTITY = "operator@example.com"
APPROVAL_PRINCIPAL_ROLE = "repo_owner_and_aws_account_human_principal"
SOURCE_TRANSPORT = "codex_chat"
SOURCE_AUTHORITY_PATH = (
    "docs/superpowers/approvals/"
    "2026-07-28-glm52-campaign-owner-approvals.md"
)
APPROVAL_RECEIVED_DATE = "2026-07-28"


_COMMON_TERMS: Dict[str, Any] = {
    "schema_version": 1,
    "account_id": ACCOUNT_ID,
    "region": REGION,
    "run_id": RUN_ID,
    "approval_status": "APPROVED",
    "approval_principal_identity": APPROVAL_PRINCIPAL_IDENTITY,
    "approval_principal_role": APPROVAL_PRINCIPAL_ROLE,
    "source_transport": SOURCE_TRANSPORT,
    "source_authority_path": SOURCE_AUTHORITY_PATH,
    "approval_received_date": APPROVAL_RECEIVED_DATE,
}

_SUPPORT_TERMS: Dict[str, Any] = {
    **_COMMON_TERMS,
    "record_type": "PRODUCTION_SUPPORT_PLANE_APPROVAL",
    "work_deadline_hours": 68,
    "delete_request_deadline_hours": 71,
    "cost_incident_deadline_hours": 72,
    "host_nat_processed_gib_limit": 10,
    "estimated_support_plane_ceiling_usd": "25.00",
    "workflow_event_limit": 12000,
    "forensic_snapshot_retention_days": 7,
    "liability_scan_interval_seconds": 60,
    "liability_scan_max_count": 43200,
    "liability_logs_gib_limit": 1,
    "liability_logs_retention_days": 30,
    "estimated_liability_watcher_ceiling_usd": "5.00",
    "exact_topology_and_cardinalities_approved": True,
    "network_and_storage_plan_approved": True,
    "logs_invocations_workflow_history_retention_approved": True,
    "eight_secret_inventory_approved": True,
    "endpoint_plan_approved": True,
    "dated_price_model_identity_approved_and_complete": True,
}

_RESIDUAL_TERMS: Dict[str, Any] = {
    **_COMMON_TERMS,
    "record_type": "GPU_RESIDUAL_LIABILITY_APPROVAL",
    "gpu_reserve_seconds": 900,
    "gpu_reserve_usd": "13.76",
    "root_volume_gib": 300,
    "root_volume_tail_usd_max": "0.01",
    "gpu_approval_hours_total": "24.00",
    "gpu_approval_usd_total": "1320.96",
    "delayed_visibility_accepted": True,
    "delayed_control_accepted": True,
    "delayed_termination_accepted": True,
    "same_token_multiplicity_accepted": True,
    "termination_only_continuation_until_settlement_accepted": True,
    "no_hard_post_acceptance_aws_billing_cap_accepted": True,
    "knowing_overage_risk_accepted": True,
}


def _source_sha256(source_authority_bytes: bytes) -> str:
    if type(source_authority_bytes) is not bytes:
        raise TypeError("source authority must be exact bytes")
    return hashlib.sha256(source_authority_bytes).hexdigest()


def _with_self_hash(body: Mapping[str, Any]) -> Dict[str, Any]:
    record = dict(body)
    record["canonical_body_sha256"] = canonical_sha256(record)
    return record


def _validate_closed_record(
    record: object,
    source_authority_bytes: bytes,
    exact_terms: Mapping[str, Any],
) -> Dict[str, Any]:
    if type(record) is not dict:
        raise ValueError("approval must be a JSON object")
    expected_fields = set(exact_terms) | {
        "source_authority_sha256",
        "canonical_body_sha256",
    }
    actual_fields = set(record)
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        unknown = sorted(actual_fields - expected_fields)
        raise ValueError(
            f"approval schema mismatch: missing={missing}, unknown={unknown}"
        )
    for field, expected in exact_terms.items():
        actual = record[field]
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"{field} does not match approved authority")
    expected_source_sha = _source_sha256(source_authority_bytes)
    source_sha = record["source_authority_sha256"]
    if type(source_sha) is not str or source_sha != expected_source_sha:
        raise ValueError("source authority SHA-256 mismatch")
    body = dict(record)
    self_hash = body.pop("canonical_body_sha256")
    if type(self_hash) is not str or self_hash != canonical_sha256(body):
        raise ValueError("canonical body SHA-256 mismatch")
    return record


def build_production_support_plane_approval(
    source_authority_bytes: bytes,
) -> Dict[str, Any]:
    """Build the exact support-plane approval bound to its source bytes."""

    return _with_self_hash(
        {
            **_SUPPORT_TERMS,
            "source_authority_sha256": _source_sha256(source_authority_bytes),
        }
    )


def validate_production_support_plane_approval(
    record: object,
    source_authority_bytes: bytes,
) -> Dict[str, Any]:
    """Validate the exact closed support-plane approval record."""

    return _validate_closed_record(record, source_authority_bytes, _SUPPORT_TERMS)


def build_gpu_residual_liability_approval(
    source_authority_bytes: bytes,
) -> Dict[str, Any]:
    """Build the exact residual-liability approval bound to its source bytes."""

    return _with_self_hash(
        {
            **_RESIDUAL_TERMS,
            "source_authority_sha256": _source_sha256(source_authority_bytes),
        }
    )


def validate_gpu_residual_liability_approval(
    record: object,
    source_authority_bytes: bytes,
) -> Dict[str, Any]:
    """Validate the exact closed residual-liability approval record."""

    return _validate_closed_record(record, source_authority_bytes, _RESIDUAL_TERMS)
