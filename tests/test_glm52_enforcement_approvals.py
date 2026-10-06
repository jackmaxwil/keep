from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from glm52_enforcement.approvals import (
    build_gpu_residual_liability_approval,
    build_production_support_plane_approval,
    validate_gpu_residual_liability_approval,
    validate_production_support_plane_approval,
)
from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256


SOURCE_BYTES = b"tracked owner approval receipt\n"
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_AUTHORITY = (
    REPO_ROOT
    / "docs/superpowers/approvals/2026-07-28-glm52-campaign-owner-approvals.md"
)
BUILD_APPROVALS_SCRIPT = (
    REPO_ROOT / "aws/glm52-gpu/scripts/build_campaign_approvals.py"
)

SUPPORT_EXACT_TERM_FIELDS = (
    "schema_version",
    "record_type",
    "approval_status",
    "approval_principal_role",
    "source_authority_path",
    "approval_received_date",
    "work_deadline_hours",
    "delete_request_deadline_hours",
    "cost_incident_deadline_hours",
    "host_nat_processed_gib_limit",
    "estimated_support_plane_ceiling_usd",
    "workflow_event_limit",
    "forensic_snapshot_retention_days",
    "liability_scan_interval_seconds",
    "liability_scan_max_count",
    "liability_logs_gib_limit",
    "liability_logs_retention_days",
    "estimated_liability_watcher_ceiling_usd",
    "exact_topology_and_cardinalities_approved",
    "network_and_storage_plan_approved",
    "logs_invocations_workflow_history_retention_approved",
    "eight_secret_inventory_approved",
    "endpoint_plan_approved",
    "dated_price_model_identity_approved_and_complete",
)

RESIDUAL_EXACT_TERM_FIELDS = (
    "schema_version",
    "record_type",
    "approval_status",
    "approval_principal_role",
    "source_authority_path",
    "approval_received_date",
    "gpu_reserve_seconds",
    "gpu_reserve_usd",
    "root_volume_gib",
    "root_volume_tail_usd_max",
    "gpu_approval_hours_total",
    "gpu_approval_usd_total",
    "delayed_visibility_accepted",
    "delayed_control_accepted",
    "delayed_termination_accepted",
    "same_token_multiplicity_accepted",
    "termination_only_continuation_until_settlement_accepted",
    "no_hard_post_acceptance_aws_billing_cap_accepted",
    "knowing_overage_risk_accepted",
)


def test_canonical_json_bytes_are_deterministic_utf8_and_hash_exactly() -> None:
    value = {
        "z": None,
        "unicode": "café",
        "ordered": [3, True, False],
        "a": {"two": 2, "one": 1},
    }
    expected = (
        b'{"a":{"one":1,"two":2},"ordered":[3,true,false],'
        b'"unicode":"caf\xc3\xa9","z":null}'
    )

    assert canonical_json_bytes(value) == expected
    assert canonical_sha256(value) == hashlib.sha256(expected).hexdigest()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_non_finite_numbers(value: float) -> None:
    with pytest.raises(ValueError, match="non-finite number"):
        canonical_json_bytes({"value": value})


def test_support_builder_binds_source_and_every_approved_term() -> None:
    approval = build_production_support_plane_approval(SOURCE_BYTES)

    assert approval == validate_production_support_plane_approval(
        approval, SOURCE_BYTES
    )
    assert approval["source_authority_sha256"] == hashlib.sha256(
        SOURCE_BYTES
    ).hexdigest()
    assert {
        key: approval[key]
        for key in (
            "work_deadline_hours",
            "delete_request_deadline_hours",
            "cost_incident_deadline_hours",
            "host_nat_processed_gib_limit",
            "estimated_support_plane_ceiling_usd",
            "workflow_event_limit",
            "forensic_snapshot_retention_days",
            "liability_scan_interval_seconds",
            "liability_scan_max_count",
            "liability_logs_gib_limit",
            "liability_logs_retention_days",
            "estimated_liability_watcher_ceiling_usd",
        )
    } == {
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
    }
    assert {
        key: approval[key]
        for key in (
            "exact_topology_and_cardinalities_approved",
            "network_and_storage_plan_approved",
            "logs_invocations_workflow_history_retention_approved",
            "eight_secret_inventory_approved",
            "endpoint_plan_approved",
            "dated_price_model_identity_approved_and_complete",
        )
    } == {
        "exact_topology_and_cardinalities_approved": True,
        "network_and_storage_plan_approved": True,
        "logs_invocations_workflow_history_retention_approved": True,
        "eight_secret_inventory_approved": True,
        "endpoint_plan_approved": True,
        "dated_price_model_identity_approved_and_complete": True,
    }


def test_residual_builder_binds_every_numeric_and_accepted_liability_term() -> None:
    approval = build_gpu_residual_liability_approval(SOURCE_BYTES)

    assert approval == validate_gpu_residual_liability_approval(
        approval, SOURCE_BYTES
    )
    assert {
        key: approval[key]
        for key in (
            "gpu_reserve_seconds",
            "gpu_reserve_usd",
            "root_volume_gib",
            "root_volume_tail_usd_max",
            "gpu_approval_hours_total",
            "gpu_approval_usd_total",
        )
    } == {
        "gpu_reserve_seconds": 900,
        "gpu_reserve_usd": "13.76",
        "root_volume_gib": 300,
        "root_volume_tail_usd_max": "0.01",
        "gpu_approval_hours_total": "24.00",
        "gpu_approval_usd_total": "1320.96",
    }
    assert {
        key: approval[key]
        for key in (
            "delayed_visibility_accepted",
            "delayed_control_accepted",
            "delayed_termination_accepted",
            "same_token_multiplicity_accepted",
            "termination_only_continuation_until_settlement_accepted",
            "no_hard_post_acceptance_aws_billing_cap_accepted",
            "knowing_overage_risk_accepted",
        )
    } == {
        "delayed_visibility_accepted": True,
        "delayed_control_accepted": True,
        "delayed_termination_accepted": True,
        "same_token_multiplicity_accepted": True,
        "termination_only_continuation_until_settlement_accepted": True,
        "no_hard_post_acceptance_aws_billing_cap_accepted": True,
        "knowing_overage_risk_accepted": True,
    }


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_closed_approval_schema_rejects_missing_and_unknown_fields(
    mutation: str,
) -> None:
    approval = build_production_support_plane_approval(SOURCE_BYTES)
    if mutation == "missing":
        del approval["work_deadline_hours"]
    else:
        approval["unapproved"] = True

    with pytest.raises(ValueError, match="schema mismatch"):
        validate_production_support_plane_approval(approval, SOURCE_BYTES)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("account_id", "000000000000"),
        ("region", "us-east-1"),
        ("run_id", "other-run"),
        ("approval_principal_identity", "other@example.com"),
        ("source_transport", "email"),
    ],
)
def test_validator_rejects_wrong_campaign_authority_coordinates(
    field: str, value: str
) -> None:
    approval = build_production_support_plane_approval(SOURCE_BYTES)
    approval[field] = value

    with pytest.raises(ValueError, match=field):
        validate_production_support_plane_approval(approval, SOURCE_BYTES)


@pytest.mark.parametrize(
    ("record_kind", "field"),
    [
        *(("support", field) for field in SUPPORT_EXACT_TERM_FIELDS),
        *(("residual", field) for field in RESIDUAL_EXACT_TERM_FIELDS),
    ],
)
def test_validators_reject_every_changed_or_type_confused_exact_term(
    record_kind: str, field: str
) -> None:
    if record_kind == "support":
        approval = build_production_support_plane_approval(SOURCE_BYTES)
        validator = validate_production_support_plane_approval
    else:
        approval = build_gpu_residual_liability_approval(SOURCE_BYTES)
        validator = validate_gpu_residual_liability_approval
    original = approval[field]
    if type(original) is int:
        approval[field] = True
    elif type(original) is bool:
        approval[field] = False
    else:
        approval[field] = f"{original}-changed"

    with pytest.raises(ValueError, match=field):
        validator(approval, SOURCE_BYTES)


def test_residual_validator_rejects_stale_13_77_with_valid_self_hash() -> None:
    approval = build_gpu_residual_liability_approval(SOURCE_BYTES)
    approval["gpu_reserve_usd"] = "13.77"
    body = dict(approval)
    body.pop("canonical_body_sha256")
    approval["canonical_body_sha256"] = canonical_sha256(body)

    with pytest.raises(ValueError, match="gpu_reserve_usd"):
        validate_gpu_residual_liability_approval(approval, SOURCE_BYTES)


@pytest.mark.parametrize("tamper", ["receipt-bytes", "bound-hash"])
def test_validator_rejects_tampered_source_authority_hash(tamper: str) -> None:
    approval = build_production_support_plane_approval(SOURCE_BYTES)
    source = SOURCE_BYTES
    if tamper == "receipt-bytes":
        source += b"tampered"
    else:
        approval["source_authority_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="source authority SHA-256 mismatch"):
        validate_production_support_plane_approval(approval, source)


def test_validator_rejects_tampered_self_hash() -> None:
    approval = build_gpu_residual_liability_approval(SOURCE_BYTES)
    approval["canonical_body_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="canonical body SHA-256 mismatch"):
        validate_gpu_residual_liability_approval(approval, SOURCE_BYTES)


def test_cli_writes_valid_records_and_manifest_hashes_exact_file_bytes(
    tmp_path: Path,
) -> None:
    support_path = tmp_path / "PRODUCTION_SUPPORT_PLANE_APPROVAL.json"
    residual_path = tmp_path / "GPU_RESIDUAL_LIABILITY_APPROVAL.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(BUILD_APPROVALS_SCRIPT),
            "--support-output",
            str(support_path),
            "--residual-output",
            str(residual_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    source_bytes = SOURCE_AUTHORITY.read_bytes()
    support_bytes = support_path.read_bytes()
    residual_bytes = residual_path.read_bytes()
    validate_production_support_plane_approval(
        json.loads(support_bytes), source_bytes
    )
    validate_gpu_residual_liability_approval(
        json.loads(residual_bytes), source_bytes
    )
    assert json.loads(completed.stdout) == {
        "outputs": [
            {
                "path": str(support_path),
                "byte_size": len(support_bytes),
                "sha256": hashlib.sha256(support_bytes).hexdigest(),
            },
            {
                "path": str(residual_path),
                "byte_size": len(residual_bytes),
                "sha256": hashlib.sha256(residual_bytes).hexdigest(),
            },
        ]
    }


@pytest.mark.parametrize("existing_kind", ["support", "residual"])
def test_cli_refuses_any_overwrite_without_creating_the_other_output(
    tmp_path: Path, existing_kind: str
) -> None:
    support_path = tmp_path / "support.json"
    residual_path = tmp_path / "residual.json"
    existing_path = support_path if existing_kind == "support" else residual_path
    other_path = residual_path if existing_kind == "support" else support_path
    sentinel = b"do not overwrite\n"
    existing_path.write_bytes(sentinel)

    completed = subprocess.run(
        [
            sys.executable,
            str(BUILD_APPROVALS_SCRIPT),
            "--support-output",
            str(support_path),
            "--residual-output",
            str(residual_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "refusing to overwrite" in completed.stderr
    assert existing_path.read_bytes() == sentinel
    assert not other_path.exists()
