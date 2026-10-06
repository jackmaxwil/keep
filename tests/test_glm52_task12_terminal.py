from __future__ import annotations

import pytest

from glm52_enforcement.canonical import canonical_sha256
from glm52_enforcement.records import RECORD_FIELDS, validate_record


SHA = "a" * 64
TS = "2026-07-29T12:00:00Z"


def _no_launch_terminal() -> dict[str, object]:
    record: dict[str, object] = {}
    arrays = {
        "final_ec2_states",
        "allocations",
        "worker_launch_evidence",
        "worker_launch_liabilities",
        "request_evidence",
    }
    nullable_objects = {
        "handoff",
        "binding",
        "final_heartbeat_identity",
        "checkpoint_identity",
        "cache_identity",
        "training_identity",
        "evaluation_identity",
        "drain_identity",
        "post_terminal_quiescence_evidence",
        "prior_terminal_v1_identity",
    }
    for field in RECORD_FIELDS["glm52_production_terminal_v2"]:
        if field == "schema_version":
            record[field] = 2
        elif field == "record_type":
            record[field] = "glm52_production_terminal_v2"
        elif field == "account_id":
            record[field] = "246813579024"
        elif field == "region":
            record[field] = "us-west-2"
        elif field == "run_id":
            record[field] = "glm52-sky-20260724"
        elif field == "activation_id":
            record[field] = "activation-1"
        elif field in {"activation_ordinal", "generation"}:
            record[field] = 1
        elif field == "generation_text":
            record[field] = "00000001"
        elif field in arrays:
            record[field] = []
        elif field in nullable_objects:
            record[field] = None
        elif field.endswith("_array_sha256"):
            record[field] = canonical_sha256([])
        elif field.endswith("_sha256"):
            record[field] = SHA
        elif field == "spend_ledger_head_identity":
            record[field] = {"identity": SHA}
        elif field == "terminal_observation_window":
            record[field] = {"identity": SHA}
        elif field == "remaining_approved_gpu_seconds":
            record[field] = 0
        elif field == "remaining_approved_gpu_usd":
            record[field] = "0.00"
        elif field in {"request_cardinality", "worker_cardinality"}:
            record[field] = "NOT_APPLICABLE" if field == "request_cardinality" else "ZERO"
        elif field == "outcome":
            record[field] = "ACTIVATION_ROLLOVER_DEPLOYMENT_FAILED_NO_LAUNCH"
        elif field == "operator_disposition_required":
            record[field] = True
        elif field == "created_at":
            record[field] = TS
        else:
            record[field] = "x"
    record["canonical_body_sha256"] = canonical_sha256(
        {key: value for key, value in record.items() if key != "canonical_body_sha256"}
    )
    return validate_record("glm52_production_terminal_v2", record)


def test_terminal_assembly_recomputes_hashes_cardinality_and_self_identity() -> None:
    from glm52_enforcement.task12_terminal import assemble_terminal_v2

    base = _no_launch_terminal()
    base["allocations_array_sha256"] = "b" * 64
    base["worker_cardinality"] = "MULTIPLE"
    base["canonical_body_sha256"] = "b" * 64

    terminal = assemble_terminal_v2(
        base_record=base,
        allocations=[],
        worker_launch_evidence=[],
        worker_launch_liabilities=[],
        request_evidence=[],
    )

    assert terminal["worker_cardinality"] == "ZERO"
    assert terminal["allocations_array_sha256"] == canonical_sha256([])
    assert validate_record("glm52_production_terminal_v2", terminal) == terminal


def test_terminal_assembly_rejects_late_allocation_instead_of_mutating_terminal() -> None:
    from glm52_enforcement.task12_terminal import (
        Task12TerminalError,
        assemble_terminal_v2,
    )

    with pytest.raises(Task12TerminalError, match="late allocation"):
        assemble_terminal_v2(
            base_record=_no_launch_terminal(),
            allocations=[],
            worker_launch_evidence=[],
            worker_launch_liabilities=[],
            request_evidence=[],
            late_allocations=[{"allocation_ordinal": 2, "instance_id": "i-late"}],
        )


def test_terminal_assembly_delegates_outcome_nullability_to_closed_record_validator() -> None:
    from glm52_enforcement.task12_terminal import Task12TerminalError, assemble_terminal_v2

    base = _no_launch_terminal()
    base["handoff"] = {"identity": SHA}

    with pytest.raises(Task12TerminalError, match="terminal-v2 record"):
        assemble_terminal_v2(
            base_record=base,
            allocations=[],
            worker_launch_evidence=[],
            worker_launch_liabilities=[],
            request_evidence=[],
        )
