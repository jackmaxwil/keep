"""Pure Task 12 terminal-v2 candidate assembly."""

from __future__ import annotations

from .canonical import canonical_sha256
from .records import RecordValidationError, validate_record


class Task12TerminalError(ValueError):
    """A terminal-v2 candidate is incomplete or tries to absorb late work."""


def _exact_array(value: object, label: str) -> list[object]:
    if type(value) is not list:
        raise Task12TerminalError(label + " must be an exact list")
    return list(value)


def assemble_terminal_v2(
    *,
    base_record: object,
    allocations: object,
    worker_launch_evidence: object,
    worker_launch_liabilities: object,
    request_evidence: object,
    late_allocations: object = (),
) -> dict[str, object]:
    """Recompute terminal arrays only; late allocations remain Task 9 records."""

    if type(base_record) is not dict:
        raise Task12TerminalError("terminal-v2 base record must be an exact mapping")
    if type(late_allocations) not in {tuple, list}:
        raise Task12TerminalError("late allocations must be a sequence")
    if late_allocations:
        raise Task12TerminalError(
            "late allocation must stay outside immutable terminal-v2"
        )
    record = dict(base_record)
    record["allocations"] = _exact_array(allocations, "terminal allocations")
    record["worker_launch_evidence"] = _exact_array(
        worker_launch_evidence, "worker-launch evidence"
    )
    record["worker_launch_liabilities"] = _exact_array(
        worker_launch_liabilities, "worker-launch liabilities"
    )
    record["request_evidence"] = _exact_array(request_evidence, "request evidence")
    record["allocations_array_sha256"] = canonical_sha256(record["allocations"])
    record["worker_launch_evidence_array_sha256"] = canonical_sha256(
        record["worker_launch_evidence"]
    )
    record["worker_launch_liabilities_array_sha256"] = canonical_sha256(
        record["worker_launch_liabilities"]
    )
    record["request_evidence_array_sha256"] = canonical_sha256(
        record["request_evidence"]
    )
    allocation_count = len(record["allocations"])
    record["worker_cardinality"] = (
        "ZERO" if allocation_count == 0 else ("ONE" if allocation_count == 1 else "MULTIPLE")
    )
    record["canonical_body_sha256"] = canonical_sha256(
        {
            name: value
            for name, value in record.items()
            if name != "canonical_body_sha256"
        }
    )
    try:
        return validate_record("glm52_production_terminal_v2", record)
    except RecordValidationError as exc:
        raise Task12TerminalError("terminal-v2 record is invalid: " + str(exc)) from exc


__all__ = ["Task12TerminalError", "assemble_terminal_v2"]
