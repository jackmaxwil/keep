"""Task 11 measured no-POST collector and verifier."""

from __future__ import annotations

from dataclasses import replace

import pytest

from glm52_enforcement.decision_closure import (
    CLOSURE_PHASE_CEILINGS,
    SUFFIX_PHASE_CEILINGS,
    PhaseSpan,
    build_rehearsal_measurement,
    verify_no_post_rehearsals,
)


PATH_PROOFS = (
    "EXACT_PRODUCTION_CLIENTS",
    "FULL_VERSION_PAGINATION",
    "NON_VPC_DECISION_PATH",
    "ISOLATED_ADMISSION_ATTESTATION_PATH",
    "HOST_NAT_PATH",
    "FROZEN_NAMESPACE_SCALE",
    "CONDITIONAL_WRITE",
    "DIRECT_RESPONSE",
    "EXACT_GET_HEAD",
    "H1E_MODELED_VALIDATION",
)
ZERO_EFFECTS = (
    ("PRODUCTION_SOURCE", 0),
    ("PRODUCTION_CLAIM", 0),
    ("PRODUCTION_DECISION", 0),
    ("PRODUCTION_ACTION_CONSUME", 0),
    ("SKY_POST", 0),
)


def _spans(
    names_and_durations: tuple[tuple[str, int], ...],
    *,
    start: int = 0,
) -> tuple[PhaseSpan, ...]:
    result = []
    cursor = start
    for name, duration in names_and_durations:
        result.append(PhaseSpan(name, cursor, cursor + duration))
        cursor += duration
    return tuple(result)


def _timing() -> tuple[tuple[PhaseSpan, ...], tuple[PhaseSpan, ...]]:
    suffix_durations = (1, 1, 10, 1, 1, 1, 1, 1)
    closure_durations = (10, 10, 10, 10, sum(suffix_durations), 10, 10, 10)
    closure = _spans(
        tuple(zip(CLOSURE_PHASE_CEILINGS, closure_durations))
    )
    suffix = _spans(
        tuple(zip(SUFFIX_PHASE_CEILINGS, suffix_durations)),
        start=closure[4].started_monotonic_seconds,
    )
    return closure, suffix


def _measurement(index: int, **changes: object):
    closure, suffix = _timing()
    faults = ("THROTTLING", "PAGINATION", "NETWORK_AMBIGUITY")
    values = {
        "rehearsal_id": "rehearsal-" + str(index),
        "provenance": "LOCAL_SIMULATOR",
        "lambda_environment_id": "environment-" + str(index),
        "cold_start": index < 5,
        "path_proofs": PATH_PROOFS,
        "canary_bucket": "keep-glm52-h1g-rehearsal",
        "canary_key": "rehearsal/canary/" + str(index) + "/decision.json",
        "canary_can_satisfy_production_authority": False,
        "canary_worker_readable": False,
        "canary_admission_readable": False,
        "effect_counts": ZERO_EFFECTS,
        "relay_call_count": 0,
        "closure_spans": closure,
        "suffix_spans": suffix,
        "first_policy_readback_monotonic_seconds": (
            suffix[2].started_monotonic_seconds
        ),
        "second_policy_readback_monotonic_seconds": (
            suffix[2].started_monotonic_seconds + 10
        ),
        "injected_failure_kind": (
            faults[index] if index < len(faults) else "NONE"
        ),
        "failure_unwind_seconds": 10 if index < len(faults) else 0,
    }
    values.update(changes)
    return build_rehearsal_measurement(**values)


def _valid_measurements():
    return tuple(_measurement(index) for index in range(20))


def test_task11_red_local_twenty_run_gate_is_validated_but_unproven() -> None:
    gate = verify_no_post_rehearsals(_valid_measurements())
    assert gate.status == "CLOSURE_BUDGET_UNPROVEN"
    assert gate.local_shape_verified is True
    assert gate.measurement_count == 20
    assert gate.cold_environment_count == 5
    assert gate.worst_closure_seconds == 87
    assert gate.worst_suffix_seconds == 17
    assert gate.production_effect_count == 0
    assert gate.failure_kinds == (
        "THROTTLING",
        "PAGINATION",
        "NETWORK_AMBIGUITY",
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "nineteen",
        "four_cold",
        "reused_cold_environment",
        "duplicate_id",
        "missing_failure_kind",
        "missing_path",
        "production_authority",
        "worker_readable",
        "admission_readable",
        "production_effect",
        "relay_call",
        "unwind_overrun",
        "closure_overrun",
        "suffix_overrun",
    ),
)
def test_task11_red_malformed_measurement_matrix_fails_closed(
    mutation: str,
) -> None:
    rows = list(_valid_measurements())
    if mutation == "nineteen":
        rows.pop()
    elif mutation == "four_cold":
        rows[4] = replace(rows[4], cold_start=False)
    elif mutation == "reused_cold_environment":
        rows[4] = replace(
            rows[4],
            lambda_environment_id=rows[3].lambda_environment_id,
        )
    elif mutation == "duplicate_id":
        rows[19] = replace(rows[19], rehearsal_id=rows[18].rehearsal_id)
    elif mutation == "missing_failure_kind":
        rows[2] = replace(rows[2], injected_failure_kind="NONE")
    elif mutation == "missing_path":
        rows[10] = replace(rows[10], path_proofs=PATH_PROOFS[:-1])
    elif mutation == "production_authority":
        rows[10] = replace(
            rows[10],
            canary_can_satisfy_production_authority=True,
        )
    elif mutation == "worker_readable":
        rows[10] = replace(rows[10], canary_worker_readable=True)
    elif mutation == "admission_readable":
        rows[10] = replace(rows[10], canary_admission_readable=True)
    elif mutation == "production_effect":
        rows[10] = replace(
            rows[10],
            effect_counts=(*ZERO_EFFECTS[:-1], ("SKY_POST", 1)),
        )
    elif mutation == "relay_call":
        rows[10] = replace(rows[10], relay_call_count=1)
    elif mutation == "unwind_overrun":
        rows[0] = replace(rows[0], failure_unwind_seconds=56)
    elif mutation == "closure_overrun":
        closure = list(rows[10].closure_spans)
        closure[0] = replace(
            closure[0],
            ended_monotonic_seconds=(
                closure[0].started_monotonic_seconds
                + CLOSURE_PHASE_CEILINGS[closure[0].phase_name]
                + 1
            ),
        )
        rows[10] = replace(rows[10], closure_spans=tuple(closure))
    else:
        suffix = list(rows[10].suffix_spans)
        suffix[0] = replace(
            suffix[0],
            ended_monotonic_seconds=(
                suffix[0].started_monotonic_seconds
                + SUFFIX_PHASE_CEILINGS[suffix[0].phase_name]
                + 1
            ),
        )
        rows[10] = replace(rows[10], suffix_spans=tuple(suffix))

    with pytest.raises(ValueError):
        verify_no_post_rehearsals(tuple(rows))
