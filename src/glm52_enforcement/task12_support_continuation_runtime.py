"""Live support-owned continuation after the synchronous Task 11 closure.

The retained Task 12 registry is deliberately descriptor-only.  This module
is the separate, versioned support runtime: every invocation materializes the
authenticated predecessor, performs exactly one named live operation, and
returns the complete successor state for Step Functions to persist.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .canonical import canonical_sha256
from .task12_correlation import (
    NumericJobBinding,
    RequestCorrelation,
    Task11HandoffAuthority,
    resolve_numeric_binding,
)
from .task12_support_continuation_adapter import (
    SupportContinuationContractError,
    SupportContinuationInvocation,
    build_support_result,
    parse_support_invocation,
)

_SHA = re.compile(r"[0-9a-f]{64}\Z")


class SupportContinuationRuntimeError(ValueError):
    """The live support continuation lost typed current-state custody."""


class SupportContinuationIncomplete(RuntimeError):
    """A bounded live observation is incomplete and must be re-entered."""

    def __init__(self, evidence: Mapping[str, object]) -> None:
        if type(evidence) is not dict:
            raise TypeError("incomplete support evidence must be exact")
        self.evidence = dict(evidence)
        super().__init__("support continuation observation is incomplete")


@dataclass(frozen=True)
class SupportEffect:
    """Authenticated evidence returned by one concrete live effect."""

    effect_kind: str
    outcome: str
    evidence: Mapping[str, object]
    canonical_identity_sha256: str


@dataclass(frozen=True)
class ZeroRequestProof:
    """Positive exhaustive-zero evidence; never a projected absence claim."""

    handoff_identity_sha256: str
    pagination_complete: bool
    first_observed_at: str
    second_observed_at: str
    request_ids: tuple[str, ...]
    job_ids: tuple[str, ...]
    worker_instance_ids: tuple[str, ...]
    allocation_ids: tuple[str, ...]
    controller_work_ids: tuple[str, ...]
    canonical_identity_sha256: str


@dataclass(frozen=True)
class SupportContinuationState:
    """Complete state threaded through the versioned support workflow."""

    phase: str
    task11_context: Mapping[str, object]
    handoff: Task11HandoffAuthority
    correlation_tuple: Mapping[str, object] | None
    correlation: RequestCorrelation | None
    request_cardinality: str | None
    binding: NumericJobBinding | None
    zero_request_proof: ZeroRequestProof | None
    effects: tuple[SupportEffect, ...]
    canonical_identity_sha256: str


class SupportContinuationServices(Protocol):
    """Concrete live boundaries required by the support continuation."""

    def hydrate(self, state: SupportContinuationState) -> None:
        ...

    def read_approved_task11_handoff(
        self, context: object
    ) -> Task11HandoffAuthority:
        ...

    def materialize_correlation_tuple(
        self, authority: Task11HandoffAuthority
    ) -> Mapping[str, object]:
        ...

    def correlate_request(
        self,
        authority: Task11HandoffAuthority,
        correlation_tuple: object,
    ) -> RequestCorrelation:
        ...

    def prove_zero_request(
        self,
        authority: Task11HandoffAuthority,
        correlation: RequestCorrelation,
    ) -> ZeroRequestProof:
        ...

    def read_numeric_binding(self, request_id: str) -> object:
        ...

    def observe_runtime(
        self, state: SupportContinuationState
    ) -> SupportEffect:
        ...

    def arm_drain(self, state: SupportContinuationState) -> SupportEffect:
        ...

    def request_drain(
        self, state: SupportContinuationState
    ) -> SupportEffect:
        ...

    def reconcile_request_cancel(
        self, state: SupportContinuationState
    ) -> SupportEffect:
        ...

    def reconcile_job_cancel(
        self, state: SupportContinuationState
    ) -> SupportEffect:
        ...

    def quiesce_controller(
        self, state: SupportContinuationState
    ) -> SupportEffect:
        ...

    def reconcile_workers(
        self, state: SupportContinuationState
    ) -> SupportEffect:
        ...

    def close_allocations(
        self, state: SupportContinuationState
    ) -> SupportEffect:
        ...

    def commit_allocation_closed(
        self, state: SupportContinuationState
    ) -> None:
        ...

    def create_terminal_v2(
        self, state: SupportContinuationState
    ) -> SupportEffect:
        ...

    def request_finalization(
        self, state: SupportContinuationState
    ) -> SupportEffect:
        ...


def _fail(message: str) -> None:
    raise SupportContinuationRuntimeError(message)


def _identity_body(value: object) -> dict[str, object]:
    body = asdict(value)
    body.pop("canonical_identity_sha256", None)
    return body


def _validate_effect(
    value: object, expected_kind: str
) -> SupportEffect:
    if type(value) is not SupportEffect:
        _fail(expected_kind + " did not return typed live evidence")
    if (
        value.effect_kind != expected_kind
        or value.outcome != "SUCCEEDED"
        or type(value.evidence) is not dict
        or _SHA.fullmatch(value.canonical_identity_sha256) is None
        or value.canonical_identity_sha256
        != canonical_sha256(_identity_body(value))
    ):
        _fail(expected_kind + " live evidence drifted")
    return value


def _validate_zero_request(value: object) -> ZeroRequestProof:
    if type(value) is not ZeroRequestProof:
        _fail("exhaustive zero-request proof is not typed")
    if (
        _SHA.fullmatch(value.handoff_identity_sha256) is None
        or value.pagination_complete is not True
        or not value.first_observed_at
        or not value.second_observed_at
        or value.second_observed_at <= value.first_observed_at
        or any(
            type(items) is not tuple or items
            for items in (
                value.request_ids,
                value.job_ids,
                value.worker_instance_ids,
                value.allocation_ids,
                value.controller_work_ids,
            )
        )
        or value.canonical_identity_sha256
        != canonical_sha256(_identity_body(value))
    ):
        _fail("exhaustive zero-request proof drifted")
    return value


def _state_body(
    *,
    phase: str,
    task11_context: Mapping[str, object],
    handoff: Task11HandoffAuthority,
    correlation_tuple: Mapping[str, object] | None,
    correlation: RequestCorrelation | None,
    request_cardinality: str | None,
    binding: NumericJobBinding | None,
    zero_request_proof: ZeroRequestProof | None,
    effects: tuple[SupportEffect, ...],
) -> dict[str, object]:
    return {
        "phase": phase,
        "task11_context": dict(task11_context),
        "handoff": asdict(handoff),
        "correlation_tuple": (
            None
            if correlation_tuple is None
            else dict(correlation_tuple)
        ),
        "correlation": (
            None if correlation is None else asdict(correlation)
        ),
        "request_cardinality": request_cardinality,
        "binding": None if binding is None else asdict(binding),
        "zero_request_proof": (
            None
            if zero_request_proof is None
            else asdict(zero_request_proof)
        ),
        "effects": [asdict(item) for item in effects],
    }


def _make_state(
    *,
    phase: str,
    task11_context: Mapping[str, object],
    handoff: Task11HandoffAuthority,
    correlation_tuple: Mapping[str, object] | None = None,
    correlation: RequestCorrelation | None = None,
    request_cardinality: str | None = None,
    binding: NumericJobBinding | None = None,
    zero_request_proof: ZeroRequestProof | None = None,
    effects: tuple[SupportEffect, ...] = (),
) -> SupportContinuationState:
    body = _state_body(
        phase=phase,
        task11_context=task11_context,
        handoff=handoff,
        correlation_tuple=correlation_tuple,
        correlation=correlation,
        request_cardinality=request_cardinality,
        binding=binding,
        zero_request_proof=zero_request_proof,
        effects=effects,
    )
    return SupportContinuationState(
        phase=phase,
        task11_context=dict(task11_context),
        handoff=handoff,
        correlation_tuple=(
            None
            if correlation_tuple is None
            else dict(correlation_tuple)
        ),
        correlation=correlation,
        request_cardinality=request_cardinality,
        binding=binding,
        zero_request_proof=zero_request_proof,
        effects=effects,
        canonical_identity_sha256=canonical_sha256(body),
    )


def _mapping_to_handoff(value: object) -> Task11HandoffAuthority:
    if type(value) is not dict:
        _fail("support handoff state is absent")
    try:
        return Task11HandoffAuthority(**value)
    except TypeError as exc:
        raise SupportContinuationRuntimeError(
            "support handoff state drifted"
        ) from exc


def _mapping_to_correlation(value: object) -> RequestCorrelation | None:
    if value is None:
        return None
    if type(value) is not dict:
        _fail("support correlation state drifted")
    matches = value.get("matches")
    if type(matches) not in {list, tuple}:
        _fail("support correlation matches drifted")
    from .task12_correlation import CorrelatedRequest

    try:
        return RequestCorrelation(
            kind=value["kind"],
            matches=tuple(CorrelatedRequest(**item) for item in matches),
        )
    except (KeyError, TypeError) as exc:
        raise SupportContinuationRuntimeError(
            "support correlation state drifted"
        ) from exc


def _mapping_to_optional(
    cls: type[Any], value: object, label: str
) -> object:
    if value is None:
        return None
    if type(value) is not dict:
        _fail(label + " state drifted")
    try:
        return cls(**value)
    except TypeError as exc:
        raise SupportContinuationRuntimeError(
            label + " state drifted"
        ) from exc


def _state_from_predecessor(
    invocation: SupportContinuationInvocation,
) -> SupportContinuationState:
    prior = invocation.operation_input.get("support_last_result")
    try:
        value = prior["result"]["continuation_state"]
    except (KeyError, TypeError) as exc:
        raise SupportContinuationRuntimeError(
            "support predecessor state is absent"
        ) from exc
    if type(value) is not dict:
        _fail("support predecessor state is not an object")
    effects_value = value.get("effects")
    if type(effects_value) not in {list, tuple}:
        _fail("support predecessor effects drifted")
    try:
        effects = tuple(SupportEffect(**item) for item in effects_value)
        for effect in effects:
            _validate_effect(effect, effect.effect_kind)
        state = SupportContinuationState(
            phase=value["phase"],
            task11_context=value["task11_context"],
            handoff=_mapping_to_handoff(value["handoff"]),
            correlation_tuple=value["correlation_tuple"],
            correlation=_mapping_to_correlation(value["correlation"]),
            request_cardinality=value["request_cardinality"],
            binding=_mapping_to_optional(
                NumericJobBinding, value["binding"], "numeric binding"
            ),
            zero_request_proof=_mapping_to_optional(
                ZeroRequestProof,
                value["zero_request_proof"],
                "zero-request proof",
            ),
            effects=effects,
            canonical_identity_sha256=value[
                "canonical_identity_sha256"
            ],
        )
    except (KeyError, TypeError) as exc:
        raise SupportContinuationRuntimeError(
            "support predecessor state field set drifted"
        ) from exc
    if (
        type(state.task11_context) is not dict
        or state.request_cardinality
        not in {None, "ZERO", "ONE", "MULTIPLE"}
        or state.canonical_identity_sha256
        != canonical_sha256(
            _state_body(
                phase=state.phase,
                task11_context=state.task11_context,
                handoff=state.handoff,
                correlation_tuple=state.correlation_tuple,
                correlation=state.correlation,
                request_cardinality=state.request_cardinality,
                binding=state.binding,
                zero_request_proof=state.zero_request_proof,
                effects=state.effects,
            )
        )
    ):
        _fail("support predecessor state identity drifted")
    return state


def _successor(
    state: SupportContinuationState,
    *,
    phase: str,
    effect: SupportEffect | None = None,
    **changes: object,
) -> SupportContinuationState:
    values: dict[str, object] = {
        "phase": phase,
        "task11_context": state.task11_context,
        "handoff": state.handoff,
        "correlation_tuple": state.correlation_tuple,
        "correlation": state.correlation,
        "request_cardinality": state.request_cardinality,
        "binding": state.binding,
        "zero_request_proof": state.zero_request_proof,
        "effects": (
            state.effects
            if effect is None
            else state.effects + (effect,)
        ),
    }
    values.update(changes)
    return _make_state(**values)


def _root_state(
    invocation: SupportContinuationInvocation,
    services: SupportContinuationServices,
) -> SupportContinuationState:
    if set(invocation.operation_input) != {"task11_context"}:
        _fail("support continuation root input drifted")
    context = invocation.operation_input["task11_context"]
    if (
        type(context) is not dict
        or type(context.get("closure_request")) is not dict
        or type(context.get("decision_result")) is not dict
        or context["decision_result"].get("status")
        not in {"CLOSED", "NUMERIC_BINDING_RECONCILIATION"}
        or type(context.get("task11_execution_arn")) is not str
        or not context["task11_execution_arn"]
    ):
        _fail("approved Task 11 context is incomplete")
    handoff = services.read_approved_task11_handoff(context)
    if type(handoff) is not Task11HandoffAuthority:
        _fail("approved Task 11 handoff is not typed")
    return _make_state(
        phase="HANDOFF_VALIDATED",
        task11_context=context,
        handoff=handoff,
    )


def execute_support_operation(
    event: object,
    *,
    services: SupportContinuationServices,
) -> dict[str, object]:
    """Execute exactly one live support operation and return its successor."""

    invocation = parse_support_invocation(event)
    operation = invocation.operation_kind
    if operation == "VALIDATE_APPROVED_TASK11_HANDOFF":
        state = _root_state(invocation, services)
    else:
        state = _state_from_predecessor(invocation)
        hydrate = getattr(services, "hydrate", None)
        if callable(hydrate):
            hydrate(state)
        if operation == "NUMERIC_BINDING_HANDOFF":
            correlation_tuple = services.materialize_correlation_tuple(
                state.handoff
            )
            if type(correlation_tuple) is not dict:
                _fail("numeric-binding handoff tuple is not exact")
            state = _successor(
                state,
                phase="NUMERIC_BINDING_HANDOFF",
                correlation_tuple=correlation_tuple,
            )
        elif operation == "BIND_NUMERIC_JOB_OR_PROVE_NONE":
            if state.correlation_tuple is None:
                _fail("numeric binding lacks its immutable tuple")
            correlation = services.correlate_request(
                state.handoff,
                state.correlation_tuple,
            )
            if (
                type(correlation) is not RequestCorrelation
                or correlation.kind not in {"ZERO", "ONE", "MULTIPLE"}
            ):
                _fail("numeric binding correlation is untyped")
            if correlation.kind == "MULTIPLE":
                _fail(
                    "multiple request correlation is a sealed incident; "
                    "no cancellation mutation is authorized"
                )
            binding = None
            zero_request_proof = None
            if correlation.kind == "ZERO":
                zero_request_proof = _validate_zero_request(
                    services.prove_zero_request(
                        state.handoff,
                        correlation,
                    )
                )
                if (
                    zero_request_proof.handoff_identity_sha256
                    != state.handoff.body_sha256
                ):
                    _fail("zero-request proof is for a foreign handoff")
            elif correlation.kind == "ONE":
                binding = resolve_numeric_binding(
                    correlation=correlation,
                    expected_sky_job_name=state.handoff.body[
                        "expected_sky_job_name"
                    ],
                    reader=services.read_numeric_binding,
                )
            state = _successor(
                state,
                phase="NUMERIC_BOUND_OR_NONE",
                correlation=correlation,
                request_cardinality=correlation.kind,
                binding=binding,
                zero_request_proof=zero_request_proof,
            )
        elif operation == "RUNTIME_OBSERVATION":
            state = _successor(
                state,
                phase="RUNTIME_OBSERVED",
                effect=_validate_effect(
                    services.observe_runtime(state),
                    "RUNTIME_OBSERVATION",
                ),
            )
        elif operation == "DRAIN_ARMED":
            state = _successor(
                state,
                phase="DRAIN_ARMED",
                effect=_validate_effect(
                    services.arm_drain(state), "DRAIN_ARMED"
                ),
            )
        elif operation == "DRAIN_REQUESTED":
            state = _successor(
                state,
                phase="DRAIN_REQUESTED",
                effect=_validate_effect(
                    services.request_drain(state), "DRAIN_REQUESTED"
                ),
            )
        elif operation == "SUPPORT_REQUEST_CANCEL_RECONCILIATION":
            effect = (
                None
                if state.request_cardinality == "ZERO"
                else _validate_effect(
                    services.reconcile_request_cancel(state),
                    "REQUEST_CANCEL",
                )
            )
            state = _successor(
                state,
                phase=(
                    "REQUEST_CANCEL_NOT_APPLICABLE"
                    if effect is None
                    else "REQUEST_CANCEL_TERMINAL"
                ),
                effect=effect,
            )
        elif operation == "SUPPORT_JOB_CANCEL_RECONCILIATION":
            effect = (
                None
                if state.binding is None
                else _validate_effect(
                    services.reconcile_job_cancel(state),
                    "JOB_CANCEL",
                )
            )
            state = _successor(
                state,
                phase=(
                    "JOB_CANCEL_NOT_APPLICABLE"
                    if effect is None
                    else "JOB_CANCEL_TERMINAL"
                ),
                effect=effect,
            )
        elif operation == (
            "SUPPORT_CONTROLLER_QUIESCE_BEFORE_FORCE_IF_REQUIRED"
        ):
            state = _successor(
                state,
                phase="CONTROLLER_QUIESCED",
                effect=_validate_effect(
                    services.quiesce_controller(state),
                    "CONTROLLER_QUIESCED",
                ),
            )
        elif operation == "WORKER_TERMINAL_RECONCILIATION":
            state = _successor(
                state,
                phase="WORKER_TERMINAL",
                effect=_validate_effect(
                    services.reconcile_workers(state),
                    "WORKER_TERMINAL",
                ),
            )
        elif operation == "ALLOCATION_CLOSED":
            state = _successor(
                state,
                phase="ALLOCATION_CLOSED",
                effect=_validate_effect(
                    services.close_allocations(state),
                    "ALLOCATION_CLOSED",
                ),
            )
            services.commit_allocation_closed(state)
        elif operation == (
            "CREATE_OR_RECONCILE_PRODUCTION_TERMINAL_V2"
        ):
            try:
                terminal_effect = services.create_terminal_v2(state)
            except SupportContinuationIncomplete as incomplete:
                body = {
                    "schema_version": 1,
                    "record_type": (
                        "glm52_task12_support_continuation_result_v1"
                    ),
                    "operation_kind": operation,
                    "outcome": "INCOMPLETE",
                    "result": {
                        "continuation_state": asdict(state),
                        "incomplete_evidence": incomplete.evidence,
                    },
                }
                return {
                    **body,
                    "canonical_body_sha256": canonical_sha256(body),
                }
            state = _successor(
                state,
                phase="TERMINAL_V2_PUBLISHED",
                effect=_validate_effect(
                    terminal_effect,
                    "TERMINAL_V2",
                ),
            )
        elif operation == "REQUEST_SUPPORT_FINALIZATION":
            state = _successor(
                state,
                phase="FINALIZATION_REQUESTED",
                effect=_validate_effect(
                    services.request_finalization(state),
                    "FINALIZATION_REQUESTED",
                ),
            )
        else:  # pragma: no cover - parser owns the closed operation set
            raise SupportContinuationContractError(
                "support continuation operation is unknown"
            )
    return build_support_result(
        invocation=invocation,
        result={"continuation_state": asdict(state)},
    )


def main(
    event: object,
    context: object,
    *,
    services: SupportContinuationServices | None = None,
) -> Mapping[str, object]:
    """Published Lambda entrypoint for the support continuation."""

    if services is None:
        from .task12_support_continuation_aws import (
            build_aws_support_continuation_services,
        )

        services = build_aws_support_continuation_services(context=context)
    return execute_support_operation(event, services=services)


__all__ = [
    "SupportContinuationIncomplete",
    "SupportContinuationRuntimeError",
    "SupportContinuationServices",
    "SupportContinuationState",
    "SupportEffect",
    "ZeroRequestProof",
    "execute_support_operation",
    "main",
]
