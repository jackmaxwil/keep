"""Typed, identity-pinned invocation envelope for Task 12 coordinators."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Mapping, Optional, TypeVar


class Task12HandlerError(ValueError):
    """Raised when an event is not the exact invocation expected by a handler."""


_MODES = frozenset(("SUPPORT", "RETAINED", "SNAPSHOT_CLEANUP"))
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ACTIVATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_LAMBDA_VERSION_ARN = re.compile(
    r"^arn:aws:lambda:us-west-2:246813579024:function:[A-Za-z0-9_-]+:[1-9][0-9]*$"
)
_STATE_MACHINE_VERSION_ARN = re.compile(
    r"^arn:aws:states:us-west-2:246813579024:stateMachine:[A-Za-z0-9_-]+:[1-9][0-9]*$"
)
_EVENT_KEYS = frozenset(
    (
        "mode",
        "activation_id",
        "activation_ordinal",
        "generation",
        "generation_text",
        "task11_handoff_identity_sha256",
        "dispatch_identity_sha256",
        "deployment_identity_sha256",
        "invoked_function_version_arn",
        "caller_state_machine_version_arn",
    )
)
_NAMED_EVENT_KEYS = _EVENT_KEYS | frozenset(("handler_kind",))


@dataclass(frozen=True)
class Task12HandlerConfig:
    mode: str
    invoked_function_version_arn: str
    caller_state_machine_version_arn: str
    deployment_identity_sha256: str


@dataclass(frozen=True)
class Task12HandlerInvocation:
    mode: str
    activation_id: str
    activation_ordinal: int
    generation: int
    generation_text: str
    task11_handoff_identity_sha256: Optional[str]
    dispatch_identity_sha256: str
    deployment_identity_sha256: str
    invoked_function_version_arn: str
    caller_state_machine_version_arn: str


T = TypeVar("T")


def _exact_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise Task12HandlerError("{0} must be a positive integer".format(name))
    return value


def _sha(value: Any, name: str, *, nullable: bool = False) -> Optional[str]:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise Task12HandlerError("{0} must be a lowercase SHA-256".format(name))
    return value


def _validate_config(config: Task12HandlerConfig) -> None:
    if config.mode not in _MODES:
        raise Task12HandlerError("handler config has an unknown mode")
    if not _LAMBDA_VERSION_ARN.fullmatch(config.invoked_function_version_arn):
        raise Task12HandlerError("handler config must name a published Lambda version")
    if not _STATE_MACHINE_VERSION_ARN.fullmatch(config.caller_state_machine_version_arn):
        raise Task12HandlerError("handler config must name a published state-machine version")
    _sha(config.deployment_identity_sha256, "config deployment_identity_sha256")


def parse_handler_invocation(
    *, event: Mapping[str, Any], config: Task12HandlerConfig
) -> Task12HandlerInvocation:
    """Parse a closed event after binding it to this deployed handler identity."""
    _validate_config(config)
    if not isinstance(event, Mapping) or set(event) != _EVENT_KEYS:
        raise Task12HandlerError("Task 12 handler event must have exactly the closed envelope fields")
    mode = event["mode"]
    if mode != config.mode:
        raise Task12HandlerError("event mode does not match the handler mode")
    activation_id = event["activation_id"]
    if not isinstance(activation_id, str) or not _ACTIVATION_ID.fullmatch(activation_id):
        raise Task12HandlerError("activation_id is invalid")
    activation_ordinal = _exact_positive_int(event["activation_ordinal"], "activation_ordinal")
    generation = _exact_positive_int(event["generation"], "generation")
    generation_text = event["generation_text"]
    if not isinstance(generation_text, str) or generation_text != "{0:08d}".format(generation):
        raise Task12HandlerError("generation_text must exactly encode generation")
    task11_handoff = _sha(
        event["task11_handoff_identity_sha256"],
        "task11_handoff_identity_sha256",
        nullable=True,
    )
    if mode == "SUPPORT" and task11_handoff is None:
        raise Task12HandlerError("support continuation requires the approved Task 11 handoff identity")
    dispatch_identity = _sha(event["dispatch_identity_sha256"], "dispatch_identity_sha256")
    deployment_identity = _sha(event["deployment_identity_sha256"], "deployment_identity_sha256")
    if deployment_identity != config.deployment_identity_sha256:
        raise Task12HandlerError("event deployment identity does not match the handler deployment")
    invoked_arn = event["invoked_function_version_arn"]
    caller_arn = event["caller_state_machine_version_arn"]
    if invoked_arn != config.invoked_function_version_arn or not _LAMBDA_VERSION_ARN.fullmatch(invoked_arn):
        raise Task12HandlerError("event invoked Lambda must be this exact published version")
    if caller_arn != config.caller_state_machine_version_arn or not _STATE_MACHINE_VERSION_ARN.fullmatch(caller_arn):
        raise Task12HandlerError("event caller must be this exact published state-machine version")
    return Task12HandlerInvocation(
        mode=mode,
        activation_id=activation_id,
        activation_ordinal=activation_ordinal,
        generation=generation,
        generation_text=generation_text,
        task11_handoff_identity_sha256=task11_handoff,
        dispatch_identity_sha256=dispatch_identity,
        deployment_identity_sha256=deployment_identity,
        invoked_function_version_arn=invoked_arn,
        caller_state_machine_version_arn=caller_arn,
    )


def dispatch_handler(
    *,
    event: Mapping[str, Any],
    config: Task12HandlerConfig,
    coordinator: Callable[[Task12HandlerInvocation], T],
) -> T:
    """Validate the envelope and delegate once to the injected coordinator."""
    if not callable(coordinator):
        raise Task12HandlerError("coordinator must be callable")
    return coordinator(parse_handler_invocation(event=event, config=config))


def _dispatch_fixed_handler(
    *,
    event: Mapping[str, Any],
    config: Task12HandlerConfig,
    coordinator: Callable[[Task12HandlerInvocation], T],
    handler_kind: str,
    mode: str,
) -> T:
    if not isinstance(event, Mapping) or set(event) != _NAMED_EVENT_KEYS:
        raise Task12HandlerError(
            "named Task 12 event must have the closed handler-kind envelope"
        )
    if event["handler_kind"] != handler_kind:
        raise Task12HandlerError("event handler kind does not match entrypoint")
    if config.mode != mode:
        raise Task12HandlerError("handler kind does not match configured mode")
    parsed_event = dict(event)
    parsed_event.pop("handler_kind")
    return dispatch_handler(
        event=parsed_event,
        config=config,
        coordinator=coordinator,
    )


def retained_execution_observer_handler(
    *,
    event: Mapping[str, Any],
    config: Task12HandlerConfig,
    coordinator: Callable[[Task12HandlerInvocation], T],
) -> T:
    return _dispatch_fixed_handler(
        event=event,
        config=config,
        coordinator=coordinator,
        handler_kind="RETAINED_EXECUTION_OBSERVER",
        mode="RETAINED",
    )


def retained_terminal_v2_handler(
    *,
    event: Mapping[str, Any],
    config: Task12HandlerConfig,
    coordinator: Callable[[Task12HandlerInvocation], T],
) -> T:
    return _dispatch_fixed_handler(
        event=event,
        config=config,
        coordinator=coordinator,
        handler_kind="RETAINED_TERMINAL_V2",
        mode="RETAINED",
    )


def retained_finalizer_handler(
    *,
    event: Mapping[str, Any],
    config: Task12HandlerConfig,
    coordinator: Callable[[Task12HandlerInvocation], T],
) -> T:
    return _dispatch_fixed_handler(
        event=event,
        config=config,
        coordinator=coordinator,
        handler_kind="RETAINED_FINALIZER",
        mode="RETAINED",
    )


def retained_h1g_drained_handler(
    *,
    event: Mapping[str, Any],
    config: Task12HandlerConfig,
    coordinator: Callable[[Task12HandlerInvocation], T],
) -> T:
    return _dispatch_fixed_handler(
        event=event,
        config=config,
        coordinator=coordinator,
        handler_kind="RETAINED_H1G_DRAINED",
        mode="RETAINED",
    )


def retained_worker_drain_handler(
    *,
    event: Mapping[str, Any],
    config: Task12HandlerConfig,
    coordinator: Callable[[Task12HandlerInvocation], T],
) -> T:
    return _dispatch_fixed_handler(
        event=event,
        config=config,
        coordinator=coordinator,
        handler_kind="RETAINED_WORKER_DRAIN",
        mode="RETAINED",
    )


def retained_operator_disposition_handler(
    *,
    event: Mapping[str, Any],
    config: Task12HandlerConfig,
    coordinator: Callable[[Task12HandlerInvocation], T],
) -> T:
    return _dispatch_fixed_handler(
        event=event,
        config=config,
        coordinator=coordinator,
        handler_kind="RETAINED_OPERATOR_DISPOSITION",
        mode="RETAINED",
    )


def retained_orphan_audit_handler(
    *,
    event: Mapping[str, Any],
    config: Task12HandlerConfig,
    coordinator: Callable[[Task12HandlerInvocation], T],
) -> T:
    return _dispatch_fixed_handler(
        event=event,
        config=config,
        coordinator=coordinator,
        handler_kind="RETAINED_ORPHAN_AUDIT",
        mode="RETAINED",
    )


def retained_snapshot_cleanup_handler(
    *,
    event: Mapping[str, Any],
    config: Task12HandlerConfig,
    coordinator: Callable[[Task12HandlerInvocation], T],
) -> T:
    return _dispatch_fixed_handler(
        event=event,
        config=config,
        coordinator=coordinator,
        handler_kind="RETAINED_SNAPSHOT_CLEANUP",
        mode="SNAPSHOT_CLEANUP",
    )
