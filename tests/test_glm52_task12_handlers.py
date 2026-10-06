from __future__ import annotations

import pytest


SHA = "a" * 64


def _config() -> object:
    from glm52_enforcement.task12_handlers import Task12HandlerConfig

    return Task12HandlerConfig(
        mode="SUPPORT",
        invoked_function_version_arn=(
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-task12-support:42"
        ),
        caller_state_machine_version_arn=(
            "arn:aws:states:us-west-2:246813579024:stateMachine:"
            "keep-glm52-h1g-support:9"
        ),
        deployment_identity_sha256=SHA,
    )


def _event() -> dict[str, object]:
    return {
        "mode": "SUPPORT",
        "activation_id": "activation-1",
        "activation_ordinal": 1,
        "generation": 1,
        "generation_text": "00000001",
        "task11_handoff_identity_sha256": SHA,
        "dispatch_identity_sha256": SHA,
        "deployment_identity_sha256": SHA,
        "invoked_function_version_arn": (
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-task12-support:42"
        ),
        "caller_state_machine_version_arn": (
            "arn:aws:states:us-west-2:246813579024:stateMachine:"
            "keep-glm52-h1g-support:9"
        ),
    }


def test_handler_parses_exact_versioned_invocation_and_dispatches_only_coordinator() -> None:
    from glm52_enforcement.task12_handlers import dispatch_handler

    seen: list[object] = []
    result = dispatch_handler(
        event=_event(),
        config=_config(),
        coordinator=lambda invocation: seen.append(invocation) or {"status": "accepted"},
    )

    assert result == {"status": "accepted"}
    assert len(seen) == 1
    assert seen[0].mode == "SUPPORT"


@pytest.mark.parametrize(
defect := "mutation",
    [
        lambda event: event.update(
            {"invoked_function_version_arn": event["invoked_function_version_arn"].rsplit(":", 1)[0]}
        ),
        lambda event: event.update({"mode": "SNAPSHOT_CLEANUP"}),
        lambda event: event.update({"generation_text": "00000002"}),
        lambda event: event.update({"caller_state_machine_version_arn": "alias"}),
    ],
)
def test_handler_rejects_alias_mode_crossover_and_identity_drift(mutation: object) -> None:
    from glm52_enforcement.task12_handlers import Task12HandlerError, dispatch_handler

    event = _event()
    mutation(event)
    with pytest.raises(Task12HandlerError):
        dispatch_handler(event=event, config=_config(), coordinator=lambda _: {"status": "bad"})


_NAMED_HANDLERS = (
    (
        "retained_execution_observer_handler",
        "RETAINED_EXECUTION_OBSERVER",
        "RETAINED",
    ),
    (
        "retained_terminal_v2_handler",
        "RETAINED_TERMINAL_V2",
        "RETAINED",
    ),
    ("retained_finalizer_handler", "RETAINED_FINALIZER", "RETAINED"),
    (
        "retained_h1g_drained_handler",
        "RETAINED_H1G_DRAINED",
        "RETAINED",
    ),
    (
        "retained_worker_drain_handler",
        "RETAINED_WORKER_DRAIN",
        "RETAINED",
    ),
    (
        "retained_operator_disposition_handler",
        "RETAINED_OPERATOR_DISPOSITION",
        "RETAINED",
    ),
    (
        "retained_orphan_audit_handler",
        "RETAINED_ORPHAN_AUDIT",
        "RETAINED",
    ),
    (
        "retained_snapshot_cleanup_handler",
        "RETAINED_SNAPSHOT_CLEANUP",
        "SNAPSHOT_CLEANUP",
    ),
)


def _named_config(mode: str) -> object:
    from glm52_enforcement.task12_handlers import Task12HandlerConfig

    return Task12HandlerConfig(
        mode=mode,
        invoked_function_version_arn=(
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-task12-retained:42"
        ),
        caller_state_machine_version_arn=(
            "arn:aws:states:us-west-2:246813579024:stateMachine:"
            "keep-glm52-h1g-retained:9"
        ),
        deployment_identity_sha256=SHA,
    )


def _named_event(kind: str, mode: str) -> dict[str, object]:
    return {
        "handler_kind": kind,
        "mode": mode,
        "activation_id": "activation-1",
        "activation_ordinal": 1,
        "generation": 1,
        "generation_text": "00000001",
        "task11_handoff_identity_sha256": None,
        "dispatch_identity_sha256": SHA,
        "deployment_identity_sha256": SHA,
        "invoked_function_version_arn": (
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-task12-retained:42"
        ),
        "caller_state_machine_version_arn": (
            "arn:aws:states:us-west-2:246813579024:stateMachine:"
            "keep-glm52-h1g-retained:9"
        ),
    }


@pytest.mark.parametrize(("entrypoint_name", "kind", "mode"), _NAMED_HANDLERS)
def test_named_handler_dispatches_only_its_fixed_kind(
    entrypoint_name: str, kind: str, mode: str
) -> None:
    from glm52_enforcement import task12_handlers

    seen: list[object] = []
    result = getattr(task12_handlers, entrypoint_name)(
        event=_named_event(kind, mode),
        config=_named_config(mode),
        coordinator=lambda invocation: seen.append(invocation)
        or {"handler_kind": kind},
    )

    assert result == {"handler_kind": kind}
    assert len(seen) == 1
    assert seen[0].mode == mode


@pytest.mark.parametrize(("entrypoint_name", "kind", "mode"), _NAMED_HANDLERS)
def test_named_handler_rejects_cross_routing_before_coordinator(
    entrypoint_name: str, kind: str, mode: str
) -> None:
    from glm52_enforcement import task12_handlers
    from glm52_enforcement.task12_handlers import Task12HandlerError

    seen: list[object] = []
    event = _named_event(kind, mode)
    event["handler_kind"] = "RETAINED_FOREIGN"
    with pytest.raises(Task12HandlerError, match="handler kind"):
        getattr(task12_handlers, entrypoint_name)(
            event=event,
            config=_named_config(mode),
            coordinator=seen.append,
        )
    assert seen == []


@pytest.mark.parametrize(("entrypoint_name", "kind", "mode"), _NAMED_HANDLERS)
def test_named_handler_rejects_generic_mode_crossover(
    entrypoint_name: str, kind: str, mode: str
) -> None:
    from glm52_enforcement import task12_handlers
    from glm52_enforcement.task12_handlers import Task12HandlerError

    foreign_mode = (
        "RETAINED" if mode == "SNAPSHOT_CLEANUP" else "SNAPSHOT_CLEANUP"
    )
    with pytest.raises(Task12HandlerError, match="configured mode"):
        getattr(task12_handlers, entrypoint_name)(
            event=_named_event(kind, foreign_mode),
            config=_named_config(foreign_mode),
            coordinator=lambda _: "must-not-run",
        )
