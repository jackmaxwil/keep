"""Deterministic planning and guarded advancement for recovery campaigns."""

from __future__ import annotations

import errno
import fcntl
import json
import os
import secrets
import stat
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .ledger import (
    LedgerError,
    LedgerUncertainCommitError,
    append_event,
    evidence_from_event,
    load_ledger,
)
from .models import (
    AdvanceResult,
    CampaignConfig,
    CampaignObservation,
    ExperimentObservation,
    LauncherRecord,
    LedgerEvent,
    Transition,
)
from .observer import observe_campaign
from .launcher import command_sha256


_STAGES = (
    "audit_complete",
    "reevaluation_complete",
    "attribution_complete",
    "experiment_complete",
)
_SHA256_CHARS = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class _OutstandingLaunch:
    experiment_name: str
    transition_name: str
    record: LauncherRecord


def _physical_snapshot(
    observation: CampaignObservation,
    experiment_name: str,
) -> ExperimentObservation | None:
    if observation.experiment_observations:
        try:
            return observation.snapshot(experiment_name)
        except KeyError:
            return None
    if observation.experiment_name != experiment_name:
        return None
    return ExperimentObservation(
        experiment_name=experiment_name,
        recovered_groups=observation.recovered_groups,
        expected_groups=observation.expected_groups,
        recovered_groups_state=observation.recovered_groups_state,
        artifact_links=observation.artifact_links,
        expected_artifact_links=observation.expected_artifact_links,
        artifact_links_state=observation.artifact_links_state,
        manifest_state=observation.manifest_state,
        manifest_file_sha256=observation.manifest_file_sha256,
        manifest_body_sha256=observation.manifest_body_sha256,
        contradictions=observation.contradictions,
    )


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _SHA256_CHARS
    )


def _stage_inventory(
    config: CampaignConfig,
    events: Sequence[LedgerEvent],
) -> tuple[
    dict[str, tuple[str, ...]],
    dict[str, tuple[LedgerEvent, ...]],
    str | None,
]:
    names = {experiment.name for experiment in config.experiments}
    experiments = {experiment.name: experiment for experiment in config.experiments}
    inventories: dict[str, list[str]] = {name: [] for name in names}
    stage_events: dict[str, list[LedgerEvent]] = {name: [] for name in names}
    completed: set[str] = set()
    evidence_by_hash: dict[str, LedgerEvent] = {}
    consumed_evidence: set[str] = set()
    consumed_evidence_payloads: set[str] = set()
    consumed_content_identities: set[str] = set()
    for event in events:
        if event.event_kind == "evidence_registered":
            evidence_by_hash[event.event_sha256] = event
            continue
        if event.event_kind not in _STAGES:
            continue
        if event.experiment not in names:
            return {}, {}, (
                f"stage event {event.event_kind} names undeclared experiment "
                f"{event.experiment}"
            )
        missing_dependencies = set(experiments[event.experiment].depends_on) - completed
        if missing_dependencies:
            return {}, {}, (
                f"stage event for {event.experiment} predates accepted dependency "
                + ", ".join(sorted(missing_dependencies))
            )
        payload = event.payload
        expected_fields = (
            {
                "campaign_config_sha256",
                "manifest_sha256",
                "evidence_event_sha256",
                "human_decision_reference",
            }
            if event.event_kind == "experiment_complete"
            else {
                "campaign_config_sha256",
                "manifest_sha256",
                "evidence_event_sha256",
            }
        )
        if (
            set(payload) != expected_fields
            or payload.get("campaign_config_sha256")
            != config.campaign_config_sha256
            or not _sha256(payload.get("manifest_sha256"))
            or not _sha256(payload.get("evidence_event_sha256"))
        ):
            return {}, {}, f"malformed {event.event_kind} stage event"
        if event.event_kind == "experiment_complete" and (
            not isinstance(payload.get("human_decision_reference"), str)
            or not str(payload["human_decision_reference"]).strip()
        ):
            return {}, {}, "experiment_complete requires a human decision reference"
        observed = inventories[event.experiment]
        if event.event_kind in observed:
            return {}, {}, f"duplicate {event.event_kind} stage event"
        expected = _STAGES[len(observed)] if len(observed) < len(_STAGES) else None
        if event.event_kind != expected:
            return {}, {}, (
                f"stage order violation for {event.experiment}: expected "
                f"{expected or 'no further event'}, got {event.event_kind}"
            )
        evidence_event = evidence_by_hash.get(str(payload["evidence_event_sha256"]))
        if evidence_event is None or evidence_event.experiment != event.experiment:
            return {}, {}, (
                f"{event.event_kind} lacks corresponding earlier same-experiment "
                "evidence registration"
            )
        if evidence_event.event_sha256 in consumed_evidence:
            return {}, {}, (
                "one evidence registration cannot satisfy multiple campaign stages"
            )
        try:
            evidence = evidence_from_event(evidence_event)
        except LedgerError as error:
            return {}, {}, f"invalid stage evidence registration: {error}"
        if evidence.manifest_identity_sha256 != payload["manifest_sha256"]:
            return {}, {}, "stage evidence manifest identity does not match"
        if any(result.exit_code != 0 for result in evidence.verification_results):
            return {}, {}, "stage evidence contains a failed verification"
        required_verification = f"{event.event_kind}_verification"
        if (
            len(evidence.verification_results) != 1
            or evidence.verification_results[0].name != required_verification
            or evidence.verification_results[0].argv != evidence.argv
        ):
            return {}, {}, (
                f"{event.event_kind} requires exactly one stage-specific "
                f"verification {required_verification!r} matching evidence argv"
            )
        canonical_evidence_payload = json.dumps(
            asdict(evidence),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if (
            canonical_evidence_payload in consumed_evidence_payloads
            or evidence.content_sha256 in consumed_content_identities
        ):
            return {}, {}, (
                "canonical evidence payload/content identity cannot satisfy "
                "multiple campaign stages"
            )
        consumed_evidence.add(evidence_event.event_sha256)
        consumed_evidence_payloads.add(canonical_evidence_payload)
        consumed_content_identities.add(evidence.content_sha256)
        observed.append(event.event_kind)
        stage_events[event.experiment].append(event)
        if event.event_kind == "experiment_complete":
            completed.add(event.experiment)
    return (
        {name: tuple(items) for name, items in inventories.items()},
        {name: tuple(items) for name, items in stage_events.items()},
        None,
    )


def _transition_history_error(
    config: CampaignConfig,
    events: Sequence[LedgerEvent],
) -> tuple[str | None, _OutstandingLaunch | None]:
    pending: tuple[str, str, str] | None = None
    outstanding: _OutstandingLaunch | None = None
    terminal_events: dict[str, LedgerEvent] = {}
    authorized_terminals: set[str] = set()
    for event in events:
        if event.event_kind not in {
            "transition_started",
            "transition_launched",
            "transition_launch_failed",
            "transition_completed",
            "transition_finished",
            "retry_authorized",
        }:
            continue
        try:
            declared = _declared_for_experiment(config, event.experiment)
        except KeyError:
            return (
                f"transition event names undeclared experiment {event.experiment}",
                None,
            )
        if declared is None:
            return (
                "transition event targets non-executable experiment "
                f"{event.experiment}",
                None,
            )
        payload = event.payload
        if event.event_kind == "retry_authorized":
            if set(payload) != {
                "campaign",
                "experiment",
                "transition",
                "terminal_event_sha256",
                "exit_code",
                "reason",
            }:
                return "malformed retry_authorized event", None
            terminal_hash = str(payload["terminal_event_sha256"])
            terminal = terminal_events.get(terminal_hash)
            if (
                payload["campaign"] != config.campaign
                or payload["experiment"] != event.experiment
                or payload["transition"] != declared.name
                or terminal is None
                or terminal.event_kind != "transition_finished"
                or terminal.experiment != event.experiment
                or payload["exit_code"] != terminal.payload["exit_code"]
            ):
                return "retry_authorized does not bind an exact earlier failure", None
            if terminal_hash in authorized_terminals:
                return "duplicate retry_authorized event for one terminal", None
            authorized_terminals.add(terminal_hash)
            continue
        if event.event_kind == "transition_started":
            if pending is not None or outstanding is not None:
                return (
                    "unmatched transition lifecycle precedes another launch attempt",
                    outstanding,
                )
            if set(payload) != {
                "transition",
                "argv",
                "expected_artifacts",
                "campaign_config_sha256",
                "launch_token",
            }:
                return "malformed transition_started event", None
            if (
                payload.get("transition") != declared.name
                or tuple(payload.get("argv", ())) != declared.argv
                or tuple(payload.get("expected_artifacts", ()))
                != declared.expected_artifacts
                or payload.get("campaign_config_sha256")
                != config.campaign_config_sha256
                or not _sha256(payload.get("launch_token"))
            ):
                return (
                    "transition_started does not match the exact declared transition",
                    None,
                )
            pending = (
                event.experiment,
                declared.name,
                str(payload["launch_token"]),
            )
            continue
        if event.event_kind in {"transition_completed", "transition_finished"}:
            if (
                outstanding is None
                or (
                    outstanding.experiment_name,
                    outstanding.transition_name,
                )
                != (event.experiment, declared.name)
                or payload.get("launch_token") != outstanding.record.launch_token
            ):
                return f"{event.event_kind} has no matching launched transition", None
            terminal_fields = {
                "transition",
                "launch_token",
                "pid",
                "supervisor_pid",
                "command_sha256",
                "exit_code",
                "campaign_config_sha256",
                "physical_complete",
                "manifest_sha256",
                "terminal_path",
            }
            if set(payload) != terminal_fields:
                return f"malformed {event.event_kind} event", None
            completed = event.event_kind == "transition_completed"
            if (
                payload.get("transition") != declared.name
                or payload.get("campaign_config_sha256")
                != config.campaign_config_sha256
                or payload.get("command_sha256") != command_sha256(declared.argv)
                or type(payload.get("exit_code")) is not int
                or type(payload.get("pid")) is not int
                or payload.get("pid") != outstanding.record.pid
                or type(payload.get("supervisor_pid")) is not int
                or payload.get("supervisor_pid")
                != outstanding.record.supervisor_pid
                or payload.get("terminal_path")
                != outstanding.record.terminal_path
                or type(payload.get("physical_complete")) is not bool
                or (
                    completed
                    and (
                        payload.get("exit_code") != 0
                        or payload.get("physical_complete") is not True
                        or not _sha256(payload.get("manifest_sha256"))
                    )
                )
                or (
                    not completed
                    and payload.get("exit_code") == 0
                    and payload.get("physical_complete") is True
                )
            ):
                return f"invalid {event.event_kind} terminal evidence", None
            outstanding = None
            terminal_events[event.event_sha256] = event
            continue
        if (
            pending is None
            or pending[:2] != (event.experiment, declared.name)
            or payload.get("launch_token") != pending[2]
        ):
            return f"{event.event_kind} has no matching transition_started event", None
        if event.event_kind == "transition_launched":
            if set(payload) != {
                "pid",
                "supervisor_pid",
                "command_sha256",
                "started_at",
                "stdout_path",
                "stderr_path",
                "terminal_path",
                "launch_token",
                "transition",
                "campaign_config_sha256",
            } or payload.get("transition") != declared.name:
                return "malformed transition_launched event", None
            if (
                payload.get("campaign_config_sha256")
                != config.campaign_config_sha256
            ):
                return "transition_launched campaign fingerprint mismatch", None
            try:
                record = LauncherRecord(
                    pid=payload["pid"],  # type: ignore[arg-type]
                    supervisor_pid=payload["supervisor_pid"],  # type: ignore[arg-type]
                    command_sha256=payload["command_sha256"],  # type: ignore[arg-type]
                    launch_token=payload["launch_token"],  # type: ignore[arg-type]
                    started_at=payload["started_at"],  # type: ignore[arg-type]
                    stdout_path=payload["stdout_path"],  # type: ignore[arg-type]
                    stderr_path=payload["stderr_path"],  # type: ignore[arg-type]
                    terminal_path=payload["terminal_path"],  # type: ignore[arg-type]
                    experiment_name=event.experiment,
                    transition_name=declared.name,
                )
            except (TypeError, ValueError):
                return "malformed transition_launched event", None
            if record.command_sha256 != command_sha256(declared.argv):
                return (
                    "transition_launched command hash does not bind declared argv",
                    None,
                )
            outstanding = _OutstandingLaunch(
                experiment_name=event.experiment,
                transition_name=declared.name,
                record=record,
            )
        else:
            if set(payload) != {
                "transition",
                "error_type",
                "error",
                "campaign_config_sha256",
            }:
                return "malformed transition_launch_failed event", None
            if (
                payload.get("transition") != declared.name
                or not isinstance(payload.get("error_type"), str)
                or not payload.get("error_type")
                or not isinstance(payload.get("error"), str)
                or not payload.get("error")
                or payload.get("campaign_config_sha256")
                != config.campaign_config_sha256
            ):
                return "malformed transition_launch_failed event", None
        pending = None
    if pending is not None:
        return (
            f"unmatched transition_started for {pending[0]}; refusing a "
            "duplicate launch",
            outstanding,
        )
    return None, outstanding


def _decision(
    *,
    action: str,
    experiment_name: str,
    reason: str,
    declared: Transition | None = None,
    exit_code: int | None = None,
) -> Transition:
    return Transition(
        name=(declared.name if declared is not None else f"{experiment_name}-{action}"),
        experiment_name=experiment_name,
        action=action,
        heavy=(
            declared.heavy
            if declared is not None and action in {"wait", "resume"}
            else False
        ),
        argv=(
            declared.argv
            if declared is not None and action in {"wait", "resume"}
            else ()
        ),
        expected_artifacts=(
            declared.expected_artifacts if declared is not None else ()
        ),
        reason=reason,
        exit_code=exit_code,
    )


def _declared_for_experiment(
    config: CampaignConfig, experiment_name: str
) -> Transition | None:
    experiment = config.experiment(experiment_name)
    if experiment.transition_name is None:
        return None
    return config.transition(experiment.transition_name)


def plan_next_transition(
    config: CampaignConfig,
    observation: CampaignObservation,
    ledger: Sequence[LedgerEvent],
) -> Transition:
    """Return one pure, fail-closed state-machine decision."""

    declared_names = {experiment.name for experiment in config.experiments}
    if observation.campaign != config.campaign:
        return _decision(
            action="blocked",
            experiment_name=config.campaign,
            reason=(
                f"observation campaign {observation.campaign!r} does not match "
                f"declared campaign {config.campaign!r}"
            ),
        )
    if (
        observation.experiment_name is not None
        and observation.experiment_name not in declared_names
    ):
        return _decision(
            action="blocked",
            experiment_name=config.campaign,
            reason=(
                "observation selected undeclared experiment "
                f"{observation.experiment_name!r}"
            ),
        )
    if observation.experiment_observations:
        snapshot_names = [
            snapshot.experiment_name
            for snapshot in observation.experiment_observations
        ]
        if (
            len(snapshot_names) != len(set(snapshot_names))
            or set(snapshot_names) != declared_names
        ):
            return _decision(
                action="blocked",
                experiment_name=config.campaign,
                reason=(
                    "physical snapshot inventory does not exactly match declared "
                    "experiments"
                ),
            )
        for snapshot in observation.experiment_observations:
            expected_groups = len(
                config.experiment(snapshot.experiment_name).recovered_layers
            ) * 3
            if (
                snapshot.expected_groups != expected_groups
                or snapshot.expected_artifact_links != 225
            ):
                return _decision(
                    action="blocked",
                    experiment_name=snapshot.experiment_name,
                    reason=(
                        "physical snapshot expected counts do not match the "
                        "campaign matrix"
                    ),
                )

    stages, bound_stage_events, stage_error = _stage_inventory(config, ledger)
    if stage_error is not None:
        return _decision(
            action="blocked",
            experiment_name=observation.experiment_name or config.campaign,
            reason=stage_error,
        )
    history_error, outstanding_launch = _transition_history_error(config, ledger)
    if history_error is not None:
        return _decision(
            action="blocked",
            experiment_name=observation.experiment_name or config.campaign,
            reason=history_error,
        )
    if observation.contradictions:
        return _decision(
            action="blocked",
            experiment_name=observation.experiment_name or config.campaign,
            reason="; ".join(observation.contradictions),
        )

    completed = {name for name, inventory in stages.items() if inventory == _STAGES}
    for experiment_name, stage_records in bound_stage_events.items():
        if not stage_records:
            continue
        snapshot = _physical_snapshot(observation, experiment_name)
        if snapshot is None or not snapshot.physically_complete:
            return _decision(
                action="blocked",
                experiment_name=experiment_name,
                reason=(
                    "stage evidence cannot bypass missing authenticated physical "
                    "manifest/group/link inventory"
                ),
            )
        if any(
            event.payload["manifest_sha256"] != snapshot.manifest_file_sha256
            for event in stage_records
        ):
            return _decision(
                action="blocked",
                experiment_name=experiment_name,
                reason="stage evidence is bound to a different manifest identity",
            )
    policy_target = next(
        (
            experiment
            for experiment in config.experiments
            if experiment.name not in completed
        ),
        None,
    )

    if observation.active_processes:
        if len(observation.active_processes) != 1:
            return _decision(
                action="blocked",
                experiment_name=observation.experiment_name or config.campaign,
                reason="overlapping declared recovery processes are active",
            )
        process = observation.active_processes[0]
        if process.experiment_name is None:
            return _decision(
                action="blocked",
                experiment_name=config.campaign,
                reason="active process has no declared experiment identity",
            )
        if policy_target is None or process.experiment_name != policy_target.name:
            return _decision(
                action="blocked",
                experiment_name=process.experiment_name,
                reason=(
                    "active process targets a later experiment than the first "
                    "ledger-incomplete policy experiment"
                ),
            )
        try:
            declared = _declared_for_experiment(config, process.experiment_name)
        except KeyError:
            declared = None
        exact = (
            declared is not None
            and process.transition_name == declared.name
            and process.command == declared.argv
        )
        if (
            not exact
            or observation.lock_state != "held"
            or not observation.lock_held
            or not observation.lock_owner_known
        ):
            return _decision(
                action="blocked",
                experiment_name=process.experiment_name,
                reason=(
                    "active recovery does not have one exact declared command "
                    "and proven heavy-lock owner"
                ),
            )
        return _decision(
            action="wait",
            experiment_name=process.experiment_name,
            reason="exact declared heavy transition is healthy and running",
            declared=declared,
        )

    if outstanding_launch is not None:
        return _decision(
            action="blocked",
            experiment_name=outstanding_launch.experiment_name,
            reason=(
                "launched transition has no durable terminal outcome and no "
                "active exact producer; PID disappearance never permits resume"
            ),
        )

    if observation.lock_held or observation.lock_state == "held":
        return _decision(
            action="blocked",
            experiment_name=observation.experiment_name or config.campaign,
            reason="heavy lock is held but its exact declared owner is unknown",
        )
    if observation.lock_state == "invalid":
        return _decision(
            action="blocked",
            experiment_name=observation.experiment_name or config.campaign,
            reason="heavy lock state is invalid",
        )

    if policy_target is not None:
        terminal_events = [
            event
            for event in ledger
            if event.experiment == policy_target.name
            and event.event_kind in {"transition_completed", "transition_finished"}
        ]
        if terminal_events:
            terminal = terminal_events[-1]
            if terminal.event_kind == "transition_finished":
                authorized = any(
                    event.sequence > terminal.sequence
                    and event.event_kind == "retry_authorized"
                    and event.payload["terminal_event_sha256"]
                    == terminal.event_sha256
                    for event in ledger
                )
                if not authorized:
                    actual_exit = int(terminal.payload["exit_code"])
                    cli_exit = (
                        actual_exit
                        if actual_exit > 0
                        else 128 + min(abs(actual_exit), 127)
                        if actual_exit < 0
                        else 3
                    )
                    return _decision(
                        action="blocked",
                        experiment_name=policy_target.name,
                        reason=(
                            "terminal recovery outcome is not an authenticated "
                            f"completion (actual exit_code={actual_exit}); explicit "
                            "retry authorization is required"
                        ),
                        exit_code=cli_exit,
                    )
            else:
                snapshot = _physical_snapshot(observation, policy_target.name)
                if (
                    snapshot is None
                    or not snapshot.physically_complete
                    or snapshot.manifest_file_sha256
                    != terminal.payload["manifest_sha256"]
                ):
                    return _decision(
                        action="blocked",
                        experiment_name=policy_target.name,
                        reason=(
                            "transition_completed terminal manifest does not match "
                            "current authenticated physical evidence"
                        ),
                    )

    if len(completed) == len(config.experiments):
        final_event = bound_stage_events[config.experiments[-1].name][-1]
        evidence_hash = str(final_event.payload["evidence_event_sha256"])
        release_evidence = evidence_from_event(
            next(event for event in ledger if event.event_sha256 == evidence_hash)
        )
        if not release_evidence.release_eligible:
            return _decision(
                action="blocked",
                experiment_name=config.campaign,
                reason="campaign completion requires release-class evidence",
            )
        return _decision(
            action="complete",
            experiment_name=config.campaign,
            reason="every declared experiment and ordered evidence packet is complete",
        )

    if policy_target is None:
        return _decision(
            action="blocked",
            experiment_name=config.campaign,
            reason="campaign policy target is absent",
        )
    experiment = policy_target
    inventory = stages[experiment.name]

    for dependency in experiment.depends_on:
        if dependency not in completed:
            return _decision(
                action="blocked",
                experiment_name=experiment.name,
                reason=f"dependency {dependency} lacks an accepted evidence packet",
            )

    snapshot = _physical_snapshot(observation, experiment.name)
    if snapshot is None:
        return _decision(
            action="blocked",
            experiment_name=experiment.name,
            reason="declared experiment lacks a physical observation snapshot",
        )
    declared = _declared_for_experiment(config, experiment.name)
    if not snapshot.physically_complete:
        if inventory:
            return _decision(
                action="blocked",
                experiment_name=experiment.name,
                reason="stage evidence exists without valid physical materialization",
            )
        if declared is None:
            return _decision(
                action="blocked",
                experiment_name=experiment.name,
                reason=(
                    f"{experiment.name} has no declared implementation transition; "
                    "implementation and human policy decisions are required"
                ),
            )
        return _decision(
            action="resume",
            experiment_name=experiment.name,
            reason="declared materialization is incomplete and no owner is active",
            declared=declared,
        )

    manifest_sha256 = snapshot.manifest_file_sha256
    if manifest_sha256 is None:
        return _decision(
            action="blocked",
            experiment_name=experiment.name,
            reason="valid manifest observation lacks its authenticated file identity",
        )
    for event in ledger:
        if event.experiment == experiment.name and event.event_kind in _STAGES:
            if event.payload["manifest_sha256"] != manifest_sha256:
                return _decision(
                    action="blocked",
                    experiment_name=experiment.name,
                    reason="stage evidence is bound to a different manifest identity",
                )

    next_action = (
        "audit",
        "reevaluate",
        "attribute",
        "ready",
    )[len(inventory)]
    reason = {
        "audit": "authenticated materialization is ready for audit",
        "reevaluate": "audit evidence is complete; reevaluation is next",
        "attribute": "reevaluation evidence is complete; attribution is next",
        "ready": "experiment packet is complete and the next decision is human-owned",
    }[next_action]
    return _decision(
        action=next_action,
        experiment_name=experiment.name,
        reason=reason,
    )


def _result(
    planned: Transition,
    *,
    dry_run: bool,
    lock_conflict: bool = False,
) -> AdvanceResult:
    exit_code = 0
    if planned.action == "blocked":
        exit_code = (
            planned.exit_code
            if planned.exit_code is not None
            else 4
            if lock_conflict and not dry_run
            else 3
        )
    elif planned.action == "wait" and not dry_run:
        exit_code = 4
    return AdvanceResult(
        action=planned.action,
        experiment_name=planned.experiment_name,
        transition_name=(
            planned.name if planned.action in {"wait", "resume"} else None
        ),
        reason=planned.reason,
        argv=planned.argv,
        heavy=planned.heavy,
        dry_run=dry_run,
        launcher_record=None,
        exit_code=exit_code,
    )


def _anchors(events: Sequence[LedgerEvent]) -> tuple[str | None, int]:
    return (events[-1].event_sha256 if events else None, len(events))


def _append_reconciled(
    path: Path,
    *,
    events: Sequence[LedgerEvent],
    event_kind: str,
    experiment: str,
    payload: dict[str, object],
    append: Callable[..., LedgerEvent],
    load_events: Callable[[Path], Sequence[LedgerEvent]],
) -> LedgerEvent:
    head, count = _anchors(events)
    try:
        return append(
            path,
            event_kind=event_kind,
            experiment=experiment,
            payload=payload,
            expected_head_sha256=head,
            expected_event_count=count,
        )
    except LedgerUncertainCommitError:
        refreshed = tuple(load_events(path))
        if len(refreshed) != count + 1:
            raise
        candidate = refreshed[-1]
        if (
            candidate.event_kind != event_kind
            or candidate.experiment != experiment
            or candidate.payload != payload
            or candidate.previous_event_sha256 != head
        ):
            raise
        return candidate


def _launcher_payload(record: LauncherRecord) -> dict[str, object]:
    return {
        "pid": record.pid,
        "command_sha256": record.command_sha256,
        "started_at": record.started_at,
        "stdout_path": record.stdout_path,
        "stderr_path": record.stderr_path,
        "transition": record.transition_name,
        "supervisor_pid": record.supervisor_pid,
        "launch_token": record.launch_token,
        "terminal_path": record.terminal_path,
    }


def _validate_launcher_record(
    config: CampaignConfig,
    transition: Transition,
    record: LauncherRecord,
    launch_token: str,
    repo_root: Path,
) -> None:
    if (
        record.experiment_name != transition.experiment_name
        or record.transition_name != transition.name
        or record.command_sha256 != command_sha256(transition.argv)
        or record.launch_token != launch_token
        or record.supervisor_pid == record.pid
    ):
        raise ValueError("launcher return identity/hash/token contract mismatch")
    prefix = f"artifacts/quality/{config.campaign}-launches/"
    paths = (record.stdout_path, record.stderr_path, record.terminal_path)
    if len(set(paths)) != 3 or any(not path.startswith(prefix) for path in paths):
        raise ValueError("launcher return log/terminal path contract mismatch")
    for relative in paths:
        metadata = (repo_root / relative).lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
            raise ValueError(
                "launcher return paths must be private regular files"
            )


def default_ledger_path(config: CampaignConfig, repo_root: Path) -> Path:
    """Compute the ignored default ledger path without touching the filesystem."""

    return repo_root / "artifacts" / "quality" / f"{config.campaign}-ledger.jsonl"


def authorize_retry(
    config: CampaignConfig,
    *,
    repo_root: str | Path,
    ledger_path: str | Path | None,
    transition_name: str,
    reason: str,
    load_events: Callable[[Path], Sequence[LedgerEvent]] = load_ledger,
    append: Callable[..., LedgerEvent] = append_event,
) -> LedgerEvent:
    """Append one authorization bound to the latest failed terminal event."""

    root = Path(repo_root).absolute()
    requested = Path(ledger_path) if ledger_path is not None else None
    path = (
        requested
        if requested is not None and requested.is_absolute()
        else root / requested
        if requested is not None
        else default_ledger_path(config, root)
    )
    try:
        declared = config.transition(transition_name)
    except KeyError as error:
        raise ValueError(f"unknown campaign transition {transition_name!r}") from error
    events = tuple(load_events(path))
    history_error, _outstanding = _transition_history_error(config, events)
    if history_error is not None:
        raise LedgerError(f"invalid transition history: {history_error}")
    terminals = [
        event
        for event in events
        if event.experiment == declared.experiment_name
        and event.event_kind in {"transition_completed", "transition_finished"}
    ]
    if not terminals:
        raise LedgerError(
            f"transition {transition_name!r} has no terminal event to authorize"
        )
    terminal = terminals[-1]
    if terminal.event_kind == "transition_completed":
        raise LedgerError(
            "latest terminal is transition_completed; there is nothing to authorize"
        )
    if any(
        event.event_kind == "retry_authorized"
        and event.payload["terminal_event_sha256"] == terminal.event_sha256
        for event in events
    ):
        raise LedgerError(
            "retry authorization for this exact terminal event already exists"
        )
    payload: dict[str, object] = {
        "campaign": config.campaign,
        "experiment": declared.experiment_name,
        "transition": declared.name,
        "terminal_event_sha256": terminal.event_sha256,
        "exit_code": terminal.payload["exit_code"],
        "reason": reason,
    }
    return _append_reconciled(
        path,
        events=events,
        event_kind="retry_authorized",
        experiment=declared.experiment_name,
        payload=payload,
        append=append,
        load_events=load_events,
    )


def controller_mutex_path(config: CampaignConfig, repo_root: Path) -> Path:
    """Return the stable repository-directory authority used by the controller."""

    del config
    return repo_root


class ControllerMutexBusy(RuntimeError):
    pass


@contextmanager
def _controller_mutex(config: CampaignConfig, repo_root: Path):
    authority = controller_mutex_path(config, repo_root)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(authority, flags)
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise LedgerError("controller mutex authority must be a real directory")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EAGAIN}:
                raise ControllerMutexBusy("campaign controller mutex is held") from error
            raise
        parent = os.dup(descriptor)
        try:
            for component in ("artifacts", "quality"):
                try:
                    os.mkdir(component, 0o700, dir_fd=parent)
                except FileExistsError:
                    pass
                child = os.open(component, flags, dir_fd=parent)
                if not stat.S_ISDIR(os.fstat(child).st_mode):
                    os.close(child)
                    raise LedgerError(
                        "controller ledger parent must be a real directory"
                    )
                os.close(parent)
                parent = child
        finally:
            os.close(parent)
        yield descriptor
    finally:
        # Close only: an inherited supervisor fd intentionally retains the same
        # flock/open-file-description until terminal recording exits.
        os.close(descriptor)


def _parent_component_is_missing(path: Path) -> bool:
    """Distinguish honest absence from a symlink/invalid ledger parent."""

    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parent.parts[1:]:
        current = current / component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return True
        if not stat.S_ISDIR(metadata.st_mode):
            return False
    return False


def advance_campaign(
    config: CampaignConfig,
    *,
    repo_root: str | Path,
    ledger_path: str | Path | None,
    dry_run: bool,
    observe: Callable[[CampaignConfig, Path], CampaignObservation] = observe_campaign,
    load_events: Callable[[Path], Sequence[LedgerEvent]] = load_ledger,
    append: Callable[..., LedgerEvent] = append_event,
    launch: Callable[..., LauncherRecord] | None = None,
) -> AdvanceResult:
    """Observe, plan, and optionally launch one exact declared transition."""

    root = Path(repo_root).absolute()
    requested = Path(ledger_path) if ledger_path is not None else None
    path = (
        requested
        if requested is not None and requested.is_absolute()
        else root / requested
        if requested is not None
        else default_ledger_path(config, root)
    )
    observation = observe(config, root)
    try:
        events = tuple(load_events(path))
    except LedgerError:
        if load_events is not load_ledger or not _parent_component_is_missing(path):
            raise
        events = ()
    planned = plan_next_transition(config, observation, events)
    if dry_run:
        return _result(planned, dry_run=True)
    canonical_path = default_ledger_path(config, root)
    if path != canonical_path:
        raise ValueError(
            "mutating advance requires the canonical campaign ledger; custom "
            "--ledger paths are read-only/dry-run only"
        )
    try:
        with _controller_mutex(config, root) as mutex_fd:
            return _advance_campaign_locked(
                config,
                root=root,
                path=canonical_path,
                observe=observe,
                load_events=load_events,
                append=append,
                launch=launch,
                controller_mutex_fd=mutex_fd,
            )
    except ControllerMutexBusy:
        if planned.action in {"wait", "resume"}:
            waiting = replace(
                planned,
                action="wait",
                reason=(
                    "campaign-global controller launch mutex is held; another "
                    "launch or terminal monitor owns the campaign"
                ),
            )
            return _result(waiting, dry_run=False, lock_conflict=True)
        blocked = replace(
            planned,
            action="blocked",
            reason=(
                "campaign-global controller launch mutex is held while the "
                "canonical ledger has an unresolved launch lifecycle"
            ),
        )
        return _result(blocked, dry_run=False, lock_conflict=True)


def _advance_campaign_locked(
    config: CampaignConfig,
    *,
    root: Path,
    path: Path,
    observe: Callable[[CampaignConfig, Path], CampaignObservation],
    load_events: Callable[[Path], Sequence[LedgerEvent]],
    append: Callable[..., LedgerEvent],
    launch: Callable[..., LauncherRecord] | None,
    controller_mutex_fd: int,
) -> AdvanceResult:
    observation = observe(config, root)
    events = tuple(load_events(path))
    planned = plan_next_transition(config, observation, events)
    if planned.action != "resume":
        return _result(
            planned,
            dry_run=False,
            lock_conflict=(
                observation.lock_held
                or observation.lock_state == "held"
                or bool(observation.active_processes)
            ),
        )

    launch_token = secrets.token_hex(32)
    started_payload: dict[str, object] = {
        "transition": planned.name,
        "argv": list(planned.argv),
        "expected_artifacts": list(planned.expected_artifacts),
        "campaign_config_sha256": config.campaign_config_sha256,
        "launch_token": launch_token,
    }
    started = _append_reconciled(
        path,
        events=events,
        event_kind="transition_started",
        experiment=planned.experiment_name,
        payload=started_payload,
        append=append,
        load_events=load_events,
    )
    events = (*events, started)

    if launch is None:
        from .launcher import launch_transition

        launch = launch_transition

    try:
        record = launch(
            planned,
            repo_root=root,
            campaign=config.campaign,
            controller_mutex_fd=controller_mutex_fd,
            terminal_context={
                "campaign_config_path": config.campaign_config_path,
                "campaign_config_sha256": config.campaign_config_sha256,
                "ledger_path": str(path),
            },
            launch_token=launch_token,
        )
    except OSError as error:
        failure_payload = {
            "transition": planned.name,
            "error_type": type(error).__name__,
            "error": str(error),
            "campaign_config_sha256": config.campaign_config_sha256,
        }
        try:
            failed = _append_reconciled(
                path,
                events=events,
                event_kind="transition_launch_failed",
                experiment=planned.experiment_name,
                payload=failure_payload,
                append=append,
                load_events=load_events,
            )
            del failed
            reason = f"transition launch failed and was recorded: {error}"
        except LedgerError as ledger_error:
            reason = (
                f"transition launch failed ({error}) and failure recording also "
                "failed: "
                f"{ledger_error}"
            )
        return AdvanceResult(
            action="blocked",
            experiment_name=planned.experiment_name,
            transition_name=planned.name,
            reason=reason,
            argv=planned.argv,
            heavy=planned.heavy,
            dry_run=False,
            launcher_record=None,
            exit_code=1,
        )

    try:
        _validate_launcher_record(config, planned, record, launch_token, root)
    except (OSError, ValueError) as error:
        return AdvanceResult(
            action="wait",
            experiment_name=planned.experiment_name,
            transition_name=planned.name,
            reason=(
                "launcher returned an untrusted post-launch identity; no launched "
                f"event was appended and manual reconciliation is required: {error}"
            ),
            argv=planned.argv,
            heavy=planned.heavy,
            dry_run=False,
            launcher_record=record,
            exit_code=4,
        )

    launched_payload = {
        **_launcher_payload(record),
        "campaign_config_sha256": config.campaign_config_sha256,
    }
    try:
        _append_reconciled(
            path,
            events=events,
            event_kind="transition_launched",
            experiment=planned.experiment_name,
            payload=launched_payload,
            append=append,
            load_events=load_events,
        )
    except LedgerError as error:
        return AdvanceResult(
            action="wait",
            experiment_name=planned.experiment_name,
            transition_name=planned.name,
            reason=(
                "process launched but its ledger record failed; do not retry until "
                f"fresh observation reconciles pid {record.pid}: {error}"
            ),
            argv=planned.argv,
            heavy=planned.heavy,
            dry_run=False,
            launcher_record=record,
            exit_code=4,
        )
    return AdvanceResult(
        action="resume",
        experiment_name=planned.experiment_name,
        transition_name=planned.name,
        reason=f"launched exact declared transition as pid {record.pid}",
        argv=planned.argv,
        heavy=planned.heavy,
        dry_run=False,
        launcher_record=record,
    )


def advance_payload(result: AdvanceResult) -> dict[str, Any]:
    record = result.launcher_record
    return {
        "state": result.action,
        "experiment": result.experiment_name,
        "transition": result.transition_name,
        "reason": result.reason,
        "command": list(result.argv) if result.argv else None,
        "heavy": result.heavy,
        "dry_run": result.dry_run,
        "exit_code": result.exit_code,
        "launcher": (
            {
                "pid": record.pid,
                "supervisor_pid": record.supervisor_pid,
                "command_sha256": record.command_sha256,
                "launch_token": record.launch_token,
                "started_at": record.started_at,
                "stdout_path": record.stdout_path,
                "stderr_path": record.stderr_path,
                "terminal_path": record.terminal_path,
            }
            if record is not None
            else None
        ),
    }


def render_advance(result: AdvanceResult, *, as_json: bool = False) -> str:
    if as_json:
        return json.dumps(
            advance_payload(result),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
    command = " ".join(result.argv) if result.argv else "absent"
    return "\n".join(
        (
            f"State: {result.action}",
            f"Experiment: {result.experiment_name}",
            f"Transition: {result.transition_name or 'absent'}",
            f"Reason: {result.reason}",
            f"Command: {command}",
            f"Dry run: {'yes' if result.dry_run else 'no'}",
        )
    ) + "\n"


__all__ = [
    "advance_campaign",
    "authorize_retry",
    "advance_payload",
    "default_ledger_path",
    "plan_next_transition",
    "render_advance",
]
