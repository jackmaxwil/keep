from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from mlx_vq.recovery_campaign import (
    CampaignObservation,
    EvidenceRecord,
    ExperimentObservation,
    ProcessObservation,
    VerificationResult,
    append_event,
    append_evidence,
    load_campaign_config,
    load_ledger,
    plan_next_transition,
)
from mlx_vq.recovery_campaign.launcher import command_sha256


RECIPE = Path(__file__).parents[1] / "recipes/glm52_recovery_campaign_v1_20260711.yaml"


def _observation(**changes: object) -> CampaignObservation:
    base = CampaignObservation(
        lock_held=False,
        lock_owner_known=False,
        active_processes=(),
        recovered_groups=0,
        expected_groups=225,
        manifest_sha256=None,
        contradictions=(),
        campaign="glm52-recovery-v1-20260711",
        experiment_name="full75-e8",
        lock_state="free",
        artifact_links=0,
        expected_artifact_links=225,
        manifest_state="absent",
        manifest_file_sha256=None,
        manifest_body_sha256=None,
        recovered_groups_state="absent",
        artifact_links_state="absent",
    )
    return replace(base, **changes)


def _stage_payload(kind: str, digest: str = "a" * 64) -> dict[str, object]:
    if kind == "transition_launched":
        config = load_campaign_config(RECIPE)
        declared = config.transition("full75-rematerialize")
        return {
            "transition": "full75-rematerialize",
            "command_sha256": command_sha256(declared.argv),
            "pid": 36724,
            "supervisor_pid": 36725,
            "started_at": "2026-07-11T12:00:00.000000Z",
            "stdout_path": "artifacts/quality/launches/out.log",
            "stderr_path": "artifacts/quality/launches/err.log",
            "terminal_path": "artifacts/quality/launches/terminal.json",
            "launch_token": "3" * 64,
            "campaign_config_sha256": config.campaign_config_sha256,
        }
    if kind == "transition_started":
        config = load_campaign_config(RECIPE)
        return {
            "transition": "full75-rematerialize",
            "argv": list(load_campaign_config(RECIPE).transition("full75-rematerialize").argv),
            "expected_artifacts": list(
                config.transition("full75-rematerialize").expected_artifacts
            ),
            "campaign_config_sha256": config.campaign_config_sha256,
            "launch_token": "3" * 64,
        }
    return {}


def _events(tmp_path: Path, *kinds: tuple[str, str]):
    identity = hashlib.sha256(repr(kinds).encode()).hexdigest()[:12]
    ledger = tmp_path / f"campaign-{identity}.jsonl"
    config = load_campaign_config(RECIPE)
    for experiment, kind in kinds:
        if kind in {
            "audit_complete",
            "reevaluation_complete",
            "attribution_complete",
            "experiment_complete",
        }:
            release = experiment == "rotations" and kind == "experiment_complete"
            evidence = append_evidence(
                ledger,
                experiment=experiment,
                evidence=EvidenceRecord(
                    artifact_path="artifacts/quality/controller-evidence.json",
                    content_sha256=hashlib.sha256(
                        f"{experiment}:{kind}".encode()
                    ).hexdigest(),
                    manifest_identity_sha256="a" * 64,
                    candidate_identity_sha256="d" * 64 if release else None,
                    baseline_identity_sha256="b" * 64 if release else None,
                    evidence_class="release" if release else "diagnostic_only",
                    release_eligible=release,
                    recovery_levers=("selection_only",),
                    argv=("python", "verify.py"),
                    verification_results=(
                        VerificationResult(
                            name=f"{kind}_verification",
                            argv=("python", "verify.py"),
                            exit_code=0,
                            stdout_sha256="1" * 64,
                            stderr_sha256="2" * 64,
                        ),
                    ),
                ),
            )
            payload: dict[str, object] = {
                "campaign_config_sha256": config.campaign_config_sha256,
                "manifest_sha256": "a" * 64,
                "evidence_event_sha256": evidence.event_sha256,
            }
            if kind == "experiment_complete":
                payload["human_decision_reference"] = f"decision:{experiment}"
        else:
            payload = _stage_payload(kind)
        append_event(
            ledger,
            event_kind=kind,
            experiment=experiment,
            payload=payload,
        )
    return load_ledger(ledger)


def _physical(name: str, *, complete: bool) -> ExperimentObservation:
    expected_groups = 225 if name == "full75-e8" else 24
    return ExperimentObservation(
        experiment_name=name,
        recovered_groups=expected_groups if complete else 0,
        expected_groups=expected_groups,
        recovered_groups_state="present" if complete else "absent",
        artifact_links=225 if complete else 0,
        expected_artifact_links=225,
        artifact_links_state="present" if complete else "absent",
        manifest_state="valid" if complete else "absent",
        manifest_file_sha256="a" * 64 if complete else None,
        manifest_body_sha256="b" * 64 if complete else None,
        contradictions=(),
    )


def _append_terminal_attempt(
    ledger: Path,
    *,
    completed: bool = False,
    exit_code: int = -15,
    token_digit: str = "3",
):
    config = load_campaign_config(RECIPE)
    started = _stage_payload("transition_started")
    started["launch_token"] = token_digit * 64
    append_event(
        ledger,
        event_kind="transition_started",
        experiment="full75-e8",
        payload=started,
    )
    launched = _stage_payload("transition_launched")
    launched["launch_token"] = token_digit * 64
    append_event(
        ledger,
        event_kind="transition_launched",
        experiment="full75-e8",
        payload=launched,
    )
    return append_event(
        ledger,
        event_kind="transition_completed" if completed else "transition_finished",
        experiment="full75-e8",
        payload={
            "transition": "full75-rematerialize",
            "launch_token": token_digit * 64,
            "pid": 36724,
            "supervisor_pid": 36725,
            "command_sha256": command_sha256(
                config.transition("full75-rematerialize").argv
            ),
            "exit_code": 0 if completed else exit_code,
            "campaign_config_sha256": config.campaign_config_sha256,
            "physical_complete": completed,
            "manifest_sha256": "a" * 64 if completed else None,
            "terminal_path": "artifacts/quality/launches/terminal.json",
        },
    )


def test_healthy_exact_active_owner_plans_wait() -> None:
    config = load_campaign_config(RECIPE)
    declared = config.transition("full75-rematerialize")
    process = ProcessObservation(
        pid=36724,
        command=declared.argv,
        elapsed_seconds=100.0,
        experiment_name=declared.experiment_name,
        transition_name=declared.name,
    )

    planned = plan_next_transition(
        config,
        _observation(
            lock_held=True,
            lock_owner_known=True,
            lock_state="held",
            active_processes=(process,),
            recovered_groups=66,
            recovered_groups_state="present",
        ),
        (),
    )

    assert planned.action == "wait"
    assert planned.name == declared.name
    assert planned.argv == declared.argv


def test_incomplete_idle_declared_transition_plans_exact_resume() -> None:
    config = load_campaign_config(RECIPE)
    declared = config.transition("full75-rematerialize")

    planned = plan_next_transition(config, _observation(), ())

    assert planned.action == "resume"
    assert planned.name == declared.name
    assert planned.argv == declared.argv
    assert planned.heavy is True
    assert planned.expected_artifacts == declared.expected_artifacts


def test_contradiction_and_unknown_lock_owner_fail_closed() -> None:
    config = load_campaign_config(RECIPE)

    contradictory = plan_next_transition(
        config,
        _observation(contradictions=("manifest lies",)),
        (),
    )
    unknown_owner = plan_next_transition(
        config,
        _observation(lock_held=True, lock_state="held"),
        (),
    )

    assert contradictory.action == "blocked"
    assert "manifest lies" in contradictory.reason
    assert unknown_owner.action == "blocked"
    assert "owner" in unknown_owner.reason


def test_declared_process_command_or_lock_mismatch_is_blocked() -> None:
    config = load_campaign_config(RECIPE)
    declared = config.transition("full75-rematerialize")
    wrong = ProcessObservation(
        pid=12,
        command=("python", "wrong.py"),
        elapsed_seconds=1.0,
        experiment_name=declared.experiment_name,
        transition_name=declared.name,
    )

    planned = plan_next_transition(
        config,
        _observation(
            lock_held=True,
            lock_owner_known=True,
            lock_state="held",
            active_processes=(wrong,),
        ),
        (),
    )

    assert planned.action == "blocked"
    assert "exact declared command" in planned.reason


def test_complete_materialization_advances_through_evidence_stages(
    tmp_path: Path,
) -> None:
    config = load_campaign_config(RECIPE)
    complete_artifact = _observation(
        manifest_state="valid",
        manifest_sha256="a" * 64,
        manifest_file_sha256="a" * 64,
        manifest_body_sha256="b" * 64,
        recovered_groups=225,
        recovered_groups_state="present",
        artifact_links=225,
        artifact_links_state="present",
    )

    assert plan_next_transition(config, complete_artifact, ()).action == "audit"
    assert (
        plan_next_transition(
            config,
            complete_artifact,
            _events(tmp_path, ("full75-e8", "audit_complete")),
        ).action
        == "reevaluate"
    )
    assert (
        plan_next_transition(
            config,
            complete_artifact,
            _events(
                tmp_path,
                ("full75-e8", "audit_complete"),
                ("full75-e8", "reevaluation_complete"),
            ),
        ).action
        == "attribute"
    )
    assert (
        plan_next_transition(
            config,
            complete_artifact,
            _events(
                tmp_path,
                ("full75-e8", "audit_complete"),
                ("full75-e8", "reevaluation_complete"),
                ("full75-e8", "attribution_complete"),
            ),
        ).action
        == "ready"
    )


def test_null_transition_is_explicit_implementation_and_human_block(
    tmp_path: Path,
) -> None:
    config = load_campaign_config(RECIPE)
    events = _events(
        tmp_path,
        ("full75-e8", "audit_complete"),
        ("full75-e8", "reevaluation_complete"),
        ("full75-e8", "attribution_complete"),
        ("full75-e8", "experiment_complete"),
        ("worst8-e8p", "audit_complete"),
        ("worst8-e8p", "reevaluation_complete"),
        ("worst8-e8p", "attribution_complete"),
        ("worst8-e8p", "experiment_complete"),
    )

    planned = plan_next_transition(
        config,
        _observation(
            experiment_name=None,
            expected_groups=0,
            experiment_observations=(
                _physical("full75-e8", complete=True),
                _physical("worst8-e8p", complete=True),
                _physical("ebss", complete=False),
                _physical("rotations", complete=False),
            ),
        ),
        events,
    )

    assert planned.action == "blocked"
    assert planned.experiment_name == "ebss"
    assert "implementation" in planned.reason
    assert "human" in planned.reason


def test_all_declared_experiments_complete_plans_complete(tmp_path: Path) -> None:
    config = load_campaign_config(RECIPE)
    stages = tuple(
        (experiment.name, stage)
        for experiment in config.experiments
        for stage in (
            "audit_complete",
            "reevaluation_complete",
            "attribution_complete",
            "experiment_complete",
        )
    )
    events = _events(tmp_path, *stages)

    planned = plan_next_transition(
        config,
        _observation(
            experiment_name=None,
            expected_groups=0,
            experiment_observations=tuple(
                _physical(experiment.name, complete=True)
                for experiment in config.experiments
            ),
        ),
        events,
    )

    assert planned.action == "complete"
    assert planned.experiment_name == config.campaign


def test_launched_without_terminal_and_stale_pid_blocks_resume(tmp_path: Path) -> None:
    config = load_campaign_config(RECIPE)
    events = _events(
        tmp_path,
        ("full75-e8", "transition_started"),
        ("full75-e8", "transition_launched"),
    )

    planned = plan_next_transition(config, _observation(), events)

    assert planned.action == "blocked"
    assert "terminal" in planned.reason


def test_failed_terminal_requires_explicit_retry_authorization(tmp_path: Path) -> None:
    config = load_campaign_config(RECIPE)
    ledger = tmp_path / "failed.jsonl"
    _append_terminal_attempt(ledger)

    planned = plan_next_transition(config, _observation(), load_ledger(ledger))

    assert planned.action == "blocked"
    assert planned.reason == (
        "terminal recovery outcome is not an authenticated completion "
        "(actual exit_code=-15); explicit retry authorization is required"
    )


def test_retry_authorization_for_latest_terminal_allows_resume(tmp_path: Path) -> None:
    config = load_campaign_config(RECIPE)
    ledger = tmp_path / "authorized.jsonl"
    terminal = _append_terminal_attempt(ledger)
    append_event(
        ledger,
        event_kind="retry_authorized",
        experiment="full75-e8",
        payload={
            "campaign": config.campaign,
            "experiment": "full75-e8",
            "transition": "full75-rematerialize",
            "terminal_event_sha256": terminal.event_sha256,
            "exit_code": -15,
            "reason": "operator confirmed safe resume",
        },
    )

    planned = plan_next_transition(config, _observation(), load_ledger(ledger))

    assert planned.action == "resume"
    assert planned.name == "full75-rematerialize"


def test_retry_authorization_for_stale_terminal_does_not_unblock_new_failure(
    tmp_path: Path,
) -> None:
    config = load_campaign_config(RECIPE)
    ledger = tmp_path / "stale-authorization.jsonl"
    first_terminal = _append_terminal_attempt(ledger, token_digit="3")
    append_event(
        ledger,
        event_kind="retry_authorized",
        experiment="full75-e8",
        payload={
            "campaign": config.campaign,
            "experiment": "full75-e8",
            "transition": "full75-rematerialize",
            "terminal_event_sha256": first_terminal.event_sha256,
            "exit_code": -15,
            "reason": "authorize only the first failure",
        },
    )
    second_terminal = _append_terminal_attempt(
        ledger,
        exit_code=9,
        token_digit="4",
    )

    planned = plan_next_transition(config, _observation(), load_ledger(ledger))

    assert second_terminal.event_sha256 != first_terminal.event_sha256
    assert planned.action == "blocked"
    assert "actual exit_code=9" in planned.reason


def test_unmatched_transition_started_blocks_duplicate_launch(tmp_path: Path) -> None:
    config = load_campaign_config(RECIPE)
    events = _events(tmp_path, ("full75-e8", "transition_started"))

    planned = plan_next_transition(config, _observation(), events)

    assert planned.action == "blocked"
    assert "unmatched transition_started" in planned.reason


def test_launched_record_hash_must_bind_exact_declared_argv(tmp_path: Path) -> None:
    config = load_campaign_config(RECIPE)
    ledger = tmp_path / "wrong-command-hash.jsonl"
    append_event(
        ledger,
        event_kind="transition_started",
        experiment="full75-e8",
        payload=_stage_payload("transition_started"),
    )
    launched = _stage_payload("transition_launched")
    launched["command_sha256"] = "f" * 64
    append_event(
        ledger,
        event_kind="transition_launched",
        experiment="full75-e8",
        payload=launched,
    )

    planned = plan_next_transition(config, _observation(), load_ledger(ledger))

    assert planned.action == "blocked"
    assert "command hash" in planned.reason


def test_stage_events_are_ordered_unique_and_bound_to_current_manifest(
    tmp_path: Path,
) -> None:
    config = load_campaign_config(RECIPE)
    complete_artifact = _observation(
        manifest_state="valid",
        manifest_sha256="a" * 64,
        manifest_file_sha256="a" * 64,
        manifest_body_sha256="b" * 64,
        recovered_groups=225,
        recovered_groups_state="present",
        artifact_links=225,
        artifact_links_state="present",
    )
    out_of_order = _events(
        tmp_path,
        ("full75-e8", "reevaluation_complete"),
    )

    planned = plan_next_transition(config, complete_artifact, out_of_order)

    assert planned.action == "blocked"
    assert "order" in planned.reason


def test_duplicate_or_foreign_stage_event_fails_closed(tmp_path: Path) -> None:
    config = load_campaign_config(RECIPE)
    duplicate = _events(
        tmp_path,
        ("full75-e8", "audit_complete"),
        ("full75-e8", "audit_complete"),
    )
    foreign = _events(tmp_path, ("not-declared", "audit_complete"))

    duplicate_plan = plan_next_transition(config, _observation(), duplicate)
    foreign_plan = plan_next_transition(config, _observation(), foreign)

    assert duplicate_plan.action == "blocked"
    assert "duplicate" in duplicate_plan.reason
    assert foreign_plan.action == "blocked"
    assert "undeclared experiment" in foreign_plan.reason


def test_bare_completion_cannot_bypass_current_physical_evidence(
    tmp_path: Path,
) -> None:
    config = load_campaign_config(RECIPE)
    complete = _events(
        tmp_path,
        ("full75-e8", "audit_complete"),
        ("full75-e8", "reevaluation_complete"),
        ("full75-e8", "attribution_complete"),
        ("full75-e8", "experiment_complete"),
    )

    planned = plan_next_transition(config, _observation(), complete)

    assert planned.action == "blocked"
    assert "physical" in planned.reason


def test_child_stage_cannot_predate_accepted_dependency(tmp_path: Path) -> None:
    config = load_campaign_config(RECIPE)
    events = _events(
        tmp_path,
        ("worst8-e8p", "audit_complete"),
        ("full75-e8", "audit_complete"),
        ("full75-e8", "reevaluation_complete"),
        ("full75-e8", "attribution_complete"),
        ("full75-e8", "experiment_complete"),
    )

    planned = plan_next_transition(config, _observation(), events)

    assert planned.action == "blocked"
    assert "dependency" in planned.reason
