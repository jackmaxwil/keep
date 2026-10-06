from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sys
import threading
import time
import warnings
from dataclasses import replace
from pathlib import Path

import pytest

from mlx_vq.recovery_campaign import (
    AdvanceResult,
    CampaignObservation,
    ExperimentObservation,
    EvidenceRecord,
    LauncherRecord,
    LedgerError,
    LedgerUncertainCommitError,
    VerificationResult,
    advance_campaign,
    append_event,
    append_evidence,
    load_campaign_config,
    plan_next_transition,
)
from mlx_vq.recovery_campaign.launcher import command_sha256, launch_transition
from mlx_vq.recovery_campaign.controller import (
    ControllerMutexBusy,
    _controller_mutex,
    controller_mutex_path,
)
from mlx_vq.recovery_campaign.supervisor import record_terminal_outcome


RECIPE = Path(__file__).parents[1] / "recipes/glm52_recovery_campaign_v1_20260711.yaml"


def _snapshot(name: str, **changes: object) -> ExperimentObservation:
    expected_groups = 225 if name == "full75-e8" else 24
    base = ExperimentObservation(
        experiment_name=name,
        recovered_groups=0,
        expected_groups=expected_groups,
        recovered_groups_state="absent",
        artifact_links=0,
        expected_artifact_links=225,
        artifact_links_state="absent",
        manifest_state="absent",
        manifest_file_sha256=None,
        manifest_body_sha256=None,
        contradictions=(),
    )
    return replace(base, **changes)


def _complete_snapshot(name: str, digest: str) -> ExperimentObservation:
    expected_groups = 225 if name == "full75-e8" else 24
    return _snapshot(
        name,
        recovered_groups=expected_groups,
        recovered_groups_state="present",
        artifact_links=225,
        artifact_links_state="present",
        manifest_state="valid",
        manifest_file_sha256=digest,
        manifest_body_sha256="e" * 64,
    )


def _observation(config, **changes: object) -> CampaignObservation:
    snapshots = tuple(_snapshot(experiment.name) for experiment in config.experiments)
    base = CampaignObservation(
        lock_held=False,
        lock_owner_known=False,
        active_processes=(),
        recovered_groups=0,
        expected_groups=225,
        manifest_sha256=None,
        contradictions=(),
        campaign=config.campaign,
        experiment_name="full75-e8",
        lock_state="free",
        manifest_state="absent",
        experiment_observations=snapshots,
    )
    return replace(base, **changes)


def _record_files(root: Path, record: LauncherRecord) -> LauncherRecord:
    for relative in (record.stdout_path, record.stderr_path, record.terminal_path):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(mode=0o600, exist_ok=True)
        path.chmod(0o600)
    return record


def _reap_supervisor(record: LauncherRecord) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        pid, _status = os.waitpid(record.supervisor_pid, os.WNOHANG)
        if pid == record.supervisor_pid:
            return
        time.sleep(0.01)
    pytest.fail(f"supervisor {record.supervisor_pid} did not exit")


def _wait_for_terminal(root: Path, record: LauncherRecord) -> dict[str, object]:
    terminal = root / record.terminal_path
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and terminal.stat().st_size == 0:
        time.sleep(0.01)
    assert terminal.stat().st_size > 0
    return json.loads(terminal.read_text())


def _fake_recipe_repo(
    root: Path,
    *,
    exit_code: int = 0,
    sleep_seconds: float = 0.0,
):
    recipe = root / "campaign.yaml"
    recipe.write_text(RECIPE.read_text())
    python = root / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    runner = root / "benchmarks" / "run_glm52_recovery_wave1.py"
    runner.parent.mkdir()
    runner.write_text(
        "import time\n"
        f"time.sleep({sleep_seconds!r})\n"
        f"raise SystemExit({exit_code})\n"
    )
    return load_campaign_config(recipe)


def _terminal_control(
    config,
    root: Path,
    ledger: Path,
    *,
    producer_pid: int = 41,
    supervisor_pid: int = 42,
) -> dict[str, object]:
    transition = config.transition("full75-rematerialize")
    launch_token = "7" * 64
    started_at = "2026-07-11T12:00:09.000000Z"
    stdout_path = "artifacts/quality/test-launches/test.stdout.log"
    stderr_path = "artifacts/quality/test-launches/test.stderr.log"
    terminal_path = "artifacts/quality/test-launches/test.terminal.json"
    append_event(
        ledger,
        event_kind="transition_started",
        experiment=transition.experiment_name,
        payload={
            "transition": transition.name,
            "argv": list(transition.argv),
            "expected_artifacts": list(transition.expected_artifacts),
            "campaign_config_sha256": config.campaign_config_sha256,
            "launch_token": launch_token,
        },
    )
    append_event(
        ledger,
        event_kind="transition_launched",
        experiment=transition.experiment_name,
        payload={
            "pid": producer_pid,
            "supervisor_pid": supervisor_pid,
            "command_sha256": command_sha256(transition.argv),
            "started_at": started_at,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "terminal_path": terminal_path,
            "launch_token": launch_token,
            "transition": transition.name,
            "campaign_config_sha256": config.campaign_config_sha256,
        },
    )
    return {
        "argv": list(transition.argv),
        "repo_root": str(root),
        "campaign": config.campaign,
        "experiment": transition.experiment_name,
        "transition": transition.name,
        "command_sha256": command_sha256(transition.argv),
        "launch_token": launch_token,
        "started_at": started_at,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "terminal_path": terminal_path,
        "terminal_context": {
            "campaign_config_path": config.campaign_config_path,
            "campaign_config_sha256": config.campaign_config_sha256,
            "ledger_path": str(ledger),
        },
    }


def _evidence(
    manifest_sha256: str,
    *,
    stage: str,
    release: bool = False,
) -> EvidenceRecord:
    return EvidenceRecord(
        artifact_path="artifacts/quality/controller-evidence.json",
        content_sha256=hashlib.sha256(
            f"{manifest_sha256}:{stage}".encode()
        ).hexdigest(),
        manifest_identity_sha256=manifest_sha256,
        candidate_identity_sha256="d" * 64 if release else None,
        baseline_identity_sha256="b" * 64 if release else None,
        evidence_class="release" if release else "diagnostic_only",
        release_eligible=release,
        recovery_levers=("selection_only",),
        argv=("python", "verify.py"),
        verification_results=(
            VerificationResult(
                name=f"{stage}_verification",
                argv=("python", "verify.py"),
                exit_code=0,
                stdout_sha256="1" * 64,
                stderr_sha256="2" * 64,
            ),
        ),
    )


def _append_stage(
    ledger: Path,
    config,
    experiment: str,
    stage: str,
    manifest_sha256: str,
    *,
    release: bool = False,
) -> None:
    evidence = append_evidence(
        ledger,
        experiment=experiment,
        evidence=_evidence(manifest_sha256, stage=stage, release=release),
    )
    payload: dict[str, object] = {
        "campaign_config_sha256": config.campaign_config_sha256,
        "manifest_sha256": manifest_sha256,
        "evidence_event_sha256": evidence.event_sha256,
    }
    if stage == "experiment_complete":
        payload["human_decision_reference"] = f"decision:{experiment}:accepted"
    append_event(
        ledger,
        event_kind=stage,
        experiment=experiment,
        payload=payload,
    )


def _append_packet(
    ledger: Path,
    config,
    experiment: str,
    manifest_sha256: str,
    *,
    release: bool = False,
) -> None:
    for stage in (
        "audit_complete",
        "reevaluation_complete",
        "attribution_complete",
        "experiment_complete",
    ):
        _append_stage(
            ledger,
            config,
            experiment,
            stage,
            manifest_sha256,
            release=release and stage == "experiment_complete",
        )


def test_campaign_config_has_stable_semantic_matrix_fingerprint(tmp_path: Path) -> None:
    first = load_campaign_config(RECIPE)
    copied = tmp_path / "campaign.yaml"
    copied.write_text(RECIPE.read_text())
    second = load_campaign_config(copied)

    assert first.campaign_config_sha256 == second.campaign_config_sha256
    assert len(first.campaign_config_sha256) == 64

    copied.write_text(copied.read_text().replace("prompt_count: 22", "prompt_count: 23"))
    changed = load_campaign_config(copied)
    assert changed.campaign_config_sha256 != first.campaign_config_sha256


def test_observation_exposes_every_experiment_including_null_transitions(
    tmp_path: Path, monkeypatch
) -> None:
    config = load_campaign_config(RECIPE)
    monkeypatch.setattr("mlx_vq.recovery_campaign.observer._read_ps_output", lambda: "")

    from mlx_vq.recovery_campaign import observe_campaign

    observation = observe_campaign(config, tmp_path)

    assert [item.experiment_name for item in observation.experiment_observations] == [
        "full75-e8",
        "worst8-e8p",
        "ebss",
        "rotations",
    ]
    assert observation.snapshot("ebss").manifest_state == "absent"
    assert observation.snapshot("rotations").manifest_state == "absent"


def test_foreign_campaign_and_unknown_selected_experiment_fail_closed() -> None:
    config = load_campaign_config(RECIPE)
    foreign = plan_next_transition(
        config,
        _observation(config, campaign="other-campaign"),
        (),
    )
    unknown = plan_next_transition(
        config,
        _observation(config, experiment_name="not-declared"),
        (),
    )

    assert foreign.action == "blocked"
    assert "campaign" in foreign.reason
    assert unknown.action == "blocked"
    assert "undeclared" in unknown.reason


def test_advance_result_rejects_forged_complete_shape() -> None:
    with pytest.raises(ValueError):
        AdvanceResult(
            action="complete",
            experiment_name="glm52-recovery-v1-20260711",
            transition_name="forged",
            reason="forged",
            argv=("python", "forged.py"),
            heavy=True,
            dry_run=False,
            launcher_record=None,
            exit_code=99,
        )


def test_explicit_relative_ledger_is_anchored_to_repo_root(
    tmp_path: Path, monkeypatch
) -> None:
    config = load_campaign_config(RECIPE)
    repo = tmp_path / "repo"
    elsewhere = tmp_path / "elsewhere"
    repo.mkdir()
    elsewhere.mkdir()
    ledger = repo / "relative.jsonl"
    append_event(
        ledger,
        event_kind="audit_complete",
        experiment="not-declared",
        payload={"manifest_sha256": "a" * 64},
    )
    monkeypatch.chdir(elsewhere)

    result = advance_campaign(
        config,
        repo_root=repo,
        ledger_path=Path("relative.jsonl"),
        dry_run=True,
        observe=lambda _config, _root: _observation(config),
    )

    assert result.action == "blocked"
    assert "undeclared" in result.reason


def test_global_controller_mutex_allows_one_canonical_launch_across_threads(
    tmp_path: Path,
) -> None:
    config = load_campaign_config(RECIPE)
    (tmp_path / "artifacts" / "quality").mkdir(parents=True)
    entered = threading.Event()
    release = threading.Event()
    launches: list[str] = []
    results: list[AdvanceResult] = []
    errors: list[BaseException] = []

    def launch(transition, **kwargs):
        launches.append(transition.name)
        entered.set()
        assert release.wait(timeout=5)
        return _record_files(tmp_path, LauncherRecord(
            pid=1234,
            command_sha256=command_sha256(transition.argv),
            started_at="2026-07-11T12:00:00.000000Z",
            stdout_path=f"artifacts/quality/{config.campaign}-launches/out.log",
            stderr_path=f"artifacts/quality/{config.campaign}-launches/err.log",
            experiment_name=transition.experiment_name,
            transition_name=transition.name,
            supervisor_pid=1235,
            launch_token=kwargs["launch_token"],
            terminal_path=f"artifacts/quality/{config.campaign}-launches/terminal.json",
        ))

    def worker(ledger: Path | None) -> None:
        try:
            results.append(
                advance_campaign(
                    config,
                    repo_root=tmp_path,
                    ledger_path=ledger,
                    dry_run=False,
                    observe=lambda _config, _root: _observation(config),
                    launch=launch,
                )
            )
        except BaseException as error:
            errors.append(error)

    first = threading.Thread(target=worker, args=(None,))
    second = threading.Thread(target=worker, args=(None,))
    first.start()
    assert entered.wait(timeout=5)
    second.start()
    second.join(timeout=5)
    release.set()
    first.join(timeout=5)

    assert errors == []
    assert launches == ["full75-rematerialize"]
    assert sorted(result.action for result in results) == ["blocked", "resume"]


def test_controller_mutex_authority_survives_artifact_ancestor_replacement(
    tmp_path: Path,
) -> None:
    config = load_campaign_config(RECIPE)

    with _controller_mutex(config, tmp_path):
        artifacts = tmp_path / "artifacts"
        artifacts.rename(tmp_path / "detached-artifacts")
        (artifacts / "quality").mkdir(parents=True)

        result: list[str] = []

        def contender() -> None:
            try:
                with _controller_mutex(config, tmp_path):
                    result.append("acquired")
            except ControllerMutexBusy:
                result.append("busy")

        thread = threading.Thread(target=contender)
        thread.start()
        thread.join(timeout=5)

        assert not thread.is_alive()
        assert result == ["busy"]


def test_two_different_custom_mutating_ledgers_cannot_launch(tmp_path: Path) -> None:
    config = load_campaign_config(RECIPE)
    launches: list[str] = []
    errors: list[BaseException] = []

    def worker(path: Path) -> None:
        try:
            advance_campaign(
                config,
                repo_root=tmp_path,
                ledger_path=path,
                dry_run=False,
                observe=lambda _config, _root: _observation(config),
                load_events=lambda _path: (),
                launch=lambda *_args, **_kwargs: launches.append("launch"),
            )
        except BaseException as error:
            errors.append(error)

    threads = [
        threading.Thread(target=worker, args=(Path("first.jsonl"),)),
        threading.Thread(target=worker, args=(Path("second.jsonl"),)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert launches == []
    assert len(errors) == 2
    assert all("canonical" in str(error) for error in errors)


def test_policy_target_comes_from_ledger_not_observer_materialization_selection() -> None:
    config = load_campaign_config(RECIPE)
    full = _complete_snapshot("full75-e8", "a" * 64)
    observation = _observation(
        config,
        experiment_name="worst8-e8p",
        experiment_observations=(
            full,
            _snapshot("worst8-e8p"),
            _snapshot("ebss"),
            _snapshot("rotations"),
        ),
    )

    planned = plan_next_transition(config, observation, ())

    assert planned.action == "audit"
    assert planned.experiment_name == "full75-e8"


def test_stage_requires_corresponding_evidence_registration(tmp_path: Path) -> None:
    config = load_campaign_config(RECIPE)
    ledger = tmp_path / "campaign.jsonl"
    append_event(
        ledger,
        event_kind="audit_complete",
        experiment="full75-e8",
        payload={
            "campaign_config_sha256": config.campaign_config_sha256,
            "manifest_sha256": "a" * 64,
            "evidence_event_sha256": "f" * 64,
        },
    )
    observation = _observation(
        config,
        experiment_observations=(
            _complete_snapshot("full75-e8", "a" * 64),
            _snapshot("worst8-e8p"),
            _snapshot("ebss"),
            _snapshot("rotations"),
        ),
    )

    from mlx_vq.recovery_campaign import load_ledger

    planned = plan_next_transition(config, observation, load_ledger(ledger))
    assert planned.action == "blocked"
    assert "evidence" in planned.reason


def test_one_evidence_registration_cannot_complete_multiple_stages(
    tmp_path: Path,
) -> None:
    config = load_campaign_config(RECIPE)
    ledger = tmp_path / "campaign.jsonl"
    evidence = append_evidence(
        ledger,
        experiment="full75-e8",
        evidence=EvidenceRecord(
            artifact_path="artifacts/quality/multi-stage.json",
            content_sha256="c" * 64,
            manifest_identity_sha256="a" * 64,
            candidate_identity_sha256=None,
            baseline_identity_sha256=None,
            evidence_class="diagnostic_only",
            release_eligible=False,
            recovery_levers=("selection_only",),
            argv=("python", "verify.py"),
            verification_results=(
                VerificationResult(
                    name="audit_complete_verification",
                    argv=("python", "verify.py"),
                    exit_code=0,
                    stdout_sha256="1" * 64,
                    stderr_sha256="2" * 64,
                ),
            ),
        ),
    )
    payload = {
        "campaign_config_sha256": config.campaign_config_sha256,
        "manifest_sha256": "a" * 64,
        "evidence_event_sha256": evidence.event_sha256,
    }
    append_event(
        ledger,
        event_kind="audit_complete",
        experiment="full75-e8",
        payload=payload,
    )
    append_event(
        ledger,
        event_kind="reevaluation_complete",
        experiment="full75-e8",
        payload=payload,
    )
    from mlx_vq.recovery_campaign import load_ledger

    observation = _observation(
        config,
        experiment_observations=(
            _complete_snapshot("full75-e8", "a" * 64),
            _snapshot("worst8-e8p"),
            _snapshot("ebss"),
            _snapshot("rotations"),
        ),
    )
    planned = plan_next_transition(config, observation, load_ledger(ledger))

    assert planned.action == "blocked"
    assert "cannot satisfy multiple" in planned.reason


def test_multi_stage_verification_profile_cannot_complete_all_stages(
    tmp_path: Path,
) -> None:
    config = load_campaign_config(RECIPE)
    ledger = tmp_path / "campaign.jsonl"
    stages = (
        "audit_complete",
        "reevaluation_complete",
        "attribution_complete",
        "experiment_complete",
    )
    shared = EvidenceRecord(
        artifact_path="artifacts/quality/reused-evidence.json",
        content_sha256="c" * 64,
        manifest_identity_sha256="a" * 64,
        candidate_identity_sha256=None,
        baseline_identity_sha256=None,
        evidence_class="diagnostic_only",
        release_eligible=False,
        recovery_levers=("selection_only",),
        argv=("python", "verify.py"),
        verification_results=tuple(
            VerificationResult(
                name=f"{stage}_verification",
                argv=("python", "verify.py", stage),
                exit_code=0,
                stdout_sha256="1" * 64,
                stderr_sha256="2" * 64,
            )
            for stage in stages
        ),
    )
    for stage in stages:
        evidence = append_evidence(
            ledger,
            experiment="full75-e8",
            evidence=shared,
        )
        payload: dict[str, object] = {
            "campaign_config_sha256": config.campaign_config_sha256,
            "manifest_sha256": "a" * 64,
            "evidence_event_sha256": evidence.event_sha256,
        }
        if stage == "experiment_complete":
            payload["human_decision_reference"] = "decision:full75-e8:accepted"
        append_event(
            ledger,
            event_kind=stage,
            experiment="full75-e8",
            payload=payload,
        )
    from mlx_vq.recovery_campaign import load_ledger

    observation = _observation(
        config,
        experiment_observations=(
            _complete_snapshot("full75-e8", "a" * 64),
            _snapshot("worst8-e8p"),
            _snapshot("ebss"),
            _snapshot("rotations"),
        ),
    )

    planned = plan_next_transition(config, observation, load_ledger(ledger))

    assert planned.action == "blocked"
    assert "exactly one stage-specific" in planned.reason


def test_reused_evidence_content_identity_cannot_complete_multiple_stages(
    tmp_path: Path,
) -> None:
    config = load_campaign_config(RECIPE)
    ledger = tmp_path / "campaign.jsonl"
    for stage in ("audit_complete", "reevaluation_complete"):
        evidence = append_evidence(
            ledger,
            experiment="full75-e8",
            evidence=EvidenceRecord(
                artifact_path=f"artifacts/quality/{stage}.json",
                content_sha256="c" * 64,
                manifest_identity_sha256="a" * 64,
                candidate_identity_sha256=None,
                baseline_identity_sha256=None,
                evidence_class="diagnostic_only",
                release_eligible=False,
                recovery_levers=("selection_only",),
                argv=("python", "verify.py", stage),
                verification_results=(
                    VerificationResult(
                        name=f"{stage}_verification",
                        argv=("python", "verify.py", stage),
                        exit_code=0,
                        stdout_sha256="1" * 64,
                        stderr_sha256="2" * 64,
                    ),
                ),
            ),
        )
        append_event(
            ledger,
            event_kind=stage,
            experiment="full75-e8",
            payload={
                "campaign_config_sha256": config.campaign_config_sha256,
                "manifest_sha256": "a" * 64,
                "evidence_event_sha256": evidence.event_sha256,
            },
        )
    from mlx_vq.recovery_campaign import load_ledger

    observation = _observation(
        config,
        experiment_observations=(
            _complete_snapshot("full75-e8", "a" * 64),
            _snapshot("worst8-e8p"),
            _snapshot("ebss"),
            _snapshot("rotations"),
        ),
    )

    planned = plan_next_transition(config, observation, load_ledger(ledger))

    assert planned.action == "blocked"
    assert "payload/content identity" in planned.reason


def test_evidence_bound_stage_advances_and_swapped_manifest_blocks(
    tmp_path: Path,
) -> None:
    config = load_campaign_config(RECIPE)
    ledger = tmp_path / "campaign.jsonl"
    _append_stage(ledger, config, "full75-e8", "audit_complete", "a" * 64)
    from mlx_vq.recovery_campaign import load_ledger

    events = load_ledger(ledger)
    valid = _observation(
        config,
        experiment_observations=(
            _complete_snapshot("full75-e8", "a" * 64),
            _snapshot("worst8-e8p"),
            _snapshot("ebss"),
            _snapshot("rotations"),
        ),
    )
    swapped = replace(
        valid,
        experiment_observations=(
            _complete_snapshot("full75-e8", "9" * 64),
            _snapshot("worst8-e8p"),
            _snapshot("ebss"),
            _snapshot("rotations"),
        ),
    )

    assert plan_next_transition(config, valid, events).action == "reevaluate"
    blocked = plan_next_transition(config, swapped, events)
    assert blocked.action == "blocked"
    assert "manifest" in blocked.reason


def test_null_transition_with_authenticated_physical_candidate_reaches_audit(
    tmp_path: Path,
) -> None:
    config = load_campaign_config(RECIPE)
    ledger = tmp_path / "campaign.jsonl"
    _append_packet(ledger, config, "full75-e8", "a" * 64)
    _append_packet(ledger, config, "worst8-e8p", "b" * 64)
    from mlx_vq.recovery_campaign import load_ledger

    observation = _observation(
        config,
        experiment_name=None,
        experiment_observations=(
            _complete_snapshot("full75-e8", "a" * 64),
            _complete_snapshot("worst8-e8p", "b" * 64),
            _complete_snapshot("ebss", "c" * 64),
            _snapshot("rotations"),
        ),
    )

    planned = plan_next_transition(config, observation, load_ledger(ledger))
    assert planned.action == "audit"
    assert planned.experiment_name == "ebss"


def test_recovery_producer_heavy_lock_is_nonblocking_by_default(
    tmp_path: Path, monkeypatch
) -> None:
    from mlx_vq.convert import glm52_recovery_materialize as materializer

    operations: list[int] = []

    def flock(_descriptor: int, operation: int) -> None:
        operations.append(operation)
        if operation & fcntl.LOCK_NB:
            raise BlockingIOError("busy")

    monkeypatch.setattr(materializer.fcntl, "flock", flock)

    with pytest.raises(materializer.RecoveryHeavyLockBusy):
        with materializer._cooperating_file_lock(tmp_path / "heavy.lock"):
            pytest.fail("busy producer lock must never queue or enter")

    assert operations == [fcntl.LOCK_EX | fcntl.LOCK_NB]


def test_recovery_producer_rejects_symlink_heavy_lock(tmp_path: Path) -> None:
    from mlx_vq.convert import glm52_recovery_materialize as materializer

    target = tmp_path / "target.lock"
    target.touch()
    link = tmp_path / "heavy.lock"
    link.symlink_to(target)

    with pytest.raises(OSError):
        with materializer._cooperating_file_lock(link):
            pytest.fail("symlink heavy lock must not be followed")


def test_recovery_producer_reauthenticates_visible_lock_after_acquisition(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from mlx_vq.convert import glm52_recovery_materialize as materializer

    lock_path = tmp_path / "heavy.lock"
    lock_path.touch()
    parked = tmp_path / "parked.lock"
    real_flock = materializer.fcntl.flock
    swapped = False

    def swap_during_acquire(descriptor: int, operation: int) -> object:
        nonlocal swapped
        if operation & fcntl.LOCK_EX and not swapped:
            swapped = True
            lock_path.rename(parked)
            lock_path.touch()
        return real_flock(descriptor, operation)

    monkeypatch.setattr(materializer.fcntl, "flock", swap_during_acquire)

    with pytest.raises(ValueError, match="lock path"):
        with materializer._cooperating_file_lock(lock_path):
            pytest.fail("detached old lock inode must never authorize mutation")


def test_planner_rejects_snapshot_supplied_expected_count() -> None:
    config = load_campaign_config(RECIPE)
    forged = _complete_snapshot("full75-e8", "a" * 64)
    forged = replace(forged, expected_groups=1, recovered_groups=1)
    observation = _observation(
        config,
        experiment_observations=(
            forged,
            _snapshot("worst8-e8p"),
            _snapshot("ebss"),
            _snapshot("rotations"),
        ),
    )

    planned = plan_next_transition(config, observation, ())
    assert planned.action == "blocked"
    assert "expected" in planned.reason


@pytest.mark.parametrize(
    ("physical_after_exit", "terminal_kind"),
    ((True, "transition_completed"), (False, "transition_finished")),
)
def test_supervisor_appends_terminal_only_after_fresh_physical_authentication(
    tmp_path: Path,
    physical_after_exit: bool,
    terminal_kind: str,
) -> None:
    config = load_campaign_config(RECIPE)
    ledger = (
        tmp_path
        / "artifacts"
        / "quality"
        / f"{config.campaign}-ledger.jsonl"
    )
    ledger.parent.mkdir(parents=True)
    control = _terminal_control(config, tmp_path, ledger)
    after = _observation(
        config,
        experiment_observations=(
            (
                _complete_snapshot("full75-e8", "a" * 64)
                if physical_after_exit
                else _snapshot("full75-e8")
            ),
            _snapshot("worst8-e8p"),
            _snapshot("ebss"),
            _snapshot("rotations"),
        ),
    )
    outcome = record_terminal_outcome(
        control,
        producer_pid=41,
        supervisor_pid=42,
        exit_code=0,
        observer=lambda _config, _root: after,
    )
    from mlx_vq.recovery_campaign import load_ledger

    assert [event.event_kind for event in load_ledger(ledger)] == [
        "transition_started",
        "transition_launched",
        terminal_kind,
    ]
    assert outcome["classification"] == (
        "completed" if physical_after_exit else "finished"
    )
    assert outcome["physical_complete"] is physical_after_exit


def test_terminal_monitor_rejects_launched_pid_identity_mismatch(
    tmp_path: Path,
) -> None:
    config = load_campaign_config(RECIPE)
    ledger = (
        tmp_path
        / "artifacts"
        / "quality"
        / f"{config.campaign}-ledger.jsonl"
    )
    ledger.parent.mkdir(parents=True)
    control = _terminal_control(config, tmp_path, ledger)

    with pytest.raises(ValueError, match="transition_launched identity mismatch"):
        record_terminal_outcome(
            control,
            producer_pid=99,
            supervisor_pid=42,
            exit_code=0,
            observer=lambda _config, _root: _observation(config),
        )


def test_terminal_monitor_retries_transient_parent_ledger_publication_race(
    tmp_path: Path,
) -> None:
    config = load_campaign_config(RECIPE)
    ledger = (
        tmp_path
        / "artifacts"
        / "quality"
        / f"{config.campaign}-ledger.jsonl"
    )
    ledger.parent.mkdir(parents=True)
    control = _terminal_control(config, tmp_path, ledger)
    from mlx_vq.recovery_campaign import load_ledger

    calls = 0

    def transient_load(path: Path):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise LedgerError("parent publication still in progress")
        return load_ledger(path)

    outcome = record_terminal_outcome(
        control,
        producer_pid=41,
        supervisor_pid=42,
        exit_code=7,
        observer=lambda _config, _root: _observation(config),
        load_events=transient_load,
    )

    assert calls >= 2
    assert outcome["classification"] == "finished"


def test_nonzero_terminal_exit_is_visible_and_never_auto_relaunches(
    tmp_path: Path,
) -> None:
    config = _fake_recipe_repo(tmp_path, exit_code=7)
    absent = _observation(config)
    observations = iter((absent, absent))
    first = advance_campaign(
        config,
        repo_root=tmp_path,
        ledger_path=None,
        dry_run=False,
        observe=lambda _config, _root: next(observations),
    )
    assert first.launcher_record is not None
    ledger = (
        tmp_path
        / "artifacts"
        / "quality"
        / f"{config.campaign}-ledger.jsonl"
    )
    from mlx_vq.recovery_campaign import load_ledger

    terminal = _wait_for_terminal(tmp_path, first.launcher_record)
    assert terminal["classification"] == "finished"
    assert terminal["exit_code"] == 7
    followup = advance_campaign(
        config,
        repo_root=tmp_path,
        ledger_path=None,
        dry_run=True,
        observe=lambda _config, _root: absent,
    )
    assert followup.action == "blocked"
    assert followup.exit_code == 7
    assert "exit_code=7" in followup.reason
    _reap_supervisor(first.launcher_record)


def test_supervisor_retains_controller_mutex_until_terminal_exit(tmp_path: Path) -> None:
    config = _fake_recipe_repo(tmp_path, sleep_seconds=0.3)
    absent = _observation(config)
    observations = iter((absent, absent))
    result = advance_campaign(
        config,
        repo_root=tmp_path,
        ledger_path=None,
        dry_run=False,
        observe=lambda _config, _root: next(observations),
    )
    assert result.launcher_record is not None
    mutex = controller_mutex_path(config, tmp_path)
    descriptor = os.open(mutex, os.O_RDONLY)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(descriptor)

    terminal = tmp_path / result.launcher_record.terminal_path
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and terminal.stat().st_size == 0:
        time.sleep(0.01)
    acquired = False
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not acquired:
        descriptor = os.open(mutex, os.O_RDONLY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            time.sleep(0.01)
        else:
            acquired = True
        finally:
            os.close(descriptor)
    assert acquired
    _reap_supervisor(result.launcher_record)


def test_terminal_monitor_reconciles_uncertain_append_without_duplicate(
    tmp_path: Path,
) -> None:
    config = load_campaign_config(RECIPE)
    ledger = (
        tmp_path
        / "artifacts"
        / "quality"
        / f"{config.campaign}-ledger.jsonl"
    )
    ledger.parent.mkdir(parents=True)
    control = _terminal_control(
        config,
        tmp_path,
        ledger,
        producer_pid=51,
        supervisor_pid=52,
    )
    complete = _observation(
        config,
        experiment_observations=(
            _complete_snapshot("full75-e8", "a" * 64),
            _snapshot("worst8-e8p"),
            _snapshot("ebss"),
            _snapshot("rotations"),
        ),
    )
    uncertain = False

    def uncertain_terminal(path, **kwargs):
        nonlocal uncertain
        from mlx_vq.recovery_campaign.ledger import append_event as real_append

        event = real_append(path, **kwargs)
        if kwargs["event_kind"] == "transition_completed" and not uncertain:
            uncertain = True
            raise LedgerUncertainCommitError("terminal published, fsync uncertain")
        return event

    outcome = record_terminal_outcome(
        control,
        producer_pid=51,
        supervisor_pid=52,
        exit_code=0,
        observer=lambda _config, _root: complete,
        append=uncertain_terminal,
    )
    from mlx_vq.recovery_campaign import load_ledger

    events = load_ledger(ledger)
    assert [event.event_kind for event in events].count("transition_completed") == 1
    assert outcome["classification"] == "completed"


def test_posix_spawn_launcher_is_warning_clean_with_an_active_thread(
    tmp_path: Path,
) -> None:
    transition = replace(
        load_campaign_config(RECIPE).transition("full75-rematerialize"),
        action="resume",
        argv=(sys.executable, "-c", "raise SystemExit(0)"),
    )
    stop = threading.Event()
    active = threading.Thread(target=stop.wait)
    active.start()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            record = launch_transition(
                transition,
                repo_root=tmp_path,
                campaign="threaded-spawn-test",
                timestamp="2026-07-11T12:00:10.000000Z",
            )
        assert _wait_for_terminal(tmp_path, record)["classification"] == "finished"
        _reap_supervisor(record)
    finally:
        stop.set()
        active.join(timeout=5)


def test_supervisor_allows_unrelated_sibling_directory_churn(tmp_path: Path) -> None:
    transition = replace(
        load_campaign_config(RECIPE).transition("full75-rematerialize"),
        action="resume",
        argv=(
            sys.executable,
            "-c",
            "from pathlib import Path; Path('artifacts/quality/sibling').mkdir()",
        ),
    )
    record = launch_transition(
        transition,
        repo_root=tmp_path,
        campaign="sibling-churn-test",
        timestamp="2026-07-11T12:00:11.000000Z",
    )

    assert _wait_for_terminal(tmp_path, record)["classification"] == "finished"
    _reap_supervisor(record)


def test_spawn_fd_staging_handles_mutex_source_equal_to_supervisor_target(
    tmp_path: Path,
) -> None:
    transition = replace(
        load_campaign_config(RECIPE).transition("full75-rematerialize"),
        action="resume",
        argv=(sys.executable, "-c", "raise SystemExit(0)"),
    )
    target_fd = 8
    try:
        backup_fd = os.dup(target_fd)
    except OSError:
        backup_fd = None
    source_fd = os.open(os.devnull, os.O_RDONLY)
    try:
        os.dup2(source_fd, target_fd)
        record = launch_transition(
            transition,
            repo_root=tmp_path,
            campaign="low-fd-staging-test",
            controller_mutex_fd=target_fd,
            timestamp="2026-07-11T12:00:12.000000Z",
        )
    finally:
        if source_fd != target_fd:
            os.close(source_fd)
        if backup_fd is None:
            if source_fd == target_fd:
                os.close(target_fd)
            else:
                try:
                    os.close(target_fd)
                except OSError:
                    pass
        else:
            os.dup2(backup_fd, target_fd)
            os.close(backup_fd)

    assert _wait_for_terminal(tmp_path, record)["classification"] == "finished"
    _reap_supervisor(record)
