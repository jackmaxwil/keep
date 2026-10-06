from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from mlx_vq.recovery_campaign import (
    AdvanceResult,
    CampaignObservation,
    LauncherRecord,
    ProcessObservation,
    advance_campaign,
    load_campaign_config,
    load_ledger,
    render_advance,
)
from mlx_vq.recovery_campaign.controller import default_ledger_path
from mlx_vq.recovery_campaign.ledger import append_event
from mlx_vq.recovery_campaign.ledger import LedgerError, LedgerUncertainCommitError
from mlx_vq.recovery_campaign.launcher import command_sha256, launch_transition


RECIPE = Path(__file__).parents[1] / "recipes/glm52_recovery_campaign_v1_20260711.yaml"


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
        manifest_state="absent",
    )
    return replace(base, **changes)


def test_dry_run_is_read_only_and_reports_exact_command(tmp_path: Path) -> None:
    config = load_campaign_config(RECIPE)
    ledger = tmp_path / "missing" / "campaign.jsonl"
    calls: list[str] = []

    result = advance_campaign(
        config,
        repo_root=tmp_path,
        ledger_path=ledger,
        dry_run=True,
        observe=lambda _config, _root: _observation(),
        load_events=lambda _path: (),
        append=lambda *_args, **_kwargs: calls.append("append"),
        launch=lambda *_args, **_kwargs: calls.append("launch"),
    )

    assert result.action == "resume"
    assert result.dry_run is True
    assert result.argv == config.transition("full75-rematerialize").argv
    assert calls == []
    assert not ledger.parent.exists()


def test_default_ledger_affects_dry_run_without_any_mutation(tmp_path: Path) -> None:
    config = load_campaign_config(RECIPE)
    ledger = default_ledger_path(config, tmp_path)
    ledger.parent.mkdir(parents=True)
    append_event(
        ledger,
        event_kind="audit_complete",
        experiment="not-declared",
        payload={"manifest_sha256": "a" * 64},
    )
    before_bytes = ledger.read_bytes()
    before_stat = ledger.stat()
    before_inventory = tuple(sorted(path.name for path in ledger.parent.iterdir()))

    result = advance_campaign(
        config,
        repo_root=tmp_path,
        ledger_path=None,
        dry_run=True,
        observe=lambda _config, _root: _observation(),
    )

    assert result.action == "blocked"
    assert ledger.read_bytes() == before_bytes
    after_stat = ledger.stat()
    assert (after_stat.st_ino, after_stat.st_size, after_stat.st_mtime_ns) == (
        before_stat.st_ino,
        before_stat.st_size,
        before_stat.st_mtime_ns,
    )
    assert tuple(sorted(path.name for path in ledger.parent.iterdir())) == before_inventory


def test_mutating_advance_reobserves_and_refuses_new_overlap(tmp_path: Path) -> None:
    config = load_campaign_config(RECIPE)
    declared = config.transition("full75-rematerialize")
    process = ProcessObservation(
        pid=77,
        command=declared.argv,
        elapsed_seconds=2.0,
        experiment_name=declared.experiment_name,
        transition_name=declared.name,
    )
    observations = iter(
        (
            _observation(),
            _observation(
                lock_held=True,
                lock_owner_known=True,
                lock_state="held",
                active_processes=(process,),
            ),
        )
    )
    calls: list[str] = []

    result = advance_campaign(
        config,
        repo_root=tmp_path,
        ledger_path=None,
        dry_run=False,
        observe=lambda _config, _root: next(observations),
        load_events=lambda _path: (),
        append=lambda *_args, **_kwargs: calls.append("append"),
        launch=lambda *_args, **_kwargs: calls.append("launch"),
    )

    assert result.action == "wait"
    assert calls == []


def test_mutating_resume_anchors_both_events_and_records_launch(tmp_path: Path) -> None:
    config = load_campaign_config(RECIPE)
    ledger = default_ledger_path(config, tmp_path)
    append_calls: list[dict[str, object]] = []

    def append(path: Path, **kwargs: object):
        append_calls.append(dict(kwargs))
        from mlx_vq.recovery_campaign.ledger import append_event

        return append_event(path, **kwargs)

    def launch(transition, **kwargs):
        return _record_files(tmp_path, LauncherRecord(
            pid=4242,
            command_sha256=command_sha256(transition.argv),
            started_at="2026-07-11T12:00:00.000000Z",
            stdout_path=f"artifacts/quality/{config.campaign}-launches/out.log",
            stderr_path=f"artifacts/quality/{config.campaign}-launches/err.log",
            experiment_name=transition.experiment_name,
            transition_name=transition.name,
            supervisor_pid=4243,
            launch_token=kwargs["launch_token"],
            terminal_path=f"artifacts/quality/{config.campaign}-launches/terminal.json",
        ))

    result = advance_campaign(
        config,
        repo_root=tmp_path,
        ledger_path=ledger,
        dry_run=False,
        observe=lambda _config, _root: _observation(),
        launch=launch,
        append=append,
    )

    assert result.action == "resume"
    assert result.launcher_record is not None
    assert result.launcher_record.pid == 4242
    assert [call["event_kind"] for call in append_calls] == [
        "transition_started",
        "transition_launched",
    ]
    assert append_calls[0]["expected_event_count"] == 0
    assert append_calls[0]["expected_head_sha256"] is None
    assert append_calls[1]["expected_event_count"] == 1
    assert append_calls[1]["expected_head_sha256"] == load_ledger(ledger)[0].event_sha256
    assert tuple(load_ledger(ledger)[0].payload["argv"]) == result.argv


def test_launcher_uses_direct_argv_without_outer_lock(tmp_path: Path) -> None:
    config = load_campaign_config(RECIPE)
    transition = config.transition("full75-rematerialize")
    observed: dict[str, object] = {}

    class Process:
        pid = 91

    def popen(argv, **kwargs):
        observed["argv"] = argv
        observed.update(kwargs)
        return Process()

    record = launch_transition(
        replace(transition, action="resume"),
        repo_root=tmp_path,
        campaign=config.campaign,
        popen_factory=popen,
        timestamp="2026-07-11T12:00:00.000000Z",
    )

    assert observed["argv"] == list(transition.argv)
    assert "shell" not in observed
    assert "preexec_fn" not in observed
    assert record.pid == 91
    assert record.command_sha256
    assert (tmp_path / record.stdout_path).is_file()
    assert (tmp_path / record.stderr_path).is_file()


def test_launcher_closes_parent_log_descriptors(tmp_path: Path) -> None:
    config = load_campaign_config(RECIPE)
    transition = replace(config.transition("full75-rematerialize"), action="resume")
    descriptors: list[int] = []

    class Process:
        pid = 92

    def popen(_argv, **kwargs):
        descriptors.extend((kwargs["stdout"].fileno(), kwargs["stderr"].fileno()))
        return Process()

    launch_transition(
        transition,
        repo_root=tmp_path,
        campaign=config.campaign,
        popen_factory=popen,
        timestamp="2026-07-11T12:00:01.000000Z",
    )

    for descriptor in descriptors:
        with pytest.raises(OSError):
            import os

            os.fstat(descriptor)


def test_launcher_rejects_symlinked_log_directory(tmp_path: Path) -> None:
    config = load_campaign_config(RECIPE)
    transition = replace(config.transition("full75-rematerialize"), action="resume")
    quality = tmp_path / "artifacts" / "quality"
    outside = tmp_path / "outside"
    quality.mkdir(parents=True)
    outside.mkdir()
    (quality / f"{config.campaign}-launches").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError):
        launch_transition(
            transition,
            repo_root=tmp_path,
            campaign=config.campaign,
            popen_factory=lambda *_args, **_kwargs: pytest.fail("must not launch"),
            timestamp="2026-07-11T12:00:02.000000Z",
        )

    assert list(outside.iterdir()) == []


def test_launcher_rejects_invalid_timestamp_before_filesystem_or_popen(
    tmp_path: Path,
) -> None:
    config = load_campaign_config(RECIPE)
    transition = replace(config.transition("full75-rematerialize"), action="resume")
    calls: list[str] = []

    with pytest.raises(ValueError):
        launch_transition(
            transition,
            repo_root=tmp_path,
            campaign=config.campaign,
            popen_factory=lambda *_args, **_kwargs: calls.append("popen"),
            timestamp="not-a-time",
        )

    assert calls == []
    assert list(tmp_path.iterdir()) == []


def test_advance_rendering_is_deterministic_json_and_text() -> None:
    result = AdvanceResult(
        action="blocked",
        experiment_name="ebss",
        transition_name=None,
        reason="implementation and human decision required",
        argv=(),
        heavy=False,
        dry_run=True,
        launcher_record=None,
        exit_code=3,
    )

    first = render_advance(result, as_json=True)
    second = render_advance(result, as_json=True)

    assert first == second
    assert json.loads(first)["state"] == "blocked"
    assert json.loads(first)["exit_code"] == 3
    assert "State: blocked" in render_advance(result)
    assert "Command: absent" in render_advance(result)


@pytest.mark.parametrize(
    "changes",
    (
        {"pid": True},
        {"pid": 0},
        {"command_sha256": "ABC"},
        {"started_at": "yesterday"},
        {"stdout_path": "/tmp/out"},
        {"stderr_path": "artifacts/quality/../err"},
        {"experiment_name": "bad\nname"},
        {"transition_name": ""},
    ),
)
def test_launcher_record_rejects_invalid_durable_identity(changes: dict[str, object]) -> None:
    values: dict[str, object] = {
        "pid": 1,
        "command_sha256": "a" * 64,
        "started_at": "2026-07-11T12:00:00.000000Z",
        "stdout_path": "artifacts/quality/glm52-recovery-v1-20260711-launches/out.log",
        "stderr_path": "artifacts/quality/glm52-recovery-v1-20260711-launches/err.log",
        "experiment_name": "full75-e8",
        "transition_name": "full75-rematerialize",
        "supervisor_pid": 2,
        "launch_token": "3" * 64,
        "terminal_path": "artifacts/quality/glm52-recovery-v1-20260711-launches/terminal.json",
    }
    values.update(changes)

    with pytest.raises((TypeError, ValueError)):
        LauncherRecord(**values)


def test_advance_result_rejects_invalid_state_or_record_consistency() -> None:
    with pytest.raises(ValueError):
        AdvanceResult(
            action="invented",
            experiment_name="full75-e8",
            transition_name=None,
            reason="bad",
            argv=(),
            heavy=False,
            dry_run=True,
            launcher_record=None,
        )
    record = LauncherRecord(
        pid=1,
        command_sha256="a" * 64,
        started_at="2026-07-11T12:00:00.000000Z",
        stdout_path="artifacts/quality/glm52-recovery-v1-20260711-launches/out.log",
        stderr_path="artifacts/quality/glm52-recovery-v1-20260711-launches/err.log",
        experiment_name="full75-e8",
        transition_name="full75-rematerialize",
        supervisor_pid=2,
        launch_token="3" * 64,
        terminal_path="artifacts/quality/glm52-recovery-v1-20260711-launches/terminal.json",
    )
    with pytest.raises(ValueError):
        AdvanceResult(
            action="ready",
            experiment_name="full75-e8",
            transition_name=None,
            reason="bad",
            argv=(),
            heavy=False,
            dry_run=True,
            launcher_record=record,
        )


def test_advance_json_keeps_full_durable_launcher_identity() -> None:
    record = LauncherRecord(
        pid=11,
        supervisor_pid=12,
        command_sha256="a" * 64,
        launch_token="b" * 64,
        started_at="2026-07-11T12:00:00.000000Z",
        stdout_path="artifacts/quality/campaign-launches/out.log",
        stderr_path="artifacts/quality/campaign-launches/err.log",
        terminal_path="artifacts/quality/campaign-launches/terminal.json",
        experiment_name="full75-e8",
        transition_name="full75-rematerialize",
    )
    result = AdvanceResult(
        action="resume",
        experiment_name="full75-e8",
        transition_name="full75-rematerialize",
        reason="launched",
        argv=("python", "producer.py"),
        heavy=True,
        dry_run=False,
        launcher_record=record,
    )

    launcher = json.loads(render_advance(result, as_json=True))["launcher"]

    assert launcher == {
        "pid": 11,
        "supervisor_pid": 12,
        "command_sha256": "a" * 64,
        "launch_token": "b" * 64,
        "started_at": "2026-07-11T12:00:00.000000Z",
        "stdout_path": "artifacts/quality/campaign-launches/out.log",
        "stderr_path": "artifacts/quality/campaign-launches/err.log",
        "terminal_path": "artifacts/quality/campaign-launches/terminal.json",
    }


def test_popen_failure_is_recorded_without_claiming_launched(tmp_path: Path) -> None:
    config = load_campaign_config(RECIPE)
    ledger = default_ledger_path(config, tmp_path)

    def fail_launch(*_args, **_kwargs):
        raise OSError("exec failed")

    result = advance_campaign(
        config,
        repo_root=tmp_path,
        ledger_path=ledger,
        dry_run=False,
        observe=lambda _config, _root: _observation(),
        launch=fail_launch,
    )

    assert result.action == "blocked"
    assert result.launcher_record is None
    assert result.exit_code != 0
    assert [event.event_kind for event in load_ledger(ledger)] == [
        "transition_started",
        "transition_launch_failed",
    ]


def test_uncertain_append_is_reconciled_by_reload_without_retry(tmp_path: Path) -> None:
    config = load_campaign_config(RECIPE)
    ledger = default_ledger_path(config, tmp_path)
    calls = 0

    def append(path, **kwargs):
        nonlocal calls
        calls += 1
        from mlx_vq.recovery_campaign.ledger import append_event

        event = append_event(path, **kwargs)
        if calls == 1:
            raise LedgerUncertainCommitError("published, durability uncertain")
        return event

    record = LauncherRecord(
        pid=9,
        command_sha256=command_sha256(
            config.transition("full75-rematerialize").argv
        ),
        started_at="2026-07-11T12:00:00.000000Z",
        stdout_path=f"artifacts/quality/{config.campaign}-launches/out.log",
        stderr_path=f"artifacts/quality/{config.campaign}-launches/err.log",
        experiment_name="full75-e8",
        transition_name="full75-rematerialize",
        supervisor_pid=10,
        launch_token="3" * 64,
        terminal_path=f"artifacts/quality/{config.campaign}-launches/terminal.json",
    )
    _record_files(tmp_path, record)

    result = advance_campaign(
        config,
        repo_root=tmp_path,
        ledger_path=ledger,
        dry_run=False,
        observe=lambda _config, _root: _observation(),
        append=append,
        launch=lambda *_args, **kwargs: replace(
            record,
            launch_token=kwargs["launch_token"],
        ),
    )

    assert result.launcher_record == replace(
        record,
        launch_token=result.launcher_record.launch_token,
    )
    assert calls == 2
    assert [event.event_kind for event in load_ledger(ledger)] == [
        "transition_started",
        "transition_launched",
    ]


def test_failed_launched_append_never_launches_twice(tmp_path: Path) -> None:
    config = load_campaign_config(RECIPE)
    ledger = default_ledger_path(config, tmp_path)
    launches = 0
    appends = 0

    def launch(*_args, **kwargs):
        nonlocal launches
        launches += 1
        return _record_files(tmp_path, LauncherRecord(
            pid=19,
            command_sha256=command_sha256(
                config.transition("full75-rematerialize").argv
            ),
            started_at="2026-07-11T12:00:00.000000Z",
            stdout_path=f"artifacts/quality/{config.campaign}-launches/out.log",
            stderr_path=f"artifacts/quality/{config.campaign}-launches/err.log",
            experiment_name="full75-e8",
            transition_name="full75-rematerialize",
            supervisor_pid=20,
            launch_token=kwargs["launch_token"],
            terminal_path=f"artifacts/quality/{config.campaign}-launches/terminal.json",
        ))

    def append(path, **kwargs):
        nonlocal appends
        appends += 1
        if appends == 2:
            raise LedgerError("anchor lost")
        from mlx_vq.recovery_campaign.ledger import append_event

        return append_event(path, **kwargs)

    result = advance_campaign(
        config,
        repo_root=tmp_path,
        ledger_path=ledger,
        dry_run=False,
        observe=lambda _config, _root: _observation(),
        append=append,
        launch=launch,
    )

    assert launches == 1
    assert result.action == "wait"
    assert result.launcher_record is not None
    assert result.exit_code == 4
    assert "do not retry" in result.reason


def test_launcher_supervisor_records_actual_terminal_exit(tmp_path: Path) -> None:
    transition = replace(
        load_campaign_config(RECIPE).transition("full75-rematerialize"),
        action="resume",
        argv=(sys.executable, "-c", "raise SystemExit(7)"),
    )

    record = launch_transition(
        transition,
        repo_root=tmp_path,
        campaign="supervisor-test",
        timestamp="2026-07-11T12:00:03.000000Z",
    )

    terminal = tmp_path / record.terminal_path
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and terminal.stat().st_size == 0:
        time.sleep(0.01)
    payload = json.loads(terminal.read_text())
    assert payload["exit_code"] == 7
    assert payload["producer_pid"] == record.pid
    assert payload["classification"] == "finished"
    _reap_supervisor(record)


def test_controller_rejects_forged_launcher_return_before_launched_append(
    tmp_path: Path,
) -> None:
    config = load_campaign_config(RECIPE)
    ledger = default_ledger_path(config, tmp_path)

    def forged(transition, **kwargs):
        return LauncherRecord(
            pid=22,
            command_sha256="f" * 64,
            started_at="2026-07-11T12:00:00.000000Z",
            stdout_path=f"artifacts/quality/{config.campaign}-launches/out.log",
            stderr_path=f"artifacts/quality/{config.campaign}-launches/err.log",
            experiment_name=transition.experiment_name,
            transition_name=transition.name,
            supervisor_pid=23,
            launch_token=kwargs["launch_token"],
            terminal_path=f"artifacts/quality/{config.campaign}-launches/terminal.json",
        )

    result = advance_campaign(
        config,
        repo_root=tmp_path,
        ledger_path=None,
        dry_run=False,
        observe=lambda _config, _root: _observation(),
        launch=forged,
    )

    assert result.action == "wait"
    assert "launcher" in result.reason
    assert [event.event_kind for event in load_ledger(ledger)] == [
        "transition_started",
    ]


def test_controller_rejects_producer_mislabeled_as_its_own_supervisor(
    tmp_path: Path,
) -> None:
    config = load_campaign_config(RECIPE)

    def naked_producer(transition, **kwargs):
        return _record_files(
            tmp_path,
            LauncherRecord(
                pid=31,
                supervisor_pid=31,
                command_sha256=command_sha256(transition.argv),
                started_at="2026-07-11T12:00:00.000000Z",
                stdout_path=(
                    f"artifacts/quality/{config.campaign}-launches/out.log"
                ),
                stderr_path=(
                    f"artifacts/quality/{config.campaign}-launches/err.log"
                ),
                terminal_path=(
                    f"artifacts/quality/{config.campaign}-launches/terminal.json"
                ),
                experiment_name=transition.experiment_name,
                transition_name=transition.name,
                launch_token=kwargs["launch_token"],
            ),
        )

    result = advance_campaign(
        config,
        repo_root=tmp_path,
        ledger_path=None,
        dry_run=False,
        observe=lambda _config, _root: _observation(),
        launch=naked_producer,
    )

    assert result.action == "wait"
    assert "untrusted" in result.reason


def test_launcher_detects_ancestor_rename_before_supervisor_popen(
    tmp_path: Path, monkeypatch
) -> None:
    config = load_campaign_config(RECIPE)
    transition = replace(config.transition("full75-rematerialize"), action="resume")

    def rename_ancestor(_chain, _stdout_fd, _stderr_fd) -> None:
        quality = tmp_path / "artifacts" / "quality"
        quality.rename(tmp_path / "artifacts" / "quality-renamed")

    monkeypatch.setattr(
        "mlx_vq.recovery_campaign.launcher._before_supervisor_launch",
        rename_ancestor,
    )

    with pytest.raises(OSError):
        launch_transition(
            transition,
            repo_root=tmp_path,
            campaign=config.campaign,
            timestamp="2026-07-11T12:00:04.000000Z",
        )


def test_supervisor_classifies_post_launch_ancestor_rename_as_uncertain(
    tmp_path: Path,
) -> None:
    transition = replace(
        load_campaign_config(RECIPE).transition("full75-rematerialize"),
        action="resume",
        argv=(
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "Path('artifacts/quality').rename('artifacts/quality-renamed')"
            ),
        ),
    )
    record = launch_transition(
        transition,
        repo_root=tmp_path,
        campaign="post-launch-test",
        timestamp="2026-07-11T12:00:05.000000Z",
    )

    terminal = (
        tmp_path
        / "artifacts"
        / "quality-renamed"
        / "post-launch-test-launches"
        / Path(record.terminal_path).name
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and (
        not terminal.exists() or terminal.stat().st_size == 0
    ):
        time.sleep(0.01)
    payload = json.loads(terminal.read_text())
    assert payload["classification"] == "uncertain"
    assert "identity_error" in payload
    _reap_supervisor(record)
