from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx_vq.build.cli import build_parser, main
from mlx_vq.recovery_campaign import AdvanceResult, CampaignObservation, ProcessObservation
from mlx_vq.recovery_campaign.verification import (
    AVAILABLE_VERIFICATION_PROFILES,
    VerificationCheckResult,
    VerificationRunResult,
)
from mlx_vq.recovery_campaign.lanes import LaneError, get_lane
from mlx_vq.recovery_campaign import cli as recovery_cli
from mlx_vq.recovery_campaign import render as recovery_render
from mlx_vq.recovery_campaign.config import load_campaign_config
from mlx_vq.recovery_campaign.ledger import append_event, load_ledger
from mlx_vq.recovery_campaign.launcher import command_sha256


RECIPE = Path(__file__).parents[1] / "recipes/glm52_recovery_campaign_v1_20260711.yaml"


def test_recovery_campaign_status_help_is_registered(capsys) -> None:
    parser = build_parser()
    args = parser.parse_args(["recovery", "campaign", "status", "--json"])

    assert args.json is True
    assert args.config.name == "glm52_recovery_campaign_v1_20260711.yaml"

    try:
        parser.parse_args(["recovery", "campaign", "status", "--help"])
    except SystemExit as error:
        assert error.code == 0
    assert "--config" in capsys.readouterr().out


def test_recovery_campaign_advance_help_is_registered(capsys) -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["recovery", "campaign", "advance", "--dry-run", "--json"]
    )

    assert args.dry_run is True
    assert args.json is True
    assert args.ledger is None

    with pytest.raises(SystemExit) as raised:
        parser.parse_args(["recovery", "campaign", "advance", "--help"])
    assert raised.value.code == 0
    output = capsys.readouterr().out
    assert "--ledger" in output
    assert "--dry-run" in output


def _append_cli_terminal(ledger: Path, *, completed: bool):
    config = load_campaign_config(RECIPE)
    transition = config.transition("full75-rematerialize")
    launch_token = "3" * 64
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
            "pid": 36724,
            "supervisor_pid": 36725,
            "command_sha256": command_sha256(transition.argv),
            "started_at": "2026-07-11T12:00:00.000000Z",
            "stdout_path": "artifacts/quality/launches/out.log",
            "stderr_path": "artifacts/quality/launches/err.log",
            "terminal_path": "artifacts/quality/launches/terminal.json",
            "launch_token": launch_token,
            "transition": transition.name,
            "campaign_config_sha256": config.campaign_config_sha256,
        },
    )
    return append_event(
        ledger,
        event_kind="transition_completed" if completed else "transition_finished",
        experiment=transition.experiment_name,
        payload={
            "transition": transition.name,
            "launch_token": launch_token,
            "pid": 36724,
            "supervisor_pid": 36725,
            "command_sha256": command_sha256(transition.argv),
            "exit_code": 0 if completed else -15,
            "campaign_config_sha256": config.campaign_config_sha256,
            "physical_complete": completed,
            "manifest_sha256": "a" * 64 if completed else None,
            "terminal_path": "artifacts/quality/launches/terminal.json",
        },
    )


def test_recovery_campaign_authorize_retry_appends_once(
    tmp_path: Path,
    capsys,
) -> None:
    ledger = tmp_path / "campaign.jsonl"
    terminal = _append_cli_terminal(ledger, completed=False)
    argv = [
        "recovery",
        "campaign",
        "authorize-retry",
        "--config",
        str(RECIPE),
        "--ledger",
        str(ledger),
        "--transition",
        "full75-rematerialize",
        "--reason",
        "operator confirmed safe resume",
    ]

    assert main(argv) == 0
    event_sha256 = capsys.readouterr().out.strip()
    authorization = load_ledger(ledger)[-1]
    assert event_sha256 == authorization.event_sha256
    assert authorization.event_kind == "retry_authorized"
    assert authorization.payload["terminal_event_sha256"] == terminal.event_sha256
    assert authorization.payload["exit_code"] == -15

    assert main(argv) == 2
    assert "already exists" in capsys.readouterr().err


def test_recovery_campaign_authorize_retry_refuses_completed_terminal(
    tmp_path: Path,
    capsys,
) -> None:
    ledger = tmp_path / "campaign.jsonl"
    _append_cli_terminal(ledger, completed=True)

    rc = main(
        [
            "recovery",
            "campaign",
            "authorize-retry",
            "--config",
            str(RECIPE),
            "--ledger",
            str(ledger),
            "--transition",
            "full75-rematerialize",
            "--reason",
            "not needed",
        ]
    )

    assert rc == 2
    assert "transition_completed" in capsys.readouterr().err


def test_recovery_campaign_status_json_is_deterministic_and_complete(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr("mlx_vq.recovery_campaign.observer._read_ps_output", lambda: "")
    monkeypatch.chdir(tmp_path)

    assert main(["recovery", "campaign", "status", "--config", str(RECIPE), "--json"]) == 0
    first = capsys.readouterr().out
    assert main(["recovery", "campaign", "status", "--config", str(RECIPE), "--json"]) == 0
    second = capsys.readouterr().out

    assert first == second
    payload = json.loads(first)
    assert payload == {
        "active_processes": [],
        "artifact_links": {"expected": 225, "observed": None, "state": "absent"},
        "campaign": "glm52-recovery-v1-20260711",
        "contradictions": [],
        "experiment": "full75-e8",
        "groups": {"expected": 225, "observed": None, "state": "absent"},
        "lock": {"owner_known": False, "state": "absent"},
        "manifest": {"body_sha256": None, "file_sha256": None, "state": "absent"},
        "throughput_groups_per_second": None,
        "eta_seconds": None,
    }


def test_recovery_campaign_status_human_render_includes_honest_absence(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr("mlx_vq.recovery_campaign.observer._read_ps_output", lambda: "")
    monkeypatch.chdir(tmp_path)

    assert main(["recovery", "campaign", "status", "--config", str(RECIPE)]) == 0

    output = capsys.readouterr().out
    assert "Campaign: glm52-recovery-v1-20260711" in output
    assert "Experiment: full75-e8" in output
    assert "Lock: absent (owner known: no)" in output
    assert "Recovered groups: absent (expected 225)" in output
    assert "Artifact links: absent (expected 225)" in output
    assert "Manifest: absent" in output
    assert "Throughput: absent" in output
    assert "ETA: absent" in output


def test_recovery_campaign_status_config_errors_exit_two(tmp_path: Path, capsys) -> None:
    rc = main(
        [
            "recovery",
            "campaign",
            "status",
            "--config",
            str(tmp_path / "missing.yaml"),
        ]
    )

    assert rc == 2
    assert "error:" in capsys.readouterr().err


def test_recovery_campaign_status_corrupt_observation_exits_three(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    output = tmp_path / "artifacts/quality/glm52-recovery-wave1-mixed75-artifact-20260710"
    output.mkdir(parents=True)
    payload = b'{"manifest_body_sha256":"' + b"0" * 64 + b'"}\n'
    (output / "conversion-manifest.json").write_bytes(payload)
    monkeypatch.setattr("mlx_vq.recovery_campaign.observer._read_ps_output", lambda: "")
    monkeypatch.chdir(tmp_path)

    rc = main(["recovery", "campaign", "status", "--config", str(RECIPE), "--json"])

    assert rc == 3
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["manifest"]["state"] == "invalid"
    assert rendered["manifest"]["file_sha256"] == hashlib.sha256(payload).hexdigest()
    assert rendered["contradictions"]


def test_recovery_campaign_advance_dry_run_default_is_read_only(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr("mlx_vq.recovery_campaign.observer._read_ps_output", lambda: "")
    monkeypatch.chdir(tmp_path)

    rc = main(
        [
            "recovery",
            "campaign",
            "advance",
            "--config",
            str(RECIPE),
            "--dry-run",
            "--json",
        ]
    )

    assert rc == 0
    assert json.loads(capsys.readouterr().out)["state"] == "resume"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("result", "argv", "expected"),
    (
        (
            AdvanceResult(
                action="blocked",
                experiment_name="full75-e8",
                transition_name=None,
                reason="contradiction",
                argv=(),
                heavy=False,
                dry_run=True,
                launcher_record=None,
                exit_code=3,
            ),
            ["--dry-run"],
            3,
        ),
        (
            AdvanceResult(
                action="wait",
                experiment_name="full75-e8",
                transition_name="full75-rematerialize",
                reason="active",
                argv=("python",),
                heavy=True,
                dry_run=False,
                launcher_record=None,
                exit_code=4,
            ),
            [],
            4,
        ),
    ),
)
def test_recovery_campaign_advance_exit_semantics(
    result: AdvanceResult,
    argv: list[str],
    expected: int,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        "mlx_vq.recovery_campaign.cli.advance_campaign",
        lambda *_args, **_kwargs: result,
    )

    rc = main(
        [
            "recovery",
            "campaign",
            "advance",
            "--config",
            str(RECIPE),
            *argv,
        ]
    )

    assert rc == expected
    assert "State:" in capsys.readouterr().out


def test_recovery_campaign_advance_config_error_exits_two(tmp_path: Path, capsys) -> None:
    rc = main(
        [
            "recovery",
            "campaign",
            "advance",
            "--config",
            str(tmp_path / "missing.yaml"),
            "--dry-run",
        ]
    )

    assert rc == 2
    assert "error:" in capsys.readouterr().err


def test_recovery_campaign_custom_mutating_ledger_exits_two(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr("mlx_vq.recovery_campaign.observer._read_ps_output", lambda: "")
    monkeypatch.chdir(tmp_path)

    rc = main(
        [
            "recovery",
            "campaign",
            "advance",
            "--config",
            str(RECIPE),
            "--ledger",
            "custom.jsonl",
        ]
    )

    assert rc == 2
    assert "canonical" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def _verification_result(*, ready: bool) -> VerificationRunResult:
    return VerificationRunResult(
        profile="glm52-recovery-campaign",
        profile_sha256="a" * 64,
        status="passed" if ready else "blocked",
        commit_ready=ready,
        staged_paths=(),
        scope_violations=(),
        checks=(
            VerificationCheckResult(
                name="adversarial-review",
                kind="review",
                status="passed" if ready else "blocked",
                reason=None if ready else "review record absent",
            ),
        ),
    )


def test_both_verification_aliases_share_the_same_runner(monkeypatch, capsys) -> None:
    calls: list[tuple[str, Path]] = []

    def fake_run(profile: str, repo_root: Path) -> VerificationRunResult:
        calls.append((profile, repo_root))
        return _verification_result(ready=True)

    monkeypatch.setattr("mlx_vq.recovery_campaign.cli.run_verification_profile", fake_run)

    assert main(["verify", "glm52-recovery-campaign", "--json"]) == 0
    first = capsys.readouterr().out
    assert main(
        ["recovery", "campaign", "verify", "glm52-recovery-campaign", "--json"]
    ) == 0
    second = capsys.readouterr().out

    assert first == second
    assert json.loads(first)["commit_ready"] is True
    assert calls == [
        ("glm52-recovery-campaign", Path.cwd()),
        ("glm52-recovery-campaign", Path.cwd()),
    ]


def test_both_verification_help_surfaces_list_all_profiles(capsys) -> None:
    parser = build_parser()
    for argv in (("verify", "--help"), ("recovery", "campaign", "verify", "--help")):
        with pytest.raises(SystemExit) as raised:
            parser.parse_args(list(argv))
        assert raised.value.code == 0
        output = capsys.readouterr().out
        assert all(name in output for name in AVAILABLE_VERIFICATION_PROFILES)


@pytest.mark.parametrize(
    "argv",
    (
        ["verify", "missing"],
        ["recovery", "campaign", "verify", "missing"],
    ),
)
def test_unknown_verification_profile_exits_two(argv: list[str], capsys) -> None:
    assert main(argv) == 2
    error = capsys.readouterr().err
    assert "unknown verification profile 'missing'" in error
    assert all(name in error for name in AVAILABLE_VERIFICATION_PROFILES)


def test_recovery_lanes_help_and_dry_run_parser_are_registered(capsys) -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["recovery", "lanes", "run", "recovery-review-audit", "--dry-run", "--json"]
    )
    assert args.dry_run is True
    assert args.json is True
    with pytest.raises(SystemExit) as raised:
        parser.parse_args(["recovery", "lanes", "run", "--help"])
    assert raised.value.code == 0
    output = capsys.readouterr().out
    assert "--dry-run" in output
    assert "recovery-review-audit" in output


def test_recovery_lanes_dry_run_is_deterministic_and_read_only(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        "mlx_vq.recovery_campaign.cli._available_lane_prerequisites",
        lambda lane, _repo_root: lane.prerequisites,
    )
    monkeypatch.chdir(tmp_path)
    before = tuple(tmp_path.iterdir())
    argv = [
        "recovery",
        "lanes",
        "run",
        "recovery-review-audit",
        "--dry-run",
        "--json",
    ]
    assert main(argv) == 0
    first = capsys.readouterr().out
    assert main(argv) == 0
    second = capsys.readouterr().out
    assert first == second
    assert tuple(tmp_path.iterdir()) == before == ()
    payload = json.loads(first)
    assert payload["worker_launched"] is False
    assert payload["status"] == "ready"


def test_recovery_lanes_missing_authenticated_prerequisites_fails_closed(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "recovery",
                "lanes",
                "run",
                "recovery-review-audit",
                "--dry-run",
            ]
        )
        == 3
    )
    assert "prerequisite" in capsys.readouterr().err


@pytest.mark.parametrize("ancestry_returncode", (1, 128))
def test_recovery_lane_commit_prerequisite_requires_current_branch_ancestry(
    ancestry_returncode: int, monkeypatch
) -> None:
    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=ancestry_returncode, stdout=b"")

    monkeypatch.setattr(recovery_cli.subprocess, "run", fake_run)
    available = recovery_cli._available_lane_prerequisites(
        get_lane("recovery-review-audit"), Path.cwd()
    )
    assert not any(item.kind == "verification" for item in available)
    assert len(calls) == 1
    assert calls[0][0] == "/usr/bin/git"
    assert calls[0][3:5] == ["merge-base", "--is-ancestor"]


def test_recovery_lane_commit_prerequisite_rejects_wrong_patch_bytes(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        calls.append(argv)
        if "merge-base" in argv:
            return SimpleNamespace(returncode=0, stdout=b"")
        return SimpleNamespace(returncode=0, stdout=b"not the approved patch")

    monkeypatch.setattr(recovery_cli.subprocess, "run", fake_run)
    available = recovery_cli._available_lane_prerequisites(
        get_lane("recovery-review-audit"), Path.cwd()
    )
    assert not any(item.kind == "verification" for item in available)
    assert len(calls) == 2
    assert all(call[0] == "/usr/bin/git" for call in calls)
    assert "--no-ext-diff" in calls[1]
    assert "--no-textconv" in calls[1]


@pytest.mark.parametrize("timeout_phase", ("ancestry", "show"))
def test_recovery_lane_commit_prerequisite_fails_closed_on_git_timeout(
    timeout_phase: str, monkeypatch
) -> None:
    timeouts: list[int] = []

    def fake_run(argv, **kwargs):
        timeouts.append(kwargs["timeout"])
        if timeout_phase == "show" and "merge-base" in argv:
            return SimpleNamespace(returncode=0, stdout=b"")
        raise recovery_cli.subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(recovery_cli.subprocess, "run", fake_run)
    available = recovery_cli._available_lane_prerequisites(
        get_lane("recovery-review-audit"), Path.cwd()
    )
    assert not any(item.kind == "verification" for item in available)
    assert timeouts == ([5] if timeout_phase == "ancestry" else [5, 5])


@pytest.mark.parametrize("leaf_kind", ("symlink", "fifo"))
def test_recovery_lane_review_prerequisite_rejects_nonregular_leaf(
    leaf_kind: str, tmp_path: Path, monkeypatch
) -> None:
    package = tmp_path / ".superpowers/sdd/recovery-campaign-task-6-review.diff"
    package.parent.mkdir(parents=True)
    if leaf_kind == "symlink":
        target = tmp_path / "review-target.diff"
        target.write_bytes(b"replacement")
        package.symlink_to(target)
    else:
        os.mkfifo(package)
    monkeypatch.setattr(
        recovery_cli.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout=b""),
    )
    available = recovery_cli._available_lane_prerequisites(
        get_lane("recovery-review-audit"), tmp_path
    )
    assert not any(item.kind == "review" for item in available)


def test_recovery_lanes_unknown_name_exits_two(capsys) -> None:
    assert main(["recovery", "lanes", "run", "missing", "--dry-run"]) == 2
    assert "unknown lane" in capsys.readouterr().err


def test_recovery_lanes_malformed_declaration_exits_two(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        recovery_cli,
        "get_lane",
        lambda _name: (_ for _ in ()).throw(LaneError("malformed declaration")),
    )
    assert main(["recovery", "lanes", "run", "broken", "--dry-run"]) == 2
    assert "malformed declaration" in capsys.readouterr().err


def test_recovery_lanes_non_dry_run_requires_injected_adapter(
    monkeypatch, capsys
) -> None:
    lane = get_lane("recovery-review-audit")
    # Isolate the adapter boundary from live prerequisite files. Later commits
    # intentionally make older verification/review identities stale, and that
    # fail-closed behavior is covered by the prerequisite tests above.
    monkeypatch.setattr(
        recovery_cli,
        "_available_lane_prerequisites",
        lambda _lane, _root: lane.prerequisites,
    )
    rc = main(["recovery", "lanes", "run", "recovery-review-audit"])
    assert rc != 0
    assert "injected companion adapter" in capsys.readouterr().err


def _handoff_observation() -> CampaignObservation:
    config = load_campaign_config(RECIPE)
    transition = config.transition("full75-rematerialize")
    return CampaignObservation(
        lock_held=True,
        lock_owner_known=True,
        active_processes=(
            ProcessObservation(
                pid=36724,
                command=transition.argv,
                elapsed_seconds=7200.0,
                experiment_name="full75-e8",
                transition_name=transition.name,
            ),
        ),
        recovered_groups=144,
        expected_groups=225,
        manifest_sha256=None,
        contradictions=(),
        campaign=config.campaign,
        experiment_name="full75-e8",
        lock_state="held",
        artifact_links=0,
        expected_artifact_links=225,
        manifest_state="absent",
        manifest_file_sha256=None,
        manifest_body_sha256=None,
        throughput_groups_per_second=0.01,
        eta_seconds=8100.0,
        recovered_groups_state="present",
        artifact_links_state="absent",
    )


def _patch_handoff_live_observers(monkeypatch) -> None:
    monkeypatch.setattr(recovery_cli, "observe_campaign", lambda *_args, **_kwargs: _handoff_observation())
    monkeypatch.setattr(
        recovery_cli,
        "observe_repository_identity",
        lambda _root: recovery_render.RepositoryIdentity(
            branch="branch", head="a" * 40
        ),
    )


def test_recovery_campaign_handoff_help_and_parser_are_registered(capsys) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "recovery", "campaign", "handoff", "--config", str(RECIPE),
            "--ledger", "ledger.jsonl", "--out", "artifacts/quality/out.md",
            "--ledger-head-sha256", "a" * 64,
            "--ledger-event-count", "3",
            "--json",
        ]
    )
    assert args.ledger == Path("ledger.jsonl")
    assert args.out == Path("artifacts/quality/out.md")
    assert args.ledger_head_sha256 == "a" * 64
    assert args.ledger_event_count == 3
    assert args.json is True
    with pytest.raises(SystemExit) as raised:
        parser.parse_args(["recovery", "campaign", "handoff", "--help"])
    assert raised.value.code == 0
    output = capsys.readouterr().out
    assert "--ledger" in output
    assert "--ledger-head-sha256" in output
    assert "--ledger-event-count" in output
    assert "--out" in output
    assert "--json" in output


def test_recovery_campaign_handoff_json_and_publication_receipt(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _patch_handoff_live_observers(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "artifacts/quality").mkdir(parents=True)
    argv = [
        "recovery", "campaign", "handoff", "--config", str(RECIPE), "--json"
    ]
    assert main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["evidence"]["ledger"]["state"] == "absent"
    assert payload["current_blocker"]["state"] == "in-flight"
    assert payload["approvals"]["human_release"]["state"] == "required"

    output = tmp_path / "artifacts/quality/handoff.md"
    assert main([*argv, "--out", str(output)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["path"] == str(output)
    assert receipt["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert output.read_text().startswith("# GLM-5.2 Recovery Campaign Handoff")


def test_recovery_campaign_handoff_error_and_contradiction_exits(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _patch_handoff_live_observers(monkeypatch)
    monkeypatch.chdir(tmp_path)
    quality = tmp_path / "artifacts/quality"
    quality.mkdir(parents=True)
    ledger = quality / "bad.jsonl"
    ledger.write_text("not-json\n")
    base = [
        "recovery", "campaign", "handoff", "--config", str(RECIPE),
        "--ledger", str(ledger), "--json",
    ]
    assert main([*base, "--ledger-head-sha256", "a" * 64]) == 2
    assert "supplied together" in capsys.readouterr().err
    assert main(base) == 2
    assert "error:" in capsys.readouterr().err

    ledger.unlink()
    monkeypatch.setattr(
        recovery_cli,
        "observe_campaign",
        lambda *_args, **_kwargs: replace(
            _handoff_observation(), contradictions=("conflicting evidence",)
        ),
    )
    assert main(base) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["contradictions"] == ["conflicting evidence"]

    monkeypatch.setattr(
        recovery_cli,
        "observe_repository_identity",
        lambda _root: (_ for _ in ()).throw(
            recovery_render.CampaignHandoffError("git unavailable")
        ),
    )
    assert main(base) == 2
    assert "git unavailable" in capsys.readouterr().err


def test_recovery_campaign_handoff_rejects_clean_tail_rollback(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _patch_handoff_live_observers(monkeypatch)
    monkeypatch.chdir(tmp_path)
    ledger = tmp_path / "campaign.jsonl"
    events = []
    for index in range(3):
        events.append(
            append_event(
                ledger,
                event_kind=f"test_event_{index}",
                experiment="full75-e8",
                payload={"index": index},
                timestamp=f"2026-07-11T12:00:0{index}Z",
            )
        )
    lines = ledger.read_bytes().splitlines(keepends=True)
    ledger.write_bytes(b"".join(lines[:-1]))
    rc = main(
        [
            "recovery", "campaign", "handoff", "--config", str(RECIPE),
            "--ledger", str(ledger),
            "--ledger-head-sha256", events[-1].event_sha256,
            "--ledger-event-count", "3",
            "--json",
        ]
    )
    assert rc == 2
    assert "mismatch" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    (
        ["verify", "glm52-recovery-campaign"],
        ["recovery", "campaign", "verify", "glm52-recovery-campaign"],
    ),
)
def test_verification_readiness_failure_exits_one(
    argv: list[str], monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        "mlx_vq.recovery_campaign.cli.run_verification_profile",
        lambda *_args, **_kwargs: _verification_result(ready=False),
    )

    assert main(argv) == 1
    output = capsys.readouterr().out
    assert "Commit ready: no" in output
    assert "review record absent" in output
