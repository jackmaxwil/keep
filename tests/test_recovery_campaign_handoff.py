from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from mlx_vq.io import authenticated_artifacts
from mlx_vq.recovery_campaign.config import load_campaign_config
from mlx_vq.recovery_campaign.ledger import (
    LedgerError,
    append_evidence,
    append_event,
    load_ledger,
)
from mlx_vq.recovery_campaign.models import (
    CampaignObservation,
    EvidenceRecord,
    ExperimentObservation,
    FrozenJsonDict,
    ProcessObservation,
    VerificationResult,
)
from mlx_vq.recovery_campaign.render import (
    CampaignHandoff,
    CampaignHandoffError,
    HandoffPublication,
    RepositoryIdentity,
    build_campaign_handoff as _build_campaign_handoff,
    handoff_fingerprint,
    handoff_payload,
    observe_repository_identity,
    publish_handoff,
    render_handoff_json,
    render_handoff_markdown,
    render_handoff_terminal,
    render_publication_receipt,
)
from mlx_vq.recovery_campaign.verification import (
    get_verification_profile,
    profile_fingerprint,
)


RECIPE = Path(__file__).parents[1] / "recipes/glm52_recovery_campaign_v1_20260711.yaml"
NOW = "2026-07-11T12:00:00Z"


def _config():
    return load_campaign_config(RECIPE)


def _repository() -> RepositoryIdentity:
    return RepositoryIdentity(
        branch="keep-glm52-pipeline-and-p1-lock",
        head="a" * 40,
    )


def build_campaign_handoff(config, ledger, observation, repository, **anchors):
    events = tuple(ledger)
    if events and not anchors:
        anchors = {
            "ledger_head_sha256": events[-1].event_sha256,
            "ledger_event_count": len(events),
        }
    return _build_campaign_handoff(
        config, events, observation, repository, **anchors
    )


def _quality_row_sha256(
    config,
    *,
    experiment: str,
    transition: str,
    split: str,
    metric: str,
    value: float,
    manifest_sha256: str,
) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "campaign_config_sha256": config.campaign_config_sha256,
                "experiment": experiment,
                "transition": transition,
                "split": split,
                "metric": metric,
                "value": value,
                "manifest_sha256": manifest_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _observation(*, active: bool = True, contradictions=()) -> CampaignObservation:
    config = _config()
    transition = config.transition("full75-rematerialize")
    processes = (
        (
            ProcessObservation(
                pid=36724,
                command=transition.argv,
                elapsed_seconds=7200.0,
                experiment_name="full75-e8",
                transition_name=transition.name,
            ),
        )
        if active
        else ()
    )
    return CampaignObservation(
        lock_held=active,
        lock_owner_known=active,
        active_processes=processes,
        recovered_groups=144,
        expected_groups=225,
        manifest_sha256=None,
        contradictions=tuple(contradictions),
        campaign=config.campaign,
        experiment_name="full75-e8",
        lock_state="held" if active else "free",
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


def _completed_observation() -> CampaignObservation:
    return replace(
        _observation(active=False),
        recovered_groups=225,
        artifact_links=225,
        artifact_links_state="present",
        manifest_sha256="f" * 64,
        manifest_state="valid",
        manifest_file_sha256="f" * 64,
        manifest_body_sha256="9" * 64,
        throughput_groups_per_second=None,
        eta_seconds=0.0,
    )


def _evidence(*, diagnostic: bool = False) -> EvidenceRecord:
    argv = ("pytest", "-q")
    return EvidenceRecord(
        artifact_path="artifacts/quality/quality-row.json",
        content_sha256="c" * 64,
        evidence_class="diagnostic_only" if diagnostic else "release",
        release_eligible=not diagnostic,
        recovery_levers=("quality-observation",),
        argv=argv,
        verification_results=(
            VerificationResult(
                name="quality-row-verification",
                argv=argv,
                exit_code=0,
                stdout_sha256="d" * 64,
                stderr_sha256="e" * 64,
            ),
        ),
        manifest_identity_sha256="f" * 64,
        candidate_identity_sha256="1" * 64 if not diagnostic else None,
        baseline_identity_sha256="2" * 64 if not diagnostic else None,
    )


def _approval_evidence(kind: str) -> EvidenceRecord:
    config = _config()
    profile = get_verification_profile("glm52-recovery-campaign")
    profile_sha = profile_fingerprint(profile)
    verification = kind == "verification"
    argv = ("pytest", "-q")
    return EvidenceRecord(
        artifact_path=f"artifacts/quality/{kind}-approval.json",
        content_sha256=("7" if verification else "8") * 64,
        evidence_class="release",
        release_eligible=True,
        recovery_levers=(
            (
                f"verification-profile:{profile.name}"
                if verification
                else f"reviewer:{profile.reviewer_authority}"
            ),
        ),
        argv=argv,
        verification_results=(
            VerificationResult(
                name=f"{kind}-approval",
                argv=argv,
                exit_code=0,
                stdout_sha256=("4" if verification else "5") * 64,
                stderr_sha256="6" * 64,
            ),
        ),
        manifest_identity_sha256=config.campaign_config_sha256,
        candidate_identity_sha256=profile_sha,
        baseline_identity_sha256=("3" if verification else "7") * 64,
    )


def _quality_ledger(
    tmp_path: Path,
    *,
    values: tuple[float, ...] = (0.81,),
    splits: tuple[str, ...] = ("selection",),
    diagnostic: bool = False,
    transition: str = "full75-rematerialize",
    reuse_evidence: bool = False,
):
    config = _config()
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "campaign.jsonl"
    assert len(values) == len(splits)
    retained_evidence = None
    for index, (value, split) in enumerate(zip(values, splits), start=1):
        row_sha256 = _quality_row_sha256(
            config,
            experiment="full75-e8",
            transition=transition,
            split=split,
            metric="top1_agreement",
            value=value,
            manifest_sha256="f" * 64,
        )
        if retained_evidence is None or not reuse_evidence:
            evidence_event = append_evidence(
                path,
                experiment="full75-e8",
                evidence=replace(
                    _evidence(diagnostic=diagnostic),
                    content_sha256=f"{index:x}" * 64,
                    recovery_levers=(
                        "quality-observation",
                        f"quality-row-sha256:{row_sha256}",
                    ),
                ),
                timestamp=f"2026-07-11T12:00:{index * 2 - 2:02d}Z",
            )
            retained_evidence = evidence_event
        else:
            evidence_event = retained_evidence
        append_event(
            path,
            event_kind="quality_observed",
            experiment="full75-e8",
            payload={
                "campaign_config_sha256": config.campaign_config_sha256,
                "transition": transition,
                "split": split,
                "metric": "top1_agreement",
                "value": value,
                "evidence_event_sha256": evidence_event.event_sha256,
                "manifest_sha256": "f" * 64,
                "quality_row_sha256": row_sha256,
            },
            timestamp=f"2026-07-11T12:00:{index * 2 - 1:02d}Z",
        )
    return load_ledger(path)


def test_handoff_projection_is_immutable_complete_and_fingerprinted() -> None:
    config = _config()
    handoff = build_campaign_handoff(config, (), _observation(), _repository())
    payload = handoff_payload(handoff)

    assert payload["campaign"] == config.campaign
    assert payload["model"]["id"] == config.model_id
    assert payload["model"]["revision"] == config.model_revision
    assert payload["repository"] == {
        "branch": "keep-glm52-pipeline-and-p1-lock",
        "head": "a" * 40,
    }
    assert payload["active_jobs"][0]["pid"] == 36724
    assert payload["active_jobs"][0]["elapsed_seconds"] == 7200.0
    assert payload["active_jobs"][0]["transition"] == "full75-rematerialize"
    assert payload["active_jobs"][0]["command"] == list(
        config.transition("full75-rematerialize").argv
    )
    assert payload["groups"] == {"observed": 144, "expected": 225, "state": "present"}
    assert payload["throughput_groups_per_second"] == 0.01
    assert payload["eta_seconds"] == 8100.0
    assert payload["evidence"]["campaign_matrix_sha256"] == config.campaign_config_sha256
    assert payload["evidence"]["authorities"] == [
        {
            "name": authority.name,
            "path": authority.path,
            "sha256": authority.sha256,
            "role": authority.role,
            "split": authority.split,
        }
        for authority in config.authorities
    ]
    assert payload["experiments"] == [
        {
            "name": experiment.name,
            "kind": experiment.kind,
            "transition": experiment.transition_name,
            "depends_on": list(experiment.depends_on),
            "output_path": experiment.output_path,
            "expected_payload_bytes": experiment.expected_payload_bytes,
        }
        for experiment in config.experiments
    ]
    assert payload["evidence"]["manifest"]["state"] == "absent"
    assert payload["evidence"]["ledger"] == {
        "state": "absent",
        "event_count": 0,
        "head_sha256": None,
        "chain_sha256": None,
        "anchor_head_sha256": None,
        "anchor_event_count": None,
    }
    assert payload["quality_curve"] == {
        "state": "absent",
        "rows": [],
    }
    assert payload["current_blocker"]["state"] == "in-flight"
    assert payload["next_action"]["action"] == "wait"
    assert payload["next_action"]["command"] == list(
        config.transition("full75-rematerialize").argv
    )
    assert payload["approvals"]["verification"]["state"] == "absent"
    assert payload["approvals"]["review"]["state"] == "absent"
    assert payload["approvals"]["human_release"]["state"] == "required"
    assert len(handoff_fingerprint(handoff)) == 64
    assert payload["handoff_sha256"] == handoff_fingerprint(handoff)
    with pytest.raises(FrozenInstanceError):
        handoff.fingerprint = "b" * 64  # type: ignore[misc]


def test_protected_paths_include_lock_runs_handoffs_and_declarations() -> None:
    config = _config()
    profile = get_verification_profile("glm52-recovery-campaign")
    payload = handoff_payload(
        build_campaign_handoff(config, (), _observation(), _repository())
    )
    protected = set(payload["protected_paths"])
    assert {
        ".keep-heavy-job.lock",
        "runs",
        "glm52-community-wow-section-handoff-20260710.md",
        "glm52-recovery-campaign-handoff-20260710.md",
        config.campaign_config_path,
        profile.required_review_record,
    } <= protected
    assert payload["evidence"]["verification_profile_sha256"] == profile_fingerprint(profile)


def test_terminal_json_and_markdown_render_the_same_canonical_facts() -> None:
    handoff = build_campaign_handoff(_config(), (), _observation(), _repository())
    payload = handoff_payload(handoff)
    encoded = render_handoff_json(handoff)
    assert json.loads(encoded) == payload
    terminal = render_handoff_terminal(handoff)
    markdown = render_handoff_markdown(handoff)
    for rendered in (terminal, markdown):
        assert payload["campaign"] in rendered
        assert payload["repository"]["branch"] in rendered
        assert payload["handoff_sha256"] in rendered
        assert "144/225" in rendered
        assert "absent" in rendered
        assert "wait" in rendered.lower()
        assert "human" in rendered.lower()
    assert markdown.startswith("# GLM-5.2 Recovery Campaign Handoff\n")
    canonical_facts = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    assert canonical_facts in terminal
    assert canonical_facts in markdown
    for representative in (
        payload["experiments"][0]["output_path"],
        payload["active_jobs"][0]["command"][0],
        str(payload["artifact_links"]["expected"]),
        payload["evidence"]["authorities"][0]["path"],
        payload["evidence"]["full_source_inventory_sha256"],
        str(payload["evidence"]["ledger"]["event_count"]),
        payload["approvals"]["human_release"]["boundaries"][0],
        payload["release_status"],
    ):
        assert representative in terminal
        assert representative in markdown
    assert encoded == render_handoff_json(handoff)
    assert markdown == render_handoff_markdown(handoff)


def test_quality_rows_require_exact_release_evidence_and_conflicts_are_absent(
    tmp_path: Path,
) -> None:
    accepted = build_campaign_handoff(
        _config(), _quality_ledger(tmp_path), _completed_observation(), _repository()
    )
    quality = handoff_payload(accepted)["quality_curve"]
    assert quality["state"] == "present"
    assert quality["rows"][0] == {
        "experiment": "full75-e8",
        "transition": "full75-rematerialize",
        "split": "selection",
        "metric": "top1_agreement",
        "value": 0.81,
        "evidence_sha256": "1" * 64,
        "manifest_sha256": "f" * 64,
    }
    ledger_identity = handoff_payload(accepted)["evidence"]["ledger"]
    ledger = _quality_ledger(tmp_path / "identity")
    assert ledger_identity["state"] == "anchored"
    assert ledger_identity["event_count"] == 2
    assert ledger_identity["head_sha256"] is not None
    assert ledger_identity["chain_sha256"] is not None
    assert len(ledger) == 2

    diagnostic = build_campaign_handoff(
        _config(),
        _quality_ledger(tmp_path / "diagnostic", diagnostic=True),
        _completed_observation(),
        _repository(),
    )
    assert handoff_payload(diagnostic)["quality_curve"] == {
        "state": "absent",
        "rows": [],
    }

    stale = build_campaign_handoff(
        _config(),
        _quality_ledger(tmp_path / "stale", transition="wrong"),
        _completed_observation(),
        _repository(),
    )
    assert handoff_payload(stale)["quality_curve"]["state"] == "absent"

    conflicting = build_campaign_handoff(
        _config(),
        _quality_ledger(
            tmp_path / "conflict",
            values=(0.81, 0.82),
            splits=("selection", "selection"),
        ),
        _completed_observation(),
        _repository(),
    )
    conflict_payload = handoff_payload(conflicting)
    assert conflict_payload["quality_curve"]["state"] == "contradictory"
    assert conflict_payload["quality_curve"]["rows"] == []
    assert any("quality" in item for item in conflict_payload["contradictions"])


def test_quality_splits_are_distinct_and_inflight_rows_stay_absent(
    tmp_path: Path,
) -> None:
    ledger = _quality_ledger(
        tmp_path,
        values=(0.81, 0.82, 0.83),
        splits=("selection", "report", "holdout"),
    )
    completed = handoff_payload(
        build_campaign_handoff(
            _config(), ledger, _completed_observation(), _repository()
        )
    )
    rows = completed["quality_curve"]["rows"]
    assert [row["split"] for row in rows] == ["holdout", "report", "selection"]
    assert completed["quality_curve"]["state"] == "present"

    inflight = handoff_payload(
        build_campaign_handoff(_config(), ledger, _observation(), _repository())
    )
    assert inflight["quality_curve"] == {"state": "absent", "rows": []}


def test_quality_requires_exact_purpose_and_single_use_evidence(
    tmp_path: Path,
) -> None:
    config = _config()
    unrelated_path = tmp_path / "unrelated.jsonl"
    unrelated = append_evidence(
        unrelated_path,
        experiment="full75-e8",
        evidence=replace(_evidence(), recovery_levers=("unrelated-release",)),
        timestamp=NOW,
    )
    unrelated_row_sha256 = _quality_row_sha256(
        config,
        experiment="full75-e8",
        transition="full75-rematerialize",
        split="selection",
        metric="top1_agreement",
        value=0.81,
        manifest_sha256="f" * 64,
    )
    append_event(
        unrelated_path,
        event_kind="quality_observed",
        experiment="full75-e8",
        payload={
            "campaign_config_sha256": config.campaign_config_sha256,
            "transition": "full75-rematerialize",
            "split": "selection",
            "metric": "top1_agreement",
            "value": 0.81,
            "evidence_event_sha256": unrelated.event_sha256,
            "manifest_sha256": "f" * 64,
            "quality_row_sha256": unrelated_row_sha256,
        },
        timestamp="2026-07-11T12:00:01Z",
    )
    unrelated_events = load_ledger(unrelated_path)
    unrelated_handoff = build_campaign_handoff(
        config,
        unrelated_events,
        _completed_observation(),
        _repository(),
        ledger_head_sha256=unrelated_events[-1].event_sha256,
        ledger_event_count=len(unrelated_events),
    )
    assert handoff_payload(unrelated_handoff)["quality_curve"]["rows"] == []

    argv_path = tmp_path / "mismatched-argv.jsonl"
    exact_evidence = _evidence()
    mismatched_result = replace(
        exact_evidence.verification_results[0], argv=("pytest", "-x")
    )
    mismatched = append_evidence(
        argv_path,
        experiment="full75-e8",
        evidence=replace(
            exact_evidence,
            recovery_levers=(
                "quality-observation",
                f"quality-row-sha256:{unrelated_row_sha256}",
            ),
            verification_results=(mismatched_result,),
        ),
        timestamp=NOW,
    )
    append_event(
        argv_path,
        event_kind="quality_observed",
        experiment="full75-e8",
        payload={
            "campaign_config_sha256": config.campaign_config_sha256,
            "transition": "full75-rematerialize",
            "split": "selection",
            "metric": "top1_agreement",
            "value": 0.81,
            "evidence_event_sha256": mismatched.event_sha256,
            "manifest_sha256": "f" * 64,
            "quality_row_sha256": unrelated_row_sha256,
        },
        timestamp="2026-07-11T12:00:01Z",
    )
    argv_events = load_ledger(argv_path)
    argv_handoff = build_campaign_handoff(
        config,
        argv_events,
        _completed_observation(),
        _repository(),
        ledger_head_sha256=argv_events[-1].event_sha256,
        ledger_event_count=len(argv_events),
    )
    assert handoff_payload(argv_handoff)["quality_curve"]["rows"] == []

    reused_events = _quality_ledger(
        tmp_path / "reused",
        values=(0.81, 0.82),
        splits=("selection", "report"),
        reuse_evidence=True,
    )
    reused_handoff = build_campaign_handoff(
        config,
        reused_events,
        _completed_observation(),
        _repository(),
        ledger_head_sha256=reused_events[-1].event_sha256,
        ledger_event_count=len(reused_events),
    )
    reused_payload = handoff_payload(reused_handoff)
    assert reused_payload["quality_curve"] == {
        "state": "contradictory",
        "rows": [],
    }
    assert any("reused" in item for item in reused_payload["contradictions"])


def test_nonempty_ledger_requires_exact_external_anchor(tmp_path: Path) -> None:
    events = _quality_ledger(tmp_path)
    with pytest.raises(CampaignHandoffError, match="anchor"):
        _build_campaign_handoff(
            _config(), events, _completed_observation(), _repository()
        )
    anchored = build_campaign_handoff(
        _config(),
        events,
        _completed_observation(),
        _repository(),
        ledger_head_sha256=events[-1].event_sha256,
        ledger_event_count=len(events),
    )
    assert handoff_payload(anchored)["evidence"]["ledger"]["state"] == "anchored"


def test_external_anchor_rejects_clean_tail_rollback(tmp_path: Path) -> None:
    events = _quality_ledger(
        tmp_path,
        values=(0.81, 0.82),
        splits=("selection", "selection"),
        reuse_evidence=True,
    )
    assert len(events) == 3
    original_head = events[-1].event_sha256
    original_count = len(events)
    path = tmp_path / "campaign.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[:-1]))

    with pytest.raises(LedgerError, match="head|count"):
        load_ledger(
            path,
            expected_head_sha256=original_head,
            expected_event_count=original_count,
        )
    prefix = load_ledger(path)
    assert len(prefix) == 2
    with pytest.raises(CampaignHandoffError, match="anchor"):
        _build_campaign_handoff(
            _config(), prefix, _completed_observation(), _repository()
        )
def test_completed_experiment_quality_survives_later_current_experiment(
    tmp_path: Path,
) -> None:
    config = _config()
    snapshots = []
    for experiment in config.experiments:
        expected = len(experiment.recovered_layers) * 3
        if experiment.name == "full75-e8":
            snapshots.append(
                ExperimentObservation(
                    experiment_name=experiment.name,
                    recovered_groups=expected,
                    expected_groups=expected,
                    recovered_groups_state="present",
                    artifact_links=225,
                    expected_artifact_links=225,
                    artifact_links_state="present",
                    manifest_state="valid",
                    manifest_file_sha256="f" * 64,
                    manifest_body_sha256="9" * 64,
                    contradictions=(),
                )
            )
        else:
            snapshots.append(
                ExperimentObservation(
                    experiment_name=experiment.name,
                    recovered_groups=0,
                    expected_groups=expected,
                    recovered_groups_state="absent",
                    artifact_links=0,
                    expected_artifact_links=225,
                    artifact_links_state="absent",
                    manifest_state="absent",
                    manifest_file_sha256=None,
                    manifest_body_sha256=None,
                    contradictions=(),
                )
            )
    later = replace(
        _observation(),
        experiment_name="worst8-e8p",
        active_processes=(
            ProcessObservation(
                pid=36724,
                command=config.transition("worst8-e8p-rematerialize").argv,
                elapsed_seconds=10.0,
                experiment_name="worst8-e8p",
                transition_name="worst8-e8p-rematerialize",
            ),
        ),
        recovered_groups=0,
        expected_groups=24,
        recovered_groups_state="absent",
        experiment_observations=tuple(snapshots),
    )
    payload = handoff_payload(
        build_campaign_handoff(
            config, _quality_ledger(tmp_path), later, _repository()
        )
    )
    assert payload["quality_curve"]["state"] == "present"
    assert payload["quality_curve"]["rows"][0]["experiment"] == "full75-e8"


def test_partial_quality_payload_is_contradictory_not_partially_trusted(
    tmp_path: Path,
) -> None:
    path = tmp_path / "campaign.jsonl"
    append_event(
        path,
        event_kind="quality_observed",
        experiment="full75-e8",
        payload={"metric": "top1_agreement", "value": 0.81},
        timestamp=NOW,
    )
    payload = handoff_payload(
        build_campaign_handoff(
            _config(), load_ledger(path), _completed_observation(), _repository()
        )
    )
    assert payload["quality_curve"] == {"state": "contradictory", "rows": []}
    assert any("quality" in item for item in payload["contradictions"])


def test_nonfinite_observation_values_are_absent_and_contradictory() -> None:
    observation = replace(
        _observation(),
        throughput_groups_per_second=math.nan,
        eta_seconds=math.inf,
    )
    payload = handoff_payload(
        build_campaign_handoff(_config(), (), observation, _repository())
    )
    assert payload["throughput_groups_per_second"] is None
    assert payload["eta_seconds"] is None
    assert any("non-finite" in item for item in payload["contradictions"])
    assert "NaN" not in render_handoff_json(
        build_campaign_handoff(_config(), (), observation, _repository())
    )


def test_ledger_approval_events_cannot_establish_independent_approval(
    tmp_path: Path,
) -> None:
    config = _config()
    profile = get_verification_profile("glm52-recovery-campaign")
    profile_sha = profile_fingerprint(profile)
    path = tmp_path / "campaign.jsonl"
    verification_evidence = append_evidence(
        path,
        experiment=config.campaign,
        evidence=_approval_evidence("verification"),
        timestamp=NOW,
    )
    append_event(
        path,
        event_kind="verification_approved",
        experiment=config.campaign,
        payload={
            "campaign_config_sha256": config.campaign_config_sha256,
            "profile": profile.name,
            "profile_sha256": profile_sha,
            "evidence_event_sha256": verification_evidence.event_sha256,
        },
        timestamp=NOW,
    )
    review_evidence = append_evidence(
        path,
        experiment=config.campaign,
        evidence=_approval_evidence("review"),
        timestamp="2026-07-11T12:00:02Z",
    )
    append_event(
        path,
        event_kind="review_approved",
        experiment=config.campaign,
        payload={
            "campaign_config_sha256": config.campaign_config_sha256,
            "profile_sha256": profile_sha,
            "review_sha256": "8" * 64,
            "reviewer_authority": profile.reviewer_authority,
            "decision": "approved",
            "evidence_event_sha256": review_evidence.event_sha256,
        },
        timestamp="2026-07-11T12:00:03Z",
    )
    payload = handoff_payload(
        build_campaign_handoff(
            config, load_ledger(path), _observation(), _repository()
        )
    )
    approvals = payload["approvals"]
    assert approvals["verification"]["state"] == "absent"
    assert approvals["verification"]["evidence_sha256"] is None
    assert approvals["review"]["state"] == "absent"
    assert approvals["review"]["evidence_sha256"] is None
    assert approvals["human_release"]["state"] == "required"
    assert any("independent" in item for item in payload["contradictions"])


def test_self_asserted_approval_events_cannot_approve_without_prior_evidence(
    tmp_path: Path,
) -> None:
    config = _config()
    profile = get_verification_profile("glm52-recovery-campaign")
    path = tmp_path / "campaign.jsonl"
    append_event(
        path,
        event_kind="verification_approved",
        experiment=config.campaign,
        payload={
            "campaign_config_sha256": config.campaign_config_sha256,
            "profile": profile.name,
            "profile_sha256": profile_fingerprint(profile),
            "evidence_sha256": "7" * 64,
        },
        timestamp=NOW,
    )
    payload = handoff_payload(
        build_campaign_handoff(
            config, load_ledger(path), _observation(), _repository()
        )
    )
    assert payload["approvals"]["verification"]["state"] == "absent"
    assert any("approval" in item for item in payload["contradictions"])


def test_conflicting_evidence_bound_approvals_are_never_positive(
    tmp_path: Path,
) -> None:
    config = _config()
    profile = get_verification_profile("glm52-recovery-campaign")
    profile_sha = profile_fingerprint(profile)
    path = tmp_path / "campaign.jsonl"
    for index, content in enumerate(("7" * 64, "9" * 64)):
        evidence_event = append_evidence(
            path,
            experiment=config.campaign,
            evidence=replace(
                _approval_evidence("verification"),
                content_sha256=content,
            ),
            timestamp=f"2026-07-11T12:00:0{index * 2}Z",
        )
        append_event(
            path,
            event_kind="verification_approved",
            experiment=config.campaign,
            payload={
                "campaign_config_sha256": config.campaign_config_sha256,
                "profile": profile.name,
                "profile_sha256": profile_sha,
                "evidence_event_sha256": evidence_event.event_sha256,
            },
            timestamp=f"2026-07-11T12:00:0{index * 2 + 1}Z",
        )
    payload = handoff_payload(
        build_campaign_handoff(
            config, load_ledger(path), _observation(), _repository()
        )
    )
    assert payload["approvals"]["verification"]["state"] == "absent"
    assert payload["approvals"]["verification"]["evidence_sha256"] is None
    assert any("independent" in item for item in payload["contradictions"])


def test_malformed_plus_valid_approval_is_never_positive(tmp_path: Path) -> None:
    config = _config()
    profile = get_verification_profile("glm52-recovery-campaign")
    profile_sha = profile_fingerprint(profile)
    path = tmp_path / "campaign.jsonl"
    append_event(
        path,
        event_kind="verification_approved",
        experiment=config.campaign,
        payload={
            "campaign_config_sha256": config.campaign_config_sha256,
            "profile": profile.name,
            "profile_sha256": profile_sha,
            "evidence_sha256": "7" * 64,
        },
        timestamp=NOW,
    )
    evidence_event = append_evidence(
        path,
        experiment=config.campaign,
        evidence=_approval_evidence("verification"),
        timestamp="2026-07-11T12:00:01Z",
    )
    append_event(
        path,
        event_kind="verification_approved",
        experiment=config.campaign,
        payload={
            "campaign_config_sha256": config.campaign_config_sha256,
            "profile": profile.name,
            "profile_sha256": profile_sha,
            "evidence_event_sha256": evidence_event.event_sha256,
        },
        timestamp="2026-07-11T12:00:02Z",
    )
    payload = handoff_payload(
        build_campaign_handoff(
            config, load_ledger(path), _observation(), _repository()
        )
    )
    assert payload["approvals"]["verification"]["state"] == "absent"
    assert payload["approvals"]["verification"]["evidence_sha256"] is None
    assert any("independent" in item for item in payload["contradictions"])


def test_any_approval_contradiction_invalidates_verification_and_review(
    tmp_path: Path,
) -> None:
    config = _config()
    profile = get_verification_profile("glm52-recovery-campaign")
    profile_sha = profile_fingerprint(profile)
    path = tmp_path / "campaign.jsonl"
    _quality_ledger(tmp_path)
    verification = append_evidence(
        path,
        experiment=config.campaign,
        evidence=_approval_evidence("verification"),
        timestamp=NOW,
    )
    append_event(
        path,
        event_kind="verification_approved",
        experiment=config.campaign,
        payload={
            "campaign_config_sha256": config.campaign_config_sha256,
            "profile": profile.name,
            "profile_sha256": profile_sha,
            "evidence_event_sha256": verification.event_sha256,
        },
        timestamp="2026-07-11T12:00:01Z",
    )
    review = append_evidence(
        path,
        experiment=config.campaign,
        evidence=_approval_evidence("review"),
        timestamp="2026-07-11T12:00:02Z",
    )
    append_event(
        path,
        event_kind="review_approved",
        experiment=config.campaign,
        payload={
            "campaign_config_sha256": config.campaign_config_sha256,
            "profile_sha256": profile_sha,
            "review_sha256": "8" * 64,
            "reviewer_authority": profile.reviewer_authority,
            "decision": "approved",
            "evidence_event_sha256": review.event_sha256,
        },
        timestamp="2026-07-11T12:00:03Z",
    )
    append_event(
        path,
        event_kind="review_approved",
        experiment=config.campaign,
        payload={"decision": "approved"},
        timestamp="2026-07-11T12:00:04Z",
    )
    events = load_ledger(path)
    payload = handoff_payload(
        build_campaign_handoff(
            config,
            events,
            _completed_observation(),
            _repository(),
            ledger_head_sha256=events[-1].event_sha256,
            ledger_event_count=len(events),
        )
    )
    assert payload["approvals"]["verification"]["state"] == "absent"
    assert payload["approvals"]["review"]["state"] == "absent"
    assert payload["quality_curve"] == {"state": "contradictory", "rows": []}
    assert any("approval" in item for item in payload["contradictions"])


def test_handoff_values_require_builder_authority_and_campaign_match() -> None:
    forged_facts = FrozenJsonDict(
        {"release_status": "human-approved", "contradictions": []}
    )
    forged = CampaignHandoff(
        facts=forged_facts,
        fingerprint=hashlib.sha256(
            json.dumps(
                {
                    "contradictions": [],
                    "release_status": "human-approved",
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    )
    with pytest.raises(CampaignHandoffError, match="builder|authorit"):
        handoff_payload(forged)
    valid = build_campaign_handoff(_config(), (), _observation(), _repository())
    with pytest.raises(CampaignHandoffError, match="builder|authorit"):
        handoff_payload(replace(valid, facts=forged_facts, fingerprint=forged.fingerprint))
    with pytest.raises(CampaignHandoffError, match="campaign"):
        build_campaign_handoff(
            _config(),
            (),
            replace(_observation(), campaign="different"),
            _repository(),
        )


def test_repository_identity_is_strict_and_live_git_is_direct() -> None:
    with pytest.raises(CampaignHandoffError):
        RepositoryIdentity(branch="bad\nbranch", head="bad")
    calls: list[tuple[str, ...]] = []

    def runner(argv, **kwargs):
        calls.append(tuple(argv))
        output = b"branch\n" if "symbolic-ref" in argv else (b"a" * 40 + b"\n")
        return subprocess.CompletedProcess(argv, 0, stdout=output, stderr=b"")

    identity = observe_repository_identity(Path("/repo"), runner=runner)
    assert identity == RepositoryIdentity(branch="branch", head="a" * 40)
    assert len(calls) == 2
    assert all(call[0] == "/usr/bin/git" for call in calls)


@pytest.mark.parametrize("failure", ("nonzero", "timeout", "malformed"))
def test_repository_identity_fails_loud_on_git_errors(failure: str) -> None:
    def runner(argv, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        if failure == "nonzero":
            return subprocess.CompletedProcess(argv, 1, stdout=b"", stderr=b"bad")
        return subprocess.CompletedProcess(argv, 0, stdout=b"not valid\n", stderr=b"")

    with pytest.raises(CampaignHandoffError, match="Git|branch|HEAD"):
        observe_repository_identity(Path("/repo"), runner=runner)


def test_atomic_publication_and_receipt_are_authenticated(tmp_path: Path) -> None:
    output = tmp_path / "artifacts/quality/handoff.md"
    output.parent.mkdir(parents=True)
    markdown = render_handoff_markdown(
        build_campaign_handoff(_config(), (), _observation(), _repository())
    )
    receipt = publish_handoff(output, markdown, repo_root=tmp_path)
    assert output.read_text() == markdown
    assert receipt.sha256 == hashlib.sha256(markdown.encode()).hexdigest()
    assert receipt.size == len(markdown.encode())
    human = render_publication_receipt(receipt, as_json=False)
    machine = json.loads(render_publication_receipt(receipt, as_json=True))
    assert str(output) in human
    assert machine == {
        "path": str(output),
        "sha256": receipt.sha256,
        "size": receipt.size,
    }
    receipt.close()


def test_publication_receipt_rejects_forgery_and_post_publish_replacement(
    tmp_path: Path,
) -> None:
    output = tmp_path / "artifacts/quality/handoff.md"
    output.parent.mkdir(parents=True)
    forged = HandoffPublication(path=output, sha256="a" * 64, size=1)
    with pytest.raises(CampaignHandoffError, match="publisher|authority"):
        render_publication_receipt(forged, as_json=False)

    receipt = publish_handoff(output, "# authenticated\n", repo_root=tmp_path)
    output.write_text("# replacement\n")
    with pytest.raises(CampaignHandoffError, match="changed|authentic"):
        render_publication_receipt(receipt, as_json=False)
    receipt.close()


def test_publication_receipt_context_close_and_fd_lifecycle(
    tmp_path: Path, monkeypatch
) -> None:
    baseline = len(os.listdir("/dev/fd"))
    for index in range(20):
        output = tmp_path / f"repo-{index}/artifacts/quality/handoff.md"
        output.parent.mkdir(parents=True)
        with publish_handoff(
            output, "# authenticated\n", repo_root=tmp_path / f"repo-{index}"
        ) as receipt:
            assert "SHA-256" in render_publication_receipt(receipt, as_json=False)
            assert json.loads(render_publication_receipt(receipt, as_json=True))["size"]
        with pytest.raises(CampaignHandoffError, match="closed"):
            render_publication_receipt(receipt, as_json=False)
    assert len(os.listdir("/dev/fd")) == baseline

    in_place_root = tmp_path / "in-place"
    in_place = in_place_root / "artifacts/quality/handoff.md"
    in_place.parent.mkdir(parents=True)
    receipt = publish_handoff(
        in_place, "# authenticated\n", repo_root=in_place_root
    )
    with in_place.open("r+b") as stream:
        stream.write(b"x" * receipt.size)
    with pytest.raises(CampaignHandoffError, match="changed|authentic"):
        render_publication_receipt(receipt, as_json=False)
    receipt.close()

    ancestor_root = tmp_path / "ancestor"
    ancestor = ancestor_root / "artifacts/quality/handoff.md"
    ancestor.parent.mkdir(parents=True)
    markdown = "# authenticated\n"
    receipt = publish_handoff(ancestor, markdown, repo_root=ancestor_root)
    (ancestor_root / "artifacts").rename(ancestor_root / "old-artifacts")
    ancestor.parent.mkdir(parents=True)
    ancestor.write_text(markdown)
    with pytest.raises(CampaignHandoffError, match="changed|authentic"):
        render_publication_receipt(receipt, as_json=False)
    receipt.close()

    fault_root = tmp_path / "fault"
    fault = fault_root / "artifacts/quality/handoff.md"
    fault.parent.mkdir(parents=True)

    def fail_seal(_receipt):
        raise RuntimeError("injected receipt construction failure")

    monkeypatch.setattr(
        "mlx_vq.recovery_campaign.render._publication_seal", fail_seal
    )
    with pytest.raises(RuntimeError, match="construction failure"):
        publish_handoff(fault, "# authenticated\n", repo_root=fault_root)
    assert len(os.listdir("/dev/fd")) == baseline


def test_publication_rejects_protected_symlink_existing_and_interrupted_targets(
    tmp_path: Path, monkeypatch
) -> None:
    markdown = "# handoff\n"
    quality = tmp_path / "artifacts/quality"
    quality.mkdir(parents=True)
    with pytest.raises(CampaignHandoffError, match="protected"):
        publish_handoff(
            tmp_path / "glm52-recovery-campaign-handoff-20260710.md",
            markdown,
            repo_root=tmp_path,
        )
    with pytest.raises(CampaignHandoffError, match="protected"):
        publish_handoff(
            tmp_path / "glm52-community-wow-section-handoff-20260710.md",
            markdown,
            repo_root=tmp_path,
        )
    with pytest.raises(CampaignHandoffError, match="inside"):
        publish_handoff(
            tmp_path.parent / "outside.md",
            markdown,
            repo_root=tmp_path,
        )
    target = quality / "handoff.md"
    target.symlink_to(quality / "missing")
    with pytest.raises(CampaignHandoffError):
        publish_handoff(target, markdown, repo_root=tmp_path)
    target.unlink()
    target.write_text("old")
    with pytest.raises(CampaignHandoffError):
        publish_handoff(target, markdown, repo_root=tmp_path)
    assert target.read_text() == "old"
    target.unlink()

    def interrupt(*_args, **_kwargs):
        raise OSError("injected interruption")

    monkeypatch.setattr(authenticated_artifacts, "_before_publication_syscall", interrupt)
    with pytest.raises(CampaignHandoffError, match="interruption"):
        publish_handoff(target, markdown, repo_root=tmp_path)
    assert not target.exists()
    assert list(quality.iterdir()) == []

    quality.rmdir()
    (tmp_path / "artifacts").rmdir()
    outside = tmp_path / "outside"
    (outside / "quality").mkdir(parents=True)
    (tmp_path / "artifacts").symlink_to(outside, target_is_directory=True)
    with pytest.raises(CampaignHandoffError):
        publish_handoff(
            tmp_path / "artifacts/quality/handoff.md",
            markdown,
            repo_root=tmp_path,
        )


def test_handoff_fingerprint_changes_with_every_authority_axis() -> None:
    base = build_campaign_handoff(_config(), (), _observation(), _repository())
    changed_repo = build_campaign_handoff(
        _config(),
        (),
        _observation(),
        RepositoryIdentity(branch="other", head="b" * 40),
    )
    changed_observation = build_campaign_handoff(
        _config(),
        (),
        replace(_observation(), recovered_groups=145),
        _repository(),
    )
    assert handoff_fingerprint(base) != handoff_fingerprint(changed_repo)
    assert handoff_fingerprint(base) != handoff_fingerprint(changed_observation)
