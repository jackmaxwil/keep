from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from mlx_vq.recovery_campaign.review import (
    ReviewEvidenceIdentity,
    ReviewRecord,
    ReviewRecordError,
    classify_finding,
    finding_fingerprint,
    load_review_record,
    parse_review_record,
    render_red_test_queue,
    review_fingerprint,
)


PROFILE = "a" * 64
BASE = "1" * 40
HEAD = "2" * 40
REVIEWER = "independent-adversarial-reviewer"


def _payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "task7-review",
        "base_commit": BASE,
        "head_commit": HEAD,
        "reviewed_content_sha256": None,
        "profile_sha256": PROFILE,
        "owned_paths": ["src/mlx_vq/recovery_campaign/review.py"],
        "frozen_decisions": ["human approval remains external"],
        "out_of_scope_paths": ["runs", ".keep-heavy-job.lock"],
        "threat_model": ["same-uid pathname replacement", "stale review authority"],
        "verification_evidence": [
            {"name": "focused-pytest", "sha256": "b" * 64}
        ],
        "resolved_finding_fingerprints": [],
        "known_diagnostic_fingerprints": [],
        "reviewer_authority": REVIEWER,
        "result_status": "changes-required",
    }


def _record(payload: dict[str, object] | None = None):
    return parse_review_record(
        payload or _payload(),
        expected_profile_sha256=PROFILE,
        expected_base=BASE,
        expected_head=HEAD,
        recognized_reviewer_authorities=(REVIEWER,),
    )


def _finding(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "title": "Destination swap can erase evidence",
        "severity": "important",
        "paths": ["src/mlx_vq/recovery_campaign/review.py"],
        "threat": "same-uid replacement between validation and mutation",
        "red_argv": [
            ".venv/bin/python", "-m", "pytest", "-q", "tests/test_review.py"
        ],
    }
    payload.update(changes)
    return payload


def test_strict_review_record_is_immutable_and_exactly_fingerprinted() -> None:
    record = _record()
    first = review_fingerprint(record)
    assert len(first) == 64
    changed = _payload()
    changed["threat_model"] = ["different threat"]
    assert review_fingerprint(_record(changed)) != first
    with pytest.raises(Exception):
        record.name = "mutated"  # type: ignore[misc]
    with pytest.raises(ReviewRecordError, match="immutable tuple"):
        replace(record, owned_paths=list(record.owned_paths))


def test_review_schema_version_rejects_boolean_alias_for_one() -> None:
    payload = _payload()
    payload["schema_version"] = True
    with pytest.raises(ReviewRecordError, match="schema"):
        _record(payload)
    with pytest.raises(ReviewRecordError, match="schema"):
        replace(_record(), schema_version=True)


@pytest.mark.parametrize(
    "mutation",
    (
        "unknown",
        "unsafe-owned",
        "overlap",
        "bad-hash",
        "contradictory-findings",
        "self-review",
        "stale-profile",
        "stale-head",
        "partial-authority",
    ),
)
def test_review_record_rejects_invalid_or_stale_authority(mutation: str) -> None:
    payload = _payload()
    expected_profile = PROFILE
    expected_base = BASE
    expected_head = HEAD
    reviewers = (REVIEWER,)
    if mutation == "unknown":
        payload["extra"] = True
    elif mutation == "unsafe-owned":
        payload["owned_paths"] = ["../escape"]
    elif mutation == "overlap":
        payload["out_of_scope_paths"] = ["src/mlx_vq"]
    elif mutation == "bad-hash":
        payload["verification_evidence"] = [{"name": "x", "sha256": "bad"}]
    elif mutation == "contradictory-findings":
        payload["resolved_finding_fingerprints"] = ["c" * 64]
        payload["known_diagnostic_fingerprints"] = ["c" * 64]
    elif mutation == "self-review":
        payload["reviewer_authority"] = "implementer-self-review"
        reviewers = (REVIEWER, "implementer-self-review")
    elif mutation == "stale-profile":
        expected_profile = "f" * 64
    elif mutation == "stale-head":
        expected_head = "3" * 40
    else:
        payload["head_commit"] = None
    with pytest.raises(ReviewRecordError):
        parse_review_record(
            payload,
            expected_profile_sha256=expected_profile,
            expected_base=expected_base,
            expected_head=expected_head,
            recognized_reviewer_authorities=reviewers,
        )


def test_content_authority_mode_is_mutually_exclusive_with_commit_range() -> None:
    payload = _payload()
    payload["base_commit"] = None
    payload["head_commit"] = None
    payload["reviewed_content_sha256"] = "d" * 64
    record = parse_review_record(
        payload,
        expected_profile_sha256=PROFILE,
        expected_content_sha256="d" * 64,
        recognized_reviewer_authorities=(REVIEWER,),
    )
    assert record.reviewed_content_sha256 == "d" * 64


@pytest.mark.parametrize(
    ("base", "head", "expected_base", "expected_head"),
    (
        (BASE, HEAD, None, HEAD),
        (BASE, HEAD, BASE, None),
        (BASE, HEAD, "3" * 40, HEAD),
        ("0" * 40, HEAD, "0" * 40, HEAD),
        (HEAD, HEAD, HEAD, HEAD),
    ),
)
def test_commit_authority_requires_exact_external_nontrivial_range(
    base: str,
    head: str,
    expected_base: str | None,
    expected_head: str | None,
) -> None:
    payload = _payload()
    payload["base_commit"] = base
    payload["head_commit"] = head
    with pytest.raises(ReviewRecordError, match="authorit|range|zero"):
        parse_review_record(
            payload,
            expected_profile_sha256=PROFILE,
            expected_base=expected_base,
            expected_head=expected_head,
            recognized_reviewer_authorities=(REVIEWER,),
        )


def test_content_authority_fails_closed_without_external_expected_digest() -> None:
    payload = _payload()
    payload["base_commit"] = None
    payload["head_commit"] = None
    payload["reviewed_content_sha256"] = "d" * 64
    with pytest.raises(ReviewRecordError, match="authorit"):
        parse_review_record(
            payload,
            expected_profile_sha256=PROFILE,
            recognized_reviewer_authorities=(REVIEWER,),
        )


def test_reviewer_authority_inventory_cannot_be_a_bare_string() -> None:
    with pytest.raises(ReviewRecordError, match="authorit"):
        parse_review_record(
            _payload(),
            expected_profile_sha256=PROFILE,
            expected_base=BASE,
            expected_head=HEAD,
            recognized_reviewer_authorities=REVIEWER,  # type: ignore[arg-type]
        )


def test_direct_review_evidence_identity_is_intrinsically_strict() -> None:
    with pytest.raises(ReviewRecordError):
        ReviewEvidenceIdentity("", "bad")


def test_direct_or_replaced_review_record_cannot_dismiss_findings() -> None:
    finding = _finding()
    fingerprint = finding_fingerprint(finding)
    values = dict(
        schema_version=1,
        name="forged",
        base_commit=BASE,
        head_commit=HEAD,
        reviewed_content_sha256=None,
        profile_sha256=PROFILE,
        owned_paths=("src/mlx_vq/recovery_campaign/review.py",),
        frozen_decisions=("x",),
        out_of_scope_paths=("runs",),
        threat_model=("t",),
        verification_evidence=(ReviewEvidenceIdentity("x", "b" * 64),),
        resolved_finding_fingerprints=(fingerprint,),
        known_diagnostic_fingerprints=(),
        result_status="approved",
    )
    with pytest.raises(ReviewRecordError):
        ReviewRecord(
            reviewer_authority="implementer-self-review",
            **values,
        )
    unparsed = ReviewRecord(reviewer_authority=REVIEWER, **values)
    assert classify_finding(finding, unparsed) == "new"
    with pytest.raises(ReviewRecordError, match="parsed|authorit"):
        review_fingerprint(unparsed)
    with pytest.raises(ReviewRecordError, match="parsed|authorit"):
        render_red_test_queue((finding,), unparsed)

    replaced = replace(
        _record(),
        resolved_finding_fingerprints=(fingerprint,),
    )
    assert classify_finding(finding, replaced) == "new"


def test_json_loader_rejects_duplicate_keys_and_nonfinite_values(tmp_path: Path) -> None:
    path = tmp_path / "review.json"
    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ReviewRecordError, match="duplicate"):
        load_review_record(
            path,
            expected_profile_sha256=PROFILE,
            recognized_reviewer_authorities=(REVIEWER,),
        )
    path.write_text(
        json.dumps(_payload()).replace(
            '"schema_version": 1', '"schema_version": NaN'
        )
    )
    with pytest.raises(ReviewRecordError, match="finite|JSON"):
        load_review_record(
            path,
            expected_profile_sha256=PROFILE,
            recognized_reviewer_authorities=(REVIEWER,),
        )


@pytest.mark.parametrize("leaf_kind", ("symlink", "fifo"))
def test_json_loader_rejects_nonregular_leaf_without_blocking(
    leaf_kind: str, tmp_path: Path
) -> None:
    path = tmp_path / "review.json"
    if leaf_kind == "symlink":
        target = tmp_path / "target.json"
        target.write_text(json.dumps(_payload()))
        path.symlink_to(target)
    else:
        os.mkfifo(path)
    with pytest.raises(ReviewRecordError, match="regular|links"):
        load_review_record(
            path,
            expected_profile_sha256=PROFILE,
            expected_base=BASE,
            expected_head=HEAD,
            recognized_reviewer_authorities=(REVIEWER,),
        )


def test_finding_fingerprint_is_canonical_not_prose_substring_matching() -> None:
    finding = _finding()
    reordered = dict(reversed(list(finding.items())))
    assert finding_fingerprint(finding) == finding_fingerprint(reordered)
    changed = _finding(threat="different actionable threat")
    assert finding_fingerprint(changed) != finding_fingerprint(finding)


def test_finding_classifications_cover_exact_states() -> None:
    finding = _finding()
    fingerprint = finding_fingerprint(finding)
    payload = _payload()
    payload["resolved_finding_fingerprints"] = [fingerprint]
    assert classify_finding(finding, _record(payload)) == "resolved"
    payload = _payload()
    payload["known_diagnostic_fingerprints"] = [fingerprint]
    assert classify_finding(finding, _record(payload)) == "known-diagnostic"
    assert (
        classify_finding(
            finding, _record(), duplicate_fingerprints=(fingerprint,)
        )
        == "duplicate"
    )
    assert (
        classify_finding(
            finding, _record(), duplicate_fingerprints=[fingerprint]
        )
        == "duplicate"
    )
    outside = _finding(paths=["runs/private.json"])
    assert classify_finding(outside, _record()) == "out-of-scope"
    assert classify_finding(finding, _record()) == "new"


def test_ambiguous_malformed_and_cross_boundary_findings_remain_new() -> None:
    record = _record()
    assert classify_finding({"title": "partial"}, record) == "new"
    crossing = _finding(
        paths=["src/mlx_vq/recovery_campaign/review.py", "runs/private.json"]
    )
    assert classify_finding(crossing, record) == "new"
    crossing_fingerprint = finding_fingerprint(crossing)
    for field in (
        "resolved_finding_fingerprints",
        "known_diagnostic_fingerprints",
    ):
        payload = _payload()
        payload[field] = [crossing_fingerprint]
        assert classify_finding(crossing, _record(payload)) == "new"
    assert (
        classify_finding(
            crossing,
            record,
            duplicate_fingerprints=(crossing_fingerprint,),
        )
        == "new"
    )
    prose_lookalike = _finding(title="resolved " + "e" * 64)
    assert classify_finding(prose_lookalike, record) == "new"


@pytest.mark.parametrize(
    "inventory",
    (
        "e" * 64,
        ("bad",),
        ("e" * 64, "e" * 64),
    ),
)
def test_invalid_duplicate_inventory_never_auto_dismisses(inventory) -> None:
    assert (
        classify_finding(
            _finding(),
            _record(),
            duplicate_fingerprints=inventory,
        )
        == "new"
    )


@pytest.mark.parametrize(
    "argv",
    (
        ["/bin/sh", "-c", "pytest"],
        ["/usr/bin/env", "bash", "-c", "pytest"],
        ["python", "-m", "pytest", "../tests/test_review.py"],
        ["python", "-m", "pytest", "tests/test_review.py\nignored"],
        ["/tmp/python", "-m", "pytest", "tests/test_review.py"],
        ["python", "-m", "pytest", "tests/test_review.py"],
        ["pytest", "tests/test_review.py"],
        [".venv/bin/python", "-m", "pytest", "/tmp/attacker_test.py"],
        [
            ".venv/bin/python", "-m", "pytest", "--rootdir=/tmp",
            "tests/test_review.py",
        ],
    ),
)
def test_finding_red_argv_rejects_wrappers_traversal_and_controls(argv) -> None:
    finding = _finding(red_argv=argv)
    with pytest.raises(ReviewRecordError, match="argv"):
        finding_fingerprint(finding)
    assert classify_finding(finding, _record()) == "new"


def test_new_findings_render_as_deterministic_red_queue_only() -> None:
    first = _finding(title="First")
    second = _finding(title="Second")
    duplicate = finding_fingerprint(second)
    queue = render_red_test_queue(
        (second, first),
        _record(),
        duplicate_fingerprints=(duplicate,),
    )
    assert len(queue) == 1
    assert queue[0]["classification"] == "new"
    assert queue[0]["title"] == "First"
    assert queue == render_red_test_queue(
        (first, second),
        _record(),
        duplicate_fingerprints=(duplicate,),
    )
    assert queue[0]["finding_sha256"] == finding_fingerprint(first)


def test_review_fingerprint_matches_canonical_body_hash() -> None:
    record = _record()
    body = record.canonical_body()
    expected = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert review_fingerprint(record) == expected
