"""Task 12 exact immutable S3 coordinates and record-family closure."""

from __future__ import annotations

import hashlib
import json

import pytest


OWNER = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
ACTIVATION = "activation-0001"
BUCKET = "keep-glm52-production"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _raw(
    *,
    run_id: str = RUN_ID,
    activation_id: str = ACTIVATION,
    generation: int = 1,
    activation_ordinal: int | None = None,
    delete_logical_attempt: int | None = None,
    handoff_hash: bool = False,
) -> bytes:
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_task12_s3_fixture_v1",
        "account_id": OWNER,
        "region": REGION,
        "run_id": run_id,
        "activation_id": activation_id,
        "generation": generation,
        "generation_text": f"{generation:08d}",
    }
    if activation_ordinal is not None:
        body["activation_ordinal"] = activation_ordinal
    if delete_logical_attempt is not None:
        body["delete_logical_attempt"] = delete_logical_attempt
        body["delete_logical_attempt_text"] = (
            f"{delete_logical_attempt:08d}"
        )
    hash_field = (
        "handoff_body_sha256"
        if handoff_hash
        else "canonical_body_sha256"
    )
    return _canonical(
        {
            **body,
            hash_field: hashlib.sha256(_canonical(body)).hexdigest(),
        }
    ) + b"\n"


def test_task12_key_helpers_emit_only_architecture_backed_coordinates() -> None:
    """Break caught: a generic namespace or root marker becomes authority."""

    from glm52_enforcement.s3_keys import (
        h1g_drained_s3_key,
        production_terminal_v2_s3_key,
        sky_job_bound_s3_key,
        sky_post_handoff_s3_key,
        sky_request_correlated_s3_key,
        snapshot_cleanup_authority_audit_s3_key,
        support_plane_finalized_s3_key,
    )

    generation_prefix = (
        f"campaigns/{RUN_ID}/submissions/production/generations/00000001"
    )
    assert sky_post_handoff_s3_key(run_id=RUN_ID, generation=1) == (
        generation_prefix + "/handoff/SKY_POST_HANDOFF.json"
    )
    assert sky_request_correlated_s3_key(
        run_id=RUN_ID, generation=1
    ) == (
        generation_prefix + "/requests/SKY_REQUEST_CORRELATED.json"
    )
    assert sky_job_bound_s3_key(run_id=RUN_ID, generation=1) == (
        generation_prefix + "/bindings/SKY_JOB_BOUND.json"
    )
    assert production_terminal_v2_s3_key(
        run_id=RUN_ID, generation=1
    ) == (
        generation_prefix + "/terminal/PRODUCTION_TERMINAL_V2.json"
    )
    activation_prefix = (
        f"campaigns/{RUN_ID}/submissions/production/activations/"
        f"{ACTIVATION}/finalization"
    )
    assert support_plane_finalized_s3_key(
        run_id=RUN_ID, activation_id=ACTIVATION
    ) == activation_prefix + "/SUPPORT_PLANE_FINALIZED.json"
    assert h1g_drained_s3_key(
        run_id=RUN_ID, activation_id=ACTIVATION
    ) == activation_prefix + "/H1G_DRAINED.json"
    assert snapshot_cleanup_authority_audit_s3_key(
        run_id=RUN_ID,
        activation_ordinal=1,
        delete_logical_attempt=2,
    ) == (
        f"campaigns/{RUN_ID}/h1g/snapshot-cleanup/00000001/"
        "audits/00000002.json"
    )


def test_task12_record_kinds_have_one_closed_physical_family() -> None:
    """Break caught: a retained writer is relabelled as generic ACTIVATION."""

    from glm52_enforcement.s3_records import (
        task12_s3_record_family,
        validate_task12_s3_record_family_rows,
    )

    rows = (
        ("sky-post-handoff", "sky-post-handoff"),
        ("recovery-handoff", "sky-post-handoff"),
        ("sky-request-correlated", "sky-request-correlated"),
        ("sky-job-bound", "sky-job-bound"),
        ("terminal-v2", "production-terminal-v2"),
        ("support-plane-finalized", "support-plane-finalized"),
        ("h1g-drained", "h1g-drained"),
        (
            "snapshot-cleanup-authority-audit",
            "snapshot-cleanup-authority-audit",
        ),
    )
    assert validate_task12_s3_record_family_rows(rows) == rows
    for record_kind, family in rows:
        assert task12_s3_record_family(record_kind) == family

    for invalid in (
        rows + (rows[0],),
        rows[:-1],
        rows[:-1] + (("unknown-kind", "sky-post-handoff"),),
        rows[:-1] + (("operator-disposition", "operator-disposition"),),
        rows[:-1]
        + (("snapshot-disposition", "snapshot-disposition"),),
        rows[:-1]
        + (("snapshot-cleanup-transition", "snapshot-cleanup"),),
    ):
        with pytest.raises(ValueError):
            validate_task12_s3_record_family_rows(invalid)
    with pytest.raises(ValueError, match="not an S3 record kind"):
        task12_s3_record_family("operator-disposition")


def test_candidates_bind_record_kind_to_exact_generation_and_activation() -> None:
    """Break caught: family-shaped keys shift generation or activation."""

    from glm52_enforcement.s3_keys import (
        h1g_drained_s3_key,
        production_terminal_v2_s3_key,
        sky_job_bound_s3_key,
        sky_post_handoff_s3_key,
        sky_request_correlated_s3_key,
        support_plane_finalized_s3_key,
    )
    from glm52_enforcement.s3_records import (
        build_immutable_json_candidate,
    )

    exact = (
        (
            "sky-post-handoff",
            sky_post_handoff_s3_key(run_id=RUN_ID, generation=1),
            _raw(handoff_hash=True),
        ),
        (
            "recovery-handoff",
            sky_post_handoff_s3_key(run_id=RUN_ID, generation=1),
            _raw(handoff_hash=True),
        ),
        (
            "sky-request-correlated",
            sky_request_correlated_s3_key(
                run_id=RUN_ID, generation=1
            ),
            _raw(),
        ),
        (
            "sky-job-bound",
            sky_job_bound_s3_key(run_id=RUN_ID, generation=1),
            _raw(),
        ),
        (
            "terminal-v2",
            production_terminal_v2_s3_key(
                run_id=RUN_ID, generation=1
            ),
            _raw(),
        ),
        (
            "support-plane-finalized",
            support_plane_finalized_s3_key(
                run_id=RUN_ID, activation_id=ACTIVATION
            ),
            _raw(),
        ),
        (
            "h1g-drained",
            h1g_drained_s3_key(
                run_id=RUN_ID, activation_id=ACTIVATION
            ),
            _raw(),
        ),
    )
    for record_kind, key, raw in exact:
        candidate = build_immutable_json_candidate(
            record_kind=record_kind,
            bucket=BUCKET,
            key=key,
            raw=raw,
            activation_id=ACTIVATION,
            generation=1,
        )
        assert candidate.key == key

    mutations = (
        (
            "sky-request-correlated",
            sky_job_bound_s3_key(run_id=RUN_ID, generation=1),
            _raw(),
        ),
        (
            "sky-job-bound",
            sky_job_bound_s3_key(run_id=RUN_ID, generation=2),
            _raw(),
        ),
        (
            "terminal-v2",
            production_terminal_v2_s3_key(
                run_id=RUN_ID, generation=2
            ),
            _raw(),
        ),
        (
            "support-plane-finalized",
            support_plane_finalized_s3_key(
                run_id=RUN_ID, activation_id="activation-0002"
            ),
            _raw(),
        ),
        (
            "h1g-drained",
            h1g_drained_s3_key(
                run_id=RUN_ID, activation_id="activation-0002"
            ),
            _raw(),
        ),
    )
    for record_kind, key, raw in mutations:
        with pytest.raises(ValueError, match="coordinate"):
            build_immutable_json_candidate(
                record_kind=record_kind,
                bucket=BUCKET,
                key=key,
                raw=raw,
                activation_id=ACTIVATION,
                generation=1,
            )


def test_snapshot_cleanup_audit_binds_ordinal_and_attempt_text() -> None:
    """Break caught: cleanup evidence is accepted for a shifted lineage slot."""

    from glm52_enforcement.s3_keys import (
        snapshot_cleanup_authority_audit_s3_key,
    )
    from glm52_enforcement.s3_records import (
        build_immutable_json_candidate,
    )

    raw = _raw(
        activation_ordinal=3,
        delete_logical_attempt=7,
    )
    key = snapshot_cleanup_authority_audit_s3_key(
        run_id=RUN_ID,
        activation_ordinal=3,
        delete_logical_attempt=7,
    )
    candidate = build_immutable_json_candidate(
        record_kind="snapshot-cleanup-authority-audit",
        bucket=BUCKET,
        key=key,
        raw=raw,
        activation_id=ACTIVATION,
        generation=1,
    )
    assert candidate.key.endswith(
        "/snapshot-cleanup/00000003/audits/00000007.json"
    )

    shifted = (
        snapshot_cleanup_authority_audit_s3_key(
            run_id=RUN_ID,
            activation_ordinal=2,
            delete_logical_attempt=7,
        ),
        snapshot_cleanup_authority_audit_s3_key(
            run_id=RUN_ID,
            activation_ordinal=3,
            delete_logical_attempt=8,
        ),
    )
    for wrong_key in shifted:
        with pytest.raises(ValueError, match="exact record coordinate"):
            build_immutable_json_candidate(
                record_kind="snapshot-cleanup-authority-audit",
                bucket=BUCKET,
                key=wrong_key,
                raw=raw,
                activation_id=ACTIVATION,
                generation=1,
            )

    bad_text = json.loads(raw)
    bad_text["delete_logical_attempt_text"] = "00000008"
    unhashed = dict(bad_text)
    del unhashed["canonical_body_sha256"]
    bad_text["canonical_body_sha256"] = hashlib.sha256(
        _canonical(unhashed)
    ).hexdigest()
    with pytest.raises(ValueError, match="logical attempt"):
        build_immutable_json_candidate(
            record_kind="snapshot-cleanup-authority-audit",
            bucket=BUCKET,
            key=key,
            raw=_canonical(bad_text) + b"\n",
            activation_id=ACTIVATION,
            generation=1,
        )


def test_task12_families_reject_foreign_run_and_dynamodb_only_kinds() -> None:
    """Break caught: a foreign run or DynamoDB record gains S3 metadata."""

    from glm52_enforcement.s3_keys import (
        sky_request_correlated_s3_key,
    )
    from glm52_enforcement.s3_records import (
        build_immutable_json_candidate,
    )

    foreign_run = "glm52-sky-20260725"
    with pytest.raises(ValueError, match="fixed production run"):
        build_immutable_json_candidate(
            record_kind="sky-request-correlated",
            bucket=BUCKET,
            key=sky_request_correlated_s3_key(
                run_id=foreign_run,
                generation=1,
            ),
            raw=_raw(run_id=foreign_run),
            activation_id=ACTIVATION,
            generation=1,
        )
    for record_kind in (
        "operator-disposition",
        "snapshot-disposition",
        "snapshot-cleanup-transition",
    ):
        with pytest.raises(ValueError, match="frozen Task 4 family"):
            build_immutable_json_candidate(
                record_kind=record_kind,
                bucket=BUCKET,
                key=(
                    f"campaigns/{RUN_ID}/submissions/production/"
                    f"activations/{ACTIVATION}/finalization/"
                    f"{record_kind.upper()}.json"
                ),
                raw=_raw(),
                activation_id=ACTIVATION,
                generation=1,
            )
