from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
import json

import pytest


OWNER = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
ACTIVATION_ID = "activation-0001"
BUCKET = "keep-glm52-production"
KEY = (
    f"campaigns/{RUN_ID}/submissions/production/generations/"
    "00000001/terminal/PRODUCTION_TERMINAL_V2.json"
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _terminal_raw() -> bytes:
    body = {
        "account_id": OWNER,
        "activation_id": ACTIVATION_ID,
        "generation": 1,
        "generation_text": "00000001",
        "record_type": "glm52_production_terminal_v2",
        "region": REGION,
        "run_id": RUN_ID,
        "schema_version": 2,
    }
    return _canonical(
        {
            **body,
            "canonical_body_sha256": hashlib.sha256(
                _canonical(body)
            ).hexdigest(),
        }
    ) + b"\n"


def _accepted_gpu_spend_snapshot_raw(
    **extra: object,
) -> tuple[dict[str, object], bytes]:
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_gpu_spend_snapshot_v1",
        "run_id": RUN_ID,
        "campaign_identity_sha256": "1" * 64,
        "descriptor_sha256": "2" * 64,
        "descriptor_body_sha256": "3" * 64,
        "approval_sha256": "4" * 64,
        "approval_body_sha256": "5" * 64,
        "gpu_spend_ledger_latest_sha256": "6" * 64,
        "gpu_spend_ledger_latest_body_sha256": "7" * 64,
        "gpu_spend_ledger_genesis_sha256": "8" * 64,
        "gpu_spend_ledger_record_count": 4,
        "gpu_spend_ledger_tip_record_sha256": "9" * 64,
        "gpu_spend_ledger_file_sha256": "a" * 64,
        "ec2_allocation_history_sha256": "b" * 64,
        "ec2_allocation_instance_ids": [
            "i-00000000000000001",
            "i-00000000000000002",
        ],
        "observed_at": "2026-07-26T11:50:00Z",
        "approved_gpu_runtime_seconds": 86_400,
        "approved_gpu_cost_usd": 1_320.96,
        "hourly_cost_usd": 55.04,
        "consumed_gpu_seconds": 10_800,
        "remaining_gpu_seconds": 75_600,
        "consumed_gpu_cost_usd": 165.12,
        "remaining_gpu_cost_usd": 1_155.84,
        "qualification_allowance_seconds": 14_400,
        "qualification_allowance_cost_usd": 220.16,
        "open_allocation_count": 0,
    }
    body.update(extra)
    body_sha = hashlib.sha256(_canonical(body)).hexdigest()
    return body, _canonical(
        {**body, "snapshot_body_sha256": body_sha}
    ) + b"\n"


def test_candidate_requires_canonical_bytes_exact_hashes_and_closed_metadata() -> None:
    from glm52_enforcement.s3_records import (
        build_immutable_json_candidate,
        validate_immutable_json_candidate,
    )

    candidate = build_immutable_json_candidate(
        record_kind="terminal-v2",
        bucket=BUCKET,
        key=KEY,
        raw=_terminal_raw(),
        activation_id=ACTIVATION_ID,
        generation=1,
    )
    assert validate_immutable_json_candidate(candidate) is candidate
    assert candidate.content_type == "application/json"
    assert dict(candidate.metadata) == {
        "glm52-account-id": OWNER,
        "glm52-activation-id": ACTIVATION_ID,
        "glm52-body-sha256": hashlib.sha256(
            _canonical(
                {
                    "account_id": OWNER,
                    "activation_id": ACTIVATION_ID,
                    "generation": 1,
                    "generation_text": "00000001",
                    "record_type": "glm52_production_terminal_v2",
                    "region": REGION,
                    "run_id": RUN_ID,
                    "schema_version": 2,
                }
            )
        ).hexdigest(),
        "glm52-candidate-identity-sha256": candidate.candidate_identity_sha256,
        "glm52-file-sha256": hashlib.sha256(_terminal_raw()).hexdigest(),
        "glm52-generation-text": "00000001",
        "glm52-record-kind": "terminal-v2",
        "glm52-region": REGION,
        "glm52-run-id": RUN_ID,
    }

    mutations = (
        replace(candidate, raw=candidate.raw[:-1]),
        replace(candidate, file_sha256="f" * 64),
        replace(candidate, body_sha256="e" * 64),
        replace(
            candidate,
            metadata=candidate.metadata + (("glm52-extra", "forbidden"),),
        ),
        replace(candidate, candidate_identity_sha256="d" * 64),
    )
    for mutation in mutations:
        with pytest.raises(ValueError):
            validate_immutable_json_candidate(mutation)


def test_terminal_marker_identity_is_exact_four_field_projection() -> None:
    from glm52_enforcement.s3_records import (
        build_immutable_json_candidate,
        build_s3_object_identity,
        terminal_marker_identity,
        validate_terminal_marker_identity,
    )

    candidate = build_immutable_json_candidate(
        record_kind="terminal-v2",
        bucket=BUCKET,
        key=KEY,
        raw=_terminal_raw(),
        activation_id=ACTIVATION_ID,
        generation=1,
    )
    object_identity = build_s3_object_identity(
        candidate=candidate,
        version_id="opaque-version-one",
        content_length=len(candidate.raw),
        etag='"0123456789abcdef0123456789abcdef"',
        last_modified="2026-07-28T12:00:00Z",
        checksum_sha256_base64=base64.b64encode(
            hashlib.sha256(candidate.raw).digest()
        ).decode("ascii"),
        checksum_type="FULL_OBJECT",
        content_type="application/json",
        metadata=candidate.metadata,
    )
    marker = terminal_marker_identity(object_identity)
    assert marker.key == KEY
    assert marker.version_id == "opaque-version-one"
    assert marker.body_sha256 == candidate.body_sha256
    assert marker.canonical_identity_sha256 == hashlib.sha256(
        _canonical(
            {
                "body_sha256": candidate.body_sha256,
                "key": KEY,
                "version_id": "opaque-version-one",
            }
        )
    ).hexdigest()
    assert validate_terminal_marker_identity(marker) is marker
    with pytest.raises(ValueError):
        validate_terminal_marker_identity(
            replace(marker, canonical_identity_sha256="f" * 64)
        )


def test_candidate_metadata_scopes_existing_source_without_rewriting_schema() -> None:
    from glm52_enforcement.s3_records import (
        build_immutable_json_candidate,
        validate_immutable_json_candidate,
    )

    body, raw = _accepted_gpu_spend_snapshot_raw()
    body_sha = hashlib.sha256(_canonical(body)).hexdigest()
    candidate = build_immutable_json_candidate(
        record_kind="gpu-spend-snapshot",
        bucket=BUCKET,
        key=f"campaigns/{RUN_ID}/spend-snapshots/{body_sha}/GPU_SPEND_SNAPSHOT.json",
        raw=raw,
        activation_id=ACTIVATION_ID,
        generation=1,
    )
    assert validate_immutable_json_candidate(candidate) is candidate
    assert "account_id" not in body
    assert "region" not in body
    assert dict(candidate.metadata)["glm52-account-id"] == OWNER
    assert dict(candidate.metadata)["glm52-region"] == REGION
    assert dict(candidate.metadata)["glm52-activation-id"] == ACTIVATION_ID
    assert dict(candidate.metadata)["glm52-generation-text"] == "00000001"

    for field, wrong in (
        ("account_id", "000000000000"),
        ("region", "us-east-1"),
    ):
        wrong_body, wrong_raw = _accepted_gpu_spend_snapshot_raw(
            **{field: wrong}
        )
        wrong_body_sha = hashlib.sha256(
            _canonical(wrong_body)
        ).hexdigest()
        with pytest.raises(ValueError, match=field):
            build_immutable_json_candidate(
                record_kind="gpu-spend-snapshot",
                bucket=BUCKET,
                key=(
                    f"campaigns/{RUN_ID}/spend-snapshots/{wrong_body_sha}/"
                    "GPU_SPEND_SNAPSHOT.json"
                ),
                raw=wrong_raw,
                activation_id=ACTIVATION_ID,
                generation=1,
            )
        with pytest.raises(ValueError, match=field):
            validate_immutable_json_candidate(
                replace(candidate, raw=wrong_raw)
            )


def test_record_kind_is_closed_to_its_exact_frozen_coordinate_family() -> None:
    from glm52_enforcement import s3_keys
    from glm52_enforcement.s3_records import build_immutable_json_candidate

    sha = "a" * 64
    _body, authority_raw = _accepted_gpu_spend_snapshot_raw()
    exact = {
        "gpu-spend-approval": s3_keys.gpu_spend_approval_s3_key(
            run_id=RUN_ID, approval_file_sha256=sha
        ),
        "gpu-spend-snapshot": s3_keys.gpu_spend_snapshot_s3_key(
            run_id=RUN_ID, snapshot_body_sha256=sha
        ),
        "production-submission-intent": (
            s3_keys.production_submission_intent_s3_key(
                run_id=RUN_ID, intent_body_sha256=sha
            )
        ),
        "production-controller-baseline": (
            s3_keys.production_controller_baseline_s3_key(
                run_id=RUN_ID, baseline_body_sha256=sha
            )
        ),
        "production-control-plane-readiness": (
            s3_keys.production_must_start_control_plane_ready_s3_key(
                run_id=RUN_ID,
                intent_body_sha256=sha,
                control_plane_ready_body_sha256=sha,
            )
        ),
        "production-submission-acquisition": (
            s3_keys.production_submission_acquired_s3_key(
                run_id=RUN_ID, descriptor_file_sha256=sha
            )
        ),
        "fence-genesis": s3_keys.fence_genesis_s3_key(run_id=RUN_ID),
        "fence-successor": s3_keys.fence_successor_s3_key(
            run_id=RUN_ID, predecessor_body_sha256=sha
        ),
        "generation-claim": s3_keys.production_generation_claim_s3_key(
            run_id=RUN_ID, generation=1
        ),
        "start-decision": (
            s3_keys.production_generation_start_decision_s3_key(
                run_id=RUN_ID, generation=1
            )
        ),
        "generation-terminal": (
            s3_keys.production_generation_terminal_s3_key(
                run_id=RUN_ID, generation=1
            )
        ),
        "terminal-v2": s3_keys.production_terminal_v2_s3_key(
            run_id=RUN_ID, generation=1
        ),
        "sky-post-handoff": s3_keys.sky_post_handoff_s3_key(
            run_id=RUN_ID, generation=1
        ),
        "bootstrap-ready": s3_keys.bootstrap_ready_s3_key(
            run_id=RUN_ID, generation=1, allocation_ordinal=1
        ),
        "worker-graceful-stop": s3_keys.worker_graceful_stop_s3_key(
            run_id=RUN_ID, generation=1, allocation_ordinal=1
        ),
        "campaign-drained": s3_keys.campaign_drained_s3_key(
            run_id=RUN_ID, generation=1
        ),
        "support-plane-finalized": (
            s3_keys.support_plane_finalized_s3_key(
                run_id=RUN_ID, activation_id=ACTIVATION_ID
            )
        ),
        "h1g-drained": s3_keys.h1g_drained_s3_key(
            run_id=RUN_ID, activation_id=ACTIVATION_ID
        ),
    }
    for record_kind, key in exact.items():
        candidate = build_immutable_json_candidate(
            record_kind=record_kind,
            bucket=BUCKET,
            key=key,
            raw=authority_raw,
            activation_id=ACTIVATION_ID,
            generation=1,
        )
        assert candidate.key == key

    invalid = (
        ("unknown-kind", KEY),
        ("terminal-v2", KEY.replace("PRODUCTION_TERMINAL_V2", "OTHER")),
        ("terminal-v2", exact["gpu-spend-snapshot"]),
        (
            "gpu-spend-snapshot",
            exact["gpu-spend-snapshot"].replace(
                "GPU_SPEND_SNAPSHOT.json", "SOURCE.json"
            ),
        ),
    )
    for record_kind, key in invalid:
        with pytest.raises(ValueError):
            build_immutable_json_candidate(
                record_kind=record_kind,
                bucket=BUCKET,
                key=key,
                raw=authority_raw,
                activation_id=ACTIVATION_ID,
                generation=1,
            )
