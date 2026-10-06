"""Focused proof for the normal Task 11 SKY_POST_HANDOFF contract."""

from __future__ import annotations

import base64
import hashlib
import io
import json
from types import SimpleNamespace

import pytest

from glm52_enforcement.canonical import canonical_json_bytes
from glm52_enforcement.s3_adapter import (
    ConditionalCreateResult,
    S3PublicationServices,
)
from glm52_enforcement.s3_records import S3ObjectIdentity
from glm52_enforcement.support_effect_writer_handler import (
    effect_write_result_from_payload,
    effect_write_result_to_payload,
    materialize_effect_builder_arguments,
)
from glm52_enforcement.task11_effect_writers import (
    EffectWriteResult,
    EffectWriterRequest,
    Task11EffectWriterError,
    prepare_effect_candidate,
    write_closure_handoff,
)
from glm52_enforcement.task11_production import (
    _build_normal_task11_handoff_bindings,
    _load_task10_production_authority,
)
from glm52_enforcement.task12_correlation import (
    build_sky_post_handoff_record,
)


def _handoff_values() -> dict[str, object]:
    return {
        "run_id": "glm52-sky-20260724",
        "campaign_identity_sha256": "1" * 64,
        "generation": 1,
        "generation_text": "00000001",
        "submit_attempt_id": "2" * 64,
        "decision_key": (
            "campaigns/glm52-sky-20260724/submissions/production/"
            "generations/00000001/START_DECISION.json"
        ),
        "decision_version_id": "decision-version-1",
        "decision_file_sha256": "3" * 64,
        "decision_body_sha256": "4" * 64,
        "sky_post_action_key": (
            "ACTIVATION#activation-1#ACTION#SKY_POST#00000001"
        ),
        "sky_post_consumed_at": "2026-07-28T11:00:00Z",
        "sky_post_outcome_class": "ACCEPTED",
        "expected_sky_job_name": "glm52-sky-20260724",
        "task_yaml_sha256": "5" * 64,
        "request_body_sha256": "6" * 64,
        "api_server_identity_sha256": "7" * 64,
        "sky_request_id": "sky-request-1",
        "post_started_at": "2026-07-28T11:00:01Z",
        "post_completed_or_lost_at": "2026-07-28T11:00:02Z",
        "binding_state": "reconcile-required",
    }


def _closure_document(
    builder_arguments: dict[str, object],
) -> dict[str, object]:
    return {
        "writer_kind": "ClosureHandoff",
        "builder_arguments": builder_arguments,
    }


def test_normal_task11_closure_emits_the_shared_frozen_body_bytes() -> None:
    values = _handoff_values()
    candidate = prepare_effect_candidate(
        EffectWriterRequest(
            writer_kind="ClosureHandoff",
            activation_id="activation-1",
            generation=1,
            action_key=(
                "ACTIVATION#activation-1#ACTION#S3_CREATE#"
                "SKY_POST_HANDOFF#00000001"
            ),
            campaign_bucket=(
                "keep-glm52-models-246813579024-us-west-2"
            ),
            builder_arguments=values,
        )
    )

    expected = build_sky_post_handoff_record(**values)
    assert candidate.raw == canonical_json_bytes(expected) + b"\n"
    assert json.loads(candidate.raw) == expected
    assert "schema_version" not in expected
    assert "record_type" not in expected
    assert "sky_post_handoff_body_sha256" not in expected


def test_real_effect_writer_round_trip_preserves_direct_transport_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch missing timing fields or substitution with the server Date."""

    from glm52_enforcement import task11_effect_writers

    request = EffectWriterRequest(
        writer_kind="ClosureHandoff",
        activation_id="activation-1",
        generation=1,
        action_key=(
            "ACTIVATION#activation-1#ACTION#S3_CREATE#"
            "SKY_POST_HANDOFF#00000001"
        ),
        campaign_bucket="keep-glm52-models-246813579024-us-west-2",
        builder_arguments=_handoff_values(),
    )
    candidate = prepare_effect_candidate(request)
    identity = S3ObjectIdentity(
        bucket=candidate.bucket,
        key=candidate.key,
        version_id="handoff-version-1",
        file_sha256=hashlib.sha256(candidate.raw).hexdigest(),
        body_sha256=json.loads(candidate.raw)["handoff_body_sha256"],
        content_length=len(candidate.raw),
        etag='"handoff-etag"',
        last_modified="2026-07-28T11:00:03Z",
        checksum_sha256_base64=base64.b64encode(
            hashlib.sha256(candidate.raw).digest()
        ).decode("ascii"),
        checksum_type="FULL_OBJECT",
        content_type="application/json",
        metadata=(),
        canonical_identity_sha256="8" * 64,
    )
    direct = ConditionalCreateResult(
        outcome="created-direct",
        object_identity=identity,
        authority_audit_body_sha256="9" * 64,
        closing_revision=41,
        authorized_revision=42,
        provenance="direct-response",
        direct_request_id="request-1",
        direct_server_date="2026-07-28T11:00:02Z",
        direct_request_started_at="2026-07-28T11:00:00Z",
        direct_response_received_at="2026-07-28T11:00:01Z",
        direct_response_authenticated=True,
    )
    monkeypatch.setattr(
        task11_effect_writers,
        "conditional_create_immutable_json",
        lambda **_kwargs: direct,
    )

    result = write_closure_handoff(
        services=S3PublicationServices(
            s3=object(),
            actions=object(),
            h1f=object(),
        ),
        request=request,
    )
    assert result.direct_request_started_at == "2026-07-28T11:00:00Z"
    assert result.direct_response_received_at == "2026-07-28T11:00:01Z"

    restored = effect_write_result_from_payload(
        effect_write_result_to_payload(result),
        expected_writer_kind="ClosureHandoff",
    )
    assert restored == result


def test_normal_task11_closure_rejects_the_compact_legacy_body() -> None:
    compact = {
        "activation_id": "activation-1",
        "generation": 1,
        "admission_identity_sha256": "1" * 64,
        "post_audit_body_sha256": "2" * 64,
        "relay_receipt_sha256": "3" * 64,
        "classification": "POST_ACCEPTED_ONCE",
        "persisted_at": "2026-07-28T11:00:02Z",
    }
    with pytest.raises(
        Task11EffectWriterError,
        match="closure handoff canonical fields drifted",
    ):
        prepare_effect_candidate(
            EffectWriterRequest(
                writer_kind="ClosureHandoff",
                activation_id="activation-1",
                generation=1,
                action_key=(
                    "ACTIVATION#activation-1#ACTION#S3_CREATE#"
                    "SKY_POST_HANDOFF#00000001"
                ),
                campaign_bucket=(
                    "keep-glm52-models-246813579024-us-west-2"
                ),
                builder_arguments=compact,
            )
        )


def test_closure_handler_requires_the_complete_frozen_runtime_binding() -> None:
    values = _handoff_values()
    assert materialize_effect_builder_arguments(
        _closure_document({"legacy": "$runtime.classification"}),
        source_publications=[],
        dependency_results=[],
        runtime_bindings=values,
    ) == values

    incomplete = dict(values)
    incomplete.pop("decision_version_id")
    with pytest.raises(ValueError, match="runtime dependency set is not closed"):
        materialize_effect_builder_arguments(
            _closure_document({}),
            source_publications=[],
            dependency_results=[],
            runtime_bindings=incomplete,
        )


def test_non_closure_writer_rejects_runtime_handoff_facts() -> None:
    with pytest.raises(ValueError, match="runtime dependency set is not closed"):
        materialize_effect_builder_arguments(
            {
                "writer_kind": "DecisionWriter",
                "builder_arguments": {},
            },
            source_publications=[],
            dependency_results=[],
            runtime_bindings=_handoff_values(),
        )


def _decision_result() -> EffectWriteResult:
    decision = {
        "record_type": (
            "glm52_sky_production_generation_start_decision_v1"
        ),
        "run_id": "glm52-sky-20260724",
        "campaign_identity_sha256": "1" * 64,
        "generation": 1,
        "generation_text": "00000001",
        "submit_attempt_id": "2" * 64,
        "sky_job_name": "glm52-sky-20260724",
        "decision": "launch-once",
        "start_decision_body_sha256": "4" * 64,
    }
    raw = canonical_json_bytes(decision) + b"\n"
    identity = S3ObjectIdentity(
        bucket="keep-glm52-models-246813579024-us-west-2",
        key=_handoff_values()["decision_key"],
        version_id="decision-version-1",
        file_sha256="3" * 64,
        body_sha256="4" * 64,
        content_length=len(raw),
        etag='"etag"',
        last_modified="2026-07-28T10:59:59Z",
        checksum_sha256_base64="checksum",
        checksum_type="FULL_OBJECT",
        content_type="application/json",
        metadata=(),
        canonical_identity_sha256="8" * 64,
    )
    return EffectWriteResult(
        writer_kind="DecisionWriter",
        record_kind="start-decision",
        object_identity=identity,
        candidate_identity_sha256="9" * 64,
        authority_audit_body_sha256="a" * 64,
        closing_revision=1,
        authorized_revision=2,
        direct_request_id="request-1",
        direct_server_date="Tue, 28 Jul 2026 10:59:59 GMT",
        direct_response_authenticated=True,
        raw=raw,
        canonical_identity_sha256="b" * 64,
    )


def _classified_action() -> dict[str, object]:
    values = _handoff_values()
    return {
        "record_type": "glm52_production_action",
        "state": "POST_CLASSIFIED",
        "action_kind": "SKY_POST",
        "action_key": values["sky_post_action_key"],
        "run_id": values["run_id"],
        "campaign_identity_sha256": values["campaign_identity_sha256"],
        "activation_id": "activation-1",
        "generation": values["generation"],
        "generation_text": values["generation_text"],
        "candidate_key": values["decision_key"],
        "candidate_file_sha256": values["decision_file_sha256"],
        "candidate_body_sha256": values["decision_body_sha256"],
        "consumed_at": values["sky_post_consumed_at"],
        "outcome_class": values["sky_post_outcome_class"],
        "sky_request_id": values["sky_request_id"],
        "post_started_at": values["post_started_at"],
        "completed_at": values["post_completed_or_lost_at"],
    }


def _production_authority() -> SimpleNamespace:
    values = _handoff_values()
    return SimpleNamespace(
        account_id="246813579024",
        region="us-west-2",
        run_id=values["run_id"],
        managed_mode="production",
        campaign_identity_sha256=values["campaign_identity_sha256"],
        activation_id="activation-1",
        generation=values["generation"],
        generation_text=values["generation_text"],
        action_key=values["sky_post_action_key"],
        action_state="CONSUMED",
        current_activation=True,
        sky_job_name=values["expected_sky_job_name"],
        task_yaml_sha256=values["task_yaml_sha256"],
        request_body_sha256=values["request_body_sha256"],
    )


def test_normal_task11_runtime_bindings_cross_close_all_authorities() -> None:
    result = _build_normal_task11_handoff_bindings(
        run_id="glm52-sky-20260724",
        activation_id="activation-1",
        generation=1,
        action_key=_handoff_values()["sky_post_action_key"],
        action=_classified_action(),
        decision_result=_decision_result(),
        production_authority=_production_authority(),
        api_server_identity_sha256="7" * 64,
        admission_classification="ACCEPTED",
        admission_request_id="sky-request-1",
    )
    assert result == _handoff_values()
    assert build_sky_post_handoff_record(**result)[
        "handoff_body_sha256"
    ]


def test_normal_task11_runtime_bindings_reject_decision_action_drift() -> None:
    action = _classified_action()
    action["candidate_file_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="authority drifted"):
        _build_normal_task11_handoff_bindings(
            run_id="glm52-sky-20260724",
            activation_id="activation-1",
            generation=1,
            action_key=_handoff_values()["sky_post_action_key"],
            action=action,
            decision_result=_decision_result(),
            production_authority=_production_authority(),
            api_server_identity_sha256="7" * 64,
            admission_classification="ACCEPTED",
            admission_request_id="sky-request-1",
        )


class _Task10AuthorityS3:
    def __init__(self, *, sibling: bool = False) -> None:
        self.document = {"canonical_identity_sha256": "a" * 64}
        self.raw = canonical_json_bytes(self.document) + b"\n"
        self.sibling = sibling
        self.calls: list[tuple[str, dict[str, object]]] = []

    @staticmethod
    def _transport(request_id: str) -> dict[str, object]:
        return {
            "HTTPStatusCode": 200,
            "RequestId": request_id,
            "RetryAttempts": 0,
        }

    def list_object_versions(self, **request: object) -> object:
        self.calls.append(("list_object_versions", dict(request)))
        versions = [
            {
                "Key": (
                    "task13/production/task10-production-authority.json"
                ),
                "VersionId": "authority-version-1",
                "IsLatest": True,
                "Size": len(self.raw),
            }
        ]
        if self.sibling:
            versions.append(
                {
                    "Key": (
                        "task13/production/"
                        "task10-production-authority.json.foreign"
                    ),
                    "VersionId": "foreign-version-1",
                    "IsLatest": True,
                    "Size": 1,
                }
            )
        return {
            "Versions": versions,
            "DeleteMarkers": [],
            "IsTruncated": False,
            "ResponseMetadata": self._transport("list-1"),
        }

    def get_object(self, **request: object) -> object:
        self.calls.append(("get_object", dict(request)))
        return {
            "Body": io.BytesIO(self.raw),
            "VersionId": "authority-version-1",
            "ContentLength": len(self.raw),
            "ContentType": "application/json",
            "ChecksumSHA256": base64.b64encode(
                hashlib.sha256(self.raw).digest()
            ).decode("ascii"),
            "ChecksumType": "FULL_OBJECT",
            "Metadata": {
                "record-type": "glm52_task10_production_authority_v1",
                "canonical-identity-sha256": "a" * 64,
            },
            "ResponseMetadata": self._transport("get-1"),
        }


def test_normal_task11_exact_loads_the_sole_task10_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from glm52_enforcement import task11_production

    sentinel = SimpleNamespace(canonical_identity_sha256="a" * 64)
    monkeypatch.setattr(
        task11_production,
        "production_authority_from_mapping",
        lambda value: sentinel if value == {"canonical_identity_sha256": "a" * 64}
        else None,
    )
    s3 = _Task10AuthorityS3()
    assert _load_task10_production_authority(
        s3=s3,
        bucket="keep-glm52-models-246813579024-us-west-2",
    ) is sentinel
    assert [name for name, _ in s3.calls] == [
        "list_object_versions",
        "get_object",
    ]


def test_normal_task11_rejects_task10_authority_prefix_siblings() -> None:
    with pytest.raises(RuntimeError, match="inventory drifted"):
        _load_task10_production_authority(
            s3=_Task10AuthorityS3(sibling=True),
            bucket=(
                "keep-glm52-models-246813579024-us-west-2"
            ),
        )
