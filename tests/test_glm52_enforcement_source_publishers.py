from __future__ import annotations

from dataclasses import fields, replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import pytest


EXPECTED_KINDS = (
    "gpu-spend-snapshot",
    "production-submission-intent",
    "production-controller-baseline",
    "production-control-plane-readiness",
    "production-submission-acquisition",
)
ROOT = Path(__file__).resolve().parents[1]
CALLER = (
    "arn:aws:iam::246813579024:role/"
    "keep-glm52-production-decision-closure"
)
BUCKET = "keep-glm52-production"
ACTIVATION_ID = "activation-0001"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _accepted_chain(tmp_path_factory: pytest.TempPathFactory):
    path = ROOT / "tests/test_glm52_enforcement_source_authorities.py"
    specification = importlib.util.spec_from_file_location(
        "_task5_authority_fixtures_for_publishers", path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module._accepted_chain(tmp_path_factory)


def _publisher_cases(tmp_path_factory: pytest.TempPathFactory):
    from glm52_enforcement.s3_adapter import S3PublicationServices
    from glm52_enforcement.source_authorities import VersionedJsonArtifact
    from glm52_enforcement import source_publishers

    (
        _production_tests,
        _prelaunch_tests,
        _accepted,
        authorities,
        chain,
        snapshot,
    ) = _accepted_chain(tmp_path_factory)

    def local(value):
        return VersionedJsonArtifact(
            key=value.key,
            raw=value.raw,
            version_id=value.version_id,
        )

    common = {
        "caller_arn": CALLER,
        "activation_id": ACTIVATION_ID,
        "generation": 1,
    }
    requests = (
        source_publishers.GpuSpendSnapshotPublicationRequest(
            **common,
            action_key="ACTIVATION#activation-0001#SOURCE#01",
            raw=_canonical(snapshot) + b"\n",
        ),
        source_publishers.ProductionSubmissionIntentPublicationRequest(
            **common,
            action_key="ACTIVATION#activation-0001#SOURCE#02",
            raw=authorities["intent"].raw,
        ),
        source_publishers.ProductionControllerBaselinePublicationRequest(
            **common,
            action_key="ACTIVATION#activation-0001#SOURCE#03",
            raw=_canonical(chain["baseline"]) + b"\n",
            descriptor=local(authorities["descriptor"]),
            intent=local(authorities["intent"]),
        ),
        source_publishers.ProductionControlPlaneReadinessPublicationRequest(
            **common,
            action_key="ACTIVATION#activation-0001#SOURCE#04",
            raw=_canonical(chain["ready"]) + b"\n",
            descriptor=local(authorities["descriptor"]),
            intent=local(authorities["intent"]),
            controller_baseline=local(chain["baseline_artifact"]),
        ),
        source_publishers.ProductionSubmissionAcquisitionPublicationRequest(
            **common,
            action_key="ACTIVATION#activation-0001#SOURCE#05",
            raw=_canonical(chain["acquisition"]) + b"\n",
            descriptor=local(authorities["descriptor"]),
            intent=local(authorities["intent"]),
            controller_baseline=local(chain["baseline_artifact"]),
            must_start_control_plane_ready=local(chain["ready_artifact"]),
            now=chain["acquired_at"],
        ),
    )
    entrypoints = (
        source_publishers.publish_gpu_spend_snapshot,
        source_publishers.publish_production_submission_intent,
        source_publishers.publish_production_controller_baseline,
        source_publishers.publish_production_control_plane_readiness,
        source_publishers.publish_production_submission_acquisition,
    )
    campaign_bucket = str(authorities["intent_record"]["bucket"])
    runtime_types = (
        source_publishers.GpuSpendSnapshotPublisherRuntime,
        source_publishers.ProductionSubmissionIntentPublisherRuntime,
        source_publishers.ProductionControllerBaselinePublisherRuntime,
        source_publishers.ProductionControlPlaneReadinessPublisherRuntime,
        source_publishers.ProductionSubmissionAcquisitionPublisherRuntime,
    )
    result = []
    for kind, request, entrypoint, runtime_type in zip(
        EXPECTED_KINDS, requests, entrypoints, runtime_types
    ):
        role = (
            "arn:aws:iam::246813579024:role/"
            f"keep-glm52-{kind}-publisher"
        )
        runtime = runtime_type(
            services=S3PublicationServices(
                s3=object(), actions=object(), h1f=object()
            ),
            closure_caller_arn=CALLER,
            publisher_role_arn=role,
            campaign_bucket=campaign_bucket,
        )
        result.append((kind, request, entrypoint, runtime))
    return tuple(result)


def _task4_result(task4_request, *, provenance: str = "direct-response"):
    from glm52_enforcement.s3_adapter import ConditionalCreateResult
    from glm52_enforcement.s3_records import S3ObjectIdentity

    candidate = task4_request.candidate
    identity = S3ObjectIdentity(
        bucket=candidate.bucket,
        key=candidate.key,
        version_id=f"version-{candidate.record_kind}",
        file_sha256=candidate.file_sha256,
        body_sha256=candidate.body_sha256,
        content_length=len(candidate.raw),
        etag=f'"{hashlib.md5(candidate.raw).hexdigest()}"',
        last_modified="2026-07-28T12:00:00Z",
        checksum_sha256_base64="checksum",
        checksum_type="FULL_OBJECT",
        content_type="application/json",
        metadata=candidate.metadata,
        canonical_identity_sha256=hashlib.sha256(
            candidate.raw + b"identity"
        ).hexdigest(),
    )
    return ConditionalCreateResult(
        outcome=(
            "created-direct"
            if provenance == "direct-response"
            else "reconciled-existing"
        ),
        object_identity=identity,
        authority_audit_body_sha256="a" * 64,
        closing_revision=1,
        authorized_revision=2,
        provenance=provenance,
        direct_request_id=(
            "request-direct" if provenance == "direct-response" else None
        ),
        direct_server_date=(
            "2026-07-28T12:00:00Z"
            if provenance == "direct-response"
            else None
        ),
        direct_response_authenticated=provenance == "direct-response",
    )


def test_five_public_entrypoints_have_fixed_unique_source_kinds() -> None:
    from glm52_enforcement import source_publishers

    entrypoints = (
        source_publishers.publish_gpu_spend_snapshot,
        source_publishers.publish_production_submission_intent,
        source_publishers.publish_production_controller_baseline,
        source_publishers.publish_production_control_plane_readiness,
        source_publishers.publish_production_submission_acquisition,
    )
    requests = (
        source_publishers.GpuSpendSnapshotPublicationRequest,
        source_publishers.ProductionSubmissionIntentPublicationRequest,
        source_publishers.ProductionControllerBaselinePublicationRequest,
        source_publishers.ProductionControlPlaneReadinessPublicationRequest,
        source_publishers.ProductionSubmissionAcquisitionPublicationRequest,
    )
    assert tuple(item.source_kind for item in requests) == EXPECTED_KINDS
    assert len(set(entrypoints)) == 5
    assert len(set(requests)) == 5


def test_generic_source_writer_and_alternate_coordinate_are_unrepresentable() -> None:
    from glm52_enforcement import source_publishers

    assert not hasattr(source_publishers, "publish_source")
    assert not hasattr(source_publishers, "SourcePublicationRequest")
    assert not hasattr(source_publishers, "PublisherRuntime")
    for request_type in (
        source_publishers.GpuSpendSnapshotPublicationRequest,
        source_publishers.ProductionSubmissionIntentPublicationRequest,
        source_publishers.ProductionControllerBaselinePublicationRequest,
        source_publishers.ProductionControlPlaneReadinessPublicationRequest,
        source_publishers.ProductionSubmissionAcquisitionPublicationRequest,
    ):
        names = {item.name for item in fields(request_type)}
        assert "key" not in names
        assert "source_kind" not in names
        assert "record_kind" not in names
        assert "alternate_coordinate" not in names
        assert "bucket" not in names
    assert "kind" not in inspect.signature(
        source_publishers.publish_gpu_spend_snapshot
    ).parameters


def test_caller_swapped_bucket_is_unrepresentable_and_never_reaches_task4(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    from glm52_enforcement import source_publishers

    names = {
        item.name
        for item in fields(
            source_publishers.GpuSpendSnapshotPublicationRequest
        )
    }
    assert "bucket" not in names
    calls = []

    def create(*, services, request):
        calls.append((services, request))
        return _task4_result(request)

    monkeypatch.setattr(
        source_publishers, "conditional_create_immutable_json", create
    )
    cases = _publisher_cases(tmp_path_factory)
    _kind, request, entrypoint, runtime = cases[1]
    alternate_runtime = replace(
        runtime, campaign_bucket="caller-swapped-bucket"
    )
    with pytest.raises(source_publishers.SourcePublicationError):
        entrypoint(runtime=alternate_runtime, request=request)
    assert calls == []


def test_invalid_runtime_bucket_never_reaches_direct_task4(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    from glm52_enforcement import source_publishers

    calls = []

    def create(*, services, request):
        calls.append((services, request))
        return _task4_result(request)

    monkeypatch.setattr(
        source_publishers, "conditional_create_immutable_json", create
    )
    _kind, request, entrypoint, runtime = _publisher_cases(
        tmp_path_factory
    )[0]
    with pytest.raises(source_publishers.SourcePublicationError):
        entrypoint(
            runtime=replace(runtime, campaign_bucket="abc..def"),
            request=request,
        )
    assert calls == []


def test_activation_owner_identity_mismatch_stops_before_next_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    (
        source_publishers,
        services,
        request,
        recorder,
        factory,
        batch,
        calls,
    ) = _coordinator_setup(monkeypatch, tmp_path_factory)
    recorder.owner_identities = tuple(
        character * 64 for character in "12345"
    )
    with pytest.raises(source_publishers.SourcePublicationError):
        source_publishers.activate_sources_sequentially(
            services=services, request=request
        )
    assert len(calls) == 1
    assert factory.calls == []
    assert batch.calls == []


def test_boolean_durable_generation_does_not_equal_integer_activation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    (
        source_publishers,
        services,
        request,
        recorder,
        factory,
        batch,
        calls,
    ) = _coordinator_setup(monkeypatch, tmp_path_factory)
    recorder.generations = (True, None, None, None, None)
    with pytest.raises(source_publishers.SourcePublicationError):
        source_publishers.activate_sources_sequentially(
            services=services, request=request
        )
    assert len(calls) == 1
    assert len(recorder.records) == 1
    assert factory.calls == []
    assert batch.calls == []


def test_boolean_generation_is_rejected_at_request_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    (
        source_publishers,
        services,
        request,
        recorder,
        factory,
        batch,
        calls,
    ) = _coordinator_setup(monkeypatch, tmp_path_factory)
    request = replace(
        request,
        generation=True,
        gpu_spend_snapshot=replace(
            request.gpu_spend_snapshot,
            generation=True,
        ),
    )
    with pytest.raises(source_publishers.SourcePublicationError):
        source_publishers.activate_sources_sequentially(
            services=services, request=request
        )
    assert calls == []
    assert recorder.records == []
    assert factory.calls == []
    assert batch.calls == []

    (
        source_publishers,
        services,
        request,
        recorder,
        factory,
        batch,
        calls,
    ) = _coordinator_setup(monkeypatch, tmp_path_factory)
    request = replace(
        request,
        gpu_spend_snapshot=replace(
            request.gpu_spend_snapshot,
            generation=True,
        ),
    )
    with pytest.raises(source_publishers.SourcePublicationError):
        source_publishers.activate_sources_sequentially(
            services=services, request=request
        )
    assert calls == []
    assert recorder.records == []
    assert factory.calls == []
    assert batch.calls == []

    (
        source_publishers,
        services,
        request,
        recorder,
        factory,
        batch,
        calls,
    ) = _coordinator_setup(monkeypatch, tmp_path_factory)
    factory.requests = (
        factory.requests[0],
        replace(factory.requests[1], generation=True),
        *factory.requests[2:],
    )
    with pytest.raises(source_publishers.SourcePublicationError):
        source_publishers.activate_sources_sequentially(
            services=services, request=request
        )
    assert len(calls) == 1
    assert len(recorder.records) == 1
    assert len(factory.calls) == 1
    assert batch.calls == []


def test_reused_or_invalid_publisher_role_preflight_causes_zero_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    (
        source_publishers,
        services,
        request,
        recorder,
        factory,
        batch,
        calls,
    ) = _coordinator_setup(monkeypatch, tmp_path_factory)
    duplicate_runtime = replace(
        services.production_submission_intent_runtime,
        publisher_role_arn=(
            services.gpu_spend_snapshot_runtime.publisher_role_arn
        ),
    )
    services = replace(
        services,
        production_submission_intent_runtime=duplicate_runtime,
    )
    with pytest.raises(source_publishers.SourcePublicationError):
        source_publishers.activate_sources_sequentially(
            services=services, request=request
        )
    assert calls == []
    assert recorder.records == []
    assert factory.calls == []
    assert batch.calls == []

    (
        source_publishers,
        services,
        request,
        recorder,
        factory,
        batch,
        calls,
    ) = _coordinator_setup(monkeypatch, tmp_path_factory)
    invalid_runtime = replace(
        services.production_submission_acquisition_runtime,
        publisher_role_arn=(
            "arn:aws:iam::246813579024:role/not valid"
        ),
    )
    services = replace(
        services,
        production_submission_acquisition_runtime=invalid_runtime,
    )
    with pytest.raises(source_publishers.SourcePublicationError):
        source_publishers.activate_sources_sequentially(
            services=services, request=request
        )
    assert calls == []
    assert recorder.records == []
    assert factory.calls == []
    assert batch.calls == []


def test_shared_wrong_runtime_bucket_preflight_causes_zero_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    (
        source_publishers,
        services,
        request,
        recorder,
        factory,
        batch,
        calls,
    ) = _coordinator_setup(monkeypatch, tmp_path_factory)
    replacements = {
        field_name: replace(
            getattr(services, field_name),
            campaign_bucket="wrong-deployment-bucket",
        )
        for field_name in (
            "gpu_spend_snapshot_runtime",
            "production_submission_intent_runtime",
            "production_controller_baseline_runtime",
            "production_control_plane_readiness_runtime",
            "production_submission_acquisition_runtime",
        )
    }
    services = replace(services, **replacements)
    with pytest.raises(source_publishers.SourcePublicationError):
        source_publishers.activate_sources_sequentially(
            services=services, request=request
        )
    assert calls == []
    assert recorder.records == []
    assert factory.calls == []
    assert batch.calls == []


@pytest.mark.parametrize(
    "bucket",
    (
        "ab",
        "a" * 64,
        "Abc",
        "abc_def",
        ".abc",
        "abc.",
        "abc..def",
        "192.168.1.1",
        "abc.-def",
        "abc-.def",
        "xn--bucket",
        "sthree-bucket",
        "amzn-s3-demo-bucket",
        "bucket-s3alias",
        "bucket--ol-s3",
        "bucket.mrap",
        "bucket--x-s3",
        "bucket--table-s3",
        "bucket-an",
    ),
)
def test_invalid_general_purpose_bucket_authority_has_zero_side_effects(
    bucket: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    (
        source_publishers,
        services,
        request,
        recorder,
        factory,
        batch,
        calls,
    ) = _coordinator_setup(monkeypatch, tmp_path_factory)
    runtime_fields = (
        "gpu_spend_snapshot_runtime",
        "production_submission_intent_runtime",
        "production_controller_baseline_runtime",
        "production_control_plane_readiness_runtime",
        "production_submission_acquisition_runtime",
    )
    services = replace(
        services,
        **{
            field_name: replace(
                getattr(services, field_name),
                campaign_bucket=bucket,
            )
            for field_name in runtime_fields
        },
    )
    request = replace(request, expected_campaign_bucket=bucket)
    with pytest.raises(source_publishers.SourcePublicationError):
        source_publishers.activate_sources_sequentially(
            services=services, request=request
        )
    assert calls == []
    assert recorder.records == []
    assert factory.calls == []
    assert batch.calls == []


def test_distinct_runtime_closure_callers_preflight_causes_zero_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    (
        source_publishers,
        services,
        request,
        recorder,
        factory,
        batch,
        calls,
    ) = _coordinator_setup(monkeypatch, tmp_path_factory)
    runtime_fields = (
        "gpu_spend_snapshot_runtime",
        "production_submission_intent_runtime",
        "production_controller_baseline_runtime",
        "production_control_plane_readiness_runtime",
        "production_submission_acquisition_runtime",
    )
    callers = tuple(
        "arn:aws:iam::246813579024:role/"
        f"keep-glm52-closure-caller-{index}"
        for index in range(1, 6)
    )
    services = replace(
        services,
        **{
            field_name: replace(
                getattr(services, field_name),
                closure_caller_arn=caller,
            )
            for field_name, caller in zip(runtime_fields, callers)
        },
    )
    request = replace(
        request,
        gpu_spend_snapshot=replace(
            request.gpu_spend_snapshot,
            caller_arn=callers[0],
        ),
    )
    with pytest.raises(source_publishers.SourcePublicationError):
        source_publishers.activate_sources_sequentially(
            services=services, request=request
        )
    assert calls == []
    assert recorder.records == []
    assert factory.calls == []
    assert batch.calls == []


@pytest.mark.parametrize(
    "caller",
    (
        "arn:aws:iam::246813579024:role/path//closure",
        "arn:aws:iam::246813579024:role/path/",
        "arn:aws:iam::246813579024:role/",
        "arn:aws:iam::246813579024:user/closure",
        "arn:aws:sts::246813579024:assumed-role/closure//session",
        "arn:aws:sts::246813579024:assumed-role/closure/",
        "arn:aws:sts::246813579024:assumed-role/closure",
        "arn:aws:sts::246813579024:federated-user/closure",
    ),
)
def test_malformed_common_closure_caller_has_zero_side_effects(
    caller: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    (
        source_publishers,
        services,
        request,
        recorder,
        factory,
        batch,
        calls,
    ) = _coordinator_setup(monkeypatch, tmp_path_factory)
    runtime_fields = (
        "gpu_spend_snapshot_runtime",
        "production_submission_intent_runtime",
        "production_controller_baseline_runtime",
        "production_control_plane_readiness_runtime",
        "production_submission_acquisition_runtime",
    )
    services = replace(
        services,
        **{
            field_name: replace(
                getattr(services, field_name),
                closure_caller_arn=caller,
            )
            for field_name in runtime_fields
        },
    )
    request = replace(
        request,
        expected_closure_caller_arn=caller,
        gpu_spend_snapshot=replace(
            request.gpu_spend_snapshot,
            caller_arn=caller,
        ),
    )
    with pytest.raises(source_publishers.SourcePublicationError):
        source_publishers.activate_sources_sequentially(
            services=services, request=request
        )
    assert calls == []
    assert recorder.records == []
    assert factory.calls == []
    assert batch.calls == []


@pytest.mark.parametrize(
    "caller",
    (
        "arn:aws:iam::246813579024:role/keep/glm52/closure",
        (
            "arn:aws:sts::246813579024:"
            "assumed-role/keep-glm52-closure/session-1"
        ),
    ),
)
def test_stable_iam_and_assumed_role_closure_callers_are_accepted(
    caller: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    (
        source_publishers,
        services,
        request,
        _recorder,
        factory,
        _batch,
        calls,
    ) = _coordinator_setup(monkeypatch, tmp_path_factory)
    runtime_fields = (
        "gpu_spend_snapshot_runtime",
        "production_submission_intent_runtime",
        "production_controller_baseline_runtime",
        "production_control_plane_readiness_runtime",
        "production_submission_acquisition_runtime",
    )
    services = replace(
        services,
        **{
            field_name: replace(
                getattr(services, field_name),
                closure_caller_arn=caller,
            )
            for field_name in runtime_fields
        },
    )
    factory.requests = tuple(
        replace(source_request, caller_arn=caller)
        for source_request in factory.requests
    )
    request = replace(
        request,
        expected_closure_caller_arn=caller,
        gpu_spend_snapshot=factory.requests[0],
    )
    result = source_publishers.activate_sources_sequentially(
        services=services, request=request
    )
    assert result.sources_authoritative is True
    assert len(calls) == 5


def test_source_requests_cannot_cross_wire_common_closure_caller(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    alternate = (
        "arn:aws:iam::246813579024:role/"
        "keep-glm52-alternate-closure-caller"
    )
    (
        source_publishers,
        services,
        request,
        recorder,
        factory,
        batch,
        calls,
    ) = _coordinator_setup(monkeypatch, tmp_path_factory)
    request = replace(
        request,
        gpu_spend_snapshot=replace(
            request.gpu_spend_snapshot,
            caller_arn=alternate,
        ),
    )
    with pytest.raises(source_publishers.SourcePublicationError):
        source_publishers.activate_sources_sequentially(
            services=services, request=request
        )
    assert calls == []
    assert recorder.records == []
    assert factory.calls == []
    assert batch.calls == []

    (
        source_publishers,
        services,
        request,
        recorder,
        factory,
        batch,
        calls,
    ) = _coordinator_setup(monkeypatch, tmp_path_factory)
    factory.requests = (
        factory.requests[0],
        replace(factory.requests[1], caller_arn=alternate),
        *factory.requests[2:],
    )
    with pytest.raises(source_publishers.SourcePublicationError):
        source_publishers.activate_sources_sequentially(
            services=services, request=request
        )
    assert len(calls) == 1
    assert len(recorder.records) == 1
    assert len(factory.calls) == 1
    assert batch.calls == []


def test_malformed_role_paths_fail_preflight_and_valid_paths_are_accepted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    for malformed in (
        "arn:aws:iam::246813579024:role/path//name",
        "arn:aws:iam::246813579024:role/path/",
        "arn:aws:iam::246813579024:role/",
    ):
        (
            source_publishers,
            services,
            request,
            recorder,
            factory,
            batch,
            calls,
        ) = _coordinator_setup(monkeypatch, tmp_path_factory)
        services = replace(
            services,
            gpu_spend_snapshot_runtime=replace(
                services.gpu_spend_snapshot_runtime,
                publisher_role_arn=malformed,
            ),
        )
        with pytest.raises(source_publishers.SourcePublicationError):
            source_publishers.activate_sources_sequentially(
                services=services, request=request
            )
        assert calls == []
        assert recorder.records == []
        assert factory.calls == []
        assert batch.calls == []

    (
        source_publishers,
        services,
        request,
        _recorder,
        _factory,
        _batch,
        calls,
    ) = _coordinator_setup(monkeypatch, tmp_path_factory)
    runtime_fields = (
        "gpu_spend_snapshot_runtime",
        "production_submission_intent_runtime",
        "production_controller_baseline_runtime",
        "production_control_plane_readiness_runtime",
        "production_submission_acquisition_runtime",
    )
    services = replace(
        services,
        **{
            field_name: replace(
                getattr(services, field_name),
                publisher_role_arn=(
                    "arn:aws:iam::246813579024:role/"
                    f"keep/glm52/publisher-{index}"
                ),
            )
            for index, field_name in enumerate(runtime_fields, start=1)
        },
    )
    result = source_publishers.activate_sources_sequentially(
        services=services, request=request
    )
    assert result.sources_authoritative is True
    assert len(calls) == 5


def test_publishers_call_fresh_conditional_create_once_for_only_their_family(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    from glm52_enforcement import source_publishers

    calls = []

    def create(*, services, request):
        calls.append((services, request))
        return _task4_result(request)

    monkeypatch.setattr(
        source_publishers, "conditional_create_immutable_json", create
    )
    for kind, request, entrypoint, runtime in _publisher_cases(
        tmp_path_factory
    ):
        before = len(calls)
        result = entrypoint(runtime=runtime, request=request)
        assert len(calls) == before + 1
        task4_request = calls[-1][1]
        assert task4_request.candidate.record_kind == kind
        assert task4_request.candidate.raw == request.raw
        assert result.object_identity.key == task4_request.candidate.key
        assert f"/{result.object_identity.body_sha256}/" in (
            result.object_identity.key
        ) or kind == "production-submission-acquisition"
    cases = _publisher_cases(tmp_path_factory)
    before = len(calls)
    with pytest.raises(TypeError):
        cases[0][2](runtime=cases[1][3], request=cases[0][1])
    assert len(calls) == before


def test_source_result_preserves_task4_custody_without_client_capability(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    from glm52_enforcement import source_publishers

    def create(*, services, request):
        del services
        return _task4_result(
            request, provenance="all-version-reconciliation"
        )

    monkeypatch.setattr(
        source_publishers, "conditional_create_immutable_json", create
    )
    kind, request, entrypoint, runtime = _publisher_cases(tmp_path_factory)[0]
    result = entrypoint(runtime=runtime, request=request)
    assert result.source_kind == kind
    assert result.provenance == "all-version-reconciliation"
    assert result.object_identity.version_id.startswith("version-")
    assert {item.name for item in fields(result)} == {
        "source_kind",
        "publisher_role_arn",
        "object_identity",
        "candidate_identity_sha256",
        "provenance",
        "authority_audit_body_sha256",
        "closing_revision",
        "authorized_revision",
        "direct_request_id",
        "direct_server_date",
        "direct_response_authenticated",
    }
    assert result.authority_audit_body_sha256 == "a" * 64
    assert result.closing_revision == 1
    assert result.authorized_revision == 2
    assert result.direct_response_authenticated is False
    for forbidden in (
        "client",
        "services",
        "action_witness",
        "authority_audit",
        "nonce",
        "put_object",
    ):
        assert not hasattr(result, forbidden)


class Recorder:
    def __init__(self) -> None:
        self.records = []
        self.fail_at = None
        self.owner_identities = ("9" * 64,) * 5
        self.generations = (None,) * 5

    def record_source(
        self, *, activation_id, generation, publication
    ):
        from glm52_enforcement.source_publishers import DurableSourceRecord

        if self.fail_at == publication.source_kind:
            raise RuntimeError("durable record failed")
        owner_identity = self.owner_identities[len(self.records)]
        generation_override = self.generations[len(self.records)]
        record = DurableSourceRecord(
            activation_id=activation_id,
            generation=(
                generation
                if generation_override is None
                else generation_override
            ),
            source_kind=publication.source_kind,
            publisher_role_arn=publication.publisher_role_arn,
            object_identity=publication.object_identity,
            candidate_identity_sha256=publication.candidate_identity_sha256,
            publication_provenance=publication.provenance,
            activation_owner_identity_sha256=owner_identity,
            authority_audit_body_sha256=(
                publication.authority_audit_body_sha256
            ),
            closing_revision=publication.closing_revision,
            authorized_revision=publication.authorized_revision,
            direct_request_id=publication.direct_request_id,
            direct_server_date=publication.direct_server_date,
            direct_response_authenticated=(
                publication.direct_response_authenticated
            ),
        )
        self.records.append(record)
        return record


class CandidateFactory:
    def __init__(self, requests) -> None:
        self.requests = requests
        self.calls = []

    def _next(self, name, predecessors, index):
        self.calls.append((name, predecessors, len(predecessors)))
        assert tuple(predecessors) == tuple(
            self.recorder.records
        )
        return self.requests[index]

    def build_production_submission_intent(self, *, predecessors):
        return self._next("intent", predecessors, 1)

    def build_production_controller_baseline(self, *, predecessors):
        return self._next("baseline", predecessors, 2)

    def build_production_control_plane_readiness(self, *, predecessors):
        return self._next("readiness", predecessors, 3)

    def build_production_submission_acquisition(self, *, predecessors):
        return self._next("acquisition", predecessors, 4)


class Batch:
    def __init__(self) -> None:
        self.calls = []
        self.partial = False
        self.one_probe = False
        self.result_overrides = {}
        self.activated_generation = None

    def activate(self, *, request):
        from glm52_enforcement.source_publishers import (
            BatchSuccessorResult,
            StabilizationProbeSet,
        )

        self.calls.append(request)
        first = StabilizationProbeSet(
            observed_at=datetime(
                2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc
            ),
            policy_sha256="a" * 64,
            readback_sha256="b" * 64,
            denied_publisher_roles=tuple(
                source.publisher_role_arn for source in request.sources
            ),
        )
        second = StabilizationProbeSet(
            observed_at=datetime(
                2026, 7, 28, 12, 0, 10, tzinfo=timezone.utc
            ),
            policy_sha256=("c" * 64 if self.partial else "a" * 64),
            readback_sha256="b" * 64,
            denied_publisher_roles=first.denied_publisher_roles,
        )
        result = BatchSuccessorResult(
            selected_successor_count=1,
            selected_precreated_successor=True,
            family_closing_policy_applied=True,
            activated_sources=request.sources,
            denied_publisher_roles=first.denied_publisher_roles,
            probe_sets=((first,) if self.one_probe else (first, second)),
            fresh_h1f_unique_zero_child_active_head=True,
            fresh_h1f_audit_body_sha256="d" * 64,
            active_head_body_sha256="e" * 64,
            active_head_version_id="batch-head-version",
        )
        if self.activated_generation is not None:
            result = replace(
                result,
                activated_sources=tuple(
                    replace(
                        source,
                        generation=self.activated_generation,
                    )
                    for source in request.sources
                ),
            )
        return replace(result, **self.result_overrides)


def _coordinator_setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
):
    from glm52_enforcement import source_publishers

    cases = _publisher_cases(tmp_path_factory)
    requests = tuple(item[1] for item in cases)
    runtimes = tuple(item[3] for item in cases)
    expected_versions = {
        EXPECTED_KINDS[0]: json.loads(requests[1].raw)[
            "gpu_spend_snapshot_version_id"
        ],
        EXPECTED_KINDS[1]: requests[2].intent.version_id,
        EXPECTED_KINDS[2]: requests[3].controller_baseline.version_id,
        EXPECTED_KINDS[3]: (
            requests[4].must_start_control_plane_ready.version_id
        ),
        EXPECTED_KINDS[4]: "production-acquisition-version-1",
    }
    calls = []
    active = False

    def create(*, services, request):
        nonlocal active
        assert not active
        active = True
        calls.append(request)
        result = _task4_result(request)
        result = result.__class__(
            outcome=result.outcome,
            object_identity=result.object_identity.__class__(
                **{
                    **result.object_identity.__dict__,
                    "version_id": expected_versions[
                        request.candidate.record_kind
                    ],
                }
            ),
            authority_audit_body_sha256=result.authority_audit_body_sha256,
            closing_revision=result.closing_revision,
            authorized_revision=result.authorized_revision,
            provenance=result.provenance,
            direct_request_id=result.direct_request_id,
            direct_server_date=result.direct_server_date,
            direct_response_authenticated=(
                result.direct_response_authenticated
            ),
        )
        active = False
        return result

    monkeypatch.setattr(
        source_publishers, "conditional_create_immutable_json", create
    )
    recorder = Recorder()
    factory = CandidateFactory(requests)
    factory.recorder = recorder
    batch = Batch()
    services = source_publishers.SequentialSourceServices(
        gpu_spend_snapshot_runtime=runtimes[0],
        production_submission_intent_runtime=runtimes[1],
        production_controller_baseline_runtime=runtimes[2],
        production_control_plane_readiness_runtime=runtimes[3],
        production_submission_acquisition_runtime=runtimes[4],
        durable_recorder=recorder,
        candidate_factory=factory,
        batch_successor=batch,
    )
    request = source_publishers.SequentialSourceActivationRequest(
        activation_id=ACTIVATION_ID,
        generation=1,
        activation_owner_identity_sha256="9" * 64,
        expected_campaign_bucket=str(json.loads(requests[1].raw)["bucket"]),
        expected_closure_caller_arn=CALLER,
        gpu_spend_snapshot=requests[0],
    )
    return source_publishers, services, request, recorder, factory, batch, calls


def test_coordinator_invokes_exact_five_source_order_without_overlap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    (
        source_publishers,
        services,
        request,
        recorder,
        factory,
        batch,
        calls,
    ) = _coordinator_setup(monkeypatch, tmp_path_factory)
    result = source_publishers.activate_sources_sequentially(
        services=services, request=request
    )
    assert tuple(
        item.candidate.record_kind for item in calls
    ) == EXPECTED_KINDS
    assert tuple(item.source_kind for item in recorder.records) == EXPECTED_KINDS
    assert [item[2] for item in factory.calls] == [1, 2, 3, 4]
    assert len(batch.calls) == 1
    assert result.sources_authoritative is True


def test_each_next_candidate_binds_all_required_service_assigned_predecessors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    (
        source_publishers,
        services,
        request,
        recorder,
        _factory,
        _batch,
        _calls,
    ) = _coordinator_setup(monkeypatch, tmp_path_factory)
    result = source_publishers.activate_sources_sequentially(
        services=services, request=request
    )
    assert len(result.durable_sources) == 5
    for index, durable in enumerate(result.durable_sources):
        assert durable.object_identity.version_id
        if index:
            assert result.durable_sources[index - 1] in recorder.records

    (
        source_publishers,
        services,
        request,
        recorder,
        factory,
        batch,
        calls,
    ) = _coordinator_setup(monkeypatch, tmp_path_factory)
    intent_request = factory.requests[1]
    intent = json.loads(intent_request.raw)
    intent["gpu_spend_snapshot_version_id"] = "precomputed-version"
    body = dict(intent)
    body.pop("intent_body_sha256")
    intent["intent_body_sha256"] = hashlib.sha256(
        _canonical(body)
    ).hexdigest()
    factory.requests = (
        factory.requests[0],
        replace(intent_request, raw=_canonical(intent) + b"\n"),
        *factory.requests[2:],
    )
    with pytest.raises(source_publishers.SourcePublicationError):
        source_publishers.activate_sources_sequentially(
            services=services, request=request
        )
    assert len(recorder.records) == 1
    assert len(calls) == 1
    assert batch.calls == []


def test_reconciled_predecessor_requires_durable_activation_record_before_next(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    (
        source_publishers,
        services,
        request,
        recorder,
        factory,
        batch,
        _calls,
    ) = _coordinator_setup(monkeypatch, tmp_path_factory)
    recorder.fail_at = EXPECTED_KINDS[0]
    with pytest.raises(source_publishers.SourcePublicationError):
        source_publishers.activate_sources_sequentially(
            services=services, request=request
        )
    assert factory.calls == []
    assert batch.calls == []


def test_ambiguous_zero_history_sibling_delete_marker_or_mismatch_stops_sequence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    for failure in (
        "absent-after-reconciliation",
        "history",
        "sibling",
        "delete-marker",
        "mismatch",
    ):
        (
            source_publishers,
            services,
            request,
            recorder,
            factory,
            batch,
            _calls,
        ) = _coordinator_setup(monkeypatch, tmp_path_factory)

        def ambiguous(*, services, request):
            del services
            result = _task4_result(request)
            identity = result.object_identity
            outcome = failure
            if failure == "mismatch":
                outcome = "reconciled-existing"
                identity = identity.__class__(
                    **{
                        **identity.__dict__,
                        "key": identity.key + ".mismatch",
                    }
                )
            else:
                identity = None
            return result.__class__(
                outcome=outcome,
                object_identity=identity,
                authority_audit_body_sha256=(
                    result.authority_audit_body_sha256
                ),
                closing_revision=result.closing_revision,
                authorized_revision=result.authorized_revision,
                provenance="all-version-reconciliation",
            )

        monkeypatch.setattr(
            source_publishers,
            "conditional_create_immutable_json",
            ambiguous,
        )
        with pytest.raises(source_publishers.SourcePublicationError):
            source_publishers.activate_sources_sequentially(
                services=services, request=request
            )
        assert recorder.records == []
        assert factory.calls == []
        assert batch.calls == []


def test_batch_successor_contains_all_five_exact_identities_and_publisher_roles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    (
        source_publishers,
        services,
        request,
        _recorder,
        _factory,
        batch,
        _calls,
    ) = _coordinator_setup(monkeypatch, tmp_path_factory)
    result = source_publishers.activate_sources_sequentially(
        services=services, request=request
    )
    batch_request = batch.calls[0]
    assert batch_request.sources == result.durable_sources
    assert batch_request.activation_owner_identity_sha256 == "9" * 64
    assert batch_request.expected_campaign_bucket == (
        request.expected_campaign_bucket
    )
    assert batch_request.expected_closure_caller_arn == CALLER
    assert all(
        source.activation_owner_identity_sha256
        == batch_request.activation_owner_identity_sha256
        for source in batch_request.sources
    )
    assert tuple(source.source_kind for source in batch_request.sources) == (
        EXPECTED_KINDS
    )
    assert len(
        {source.publisher_role_arn for source in batch_request.sources}
    ) == 5
    for source in batch_request.sources:
        assert source.object_identity.key
        assert source.object_identity.version_id
        assert source.object_identity.file_sha256
        assert source.object_identity.body_sha256
        assert source.candidate_identity_sha256


def test_no_source_is_authoritative_before_batch_successor_stabilizes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    (
        source_publishers,
        services,
        request,
        recorder,
        _factory,
        batch,
        _calls,
    ) = _coordinator_setup(monkeypatch, tmp_path_factory)
    for item in recorder.records:
        assert not hasattr(item, "authoritative")
    result = source_publishers.activate_sources_sequentially(
        services=services, request=request
    )
    assert len(batch.calls) == 1
    assert result.sources_authoritative is True
    assert all(
        not hasattr(source, "authoritative")
        for source in result.durable_sources
    )


def test_partial_policy_or_one_probe_set_cannot_activate_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    (
        source_publishers,
        services,
        request,
        _recorder,
        _factory,
        batch,
        _calls,
    ) = _coordinator_setup(monkeypatch, tmp_path_factory)
    batch.partial = True
    result = source_publishers.activate_sources_sequentially(
        services=services, request=request
    )
    assert result.sources_authoritative is False

    (
        source_publishers,
        services,
        request,
        _recorder,
        _factory,
        batch,
        _calls,
    ) = _coordinator_setup(monkeypatch, tmp_path_factory)
    batch.one_probe = True
    result = source_publishers.activate_sources_sequentially(
        services=services, request=request
    )
    assert result.sources_authoritative is False


@pytest.mark.parametrize(
    ("field_name", "wrong_type_value"),
    (
        ("selected_successor_count", True),
        ("selected_precreated_successor", 1),
        ("family_closing_policy_applied", 1),
        ("fresh_h1f_unique_zero_child_active_head", 1),
    ),
)
def test_batch_proof_int_and_bool_fields_require_exact_types(
    field_name: str,
    wrong_type_value: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    (
        source_publishers,
        services,
        request,
        _recorder,
        _factory,
        batch,
        _calls,
    ) = _coordinator_setup(monkeypatch, tmp_path_factory)
    batch.result_overrides = {field_name: wrong_type_value}
    result = source_publishers.activate_sources_sequentially(
        services=services, request=request
    )
    assert result.sources_authoritative is False


def test_batch_activated_source_generation_requires_exact_integer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    (
        source_publishers,
        services,
        request,
        _recorder,
        _factory,
        batch,
        _calls,
    ) = _coordinator_setup(monkeypatch, tmp_path_factory)
    batch.activated_generation = True
    result = source_publishers.activate_sources_sequentially(
        services=services, request=request
    )
    assert result.sources_authoritative is False
