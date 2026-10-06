from __future__ import annotations

import importlib.util
import base64
import hashlib
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from glm52_enforcement.canonical import canonical_sha256
from glm52_enforcement.records import canonical_record_identity
from glm52_enforcement.task12_operations import (
    OPERATION_SOURCE_PRODUCERS,
)


_SOURCE_TYPES = {
    "finalization_control": "glm52_production_finalization_control",
    "support_finalized": "glm52_production_support_plane_finalized",
    "snapshot_cleanup_control": (
        "glm52_production_snapshot_cleanup_control"
    ),
    "snapshot_cleanup_action": "glm52_production_snapshot_cleanup_action",
    "snapshot_cleanup_transition": (
        "glm52_production_snapshot_cleanup_transition"
    ),
}


def _record_fixtures() -> object:
    spec = importlib.util.spec_from_file_location(
        "_task12_snapshot_finalization_record_fixtures",
        Path(__file__).with_name("test_glm52_enforcement_dynamodb.py"),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _invocation(operation: str) -> object:
    return SimpleNamespace(
        operation_kind=operation,
        activation_id="activation-1",
        activation_ordinal=1,
        generation=1,
        generation_text="00000001",
        dispatch_identity_sha256="d" * 64,
        invoked_function_version_arn=(
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-task12:1"
        ),
        caller_state_machine_version_arn=(
            "arn:aws:states:us-west-2:246813579024:"
            "stateMachine:keep-glm52-h1g-retainedlifecycle:1"
        ),
        state_machine_execution_arn=(
            "arn:aws:states:us-west-2:246813579024:execution:"
            "keep-glm52-h1g-retainedlifecycle:execution-1"
        ),
        operation_input={},
    )


def _rehash(value: dict[str, object]) -> dict[str, object]:
    body = dict(value)
    body.pop("canonical_body_sha256")
    return {**body, "canonical_body_sha256": canonical_sha256(body)}


def _source(record_type: str, **overrides: object) -> dict[str, object]:
    return _record_fixtures()._closed_record(
        record_type,
        activation_id="activation-1",
        **overrides,
    )


def _sources_for(operation: str) -> dict[str, dict[str, object]]:
    return {
        alias: _source(_SOURCE_TYPES[alias])
        for alias, _producer in OPERATION_SOURCE_PRODUCERS[operation]
    }


class _NoEffects:
    def __init__(self) -> None:
        self.calls = 0

    def client(self, _service: str) -> object:
        self.calls += 1
        raise AssertionError("no live boundary may run")


class _Ec2:
    def __init__(self, snapshot: dict[str, object]) -> None:
        self.snapshot = snapshot
        self.calls: list[dict[str, object]] = []

    def describe_snapshots(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        return {
            "Snapshots": [self.snapshot],
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "describe-1",
                "RetryAttempts": 0,
            },
        }


class _Ec2Ports:
    def __init__(self, ec2: _Ec2) -> None:
        self.ec2 = ec2

    def client(self, service: str) -> object:
        assert service == "ec2"
        return self.ec2


def _armed_control() -> tuple[dict[str, object], dict[str, object]]:
    from glm52_enforcement.task12_snapshot_cleanup import SnapshotObservation

    snapshot = {
        "SnapshotId": "snap-0123456789abcdef0",
        "VolumeId": "vol-0123456789abcdef0",
        "KmsKeyId": (
            "arn:aws:kms:us-west-2:246813579024:key/"
            "12345678-1234-1234-1234-123456789012"
        ),
        "Encrypted": True,
        "State": "completed",
        "StartTime": datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc),
        "Tags": [{"Key": "RunId", "Value": "glm52-sky-20260724"}],
    }
    tag_identity = canonical_sha256(snapshot["Tags"])
    observation = SnapshotObservation(
        snapshot_id=snapshot["SnapshotId"],
        source_volume_id=snapshot["VolumeId"],
        kms_key_arn=snapshot["KmsKeyId"],
        snapshot_tags_sha256=tag_identity,
        encrypted=True,
        state="completed",
        observed_at="2026-07-28T12:00:00Z",
        describe_request_id="describe-1",
        describe_response_sha256="d" * 64,
    )
    control = _source(
        "glm52_production_snapshot_cleanup_control",
        state="ARMED",
        snapshot_id=observation.snapshot_id,
        snapshot_identity_sha256=observation.snapshot_identity_sha256,
        source_volume_id=observation.source_volume_id,
        snapshot_tags_sha256=observation.snapshot_tags_sha256,
        delete_not_before="2026-08-04T12:00:00Z",
        schedule_arn=(
            "arn:aws:scheduler:us-west-2:246813579024:"
            "schedule/default/keep-glm52-h1g-snapshot-cleanup-activation-1"
        ),
        schedule_input_sha256=canonical_sha256({}),
    )
    return control, snapshot


def _send_sources() -> dict[str, dict[str, object]]:
    armed, _snapshot = _armed_control()
    control = _source(
        "glm52_production_snapshot_cleanup_control",
        state="DELETE_POSSIBLY_SENT",
        revision=5,
        snapshot_id=armed["snapshot_id"],
        snapshot_identity_sha256=armed["snapshot_identity_sha256"],
        source_volume_id=armed["source_volume_id"],
        snapshot_tags_sha256=armed["snapshot_tags_sha256"],
        delete_not_before=armed["delete_not_before"],
        schedule_arn=armed["schedule_arn"],
        schedule_input_sha256=armed["schedule_input_sha256"],
        delete_logical_attempt=1,
        delete_call_count=1,
    )
    transition = _source(
        "glm52_production_snapshot_cleanup_transition",
        from_state="OWNED",
        to_state="DELETE_POSSIBLY_SENT",
        from_revision=4,
        to_revision=5,
        owner_attempt=control["owner_attempt"],
        owner_execution_arn=control["owner_execution_arn"],
        owner_state_machine_version_arn=control[
            "owner_state_machine_version_arn"
        ],
        owner_dispatch_identity_sha256=control[
            "owner_dispatch_identity_sha256"
        ],
        owner_invocation_nonce_sha256=control[
            "owner_invocation_nonce_sha256"
        ],
        action_identity_sha256=control[
            "latest_delete_action_identity_sha256"
        ],
    )
    return {
        "snapshot_cleanup_control": control,
        "snapshot_cleanup_transition": transition,
    }


def test_direct_requests_use_exact_current_canonical_sources() -> None:
    from glm52_enforcement.cloudformation_stacks import StackKind, stack_name
    from glm52_enforcement.task12_live_snapshot_finalization import (
        materialize_live_request,
    )

    operation = "RETAINED_RECONCILE_DELETE_UNTIL_ABSENT"
    no_effects = _NoEffects()
    delete_request = materialize_live_request(
        operation_kind=operation,
        invocation=_invocation(operation),
        live_sources=_sources_for(operation),
        ports=no_effects,
    )
    assert delete_request == {"stack_id": stack_name(StackKind.SUPPORT)}
    assert no_effects.calls == 0

    operation = "SNAPSHOT_CLEANUP_AUDIT_DELETE_AUTHORITY"
    action = _source("glm52_production_snapshot_cleanup_action")
    audit_request = materialize_live_request(
        operation_kind=operation,
        invocation=_invocation(operation),
        live_sources={"snapshot_cleanup_action": action},
        ports=no_effects,
    )
    assert audit_request["sort_key"].endswith(
        "#SNAPSHOT_CLEANUP_ACTION#00000001"
    )
    assert audit_request["expected_record_identity_sha256"] == (
        canonical_record_identity(
            "glm52_production_snapshot_cleanup_action", action
        )
    )

    operation = "SNAPSHOT_CLEANUP_SEND_SAME_ID_OR_READ_BACK"
    send_request = materialize_live_request(
        operation_kind=operation,
        invocation=_invocation(operation),
        live_sources=_send_sources(),
        ports=no_effects,
    )
    assert send_request == {"snapshot_id": "snap-0123456789abcdef0"}
    assert no_effects.calls == 0


def test_schedule_validation_rebuilds_capture_from_live_ec2() -> None:
    from glm52_enforcement.task12_live_snapshot_finalization import (
        materialize_live_request,
    )

    operation = "SNAPSHOT_CLEANUP_VALIDATE_SCHEDULE_AND_DEADLINE"
    control, snapshot = _armed_control()
    ec2 = _Ec2(snapshot)
    request = materialize_live_request(
        operation_kind=operation,
        invocation=_invocation(operation),
        live_sources={"snapshot_cleanup_control": control},
        ports=_Ec2Ports(ec2),
    )

    assert request["capture"]["snapshot_identity_sha256"] == control[
        "snapshot_identity_sha256"
    ]
    assert request["schedule"]["schedule_arn"] == control["schedule_arn"]
    assert ec2.calls == [{"SnapshotIds": [control["snapshot_id"]]}]


def test_schedule_validation_rejects_naive_snapshot_start_time() -> None:
    from glm52_enforcement.task12_live_snapshot_finalization import (
        materialize_live_request,
    )

    operation = "SNAPSHOT_CLEANUP_VALIDATE_SCHEDULE_AND_DEADLINE"
    control, snapshot = _armed_control()
    snapshot["StartTime"] = datetime(2026, 7, 28, 12, 0)
    ec2 = _Ec2(snapshot)

    with pytest.raises(ValueError, match="start time is not normalized"):
        materialize_live_request(
            operation_kind=operation,
            invocation=_invocation(operation),
            live_sources={"snapshot_cleanup_control": control},
            ports=_Ec2Ports(ec2),
        )
    assert ec2.calls == [{"SnapshotIds": [control["snapshot_id"]]}]


def test_schedule_validation_normalizes_aware_start_time_to_utc() -> None:
    from glm52_enforcement.task12_live_snapshot_finalization import (
        materialize_live_request,
    )

    operation = "SNAPSHOT_CLEANUP_VALIDATE_SCHEDULE_AND_DEADLINE"
    control, snapshot = _armed_control()
    snapshot["StartTime"] = datetime(
        2026,
        7,
        28,
        17,
        0,
        tzinfo=timezone(timedelta(hours=5)),
    )
    ec2 = _Ec2(snapshot)

    request = materialize_live_request(
        operation_kind=operation,
        invocation=_invocation(operation),
        live_sources={"snapshot_cleanup_control": control},
        ports=_Ec2Ports(ec2),
    )

    assert request["capture"]["observed_at"] == "2026-07-28T12:00:00Z"
    assert request["schedule"]["delete_not_before"] == "2026-08-04T12:00:00Z"
    assert ec2.calls == [{"SnapshotIds": [control["snapshot_id"]]}]


def test_stale_transition_is_rejected_before_snapshot_effect() -> None:
    from glm52_enforcement.task12_live_snapshot_finalization import (
        materialize_live_request,
    )

    operation = "SNAPSHOT_CLEANUP_SEND_SAME_ID_OR_READ_BACK"
    sources = _send_sources()
    sources["snapshot_cleanup_transition"] = _rehash(
        {
            **sources["snapshot_cleanup_transition"],
            "to_revision": sources["snapshot_cleanup_control"]["revision"] - 1,
        }
    )
    ports = _NoEffects()

    with pytest.raises(ValueError, match="causal source lineage drifted"):
        materialize_live_request(
            operation_kind=operation,
            invocation=_invocation(operation),
            live_sources=sources,
            ports=ports,
        )
    assert ports.calls == 0


def test_production_adapter_dispatches_through_snapshot_live_builder() -> None:
    from glm52_enforcement.task12_lambda_adapters import (
        _materialize_live_operation_request,
    )

    operation = "SNAPSHOT_CLEANUP_SEND_SAME_ID_OR_READ_BACK"
    sources = _send_sources()
    sources["snapshot_cleanup_transition"] = _rehash(
        {
            **sources["snapshot_cleanup_transition"],
            "to_revision": sources["snapshot_cleanup_control"]["revision"] - 1,
        }
    )
    ports = _NoEffects()

    with pytest.raises(
        ValueError,
        match="live materializer rejected canonical sources",
    ) as caught:
        _materialize_live_operation_request(
            ports=ports,
            invocation=_invocation(operation),
            live_sources=sources,
        )
    assert caught.value.__cause__ is not None
    assert "causal source lineage drifted" in str(caught.value.__cause__)
    assert ports.calls == 0


def test_audit_request_uses_shared_closed_request_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from glm52_enforcement import task12_live_snapshot_finalization as live

    operation = "SNAPSHOT_CLEANUP_AUDIT_DELETE_AUTHORITY"
    ports = _NoEffects()
    monkeypatch.setattr(live, "ledger_pk", lambda _run_id: " RUN#invalid")

    with pytest.raises(ValueError, match="partition_key.*not typed"):
        live.materialize_live_request(
            operation_kind=operation,
            invocation=_invocation(operation),
            live_sources={
                "snapshot_cleanup_action": _source(
                    "glm52_production_snapshot_cleanup_action"
                )
            },
            ports=ports,
        )
    assert ports.calls == 0


def test_preseeded_or_extra_source_is_rejected_before_missing_producer() -> None:
    from glm52_enforcement.task12_live_snapshot_finalization import (
        materialize_live_request,
    )

    operation = "RETAINED_PUBLISH_SUPPORT_PLANE_FINALIZED"
    sources = _sources_for(operation)
    sources["operation_projection"] = {
        "payload": {"future": "request"},
        "canonical_body_sha256": "a" * 64,
    }
    ports = _NoEffects()

    with pytest.raises(ValueError, match="live source set drifted"):
        materialize_live_request(
            operation_kind=operation,
            invocation=_invocation(operation),
            live_sources=sources,
            ports=ports,
        )
    assert ports.calls == 0


class _WriterPorts:
    def __init__(
        self, campaign_bucket: str = "keep-glm52-campaign"
    ) -> None:
        self.calls = 0
        self.deployment = SimpleNamespace(
            role_coordinates={"campaign_bucket": campaign_bucket}
        )

    def client(self, _service: str) -> object:
        self.calls += 1
        raise AssertionError("writer request materialization is effect-free")


def test_production_dispatch_builds_candidate_bound_support_writer() -> None:
    from glm52_enforcement.task12_lambda_adapters import (
        _materialize_live_operation_request,
    )
    from glm52_enforcement.task12_writers import (
        RetainedWriterActionAuthority,
        RetainedWriterAuditAuthority,
        build_retained_writer_candidate,
        validate_retained_writer_authority,
    )
    operation = "RETAINED_PUBLISH_SUPPORT_PLANE_FINALIZED"
    invocation = _invocation(operation)
    control = _source(
        "glm52_production_finalization_control",
        state="OWNED",
        owner_invocation_nonce_sha256=hashlib.sha256(b"n" * 32).hexdigest(),
    )
    ports = _WriterPorts()

    request = _materialize_live_operation_request(
        ports=ports,
        invocation=invocation,
        live_sources={"finalization_control": control},
    )

    candidate = build_retained_writer_candidate(
        writer_kind="SupportPlaneFinalized",
        campaign_bucket="keep-glm52-campaign",
        activation_id="activation-1",
        generation=1,
        authority_domain="FINALIZATION",
        record=request["record"],
    )
    validate_retained_writer_authority(
        candidate=candidate,
        action=RetainedWriterActionAuthority(**request["action"]),
        audit=RetainedWriterAuditAuthority(**request["audit"]),
    )
    assert request["record"]["writer_dispatch_identity_sha256"] == "d" * 64
    assert request["record"]["writer_invocation_nonce_sha256"] == (
        control["owner_invocation_nonce_sha256"]
    )
    assert ports.calls == 0


def test_swapped_finalization_source_fails_before_writer_boundary() -> None:
    from glm52_enforcement.task12_live_snapshot_finalization import (
        materialize_live_request,
    )

    operation = "RETAINED_PUBLISH_SUPPORT_PLANE_FINALIZED"
    invocation = _invocation(operation)
    control = {
        **_source(
            "glm52_production_finalization_control",
            state="OWNED",
        ),
        "activation_id": "activation-foreign",
    }
    ports = _WriterPorts()

    with pytest.raises(ValueError, match="activation source drifted"):
        materialize_live_request(
            operation_kind=operation,
            invocation=invocation,
            live_sources={"finalization_control": control},
            ports=ports,
        )
    assert ports.calls == 0


def test_writer_successor_rejects_non_effect_before_dynamodb() -> None:
    from glm52_enforcement.task12_live_snapshot_finalization import (
        persist_live_successors,
    )

    operation = "RETAINED_PUBLISH_SUPPORT_PLANE_FINALIZED"
    invocation = _invocation(operation)
    control = _source(
        "glm52_production_finalization_control",
        state="OWNED",
        owner_invocation_nonce_sha256=hashlib.sha256(b"n" * 32).hexdigest(),
    )
    ports = _WriterPorts()
    request = {
        "writer_kind": "SupportPlaneFinalized",
        "authority_domain": "FINALIZATION",
        "record": {},
        "action": {},
        "audit": {},
    }

    with pytest.raises(ValueError, match="requires the exact domain effect"):
        persist_live_successors(
            operation_kind=operation,
            invocation=invocation,
            live_sources={"finalization_control": control},
            request=request,
            domain_result={"projected": "success"},
            ports=ports,
        )
    assert ports.calls == 0


class _DiscoveryCloudFormation:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def list_stack_resources(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        return {
            "StackResourceSummaries": [
                {
                    "LogicalResourceId": "CombinedHostDataVolume",
                    "PhysicalResourceId": "vol-0123456789abcdef0",
                    "ResourceType": "AWS::EC2::Volume",
                }
            ],
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "cfn-list-1",
                "RetryAttempts": 0,
            },
        }


class _DiscoveryPorts:
    def __init__(self, snapshot: dict[str, object]) -> None:
        self.cloudformation = _DiscoveryCloudFormation()
        self.ec2 = _Ec2(snapshot)
        self.deployment = SimpleNamespace(
            role_coordinates={
                "kms_key_id": snapshot["KmsKeyId"],
                "ledger_table_name": "keep-glm52-ledger",
                "support_stack_id": (
                    "arn:aws:cloudformation:us-west-2:246813579024:"
                    "stack/keep-glm52-h1g-support/"
                    "12345678-1234-1234-1234-123456789012"
                ),
                "snapshot_cleanup_schedule_group_name": "default",
                "snapshot_cleanup_schedule_name": (
                    "keep-glm52-h1g-snapshot-cleanup-activation-1"
                ),
            }
        )

    def client(self, service: str) -> object:
        return {
            "cloudformation": self.cloudformation,
            "ec2": self.ec2,
        }[service]


def test_production_dispatch_discovers_snapshot_from_current_aws() -> None:
    from glm52_enforcement.task12_lambda_adapters import (
        _materialize_live_operation_request,
    )

    operation = "RETAINED_DISCOVER_EXACT_RECOVERY_SNAPSHOT"
    tags = [
        {"Key": "ActivationId", "Value": "activation-1"},
        {"Key": "Authority", "Value": "H1g"},
        {"Key": "Campaign", "Value": "GLM-5.2"},
        {"Key": "ManagedBy", "Value": "CloudFormation"},
        {"Key": "Project", "Value": "KEEP"},
        {"Key": "Purpose", "Value": "forensic-data-volume"},
        {"Key": "RunId", "Value": "glm52-sky-20260724"},
    ]
    snapshot = {
        "SnapshotId": "snap-0123456789abcdef0",
        "VolumeId": "vol-0123456789abcdef0",
        "KmsKeyId": (
            "arn:aws:kms:us-west-2:246813579024:key/"
            "12345678-1234-1234-1234-123456789012"
        ),
        "Encrypted": True,
        "State": "completed",
        "StartTime": datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc),
        "Tags": tags,
    }
    ports = _DiscoveryPorts(snapshot)

    request = _materialize_live_operation_request(
        ports=ports,
        invocation=_invocation(operation),
        live_sources={"support_finalized": _source(
            "glm52_production_support_plane_finalized"
        )},
    )

    assert request["expected_source_volume_id"] == snapshot["VolumeId"]
    assert request["expected_kms_key_arn"] == snapshot["KmsKeyId"]
    assert request["expected_snapshot_tags_sha256"] == canonical_sha256(tags)
    assert ports.cloudformation.calls == [
        {
            "StackName": (
                "arn:aws:cloudformation:us-west-2:246813579024:"
                "stack/keep-glm52-h1g-support/"
                "12345678-1234-1234-1234-123456789012"
            )
        }
    ]
    assert ports.ec2.calls == [
        {
            "OwnerIds": ["self"],
            "Filters": [
                {
                    "Name": "volume-id",
                    "Values": [snapshot["VolumeId"]],
                }
            ],
        }
    ]


def test_snapshot_arm_builds_exact_current_transition_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from glm52_enforcement import task12_live_snapshot_finalization as live
    from glm52_enforcement.task12_lambda_adapters import (
        _materialize_live_operation_request,
    )
    from glm52_enforcement.task12_snapshot_cleanup import (
        SnapshotCapture,
        SnapshotObservation,
    )

    operation = "RETAINED_ARM_SNAPSHOT_CLEANUP_CONTROL_AND_SCHEDULE"
    support = _source("glm52_production_support_plane_finalized")
    raw_nonce = b"n" * 32
    finalization = _source(
        "glm52_production_finalization_control",
        state="SUPPORT_FINALIZED",
        owner_invocation_nonce_sha256=hashlib.sha256(raw_nonce).hexdigest(),
        support_plane_finalized_identity_sha256=canonical_record_identity(
            "glm52_production_support_plane_finalized", support
        ),
    )
    observation = SnapshotObservation(
        snapshot_id="snap-0123456789abcdef0",
        source_volume_id="vol-0123456789abcdef0",
        kms_key_arn=(
            "arn:aws:kms:us-west-2:246813579024:key/"
            "12345678-1234-1234-1234-123456789012"
        ),
        snapshot_tags_sha256="7" * 64,
        encrypted=True,
        state="completed",
        observed_at="2026-07-29T12:00:00Z",
        describe_request_id="describe-1",
        describe_response_sha256="8" * 64,
    )
    capture = SnapshotCapture.discover(
        observations=(observation,),
        expected_source_volume_id=observation.source_volume_id,
        expected_kms_key_arn=observation.kms_key_arn,
        expected_snapshot_tags_sha256=observation.snapshot_tags_sha256,
    )
    recovery = _source(
        "glm52_production_recovery_control",
        state="RECOVERY_COMPLETE",
        cleanup_control_root_identity_sha256="2" * 64,
        cleanup_transition_chain_head_sha256="3" * 64,
        cleanup_transition_chain_length=1,
    )
    cleanup = _source("glm52_production_snapshot_cleanup_control")
    records = {
        "glm52_production_activation_index": (
            _record_fixtures()._activation_index()
        ),
        "glm52_production_recovery_control": recovery,
        "glm52_production_snapshot_cleanup_control": cleanup,
        "glm52_production_finalization_control": finalization,
    }
    monkeypatch.setattr(
        live,
        "_retained_read_snapshot_effect",
        lambda **_kwargs: capture,
    )
    monkeypatch.setattr(
        live,
        "_retained_read_record",
        lambda **kwargs: records[kwargs["record_type"]],
    )
    ports = _DiscoveryPorts(
        {
            "KmsKeyId": observation.kms_key_arn,
        }
    )
    capsule_body = {
        "schema_version": 1,
        "record_type": "glm52_task12_owner_nonce_capsule_v1",
        "kms_key_id": observation.kms_key_arn,
        "encryption_context": {
            key: value
            for key, value in sorted(
                {
                    "account_id": finalization["account_id"],
                    "region": finalization["region"],
                    "run_id": finalization["run_id"],
                    "activation_id": finalization["activation_id"],
                    "authority_domain": "FINALIZATION",
                    "owner_execution_arn": finalization[
                        "owner_execution_arn"
                    ],
                    "owner_state_machine_version_arn": finalization[
                        "owner_state_machine_version_arn"
                    ],
                    "owner_attempt": str(finalization["owner_attempt"]),
                    "barrier_nonce_sha256": finalization[
                        "finalization_barrier_nonce_sha256"
                    ],
                    "control_revision": str(finalization["revision"]),
                    "owner_hard_expires_at": finalization[
                        "owner_hard_expires_at"
                    ],
                }.items()
            )
        },
        "ciphertext_base64": "Y2lwaGVydGV4dA==",
        "nonce_sha256": finalization[
            "owner_invocation_nonce_sha256"
        ],
    }
    capsule = {
        **capsule_body,
        "canonical_body_sha256": canonical_sha256(capsule_body),
    }
    invocation = _invocation(operation)
    invocation.operation_input = {
        "task12_last_result": {
            "result": {"owner_nonce_capsule": capsule}
        }
    }

    request = _materialize_live_operation_request(
        ports=ports,
        invocation=invocation,
        live_sources={
            "finalization_control": finalization,
            "support_finalized": support,
        },
    )

    assert request["capture"]["snapshot_identity_sha256"] == (
        capture.snapshot_identity_sha256
    )
    assert request["plan"]["cleanup_control"]["before"] == cleanup
    assert request["plan"]["cleanup_control"]["after"]["state"] == "ARMED"
    assert request["plan"]["cleanup_transition"]["item"]["to_state"] == "ARMED"
    assert request["owner_nonce_capsule"] == capsule


def test_snapshot_discovery_successor_requires_exact_effect_before_write() -> None:
    from glm52_enforcement.task12_live_snapshot_finalization import (
        Task12LiveSnapshotFinalizationError,
        persist_live_successors,
    )

    ports = _NoEffects()
    operation = "RETAINED_DISCOVER_EXACT_RECOVERY_SNAPSHOT"

    with pytest.raises(
        Task12LiveSnapshotFinalizationError,
        match="snapshot successor requires the exact domain effect",
    ):
        persist_live_successors(
            operation_kind=operation,
            invocation=_invocation(operation),
            live_sources={
                "support_finalized": _source(
                    "glm52_production_support_plane_finalized"
                )
            },
            request={},
            domain_result={"snapshot_id": "projected"},
            ports=ports,
        )

    assert ports.calls == 0


def test_snapshot_discovery_successor_persists_only_actual_capture() -> None:
    from glm52_enforcement.dynamodb import decode_item
    from glm52_enforcement.task12_live_snapshot_finalization import (
        persist_live_successors,
    )
    from glm52_enforcement.task12_snapshot_cleanup import (
        SnapshotCapture,
        SnapshotObservation,
    )

    observation = SnapshotObservation(
        snapshot_id="snap-0123456789abcdef0",
        source_volume_id="vol-0123456789abcdef0",
        kms_key_arn=(
            "arn:aws:kms:us-west-2:246813579024:key/"
            "12345678-1234-1234-1234-123456789012"
        ),
        snapshot_tags_sha256="7" * 64,
        encrypted=True,
        state="completed",
        observed_at="2026-07-29T12:00:00Z",
        describe_request_id="describe-1",
        describe_response_sha256="8" * 64,
    )
    capture = SnapshotCapture.discover(
        observations=(observation,),
        expected_source_volume_id=observation.source_volume_id,
        expected_kms_key_arn=observation.kms_key_arn,
        expected_snapshot_tags_sha256=observation.snapshot_tags_sha256,
    )
    support = _source("glm52_production_support_plane_finalized")

    class _Dynamo:
        def __init__(self) -> None:
            self.requests: list[dict[str, object]] = []

        def put_item(self, **kwargs: object) -> dict[str, object]:
            self.requests.append(dict(kwargs))
            return {
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "put-snapshot-effect-1",
                    "RetryAttempts": 0,
                }
            }

    dynamo = _Dynamo()
    ports = SimpleNamespace(
        deployment=SimpleNamespace(
            role_coordinates={"ledger_table_name": "keep-glm52-ledger"}
        ),
        client=lambda service: (
            dynamo
            if service == "dynamodb"
            else pytest.fail("unexpected live boundary")
        ),
    )
    operation = "RETAINED_DISCOVER_EXACT_RECOVERY_SNAPSHOT"

    assert persist_live_successors(
        operation_kind=operation,
        invocation=_invocation(operation),
        live_sources={"support_finalized": support},
        request={},
        domain_result=capture,
        ports=ports,
    )

    assert len(dynamo.requests) == 1
    item = decode_item(dynamo.requests[0]["Item"])
    assert item["capture"] == vars(capture)
    assert item["support_plane_finalized_identity_sha256"] == (
        canonical_record_identity(
            "glm52_production_support_plane_finalized", support
        )
    )


def test_production_dispatch_builds_marker_last_h1g_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from glm52_enforcement import task12_live_snapshot_finalization as live
    from glm52_enforcement.records import validate_record
    from glm52_enforcement.task12_lambda_adapters import (
        _materialize_live_operation_request,
    )
    from glm52_enforcement.task12_writers import (
        RetainedWriterActionAuthority,
        RetainedWriterAuditAuthority,
        build_retained_writer_candidate,
        validate_retained_writer_authority,
    )

    writer_spec = importlib.util.spec_from_file_location(
        "_task12_h1g_writer_fixtures",
        Path(__file__).with_name("test_glm52_task12_writers.py"),
    )
    assert writer_spec is not None and writer_spec.loader is not None
    writer_fixtures = importlib.util.module_from_spec(writer_spec)
    writer_spec.loader.exec_module(writer_fixtures)
    terminal = writer_fixtures._terminal_v2()
    terminal = validate_record("glm52_production_terminal_v2", terminal)

    orphan_spec = importlib.util.spec_from_file_location(
        "_task12_h1g_orphan_fixtures",
        Path(__file__).with_name("test_glm52_task12_orphan_audit.py"),
    )
    assert orphan_spec is not None and orphan_spec.loader is not None
    orphan_fixtures = importlib.util.module_from_spec(orphan_spec)
    orphan_spec.loader.exec_module(orphan_fixtures)
    orphan_proof = orphan_fixtures._audit()

    operation = "RETAINED_INVOKE_H1G_DRAINED_WRITER"
    support = _source("glm52_production_support_plane_finalized")
    cleanup, _snapshot = _armed_control()
    finalization = _source(
        "glm52_production_finalization_control",
        state="SNAPSHOT_DISPOSITION_RECORDED",
        terminal_v2_identity_sha256=canonical_record_identity(
            "glm52_production_terminal_v2", terminal
        ),
        support_plane_finalized_identity_sha256=canonical_record_identity(
            "glm52_production_support_plane_finalized", support
        ),
        snapshot_disposition_identity_sha256=canonical_record_identity(
            "glm52_production_snapshot_cleanup_control", cleanup
        ),
    )
    terminal_control = _source(
        "glm52_task12_versioned_writer_control_v1",
        generation=1,
        generation_text="00000001",
        writer_kind="TerminalV2",
        campaign_bucket=(
            "keep-glm52-models-246813579024-us-west-2"
        ),
        object_key=(
            "campaigns/glm52-sky-20260724/submissions/production/"
            "generations/00000001/terminal/PRODUCTION_TERMINAL_V2.json"
        ),
        object_version_id="terminal-v2-version-1",
        file_sha256=hashlib.sha256(b"terminal-v2\n").hexdigest(),
        body_sha256=terminal["canonical_body_sha256"],
    )
    spend_body = {
        "bucket": "keep-glm52-models-246813579024-us-west-2",
        "key": (
            "campaigns/glm52-sky-20260724/runtime/"
            "GPU_SPEND_LEDGER.jsonl"
        ),
        "version_id": "spend-ledger-version-1",
        "file_sha256": "7" * 64,
        "body_sha256": "8" * 64,
        "head_record_sha256": terminal["spend_ledger_head_identity"][
            "identity"
        ],
    }
    spend_identity = {
        **spend_body,
        "canonical_identity_sha256": canonical_sha256(spend_body),
    }
    deletion_body = {
        "stack_id": (
            "arn:aws:cloudformation:us-west-2:246813579024:stack/"
            "keep-glm52-h1g-support/"
            "12345678-1234-1234-1234-123456789012"
        ),
        "observation": "ABSENT_VALIDATION_ERROR",
        "request_id": "support-stack-absence-request-1",
    }
    deletion_identity = {
        **deletion_body,
        "canonical_identity_sha256": canonical_sha256(deletion_body),
    }
    monkeypatch.setattr(
        live,
        "_retained_read_record",
        lambda **_kwargs: finalization,
    )
    monkeypatch.setattr(
        live,
        "_retained_read_terminal",
        lambda **_kwargs: (terminal, terminal_control),
    )
    monkeypatch.setattr(
        live,
        "_retained_read_spend_ledger_head",
        lambda **_kwargs: spend_identity,
    )
    monkeypatch.setattr(
        live,
        "_retained_read_orphan_effect",
        lambda **_kwargs: orphan_proof,
    )
    monkeypatch.setattr(
        live,
        "_retained_support_stack_absent",
        lambda _ports: deletion_identity,
    )
    ports = _WriterPorts(
        "keep-glm52-models-246813579024-us-west-2"
    )

    request = _materialize_live_operation_request(
        ports=ports,
        invocation=_invocation(operation),
        live_sources={
            "snapshot_cleanup_control": cleanup,
            "support_finalized": support,
        },
    )

    candidate = build_retained_writer_candidate(
        writer_kind="H1GDrained",
        campaign_bucket=(
            "keep-glm52-models-246813579024-us-west-2"
        ),
        activation_id="activation-1",
        generation=1,
        authority_domain="FINALIZATION",
        record=request["record"],
    )
    validate_retained_writer_authority(
        candidate=candidate,
        action=RetainedWriterActionAuthority(**request["action"]),
        audit=RetainedWriterAuditAuthority(**request["audit"]),
    )
    assert request["writer_kind"] == "H1GDrained"
    assert request["record"]["terminal_v2_identity"] == {
        "generation": 1,
        "generation_text": "00000001",
        "bucket": terminal_control["campaign_bucket"],
        "key": terminal_control["object_key"],
        "version_id": terminal_control["object_version_id"],
        "file_sha256": terminal_control["file_sha256"],
        "body_sha256": terminal_control["body_sha256"],
        "canonical_identity_sha256": canonical_sha256(
            {
                "generation": 1,
                "generation_text": "00000001",
                "bucket": terminal_control["campaign_bucket"],
                "key": terminal_control["object_key"],
                "version_id": terminal_control["object_version_id"],
                "file_sha256": terminal_control["file_sha256"],
                "body_sha256": terminal_control["body_sha256"],
            }
        ),
    }
    assert request["record"]["spend_ledger_head_identity"] == spend_identity
    assert (
        request["record"]["support_stack_deletion_identity"]
        == deletion_identity
    )
    assert request["record"]["orphan_audit_identity"] == (
        orphan_proof.canonical_identity_sha256
    )
    assert request["record"]["snapshot_cleanup_schedule_identity"][
        "schedule_arn"
    ] == cleanup["schedule_arn"]
    assert request["record"]["worker_launch_liabilities"] == (
        terminal["worker_launch_liabilities"]
    )
    assert request["record"]["retained_resource_inventory_identity"][
        "retained_resources"
    ] == [asdict(item) for item in orphan_proof.retained_resources]
    assert (
        validate_record(
            "glm52_production_h1g_drained",
            request["record"],
        )
        == request["record"]
    )
    assert ports.calls == 0


def _spend_ledger_pair() -> tuple[bytes, str]:
    start_body = {
        "record_type": "glm52_gpu_spend_event_v1",
        "run_id": "glm52-sky-20260724",
        "approval_sha256": "6" * 64,
        "event": "allocation_started",
        "instance_id": "i-0123456789abcdef0",
        "job_id": "glm52-production",
        "timestamp": "2026-07-29T20:00:00Z",
        "prior_record_sha256": "5" * 64,
    }
    start = {
        **start_body,
        "record_sha256": canonical_sha256(start_body),
    }
    end_body = {
        "record_type": "glm52_gpu_spend_event_v1",
        "run_id": "glm52-sky-20260724",
        "approval_sha256": "6" * 64,
        "event": "allocation_ended",
        "instance_id": "i-0123456789abcdef0",
        "timestamp": "2026-07-29T20:15:00Z",
        "prior_record_sha256": start["record_sha256"],
    }
    end = {
        **end_body,
        "record_sha256": canonical_sha256(end_body),
    }
    from glm52_enforcement.canonical import canonical_json_bytes

    raw = canonical_json_bytes(start) + b"\n"
    raw += canonical_json_bytes(end) + b"\n"
    return raw, end["record_sha256"]


def test_h1g_spend_reader_binds_one_exact_current_version_and_head() -> None:
    from glm52_enforcement.task12_live_snapshot_finalization import (
        _retained_read_spend_ledger_head,
    )

    raw, head = _spend_ledger_pair()
    metadata = {
        "HTTPStatusCode": 200,
        "RequestId": "spend-request-1",
        "RetryAttempts": 0,
    }
    requests: list[tuple[str, dict[str, object]]] = []

    class _S3:
        def list_object_versions(self, **kwargs: object) -> object:
            requests.append(("list_object_versions", dict(kwargs)))
            return {
                "Versions": [
                    {
                        "Key": (
                            "campaigns/glm52-sky-20260724/runtime/"
                            "GPU_SPEND_LEDGER.jsonl"
                        ),
                        "VersionId": "spend-ledger-version-1",
                        "IsLatest": True,
                    }
                ],
                "DeleteMarkers": [],
                "IsTruncated": False,
                "ResponseMetadata": metadata,
            }

        def get_object(self, **kwargs: object) -> object:
            requests.append(("get_object", dict(kwargs)))
            return {
                "Body": BytesIO(raw),
                "VersionId": "spend-ledger-version-1",
                "ChecksumSHA256": base64.b64encode(
                    hashlib.sha256(raw).digest()
                ).decode("ascii"),
                "ResponseMetadata": metadata,
            }

    class _Ports:
        def client(self, service: str) -> object:
            assert service == "s3"
            return _S3()

    identity = _retained_read_spend_ledger_head(
        ports=_Ports(),
        expected_head_record_sha256=head,
    )

    assert identity["version_id"] == "spend-ledger-version-1"
    assert identity["file_sha256"] == hashlib.sha256(raw).hexdigest()
    assert identity["head_record_sha256"] == head
    spend_key = (
        "campaigns/glm52-sky-20260724/runtime/GPU_SPEND_LEDGER.jsonl"
    )
    assert requests == [
        (
            "list_object_versions",
            {
                "Bucket": (
                    "keep-glm52-models-246813579024-us-west-2"
                ),
                "Prefix": spend_key,
                "ExpectedBucketOwner": "246813579024",
            },
        ),
        (
            "get_object",
            {
                "Bucket": (
                    "keep-glm52-models-246813579024-us-west-2"
                ),
                "Key": spend_key,
                "VersionId": "spend-ledger-version-1",
                "ExpectedBucketOwner": "246813579024",
                "ChecksumMode": "ENABLED",
            },
        ),
    ]
    assert identity["canonical_identity_sha256"] == canonical_sha256(
        {
            key: value
            for key, value in identity.items()
            if key != "canonical_identity_sha256"
        }
    )
    with pytest.raises(ValueError, match="terminal head drifted"):
        _retained_read_spend_ledger_head(
            ports=_Ports(),
            expected_head_record_sha256="0" * 64,
        )


def test_h1g_support_stack_absence_retains_authenticated_observation() -> None:
    from glm52_enforcement.task12_live_snapshot_finalization import (
        _retained_support_stack_absent,
    )

    stack_id = (
        "arn:aws:cloudformation:us-west-2:246813579024:stack/"
        "keep-glm52-h1g-support/"
        "12345678-1234-1234-1234-123456789012"
    )

    class _CloudFormation:
        def describe_stacks(self, **_kwargs: object) -> object:
            return {
                "Stacks": [
                    {
                        "StackId": stack_id,
                        "StackStatus": "DELETE_COMPLETE",
                    }
                ],
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "support-delete-request-1",
                    "RetryAttempts": 0,
                },
            }

    class _Ports:
        deployment = SimpleNamespace(
            role_coordinates={"support_stack_id": stack_id}
        )

        def client(self, service: str) -> object:
            assert service == "cloudformation"
            return _CloudFormation()

    proof = _retained_support_stack_absent(_Ports())

    assert proof["stack_id"] == stack_id
    assert proof["observation"] == "DELETE_COMPLETE"
    assert proof["request_id"] == "support-delete-request-1"
    assert proof["canonical_identity_sha256"] == canonical_sha256(
        {
            key: value
            for key, value in proof.items()
            if key != "canonical_identity_sha256"
        }
    )


def test_orphan_operation_consumes_activation_authority_and_runtime_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: production dispatch retains the deliberate missing-producer path."""

    from dataclasses import asdict

    from glm52_enforcement import task12_orphan_authority as authority_module
    from glm52_enforcement import task12_resource_scanner as scanner_module
    from glm52_enforcement.task12_live_snapshot_finalization import (
        materialize_live_request,
    )

    spec = importlib.util.spec_from_file_location(
        "_task12_orphan_runtime_fixtures",
        Path(__file__).with_name("test_glm52_task12_orphan_audit.py"),
    )
    assert spec is not None and spec.loader is not None
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    baseline = (fixtures._grant("baseline"),)
    direct = fixtures._direct()
    service = fixtures._service()
    request = {
        "expected_retained": [
            asdict(item) for item in fixtures._expected()
        ],
        "inventory": asdict(fixtures._inventory()),
        "retained_grant_baseline": [
            asdict(item) for item in baseline
        ],
        "pre_cleanup_grants": asdict(
            fixtures._grant_scan(
                baseline + (direct.grant, service.grant),
                observed_at="2026-07-29T12:01:00Z",
            )
        ),
        "direct_grants": [asdict(direct)],
        "service_grants": [asdict(service)],
        "settling_deadline": "2026-07-29T12:04:00Z",
    }
    calls = []
    monkeypatch.setattr(
        authority_module,
        "read_activation_authority",
        lambda **kwargs: calls.append(("read", kwargs)) or {"authority": True},
    )
    monkeypatch.setattr(
        scanner_module,
        "scan_task12_resources",
        lambda **kwargs: calls.append(("scan", kwargs)) or fixtures._inventory(),
    )
    monkeypatch.setattr(
        authority_module,
        "cleanup_activation_orphans",
        lambda **kwargs: calls.append(("cleanup", kwargs))
        or {"audit_orphans_request": request},
    )

    class Ports:
        deployment = SimpleNamespace(
            role_coordinates={
                "ledger_table_name": "keep-glm52-ledger"
            }
        )

        def client(self, service: str) -> object:
            return "client-" + service

    result = materialize_live_request(
        operation_kind="RETAINED_AUDIT_SUPPORT_ORPHANS",
        invocation=_invocation("RETAINED_AUDIT_SUPPORT_ORPHANS"),
        live_sources=_sources_for("RETAINED_AUDIT_SUPPORT_ORPHANS"),
        ports=Ports(),
    )
    assert canonical_sha256(result) == canonical_sha256(request)
    assert [item[0] for item in calls] == ["read", "scan", "cleanup"]


def test_successor_hook_never_persists_a_future_request() -> None:
    from glm52_enforcement.task12_live_snapshot_finalization import (
        persist_live_successors,
    )

    operation = "SNAPSHOT_CLEANUP_SEND_SAME_ID_OR_READ_BACK"
    ports = _NoEffects()
    assert persist_live_successors(
        operation_kind=operation,
        invocation=_invocation(operation),
        live_sources=_send_sources(),
        request={"snapshot_id": "snap-0123456789abcdef0"},
        domain_result={"outcome": "DELETE_SENT"},
        ports=ports,
    ) is True
    assert ports.calls == 0
