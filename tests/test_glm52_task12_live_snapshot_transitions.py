from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from glm52_enforcement.canonical import canonical_sha256
from glm52_enforcement.dynamodb import decode_item, encode_item
from glm52_enforcement.records import ledger_sk


KMS_KEY = (
    "arn:aws:kms:us-west-2:246813579024:key/"
    "12345678-1234-4234-8234-1234567890ab"
)
RUN_ID = "glm52-sky-20260724"
ACTIVATION_ID = "activation-1"
OWNER_EXECUTION = (
    "arn:aws:states:us-west-2:246813579024:"
    "execution:keep-glm52-h1g-snapshot-cleanup:cleanup-1"
)
OWNER_VERSION = (
    "arn:aws:states:us-west-2:246813579024:"
    "stateMachine:keep-glm52-h1g-snapshot-cleanup:7"
)


def _record_fixtures() -> object:
    spec = importlib.util.spec_from_file_location(
        "_task12_snapshot_transition_record_fixtures",
        Path(__file__).with_name("test_glm52_enforcement_dynamodb.py"),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _invocation(
    operation: str,
    *,
    capsule: dict[str, object] | None = None,
    domain_result: dict[str, object] | None = None,
    delete_ambiguity: dict[str, str] | None = None,
    retained_reconciliation: dict[str, object] | None = None,
) -> object:
    operation_input: dict[str, object] = {}
    if capsule is not None:
        result: dict[str, object] = {"owner_nonce_capsule": capsule}
        if domain_result is not None:
            result["domain_result"] = domain_result
        operation_input["task12_last_result"] = {
            "result": result
        }
    if delete_ambiguity is not None:
        operation_input["delete_ambiguity"] = delete_ambiguity
    if retained_reconciliation is not None:
        retained_body = {
            "schema_version": 1,
            "record_type": "glm52_task12_lambda_result_v1",
            "handler_kind": "RETAINED_SNAPSHOT_CLEANUP",
            "operation_kind": "SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE",
            "operation_input_identity_sha256": "5" * 64,
            "outcome": "SUCCEEDED",
            "result": {"domain_result": retained_reconciliation},
        }
        operation_input["snapshot_reconciliation_result"] = {
            **retained_body,
            "canonical_body_sha256": canonical_sha256(retained_body),
        }
    return SimpleNamespace(
        operation_kind=operation,
        activation_id=ACTIVATION_ID,
        activation_ordinal=1,
        generation=1,
        generation_text="00000001",
        dispatch_identity_sha256="b" * 64,
        state_machine_execution_arn=OWNER_EXECUTION,
        caller_state_machine_version_arn=OWNER_VERSION,
        operation_input=operation_input,
    )


class _Dynamo:
    def __init__(self, records: dict[str, dict[str, object]]) -> None:
        self.records = records
        self.calls: list[dict[str, object]] = []

    def get_item(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        key = decode_item(kwargs["Key"])
        record = self.records.get(key["SK"])
        response: dict[str, object] = {
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "ddb-" + str(len(self.calls)),
                "RetryAttempts": 0,
            }
        }
        if record is not None:
            response["Item"] = encode_item(
                {"PK": key["PK"], "SK": key["SK"], **record}
            )
        return response


class _Kms:
    def __init__(self) -> None:
        self.plaintext = b"n" * 32
        self.context: dict[str, str] | None = None
        self.generate_calls = 0
        self.decrypt_calls = 0

    def generate_data_key(self, **kwargs: object) -> dict[str, object]:
        self.generate_calls += 1
        self.context = dict(kwargs["EncryptionContext"])
        return {
            "KeyId": KMS_KEY,
            "Plaintext": self.plaintext,
            "CiphertextBlob": b"snapshot-owner-ciphertext",
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "generate-1",
                "RetryAttempts": 0,
            },
        }

    def decrypt(self, **kwargs: object) -> dict[str, object]:
        self.decrypt_calls += 1
        assert kwargs["EncryptionContext"] == self.context
        return {
            "KeyId": KMS_KEY,
            "Plaintext": self.plaintext,
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "decrypt-1",
                "RetryAttempts": 0,
            },
        }


class _Ports:
    def __init__(
        self,
        *,
        records: dict[str, dict[str, object]],
        kms: _Kms,
        ec2: object | None = None,
    ) -> None:
        self.dynamodb = _Dynamo(records)
        self.kms = kms
        self.ec2 = ec2
        self.deployment = SimpleNamespace(
            role_coordinates={
                "ledger_table_name": "keep-glm52-h1g-ledger",
                "kms_key_id": KMS_KEY,
            }
        )

    def client(self, service: str) -> object:
        if service == "dynamodb":
            return self.dynamodb
        if service == "kms":
            return self.kms
        if service == "ec2" and self.ec2 is not None:
            return self.ec2
        raise AssertionError("foreign service: " + service)

    @staticmethod
    def now_utc() -> datetime:
        return datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)


def _authorities() -> dict[str, dict[str, object]]:
    fixtures = _record_fixtures()
    return {
        "ACTIVATION_INDEX": fixtures._activation_index(),
        ledger_sk(
            "glm52_production_control", activation_id=ACTIVATION_ID
        ): fixtures._control(
            phase="TEARDOWN_SEALED",
            revision=5,
            updated_at="2026-08-04T12:00:00Z",
        ),
        ledger_sk(
            "glm52_production_recovery_control",
            activation_id=ACTIVATION_ID,
        ): fixtures._closed_record(
            "glm52_production_recovery_control",
            activation_id=ACTIVATION_ID,
            state="RECOVERY_COMPLETE",
            revision=5,
            updated_at="2026-08-04T12:00:00Z",
        ),
    }


def _armed_control() -> dict[str, object]:
    from glm52_enforcement.task12_snapshot_cleanup import SnapshotObservation

    fixtures = _record_fixtures()
    tags = [{"Key": "RunId", "Value": RUN_ID}]
    observation = SnapshotObservation(
        snapshot_id="snap-0123456789abcdef0",
        source_volume_id="vol-0123456789abcdef0",
        kms_key_arn=KMS_KEY,
        snapshot_tags_sha256=canonical_sha256(tags),
        encrypted=True,
        state="completed",
        observed_at="2026-07-28T12:00:00Z",
        describe_request_id="describe-seed-1",
        describe_response_sha256="d" * 64,
    )
    return fixtures._closed_record(
        "glm52_production_snapshot_cleanup_control",
        activation_id=ACTIVATION_ID,
        state="ARMED",
        revision=2,
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


def test_acquire_builds_current_transition_and_ciphertext_only_capsule() -> None:
    from glm52_enforcement.task12_live_snapshot_finalization import (
        materialize_live_request,
    )

    operation = "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER"
    kms = _Kms()
    ports = _Ports(records=_authorities(), kms=kms)
    ec2 = _SnapshotEc2(None)
    ports.ec2 = ec2

    request = materialize_live_request(
        operation_kind=operation,
        invocation=_invocation(operation),
        live_sources={"snapshot_cleanup_control": _armed_control()},
        ports=ports,
    )

    assert request["plan"]["cleanup_control"]["before"]["state"] == "ARMED"
    assert request["plan"]["cleanup_control"]["after"]["state"] == "OWNED"
    assert request["observed_at"] == "2026-08-04T12:00:00Z"
    assert request["owner_nonce_capsule"]["nonce_sha256"] == (
        "fe06271acc7d35b9406d4397b7b658ad91a21e4a47a40dc5805294d6fc77a028"
    )
    assert "plaintext" not in str(request).lower()
    assert "raw_owner_nonce_hex" not in request
    assert kms.generate_calls == 1


def test_acquire_rejects_stale_authority_before_generating_nonce() -> None:
    from glm52_enforcement.task12_live_snapshot_finalization import (
        materialize_live_request,
    )

    operation = "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER"
    authorities = _authorities()
    authorities["ACTIVATION_INDEX"] = {
        **authorities["ACTIVATION_INDEX"],
        "current_activation_id": "activation-foreign",
    }
    kms = _Kms()
    ports = _Ports(records=authorities, kms=kms)

    try:
        materialize_live_request(
            operation_kind=operation,
            invocation=_invocation(operation),
            live_sources={"snapshot_cleanup_control": _armed_control()},
            ports=ports,
        )
    except ValueError as exc:
        assert "activation" in str(exc)
    else:  # pragma: no cover - explicit RED/GREEN failure
        raise AssertionError("stale activation authority was accepted")
    assert kms.generate_calls == 0


def test_acquire_accepts_retained_cleanup_after_activation_rollover() -> None:
    from glm52_enforcement.task12_live_snapshot_finalization import (
        materialize_live_request,
    )

    operation = "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER"
    authorities = _authorities()
    authorities["ACTIVATION_INDEX"] = {
        **authorities["ACTIVATION_INDEX"],
        "current_activation_id": "activation-2",
        "current_activation_ordinal": 2,
    }
    kms = _Kms()
    ports = _Ports(records=authorities, kms=kms)

    request = materialize_live_request(
        operation_kind=operation,
        invocation=_invocation(operation),
        live_sources={"snapshot_cleanup_control": _armed_control()},
        ports=ports,
    )

    assert request["plan"]["cleanup_control"]["after"]["state"] == "OWNED"
    assert request["plan"]["index"]["expected"][
        "current_activation_ordinal"
    ] == 2
    assert kms.generate_calls == 1


def test_acquire_rejects_cleanup_more_than_one_activation_behind() -> None:
    from glm52_enforcement.task12_live_snapshot_finalization import (
        materialize_live_request,
    )

    operation = "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER"
    authorities = _authorities()
    authorities["ACTIVATION_INDEX"] = {
        **authorities["ACTIVATION_INDEX"],
        "current_activation_id": "activation-3",
        "current_activation_ordinal": 3,
    }
    kms = _Kms()
    ports = _Ports(records=authorities, kms=kms)

    with pytest.raises(ValueError, match="activation authority"):
        materialize_live_request(
            operation_kind=operation,
            invocation=_invocation(operation),
            live_sources={"snapshot_cleanup_control": _armed_control()},
            ports=ports,
        )
    assert kms.generate_calls == 0


def _expired_owned_control(
    state: str,
) -> dict[str, object]:
    from glm52_enforcement.records import validate_record

    fixtures = _record_fixtures()
    common = {
        "activation_id": ACTIVATION_ID,
        "state": state,
        "revision": 5,
        "snapshot_id": _armed_control()["snapshot_id"],
        "snapshot_identity_sha256": _armed_control()[
            "snapshot_identity_sha256"
        ],
        "source_volume_id": _armed_control()["source_volume_id"],
        "snapshot_tags_sha256": _armed_control()[
            "snapshot_tags_sha256"
        ],
        "delete_not_before": "2026-08-04T12:00:00Z",
        "schedule_arn": _armed_control()["schedule_arn"],
        "schedule_input_sha256": canonical_sha256({}),
        "owner_attempt": 1,
        "owner_execution_arn": (
            "arn:aws:states:us-west-2:246813579024:"
            "execution:keep-glm52-h1g-snapshot-cleanup:cleanup-old"
        ),
        "owner_state_machine_version_arn": OWNER_VERSION,
        "owner_dispatch_identity_sha256": "8" * 64,
        "owner_invocation_nonce_sha256": "9" * 64,
        "owner_hard_expires_at": "2026-08-04T11:59:59Z",
        "updated_at": "2026-08-04T11:00:00Z",
    }
    if state == "OWNED":
        return fixtures._closed_record(
            "glm52_production_snapshot_cleanup_control",
            **common,
        )
    later = {
        "cleanup_authority_audit_identities": ["1" * 64],
        "latest_cleanup_authority_audit_identity_sha256": "1" * 64,
        "delete_action_identities": ["2" * 64],
        "latest_delete_action_identity_sha256": "2" * 64,
        "delete_logical_attempt": 1,
        "delete_call_count": 1,
    }
    if state == "DELETE_RECONCILING":
        later.update(
            last_describe_request_id="describe-old-1",
            last_describe_response_sha256="4" * 64,
        )
    return validate_record(
        "glm52_production_snapshot_cleanup_control",
        fixtures._closed_record(
            "glm52_production_snapshot_cleanup_control",
            **common,
            **later,
        ),
    )


def test_takeover_rejects_live_owner_before_generating_nonce() -> None:
    import pytest

    from glm52_enforcement.task12_live_snapshot_finalization import (
        materialize_live_request,
    )

    operation = "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER"
    control = {
        **_expired_owned_control("OWNED"),
        "owner_hard_expires_at": "2026-08-04T12:00:01Z",
    }
    kms = _Kms()
    ports = _Ports(records=_authorities(), kms=kms)

    with pytest.raises(ValueError, match="live owner"):
        materialize_live_request(
            operation_kind=operation,
            invocation=_invocation(operation),
            live_sources={"snapshot_cleanup_control": control},
            ports=ports,
        )
    assert kms.generate_calls == 0


@pytest.mark.parametrize(
    "state", ["OWNED", "DELETE_POSSIBLY_SENT", "DELETE_RECONCILING"]
)
def test_takeover_replaces_only_expired_owner_and_preserves_lineage(
    state: str,
) -> None:
    from glm52_enforcement.task12_live_snapshot_finalization import (
        materialize_live_request,
    )

    operation = "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER"
    control = _expired_owned_control(state)
    kms = _Kms()
    ports = _Ports(records=_authorities(), kms=kms)
    request = materialize_live_request(
        operation_kind=operation,
        invocation=_invocation(operation),
        live_sources={"snapshot_cleanup_control": control},
        ports=ports,
    )

    assert set(request["plan"]) == {
        "index",
        "authority",
        "cleanup_control",
    }
    before = request["plan"]["cleanup_control"]["before"]
    after = request["plan"]["cleanup_control"]["after"]
    assert before == control
    assert after["state"] == state
    assert after["owner_attempt"] == control["owner_attempt"] + 1
    assert after["owner_execution_arn"] == OWNER_EXECUTION
    assert after["revision"] == control["revision"] + 1
    for field in (
        "delete_logical_attempt",
        "delete_call_count",
        "cleanup_authority_audit_identities",
        "delete_action_identities",
        "latest_delete_action_identity_sha256",
    ):
        assert after[field] == control[field]
    assert request["owner_nonce_capsule"]["nonce_sha256"] == (
        after["owner_invocation_nonce_sha256"]
    )
    assert "raw_owner_nonce_hex" not in request
    assert kms.generate_calls == 1


def test_recovered_reconciling_owner_builds_readback_from_persisted_action() -> None:
    from glm52_enforcement.records import (
        canonical_record_identity,
        validate_record,
    )
    from glm52_enforcement.task12_live_snapshot_finalization import (
        materialize_live_request,
    )

    acquire = "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER"
    prior_control = _expired_owned_control("DELETE_RECONCILING")
    fixtures = _record_fixtures()
    action = fixtures._closed_record(
        "glm52_production_snapshot_cleanup_action",
        activation_id=ACTIVATION_ID,
        state="CONSUMED",
        attempt=prior_control["delete_logical_attempt"],
        owner_attempt=prior_control["owner_attempt"],
        owner_execution_arn=prior_control["owner_execution_arn"],
        owner_state_machine_version_arn=prior_control[
            "owner_state_machine_version_arn"
        ],
        owner_dispatch_identity_sha256=prior_control[
            "owner_dispatch_identity_sha256"
        ],
        owner_invocation_nonce_sha256=prior_control[
            "owner_invocation_nonce_sha256"
        ],
        owner_hard_expires_at=prior_control["owner_hard_expires_at"],
        authority_barrier_nonce_sha256=prior_control[
            "cleanup_barrier_nonce_sha256"
        ],
    )
    action_identity = canonical_record_identity(
        "glm52_production_snapshot_cleanup_action", action
    )
    control_body = {
        **prior_control,
        "cleanup_authority_audit_identities": [action_identity],
        "latest_cleanup_authority_audit_identity_sha256": action_identity,
        "delete_action_identities": [action_identity],
        "latest_delete_action_identity_sha256": action_identity,
    }
    prior_control = validate_record(
        "glm52_production_snapshot_cleanup_control",
        control_body,
    )
    kms = _Kms()
    ports = _Ports(records=_authorities(), kms=kms)
    ec2 = _SnapshotEc2(None)
    ports.ec2 = ec2
    takeover = materialize_live_request(
        operation_kind=acquire,
        invocation=_invocation(acquire),
        live_sources={"snapshot_cleanup_control": prior_control},
        ports=ports,
    )
    control = takeover["plan"]["cleanup_control"]["after"]
    capsule = takeover["owner_nonce_capsule"]

    operation = "SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE"
    request = materialize_live_request(
        operation_kind=operation,
        invocation=_invocation(
            operation,
            capsule=capsule,
            domain_result={
                "cleanup_state": "DELETE_RECONCILING",
                "resolution": {"outcome": "EXACT_LIVE_OWNER_COMMIT"},
            },
        ),
        live_sources={
            "snapshot_cleanup_control": control,
            "snapshot_cleanup_action": action,
        },
        ports=ports,
    )

    assert set(request["plan"]) == {
        "index",
        "authority",
        "cleanup_control",
        "cleanup_action",
    }
    assert request["plan"]["cleanup_control"]["expected"] == control
    assert request["plan"]["cleanup_action"]["expected"] == action
    assert request["owner_nonce_capsule"] == capsule
    assert ec2.calls == []


def test_recovered_action_close_keeps_prior_owner_lineage_immutable() -> None:
    from glm52_enforcement.records import (
        canonical_record_identity,
        validate_record,
    )
    from glm52_enforcement.task12_live_snapshot_finalization import (
        materialize_live_request,
    )

    acquire = "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER"
    prior_control = _expired_owned_control("DELETE_RECONCILING")
    fixtures = _record_fixtures()
    action = fixtures._closed_record(
        "glm52_production_snapshot_cleanup_action",
        activation_id=ACTIVATION_ID,
        state="CONSUMED",
        attempt=prior_control["delete_logical_attempt"],
        owner_attempt=prior_control["owner_attempt"],
        owner_execution_arn=prior_control["owner_execution_arn"],
        owner_state_machine_version_arn=prior_control[
            "owner_state_machine_version_arn"
        ],
        owner_dispatch_identity_sha256=prior_control[
            "owner_dispatch_identity_sha256"
        ],
        owner_invocation_nonce_sha256=prior_control[
            "owner_invocation_nonce_sha256"
        ],
        owner_hard_expires_at=prior_control["owner_hard_expires_at"],
        authority_barrier_nonce_sha256=prior_control[
            "cleanup_barrier_nonce_sha256"
        ],
    )
    action_identity = canonical_record_identity(
        "glm52_production_snapshot_cleanup_action", action
    )
    control_body = {
        **prior_control,
        "cleanup_authority_audit_identities": [action_identity],
        "latest_cleanup_authority_audit_identity_sha256": action_identity,
        "delete_action_identities": [action_identity],
        "latest_delete_action_identity_sha256": action_identity,
    }
    prior_control = validate_record(
        "glm52_production_snapshot_cleanup_control",
        control_body,
    )
    kms = _Kms()
    ports = _Ports(records=_authorities(), kms=kms)
    takeover = materialize_live_request(
        operation_kind=acquire,
        invocation=_invocation(acquire),
        live_sources={"snapshot_cleanup_control": prior_control},
        ports=ports,
    )
    control = takeover["plan"]["cleanup_control"]["after"]
    capsule = takeover["owner_nonce_capsule"]
    operation = "SNAPSHOT_CLEANUP_CLOSE_AMBIGUOUS_ATTEMPT"
    request = materialize_live_request(
        operation_kind=operation,
        invocation=_invocation(
            operation,
            capsule=capsule,
            retained_reconciliation={
                "source_state": "DELETE_RECONCILING",
                "target_state": "DELETE_RECONCILING",
                "committed": False,
                "snapshot_present": True,
                "delete_logical_attempt": control["delete_logical_attempt"],
                "resolution": None,
            },
        ),
        live_sources={
            "snapshot_cleanup_control": control,
            "snapshot_cleanup_action": action,
        },
        ports=ports,
    )

    before = request["plan"]["action"]["before"]
    after = request["plan"]["action"]["after"]
    for field in (
        "owner_attempt",
        "owner_execution_arn",
        "owner_state_machine_version_arn",
        "owner_dispatch_identity_sha256",
        "owner_invocation_nonce_sha256",
    ):
        assert before[field] == action[field]
        assert after[field] == action[field]
    assert (
        after["owner_attempt"],
        after["owner_execution_arn"],
        after["owner_invocation_nonce_sha256"],
    ) != (
        control["owner_attempt"],
        control["owner_execution_arn"],
        control["owner_invocation_nonce_sha256"],
    )
    assert after["state"] == "AMBIGUOUS"


def test_arm_delete_action_builds_action_only_put_from_live_owner() -> None:
    from glm52_enforcement.task12_live_snapshot_finalization import (
        materialize_live_request,
    )

    kms = _Kms()
    ports = _Ports(records=_authorities(), kms=kms)
    acquire = "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER"
    acquired = materialize_live_request(
        operation_kind=acquire,
        invocation=_invocation(acquire),
        live_sources={"snapshot_cleanup_control": _armed_control()},
        ports=ports,
    )
    control = acquired["plan"]["cleanup_control"]["after"]
    capsule = acquired["owner_nonce_capsule"]

    operation = "SNAPSHOT_CLEANUP_ARM_DELETE_ACTION"
    request = materialize_live_request(
        operation_kind=operation,
        invocation=_invocation(operation, capsule=capsule),
        live_sources={"snapshot_cleanup_control": control},
        ports=ports,
    )

    assert set(request["plan"]) == {"index", "control", "action"}
    assert request["plan"]["control"]["expected"]["state"] == "OWNED"
    assert request["plan"]["action"]["item"]["state"] == "ARMED"
    assert request["plan"]["action"]["item"]["attempt"] == 1
    assert request["sort_key"].endswith(
        "#SNAPSHOT_CLEANUP_ACTION#00000001"
    )
    assert request["owner_nonce_capsule"] == capsule
    assert "raw_owner_nonce_hex" not in request
    assert "plaintext" not in str(request).lower()
    assert kms.decrypt_calls == 1


def test_atomic_stage_consumes_live_action_and_advances_control_once() -> None:
    from glm52_enforcement.task12_live_snapshot_finalization import (
        materialize_live_request,
    )

    kms = _Kms()
    ports = _Ports(records=_authorities(), kms=kms)
    acquire = "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER"
    acquired = materialize_live_request(
        operation_kind=acquire,
        invocation=_invocation(acquire),
        live_sources={"snapshot_cleanup_control": _armed_control()},
        ports=ports,
    )
    control = acquired["plan"]["cleanup_control"]["after"]
    capsule = acquired["owner_nonce_capsule"]
    fixtures = _record_fixtures()
    action = fixtures._closed_record(
        "glm52_production_snapshot_cleanup_action",
        activation_id=ACTIVATION_ID,
        state="ARMED",
        attempt=1,
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
        owner_hard_expires_at=control["owner_hard_expires_at"],
        authority_barrier_nonce_sha256=control[
            "cleanup_barrier_nonce_sha256"
        ],
    )

    operation = "SNAPSHOT_CLEANUP_ATOMIC_CONSUME_AND_STAGE_POSSIBLY_SENT"
    request = materialize_live_request(
        operation_kind=operation,
        invocation=_invocation(operation, capsule=capsule),
        live_sources={
            "snapshot_cleanup_action": action,
            "snapshot_cleanup_control": control,
        },
        ports=ports,
    )

    plan = request["plan"]
    assert plan["cleanup_control"]["after"]["state"] == (
        "DELETE_POSSIBLY_SENT"
    )
    assert plan["cleanup_control"]["after"]["delete_call_count"] == 1
    assert plan["cleanup_action"]["after"]["state"] == "CONSUMED"
    assert plan["cleanup_transition"]["item"]["from_state"] == "OWNED"
    assert plan["cleanup_transition"]["item"]["to_state"] == (
        "DELETE_POSSIBLY_SENT"
    )
    assert request["owner_nonce_capsule"] == capsule
    assert "raw_owner_nonce_hex" not in request
    assert kms.decrypt_calls == 1


class _SnapshotEc2:
    def __init__(self, snapshot: dict[str, object] | None) -> None:
        self.snapshot = snapshot
        self.calls: list[dict[str, object]] = []

    def describe_snapshots(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        return {
            "Snapshots": [] if self.snapshot is None else [self.snapshot],
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "describe-reconcile-1",
                "RetryAttempts": 0,
            },
        }


class _SnapshotNotFoundError(Exception):
    def __init__(self) -> None:
        super().__init__("snapshot not found")
        self.response = {
            "Error": {
                "Code": "InvalidSnapshot.NotFound",
                "Message": (
                    "The snapshot snap-0123456789abcdef0 does not exist"
                ),
            },
            "ResponseMetadata": {
                "HTTPStatusCode": 400,
                "RequestId": "describe-not-found-1",
                "RetryAttempts": 0,
            },
        }


class _SnapshotNotFoundEc2:
    @staticmethod
    def describe_snapshots(**kwargs: object) -> dict[str, object]:
        assert kwargs == {"SnapshotIds": ["snap-0123456789abcdef0"]}
        raise _SnapshotNotFoundError()


def test_reconcile_capture_accepts_exact_ec2_not_found_error() -> None:
    from glm52_enforcement.task12_live_snapshot_finalization import (
        _snapshot_capture_and_schedule,
    )

    ports = _Ports(
        records=_authorities(),
        kms=_Kms(),
        ec2=_SnapshotNotFoundEc2(),
    )
    result = _snapshot_capture_and_schedule(
        control=_armed_control(),
        ports=ports,
        include_describe_evidence=True,
    )

    assert result["snapshot_present"] is False
    assert result["describe_request_id"] == "describe-not-found-1"
    assert result["describe_response_sha256"] is not None


def test_reconcile_describe_builds_transition_from_current_live_observation() -> None:
    from glm52_enforcement.task12_live_snapshot_finalization import (
        materialize_live_request,
    )

    kms = _Kms()
    ports = _Ports(records=_authorities(), kms=kms)
    acquire = "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER"
    acquired = materialize_live_request(
        operation_kind=acquire,
        invocation=_invocation(acquire),
        live_sources={"snapshot_cleanup_control": _armed_control()},
        ports=ports,
    )
    control = acquired["plan"]["cleanup_control"]["after"]
    capsule = acquired["owner_nonce_capsule"]
    fixtures = _record_fixtures()
    action = fixtures._closed_record(
        "glm52_production_snapshot_cleanup_action",
        activation_id=ACTIVATION_ID,
        state="ARMED",
        attempt=1,
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
        owner_hard_expires_at=control["owner_hard_expires_at"],
        authority_barrier_nonce_sha256=control[
            "cleanup_barrier_nonce_sha256"
        ],
    )
    staged_request = materialize_live_request(
        operation_kind=(
            "SNAPSHOT_CLEANUP_ATOMIC_CONSUME_AND_STAGE_POSSIBLY_SENT"
        ),
        invocation=_invocation(
            "SNAPSHOT_CLEANUP_ATOMIC_CONSUME_AND_STAGE_POSSIBLY_SENT",
            capsule=capsule,
        ),
        live_sources={
            "snapshot_cleanup_action": action,
            "snapshot_cleanup_control": control,
        },
        ports=ports,
    )
    staged = staged_request["plan"]["cleanup_control"]["after"]
    consumed_action = staged_request["plan"]["cleanup_action"]["after"]
    snapshot = {
        "SnapshotId": staged["snapshot_id"],
        "VolumeId": staged["source_volume_id"],
        "KmsKeyId": (
            "arn:aws:kms:us-west-2:246813579024:key/"
            "12345678-1234-4234-8234-1234567890ab"
        ),
        "Encrypted": True,
        "State": "completed",
        "StartTime": datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc),
        "Tags": [{"Key": "RunId", "Value": RUN_ID}],
    }
    ec2 = _SnapshotEc2(snapshot)
    ports.ec2 = ec2
    operation = "SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE"
    request = materialize_live_request(
        operation_kind=operation,
        invocation=_invocation(
            operation,
            capsule=capsule,
            domain_result={
                "outcome": "DELETION_IN_PROGRESS",
                "request_id": "delete-1",
                "response_sha256": "d" * 64,
            },
        ),
        live_sources={
            "snapshot_cleanup_control": staged,
            "snapshot_cleanup_action": consumed_action,
        },
        ports=ports,
    )

    after = request["plan"]["cleanup_control"]["after"]
    assert after["state"] == "DELETE_RECONCILING"
    assert after["latest_delete_request_id"] == "delete-1"
    assert after["latest_delete_response_sha256"] == "d" * 64
    assert after["last_describe_request_id"] == "describe-reconcile-1"
    assert request["owner_nonce_capsule"] == capsule
    assert ec2.calls == [{"SnapshotIds": [staged["snapshot_id"]]}]


def test_reconcile_describe_accepts_authenticated_snapshot_absence() -> None:
    from glm52_enforcement.records import (
        canonical_record_identity,
        validate_record,
    )
    from glm52_enforcement.task12_live_snapshot_finalization import (
        materialize_live_request,
    )

    kms = _Kms()
    ports = _Ports(records=_authorities(), kms=kms)
    acquire = "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER"
    acquired = materialize_live_request(
        operation_kind=acquire,
        invocation=_invocation(acquire),
        live_sources={"snapshot_cleanup_control": _armed_control()},
        ports=ports,
    )
    owned = acquired["plan"]["cleanup_control"]["after"]
    capsule = acquired["owner_nonce_capsule"]
    fixtures = _record_fixtures()
    action = fixtures._closed_record(
        "glm52_production_snapshot_cleanup_action",
        activation_id=ACTIVATION_ID,
        state="CONSUMED",
        attempt=1,
        owner_attempt=owned["owner_attempt"],
        owner_execution_arn=owned["owner_execution_arn"],
        owner_state_machine_version_arn=owned[
            "owner_state_machine_version_arn"
        ],
        owner_dispatch_identity_sha256=owned[
            "owner_dispatch_identity_sha256"
        ],
        owner_invocation_nonce_sha256=owned[
            "owner_invocation_nonce_sha256"
        ],
        owner_hard_expires_at=owned["owner_hard_expires_at"],
        authority_barrier_nonce_sha256=owned[
            "cleanup_barrier_nonce_sha256"
        ],
    )
    action_identity = canonical_record_identity(
        "glm52_production_snapshot_cleanup_action", action
    )
    possibly_sent = validate_record(
        "glm52_production_snapshot_cleanup_control",
        {
            **owned,
            "state": "DELETE_POSSIBLY_SENT",
            "cleanup_authority_audit_identities": [action_identity],
            "latest_cleanup_authority_audit_identity_sha256": (
                action_identity
            ),
            "delete_action_identities": [action_identity],
            "latest_delete_action_identity_sha256": action_identity,
            "delete_logical_attempt": 1,
            "delete_call_count": 1,
            "revision": owned["revision"] + 1,
            "updated_at": "2026-08-04T12:00:00Z",
        },
    )
    ec2 = _SnapshotEc2(None)
    ports.ec2 = ec2
    operation = "SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE"
    request = materialize_live_request(
        operation_kind=operation,
        invocation=_invocation(
            operation,
            capsule=capsule,
            domain_result={
                "outcome": "ACCEPTED",
                "request_id": "delete-1",
                "response_sha256": "d" * 64,
            },
        ),
        live_sources={
            "snapshot_cleanup_control": possibly_sent,
            "snapshot_cleanup_action": action,
        },
        ports=ports,
    )

    after = request["plan"]["cleanup_control"]["after"]
    assert after["state"] == "DELETE_RECONCILING"
    assert after["last_describe_request_id"] == "describe-reconcile-1"
    assert after["last_describe_response_sha256"] is not None
    assert ec2.calls == [{"SnapshotIds": [possibly_sent["snapshot_id"]]}]


def test_terminal_evidence_closes_absent_snapshot_and_owner() -> None:
    from glm52_enforcement.records import validate_record
    from glm52_enforcement.task12_live_snapshot_finalization import (
        materialize_live_request,
    )

    kms = _Kms()
    ports = _Ports(records=_authorities(), kms=kms)
    acquire = "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER"
    acquired = materialize_live_request(
        operation_kind=acquire,
        invocation=_invocation(acquire),
        live_sources={"snapshot_cleanup_control": _armed_control()},
        ports=ports,
    )
    owned = acquired["plan"]["cleanup_control"]["after"]
    capsule = acquired["owner_nonce_capsule"]
    reconciling = validate_record(
        "glm52_production_snapshot_cleanup_control",
        {
            **owned,
            "state": "DELETE_RECONCILING",
            "cleanup_authority_audit_identities": ["1" * 64],
            "latest_cleanup_authority_audit_identity_sha256": "1" * 64,
            "delete_action_identities": ["2" * 64],
            "latest_delete_action_identity_sha256": "2" * 64,
            "delete_logical_attempt": 1,
            "delete_call_count": 1,
            "latest_delete_request_id": "delete-1",
            "latest_delete_response_sha256": "3" * 64,
            "last_describe_request_id": "describe-1",
            "last_describe_response_sha256": "4" * 64,
            "revision": owned["revision"] + 2,
            "updated_at": "2026-08-04T12:00:00Z",
        },
    )
    operation = "SNAPSHOT_CLEANUP_RECORD_TERMINAL_EVIDENCE"
    request = materialize_live_request(
        operation_kind=operation,
        invocation=_invocation(
            operation,
            capsule=capsule,
            domain_result={
                "outcome": "EXACT_DURABLE_ADOPTION",
            },
            retained_reconciliation={
                "source_state": "DELETE_POSSIBLY_SENT",
                "target_state": "DELETE_RECONCILING",
                "committed": True,
                "snapshot_present": False,
                "delete_logical_attempt": 1,
                "resolution": {"outcome": "EXACT_LIVE_OWNER_COMMIT"},
            },
        ),
        live_sources={"snapshot_cleanup_control": reconciling},
        ports=ports,
    )

    after = request["plan"]["cleanup_control"]["after"]
    assert after["state"] == "DELETED"
    assert after["terminal_evidence_sha256"] is not None
    assert after["owner_execution_arn"] is None
    assert after["owner_invocation_nonce_sha256"] is None
    assert request["owner_nonce_capsule"] == capsule
    assert "raw_owner_nonce_hex" not in request


def test_close_ambiguous_attempt_updates_only_current_consumed_action() -> None:
    from glm52_enforcement.records import validate_record
    from glm52_enforcement.task12_live_snapshot_finalization import (
        materialize_live_request,
    )

    kms = _Kms()
    ports = _Ports(records=_authorities(), kms=kms)
    acquire = "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER"
    acquired = materialize_live_request(
        operation_kind=acquire,
        invocation=_invocation(acquire),
        live_sources={"snapshot_cleanup_control": _armed_control()},
        ports=ports,
    )
    owned = acquired["plan"]["cleanup_control"]["after"]
    capsule = acquired["owner_nonce_capsule"]
    reconciling = validate_record(
        "glm52_production_snapshot_cleanup_control",
        {
            **owned,
            "state": "DELETE_RECONCILING",
            "cleanup_authority_audit_identities": ["1" * 64],
            "latest_cleanup_authority_audit_identity_sha256": "1" * 64,
            "delete_action_identities": ["2" * 64],
            "latest_delete_action_identity_sha256": "2" * 64,
            "delete_logical_attempt": 1,
            "delete_call_count": 1,
            "last_describe_request_id": "describe-1",
            "last_describe_response_sha256": "4" * 64,
            "revision": owned["revision"] + 2,
            "updated_at": "2026-08-04T12:00:00Z",
        },
    )
    fixtures = _record_fixtures()
    action = fixtures._closed_record(
        "glm52_production_snapshot_cleanup_action",
        activation_id=ACTIVATION_ID,
        state="CONSUMED",
        attempt=1,
        owner_attempt=reconciling["owner_attempt"],
        owner_execution_arn=reconciling["owner_execution_arn"],
        owner_state_machine_version_arn=reconciling[
            "owner_state_machine_version_arn"
        ],
        owner_dispatch_identity_sha256=reconciling[
            "owner_dispatch_identity_sha256"
        ],
        owner_invocation_nonce_sha256=reconciling[
            "owner_invocation_nonce_sha256"
        ],
        owner_hard_expires_at=reconciling["owner_hard_expires_at"],
        authority_barrier_nonce_sha256=reconciling[
            "cleanup_barrier_nonce_sha256"
        ],
    )
    operation = "SNAPSHOT_CLEANUP_CLOSE_AMBIGUOUS_ATTEMPT"
    request = materialize_live_request(
        operation_kind=operation,
        invocation=_invocation(
            operation,
            capsule=capsule,
            domain_result={
                "outcome": "EXACT_DURABLE_ADOPTION",
            },
            retained_reconciliation={
                "source_state": "DELETE_POSSIBLY_SENT",
                "target_state": "DELETE_RECONCILING",
                "committed": True,
                "snapshot_present": True,
                "delete_logical_attempt": 1,
                "resolution": {"outcome": "EXACT_LIVE_OWNER_COMMIT"},
            },
        ),
        live_sources={
            "snapshot_cleanup_control": reconciling,
            "snapshot_cleanup_action": action,
        },
        ports=ports,
    )

    assert set(request["plan"]) == {"index", "control", "action"}
    assert request["plan"]["control"]["expected"] == reconciling
    assert request["plan"]["action"]["before"]["state"] == "CONSUMED"
    assert request["plan"]["action"]["after"]["state"] == "AMBIGUOUS"
    assert request["plan"]["action"]["after"][
        "reconciliation_identity_sha256"
    ] == reconciling["last_describe_response_sha256"]
    assert request["owner_nonce_capsule"] == capsule


def test_arm_next_same_id_attempt_puts_attempt_two_without_control_update() -> None:
    from glm52_enforcement.records import validate_record
    from glm52_enforcement.task12_live_snapshot_finalization import (
        materialize_live_request,
    )

    kms = _Kms()
    ports = _Ports(records=_authorities(), kms=kms)
    acquire = "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER"
    acquired = materialize_live_request(
        operation_kind=acquire,
        invocation=_invocation(acquire),
        live_sources={"snapshot_cleanup_control": _armed_control()},
        ports=ports,
    )
    owned = acquired["plan"]["cleanup_control"]["after"]
    capsule = acquired["owner_nonce_capsule"]
    reconciling = validate_record(
        "glm52_production_snapshot_cleanup_control",
        {
            **owned,
            "state": "DELETE_RECONCILING",
            "cleanup_authority_audit_identities": ["1" * 64],
            "latest_cleanup_authority_audit_identity_sha256": "1" * 64,
            "delete_action_identities": ["2" * 64],
            "latest_delete_action_identity_sha256": "2" * 64,
            "delete_logical_attempt": 1,
            "delete_call_count": 1,
            "last_describe_request_id": "describe-1",
            "last_describe_response_sha256": "4" * 64,
            "revision": owned["revision"] + 2,
            "updated_at": "2026-08-04T12:00:00Z",
        },
    )
    operation = "SNAPSHOT_CLEANUP_ARM_NEXT_SAME_ID_ATTEMPT"
    request = materialize_live_request(
        operation_kind=operation,
        invocation=_invocation(
            operation,
            capsule=capsule,
            domain_result={
                "outcome": "EXACT_DURABLE_ADOPTION",
            },
            retained_reconciliation={
                "source_state": "DELETE_POSSIBLY_SENT",
                "target_state": "DELETE_RECONCILING",
                "committed": True,
                "snapshot_present": True,
                "delete_logical_attempt": 1,
                "resolution": {"outcome": "EXACT_LIVE_OWNER_COMMIT"},
            },
        ),
        live_sources={"snapshot_cleanup_control": reconciling},
        ports=ports,
    )

    assert set(request["plan"]) == {"index", "control", "action"}
    assert request["plan"]["control"]["expected"] == reconciling
    assert request["plan"]["action"]["item"]["attempt"] == 2
    assert request["plan"]["action"]["item"]["state"] == "ARMED"
    assert request["owner_nonce_capsule"] == capsule


def test_successor_hook_accepts_only_authenticated_committed_readback() -> None:
    import pytest

    from glm52_enforcement.dynamodb import (
        TransactionResolution,
        WriteOutcome,
    )
    from glm52_enforcement.task12_live_snapshot_finalization import (
        materialize_live_request,
        persist_live_successors,
    )
    from glm52_enforcement.task12_snapshot_cleanup import (
        SnapshotOwnerAcquisitionResult,
    )

    operation = "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER"
    kms = _Kms()
    ports = _Ports(records=_authorities(), kms=kms)
    invocation = _invocation(operation)
    live_sources = {"snapshot_cleanup_control": _armed_control()}
    request = materialize_live_request(
        operation_kind=operation,
        invocation=invocation,
        live_sources=live_sources,
        ports=ports,
    )
    records = tuple(
        member.get("after", member.get("item", member.get("expected")))
        for member in request["plan"].values()
        if member is not None
    )

    assert persist_live_successors(
        operation_kind=operation,
        invocation=invocation,
        live_sources=live_sources,
        request=request,
        domain_result=SnapshotOwnerAcquisitionResult(
            cleanup_state="OWNED",
            resolution=TransactionResolution(
                WriteOutcome.EXACT_LIVE_OWNER_COMMIT,
                records,
                "ddb-commit-1",
                None,
                (),
            ),
        ),
        ports=ports,
    )
    with pytest.raises(ValueError, match="authenticated committed readback"):
        persist_live_successors(
            operation_kind=operation,
            invocation=invocation,
            live_sources=live_sources,
            request=request,
            domain_result=SnapshotOwnerAcquisitionResult(
                cleanup_state="OWNED",
                resolution=TransactionResolution(
                    WriteOutcome.MISMATCHED_COMMIT,
                    (),
                    "ddb-mismatch-1",
                    None,
                    (),
                ),
            ),
            ports=ports,
        )


def test_snapshot_client_accepts_only_authenticated_exact_not_found() -> None:
    from glm52_enforcement.task12_lambda_adapters import _SnapshotClient

    snapshot_id = "snap-0123456789abcdef0"

    class Ec2:
        def __init__(self, retries: int) -> None:
            self.retries = retries

        def describe_snapshots(self, **request: object) -> object:
            assert request == {"SnapshotIds": [snapshot_id]}
            error = RuntimeError("not found")
            error.response = {
                "Error": {
                    "Code": "InvalidSnapshot.NotFound",
                    "Message": "The snapshot " + snapshot_id + " does not exist",
                },
                "ResponseMetadata": {
                    "HTTPStatusCode": 400,
                    "RequestId": "describe-not-found",
                    "RetryAttempts": self.retries,
                },
            }
            raise error

    ports = SimpleNamespace(client=lambda service: Ec2(0))
    assert _SnapshotClient(ports).describe_snapshot(
        snapshot_id=snapshot_id
    ) is None

    with pytest.raises(RuntimeError, match="not found"):
        _SnapshotClient(
            SimpleNamespace(client=lambda service: Ec2(1))
        ).describe_snapshot(snapshot_id=snapshot_id)
