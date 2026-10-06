from __future__ import annotations

import base64
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import pytest

from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256
from glm52_enforcement.dynamodb import (
    TransactionResolution,
    WriteOutcome,
    encode_item,
)
from glm52_enforcement.records import ledger_pk


SHA = "a" * 64
SHA_B = "b" * 64
ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
CALLER_ARN = (
    "arn:aws:states:us-west-2:246813579024:stateMachine:"
    "keep-glm52-h1g-retained:9"
)
SNAPSHOT_CALLER_ARN = (
    "arn:aws:states:us-west-2:246813579024:stateMachine:"
    "keep-glm52-h1g-snapshot-cleanup:11"
)
ROOT = Path(__file__).resolve().parent


CASES = (
    (
        "RETAINED_EXECUTION_OBSERVER",
        "RETAINED",
        "keep-glm52-h1g-execution-observer",
        (
            "authority",
            "campaign_bucket",
            "spend_runtime_prefix",
            "campaign_descriptor_key",
            "campaign_descriptor_version_id",
            "campaign_descriptor_file_sha256",
            "gpu_spend_approval_key",
            "gpu_spend_approval_version_id",
            "gpu_spend_approval_file_sha256",
            "kms_key_id",
            "ledger_table_name",
            "numeric_binding_version_arn",
            "task9_deployed_identity_coordinate",
            "task9_deployed_identity_sha256",
        ),
        "make_retained_execution_observer_coordinator",
        "glm52_task12_retained_execution_observer_input_v1",
    ),
    (
        "RETAINED_TERMINAL_V2",
        "RETAINED",
        "keep-glm52-h1g-terminal-v2-writer",
        (
            "authority",
            "campaign_bucket",
            "spend_runtime_prefix",
            "campaign_descriptor_key",
            "campaign_descriptor_version_id",
            "campaign_descriptor_file_sha256",
            "gpu_spend_approval_key",
            "gpu_spend_approval_version_id",
            "gpu_spend_approval_file_sha256",
            "kms_key_id",
            "ledger_table_name",
            "numeric_binding_version_arn",
            "task9_deployed_identity_coordinate",
            "task9_deployed_identity_sha256",
            "task10_worker_descriptor_coordinate",
            "terminal_evidence_prefix",
        ),
        "make_retained_terminal_v2_coordinator",
        "glm52_task12_retained_terminal_v2_input_v1",
    ),
    (
        "RETAINED_FINALIZER",
        "RETAINED",
        "keep-glm52-h1g-finalizer",
        ("authority", "campaign_bucket", "ledger_table_name"),
        "make_retained_finalizer_coordinator",
        "glm52_task12_retained_finalizer_input_v1",
    ),
    (
        "RETAINED_H1G_DRAINED",
        "RETAINED",
        "keep-glm52-h1g-h1g-drained-writer",
        ("authority", "campaign_bucket", "ledger_table_name"),
        "make_retained_h1g_drained_coordinator",
        "glm52_task12_retained_h1g_drained_input_v1",
    ),
    (
        "RETAINED_WORKER_DRAIN",
        "RETAINED",
        "keep-glm52-h1g-worker-drain-signal",
        (
            "authority",
            "campaign_bucket",
            "spend_runtime_prefix",
            "campaign_descriptor_key",
            "campaign_descriptor_version_id",
            "campaign_descriptor_file_sha256",
            "gpu_spend_approval_key",
            "gpu_spend_approval_version_id",
            "gpu_spend_approval_file_sha256",
            "kms_key_id",
            "ledger_table_name",
            "numeric_binding_version_arn",
            "task9_deployed_identity_coordinate",
            "task9_deployed_identity_sha256",
            "task10_worker_descriptor_coordinate",
            "worker_script_hashes",
            "worker_drain_document_name",
            "worker_drain_document_version",
        ),
        "make_retained_worker_drain_coordinator",
        "glm52_task12_retained_worker_drain_input_v1",
    ),
    (
        "RETAINED_OPERATOR_DISPOSITION",
        "RETAINED",
        "keep-glm52-h1g-operator-disposition-writer",
        (
            "authority",
            "campaign_bucket",
            "ledger_table_name",
            "numeric_binding_version_arn",
            "retained_cancellation_version_arn",
            "task9_deployed_identity_coordinate",
            "task9_deployed_identity_sha256",
        ),
        "make_retained_operator_disposition_coordinator",
        "glm52_task12_retained_operator_disposition_input_v1",
    ),
    (
        "RETAINED_ORPHAN_AUDIT",
        "RETAINED",
        "keep-glm52-h1g-orphan-audit",
        ("authority", "kms_key_id", "ledger_table_name"),
        "make_retained_orphan_audit_coordinator",
        "glm52_task12_retained_orphan_audit_input_v1",
    ),
    (
        "RETAINED_SNAPSHOT_CLEANUP",
        "SNAPSHOT_CLEANUP",
        "keep-glm52-h1g-snapshot-cleanup",
        (
            "authority",
            "kms_key_id",
            "ledger_table_name",
            "snapshot_cleanup_state_machine_arn",
            "snapshot_cleanup_state_machine_version",
            "snapshot_cleanup_schedule_invoke_role_arn",
            "snapshot_cleanup_schedule_group_name",
            "snapshot_cleanup_schedule_name",
        ),
        "make_retained_snapshot_cleanup_coordinator",
        "glm52_task12_retained_snapshot_cleanup_input_v1",
    ),
)

DEFAULT_OPERATIONS = {
    "RETAINED_EXECUTION_OBSERVER": "RECONCILE_RETAINED_LIFECYCLE_TRIGGER",
    "RETAINED_TERMINAL_V2": "RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2",
    "RETAINED_FINALIZER": "RETAINED_PUBLISH_SUPPORT_PLANE_FINALIZED",
    "RETAINED_H1G_DRAINED": "RETAINED_INVOKE_H1G_DRAINED_WRITER",
    "RETAINED_WORKER_DRAIN": "RETAINED_RECONCILE_WORKERS_AND_ALLOCATIONS",
    "RETAINED_OPERATOR_DISPOSITION": (
        "RETAINED_REQUEST_AND_JOB_CANCEL_RECONCILIATION"
    ),
    "RETAINED_ORPHAN_AUDIT": "RETAINED_AUDIT_SUPPORT_ORPHANS",
    "RETAINED_SNAPSHOT_CLEANUP": (
        "SNAPSHOT_CLEANUP_SEND_SAME_ID_OR_READ_BACK"
    ),
}


def _load_test_module(filename: str) -> object:
    spec = importlib.util.spec_from_file_location(
        "_task12_factory_fixture_" + filename.removesuffix(".py"),
        ROOT / filename,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WRITER_FIXTURES = _load_test_module("test_glm52_task12_writers.py")
RUNTIME_FIXTURES = _load_test_module("test_glm52_task12_runtime.py")
WORKER_FIXTURES = _load_test_module("test_glm52_task12_worker_drain.py")
ORPHAN_FIXTURES = _load_test_module("test_glm52_task12_orphan_audit.py")
SNAPSHOT_FIXTURES = _load_test_module("test_glm52_task12_snapshot_cleanup.py")


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _coordinates(names: tuple[str, ...]) -> dict[str, object]:
    from glm52_enforcement.task11_boundary import (
        build_task11_input_coordinate,
    )

    available: dict[str, object] = {
        "authority": {
            "bucket": "keep-glm52-retained",
            "key": "task12/activation-1/authority.json",
            "version_id": "authority-version-1",
            "file_sha256": SHA,
        },
        "campaign_bucket": "keep-glm52-campaign",
        "spend_runtime_prefix": (
            "campaigns/glm52-sky-20260724/runtime"
        ),
        "campaign_descriptor_key": (
            "campaigns/glm52-sky-20260724/submissions/production/"
            "descriptor.json"
        ),
        "campaign_descriptor_version_id": "descriptor-version-1",
        "campaign_descriptor_file_sha256": "d" * 64,
        "gpu_spend_approval_key": (
            "campaigns/glm52-sky-20260724/authorities/"
            "GPU_SPEND_APPROVAL.json"
        ),
        "gpu_spend_approval_version_id": "approval-version-1",
        "gpu_spend_approval_file_sha256": "e" * 64,
        "ledger_table_name": "keep-glm52-ledger",
        "numeric_binding_version_arn": (
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-h1g-numeric-binding:17"
        ),
        "retained_cancellation_version_arn": (
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-h1g-retained-cancellation:19"
        ),
        "task9_deployed_identity_coordinate": asdict(
            build_task11_input_coordinate(
                input_kind="TASK9_DEPLOYED_IDENTITY_COORDINATE",
                bucket="keep-glm52-campaign",
                key=(
                    "campaigns/glm52-sky-20260724/authorities/task9/"
                    "activation-1/TASK9_DEPLOYED_IDENTITY.json"
                ),
                version_id="task9-version-1",
                file_sha256="b" * 64,
                body_sha256="c" * 64,
            )
        ),
        "task9_deployed_identity_sha256": "c" * 64,
        "task10_worker_descriptor_coordinate": {
            "bucket": "keep-glm52-campaign",
            "key": "task13/production/task10-worker-descriptor.json",
            "version_id": "task10-worker-version-1",
            "file_sha256": "d" * 64,
            "body_sha256": "e" * 64,
        },
        "worker_script_hashes": {
            "glm52_checkpoint_commit.py": "1" * 64,
            "glm52_drain_and_stop.py": "2" * 64,
            "glm52_deadline_guard.py": "3" * 64,
        },
        "terminal_evidence_prefix": (
            "campaigns/glm52-sky-20260724/submissions/production/"
            "generations/00000001/terminal-evidence/"
        ),
        "state_machine_execution_arn": (
            "arn:aws:states:us-west-2:246813579024:execution:"
            "keep-glm52-h1g-retained:activation-1"
        ),
        "worker_drain_document_name": "KeepGlm52GracefulStopV1",
        "worker_drain_document_version": "7",
        "kms_key_id": (
            "arn:aws:kms:us-west-2:246813579024:key/"
            "11111111-2222-3333-4444-555555555555"
        ),
        "snapshot_cleanup_state_machine_arn": (
            "arn:aws:states:us-west-2:246813579024:stateMachine:"
            "keep-glm52-h1g-snapshot-cleanup"
        ),
        "snapshot_cleanup_state_machine_version": "11",
        "snapshot_cleanup_schedule_invoke_role_arn": (
            "arn:aws:iam::246813579024:role/"
            "keep-glm52-h1g-snapshot-cleanup-schedule-invoke"
        ),
        "snapshot_cleanup_schedule_group_name": "default",
        "snapshot_cleanup_schedule_name": (
            "keep-glm52-h1g-snapshot-cleanup-activation-1"
        ),
    }
    return {name: available[name] for name in names}


def _state_sort_key(kind: str, operation_kind: str) -> str:
    return (
        "ACTIVATION#activation-1#TASK12_LAMBDA_INPUT#"
        + kind
        + "#"
        + operation_kind
        + "#00000001"
    )


def _with_hash(body: Mapping[str, object]) -> dict[str, object]:
    return {
        **body,
        "canonical_body_sha256": canonical_sha256(body),
    }


def _operation_input(operation_kind: str) -> dict[str, object]:
    from glm52_enforcement import task12_lambda_adapters as adapters

    predecessor = adapters._OPERATION_PREDECESSOR[operation_kind]
    if predecessor is None:
        return {}
    prior_handler = next(
        kind
        for kind, operations in adapters._OPERATIONS.items()
        if predecessor in operations
    )
    prior_body = {
        "schema_version": 1,
        "record_type": "glm52_task12_lambda_result_v1",
        "handler_kind": prior_handler,
        "operation_kind": predecessor,
        "operation_input_identity_sha256": canonical_sha256({}),
        "outcome": "SUCCEEDED",
        "result": {},
    }
    return {
        "task12_last_result": {
            **prior_body,
            "canonical_body_sha256": canonical_sha256(prior_body),
        }
    }


def _writer_authority(
    *,
    writer_kind: str,
    authority_domain: str,
    record: Mapping[str, object],
) -> tuple[object, object]:
    from glm52_enforcement.task12_writers import (
        build_retained_writer_candidate,
    )

    candidate = build_retained_writer_candidate(
        writer_kind=writer_kind,
        campaign_bucket="keep-glm52-campaign",
        activation_id="activation-1",
        generation=1,
        authority_domain=authority_domain,
        record=dict(record),
    )
    return WRITER_FIXTURES._authorities(candidate)


def _execution_input() -> dict[str, object]:
    scan = RUNTIME_FIXTURES._scan(observed_at="2026-07-29T12:00:00Z")
    return {
        "correlation_identity_sha256": SHA,
        "runtime_scan": _jsonable(scan),
    }


def _terminal_input() -> dict[str, object]:
    record = WRITER_FIXTURES._terminal_v2()
    action, audit = _writer_authority(
        writer_kind="TerminalV2",
        authority_domain="RECOVERY",
        record=record,
    )
    return {
        "base_record": record,
        "allocations": record["allocations"],
        "worker_launch_evidence": record["worker_launch_evidence"],
        "worker_launch_liabilities": record[
            "worker_launch_liabilities"
        ],
        "request_evidence": record["request_evidence"],
        "late_allocations": [],
        "action": _jsonable(action),
        "audit": _jsonable(audit),
    }


def _finalizer_input() -> dict[str, object]:
    record = WRITER_FIXTURES._simple_record(
        "glm52_production_support_plane_finalized"
    )
    action, audit = _writer_authority(
        writer_kind="SupportPlaneFinalized",
        authority_domain="FINALIZATION",
        record=record,
    )
    return {
        "first_runtime_scan": _jsonable(
            RUNTIME_FIXTURES._scan(observed_at="2026-07-29T12:00:00Z")
        ),
        "second_runtime_scan": _jsonable(
            RUNTIME_FIXTURES._scan(observed_at="2026-07-29T12:01:00Z")
        ),
        "minimum_quiet_seconds": 60,
        "record": record,
        "action": _jsonable(action),
        "audit": _jsonable(audit),
    }


def _h1g_input() -> dict[str, object]:
    from glm52_enforcement.task12_orphan_audit import MarkerLastPrerequisites

    prerequisites = MarkerLastPrerequisites(
        terminal_v2_identity_sha256=SHA,
        finalization_identity_sha256=SHA_B,
        snapshot_cleanup_control_identity_sha256=SHA,
        controller_quiesced_identity_sha256=SHA_B,
        spend_ledger_head_identity_sha256=SHA,
        orphan_audit=ORPHAN_FIXTURES._audit(),
        support_stack_absent=True,
        snapshot_cleanup_armed=True,
        all_workers_terminal=True,
        all_allocations_closed=True,
        liability_state="SETTLED",
        liability_identity_sha256=SHA_B,
    )
    record = WRITER_FIXTURES._h1g_drained()
    action, audit = _writer_authority(
        writer_kind="H1GDrained",
        authority_domain="FINALIZATION",
        record=record,
    )
    return {
        "prerequisites": _jsonable(prerequisites),
        "record": record,
        "action": _jsonable(action),
        "audit": _jsonable(audit),
    }


def _worker_input() -> dict[str, object]:
    candidate = WORKER_FIXTURES._candidate()
    authority = WORKER_FIXTURES._authority(candidate)
    return {
        "candidate": _jsonable(candidate),
        "action": _jsonable(authority.action),
        "audit": _jsonable(authority.audit),
    }


def _operator_input() -> dict[str, object]:
    record = WRITER_FIXTURES._operator_disposition()
    action, audit = _writer_authority(
        writer_kind="OperatorDisposition",
        authority_domain="OPERATOR_DISPOSITION",
        record=record,
    )
    return {
        "record": record,
        "action": _jsonable(action),
        "audit": _jsonable(audit),
    }


def _orphan_input() -> dict[str, object]:
    baseline_grant = replace(
        ORPHAN_FIXTURES._grant("baseline"),
        encryption_context_identity_sha256=canonical_sha256({}),
    )
    baseline = (baseline_grant,)
    direct = ORPHAN_FIXTURES._direct()
    service = ORPHAN_FIXTURES._service()
    return {
        "expected_retained": _jsonable(ORPHAN_FIXTURES._expected()),
        "inventory": _jsonable(ORPHAN_FIXTURES._inventory()),
        "retained_grant_baseline": _jsonable(baseline),
        "pre_cleanup_grants": _jsonable(
            ORPHAN_FIXTURES._grant_scan(
                baseline + (direct.grant, service.grant),
                observed_at="2026-07-29T12:01:00Z",
            )
        ),
        "direct_grants": _jsonable((direct,)),
        "service_grants": _jsonable((service,)),
        "settling_deadline": "2026-07-29T12:04:00Z",
    }


def _snapshot_input() -> dict[str, object]:
    plan = SNAPSHOT_FIXTURES._typed_plan(
        before_state="OWNED",
        after_state="DELETE_POSSIBLY_SENT",
        after_attempt=1,
        after_call_count=1,
    )
    return {
        "operation": "DELETE_ONCE",
        "plan": _jsonable(plan),
        "observed_at": "2026-08-04T12:00:00Z",
        "domain": "TASK12_SNAPSHOT_CLEANUP",
        "operation_identity_sha256": "9" * 64,
        "raw_owner_nonce_hex": (b"n" * 32).hex(),
    }


def _snapshot_arm_input() -> dict[str, object]:
    from glm52_enforcement.task12_snapshot_cleanup import SnapshotSchedule

    capture, _ = SNAPSHOT_FIXTURES._capture_and_schedule()
    schedule = SnapshotSchedule.create(
        capture=capture,
        schedule_arn=(
            "arn:aws:scheduler:us-west-2:246813579024:schedule/default/"
            "keep-glm52-h1g-snapshot-cleanup-activation-1"
        ),
    )
    plan = _jsonable(
        SNAPSHOT_FIXTURES._typed_plan(
            before_state="DORMANT",
            after_state="ARMED",
        )
    )
    for side in ("before", "after"):
        plan["cleanup_control"][side]["schedule_arn"] = schedule.schedule_arn
    return {
        "operation": "ARM_SCHEDULE",
        "plan": plan,
        "capture": _jsonable(capture),
        "schedule": _jsonable(schedule),
        "domain": "TASK12_SNAPSHOT_CLEANUP",
        "operation_identity_sha256": "8" * 64,
        "raw_owner_nonce_hex": (b"a" * 32).hex(),
    }


_INPUT_BUILDERS = {
    "RETAINED_EXECUTION_OBSERVER": _execution_input,
    "RETAINED_TERMINAL_V2": _terminal_input,
    "RETAINED_FINALIZER": _finalizer_input,
    "RETAINED_H1G_DRAINED": _h1g_input,
    "RETAINED_WORKER_DRAIN": _worker_input,
    "RETAINED_OPERATOR_DISPOSITION": _operator_input,
    "RETAINED_ORPHAN_AUDIT": _orphan_input,
    "RETAINED_SNAPSHOT_CLEANUP": _snapshot_input,
}


def _spy_contract(
    *,
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
    calls: list[str],
) -> None:
    targets = {
        "RETAINED_EXECUTION_OBSERVER": (
            "glm52_enforcement.task12_runtime",
            "collect_runtime_scan",
        ),
        "RETAINED_TERMINAL_V2": (
            "glm52_enforcement.task12_terminal",
            "assemble_terminal_v2",
        ),
        "RETAINED_FINALIZER": (
            "glm52_enforcement.task12_runtime",
            "prove_runtime_terminal",
        ),
        "RETAINED_H1G_DRAINED": (
            "glm52_enforcement.task12_orphan_audit",
            "build_h1g_drained_prerequisites",
        ),
        "RETAINED_WORKER_DRAIN": (
            "glm52_enforcement.task12_worker_drain",
            "dispatch_worker_drain",
        ),
        "RETAINED_OPERATOR_DISPOSITION": (
            "glm52_enforcement.task12_writers",
            "write_retained_candidate",
        ),
        "RETAINED_ORPHAN_AUDIT": (
            "glm52_enforcement.task12_orphan_audit",
            "audit_orphans",
        ),
    }
    if kind == "RETAINED_SNAPSHOT_CLEANUP":
        from glm52_enforcement.task12_snapshot_cleanup import (
            SnapshotCleanupCoordinator,
        )

        original_method = SnapshotCleanupCoordinator.delete_once

        def method_spy(self: object, **request: object) -> object:
            calls.append("delete_once")
            return original_method(self, **request)

        monkeypatch.setattr(
            SnapshotCleanupCoordinator,
            "delete_once",
            method_spy,
        )
        return
    module_name, function_name = targets[kind]
    module = __import__(module_name, fromlist=[function_name])
    original = getattr(module, function_name)

    def function_spy(*args: object, **request: object) -> object:
        calls.append(function_name)
        return original(*args, **request)

    monkeypatch.setattr(module, function_name, function_spy)


class _Body:
    def __init__(self, raw: bytes) -> None:
        self.raw = raw

    def read(self) -> bytes:
        return self.raw


class _AwsHarness:
    def __init__(
        self,
        *,
        kind: str,
        state_record_type: str,
        state_payload: Mapping[str, object],
        mode: str,
        function_name: str,
        coordinate_names: tuple[str, ...],
        foreign_authority_coordinate: bool = False,
        wrong_state: bool = False,
        operation_kind: str | None = None,
    ) -> None:
        from glm52_enforcement import task12_lambda_adapters as adapters

        self.calls: list[tuple[str, str]] = []
        self.mutations: list[str] = []
        self.operation_kind = operation_kind or DEFAULT_OPERATIONS[kind]
        self.caller_arn = (
            SNAPSHOT_CALLER_ARN
            if self.operation_kind.startswith("SNAPSHOT_CLEANUP_")
            else CALLER_ARN
        )
        state_key = _state_sort_key(kind, self.operation_kind)
        state_body = {
            "schema_version": 1,
            "record_type": (
                "glm52_task12_foreign_input_v1"
                if wrong_state
                else state_record_type
            ),
            "handler_kind": kind,
            "activation_id": "activation-1",
            "activation_ordinal": 1,
            "generation": 1,
            "generation_text": "00000001",
            "operation_kind": self.operation_kind,
            **dict(state_payload),
        }
        self.state = _with_hash(state_body)
        authority_key = (
            state_key + "#FOREIGN"
            if foreign_authority_coordinate
            else state_key
        )
        authority_body = {
            "schema_version": 1,
            "record_type": (
                "glm52_task12_lambda_invocation_authority_v1"
            ),
            "handler_kind": kind,
            "activation_id": "activation-1",
            "activation_ordinal": 1,
            "generation": 1,
            "generation_text": "00000001",
            "operations": {
                operation: {
                    "state_sort_key": (
                        authority_key
                        if operation == self.operation_kind
                        else _state_sort_key(kind, operation)
                    ),
                    "state_body_sha256": (
                        self.state["canonical_body_sha256"]
                        if operation == self.operation_kind
                        else SHA
                    ),
                }
                for operation in (
                    adapters._caller_edge_operations(kind, self.caller_arn)
                )
            },
        }
        self.authority = _with_hash(authority_body)
        self.authority_raw = canonical_json_bytes(self.authority)
        coordinates = _coordinates(coordinate_names)
        coordinates["authority"]["file_sha256"] = hashlib.sha256(
            self.authority_raw
        ).hexdigest()
        function_arn = (
            "arn:aws:lambda:us-west-2:246813579024:function:"
            + function_name
            + ":42"
        )
        config_body = {
            "schema_version": 1,
            "record_type": "glm52_task12_lambda_deployment_v1",
            "account_id": ACCOUNT_ID,
            "region": REGION,
            "handler_kind": kind,
            "mode": mode,
            "activation_id": "activation-1",
            "activation_ordinal": 1,
            "generation": 1,
            "generation_text": "00000001",
            "function_name": function_name,
            "function_version": "42",
            "invoked_function_version_arn": function_arn,
            "caller_state_machine_version_arn": self.caller_arn,
            "role_coordinates": coordinates,
        }
        config = _with_hash(config_body)
        raw_config = canonical_json_bytes(config).decode("ascii")
        self.raw_config = raw_config
        self.deployment = adapters._load_deployment(
            environ={"GLM52_TASK12_DEPLOYMENT_CONFIG": raw_config},
            expected_kind=kind,
        )
        self.invocation = adapters._parse_invocation(
            event={
                "activation_id": "activation-1",
                "activation_ordinal": 1,
                "generation": 1,
                "generation_text": "00000001",
                "dispatch_identity_sha256": SHA,
                "caller_state_machine_arn": self.caller_arn.rsplit(":", 1)[0],
                "state_machine_execution_arn": (
                    "arn:aws:states:us-west-2:246813579024:execution:"
                    + self.caller_arn.split(":stateMachine:", 1)[1].rsplit(
                        ":", 1
                    )[0]
                    + ":activation-1"
                ),
                "operation_kind": self.operation_kind,
                "operation_input": _operation_input(self.operation_kind),
            },
            context=SimpleNamespace(invoked_function_arn=function_arn),
            deployment=self.deployment,
        )

    def client(
        self,
        service: str,
        *,
        region_name: str,
        config: object,
    ) -> object:
        assert region_name == REGION
        assert config.retries == {
            "mode": "standard",
            "total_max_attempts": 1,
        }
        return {
            "s3": self,
            "dynamodb": self,
            "ssm": self,
            "ec2": self,
            "stepfunctions": self,
            "kms": self,
            "scheduler": self,
        }[service]

    def get_object(self, **request: object) -> object:
        self.calls.append(("s3", "get_object"))
        assert request["ExpectedBucketOwner"] == ACCOUNT_ID
        if request.get("VersionId") == "authority-version-1":
            return {
                "Body": _Body(self.authority_raw),
                "VersionId": "authority-version-1",
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "s3-authority-read",
                    "RetryAttempts": 0,
                },
            }
        return {
            "Body": _Body(self.writer_raw),
            "VersionId": "writer-version-1",
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "s3-writer-read",
            },
        }

    def get_item(self, **request: object) -> object:
        self.calls.append(("dynamodb", "get_item"))
        if request.get("ReturnConsumedCapacity") == "NONE":
            assert request["ConsistentRead"] is True
            return {
                "Item": encode_item(
                    {
                        "PK": ledger_pk(RUN_ID),
                        "SK": self.authority["operations"][
                            self.operation_kind
                        ]["state_sort_key"],
                        **self.state,
                    }
                ),
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "ddb-state-read",
                    "RetryAttempts": 0,
                },
            }
        raise AssertionError("unexpected DynamoDB read")

    def put_object(self, **request: object) -> object:
        self.calls.append(("s3", "put_object"))
        self.mutations.append("s3:put_object")
        assert request["IfNoneMatch"] == "*"
        self.writer_raw = request["Body"]
        return {
            "VersionId": "writer-version-1",
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "s3-writer-create",
                "RetryAttempts": 0,
            },
        }

    def describe_execution(self, **request: object) -> object:
        self.calls.append(("stepfunctions", "describe_execution"))
        assert request == {
            "executionArn": (
                "arn:aws:states:us-west-2:246813579024:execution:"
                + self.caller_arn.split(":stateMachine:", 1)[1].rsplit(
                    ":", 1
                )[0]
                + ":activation-1"
            )
        }
        return {
            "executionArn": request["executionArn"],
            "stateMachineArn": self.caller_arn.rsplit(":", 1)[0],
            "stateMachineVersionArn": self.caller_arn,
            "status": "SUCCEEDED",
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "states-describe-execution",
                "RetryAttempts": 0,
            },
        }

    def list_grants(self, **request: object) -> object:
        self.calls.append(("kms", "list_grants"))
        assert request == {
            "KeyId": self.deployment.role_coordinates["kms_key_id"],
            "Limit": 100,
        }
        return {
            "Grants": [
                {
                    "GrantId": "baseline",
                    "Name": "grant-baseline",
                    "GranteePrincipal": (
                        "arn:aws:iam::246813579024:role/grantee"
                    ),
                    "RetiringPrincipal": (
                        "arn:aws:iam::246813579024:role/retirer"
                    ),
                    "Operations": ["Decrypt"],
                    "Constraints": {},
                }
            ],
            "Truncated": False,
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "kms-list-grants",
                "RetryAttempts": 0,
            },
        }

    def create_schedule(self, **request: object) -> object:
        self.calls.append(("scheduler", "create_schedule"))
        self.mutations.append("scheduler:create_schedule")
        assert request == {
            "Name": "keep-glm52-h1g-snapshot-cleanup-activation-1",
            "GroupName": "default",
            "ScheduleExpression": "at(2026-08-04T12:00:00)",
            "FlexibleTimeWindow": {"Mode": "OFF"},
            "ActionAfterCompletion": "DELETE",
            "State": "ENABLED",
            "Target": {
                "Arn": (
                    "arn:aws:states:us-west-2:246813579024:stateMachine:"
                    "keep-glm52-h1g-snapshot-cleanup:11"
                ),
                "RoleArn": (
                    "arn:aws:iam::246813579024:role/"
                    "keep-glm52-h1g-snapshot-cleanup-schedule-invoke"
                ),
                "Input": "{}",
                "RetryPolicy": {
                    "MaximumEventAgeInSeconds": 60,
                    "MaximumRetryAttempts": 0,
                },
            },
            "ClientToken": "8" * 64,
        }
        return {
            "ScheduleArn": (
                "arn:aws:scheduler:us-west-2:246813579024:schedule/default/"
                "keep-glm52-h1g-snapshot-cleanup-activation-1"
            ),
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "scheduler-create",
                "RetryAttempts": 0,
            },
        }

    def get_schedule(self, **request: object) -> object:
        self.calls.append(("scheduler", "get_schedule"))
        return {
            "Arn": (
                "arn:aws:scheduler:us-west-2:246813579024:schedule/default/"
                "keep-glm52-h1g-snapshot-cleanup-activation-1"
            ),
            "Name": request["Name"],
            "GroupName": request["GroupName"],
            "ScheduleExpression": "at(2026-08-04T12:00:00)",
            "FlexibleTimeWindow": {"Mode": "OFF"},
            "ActionAfterCompletion": "DELETE",
            "State": "ENABLED",
            "Target": {
                "Arn": (
                    "arn:aws:states:us-west-2:246813579024:stateMachine:"
                    "keep-glm52-h1g-snapshot-cleanup:11"
                ),
                "RoleArn": (
                    "arn:aws:iam::246813579024:role/"
                    "keep-glm52-h1g-snapshot-cleanup-schedule-invoke"
                ),
                "Input": "{}",
                "RetryPolicy": {
                    "MaximumEventAgeInSeconds": 60,
                    "MaximumRetryAttempts": 0,
                },
            },
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "scheduler-get",
                "RetryAttempts": 0,
            },
        }

    def delete_schedule(self, **request: object) -> object:
        self.calls.append(("scheduler", "delete_schedule"))
        self.mutations.append("scheduler:delete_schedule")
        return {
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "scheduler-delete",
                "RetryAttempts": 0,
            }
        }

    def put_item(self, **request: object) -> object:
        self.calls.append(("dynamodb", "put_item"))
        self.mutations.append("dynamodb:put_item")
        assert "attribute_not_exists" in request["ConditionExpression"]
        return {
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "ddb-writer-create",
                "RetryAttempts": 0,
            }
        }

    def send_command(self, **request: object) -> object:
        self.calls.append(("ssm", "send_command"))
        self.mutations.append("ssm:send_command")
        return {
            "Command": {
                "CommandId": "11111111-2222-3333-4444-555555555555",
                "DocumentName": request["DocumentName"],
                "DocumentVersion": request["DocumentVersion"],
                "InstanceIds": request["InstanceIds"],
                "Status": "Pending",
            },
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "ssm-drain-send",
                "RetryAttempts": 0,
            },
        }

    def delete_snapshot(self, **request: object) -> object:
        self.calls.append(("ec2", "delete_snapshot"))
        self.mutations.append("ec2:delete_snapshot")
        return {
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "ec2-delete-snapshot",
                "RetryAttempts": 0,
            }
        }


@pytest.mark.parametrize(
    (
        "kind",
        "mode",
        "function_name",
        "coordinate_names",
        "factory_name",
        "state_record_type",
    ),
    CASES,
)
def test_all_named_factories_traverse_pinned_state_and_typed_boundary(
    kind: str,
    mode: str,
    function_name: str,
    coordinate_names: tuple[str, ...],
    factory_name: str,
    state_record_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from glm52_enforcement import task12_lambda_adapters as adapters

    contract_calls: list[str] = []
    _spy_contract(
        kind=kind,
        monkeypatch=monkeypatch,
        calls=contract_calls,
    )
    harness = _AwsHarness(
        kind=kind,
        state_record_type=state_record_type,
        state_payload=_INPUT_BUILDERS[kind](),
        mode=mode,
        function_name=function_name,
        coordinate_names=coordinate_names,
    )
    if kind == "RETAINED_SNAPSHOT_CLEANUP":
        mutations = harness.mutations

        class _FakeDynamoLedgerAdapter:
            def __init__(self, *, client: object, table_name: str) -> None:
                assert client._client is harness
                assert table_name == "keep-glm52-ledger"

            def commit_snapshot_cleanup_transition(
                self, **request: object
            ) -> TransactionResolution:
                assert request["plan"].cleanup_control.after["state"] == (
                    "DELETE_POSSIBLY_SENT"
                )
                mutations.append("dynamodb:transact_write_items")
                return TransactionResolution(
                    outcome=WriteOutcome.EXACT_LIVE_OWNER_COMMIT,
                    records=(),
                    request_id="ddb-cleanup-write",
                    error_code=None,
                    cancellation_reasons=(),
                )

        import glm52_enforcement.dynamodb as dynamodb

        monkeypatch.setattr(
            dynamodb,
            "DynamoLedgerAdapter",
            _FakeDynamoLedgerAdapter,
        )
    coordinator = getattr(adapters, factory_name)(
        harness.deployment,
        _session=harness,
    )

    result = coordinator(harness.invocation)

    assert result is not None
    assert len(contract_calls) == 1
    assert harness.calls[:2] == [
        ("s3", "get_object"),
        ("dynamodb", "get_item"),
    ]
    expected_mutations = {
        "RETAINED_EXECUTION_OBSERVER": [],
        "RETAINED_TERMINAL_V2": ["s3:put_object", "dynamodb:put_item"],
        "RETAINED_FINALIZER": ["s3:put_object", "dynamodb:put_item"],
        "RETAINED_H1G_DRAINED": ["s3:put_object", "dynamodb:put_item"],
        "RETAINED_WORKER_DRAIN": ["ssm:send_command"],
        "RETAINED_OPERATOR_DISPOSITION": ["dynamodb:put_item"],
        "RETAINED_ORPHAN_AUDIT": [],
        "RETAINED_SNAPSHOT_CLEANUP": [
            "dynamodb:transact_write_items",
            "ec2:delete_snapshot",
        ],
    }
    assert harness.mutations == expected_mutations[kind]


@pytest.mark.parametrize(
    (
        "kind",
        "mode",
        "function_name",
        "coordinate_names",
        "factory_name",
        "state_record_type",
    ),
    CASES,
)
def test_all_named_factories_reject_wrong_state_before_side_effect(
    kind: str,
    mode: str,
    function_name: str,
    coordinate_names: tuple[str, ...],
    factory_name: str,
    state_record_type: str,
) -> None:
    from glm52_enforcement import task12_lambda_adapters as adapters

    harness = _AwsHarness(
        kind=kind,
        state_record_type=state_record_type,
        state_payload=_INPUT_BUILDERS[kind](),
        mode=mode,
        function_name=function_name,
        coordinate_names=coordinate_names,
        wrong_state=True,
    )
    coordinator = getattr(adapters, factory_name)(
        harness.deployment,
        _session=harness,
    )

    with pytest.raises(
        adapters.Task12LambdaAdapterError,
        match="retained coordinator input is foreign",
    ):
        coordinator(harness.invocation)

    assert harness.mutations == []


@pytest.mark.parametrize(
    (
        "kind",
        "mode",
        "function_name",
        "coordinate_names",
        "factory_name",
        "state_record_type",
    ),
    CASES,
)
def test_all_named_factories_reject_foreign_authority_coordinate_before_state_read(
    kind: str,
    mode: str,
    function_name: str,
    coordinate_names: tuple[str, ...],
    factory_name: str,
    state_record_type: str,
) -> None:
    from glm52_enforcement import task12_lambda_adapters as adapters

    harness = _AwsHarness(
        kind=kind,
        state_record_type=state_record_type,
        state_payload=_INPUT_BUILDERS[kind](),
        mode=mode,
        function_name=function_name,
        coordinate_names=coordinate_names,
        foreign_authority_coordinate=True,
    )
    coordinator = getattr(adapters, factory_name)(
        harness.deployment,
        _session=harness,
    )

    with pytest.raises(
        adapters.Task12LambdaAdapterError,
        match="coordinator authority operation manifest drifted",
    ):
        coordinator(harness.invocation)

    assert harness.calls == [("s3", "get_object")]
    assert harness.mutations == []


def test_operation_scoped_state_rejects_cross_operation_replay_before_effect() -> None:
    from glm52_enforcement import task12_lambda_adapters as adapters

    kind, mode, function_name, coordinates, factory, record_type = CASES[4]
    harness = _AwsHarness(
        kind=kind,
        state_record_type=record_type,
        state_payload=_INPUT_BUILDERS[kind](),
        mode=mode,
        function_name=function_name,
        coordinate_names=coordinates,
    )
    foreign_operation = "RETAINED_QUIESCE_CONTROLLER_IF_REQUIRED"
    state_body = {
        name: value
        for name, value in harness.state.items()
        if name != "canonical_body_sha256"
    }
    state_body["operation_kind"] = foreign_operation
    harness.state = _with_hash(state_body)
    authority_body = {
        name: value
        for name, value in harness.authority.items()
        if name != "canonical_body_sha256"
    }
    authority_body["operations"][harness.operation_kind][
        "state_body_sha256"
    ] = harness.state["canonical_body_sha256"]
    harness.authority = _with_hash(authority_body)
    harness.authority_raw = canonical_json_bytes(harness.authority)
    harness.deployment = replace(
        harness.deployment,
        authority=replace(
            harness.deployment.authority,
            file_sha256=hashlib.sha256(harness.authority_raw).hexdigest(),
        ),
    )
    coordinator = getattr(adapters, factory)(
        harness.deployment,
        _session=harness,
    )

    with pytest.raises(
        adapters.Task12LambdaAdapterError,
        match="retained coordinator input is foreign",
    ):
        coordinator(harness.invocation)

    assert harness.mutations == []


def test_handler_rejects_cross_handler_operation_before_factory_or_effect() -> None:
    from glm52_enforcement import task12_lambda_adapters as adapters

    kind, mode, function_name, coordinates, factory_name, record_type = CASES[0]
    harness = _AwsHarness(
        kind=kind,
        state_record_type=record_type,
        state_payload=_INPUT_BUILDERS[kind](),
        mode=mode,
        function_name=function_name,
        coordinate_names=coordinates,
    )
    factory_calls = 0

    def factory(deployment: object) -> object:
        nonlocal factory_calls
        factory_calls += 1
        return getattr(adapters, factory_name)(deployment, _session=harness)

    handler = adapters.build_lambda_handler(
        handler_kind=kind,
        coordinator_factory=factory,
    )
    event = {
        "activation_id": "activation-1",
        "activation_ordinal": 1,
        "generation": 1,
        "generation_text": "00000001",
        "dispatch_identity_sha256": SHA,
        "caller_state_machine_arn": CALLER_ARN.rsplit(":", 1)[0],
        "state_machine_execution_arn": (
            "arn:aws:states:us-west-2:246813579024:execution:"
            "keep-glm52-h1g-retained:activation-1"
        ),
        "operation_kind": "RETAINED_AUDIT_SUPPORT_ORPHANS",
        "operation_input": {},
    }

    result = handler(
        event,
        SimpleNamespace(
            invoked_function_arn=harness.deployment.invoked_function_version_arn
        ),
        _environ={"GLM52_TASK12_DEPLOYMENT_CONFIG": harness.raw_config},
        _deployment_session=harness,
    )

    assert result["outcome"] == "REJECTED"
    assert factory_calls == 0
    assert harness.calls == []
    assert harness.mutations == []


@pytest.mark.parametrize(
    (
        "kind",
        "mode",
        "function_name",
        "coordinate_names",
        "factory_name",
        "state_record_type",
    ),
    CASES,
)
def test_all_lambda_adapters_validate_context_before_factory_or_aws(
    kind: str,
    mode: str,
    function_name: str,
    coordinate_names: tuple[str, ...],
    factory_name: str,
    state_record_type: str,
) -> None:
    from glm52_enforcement import task12_lambda_adapters as adapters

    harness = _AwsHarness(
        kind=kind,
        state_record_type=state_record_type,
        state_payload=_INPUT_BUILDERS[kind](),
        mode=mode,
        function_name=function_name,
        coordinate_names=coordinate_names,
    )
    factory_calls = 0

    def production_factory(deployment: object) -> object:
        nonlocal factory_calls
        factory_calls += 1
        return getattr(adapters, factory_name)(
            deployment,
            _session=harness,
        )

    handler = adapters.build_lambda_handler(
        handler_kind=kind,
        coordinator_factory=production_factory,
    )
    event = {
        "activation_id": "activation-1",
        "activation_ordinal": 1,
        "generation": 1,
        "generation_text": "00000001",
        "dispatch_identity_sha256": SHA,
        "caller_state_machine_arn": CALLER_ARN.rsplit(":", 1)[0],
        "state_machine_execution_arn": (
            "arn:aws:states:us-west-2:246813579024:execution:"
            "keep-glm52-h1g-retained:activation-1"
        ),
        "operation_kind": DEFAULT_OPERATIONS[kind],
        "operation_input": _operation_input(DEFAULT_OPERATIONS[kind]),
    }

    result = handler(
        event,
        SimpleNamespace(
            invoked_function_arn=(
                harness.deployment.invoked_function_version_arn[:-2] + "43"
            )
        ),
        _environ={
            "GLM52_TASK12_DEPLOYMENT_CONFIG": harness.raw_config
        },
    )

    assert result["outcome"] == "REJECTED"
    assert factory_calls == 0
    assert harness.calls == []
    assert harness.mutations == []


def test_s3_writer_closes_owner_version_and_authenticated_transport() -> None:
    from glm52_enforcement import task12_lambda_adapters as adapters
    from glm52_enforcement.task12_writers import (
        build_retained_writer_candidate,
    )

    requests: list[tuple[str, dict[str, object]]] = []
    candidate = build_retained_writer_candidate(
        writer_kind="SupportPlaneFinalized",
        campaign_bucket="keep-glm52-campaign",
        activation_id="activation-1",
        generation=1,
        authority_domain="FINALIZATION",
        record=WRITER_FIXTURES._simple_record(
            "glm52_production_support_plane_finalized"
        ),
    )

    class _S3:
        def put_object(self, **request: object) -> object:
            requests.append(("put_object", dict(request)))
            return {
                "VersionId": "writer-version-1",
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "writer-put",
                    "RetryAttempts": 0,
                },
            }

        def list_object_versions(self, **request: object) -> object:
            requests.append(("list_object_versions", dict(request)))
            return {
                "IsTruncated": False,
                "Versions": [
                    {
                        "Key": candidate.coordinate[5:].split("/", 1)[1],
                        "VersionId": "writer-version-1",
                        "IsLatest": True,
                        "ETag": '"writer-etag"',
                        "Size": len(candidate.raw),
                        "LastModified": datetime(
                            2026, 7, 29, 12, tzinfo=timezone.utc
                        ),
                    }
                ],
                "DeleteMarkers": [],
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "writer-list",
                    "RetryAttempts": 0,
                },
            }

        @staticmethod
        def _transport(request_id: str) -> dict[str, object]:
            exact = adapters._RetainedWriteBoundary._s3_candidate(candidate)
            checksum = base64.b64encode(
                hashlib.sha256(candidate.raw).digest()
            ).decode("ascii")
            return {
                "VersionId": "writer-version-1",
                "ETag": '"writer-etag"',
                "ContentLength": len(candidate.raw),
                "LastModified": datetime(
                    2026, 7, 29, 12, tzinfo=timezone.utc
                ),
                "ChecksumSHA256": checksum,
                "ChecksumType": "FULL_OBJECT",
                "ContentType": "application/json",
                "Metadata": dict(exact.metadata),
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": request_id,
                    "RetryAttempts": 0,
                },
            }

        def get_object(self, **request: object) -> object:
            requests.append(("get_object", dict(request)))

            class _Body:
                def read(self) -> bytes:
                    return candidate.raw

                def close(self) -> None:
                    return None

            return {**self._transport("writer-get"), "Body": _Body()}

        def head_object(self, **request: object) -> object:
            requests.append(("head_object", dict(request)))
            return self._transport("writer-head")

    s3 = _S3()

    class _Ports:
        def client(self, service: str) -> object:
            assert service == "s3"
            return s3

    boundary = adapters._RetainedWriteBoundary(_Ports())

    created = boundary.conditional_create(
        candidate=candidate,
        action=object(),
        audit=object(),
    )
    reconciled = boundary.reconcile_exact(candidate=candidate)

    assert created.authenticated is True
    assert reconciled.authenticated is True
    assert [name for name, _request in requests] == [
        "put_object",
        "list_object_versions",
        "get_object",
        "head_object",
    ]
    put = requests[0][1]
    assert put["ChecksumAlgorithm"] == "SHA256"
    assert put["ChecksumSHA256"] == base64.b64encode(
        hashlib.sha256(candidate.raw).digest()
    ).decode("ascii")
    assert put["Metadata"] == dict(
        adapters._RetainedWriteBoundary._s3_candidate(candidate).metadata
    )


@pytest.mark.parametrize(
    "retry_attempts",
    [None, 1],
    ids=["missing-retry-metadata", "nonzero-retry-metadata"],
)
def test_retained_dynamodb_input_rejects_unauthenticated_retry_metadata(
    retry_attempts: object,
) -> None:
    from glm52_enforcement import task12_lambda_adapters as adapters

    kind, mode, function_name, coordinates, factory, record_type = CASES[0]
    harness = _AwsHarness(
        kind=kind,
        state_record_type=record_type,
        state_payload=_INPUT_BUILDERS[kind](),
        mode=mode,
        function_name=function_name,
        coordinate_names=coordinates,
    )
    get_item = harness.get_item

    def unauthenticated_get_item(**request: object) -> object:
        response = get_item(**request)
        metadata = response["ResponseMetadata"]
        if retry_attempts is None:
            metadata.pop("RetryAttempts")
        else:
            metadata["RetryAttempts"] = retry_attempts
        return response

    harness.get_item = unauthenticated_get_item  # type: ignore[method-assign]
    coordinator = getattr(adapters, factory)(
        harness.deployment,
        _session=harness,
    )

    with pytest.raises(
        adapters.Task12LambdaAdapterError,
        match="retained coordinator input is unavailable",
    ):
        coordinator(harness.invocation)

    assert harness.mutations == []


def _dynamodb_writer_boundary(
    metadata: Mapping[str, object],
) -> tuple[object, SimpleNamespace]:
    from glm52_enforcement import task12_lambda_adapters as adapters

    record = {"record_type": "exact"}
    candidate = SimpleNamespace(
        coordinate="dynamodb://RUN#task/SK#result",
        record=record,
        raw=canonical_json_bytes(record) + b"\n",
        candidate_identity_sha256=SHA,
    )

    class _DynamoDB:
        def put_item(self, **request: object) -> object:
            assert request["ConditionExpression"] == (
                "attribute_not_exists(#pk) AND attribute_not_exists(#sk)"
            )
            return {"ResponseMetadata": dict(metadata)}

        def get_item(self, **request: object) -> object:
            assert request["ConsistentRead"] is True
            return {
                "Item": encode_item(
                    {
                        "PK": "RUN#task",
                        "SK": "SK#result",
                        **record,
                    }
                ),
                "ResponseMetadata": dict(metadata),
            }

    deployment = SimpleNamespace(
        role_coordinates={"ledger_table_name": "keep-glm52-ledger"}
    )
    ports = SimpleNamespace(
        deployment=deployment,
        client=lambda service: _DynamoDB(),
    )
    return adapters._RetainedWriteBoundary(ports), candidate


@pytest.mark.parametrize(
    "metadata",
    [
        {"HTTPStatusCode": 200, "RequestId": "ddb-put"},
        {
            "HTTPStatusCode": 200,
            "RequestId": "ddb-put",
            "RetryAttempts": 1,
        },
    ],
    ids=["missing-retry-metadata", "nonzero-retry-metadata"],
)
def test_retained_dynamodb_create_does_not_authenticate_retry_drift(
    metadata: Mapping[str, object],
) -> None:
    boundary, candidate = _dynamodb_writer_boundary(metadata)

    response = boundary.conditional_create(
        candidate=candidate,
        action=object(),
        audit=object(),
    )

    assert response.classification == "AMBIGUOUS"
    assert response.authenticated is False


@pytest.mark.parametrize(
    "metadata",
    [
        {"HTTPStatusCode": 200, "RequestId": "ddb-get"},
        {
            "HTTPStatusCode": 200,
            "RequestId": "ddb-get",
            "RetryAttempts": 1,
        },
    ],
    ids=["missing-retry-metadata", "nonzero-retry-metadata"],
)
def test_retained_dynamodb_reconcile_does_not_authenticate_retry_drift(
    metadata: Mapping[str, object],
) -> None:
    boundary, candidate = _dynamodb_writer_boundary(metadata)

    response = boundary.reconcile_exact(candidate=candidate)

    assert response.authenticated is False


@pytest.mark.parametrize(
    "retry_attempts",
    [None, 1],
    ids=["missing-retry-metadata", "nonzero-retry-metadata"],
)
def test_worker_drain_rejects_unauthenticated_retry_metadata(
    retry_attempts: object,
) -> None:
    from glm52_enforcement import task12_lambda_adapters as adapters
    from glm52_enforcement.task12_worker_drain import Task12WorkerDrainError

    kind, mode, function_name, coordinates, factory, record_type = CASES[4]
    harness = _AwsHarness(
        kind=kind,
        state_record_type=record_type,
        state_payload=_INPUT_BUILDERS[kind](),
        mode=mode,
        function_name=function_name,
        coordinate_names=coordinates,
    )
    send_command = harness.send_command

    def unauthenticated_send_command(**request: object) -> object:
        response = send_command(**request)
        metadata = response["ResponseMetadata"]
        if retry_attempts is None:
            metadata.pop("RetryAttempts")
        else:
            metadata["RetryAttempts"] = retry_attempts
        return response

    harness.send_command = unauthenticated_send_command  # type: ignore[method-assign]
    coordinator = getattr(adapters, factory)(
        harness.deployment,
        _session=harness,
    )

    with pytest.raises(
        Task12WorkerDrainError,
        match="exact drain request",
    ):
        coordinator(harness.invocation)


@pytest.mark.parametrize(
    ("operation", "retry_attempts"),
    [
        ("delete", None),
        ("delete", 1),
        ("describe", None),
        ("describe", 1),
    ],
    ids=[
        "delete-missing-retry",
        "delete-nonzero-retry",
        "describe-missing-retry",
        "describe-nonzero-retry",
    ],
)
def test_snapshot_client_rejects_unauthenticated_retry_metadata(
    operation: str,
    retry_attempts: object,
) -> None:
    from glm52_enforcement import task12_lambda_adapters as adapters

    metadata: dict[str, object] = {
        "HTTPStatusCode": 200,
        "RequestId": "ec2-snapshot",
    }
    if retry_attempts is not None:
        metadata["RetryAttempts"] = retry_attempts

    class _Ec2:
        def delete_snapshot(self, **request: object) -> object:
            assert request == {"SnapshotId": "snap-0123456789abcdef0"}
            return {"ResponseMetadata": dict(metadata)}

        def describe_snapshots(self, **request: object) -> object:
            assert request == {"SnapshotIds": ["snap-0123456789abcdef0"]}
            return {
                "Snapshots": [],
                "ResponseMetadata": dict(metadata),
            }

    ports = SimpleNamespace(client=lambda service: _Ec2())
    client = adapters._SnapshotClient(ports)

    with pytest.raises(
        adapters.Task12LambdaAdapterError,
        match="snapshot .* response is unauthenticated",
    ):
        if operation == "delete":
            client.delete_snapshot(snapshot_id="snap-0123456789abcdef0")
        else:
            client.describe_snapshot(snapshot_id="snap-0123456789abcdef0")


@pytest.mark.parametrize(
    "retry_attempts",
    [None, 1],
    ids=["missing-retry-metadata", "nonzero-retry-metadata"],
)
def test_snapshot_dynamodb_transactions_reject_retry_metadata_drift(
    retry_attempts: object,
) -> None:
    from glm52_enforcement import task12_lambda_adapters as adapters

    metadata: dict[str, object] = {
        "HTTPStatusCode": 200,
        "RequestId": "ddb-transaction",
    }
    if retry_attempts is not None:
        metadata["RetryAttempts"] = retry_attempts

    class _DynamoDB:
        def transact_write_items(self, **request: object) -> object:
            return {"ResponseMetadata": dict(metadata)}

    client = adapters._AuthenticatedDynamoClient(_DynamoDB())

    with pytest.raises(
        adapters.Task12LambdaAdapterError,
        match="DynamoDB transact_write_items response is unauthenticated",
    ):
        client.transact_write_items(TransactItems=[])


def test_execution_observer_reads_exact_live_state_machine_execution() -> None:
    from glm52_enforcement import task12_lambda_adapters as adapters

    kind, mode, function_name, coordinates, factory, record_type = CASES[0]
    harness = _AwsHarness(
        kind=kind,
        state_record_type=record_type,
        state_payload=_INPUT_BUILDERS[kind](),
        mode=mode,
        function_name=function_name,
        coordinate_names=coordinates,
    )

    coordinator = getattr(adapters, factory)(
        harness.deployment,
        _session=harness,
    )
    coordinator(harness.invocation)

    assert ("stepfunctions", "describe_execution") in harness.calls


def test_execution_observer_reuses_common_authenticated_execution_once() -> None:
    from glm52_enforcement import task12_lambda_adapters as adapters

    kind, mode, function_name, coordinates, factory, record_type = CASES[0]
    harness = _AwsHarness(
        kind=kind,
        state_record_type=record_type,
        state_payload=_INPUT_BUILDERS[kind](),
        mode=mode,
        function_name=function_name,
        coordinate_names=coordinates,
    )
    invocation = replace(
        harness.invocation,
        authenticated_execution={
            "execution_arn": harness.invocation.state_machine_execution_arn,
            "state_machine_arn": harness.invocation.caller_state_machine_arn,
            "state_machine_version_arn": CALLER_ARN,
            "status": "RUNNING",
            "request_id": "common-authentication-read",
        },
    )

    coordinator = getattr(adapters, factory)(
        harness.deployment,
        _session=harness,
    )
    coordinator(invocation)

    assert ("stepfunctions", "describe_execution") not in harness.calls


def test_orphan_auditor_uses_live_paginated_kms_grants() -> None:
    from glm52_enforcement import task12_lambda_adapters as adapters

    kind, mode, function_name, coordinates, factory, record_type = CASES[6]
    harness = _AwsHarness(
        kind=kind,
        state_record_type=record_type,
        state_payload=_INPUT_BUILDERS[kind](),
        mode=mode,
        function_name=function_name,
        coordinate_names=coordinates,
    )
    requests: list[dict[str, object]] = []

    def grant(grant_id: str) -> dict[str, object]:
        return {
            "GrantId": grant_id,
            "Name": "grant-" + grant_id,
            "GranteePrincipal": (
                "arn:aws:iam::246813579024:role/grantee"
            ),
            "RetiringPrincipal": (
                "arn:aws:iam::246813579024:role/retirer"
            ),
            "Operations": ["Decrypt"],
            "Constraints": {},
        }

    def list_grants(**request: object) -> object:
        requests.append(dict(request))
        if "Marker" not in request:
            return {
                "Grants": [grant("z-retained")],
                "Truncated": True,
                "NextMarker": "page-2",
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "kms-page-1",
                    "RetryAttempts": 0,
                },
            }
        assert request["Marker"] == "page-2"
        return {
            "Grants": [grant("a-retained")],
            "Truncated": False,
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "kms-page-2",
                "RetryAttempts": 0,
            },
        }

    harness.list_grants = list_grants  # type: ignore[method-assign]

    result = adapters._read_live_kms_grants(
        adapters._ports(harness.deployment, harness)
    )

    assert requests == [
        {
            "KeyId": harness.deployment.role_coordinates["kms_key_id"],
            "Limit": 100,
        },
        {
            "KeyId": harness.deployment.role_coordinates["kms_key_id"],
            "Limit": 100,
            "Marker": "page-2",
        },
    ]
    assert [grant.grant_id for grant in result.grants] == [
        "a-retained",
        "z-retained",
    ]
    assert result.list_grants_identity_sha256 != SHA


@pytest.mark.parametrize("next_marker", [None, "", 7])
def test_live_kms_rejects_truncated_page_without_exact_next_marker(
    next_marker: object,
) -> None:
    from glm52_enforcement import task12_lambda_adapters as adapters

    class _Kms:
        def list_grants(self, **request: object) -> object:
            response: dict[str, object] = {
                "Grants": [],
                "Truncated": True,
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "kms-truncated",
                    "RetryAttempts": 0,
                },
            }
            if next_marker is not None:
                response["NextMarker"] = next_marker
            return response

    ports = SimpleNamespace(
        deployment=SimpleNamespace(
            role_coordinates={
                "kms_key_id": (
                    "arn:aws:kms:us-west-2:246813579024:key/"
                    "11111111-2222-3333-4444-555555555555"
                )
            }
        ),
        client=lambda service: _Kms(),
    )

    with pytest.raises(
        adapters.Task12LambdaAdapterError,
        match="pagination is not exact",
    ):
        adapters._read_live_kms_grants(ports)


def test_snapshot_adapter_arms_exact_one_time_scheduler_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from glm52_enforcement import task12_lambda_adapters as adapters

    kind, mode, function_name, coordinates, factory, record_type = CASES[7]
    harness = _AwsHarness(
        kind=kind,
        state_record_type=record_type,
        state_payload=_snapshot_arm_input(),
        mode=mode,
        function_name=function_name,
        coordinate_names=coordinates,
        operation_kind=(
            "RETAINED_ARM_SNAPSHOT_CLEANUP_CONTROL_AND_SCHEDULE"
        ),
    )

    class _FakeDynamoLedgerAdapter:
        def __init__(self, *, client: object, table_name: str) -> None:
            assert client._client is harness
            assert table_name == "keep-glm52-ledger"

        def commit_snapshot_cleanup_transition(
            self, **request: object
        ) -> TransactionResolution:
            assert request["plan"].cleanup_control.after["state"] == "ARMED"
            harness.mutations.append("dynamodb:transact_write_items")
            return TransactionResolution(
                outcome=WriteOutcome.EXACT_LIVE_OWNER_COMMIT,
                records=(),
                request_id="ddb-arm-schedule",
                error_code=None,
                cancellation_reasons=(),
            )

    import glm52_enforcement.dynamodb as dynamodb

    monkeypatch.setattr(
        dynamodb,
        "DynamoLedgerAdapter",
        _FakeDynamoLedgerAdapter,
    )
    coordinator = getattr(adapters, factory)(
        harness.deployment,
        _session=harness,
    )

    result = coordinator(harness.invocation)

    assert result["schedule"]["outcome"] == "CREATED"
    assert harness.mutations == [
        "dynamodb:transact_write_items",
        "scheduler:create_schedule",
    ]


def test_snapshot_scheduler_reconciles_then_deletes_only_exact_schedule() -> None:
    from glm52_enforcement import task12_lambda_adapters as adapters
    from glm52_enforcement.task12_snapshot_cleanup import SnapshotSchedule

    kind, mode, function_name, coordinates, _, record_type = CASES[7]
    state = _snapshot_arm_input()
    harness = _AwsHarness(
        kind=kind,
        state_record_type=record_type,
        state_payload=state,
        mode=mode,
        function_name=function_name,
        coordinate_names=coordinates,
    )
    scheduler = adapters._SchedulerClient(
        adapters._ports(harness.deployment, harness)
    )
    schedule = adapters._materialize_dataclass(
        SnapshotSchedule, state["schedule"]
    )

    result = scheduler.delete(schedule=schedule)

    assert result["outcome"] == "DELETED"
    assert harness.calls == [
        ("scheduler", "get_schedule"),
        ("scheduler", "delete_schedule"),
    ]
    assert harness.mutations == ["scheduler:delete_schedule"]
