from __future__ import annotations

import copy
from dataclasses import fields
import json
from pathlib import Path

import pytest


def _inputs() -> object:
    from glm52_enforcement.task12_support_plane import Task12SupportPlaneInputs

    return Task12SupportPlaneInputs(
        activation_id="h1g-act-20260729-0001",
        lambda_code_sha256="a" * 64,
        lambda_code_bucket="keep-glm52-runtime-artifacts",
        lambda_code_key="releases/task12-runtime.zip",
        lambda_code_version="3HL4kqtJlcpXroDTDmJ+rmSpXd3Ly3B/",
        ledger_table_arn=(
            "arn:aws:dynamodb:us-west-2:246813579024:table/keep-glm52-h1g-ledger-v1"
        ),
        retained_kms_key_arn=(
            "arn:aws:kms:us-west-2:246813579024:key/"
            "12345678-1234-4234-8234-1234567890ab"
        ),
        model_bucket_arn=(
            "arn:aws:s3:::"
            "keep-glm52-models-246813579024-us-west-2"
        ),
        worker_drain_document_arn=(
            "arn:aws:ssm:us-west-2:246813579024:document/KeepGlm52GracefulStopV1"
        ),
    )


def _published_versions() -> dict[str, str]:
    names = (
        "recovery_coordinator",
        "recovery_cancellation",
        "recovery_drain",
        "recovery_terminal_writer",
        "teardown_coordinator",
        "finalizer",
        "orphan_auditor",
        "snapshot_scheduler",
        "h1g_drained_writer",
        "snapshot_cleanup_coordinator",
        "snapshot_cleanup_writer",
    )
    return {
        name: (
            "arn:aws:lambda:us-west-2:246813579024:function:"
            f"keep-glm52-{name}:42"
        )
        for name in names
    }


_VERSION_SUBSTITUTIONS = {
    "ExecutionObserverVersionArn": "recovery_coordinator",
    "OperatorDispositionVersionArn": "recovery_cancellation",
    "WorkerDrainVersionArn": "recovery_drain",
    "TerminalV2VersionArn": "recovery_terminal_writer",
    "FinalizerVersionArn": "finalizer",
    "OrphanAuditVersionArn": "orphan_auditor",
    "SnapshotCleanupVersionArn": "snapshot_scheduler",
    "H1GDrainedVersionArn": "h1g_drained_writer",
}


def _production_versions() -> dict[str, str]:
    suffixes = {
        "recovery_coordinator": "execution-observer",
        "recovery_cancellation": "operator-disposition-writer",
        "recovery_drain": "worker-drain-signal",
        "recovery_terminal_writer": "terminal-v2-writer",
        "teardown_coordinator": "execution-observer",
        "finalizer": "finalizer",
        "orphan_auditor": "orphan-audit",
        "snapshot_scheduler": "snapshot-cleanup",
        "h1g_drained_writer": "h1g-drained-writer",
        "snapshot_cleanup_coordinator": "snapshot-cleanup",
        "snapshot_cleanup_writer": "snapshot-cleanup",
    }
    return {
        key: (
            "arn:aws:lambda:us-west-2:246813579024:function:"
            f"keep-glm52-h1g-{suffix}:42"
        )
        for key, suffix in suffixes.items()
    }


def _materialize_definition(definition: str) -> dict[str, object]:
    published = _production_versions()
    replacements = {
        "${" + substitution + "}": published[key]
        for substitution, key in _VERSION_SUBSTITUTIONS.items()
    }
    replacements["${SupportDeletionWorkflowVersionArn}"] = (
        "arn:aws:states:us-west-2:246813579024:stateMachine:"
        "keep-glm52-h1g-retained-support-delete:7"
    )

    def replace(value: object) -> object:
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, str):
            return replacements.get(value, value)
        return value

    materialized = replace(json.loads(definition))
    assert isinstance(materialized, dict)
    return materialized


def _normalize_builder_definition(
    definition: dict[str, object],
) -> dict[str, object]:
    replacements = {
        value: _production_versions()[key]
        for key, value in _published_versions().items()
    }

    def replace(value: object) -> object:
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, str):
            return replacements.get(value, value)
        return value

    normalized = replace(definition)
    assert isinstance(normalized, dict)
    envelope = {
        "activation_id": _inputs().activation_id,
        "activation_ordinal.$": (
            "States.StringToJson('${Task12ActivationOrdinal}')"
        ),
        "generation.$": "States.StringToJson('${Task12Generation}')",
        "generation_text": "${Task12GenerationText}",
        "dispatch_identity_sha256": "${Task12DispatchIdentitySha256}",
        "caller_state_machine_arn.$": "$$.StateMachine.Id",
        "state_machine_execution_arn.$": "$$.Execution.Id",
    }
    for state_name, state in normalized["States"].items():
        if (
            state.get("Type") == "Task"
            and isinstance(state.get("Resource"), str)
            and state["Resource"].startswith("arn:aws:lambda:")
        ):
            state["Parameters"] = {
                **envelope,
                "operation_kind": state_name,
                "operation_input.$": "$",
            }
            state.setdefault("ResultPath", "$.task12_last_result")
    return normalized


def test_deployment_inputs_defer_worker_and_snapshot_identity_to_typed_runtime_authority() -> None:
    from glm52_enforcement.task12_support_plane import (
        Task12SupportPlaneInputs,
        render_task12_support_plane_fragment,
    )

    input_fields = {field.name for field in fields(Task12SupportPlaneInputs)}
    assert "worker_instance_arn" not in input_fields
    assert "snapshot_arn" not in input_fields
    assert "support_deletion_workflow_version_arn" not in input_fields

    fragment = render_task12_support_plane_fragment(inputs=_inputs())
    encoded = json.dumps(fragment, sort_keys=True)
    assert "i-0123456789abcdef0" not in encoded
    assert "snap-0123456789abcdef0" not in encoded

    resources = fragment["Resources"]
    worker_statements = resources["WorkerDrainRole"]["Properties"]["Policies"][0][
        "PolicyDocument"
    ]["Statement"]
    worker_authority = next(
        statement
        for statement in worker_statements
        if statement.get("Sid") == "DrainAuthenticatedWorker"
    )
    assert worker_authority["Resource"] == (
        "arn:aws:ec2:us-west-2:246813579024:instance/*"
    )
    assert worker_authority["Condition"]["StringEquals"][
        "ssm:resourceTag/ActivationId"
    ] == _inputs().activation_id

    snapshot_statements = resources["SnapshotCleanupRole"]["Properties"][
        "Policies"
    ][0]["PolicyDocument"]["Statement"]
    snapshot_authority = next(
        statement
        for statement in snapshot_statements
        if statement.get("Sid") == "DeleteAuthenticatedSnapshot"
    )
    assert snapshot_authority["Resource"] == (
        "arn:aws:ec2:us-west-2:246813579024:snapshot/*"
    )
    assert snapshot_authority["Condition"]["StringEquals"][
        "ec2:ResourceTag/ActivationId"
    ] == _inputs().activation_id


def test_renderer_is_deterministic_and_has_exact_retained_runtime_inventory() -> None:
    from glm52_enforcement.task12_support_plane import (
        render_task12_support_plane_fragment,
    )

    first = render_task12_support_plane_fragment(inputs=_inputs())
    assert first == render_task12_support_plane_fragment(inputs=_inputs())
    resources = first["Resources"]
    assert len(resources) == 60
    assert {
        resource_type: sum(
            resource["Type"] == resource_type
            for resource in resources.values()
        )
        for resource_type in {
            "AWS::CloudWatch::Alarm",
            "AWS::Events::Rule",
            "AWS::IAM::Role",
            "AWS::Lambda::Function",
            "AWS::Lambda::Permission",
            "AWS::Lambda::Version",
            "AWS::Logs::LogGroup",
            "AWS::StepFunctions::StateMachine",
            "AWS::StepFunctions::StateMachineVersion",
        }
    } == {
        "AWS::CloudWatch::Alarm": 10,
        "AWS::Events::Rule": 1,
        "AWS::IAM::Role": 12,
        "AWS::Lambda::Function": 8,
        "AWS::Lambda::Permission": 9,
        "AWS::Lambda::Version": 8,
        "AWS::Logs::LogGroup": 8,
        "AWS::StepFunctions::StateMachine": 2,
        "AWS::StepFunctions::StateMachineVersion": 2,
    }
    roles = {
        name
        for name, resource in resources.items()
        if resource["Type"] == "AWS::IAM::Role" and name.endswith("Role")
    }
    assert {
        "ExecutionObserverRole",
        "TerminalV2Role",
        "FinalizerRole",
        "H1GDrainedRole",
        "WorkerDrainRole",
        "OperatorDispositionRole",
        "OrphanAuditRole",
        "SnapshotCleanupRole",
    }.issubset(roles)
    for name in (
        "ExecutionObserver",
        "TerminalV2",
        "Finalizer",
        "H1GDrained",
        "WorkerDrain",
        "OperatorDisposition",
        "OrphanAudit",
        "SnapshotCleanup",
    ):
        assert resources[f"{name}Function"]["Properties"]["ReservedConcurrentExecutions"] == 1
        assert resources[f"{name}Version"]["Type"] == "AWS::Lambda::Version"
        assert resources[f"{name}LogGroup"]["Type"] == "AWS::Logs::LogGroup"
        assert resources[f"{name}ErrorAlarm"]["Type"] == "AWS::CloudWatch::Alarm"
        matching_permissions = [
            resource
            for logical_id, resource in resources.items()
            if logical_id.startswith(name)
            and resource["Type"] == "AWS::Lambda::Permission"
        ]
        assert matching_permissions
        assert all(
            permission["Properties"]["FunctionName"]
            == {"Ref": f"{name}Version"}
            for permission in matching_permissions
        )
    assert first["Metadata"]["task12_retained_ownership"]["support_owned_resources"] == []


def test_rendered_workflows_are_mechanically_identical_to_authoritative_builders() -> None:
    from glm52_enforcement.task12_support_plane import (
        render_task12_support_plane_fragment,
    )
    from glm52_enforcement.task12_workflows import (
        build_retained_lifecycle_workflow,
        build_snapshot_cleanup_workflow,
    )

    resources = render_task12_support_plane_fragment(inputs=_inputs())["Resources"]
    for name in ("RetainedLifecycle", "SnapshotCleanup"):
        workflow = resources[f"{name}StateMachine"]
        assert workflow["Properties"]["StateMachineType"] == "STANDARD"
        assert resources[f"{name}StateMachineVersion"]["Type"] == (
            "AWS::StepFunctions::StateMachineVersion"
        )
        definition = workflow["Properties"]["DefinitionString"]
        assert "Retry" not in definition
        assert ":live" not in definition
        assert all(
            substitution.startswith("${") is False
            for substitution in workflow["Properties"]["DefinitionSubstitutions"]
        )
    deletion_arn = (
        "arn:aws:states:us-west-2:246813579024:stateMachine:"
        "keep-glm52-h1g-retained-support-delete:7"
    )
    assert _materialize_definition(
        resources["RetainedLifecycleStateMachine"]["Properties"]["DefinitionString"]
    ) == _normalize_builder_definition(
        build_retained_lifecycle_workflow(
            versions=_published_versions(),
            support_deletion_workflow_version_arn=deletion_arn,
        )
    )
    assert _materialize_definition(
        resources["SnapshotCleanupStateMachine"]["Properties"]["DefinitionString"]
    ) == _normalize_builder_definition(
        build_snapshot_cleanup_workflow(versions=_published_versions())
    )
    assert resources["RetainedLifecycleStateMachine"]["Properties"][
        "DefinitionSubstitutions"
    ]["SupportDeletionWorkflowVersionArn"] == {
        "Ref": "H1gRetainedLifecycleStateMachineVersion"
    }
    retained_role_statements = resources["RetainedLifecycleWorkflowRole"]["Properties"][
        "Policies"
    ][0]["PolicyDocument"]["Statement"]
    assert next(
        statement
        for statement in retained_role_statements
        if statement["Action"] == "states:StartExecution"
    ) == {
        "Sid": "StartExactSupportDelete",
        "Effect": "Allow",
        "Action": "states:StartExecution",
        "Resource": {"Ref": "H1gRetainedLifecycleStateMachine"},
        "Condition": {
            "ForAnyValue:StringEquals": {
                "states:StateMachineQualifier": [
                    {
                        "Fn::Select": [
                            7,
                            {
                                "Fn::Split": [
                                    ":",
                                    {
                                        "Ref": (
                                            "H1gRetainedLifecycleStateMachineVersion"
                                        )
                                    },
                                ]
                            },
                        ]
                    }
                ]
            }
        },
    }
    assert resources["RetainedLifecycleEventRule"]["Properties"]["Targets"][0]["Arn"] == {
        "Ref": "RetainedLifecycleStateMachineVersion"
    }


def test_snapshot_schedule_is_one_time_and_authenticated_by_cleanup_control() -> None:
    from glm52_enforcement.task12_support_plane import (
        render_task12_support_plane_fragment,
    )

    resources = render_task12_support_plane_fragment(inputs=_inputs())["Resources"]
    assert not any(
        resource["Type"] == "AWS::Scheduler::Schedule"
        for resource in resources.values()
    )
    assert "rate(1 hour)" not in json.dumps(resources, sort_keys=True)
    environment = resources["SnapshotCleanupFunction"]["Properties"]["Environment"][
        "Variables"
    ]
    assert environment == {
        "GLM52_ACTIVATION_ID": _inputs().activation_id,
        "GLM52_TASK12_RETAINED_OWNER": "1",
        "GLM52_TASK12_DEPLOYMENT_TABLE_NAME": "keep-glm52-h1g-ledger-v1",
        "GLM52_TASK12_DEPLOYMENT_PARTITION_KEY": (
            "RUN#glm52-sky-20260724"
        ),
        "GLM52_TASK12_DEPLOYMENT_SORT_KEY_PREFIX": (
            "TASK12_LAMBDA_DEPLOYMENT#"
        ),
    }
    authority = (
        render_task12_support_plane_fragment(inputs=_inputs())["Metadata"][
            "task12_deployment_authority_materialization"
        ]
    )
    snapshot_edges = [
        record
        for record in authority["records"]
        if record["handler_kind"] == "RETAINED_SNAPSHOT_CLEANUP"
    ]
    assert [record["operation_family"] for record in snapshot_edges] == [
        "ARM_SCHEDULE",
        "DELETE_OR_RECONCILE",
    ]
    assert all(
        record["role_coordinate_fields"]
        == [
            "authority",
            "kms_key_id",
            "ledger_table_name",
            "snapshot_cleanup_state_machine_arn",
            "snapshot_cleanup_state_machine_version",
            "snapshot_cleanup_schedule_invoke_role_arn",
            "snapshot_cleanup_schedule_group_name",
            "snapshot_cleanup_schedule_name",
        ]
        for record in snapshot_edges
    )
    assert all(
        record["resolved_role_coordinates"][
            "snapshot_cleanup_state_machine_arn"
        ]
        == {"Ref": "SnapshotCleanupStateMachine"}
        for record in snapshot_edges
    )


def test_runtime_authority_reads_are_prefix_closed_and_orphan_scoped() -> None:
    """Break caught: retained handlers regain bucket-wide GetObject authority."""

    from glm52_enforcement.task12_support_plane import (
        render_task12_support_plane_fragment,
    )

    resources = render_task12_support_plane_fragment(inputs=_inputs())[
        "Resources"
    ]
    deployment_prefix = (
        _inputs().model_bucket_arn
        + "/campaigns/glm52-sky-20260724/task12/deployment/"
        + _inputs().activation_id
        + "/*"
    )
    orphan_prefix = (
        _inputs().model_bucket_arn
        + "/campaigns/glm52-sky-20260724/task12/orphans/"
        + _inputs().activation_id
        + "/*"
    )
    for logical in (
        "ExecutionObserver",
        "TerminalV2",
        "Finalizer",
        "H1GDrained",
        "WorkerDrain",
        "OperatorDisposition",
        "SnapshotCleanup",
    ):
        statement = next(
            item
            for item in resources[logical + "Role"]["Properties"][
                "Policies"
            ][0]["PolicyDocument"]["Statement"]
            if item["Sid"] == "ReadExactVersionedCoordinatorAuthority"
        )
        assert statement["Resource"] == deployment_prefix
        assert statement["Resource"] != _inputs().model_bucket_arn + "/*"
    orphan = next(
        item
        for item in resources["OrphanAuditRole"]["Properties"]["Policies"][
            0
        ]["PolicyDocument"]["Statement"]
        if item["Sid"] == "ReadExactVersionedCoordinatorAuthority"
    )
    assert orphan["Resource"] == [deployment_prefix, orphan_prefix]


def test_validator_denies_bucket_wide_runtime_authority_read() -> None:
    """Break caught: exact activation prefixes silently widen to the bucket."""

    from glm52_enforcement.task12_support_plane import (
        Task12SupportPlaneError,
        render_task12_support_plane_fragment,
        validate_task12_support_plane_fragment,
    )

    fragment = copy.deepcopy(
        render_task12_support_plane_fragment(inputs=_inputs())
    )
    statements = fragment["Resources"]["OrphanAuditRole"]["Properties"][
        "Policies"
    ][0]["PolicyDocument"]["Statement"]
    next(
        item
        for item in statements
        if item["Sid"] == "ReadExactVersionedCoordinatorAuthority"
    )["Resource"] = _inputs().model_bucket_arn + "/*"
    with pytest.raises(Task12SupportPlaneError):
        validate_task12_support_plane_fragment(
            fragment,
            inputs=_inputs(),
        )


def test_sync_delete_and_runtime_snapshot_policies_are_executable_and_scoped() -> None:
    from glm52_enforcement.task12_support_plane import render_task12_support_plane_fragment

    resources = render_task12_support_plane_fragment(inputs=_inputs())["Resources"]
    statements = resources["RetainedLifecycleWorkflowRole"]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
    assert next(item for item in statements if item.get("Sid") == "StartExactSupportDelete") == {
        "Sid": "StartExactSupportDelete",
        "Effect": "Allow",
        "Action": "states:StartExecution",
        "Resource": {"Ref": "H1gRetainedLifecycleStateMachine"},
        "Condition": {
            "ForAnyValue:StringEquals": {
                "states:StateMachineQualifier": [
                    {
                        "Fn::Select": [
                            7,
                            {
                                "Fn::Split": [
                                    ":",
                                    {
                                        "Ref": (
                                            "H1gRetainedLifecycleStateMachineVersion"
                                        )
                                    },
                                ]
                            },
                        ]
                    }
                ]
            }
        },
    }
    assert next(item for item in statements if item.get("Sid") == "AwaitExactSupportDelete")["Action"] == ["states:DescribeExecution", "states:StopExecution"]
    assert next(item for item in statements if item.get("Sid") == "ManageExactSyncRule")["Action"] == ["events:PutTargets", "events:PutRule", "events:DescribeRule"]
    for role_name, state_machine_name in (
        ("RetainedLifecycleEventInvokeRole", "RetainedLifecycle"),
        ("SnapshotCleanupScheduleInvokeRole", "SnapshotCleanup"),
    ):
        start = resources[role_name]["Properties"]["Policies"][0][
            "PolicyDocument"
        ]["Statement"][0]
        assert start == {
            "Effect": "Allow",
            "Action": "states:StartExecution",
            "Resource": {"Ref": f"{state_machine_name}StateMachine"},
            "Condition": {
                "ForAnyValue:StringEquals": {
                    "states:StateMachineQualifier": [
                        {
                            "Fn::Select": [
                                7,
                                {
                                    "Fn::Split": [
                                        ":",
                                        {
                                            "Ref": (
                                                f"{state_machine_name}"
                                                "StateMachineVersion"
                                            )
                                        },
                                    ]
                                },
                            ]
                        }
                    ]
                }
            },
        }
    snapshot = resources["SnapshotCleanupRole"]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
    assert next(item for item in snapshot if item["Sid"] == "DeleteAuthenticatedSnapshot")["Action"] == "ec2:DeleteSnapshot"
    assert next(item for item in snapshot if item["Sid"] == "ReadBackAuthenticatedSnapshot")["Resource"] == "*"
    drain = resources["WorkerDrainRole"]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
    assert next(item for item in drain if item["Sid"] == "DrainAuthenticatedWorker")["Condition"]["StringEquals"]["ssm:resourceTag/ActivationId"] == _inputs().activation_id
    assert next(item for item in drain if item["Sid"] == "DecryptOwnerNonceCapsule") == {
        "Sid": "DecryptOwnerNonceCapsule",
        "Effect": "Allow",
        "Action": "kms:Decrypt",
        "Resource": _inputs().retained_kms_key_arn,
    }
    assert next(
        item for item in drain if item["Sid"] == "ReadAndCommitExactDrainLedger"
    )["Action"] == [
        "dynamodb:GetItem",
        "dynamodb:Query",
        "dynamodb:TransactWriteItems",
    ]
    assert next(
        item
        for item in drain
        if item["Sid"] == "StopAuthenticatedRuntimeInstances"
    )["Condition"] == {
        "StringEquals": {
            "ec2:ResourceTag/ActivationId": _inputs().activation_id
        }
    }
    assert next(
        item
        for item in drain
        if item["Sid"] == "ReadExactWorkerDrainVersions"
    )["Resource"] == [
        _inputs().model_bucket_arn
        + "/campaigns/glm52-sky-20260724/submissions/production/"
        "descriptor.json",
        _inputs().model_bucket_arn
        + "/campaigns/glm52-sky-20260724/authorities/"
        "GPU_SPEND_APPROVAL.json",
        _inputs().model_bucket_arn
        + "/campaigns/glm52-sky-20260724/runtime/"
        "GPU_SPEND_LEDGER_LATEST.json",
        _inputs().model_bucket_arn
        + "/campaigns/glm52-sky-20260724/runtime/"
        "spend-ledger/records/*",
        _inputs().model_bucket_arn
        + "/task13/production/task10-worker-descriptor.json",
    ]
    assert next(
        item
        for item in drain
        if item["Sid"] == "ObserveControllerSsmOffline"
    ) == {
        "Sid": "ObserveControllerSsmOffline",
        "Effect": "Allow",
        "Action": "ssm:DescribeInstanceInformation",
        "Resource": "*",
    }
    assert not any(
        item.get("Sid") == "PublishExactGenerationCampaignDrained"
        for item in drain
    )
    terminal = resources["TerminalV2Role"]["Properties"]["Policies"][0][
        "PolicyDocument"
    ]["Statement"]
    assert next(
        item
        for item in terminal
        if item["Sid"] == "QueryCompleteTerminalLedgerFamilies"
    )["Action"] == "dynamodb:Query"
    assert next(
        item
        for item in terminal
        if item["Sid"] == "InvokeExactNumericBindingVersion"
    )["Resource"] == {"Ref": "NumericBindingVersion"}
    assert next(
        item
        for item in terminal
        if item["Sid"] == "ObserveTerminalRuntimeInstances"
    )["Action"] == "ec2:DescribeInstances"
    assert next(
        item
        for item in terminal
        if item["Sid"] == "ListTerminalRuntimeVersions"
    ) == {
        "Sid": "ListTerminalRuntimeVersions",
        "Effect": "Allow",
        "Action": "s3:ListBucketVersions",
        "Resource": _inputs().model_bucket_arn,
        "Condition": {
            "ForAnyValue:StringLike": {
                "s3:prefix": [
                    "campaigns/glm52-sky-20260724/"
                    "CAMPAIGN_DRAINED.json",
                    "campaigns/glm52-sky-20260724/ledger/"
                    "campaign-ledger.jsonl",
                    "campaigns/glm52-sky-20260724/runtime/"
                    "GPU_SPEND_LEDGER.jsonl",
                    "campaigns/glm52-sky-20260724/runtime/"
                    "GPU_SPEND_LEDGER_LATEST.json",
                    "campaigns/glm52-sky-20260724/runtime/"
                    "spend-ledger/records/*",
                    "task13/production/"
                    "task10-worker-descriptor.json",
                    "campaigns/glm52-sky-20260724/submissions/production/"
                    "generations/*/terminal-evidence/*",
                    "campaigns/glm52-sky-20260724/submissions/production/"
                    "generations/*/allocations/*/"
                    "WORKER_GRACEFUL_STOP.json",
                    "campaigns/glm52-sky-20260724/submissions/production/"
                    "generations/*/terminal/CAMPAIGN_DRAINED.json",
                    "campaigns/glm52-sky-20260724/submissions/production/"
                    "generations/*/terminal/PRODUCTION_TERMINAL_V2.json",
                ]
            }
        },
    }
    terminal_versions = next(
        item
        for item in terminal
        if item["Sid"] == "ReadExactTerminalRuntimeVersions"
    )["Resource"]
    assert (
        _inputs().model_bucket_arn
        + "/campaigns/glm52-sky-20260724/authorities/task9/"
        + _inputs().activation_id
        + "/TASK9_DEPLOYED_IDENTITY.json"
    ) in terminal_versions
    assert (
        _inputs().model_bucket_arn
        + "/task13/production/task10-worker-descriptor.json"
    ) in terminal_versions
    for source_key in (
        "/campaigns/glm52-sky-20260724/CAMPAIGN_DRAINED.json",
        "/campaigns/glm52-sky-20260724/ledger/"
        "campaign-ledger.jsonl",
        "/campaigns/glm52-sky-20260724/runtime/"
        "GPU_SPEND_LEDGER.jsonl",
    ):
        assert _inputs().model_bucket_arn + source_key in terminal_versions
    terminal_key = (
        _inputs().model_bucket_arn
        + "/campaigns/glm52-sky-20260724/submissions/production/"
        "generations/*/terminal/PRODUCTION_TERMINAL_V2.json"
    )
    assert terminal_key in terminal_versions
    terminal_write = next(
        item
        for item in terminal
        if item["Sid"] == "WriteExactRetainedRecord"
    )
    assert terminal_write == {
        "Sid": "WriteExactRetainedRecord",
        "Effect": "Allow",
        "Action": ["s3:GetObjectVersion", "s3:PutObject"],
        "Resource": terminal_key,
    }
    generation_key = (
        _inputs().model_bucket_arn
        + "/campaigns/glm52-sky-20260724/submissions/production/"
        "generations/*/terminal/CAMPAIGN_DRAINED.json"
    )
    assert next(
        item
        for item in terminal
        if item["Sid"] == "PublishExactGenerationCampaignDrained"
    ) == {
        "Sid": "PublishExactGenerationCampaignDrained",
        "Effect": "Allow",
        "Action": ["s3:GetObjectVersion", "s3:PutObject"],
        "Resource": generation_key,
    }
    assert generation_key in terminal_versions
    assert not any(
        item.get("Resource") == _inputs().model_bucket_arn + "/*"
        for item in terminal
    )
    finalization_prefix = (
        "campaigns/glm52-sky-20260724/submissions/production/"
        "activations/"
        + _inputs().activation_id
        + "/finalization/"
    )
    support_finalized_key = (
        finalization_prefix + "SUPPORT_PLANE_FINALIZED.json"
    )
    h1g_drained_key = finalization_prefix + "H1G_DRAINED.json"
    terminal_v2_key = (
        "campaigns/glm52-sky-20260724/submissions/production/"
        "generations/00000001/terminal/PRODUCTION_TERMINAL_V2.json"
    )
    finalizer = resources["FinalizerRole"]["Properties"]["Policies"][0][
        "PolicyDocument"
    ]["Statement"]
    assert next(
        item
        for item in finalizer
        if item["Sid"] == "WriteExactRetainedRecord"
    ) == {
        "Sid": "WriteExactRetainedRecord",
        "Effect": "Allow",
        "Action": ["s3:GetObjectVersion", "s3:PutObject"],
        "Resource": (
            _inputs().model_bucket_arn + "/" + support_finalized_key
        ),
    }
    assert next(
        item
        for item in finalizer
        if item["Sid"] == "ListExactRetainedRecordVersion"
    ) == {
        "Sid": "ListExactRetainedRecordVersion",
        "Effect": "Allow",
        "Action": "s3:ListBucketVersions",
        "Resource": _inputs().model_bucket_arn,
        "Condition": {
            "StringEquals": {"s3:prefix": support_finalized_key}
        },
    }
    h1g = resources["H1GDrainedRole"]["Properties"]["Policies"][0][
        "PolicyDocument"
    ]["Statement"]
    assert next(
        item
        for item in h1g
        if item["Sid"] == "WriteExactRetainedRecord"
    ) == {
        "Sid": "WriteExactRetainedRecord",
        "Effect": "Allow",
        "Action": ["s3:GetObjectVersion", "s3:PutObject"],
        "Resource": _inputs().model_bucket_arn + "/" + h1g_drained_key,
    }
    assert next(
        item
        for item in h1g
        if item["Sid"] == "ListExactRetainedRecordVersion"
    ) == {
        "Sid": "ListExactRetainedRecordVersion",
        "Effect": "Allow",
        "Action": "s3:ListBucketVersions",
        "Resource": _inputs().model_bucket_arn,
        "Condition": {"StringEquals": {"s3:prefix": h1g_drained_key}},
    }
    assert next(
        item
        for item in h1g
        if item["Sid"] == "ReadExactFinalizationVersions"
    ) == {
        "Sid": "ReadExactFinalizationVersions",
        "Effect": "Allow",
        "Action": "s3:GetObjectVersion",
        "Resource": (
            _inputs().model_bucket_arn + "/" + support_finalized_key
        ),
    }
    assert next(
        item
        for item in h1g
        if item["Sid"] == "ReadExactTerminalV2Version"
    ) == {
        "Sid": "ReadExactTerminalV2Version",
        "Effect": "Allow",
        "Action": "s3:GetObjectVersion",
        "Resource": _inputs().model_bucket_arn + "/" + terminal_v2_key,
    }
    operator = resources["OperatorDispositionRole"]["Properties"][
        "Policies"
    ][0]["PolicyDocument"]["Statement"]
    assert not any(
        item.get("Sid") == "WriteExactRetainedRecord"
        for item in operator
    )
    for statements in (finalizer, h1g, operator):
        assert not any(
            item.get("Resource") == _inputs().model_bucket_arn + "/*"
            for item in statements
        )
    observer = resources["ExecutionObserverRole"]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
    observer_ledger = next(
        item
        for item in observer
        if item["Sid"] == "ReadAndConditionallyCreateExactLedger"
    )
    assert observer_ledger["Resource"] == _inputs().ledger_table_arn
    assert observer_ledger["Action"] == [
        "dynamodb:GetItem",
        "dynamodb:Query",
        "dynamodb:PutItem",
        "dynamodb:UpdateItem",
    ]
    assert next(
        item
        for item in observer
        if item["Sid"] == "StopExactAuthenticatedSupportExecution"
    ) == {
        "Sid": "StopExactAuthenticatedSupportExecution",
        "Effect": "Allow",
        "Action": "states:StopExecution",
        "Resource": (
            "arn:aws:states:us-west-2:246813579024:execution:"
            "keep-glm52-h1g-retainedlifecycle:*"
        ),
    }
    for writer in ("Finalizer", "H1GDrained"):
        writer_statements = resources[writer + "Role"]["Properties"][
            "Policies"
        ][0]["PolicyDocument"]["Statement"]
        assert "dynamodb:TransactWriteItems" in next(
            item for item in writer_statements if item["Sid"] == "WriteExactLedger"
        )["Action"]
        assert next(
            item
            for item in writer_statements
            if item["Sid"] == "DecryptOwnerNonceCapsule"
        )["Resource"] == _inputs().retained_kms_key_arn
    h1g = resources["H1GDrainedRole"]["Properties"]["Policies"][0][
        "PolicyDocument"
    ]["Statement"]
    spend_key = (
        "campaigns/glm52-sky-20260724/runtime/GPU_SPEND_LEDGER.jsonl"
    )
    assert next(
        item
        for item in h1g
        if item["Sid"] == "ListExactGpuSpendLedgerVersion"
    ) == {
        "Sid": "ListExactGpuSpendLedgerVersion",
        "Effect": "Allow",
        "Action": "s3:ListBucketVersions",
        "Resource": _inputs().model_bucket_arn,
        "Condition": {"StringEquals": {"s3:prefix": spend_key}},
    }
    assert next(
        item
        for item in h1g
        if item["Sid"] == "ReadExactGpuSpendLedgerVersion"
    ) == {
        "Sid": "ReadExactGpuSpendLedgerVersion",
        "Effect": "Allow",
        "Action": "s3:GetObjectVersion",
        "Resource": _inputs().model_bucket_arn + "/" + spend_key,
    }
    orphan = resources["OrphanAuditRole"]["Properties"]["Policies"][0][
        "PolicyDocument"
    ]["Statement"]
    assert next(
        item
        for item in orphan
        if item["Sid"] == "ReadAndPersistExactRetainedLedger"
    )["Resource"] == _inputs().ledger_table_arn
    assert next(
        item
        for item in orphan
        if item["Sid"] == "ReadExactSupportStackResources"
    )["Resource"] == (
        "arn:aws:cloudformation:us-west-2:246813579024:"
        "stack/keep-glm52-h1g-support/*"
    )
    assert next(
        item
        for item in orphan
        if item["Sid"] == "ReadAndCleanExactKmsGrants"
    )["Action"] == [
        "kms:ListGrants",
        "kms:DescribeKey",
        "kms:RevokeGrant",
        "kms:RetireGrant",
    ]
    regional = next(
        item
        for item in orphan
        if item["Sid"] == "ScanExactRegionalRuntimeFamilies"
    )
    assert regional["Resource"] == "*"
    assert regional["Condition"] == {
        "StringEquals": {"aws:RequestedRegion": "us-west-2"}
    }
    assert {
        "lambda:ListFunctions",
        "iam:ListRoles",
        "states:ListStateMachines",
        "scheduler:ListSchedules",
        "events:ListRules",
        "cloudwatch:DescribeAlarms",
        "logs:DescribeLogGroups",
        "sqs:ListQueues",
        "s3:ListAllMyBuckets",
        "secretsmanager:ListSecrets",
    }.issubset(
        {
            action
            for statement in orphan
            for action in (
                statement["Action"]
                if type(statement["Action"]) is list
                else [statement["Action"]]
            )
        }
    )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda fragment: fragment["Resources"]["ExecutionObserverRole"]["Properties"][
            "Policies"
        ][0]["PolicyDocument"]["Statement"][0].update(
            {"Action": "*", "Resource": "*"}
        ),
        lambda fragment: fragment["Resources"]["ExecutionObserverRetainedWorkflowInvokePermission"][
            "Properties"
        ].update({"FunctionName": {"Ref": "ExecutionObserverFunction"}}),
        lambda fragment: fragment["Resources"][
            "ExecutionObserverRetainedWorkflowInvokePermission"
        ]["Properties"].update(
            {"SourceArn": {"Ref": "RetainedLifecycleStateMachine"}}
        ),
        lambda fragment: fragment["Resources"]["FinalizerFunction"].update(
            {"DeletionPolicy": "Delete"}
        ),
        lambda fragment: fragment["Resources"]["SnapshotCleanupStateMachine"]["Properties"].update(
            {"DefinitionString": '{"Retry":["States.ALL"]}'}
        ),
        lambda fragment: fragment["Resources"]["RetainedLifecycleStateMachine"][
            "Properties"
        ].update({"DefinitionString": '{"StartAt":"VALIDATE_APPROVED_TASK11_HANDOFF"}'}),
        lambda fragment: fragment["Resources"]["SnapshotCleanupStateMachine"][
            "Properties"
        ].update(
            {
                "DefinitionString": (
                    '{"StartAt":"SNAPSHOT_CLEANUP_SEND_SAME_ID_OR_READ_BACK",'
                    '"States":{"SNAPSHOT_CLEANUP_SEND_SAME_ID_OR_READ_BACK":'
                    '{"Type":"Task","Retry":[{"ErrorEquals":["States.ALL"]}],'
                    '"End":true}}}'
                )
            }
        ),
        lambda fragment: fragment["Resources"]["RetainedLifecycleStateMachine"][
            "Properties"
        ]["DefinitionSubstitutions"].update(
            {"ExecutionObserverVersionArn": {"Ref": "ExecutionObserverFunction"}}
        ),
        lambda fragment: next(
            statement
            for statement in fragment["Resources"][
                "RetainedLifecycleWorkflowRole"
            ]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
            if statement["Action"] == "lambda:InvokeFunction"
        ).update({"Resource": [{"Ref": "Task12ActivationOrdinal"}]}),
        lambda fragment: next(
            statement
            for statement in fragment["Resources"][
                "SnapshotCleanupWorkflowRole"
            ]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
            if statement["Action"] == "lambda:InvokeFunction"
        ).update({"Resource": [{"Ref": "SnapshotCleanupFunction"}]}),
    ],
)
def test_validator_rejects_authority_and_graph_mutants(mutator: object) -> None:
    from glm52_enforcement.task12_support_plane import (
        Task12SupportPlaneError,
        render_task12_support_plane_fragment,
        validate_task12_support_plane_fragment,
    )

    fragment = copy.deepcopy(render_task12_support_plane_fragment(inputs=_inputs()))
    mutator(fragment)

    with pytest.raises(Task12SupportPlaneError):
        validate_task12_support_plane_fragment(fragment, inputs=_inputs())


def test_all_resource_policies_use_versioned_function_targets() -> None:
    from glm52_enforcement.task12_support_plane import render_task12_support_plane_fragment

    resources = render_task12_support_plane_fragment(inputs=_inputs())["Resources"]
    permissions = [
        resource
        for resource in resources.values()
        if resource["Type"] == "AWS::Lambda::Permission"
    ]
    assert len(permissions) == 9
    assert all(
        isinstance(resource["Properties"]["FunctionName"], dict)
        and resource["Properties"]["FunctionName"]["Ref"].endswith("Version")
        for resource in permissions
    )
    assert all(
        resource["Properties"]["SourceArn"]
        in (
            {"Ref": "RetainedLifecycleStateMachineVersion"},
            {"Ref": "SnapshotCleanupStateMachineVersion"},
        )
        for resource in permissions
    )
    assert all(
        policy["PolicyDocument"]["Statement"][0]["Resource"] != "*"
        for resource in resources.values()
        if resource["Type"] == "AWS::IAM::Role"
        for policy in resource["Properties"].get("Policies", [])
        if policy["PolicyDocument"]["Statement"]
    )


def test_production_fragment_uses_exactly_eight_named_deployable_wrappers() -> None:
    """Break caught: the retained graph points at inert generic capabilities."""

    from glm52_enforcement.task12_support_plane import (
        render_task12_support_plane_fragment,
    )

    resources = render_task12_support_plane_fragment(inputs=_inputs())["Resources"]
    expected = {
        "ExecutionObserver": "retained_execution_observer_handler.main",
        "TerminalV2": "retained_terminal_v2_handler.main",
        "Finalizer": "retained_finalizer_handler.main",
        "H1GDrained": "retained_h1g_drained_handler.main",
        "WorkerDrain": "retained_worker_drain_handler.main",
        "OperatorDisposition": "retained_operator_disposition_handler.main",
        "OrphanAudit": "retained_orphan_audit_handler.main",
        "SnapshotCleanup": "retained_snapshot_cleanup_handler.main",
    }
    functions = {
        logical_id.removesuffix("Function"): resource["Properties"]["Handler"]
        for logical_id, resource in resources.items()
        if resource["Type"] == "AWS::Lambda::Function"
    }

    assert functions == expected
    assert "task12_handlers.dispatch_handler" not in json.dumps(
        resources, sort_keys=True
    )


def test_task12_archive_contains_all_named_wrappers_deterministically(
    tmp_path: Path,
) -> None:
    """Break caught: CloudFormation names a handler absent from the Lambda zip."""

    import importlib.util
    import zipfile

    script = (
        Path(__file__).resolve().parents[1]
        / "aws/glm52-gpu/scripts/package_h1g_support_lambdas.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_task12_package_support_lambdas", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    module.build(first)
    module.build(second)

    expected = {
        "retained_execution_observer_handler.py",
        "retained_terminal_v2_handler.py",
        "retained_finalizer_handler.py",
        "retained_h1g_drained_handler.py",
        "retained_worker_drain_handler.py",
        "retained_operator_disposition_handler.py",
        "retained_orphan_audit_handler.py",
        "retained_snapshot_cleanup_handler.py",
    }
    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
        assert expected.issubset(names)
        assert len(names) == len(set(names))
        assert all(
            archive.getinfo(name).date_time == (1980, 1, 1, 0, 0, 0)
            for name in names
        )
    assert first.read_bytes() == second.read_bytes()


def test_function_publication_graph_has_no_state_machine_version_cycle() -> None:
    """Break caught: Lambda version config depends on its caller workflow version."""

    from glm52_enforcement.task12_support_plane import (
        render_task12_support_plane_fragment,
    )

    resources = render_task12_support_plane_fragment(inputs=_inputs())["Resources"]
    for logical_id, resource in resources.items():
        if resource["Type"] in {"AWS::Lambda::Function", "AWS::Lambda::Version"}:
            assert "StateMachineVersion" not in json.dumps(
                resource, sort_keys=True
            )
    pass_role = next(
        statement
        for statement in resources["SnapshotCleanupRole"]["Properties"][
            "Policies"
        ][0]["PolicyDocument"]["Statement"]
        if statement.get("Sid") == "PassOnlyCleanupScheduleRole"
    )
    assert pass_role["Resource"] == (
        "arn:aws:iam::246813579024:role/"
        "keep-glm52-h1g-snapshot-cleanup-schedule-invoke"
    )


def test_every_wrapper_role_can_authenticate_its_exact_caller_execution() -> None:
    """Break caught: only the first wrapper proves the caller's exact version."""

    from glm52_enforcement.task12_support_plane import (
        render_task12_support_plane_fragment,
    )

    resources = render_task12_support_plane_fragment(inputs=_inputs())["Resources"]
    for logical in (
        "ExecutionObserver",
        "TerminalV2",
        "Finalizer",
        "H1GDrained",
        "WorkerDrain",
        "OperatorDisposition",
        "OrphanAudit",
        "SnapshotCleanup",
    ):
        statements = resources[f"{logical}Role"]["Properties"]["Policies"][0][
            "PolicyDocument"
        ]["Statement"]
        caller = next(
            statement
            for statement in statements
            if statement.get("Sid") == "AuthenticateExactCallerVersion"
        )
        assert caller["Action"] == "states:DescribeExecution"
        encoded = json.dumps(caller["Resource"], sort_keys=True)
        assert ":execution:keep-glm52-h1g-retainedlifecycle:*" in encoded
        if logical == "SnapshotCleanup":
            assert ":execution:keep-glm52-h1g-snapshotcleanup:*" in encoded


def test_workflow_invoke_iam_contains_only_exact_lambda_version_refs() -> None:
    """Break caught: ASL parameter refs accidentally become Lambda IAM resources."""

    from glm52_enforcement.task12_support_plane import (
        render_task12_support_plane_fragment,
    )

    resources = render_task12_support_plane_fragment(inputs=_inputs())["Resources"]
    expected = {
        "RetainedLifecycleWorkflowRole": {
            "ExecutionObserverVersion",
            "TerminalV2Version",
            "FinalizerVersion",
            "H1GDrainedVersion",
            "WorkerDrainVersion",
            "OperatorDispositionVersion",
            "OrphanAuditVersion",
            "SnapshotCleanupVersion",
        },
        "SnapshotCleanupWorkflowRole": {"SnapshotCleanupVersion"},
    }
    for role_name, exact_versions in expected.items():
        statement = next(
            item
            for item in resources[role_name]["Properties"]["Policies"][0][
                "PolicyDocument"
            ]["Statement"]
            if item["Action"] == "lambda:InvokeFunction"
        )
        assert statement["Resource"] == [
            {"Ref": version} for version in sorted(exact_versions)
        ]
        assert all(
            resource["Ref"].endswith("Version")
            and not resource["Ref"].startswith("Task12")
            for resource in statement["Resource"]
        )


def test_every_lambda_state_dispatches_its_exact_operation_and_carries_prior_output() -> None:
    """Break caught: conceptual states all invoke one fixture-preseeded operation."""

    from glm52_enforcement.task12_support_plane import (
        render_task12_support_plane_fragment,
    )

    resources = render_task12_support_plane_fragment(inputs=_inputs())["Resources"]
    counts = {"RetainedLifecycle": 22, "SnapshotCleanup": 10}
    for workflow_name, expected_count in counts.items():
        definition = json.loads(
            resources[f"{workflow_name}StateMachine"]["Properties"][
                "DefinitionString"
            ]
        )
        lambda_states = {
            state_name: state
            for state_name, state in definition["States"].items()
            if state.get("Type") == "Task"
            and isinstance(state.get("Resource"), str)
            and state["Resource"].startswith("${")
        }
        assert len(lambda_states) == expected_count
        for state_name, state in lambda_states.items():
            assert state["Parameters"]["operation_kind"] == state_name
            assert state["Parameters"]["operation_input.$"] == "$"
            assert state["ResultPath"] == "$.task12_last_result"
    retained = json.loads(
        resources["RetainedLifecycleStateMachine"]["Properties"][
            "DefinitionString"
        ]
    )
    assert retained["States"]["RETAINED_SUPPORT_DELETE_REQUESTED"][
        "ResultPath"
    ] == "$.support_delete_result"
