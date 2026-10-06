"""Closed retained-runtime CloudFormation fragment for Task 12.

This module owns only the retained recovery plane.  The support continuation
is deliberately rendered elsewhere: it is a distinct mode with a different
authority boundary.  Both retained ASL documents are produced from the
authoritative workflow builders below, rather than maintained as a second
hand-written graph.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .task12_workflows import (
    build_retained_lifecycle_workflow,
    build_snapshot_cleanup_workflow,
)


class Task12SupportPlaneError(ValueError):
    """Raised when a Task 12 retained fragment drifts from its closed contract."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RETAIN = {"DeletionPolicy": "Retain", "UpdateReplacePolicy": "Retain"}
_ACCOUNT = "246813579024"
_REGION = "us-west-2"
_RUN_ID = "glm52-sky-20260724"

# These are the eight deployable Task 12 adapters.  The workflow builders use
# finer-grained capability names, but multiple states may intentionally invoke
# one of these named adapters with a different authenticated retained input.
_RUNTIMES = (
    (
        "ExecutionObserver",
        "execution-observer",
        "retained_execution_observer_handler.main",
        ("retained",),
    ),
    (
        "TerminalV2",
        "terminal-v2-writer",
        "retained_terminal_v2_handler.main",
        ("retained",),
    ),
    (
        "Finalizer",
        "finalizer",
        "retained_finalizer_handler.main",
        ("retained",),
    ),
    (
        "H1GDrained",
        "h1g-drained-writer",
        "retained_h1g_drained_handler.main",
        ("retained",),
    ),
    (
        "WorkerDrain",
        "worker-drain-signal",
        "retained_worker_drain_handler.main",
        ("retained",),
    ),
    (
        "OperatorDisposition",
        "operator-disposition-writer",
        "retained_operator_disposition_handler.main",
        ("retained",),
    ),
    (
        "OrphanAudit",
        "orphan-audit",
        "retained_orphan_audit_handler.main",
        ("retained",),
    ),
    (
        "SnapshotCleanup",
        "snapshot-cleanup",
        "retained_snapshot_cleanup_handler.main",
        ("retained", "snapshot"),
    ),
)
_RETAINED_BINDINGS = {
    "recovery_coordinator": "ExecutionObserverVersionArn",
    "recovery_cancellation": "OperatorDispositionVersionArn",
    "recovery_drain": "WorkerDrainVersionArn",
    "recovery_terminal_writer": "TerminalV2VersionArn",
    "teardown_coordinator": "ExecutionObserverVersionArn",
    "finalizer": "FinalizerVersionArn",
    "orphan_auditor": "OrphanAuditVersionArn",
    "snapshot_scheduler": "SnapshotCleanupVersionArn",
    "h1g_drained_writer": "H1GDrainedVersionArn",
}
_SNAPSHOT_BINDINGS = {
    "snapshot_cleanup_coordinator": "SnapshotCleanupVersionArn",
    "snapshot_cleanup_writer": "SnapshotCleanupVersionArn",
}
_DEPLOYMENT_SPECS = {
    "ExecutionObserver": (
        "RETAINED_EXECUTION_OBSERVER",
        "RETAINED",
        [
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
        ],
    ),
    "TerminalV2": (
        "RETAINED_TERMINAL_V2",
        "RETAINED",
        [
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
        ],
    ),
    "Finalizer": (
        "RETAINED_FINALIZER",
        "RETAINED",
        ["authority", "campaign_bucket", "kms_key_id", "ledger_table_name"],
    ),
    "H1GDrained": (
        "RETAINED_H1G_DRAINED",
        "RETAINED",
        [
            "authority",
            "campaign_bucket",
            "kms_key_id",
            "ledger_table_name",
            "support_stack_id",
        ],
    ),
    "WorkerDrain": (
        "RETAINED_WORKER_DRAIN",
        "RETAINED",
        [
            "authority",
            "campaign_bucket",
            "kms_key_id",
            "ledger_table_name",
            "numeric_binding_version_arn",
            "task9_deployed_identity_coordinate",
            "task9_deployed_identity_sha256",
            "spend_runtime_prefix",
            "campaign_descriptor_key",
            "campaign_descriptor_version_id",
            "campaign_descriptor_file_sha256",
            "gpu_spend_approval_key",
            "gpu_spend_approval_version_id",
            "gpu_spend_approval_file_sha256",
            "task10_worker_descriptor_coordinate",
            "worker_script_hashes",
            "worker_drain_document_name",
            "worker_drain_document_version",
        ],
    ),
    "OperatorDisposition": (
        "RETAINED_OPERATOR_DISPOSITION",
        "RETAINED",
        [
            "authority",
            "campaign_bucket",
            "ledger_table_name",
            "numeric_binding_version_arn",
            "retained_cancellation_version_arn",
            "task9_deployed_identity_coordinate",
            "task9_deployed_identity_sha256",
        ],
    ),
    "OrphanAudit": (
        "RETAINED_ORPHAN_AUDIT",
        "RETAINED",
        ["authority", "kms_key_id", "ledger_table_name", "support_stack_id"],
    ),
    "SnapshotCleanup": (
        "RETAINED_SNAPSHOT_CLEANUP",
        "SNAPSHOT_CLEANUP",
        [
            "authority",
            "kms_key_id",
            "ledger_table_name",
            "snapshot_cleanup_state_machine_arn",
            "snapshot_cleanup_state_machine_version",
            "snapshot_cleanup_schedule_invoke_role_arn",
            "snapshot_cleanup_schedule_group_name",
            "snapshot_cleanup_schedule_name",
        ],
    ),
}


@dataclass(frozen=True)
class Task12SupportPlaneInputs:
    activation_id: str
    lambda_code_sha256: str
    lambda_code_bucket: str
    lambda_code_key: str
    lambda_code_version: str
    ledger_table_arn: str
    retained_kms_key_arn: str
    model_bucket_arn: str
    worker_drain_document_arn: str


def _fail(message: str) -> None:
    raise Task12SupportPlaneError(message)


def _validate_inputs(inputs: Task12SupportPlaneInputs) -> None:
    if type(inputs) is not Task12SupportPlaneInputs:
        _fail("Task 12 support-plane inputs must be typed")
    if not _SHA256.fullmatch(inputs.lambda_code_sha256):
        _fail("lambda code SHA-256 must be exact")
    prefixes = {
        "ledger_table_arn": f"arn:aws:dynamodb:{_REGION}:{_ACCOUNT}:table/",
        "retained_kms_key_arn": f"arn:aws:kms:{_REGION}:{_ACCOUNT}:key/",
        "model_bucket_arn": "arn:aws:s3:::",
        "worker_drain_document_arn": f"arn:aws:ssm:{_REGION}:{_ACCOUNT}:document/",
    }
    for field, prefix in prefixes.items():
        value = getattr(inputs, field)
        if not isinstance(value, str) or not value.startswith(prefix):
            _fail(field + " must be a nonempty exact retained resource kind")
    for field in ("activation_id", "lambda_code_bucket", "lambda_code_key", "lambda_code_version"):
        if not isinstance(getattr(inputs, field), str) or not getattr(inputs, field):
            _fail(field + " must be a nonempty exact string")


def _role(name: str, statements: list[dict[str, Any]], *, service: str = "lambda.amazonaws.com") -> dict[str, Any]:
    return {
        **_RETAIN,
        "Type": "AWS::IAM::Role",
        "Properties": {
            "RoleName": f"keep-glm52-h1g-{name}",
            "AssumeRolePolicyDocument": {
                "Version": "2012-10-17",
                "Statement": [{"Effect": "Allow", "Principal": {"Service": service}, "Action": "sts:AssumeRole"}],
            },
            "Policies": [{"PolicyName": "Task12ExactAuthority", "PolicyDocument": {"Version": "2012-10-17", "Statement": statements}}],
        },
    }


def _log_statement(function_name: str) -> dict[str, Any]:
    return {
        "Sid": "WriteOnlyOwnLogStream",
        "Effect": "Allow",
        "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
        "Resource": f"arn:aws:logs:{_REGION}:{_ACCOUNT}:log-group:/aws/lambda/{function_name}:*",
    }


def _start_exact_state_machine_version(
    *,
    state_machine: str,
    version: str,
    sid: str | None = None,
) -> dict[str, Any]:
    statement: dict[str, Any] = {
        "Effect": "Allow",
        "Action": "states:StartExecution",
        "Resource": {"Ref": state_machine},
        "Condition": {
            "ForAnyValue:StringEquals": {
                "states:StateMachineQualifier": [
                    {
                        "Fn::Select": [
                            7,
                            {"Fn::Split": [":", {"Ref": version}]},
                        ]
                    }
                ]
            }
        },
    }
    if sid is not None:
        statement["Sid"] = sid
    return statement


def _runtime_statements(inputs: Task12SupportPlaneInputs, logical: str, function_name: str) -> list[dict[str, Any]]:
    coordinator_authority_resources = [
        (
            inputs.model_bucket_arn
            + f"/campaigns/{_RUN_ID}/task12/deployment/"
            + inputs.activation_id
            + "/*"
        )
    ]
    if logical == "OrphanAudit":
        coordinator_authority_resources.append(
            inputs.model_bucket_arn
            + f"/campaigns/{_RUN_ID}/task12/orphans/"
            + inputs.activation_id
            + "/*"
        )
    statements: list[dict[str, Any]] = [
        _log_statement(function_name),
        {
            "Sid": "ReadExactDeploymentAndInvocationAuthority",
            "Effect": "Allow",
            "Action": "dynamodb:GetItem",
            "Resource": inputs.ledger_table_arn,
            "Condition": {
                "ForAllValues:StringEquals": {
                    "dynamodb:LeadingKeys": ["RUN#glm52-sky-20260724"]
                }
            },
        },
        {
            "Sid": "ReadExactVersionedCoordinatorAuthority",
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:GetObjectVersion"],
            "Resource": (
                coordinator_authority_resources[0]
                if len(coordinator_authority_resources) == 1
                else coordinator_authority_resources
            ),
        },
    ]
    caller_execution_resources = [
        (
            f"arn:aws:states:{_REGION}:{_ACCOUNT}:execution:"
            "keep-glm52-h1g-retainedlifecycle:*"
        )
    ]
    if logical == "SnapshotCleanup":
        caller_execution_resources.append(
            f"arn:aws:states:{_REGION}:{_ACCOUNT}:execution:"
            "keep-glm52-h1g-snapshotcleanup:*"
        )
    if logical == "TerminalV2":
        caller_execution_resources.append(
            f"arn:aws:states:{_REGION}:{_ACCOUNT}:execution:"
            "keep-glm52-h1g-support:*"
        )
    statements.append(
        {
            "Sid": "AuthenticateExactCallerVersion",
            "Effect": "Allow",
            "Action": "states:DescribeExecution",
            "Resource": (
                caller_execution_resources[0]
                if len(caller_execution_resources) == 1
                else caller_execution_resources
            ),
        }
    )
    ledger_read = ["dynamodb:GetItem", "dynamodb:Query"]
    if logical == "ExecutionObserver":
        statements.extend((
            {"Sid": "ReadAndConditionallyCreateExactLedger", "Effect": "Allow", "Action": [*ledger_read, "dynamodb:PutItem", "dynamodb:UpdateItem"], "Resource": inputs.ledger_table_arn},
            {"Sid": "StopExactAuthenticatedSupportExecution", "Effect": "Allow", "Action": "states:StopExecution", "Resource": caller_execution_resources[0]},
            {"Sid": "InvokeExactNumericBindingVersion", "Effect": "Allow", "Action": "lambda:InvokeFunction", "Resource": {"Ref": "NumericBindingVersion"}},
            {"Sid": "ListOnlyGpuSpendRuntimeVersions", "Effect": "Allow", "Action": "s3:ListBucketVersions", "Resource": inputs.model_bucket_arn, "Condition": {"ForAnyValue:StringLike": {"s3:prefix": [
                "campaigns/glm52-sky-20260724/runtime/GPU_SPEND_LEDGER_LATEST.json",
                "campaigns/glm52-sky-20260724/runtime/spend-ledger/records/*",
            ]}}},
            {"Sid": "ReadExactGpuSpendAuthorityVersions", "Effect": "Allow", "Action": "s3:GetObjectVersion", "Resource": [
                inputs.model_bucket_arn + "/campaigns/glm52-sky-20260724/submissions/production/descriptor.json",
                inputs.model_bucket_arn + "/campaigns/glm52-sky-20260724/authorities/GPU_SPEND_APPROVAL.json",
                inputs.model_bucket_arn + "/campaigns/glm52-sky-20260724/runtime/GPU_SPEND_LEDGER_LATEST.json",
                inputs.model_bucket_arn + "/campaigns/glm52-sky-20260724/runtime/spend-ledger/records/*",
            ]},
            {"Sid": "ObserveAuthenticatedRuntimeInstances", "Effect": "Allow", "Action": "ec2:DescribeInstances", "Resource": "*"},
            {"Sid": "GenerateAndDecryptOwnerNonceCapsule", "Effect": "Allow", "Action": ["kms:GenerateDataKey", "kms:Decrypt"], "Resource": inputs.retained_kms_key_arn},
        ))
    elif logical == "WorkerDrain":
        statements.extend((
            {"Sid": "ReadAndCommitExactDrainLedger", "Effect": "Allow", "Action": [*ledger_read, "dynamodb:TransactWriteItems"], "Resource": inputs.ledger_table_arn},
            {"Sid": "InvokeExactNumericBindingVersion", "Effect": "Allow", "Action": "lambda:InvokeFunction", "Resource": {"Ref": "NumericBindingVersion"}},
            {"Sid": "UseExactGracefulStopDocument", "Effect": "Allow", "Action": "ssm:SendCommand", "Resource": inputs.worker_drain_document_arn},
            {"Sid": "DrainAuthenticatedWorker", "Effect": "Allow", "Action": "ssm:SendCommand", "Resource": f"arn:aws:ec2:{_REGION}:{_ACCOUNT}:instance/*", "Condition": {"StringEquals": {"ssm:resourceTag/ActivationId": inputs.activation_id}}},
            {"Sid": "ObserveControllerSsmOffline", "Effect": "Allow", "Action": "ssm:DescribeInstanceInformation", "Resource": "*"},
            {"Sid": "ObserveRuntimeInstances", "Effect": "Allow", "Action": "ec2:DescribeInstances", "Resource": "*"},
            {"Sid": "StopAuthenticatedRuntimeInstances", "Effect": "Allow", "Action": "ec2:StopInstances", "Resource": f"arn:aws:ec2:{_REGION}:{_ACCOUNT}:instance/*", "Condition": {"StringEquals": {"ec2:ResourceTag/ActivationId": inputs.activation_id}}},
            {"Sid": "ListOnlyGpuSpendRuntimeVersions", "Effect": "Allow", "Action": "s3:ListBucketVersions", "Resource": inputs.model_bucket_arn, "Condition": {"ForAnyValue:StringLike": {"s3:prefix": [
                "campaigns/glm52-sky-20260724/runtime/GPU_SPEND_LEDGER_LATEST.json",
                "campaigns/glm52-sky-20260724/runtime/spend-ledger/records/*",
            ]}}},
            {"Sid": "ReadExactWorkerDrainVersions", "Effect": "Allow", "Action": "s3:GetObjectVersion", "Resource": [
                inputs.model_bucket_arn + "/campaigns/glm52-sky-20260724/submissions/production/descriptor.json",
                inputs.model_bucket_arn + "/campaigns/glm52-sky-20260724/authorities/GPU_SPEND_APPROVAL.json",
                inputs.model_bucket_arn + "/campaigns/glm52-sky-20260724/runtime/GPU_SPEND_LEDGER_LATEST.json",
                inputs.model_bucket_arn + "/campaigns/glm52-sky-20260724/runtime/spend-ledger/records/*",
                inputs.model_bucket_arn + "/task13/production/task10-worker-descriptor.json",
            ]},
            {"Sid": "DecryptOwnerNonceCapsule", "Effect": "Allow", "Action": "kms:Decrypt", "Resource": inputs.retained_kms_key_arn},
        ))
    elif logical in {"TerminalV2", "Finalizer", "H1GDrained"}:
        statements.append(
            {"Sid": "WriteExactLedger", "Effect": "Allow", "Action": ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:TransactWriteItems"], "Resource": inputs.ledger_table_arn}
        )
        if logical == "TerminalV2":
            terminal_key = (
                inputs.model_bucket_arn
                + "/campaigns/glm52-sky-20260724/submissions/production/"
                "generations/*/terminal/PRODUCTION_TERMINAL_V2.json"
            )
            campaign_drained_key = (
                inputs.model_bucket_arn
                + "/campaigns/glm52-sky-20260724/submissions/production/"
                "generations/*/terminal/CAMPAIGN_DRAINED.json"
            )
            statements.extend((
                {"Sid": "WriteExactRetainedRecord", "Effect": "Allow", "Action": ["s3:GetObjectVersion", "s3:PutObject"], "Resource": terminal_key},
                {"Sid": "PublishExactGenerationCampaignDrained", "Effect": "Allow", "Action": ["s3:GetObjectVersion", "s3:PutObject"], "Resource": campaign_drained_key},
                {"Sid": "DecryptOwnerNonceCapsule", "Effect": "Allow", "Action": "kms:Decrypt", "Resource": inputs.retained_kms_key_arn},
                {"Sid": "QueryCompleteTerminalLedgerFamilies", "Effect": "Allow", "Action": "dynamodb:Query", "Resource": inputs.ledger_table_arn},
                {"Sid": "InvokeExactNumericBindingVersion", "Effect": "Allow", "Action": "lambda:InvokeFunction", "Resource": {"Ref": "NumericBindingVersion"}},
                {"Sid": "ObserveTerminalRuntimeInstances", "Effect": "Allow", "Action": "ec2:DescribeInstances", "Resource": "*"},
                {"Sid": "ListTerminalRuntimeVersions", "Effect": "Allow", "Action": "s3:ListBucketVersions", "Resource": inputs.model_bucket_arn, "Condition": {"ForAnyValue:StringLike": {"s3:prefix": [
                    "campaigns/glm52-sky-20260724/CAMPAIGN_DRAINED.json",
                    "campaigns/glm52-sky-20260724/ledger/campaign-ledger.jsonl",
                    "campaigns/glm52-sky-20260724/runtime/GPU_SPEND_LEDGER.jsonl",
                    "campaigns/glm52-sky-20260724/runtime/GPU_SPEND_LEDGER_LATEST.json",
                    "campaigns/glm52-sky-20260724/runtime/spend-ledger/records/*",
                    "task13/production/task10-worker-descriptor.json",
                    "campaigns/glm52-sky-20260724/submissions/production/generations/*/terminal-evidence/*",
                    "campaigns/glm52-sky-20260724/submissions/production/generations/*/allocations/*/WORKER_GRACEFUL_STOP.json",
                    "campaigns/glm52-sky-20260724/submissions/production/generations/*/terminal/CAMPAIGN_DRAINED.json",
                    "campaigns/glm52-sky-20260724/submissions/production/generations/*/terminal/PRODUCTION_TERMINAL_V2.json",
                ]}}},
                {"Sid": "ReadExactTerminalRuntimeVersions", "Effect": "Allow", "Action": "s3:GetObjectVersion", "Resource": [
                    inputs.model_bucket_arn + "/campaigns/glm52-sky-20260724/CAMPAIGN_DRAINED.json",
                    inputs.model_bucket_arn + "/campaigns/glm52-sky-20260724/ledger/campaign-ledger.jsonl",
                    inputs.model_bucket_arn + "/campaigns/glm52-sky-20260724/runtime/GPU_SPEND_LEDGER.jsonl",
                    inputs.model_bucket_arn + "/campaigns/glm52-sky-20260724/submissions/production/descriptor.json",
                    inputs.model_bucket_arn + "/campaigns/glm52-sky-20260724/authorities/GPU_SPEND_APPROVAL.json",
                    inputs.model_bucket_arn + "/campaigns/glm52-sky-20260724/authorities/task9/" + inputs.activation_id + "/TASK9_DEPLOYED_IDENTITY.json",
                    inputs.model_bucket_arn + "/campaigns/glm52-sky-20260724/runtime/GPU_SPEND_LEDGER_LATEST.json",
                    inputs.model_bucket_arn + "/campaigns/glm52-sky-20260724/runtime/spend-ledger/records/*",
                    inputs.model_bucket_arn + "/campaigns/glm52-sky-20260724/submissions/production/generations/*/terminal-evidence/*",
                    inputs.model_bucket_arn + "/campaigns/glm52-sky-20260724/submissions/production/generations/*/allocations/*/WORKER_GRACEFUL_STOP.json",
                    inputs.model_bucket_arn + "/task13/production/task10-worker-descriptor.json",
                    campaign_drained_key,
                    terminal_key,
                ]},
            ))
        elif logical in {"Finalizer", "H1GDrained"}:
            finalization_key = (
                "campaigns/glm52-sky-20260724/submissions/production/"
                "activations/"
                + inputs.activation_id
                + "/finalization/"
                + (
                    "SUPPORT_PLANE_FINALIZED.json"
                    if logical == "Finalizer"
                    else "H1G_DRAINED.json"
                )
            )
            statements.extend((
                {
                    "Sid": "WriteExactRetainedRecord",
                    "Effect": "Allow",
                    "Action": ["s3:GetObjectVersion", "s3:PutObject"],
                    "Resource": (
                        inputs.model_bucket_arn + "/" + finalization_key
                    ),
                },
                {
                    "Sid": "ListExactRetainedRecordVersion",
                    "Effect": "Allow",
                    "Action": "s3:ListBucketVersions",
                    "Resource": inputs.model_bucket_arn,
                    "Condition": {
                        "StringEquals": {"s3:prefix": finalization_key}
                    },
                },
                {"Sid": "DecryptOwnerNonceCapsule", "Effect": "Allow", "Action": "kms:Decrypt", "Resource": inputs.retained_kms_key_arn},
            ))
        if logical == "H1GDrained":
            support_finalized_key = (
                "campaigns/glm52-sky-20260724/submissions/production/"
                "activations/"
                + inputs.activation_id
                + "/finalization/SUPPORT_PLANE_FINALIZED.json"
            )
            terminal_v2_key = (
                "campaigns/glm52-sky-20260724/submissions/production/"
                "generations/00000001/terminal/"
                "PRODUCTION_TERMINAL_V2.json"
            )
            spend_ledger_key = (
                "campaigns/glm52-sky-20260724/runtime/"
                "GPU_SPEND_LEDGER.jsonl"
            )
            statements.extend((
                {"Sid": "ReadExactSupportStack", "Effect": "Allow", "Action": "cloudformation:DescribeStacks", "Resource": f"arn:aws:cloudformation:{_REGION}:{_ACCOUNT}:stack/keep-glm52-h1g-support/*"},
                {
                    "Sid": "ReadExactFinalizationVersions",
                    "Effect": "Allow",
                    "Action": "s3:GetObjectVersion",
                    "Resource": (
                        inputs.model_bucket_arn
                        + "/"
                        + support_finalized_key
                    ),
                },
                {
                    "Sid": "ReadExactTerminalV2Version",
                    "Effect": "Allow",
                    "Action": "s3:GetObjectVersion",
                    "Resource": (
                        inputs.model_bucket_arn + "/" + terminal_v2_key
                    ),
                },
                {
                    "Sid": "ListExactGpuSpendLedgerVersion",
                    "Effect": "Allow",
                    "Action": "s3:ListBucketVersions",
                    "Resource": inputs.model_bucket_arn,
                    "Condition": {
                        "StringEquals": {"s3:prefix": spend_ledger_key}
                    },
                },
                {
                    "Sid": "ReadExactGpuSpendLedgerVersion",
                    "Effect": "Allow",
                    "Action": "s3:GetObjectVersion",
                    "Resource": (
                        inputs.model_bucket_arn + "/" + spend_ledger_key
                    ),
                },
            ))
    elif logical == "OperatorDisposition":
        statements.extend((
            {"Sid": "WriteExactLedger", "Effect": "Allow", "Action": ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:Query"], "Resource": inputs.ledger_table_arn},
            {"Sid": "InvokeExactNumericBindingVersion", "Effect": "Allow", "Action": "lambda:InvokeFunction", "Resource": {"Ref": "NumericBindingVersion"}},
            {"Sid": "InvokeExactRetainedCancellationVersion", "Effect": "Allow", "Action": "lambda:InvokeFunction", "Resource": {"Ref": "RetainedCancellationVersion"}},
        ))
    elif logical == "OrphanAudit":
        statements.extend((
            {"Sid": "ReadAndPersistExactRetainedLedger", "Effect": "Allow", "Action": ["dynamodb:GetItem", "dynamodb:Query", "dynamodb:Scan", "dynamodb:PutItem"], "Resource": inputs.ledger_table_arn},
            {"Sid": "ReadAndCleanExactKmsGrants", "Effect": "Allow", "Action": ["kms:ListGrants", "kms:DescribeKey", "kms:RevokeGrant", "kms:RetireGrant"], "Resource": inputs.retained_kms_key_arn},
            {"Sid": "ReadExactSupportStackResources", "Effect": "Allow", "Action": "cloudformation:ListStackResources", "Resource": f"arn:aws:cloudformation:{_REGION}:{_ACCOUNT}:stack/keep-glm52-h1g-support/*"},
            {
                "Sid": "ScanExactRegionalEc2Families",
                "Effect": "Allow",
                "Action": [
                    "ec2:DescribeAddresses",
                    "ec2:DescribeInstances",
                    "ec2:DescribeNatGateways",
                    "ec2:DescribeNetworkInterfaces",
                    "ec2:DescribeRouteTables",
                    "ec2:DescribeSecurityGroups",
                    "ec2:DescribeSnapshots",
                    "ec2:DescribeSubnets",
                    "ec2:DescribeVolumes",
                    "ec2:DescribeVpcEndpoints",
                ],
                "Resource": "*",
                "Condition": {
                    "StringEquals": {"aws:RequestedRegion": _REGION}
                },
            },
            {
                "Sid": "ScanExactRegionalRuntimeFamilies",
                "Effect": "Allow",
                "Action": [
                    "cloudwatch:DescribeAlarms",
                    "events:ListRules",
                    "lambda:GetPolicy",
                    "lambda:ListFunctions",
                    "lambda:ListVersionsByFunction",
                    "logs:DescribeLogGroups",
                    "scheduler:GetSchedule",
                    "scheduler:ListSchedules",
                    "secretsmanager:ListSecrets",
                    "sqs:ListQueues",
                    "states:DescribeStateMachine",
                    "states:ListExecutions",
                    "states:ListStateMachines",
                    "states:ListStateMachineVersions",
                ],
                "Resource": "*",
                "Condition": {
                    "StringEquals": {"aws:RequestedRegion": _REGION}
                },
            },
            {
                "Sid": "ScanExactGlobalIdentityFamilies",
                "Effect": "Allow",
                "Action": [
                    "iam:GetRole",
                    "iam:ListInstanceProfiles",
                    "iam:ListPolicies",
                    "iam:ListRoles",
                    "s3:ListAllMyBuckets",
                ],
                "Resource": "*",
            },
            {
                "Sid": "ReadExactRetainedEvidenceBucketState",
                "Effect": "Allow",
                "Action": "s3:GetBucketVersioning",
                "Resource": inputs.model_bucket_arn,
            },
            {
                "Sid": "ReadExactRetainedLedgerState",
                "Effect": "Allow",
                "Action": "dynamodb:DescribeTable",
                "Resource": inputs.ledger_table_arn,
            },
            {
                "Sid": "ReadExactRetainedFenceStack",
                "Effect": "Allow",
                "Action": "cloudformation:DescribeStacks",
                "Resource": (
                    f"arn:aws:cloudformation:{_REGION}:{_ACCOUNT}:"
                    "stack/keep-glm52-production-fence/*"
                ),
            },
        ))
    elif logical == "SnapshotCleanup":
        statements.extend((
            {"Sid": "ReadAuthenticatedCleanupControl", "Effect": "Allow", "Action": "dynamodb:GetItem", "Resource": inputs.ledger_table_arn},
            {"Sid": "WriteExactCleanupControl", "Effect": "Allow", "Action": "dynamodb:TransactWriteItems", "Resource": inputs.ledger_table_arn},
            {"Sid": "CreateOneTimeCleanupSchedule", "Effect": "Allow", "Action": ["scheduler:CreateSchedule", "scheduler:DeleteSchedule", "scheduler:GetSchedule"], "Resource": f"arn:aws:scheduler:{_REGION}:{_ACCOUNT}:schedule/default/keep-glm52-h1g-*"},
            {"Sid": "PassOnlyCleanupScheduleRole", "Effect": "Allow", "Action": "iam:PassRole", "Resource": f"arn:aws:iam::{_ACCOUNT}:role/keep-glm52-h1g-snapshot-cleanup-schedule-invoke"},
            {"Sid": "DeleteAuthenticatedSnapshot", "Effect": "Allow", "Action": "ec2:DeleteSnapshot", "Resource": f"arn:aws:ec2:{_REGION}:{_ACCOUNT}:snapshot/*", "Condition": {"StringEquals": {"ec2:ResourceTag/ActivationId": inputs.activation_id}}},
            {"Sid": "ReadBackAuthenticatedSnapshot", "Effect": "Allow", "Action": "ec2:DescribeSnapshots", "Resource": "*"},
            {"Sid": "GenerateAndDecryptOwnerNonceCapsule", "Effect": "Allow", "Action": ["kms:GenerateDataKey", "kms:Decrypt"], "Resource": inputs.retained_kms_key_arn},
        ))
    else:
        _fail("unknown Task 12 runtime role")
    return statements


def _runtime_environment(inputs: Task12SupportPlaneInputs, logical: str) -> dict[str, Any]:
    environment = {
        "GLM52_ACTIVATION_ID": inputs.activation_id,
        "GLM52_TASK12_RETAINED_OWNER": "1",
        "GLM52_TASK12_DEPLOYMENT_TABLE_NAME": inputs.ledger_table_arn.rsplit("/", 1)[-1],
        "GLM52_TASK12_DEPLOYMENT_PARTITION_KEY": "RUN#glm52-sky-20260724",
        "GLM52_TASK12_DEPLOYMENT_SORT_KEY_PREFIX": "TASK12_LAMBDA_DEPLOYMENT#",
    }
    if logical == "TerminalV2":
        environment.update(
            {
                "GLM52_ACCOUNT_ID": _ACCOUNT,
                "GLM52_RUN_ID": _RUN_ID,
                "GLM52_CAMPAIGN_BUCKET": inputs.model_bucket_arn.rsplit(
                    ":::", 1
                )[-1],
            }
        )
    return environment


def _runtime_resources(inputs: Task12SupportPlaneInputs) -> dict[str, Any]:
    resources: dict[str, Any] = {}
    for logical, suffix, handler, workflows in _RUNTIMES:
        function_name = f"keep-glm52-h1g-{suffix}"
        resources[f"{logical}Role"] = _role(suffix, _runtime_statements(inputs, logical, function_name))
        resources[f"{logical}Function"] = {
            **_RETAIN, "Type": "AWS::Lambda::Function",
            "Properties": {"FunctionName": function_name, "Code": {"S3Bucket": inputs.lambda_code_bucket, "S3Key": inputs.lambda_code_key, "S3ObjectVersion": inputs.lambda_code_version}, "Handler": handler, "Runtime": "python3.12", "MemorySize": 256, "Timeout": 540, "ReservedConcurrentExecutions": 1, "Role": {"Fn::GetAtt": [f"{logical}Role", "Arn"]}, "Environment": {"Variables": _runtime_environment(inputs, logical)}},
        }
        resources[f"{logical}Version"] = {**_RETAIN, "Type": "AWS::Lambda::Version", "Properties": {"FunctionName": {"Ref": f"{logical}Function"}, "Description": "code=" + inputs.lambda_code_sha256}}
        resources[f"{logical}LogGroup"] = {**_RETAIN, "Type": "AWS::Logs::LogGroup", "Properties": {"LogGroupName": "/aws/lambda/" + function_name, "RetentionInDays": 14}}
        resources[f"{logical}ErrorAlarm"] = {**_RETAIN, "Type": "AWS::CloudWatch::Alarm", "Properties": {"AlarmName": function_name + "-errors", "ComparisonOperator": "GreaterThanThreshold", "Dimensions": [{"Name": "FunctionName", "Value": {"Ref": f"{logical}Function"}}], "EvaluationPeriods": 1, "MetricName": "Errors", "Namespace": "AWS/Lambda", "Period": 60, "Statistic": "Sum", "Threshold": 0, "TreatMissingData": "breaching"}}
        for workflow in workflows:
            source_version = "RetainedLifecycleStateMachineVersion" if workflow == "retained" else "SnapshotCleanupStateMachineVersion"
            suffix_name = "Retained" if workflow == "retained" else "Snapshot"
            resources[f"{logical}{suffix_name}WorkflowInvokePermission"] = {**_RETAIN, "Type": "AWS::Lambda::Permission", "Properties": {"Action": "lambda:InvokeFunction", "FunctionName": {"Ref": f"{logical}Version"}, "Principal": "states.amazonaws.com", "SourceAccount": _ACCOUNT, "SourceArn": {"Ref": source_version}}}
    return resources


def _definition_and_substitutions(inputs: Task12SupportPlaneInputs, name: str) -> tuple[str, dict[str, Any], tuple[str, ...]]:
    def as_substitutions(
        definition: Mapping[str, Any], replacements: Mapping[str, str]
    ) -> dict[str, Any]:
        def replace(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: replace(item) for key, item in value.items()}
            if isinstance(value, list):
                return [replace(item) for item in value]
            if isinstance(value, str):
                return replacements.get(value, value)
            return value

        rendered = replace(definition)
        assert isinstance(rendered, dict)
        return rendered

    if name == "RetainedLifecycle":
        bindings = _RETAINED_BINDINGS
        substitutions = {
            "ExecutionObserverVersionArn": {"Ref": "ExecutionObserverVersion"},
            "OperatorDispositionVersionArn": {"Ref": "OperatorDispositionVersion"},
            "WorkerDrainVersionArn": {"Ref": "WorkerDrainVersion"},
            "TerminalV2VersionArn": {"Ref": "TerminalV2Version"},
            "FinalizerVersionArn": {"Ref": "FinalizerVersion"},
            "OrphanAuditVersionArn": {"Ref": "OrphanAuditVersion"},
            "SnapshotCleanupVersionArn": {"Ref": "SnapshotCleanupVersion"},
            "H1GDrainedVersionArn": {"Ref": "H1GDrainedVersion"},
            "SupportDeletionWorkflowVersionArn": {"Ref": "H1gRetainedLifecycleStateMachineVersion"},
        }
        concrete = {
            key: f"arn:aws:lambda:{_REGION}:{_ACCOUNT}:function:keep-glm52-render-{key}:{index + 1}"
            for index, key in enumerate(bindings)
        }
        deletion_target = (
            f"arn:aws:states:{_REGION}:{_ACCOUNT}:stateMachine:"
            "keep-glm52-h1g-retained-support-delete:1"
        )
        definition = as_substitutions(
            build_retained_lifecycle_workflow(
                versions=concrete, support_deletion_workflow_version_arn=deletion_target
            ),
            {
                **{
                    concrete[key]: "${" + bindings[key] + "}"
                    for key in bindings
                },
                deletion_target: "${SupportDeletionWorkflowVersionArn}",
            },
        )
    elif name == "SnapshotCleanup":
        bindings = _SNAPSHOT_BINDINGS
        substitutions = {
            "SnapshotCleanupVersionArn": {"Ref": "SnapshotCleanupVersion"}
        }
        concrete = {
            key: f"arn:aws:lambda:{_REGION}:{_ACCOUNT}:function:keep-glm52-render-{key}:{index + 1}"
            for index, key in enumerate(bindings)
        }
        definition = as_substitutions(
            build_snapshot_cleanup_workflow(versions=concrete),
            {
                concrete[key]: "${" + bindings[key] + "}"
                for key in bindings
            },
        )
    else:
        _fail("unknown Task 12 workflow")
    envelope = {
        "activation_id": inputs.activation_id,
        "activation_ordinal.$": (
            "States.StringToJson('${Task12ActivationOrdinal}')"
        ),
        "generation.$": "States.StringToJson('${Task12Generation}')",
        "generation_text": "${Task12GenerationText}",
        "dispatch_identity_sha256": "${Task12DispatchIdentitySha256}",
        "caller_state_machine_arn.$": "$$.StateMachine.Id",
        "state_machine_execution_arn.$": "$$.Execution.Id",
    }
    for state_name, state in definition["States"].items():
        resource = state.get("Resource")
        if (
            state.get("Type") == "Task"
            and isinstance(resource, str)
            and resource.startswith("${")
        ):
            state["Parameters"] = {
                **envelope,
                "operation_kind": state_name,
                "operation_input.$": "$",
            }
            state.setdefault("ResultPath", "$.task12_last_result")
    substitutions.update(
        {
            "Task12ActivationOrdinal": {"Ref": "Task12ActivationOrdinal"},
            "Task12Generation": {"Ref": "Task12Generation"},
            "Task12GenerationText": {"Ref": "Task12GenerationText"},
            "Task12DispatchIdentitySha256": {
                "Ref": "Task12DispatchIdentitySha256"
            },
        }
    )
    return json.dumps(definition, sort_keys=True, separators=(",", ":")), substitutions, tuple(bindings)


def _workflow_resources(inputs: Task12SupportPlaneInputs) -> dict[str, Any]:
    resources: dict[str, Any] = {}
    for name in ("RetainedLifecycle", "SnapshotCleanup"):
        definition, substitutions, _keys = _definition_and_substitutions(
            inputs, name
        )
        version_refs_by_logical_id = {
            value["Ref"]: value
            for key, value in substitutions.items()
            if key.endswith("VersionArn")
            and key != "SupportDeletionWorkflowVersionArn"
            and isinstance(value, dict)
            and set(value) == {"Ref"}
            and isinstance(value["Ref"], str)
            and value["Ref"].endswith("Version")
        }
        version_refs = [
            version_refs_by_logical_id[logical_id]
            for logical_id in sorted(version_refs_by_logical_id)
        ]
        statements: list[dict[str, Any]] = [{"Effect": "Allow", "Action": "lambda:InvokeFunction", "Resource": version_refs}]
        if name == "RetainedLifecycle":
            statements.extend((
                _start_exact_state_machine_version(
                    state_machine="H1gRetainedLifecycleStateMachine",
                    version="H1gRetainedLifecycleStateMachineVersion",
                    sid="StartExactSupportDelete",
                ),
                {"Sid": "AwaitExactSupportDelete", "Effect": "Allow", "Action": ["states:DescribeExecution", "states:StopExecution"], "Resource": {"Fn::Sub": f"arn:aws:states:{_REGION}:{_ACCOUNT}:execution:keep-glm52-h1g-retained-support-delete:*"}},
                {"Sid": "ManageExactSyncRule", "Effect": "Allow", "Action": ["events:PutTargets", "events:PutRule", "events:DescribeRule"], "Resource": f"arn:aws:events:{_REGION}:{_ACCOUNT}:rule/StepFunctionsGetEventsForStepFunctionsExecutionRule"},
            ))
        resources[f"{name}WorkflowRole"] = _role(name.lower() + "-workflow", statements, service="states.amazonaws.com")
        resources[f"{name}StateMachine"] = {**_RETAIN, "Type": "AWS::StepFunctions::StateMachine", "Properties": {"StateMachineName": f"keep-glm52-h1g-{name.lower()}", "StateMachineType": "STANDARD", "RoleArn": {"Fn::GetAtt": [f"{name}WorkflowRole", "Arn"]}, "DefinitionString": definition, "DefinitionSubstitutions": substitutions}}
        resources[f"{name}StateMachineVersion"] = {**_RETAIN, "Type": "AWS::StepFunctions::StateMachineVersion", "Properties": {"StateMachineArn": {"Ref": f"{name}StateMachine"}, "StateMachineRevisionId": {"Fn::GetAtt": [f"{name}StateMachine", "StateMachineRevisionId"]}}}
        resources[f"{name}FailedExecutionAlarm"] = {**_RETAIN, "Type": "AWS::CloudWatch::Alarm", "Properties": {"AlarmName": f"keep-glm52-h1g-{name.lower()}-failed", "ComparisonOperator": "GreaterThanThreshold", "Dimensions": [{"Name": "StateMachineArn", "Value": {"Ref": f"{name}StateMachine"}}], "EvaluationPeriods": 1, "MetricName": "ExecutionsFailed", "Namespace": "AWS/States", "Period": 60, "Statistic": "Sum", "Threshold": 0, "TreatMissingData": "breaching"}}
    resources["RetainedLifecycleEventInvokeRole"] = _role(
        "retained-lifecycle-event-invoke",
        [
            _start_exact_state_machine_version(
                state_machine="RetainedLifecycleStateMachine",
                version="RetainedLifecycleStateMachineVersion",
            )
        ],
        service="events.amazonaws.com",
    )
    resources["RetainedLifecycleEventRule"] = {
        **_RETAIN,
        "Type": "AWS::Events::Rule",
        "Properties": {
            "EventPattern": {
                "source": ["keep.glm52.task12"],
                "detail-type": ["RETAINED_LIFECYCLE_REQUESTED"],
                "detail": {
                    "record_type": [
                        "glm52_task12_retained_lifecycle_request_v1"
                    ],
                    "reason": ["SUPPORT_FINALIZATION_REQUESTED"],
                },
            },
            "State": "ENABLED",
            "Targets": [
                {
                    "Id": "RetainedLifecycleExactVersion",
                    "Arn": {"Ref": "RetainedLifecycleStateMachineVersion"},
                    "RoleArn": {
                        "Fn::GetAtt": [
                            "RetainedLifecycleEventInvokeRole",
                            "Arn",
                        ]
                    },
                    "InputPath": "$.detail",
                }
            ],
        },
    }
    resources["SnapshotCleanupScheduleInvokeRole"] = _role(
        "snapshot-cleanup-schedule-invoke",
        [
            _start_exact_state_machine_version(
                state_machine="SnapshotCleanupStateMachine",
                version="SnapshotCleanupStateMachineVersion",
            )
        ],
        service="scheduler.amazonaws.com",
    )
    return resources


def _deployment_authority_materialization() -> dict[str, Any]:
    from .task12_postpublication import _OPERATIONS

    records: list[dict[str, Any]] = []
    for logical, (kind, mode, coordinates) in _DEPLOYMENT_SPECS.items():
        callers = (
            (
                "RetainedLifecycleStateMachine",
                "RetainedLifecycleStateMachineVersion",
                "ARM_SCHEDULE",
            ),
            (
                "SnapshotCleanupStateMachine",
                "SnapshotCleanupStateMachineVersion",
                "DELETE_OR_RECONCILE",
            ),
        ) if logical == "SnapshotCleanup" else (
            (
                "RetainedLifecycleStateMachine",
                "RetainedLifecycleStateMachineVersion",
                "RETAINED",
            ),
        )
        for caller, caller_version, operation in callers:
            operation_inventory_key = (
                "SnapshotCleanupSnapshot"
                if logical == "SnapshotCleanup"
                and caller == "SnapshotCleanupStateMachine"
                else (
                    "SnapshotCleanupRetained"
                    if logical == "SnapshotCleanup"
                    else logical
                )
            )
            records.append(
                {
                    "handler_kind": kind,
                    "mode": mode,
                    "operation_family": operation,
                    "operation_kinds": list(
                        _OPERATIONS[operation_inventory_key]
                    ),
                    "function_version_arn": {"Ref": f"{logical}Version"},
                    "lookup_caller_state_machine_arn": {"Ref": caller},
                    "caller_state_machine_version_arn": {
                        "Ref": caller_version
                    },
                    "role_coordinate_fields": coordinates,
                    "resolved_role_coordinates": (
                        {
                            "ledger_table_name": (
                                "FROM_AUTHENTICATED_RETAINED_INPUT"
                            ),
                            "snapshot_cleanup_state_machine_arn": {
                                "Ref": "SnapshotCleanupStateMachine"
                            },
                            "snapshot_cleanup_state_machine_version": {
                                "Fn::Select": [
                                    7,
                                    {
                                        "Fn::Split": [
                                            ":",
                                            {
                                                "Ref": (
                                                    "SnapshotCleanup"
                                                    "StateMachineVersion"
                                                )
                                            },
                                        ]
                                    },
                                ]
                            },
                            "snapshot_cleanup_schedule_invoke_role_arn": {
                                "Fn::GetAtt": [
                                    "SnapshotCleanupScheduleInvokeRole",
                                    "Arn",
                                ]
                            },
                            "snapshot_cleanup_schedule_group_name": "default",
                            "snapshot_cleanup_schedule_name": (
                                "keep-glm52-h1g-snapshot-cleanup-"
                                "${activation_id}"
                            ),
                            "authority": (
                                "FROM_AUTHENTICATED_VERSIONED_S3_COORDINATE"
                            ),
                        }
                        if logical == "SnapshotCleanup"
                        else {
                            field: (
                                "FROM_AUTHENTICATED_POSTCREATE_COORDINATE"
                            )
                            for field in coordinates
                        }
                    ),
                }
            )
    records.append(
        {
            "handler_kind": "RETAINED_TERMINAL_V2",
            "mode": "SUPPORT_OBSERVER",
            "operation_family": "SUPPORT_OBSERVER",
            "operation_kinds": [
                "RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2"
            ],
            "function_version_arn": {"Ref": "TerminalV2Version"},
            "lookup_caller_state_machine_arn": (
                "FROM_AUTHENTICATED_SUPPORT_STACK_READBACK"
            ),
            "caller_state_machine_version_arn": (
                "FROM_AUTHENTICATED_SUPPORT_STACK_READBACK"
            ),
            "role_coordinate_fields": [
                *_DEPLOYMENT_SPECS["TerminalV2"][2],
                "support_observer_version_arn",
            ],
            "resolved_role_coordinates": (
                "FROM_AUTHENTICATED_POSTCREATE_AND_SUPPORT_STACK_COORDINATES"
            ),
        }
    )
    return {
        "record_type": "glm52_task12_deployment_authority_materialization_v1",
        "materialization_phase": "AFTER_RETAINED_STACK_VERSION_PUBLICATION",
        "table_partition_key": "RUN#glm52-sky-20260724",
        "table_sort_key_prefix": "TASK12_LAMBDA_DEPLOYMENT#",
        "lookup_consistency": "STRONGLY_CONSISTENT",
        "lookup_key_fields": [
            "context.invoked_function_arn",
            "event.caller_state_machine_arn",
            "event.activation_id",
            "event.generation_text",
        ],
        "operation_input_sort_key_fields": [
            "event.activation_id",
            "handler_kind",
            "event.operation_kind",
            "event.generation_text",
        ],
        "operation_input_record_count": 32,
        "authority_manifest_count": 10,
        "production_materializer": (
            "coordinate_support_task12_postpublication"
        ),
        "caller_version_proof": (
            "DescribeExecution.stateMachineVersionArn equals deployment record"
        ),
        "inline_config_allowed_in_production": False,
        "records": records,
    }


def render_task12_support_plane_fragment(*, inputs: Task12SupportPlaneInputs) -> dict[str, Any]:
    """Render the deterministic retained-only Task 12 runtime fragment."""
    _validate_inputs(inputs)
    resources = _runtime_resources(inputs)
    resources.update(_workflow_resources(inputs))
    return {
        "Parameters": {
            "Task12ActivationOrdinal": {
                "Type": "String",
                "AllowedPattern": "^[1-9][0-9]*$",
            },
            "Task12Generation": {
                "Type": "String",
                "AllowedPattern": "^[1-9][0-9]*$",
            },
            "Task12GenerationText": {
                "Type": "String",
                "AllowedPattern": "^[0-9]{8}$",
            },
            "Task12DispatchIdentitySha256": {
                "Type": "String",
                "AllowedPattern": "^[0-9a-f]{64}$",
            },
        },
        "Metadata": {
            "task12_retained_ownership": {
                "retained_owned_resources": sorted(resources),
                "support_owned_resources": [],
                "support_stack_may_delete_retained_resources": False,
            },
            "task12_deployment_authority_materialization": (
                _deployment_authority_materialization()
            ),
        },
        "Resources": resources,
    }


def validate_task12_support_plane_fragment(fragment: Mapping[str, Any], *, inputs: Task12SupportPlaneInputs) -> None:
    """Fail closed on any authority, version pin, retention, or graph drift."""
    expected = render_task12_support_plane_fragment(inputs=inputs)
    if not isinstance(fragment, Mapping) or dict(fragment) != expected:
        _fail("Task 12 support-plane fragment must match the closed retained inventory")
