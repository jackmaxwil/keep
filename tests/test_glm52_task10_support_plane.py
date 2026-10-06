"""Task 10 retained sole-sender CloudFormation and workflow contracts."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from glm52_enforcement.task10_support_plane import (
    Task10SupportInputs,
    Task10SupportPlaneError,
    render_task10_support_plane_fragment,
    validate_task10_support_plane_fragment,
)


def _inputs() -> Task10SupportInputs:
    return Task10SupportInputs(
        lambda_code_bucket="keep-glm52-code",
        lambda_code_key="support/support.zip",
        lambda_code_version="opaque-version-1",
        lambda_code_sha256="a" * 64,
        ledger_table_arn=(
            "arn:aws:dynamodb:us-west-2:246813579024:"
            "table/keep-glm52-h1g-ledger-v1"
        ),
        campaign_bucket_arn=(
            "arn:aws:s3:::keep-glm52-models-246813579024-us-west-2"
        ),
    )


def test_retained_task10_rejects_absent_campaign_bucket_identity() -> None:
    """Break caught: Task 10 accepts a bucket absent from retained truth."""

    stale = Task10SupportInputs(
        lambda_code_bucket="keep-glm52-code",
        lambda_code_key="support/support.zip",
        lambda_code_version="opaque-version-1",
        lambda_code_sha256="a" * 64,
        ledger_table_arn=(
            "arn:aws:dynamodb:us-west-2:246813579024:"
            "table/keep-glm52-h1g-ledger-v1"
        ),
        campaign_bucket_arn=(
            "arn:aws:s3:::keep-glm52-h1g-campaign-246813579024-us-west-2"
        ),
    )

    assert validate_task10_support_plane_fragment(
        render_task10_support_plane_fragment(inputs=_inputs()),
        inputs=_inputs(),
    )
    with pytest.raises(Task10SupportPlaneError):
        render_task10_support_plane_fragment(inputs=stale)


def test_retained_task10_has_one_versioned_sender_and_one_watcher() -> None:
    """Break caught: the workflow self-describes instead of launching capacity."""

    fragment = render_task10_support_plane_fragment(inputs=_inputs())
    assert validate_task10_support_plane_fragment(
        fragment,
        inputs=_inputs(),
    )
    resources = fragment["Resources"]
    assert len(resources) == 14
    function = resources["Task10ProductionReconciliationFunction"][
        "Properties"
    ]
    assert function["Handler"] == "task10_sole_sender_handler.main"
    assert function["ReservedConcurrentExecutions"] == 1
    assert function["Timeout"] == 900
    assert function["Environment"]["Variables"][
        "GLM52_TASK10_CAMPAIGN_BUCKET"
    ] == "keep-glm52-models-246813579024-us-west-2"
    definition = json.loads(
        resources["Task10ProductionStateMachine"]["Properties"][
            "DefinitionString"
        ]
    )
    assert definition == {
        "StartAt": "RunInternalSixAzSoleSender",
        "TimeoutSeconds": 1200,
        "States": {
            "RunInternalSixAzSoleSender": {
                "Type": "Task",
                "Resource": "${Task10ProductionReconciliationVersionArn}",
                "Parameters": {
                    "activation_id.$": "$.activation_id",
                    "generation.$": "$.generation",
                    "generation_text.$": "$.generation_text",
                    "execution_arn.$": "$$.Execution.Id",
                },
                "End": True,
            }
        },
    }
    assert "states:describeExecution" not in json.dumps(fragment)
    assert "SoleSenderCapacityPlan" not in json.dumps(fragment)
    watcher = resources["Task9LiabilityWatcherFunction"]["Properties"]
    assert watcher["FunctionName"] == (
        "keep-glm52-h1g-worker-launch-custody"
    )
    assert watcher["Handler"] == "task9_liability_watcher_handler.main"
    assert watcher["ReservedConcurrentExecutions"] == 1
    assert resources["Task9LiabilityWatcherFunctionVersion"][
        "Properties"
    ] == {
        "FunctionName": {"Ref": "Task9LiabilityWatcherFunction"},
        "Description": "code=" + _inputs().lambda_code_sha256,
    }
    assert not any(
        resource["Type"] == "AWS::Lambda::Alias"
        for resource in resources.values()
    )


def test_task9_watcher_is_retained_but_disabled_until_exact_invocation() -> None:
    """Break caught: rollback deletes custody or a schedule arms it early."""

    fragment = render_task10_support_plane_fragment(inputs=_inputs())
    resources = fragment["Resources"]
    for logical_id in (
        "Task9LiabilityWatcherRole",
        "Task9LiabilityWatcherLogGroup",
        "Task9LiabilityWatcherFunction",
        "Task9LiabilityWatcherFunctionVersion",
        "Task9LiabilityWatcherErrorAlarm",
    ):
        assert resources[logical_id]["DeletionPolicy"] == "Retain"
        assert resources[logical_id]["UpdateReplacePolicy"] == "Retain"
    assert not any(
        resource["Type"]
        in {
            "AWS::Events::Rule",
            "AWS::Lambda::EventSourceMapping",
        }
        for resource in resources.values()
    )
    mutant = json.loads(json.dumps(fragment))
    del mutant["Resources"]["Task9LiabilityWatcherFunction"][
        "DeletionPolicy"
    ]
    with pytest.raises(Task10SupportPlaneError, match="drifted|retained"):
        validate_task10_support_plane_fragment(
            mutant,
            inputs=_inputs(),
        )


def test_sole_sender_iam_is_closed_to_ledger_capacity_and_terminal_output() -> None:
    """Break caught: Task 10 gains broad AWS mutation or replay authority."""

    resources = render_task10_support_plane_fragment(inputs=_inputs())[
        "Resources"
    ]
    statements = resources["Task10ProductionReconciliationRole"]["Properties"][
        "Policies"
    ][0]["PolicyDocument"]["Statement"]
    actions = {
        action
        for statement in statements
        for action in (
            statement["Action"]
            if isinstance(statement["Action"], list)
            else [statement["Action"]]
        )
    }
    assert actions == {
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:UpdateItem",
        "ec2:RunInstances",
        "ec2:DescribeInstances",
        "ec2:DescribeVolumes",
        "ec2:CreateTags",
        "iam:PassRole",
        "s3:PutObject",
        "s3:GetObject",
        "s3:GetObjectVersion",
        "s3:ListBucketVersions",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "sts:GetCallerIdentity",
    }
    invoke = next(
        statement
        for statement in resources["Task10ProductionWorkflowRole"]["Properties"][
            "Policies"
        ][0]["PolicyDocument"]["Statement"]
        if statement["Action"] == "lambda:InvokeFunction"
    )
    assert invoke["Resource"] == {
        "Ref": "Task10ProductionReconciliationFunctionVersion"
    }
    encoded = json.dumps(statements, sort_keys=True)
    wildcard = [
        statement
        for statement in statements
        if statement["Resource"] == "*"
    ]
    assert wildcard == [
        next(
            statement
            for statement in statements
            if statement["Action"] == "sts:GetCallerIdentity"
        ),
        next(
            statement
            for statement in statements
            if statement["Action"]
            == ["ec2:DescribeInstances", "ec2:DescribeVolumes"]
        ),
    ]
    assert "ec2:RequestSpotInstances" not in encoded
    assert "ec2:PurchaseCapacityBlock" not in encoded
    pass_role = next(
        statement
        for statement in statements
        if statement["Action"] == "iam:PassRole"
    )
    assert pass_role == {
        "Sid": "PassOnlyApprovedGpuWorkerRole",
        "Effect": "Allow",
        "Action": "iam:PassRole",
        "Resource": (
            "arn:aws:iam::246813579024:"
            "role/keep-glm52-gpu-worker"
        ),
        "Condition": {
            "StringEquals": {
                "iam:PassedToService": "ec2.amazonaws.com"
            }
        },
    }
    watcher_statements = resources["Task9LiabilityWatcherRole"][
        "Properties"
    ]["Policies"][0]["PolicyDocument"]["Statement"]
    assert {
        action
        for statement in watcher_statements
        for action in (
            statement["Action"]
            if isinstance(statement["Action"], list)
            else [statement["Action"]]
        )
    } == {
        "sts:GetCallerIdentity",
        "dynamodb:GetItem",
        "dynamodb:Query",
        "dynamodb:TransactWriteItems",
        "dynamodb:UpdateItem",
        "ec2:DescribeInstances",
        "ec2:TerminateInstances",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
    }
    task9_ledger = next(
        statement
        for statement in watcher_statements
        if "dynamodb:GetItem" in statement["Action"]
    )
    assert task9_ledger["Condition"] == {
        "ForAllValues:StringLike": {
            "dynamodb:LeadingKeys": [
                "RUN#glm52-sky-20260724"
            ]
        }
    }


def test_permission_binds_only_exact_published_workflow_version() -> None:
    """Break caught: an unversioned or foreign workflow can invoke the sender."""

    permission = render_task10_support_plane_fragment(inputs=_inputs())[
        "Resources"
    ]["Task10ProductionReconciliationInvokePermission"]["Properties"]
    assert permission == {
        "Action": "lambda:InvokeFunction",
        "FunctionName": {
            "Ref": "Task10ProductionReconciliationFunctionVersion"
        },
        "Principal": "states.amazonaws.com",
        "SourceAccount": "246813579024",
        "SourceArn": {"Ref": "Task10ProductionStateMachineVersion"},
    }


def test_support_archive_contains_executable_task10_sender_deterministically(
    tmp_path: Path,
) -> None:
    """Break caught: CloudFormation names a Task 10 handler absent from the zip."""

    script = (
        Path(__file__).resolve().parents[1]
        / "aws/glm52-gpu/scripts/package_h1g_support_lambdas.py"
    )
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    subprocess.run(
        [str(Path(__file__).resolve().parents[1] / ".venv/bin/python"), str(script), str(first)],
        check=True,
    )
    subprocess.run(
        [str(Path(__file__).resolve().parents[1] / ".venv/bin/python"), str(script), str(second)],
        check=True,
    )
    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
        assert "task10_sole_sender_handler.py" in names
        assert "task10_sole_sender_runtime.py" in names
        assert "task9_liability_watcher_handler.py" in names
        assert "task9_liability_watcher_runtime.py" in names
        assert "glm52_enforcement/task10_sole_sender.py" in names
        assert len(names) == len(set(names))
    assert first.read_bytes() == second.read_bytes()
    imported = subprocess.run(
        [
            str(
                Path(__file__).resolve().parents[1]
                / ".venv/bin/python"
            ),
            "-c",
            (
                "import sys;"
                f"sys.path.insert(0, {str(first)!r});"
                "import task9_liability_watcher_handler;"
                "assert callable("
                "task9_liability_watcher_handler.main)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert imported.returncode == 0, imported.stderr
