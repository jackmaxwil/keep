from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from botocore.exceptions import EndpointConnectionError

import glm52_enforcement.task13_retained_foundation as foundation_api
from glm52_enforcement.task13_retained_foundation import (
    CHANGE_SET_NAME as PRODUCTION_CHANGE_SET_NAME,
)
from glm52_enforcement.task13_retained_foundation import (
    DEPLOYMENT_ROLE_NAME,
    FENCE_ROLE_NAME,
    FoundationError,
    FoundationServices,
    apply_retained_foundation,
    compose_retained_foundation_template,
    recover_retained_foundation,
    validate_foundation_change_set,
)
from glm52_enforcement.task13_reviewed_artifacts import REVIEWED_ARTIFACT_KEYS

ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
STACK_NAME = "keep-glm52-gpu"
STACK_ID = (
    "arn:aws:cloudformation:us-west-2:246813579024:stack/"
    "keep-glm52-gpu/aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
)
CHANGE_SET_NAME = "glm52-task13-retained-foundation-v1"
CHANGE_SET_ID = (
    "arn:aws:cloudformation:us-west-2:246813579024:changeSet/"
    f"{CHANGE_SET_NAME}/11111111-2222-4333-8444-555555555555"
)
ROOT = Path(__file__).resolve().parents[1]
INVOCATION_PREPARER = (
    ROOT
    / "aws/glm52-gpu/scripts"
    / "prepare_glm52_task13_staged_deployment_invocation.py"
)
SCRIPT = ROOT / "aws/glm52-gpu/scripts/apply_glm52_task13_retained_foundation.py"


def test_retained_foundation_production_module_exists() -> None:
    """Break caught: the guarded retained-foundation route is absent."""

    assert (
        importlib.util.find_spec("glm52_enforcement.task13_retained_foundation")
        is not None
    )


def test_retained_foundation_one_shot_cli_exists() -> None:
    """Break caught: no operator entrypoint can run the guarded transaction."""

    assert SCRIPT.is_file()


def test_foundation_transport_does_not_consume_final_retained_template_key() -> None:
    """Break caught: Phase 1 prevents the later final retained publication."""

    assert foundation_api.FOUNDATION_TEMPLATE_ARTIFACT_KIND == (
        "RETAINED_FOUNDATION_TEMPLATE"
    )
    assert foundation_api.FOUNDATION_TEMPLATE_KEY == (
        "task13/templates/retained-foundation.yaml"
    )
    assert REVIEWED_ARTIFACT_KEYS["RETAINED_TEMPLATE"] == (
        "task13/templates/retained.yaml"
    )
    assert (
        foundation_api.FOUNDATION_TEMPLATE_KEY
        != (REVIEWED_ARTIFACT_KEYS["RETAINED_TEMPLATE"])
    )


def _original_template() -> dict[str, object]:
    return {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Description": "authoritative live original",
        "Parameters": {
            "ProjectTag": {"Type": "String", "Default": "keep-glm52"},
            "GpuAmiId": {"Type": "AWS::EC2::Image::Id"},
        },
        "Mappings": {"RegionMap": {"us-west-2": {"Arch": "x86_64"}}},
        "Conditions": {"Never": {"Fn::Equals": ["a", "b"]}},
        "Resources": {
            "Vpc": {
                "Type": "AWS::EC2::VPC",
                "Properties": {"CidrBlock": "10.20.0.0/16"},
            },
            "PublicSubnet": {
                "Type": "AWS::EC2::Subnet",
                "Properties": {"VpcId": {"Ref": "Vpc"}},
            },
            "PublicRouteTable": {
                "Type": "AWS::EC2::RouteTable",
                "Properties": {"VpcId": {"Ref": "Vpc"}},
            },
            "S3GatewayEndpoint": {
                "Type": "AWS::EC2::VPCEndpoint",
                "Properties": {
                    "VpcId": {"Ref": "Vpc"},
                    "VpcEndpointType": "Gateway",
                    "RouteTableIds": [{"Ref": "PublicRouteTable"}],
                },
            },
            "ModelBucket": {
                "Type": "AWS::S3::Bucket",
                "DeletionPolicy": "Retain",
                "UpdateReplacePolicy": "Retain",
                "Properties": {
                    "BucketName": {
                        "Fn::Sub": (
                            "keep-glm52-models-${AWS::AccountId}-${AWS::Region}"
                        )
                    }
                },
            },
            "GpuLaunchTemplate": {
                "Type": "AWS::EC2::LaunchTemplate",
                "Properties": {"LaunchTemplateName": "keep-glm52-gpu-node"},
            },
            "DormantController": {
                "Type": "AWS::Lambda::Function",
                "Condition": "Never",
            },
        },
        "Outputs": {
            "VpcId": {"Value": {"Ref": "Vpc"}},
            "SubnetId": {"Value": {"Ref": "PublicSubnet"}},
            "ModelBucketName": {
                "Value": {"Ref": "ModelBucket"},
                "Export": {"Name": {"Fn::Sub": "${AWS::StackName}-ModelBucket"}},
            },
            "LaunchTemplateId": {
                "Value": {"Ref": "GpuLaunchTemplate"},
                "Export": {"Name": {"Fn::Sub": "${AWS::StackName}-LaunchTemplate"}},
            },
        },
    }


def test_active_resource_inventory_evaluates_cloudformation_conditions() -> None:
    """Break caught: false resources are required or true resources are omitted."""

    template = {
        "Conditions": {
            "FeatureEnabled": {"Fn::Equals": [{"Ref": "EnableFeature"}, "true"]},
            "NestedEnabled": {
                "Fn::And": [
                    {"Condition": "FeatureEnabled"},
                    {
                        "Fn::Or": [
                            {"Fn::Equals": [{"Ref": "AWS::Region"}, "us-west-2"]},
                            {"Fn::Equals": ["a", "b"]},
                        ]
                    },
                    {"Fn::Not": [{"Fn::Equals": ["a", "b"]}]},
                ]
            },
        },
        "Resources": {
            "Always": {"Type": "AWS::S3::Bucket"},
            "Active": {
                "Type": "AWS::Lambda::Function",
                "Condition": "NestedEnabled",
            },
            "Dormant": {
                "Type": "AWS::SQS::Queue",
                "Condition": "FeatureEnabled",
            },
        },
    }

    assert foundation_api._active_resource_types(
        template,
        [{"ParameterKey": "EnableFeature", "ParameterValue": "true"}],
    ) == {
        "Active": "AWS::Lambda::Function",
        "Always": "AWS::S3::Bucket",
        "Dormant": "AWS::SQS::Queue",
    }
    assert foundation_api._active_resource_types(
        template,
        [{"ParameterKey": "EnableFeature", "ParameterValue": "false"}],
    ) == {"Always": "AWS::S3::Bucket"}


def test_active_resource_inventory_rejects_unsupported_condition_intrinsic() -> None:
    """Break caught: an unevaluated condition silently weakens live proof."""

    with pytest.raises(FoundationError, match="condition expression"):
        foundation_api._active_resource_types(
            {
                "Conditions": {"Unsupported": {"Fn::If": ["Other", True, False]}},
                "Resources": {"Always": {"Type": "AWS::S3::Bucket"}},
            },
            [],
        )


def test_active_resource_inventory_rejects_non_string_equals_operands() -> None:
    """Break caught: Python equality diverges from the reviewed string contract."""

    with pytest.raises(FoundationError, match="Fn::Equals operands"):
        foundation_api._active_resource_types(
            {
                "Conditions": {
                    "Unsupported": {"Fn::Equals": [{"Ref": "FeatureFlag"}, True]}
                },
                "Resources": {
                    "Guarded": {
                        "Type": "AWS::S3::Bucket",
                        "Condition": "Unsupported",
                    }
                },
            },
            [{"ParameterKey": "FeatureFlag", "ParameterValue": "true"}],
        )


def _expected_changes() -> list[dict[str, object]]:
    return [
        {
            "Type": "Resource",
            "ResourceChange": {
                "Action": "Add",
                "LogicalResourceId": logical_id,
                "ResourceType": resource_type,
            },
        }
        for logical_id, resource_type in (
            ("CampaignKmsKey", "AWS::KMS::Key"),
            ("H1gCloudFormationDeploymentRole", "AWS::IAM::Role"),
            ("H1gFenceServiceRole", "AWS::IAM::Role"),
            ("H1gLedger", "AWS::DynamoDB::Table"),
        )
    ]


def _safe_launch_template_refresh() -> dict[str, object]:
    return {
        "Type": "Resource",
        "ResourceChange": {
            "Action": "Modify",
            "LogicalResourceId": "GpuLaunchTemplate",
            "ResourceType": "AWS::EC2::LaunchTemplate",
            "Replacement": "False",
            "Details": [
                {
                    "Target": {
                        "Attribute": "Properties",
                        "Name": "LaunchTemplateData",
                        "RequiresRecreation": "Never",
                        "Path": "/Properties/LaunchTemplateData",
                        "BeforeValue": "(Truncated-Signature):before",
                        "AfterValue": "(Truncated-Signature):after",
                        "AttributeChangeType": "Modify",
                    },
                    "Evaluation": "Static",
                    "ChangeSource": "DirectModification",
                }
            ],
        },
    }


def _change_set(
    *,
    changes: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "ChangeSetId": CHANGE_SET_ID,
        "ChangeSetName": CHANGE_SET_NAME,
        "StackId": STACK_ID,
        "StackName": STACK_NAME,
        "Status": "CREATE_COMPLETE",
        "ExecutionStatus": "AVAILABLE",
        "Parameters": [
            {"ParameterKey": "GpuAmiId", "ParameterValue": "ami-exact"},
            {"ParameterKey": "ProjectTag", "ParameterValue": "keep-glm52"},
        ],
        "Capabilities": ["CAPABILITY_NAMED_IAM"],
        "Changes": _expected_changes() if changes is None else changes,
    }


def test_composer_preserves_complete_original_and_adds_only_reviewed_foundation() -> (
    None
):
    """Break caught: composing drops live sections, resources, or legacy exports."""

    original = _original_template()
    frozen = deepcopy(original)
    composed = compose_retained_foundation_template(original)

    assert original == frozen
    assert composed["AWSTemplateFormatVersion"] == original["AWSTemplateFormatVersion"]
    assert composed["Description"] == original["Description"]
    assert composed["Parameters"] == original["Parameters"]
    assert composed["Mappings"] == original["Mappings"]
    assert composed["Conditions"] == original["Conditions"]

    additions = {
        "CampaignKmsKey",
        "H1gCloudFormationDeploymentRole",
        "H1gFenceServiceRole",
        "H1gLedger",
    }
    assert set(composed["Resources"]) == set(original["Resources"]) | additions
    for logical_id, resource in original["Resources"].items():
        assert composed["Resources"][logical_id] == resource
    assert "CampaignKmsAlias" not in composed["Resources"]
    assert (
        sum(
            resource["Type"] == "AWS::EC2::VPCEndpoint"
            for resource in composed["Resources"].values()
        )
        == 1
    )
    assert (
        composed["Resources"]["S3GatewayEndpoint"]
        == original["Resources"]["S3GatewayEndpoint"]
    )
    assert composed["Resources"]["ModelBucket"] == original["Resources"]["ModelBucket"]
    assert (
        composed["Resources"]["GpuLaunchTemplate"]
        == original["Resources"]["GpuLaunchTemplate"]
    )

    assert (
        composed["Outputs"]["ModelBucketName"] == original["Outputs"]["ModelBucketName"]
    )
    assert (
        composed["Outputs"]["LaunchTemplateId"]
        == original["Outputs"]["LaunchTemplateId"]
    )
    assert composed["Outputs"]["VpcId"] == {
        "Value": {"Ref": "Vpc"},
        "Export": {"Name": "KeepGlm52VpcId"},
    }
    assert composed["Outputs"]["SubnetId"] == {
        "Value": {"Ref": "PublicSubnet"},
        "Export": {"Name": "KeepGlm52PrimaryPublicSubnetId"},
    }
    assert composed["Outputs"]["CampaignKmsKeyArn"] == {
        "Value": {"Fn::GetAtt": ["CampaignKmsKey", "Arn"]},
        "Export": {"Name": "KeepGlm52CampaignKmsKeyArn"},
    }
    assert composed["Outputs"]["H1gLedgerArn"] == {
        "Value": {"Fn::GetAtt": ["H1gLedger", "Arn"]},
        "Export": {"Name": "KeepGlm52H1gLedgerArn"},
    }


def test_original_template_parser_preserves_cloudformation_yaml_intrinsics() -> None:
    """Break caught: GetTemplate Original YAML is rejected or loses intrinsic meaning."""

    parsed = foundation_api._template_body(
        """
AWSTemplateFormatVersion: "2010-09-09"
Resources:
  Vpc:
    Type: AWS::EC2::VPC
  Bucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub "bucket-${AWS::AccountId}"
Outputs:
  VpcId:
    Value: !Ref Vpc
  BucketArn:
    Value: !GetAtt Bucket.Arn
"""
    )
    assert parsed["Resources"]["Bucket"]["Properties"]["BucketName"] == {
        "Fn::Sub": "bucket-${AWS::AccountId}"
    }
    assert parsed["Outputs"]["VpcId"]["Value"] == {"Ref": "Vpc"}
    assert parsed["Outputs"]["BucketArn"]["Value"] == {"Fn::GetAtt": ["Bucket", "Arn"]}


def test_foundation_resources_are_retained_encrypted_and_named_exactly() -> None:
    """Break caught: the durable key/table can be deleted or drift in identity."""

    resources = compose_retained_foundation_template(_original_template())["Resources"]
    key = resources["CampaignKmsKey"]
    assert key["Type"] == "AWS::KMS::Key"
    assert key["DeletionPolicy"] == "Retain"
    assert key["UpdateReplacePolicy"] == "Retain"
    assert key["Properties"]["EnableKeyRotation"] is True
    assert key["Properties"]["PendingWindowInDays"] == 30

    ledger = resources["H1gLedger"]
    assert ledger == {
        "Type": "AWS::DynamoDB::Table",
        "DeletionPolicy": "Retain",
        "UpdateReplacePolicy": "Retain",
        "Properties": {
            "TableName": "keep-glm52-h1g-ledger-v1",
            "BillingMode": "PAY_PER_REQUEST",
            "AttributeDefinitions": [
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
            ],
            "KeySchema": [
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            "DeletionProtectionEnabled": True,
            "PointInTimeRecoverySpecification": {"PointInTimeRecoveryEnabled": True},
            "SSESpecification": {
                "SSEEnabled": True,
                "SSEType": "KMS",
                "KMSMasterKeyId": {"Fn::GetAtt": ["CampaignKmsKey", "Arn"]},
            },
        },
    }


def test_role_policies_are_explicit_separate_and_have_no_admin_wildcard() -> None:
    """Break caught: the fence role gains deployment power or either role is admin."""

    resources = compose_retained_foundation_template(_original_template())["Resources"]
    deployment = resources["H1gCloudFormationDeploymentRole"]
    fence = resources["H1gFenceServiceRole"]
    assert deployment["Properties"]["RoleName"] == (
        "keep-glm52-h1g-cloudformation-deployment"
    )
    assert fence["Properties"]["RoleName"] == ("keep-glm52-h1g-fence-service")
    assert "ManagedPolicyArns" not in deployment["Properties"]
    assert "ManagedPolicyArns" not in fence["Properties"]

    deployment_actions: set[str] = set()
    for policy in deployment["Properties"]["Policies"]:
        for statement in policy["PolicyDocument"]["Statement"]:
            action = statement["Action"]
            deployment_actions.update([action] if isinstance(action, str) else action)
    fence_actions: set[str] = set()
    for policy in fence["Properties"]["Policies"]:
        for statement in policy["PolicyDocument"]["Statement"]:
            action = statement["Action"]
            fence_actions.update([action] if isinstance(action, str) else action)

    assert "*" not in deployment_actions
    assert "*" not in fence_actions
    assert all(not action.endswith(":*") for action in deployment_actions)
    assert all(not action.endswith(":*") for action in fence_actions)
    assert fence_actions == {"s3:GetBucketPolicy", "s3:PutBucketPolicy"}
    assert fence_actions.isdisjoint(deployment_actions)
    assert "iam:PassRole" in deployment_actions
    assert "cloudformation:ExecuteChangeSet" not in fence_actions
    assert "arn:aws:iam::aws:policy/AdministratorAccess" not in repr(resources)


def _current_retained_runtime_fragments() -> tuple[
    dict[str, object],
    dict[str, object],
]:
    fixture_path = ROOT / "tests/test_glm52_enforcement_support_plane.py"
    spec = importlib.util.spec_from_file_location(
        "_task13_retained_support_fixture",
        fixture_path,
    )
    assert spec is not None and spec.loader is not None
    fixture = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fixture
    spec.loader.exec_module(fixture)
    from glm52_enforcement.support_plane import (
        build_postcreate_retained_support_fragment,
        build_pre_support_retained_runtime_fragment,
        pre_support_runtime_inputs_from_mapping,
        support_inputs_from_mapping,
    )

    pre_support = build_pre_support_retained_runtime_fragment(
        pre_support_runtime_inputs_from_mapping(
            fixture._pre_support_runtime_input_mapping()
        )
    )
    postcreate = build_postcreate_retained_support_fragment(
        support_inputs_from_mapping(fixture._support_inputs())
    )
    assert isinstance(pre_support, dict)
    assert isinstance(postcreate, dict)
    return pre_support, postcreate


def _deployment_statements() -> list[dict[str, object]]:
    role = foundation_api._deployment_role()
    return [
        statement
        for policy in role["Properties"]["Policies"]
        for statement in policy["PolicyDocument"]["Statement"]
    ]


def test_deployment_role_covers_exact_current_retained_resource_lifecycle() -> None:
    """Break caught: the retained service role cannot deploy the reviewed graph."""

    pre_support, postcreate = _current_retained_runtime_fragments()
    pre_counts = Counter(
        resource["Type"] for resource in pre_support["Resources"].values()
    )
    post_counts = Counter(
        resource["Type"] for resource in postcreate["Resources"].values()
    )
    assert pre_counts == {
        "AWS::CloudWatch::Alarm": 12,
        "AWS::Events::Rule": 1,
        "AWS::IAM::Role": 15,
        "AWS::Lambda::Function": 10,
        "AWS::Lambda::Permission": 10,
        "AWS::Lambda::Version": 10,
        "AWS::Logs::LogGroup": 10,
        "AWS::StepFunctions::StateMachine": 3,
        "AWS::StepFunctions::StateMachineVersion": 3,
    }
    assert post_counts == {
        "AWS::CloudWatch::Alarm": 1,
        "AWS::IAM::Role": 4,
        "AWS::Lambda::Function": 1,
        "AWS::Lambda::Version": 1,
        "AWS::Scheduler::Schedule": 4,
        "AWS::SNS::Topic": 1,
        "AWS::StepFunctions::StateMachine": 1,
        "AWS::StepFunctions::StateMachineVersion": 1,
    }
    required_by_type = {
        "AWS::CloudWatch::Alarm": {
            "cloudwatch:DeleteAlarms",
            "cloudwatch:DescribeAlarms",
            "cloudwatch:ListTagsForResource",
            "cloudwatch:PutMetricAlarm",
            "cloudwatch:TagResource",
            "cloudwatch:UntagResource",
        },
        "AWS::Events::Rule": {
            "events:DeleteRule",
            "events:DescribeRule",
            "events:ListTagsForResource",
            "events:ListTargetsByRule",
            "events:PutRule",
            "events:PutTargets",
            "events:RemoveTargets",
            "events:TagResource",
            "events:UntagResource",
        },
        "AWS::IAM::Role": {
            "iam:CreateRole",
            "iam:DeleteRole",
            "iam:DeleteRolePolicy",
            "iam:GetRole",
            "iam:GetRolePolicy",
            "iam:ListRolePolicies",
            "iam:ListRoleTags",
            "iam:PutRolePolicy",
            "iam:TagRole",
            "iam:UntagRole",
            "iam:UpdateAssumeRolePolicy",
            "iam:UpdateRoleDescription",
        },
        "AWS::Lambda::Function": {
            "lambda:CreateFunction",
            "lambda:DeleteFunction",
            "lambda:DeleteFunctionConcurrency",
            "lambda:GetFunction",
            "lambda:GetFunctionConcurrency",
            "lambda:ListTags",
            "lambda:PutFunctionConcurrency",
            "lambda:TagResource",
            "lambda:UntagResource",
            "lambda:UpdateFunctionCode",
            "lambda:UpdateFunctionConfiguration",
        },
        "AWS::Lambda::Permission": {
            "lambda:AddPermission",
            "lambda:GetPolicy",
            "lambda:RemovePermission",
        },
        "AWS::Lambda::Version": {
            "lambda:DeleteFunction",
            "lambda:GetFunction",
            "lambda:ListVersionsByFunction",
            "lambda:PublishVersion",
        },
        "AWS::Logs::LogGroup": {
            "logs:CreateLogGroup",
            "logs:DeleteLogGroup",
            "logs:DeleteRetentionPolicy",
            "logs:DescribeLogGroups",
            "logs:ListTagsForResource",
            "logs:PutRetentionPolicy",
            "logs:TagResource",
            "logs:UntagResource",
        },
        "AWS::Scheduler::Schedule": {
            "scheduler:CreateSchedule",
            "scheduler:DeleteSchedule",
            "scheduler:GetSchedule",
            "scheduler:UpdateSchedule",
        },
        "AWS::SNS::Topic": {
            "sns:CreateTopic",
            "sns:DeleteTopic",
            "sns:GetTopicAttributes",
            "sns:ListTagsForResource",
            "sns:SetTopicAttributes",
            "sns:TagResource",
            "sns:UntagResource",
        },
        "AWS::StepFunctions::StateMachine": {
            "states:CreateStateMachine",
            "states:DeleteStateMachine",
            "states:DescribeStateMachine",
            "states:ListTagsForResource",
            "states:TagResource",
            "states:UntagResource",
            "states:UpdateStateMachine",
        },
        "AWS::StepFunctions::StateMachineVersion": {
            "states:DeleteStateMachineVersion",
            "states:ListStateMachineVersions",
            "states:PublishStateMachineVersion",
        },
    }
    assert set(pre_counts) | set(post_counts) == set(required_by_type)

    foundation_actions = {
        "dynamodb:DescribeContinuousBackups",
        "dynamodb:DescribeTable",
        "dynamodb:DescribeTimeToLive",
        "dynamodb:ListTagsOfResource",
        "dynamodb:TagResource",
        "dynamodb:UntagResource",
        "dynamodb:UpdateContinuousBackups",
        "dynamodb:UpdateTable",
        "kms:DescribeKey",
        "kms:EnableKeyRotation",
        "kms:GetKeyPolicy",
        "kms:GetKeyRotationStatus",
        "kms:ListResourceTags",
        "kms:PutKeyPolicy",
        "kms:TagResource",
        "kms:UntagResource",
        "kms:UpdateKeyDescription",
    }
    expected_actions = (
        {action for actions in required_by_type.values() for action in actions}
        | foundation_actions
        | {"iam:PassRole"}
    )
    observed_actions = {
        action
        for statement in _deployment_statements()
        for action in (
            [statement["Action"]]
            if isinstance(statement["Action"], str)
            else statement["Action"]
        )
    }
    assert observed_actions == expected_actions


def test_deployment_role_passes_only_roles_used_by_retained_resources() -> None:
    """Break caught: CloudFormation can pass an unrelated or launch-capable role."""

    statements = {statement["Sid"]: statement for statement in _deployment_statements()}
    expected = {
        "PassRetainedLambdaExecutionRoles": (
            "lambda.amazonaws.com",
            {
                "keep-glm52-h1g-execution-observer",
                "keep-glm52-h1g-finalizer",
                "keep-glm52-h1g-h1g-drained-writer",
                "keep-glm52-h1g-operator-disposition-writer",
                "keep-glm52-h1g-orphan-audit",
                "keep-glm52-h1g-retained-support-lifecycle",
                "keep-glm52-h1g-snapshot-cleanup",
                "keep-glm52-h1g-task10-capacity-reconciliation",
                "keep-glm52-h1g-terminal-v2-writer",
                "keep-glm52-h1g-worker-drain-signal",
            },
        ),
        "PassRetainedStateMachineRoles": (
            "states.amazonaws.com",
            {
                "keep-glm52-h1g-production-workflow",
                "keep-glm52-h1g-retained-lifecycle-workflow",
                "keep-glm52-h1g-retainedlifecycle-workflow",
                "keep-glm52-h1g-snapshotcleanup-workflow",
            },
        ),
        "PassRetainedEventInvokeRole": (
            "events.amazonaws.com",
            {"keep-glm52-h1g-retained-lifecycle-event-invoke"},
        ),
        "PassRetainedSchedulerInvokeRole": (
            "scheduler.amazonaws.com",
            {"keep-glm52-h1g-retained-support-schedule-invoke"},
        ),
    }
    pass_statements = {
        sid: statement
        for sid, statement in statements.items()
        if statement["Action"] == "iam:PassRole"
    }
    assert set(pass_statements) == set(expected)
    for sid, (service, role_names) in expected.items():
        statement = pass_statements[sid]
        assert statement["Condition"] == {
            "StringEquals": {"iam:PassedToService": service}
        }
        resources = statement["Resource"]
        assert isinstance(resources, list)
        assert {
            resource["Fn::Sub"].rsplit("/", 1)[-1] for resource in resources
        } == role_names

    forbidden = {
        "ec2:CreateFleet",
        "ec2:RequestSpotInstances",
        "ec2:RunInstances",
        "ec2:TerminateInstances",
        "s3:DeleteObject",
        "s3:PutObject",
        "sqs:SendMessage",
    }
    actions = {
        action
        for statement in statements.values()
        for action in (
            [statement["Action"]]
            if isinstance(statement["Action"], str)
            else statement["Action"]
        )
    }
    assert actions.isdisjoint(forbidden)
    star_statements = [
        statement for statement in statements.values() if statement["Resource"] == "*"
    ]
    assert star_statements == [
        {
            "Sid": "DescribeRetainedLogGroups",
            "Effect": "Allow",
            "Action": "logs:DescribeLogGroups",
            "Resource": "*",
            "Condition": {"StringEquals": {"aws:RequestedRegion": REGION}},
        }
    ]


def test_deployment_role_inline_policy_fits_live_iam_quota() -> None:
    """Break caught: the exact policy cannot be created by live IAM."""

    role = foundation_api._deployment_role()
    compact_characters = sum(
        len(
            json.dumps(
                policy["PolicyDocument"],
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        for policy in role["Properties"]["Policies"]
    )
    assert compact_characters <= 10_240


def test_cloudformation_fake_exposes_only_the_real_termination_field_api() -> None:
    """Break caught: tests green-light a nonexistent CloudFormation operation."""

    cloudformation = _FakeCloudFormation()
    assert not hasattr(cloudformation, "describe_termination_protection")
    stack = cloudformation.describe_stacks(StackName=STACK_NAME)["Stacks"][0]
    assert stack["EnableTerminationProtection"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        "logical-id",
        "export-name",
        "missing-vpc",
        "duplicate-endpoint",
    ),
)
def test_composer_rejects_preexisting_conflicts_and_absent_live_anchors(
    mutation: str,
) -> None:
    """Break caught: a conflict is silently adopted or a live anchor is guessed."""

    original = _original_template()
    if mutation == "logical-id":
        original["Resources"]["CampaignKmsKey"] = {"Type": "AWS::KMS::Key"}
    elif mutation == "export-name":
        original["Outputs"]["ForeignVpc"] = {
            "Value": "vpc-foreign",
            "Export": {"Name": "KeepGlm52VpcId"},
        }
    elif mutation == "missing-vpc":
        del original["Resources"]["Vpc"]
    else:
        original["Resources"]["SecondS3Endpoint"] = {
            "Type": "AWS::EC2::VPCEndpoint",
            "Properties": {"VpcEndpointType": "Gateway"},
        }
    with pytest.raises(FoundationError):
        compose_retained_foundation_template(original)


def test_execution_template_preserves_live_yaml_bytes_and_adds_only_foundation() -> (
    None
):
    """Break caught: parsing and canonicalizing the live template creates phantom changes."""

    original = _original_template()
    raw = "# exact live Original bytes; do not reserialize\n" + yaml.safe_dump(
        original, sort_keys=False
    )
    composed = compose_retained_foundation_template(original)

    execution = foundation_api._execution_template_bytes(
        raw_template_body=raw,
        original_template=original,
        retained_template=composed,
    )
    decoded = execution.decode("utf-8")

    assert decoded.startswith("# exact live Original bytes; do not reserialize\n")
    assert foundation_api._template_body(decoded) == composed
    assert decoded.count("  CampaignKmsKey:\n") == 1
    assert decoded.count("  H1gLedger:\n") == 1
    assert decoded.count("  H1gCloudFormationDeploymentRole:\n") == 1
    assert decoded.count("  H1gFenceServiceRole:\n") == 1
    assert decoded.count("  CampaignKmsKeyArn:\n") == 1
    assert decoded.count("  H1gLedgerArn:\n") == 1


def test_change_set_accepts_exact_nonreplacement_launch_template_refresh() -> None:
    """Break caught: mutable SSM AMI resolution makes an exact stack update impossible."""

    validated = validate_foundation_change_set(
        _change_set(changes=_expected_changes() + [_safe_launch_template_refresh()]),
        stack_id=STACK_ID,
        change_set_name=CHANGE_SET_NAME,
        expected_parameters=[
            {"ParameterKey": "GpuAmiId", "ParameterValue": "ami-exact"},
            {"ParameterKey": "ProjectTag", "ParameterValue": "keep-glm52"},
        ],
    )

    assert len(validated["Changes"]) == 5


def test_change_set_accepts_gpu_resolution_drift_only_with_safe_refresh() -> None:
    """Break caught: SSM AMI re-resolution is rejected before its exact refresh is checked."""

    change_set = _change_set(
        changes=_expected_changes() + [_safe_launch_template_refresh()]
    )
    change_set["Parameters"][0]["ResolvedValue"] = "ami-new"

    validated = validate_foundation_change_set(
        change_set,
        stack_id=STACK_ID,
        change_set_name=CHANGE_SET_NAME,
        expected_parameters=[
            {
                "ParameterKey": "GpuAmiId",
                "ParameterValue": "ami-exact",
                "ResolvedValue": "ami-old",
            },
            {"ParameterKey": "ProjectTag", "ParameterValue": "keep-glm52"},
        ],
    )

    assert len(validated["Changes"]) == 5


@pytest.mark.parametrize("drifted_parameter", ("GpuAmiId", "ProjectTag"))
def test_change_set_rejects_resolved_drift_without_matching_safe_refresh(
    drifted_parameter: str,
) -> None:
    """Break caught: one parameter resolution drifts without its exact resource mutation."""

    change_set = _change_set()
    expected_parameters = deepcopy(change_set["Parameters"])
    row = next(
        parameter
        for parameter in change_set["Parameters"]
        if parameter["ParameterKey"] == drifted_parameter
    )
    expected_row = next(
        parameter
        for parameter in expected_parameters
        if parameter["ParameterKey"] == drifted_parameter
    )
    row["ResolvedValue"] = "resolved-new"
    expected_row["ResolvedValue"] = "resolved-old"

    with pytest.raises(FoundationError, match="parameter"):
        validate_foundation_change_set(
            change_set,
            stack_id=STACK_ID,
            change_set_name=CHANGE_SET_NAME,
            expected_parameters=expected_parameters,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("Action", "Remove"),
        ("LogicalResourceId", "ModelBucket"),
        ("ResourceType", "AWS::S3::Bucket"),
        ("Replacement", "True"),
    ),
)
def test_change_set_rejects_hostile_launch_template_refresh(
    field: str,
    value: str,
) -> None:
    """Break caught: the SSM refresh exception admits a broader stack mutation."""

    refresh = _safe_launch_template_refresh()
    refresh["ResourceChange"][field] = value
    with pytest.raises(FoundationError):
        validate_foundation_change_set(
            _change_set(changes=_expected_changes() + [refresh]),
            stack_id=STACK_ID,
            change_set_name=CHANGE_SET_NAME,
            expected_parameters=[
                {"ParameterKey": "GpuAmiId", "ParameterValue": "ami-exact"},
                {"ParameterKey": "ProjectTag", "ParameterValue": "keep-glm52"},
            ],
        )


def test_change_set_rejects_launch_template_refresh_detail_drift() -> None:
    """Break caught: a broad launch-template mutation hides behind the AMI refresh."""

    refresh = _safe_launch_template_refresh()
    refresh["ResourceChange"]["Details"][0]["Target"]["Name"] = "Tags"
    with pytest.raises(FoundationError):
        validate_foundation_change_set(
            _change_set(changes=_expected_changes() + [refresh]),
            stack_id=STACK_ID,
            change_set_name=CHANGE_SET_NAME,
            expected_parameters=[
                {"ParameterKey": "GpuAmiId", "ParameterValue": "ami-exact"},
                {"ParameterKey": "ProjectTag", "ParameterValue": "keep-glm52"},
            ],
        )


def test_change_set_accepts_only_four_additions_and_no_replace_or_delete() -> None:
    """Break caught: a retained resource is modified, replaced, or deleted."""

    validated = validate_foundation_change_set(
        _change_set(),
        stack_id=STACK_ID,
        change_set_name=CHANGE_SET_NAME,
        expected_parameters=[
            {"ParameterKey": "GpuAmiId", "ParameterValue": "ami-exact"},
            {"ParameterKey": "ProjectTag", "ParameterValue": "keep-glm52"},
        ],
    )
    assert validated["ChangeSetId"] == CHANGE_SET_ID

    for action, replacement in (
        ("Modify", "False"),
        ("Modify", "True"),
        ("Remove", "False"),
    ):
        hostile = _expected_changes()
        hostile[0] = {
            "Type": "Resource",
            "ResourceChange": {
                "Action": action,
                "LogicalResourceId": "ModelBucket",
                "ResourceType": "AWS::S3::Bucket",
                "Replacement": replacement,
            },
        }
        with pytest.raises(FoundationError):
            validate_foundation_change_set(
                _change_set(changes=hostile),
                stack_id=STACK_ID,
                change_set_name=CHANGE_SET_NAME,
                expected_parameters=[
                    {"ParameterKey": "GpuAmiId", "ParameterValue": "ami-exact"},
                    {"ParameterKey": "ProjectTag", "ParameterValue": "keep-glm52"},
                ],
            )


def test_foundation_change_set_rejects_parameter_readback_drift() -> None:
    """Break caught: foundation change set does not preserve live parameters."""

    change_set = _change_set()
    change_set["Parameters"] = []

    with pytest.raises(FoundationError, match="parameter"):
        validate_foundation_change_set(
            change_set,
            stack_id=STACK_ID,
            change_set_name=CHANGE_SET_NAME,
            expected_parameters=[
                {"ParameterKey": "GpuAmiId", "ParameterValue": "ami-exact"},
                {"ParameterKey": "ProjectTag", "ParameterValue": "keep-glm52"},
            ],
        )


def test_foundation_change_set_rejects_request_shape_as_parameter_readback() -> None:
    """Break caught: tests model request fields that AWS never reads back."""

    change_set = _change_set()
    change_set["Parameters"] = [
        {"ParameterKey": "GpuAmiId", "UsePreviousValue": True},
        {"ParameterKey": "ProjectTag", "UsePreviousValue": True},
    ]

    with pytest.raises(FoundationError, match="parameter"):
        validate_foundation_change_set(
            change_set,
            stack_id=STACK_ID,
            change_set_name=CHANGE_SET_NAME,
            expected_parameters=[
                {"ParameterKey": "GpuAmiId", "ParameterValue": "ami-exact"},
                {"ParameterKey": "ProjectTag", "ParameterValue": "keep-glm52"},
            ],
        )


class _AwsError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


def _response_metadata(request_id: str) -> dict[str, object]:
    return {
        "HTTPStatusCode": 200,
        "RetryAttempts": 0,
        "RequestId": request_id,
    }


class _FakeSts:
    def get_caller_identity(self) -> dict[str, object]:
        return {
            "Account": ACCOUNT_ID,
            "Arn": f"arn:aws:iam::{ACCOUNT_ID}:user/test",
            "UserId": "AIDATEST",
            "ResponseMetadata": _response_metadata("sts-request"),
        }


def _resolved_iam_policy(value: object) -> object:
    if type(value) is dict:
        if set(value) == {"Fn::GetAtt"}:
            target = value["Fn::GetAtt"]
            arns = {
                ("CampaignKmsKey", "Arn"): (
                    "arn:aws:kms:us-west-2:246813579024:key/"
                    "11111111-2222-4333-8444-555555555555"
                ),
                ("H1gLedger", "Arn"): (
                    "arn:aws:dynamodb:us-west-2:246813579024:"
                    "table/keep-glm52-h1g-ledger-v1"
                ),
                ("ModelBucket", "Arn"): (
                    "arn:aws:s3:::keep-glm52-models-246813579024-us-west-2"
                ),
            }
            assert type(target) is list and tuple(target) in arns
            return arns[tuple(target)]
        if set(value) == {"Fn::Sub"}:
            template = value["Fn::Sub"]
            assert type(template) is str
            resolved = (
                template.replace("${AWS::Partition}", "aws")
                .replace("${AWS::Region}", REGION)
                .replace("${AWS::AccountId}", ACCOUNT_ID)
            )
            assert "${" not in resolved
            return resolved
        return {key: _resolved_iam_policy(item) for key, item in value.items()}
    if type(value) is list:
        return [_resolved_iam_policy(item) for item in value]
    return deepcopy(value)


class _FakeIam:
    def __init__(self, owner: _FakeCloudFormation) -> None:
        self.owner = owner
        self.conflicting_role: str | None = None
        self.policy_readbacks: list[tuple[str, str]] = []

    def get_role(self, *, RoleName: str) -> dict[str, object]:
        if RoleName == self.conflicting_role:
            return {
                "Role": {
                    "RoleName": RoleName,
                    "RoleId": "AROACONFLICTINGROLE01",
                    "Arn": f"arn:aws:iam::{ACCOUNT_ID}:role/{RoleName}",
                }
            }
        if RoleName == "existing-retained-service-role":
            return {
                "Role": {
                    "RoleName": RoleName,
                    "RoleId": "AROAEXISTINGROLE0001",
                    "Arn": f"arn:aws:iam::{ACCOUNT_ID}:role/{RoleName}",
                }
            }
        if not self.owner.committed:
            raise _AwsError("NoSuchEntity")
        if RoleName not in {DEPLOYMENT_ROLE_NAME, FENCE_ROLE_NAME}:
            raise AssertionError(RoleName)
        return {
            "Role": {
                "RoleName": RoleName,
                "RoleId": (
                    "AROADEPLOYMENTROLE01"
                    if RoleName == DEPLOYMENT_ROLE_NAME
                    else "AROAFENCESERVICE0001"
                ),
                "Arn": f"arn:aws:iam::{ACCOUNT_ID}:role/{RoleName}",
                "AssumeRolePolicyDocument": (
                    self.owner.after["Resources"][
                        (
                            "H1gCloudFormationDeploymentRole"
                            if RoleName == DEPLOYMENT_ROLE_NAME
                            else "H1gFenceServiceRole"
                        )
                    ]["Properties"]["AssumeRolePolicyDocument"]
                ),
            }
        }

    def list_attached_role_policies(self, *, RoleName: str) -> dict[str, object]:
        self.policy_readbacks.append(("list-attached", RoleName))
        return {"AttachedPolicies": [], "IsTruncated": False}

    def list_role_policies(self, *, RoleName: str) -> dict[str, object]:
        self.policy_readbacks.append(("list-inline", RoleName))
        logical_id = (
            "H1gCloudFormationDeploymentRole"
            if RoleName == DEPLOYMENT_ROLE_NAME
            else "H1gFenceServiceRole"
        )
        policies = self.owner.after["Resources"][logical_id]["Properties"]["Policies"]
        return {
            "PolicyNames": [policy["PolicyName"] for policy in policies],
            "IsTruncated": False,
        }

    def get_role_policy(self, *, RoleName: str, PolicyName: str) -> dict[str, object]:
        self.policy_readbacks.append(("get-inline", RoleName))
        logical_id = (
            "H1gCloudFormationDeploymentRole"
            if RoleName == DEPLOYMENT_ROLE_NAME
            else "H1gFenceServiceRole"
        )
        policies = self.owner.after["Resources"][logical_id]["Properties"]["Policies"]
        policy = next(
            policy for policy in policies if policy["PolicyName"] == PolicyName
        )
        return {
            "RoleName": RoleName,
            "PolicyName": PolicyName,
            "PolicyDocument": _resolved_iam_policy(policy["PolicyDocument"]),
        }


class _FakeDynamoDb:
    def __init__(self, owner: _FakeCloudFormation) -> None:
        self.owner = owner
        self.conflict = False

    def describe_table(self, *, TableName: str) -> dict[str, object]:
        assert TableName == "keep-glm52-h1g-ledger-v1"
        if not self.owner.committed and not self.conflict:
            raise _AwsError("ResourceNotFoundException")
        return {
            "Table": {
                "TableName": TableName,
                "TableArn": (
                    "arn:aws:dynamodb:us-west-2:246813579024:table/"
                    "keep-glm52-h1g-ledger-v1"
                ),
                "TableStatus": "ACTIVE",
                "DeletionProtectionEnabled": True,
                "SSEDescription": {
                    "Status": "ENABLED",
                    "SSEType": "KMS",
                    "KMSMasterKeyArn": (
                        "arn:aws:kms:us-west-2:246813579024:key/"
                        "11111111-2222-4333-8444-555555555555"
                    ),
                },
            }
        }


class _FakeKms:
    def describe_key(self, *, KeyId: str) -> dict[str, object]:
        assert KeyId == (
            "arn:aws:kms:us-west-2:246813579024:key/"
            "11111111-2222-4333-8444-555555555555"
        )
        return {
            "KeyMetadata": {
                "AWSAccountId": ACCOUNT_ID,
                "Arn": KeyId,
                "KeyId": "11111111-2222-4333-8444-555555555555",
                "Enabled": True,
                "KeyManager": "CUSTOMER",
                "KeyState": "Enabled",
            }
        }


class _FakeS3:
    def __init__(self) -> None:
        self.raw: bytes | None = None
        self.metadata: dict[str, str] | None = None
        self.put_requests: list[dict[str, object]] = []
        self.version_id = "retained-template-version-0001"

    def get_bucket_versioning(self, **request: object) -> dict[str, object]:
        assert request == {
            "Bucket": "keep-glm52-models-246813579024-us-west-2",
            "ExpectedBucketOwner": ACCOUNT_ID,
        }
        return {
            "Status": "Enabled",
            "ResponseMetadata": _response_metadata("versioning-request"),
        }

    def list_object_versions(self, **request: object) -> dict[str, object]:
        assert request == {
            "Bucket": "keep-glm52-models-246813579024-us-west-2",
            "Prefix": "task13/templates/retained-foundation.yaml",
            "MaxKeys": 1000,
            "ExpectedBucketOwner": ACCOUNT_ID,
        }
        versions = (
            []
            if self.raw is None
            else [
                {
                    "Key": "task13/templates/retained-foundation.yaml",
                    "VersionId": self.version_id,
                    "Size": len(self.raw),
                }
            ]
        )
        return {
            "Versions": versions,
            "DeleteMarkers": [],
            "IsTruncated": False,
            "ResponseMetadata": _response_metadata("versions-request"),
        }

    def put_object(self, **request: object) -> dict[str, object]:
        self.put_requests.append(dict(request))
        assert request["Bucket"] == ("keep-glm52-models-246813579024-us-west-2")
        assert request["Key"] == "task13/templates/retained-foundation.yaml"
        assert request["IfNoneMatch"] == "*"
        assert request["ExpectedBucketOwner"] == ACCOUNT_ID
        assert request["ChecksumAlgorithm"] == "SHA256"
        raw = request["Body"]
        assert isinstance(raw, bytes)
        self.raw = raw
        metadata = request["Metadata"]
        assert isinstance(metadata, dict)
        self.metadata = dict(metadata)
        checksum = base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
        assert request["ChecksumSHA256"] == checksum
        return {
            "VersionId": self.version_id,
            "ChecksumSHA256": checksum,
            "ResponseMetadata": _response_metadata("put-request"),
        }

    def get_object(self, **request: object) -> dict[str, object]:
        assert request == {
            "Bucket": "keep-glm52-models-246813579024-us-west-2",
            "Key": "task13/templates/retained-foundation.yaml",
            "VersionId": self.version_id,
            "ExpectedBucketOwner": ACCOUNT_ID,
            "ChecksumMode": "ENABLED",
        }
        assert self.raw is not None
        checksum = base64.b64encode(hashlib.sha256(self.raw).digest()).decode("ascii")
        return {
            "Body": io.BytesIO(self.raw),
            "ContentLength": len(self.raw),
            "VersionId": self.version_id,
            "ChecksumSHA256": checksum,
            "Metadata": self.metadata,
            "ResponseMetadata": _response_metadata("get-request"),
        }


def test_botocore_transport_failure_is_ambiguous() -> None:
    """Break caught: botocore transport loss bypasses readback reconciliation."""
    error = EndpointConnectionError(endpoint_url="https://example.invalid")
    assert foundation_api._ambiguous_exception(error) is True


class _FakeCloudFormation:
    def __init__(
        self,
        *,
        ambiguous_create: bool = False,
        ambiguous_execute: bool = False,
        existing_role: bool = False,
        original_as_yaml: bool = False,
        termination_protection: bool = False,
        ambiguous_protection: bool = False,
        gpu_resolved_before: str | None = None,
        gpu_resolved_after: str | None = None,
    ) -> None:
        self.original = _original_template()
        self.after = compose_retained_foundation_template(self.original)
        self.ambiguous_create = ambiguous_create
        self.ambiguous_execute = ambiguous_execute
        self.ambiguous_protection = ambiguous_protection
        self.existing_role = existing_role
        self.original_source = (
            "# exact live Original bytes; do not reserialize\n"
            + yaml.safe_dump(self.original, sort_keys=False)
        )
        self.after_source = foundation_api._execution_template_bytes(
            raw_template_body=self.original_source,
            original_template=self.original,
            retained_template=self.after,
        ).decode("utf-8")
        self.original_as_yaml = original_as_yaml
        self.termination_protection = termination_protection
        self.gpu_resolved_before = gpu_resolved_before
        self.gpu_resolved_after = gpu_resolved_after
        self.created = False
        self.executed = False
        self.committed = False
        self.conflicting_export = False
        self.post_execute_describes = 0
        self.create_requests: list[dict[str, object]] = []
        self.execute_requests: list[dict[str, object]] = []
        self.update_termination_requests: list[dict[str, object]] = []

    def _outputs(self, complete: bool) -> list[dict[str, object]]:
        base = [
            {"OutputKey": "VpcId", "OutputValue": "vpc-0123456789abcdef0"},
            {
                "OutputKey": "SubnetId",
                "OutputValue": "subnet-0123456789abcdef0",
            },
            {
                "OutputKey": "ModelBucketName",
                "OutputValue": ("keep-glm52-models-246813579024-us-west-2"),
                "ExportName": "keep-glm52-gpu-ModelBucket",
            },
            {
                "OutputKey": "LaunchTemplateId",
                "OutputValue": "lt-0123456789abcdef0",
                "ExportName": "keep-glm52-gpu-LaunchTemplate",
            },
        ]
        if complete:
            base[0]["ExportName"] = "KeepGlm52VpcId"
            base[1]["ExportName"] = "KeepGlm52PrimaryPublicSubnetId"
            base.extend(
                [
                    {
                        "OutputKey": "CampaignKmsKeyArn",
                        "OutputValue": (
                            "arn:aws:kms:us-west-2:246813579024:key/"
                            "11111111-2222-4333-8444-555555555555"
                        ),
                        "ExportName": "KeepGlm52CampaignKmsKeyArn",
                    },
                    {
                        "OutputKey": "H1gLedgerArn",
                        "OutputValue": (
                            "arn:aws:dynamodb:us-west-2:246813579024:"
                            "table/keep-glm52-h1g-ledger-v1"
                        ),
                        "ExportName": "KeepGlm52H1gLedgerArn",
                    },
                ]
            )
        return base

    def describe_stacks(self, *, StackName: str) -> dict[str, object]:
        assert StackName in {STACK_NAME, STACK_ID}
        if self.executed:
            self.post_execute_describes += 1
            if self.post_execute_describes >= 2:
                self.committed = True
        stack: dict[str, object] = {
            "StackId": STACK_ID,
            "StackName": STACK_NAME,
            "StackStatus": (
                "UPDATE_COMPLETE"
                if self.committed or not self.executed
                else "UPDATE_IN_PROGRESS"
            ),
            "Parameters": [
                {
                    "ParameterKey": "ProjectTag",
                    "ParameterValue": "keep-glm52",
                },
                {
                    "ParameterKey": "GpuAmiId",
                    "ParameterValue": "ami-exact",
                    **(
                        {"ResolvedValue": resolved_value}
                        if (
                            resolved_value := (
                                self.gpu_resolved_after
                                if self.committed
                                else self.gpu_resolved_before
                            )
                        )
                        is not None
                        else {}
                    ),
                },
            ],
            "Outputs": self._outputs(self.committed),
            "Tags": [{"Key": "project", "Value": "keep-glm52"}],
            "EnableTerminationProtection": self.termination_protection,
        }
        if self.existing_role:
            stack["RoleARN"] = (
                f"arn:aws:iam::{ACCOUNT_ID}:role/existing-retained-service-role"
            )
        return {"Stacks": [stack]}

    def get_template(
        self,
        *,
        TemplateStage: str,
        StackName: str | None = None,
        ChangeSetName: str | None = None,
    ) -> dict[str, object]:
        assert TemplateStage == "Original"
        if ChangeSetName is not None:
            assert ChangeSetName == CHANGE_SET_ID
            return {
                "TemplateBody": (
                    self.after_source if self.original_as_yaml else deepcopy(self.after)
                )
            }
        if self.original_as_yaml:
            return {
                "TemplateBody": (
                    self.after_source if self.committed else self.original_source
                )
            }
        return {
            "TemplateBody": deepcopy(self.after if self.committed else self.original)
        }

    def describe_change_set(
        self,
        *,
        ChangeSetName: str,
        StackName: str | None = None,
        IncludePropertyValues: bool = False,
    ) -> dict[str, object]:
        assert IncludePropertyValues is True
        if not self.created:
            raise _AwsError("ChangeSetNotFound")
        assert ChangeSetName in {CHANGE_SET_NAME, CHANGE_SET_ID}
        if StackName is not None:
            assert StackName == STACK_ID
        response = _change_set()
        if self.gpu_resolved_after is not None:
            response["Parameters"][0]["ResolvedValue"] = self.gpu_resolved_after
        if self.gpu_resolved_after != self.gpu_resolved_before:
            response["Changes"] = _expected_changes() + [
                _safe_launch_template_refresh()
            ]
        if self.executed:
            response["ExecutionStatus"] = "EXECUTE_COMPLETE"
        return response

    def create_change_set(self, **request: object) -> dict[str, object]:
        self.create_requests.append(dict(request))
        self.created = True
        if self.ambiguous_create:
            raise TimeoutError("response lost after create")
        return {"Id": CHANGE_SET_ID, "StackId": STACK_ID}

    def execute_change_set(self, **request: object) -> dict[str, object]:
        self.execute_requests.append(dict(request))
        self.executed = True
        if self.ambiguous_execute:
            raise TimeoutError("response lost after execute")
        return {}

    def update_termination_protection(
        self,
        *,
        StackName: str,
        EnableTerminationProtection: bool,
    ) -> dict[str, object]:
        self.update_termination_requests.append(
            {
                "StackName": StackName,
                "EnableTerminationProtection": EnableTerminationProtection,
            }
        )
        assert StackName == STACK_ID
        assert EnableTerminationProtection is True
        self.termination_protection = True
        if self.ambiguous_protection:
            raise TimeoutError("response lost after termination-protection update")
        return {}

    def list_stack_resources(
        self, *, StackName: str, NextToken: str | None = None
    ) -> dict[str, object]:
        assert StackName == STACK_ID
        assert NextToken is None
        active_original = foundation_api._active_resource_types(
            self.original,
            _change_set()["Parameters"],
        )
        originals = [
            {
                "LogicalResourceId": logical_id,
                "PhysicalResourceId": f"physical-{logical_id}",
                "ResourceType": resource_type,
                "ResourceStatus": "CREATE_COMPLETE",
            }
            for logical_id, resource_type in active_original.items()
        ]
        additions = [
            {
                "LogicalResourceId": "CampaignKmsKey",
                "PhysicalResourceId": ("11111111-2222-4333-8444-555555555555"),
                "ResourceType": "AWS::KMS::Key",
                "ResourceStatus": "CREATE_COMPLETE",
            },
            {
                "LogicalResourceId": "H1gLedger",
                "PhysicalResourceId": "keep-glm52-h1g-ledger-v1",
                "ResourceType": "AWS::DynamoDB::Table",
                "ResourceStatus": "CREATE_COMPLETE",
            },
            {
                "LogicalResourceId": "H1gCloudFormationDeploymentRole",
                "PhysicalResourceId": DEPLOYMENT_ROLE_NAME,
                "ResourceType": "AWS::IAM::Role",
                "ResourceStatus": "CREATE_COMPLETE",
            },
            {
                "LogicalResourceId": "H1gFenceServiceRole",
                "PhysicalResourceId": FENCE_ROLE_NAME,
                "ResourceType": "AWS::IAM::Role",
                "ResourceStatus": "CREATE_COMPLETE",
            },
        ]
        return {"StackResourceSummaries": originals + additions}

    def list_exports(self, *, NextToken: str | None = None) -> dict[str, object]:
        assert NextToken is None
        exports = []
        if self.conflicting_export:
            exports.append(
                {
                    "Name": "KeepGlm52CampaignKmsKeyArn",
                    "Value": "arn:aws:kms:us-west-2:246813579024:key/foreign",
                    "ExportingStackId": (
                        "arn:aws:cloudformation:us-west-2:246813579024:"
                        "stack/foreign/11111111-2222-4333-8444-555555555555"
                    ),
                }
            )
        return {"Exports": exports}


class _FakeCloudTrail:
    def __init__(
        self,
        *,
        exact: bool = True,
        include_failed_exact_attempt: bool = False,
    ) -> None:
        self.exact = exact
        self.include_failed_exact_attempt = include_failed_exact_attempt

    def lookup_events(self, **request: object) -> dict[str, object]:
        assert request == {
            "LookupAttributes": [
                {
                    "AttributeKey": "EventName",
                    "AttributeValue": "ExecuteChangeSet",
                }
            ],
            "MaxResults": 50,
        }
        event_id = "c1cec0e1-dfff-4648-a1a1-6aaf310e835a"
        event = {
            "eventID": event_id,
            "eventTime": "2026-07-31T23:02:23Z",
            "eventName": "ExecuteChangeSet",
            "eventSource": "cloudformation.amazonaws.com",
            "userIdentity": {"accountId": ACCOUNT_ID},
            "requestParameters": {
                "changeSetName": (
                    CHANGE_SET_ID
                    if self.exact
                    else CHANGE_SET_ID.replace("11111111", "99999999")
                ),
                "stackName": STACK_ID,
            },
            "responseElements": None,
            "readOnly": False,
            "managementEvent": True,
            "recipientAccountId": ACCOUNT_ID,
        }
        events = [
            {
                "EventId": event_id,
                "EventName": "ExecuteChangeSet",
                "CloudTrailEvent": json.dumps(event),
            }
        ]
        if self.include_failed_exact_attempt:
            failed_event = deepcopy(event)
            failed_event["eventID"] = "d2dfd1f2-e111-4759-b2b2-7bbf421f946b"
            failed_event["errorCode"] = "InvalidChangeSetStatus"
            failed_event["errorMessage"] = "ChangeSet cannot be executed"
            events.insert(
                0,
                {
                    "EventId": failed_event["eventID"],
                    "EventName": "ExecuteChangeSet",
                    "CloudTrailEvent": json.dumps(failed_event),
                },
            )
        return {"Events": events}


def _services(
    *,
    ambiguous_create: bool = False,
    ambiguous_execute: bool = False,
    existing_role: bool = False,
    original_as_yaml: bool = False,
    termination_protection: bool = False,
    ambiguous_protection: bool = False,
    cloudtrail_exact: bool = True,
    cloudtrail_failed_exact_attempt: bool = False,
    gpu_resolved_before: str | None = None,
    gpu_resolved_after: str | None = None,
) -> tuple[FoundationServices, _FakeCloudFormation]:
    cloudformation = _FakeCloudFormation(
        ambiguous_create=ambiguous_create,
        ambiguous_execute=ambiguous_execute,
        ambiguous_protection=ambiguous_protection,
        existing_role=existing_role,
        original_as_yaml=original_as_yaml,
        termination_protection=termination_protection,
        gpu_resolved_before=gpu_resolved_before,
        gpu_resolved_after=gpu_resolved_after,
    )
    return (
        FoundationServices(
            sts=_FakeSts(),
            cloudformation=cloudformation,
            cloudtrail=_FakeCloudTrail(
                exact=cloudtrail_exact,
                include_failed_exact_attempt=cloudtrail_failed_exact_attempt,
            ),
            iam=_FakeIam(cloudformation),
            dynamodb=_FakeDynamoDb(cloudformation),
            kms=_FakeKms(),
            s3=_FakeS3(),
            total_max_attempts=1,
        ),
        cloudformation,
    )


def test_one_shot_route_seals_templates_change_set_and_exact_live_readback(
    tmp_path: Path,
) -> None:
    """Break caught: the production route executes before exact sealed review."""

    services, cloudformation = _services()
    result = apply_retained_foundation(
        services,
        output_directory=tmp_path.resolve(),
        sleep=lambda _seconds: None,
        max_polls=4,
    )

    assert result["status"] == "UPDATE_COMPLETE"
    assert len(cloudformation.create_requests) == 1
    request = cloudformation.create_requests[0]
    assert request["StackName"] == STACK_ID
    assert request["ChangeSetName"] == PRODUCTION_CHANGE_SET_NAME
    assert request["ChangeSetType"] == "UPDATE"
    assert request["Capabilities"] == ["CAPABILITY_NAMED_IAM"]
    assert request["IncludeNestedStacks"] is False
    assert "RoleARN" not in request
    assert "TemplateBody" not in request
    assert request["TemplateURL"] == (
        "https://keep-glm52-models-246813579024-us-west-2."
        "s3.us-west-2.amazonaws.com/task13/templates/retained-foundation.yaml"
        "?versionId=retained-template-version-0001"
    )
    assert request["Parameters"] == [
        {"ParameterKey": "GpuAmiId", "UsePreviousValue": True},
        {"ParameterKey": "ProjectTag", "UsePreviousValue": True},
    ]
    assert len(cloudformation.execute_requests) == 1
    assert cloudformation.execute_requests[0]["ChangeSetName"] == CHANGE_SET_ID
    assert cloudformation.execute_requests[0]["StackName"] == STACK_ID
    assert len(services.iam.policy_readbacks) == 6

    assert {path.name for path in tmp_path.iterdir()} == {
        "retained-before-template.json",
        "retained-after-template.json",
        "retained-change-set-evidence.json",
        "retained-readback-evidence.json",
    }
    for path in tmp_path.iterdir():
        assert path.stat().st_mode & 0o777 == 0o600
        raw = path.read_bytes()
        assert raw == (
            json.dumps(
                json.loads(raw),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
            + b"\n"
        )
    assert (
        json.loads((tmp_path / "retained-before-template.json").read_bytes())
        == cloudformation.original
    )
    assert (
        json.loads((tmp_path / "retained-after-template.json").read_bytes())
        == cloudformation.after
    )


def test_recovery_seals_live_protection_and_emits_explicit_evidence(
    tmp_path: Path,
) -> None:
    """Break caught: missing post-update output causes a second template update."""

    transaction = tmp_path / "transaction"
    transaction.mkdir()
    services, cloudformation = _services()
    apply_retained_foundation(
        services,
        output_directory=transaction.resolve(),
        sleep=lambda _seconds: None,
        max_polls=4,
    )
    change_path = transaction / "retained-change-set-evidence.json"
    recovery_path = tmp_path / "recovery-readback.json"

    result = recover_retained_foundation(
        services,
        change_set_evidence=change_path.resolve(),
        readback_output=recovery_path.resolve(),
    )

    change_raw = change_path.read_bytes()
    readback_raw = recovery_path.read_bytes()
    readback = json.loads(readback_raw)
    assert readback_raw == (
        json.dumps(
            readback,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        + b"\n"
    )
    assert recovery_path.stat().st_mode & 0o777 == 0o600
    assert readback["record_type"] == ("glm52_retained_foundation_recovery_readback_v1")
    spec = importlib.util.spec_from_file_location(
        "task13_staged_invocation_preparer_recovery",
        INVOCATION_PREPARER,
    )
    assert spec is not None and spec.loader is not None
    preparer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preparer)
    compact, coordinate = preparer._compact_foundation_evidence(
        change_value=json.loads(change_path.read_bytes()),
        readback_value=readback,
        retained_stack_id=STACK_ID,
    )
    assert compact["stack_status"] == "UPDATE_COMPLETE"
    assert coordinate == readback["template_coordinate"]
    assert set(compact) == {
        "stack_id",
        "stack_status",
        "change_set_id",
        "template_coordinate",
        "template_body_sha256",
        "readback_sha256",
    }
    assert readback["execute_cloudtrail_event_id"] == (
        "c1cec0e1-dfff-4648-a1a1-6aaf310e835a"
    )
    assert readback["execute_cloudtrail_event_time"] == "2026-07-31T23:02:23Z"
    assert (
        readback["source_change_set_evidence_sha256"]
        == hashlib.sha256(change_raw[:-1]).hexdigest()
    )
    assert readback["historical_termination_protection"] is False
    assert readback["termination_protection"] is True
    assert readback["termination_protection_sealed_during_recovery"] is True
    assert readback["execute_response_was_ambiguous"] is None
    assert cloudformation.update_termination_requests == [
        {
            "StackName": STACK_ID,
            "EnableTerminationProtection": True,
        }
    ]
    assert len(cloudformation.create_requests) == 1
    assert len(cloudformation.execute_requests) == 1
    assert result["status"] == "UPDATE_COMPLETE"
    assert result["stack_id"] == STACK_ID
    assert result["readback_sha256"] == hashlib.sha256(readback_raw[:-1]).hexdigest()


@pytest.mark.parametrize(
    ("initially_protected", "ambiguous_update", "expected_updates"),
    ((True, False, 0), (False, True, 1)),
)
def test_recovery_adopts_exact_protection_state_without_replay(
    tmp_path: Path,
    initially_protected: bool,
    ambiguous_update: bool,
    expected_updates: int,
) -> None:
    """Break caught: recovery repeats a sealed or response-lost protection write."""
    transaction = tmp_path / "transaction"
    transaction.mkdir()
    services, cloudformation = _services(
        termination_protection=initially_protected,
        ambiguous_protection=ambiguous_update,
    )
    apply_retained_foundation(
        services,
        output_directory=transaction.resolve(),
        sleep=lambda _seconds: None,
        max_polls=4,
    )

    recover_retained_foundation(
        services,
        change_set_evidence=(
            transaction / "retained-change-set-evidence.json"
        ).resolve(),
        readback_output=(tmp_path / "recovered.json").resolve(),
    )

    assert len(cloudformation.update_termination_requests) == expected_updates
    assert len(cloudformation.create_requests) == 1
    assert len(cloudformation.execute_requests) == 1


def test_recovery_ignores_failed_reexecute_events(
    tmp_path: Path,
) -> None:
    """Break caught: a later failed re-execute attempt bricks valid adoption."""
    transaction = tmp_path / "transaction"
    transaction.mkdir()
    services, _cloudformation = _services(
        cloudtrail_failed_exact_attempt=True,
    )
    apply_retained_foundation(
        services,
        output_directory=transaction.resolve(),
        sleep=lambda _seconds: None,
        max_polls=4,
    )
    recovery_path = tmp_path / "recovered.json"

    recovered = recover_retained_foundation(
        services,
        change_set_evidence=(
            transaction / "retained-change-set-evidence.json"
        ).resolve(),
        readback_output=recovery_path.resolve(),
    )

    assert recovered["status"] == "UPDATE_COMPLETE"
    readback = json.loads(recovery_path.read_bytes())
    assert readback["execute_cloudtrail_event_id"] == (
        "c1cec0e1-dfff-4648-a1a1-6aaf310e835a"
    )


def test_recovery_requires_the_exact_cloudtrail_execute_event(
    tmp_path: Path,
) -> None:
    """Break caught: live byte equality is misattributed to another change set."""
    transaction = tmp_path / "transaction"
    transaction.mkdir()
    services, cloudformation = _services(cloudtrail_exact=False)
    apply_retained_foundation(
        services,
        output_directory=transaction.resolve(),
        sleep=lambda _seconds: None,
        max_polls=4,
    )
    recovery_path = tmp_path / "recovered.json"

    with pytest.raises(FoundationError, match="no unique exact"):
        recover_retained_foundation(
            services,
            change_set_evidence=(
                transaction / "retained-change-set-evidence.json"
            ).resolve(),
            readback_output=recovery_path.resolve(),
        )

    assert cloudformation.update_termination_requests == []
    assert not recovery_path.exists()


def test_recovery_rejects_drifted_versioned_template_without_output(
    tmp_path: Path,
) -> None:
    """Break caught: recovery trusts a mutable or mismatched template payload."""
    transaction = tmp_path / "transaction"
    transaction.mkdir()
    services, _cloudformation = _services()
    apply_retained_foundation(
        services,
        output_directory=transaction.resolve(),
        sleep=lambda _seconds: None,
        max_polls=4,
    )
    assert isinstance(services.s3, _FakeS3)
    assert services.s3.raw is not None
    services.s3.raw += b"tampered"
    recovery_path = tmp_path / "recovered.json"

    with pytest.raises(FoundationError, match="versioned readback drifted"):
        recover_retained_foundation(
            services,
            change_set_evidence=(
                transaction / "retained-change-set-evidence.json"
            ).resolve(),
            readback_output=recovery_path.resolve(),
        )

    assert not recovery_path.exists()


def test_recovery_rejects_tampered_change_evidence_before_live_mutation(
    tmp_path: Path,
) -> None:
    """Break caught: arbitrary local JSON can authorize foundation adoption."""

    transaction = tmp_path / "transaction"
    transaction.mkdir()
    services, cloudformation = _services()
    apply_retained_foundation(
        services,
        output_directory=transaction.resolve(),
        sleep=lambda _seconds: None,
        max_polls=4,
    )
    change_path = transaction / "retained-change-set-evidence.json"
    change = json.loads(change_path.read_bytes())
    change["after_template_sha256"] = "f" * 64
    change_path.write_bytes(
        json.dumps(
            change,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        + b"\n"
    )
    recovery_path = tmp_path / "recovery-readback.json"

    with pytest.raises(FoundationError, match="change evidence"):
        recover_retained_foundation(
            services,
            change_set_evidence=change_path.resolve(),
            readback_output=recovery_path.resolve(),
        )

    assert cloudformation.update_termination_requests == []
    assert not recovery_path.exists()


def test_recovery_binds_source_parameters_to_the_live_stack(
    tmp_path: Path,
) -> None:
    """Break caught: a self-consistent source request names foreign parameters."""
    transaction = tmp_path / "transaction"
    transaction.mkdir()
    services, cloudformation = _services()
    apply_retained_foundation(
        services,
        output_directory=transaction.resolve(),
        sleep=lambda _seconds: None,
        max_polls=4,
    )
    change_path = transaction / "retained-change-set-evidence.json"
    change = json.loads(change_path.read_bytes())
    change["parameters"] = change["parameters"][:-1]
    change["client_token"] = hashlib.sha256(
        foundation_api.canonical_json_bytes(
            {
                "stack_id": change["stack_id"],
                "change_set_name": change["change_set_name"],
                "template_coordinate": change["template_coordinate"],
                "parameters": change["parameters"],
                "role_arn": change["role_arn"],
            }
        )
    ).hexdigest()
    change_path.write_bytes(foundation_api.canonical_json_bytes(change) + b"\n")
    recovery_path = tmp_path / "recovery-readback.json"

    with pytest.raises(FoundationError, match="parameters drifted"):
        recover_retained_foundation(
            services,
            change_set_evidence=change_path.resolve(),
            readback_output=recovery_path.resolve(),
        )

    assert cloudformation.update_termination_requests == []
    assert not recovery_path.exists()


def test_live_readback_accepts_validated_ssm_ami_resolution_after_execute(
    tmp_path: Path,
) -> None:
    """Break caught: safe SSM AMI refresh executes, then final readback rejects it."""

    services, cloudformation = _services(
        gpu_resolved_before="ami-old",
        gpu_resolved_after="ami-new",
    )

    result = apply_retained_foundation(
        services,
        output_directory=tmp_path.resolve(),
        sleep=lambda _seconds: None,
        max_polls=4,
    )

    assert result["status"] == "UPDATE_COMPLETE"
    assert cloudformation.committed is True


def test_one_shot_route_uses_raw_live_yaml_as_execution_transport(
    tmp_path: Path,
) -> None:
    """Break caught: the guarded route republishes parsed YAML and widens the change set."""

    services, cloudformation = _services(
        original_as_yaml=True,
        termination_protection=True,
    )
    result = apply_retained_foundation(
        services,
        output_directory=tmp_path.resolve(),
        sleep=lambda _seconds: None,
        max_polls=4,
    )

    assert result["status"] == "UPDATE_COMPLETE"
    assert isinstance(services.s3, _FakeS3)
    assert services.s3.raw is not None
    assert services.s3.raw.startswith(
        b"# exact live Original bytes; do not reserialize\n"
    )
    assert foundation_api._template_body(services.s3.raw.decode("utf-8")) == (
        cloudformation.after
    )
    change_evidence = json.loads(
        (tmp_path / "retained-change-set-evidence.json").read_bytes()
    )
    readback_evidence = json.loads(
        (tmp_path / "retained-readback-evidence.json").read_bytes()
    )
    body_sha256 = change_evidence["template_coordinate"]["body_sha256"]
    assert change_evidence["after_template_sha256"] == body_sha256
    assert readback_evidence["template_sha256"] == body_sha256

    spec = importlib.util.spec_from_file_location(
        "task13_staged_invocation_preparer_e2e",
        INVOCATION_PREPARER,
    )
    assert spec is not None and spec.loader is not None
    preparer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preparer)
    compact, coordinate = preparer._compact_foundation_evidence(
        change_value=change_evidence,
        readback_value=readback_evidence,
        retained_stack_id=STACK_ID,
    )
    assert compact["template_body_sha256"] == body_sha256
    assert coordinate == change_evidence["template_coordinate"]


@pytest.mark.parametrize("ambiguous_stage", ("create", "execute"))
def test_ambiguous_mutation_is_adopted_only_by_exact_describe_reconciliation(
    tmp_path: Path,
    ambiguous_stage: str,
) -> None:
    """Break caught: a lost response replays the create or execute mutation."""

    services, cloudformation = _services(
        ambiguous_create=ambiguous_stage == "create",
        ambiguous_execute=ambiguous_stage == "execute",
    )
    result = apply_retained_foundation(
        services,
        output_directory=tmp_path.resolve(),
        sleep=lambda _seconds: None,
        max_polls=4,
    )
    assert result["status"] == "UPDATE_COMPLETE"
    assert len(cloudformation.create_requests) == 1
    assert len(cloudformation.execute_requests) == 1


def test_existing_retained_role_is_authenticated_and_reused_but_new_role_is_not(
    tmp_path: Path,
) -> None:
    """Break caught: bootstrap passes a same-change-set or unauthenticated RoleARN."""

    services, cloudformation = _services(existing_role=True)
    apply_retained_foundation(
        services,
        output_directory=tmp_path.resolve(),
        sleep=lambda _seconds: None,
        max_polls=4,
    )
    assert cloudformation.create_requests[0]["RoleARN"] == (
        f"arn:aws:iam::{ACCOUNT_ID}:role/existing-retained-service-role"
    )


def test_preexisting_physical_role_blocks_before_change_set_creation(
    tmp_path: Path,
) -> None:
    """Break caught: a foreign account-global role name is adopted by CloudFormation."""

    services, cloudformation = _services()
    services.iam.conflicting_role = DEPLOYMENT_ROLE_NAME
    with pytest.raises(FoundationError, match="already exists"):
        apply_retained_foundation(
            services,
            output_directory=tmp_path.resolve(),
            sleep=lambda _seconds: None,
            max_polls=4,
        )
    assert cloudformation.create_requests == []
    assert cloudformation.execute_requests == []


def test_preexisting_account_export_blocks_before_change_set_creation(
    tmp_path: Path,
) -> None:
    """Break caught: an export owned by another stack is discovered only at execute."""

    services, cloudformation = _services()
    cloudformation.conflicting_export = True
    with pytest.raises(FoundationError, match="export"):
        apply_retained_foundation(
            services,
            output_directory=tmp_path.resolve(),
            sleep=lambda _seconds: None,
            max_polls=4,
        )
    assert cloudformation.create_requests == []
    assert cloudformation.execute_requests == []


def _load_cli() -> object:
    spec = importlib.util.spec_from_file_location(
        "apply_glm52_task13_retained_foundation",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_builds_only_exact_profile_region_and_zero_retry_clients() -> None:
    """Break caught: SDK defaults can target another account path or replay calls."""

    module = _load_cli()
    configs: list[dict[str, object]] = []
    clients: list[tuple[str, object]] = []

    def config_factory(**kwargs: object) -> object:
        configs.append(dict(kwargs))
        return object()

    class Session:
        region_name = REGION

        def __init__(self, **kwargs: object) -> None:
            assert kwargs == {
                "profile_name": "keep-gpu",
                "region_name": REGION,
            }

        def client(self, service: str, *, config: object) -> object:
            clients.append((service, config))
            return object()

    services = module._build_services(
        session_factory=Session,
        config_factory=config_factory,
    )
    assert configs == [
        {
            "region_name": REGION,
            "connect_timeout": 5,
            "read_timeout": 30,
            "retries": {
                "mode": "standard",
                "total_max_attempts": 1,
            },
        }
    ]
    assert [service for service, _config in clients] == [
        "sts",
        "cloudformation",
        "iam",
        "cloudtrail",
        "dynamodb",
        "kms",
        "s3",
    ]
    assert all(config is clients[0][1] for _service, config in clients)
    assert services.total_max_attempts == 1


def test_cli_recovers_without_replaying_the_foundation_change_set(
    tmp_path: Path,
) -> None:
    """Break caught: the recovery CLI routes through the one-shot apply path."""
    module = _load_cli()
    transaction = tmp_path / "transaction"
    transaction.mkdir()
    services, cloudformation = _services()
    apply_retained_foundation(
        services,
        output_directory=transaction.resolve(),
        sleep=lambda _seconds: None,
        max_polls=4,
    )

    result = module.main(
        [
            "--output-directory",
            str(tmp_path.resolve()),
            "--recover-change-set-evidence",
            str((transaction / "retained-change-set-evidence.json").resolve()),
        ],
        services_factory=lambda: services,
        sleep=lambda _seconds: None,
    )

    assert result == 0
    assert len(cloudformation.create_requests) == 1
    assert len(cloudformation.execute_requests) == 1
    assert (tmp_path / "retained-foundation-recovery-readback.json").is_file()


def test_cli_runs_the_complete_one_shot_route_without_stage_replay(
    tmp_path: Path,
) -> None:
    """Break caught: CLI exposes partial create/execute stages that can be replayed."""

    module = _load_cli()
    services, cloudformation = _services()
    result = module.main(
        ["--output-directory", str(tmp_path.resolve())],
        services_factory=lambda: services,
        sleep=lambda _seconds: None,
    )
    assert result == 0
    assert len(cloudformation.create_requests) == 1
    assert len(cloudformation.execute_requests) == 1
