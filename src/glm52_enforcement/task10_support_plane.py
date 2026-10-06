"""Retained CloudFormation surface for the Task 10 sole sender."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
FUNCTION_NAME = "keep-glm52-h1g-task10-capacity-reconciliation"
LIABILITY_WATCHER_FUNCTION_NAME = (
    "keep-glm52-h1g-worker-launch-custody"
)
WORKFLOW_NAME = "keep-glm52-h1g-production"
CAMPAIGN_BUCKET_ARN = (
    "arn:aws:s3:::keep-glm52-models-246813579024-us-west-2"
)
_ABSENT_CAMPAIGN_BUCKET_TOKEN = "h1g-" + "campaign"
OUTPUT_PREFIX = (
    "campaigns/glm52-sky-20260724/submissions/production/generations/"
)
_SHA = re.compile(r"^[0-9a-f]{64}$")
_S3_BUCKET_ARN = re.compile(
    r"^arn:aws:s3:::[a-z0-9](?:[a-z0-9.-]{1,61}[a-z0-9])?$"
)
_RETAIN = {"DeletionPolicy": "Retain", "UpdateReplacePolicy": "Retain"}


class Task10SupportPlaneError(ValueError):
    """Task 10 retained deployment coordinates or resources drifted."""


@dataclass(frozen=True)
class Task10SupportInputs:
    lambda_code_bucket: str
    lambda_code_key: str
    lambda_code_version: str
    lambda_code_sha256: str
    ledger_table_arn: str
    campaign_bucket_arn: str


def _validate_inputs(inputs: object) -> Task10SupportInputs:
    if not isinstance(inputs, Task10SupportInputs):
        raise Task10SupportPlaneError("Task 10 support inputs must be typed")
    if (
        type(inputs.lambda_code_bucket) is not str
        or not inputs.lambda_code_bucket
        or type(inputs.lambda_code_key) is not str
        or not inputs.lambda_code_key
        or type(inputs.lambda_code_version) is not str
        or not inputs.lambda_code_version
        or _SHA.fullmatch(inputs.lambda_code_sha256) is None
        or inputs.ledger_table_arn
        != (
            "arn:aws:dynamodb:us-west-2:246813579024:"
            "table/keep-glm52-h1g-ledger-v1"
        )
        or _S3_BUCKET_ARN.fullmatch(inputs.campaign_bucket_arn) is None
        or _ABSENT_CAMPAIGN_BUCKET_TOKEN in inputs.campaign_bucket_arn
    ):
        raise Task10SupportPlaneError("Task 10 support inputs drifted")
    return inputs


def build_task10_production_workflow() -> dict[str, object]:
    """Build the single-invocation Standard Workflow."""

    return {
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


def _lambda_statements(inputs: Task10SupportInputs) -> list[dict[str, object]]:
    log_arn = (
        f"arn:aws:logs:{REGION}:{ACCOUNT_ID}:log-group:"
        f"/aws/lambda/{FUNCTION_NAME}:*"
    )
    return [
        {
            "Sid": "AuthenticateExactAwsAccount",
            "Effect": "Allow",
            "Action": "sts:GetCallerIdentity",
            "Resource": "*",
        },
        {
            "Sid": "ReadAndConsumeOnlyRetainedTask8Task9Authority",
            "Effect": "Allow",
            "Action": [
                "dynamodb:GetItem",
                "dynamodb:PutItem",
                "dynamodb:UpdateItem",
            ],
            "Resource": inputs.ledger_table_arn,
            "Condition": {
                "ForAllValues:StringLike": {
                    "dynamodb:LeadingKeys": [f"RUN#{RUN_ID}"]
                }
            },
        },
        {
            "Sid": "OneAttemptReviewedOnDemandP5Launch",
            "Effect": "Allow",
            "Action": "ec2:RunInstances",
            "Resource": [
                f"arn:aws:ec2:{REGION}::image/*",
                f"arn:aws:ec2:{REGION}:{ACCOUNT_ID}:instance/*",
                f"arn:aws:ec2:{REGION}:{ACCOUNT_ID}:network-interface/*",
                f"arn:aws:ec2:{REGION}:{ACCOUNT_ID}:security-group/*",
                f"arn:aws:ec2:{REGION}:{ACCOUNT_ID}:subnet/*",
                f"arn:aws:ec2:{REGION}:{ACCOUNT_ID}:volume/*",
            ],
            "Condition": {
                "StringEquals": {
                    "aws:RequestedRegion": REGION,
                    "ec2:InstanceType": "p5.48xlarge",
                    "ec2:MetadataHttpTokens": "required",
                    "aws:RequestTag/RunId": RUN_ID,
                    "aws:RequestTag/Market": "on-demand",
                }
            },
        },
        {
            "Sid": "CreateOnlyReviewedLaunchTags",
            "Effect": "Allow",
            "Action": "ec2:CreateTags",
            "Resource": [
                f"arn:aws:ec2:{REGION}:{ACCOUNT_ID}:instance/*",
                f"arn:aws:ec2:{REGION}:{ACCOUNT_ID}:volume/*",
            ],
            "Condition": {
                "StringEquals": {
                    "ec2:CreateAction": "RunInstances",
                    "aws:RequestTag/RunId": RUN_ID,
                    "aws:RequestTag/Market": "on-demand",
                },
                "ForAllValues:StringEquals": {
                    "aws:TagKeys": [
                        "Campaign",
                        "Market",
                        "Project",
                        "RunId",
                        "action-key",
                        "activation-id",
                        "activation-ordinal-text",
                        "allocation-ordinal-text",
                        "campaign-identity-sha256",
                        "generation-text",
                        "request-body-sha256",
                        "sky-job-name",
                        "sky-request-id",
                        "sky-task-name",
                        "task-yaml-sha256",
                    ]
                },
            },
        },
        {
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
        },
        {
            "Sid": "AuthenticatedCapacityReadback",
            "Effect": "Allow",
            "Action": ["ec2:DescribeInstances", "ec2:DescribeVolumes"],
            "Resource": "*",
            "Condition": {
                "StringEquals": {"aws:RequestedRegion": REGION}
            },
        },
        {
            "Sid": "ListOnlyTask10TerminalVersions",
            "Effect": "Allow",
            "Action": "s3:ListBucketVersions",
            "Resource": inputs.campaign_bucket_arn,
            "Condition": {
                "StringLike": {"s3:prefix": OUTPUT_PREFIX + "*"}
            },
        },
        {
            "Sid": "PublishAndReconcileExactTask10Terminal",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:GetObjectVersion",
                "s3:PutObject",
            ],
            "Resource": (
                inputs.campaign_bucket_arn + "/" + OUTPUT_PREFIX + "*"
            ),
        },
        {
            "Sid": "WriteOnlyOwnRetainedLogGroup",
            "Effect": "Allow",
            "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
            "Resource": log_arn,
        },
    ]


def _liability_watcher_statements(
    inputs: Task10SupportInputs,
) -> list[dict[str, object]]:
    log_arn = (
        f"arn:aws:logs:{REGION}:{ACCOUNT_ID}:log-group:"
        f"/aws/lambda/{LIABILITY_WATCHER_FUNCTION_NAME}:*"
    )
    return [
        {
            "Sid": "AuthenticateExactAwsAccount",
            "Effect": "Allow",
            "Action": "sts:GetCallerIdentity",
            "Resource": "*",
        },
        {
            "Sid": "ReadAndConsumeOnlyRetainedTask9Liability",
            "Effect": "Allow",
            "Action": [
                "dynamodb:GetItem",
                "dynamodb:Query",
                "dynamodb:TransactWriteItems",
                "dynamodb:UpdateItem",
            ],
            "Resource": inputs.ledger_table_arn,
            "Condition": {
                "ForAllValues:StringLike": {
                    "dynamodb:LeadingKeys": [f"RUN#{RUN_ID}"]
                }
            },
        },
        {
            "Sid": "DiscoverOnlyReviewedRetainedWorkers",
            "Effect": "Allow",
            "Action": "ec2:DescribeInstances",
            "Resource": "*",
            "Condition": {
                "StringEquals": {"aws:RequestedRegion": REGION}
            },
        },
        {
            "Sid": "TerminateOnlyExactRetainedWorkers",
            "Effect": "Allow",
            "Action": "ec2:TerminateInstances",
            "Resource": (
                f"arn:aws:ec2:{REGION}:{ACCOUNT_ID}:instance/*"
            ),
            "Condition": {
                "StringEquals": {
                    "aws:RequestedRegion": REGION,
                    "ec2:ResourceTag/RunId": RUN_ID,
                    "ec2:ResourceTag/Market": "on-demand",
                }
            },
        },
        {
            "Sid": "WriteOnlyOwnRetainedLogGroup",
            "Effect": "Allow",
            "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
            "Resource": log_arn,
        },
    ]


def render_task10_support_plane_fragment(
    *, inputs: Task10SupportInputs
) -> dict[str, Any]:
    """Render one retained Lambda/version and one published workflow version."""

    inputs = _validate_inputs(inputs)
    table_name = inputs.ledger_table_arn.rsplit("/", 1)[-1]
    bucket_name = inputs.campaign_bucket_arn.removeprefix("arn:aws:s3:::")
    workflow = build_task10_production_workflow()
    resources: dict[str, object] = {
        "Task10ProductionReconciliationRole": {
            **_RETAIN,
            "Type": "AWS::IAM::Role",
            "Properties": {
                "RoleName": FUNCTION_NAME,
                "AssumeRolePolicyDocument": {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "lambda.amazonaws.com"},
                            "Action": "sts:AssumeRole",
                        }
                    ],
                },
                "Policies": [
                    {
                        "PolicyName": "Task10RetainedSoleSender",
                        "PolicyDocument": {
                            "Version": "2012-10-17",
                            "Statement": _lambda_statements(inputs),
                        },
                    }
                ],
            },
        },
        "Task10ProductionReconciliationLogGroup": {
            **_RETAIN,
            "Type": "AWS::Logs::LogGroup",
            "Properties": {
                "LogGroupName": f"/aws/lambda/{FUNCTION_NAME}",
                "RetentionInDays": 14,
            },
        },
        "Task10ProductionReconciliationFunction": {
            **_RETAIN,
            "Type": "AWS::Lambda::Function",
            "DependsOn": "Task10ProductionReconciliationLogGroup",
            "Properties": {
                "FunctionName": FUNCTION_NAME,
                "Code": {
                    "S3Bucket": inputs.lambda_code_bucket,
                    "S3Key": inputs.lambda_code_key,
                    "S3ObjectVersion": inputs.lambda_code_version,
                },
                "Handler": "task10_sole_sender_handler.main",
                "Runtime": "python3.12",
                "Architectures": ["arm64"],
                "Role": {
                    "Fn::GetAtt": [
                        "Task10ProductionReconciliationRole",
                        "Arn",
                    ]
                },
                "ReservedConcurrentExecutions": 1,
                "Timeout": 900,
                "MemorySize": 512,
                "Environment": {
                    "Variables": {
                        "GLM52_TASK10_LEDGER_TABLE": table_name,
                        "GLM52_TASK10_CAMPAIGN_BUCKET": bucket_name,
                        "GLM52_TASK10_REGION": REGION,
                    }
                },
            },
        },
        "Task10ProductionReconciliationFunctionVersion": {
            **_RETAIN,
            "Type": "AWS::Lambda::Version",
            "Properties": {
                "FunctionName": {
                    "Ref": "Task10ProductionReconciliationFunction"
                },
                "Description": "code=" + inputs.lambda_code_sha256,
            },
        },
        "Task10ProductionReconciliationErrorAlarm": {
            **_RETAIN,
            "Type": "AWS::CloudWatch::Alarm",
            "Properties": {
                "AlarmName": FUNCTION_NAME + "-errors",
                "Namespace": "AWS/Lambda",
                "MetricName": "Errors",
                "Dimensions": [
                    {
                        "Name": "FunctionName",
                        "Value": FUNCTION_NAME,
                    }
                ],
                "Statistic": "Sum",
                "Period": 60,
                "EvaluationPeriods": 1,
                "DatapointsToAlarm": 1,
                "Threshold": 0,
                "ComparisonOperator": "GreaterThanThreshold",
                "TreatMissingData": "notBreaching",
            },
        },
        "Task10ProductionWorkflowRole": {
            **_RETAIN,
            "Type": "AWS::IAM::Role",
            "Properties": {
                "RoleName": "keep-glm52-h1g-production-workflow",
                "AssumeRolePolicyDocument": {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {
                                "Service": "states.amazonaws.com"
                            },
                            "Action": "sts:AssumeRole",
                        }
                    ],
                },
                "Policies": [
                    {
                        "PolicyName": "InvokeExactTask10SoleSender",
                        "PolicyDocument": {
                            "Version": "2012-10-17",
                            "Statement": [
                                {
                                    "Effect": "Allow",
                                    "Action": "lambda:InvokeFunction",
                                    "Resource": {
                                        "Ref": (
                                            "Task10ProductionReconciliation"
                                            "FunctionVersion"
                                        )
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
        },
        "Task10ProductionStateMachine": {
            **_RETAIN,
            "Type": "AWS::StepFunctions::StateMachine",
            "Properties": {
                "StateMachineName": WORKFLOW_NAME,
                "StateMachineType": "STANDARD",
                "RoleArn": {
                    "Fn::GetAtt": ["Task10ProductionWorkflowRole", "Arn"]
                },
                "DefinitionString": json.dumps(
                    workflow,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "DefinitionSubstitutions": {
                    "Task10ProductionReconciliationVersionArn": {
                        "Ref": (
                            "Task10ProductionReconciliationFunctionVersion"
                        )
                    }
                },
            },
        },
        "Task10ProductionStateMachineVersion": {
            **_RETAIN,
            "Type": "AWS::StepFunctions::StateMachineVersion",
            "Properties": {
                "StateMachineArn": {"Ref": "Task10ProductionStateMachine"},
                "StateMachineRevisionId": {
                    "Fn::GetAtt": [
                        "Task10ProductionStateMachine",
                        "StateMachineRevisionId",
                    ]
                },
            },
        },
        "Task10ProductionReconciliationInvokePermission": {
            **_RETAIN,
            "Type": "AWS::Lambda::Permission",
            "Properties": {
                "Action": "lambda:InvokeFunction",
                "FunctionName": {
                    "Ref": "Task10ProductionReconciliationFunctionVersion"
                },
                "Principal": "states.amazonaws.com",
                "SourceAccount": ACCOUNT_ID,
                "SourceArn": {
                    "Ref": "Task10ProductionStateMachineVersion"
                },
            },
        },
        "Task9LiabilityWatcherRole": {
            **_RETAIN,
            "Type": "AWS::IAM::Role",
            "Properties": {
                "RoleName": LIABILITY_WATCHER_FUNCTION_NAME,
                "AssumeRolePolicyDocument": {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {
                                "Service": "lambda.amazonaws.com"
                            },
                            "Action": "sts:AssumeRole",
                        }
                    ],
                },
                "Policies": [
                    {
                        "PolicyName": "Task9RetainedLiabilityWatcher",
                        "PolicyDocument": {
                            "Version": "2012-10-17",
                            "Statement": _liability_watcher_statements(
                                inputs
                            ),
                        },
                    }
                ],
            },
        },
        "Task9LiabilityWatcherLogGroup": {
            **_RETAIN,
            "Type": "AWS::Logs::LogGroup",
            "Properties": {
                "LogGroupName": (
                    "/aws/lambda/" + LIABILITY_WATCHER_FUNCTION_NAME
                ),
                "RetentionInDays": 14,
            },
        },
        "Task9LiabilityWatcherFunction": {
            **_RETAIN,
            "Type": "AWS::Lambda::Function",
            "DependsOn": "Task9LiabilityWatcherLogGroup",
            "Properties": {
                "FunctionName": LIABILITY_WATCHER_FUNCTION_NAME,
                "Code": {
                    "S3Bucket": inputs.lambda_code_bucket,
                    "S3Key": inputs.lambda_code_key,
                    "S3ObjectVersion": inputs.lambda_code_version,
                },
                "Handler": "task9_liability_watcher_handler.main",
                "Runtime": "python3.12",
                "Architectures": ["arm64"],
                "Role": {
                    "Fn::GetAtt": ["Task9LiabilityWatcherRole", "Arn"]
                },
                "ReservedConcurrentExecutions": 1,
                "Timeout": 60,
                "MemorySize": 256,
                "Environment": {
                    "Variables": {
                        "GLM52_TASK9_LEDGER_TABLE": table_name,
                        "GLM52_TASK9_REGION": REGION,
                    }
                },
            },
        },
        "Task9LiabilityWatcherFunctionVersion": {
            **_RETAIN,
            "Type": "AWS::Lambda::Version",
            "Properties": {
                "FunctionName": {
                    "Ref": "Task9LiabilityWatcherFunction"
                },
                "Description": "code=" + inputs.lambda_code_sha256,
            },
        },
        "Task9LiabilityWatcherErrorAlarm": {
            **_RETAIN,
            "Type": "AWS::CloudWatch::Alarm",
            "Properties": {
                "AlarmName": LIABILITY_WATCHER_FUNCTION_NAME + "-errors",
                "Namespace": "AWS/Lambda",
                "MetricName": "Errors",
                "Dimensions": [
                    {
                        "Name": "FunctionName",
                        "Value": LIABILITY_WATCHER_FUNCTION_NAME,
                    }
                ],
                "Statistic": "Sum",
                "Period": 60,
                "EvaluationPeriods": 1,
                "DatapointsToAlarm": 1,
                "Threshold": 0,
                "ComparisonOperator": "GreaterThanThreshold",
                "TreatMissingData": "notBreaching",
            },
        },
    }
    return {
        "Parameters": {},
        "Resources": resources,
        "Metadata": {
            "Task10ProductionWorkflow": "STANDARD",
            "Task10ProductionSoleSenderState": (
                "RunInternalSixAzSoleSender"
            ),
            "Task10ProductionWriterState": (
                "RunInternalSixAzSoleSender"
            ),
            "Task10ProductionResourceCount": 14,
            "Task10ProductionMaximumEc2Calls": 6,
            "Task10ProductionOrderedAvailabilityZones": list(
                "us-west-2" + letter for letter in "abcdef"
            ),
            "Task10ProductionTerminationOnlyLiability": True,
        },
    }


def validate_task10_support_plane_fragment(
    fragment: Mapping[str, object],
    *,
    inputs: Task10SupportInputs,
) -> bool:
    """Reject any resource, ASL, IAM, or version-coordinate widening."""

    expected = render_task10_support_plane_fragment(inputs=inputs)
    if fragment != expected:
        raise Task10SupportPlaneError(
            "Task 10 retained support-plane fragment drifted"
        )
    resources = fragment["Resources"]
    for logical_id, resource in resources.items():
        if (
            resource.get("DeletionPolicy") != "Retain"
            or resource.get("UpdateReplacePolicy") != "Retain"
        ):
            raise Task10SupportPlaneError(logical_id + " is not retained")
    return True


__all__ = [
    "CAMPAIGN_BUCKET_ARN",
    "Task10SupportInputs",
    "Task10SupportPlaneError",
    "build_task10_production_workflow",
    "render_task10_support_plane_fragment",
    "validate_task10_support_plane_fragment",
]
