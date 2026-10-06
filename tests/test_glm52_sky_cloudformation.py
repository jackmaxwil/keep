from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "aws/glm52-gpu/cfn/gpu-teacher-stack.yaml"
DEPLOY_SCRIPT = ROOT / "aws/glm52-gpu/scripts/deploy_sky_control_plane.sh"

WORKER_START_V2_PARAMETERS = {
    "EnableSkyWorkerStartV2Coordinator",
    "SkyWorkerStartV2FoundationRevision",
    "SkyWorkerStartV2CodeSha256",
    "SkyWorkerStartV2CodeVersionId",
    "SkyWorkerStartV2DescriptorRelativeKey",
    "SkyWorkerStartV2DescriptorFileSha256",
    "SkyWorkerStartV2IntentFileSha256",
    "SkyWorkerStartV2IntentBodySha256",
    "SkyWorkerStartV2ControllerInstanceId",
    "SkyWorkerStartV2ControllerClusterName",
}
WORKER_START_V2_RESOURCES = {
    "SkyWorkerStartV2DeadLetterQueue",
    "SkyWorkerStartV2Role",
    "SkyWorkerStartV2Function",
    "SkyWorkerStartV2EventInvokeConfig",
    "SkyWorkerStartV2LogGroup",
    "SkyWorkerStartV2DeadLetterQueuePolicy",
    "SkyWorkerStartV2ObjectCreatedRule",
    "SkyWorkerStartV2InvokePermission",
    "SkyWorkerStartV2AuthorityBucketPolicy",
    "SkyWorkerStartV2ErrorsAlarm",
    "SkyWorkerStartV2ThrottlesAlarm",
    "SkyWorkerStartV2AgeAlarm",
    "SkyWorkerStartV2DlqAlarm",
    "SkyWorkerStartV2FailedInvocationsAlarm",
}
WORKER_START_V2_OUTPUTS = {
    "SkyWorkerStartV2FunctionArn",
    "SkyWorkerStartV2DeadLetterQueueUrl",
    "SkyWorkerStartV2ObjectCreatedRuleName",
}
WORKER_START_V2_WRITER_ROLES = (
    "InstanceRole",
    "SkyPilotWorkerRole",
    "SkyPilotControllerRole",
    "SkyWatchdogRole",
)


class _Loader(yaml.SafeLoader):
    pass


def _unknown(loader: _Loader, suffix: str, node: yaml.Node):
    if isinstance(node, yaml.ScalarNode):
        return f"!{suffix} {loader.construct_scalar(node)}"
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_mapping(node)


_Loader.add_multi_constructor("!", _unknown)


def _template() -> dict[str, object]:
    return yaml.load(TEMPLATE.read_text(), Loader=_Loader)


def _actions(statement: dict[str, object]) -> set[str]:
    value = statement["Action"]
    return set(value if isinstance(value, list) else [value])


def _legacy_template_projection() -> dict[str, object]:
    template = copy.deepcopy(_template())
    for name in WORKER_START_V2_PARAMETERS:
        template["Parameters"].pop(name, None)
    template["Rules"].pop("SkyWorkerStartV2RequiresExactAuthority", None)
    template["Conditions"].pop("SkyWorkerStartV2CoordinatorEnabled", None)
    for name in WORKER_START_V2_RESOURCES:
        template["Resources"].pop(name, None)
    for name in WORKER_START_V2_OUTPUTS:
        template["Outputs"].pop(name, None)
    template["Resources"]["ModelBucket"]["Properties"].pop(
        "NotificationConfiguration",
        None,
    )
    for role_name in WORKER_START_V2_WRITER_ROLES:
        statements = template["Resources"][role_name]["Properties"]["Policies"][0][
            "PolicyDocument"
        ]["Statement"]
        statements[:] = [
            statement
            for statement in statements
            if statement.get("Sid")
            not in {
                "DenyWritesToSkyWorkerStartV2Evidence",
                "DenyWritesToSkyWorkerStartV2LambdaCode",
            }
        ]
    return template


def test_frozen_template_is_strict_utf8_with_known_service_projection() -> None:
    template_text = TEMPLATE.read_bytes().decode("utf-8")
    assert template_text.count("—") == 10
    assert {character for character in template_text if not character.isascii()} == {
        "—"
    }
    projected = template_text.replace("—", "?").encode()
    assert hashlib.sha256(projected).hexdigest() == (
        "9e57301564efb325f16caecdb4d8a5ff372a7a306e823e3d4ac87aa502c7ad69"
    )


def test_frozen_template_matches_original_uploaded_bytes() -> None:
    assert hashlib.sha256(TEMPLATE.read_bytes()).hexdigest() == (
        "b4c0bd8c725c3c4db951fccadd32707a0b2f8064c2519d264abfbc2f8d2924a4"
    )


def test_worker_start_v2_parameters_condition_and_enable_rule_are_exact() -> None:
    template = _template()
    parameters = template["Parameters"]
    assert {
        name
        for name in parameters
        if name.startswith("SkyWorkerStartV2")
        or name == "EnableSkyWorkerStartV2Coordinator"
    } == WORKER_START_V2_PARAMETERS
    assert parameters["EnableSkyWorkerStartV2Coordinator"] == {
        "Type": "String",
        "Default": "false",
        "AllowedValues": ["true", "false"],
    }
    assert parameters["SkyWorkerStartV2FoundationRevision"] == {
        "Type": "String",
        "Default": "disabled",
        "AllowedValues": ["disabled", "versioned-code-v1"],
    }

    hash_names = (
        "SkyWorkerStartV2CodeSha256",
        "SkyWorkerStartV2DescriptorFileSha256",
        "SkyWorkerStartV2IntentFileSha256",
        "SkyWorkerStartV2IntentBodySha256",
    )
    for name in hash_names:
        parameter = parameters[name]
        assert parameter["Type"] == "String"
        assert parameter["Default"] == "disabled"
        pattern = re.compile(parameter["AllowedPattern"])
        assert pattern.fullmatch("disabled")
        assert pattern.fullmatch("a" * 64)
        assert pattern.fullmatch("A" * 64) is None
        assert pattern.fullmatch("a" * 63) is None

    version = parameters["SkyWorkerStartV2CodeVersionId"]
    assert version["Type"] == "String"
    assert version["Default"] == "disabled"
    version_pattern = re.compile(version["AllowedPattern"])
    assert version_pattern.fullmatch("disabled")
    assert version_pattern.fullmatch(
        "3/L4kqtJlcpXroDTDmJ+rmSpXd3dIbrHY+MTRCxf3vjVBH40="
    )
    for invalid in (
        "",
        "*",
        "../version",
        "version/../id",
        " version",
        "null",
        "None",
    ):
        assert version_pattern.fullmatch(invalid) is None

    descriptor = parameters["SkyWorkerStartV2DescriptorRelativeKey"]
    assert descriptor["Type"] == "String"
    assert descriptor["Default"] == "disabled"
    descriptor_pattern = re.compile(descriptor["AllowedPattern"])
    assert descriptor_pattern.fullmatch("disabled")
    assert descriptor_pattern.fullmatch(
        "submissions/qualification/campaign-descriptor-v2.json"
    )
    assert descriptor_pattern.fullmatch(
        "submissions/qualification/releases/v2/campaign-descriptor-v2.json"
    )
    for invalid in (
        "submissions/seed/campaign-descriptor-v2.json",
        "campaigns/run/submissions/qualification/descriptor.json",
        "submissions/qualification/../descriptor.json",
        "submissions/qualification/descriptor?.json",
        "submissions/qualification/a//descriptor.json",
        "submissions/qualification/a/./descriptor.json",
        "submissions/qualification/a/.hidden.json",
        "submissions/qualification/.json",
        f"submissions/qualification/{'a' * 257}/descriptor.json",
        f"submissions/qualification/{'a' * 252}.json",
    ):
        assert descriptor_pattern.fullmatch(invalid) is None

    instance = parameters["SkyWorkerStartV2ControllerInstanceId"]
    assert instance["Default"] == "disabled"
    instance_pattern = re.compile(instance["AllowedPattern"])
    assert instance_pattern.fullmatch("disabled")
    assert instance_pattern.fullmatch("i-0123456789abcdef0")
    assert instance_pattern.fullmatch("i-01234567") is None
    assert instance_pattern.fullmatch("i-0123456789abcdef01") is None

    cluster = parameters["SkyWorkerStartV2ControllerClusterName"]
    assert cluster["Default"] == "disabled"
    cluster_pattern = re.compile(cluster["AllowedPattern"])
    assert cluster_pattern.fullmatch("disabled")
    assert cluster_pattern.fullmatch("sky-jobs-controller-9d9f31a9")
    assert cluster_pattern.fullmatch("sky-jobs-controller-../other") is None
    assert cluster_pattern.fullmatch("controller-9d9f31a9") is None

    assert template["Conditions"]["SkyWorkerStartV2CoordinatorEnabled"] == [
        "!Condition SkyPilotSupportEnabled",
        ["!Ref EnableSkyWorkerStartV2Coordinator", "true"],
        ["!Ref AWS::AccountId", "246813579024"],
        ["!Ref AWS::Region", "us-west-2"],
    ]
    rule = template["Rules"]["SkyWorkerStartV2RequiresExactAuthority"]
    assert rule["RuleCondition"] == [
        "!Ref EnableSkyWorkerStartV2Coordinator",
        "true",
    ]
    assert [assertion["Assert"] for assertion in rule["Assertions"]] == [
        ["!Ref EnableSkyPilotSupport", "true"],
        ["!Ref AWS::AccountId", "246813579024"],
        ["!Ref AWS::Region", "us-west-2"],
        [["!Ref SkyCampaignRunId", "disabled"]],
        [
            "!Ref SkyWorkerStartV2FoundationRevision",
            "versioned-code-v1",
        ],
        [["!Ref SkyWorkerStartV2CodeSha256", "disabled"]],
        [["!Ref SkyWorkerStartV2CodeVersionId", "disabled"]],
        [["!Ref SkyWorkerStartV2DescriptorRelativeKey", "disabled"]],
        [["!Ref SkyWorkerStartV2DescriptorFileSha256", "disabled"]],
        [["!Ref SkyWorkerStartV2IntentFileSha256", "disabled"]],
        [["!Ref SkyWorkerStartV2IntentBodySha256", "disabled"]],
        [["!Ref SkyWorkerStartV2ControllerInstanceId", "disabled"]],
        [["!Ref SkyWorkerStartV2ControllerClusterName", "disabled"]],
    ]


def test_worker_start_v2_resources_are_independent_and_bucket_notification_is_exact() -> (
    None
):
    template = _template()
    resources = template["Resources"]
    assert {
        name for name in resources if name.startswith("SkyWorkerStartV2")
    } == WORKER_START_V2_RESOURCES
    expected_types = {
        "SkyWorkerStartV2DeadLetterQueue": "AWS::SQS::Queue",
        "SkyWorkerStartV2Role": "AWS::IAM::Role",
        "SkyWorkerStartV2Function": "AWS::Lambda::Function",
        "SkyWorkerStartV2EventInvokeConfig": "AWS::Lambda::EventInvokeConfig",
        "SkyWorkerStartV2LogGroup": "AWS::Logs::LogGroup",
        "SkyWorkerStartV2DeadLetterQueuePolicy": "AWS::SQS::QueuePolicy",
        "SkyWorkerStartV2ObjectCreatedRule": "AWS::Events::Rule",
        "SkyWorkerStartV2InvokePermission": "AWS::Lambda::Permission",
        "SkyWorkerStartV2AuthorityBucketPolicy": "AWS::S3::BucketPolicy",
        "SkyWorkerStartV2ErrorsAlarm": "AWS::CloudWatch::Alarm",
        "SkyWorkerStartV2ThrottlesAlarm": "AWS::CloudWatch::Alarm",
        "SkyWorkerStartV2AgeAlarm": "AWS::CloudWatch::Alarm",
        "SkyWorkerStartV2DlqAlarm": "AWS::CloudWatch::Alarm",
        "SkyWorkerStartV2FailedInvocationsAlarm": "AWS::CloudWatch::Alarm",
    }
    for name, resource_type in expected_types.items():
        assert resources[name]["Type"] == resource_type
        assert resources[name]["Condition"] == ("SkyWorkerStartV2CoordinatorEnabled")
    assert resources["ModelBucket"]["Properties"]["NotificationConfiguration"] == [
        "SkyWorkerStartV2CoordinatorEnabled",
        {"EventBridgeConfiguration": {"EventBridgeEnabled": True}},
        "!Ref AWS::NoValue",
    ]


def test_worker_start_v2_function_event_and_retry_contract_are_exact() -> None:
    resources = _template()["Resources"]
    function = resources["SkyWorkerStartV2Function"]["Properties"]
    assert function["FunctionName"] == "!Sub ${ProjectTag}-sky-worker-start-v2"
    assert function["Runtime"] == "python3.13"
    assert function["Handler"] == "handler.lambda_handler"
    assert function["Role"] == "!GetAtt SkyWorkerStartV2Role.Arn"
    assert function["ReservedConcurrentExecutions"] == 1
    assert function["Timeout"] == 300
    assert function["MemorySize"] == 256
    assert function["DeadLetterConfig"] == {
        "TargetArn": "!GetAtt SkyWorkerStartV2DeadLetterQueue.Arn"
    }
    assert function["Code"] == {
        "S3Bucket": "!Ref ModelBucket",
        "S3Key": ("!Sub lambda/sky-worker-start-v2/${SkyWorkerStartV2CodeSha256}.zip"),
        "S3ObjectVersion": "!Ref SkyWorkerStartV2CodeVersionId",
    }
    assert function["Environment"]["Variables"] == {
        "WORKER_START_EXPECTED_ACCOUNT_ID": "246813579024",
        "WORKER_START_BUCKET": "!Ref ModelBucket",
        "WORKER_START_DESCRIPTOR_KEY": (
            "!Sub campaigns/${SkyCampaignRunId}/"
            "${SkyWorkerStartV2DescriptorRelativeKey}"
        ),
        "WORKER_START_DESCRIPTOR_FILE_SHA256": (
            "!Ref SkyWorkerStartV2DescriptorFileSha256"
        ),
        "WORKER_START_INTENT_KEY": (
            "!Sub campaigns/${SkyCampaignRunId}/submissions/qualification/"
            "intents/${SkyWorkerStartV2IntentBodySha256}/"
            "SKYPILOT_SUBMISSION_INTENT.json"
        ),
        "WORKER_START_INTENT_FILE_SHA256": ("!Ref SkyWorkerStartV2IntentFileSha256"),
        "WORKER_START_INTENT_BODY_SHA256": ("!Ref SkyWorkerStartV2IntentBodySha256"),
    }
    assert resources["SkyWorkerStartV2LogGroup"]["Properties"] == {
        "LogGroupName": "!Sub /aws/lambda/${ProjectTag}-sky-worker-start-v2",
        "RetentionInDays": 14,
    }
    assert resources["SkyWorkerStartV2EventInvokeConfig"]["Properties"] == {
        "FunctionName": "!Ref SkyWorkerStartV2Function",
        "Qualifier": "$LATEST",
        "MaximumRetryAttempts": 2,
        "MaximumEventAgeInSeconds": 900,
    }

    rule = resources["SkyWorkerStartV2ObjectCreatedRule"]
    assert rule["DependsOn"] == "SkyWorkerStartV2DeadLetterQueuePolicy"
    properties = rule["Properties"]
    assert properties["Name"] == "!Sub ${ProjectTag}-sky-worker-start-v2"
    assert properties["State"] == "ENABLED"
    assert properties["EventPattern"] == {
        "account": ["!Ref AWS::AccountId"],
        "region": ["!Ref AWS::Region"],
        "source": ["aws.s3"],
        "detail-type": ["Object Created"],
        "resources": ["!GetAtt ModelBucket.Arn"],
        "detail": {
            "bucket": {"name": ["!Ref ModelBucket"]},
            "reason": ["PutObject"],
            "object": {
                "key": [
                    {
                        "wildcard": (
                            "!Sub campaigns/${SkyCampaignRunId}/monitor/"
                            "must-start/qualification/"
                            "${SkyWorkerStartV2IntentBodySha256}/"
                            "worker-latches/i-*/*.json"
                        )
                    }
                ],
                "version-id": [{"exists": True}],
            },
        },
    }
    assert len(properties["Targets"]) == 1
    target = properties["Targets"][0]
    assert target["Arn"] == "!GetAtt SkyWorkerStartV2Function.Arn"
    assert target["Id"] == "sky-worker-start-v2"
    assert target["RetryPolicy"] == {
        "MaximumEventAgeInSeconds": 900,
        "MaximumRetryAttempts": 2,
    }
    assert target["DeadLetterConfig"] == {
        "Arn": "!GetAtt SkyWorkerStartV2DeadLetterQueue.Arn"
    }
    transformer = target["InputTransformer"]
    assert transformer["InputPathsMap"] == {
        "account_id": "$.account",
        "region": "$.region",
        "bucket": "$.detail.bucket.name",
        "worker_latch_key": "$.detail.object.key",
        "worker_latch_version_id": "$.detail.object.version-id",
    }
    input_template = transformer["InputTemplate"]
    assert input_template.startswith("!Sub ")
    rendered = input_template.removeprefix("!Sub ")
    dynamic_values = {
        "account_id": "246813579024",
        "region": "us-west-2",
        "bucket": "keep-glm52-models-246813579024-us-west-2",
        "worker_latch_key": (
            "campaigns/run/monitor/must-start/qualification/"
            f"{'b' * 64}/worker-latches/i-0123456789abcdef0/"
            f"{'c' * 64}.json"
        ),
        "worker_latch_version_id": "exact-version",
    }
    for name, value in dynamic_values.items():
        rendered = rendered.replace(f"<{name}>", json.dumps(value))
    assert json.loads(rendered) == {
        "schema_version": 1,
        "record_type": "glm52_sky_worker_start_v2_invocation_v1",
        "account_id": dynamic_values["account_id"],
        "region": dynamic_values["region"],
        "bucket": dynamic_values["bucket"],
        "descriptor_key": (
            "campaigns/${SkyCampaignRunId}/${SkyWorkerStartV2DescriptorRelativeKey}"
        ),
        "descriptor_file_sha256": ("${SkyWorkerStartV2DescriptorFileSha256}"),
        "intent_key": (
            "campaigns/${SkyCampaignRunId}/submissions/qualification/"
            "intents/${SkyWorkerStartV2IntentBodySha256}/"
            "SKYPILOT_SUBMISSION_INTENT.json"
        ),
        "intent_file_sha256": "${SkyWorkerStartV2IntentFileSha256}",
        "intent_body_sha256": "${SkyWorkerStartV2IntentBodySha256}",
        "worker_latch_key": dynamic_values["worker_latch_key"],
        "worker_latch_version_id": dynamic_values["worker_latch_version_id"],
    }
    assert resources["SkyWorkerStartV2InvokePermission"]["Properties"] == {
        "Action": "lambda:InvokeFunction",
        "FunctionName": "!Ref SkyWorkerStartV2Function",
        "Principal": "events.amazonaws.com",
        "SourceArn": "!GetAtt SkyWorkerStartV2ObjectCreatedRule.Arn",
        "SourceAccount": "!Ref AWS::AccountId",
    }


def test_worker_start_v2_role_is_least_privilege() -> None:
    resources = _template()["Resources"]
    role = resources["SkyWorkerStartV2Role"]["Properties"]
    assert role["ManagedPolicyArns"] == [
        "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
    ]
    assert role["AssumeRolePolicyDocument"]["Statement"] == [
        {
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }
    ]
    assert len(role["Policies"]) == 1
    statements = role["Policies"][0]["PolicyDocument"]["Statement"]
    assert len(statements) == 8
    dynamic_prefix = (
        "!Sub ${ModelBucket.Arn}/campaigns/${SkyCampaignRunId}/monitor/"
        "must-start/qualification/${SkyWorkerStartV2IntentBodySha256}/"
    )

    reads = next(
        statement
        for statement in statements
        if _actions(statement) == {"s3:GetObject", "s3:GetObjectVersion"}
    )
    assert reads["Effect"] == "Allow"
    assert reads["Resource"] == [
        (
            "!Sub ${ModelBucket.Arn}/campaigns/${SkyCampaignRunId}/"
            "${SkyWorkerStartV2DescriptorRelativeKey}"
        ),
        (
            "!Sub ${ModelBucket.Arn}/campaigns/${SkyCampaignRunId}/"
            "submissions/qualification/intents/"
            "${SkyWorkerStartV2IntentBodySha256}/"
            "SKYPILOT_SUBMISSION_INTENT.json"
        ),
        (
            "!Sub ${ModelBucket.Arn}/campaigns/${SkyCampaignRunId}/"
            "submissions/qualification/acquisitions/"
            "${SkyWorkerStartV2DescriptorFileSha256}/"
            "SUBMISSION_ACQUIRED.json"
        ),
        (
            "!Sub ${ModelBucket.Arn}/campaigns/${SkyCampaignRunId}/"
            "qualification/controller-baselines/*/CONTROLLER_BASELINE.json"
        ),
        (
            "!Sub ${ModelBucket.Arn}/campaigns/${SkyCampaignRunId}/"
            "submissions/qualification/accepted/*/"
            "SKYPILOT_SUBMISSION_ACCEPTED.json"
        ),
        dynamic_prefix + "DYNAMIC_JOB_BINDING.json",
        dynamic_prefix + "observations/*.json",
        dynamic_prefix + "worker-latches/*/*.json",
        dynamic_prefix + "worker-controller-observations/*/*.json",
        (dynamic_prefix + "worker-acceptances/*/*/WORKER_START_ACCEPTED.json"),
    ]

    lists = next(
        statement
        for statement in statements
        if _actions(statement) == {"s3:ListBucket"}
    )
    assert lists == {
        "Sid": "ListExactWorkerStartV2Authority",
        "Effect": "Allow",
        "Action": "s3:ListBucket",
        "Resource": "!GetAtt ModelBucket.Arn",
        "Condition": {
            "StringEquals": {
                "s3:prefix": [
                    (
                        "!Sub campaigns/${SkyCampaignRunId}/submissions/"
                        "qualification/accepted/"
                    ),
                    (
                        "!Sub campaigns/${SkyCampaignRunId}/monitor/"
                        "must-start/qualification/"
                        "${SkyWorkerStartV2IntentBodySha256}/"
                    ),
                    (
                        "!Sub campaigns/${SkyCampaignRunId}/monitor/"
                        "must-start/qualification/"
                        "${SkyWorkerStartV2IntentBodySha256}/"
                        "worker-acceptances/"
                    ),
                ]
            }
        },
    }

    writes = next(
        statement for statement in statements if _actions(statement) == {"s3:PutObject"}
    )
    assert writes["Effect"] == "Allow"
    assert writes["Resource"] == [
        dynamic_prefix + "worker-controller-observations/*/*.json",
        dynamic_prefix + "worker-acceptances/*/*/WORKER_START_ACCEPTED.json",
    ]

    describe_instances = next(
        statement
        for statement in statements
        if _actions(statement) == {"ec2:DescribeInstances"}
    )
    assert describe_instances["Resource"] == "*"
    assert describe_instances["Condition"] == {
        "StringEquals": {"aws:RequestedRegion": "!Ref AWS::Region"}
    }
    ssm_reads = next(
        statement
        for statement in statements
        if _actions(statement)
        == {
            "ssm:DescribeInstanceInformation",
            "ssm:GetCommandInvocation",
        }
    )
    assert ssm_reads["Resource"] == "*"
    assert ssm_reads["Condition"] == {
        "StringEquals": {"aws:RequestedRegion": "!Ref AWS::Region"}
    }
    send_document = next(
        statement
        for statement in statements
        if _actions(statement) == {"ssm:SendCommand"}
        and str(statement["Resource"]).endswith(
            ":ssm:${AWS::Region}::document/AWS-RunShellScript"
        )
    )
    assert send_document["Resource"] == (
        "!Sub arn:${AWS::Partition}:ssm:${AWS::Region}::document/AWS-RunShellScript"
    )
    send_instance = next(
        statement
        for statement in statements
        if _actions(statement) == {"ssm:SendCommand"}
        and str(statement["Resource"]).endswith(
            "instance/${SkyWorkerStartV2ControllerInstanceId}"
        )
    )
    assert send_instance["Resource"] == (
        "!Sub arn:${AWS::Partition}:ec2:${AWS::Region}:"
        "${AWS::AccountId}:instance/"
        "${SkyWorkerStartV2ControllerInstanceId}"
    )
    assert send_instance["Condition"] == {
        "StringEquals": {
            "ssm:resourceTag/project": "!Ref ProjectTag",
            "ssm:resourceTag/owner": "!Ref OwnerTag",
            "ssm:resourceTag/model": "glm-5.2",
            "ssm:resourceTag/campaign-run-id": "!Ref SkyCampaignRunId",
            "ssm:resourceTag/cost-allocation": "glm52-sky-campaign",
            "ssm:resourceTag/ray-cluster-name": (
                "!Ref SkyWorkerStartV2ControllerClusterName"
            ),
        }
    }
    dead_letter = next(
        statement
        for statement in statements
        if _actions(statement) == {"sqs:SendMessage"}
    )
    assert dead_letter["Resource"] == ("!GetAtt SkyWorkerStartV2DeadLetterQueue.Arn")

    allowed_actions = {
        action
        for statement in statements
        if statement["Effect"] == "Allow"
        for action in _actions(statement)
    }
    assert allowed_actions == {
        "s3:GetObject",
        "s3:GetObjectVersion",
        "s3:ListBucket",
        "s3:PutObject",
        "ec2:DescribeInstances",
        "ssm:DescribeInstanceInformation",
        "ssm:GetCommandInvocation",
        "ssm:SendCommand",
        "sqs:SendMessage",
    }
    assert "sts:GetCallerIdentity" not in allowed_actions


def test_worker_start_v2_output_namespaces_have_one_writer_authority() -> None:
    resources = _template()["Resources"]
    dynamic_prefix = (
        "!Sub ${ModelBucket.Arn}/campaigns/${SkyCampaignRunId}/monitor/"
        "must-start/qualification/${SkyWorkerStartV2IntentBodySha256}/"
    )
    evidence_deny = {
        "Sid": "DenyWritesToSkyWorkerStartV2Evidence",
        "Effect": "Deny",
        "Action": [
            "s3:PutObject",
            "s3:DeleteObject",
            "s3:DeleteObjectVersion",
        ],
        "Resource": [
            dynamic_prefix + "worker-controller-observations/*",
            dynamic_prefix + "worker-acceptances/*",
        ],
    }
    code_deny = {
        "Sid": "DenyWritesToSkyWorkerStartV2LambdaCode",
        "Effect": "Deny",
        "Action": [
            "s3:PutObject",
            "s3:DeleteObject",
            "s3:DeleteObjectVersion",
        ],
        "Resource": [
            "!Sub ${ModelBucket.Arn}/lambda/sky-worker-start-v2/*",
        ],
    }
    for role_name in WORKER_START_V2_WRITER_ROLES:
        statements = resources[role_name]["Properties"]["Policies"][0][
            "PolicyDocument"
        ]["Statement"]
        assert (
            next(
                statement
                for statement in statements
                if statement.get("Sid") == evidence_deny["Sid"]
            )
            == evidence_deny
        )
        assert (
            next(
                statement
                for statement in statements
                if statement.get("Sid") == code_deny["Sid"]
            )
            == code_deny
        )

    bucket_policy = resources["SkyWorkerStartV2AuthorityBucketPolicy"]
    assert bucket_policy["Properties"]["Bucket"] == "!Ref ModelBucket"
    assert bucket_policy["Properties"]["PolicyDocument"] == {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "DenyNonCoordinatorWorkerStartV2EvidenceWrites",
                "Effect": "Deny",
                "Principal": "*",
                "Action": [
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:DeleteObjectVersion",
                ],
                "Resource": [
                    dynamic_prefix + "worker-controller-observations/*",
                    dynamic_prefix + "worker-acceptances/*",
                ],
                "Condition": {
                    "ArnNotEquals": {
                        "aws:PrincipalArn": ("!GetAtt SkyWorkerStartV2Role.Arn")
                    }
                },
            }
        ],
    }


def test_worker_start_v2_dlq_and_alarm_family_are_exact() -> None:
    resources = _template()["Resources"]
    queue = resources["SkyWorkerStartV2DeadLetterQueue"]["Properties"]
    assert queue == {
        "QueueName": "!Sub ${ProjectTag}-sky-worker-start-v2-dlq",
        "MessageRetentionPeriod": 1209600,
        "SqsManagedSseEnabled": True,
        "Tags": [
            {"Key": "project", "Value": "!Ref ProjectTag"},
            {"Key": "owner", "Value": "!Ref OwnerTag"},
            {"Key": "model", "Value": "glm-5.2"},
            {"Key": "campaign-run-id", "Value": "!Ref SkyCampaignRunId"},
        ],
    }
    queue_policy = resources["SkyWorkerStartV2DeadLetterQueuePolicy"]["Properties"]
    assert queue_policy["Queues"] == ["!Ref SkyWorkerStartV2DeadLetterQueue"]
    assert queue_policy["PolicyDocument"] == {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "events.amazonaws.com"},
                "Action": "sqs:SendMessage",
                "Resource": "!GetAtt SkyWorkerStartV2DeadLetterQueue.Arn",
                "Condition": {
                    "StringEquals": {"aws:SourceAccount": "!Ref AWS::AccountId"},
                    "ArnEquals": {
                        "aws:SourceArn": (
                            "!Sub arn:${AWS::Partition}:events:"
                            "${AWS::Region}:${AWS::AccountId}:rule/"
                            "${ProjectTag}-sky-worker-start-v2"
                        )
                    },
                },
            }
        ],
    }

    alarm_names = {
        "SkyWorkerStartV2ErrorsAlarm",
        "SkyWorkerStartV2ThrottlesAlarm",
        "SkyWorkerStartV2AgeAlarm",
        "SkyWorkerStartV2DlqAlarm",
        "SkyWorkerStartV2FailedInvocationsAlarm",
    }
    assert {
        name
        for name in resources
        if name.startswith("SkyWorkerStartV2") and name.endswith("Alarm")
    } == alarm_names
    for name in alarm_names:
        alarm = resources[name]
        assert alarm["DependsOn"] == "CampaignAlertTopicPolicy"
        properties = alarm["Properties"]
        assert properties["Period"] == 60
        assert properties["Threshold"] in {0, 120000}
        assert properties["ComparisonOperator"] == "GreaterThanThreshold"
        assert properties["TreatMissingData"] == "notBreaching"
        assert properties["AlarmActions"] == ["!Ref CampaignAlertTopic"]

    errors = resources["SkyWorkerStartV2ErrorsAlarm"]["Properties"]
    assert errors["Namespace"] == "AWS/Lambda"
    assert errors["MetricName"] == "Errors"
    assert errors["Statistic"] == "Sum"
    assert errors["EvaluationPeriods"] == 5
    assert errors["DatapointsToAlarm"] == 3
    assert errors["Dimensions"] == [
        {"Name": "FunctionName", "Value": "!Ref SkyWorkerStartV2Function"}
    ]

    throttles = resources["SkyWorkerStartV2ThrottlesAlarm"]["Properties"]
    assert throttles["Namespace"] == "AWS/Lambda"
    assert throttles["MetricName"] == "Throttles"
    assert throttles["EvaluationPeriods"] == 1
    assert throttles["DatapointsToAlarm"] == 1
    assert throttles["Dimensions"] == [
        {"Name": "FunctionName", "Value": "!Ref SkyWorkerStartV2Function"}
    ]

    age = resources["SkyWorkerStartV2AgeAlarm"]["Properties"]
    assert age["Namespace"] == "AWS/Lambda"
    assert age["MetricName"] == "AsyncEventAge"
    assert age["Statistic"] == "Maximum"
    assert age["EvaluationPeriods"] == 1
    assert age["DatapointsToAlarm"] == 1
    assert age["Threshold"] == 120000
    assert age["Dimensions"] == [
        {"Name": "FunctionName", "Value": "!Ref SkyWorkerStartV2Function"}
    ]

    dlq = resources["SkyWorkerStartV2DlqAlarm"]["Properties"]
    assert dlq["Namespace"] == "AWS/SQS"
    assert dlq["MetricName"] == "ApproximateNumberOfMessagesVisible"
    assert dlq["Statistic"] == "Maximum"
    assert dlq["EvaluationPeriods"] == 1
    assert dlq["DatapointsToAlarm"] == 1
    assert dlq["Dimensions"] == [
        {
            "Name": "QueueName",
            "Value": "!GetAtt SkyWorkerStartV2DeadLetterQueue.QueueName",
        }
    ]

    failed = resources["SkyWorkerStartV2FailedInvocationsAlarm"]["Properties"]
    assert failed["Namespace"] == "AWS/Events"
    assert failed["MetricName"] == "FailedInvocations"
    assert failed["Statistic"] == "Sum"
    assert failed["EvaluationPeriods"] == 1
    assert failed["DatapointsToAlarm"] == 1
    assert failed["Dimensions"] == [
        {
            "Name": "RuleName",
            "Value": "!Ref SkyWorkerStartV2ObjectCreatedRule",
        }
    ]


def test_worker_start_v2_outputs_are_conditioned_and_legacy_projection_is_frozen() -> (
    None
):
    template = _template()
    assert {
        name for name in template["Outputs"] if name.startswith("SkyWorkerStartV2")
    } == WORKER_START_V2_OUTPUTS
    assert template["Outputs"]["SkyWorkerStartV2FunctionArn"] == {
        "Condition": "SkyWorkerStartV2CoordinatorEnabled",
        "Value": "!GetAtt SkyWorkerStartV2Function.Arn",
    }
    assert template["Outputs"]["SkyWorkerStartV2DeadLetterQueueUrl"] == {
        "Condition": "SkyWorkerStartV2CoordinatorEnabled",
        "Value": "!Ref SkyWorkerStartV2DeadLetterQueue",
    }
    assert template["Outputs"]["SkyWorkerStartV2ObjectCreatedRuleName"] == {
        "Condition": "SkyWorkerStartV2CoordinatorEnabled",
        "Value": "!Ref SkyWorkerStartV2ObjectCreatedRule",
    }

    projection = json.dumps(
        _legacy_template_projection(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    assert hashlib.sha256(projection).hexdigest() == (
        "8a590e906013ac0b61c73172bca55e8f070c1df99570baf212f9a7d05fb18074"
    )


def test_sky_support_is_independent_and_legacy_controller_defaults_disabled() -> None:
    template = _template()
    parameters = template["Parameters"]
    conditions = template["Conditions"]
    assert parameters["EnableCampaignController"]["Default"] == "false"
    assert parameters["EnableSkyPilotSupport"]["Default"] == "false"
    assert "CampaignControllerEnabled" in conditions
    assert "SkyPilotSupportEnabled" in conditions
    assert "CampaignAlertsEnabled" in conditions
    resources = template["Resources"]
    assert resources["CapacityBlockDeliveredRule"]["Condition"] == (
        "CampaignControllerEnabled"
    )
    assert resources["SkyWatchdogRule"]["Condition"] == "SkyPilotSupportEnabled"


def test_sky_roles_have_exact_names_and_forbid_spot_purchase_and_worker_termination() -> None:
    resources = _template()["Resources"]
    security_group = resources["InstanceSecurityGroup"]["Properties"]
    assert security_group["GroupName"] == "keep-glm52-gpu-sg"
    worker = resources["SkyPilotWorkerRole"]["Properties"]
    controller = resources["SkyPilotControllerRole"]["Properties"]
    assert worker["RoleName"] == "keep-glm52-gpu-worker"
    assert controller["RoleName"] == "keep-glm52-skypilot-controller"
    worker_statements = worker["Policies"][0]["PolicyDocument"]["Statement"]
    controller_statements = controller["Policies"][0]["PolicyDocument"]["Statement"]
    worker_deny = next(
        statement
        for statement in worker_statements
        if statement.get("Sid") == "DenyUnapprovedGpuMarketsAndSelfTermination"
    )
    controller_deny = next(
        statement
        for statement in controller_statements
        if statement.get("Sid") == "DenyUnapprovedPurchaseAndSpot"
    )
    forbidden = {
        "ec2:PurchaseCapacityBlock",
        "ec2:PurchaseCapacityBlockExtension",
        "ec2:RequestSpotInstances",
        "ec2:CreateFleet",
    }
    assert forbidden.issubset(set(worker_deny["Action"]))
    assert forbidden.issubset(set(controller_deny["Action"]))
    assert "ec2:TerminateInstances" in worker_deny["Action"]
    assert all(
        not (
            statement.get("Effect") == "Allow"
            and "ec2:TerminateInstances"
            in (
                statement.get("Action")
                if isinstance(statement.get("Action"), list)
                else [statement.get("Action")]
            )
        )
        for statement in worker_statements
    )
    writes = [
        statement
        for statement in worker_statements
        if statement.get("Effect") == "Allow"
        and "s3:PutObject"
        in (
            statement.get("Action")
            if isinstance(statement.get("Action"), list)
            else [statement.get("Action")]
        )
    ]
    assert writes == [
        {
            "Effect": "Allow",
            "Action": ["s3:PutObject", "s3:AbortMultipartUpload"],
            "Resource": [
                "!Sub ${ModelBucket.Arn}/campaigns/${SkyCampaignRunId}/*",
                "!Sub ${ModelBucket.Arn}/qualification-cache/seeds/${SkyCampaignRunId}/*",
            ],
        }
    ]
    assert next(
        statement
        for statement in worker_statements
        if statement.get("Sid") == "DenyGenerationTerminalPublication"
    ) == {
        "Sid": "DenyGenerationTerminalPublication",
        "Effect": "Deny",
        "Action": [
            "s3:PutObject",
            "s3:DeleteObject",
            "s3:DeleteObjectVersion",
        ],
        "Resource": (
            "!Sub ${ModelBucket.Arn}/campaigns/${SkyCampaignRunId}/"
            "submissions/production/generations/*/terminal/"
            "CAMPAIGN_DRAINED.json"
        ),
    }
    run = next(
        statement
        for statement in controller_statements
        if statement.get("Sid") == "RunOneTaggedP5Instance"
    )
    assert run["Condition"]["StringEquals"]["ec2:InstanceType"] == "p5.48xlarge"
    assert run["Condition"]["StringEquals"]["aws:RequestedRegion"] == (
        "!Ref AWS::Region"
    )
    assert all(
        statement.get("Sid") != "ManageCampaignSecurityGroups"
        for statement in controller_statements
    )


def test_sky_runtime_roles_cannot_write_coordinator_must_start_evidence() -> None:
    resources = _template()["Resources"]
    protected_prefix = (
        "!Sub ${ModelBucket.Arn}/campaigns/${SkyCampaignRunId}/monitor/"
        "must-start/${SkyMustStartManagedMode}/"
        "${SkyMustStartSubmissionBodySha256}/"
    )
    protected_resources = [
        protected_prefix + "CANCEL_REQUESTED.json",
        protected_prefix + "CANCEL_COMPLETED.json",
        protected_prefix + "TARGET_BOUND.json",
        protected_prefix + "JOB_BINDING.json",
        protected_prefix + "TIMELY_START_ACCEPTED.json",
        protected_prefix + "CANCEL_SUPERSEDED.json",
        protected_prefix + "REQUEST_ALERTED.json",
        protected_prefix + "COMPLETION_ALERTED.json",
        protected_prefix + "observations/*",
    ]
    expected = {
        "Sid": "DenyWritesToCoordinatorMustStartEvidence",
        "Effect": "Deny",
        "Action": ["s3:PutObject", "s3:DeleteObject"],
        "Resource": protected_resources,
    }
    for role_name in (
        "InstanceRole",
        "SkyPilotWorkerRole",
        "SkyPilotControllerRole",
        "SkyWatchdogRole",
    ):
        statements = resources[role_name]["Properties"]["Policies"][0][
            "PolicyDocument"
        ]["Statement"]
        deny = next(
            (
                statement
                for statement in statements
                if statement.get("Sid")
                == "DenyWritesToCoordinatorMustStartEvidence"
            ),
            None,
        )
        assert deny == expected


def test_sky_runtime_roles_cannot_overwrite_versioned_must_start_lambda_code() -> None:
    resources = _template()["Resources"]
    expected = {
        "Sid": "DenyWritesToMustStartLambdaCode",
        "Effect": "Deny",
        "Action": ["s3:PutObject", "s3:DeleteObject"],
        "Resource": [
            "!Sub ${ModelBucket.Arn}/lambda/sky-must-start-cancel/*"
        ],
    }
    for role_name in (
        "InstanceRole",
        "SkyPilotWorkerRole",
        "SkyPilotControllerRole",
        "SkyWatchdogRole",
    ):
        statements = resources[role_name]["Properties"]["Policies"][0][
            "PolicyDocument"
        ]["Statement"]
        deny = next(
            (
                statement
                for statement in statements
                if statement.get("Sid") == "DenyWritesToMustStartLambdaCode"
            ),
            None,
        )
        assert deny == expected


def test_sky_controller_can_discover_regions_only_through_catalog_home() -> None:
    resources = _template()["Resources"]
    controller = resources["SkyPilotControllerRole"]["Properties"]
    statements = controller["Policies"][0]["PolicyDocument"]["Statement"]
    discovery = next(
        statement
        for statement in statements
        if statement.get("Sid") == "SkyPilotRegionDiscovery"
    )
    assert discovery == {
        "Sid": "SkyPilotRegionDiscovery",
        "Effect": "Allow",
        "Action": "ec2:DescribeRegions",
        "Resource": "*",
        "Condition": {
            "StringEquals": {
                "aws:RequestedRegion": "us-east-1",
            }
        },
    }


def test_sky_controller_separates_p5_instance_from_launch_dependencies() -> None:
    resources = _template()["Resources"]
    controller = resources["SkyPilotControllerRole"]["Properties"]
    statements = controller["Policies"][0]["PolicyDocument"]["Statement"]
    image = next(
        statement
        for statement in statements
        if statement.get("Sid") == "RunApprovedImage"
    )
    instance = next(
        statement
        for statement in statements
        if statement.get("Sid") == "RunOneTaggedP5Instance"
    )
    dependencies = next(
        statement
        for statement in statements
        if statement.get("Sid") == "UseApprovedP5LaunchDependencies"
    )
    assert image["Resource"] == (
        "!Sub arn:${AWS::Partition}:ec2:${AWS::Region}::image/ami-*"
    )
    assert image["Condition"] == {
        "StringEquals": {
            "aws:RequestedRegion": "!Ref AWS::Region",
        }
    }
    assert instance["Resource"] == (
        "!Sub arn:${AWS::Partition}:ec2:${AWS::Region}:${AWS::AccountId}:instance/*"
    )
    assert instance["Condition"]["StringEquals"] == {
        "ec2:InstanceType": "p5.48xlarge",
        "aws:RequestedRegion": "!Ref AWS::Region",
        "aws:RequestTag/project": "!Ref ProjectTag",
        "aws:RequestTag/owner": "!Ref OwnerTag",
        "aws:RequestTag/model": "glm-5.2",
        "aws:RequestTag/campaign-run-id": "!Ref SkyCampaignRunId",
        "aws:RequestTag/cost-allocation": "glm52-sky-campaign",
    }
    assert dependencies["Resource"] == [
        "!Sub arn:${AWS::Partition}:ec2:${AWS::Region}:${AWS::AccountId}:network-interface/*",
        "!Sub arn:${AWS::Partition}:ec2:${AWS::Region}:${AWS::AccountId}:volume/*",
        "!Sub arn:${AWS::Partition}:ec2:${AWS::Region}:${AWS::AccountId}:subnet/${PublicSubnet}",
        "!Sub arn:${AWS::Partition}:ec2:${AWS::Region}:${AWS::AccountId}:subnet/${PublicSubnetAltAz}",
        "!Sub arn:${AWS::Partition}:ec2:${AWS::Region}:${AWS::AccountId}:subnet/${PublicSubnetAz3}",
        "!Sub arn:${AWS::Partition}:ec2:${AWS::Region}:${AWS::AccountId}:subnet/${PublicSubnetAz4}",
        "!Sub arn:${AWS::Partition}:ec2:${AWS::Region}:${AWS::AccountId}:security-group/${InstanceSecurityGroup.GroupId}",
    ]
    assert dependencies["Condition"] == {
        "StringEquals": {
            "aws:RequestedRegion": "!Ref AWS::Region",
        }
    }
    assert all(
        "ec2:CreateSecurityGroup"
        not in (
            statement.get("Action")
            if isinstance(statement.get("Action"), list)
            else [statement.get("Action")]
        )
        for statement in statements
    )


def test_watchdog_has_ten_minute_schedule_dlq_alarms_and_14_day_logs() -> None:
    resources = _template()["Resources"]
    assert (
        resources["SkyWatchdogRule"]["Properties"]["ScheduleExpression"]
        == "rate(10 minutes)"
    )
    target = resources["SkyWatchdogRule"]["Properties"]["Targets"][0]
    assert "DeadLetterConfig" in target
    function = resources["SkyWatchdogFunction"]["Properties"]
    assert function["DeadLetterConfig"] == {
        "TargetArn": "!GetAtt SkyWatchdogDeadLetterQueue.Arn"
    }
    role_statements = resources["SkyWatchdogRole"]["Properties"]["Policies"][0][
        "PolicyDocument"
    ]["Statement"]
    queue_access = next(
        statement
        for statement in role_statements
        if statement.get("Resource") == "!GetAtt SkyWatchdogDeadLetterQueue.Arn"
    )
    assert set(queue_access["Action"]) == {
        "sqs:GetQueueAttributes",
        "sqs:SendMessage",
    }
    campaign_list = next(
        statement
        for statement in role_statements
        if statement.get("Sid") == "ListCampaignMarkers"
    )
    assert campaign_list["Action"] == "s3:ListBucket"
    assert campaign_list["Resource"] == "!GetAtt ModelBucket.Arn"
    assert campaign_list["Condition"] == {
        "StringLike": {
            "s3:prefix": [
                "!Sub campaigns/${SkyCampaignRunId}",
                "!Sub campaigns/${SkyCampaignRunId}/*",
            ]
        }
    }
    assert resources["SkyWatchdogLogGroup"]["Properties"]["RetentionInDays"] == 14
    for alarm in (
        "SkyWatchdogErrorsAlarm",
        "SkyWatchdogThrottlesAlarm",
        "SkyWatchdogDlqAlarm",
        "SkyWatchdogFailedInvocationsAlarm",
    ):
        assert resources[alarm]["Properties"]["AlarmActions"] == [
            "!Ref CampaignAlertTopic"
        ]
    subscription = resources["CampaignAlertEmailSubscription"]["Properties"]
    assert subscription["Endpoint"] == "!Ref CampaignAlertEmail"
    assert resources["CampaignAlertTopic"]["Condition"] == "CampaignAlertsEnabled"


def test_must_start_coordinator_is_one_minute_run_bound_and_independent() -> None:
    template = _template()
    parameters = template["Parameters"]
    resources = template["Resources"]
    assert parameters["SkyMustStartCancelCodeSha256"]["Default"] == "disabled"
    assert parameters["SkyMustStartCancelCodeVersionId"]["Default"] == "disabled"
    assert parameters["EnableSkyMustStartCancel"]["Default"] == "false"
    assert parameters["SkyMustStartSubmissionBodySha256"]["Default"] == "disabled"
    baseline_parameter = parameters["SkyMustStartControllerBaselineBodySha256"]
    assert baseline_parameter["Default"] == "disabled"
    baseline_pattern = re.compile(baseline_parameter["AllowedPattern"])
    assert baseline_pattern.fullmatch("a" * 64)
    assert baseline_pattern.fullmatch("A" * 64) is None
    target_parameter = parameters["SkyMustStartTargetJobId"]
    assert target_parameter["Type"] == "String"
    assert target_parameter["Default"] == "-1"
    target_pattern = re.compile(target_parameter["AllowedPattern"])
    assert target_pattern.fullmatch("-1")
    assert target_pattern.fullmatch("1")
    assert target_pattern.fullmatch("3")
    assert target_pattern.fullmatch("0") is None
    assert target_pattern.fullmatch("-2") is None
    assert template["Conditions"]["SkyMustStartCancelEnabled"] == [
        "!Condition SkyPilotSupportEnabled",
        ["!Ref EnableSkyMustStartCancel", "true"],
    ]
    assert template["Conditions"]["SkyMustStartCoordinatorEnabled"] == [
        "!Condition SkyPilotSupportEnabled",
        [
            ["!Ref EnableSkyMustStartObserve", "true"],
            ["!Ref EnableSkyMustStartCancel", "true"],
        ],
    ]
    target_rule = template["Rules"][
        "SkyMustStartCancelRequiresObservedTarget"
    ]
    assert target_rule["RuleCondition"] == [
        ["!Ref EnableSkyMustStartObserve", "true"],
        ["!Ref EnableSkyMustStartCancel", "true"],
    ]
    assert target_rule["Assertions"][0]["Assert"] == [
        ["!Ref SkyMustStartTargetJobId", "-1"]
    ]
    for name in (
        "SkyMustStartCancelFunction",
        "SkyMustStartCancelRole",
        "SkyMustStartCancelRule",
        "SkyMustStartCancelDeadLetterQueue",
        "SkyMustStartCancelLogGroup",
        "SkyMustStartCancelInvokePermission",
        "SkyMustStartSchedulerRole",
        "SkyMustStartDeadlineSchedule",
    ):
        assert resources[name]["Condition"] == "SkyMustStartCoordinatorEnabled"
    assert parameters["SkyMustStartManagedMode"]["AllowedValues"] == [
        "production",
        "qualification",
        "cache-seed",
    ]
    function = resources["SkyMustStartCancelFunction"]["Properties"]
    assert function["ReservedConcurrentExecutions"] == 1
    assert function["Timeout"] == 300
    assert function["DeadLetterConfig"] == {
        "TargetArn": "!GetAtt SkyMustStartCancelDeadLetterQueue.Arn"
    }
    assert function["Environment"]["Variables"] == {
        "EXPECTED_ACCOUNT_ID": "246813579024",
        "CAMPAIGN_BUCKET": "!Ref ModelBucket",
        "CAMPAIGN_DESCRIPTOR_KEY": (
            "!Sub campaigns/${SkyCampaignRunId}/"
            "${SkyMustStartDescriptorRelativeKey}"
        ),
        "IMMUTABLE_SUBMISSION_KEY": (
            "!Sub campaigns/${SkyCampaignRunId}/monitor/submission-locks/"
            "${SkyMustStartDescriptorFileSha256}-"
            "${SkyMustStartManagedMode}.json"
        ),
        "SUBMISSION_BODY_SHA256": "!Ref SkyMustStartSubmissionBodySha256",
        "EXPECTED_CONTROLLER_BASELINE_BODY_SHA256": (
            "!Ref SkyMustStartControllerBaselineBodySha256"
        ),
        "EXPECTED_TARGET_JOB_ID": "!Ref SkyMustStartTargetJobId",
        "OBSERVE_ONLY": ["SkyMustStartCancelEnabled", "false", "true"],
        "EXPECTED_WORKSPACE": "!Ref SkyMustStartWorkspace",
        "EXPECTED_DESCRIPTOR_FILE_SHA256": (
            "!Ref SkyMustStartDescriptorFileSha256"
        ),
        "EXPECTED_SUBMISSION_SUBMITTED_AT": (
            "!Ref SkyMustStartSubmissionSubmittedAt"
        ),
        "EXPECTED_CONTROLLER_INSTANCE_ID": (
            "!Ref SkyMustStartControllerInstanceId"
        ),
        "EXPECTED_CONTROLLER_INSTANCE_TYPE": (
            "!Ref SkyMustStartControllerInstanceType"
        ),
        "EXPECTED_CONTROLLER_PROFILE_ARN": (
            "!Ref SkyMustStartControllerProfileArn"
        ),
        "EXPECTED_CONTROLLER_CLUSTER_NAME": (
            "!Ref SkyMustStartControllerClusterName"
        ),
        "STARTING_GRACE_SECONDS": "!Ref SkyMustStartStartingGraceSeconds",
        "PRIMARY_WAKE_MAX_WAIT_SECONDS": "120",
        "RECONCILIATION_RULE_NAME": (
            "!Sub ${ProjectTag}-sky-must-start-cancel"
        ),
        "EXPECTED_ACTIVATION_JOB_BINDING_SHA256": (
            "!Ref SkyMustStartActivationJobBindingSha256"
        ),
        "EXPECTED_ACTIVATION_OBSERVATION_SHA256": (
            "!Ref SkyMustStartActivationObservationSha256"
        ),
        "CAMPAIGN_RUN_ID": "!Ref SkyCampaignRunId",
        "MANAGED_MODE": "!Ref SkyMustStartManagedMode",
        "SKY_JOB_NAME": "!Ref SkyMustStartJobName",
        "MUST_START_BY": "!Ref SkyMustStartBy",
        "ALERT_TOPIC_ARN": "!Ref CampaignAlertTopic",
    }
    assert resources["SkyMustStartCancelLogGroup"]["Properties"][
        "RetentionInDays"
    ] == 14
    rule = resources["SkyMustStartCancelRule"]["Properties"]
    assert rule["ScheduleExpression"] == "rate(1 minute)"
    target = rule["Targets"][0]
    assert target["Input"] == (
        '!Sub {"campaign_run_id":"${SkyCampaignRunId}",'
        '"managed_mode":"${SkyMustStartManagedMode}",'
        '"sky_job_name":"${SkyMustStartJobName}",'
        '"must_start_by":"${SkyMustStartBy}",'
        '"trigger":"reconcile"}'
    )
    assert target["RetryPolicy"] == {
        "MaximumEventAgeInSeconds": 300,
        "MaximumRetryAttempts": 2,
    }
    assert target["DeadLetterConfig"] == {
        "Arn": "!GetAtt SkyMustStartCancelDeadLetterQueue.Arn"
    }


def test_must_start_role_can_only_read_exact_authorities_write_markers_and_cancel_via_ssm() -> None:
    resources = _template()["Resources"]
    statements = resources["SkyMustStartCancelRole"]["Properties"]["Policies"][0][
        "PolicyDocument"
    ]["Statement"]
    reads = next(item for item in statements if item["Sid"] == "ReadMustStartAuthority")
    prefix = (
        "!Sub ${ModelBucket.Arn}/campaigns/${SkyCampaignRunId}/monitor/"
        "must-start/${SkyMustStartManagedMode}/"
        "${SkyMustStartSubmissionBodySha256}/"
    )
    assert reads == {
        "Sid": "ReadMustStartAuthority",
        "Effect": "Allow",
        "Action": "s3:GetObject",
        "Resource": [
            (
                "!Sub ${ModelBucket.Arn}/campaigns/${SkyCampaignRunId}/"
                "${SkyMustStartDescriptorRelativeKey}"
            ),
            (
                "!Sub ${ModelBucket.Arn}/campaigns/${SkyCampaignRunId}/"
                "monitor/submission-locks/"
                "${SkyMustStartDescriptorFileSha256}-"
                "${SkyMustStartManagedMode}.json"
            ),
            prefix + "TIMELY_START_LATCH.json",
            prefix + "CANCEL_REQUESTED.json",
            prefix + "CANCEL_COMPLETED.json",
            prefix + "TARGET_BOUND.json",
            prefix + "JOB_BINDING.json",
            prefix + "TIMELY_START_ACCEPTED.json",
            prefix + "CANCEL_SUPERSEDED.json",
            prefix + "REQUEST_ALERTED.json",
            prefix + "COMPLETION_ALERTED.json",
            prefix + "observations/*",
            prefix + "worker-latches/*",
        ],
    }
    evidence_list = next(
        item
        for item in statements
        if item["Sid"] == "ListExactMustStartEvidenceNamespace"
    )
    assert evidence_list == {
        "Sid": "ListExactMustStartEvidenceNamespace",
        "Effect": "Allow",
        "Action": "s3:ListBucket",
        "Resource": "!GetAtt ModelBucket.Arn",
        "Condition": {
            "StringLike": {
                "s3:prefix": (
                    "!Sub campaigns/${SkyCampaignRunId}/monitor/"
                    "must-start/${SkyMustStartManagedMode}/"
                    "${SkyMustStartSubmissionBodySha256}/*"
                )
            }
        },
    }
    assert [
        item
        for item in statements
        if "s3:ListBucket"
        in (
            item["Action"]
            if isinstance(item["Action"], list)
            else [item["Action"]]
        )
    ] == [evidence_list]
    writes = next(item for item in statements if item["Sid"] == "WriteMustStartMarkers")
    assert writes["Action"] == "s3:PutObject"
    assert writes["Resource"] == [
        prefix + "CANCEL_REQUESTED.json",
        prefix + "CANCEL_COMPLETED.json",
        prefix + "TARGET_BOUND.json",
        prefix + "JOB_BINDING.json",
        prefix + "TIMELY_START_ACCEPTED.json",
        prefix + "CANCEL_SUPERSEDED.json",
        prefix + "REQUEST_ALERTED.json",
        prefix + "COMPLETION_ALERTED.json",
        prefix + "observations/*",
    ]
    cancel = next(
        item
        for item in statements
        if item["Sid"] == "CancelOnlyThroughTaggedSkyJobsController"
    )
    assert cancel["Condition"]["StringLike"] == {
        "ssm:resourceTag/ray-cluster-name": (
            "!Ref SkyMustStartControllerClusterName"
        )
    }
    assert cancel["Resource"].endswith(
        "instance/${SkyMustStartControllerInstanceId}"
    )
    allowed_actions = {
        action
        for statement in statements
        if statement.get("Effect") == "Allow"
        for action in (
            statement["Action"]
            if isinstance(statement["Action"], list)
            else [statement["Action"]]
        )
    }
    forbidden = {
        "ec2:TerminateInstances",
        "ec2:RunInstances",
        "ec2:StartInstances",
        "ec2:StopInstances",
        "ec2:CreateTags",
        "iam:PassRole",
        "ec2:CreateFleet",
        "ec2:RequestSpotInstances",
        "ec2:PurchaseCapacityBlock",
    }
    assert allowed_actions.isdisjoint(forbidden)
    assert {
        "ec2:DescribeInstances",
        "ssm:SendCommand",
        "ssm:GetCommandInvocation",
        "ssm:DescribeInstanceInformation",
        "s3:GetObject",
        "s3:ListBucket",
        "s3:PutObject",
        "sns:Publish",
    }.issubset(allowed_actions)


def test_must_start_observe_only_exact_controller_and_deadline_scheduler() -> None:
    template = _template()
    parameters = template["Parameters"]
    resources = template["Resources"]
    assert parameters["EnableSkyMustStartObserve"]["Default"] == "false"
    assert parameters["SkyMustStartWorkspace"]["Default"] == "default"
    assert parameters["SkyMustStartStartingGraceSeconds"] == {
        "Type": "Number",
        "Default": 20,
        "MinValue": 15,
        "MaxValue": 30,
    }
    variables = resources["SkyMustStartCancelFunction"]["Properties"][
        "Environment"
    ]["Variables"]
    assert variables["OBSERVE_ONLY"] == [
        "SkyMustStartCancelEnabled",
        "false",
        "true",
    ]
    assert variables["EXPECTED_WORKSPACE"] == "!Ref SkyMustStartWorkspace"
    assert variables["EXPECTED_DESCRIPTOR_FILE_SHA256"] == (
        "!Ref SkyMustStartDescriptorFileSha256"
    )
    assert variables["EXPECTED_SUBMISSION_SUBMITTED_AT"] == (
        "!Ref SkyMustStartSubmissionSubmittedAt"
    )
    assert variables["EXPECTED_CONTROLLER_INSTANCE_ID"] == (
        "!Ref SkyMustStartControllerInstanceId"
    )
    assert variables["EXPECTED_CONTROLLER_INSTANCE_TYPE"] == (
        "!Ref SkyMustStartControllerInstanceType"
    )
    assert variables["EXPECTED_CONTROLLER_PROFILE_ARN"] == (
        "!Ref SkyMustStartControllerProfileArn"
    )
    assert variables["EXPECTED_CONTROLLER_CLUSTER_NAME"] == (
        "!Ref SkyMustStartControllerClusterName"
    )
    assert variables["STARTING_GRACE_SECONDS"] == (
        "!Ref SkyMustStartStartingGraceSeconds"
    )

    schedule = resources["SkyMustStartDeadlineSchedule"]
    assert schedule["Type"] == "AWS::Scheduler::Schedule"
    assert schedule["Condition"] == "SkyMustStartCoordinatorEnabled"
    assert schedule["Properties"]["ScheduleExpression"] == [
        "",
        [
            "at(",
            [0, ["Z", "!Ref SkyMustStartPrimaryWakeAt"]],
            ")",
        ],
    ]
    assert schedule["Properties"]["FlexibleTimeWindow"] == {"Mode": "OFF"}
    assert "ActionAfterCompletion" not in schedule["Properties"]
    target = schedule["Properties"]["Target"]
    assert target["Arn"] == "!GetAtt SkyMustStartCancelFunction.Arn"
    assert target["RoleArn"] == "!GetAtt SkyMustStartSchedulerRole.Arn"
    primary_input = (
        '!Sub {"campaign_run_id":"${SkyCampaignRunId}",'
        '"managed_mode":"${SkyMustStartManagedMode}",'
        '"sky_job_name":"${SkyMustStartJobName}",'
        '"must_start_by":"${SkyMustStartBy}",'
        '"trigger":"primary-deadline"}'
    )
    reconciliation_input = (
        '!Sub {"campaign_run_id":"${SkyCampaignRunId}",'
        '"managed_mode":"${SkyMustStartManagedMode}",'
        '"sky_job_name":"${SkyMustStartJobName}",'
        '"must_start_by":"${SkyMustStartBy}",'
        '"trigger":"reconcile"}'
    )
    assert target["Input"] == primary_input
    assert resources["SkyMustStartCancelRule"]["Properties"]["Targets"][0][
        "Input"
    ] == reconciliation_input


def test_must_start_primary_wake_and_lifecycle_shutdown_are_narrowly_scoped() -> None:
    template = _template()
    parameters = template["Parameters"]
    resources = template["Resources"]
    assert parameters["SkyMustStartPrimaryWakeAt"]["AllowedPattern"] == (
        "^[0-9]{4}-[0-9]{2}-[0-9]{2}T"
        "[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
    )
    function = resources["SkyMustStartCancelFunction"]["Properties"]
    assert function["Timeout"] >= 180
    assert function["Environment"]["Variables"][
        "PRIMARY_WAKE_MAX_WAIT_SECONDS"
    ] == "120"
    assert function["Environment"]["Variables"]["RECONCILIATION_RULE_NAME"] == (
        "!Sub ${ProjectTag}-sky-must-start-cancel"
    )

    coordinator_statements = resources["SkyMustStartCancelRole"]["Properties"][
        "Policies"
    ][0]["PolicyDocument"]["Statement"]
    disable = next(
        statement
        for statement in coordinator_statements
        if statement.get("Sid") == "DisableExactMustStartReconciliation"
    )
    assert disable == {
        "Sid": "DisableExactMustStartReconciliation",
        "Effect": "Allow",
        "Action": "events:DisableRule",
        "Resource": (
            "!Sub arn:${AWS::Partition}:events:${AWS::Region}:"
            "${AWS::AccountId}:rule/${ProjectTag}-sky-must-start-cancel"
        ),
    }

    scheduler_role = resources["SkyMustStartSchedulerRole"]["Properties"]
    trust = scheduler_role["AssumeRolePolicyDocument"]["Statement"][0]
    assert trust["Condition"] == {
        "StringEquals": {"aws:SourceAccount": "!Ref AWS::AccountId"},
        "ArnEquals": {
            "aws:SourceArn": (
                "!Sub arn:${AWS::Partition}:scheduler:${AWS::Region}:"
                "${AWS::AccountId}:schedule-group/default"
            )
        },
    }
    scheduler_statements = scheduler_role["Policies"][0]["PolicyDocument"][
        "Statement"
    ]
    assert scheduler_statements == [
        {
            "Sid": "InvokeExactMustStartFunction",
            "Effect": "Allow",
            "Action": "lambda:InvokeFunction",
            "Resource": "!GetAtt SkyMustStartCancelFunction.Arn",
        },
        {
            "Sid": "SendExactMustStartDeadLetter",
            "Effect": "Allow",
            "Action": "sqs:SendMessage",
            "Resource": "!GetAtt SkyMustStartCancelDeadLetterQueue.Arn",
        },
    ]

    hardener = resources["SkyMustStartDeadlineScheduleAutoDelete"]
    assert hardener["Type"] == "Custom::SchedulerActionAfterCompletion"
    assert hardener["Condition"] == "SkyMustStartCoordinatorEnabled"
    assert hardener["Properties"]["ActionAfterCompletion"] == "DELETE"
    assert hardener["Properties"]["ScheduleName"] == (
        "!Ref SkyMustStartDeadlineSchedule"
    )
    hardener_statements = resources[
        "SkyMustStartScheduleAutoDeleteRole"
    ]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
    assert hardener_statements == [
        {
            "Sid": "UpdateExactMustStartSchedule",
            "Effect": "Allow",
            "Action": "scheduler:UpdateSchedule",
            "Resource": (
                "!Sub arn:${AWS::Partition}:scheduler:${AWS::Region}:"
                "${AWS::AccountId}:schedule/default/"
                "${ProjectTag}-sky-must-start-deadline"
            ),
        },
        {
            "Sid": "PassExactSchedulerExecutionRole",
            "Effect": "Allow",
            "Action": "iam:PassRole",
            "Resource": "!GetAtt SkyMustStartSchedulerRole.Arn",
            "Condition": {
                "StringEquals": {
                    "iam:PassedToService": "scheduler.amazonaws.com"
                }
            },
        },
    ]


def test_raw_must_start_enable_requires_complete_exact_controller_authority() -> None:
    template = _template()
    rules = template["Rules"]
    rule = rules["SkyMustStartCoordinatorRequiresExactAuthority"]
    assert rule["RuleCondition"] == [
        ["!Ref EnableSkyMustStartObserve", "true"],
        ["!Ref EnableSkyMustStartCancel", "true"],
    ]
    expected_nondefaults = {
        ("SkyCampaignRunId", "disabled"),
        ("SkyMustStartFoundationRevision", "disabled"),
        ("SkyMustStartCancelCodeSha256", "disabled"),
        ("SkyMustStartCancelCodeVersionId", "disabled"),
        ("SkyMustStartDescriptorRelativeKey", "disabled"),
        ("SkyMustStartSubmissionBodySha256", "disabled"),
        ("SkyMustStartControllerBaselineBodySha256", "disabled"),
        ("SkyMustStartTargetJobId", "-1"),
        ("SkyMustStartJobName", "disabled-cache-seed"),
        ("SkyMustStartBy", "1970-01-01T00:00:00Z"),
        ("SkyMustStartPrimaryWakeAt", "1970-01-01T00:00:00Z"),
        ("SkyMustStartDescriptorFileSha256", "disabled"),
        ("SkyMustStartSubmissionSubmittedAt", "1970-01-01T00:00:00Z"),
        ("SkyMustStartControllerInstanceId", "disabled"),
        ("SkyMustStartControllerInstanceType", "disabled"),
        ("SkyMustStartControllerProfileArn", "disabled"),
        ("SkyMustStartControllerClusterName", "disabled"),
    }
    actual_nondefaults = {
        tuple(assertion["Assert"][0])
        for assertion in rule["Assertions"]
    }
    assert actual_nondefaults == {
        (f"!Ref {parameter}", default)
        for parameter, default in expected_nondefaults
    }


def test_must_start_lambda_code_is_bucket_versioned_and_exact_version_pinned() -> None:
    template = _template()
    parameters = template["Parameters"]
    resources = template["Resources"]

    assert resources["ModelBucket"]["Properties"]["VersioningConfiguration"] == {
        "Status": "Enabled"
    }
    version = parameters["SkyMustStartCancelCodeVersionId"]
    assert version["Default"] == "disabled"
    pattern = re.compile(version["AllowedPattern"])
    assert pattern.fullmatch("disabled")
    assert pattern.fullmatch("3/L4kqtJlcpXroDTDmJ+rmSpXd3dIbrHY+MTRCxf3vjVBH40=")
    for invalid in (
        "",
        "*",
        "../version",
        "version/../id",
        " version",
        "null",
        "None",
    ):
        assert pattern.fullmatch(invalid) is None
    revision = parameters["SkyMustStartFoundationRevision"]
    assert revision["Type"] == "String"
    assert revision["Default"] == "disabled"
    assert revision["AllowedValues"] == ["disabled", "versioned-code-v1"]
    assert resources["SkyMustStartCancelFunction"]["Properties"]["Code"] == {
        "S3Bucket": "!Ref ModelBucket",
        "S3Key": (
            "!Sub lambda/sky-must-start-cancel/"
            "${SkyMustStartCancelCodeSha256}.zip"
        ),
        "S3ObjectVersion": "!Ref SkyMustStartCancelCodeVersionId",
    }


def test_must_start_iam_inputs_and_code_artifact_are_strictly_scoped() -> None:
    template = _template()
    parameters = template["Parameters"]
    run_id = "glm52-sky-20260724"

    accepted = {
        "ProjectTag": "keep-glm52",
        "OwnerTag": "jack.mazac",
        "SkyCampaignRunId": run_id,
        "SkyCampaignDescriptorKey": (
            f"campaigns/{run_id}/submissions/seed/"
            "campaign-descriptor-v2.json"
        ),
        "SkyMustStartDescriptorRelativeKey": (
            "submissions/seed/campaign-descriptor-v2.json"
        ),
        "SkyMustStartSubmissionBodySha256": "b" * 64,
        "SkyMustStartControllerBaselineBodySha256": "f" * 64,
        "SkyMustStartDescriptorFileSha256": "a" * 64,
        "SkyMustStartCancelCodeSha256": "c" * 64,
        "SkyMustStartCancelCodeVersionId": (
            "3/L4kqtJlcpXroDTDmJ+rmSpXd3dIbrHY+MTRCxf3vjVBH40="
        ),
        "SkyMustStartActivationJobBindingSha256": "d" * 64,
        "SkyMustStartActivationObservationSha256": "e" * 64,
        "SkyMustStartJobName": f"{run_id}-cache-seed",
        "SkyMustStartControllerInstanceType": "c6a.xlarge",
        "SkyMustStartControllerProfileArn": (
            "arn:aws:iam::246813579024:instance-profile/"
            "keep-glm52-skypilot-controller"
        ),
        "SkyMustStartControllerClusterName": (
            "sky-jobs-controller-9d9f31a9-9d9f31a9"
        ),
    }
    for name, value in accepted.items():
        pattern = re.compile(parameters[name]["AllowedPattern"])
        assert pattern.fullmatch(value), name

    malicious = {
        "ProjectTag": ["*", "keep?glm52", "../keep"],
        "OwnerTag": ["*", "jack?mazac", "../jack"],
        "SkyCampaignRunId": ["*", "run?", "../run", "run/other"],
        "SkyCampaignDescriptorKey": [
            "campaigns/*/submissions/seed/descriptor.json",
            "campaigns/run?/submissions/seed/descriptor.json",
            "campaigns/run/submissions/../descriptor.json",
        ],
        "SkyMustStartDescriptorRelativeKey": [
            "*",
            "submissions/seed/descriptor?.json",
            "submissions/../descriptor.json",
            "campaigns/other/submissions/descriptor.json",
        ],
        "SkyMustStartSubmissionBodySha256": ["*", "B" * 64, "b" * 63],
        "SkyMustStartControllerBaselineBodySha256": [
            "*",
            "F" * 64,
            "f" * 63,
        ],
        "SkyMustStartDescriptorFileSha256": ["*", "A" * 64, "a" * 65],
        "SkyMustStartCancelCodeSha256": ["*", "C" * 64, "c" * 63],
        "SkyMustStartCancelCodeVersionId": [
            "*",
            "../version",
            "version/../id",
            " version",
            "null",
            "None",
        ],
        "SkyMustStartActivationJobBindingSha256": [
            "*",
            "D" * 64,
            "d" * 63,
        ],
        "SkyMustStartActivationObservationSha256": [
            "*",
            "E" * 64,
            "e" * 63,
        ],
        "SkyMustStartJobName": ["*", "job?", "../job"],
        "SkyMustStartControllerInstanceType": ["*", "c6a?xlarge", "../c6a"],
        "SkyMustStartControllerProfileArn": [
            "*",
            "arn:aws:iam::*:instance-profile/controller",
            "arn:aws:iam::246813579024:instance-profile/../controller",
        ],
        "SkyMustStartControllerClusterName": ["*", "cluster?", "../cluster"],
    }
    for name, values in malicious.items():
        pattern = re.compile(parameters[name]["AllowedPattern"])
        for value in values:
            assert pattern.fullmatch(value) is None, (name, value)

    assert "SkyMustStartCancelCodeS3Key" not in parameters
    assert "SkyMustStartSubmissionKey" not in parameters
    assert template["Resources"]["SkyMustStartCancelFunction"]["Properties"][
        "Code"
    ] == {
        "S3Bucket": "!Ref ModelBucket",
        "S3Key": (
            "!Sub lambda/sky-must-start-cancel/"
            "${SkyMustStartCancelCodeSha256}.zip"
        ),
        "S3ObjectVersion": "!Ref SkyMustStartCancelCodeVersionId",
    }


def test_must_start_alarms_cover_lambda_age_events_and_dlq() -> None:
    resources = _template()["Resources"]
    for name in (
        "SkyMustStartCancelErrorsAlarm",
        "SkyMustStartCancelAgeAlarm",
        "SkyMustStartCancelFailedInvocationsAlarm",
        "SkyMustStartCancelDlqAlarm",
    ):
        assert resources[name]["Properties"]["AlarmActions"] == [
            "!Ref CampaignAlertTopic"
        ]


def test_cost_backstops_use_tag_budget_and_member_account_service_monitor() -> None:
    resources = _template()["Resources"]
    budget = resources["SkyGpuBudget"]["Properties"]["Budget"]
    assert budget["BudgetLimit"] == {"Amount": 1320.96, "Unit": "USD"}
    assert budget["CostFilters"]["TagKeyValue"] == [
        "user:cost-allocation$glm52-sky-campaign"
    ]
    monitor = resources["SkyCostAnomalyMonitor"]["Properties"]
    assert monitor["MonitorType"] == "DIMENSIONAL"
    assert monitor["MonitorDimension"] == "SERVICE"
    assert "MonitorSpecification" not in monitor
    assert (
        resources["SkyCostAnomalySubscription"]["Properties"]["Subscribers"][0][
            "Address"
        ]
        == "!Ref CampaignAlertEmail"
    )


def test_s3_batch_checksum_role_is_least_privilege_and_always_available() -> None:
    template = _template()
    resources = template["Resources"]
    role = resources["S3BatchChecksumRole"]
    assert "Condition" not in role
    properties = role["Properties"]
    assert properties["RoleName"] == "keep-glm52-s3-checksum-auditor"
    trust = properties["AssumeRolePolicyDocument"]["Statement"]
    assert trust == [
        {
            "Effect": "Allow",
            "Principal": {"Service": "batchoperations.s3.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }
    ]
    statements = properties["Policies"][0]["PolicyDocument"]["Statement"]
    read = next(item for item in statements if item["Sid"] == "ReadAuditInputs")
    assert set(read["Action"]) == {
        "s3:GetObject",
        "s3:GetObjectVersion",
        "s3:RestoreObject",
    }
    assert read["Resource"] == ["!Sub ${ModelBucket.Arn}/*"]
    report = next(item for item in statements if item["Sid"] == "WriteAuditReports")
    assert report["Action"] == "s3:PutObject"
    assert report["Resource"] == [
        "!Sub ${ModelBucket.Arn}/campaign-audits/*",
    ]
    assert template["Outputs"]["S3BatchChecksumRoleArn"]["Value"] == (
        "!GetAtt S3BatchChecksumRole.Arn"
    )


def test_controller_baseline_digest_is_plumbed_through_guarded_deploy() -> None:
    script = DEPLOY_SCRIPT.read_text()
    assert (
        ': "${MUST_START_CONTROLLER_BASELINE_BODY_SHA256:'
        '?set MUST_START_CONTROLLER_BASELINE_BODY_SHA256}"'
    ) in script
    assert (
        "require_sha256 \\\n"
        "    MUST_START_CONTROLLER_BASELINE_BODY_SHA256 \\\n"
        '    "$MUST_START_CONTROLLER_BASELINE_BODY_SHA256"'
    ) in script
    assert (
        '"SkyMustStartControllerBaselineBodySha256='
        '$MUST_START_CONTROLLER_BASELINE_BODY_SHA256"'
    ) in script
    assert (
        "ParameterKey=='SkyMustStartControllerBaselineBodySha256'"
    ) in script
    assert (
        '"$MUST_START_CONTROLLER_BASELINE_BODY_SHA256" \\'
    ) in script
