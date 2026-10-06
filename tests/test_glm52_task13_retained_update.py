from __future__ import annotations

import base64
import importlib.util
import json
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from urllib.parse import parse_qs, urlparse

import pytest

from glm52_enforcement import task13_retained_update as retained_update
from glm52_enforcement.task13_retained_update import (
    ACCOUNT_ID,
    DEPLOYMENT_ROLE_ARN,
    PROFILE,
    REGION,
    RETAINED_MODELS_BUCKET,
    RETAINED_STACK_NAME,
    RetainedUpdateError,
    RetainedUpdateServices,
    apply_retained_update,
    validate_retained_update_change_set,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "aws/glm52-gpu/scripts/apply_glm52_task13_retained_update.py"
STACK_ID = (
    "arn:aws:cloudformation:us-west-2:246813579024:"
    "stack/keep-glm52-gpu/11111111-2222-3333-4444-555555555555"
)
CHANGE_SET_ID = (
    "arn:aws:cloudformation:us-west-2:246813579024:"
    "changeSet/glm52-task13-retained-pre-support-v1/"
    "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
)
FINAL_CHANGE_SET_ID = (
    "arn:aws:cloudformation:us-west-2:246813579024:"
    "changeSet/glm52-task13-retained-final-v1/"
    "ffffffff-bbbb-cccc-dddd-eeeeeeeeeeee"
)


def _base_template() -> dict[str, object]:
    return {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Description": "retained base",
        "Parameters": {
            "LegacyMode": {
                "Type": "String",
                "Default": "disabled",
            }
        },
        "Resources": {
            "Vpc": {
                "Type": "AWS::EC2::VPC",
                "Properties": {"CidrBlock": "10.0.0.0/16"},
            },
            "ModelBucket": {
                "Type": "AWS::S3::Bucket",
                "Properties": {
                    "BucketName": RETAINED_MODELS_BUCKET,
                    "VersioningConfiguration": {"Status": "Enabled"},
                },
            },
        },
        "Outputs": {
            "VpcId": {
                "Value": {"Ref": "Vpc"},
                "Export": {"Name": "KeepGlm52VpcId"},
            },
            "ModelBucketName": {
                "Value": {"Ref": "ModelBucket"},
                "Export": {"Name": "KeepGlm52ModelBucket"},
            },
        },
    }


def test_template_body_accepts_botocore_ordered_mapping() -> None:
    value = OrderedDict(
        [
            ("AWSTemplateFormatVersion", "2010-09-09"),
            ("Resources", OrderedDict([("Vpc", {"Type": "AWS::EC2::VPC"})])),
        ]
    )

    assert retained_update._template_body(value) == {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Resources": {"Vpc": {"Type": "AWS::EC2::VPC"}},
    }


def _fragment() -> dict[str, object]:
    return {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Parameters": {},
        "Resources": {
            "RetainedQueue": {
                "Type": "AWS::SQS::Queue",
                "DeletionPolicy": "Retain",
                "UpdateReplacePolicy": "Retain",
                "Properties": {
                    "QueueName": "keep-glm52-h1g-retained-test",
                },
            }
        },
        "Outputs": {
            "RetainedQueueArn": {
                "Value": {"Fn::GetAtt": ["RetainedQueue", "Arn"]},
                "Export": {"Name": "KeepGlm52RetainedQueueArn"},
            }
        },
    }


def _before_outputs() -> list[dict[str, object]]:
    return [
        {
            "OutputKey": "VpcId",
            "OutputValue": "vpc-0123456789abcdef0",
            "ExportName": "KeepGlm52VpcId",
        },
        {
            "OutputKey": "ModelBucketName",
            "OutputValue": RETAINED_MODELS_BUCKET,
            "ExportName": "KeepGlm52ModelBucket",
        },
    ]


def _after_outputs() -> list[dict[str, object]]:
    return [
        *_before_outputs(),
        {
            "OutputKey": "RetainedQueueArn",
            "OutputValue": (
                "arn:aws:sqs:us-west-2:246813579024:keep-glm52-h1g-retained-test"
            ),
            "ExportName": "KeepGlm52RetainedQueueArn",
        },
    ]


def _stack(
    *,
    status: str,
    outputs: list[dict[str, object]],
    role_arn: str | None,
    parameters: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "StackId": STACK_ID,
        "StackName": RETAINED_STACK_NAME,
        "StackStatus": status,
        "EnableTerminationProtection": False,
        "Parameters": (
            [
                {
                    "ParameterKey": "LegacyMode",
                    "ParameterValue": "disabled",
                }
            ]
            if parameters is None
            else deepcopy(parameters)
        ),
        "Outputs": outputs,
        "Tags": [{"Key": "Project", "Value": "KEEP"}],
    }
    if role_arn is not None:
        value["RoleARN"] = role_arn

    return value


def test_stack_identity_guard_ignores_only_readback_order() -> None:
    original = _stack(
        status="UPDATE_COMPLETE",
        outputs=_after_outputs(),
        role_arn=None,
        parameters=_intentional_stack_parameters(),
    )
    reordered = deepcopy(original)
    reordered["Parameters"].reverse()
    reordered["Outputs"].reverse()
    reordered["Tags"].reverse()

    assert retained_update._initial_stack_guard(
        reordered
    ) == retained_update._initial_stack_guard(original)

    reordered["Outputs"][0]["OutputValue"] = "foreign"
    assert retained_update._initial_stack_guard(
        reordered
    ) != retained_update._initial_stack_guard(original)


class _Body:
    def __init__(self, raw: bytes):
        self._raw = raw

    def read(self) -> bytes:
        return self._raw

    def close(self) -> None:
        return None


class _Sts:
    def get_caller_identity(self) -> dict[str, object]:
        return {
            "Account": ACCOUNT_ID,
            "Arn": f"arn:aws:sts::{ACCOUNT_ID}:assumed-role/operator/session",
            "UserId": "AROATEST:session",
        }


def _cancel_policy_document() -> dict[str, object]:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "events:DisableRule",
                "Resource": (
                    "arn:aws:events:us-west-2:246813579024:rule/keep-glm52-cancel"
                ),
            }
        ],
    }


def _controller_policy_document() -> dict[str, object]:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "ec2:DescribeInstances",
                "Resource": "*",
                "Condition": {
                    "StringEquals": {
                        "aws:RequestTag/project": "keep-glm52",
                    }
                },
            }
        ],
    }


def _launch_template_data() -> dict[str, object]:
    return {
        "ImageId": "ami-02b19745d2b303803",
        "BlockDeviceMappings": [
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "VolumeSize": 300,
                    "VolumeType": "gp3",
                },
            }
        ],
        "UserData": base64.b64encode(b"project=keep-glm52").decode("ascii"),
    }


class _Iam:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []
        self.policy_documents = {
            ("keep-glm52-cancel-role", "sky-must-start-cancel-only"): (
                _cancel_policy_document()
            ),
            ("keep-glm52-controller-role", "sky-controller-one-p5"): (
                _controller_policy_document()
            ),
        }

    def get_role(self, **request: object) -> dict[str, object]:
        self.requests.append(dict(request))
        assert request == {"RoleName": DEPLOYMENT_ROLE_ARN.rpartition("/")[2]}
        return {
            "Role": {
                "Arn": DEPLOYMENT_ROLE_ARN,
                "RoleName": DEPLOYMENT_ROLE_ARN.rpartition("/")[2],
                "RoleId": "AROARETAINEDDEPLOYMENT",
                "AssumeRolePolicyDocument": {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "cloudformation.amazonaws.com"},
                            "Action": "sts:AssumeRole",
                        }
                    ],
                },
            }
        }

    def get_role_policy(self, **request: object) -> dict[str, object]:
        self.requests.append(dict(request))
        key = (request.get("RoleName"), request.get("PolicyName"))
        assert key in self.policy_documents
        return {
            "RoleName": key[0],
            "PolicyName": key[1],
            "PolicyDocument": deepcopy(self.policy_documents[key]),
        }


class _Ec2:
    def __init__(self) -> None:
        self.version_number = 7
        self.launch_template_data = _launch_template_data()

    def describe_launch_template_versions(
        self,
        **request: object,
    ) -> dict[str, object]:
        assert request == {
            "LaunchTemplateId": "lt-0123456789abcdef0",
            "Versions": ["$Latest"],
        }
        return {
            "LaunchTemplateVersions": [
                {
                    "LaunchTemplateId": "lt-0123456789abcdef0",
                    "VersionNumber": self.version_number,
                    "DefaultVersion": False,
                    "LaunchTemplateData": deepcopy(self.launch_template_data),
                }
            ]
        }


class _Ssm:
    def __init__(
        self,
        *,
        value: str = "ami-0f965e1c95dfa5a23",
        version: int = 174,
    ) -> None:
        self.value = value
        self.version = version

    def get_parameter(self, **request: object) -> dict[str, object]:
        assert request == {
            "Name": (
                "/aws/service/deeplearning/ami/x86_64/"
                "base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
            ),
            "WithDecryption": False,
        }
        return {
            "Parameter": {
                "Name": request["Name"],
                "Type": "String",
                "Value": self.value,
                "Version": self.version,
            }
        }


class _S3:
    def __init__(self) -> None:
        self.raw: bytes | None = None
        self.metadata: dict[str, str] | None = None
        self.key: str | None = None
        self.put_requests: list[dict[str, object]] = []
        self.version_id = "pre-support-version-1"

    def get_bucket_versioning(self, **request: object) -> dict[str, object]:
        assert request == {
            "Bucket": RETAINED_MODELS_BUCKET,
            "ExpectedBucketOwner": ACCOUNT_ID,
        }
        return {"Status": "Enabled"}

    def list_object_versions(self, **request: object) -> dict[str, object]:
        assert request["Bucket"] == RETAINED_MODELS_BUCKET
        assert request["ExpectedBucketOwner"] == ACCOUNT_ID
        assert request["Prefix"] == request.get("Prefix")
        versions: list[dict[str, object]] = []
        if self.raw is not None:
            versions = [
                {
                    "Key": self.key,
                    "VersionId": self.version_id,
                    "Size": len(self.raw),
                    "IsLatest": True,
                }
            ]
        return {
            "Versions": versions,
            "DeleteMarkers": [],
            "IsTruncated": False,
        }

    def put_object(self, **request: object) -> dict[str, object]:
        self.put_requests.append(dict(request))
        assert request["IfNoneMatch"] == "*"
        assert request["ExpectedBucketOwner"] == ACCOUNT_ID
        assert request["ChecksumAlgorithm"] == "SHA256"
        self.raw = request["Body"]  # type: ignore[assignment]
        self.metadata = dict(request["Metadata"])  # type: ignore[arg-type]
        self.key = str(request["Key"])
        return {
            "VersionId": self.version_id,
            "ChecksumSHA256": request["ChecksumSHA256"],
        }

    def get_object(self, **request: object) -> dict[str, object]:
        assert request == {
            "Bucket": RETAINED_MODELS_BUCKET,
            "Key": self.key,
            "VersionId": self.version_id,
            "ExpectedBucketOwner": ACCOUNT_ID,
            "ChecksumMode": "ENABLED",
        }
        assert self.raw is not None
        return {
            "Body": _Body(self.raw),
            "ContentLength": len(self.raw),
            "VersionId": self.version_id,
            "ChecksumSHA256": self.put_requests[-1]["ChecksumSHA256"],
            "Metadata": self.metadata,
        }


class _CloudFormation:
    def __init__(
        self,
        *,
        ambiguous_create: bool = False,
        ambiguous_execute: bool = False,
        replacement: str | None = None,
        action: str = "Add",
        parameters: list[dict[str, object]] | None = None,
        phase: str = "pre-support",
        legacy_wildcard_action: bool = False,
    ) -> None:
        assert phase in {"pre-support", "final"}
        self.before_template = _base_template()
        self.after_template = deepcopy(self.before_template)
        after_resources = self.after_template["Resources"]
        after_outputs = self.after_template["Outputs"]
        fragment = _fragment()
        assert isinstance(after_resources, dict)
        assert isinstance(after_outputs, dict)
        assert isinstance(fragment["Resources"], dict)
        assert isinstance(fragment["Outputs"], dict)
        after_resources["RetainedQueue"] = deepcopy(
            fragment["Resources"]["RetainedQueue"]
        )
        after_outputs["RetainedQueueArn"] = deepcopy(
            fragment["Outputs"]["RetainedQueueArn"]
        )
        if legacy_wildcard_action:
            legacy_role = {
                "Type": "AWS::IAM::Role",
                "Properties": {
                    "AssumeRolePolicyDocument": {
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {"Service": "lambda.amazonaws.com"},
                                "Action": "sts:AssumeRole",
                            }
                        ]
                    },
                    "Policies": [
                        {
                            "PolicyName": "legacy-wildcard",
                            "PolicyDocument": {
                                "Statement": [
                                    {
                                        "Effect": "Allow",
                                        "Action": "iam:*",
                                        "Resource": "*",
                                    }
                                ]
                            },
                        }
                    ],
                },
            }
            self.before_template["Resources"]["LegacyRole"] = deepcopy(legacy_role)
            self.after_template["Resources"]["LegacyRole"] = deepcopy(legacy_role)
        self.ambiguous_create = ambiguous_create
        self.ambiguous_execute = ambiguous_execute
        self.replacement = replacement
        self.action = action
        self.parameters = parameters
        self.change_set_name = (
            "glm52-task13-retained-pre-support-v1"
            if phase == "pre-support"
            else "glm52-task13-retained-final-v1"
        )
        self.change_set_id = (
            CHANGE_SET_ID if phase == "pre-support" else FINAL_CHANGE_SET_ID
        )
        self.expected_template_path = (
            "/task13/templates/retained-pre-support.yaml"
            if phase == "pre-support"
            else "/task13/templates/retained.yaml"
        )
        self.created = False
        self.executed = False
        self.create_requests: list[dict[str, object]] = []
        self.execute_requests: list[dict[str, object]] = []
        self.get_template_requests: list[dict[str, object]] = []

    def describe_stacks(self, **request: object) -> dict[str, object]:
        assert request["StackName"] in {RETAINED_STACK_NAME, STACK_ID}
        if self.executed:
            return {
                "Stacks": [
                    _stack(
                        status="UPDATE_COMPLETE",
                        outputs=_after_outputs(),
                        role_arn=DEPLOYMENT_ROLE_ARN,
                        parameters=self.parameters,
                    )
                ]
            }
        return {
            "Stacks": [
                _stack(
                    status="UPDATE_COMPLETE",
                    outputs=_before_outputs(),
                    role_arn=None,
                    parameters=self.parameters,
                )
            ]
        }

    def get_template(self, **request: object) -> dict[str, object]:
        self.get_template_requests.append(dict(request))
        assert request["TemplateStage"] == "Original"
        if "ChangeSetName" in request:
            assert request["ChangeSetName"] == self.change_set_id
            assert self.after_template is not None
            return {"TemplateBody": self.after_template}
        assert request["StackName"] == STACK_ID
        return {
            "TemplateBody": (
                self.after_template if self.executed else self.before_template
            )
        }

    def create_change_set(self, **request: object) -> dict[str, object]:
        self.create_requests.append(dict(request))
        self.created = True
        parsed = urlparse(str(request["TemplateURL"]))
        assert parsed.path == self.expected_template_path
        assert parse_qs(parsed.query) == {"versionId": ["pre-support-version-1"]}
        if self.ambiguous_create:
            raise TimeoutError("response lost after durable create")
        return {"Id": self.change_set_id, "StackId": STACK_ID}

    def describe_change_set(self, **request: object) -> dict[str, object]:
        assert self.created
        assert request["StackName"] == STACK_ID
        assert request["IncludePropertyValues"] is True
        resource_change = {
            "Action": self.action,
            "LogicalResourceId": "RetainedQueue",
            "ResourceType": "AWS::SQS::Queue",
        }
        if self.replacement is not None:
            resource_change["Replacement"] = self.replacement
        return {
            "StackId": STACK_ID,
            "StackName": RETAINED_STACK_NAME,
            "ChangeSetId": self.change_set_id,
            "ChangeSetName": self.change_set_name,
            "Status": "CREATE_COMPLETE",
            "ExecutionStatus": ("EXECUTE_COMPLETE" if self.executed else "AVAILABLE"),
            "Capabilities": [],
            "Parameters": [
                {
                    "ParameterKey": "LegacyMode",
                    "ParameterValue": "disabled",
                }
            ],
            "RoleARN": DEPLOYMENT_ROLE_ARN,
            "Changes": [
                {
                    "Type": "Resource",
                    "ResourceChange": resource_change,
                }
            ],
        }

    def execute_change_set(self, **request: object) -> dict[str, object]:
        self.execute_requests.append(dict(request))
        self.executed = True
        if self.ambiguous_execute:
            raise TimeoutError("response lost after durable execute")
        return {}

    def list_stack_resources(self, **request: object) -> dict[str, object]:
        assert request == {"StackName": STACK_ID}
        rows = [
            {
                "LogicalResourceId": "Vpc",
                "PhysicalResourceId": "vpc-0123456789abcdef0",
                "ResourceType": "AWS::EC2::VPC",
                "ResourceStatus": "UPDATE_COMPLETE",
            },
            {
                "LogicalResourceId": "ModelBucket",
                "PhysicalResourceId": RETAINED_MODELS_BUCKET,
                "ResourceType": "AWS::S3::Bucket",
                "ResourceStatus": "UPDATE_COMPLETE",
            },
        ]
        if "LegacyRole" in self.before_template["Resources"]:
            rows.append(
                {
                    "LogicalResourceId": "LegacyRole",
                    "PhysicalResourceId": "keep-glm52-legacy-role",
                    "ResourceType": "AWS::IAM::Role",
                    "ResourceStatus": "UPDATE_COMPLETE",
                }
            )
        if self.executed:
            rows.append(
                {
                    "LogicalResourceId": "RetainedQueue",
                    "PhysicalResourceId": (
                        "https://sqs.us-west-2.amazonaws.com/246813579024/"
                        "keep-glm52-h1g-retained-test"
                    ),
                    "ResourceType": "AWS::SQS::Queue",
                    "ResourceStatus": "CREATE_COMPLETE",
                }
            )
        return {"StackResourceSummaries": rows}

    def list_exports(self, **request: object) -> dict[str, object]:
        assert request == {}
        return {
            "Exports": [
                {
                    "Name": "KeepGlm52VpcId",
                    "Value": "vpc-0123456789abcdef0",
                    "ExportingStackId": STACK_ID,
                },
                {
                    "Name": "KeepGlm52ModelBucket",
                    "Value": RETAINED_MODELS_BUCKET,
                    "ExportingStackId": STACK_ID,
                },
                {
                    "Name": "KeepGlm52RetainedQueueArn",
                    "Value": (
                        "arn:aws:sqs:us-west-2:246813579024:"
                        "keep-glm52-h1g-retained-test"
                    ),
                    "ExportingStackId": STACK_ID,
                },
            ]
        }


def _services(
    cloudformation: _CloudFormation | None = None,
) -> RetainedUpdateServices:
    return RetainedUpdateServices(
        sts=_Sts(),
        cloudformation=cloudformation or _CloudFormation(),
        iam=_Iam(),
        s3=_S3(),
        ec2=_Ec2(),
        ssm=_Ssm(),
        total_max_attempts=1,
    )


def test_pre_support_update_is_one_versioned_add_only_transaction(
    tmp_path: Path,
) -> None:
    cloudformation = _CloudFormation()
    services = _services(cloudformation)

    result = apply_retained_update(
        services,
        phase="pre-support",
        fragment=_fragment(),
        output_directory=tmp_path,
        sleep=lambda _seconds: None,
        max_polls=4,
    )

    assert result == {
        "status": "UPDATE_COMPLETE",
        "stack_id": STACK_ID,
        "change_set_id": CHANGE_SET_ID,
        "phase": "pre-support",
        "template_key": "task13/templates/retained-pre-support.yaml",
        "template_version_id": "pre-support-version-1",
    }
    s3 = services.s3
    assert isinstance(s3, _S3)
    assert len(s3.put_requests) == 1
    assert len(cloudformation.create_requests) == 1
    assert len(cloudformation.execute_requests) == 1
    request = cloudformation.create_requests[0]
    assert request["StackName"] == STACK_ID
    assert request["ChangeSetType"] == "UPDATE"
    assert request["RoleARN"] == DEPLOYMENT_ROLE_ARN
    assert request["Parameters"] == [
        {
            "ParameterKey": "LegacyMode",
            "UsePreviousValue": True,
        }
    ]
    assert request["Capabilities"] == []
    assert request["IncludeNestedStacks"] is False
    assert "ImportExistingResources" not in request
    assert "ResourcesToImport" not in request
    parsed = urlparse(str(request["TemplateURL"]))
    assert parsed.path == "/task13/templates/retained-pre-support.yaml"
    assert parse_qs(parsed.query) == {"versionId": ["pre-support-version-1"]}
    assert services.iam.requests == [
        {"RoleName": "keep-glm52-h1g-cloudformation-deployment"}
    ]
    assert any(
        request
        == {
            "StackName": STACK_ID,
            "TemplateStage": "Original",
        }
        for request in cloudformation.get_template_requests
    )
    assert any(
        request
        == {
            "ChangeSetName": CHANGE_SET_ID,
            "TemplateStage": "Original",
        }
        for request in cloudformation.get_template_requests
    )
    expected_files = {
        "retained-pre-support-before-template.json",
        "retained-pre-support-after-template.json",
        "retained-pre-support-change-set.json",
        "retained-pre-support-readback.json",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected_files
    for path in tmp_path.iterdir():
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.read_bytes().endswith(b"\n")
        assert json.loads(path.read_text(encoding="ascii"))


@pytest.mark.parametrize("stage", ["create", "execute"])
def test_ambiguous_mutation_is_adopted_without_resend(
    tmp_path: Path,
    stage: str,
) -> None:
    cloudformation = _CloudFormation(
        ambiguous_create=stage == "create",
        ambiguous_execute=stage == "execute",
    )

    result = apply_retained_update(
        _services(cloudformation),
        phase="pre-support",
        fragment=_fragment(),
        output_directory=tmp_path,
        sleep=lambda _seconds: None,
        max_polls=4,
    )

    assert result["status"] == "UPDATE_COMPLETE"
    assert len(cloudformation.create_requests) == 1
    assert len(cloudformation.execute_requests) == 1


def test_final_phase_uses_only_the_fixed_retained_template_key(
    tmp_path: Path,
) -> None:
    cloudformation = _CloudFormation(phase="final")

    result = apply_retained_update(
        _services(cloudformation),
        phase="final",
        fragment=_fragment(),
        output_directory=tmp_path,
        sleep=lambda _seconds: None,
        max_polls=4,
    )

    assert result["change_set_id"] == FINAL_CHANGE_SET_ID
    assert result["template_key"] == "task13/templates/retained.yaml"
    assert (
        cloudformation.create_requests[0]["ChangeSetName"]
        == "glm52-task13-retained-final-v1"
    )


@pytest.mark.parametrize(
    ("action", "replacement"),
    [("Modify", "False"), ("Remove", "False"), ("Import", "False"), ("Add", "True")],
)
def test_change_set_rejects_non_add_or_replacement(
    action: str,
    replacement: str,
) -> None:
    cloudformation = _CloudFormation(action=action, replacement=replacement)
    cloudformation.created = True
    change_set = cloudformation.describe_change_set(
        ChangeSetName="glm52-task13-retained-pre-support-v1",
        StackName=STACK_ID,
        IncludePropertyValues=True,
    )

    with pytest.raises(RetainedUpdateError):
        validate_retained_update_change_set(
            change_set,
            phase="pre-support",
            stack_id=STACK_ID,
            expected_resources={"RetainedQueue": "AWS::SQS::Queue"},
            expected_parameters=[
                {
                    "ParameterKey": "LegacyMode",
                    "ParameterValue": "disabled",
                }
            ],
        )


def test_change_set_rejects_non_object_input_as_malformed() -> None:
    with pytest.raises(RetainedUpdateError, match="inputs are malformed"):
        validate_retained_update_change_set(
            None,  # type: ignore[arg-type]
            phase="fence-bootstrap-v9",
            stack_id=STACK_ID,
            expected_resources={"RetainedQueue": "AWS::SQS::Queue"},
            expected_parameters=_intentional_stack_parameters(),
        )


@pytest.mark.parametrize(
    "fragment",
    [
        {
            **_fragment(),
            "Parameters": {"Forbidden": {"Type": "String"}},
        },
        {
            **_fragment(),
            "Resources": {
                "BadRole": {
                    "Type": "AWS::IAM::Role",
                    "Properties": {
                        "Policies": [
                            {
                                "PolicyName": "bad",
                                "PolicyDocument": {
                                    "Statement": [
                                        {
                                            "Effect": "Allow",
                                            "Action": "iam:*",
                                            "Resource": "*",
                                        }
                                    ]
                                },
                            }
                        ]
                    },
                }
            },
        },
    ],
)
def test_fragment_parameters_or_wildcard_iam_action_fail_before_mutation(
    tmp_path: Path,
    fragment: dict[str, object],
) -> None:
    services = _services()

    with pytest.raises(RetainedUpdateError):
        apply_retained_update(
            services,
            phase="pre-support",
            fragment=fragment,
            output_directory=tmp_path,
            sleep=lambda _seconds: None,
            max_polls=4,
        )

    assert services.cloudformation.create_requests == []
    assert services.s3.put_requests == []


def test_legacy_wildcard_iam_action_is_preserved_without_blocking_addition(
    tmp_path: Path,
) -> None:
    cloudformation = _CloudFormation(legacy_wildcard_action=True)
    services = _services(cloudformation)

    result = apply_retained_update(
        services,
        phase="pre-support",
        fragment=_fragment(),
        output_directory=tmp_path,
        sleep=lambda _seconds: None,
        max_polls=4,
    )

    assert result["status"] == "UPDATE_COMPLETE"
    assert len(cloudformation.create_requests) == 1
    assert len(cloudformation.execute_requests) == 1


@pytest.mark.parametrize(
    "parameters",
    [
        [],
        [
            {
                "ParameterKey": "LegacyMode",
                "ParameterValue": "disabled",
            },
            {
                "ParameterKey": "Foreign",
                "ParameterValue": "foreign",
            },
        ],
    ],
)
def test_live_parameter_omission_or_addition_fails_before_mutation(
    tmp_path: Path,
    parameters: list[dict[str, object]],
) -> None:
    cloudformation = _CloudFormation(parameters=parameters)
    services = _services(cloudformation)

    with pytest.raises(RetainedUpdateError):
        apply_retained_update(
            services,
            phase="pre-support",
            fragment=_fragment(),
            output_directory=tmp_path,
            sleep=lambda _seconds: None,
            max_polls=4,
        )

    assert cloudformation.create_requests == []
    assert services.s3.put_requests == []


def test_iam_additions_require_only_named_iam_capability() -> None:
    cloudformation = _CloudFormation()
    cloudformation.created = True
    change_set = cloudformation.describe_change_set(
        ChangeSetName="glm52-task13-retained-pre-support-v1",
        StackName=STACK_ID,
        IncludePropertyValues=True,
    )
    change_set["Capabilities"] = ["CAPABILITY_NAMED_IAM"]
    change_set["Changes"] = [
        {
            "Type": "Resource",
            "ResourceChange": {
                "Action": "Add",
                "LogicalResourceId": "RetainedRole",
                "ResourceType": "AWS::IAM::Role",
                "Replacement": "False",
            },
        }
    ]

    assert validate_retained_update_change_set(
        change_set,
        phase="pre-support",
        stack_id=STACK_ID,
        expected_resources={"RetainedRole": "AWS::IAM::Role"},
        expected_parameters=[
            {
                "ParameterKey": "LegacyMode",
                "ParameterValue": "disabled",
            }
        ],
    )["Capabilities"] == ["CAPABILITY_NAMED_IAM"]


def test_change_set_accepts_live_api_omitted_role_arn() -> None:
    cloudformation = _CloudFormation()
    cloudformation.created = True
    change_set = cloudformation.describe_change_set(
        ChangeSetName="glm52-task13-retained-pre-support-v1",
        StackName=STACK_ID,
        IncludePropertyValues=True,
    )
    change_set.pop("RoleARN")

    assert (
        validate_retained_update_change_set(
            change_set,
            phase="pre-support",
            stack_id=STACK_ID,
            expected_resources={"RetainedQueue": "AWS::SQS::Queue"},
            expected_parameters=[
                {
                    "ParameterKey": "LegacyMode",
                    "ParameterValue": "disabled",
                }
            ],
        )["ChangeSetId"]
        == CHANGE_SET_ID
    )


def test_change_set_rejects_foreign_role_arn() -> None:
    cloudformation = _CloudFormation()
    cloudformation.created = True
    change_set = cloudformation.describe_change_set(
        ChangeSetName="glm52-task13-retained-pre-support-v1",
        StackName=STACK_ID,
        IncludePropertyValues=True,
    )
    change_set["RoleARN"] = "arn:aws:iam::246813579024:role/foreign-cloudformation-role"

    with pytest.raises(RetainedUpdateError, match="identity or authority"):
        validate_retained_update_change_set(
            change_set,
            phase="pre-support",
            stack_id=STACK_ID,
            expected_resources={"RetainedQueue": "AWS::SQS::Queue"},
            expected_parameters=[
                {
                    "ParameterKey": "LegacyMode",
                    "ParameterValue": "disabled",
                }
            ],
        )


def test_change_set_rejects_parameter_readback_drift() -> None:
    """Break caught: change set silently drops a live legacy parameter."""

    cloudformation = _CloudFormation()
    cloudformation.created = True
    change_set = cloudformation.describe_change_set(
        ChangeSetName="glm52-task13-retained-pre-support-v1",
        StackName=STACK_ID,
        IncludePropertyValues=True,
    )
    change_set["Parameters"] = []

    with pytest.raises(RetainedUpdateError, match="parameter"):
        validate_retained_update_change_set(
            change_set,
            phase="pre-support",
            stack_id=STACK_ID,
            expected_resources={"RetainedQueue": "AWS::SQS::Queue"},
            expected_parameters=[
                {
                    "ParameterKey": "LegacyMode",
                    "ParameterValue": "disabled",
                }
            ],
        )


def test_change_set_rejects_request_shape_as_parameter_readback() -> None:
    """Break caught: tests accept fields that DescribeChangeSet never returns."""

    cloudformation = _CloudFormation()
    cloudformation.created = True
    change_set = cloudformation.describe_change_set(
        ChangeSetName="glm52-task13-retained-pre-support-v1",
        StackName=STACK_ID,
        IncludePropertyValues=True,
    )
    change_set["Parameters"] = [
        {"ParameterKey": "LegacyMode", "UsePreviousValue": True}
    ]

    with pytest.raises(RetainedUpdateError, match="parameter"):
        validate_retained_update_change_set(
            change_set,
            phase="pre-support",
            stack_id=STACK_ID,
            expected_resources={"RetainedQueue": "AWS::SQS::Queue"},
            expected_parameters=[
                {
                    "ParameterKey": "LegacyMode",
                    "ParameterValue": "disabled",
                }
            ],
        )


def test_cli_builds_exact_single_attempt_profile_region_clients() -> None:
    assert SCRIPT.is_file()
    spec = importlib.util.spec_from_file_location("retained_update_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert isinstance(module, ModuleType)

    client_calls: list[tuple[str, object]] = []

    class _Session:
        region_name = REGION

        def client(self, name: str, *, config: object) -> object:
            client_calls.append((name, config))
            return object()

    session_calls: list[dict[str, object]] = []

    def session_factory(**kwargs: object) -> _Session:
        session_calls.append(dict(kwargs))
        return _Session()

    config_calls: list[dict[str, object]] = []

    def config_factory(**kwargs: object) -> object:
        config_calls.append(dict(kwargs))
        return object()

    services = module._build_services(
        session_factory=session_factory,
        config_factory=config_factory,
    )

    assert isinstance(services, RetainedUpdateServices)
    assert session_calls == [{"profile_name": PROFILE, "region_name": REGION}]
    assert config_calls == [
        {
            "region_name": REGION,
            "connect_timeout": 5,
            "read_timeout": 30,
            "retries": {"mode": "standard", "total_max_attempts": 1},
        }
    ]
    assert [name for name, _config in client_calls] == [
        "sts",
        "cloudformation",
        "ec2",
        "ssm",
        "iam",
        "s3",
    ]
    assert services.total_max_attempts == 1


def test_retained_bootstrap_phase_uses_fresh_immutable_v9_identity() -> None:
    assert retained_update._PHASES["fence-bootstrap-v9"] == {
        "artifact_kind": "RETAINED_FENCE_BOOTSTRAP_TEMPLATE_V9",
        "change_set_name": "glm52-h1g-retained-fence-bootstrap-v9",
        "key": "task13/templates/retained-fence-bootstrap-v9.json",
    }
    assert "fence-bootstrap" not in retained_update._PHASES
    assert "fence-bootstrap-v3" not in retained_update._PHASES
    assert "fence-bootstrap-v4" not in retained_update._PHASES
    assert "fence-bootstrap-v5" not in retained_update._PHASES
    assert "fence-bootstrap-v6" not in retained_update._PHASES


def test_cli_accepts_only_settlement_and_bootstrap_v9_predeployment_phases() -> None:
    assert SCRIPT.is_file()
    spec = importlib.util.spec_from_file_location("retained_update_phase_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for live_phase in (
        "intentional-drift-settlement-v1",
        "fence-bootstrap-v9",
    ):
        parsed = module._parser().parse_args(
            [
                "--phase",
                live_phase,
                "--fragment",
                "/tmp/fragment.json",
                "--output-directory",
                "/tmp/production-output",
            ]
        )
        assert parsed.phase == live_phase

    for retired_phase in (
        "fence-bootstrap",
        "fence-bootstrap-v3",
        "fence-bootstrap-v4",
        "fence-bootstrap-v5",
        "fence-bootstrap-v6",
        "fence-runtime",
    ):
        with pytest.raises(SystemExit):
            module._parser().parse_args(
                [
                    "--phase",
                    retired_phase,
                    "--fragment",
                    "/tmp/fragment.json",
                    "--output-directory",
                    "/tmp/production-output",
                ]
            )


def test_bootstrap_cli_derives_canonical_staged_evidence_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert SCRIPT.is_file()
    spec = importlib.util.spec_from_file_location("retained_update_path_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fragment_path = tmp_path / "fragment.json"
    fragment = {
        "Resources": {
            "BootstrapRuntime": {
                "Type": "AWS::Lambda::Function",
            }
        }
    }
    fragment_path.write_bytes(module.canonical_json_bytes(fragment) + b"\n")
    production_output = tmp_path / "production-output"
    production_output.mkdir()
    captured: dict[str, object] = {}

    def apply(_services: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "status": "UPDATE_COMPLETE",
            "stack_id": STACK_ID,
            "template_version_id": "template-v1",
        }

    monkeypatch.setattr(module, "apply_retained_update", apply)
    assert (
        module.main(
            [
                "--phase",
                "fence-bootstrap-v9",
                "--fragment",
                str(fragment_path),
                "--output-directory",
                str(production_output),
            ],
            services_factory=lambda: object(),
            sleep=lambda _seconds: None,
        )
        == 0
    )
    expected = production_output / "retained-fence-bootstrap-v9"
    assert captured["output_directory"] == expected
    assert expected.is_dir()


def _intentional_drift_template() -> dict[str, object]:
    template = _base_template()
    parameters = template["Parameters"]
    assert isinstance(parameters, dict)
    parameters.update(
        {
            "GpuAmiId": {
                "Type": "AWS::SSM::Parameter::Value<AWS::EC2::Image::Id>",
            },
            "RootVolumeGiB": {"Type": "Number"},
            "ProjectTag": {"Type": "String"},
        }
    )
    resources = template["Resources"]
    assert isinstance(resources, dict)
    resources["GpuLaunchTemplate"] = {
        "Type": "AWS::EC2::LaunchTemplate",
        "Properties": {
            "LaunchTemplateData": {
                "ImageId": {"Ref": "GpuAmiId"},
                "BlockDeviceMappings": [
                    {
                        "DeviceName": "/dev/sda1",
                        "Ebs": {
                            "VolumeSize": {"Ref": "RootVolumeGiB"},
                            "VolumeType": "gp3",
                        },
                    }
                ],
                "UserData": {
                    "Fn::Base64": {
                        "Fn::Sub": "project=${ProjectTag}",
                    }
                },
            }
        },
    }
    resources["SkyMustStartCancelRole"] = {
        "Type": "AWS::IAM::Role",
        "Properties": {
            "Policies": [
                {
                    "PolicyName": "sky-must-start-cancel-only",
                    "PolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Action": "events:DisableRule",
                                "Resource": {
                                    "Fn::Sub": (
                                        "arn:${AWS::Partition}:events:${AWS::Region}:"
                                        "${AWS::AccountId}:rule/${ProjectTag}-cancel"
                                    )
                                },
                            }
                        ],
                    },
                }
            ]
        },
    }
    resources["SkyPilotControllerRole"] = {
        "Type": "AWS::IAM::Role",
        "Properties": {
            "Policies": [
                {
                    "PolicyName": "sky-controller-one-p5",
                    "PolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Action": "ec2:DescribeInstances",
                                "Resource": "*",
                                "Condition": {
                                    "StringEquals": {
                                        "aws:RequestTag/project": {
                                            "Ref": "ProjectTag",
                                        }
                                    }
                                },
                            }
                        ],
                    },
                }
            ]
        },
    }
    resources["SkyMustStartCancelRule"] = {
        "Type": "AWS::Events::Rule",
        "Properties": {
            "Name": "keep-glm52-sky-must-start-cancel",
            "State": "ENABLED",
        },
    }
    resources["SkyMustStartDeadlineSchedule"] = {
        "Type": "AWS::Scheduler::Schedule",
        "Properties": {
            "Name": "keep-glm52-sky-must-start-deadline",
            "State": "ENABLED",
        },
    }
    resources["SkyMustStartDeadlineScheduleAutoDelete"] = {
        "Type": "Custom::SchedulerActionAfterCompletion",
        "Properties": {
            "ServiceToken": "arn:aws:lambda:us-west-2:246813579024:function:auto-delete",
            "ScheduleName": {"Ref": "SkyMustStartDeadlineSchedule"},
        },
    }
    return template


def _intentional_drift_rows() -> list[dict[str, object]]:
    return [
        {
            "LogicalResourceId": "SkyMustStartCancelRule",
            "ResourceType": "AWS::Events::Rule",
            "StackResourceDriftStatus": "MODIFIED",
            "PropertyDifferences": [
                {
                    "PropertyPath": "/State",
                    "ExpectedValue": "ENABLED",
                    "ActualValue": "DISABLED",
                    "DifferenceType": "NOT_EQUAL",
                }
            ],
        },
        {
            "LogicalResourceId": "SkyMustStartDeadlineSchedule",
            "ResourceType": "AWS::Scheduler::Schedule",
            "StackResourceDriftStatus": "DELETED",
            "PropertyDifferences": [],
        },
    ]


def test_intentional_drift_settlement_changes_only_exact_observed_lifecycle() -> None:
    before = _intentional_drift_template()
    original = deepcopy(before)
    after = retained_update.compose_intentional_drift_settlement_template(
        before,
        _intentional_drift_rows(),
    )

    assert before == original
    expected = deepcopy(original)
    resources = expected["Resources"]
    assert isinstance(resources, dict)
    cancel_rule = resources["SkyMustStartCancelRule"]
    assert isinstance(cancel_rule, dict)
    properties = cancel_rule["Properties"]
    assert isinstance(properties, dict)
    properties["State"] = "DISABLED"
    del resources["SkyMustStartDeadlineSchedule"]
    del resources["SkyMustStartDeadlineScheduleAutoDelete"]
    assert after == expected


def test_intentional_drift_settlement_rejects_any_extra_drift() -> None:
    drifts = _intentional_drift_rows()
    drifts.append(
        {
            "LogicalResourceId": "Vpc",
            "ResourceType": "AWS::EC2::VPC",
            "StackResourceDriftStatus": "MODIFIED",
            "PropertyDifferences": [],
        }
    )
    with pytest.raises(RetainedUpdateError, match="intentional drift"):
        retained_update.compose_intentional_drift_settlement_template(
            _intentional_drift_template(),
            drifts,
        )


def _intentional_stack_parameters() -> list[dict[str, object]]:
    return [
        {
            "ParameterKey": "LegacyMode",
            "ParameterValue": "disabled",
        },
        {
            "ParameterKey": "GpuAmiId",
            "ParameterValue": (
                "/aws/service/deeplearning/ami/x86_64/"
                "base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
            ),
            "ResolvedValue": "ami-02b19745d2b303803",
        },
        {
            "ParameterKey": "RootVolumeGiB",
            "ParameterValue": "300",
        },
        {
            "ParameterKey": "ProjectTag",
            "ParameterValue": "keep-glm52",
        },
    ]


def _intentional_roundtrip_changes() -> list[dict[str, object]]:
    return [
        {
            "Type": "Resource",
            "ResourceChange": {
                "Action": "Modify",
                "LogicalResourceId": "GpuLaunchTemplate",
                "ResourceType": "AWS::EC2::LaunchTemplate",
                "Replacement": "False",
                "Scope": ["Properties"],
                "Details": [
                    {
                        "Target": {
                            "Attribute": "Properties",
                            "Name": "LaunchTemplateData",
                            "RequiresRecreation": "Never",
                            "Path": "/Properties/LaunchTemplateData",
                            "BeforeValue": (
                                "(Truncated-Signature):"
                                "8892463b805f7adf3b05042cee62f6a4838675933f7ca5f0f36c233de45b6e3b"
                            ),
                            "AfterValue": (
                                "(Truncated-Signature):"
                                "de795a4a649013332ccb10da08f3f9a6b4295e20e60bc93205e5e026f47db1a0"
                            ),
                            "AttributeChangeType": "Modify",
                        },
                        "Evaluation": "Static",
                        "ChangeSource": "DirectModification",
                    }
                ],
            },
        },
        {
            "Type": "Resource",
            "ResourceChange": {
                "Action": "Modify",
                "LogicalResourceId": "SkyMustStartCancelRole",
                "ResourceType": "AWS::IAM::Role",
                "Replacement": "False",
                "Scope": ["Properties"],
                "Details": [
                    {
                        "Target": {
                            "Attribute": "Properties",
                            "Name": "Policies",
                            "RequiresRecreation": "Never",
                            "Path": "/Properties/Policies",
                            "BeforeValue": (
                                "(Truncated-Signature):"
                                "1d47a8c5906537dd3baa0729fe9e45b3b79f11df5e07ad85ac80c01c8b45c526"
                            ),
                            "AfterValue": (
                                "(Truncated-Signature):"
                                "6d69414bef0f62fc13f9c739e0e62047dcabbedb3394cc619350738fbe38c4c0"
                            ),
                            "AttributeChangeType": "Modify",
                        },
                        "Evaluation": "Static",
                        "ChangeSource": "DirectModification",
                    }
                ],
            },
        },
        {
            "Type": "Resource",
            "ResourceChange": {
                "Action": "Modify",
                "LogicalResourceId": "SkyPilotControllerRole",
                "ResourceType": "AWS::IAM::Role",
                "Replacement": "False",
                "Scope": ["Properties"],
                "Details": [
                    {
                        "Target": {
                            "Attribute": "Properties",
                            "Name": "Policies",
                            "RequiresRecreation": "Never",
                            "Path": "/Properties/Policies",
                            "BeforeValue": (
                                "(Truncated-Signature):"
                                "7a77bb24011959c4dc1b961c52c046914a401a1e57435bc40c6508fb2e0139d9"
                            ),
                            "AfterValue": (
                                "(Truncated-Signature):"
                                "943de9e40066cc77ba2625a0a65fb73835850879a3ea85aab0a321e9c8b0729f"
                            ),
                            "AttributeChangeType": "Modify",
                        },
                        "Evaluation": "Static",
                        "ChangeSource": "DirectModification",
                    }
                ],
            },
        },
    ]


def _bootstrap_v9_parameters() -> list[dict[str, object]]:
    parameters = deepcopy(_intentional_stack_parameters())
    gpu_ami = next(row for row in parameters if row["ParameterKey"] == "GpuAmiId")
    gpu_ami["ResolvedValue"] = "ami-0f965e1c95dfa5a23"
    return parameters


def _bootstrap_v9_change_set() -> dict[str, object]:
    return {
        "StackId": STACK_ID,
        "StackName": RETAINED_STACK_NAME,
        "ChangeSetId": (
            "arn:aws:cloudformation:us-west-2:246813579024:changeSet/"
            "glm52-h1g-retained-fence-bootstrap-v9/"
            "77777777-bbbb-cccc-dddd-eeeeeeeeeeee"
        ),
        "ChangeSetName": "glm52-h1g-retained-fence-bootstrap-v9",
        "Status": "CREATE_COMPLETE",
        "ExecutionStatus": "AVAILABLE",
        "Capabilities": [],
        "Parameters": _bootstrap_v9_parameters(),
        "RoleARN": None,
        "Changes": [
            *_intentional_roundtrip_changes(),
            {
                "Type": "Resource",
                "ResourceChange": {
                    "Action": "Add",
                    "LogicalResourceId": "RetainedQueue",
                    "ResourceType": "AWS::SQS::Queue",
                    "Replacement": "False",
                },
            },
        ],
    }


def test_bootstrap_v9_change_set_accepts_only_additions_and_proven_roundtrips() -> None:
    validated = validate_retained_update_change_set(
        _bootstrap_v9_change_set(),
        phase="fence-bootstrap-v9",
        stack_id=STACK_ID,
        expected_resources={"RetainedQueue": "AWS::SQS::Queue"},
        expected_parameters=_intentional_stack_parameters(),
        expected_gpu_ami_resolution="ami-0f965e1c95dfa5a23",
    )

    assert validated["RoleARN"] is None


def test_bootstrap_v9_change_set_rejects_service_role_transition() -> None:
    change_set = _bootstrap_v9_change_set()
    change_set["RoleARN"] = DEPLOYMENT_ROLE_ARN

    with pytest.raises(RetainedUpdateError, match="authority"):
        validate_retained_update_change_set(
            change_set,
            phase="fence-bootstrap-v9",
            stack_id=STACK_ID,
            expected_resources={"RetainedQueue": "AWS::SQS::Queue"},
            expected_parameters=_intentional_stack_parameters(),
            expected_gpu_ami_resolution="ami-0f965e1c95dfa5a23",
        )


def test_bootstrap_v9_change_set_rejects_roundtrip_signature_drift() -> None:
    change_set = _bootstrap_v9_change_set()
    changes = change_set["Changes"]
    assert isinstance(changes, list)
    launch_template = changes[0]["ResourceChange"]
    launch_template["Details"][0]["Target"]["AfterValue"] = (
        "(Truncated-Signature):foreign"
    )

    with pytest.raises(RetainedUpdateError, match="semantic roundtrip"):
        validate_retained_update_change_set(
            change_set,
            phase="fence-bootstrap-v9",
            stack_id=STACK_ID,
            expected_resources={"RetainedQueue": "AWS::SQS::Queue"},
            expected_parameters=_intentional_stack_parameters(),
            expected_gpu_ami_resolution="ami-0f965e1c95dfa5a23",
        )


def _intentional_drift_change_set() -> dict[str, object]:
    return {
        "StackId": STACK_ID,
        "StackName": RETAINED_STACK_NAME,
        "ChangeSetId": (
            "arn:aws:cloudformation:us-west-2:246813579024:changeSet/"
            "glm52-task13-retained-drift-settlement-v1/"
            "dddddddd-bbbb-cccc-dddd-eeeeeeeeeeee"
        ),
        "ChangeSetName": "glm52-task13-retained-drift-settlement-v1",
        "Status": "CREATE_COMPLETE",
        "ExecutionStatus": "AVAILABLE",
        "Capabilities": ["CAPABILITY_NAMED_IAM"],
        "Parameters": deepcopy(_intentional_stack_parameters()),
        "RoleARN": None,
        "Changes": [
            *_intentional_roundtrip_changes(),
            {
                "Type": "Resource",
                "ResourceChange": {
                    "Action": "Modify",
                    "LogicalResourceId": "SkyMustStartCancelRule",
                    "ResourceType": "AWS::Events::Rule",
                    "Replacement": "False",
                    "Scope": ["Properties"],
                    "Details": [
                        {
                            "Target": {
                                "Attribute": "Properties",
                                "Name": "State",
                                "RequiresRecreation": "Never",
                                "Path": "/Properties/State",
                                "BeforeValue": "ENABLED",
                                "AfterValue": "DISABLED",
                                "AttributeChangeType": "Modify",
                            },
                            "Evaluation": "Static",
                            "ChangeSource": "DirectModification",
                        }
                    ],
                },
            },
            {
                "Type": "Resource",
                "ResourceChange": {
                    "Action": "Remove",
                    "LogicalResourceId": "SkyMustStartDeadlineSchedule",
                    "ResourceType": "AWS::Scheduler::Schedule",
                },
            },
            {
                "Type": "Resource",
                "ResourceChange": {
                    "Action": "Remove",
                    "LogicalResourceId": "SkyMustStartDeadlineScheduleAutoDelete",
                    "ResourceType": "Custom::SchedulerActionAfterCompletion",
                },
            },
        ],
    }


def test_intentional_drift_change_set_accepts_only_six_exact_actions() -> None:
    retained_update.validate_intentional_drift_settlement_change_set(
        _intentional_drift_change_set(),
        stack_id=STACK_ID,
        expected_parameters=_intentional_stack_parameters(),
    )

    extra = _intentional_drift_change_set()
    changes = extra["Changes"]
    assert isinstance(changes, list)
    changes.append(
        {
            "Type": "Resource",
            "ResourceChange": {
                "Action": "Modify",
                "LogicalResourceId": "Vpc",
                "ResourceType": "AWS::EC2::VPC",
                "Replacement": "False",
            },
        }
    )
    with pytest.raises(RetainedUpdateError, match="six exact actions"):
        retained_update.validate_intentional_drift_settlement_change_set(
            extra,
            stack_id=STACK_ID,
            expected_parameters=_intentional_stack_parameters(),
        )


def test_intentional_drift_change_set_rejects_any_service_role() -> None:
    change_set = _intentional_drift_change_set()
    change_set["RoleARN"] = DEPLOYMENT_ROLE_ARN
    with pytest.raises(RetainedUpdateError, match="authority"):
        retained_update.validate_intentional_drift_settlement_change_set(
            change_set,
            stack_id=STACK_ID,
            expected_parameters=_intentional_stack_parameters(),
        )


def test_intentional_drift_change_set_rejects_roundtrip_signature_drift() -> None:
    change_set = _intentional_drift_change_set()
    changes = change_set["Changes"]
    assert isinstance(changes, list)
    launch_template = changes[0]["ResourceChange"]
    details = launch_template["Details"]
    details[0]["Target"]["AfterValue"] = "(Truncated-Signature):foreign"
    with pytest.raises(RetainedUpdateError, match="six exact actions"):
        retained_update.validate_intentional_drift_settlement_change_set(
            change_set,
            stack_id=STACK_ID,
            expected_parameters=_intentional_stack_parameters(),
        )


class _IntentionalDriftCloudFormation:
    def __init__(self) -> None:
        self.before_template = _intentional_drift_template()
        self.after_template = (
            retained_update.compose_intentional_drift_settlement_template(
                self.before_template,
                _intentional_drift_rows(),
            )
        )
        self.created = False
        self.executed = False
        self.create_requests: list[dict[str, object]] = []
        self.execute_requests: list[dict[str, object]] = []
        self.drift_detection_count = 0
        self.final_stack_drift_status = "IN_SYNC"
        self.final_outputs = _before_outputs()
        self.omitted_final_resource: str | None = None
        self.inactive_final_resources: set[str] = set()
        self.stack_parameters = _intentional_stack_parameters()
        self.initial_status = "UPDATE_COMPLETE"

    def describe_stacks(self, **request: object) -> dict[str, object]:
        assert request["StackName"] in {RETAINED_STACK_NAME, STACK_ID}
        return {
            "Stacks": [
                _stack(
                    status=(
                        "UPDATE_COMPLETE" if self.executed else self.initial_status
                    ),
                    outputs=(
                        self.final_outputs if self.executed else _before_outputs()
                    ),
                    parameters=self.stack_parameters,
                    role_arn=None,
                )
            ]
        }

    def get_template(self, **request: object) -> dict[str, object]:
        assert request["TemplateStage"] == "Original"
        if "ChangeSetName" in request:
            assert (
                request["ChangeSetName"]
                == _intentional_drift_change_set()["ChangeSetId"]
            )
            return {"TemplateBody": self.after_template}
        assert request["StackName"] == STACK_ID
        return {
            "TemplateBody": (
                self.after_template if self.executed else self.before_template
            )
        }

    def create_change_set(self, **request: object) -> dict[str, object]:
        self.create_requests.append(dict(request))
        parsed = urlparse(str(request["TemplateURL"]))
        assert parsed.path == "/task13/templates/retained-drift-settlement-v1.json"
        assert parse_qs(parsed.query) == {"versionId": ["pre-support-version-1"]}
        assert request["Capabilities"] == ["CAPABILITY_NAMED_IAM"]
        assert "RoleARN" not in request
        self.created = True
        return {
            "Id": _intentional_drift_change_set()["ChangeSetId"],
            "StackId": STACK_ID,
        }

    def describe_change_set(self, **request: object) -> dict[str, object]:
        assert self.created
        assert request["StackName"] == STACK_ID
        assert request["IncludePropertyValues"] is True
        value = _intentional_drift_change_set()
        value["ExecutionStatus"] = "EXECUTE_COMPLETE" if self.executed else "AVAILABLE"
        value["Parameters"] = deepcopy(self.stack_parameters)
        return value

    def execute_change_set(self, **request: object) -> dict[str, object]:
        self.execute_requests.append(dict(request))
        self.executed = True
        return {}

    def list_stack_resources(self, **request: object) -> dict[str, object]:
        assert request == {"StackName": STACK_ID}
        resources = (
            self.after_template["Resources"]
            if self.executed
            else (self.before_template["Resources"])
        )
        assert isinstance(resources, dict)
        physical_ids = {
            "Vpc": "vpc-0123456789abcdef0",
            "ModelBucket": RETAINED_MODELS_BUCKET,
            "GpuLaunchTemplate": "lt-0123456789abcdef0",
            "SkyMustStartCancelRole": "keep-glm52-cancel-role",
            "SkyPilotControllerRole": "keep-glm52-controller-role",
            "SkyMustStartCancelRule": "keep-glm52-sky-must-start-cancel",
            "SkyMustStartDeadlineSchedule": ("keep-glm52-sky-must-start-deadline"),
            "SkyMustStartDeadlineScheduleAutoDelete": (
                "must-start-schedule-auto-delete:keep-glm52-sky-must-start-deadline"
            ),
        }
        return {
            "StackResourceSummaries": [
                {
                    "LogicalResourceId": logical_id,
                    "PhysicalResourceId": physical_ids[logical_id],
                    "ResourceType": resource["Type"],
                    "ResourceStatus": "UPDATE_COMPLETE",
                }
                for logical_id, resource in resources.items()
                if logical_id not in self.inactive_final_resources
                and not (
                    self.executed
                    and logical_id == self.omitted_final_resource
                )
            ]
        }

    def detect_stack_drift(self, **request: object) -> dict[str, object]:
        assert request == {"StackName": STACK_ID}
        self.drift_detection_count += 1
        return {
            "StackDriftDetectionId": (f"drift-detection-{self.drift_detection_count}")
        }

    def describe_stack_drift_detection_status(
        self,
        **request: object,
    ) -> dict[str, object]:
        assert request["StackDriftDetectionId"] == (
            f"drift-detection-{self.drift_detection_count}"
        )
        return {
            "DetectionStatus": "DETECTION_COMPLETE",
            "StackDriftStatus": (
                self.final_stack_drift_status if self.executed else "DRIFTED"
            ),
        }

    def describe_stack_resource_drifts(
        self,
        **request: object,
    ) -> dict[str, object]:
        rows = [] if self.executed else _intentional_drift_rows()
        for row in rows:
            row["Timestamp"] = object()
            row["ExpectedProperties"] = "{}"
            row["ActualProperties"] = "{}"
        return {"StackResourceDrifts": rows}


class _BootstrapV9CloudFormation(_IntentionalDriftCloudFormation):
    def __init__(self) -> None:
        super().__init__()
        self.before_template = deepcopy(self.after_template)
        self.after_template = retained_update.compose_complete_retained_template(
            self.before_template,
            _fragment(),
        )
        self.after_template["Resources"]["GpuLaunchTemplate"]["Properties"][
            "LaunchTemplateData"
        ]["ImageId"] = "ami-02b19745d2b303803"
        self.created = False
        self.executed = False
        self.create_requests = []
        self.execute_requests = []
        self.final_outputs = _after_outputs()
        self.on_execute: object | None = None

    def get_template(self, **request: object) -> dict[str, object]:
        assert request["TemplateStage"] == "Original"
        if "ChangeSetName" in request:
            assert request["ChangeSetName"] == _bootstrap_v9_change_set()["ChangeSetId"]
            return {"TemplateBody": self.after_template}
        assert request["StackName"] == STACK_ID
        return {
            "TemplateBody": (
                self.after_template if self.executed else self.before_template
            )
        }

    def create_change_set(self, **request: object) -> dict[str, object]:
        self.create_requests.append(dict(request))
        parsed = urlparse(str(request["TemplateURL"]))
        assert parsed.path == "/task13/templates/retained-fence-bootstrap-v9.json"
        assert parse_qs(parsed.query) == {"versionId": ["pre-support-version-1"]}
        assert request["Capabilities"] == []
        assert "RoleARN" not in request
        self.created = True
        return {
            "Id": _bootstrap_v9_change_set()["ChangeSetId"],
            "StackId": STACK_ID,
        }

    def describe_change_set(self, **request: object) -> dict[str, object]:
        assert self.created
        assert request["StackName"] == STACK_ID
        assert request["IncludePropertyValues"] is True
        value = _bootstrap_v9_change_set()
        value["ExecutionStatus"] = "EXECUTE_COMPLETE" if self.executed else "AVAILABLE"
        return value

    def execute_change_set(self, **request: object) -> dict[str, object]:
        self.execute_requests.append(dict(request))
        self.executed = True
        self.stack_parameters = _bootstrap_v9_parameters()
        if callable(self.on_execute):
            self.on_execute()
        return {}

    def list_stack_resources(self, **request: object) -> dict[str, object]:
        assert request == {"StackName": STACK_ID}
        resources = (
            self.after_template["Resources"]
            if self.executed
            else self.before_template["Resources"]
        )
        assert isinstance(resources, dict)
        physical_ids = {
            "Vpc": "vpc-0123456789abcdef0",
            "ModelBucket": RETAINED_MODELS_BUCKET,
            "GpuLaunchTemplate": "lt-0123456789abcdef0",
            "SkyMustStartCancelRole": "keep-glm52-cancel-role",
            "SkyPilotControllerRole": "keep-glm52-controller-role",
            "SkyMustStartCancelRule": "keep-glm52-sky-must-start-cancel",
            "RetainedQueue": (
                "https://sqs.us-west-2.amazonaws.com/246813579024/"
                "keep-glm52-h1g-retained-test"
            ),
        }
        return {
            "StackResourceSummaries": [
                {
                    "LogicalResourceId": logical_id,
                    "PhysicalResourceId": physical_ids[logical_id],
                    "ResourceType": resource["Type"],
                    "ResourceStatus": (
                        "CREATE_COMPLETE"
                        if logical_id == "RetainedQueue"
                        else "UPDATE_COMPLETE"
                    ),
                }
                for logical_id, resource in resources.items()
            ]
        }

    def list_exports(self, **request: object) -> dict[str, object]:
        assert request == {}
        return {
            "Exports": [
                {
                    "Name": row["ExportName"],
                    "Value": row["OutputValue"],
                    "ExportingStackId": STACK_ID,
                }
                for row in _after_outputs()
            ]
        }


def test_bootstrap_v9_update_uses_caller_and_preserves_roundtrip_semantics(
    tmp_path: Path,
) -> None:
    cloudformation = _BootstrapV9CloudFormation()
    services = _services(cloudformation)

    result = apply_retained_update(
        services,
        phase="fence-bootstrap-v9",
        fragment=_fragment(),
        output_directory=tmp_path,
        sleep=lambda _seconds: None,
        max_polls=8,
    )

    assert result["status"] == "UPDATE_COMPLETE"
    assert "RoleARN" not in cloudformation.create_requests[0]
    assert len(cloudformation.execute_requests) == 1
    change_evidence = json.loads(
        (tmp_path / "retained-fence-bootstrap-v9-change-set.json").read_bytes()
    )
    readback = json.loads(
        (tmp_path / "retained-fence-bootstrap-v9-readback.json").read_bytes()
    )
    assert change_evidence["change_set_role_arn"] is None
    assert (
        change_evidence["initial_semantics"]
        == (change_evidence["pre_execute_semantics"])
    )
    assert readback["role_arn"] is None
    assert (
        readback["initial_semantics"]["launch_template_data_sha256"]
        == readback["final_semantics"]["launch_template_data_sha256"]
    )
    assert (
        readback["initial_semantics"]["iam_policies"]
        == readback["final_semantics"]["iam_policies"]
    )


def test_bootstrap_v9_accepts_completed_rollback_as_recovery_source(
    tmp_path: Path,
) -> None:
    cloudformation = _BootstrapV9CloudFormation()
    cloudformation.initial_status = "UPDATE_ROLLBACK_COMPLETE"

    result = apply_retained_update(
        _services(cloudformation),
        phase="fence-bootstrap-v9",
        fragment=_fragment(),
        output_directory=tmp_path,
        sleep=lambda _seconds: None,
        max_polls=8,
    )

    assert result["status"] == "UPDATE_COMPLETE"


def test_bootstrap_v9_update_rejects_post_execute_semantic_drift(
    tmp_path: Path,
) -> None:
    cloudformation = _BootstrapV9CloudFormation()
    services = _services(cloudformation)
    ec2 = services.ec2
    assert isinstance(ec2, _Ec2)
    cloudformation.on_execute = lambda: setattr(ec2, "version_number", 9)

    with pytest.raises(RetainedUpdateError, match="semantic"):
        apply_retained_update(
            services,
            phase="fence-bootstrap-v9",
            fragment=_fragment(),
            output_directory=tmp_path,
            sleep=lambda _seconds: None,
            max_polls=8,
        )

    assert not (tmp_path / "retained-fence-bootstrap-v9-readback.json").exists()


def test_bootstrap_v9_preserves_stack_ami_when_public_ssm_latest_advances(
    tmp_path: Path,
) -> None:
    cloudformation = _BootstrapV9CloudFormation()
    services = _services(cloudformation)
    object.__setattr__(
        services,
        "ssm",
        _Ssm(value="ami-0f965e1c95dfa5a23", version=174),
    )

    result = apply_retained_update(
        services,
        phase="fence-bootstrap-v9",
        fragment=_fragment(),
        output_directory=tmp_path,
        sleep=lambda _seconds: None,
        max_polls=8,
    )

    assert result["status"] == "UPDATE_COMPLETE"
    readback = json.loads(
        (tmp_path / "retained-fence-bootstrap-v9-readback.json").read_bytes()
    )
    semantics = readback["final_semantics"]
    assert semantics["resolved_ami_id"] == "ami-0f965e1c95dfa5a23"
    assert semantics["ssm_parameter_value"] == "ami-0f965e1c95dfa5a23"
    assert semantics["ssm_parameter_version"] == 174
    assert semantics["ssm_matches_resolved_ami"] is True
    assert (
        readback["initial_semantics"]["launch_template_data_sha256"]
        == semantics["launch_template_data_sha256"]
    )


def test_intentional_drift_settlement_is_one_exact_transaction(
    tmp_path: Path,
) -> None:
    cloudformation = _IntentionalDriftCloudFormation()
    output = tmp_path / "intentional-drift"
    output.mkdir()
    services = _services(cloudformation)

    result = retained_update.apply_intentional_drift_settlement(
        services,
        output_directory=output,
        sleep=lambda _seconds: None,
        max_polls=8,
    )

    assert result == {
        "status": "UPDATE_COMPLETE",
        "stack_id": STACK_ID,
        "change_set_id": _intentional_drift_change_set()["ChangeSetId"],
        "phase": "intentional-drift-settlement-v1",
        "template_key": "task13/templates/retained-drift-settlement-v1.json",
        "template_version_id": "pre-support-version-1",
    }
    assert len(cloudformation.create_requests) == 1
    assert len(cloudformation.execute_requests) == 1
    assert cloudformation.drift_detection_count == 3
    assert {path.name for path in output.iterdir()} == {
        "retained-intentional-drift-settlement-v1-before-template.json",
        "retained-intentional-drift-settlement-v1-after-template.json",
        "retained-intentional-drift-settlement-v1-change-set.json",
        "retained-intentional-drift-settlement-v1-readback.json",
    }


def test_intentional_drift_settlement_recovers_exact_post_execute_readback(
    tmp_path: Path,
) -> None:
    cloudformation = _IntentionalDriftCloudFormation()
    output = tmp_path / "recover-post-execute"
    output.mkdir()
    services = _services(cloudformation)
    first = retained_update.apply_intentional_drift_settlement(
        services,
        output_directory=output,
        sleep=lambda _seconds: None,
        max_polls=8,
    )
    readback = output / "retained-intentional-drift-settlement-v1-readback.json"
    readback.unlink()

    recovered = retained_update.apply_intentional_drift_settlement(
        services,
        output_directory=output,
        sleep=lambda _seconds: None,
        max_polls=8,
    )

    assert recovered == first
    assert readback.is_file()
    assert len(cloudformation.execute_requests) == 1
    recovered_readback = json.loads(readback.read_bytes())
    assert recovered_readback["execute_response_observation"] == (
        "UNKNOWN_RECOVERED_FROM_EXACT_FINAL_STATE"
    )
    assert "execute_response_was_ambiguous" not in recovered_readback
    assert recovered_readback["recovered_from_exact_final_state"] is True
    assert "recovered_after_exact_execute" not in recovered_readback


def test_intentional_drift_settlement_recovery_accepts_inactive_conditionals(
    tmp_path: Path,
) -> None:
    cloudformation = _IntentionalDriftCloudFormation()
    parameters = cloudformation.before_template["Parameters"]
    conditions = cloudformation.before_template.setdefault("Conditions", {})
    resources = cloudformation.before_template["Resources"]
    assert isinstance(parameters, dict)
    assert isinstance(conditions, dict)
    assert isinstance(resources, dict)
    parameters["InactiveTestResourceEnabled"] = {"Type": "String"}
    conditions["InactiveTestResourceCondition"] = {
        "Fn::Equals": [
            {"Ref": "InactiveTestResourceEnabled"},
            "true",
        ]
    }
    resources["InactiveTestResource"] = {
        "Type": "AWS::SNS::Topic",
        "Condition": "InactiveTestResourceCondition",
    }
    cloudformation.after_template = (
        retained_update.compose_intentional_drift_settlement_template(
            cloudformation.before_template,
            _intentional_drift_rows(),
        )
    )
    cloudformation.stack_parameters.append(
        {
            "ParameterKey": "InactiveTestResourceEnabled",
            "ParameterValue": "false",
        }
    )
    cloudformation.inactive_final_resources.add("InactiveTestResource")
    output = tmp_path / "recover-inactive-conditionals"
    output.mkdir()
    services = _services(cloudformation)
    retained_update.apply_intentional_drift_settlement(
        services,
        output_directory=output,
        sleep=lambda _seconds: None,
        max_polls=8,
    )
    readback = output / "retained-intentional-drift-settlement-v1-readback.json"
    readback.unlink()

    retained_update.apply_intentional_drift_settlement(
        services,
        output_directory=output,
        sleep=lambda _seconds: None,
        max_polls=8,
    )

    assert readback.is_file()


def test_intentional_drift_settlement_recovery_rejects_incomplete_inventory(
    tmp_path: Path,
) -> None:
    cloudformation = _IntentionalDriftCloudFormation()
    output = tmp_path / "recover-incomplete-inventory"
    output.mkdir()
    services = _services(cloudformation)
    retained_update.apply_intentional_drift_settlement(
        services,
        output_directory=output,
        sleep=lambda _seconds: None,
        max_polls=8,
    )
    readback = output / "retained-intentional-drift-settlement-v1-readback.json"
    readback.unlink()
    cloudformation.omitted_final_resource = "Vpc"

    with pytest.raises(RetainedUpdateError, match="resource inventory"):
        retained_update.apply_intentional_drift_settlement(
            services,
            output_directory=output,
            sleep=lambda _seconds: None,
            max_polls=8,
        )
    assert not readback.exists()


def test_intentional_drift_settlement_rejects_live_policy_semantic_drift(
    tmp_path: Path,
) -> None:
    cloudformation = _IntentionalDriftCloudFormation()
    services = _services(cloudformation)
    iam = services.iam
    assert isinstance(iam, _Iam)
    policy = iam.policy_documents[
        ("keep-glm52-controller-role", "sky-controller-one-p5")
    ]
    policy["Statement"][0]["Action"] = "ec2:RunInstances"
    output = tmp_path / "policy-drift"
    output.mkdir()

    with pytest.raises(RetainedUpdateError, match="semantic"):
        retained_update.apply_intentional_drift_settlement(
            services,
            output_directory=output,
            sleep=lambda _seconds: None,
            max_polls=8,
        )
    assert cloudformation.create_requests == []


def test_intentional_drift_settlement_rejects_live_launch_template_drift(
    tmp_path: Path,
) -> None:
    cloudformation = _IntentionalDriftCloudFormation()
    services = _services(cloudformation)
    ec2 = services.ec2
    assert isinstance(ec2, _Ec2)
    ec2.launch_template_data["ImageId"] = "ami-foreign"
    output = tmp_path / "launch-drift"
    output.mkdir()

    with pytest.raises(RetainedUpdateError, match="semantic"):
        retained_update.apply_intentional_drift_settlement(
            services,
            output_directory=output,
            sleep=lambda _seconds: None,
            max_polls=8,
        )
    assert cloudformation.create_requests == []


def test_intentional_drift_settlement_requires_final_stack_in_sync(
    tmp_path: Path,
) -> None:
    cloudformation = _IntentionalDriftCloudFormation()
    cloudformation.final_stack_drift_status = "DRIFTED"
    output = tmp_path / "drift-remains"
    output.mkdir()

    with pytest.raises(RetainedUpdateError, match="stack drift status"):
        retained_update.apply_intentional_drift_settlement(
            _services(cloudformation),
            output_directory=output,
            sleep=lambda _seconds: None,
            max_polls=8,
        )


def test_intentional_drift_settlement_preserves_stack_outputs(
    tmp_path: Path,
) -> None:
    cloudformation = _IntentionalDriftCloudFormation()
    cloudformation.final_outputs = [
        {
            **_before_outputs()[0],
            "OutputValue": "vpc-foreign",
        },
        _before_outputs()[1],
    ]
    output = tmp_path / "output-drift"
    output.mkdir()

    with pytest.raises(RetainedUpdateError, match="stack identity"):
        retained_update.apply_intentional_drift_settlement(
            _services(cloudformation),
            output_directory=output,
            sleep=lambda _seconds: None,
            max_polls=8,
        )
